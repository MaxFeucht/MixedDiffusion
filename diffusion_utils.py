import torch
import torch.nn as nn
import torch.nn.functional as F
import torchgeometry as tgm
from sklearn.mixture import GaussianMixture

import math
from tqdm import tqdm

import warnings

### Forward Process ###

class Scheduler:
    def __init__(self):
        pass

    def linear(self, timesteps):  # Problematic when using < 20 timesteps, as betas are then surpassing 1.0
        """
        linear schedule, proposed in original ddpm paper
        """
        scale = 1000 / timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float32)

    def cosine(self, timesteps, s = 0.008):
        """
        cosine schedule
        as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
        """
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps, dtype = torch.float32) / timesteps
        alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
        
    def sigmoid(self, timesteps, start = -3, end = 3, tau = 1):
        """
        sigmoid schedule
        proposed in https://arxiv.org/abs/2212.11972 - Figure 8
        better for images > 64x64, when used during training
        """
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps, dtype = torch.float32) / timesteps
        v_start = torch.tensor(start / tau).sigmoid()
        v_end = torch.tensor(end / tau).sigmoid()
        alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)

    def get_noise_schedule(self, timesteps, noise_schedule):
        if noise_schedule == 'linear':
            return self.linear(timesteps)
        elif noise_schedule == 'cosine':
            return self.cosine(timesteps)
        elif noise_schedule == 'sigmoid':
            return self.sigmoid(timesteps)
        else:
            raise ValueError('Invalid schedule type')

    
    def get_blur_schedule(self, timesteps, std, type = 'constant'):
        
        # 0.01 * t + 0.35 CIFAR-10
        
        # MNIST:
        # Rissanen: Max of 20, min of 0.5, interpolated --> Investigate more clearly in the future and also use Hoogeboom for further explanation (blurring schedule slightly better explained there)
        # Bansal: Constant 7, recursive application

        # The standard deviation of the kernel starts at 1 and increases exponentially at the rate of 0.01.        
        if type == 'constant':
            return torch.ones(timesteps, dtype=torch.float32) * std

        if type == 'exponential':
            return torch.exp(std * torch.arange(timesteps, dtype=torch.float32))
       
        
        
class Degradation:
    
    def __init__(self, timesteps, degradation, noise_schedule, dataset, **kwargs):
        
        self.timesteps = timesteps
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

        
        assert degradation in ['noise', 'blur', 'fadeblack', 'fadeblack_blur'], 'Invalid degradation type, choose from noise, blur, fadeblack, fadeblack_blur'
        self.degradation = degradation
        
        assert dataset in ['mnist', 'cifar10', 'celeba', 'lsun_churches'],f"Invalid dataset, choose from ['mnist', 'cifar10', 'celeba', 'lsun_churches']"
        
        # Default settings
        blur_kwargs = {'channels': 1 if dataset == 'mnist' else 3, 
                        'kernel_size': 5, # Change to 11 for non-cold start but for conditional sampling (only blurring for 40 steps)
                        'kernel_std': 0.001 if dataset != 'mnist' else 2, # Std has a different interpretation for constant schedule and exponential schedule: constant schedule is the actual std, exponential schedule is the rate of increase # 7 if dataset == 'mnist' else 0.01
                        'timesteps': timesteps, 
                        'blur_routine': 'constant' if dataset == 'mnist' else 'exponential'} # 'constant' if dataset == 'mnist' else 'exponential'}
            
        self.blur = Blurring(**blur_kwargs)
        self.noise_coefs = DenoisingCoefs(timesteps=timesteps, noise_schedule=noise_schedule, device = self.device)


    def noising(self, x_0, t, noise = None):
        """
        Function to add noise to an image x at time t.
        
        :param torch.Tensor x_0: The original image
        :param int t: The time step
        :return torch.Tensor: The degraded image at time t
        """

        if noise is None:
            noise = torch.randn_like(x_0, device = self.device)
            warnings.warn('Noise not provided, using random noise')

        x_0_coef, residual_coef = self.noise_coefs.forward_process(t)
        x_0_coef, residual_coef = x_0_coef.to(self.device), residual_coef.to(self.device)
        x_t = x_0_coef * x_0 + residual_coef * noise
        return x_t


    def blurring(self, x_0, t):
        """
        Function to blur an image x at time t.
        
        :param torch.Tensor x_0: The original image
        :param int t: The time step
        :return torch.Tensor: The degraded image at time t
        """
        
        gaussian_kernels = nn.ModuleList(self.blur.get_kernels())

        # Freeze kernels
        for kernel in gaussian_kernels:
            kernel.requires_grad = False

        gaussian_kernels.to(self.device)  # Move kernels to GPU
        x = x_0
        
        # Keep gradients for the original image for backpropagation
        if x_0.requires_grad:
            x.retain_grad()

        t_max = torch.max(t)

        # Blur all images to the max, but store all intermediate blurs for later retrieval
        max_blurs = []
        for i in range(t_max + 1):
            x = x.unsqueeze(0) if len(x.shape) == 2  else x
            x = gaussian_kernels[i](x).squeeze(0) 

            # Make sure gradients are retained and kernels are frozen
            if x_0.requires_grad:      
                assert gaussian_kernels[i].requires_grad == False
                assert x.requires_grad == True 

            max_blurs.append(x)
        
        max_blurs = torch.stack(max_blurs)

        # Choose the correct blur for each image in the batch
        blur_t = []
        for step in range(t.shape[0]):
            blur_t.append(max_blurs[t[step], step])
            assert max_blurs[t[step], step].shape == x_0.shape[1:], f"Shape mismatch: {max_blurs[t[step], step].shape} and {x_0.shape} at time step {i}"

        return torch.stack(blur_t)
    


    def blacking(self, x_0, t):
        """
        Function to fade an image x to black at time t.
        
        :param torch.Tensor x_0: The original image
        :param int t: The time step
        :return torch.Tensor: The degraded image at time t
        """
        multiplier = (1 - (t+1) / self.timesteps).reshape(-1, 1, 1, 1)  # +1 bc of zero indexing
        x_t = multiplier * x_0 
        return x_t
    
    
    def degrade(self, x, t, noise = None):
        """
        Function to degrade an image x at time t.
        
        :param x: torch.Tensor
            The image at time t
        :param t: int
            The time step
            
        :return: torch.Tensor
            The degraded image at time t
        """
        if self.degradation == 'noise':
            return self.noising(x, t, noise)
        elif self.degradation == 'blur':
            return self.blurring(x, t)
        elif self.degradation == 'fadeblack':
            return self.blacking(x, t)
        elif self.degradation == 'fadeblack_blur':
            return self.blacking(self.blurring(x, t),t)


class Blurring:
    
    def __init__(self, timesteps, channels, kernel_size, kernel_std, blur_routine):
            """
            Initializes the Blurring class.

            Args:
                channels (int): Number of channels in the input image. Default is 3.
                kernel_size (int): Size of the kernel used for blurring. Default is 11.
                kernel_std (int): Standard deviation of the kernel used for blurring. Default is 7.
                num_timesteps (int): Number of diffusion timesteps. Default is 40.
                blur_routine (str): Routine used for blurring. Default is 'Constant'.
            """
            scheduler = Scheduler()

            self.channels = channels
            self.kernel_size = kernel_size
            self.kernel_stds = scheduler.get_blur_schedule(timesteps = timesteps, std = kernel_std, type = blur_routine) 
            self.num_timesteps = timesteps
            self.blur_routine = blur_routine
        

    def get_conv(self, dims, std):
        """
        Function to obtain a 2D convolutional layer with a Gaussian Blurring kernel.
        
        :param tuple dims: The dimensions of the kernel
        :param tuple std: The standard deviation of the kernel
        :return nn.Conv2d: The 2D convolutional layer with the Gaussian Blurring kernel
        """
        
        kernel = tgm.image.get_gaussian_kernel2d(dims, std) 
        conv = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=dims, padding=int((dims[0]-1)/2), padding_mode='circular',
                         bias=False, groups=self.channels)
        
        kernel = torch.unsqueeze(kernel, 0)
        kernel = torch.unsqueeze(kernel, 0)
        kernel = kernel.repeat(self.channels, 1, 1, 1)
        conv.weight = nn.Parameter(kernel, requires_grad=False)

        # with torch.no_grad():
        #     kernel = torch.unsqueeze(kernel, 0)
        #     kernel = torch.unsqueeze(kernel, 0)
        #     kernel = kernel.repeat(self.channels, 1, 1, 1)
        #     conv.weight = nn.Parameter(kernel)

        return conv


    def get_kernels(self):
        """
        Function to obtain a list of 2D convolutional layers with Gaussian Blurring kernels following a certain routine.
        
        :return list: A list of 2D convolutional layers with Gaussian Blurring kernels
        """
        
        kernels = []
        for i in range(self.num_timesteps):
            kernels.append(self.get_conv((self.kernel_size, self.kernel_size), (self.kernel_stds[i], self.kernel_stds[i]))) 
        return kernels
    


class DenoisingCoefs:
    
    
    def __init__(self, timesteps, noise_schedule, device, **kwargs):
        self.timesteps = timesteps
        self.scheduler = Scheduler()
        
        self.betas = self.scheduler.get_noise_schedule(self.timesteps, noise_schedule=noise_schedule).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.], device=device), self.alphas_cumprod[:-1]])
    
    
    def forward_process(self, t):
        """
        Function to obtain the coefficients for the standard Denoising Diffusion process xt = sqrt(alphas_cumprod) * x0 + sqrt(1 - alphas_cumprod) * N(0, I).
        
        :param x: torch.Tensor
            The image at time t
        :param t: int
            The time step
            
        :return: tuple
            The coefficients for the Denoising Diffusion process
        """
        alpha_t = self.alphas_cumprod.gather(-1, t).reshape(-1, 1, 1, 1)
        x0_coef = torch.sqrt(alpha_t)
        residual_coef =  torch.sqrt(1. - alpha_t)
        return x0_coef, residual_coef
    
    
    def posterior(self, t):
        """
        Function to obtain the coefficients for the Denoising Diffusion posterior 
        q(x_{t-1} | x_t, x_0).
        """
        beta_t = self.betas.gather(-1, t).reshape(-1, 1, 1, 1)
        alphas_cumprod_t = self.alphas_cumprod.gather(-1, t).reshape(-1, 1, 1, 1)
        alphas_cumprod_prev_t = self.alphas_cumprod_prev.gather(-1, t).reshape(-1, 1, 1, 1)
        
        posterior_variance = beta_t * (1. - alphas_cumprod_prev_t) / (1. - alphas_cumprod_t) # beta_hat
        posterior_mean_x0 = beta_t * torch.sqrt(alphas_cumprod_prev_t) / (1. - alphas_cumprod_t) #x_0
        posterior_mean_xt = (1. - alphas_cumprod_prev_t) * torch.sqrt(self.alphas.gather(-1,t).reshape(-1, 1, 1, 1)) / (1. - alphas_cumprod_t) #x_t

        return posterior_mean_x0, posterior_mean_xt, posterior_variance
    
    
    def x0_restore(self, t):
        """
        Function to obtain the coefficients for the Denoising Diffusion reconstruction.
        
        :param int t: The time step
        :return tuple: The coefficients for the Denoising Diffusion process
        """
        
        xt_coef = torch.sqrt(1. / self.alphas_cumprod.gather(-1, t)).reshape(-1, 1, 1, 1)
        residual_coef = torch.sqrt(1. / self.alphas_cumprod.gather(-1, t) - 1).reshape(-1, 1, 1, 1)

        return xt_coef, residual_coef
    
    

class Reconstruction:
    
    def __init__(self, prediction, degradation, **kwargs):
        self.prediction = prediction
        self.deterministic = True if degradation in ['blur', 'fadeblack', 'fadeblack_blur'] else False
        self.coefs = DenoisingCoefs(**kwargs)

    def reform_pred(self, model_output, x_t, t, return_x0 = False):
        """
        Function to reform predictions for a given degraded image x_t at time t, using the output of a trained model and a degrader function.
        
        :param torch.Tensor model_output: The output of the model - either the residual or the x0 estimate
        :param torch.Tensor x_t: The degraded image at time t
        :param int t: The time step
        :return torch.Tensor: The predicted image at time t-1
        """
        
        if not self.deterministic: 
            xt_coef, residual_coef = self.coefs.x0_restore(t) # Get coefficients for the Denoising Diffusion process
        else:
            xt_coef, residual_coef = torch.tensor(1.), torch.tensor(1.) # Coefficients are 1 for deterministic degradation
                
        if self.prediction == 'residual':
            residual = model_output
            if not return_x0:
                return residual
            else:
                x0_estimate = xt_coef * x_t - residual_coef * residual 
                return x0_estimate      
        

        elif self.prediction == 'x0':
            x0_estimate = model_output
            if return_x0:
                return x0_estimate
            else:
                residual = (xt_coef * x_t - x0_estimate) / residual_coef
                return residual      
        
        else:
            raise ValueError('Invalid prediction type')
        

class Loss:
    
    def __init__(self, **kwargs):
        self.degradation = Degradation(**kwargs)
    
    def mse_loss(self, target, pred):    
        return F.mse_loss(pred, target, reduction='mean')
    
    def cold_loss(self, target, pred, t):
        diff = pred - target
        return diff.abs().mean()  # Mean over batch dimension

    def darras_loss(self, target, pred, t):
        diff = pred - target # difference between the predicted and the target image / residual
        degraded_diff = self.degradation.degrade(diff, t)**2 # Move this difference into the degradation space and square it
        return degraded_diff.mean()  # Squared error loss, averaged over batch dimension



class Trainer:
    
    def __init__(self, model, lr, timesteps, prediction, degradation, noise_schedule, **kwargs):

        self.device = kwargs['device']
        self.model = model.to(self.device)
        self.prediction = prediction
        self.timesteps = timesteps
        self.deterministic = True if degradation in ['blur', 'fadeblack', 'fadeblack_blur'] else False

        general_kwargs = {'timesteps': timesteps, 
                          'prediction': prediction,
                          'degradation': degradation, 
                          'noise_schedule': noise_schedule, 
                          'device': self.device,
                          'dataset': kwargs['dataset']}
        
        self.schedule = Scheduler()
        self.degrader = Degradation(**general_kwargs)
        self.reconstruction = Reconstruction(**general_kwargs)
        self.loss = Loss(**general_kwargs)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.99))
        self.apply_ema = not kwargs['skip_ema']
        self.test_run = kwargs['test_run']

        # Define Model EMA
        if self.apply_ema:
            self.ema_steps = kwargs['model_ema_steps']
            adjust = 1 * kwargs['batch_size'] * self.ema_steps / kwargs['epochs'] # The 1 in the beginning is symbolizing the number of distributed processes (1 for single GPU) 
            alpha = 1.0 - kwargs['model_ema_decay']
            alpha = min(1.0, alpha * adjust)
            self.model_ema = ExponentialMovingAverage(self.model, device=kwargs['device'], decay=1.0 - alpha).to(self.device)
        else:
            self.model_ema = model
            warnings.warn('No EMA applied')


    def train_iter(self, x_0):

        self.model.train()

        t = torch.randint(0, self.timesteps, (x_0.shape[0],), dtype=torch.long, device=self.device) # Randomly sample time steps

        if self.deterministic:
            x_t = self.degrader.degrade(x_0, t)
            residual = x_0 - x_t
        else:
            noise = torch.randn_like(x_0, device=self.device) # Important: Noise to degrade with must be the same as the noise that should be predicted
            x_t = self.degrader.degrade(x_0, t, noise=noise)
            residual = noise

        if self.prediction == 'residual':
            target = residual
            ret_x0 = False
        elif self.prediction == 'x0':
            target = x_0
            ret_x0 = True
        else:
            raise ValueError('Invalid prediction type')
        
        # Get Model prediction with correct output
        model_pred = self.model(x_t, t)
        pred = self.reconstruction.reform_pred(model_pred, x_t, t, return_x0=ret_x0) # Model prediction in correct form with coefficients applied

        if self.deterministic: 
            loss = self.loss.mse_loss(target, pred)
            #loss = self.loss.cold_loss(target, pred, t)
            #loss = self.loss.darras_loss(target, pred, t)
        else:
            loss = self.loss.mse_loss(target, pred)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
    
        return loss.item()
    
    
    def train_epoch(self, trainloader, valloader, val = False):
        
        epoch_loss = 0  
        for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
            x_0, _ = data
            x_0 = x_0.to(self.device)
            epoch_loss += self.train_iter(x_0)

            if self.apply_ema and i % self.ema_steps==0:
                self.model_ema.update_parameters(self.model)

            if self.test_run:
                break

        val_loss = 0
        if val:
            print("Validation")
            for x_0, _ in tqdm(valloader, total=len(valloader)):
                x_0 = x_0.to(self.device)
                val_loss += self.train_iter(x_0)
        
            
        return epoch_loss/len(trainloader), val_loss/len(valloader)


    
class Sampler:
    
    def __init__(self, timesteps, prediction, noise_schedule, degradation, fixed_seed = True, **kwargs):
        self.degradation = Degradation(timesteps=timesteps, degradation=degradation, prediction=prediction, noise_schedule=noise_schedule, **kwargs)
        self.reconstruction = Reconstruction(timesteps=timesteps, prediction=prediction, degradation = degradation, noise_schedule=noise_schedule, **kwargs)
        self.prediction = prediction
        self.timesteps = timesteps
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
        self.deterministic = True if degradation in ['blur', 'fadeblack', 'fadeblack_blur'] else False
        self.fixed_seed = fixed_seed
        self.gmm = None
        self.break_symmetry = kwargs['add_noise']

        if fixed_seed:
            torch.manual_seed(torch.randint(100000, (1,)).item())
        

    def fit_gmm(self, dataloader, clusters = 1, sample = False):
        """
        Function to fit a Gaussian Mixture Model to the mean of data in the dataloader. 
        Used to sample cold start images for deblurring diffusion.

        :param GMM: The Gaussian Mixture Model class
        :param DataLoader dataloader: The dataloader containing the data
        :param int clusters: The number of clusters in the Gaussian Mixture Model
        """

        # Fit GMM for cold sampling
        all_samples = None # Initialize all_samples
        for i, data in enumerate(dataloader, 0):
            img, _ = data
            img = torch.mean(img, [2, 3])
            if all_samples is None:
                all_samples = img
            else:
                all_samples = torch.cat((all_samples, img), dim=0)

        all_samples = all_samples

        self.gmm = GaussianMixture(n_components=clusters, covariance_type='full', tol = 0.001)

        self.gmm.fit(all_samples)
        print("GMM fitted")


    def sample_x_T(self, batch_size, channels, image_size):
        """
        Function to sample x_T either from a Gaussian Mixture Model or from a random normal distribution.

        :param int batch_size: The batch size of the samples
        :param int channels: The number of channels in the samples
        :param int image_size: The size of the images in the samples
        """

        if self.deterministic:

            # Sample x_T from GMM
            if self.gmm is None:
                raise ValueError('GMM not fitted, please fit GMM before cold sampling')
            else:
                assert isinstance(self.gmm, GaussianMixture), 'GMM not fitted correctly'
                channel_means = self.gmm.sample(n_samples=batch_size)[0] # Sample from GMM
                channel_means = torch.tensor(channel_means, dtype=torch.float32, device=self.device)
                channel_means = channel_means.unsqueeze(2).unsqueeze(3)
                x_t = channel_means.expand(batch_size, channels, image_size, image_size) # Expand the channel-wise means to the correct dimensions to build x_T
                x_t = x_t.float()
                    
            # Noise injection for breaking symmetry
            # Original code: noise_levels = [0.001, 0.002, 0.003, 0.004] # THIS GIVES US A HINT THAT THE NOISE LEVELS HAVE TO BE FINELY TUNED
            noise_level = 0.002
            if self.break_symmetry:
                x_t = x_t + torch.randn_like(x_t, device=self.device) * noise_level
        
        else:
            # Sample x_T from random normal distribution
            x_t = torch.randn((batch_size, channels, image_size, image_size), device=self.device)
        
        self.x_T = x_t

        print("x_T sampled and fixed")


    @torch.no_grad() 
    def sample_ddpm(self, model, batch_size, return_trajectory):

        model.eval()

        # Sample x_T either every time new or once and keep it fixed 
        if self.x_T is None:
            x_t = self.sample_x_T(batch_size, model.channels, model.image_size)
        else:
            x_t = self.x_T
        
        samples = []
        for t in tqdm(reversed(range(self.timesteps)), desc="DDPM Sampling"):
            samples.append(x_t) 
            t = torch.full((batch_size,), t).to(self.device)
            z = torch.randn((batch_size, model.channels, model.image_size, model.image_size)).to(self.device)
            posterior_mean_x0, posterior_mean_xt, posterior_var = self.reconstruction.coefs.posterior(t) # Get coefficients for the posterior distribution q(x_{t-1} | x_t, x_0)
            model_pred = model(x_t, t)
            x_0_hat = self.reconstruction.reform_pred(model_pred, x_t, t, return_x0 = True) # Obtain the estimate of x_0 at time t to sample from the posterior distribution q(x_{t-1} | x_t, x_0)
            x_0_hat.clamp_(-1, 1) # Clip the estimate to the range [-1, 1]
            x_t_m1 = posterior_mean_xt * x_t + posterior_mean_x0 * x_0_hat + torch.sqrt(posterior_var) * z # Sample x_{t-1} from the posterior distribution q(x_{t-1} | x_t, x_0)
            x_t = x_t_m1
        
        return x_t.unsqueeze(0) if not return_trajectory else samples
    

    @torch.no_grad() 
    def sample_ddim(self, model, batch_size, return_trajectory):
        
        # Sample x_T either every time new or once and keep it fixed 
        if self.x_T is None:
            x_t = self.sample_x_T(batch_size, model.channels, model.image_size)
        else:
            x_t = self.x_T
        
        # To be implemented
            
        pass


    @torch.no_grad()
    def sample_cold(self, model, batch_size, return_trajectory = True):
        
        # Sample x_T either every time new or once and keep it fixed 
        if self.x_T is None:
            x_t = self.sample_x_T(batch_size, model.channels, model.image_size)
        else:
            x_t = self.x_T

        symm_string = 'with broken symmetry' if self.break_symmetry else ''

        samples = []
        for t in tqdm(reversed(range(1, self.timesteps)), desc=f"Cold Sampling {symm_string}"):
            samples.append(x_t) 
            t_tensor = torch.tensor([t], dtype=torch.long).repeat(x_t.shape[0]).to(self.device)
            model_pred = model(x_t, t_tensor)
            x_0_hat = self.reconstruction.reform_pred(model_pred, x_t, t_tensor, return_x0 = True) # Obtain the estimate of x_0 at time t to sample from the posterior distribution q(x_{t-1} | x_t, x_0)
            x_tm1 = x_t - self.degradation.degrade(x_0_hat, t_tensor) + self.degradation.degrade(x_0_hat, t_tensor - 1)
            x_t = x_tm1 

        samples.append(x_t)
        return x_t.unsqueeze(0) if not return_trajectory else samples


    def sample(self, model, batch_size, return_trajectory = True):
        if self.deterministic:
            return self.sample_cold(model, batch_size, return_trajectory)
        else:
            return self.sample_ddpm(model, batch_size, return_trajectory)
        

                
class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    def __init__(self, model, decay, device="mps"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=True)




## Current Problem: It seems that in the standard Denoising Diffusion, we subtract the complete noise to obtain an estimate of x0 at each sampling step, while for deblurring, it seems that we only subtract the noise from the previous step. This is not clear in the paper.