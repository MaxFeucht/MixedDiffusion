import torch
import torch.nn as nn
import torch.nn.functional as F
import torchgeometry as tgm
import math
from tqdm import tqdm


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

    
    def get_blur_schedule(self, timesteps):
        
        # 0.01 * t + 0.35 CIFAR-10
        
        # MNIST:
        # Rissanen: Max of 20, min of 0.5, interpolated --> Investigate more clearly in the future and also use Hoogeboom for further explanation (blurring schedule slightly better explained there)
        # Bansal: Constant 7, recursive application
        
        return torch.linspace(0, timesteps-1, timesteps)

        
        
class Degradation:
    
    def __init__(self, timesteps, degradation, noise_schedule, dataset, **kwargs):
        
        self.timesteps = timesteps
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

        
        assert degradation in ['noise', 'blur', 'fadeblack', 'fadeblack_blur'], 'Invalid degradation type, choose from noise, blur, fadeblack, fadeblack_blur'
        self.degradation = degradation
        
        assert dataset in ['mnist', 'cifar', 'celeba'], 'Invalid dataset'
        
        # Default MNIST settings
        blur_kwargs = {'channels': 1, 
                        'kernel_size': 11, 
                        'kernel_std': 2, 
                        'timesteps': timesteps, 
                        'blur_routine': 'Constant'}
            
        self.blur = Blurring(**blur_kwargs)
        self.noise_coefs = DenoisingCoefs(timesteps=timesteps, noise_schedule=noise_schedule, device = self.device)


    def noising(self, x_0, t):
        """
        Function to add noise to an image x at time t.
        
        :param torch.Tensor x_0: The original image
        :param int t: The time step
        :return torch.Tensor: The degraded image at time t
        """

        x_0_coef, residual_coef = self.noise_coefs.forward_process(t)
        x_0_coef, residual_coef = x_0_coef.to(self.device), residual_coef.to(self.device)
        x_t = x_0_coef * x_0 + residual_coef * torch.randn_like(x_0, device = self.device)
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

        # Apply Blurring with the same intensity (t) to all images in batch
        if isinstance(t, torch.Tensor):
            t = t[0].item()

        for i in range(t):
            x = x.unsqueeze(0) if len(x.shape) == 2  else x
            x = gaussian_kernels[i](x).squeeze(0) 

            # Make sure gradients are retained and kernels are frozen
            if x_0.requires_grad:      
                assert gaussian_kernels[i].requires_grad == False
                assert x.requires_grad == True 
        return x

    def blacking(self, x_0, t):
        """
        Function to fade an image x to black at time t.
        
        :param torch.Tensor x_0: The original image
        :param int t: The time step
        :return torch.Tensor: The degraded image at time t
        """
        
        x_t = x_0 * (1 - (t+1) / self.timesteps) # +1 bc of zero indexing
        return x_t
    
    
    def degrade(self, x, t):
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
            return self.noising(x, t)
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
            
            self.channels = channels
            self.kernel_size = kernel_size
            self.kernel_std = kernel_std
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
            if self.blur_routine == 'Incremental':
                kernels.append(self.get_conv((self.kernel_size, self.kernel_size), (self.kernel_std*(i+1), self.kernel_std*(i+1)) ) )
            elif self.blur_routine == 'Constant': # For MNIST
                kernels.append(self.get_conv((self.kernel_size, self.kernel_size), (self.kernel_std, self.kernel_std)))
            elif self.blur_routine == 'Exponential':
                ks = self.kernel_size
                kstd = torch.exp(self.kernel_std * i)
                kernels.append(self.get_conv((ks, ks), (kstd, kstd)))
        
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
        
        x0_coef = torch.sqrt(self.alphas_cumprod.gather(-1, t)).reshape(-1, 1, 1, 1)
        residual_coef =  torch.sqrt(1. - self.alphas_cumprod.gather(-1, t)).reshape(-1, 1, 1, 1)
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

    def model_prediction(self, model, x_t, t, return_x0 = False):
        """
        Function to obtain predictions for a given degraded image x_t at time t, using a trained model and a degrader function.
        
        :param torch.Tensor x_t: The degraded image at time t
        :param int t: The time step
        :return torch.Tensor: The predicted image at time t-1
        """
        
        if not self.deterministic: 
            xt_coef, residual_coef = self.coefs.x0_restore(t)
        else:
            xt_coef, residual_coef = torch.tensor(1.), torch.tensor(1.)
                
        if self.prediction == 'residual':
            residual = model(x_t, t)
            if not return_x0:
                return residual
            else:
                x0_estimate = xt_coef * x_t - residual_coef * residual 
                return x0_estimate      
            
        elif self.prediction == 'x0':
            x0_estimate = model(x_t, t)
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
        return F.mse_loss(pred, target)
    
    def cold_loss(self, target, pred, t):
        diff = pred - target
        return diff.mean()  # Mean over batch dimension

    def darras_loss(self, target, pred, t):
        diff = pred - target # difference between the predicted and the target image / residual
        degraded_diff = self.degradation.degrade(diff, t)**2 # Move this difference into the degradation space and square it
        return degraded_diff.mean()  # Squared error loss, averaged over batch dimension



class Trainer:
    
    def __init__(self, model, model_ema, lr, timesteps, prediction, degradation, noise_schedule, **kwargs):

        self.device = kwargs['device']
        self.model = model.to(self.device)
        self.model_ema = model_ema.to(self.device)
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
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.apply_ema = not kwargs['skip_ema']
        self.ema_steps = kwargs['model_ema_steps']

        # To Do: Save and load model and optimizer
        # if os.path.exists('./checkpoints'):
        #     self.model.load_state_dict(torch.load('./checkpoints/model.pth'))
        #     self.optimizer.load_state_dict(torch.load('./checkpoints/optimizer.pth'))
        #     print("Model and optimizer loaded")


    def train_iter(self, x_0):
        
        t = torch.randint(0, self.timesteps, (x_0.shape[0],), dtype=torch.long, device=self.device) # Randomly sample time steps
        x_t = self.degrader.degrade(x_0, t)
        pred = self.reconstruction.model_prediction(self.model, x_t, t, return_x0=False) # Model prediction in correct form with coefficients applied

        if self.deterministic: # Residual either random noise for denoising diffusion or difference between x_0 and x_t for deterministic degradation
            residual = x_0 - x_t
        else:
            residual = torch.randn_like(x_0)
        
        if self.prediction == 'residual':
            target = residual
        elif self.prediction == 'x0':
            target = x_0
        else:
            raise ValueError('Invalid prediction type')
        
        if self.deterministic: 
            #loss = self.loss.cold_loss(target, pred, t)
            #loss = self.loss.darras_loss(target, pred, t)
            loss = self.loss.mse_loss(target, pred)
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

        if fixed_seed:
            torch.manual_seed(torch.randint(100000, (1,)).item())

    @torch.no_grad() 
    def sample_ddpm(self, model, batch_size, return_trajectory = True):
        
        x_t = torch.randn((batch_size, model.channels, model.image_size, model.image_size)).to(self.device)
        samples = []
        
        for t in tqdm(reversed(range(self.timesteps)), desc="DDPM Sampling"):
            samples.append(x_t) 
            t = torch.full((batch_size,), t).to(self.device)
            z = torch.randn((batch_size, model.channels, model.image_size, model.image_size)).to(self.device)
            posterior_mean_x0, posterior_mean_xt, posterior_var = self.reconstruction.coefs.posterior(t) # Get coefficients for the posterior distribution q(x_{t-1} | x_t, x_0)
            x0_estimate = self.reconstruction.model_prediction(model, x_t, t, return_x0 = True) # Obtain the estimate of x_0 at time t to sample from the posterior distribution q(x_{t-1} | x_t, x_0)
            x0_estimate.clamp_(-1, 1) # Clip the estimate to the range [-1, 1]
            x_t_m1 = posterior_mean_xt * x_t + posterior_mean_x0 * x0_estimate + torch.sqrt(posterior_var) * z # Sample x_{t-1} from the posterior distribution q(x_{t-1} | x_t, x_0)

            x_t = x_t_m1
            
        return x_t if not return_trajectory else samples
    

    @torch.no_grad() 
    def sample_ddim(self, model, batch_size, return_trajectory = True):
        ## To be implemented
        pass


    @torch.no_grad()
    def sample_cold(self, model, batch_size, return_trajectory = True):

        # Initialize an empty tensor to store the batch
        x_t = torch.empty(batch_size, model.channels, model.image_size, model.image_size, device=self.device)

        # Fill each depth slice with a single decimal drawn uniformly from [0, 1]
        for i in range(batch_size):
            for j in range(model.channels):
                x_t[i, j, :, :] = torch.full((model.image_size, model.image_size), torch.rand(1).item(), dtype=torch.float32)

        samples = []
        for t in tqdm(reversed(range(self.timesteps)), desc="Cold Sampling"):
            samples.append(x_t) 
            t_tensor = torch.tensor([t]).repeat(x_t.shape[0]).float().to(self.device)
            x_0_hat = model(x_t, t_tensor)
            x_tm1 = x_t -  self.degradation.degrade(x_0_hat, t) + self.degradation.degrade(x_0_hat, t-1)
            x_t = x_tm1 

        return x_t if not return_trajectory else x_t


    def sample(self, model, batch_size, return_trajectory = False):
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

    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=True)




## Current Problem: It seems that in the standard Denoising Diffusion, we subtract the complete noise to obtain an estimate of x0 at each sampling step, while for deblurring, it seems that we only subtract the noise from the previous step. This is not clear in the paper.