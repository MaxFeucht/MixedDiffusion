import torch
import torch.nn as nn
import torch.nn.functional as F
import torchgeometry as tgm
from sklearn.mixture import GaussianMixture

import math
from tqdm import tqdm
from utils import save_image
import os

import dct_blur as torch_dct
import numpy as np

from scripts.vae_unet_full import VAEEncoderStandAlone as VAEEncoder

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
        
        if type == 'cifar':
            return torch.arange(timesteps, dtype=torch.float32)/100 + 0.35
       
        
        
class Degradation:
    
    def __init__(self, timesteps, degradation, noise_schedule, dataset, **kwargs):
        
        self.timesteps = timesteps
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

        
        assert degradation in ['noise', 'blur', 'fadeblack', 'fadeblack_blur'], 'Invalid degradation type, choose from noise, blur, fadeblack, fadeblack_blur'
        self.degradation = degradation
        
        assert dataset in ['mnist', 'cifar10', 'celeba', 'lsun_churches'],f"Invalid dataset, choose from ['mnist', 'cifar10', 'celeba', 'lsun_churches']"
        
        # Default settings
        blur_kwargs = {'channels': 1 if dataset == 'mnist' else 3, 
                        'kernel_size': 11 if dataset == 'mnist' else 11, # Change to 11 for non-cold start but for conditional sampling (only blurring for 40 steps)
                        'kernel_std': 7 if dataset == 'mnist' else 7, # if dataset == 'mnist' else 0.001, # Std has a different interpretation for constant schedule and exponential schedule: constant schedule is the actual std, exponential schedule is the rate of increase # 7 if dataset == 'mnist' else 0.01
                        'timesteps': timesteps, 
                        'blur_routine': 'cifar' if dataset == 'cifar10' else 'constant' if dataset == 'mnist' else 'exponential',
                        'mode': 'circular' if dataset == 'mnist' else 'reflect',
                        'device': self.device} # if dataset == 'mnist' else 'exponential'} # 'constant' if dataset == 'mnist' else 'exponential'}
            
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

        if not hasattr(self.blur, 'gaussian_kernels'):
            self.blur.get_kernels()
        
        assert self.blur.gaussian_kernels is not None, 'Gaussian Kernels not initialized'

        # Freeze kernels
        for kernel in self.blur.gaussian_kernels:
            kernel.requires_grad = False

        self.blur.gaussian_kernels.to(self.device)  # Move kernels to GPU
        x = x_0
        
        # Keep gradients for the original image for backpropagation
        if x_0.requires_grad:
            x.retain_grad()

        t_max = torch.max(t)

        # Blur all images to the max, but store all intermediate blurs for later retrieval         
        max_blurs = []

        if t_max+1 == 0:
            max_blurs.append(x)
        else:
            for i in range(t_max + 1): ## +1 to account for zero indexing of range
                x = x.unsqueeze(0) if len(x.shape) == 2  else x
                x = self.blur.gaussian_kernels[i](x).squeeze(0) 
                if i == (self.timesteps-1):
                    x = torch.mean(x, [2, 3], keepdim=True)
                    x = x.expand(x_0.shape[0], x_0.shape[1], x_0.shape[2], x_0.shape[3])

                max_blurs.append(x)
        
        max_blurs = torch.stack(max_blurs)

        # Choose the correct blur for each image in the batch
        blur_t = []
        for step in range(t.shape[0]):
            if t[step] != -1:
                blur_t.append(max_blurs[t[step], step])
            else:
                blur_t.append(x_0[step])

        return torch.stack(blur_t)


    def dct_blurring(self, x_0, t):

        sigmas = self.blur.dct_sigmas

        if not hasattr(self, 'dct_blur'):
            self.dct_blur = DCTBlur(sigmas, x_0.shape[-1], self.device)

        xt = self.dct_blur(x_0, t).float()
        return xt
        


    def blacking(self, x_0, t, factor = 0.95, mode = 'linear'):
        """
        Function to fade an image x to black at time t.
        
        :param torch.Tensor x_0: The original image
        :param int t: The time step
        :return torch.Tensor: The degraded image at time t
        """
        # if t == self.timesteps-1:
        #     multiplier = 0.0
        # else:
        #     multiplier = factor ** (t)  # +1 bc of zero indexing
        
        if mode == 'linear':
            multiplier = (1 - (t+1) / (self.timesteps)).reshape(-1, 1, 1, 1)  # +1 bc of zero indexing
        
        elif mode == 'exponential':
            if t == self.timesteps-1:
                multiplier = 0.0
            else:
                multiplier = factor ** (t)  # +1 bc of zero indexing

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
    
    def __init__(self, timesteps, channels, kernel_size, kernel_std, blur_routine, mode, device):
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
            self.mode = mode
            self.device = device
        

    def get_conv(self, dims, std, mode):
        """
        Function to obtain a 2D convolutional layer with a Gaussian Blurring kernel.
        
        :param tuple dims: The dimensions of the kernel
        :param tuple std: The standard deviation of the kernel
        :param str mode: The padding mode
        :return nn.Conv2d: The 2D convolutional layer with the Gaussian Blurring kernel
        """
        
        kernel = tgm.image.get_gaussian_kernel2d(dims, std) 
        conv = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=dims, padding=int((dims[0]-1)/2), padding_mode=mode,
                         bias=False, groups=self.channels)
        
        kernel = torch.unsqueeze(kernel, 0)
        kernel = torch.unsqueeze(kernel, 0)
        kernel = kernel.repeat(self.channels, 1, 1, 1)
        conv.weight = nn.Parameter(kernel, requires_grad=False)

        return conv


    def get_kernels(self):
        """
        Function to obtain a list of 2D convolutional layers with Gaussian Blurring kernels following a certain routine.
        
        :return list: A list of 2D convolutional layers with Gaussian Blurring kernels
        """
        
        kernels = []
        for i in range(self.num_timesteps):
            kernels.append(self.get_conv((self.kernel_size, self.kernel_size), (self.kernel_stds[i], self.kernel_stds[i]), mode = self.mode)) 
        
        self.gaussian_kernels = nn.ModuleList(kernels).to(self.device)
    
    def get_dct_sigmas(self, image_size):

        # Check how Risannen does it, and experiment with Simon's script

        sigmas = []
        for i in range(self.num_timesteps):
            #sigma = 0.1 * (image_size / 64) * (i + 1)
            sigmas.append(sigma)
        return sigmas
    


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
    
    def __init__(self, model, lr, timesteps, prediction, degradation, noise_schedule, vae, vae_alpha, **kwargs):

        self.device = kwargs['device']
        self.model = model.to(self.device)
        self.prediction = prediction
        self.timesteps = timesteps
        self.deterministic = True if degradation in ['blur', 'fadeblack', 'fadeblack_blur'] else False
        self.vae = vae
        self.vae_alpha = vae_alpha
        self.noise_scale = kwargs['noise_scale']
        self.vae_xt = kwargs['vae_full']
        self.vae_downsample = kwargs['vae_downsample']

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
        self.test_run = True if kwargs['test_run'] else False

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

        if self.vae_xt:

            latent_dim = model.channels * model.image_size * model.image_size
            latent_dim = int(latent_dim // kwargs['vae_downsample'])

            self.vae_model = VAEEncoder(dim=model.ch, 
                            dim_mult=model.ch_mult, 
                            latent_dim=latent_dim, # Returns flattened image latents
                            image_size=model.image_size, 
                            channels=model.channels, 
                            temb_channels=model.temb_ch,
                            dropout=0)
            self.vae_model = self.vae_model.to(self.device)


    def train_iter(self, x_0, t):

        # Degrade and obtain residual
        if self.deterministic:
            x_t = self.degrader.degrade(x_0, t) 
            
            # Add noise to degrade with - Noise injection a la Bansal
            # if not self.vae_xt:
            #     x_t = x_t + torch.randn_like(x_0, device=self.device) * self.noise_scale

            x_tm1 = self.degrader.degrade(x_0, t-1)
            residual = x_tm1 - x_t 
        else:
            noise = torch.randn_like(x_0, device=self.device) # Important: Noise to degrade with must be the same as the noise that should be predicted
            x_t = self.degrader.degrade(x_0, t, noise=noise)
            residual = noise

        # Define prediction and target
        if self.prediction == 'residual':
            target = residual
            ret_x0 = False
        elif self.prediction == 'x0':
            target = x_0
            ret_x0 = True
        elif self.prediction == 'xtm1':
            assert self.deterministic, 'xtm1 prediction not compatible with denoising diffusion'
            # if t[0].item() == 0:
            #     assert torch.all(torch.eq(x_tm1[0], x_0[0])), 'x_tm1 not equal to x_0 for last step'
            target = x_tm1
        else:
            raise ValueError('Invalid prediction type')
        
        # Get Model prediction with correct output and select appropriate loss
        if self.vae: 
            model_pred = self.model(x_t, t, x_tm1) # VAE Model needs x_tm1 for prediction. Important: x_tm1 is needed for VAE to run in training mode

            # Testing to include VAE Noise into Loss, just as in Risannen. 
            # We do this by adding the noise to x_t and let the model optimize for the difference between the perturbed x_t and xtm1.
            x_t = x_t + self.model.vae_noise 

            pred = (x_t + model_pred) if self.prediction == 'xtm1' else model_pred # According to Risannen, the model predicts the residual
            
            reconstruction = self.loss.mse_loss(target, pred)
            kl_div = self.model.kl_div
            loss = 2 * (self.vae_alpha * reconstruction + (1-self.vae_alpha) * kl_div) #* self.noise_scale)
            return loss, reconstruction, kl_div
        
        else:

            # Full scale external VAE injection
            if self.vae_xt:
                vae_mu, vae_logvar = self.vae_model(x_t, x_tm1, t)
                z_sample = torch.randn_like(vae_mu) * torch.exp(0.5*vae_logvar) + vae_mu

                # Repeat z_sample to match image dimensions
                if self.vae_downsample > 1:
                    z_sample = z_sample.repeat(1, self.vae_downsample) # Flattened image latents, repeated on non-batch dimensions

                # KL Divergence for VAE Encoder
                kl_div = 0.5 * (vae_mu.pow(2) + vae_logvar.exp() - 1 - vae_logvar).sum(1).mean()

                bs, ch, res = x_t.shape[0], x_t.shape[1], x_t.shape[2]
                x_t = x_t + z_sample.reshape(bs, ch, res, res) * self.noise_scale

            model_pred = self.model(x_t, t)

            # Risannen Loss Formulation - Here the slightly perturbed x_t is used for the prediction
            pred = (x_t + model_pred) if self.prediction == 'xtm1' else model_pred

            if not self.deterministic:
                pred = self.reconstruction.reform_pred(pred, x_t, t, return_x0=ret_x0) # Model prediction in correct form with coefficients applied
            loss = self.loss.mse_loss(target, pred)

            if self.vae_xt:
                loss = self.vae_alpha * loss + (1-self.vae_alpha) * kl_div

            return loss
    
    
    def train_epoch(self, dataloader, val = False):
        
        # Set model to train mode
        if not val:
            assert self.model.train(), 'Model not in training mode'
        else:
            assert self.model.eval(), 'Model not in evaluation mode'

        # Iterate through trainloader
        epoch_loss = 0  
        epoch_reconstruction = 0
        epoch_kl_div = 0
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            x_0, _ = data
            x_0 = x_0.to(self.device)
            
            # Sample t
            t = torch.randint(0, self.timesteps, (x_0.shape[0],), dtype=torch.long, device=self.device) # Randomly sample time steps
            #t = torch.randint(10, 11, (x_0.shape[0],), dtype=torch.long, device=self.device) # Randomly sample time steps

            if self.vae:
                loss, reconstruction, kl_div = self.train_iter(x_0, t)
                epoch_reconstruction += reconstruction.item()
                epoch_kl_div += kl_div.item()
            else:
                loss = self.train_iter(x_0, t)
            
            epoch_loss += loss.item()

            # Implement Gradient Accumulation
            if not val:
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.apply_ema and i % self.ema_steps==0:
                    self.model_ema.update_parameters(self.model)
            
            # Break prematurely if args.test_run
            if self.test_run:
                break
        
        if self.vae:
            return epoch_loss/len(dataloader), epoch_reconstruction/len(dataloader), epoch_kl_div/len(dataloader)
        else:
            return epoch_loss/len(dataloader)

    
class Sampler:
    
    def __init__(self, timesteps, prediction, noise_schedule, degradation, **kwargs):
        self.degradation = Degradation(timesteps=timesteps, degradation=degradation, prediction=prediction, noise_schedule=noise_schedule, **kwargs)
        self.reconstruction = Reconstruction(timesteps=timesteps, prediction=prediction, degradation = degradation, noise_schedule=noise_schedule, **kwargs)
        self.prediction = prediction
        self.timesteps = timesteps
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
        self.deterministic = True if degradation in ['blur', 'fadeblack', 'fadeblack_blur'] else False
        self.black = True if degradation in ['fadeblack', 'fadeblack_blur'] else False
        self.gmm = None
        self.break_symmetry = kwargs['add_noise']
        self.vae = kwargs['vae']
        self.vae_xt = kwargs['vae_full']
        self.noise_scale = kwargs['noise_scale']
        self.vae_downsample = kwargs['vae_downsample']
        

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

        if self.deterministic and not self.black:

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
            noise_level = 0.001
            if self.break_symmetry:
                x_t = x_t + torch.randn_like(x_t, device=self.device) * noise_level
        
        elif self.black:
            # Sample x_T from R0
            x_t = torch.zeros((batch_size, channels, image_size, image_size), device=self.device) 
            
            # # Noise injection for breaking symmetry
            # noise_level = 0.001
            # if self.break_symmetry:
            #     x_t = x_t + torch.randn_like(x_t, device=self.device) * noise_level
        
        elif not self.deterministic:
            # Sample x_T from random normal distribution
            x_t = torch.randn((batch_size, channels, image_size, image_size), device=self.device)
        
        self.x_T = x_t

        print("x_T sampled and fixed")


    @torch.no_grad() 
    def sample_ddpm(self, model, batch_size):

        model.eval()

        # Sample x_T either every time new or once and keep it fixed 
        if self.x_T is None:
            x_t = self.sample_x_T(batch_size, model.channels, model.image_size)
        else:
            x_t = self.x_T
        
        ret_x_T = x_t

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
        
        return samples, ret_x_T


    @torch.no_grad() 
    def sample_ddim(self, model, batch_size):
        
        # Sample x_T either every time new or once and keep it fixed 
        if self.x_T is None:
            x_t = self.sample_x_T(batch_size, model.channels, model.image_size)
        else:
            x_t = self.x_T
        
        # To be implemented
            
        pass


    @torch.no_grad()
    def sample_cold(self, model, batch_size = 16, x0=None, generate=False, prior=None, t_inject=None):

        model.eval()

        t=self.timesteps

        if not hasattr(self.degradation.blur, 'gaussian_kernels'):
            self.degradation.blur.get_kernels()

        # Sample x_T either every time new or once and keep it fixed 
        if generate:
            if self.x_T is None:
                xT = self.sample_x_T(batch_size, model.channels, model.image_size)
            else:
                xT = self.x_T
        else:
            t_tensor = torch.full((batch_size,), t-1, dtype=torch.long).to(self.device) # t-1 to account for 0 indexing and the resulting t+1 in the degradation operation
            xT = self.degradation.degrade(x0, t_tensor) # Adaption due to explanation below (0 indexing)
            
        xt = xT

        direct_recons = None
        samples = []
        samples.append(xT) 

        for t in tqdm(reversed(range(self.timesteps)), desc=f"Cold Sampling"):
            t_tensor = torch.full((batch_size,), t, dtype=torch.long).to(self.device) # t-1 to account for 0 indexing that the model is seeing during training
            
            # External VAE Injection - THIS IS SAMPLING NOISE
            # if self.vae_xt:
                    
            #         # Sample from VAE prior (and repeat to match image dimensions in necessary)
            #         latent_dim = (model.channels * model.image_size * model.image_size) // self.vae_downsample
            #         noise = torch.randn((batch_size, latent_dim), device=self.device) * self.noise_scale

            #         bs, ch, res = xt.shape[0], xt.shape[1], xt.shape[2]
            #         noise = noise.reshape(bs, ch, res, res) * self.noise_scale
                    
            #         if self.vae_downsample > 1:
            #             noise = noise.repeat(1, self.vae_downsample) # Flattened image latents, repeated on non-batch dimensions

            #         xt = xt + noise

            if self.vae:
                if generate:
                    if t_inject is not None: # T_Inject aims to assess the effect of manipulated injections at different timestep
                        pred = model(xt, t_tensor, xtm1=None, prior=prior if t == t_inject else None)
                    else: # If no t_inject is provided, we always use the provided prior
                        pred = model(xt, t_tensor, xtm1=None, prior=prior)
                else:
                    # Reconstruction with encoded latent from x0 ground truth
                    xtm1 = self.degradation.degrade(x0, t_tensor-1) # Adaption due to explanation below (0 indexing) 
                    pred = model(xt, t_tensor, xtm1)
                
                xt = xt + model.vae_noise

            else:                    
                pred = model(xt, t_tensor)

            # BANSAL ALGORITHM 2
            if self.prediction == 'x0':
                x0_hat = pred
                xt_hat = self.degradation.degrade(x0_hat, t_tensor)
                xtm1_hat = self.degradation.degrade(x0_hat, t_tensor - 1) # This returns x0_hat for t=0
                xtm1 = xt - xt_hat + xtm1_hat
                xt = xtm1
                samples.append(xt)

                if direct_recons == None:
                    direct_recons = x0_hat
    

            # OURS with xt prediction
            elif self.prediction == 'xtm1':
                xtm1_hat = xt + pred # According to Risannen, the model predicts the residual
                xt = xtm1_hat
                samples.append(xt)

                # if x0 is not None:
                #     x0_hat = self.degradation.degrade(xt, t_tensor-1)
                #     samples.append(x0_hat)

        

            # OURS with residual prediction
            elif self.prediction == 'residual':
                residual = pred
                xt = xt + residual
                samples.append(xt)

                
        return samples, xT, direct_recons, xt

    
    @torch.no_grad()
    def sample_cold_orig(self, model, batch_size = 16, x0=None, generate=False, prior=None, t_inject=None):

        model.eval()

        t=self.timesteps

        if not hasattr(self.degradation.blur, 'gaussian_kernels'):
            self.degradation.blur.get_kernels()
        
        # Decide whether to generate x_T or use the degraded input image (reconstruction)
        if generate:
            # Sample x_T either every time new or once and keep it fixed 
            if self.x_T is None:
                xT = self.sample_x_T(batch_size, model.channels, model.image_size)
            else:
                xT = self.x_T
        else:
            # for i in range(t):
            #     with torch.no_grad():
            #         img = self.degradation.blur.gaussian_kernels[i](img)
            #         if i == (self.timesteps-1):
            #             img = torch.mean(img, [2, 3], keepdim=True)
            #             img = img.expand(x0.shape[0], x0.shape[1], x0.shape[2], x0.shape[3])
            
            t_tensor = torch.full((batch_size,), t-1, dtype=torch.long).to(self.device) # t-1 to account for 0 indexing and the resulting t+1 in the degradation operation
            xT = self.degradation.degrade(x0, t_tensor) # Adaption due to explanation below (0 indexing)
            

        img = xT

        direct_recons = None
        #while(t):
        samples = []
        samples.append(xT) 
        for t_step in tqdm(reversed(range(t)), desc="Cold Original Sampling"):
            step = torch.full((batch_size,), t_step, dtype=torch.long).to(self.device) # t-1 to account for 0 indexing that the model is seeing during training
            
            if generate:
                # if t_step == t_inject: # Check what injections at different timesteps do
                #     x0_hat = model(img, step, prior=prior)
                # else:
                #     x0_hat = model(img, step, prior=None)
                
                x0_hat = model(img, step, prior=prior)

            else:
                # Bootstrapped sampling by using VAE-encoding derived from x0 predictions
                # Sample first x0_hat using VAE prior
                # if t_step == (t-1):
                #     x0_hat = model(img, step, prior=prior)

                # Reconstruction with encoded latent from x0 predictions
                xtm1_hat = self.degradation.degrade(x0, step-1) # Adaption due to explanation below (0 indexing)
                x0_hat = model(img, step, xtm1_hat)
                
            if direct_recons == None:
                direct_recons = x0_hat

            # x_times = x0_hat
            x_times = self.degradation.degrade(x0_hat, step) 
            # for i in range(t_step): #t+1 to account for 0 indexing of range
            #     with torch.no_grad():
            #         x_times = self.degradation.blur.gaussian_kernels[i](x_times)
            #         if i == (self.timesteps-1):
            #             x_times = torch.mean(x_times, [2, 3], keepdim=True)
            #             x_times = x_times.expand(temp.shape[0], temp.shape[1], temp.shape[2], temp.shape[3])


            # x_times_sub_1 = x0_hat
            x_times_sub_1 = self.degradation.degrade(x0_hat, step - 1)
            # for i in range(t_step-1): # actually t-1 but t because of 0 indexing
            #     with torch.no_grad():
            #         x_times_sub_1 = self.degradation.blur.gaussian_kernels[i](x_times_sub_1)

            xt_hat = img - x_times + x_times_sub_1
            img = xt_hat
            samples.append(xt_hat)
            #t = t -1
        
        model.train()

        return samples, xT, direct_recons, img



    def sample(self, model, batch_size, x0=None, prior=None, generate=False, t_inject=None):
        if self.deterministic:
            return self.sample_cold(model, 
                                    batch_size, 
                                    x0, 
                                    generate=generate, 
                                    prior=prior, 
                                    t_inject=t_inject)
        else:
            return self.sample_ddpm(model, batch_size)
        

                
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


class DCTBlur(nn.Module):

    def __init__(self, blur_sigmas, image_size, device):
        super(DCTBlur, self).__init__()
        self.blur_sigmas = torch.tensor(blur_sigmas).to(device)
        freqs = np.pi*torch.linspace(0, image_size-1,
                                    image_size).to(device)/image_size
        self.frequencies_squared = freqs[:, None]**2 + freqs[None, :]**2

    def forward(self, x, fwd_steps):
        if len(x.shape) == 4:
            sigmas = self.blur_sigmas[fwd_steps][:, None, None, None]
        elif len(x.shape) == 3:
            sigmas = self.blur_sigmas[fwd_steps][:, None, None]
        t = sigmas**2/2
        dct_coefs = torch_dct.dct_2d(x, norm='ortho')
        dct_coefs = dct_coefs * torch.exp(- self.frequencies_squared * t)
        return torch_dct.idct_2d(dct_coefs, norm='ortho')




## Current Problem: It seems that in the standard Denoising Diffusion, we subtract the complete noise to obtain an estimate of x0 at each sampling step, while for deblurring, it seems that we only subtract the noise from the previous step. This is not clear in the paper.