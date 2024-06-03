import argparse
import numpy as np
import os
import sys
import time
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets
from torchvision import transforms as T
from torchvision.utils import save_image, make_grid

from unet import UNet
from mnist_unet import MNISTUnet
#from scripts.karras_unet import KarrasUnet
from scripts.bansal_unet import BansalUnet
from scripts.vae_unet import VAEUNet
from scripts.risannen_unet import RisannenUnet
from scripts.risannen_unet_vae import VAEUnet
from diffusion_utils import Degradation, Trainer, Sampler, ExponentialMovingAverage
from utils import load_dataset, create_dirs, save_video, save_gif, MyCelebA

import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

# Check if ipykernel is running to check if we're working locally or on the cluster
import sys
if 'ipykernel' in sys.modules:
    sys.argv = ['']


#%%

parser = argparse.ArgumentParser(description='Diffusion Models')

# General Diffusion Parameters
parser.add_argument('--timesteps', '--t', type=int, default=50, help='Degradation timesteps')
parser.add_argument('--prediction', '--pred', type=str, default='vxt', help='Prediction method, choose one of [x0, xt, residual]')
parser.add_argument('--dataset', type=str, default='mnist', help='Dataset to run Diffusion on. Choose one of [mnist, cifar10, celeba, lsun_churches]')
parser.add_argument('--degradation', '--deg', type=str, default='fadeblack_blur_bansal', help='Degradation method')
parser.add_argument('--batch_size', '--b', type=int, default=64, help='Batch size')
parser.add_argument('--dim', '--d', type=int , default=128, help='Model dimension')
parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
parser.add_argument('--epochs', '--e', type=int, default=20, help='Number of Training Epochs')
parser.add_argument('--noise_schedule', '--sched', type=str, default='cosine', help='Noise schedule')
parser.add_argument('--xt_weighting', action='store_true', help='Whether to use weighting for xt in loss')
parser.add_argument('--var_sampling_step', type=int, default = -1, help='How to sample var timestep model - int > 0 indicates t difference to predict, -1 indicates x0 prediction')
parser.add_argument('--baseline', '--base', type=str, default='xxx', help='Whether to run a baseline model - Risannen, Bansal, VAE')

# Noise Injection Parameters
parser.add_argument('--vae', action='store_false', help='Whether to use VAE Noise injections')
parser.add_argument('--vae_alpha', type=float, default = 0.999, help='Trade-off parameter for weight of Reconstruction and KL Div')
parser.add_argument('--latent_dim', type=int, default=4, help='Which dimension the VAE latent space is supposed to have')
parser.add_argument('--add_noise', action='store_true', help='Whether to add noise Risannen et al. style')
parser.add_argument('--break_symmetry', action='store_true', help='Whether to add noise to xT Bansal et al. style')
parser.add_argument('--noise_scale', type=float, default = 0.01, help='How much Noise to add to the input')
parser.add_argument('--vae_loc', type=str, default = 'start', help='Where to inject VAE Noise. One of [start, bottleneck, emb].')
parser.add_argument('--vae_inject', type=str, default = 'add', help='How to inject VAE Noise. One of [concat, add].')
parser.add_argument('--xt_dropout', type=float, default = 0.2, help='How much of xt is dropped out at every step (to foster reliance on VAE injections)')

# Housekeeping Parameters
parser.add_argument('--load_checkpoint', action='store_false', help='Whether to try to load a checkpoint')
parser.add_argument('--sample_interval', type=int, help='After how many epochs to sample', default=1)
parser.add_argument('--n_samples', type=int, default=10, help='Number of samples to generate')
parser.add_argument('--fix_sample', action='store_false', help='Whether to fix x_T for sampling, to see sample progression')
parser.add_argument('--skip_ema', action='store_true', help='Whether to skip model EMA')
parser.add_argument('--model_ema_decay', type=float, default=0.997, help='Model EMA decay')
parser.add_argument('--cluster', action='store_true', help='Whether to run script locally')
parser.add_argument('--verbose', '--v', action='store_true', help='Verbose mode')

parser.add_argument('--test_run', action='store_false', help='Whether to test run the pipeline')

args = parser.parse_args()

args.num_downsamples = 2 if args.dataset == 'mnist' else 3
args.device = 'cuda' if torch.cuda.is_available() else 'mps'

if args.dataset == 'mnist':
    args.image_size = 28
elif args.dataset == 'cifar10':
    args.image_size = 32
elif args.dataset == 'afhq':
    args.image_size = 64

if args.vae:
    print("Using VAE Noise Injections")
    assert not args.add_noise, "Cannot use VAE and add noise at the same time"
else:
    if args.add_noise:
        print("Using Risannen Noise Injections")
    else:
        print("Using Normal U-Net")

if not args.cluster:
    print("Running locally, Cluster =", args.cluster)
    # args.dim = int(args.dim/2)
    if args.device == 'cuda':
        warnings.warn('Consider running model on cluster-scale if CUDA is available')

if args.test_run:
    print("Running Test Run with only one iter per epoch")

print("Device: ", args.device)


kwargs = vars(args)

trainloader, valloader = load_dataset(kwargs['batch_size'], kwargs['dataset'])

x, _ = next(iter(trainloader))   
channels = x[0].shape[0]

# Model Configuration
if kwargs['dataset'] == 'mnist':
    attention_levels = (2,)
    ch_mult = (1,2,2)
    num_res_blocks = 2
elif kwargs['dataset'] == 'cifar10':
    attention_levels = (2,3)
    ch_mult = (1, 2, 2, 2)
    num_res_blocks = 4
elif kwargs['dataset'] == 'afhq':
    attention_levels = (2,3)
    ch_mult = (1, 2, 3, 4)
    num_res_blocks = 2
elif kwargs['dataset'] == 'celeba':
    attention_levels = (2,3)
    ch_mult = (1, 2, 2, 2)
elif kwargs['dataset'] == 'lsun_churches':
    attention_levels = (2,3,4)
    ch_mult = (1, 2, 3, 4, 5)
    num_res_blocks = 4


# Define Model
if kwargs['vae']:


    # Risannen Version
    unet = VAEUnet(image_size=kwargs["image_size"],
                    in_channels=channels,
                    dim=kwargs['dim'],
                    num_res_blocks=num_res_blocks,
                    attention_levels=attention_levels,
                    dropout=0,
                    ch_mult=ch_mult,
                    latent_dim = kwargs['latent_dim'],
                    noise_scale= kwargs['noise_scale'],
                    var_timestep=True if kwargs['prediction'] == 'vxt' else False,
                    vae_loc = kwargs['vae_loc'],
                    vae_inject = kwargs['vae_inject'],
                    xt_dropout = kwargs['xt_dropout'])

else:

    unet = RisannenUnet(image_size=kwargs['image_size'],
                        in_channels=channels,
                        dim=kwargs['dim'],
                        num_res_blocks=num_res_blocks,
                        attention_levels=attention_levels,
                        dropout=0.1,
                        ch_mult=ch_mult,
                        var_timestep=kwargs['var_timestep'])



# Define Trainer and Sampler
trainer = Trainer(model = unet, **kwargs)
sampler = Sampler(**kwargs)

# Fit GMM for cold sampling in deblurring diffusion
if kwargs['degradation'] == 'blur':
    sampler.fit_gmm(trainloader, clusters=1)

# Fix x_T for sampling
if kwargs['fix_sample']:
    sampler.sample_x_T(kwargs['n_samples'], channels, kwargs['image_size'])

# Create directories
imgpath, modelpath = create_dirs(**kwargs)

# imgpath = imgpath.replace('imgs', 'imgs/control').split('run')[0]
# imgpath = os.path.join(imgpath, 'control')
# if not os.path.exists(imgpath):
#     os.makedirs(imgpath)

ema_flag = '' if kwargs['skip_ema'] else '_ema'

# Load Checkpoint
try:
    chkpt = torch.load(os.path.join("./models/models_to_test/", f"chpkt_{kwargs['dim']}_{kwargs['timesteps']}_{kwargs['prediction']}{ema_flag}.pt"), map_location=kwargs['device'])
    trainer.model.load_state_dict(chkpt['model_state_dict'])
    trainer.optimizer.load_state_dict(chkpt['optimizer_state_dict'])
    trainer.model_ema.load_state_dict(chkpt['ema_state_dict'])
        
    # Replace model params with EMA params 
    trainer.model_ema.copy_to(trainer.model.parameters()) # Copy EMA params to model
    
    print("Checkpoint loaded, model trained until epoch", chkpt['epoch'])
except Exception as e:
    raise ValueError("No checkpoint found, please choose pretrained variable timestep model to control VAE injections.")

# EMA Transfer
trainer.model.eval()
trainer.model_ema.copy_to(trainer.model.parameters()) # Copy EMA params to model for inference
print("EMA transferred to model")




#%%

class ModelInteract():

    def __init__(self, model, timesteps):
        self.model = model
        self.t = torch.full((kwargs['n_samples'],), timesteps, dtype=torch.long).to(kwargs['device'])
        self.xt = torch.zeros((kwargs['n_samples'], channels, kwargs['image_size'], kwargs['image_size']), device=kwargs['device']) 

    @torch.no_grad()
    def predict(self, t_diff):
        t = self.t[0].item()
        t2 = int(t - t_diff)
        print(t2)
        self.t2 = torch.full((kwargs['n_samples'],), t2, dtype=torch.long).to(kwargs['device'])
        self.pred = self.model(self.xt, self.t, cond=None, t2=self.t2)
        self.plot_pred()
        
    def update_xt(self, b):
        self.xt = (self.xt + self.pred) if kwargs['add_noise'] or kwargs['vae'] else pred
        self.t = self.t2
    
    def plot_pred(self):
        grid = make_grid(self.pred[:10], nrow=10, padding=0)
        plt.imshow(grid.cpu().detach().numpy().transpose(1, 2, 0))
        plt.axis('off')
        plt.suptitle(f'\nVariable Step Reconstruction', fontsize=12)
        plt.show()
        



#%%


import ipywidgets as widgets
from IPython.display import display


#samples = torch.cat((samples, xt[0:num_samples]), dim=0)

interactor = ModelInteract(trainer.model, kwargs['timesteps'])

# Create the button widget
update_button = widgets.Button(description='Update xt')
update_button.on_click(interactor.update_xt)

# Create the slider widget
tdiff_slider = widgets.IntSlider(value=0, min=-1, max=kwargs['timesteps'], step=1, description='t difference')

# Create the interactive window
interactive_window = widgets.interactive(interactor.predict, t_diff=tdiff_slider)

# Display the interactive window
display(interactive_window, update_button)






#%%

# Sample t and define the difference
t = torch.ones((kwargs['n_samples'],), dtype=torch.long).to(kwargs['device']) * kwargs['timesteps'] # Start from T
t2 = torch.ones((kwargs['n_samples'],), dtype=torch.long).to(kwargs['device'])
for i, j in enumerate(reversed(range(kwargs['n_samples']))):
    t2[j] = i

trainer.model.eval()
xT = sampler.x_T
xt = xT

sampling_noise = None

pred = trainer.model(xt, t, cond=None, t2=t2)

# OURS with xt prediction
xtm1 = (xt + pred) if kwargs['add_noise'] or kwargs['vae'] else pred # According to Risannen, the model predicts the residual, which stabilizes the training

# In Risannen the noise is added to the predicted image, AFTER the model prediction
if kwargs['add_noise'] and not t == 0:
    sampling_noise = torch.randn_like(xt, device=kwargs['device']) * kwargs['noise_scale'] * 1.25 # 1.25 is a scaling factor from the original Risannen Code (delta = 1.25 * sigma)
    xtm1 = xtm1 + sampling_noise   
            
# Change from xtm1 to xt for next iteration
xt = xtm1

# Plot
grid = make_grid(xt, nrow=10, padding=0)
plt.imshow(grid.cpu().detach().numpy().transpose(1, 2, 0))
#plt.imshow(xt[0].cpu().detach().numpy().transpose(1, 2, 0))
plt.axis('off')
plt.suptitle(f'\nVariable Step Reconstruction with from xT up to x0', fontsize=12)
plt.show()


#%%







#%%


#%%

diff1 = 20
diff2 = 2

num_samples = 10
xT = sampler.x_T
samples = torch.Tensor(xT[0:num_samples])
xt = xT

skip = 50

# Detach the prediction from the graph
trainer.model.eval()
for t in tqdm(reversed(range(0, skip+1, diff1)), desc=f"Cold Sampling"):
    
    t_tensor = torch.full((kwargs['n_samples'],), t, dtype=torch.long).to(kwargs['device']) # t-1 to account for 0 indexing that the model is seeing during training
    t2 = torch.full((kwargs['n_samples'],), t-diff1, dtype=torch.long).to(kwargs['device']) # t-1 to account for 0 indexing that the model is seeing during training
    
    if t2[0].item() < 0: 
        t2 = torch.zeros((kwargs['n_samples'],), dtype=torch.long).to(kwargs['device']) - 1

    pred = trainer.model(xt, t_tensor, cond=None, t2=t2)
    pred = pred.detach()

    # OURS with xt prediction
    xtm1 = (xt + pred) if kwargs['add_noise'] or kwargs['vae'] else pred # According to Risannen, the model predicts the residual, which stabilizes the training

    # In Risannen the noise is added to the predicted image, AFTER the model prediction
    if kwargs['add_noise'] and not t == 0:
        sampling_noise = torch.randn_like(xt, device=kwargs['device']) * kwargs['noise_scale'] * 1.25 # 1.25 is a scaling factor from the original Risannen Code (delta = 1.25 * sigma)
        xtm1 = xtm1 + sampling_noise   
                
    # Change from xtm1 to xt for next iteration
    xt = xtm1
    samples = torch.cat((samples, xt[0:num_samples]), dim=0)

    if t2[0].item() < 0: 
        break


# Detach the prediction from the graph
trainer.model.eval()
for t in tqdm(reversed(range(skip, kwargs['timesteps']+1, diff2)), desc=f"Cold Sampling"):
    
    t_tensor = torch.full((kwargs['n_samples'],), t, dtype=torch.long).to(kwargs['device']) # t-1 to account for 0 indexing that the model is seeing during training
    t2 = torch.full((kwargs['n_samples'],), t-diff2, dtype=torch.long).to(kwargs['device']) # t-1 to account for 0 indexing that the model is seeing during training
    
    if t2[0].item() < 0: 
        t2 = torch.zeros((kwargs['n_samples'],), dtype=torch.long).to(kwargs['device']) - 1

    pred = trainer.model(xt, t_tensor, cond=None, t2=t2)
    pred = pred.detach()

    # OURS with xt prediction
    xtm1 = (xt + pred) if kwargs['add_noise'] or kwargs['vae'] else pred # According to Risannen, the model predicts the residual, which stabilizes the training

    # In Risannen the noise is added to the predicted image, AFTER the model prediction
    if kwargs['add_noise'] and not t == 0:
        sampling_noise = torch.randn_like(xt, device=kwargs['device']) * kwargs['noise_scale'] * 1.25 # 1.25 is a scaling factor from the original Risannen Code (delta = 1.25 * sigma)
        xtm1 = xtm1 + sampling_noise   
                
    # Change from xtm1 to xt for next iteration
    xt = xtm1
    samples = torch.cat((samples, xt[0:num_samples]), dim=0)

    if t2[0].item() < 0: 
        break


# Plot
# Permute such that the first, sixth, eleventh, ... image is shown in the first row
grid = make_grid(samples, nrow = 10, padding=0)
plt.imshow(grid.cpu().detach().numpy().transpose(1, 2, 0))
plt.axis('off')
plt.suptitle(f'\nVariable Step Reconstruction with diff1 = {diff1} and diff2 = {diff2}', fontsize=12)
plt.show()

# #%%
#     # Sampling from pretrained Model
        
#     # Train
#     trainer.model.eval()

#     # Sample
#     nrow = 6

#     flag = f'_mean_{kwargs["prior_mean"]}_std_{kwargs["prior_std"]}'
#     if kwargs['degradation'] == 'noise': # Noise Sampling
#         samples, xt = sampler.sample(trainer.model, kwargs['batch_size'], prior=prior)
        
#         save_image(samples[-1], os.path.join(imgpath, f'sample_{flag}.png'), nrow=nrow) #int(math.sqrt(kwargs['n_samples']))
#         save_video(samples, imgpath, nrow, f'sample_{flag}.mp4')
    
#     else: # Cold Sampling

#         if kwargs['plot_imgs']:

#             if kwargs['latent_walk']:

#                 if kwargs['dimension_walk']:
                    
#                     plt.figure(figsize=(20,20))
#                     imgs = []
#                     for dimension in tqdm(range(res*res)):
                        
#                         # Enhance one dimension of the prior for each timestep
#                         new_prior = standard_prior.flatten(1)
#                         new_prior[:, dimension] = new_prior[:, dimension] + kwargs['prior_mean']
#                         new_prior = new_prior.reshape(kwargs['batch_size'], res, res)

#                         gen_samples, _, _, manip_img = sampler.sample(model = trainer.model, 
#                                                                         batch_size = 1, 
#                                                                         generate=True, 
#                                                                         prior=new_prior)
#                         #plt.subplot(res, res, dimension+1)
#                         #plt.imshow(manip_img[0].cpu().detach().numpy().transpose(1, 2, 0).squeeze(), cmap='gray')
#                         #plt.axis('off')
#                         imgs.append(manip_img[0])
                    
#                     grid = make_grid(imgs, nrow=res, padding=0)
#                     plt.imshow(grid.cpu().detach().numpy().transpose(1, 2, 0))
#                     plt.axis('off')
#                     plt.suptitle(f'\nLatent Dimension Walk', fontsize=20)
#                     plt.show()
                
#                 else:
#                     plt.figure(figsize=(20,20))
#                     sqrt = int(math.sqrt(kwargs['timesteps']))
#                     rest = math.ceil((kwargs['timesteps'] - sqrt**2) / sqrt)
#                     for timestep in tqdm(range(kwargs['timesteps'])):
#                         gen_samples, _, _, manip_img = sampler.sample(model = trainer.model, 
#                                                                     batch_size = 1, 
#                                                                     generate=True, 
#                                                                     prior=manipulated_prior, 
#                                                                     t_inject=timestep)

#                         imgs.append(manip_img[0])
                    
#                     grid = make_grid(imgs, nrow=sqrt, padding=0)
#                     plt.imshow(grid.cpu().detach().numpy().transpose(1, 2, 0))
#                     plt.axis('off')
#                     plt.suptitle(f'\n Latent Timestep Walk', fontsize=16)
#                     plt.show()


#             else:

#                 og_img = next(iter(trainloader))[0][:kwargs['batch_size']].to(kwargs['device'])
#                 #_, xt, direct_recons, all_images = sampler.sample(model = trainer.model, img = og_img, batch_size = kwargs['n_samples'], generate = False)
#                 _, _, direct_recons, standard_imgs = sampler.sample(model = trainer.model, 
#                                                                     x0 = og_img, 
#                                                                     batch_size = kwargs['batch_size'], 
#                                                                     generate=True, 
#                                                                     prior=standard_prior)
                
#                 gen_samples, _, _, manip_imgs = sampler.sample(model = trainer.model, 
#                                                                x0 = og_img, 
#                                                                batch_size = kwargs['batch_size'], 
#                                                                generate=True, 
#                                                                prior=manipulated_prior)

#                 plt.figure(figsize=(10,8))
#                 plt.subplot(1, 2, 1)
#                 grid = make_grid(standard_imgs, nrow=nrow, padding=0)
#                 plt.imshow(grid.cpu().detach().numpy().transpose(1, 2, 0))
#                 plt.axis('off')
#                 plt.title('Standard Normal Prior', fontsize=14)

#                 plt.subplot(1, 2, 2)
#                 grid = make_grid(manip_imgs, nrow=nrow, padding=0)
#                 plt.imshow(grid.cpu().detach().numpy().transpose(1, 2, 0))
#                 plt.axis('off')
#                 plt.title(f"Prior ~ N({kwargs['prior_mean']},{kwargs['prior_std']})", fontsize=14)

#                 plt.suptitle(f'\n Controlled Sampling', fontsize=16)
                

#         if kwargs['save_imgs']:

#             # Training Process conditional generation
#             save_image(og_img, os.path.join(imgpath, f'orig_{flag}.png'), nrow=nrow)
#             save_image(standard_imgs, os.path.join(imgpath, f'normal_sample_{flag}.png'), nrow=nrow)
#             save_image(direct_recons, os.path.join(imgpath, f'direct_recon_{flag}.png'), nrow=nrow)

#             # Training Process unconditional generation
#             save_image(manip_imgs, os.path.join(imgpath, f'manip_sample_regular_{flag}.png'), nrow=nrow)
#             save_video(gen_samples, imgpath, nrow, f'sample_{flag}.mp4')


#         # save_gif(samples, imgpath, nrow, f'sample_{e}.gif')







    


# To Do Today:

# Debug Blurring Diffusion by comparing training and sampling 1:1 with Bansal et al.




### Difference: ema.module vs. unet - trainer.model vs. unet - currently investigated: No!
    # Small Batch Size
    # Only 25 timesteps - currently investigated: No!
    # 
# %%
