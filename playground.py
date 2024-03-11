import argparse
import numpy as np
import os
import sys
import time
import math
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torchvision import datasets
from torchvision import transforms as T

from unet import UNet
from scripts.karras_unet import KarrasUnet
from diffusion_utils import Degradation, Scheduler, Reconstruction, Trainer, Sampler, Blurring, DenoisingCoefs
from utils import create_dir

import sys
sys.argv = ['']

def load_data(batch_size = 32):
    
    # Check if directory exists
    if not os.path.exists('./data/MNIST'):
        os.makedirs('./data/MNIST')
    
    # Set up training data
    training_data = datasets.MNIST(root='./data/MNIST', train=True, download=True, transform=T.Compose([
                                                                                                    T.ToTensor()
                                                                                                ]))
    # Set up validation data
    val_data = datasets.MNIST(root='./data/MNIST', train=False, download=True, transform=T.Compose([
                                                                                                    T.ToTensor()
                                                                                                ]))

    # Set up data loaders
    train_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader


def plot_degradation(timesteps, train_loader):
    
    noise = Degradation(timesteps = timesteps, degradation = 'noise', noise_schedule='cosine')
    blur = Degradation(timesteps = timesteps, degradation = 'blur')
    black = Degradation(timesteps = timesteps, degradation = 'fadeblack')
    black_blur = Degradation(timesteps = timesteps, degradation = 'fadeblack_blur')

    plt.figure(figsize=(16, 5))
    for i in range(timesteps):

        x, y = next(iter(train_loader))   
        x = x[0].squeeze().numpy()
        
        plt.subplot(5, timesteps, 0*timesteps+i+1)
        plt.imshow(x)
        plt.axis('off')

        plt.subplot(5, timesteps, 1*timesteps+i+1)
        plt.imshow(noise.degrade(x, i))
        plt.axis('off')
    
        plt.subplot(5, timesteps, 2*timesteps+i+1)
        plt.imshow(blur.degrade(x, i).cpu(), vmin=0, vmax=1)
        plt.axis('off')
        
        plt.subplot(5, timesteps, 3*timesteps+i+1)
        plt.imshow(black.degrade(x, i).cpu(), vmin=0, vmax=1)   
        plt.axis('off')
        
        plt.subplot(5, timesteps, 4*timesteps+i+1)
        plt.imshow(black_blur.degrade(x, i).cpu(), vmin=0, vmax=1)   
        plt.axis('off')

    # axis off
    plt.suptitle('Image degradation', size = 18)


#%%

default_args = {
    'timesteps': 1000,
    'lr': 1e-4,
    'epochs': 500,
    'batch_size': 32,
    'dim': 32,
    'num_downsamples': 2,
    'prediction': 'residual',
    'degradation': 'noise',
    'noise_schedule': 'cosine',
    'dataset': 'mnist',
    'verbose': False,
    'device': 'cuda' if torch.cuda.is_available() else 'mps'
}

trainloader, valloader = load_data(default_args['batch_size'])

if default_args['verbose']:
    plot_degradation(10, trainloader)

x, _ = next(iter(trainloader))   
channels, imsize = x[0].shape[0], x[0].shape[-1]

# Define Model
unet = UNet(image_size=imsize, channels=channels, num_downsamples=default_args['num_downsamples'], dim=default_args['dim'], dim_max=default_args['dim']*2**default_args['num_downsamples'])

# Define Trainer and Sampler
trainer = Trainer(model = unet, **default_args)
sampler = Sampler(**default_args)

# Training Loop
for e in range(default_args['epochs']):
    
    val_flag = True if (e+1) % 10 == 0 else False
    trainloss, valloss = trainer.train_epoch(trainloader, valloader, val=val_flag)
    
    print(f"Epoch {e} Train Loss: {trainloss}")
    if val_flag:
        
        # Create directory for images
        if e < 10:
            path = create_dir(**default_args)

        print(f"Epoch {e} Validation Loss: {valloss}")
    
        # Save 10 images generated from the model at the end of each epoch
        samples = sampler.sample_ddpm(unet, 10)
        
        # Save all 10 images in a folder
        for i, img in enumerate(samples):
            plt.imsave(path + f'epoch_{e+1}_img_{i}.png', img.squeeze().detach().cpu().numpy())

        # Save model
        torch.save(trainer.model.state_dict(), f'./models/mnist_noise/unet_{default_args["dataset"]}_{default_args["degradation"]}_{default_args["dim"]}_{default_args["epochs"]}.pt')


#%%
hfont = {'fontname':'AppleGothic'}

## Check whether image from 0 to 255 or 0 to 1
default_args = {
    'timesteps': 1000,
    'lr': 1e-4,
    'epochs': 500,
    'batch_size': 32,
    'dim': 32,
    'num_downsamples': 2,
    'prediction': 'residual',
    'degradation': 'fadeblack_blur',
    'noise_schedule': 'cosine',
    'dataset': 'mnist',
    'verbose': False,
    'device': 'cuda' if torch.cuda.is_available() else 'mps'
}

batch_size = 32
timesteps = 40
model = UNet(image_size=28, channels=1, num_downsamples=2, dim=32, dim_max=128)
model.to(default_args['device'])
degrader = Degradation(**default_args)

# Initialize an empty tensor to store the batch
x_t = torch.empty(batch_size, model.channels, model.image_size, model.image_size, device=default_args['device'])

# Fill each depth slice with a single integer drawn uniformly from [0, 255]
for i in range(batch_size):
    for j in range(model.channels):
        x_t[i, j, :, :] = torch.full((model.image_size, model.image_size), torch.rand(1).item(), dtype=torch.float32)

#degrader.degrade(x_t, 0)
        
x, _ = next(iter(trainloader))

plt.figure(figsize=(10, 7))
with torch.no_grad():
    for t in range(timesteps, 0, -1):
        t_tensor = torch.tensor([t]).repeat(x_t.shape[0]).float().to(default_args['device'])
        x_0_hat = model(x_t,t_tensor)
        x_tm1 = x_t -  degrader.degrade(x_0_hat, t) + degrader.degrade(x_0_hat, t-1)

        plt.subplot(5, timesteps//5, timesteps - t + 1)
        plt.imshow(x_t[0].T.squeeze().detach().cpu().numpy(), vmin=0, vmax=1)
        plt.title(f't = {t}', size = 8, **hfont) 
        plt.axis('off')

        x_t = x_tm1


plt.suptitle('Denoising Diffusion Process', size = 16, **hfont)
plt.show()

#%%
        
# Get batch from dataloader
x, _ = next(iter(trainloader))

x.shape
x_0_hat.shape

# %%
