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
from karras_unet import KarrasUnet
from diffusion_utils import Degradation, Scheduler, Reconstruction, Trainer, Sampler, Blurring, DenoisingCoefs



def load_data(batch_size = 32):
    
    # Set up training data
    training_data = datasets.MNIST(root='./data/MNIST/train', train=True, download=False, transform=T.Compose([
                                                                                                    T.ToTensor()
                                                                                                ]))
    # Set up validation data
    val_data = datasets.MNIST(root='./data/MNIST/test', train=False, download=False, transform=T.Compose([
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
        plt.imshow(blur.degrade(x, i), vmin=0, vmax=1)
        plt.axis('off')
        
        plt.subplot(5, timesteps, 3*timesteps+i+1)
        plt.imshow(black.degrade(x, i), vmin=0, vmax=1)   
        plt.axis('off')
        
        plt.subplot(5, timesteps, 4*timesteps+i+1)
        plt.imshow(black_blur.degrade(x, i), vmin=0, vmax=1)   
        plt.axis('off')

    # axis off
    plt.suptitle('Image degradation', size = 18)


#%%

# Next stage: Training a model
unet = UNet(image_size=28, channels=1, num_downsamples=2, dim = 64, dim_max = 256)
#unet = KarrasUnet(image_size=28, channels=1)
timesteps = 10
lr = 1e-4
prediction = 'residual'
degradation = 'noise'
noise_schedule = 'cosine'
dataset = 'mnist'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

general_kwargs = {'model': unet, 'lr': lr, 'timesteps': timesteps, 'prediction': prediction, 'degradation': degradation, 'noise_schedule': noise_schedule, 'dataset': dataset, 'device': device}

trainer = Trainer()

general_kwargs['timesteps']

# Training Loop
loss_avg = 0
for i, train_data in tqdm(enumerate(train_loader)):
    x_0 = train_data[0].to(device)
    loss_avg += trainer.train_iter(x_0.float())
    if i % 10 == 0:
        print("Loss:", loss_avg/10)
        loss_avg = 0
        


#%%
# Training Loop
timesteps = 100
degrader = Degradation(timesteps = timesteps, degradation = 'noise', noise_schedule='cosine')
coefs = DenoisingCoefs(timesteps = timesteps, degradation = 'noise', noise_schedule='cosine')
unet = UNet(image_size=28, channels=1, num_downsamples=2, dim = 32, dim_max = 256)
optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4)
train_loader, val_loader = load_data()

for i, train_data in tqdm(enumerate(train_loader), total=len(train_loader)):
    x_0 = train_data[0].to(device)
    t = torch.randint(0, timesteps, (1,), dtype=torch.long) # Randomly sample a time step
    x_t = degrader.degrade(x_0, t)
    residual = torch.randn_like(x_0) # Residual either difference between x_0 and x_t for deterministic degradation or random noise for denoising diffusion
    t_tensor = t.repeat(x_t.shape[0]).float() # Repeat time step tensor to match the batch size
    pred = unet(x_t, t_tensor)
    xt_coef, residual_coef = coefs.x0_restore(t)
    x0_estimate = xt_coef * x_t - residual_coef * pred 
    target = residual
    loss = torch.nn.functional.mse_loss(x0_estimate, target)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if i % 10 == 0:
        print("Loss:", loss.item())

plt.imshow(pred[5].squeeze().detach().numpy(), cmap='gray')

# %%

sampler = Sampler(timesteps = timesteps, degradation = 'noise', noise_schedule='cosine')
sampler.sample_ddpm(unet, 10, return_trajectory=True)


    
# To Do:

# Make sure Sampler works

# %%
