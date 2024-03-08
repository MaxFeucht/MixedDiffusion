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


def main(**kwargs):
    
    trainloader, valloader = load_data(kwargs['batch_size'])
    
    if kwargs['verbose']:
        plot_degradation(10, trainloader)
    
    x, _ = next(iter(trainloader))   
    channels, imsize = x[0].shape[0], x[0].shape[-1]
    
    # Define Model
    unet = UNet(image_size=imsize, channels=channels, num_downsamples=kwargs['num_downsamples'], dim = kwargs['dim'], dim_max =  kwargs['dim']*2**kwargs['num_downsamples'])

    # Define Trainer and Sampler
    trainer = Trainer(model = unet, **kwargs)
    sampler = Sampler(**kwargs)

    # Check if directory for imgs exists
    for i in range(10000):
        path = f'./imgs/{kwargs["dataset"]}_{kwargs["degradation"]}/run_{i}/'
        if not os.path.exists(path):
            os.makedirs(path)
            break

    # Training Loop
    for e in range(kwargs['epochs']):
        
        val_flag = True if e % 10 == 0 and e != 0 else False
        trainloss, valloss = trainer.train_epoch(trainloader, valloader, val=val_flag)
        
        print(f"Epoch {e} Train Loss: {trainloss}")
        if val_flag:
            print(f"Epoch {e} Validation Loss: {valloss}")
        
            # Save 10 images generated from the model at the end of each epoch
            samples = sampler.sample_ddpm(unet, 10)
            
            # Save all 10 images in a folder
            for i, img in enumerate(samples):
                plt.imsave(path + f'epoch_{e+1}_img_{i}.png', img.squeeze().detach().cpu().numpy())

            # Save model
            torch.save(trainer.model.state_dict(), f'./unet_{kwargs["dataset"]}_{kwargs["degradation"]}_{kwargs["dim"]}_{kwargs["epochs"]}.pt')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Diffusion Models')
    parser.add_argument('--timesteps', '--t', type=int, default=100, help='Degradation timesteps')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', '--e', type=int, default=10, help='Number of Training Epochs')
    parser.add_argument('--batch_size', '--b', type=int, default=32, help='Batch size')
    parser.add_argument('--dim', '--d', type=int, default=32, help='Model dimension')
    parser.add_argument('--num_downsamples', '--down', type=int, default=2, help='Number of downsamples')
    parser.add_argument('--prediction', '--pred', type=str, default='residual', help='Prediction method')
    parser.add_argument('--degradation', '--deg', type=str, default='blur', help='Degradation method')
    parser.add_argument('--noise_schedule', '--sched', type=str, default='cosine', help='Noise schedule')
    parser.add_argument('--dataset', type=str, default='mnist', help='Dataset')
    parser.add_argument('--verbose', '--v', action='store_true', help='Verbose mode')

    args = parser.parse_args()
    
    args.num_downsamples = 2 if args.dataset == 'mnist' else 3
    args.device = 'cuda' if torch.cuda.is_available() else 'mps'
    
    main(**vars(args))
    



    

