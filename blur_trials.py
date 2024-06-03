import argparse
import numpy as np
import os
import sys
import time
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import wandb

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets
from torchvision import transforms as T
from torchvision.utils import save_image

from scripts.datasets import load_data

from unet import UNet
from mnist_unet import MNISTUnet
from scripts.bansal_unet import BansalUnet
from scripts.risannen_unet import RisannenUnet
from scripts.risannen_unet_vae import VAEUnet
#from scripts.vae_unet_full import VAEUnet

from diffusion_utils import Degradation, Trainer, Sampler, ExponentialMovingAverage
from utils import create_dirs, save_video, save_gif, MyCelebA

import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

# Check if ipykernel is running to check if we're working locally or on the cluster
import sys
if 'ipykernel' in sys.modules:
    sys.argv = ['']


parser = argparse.ArgumentParser(description='Diffusion Models')

# General Diffusion Parameters
parser.add_argument('--timesteps', '--t', type=int, default=100, help='Degradation timesteps')
parser.add_argument('--prediction', '--pred', type=str, default='vxt', help='Prediction method, choose one of [x0, xt, residual]')
parser.add_argument('--dataset', type=str, default='mnist', help='Dataset to run Diffusion on. Choose one of [mnist, cifar10, celeba, lsun_churches]')
parser.add_argument('--degradation', '--deg', type=str, default='fadeblack_blur', help='Degradation method')
parser.add_argument('--batch_size', '--b', type=int, default=5, help='Batch size')
parser.add_argument('--dim', '--d', type=int , default=32, help='Model dimension')
parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
parser.add_argument('--epochs', '--e', type=int, default=20, help='Number of Training Epochs')
parser.add_argument('--noise_schedule', '--sched', type=str, default='cosine', help='Noise schedule')
parser.add_argument('--xt_weighting', action='store_true', help='Whether to use weighting for xt in loss')
parser.add_argument('--var_sampling_step', type=int, default = 1, help='How to sample var timestep model - int > 0 indicates t difference to predict, -1 indicates x0 prediction')
parser.add_argument('--baseline', '--base', type=str, default='xxx', help='Whether to run a baseline model - Risannen, Bansal, VAE')

# Noise Injection Parameters
parser.add_argument('--vae', action='store_false', help='Whether to use VAE Noise injections')
parser.add_argument('--vae_alpha', type=float, default = 0.999, help='Trade-off parameter for weight of Reconstruction and KL Div')
parser.add_argument('--latent_dim', type=int, default=32, help='Which dimension the VAE latent space is supposed to have')
parser.add_argument('--add_noise', action='store_true', help='Whether to add noise Risannen et al. style')
parser.add_argument('--break_symmetry', action='store_true', help='Whether to add noise to xT Bansal et al. style')
parser.add_argument('--noise_scale', type=float, default = 0.01, help='How much Noise to add to the input')
parser.add_argument('--vae_loc', type=str, default = 'start', help='Where to inject VAE Noise. One of [start, bottleneck, emb].')
parser.add_argument('--vae_inject', type=str, default = 'add', help='How to inject VAE Noise. One of [concat, add].')
parser.add_argument('--xt_dropout', type=float, default = 0.3, help='How much of xt is dropped out at every step (to foster reliance on VAE injections)')

# Housekeeping Parameters
parser.add_argument('--load_checkpoint', action='store_false', help='Whether to try to load a checkpoint')
parser.add_argument('--sample_interval', type=int, help='After how many epochs to sample', default=1)
parser.add_argument('--n_samples', type=int, default=60, help='Number of samples to generate')
parser.add_argument('--fix_sample', action='store_false', help='Whether to fix x_T for sampling, to see sample progression')
parser.add_argument('--skip_ema', action='store_true', help='Whether to skip model EMA')
parser.add_argument('--model_ema_decay', type=float, default=0.997, help='Model EMA decay')
parser.add_argument('--cluster', action='store_true', help='Whether to run script locally')
parser.add_argument('--verbose', '--v', action='store_true', help='Verbose mode')

parser.add_argument('--test_run', action='store_true', help='Whether to test run the pipeline')

args = parser.parse_args()

args.num_downsamples = 2 if args.dataset == 'mnist' else 3
args.device = 'cuda' if torch.cuda.is_available() else 'mps'

if args.dataset == 'mnist':
    args.image_size = 28
elif args.dataset == 'cifar10':
    args.image_size = 32
elif args.dataset == 'afhq':
    args.image_size = 64

kwargs = vars(args)

def load_dataset(batch_size = 32, dataset = 'mnist'):
    
    assert dataset in ['mnist', 'cifar10', 'celeba', 'lsun_churches', 'afhq'],f"Invalid dataset, choose from ['mnist', 'cifar10', 'celeba', 'lsun_churches', 'afhq']"

    # Check if directory exists
    if not os.path.exists(f'./data/{dataset.split("_")[0].upper()}'):
        os.makedirs(f'./data/{dataset.split("_")[0].upper()}')

    
    if dataset == 'mnist':

        training_data = datasets.MNIST(root='./data/MNIST', 
                                    train=True, 
                                    download=True, 
                                    transform=T.Compose([T.ToTensor()]))
        val_data = datasets.MNIST(root='./data/MNIST', 
                                    train=False, 
                                    download=True, 
                                    transform=T.Compose([T.ToTensor()]))
    
    elif dataset == 'cifar10':

        training_data = datasets.CIFAR10(root='./data/CIFAR10', 
                                    train=True, 
                                    download=True, 
                                    transform=T.Compose([T.ToTensor()]))
        val_data = datasets.CIFAR10(root='./data/CIFAR10', 
                                    train=False, 
                                    download=True, 
                                    transform=T.Compose([T.ToTensor()]))
    
    elif dataset == 'celeba':
        
        train_transformation = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor()])
        
        scriptdir = os.path.dirname(__file__)
        datadir = os.path.join(scriptdir,'data')

        # Adapt path to data directory for DAS-6
        if 'scratch' in datadir:
            datadir = datadir.replace('MixedDiffusion/', '')

        print("Data Directory: ", datadir)

        training_data = MyCelebA(
            datadir,
            split='train',
            transform=train_transformation,
            download=False,
        )
        
        # Replace CelebA with your dataset
        val_data = MyCelebA(
            datadir,
            split='test',
            transform=train_transformation,
            download=False,
        )
    

    elif dataset == 'lsun_churches':
        scriptdir = os.path.dirname(__file__)
        datadir = os.path.join(scriptdir,'data/LSUN_CHURCHES')
        training_data = datasets.LSUN(root=datadir,
                                    classes=['church_outdoor_train'], 
                                    transform=T.Compose([T.ToTensor()]))
        val_data = datasets.LSUN(root=datadir,
                                    classes=['church_outdoor_val'], 
                                    transform=T.Compose([T.ToTensor()]))
    
    elif dataset == 'afhq':
        train_loader = load_data(data_dir="./data/AFHQ_64/train",
                                batch_size=batch_size, image_size=64,
                                random_flip=False, num_workers=0)
        val_loader = load_data(data_dir="./data/AFHQ_64/test",
                               batch_size=batch_size, image_size=64,
                               random_flip=False, num_workers=0)
        
        return train_loader, val_loader


    # Set up data loaders
    train_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader


def plot_degradation(train_loader, **kwargs):

    timesteps = kwargs.pop('timesteps')
    kwargs.pop('degradation')
    noise = Degradation(timesteps = timesteps, degradation = 'noise', **kwargs)
    blur = Degradation(timesteps = timesteps, degradation = 'blur', **kwargs)
    black = Degradation(timesteps = timesteps, degradation = 'fadeblack', **kwargs)
    black_blur = Degradation(timesteps = timesteps, degradation = 'fadeblack_blur', **kwargs)
    blur_ban = Degradation(timesteps = timesteps, degradation = 'blur_bansal', **kwargs)
    black_blur_ban = Degradation(timesteps = timesteps, degradation = 'fadeblack_blur_bansal', **kwargs)

    #timesteps = min(50, timesteps)

    plt.figure(figsize=(16, 10))
    for i, j in enumerate(range(1, timesteps + 1, 10)):

        ind = i + 1
        x, y = next(iter(train_loader)) 
        t = torch.tensor([j]).repeat(x.shape[0],).to('mps')
        x = x.to('mps')

        plt.subplot(7, timesteps//10, 0*timesteps//10+ind)
        x_plain = x[0].unsqueeze(0) if len(x[0].shape) == 2 else x
        plt.imshow(x_plain[0].cpu().permute(1, 2, 0))
        plt.axis('off')

        plt.subplot(7, timesteps//10, 1*timesteps//10+ind)
        x_noise = noise.degrade(x, t).cpu()
        x_noise = x_noise[0].unsqueeze(0) if len(x_noise[0].shape) == 2 else x_noise[0]
        plt.imshow(x_noise.permute(1, 2, 0), vmin=0, vmax=1)   
        plt.axis('off')
    
        plt.subplot(7, timesteps//10, 2*timesteps//10+ind)
        x_blur = blur.degrade(x, t).cpu()
        x_blur = x_blur[0].unsqueeze(0) if len(x_blur[0].shape) == 2 else x_blur[0]
        plt.imshow(x_blur.permute(1, 2, 0), vmin=0, vmax=1)
        plt.axis('off')
        
        plt.subplot(7, timesteps//10, 3*timesteps//10+ind)
        x_black = black.degrade(x, t).cpu()
        x_black = x_black[0].unsqueeze(0) if len(x_black[0].shape) == 2 else x_black[0]
        plt.imshow(x_black.permute(1, 2, 0), vmin=0, vmax=1)   
        plt.axis('off')
    
        plt.subplot(7, timesteps//10, 4*timesteps//10+ind)
        x_blackblur = black_blur.degrade(x, t).cpu()
        x_blackblur = x_blackblur[0].unsqueeze(0) if len(x_blackblur[0].shape) == 2 else x_blackblur[0]
        plt.imshow(x_blackblur.permute(1, 2, 0), vmin=0, vmax=1)   
        plt.axis('off')

        plt.subplot(7, timesteps//10, 5*timesteps//10+ind)
        x_blur_ban = blur_ban.degrade(x, t).cpu()
        x_blur_ban = x_blur_ban[0].unsqueeze(0) if len(x_blur_ban[0].shape) == 2 else x_blur_ban[0]
        plt.imshow(x_blur_ban.permute(1, 2, 0), vmin=0, vmax=1)   
        plt.axis('off')

        plt.subplot(7, timesteps//10, 6*timesteps//10+ind)
        x_blackblur_ban = black_blur_ban.degrade(x, t).cpu()
        x_blackblur_ban = x_blackblur_ban[0].unsqueeze(0) if len(x_blackblur_ban[0].shape) == 2 else x_blackblur_ban[0]
        plt.imshow(x_blackblur_ban.permute(1, 2, 0), vmin=0, vmax=1)   
        plt.axis('off')
        

    # axis off
    plt.suptitle('Image degradation', size = 18)


trainloader, valloader = load_dataset(kwargs['batch_size'], kwargs['dataset'])
plot_degradation(trainloader, **kwargs)





