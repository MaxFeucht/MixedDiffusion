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
from torchvision.utils import save_image, make_grid

from scripts.datasets import load_data

from unet import UNet
from mnist_unet import MNISTUnet
from scripts.bansal_unet import BansalUnet
from scripts.risannen_unet import RisannenUnet
from scripts.risannen_unet_vae import VAEUnet
#from scripts.vae_unet_full import VAEUnet

from diffusion_utils import Degradation, Trainer, Sampler, ExponentialMovingAverage
from utils import create_dirs, save_video, save_gif, MyCelebA

# Check if ipykernel is running to check if we're working locally or on the cluster
import sys
if 'ipykernel' in sys.modules:
    sys.argv = ['']


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
                                random_flip=False)
        val_loader = load_data(data_dir="./data/AFHQ_64/test",
                               batch_size=batch_size, image_size=64,
                               random_flip=False)
        
        return train_loader, val_loader


    # Set up data loaders
    train_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader


def plot_degradation(timesteps, train_loader, **kwargs):

    kwargs.pop('degradation')
    noise = Degradation(timesteps = timesteps, degradation = 'noise', **kwargs)
    blur = Degradation(timesteps = timesteps, degradation = 'blur', **kwargs)
    black = Degradation(timesteps = timesteps, degradation = 'fadeblack', **kwargs)
    black_blur = Degradation(timesteps = timesteps, degradation = 'fadeblack_blur', **kwargs)

    timesteps = min(50, timesteps)

    plt.figure(figsize=(16, 5))
    for i in range(timesteps):

        ind = i
        x, y = next(iter(train_loader)) 
        t = torch.tensor([i]).repeat(x.shape[0],).to('mps')
        x = x.to('mps')

        plt.subplot(5, timesteps, 0*timesteps+ind+1)
        x_plain = x[0].unsqueeze(0) if len(x[0].shape) == 2 else x
        plt.imshow(x_plain[0].cpu().permute(1, 2, 0))
        plt.axis('off')

        plt.subplot(5, timesteps, 1*timesteps+ind+1)
        x_noise = noise.degrade(x, t).cpu()
        x_noise = x_noise[0].unsqueeze(0) if len(x_noise[0].shape) == 2 else x_noise[0]
        plt.imshow(x_noise.permute(1, 2, 0), vmin=0, vmax=1)   
        plt.axis('off')
    
        plt.subplot(5, timesteps, 2*timesteps+ind+1)
        x_blur = blur.degrade(x, t).cpu()
        x_blur = x_blur[0].unsqueeze(0) if len(x_blur[0].shape) == 2 else x_blur[0]
        plt.imshow(x_blur.permute(1, 2, 0), vmin=0, vmax=1)
        plt.axis('off')
        
        plt.subplot(5, timesteps, 3*timesteps+ind+1)
        x_black = black.degrade(x, t).cpu()
        x_black = x_black[0].unsqueeze(0) if len(x_black[0].shape) == 2 else x_black[0]
        plt.imshow(x_black.permute(1, 2, 0), vmin=0, vmax=1)   
        plt.axis('off')
    
        plt.subplot(5, timesteps, 4*timesteps+ind+1)
        x_blackblur = black_blur.degrade(x, t).cpu()
        x_blackblur = x_blackblur[0].unsqueeze(0) if len(x_blackblur[0].shape) == 2 else x_blackblur[0]
        plt.imshow(x_blackblur.permute(1, 2, 0), vmin=0, vmax=1)   
        plt.axis('off')
        

    # axis off
    plt.suptitle('Image degradation', size = 18)


def plot_grid(samples, timesteps):
    grid = make_grid(samples[1:timesteps+1], nrow = 20, padding=0)
    plt.imshow(grid.cpu().detach().numpy().transpose(1, 2, 0))
    plt.axis('off')
    plt.suptitle(f'\nLatent Walk with Standard Normal', fontsize=12)
    plt.show()


def main(**kwargs):
    
    trainloader, valloader = load_dataset(kwargs['batch_size'], kwargs['dataset'])
    
    if kwargs['verbose']:
        plot_degradation(train_loader=trainloader, **kwargs)
    
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

        # Bansal Version
        # unet = VAEUnet(image_size=imsize,
        #                 channels=channels,
        #                 out_ch=channels,
        #                 ch=kwargs['dim'],
        #                 ch_mult= ch_mult,
        #                 num_res_blocks=num_res_blocks,
        #                 attn_resolutions=(14,) if kwargs['dataset'] == 'mnist' else (16,),
        #                 latent_dim=int(channels*imsize*imsize//kwargs['vae_downsample']),
        #                 noise_scale=kwargs['noise_scale'],
        #                 dropout=0)

        # Risannen Version
        unet = VAEUnet(image_size=kwargs["image_size"],
                        in_channels=channels,
                        dim=kwargs['dim'],
                        num_res_blocks=num_res_blocks,
                        attention_levels=attention_levels,
                        dropout=0.1,
                        ch_mult=ch_mult,
                        latent_dim=kwargs['latent_dim'],
                        noise_scale=kwargs['noise_scale'],
                        vae_inject = kwargs['vae_inject'],
                        xt_dropout = kwargs['xt_dropout'])

    else:
        # unet = BansalUnet(image_size=imsize,
        #         channels=channels,
        #         out_ch=channels,
        #         ch=kwargs['dim'],
        #         ch_mult= ch_mult,
        #         num_res_blocks=num_res_blocks,
        #         attn_resolutions=(14,) if kwargs['dataset'] == 'mnist' else (16,),
        #         dropout=0)
    
        unet = RisannenUnet(image_size=kwargs["image_size"],
                            in_channels=channels,
                            dim=kwargs['dim'],
                            num_res_blocks=num_res_blocks,
                            attention_levels=attention_levels,
                            dropout=0.1,
                            ch_mult=ch_mult)


    # # Enable Multi-GPU training
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     unet = nn.DataParallel(unet)

    # Define Trainer and Sampler
    trainer = Trainer(model = unet, **kwargs)
    sampler = Sampler(**kwargs)
    
    # Fit GMM for cold sampling in deblurring diffusion
    if kwargs['degradation'] == 'blur':
        sampler.fit_gmm(trainloader, clusters=1)

    # Fix x_T for sampling
    if kwargs['fix_sample']:
        sampler.sample_x_T(kwargs['n_samples'], channels, kwargs["image_size"])

    # Fix Prior for VAE
    prior = torch.randn((kwargs['n_samples'], kwargs['latent_dim'])).to(kwargs['device'])        
    prior[:, 0] = prior[:, 0] + 2

    # Create directories
    imgpath, modelpath = create_dirs(**kwargs)
    ema_flag = '' if kwargs['skip_ema'] else '_ema'

    # Load Checkpoint
    if kwargs['load_checkpoint']:
        try:
            chkpt = torch.load(os.path.join(modelpath, f"chpkt_{kwargs['dim']}_{kwargs['timesteps']}_{kwargs['prediction']}{ema_flag}.pt"), map_location=kwargs['device'])
            trainer.model.load_state_dict(chkpt['model_state_dict'])
            trainer.optimizer.load_state_dict(chkpt['optimizer_state_dict'])
            trainer.model_ema.load_state_dict(chkpt['ema_state_dict'])
            epoch_offset = chkpt['epoch']
            
            # Replace model params with EMA params 
            trainer.model_ema.copy_to(trainer.model.parameters()) # Copy EMA params to model
            
            print("Checkpoint loaded, continuing training from epoch", epoch_offset)

        except Exception as e:
            print("No checkpoint found: ", e)
            epoch_offset = 0
    else:
        epoch_offset = 0


    prior = torch.randn((kwargs['n_samples'], kwargs['latent_dim'])).to(kwargs['device'])        
    trainer.model.eval()

    t=kwargs['timesteps']

    xT = sampler.x_T
    xt = xT

    direct_recons = None
    sampling_noise = None

    global samples
    samples1 = torch.Tensor(xT[0].unsqueeze(0))
    samples2 = torch.Tensor(xT[1].unsqueeze(0))
    samples3 = torch.Tensor(xT[2].unsqueeze(0))
    samples4 = torch.Tensor(xT[3].unsqueeze(0))
    samples5 = torch.Tensor(xT[4].unsqueeze(0))

    for t in tqdm(reversed(range(kwargs['timesteps'])), desc=f"Cold Sampling"):
        t_tensor = torch.full((kwargs['n_samples'],), t, dtype=torch.long).to(kwargs['device']) # t-1 to account for 0 indexing that the model is seeing during training
                
        pred = trainer.model(xt, t_tensor, xtm1=None, prior=prior)
        pred = pred.detach()

        # BANSAL ALGORITHM 2
        x0_hat = pred
        xt_hat = sampler.degradation.degrade(x0_hat, t_tensor)
        xtm1_hat = sampler.degradation.degrade(x0_hat, t_tensor - 1) # This returns x0_hat for t=0
        xtm1 = xt - xt_hat + xtm1_hat

        if direct_recons == None:
            direct_recons = x0_hat
        
        xt = xtm1

        samples1 = torch.cat((samples1, xt[0].unsqueeze(0)), dim=0)
        samples2 = torch.cat((samples2, xt[1].unsqueeze(0)), dim=0)
        samples3 = torch.cat((samples3, xt[2].unsqueeze(0)), dim=0)
        samples4 = torch.cat((samples4, xt[3].unsqueeze(0)), dim=0)
        samples5 = torch.cat((samples5, xt[4].unsqueeze(0)), dim=0)

    
    # Plot
    # Permute such that the first, sixth, eleventh, ... image is shown in the first row
    plot_grid(samples1, kwargs['timesteps'])
    plot_grid(samples2, kwargs['timesteps'])
    plot_grid(samples3, kwargs['timesteps'])
    plot_grid(samples4, kwargs['timesteps'])
    plot_grid(samples5, kwargs['timesteps'])



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Diffusion Models')

    # General Diffusion Parameters
    parser.add_argument('--timesteps', '--t', type=int, default=200, help='Degradation timesteps')
    parser.add_argument('--prediction', '--pred', type=str, default='x0', help='Prediction method, choose one of [x0, xtm1, residual]')
    parser.add_argument('--dataset', type=str, default='afhq', help='Dataset to run Diffusion on. Choose one of [mnist, cifar10, celeba, lsun_churches]')
    parser.add_argument('--degradation', '--deg', type=str, default='fadeblack_blur', help='Degradation method')
    parser.add_argument('--batch_size', '--b', type=int, default=32, help='Batch size')
    parser.add_argument('--dim', '--d', type=int , default=128, help='Model dimension')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--epochs', '--e', type=int, default=50, help='Number of Training Epochs')
    parser.add_argument('--noise_schedule', '--sched', type=str, default='cosine', help='Noise schedule')
    parser.add_argument('--xt_weighting', action='store_true', help='Whether to use weighting for xt in loss')
    parser.add_argument('--var_timestep', action='store_true', help='Whether to use variable timestep diffusion')

    # Noise Injection Parameters
    parser.add_argument('--vae', action='store_false', help='Whether to use VAE Noise injections')
    parser.add_argument('--vae_alpha', type=float, default = 0.999, help='Trade-off parameter for weight of Reconstruction and KL Div')
    parser.add_argument('--latent_dim', type=int, default=10, help='Which dimension the VAE latent space is supposed to have')
    parser.add_argument('--add_noise', action='store_true', help='Whether to add noise Risannen et al. style')
    parser.add_argument('--break_symmetry', action='store_true', help='Whether to add noise to xT Bansal et al. style')
    parser.add_argument('--noise_scale', type=float, default = 0.01, help='How much Noise to add to the input')
    parser.add_argument('--vae_inject', type=str, default = 'emb', help='Where to inject VAE Noise. One of [start, bottleneck, emb].')
    parser.add_argument('--xt_dropout', type=float, default = 0, help='How much of xt is dropped out at every step (to foster reliance on VAE injections)')

    # Housekeeping Parameters
    parser.add_argument('--load_checkpoint', action='store_false', help='Whether to try to load a checkpoint')
    parser.add_argument('--sample_interval', type=int, help='After how many epochs to sample', default=1)
    parser.add_argument('--n_samples', type=int, default=5, help='Number of samples to generate')
    parser.add_argument('--fix_sample', action='store_false', help='Whether to fix x_T for sampling, to see sample progression')
    parser.add_argument('--skip_ema', action='store_true', help='Whether to skip model EMA')
    parser.add_argument('--model_ema_steps', type=int, default=10, help='Model EMA steps')
    parser.add_argument('--model_ema_decay', type=float, default=0.995, help='Model EMA decay')
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

    if args.var_timestep:
        var_string = "Running Variable Timestep Diffusion"
    else:
        var_string = "Running Sequential Diffusion"

    if args.vae:
        setup_string = "using VAE Noise Injections"
        assert not args.add_noise, "Cannot use VAE and add noise at the same time"
    else:
        if args.add_noise:
            setup_string = "with Risannen Noise Injections"
        else:
            setup_string = "with Normal U-Net"
    
    print(var_string + " " + setup_string)


    if not args.cluster:
        print("Running locally, Cluster =", args.cluster)
        # args.dim = int(args.dim/2)
        if args.device == 'cuda':
            warnings.warn('Consider running model on cluster-scale if CUDA is available')
    
    if args.test_run:
        print("Running Test Run with only one iter per epoch")
    
    print("Device: ", args.device)

    print(vars(args))

    # Run main function

    main(**vars(args))


    print("Finished Training")


#%%