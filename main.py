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

#from unet import UNet
#from mnist_unet import MNISTUnet
from scripts.karras_unet import KarrasUnet
from diffusion_utils import Degradation, Trainer, Sampler, ExponentialMovingAverage
from utils import create_dirs, save_video, save_gif, MyCelebA

# Check if ipykernel is running to check if we're working locally or on the cluster
import sys
if 'ipykernel' in sys.modules:
    sys.argv = ['']

def load_data(batch_size = 32, dataset = 'mnist'):
    
    assert dataset in ['mnist', 'cifar10', 'celeba', 'lsun_churches'],f"Invalid dataset, choose from ['mnist', 'cifar10', 'celeba', 'lsun_churches']"

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
    

    # Set up data loaders
    train_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader


def plot_degradation(timesteps, train_loader, **kwargs):

    noise = Degradation(timesteps = timesteps, degradation = 'noise', noise_schedule='cosine', dataset=kwargs['dataset'])
    blur = Degradation(timesteps = timesteps, degradation = 'blur', noise_schedule='cosine', dataset=kwargs['dataset'])
    black = Degradation(timesteps = timesteps, degradation = 'fadeblack', noise_schedule='cosine', dataset=kwargs['dataset'])
    black_blur = Degradation(timesteps = timesteps, degradation = 'fadeblack_blur', noise_schedule='cosine', dataset=kwargs['dataset'])

    timesteps = min(50, timesteps)

    plt.figure(figsize=(16, 5))
    for i in tqdm(range(timesteps), total = timesteps):

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
        plt.axis('off')

    # axis off
    plt.suptitle('Image degradation', size = 18)



def main(**kwargs):
    
    trainloader, valloader = load_data(kwargs['batch_size'], kwargs['dataset'])
    
    if kwargs['verbose']:
        plot_degradation(train_loader=trainloader, **kwargs)
        raise ValueError("Plotted degradation, exiting")
    
    x, _ = next(iter(trainloader))   
    channels, imsize = x[0].shape[0], x[0].shape[-1]
    
    # # Define Model
    # unet = UNet(image_size=imsize, 
    #             channels=channels, 
    #             num_downsamples=kwargs['num_downsamples'], 
    #             dim = kwargs['dim'], 
    #             dim_max =  kwargs['dim']*2**kwargs['num_downsamples'],
    #             dropout = 0)
    
    # unet = MNISTUnet(timesteps=kwargs['timesteps'],
    #                  in_channels=channels,
    #                  out_channels=channels,
    #                  time_embedding_dim=64,
    #                  dim_mults=[2,4],
    #                  base_dim=kwargs['dim'])

    unet = KarrasUnet(image_size=imsize, 
                    channels=channels, 
                    num_downsamples=kwargs['num_downsamples'], 
                    dim = kwargs['dim'], 
                    dim_max = kwargs['dim']*2**kwargs['num_downsamples'],
                    dropout = 0)
    
    
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
        sampler.sample_x_T(kwargs['n_samples'], channels, imsize)

    # Create directories
    imgpath, modelpath = create_dirs(**kwargs)
    ema_flag = '' if kwargs['skip_ema'] else '_ema'

    # Load Checkpoint
    if kwargs['load_checkpoint']:
        try:
            chkpt = torch.load(os.path.join(modelpath, f"chpkt_{kwargs['dim']}_{kwargs['timesteps']}_{kwargs['prediction']}{ema_flag}.pt"))
            trainer.model.load_state_dict(chkpt['model_state_dict'])
            trainer.optimizer.load_state_dict(chkpt['optimizer_state_dict'])
            trainer.model_ema.load_state_dict(chkpt['ema_state_dict'])
            epoch_offset = chkpt['epoch']
            print("Checkpoint loaded, continuing training from epoch", epoch_offset)
        except Exception as e:
            print("No checkpoint found: ", e)
            epoch_offset = 0
    else:
        epoch_offset = 0


    # Training Loop
    for e in range(epoch_offset + 1, kwargs['epochs']):
        
        sample_flag = True if (e) % kwargs['sample_interval'] == 0 else False 

        # Train
        trainer.model.train()
        trainloss = trainer.train_epoch(trainloader, val=False) # ATTENTION: CURRENTLY NO VALIDATION LOSS
        print(f"Epoch {e} Train Loss: {trainloss}")

        if sample_flag:

            # Validation
            trainer.model.eval()
            # valloss = trainer.train_epoch(valloader, val=True) # ATTENTION: CURRENTLY NO VALIDATION LOSS
            # print(f"Epoch {e} Validation Loss: {valloss}")
        
            # Sample
            nrow = 6
            samples = sampler.sample(trainer.model, kwargs['n_samples'])
            save_image(samples[-1], os.path.join(imgpath, f'epoch_{e}.png'), nrow=nrow) #int(math.sqrt(kwargs['n_samples']))
            save_video(samples, imgpath, nrow, f'epoch_{e}.mp4')
            save_gif(samples, imgpath, nrow, f'epoch_{e}.gif')

            # Save checkpoint
            if not kwargs['test_run']:
                chkpt = {
                    'epoch': e,
                    'model_state_dict': trainer.model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'ema_state_dict': trainer.model_ema.state_dict(),
                }
                torch.save(chkpt, os.path.join(modelpath, f"chpkt_{kwargs['dim']}_{kwargs['timesteps']}_{kwargs['prediction']}{ema_flag}.pt"))

        # Log to wandb
        wandb.log({"train loss": trainloss}, step=e)



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Diffusion Models')
    parser.add_argument('--timesteps', '--t', type=int, default=1000, help='Degradation timesteps')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--epochs', '--e', type=int, default=1000, help='Number of Training Epochs')
    parser.add_argument('--batch_size', '--b', type=int, default=64, help='Batch size')
    parser.add_argument('--dim', '--d', type=int, default=128, help='Model dimension')
    parser.add_argument('--prediction', '--pred', type=str, default='residual', help='Prediction method')
    parser.add_argument('--degradation', '--deg', type=str, default='noise', help='Degradation method')
    parser.add_argument('--noise_schedule', '--sched', type=str, default='cosine', help='Noise schedule')
    parser.add_argument('--dataset', type=str, default='mnist', help='Dataset to run Diffusion on. Choose one of [mnist, cifar10, celeba, lsun_churches]')
    parser.add_argument('--verbose', '--v', action='store_true', help='Verbose mode')
    parser.add_argument('--sample_interval', type=int, help='After how many epochs to sample', default=1)
    parser.add_argument('--cluster', '--clust', action='store_true', help='Whether to run script locally')
    parser.add_argument('--n_samples', type=int, default=60, help='Number of samples to generate')
    parser.add_argument('--load_checkpoint', action='store_true', help='Whether to try to load a checkpoint')
    parser.add_argument('--fix_sample', action='store_false', help='Whether to fix x_T for sampling, to see sample progression')
    parser.add_argument('--skip_ema', action='store_true', help='Whether to skip model EMA')
    parser.add_argument('--model_ema_steps', type=int, default=10, help='Model EMA steps')
    parser.add_argument('--model_ema_decay', type=float, default=0.995, help='Model EMA decay')
    parser.add_argument('--num_train_steps', type=int, default=700000, help='Number of training steps')
    parser.add_argument('--kernel_size', type=int, default=5, help='Number of training steps')
    parser.add_argument('--kernel_std', type=float, default=0.01, help='Number of training steps')
    parser.add_argument('--blur_routine', type=str, default='exponential', help='Number of training steps')
    parser.add_argument('--test_run', action='store_true', help='Whether to test run the pipeline')

    parser.add_argument('--add_noise', action='store_true', help='Whether to add noise to the deterministic sampling')

    args = parser.parse_args()

    args.num_downsamples = 2 if args.dataset == 'mnist' else 3
    args.device = 'cuda' if torch.cuda.is_available() else 'mps'

    if not args.cluster:
        print("Running locally, Cluster =", args.cluster)
        args.dim = int(args.dim/2)
        if args.device == 'cuda':
            warnings.warn('Consider running model on cluster-scale if CUDA is available')
    
    if args.test_run:
        print("Running Test Run with only one iter per epoch")

    print("Device: ", args.device)
    print("Arguments: ", args)

    # Initialize wandb
    wandb.init(
    project="Diffusion Thesis",
    config=vars(args))

    # Run main function
    main(**vars(args))

    # Finish wandb run
    wandb.finish()

    


# To Do Today:

# Debug Blurring Diffusion by comparing training and sampling 1:1 with Bansal et al.




### Difference: ema.module vs. unet - trainer.model vs. unet - currently investigated: No!
    # Small Batch Size
    # Only 25 timesteps - currently investigated: No!
    # 