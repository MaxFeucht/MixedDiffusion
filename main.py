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

# Check if ipykernel is running to check if we're working locally or on the cluster
import sys
if 'ipykernel' in sys.modules:
    sys.argv = ['']


def load_dataset(batch_size = 32, dataset = 'mnist'):
    
    assert dataset in ['mnist', 'cifar10', 'celeba', 'lsun_churches', 'afhq'],f"Invalid dataset, choose from ['mnist', 'cifar10', 'celeba', 'lsun_churches']"

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

    noise = Degradation(timesteps = timesteps, degradation = 'noise', noise_schedule='cosine', dataset=kwargs['dataset'], device=kwargs['device'])
    blur = Degradation(timesteps = timesteps, degradation = 'blur', noise_schedule='cosine', dataset=kwargs['dataset'], device=kwargs['device'])
    black = Degradation(timesteps = timesteps, degradation = 'fadeblack', noise_schedule='cosine', dataset=kwargs['dataset'], device=kwargs['device'])
    black_blur = Degradation(timesteps = timesteps, degradation = 'fadeblack_blur', noise_schedule='cosine', dataset=kwargs['dataset'], device=kwargs['device'])

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
        x_blackblur = black_blur.degrade(x, t).cpu()
        x_blackblur = x_blackblur[0].unsqueeze(0) if len(x_blackblur[0].shape) == 2 else x_blackblur[0]
        plt.imshow(x_blackblur.permute(1, 2, 0), vmin=0, vmax=1)   
        plt.axis('off')
        

    # axis off
    plt.suptitle('Image degradation', size = 18)



def main(**kwargs):
    
    trainloader, valloader = load_dataset(kwargs['batch_size'], kwargs['dataset'])
    
    if kwargs['verbose']:
        plot_degradation(train_loader=trainloader, **kwargs)
    
    x, _ = next(iter(trainloader))   
    channels = x[0].shape[0]
    
    # Define Model
    if kwargs['vae']:

        # Bansal Version
        # unet = VAEUnet(image_size=imsize,
        #                 channels=channels,
        #                 out_ch=channels,
        #                 ch=kwargs['dim'],
        #                 ch_mult= (1,2,2) if kwargs['dataset'] == 'mnist' else (1,2,2,2),
        #                 num_res_blocks=2,
        #                 attn_resolutions=(14,) if kwargs['dataset'] == 'mnist' else (16,),
        #                 latent_dim=int(channels*imsize*imsize//kwargs['vae_downsample']),
        #                 noise_scale=kwargs['noise_scale'],
        #                 dropout=0)


        # Risannen Version
        unet = VAEUnet(image_size=kwargs['image_size'],
                        in_channels=channels,
                        dim=kwargs['dim'],
                        num_res_blocks=2,
                        attention_levels=(2,) if kwargs['dataset'] == 'mnist' else (2,3),
                        dropout=0.1,
                        ch_mult=(1,2,2) if kwargs['dataset'] == 'mnist' else (1,2,2,2),
                        latent_dim=int(channels*kwargs['image_size']*kwargs['image_size']//kwargs['vae_downsample']),
                        noise_scale=kwargs['noise_scale'])

    else:
        # unet = BansalUnet(image_size=imsize,
        #         channels=channels,
        #         out_ch=channels,
        #         ch=kwargs['dim'],
        #         ch_mult= (1,2,2) if kwargs['dataset'] == 'mnist' else (1,2,2,2),
        #         num_res_blocks=2,
        #         attn_resolutions=(14,) if kwargs['dataset'] == 'mnist' else (16,),
        #         dropout=0)
    
        unet = RisannenUnet(image_size=kwargs['image_size'],
                            in_channels=channels,
                            dim=kwargs['dim'],
                            num_res_blocks=2,
                            attention_levels=(2,) if kwargs['dataset'] == 'mnist' else (2,3),
                            dropout=0.1,
                            ch_mult=(1,2,2) if kwargs['dataset'] == 'mnist' else (1,2,2,2))


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
        sampler.sample_x_T(kwargs['n_samples'], channels, kwargs['image_size'])

        # Fix Prior for VAE
        latent_dim = int(channels*kwargs['image_size']*kwargs['image_size'])#//kwargs['vae_downsample'])
        prior = torch.randn(kwargs['n_samples'], latent_dim).to(kwargs['device'])
        
        #prior = torch.randn(kwargs['n_samples'], imsize).to(kwargs['device'])
        #res = imsize//2**kwargs['num_downsamples']
        #prior = torch.randn(kwargs['n_samples'], res, res).to(kwargs['device'])
        

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
        if kwargs['vae']:
            trainloss, reconstruction, kl_div = trainer.train_epoch(trainloader, val=False) # ATTENTION: CURRENTLY NO VALIDATION LOSS
            #if not kwargs['test_run']:
            wandb.log({"train loss": trainloss,
                    "reconstruction loss": reconstruction,
                        "kl divergence": kl_div}, step = e)
        else:
            trainloss = trainer.train_epoch(trainloader, val=False)
            #if not kwargs['test_run']:
            wandb.log({"train loss": trainloss}, step=e)

        print(f"Epoch {e} Train Loss: {trainloss}")

        if sample_flag:

            # Validation
            trainer.model.eval()
            # valloss = trainer.train_epoch(valloader, val=True) # ATTENTION: CURRENTLY NO VALIDATION LOSS
            # print(f"Epoch {e} Validation Loss: {valloss}")
        
            # Sample
            nrow = 6

            if kwargs['degradation'] == 'noise': # Noise Sampling
                samples, xt = sampler.sample(trainer.model, kwargs['n_samples'])
                
                save_image(samples[-1], os.path.join(imgpath, f'sample_{e}.png'), nrow=nrow) #int(math.sqrt(kwargs['n_samples']))
                save_video(samples, imgpath, nrow, f'sample_{e}.mp4')
            
            else: # Cold Sampling
                og_img = next(iter(trainloader))[0][:kwargs['n_samples']].to(kwargs['device'])
                _, xt, direct_recons, all_images = sampler.sample(model = trainer.model, 
                                                                    #x0=og_img, 
                                                                    generate=True, 
                                                                    batch_size = kwargs['n_samples'])

                # Prior is defined above under "fix_sample"
                gen_samples, gen_xt, _, gen_all_images = sampler.sample(model = trainer.model, 
                                                                        batch_size = kwargs['n_samples'], 
                                                                        generate=True, 
                                                                        prior=prior)
                
                # Training Process conditional generation
                #save_image(og_img, os.path.join(imgpath, f'orig_{e}.png'), nrow=nrow)
                #save_image(xt, os.path.join(imgpath, f'xt_{e}.png'), nrow=nrow)
                save_image(all_images, os.path.join(imgpath, f'sample_regular_{e}.png'), nrow=nrow)
                if kwargs['prediction'] == 'x0':
                    save_image(direct_recons, os.path.join(imgpath, f'direct_recon_{e}.png'), nrow=nrow)

                # Training Process unconditional generation
                #save_image(gen_xt, os.path.join(imgpath, f'gen_xt_{e}.png'), nrow=nrow)
                save_image(gen_all_images, os.path.join(imgpath, f'gen_sample_regular_{e}.png'), nrow=nrow)
                save_video(gen_samples, imgpath, nrow, f'sample_{e}.mp4')


            # save_gif(samples, imgpath, nrow, f'sample_{e}.gif')

            # Save checkpoint
            if not kwargs['test_run']:
                chkpt = {
                    'epoch': e,
                    'model_state_dict': trainer.model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'ema_state_dict': trainer.model_ema.state_dict(),
                }
                torch.save(chkpt, os.path.join(modelpath, f"chpkt_{kwargs['dim']}_{kwargs['timesteps']}_{kwargs['prediction']}{ema_flag}.pt"))





if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Diffusion Models')

    # General Diffusion Parameters
    parser.add_argument('--timesteps', '--t', type=int, default=10, help='Degradation timesteps')
    parser.add_argument('--prediction', '--pred', type=str, default='xtm1', help='Prediction method, choose one of [x0, xtm1, residual]')
    parser.add_argument('--dataset', type=str, default='mnist', help='Dataset to run Diffusion on. Choose one of [mnist, cifar10, celeba, lsun_churches]')
    parser.add_argument('--degradation', '--deg', type=str, default='fadeblack_blur', help='Degradation method')
    parser.add_argument('--batch_size', '--b', type=int, default=64, help='Batch size')
    parser.add_argument('--dim', '--d', type=int , default=64, help='Model dimension')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--epochs', '--e', type=int, default=20, help='Number of Training Epochs')
    parser.add_argument('--noise_schedule', '--sched', type=str, default='cosine', help='Noise schedule')
    parser.add_argument('--xt_weighting', action='store_true', help='Whether to use weighting for xt in loss')
    #parser.add_argument('--recursive_x0', action='store_', help='Whether to predict x0 recursively as during sampling')

    # Noise Injection Parameters
    parser.add_argument('--vae', action='store_false', help='Whether to use VAE Noise injections')
    parser.add_argument('--vae_alpha', type=float, default = 0.999, help='Trade-off parameter for weight of Reconstruction and KL Div')
    parser.add_argument('--vae_downsample', type=float, default=1, help='To which degree to downsample and repeat the VAE noise injections')
    parser.add_argument('--add_noise', action='store_true', help='Whether to add noise Risannen et al. style')
    parser.add_argument('--noise_scale', type=float, default = 0.01, help='How much Noise to add to the input')

    # Housekeeping Parameters
    parser.add_argument('--load_checkpoint', action='store_true', help='Whether to try to load a checkpoint')
    parser.add_argument('--sample_interval', type=int, help='After how many epochs to sample', default=1)
    parser.add_argument('--n_samples', type=int, default=60, help='Number of samples to generate')
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
        args.dim = int(args.dim/2)
        if args.device == 'cuda':
            warnings.warn('Consider running model on cluster-scale if CUDA is available')
    
    if args.test_run:
        print("Running Test Run with only one iter per epoch")

    # Initialize wandb
    wandb.init(
    project="Diffusion Thesis",
    config=vars(args))
    
    print("Device: ", args.device)


    # Run main function

    main(**vars(args))

    # Finish wandb run
    #if not args.test_run:
    wandb.finish()

    print("Finished Training")