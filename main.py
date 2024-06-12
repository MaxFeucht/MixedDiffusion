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

from unet import UNet
from mnist_unet import MNISTUnet
from scripts.bansal_unet import BansalUnet
from scripts.risannen_unet import RisannenUnet
from scripts.risannen_unet_vae import VAEUnet
#from scripts.efficient_diffusion_vae import VAEUnet

#from scripts.vae_unet_full import VAEUnet

from diffusion_utils import Degradation, Trainer, Sampler, ExponentialMovingAverage
from utils import load_dataset, plot_degradation, create_dirs, save_video, save_gif, MyCelebA

from sampler import sample_func

import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

# Check if ipykernel is running to check if we're working locally or on the cluster
import sys
if 'ipykernel' in sys.modules:
    sys.argv = ['']


def main(**kwargs):
    
    trainloader, valloader = load_dataset(kwargs['batch_size'], kwargs['dataset'])
    
    if kwargs['verbose']:
        plot_degradation(train_loader=trainloader, **kwargs)
    
    x, _ = next(iter(trainloader))   
    channels = x[0].shape[0]

    # Model Configuration
    if 'mnist' in kwargs['dataset']:
        attention_levels = (2,)
        ch_mult = (1,2,2)
        num_res_blocks = 2
        dropout = 0.1
    elif kwargs['dataset'] == 'cifar10':
        attention_levels = (2,3)
        ch_mult = (1, 2, 2, 2)
        num_res_blocks = 4
        dropout = 0.1
    elif kwargs['dataset'] == 'afhq':
        attention_levels = (2,3)
        ch_mult = (1, 2, 3, 4)
        num_res_blocks = 2
        dropout = 0.2
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

        print(kwargs['dim'])

        # Risannen Version
        unet = VAEUnet(image_size=kwargs["image_size"],
                        in_channels=channels,
                        dim=kwargs['dim'],
                        num_res_blocks=num_res_blocks,
                        attention_levels=attention_levels,
                        dropout=dropout,
                        ch_mult=ch_mult,
                        latent_dim = kwargs['latent_dim'],
                        noise_scale= kwargs['noise_scale'],
                        var_timestep=True if kwargs['prediction'] in ['xt', 'vxt'] else False,
                        vae_loc = kwargs['vae_loc'],
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
                            dropout=dropout,
                            ch_mult=ch_mult,
                            t2=True if kwargs['prediction'] in ['xt', 'vxt'] else False)


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
            if not kwargs['skip_wandb']:
                wandb.log({"train loss": trainloss,
                        "reconstruction loss": reconstruction,
                            "kl divergence": kl_div}, step = e)
            print(f"Epoch {e} Train Loss: {trainloss}, \nReconstruction Loss: {reconstruction}, \nKL Divergence: {kl_div}")
        else:
            trainloss = trainer.train_epoch(trainloader, val=False)
            if not kwargs['skip_wandb']:
                wandb.log({"train loss": trainloss}, step=e)
            print(f"Epoch {e} Train Loss: {trainloss}")

        if sample_flag:

            # Validation
            
            # Sample from model using EMA parameters
            trainer.model.eval()
            trainer.model_ema.store(trainer.model.parameters()) # Store model params
            trainer.model_ema.copy_to(trainer.model.parameters()) # Copy EMA params to model

            # Sample
            nrow = 6

            if kwargs['degradation'] in ['noise', 'fadeblack_noise'] : # Noise Sampling
                
                samples, xt = sampler.sample(trainer.model, kwargs['n_samples'])

                save_image(samples[-1], os.path.join(imgpath, f'sample_{e}.png'), nrow=nrow) #int(math.sqrt(kwargs['n_samples']))
                save_video(samples, imgpath, nrow, f'sample_{e}.mp4')
            
            else: 
                # Cold Sampling
                #og_img = next(iter(trainloader))[0][:kwargs['n_samples']].to(kwargs['device'])
                t_diff_var = kwargs['var_sampling_step'] if e % 2 != 0 else -1 # Alternate between sampling xt-t_diff style and x0 style
                t_diff_xt = kwargs['var_sampling_step'] if kwargs['prediction'] == 'xt' else 1 # Sampling xt-1 style but with bigger steps 
                _, xt, direct_recons, all_images = sampler.sample(model=trainer.model, 
                                                                        generate=True, 
                                                                        batch_size = kwargs['n_samples'],
                                                                        t_diff=t_diff_var if kwargs['prediction'] == 'vxt' else t_diff_xt) # Sample xt-1 style every second epoch

                # Prior is defined above under "fix_sample"
                gen_samples, gen_xt, _, gen_all_images = sampler.sample(model = trainer.model,
                                                                        generate=True,
                                                                        batch_size = kwargs['n_samples'], 
                                                                        prior=prior,
                                                                        t_diff=t_diff_var if kwargs['prediction'] == 'vxt' else t_diff_xt) # Sample xt-1 style every second epoch
                
                # Training Process conditional generation
                #save_image(og_img, os.path.join(imgpath, f'orig_{e}.png'), nrow=nrow)
                #save_image(xt, os.path.join(imgpath, f'xt_{e}.png'), nrow=nrow)
                save_image(all_images, os.path.join(imgpath, f'sample_regular_{e}.png'), nrow=nrow)
                if kwargs['prediction'] == 'x0':
                    save_image(direct_recons, os.path.join(imgpath, f'direct_recon_{e}.png'), nrow=nrow)

                # Training Process unconditional generation
                #save_image(gen_xt, os.path.join(imgpath, f'gen_xt_{e}.png'), nrow=nrow)
                save_image(gen_all_images, os.path.join(imgpath, f'gen_sample_regular_{e}.png'), nrow=nrow)
                #save_video(gen_samples, imgpath, nrow, f'sample_{e}.mp4')

                # After sampling, restore model parameters
                trainer.model_ema.restore(trainer.model.parameters()) # Restore model params

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
        
        # Sample Function for VXT
        if kwargs['prediction'] == 'vxt':
            if e % 2 == 0:
                sample_args = kwargs.copy()
                sample_args['trainer'] = trainer
                sample_args['cluster'] = True
                sample_args['sampling_steps'] = [20,50,70,100,150,200]
                sample_args['e'] = e
                sample_func(**sample_args)





if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Diffusion Models')

    # General Diffusion Parameters
    parser.add_argument('--timesteps', '--t', type=int, default=50, help='Degradation timesteps')
    parser.add_argument('--prediction', '--pred', type=str, default='vxt', help='Prediction method, choose one of [x0, xt, residual]')
    parser.add_argument('--dataset', type=str, default='fashionmnist', help='Dataset to run Diffusion on. Choose one of [mnist, cifar10, celeba, lsun_churches]')
    parser.add_argument('--degradation', '--deg', type=str, default='fadeblack_blur', help='Degradation method')
    parser.add_argument('--batch_size', '--b', type=int, default=64, help='Batch size')
    parser.add_argument('--dim', '--d', type=int , default=64, help='Model dimension')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--epochs', '--e', type=int, default=20, help='Number of Training Epochs')
    parser.add_argument('--noise_schedule', '--sched', type=str, default='cosine', help='Noise schedule')
    parser.add_argument('--loss_weighting', action='store_true', help='Whether to use weighting for reconstruction loss')
    parser.add_argument('--var_sampling_step', type=int, default = 1, help='How to sample var timestep model - int > 0 indicates t difference to predict, -1 indicates x0 prediction')
    parser.add_argument('--min_t2_step', type=int, default=1, help='With what min step size to discretize t2 in variational timestep model')
    parser.add_argument('--baseline', '--base', type=str, default='xxxx', help='Whether to run a baseline model - Risannen, Bansal, VAE')

    # Noise Injection Parameters
    parser.add_argument('--vae', action='store_true', help='Whether to use VAE Noise injections')
    parser.add_argument('--vae_alpha', type=float, default = 0.999, help='Trade-off parameter for weight of Reconstruction and KL Div')
    parser.add_argument('--latent_dim', type=int, default=32, help='Which dimension the VAE latent space is supposed to have')
    parser.add_argument('--add_noise', action='store_true', help='Whether to add noise Risannen et al. style')
    parser.add_argument('--break_symmetry', action='store_true', help='Whether to add noise to xT Bansal et al. style')
    parser.add_argument('--noise_scale', type=float, default = 0.01, help='How much Noise to add to the input')
    parser.add_argument('--vae_loc', type=str, default = 'start', help='Where to inject VAE Noise. One of [start, bottleneck, emb].')
    parser.add_argument('--vae_inject', type=str, default = 'add', help='How to inject VAE Noise. One of [concat, add].')
    parser.add_argument('--xt_dropout', type=float, default = 0.2, help='How much of xt is dropped out at every step (to foster reliance on VAE injections)')

    # Housekeeping Parameters
    parser.add_argument('--load_checkpoint', action='store_true', help='Whether to try to load a checkpoint')
    parser.add_argument('--sample_interval', type=int, help='After how many epochs to sample', default=1)
    parser.add_argument('--n_samples', type=int, default=60, help='Number of samples to generate')
    parser.add_argument('--fix_sample', action='store_false', help='Whether to fix x_T for sampling, to see sample progression')
    parser.add_argument('--skip_ema', action='store_true', help='Whether to skip model EMA')
    parser.add_argument('--model_ema_decay', type=float, default=0.997, help='Model EMA decay')
    parser.add_argument('--cluster', action='store_true', help='Whether to run script locally')
    parser.add_argument('--skip_wandb', action='store_true', help='Whether to skip wandb logging')
    parser.add_argument('--verbose', '--v', action='store_true', help='Verbose mode')

    parser.add_argument('--test_run', action='store_true', help='Whether to test run the pipeline')

    args = parser.parse_args()

    args.num_downsamples = 2 if args.dataset == 'mnist' else 3
    args.device = 'cuda' if torch.cuda.is_available() else 'mps'

    if 'mnist' in args.dataset:
        args.image_size = 28
    elif args.dataset == 'cifar10':
        args.image_size = 32
    elif args.dataset == 'afhq':
        args.image_size = 64


    if args.baseline == 'ddpm':
        args.prediction = 'residual'
        args.degradation = 'noise'
        args.vae = False
        args.add_noise = False
        args.break_symmetry = False
    elif args.baseline == 'risannen':
        args.vae = False
        args.add_noise = True
        args.break_symmetry = False
        args.prediction = 'xt'
        args.noise_scale = 0.01
    elif args.baseline == 'bansal':
        args.vae = False
        args.add_noise = False
        args.break_symmetry = True
        args.prediction = 'x0'
        args.noise_scale = 0.002
    elif args.baseline == 'vae_xt':
        args.vae = True
        args.add_noise = False
        args.break_symmetry = False
        args.prediction = 'xt'
    elif args.baseline == 'vae_x0':
        args.vae = True
        args.add_noise = False
        args.break_symmetry = False
        args.prediction = 'x0'


    if args.prediction == 'vxt':
        var_string = "Running Variable Timestep Diffusion"
    else:
        var_string = "Running Sequential Diffusion"

    if not args.cluster:
        print("Running locally, Cluster =", args.cluster)
        if args.device == 'cuda':
            warnings.warn('Consider running model on cluster-scale if CUDA is available')
    
    if args.test_run:
        print("Running Test Run with only one iter per epoch")

    if args.vae:
        setup_string = "using VAE Noise Injections"
        assert not args.add_noise, "Cannot use VAE and add noise at the same time"
    else:
        if args.add_noise:
            setup_string = "with Risannen Noise Injections"
        else:
            setup_string = "with Normal U-Net"
    
    print(var_string + " " + setup_string)
    
    # Initialize wandb
    if not args.skip_wandb:
        wandb.init(
        project="Diffusion Thesis",
        config=vars(args))
    
    print("Device: ", args.device)


    # Run main function
    main(**vars(args))

    # Finish wandb run
    if not args.test_run:
        wandb.finish()

    print("Finished Training")