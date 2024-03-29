#!/bin/bash
#SBATCH --job-name=blur_residual_test
#SBATCH --time=10:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=defq
#SBATCH -C A4000
#SBATCH --gres=gpu:1
#SBATCH -o output.out

## in the list above, the partition name depends on where you are running your job. 
## On DAS5 the default would be `defq` on Lisa the default would be `gpu` or `gpu_shared`
## Typing `sinfo` on the server command line gives a column called PARTITION.  There, one can find the name of a specific node, the state (down, alloc, idle etc), the availability and how long is the time limit . Ask your supervisor before running jobs on queues you do not know.

# Load GPU drivers

## Enable the following two lines for DAS5
# module load cuda10.0/toolkit
# module load cuDNN/cuda10.0

## Enable the following line for DAS6
module load cuda11.3/toolkit/11.3.1

## For Lisa and Snellius, modules are usually not needed
## https://userinfo.surfsara.nl/systems/shared/modules 

# This loads the anaconda virtual environment with our packages
source $HOME/.bashrc
conda activate

# Scratch directory has far more space than home directory.
mkdir /var/scratch/mft520/experiments
cd /var/scratch/mft520/experiments

# # Base directory for the experiment
# mkdir $HOME/experiments
# cd $HOME/experiments

# Simple trick to create a unique directory for each run of the script
# echo $$
# mkdir o`echo $$`
# cd o`echo $$`

## Set Vars

lr=2e-5
batch_size=128
timesteps=200
dim=128
epochs=1000
prediction="x0"
degradation="blur"
noise_schedule="cosine"
dataset="cifar10"
sample_interval=1
n_samples=72
model_ema_steps=10
model_ema_decay=0.995
num_train_steps=700000


# Run the actual experiment. 
python /var/scratch/mft520/MixedDiffusion/main.py --epochs $epochs --batch_size $batch_size --timesteps $timesteps --dim $dim \
                                                --lr $lr --prediction $prediction --degradation $degradation \
                                                --noise_schedule $noise_schedule --dataset $dataset --sample_interval $sample_interval \
                                                --n_samples $n_samples --num_train_steps $num_train_steps \
                                                --model_ema_steps $model_ema_steps --model_ema_decay $model_ema_decay \
                                                --load_checkpoint --cluster --add_noise --skip_ema
echo "Script finished"
