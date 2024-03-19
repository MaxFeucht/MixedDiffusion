#!/bin/bash
#SBATCH --job-name=cramming
#SBATCH --time=72:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=defq
#SBATCH -C A4000
#SBATCH --gres=gpu:1

module load cuda11.7/toolkit
module load cuDNN/cuda11.7

source /home/pbloem/.bashrc

cd /var/scratch/pbloem

lr=1e-4
acc=24
bs=320_000
up=true
hrs=24
reuse=true
wm=10e5 #${SLURM_ARRAY_TASK_ID}e5
layers=8
reset=0.6 #${SLURM_ARRAY_TASK_ID}
mix=0.4
decay=5e-6 #${SLURM_ARRAY_TASK_ID}e-6

name="m$mix-d$decay-320snapshot"

if [ "$up" = false ]; then
   name="cramming-baseline-h$hrs"
fi

mkdir $name
cd $name

export HF_DATASETS_CACHE="/var/scratch/pbloem"
export TRANSFORMERS_CACHE="/var/scratch/pbloem"

python /home/pbloem/git/cramming/pretrain.py name=$name train=bert-o4 data=pile-readymade arch=crammed-bert data=pile-readymade \
                   budget=24 up.warmup=3_000_000 up.acc_warmup=3_000_000 up.enabled=$up up.num_batches=$bs up.lr=$lr up.accumulate=$acc up.batch_size=46 \
                   up.buffer_size=10_000 up.spinup=500 up.source_layers=$layers up.sample_batch_size=50 up.print_every=2_000 \
                   up.temperature=0.0 up.init_mult_max=$wm up.eval_ood_every=5_000 up.source_mode=nn up.reset_prob=$reset \
                   up.cooldown=9_000_000 up.reuse_opt=$reuse up.opt_mult=1.0 up.weight_decay=0.0 up.betas="[0.8,0.99]" impl.troubleshoot_strategy=['dump_nan_grads'] \
                   up.iterations=1 up.init_mode=minimal up.transfer=discrete up.loss_mask_scale=0.0 up.bid_source=True \
                   train.scheduler="budget-triangle2" up.up_mix=$mix up.up_mix_decay=$decay up.nrehearsal=10_000 \
                   wandb.enabled=true wandb.entity=pboemesquire wandb.project=up-cramming train.optim.lr=0.001 \
                   up.snapshot="/var/scratch/pbloem/snapshots/b320k_reducedbetas.pt" up.reset_betas=false

python /home/pbloem/git/cramming/eval.py eval=GLUE_sane name=$name eval.checkpoint=latest \
                   impl.microbatch_size=16 impl.shuffle_in_dataloader=True impl.compile_torch=False \
                   wandb.enabled=true wandb.entity=pboemesquire wandb.project=up-cramming