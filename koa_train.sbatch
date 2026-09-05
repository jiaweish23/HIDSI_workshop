#!/bin/bash
# =============================================================================
# SCIPE workshop -- full training of the 3D fracture LDM on Koa (SLURM).
#
# This is the SHOW-ONLY job from section 9 of the notebook.  Training from
# scratch takes hours on a GPU, so we DO NOT run it during the workshop -- we
# load the pretrained checkpoints instead.  Kept here so you can see how a real
# training job is submitted on Koa.
#
#   module load lang/Anaconda3
#   conda env create -f environment.yml         # one-time
#   sbatch koa_train.sbatch                      # edit the two TODO lines first
#   squeue -u $USER ;  tail -f slurm-<jobid>.out
# =============================================================================
#SBATCH --job-name=scipe-ldm
#SBATCH --account=YOUR_ACCOUNT          # TODO: your Koa allocation/account
#SBATCH --partition=YOUR_GPU_PARTITION  # TODO: a GPU partition on Koa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out

set -e
module purge
module load lang/Anaconda3
source activate scipe-ldm            # env from environment.yml

echo "Job $SLURM_JOB_ID on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ---------------------------------------------------------------------------
# The training SCRIPTS + CONFIGS live in the research repo (fracture_ldm_reproduce),
# not in this self-contained workshop folder.  Point RES_REPO at your checkout.
# ---------------------------------------------------------------------------
RES_REPO=${RES_REPO:-$HOME/fracture_ldm_reproduce}
cd "$RES_REPO"

# Stage A -- train the VAE (skip if runs/3d_vae_binfromtudf/UnetVAE.pt exists)
python train_vae.py --config configs/3d_vae_binfromtudf.yml --ndim 3

# Stage B -- train the hard-data (1-2 well) conditioned LDM in the frozen latent
python train_ldm_harddata_edm_3d.py \
    --config configs/3d_harddata_binfromtudf_w12_lambda_cons0.yml

echo "Training finished."
