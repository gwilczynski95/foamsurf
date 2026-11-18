#!/usr/bin/env bash
#SBATCH --partition=rtx4090_batch
#SBATCH --qos=batch
#SBATCH --exclude=c22,c11
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm-%j.out

# sbatch --chdir in reproduce.sh supplies the repository; no site-specific path.
# Activate the environment and export dataset/asset variables before submission.
set -Eeuo pipefail
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MAX_JOBS="${MAX_JOBS:-4}"
exec bash scripts/reproduce.sh "$@"
