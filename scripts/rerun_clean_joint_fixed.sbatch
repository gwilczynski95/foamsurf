#!/usr/bin/env bash
#SBATCH --job-name=fs_joint_fix
#SBATCH --output=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/joint_fix_%j.out
#SBATCH --error=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/joint_fix_%j.err
#SBATCH --partition=rtx4090_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --qos=batch
#SBATCH --exclude=c22,c11
#SBATCH --time=24:00:00

set -Eeuo pipefail

ROOT=/shared/results/gmwilczn/foamsurf/foamsurf
RUN_ID="${SLURM_JOB_ID:-local}_$(date +%Y%m%d_%H%M%S)"
export OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/baseline_clean_joint_fixed/seed0_$RUN_ID}"
export WANDB_NAME="${WANDB_NAME:-clean_joint_fixed_seed0_$RUN_ID}"

# This delegates to the original launcher so all scientific settings remain
# byte-for-byte identical to the failed baseline. Only the output identity and
# the repaired replace_population implementation differ.
exec bash "$ROOT/scripts/train_clean_joint.sbatch"
