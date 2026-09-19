#!/usr/bin/env bash
#SBATCH --job-name=fs_warm_analyze
#SBATCH --output=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/warmup_analyze_%j.out
#SBATCH --error=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/warmup_analyze_%j.err
#SBATCH --partition=rtx4090_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --qos=batch
#SBATCH --exclude=c22,c11
#SBATCH --time=12:00:00
set -Eeuo pipefail
: "${WARMUP_RUN:?Set WARMUP_RUN to a completed run directory}"
ROOT=/shared/results/gmwilczn/foamsurf/foamsurf
FS_ENV=/shared/results/gmwilczn/conda/envs/foamsurf
DRIVER=/shared/results/gmwilczn/drivers/NVIDIA-Linux-x86_64-535.183.01
export PATH="$FS_ENV/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export CUDA_HOME="$FS_ENV"
export FOAMSURF_DRIVER_LIBRARY_DIRECTORY="$DRIVER"
export LD_LIBRARY_PATH="$DRIVER:$FS_ENV/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST=8.9
export MAX_JOBS=2
export TORCH_EXTENSIONS_DIR="$ROOT/outputs/warmup_analysis_build_${SLURM_JOB_ID}"
cd "$ROOT"
mkdir -p "$TORCH_EXTENSIONS_DIR"
python scripts/analyze_warmup_radfoam_like.py --run "$WARMUP_RUN"
