#!/usr/bin/env bash
#SBATCH --job-name=fs_smoke
#SBATCH --output=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/smoke_%j.out
#SBATCH --error=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/smoke_%j.err
#SBATCH --partition=rtx4090_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --qos=batch
#SBATCH --exclude=c22,c11
#SBATCH --time=01:00:00
# Full test suite (incl. short GPU trainings) on an RTX4090. Gate sweeps on it:
#   smoke=$(sbatch --parsable scripts/smoke_tests.sbatch)
#   FS_VARIANT_SET=... sbatch --dependency=afterok:$smoke ... scripts/sweep_medium_fix.sbatch
set -Eeuo pipefail
ROOT=/shared/results/gmwilczn/foamsurf/foamsurf
FS_ENV=/shared/results/gmwilczn/conda/envs/foamsurf
DRIVER=/shared/results/gmwilczn/drivers/NVIDIA-Linux-x86_64-535.183.01
export PATH="$FS_ENV/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export CUDA_HOME="$FS_ENV"
export FOAMSURF_DRIVER_LIBRARY_DIRECTORY="$DRIVER"
export LD_LIBRARY_PATH="$DRIVER:$FS_ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export TORCH_CUDA_ARCH_LIST=8.9
cd "$ROOT"
nvidia-smi --query-gpu=name --format=csv,noheader
# Fail loudly if CUDA is missing: GPU tests would otherwise be skipped silently.
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"
python -m pytest -q tests/ -rs
# Renderer kernels (fused RGB vs six-pass and the autograd oracle).
(cd renderer && python -m pytest -q tests/test_rgb_medium_renderer.py -rs)
