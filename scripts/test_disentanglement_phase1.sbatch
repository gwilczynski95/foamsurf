#!/usr/bin/env bash
#SBATCH --job-name=fs_dis_tests
#SBATCH --output=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/dis_tests_%j.out
#SBATCH --error=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/dis_tests_%j.err
#SBATCH --partition=rtx4090_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --qos=batch
#SBATCH --exclude=c22,c11
#SBATCH --time=02:00:00

set -Eeuo pipefail

ROOT=/shared/results/gmwilczn/foamsurf/foamsurf
ENV=/shared/results/gmwilczn/conda/envs/foamsurf
DRIVER=/shared/results/gmwilczn/drivers/NVIDIA-Linux-x86_64-535.183.01

export PATH="$ENV/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export CUDA_HOME="$ENV"
export FOAMSURF_DRIVER_LIBRARY_DIRECTORY="$DRIVER"
export LD_LIBRARY_PATH="$DRIVER:$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p "$ROOT/slurm_logs"
cd "$ROOT"

for K in 16 64; do
  BACKEND="$ROOT/renderer/foamsurf/_optix_k$K"
  if [[ ! -s "$BACKEND/PYOPTIXFLARERENDERER.so" || ! -s "$BACKEND/shaders.cu.ptx" ]]; then
    HIT_BUFFER_SIZE="$K" \
      OPTIX_OUTPUT_DIRECTORY="$BACKEND" \
      CONDA_ENV_PATH="$ENV" \
      bash renderer/build_optix.sh
  fi
done

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"$ENV/bin/python" -m pytest -q tests/b3
"$ENV/bin/python" -m pytest -q renderer/tests/test_b01_diagnostics.py
