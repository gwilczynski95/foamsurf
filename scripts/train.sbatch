#!/usr/bin/env bash
#SBATCH --job-name=foamsurf
#SBATCH --output=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/train_%j.out
#SBATCH --error=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/train_%j.err
#SBATCH --partition=rtx4090_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --qos=batch
#SBATCH --exclude=c22,c11
#SBATCH --time=24:00:00
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
export MAX_JOBS=2
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
cd "$ROOT"
CONFIG="${FS_CONFIG:-configs/iui3_5k.yaml}"
OUTPUT="${FS_OUTPUT:-outputs/train/${SLURM_JOB_ID}}"
mkdir -p slurm_logs
export TORCH_EXTENSIONS_DIR="$ROOT/outputs/build_${SLURM_JOB_ID}"
mkdir -p "$TORCH_EXTENSIONS_DIR"
ARGS=(--config "$CONFIG" --output-dir "$OUTPUT")
if [[ "${FS_RESUME:-0}" == 1 ]]; then ARGS+=(--resume); fi
python train.py "${ARGS[@]}"
