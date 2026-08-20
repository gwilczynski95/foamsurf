#!/usr/bin/env bash
#SBATCH --job-name=fs_best_2dgs
#SBATCH --output=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/best_2dgs_%j.out
#SBATCH --error=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/best_2dgs_%j.err
#SBATCH --partition=rtx4090_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --qos=batch
#SBATCH --exclude=c22,c11
#SBATCH --time=24:00:00

set -Eeuo pipefail

ROOT=/shared/results/gmwilczn/foamsurf/foamsurf
ENV=/shared/results/gmwilczn/conda/envs/foamsurf
DRIVER=/shared/results/gmwilczn/drivers/NVIDIA-Linux-x86_64-535.183.01
BACKEND="$ROOT/renderer/foamsurf/_optix_k64"
RUN_ID="${SLURM_JOB_ID:-local}_$(date +%Y%m%d_%H%M%S)"
OUTPUT="$ROOT/outputs/baseline_best_2d_gaussians/$RUN_ID"

export PATH="$ENV/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export CUDA_HOME="$ENV"
export FOAMSURF_DRIVER_LIBRARY_DIRECTORY="$DRIVER"
export LD_LIBRARY_PATH="$DRIVER:$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p "$ROOT/slurm_logs"
cd "$ROOT"

if [[ ! -s "$BACKEND/PYOPTIXFLARERENDERER.so" || ! -s "$BACKEND/shaders.cu.ptx" ]]; then
  echo "K=64 OptiX backend is missing; building it now."
  HIT_BUFFER_SIZE=64 \
    OPTIX_OUTPUT_DIRECTORY="$BACKEND" \
    CONDA_ENV_PATH="$ENV" \
    bash renderer/build_optix.sh
fi

echo "node=$(hostname) K=64 backend=$BACKEND output=$OUTPUT"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"$ENV/bin/python" -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); import foamsurf_topology; print('foamsurf_topology import OK')"

"$ENV/bin/python" train_b0.py \
  --config configs/baseline_best_2d_gaussians.yaml \
  --output-dir "$OUTPUT" \
  --wandb-mode "${WANDB_MODE:-online}" \
  --wandb-name "${WANDB_NAME:-best_2dgs_$RUN_ID}"

test -f "$OUTPUT/summary.json"
test -f "$OUTPUT/checkpoints/best.pt"
