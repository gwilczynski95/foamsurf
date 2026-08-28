#!/usr/bin/env bash
#SBATCH --job-name=fs_g_k
#SBATCH --output=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/gk_%A_%a.out
#SBATCH --error=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/gk_%A_%a.err
#SBATCH --array=0-47%8
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
TASK=$SLURM_ARRAY_TASK_ID
K_VALUES=(32 64 128 16)
K=${K_VALUES[$((TASK / 12))]}
RUN="gk_${SLURM_ARRAY_JOB_ID}_${TASK}"
OUTPUT="$ROOT/outputs/stage1_gaussian_k_sweep/$RUN"
BACKEND="$ROOT/renderer/foamsurf/_optix_k$K"

export PATH="$ENV/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export CUDA_HOME="$ENV"
export FOAMSURF_DRIVER_LIBRARY_DIRECTORY="$DRIVER"
export LD_LIBRARY_PATH="$DRIVER:$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p "$ROOT/slurm_logs"
cd "$ROOT"

test -s "$BACKEND/PYOPTIXFLARERENDERER.so"
test -s "$BACKEND/shaders.cu.ptx"

echo "node=$(hostname) task=$TASK K=$K backend=$BACKEND"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"$ENV/bin/python" -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); import foamsurf_topology; print('foamsurf_topology import OK')"

"$ENV/bin/python" scripts/run_gaussian_k_sweep.py \
    --config configs/stage1_gaussian_flare_control.yaml \
    --index "$TASK" \
    --output-dir "$OUTPUT" \
    --wandb-name "$RUN" \
    --wandb-mode "${WANDB_MODE:-online}"

test -f "$OUTPUT/summary.json"
test -f "$OUTPUT/checkpoints/best.pt"
