#!/usr/bin/env bash
#SBATCH --job-name=fs_g_flare_ctrl
#SBATCH --output=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/g_flare_ctrl_%j.out
#SBATCH --error=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/g_flare_ctrl_%j.err
#SBATCH --partition=rtx4090_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --qos=batch
#SBATCH --exclude=c22,c11
#SBATCH --time=08:00:00
set -Eeuo pipefail

ROOT=/shared/results/gmwilczn/foamsurf/foamsurf
ENV=/shared/results/gmwilczn/conda/envs/foamsurf
DRIVER=/shared/results/gmwilczn/drivers/NVIDIA-Linux-x86_64-535.183.01
RUN_ID="${SLURM_JOB_ID:-local}_$(date +%Y%m%d_%H%M%S)"
OUTPUT="$ROOT/outputs/stage1_gaussian_flare_control/$RUN_ID"

export PATH="$ENV/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export CUDA_HOME="$ENV"
export FOAMSURF_DRIVER_LIBRARY_DIRECTORY="$DRIVER"
export LD_LIBRARY_PATH="$DRIVER:$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

if [[ ! -x "$ENV/bin/python" ]]; then
  echo "Python not found: $ENV/bin/python" >&2
  exit 1
fi
if [[ ! -f "$ROOT/renderer/foamsurf/_C.so" ]]; then
  echo "Renderer is not built. Run ./build.sh before sbatch." >&2
  exit 1
fi
if [[ ! -f "$ROOT/renderer/foamsurf/_optix/PYOPTIXFLARERENDERER.so" ]]; then
  echo "OptiX renderer is not built. Run ./build.sh before sbatch." >&2
  exit 1
fi

mkdir -p "$ROOT/slurm_logs"
cd "$ROOT"
echo "node=$(hostname)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
"$ENV/bin/python" -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); import foamsurf_topology; print('foamsurf_topology import OK')"
"$ENV/bin/python" train_b0.py \
  --config configs/stage1_gaussian_flare_control.yaml \
  --output-dir "$OUTPUT" \
  --wandb-mode "${WANDB_MODE:-online}" \
  --wandb-name "${WANDB_NAME:-g_flare_ctrl_$RUN_ID}"

test -f "$OUTPUT/summary.json"
test -f "$OUTPUT/checkpoints/best.pt"
