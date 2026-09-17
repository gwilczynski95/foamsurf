#!/usr/bin/env bash
#SBATCH --job-name=fs_dis_evict
#SBATCH --output=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/dis_evict_%A_%a.out
#SBATCH --error=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/dis_evict_%A_%a.err
#SBATCH --array=0-99%8
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
TASK=$SLURM_ARRAY_TASK_ID
RUN="dis_evict_${SLURM_ARRAY_JOB_ID}_${TASK}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/disentanglement_eviction}"
OUTPUT="$OUTPUT_ROOT/$RUN"

export PATH="$ENV/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export CUDA_HOME="$ENV"
export FOAMSURF_DRIVER_LIBRARY_DIRECTORY="$DRIVER"
export LD_LIBRARY_PATH="$DRIVER:$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p "$ROOT/slurm_logs"
cd "$ROOT"

if [[ ! -s "$BACKEND/PYOPTIXFLARERENDERER.so" || ! -s "$BACKEND/shaders.cu.ptx" ]]; then
  (
    flock 9
    if [[ ! -s "$BACKEND/PYOPTIXFLARERENDERER.so" || ! -s "$BACKEND/shaders.cu.ptx" ]]; then
      HIT_BUFFER_SIZE=64 \
        OPTIX_OUTPUT_DIRECTORY="$BACKEND" \
        CONDA_ENV_PATH="$ENV" \
        bash renderer/build_optix.sh
    fi
  ) 9>"$ROOT/slurm_logs/optix_k64_build.lock"
fi

echo "FoamSurf Gaussian-eviction sweep task=$TASK/99 K=64 seed=0"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"$ENV/bin/python" -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"

"$ENV/bin/python" scripts/run_disentanglement_eviction.py \
  --config configs/baseline_clean_joint.yaml \
  --index "$TASK" \
  --output-dir "$OUTPUT" \
  --wandb-name "$RUN" \
  --wandb-mode "${WANDB_MODE:-online}"

test -f "$OUTPUT/config.yaml"
test -f "$OUTPUT/checkpoints/best.pt"
test -f "$OUTPUT/checkpoints/last.pt"
test -f "$OUTPUT/metrics/train.csv"
test -f "$OUTPUT/metrics/eval.csv"
test -f "$OUTPUT/summary.json"
