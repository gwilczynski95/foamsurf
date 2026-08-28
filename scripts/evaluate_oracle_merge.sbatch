#!/usr/bin/env bash
#SBATCH --job-name=fs_oracle_merge
#SBATCH --output=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/oracle_merge_%j.out
#SBATCH --error=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/oracle_merge_%j.err
#SBATCH --partition=rtx4090_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --qos=batch
#SBATCH --exclude=c22,c11
#SBATCH --time=04:00:00

set -Eeuo pipefail

ROOT=/shared/results/gmwilczn/foamsurf/foamsurf
ENV=/shared/results/gmwilczn/conda/envs/foamsurf
DRIVER=/shared/results/gmwilczn/drivers/NVIDIA-Linux-x86_64-535.183.01
BACKEND="$ROOT/renderer/foamsurf/_optix_k64"

GAUSSIAN_CHECKPOINT="${GAUSSIAN_CHECKPOINT:-$ROOT/outputs/stage1_gaussian_k_sweep/gk_458910_17/checkpoints/best.pt}"
VORONOI_CHECKPOINT="${VORONOI_CHECKPOINT:-$ROOT/outputs/stage2_voronoi_refinement_sweep/vref_462246_28/checkpoints/best.pt}"

RUN_ID="${SLURM_JOB_ID:-local}_$(date +%Y%m%d_%H%M%S)"
OUTPUT="${OUTPUT_DIR:-$ROOT/outputs/oracle_merge/$RUN_ID}"

export PATH="$ENV/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export CUDA_HOME="$ENV"
export FOAMSURF_DRIVER_LIBRARY_DIRECTORY="$DRIVER"
export LD_LIBRARY_PATH="$DRIVER:$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p "$ROOT/slurm_logs"
cd "$ROOT"

test -s "$GAUSSIAN_CHECKPOINT"
test -s "$VORONOI_CHECKPOINT"

if [[ ! -s "$BACKEND/PYOPTIXFLARERENDERER.so" || ! -s "$BACKEND/shaders.cu.ptx" ]]; then
  echo "K=64 OptiX backend is missing; building it now."
  HIT_BUFFER_SIZE=64 \
    OPTIX_OUTPUT_DIRECTORY="$BACKEND" \
    CONDA_ENV_PATH="$ENV" \
    bash renderer/build_optix.sh
fi

echo "node=$(hostname) K=64 backend=$BACKEND output=$OUTPUT"
echo "gaussian_checkpoint=$GAUSSIAN_CHECKPOINT"
echo "voronoi_checkpoint=$VORONOI_CHECKPOINT"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"$ENV/bin/python" -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); import foamsurf_topology; print('foamsurf_topology import OK')"

"$ENV/bin/python" evaluate_oracle_merge.py \
  --config configs/baseline_best_2d_gaussians.yaml \
  --gaussian-checkpoint "$GAUSSIAN_CHECKPOINT" \
  --voronoi-checkpoint "$VORONOI_CHECKPOINT" \
  --output-dir "$OUTPUT" \
  --max-eval-views "${MAX_EVAL_VIEWS:--1}" \
  --image-views "${IMAGE_VIEWS:-4}" \
  --reproduction-tolerance "${REPRODUCTION_TOLERANCE:-0.0005}" \
  --wandb-mode "${WANDB_MODE:-online}" \
  --wandb-name "${WANDB_NAME:-oracle_merge_$RUN_ID}"

test -f "$OUTPUT/config.yaml"
test -f "$OUTPUT/oracle_manifest.json"
test -f "$OUTPUT/metrics/eval.csv"
test -f "$OUTPUT/summary.json"
