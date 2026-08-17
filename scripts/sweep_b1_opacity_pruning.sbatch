#!/usr/bin/env bash
#SBATCH --job-name=fs_b1_opacity
#SBATCH --output=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/b1_opacity_%A_%a.out
#SBATCH --error=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/b1_opacity_%A_%a.err
#SBATCH --array=0-6%7
#SBATCH --partition=rtx4090_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --qos=batch
#SBATCH --exclude=c22,c11
#SBATCH --time=08:00:00

set -Eeuo pipefail

PROJECT_DIRECTORY="/shared/results/gmwilczn/foamsurf/foamsurf"
CONDA_ENVIRONMENT="/shared/results/gmwilczn/conda/envs/foamsurf"
PYTHON="$CONDA_ENVIRONMENT/bin/python"
CONFIGURATION="$PROJECT_DIRECTORY/configs/b1_lego_layers.yaml"
DRIVER_DIRECTORY="/shared/results/gmwilczn/drivers/NVIDIA-Linux-x86_64-535.183.01"
WANDB_MODE="${WANDB_MODE:-online}"
SWEEP_ITERATIONS="${SWEEP_ITERATIONS:-5000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"

case "$SLURM_ARRAY_TASK_ID" in
    0) LABEL="baseline_t005_s0500"; MIN_OPACITY="0.005";  PRUNE_FROM="500" ;;
    1) LABEL="t0025_s0500";        MIN_OPACITY="0.0025"; PRUNE_FROM="500" ;;
    2) LABEL="t001_s0500";         MIN_OPACITY="0.001";  PRUNE_FROM="500" ;;
    3) LABEL="t005_s3000";         MIN_OPACITY="0.005";  PRUNE_FROM="3000" ;;
    4) LABEL="t0025_s3000";        MIN_OPACITY="0.0025"; PRUNE_FROM="3000" ;;
    5) LABEL="t001_s3000";         MIN_OPACITY="0.001";  PRUNE_FROM="3000" ;;
    6) LABEL="no_opacity_prune";   MIN_OPACITY="0.0";    PRUNE_FROM="500" ;;
    *) echo "Unknown array task: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

RUN_IDENTIFIER="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
OUTPUT_DIRECTORY="${OUTPUT_ROOT:-$PROJECT_DIRECTORY/outputs/b1-opacity-sweep}/${LABEL}_${RUN_IDENTIFIER}"
WANDB_NAME="b1-opacity-${LABEL}-${RUN_IDENTIFIER}"

export PATH="$CONDA_ENVIRONMENT/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export CUDA_HOME="$CONDA_ENVIRONMENT"
export FOAMSURF_DRIVER_LIBRARY_DIRECTORY="$DRIVER_DIRECTORY"
export LD_LIBRARY_PATH="$DRIVER_DIRECTORY:$CONDA_ENVIRONMENT/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

if [[ ! -x "$PYTHON" ]]; then
    echo "Python not found: $PYTHON" >&2
    exit 1
fi
if [[ ! -f "$PROJECT_DIRECTORY/renderer/foamsurf/_C.so" ]]; then
    echo "Renderer is not built. Run ./build.sh before sbatch." >&2
    exit 1
fi
if [[ ! -f "$PROJECT_DIRECTORY/renderer/foamsurf/_optix/PYOPTIXFLARERENDERER.so" ]]; then
    echo "OptiX renderer is not built. Run ./build.sh before sbatch." >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIRECTORY" "$PROJECT_DIRECTORY/slurm_logs"
cd "$PROJECT_DIRECTORY"

echo "FoamSurf B1 opacity-pruning sweep"
echo "array_job_id=$SLURM_ARRAY_JOB_ID"
echo "array_task_id=$SLURM_ARRAY_TASK_ID"
echo "node=${SLURMD_NODENAME:-unknown}"
echo "label=$LABEL"
echo "min_opacity=$MIN_OPACITY"
echo "opacity_prune_from=$PRUNE_FROM"
echo "iterations=$SWEEP_ITERATIONS"
echo "eval_interval=$EVAL_INTERVAL"
echo "output=$OUTPUT_DIRECTORY"
echo "wandb_name=$WANDB_NAME"
nvidia-smi

"$PYTHON" train_b0.py \
    --config "$CONFIGURATION" \
    --output-dir "$OUTPUT_DIRECTORY" \
    --iterations "$SWEEP_ITERATIONS" \
    --eval-interval "$EVAL_INTERVAL" \
    --experiment-name "b1-opacity-$LABEL" \
    --wandb-name "$WANDB_NAME" \
    --wandb-mode "$WANDB_MODE" \
    --min-opacity "$MIN_OPACITY" \
    --opacity-prune-from "$PRUNE_FROM"

test -f "$OUTPUT_DIRECTORY/checkpoints/last.pt"
test -f "$OUTPUT_DIRECTORY/checkpoints/best.pt"
test -f "$OUTPUT_DIRECTORY/metrics/train.csv"
test -f "$OUTPUT_DIRECTORY/metrics/eval.csv"
test -f "$OUTPUT_DIRECTORY/summary.json"

echo "FoamSurf B1 sweep task completed: $OUTPUT_DIRECTORY"
