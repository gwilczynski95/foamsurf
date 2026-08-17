#!/usr/bin/env bash
#SBATCH --job-name=fs_b1_op10k
#SBATCH --output=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/b1_op10k_%A_%a.out
#SBATCH --error=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/b1_op10k_%A_%a.err
#SBATCH --array=0-9%10
#SBATCH --partition=rtx4090_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --qos=batch
#SBATCH --exclude=c22,c11
#SBATCH --time=16:00:00

set -Eeuo pipefail

PROJECT_DIRECTORY="/shared/results/gmwilczn/foamsurf/foamsurf"
CONDA_ENVIRONMENT="/shared/results/gmwilczn/conda/envs/foamsurf"
PYTHON="$CONDA_ENVIRONMENT/bin/python"
CONFIGURATION="$PROJECT_DIRECTORY/configs/b1_lego_layers.yaml"
DRIVER_DIRECTORY="/shared/results/gmwilczn/drivers/NVIDIA-Linux-x86_64-535.183.01"
WANDB_MODE="${WANDB_MODE:-online}"
SWEEP_ITERATIONS="${SWEEP_ITERATIONS:-10000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"

THRESHOLDS=("0.003" "0.0035" "0.004" "0.0045" "0.005")
THRESHOLD_LABELS=("t003" "t0035" "t004" "t0045" "t005")
SEEDS=("42" "1337")

THRESHOLD_INDEX=$((SLURM_ARRAY_TASK_ID / 2))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % 2))
MIN_OPACITY="${THRESHOLDS[$THRESHOLD_INDEX]}"
THRESHOLD_LABEL="${THRESHOLD_LABELS[$THRESHOLD_INDEX]}"
SEED="${SEEDS[$SEED_INDEX]}"
LABEL="${THRESHOLD_LABEL}_s0500_seed${SEED}"

RUN_IDENTIFIER="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
OUTPUT_DIRECTORY="${OUTPUT_ROOT:-$PROJECT_DIRECTORY/outputs/b1-opacity-thresholds-10k}/${LABEL}_${RUN_IDENTIFIER}"
WANDB_NAME="b1-op10k-${LABEL}-${RUN_IDENTIFIER}"

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

echo "FoamSurf B1 opacity-threshold sweep (10k)"
echo "array_job_id=$SLURM_ARRAY_JOB_ID"
echo "array_task_id=$SLURM_ARRAY_TASK_ID"
echo "node=${SLURMD_NODENAME:-unknown}"
echo "label=$LABEL"
echo "seed=$SEED"
echo "min_opacity=$MIN_OPACITY"
echo "opacity_prune_from=500"
echo "iterations=$SWEEP_ITERATIONS"
echo "eval_interval=$EVAL_INTERVAL"
echo "output=$OUTPUT_DIRECTORY"
echo "wandb_name=$WANDB_NAME"
nvidia-smi

"$PYTHON" train_b0.py \
    --config "$CONFIGURATION" \
    --output-dir "$OUTPUT_DIRECTORY" \
    --iterations "$SWEEP_ITERATIONS" \
    --seed "$SEED" \
    --eval-interval "$EVAL_INTERVAL" \
    --experiment-name "b1-op10k-$LABEL" \
    --wandb-name "$WANDB_NAME" \
    --wandb-mode "$WANDB_MODE" \
    --min-opacity "$MIN_OPACITY" \
    --opacity-prune-from 500

test -f "$OUTPUT_DIRECTORY/checkpoints/last.pt"
test -f "$OUTPUT_DIRECTORY/checkpoints/best.pt"
test -f "$OUTPUT_DIRECTORY/metrics/train.csv"
test -f "$OUTPUT_DIRECTORY/metrics/eval.csv"
test -f "$OUTPUT_DIRECTORY/summary.json"
"$PYTHON" -c 'import json,sys; s=json.load(open(sys.argv[1])); assert int(s["final_iteration"]) == int(sys.argv[2]), s; assert 0 < int(s["gaussian_count"]) <= 200000, s' "$OUTPUT_DIRECTORY/summary.json" "$SWEEP_ITERATIONS"

echo "FoamSurf B1 10k sweep task completed: $OUTPUT_DIRECTORY"
