#!/usr/bin/env bash
#SBATCH --job-name=fs_b1_full_unlim
#SBATCH --output=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/b1_full_unlim_%j.out
#SBATCH --error=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/b1_full_unlim_%j.err
#SBATCH --partition=rtx4090_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
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
ITERATIONS="${ITERATIONS:-10000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
SEED="${SEED:-42}"
MIN_OPACITY="${MIN_OPACITY:-0.005}"

RUN_IDENTIFIER="${SLURM_JOB_ID:-manual}"
LABEL="full800-unlimited-t005-seed${SEED}"
OUTPUT_DIRECTORY="${OUTPUT_ROOT:-$PROJECT_DIRECTORY/outputs/b1-fullimage-unlimited-10k/${LABEL}_${RUN_IDENTIFIER}}"
WANDB_NAME="b1-full800-unlimited-t005-seed${SEED}-${RUN_IDENTIFIER}"

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

echo "FoamSurf B1 full-image unlimited baseline"
echo "job_id=${SLURM_JOB_ID:-manual}"
echo "node=${SLURMD_NODENAME:-unknown}"
echo "seed=$SEED"
echo "iterations=$ITERATIONS"
echo "image_batch=800x800=640000 unique rays"
echo "camera_sampling=shuffled_without_replacement"
echo "max_points=none"
echo "min_opacity=$MIN_OPACITY"
echo "output=$OUTPUT_DIRECTORY"
echo "wandb_name=$WANDB_NAME"
nvidia-smi

"$PYTHON" train_b0.py \
    --config "$CONFIGURATION" \
    --output-dir "$OUTPUT_DIRECTORY" \
    --iterations "$ITERATIONS" \
    --seed "$SEED" \
    --eval-interval "$EVAL_INTERVAL" \
    --experiment-name "b1-full800-unlimited-t005" \
    --wandb-name "$WANDB_NAME" \
    --wandb-mode "$WANDB_MODE" \
    --min-opacity "$MIN_OPACITY" \
    --opacity-prune-from 500 \
    --rays-per-batch 640000 \
    --full-image-batch \
    --cameras-without-replacement \
    --no-max-points

test -f "$OUTPUT_DIRECTORY/checkpoints/last.pt"
test -f "$OUTPUT_DIRECTORY/checkpoints/best.pt"
test -f "$OUTPUT_DIRECTORY/metrics/train.csv"
test -f "$OUTPUT_DIRECTORY/metrics/eval.csv"
test -f "$OUTPUT_DIRECTORY/summary.json"
"$PYTHON" -c 'import json,sys; s=json.load(open(sys.argv[1])); assert int(s["final_iteration"]) == int(sys.argv[2]), s; assert int(s["gaussian_count"]) > 0, s' "$OUTPUT_DIRECTORY/summary.json" "$ITERATIONS"

echo "FoamSurf B1 full-image unlimited baseline completed: $OUTPUT_DIRECTORY"
