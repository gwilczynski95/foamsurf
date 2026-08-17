#!/usr/bin/env bash
#SBATCH --job-name=fs_det_train
#SBATCH --output=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/determinism_training_%A_%a.out
#SBATCH --error=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/determinism_training_%A_%a.err
#SBATCH --array=0-5%1
#SBATCH --partition=rtx4090_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --qos=batch
#SBATCH --exclude=c22,c11
#SBATCH --time=06:00:00

set -Eeuo pipefail

PROJECT_DIRECTORY="/shared/results/gmwilczn/foamsurf/foamsurf"
CONDA_ENVIRONMENT="/shared/results/gmwilczn/conda/envs/foamsurf"
PYTHON="$CONDA_ENVIRONMENT/bin/python"
CONFIGURATION="$PROJECT_DIRECTORY/configs/b1_lego_layers.yaml"
DRIVER_DIRECTORY="/shared/results/gmwilczn/drivers/NVIDIA-Linux-x86_64-535.183.01"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-foamsurf_determinism_audit}"
SEED="${SEED:-42}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
NODE_NAME="${SLURMD_NODENAME:-manual}"

if (( TASK_ID < 3 )); then
    MODE="off"
    REPLICATE="$TASK_ID"
    ITERATIONS=1000
else
    MODE="on"
    REPLICATE="$((TASK_ID - 3))"
    ITERATIONS=12000
fi

RUN_IDENTIFIER="${SLURM_ARRAY_JOB_ID:-manual}_${TASK_ID}"
OUTPUT_DIRECTORY="${OUTPUT_ROOT:-$PROJECT_DIRECTORY/outputs/foamsurf_determinism_audit}/training_${MODE}_${NODE_NAME}_${RUN_IDENTIFIER}"
WANDB_NAME="det-training-${MODE}-${NODE_NAME}-r${REPLICATE}-${RUN_IDENTIFIER}"

export PATH="$CONDA_ENVIRONMENT/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export CUDA_HOME="$CONDA_ENVIRONMENT"
export FOAMSURF_DRIVER_LIBRARY_DIRECTORY="$DRIVER_DIRECTORY"
export LD_LIBRARY_PATH="$DRIVER_DIRECTORY:$CONDA_ENVIRONMENT/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

mkdir -p "$OUTPUT_DIRECTORY" "$PROJECT_DIRECTORY/slurm_logs"
cd "$PROJECT_DIRECTORY"

echo "FoamSurf replicated training determinism audit"
echo "node=$NODE_NAME"
echo "mode=$MODE"
echo "replicate=$REPLICATE"
echo "iterations=$ITERATIONS"
echo "seed=$SEED"
echo "wandb_project=$WANDB_PROJECT"
echo "output=$OUTPUT_DIRECTORY"
nvidia-smi

"$PYTHON" scripts/run_determinism_training.py \
    --config "$CONFIGURATION" \
    --output-dir "$OUTPUT_DIRECTORY" \
    --wandb-name "$WANDB_NAME" \
    --wandb-mode "$WANDB_MODE" \
    --wandb-project "$WANDB_PROJECT" \
    --seed "$SEED" \
    --iterations "$ITERATIONS" \
    --adaptation "$MODE"

test -f "$OUTPUT_DIRECTORY/summary.json"
test -f "$OUTPUT_DIRECTORY/audit_summary.json"
test -f "$OUTPUT_DIRECTORY/provenance.json"
echo "Training audit finished: $OUTPUT_DIRECTORY/audit_summary.json"
