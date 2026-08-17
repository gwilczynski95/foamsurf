#!/usr/bin/env bash
#SBATCH --job-name=fs_2dgs_b1
#SBATCH --output=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/2dgs_baseline_%j.out
#SBATCH --error=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/2dgs_baseline_%j.err
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
WANDB_PROJECT="${WANDB_PROJECT:-foamsurf_2dgs_baseline}"
SEED="${SEED:-42}"
ITERATIONS="${ITERATIONS:-30000}"
RUN_IDENTIFIER="${SLURM_JOB_ID:-manual}"
OUTPUT_DIRECTORY="${OUTPUT_ROOT:-$PROJECT_DIRECTORY/outputs/foamsurf_2dgs_baseline}/run_$RUN_IDENTIFIER"
WANDB_NAME="2dgs-baseline-$RUN_IDENTIFIER"

export PATH="$CONDA_ENVIRONMENT/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export CUDA_HOME="$CONDA_ENVIRONMENT"
export FOAMSURF_DRIVER_LIBRARY_DIRECTORY="$DRIVER_DIRECTORY"
export LD_LIBRARY_PATH="$DRIVER_DIRECTORY:$CONDA_ENVIRONMENT/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p "$OUTPUT_DIRECTORY" "$PROJECT_DIRECTORY/slurm_logs"
cd "$PROJECT_DIRECTORY"

echo "FoamSurf B1 official-2DGS behavior baseline"
echo "node=${SLURMD_NODENAME:-manual}"
echo "iterations=$ITERATIONS"
echo "seed=$SEED"
echo "wandb_project=$WANDB_PROJECT"
echo "output=$OUTPUT_DIRECTORY"
nvidia-smi

"$PYTHON" scripts/run_2dgs_baseline.py \
    --config "$CONFIGURATION" \
    --output-dir "$OUTPUT_DIRECTORY" \
    --wandb-name "$WANDB_NAME" \
    --wandb-mode "$WANDB_MODE" \
    --wandb-project "$WANDB_PROJECT" \
    --seed "$SEED" \
    --iterations "$ITERATIONS"

test -f "$OUTPUT_DIRECTORY/summary.json"
test -f "$OUTPUT_DIRECTORY/checkpoints/last.pt"
echo "2DGS baseline finished: $OUTPUT_DIRECTORY"
