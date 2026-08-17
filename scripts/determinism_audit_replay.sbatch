#!/usr/bin/env bash
#SBATCH --job-name=fs_det_replay
#SBATCH --output=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/determinism_replay_%A_%a.out
#SBATCH --error=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/determinism_replay_%A_%a.err
#SBATCH --array=0-2%1
#SBATCH --partition=rtx4090_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --qos=batch
#SBATCH --exclude=c22,c11
#SBATCH --time=02:00:00

set -Eeuo pipefail

PROJECT_DIRECTORY="/shared/results/gmwilczn/foamsurf/foamsurf"
CONDA_ENVIRONMENT="/shared/results/gmwilczn/conda/envs/foamsurf"
PYTHON="$CONDA_ENVIRONMENT/bin/python"
CONFIGURATION="$PROJECT_DIRECTORY/configs/b1_lego_layers.yaml"
DRIVER_DIRECTORY="/shared/results/gmwilczn/drivers/NVIDIA-Linux-x86_64-535.183.01"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-foamsurf_determinism_audit}"
SEED="${SEED:-42}"
REPLICATE="${SLURM_ARRAY_TASK_ID:-0}"
NODE_NAME="${SLURMD_NODENAME:-manual}"
RUN_IDENTIFIER="${SLURM_ARRAY_JOB_ID:-manual}_${REPLICATE}"
OUTPUT_DIRECTORY="${OUTPUT_ROOT:-$PROJECT_DIRECTORY/outputs/foamsurf_determinism_audit}/replay_${NODE_NAME}_${RUN_IDENTIFIER}"
WANDB_NAME="det-replay-${NODE_NAME}-${RUN_IDENTIFIER}"

export PATH="$CONDA_ENVIRONMENT/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export CUDA_HOME="$CONDA_ENVIRONMENT"
export FOAMSURF_DRIVER_LIBRARY_DIRECTORY="$DRIVER_DIRECTORY"
export LD_LIBRARY_PATH="$DRIVER_DIRECTORY:$CONDA_ENVIRONMENT/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

mkdir -p "$OUTPUT_DIRECTORY" "$PROJECT_DIRECTORY/slurm_logs"
cd "$PROJECT_DIRECTORY"

echo "FoamSurf determinism replay audit"
echo "node=$NODE_NAME"
echo "replicate=$REPLICATE"
echo "seed=$SEED"
echo "wandb_project=$WANDB_PROJECT"
echo "output=$OUTPUT_DIRECTORY"
nvidia-smi

"$PYTHON" scripts/run_determinism_replay.py \
    --config "$CONFIGURATION" \
    --output-dir "$OUTPUT_DIRECTORY" \
    --wandb-name "$WANDB_NAME" \
    --wandb-mode "$WANDB_MODE" \
    --wandb-project "$WANDB_PROJECT" \
    --seed "$SEED" \
    --replays 5

test -f "$OUTPUT_DIRECTORY/audit.json"
test -f "$OUTPUT_DIRECTORY/provenance.json"
echo "Replay audit finished: $OUTPUT_DIRECTORY/audit.json"
