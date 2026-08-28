#!/usr/bin/env bash
#SBATCH --job-name=fs_vcap
#SBATCH --output=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/vcap_%A_%a.out
#SBATCH --error=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/vcap_%A_%a.err
#SBATCH --array=0-15%8
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
RUN="vcap_${SLURM_ARRAY_JOB_ID}_${TASK}"
OUTPUT="$ROOT/outputs/stage2_voronoi_capacity_sweep/$RUN"

export PATH="$ENV/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export CUDA_HOME="$ENV"
export FOAMSURF_DRIVER_LIBRARY_DIRECTORY="$DRIVER"
export LD_LIBRARY_PATH="$DRIVER:$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p "$ROOT/slurm_logs"
cd "$ROOT"

"$ENV/bin/python" scripts/run_voronoi_capacity_sweep.py \
  --config configs/stage2_voronoi_capacity.yaml \
  --index "$TASK" \
  --output-dir "$OUTPUT" \
  --wandb-name "$RUN" \
  --wandb-mode "${WANDB_MODE:-online}"

test -f "$OUTPUT/summary.json"
test -f "$OUTPUT/checkpoints/best.pt"
