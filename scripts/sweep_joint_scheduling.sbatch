#!/usr/bin/env bash
#SBATCH --job-name=fs_joint_sched
#SBATCH --output=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/joint_sched_%A_%a.out
#SBATCH --error=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/joint_sched_%A_%a.err
#SBATCH --array=0-3%4
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
RUN="joint_sched_${SLURM_ARRAY_JOB_ID}_${TASK}"
OUTPUT="${OUTPUT_DIR:-$ROOT/outputs/joint_scheduling/$RUN}"

export PATH="$ENV/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export CUDA_HOME="$ENV"
export FOAMSURF_DRIVER_LIBRARY_DIRECTORY="$DRIVER"
export LD_LIBRARY_PATH="$DRIVER:$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p "$ROOT/slurm_logs"
cd "$ROOT"

if [[ ! -s "$BACKEND/PYOPTIXFLARERENDERER.so" || ! -s "$BACKEND/shaders.cu.ptx" ]]; then
  HIT_BUFFER_SIZE=64 \
    OPTIX_OUTPUT_DIRECTORY="$BACKEND" \
    CONDA_ENV_PATH="$ENV" \
    bash renderer/build_optix.sh
fi

echo "Joint scheduling test task=$TASK; renderer K=64; seed=0"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

"$ENV/bin/python" scripts/run_joint_scheduling_sweep.py \
  --config configs/baseline_clean_joint.yaml \
  --index "$TASK" \
  --output-dir "$OUTPUT" \
  --wandb-name "$RUN" \
  --wandb-mode "${WANDB_MODE:-online}"

test -f "$OUTPUT/summary.json"
test -f "$OUTPUT/checkpoints/best.pt"
