#!/usr/bin/env bash
#SBATCH --job-name=fs_sigmoid
#SBATCH --output=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/sigmoid_%A_%a.out
#SBATCH --error=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/sigmoid_%A_%a.err
#SBATCH --partition=rtx4090_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --qos=batch
#SBATCH --exclude=c22,c11
#SBATCH --time=24:00:00
#SBATCH --array=0-2%3
set -Eeuo pipefail

ROOT=/shared/results/gmwilczn/foamsurf/foamsurf
FS_ENV=/shared/results/gmwilczn/conda/envs/foamsurf
DRIVER=/shared/results/gmwilczn/drivers/NVIDIA-Linux-x86_64-535.183.01
export PATH="$FS_ENV/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export CUDA_HOME="$FS_ENV"
export FOAMSURF_DRIVER_LIBRARY_DIRECTORY="$DRIVER"
export LD_LIBRARY_PATH="$DRIVER:$FS_ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export TORCH_CUDA_ARCH_LIST=8.9
export MAX_JOBS=2
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

case "$SLURM_ARRAY_TASK_ID" in
  0) VARIANT=gaussian_sigmoid; GAUSSIAN=sigmoid; MEDIUM=dc ;;
  1) VARIANT=medium_sigmoid; GAUSSIAN=identity; MEDIUM=sigmoid ;;
  2) VARIANT=both_sigmoid; GAUSSIAN=sigmoid; MEDIUM=sigmoid ;;
  *) echo "Unexpected array index: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

cd "$ROOT"
mkdir -p slurm_logs
export TORCH_EXTENSIONS_DIR="$ROOT/outputs/build_sigmoid_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TORCH_EXTENSIONS_DIR"
RUN_ID="${FS_RUN_ID:-${SLURM_ARRAY_JOB_ID}}"
OUTPUT="${FS_OUTPUT_ROOT:-outputs/iui3_sigmoid}/${RUN_ID}/${VARIANT}"
ARGS=(
  --config configs/iui3_5k.yaml
  --output-dir "$OUTPUT"
  --gaussian-rgb-activation "$GAUSSIAN"
  --medium-color-activation "$MEDIUM"
  --wandb-project "${FS_WANDB_PROJECT:-foamsurf_iui3_sigmoid}"
)
if [[ "${FS_RESUME:-0}" == 1 ]]; then ARGS+=(--resume); fi
python train.py "${ARGS[@]}"
