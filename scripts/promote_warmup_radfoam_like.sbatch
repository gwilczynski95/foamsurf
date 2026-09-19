#!/usr/bin/env bash
#SBATCH --job-name=fs_warm_long
#SBATCH --output=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/warmup_long_%A_%a.out
#SBATCH --error=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/warmup_long_%A_%a.err
#SBATCH --partition=rtx4090_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --qos=batch
#SBATCH --exclude=c22,c11
#SBATCH --time=24:00:00
#SBATCH --array=0-14%5
set -Eeuo pipefail
: "${WARMUP_TEST_CERTIFICATE:?Passed test certificate required}"
: "${WARMUP_SELECTED_IDS:?Provide five selected IDs separated by spaces}"
read -r -a SELECTED <<< "$WARMUP_SELECTED_IDS"
test "${#SELECTED[@]}" -eq 5
INDEX="${SELECTED[$((SLURM_ARRAY_TASK_ID / 3))]}"
SEED=$((SLURM_ARRAY_TASK_ID % 3))
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
export TORCH_EXTENSIONS_DIR="$ROOT/outputs/warmup_long_build_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
cd "$ROOT"
python scripts/warmup_gate.py check --certificate "$WARMUP_TEST_CERTIFICATE"
mkdir -p "$TORCH_EXTENSIONS_DIR"
OUTPUT="${OUTPUT_ROOT:-$ROOT/outputs/warmup_promoted/${SLURM_ARRAY_JOB_ID}}/run_${INDEX}_seed_${SEED}"
python scripts/run_warmup_radfoam_like.py --index "$INDEX" --seed "$SEED" --iterations 20000 \
  --output-dir "$OUTPUT" --certificate "$WARMUP_TEST_CERTIFICATE" --wandb-mode "${WANDB_MODE:-offline}"
python scripts/analyze_warmup_radfoam_like.py --run "$OUTPUT"
