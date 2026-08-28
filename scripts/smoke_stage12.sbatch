#!/usr/bin/env bash
#SBATCH --job-name=fs_stage12_smoke
#SBATCH --output=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/stage12_smoke_%A_%a.out
#SBATCH --error=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/stage12_smoke_%A_%a.err
#SBATCH --array=0-1%1
#SBATCH --partition=rtx4090_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --qos=batch
#SBATCH --time=02:00:00
set -Eeuo pipefail
ROOT=/shared/results/gmwilczn/foamsurf/foamsurf
ENV=/shared/results/gmwilczn/conda/envs/foamsurf
DRIVER=/shared/results/gmwilczn/drivers/NVIDIA-Linux-x86_64-535.183.01
PHASES=(gaussian voronoi)
PHASE=${PHASES[$SLURM_ARRAY_TASK_ID]}
OUTPUT="$ROOT/outputs/stage12_smoke/${PHASE}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export PATH="$ENV/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export CUDA_HOME="$ENV"
export FOAMSURF_DRIVER_LIBRARY_DIRECTORY="$DRIVER"
export LD_LIBRARY_PATH="$DRIVER:$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
mkdir -p "$ROOT/slurm_logs" "$OUTPUT"
rmdir "$OUTPUT"
cd "$ROOT"
"$ENV/bin/python" scripts/run_stage12_smoke.py --phase "$PHASE" --output-dir "$OUTPUT"
test -f "$OUTPUT/checkpoints/last.pt"
test -f "$OUTPUT/checkpoints/best.pt"
test -f "$OUTPUT/metrics/train.csv"
test -f "$OUTPUT/metrics/eval.csv"
