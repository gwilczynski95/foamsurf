#!/usr/bin/env bash
#SBATCH --job-name=fs_stage2_v
#SBATCH --output=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/stage2_v_%A_%a.out
#SBATCH --error=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/stage2_v_%A_%a.err
#SBATCH --array=0-69%8
#SBATCH --partition=rtx4090_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --qos=batch
#SBATCH --exclude=c22,c11
#SBATCH --time=24:00:00
set -Eeuo pipefail
ROOT=/shared/results/gmwilczn/foamsurf/foamsurf; ENV=/shared/results/gmwilczn/conda/envs/foamsurf; DRIVER=/shared/results/gmwilczn/drivers/NVIDIA-Linux-x86_64-535.183.01
TASK=$SLURM_ARRAY_TASK_ID; RUN="v_${SLURM_ARRAY_JOB_ID}_${TASK}"; OUTPUT="$ROOT/outputs/stage2_voronoi_sweep/$RUN"
export PATH="$ENV/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" CUDA_HOME="$ENV" FOAMSURF_DRIVER_LIBRARY_DIRECTORY="$DRIVER" LD_LIBRARY_PATH="$DRIVER:$ENV/lib:${LD_LIBRARY_PATH:-}" PYTHONUNBUFFERED=1
mkdir -p "$ROOT/slurm_logs"; cd "$ROOT"
"$ENV/bin/python" scripts/run_standalone_sweep.py --phase voronoi --config configs/stage2_voronoi_only.yaml --index "$TASK" --output-dir "$OUTPUT" --wandb-name "$RUN" --wandb-mode "${WANDB_MODE:-online}"
test -f "$OUTPUT/summary.json"; test -f "$OUTPUT/checkpoints/best.pt"
