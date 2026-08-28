#!/usr/bin/env bash
#SBATCH --job-name=fs_stage1_g
#SBATCH --output=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/stage1_g_%A_%a.out
#SBATCH --error=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/stage1_g_%A_%a.err
#SBATCH --array=0-371%8
#SBATCH --partition=rtx4090_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --qos=batch
#SBATCH --exclude=c22,c11
#SBATCH --time=24:00:00
set -Eeuo pipefail
ROOT=/shared/results/gmwilczn/foamsurf/foamsurf; ENV=/shared/results/gmwilczn/conda/envs/foamsurf; DRIVER=/shared/results/gmwilczn/drivers/NVIDIA-Linux-x86_64-535.183.01
TASK=$SLURM_ARRAY_TASK_ID; RUN="g_${SLURM_ARRAY_JOB_ID}_${TASK}"; OUTPUT="$ROOT/outputs/stage1_gaussian_sweep/$RUN"
export PATH="$ENV/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" CUDA_HOME="$ENV" FOAMSURF_DRIVER_LIBRARY_DIRECTORY="$DRIVER" LD_LIBRARY_PATH="$DRIVER:$ENV/lib:${LD_LIBRARY_PATH:-}" PYTHONUNBUFFERED=1
mkdir -p "$ROOT/slurm_logs"; cd "$ROOT"
echo "node=$(hostname) task=$TASK"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
"$ENV/bin/python" -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); import foamsurf_topology; print('foamsurf_topology import OK')"
"$ENV/bin/python" scripts/run_standalone_sweep.py --phase gaussian \
  --config configs/stage1_gaussian_flare_control.yaml --index "$TASK" \
  --output-dir "$OUTPUT" --wandb-name "$RUN" \
  --wandb-mode "${WANDB_MODE:-online}"
test -f "$OUTPUT/summary.json"; test -f "$OUTPUT/checkpoints/best.pt"
