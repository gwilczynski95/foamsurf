#!/usr/bin/env bash
#SBATCH --job-name=fs_joint_adaptation
#SBATCH --output=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/joint_adaptation_%A_%a.out
#SBATCH --error=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/joint_adaptation_%A_%a.err
#SBATCH --partition=rtx4090_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --qos=batch
#SBATCH --exclude=c22,c11
#SBATCH --time=24:00:00
#SBATCH --array=0-6%3
set -Eeuo pipefail
: "${JOINT_ADAPTATION_TEST_CERTIFICATE:?Set JOINT_ADAPTATION_TEST_CERTIFICATE to the passed joint-adaptation test certificate}"
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
export TORCH_EXTENSIONS_DIR="$ROOT/outputs/joint_adaptation_build_${SLURM_JOB_ID}"
cd "$ROOT"
mkdir -p "$TORCH_EXTENSIONS_DIR"
python scripts/joint_adaptation_gate.py check --certificate "$JOINT_ADAPTATION_TEST_CERTIFICATE"
export WANDB_RUN_ID="ja-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
export WANDB_RUN_GROUP="joint-adaptation-${SLURM_ARRAY_JOB_ID}"
python scripts/train_joint_adaptation.py --config "${SWEEP_CONFIG:-$ROOT/configs/joint_adaptation_sweep.yaml}" --variant "${SLURM_ARRAY_TASK_ID}" --output-dir "$ROOT/outputs/joint_adaptation/${SLURM_ARRAY_JOB_ID}/run_${SLURM_ARRAY_TASK_ID}"
