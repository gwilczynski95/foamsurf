#!/usr/bin/env bash
#SBATCH --job-name=fs_end_to_end
#SBATCH --output=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/end_to_end_%j.out
#SBATCH --error=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/end_to_end_%j.err
#SBATCH --partition=rtx4090_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --qos=batch
#SBATCH --exclude=c22,c11
#SBATCH --time=24:00:00
set -Eeuo pipefail
: "${END_TO_END_TEST_CERTIFICATE:?Set END_TO_END_TEST_CERTIFICATE to the passed end-to-end test certificate}"
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
export TORCH_EXTENSIONS_DIR="$ROOT/outputs/end_to_end_build_${SLURM_JOB_ID}"
cd "$ROOT"
mkdir -p "$TORCH_EXTENSIONS_DIR"
python scripts/end_to_end_gate.py check --certificate "$END_TO_END_TEST_CERTIFICATE"
ARGS=(--config "${E2E_CONFIG:-$ROOT/configs/end_to_end.yaml}" --output-dir "${OUTPUT_DIR:-$ROOT/outputs/end_to_end/${SLURM_JOB_ID}}")
if [[ "${RESUME:-0}" == 1 ]]; then ARGS+=(--resume); fi
python scripts/train_end_to_end.py "${ARGS[@]}"
