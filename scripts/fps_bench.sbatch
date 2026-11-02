#!/usr/bin/env bash
#SBATCH --job-name=fs_fps
#SBATCH --output=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/fps_%j.out
#SBATCH --error=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/fps_%j.err
#SBATCH --partition=rtx4090_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --qos=batch
#SBATCH --exclude=c22,c11
#SBATCH --time=02:00:00
# Rendering speed (scripts/fps_bench.py) of trained runs on an RTX 4090: every test view, frame = colours +
# render + PPISP and the evaluator's render-only number, once per OptiX K-hit buffer in FS_KS (default 64).
#   FS_RUNS="outputs/seathru/551590/*" FS_KS="64 32" sbatch scripts/fps_bench.sbatch
# FS_THRESHOLDS="0.01 0.0001": also once per evaluation transmittance threshold (default: the run's own).
set -Eeuo pipefail
ROOT=/shared/results/gmwilczn/foamsurf/foamsurf
FS_ENV=/shared/results/gmwilczn/conda/envs/foamsurf
DRIVER=/shared/results/gmwilczn/drivers/NVIDIA-Linux-x86_64-535.183.01
export PATH="$FS_ENV/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export CUDA_HOME="$FS_ENV"
export FOAMSURF_DRIVER_LIBRARY_DIRECTORY="$DRIVER"
export LD_LIBRARY_PATH="$DRIVER:$FS_ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1 TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=2 OMP_NUM_THREADS=4
export TORCH_EXTENSIONS_DIR="$ROOT/outputs/build_fps_${SLURM_JOB_ID}"
mkdir -p "$TORCH_EXTENSIONS_DIR"
cd "$ROOT"
# One process per run and buffer: every Trainer creates its own OptiX context.
for k in ${FS_KS:-64}; do
  for t in ${FS_THRESHOLDS:-run}; do
    for run in ${FS_RUNS:?set FS_RUNS}; do
      threshold=(); [[ "$t" != run ]] && threshold=(--threshold "$t")
      python scripts/fps_bench.py "$run" --hit-buffer "$k" "${threshold[@]}" \
        --json "outputs/fps_bench/fps_${SLURM_JOB_ID}_k${k}_t${t}_$(basename "$run").json"
    done
  done
done
rm -rf "$TORCH_EXTENSIONS_DIR" outputs/fps_bench/_trainer
