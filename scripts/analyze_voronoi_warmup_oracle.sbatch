#!/usr/bin/env bash
#SBATCH --job-name=fs_vw_or
#SBATCH --output=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/voronoi_warmup_oracle_%j.out
#SBATCH --error=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/voronoi_warmup_oracle_%j.err
#SBATCH --partition=rtx4090_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --qos=batch
#SBATCH --exclude=c22,c11
#SBATCH --time=01:00:00

set -Eeuo pipefail
ROOT=/shared/results/gmwilczn/foamsurf/foamsurf
ENV=/shared/results/gmwilczn/conda/envs/foamsurf
DRIVER=/shared/results/gmwilczn/drivers/NVIDIA-Linux-x86_64-535.183.01
RUN=${RUN:-$ROOT/outputs/voronoi_prior_warmup/voronoi_prior_warmup_503195}
export PATH="$ENV/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export CUDA_HOME="$ENV"
export FOAMSURF_DRIVER_LIBRARY_DIRECTORY="$DRIVER"
export LD_LIBRARY_PATH="$DRIVER:$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
cd "$ROOT"
"$ENV/bin/python" scripts/analyze_voronoi_warmup_oracle.py \
  --run "$RUN" --iterations 4000 5000 --image-views 4
test -s "$RUN/analysis/oracle_analysis.json"
test -s "$RUN/analysis/iteration_004000/oracle/view_00_separation.png"
test -s "$RUN/analysis/iteration_005000/oracle/view_00_separation.png"
