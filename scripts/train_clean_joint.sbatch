#!/usr/bin/env bash
#SBATCH --job-name=fs_clean_joint
#SBATCH --output=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/clean_joint_%j.out
#SBATCH --error=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/clean_joint_%j.err
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
RUN_ID="${SLURM_JOB_ID:-local}_$(date +%Y%m%d_%H%M%S)"
SEED="${SEED:-0}"
ITERATIONS="${ITERATIONS:-30000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
OUTPUT="${OUTPUT_DIR:-$ROOT/outputs/baseline_clean_joint/seed${SEED}_$RUN_ID}"

export PATH="$ENV/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export CUDA_HOME="$ENV"
export FOAMSURF_DRIVER_LIBRARY_DIRECTORY="$DRIVER"
export LD_LIBRARY_PATH="$DRIVER:$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p "$ROOT/slurm_logs"
cd "$ROOT"

if [[ ! -s "$BACKEND/PYOPTIXFLARERENDERER.so" || ! -s "$BACKEND/shaders.cu.ptx" ]]; then
  echo "K=64 OptiX backend is missing; building it now."
  HIT_BUFFER_SIZE=64 \
    OPTIX_OUTPUT_DIRECTORY="$BACKEND" \
    CONDA_ENV_PATH="$ENV" \
    bash renderer/build_optix.sh
fi

echo "FoamSurf Stage 3 clean joint baseline"
echo "node=$(hostname) seed=$SEED iterations=$ITERATIONS output=$OUTPUT"
echo "training_target=lego_both layer_targets=evaluation_only"
echo "initialization=fresh oracle_checkpoint_import=disabled"
echo "loss=full-image-L1-plus-0.2-DSSIM disentanglement_losses=disabled"
echo "gaussians=100000-adaptive-unlimited voronoi=2048-adaptive-to-8192"
echo "renderer=baseline_best_2d_gaussians K=64 backend=$BACKEND"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"$ENV/bin/python" -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); import foamsurf_topology; print('foamsurf_topology import OK')"

"$ENV/bin/python" train_b0.py \
  --config configs/baseline_clean_joint.yaml \
  --output-dir "$OUTPUT" \
  --iterations "$ITERATIONS" \
  --eval-interval "$EVAL_INTERVAL" \
  --seed "$SEED" \
  --experiment-name clean-joint-baseline \
  --wandb-mode "${WANDB_MODE:-online}" \
  --wandb-name "${WANDB_NAME:-clean_joint_seed${SEED}_$RUN_ID}"

test -f "$OUTPUT/config.yaml"
test -f "$OUTPUT/checkpoints/last.pt"
test -f "$OUTPUT/checkpoints/best.pt"
test -f "$OUTPUT/metrics/train.csv"
test -f "$OUTPUT/metrics/eval.csv"
test -f "$OUTPUT/summary.json"
"$ENV/bin/python" -c 'import json,sys; s=json.load(open(sys.argv[1])); assert int(s["final_iteration"]) == int(sys.argv[2]), s; assert s["representation_mode"] == "hybrid", s; assert int(s["gaussian_count"]) > 0, s; assert int(s["voronoi_count"]) > 0, s' "$OUTPUT/summary.json" "$ITERATIONS"

echo "Clean joint baseline completed: $OUTPUT"
