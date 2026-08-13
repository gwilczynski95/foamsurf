#!/usr/bin/env bash
#SBATCH --job-name=foamsurf_b01_layers
#SBATCH --output=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/b01_layers_%j.out
#SBATCH --error=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/b01_layers_%j.err
#SBATCH --partition=rtx4090_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --qos=batch
#SBATCH --exclude=c22,c11
#SBATCH --time=24:00:00

set -Eeuo pipefail

PROJECT_DIRECTORY="/shared/results/gmwilczn/foamsurf/foamsurf"
CONDA_ENVIRONMENT="/shared/results/gmwilczn/conda/envs/foamsurf"
PYTHON="$CONDA_ENVIRONMENT/bin/python"
CONFIGURATION="$PROJECT_DIRECTORY/configs/b01_lego_layers.yaml"
DRIVER_DIRECTORY="/shared/results/gmwilczn/drivers/NVIDIA-Linux-x86_64-535.183.01"

OUTPUT_DIRECTORY="${OUTPUT_DIRECTORY:-$PROJECT_DIRECTORY/outputs/b01-lego-layers_${SLURM_JOB_ID}}"
WANDB_MODE="${WANDB_MODE:-online}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

export PATH="$CONDA_ENVIRONMENT/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export CUDA_HOME="$CONDA_ENVIRONMENT"
export FOAMSURF_DRIVER_LIBRARY_DIRECTORY="$DRIVER_DIRECTORY"
export LD_LIBRARY_PATH="$DRIVER_DIRECTORY:$CONDA_ENVIRONMENT/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

if [[ ! -x "$PYTHON" ]]; then
    echo "Python not found: $PYTHON" >&2
    exit 1
fi
if [[ ! -f "$PROJECT_DIRECTORY/renderer/foamsurf/_C.so" ]]; then
    echo "Renderer is not built. Run ./build.sh before sbatch." >&2
    exit 1
fi
if [[ ! -f "$PROJECT_DIRECTORY/renderer/foamsurf/_optix/PYOPTIXFLARERENDERER.so" ]]; then
    echo "OptiX renderer is not built. Run ./build.sh before sbatch." >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIRECTORY" "$PROJECT_DIRECTORY/slurm_logs"
cd "$PROJECT_DIRECTORY"

arguments=(
    --config "$CONFIGURATION"
    --output-dir "$OUTPUT_DIRECTORY"
    --wandb-mode "$WANDB_MODE"
)
if [[ -n "$RESUME_CHECKPOINT" ]]; then
    arguments+=(--resume "$RESUME_CHECKPOINT")
fi

echo "FoamSurf B0.1 layered entanglement evaluation"
echo "job_id=$SLURM_JOB_ID"
echo "node=${SLURMD_NODENAME:-unknown}"
echo "training_dataset=/shared/sets/datasets/foamsurf/lego/lego_both"
echo "gaussian_ground_truth=/shared/sets/datasets/foamsurf/lego/lego_lego"
echo "medium_ground_truth=/shared/sets/datasets/foamsurf/lego/lego_clouds"
echo "output=$OUTPUT_DIRECTORY"
echo "wandb_mode=$WANDB_MODE"
echo "resume=${RESUME_CHECKPOINT:-none}"
nvidia-smi

"$PYTHON" train_b0.py "${arguments[@]}"

test -f "$OUTPUT_DIRECTORY/checkpoints/last.pt"
test -f "$OUTPUT_DIRECTORY/checkpoints/best.pt"
test -f "$OUTPUT_DIRECTORY/metrics/train.csv"
test -f "$OUTPUT_DIRECTORY/metrics/eval.csv"
test -f "$OUTPUT_DIRECTORY/summary.json"
test -f "$OUTPUT_DIRECTORY/evaluation/iteration_035000/per_view.csv"
test -f "$OUTPUT_DIRECTORY/evaluation/iteration_035000/0000_r_0_lego_lego.png"
test -f "$OUTPUT_DIRECTORY/evaluation/iteration_035000/0000_r_0_lego_clouds.png"
test -f "$OUTPUT_DIRECTORY/evaluation/iteration_035000/0000_r_0_layer_masks.png"

echo "FoamSurf layered B0.1 completed successfully: $OUTPUT_DIRECTORY"
