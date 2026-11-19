#!/bin/bash
#SBATCH --job-name=foamsurf
#SBATCH --partition=rtx4090_batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%A_%a.out
# Train FoamSurf on the four SeaThru-NeRF scenes (one array task per scene):
#   sbatch --array=0-3 scripts/slurm_train.sbatch
#   FS_CONFIG=configs/garden_fog.yaml FS_DATA=/data/foggy_garden FS_SCENES="garden_easy garden_hard" \
#     sbatch --array=0-1 scripts/slurm_train.sbatch
# Settings (environment): FS_DATA (scene parent directory), FS_SCENES, FS_CONFIG, FS_SEED, FS_OUT (run parent
# directory), FS_ENV (conda environment), FS_WANDB (W&B project; logging off when empty), FS_DRIVER (directory
# with libcuda / libnvoptix when they are not in the system library path), TORCH_CUDA_ARCH_LIST (8.9 = RTX 4090),
# FS_RUN_ID (run directory of an earlier job, to resume it), FS_EXTRA (more train.py arguments, e.g. "--set joint.dataset.gaussian_ground_truth_path=...").
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
ENV="${FS_ENV:-${CONDA_PREFIX:?activate the conda environment or set FS_ENV}}"
DATA="${FS_DATA:-data/SeathruNeRF_dataset}"
read -r -a SCENES <<< "${FS_SCENES:-IUI3-RedSea-Undistorted Curasao-Undistorted JapaneseGradens-RedSea-Undistorted Panama-Undisorted}"
SCENE="${SCENES[${SLURM_ARRAY_TASK_ID:-0}]}"
SEED="${FS_SEED:-42}"
OUT="${FS_OUT:-$ROOT/outputs}/${FS_RUN_ID:-${SLURM_ARRAY_JOB_ID:-local}}/${SCENE}_seed${SEED}"  # FS_RUN_ID: resume an earlier job

export PATH="$ENV/bin:$PATH" CUDA_HOME="$ENV" LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
if [[ -n "${FS_DRIVER:-}" ]]; then
  export FOAMSURF_DRIVER_LIBRARY_DIRECTORY="$FS_DRIVER" LD_LIBRARY_PATH="$FS_DRIVER:$LD_LIBRARY_PATH"
fi
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
export TORCH_EXTENSIONS_DIR="$ROOT/outputs/.torch_extensions/${SLURM_JOB_ID:-local}"  # PPISP / SH kernels, per job
cd "$ROOT"
ARGS=(--scene "$DATA/$SCENE" --out "$OUT" --config "${FS_CONFIG:-configs/seathru.yaml}" --seed "$SEED")
[[ -n "${FS_WANDB:-}" ]] && ARGS+=(--wandb "$FS_WANDB")
if [[ -n "${FS_EXTRA:-}" ]]; then read -r -a EXTRA <<< "$FS_EXTRA"; ARGS+=("${EXTRA[@]}"); fi
[[ -d "$OUT" ]] && ARGS+=(--resume)
mkdir -p "$(dirname "$OUT")"  # here, not in Python: array tasks starting together race on the parent
echo "scene=$SCENE seed=$SEED out=$OUT"
python train.py "${ARGS[@]}"
rm -rf "$TORCH_EXTENSIONS_DIR"
