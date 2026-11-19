#!/bin/bash
#SBATCH --job-name=blender_sfm
#SBATCH --partition=rtx4090_batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%A_%a.out
# SfM points for Blender scenes with their known poses (scripts/blender_sfm.py), one array task per scene:
#   sbatch --array=0-1 scripts/blender_sfm.sbatch
# Settings (environment): FS_DATA (parent of the scenes), FS_SCENES (scene paths under FS_DATA), FS_SFM_OUT (output
# parent; one directory per scene, must not exist yet), FS_ENV (Python with numpy / PIL), FS_COLMAP_ENV (conda
# environment with COLMAP built with CUDA). No display is needed: SIFT extraction and matching run on CUDA, Qt is
# set to offscreen.
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
ENV="${FS_ENV:?set FS_ENV to the conda environment}"
CENV="${FS_COLMAP_ENV:?set FS_COLMAP_ENV to an environment with COLMAP}"
DATA="${FS_DATA:-data/foamsurf}"
read -r -a SCENES <<< "${FS_SCENES:-lego/lego_both ficus/ficus_both}"
SCENE="${SCENES[${SLURM_ARRAY_TASK_ID:-0}]}"
OUT="${FS_SFM_OUT:-data/synthetic_sfm}/$(basename "$SCENE")"

export QT_QPA_PLATFORM=offscreen PYTHONUNBUFFERED=1
unset DISPLAY
colmap() { LD_LIBRARY_PATH="$CENV/lib:${LD_LIBRARY_PATH:-}" "$CENV/bin/colmap" "$@"; }
python() { "$ENV/bin/python" "$@"; }

echo "scene=$DATA/$SCENE out=$OUT host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd "$ROOT"
python scripts/blender_sfm.py prepare "$DATA/$SCENE" "$OUT"
PARAMS=$(python -c "import json; c = json.load(open('$OUT/camera.json')); print(','.join(repr(c[k]) for k in ('fx', 'fy', 'cx', 'cy')))")

extract() {
  colmap feature_extractor --database_path "$OUT/database.db" --image_path "$OUT/images" \
    --ImageReader.single_camera 1 --ImageReader.camera_model PINHOLE --ImageReader.camera_params "$PARAMS" \
    --FeatureExtraction.use_gpu "$1" --FeatureExtraction.gpu_index 0
}
match() { colmap exhaustive_matcher --database_path "$OUT/database.db" --FeatureMatching.use_gpu "$1" --FeatureMatching.gpu_index 0; }
# GPU (CUDA) first; a failure there falls back to the CPU of this node instead of ending the job
if ! extract 1; then
  echo "WARNING: GPU feature extraction failed, extracting on the CPU"
  rm -f "$OUT/database.db"*  # this job's own partial database
  extract 0
fi
if ! match 1; then
  echo "WARNING: GPU matching failed, matching on the CPU"
  match 0
fi
python scripts/blender_sfm.py model "$OUT"
mkdir -p "$OUT/sparse/0"
colmap point_triangulator --database_path "$OUT/database.db" --image_path "$OUT/images" \
  --input_path "$OUT/sparse_known" --output_path "$OUT/sparse/0" --refine_intrinsics 0
colmap model_analyzer --path "$OUT/sparse/0" || echo "WARNING: model_analyzer failed (statistics only)"
[[ -s "$OUT/sparse/0/points3D.bin" ]] || { echo "ERROR: no points3D.bin"; exit 1; }
echo "done: $OUT/sparse/0"
