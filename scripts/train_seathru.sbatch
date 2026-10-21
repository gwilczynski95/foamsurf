#!/usr/bin/env bash
#SBATCH --job-name=fs_seathru
#SBATCH --output=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/seathru_%A_%a.out
#SBATCH --error=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/seathru_%A_%a.err
#SBATCH --partition=rtx4090_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --qos=batch
#SBATCH --exclude=c22,c11
#SBATCH --time=24:00:00
#SBATCH --array=0-7%7
# The default configuration (configs/iui3_default.yaml) on every SeaThru-NeRF scene,
# seeds 42 and 43: task = 2 * scene index + seed index. Each task trains, then audits
# best.pt and last.pt (water/seabed metrics on configs/water_sam/<scene>.npz).
#
#   sbatch scripts/train_seathru.sbatch
# One scene / other seeds:  FS_SCENES="Curasao-Undistorted" FS_SEEDS="42" sbatch --array=0 ...
# Resume:                   FS_RUN_ID=<job id> FS_RESUME=1 sbatch scripts/train_seathru.sbatch
# Another config:           FS_CONFIG=configs/<file>.yaml sbatch scripts/train_seathru.sbatch
# Voronoi layout x colour smoothing on IUI3 and Panama, seed 42 (18 tasks):
#   FS_VARIANT_SET=cells sbatch --array=0-17%7 scripts/train_seathru.sbatch
# Variant suffixes (on top of the config): _view75 = 75% of the cells uniform in the volume the
# training cameras see, _c20k = 20000 cells, _cs<N> = medium colour smoothing weight N (default 1),
# _sh3 = sigmoid(SH degree 3) Gaussian colour (degree +1 every 1000 iterations), _g<N>k = at most
# N thousand Gaussians, _dres = densification threshold scaled to the image width (IUI3 = 1383).
# Seed 42 on every scene for SH / capacity / resolution-scaled densification (20 tasks; the
# control is the default run of the 'default' set):
#   FS_VARIANT_SET=sh sbatch --array=0-19%7 scripts/train_seathru.sbatch
# _opt = 3DGS optimiser (Adam eps 1e-15, scales LR 0.005, quaternions LR 0.001, no scale
# regularisation), _rel = WaterSplatting loss (L1 and SSIM divided by the detached prediction),
# _noreg = no regularisers on the Gaussians (water clearing, seabed coverage, opacity binarisation).
# _wsop = WaterSplatting opacity schedule (reset to 0.5 every 500 and cull below 0.5 until 10000,
# then below 0.1 every 100; densify until 10000), _c2f = coarse-to-fine resolution (1/4, 1/2, full
# from 6000).
# Towards WaterSplatting on SH + resolution-scaled densification, seeds 42 and 43 (40 tasks):
#   FS_VARIANT_SET=ws sbatch --array=0-39%7 scripts/train_seathru.sbatch
set -Eeuo pipefail

ROOT=/shared/results/gmwilczn/foamsurf/foamsurf
FS_ENV=/shared/results/gmwilczn/conda/envs/foamsurf
DRIVER=/shared/results/gmwilczn/drivers/NVIDIA-Linux-x86_64-535.183.01
DATASETS=/shared/sets/datasets/SeathruNeRF_dataset
export PATH="$FS_ENV/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export CUDA_HOME="$FS_ENV"
export FOAMSURF_DRIVER_LIBRARY_DIRECTORY="$DRIVER"
export LD_LIBRARY_PATH="$DRIVER:$FS_ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export TORCH_CUDA_ARCH_LIST=8.9
export MAX_JOBS=2
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

# Panama's directory name is misspelled in the dataset.
ALL_SCENES="IUI3-RedSea-Undistorted Curasao-Undistorted JapaneseGradens-RedSea-Undistorted Panama-Undisorted"
case "${FS_VARIANT_SET:-default}" in
  default) VARIANTS=(""); DEFAULT_SCENES=$ALL_SCENES; DEFAULT_SEEDS="42 43" ;;
  cells)
    VARIANTS=("" _cs5 _cs10 _view75 _view75_cs5 _view75_cs10 _view75_c20k _view75_c20k_cs5 _view75_c20k_cs10)
    DEFAULT_SCENES="IUI3-RedSea-Undistorted Panama-Undisorted"; DEFAULT_SEEDS="42" ;;
  sh)
    VARIANTS=(_sh3 _g1200k _dres _sh3_g1200k _sh3_g1200k_dres)
    DEFAULT_SCENES=$ALL_SCENES; DEFAULT_SEEDS="42" ;;
  ws)
    VARIANTS=(_sh3_dres _sh3_dres_opt _sh3_dres_rel _sh3_dres_opt_rel _sh3_dres_opt_rel_noreg)
    DEFAULT_SCENES=$ALL_SCENES; DEFAULT_SEEDS="42 43" ;;
  *) echo "Unknown FS_VARIANT_SET: $FS_VARIANT_SET" >&2; exit 2 ;;
esac
read -r -a SCENES <<< "${FS_SCENES:-$DEFAULT_SCENES}"
read -r -a SEEDS <<< "${FS_SEEDS:-$DEFAULT_SEEDS}"

# Prints one manifest override per line for a variant suffix.
variant_overrides() {
  local rest=$1
  while [[ -n "$rest" ]]; do
    if [[ "$rest" =~ ^_view([0-9]{2})(.*)$ ]]; then
      echo "profile.spatial_view_fraction=0.${BASH_REMATCH[1]}"; rest=${BASH_REMATCH[2]}
    elif [[ "$rest" =~ ^_c([0-9]+)k(.*)$ ]]; then
      echo "profile.spatial_count=$(( BASH_REMATCH[1] * 1000 ))"
      echo "joint.voronoi_adaptation.max_points=$(( BASH_REMATCH[1] * 1000 ))"; rest=${BASH_REMATCH[2]}
    elif [[ "$rest" =~ ^_cs([0-9]+)(.*)$ ]]; then
      echo "rgb_smoothing.color=${BASH_REMATCH[1]}.0"; rest=${BASH_REMATCH[2]}
    elif [[ "$rest" =~ ^_sh([0-9])(.*)$ ]]; then
      echo "joint.model.appearance=sh"
      echo "joint.model.sh_degree=${BASH_REMATCH[1]}"
      echo "joint.optimization.gaussian_sh_degree_interval=1000"; rest=${BASH_REMATCH[2]}
    elif [[ "$rest" =~ ^_g([0-9]+)k(.*)$ ]]; then
      echo "joint.gaussian_adaptation.max_points=$(( BASH_REMATCH[1] * 1000 ))"; rest=${BASH_REMATCH[2]}
    elif [[ "$rest" =~ ^_opt(.*)$ ]]; then
      echo "joint.optimization.gaussian_adam_eps=1.0e-15"
      echo "joint.optimization.gaussian_scales_lr=0.005"
      echo "joint.optimization.gaussian_scales_lr_final=0.005"
      echo "joint.optimization.gaussian_quaternions_lr=0.001"
      echo "joint.optimization.gaussian_quaternions_lr_final=0.001"
      echo "joint.optimization.gaussian_scale_regularization=0.0"; rest=${BASH_REMATCH[1]}
    elif [[ "$rest" =~ ^_rel(.*)$ ]]; then
      echo "joint.optimization.reconstruction_loss=reg_l1_reg_dssim"; rest=${BASH_REMATCH[1]}
    elif [[ "$rest" =~ ^_noreg(.*)$ ]]; then
      echo "medium_fix.water_clear_weight=0.0"
      echo "medium_fix.surface_coverage_weight=0.0"
      echo "joint.disentanglement.opacity_binarization_weight=0.0"; rest=${BASH_REMATCH[1]}
    elif [[ "$rest" =~ ^_wsop(.*)$ ]]; then
      echo "joint.gaussian_adaptation.densify_until=10000"
      echo "joint.gaussian_adaptation.opacity_reset_interval=500"
      echo "joint.gaussian_adaptation.opacity_reset_value=0.5"
      echo "joint.gaussian_adaptation.min_opacity=0.5"
      echo "joint.gaussian_adaptation.min_opacity_initial=0.5"
      echo "joint.gaussian_adaptation.opacity_prune_ramp=0"
      echo "joint.gaussian_adaptation.min_opacity_late=0.1"
      echo "joint.gaussian_adaptation.min_opacity_late_from=10000"
      echo "joint.gaussian_adaptation.opacity_prune_interval=100"
      echo "joint.gaussian_adaptation.max_opacity_prune_fraction=1.0"; rest=${BASH_REMATCH[1]}
    elif [[ "$rest" =~ ^_c2f(.*)$ ]]; then
      echo "joint.optimization.training_downscale_levels=2"
      echo "joint.optimization.training_downscale_interval=3000"; rest=${BASH_REMATCH[1]}
    elif [[ "$rest" =~ ^_dres(.*)$ ]]; then
      echo "joint.gaussian_adaptation.gradient_threshold_reference_width=1383"; rest=${BASH_REMATCH[1]}
    else
      echo "Unknown variant suffix: $rest" >&2
      return 2
    fi
  done
}

TASK=${SLURM_ARRAY_TASK_ID:?run through sbatch}
PER_SCENE=$(( ${#VARIANTS[@]} * ${#SEEDS[@]} ))
(( TASK < ${#SCENES[@]} * PER_SCENE )) || { echo "Array index $TASK out of range" >&2; exit 2; }
SCENE=${SCENES[TASK / PER_SCENE]}
VARIANT=${VARIANTS[(TASK % PER_SCENE) / ${#SEEDS[@]}]}
SEED=${SEEDS[TASK % ${#SEEDS[@]}]}
[[ -d "$DATASETS/$SCENE" ]] || { echo "Missing dataset $DATASETS/$SCENE" >&2; exit 2; }
variant_overrides "$VARIANT" > /dev/null  # fail loudly on an unknown suffix
OVERRIDES=()
mapfile -t OVERRIDES < <(variant_overrides "$VARIANT")
NAME="${SCENE}${VARIANT}"
[[ "$SEED" == 42 ]] || NAME="${NAME}_seed${SEED}"

cd "$ROOT"
mkdir -p slurm_logs
export TORCH_EXTENSIONS_DIR="$ROOT/outputs/build_seathru_${SLURM_ARRAY_JOB_ID}_${TASK}"
mkdir -p "$TORCH_EXTENSIONS_DIR"
# Temporary files (W&B media) on shared storage: a node's local /tmp can fill up.
export TMPDIR="$ROOT/outputs/tmp/seathru_${SLURM_ARRAY_JOB_ID}_${TASK}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
RUN_ID="${FS_RUN_ID:-${SLURM_ARRAY_JOB_ID}}"
OUTPUT="outputs/seathru/${RUN_ID}/${NAME}"

ARGS=(
  --config "${FS_CONFIG:-configs/iui3_default.yaml}"
  --output-dir "$OUTPUT"
  --run-suffix "$NAME"
  --set "warmup.dataset.source_path=$DATASETS/$SCENE"
  --set "joint.dataset.source_path=$DATASETS/$SCENE"
  --set "variant.seed=$SEED" --set "warmup.seed=$SEED" --set "joint.seed=$SEED"
)
for item in "${OVERRIDES[@]}"; do ARGS+=(--set "$item"); done
if [[ "${FS_RESUME:-0}" == 1 ]]; then ARGS+=(--resume); fi

echo "scene=$SCENE variant=${VARIANT:-none} seed=$SEED output=$OUTPUT"
if (( ${#OVERRIDES[@]} )); then printf '  --set %s\n' "${OVERRIDES[@]}"; fi
if [[ "${FS_DRY_RUN:-0}" == 1 ]]; then
  python train.py "${ARGS[@]}" --dry-run > /dev/null
  echo "dry run ok"
  exit 0
fi
python train.py "${ARGS[@]}"
for checkpoint in best last; do
  python scripts/medium_audit.py --run "$OUTPUT" --checkpoint "$checkpoint"
done
