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
# Only the audit (+ slimming) of finished runs: FS_RUN_ID=<job id> FS_AUDIT_ONLY=1 sbatch --array=<tasks> ...
# Space: W&B is off unless FS_WANDB=online; after the audit each run is slimmed (scripts/slim_run.py:
# evaluation renders every 5000 + last + best, no initialization.pt, no local W&B), unless FS_SLIM=0.
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
# _abs<N> = absolute (AbsGS / WaterSplatting) densification gradient with threshold N x 1e-4
# (plain gradient: 2e-4; WaterSplatting: abs 8e-4 = 4x its plain 2e-4).
# Absolute-gradient densification on the default config, seeds 42 and 43 (24 tasks; the control is
# the 'default' set):
#   FS_VARIANT_SET=abs sbatch --array=0-23%7 scripts/train_seathru.sbatch
# _plain = plain baseline: only 0.8 L1 + 0.2 D-SSIM. No medium priors (chroma, seabed coverage, water
# clearing), no opacity binarisation, no medium/surface overlap, no Voronoi smoothing, no Gaussian
# scale regularisation, no attenuation-compensated densification; pruning and densification as is.
# Plain baseline on every scene, seeds 42 and 43 (8 tasks):
#   FS_VARIANT_SET=plain sbatch --array=0-7%7 scripts/train_seathru.sbatch
# _gonly = Gaussians alone (representation_mode=gaussian_only, no medium; needs _plain). Is our Gaussian
# part as good as 2DGS/3DGS? Mip-NeRF 360 garden and bicycle (images_4, as 2DGS/3DGS) and Panama,
# seeds 42 and 43, SH degree 3, no 700k cap (_g6000k) (6 tasks):
#   FS_VARIANT_SET=gonly sbatch --array=0-5%7 scripts/train_seathru.sbatch
# Does the medium overfit the few training views? _cells<N> = N medium cells (fixed population,
# default 5000), _ss<N> = attenuation/backscatter smoothing N (default 0.1). Default config, every scene,
# seeds 42 and 43 (40 tasks):
#   FS_VARIANT_SET=overfit sbatch --array=0-39%7 scripts/train_seathru.sbatch
# Gaussians alone on the other SeaThru scenes (Panama is in 'gonly'), seeds 42 and 43 (6 tasks):
#   FS_VARIANT_SET=gonly_seathru sbatch --array=0-5%7 scripts/train_seathru.sbatch
# Calmer densification (the default fills the 700k cap by iteration 4-7k): _ramp = cap grows linearly
# to max_points at 15000, _pd01 = percent_dense 0.01 (3DGS/2DGS: more splits, fewer clones),
# _nocomp = no attenuation-compensated densification. Default config (SH3 since 2026-10-24) and the four
# variants, every scene, seeds 42, 43 (40 tasks):
#   FS_VARIANT_SET=calm sbatch --array=0-39%7 scripts/train_seathru.sbatch
# A calmer start (Gaussians covered the whole frame by iteration 500 to hide the grey initial medium):
# _clamp<NN> = every Gaussian scale capped at 0.NN x scene extent from iteration 0, _minit = medium
# starts with the mean colour of the photos, _scl005 = Gaussian scale LR 0.005 (3DGS). Control and 7
# variants, every scene, seeds 42 and 43 (64 tasks); keep the evaluations every 500 up to 5000:
#   FS_VARIANT_SET=start FS_SLIM_KEEP_UNTIL=5000 sbatch --array=0-63%7 scripts/train_seathru.sbatch
# Which part pulls the medium towards transparency (diagnostic, 10k iterations, every evaluation kept)?
# On top of _minit, one element off at a time: _nooverlap (medium/surface overlap), _nochroma
# (backscatter chroma prior), _nocover (seabed coverage), _noclear (water clearing), _nobin (opacity
# binarisation), _nosmooth (Voronoi smoothing); plus _plain_minit (all off) and _minit alone.
# Japanese Gardens and Curaçao, seed 42 (16 tasks); measure with scripts/medium_attenuation.py:
#   FS_VARIANT_SET=diag FS_SLIM=0 sbatch --array=0-15%7 scripts/train_seathru.sbatch
#   (another machine: FS_VARIANT_SET=diag FS_SLIM=0 scripts/run_local.sh 0 15)
# _mdc = medium colour without the sigmoid (0.5 + C0 * raw, as before 2026-10-20). With _minit, same scenes
# and seed, into the same folder as 'diag' (2 tasks):
#   FS_VARIANT_SET=diag_mdc FS_SLIM=0 FS_RUN_ID=<diag job> sbatch --array=0-1 scripts/train_seathru.sbatch
set -Eeuo pipefail

# Cluster paths by default; another machine (scripts/run_local.sh) sets FS_ROOT, FS_ENV, FS_DRIVER,
# FS_DATASETS, FS_MIPNERF and TORCH_CUDA_ARCH_LIST (e.g. 8.6 on an RTX 3090).
ROOT=${FS_ROOT:-/shared/results/gmwilczn/foamsurf/foamsurf}
FS_ENV=${FS_ENV:-/shared/results/gmwilczn/conda/envs/foamsurf}
DRIVER=${FS_DRIVER:-/shared/results/gmwilczn/drivers/NVIDIA-Linux-x86_64-535.183.01}
DATASETS=${FS_DATASETS:-/shared/sets/datasets/SeathruNeRF_dataset}
# Mip-NeRF 360 outdoor scenes, trained on images_4 like 3DGS and 2DGS.
MIPNERF=${FS_MIPNERF:-/shared/sets/datasets/mipnerf_ds}
MIPNERF_OUTDOOR=" bicycle flowers garden stump treehill "
export PATH="$FS_ENV/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export CUDA_HOME="$FS_ENV"
export FOAMSURF_DRIVER_LIBRARY_DIRECTORY="$DRIVER"
export LD_LIBRARY_PATH="$DRIVER:$FS_ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.9}
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
  gonly)
    VARIANTS=(_plain_sh3_g6000k_gonly)
    DEFAULT_SCENES="garden bicycle Panama-Undisorted"; DEFAULT_SEEDS="42 43" ;;
  diag)
    VARIANTS=(_minit_it10k _minit_nooverlap_it10k _minit_nochroma_it10k _minit_nocover_it10k
              _minit_noclear_it10k _minit_nobin_it10k _minit_nosmooth_it10k _plain_minit_it10k)
    DEFAULT_SCENES="JapaneseGradens-RedSea-Undistorted Curasao-Undistorted"; DEFAULT_SEEDS="42" ;;
  diag_mdc)
    VARIANTS=(_minit_mdc_it10k)
    DEFAULT_SCENES="JapaneseGradens-RedSea-Undistorted Curasao-Undistorted"; DEFAULT_SEEDS="42" ;;
  start)
    VARIANTS=("" _clamp10 _clamp05 _minit _clamp10_minit _clamp05_minit _clamp10_minit_scl005 _clamp05_minit_scl005)
    DEFAULT_SCENES=$ALL_SCENES; DEFAULT_SEEDS="42 43" ;;
  calm)
    VARIANTS=("" _ramp _pd01 _ramp_pd01 _nocomp)
    DEFAULT_SCENES=$ALL_SCENES; DEFAULT_SEEDS="42 43" ;;
  overfit)
    VARIANTS=(_cells1000 _cells250 _cells50 _ss1 _ss10)
    DEFAULT_SCENES=$ALL_SCENES; DEFAULT_SEEDS="42 43" ;;
  gonly_seathru)
    VARIANTS=(_plain_sh3_g6000k_gonly)
    DEFAULT_SCENES="IUI3-RedSea-Undistorted Curasao-Undistorted JapaneseGradens-RedSea-Undistorted"; DEFAULT_SEEDS="42 43" ;;
  plain)
    VARIANTS=(_plain)
    DEFAULT_SCENES=$ALL_SCENES; DEFAULT_SEEDS="42 43" ;;
  abs)
    VARIANTS=(_abs8 _abs8_g1200k _abs6_g1200k)
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
    elif [[ "$rest" =~ ^_it([0-9]+)k(.*)$ ]]; then
      # Shorter training: the schedules that end at 30000 end with it (densification still stops at 15000).
      echo "joint.optimization.iterations=$(( BASH_REMATCH[1] * 1000 ))"
      echo "joint.gaussian_adaptation.opacity_prune_until=$(( BASH_REMATCH[1] * 1000 ))"; rest=${BASH_REMATCH[2]}
    elif [[ "$rest" =~ ^_mdc(.*)$ ]]; then
      echo "medium_color_activation=dc"; rest=${BASH_REMATCH[1]}
    elif [[ "$rest" =~ ^_nooverlap(.*)$ ]]; then
      echo "rgb_surface.weight=0.0"; rest=${BASH_REMATCH[1]}
    elif [[ "$rest" =~ ^_nochroma(.*)$ ]]; then
      echo "medium_fix.chroma_weight=0.0"; rest=${BASH_REMATCH[1]}
    elif [[ "$rest" =~ ^_nocover(.*)$ ]]; then
      echo "medium_fix.surface_coverage_weight=0.0"; rest=${BASH_REMATCH[1]}
    elif [[ "$rest" =~ ^_noclear(.*)$ ]]; then
      echo "medium_fix.water_clear_weight=0.0"; rest=${BASH_REMATCH[1]}
    elif [[ "$rest" =~ ^_nobin(.*)$ ]]; then
      echo "joint.disentanglement.opacity_binarization_weight=0.0"; rest=${BASH_REMATCH[1]}
    elif [[ "$rest" =~ ^_nosmooth(.*)$ ]]; then
      echo "rgb_smoothing.attenuation=0.0"
      echo "rgb_smoothing.backscatter=0.0"
      echo "rgb_smoothing.color=0.0"; rest=${BASH_REMATCH[1]}
    elif [[ "$rest" =~ ^_clamp([0-9]{2})(.*)$ ]]; then
      echo "joint.optimization.gaussian_max_scale_scene_fraction=0.${BASH_REMATCH[1]}"; rest=${BASH_REMATCH[2]}
    elif [[ "$rest" =~ ^_minit(.*)$ ]]; then
      echo "joint.model.medium_initial_color_from_data=true"; rest=${BASH_REMATCH[1]}
    elif [[ "$rest" =~ ^_scl005(.*)$ ]]; then
      echo "joint.optimization.gaussian_scales_lr=0.005"
      echo "joint.optimization.gaussian_scales_lr_final=0.005"; rest=${BASH_REMATCH[1]}
    elif [[ "$rest" =~ ^_ramp(.*)$ ]]; then
      echo "joint.gaussian_adaptation.max_points_ramp_until=15000"; rest=${BASH_REMATCH[1]}
    elif [[ "$rest" =~ ^_pd01(.*)$ ]]; then
      echo "joint.gaussian_adaptation.percent_dense=0.01"; rest=${BASH_REMATCH[1]}
    elif [[ "$rest" =~ ^_nocomp(.*)$ ]]; then
      echo "joint.gaussian_adaptation.gradient_attenuation_compensation=false"; rest=${BASH_REMATCH[1]}
    elif [[ "$rest" =~ ^_cells([0-9]+)(.*)$ ]]; then
      echo "profile.spatial_count=${BASH_REMATCH[1]}"
      echo "warmup.model.medium_points=${BASH_REMATCH[1]}"
      echo "joint.model.medium_points=${BASH_REMATCH[1]}"
      echo "joint.voronoi_adaptation.max_points=${BASH_REMATCH[1]}"
      echo "joint.voronoi_adaptation.min_points=${BASH_REMATCH[1]}"; rest=${BASH_REMATCH[2]}
    elif [[ "$rest" =~ ^_ss([0-9]+)(.*)$ ]]; then
      echo "rgb_smoothing.attenuation=${BASH_REMATCH[1]}.0"
      echo "rgb_smoothing.backscatter=${BASH_REMATCH[1]}.0"; rest=${BASH_REMATCH[2]}
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
    elif [[ "$rest" =~ ^_plain(.*)$ ]]; then
      echo "medium_fix.chroma_weight=0.0"
      echo "medium_fix.surface_coverage_weight=0.0"
      echo "medium_fix.water_clear_weight=0.0"
      echo "medium_fix.under_surface_weight=0.0"
      echo "joint.disentanglement.opacity_binarization_weight=0.0"
      echo "rgb_surface.weight=0.0"
      echo "rgb_smoothing.attenuation=0.0"
      echo "rgb_smoothing.backscatter=0.0"
      echo "rgb_smoothing.color=0.0"
      echo "capacity_loss.repulsion_weight=0.0"
      echo "capacity_loss.direct_weight=0.0"
      echo "joint.optimization.gaussian_scale_regularization=0.0"
      echo "joint.gaussian_adaptation.gradient_attenuation_compensation=false"; rest=${BASH_REMATCH[1]}
    elif [[ "$rest" =~ ^_gonly(.*)$ ]]; then
      echo "joint.representation_mode=gaussian_only"
      echo "joint.voronoi_adaptation.enabled=false"; rest=${BASH_REMATCH[1]}
    elif [[ "$rest" =~ ^_abs([0-9]+)(.*)$ ]]; then
      echo "joint.gaussian_adaptation.absolute_gradient=true"
      echo "joint.gaussian_adaptation.gradient_threshold=${BASH_REMATCH[1]}.0e-4"; rest=${BASH_REMATCH[2]}
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
SOURCE="$DATASETS/$SCENE"
[[ "$MIPNERF_OUTDOOR" == *" $SCENE "* ]] && SOURCE="$MIPNERF/$SCENE"
[[ -d "$SOURCE" ]] || { echo "Missing dataset $SOURCE" >&2; exit 2; }
variant_overrides "$VARIANT" > /dev/null  # fail loudly on an unknown suffix
OVERRIDES=()
mapfile -t OVERRIDES < <(variant_overrides "$VARIANT")
if [[ "$SOURCE" == "$MIPNERF/"* ]]; then
  OVERRIDES+=("warmup.dataset.resolution=4" "joint.dataset.resolution=4")
fi
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
WANDB="${FS_WANDB:-disabled}"
OUTPUT="outputs/seathru/${RUN_ID}/${NAME}"

ARGS=(
  --config "${FS_CONFIG:-configs/iui3_default.yaml}"
  --output-dir "$OUTPUT"
  --run-suffix "$NAME"
  --set "warmup.dataset.source_path=$SOURCE"
  --set "joint.dataset.source_path=$SOURCE"
  --set "variant.seed=$SEED" --set "warmup.seed=$SEED" --set "joint.seed=$SEED"
  --set "warmup.wandb.mode=$WANDB" --set "joint.wandb.mode=$WANDB"
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
[[ "${FS_AUDIT_ONLY:-0}" == 1 ]] || python train.py "${ARGS[@]}"
# The medium audit needs the medium and the SAM masks of the SeaThru test views.
if [[ "$VARIANT" != *_gonly* && "$SOURCE" == "$DATASETS/"* ]]; then
  for checkpoint in best last; do
    python scripts/medium_audit.py --run "$OUTPUT" --checkpoint "$checkpoint"
  done
fi
if [[ "${FS_SLIM:-1}" == 1 ]]; then
  python scripts/slim_run.py --run "$OUTPUT" --keep-until "${FS_SLIM_KEEP_UNTIL:-0}" --apply
fi
