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
# The default configuration (configs/seathru_default.yaml: OceanSplat + PPISP) on every SeaThru-NeRF
# scene, seeds 42 and 43: task = 2 * scene index + seed index. Each task trains, then audits
# best.pt and last.pt (water/seabed metrics on configs/water_sam/<scene>.npz).
# Every other variant set builds its suffixes on configs/iui3_default.yaml (FS_CONFIG overrides both).
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
# Water fitted on the training photos (training/water_fit.py, COLMAP point tracks): _wfit = every cell
# starts with the fitted attenuation and backscatter (colour: _minit mean), _wfitc = also the fitted
# B_inf colour. Organic, calmer densification: _cmax<N> = attenuation compensation capped at N (default
# 20), _gt<N> = gradient threshold N x 1e-4 (default 2e-4). 2 x 2 grid (water start x densification)
# on _clamp05_minit (best so far), every scene, seeds 42 and 43 (32 tasks):
#   FS_VARIANT_SET=water FS_SLIM_KEEP_UNTIL=5000 sbatch --array=0-31%7 scripts/train_seathru.sbatch
# Overfitting to individual training photos (view-specific dark Gaussians and medium cells near the
# cameras). _gdk = Gaussian-DK near-camera densification gradient scaling with the paper's schedule
# (x0.5 from 4000, off from 7000), _gdkw = the same weight without the schedule, _tri = OceanSplat
# trinocular view consistency (lambda 0.1, 4000-24000). On _clamp05_minit (controls: water set,
# job 548623), every scene, seeds 42 and 43 (32 tasks):
#   FS_VARIANT_SET=consist FS_SLIM_KEEP_UNTIL=5000 sbatch --array=0-31%7 scripts/train_seathru.sbatch
# Full OceanSplat (training/oceansplat.py): _rs = their resolution schedule (x1/4, x1/2 from 6000, full
# from 12000), _ocean = trinocular consistency warped by the rendered depth + disparity smoothness
# (lambda 0.1), epipolar depth prior (0.4 -> 0.2), depth residual (0.01), all 12000-24000, and the
# depth-aware opacity until 12000. Both arms use their photometric loss (_rel = regularised L1 +
# regularised D-SSIM 0.2, as WaterSplatting / OceanSplat), so lambda_tri 0.1 means what it means in
# the paper. _oceane = the same with the loss window from 4000 (the paper's start in iterations; our
# JG overfitting starts at ~3k). 3 x 2 grid (no OceanSplat / window 12k-24k / window 4k-24k) x (without /
# with _rs) on _clamp05_minit, every scene, seeds 42 and 43 (48 tasks):
#   FS_VARIANT_SET=ocean sbatch --array=0-47 scripts/train_seathru.sbatch
# Water baked into the Gaussians (Panama MTN_1529: medium transmittance to the seabed 0.83-0.98 and grey,
# the seed-dependent colour offset sits in the Gaussian colours): the medium/surface overlap prior is the
# main pull towards a transparent medium (diag 548464), never run to 30k. _nooverlap on the 'ocean'
# control _clamp05_minit_rel, every scene, seeds 42 and 43 (8 tasks):
#   FS_VARIANT_SET=overlap sbatch --array=0-7 scripts/train_seathru.sbatch
# Sub-pixel Gaussians (Panama: 56% of the opacity below 0.5 px at the nearest training camera; the
# pixel-area render beats the pixel-centre render on test views, +0.25 dB, and loses on training views):
# _jit = training rays through a random point of their pixel (evaluation: centres), _mip = Mip-Splatting
# 3D filter (variance 0.2 px^2). On the 'ocean' control _clamp05_minit_rel, every scene, seeds 42 and
# 43 (24 tasks):
#   FS_VARIANT_SET=alias sbatch --array=0-23 scripts/train_seathru.sbatch
# The same on the best OceanSplat variant (550540: _clamp05_minit_rs_rel_oceane, control in 550540),
# plus that variant without _rel (our photometric loss: _rel alone was -0.22 dB), every scene, seeds 42
# and 43 (32 tasks):
#   FS_VARIANT_SET=alias2 sbatch --array=0-31 scripts/train_seathru.sbatch
# The camera after the water: _isp = PPISP (training/camera_isp.py, CVPR 2026) with per-photo exposure and
# colour, shared vignetting and response curve; test views neutral. Fitted after training on a frozen
# 550540 model it already gave +0.26 dB (Curacao +0.48, IUI3 +0.43). On _clamp05_minit_rs_rel_oceane
# (control in 550540), every scene, seeds 42 and 43 (8 tasks):
#   FS_VARIANT_SET=isp sbatch --array=0-7 scripts/train_seathru.sbatch
# Mip-Splatting 3D filter on the new default (configs/seathru_default.yaml = OceanSplat + PPISP): on
# _rs_rel_oceane it gave Panama +0.24 dB (sub-pixel Gaussians), IUI3 / JG about 0; the Curacao loss was the
# camera-cell exposure view that PPISP fixes. Every scene, seeds 42 and 43 (8 tasks):
#   FS_VARIANT_SET=mip sbatch --array=0-7%7 scripts/train_seathru.sbatch
# On the new default: _camw = homogeneous water at the cameras (medium_fix.camera_water_*: the free-water
# cells within 2 camera spacings of a training camera pulled to the median water, weight 0.01; each
# camera's cell had become a per-photo brightness knob, JG MTN_1098), _oe2_it20k = OceanSplat losses every
# 2nd step (x2) and 20k iterations with every schedule scaled by 2/3 (the window costs 40-45% of the
# training time; JG overfits after ~16k), _g400k = at most 400k Gaussians (rendering speed). Every scene,
# seeds 42 and 43 (24 tasks):
#   FS_VARIANT_SET=camw_speed sbatch --array=0-23%7 scripts/train_seathru.sbatch
# camw_speed (552535) on the default: _oe2_it20k 30.14 dB in 29 min (default 29.95 in 53 min), _camw 30.10
# (JG +0.49, Panama -0.19), _mip 30.00 (552525, Panama +0.19), _g400k 29.87. Do they add up (16 tasks)?
#   FS_VARIANT_SET=combo sbatch --array=0-15%7 scripts/train_seathru.sbatch
# Since 2026-10-29 configs/seathru_default.yaml is combo's _camw_oe2_it20k_mip (30.22 dB, 31 min), so the
# mip / camw_speed / combo sets on it now mean their suffixes on top of that (the 552525 / 552535 / 552797
# runs were on the earlier default). Faster training on the new default: _oe4 = OceanSplat every 4th step
# (x4), _it15k = 15k iterations with every schedule scaled (16 tasks):
#   FS_VARIANT_SET=speed2 sbatch --array=0-15%7 scripts/train_seathru.sbatch
# More Gaussians on the new default: every scene hits the 700k cap (about 52 densifications skipped per
# run) and WaterSplatting uses 0.8-1.14M (Panama 1.14M, where we lose on seabed detail) (16 tasks):
#   FS_VARIANT_SET=capacity sbatch --array=0-15%7 scripts/train_seathru.sbatch
# Lower SH degree on the new default: seabed Gaussians seen alone change colour with the camera distance
# (Panama MTN_1547: 105 at 8.6 units, 69 at 16), i.e. the direction-only SH colour carries part of the
# water's attenuation and cannot brighten a closer test view (16 tasks):
#   FS_VARIANT_SET=shdeg sbatch --array=0-15%7 scripts/train_seathru.sbatch
# Water attenuation from the data on the new default: _wfit = the cells start from the fit on the COLMAP
# tracks, _wanc = that fitted beta_D also anchors the level of the free water during training (weight 0.1).
# On foggy Garden the fit is close to the truth (beta_D 0.75 / 1.28 vs 0.60 / 1.10) while the trained medium
# ends at half (0.27 / 0.53), the Gaussian colours taking the rest. SeaThru (16 tasks) and foggy Garden (8):
#   FS_VARIANT_SET=wfit sbatch --array=0-15%7 scripts/train_seathru.sbatch
#   FS_VARIANT_SET=wfit_foggy sbatch --array=0-7 scripts/train_seathru.sbatch
# Foggy Garden is ~3 dB below WaterSplatting, 95% of it on the near, covered scene (detail and low
# frequency), not in the empty background. garden_diag = the default on the same photos without fog (is it
# the representation or the fog?); garden_tune = Garden settings chosen on held-out training photos (_val8:
# every 8th training photo evaluated, the test views unused): without the underwater medium priors (_plain),
# lower densification threshold (_dth10 = 1e-4), both (2 + 8 tasks):
#   FS_VARIANT_SET=garden_diag sbatch --array=0-1 scripts/train_seathru.sbatch
#   FS_VARIANT_SET=garden_tune sbatch --array=0-7 scripts/train_seathru.sbatch
# Since 2026-10-30 the default includes _wfit (553596: 30.36 dB, every scene up). Garden round 2 on held-out
# training photos: capacity (WaterSplatting puts 2.67M Gaussians on clear Garden, we stop at the 700k cap;
# clear Garden 25.2 vs 26.7 dB) and the attenuation anchor at 0.03 vs 0.1 (0.1 fixes the fog attenuation,
# 0.58 vs truth 0.60, but costs 0.7 dB on hard) (9 + 4 tasks):
#   FS_VARIANT_SET=garden_cap sbatch --array=0-8 scripts/train_seathru.sbatch
#   FS_VARIANT_SET=garden_anchor sbatch --array=0-3 scripts/train_seathru.sbatch
# Which underwater prior removes the Gaussians from the fogged background? Without the medium priors
# (_plain) Garden coverage (alpha > 0.5) goes 74 -> 91% (easy) and 54 -> 86% (hard) at the same PSNR;
# leave one out on held-out photos (8 tasks):
#   FS_VARIANT_SET=garden_prior sbatch --array=0-7 scripts/train_seathru.sbatch
# garden_prior result: the water clearing prior removes the fogged background (without it coverage 75 -> 92%
# easy, 54 -> 81% hard) and costs PSNR on held-out photos (+0.10 / +0.49 without it); the other priors do
# not matter there. Test it on the Garden test views (with the default as control, both on the current code)
# and on SeaThru, where it was introduced for open water (8 + 8 tasks):
#   FS_VARIANT_SET=noclear_foggy sbatch --array=0-7 scripts/train_seathru.sbatch
#   FS_VARIANT_SET=noclear sbatch --array=0-7%7 scripts/train_seathru.sbatch
# Garden underfits: training PSNR = test PSNR (clear 25.7 / 25.2), renders blurred (Laplacian energy 0.39 of
# the photo vs WS 0.56), the training curve still rising by ~3 dB at the end (20k, 161 photos = ~120 visits
# each) and flat during the OceanSplat window. Longer training, no OceanSplat losses, no Mip filter, on
# held-out photos (10 tasks):
#   FS_VARIANT_SET=garden_fit sbatch --array=0-9 scripts/train_seathru.sbatch
# garden_fit: without the OceanSplat losses Garden easy goes 31.6 -> 34.0 dB on held-out photos (34.5 with
# 40k), clear 25.1 -> 25.8 (26.0); _noclear is neutral on SeaThru (30.35 vs 30.36). Test views (12 tasks), and
# which OceanSplat part hurts Garden (depth-aware opacity, trinocular, epipolar + residual) (6 tasks):
#   FS_VARIANT_SET=garden_final sbatch --array=0-11 scripts/train_seathru.sbatch
#   FS_VARIANT_SET=garden_ocean sbatch --array=0-5 scripts/train_seathru.sbatch
#   FS_VARIANT_SET=garden_anchor2 sbatch --array=0-5 scripts/train_seathru.sbatch
#   FS_VARIANT_SET=garden_final2 sbatch --array=0-7 scripts/train_seathru.sbatch
#   FS_VARIANT_SET=unify sbatch --array=0-15 scripts/train_seathru.sbatch
#   FS_VARIANT_SET=garden_epivis sbatch --array=0-1 scripts/train_seathru.sbatch
#   FS_VARIANT_SET=epivis sbatch --array=0-7 scripts/train_seathru.sbatch
#   FS_VARIANT_SET=epivis_wanc sbatch --array=0-11 scripts/train_seathru.sbatch
#   FS_VARIANT_SET=fitlr sbatch --array=0-19 scripts/train_seathru.sbatch
#   FS_VARIANT_SET=adameps sbatch --array=0-7 scripts/train_seathru.sbatch
#   FS_VARIANT_SET=owin sbatch --array=0-7 scripts/train_seathru.sbatch
#   FS_VARIANT_SET=wsapp_wdir sbatch --array=0-15 scripts/train_seathru.sbatch
#   FS_VARIANT_SET=classic sbatch --array=0-23 scripts/train_seathru.sbatch
#   FS_VARIANT_SET=cull sbatch --array=0-23 scripts/train_seathru.sbatch
#   FS_VARIANT_SET=fuzz sbatch --array=0-31 scripts/train_seathru.sbatch
#   FS_VARIANT_SET=reset sbatch --array=0-23 scripts/train_seathru.sbatch
#   FS_VARIANT_SET=cover sbatch --array=0-23 scripts/train_seathru.sbatch
#   FS_VARIANT_SET=phase sbatch --array=0-7 scripts/train_seathru.sbatch
# The default on foggy Garden (easy, hard), seeds 42 and 43 (4 tasks; no SAM audit there):
#   FS_VARIANT_SET=foggy sbatch --array=0-3 scripts/train_seathru.sbatch
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
# Mip-NeRF 360 Garden in simulated fog (WaterSplatting Sec. 4; 1297x840 = images_4, test = sorted[::8]).
FOGGY=${FS_FOGGY:-/shared/sets/datasets/foamsurf/foggy_garden}
FOGGY_SCENES=" garden_easy garden_hard "
# The same Garden photos without fog (symlinks to foggy_garden/garden_easy/clear and its COLMAP model).
GARDEN_CLEAR=${FS_GARDEN_CLEAR:-/shared/results/gmwilczn/foamsurf/datasets/garden_clear}
export PATH="$FS_ENV/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export CUDA_HOME="$FS_ENV"
export FOAMSURF_DRIVER_LIBRARY_DIRECTORY="$DRIVER"
export LD_LIBRARY_PATH="$DRIVER:$FS_ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
# Fewer unusable reserved blocks (foggy Garden ran out of memory with 2.5 GB reserved but unallocated).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.9}
export MAX_JOBS=2
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

# Panama's directory name is misspelled in the dataset.
ALL_SCENES="IUI3-RedSea-Undistorted Curasao-Undistorted JapaneseGradens-RedSea-Undistorted Panama-Undisorted"
case "${FS_VARIANT_SET:-default}" in
  default) VARIANTS=(""); DEFAULT_SCENES=$ALL_SCENES; DEFAULT_SEEDS="42 43"; SET_CONFIG=configs/seathru_default.yaml ;;
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
  water)
    VARIANTS=(_clamp05_minit _clamp05_minit_cmax5 _clamp05_minit_wfit _clamp05_minit_wfit_cmax5)
    DEFAULT_SCENES=$ALL_SCENES; DEFAULT_SEEDS="42 43" ;;
  consist)
    VARIANTS=(_clamp05_minit_gdk _clamp05_minit_gdkw _clamp05_minit_tri _clamp05_minit_tri_gdkw)
    DEFAULT_SCENES=$ALL_SCENES; DEFAULT_SEEDS="42 43" ;;
  ocean)
    VARIANTS=(_clamp05_minit_rel _clamp05_minit_rel_ocean _clamp05_minit_rel_oceane
              _clamp05_minit_rs_rel _clamp05_minit_rs_rel_ocean _clamp05_minit_rs_rel_oceane)
    DEFAULT_SCENES=$ALL_SCENES; DEFAULT_SEEDS="42 43" ;;
  overlap)
    VARIANTS=(_clamp05_minit_rel_nooverlap)
    DEFAULT_SCENES=$ALL_SCENES; DEFAULT_SEEDS="42 43" ;;
  alias)
    VARIANTS=(_clamp05_minit_rel_jit _clamp05_minit_rel_mip _clamp05_minit_rel_jit_mip)
    DEFAULT_SCENES=$ALL_SCENES; DEFAULT_SEEDS="42 43" ;;
  isp)
    VARIANTS=(_clamp05_minit_rs_rel_oceane_isp)
    DEFAULT_SCENES=$ALL_SCENES; DEFAULT_SEEDS="42 43" ;;
  mip)
    VARIANTS=(_mip); SET_CONFIG=configs/seathru_default.yaml
    DEFAULT_SCENES=$ALL_SCENES; DEFAULT_SEEDS="42 43" ;;
  camw_speed)
    VARIANTS=(_camw _oe2_it20k _g400k); SET_CONFIG=configs/seathru_default.yaml
    DEFAULT_SCENES=$ALL_SCENES; DEFAULT_SEEDS="42 43" ;;
  combo)
    VARIANTS=(_camw_oe2_it20k _camw_oe2_it20k_mip); SET_CONFIG=configs/seathru_default.yaml
    DEFAULT_SCENES=$ALL_SCENES; DEFAULT_SEEDS="42 43" ;;
  speed2)
    VARIANTS=(_oe4 _it15k); SET_CONFIG=configs/seathru_default.yaml
    DEFAULT_SCENES=$ALL_SCENES; DEFAULT_SEEDS="42 43" ;;
  capacity)
    VARIANTS=(_g1000k _g1200k); SET_CONFIG=configs/seathru_default.yaml
    DEFAULT_SCENES=$ALL_SCENES; DEFAULT_SEEDS="42 43" ;;
  shdeg)
    VARIANTS=(_sh1 _sh2); SET_CONFIG=configs/seathru_default.yaml
    DEFAULT_SCENES=$ALL_SCENES; DEFAULT_SEEDS="42 43" ;;
  wfit)
    VARIANTS=(_wfit _wfit_wanc); SET_CONFIG=configs/seathru_default.yaml
    DEFAULT_SCENES=$ALL_SCENES; DEFAULT_SEEDS="42 43" ;;
  wfit_foggy)
    VARIANTS=(_wfit _wfit_wanc); SET_CONFIG=configs/seathru_default.yaml
    DEFAULT_SCENES="garden_easy garden_hard"; DEFAULT_SEEDS="42 43" ;;
  garden_diag)
    VARIANTS=(""); SET_CONFIG=configs/seathru_default.yaml
    DEFAULT_SCENES="garden_clear"; DEFAULT_SEEDS="42 43" ;;
  garden_tune)
    VARIANTS=(_val8 _plain_val8 _dth10_val8 _plain_dth10_val8); SET_CONFIG=configs/seathru_default.yaml
    DEFAULT_SCENES="garden_easy garden_hard"; DEFAULT_SEEDS="42" ;;
  garden_cap)
    VARIANTS=(_val8 _g1500k_val8 _g3000k_val8); SET_CONFIG=configs/seathru_default.yaml
    DEFAULT_SCENES="garden_easy garden_hard garden_clear"; DEFAULT_SEEDS="42" ;;
  garden_anchor)
    VARIANTS=(_wanc03_val8 _wanc_val8); SET_CONFIG=configs/seathru_default.yaml
    DEFAULT_SCENES="garden_easy garden_hard"; DEFAULT_SEEDS="42" ;;
  garden_prior)
    VARIANTS=(_noclear_val8 _nobin_val8 _nochroma_val8 _nocover_val8); SET_CONFIG=configs/seathru_default.yaml
    DEFAULT_SCENES="garden_easy garden_hard"; DEFAULT_SEEDS="42" ;;
  noclear)
    VARIANTS=(_noclear); SET_CONFIG=configs/seathru_default.yaml
    DEFAULT_SCENES=$ALL_SCENES; DEFAULT_SEEDS="42 43" ;;
  noclear_foggy)
    VARIANTS=("" _noclear); SET_CONFIG=configs/seathru_default.yaml
    DEFAULT_SCENES="garden_easy garden_hard"; DEFAULT_SEEDS="42 43" ;;
  garden_fit)
    VARIANTS=(_noclear_val8 _noclear_it40k_val8 _noclear_noocean_val8 _noclear_nomip_val8 _noclear_noocean_it40k_val8)
    SET_CONFIG=configs/seathru_default.yaml; DEFAULT_SCENES="garden_easy garden_clear"; DEFAULT_SEEDS="42" ;;
  garden_final)
    VARIANTS=(_noclear_noocean _noclear_noocean_it40k); SET_CONFIG=configs/seathru_default.yaml
    DEFAULT_SCENES="garden_easy garden_hard garden_clear"; DEFAULT_SEEDS="42 43" ;;
  garden_ocean)
    VARIANTS=(_noclear_noalpha_val8 _noclear_notri_val8 _noclear_noepi_val8); SET_CONFIG=configs/seathru_default.yaml
    DEFAULT_SCENES="garden_easy garden_hard"; DEFAULT_SEEDS="42" ;;
  garden_final2)
    VARIANTS=(_noclear_noocean_wanc03 _noclear_noocean_wanc03_it40k); SET_CONFIG=configs/seathru_default.yaml
    DEFAULT_SCENES="garden_easy garden_hard"; DEFAULT_SEEDS="42 43" ;;
  garden_epivis)
    VARIANTS=(_noclear_epivis_val8); SET_CONFIG=configs/seathru_default.yaml
    DEFAULT_SCENES="garden_easy garden_hard"; DEFAULT_SEEDS="42" ;;
  epivis)
    VARIANTS=(_noclear_epivis); SET_CONFIG=configs/seathru_default.yaml
    DEFAULT_SCENES=$ALL_SCENES; DEFAULT_SEEDS="42 43" ;;
  phase)
    # directional water light on top of the best run so far (_cull05r, 30.50)
    VARIANTS=(_cull05r_wphase); SET_CONFIG=configs/seathru_default.yaml
    DEFAULT_SCENES=$ALL_SCENES; DEFAULT_SEEDS="42 43" ;;
  cover)
    # seabed coverage without Gaussians in the water: coverage prior from alpha 0.7 / 0.8 / 0.9 at weight 0.03
    VARIANTS=(_cov07 _cov08 _cov09); SET_CONFIG=configs/seathru_default.yaml
    DEFAULT_SCENES=$ALL_SCENES; DEFAULT_SEEDS="42 43" ;;
  reset)
    # the two winners of 557114/557123 combined, and the opacity reset without the 0.5 culling
    VARIANTS=(_cull05r_bin01 _reset _reset_bin01); SET_CONFIG=configs/seathru_default.yaml
    DEFAULT_SCENES=$ALL_SCENES; DEFAULT_SEEDS="42 43" ;;
  fuzz)
    # the half-transparent seabed, one mechanism at a time (on the default, which evaluates with _vt)
    VARIANTS=(_bin01 _kfix _cov05 _nooverlap); SET_CONFIG=configs/seathru_default.yaml
    DEFAULT_SCENES=$ALL_SCENES; DEFAULT_SEEDS="42 43" ;;
  cull)
    # half-transparent seabed (a fuzz of ~11 surfels at opacity 0.5-0.7): WaterSplatting's opacity culling
    VARIANTS=(_cull05 _cull05r _wsop); SET_CONFIG=configs/seathru_default.yaml
    DEFAULT_SCENES=$ALL_SCENES; DEFAULT_SEEDS="42 43" ;;
  classic)
    # seabed coverage: mip filter without the opacity scale-down (as WaterSplatting), and no filter at all
    VARIANTS=(_classic _classic3 _nomip); SET_CONFIG=configs/seathru_default.yaml
    DEFAULT_SCENES=$ALL_SCENES; DEFAULT_SEEDS="42 43" ;;
  wsapp_wdir)
    VARIANTS=(_wsapp_dth10 _wdir); SET_CONFIG=configs/seathru_default.yaml
    DEFAULT_SCENES=$ALL_SCENES; DEFAULT_SEEDS="42 43" ;;
  owin)
    VARIANTS=(_owin); SET_CONFIG=configs/seathru_default.yaml
    DEFAULT_SCENES=$ALL_SCENES; DEFAULT_SEEDS="42 43" ;;
  adameps)
    # Adam epsilon: with 1e-8 the colour/opacity steps of Gaussians with small gradients are damped (sqrt(v) ~ 1e-8)
    VARIANTS=(_eps15 _eps12); SET_CONFIG=configs/seathru_default.yaml
    DEFAULT_SCENES="Curasao-Undistorted Panama-Undisorted"; DEFAULT_SEEDS="42 43" ;;
  fitlr)
    # training-photo fit vs WaterSplatting (colour learning rates, densification) on the two scenes with the
    # largest training-PSNR gap
    VARIANTS=(_aconst _wsapp _shr20 _dth10 _dint150); SET_CONFIG=configs/seathru_default.yaml
    DEFAULT_SCENES="Curasao-Undistorted Panama-Undisorted"; DEFAULT_SEEDS="42 43" ;;
  epivis_wanc)
    VARIANTS=(_noclear_epivis_wanc03); SET_CONFIG=configs/seathru_default.yaml
    DEFAULT_SCENES="$ALL_SCENES garden_easy garden_hard"; DEFAULT_SEEDS="42 43" ;;
  unify)
    VARIANTS=(_noclear_noocean_wanc03 _noclear_noepi_wanc03); SET_CONFIG=configs/seathru_default.yaml
    DEFAULT_SCENES=$ALL_SCENES; DEFAULT_SEEDS="42 43" ;;
  garden_anchor2)
    VARIANTS=(_noclear_noocean_val8 _noclear_noocean_wanc_val8 _noclear_noocean_wanc03_val8); SET_CONFIG=configs/seathru_default.yaml
    DEFAULT_SCENES="garden_easy garden_hard"; DEFAULT_SEEDS="42" ;;
  foggy)
    VARIANTS=(""); SET_CONFIG=configs/seathru_default.yaml
    DEFAULT_SCENES="garden_easy garden_hard"; DEFAULT_SEEDS="42 43" ;;
  alias2)
    VARIANTS=(_clamp05_minit_rs_rel_oceane_jit _clamp05_minit_rs_rel_oceane_mip
              _clamp05_minit_rs_rel_oceane_jit_mip _clamp05_minit_rs_oceane)
    DEFAULT_SCENES=$ALL_SCENES; DEFAULT_SEEDS="42 43" ;;
  diag)
    VARIANTS=(_minit_cut10k _minit_nooverlap_cut10k _minit_nochroma_cut10k _minit_nocover_cut10k
              _minit_noclear_cut10k _minit_nobin_cut10k _minit_nosmooth_cut10k _plain_minit_cut10k)
    DEFAULT_SCENES="JapaneseGradens-RedSea-Undistorted Curasao-Undistorted"; DEFAULT_SEEDS="42" ;;
  diag_mdc)
    VARIANTS=(_minit_mdc_cut10k)
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
      # Training of N thousand iterations with every schedule scaled by the same factor (training/schedule.py).
      echo "schedule.iterations=$(( BASH_REMATCH[1] * 1000 ))"; rest=${BASH_REMATCH[2]}
    elif [[ "$rest" =~ ^_cut([0-9]+)k(.*)$ ]]; then
      # Training cut at N thousand iterations: the schedules that end at 30000 end with it (densification
      # still stops at 15000). The diag set's _it10k before 2026-10-29.
      echo "joint.optimization.iterations=$(( BASH_REMATCH[1] * 1000 ))"
      echo "joint.gaussian_adaptation.opacity_prune_until=$(( BASH_REMATCH[1] * 1000 ))"; rest=${BASH_REMATCH[2]}
    elif [[ "$rest" =~ ^_oe([0-9]+)(.*)$ ]]; then
      echo "joint.optimization.ocean_every=${BASH_REMATCH[1]}"; rest=${BASH_REMATCH[2]}
    elif [[ "$rest" =~ ^_val([0-9]+)(.*)$ ]]; then
      # evaluate on every N-th training photo (held out) instead of the test views: tuning only
      echo "warmup.dataset.validation_hold=${BASH_REMATCH[1]}"
      echo "joint.dataset.validation_hold=${BASH_REMATCH[1]}"; rest=${BASH_REMATCH[2]}
    elif [[ "$rest" =~ ^_wphase(.*)$ ]]; then
      # Henyey-Greenstein phase function on the backscatter (training/water_phase.py)
      echo "joint.optimization.water_phase=true"; rest=${BASH_REMATCH[1]}
    elif [[ "$rest" =~ ^_wdir(.*)$ ]]; then
      # direction-dependent water light (training/water_direction.py)
      echo "joint.optimization.water_direction=true"; rest=${BASH_REMATCH[1]}
    elif [[ "$rest" =~ ^_owin(.*)$ ]]; then
      # OceanSplat window as in the paper (4k-8k of 10k): 12k-24k of the 30k schedule = 8k-16k at the 20k default
      echo "joint.optimization.ocean_from=12000"
      echo "joint.optimization.ocean_until=24000"; rest=${BASH_REMATCH[1]}
    elif [[ "$rest" =~ ^_eps([0-9]+)(.*)$ ]]; then
      # Adam epsilon of the Gaussian parameters 1e-N (default 1e-8; WaterSplatting / 3DGS: 1e-15)
      echo "joint.optimization.gaussian_adam_eps=1.0e-${BASH_REMATCH[1]}"; rest=${BASH_REMATCH[2]}
    elif [[ "$rest" =~ ^_aconst(.*)$ ]]; then
      # Gaussian colour learning rate kept at its start value (no decay)
      echo "joint.optimization.gaussian_appearance_lr_decay=0.0"; rest=${BASH_REMATCH[1]}
    elif [[ "$rest" =~ ^_wsapp(.*)$ ]]; then
      # WaterSplatting's colour learning rates: DC 0.0025 constant, higher SH degrees x 1/20
      echo "joint.optimization.gaussian_appearance_lr=0.0025"
      echo "joint.optimization.gaussian_appearance_lr_decay=0.0"
      echo "joint.optimization.gaussian_appearance_rest_lr_scale=0.05"; rest=${BASH_REMATCH[1]}
    elif [[ "$rest" =~ ^_shr20(.*)$ ]]; then
      # higher SH degrees learn 20x slower than the DC colour (WaterSplatting's ratio)
      echo "joint.optimization.gaussian_appearance_rest_lr_scale=0.05"; rest=${BASH_REMATCH[1]}
    elif [[ "$rest" =~ ^_dint([0-9]+)(.*)$ ]]; then
      # densification interval before schedule scaling (the 20k default scales 30k-schedule values by 2/3)
      echo "joint.gaussian_adaptation.densification_interval=${BASH_REMATCH[1]}"; rest=${BASH_REMATCH[2]}
    elif [[ "$rest" =~ ^_dth([0-9]+)(.*)$ ]]; then
      echo "joint.gaussian_adaptation.gradient_threshold=${BASH_REMATCH[1]}.0e-5"; rest=${BASH_REMATCH[2]}
    elif [[ "$rest" =~ ^_noocean(.*)$ ]]; then
      echo "joint.optimization.ocean_tri_weight=0.0"
      echo "joint.optimization.ocean_epi_weight_start=0.0"
      echo "joint.optimization.ocean_epi_weight_end=0.0"
      echo "joint.optimization.ocean_res_weight=0.0"
      echo "joint.optimization.ocean_alpha_until=0"; rest=${BASH_REMATCH[1]}
    elif [[ "$rest" =~ ^_noalpha(.*)$ ]]; then
      echo "joint.optimization.ocean_alpha_until=0"; rest=${BASH_REMATCH[1]}
    elif [[ "$rest" =~ ^_notri(.*)$ ]]; then
      echo "joint.optimization.ocean_tri_weight=0.0"; rest=${BASH_REMATCH[1]}
    elif [[ "$rest" =~ ^_noepi(.*)$ ]]; then
      echo "joint.optimization.ocean_epi_weight_start=0.0"
      echo "joint.optimization.ocean_epi_weight_end=0.0"
      echo "joint.optimization.ocean_res_weight=0.0"; rest=${BASH_REMATCH[1]}
    elif [[ "$rest" =~ ^_epivis(.*)$ ]]; then
      echo "joint.optimization.ocean_visible=0.1"; rest=${BASH_REMATCH[1]}
    elif [[ "$rest" =~ ^_classic([0-9]*)(.*)$ ]]; then
      # mip filter as a dilation only (opacity kept, 3DGS "classic" as WaterSplatting); _classic3 = variance 0.3
      echo "joint.model.gaussian_mip_compensate=false"
      [[ -n "${BASH_REMATCH[1]}" ]] && echo "joint.model.gaussian_mip_filter=0.${BASH_REMATCH[1]}"
      rest=${BASH_REMATCH[2]}
    elif [[ "$rest" =~ ^_nomip(.*)$ ]]; then
      echo "joint.model.gaussian_mip_filter=0.0"; rest=${BASH_REMATCH[1]}
    elif [[ "$rest" =~ ^_wanc([0-9]*)(.*)$ ]]; then
      # _wanc = weight 0.1, _wanc<digits> = 0.<digits> (e.g. _wanc03 = 0.03)
      echo "medium_fix.water_anchor_weight=0.${BASH_REMATCH[1]:-1}"; rest=${BASH_REMATCH[2]}
    elif [[ "$rest" =~ ^_camw(.*)$ ]]; then
      echo "medium_fix.camera_water_weight=0.01"; rest=${BASH_REMATCH[1]}
    elif [[ "$rest" =~ ^_isp(.*)$ ]]; then
      echo "joint.optimization.camera_isp=true"; rest=${BASH_REMATCH[1]}
    elif [[ "$rest" =~ ^_jit(.*)$ ]]; then
      echo "joint.optimization.ray_jitter=true"; rest=${BASH_REMATCH[1]}
    elif [[ "$rest" =~ ^_mip(.*)$ ]]; then
      echo "joint.model.gaussian_mip_filter=0.2"; rest=${BASH_REMATCH[1]}
    elif [[ "$rest" =~ ^_rs(.*)$ ]]; then
      echo "joint.optimization.training_downscale_levels=2"
      echo "joint.optimization.training_downscale_interval=6000"; rest=${BASH_REMATCH[1]}
    elif [[ "$rest" =~ ^_oceane(.*)$ ]]; then
      echo "joint.optimization.ocean_tri_weight=0.1"
      echo "joint.optimization.ocean_epi_weight_start=0.4"
      echo "joint.optimization.ocean_epi_weight_end=0.2"
      echo "joint.optimization.ocean_res_weight=0.01"
      echo "joint.optimization.ocean_alpha_until=12000"
      echo "joint.optimization.ocean_from=4000"; rest=${BASH_REMATCH[1]}
    elif [[ "$rest" =~ ^_ocean(.*)$ ]]; then
      echo "joint.optimization.ocean_tri_weight=0.1"
      echo "joint.optimization.ocean_epi_weight_start=0.4"
      echo "joint.optimization.ocean_epi_weight_end=0.2"
      echo "joint.optimization.ocean_res_weight=0.01"
      echo "joint.optimization.ocean_alpha_until=12000"; rest=${BASH_REMATCH[1]}
    elif [[ "$rest" =~ ^_gdkw(.*)$ ]]; then
      echo "joint.gaussian_adaptation.near_camera_gradient_scaling=true"; rest=${BASH_REMATCH[1]}
    elif [[ "$rest" =~ ^_gdk(.*)$ ]]; then
      echo "joint.gaussian_adaptation.near_camera_gradient_scaling=true"
      echo "joint.gaussian_adaptation.near_camera_steps=[4000, 7000]"; rest=${BASH_REMATCH[1]}
    elif [[ "$rest" =~ ^_tri(.*)$ ]]; then
      echo "joint.optimization.trinocular_weight=0.1"; rest=${BASH_REMATCH[1]}
    elif [[ "$rest" =~ ^_wfitc(.*)$ ]]; then
      echo "joint.model.medium_initial_water_fit=all"; rest=${BASH_REMATCH[1]}
    elif [[ "$rest" =~ ^_wfit(.*)$ ]]; then
      echo "joint.model.medium_initial_water_fit=coefficients"; rest=${BASH_REMATCH[1]}
    elif [[ "$rest" =~ ^_cmax([0-9]+)(.*)$ ]]; then
      echo "joint.gaussian_adaptation.gradient_compensation_max=${BASH_REMATCH[1]}.0"; rest=${BASH_REMATCH[2]}
    elif [[ "$rest" =~ ^_gt([0-9]+)(.*)$ ]]; then
      echo "joint.gaussian_adaptation.gradient_threshold=${BASH_REMATCH[1]}.0e-4"; rest=${BASH_REMATCH[2]}
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
    elif [[ "$rest" =~ ^_bin01(.*)$ ]]; then
      # opacity binarisation prior x10 (0.001 -> 0.01): against the half-transparent seabed fuzz
      echo "joint.disentanglement.opacity_binarization_weight=0.01"; rest=${BASH_REMATCH[1]}
    elif [[ "$rest" =~ ^_kfix(.*)$ ]]; then
      # kernel shape frozen at its start, kappa = 1.01 (plain Gaussians, as WaterSplatting)
      echo "joint.optimization.gaussian_kappa_lr=1.0e-12"
      echo "joint.optimization.gaussian_kappa_lr_final=1.0e-12"; rest=${BASH_REMATCH[1]}
    elif [[ "$rest" =~ ^_cov([0-9])([0-9])(.*)$ ]]; then
      # seabed coverage prior from accumulated alpha 0.<NN> on (default 0.9), weight x10 (0.03); _cov05 pulls
      # water rays with a few stray Gaussians to opaque, so higher thresholds keep the water clean
      echo "medium_fix.surface_coverage_threshold=${BASH_REMATCH[1]}.${BASH_REMATCH[2]}"
      echo "medium_fix.surface_coverage_weight=0.03"; rest=${BASH_REMATCH[3]}
    elif [[ "$rest" =~ ^_vt(.*)$ ]]; then
      # evaluation rays end at the training transmittance threshold (1e-4) instead of 1e-2
      echo "joint.renderer.validation_transmittance_threshold=1.0e-4"; rest=${BASH_REMATCH[1]}
    elif [[ "$rest" =~ ^_reset(.*)$ ]]; then
      # WaterSplatting's opacity reset alone (to 0.5 every 500 steps = 750 of the 30k schedule), culling unchanged
      echo "joint.gaussian_adaptation.opacity_reset_interval=750"
      echo "joint.gaussian_adaptation.opacity_reset_value=0.5"; rest=${BASH_REMATCH[1]}
    elif [[ "$rest" =~ ^_cull05(r?)(.*)$ ]]; then
      # WaterSplatting's opacity culling only: Gaussians below 0.5 removed while densifying (to the end of our
      # densification, 15000 of the 30k schedule = 10000 at 20k), below 0.1 after; _cull05r also resets the
      # opacity to 0.5 every 500 steps (750 of the 30k schedule) as WaterSplatting
      echo "joint.gaussian_adaptation.min_opacity=0.5"
      echo "joint.gaussian_adaptation.min_opacity_initial=0.5"
      echo "joint.gaussian_adaptation.opacity_prune_ramp=0"
      echo "joint.gaussian_adaptation.min_opacity_late=0.1"
      echo "joint.gaussian_adaptation.min_opacity_late_from=15000"
      echo "joint.gaussian_adaptation.max_opacity_prune_fraction=1.0"
      if [[ -n "${BASH_REMATCH[1]}" ]]; then
        echo "joint.gaussian_adaptation.opacity_reset_interval=750"
        echo "joint.gaussian_adaptation.opacity_reset_value=0.5"
      fi
      rest=${BASH_REMATCH[2]}
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
[[ "$FOGGY_SCENES" == *" $SCENE "* ]] && SOURCE="$FOGGY/$SCENE"
[[ "$SCENE" == garden_clear ]] && SOURCE="$GARDEN_CLEAR"
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
  --config "${FS_CONFIG:-${SET_CONFIG:-configs/iui3_default.yaml}}"
  --output-dir "$OUTPUT"
  --run-suffix "$NAME"
  --set "warmup.dataset.source_path=$SOURCE"
  --set "joint.dataset.source_path=$SOURCE"
  --set "variant.seed=$SEED" --set "warmup.seed=$SEED" --set "joint.seed=$SEED"
  --set "warmup.wandb.mode=$WANDB" --set "joint.wandb.mode=$WANDB"
)
for item in "${OVERRIDES[@]}"; do ARGS+=(--set "$item"); done
if [[ "${FS_RESUME:-0}" == 1 ]]; then ARGS+=(--resume); fi

echo "scene=$SCENE variant=${VARIANT:-none} seed=$SEED config=${ARGS[1]} output=$OUTPUT"
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
