# RadFoam as a spatial surface/medium prior for FoamSurf

## 1. Purpose of this document

This document is a complete handoff for implementing and running the next
FoamSurf experiment. It is intended to be understandable without access to the
preceding chat.

The immediate research question is:

> Can a normally trained, single-representation RadFoam model reveal—before
> the actual FoamSurf training—which parts of 3D space behave like solid
> surfaces, which parts behave like participating medium, and which parts are
> empty?

The experiment is motivated by a persistent failure in the best current
FoamSurf disentanglement runs. Opacity binarization, aggressive Gaussian
pruning, and a medium-only warmup move most clouds from the 2D Gaussian branch
to the Voronoi medium branch. Nevertheless, one compact cloud is repeatedly
reconstructed by the Gaussian branch in nearly every good run. It appears to
be sufficiently compact and sharp that 2D Gaussian optimization treats it as a
pseudo-surface.

The proposed solution is not yet another global loss. Instead, we want to run
RadFoam first as a scene-probing/teacher model, inspect its volumetric field at
multiple points in training, and determine whether it contains enough
information to construct a frozen 3D prior:

```text
SURFACE  -> use/allow 2D Gaussians
MEDIUM   -> use/allow FoamSurf Voronoi medium
EMPTY    -> use neither representation
```

The separated `lego_lego` and `lego_clouds` datasets must not supervise the
RadFoam training in this experiment. They are privileged ground truth used to
measure whether a proposed checkpoint separation is correct.

This document specifies the diagnostic RadFoam experiment. Do not yet integrate
the resulting prior into FoamSurf training. First prove that the information is
present and separable.

## 2. Scientific context and required interpretation

FoamSurf has two representations with deliberately different responsibilities:

- 2D Gaussians should represent opaque solid geometry, sharp boundaries,
  texture, and view-dependent appearance on solid objects.
- Voronoi cells should represent participating medium: absorption, scattering,
  backscatter, turbidity, smooth volumetric density changes, and low-frequency
  color shifts introduced by the medium.
- There is no visible water surface in the target scenes, so a water interface
  does not need special handling.

Previous sweeps showed that Gaussians learn faster and tend to take over both
solid objects and clouds. The current mechanisms—opacity binarization and
pruning—can remove many cloud Gaussians, especially when preceded by a
medium-only warmup, but do not reliably remove one particular cloud.

The hypothesis for this experiment is that RadFoam can act as a pretraining
probe. RadFoam represents the whole scene as a volumetric Voronoi radiance
field. Ideally, solid surfaces and diffuse media produce different spatial and
ray-wise statistics:

### Expected surface-like behavior

- optical opacity concentrated in a narrow depth interval;
- a single cell or a short sequence of cells dominates a ray;
- stable depth across nearby views;
- a strong local density transition;
- small cells/sites concentrated around a thin 3D manifold;
- a relatively small distance between opacity depth quantiles such as
  `d10` and `d90`.

### Expected medium-like behavior

- opacity distributed over a longer ray interval;
- several cells contribute nontrivially to the same ray;
- no single stable, thin surface depth;
- smoother density changes over neighboring cells;
- a larger distance between opacity depth quantiles;
- consistent nonzero image contribution despite low surface confidence.

### Expected empty-space behavior

- negligible optical thickness;
- negligible accumulated contribution across views;
- no consistent target alpha in either separated ground-truth layer.

The goal is not to assume that raw RadFoam density is physical material
density. A single global `sigma` threshold is only a baseline and is not the
desired final method. Density is coupled to cell size and ray path length. The
relevant quantity for a ray segment is optical thickness:

\[
\tau_i = \sigma_i \Delta s_i,
\]

and its opacity is:

\[
\alpha_i = 1 - \exp(-\tau_i).
\]

A large cell with moderate density can be more opaque than a very small cell
with a larger density. Conversely, a compact cloud can acquire a large learned
density. Therefore the analysis must use contribution, geometry, neighborhood,
and ray-depth concentration in addition to `sigma`.

## 3. Repository, environment, data, and hardware

### 3.1 Main FoamSurf repository

```text
/shared/results/gmwilczn/foamsurf/foamsurf
```

Important context files:

```text
/shared/results/gmwilczn/foamsurf/foamsurf/context_cornerstone.md
/shared/results/gmwilczn/foamsurf/foamsurf/docs/QUALIFIED_BASELINES_AND_INTEGRATION_PLAN.md
/shared/results/gmwilczn/foamsurf/foamsurf/docs/DISENTANGLEMENT_PHASE1.md
/shared/results/gmwilczn/foamsurf/foamsurf/docs/DISENTANGLEMENT_GAUSSIAN_EVICTION_SWEEP.md
/shared/results/gmwilczn/foamsurf/foamsurf/docs/DISENTANGLEMENT_MEDIUM_HANDOFF_SWEEP.md
```

The canonical renderer settings used in FoamSurf come from:

```text
/shared/results/gmwilczn/foamsurf/foamsurf/configs/baseline_best_2d_gaussians.yaml
```

The clean joint configuration and layer paths are in:

```text
/shared/results/gmwilczn/foamsurf/foamsurf/configs/baseline_clean_joint.yaml
```

The best medium-handoff sweep results are under:

```text
/shared/results/gmwilczn/foamsurf/foamsurf/outputs/medium_handoff
```

Previous candidate run IDs of interest were 30, 41, and 61 in the Gaussian
eviction sweep; the subsequent medium-handoff sweep contains the warmup/freeze
experiments based on these configurations. This RadFoam experiment is motivated
by visual inspection of those runs, but it must not alter their outputs.

At the time this handoff was written, the FoamSurf worktree contained existing
tracked and untracked changes from the disentanglement implementation and
sweeps. These are user work. Do not reset, discard, overwrite, or broadly
reformat them. Inspect `git status --short` before editing.

### 3.2 RadFoam repository to modify and run

```text
/shared/results/gmwilczn/foamsurf/radfoam
```

This is based on upstream RadFoam commit:

```text
3e7b52cf74e37ab2ab5e695f53570f515f537e3d
```

It already contains local changes for Blender synthetic initialization,
evaluation, timing, and Slurm execution. The worktree is dirty. In particular,
local modifications/untracked files include at least:

```text
configs/__init__.py
configs/mipnerf360_indoor.yaml
radfoam_model/scene.py
requirements.txt
train.py
configs/nerf_synthetic_lego.yaml
evaluate.py
radfoam_lego_initializations.sbatch
scripts/compare_lego_initializations.py
scripts/smoke_test_training.py
```

Preserve all existing changes. Do not reset RadFoam to upstream.

Relevant RadFoam files:

```text
/shared/results/gmwilczn/foamsurf/radfoam/train.py
/shared/results/gmwilczn/foamsurf/radfoam/evaluate.py
/shared/results/gmwilczn/foamsurf/radfoam/test.py
/shared/results/gmwilczn/foamsurf/radfoam/configs/__init__.py
/shared/results/gmwilczn/foamsurf/radfoam/configs/nerf_synthetic_lego.yaml
/shared/results/gmwilczn/foamsurf/radfoam/data_loader/blender.py
/shared/results/gmwilczn/foamsurf/radfoam/radfoam_model/scene.py
/shared/results/gmwilczn/foamsurf/radfoam/radfoam_model/render.py
/shared/results/gmwilczn/foamsurf/radfoam/src/tracing/pipeline.cu
/shared/results/gmwilczn/foamsurf/radfoam/radfoam_lego_initializations.sbatch
```

### 3.3 Conda environment

Use the existing `foamsurf` environment for both repositories:

```text
/shared/results/gmwilczn/conda/envs/foamsurf
```

For interactive work:

```bash
conda activate /shared/results/gmwilczn/conda/envs/foamsurf
```

For scripts and Slurm, prefer the absolute Python executable:

```text
/shared/results/gmwilczn/conda/envs/foamsurf/bin/python
```

Do not install a new environment unless the existing environment is proven
insufficient. Do not silently switch to `base`.

### 3.4 GPU and scheduler

The current interactive node has a GPU and can be used for GPU smoke tests. At
the time of writing it reports:

```text
NVIDIA GeForce RTX 4090
24,564 MiB
driver 610.43.02
```

Long training and full checkpoint analysis should still be supplied as Slurm
jobs. Existing project jobs use:

```text
partition: rtx4090_batch
qos: batch
gres: gpu:1
cpus-per-task: 8
exclude: c11,c22
```

Use at least 32 GB host memory; 48 GB is safer for the complete analysis. A
12-hour limit was sufficient for earlier RadFoam jobs. One previous 20k
RadFoam training took approximately 4,157 seconds (about 69 minutes), but that
number is only a reference.

### 3.5 Datasets

The correct full-scene training dataset is:

```text
/shared/sets/datasets/foamsurf/lego/lego_both
```

The separated evaluation-only layers are:

```text
/shared/sets/datasets/foamsurf/lego/lego_lego
/shared/sets/datasets/foamsurf/lego/lego_clouds
```

All three are Blender/NeRF-style RGBA datasets with:

```text
transforms_train.json
transforms_test.json
train/r_*.png
test/r_*.png
```

Verified dataset properties:

- 200 training views per layer;
- 50 test views per layer;
- 800 x 800 native resolution;
- frame ordering and camera transform matrices are identical across
  `lego_both`, `lego_lego`, and `lego_clouds` for both splits;
- the cloud alpha is genuinely translucent (for an inspected view its maximum
  was approximately 0.6745), so alpha-aware metrics are mandatory.

The RadFoam `DataHandler` constructs a path as:

```python
os.path.join(data_path, scene)
```

Therefore the new RadFoam config must use:

```yaml
dataset: blender
data_path: /shared/sets/datasets/foamsurf/lego
scene: lego_both
```

Do not train this experiment on:

```text
/home/gmwilczn/nerf_synth_mipnerf/lego
```

That is an older/different Lego dataset. An existing checkpoint at:

```text
/shared/results/gmwilczn/foamsurf/radfoam/output/lego_radfoam_synthetic_bbox_448307
```

was trained using that older path and contains only the final checkpoint. It is
useful as evidence that the local RadFoam port runs, but it does not answer the
current question and must not replace the new `lego_both` run.

## 4. Critical background-compositing issue to fix first

FoamSurf trains this scene with a black background:

```yaml
background: [0.0, 0.0, 0.0]
```

FoamSurf stores premultiplied foreground and composes it as:

```python
foreground = rgb * alpha
target = foreground + background * (1 - alpha)
```

The current RadFoam Blender loader instead always produces white-background
targets:

```python
rgbs = rgb * alpha + (1 - alpha)
```

This happens regardless of the `white_background` setting later used in
`train.py`. Merely setting `white_background: false` in the YAML is therefore
not sufficient: the current loader would still supply white-composited targets
while the renderer would be interpreted as black-composited.

Before the experiment, make background behavior explicit and consistent in all
of the following:

- RadFoam Blender training target construction;
- built-in training validation;
- standalone `evaluate.py`;
- separated-layer evaluator;
- saved visualization panels.

For this experiment use a black background and the direct premultiplied target:

```python
target_rgb = rgba[..., :3] * rgba[..., 3:4]
```

Retain alpha separately. The implementation should be configurable rather than
hard-coding black and should preserve legacy white-background behavior for old
configs.

Add a test that constructs a known RGBA pixel and verifies exact black and
white composition. Without this fix, do not trust PSNR or the separated renders.

## 5. Required RadFoam training run

### 5.1 Base configuration

Create a dedicated config rather than changing the meaning of the existing
Lego config. A suggested location is:

```text
/shared/results/gmwilczn/foamsurf/radfoam/configs/nerf_synthetic_lego_both_prior.yaml
```

Start from the local configuration:

```text
/shared/results/gmwilczn/foamsurf/radfoam/configs/nerf_synthetic_lego.yaml
```

Keep the initial experiment deliberately close to vanilla RadFoam:

```yaml
sh_degree: 3
init_points: 131072
final_points: 2097152
activation_scale: 1.0
init_method: synthetic_bbox
synthetic_bbox_min: [-1.5, -1.5, -1.5]
synthetic_bbox_max: [1.5, 1.5, 1.5]
synthetic_outer_points: 5000
synthetic_outer_scale: 10.0
synthetic_inner_density: 0.0
synthetic_outer_density: -0.5
device: cuda

iterations: 20000
densify_from: 2000
densify_until: 11000
densify_factor: 1.15
quantile_weight: 1.0e-4
eval_interval: 1000
eval_num_views: 8
viewer: false
debug: false

points_lr_init: 2.0e-4
points_lr_final: 5.0e-6
density_lr_init: 1.0e-1
density_lr_final: 1.0e-2
attributes_lr_init: 5.0e-3
attributes_lr_final: 5.0e-4
sh_factor: 0.1
freeze_points: 18000

dataset: blender
data_path: /shared/sets/datasets/foamsurf/lego
scene: lego_both
patch_based: false
downsample: [4, 2, 1]
downsample_iterations: [0, 150, 500]
white_background: false
```

Do not introduce a large hyperparameter sweep yet. First determine whether one
normal training trajectory contains a separable signal.

### 5.2 Intermediate checkpoint schedule

The current `train.py` only writes a final `model.pt`. Extend it with an explicit
configuration/CLI option for analysis checkpoint iterations. The suggested
schedule is:

```text
500, 1000, 2000, 4000, 6000, 8000, 11000, 15000, 18000, 20000
```

These points cover:

- very early field formation;
- the beginning of densification at 2k;
- several stages during densification;
- the end of densification at 11k;
- later density/appearance refinement;
- point freezing at 18k;
- the final 20k model.

Save analysis snapshots as:

```text
<run>/checkpoints/iter_00000500.pt
<run>/checkpoints/iter_00001000.pt
...
<run>/checkpoints/iter_00020000.pt
```

Use `RadFoamScene.save_pt()`, which currently stores:

```text
xyz
density                  # raw density parameter, before activation
color_dc
color_sh
adjacency
adjacency_offsets
```

Requirements:

- save after completing the selected training iteration;
- ensure the stored adjacency corresponds exactly to the current site order;
- perform atomic writing via a temporary file followed by rename, so a killed
  job cannot leave a valid-looking partial checkpoint;
- also retain the conventional final `<run>/model.pt`;
- do not write a full PLY at every intermediate checkpoint—the final PLY is
  already very large and intermediate PLYs are unnecessary;
- write a manifest containing iteration, point count, config path, elapsed
  time, and checkpoint path;
- no optimizer state is required for this diagnostic analysis unless resume
  support is deliberately added as a separate feature.

Expect several GB of checkpoints. The existing final two-million-point
`model.pt` is approximately 636 MB.

### 5.3 Important caveat about RadFoam's quantile loss

The local RadFoam training samples two opacity quantiles per ray and minimizes:

\[
L_q = |d(q_1)-d(q_2)|.
\]

Its weight ramps during training. This encourages opacity to concentrate at a
single depth, which is useful for solid surfaces but can collapse a real cloud
into a pseudo-surface. This is exactly why intermediate checkpoints matter.

Do not disable the quantile loss in the first run: the first question concerns
normal RadFoam. If the final checkpoint is entangled but an earlier checkpoint
is separable, that is already a successful result. A second run with
`quantile_weight: 0` or a weaker value is a later ablation, only if necessary.

## 6. Separation and rendering architecture

Implement the analyzer as a standalone, inference-only pipeline. Do not mix
layer-ground-truth logic into `train.py`.

Suggested entry point:

```text
/shared/results/gmwilczn/foamsurf/radfoam/scripts/analyze_checkpoint_separation.py
```

It should accept at least:

```text
--checkpoint
--config
--full-dataset
--surface-dataset
--medium-dataset
--split train|test
--output-dir
--classifier
--surface-threshold
--medium-threshold
--alpha-threshold
--downsample
```

The script should be reusable for every intermediate checkpoint and should not
modify the checkpoint in place.

### 6.1 Do not delete cells or rebuild topology for isolated renders

Removing points would change the Delaunay/Voronoi topology and all neighboring
cell shapes. Such a render would no longer isolate a subset of the original
trained representation.

For every isolated render:

- keep all primal points;
- keep the original adjacency and offsets;
- keep the original colors;
- keep the original AABB tree;
- multiply the activated per-cell density by a cell mask before calling the
  renderer.

For a binary mask `m_i`:

\[
\sigma_i^{masked} = m_i \sigma_i.
\]

Do not emulate zero density by setting the raw softplus parameter to an
arbitrary negative number. It is approximate, mutates model state, and can
introduce checkpoint-dependent residual opacity. Add a clean trace-data or
rendering path that accepts an activated density multiplier/override.

The following invariants must hold numerically:

- an all-ones mask produces the same RGBA as an unmasked render;
- an all-zero mask produces zero premultiplied RGB and zero alpha;
- masks never reorder cells;
- masks never trigger triangulation reconstruction;
- analysis does not modify model parameters.

### 6.2 Required render modes

For each checkpoint and classifier produce:

```text
full
surface_only
medium_only
ambiguous_only
empty_only                   # primarily a leakage/debug render
surface_plus_medium
classification_debug
```

`surface_plus_medium` uses the union of both masks in one renderer invocation.
Do not add independently rendered surface and medium RGB images: volume
compositing is not additive because transmittance and occlusion depend on
ordering.

`classification_debug` should provide an interpretable visualization, for
example surface contribution in red, medium in blue, ambiguous in yellow, and
empty in black.

Save raw alpha maps as lossless images or NumPy arrays as well as visible RGB
panels. The visible panel should contain at least:

```text
full target | full prediction | surface target | surface-only prediction |
medium target | medium-only prediction | absolute errors | alpha maps
```

## 7. Two distinct separation tracks

The implementation must keep `oracle` and `checkpoint-only` classifiers
explicitly separate. Results from them answer different questions and must not
be mixed in tables.

### 7.1 Track A: oracle cell attribution and oracle upper bound

Purpose:

> Determine whether the trained RadFoam checkpoint internally contains cells
> that can be separated into solid and medium at all.

Use the separated layers only to construct ray masks for attribution. Derive
masks from RGBA alpha, not RGB brightness, because clouds may be dark or
translucent.

For an alpha threshold `eps`, define:

```text
solid_only  = alpha_lego   > eps and alpha_cloud <= eps
medium_only = alpha_cloud  > eps and alpha_lego  <= eps
overlap     = alpha_lego   > eps and alpha_cloud >  eps
empty       = alpha_lego  <= eps and alpha_cloud <= eps
```

Start with `eps = 0.01`, but report sensitivity to a few nearby values because
cloud alpha is continuous.

The existing RadFoam renderer already supports `return_contribution=True`. In
the CUDA kernel, per-point contribution is accumulated using `atomicAdd`, so a
render of a selected set of rays gives a sum of actual compositing weights for
each cell. This is directly useful.

For each cell `i`, accumulate over training views:

```text
C_solid[i]   = contribution on solid-only rays
C_medium[i]  = contribution on medium-only rays
C_overlap[i] = contribution on overlap rays
C_empty[i]   = contribution on empty rays
C_all[i]     = contribution on all rays
```

Normalize each accumulation by the number of rays in that category so that a
large pixel category does not automatically dominate a small cloud category.
Also retain the unnormalized sums because they express total rendered mass.

Define a basic responsibility ratio:

\[
r_i = \frac{\bar C_i^{solid}}
{\bar C_i^{solid} + \bar C_i^{medium} + \epsilon}.
\]

A starting oracle rule is:

```text
inactive/empty: total relevant contribution below an activity threshold
surface:        active and r_i >= t_surface
medium:         active and r_i <= t_medium
ambiguous:      active and t_medium < r_i < t_surface
```

Evaluate several pairs such as:

```text
(t_medium, t_surface) =
(0.10, 0.90),
(0.20, 0.80),
(0.30, 0.70),
(0.40, 0.60)
```

This is a small analysis threshold sweep, not a new training sweep.

Use `train` layer masks to derive oracle cell labels and `test` views to measure
the resulting isolated renders. Do not assign cells using test targets and then
report the same test targets as generalization evidence.

The overlap category must remain separate in the first analysis. Do not
silently force every overlap cell to surface or medium. Report how much
contribution is ambiguous. A high ambiguous fraction means that the RadFoam
topology itself does not spatially resolve the two layers.

Interpretation:

- Good oracle separation means the checkpoint contains a usable signal and the
  remaining problem is finding an unsupervised classifier.
- Poor oracle separation means the checkpoint's cells jointly explain both
  layers, so no thresholding of cell features can recover a clean split. The
  representation, topology, or training setup must then change.

The oracle classifier is a diagnostic upper bound. It is not an admissible
final FoamSurf prior because it uses layer supervision.

### 7.2 Track B: checkpoint-only, unsupervised separation

Purpose:

> Determine whether the same split can be inferred from the RadFoam checkpoint
> and full-scene cameras alone, without `lego_lego` or `lego_clouds`.

Extract a per-cell feature table. At minimum include:

```text
cell_index
x, y, z
raw_density
activated_sigma
cell_radius or a documented geometry proxy
log_sigma
tau_proxy = sigma * 2 * cell_radius
all-ray contribution
visibility/ray count if available
contribution per visible ray
number of neighbors
neighbor sigma mean/median/std
absolute and relative difference from neighbor sigma
local graph gradient / TV term
color DC
color/SH norm
```

`cell_radius` can use the existing farthest-neighbor geometry routine already
used by RadFoam densification, provided its exact meaning is documented. It is
only a proxy for path length, not an exact segment length.

Also produce ray-wise maps for the full test views:

```text
opacity
expected or median depth
d10
d50
d90
d90 - d10
maximum-cell concentration if available
number/effective number of contributing cells if available
```

The current renderer accepts arbitrary depth quantiles and returns their depths.
It does not expose all traversal IDs through the public Python output. Start by
using quantile width as a ray-level diagnostic. Only extend the CUDA/Python API
to expose quantile cell IDs or additional per-cell path statistics if the
existing cell features and ray maps are insufficient. Do not assume that the
current public API already returns a complete cell sequence.

Start with interpretable classifiers:

1. `sigma_threshold`: raw density-only baseline.
2. `tau_threshold`: optical-thickness proxy baseline.
3. `local_concentration`: combines `tau_proxy`, local density contrast, cell
   radius, and contribution concentration.
4. `feature_clustering`: a simple standardized two- or three-component mixture
   or clustering model over log-scaled checkpoint features.

Do not begin with another neural classifier. First establish which physical or
geometric statistics carry the signal.

The layer datasets may be loaded only after predictions are fixed, for scoring
and plots. Any threshold selected using layer metrics must be clearly marked as
validation-selected/oracle-tuned, not fully unsupervised.

## 8. Metrics and evaluation protocol

### 8.1 Full-scene reconstruction

For every checkpoint report full RadFoam quality against `lego_both`:

- PSNR;
- SSIM;
- LPIPS if practical;
- RGB MAE;
- alpha MAE;
- point count.

High full-scene PSNR is necessary but not sufficient. A checkpoint with lower
full PSNR may be a better teacher if it produces a much cleaner surface/medium
split.

### 8.2 Correct branch reconstruction

Evaluate:

```text
surface_only vs lego_lego
medium_only  vs lego_clouds
```

Report:

- PSNR and SSIM on premultiplied RGB;
- alpha MAE;
- foreground-masked PSNR;
- alpha-weighted RGB MAE;
- optional LPIPS for the surface layer;
- cloud-layer PSNR/MAE with enough precision to compare checkpoints.

Do not rely only on RGB composed against a background. A transparent black
pixel and an opaque black pixel have identical RGB but radically different
meaning.

### 8.3 Cross-layer leakage

At minimum report:

```text
surface alpha/contribution on cloud-only pixels
medium alpha/contribution on Lego-only pixels
surface RGB error on cloud-only pixels
medium RGB error on Lego-only pixels
```

Lower leakage is the central goal.

### 8.4 Cell purity and ambiguity

For every checkpoint and separation method report:

- number and fraction of cells assigned to each class;
- fraction of total contribution assigned to each class;
- fraction of contribution belonging to ambiguous cells;
- distribution of `r_i` for the oracle track;
- cells with significant contribution to both surface-only and medium-only
  rays;
- feature histograms split by oracle responsibility, used only for diagnosis.

AUROC and average precision can be computed for checkpoint-only surface and
medium scores against high-confidence oracle cell labels. Do not force
ambiguous oracle cells into either binary class for these metrics.

### 8.5 Persistent-cloud ROI

Global averages can conceal the exact cloud that motivates the experiment.
Create an explicit ROI protocol for it.

Prefer a reproducible ROI generated from the cloud-only alpha component rather
than a hand-written rectangle tied to one view:

1. select connected components of `lego_clouds` alpha above a documented
   threshold;
2. identify the persistent component in one or more reference views;
3. save the selected view names, masks, component areas, and selection rule;
4. measure the corresponding component across all views in which it is visible.

If automatic cross-view component identity is initially difficult, store an
explicit list of reference view names and masks as a versioned diagnostic
artifact. Never rely on an undocumented crop.

Within the ROI report:

- fraction of target cloud alpha recovered by `medium_only`;
- fraction incorrectly recovered by `surface_only`;
- medium-only alpha MAE;
- surface leakage alpha;
- RGB PSNR/MAE;
- which cells dominate the incorrect surface render;
- whether those cells are pure surface, pure medium, or ambiguous according to
  oracle contribution.

### 8.6 Conservation and integrity checks

The cell labels must be mutually exclusive and exhaustive:

```text
surface | medium | ambiguous | empty = all cells
```

Also verify:

- no cell belongs to two classes;
- full unmasked rendering is bitwise or numerically unchanged by merely
  enabling diagnostics;
- all-ones masked rendering matches the unmasked rendering;
- the union render behaves as expected;
- isolated outputs are finite and alpha stays in `[0, 1]` within tolerance;
- metric view names match across all three datasets.

## 9. Expected output layout

Use a self-contained run directory such as:

```text
/shared/results/gmwilczn/foamsurf/radfoam/output/lego_both_prior_<JOB_ID>
```

Suggested structure:

```text
lego_both_prior_<JOB_ID>/
  config.yaml
  experiment_info.json
  training_time.json
  model.pt
  scene.ply
  checkpoints/
    manifest.json
    iter_00000500.pt
    iter_00001000.pt
    ...
    iter_00020000.pt
  separation/
    iter_00000500/
      cell_features.pt
      cell_features.csv_or_parquet
      oracle/
        classifier.json
        cell_labels.pt
        metrics.json
        metrics_per_view.csv
        renders/
      checkpoint_only/
        <classifier_name>/
          classifier.json
          cell_labels.pt
          metrics.json
          metrics_per_view.csv
          renders/
    ...
    summary.csv
    summary.json
    checkpoint_tradeoff_plots/
    persistent_cloud_roi/
  DONE
```

The summary must make checkpoint selection possible without opening every
directory manually.

## 10. Slurm deliverable

Create a new sbatch script in the RadFoam repository, for example:

```text
/shared/results/gmwilczn/foamsurf/radfoam/radfoam_checkpoint_separation.sbatch
```

It should:

1. use one RTX 4090;
2. activate/use `/shared/results/gmwilczn/conda/envs/foamsurf` by absolute path;
3. verify `torch.cuda.is_available()` and print GPU information;
4. validate all three dataset roots and transform files;
5. create a unique run directory based on `SLURM_JOB_ID`;
6. train exactly one RadFoam on `lego_both` and save intermediate checkpoints;
7. verify every scheduled checkpoint exists and is nonempty;
8. analyze all checkpoints in a deterministic order;
9. write a consolidated summary;
10. create `DONE` only after training and every required analysis succeed.

Recommended header:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=rf_sep_probe
#SBATCH --partition=rtx4090_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --qos=batch
#SBATCH --exclude=c11,c22
#SBATCH --time=12:00:00
```

Use `set -Eeuo pipefail`. Save stdout/stderr under:

```text
/shared/results/gmwilczn/foamsurf/radfoam/slurm_logs
```

Do not submit the job automatically unless explicitly requested. The user will
run `sbatch`.

Because the final model has roughly two million cells and full-resolution
evaluation has 50 views, analysis should process views/rays in bounded batches.
Do not retain every per-ray/per-cell dense matrix in GPU memory. Aggregate
statistics online and transfer completed summaries to CPU.

## 11. Testing requirements

Correctness matters more than quickly producing an attractive render. Add
tests before trusting the experiment.

### 11.1 CPU tests

- black and white RGBA compositing with known pixels;
- layer camera/frame alignment validation;
- alpha-mask category construction including overlap and empty pixels;
- oracle responsibility ratio and threshold boundary behavior;
- mutually exclusive/exhaustive cell labeling;
- density-mask construction without mutating source tensors;
- metrics on small synthetic RGBA arrays;
- checkpoint naming and manifest parsing;
- classifier serialization and deterministic reload;
- ROI connected-component selection on synthetic masks.

### 11.2 GPU smoke tests on the current node

Use a tiny scene/model configuration, not the two-million-cell production
model. Existing smoke-test infrastructure is located at:

```text
/shared/results/gmwilczn/foamsurf/radfoam/scripts/smoke_test_training.py
```

Required GPU checks:

- tiny RadFoam training still runs after the background change;
- an intermediate checkpoint saves and reloads;
- unmasked render before and after reload agrees;
- all-ones cell mask matches unmasked RGBA;
- all-zero cell mask gives zero premultiplied RGB/alpha;
- complementary masks do not rebuild or reorder topology;
- `return_contribution=True` produces finite nonnegative per-cell values;
- selecting a subset of rays changes contribution only through those rays;
- quantile depths have correct shape/order and are finite where valid;
- separated evaluation runs on a few aligned views and writes all expected
  artifacts.

Avoid comparing separately rendered RGB images by addition; use the union mask
in one volume-renderer call for compositing checks.

### 11.3 Regression tests

- legacy white-background RadFoam behavior must remain available;
- existing final-checkpoint evaluation must continue to load old `model.pt`
  files;
- diagnostics disabled must not alter the full render;
- saving intermediate inference checkpoints must not change the training
  trajectory other than unavoidable timing/I/O effects.

## 12. Analysis order and decision tree

Analyze results in this order.

### Step 1: validate the training run

- Confirm the model trained on `lego_both`, not the old Lego path.
- Confirm black-background composition was used consistently.
- Confirm all scheduled checkpoints and point counts.
- Plot full reconstruction quality against iteration.
- Inspect full renders for the persistent cloud.

### Step 2: inspect ray-depth behavior

For Lego-only, cloud-only, overlap, and empty masks, compare distributions of:

- `d90 - d10`;
- opacity;
- maximum or effective contribution concentration;
- number of intersections;
- any available significant-cell-count proxy.

Question: Is there a stage where solid rays are already depth-concentrated but
cloud rays remain broad?

### Step 3: oracle upper bound

- Derive cell attribution from training-layer masks.
- Render the resulting masks on test views.
- Measure surface quality, medium quality, leakage, ambiguity, and the
  persistent-cloud ROI.

Question: Does a cell-level partition of this checkpoint exist?

### Step 4: checkpoint-only classifiers

- Compare simple `sigma`, `tau_proxy`, local-concentration, and clustering
  methods.
- Compare their cell predictions with high-confidence oracle labels.
- Render all candidates rather than judging cell AUROC alone.

Question: Can checkpoint features recover the usable oracle split without
layer labels?

### Step 5: choose a teacher checkpoint

The selected checkpoint should balance:

- adequate full-scene reconstruction;
- strong `surface_only` reconstruction of `lego_lego`;
- strong `medium_only` reconstruction of `lego_clouds`;
- low cross-layer leakage;
- low ambiguous contribution mass;
- successful assignment of the persistent cloud to medium.

Do not automatically select the checkpoint with the best full PSNR.

### Decision A: oracle works and checkpoint-only works

This is the desired outcome. Export the selected checkpoint's spatial
surface/medium/empty scores as a frozen 3D prior for a later FoamSurf stage.
That later integration can guide:

- initial Gaussian placement;
- Gaussian clone/split eligibility during densification;
- Gaussian location/opacity regularization in medium regions;
- opacity pruning priority;
- initial/adaptive Voronoi site allocation in medium regions.

### Decision B: oracle works but checkpoint-only fails

RadFoam contains the separation, but current observable features do not recover
it. Inspect feature distributions and add better ray-to-cell statistics, such
as quantile-owning cell IDs, exact segment path length, or per-cell depth
concentration. Do not immediately change FoamSurf.

### Decision C: early oracle works, late oracle fails

This strongly suggests RadFoam's later optimization, likely including depth
quantile concentration, turns clouds into pseudo-surfaces. Use the best early
checkpoint as teacher and then test a weaker/no-quantile RadFoam ablation.

### Decision D: oracle fails at every checkpoint

The cells themselves mix solid and medium or fail to resolve the cloud. A
post-hoc classifier cannot solve this. Investigate:

- RadFoam spatial resolution around the persistent cloud;
- initialization bounds and site allocation;
- whether the same cells contribute to both layers;
- weaker/no quantile regularization;
- different checkpoint topology or a targeted site-allocation experiment.

### Decision E: density threshold alone works

Treat this as an excellent empirical result, but still validate it across
checkpoints and views using optical-thickness and leakage metrics. Do not claim
that raw learned `sigma` is physical density.

## 13. What is explicitly out of scope for this stage

- Do not train FoamSurf using `lego_lego` or `lego_clouds` supervision.
- Do not integrate a RadFoam prior into Gaussian losses yet.
- Do not launch another 100-run FoamSurf sweep.
- Do not change Gaussian `K`; previous discussion explicitly deferred testing
  another `K`.
- Do not delete/retriangulate RadFoam cells merely to render a subset.
- Do not select the final RadFoam checkpoint only because it has the best PSNR.
- Do not claim unsupervised disentanglement from oracle-attributed cells.
- Do not overwrite old RadFoam or FoamSurf outputs.
- Do not discard existing dirty-worktree changes.

## 14. Required final deliverables from the implementation

The implementation task is complete only when it provides:

1. configurable, tested black/white background handling in RadFoam;
2. a dedicated `lego_both` RadFoam config;
3. intermediate inference-checkpoint saving with a manifest;
4. topology-preserving density-masked rendering;
5. aligned loading of full/surface/medium datasets;
6. oracle cell attribution and isolated renders;
7. checkpoint-only feature extraction and baseline classifiers;
8. layer, leakage, cell-purity, and persistent-cloud metrics;
9. visual panels and a consolidated checkpoint comparison;
10. CPU tests and small GPU smoke tests;
11. an sbatch script for the complete one-run experiment;
12. a concise run README or command record inside the output directory.

After running, the report must answer these questions explicitly:

1. At which checkpoint, if any, are Lego and clouds most separable?
2. Can the persistent cloud be isolated as medium?
3. Does oracle cell attribution succeed?
4. Does a classifier using only checkpoint statistics approach the oracle?
5. Which features carry the useful signal: density, optical thickness, cell
   scale, contribution, neighborhood contrast, or depth concentration?
6. Does RadFoam later collapse the cloud into a pseudo-surface?
7. Is the failure one of classification, optimization, or insufficient
   spatial/topological resolution?
8. Is the evidence strong enough to justify exporting a RadFoam-derived prior
   into FoamSurf?

Only after those questions are answered should the project proceed to actual
RadFoam-guided FoamSurf training.
