# FoamSurf: initialization-only Voronoi warmup, 50 experiments

## Status and scope

Implementation prepared on a node without RTX4090. No tests, GPU compilation,
training, or Slurm submission were performed on that node. GPU validation is a
required separate job, submitted by the user. A successful test certificate for
the current source content is mandatory before the experiment runner starts.

All changes belong to scene INITIALIZATION. The existing joint Gaussian +
Voronoi trainer, medium activation, optimizer, reconstruction loss and renderer
are not changed. This profile is a separate `WarmupTrainer` subclass, called only
by the new warmup entrypoint. It rejects hybrid mode. The quantile extension is
compiled separately; it reuses geometry helpers by including the existing CUDA
source, but does not replace the installed `foamsurf._C` extension.

Repository: `/shared/results/gmwilczn/foamsurf/foamsurf`.
Conda environment: `/shared/results/gmwilczn/conda/envs/foamsurf` (name `foamsurf`).
Reference source, read-only: `/shared/results/gmwilczn/foamsurf/radfoam`.
Training RGB: `/shared/sets/datasets/foamsurf/lego/lego_both`.
Evaluation-only layers: `/shared/sets/datasets/foamsurf/lego/lego_lego` and
`/shared/sets/datasets/foamsurf/lego/lego_clouds`.

The starting native experiment is
`outputs/voronoi_prior_warmup/voronoi_prior_warmup_503195`; its report is
`analysis/ANALYSIS_REPORT.md`. It learned a spatially useful oracle partition,
but local-density-TV Otsu assigned nearly all cloud contribution to SURFACE.
The hypothesis here is that density parameterization, optimization, outer
scaffold and depth concentration regularization can recover a useful partition
with RadFoam-compatible opacity supervision but without SH.

## Commands, in order

Prepare the log directory before sbatch (Slurm opens log files before the script):

```bash
cd /shared/results/gmwilczn/foamsurf/foamsurf
mkdir -p slurm_logs
sbatch scripts/test_warmup_radfoam_like.sbatch
```

The test job requires an RTX4090, uses the existing CUDA/OptiX installations,
builds only the new quantile extension, and runs CPU and GPU tests together.
It writes:

```text
outputs/warmup_tests/TEST_JOB_ID/passed.json
outputs/warmup_tests/TEST_JOB_ID/passed.xml
slurm_logs/warmup_test_TEST_JOB_ID.out
slurm_logs/warmup_test_TEST_JOB_ID.err
```

The JSON is marked `passed` only after pytest succeeds, no required tests are
skipped, and the source fingerprint remains unchanged. A failing job writes
`failed`, not a success certificate. The reference RadFoam Python package must
be importable in the same environment for the parity test; it is never trained
or modified by these jobs.

AFTER reviewing the test result, manually submit:

```bash
export WARMUP_TEST_CERTIFICATE=/shared/results/gmwilczn/foamsurf/foamsurf/outputs/warmup_tests/TEST_JOB_ID/passed.json
sbatch --export=ALL scripts/sweep_warmup_radfoam_like.sbatch
```

This is an array `0-49%8`: 50 experiments, at most eight concurrently, one GPU
per task. No script automatically submits the sweep after tests. Both the
sbatch and Python runner verify the certificate. Changes to relevant Python,
CUDA, configs, tests or launch scripts require a fresh test job. Each task has
its own CUDA extension build directory to avoid concurrent compilation races.

Outputs:

```text
outputs/warmup_radfoam_like/ARRAY_JOB_ID/run_INDEX/
  config.yaml                  # legacy inference-compatible experiment schema
  warmup_profile.json           # actual warmup activation/LR/loss/resolutions
  sweep_variant.json            # ID, overrides, seed, provenance/test certificate
  checkpoints/iter_XXXXXXXX.pt
  checkpoints/preadapt_XXXXXXXX.pt
  checkpoints/best.pt
  checkpoints/last.pt
  metrics/train.csv
  metrics/eval.csv
  summary.json
  warmup_analysis/analysis.json
  warmup_analysis/iteration_XXXXXX/
```

The profile JSON is authoritative for warmup losses and schedules. `config.yaml`
retains the old schema so legacy evaluation and scene initialization can read
the canonical checkpoint. It is NOT sufficient to reproduce a profiled run on
its own. Run 0 uses the original Trainer and has no profile JSON.

Inspect the executable manifest (does not start a run):

```bash
conda activate foamsurf
python scripts/run_warmup_radfoam_like.py --manifest
```

One reference run, after tests:

```bash
python scripts/run_warmup_radfoam_like.py \
  --config configs/voronoi_warmup_radfoam_like.yaml \
  --output-dir outputs/warmup_radfoam_like/manual_reference \
  --certificate "$WARMUP_TEST_CERTIFICATE" --wandb-mode offline
```

This command requires a GPU allocation; the array sbatch is the normal launcher.
W&B defaults to offline; use `WANDB_MODE=online` in the job environment if desired.

Resume a warmup with the SAME index/profile and `--resume CHECKPOINT`; a new
output directory is recommended for unambiguous logs. `iter_*.pt` and `last.pt`
are resumable. `preadapt_*.pt` are diagnostic snapshots taken between the
optimizer update and topology adaptation and are deliberately rejected for
resume. The old generic Trainer is for evaluation of canonical tensors only,
not for resuming warmup optimizer moments.

Re-run interrupted analysis without training:

```bash
export WARMUP_RUN=/shared/results/gmwilczn/foamsurf/foamsurf/outputs/warmup_radfoam_like/ARRAY_JOB_ID/run_INDEX
sbatch --export=ALL scripts/analyze_warmup_radfoam_like.sbatch
```

Completed per-checkpoint analyses are reused. To compare all 50 runs on CPU:

```bash
python scripts/summarize_warmup_radfoam_like.py \
  --root outputs/warmup_radfoam_like/ARRAY_JOB_ID
```

This writes `checkpoint_metrics.csv` and `SWEEP_REPORT.md`, and lists missing
runs explicitly. A low leakage score alone is not a selection criterion.

## Parameterization and supervision

The warmup medium uses `scale * softplus(raw, beta)`. Its default beta is 10,
scale 1, initial INNER physical sigma `log(2)/10 = 0.06931471805599453`
(equivalent to RadFoam raw=0). Outer sigma is
`log(1+exp(-5))/10 = 0.0006715348489118068` (RadFoam raw=-0.5).
Configuration uses physical sigma explicitly, avoiding ambiguous raw-vs-sigma
fields. Changing beta in a matched-sigma ablation preserves the physical field.

RGB uses SmoothL1 beta=1, averaged over pixels/channels. SSIM is reported but
does not enter the SmoothL1 objective. The reference warmup now matches native
RadFoam's opacity objective exactly:

```text
L = SmoothL1(rendered_rgb, target_rgb)
    + opacity_weight * mean((rendered_alpha - target_alpha)^2)
    + quantile_weight(step) * depth_quantile_width
```

The default `opacity_weight` is 1.0. Blender/RGBA cameras provide their raw
per-pixel alpha. Ordinary COLMAP RGB images are converted to RGBA with alpha=1,
so the same term becomes `mean((1-rendered_alpha)^2)`, matching native RadFoam
on real scenes without requiring an unavailable alpha channel. Sampled alpha is
stored in `TrainingBatch`; the existing joint Trainer ignores this metadata and
its objective remains unchanged. Evaluation-only lego/cloud layer ground truth
never enters the optimizer.

This distinction is essential: the first sweep (array 504683) omitted opacity
loss. It showed that depth-quantile width alone drives density into transparent
background. Those results are an ablation, not a valid RadFoam-like reference.
There is still no SH in the warmup.

Warmup positions/density/RGB are ordinary trainable parameters. Outer sites
are protected by the existing permutation-safe guard mask; `outer_mask` is an
alias in the warmup medium. Default outer positions are fixed, while density
and color learn. Frozen rows have gradients and Adam moments cleared and their
values restored after optimizer updates. They cannot be pruned or parents of
new inner cells. Inner population limits exclude the outer count.

Training logs include raw and weighted RGB/opacity/quantile losses plus target
and rendered alpha means. `opacity_weight=0` is an explicit RGB-only ablation.

The random scaffold uses `center + N(0, I)*scale`, like the RadFoam Gaussian
initializer. This is a scaffold origin mask, not a guarantee every sampled site
is outside the bbox: a small number can land inside. Corner-guard run 35 is the
strictly outside alternative. Its eight nominal cube corners receive a
deterministic sub-per-mille radial staggering: exact cube corners are
co-spherical/coplanar and caused an ambiguous Delaunay triangulation in array
504683. Outer sites receive label -1 during analysis and
are excluded from semantic classes and Gaussian initialization candidates.

The topology replacement wrapper expands temporary retry bounds to encompass
outer sites. Child generation still uses the inner bbox. This prevents the
shared retry/clamp code from pulling outer sites into the object volume.

## Quantile definition and gradients

Two descending thresholds q are sampled uniformly in (0,1) for each ray. They
are thresholds of TRANSMITTANCE, not quantiles normalized by final opacity.
For a constant-density segment starting at t0 with incoming transmittance T:

```text
T_next = T * exp(-sigma * length)
if T_next < q <= T:
    depth_q = t0 + log(T/q)/sigma
otherwise, if never reached: depth_q = -1, owner = -1
```

Loss: `mean(abs(d0-d1) * both_depths_positive)`, averaged over ALL rays.
Do not replace this with a mean over valid rays. At opacity alpha, two random
thresholds are both reached with probability approximately alpha squared.
This naturally makes the penalty more frequent on opaque rays; it is not a
semantic mask and cannot guarantee clouds stay volumetric.

The separate CUDA backward differentiates density in the owning cell, density
in every preceding segment, and site positions through segment boundaries.
Camera rays are fixed and ray gradients are explicitly unsupported. Colors do
not receive a direct quantile gradient. Providing Gaussians to this quantile
API raises an error. RGB/alpha kernels in the joint renderer are untouched.

Tests compare analytical two-cell derivatives, finite differences, invalid
quantiles, clipping at near plane, diffuse-vs-concentrated widths, no color
gradient, zero weight, RGB/alpha/gradient invariance and RadFoam reference depths
and density gradients. Site gradients are checked against an independent
analytic oracle, not assumed correct because RadFoam produces them.

## Schedules

Schedules use zero-based `step = completed_iteration - 1`, applied BEFORE each
optimizer step. This deliberately avoids RadFoam's anomalous first full-LR step
followed by an LR=0 update. The density warmup starts at zero.

| Parameter | Initial LR | Final LR | Linear warmup | Cosine horizon |
|---|---:|---:|---:|---:|
| sites | 2e-4 | 5e-6 | 0 | 18000 |
| density | 0.1 | 0.01 | 2000 | 20000 |
| color logits | 0.005 | 0.0005 | 0 | 20000 |

After a horizon, LR is zero; at the horizon it is the final LR. Adam epsilon
is 1e-15 for the reference warmup (legacy ladder starts at 1e-8). Color remains
sigmoid RGB, so its parameterization still differs from RadFoam SH DC.

Progressive training factors: 4 at step 0, 2 at step 150, 1 at step 500.
Validation stays at base resolution. A separate training dataset/sampler is
replaced at each transition while retaining camera order. Checkpoint/resume
saves Python, NumPy, Torch and CUDA RNG, sampler stack, optimizer moments and
adaptation accumulators. Loading reconstructs the current resolution from the
schedule, then restores the saved RNG and sampler state.

Reference quantile weight = 1e-4, linear ramp from step 0 through 10000; thus
at step 5000 effective weight is 5e-5. Stopping the screening at 5k does not
compress the LR or quantile schedule into 5k.

## 50-run manifest

Reference R (ID7): 350000 inner + 5000 outer sites, beta10, inner sigma log2/10,
SmoothL1, opacity weight 1, cosine schedules above, fixed outer positions,
progressive resolution, quantile 1e-4 with 10k ramp, RGB appearance, seed 0.

| IDs | Changes |
|---|---|
| 0 | Original native warmup Trainer/control, beta1, sigma .005, L1+DSSIM, constant LR, no opacity/outer/progressive/quantile |
| 1 | Ladder: SmoothL1, otherwise old profile; opacity weight 0 |
| 2 | ID1 + beta10 preserving sigma .005; opacity weight 0 |
| 3 | Reference opacity weight 1 plus cosine optimizer/initialization bundle including eps 1e-15 |
| 4 | ID3 + 5000 outer sites |
| 5 | ID4 + progressive 4/2/1 |
| 6 | ID5 + quantile loss, initial sigma still .005 |
| 7 | R: ID6 + initial sigma log2/10 |
| 8 | R without quantile |
| 9 | R without outer |
| 10,11,12 | R beta1, beta5, beta20, same physical initial sigma |
| 13,14,15,16,17,18 | R initial sigma .0006715, .01, .02, .04, .10, .20 |
| 19 | R without density-LR warmup |
| 20–25 | R quantile weight 1e-6, 3e-6, 1e-5, 3e-5, 3e-4, 1e-3 |
| 26–30 | R quantile ramp 0, 500, 1000, 2000, 5000 steps |
| 31–33 | R quantile start 500,1000,2000; ramp duration 2000 |
| 34 | R fixed q=[.9,.5] instead of random thresholds |
| 35 | R with eight corner guards instead of random outer sites |
| 36–38 | R outer count 500,2000,10000 |
| 39,40 | R outer scale 5,20 |
| 41 | R with trainable outer positions |
| 42 | R full resolution from start |
| 43 | R progressive milestones 0,500,1500 |
| 44 | R progressive 8/4/2/1 at 0/150/500/1500 |
| 45 | R with 131072 inner sites, maximum 500k inner |
| 46 | R with 500000 inner sites, maximum 700k inner |
| 47 | R fixed inner population (capacity freeze, adaptation controller inactive) |
| 48 | R densification .15 from 2k every500, maximum 1m inner |
| 49 | R replicated at seed1 |

All non-legacy variants from ID3 onward use opacity weight 1; ID8 removes
quantile loss but retains opacity supervision, giving the key paired ablation
against ID7. The previous discussion duplicated ID6 at ID14. ID14 now uses sigma .01;
manifest tests require all 50 effective profile/override combinations unique.
Run 0 alone uses the original training loop. The new warmup loop applies an
optimizer update before adaptation; therefore the first ladder step changes
that ordering as well as its RGB loss. Interpret the ladder accordingly, not
as a perfect one-factor causal experiment. Paired R ablations isolate their
listed factors within the same new training loop.

Inner adaptation otherwise keeps FoamSurf's existing score/pruning rule and
adds 2% every500 from iteration500 through5000. This is not a port of RadFoam's
population algorithm or raw-density reset. No reset to raw=-1 is introduced.
For 20k promoted runs adaptation continues through11000; schedule horizons are
unchanged. Initial count350k is not a fixed target; inner/outer counts are logged
separately. ID47 uses the existing freeze-at-capacity mechanism because the
legacy warmup config contract requires an enabled adaptation controller.

## Checkpoints and later scene initialization

Milestones: 0,150,500,1000,2000,3000,4000,5000; promoted runs additionally
8000,11000,15000,20000. Each adaptation has a pre-update snapshot and milestone
post-adaptation state. `best.pt` is selected by the existing full RGB PSNR
evaluator; it is not asserted to be the best separation checkpoint.

Canonical `model.medium.raw_density` is inverse-softplus(beta1) of the learned
physical sigma. Existing inference uses the same physical field. The additional
`warmup` payload contains the original raw parameter, beta/scale/profile,
optimizer interpretation, outer mask and exact stochastic/adaptation state.
Only `WarmupTrainer` resumes these optimizer moments correctly.

Changing the parameterization of a saved model without changing sigma does
not change renders or Otsu. The benefit under study is the training trajectory.
No Gaussian spawning or whole-representation training is performed by any new
script. The future handoff must consume canonical model parameters and fresh
joint optimizer state, while excluding outer sites from spawning candidates.

## Analysis and visualization

Each array task analyzes its completed checkpoints in a separate process.
Checkpoints are loaded as canonical beta1 fields by the existing evaluation
Trainer; no optimization is performed. Active/empty classification uses Otsu on
log10 contribution accumulated from TRAIN cameras. Surface/medium classification
uses Otsu on log10 mean absolute density differences along INNER-INNER graph
edges. Outer sites do not influence either histogram or semantic TV edges.
This fixes the old `contribution>0` rule; it is a hypothesis to be evaluated,
not a guarantee of semantic separation.

Only after checkpoint-only labels are fixed, evaluation-only layer targets
produce oracle labels and diagnostics. Oracle uses contribution on exclusive
LEGO-only/cloud-only test rays. It is an imperfect upper-bound diagnostic for
occluded/mixed cells, not deployment ground truth.

Per checkpoint output includes:

- full/surface/empty/medium renders and fixed-density occupancy panels for four
  views, with full target and class overlays;
- the same decomposition for the privileged oracle;
- ordinary and contribution-weighted confusion;
- cloud contribution incorrectly assigned to SURFACE;
- full reconstruction PSNR and component PSNR/SSIM/leakage;
- inner/outer raw density, sigma, local-TV and contribution distributions;
- stochastic quantile validity/width statistics over deterministic subsamples
  of all test views, also separated by oracle image region;
- quantile-owner counts by oracle class and per cell;
- quantile validity/width maps for fixed q=.9,.5 on four complete views;
- density/TV/contribution histograms and separation-over-time plots;
- `cell_features.pt` for subsequent analysis, with an explicit map from current
  site order to source checkpoint indices;
- `cell_labels.pt` in ORIGINAL checkpoint order for future initialization (outer
  label -1), preventing a triangulation permutation from misassigning labels.

`raw_density` distributions are recovered using the saved warmup beta/scale;
inference canonical raw values must not be mistaken for training raw values.
The principal classifier remains independent of layer images. Oracle-dependent
metrics can compare algorithms on this benchmark but cannot be used as an
automatic real-photo checkpoint selection rule.

Compare full reconstruction, cloud reconstruction, leakage, contribution
confusion and oracle gap together. A nearly empty predicted medium can have
low medium→LEGO leakage; it is not a successful separation. Inspect the entire
trajectory: LEGO may become concentrated before clouds, with separation later
degrading as quantile regularization strengthens. A checkpoint with good RGB
PSNR alone is not sufficient for Gaussian spawning.

## Promotion after the screening

After manual review, supply five selected IDs (space-separated). The promotion
job runs all five from initialization at seeds0,1,2 to20k. It does not inherit an
optimizer/checkpoint from a different configuration or seed.

```bash
export WARMUP_SELECTED_IDS='ID_A ID_B ID_C ID_D ID_E'
sbatch --export=ALL scripts/promote_warmup_radfoam_like.sbatch
```

Replace the symbolic IDs with integers. The same test certificate is required.
Promotion is not submitted automatically. Full-run results determine whether
separation persists, whether clouds collapse later, and whether the apparent
benefit is robust across seeds. No automatic best-checkpoint-to-Gaussians action
is included.

## Implementation map

- `training/warmup_profile.py`: profile validation, inverse activation, schedules,
  quantile loss normalization.
- `training/warmup_trainer.py`: isolated initialization trainer, outer mask,
  progressive data, checkpoint conversion/resume.
- `training/warmup_quantiles.py`: warmup-only CUDA extension loader/autograd API.
- `renderer/native/warmup_quantiles.cu`: independent differentiable quantile pass.
- `training/warmup_sweep.py`: executable 50-run manifest and config construction.
- `configs/voronoi_warmup_radfoam_like.yaml`: optional reference wrapper config.
- `scripts/run_warmup_radfoam_like.py`: verified launch/resume/manifest CLI.
- `scripts/warmup_gate.py`: RTX4090 tests and source-content certificate.
- `scripts/analyze_warmup_radfoam_like.py`: checkpoint analysis and visualizations.
- `scripts/summarize_warmup_radfoam_like.py`: CPU sweep tables/inventory.
- `tests/warmup_radfoam_like/`: CPU, CUDA, integration and isolation contracts.
- `scripts/*warmup_radfoam_like.sbatch`: test, screening, analysis and promotion jobs.
