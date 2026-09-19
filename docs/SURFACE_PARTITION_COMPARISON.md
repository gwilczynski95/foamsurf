# Binary surface partition comparison

Historical report of job 506970 (14 variants). The active implementation now
retains only `weighted_valley_w1_s2`, `otsu_w1`, and `gmm_2`; see
[NATIVE_END_TO_END_TRAINING.md](NATIVE_END_TO_END_TRAINING.md). Historical outputs
remain unchanged. Rerunning the comparison now produces these three variants.

## Scope

Postprocessing only. No training, renderer or joint initialization code is changed.
Repository: `/shared/results/gmwilczn/foamsurf/foamsurf`.
Conda environment: `/shared/results/gmwilczn/conda/envs/foamsurf`.
Input: existing `warmup_analysis/iteration_005000/cell_features.pt` from organic
sweep `outputs/warmup_organic/506329/run_ID`, and corresponding
`checkpoints/iter_00005000.pt` for rendering.

The binary output is SURFACE / NON-SURFACE. All outer-provenance and
zero-contribution cells are NON-SURFACE, not discarded. This is a conservative
prior hypothesis: outer sites may have moved into useful geometry during training.
No Gaussians are spawned by this comparison.

## Four method families, fourteen variants

Features used to decide labels: physical sigma, local TV, accumulated TRAIN-view
contribution, outer provenance. Oracle labels and separated layers are NOT inputs.
Logarithms clamp at 1e-12. Histograms use 256 bins over active inner sites.
Active means contribution > 0; no preliminary contribution-Otsu cut.

1. **Valley**, three variants: Gaussian histogram smoothing of 1, 2 or 4 bins.
   Detect peaks with prominence >= 3% of maximum smoothed histogram height.
   Select the two rightmost prominent peaks, then the minimum between them.
   Require a valley at least 15% below the smaller peak. No valid valley means
   uncertain, all cells NON-SURFACE; no hidden Otsu fallback.
2. **Weighted valley**, six variants: the same smoothing values, histogram
   weights contribution^0.5 or contribution^1. Normalize contribution by its
   maximum before exponentiation; this does not affect relative weights.
3. **Weighted Otsu**, two variants: the same two weight powers. Maximize
   between-class variance; for a tied plateau take its central cutoff.
   Otsu always proposes a split for nonconstant data; it is NOT a semantic
   confidence test and can split a unimodal scene.
4. **Two-dimensional Gaussian mixtures**, three variants: fixed 2 components,
   fixed 3, or BIC choice among 1..4. Features are log sigma and log TV.
   Fit on a deterministic contribution-weighted resample of up to 20k sites
   (with replacement, seed 0), standardized using sample median/std.
   sklearn does not accept sample_weight here. BIC refers to the resampled
   distribution, not an exact likelihood of all cells.
   Surface is the component with greatest sum of standardized feature means,
   provided neither mean is >0.25 standardized units below the maximum on that
   axis. Use posterior >=0.5. Single-component, failed convergence or ambiguous
   ordering yields uncertain. This conservative prototype selects ONE component;
   additional surface modes can therefore be missed by 3/4-component fits.

All methods have explicit structural assumptions; none proves solid-body semantics.
The valley method assumes the rightmost prominent mode is surface and may split
multiple surface modes. Parameter variations are diagnostic, not GT-tuned choices.
Do not treat these heuristics as validated for arbitrary scenes yet.

## Outputs

`run_ID/surface_partitions/iteration_005000/VARIANT/` contains:

- `histograms.png`: sigma and TV histograms colored by predicted class; fit
  histogram and smoothed curve; vertical TV cutoff for scalar methods.
- `decision_boundary.png`: sigma/TV scatter, black TV cutoff or actual GMM
  posterior=0.5 contour. GMM has no honest single vertical cutoff; none is invented.
- `surface_mask.pt`: boolean SURFACE mask in ORIGINAL checkpoint order,
  classifier metadata, source checkpoint path. Complement means NON-SURFACE.
- `result.json`: settings, thresholds/status/counts and optional oracle evaluation.
- With `--render`: `view_00_partition.png` through `view_03_partition.png`,
  plus individual full/surface/non-surface/overlay and fixed-density occupancy PNGs.
  Overlay: red SURFACE, blue NON-SURFACE. Occupancy uses artificial sigma=0.5,
  white color and black background; it is not physical trained opacity.

`summary.json` collects all variant results. Oracle evaluation reports original
field LEGO contribution retained and cloud contribution assigned to SURFACE.
These are NOT PSNR, cell-count precision or rerendered layer metrics. Removing
cells changes transmittance, so render inspection remains necessary.

Features are in analyzer topology order. Mask export uses checkpoint_index;
rendering applies the fresh topology permutation and checks positions against
the cached features before rendering. Cached features must match the checkpoint.

## Running

CPU tests (23 tests):

```bash
cd /shared/results/gmwilczn/foamsurf/foamsurf
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
 /shared/results/gmwilczn/conda/envs/foamsurf/bin/python -m pytest -q tests/test_surface_partition.py
```

CPU histograms/masks, without rendering:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
 /shared/results/gmwilczn/conda/envs/foamsurf/bin/python scripts/compare_surface_partitions.py \
 --run outputs/warmup_organic/506329/run_12 --iteration 5000
```

RTX4090 tests + all renders (array tasks map to runs 12,13,2,3):

```bash
sbatch scripts/compare_surface_partitions.sbatch
```

Submitted job: `506970`. Logs: `slurm_logs/partition_506970_TASK.{out,err}`.
The job repeats CPU tests first, then performs real GPU rendering of all variants.
No GPU result is certified merely by CPU success; inspect SLURM exit status and
the four panel files per variant. No training is submitted.

Single custom run:

```bash
sbatch --array=0 --export=ALL,WARMUP_RUN=/absolute/path/to/run,ITERATION=5000 \
 scripts/compare_surface_partitions.sbatch
```

Renderer uses sequential Voronoi-only passes and explicit 8192-ray chunks;
completed images are moved to CPU. It avoids the full evaluator and its
unnecessary Gaussian-only rendering that failed in the old analysis. It still
loads the standard Trainer checkpoint for compatible topology/activation.

## Initial CPU observation, run 12 at 5k

- Weighted valley, smoothing 2: LEGO contribution retained 99.675%, cloud
  contribution assigned to surface 0.162%; log10 TV cutoff -0.1104.
- Weighted Otsu: 98.935% / 0.122%; cutoff 0.3674.
- Unweighted valley, smoothing 2: 98.619% / 0.116%; cutoff 0.5641.
- GMM 2: 98.451% / 0.112%.
- GMM auto chooses 4 components but retains only 62.036% of LEGO contribution.
  Better density fit does not imply better semantic grouping. This is an
  important failure mode of selecting only the highest component.

These diagnostics were produced without using oracle data to select thresholds.
They are not a multi-scene validation. Compare both seeds and quantile/no-quantile
variants, then test other scenes before promoting a default.
