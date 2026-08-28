# FoamSurf handoff for a new chat

## Where we are

FoamSurf is now a standalone research repo, not a thin RadFoam/FALCON wrapper.
The active focus is to understand and stabilize the two base representations
before doing joint disentanglement work:

- Gaussian-only surface representation
- Voronoi-only medium representation

The roadmap and exact stage definitions live in:

- `docs/IMPLEMENTATION_ROADMAP.md`
- `docs/STANDALONE_REPRESENTATIONS.md`
- `docs/B01_DIAGNOSTICS.md`
- `docs/B1_2DGS_BASELINE.md`
- `docs/DETERMINISM_AUDIT.md`

The repo walkthrough is in:

- `docs/REPOSITORY_WALKTHROUGH.md`

## Current project structure

Important directories and entry points:

- `train_b0.py` is the main training entry point
- `configs/` contains the canonical stage configs
- `training/` contains trainer, loss, optimizer, logging, checkpointing
- `evaluation/` contains the three-mode validation logic
- `foamsurf_model/` contains the Gaussian and Voronoi models plus topology
- `foamsurf_data/` contains Blender and COLMAP dataset loading
- `renderer/` is the standalone CUDA/OptiX renderer
- `tests/b0/` contains B0 contract tests

The renderer vendors the needed FlaRe/FALCON base OptiX sources locally.
It does not depend on a neighboring RadFoam or flare checkout at runtime.

## What is implemented

### B0 baseline

B0 trains both representations together with a single RGB MSE loss:

- one camera per batch
- both renderers composed in one depth-ordered pass
- separate Adam optimizers for Gaussians and Voronoi
- validation modes: `hybrid`, `gaussian_only`, `voronoi_only`
- CSV logging, W&B logging, checkpoint save/resume

### B0.1 diagnostics

B0.1 adds diagnostics without changing the loss:

- responsibility / contribution / visibility style metrics
- per-view panels and CSVs
- layer-aligned evaluation on `lego_lego` and `lego_clouds`

### Stage 1 / Stage 2 qualification

The repo now contains separate qualification paths for:

- Gaussian-only experiments on `lego_lego`
- Voronoi-only experiments on `lego_clouds`

The intent is to qualify each representation separately before trying to
explain joint entanglement.

## Important design choices

- Do not use `lego_lego` or `lego_clouds` as training targets in later hybrid
  stages. They are evaluation-only beyond the qualification stages.
- Do not use oracle / layer-supervised weights to initialize the hybrid model.
- Do not mix representation changes, topology changes, and disentanglement
  loss changes in one experiment.
- `set_geometry` is intentionally separate from `render()`.
- The renderer should remain standalone and importable as `foamsurf`.

## What we learned so far

The current Gaussian-only baseline is still not behaving like a healthy
2DGS-style surface model. The main symptoms observed in runs were:

- strong early performance, then PSNR drops later
- opacity / scale growth becoming unstable
- pruning sometimes removing sensible Gaussians
- some runs producing overly blurry splats
- a mismatch between intuition from 3DGS / 2DGS and the current behavior

The most important suspected contributors were:

1. Initialization mismatch against official 2DGS / flare_new behavior
2. Opacity-dependent support size in the renderer
3. Missing or different scale safeguards compared with 2DGS / flare_new
4. Learning-rate and schedule differences
5. The current hybrid setup interfering with a clean Gaussian-only diagnosis

## Active debugging direction

The current recommendation is to debug the Gaussian-only baseline in this
order:

1. Make the Gaussian-only run as close as possible to the pinned 2DGS recipe
2. Verify init scales against the official 2DGS logic
3. Freeze away extra degrees of freedom when testing the baseline
4. Test with a single-view overfit before trusting the full 30k run
5. Only then re-enable densification, pruning, opacity reset, and hybrids

## Active runs and what to watch

There was a Gaussian-only run that was manually terminated after it started
degrading badly. The key issue was that the model did not remain stable:

- Gaussians became too large
- opacity collapsed or saturated in a bad way
- pruning / adaptation no longer behaved like the intended 2DGS-style policy

There is also an active Voronoi-only baseline run that should be left alone
unless explicitly requested by the user.

## Good next step for a new chat

If the new chat continues the investigation, start by reading:

1. `docs/STANDALONE_REPRESENTATIONS.md`
2. `docs/B1_2DGS_BASELINE.md`
3. `docs/DETERMINISM_AUDIT.md`
4. `renderer/README.md`
5. `docs/IMPLEMENTATION_ROADMAP.md`

Then inspect the Gaussian-only configuration and the renderer support-size
logic before changing anything else.
