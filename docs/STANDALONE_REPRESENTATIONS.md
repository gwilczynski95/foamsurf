# Standalone representation qualification (Stages 1 and 2)

This milestone adds real, testable Gaussian-only and Voronoi-only training to
FoamSurf. It is infrastructure for the first two qualification stages in
`IMPLEMENTATION_ROADMAP.md`; it does not claim that either representation has
already passed its quality threshold.

## Mode and target contract

Two independent config fields make accidental oracle supervision difficult:

| `representation_mode` | `dataset.train_target` | training images | native forward |
|---|---|---|---|
| `hybrid` | `both` | `source_path` (`lego_both`) | Gaussian + Voronoi |
| `gaussian_only` | `lego` | `gaussian_ground_truth_path` (`lego_lego`) | Gaussian only |
| `voronoi_only` | `clouds` | `medium_ground_truth_path` (`lego_clouds`) | Voronoi only |

Every other pairing is rejected by `ExperimentConfig.validate()`. In a pure
mode the inactive module has `requires_grad=False`, its optimizer is not
stepped, its adaptation events are disabled, and it is omitted from the
training renderer call. It is therefore not an opacity/density-zero trick.
The inactive model is retained as a very small frozen object so the shared
checkpoint/evaluator schema remains compatible.

Validation still writes the familiar hybrid/Gaussian/Voronoi comparison
files, but `eval/primary_psnr` and `eval/primary_ssim` select the active pure
branch. `best.pt` is selected by `eval/primary_psnr`, not by a render containing
the inactive representation.

## Adaptive Voronoi medium

`foamsurf_model/voronoi_adaptation.py` implements population changes without
RadFoam's depth-quantile or solid-surface regularization. Between topology
events it accumulates native renderer statistics per cell:

- compositing contribution;
- visibility/segment count;
- maximum RGB ray residual;
- combined site, density and color gradient magnitude.

The current split score is proportional to

```text
cell_radius * (max_error * sqrt(contribution / visibility) * sqrt(weighted_path / visibility)
               + gradient / visibility)
```

and is defined only for sufficiently observed, non-guard cells. Cell radius
and the farthest Delaunay neighbour are computed from CSR adjacency. A child
is placed halfway toward that neighbour, receives a small scale-relative
jitter, is clamped to the medium bounding box, and copies the parent's density
and color. This is a volumetric resolution operation; it does not optimize a
depth quantile or force a hard interface.

Pruning is intentionally conservative. A site must remain inactive for
`prune_patience` consecutive adaptation windows, have no active neighbour,
occupy a small cell, and not be a guard site. A single event can remove at most
`max_prune_fraction` and never crosses `min_points`. This preserves coarse
coverage while allowing redundant dense regions to become sparse.

`VoronoiTopologyController.replace_population()` applies prune and split as
one transaction. Survivor Adam moments are indexed with the survivors; child
moments start at zero. The guard mask follows every permutation. The method
then performs a full Delaunay rebuild, applies its permutation to parameters
and optimizer state, refreshes adjacency and synchronizes the native renderer.
Variable-population checkpoints restore the model, both optimizers, guard
mask, adaptation counters and pruning streak; resume rejects a different
representation/target pair.

## Baseline configs

- `configs/stage1_gaussian_only.yaml`: 30k, `lego_lego`, full 800x800 image,
  cameras without replacement, L1+DSSIM and the current 2DGS-aligned Gaussian
  adaptation profile.
- `configs/stage2_voronoi_only.yaml`: 30k, `lego_clouds`, 8,192 initial sites,
  adaptive split/prune to at most 32,768 sites, RGB MSE and no surface loss.

The Voronoi initial population is a sweep variable, not a fixed assumption.
The supplied sweep explicitly includes 2k, 4k, 8k, 16k and 32k starts.

## Tests and smoke runs

Submit the full CPU/CUDA regression suite:

```bash
sbatch scripts/test_stage12.sbatch
```

Submit two four-step end-to-end jobs (Gaussian-only and Voronoi-only):

```bash
sbatch scripts/smoke_stage12.sbatch
```

The smoke jobs verify train/eval CSV files plus `last.pt` and `best.pt`.
Focused tests in `tests/b2/` cover:

- legal and illegal mode/target pairs;
- 16k and 32k initial Voronoi configurations;
- native pure-forward gradient isolation;
- split budget, prune cap and immutable guard sites;
- Delaunay rebuild and Adam-moment migration after a real CUDA mutation;
- adaptation-state roundtrip;
- validity and uniqueness of every sweep config;
- Slurm array bounds.

## Overnight sweeps

Stage 1 contains 24 jobs (8 controlled 2DGS-profile variants × 3 seeds):

```bash
sbatch scripts/sweep_stage1_gaussians.sbatch
```

It tests the baseline, MSE ablation, two gradient thresholds, two opacity
thresholds, no opacity reset, and a smaller clone/split scale boundary. W&B
project: `foamsurf_stage1_gaussian_sweep`.

Stage 2 contains 70 jobs:

```bash
sbatch scripts/sweep_stage2_voronoi.sbatch
```

Ten are fixed-population controls (five initial counts × two seeds). Sixty are
adaptive: five initial counts × three site learning rates × two density
learning rates × two seeds. The growth budget is 4× the initial population,
capped at 65,536. W&B project: `foamsurf_stage2_voronoi_sweep`.

Set `WANDB_MODE=offline` or `WANDB_MODE=disabled` when submitting if needed.
All runs retain `metrics/train.csv`, `metrics/eval.csv`, validation images,
`summary.json`, and last/best checkpoints.

## What to inspect before Stage 3

For Gaussians, compare test PSNR/SSIM, sharp crops, population events, opacity,
screen radius and scale stability against the external 2DGS baseline. For
Voronoi, compare adaptive runs to the fixed control at each starting count,
then inspect whether added sites concentrate in cloud volume rather than
background. The implementation is ready for these experiments; their results,
not a smoke test, decide whether Stages 1 and 2 are complete.
