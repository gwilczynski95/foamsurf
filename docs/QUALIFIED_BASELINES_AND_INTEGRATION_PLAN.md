# Qualified standalone baselines and integration plan

## Purpose

The surface Gaussian and volumetric Voronoi representations have now been
qualified separately on the layered Lego development dataset. The next stage
is integration. It must first establish whether the already working
representations compose correctly and how they behave under the ambiguous
combined RGB objective. Disentanglement losses are deliberately out of scope
until that behaviour has been measured.

The layer-separated datasets are a development convenience. They may be used
for standalone qualification and evaluation diagnostics, but the final hybrid
method must not require them for training, initialization, model selection, or
deployment.

## Qualified Gaussian-only baseline

The canonical configuration is
`configs/baseline_best_2d_gaussians.yaml`, selected from run
`gk_458910_17`.

- target: `lego_lego`
- result: 37.1242 PSNR at 30k iterations (seed 0)
- black background and full-image L1 + 0.2 DSSIM
- RGB appearance and Gaussian hit buffer K=64
- 100k initial Gaussians
- GraphDECO-style clone/split/prune from 500 to 15k
- no Gaussian population limit
- no opacity reset
- Gaussian scale regularization and scale guards enabled

Run it with:

```bash
sbatch scripts/train_best_2d_gaussians.sbatch
```

## Qualified Voronoi-only baseline

The canonical configuration is `configs/baseline_best_voronoi.yaml`. It is the
exact `combo_slow_sched` recipe selected from run `vref_462246_28` and retained
after the larger-capacity sweep.

- target: `lego_clouds`
- result: 26.632 peak PSNR at 28k and 26.629 at 30k (seed 42)
- black background and full-image L1 + 0.2 DSSIM
- 2,048 initial sites, adaptively grown to at most 8,192
- 5% densification every 500 iterations from 1k to 20k
- initial density 0.005 and child jitter 0.025
- site, density, and color learning rates decay between 15k and 30k
- exact duplicate child sites are repaired before Delaunay reconstruction

The capacity sweep showed no quality gain from allowing 16k, 32k, or 64k
sites. Their best PSNR was similar, while their final quality became
progressively less stable. Therefore 8,192 is the qualified capacity, not an
arbitrary implementation limit.

Run it with:

```bash
sbatch scripts/train_best_voronoi.sbatch
```

## What is already integrated

`HybridSceneModel` already owns both representations. The native renderer
places Gaussian hit events and Voronoi segments in one depth-ordered stream
and composites them with shared transmittance. Training already provides
separate Adam optimizers and one backward pass from the final image.

The remaining integration problem is therefore not basic rendering plumbing.
It is initialization, training dynamics, adaptation scheduling, and the
ambiguous division of responsibility under a combined RGB loss.

## Experiment A: oracle merge and frozen render

This is a validation experiment, not the proposed training method.

Implement a branch-level checkpoint importer that constructs one hybrid model
from:

1. the Gaussian branch of the qualified Gaussian-only checkpoint; and
2. the medium branch of the qualified Voronoi-only checkpoint.

No optimizer step or adaptation event is allowed in this experiment. Render
the aligned `lego_both` test cameras in hybrid, Gaussian-only, and Voronoi-only
modes. The importer must first verify compatible appearance mode, cameras,
coordinate system, bounds, and renderer settings. Each imported branch rendered
alone must reproduce its source checkpoint within numerical tolerance.

This experiment answers a narrow but important question: do two independently
working scene representations produce the expected image when passed through
the joint depth/transmittance compositor?

- If the frozen hybrid is already wrong, investigate checkpoint import,
  coordinates, bounds, depth ordering, support, and transmittance.
- If it is correct but degrades after optimization, the problem is joint
  training rather than composition.

Because the branch checkpoints were trained using separated targets, this
oracle merge must be labelled clearly in outputs and must never become the
initialization of the final unsupervised hybrid result.

## Experiment B: clean joint baseline

Train a new hybrid model using only images from `lego_both`. Do not initialize
it from either standalone checkpoint. The separated datasets may be loaded on
CPU for test-only diagnostics, but must have no path to the training loss.

The first run should combine the qualified settings without adding a new loss:

- black background
- RGB appearance
- full-image L1 + 0.2 DSSIM
- Gaussian hit buffer K=64
- GraphDECO Gaussian adaptation without opacity reset
- adaptive Voronoi from 2,048 to at most 8,192 sites
- the qualified learning-rate schedules and existing scale regularization
- one shared reconstruction loss on the final hybrid image
- 30k iterations

Do not add alpha/foreground supervision, separation regularization, sparsity
losses, or other disentanglement terms to this run. Its purpose is to reveal
the natural failure mode before attempting to fix it.

## Required diagnostics for the clean joint run

The primary metric is reconstruction quality on `lego_both`, but it is not
sufficient to diagnose representation collapse. Record at least:

- hybrid, Gaussian-only, and Voronoi-only renders on the same test views
- hybrid PSNR/SSIM and test-only layer-aligned metrics
- contribution, alpha, visibility, and active-population statistics per branch
- Gaussian and medium gradient norms and learning rates
- Gaussian count, Voronoi site count, and every adaptation event
- same-batch quality changes around adaptation events
- Gaussian K=64 hit-count percentiles and buffer-full fraction

Layer-aligned diagnostics may explain leakage or collapse, but they must not be
used in the loss or to select the final method.

## Decision after the first joint run

Only after inspecting the clean run should the next intervention be selected.
The likely cases are:

- correct composition but one branch receives little gradient;
- both branches explain the same pixels;
- Gaussian adaptation or Voronoi topology changes destabilize the other branch;
- strong reconstruction with poor representation separation;
- a renderer capacity or transmittance problem.

The next sweep should be small and targeted at the observed case, primarily
testing update/adaptation scheduling before introducing new loss terms. This
keeps representation qualification, topology behaviour, and disentanglement
loss design as separate experimental variables.
