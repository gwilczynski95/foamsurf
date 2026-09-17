# FoamSurf

FoamSurf is a research codebase for jointly learning two scene representations:

- **surface:** oriented 2D Gaussian splats derived from the FlaRe/FALCON base
  representation;
- **medium:** a piecewise-constant emission/absorption field carried by 3D
  Voronoi cells derived from Radiant Foam.

The standalone representations have been qualified, and the current
disentanglement baseline initializes FoamSurf from a separated RadFoam spatial
prior. On the reference Lego run it reaches 34.824 dB hybrid PSNR while
assigning 99.71% of cloud-only responsibility to Voronoi and 99.45% of
Lego-only responsibility to Gaussians. See
[the current RadFoam-prior baseline](docs/RADFOAM_PRIOR_HANDOFF_TRAINING.md)
for the exact inputs, initialization, launch command, results and analysis
protocol. The earlier standalone stage is documented in
[the Stage 1/2 guide](docs/STANDALONE_REPRESENTATIONS.md).

## What B0 does—and does not do

At every step B0 samples rays from one camera, evaluates directional Gaussian
appearance, renders both representations in one depth-ordered pass, computes

```text
L_B0 = mean((RGB_hybrid - RGB_target)^2),
```

then steps separate Gaussian and medium Adam optimizers from the same backward.

B0 deliberately has no depth-quantile, opacity, density, sparsity, separation,
or other regularizer. Gaussian adaptation is available in B1. Adaptive Voronoi split/prune is now available for Stage 2 and remains disabled unless `voronoi_adaptation.enabled` is set. Neither mechanism adds a surface-aware or oracle loss; see [the implementation roadmap](docs/IMPLEMENTATION_ROADMAP.md).

## Repository map

```text
train_b0.py                 B0 command-line entry point
configs/b0_lego.yaml        canonical B0 configuration
configs/b01_lego.yaml       B0.1 Lego diagnostics configuration
foamsurf_data/              Blender and COLMAP cameras, images and rays
foamsurf_model/             Gaussian model, Voronoi medium and topology owner
training/                   trainer, loss, optimizers, logging and checkpoints
evaluation/                 three-mode validation, metrics and images
renderer/                   standalone `foamsurf` Python/CUDA/OptiX renderer
tests/b0/                   B0 contract tests
scripts/train_b01_lego.sbatch Slurm entry point for B0.1 Lego
scripts/smoke_test_b0.py    tiny end-to-end train/resume smoke test
scripts/test_stage12.sbatch  standalone contract and CUDA regression tests
scripts/sweep_stage{1,2}_*.sbatch  overnight standalone sweeps
docs/                       developer walkthrough and implementation roadmap
```

The active codebase is FoamSurf. The CUDA Delaunay/AABB topology component is
adapted from Radiant Foam and exposed only as `foamsurf_topology`; provenance
is documented in [src/TOPOLOGY_PROVENANCE.md](src/TOPOLOGY_PROVENANCE.md).

Read [the repository walkthrough](docs/REPOSITORY_WALKTHROUGH.md) before
developing the model, and [the renderer documentation](renderer/README.md)
before changing the CUDA/OptiX path.

## Build

Requirements are Linux, Python 3.10+, CUDA-enabled PyTorch, a matching CUDA
toolkit, NVIDIA OptiX headers and a compatible driver. Initialize the retained
Eigen submodule after cloning with `git submodule update --init --recursive`. In the `foamsurf` conda
environment run:

```bash
cd /shared/results/gmwilczn/foamsurf/foamsurf
CUDA_VISIBLE_DEVICES=1 ./build.sh
```

`build.sh` installs requirements, builds the local Delaunay binding, builds the
standalone renderer and verifies the imports. The renderer vendors every
required source file; it does not search for a neighbouring RadFoam, FALCON or
`flare_new` checkout at runtime.

## Qualify representations separately

Use `configs/stage1_gaussian_only.yaml` and `configs/stage2_voronoi_only.yaml`. The exact mode contract, adaptive Voronoi policy, tests and 94-run sweep matrix are documented in [the standalone qualification guide](docs/STANDALONE_REPRESENTATIONS.md).

## Train B0.1 diagnostics on NeRF Synthetic Lego

The B0.1 run keeps the exact B0 loss/model and enables reduced per-ray and
per-primitive diagnostics:

~~~bash
sbatch scripts/train_b01_lego.sbatch
~~~

For an interactive run use config configs/b01_lego.yaml. See
[the B0.1 diagnostics guide](docs/B01_DIAGNOSTICS.md) for all metrics, images,
CSV schemas and interpretation.

### Layer-ground-truth entanglement experiment

The aligned FoamSurf Lego dataset trains only on `lego_both`. Its `lego_lego`
and `lego_clouds` test images are evaluation-only targets for the Gaussian and
Voronoi branches respectively:

~~~bash
sbatch scripts/train_b01_lego_layers.sbatch
~~~

W&B exposes separate `lego_lego`, `lego_clouds`, and `entanglement` sections.
The same scalar results are retained in `metrics/eval.csv`; aligned images and
region masks are saved under each `evaluation/iteration_*` directory. The
layer targets are kept on CPU and one view is moved to the GPU at evaluation
time, so they do not triple persistent GPU image memory.

## Train B1 Gaussian adaptation

B1 keeps the Voronoi topology fixed and applies the effective policy of the
pinned official 2DGS implementation to the surface Gaussians: full images,
cameras without replacement, L1+DSSIM, opacity reset, and clone/split/prune
without a 200k cap:

~~~bash
sbatch scripts/train_b1_2dgs_baseline.sbatch
~~~

The 30k run logs operation counts, Gaussian count, memory, layer diagnostics,
images and metrics to CSV and W&B. It saves both last and best-PSNR
checkpoints. See [the 2DGS baseline guide](docs/B1_2DGS_BASELINE.md) for the
exact pinned sources, policy, and intentional hybrid-model differences.

## Train B0 on NeRF Synthetic Lego

The supplied configuration enables W&B online logging:

```bash
CUDA_VISIBLE_DEVICES=1 python train_b0.py \
  --config configs/b0_lego.yaml
```

Useful one-off overrides are:

```bash
CUDA_VISIBLE_DEVICES=1 python train_b0.py \
  --config configs/b0_lego.yaml \
  --source-path /home/gmwilczn/nerf_synth_mipnerf/lego \
  --output-dir outputs/my-b0-lego \
  --wandb-mode offline
```

Use `--wandb-mode disabled` for no W&B process. An offline run can later be
uploaded with `wandb sync`.

Resume while preserving both model branches, both Adam optimizers, RNG state,
elapsed training time and topology revision:

```bash
CUDA_VISIBLE_DEVICES=1 python train_b0.py \
  --config outputs/b0-lego/config.yaml \
  --resume outputs/b0-lego/checkpoints/last.pt \
  --iterations 35000
```

`optimization.iterations` is the final iteration number, not the number of
additional steps after resume.

## Outputs

```text
outputs/b0-lego/
  config.yaml
  summary.json
  checkpoints/
    last.pt                  newest periodic/evaluated state
    best.pt                  highest validation hybrid PSNR
  metrics/
    train.csv                loss, PSNR, gradients and parameter diagnostics
    eval.csv                 metrics/FPS for all three render modes
  evaluation/
    iteration_XXXXXX/
      per_view.csv
      *_panel.png            target | hybrid | error | Gaussian | Voronoi | T
      *_hybrid.png
      *_gaussian_only.png
      *_voronoi_only.png
      *_transmittance.png
      *_error.png
      *_responsibility.png   Gaussian | medium | medium fraction
      gaussian_primitives.csv
      medium_primitives.csv
  wandb/                     online/offline W&B run data
```

Validation always renders **hybrid**, **Gaussian-only** and **Voronoi-only**
from identical rays. W&B receives the CSV scalars, comparison panels and
histograms of Gaussian opacity/scale and medium density/color. B0.1 additionally logs contribution, visibility, path length, max ray error and per-primitive gradient magnitude.

## Tests

```bash
# Fast B0 contracts.
python -m pytest -q tests/b0

# Native train, validation, checkpoint and resume.
CUDA_VISIBLE_DEVICES=1 python scripts/smoke_test_b0.py

# Complete standalone renderer suite.
cd renderer
CUDA_VISIBLE_DEVICES=1 python -m pytest -q
```

## Data

`foamsurf_data` accepts:

- Blender/NeRF Synthetic datasets with `transforms_train.json` and
  `transforms_test.json`;
- COLMAP datasets in the common 3DGS/Mip-NeRF 360 layout, including large
  scenes and LLFF-style holdout selection.

## Provenance

FoamSurf adapts its CUDA Delaunay/AABB topology component from the official
[Radiant Foam](https://github.com/theialab/radfoam) implementation (upstream
commit `3e7b52cf74e37ab2ab5e695f53570f515f537e3d`). The standalone renderer also
contains FlaRe/FALCON base OptiX hit-generation sources at the provenance
recorded in `renderer/third_party/flare_renderer/README.md`. Original license
and attribution files are retained. Cite the relevant upstream methods when
publishing FoamSurf results.
