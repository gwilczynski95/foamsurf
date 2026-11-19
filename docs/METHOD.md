# FoamSurf method and implementation

FoamSurf combines oriented 2D Gaussian surfels with a spatially varying Voronoi medium.
This guide describes the release implementation; the resolved YAML configurations define
the actual numerical settings. See [references.bib](references.bib) for related methods.

## Representation and image formation

Surfels carry position, orientation, two tangent scales, opacity and spherical-harmonic
appearance. Voronoi sites define piecewise-constant RGB attenuation coefficients
`sigma_a`, backscatter coefficients `sigma_b`, and medium color `B`. The coefficients
for direct attenuation and backscatter are separate, following the SeaThru model.

For an ordered ray segment of length `l`, let `G` be accumulated surface transmittance
and `A`, `W` the RGB direct and backscatter transmittances. Initially all are one.
Within a cell, the renderer accumulates:

```text
scattered += G * W * B * (1 - exp(-sigma_b * l))
A *= exp(-sigma_a * l)
W *= exp(-sigma_b * l)
```

At the next surfel hit with opacity `alpha` and color `c`:

```text
direct += G * alpha * A * c
G *= 1 - alpha
```

Multiplication is componentwise for RGB quantities. A terminal background contributes
`G * A * background`. Cell boundaries and surfel hits share an ordered event sequence;
the medium integral is analytic within each constant cell. This avoids numerical
quadrature of the medium. Hit-buffer limits and traversal tolerances remain implementation
constraints, so this is not a claim of unlimited exact geometric traversal.

The CUDA/OptiX implementation and its gradients are in [renderer](../renderer/README.md).
[NativeHybridRenderer](../renderer/foamsurf/cuda.py) exposes hybrid, surface-only and
medium-only rendering. [gaussians.py](../foamsurf_model/gaussians.py) implements surfel
parameters and Mip filtering; [medium.py](../foamsurf_model/medium.py) and
[topology.py](../foamsurf_model/topology.py) manage medium coefficients and adjacency.

## Reconstruction

[COLMAP initialization](../training/colmap_joint.py) initializes the surface from sparse
geometry and fits water coefficients using tracks. The SeaThru configuration then jointly
optimizes for 20,000 iterations. Synthetic configurations use a 5,000-step warmup, followed
by 20,000 joint steps for the hybrid initialization or 30,000 for Voronoi-first initialization.
These schedules are separate experiments, not interchangeable defaults.

The [trainer](../training/trainer.py) combines image reconstruction with geometric and
surface–medium priors. The SeaThru configuration uses prediction-normalized L1 and D-SSIM,
OceanSplat terms, opacity binarization, and priors intended to keep surfels on surfaces and
water coefficients smooth. Gaussian densification/pruning and Voronoi adaptation change
representation capacity during training. See [surface–medium priors](../training/colmap_joint_trainer.py)
and [Gaussian adaptation](../foamsurf_model/gaussian_adaptation.py).

PPISP models camera photometric response. Water-light handling includes a
Henyey–Greenstein phase function and an evaluation extension to directions outside the
training views. These operations belong to the full evaluation image and must not be
silently omitted when comparing to the paper. Synthetic insertion controls disable these
corrections. See [camera_isp.py](../training/camera_isp.py),
[water_phase.py](../training/water_phase.py) and
[water_extrapolation.py](../training/water_extrapolation.py).

Print the complete settings instead of copying a partial parameter list:

```bash
python train.py --config configs/seathru.yaml --scene /path/to/undistorted/scene --dry-run
```

The seven ablation configs disable individual components while preserving the remaining
paper settings. Configuration regression tests compare resolved values to paper manifests.

## Evaluation and applications

Evaluation always loads `last.pt`, not a checkpoint selected by test PSNR. SeaThru holds
out every eighth image. Full-image PSNR/SSIM/LPIPS operate on float images; PNG exports
are visualizations. Separation uses provided masks only at evaluation time. Synthetic
layer and transmittance evaluations use the separate ground-truth layers. FPS uses the
paper's K=32 renderer; training uses K=64.

[Insertion](../examples/insertion/) traces objects through a frozen reconstruction.
[Editing](../edit.py) changes surfel geometry or medium parameters without retraining.
[Style transfer](../stylize.py) optimizes a selected component with CLIP losses, then
[composition](../examples/style/compose_style.py) combines surface and water styles from the
same base model. The application runner contains the available paper presets. Bounded
medium removal/draining still requires integration. Reproduction of the paper demo is
outside the release scope; the existing insertion shot remains an optional demo.
