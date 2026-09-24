# Ray concentration diagnostics — CPU, isolated from training

Code lives ONLY in `experiments/ray_heatmaps/`. Neither the old RGB gate nor
the pending RGB-smoothing gate hashes this sibling directory. No model,
renderer, training/configuration, or existing test files are changed.
This is a read-only checkpoint analysis, NOT a loss or a segmentation label.

Environment: `/shared/results/gmwilczn/conda/envs/foamsurf`.
Checkpoint: `outputs/rgb_medium/512923/run_3/joint/checkpoints/last.pt`, step 30000.
Dataset: `/shared/sets/datasets/SeathruNeRF_dataset/IUI3-RedSea-Undistorted`.
All four held-out cameras are analyzed. No GPU allocation is needed.

## Reproduce

From `/shared/results/gmwilczn/foamsurf/foamsurf`:

```bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
/shared/results/gmwilczn/conda/envs/foamsurf/bin/python -m pytest -q experiments/ray_heatmaps/test_diagnostic.py
/shared/results/gmwilczn/conda/envs/foamsurf/bin/python experiments/ray_heatmaps/analyze.py \
  --run outputs/rgb_medium/512923/run_3 \
  --output outputs/ray_heatmaps/512923_run3_30k_full \
  --resolution 8 --samples 1024 --range-multiplier 8
```

`g++` compiles a small CPU-only helper into the output directory. It uses four
OpenMP threads. No native training extensions are rebuilt.

## Criteria

All diagnostic rays use the original COLMAP camera calibration, scaled to the
diagnostic image size (173 x 115). Depth is Euclidean distance along normalized
rays, in scene units, NOT meters. Analyze 1024 equal-width bins from zero to
L = min(renderer far, 8 x training-camera extent), about 49.32 scene units.
The camera extent is 1.1 times maximum camera distance from their mean.
This finite range is explicit and shared across all views. It is NOT the full
renderer interval (far=1000), nor a physical water boundary.

Nearest-site queries at bin midpoints sample the actual Voronoi field. Physical
aRGB and bRGB are softplus(raw_density), using joint-training beta=1. B is read
with the checkpoint's color activation: here max(0, 0.5 + C0*raw_color).
No alpha GT or semantic labels are used.

For any nonnegative bin mass m, define an effective support length and
concentration:

    Leff = (sum m)^2 / sum(m^2 / dt)
    C = 1 - Leff/L

Empty signals give zero. Uniform mass per unit length gives C=0. A signal
confined to a thin depth interval approaches C=1. This is invariant to uniform
subdivision of a constant field; it does not count cells or Gaussian hits.

Maps (all raw values in [0,1]):

1. `attenuation_concentration`: C(mean(aRGB)*dt), multiplied by
   1-exp(-sum(mean(aRGB)*dt)). Thus insignificant optical thickness stays dark.
   This is concentration of intrinsic attenuation, not rendered opacity.
2. `backscatter_concentration`: concentration of approximate visible backscatter
   bin mass q, multiplied by S=sum(q)/(sum(q)+0.05).
3. `intrinsic_source_concentration`: concentration of mean(bRGB*B)*dt, multiplied
   by the same saturation form applied to its integral. Unlike q this does not
   include medium transmission or Gaussian occlusion. It distinguishes field
   structure from concentration produced purely by visibility.
4. `gaussian_opacity`: Gaussian-only accumulated opacity A in [0,L].
5. `gaussian_surface_confidence`: A*exp(-(std_depth/h)^2), h=0.02L (~0.986).
   Depth mean and standard deviation use front-to-back Gaussian-only weights.
   High opacity distributed over many depths therefore has low confidence.
6. `medium_near_surface`: fraction of q within +/- h of the weighted Gaussian
   depth. It is saved separately, even though it is not in the main 3x3 panel.
7. `suspect_overlap`: C(q)*S*medium_near_surface*gaussian_surface_confidence.
   It asks whether visible medium is concentrated near a confident Gaussian
   layer. It does NOT establish that the medium is incorrectly modeling solid.

q uses separate RGB backscatter transmission exp(-integral b) and Gaussian
transmission; bin emission is B*(1-exp(-b*dt)). Gaussian transmission is evaluated
at the midpoint of accumulated bin optical thickness. This is an approximation
when multiple events share a bin, checked by doubling the sample count.

Gaussian intersections use the actual planar means, quaternion frame, scales,
opacity and kappa. The shape matches alpha = sigmoid(logit)*
exp(-0.5*(u^2+v^2)^kappa/kappa), kappa=1+softplus(raw_kappa).
Accepted support is a circle u^2+v^2 <= 11.3449 in local coordinates, not the
native OptiX octagonal support. There is no hit-count cap or early stop here.
Consequently these are diagnostic approximations, NOT bit-exact native renders.
Gaussian colors are unnecessary for Gaussian-only opacity/depth.

## Reading the images

`*_panel.png` has actual saved native renders in row 1: hybrid, backscatter,
direct-with-background. Rows 2–3 contain the six diagnostic maps listed above
(excluding medium_near_surface). Brighter yellow/white = stronger signal,
dark purple/black = weaker. Colormaps are shared across cameras: [0,1], except
suspect_overlap in the panel uses [0,0.25] to make low scores legible. Values
above 0.25 saturate; the colorbar explicitly shows that range. Individual map
PNGs always use [0,1]. No per-image percentile normalization is applied.

`*_profiles.png` plots a,b and visible backscatter / Gaussian weight per unit
length for four rays. `*_probe_locations.png` marks their locations. Probe 0
is the maximum suspect_overlap pixel; probes 1–3 are fixed image coordinates,
not hand-picked surfaces. The dashed line marks weighted Gaussian depth (not
meaningful if Gaussian opacity is zero).

`*_data.npz` stores raw float maps and per-ray profiles. `report.json` contains
parameters, map statistics and 1024-vs-2048 sample convergence on every eighth
pixel in each image direction. This subsampling is a diagnostic, not a guarantee
that every narrow feature is resolved.

## Limitations and tests

- Uniform dense water can concentrate visible q near the camera. Inspect
  intrinsic_source_concentration too; a hot q map alone is not a surface mask.
- Real water can be present immediately in front of a solid surface. Overlap
  alone is not an error and must not automatically trigger removal.
- A weak overlap score can mean missing/broad/wrong Gaussian geometry, not
  absence of incorrectly represented solid in the medium.
- No multi-view consistency, local graph outlier detection, or semantic GT is
  implemented here. These maps test whether the elementary ray signals help.
- Changing L or h changes the hypothesis. Neither is a learned or calibrated
  physical threshold. The initial exploratory folder without `_full` used too
  short a range and should not be used as the final result.
- CPU tests cover uniform/empty/thin support, subdivision invariance, analytic
  homogeneous backscatter, planar Gaussian alpha/missed rays, and overlap
  localization. Spatial and sampling approximations remain as described above.
