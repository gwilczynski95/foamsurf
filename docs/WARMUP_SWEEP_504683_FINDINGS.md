# Warmup array 504683: why it is an ablation, not the reference

Array 504683 used source fingerprint
`4a99fdf3c095adc105ab1754cdd1f4088f44126b6a0e2e2101a90a78620a8813`.
Runs 0-29 completed; 30-34 and 36-49 were cancelled; run 35 failed during
initial triangulation. Its outputs remain immutable evidence for the objective
that was actually run.

## Main result

The implementation matched RadFoam's depth-quantile forward/backward, but the
warmup objective omitted RadFoam's opacity loss. The omission was discovered by
checking the upstream RadFoam history:

- initial RadFoam: `mean((1-opacity)^2)`;
- Blender support commit 77f5cc8: `mean((alpha-opacity)^2)`;
- COLMAP has no alpha and native RadFoam supplies alpha=1.

Consequently, quantile width alone found an invalid shortcut: make both
transmittance thresholds reachable on transparent/background rays by raising
density. Selected final results:

| ID | variant | PSNR | background alpha | fixed-q valid fraction |
|---:|---|---:|---:|---:|
| 8 | no quantile | 24.474 | 0.023 | 0.216 |
| 20 | q weight 1e-6 | 24.483 | 0.028 | 0.226 |
| 22 | q weight 1e-5 | 24.178 | 0.152 | 0.350 |
| 23 | q weight 3e-5 | 22.793 | 0.363 | 0.535 |
| 7 | q weight 1e-4, ramp 10k | 20.411 | 0.832 | 0.875 |
| 26 | q weight 1e-4, no ramp | 23.941 | 0.998 | 0.996 |

Run 7 peaked near 1-2k and then degraded while background alpha rose. These
runs remain useful opacity-loss ablations but cannot judge the intended
RadFoam-like warmup.

## Technical failure

ID35 used eight exact cube corners. They are co-spherical and cube faces are
co-planar, so Delaunay construction failed with
`TriangulationFailedError: ambiguous triangulation`. The warmup now applies a
deterministic sub-per-mille radial staggering to corner guards and tests that
initialization/training preserves them.

## Corrected objective

The initialization-only trainer now uses:

```text
SmoothL1 RGB + opacity_weight * MSE(rendered alpha, sampled target alpha)
             + scheduled depth-quantile width
```

Default opacity weight is 1. Blender uses raw GT alpha; ordinary COLMAP images
produce alpha=1. The full Gaussian+Voronoi Trainer is unchanged. A new source
fingerprint requires a fresh RTX4090 test certificate before any corrected
experiments.
