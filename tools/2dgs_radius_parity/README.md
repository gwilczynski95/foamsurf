# FoamSurf / official 2DGS radius parity

This standalone CUDA smoke test compares the screen-space radius used by
FoamSurf densification and pruning with the integer `radii` returned by the
official 2DGS `diff-surfel-rasterization` CUDA kernel.

The reference is pinned to `hbb1/2d-gaussian-splatting` commit
`335ad612f2e783a4e57b9cbc4d1e167bd599fc98` and its rasterizer submodule commit
`e0ed0207b3e0669960cfad70852200a4a5847f61`.

No upstream source or compiled artifact is stored in FoamSurf. The installer
uses a temporary directory. For CUDA 12.4 it adds only the missing `<cstdint>`
include required by the pinned source.

## Install and run

```bash
conda activate foamsurf
cd /shared/results/gmwilczn/foamsurf/foamsurf
bash tools/2dgs_radius_parity/install_official_rasterizer.sh
python tools/2dgs_radius_parity/radius_parity.py
```

The contract permits at most one pixel of error per visible Gaussian. The
script covers named deterministic cases and 2048 seeded random, in-frustum
planar Gaussians at 800x800, then exits nonzero when parity is violated.

The independent pytest entry point is:

```bash
python -m pytest -q tools/2dgs_radius_parity/test_radius_parity.py
```

The test uses actual upstream CUDA output rather than a Python rewrite of its
equations, so it also detects convention, culling, filtering and rounding
differences.
