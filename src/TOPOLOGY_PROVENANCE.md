# FoamSurf topology component: provenance and scope

FoamSurf uses an adapted subset of the CUDA Delaunay implementation from
[Radiant Foam](https://github.com/theialab/radfoam), upstream commit
`3e7b52cf74e37ab2ab5e695f53570f515f537e3d`.

## What is retained

Only code required to construct and update the Voronoi/Delaunay topology is
part of the active FoamSurf build:

- `src/delaunay/`: CUDA Delaunay construction and connectivity;
- `src/aabb_tree/`: spatial sorting/search required internally by Delaunay;
- `src/utils/`: low-level CUDA containers, geometry and launch helpers;
- `torch_bindings/triangulation_bindings.*`: PyTorch-facing topology API;
- the Eigen submodule used by the geometry types.

The Python package is named `foamsurf_topology`. The CMake project,
distribution metadata, native target and binding documentation are all named
for FoamSurf.

## What is intentionally not retained

The upstream training application, renderer pipeline, viewer, batch fetcher,
dataset loader, configurations and GUI/OpenGL dependencies are not part of
FoamSurf. The FoamSurf renderer is the standalone package under `renderer/`.

## Why internal `radfoam::` namespaces remain

The adapted CUDA/C++ source retains upstream `radfoam::` namespaces and
`RADFOAM_HD` macros internally. They are implementation-level provenance, not
the product or Python API. Mechanically renaming thousands of internal symbols
would:

- obscure line-by-line comparison with upstream;
- make future attribution and security review harder;
- create a large, behaviour-neutral diff with additional regression risk.

No public FoamSurf command or Python module imports `radfoam`. The only
public topology import is:

```python
import foamsurf_topology
triangulation = foamsurf_topology.Triangulation(points)
```

## License

The retained component remains covered by the upstream Apache-2.0 license in
the repository root. FoamSurf documentation and publications must continue to
credit Radiant Foam for this adapted topology implementation.
