# Code from other projects

FoamSurf uses or adapts code from the following projects. Each keeps its own license; the repository
[LICENSE](../LICENSE) covers the rest.

| Component | Used for | Source |
|---|---|---|
| Radiant Foam | CUDA Delaunay triangulation of the Voronoi cells (`src/`, adapted subset) | [theialab/radfoam](https://github.com/theialab/radfoam); see [src/TOPOLOGY_PROVENANCE.md](../src/TOPOLOGY_PROVENANCE.md) |
| FlaRe | OptiX ray tracer for the surfels (`renderer/third_party/flare_renderer`) | see its [README](../renderer/third_party/flare_renderer/README.md) |
| PPISP | Camera model (`third_party/ppisp`, Apache-2.0) | see [third_party/ppisp](../third_party/ppisp/README.md) |
| Eigen | Linear algebra in the native code (git submodule) | [libeigen/eigen](https://gitlab.com/libeigen/eigen) |
| CLIP | Style loss of `stylize.py` (optional dependency) | [openai/CLIP](https://github.com/openai/CLIP) |
| Warp | Triangle-mesh intersection in `insert.py` (optional dependency) | [NVIDIA/warp](https://github.com/NVIDIA/warp) |
| NVIDIA OptiX SDK 8.0 | Ray tracing (downloaded separately) | NVIDIA |
