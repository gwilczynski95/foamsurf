# Vendored FlaRe-base OptiX component

This directory contains the CUDA/OptiX renderer imported from:

- branch: `origin/generalized-depth-regularization-compound`
- commit: `5510485`

`PYDATALOADER` from the renderer-only branch is legacy and is intentionally not
included.

Build through the standalone FoamSurf renderer:

```bash
cd renderer
./build_optix.sh
```

The build uses the active Python environment and produces:

```text
foamsurf/_optix/PYOPTIXFLARERENDERER.so
foamsurf/_optix/shaders.cu.ptx
```

The public `foamsurf.NativeHybridRenderer` loads this private module. Internal
class, filename and symbol names are retained from upstream to keep this vendored
component directly comparable with its source. They are not public FoamSurf API.
The native module resolves `shaders.cu.ptx` relative to its own
location, so execution does not depend on the current working directory.
