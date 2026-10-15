# Standalone FoamSurf hybrid renderer

This directory is a self-contained Python/CUDA/OptiX project. Its public API is
`import foamsurf`; the runtime does not import RadFoam, FALCON or `flare_new`,
and it does not search for neighbouring source trees. The FlaRe-base OptiX
sources required to generate Gaussian intersections are vendored in
`third_party/flare_renderer/`.

This document is the implementation contract for work on the CUDA renderer. It
describes the current code, including its limitations, rather than a proposed
one-kernel design.

## Current architecture

One Python render call currently performs two native stages:

```text
NativeHybridRenderer.render(...)
  |
  +-- optional: exact nearest Voronoi site for every unique ray origin
  |
  +-- FlaRe-base OptiX Render
  |     output: sorted Gaussian hit_depths and hit_indices, shape [64, R]
  |
  +-- foamsurf._C.hybrid_forward (scalar medium)
  |     hybrid_forward_kernel, 128 threads/block, one thread/ray
  |     Voronoi traversal + Gaussian evaluation + fused depth compositing
  |     output: RGBA [R, 4] and traversal count [R]
  |   or foamsurf._C.rgb_forward (RGB medium, see "RGB medium" below)
  |     output: direct, backscatter, direct transmittance [R, 3] each
  |
  +-- foamsurf._C.surface_hits: first valid Gaussian hit per ray
```

Thus “hybrid/fused renderer” means that Voronoi segments and Gaussian events
are merged and composited in one CUDA traversal. It is not yet one literal
OptiX launch. Removing the intermediate hit buffer and moving traversal into
one OptiX program is a future optimization and must preserve this contract.

## Source map

```text
renderer/
  foamsurf/
    __init__.py       exported public API
    types.py          GaussianScene, VoronoiScene, HybridRender contracts
    cuda.py           native lifecycle, validation and custom autograd bridge
    hybrid.py         slow differentiable PyTorch correctness oracle
    benchmark.py      complete-call benchmark
  native/
    bindings.cpp      PyTorch extension checks and Python binding
    hybrid.h          native launch interface
    hybrid_cuda.cu    forward/backward traversal and compositing
  third_party/
    flare_renderer/   vendored FlaRe-base OptiX hit-generation implementation
  tests/
    test_hybrid_renderer.py
    test_native_hybrid_renderer.py
    test_renderer_native_equivalence.py
  build.sh            builds CUDA extension and vendored OptiX backend
  build_optix.sh      builds only the OptiX backend
  setup.py            independent Python extension build
```

The generated files `foamsurf/_C.so`, `foamsurf/_optix_k64/` and `build/` are
ignored by Git and must never be committed.

## Public scene contract

All native inputs are contiguous CUDA `float32`, unless a different dtype is
specified below. Rays have shape `[..., 6]` and contain
`(origin_x, origin_y, origin_z, direction_x, direction_y, direction_z)`.
Directions are normalized by the Python frontend.

### Gaussian surface

For `G` oriented planar Gaussians, `GaussianScene` contains:

| tensor | shape | stored value | value used by renderer |
|---|---:|---|---|
| `means` | `[G,3]` | centre in world coordinates | unchanged |
| `log_scales` | `[G,2]` | log in-plane radii | `exp(log_scales)` |
| `quaternions` | `[G,4]` | `(w,x,y,z)` | normalized rotation |
| `colors` | `[G,3]` | RGB for this camera | unchanged |
| `opacity_logits` | `[G,1]` | unconstrained | `sigmoid(logit)` |
| `raw_kappa` | `[G,1]` | unconstrained | `1 + softplus(raw)` |

`colors` are already view-dependent values. If the model stores spherical
harmonics, Python evaluates SH once per camera using the camera origin and
Gaussian positions before calling the renderer. SH is intentionally not
implemented inside the kernel.

### Voronoi medium

For `V` sites, `VoronoiScene` contains:

| tensor | shape | meaning |
|---|---:|---|
| `sites` | `[V,3]` | Voronoi generators in world coordinates |
| `densities` | `[V,1]` | non-negative extinction coefficient |
| `colors` | `[V,3]` | constant emitted RGB in a cell |

Topology is a CUDA CSR graph:

- `adjacency: uint32[E]` stores neighbouring site indices;
- `offsets: uint32[V+1]` gives the interval for each site;
- `start_points: uint32[R]` identifies the cell containing every ray origin.

If `start_points` is omitted, Python finds the exact nearest site for each
unique camera origin, then broadcasts it to that camera's rays. When all rays
share one origin (one camera) this is a single lookup without `torch.unique`.
The lookup is chunked brute force and does not call RadFoam.

### RGB medium

`RGBVoronoiScene` adds per-channel `attenuation` and `backscatter` `[V,3]`.
Per ray and channel `c` the renderer computes a direct pass (density
`attenuation[:, c]`, medium colour 0) and a backscatter pass (density
`backscatter[:, c]`, Gaussian colour and background 0); see
`rgb_medium.reference_segments`. `foamsurf._C.rgb_forward`/`rgb_backward`
(`rgb_forward_kernel`, `rgb_backward_kernel`) do all six passes in one
traversal: each pass keeps its own transmittance, updated with the scalar
kernel's float operations and checked at the same points, so the forward
output equals six scalar launches bit for bit. Backward sums gradients that go
to the same address in registers (a Gaussian event's alpha and endpoint terms,
a boundary's sites, a cell's medium parameters) before the global atomics.
`NativeHybridRenderer.rgb_medium_backend = "six_pass"` (or
`FOAMSURF_RGB_MEDIUM_BACKEND=six_pass`) selects the former six launches of the
scalar kernel (`rgb_medium.compose_cuda`) for comparisons.

## Geometry lifecycle

Geometry synchronization is deliberately outside the render hot path:

```python
renderer = foamsurf.NativeHybridRenderer(max_batch_size=800 * 800)
renderer.set_gaussian_geometry(gaussian_scene)
renderer.set_voronoi_geometry(medium_scene, adjacency, offsets)

result = renderer.render(
    rays,
    gaussians=gaussian_scene,
    voronoi=medium_scene,
)
```

Call `set_gaussian_geometry` after Gaussian positions, scales, orientations,
opacity or kappa change. This rebuilds/refits the OptiX-side geometry. Gaussian
color-only changes do not require it.

Call `set_voronoi_geometry` after site storage or Delaunay connectivity
changes. Density/color-only changes do not require it. The frontend checks the
site and Gaussian data pointers to catch stale geometry.

Pure modes use the same kernel and an empty opposite representation:

```python
hybrid = renderer.render(rays, gaussians=g, voronoi=v)
surface = renderer.render_gaussians(rays, g)
medium = renderer.render_voronoi(rays, v)
```

This makes isolated validation directly comparable to hybrid rendering.

## Gaussian event

Let ray `x(t) = o + td`, Gaussian centre `m`, and the normalized quaternion
rotation rows be `r0, r1, r2`; `r2` is the plane normal. The saved OptiX hit
depth is used in forward. The analytic backward reconstructs

```text
t = dot(r2, m - o) / dot(r2, d)
p = o + t d - m
u = exp(-s_x) dot(r0, p)
v = exp(-s_y) dot(r1, p)
q = max(u^2 + v^2, epsilon)
kappa = 1 + softplus(raw_kappa)
response = exp(-0.5 * q^kappa / kappa)
alpha_g = sigmoid(opacity_logit) * response
```

and composites one event with the Gaussian RGB. The OptiX intersection radius
is controlled independently by `gaussian_radius_squared` (default 11.3449)
and `number_of_sides` (default 8).

## Voronoi boundary and medium segment

For current site `p_i` and neighbour `p_j`, their bisector plane has normal
`n = p_j - p_i` and contains `(p_i+p_j)/2`. Only faces facing the ray
(`dot(n,d) > 0`) are candidates. Their depths are

```text
t_ij = dot((p_i + p_j)/2 - o, n) / dot(n, d).
```

The smallest valid depth is the exit face and `j` becomes the next cell. For
a cell-`i` segment `[t0,t1]`:

```text
delta = max(t1 - t0, 0)
alpha_m = 1 - exp(-density_i * delta).
```

This is piecewise-constant emission/absorption, not multiple scattering or a
physical participating-media solver.

## Interleaving and compositing

The hit buffer is depth-sorted. For each ray the CUDA thread repeatedly:

1. finds the next Voronoi boundary;
2. consumes every Gaussian hit before that boundary;
3. composites the medium from the cursor to the Gaussian;
4. composites the Gaussian;
5. after all earlier Gaussians, composites medium to the boundary;
6. enters the adjacent Voronoi cell and repeats.

Every event uses front-to-back alpha compositing:

```text
weight = T * alpha
C = C + weight * color
T = T * (1 - alpha).
```

At termination, `C = C + T * background` and output alpha is `1-T`.
Traversal stops at the far plane, below `transmittance_threshold`, or after
`max_intersections` cell-loop iterations. `near` and `far` apply to both
representations.

## Backward contract

`_HybridCudaFunction` is a custom PyTorch autograd operation. PyTorch does not
record the internal CUDA arithmetic and does not differentiate through OptiX.
The function saves scene tensors, rays, hit buffers and forward RGBA, then
launches `hybrid_backward_kernel`, again with one CUDA thread per ray.

Backward remarches the same ordered events and analytically reverses
front-to-back compositing. Gaussian geometry uses a small forward-mode dual
number internally for its 11 scalar geometry/shape variables; accumulated
per-ray derivatives are atomically added to global parameter gradients.
Voronoi site derivatives include how both endpoints change a segment length
through the bisector-plane depth.

Gradients are produced for:

- Voronoi sites, densities and colors;
- Gaussian means, log-scales, quaternions, colors, opacity logits and kappa.

There are intentionally no gradients for CSR topology, start indices, hit
indices/depth buffers, rays, or background. Connectivity is a discrete
structure and is periodically rebuilt by the model layer. Any CUDA rewrite
must continue to pass gradients to **both** representations from one hybrid
loss.

## Returned values

`NativeHybridRender` exposes:

- `image [...,3]`: composited RGB;
- `alpha [...,1]` and `transmittance = 1-alpha`;
- `surface_depth [...,1]`: first valid Gaussian hit or `-1`;
- `surface_index [...,1]`: its Gaussian index or `-1`;
- raw `hit_depths`, `hit_indices`, and per-ray Voronoi
  `num_intersections`.

## Current hard limits and CUDA work queue

- The vendored OptiX hit buffer has `HIT_BUFFER_SIZE = 64` (the only size
  built and used). More than 64 Gaussian intersections on one ray are truncated
  before hybrid compositing.
- A complete render has two native stages and materializes `[64,R]` depth
  and index buffers (`surface_depth`/`surface_index` are read from them by
  `foamsurf._C.surface_hits`).
- Nearest starting-cell lookup is Python/PyTorch brute force (once per
  distinct ray origin).
- The native backend is CUDA float32 and single-GPU.
- Topology construction/rebuild is outside this standalone renderer.
- Rays/background and discrete topology are non-differentiable.
- SH is evaluated by the model before rendering.
- Atomic parameter accumulation makes gradients numerically close, not
  bitwise deterministic.

The highest-value renderer optimization is a single OptiX traversal that emits
Gaussian and Voronoi events directly without the fixed hit buffer. It should be
introduced behind the same Python/types contract and compared against both the
PyTorch oracle and this implementation before replacing it.

## Build and tests

```bash
cd renderer
./build.sh            # TORCH_CUDA_ARCH_LIST selects the GPU architecture (8.9 default, 8.6 for RTX 3090)

# Oracle, native values/gradients and external baseline equivalence
# (from the repository root: some tests import the training code).
cd .. && python -m pytest -q renderer/tests

# Synthetic full-call benchmark; resolution means an R x R ray grid.
python -m foamsurf.benchmark --resolution 800
# Real-checkpoint equivalence and benchmark: scripts/renderer_check.py (repository root).
```

The external-equivalence test may import original implementations as test
oracles. Those imports are test-only and are not package dependencies.

## Optional B0.1 responsibility diagnostics

Calling render with collect_diagnostics=True and an optional diagnostic_target
adds a separate forward-only CUDA instrumentation launch after the ordinary
hybrid forward. It does not participate in autograd, alter RGBA, or allocate
R x G / R x V storage. With the default collect_diagnostics=False, the binding
is not called at all.

NativeHybridRender.diagnostics then contains:

| tensor | shape | columns |
|---|---:|---|
| ray | [...,8] | Gaussian weight, medium weight, medium path, transmittance-weighted path, Gaussian event count, medium segment count, final T, traversal-truncated flag |
| gaussian | [G,4] | contribution, visibility, alpha sum, max touching-ray RGB error |
| medium | [V,5] | contribution, visibility, path, transmittance-weighted path, max traversing-ray RGB error |

Contribution is accumulated T_before * alpha. Positive maximum errors use
atomic max; other primitive columns use atomic add. With no diagnostic target,
max-error values remain zero. The normal render remains the source of RGB and
gradients. B0.1 times it separately from the instrumented pass so hybrid FPS
is not contaminated by diagnostics overhead.

Any CUDA rewrite must pass tests/test_b01_diagnostics.py: statistics on/off
must produce equal RGB, alpha and gradients for every trainable Gaussian and
Voronoi tensor.
