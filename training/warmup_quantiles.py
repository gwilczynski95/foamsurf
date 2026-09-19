"""Warmup-only autograd pass; no change to installed hybrid renderer."""
from functools import lru_cache
from pathlib import Path
import torch


@lru_cache(maxsize=1)
def bindings():
    from torch.utils.cpp_extension import load
    source = Path(__file__).resolve().parents[1] / "renderer/native/warmup_quantiles.cu"
    return load(name="foamsurf_warmup_quantiles", sources=[str(source)],
                extra_cuda_cflags=["-O3", "-lineinfo"], extra_cflags=["-O3"], verbose=True)


class _Quantiles(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, sigma, adjacency, offsets, rays, starts, q, settings):
        tensors = tuple(t.contiguous() for t in (points, sigma, adjacency, offsets, rays, starts, q))
        depth, owner = bindings().forward(list(tensors), *settings)
        ctx.save_for_backward(*tensors, depth, owner)
        ctx.settings = settings
        ctx.mark_non_differentiable(owner)
        return depth, owner

    @staticmethod
    def backward(ctx, grad_depth, grad_owner):
        *tensors, depth, owner = ctx.saved_tensors
        pg, dg = bindings().backward(tensors, depth, owner, grad_depth.contiguous(), *ctx.settings)
        return pg, dg, None, None, None, None, None, None


def render_quantiles(renderer, scene, rays, q, *, gaussians=None):
    if gaussians is not None:
        raise ValueError("depth quantiles support Voronoi-only initialization")
    if rays.requires_grad:
        raise ValueError("warmup requires fixed camera rays")
    if q.ndim != 2 or q.shape[0] != rays.reshape(-1, 6).shape[0] or q.shape[1] == 0:
        raise ValueError("quantiles must have shape [R,Q>0]")
    if not bool(torch.isfinite(q).all()) or bool(((q <= 0) | (q >= 1)).any()) or bool((q[:, 1:] > q[:, :-1]).any()):
        raise ValueError("quantiles must be descending and strictly inside (0,1)")
    if not rays.is_cuda or rays.dtype != torch.float32 or q.device != rays.device or q.dtype != torch.float32:
        raise ValueError("rays/quantiles must be CUDA float32 on the same device")
    scene.validate()
    topology = renderer._voronoi_topology
    if topology is None or topology.site_data_ptr != scene.sites.data_ptr():
        raise ValueError("synchronize Voronoi topology first")
    rays = rays.reshape(-1, 6).contiguous()
    if rays.shape[0] > renderer.max_batch_size:
        raise ValueError("too many rays")
    with torch.no_grad():
        origins, inverse = torch.unique(rays[:, :3], dim=0, return_inverse=True)
        starts = renderer._nearest_sites(scene.sites, origins)[inverse].to(torch.uint32)
    return _Quantiles.apply(scene.sites, scene.densities, topology.adjacency,
        topology.offsets, rays, starts, q,
        (renderer.near, renderer.far, renderer.transmittance_threshold, renderer.max_intersections))
