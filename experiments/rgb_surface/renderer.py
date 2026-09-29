"""Opt-in exact cell-boundary gradient using the existing scalar CUDA tracer."""
import time

import torch
import torch.nn.functional as F

from foamsurf import NativeHybridRenderer, VoronoiScene
from .loss import event_moments, local_rays, occupancy_loss, optical_density


@torch.no_grad()
def gaussian_moments(scene, rays, hit_depths, hit_indices, near, length, band):
    ids = hit_indices.t().long()
    depth = hit_depths.t()
    if not ids.numel() or not len(scene.means):
        zero = rays.new_zeros(len(rays))
        return zero, zero, zero
    valid = (ids >= 0) & (ids < len(scene.means)) & torch.isfinite(depth) & (depth >= near) & (depth < length)
    depth = torch.where(valid, depth, torch.zeros_like(depth))
    safe = ids.clamp(0, len(scene.means) - 1)
    w, x, y, z = F.normalize(scene.quaternions.detach()[safe], dim=-1).unbind(-1)
    u = torch.stack((1-2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y)), -1)
    v = torch.stack((2*(x*y-w*z), 1-2*(x*x+z*z), 2*(y*z+w*x)), -1)
    hit = rays[:, None, :3] + depth[..., None] * F.normalize(rays[:, 3:], dim=-1)[:, None]
    offset = hit - scene.means.detach()[safe]
    scale = scene.log_scales.detach()[safe].exp()
    r2 = ((offset*u).sum(-1)/scale[..., 0]).square() + ((offset*v).sum(-1)/scale[..., 1]).square()
    kappa = 1 + F.softplus(scene.raw_kappa.detach()[safe].squeeze(-1))
    alpha = scene.opacity_logits.detach()[safe].sigmoid().squeeze(-1) * torch.exp(-.5*r2.clamp_min(1e-8).pow(kappa)/kappa)
    return event_moments(depth, alpha, valid, band)


class LocalOverlap:
    def __init__(self):
        self.renderer = None
        self.key = None

    def compute(self, native_render, rays, scene, medium, topology, config,
                iteration, extent, far, near, seed):
        weight = config.weight_at(iteration)
        zero = medium.raw_density.new_zeros(())
        logs = dict(raw=0., weighted=0., weight=weight, confidence=0., gaussian_opacity=0.,
                    occupancy=0., saturated_fraction=0., rays=0, hit_full_fraction=0.,
                    traversal_truncated_fraction=0., band_half_width=0., seconds=0.)
        if weight == 0:
            return zero, logs
        started = time.perf_counter()
        count = min(config.rays, len(rays))
        rng = torch.Generator(device=rays.device).manual_seed(seed*1000003+iteration)
        indices = torch.randint(len(rays), (count,), device=rays.device, generator=rng)
        chosen = rays[indices].detach().contiguous()
        length = min(far, config.range_multiplier*extent)
        if not 0 < length < float('inf'):
            raise ValueError('Invalid overlap ray extent')
        band = config.band_fraction*length
        with torch.no_grad():
            g = native_render(chosen, gaussians=scene, background=(0., 0., 0.), collect_diagnostics=False)
            depth, confidence, opacity = gaussian_moments(scene, chosen, g.hit_depths, g.hit_indices, near, length, band)
            windows = local_rays(chosen, depth, band, length)
        # No Gaussian tensors enter the differentiable auxiliary render.
        proxy = VoronoiScene(medium.sites, optical_density(medium.optical_coefficients,
                             medium.colors, config.optical_scale), torch.zeros_like(medium.colors))
        if self.renderer is None:
            self.renderer = NativeHybridRenderer(max_batch_size=config.rays, near=0., far=2*band,
                                                transmittance_threshold=0., max_intersections=4096)
        if self.renderer.far != 2*band:
            raise ValueError('Overlap extent changed within a training run')
        key = (topology.revision, topology.adjacency.data_ptr(), topology.offsets.data_ptr(),
               medium.sites.data_ptr(), len(medium.sites))
        if key != self.key:
            self.renderer.set_voronoi_geometry(proxy, topology.adjacency, topology.offsets)
            self.key = key
        result = self.renderer.render(windows, voronoi=proxy, background=(0., 0., 0.))
        # Native traversal counts are uint32; PyTorch CUDA does not implement
        # comparisons for that dtype. Match the main renderer's int64 check.
        truncated = (result.num_intersections.to(torch.int64) >= self.renderer.max_intersections).float().mean()
        if bool(truncated > 0):
            raise RuntimeError('Local overlap traversal truncated; do not train with incomplete gradients')
        loss = occupancy_loss(result.alpha, confidence)
        logs.update(raw=float(loss.detach()), weighted=float(weight*loss.detach()),
                    confidence=float(confidence.mean()), gaussian_opacity=float(opacity.mean()),
                    occupancy=float(result.alpha.detach().mean()),
                    saturated_fraction=float((result.alpha.detach() > .99).float().mean()),
                    rays=count, band_half_width=band, traversal_truncated_fraction=float(truncated),
                    hit_full_fraction=float((g.hit_indices >= 0).all(dim=0).float().mean()),
                    seconds=time.perf_counter()-started)
        return weight*loss, logs
