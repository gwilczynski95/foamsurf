"""Per-ray Gaussian hit alphas and the depth where the Gaussian surface becomes opaque."""
from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def sorted_hit_alphas(gaussians, hit_depths, hit_indices, origins, directions, radius_squared):
    """Alpha of every OptiX hit, same planar response as renderer/foamsurf/hybrid.py.

    Inputs are [K, R] hit buffers. Returns (depth, alpha, index), each [K, R],
    sorted front to back; invalid slots get depth=inf and alpha=0.
    """
    valid = hit_indices >= 0
    index = hit_indices.clamp_min(0).long()
    q = F.normalize(gaussians.quaternions.detach(), dim=-1)
    a, b, c, d = q.unbind(-1)
    rows = torch.stack((1 - 2 * (c * c + d * d), 2 * (b * c + a * d), 2 * (b * d - a * c),
                        2 * (b * c - a * d), 1 - 2 * (b * b + d * d), 2 * (c * d + a * b)),
                       -1).reshape(-1, 2, 3)
    local = origins[None] + hit_depths[..., None] * directions[None] - gaussians.means.detach()[index]
    inverse = torch.exp(-gaussians.log_scales.detach()[index])
    r = rows[index]
    u = (r[..., 0, :] * local).sum(-1) * inverse[..., 0]
    v = (r[..., 1, :] * local).sum(-1) * inverse[..., 1]
    r2 = u * u + v * v
    kappa = 1 + F.softplus(gaussians.raw_kappa.detach()[index, 0])
    alpha = (torch.sigmoid(gaussians.opacity_logits.detach()[index, 0])
             * torch.exp(-0.5 * r2.clamp_min(1e-8).pow(kappa) / kappa))
    ok = valid & (r2 <= radius_squared) & (hit_depths >= 0)
    alpha = torch.where(ok, alpha, torch.zeros_like(alpha))
    depth = torch.where(ok, hit_depths, torch.full_like(hit_depths, float("inf")))
    depth, order = depth.sort(0)
    return depth, alpha.gather(0, order), index.gather(0, order)


@torch.no_grad()
def opaque_depth(depth, alpha, threshold=0.5):
    """First hit depth at which the Gaussian transmittance drops below ``threshold``.

    ``depth``/``alpha`` are sorted [K, R] (see sorted_hit_alphas). Returns
    (depth [R], found [R]); depth is +inf where the surface never gets opaque.
    """
    survival = torch.cumprod(1 - alpha, 0)
    below = survival < threshold
    found = below.any(0)
    first = below.float().argmax(0)
    surface = depth.gather(0, first[None])[0]
    return torch.where(found, surface, torch.full_like(surface, float("inf"))), found
