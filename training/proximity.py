"""Slower Voronoi updates for cells that contain the Gaussian surface.

A cell is a surface cell when it contains the center of at least one eligible
Gaussian (opacity above a threshold, close to the COLMAP points), optionally
dilated by one ring of Delaunay neighbors. Its optimizer step is scaled by a
multiplier m after Adam has run (Adam is invariant to gradient scale, so a
gradient multiplier would have no effect).
"""
from __future__ import annotations

import torch


@torch.no_grad()
def nearest_sites(points: torch.Tensor, sites: torch.Tensor) -> torch.Tensor:
    """Voronoi cell index of every point (exact nearest site, chunked)."""
    out = torch.empty(points.shape[0], dtype=torch.long, device=points.device)
    chunk = max(1024, int(1.5e8 // max(sites.shape[0], 1)))
    for start in range(0, points.shape[0], chunk):
        out[start:start + chunk] = torch.cdist(points[start:start + chunk], sites).argmin(1)
    return out


@torch.no_grad()
def surface_cell_mask(sites, gaussian_means, eligible, first=None, second=None):
    """Cells containing an eligible Gaussian center; dilated if edges are given."""
    mask = torch.zeros(sites.shape[0], dtype=torch.bool, device=sites.device)
    points = gaussian_means[eligible]
    if points.numel():
        mask[nearest_sites(points, sites)] = True
    if first is not None and first.numel():
        grown = mask.clone()
        grown[second[mask[first]]] = True
        grown[first[mask[second]]] = True
        mask = grown
    return mask


@torch.no_grad()
def damp_step(parameter: torch.Tensor, before: torch.Tensor, mask: torch.Tensor, multiplier: float) -> None:
    """Keep only ``multiplier`` of the last step for rows in ``mask``.

    ``before`` holds the masked rows as they were before the optimizer step.
    """
    after = parameter.data[mask]
    parameter.data[mask] = before + multiplier * (after - before)


def proximity_multiplier(config, iteration: int) -> float:
    """1 before proximity_start, then a linear ramp down to proximity_multiplier."""
    target = float(config.proximity_multiplier)
    if target >= 1.0 or iteration < config.proximity_start:
        return 1.0
    if config.proximity_ramp == 0:
        return target
    progress = min((iteration - config.proximity_start) / config.proximity_ramp, 1.0)
    return 1.0 - (1.0 - target) * progress
