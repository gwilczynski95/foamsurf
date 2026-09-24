"""COLMAP initialization matching local RadFoam initialize_from_pcd.

No spatial normalization/clamping. Keep attributes aligned through subsequent
topology permutations; the original point colors are intentionally unused:
RadFoam starts its SH/DC coefficients at zero (RGB 0.5).
"""
import numpy as np
import torch
from torch import nn
from foamsurf_model.medium import scene_bounds


def sample_pcd(points, outer_points=5000, outer_scale=10.0):
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 5:
        raise ValueError('PCD initialization needs at least five 3D points')
    if not bool(torch.isfinite(points).all()):
        raise ValueError('Non-finite COLMAP points')
    # Order follows RadFoam: outer normal, sampled indices, jitter, density.
    outer = torch.randn(outer_points, 3, device=points.device) * outer_scale
    indices = torch.randint(len(points), (int(0.9 * len(points)),), device=points.device)
    inner = points[indices] + torch.randn_like(points[indices]) * 1e-2
    sites = torch.cat((inner, outer))
    raw = torch.cat((torch.rand(len(inner), 1, device=points.device),
                     torch.full((outer_points, 1), -0.5, device=points.device)))
    mask = torch.arange(len(sites), device=points.device) >= len(inner)
    return sites, raw, mask


@torch.no_grad()
def initialize_pcd_medium(medium, dataset, config, profile):
    if dataset.__class__.__name__ != 'ColmapDataset':
        raise ValueError('PCD warmup requires a COLMAP dataset')
    if config.model.medium_bbox_min is not None or config.model.medium_bbox_max is not None:
        raise ValueError('PCD warmup must not use a manual bounding box')
    points, _ = dataset.initial_point_cloud()
    points = np.asarray(points, dtype=np.float32)
    origins = torch.stack([c.origin for c in dataset.train + dataset.test])
    # Metadata bounds only, not clipping bounds for organic warmup growth.
    medium.bbox_min, medium.bbox_max = scene_bounds(points, origins, config.model)
    sites, density, mask = sample_pcd(torch.as_tensor(points, device=medium.sites.device),
                                     profile.outer_points, profile.outer_scale)
    medium.sites = nn.Parameter(sites)
    medium.raw_density = nn.Parameter(density)
    medium.raw_color = nn.Parameter(torch.zeros_like(sites))
    medium.guard_mask = mask
    config.model.medium_points = int((~mask).sum())


def pcd_initial_count(root):
    from pathlib import Path
    import struct
    from foamsurf_data.colmap import _read_points_text
    sparse = Path(root) / "sparse" / "0"
    if (sparse / "points3D.bin").is_file():
        with (sparse / "points3D.bin").open("rb") as handle:
            count = struct.unpack("<Q", handle.read(8))[0]
    else:
        points, _ = _read_points_text(sparse / "points3D.txt")
        count = len(points)
    if count < 5:
        raise ValueError("PCD warmup requires at least five COLMAP points")
    return int(0.9 * count)
