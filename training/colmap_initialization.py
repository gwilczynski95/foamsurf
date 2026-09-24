"""Spatial Voronoi initialization: no guards, no semantic partition."""
import numpy as np
import torch
from torch import nn


def sample_spatial(points, cameras, count, fraction, extent):
    points = np.asarray(points, dtype=np.float32)
    cameras = np.asarray(cameras, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 5:
        raise ValueError('Need at least five COLMAP points')
    if cameras.ndim != 2 or cameras.shape[1] != 3 or len(cameras) == 0:
        raise ValueError('Need camera centers')
    if not np.isfinite(points).all() or not np.isfinite(cameras).all():
        raise ValueError('Nonfinite geometry')
    if count < 32 or not 0 <= fraction <= 1 or not np.isfinite(extent) or extent <= 0:
        raise ValueError('Invalid spatial initialization')
    lo = np.minimum(np.percentile(points,1,axis=0),cameras.min(axis=0))
    hi = np.maximum(np.percentile(points,99,axis=0),cameras.max(axis=0))
    center = (lo+hi)/2
    # Axis-aligned Gaussian: one standard deviation is half the robust extent.
    # extent=1 and 2 give moderate/wide coverage, without clipping to this box.
    scale = np.maximum((hi-lo)/2,1e-3)*extent
    n = int(count*fraction)
    selected = np.random.randint(len(points),size=n)
    jitter = max(float(np.linalg.norm(hi-lo))*1e-3,1e-5)
    near = points[selected] + np.random.normal(size=(n,3))*jitter
    spatial = center + np.random.normal(size=(count-n,3))*scale
    sites = np.concatenate((near,spatial)).astype(np.float32)
    return sites, dict(center=center.tolist(),std=scale.tolist(),colmap_count=n,
                       spatial_count=count-n,jitter=jitter,robust_lower=lo.tolist(),robust_upper=hi.tolist())


@torch.no_grad()
def initialize_spatial_medium(medium,dataset,config,profile):
    if dataset.__class__.__name__ != 'ColmapDataset':
        raise ValueError('Spatial initialization requires COLMAP')
    if config.model.medium_bbox_min is not None or config.model.medium_bbox_max is not None:
        raise ValueError('Spatial initialization must not have manual bounds')
    points,_ = dataset.initial_point_cloud()
    cameras = torch.stack([c.origin for c in dataset.train]).cpu().numpy()
    sites, report = sample_spatial(points,cameras,profile.spatial_count,
                                  profile.spatial_colmap_fraction,profile.spatial_extent)
    device=medium.sites.device
    medium.sites=nn.Parameter(torch.as_tensor(sites,device=device))
    # All are ordinary learnable medium cells, with identical density law.
    medium.raw_density=nn.Parameter(torch.rand(len(sites),1,device=device))
    medium.raw_color=nn.Parameter(torch.zeros(len(sites),3,device=device))
    medium.guard_mask=torch.zeros(len(sites),dtype=torch.bool,device=device)
    medium.bbox_min=medium.sites.detach().amin(0)
    medium.bbox_max=medium.sites.detach().amax(0)
    config.model.medium_points=len(sites)
    medium.initialization_report=report
