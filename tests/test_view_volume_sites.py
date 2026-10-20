"""Voronoi sites sampled in the volume the training cameras see (profile.spatial_view_*)."""
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from training.warmup_profile import WarmupProfile
from training.warmup_trainer import initialize_spatial_medium, sample_view_volume

IUI3 = Path("/shared/sets/datasets/SeathruNeRF_dataset/IUI3-RedSea-Undistorted")


def camera(origin):
    return dict(origin=np.array(origin, float), right=np.array([1.0, 0, 0]), down=np.array([0, 1.0, 0]),
                forward=np.array([0, 0, 1.0]), fov_x=math.radians(90), fov_y=math.radians(60))


def test_sites_lie_in_a_frustum_within_the_depth_range():
    np.random.seed(0)
    cameras = [camera([0, 0, 0]), camera([10, 0, 0])]
    points = np.array([[0, 0, 4.0], [10, 0, 4.0], [5, 0, 4.0]])  # median distance to a camera: 4
    sites, report = sample_view_volume(points, cameras, 4000, 2.0)
    assert sites.shape == (4000, 3) and report["view_far"] == pytest.approx(8.0)
    depth = sites[:, 2]
    assert depth.min() >= 0.2 - 1e-5 and depth.max() <= 8.0 + 1e-5
    lateral = np.minimum(np.abs(sites[:, 0]), np.abs(sites[:, 0] - 10))
    assert (lateral <= depth * math.tan(math.radians(45)) + 1e-4).all()
    assert (np.abs(sites[:, 1]) <= depth * math.tan(math.radians(30)) + 1e-4).all()
    # Uniform in volume: the frustum's far half (depth 4-8) holds 7/8 of the volume.
    assert abs((depth > 4).mean() - 7 / 8) < 0.03


def test_profile_validates_view_settings():
    WarmupProfile.from_dict(dict(site_initialization="spatial", outer_points=0, spatial_view_fraction=0.75))
    for bad in (dict(spatial_view_fraction=1.5), dict(spatial_view_depth=0.0)):
        with pytest.raises(ValueError, match="view"):
            WarmupProfile.from_dict(dict(site_initialization="spatial", outer_points=0, **bad))


@pytest.mark.skipif(not IUI3.is_dir(), reason="dataset not on this machine")
def test_initialization_puts_the_fraction_in_view():
    from types import SimpleNamespace
    from foamsurf_data.colmap import ColmapDataset
    np.random.seed(1)
    dataset = ColmapDataset(str(IUI3), resolution=16, device="cpu", llffhold=8)
    medium = SimpleNamespace(sites=torch.zeros(1, 3))
    config = SimpleNamespace(model=SimpleNamespace(medium_bbox_min=None, medium_bbox_max=None, medium_points=0))
    profile = WarmupProfile(site_initialization="spatial", outer_points=0, spatial_count=2000,
                            spatial_colmap_fraction=0.0, spatial_extent=2.0, spatial_view_fraction=0.75)
    initialize_spatial_medium(medium, dataset, config, profile)
    report = medium.initialization_report
    assert medium.sites.shape == (2000, 3) and config.model.medium_points == 2000
    assert report["view_count"] == 1500 and report["spatial_count"] == 500
    cameras = torch.stack([c.origin for c in dataset.train])
    distance = torch.cdist(medium.sites.detach(), cameras).amin(1)
    assert (distance[-1500:] <= report["view_far"] * 1.6).all()  # frustum corners reach ~1.5x depth
