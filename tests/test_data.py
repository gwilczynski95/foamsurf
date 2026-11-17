"""Datasets: COLMAP / Blender loading, splits, rays, medium site initialization."""
import os
import subprocess
import sys
from pathlib import Path
import pytest
from foamsurf_data.colmap import ColmapDataset
from train import resolve_iui3
import numpy as np
import torch
from PIL import Image
import math
from training.warmup_profile import WarmupProfile
from training.warmup_trainer import initialize_spatial_medium, sample_view_volume
from helpers import LEGO, ROOT, requires_gpu, requires_lego  # noqa: F401


DATA = Path(os.environ.get("FS_DATASETS", "/shared/sets/datasets/SeathruNeRF_dataset")) / "IUI3-RedSea-Undistorted"


@pytest.mark.skipif(not DATA.is_dir(), reason="needs the SeaThru dataset")
def test_validation_holdout_replaces_the_test_views():
    plain = ColmapDataset(DATA, resolution=8, device="cpu")
    held = ColmapDataset(DATA, resolution=8, device="cpu", validation_hold=8)
    train, test = [c.name for c in plain.train], {c.name for c in plain.test}
    assert [c.name for c in held.test] == train[::8]  # every 8th training photo
    assert [c.name for c in held.train] == [n for i, n in enumerate(train) if i % 8]
    assert not test & {c.name for c in held.train + held.test}  # the test views are not used at all
    with pytest.raises(ValueError):
        ColmapDataset(DATA, resolution=8, device="cpu", validation_hold=1)


pytestmark = pytest.mark.skipif(not DATA.is_dir(), reason="needs the SeaThru dataset")


def test_photos_stored_once_with_unchanged_values():
    dataset = ColmapDataset(DATA, resolution=4, device="cpu")
    camera = dataset.train[0]
    H, W = camera.height, camera.width
    with Image.open(next((DATA / "images_4").glob(camera.name + ".*"))) as image:  # the prescaled photos
        rgba = image.convert("RGBA")
    assert rgba.size == (W, H)
    expected = torch.from_numpy(np.asarray(rgba, dtype=np.float32).copy()) / 255.0
    torch.testing.assert_close(camera.foreground, expected[..., :3] * expected[..., 3:4], atol=0, rtol=0)
    torch.testing.assert_close(camera.ground_truth((0.0, 0.0, 0.0)), expected[..., :3], atol=0, rtol=0)
    assert camera.alpha.shape == (H, W, 1) and bool((camera.alpha == 1).all())
    assert camera.alpha.untyped_storage().nbytes() == 4  # one float, broadcast
    assert camera.foreground.untyped_storage().nbytes() == H * W * 3 * 4  # no RGBA tensor kept alive
    assert camera.alpha.reshape(-1, 1)[torch.tensor([0, H * W - 1])].tolist() == [[1.0], [1.0]]


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


@requires_lego
def test_blender_scene_medium_sites_around_the_object_box_and_cameras():
    """Blender scenes have no COLMAP points: the sites are placed with the same rule around the object box
    [-1.3, 1.3]^3 (the initial point cloud) and the cameras; spatial_colmap_fraction puts that share in the box."""
    from types import SimpleNamespace
    from foamsurf_data.blender import BlenderDataset
    dataset = BlenderDataset(LEGO / "lego_both", resolution=16, device="cpu", splits=("train",))
    config = SimpleNamespace(model=SimpleNamespace(medium_bbox_min=None, medium_bbox_max=None, medium_points=0))
    inside = {}
    for fraction in (0.0, 0.5):
        np.random.seed(0)
        medium = SimpleNamespace(sites=torch.zeros(1, 3))
        profile = WarmupProfile.from_dict(dict(site_initialization="spatial", outer_points=0, spatial_count=5000,
                                               spatial_extent=2.0, spatial_colmap_fraction=fraction))
        initialize_spatial_medium(medium, dataset, config, profile)
        assert medium.sites.shape == (5000, 3) and config.model.medium_points == 5000
        inside[fraction] = int((medium.sites.abs() <= 1.3).all(1).sum())
    assert inside[0.0] < 50  # the cameras (radius 5) widen the box: the object region gets almost no cells
    assert inside[0.5] >= 2400  # half of them in the object box (a few jittered just past its faces)


@requires_lego
def test_blender_sfm_poses_are_the_cameras_of_the_loader(tmp_path):
    """scripts/blender_sfm.py: the COLMAP poses of the prepared photos give the loader's camera centres and axes,
    the photos are the training targets on black, and sparse_known carries the database's image ids."""
    import json
    import sqlite3
    sys.path.insert(0, str(ROOT / "scripts"))
    import blender_sfm
    from foamsurf_data.blender import BlenderDataset
    dataset = BlenderDataset(LEGO / "lego_both", resolution=1, device="cpu", splits=("train",))
    out = tmp_path / "sfm"
    blender_sfm.prepare(LEGO / "lego_both", out)
    with pytest.raises(FileExistsError):
        blender_sfm.prepare(LEGO / "lego_both", out)
    poses = json.loads((out / "poses.json").read_text())
    for camera in dataset.train[::40]:
        pose = poses[camera.name + ".png"]
        w, x, y, z = pose["qvec"]
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        np.testing.assert_allclose(-R.T @ np.array(pose["tvec"]), camera.origin.numpy(), atol=1e-5)
        for row, axis in zip(R, (camera.right, camera.down, camera.forward)):
            np.testing.assert_allclose(row, axis.numpy(), atol=1e-5)
        photo = np.asarray(Image.open(out / "images" / (camera.name + ".png")), dtype=np.float32) / 255.0
        np.testing.assert_allclose(photo, camera.ground_truth((0.0, 0.0, 0.0)).numpy(), atol=0.5 / 255 + 1e-6)
    focal = json.loads((out / "camera.json").read_text())["fx"]
    assert focal == pytest.approx(0.5 * dataset.train[0].width / math.tan(0.5 * dataset.train[0].fov_x), rel=1e-6)
    # the database assigns its own ids; sparse_known must use them
    names = sorted(poses)
    with sqlite3.connect(out / "database.db") as db:
        db.execute("CREATE TABLE cameras (camera_id INTEGER)")
        db.execute("CREATE TABLE images (image_id INTEGER, name TEXT, camera_id INTEGER)")
        db.execute("INSERT INTO cameras VALUES (1)")
        db.executemany("INSERT INTO images VALUES (?, ?, 1)", [(100 + i, n) for i, n in enumerate(reversed(names))])
    blender_sfm.model(out)
    lines = [l for l in (out / "sparse_known/images.txt").read_text().splitlines() if l]
    assert len(lines) == len(names)
    first = lines[0].split()
    assert first[0] == "100" and first[-1] == names[-1] and first[-2] == "1"
    np.testing.assert_allclose([float(v) for v in first[1:5]], poses[names[-1]]["qvec"])


def test_profile_validates_view_settings():
    WarmupProfile.from_dict(dict(site_initialization="spatial", outer_points=0, spatial_view_fraction=0.75))
    for bad in (dict(spatial_view_fraction=1.5), dict(spatial_view_depth=0.0)):
        with pytest.raises(ValueError, match="view"):
            WarmupProfile.from_dict(dict(site_initialization="spatial", outer_points=0, **bad))
