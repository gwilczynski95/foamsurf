"""Coarse-to-fine training resolution and the WaterSplatting-like opacity schedule."""
import csv
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from foamsurf_model.gaussian_adaptation import scheduled_opacity_threshold
from train import fit_iui3, resolve_iui3
from training.config import ExperimentConfig, GaussianAdaptationConfig
from training.trainer import training_downscale

ROOT = Path(__file__).resolve().parents[1]
IUI3 = Path("/shared/sets/datasets/SeathruNeRF_dataset/IUI3-RedSea-Undistorted")


def test_downscale_schedule_matches_watersplatting():
    off = SimpleNamespace(training_downscale_levels=0, training_downscale_interval=3000)
    ws = SimpleNamespace(training_downscale_levels=2, training_downscale_interval=3000)
    assert training_downscale(off, 0) == 1
    assert [training_downscale(ws, i) for i in (0, 2999, 3000, 5999, 6000, 30000)] == [4, 4, 2, 2, 1, 1]


@pytest.mark.skipif(not IUI3.is_dir(), reason="dataset not on this machine")
def test_downscaled_batch_is_the_whole_image_at_lower_resolution():
    import torch.nn.functional as F
    from foamsurf_data.colmap import ColmapDataset
    from training.sampler import RandomCameraRaySampler
    dataset = ColmapDataset(str(IUI3), resolution=8, device="cpu", llffhold=8)
    sampler = RandomCameraRaySampler(dataset, 1, (0.0, 0.0, 0.0), full_image=True, cameras_without_replacement=True)
    full = sampler.next()
    camera = full.camera
    sampler._camera_stack = [dataset.train.index(camera)]  # same camera again
    sampler.downscale = 2
    batch = sampler.next()
    height, width = camera.height // 2, camera.width // 2
    assert batch.image_size == (height, width) and batch.rays.directions.shape == (height * width, 3)
    expected = F.interpolate(camera.foreground.permute(2, 0, 1)[None], size=(height, width), mode="bilinear",
                             antialias=True, align_corners=False)[0].permute(1, 2, 0).reshape(-1, 3)
    torch.testing.assert_close(batch.target, expected)
    # Pixel (0, 0) at half resolution covers full-resolution pixels (0..1, 0..1): its ray lies between theirs.
    grid = full.rays.directions.reshape(camera.height, camera.width, 3)
    corner = torch.nn.functional.normalize(grid[:2, :2].reshape(-1, 3).mean(0), dim=0)
    assert torch.allclose(batch.rays.directions[0], corner, atol=1e-3)
    sampler.full_image = False
    with pytest.raises(ValueError, match="full-image"):
        sampler.next()


def test_late_opacity_threshold():
    config = GaussianAdaptationConfig(min_opacity=0.5, min_opacity_initial=0.5, opacity_prune_ramp=0,
                                      min_opacity_late=0.1, min_opacity_late_from=10000)
    assert scheduled_opacity_threshold(config, 9999) == 0.5 and scheduled_opacity_threshold(config, 10000) == 0.1
    assert scheduled_opacity_threshold(GaussianAdaptationConfig(min_opacity=0.1), 20000) == 0.1


def test_config_validation(tmp_path):
    config = ExperimentConfig()
    config.dataset.source_path = str(tmp_path)
    config.optimization.training_downscale_levels = 2
    with pytest.raises(ValueError, match="full_image_batch"):
        config.validate()
    config.optimization.full_image_batch = True
    config.validate()
    config.gaussian_adaptation.min_opacity_late = 1.5
    with pytest.raises(ValueError, match="min_opacity_late"):
        config.validate()


def overrides(variant):
    script = (ROOT / "scripts/train_seathru.sbatch").read_text()
    start = script.index("variant_overrides() {")
    function = script[start:script.index("\n}\n", start) + 3]
    return subprocess.run(["bash", "-c", function + f"\nvariant_overrides {variant}"],
                          capture_output=True, text=True, check=True).stdout.splitlines()


def test_suffixes_resolve(tmp_path):
    m = resolve_iui3(ROOT / "configs/iui3_default.yaml", tmp_path / "r", overrides=overrides("_wsop_c2f"))
    a, o = m["joint"]["gaussian_adaptation"], m["joint"]["optimization"]
    assert (a["opacity_reset_interval"], a["opacity_reset_value"], a["min_opacity"]) == (500, 0.5, 0.5)
    assert (a["min_opacity_late"], a["min_opacity_late_from"], a["densify_until"]) == (0.1, 10000, 10000)
    assert (o["training_downscale_levels"], o["training_downscale_interval"]) == (2, 3000)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_c2f_wsop_short_real_training(tmp_path):
    from test_medium_fix import _short_manifest
    m = _short_manifest(tmp_path, "c2f", [
        "joint.optimization.training_downscale_levels=2", "joint.optimization.training_downscale_interval=2",
        "joint.gaussian_adaptation.opacity_reset_interval=2", "joint.gaussian_adaptation.opacity_reset_value=0.5",
        "joint.gaussian_adaptation.min_opacity_late=0.1", "joint.gaussian_adaptation.min_opacity_late_from=3",
        "warmup.background=[0.0, 0.0, 0.0]", "joint.background=[0.0, 0.0, 0.0]"])
    fit_iui3(m)
    rows = list(csv.DictReader((tmp_path / "c2f/joint/metrics/train.csv").open()))
    assert len(rows) == 4 and all(float(r["train/loss"]) == float(r["train/loss"]) for r in rows)
