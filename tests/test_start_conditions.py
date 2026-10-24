"""A calmer start: scale cap relative to the scene from iteration 0, medium colour from the photos."""
import csv
import math
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from foamsurf_model.gaussians import SurfaceGaussianModel
from foamsurf_model.medium import VoronoiMediumModel
from train import fit_iui3, resolve_iui3
from training.config import ExperimentConfig, ModelConfig
from training.trainer import gaussian_scale_cap

ROOT = Path(__file__).resolve().parents[1]


def test_scale_cap_follows_the_scene_extent():
    options = SimpleNamespace(gaussian_max_scale_scene_fraction=0.05, gaussian_min_scale=1e-4)
    assert gaussian_scale_cap(options, 6.8) == (1e-4, pytest.approx(0.34))
    options.gaussian_min_scale = None
    assert gaussian_scale_cap(options, 6.8)[0] > 0
    options.gaussian_max_scale_scene_fraction = None
    assert gaussian_scale_cap(options, 6.8) is None


def test_clamp_caps_initial_outlier_scales():
    points = np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01], [40, 0, 0]], dtype=np.float32)
    model = SurfaceGaussianModel(points, np.full_like(points, 0.5),
                                 ModelConfig(appearance="rgb", gaussian_initial_points=5), "cpu")
    assert model.scales.max() > 5  # the far outlier starts huge
    model.clamp_scales_(1e-7, 0.34)
    assert float(model.scales.max()) == pytest.approx(0.34, rel=1e-5)


def test_scene_fraction_validation(tmp_path):
    config = ExperimentConfig()
    config.dataset.source_path = str(tmp_path)
    config.optimization.gaussian_max_scale_scene_fraction = 0.0
    with pytest.raises(ValueError, match="gaussian_max_scale_scene_fraction"):
        config.validate()


@pytest.mark.parametrize("activation", ["sigmoid", "dc"])
def test_uniform_medium_colour(activation):
    config = ModelConfig(medium_type="rgb_backscatter", medium_points=16)
    medium = VoronoiMediumModel(torch.zeros(3), torch.ones(3), config, "cpu")
    medium.color_activation = activation
    target = torch.tensor([0.12, 0.35, 0.47])
    medium.set_uniform_color_(target)
    torch.testing.assert_close(medium.colors, target.expand(16, 3), atol=1e-5, rtol=0)


def overrides(variant):
    script = (ROOT / "scripts/train_seathru.sbatch").read_text()
    start = script.index("variant_overrides() {")
    function = script[start:script.index("\n}\n", start) + 3]
    return subprocess.run(["bash", "-c", function + f"\nvariant_overrides {variant}"],
                          capture_output=True, text=True, check=True).stdout.splitlines()


def test_start_set(tmp_path):
    m = resolve_iui3(ROOT / "configs/iui3_default.yaml", tmp_path / "r", overrides=overrides("_clamp05_minit_scl005"))
    joint = m["joint"]
    assert joint["optimization"]["gaussian_max_scale_scene_fraction"] == 0.05
    assert joint["model"]["medium_initial_color_from_data"] is True
    assert joint["optimization"]["gaussian_scales_lr"] == joint["optimization"]["gaussian_scales_lr_final"] == 0.005
    assert overrides("_clamp10") == ["joint.optimization.gaussian_max_scale_scene_fraction=0.10"]
    run = subprocess.run(["bash", "scripts/train_seathru.sbatch"], cwd=ROOT, capture_output=True, text=True,
                         env={"PATH": "/usr/bin:/bin", "FS_VARIANT_SET": "start", "SLURM_ARRAY_TASK_ID": "64",
                              "SLURM_ARRAY_JOB_ID": "test"})
    assert "out of range" in run.stderr  # 8 variants x 4 scenes x 2 seeds = tasks 0-63


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_start_short_real_training(tmp_path):
    from test_medium_fix import _short_manifest
    m = _short_manifest(tmp_path, "start", overrides("_clamp05_minit_scl005") + [
        "warmup.background=[0.0, 0.0, 0.0]", "joint.background=[0.0, 0.0, 0.0]"])
    fit_iui3(m)
    root = tmp_path / "start/joint"
    rows = list(csv.DictReader((root / "metrics/train.csv").open()))
    assert len(rows) == 4 and all(math.isfinite(float(r["train/loss"])) for r in rows)
    initial = torch.load(root / "checkpoints/initialization.pt", map_location="cpu", weights_only=False)["model"]
    extent = float(json_extent(root))
    assert initial["gaussians.log_scales"].exp().max() <= 0.05 * extent * (1 + 1e-5)
    colors = torch.sigmoid(initial["medium.raw_color"])
    assert torch.allclose(colors, colors[0].expand_as(colors), atol=1e-5)  # one colour, from the photos
    assert not torch.allclose(colors[0], torch.full((3,), 0.5), atol=1e-2)


def json_extent(root):
    from foamsurf_data import load_dataset
    from foamsurf_data.camera_geometry import camera_normalization
    config = ExperimentConfig.load(root / "config.yaml")
    dataset = load_dataset(config.dataset.source_path, resolution=config.dataset.resolution, device="cpu",
                           llffhold=config.dataset.llffhold)
    return camera_normalization(dataset.train)[1]
