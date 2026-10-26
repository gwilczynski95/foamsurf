"""training/water_fit.py: water from COLMAP point tracks; medium initialisation; the 'water' set."""
import json
import math
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from foamsurf_model.medium import VoronoiMediumModel
from train import fit_iui3, resolve_iui3
from training import water_fit
from training.config import ExperimentConfig, ModelConfig

ROOT = Path(__file__).resolve().parents[1]


def synthetic(beta_d, beta_b, b_inf, points=400, seed=0):
    rng = np.random.default_rng(seed)
    index, colours, distances = [], [], []
    for k in range(points):
        j = rng.uniform(0.2, 0.9, 3)
        for d in rng.uniform(3.0, 20.0, 4):
            colours.append(np.asarray(b_inf) * (1 - np.exp(-np.asarray(beta_b) * d)) + j * np.exp(-np.asarray(beta_d) * d))
            index.append(k)
            distances.append(d)
    return np.array(index), np.array(colours), np.array(distances)


def test_recovers_known_water_per_channel():
    grid = water_fit.GRID
    beta_d, beta_b, b_inf = [grid[15], grid[12], grid[9]], [grid[10], grid[13], grid[14]], [0.2, 0.35, 0.45]
    data = synthetic(beta_d, beta_b, b_inf)
    fit = water_fit.fit_water(*data)
    for c in range(3):
        assert math.isclose(fit["beta_d"][c], beta_d[c], rel_tol=1e-6)
        assert math.isclose(fit["beta_b"][c], beta_b[c], rel_tol=1e-6)
        assert abs(fit["b_inf"][c] - b_inf[c]) < 1e-3
    fixed = water_fit.fit_water(*data, b_inf=b_inf)  # B_inf known: the same betas, B_inf kept
    assert fixed["beta_d"] == fit["beta_d"] and fixed["beta_b"] == fit["beta_b"]
    assert fixed["b_inf"] == pytest.approx(b_inf)
    brighter = water_fit.fit_water(*data, b_inf=[0.4, 0.7, 0.9])  # brighter water: weaker haze for it
    assert all(brighter["beta_b"][c] < beta_b[c] for c in range(3))


def write_model(folder, centres, points, keypoints, width=40):
    """Minimal binary COLMAP model: one PINHOLE camera, identity rotations."""
    folder.mkdir(parents=True)
    with open(folder / "cameras.bin", "wb") as f:
        f.write(struct.pack("<Q", 1) + struct.pack("<iiQQ", 1, 1, width, width) + struct.pack("<dddd", 20, 20, 20, 20))
    with open(folder / "images.bin", "wb") as f:
        f.write(struct.pack("<Q", len(centres)))
        for i, (centre, kps) in enumerate(zip(centres, keypoints), start=1):
            t = -np.asarray(centre, float)  # R = I, centre = -t
            f.write(struct.pack("<idddddddi", i, 1, 0, 0, 0, *t, 1) + f"v{i}.png".encode() + b"\x00")
            f.write(struct.pack("<Q", len(kps)) + b"".join(struct.pack("<ddq", x, y, p) for x, y, p in kps))
    with open(folder / "points3D.bin", "wb") as f:
        f.write(struct.pack("<Q", len(points)))
        for pid, xyz in points.items():
            f.write(struct.pack("<QdddBBBd", pid, *xyz, 0, 0, 0, 0.0) + struct.pack("<Q", 0))


def test_tracks_and_observations(tmp_path):
    points = {7: (0.0, 0.0, 10.0), 8: (1.0, 0.0, 10.0)}
    centres = [(0, 0, 0), (0, 0, 4), (0, 0, -6)]
    keypoints = [[(20, 20, 7), (30, 20, 8)], [(20, 20, 7), (35, 20, 8)], [(20, 20, 7), (-1, 20, -1)]]
    write_model(tmp_path / "sparse/0", centres, points, keypoints)
    images, pts, widths = water_fit.read_tracks(tmp_path / "sparse/0")
    assert len(images) == 3 and set(pts) == {7, 8} and widths == {1: 40}
    photos = {f"v{i}": np.full((20, 20, 3), 0.1 * i) for i in (1, 2, 3)}  # half resolution: keypoints scaled
    index, colours, distances = water_fit.observations(images, pts, widths, photos, min_views=3, min_spread=1.3)
    assert index.tolist() == [0, 0, 0]  # only point 7 has three views (point 8: two, one out of frame)
    assert sorted(np.round(distances, 3).tolist()) == [6.0, 10.0, 16.0]
    assert sorted(np.round(colours[:, 0], 3).tolist()) == [0.1, 0.2, 0.3]


def test_uniform_medium_coefficients():
    medium = VoronoiMediumModel(torch.zeros(3), torch.ones(3), ModelConfig(medium_type="rgb_backscatter", medium_points=8), "cpu")
    medium.set_uniform_coefficients_(torch.tensor([0.17, 0.13, 0.1]), torch.tensor([0.06, 0.1, 0.13]))
    coefficients = medium.optical_coefficients
    torch.testing.assert_close(coefficients, torch.tensor([0.17, 0.13, 0.1, 0.06, 0.1, 0.13]).expand(8, 6), rtol=1e-5, atol=1e-6)


def overrides(variant):
    script = (ROOT / "scripts/train_seathru.sbatch").read_text()
    start = script.index("variant_overrides() {")
    function = script[start:script.index("\n}\n", start) + 3]
    return subprocess.run(["bash", "-c", function + f"\nvariant_overrides {variant}"],
                          capture_output=True, text=True, check=True).stdout.splitlines()


def test_suffixes_and_validation(tmp_path):
    assert overrides("_wfit") == ["joint.model.medium_initial_water_fit=coefficients"]
    assert overrides("_wfitc") == ["joint.model.medium_initial_water_fit=all"]
    assert overrides("_cmax5") == ["joint.gaussian_adaptation.gradient_compensation_max=5.0"]
    assert overrides("_gt4") == ["joint.gaussian_adaptation.gradient_threshold=4.0e-4"]
    m = resolve_iui3(ROOT / "configs/iui3_default.yaml", tmp_path / "r", overrides=overrides("_clamp05_minit_wfit_cmax5"))
    assert m["joint"]["model"]["medium_initial_water_fit"] == "coefficients"
    assert m["joint"]["gaussian_adaptation"]["gradient_compensation_max"] == 5.0
    config = ExperimentConfig()
    config.dataset.source_path = str(tmp_path)
    config.model.medium_initial_water_fit = "yes"
    with pytest.raises(ValueError, match="medium_initial_water_fit"):
        config.validate()
    env = {"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin", "FS_VARIANT_SET": "water",
           "SLURM_ARRAY_JOB_ID": "test", "FS_DRY_RUN": "1",
           "FS_DATASETS": os.environ.get("FS_DATASETS", "/shared/sets/datasets/SeathruNeRF_dataset")}
    assert "out of range" in subprocess.run(["bash", "scripts/train_seathru.sbatch"], cwd=ROOT, capture_output=True,
                                            text=True, env={**env, "SLURM_ARRAY_TASK_ID": "32"}).stderr  # 4 variants x 4 scenes x 2 seeds


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_water_fit_short_real_training(tmp_path):
    from test_medium_fix import _short_manifest
    m = _short_manifest(tmp_path, "water", overrides("_minit_wfitc") + [
        "warmup.background=[0.0, 0.0, 0.0]", "joint.background=[0.0, 0.0, 0.0]"])
    fit_iui3(m)
    fit = json.loads((tmp_path / "water/water_fit.json").read_text())
    initial = torch.load(tmp_path / "water/joint/checkpoints/initialization.pt", map_location="cpu", weights_only=False)
    coefficients = torch.nn.functional.softplus(initial["model"]["medium.raw_density"])
    torch.testing.assert_close(coefficients[0], torch.tensor(fit["beta_d"] + fit["beta_b"]), rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(torch.sigmoid(initial["model"]["medium.raw_color"][0]), torch.tensor(fit["b_inf"]),
                               rtol=1e-4, atol=1e-4)
