"""Direction-dependent water light (optimization.water_direction, the _wdir suffix): the backscatter of each ray is
scaled by exp(W d), nine parameters starting at 0 (= the plain model)."""
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from train import fit_iui3, resolve_iui3
from training.config import ExperimentConfig
from training.water_direction import WaterDirectionGain
from test_oceansplat import overrides

ROOT = Path(__file__).resolve().parents[1]


def test_starts_as_the_plain_model_and_scales_only_the_backscatter():
    gain = WaterDirectionGain("cpu")
    image, backscatter = torch.rand(5, 3), torch.rand(5, 3) * 0.2
    directions = torch.randn(5, 3)
    torch.testing.assert_close(gain.apply(image, backscatter, directions), image)
    with torch.no_grad():
        gain.weights.copy_(torch.tensor([[0.0, 0.0, 0.5]] * 3))  # brighter toward +z
    up, down = torch.tensor([[0.0, 0.0, 2.0]]), torch.tensor([[0.0, 0.0, -1.0]])
    torch.testing.assert_close(gain.gain(up), torch.full((1, 3), float(torch.exp(torch.tensor(0.5)))))  # unit length
    out_up = gain.apply(image[:1], backscatter[:1], up)
    out_down = gain.apply(image[:1], backscatter[:1], down)
    assert bool((out_up > image[:1]).all()) and bool((out_down < image[:1]).all())
    direct = image[:1] - backscatter[:1]
    torch.testing.assert_close(out_up - direct, backscatter[:1] * torch.exp(torch.tensor(0.5)))  # direct untouched


def test_learns_a_brightening_toward_the_surface_and_round_trips():
    torch.manual_seed(0)
    gain = WaterDirectionGain("cpu", lr=0.05)
    directions = torch.nn.functional.normalize(torch.randn(512, 3), dim=-1)
    backscatter = torch.full((512, 3), 0.3)
    target = 0.1 + backscatter * torch.exp(0.4 * directions[:, 2:3])  # water brighter for rays looking up (+z)
    for _ in range(300):
        gain.zero_grad()
        loss = ((gain.apply(0.1 + backscatter, backscatter, directions) - target) ** 2).mean()
        loss.backward()
        gain.step()
    torch.testing.assert_close(gain.weights.detach(), torch.tensor([[0.0, 0.0, 0.4]] * 3), atol=0.02, rtol=0)
    copy = WaterDirectionGain("cpu")
    copy.load_state_dict(gain.state_dict())
    torch.testing.assert_close(copy.weights, gain.weights)


def test_suffix_config_and_set(tmp_path):
    assert overrides("_wdir") == ["joint.optimization.water_direction=true"]
    assert ExperimentConfig().optimization.water_direction is False
    m = resolve_iui3(ROOT / "configs/seathru_default.yaml", tmp_path / "r", overrides=overrides("_wdir"))
    assert m["joint"]["optimization"]["water_direction"] is True
    env = {"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin", "SLURM_ARRAY_JOB_ID": "test", "FS_DRY_RUN": "1",
           "FS_VARIANT_SET": "wsapp_wdir",
           "FS_DATASETS": os.environ.get("FS_DATASETS", "/shared/sets/datasets/SeathruNeRF_dataset")}
    run = lambda task: subprocess.run(["bash", "scripts/train_seathru.sbatch"], cwd=ROOT, capture_output=True,
                                      text=True, env={**env, "SLURM_ARRAY_TASK_ID": str(task)})
    first, last = run(0), run(15)
    assert first.returncode == 0 and "IUI3-RedSea-Undistorted_wsapp_dth10" in first.stdout
    assert "gaussian_appearance_rest_lr_scale=0.05" in first.stdout and "gradient_threshold=10.0e-5" in first.stdout
    assert last.returncode == 0 and "Panama-Undisorted_wdir_seed43" in last.stdout and "out of range" in run(16).stderr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_short_real_training_with_water_direction(tmp_path):
    from test_medium_fix import _short_manifest
    m = _short_manifest(tmp_path, "wdir", overrides("_wdir"))
    fit_iui3(m)
    checkpoint = torch.load(tmp_path / "wdir/joint/checkpoints/last.pt", map_location="cpu", weights_only=False)
    weights = checkpoint["water_direction"]["weights"]
    assert weights.shape == (3, 3) and bool(torch.isfinite(weights).all())
    assert float(weights.abs().sum()) > 0  # it was trained
