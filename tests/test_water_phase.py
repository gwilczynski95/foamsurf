"""Directional water light (optimization.water_phase, the _wphase suffix): the backscatter of each ray is scaled by a
Henyey-Greenstein phase function over the isotropic one, with a learned direction toward the light and anisotropy
g per colour channel; g = 0 at the start (= the plain model)."""
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from train import fit_iui3, resolve_iui3
from training.config import ExperimentConfig
from training.water_phase import WaterPhase
from test_oceansplat import overrides

ROOT = Path(__file__).resolve().parents[1]
UP = torch.tensor([0.0, 0.0, 1.0])


def test_starts_as_the_plain_model_and_scales_only_the_backscatter():
    phase = WaterPhase(UP * 3.0, "cpu")  # any length: normalised
    image, backscatter = torch.rand(6, 3), torch.rand(6, 3) * 0.2
    directions = torch.randn(6, 3)
    torch.testing.assert_close(phase.apply(image, backscatter, directions), image)  # g = 0: isotropic
    torch.testing.assert_close(phase.toward_light.detach(), UP)
    with torch.no_grad():
        phase.raw_anisotropy.fill_(math.atanh(0.5 / 0.95))  # g = 0.5, forward scattering
    looking_up, sideways, down = torch.tensor([[0.0, 0.0, 2.0]]), torch.tensor([[1.0, 0.0, 0.0]]), torch.tensor([[0.0, 0.0, -1.0]])
    torch.testing.assert_close(phase.gain(looking_up), torch.full((1, 3), (1 - 0.25) / (1 - 0.5) ** 3))  # = 6
    torch.testing.assert_close(phase.gain(sideways), torch.full((1, 3), 0.75 / 1.25 ** 1.5))
    assert bool((phase.gain(looking_up) > phase.gain(sideways)).all()) and bool((phase.gain(sideways) > phase.gain(down)).all())
    direct = image[:1] - backscatter[:1]
    torch.testing.assert_close(phase.apply(image[:1], backscatter[:1], looking_up) - direct, backscatter[:1] * 6.0)


def test_phase_integrates_to_the_isotropic_one():
    # the HG phase over the isotropic one averages to 1 over the sphere: brightening toward the light is
    # paid for by darkening away from it (no free gain in total scattered light)
    phase = WaterPhase(UP, "cpu")
    with torch.no_grad():
        phase.raw_anisotropy.copy_(torch.tensor([0.2, 0.6, -0.4]))
    torch.manual_seed(0)
    directions = torch.nn.functional.normalize(torch.randn(400_000, 3), dim=-1)
    torch.testing.assert_close(phase.gain(directions).mean(0), torch.ones(3), atol=0.02, rtol=0)


def test_learns_the_light_direction_and_anisotropy():
    torch.manual_seed(0)
    light = torch.nn.functional.normalize(torch.tensor([0.3, 0.1, 1.0]), dim=0)
    truth = WaterPhase(light, "cpu")
    with torch.no_grad():
        truth.raw_anisotropy.fill_(math.atanh(0.4 / 0.95))
    directions = torch.nn.functional.normalize(torch.randn(2048, 3), dim=-1)
    directions = directions[directions[:, 2] < 0.6]  # like the training photos: never looking far up
    backscatter = torch.full((len(directions), 3), 0.3)
    target = truth.apply(0.1 + backscatter, backscatter, directions).detach()
    phase = WaterPhase(UP, "cpu", lr=0.02)
    for _ in range(800):
        phase.zero_grad()
        ((phase.apply(0.1 + backscatter, backscatter, directions) - target) ** 2).mean().backward()
        phase.step()
    torch.testing.assert_close(phase.anisotropy.detach(), torch.full((3,), 0.4), atol=0.03, rtol=0)
    found = torch.nn.functional.normalize(phase.toward_light.detach(), dim=0)
    assert float(found @ light) > 0.99
    # and it extrapolates: looking straight up (never in the data) gets the true gain
    up = torch.tensor([[0.0, 0.0, 1.0]])
    torch.testing.assert_close(phase.gain(up).detach(), truth.gain(up).detach(), rtol=0.1, atol=0)
    copy = WaterPhase(UP, "cpu")
    copy.load_state_dict(phase.state_dict())
    torch.testing.assert_close(copy.gain(up), phase.gain(up))


def test_fixed_light_learns_only_the_anisotropy():
    torch.manual_seed(0)
    truth = WaterPhase(UP, "cpu")
    with torch.no_grad():
        truth.raw_anisotropy.fill_(math.atanh(0.6 / 0.95))
    directions = torch.nn.functional.normalize(torch.randn(2048, 3), dim=-1)
    backscatter = torch.full((len(directions), 3), 0.3)
    target = truth.apply(0.1 + backscatter, backscatter, directions).detach()
    start = torch.nn.functional.normalize(torch.tensor([0.2, 0.0, 1.0]), dim=0)  # a tilted "up"
    phase = WaterPhase(start, "cpu", lr=0.05, fixed_light=True)
    for _ in range(300):
        phase.zero_grad()
        ((phase.apply(0.1 + backscatter, backscatter, directions) - target) ** 2).mean().backward()
        phase.step()
    torch.testing.assert_close(phase.toward_light, start)  # never moved
    assert not phase.toward_light.requires_grad
    assert float(phase.anisotropy.mean()) > 0.4  # g still found, as well as a tilted light allows
    copy = WaterPhase(start, "cpu", fixed_light=True)
    copy.load_state_dict(phase.state_dict())
    torch.testing.assert_close(copy.raw_anisotropy, phase.raw_anisotropy.detach())


def test_curwater_suffixes_and_set(tmp_path):
    assert overrides("_wphaseup") == ["joint.optimization.water_phase=true", "joint.optimization.water_phase_fixed_light=true"]
    assert overrides("_wphase_wphlr05") == ["joint.optimization.water_phase=true", "joint.optimization.water_phase_lr=0.05"]
    assert ExperimentConfig().optimization.water_phase_fixed_light is False
    m = resolve_iui3(ROOT / "configs/seathru_default.yaml", tmp_path / "r", overrides=overrides("_cull05r_bin01_wphaseup_wphlr05"))
    o = m["joint"]["optimization"]
    assert o["water_phase"] is True and o["water_phase_fixed_light"] is True and o["water_phase_lr"] == 0.05
    m = resolve_iui3(ROOT / "configs/seathru_default.yaml", tmp_path / "c", overrides=overrides("_cull05r_bin01_wphase_cov07late"))
    assert m["joint"]["optimization"]["water_phase"] is True and m["medium_fix"]["surface_coverage_threshold"] == 0.7
    env = {"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin", "SLURM_ARRAY_JOB_ID": "test", "FS_DRY_RUN": "1",
           "FS_VARIANT_SET": "curwater",
           "FS_DATASETS": os.environ.get("FS_DATASETS", "/shared/sets/datasets/SeathruNeRF_dataset")}
    run = lambda task: subprocess.run(["bash", "scripts/train_seathru.sbatch"], cwd=ROOT, capture_output=True,
                                      text=True, env={**env, "SLURM_ARRAY_TASK_ID": str(task)})
    assert run(31).returncode == 0 and "Panama-Undisorted_cull05r_bin01_wphase_cov07late_seed43" in run(31).stdout
    assert "Curasao-Undistorted_cull05r_bin01_wphaseup" in run(8).stdout
    assert "out of range" in run(32).stderr


def test_suffix_config_and_set(tmp_path):
    assert overrides("_wphase") == ["joint.optimization.water_phase=true"]
    assert ExperimentConfig().optimization.water_phase is False
    m = resolve_iui3(ROOT / "configs/seathru_default.yaml", tmp_path / "r", overrides=overrides("_cull05r_wphase"))
    assert m["joint"]["optimization"]["water_phase"] is True and m["joint"]["gaussian_adaptation"]["opacity_reset_interval"] == 500
    env = {"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin", "SLURM_ARRAY_JOB_ID": "test", "FS_DRY_RUN": "1",
           "FS_VARIANT_SET": "phase",
           "FS_DATASETS": os.environ.get("FS_DATASETS", "/shared/sets/datasets/SeathruNeRF_dataset")}
    run = lambda task: subprocess.run(["bash", "scripts/train_seathru.sbatch"], cwd=ROOT, capture_output=True,
                                      text=True, env={**env, "SLURM_ARRAY_TASK_ID": str(task)})
    assert run(7).returncode == 0 and "Panama-Undisorted_cull05r_wphase_seed43" in run(7).stdout
    assert "out of range" in run(8).stderr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_short_real_training_with_the_phase_function_and_resume(tmp_path):
    from test_medium_fix import _short_manifest
    from training.trainer import Trainer
    m = _short_manifest(tmp_path, "phase", overrides("_wphase_wdir"))  # both water light terms together
    fit_iui3(m)
    checkpoint = torch.load(tmp_path / "phase/joint/checkpoints/last.pt", map_location="cpu", weights_only=False)
    state = checkpoint["water_phase"]
    assert bool(torch.isfinite(state["toward_light"]).all()) and float(state["raw_anisotropy"].abs().sum()) > 0
    assert "water_direction" in checkpoint
    root = tmp_path / "phase/joint"
    config = ExperimentConfig.load(root / "config.yaml")
    config.output_dir = str(tmp_path / "resumed")
    trainer = Trainer(config, resume=root / "checkpoints/last.pt")
    torch.testing.assert_close(trainer.water_phase.raw_anisotropy.detach().cpu(), state["raw_anisotropy"])
    summary, _, _ = trainer.evaluator.evaluate(4)
    assert math.isfinite(summary["eval/primary_psnr"])
