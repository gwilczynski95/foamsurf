"""The camw_speed set on the new default: homogeneous water at the cameras (medium_fix.camera_water_*),
OceanSplat losses every k-th step (ocean_every) and a shorter training with every schedule scaled
(schedule.iterations, the _it<N>k suffix), plus the sbatch suffixes and set."""
import csv
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from train import fit_iui3, resolve_iui3
from training.config import ExperimentConfig
from training.end_to_end import update_config
from training.losses import MediumFixConfig, camera_water_loss
from test_oceansplat import overrides

ROOT = Path(__file__).resolve().parents[1]
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")


def test_suffixes():
    assert overrides("_it20k") == ["schedule.iterations=20000"]
    assert overrides("_cut10k") == ["joint.optimization.iterations=10000",
                                    "joint.gaussian_adaptation.opacity_prune_until=10000"]
    assert overrides("_oe2") == ["joint.optimization.ocean_every=2"]
    assert overrides("_camw") == ["medium_fix.camera_water_weight=0.01"]
    assert overrides("_oe2_it20k") == ["joint.optimization.ocean_every=2", "schedule.iterations=20000"]


def test_schedule_scaled_after_every_override(tmp_path):
    # The default trains 20k iterations; _it15k on it gives 15k (both scaled from the same base schedules).
    full = resolve_iui3(ROOT / "configs/seathru_default.yaml", tmp_path / "a")
    short = resolve_iui3(ROOT / "configs/seathru_default.yaml", tmp_path / "a", overrides=overrides("_it15k"))
    o, s = full["joint"]["optimization"], short["joint"]["optimization"]
    assert (o["iterations"], s["iterations"]) == (20000, 15000)
    assert (o["ocean_from"], o["ocean_until"], o["ocean_alpha_until"]) == (2667, 16000, 8000)
    assert (s["ocean_from"], s["ocean_until"], s["ocean_alpha_until"]) == (2000, 12000, 6000)
    assert (o["training_downscale_interval"], s["training_downscale_interval"]) == (4000, 3000)
    assert s["medium_lr_decay_until"] == 15000 and s["gaussian_sh_degree_interval"] == 500
    a, b = full["joint"]["gaussian_adaptation"], short["joint"]["gaussian_adaptation"]
    assert (a["densify_until"], b["densify_until"]) == (10000, 7500) and b["opacity_prune_until"] == 15000
    assert b["densification_interval"] >= 1
    assert (full["medium_fix"]["water_clear_start"], short["medium_fix"]["water_clear_start"]) == (2667, 2000)
    # cadence and non-schedule values stay
    assert short["joint"]["validation"] == full["joint"]["validation"]
    assert s["ocean_tri_weight"] == o["ocean_tri_weight"] and s["camera_isp"] is True
    # the order of the suffixes does not matter: the scaling runs after all overrides
    late = resolve_iui3(ROOT / "configs/iui3_default.yaml", tmp_path / "b",
                        overrides=["schedule.iterations=20000"] + overrides("_oceane"))
    assert late["joint"]["optimization"]["ocean_from"] == 2667
    checked = ExperimentConfig(); update_config(checked, short["joint"]); checked.validate()
    with pytest.raises(ValueError, match="schedule.iterations"):
        resolve_iui3(ROOT / "configs/iui3_default.yaml", tmp_path / "c", overrides=["schedule.iterations=0"])


def test_camera_water_loss_pulls_only_the_cells_at_the_cameras():
    coefficients = torch.full((6, 6), 0.05)
    coefficients[0] = 0.4  # dense cell at a camera
    coefficients[1] = 0.005  # clear cell at a camera
    coefficients = coefficients.requires_grad_(True)
    colour = torch.zeros(6, 3, requires_grad=True)
    near = torch.tensor([True, True, False, False, False, False])
    reference = torch.tensor([False, False, True, True, True, False])
    loss = camera_water_loss(coefficients, colour, near, reference)
    loss.backward()
    expected = ((torch.log(torch.tensor(0.4 / 0.05))) ** 2 + (torch.log(torch.tensor(0.005 / 0.05))) ** 2) / 2
    torch.testing.assert_close(loss.detach(), expected)
    assert coefficients.grad[0].min() > 0 and coefficients.grad[1].max() < 0  # towards the water, both ways
    assert coefficients.grad[2:].abs().sum() == 0  # the reference is detached, other cells untouched
    assert camera_water_loss(coefficients, colour, torch.zeros(6, dtype=torch.bool), reference) is None
    assert MediumFixConfig().camera_water_weight == 0.0  # off by default
    with pytest.raises(ValueError):
        MediumFixConfig(camera_water_radius=0.0)
    assert MediumFixConfig(camera_water_weight=1.0).camera_water_weight_at(500) == 0.0
    assert MediumFixConfig(camera_water_weight=1.0).camera_water_weight_at(2000) == 1.0


def test_camera_water_cells_near_cameras_in_free_water():
    from training.colmap_joint_trainer import MediumFixTrainer
    cameras = [SimpleNamespace(origin=torch.tensor([float(x), 0.0, 0.0])) for x in (0, 1, 2)]  # spacing 1
    sites = torch.tensor([[-0.5, 0.8, 0.0],   # near a camera, free water (1.7 from the Gaussian)
                          [1.0, 0.2, 0.0],    # near a camera but next to a Gaussian
                          [1.0, 10.0, 0.0],   # far free water: reference
                          [1.0, 2.5, 0.0]])   # between: neither
    gaussians = torch.tensor([[1.0, 0.0, 0.0], [5.0, -50.0, 0.0]])
    fake = SimpleNamespace(medium_fix=MediumFixConfig(camera_water_weight=0.01), _camera_water_cells=None,
                           dataset=SimpleNamespace(train=cameras))
    model = SimpleNamespace(medium=SimpleNamespace(sites=sites), gaussians=SimpleNamespace(means=gaussians))
    near, reference = MediumFixTrainer._camera_water_masks(fake, 0, model)
    assert near.tolist() == [True, False, False, False] and reference.tolist() == [False, False, True, False]
    model.medium.sites = sites * 100  # cached until the refresh interval
    assert MediumFixTrainer._camera_water_masks(fake, 100, model)[0].tolist() == [True, False, False, False]
    assert MediumFixTrainer._camera_water_masks(fake, 250, model)[0].tolist() == [False] * 4


def test_camw_speed_set_on_the_new_default():
    env = {"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin", "SLURM_ARRAY_JOB_ID": "test", "FS_DRY_RUN": "1",
           "FS_VARIANT_SET": "camw_speed",
           "FS_DATASETS": os.environ.get("FS_DATASETS", "/shared/sets/datasets/SeathruNeRF_dataset")}
    run = lambda task: subprocess.run(["bash", "scripts/train_seathru.sbatch"], cwd=ROOT, capture_output=True,
                                      text=True, env={**env, "SLURM_ARRAY_TASK_ID": str(task)})
    first, last = run(0), run(23)
    assert first.returncode == 0 and "config=configs/seathru_default.yaml" in first.stdout and "_camw" in first.stdout
    assert last.returncode == 0 and "Panama-Undisorted_g400k_seed43" in last.stdout
    assert "schedule.iterations=20000" in run(2).stdout and "out of range" in run(24).stderr


@gpu
def test_short_real_training_with_camera_water_and_ocean_every(tmp_path):
    from test_medium_fix import _short_manifest
    m = _short_manifest(tmp_path, "cs", overrides("_oceane_camw") + [
        "joint.optimization.ocean_from=1", "joint.optimization.ocean_until=5", "joint.optimization.ocean_downscale=1",
        "joint.optimization.ocean_every=2", "medium_fix.camera_water_start=0", "medium_fix.camera_water_ramp=0",
        "medium_fix.camera_water_radius=1000.0", "medium_fix.camera_water_free=0.0",
        "warmup.background=[0.0, 0.0, 0.0]", "joint.background=[0.0, 0.0, 0.0]"])
    fit_iui3(m)
    rows = {int(r["iteration"]): r for r in csv.DictReader(open(tmp_path / "cs/joint/metrics/train.csv"))}
    ocean = {i: rows[i]["ocean/obj"] for i in rows if 1 <= i < 5}
    # OceanSplat terms on steps 1 and 3 only (every 2nd step of the window from 1)
    assert ocean[1] != "nan" and ocean[3] != "nan" and ocean[2] == "nan" and ocean[4] == "nan"
    assert all(r["medium_fix/camera_water_weight"] == "0.01" for r in rows.values() if int(r["iteration"]) >= 1)
    cells = [float(r["medium_fix/camera_water_cells"]) for r in rows.values() if r["medium_fix/camera_water_cells"] != "nan"]
    assert cells and max(cells) > 0


def test_combo_set():
    env = {"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin", "SLURM_ARRAY_JOB_ID": "test", "FS_DRY_RUN": "1",
           "FS_VARIANT_SET": "combo",
           "FS_DATASETS": os.environ.get("FS_DATASETS", "/shared/sets/datasets/SeathruNeRF_dataset")}
    run = lambda task: subprocess.run(["bash", "scripts/train_seathru.sbatch"], cwd=ROOT, capture_output=True,
                                      text=True, env={**env, "SLURM_ARRAY_TASK_ID": str(task)})
    first, last = run(0), run(15)
    assert first.returncode == 0 and "config=configs/seathru_default.yaml" in first.stdout
    assert all(line in first.stdout for line in ("medium_fix.camera_water_weight=0.01",
                                                 "joint.optimization.ocean_every=2", "schedule.iterations=20000"))
    assert last.returncode == 0 and "Panama-Undisorted_camw_oe2_it20k_mip_seed43" in last.stdout
    assert "joint.model.gaussian_mip_filter=0.2" in last.stdout and "out of range" in run(16).stderr


def test_speed2_and_foggy_sets_on_the_new_default():
    env = {"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin", "SLURM_ARRAY_JOB_ID": "test", "FS_DRY_RUN": "1",
           "FS_DATASETS": os.environ.get("FS_DATASETS", "/shared/sets/datasets/SeathruNeRF_dataset")}
    run = lambda task, name: subprocess.run(["bash", "scripts/train_seathru.sbatch"], cwd=ROOT, capture_output=True,
                                            text=True, env={**env, "FS_VARIANT_SET": name, "SLURM_ARRAY_TASK_ID": str(task)})
    oe4, it15 = run(0, "speed2"), run(2, "speed2")
    assert oe4.returncode == 0 and "joint.optimization.ocean_every=4" in oe4.stdout
    assert it15.returncode == 0 and "schedule.iterations=15000" in it15.stdout and "out of range" in run(16, "speed2").stderr
    m = resolve_iui3(ROOT / "configs/seathru_default.yaml", Path("/tmp/unused"), overrides=overrides("_it15k_oe4"))
    o = m["joint"]["optimization"]
    assert o["iterations"] == 15000 and o["ocean_every"] == 4 and o["ocean_from"] == 2000  # 4000 x 15/30
    foggy = run(3, "foggy")
    assert foggy.returncode == 0 and "garden_hard_seed43" in foggy.stdout and "dry run ok" in foggy.stdout
    assert "config=configs/seathru_default.yaml" in foggy.stdout and "out of range" in run(4, "foggy").stderr


def test_capacity_set():
    env = {"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin", "SLURM_ARRAY_JOB_ID": "test", "FS_DRY_RUN": "1",
           "FS_VARIANT_SET": "capacity",
           "FS_DATASETS": os.environ.get("FS_DATASETS", "/shared/sets/datasets/SeathruNeRF_dataset")}
    run = lambda task: subprocess.run(["bash", "scripts/train_seathru.sbatch"], cwd=ROOT, capture_output=True,
                                      text=True, env={**env, "SLURM_ARRAY_TASK_ID": str(task)})
    assert "joint.gaussian_adaptation.max_points=1000000" in run(0).stdout
    last = run(15)
    assert last.returncode == 0 and "joint.gaussian_adaptation.max_points=1200000" in last.stdout
    assert "config=configs/seathru_default.yaml" in last.stdout and "out of range" in run(16).stderr


def test_shdeg_set(tmp_path):
    env = {"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin", "SLURM_ARRAY_JOB_ID": "test", "FS_DRY_RUN": "1",
           "FS_VARIANT_SET": "shdeg",
           "FS_DATASETS": os.environ.get("FS_DATASETS", "/shared/sets/datasets/SeathruNeRF_dataset")}
    run = lambda task: subprocess.run(["bash", "scripts/train_seathru.sbatch"], cwd=ROOT, capture_output=True,
                                      text=True, env={**env, "SLURM_ARRAY_TASK_ID": str(task)})
    first, last = run(0), run(15)
    assert first.returncode == 0 and "joint.model.sh_degree=1" in first.stdout
    assert last.returncode == 0 and "joint.model.sh_degree=2" in last.stdout and "out of range" in run(16).stderr
    m = resolve_iui3(ROOT / "configs/seathru_default.yaml", tmp_path / "s", overrides=overrides("_sh1"))
    assert m["joint"]["model"]["sh_degree"] == 1 and m["joint"]["optimization"]["gaussian_sh_degree_interval"] == 667
