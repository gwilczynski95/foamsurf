"""Water attenuation anchored to the COLMAP-track fit (medium_fix.water_anchor_*, the _wanc suffix): the level
of the free water moves to the fitted beta_D, the differences between cells stay."""
import csv
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from train import fit_iui3, resolve_iui3
from training.losses import MediumFixConfig, water_anchor_loss
from test_oceansplat import overrides

ROOT = Path(__file__).resolve().parents[1]
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")


def test_anchor_moves_the_level_not_the_differences():
    coefficients = torch.tensor([[0.1, 0.1, 0.1, 1, 1, 1], [0.4, 0.4, 0.4, 1, 1, 1], [9.0, 9, 9, 1, 1, 1]],
                                requires_grad=True)
    reference = torch.tensor([True, True, False])
    target = torch.tensor([0.4, 0.4, 0.4])
    loss = water_anchor_loss(coefficients, reference, target)
    level = (torch.log(torch.tensor(0.1)) + torch.log(torch.tensor(0.4))) / 2  # mean log = log 0.2
    torch.testing.assert_close(loss.detach(), 3 * (level - torch.log(torch.tensor(0.4))) ** 2)
    loss.backward()
    grad = coefficients.grad
    # d/d log sigma is the same for every reference cell: the level moves, the ratio 0.1 : 0.4 stays
    torch.testing.assert_close(grad[0, :3] * 0.1, grad[1, :3] * 0.4)
    assert bool((grad[:2, :3] < 0).all())  # towards the larger fitted attenuation
    assert grad[2].abs().sum() == 0 and grad[:, 3:].abs().sum() == 0  # other cells, backscatter untouched
    assert water_anchor_loss(coefficients, torch.zeros(3, dtype=torch.bool), target) is None
    assert MediumFixConfig().water_anchor_weight == 0.0
    assert MediumFixConfig(water_anchor_weight=0.1).water_anchor_weight_at(0) == 0.1


def test_suffix_sets_and_schedule(tmp_path):
    assert overrides("_wanc") == ["medium_fix.water_anchor_weight=0.1"]
    assert overrides("_wanc03") == ["medium_fix.water_anchor_weight=0.03"]
    assert overrides("_wanc03_val8")[0] == "medium_fix.water_anchor_weight=0.03"
    assert overrides("_wfit_wanc") == ["joint.model.medium_initial_water_fit=coefficients",
                                       "medium_fix.water_anchor_weight=0.1"]
    m = resolve_iui3(ROOT / "configs/seathru_default.yaml", tmp_path / "r", overrides=overrides("_wfit_wanc"))
    assert m["medium_fix"]["water_anchor_weight"] == 0.1 and m["joint"]["model"]["medium_initial_water_fit"] == "coefficients"
    env = {"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin", "SLURM_ARRAY_JOB_ID": "test", "FS_DRY_RUN": "1",
           "FS_DATASETS": os.environ.get("FS_DATASETS", "/shared/sets/datasets/SeathruNeRF_dataset")}
    run = lambda name, task: subprocess.run(["bash", "scripts/train_seathru.sbatch"], cwd=ROOT, capture_output=True,
                                            text=True, env={**env, "FS_VARIANT_SET": name, "SLURM_ARRAY_TASK_ID": str(task)})
    last = run("wfit", 15)
    assert last.returncode == 0 and "Panama-Undisorted_wfit_wanc_seed43" in last.stdout and "out of range" in run("wfit", 16).stderr
    foggy = run("wfit_foggy", 7)
    assert foggy.returncode == 0 and "garden_hard_wfit_wanc_seed43" in foggy.stdout and "out of range" in run("wfit_foggy", 8).stderr


@gpu
def test_short_real_training_with_the_anchor(tmp_path):
    from test_medium_fix import _short_manifest
    m = _short_manifest(tmp_path, "anchor", overrides("_wfit_wanc") + [
        "medium_fix.camera_water_free=0.0", "warmup.background=[0.0, 0.0, 0.0]", "joint.background=[0.0, 0.0, 0.0]"])
    fit_iui3(m)
    assert (tmp_path / "anchor/water_fit.json").is_file()
    rows = list(csv.DictReader(open(tmp_path / "anchor/joint/metrics/train.csv")))
    raw = [float(r["medium_fix/water_anchor_raw"]) for r in rows if r["medium_fix/water_anchor_raw"] != "nan"]
    assert raw and all(torch.isfinite(torch.tensor(raw)))
    assert all(float(r["medium_fix/water_anchor_attenuation_r"]) > 0 for r in rows
               if r["medium_fix/water_anchor_attenuation_r"] != "nan")


@gpu
def test_anchor_without_the_fit_is_an_error(tmp_path):
    from test_medium_fix import _short_manifest
    m = _short_manifest(tmp_path, "nofit", overrides("_wanc"))
    with pytest.raises(Exception):
        fit_iui3(m)


def test_garden_round_two_sets():
    env = {"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin", "SLURM_ARRAY_JOB_ID": "test", "FS_DRY_RUN": "1",
           "FS_DATASETS": os.environ.get("FS_DATASETS", "/shared/sets/datasets/SeathruNeRF_dataset")}
    run = lambda name, task: subprocess.run(["bash", "scripts/train_seathru.sbatch"], cwd=ROOT, capture_output=True,
                                            text=True, env={**env, "FS_VARIANT_SET": name, "SLURM_ARRAY_TASK_ID": str(task)})
    cap = run("garden_cap", 8)
    assert cap.returncode == 0 and "garden_clear_g3000k_val8" in cap.stdout and "max_points=3000000" in cap.stdout
    assert "out of range" in run("garden_cap", 9).stderr
    anchor = run("garden_anchor", 3)
    assert anchor.returncode == 0 and "garden_hard_wanc_val8" in anchor.stdout and "out of range" in run("garden_anchor", 4).stderr
