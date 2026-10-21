"""WaterSplatting-like recipe: relative loss, 3DGS optimiser and the 'ws' scene set."""
import csv
import math
import subprocess
from pathlib import Path

import pytest
import torch
from torchmetrics.functional.image import structural_similarity_index_measure

from train import fit_iui3, resolve_iui3
from training.losses import b0_reconstruction_loss

ROOT = Path(__file__).resolve().parents[1]


def test_relative_loss_matches_watersplatting():
    torch.manual_seed(0)
    prediction = torch.rand(24, 32, 3, requires_grad=True)
    target = torch.rand(24, 32, 3)
    loss = b0_reconstruction_loss(prediction, target, mode="reg_l1_reg_dssim", dssim_weight=0.2)
    weight = 1.0 / (prediction.detach() + 1e-3)
    l1 = (torch.abs(prediction - target) * weight).mean()
    nchw = lambda image: image.permute(2, 0, 1).unsqueeze(0)  # noqa: E731
    ssim = structural_similarity_index_measure(nchw(prediction * weight), nchw(target * weight), data_range=1.0)
    torch.testing.assert_close(loss.total, 0.8 * l1 + 0.2 * (1 - ssim))
    # Dark pixels weigh more: the same absolute error costs more where the prediction is dark.
    dark, bright = torch.full((8, 8, 3), 0.05), torch.full((8, 8, 3), 0.8)
    assert (b0_reconstruction_loss(dark, dark + 0.02, mode="reg_l1_reg_dssim").total
            > b0_reconstruction_loss(bright, bright + 0.02, mode="reg_l1_reg_dssim").total)
    # The weight is detached: the gradient of the L1 part is sign / (pred + 1e-3).
    loss = b0_reconstruction_loss(prediction, target, mode="reg_l1_reg_dssim", dssim_weight=0.0)
    loss.total.backward()
    expected = torch.sign(prediction.detach() - target) * weight / prediction.numel()
    torch.testing.assert_close(prediction.grad, expected)


def overrides(variant):
    script = (ROOT / "scripts/train_seathru.sbatch").read_text()
    start = script.index("variant_overrides() {")
    function = script[start:script.index("\n}\n", start) + 3]
    return subprocess.run(["bash", "-c", function + f"\nvariant_overrides {variant}"],
                          capture_output=True, text=True, check=True).stdout.splitlines()


def test_ws_set_overrides(tmp_path):
    m = resolve_iui3(ROOT / "configs/iui3_default.yaml", tmp_path / "r",
                     overrides=overrides("_sh3_dres_opt_rel_noreg"))
    o = m["joint"]["optimization"]
    assert o["gaussian_adam_eps"] == 1e-15 and o["gaussian_scale_regularization"] == 0.0
    assert (o["gaussian_scales_lr"], o["gaussian_scales_lr_final"]) == (0.005, 0.005)
    assert (o["gaussian_quaternions_lr"], o["gaussian_quaternions_lr_final"]) == (0.001, 0.001)
    assert o["reconstruction_loss"] == "reg_l1_reg_dssim"
    assert m["medium_fix"]["water_clear_weight"] == 0.0 and m["medium_fix"]["surface_coverage_weight"] == 0.0
    assert m["joint"]["disentanglement"]["opacity_binarization_weight"] == 0.0
    assert m["medium_fix"]["chroma_weight"] == 1.0  # the medium priors stay
    base = resolve_iui3(ROOT / "configs/iui3_default.yaml", tmp_path / "r", overrides=overrides("_sh3_dres"))
    assert base["joint"]["optimization"]["reconstruction_loss"] == "l1_dssim"
    assert base["joint"]["optimization"]["gaussian_adam_eps"] == 1e-8
    run = subprocess.run(["bash", "scripts/train_seathru.sbatch"], cwd=ROOT, capture_output=True, text=True,
                         env={"PATH": "/usr/bin:/bin", "FS_VARIANT_SET": "ws", "SLURM_ARRAY_TASK_ID": "40",
                              "SLURM_ARRAY_JOB_ID": "test"})
    assert "out of range" in run.stderr  # 5 variants x 4 scenes x 2 seeds = tasks 0-39


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_ws_recipe_short_real_training(tmp_path):
    from test_medium_fix import _short_manifest
    m = _short_manifest(tmp_path, "ws", overrides("_sh3_dres_opt_rel_noreg") + [
        "warmup.background=[0.0, 0.0, 0.0]", "joint.background=[0.0, 0.0, 0.0]"])
    fit_iui3(m)
    rows = list(csv.DictReader((tmp_path / "ws/joint/metrics/train.csv").open()))
    assert len(rows) == 4 and all(math.isfinite(float(r["train/loss"])) for r in rows)
