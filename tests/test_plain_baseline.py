"""Plain baseline ('_plain'): only L1 + D-SSIM, every extra loss and the compensated densification off."""
import csv
import math
import subprocess
from pathlib import Path

import pytest
import torch

from train import fit_iui3, resolve_iui3

ROOT = Path(__file__).resolve().parents[1]


def overrides(variant):
    script = (ROOT / "scripts/train_seathru.sbatch").read_text()
    start = script.index("variant_overrides() {")
    function = script[start:script.index("\n}\n", start) + 3]
    return subprocess.run(["bash", "-c", function + f"\nvariant_overrides {variant}"],
                          capture_output=True, text=True, check=True).stdout.splitlines()


def test_plain_turns_off_every_extra_loss(tmp_path):
    m = resolve_iui3(ROOT / "configs/iui3_default.yaml", tmp_path / "r", overrides=overrides("_plain"))
    fix = m["medium_fix"]
    assert all(fix[k] == 0.0 for k in ("chroma_weight", "surface_coverage_weight", "water_clear_weight",
                                       "under_surface_weight"))
    assert m["rgb_surface"]["weight"] == 0.0
    assert all(m["rgb_smoothing"][k] == 0.0 for k in ("attenuation", "backscatter", "color"))
    assert m["capacity_loss"]["repulsion_weight"] == 0.0 and m["capacity_loss"]["direct_weight"] == 0.0
    joint = m["joint"]
    assert joint["disentanglement"]["opacity_binarization_weight"] == 0.0
    assert joint["optimization"]["gaussian_scale_regularization"] == 0.0
    assert joint["optimization"]["reconstruction_loss"] == "l1_dssim" and joint["optimization"]["dssim_weight"] == 0.2
    adaptation = joint["gaussian_adaptation"]
    assert adaptation["gradient_attenuation_compensation"] is False
    default = resolve_iui3(ROOT / "configs/iui3_default.yaml", tmp_path / "r", overrides=[])["joint"]["gaussian_adaptation"]
    for key in ("densify_from", "densify_until", "gradient_threshold", "max_points", "min_opacity",
                "opacity_prune_from", "opacity_prune_interval"):  # densification and pruning as in the default
        assert adaptation[key] == default[key], key
    run = subprocess.run(["bash", "scripts/train_seathru.sbatch"], cwd=ROOT, capture_output=True, text=True,
                         env={"PATH": "/usr/bin:/bin", "FS_VARIANT_SET": "plain", "SLURM_ARRAY_TASK_ID": "8",
                              "SLURM_ARRAY_JOB_ID": "test"})
    assert "out of range" in run.stderr  # 4 scenes x 2 seeds = tasks 0-7


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_plain_short_real_training_has_only_the_image_loss(tmp_path):
    from test_medium_fix import _short_manifest
    m = _short_manifest(tmp_path, "plain", overrides("_plain") + [
        "warmup.background=[0.0, 0.0, 0.0]", "joint.background=[0.0, 0.0, 0.0]"])
    fit_iui3(m)
    rows = list(csv.DictReader((tmp_path / "plain/joint/metrics/train.csv").open()))
    assert len(rows) == 4 and all(math.isfinite(float(r["train/loss"])) for r in rows)
    assert not any(k.startswith("medium_fix/") for k in rows[0])
    for row in rows:  # every extra term contributes exactly zero
        for key in ("rgb_smooth/weighted", "surface_overlap/weighted", "train/gaussian_scale_regularization",
                    "disentanglement/opacity_binarization_weighted", "capacity/repulsion_weighted",
                    "capacity/direct_weighted"):
            if key in row:
                assert float(row[key]) == 0.0, key
        image_loss = 0.8 * float(row["train/rgb_l1"]) + 0.2 * (1.0 - float(row["train/ssim"]))
        assert math.isclose(float(row["train/loss"]), image_loss, rel_tol=1e-4, abs_tol=1e-6)
