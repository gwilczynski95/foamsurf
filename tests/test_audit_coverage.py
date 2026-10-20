"""Coverage metrics of scripts/medium_audit.py (open water, seabed center vs borders)."""
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from medium_audit import OPEN_WATER, coverage_metrics, load_open_water  # noqa: E402


def test_seabed_center_and_borders():
    alpha = torch.zeros(10, 8)
    alpha[6:, 2:6] = 1.0  # opaque seabed only in the central half
    row = coverage_metrics(alpha, 0.4, None)
    assert row["bottom_center_alpha_gt_0.99"] == 1.0
    assert row["bottom_border_alpha_gt_0.99"] == 0.0
    assert math.isnan(row["open_water_alpha_mean"]) and math.isnan(row["open_water_alpha_gt_0.5"])
    alpha[6:, :1] = 1.0  # a quarter of the border columns
    assert coverage_metrics(alpha, 0.4)["bottom_border_alpha_gt_0.99"] == 0.25


def test_open_water_share_taken_by_gaussians():
    alpha = torch.zeros(4, 4)
    water = torch.zeros(4, 4, dtype=torch.bool)
    water[:2] = True
    alpha[0, :2] = 0.8  # 2 of 8 open-water pixels invaded
    alpha[3] = 1.0  # seabed outside the mask does not count
    row = coverage_metrics(alpha, 0.25, water)
    assert row["open_water_alpha_gt_0.5"] == 0.25
    assert row["seabed_alpha_gt_0.99"] == 0.5 and row["seabed_alpha_gt_0.5"] == 0.5  # row 3 of rows 2-3
    assert row["open_water_alpha_mean"] == torch.tensor(0.2).item()


def test_audit_loads_the_masks_of_every_scene(tmp_path):
    masks = load_open_water(OPEN_WATER)
    assert {"MTN_5894", "MTN_5903", "MTN_5911", "MTN_5928", "MTN_1288", "MTN_1090", "MTN_1529"} <= set(masks)
    for view in ("MTN_5894", "MTN_5903", "MTN_5911", "MTN_5928"):
        assert masks[view].dtype == bool and masks[view].shape == (917, 1383) and 0.1 < masks[view].mean() < 0.9
    assert load_open_water(tmp_path / "missing.npz") == {}
    path = tmp_path / "m.npz"
    np.savez(path, meta="{}", A=np.ones((2, 2), bool))
    assert list(load_open_water(path)) == ["A"]
