"""Helpers of scripts/variants_vs_ws_report.py."""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import variants_vs_ws_report as v  # noqa: E402


def test_frequency_split_separates_colour_shift_from_detail():
    target = np.zeros((64, 64, 3))
    low, high = v.frequency_split(target + 0.1, target)  # uniform colour shift: all low frequency
    assert math.isclose(low, 0.01, rel_tol=1e-6) and high < 1e-10
    checker = (np.indices((64, 64)).sum(0) % 2)[..., None] * 0.2 - 0.1  # pixel-level detail, zero mean
    low, high = v.frequency_split(target + checker, target)
    assert low < 1e-6 and math.isclose(high, 0.01, rel_tol=1e-2)


def test_scene_table_marks_gap_to_watersplatting():
    ws = dict(all=30.0, seabed=29.0, water=35.0, low=1e-4, high=2e-4, fps=100, gaussians=800)
    ours = dict(all=29.0, seabed=29.5, water=35.0, low=2e-4, high=2e-4, fps=80, gaussians=700)
    table = v.scene_table([("WaterSplatting", ws, "ws"), ("_sh3", ours, "pick")], ours, ws)
    assert table.count("vs WS") == 3  # only our row, only the PSNR columns
    assert "1,00 vs WS" in table and "+0,50 vs WS" in table  # Polish decimal comma
    assert "<td class='num'>2,0</td>" in table  # MSE shown ×1e-4
