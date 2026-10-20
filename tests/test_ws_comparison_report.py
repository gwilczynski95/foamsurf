"""Helpers of scripts/ws_comparison_report.py."""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import ws_comparison_report as w  # noqa: E402


def test_psnr_regions():
    target = np.zeros((4, 4, 3))
    prediction = target.copy()
    prediction[:2] = 0.1  # error only in the top half
    mask = np.zeros((4, 4), bool)
    mask[:2] = True
    assert math.isclose(w.psnr(prediction, target, mask), 20.0)
    assert w.psnr(prediction, target, ~mask) > 100 and math.isnan(w.psnr(prediction, target, np.zeros((4, 4), bool)))


def test_worst_blocks_find_where_ours_loses():
    target = np.zeros((40, 60, 3))
    ws = target.copy()
    ours = target.copy()
    ours[20:30, 40:50] = 0.5  # FoamSurf wrong in one block
    ws[0:10, 0:10] = 0.5      # WaterSplatting wrong elsewhere
    blocks = w.worst_blocks(ours, ws, target, count=2, size=10)
    assert blocks[0][:2] == (20, 40) and blocks[0][2] > 0
    assert blocks[-1][2] <= blocks[0][2]


def test_difference_map_colours():
    target = np.zeros((1, 3, 3))
    ours = np.array([[[0.0] * 3, [0.2] * 3, [0.1] * 3]])
    ws = np.array([[[0.2] * 3, [0.0] * 3, [0.1] * 3]])
    picture = np.asarray(w.difference_map(ours, ws, target)).astype(int)
    assert (picture[0, 0] == w.OURS_BETTER).all() and (picture[0, 1] == w.WS_BETTER).all()
    assert (picture[0, 2] == w.NEUTRAL).all()
