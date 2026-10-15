"""Comparison logic of scripts/renderer_check.py (the renderer itself needs CUDA)."""
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import renderer_check as check  # noqa: E402


def snapshot(shift=0.0, outliers=0, grad_scale=1.0, hits=3):
    torch.manual_seed(0)
    image = torch.rand(20, 50, 3)
    view = dict(image=image + shift, backscatter=image * 0.2, alpha=torch.rand(20, 50, 1),
                gaussian_alpha=torch.rand(20, 50, 1), hit_count=torch.full((20, 50), hits, dtype=torch.int16),
                hit_buffer_size=64)
    view["image"].view(-1)[:outliers] += 0.01
    grads = {"gaussians.means": torch.randn(10, 3) * grad_scale, "medium.sites": torch.randn(4, 3)}
    return dict(views={"A": view}, gradient=dict(grads=grads), gaussians=10, cells=4)


def test_identical_snapshots_pass():
    images = check.compare_images(snapshot(), snapshot())
    gradients = check.compare_gradients(snapshot(), snapshot())
    assert all(row["ok"] for row in images + gradients)
    assert math.isinf(images[0]["psnr"]) and gradients[0]["relative_l2"] == 0
    assert check.report(images, gradients)


def test_tolerances():
    # 1e-6 everywhere passes, 1e-4 everywhere fails on the mean.
    assert all(r["ok"] for r in check.compare_images(snapshot(), snapshot(shift=1e-6)))
    assert not check.compare_images(snapshot(), snapshot(shift=1e-4))[0]["ok"]
    # Values off by 1e-2 on 0.5% of the image fail although the mean is small.
    rows = check.compare_images(snapshot(), snapshot(outliers=15))  # 15 of 3000 values
    assert rows[0]["outlier_fraction"] == pytest.approx(15 / 3000) and not rows[0]["ok"]
    gradients = check.compare_gradients(snapshot(), snapshot(grad_scale=1.01))
    assert not gradients[0]["ok"] and abs(gradients[0]["relative_l2"] - 0.01) < 1e-6
    assert gradients[0]["cosine"] > 0.9999 and gradients[1]["ok"]


def test_missing_gradient_and_full_hit_buffer_are_reported():
    actual = snapshot()
    del actual["gradient"]["grads"]["medium.sites"]
    rows = {row["name"]: row for row in check.compare_gradients(snapshot(), actual)}
    assert not rows["medium.sites"]["ok"] and rows["gaussians.means"]["ok"]
    assert check.compare_images(snapshot(hits=64), snapshot(hits=64))[0]["hit_buffer_full_fraction"] == 1.0
    assert not check.report([], list(rows.values()))
