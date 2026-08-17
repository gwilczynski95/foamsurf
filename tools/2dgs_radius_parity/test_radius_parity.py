"""CUDA smoke test against the pinned official 2DGS rasterizer."""

import pytest
import torch

from radius_parity import deterministic_cases, randomized_cases, summary


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="official 2DGS rasterizer needs CUDA"
)


def test_deterministic_radii_match_2dgs_within_one_pixel() -> None:
    result = deterministic_cases(torch.device("cuda"))
    assert summary(result)["max_absolute_error"] <= 1.0


def test_random_radii_match_2dgs_within_one_pixel() -> None:
    result = randomized_cases(torch.device("cuda"), 2048, 2025)
    assert summary(result)["max_absolute_error"] <= 1.0
