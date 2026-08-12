from __future__ import annotations

import csv

import numpy as np
import pytest
import torch

from foamsurf import HybridRenderer
from foamsurf_model.gaussians import SurfaceGaussianModel
from foamsurf_model.medium import VoronoiMediumModel
from training.config import ModelConfig
from training.experiment_logging import CsvTable
from training.losses import b0_reconstruction_loss


def test_b0_loss_is_exactly_hybrid_rgb_mse() -> None:
    prediction = torch.tensor([[0.2, 0.4, 0.8]])
    target = torch.tensor([[0.1, 0.7, 0.5]])
    result = b0_reconstruction_loss(prediction, target)
    expected = ((prediction - target) ** 2).mean()
    torch.testing.assert_close(result.total, expected)
    torch.testing.assert_close(result.rgb_mse, expected)


def test_oracle_b0_loss_reaches_both_representations() -> None:
    config = ModelConfig(
        appearance="rgb",
        gaussian_initial_points=1,
        medium_points=8,
        medium_guard_points=0,
        medium_initial_density=0.05,
    )
    gaussians = SurfaceGaussianModel(
        np.array([[0.0, 0.0, 2.0]], dtype=np.float32),
        np.array([[0.8, 0.2, 0.1]], dtype=np.float32),
        config,
        "cpu",
    )
    with torch.no_grad():
        gaussians.quaternions.copy_(
            torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        )
        gaussians.log_scales.fill_(-0.5)
        gaussians.opacity_logits.fill_(1.0)
    medium = VoronoiMediumModel(
        torch.tensor([-1.0, -1.0, 0.0]),
        torch.tensor([1.0, 1.0, 4.0]),
        config,
        "cpu",
    )
    rays = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    render = HybridRenderer(far=4.0)(
        rays,
        gaussians=gaussians.render_scene(None),
        voronoi=medium.render_scene(),
    )
    loss = b0_reconstruction_loss(
        render.image, torch.tensor([[0.3, 0.6, 0.2]])
    )
    loss.total.backward()

    assert gaussians.appearance.grad is not None
    assert gaussians.appearance.grad.abs().sum() > 0
    assert medium.raw_density.grad is not None
    assert medium.raw_density.grad.abs().sum() > 0
    assert medium.raw_color.grad is not None
    assert medium.raw_color.grad.abs().sum() > 0


def test_medium_initialization_is_sparse_positive_and_volumetric() -> None:
    config = ModelConfig(
        medium_points=64,
        medium_guard_points=8,
        medium_initial_density=0.01,
    )
    lower = torch.tensor([-2.0, -1.0, 0.0])
    upper = torch.tensor([2.0, 3.0, 6.0])
    medium = VoronoiMediumModel(
        lower, upper, config, device="cpu"
    )

    assert medium.point_count == 64
    torch.testing.assert_close(
        medium.densities,
        torch.full((64, 1), 0.01),
        atol=1.0e-7,
        rtol=1.0e-6,
    )
    assert torch.all(medium.densities > 0)
    span = medium.sites.amax(0) - medium.sites.amin(0)
    assert torch.all(span > 0.9 * (upper - lower))


def test_csv_table_preserves_stable_schema(tmp_path) -> None:
    path = tmp_path / "metrics.csv"
    table = CsvTable(path)
    table.append({"iteration": 1, "loss": 0.5})
    table.append({"iteration": 2, "loss": 0.25})
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert rows == [
        {"iteration": "1", "loss": "0.5"},
        {"iteration": "2", "loss": "0.25"},
    ]
    with pytest.raises(ValueError, match="schema mismatch"):
        table.append({"iteration": 3, "psnr": 10.0})
