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

    assert gaussians.appearance_dc.grad is not None
    assert gaussians.appearance_dc.grad.abs().sum() > 0
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


def test_b0_and_b01_configs_keep_diagnostics_explicit() -> None:
    from pathlib import Path

    from training.config import ExperimentConfig

    root = Path(__file__).resolve().parents[2]
    b0 = ExperimentConfig.load(root / "configs" / "b0_lego.yaml")
    b01 = ExperimentConfig.load(root / "configs" / "b01_lego.yaml")
    assert b0.diagnostics.enabled is False
    assert b01.diagnostics.enabled is True
    assert b0.optimization == b01.optimization
    assert b0.model == b01.model


def test_b1_uses_graphdeco_gaussian_optimization_from_layered_b01() -> None:
    from pathlib import Path

    from training.config import ExperimentConfig

    root = Path(__file__).resolve().parents[2]
    b01 = ExperimentConfig.load(root / "configs" / "b01_lego_layers.yaml")
    b1 = ExperimentConfig.load(root / "configs" / "b1_lego_layers.yaml")
    assert b01.gaussian_adaptation.enabled is False
    assert b1.gaussian_adaptation.enabled is True
    assert b1.model.gaussian_initial_points == 100_000
    assert b1.gaussian_adaptation.max_points == 200_000
    assert b1.gaussian_adaptation.densification_interval == 100
    assert b1.gaussian_adaptation.gradient_threshold == 2.0e-4
    assert b1.model.gaussian_initial_opacity == 0.1
    assert b1.optimization.gaussian_appearance_lr == 0.0025
    assert b1.optimization.gaussian_appearance_rest_lr_scale == 0.05
    assert b1.optimization.gaussian_opacity_lr == 0.025
    assert b1.optimization.gaussian_means_lr == 0.00016
    assert b1.optimization.gaussian_means_lr_final == 0.0000016
    assert b1.optimization.gaussian_means_spatial_scale is True
    assert b1.optimization.gaussian_scales_lr == 0.005
    assert b1.optimization.gaussian_quaternions_lr == 0.001
    assert b1.optimization.gaussian_adam_eps == 1.0e-15
    assert b1.optimization.gaussian_sh_degree_interval == 1000
    assert b01.model.medium_points == b1.model.medium_points
    assert b01.dataset == b1.dataset
    assert b01.renderer == b1.renderer
