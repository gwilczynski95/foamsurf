from __future__ import annotations

import pytest
import torch

from training import ExperimentConfig
from training.config import OptimizationConfig
from training.optimizer import update_medium_learning_rates


def _optimizer() -> torch.optim.Adam:
    parameters = [torch.nn.Parameter(torch.zeros(1)) for _ in range(3)]
    return torch.optim.Adam([
        {"params": [parameters[0]], "lr": 1.0, "name": "medium_sites"},
        {"params": [parameters[1]], "lr": 1.0, "name": "medium_density"},
        {"params": [parameters[2]], "lr": 1.0, "name": "medium_color"},
    ])


def test_best_voronoi_config_pins_qualified_recipe() -> None:
    config = ExperimentConfig.load("configs/baseline_best_voronoi.yaml")
    config.validate()

    assert config.representation_mode == "voronoi_only"
    assert config.dataset.train_target == "clouds"
    assert config.background == [0.0, 0.0, 0.0]
    assert config.optimization.reconstruction_loss == "l1_dssim"
    assert config.optimization.dssim_weight == pytest.approx(0.2)
    assert config.model.medium_points == 2_048
    assert config.model.medium_initial_density == pytest.approx(0.005)
    assert config.voronoi_adaptation.max_points == 8_192
    assert config.voronoi_adaptation.densify_fraction == pytest.approx(0.05)
    assert config.voronoi_adaptation.interval == 500
    assert config.voronoi_adaptation.child_jitter_fraction == pytest.approx(
        0.025
    )


def test_qualified_medium_scheduler_holds_then_reaches_final_rates() -> None:
    optimizer = _optimizer()
    config = OptimizationConfig(
        medium_sites_lr=5.0e-5,
        medium_density_lr=1.0e-2,
        medium_color_lr=2.5e-3,
        medium_sites_lr_final=5.0e-6,
        medium_density_lr_final=1.0e-3,
        medium_color_lr_final=5.0e-4,
        medium_lr_decay_from=15_000,
        medium_lr_decay_until=30_000,
    )

    held = update_medium_learning_rates(optimizer, config, 15_000)
    assert held["medium_sites"] == pytest.approx(5.0e-5)
    middle = update_medium_learning_rates(optimizer, config, 22_500)
    assert middle["medium_sites"] == pytest.approx((5.0e-5 * 5.0e-6) ** 0.5)
    final = update_medium_learning_rates(optimizer, config, 30_000)
    assert final == pytest.approx({
        "medium_sites": 5.0e-6,
        "medium_density": 1.0e-3,
        "medium_color": 5.0e-4,
    })
