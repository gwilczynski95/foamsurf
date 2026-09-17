from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from training import ExperimentConfig
from training.config import DisentanglementConfig
from training.disentanglement import (
    DisentanglementRegularizer,
    optical_thickness_loss,
    undirected_edges,
)
from tests.b3.test_disentanglement_losses import _Model, _triangle_topology


def test_disentanglement_config_yaml_round_trip(tmp_path) -> None:
    config = ExperimentConfig.load("configs/baseline_clean_joint.yaml")
    config.disentanglement = DisentanglementConfig(
        opacity_binarization_weight=0.003,
        opacity_binarization_start=500,
        opacity_binarization_ramp=2_000,
        opacity_binarization_ramp_shape="cosine",
        graph_tv_weight=0.01,
        graph_tv_mode="huber",
        graph_tv_huber_delta=0.005,
        density_cap_weight=0.01,
        density_cap_policy="warmup_freeze",
    )
    path = tmp_path / "config.yaml"
    config.save(path)
    restored = ExperimentConfig.load(path)
    assert restored == config


def test_missing_opaque_diagnostics_fails_loudly() -> None:
    regularizer = DisentanglementRegularizer(
        DisentanglementConfig(opaque_if_used_weight=1.0)
    )
    with pytest.raises(ValueError, match="requires render diagnostics"):
        regularizer.compute(
            1, _Model([0.5], [1.0, 2.0, 3.0]), _triangle_topology()
        )


def test_missing_medium_transmittance_fails_loudly() -> None:
    regularizer = DisentanglementRegularizer(
        DisentanglementConfig(optical_thickness_weight=1.0)
    )
    with pytest.raises(ValueError, match="requires medium transmittance"):
        regularizer.compute(
            1, _Model([0.5], [1.0, 2.0, 3.0]), _triangle_topology()
        )


def test_rolling_cap_follows_current_detached_quantile() -> None:
    config = DisentanglementConfig(
        density_cap_weight=1.0,
        density_cap_policy="rolling",
        density_cap_quantile=0.5,
        density_cap_multiplier=2.0,
        density_cap_update_interval=1,
    )
    model = _Model([0.5], [1.0, 2.0, 8.0])
    regularizer = DisentanglementRegularizer(config)
    first = regularizer.compute(1, model, _triangle_topology())
    with torch.no_grad():
        model.medium.density_parameter.mul_(3.0)
    second = regularizer.compute(2, model, _triangle_topology())
    assert first.density_cap_value == pytest.approx(4.0)
    assert second.density_cap_value == pytest.approx(12.0)
    assert regularizer.state_dict()["density_stat_ema"] is None


def test_malformed_csr_topology_is_rejected() -> None:
    with pytest.raises(ValueError, match="inconsistent"):
        undirected_edges(
            torch.tensor([1, 0]), torch.tensor([0, 2, 1]), point_count=2
        )


def test_optical_thickness_clamps_extreme_transmittance_to_finite_loss() -> None:
    transmittance = torch.tensor([[0.0], [1.0], [2.0]])
    loss = optical_thickness_loss(transmittance, epsilon=1.0e-6)
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(-torch.log(torch.tensor(1.0e-6)).item() / 3.0)


def test_zero_weight_regularizer_needs_no_auxiliary_inputs() -> None:
    model = _Model([0.5], [1.0, 2.0, 3.0])
    result = DisentanglementRegularizer(
        DisentanglementConfig()
    ).compute(1, model, _triangle_topology())
    assert result.total.item() == 0.0
    assert all(
        value == 0.0
        for key, value in result.metrics().items()
        if key.endswith(("_raw", "_weight", "_weighted"))
    )
