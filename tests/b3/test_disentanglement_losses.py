from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from training.config import DisentanglementConfig
from training.disentanglement import (
    DisentanglementRegularizer,
    graph_total_variation_loss,
    opacity_binarization_loss,
    opaque_if_used_loss,
    optical_thickness_loss,
    relative_density_cap_loss,
    scheduled_weight,
    undirected_edges,
)


class _Gaussians(nn.Module):
    def __init__(self, opacity: list[float]) -> None:
        super().__init__()
        values = torch.tensor(opacity, dtype=torch.float64).reshape(-1, 1)
        self.opacity_logits = nn.Parameter(torch.logit(values))

    @property
    def opacities(self) -> torch.Tensor:
        return self.opacity_logits.sigmoid()


class _Medium(nn.Module):
    def __init__(
        self, density: list[float], guard: list[bool] | None = None
    ) -> None:
        super().__init__()
        self.density_parameter = nn.Parameter(
            torch.tensor(density, dtype=torch.float64).reshape(-1, 1)
        )
        if guard is None:
            guard = [False] * len(density)
        self.register_buffer("guard_mask", torch.tensor(guard))

    @property
    def densities(self) -> torch.Tensor:
        return self.density_parameter


class _Model(nn.Module):
    def __init__(
        self,
        opacity: list[float],
        density: list[float],
        guard: list[bool] | None = None,
    ) -> None:
        super().__init__()
        self.gaussians = _Gaussians(opacity)
        self.medium = _Medium(density, guard)


def _triangle_topology() -> SimpleNamespace:
    return SimpleNamespace(
        adjacency=torch.tensor([1, 2, 0, 2, 0, 1]),
        offsets=torch.tensor([0, 2, 4, 6]),
    )


@pytest.mark.parametrize("shape", ["linear", "cosine", "quadratic"])
def test_schedule_has_exact_endpoints_and_is_monotonic(shape: str) -> None:
    values = [
        scheduled_weight(0.03, iteration, 100, 200, shape)
        for iteration in range(0, 401)
    ]
    assert values[99] == 0.0
    assert values[100] == 0.0
    assert values[300] == pytest.approx(0.03)
    assert values[-1] == pytest.approx(0.03)
    assert all(a <= b for a, b in zip(values, values[1:]))


def test_zero_length_schedule_is_a_delayed_step() -> None:
    assert scheduled_weight(0.2, 9, 10, 0, "linear") == 0.0
    assert scheduled_weight(0.2, 10, 10, 0, "linear") == 0.2


def test_schedule_can_stop_or_keep_a_maintenance_weight() -> None:
    stopped = [
        scheduled_weight(
            0.2, iteration, 100, 200, "linear",
            end=500, maintenance_scale=0.0,
        )
        for iteration in (499, 500, 900)
    ]
    assert stopped == pytest.approx([0.2, 0.0, 0.0])

    maintained = scheduled_weight(
        0.2, 501, 100, 200, "linear",
        end=500, maintenance_scale=0.1,
    )
    assert maintained == pytest.approx(0.02)


@pytest.mark.parametrize("shape", ["linear", "cosine", "quadratic"])
def test_schedule_end_ramp_has_exact_endpoints(shape: str) -> None:
    before = scheduled_weight(
        0.2, 499, 100, 200, "linear",
        end=500, end_ramp=100, end_ramp_shape=shape,
        maintenance_scale=0.1,
    )
    at_start = scheduled_weight(
        0.2, 500, 100, 200, "linear",
        end=500, end_ramp=100, end_ramp_shape=shape,
        maintenance_scale=0.1,
    )
    at_end = scheduled_weight(
        0.2, 600, 100, 200, "linear",
        end=500, end_ramp=100, end_ramp_shape=shape,
        maintenance_scale=0.1,
    )
    assert before == pytest.approx(0.2)
    assert at_start == pytest.approx(0.2)
    assert at_end == pytest.approx(0.02)


def test_opacity_binarization_value_and_gradient_direction() -> None:
    logits = nn.Parameter(
        torch.logit(torch.tensor([[0.1], [0.9]], dtype=torch.float64))
    )
    loss = opacity_binarization_loss(logits.sigmoid())
    assert loss.item() == pytest.approx(0.09)
    loss.backward()
    assert logits.grad[0].item() > 0.0
    assert logits.grad[1].item() < 0.0


def test_opaque_if_used_is_normalized_and_stops_contribution_gradient() -> None:
    opacity = torch.tensor([0.25, 0.5], dtype=torch.float64, requires_grad=True)
    contribution = torch.tensor(
        [1.0, 3.0], dtype=torch.float64, requires_grad=True
    )
    loss = opaque_if_used_loss(opacity, contribution)
    expected = (-math.log(0.25) - 3.0 * math.log(0.5)) / 4.0
    assert loss.item() == pytest.approx(expected)
    loss.backward()
    assert contribution.grad is None
    assert opacity.grad.tolist() == pytest.approx([-1.0, -1.5])


def test_zero_contribution_has_zero_opaque_loss_and_gradient() -> None:
    opacity = torch.tensor([0.2, 0.8], requires_grad=True)
    loss = opaque_if_used_loss(opacity, torch.zeros(2))
    assert loss.item() == 0.0
    loss.backward()
    torch.testing.assert_close(opacity.grad, torch.zeros_like(opacity))


def test_undirected_edges_deduplicate_symmetric_adjacency() -> None:
    topology = _triangle_topology()
    first, second = undirected_edges(
        topology.adjacency, topology.offsets, 3
    )
    assert list(zip(first.tolist(), second.tolist())) == [
        (0, 1), (0, 2), (1, 2)
    ]


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("l1", 2.0), ("l2", 14.0 / 3.0), ("huber", 1.5)],
)
def test_graph_tv_exact_values_and_gradients(mode: str, expected: float) -> None:
    density = torch.tensor(
        [1.0, 2.0, 4.0], dtype=torch.float64, requires_grad=True
    )
    topology = _triangle_topology()
    loss, edge_count = graph_total_variation_loss(
        density,
        topology.adjacency,
        topology.offsets,
        mode=mode,
        huber_delta=1.0,
    )
    assert edge_count == 3
    assert loss.item() == pytest.approx(expected)
    loss.backward()
    assert density.grad is not None
    assert density.grad.sum().item() == pytest.approx(0.0, abs=1.0e-12)


def test_density_cap_is_relative_and_excludes_values_below_cap() -> None:
    density = torch.tensor(
        [1.0, 2.0, 8.0], dtype=torch.float64, requires_grad=True
    )
    loss, fraction = relative_density_cap_loss(density, 4.0)
    assert loss.item() == pytest.approx(1.0 / 3.0)
    assert fraction.item() == pytest.approx(1.0 / 3.0)
    loss.backward()
    assert density.grad[:2].tolist() == [0.0, 0.0]
    assert density.grad[2].item() > 0.0


def test_optical_thickness_is_negative_log_medium_transmittance() -> None:
    transmittance = torch.tensor(
        [[math.exp(-1.0)], [math.exp(-3.0)]],
        dtype=torch.float64,
        requires_grad=True,
    )
    loss = optical_thickness_loss(transmittance)
    assert loss.item() == pytest.approx(2.0)
    loss.backward()
    expected = -0.5 / transmittance.detach()
    torch.testing.assert_close(transmittance.grad, expected)


def test_combined_regularizer_matches_sum_of_normalized_terms() -> None:
    config = DisentanglementConfig(
        opacity_binarization_weight=0.1,
        opaque_if_used_weight=0.2,
        graph_tv_weight=0.3,
        graph_tv_mode="l1",
        density_cap_weight=0.4,
        density_cap_policy="fixed",
        density_cap_fixed_value=2.0,
        optical_thickness_weight=0.5,
    )
    model = _Model([0.25, 0.75], [1.0, 2.0, 4.0])
    diagnostics = SimpleNamespace(
        gaussian=torch.tensor([[1.0, 0, 0, 0], [3.0, 0, 0, 0]])
    )
    transmittance = torch.tensor([[math.exp(-2.0)]], dtype=torch.float64)
    regularizer = DisentanglementRegularizer(config)
    result = regularizer.compute(
        1,
        model,
        _triangle_topology(),
        diagnostics=diagnostics,
        medium_transmittance=transmittance,
    )
    expected = (
        0.1 * result.opacity_binarization
        + 0.2 * result.opaque_if_used
        + 0.3 * result.graph_tv
        + 0.4 * result.density_cap
        + 0.5 * result.optical_thickness
    )
    torch.testing.assert_close(result.total, expected)
    assert set(result.metrics()) == {
        "disentanglement/total",
        "disentanglement/density_cap_value",
        "disentanglement/density_cap_quantile",
        "disentanglement/density_cap_violation_fraction",
        "disentanglement/graph_edge_count",
        *{
            f"disentanglement/{name}_{suffix}"
            for name in (
                "opacity_binarization", "opaque_if_used", "graph_tv",
                "density_cap", "optical_thickness",
            )
            for suffix in ("raw", "weight", "weighted")
        },
    }


def test_density_cap_ema_state_round_trips_exactly() -> None:
    config = DisentanglementConfig(
        density_cap_weight=1.0,
        density_cap_policy="ema",
        density_cap_quantile=0.5,
        density_cap_multiplier=2.0,
        density_cap_minimum=0.1,
        density_cap_update_interval=1,
        density_cap_ema_decay=0.5,
    )
    model = _Model([0.5], [1.0, 2.0, 8.0, 100.0], [False] * 3 + [True])
    first = DisentanglementRegularizer(config)
    first_result = first.compute(1, model, _triangle_topology())
    assert first_result.density_cap_value == pytest.approx(4.0)
    with torch.no_grad():
        model.medium.density_parameter[:3] = torch.tensor(
            [[2.0], [4.0], [16.0]], dtype=torch.float64
        )
    second_result = first.compute(2, model, _triangle_topology())
    assert second_result.density_cap_value == pytest.approx(6.0)

    restored = DisentanglementRegularizer(config)
    restored.load_state_dict(first.state_dict())
    assert restored.state_dict() == first.state_dict()


def test_warmup_freeze_cap_stops_following_distribution() -> None:
    config = DisentanglementConfig(
        density_cap_weight=1.0,
        density_cap_policy="warmup_freeze",
        density_cap_quantile=0.5,
        density_cap_multiplier=2.0,
        density_cap_minimum=0.1,
        density_cap_update_interval=1,
        density_cap_ema_decay=0.0,
        density_cap_warmup_until=2,
    )
    model = _Model([0.5], [1.0, 2.0, 8.0])
    regularizer = DisentanglementRegularizer(config)
    regularizer.compute(1, model, _triangle_topology())
    at_freeze = regularizer.compute(2, model, _triangle_topology())
    with torch.no_grad():
        model.medium.density_parameter.mul_(10.0)
    after = regularizer.compute(3, model, _triangle_topology())
    assert after.density_cap_value == at_freeze.density_cap_value


def test_regularizer_declares_only_required_extra_renderer_work() -> None:
    config = DisentanglementConfig(
        opaque_if_used_weight=1.0,
        opaque_if_used_start=100,
        optical_thickness_weight=1.0,
        optical_thickness_start=200,
    )
    regularizer = DisentanglementRegularizer(config)
    assert not regularizer.requires_contribution_diagnostics(99)
    assert regularizer.requires_contribution_diagnostics(100)
    assert not regularizer.requires_medium_transmittance(199)
    assert regularizer.requires_medium_transmittance(200)
