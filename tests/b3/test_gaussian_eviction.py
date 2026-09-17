from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from foamsurf_model.gaussian_adaptation import (
    GaussianAdaptationController,
    scheduled_opacity_threshold,
)
from foamsurf_model.gaussians import SurfaceGaussianModel
from training.config import GaussianAdaptationConfig, ModelConfig


def _model(count: int = 5) -> SurfaceGaussianModel:
    points = np.stack(
        (
            np.linspace(-0.5, 0.5, count),
            np.zeros(count),
            np.full(count, 2.0),
        ),
        axis=-1,
    ).astype(np.float32)
    colors = np.full((count, 3), 0.5, dtype=np.float32)
    return SurfaceGaussianModel(
        points,
        colors,
        ModelConfig(appearance="rgb", gaussian_initial_points=count),
        "cpu",
    )


def _controller(model: SurfaceGaussianModel, **overrides):
    values = {
        "enabled": True,
        "policy": "graphdeco",
        "max_points": None,
        "densify_from": 500,
        "densify_until": 15_000,
        "densification_interval": 250,
        "gradient_threshold": 0.1,
        "min_opacity": 0.1,
        "opacity_prune_from": 3_000,
        "opacity_prune_until": 30_000,
        "opacity_prune_interval": 250,
        "size_prune_from": 3_000,
        "screen_pruning_enabled": False,
        "max_world_scale_fraction": 10.0,
    }
    values.update(overrides)
    optimizer = torch.optim.Adam(
        [
            {
                "params": [parameter],
                "name": f"gaussian_{name}",
                "lr": 1.0e-3,
            }
            for name, parameter in model.named_primitive_parameters().items()
        ]
    )
    return GaussianAdaptationController(
        model,
        optimizer,
        GaussianAdaptationConfig(**values),
        scene_extent=10.0,
    )


@pytest.mark.parametrize("shape", ["linear", "cosine", "quadratic"])
def test_opacity_threshold_ramp_has_exact_endpoints(shape: str) -> None:
    config = GaussianAdaptationConfig(
        min_opacity=0.15,
        min_opacity_initial=0.01,
        opacity_prune_from=3_000,
        opacity_prune_ramp=2_000,
        opacity_prune_ramp_shape=shape,
    )
    values = [
        scheduled_opacity_threshold(config, iteration)
        for iteration in range(2_900, 5_101, 100)
    ]
    assert values[0] == pytest.approx(0.01)
    assert scheduled_opacity_threshold(config, 3_000) == pytest.approx(0.01)
    assert scheduled_opacity_threshold(config, 5_000) == pytest.approx(0.15)
    assert values[-1] == pytest.approx(0.15)
    assert all(a <= b for a, b in zip(values, values[1:]))


def test_late_pruning_does_not_restart_densification_or_size_pruning() -> None:
    model = _model(4)
    controller = _controller(model)
    with torch.no_grad():
        model.opacity_logits[0].fill_(torch.logit(torch.tensor(0.01)))
        model.log_scales[1].fill_(math.log(200.0))
    controller.gradient_sum.fill_(1.0)
    controller.gradient_denom.fill_(1.0)

    assert controller.should_prune(16_000)
    assert not controller.should_densify(16_000)
    assert controller.should_adapt(16_000)
    plan = controller.propose_plan(16_000)

    assert plan.prune_count == 1
    assert plan.clone_count == 0
    assert plan.split_count == 0
    assert 0 not in plan.keep_indices.tolist()
    assert 1 in plan.keep_indices.tolist()
    assert controller.last_plan_statistics["world_prune"] == 0
    assert controller.last_plan_statistics["densification_active"] == 0
    assert controller.last_plan_statistics["opacity_pruning_active"] == 1


def test_pruning_end_is_exclusive_and_defaults_to_densification_end() -> None:
    model = _model(2)
    explicit = _controller(model, opacity_prune_until=30_000)
    assert explicit.should_prune(29_750)
    assert not explicit.should_prune(30_000)

    legacy = _controller(model, opacity_prune_until=None)
    assert legacy.should_prune(14_750)
    assert not legacy.should_prune(15_000)


def test_opacity_pruning_preserves_minimum_population() -> None:
    model = _model(5)
    controller = _controller(model, min_points=2)
    with torch.no_grad():
        opacities = torch.tensor([0.01, 0.02, 0.03, 0.04, 0.05])
        model.opacity_logits.copy_(torch.logit(opacities[:, None]))

    plan = controller.propose_plan(4_000)
    assert plan.prune_count == 3
    assert plan.final_count == 2
    assert controller.last_plan_statistics["opacity_prune_raw"] == 5
    assert controller.last_plan_statistics["opacity_prune_rejected"] == 2
    assert plan.keep_indices.tolist() == [3, 4]


def test_opacity_pruning_respects_per_event_fraction() -> None:
    model = _model(5)
    controller = _controller(model, max_opacity_prune_fraction=0.4)
    with torch.no_grad():
        opacities = torch.tensor([0.01, 0.02, 0.03, 0.04, 0.05])
        model.opacity_logits.copy_(torch.logit(opacities[:, None]))

    plan = controller.propose_plan(4_000)
    assert plan.prune_count == 2
    assert plan.keep_indices.tolist() == [2, 3, 4]
    assert controller.last_plan_statistics["opacity_prune_rejected"] == 3


def test_freeze_disables_densification_but_keeps_opacity_pruning() -> None:
    model = _model(4)
    controller = _controller(model)
    with torch.no_grad():
        model.opacity_logits[0].fill_(torch.logit(torch.tensor(0.01)))
    controller.gradient_sum.fill_(1.0)
    controller.gradient_denom.fill_(1.0)

    plan = controller.propose_plan(4_000, allow_densification=False)

    assert plan.clone_count == 0
    assert plan.split_count == 0
    assert plan.prune_count == 1
    assert 0 not in plan.keep_indices.tolist()
    assert controller.last_plan_statistics["densification_active"] == 0
    assert controller.last_plan_statistics["opacity_pruning_active"] == 1
