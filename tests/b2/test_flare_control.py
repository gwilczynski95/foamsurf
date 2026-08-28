from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch import nn

from foamsurf_model.gaussian_adaptation import GaussianAdaptationController
from foamsurf_model.gaussians import SurfaceGaussianModel
from training.config import (
    ExperimentConfig, GaussianAdaptationConfig, ModelConfig, OptimizationConfig,
)
from training.optimizer import update_gaussian_learning_rates


def test_flare_control_pins_working_core_without_topology() -> None:
    config = ExperimentConfig.load(
        "configs/stage1_gaussian_flare_control.yaml"
    )
    assert config.model.appearance == "rgb"
    assert config.model.gaussian_initial_opacity == pytest.approx(0.01)
    assert config.model.gaussian_initial_scale_neighbors == 3
    assert config.optimization.reconstruction_loss == "l1_dssim"
    assert config.optimization.gaussian_scale_regularization == pytest.approx(
        0.001
    )
    assert config.optimization.gaussian_min_scale == pytest.approx(0.0001)
    assert config.optimization.gaussian_max_scale_fraction == pytest.approx(
        0.05
    )
    assert not config.gaussian_adaptation.enabled
    assert config.gaussian_adaptation.opacity_reset_interval == 0


def test_flare_learning_rate_schedule_updates_every_gaussian_group() -> None:
    config = OptimizationConfig(
        gaussian_appearance_lr=0.005,
        gaussian_appearance_rest_lr_scale=1.0,
        gaussian_appearance_lr_decay=-0.1,
        gaussian_appearance_lr_final=0.001,
        gaussian_opacity_lr=0.05,
        gaussian_opacity_lr_decay=-0.1,
        gaussian_opacity_lr_final=0.025,
        gaussian_kappa_lr=0.1,
        gaussian_kappa_lr_decay=-0.1,
        gaussian_kappa_lr_final=0.1,
        gaussian_means_lr=0.0025,
        gaussian_means_lr_decay=-0.15,
        gaussian_means_lr_final=1.0e-6,
        gaussian_means_spatial_scale=False,
        gaussian_scales_lr=0.04,
        gaussian_scales_lr_decay=-0.1,
        gaussian_scales_lr_final=0.04,
        gaussian_quaternions_lr=0.01,
        gaussian_quaternions_lr_decay=-0.1,
        gaussian_quaternions_lr_final=0.001,
    )
    names = (
        "gaussian_appearance_dc",
        "gaussian_appearance_rest",
        "gaussian_opacity_logits",
        "gaussian_raw_kappa",
        "gaussian_means",
        "gaussian_log_scales",
        "gaussian_quaternions",
    )
    parameters = [nn.Parameter(torch.zeros(())) for _ in names]
    optimizer = torch.optim.Adam(
        [
            {"params": [parameter], "lr": 1.0, "name": name}
            for name, parameter in zip(names, parameters)
        ]
    )
    means_rate = update_gaussian_learning_rates(
        optimizer, config, scene_extent=9.0, iteration=10
    )
    rates = {group["name"]: group["lr"] for group in optimizer.param_groups}
    assert means_rate == pytest.approx(0.0025 * math.exp(-1.5))
    assert rates["gaussian_appearance_dc"] == pytest.approx(
        0.005 * math.exp(-1.0)
    )
    assert rates["gaussian_opacity_logits"] == pytest.approx(0.025)
    assert rates["gaussian_raw_kappa"] == pytest.approx(0.1)
    assert rates["gaussian_log_scales"] == pytest.approx(0.04)


def test_three_neighbor_initial_scale_matches_flare_definition() -> None:
    points = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
         [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
        dtype=np.float32,
    )
    log_scales = SurfaceGaussianModel._initial_log_scales(
        points, neighbor_count=3
    )
    expected = math.sqrt((1.0 + 4.0 + 9.0) / 3.0)
    assert math.exp(float(log_scales[0, 0])) == pytest.approx(expected)
    assert np.array_equal(log_scales[:, 0], log_scales[:, 1])



def test_flare_movement_duplicates_pre_and_post_step_and_maps_adam_state() -> None:
    points = np.asarray(
        [[-0.2, 0.0, 2.0], [0.0, 0.0, 2.0], [0.2, 0.0, 2.0]],
        dtype=np.float32,
    )
    colors = np.full((3, 3), 0.5, dtype=np.float32)
    model = SurfaceGaussianModel(
        points,
        colors,
        ModelConfig(
            appearance="rgb",
            gaussian_initial_points=3,
            gaussian_initial_opacity=0.1,
        ),
        "cpu",
    )
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
    sum(parameter.sum() for parameter in model.parameters()).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    controller = GaussianAdaptationController(
        model,
        optimizer,
        GaussianAdaptationConfig(
            enabled=True,
            policy="flare_movement",
            max_points=None,
            densify_from=800,
            densify_until=24_000,
            densification_interval=100,
            movement_threshold=1.0e-3,
            min_opacity=0.005,
            min_effective_scale=0.0,
        ),
        scene_extent=2.0,
    )
    assert not controller.should_collect(2)
    assert controller.should_adapt(800)
    assert controller.should_adapt(24_000)
    assert not controller.should_adapt(700)
    assert not controller.should_adapt(24_100)
    snapshot = controller.snapshot_movement()
    with torch.no_grad():
        model.means[1, 0] += 0.01
        model.opacity_logits[0].fill_(torch.logit(torch.tensor(0.001)))

    event = controller.apply_movement(snapshot)

    assert event == {
        "clone": 0, "split": 1, "prune": 1, "budget_skip": 0,
    }
    assert model.point_count == 3
    torch.testing.assert_close(model.means[1], snapshot.means[1])
    assert float(model.means[2, 0]) == pytest.approx(
        float(snapshot.means[1, 0] + 0.01)
    )
    for parameter in model.named_primitive_parameters().values():
        state = optimizer.state[parameter]
        assert state["exp_avg"].shape == parameter.shape
        assert state["exp_avg_sq"].shape == parameter.shape
