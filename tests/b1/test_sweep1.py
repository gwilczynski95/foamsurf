from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from foamsurf_model.gaussian_adaptation import (
    GaussianAdaptationController,
    GaussianTopologyPlan,
)
from foamsurf_model.gaussians import SurfaceGaussianModel
from training.config import ExperimentConfig, GaussianAdaptationConfig, ModelConfig
from training.losses import b0_reconstruction_loss
from training.sweep1 import SWEEP1_SPECS, build_sweep1_config


def _model(count: int = 2) -> SurfaceGaussianModel:
    points = np.zeros((count, 3), dtype=np.float32)
    points[:, 2] = 2.0
    colors = np.full((count, 3), 0.5, dtype=np.float32)
    return SurfaceGaussianModel(
        points,
        colors,
        ModelConfig(appearance="rgb", gaussian_initial_points=count),
        "cpu",
    )


def _optimizer(model: SurfaceGaussianModel) -> torch.optim.Adam:
    return torch.optim.Adam(
        [
            {
                "params": [parameter],
                "name": f"gaussian_{name}",
                "lr": 1.0e-3,
            }
            for name, parameter in model.named_primitive_parameters().items()
        ]
    )


def _controller(model, optimizer, **overrides):
    values = dict(
        enabled=True,
        max_points=None,
        densify_until=15_000,
        opacity_reset_interval=3_000,
        opacity_reset_value=0.01,
    )
    values.update(overrides)
    return GaussianAdaptationController(
        model,
        optimizer,
        GaussianAdaptationConfig(**values),
        scene_extent=4.0,
    )


def test_sweep1_has_twenty_unique_resolved_30k_configs(tmp_path) -> None:
    root = Path(__file__).resolve().parents[2]
    base = ExperimentConfig.load(root / "configs" / "b1_lego_layers.yaml")
    assert len(SWEEP1_SPECS) == 20
    assert len({spec.label for spec in SWEEP1_SPECS}) == 20
    for index, spec in enumerate(SWEEP1_SPECS):
        config = build_sweep1_config(
            base,
            index,
            tmp_path / spec.label,
            f"test-{spec.label}",
        )
        assert config.optimization.iterations == 30_000
        assert config.optimization.full_image_batch is True
        assert config.optimization.cameras_without_replacement is True
        assert config.gaussian_adaptation.densify_until == 15_000
        assert config.gaussian_adaptation.max_points is None
        assert config.gaussian_adaptation.runaway_max_points == 1_000_000
        assert config.gaussian_adaptation.screen_pruning_enabled is True
        assert config.wandb.project == "foamsurf_sweep1"


def test_sweep1_sbatch_labels_match_python_specs() -> None:
    root = Path(__file__).resolve().parents[2]
    script = (root / "scripts" / "sweep1_30k.sbatch").read_text()
    for spec in SWEEP1_SPECS:
        assert f'"{spec.label}"' in script
    assert "#SBATCH --array=0-19" in script


def test_l1_dssim_accepts_flattened_full_image_and_backpropagates() -> None:
    prediction = torch.full((11 * 11, 3), 0.4, requires_grad=True)
    target = torch.full((11 * 11, 3), 0.6)
    loss = b0_reconstruction_loss(
        prediction,
        target,
        mode="l1_dssim",
        dssim_weight=0.2,
        image_size=(11, 11),
    )
    assert torch.isfinite(loss.total)
    assert torch.isfinite(loss.ssim)
    loss.total.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_opacity_reset_schedule_clamps_and_clears_adam_state() -> None:
    model = _model(2)
    optimizer = _optimizer(model)
    sum(parameter.square().sum() for parameter in model.parameters()).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.opacity_logits.fill_(torch.logit(torch.tensor(0.8)))
    model.opacity_logits.sum().backward()
    controller = _controller(model, optimizer)
    assert not controller.should_reset_opacity(2_999)
    assert controller.should_reset_opacity(3_000)
    assert not controller.should_reset_opacity(15_000)
    changed = controller.reset_opacity()
    assert changed == 2
    torch.testing.assert_close(model.opacities, torch.full((2, 1), 0.01))
    assert model.opacity_logits.grad is None
    state = optimizer.state[model.opacity_logits]
    assert torch.count_nonzero(state["exp_avg"]) == 0
    assert torch.count_nonzero(state["exp_avg_sq"]) == 0


def test_keep_adam_reset_preserves_moments() -> None:
    model = _model(1)
    optimizer = _optimizer(model)
    sum(parameter.square().sum() for parameter in model.parameters()).backward()
    optimizer.step()
    before = optimizer.state[model.opacity_logits]["exp_avg"].clone()
    controller = _controller(
        model, optimizer, opacity_reset_optimizer_state=False
    )
    controller.reset_opacity()
    torch.testing.assert_close(
        optimizer.state[model.opacity_logits]["exp_avg"], before
    )


def test_opacity_preserving_clone_keeps_combined_absorption() -> None:
    model = _model(1)
    with torch.no_grad():
        model.opacity_logits.fill_(torch.logit(torch.tensor(0.8)))
    before = model.opacities.item()
    controller = _controller(
        model, _optimizer(model), preserve_clone_opacity=True
    )
    controller.apply_plan(
        GaussianTopologyPlan(
            keep_indices=torch.tensor([0]),
            clone_indices=torch.tensor([0]),
            split_indices=torch.tensor([], dtype=torch.long),
            original_count=1,
        )
    )
    alpha = model.opacities.flatten()
    combined = 1.0 - torch.prod(1.0 - alpha)
    assert combined.item() == pytest.approx(before, abs=1.0e-6)


def test_opacity_preserving_split_keeps_combined_absorption() -> None:
    model = _model(1)
    with torch.no_grad():
        model.opacity_logits.fill_(torch.logit(torch.tensor(0.7)))
    before = model.opacities.item()
    controller = _controller(
        model, _optimizer(model), preserve_split_opacity=True
    )
    controller.apply_plan(
        GaussianTopologyPlan(
            keep_indices=torch.tensor([], dtype=torch.long),
            clone_indices=torch.tensor([], dtype=torch.long),
            split_indices=torch.tensor([0]),
            original_count=1,
        )
    )
    alpha = model.opacities.flatten()
    combined = 1.0 - torch.prod(1.0 - alpha)
    assert combined.item() == pytest.approx(before, abs=1.0e-6)


def test_replacement_clone_retains_parent_opacity() -> None:
    model = _model(1)
    with torch.no_grad():
        model.opacity_logits.fill_(torch.logit(torch.tensor(0.6)))
    before = model.opacities.item()
    controller = _controller(
        model, _optimizer(model), preserve_clone_opacity=True
    )
    controller.apply_plan(
        GaussianTopologyPlan(
            keep_indices=torch.tensor([], dtype=torch.long),
            clone_indices=torch.tensor([0]),
            split_indices=torch.tensor([], dtype=torch.long),
            original_count=1,
        )
    )
    assert model.opacities.item() == pytest.approx(before, abs=1.0e-6)
