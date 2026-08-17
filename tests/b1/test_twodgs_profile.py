from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from foamsurf_model.gaussian_adaptation import GaussianAdaptationController
from foamsurf_model.gaussians import SurfaceGaussianModel
from training.config import ExperimentConfig, ModelConfig
from training.twodgs_profile import build_twodgs_profile


def _model() -> SurfaceGaussianModel:
    points = np.array([[0.0, 0.0, 2.0]], dtype=np.float32)
    colors = np.full((1, 3), 0.5, dtype=np.float32)
    model = SurfaceGaussianModel(
        points,
        colors,
        ModelConfig(appearance="rgb", gaussian_initial_points=1),
        "cpu",
    )
    with torch.no_grad():
        model.quaternions[:] = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        model.log_scales.fill_(np.log(0.1))
    return model


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


def test_twodgs_profile_matches_reference_population_defaults(tmp_path) -> None:
    base = ExperimentConfig()
    base.dataset.source_path = str(tmp_path)
    config = build_twodgs_profile(
        base,
        tmp_path / "run",
        "test",
        wandb_mode="disabled",
    )
    assert config.optimization.full_image_batch
    assert config.optimization.cameras_without_replacement
    assert config.optimization.reconstruction_loss == "l1_dssim"
    assert config.optimization.dssim_weight == 0.2
    assert config.optimization.gaussian_opacity_lr == 0.05
    adaptation = config.gaussian_adaptation
    assert adaptation.max_points is None
    assert adaptation.gradient_threshold == 2.0e-4
    assert adaptation.min_opacity == 0.05
    assert adaptation.opacity_reset_interval == 3_000
    assert adaptation.opacity_reset_value == 0.01
    assert adaptation.opacity_reset_at_densify_from
    assert not adaptation.screen_pruning_enabled
    assert adaptation.max_world_scale_fraction == 0.10


def test_twodgs_white_background_reset_schedule(tmp_path) -> None:
    base = ExperimentConfig()
    base.dataset.source_path = str(tmp_path)
    config = build_twodgs_profile(
        base,
        tmp_path / "run",
        "test",
        wandb_mode="disabled",
    )
    model = _model()
    controller = GaussianAdaptationController(
        model, _optimizer(model), config.gaussian_adaptation, scene_extent=10.0
    )

    assert controller.should_reset_opacity(500)
    assert controller.should_reset_opacity(3_000)
    assert not controller.should_reset_opacity(501)


def test_twodgs_profile_does_not_apply_screen_pruning(tmp_path) -> None:
    base = ExperimentConfig()
    base.dataset.source_path = str(tmp_path)
    config = build_twodgs_profile(
        base,
        tmp_path / "run",
        "test",
        wandb_mode="disabled",
    )
    model = _model()
    controller = GaussianAdaptationController(
        model,
        _optimizer(model),
        config.gaussian_adaptation,
        scene_extent=10.0,
    )
    controller.gradient_denom.fill_(1.0)
    controller.max_radii2d.fill_(1_000.0)

    plan = controller.propose_plan(iteration=3_100)

    assert plan.prune_count == 0
    assert controller.last_plan_statistics["screen_prune"] == 0


def test_screen_pruning_remains_available_for_explicit_ablations() -> None:
    config = ExperimentConfig().gaussian_adaptation
    config.enabled = True
    config.screen_pruning_enabled = True
    config.size_prune_from = 3_000
    config.max_screen_size = 20.0
    model = _model()
    controller = GaussianAdaptationController(
        model, _optimizer(model), config, scene_extent=10.0
    )
    controller.gradient_denom.fill_(1.0)
    controller.max_radii2d.fill_(21.0)

    plan = controller.propose_plan(iteration=3_100)

    assert plan.prune_count == 1
    assert controller.last_plan_statistics["screen_prune"] == 1


def test_sbatch_uses_separate_wandb_project_and_30k_default() -> None:
    root = Path(__file__).resolve().parents[2]
    script = (root / "scripts" / "train_b1_2dgs_baseline.sbatch").read_text()
    assert "foamsurf_2dgs_baseline" in script
    assert "ITERATIONS=" in script
    assert "30000" in script
    assert "run_2dgs_baseline.py" in script
