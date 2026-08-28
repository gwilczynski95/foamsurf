from __future__ import annotations

from dataclasses import asdict

import pytest

from training import ExperimentConfig


def test_clean_joint_combines_the_two_qualified_recipes_without_oracle_training() -> None:
    joint = ExperimentConfig.load("configs/baseline_clean_joint.yaml")
    gaussian = ExperimentConfig.load("configs/baseline_best_2d_gaussians.yaml")
    medium = ExperimentConfig.load("configs/baseline_best_voronoi.yaml")

    assert joint.representation_mode == "hybrid"
    assert joint.dataset.train_target == "both"
    assert joint.dataset.source_path.endswith("/lego_both")
    assert joint.background == [0.0, 0.0, 0.0]
    assert asdict(joint.renderer) == asdict(gaussian.renderer)

    assert joint.model.gaussian_initial_points == 100_000
    assert joint.model.medium_points == 2_048
    assert joint.model.medium_guard_points == 8
    assert joint.model.medium_initial_density == pytest.approx(0.005)

    assert joint.optimization.reconstruction_loss == "l1_dssim"
    assert joint.optimization.dssim_weight == pytest.approx(0.2)
    assert joint.optimization.full_image_batch is True
    assert joint.optimization.cameras_without_replacement is True
    assert joint.optimization.gaussian_means_lr == pytest.approx(
        gaussian.optimization.gaussian_means_lr
    )
    assert joint.optimization.gaussian_scale_regularization == pytest.approx(
        gaussian.optimization.gaussian_scale_regularization
    )
    assert joint.optimization.medium_sites_lr == pytest.approx(
        medium.optimization.medium_sites_lr
    )
    assert joint.optimization.medium_density_lr_final == pytest.approx(
        medium.optimization.medium_density_lr_final
    )

    assert joint.gaussian_adaptation.enabled is True
    assert joint.gaussian_adaptation == gaussian.gaussian_adaptation
    assert joint.voronoi_adaptation.enabled is True
    assert joint.voronoi_adaptation == medium.voronoi_adaptation
    assert joint.diagnostics.enabled is True
