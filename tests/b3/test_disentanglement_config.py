from __future__ import annotations

from copy import deepcopy

import pytest

from training import ExperimentConfig


def test_baseline_defaults_disable_every_disentanglement_term() -> None:
    config = ExperimentConfig.load("configs/baseline_clean_joint.yaml")
    values = config.disentanglement
    assert values.opacity_binarization_weight == 0.0
    assert values.opaque_if_used_weight == 0.0
    assert values.graph_tv_weight == 0.0
    assert values.density_cap_weight == 0.0
    assert values.optical_thickness_weight == 0.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("opacity_binarization_weight", -1.0),
        ("opacity_binarization_end_ramp", -1),
        ("opacity_binarization_end_ramp_shape", "bad"),
        ("opacity_binarization_maintenance_scale", 1.1),
        ("graph_tv_start", -1),
        ("graph_tv_ramp", -1),
        ("graph_tv_ramp_shape", "bad"),
        ("graph_tv_mode", "bad"),
        ("graph_tv_huber_delta", 0.0),
        ("density_cap_policy", "bad"),
        ("density_cap_fixed_value", 0.0),
        ("density_cap_quantile", 1.0),
        ("density_cap_multiplier", 0.0),
        ("density_cap_minimum", 0.0),
        ("density_cap_update_interval", 0),
        ("density_cap_ema_decay", 1.0),
        ("density_cap_warmup_until", -1),
        ("epsilon", 0.0),
    ],
)
def test_invalid_disentanglement_configuration_is_rejected(
    field: str, value
) -> None:
    config = ExperimentConfig.load("configs/baseline_clean_joint.yaml")
    setattr(config.disentanglement, field, value)
    with pytest.raises(ValueError):
        config.validate()


def test_losses_cannot_target_a_missing_representation() -> None:
    hybrid = ExperimentConfig.load("configs/baseline_clean_joint.yaml")

    gaussian_only = deepcopy(hybrid)
    gaussian_only.representation_mode = "gaussian_only"
    gaussian_only.dataset.train_target = "lego"
    gaussian_only.disentanglement.graph_tv_weight = 1.0
    with pytest.raises(ValueError, match="require Voronoi"):
        gaussian_only.validate()

    voronoi_only = deepcopy(hybrid)
    voronoi_only.representation_mode = "voronoi_only"
    voronoi_only.dataset.train_target = "clouds"
    voronoi_only.disentanglement.opacity_binarization_weight = 1.0
    with pytest.raises(ValueError, match="require Gaussians"):
        voronoi_only.validate()


def test_opacity_binarization_end_schedule_is_validated() -> None:
    config = ExperimentConfig.load("configs/baseline_clean_joint.yaml")
    config.disentanglement.opacity_binarization_start = 5_000
    config.disentanglement.opacity_binarization_ramp = 5_000
    config.disentanglement.opacity_binarization_end = 9_999
    with pytest.raises(ValueError, match="overlap"):
        config.validate()

    config.disentanglement.opacity_binarization_end = 29_000
    config.disentanglement.opacity_binarization_end_ramp = 1_001
    with pytest.raises(ValueError, match="exceeds training"):
        config.validate()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("opacity_prune_ramp", -1),
        ("opacity_prune_ramp_shape", "bad"),
        ("opacity_prune_interval", 0),
        ("opacity_prune_until", 30_001),
        ("min_points", 0),
        ("max_opacity_prune_fraction", 0.0),
        ("max_opacity_prune_fraction", 1.1),
        ("min_opacity_initial", -0.1),
        ("min_opacity_initial", 1.0),
    ],
)
def test_invalid_opacity_pruning_schedule_is_rejected(field, value) -> None:
    config = ExperimentConfig.load("configs/baseline_clean_joint.yaml")
    setattr(config.gaussian_adaptation, field, value)
    with pytest.raises(ValueError):
        config.validate()
