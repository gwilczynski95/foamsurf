from __future__ import annotations

from collections import Counter

import pytest
import torch

from training import ExperimentConfig
from training.config import MediumHandoffConfig, OptimizationConfig
from training.handoff import (
    apply_medium_lr_multipliers,
    gaussian_is_frozen,
    handoff_boundary_iterations,
    handoff_freeze_active,
    medium_lr_multipliers,
)
from training.medium_handoff_sweeps import (
    MEDIUM_HANDOFF_SWEEP_SPECS,
    build_medium_handoff_config,
    medium_handoff_spec_payload,
)


EXPECTED = {
    "base_control": 3,
    "medium_prefix_reset": 18,
    "medium_prefix_retain": 9,
    "freeze_only": 36,
    "freeze_density_color_boost": 54,
    "component_boost": 6,
    "prefix_freeze_boost": 18,
    "joint_prefix_control": 6,
}


def test_handoff_matrix_has_exactly_150_unique_runs() -> None:
    specs = MEDIUM_HANDOFF_SWEEP_SPECS
    assert len(specs) == 150
    assert len({spec.label for spec in specs}) == 150
    assert Counter(spec.family for spec in specs) == EXPECTED
    assert [spec.family for spec in specs[:3]] == ["base_control"] * 3
    assert specs[29].family == "medium_prefix_retain"
    assert specs[30].family == "freeze_only"
    assert specs[65].family == "freeze_only"
    assert specs[66].family == "freeze_density_color_boost"
    assert specs[119].family == "freeze_density_color_boost"
    assert specs[120].family == "component_boost"
    assert specs[126].family == "prefix_freeze_boost"
    assert specs[144].family == "joint_prefix_control"


def test_all_handoff_configs_validate_and_reproduce_selected_bases() -> None:
    base = ExperimentConfig.load("configs/baseline_clean_joint.yaml")
    for index, spec in enumerate(MEDIUM_HANDOFF_SWEEP_SPECS):
        config = build_medium_handoff_config(
            base, index, f"/tmp/handoff-{index}", f"handoff-{index}",
            "disabled",
        )
        config.validate()
        assert config.optimization.iterations == 30_000
        assert config.renderer.gaussian_hit_buffer_size == 64
        assert config.seed == 0
        assert medium_handoff_spec_payload(index)["base_index"] in {30, 41, 61}
    assert build_medium_handoff_config(
        base, 0, "/tmp/a", "a", "disabled"
    ).gaussian_adaptation.min_opacity == pytest.approx(0.20)
    assert build_medium_handoff_config(
        base, 1, "/tmp/b", "b", "disabled"
    ).gaussian_adaptation.min_opacity == pytest.approx(0.15)
    base61 = build_medium_handoff_config(
        base, 2, "/tmp/c", "c", "disabled"
    )
    assert base61.gaussian_adaptation.min_opacity == pytest.approx(0.10)
    assert base61.disentanglement.opacity_binarization_weight == pytest.approx(1e-4)


def test_freeze_boundaries_and_component_lr_boosts_are_exact() -> None:
    handoff = MediumHandoffConfig(
        gaussian_freeze_start=15_000,
        gaussian_freeze_duration=1_000,
        medium_density_lr_multiplier=5.0,
        medium_color_lr_multiplier=2.0,
    )
    assert not handoff_freeze_active(handoff, 14_999)
    assert handoff_freeze_active(handoff, 15_000)
    assert handoff_freeze_active(handoff, 15_999)
    assert not handoff_freeze_active(handoff, 16_000)
    assert handoff_boundary_iterations(handoff) == {
        14_999, 15_000, 15_999, 16_000
    }
    assert not gaussian_is_frozen(OptimizationConfig(), handoff, 14_999)
    assert gaussian_is_frozen(OptimizationConfig(), handoff, 15_000)
    assert medium_lr_multipliers(handoff, 15_500) == {
        "medium_sites": 1.0,
        "medium_density": 5.0,
        "medium_color": 2.0,
    }
    assert all(
        value == 1.0
        for value in medium_lr_multipliers(handoff, 16_000).values()
    )


def test_lr_multiplier_changes_optimizer_and_reported_rates() -> None:
    parameters = [
        torch.nn.Parameter(torch.tensor([float(index)]))
        for index in range(3)
    ]
    optimizer = torch.optim.Adam([
        {"params": [parameters[0]], "name": "medium_sites", "lr": 1e-4},
        {"params": [parameters[1]], "name": "medium_density", "lr": 1e-2},
        {"params": [parameters[2]], "name": "medium_color", "lr": 2e-3},
    ])
    rates = {"medium_sites": 1e-4, "medium_density": 1e-2,
             "medium_color": 2e-3}
    adjusted = apply_medium_lr_multipliers(
        optimizer, rates,
        {"medium_sites": 1.0, "medium_density": 5.0,
         "medium_color": 2.0},
    )
    assert adjusted == pytest.approx({
        "medium_sites": 1e-4, "medium_density": 5e-2,
        "medium_color": 4e-3,
    })
    assert {group["name"]: group["lr"] for group in optimizer.param_groups} == pytest.approx(adjusted)


def test_handoff_yaml_roundtrip_and_invalid_windows(tmp_path) -> None:
    config = ExperimentConfig.load("configs/baseline_clean_joint.yaml")
    config.medium_handoff = MediumHandoffConfig(
        prefix_mode="medium_only", prefix_iterations=2_000,
        prefix_reset_medium_optimizer=False,
        gaussian_freeze_start=15_000, gaussian_freeze_duration=2_000,
        medium_density_lr_multiplier=5.0,
        medium_color_lr_multiplier=5.0,
    )
    path = tmp_path / "config.yaml"
    config.save(path)
    assert ExperimentConfig.load(path) == config

    config.medium_handoff = MediumHandoffConfig(
        prefix_mode="none", prefix_iterations=1
    )
    with pytest.raises(ValueError, match="prefix_mode none"):
        config.validate()
