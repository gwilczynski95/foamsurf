from __future__ import annotations

from training import ExperimentConfig
from training.voronoi_capacity_sweeps import (
    VORONOI_CAPACITY_SWEEP_SPECS,
    build_voronoi_capacity_config,
)


def _project_population(config: ExperimentConfig) -> int:
    count = config.model.medium_points
    adaptation = config.voronoi_adaptation
    for iteration in range(
        adaptation.densify_from,
        adaptation.densify_until,
        adaptation.interval,
    ):
        del iteration
        non_guard = count - config.model.medium_guard_points
        requested = int(adaptation.densify_fraction * non_guard)
        if adaptation.max_points is not None:
            requested = min(requested, adaptation.max_points - count)
        count += max(0, requested)
    return count


def test_capacity_sweep_has_16_valid_reachable_configs() -> None:
    base = ExperimentConfig.load("configs/stage2_voronoi_capacity.yaml")
    assert len(VORONOI_CAPACITY_SWEEP_SPECS) == 16
    assert len({spec.label for spec in VORONOI_CAPACITY_SWEEP_SPECS}) == 16
    assert {spec.max_points for spec in VORONOI_CAPACITY_SWEEP_SPECS} == {
        8_192, 16_384, 32_768, 65_536,
    }
    assert {spec.seed for spec in VORONOI_CAPACITY_SWEEP_SPECS} == {42, 1337}
    assert {
        spec.child_jitter_fraction for spec in VORONOI_CAPACITY_SWEEP_SPECS
    } == {0.025, 0.10}

    for index, spec in enumerate(VORONOI_CAPACITY_SWEEP_SPECS):
        config = build_voronoi_capacity_config(
            base, index, f"outputs/test-{index}", f"test-{index}", "disabled"
        )
        assert config.optimization.reconstruction_loss == "l1_dssim"
        assert config.optimization.dssim_weight == 0.2
        assert config.model.medium_initial_density == 0.005
        assert config.optimization.medium_color_lr == 0.0025
        assert config.optimization.medium_lr_decay_until == spec.iterations
        assert config.voronoi_adaptation.max_points == spec.max_points
        assert _project_population(config) == spec.max_points
