from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict

from training import ExperimentConfig
from training.disentanglement_eviction_sweeps import (
    GAUSSIAN_EVICTION_SWEEP_SPECS,
    build_gaussian_eviction_config,
    gaussian_eviction_spec_payload,
)


EXPECTED_COUNTS = {
    "pruning_main": 36,
    "pruning_start": 8,
    "pruning_control": 2,
    "pruning_interval": 4,
    "binarization_pruning": 48,
    "binarization_early": 2,
}


def _get_nested(config, path: str):
    section, field = path.split(".", 1)
    return getattr(getattr(config, section), field)


def test_sweep_has_100_unique_scientific_configurations() -> None:
    specs = GAUSSIAN_EVICTION_SWEEP_SPECS
    assert len(specs) == 100
    assert Counter(spec.family for spec in specs) == EXPECTED_COUNTS
    assert len({spec.label for spec in specs}) == 100
    assert len({spec.overrides for spec in specs}) == 100
    assert sum(spec.family.startswith("pruning") for spec in specs) == 50
    assert sum(not spec.family.startswith("pruning") for spec in specs) == 50


def test_phase1_winner_is_exactly_reproduced_in_the_new_sweep() -> None:
    spec = GAUSSIAN_EVICTION_SWEEP_SPECS[12]
    assert spec.label == "prune_a0.1_s3k_r0k_u15k"
    overrides = dict(spec.overrides)
    assert overrides["gaussian_adaptation.min_opacity"] == 0.10
    assert overrides["gaussian_adaptation.opacity_prune_from"] == 3_000
    assert overrides["gaussian_adaptation.opacity_prune_ramp"] == 0
    assert overrides["gaussian_adaptation.opacity_prune_until"] == 15_000
    assert overrides["gaussian_adaptation.opacity_prune_interval"] == 250


def test_all_configs_validate_and_change_only_declared_fields() -> None:
    base = ExperimentConfig.load("configs/baseline_clean_joint.yaml")
    ignored_top_level = {"output_dir", "experiment_name", "wandb"}
    base_dict = asdict(base)
    for key in ignored_top_level:
        base_dict.pop(key)

    for index, spec in enumerate(GAUSSIAN_EVICTION_SWEEP_SPECS):
        config = build_gaussian_eviction_config(
            base, index, f"/tmp/evict{index}", f"evict{index}", "disabled"
        )
        config.validate()
        assert config.seed == 0
        assert config.renderer.gaussian_hit_buffer_size == 64
        assert config.dataset.train_target == "both"
        assert config.optimization.iterations == 30_000
        assert config.gaussian_adaptation.densify_until == 15_000
        assert config.diagnostics.save_per_primitive is False
        for path, expected in spec.overrides:
            assert _get_nested(config, path) == expected

        candidate = asdict(config)
        for key in ignored_top_level:
            candidate.pop(key)
        candidate["diagnostics"]["save_per_primitive"] = base_dict[
            "diagnostics"
        ]["save_per_primitive"]
        for path, _ in spec.overrides:
            section, field = path.split(".", 1)
            candidate[section][field] = base_dict[section][field]
        assert candidate == base_dict, spec.label


def test_payload_is_json_ready_and_builder_does_not_mutate_base() -> None:
    base = ExperimentConfig.load("configs/baseline_clean_joint.yaml")
    before = deepcopy(base)
    for index, spec in enumerate(GAUSSIAN_EVICTION_SWEEP_SPECS):
        payload = gaussian_eviction_spec_payload(index)
        assert payload["index"] == index
        assert payload["label"] == spec.label
        assert payload["family"] == spec.family
        assert payload["factors"] == dict(spec.factors)
        assert payload["overrides"] == dict(spec.overrides)
    build_gaussian_eviction_config(
        base, 99, "/tmp/output", "name", "disabled"
    )
    assert base == before
