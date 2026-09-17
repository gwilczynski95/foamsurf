from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict

from training import ExperimentConfig
from training.disentanglement_sweeps import (
    DISENTANGLEMENT_SWEEP_SPECS,
    build_disentanglement_config,
    disentanglement_spec_payload,
)


EXPECTED_COUNTS = {
    "control": 1,
    "opacity_binarization": 40,
    "graph_tv": 32,
    "density_cap": 12,
    "optical_thickness": 4,
    "opaque_if_used": 4,
    "opacity_pruning": 4,
}


def _get_nested(config, path: str):
    section, field = path.split(".", 1)
    return getattr(getattr(config, section), field)


def test_phase1_has_exactly_one_control_and_96_single_mechanism_runs() -> None:
    assert len(DISENTANGLEMENT_SWEEP_SPECS) == 97
    assert Counter(spec.family for spec in DISENTANGLEMENT_SWEEP_SPECS) == (
        EXPECTED_COUNTS
    )
    assert len({spec.label for spec in DISENTANGLEMENT_SWEEP_SPECS}) == 97
    assert DISENTANGLEMENT_SWEEP_SPECS[0].label == "control"
    assert DISENTANGLEMENT_SWEEP_SPECS[0].overrides == ()


def test_every_generated_config_is_valid_pinned_to_k64_and_isolated() -> None:
    base = ExperimentConfig.load("configs/baseline_clean_joint.yaml")
    ignored_top_level = {"output_dir", "experiment_name", "wandb"}
    base_dict = asdict(base)
    for key in ignored_top_level:
        base_dict.pop(key)

    for index, spec in enumerate(DISENTANGLEMENT_SWEEP_SPECS):
        config = build_disentanglement_config(
            base, index, f"/tmp/dis{index}", f"dis{index}", "disabled"
        )
        config.validate()
        assert config.seed == 0
        assert config.renderer.gaussian_hit_buffer_size == 64
        assert config.dataset.train_target == "both"
        assert config.dataset.source_path.endswith("/lego_both")
        assert config.optimization.reconstruction_loss == "l1_dssim"
        assert config.optimization.dssim_weight == 0.2
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


def test_spec_payload_is_json_ready_and_complete() -> None:
    for index, spec in enumerate(DISENTANGLEMENT_SWEEP_SPECS):
        payload = disentanglement_spec_payload(index)
        assert payload["index"] == index
        assert payload["label"] == spec.label
        assert payload["family"] == spec.family
        assert payload["overrides"] == dict(spec.overrides)


def test_builder_does_not_mutate_base_config() -> None:
    base = ExperimentConfig.load("configs/baseline_clean_joint.yaml")
    before = deepcopy(base)
    build_disentanglement_config(
        base, 42, "/tmp/output", "name", "disabled"
    )
    assert base == before
