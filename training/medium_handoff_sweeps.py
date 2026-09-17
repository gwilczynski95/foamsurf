from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from .config import ExperimentConfig
from .disentanglement_eviction_sweeps import (
    GAUSSIAN_EVICTION_SWEEP_SPECS,
)


@dataclass(frozen=True)
class MediumHandoffSpec:
    label: str
    family: str
    base_index: int
    factors: tuple[tuple[str, Any], ...]
    overrides: tuple[tuple[str, Any], ...]


BASES = ((30, "b30"), (41, "b41"), (61, "b61"))


def _spec(label, family, base_index, factors, overrides):
    return MediumHandoffSpec(
        label=label,
        family=family,
        base_index=base_index,
        factors=tuple(sorted(factors.items())),
        overrides=tuple(sorted(overrides.items())),
    )


def _prefix(mode: str, iterations: int, reset: bool = True):
    return {
        "medium_handoff.prefix_mode": mode,
        "medium_handoff.prefix_iterations": iterations,
        "medium_handoff.prefix_reset_medium_optimizer": reset,
    }


def _freeze(start: int, duration: int, *, sites=1.0, density=1.0, color=1.0):
    return {
        "medium_handoff.gaussian_freeze_start": start,
        "medium_handoff.gaussian_freeze_duration": duration,
        "medium_handoff.medium_sites_lr_multiplier": sites,
        "medium_handoff.medium_density_lr_multiplier": density,
        "medium_handoff.medium_color_lr_multiplier": color,
    }


_specs = []

# A (0-2): exact reruns of the three selected eviction bases.
for base_index, base_name in BASES:
    _specs.append(_spec(
        f"{base_name}_control", "base_control", base_index,
        {"base": base_index}, {},
    ))

# B (3-29): medium-only prefix, fixed population; reset and retained Adam.
for base_index, base_name in BASES:
    for length in (250, 500, 1_000, 2_000, 3_000, 5_000):
        _specs.append(_spec(
            f"{base_name}_mw{length}_reset", "medium_prefix_reset",
            base_index,
            {"base": base_index, "prefix": length, "optimizer": "reset"},
            _prefix("medium_only", length, True),
        ))
for base_index, base_name in BASES:
    for length in (1_000, 2_000, 3_000):
        _specs.append(_spec(
            f"{base_name}_mw{length}_retain", "medium_prefix_retain",
            base_index,
            {"base": base_index, "prefix": length, "optimizer": "retain"},
            _prefix("medium_only", length, False),
        ))

# C (30-65): freeze timing/duration without an LR boost.
for base_index, base_name in BASES:
    for start in (15_000, 17_500, 20_000):
        for duration in (500, 1_000, 2_000, 4_000):
            _specs.append(_spec(
                f"{base_name}_f{start}_{duration}_x1", "freeze_only",
                base_index,
                {"base": base_index, "freeze_start": start,
                 "freeze_duration": duration, "boost": 1.0},
                _freeze(start, duration),
            ))

# D (66-119): freeze plus density+color boost; sites stay on normal LR.
for base_index, base_name in BASES:
    for start in (15_000, 20_000):
        for duration in (1_000, 2_000, 4_000):
            for multiplier in (2.0, 5.0, 10.0):
                _specs.append(_spec(
                    f"{base_name}_f{start}_{duration}_dcx{multiplier:g}",
                    "freeze_density_color_boost", base_index,
                    {"base": base_index, "freeze_start": start,
                     "freeze_duration": duration, "boost": multiplier,
                     "boost_scope": "density_color"},
                    _freeze(
                        start, duration, density=multiplier, color=multiplier
                    ),
                ))

# E (120-125): component attribution at the central freeze setting.
for base_index, base_name in BASES:
    for scope in ("density", "color"):
        values = {"density": 5.0, "color": 1.0}
        if scope == "color":
            values = {"density": 1.0, "color": 5.0}
        _specs.append(_spec(
            f"{base_name}_f15000_2000_{scope}x5", "component_boost",
            base_index,
            {"base": base_index, "freeze_start": 15_000,
             "freeze_duration": 2_000, "boost": 5.0,
             "boost_scope": scope},
            _freeze(15_000, 2_000, **values),
        ))

# F (126-143): promising mechanisms composed, but still factorizable.
for base_index, base_name in BASES:
    for warmup in (1_000, 2_000, 3_000):
        for duration in (1_000, 2_000):
            overrides = _prefix("medium_only", warmup, True)
            overrides.update(_freeze(
                15_000, duration, density=5.0, color=5.0
            ))
            _specs.append(_spec(
                f"{base_name}_mw{warmup}_f15000_{duration}_dcx5",
                "prefix_freeze_boost", base_index,
                {"base": base_index, "prefix": warmup,
                 "optimizer": "reset", "freeze_start": 15_000,
                 "freeze_duration": duration, "boost": 5.0,
                 "boost_scope": "density_color"},
                overrides,
            ))

# G (144-149): equal-extra-compute controls, both representations train.
for base_index, base_name in BASES:
    for length in (1_000, 3_000):
        _specs.append(_spec(
            f"{base_name}_jw{length}_reset", "joint_prefix_control",
            base_index,
            {"base": base_index, "prefix": length,
             "optimizer": "reset", "prefix_mode": "joint"},
            _prefix("joint", length, True),
        ))

MEDIUM_HANDOFF_SWEEP_SPECS = tuple(_specs)
if len(MEDIUM_HANDOFF_SWEEP_SPECS) != 150:
    raise RuntimeError("medium-handoff matrix must contain 150 runs")
if len({spec.label for spec in MEDIUM_HANDOFF_SWEEP_SPECS}) != 150:
    raise RuntimeError("medium-handoff labels must be unique")


def _set_nested(config: ExperimentConfig, path: str, value: Any) -> None:
    section, field = path.split(".", 1)
    target = getattr(config, section)
    if not hasattr(target, field):
        raise ValueError(f"unknown override path: {path}")
    setattr(target, field, value)


def build_medium_handoff_config(
    base: ExperimentConfig,
    index: int,
    output_dir: str,
    wandb_name: str,
    wandb_mode: str = "online",
) -> ExperimentConfig:
    if not 0 <= index < len(MEDIUM_HANDOFF_SWEEP_SPECS):
        raise IndexError("medium-handoff sweep index out of range")
    config = deepcopy(base)
    spec = MEDIUM_HANDOFF_SWEEP_SPECS[index]
    base_spec = GAUSSIAN_EVICTION_SWEEP_SPECS[spec.base_index]
    for path, value in base_spec.overrides:
        _set_nested(config, path, value)
    for path, value in spec.overrides:
        _set_nested(config, path, value)

    config.seed = 0
    config.output_dir = output_dir
    config.experiment_name = wandb_name
    config.wandb.name = wandb_name
    config.wandb.mode = wandb_mode
    config.wandb.project = "foamsurf_medium_handoff"
    config.wandb.tags = [
        "disentanglement", "medium-handoff", "phase3", "k64",
        f"base-{spec.base_index}", spec.family, spec.label,
    ]
    config.diagnostics.save_per_primitive = False

    if config.renderer.gaussian_hit_buffer_size != 64:
        raise ValueError("medium-handoff sweep requires K=64")
    if config.dataset.train_target != "both":
        raise ValueError("sweep must train only on lego_both")
    if config.optimization.iterations != 30_000:
        raise ValueError("main schedule must remain exactly 30k")
    if config.gaussian_adaptation.densify_until != 15_000:
        raise ValueError("sweep assumes Gaussian densification ends at 15k")
    config.validate()
    return config


def medium_handoff_spec_payload(index: int) -> dict[str, Any]:
    spec = MEDIUM_HANDOFF_SWEEP_SPECS[index]
    base_spec = GAUSSIAN_EVICTION_SWEEP_SPECS[spec.base_index]
    return {
        "index": index,
        "label": spec.label,
        "family": spec.family,
        "base_index": spec.base_index,
        "base_label": base_spec.label,
        "factors": dict(spec.factors),
        "base_overrides": dict(base_spec.overrides),
        "handoff_overrides": dict(spec.overrides),
        "main_iterations": 30_000,
        "extra_prefix_iterations": dict(spec.factors).get("prefix", 0),
    }
