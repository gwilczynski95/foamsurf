from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from .config import ExperimentConfig


@dataclass(frozen=True)
class GaussianEvictionSpec:
    """One deterministic Phase 2 Gaussian-eviction experiment."""

    label: str
    family: str
    factors: tuple[tuple[str, Any], ...]
    overrides: tuple[tuple[str, Any], ...]


def _spec(
    label: str,
    family: str,
    factors: dict[str, Any],
    overrides: dict[str, Any],
) -> GaussianEvictionSpec:
    return GaussianEvictionSpec(
        label=label,
        family=family,
        factors=tuple(sorted(factors.items())),
        overrides=tuple(sorted(overrides.items())),
    )


def _pruning_overrides(
    target: float,
    start: int,
    ramp: int,
    until: int,
    interval: int = 250,
) -> dict[str, Any]:
    return {
        "gaussian_adaptation.min_opacity": target,
        "gaussian_adaptation.min_opacity_initial": (
            target if ramp == 0 else 0.01
        ),
        "gaussian_adaptation.opacity_prune_from": start,
        "gaussian_adaptation.opacity_prune_ramp": ramp,
        "gaussian_adaptation.opacity_prune_ramp_shape": "linear",
        "gaussian_adaptation.opacity_prune_until": until,
        "gaussian_adaptation.opacity_prune_interval": interval,
    }


_SPECS: list[GaussianEvictionSpec] = []

# A1: factorial study of threshold, threshold ramp and pruning duration.
for target in (0.05, 0.075, 0.10, 0.125, 0.15, 0.20):
    for ramp in (0, 2_000, 5_000):
        for until in (15_000, 30_000):
            factors = {
                "prune_target": target,
                "prune_start": 3_000,
                "prune_ramp": ramp,
                "prune_until": until,
                "prune_interval": 250,
            }
            _SPECS.append(
                _spec(
                    "prune_"
                    f"a{target:g}_s3k_r{ramp // 1000}k_u{until // 1000}k",
                    "pruning_main",
                    factors,
                    _pruning_overrides(target, 3_000, ramp, until),
                )
            )

# A2: sensitivity to starting the threshold ramp before or after 3k.
for target in (0.10, 0.15):
    for start in (1_000, 5_000):
        for ramp in (2_000, 5_000):
            factors = {
                "prune_target": target,
                "prune_start": start,
                "prune_ramp": ramp,
                "prune_until": 30_000,
                "prune_interval": 250,
            }
            _SPECS.append(
                _spec(
                    "prune_start_"
                    f"a{target:g}_s{start // 1000}k_r{ramp // 1000}k_u30k",
                    "pruning_start",
                    factors,
                    _pruning_overrides(target, start, ramp, 30_000),
                )
            )

# A3: controls at the original 0.01 threshold, with short and long pruning.
for until in (15_000, 30_000):
    factors = {
        "prune_target": 0.01,
        "prune_start": 3_000,
        "prune_ramp": 0,
        "prune_until": until,
        "prune_interval": 250,
    }
    _SPECS.append(
        _spec(
            f"prune_control_a0.01_s3k_r0k_u{until // 1000}k",
            "pruning_control",
            factors,
            _pruning_overrides(0.01, 3_000, 0, until),
        )
    )

# A4: how often pruning is allowed after densification ends.
for target in (0.10, 0.15):
    for interval in (500, 1_000):
        factors = {
            "prune_target": target,
            "prune_start": 3_000,
            "prune_ramp": 2_000,
            "prune_until": 30_000,
            "prune_interval": interval,
        }
        _SPECS.append(
            _spec(
                f"prune_interval_a{target:g}_s3k_r2k_u30k_i{interval}",
                "pruning_interval",
                factors,
                _pruning_overrides(
                    target, 3_000, 2_000, 30_000, interval
                ),
            )
        )

if len(_SPECS) != 50:
    raise RuntimeError("Phase 2 pruning block must contain exactly 50 runs")


_BIN_POLICIES = (
    ("stop15_prune15", 15_000, 0.0, 15_000),
    ("stop15_prune30", 15_000, 0.0, 30_000),
    ("maint10_prune30", 15_000, 0.1, 30_000),
    ("full30_prune30", None, 1.0, 30_000),
)

# B1: binarization strength/start crossed with four late-training policies.
for weight in (1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3):
    for start in (1_000, 3_000, 5_000):
        for policy, end, maintenance, prune_until in _BIN_POLICIES:
            factors = {
                "bin_weight": weight,
                "bin_start": start,
                "bin_ramp": 3_000,
                "bin_policy": policy,
                "prune_target": 0.10,
                "prune_until": prune_until,
            }
            overrides = _pruning_overrides(
                0.10, 3_000, 0, prune_until
            )
            overrides.update(
                {
                    "disentanglement.opacity_binarization_weight": weight,
                    "disentanglement.opacity_binarization_start": start,
                    "disentanglement.opacity_binarization_ramp": 3_000,
                    "disentanglement.opacity_binarization_ramp_shape": "cosine",
                    "disentanglement.opacity_binarization_end": end,
                    "disentanglement.opacity_binarization_end_ramp": 0,
                    "disentanglement."
                    "opacity_binarization_maintenance_scale": maintenance,
                }
            )
            _SPECS.append(
                _spec(
                    f"bin_w{weight:g}_s{start // 1000}k_r3k_{policy}",
                    "binarization_pruning",
                    factors,
                    overrides,
                )
            )

# B2: two early, slow binarization probes outside the B1 factorial grid.
for weight in (3.0e-4, 1.0e-3):
    factors = {
        "bin_weight": weight,
        "bin_start": 0,
        "bin_ramp": 5_000,
        "bin_policy": "stop15_prune30",
        "prune_target": 0.10,
        "prune_until": 30_000,
    }
    overrides = _pruning_overrides(0.10, 3_000, 0, 30_000)
    overrides.update(
        {
            "disentanglement.opacity_binarization_weight": weight,
            "disentanglement.opacity_binarization_start": 0,
            "disentanglement.opacity_binarization_ramp": 5_000,
            "disentanglement.opacity_binarization_ramp_shape": "cosine",
            "disentanglement.opacity_binarization_end": 15_000,
            "disentanglement.opacity_binarization_end_ramp": 0,
            "disentanglement.opacity_binarization_maintenance_scale": 0.0,
        }
    )
    _SPECS.append(
        _spec(
            f"bin_early_w{weight:g}_s0_r5k_stop15_prune30",
            "binarization_early",
            factors,
            overrides,
        )
    )


GAUSSIAN_EVICTION_SWEEP_SPECS = tuple(_SPECS)

if len(GAUSSIAN_EVICTION_SWEEP_SPECS) != 100:
    raise RuntimeError("Phase 2 matrix must contain exactly 100 runs")
if len({spec.label for spec in GAUSSIAN_EVICTION_SWEEP_SPECS}) != 100:
    raise RuntimeError("Phase 2 labels must be unique")
if len({spec.overrides for spec in GAUSSIAN_EVICTION_SWEEP_SPECS}) != 100:
    raise RuntimeError("Phase 2 scientific configurations must be unique")


def _set_nested(config: ExperimentConfig, path: str, value: Any) -> None:
    section, field = path.split(".", 1)
    target = getattr(config, section)
    if not hasattr(target, field):
        raise ValueError(f"unknown override path: {path}")
    setattr(target, field, value)


def build_gaussian_eviction_config(
    base: ExperimentConfig,
    index: int,
    output_dir: str,
    wandb_name: str,
    wandb_mode: str = "online",
) -> ExperimentConfig:
    if not 0 <= index < len(GAUSSIAN_EVICTION_SWEEP_SPECS):
        raise IndexError("Gaussian-eviction sweep index out of range")
    config = deepcopy(base)
    spec = GAUSSIAN_EVICTION_SWEEP_SPECS[index]
    for path, value in spec.overrides:
        _set_nested(config, path, value)

    config.seed = 0
    config.output_dir = output_dir
    config.experiment_name = wandb_name
    config.wandb.name = wandb_name
    config.wandb.mode = wandb_mode
    config.wandb.project = "foamsurf_disentanglement_eviction"
    config.wandb.tags = [
        "disentanglement",
        "gaussian-eviction",
        "phase2",
        "k64",
        spec.family,
        spec.label,
    ]
    config.diagnostics.save_per_primitive = False

    if config.renderer.gaussian_hit_buffer_size != 64:
        raise ValueError("Gaussian-eviction sweep requires K=64")
    if config.dataset.train_target != "both":
        raise ValueError("Sweep must train only on lego_both")
    if config.optimization.reconstruction_loss != "l1_dssim":
        raise ValueError("Sweep requires qualified L1+DSSIM")
    if config.optimization.iterations != 30_000:
        raise ValueError("Sweep schedules assume exactly 30k iterations")
    if config.gaussian_adaptation.densify_until != 15_000:
        raise ValueError("Sweep assumes densification ends at 15k")
    config.validate()
    return config


def gaussian_eviction_spec_payload(index: int) -> dict[str, Any]:
    spec = GAUSSIAN_EVICTION_SWEEP_SPECS[index]
    return {
        "index": index,
        "label": spec.label,
        "family": spec.family,
        "factors": dict(spec.factors),
        "overrides": dict(spec.overrides),
    }
