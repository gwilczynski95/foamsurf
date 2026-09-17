from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from .config import ExperimentConfig


@dataclass(frozen=True)
class DisentanglementSpec:
    label: str
    family: str
    overrides: tuple[tuple[str, Any], ...] = ()


def _spec(
    label: str, family: str, **overrides: Any
) -> DisentanglementSpec:
    return DisentanglementSpec(
        label=label,
        family=family,
        overrides=tuple(sorted(overrides.items())),
    )


_OPACITY_WEIGHTS = (3.0e-4, 1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2)
_OPACITY_SCHEDULES = (
    ("lin_s0_r2k", 0, 2_000, "linear"),
    ("lin_s500_r2k", 500, 2_000, "linear"),
    ("lin_s2k_r2k", 2_000, 2_000, "linear"),
    ("lin_s5k_r2k", 5_000, 2_000, "linear"),
    ("cos_s500_r500", 500, 500, "cosine"),
    ("cos_s500_r2k", 500, 2_000, "cosine"),
    ("cos_s500_r5k", 500, 5_000, "cosine"),
    ("cos_s2k_r5k", 2_000, 5_000, "cosine"),
)

_SPECS: list[DisentanglementSpec] = [
    _spec("control", "control"),
]

for weight in _OPACITY_WEIGHTS:
    for schedule_label, start, ramp, shape in _OPACITY_SCHEDULES:
        _SPECS.append(
            _spec(
                f"obin_w{weight:g}_{schedule_label}",
                "opacity_binarization",
                **{
                    "disentanglement.opacity_binarization_weight": weight,
                    "disentanglement.opacity_binarization_start": start,
                    "disentanglement.opacity_binarization_ramp": ramp,
                    "disentanglement.opacity_binarization_ramp_shape": shape,
                },
            )
        )

_TV_WEIGHTS = (1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3, 1.0e-2)
_TV_DELTAS = (1.0e-3, 5.0e-3, 2.0e-2, 1.0e-1)
for weight in _TV_WEIGHTS:
    for delta in _TV_DELTAS:
        _SPECS.append(
            _spec(
                f"tv_huber_w{weight:g}_d{delta:g}_const",
                "graph_tv",
                **{
                    "disentanglement.graph_tv_weight": weight,
                    "disentanglement.graph_tv_mode": "huber",
                    "disentanglement.graph_tv_huber_delta": delta,
                },
            )
        )

for weight in (3.0e-4, 1.0e-3, 3.0e-3, 1.0e-2):
    for label, start, ramp, shape in (
        ("lin_s0_r5k", 0, 5_000, "linear"),
        ("cos_s1k_r5k", 1_000, 5_000, "cosine"),
    ):
        _SPECS.append(
            _spec(
                f"tv_huber_w{weight:g}_d0.02_{label}",
                "graph_tv",
                **{
                    "disentanglement.graph_tv_weight": weight,
                    "disentanglement.graph_tv_mode": "huber",
                    "disentanglement.graph_tv_huber_delta": 2.0e-2,
                    "disentanglement.graph_tv_start": start,
                    "disentanglement.graph_tv_ramp": ramp,
                    "disentanglement.graph_tv_ramp_shape": shape,
                },
            )
        )

for mode in ("l1", "l2"):
    for weight in (1.0e-3, 1.0e-2):
        _SPECS.append(
            _spec(
                f"tv_{mode}_w{weight:g}",
                "graph_tv",
                **{
                    "disentanglement.graph_tv_weight": weight,
                    "disentanglement.graph_tv_mode": mode,
                },
            )
        )

for policy in ("rolling", "ema", "warmup_freeze"):
    for quantile in (0.95, 0.99):
        for multiplier in (1.5, 3.0):
            _SPECS.append(
                _spec(
                    f"cap_{policy}_q{quantile:g}_m{multiplier:g}",
                    "density_cap",
                    **{
                        "disentanglement.density_cap_weight": 1.0e-2,
                        "disentanglement.density_cap_start": 5_000,
                        "disentanglement.density_cap_ramp": 1_000,
                        "disentanglement.density_cap_ramp_shape": "cosine",
                        "disentanglement.density_cap_policy": policy,
                        "disentanglement.density_cap_quantile": quantile,
                        "disentanglement.density_cap_multiplier": multiplier,
                        "disentanglement.density_cap_minimum": 5.0e-3,
                        "disentanglement.density_cap_update_interval": 25,
                        "disentanglement.density_cap_ema_decay": 0.99,
                        "disentanglement.density_cap_warmup_until": 5_000,
                    },
                )
            )

for weight in (1.0e-5, 3.0e-5, 1.0e-4, 3.0e-4):
    _SPECS.append(
        _spec(
            f"tau_w{weight:g}",
            "optical_thickness",
            **{
                "disentanglement.optical_thickness_weight": weight,
                "disentanglement.optical_thickness_start": 1_000,
                "disentanglement.optical_thickness_ramp": 3_000,
                "disentanglement.optical_thickness_ramp_shape": "cosine",
            },
        )
    )

for weight in (3.0e-4, 1.0e-3, 3.0e-3, 1.0e-2):
    _SPECS.append(
        _spec(
            f"opaque_w{weight:g}",
            "opaque_if_used",
            **{
                "disentanglement.opaque_if_used_weight": weight,
                "disentanglement.opaque_if_used_start": 500,
                "disentanglement.opaque_if_used_ramp": 2_000,
                "disentanglement.opaque_if_used_ramp_shape": "cosine",
            },
        )
    )

for threshold in (0.015, 0.025, 0.05, 0.10):
    _SPECS.append(
        _spec(
            f"prune_a{threshold:g}_from3k",
            "opacity_pruning",
            **{
                "gaussian_adaptation.min_opacity": threshold,
                "gaussian_adaptation.opacity_prune_from": 3_000,
            },
        )
    )

DISENTANGLEMENT_SWEEP_SPECS = tuple(_SPECS)

if len(DISENTANGLEMENT_SWEEP_SPECS) != 97:
    raise RuntimeError(
        "Phase 1 disentanglement matrix must contain exactly 97 runs"
    )
if len({spec.label for spec in DISENTANGLEMENT_SWEEP_SPECS}) != 97:
    raise RuntimeError("Phase 1 disentanglement labels must be unique")


def _set_nested(config: ExperimentConfig, path: str, value: Any) -> None:
    section, field = path.split(".", 1)
    target = getattr(config, section)
    if not hasattr(target, field):
        raise ValueError(f"unknown override path: {path}")
    setattr(target, field, value)


def build_disentanglement_config(
    base: ExperimentConfig,
    index: int,
    output_dir: str,
    wandb_name: str,
    wandb_mode: str = "online",
) -> ExperimentConfig:
    if not 0 <= index < len(DISENTANGLEMENT_SWEEP_SPECS):
        raise IndexError("disentanglement sweep index out of range")
    config = deepcopy(base)
    spec = DISENTANGLEMENT_SWEEP_SPECS[index]
    for path, value in spec.overrides:
        _set_nested(config, path, value)

    config.seed = 0
    config.output_dir = output_dir
    config.experiment_name = wandb_name
    config.wandb.name = wandb_name
    config.wandb.mode = wandb_mode
    config.wandb.project = "foamsurf_disentanglement_phase1"
    config.wandb.tags = [
        "disentanglement",
        "phase1",
        "single-mechanism",
        "k64",
        spec.family,
        spec.label,
    ]
    config.diagnostics.save_per_primitive = False

    if config.renderer.gaussian_hit_buffer_size != 64:
        raise ValueError("Phase 1 disentanglement requires K=64")
    if config.dataset.train_target != "both":
        raise ValueError("Phase 1 must train only on lego_both")
    if config.optimization.reconstruction_loss != "l1_dssim":
        raise ValueError("Phase 1 requires the qualified L1+DSSIM objective")
    config.validate()
    return config


def disentanglement_spec_payload(index: int) -> dict[str, Any]:
    spec = DISENTANGLEMENT_SWEEP_SPECS[index]
    return {
        "index": index,
        "label": spec.label,
        "family": spec.family,
        "overrides": dict(spec.overrides),
    }
