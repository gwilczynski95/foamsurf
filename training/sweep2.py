from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ExperimentConfig


@dataclass(frozen=True)
class Sweep2Spec:
    label: str
    question: str
    overrides: dict[str, Any]


SWEEP2_SPECS = (
    Sweep2Spec(
        "00_control_mse",
        "Repeat sweep1/05 (opacity reset 0.05 and MSE) as the control.",
        {},
    ),
    Sweep2Spec(
        "01_loss_l1",
        "Measure whether L1 improves foreground detail over MSE.",
        {"optimization.reconstruction_loss": "l1"},
    ),
    Sweep2Spec(
        "02_loss_l1dssim",
        "Measure 3DGS-style 0.8 L1 + 0.2 DSSIM sharpness.",
        {
            "optimization.reconstruction_loss": "l1_dssim",
            "optimization.dssim_weight": 0.2,
        },
    ),
    Sweep2Spec(
        "03_preserve_split",
        "Preserve absorption only when a Gaussian is split.",
        {"gaussian_adaptation.preserve_split_opacity": True},
    ),
    Sweep2Spec(
        "04_preserve_clone",
        "Preserve absorption only when a retained parent is cloned.",
        {"gaussian_adaptation.preserve_clone_opacity": True},
    ),
    Sweep2Spec(
        "05_preserve_clone_split",
        "Preserve absorption for both cloning and splitting.",
        {
            "gaussian_adaptation.preserve_clone_opacity": True,
            "gaussian_adaptation.preserve_split_opacity": True,
        },
    ),
    Sweep2Spec(
        "06_l1dssim_preserve_split",
        "Combine the sharpness loss with opacity-preserving splitting.",
        {
            "optimization.reconstruction_loss": "l1_dssim",
            "optimization.dssim_weight": 0.2,
            "gaussian_adaptation.preserve_split_opacity": True,
        },
    ),
    Sweep2Spec(
        "07_reset0075_mse",
        "Test an opacity reset halfway between 0.05 and 0.10.",
        {"gaussian_adaptation.opacity_reset_value": 0.075},
    ),
    Sweep2Spec(
        "08_reset0075_l1dssim",
        "Combine the intermediate reset with L1 + DSSIM.",
        {
            "optimization.reconstruction_loss": "l1_dssim",
            "optimization.dssim_weight": 0.2,
            "gaussian_adaptation.opacity_reset_value": 0.075,
        },
    ),
    Sweep2Spec(
        "09_reset010_l1dssim",
        "Test the sharper loss with the sweep1/07 reset value.",
        {
            "optimization.reconstruction_loss": "l1_dssim",
            "optimization.dssim_weight": 0.2,
            "gaussian_adaptation.opacity_reset_value": 0.10,
        },
    ),
)


def _set_dotted(config: ExperimentConfig, name: str, value: Any) -> None:
    section_name, field_name = name.split(".", 1)
    setattr(getattr(config, section_name), field_name, value)


def build_sweep2_config(
    base: ExperimentConfig,
    index: int,
    output_dir: str | Path,
    wandb_name: str,
    wandb_mode: str = "online",
    wandb_project: str = "foamsurf_sweep2",
    seed: int = 42,
) -> ExperimentConfig:
    if not 0 <= index < len(SWEEP2_SPECS):
        raise IndexError(
            f"sweep2 index must be in [0, {len(SWEEP2_SPECS) - 1}]"
        )
    spec = SWEEP2_SPECS[index]
    config = deepcopy(base)
    config.experiment_name = f"sweep2-{spec.label}"
    config.output_dir = str(output_dir)
    config.seed = int(seed)

    # Common sweep1/05 baseline. Keep these assignments explicit so the
    # experiment does not silently change when the source YAML evolves.
    config.optimization.iterations = 30_000
    config.optimization.rays_per_batch = 640_000
    config.optimization.full_image_batch = True
    config.optimization.cameras_without_replacement = True
    config.optimization.reconstruction_loss = "mse"
    config.optimization.dssim_weight = 0.2
    config.validation.eval_interval = 500
    config.validation.log_interval = 10
    config.validation.checkpoint_interval = 1_000

    adaptation = config.gaussian_adaptation
    adaptation.enabled = True
    adaptation.max_points = None
    adaptation.densify_from = 500
    adaptation.densify_until = 15_000
    adaptation.densification_interval = 100
    adaptation.gradient_threshold = 2.0e-4
    adaptation.min_opacity = 0.005
    adaptation.opacity_prune_from = 500
    # Preserve historical sweep semantics; the dedicated 2DGS profile carries
    # the corrected upstream-compatible screen-pruning behavior.
    adaptation.screen_pruning_enabled = True
    adaptation.opacity_reset_interval = 3_000
    adaptation.opacity_reset_value = 0.05
    adaptation.opacity_reset_at_densify_from = False
    adaptation.opacity_reset_optimizer_state = True
    adaptation.medium_freeze_after_reset = 0
    adaptation.medium_freeze_interval = 0
    adaptation.medium_freeze_from = 0
    adaptation.medium_freeze_steps = 0
    adaptation.preserve_clone_opacity = False
    adaptation.preserve_split_opacity = False
    adaptation.runaway_max_points = 1_000_000

    for name, value in spec.overrides.items():
        _set_dotted(config, name, value)

    config.wandb.mode = wandb_mode
    config.wandb.project = wandb_project
    config.wandb.name = wandb_name
    config.wandb.tags = [
        "sweep2",
        "B1",
        "lego-layers",
        "30k",
        "full-image",
        "reset005-baseline",
        spec.label,
    ]
    config.validate()
    return config
