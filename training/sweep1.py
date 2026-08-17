from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ExperimentConfig


@dataclass(frozen=True)
class Sweep1Spec:
    label: str
    question: str
    overrides: dict[str, Any]


SWEEP1_SPECS = (
    Sweep1Spec(
        "00_noreset",
        "Control: unrestricted adaptation without opacity reset.",
        {},
    ),
    Sweep1Spec(
        "01_freeze_only200",
        "Isolate freezing medium for 200 steps every 3000 iterations.",
        {
            "gaussian_adaptation.medium_freeze_interval": 3000,
            "gaussian_adaptation.medium_freeze_from": 3000,
            "gaussian_adaptation.medium_freeze_steps": 200,
        },
    ),
    Sweep1Spec(
        "02_reset001",
        "Reference 0.01 reset without freezing medium.",
        {
            "gaussian_adaptation.opacity_reset_interval": 3000,
            "gaussian_adaptation.opacity_reset_value": 0.01,
        },
    ),
    Sweep1Spec(
        "03_reset001_f200",
        "Reference 0.01 reset and freeze medium for 200 steps.",
        {
            "gaussian_adaptation.opacity_reset_interval": 3000,
            "gaussian_adaptation.opacity_reset_value": 0.01,
            "gaussian_adaptation.medium_freeze_after_reset": 200,
        },
    ),
    Sweep1Spec(
        "04_reset001_f500",
        "Reference 0.01 reset and freeze medium for 500 steps.",
        {
            "gaussian_adaptation.opacity_reset_interval": 3000,
            "gaussian_adaptation.opacity_reset_value": 0.01,
            "gaussian_adaptation.medium_freeze_after_reset": 500,
        },
    ),
    Sweep1Spec(
        "05_reset005",
        "Softer 0.05 reset without freezing medium.",
        {
            "gaussian_adaptation.opacity_reset_interval": 3000,
            "gaussian_adaptation.opacity_reset_value": 0.05,
        },
    ),
    Sweep1Spec(
        "06_reset005_f200",
        "Softer 0.05 reset and freeze medium for 200 steps.",
        {
            "gaussian_adaptation.opacity_reset_interval": 3000,
            "gaussian_adaptation.opacity_reset_value": 0.05,
            "gaussian_adaptation.medium_freeze_after_reset": 200,
        },
    ),
    Sweep1Spec(
        "07_reset010",
        "Soft 0.10 reset without freezing medium.",
        {
            "gaussian_adaptation.opacity_reset_interval": 3000,
            "gaussian_adaptation.opacity_reset_value": 0.10,
        },
    ),
    Sweep1Spec(
        "08_reset010_f200",
        "Soft 0.10 reset and freeze medium for 200 steps.",
        {
            "gaussian_adaptation.opacity_reset_interval": 3000,
            "gaussian_adaptation.opacity_reset_value": 0.10,
            "gaussian_adaptation.medium_freeze_after_reset": 200,
        },
    ),
    Sweep1Spec(
        "09_reset010_f500",
        "Soft 0.10 reset and freeze medium for 500 steps.",
        {
            "gaussian_adaptation.opacity_reset_interval": 3000,
            "gaussian_adaptation.opacity_reset_value": 0.10,
            "gaussian_adaptation.medium_freeze_after_reset": 500,
        },
    ),
    Sweep1Spec(
        "10_reset010_f200_keepadam",
        "Separate the opacity clamp from clearing Adam moments.",
        {
            "gaussian_adaptation.opacity_reset_interval": 3000,
            "gaussian_adaptation.opacity_reset_value": 0.10,
            "gaussian_adaptation.medium_freeze_after_reset": 200,
            "gaussian_adaptation.opacity_reset_optimizer_state": False,
        },
    ),
    Sweep1Spec(
        "11_official_white_reset",
        "Exact white-background reset at 500 and every 3000 steps.",
        {
            "gaussian_adaptation.opacity_reset_interval": 3000,
            "gaussian_adaptation.opacity_reset_value": 0.01,
            "gaussian_adaptation.opacity_reset_at_densify_from": True,
        },
    ),
    Sweep1Spec(
        "12_loss_l1",
        "Isolate the L1 image objective.",
        {"optimization.reconstruction_loss": "l1"},
    ),
    Sweep1Spec(
        "13_loss_l1dssim",
        "Reference 3DGS/2DGS image objective: 0.8 L1 + 0.2 DSSIM.",
        {
            "optimization.reconstruction_loss": "l1_dssim",
            "optimization.dssim_weight": 0.2,
        },
    ),
    Sweep1Spec(
        "14_opacity_lr005",
        "Isolate the 2DGS opacity learning rate 0.05.",
        {"optimization.gaussian_opacity_lr": 0.05},
    ),
    Sweep1Spec(
        "15_opacity_cull005",
        "Isolate the 2DGS opacity pruning threshold 0.05.",
        {"gaussian_adaptation.min_opacity": 0.05},
    ),
    Sweep1Spec(
        "16_grad0005",
        "Reduce clone candidates with gradient threshold 0.0005.",
        {"gaussian_adaptation.gradient_threshold": 5.0e-4},
    ),
    Sweep1Spec(
        "17_grad0010",
        "Reduce clone candidates with gradient threshold 0.0010.",
        {"gaussian_adaptation.gradient_threshold": 1.0e-3},
    ),
    Sweep1Spec(
        "18_alpha_clone",
        "Preserve combined opacity when a retained Gaussian is cloned.",
        {"gaussian_adaptation.preserve_clone_opacity": True},
    ),
    Sweep1Spec(
        "19_alpha_clone_split",
        "Preserve combined opacity for both clone and split.",
        {
            "gaussian_adaptation.preserve_clone_opacity": True,
            "gaussian_adaptation.preserve_split_opacity": True,
        },
    ),
)


def _set_dotted(config: ExperimentConfig, name: str, value: Any) -> None:
    section_name, field_name = name.split(".", 1)
    setattr(getattr(config, section_name), field_name, value)


def build_sweep1_config(
    base: ExperimentConfig,
    index: int,
    output_dir: str | Path,
    wandb_name: str,
    wandb_mode: str = "online",
    wandb_project: str = "foamsurf_sweep1",
    seed: int = 42,
) -> ExperimentConfig:
    if not 0 <= index < len(SWEEP1_SPECS):
        raise IndexError(
            f"sweep1 index must be in [0, {len(SWEEP1_SPECS) - 1}]"
        )
    spec = SWEEP1_SPECS[index]
    config = deepcopy(base)
    config.experiment_name = f"sweep1-{spec.label}"
    config.output_dir = str(output_dir)
    config.seed = int(seed)
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
    # Preserve the historical sweep semantics. The new primary 2DGS profile
    # disables this because pinned upstream 2DGS clears max_radii2D before the
    # final prune in a densification event.
    adaptation.screen_pruning_enabled = True
    adaptation.opacity_reset_interval = 0
    adaptation.opacity_reset_value = 0.01
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
        "sweep1",
        "B1",
        "lego-layers",
        "30k",
        "full-image",
        spec.label,
    ]
    config.validate()
    return config
