from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from .config import ExperimentConfig


@dataclass(frozen=True)
class JointSchedulingSpec:
    label: str
    gaussian_freeze_until: int = 0
    voronoi_adaptation_offset: int = 0
    gaussian_densify_until: int = 15_000
    freeze_voronoi_at_capacity: bool = False


JOINT_SCHEDULING_SPECS = (
    JointSchedulingSpec(
        label="voronoi_warmup_2k",
        gaussian_freeze_until=2_000,
    ),
    JointSchedulingSpec(
        label="voronoi_offset_100",
        voronoi_adaptation_offset=100,
    ),
    JointSchedulingSpec(
        label="gaussian_adapt_until_10k",
        gaussian_densify_until=10_000,
    ),
    JointSchedulingSpec(
        label="freeze_voronoi_at_8k",
        freeze_voronoi_at_capacity=True,
    ),
)


def build_joint_scheduling_config(
    base: ExperimentConfig,
    index: int,
    output_dir: str,
    wandb_name: str,
    wandb_mode: str = "online",
) -> ExperimentConfig:
    config = deepcopy(base)
    spec = JOINT_SCHEDULING_SPECS[index]
    config.output_dir = output_dir
    config.experiment_name = wandb_name
    config.wandb.name = wandb_name
    config.wandb.mode = wandb_mode
    config.wandb.project = "foamsurf_clean_joint_scheduling"
    config.wandb.tags = list(config.wandb.tags) + ["scheduling", spec.label]

    config.optimization.gaussian_freeze_until = spec.gaussian_freeze_until
    config.gaussian_adaptation.densify_until = spec.gaussian_densify_until
    config.voronoi_adaptation.adaptation_offset = (
        spec.voronoi_adaptation_offset
    )
    config.voronoi_adaptation.freeze_at_capacity = (
        spec.freeze_voronoi_at_capacity
    )

    if config.renderer.gaussian_hit_buffer_size != 64:
        raise ValueError("joint scheduling tests require K=64")
    if config.dataset.train_target != "both":
        raise ValueError("joint scheduling tests require lego_both supervision")
    if config.optimization.reconstruction_loss != "l1_dssim":
        raise ValueError("joint scheduling tests require L1+DSSIM")
    config.validate()
    return config
