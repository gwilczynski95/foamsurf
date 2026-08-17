from __future__ import annotations

import copy
from pathlib import Path

from .config import ExperimentConfig


def build_twodgs_profile(
    base: ExperimentConfig,
    output_dir: str | Path,
    wandb_name: str,
    *,
    wandb_mode: str = "online",
    wandb_project: str = "foamsurf_2dgs_baseline",
    seed: int = 42,
    iterations: int = 30_000,
) -> ExperimentConfig:
    """Apply the optimization/adaptation defaults of pinned official 2DGS.

    The FoamSurf renderer and jointly trained medium remain part of the model.
    Surface regularizers from the 2DGS paper are intentionally outside B1;
    this profile matches its reconstruction loss and Gaussian population
    dynamics without adding normal or distortion supervision.
    """

    config = copy.deepcopy(base)
    config.experiment_name = "b1-2dgs-baseline"
    config.output_dir = str(Path(output_dir).resolve())
    config.seed = int(seed)

    optimization = config.optimization
    optimization.iterations = int(iterations)
    optimization.rays_per_batch = 640_000
    optimization.full_image_batch = True
    optimization.cameras_without_replacement = True
    optimization.reconstruction_loss = "l1_dssim"
    optimization.dssim_weight = 0.2
    optimization.gaussian_opacity_lr = 0.05

    adaptation = config.gaussian_adaptation
    adaptation.enabled = True
    adaptation.max_points = None
    adaptation.densify_from = 500
    adaptation.densify_until = 15_000
    adaptation.densification_interval = 100
    adaptation.gradient_threshold = 2.0e-4
    adaptation.min_opacity = 0.05
    adaptation.opacity_prune_from = 500
    adaptation.percent_dense = 0.01
    adaptation.size_prune_from = 3_000
    adaptation.screen_pruning_enabled = False
    adaptation.max_screen_size = 20.0
    adaptation.max_world_scale_fraction = 0.10
    adaptation.opacity_reset_interval = 3_000
    adaptation.opacity_reset_value = 0.01
    # NeRF Synthetic is trained on white background. Official 2DGS also
    # clamps opacity at densify_from for that case, in addition to 3k resets.
    adaptation.opacity_reset_at_densify_from = True
    adaptation.opacity_reset_optimizer_state = True
    adaptation.medium_freeze_after_reset = 0
    adaptation.medium_freeze_interval = 0
    adaptation.medium_freeze_from = 0
    adaptation.medium_freeze_steps = 0
    adaptation.preserve_clone_opacity = False
    adaptation.preserve_split_opacity = False
    # Safety stop only; it does not affect a healthy 2DGS trajectory.
    adaptation.runaway_max_points = 1_000_000

    config.wandb.mode = wandb_mode
    config.wandb.project = wandb_project
    config.wandb.name = wandb_name
    config.wandb.tags = [
        "B1",
        "2dgs-baseline",
        "full-image",
        "camera-without-replacement",
        "l1-dssim",
        f"seed-{seed}",
    ]
    config.validate()
    return config
