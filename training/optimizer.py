from __future__ import annotations

import math

import torch

from foamsurf_model.scene import HybridSceneModel
from .config import OptimizationConfig


def build_optimizers(
    model: HybridSceneModel,
    config: OptimizationConfig,
    scene_extent: float = 1.0,
) -> tuple[torch.optim.Adam, torch.optim.Adam]:
    spatial_scale = (
        scene_extent if config.gaussian_means_spatial_scale else 1.0
    )
    gaussian_rates = {
        "appearance_dc": config.gaussian_appearance_lr,
        "appearance_rest": (
            config.gaussian_appearance_lr
            * config.gaussian_appearance_rest_lr_scale
        ),
        "opacity_logits": config.gaussian_opacity_lr,
        "raw_kappa": config.gaussian_kappa_lr,
        "means": config.gaussian_means_lr * spatial_scale,
        "log_scales": config.gaussian_scales_lr,
        "quaternions": config.gaussian_quaternions_lr,
    }
    gaussian_groups = [
        {
            "params": [parameter],
            "lr": gaussian_rates[name],
            "name": f"gaussian_{name}",
        }
        for name, parameter in (
            model.gaussians.named_primitive_parameters().items()
        )
    ]
    medium_groups = [
        {
            "params": [model.medium.sites],
            "lr": config.medium_sites_lr,
            "name": "medium_sites",
        },
        {
            "params": [model.medium.raw_density],
            "lr": config.medium_density_lr,
            "name": "medium_density",
        },
        {
            "params": [model.medium.raw_color],
            "lr": config.medium_color_lr,
            "name": "medium_color",
        },
    ]
    return (
        torch.optim.Adam(gaussian_groups, eps=config.gaussian_adam_eps),
        torch.optim.Adam(medium_groups),
    )


def _exponential_lr(
    step: int,
    lr_init: float,
    lr_final: float,
    delay_steps: int,
    delay_mult: float,
    max_steps: int,
) -> float:
    """GraphDECO/Plenoxels log-linear position LR schedule."""

    if delay_steps > 0:
        progress = min(max(step / delay_steps, 0.0), 1.0)
        delay_rate = delay_mult + (1.0 - delay_mult) * math.sin(
            0.5 * math.pi * progress
        )
    else:
        delay_rate = 1.0
    progress = min(max(step / max_steps, 0.0), 1.0)
    log_lerp = math.exp(
        math.log(lr_init) * (1.0 - progress)
        + math.log(lr_final) * progress
    )
    return delay_rate * log_lerp


def update_gaussian_learning_rate(
    optimizer: torch.optim.Adam,
    config: OptimizationConfig,
    scene_extent: float,
    iteration: int,
) -> float:
    """Update the position group exactly as in GraphDECO 3DGS."""

    spatial_scale = (
        scene_extent if config.gaussian_means_spatial_scale else 1.0
    )
    if config.gaussian_means_lr_final is None:
        rate = config.gaussian_means_lr * spatial_scale
    else:
        rate = _exponential_lr(
            iteration,
            config.gaussian_means_lr * spatial_scale,
            config.gaussian_means_lr_final * spatial_scale,
            config.gaussian_means_lr_delay_steps,
            config.gaussian_means_lr_delay_mult,
            config.gaussian_means_lr_max_steps,
        )
    group = next(
        group
        for group in optimizer.param_groups
        if group.get("name") == "gaussian_means"
    )
    group["lr"] = rate
    return rate


def gradient_norm(parameters) -> float:
    squared = None
    for parameter in parameters:
        if parameter.grad is None:
            continue
        value = parameter.grad.detach().float().square().sum()
        squared = value if squared is None else squared + value
    if squared is None:
        return 0.0
    return float(torch.sqrt(squared).item())
