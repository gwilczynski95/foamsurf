from __future__ import annotations

import torch

from foamsurf_model.scene import HybridSceneModel
from .config import OptimizationConfig


def build_optimizers(
    model: HybridSceneModel,
    config: OptimizationConfig,
) -> tuple[torch.optim.Adam, torch.optim.Adam]:
    gaussian_rates = {
        "appearance": config.gaussian_appearance_lr,
        "opacity_logits": config.gaussian_opacity_lr,
        "raw_kappa": config.gaussian_kappa_lr,
        "means": config.gaussian_means_lr,
        "log_scales": config.gaussian_scales_lr,
        "quaternions": config.gaussian_quaternions_lr,
    }
    gaussian_groups = [
        {
            "params": [parameter],
            "lr": gaussian_rates[name],
            "name": f"gaussian_{name}",
        }
        for name, parameter
        in model.gaussians.named_primitive_parameters().items()
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
        torch.optim.Adam(gaussian_groups),
        torch.optim.Adam(medium_groups),
    )


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
