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
    return (
        torch.optim.Adam(gaussian_groups, eps=config.gaussian_adam_eps),
        build_medium_optimizer(model, config),
    )


def build_medium_optimizer(
    model: HybridSceneModel, config: OptimizationConfig
) -> torch.optim.Adam:
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
    if model.medium.color_mode == "global":
        # A separate group keeps topology permutations limited to per-cell
        # tensors; it follows the medium_color schedule.
        medium_groups.append(
            {
                "params": [model.medium.raw_global_color],
                "lr": config.medium_color_lr,
                "name": MEDIUM_GLOBAL_COLOR_GROUP,
            }
        )
    return torch.optim.Adam(medium_groups)


MEDIUM_GLOBAL_COLOR_GROUP = "medium_global_color"


def medium_schedule_name(group_name: str | None) -> str | None:
    """Map a medium optimizer group to the LR schedule it follows."""
    return "medium_color" if group_name == MEDIUM_GLOBAL_COLOR_GROUP else group_name


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


def _flare_exponential_lr(
    step: int,
    initial: float,
    decay: float,
    final: float | None,
) -> float:
    """Per-parameter schedule used by the working flare_new trainer."""

    rate = initial * math.exp(decay * step)
    if final is None:
        return rate
    return max(rate, final) if decay <= 0.0 else min(rate, final)


def update_gaussian_learning_rates(
    optimizer: torch.optim.Adam,
    config: OptimizationConfig,
    scene_extent: float,
    iteration: int,
) -> float:
    """Update Gaussian groups while preserving the GraphDECO defaults."""

    spatial_scale = (
        scene_extent if config.gaussian_means_spatial_scale else 1.0
    )
    if config.gaussian_means_lr_decay is not None:
        rate = _flare_exponential_lr(
            iteration,
            config.gaussian_means_lr * spatial_scale,
            config.gaussian_means_lr_decay,
            (
                None
                if config.gaussian_means_lr_final is None
                else config.gaussian_means_lr_final * spatial_scale
            ),
        )
    elif config.gaussian_means_lr_final is None:
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

    rates = {
        "gaussian_appearance_dc": _flare_exponential_lr(
            iteration,
            config.gaussian_appearance_lr,
            config.gaussian_appearance_lr_decay,
            config.gaussian_appearance_lr_final,
        ),
        "gaussian_appearance_rest": _flare_exponential_lr(
            iteration,
            config.gaussian_appearance_lr
            * config.gaussian_appearance_rest_lr_scale,
            config.gaussian_appearance_lr_decay,
            (
                None
                if config.gaussian_appearance_lr_final is None
                else config.gaussian_appearance_lr_final
                * config.gaussian_appearance_rest_lr_scale
            ),
        ),
        "gaussian_opacity_logits": _flare_exponential_lr(
            iteration,
            config.gaussian_opacity_lr,
            config.gaussian_opacity_lr_decay,
            config.gaussian_opacity_lr_final,
        ),
        "gaussian_raw_kappa": _flare_exponential_lr(
            iteration,
            config.gaussian_kappa_lr,
            config.gaussian_kappa_lr_decay,
            config.gaussian_kappa_lr_final,
        ),
        "gaussian_means": rate,
        "gaussian_log_scales": _flare_exponential_lr(
            iteration,
            config.gaussian_scales_lr,
            config.gaussian_scales_lr_decay,
            config.gaussian_scales_lr_final,
        ),
        "gaussian_quaternions": _flare_exponential_lr(
            iteration,
            config.gaussian_quaternions_lr,
            config.gaussian_quaternions_lr_decay,
            config.gaussian_quaternions_lr_final,
        ),
    }
    for group in optimizer.param_groups:
        name = group.get("name")
        if name in rates:
            group["lr"] = rates[name]
    return rate


def update_medium_learning_rates(
    optimizer: torch.optim.Adam,
    config: OptimizationConfig,
    iteration: int,
) -> dict[str, float]:
    """Apply an optional log-linear decay to each Voronoi parameter group."""

    progress_step = max(0, iteration - config.medium_lr_decay_from)
    schedule_steps = max(
        1, config.medium_lr_decay_until - config.medium_lr_decay_from
    )
    rates: dict[str, float] = {}
    for group_name, initial_name, final_name in (
        ("medium_sites", "medium_sites_lr", "medium_sites_lr_final"),
        ("medium_density", "medium_density_lr", "medium_density_lr_final"),
        ("medium_color", "medium_color_lr", "medium_color_lr_final"),
    ):
        initial = float(getattr(config, initial_name))
        final = getattr(config, final_name)
        if final is None or iteration <= config.medium_lr_decay_from:
            rate = initial
        else:
            rate = _exponential_lr(
                progress_step,
                initial,
                float(final),
                delay_steps=0,
                delay_mult=1.0,
                max_steps=schedule_steps,
            )
        rates[group_name] = rate

    for group in optimizer.param_groups:
        name = medium_schedule_name(group.get("name"))
        if name in rates:
            group["lr"] = rates[name]
    return rates


update_gaussian_learning_rate = update_gaussian_learning_rates


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
