from __future__ import annotations

import torch

from .metrics import psnr, ssim


def layer_region_masks(
    lego_alpha: torch.Tensor,
    cloud_alpha: torch.Tensor,
    threshold: float,
) -> dict[str, torch.Tensor]:
    lego = lego_alpha > threshold
    cloud = cloud_alpha > threshold
    return {
        "lego_only": lego & ~cloud,
        "cloud_only": cloud & ~lego,
        "overlap": lego & cloud,
        "background": ~lego & ~cloud,
    }


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.expand_as(value)
    if not bool(expanded.any()):
        return value.new_tensor(float("nan"))
    return value[expanded].mean()


def _masked_psnr(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    error = _masked_mean((prediction - target).square(), mask)
    return -10.0 * torch.log10(error.clamp_min(1.0e-12))


def decomposition_metrics(
    gaussian_image: torch.Tensor,
    gaussian_alpha: torch.Tensor,
    medium_image: torch.Tensor,
    medium_alpha: torch.Tensor,
    lego_target: torch.Tensor,
    lego_alpha: torch.Tensor,
    cloud_target: torch.Tensor,
    cloud_alpha: torch.Tensor,
    alpha_threshold: float,
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    """Compare isolated branches with aligned decomposition ground truth."""
    masks = layer_region_masks(lego_alpha, cloud_alpha, alpha_threshold)
    values = {
        "lego_lego/psnr": psnr(gaussian_image, lego_target),
        "lego_lego/ssim": ssim(gaussian_image, lego_target),
        "lego_lego/rgb_mae": (gaussian_image - lego_target).abs().mean(),
        "lego_lego/alpha_mae": (gaussian_alpha - lego_alpha).abs().mean(),
        "lego_lego/foreground_psnr": _masked_psnr(
            gaussian_image, lego_target, lego_alpha > alpha_threshold
        ),
        "lego_clouds/psnr": psnr(medium_image, cloud_target),
        "lego_clouds/ssim": ssim(medium_image, cloud_target),
        "lego_clouds/rgb_mae": (medium_image - cloud_target).abs().mean(),
        "lego_clouds/alpha_mae": (medium_alpha - cloud_alpha).abs().mean(),
        "lego_clouds/foreground_psnr": _masked_psnr(
            medium_image, cloud_target, cloud_alpha > alpha_threshold
        ),
        "entanglement/medium_alpha_on_lego_only": _masked_mean(
            medium_alpha, masks["lego_only"]
        ),
        "entanglement/gaussian_alpha_on_cloud_only": _masked_mean(
            gaussian_alpha, masks["cloud_only"]
        ),
        "entanglement/gaussian_alpha_on_background": _masked_mean(
            gaussian_alpha, masks["background"]
        ),
        "entanglement/medium_alpha_on_background": _masked_mean(
            medium_alpha, masks["background"]
        ),
        "entanglement/gaussian_dominance_on_lego_only": _masked_mean(
            (gaussian_alpha > medium_alpha).float(), masks["lego_only"]
        ),
        "entanglement/medium_dominance_on_cloud_only": _masked_mean(
            (medium_alpha > gaussian_alpha).float(), masks["cloud_only"]
        ),
    }
    for name, mask in masks.items():
        values[f"regions/{name}_fraction"] = mask.float().mean()
    return {
        name: float(value.item()) for name, value in values.items()
    }, masks
