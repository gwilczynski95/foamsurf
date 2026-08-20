from __future__ import annotations

from dataclasses import dataclass

import torch
from torchmetrics.functional.image import (
    structural_similarity_index_measure,
)


@dataclass(frozen=True)
class B0Loss:
    total: torch.Tensor
    rgb_mse: torch.Tensor
    psnr: torch.Tensor
    rgb_l1: torch.Tensor
    ssim: torch.Tensor
    gaussian_scale_regularization: torch.Tensor


def b0_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mode: str = "mse",
    dssim_weight: float = 0.2,
    image_size: tuple[int, int] | None = None,
) -> B0Loss:
    """Unregularized image objective used by B0/B1 ablations."""
    rgb_mse = torch.mean((prediction - target) ** 2)
    rgb_l1 = torch.mean(torch.abs(prediction - target))
    psnr = -10.0 * torch.log10(rgb_mse.clamp_min(1.0e-12))
    if mode == "mse":
        total = rgb_mse
        ssim = rgb_mse.new_tensor(float("nan"))
    elif mode == "l1":
        total = rgb_l1
        ssim = rgb_mse.new_tensor(float("nan"))
    elif mode == "l1_dssim":
        if prediction.ndim == 2 and image_size is not None:
            height, width = image_size
            prediction_image = prediction.reshape(height, width, 3)
            target_image = target.reshape(height, width, 3)
        elif prediction.ndim == 3 and prediction.shape[-1] == 3:
            prediction_image = prediction
            target_image = target
        else:
            raise ValueError(
                "l1_dssim requires a full-image batch and image_size"
            )
        prediction_nchw = prediction_image.permute(2, 0, 1).unsqueeze(0)
        target_nchw = target_image.permute(2, 0, 1).unsqueeze(0)
        ssim = structural_similarity_index_measure(
            prediction_nchw, target_nchw, data_range=1.0
        )
        total = (1.0 - dssim_weight) * rgb_l1 + dssim_weight * (
            1.0 - ssim
        )
    else:
        raise ValueError(f"unknown reconstruction loss: {mode}")
    return B0Loss(
        total=total,
        rgb_mse=rgb_mse,
        psnr=psnr,
        rgb_l1=rgb_l1,
        ssim=ssim,
        gaussian_scale_regularization=rgb_mse.new_zeros(()),
    )
