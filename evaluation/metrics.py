from __future__ import annotations

import torch
from torchmetrics.functional.image import (
    structural_similarity_index_measure,
)


def mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((prediction - target) ** 2)


def psnr(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -10.0 * torch.log10(
        mse(prediction, target).clamp_min(1.0e-12)
    )


def ssim(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction_nchw = prediction.permute(2, 0, 1).unsqueeze(0)
    target_nchw = target.permute(2, 0, 1).unsqueeze(0)
    return structural_similarity_index_measure(
        prediction_nchw, target_nchw, data_range=1.0
    )
