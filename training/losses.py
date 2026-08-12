from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class B0Loss:
    total: torch.Tensor
    rgb_mse: torch.Tensor
    psnr: torch.Tensor


def b0_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> B0Loss:
    """The complete B0 objective: hybrid RGB mean squared error only."""
    rgb_mse = torch.mean((prediction - target) ** 2)
    psnr = -10.0 * torch.log10(rgb_mse.clamp_min(1.0e-12))
    return B0Loss(total=rgb_mse, rgb_mse=rgb_mse, psnr=psnr)
