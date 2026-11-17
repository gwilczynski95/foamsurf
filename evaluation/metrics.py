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


class ImageMetrics:
    """Paper metrics on float HWC images; VGG LPIPS receives normalized [0, 1] RGB."""

    def __init__(self, device="cpu", *, perceptual=True):
        self.lpips = None
        if perceptual:
            from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
            self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to(device).eval()

    @torch.no_grad()
    def __call__(self, prediction, target):
        prediction, target = prediction.clamp(0, 1), target.clamp(0, 1)
        result = {"psnr": float(psnr(prediction, target)), "ssim": float(ssim(prediction, target))}
        if self.lpips is not None:
            device = next(self.lpips.parameters()).device
            batch = lambda x: x.permute(2, 0, 1)[None].float().to(device)
            result["lpips_vgg"] = float(self.lpips(batch(target), batch(prediction)))
            self.lpips.reset()
        return result


def view_opacity(alpha, water):
    """Mean surfel opacity per class; a water-free view contributes only to seabed coverage."""
    import numpy as np
    alpha, water = np.asarray(alpha), np.asarray(water, dtype=bool)
    if alpha.size != water.size:
        raise ValueError("opacity and mask sizes differ")
    alpha = alpha.reshape(water.shape)
    if not (~water).any():
        raise ValueError("view without seabed pixels")
    return float(alpha[~water].mean()), float(alpha[water].mean()) if water.any() else None


def depth_errors(depth, truth):
    valid = torch.isfinite(depth) & (depth > 0) & torch.isfinite(truth) & (truth > 0)
    d, t = depth[valid], truth[valid]
    if not d.numel():
        raise ValueError("no valid positive depth pixels")
    scaled = d * (t / d).median()
    return {"depth_absrel": float(((d - t).abs() / t).mean()),
            "depth_absrel_median_scaled": float(((scaled - t).abs() / t).mean()),
            "depth_ratio_median": float((d / t).median())}


def check_psnr(measured, saved, tolerance=0.001):
    if set(measured) != set(saved) or not measured:
        raise ValueError("views differ or are empty")
    gap = max(abs(measured[v] - saved[v]) for v in measured)
    if gap > tolerance:
        raise ValueError(f"frame PSNR differs from the saved evaluation by {gap:.4f} dB")
    return gap
