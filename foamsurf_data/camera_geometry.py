from __future__ import annotations

import torch


def camera_normalization(cameras) -> tuple[torch.Tensor, float]:
    """Return the GraphDECO camera center and 1.1x camera radius."""

    if not cameras:
        raise ValueError("at least one training camera is required")
    origins = torch.stack([camera.origin for camera in cameras])
    center = origins.mean(dim=0)
    radius = 1.1 * torch.linalg.vector_norm(
        origins - center, dim=-1
    ).max()
    if not torch.isfinite(radius) or radius <= 0:
        raise ValueError("training cameras must have a positive finite extent")
    return center, float(radius.item())
