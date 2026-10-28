from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RayBundle:
    origins: torch.Tensor
    directions: torch.Tensor
    camera_indices: torch.Tensor | None = None
    pixel_indices: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.origins.shape != self.directions.shape:
            raise ValueError("ray origins and directions must have equal shapes")
        if self.origins.shape[-1] != 3:
            raise ValueError("ray tensors must end in three coordinates")


def generate_camera_rays(
    origin: torch.Tensor,
    right: torch.Tensor,
    down: torch.Tensor,
    forward: torch.Tensor,
    width: int,
    height: int,
    fov_x: float,
    fov_y: float,
    intrinsics=None,
    pixel_offsets: torch.Tensor | None = None,
) -> RayBundle:
    """Rays through the pixel centres, or through centre + pixel_offsets [H, W, 2] (x, y in pixels)."""
    device = origin.device
    dtype = origin.dtype
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    if pixel_offsets is not None:
        x = x + pixel_offsets[..., 0].to(dtype)
        y = y + pixel_offsets[..., 1].to(dtype)
    dx = (-0.5 + (x + 0.5) / width) * (2.0 * torch.tan(
        torch.tensor(0.5 * fov_x, device=device, dtype=dtype)
    ))
    dy = (-0.5 + (y + 0.5) / height) * (2.0 * torch.tan(
        torch.tensor(0.5 * fov_y, device=device, dtype=dtype)
    ))
    if intrinsics is not None:
        fx, fy, cx, cy = intrinsics
        dx = (x + 0.5 - cx) / fx
        dy = (y + 0.5 - cy) / fy
    directions = (
        right.reshape(1, 1, 3) * dx[..., None]
        + down.reshape(1, 1, 3) * dy[..., None]
        + forward.reshape(1, 1, 3)
    )
    directions = F.normalize(directions, dim=-1)
    origins = origin.reshape(1, 1, 3).expand_as(directions)
    return RayBundle(origins, directions)


def generate_indexed_rays(
    flat_indices: torch.Tensor,
    origins: torch.Tensor,
    rights: torch.Tensor,
    downs: torch.Tensor,
    forwards: torch.Tensor,
    fov_x: torch.Tensor,
    fov_y: torch.Tensor,
    width: int,
    height: int,
    intrinsics=None,
    pixel_offsets: torch.Tensor | None = None,
) -> RayBundle:
    """Rays through the pixel centres, or through centre + pixel_offsets [R, 2] (x, y in pixels)."""
    flat_indices = flat_indices.to(torch.int64)
    pixels_per_camera = width * height
    camera_indices = torch.div(
        flat_indices, pixels_per_camera, rounding_mode="floor"
    )
    pixel_indices = flat_indices.remainder(pixels_per_camera)
    y = torch.div(pixel_indices, width, rounding_mode="floor")
    x = pixel_indices.remainder(width)

    dtype = origins.dtype
    x = x.to(dtype)
    y = y.to(dtype)
    if pixel_offsets is not None:
        x = x + pixel_offsets[:, 0].to(dtype)
        y = y + pixel_offsets[:, 1].to(dtype)
    tan_x = torch.tan(0.5 * fov_x[camera_indices])
    tan_y = torch.tan(0.5 * fov_y[camera_indices])
    dx = (-0.5 + (x + 0.5) / width) * (2.0 * tan_x)
    dy = (-0.5 + (y + 0.5) / height) * (2.0 * tan_y)

    if intrinsics is not None:
        fx, fy, cx, cy = intrinsics[camera_indices].unbind(-1)
        dx = (x + 0.5 - cx) / fx
        dy = (y + 0.5 - cy) / fy
    ray_origins = origins[camera_indices]
    directions = (
        rights[camera_indices] * dx[:, None]
        + downs[camera_indices] * dy[:, None]
        + forwards[camera_indices]
    )
    directions = F.normalize(directions, dim=-1)
    return RayBundle(
        ray_origins,
        directions,
        camera_indices=camera_indices,
        pixel_indices=pixel_indices,
    )
