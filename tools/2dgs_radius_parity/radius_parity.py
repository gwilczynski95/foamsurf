#!/usr/bin/env python3
"""Compare FoamSurf projected radii with the official 2DGS rasterizer."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
from types import SimpleNamespace
import sys

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from foamsurf_model.gaussian_adaptation import (  # noqa: E402
    GaussianAdaptationController,
)

try:
    from diff_surfel_rasterization import (  # noqa: E402
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )
except ImportError as error:  # pragma: no cover
    raise SystemExit(
        "Official 2DGS rasterizer is not installed. Run "
        "tools/2dgs_radius_parity/install_official_rasterizer.sh "
        "inside the foamsurf conda environment."
    ) from error


@dataclass(frozen=True)
class RadiusComparison:
    names: list[str]
    reference: torch.Tensor
    foamsurf: torch.Tensor

    @property
    def visible(self) -> torch.Tensor:
        return self.reference > 0

    @property
    def difference(self) -> torch.Tensor:
        return self.foamsurf - self.reference.to(self.foamsurf.dtype)


def _projection_matrix(
    fov_x: float,
    fov_y: float,
    *,
    z_near: float = 0.01,
    z_far: float = 100.0,
    device: torch.device,
) -> torch.Tensor:
    projection = torch.zeros((4, 4), device=device)
    projection[0, 0] = 1.0 / math.tan(0.5 * fov_x)
    projection[1, 1] = 1.0 / math.tan(0.5 * fov_y)
    projection[3, 2] = 1.0
    projection[2, 2] = z_far / (z_far - z_near)
    projection[2, 3] = -(z_far * z_near) / (z_far - z_near)
    return projection


def _camera(
    *, width: int, height: int, fov_x: float, device: torch.device
) -> tuple[SimpleNamespace, torch.Tensor, torch.Tensor]:
    focal = width / (2.0 * math.tan(0.5 * fov_x))
    fov_y = 2.0 * math.atan(height / (2.0 * focal))
    camera = SimpleNamespace(
        origin=torch.zeros(3, device=device),
        right=torch.tensor([1.0, 0.0, 0.0], device=device),
        down=torch.tensor([0.0, 1.0, 0.0], device=device),
        forward=torch.tensor([0.0, 0.0, 1.0], device=device),
        width=width,
        height=height,
        fov_x=fov_x,
        fov_y=fov_y,
    )
    # GraphDECO stores matrices transposed for row-vector multiplication.
    world_view = torch.eye(4, device=device).transpose(0, 1).contiguous()
    projection = _projection_matrix(
        fov_x, fov_y, device=device
    ).transpose(0, 1).contiguous()
    return camera, world_view, world_view @ projection


def _official_radii(
    means: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    opacities: torch.Tensor,
    *,
    camera: SimpleNamespace,
    world_view: torch.Tensor,
    full_projection: torch.Tensor,
) -> torch.Tensor:
    settings = GaussianRasterizationSettings(
        image_height=camera.height,
        image_width=camera.width,
        tanfovx=math.tan(0.5 * camera.fov_x),
        tanfovy=math.tan(0.5 * camera.fov_y),
        bg=torch.zeros(3, device=means.device),
        scale_modifier=1.0,
        viewmatrix=world_view,
        projmatrix=full_projection,
        sh_degree=0,
        campos=camera.origin,
        prefiltered=False,
        debug=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=settings)
    with torch.no_grad():
        _, radii, _ = rasterizer(
            means3D=means.contiguous(),
            means2D=torch.zeros_like(means),
            colors_precomp=torch.ones_like(means),
            opacities=opacities.contiguous(),
            scales=scales.contiguous(),
            rotations=rotations.contiguous(),
            cov3D_precomp=None,
        )
    return radii


def _foamsurf_radii(
    means: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    camera: SimpleNamespace,
) -> torch.Tensor:
    controller = object.__new__(GaussianAdaptationController)
    controller.model = SimpleNamespace(
        means=means, scales=scales, quaternions=rotations
    )
    controller.support_radius = 3.0
    return controller.projected_radii(camera)


def compare(
    names: list[str],
    means: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    *,
    width: int = 800,
    height: int = 800,
    fov_x: float = math.radians(49.1),
) -> RadiusComparison:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the official 2DGS rasterizer")
    camera, world_view, full_projection = _camera(
        width=width, height=height, fov_x=fov_x, device=means.device
    )
    rotations = torch.nn.functional.normalize(rotations, dim=-1)
    opacities = torch.full(
        (means.shape[0], 1), 0.1, device=means.device, dtype=means.dtype
    )
    reference = _official_radii(
        means,
        scales,
        rotations,
        opacities,
        camera=camera,
        world_view=world_view,
        full_projection=full_projection,
    )
    candidate = _foamsurf_radii(means, scales, rotations, camera)
    return RadiusComparison(names, reference, candidate)


def deterministic_cases(device: torch.device) -> RadiusComparison:
    names = [
        "tiny_center",
        "isotropic_center",
        "off_axis",
        "near_camera",
        "far_camera",
        "anisotropic_front",
        "anisotropic_tilt_y_60",
        "anisotropic_tilt_y_80",
    ]
    means = torch.tensor(
        [
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 2.0],
            [0.6, -0.3, 2.5],
            [0.05, 0.02, 0.5],
            [0.0, 0.0, 8.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 2.0],
        ],
        device=device,
    )
    scales = torch.tensor(
        [
            [0.0001, 0.0001],
            [0.05, 0.05],
            [0.08, 0.03],
            [0.02, 0.04],
            [0.2, 0.2],
            [0.15, 0.02],
            [0.15, 0.02],
            [0.15, 0.02],
        ],
        device=device,
    )
    half_60 = math.radians(60.0) / 2.0
    half_80 = math.radians(80.0) / 2.0
    rotations = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [math.cos(half_60), 0.0, math.sin(half_60), 0.0],
            [math.cos(half_80), 0.0, math.sin(half_80), 0.0],
        ],
        device=device,
    )
    return compare(names, means, scales, rotations)


def randomized_cases(
    device: torch.device, count: int, seed: int
) -> RadiusComparison:
    generator = torch.Generator(device=device).manual_seed(seed)
    depth = torch.empty(count, device=device).uniform_(
        0.5, 8.0, generator=generator
    )
    tan_half_fov = math.tan(0.5 * math.radians(49.1))
    xy = (
        torch.empty((count, 2), device=device).uniform_(
            -0.65, 0.65, generator=generator
        )
        * depth[:, None]
        * tan_half_fov
    )
    means = torch.cat((xy, depth[:, None]), dim=-1)
    log_scales = torch.empty((count, 2), device=device).uniform_(
        math.log(0.002), math.log(0.25), generator=generator
    )
    scales = log_scales.exp()
    rotations = torch.randn(
        (count, 4), device=device, generator=generator
    )
    return compare(
        [f"random_{index:04d}" for index in range(count)],
        means,
        scales,
        rotations,
    )


def summary(comparison: RadiusComparison) -> dict[str, float]:
    visible = comparison.visible
    if not bool(visible.any()):
        raise RuntimeError("official rasterizer reported no visible cases")
    absolute = comparison.difference[visible].abs()
    return {
        "visible": float(visible.sum().item()),
        "exact_fraction": float((absolute == 0).float().mean().item()),
        "within_one_fraction": float((absolute <= 1).float().mean().item()),
        "mean_absolute_error": float(absolute.float().mean().item()),
        "p95_absolute_error": float(
            torch.quantile(absolute.float(), 0.95).item()
        ),
        "max_absolute_error": float(absolute.max().item()),
    }


def _print_deterministic(comparison: RadiusComparison) -> None:
    print("\nDeterministic radius cases (pixels)")
    print(f"{'case':30s} {'2DGS':>8s} {'FoamSurf':>10s} {'delta':>8s}")
    for name, reference, candidate, difference in zip(
        comparison.names,
        comparison.reference.tolist(),
        comparison.foamsurf.tolist(),
        comparison.difference.tolist(),
        strict=True,
    ):
        print(
            f"{name:30s} {reference:8d} "
            f"{candidate:10.1f} {difference:8.1f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-cases", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--max-allowed-error-px", type=float, default=1.0)
    arguments = parser.parse_args()
    device = torch.device("cuda")

    deterministic = deterministic_cases(device)
    randomized = randomized_cases(
        device, arguments.random_cases, arguments.seed
    )
    _print_deterministic(deterministic)
    randomized_summary = summary(randomized)
    print("\nRandomized in-frustum comparison")
    for key, value in randomized_summary.items():
        print(f"{key:28s}: {value:.6f}")

    passed = (
        randomized_summary["max_absolute_error"]
        <= arguments.max_allowed_error_px
        and summary(deterministic)["max_absolute_error"]
        <= arguments.max_allowed_error_px
    )
    print("\nPARITY:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
