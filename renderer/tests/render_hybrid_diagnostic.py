from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from foamsurf import GaussianScene, HybridRenderer, VoronoiScene


def make_rays(resolution: int, extent: float) -> torch.Tensor:
    coordinates = torch.linspace(-extent, extent, resolution)
    y, x = torch.meshgrid(coordinates, coordinates, indexing="ij")
    origins = torch.stack((x, -y, torch.zeros_like(x)), dim=-1)
    directions = torch.zeros_like(origins)
    directions[..., 2] = 1.0
    return torch.cat((origins, directions), dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=128)
    args = parser.parse_args()

    dtype = torch.float32
    surface = GaussianScene(
        means=torch.tensor(
            [[-0.35, 0.0, 2.0], [0.35, 0.0, 2.6]], dtype=dtype
        ),
        log_scales=torch.full((2, 2), math.log(0.16), dtype=dtype),
        quaternions=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=dtype
        ),
        colors=torch.tensor(
            [[1.0, 0.15, 0.05], [1.0, 0.85, 0.05]], dtype=dtype
        ),
        opacity_logits=torch.full((2, 1), 3.0, dtype=dtype),
        raw_kappa=torch.zeros((2, 1), dtype=dtype),
    )
    medium = VoronoiScene(
        sites=torch.tensor(
            [[-0.5, 0.0, 2.0], [0.5, 0.0, 2.0]], dtype=dtype
        ),
        densities=torch.tensor([[0.12], [0.28]], dtype=dtype),
        colors=torch.tensor(
            [[0.05, 0.2, 0.9], [0.05, 0.85, 0.3]], dtype=dtype
        ),
    )
    renderer = HybridRenderer(far=4.0)
    rays = make_rays(args.resolution, 0.8)
    panels = (
        renderer.render_gaussians(rays, surface).image,
        renderer.render_voronoi(rays, medium).image,
        renderer(rays, gaussians=surface, voronoi=medium).image,
    )
    image = torch.cat(panels, dim=1).clamp(0.0, 1.0)
    pixels = (image.detach().cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(args.output)
    print(f"wrote {args.output} (surface | medium | hybrid)")


if __name__ == "__main__":
    main()
