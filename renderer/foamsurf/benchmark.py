from __future__ import annotations

import argparse

import torch

from foamsurf import GaussianScene, NativeHybridRenderer, VoronoiScene


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthetic full-call benchmark for NativeHybridRenderer"
    )
    parser.add_argument("--resolution", type=int, default=800)
    parser.add_argument("--sites", type=int, default=96)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(23)
    device = torch.device("cuda")
    points = torch.randn(args.sites, 3, device=device) * 3.0
    indices = torch.arange(args.sites, device=device)
    adjacency = indices.repeat(args.sites, 1)
    adjacency = adjacency[adjacency != indices[:, None]].to(torch.uint32)
    offsets = torch.arange(
        0, args.sites * (args.sites - 1) + 1, args.sites - 1,
        device=device,
    ).to(torch.uint32)
    medium = VoronoiScene(
        sites=points,
        densities=torch.full((args.sites, 1), 0.08, device=device),
        colors=torch.rand(args.sites, 3, device=device) * 0.4 + 0.1,
    )
    gaussians = GaussianScene(
        means=torch.tensor([[0.0, 0.0, 3.0]], device=device),
        log_scales=torch.tensor([[0.2, 0.2]], device=device),
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device),
        colors=torch.tensor([[0.8, 0.2, 0.1]], device=device),
        opacity_logits=torch.tensor([[2.0]], device=device),
        raw_kappa=torch.tensor([[0.0]], device=device),
    )
    coordinates = torch.linspace(-1.0, 1.0, args.resolution, device=device)
    y, x = torch.meshgrid(coordinates, coordinates, indexing="ij")
    ray_count = args.resolution**2
    origins = torch.zeros((ray_count, 3), device=device)
    directions = torch.stack(
        (x.flatten() * 0.25, y.flatten() * 0.25, torch.ones(ray_count, device=device)),
        dim=-1,
    )
    rays = torch.cat((origins, directions), dim=-1)
    nearest = points.square().sum(dim=-1).argmin().to(torch.uint32).reshape(1)
    start_points = nearest.expand(ray_count).contiguous()
    renderer = NativeHybridRenderer(
        max_batch_size=ray_count,
        far=8.0,
        transmittance_threshold=0.001,
    )
    renderer.set_gaussian_geometry(gaussians)
    renderer.set_voronoi_geometry(
        medium,
        adjacency.contiguous(),
        offsets.contiguous(),
    )

    with torch.no_grad():
        for _ in range(args.warmup):
            renderer.render(
                rays,
                gaussians=gaussians,
                voronoi=medium,
                start_points=start_points,
            )
        torch.cuda.synchronize()
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(args.iterations):
            result = renderer.render(
                rays,
                gaussians=gaussians,
                voronoi=medium,
                start_points=start_points,
            )
        end.record()
        torch.cuda.synchronize()

    milliseconds = begin.elapsed_time(end) / args.iterations
    print(
        f"resolution={args.resolution} rays={ray_count} "
        f"mean_ms={milliseconds:.3f} fps={1000.0 / milliseconds:.2f} "
        f"image_mean={result.image.mean().item():.6f}"
    )


if __name__ == "__main__":
    main()
