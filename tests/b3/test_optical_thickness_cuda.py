from __future__ import annotations

import pytest
import torch

from foamsurf import GaussianScene, NativeHybridRenderer, VoronoiScene
from training.disentanglement import optical_thickness_loss


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="optical-thickness renderer integration requires CUDA",
)


def _requirements() -> None:
    try:
        import foamsurf._C  # noqa: F401
    except ImportError as error:
        pytest.skip(f"FoamSurf native bindings are unavailable: {error}")


def _adjacency(point_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.arange(point_count, device="cuda")
    neighbors = indices.repeat(point_count, 1)
    neighbors = neighbors[neighbors != indices[:, None]].to(torch.uint32)
    offsets = torch.arange(
        0,
        point_count * (point_count - 1) + 1,
        point_count - 1,
        device="cuda",
    ).to(torch.uint32)
    return neighbors.contiguous(), offsets.contiguous()


def test_medium_only_transmittance_produces_correct_density_gradient() -> None:
    _requirements()
    torch.manual_seed(913)
    sites = torch.randn(40, 3, device="cuda") * 1.5
    density = (
        torch.rand(40, 1, device="cuda") * 0.1 + 0.05
    ).requires_grad_()
    colors = torch.full((40, 3), 0.5, device="cuda")
    scene = VoronoiScene(sites, density, colors)
    adjacency, offsets = _adjacency(40)
    renderer = NativeHybridRenderer(
        max_batch_size=8,
        far=5.0,
        transmittance_threshold=0.0,
        gaussian_hit_buffer_size=64,
    )
    renderer.set_voronoi_geometry(scene, adjacency, offsets)
    rays = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.01, 0.02, 1.0],
            [0.1, -0.1, 0.0, -0.02, 0.01, 1.0],
        ],
        device="cuda",
    )

    result = renderer.render(rays, voronoi=scene)
    loss = optical_thickness_loss(result.transmittance)
    loss.backward()

    assert torch.isfinite(loss)
    assert loss.item() > 0.0
    assert density.grad is not None
    assert torch.isfinite(density.grad).all()
    assert density.grad.sum().item() > 0.0

    index = int(density.grad.abs().argmax().item())
    analytic = float(density.grad.reshape(-1)[index].item())
    epsilon = 1.0e-3
    with torch.no_grad():
        density.reshape(-1)[index] += epsilon
        plus = float(
            optical_thickness_loss(
                renderer.render(rays, voronoi=scene).transmittance
            ).item()
        )
        density.reshape(-1)[index] -= 2.0 * epsilon
        minus = float(
            optical_thickness_loss(
                renderer.render(rays, voronoi=scene).transmittance
            ).item()
        )
        density.reshape(-1)[index] += epsilon
    numerical = (plus - minus) / (2.0 * epsilon)
    assert numerical == pytest.approx(analytic, rel=2.0e-2, abs=2.0e-3)

def test_hybrid_and_medium_only_renders_share_one_backward_safely() -> None:
    """Match the two-forward graph used by Trainer for the tau loss."""
    _requirements()
    torch.manual_seed(914)
    sites = torch.randn(40, 3, device="cuda") * 1.5
    density = (
        torch.rand(40, 1, device="cuda") * 0.1 + 0.05
    ).requires_grad_()
    colors = torch.full((40, 3), 0.5, device="cuda")
    medium = VoronoiScene(sites, density, colors)
    gaussians = GaussianScene(
        torch.tensor([[0.0, 0.0, 2.0]], device="cuda", requires_grad=True),
        torch.tensor([[-1.5, -1.5]], device="cuda", requires_grad=True),
        torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]], device="cuda", requires_grad=True
        ),
        torch.tensor([[0.8, 0.2, 0.1]], device="cuda", requires_grad=True),
        torch.tensor([[1.0]], device="cuda", requires_grad=True),
        torch.tensor([[0.1]], device="cuda", requires_grad=True),
    )
    adjacency, offsets = _adjacency(40)
    renderer = NativeHybridRenderer(
        max_batch_size=8,
        far=5.0,
        transmittance_threshold=0.0,
        gaussian_hit_buffer_size=64,
    )
    renderer.set_gaussian_geometry(gaussians)
    renderer.set_voronoi_geometry(medium, adjacency, offsets)
    rays = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.01, 0.02, 1.0],
            [0.1, -0.1, 0.0, -0.02, 0.01, 1.0],
        ],
        device="cuda",
    )

    hybrid = renderer.render(rays, gaussians=gaussians, voronoi=medium)
    medium_only = renderer.render(rays, voronoi=medium)
    loss = hybrid.image.mean() + 1.0e-4 * optical_thickness_loss(
        medium_only.transmittance
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert density.grad is not None
    assert torch.isfinite(density.grad).all()
    assert gaussians.opacity_logits.grad is not None
    assert torch.isfinite(gaussians.opacity_logits.grad).all()
