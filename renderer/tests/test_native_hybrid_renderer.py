from __future__ import annotations

import pytest
import torch

from foamsurf import (
    GaussianScene,
    HybridRenderer,
    NativeHybridRenderer,
    VoronoiScene,
)


_SHARED_NATIVE = None


def _native_renderer() -> NativeHybridRenderer:
    global _SHARED_NATIVE
    if _SHARED_NATIVE is None:
        _SHARED_NATIVE = NativeHybridRenderer(
            max_batch_size=16, far=5.0, transmittance_threshold=0.0
        )
    return _SHARED_NATIVE


def _requirements():
    if not torch.cuda.is_available():
        pytest.skip("native hybrid renderer requires CUDA")
    try:
        import foamsurf._C as bindings
    except ImportError as error:
        pytest.skip(f"FoamSurf native bindings are unavailable: {error}")
    return bindings


def _gaussians(requires_grad: bool = False) -> GaussianScene:
    values = (
        torch.tensor([[0.04, -0.03, 2.4]], device="cuda"),
        torch.tensor([[-0.9, -1.1]], device="cuda"),
        torch.tensor([[0.97, 0.08, -0.12, 0.05]], device="cuda"),
        torch.tensor([[0.8, 0.2, 0.1]], device="cuda"),
        torch.tensor([[1.2]], device="cuda"),
        torch.tensor([[0.2]], device="cuda"),
    )
    if requires_grad:
        values = tuple(value.requires_grad_() for value in values)
    return GaussianScene(*values)


def _complete_adjacency(point_count: int, device: torch.device):
    indices = torch.arange(point_count, device=device)
    neighbors = indices.repeat(point_count, 1)
    neighbors = neighbors[neighbors != indices[:, None]].to(torch.uint32)
    offsets = torch.arange(
        0, point_count * (point_count - 1) + 1, point_count - 1,
        device=device,
    ).to(torch.uint32)
    return neighbors.contiguous(), offsets.contiguous()

def _medium(requires_grad: bool = False):
    _requirements()
    torch.manual_seed(19)
    sites = torch.randn(48, 3, device="cuda") * 2.0
    densities = torch.rand(48, 1, device="cuda") * 0.25 + 0.05
    colors = torch.rand(48, 3, device="cuda") * 0.7 + 0.1
    if requires_grad:
        sites = sites.detach().requires_grad_()
        densities.requires_grad_()
        colors.requires_grad_()
    return (
        VoronoiScene(sites, densities, colors),
        *_complete_adjacency(sites.size(0), sites.device),
    )


def _rays() -> torch.Tensor:
    return torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.03, 0.02, 1.0],
            [0.1, -0.08, 0.0, -0.02, 0.01, 1.0],
        ],
        device="cuda",
    )


def test_native_pure_modes_match_their_reference_renderers() -> None:
    _requirements()
    rays = _rays()
    gaussians = _gaussians()
    medium, adjacency, offsets = _medium()
    native = _native_renderer()
    native.set_gaussian_geometry(gaussians)
    native.set_voronoi_geometry(medium, adjacency, offsets)
    oracle = HybridRenderer(far=5.0)

    gaussian_result = native.render_gaussians(
        rays, gaussians, background=(0.05, 0.1, 0.15)
    )
    gaussian_reference = oracle.render_gaussians(
        rays, gaussians, background=(0.05, 0.1, 0.15)
    )
    torch.testing.assert_close(
        gaussian_result.image, gaussian_reference.image, atol=2.0e-6, rtol=2.0e-6
    )
    torch.testing.assert_close(
        gaussian_result.alpha, gaussian_reference.alpha, atol=2.0e-6, rtol=2.0e-6
    )

    medium_result = native.render_voronoi(rays, medium)
    medium_reference = oracle.render_voronoi(rays, medium)
    torch.testing.assert_close(
        medium_result.image, medium_reference.image, atol=2.0e-6, rtol=2.0e-6
    )
    torch.testing.assert_close(
        medium_result.alpha, medium_reference.alpha, atol=2.0e-6, rtol=2.0e-6
    )


def test_native_hybrid_forward_and_every_gradient_match_oracle() -> None:
    _requirements()
    rays = _rays()
    gaussians = _gaussians(requires_grad=True)
    medium, adjacency, offsets = _medium(requires_grad=True)
    native = _native_renderer()
    native.set_gaussian_geometry(gaussians)
    native.set_voronoi_geometry(medium, adjacency, offsets)
    background = (0.05, 0.1, 0.15)
    result = native.render(
        rays, gaussians=gaussians, voronoi=medium, background=background
    )

    gaussian_copies = [
        value.detach().clone().requires_grad_()
        for value in (
            gaussians.means,
            gaussians.log_scales,
            gaussians.quaternions,
            gaussians.colors,
            gaussians.opacity_logits,
            gaussians.raw_kappa,
        )
    ]
    medium_copies = [
        value.detach().clone().requires_grad_()
        for value in (medium.sites, medium.densities, medium.colors)
    ]
    reference = HybridRenderer(far=5.0)(
        rays,
        gaussians=GaussianScene(*gaussian_copies),
        voronoi=VoronoiScene(*medium_copies),
        background=background,
    )
    torch.testing.assert_close(result.image, reference.image, atol=2.0e-6, rtol=3.0e-6)
    torch.testing.assert_close(result.alpha, reference.alpha, atol=2.0e-6, rtol=3.0e-6)

    image_gradient = torch.tensor(
        [[0.4, -0.2, 0.7], [-0.3, 0.8, 0.1]], device="cuda"
    )
    alpha_gradient = torch.tensor([[0.3], [-0.2]], device="cuda")
    native_loss = (result.image * image_gradient).sum()
    native_loss = native_loss + (result.alpha * alpha_gradient).sum()
    native_loss.backward()
    reference_loss = (reference.image * image_gradient).sum()
    reference_loss = reference_loss + (reference.alpha * alpha_gradient).sum()
    reference_loss.backward()

    native_parameters = (
        medium.sites,
        medium.densities,
        medium.colors,
        gaussians.means,
        gaussians.log_scales,
        gaussians.quaternions,
        gaussians.colors,
        gaussians.opacity_logits,
        gaussians.raw_kappa,
    )
    reference_parameters = (*medium_copies, *gaussian_copies)
    for native_parameter, reference_parameter in zip(
        native_parameters, reference_parameters
    ):
        torch.testing.assert_close(
            native_parameter.grad,
            reference_parameter.grad,
            atol=3.0e-6,
            rtol=2.0e-4,
        )


def test_native_renderer_requires_explicit_geometry_sync() -> None:
    _requirements()
    native = NativeHybridRenderer(max_batch_size=4, far=5.0)
    with pytest.raises(RuntimeError, match="set_gaussian_geometry"):
        native.render_gaussians(_rays(), _gaussians())
    medium, _, _ = _medium()
    with pytest.raises(RuntimeError, match="set_voronoi_geometry"):
        native.render_voronoi(_rays(), medium)
