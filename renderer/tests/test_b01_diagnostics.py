from __future__ import annotations

import pytest
import torch

from foamsurf import GaussianScene, NativeHybridRenderer, VoronoiScene


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="B0.1 diagnostics require CUDA"
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


def _scene(requires_grad: bool):
    torch.manual_seed(123)
    gaussian_values = (
        torch.tensor([[0.02, -0.01, 2.2]], device="cuda"),
        torch.tensor([[-0.8, -0.9]], device="cuda"),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda"),
        torch.tensor([[0.8, 0.2, 0.1]], device="cuda"),
        torch.tensor([[1.0]], device="cuda"),
        torch.tensor([[0.1]], device="cuda"),
    )
    sites = torch.randn(40, 3, device="cuda") * 1.5
    medium_values = (
        sites,
        torch.rand(40, 1, device="cuda") * 0.2 + 0.05,
        torch.rand(40, 3, device="cuda") * 0.8 + 0.1,
    )
    if requires_grad:
        gaussian_values = tuple(
            value.detach().clone().requires_grad_()
            for value in gaussian_values
        )
        medium_values = tuple(
            value.detach().clone().requires_grad_()
            for value in medium_values
        )
    return GaussianScene(*gaussian_values), VoronoiScene(*medium_values)


def _rays() -> torch.Tensor:
    return torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.01, 0.02, 1.0],
            [0.1, -0.1, 0.0, -0.02, 0.01, 1.0],
        ],
        device="cuda",
    )


def _parameters(gaussians, medium):
    return (
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


def test_statistics_on_off_preserves_rgb_alpha_and_every_gradient() -> None:
    _requirements()
    renderer = NativeHybridRenderer(
        max_batch_size=8, far=5.0, transmittance_threshold=0.0
    )
    rays = _rays()
    upstream_rgb = torch.tensor(
        [[0.4, -0.2, 0.7], [-0.3, 0.8, 0.1]], device="cuda"
    )
    upstream_alpha = torch.tensor([[0.2], [-0.1]], device="cuda")

    plain_gaussians, plain_medium = _scene(requires_grad=True)
    adjacency, offsets = _adjacency(plain_medium.sites.shape[0])
    renderer.set_gaussian_geometry(plain_gaussians)
    renderer.set_voronoi_geometry(plain_medium, adjacency, offsets)
    plain = renderer.render(
        rays, gaussians=plain_gaussians, voronoi=plain_medium
    )
    plain_loss = (
        (plain.image * upstream_rgb).sum()
        + (plain.alpha * upstream_alpha).sum()
    )
    plain_loss.backward()

    diagnostic_gaussians, diagnostic_medium = _scene(requires_grad=True)
    renderer.set_gaussian_geometry(diagnostic_gaussians)
    renderer.set_voronoi_geometry(
        diagnostic_medium, adjacency, offsets
    )
    diagnostic = renderer.render(
        rays,
        gaussians=diagnostic_gaussians,
        voronoi=diagnostic_medium,
        collect_diagnostics=True,
        diagnostic_target=torch.zeros_like(upstream_rgb),
    )
    diagnostic_loss = (
        (diagnostic.image * upstream_rgb).sum()
        + (diagnostic.alpha * upstream_alpha).sum()
    )
    diagnostic_loss.backward()

    torch.testing.assert_close(diagnostic.image, plain.image)
    torch.testing.assert_close(diagnostic.alpha, plain.alpha)
    for actual, expected in zip(
        _parameters(diagnostic_gaussians, diagnostic_medium),
        _parameters(plain_gaussians, plain_medium),
    ):
        torch.testing.assert_close(actual.grad, expected.grad)


def test_diagnostics_are_reduced_and_conserve_compositing_weight() -> None:
    _requirements()
    gaussians, medium = _scene(requires_grad=False)
    adjacency, offsets = _adjacency(medium.sites.shape[0])
    renderer = NativeHybridRenderer(
        max_batch_size=8, far=5.0, transmittance_threshold=0.0
    )
    renderer.set_gaussian_geometry(gaussians)
    renderer.set_voronoi_geometry(medium, adjacency, offsets)
    target = torch.zeros((2, 3), device="cuda")
    result = renderer.render(
        _rays(),
        gaussians=gaussians,
        voronoi=medium,
        collect_diagnostics=True,
        diagnostic_target=target,
    )
    diagnostics = result.diagnostics
    assert diagnostics is not None
    assert diagnostics.ray.shape == (2, 8)
    assert diagnostics.gaussian.shape == (1, 4)
    assert diagnostics.medium.shape == (40, 5)
    assert diagnostics.ray.numel() == 2 * 8
    assert diagnostics.gaussian.numel() == 1 * 4
    assert diagnostics.medium.numel() == 40 * 5

    torch.testing.assert_close(
        diagnostics.gaussian_weight.sum(),
        diagnostics.gaussian[:, 0].sum(),
        atol=2.0e-6,
        rtol=2.0e-6,
    )
    torch.testing.assert_close(
        diagnostics.medium_weight.sum(),
        diagnostics.medium[:, 0].sum(),
        atol=2.0e-6,
        rtol=2.0e-6,
    )
    torch.testing.assert_close(
        diagnostics.gaussian_weight
        + diagnostics.medium_weight
        + diagnostics.final_transmittance,
        torch.ones_like(diagnostics.final_transmittance),
        atol=2.0e-6,
        rtol=2.0e-6,
    )
    assert diagnostics.gaussian[:, 3].max() > 0
    assert diagnostics.medium[:, 4].max() > 0


class _BindingSpy:
    def __init__(self, wrapped) -> None:
        self.wrapped = wrapped
        self.diagnostic_calls = 0

    def hybrid_forward(self, *args, **kwargs):
        return self.wrapped.hybrid_forward(*args, **kwargs)

    def hybrid_backward(self, *args, **kwargs):
        return self.wrapped.hybrid_backward(*args, **kwargs)

    def hybrid_diagnostics(self, *args, **kwargs):
        self.diagnostic_calls += 1
        return self.wrapped.hybrid_diagnostics(*args, **kwargs)


def test_disabled_statistics_have_no_diagnostic_launch_or_output() -> None:
    _requirements()
    gaussians, medium = _scene(requires_grad=False)
    adjacency, offsets = _adjacency(medium.sites.shape[0])
    renderer = NativeHybridRenderer(
        max_batch_size=8, far=5.0, transmittance_threshold=0.0
    )
    renderer.set_gaussian_geometry(gaussians)
    renderer.set_voronoi_geometry(medium, adjacency, offsets)
    spy = _BindingSpy(renderer._bindings)
    renderer._bindings = spy

    result = renderer.render(
        _rays(), gaussians=gaussians, voronoi=medium
    )
    assert result.diagnostics is None
    assert spy.diagnostic_calls == 0

    result = renderer.render(
        _rays(),
        gaussians=gaussians,
        voronoi=medium,
        collect_diagnostics=True,
    )
    assert result.diagnostics is not None
    assert spy.diagnostic_calls == 1
