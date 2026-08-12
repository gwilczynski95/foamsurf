from __future__ import annotations

import math

import pytest
import torch

from foamsurf import GaussianScene, HybridRenderer, VoronoiScene


DTYPE = torch.float64


def _rays(x_coordinates: list[float]) -> torch.Tensor:
    rays = torch.zeros((len(x_coordinates), 6), dtype=DTYPE)
    rays[:, 0] = torch.tensor(x_coordinates, dtype=DTYPE)
    rays[:, 5] = 1.0
    return rays


def _gaussian(
    *,
    center: tuple[float, float, float] = (0.0, 0.0, 2.0),
    color: tuple[float, float, float] = (1.0, 0.0, 0.0),
    scale: float = 0.25,
    opacity_logit: float = 2.0,
    raw_kappa: float = 0.0,
) -> GaussianScene:
    return GaussianScene(
        means=torch.tensor([center], dtype=DTYPE),
        log_scales=torch.full((1, 2), math.log(scale), dtype=DTYPE),
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=DTYPE),
        colors=torch.tensor([color], dtype=DTYPE),
        opacity_logits=torch.tensor([[opacity_logit]], dtype=DTYPE),
        raw_kappa=torch.tensor([[raw_kappa]], dtype=DTYPE),
    )


def test_gaussian_only_has_expected_location_and_profile() -> None:
    renderer = HybridRenderer(near=0.0, far=4.0)
    scene = _gaussian()
    rays = _rays([0.0, 0.20, 1.0])

    result = renderer.render_gaussians(rays, scene)

    kappa = 1.0 + torch.nn.functional.softplus(scene.raw_kappa[0, 0])
    radii_squared = torch.tensor([0.0, 0.20**2 / 0.25**2], dtype=DTYPE)
    expected_alpha = torch.sigmoid(scene.opacity_logits[0, 0]) * torch.exp(
        -0.5 * radii_squared.pow(kappa) / kappa
    )
    torch.testing.assert_close(result.image[:2, 0], expected_alpha)
    torch.testing.assert_close(
        result.image[:2, 1:], torch.zeros((2, 2), dtype=DTYPE)
    )
    torch.testing.assert_close(result.image[2], torch.zeros(3, dtype=DTYPE))
    assert result.surface_index[:, 0].tolist() == [0, 0, -1]
    torch.testing.assert_close(
        result.surface_depth[:2], torch.full((2, 1), 2.0, dtype=DTYPE)
    )


def test_voronoi_only_selects_spatially_correct_cells() -> None:
    renderer = HybridRenderer(near=0.0, far=4.0)
    scene = VoronoiScene(
        sites=torch.tensor(
            [[-1.0, 0.0, 2.0], [1.0, 0.0, 2.0]], dtype=DTYPE
        ),
        densities=torch.tensor([[0.2], [0.4]], dtype=DTYPE),
        colors=torch.tensor(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=DTYPE
        ),
    )

    result = renderer.render_voronoi(_rays([-0.75, 0.75]), scene)

    left_alpha = 1.0 - math.exp(-0.2 * 4.0)
    right_alpha = 1.0 - math.exp(-0.4 * 4.0)
    torch.testing.assert_close(
        result.image,
        torch.tensor(
            [[0.0, 0.0, left_alpha], [0.0, right_alpha, 0.0]], dtype=DTYPE
        ),
    )
    torch.testing.assert_close(
        result.alpha[:, 0], torch.tensor([left_alpha, right_alpha], dtype=DTYPE)
    )


def test_hybrid_orders_medium_surface_and_medium_analytically() -> None:
    near, surface_depth, far = 0.0, 2.0, 5.0
    density = 0.3
    medium_color = torch.tensor([0.0, 0.0, 1.0], dtype=DTYPE)
    background = torch.tensor([0.1, 0.2, 0.3], dtype=DTYPE)
    gaussian = _gaussian(opacity_logit=1.5)
    medium = VoronoiScene(
        sites=torch.tensor([[0.0, 0.0, 2.0]], dtype=DTYPE),
        densities=torch.tensor([[density]], dtype=DTYPE),
        colors=medium_color[None],
    )
    renderer = HybridRenderer(near=near, far=far)

    result = renderer(
        _rays([0.0]),
        gaussians=gaussian,
        voronoi=medium,
        background=background,
    )

    surface_alpha = torch.sigmoid(gaussian.opacity_logits[0, 0])
    before_alpha = 1.0 - math.exp(-density * (surface_depth - near))
    before_t = math.exp(-density * (surface_depth - near))
    after_alpha = 1.0 - math.exp(-density * (far - surface_depth))
    final_t = before_t * (1.0 - surface_alpha) * (1.0 - after_alpha)
    expected = (
        before_alpha * medium_color
        + before_t * surface_alpha * gaussian.colors[0]
        + before_t * (1.0 - surface_alpha) * after_alpha * medium_color
        + final_t * background
    )
    torch.testing.assert_close(result.image[0], expected)
    torch.testing.assert_close(result.transmittance[0, 0], final_t)
    assert result.surface_index.item() == 0
    assert result.surface_depth.item() == pytest.approx(surface_depth)


def test_mode_helpers_equal_the_corresponding_general_mode() -> None:
    renderer = HybridRenderer(far=4.0)
    rays = _rays([-0.1, 0.1])
    gaussian = _gaussian()
    medium = VoronoiScene(
        sites=torch.tensor([[0.0, 0.0, 1.0]], dtype=DTYPE),
        densities=torch.tensor([[0.25]], dtype=DTYPE),
        colors=torch.tensor([[0.1, 0.2, 0.3]], dtype=DTYPE),
    )

    torch.testing.assert_close(
        renderer.render_gaussians(rays, gaussian).image,
        renderer(rays, gaussians=gaussian).image,
    )
    torch.testing.assert_close(
        renderer.render_voronoi(rays, medium).image,
        renderer(rays, voronoi=medium).image,
    )


def test_gaussian_parameters_pass_double_precision_gradcheck() -> None:
    renderer = HybridRenderer(far=5.0, epsilon=1.0e-12)
    rays = torch.tensor(
        [[0.13, -0.07, 0.0, 0.01, 0.02, 1.0]], dtype=DTYPE
    )
    parameters = (
        torch.tensor(
            [[0.02, -0.01, 2.1]], dtype=DTYPE, requires_grad=True
        ),
        torch.tensor([[-0.7, -0.9]], dtype=DTYPE, requires_grad=True),
        torch.tensor(
            [[0.96, 0.08, -0.12, 0.05]], dtype=DTYPE, requires_grad=True
        ),
        torch.tensor([[0.7, 0.2, 0.1]], dtype=DTYPE, requires_grad=True),
        torch.tensor([[1.2]], dtype=DTYPE, requires_grad=True),
        torch.tensor([[0.3]], dtype=DTYPE, requires_grad=True),
    )

    def render(*values: torch.Tensor) -> torch.Tensor:
        return renderer.render_gaussians(rays, GaussianScene(*values)).image

    assert torch.autograd.gradcheck(
        render, parameters, eps=1.0e-6, atol=2.0e-5, rtol=2.0e-4
    )


def test_voronoi_sites_density_and_color_pass_gradcheck() -> None:
    renderer = HybridRenderer(far=5.0, epsilon=1.0e-12)
    rays = torch.tensor(
        [[0.05, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=DTYPE
    )
    parameters = (
        torch.tensor(
            [[0.0, 0.0, 1.0], [0.2, 0.0, 3.2]],
            dtype=DTYPE,
            requires_grad=True,
        ),
        torch.tensor([[0.15], [0.45]], dtype=DTYPE, requires_grad=True),
        torch.tensor(
            [[0.1, 0.3, 0.7], [0.8, 0.2, 0.1]],
            dtype=DTYPE,
            requires_grad=True,
        ),
    )

    def render(*values: torch.Tensor) -> torch.Tensor:
        return renderer.render_voronoi(rays, VoronoiScene(*values)).image

    assert torch.autograd.gradcheck(
        render, parameters, eps=1.0e-6, atol=3.0e-5, rtol=3.0e-4
    )


def test_hybrid_backward_reaches_every_parameter_group() -> None:
    renderer = HybridRenderer(far=5.0)
    rays = torch.tensor(
        [[0.11, -0.04, 0.0, 0.02, 0.01, 1.0]], dtype=DTYPE
    )
    gaussian_values = [
        torch.tensor([[0.0, 0.0, 2.6]], dtype=DTYPE, requires_grad=True),
        torch.tensor([[-0.6, -0.8]], dtype=DTYPE, requires_grad=True),
        torch.tensor(
            [[0.97, 0.07, -0.1, 0.04]], dtype=DTYPE, requires_grad=True
        ),
        torch.tensor([[0.7, 0.4, 0.2]], dtype=DTYPE, requires_grad=True),
        torch.tensor([[1.0]], dtype=DTYPE, requires_grad=True),
        torch.tensor([[0.2]], dtype=DTYPE, requires_grad=True),
    ]
    voronoi_values = [
        torch.tensor(
            [[0.0, 0.0, 1.0], [0.2, 0.0, 3.3]],
            dtype=DTYPE,
            requires_grad=True,
        ),
        torch.tensor([[0.12], [0.5]], dtype=DTYPE, requires_grad=True),
        torch.tensor(
            [[0.1, 0.2, 0.8], [0.7, 0.2, 0.1]],
            dtype=DTYPE,
            requires_grad=True,
        ),
    ]
    result = renderer(
        rays,
        gaussians=GaussianScene(*gaussian_values),
        voronoi=VoronoiScene(*voronoi_values),
        background=(0.05, 0.1, 0.15),
    )
    weights = torch.tensor([[0.7, -0.3, 0.9]], dtype=DTYPE)
    (result.image * weights).sum().backward()

    for parameter in gaussian_values + voronoi_values:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0.0


def test_rejects_zero_direction_and_empty_voronoi_scene() -> None:
    renderer = HybridRenderer(far=2.0)
    zero_ray = torch.zeros((1, 6), dtype=DTYPE)
    with pytest.raises(ValueError, match="non-zero norm"):
        renderer.render_gaussians(zero_ray, _gaussian())

    empty = VoronoiScene(
        sites=torch.empty((0, 3), dtype=DTYPE),
        densities=torch.empty((0, 1), dtype=DTYPE),
        colors=torch.empty((0, 3), dtype=DTYPE),
    )
    with pytest.raises(ValueError, match="at least one site"):
        renderer.render_voronoi(_rays([0.0]), empty)
