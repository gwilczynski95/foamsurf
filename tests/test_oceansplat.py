"""OceanSplat terms (trinocular, epipolar depth, residual, depth-aware opacity) and view-consistency helpers."""
import csv
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from foamsurf.types import GaussianScene
from train import fit_iui3, resolve_iui3
from training import oceansplat as oc
from training.config import ExperimentConfig, OptimizationConfig
from training.trainer import training_downscale
from foamsurf_model.gaussian_adaptation import GaussianAdaptationController
from foamsurf_model.gaussians import SurfaceGaussianModel
from training import view_consistency as vc
from training.config import ExperimentConfig, GaussianAdaptationConfig, ModelConfig
from helpers import ROOT, requires_gpu  # noqa: F401


def scene_of(means, log_scales, quaternions, opacity_logits, dtype=torch.float64):
    n = means.shape[0]
    return GaussianScene(means=means, log_scales=log_scales, quaternions=quaternions,
                         colors=torch.full((n, 3), 0.5, dtype=dtype), opacity_logits=opacity_logits,
                         raw_kappa=torch.zeros(n, 1, dtype=dtype))


def plane_depths(scene, rays, indices):
    """Exact ray / surfel-plane distances for the buffered indices (what OptiX returns)."""
    with torch.no_grad():
        _, _, n = oc.rotation_columns(scene.quaternions[indices.clamp_min(0).long()])
        means = scene.means[indices.clamp_min(0).long()]
        o, d = rays[None, :, :3], torch.nn.functional.normalize(rays[None, :, 3:], dim=-1)
        return (n * (means - o)).sum(-1) / (n * d).sum(-1)


def test_blended_depth_is_not_the_first_hit():
    # faint veil (alpha 0.1) at distance 1, opaque surfel at 5
    scene = scene_of(torch.tensor([[0.0, 0, 1], [0.0, 0, 5]], dtype=torch.float64),
                     torch.full((2, 2), math.log(100.0), dtype=torch.float64),
                     torch.tensor([[1.0, 0, 0, 0]] * 2, dtype=torch.float64),
                     torch.tensor([[math.log(0.1 / 0.9)], [20.0]], dtype=torch.float64))
    rays = torch.tensor([[0.0, 0, 0, 0, 0, 1]], dtype=torch.float64)
    indices = torch.tensor([[1], [0], [-1]], dtype=torch.int32)  # unsorted, one empty slot
    hits = torch.cat((plane_depths(scene, rays, indices[:2]), torch.tensor([[float("inf")]], dtype=torch.float64)))
    depth, accumulated = oc.blended_depth(scene, rays, hits, indices)
    assert depth.item() == pytest.approx(0.1 * 1 + 0.9 * 5, rel=1e-4)
    assert accumulated.item() == pytest.approx(1.0, abs=1e-4)


def test_blended_depth_gradient_matches_finite_differences():
    torch.manual_seed(0)
    rays = torch.tensor([[0.0, 0, 0, 0.05, -0.03, 1], [0.1, 0.0, 0, -0.02, 0.04, 1]], dtype=torch.float64)
    q = torch.nn.functional.normalize(torch.tensor([[1.0, 0.1, -0.05, 0.02], [1.0, -0.08, 0.06, 0.1]], dtype=torch.float64), dim=-1)
    indices = torch.tensor([[0, 0], [1, 1]], dtype=torch.int32)

    def depth_of(means, log_scales, quaternions, opacity_logits):
        scene = scene_of(means, log_scales, quaternions, opacity_logits)
        return oc.blended_depth(scene, rays, plane_depths(scene, rays, indices), indices)[0]

    inputs = (torch.tensor([[0.02, 0.01, 2.0], [0.0, -0.02, 4.0]], dtype=torch.float64, requires_grad=True),
              torch.tensor([[-1.0, -1.2], [-0.5, -0.7]], dtype=torch.float64, requires_grad=True),
              q.clone().requires_grad_(True),
              torch.tensor([[0.3], [1.5]], dtype=torch.float64, requires_grad=True))
    assert torch.autograd.gradcheck(depth_of, inputs, eps=1e-6, atol=1e-5)


def test_empty_slots_give_finite_gradients():
    scene = scene_of(torch.tensor([[0.0, 0, 3]], requires_grad=True), torch.zeros(1, 2), torch.tensor([[1.0, 0, 0, 0]]),
                     torch.tensor([[2.0]]), dtype=torch.float32)
    rays = torch.tensor([[0.0, 0, 0, 0, 0, 1]])
    depth, _ = oc.blended_depth(scene, rays, torch.tensor([[3.0], [float("inf")]]), torch.tensor([[0], [-1]], dtype=torch.int32))
    depth.sum().backward()
    assert torch.isfinite(scene.means.grad).all()


def texture(p):
    return torch.stack((0.5 + 0.3 * torch.sin(0.6 * p[:, 0]), 0.5 + 0.3 * torch.cos(0.5 * p[:, 1]),
                        0.4 + 0.1 * torch.sin(0.3 * (p[:, 0] + p[:, 1]))), -1)


class PlaneRenderer:
    """Opaque plane z = Z (Gaussian 0, a huge surfel) with a Lambertian texture; medium 0.05."""
    Z = 5.0

    def __init__(self, darken_origin=None):
        self.darken_origin = darken_origin

    def set_gaussian_geometry(self, scene):
        pass

    def render(self, rays, gaussians=None, voronoi=None, background=None):
        o, d = rays[:, :3], torch.nn.functional.normalize(rays[:, 3:], dim=-1)
        t = (self.Z - o[:, 2]) / d[:, 2]
        direct = texture(o + d * t[:, None])
        if self.darken_origin is not None and torch.allclose(o[0], self.darken_origin):
            direct = direct * 0.5  # something only this view sees
        return SimpleNamespace(direct=direct, backscatter=torch.full_like(direct, 0.05), hit_depths=t[None],
                               hit_indices=torch.zeros(1, len(t), dtype=torch.int32))


def plane_camera(size=(48, 64)):
    height, width = size
    return SimpleNamespace(origin=torch.zeros(3), right=torch.tensor([1.0, 0, 0]), down=torch.tensor([0.0, 1, 0]),
                           forward=torch.tensor([0.0, 0, 1]), width=width, height=height, fov_x=1.0, fov_y=1.0,
                           intrinsics=(40.0, 40.0, width / 2, height / 2))


def plane_scene(extra_means=()):
    means = torch.tensor([[0.0, 0.0, PlaneRenderer.Z], *extra_means], requires_grad=True)
    n = means.shape[0]
    return scene_of(means, torch.full((n, 2), math.log(1e3)), torch.tensor([[1.0, 0, 0, 0]] * n),
                    torch.full((n, 1), 20.0), dtype=torch.float32)


def run_terms(renderer, scene, baseline=0.6, seed=1):
    camera = plane_camera()
    target = renderer.render(oc.image_rays(camera, 1)).direct.reshape(48, 64, 3) + 0.05
    return oc.ocean_terms(renderer, lambda origin: scene, None, (0, 0, 0), camera, target, baseline, 1, 1e-3, 1.0,
                          1000, generator=torch.Generator().manual_seed(seed))


def test_consistent_scene_has_no_stereo_error_and_depth_gets_gradients():
    scene = plane_scene(extra_means=[[0.5, 0.3, PlaneRenderer.Z], [-0.4, -0.2, PlaneRenderer.Z]])
    terms = run_terms(PlaneRenderer(), scene)
    assert terms["obj"].item() < 2e-3 and terms["full"].item() < 2e-3
    assert terms["smooth"].item() < 1e-3  # constant disparity on a plane facing the camera
    assert terms["res"].item() < 1e-3 and terms["epi"].item() < 1e-3  # Gaussians on the surface
    (terms["obj"] + terms["full"]).backward()
    assert scene.means.grad is not None and torch.isfinite(scene.means.grad).all()


def test_view_specific_darkening_and_stray_gaussians_are_penalised():
    terms = run_terms(PlaneRenderer(darken_origin=torch.zeros(3)), plane_scene())
    assert terms["obj"].item() > 0.3 and terms["full"].item() > 0.3
    stray = run_terms(PlaneRenderer(), plane_scene(extra_means=[[0.2, 0.1, 2.0]]))
    assert stray["res"].item() > 1.0


def test_warp_direction_matches_disparity():
    # moving the camera right by b: a point at depth Z appears f b / Z pixels further left
    camera = plane_camera()
    renderer = PlaneRenderer()
    b, z, f = 0.5, PlaneRenderer.Z, 40.0
    shifted = renderer.render(oc.image_rays(camera, 1, torch.tensor([b, 0.0, 0.0]))).direct.reshape(48, 64, 3)
    central = renderer.render(oc.image_rays(camera, 1)).direct.reshape(48, 64, 3)
    ys, xs = torch.meshgrid(torch.arange(48.0) + 0.5, torch.arange(64.0) + 0.5, indexing="ij")
    warped, inside = oc.sample_image(shifted, xs - f * b / z, ys)
    assert (warped[inside] - central[inside]).abs().max().item() < 0.02


def test_smoothness_is_edge_aware():
    image = torch.zeros(8, 8, 3)
    mask = torch.ones(8, 8, dtype=torch.bool)
    assert oc.smoothness(torch.full((8, 8), 3.0), image, mask, 1.0).item() == 0.0
    step = torch.zeros(8, 8)
    step[:, 4:] = 1.0
    flat = oc.smoothness(step, image, mask, 1.0).item()
    edge_image = image.clone()
    edge_image[:, 4:] = 1.0
    assert 0 < oc.smoothness(step, edge_image, mask, 1.0).item() < flat


def test_depth_aware_opacity_blend_and_gradient():
    torch.manual_seed(0)
    mlp = oc.DepthAwareOpacity()
    scene = plane_scene(extra_means=[[0.3, 0.0, 2.0]])
    same = oc.adjusted_scene(scene, mlp, 0.0, torch.zeros(3), torch.tensor([0.0, 0, 1]), 10.0)
    assert same is scene
    blended = oc.adjusted_scene(scene, mlp, 0.5, torch.zeros(3), torch.tensor([0.0, 0, 1]), 10.0)
    opacity = torch.sigmoid(scene.opacity_logits)
    depth = scene.means[:, 2:] / 10.0
    direction = torch.nn.functional.normalize(scene.means, dim=-1)
    expected = 0.5 * opacity + 0.5 * mlp(opacity, depth, direction)
    torch.testing.assert_close(torch.sigmoid(blended.opacity_logits), expected.clamp(1e-6, 1 - 1e-6), rtol=1e-4, atol=1e-5)
    blended.opacity_logits.sum().backward()
    assert all(p.grad is not None for p in mlp.parameters())


def test_resolution_schedule_matches_oceansplat():
    optimization = OptimizationConfig(training_downscale_levels=2, training_downscale_interval=6000)
    assert [training_downscale(optimization, it) for it in (0, 5999, 6000, 11999, 12000, 30000)] == [4, 4, 2, 2, 1, 1]


def test_hidden_gaussians_pull_the_surface_back_unless_visible_is_set():
    # a Gaussian 3 units behind the opaque plane: without the visibility test L_epi and L_res pull the rendered
    # depth to it (Garden: ~50% of the Gaussians in L_epi are hidden and give ~90% of it); with visible > 0 it is skipped
    camera = plane_camera()
    renderer = PlaneRenderer()
    target = renderer.render(oc.image_rays(camera, 1)).direct.reshape(48, 64, 3) + 0.05
    scene = plane_scene(extra_means=[[0.2, 0.1, PlaneRenderer.Z + 3.0]])
    terms = lambda visible: oc.ocean_terms(renderer, lambda origin: scene, None, (0, 0, 0), camera, target, 0.6, 1,
                                           1e-3, 1.0, 1000, generator=torch.Generator().manual_seed(1), visible=visible)
    hidden, skipped = terms(0.0), terms(0.1)
    assert hidden["res"].item() > 1.0 and hidden["epi"].item() > 0.1
    assert skipped["res"].item() < 1e-3 and skipped["epi"].item() < 1e-3
    near = plane_scene(extra_means=[[0.2, 0.1, PlaneRenderer.Z * 1.05]])  # within the tolerance: still used
    kept = oc.ocean_terms(renderer, lambda origin: near, None, (0, 0, 0), camera, target, 0.6, 1, 1e-3, 1.0, 1000,
                          generator=torch.Generator().manual_seed(1), visible=0.1)
    assert kept["res"].item() > 0.05


def camera(origin=(0.0, 0.0, 0.0)):
    return SimpleNamespace(origin=torch.tensor(origin), forward=torch.tensor([0.0, 0.0, 1.0]),
                           right=torch.tensor([1.0, 0.0, 0.0]), down=torch.tensor([0.0, 1.0, 0.0]))


def surfels(means, scales, quaternions=None, opacity=0.5, raw_kappa=0.0):
    n = len(means)
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * n) if quaternions is None else torch.tensor(quaternions)
    logit = math.log(opacity / (1 - opacity))
    return SimpleNamespace(means=torch.tensor(means), log_scales=torch.log(torch.tensor(scales)),
                           quaternions=q, opacity_logits=torch.full((n, 1), logit),
                           raw_kappa=torch.full((n, 1), raw_kappa))


CAMERAS = torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])


DEPTHS = [0.55, 1.1, 5.0]


def controller(**overrides):
    points = np.array([[0.0, 0.0, z] for z in DEPTHS], dtype=np.float32)
    model = SurfaceGaussianModel(points, np.full_like(points, 0.5),
                                 ModelConfig(appearance="rgb", gaussian_initial_points=len(points)), "cpu")
    optimizer = torch.optim.Adam([{"params": [p], "name": f"gaussian_{n}", "lr": 1e-3}
                                  for n, p in model.named_primitive_parameters().items()])
    config = GaussianAdaptationConfig(**{"enabled": True, "max_points": 100, "near_camera_gradient_scaling": True,
                                         **overrides})
    return GaussianAdaptationController(model, optimizer, config, scene_extent=10.0, camera_centers=CAMERAS)


def test_near_camera_weight_is_squared_depth_over_camera_spread():
    weight = controller().near_camera_weight(camera(), iteration=10_000)  # no schedule: the weight alone
    torch.testing.assert_close(weight, torch.tensor([0.25, 1.0, 1.0]))


def test_near_camera_weight_paper_schedule():
    adaptation = controller(near_camera_steps=[4000, 7000])
    factor = lambda it: (adaptation.near_camera_weight(camera(), it) / torch.tensor([0.25, 1.0, 1.0])).tolist()
    assert factor(3999) == pytest.approx([1.0] * 3)
    assert factor(4000) == pytest.approx([0.5] * 3) and factor(6999) == pytest.approx([0.5] * 3)
    assert factor(7000) == pytest.approx([0.0] * 3)


def test_collect_statistics_applies_the_weight():
    adaptation = controller()
    adaptation.projected_radii = lambda cam: torch.ones(3)
    gradient = torch.tensor([[3.0, 4.0]] * 3)  # norm 5
    adaptation.collect_statistics(gradient, torch.zeros(3, 4), camera(), iteration=100)
    torch.testing.assert_close(adaptation.gradient_sum, torch.tensor([1.25, 5.0, 5.0]))
    plain = controller(near_camera_gradient_scaling=False)
    plain.projected_radii = lambda cam: torch.ones(3)
    plain.collect_statistics(gradient, torch.zeros(3, 4), camera(), iteration=100)
    torch.testing.assert_close(plain.gradient_sum, torch.full((3,), 5.0))


def test_nerfstudio_scale():
    origins = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 3.0], [2.0, 6.0, 3.0]])  # centred max |coord| 8/3
    assert vc.nerfstudio_scale(origins) == pytest.approx(3.0 / 8.0)


def test_hit_alpha_matches_the_renderer_formula():
    # surfel at z=2 facing the camera; hits at the centre, one x-scale off, and rotated 90 deg about z
    rotated = [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]
    scene = surfels([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]], [[0.5, 2.0], [0.5, 2.0]], [[1, 0, 0, 0], rotated],
                    opacity=0.8, raw_kappa=0.0)
    kappa = 1.0 + math.log(2.0)
    rays = torch.tensor([[0.0, 0, 0, 0, 0, 1], [0.5, 0, 0, 0, 0, 1], [0.5, 0, 0, 0, 0, 1]])
    alpha = vc.hit_alphas(scene, rays, torch.full((1, 3), 2.0), torch.tensor([[0, 0, 1]], dtype=torch.int32))
    expected = [0.8 * math.exp(-0.5 * 1e-12 ** kappa / kappa),
                0.8 * math.exp(-0.5 * 1.0 / kappa),               # x offset = 1 x-scale
                0.8 * math.exp(-0.5 * (0.5 / 2.0) ** (2 * kappa) / kappa)]  # rotated: the y-scale applies
    torch.testing.assert_close(alpha[0], torch.tensor(expected), rtol=1e-5, atol=1e-6)
    empty = vc.hit_alphas(scene, rays[:1], torch.full((1, 1), 2.0), torch.tensor([[-1]], dtype=torch.int32))
    assert empty.item() == 0.0


def test_blended_depth_sorts_hits_and_weights_by_transmittance():
    scene = surfels([[0.0, 0.0, 3.0], [0.0, 0.0, 1.0]], [[1.0, 1.0]] * 2, opacity=0.5)
    scene.opacity_logits[0] = 20.0  # back surfel opaque
    rays = torch.tensor([[0.0, 0, 0, 0, 0, 1]])
    depth, accumulated = vc.blended_depth(scene, rays, torch.tensor([[3.0], [1.0], [9.0]]),
                                          torch.tensor([[0], [1], [-1]], dtype=torch.int32))  # back first, one empty
    assert depth.item() == pytest.approx(2.0, abs=1e-4)  # weights 0.5 at 1, 0.5 at 3
    assert accumulated.item() == pytest.approx(1.0, abs=1e-4)


def test_virtual_rays_point_at_the_central_surface_point():
    rays = torch.tensor([[0.0, 0, 0, 0.3, -0.2, 1.0]])
    depth = torch.tensor([4.0])
    shifted = vc.virtual_rays(rays, depth, torch.tensor([0.7, 0.0, 0.0]))
    point = torch.nn.functional.normalize(rays[:, 3:], dim=-1) * 4.0
    origin, direction = shifted[:, :3], shifted[:, 3:]
    torch.testing.assert_close(origin, torch.tensor([[0.7, 0.0, 0.0]]))
    along = ((point - origin) * direction).sum(-1, keepdim=True)
    torch.testing.assert_close(origin + along * direction, point, rtol=1e-5, atol=1e-6)


def test_regularized_l1_stops_the_gradient_of_the_reference():
    prediction = torch.tensor([0.6], requires_grad=True)
    reference = torch.tensor([0.5], requires_grad=True)
    loss = vc.regularized_l1(prediction, torch.tensor([0.4]), reference, 0.5)
    assert loss.item() == pytest.approx(0.2)
    loss.backward()
    assert prediction.grad.item() == pytest.approx(1.0) and reference.grad is None


class StereoPlaneRenderer:
    """Lambertian textured plane z = 5 (colour from x, y) in front of a black medium: every view agrees."""
    def __init__(self):
        self.origins = []

    def set_gaussian_geometry(self, scene):
        pass

    def render(self, rays, gaussians=None, voronoi=None, background=None):
        o, d = rays[:, :3], rays[:, 3:]
        self.origins.append(o[0].clone())
        p = o + d * ((5.0 - o[:, 2:]) / d[:, 2:])
        return SimpleNamespace(direct=torch.stack((torch.sin(p[:, 0]), torch.cos(p[:, 1]), 0.5 + 0 * p[:, 0]), -1) * 0.4 + 0.5)


def test_trinocular_loss_is_zero_for_a_consistent_scene_and_shifts_along_camera_axes():
    renderer = StereoPlaneRenderer()
    gaussians = SimpleNamespace(render_scene=lambda origin: None)
    xy = torch.stack(torch.meshgrid(torch.linspace(-0.4, 0.4, 7), torch.linspace(-0.3, 0.3, 5), indexing="ij"), -1).reshape(-1, 2)
    rays = torch.cat((torch.zeros(len(xy), 3), torch.nn.functional.normalize(torch.cat((xy, torch.ones(len(xy), 1)), -1), dim=-1)), -1)
    depth = 5.0 / rays[:, 5]
    direct = renderer.render(rays).direct
    renderer.origins.clear()
    backscatter = torch.full_like(direct, 0.05)
    generator = torch.Generator().manual_seed(3)
    loss, values = vc.trinocular_loss(renderer, gaussians, None, (0, 0, 0), camera(), rays, depth, direct,
                                      backscatter, direct + backscatter, baseline=0.8, epsilon=1e-3,
                                      generator=generator)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)
    b_v = values["baseline"]
    assert abs(b_v) <= 0.8
    torch.testing.assert_close(renderer.origins[0], torch.tensor([1.5 * b_v, 0.0, 0.0]))
    torch.testing.assert_close(renderer.origins[1], torch.tensor([0.0, b_v, 0.0]))
    # a floater only the central view sees (darker central object): penalised
    loss, values = vc.trinocular_loss(renderer, gaussians, None, (0, 0, 0), camera(), rays, depth, direct * 0.5,
                                      backscatter, direct * 0.5 + backscatter, baseline=0.8, epsilon=1e-3)
    assert values["obj"] > 0.5 and values["full"] > 0.5
