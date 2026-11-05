"""Losses and priors: image loss, opacity schedule and binarization, seabed coverage, water clearing, backscatter chroma, water at the cameras."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from train import apply_overrides, fit_iui3, resolve_iui3
from training.config import ModelConfig
import math
import subprocess
import sys
import numpy as np
from train import fit_iui3, resolve_iui3
from training.losses import MediumFixConfig, surface_coverage_loss, water_clear_loss
import os
from training.config import ExperimentConfig
from training.losses import MediumFixConfig, camera_water_loss
from foamsurf_model.gaussian_adaptation import scheduled_opacity_threshold
from training.config import ExperimentConfig, GaussianAdaptationConfig
from training.trainer import training_downscale
from foamsurf_model.gaussian_adaptation import GaussianAdaptationController
from foamsurf_model.gaussians import SurfaceGaussianModel
from training.config import ExperimentConfig, GaussianAdaptationConfig, ModelConfig
from helpers import ROOT, requires_gpu  # noqa: F401


def _medium(activation="sigmoid"):
    from foamsurf_model.medium import VoronoiMediumModel

    config = ModelConfig(medium_type="rgb_backscatter", medium_points=32, medium_guard_points=0)
    medium = VoronoiMediumModel(torch.zeros(3), torch.ones(3), config, "cpu")
    medium.color_activation = activation
    with torch.no_grad():
        medium.raw_color.copy_(torch.randn_like(medium.raw_color))
    return medium


def test_medium_colors_follow_activation():
    medium = _medium("sigmoid")
    assert torch.allclose(medium.colors, torch.sigmoid(medium.raw_color))
    dc = _medium("dc")
    assert torch.allclose(dc.colors, (0.5 + 0.28209479177387814 * dc.raw_color).clamp_min(0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_gaussian_only_alpha_gradient_matches_finite_difference():
    """The coverage prior relies on gradients through the native alpha channel."""
    from foamsurf import GaussianScene, NativeHybridRenderer

    torch.manual_seed(0)
    device = torch.device("cuda")
    count = 6
    means = torch.zeros(count, 3, device=device)
    means[:, 2] = torch.linspace(2.0, 3.0, count, device=device)
    means[:, :2] = 0.05 * torch.randn(count, 2, device=device)
    logits = torch.linspace(-1.0, 1.0, count, device=device).unsqueeze(-1)
    scene_args = dict(means=means, log_scales=torch.full((count, 2), -1.0, device=device),
                      quaternions=torch.tensor([[1.0, 0, 0, 0]], device=device).expand(count, 4).contiguous(),
                      colors=torch.full((count, 3), 0.5, device=device),
                      raw_kappa=torch.zeros(count, 1, device=device))
    rays = torch.zeros(64, 6, device=device)
    rays[:, :2] = 0.1 * torch.randn(64, 2, device=device)
    rays[:, 5] = 1.0
    renderer = NativeHybridRenderer(max_batch_size=64, near=0.0, far=10.0, gaussian_hit_buffer_size=64)

    def loss_of(opacity_logits):
        scene = GaussianScene(opacity_logits=opacity_logits, **scene_args)
        renderer.set_gaussian_geometry(scene)
        alpha = renderer.render_gaussians(rays, scene).alpha
        return (alpha * (1 - alpha)).mean()

    parameter = logits.clone().requires_grad_(True)
    loss_of(parameter).backward()
    analytic = parameter.grad.flatten()
    numeric = torch.zeros(count, device=device)
    for index in range(count):
        step = torch.zeros_like(logits)
        step[index] = 1e-2
        with torch.no_grad():
            numeric[index] = (loss_of(logits + step) - loss_of(logits - step)) / 2e-2
    assert analytic.abs().max() > 1e-4
    assert torch.allclose(analytic, numeric, rtol=0.05, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_backscatter_chroma_loss_is_hue_only_and_ignores_open_water():
    import math
    from training.losses import backscatter_chroma_loss, water_direction

    water = torch.tensor([0.26, 0.32, 0.40])
    sand = torch.tensor([0.84, 0.79, 0.72])
    backscatter = torch.stack((0.2 * water, 0.5 * sand, 0.05 * sand, 0.3 * water)).requires_grad_(True)
    alpha = torch.tensor([0.9, 0.9, 0.9, 0.0])  # last ray: open water
    direction = water_direction(backscatter, alpha, 0.01)
    assert torch.allclose(direction, water / water.norm(), atol=1e-6)
    loss, angle = backscatter_chroma_loss(backscatter, alpha, direction, 0.5, 0.0)
    expected_sine2 = 1 - float((sand @ direction) / sand.norm()) ** 2
    weights = torch.tensor([0.2 * water.norm(), 0.5 * sand.norm(), 0.05 * sand.norm()])
    weights = weights / weights.mean()
    assert float(loss) == pytest.approx(float((weights[1:] * expected_sine2).sum() / 3), rel=1e-4)
    assert 0 < angle < 90
    loss.backward()
    gradient = backscatter.grad
    # Scale invariance: no gradient along each ray's own backscatter (cannot win by darkening).
    assert torch.allclose((gradient * backscatter.detach()).sum(-1), torch.zeros(4), atol=1e-6)
    assert gradient[0].abs().max() < 1e-6 and gradient[3].abs().max() < 1e-12  # water hue / unselected
    assert gradient[1].abs().max() > 0
    # Darkening sand-colored haze leaves the per-ray penalty unchanged.
    dim = torch.stack((0.2 * water, 0.1 * sand, 0.05 * sand, 0.3 * water))
    bright = torch.stack((0.2 * water, 0.5 * sand, 0.05 * sand, 0.3 * water))
    margin_loss = backscatter_chroma_loss(bright, alpha, direction, 0.5, 89.0)[0]
    assert float(margin_loss) == 0.0
    assert backscatter_chroma_loss(dim, alpha, None)[0].item() == 0.0
    assert math.isnan(backscatter_chroma_loss(dim, torch.zeros(4), direction)[1])


DEFAULT_RUN = "B0_chroma100_chthr02_scov3e3_cthr90_g700k_tden_water1e2_bk"


def test_surface_coverage_loss_is_one_sided():
    from training.losses import surface_coverage_loss

    alpha = torch.tensor([0.0, 0.3, 0.6, 0.9, 1.0], requires_grad=True)
    loss, fraction = surface_coverage_loss(alpha, 0.5)
    assert float(loss) == pytest.approx((0.4 + 0.1 + 0.0) / 3)
    assert fraction == pytest.approx(3 / 5)
    loss.backward()
    # Only rays above the threshold are pushed, and only upward (gradient of loss < 0).
    assert alpha.grad[:2].abs().max() == 0
    assert bool((alpha.grad[2:] < 0).all())
    empty, empty_fraction = surface_coverage_loss(torch.tensor([0.1, 0.2]), 0.5)
    assert float(empty) == 0 and empty_fraction == 0


def test_under_surface_loss_and_opaque_depth():
    from training.losses import under_surface_loss
    from training.surface_depth import opaque_depth

    backscatter = torch.tensor([[0.3, 0.3, 0.3], [0.0, 0.0, 0.0]], requires_grad=True)
    loss = under_surface_loss(backscatter)
    assert float(loss) == pytest.approx(0.15)
    loss.backward()
    assert bool((backscatter.grad > 0).all())
    assert float(under_surface_loss(torch.zeros(0, 3))) == 0
    inf = float("inf")
    # Ray 0: two 0.4 hits -> T = 0.6 then 0.36 -> opaque at the 2nd hit (depth 3).
    # Ray 1: single 0.9 hit at depth 5. Ray 2: never opaque.
    depth = torch.tensor([[2.0, 5.0, 1.0], [3.0, inf, inf], [inf, inf, inf]])
    alpha = torch.tensor([[0.4, 0.9, 0.2], [0.4, 0.0, 0.0], [0.0, 0.0, 0.0]])
    surface, found = opaque_depth(depth, alpha, 0.5)
    assert found.tolist() == [True, True, False]
    assert surface[:2].tolist() == [3.0, 5.0] and surface[2] == inf


def test_under_surface_passes_chunked_start_cells_to_renderer():
    """Regression: without start_points the renderer builds a [rays, sites, 3] tensor (OOM)."""
    from types import SimpleNamespace
    from training.colmap_joint_trainer import MediumFixTrainer
    from training.losses import MediumFixConfig

    torch.manual_seed(0)
    count = 300
    gaussians = SimpleNamespace(
        means=torch.tensor([[0.0, 0.0, 2.0]]), log_scales=torch.zeros(1, 2),
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]), opacity_logits=torch.full((1, 1), 10.0),
        raw_kappa=torch.zeros(1, 1))
    sites = torch.randn(50, 3) * 5
    calls = {}

    class Medium:
        raw_density = torch.zeros(50, 6, requires_grad=True)

        def render_scene(self):
            return SimpleNamespace(sites=sites)

    def render(rays, **kwargs):
        calls.update(kwargs, rays=rays)
        return SimpleNamespace(backscatter=torch.full((rays.shape[0], 3), 0.2))

    rays = torch.cat((torch.randn(count, 2) * 0.1, torch.zeros(count, 1), torch.zeros(count, 2), torch.ones(count, 1)), 1)
    only = SimpleNamespace(hit_depths=torch.full((1, count), 2.0), hit_indices=torch.zeros(1, count, dtype=torch.int32))
    trainer = SimpleNamespace(config=SimpleNamespace(renderer=SimpleNamespace(gaussian_radius_squared=11.3449)),
                              medium_fix=MediumFixConfig(under_surface_weight=1.0), renderer=SimpleNamespace(render=render))
    loss, fraction = MediumFixTrainer._under_surface(trainer, rays, only, SimpleNamespace(gaussians=gaussians, medium=Medium()))
    assert fraction == 1.0 and float(loss) == pytest.approx(0.2)
    start = calls["start_points"]
    assert start.dtype == torch.uint32 and start.shape == (count,)
    origins = calls["rays"][:, :3]
    assert torch.allclose(origins[:, 2], torch.full((count,), 2.0 + 0.01))  # just behind the surface
    expected = torch.cdist(origins, sites).argmin(1)
    assert torch.equal(start.long(), expected)


def test_colmap_model_dir_accepts_classic_and_flat_layouts(tmp_path):
    from foamsurf_data.colmap import colmap_model_dir
    from foamsurf_data.factory import detect_dataset_type

    flat = tmp_path / "flat"
    (flat / "sparse").mkdir(parents=True)
    for name in ("cameras.bin", "images.bin", "points3D.bin", "rigs.bin", "frames.bin"):
        (flat / "sparse" / name).write_bytes(b"")
    assert colmap_model_dir(flat) == flat / "sparse"
    assert detect_dataset_type(flat) == "colmap"
    classic = tmp_path / "classic"
    (classic / "sparse" / "0").mkdir(parents=True)
    for name in ("cameras.txt", "images.txt", "points3D.txt"):
        (classic / "sparse" / "0" / name).write_text("")
    (classic / "sparse" / "cameras.bin").write_bytes(b"")  # incomplete flat model is ignored
    assert colmap_model_dir(classic) == classic / "sparse" / "0"
    empty = tmp_path / "empty"
    (empty / "sparse").mkdir(parents=True)
    assert colmap_model_dir(empty) is None
    with pytest.raises(ValueError, match="sparse/0 or sparse"):
        detect_dataset_type(empty)


def test_water_clear_loss_pushes_only_rays_below_threshold():
    alpha = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.6, 0.95], requires_grad=True)
    loss, fraction = water_clear_loss(alpha, 0.3)
    assert loss.item() == pytest.approx(0.1) and fraction == pytest.approx(0.5)
    loss.backward()
    assert torch.allclose(alpha.grad, torch.tensor([1, 1, 1, 0, 0, 0]) / 3.0)
    # Mirror of the coverage loss: together with a higher coverage threshold no ray gets both.
    cover, _ = surface_coverage_loss(alpha, 0.9)
    assert cover.item() == pytest.approx(0.05)
    empty, none = water_clear_loss(torch.ones(4, requires_grad=True), 0.3)
    assert empty.item() == 0 and none == 0.0


def test_water_clear_schedule_and_coverage_end():
    config = MediumFixConfig(water_clear_weight=0.1, surface_coverage_weight=3e-3, surface_coverage_threshold=0.9,
                             surface_coverage_end=15000)
    assert config.water_clear_weight_at(3999) == 0
    assert config.water_clear_weight_at(5000) == pytest.approx(0.05)
    assert config.water_clear_weight_at(6000) == pytest.approx(0.1)
    assert config.surface_coverage_weight_at(14999) == pytest.approx(3e-3)
    assert config.surface_coverage_weight_at(15000) == 0
    assert MediumFixConfig().water_clear_weight == 0 and MediumFixConfig().surface_coverage_end is None


@pytest.mark.parametrize("bad", [
    dict(water_clear_weight=-1.0), dict(water_clear_threshold=0.0), dict(water_clear_ramp=-1),
    dict(surface_coverage_end=500), dict(surface_coverage_end=2.5e4),
    # Opposite pulls on the same rays.
    dict(water_clear_weight=0.1, surface_coverage_weight=3e-3, water_clear_threshold=0.5),
])
def test_water_clear_validation(bad):
    with pytest.raises(ValueError):
        MediumFixConfig(**bad)


gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")


def test_camera_water_loss_pulls_only_the_cells_at_the_cameras():
    coefficients = torch.full((6, 6), 0.05)
    coefficients[0] = 0.4  # dense cell at a camera
    coefficients[1] = 0.005  # clear cell at a camera
    coefficients = coefficients.requires_grad_(True)
    colour = torch.zeros(6, 3, requires_grad=True)
    near = torch.tensor([True, True, False, False, False, False])
    reference = torch.tensor([False, False, True, True, True, False])
    loss = camera_water_loss(coefficients, colour, near, reference)
    loss.backward()
    expected = ((torch.log(torch.tensor(0.4 / 0.05))) ** 2 + (torch.log(torch.tensor(0.005 / 0.05))) ** 2) / 2
    torch.testing.assert_close(loss.detach(), expected)
    assert coefficients.grad[0].min() > 0 and coefficients.grad[1].max() < 0  # towards the water, both ways
    assert coefficients.grad[2:].abs().sum() == 0  # the reference is detached, other cells untouched
    assert camera_water_loss(coefficients, colour, torch.zeros(6, dtype=torch.bool), reference) is None
    assert MediumFixConfig().camera_water_weight == 0.0  # off by default
    with pytest.raises(ValueError):
        MediumFixConfig(camera_water_radius=0.0)
    assert MediumFixConfig(camera_water_weight=1.0).camera_water_weight_at(500) == 0.0
    assert MediumFixConfig(camera_water_weight=1.0).camera_water_weight_at(2000) == 1.0


def test_camera_water_cells_near_cameras_in_free_water():
    from training.colmap_joint_trainer import MediumFixTrainer
    cameras = [SimpleNamespace(origin=torch.tensor([float(x), 0.0, 0.0])) for x in (0, 1, 2)]  # spacing 1
    sites = torch.tensor([[-0.5, 0.8, 0.0],   # near a camera, free water (1.7 from the Gaussian)
                          [1.0, 0.2, 0.0],    # near a camera but next to a Gaussian
                          [1.0, 10.0, 0.0],   # far free water: reference
                          [1.0, 2.5, 0.0]])   # between: neither
    gaussians = torch.tensor([[1.0, 0.0, 0.0], [5.0, -50.0, 0.0]])
    fake = SimpleNamespace(medium_fix=MediumFixConfig(camera_water_weight=0.01), _camera_water_cells=None,
                           dataset=SimpleNamespace(train=cameras))
    model = SimpleNamespace(medium=SimpleNamespace(sites=sites), gaussians=SimpleNamespace(means=gaussians))
    near, reference = MediumFixTrainer._camera_water_masks(fake, 0, model)
    assert near.tolist() == [True, False, False, False] and reference.tolist() == [False, False, True, False]
    model.medium.sites = sites * 100  # cached until the refresh interval
    assert MediumFixTrainer._camera_water_masks(fake, 100, model)[0].tolist() == [True, False, False, False]
    assert MediumFixTrainer._camera_water_masks(fake, 250, model)[0].tolist() == [False] * 4


IUI3 = Path("/shared/sets/datasets/SeathruNeRF_dataset/IUI3-RedSea-Undistorted")


def test_downscale_schedule_matches_watersplatting():
    off = SimpleNamespace(training_downscale_levels=0, training_downscale_interval=3000)
    ws = SimpleNamespace(training_downscale_levels=2, training_downscale_interval=3000)
    assert training_downscale(off, 0) == 1
    assert [training_downscale(ws, i) for i in (0, 2999, 3000, 5999, 6000, 30000)] == [4, 4, 2, 2, 1, 1]


@pytest.mark.skipif(not IUI3.is_dir(), reason="dataset not on this machine")
def test_downscaled_batch_is_the_whole_image_at_lower_resolution():
    import torch.nn.functional as F
    from foamsurf_data.colmap import ColmapDataset
    from training.sampler import RandomCameraRaySampler
    dataset = ColmapDataset(str(IUI3), resolution=8, device="cpu", llffhold=8)
    sampler = RandomCameraRaySampler(dataset, 1, (0.0, 0.0, 0.0), full_image=True, cameras_without_replacement=True)
    full = sampler.next()
    camera = full.camera
    sampler._camera_stack = [dataset.train.index(camera)]  # same camera again
    sampler.downscale = 2
    batch = sampler.next()
    height, width = camera.height // 2, camera.width // 2
    assert batch.image_size == (height, width) and batch.rays.directions.shape == (height * width, 3)
    expected = F.interpolate(camera.foreground.permute(2, 0, 1)[None], size=(height, width), mode="bilinear",
                             antialias=True, align_corners=False)[0].permute(1, 2, 0).reshape(-1, 3)
    torch.testing.assert_close(batch.target, expected)
    # Pixel (0, 0) at half resolution covers full-resolution pixels (0..1, 0..1): its ray lies between theirs.
    grid = full.rays.directions.reshape(camera.height, camera.width, 3)
    corner = torch.nn.functional.normalize(grid[:2, :2].reshape(-1, 3).mean(0), dim=0)
    assert torch.allclose(batch.rays.directions[0], corner, atol=1e-3)
    sampler.full_image = False
    with pytest.raises(ValueError, match="full-image"):
        sampler.next()


def test_late_opacity_threshold():
    config = GaussianAdaptationConfig(min_opacity=0.5, min_opacity_initial=0.5, opacity_prune_ramp=0,
                                      min_opacity_late=0.1, min_opacity_late_from=10000)
    assert scheduled_opacity_threshold(config, 9999) == 0.5 and scheduled_opacity_threshold(config, 10000) == 0.1
    assert scheduled_opacity_threshold(GaussianAdaptationConfig(min_opacity=0.1), 20000) == 0.1


def test_config_validation(tmp_path):
    config = ExperimentConfig()
    config.dataset.source_path = str(tmp_path)
    config.optimization.training_downscale_levels = 2
    with pytest.raises(ValueError, match="full_image_batch"):
        config.validate()
    config.optimization.full_image_batch = True
    config.validate()
    config.gaussian_adaptation.min_opacity_late = 1.5
    with pytest.raises(ValueError, match="min_opacity_late"):
        config.validate()


POINTS = np.array([[0, 0, 1], [0, 0, 10], [0, 0, 100]], dtype=np.float32)


def controller(**settings):
    model = SurfaceGaussianModel(POINTS, np.full_like(POINTS, 0.5),
                                 ModelConfig(appearance="rgb", gaussian_initial_points=len(POINTS)), "cpu")
    optimizer = torch.optim.Adam([{"params": [p], "name": f"gaussian_{n}", "lr": 1e-3}
                                  for n, p in model.named_primitive_parameters().items()])
    return GaussianAdaptationController(model, optimizer, GaussianAdaptationConfig(enabled=True, **settings), 10.0)


def test_compensation_is_inverse_mean_transmittance_capped():
    adaptation = controller(gradient_attenuation_compensation=True, gradient_compensation_max=20.0)
    adaptation.medium_attenuation = torch.tensor([0.1, 0.05, 0.02])
    factor = adaptation.attenuation_compensation(torch.zeros(3))
    for distance, value in zip((1.0, 10.0), factor[:2].tolist()):
        expected = 1 / np.mean(np.exp(-distance * np.array([0.1, 0.05, 0.02])))
        assert value == pytest.approx(expected, rel=1e-5)
    assert factor[2].item() == pytest.approx(20.0)  # 100 units of water: capped
    adaptation.medium_attenuation = None
    with pytest.raises(RuntimeError, match="medium_attenuation"):
        adaptation.attenuation_compensation(torch.zeros(3))


@pytest.mark.parametrize("compensate", [False, True])
def test_collect_statistics_scales_the_gradient(compensate):
    adaptation = controller(gradient_attenuation_compensation=compensate)
    adaptation.medium_attenuation = torch.tensor([0.05, 0.05, 0.05])
    adaptation.projected_radii = lambda camera: torch.ones(3)  # all visible
    gradient = torch.ones(3, 2)
    camera = SimpleNamespace(origin=torch.zeros(3))
    adaptation.collect_statistics(gradient, torch.zeros(3, 4), camera)
    expected = np.sqrt(2) * (np.exp(0.05 * np.array([1.0, 10.0, 100.0])).clip(max=20.0) if compensate else 1.0)
    assert np.allclose(adaptation.gradient_sum.numpy(), expected, rtol=1e-5)
    assert adaptation.gradient_denom.tolist() == [1, 1, 1]


def test_water_attenuation_is_median_over_non_guard_cells():
    from training.trainer import Trainer
    raw = torch.tensor([[0.1, 0.2, 0.3, 9, 9, 9], [0.2, 0.4, 0.6, 9, 9, 9], [0.3, 0.6, 0.9, 9, 9, 9],
                        [50.0, 50.0, 50.0, 9, 9, 9]])
    medium = SimpleNamespace(optical_coefficients=raw, is_rgb=True,
                             guard_mask=torch.tensor([False, False, False, True]))
    fake = SimpleNamespace(model=SimpleNamespace(medium=medium))
    assert torch.allclose(Trainer._water_attenuation(fake), torch.tensor([0.2, 0.4, 0.6]))


def test_compensation_config(tmp_path):
    assert GaussianAdaptationConfig().gradient_attenuation_compensation is False
    config = ExperimentConfig()
    config.dataset.source_path = str(tmp_path)
    config.gaussian_adaptation.gradient_compensation_max = 0.5
    with pytest.raises(ValueError, match="gradient_compensation_max"):
        config.validate()


