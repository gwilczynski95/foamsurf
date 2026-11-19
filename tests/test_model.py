"""Gaussians (surfels): SH colour, the colour ceiling, the CUDA colour kernel, the Mip filter, checkpoints."""
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from foamsurf_model.gaussians import SurfaceGaussianModel
from train import resolve_iui3
from training.config import ExperimentConfig, ModelConfig
from foamsurf_model.sh import C0, C1, C2, C3, evaluate_sh, sh_basis
import csv
from foamsurf_model.gaussian_adaptation import GaussianAdaptationController
from train import fit_iui3, resolve_iui3
from training.config import ExperimentConfig, GaussianAdaptationConfig, ModelConfig
from foamsurf_model.sh import evaluate_sh
from types import SimpleNamespace
from foamsurf_data.rays import generate_camera_rays, generate_indexed_rays
from training.sampler import RandomCameraRaySampler
from helpers import ROOT, requires_gpu  # noqa: F401


POINTS = np.array([[0, 0, 2.0], [1, 0, 3.0], [0, 1, 4.0]], dtype=np.float32)


RGB = np.array([[0.1, 0.5, 0.9], [0.7, 0.2, 0.3], [0.95, 0.99, 0.4]], dtype=np.float32)


def sh_model(appearance, degree=3):
    """Plain sigmoid SH colour (scale 1) on the shared points and colours."""
    config = ModelConfig(appearance=appearance, sh_degree=degree, gaussian_rgb_activation="sigmoid",
                         gaussian_initial_points=len(POINTS))
    return SurfaceGaussianModel(POINTS, RGB, config, "cpu")


def model(appearance, scale, device="cpu"):
    config = ModelConfig(appearance=appearance, sh_degree=3, gaussian_rgb_activation="sigmoid",
                         gaussian_rgb_scale=scale, gaussian_initial_points=len(POINTS))
    return SurfaceGaussianModel(POINTS, RGB, config, device)


@pytest.mark.parametrize("appearance", ["sh", "rgb"])
def test_starts_from_the_same_colours_and_the_ceiling_moves(appearance):
    for scale in (1.0, 1.5, 2.0):
        m = model(appearance, scale)
        torch.testing.assert_close(m.colors(torch.zeros(3)), torch.tensor(RGB), atol=1e-5, rtol=0)
        with torch.no_grad():
            m.appearance_dc.fill_(50.0)  # far into saturation
        top = m.colors(torch.tensor([2.0, -1.0, 0.0]))
        torch.testing.assert_close(top, torch.full_like(top, scale), atol=1e-4, rtol=0)  # bounded at the scale


def test_gradient_still_flows_near_one_with_a_lifted_ceiling():
    # at colour 0.99 the plain sigmoid is nearly flat; with scale 1.5 the same colour sits mid-curve
    plain, lifted = model("sh", 1.0), model("sh", 1.5)
    for m in (plain, lifted):
        m.colors(torch.zeros(3))[2, 1].backward()
    assert float(lifted.appearance_dc.grad[2, 1]) > 10 * float(plain.appearance_dc.grad[2, 1])


def test_checkpoint_keeps_the_scale_and_older_checkpoints_mean_one():
    m = model("sh", 1.5)
    with torch.no_grad():
        m.appearance_rest.normal_(0, 1.0)
    state = {k: v.detach().clone() for k, v in m.state_dict().items()}
    meta = m.checkpoint_metadata()
    assert meta["rgb_scale"] == 1.5
    loaded = SurfaceGaussianModel.from_checkpoint(state, meta, "cpu")
    origin = torch.tensor([1.0, 2.0, -3.0])
    torch.testing.assert_close(loaded.colors(origin), m.colors(origin))
    old = dict(meta); old.pop("rgb_scale")
    assert SurfaceGaussianModel.from_checkpoint(state, old, "cpu").rgb_scale == 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_cuda_kernel_matches_the_python_colours():
    m = model("sh", 1.5, "cuda")
    with torch.no_grad():
        m.appearance_rest.normal_(0, 1.0)
        m.appearance_dc.normal_(2.0, 3.0)
    origin = torch.tensor([1.0, 2.0, -3.0], device="cuda")
    with torch.no_grad():
        kernel = m.colors(origin)  # no grad on CUDA: the CUDA kernel path
    python = m.colors(origin)  # grad enabled: the PyTorch path
    torch.testing.assert_close(kernel, python.detach(), atol=1e-5, rtol=0)
    assert float(kernel.max()) > 1.0


def test_config_validation():
    config = ExperimentConfig()
    config.model.appearance, config.model.gaussian_rgb_activation = "sh", "sigmoid"
    config.model.gaussian_rgb_scale = 1.5
    config.dataset.source_path = "/tmp"
    config.validate()
    config.model.gaussian_rgb_scale = 0.8
    with pytest.raises(ValueError, match="gaussian_rgb_scale"):
        config.validate()
    config.model.gaussian_rgb_scale, config.model.gaussian_rgb_activation = 1.5, "identity"
    with pytest.raises(ValueError, match="gaussian_rgb_scale"):
        config.validate()
    assert ModelConfig().gaussian_rgb_scale == 1.0


def reference(degree, c, d):
    """The previous term-by-term implementation (3DGS convention)."""
    x, y, z = d[..., 0:1], d[..., 1:2], d[..., 2:3]
    r = C0 * c[..., 0]
    if degree > 0:
        r = r - C1 * y * c[..., 1] + C1 * z * c[..., 2] - C1 * x * c[..., 3]
    if degree > 1:
        xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
        r = (r + C2[0] * xy * c[..., 4] + C2[1] * yz * c[..., 5] + C2[2] * (2 * zz - xx - yy) * c[..., 6]
             + C2[3] * xz * c[..., 7] + C2[4] * (xx - yy) * c[..., 8])
    if degree > 2:
        r = (r + C3[0] * y * (3 * xx - yy) * c[..., 9] + C3[1] * xy * z * c[..., 10]
             + C3[2] * y * (4 * zz - xx - yy) * c[..., 11] + C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * c[..., 12]
             + C3[4] * x * (4 * zz - xx - yy) * c[..., 13] + C3[5] * z * (xx - yy) * c[..., 14]
             + C3[6] * x * (xx - 3 * yy) * c[..., 15])
    return r


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_values_and_gradients_match_the_term_by_term_formula(degree):
    torch.manual_seed(degree)
    coefficients = torch.randn(50, 3, 16, dtype=torch.float64, requires_grad=True)
    directions = torch.nn.functional.normalize(torch.randn(50, 3, dtype=torch.float64), dim=-1).requires_grad_(True)
    new = evaluate_sh(degree, coefficients, directions)
    old = reference(degree, coefficients, directions)
    torch.testing.assert_close(new, old)
    weights = torch.randn_like(new)
    grads_new = torch.autograd.grad((new * weights).sum(), (coefficients, directions), allow_unused=True)
    grads_old = torch.autograd.grad((old * weights).sum(), (coefficients, directions), allow_unused=True)
    for a, b, like in zip(grads_new, grads_old, (coefficients, directions)):  # degree 0: no direction gradient
        torch.testing.assert_close(torch.zeros_like(like) if a is None else a, torch.zeros_like(like) if b is None else b)


def test_basis_size_and_dc():
    d = torch.tensor([[0.0, 0.0, 1.0]])
    assert [sh_basis(k, d).shape[-1] for k in range(4)] == [1, 4, 9, 16]
    assert sh_basis(0, d).item() == pytest.approx(C0)


def test_sigmoid_sh_starts_with_the_rgb_colours_from_every_direction():
    sh, rgb = sh_model("sh"), sh_model("rgb")
    for origin in (torch.zeros(3), torch.tensor([5.0, -3.0, 1.0])):
        torch.testing.assert_close(sh.colors(origin), torch.tensor(RGB), atol=1e-5, rtol=0)
        torch.testing.assert_close(sh.colors(origin), rgb.colors(origin), atol=1e-5, rtol=0)
    assert sh.appearance_rest.shape == (3, 3, 15)


def test_sigmoid_sh_is_view_dependent_bounded_and_trainable():
    sh = sh_model("sh")
    with torch.no_grad():
        sh.appearance_rest.normal_(0, 2.0)
    a, b = sh.colors(torch.zeros(3)), sh.colors(torch.tensor([3.0, 3.0, 0.0]))
    assert (a - b).abs().max() > 0.05
    assert ((a > 0) & (a < 1)).all() and ((b > 0) & (b < 1)).all()
    a.sum().backward()
    assert sh.appearance_rest.grad.abs().sum() > 0 and sh.appearance_dc.grad.abs().sum() > 0
    sh.active_sh_degree = 0  # only the DC term: the colour no longer depends on the view
    torch.testing.assert_close(sh.colors(torch.zeros(3)), sh.colors(torch.tensor([3.0, 3.0, 0.0])))


def test_config_accepts_sigmoid_sh(tmp_path):
    config = ExperimentConfig()
    config.dataset.source_path = str(tmp_path)
    config.model.appearance, config.model.gaussian_rgb_activation = "sh", "sigmoid"
    config.validate()
    config.gaussian_adaptation.gradient_threshold_reference_width = 0
    with pytest.raises(ValueError, match="reference_width"):
        config.validate()


def adaptation(**settings):
    gaussians = sh_model("rgb")
    optimizer = torch.optim.Adam([{"params": [p], "name": f"gaussian_{n}", "lr": 1e-3}
                                  for n, p in gaussians.named_primitive_parameters().items()])
    config = GaussianAdaptationConfig(enabled=True, gradient_threshold=2e-4,
                                      **{k: v for k, v in settings.items() if k != "width"})
    return GaussianAdaptationController(gaussians, optimizer, config, 10.0, image_width=settings.get("width"))


def test_threshold_scales_with_the_image_width():
    assert adaptation().gradient_threshold == 2e-4
    assert adaptation(width=1794).gradient_threshold == 2e-4  # off by default
    assert adaptation(gradient_threshold_reference_width=1383, width=1383).gradient_threshold == pytest.approx(2e-4)
    assert adaptation(gradient_threshold_reference_width=1383, width=1794).gradient_threshold == pytest.approx(
        2e-4 * 1383 / 1794)
    with pytest.raises(ValueError, match="image width"):
        adaptation(gradient_threshold_reference_width=1383)


def test_threshold_decides_densification():
    controller = adaptation(gradient_threshold_reference_width=1000, width=2000)  # threshold 1e-4
    controller.gradient_sum[:] = torch.tensor([1.5e-4, 0.5e-4, 3e-4])
    controller.gradient_denom[:] = 1
    plan = controller.propose_plan(controller.config.densify_from + controller.config.densification_interval)
    chosen = set(plan.clone_indices.tolist()) | set(plan.split_indices.tolist())
    assert chosen == {0, 2}


gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")


@gpu
@pytest.mark.parametrize("degree", [0, 1, 2, 3])
@pytest.mark.parametrize("sigmoid", [True, False])
def test_sh_kernel_matches_python(degree, sigmoid):
    from foamsurf_model.sh_cuda import sh_colors
    g = torch.Generator(device="cuda").manual_seed(degree)
    n = 5000
    means = torch.randn(n, 3, device="cuda", generator=g) * 3
    dc = torch.randn(n, 3, 1, device="cuda", generator=g)
    rest = torch.randn(n, 3, 15, device="cuda", generator=g) * 0.3
    origin = torch.tensor([0.5, -1.0, 2.0], device="cuda")
    directions = torch.nn.functional.normalize(origin[None] - means, dim=-1)
    value = evaluate_sh(degree, torch.cat((dc, rest), -1)[..., :(degree + 1) ** 2], directions)
    expected = torch.sigmoid(value) if sigmoid else 0.5 + value
    torch.testing.assert_close(sh_colors(means, dc, rest, degree, origin, sigmoid), expected, atol=2e-6, rtol=1e-5)


@gpu
def test_model_colours_fast_only_without_gradients():
    import numpy as np
    from foamsurf_model.gaussians import SurfaceGaussianModel
    from training.config import ModelConfig
    config = ModelConfig(appearance="sh", sh_degree=3, gaussian_rgb_activation="sigmoid")
    rng = np.random.default_rng(0)
    model = SurfaceGaussianModel(rng.normal(size=(300, 3)).astype(np.float32), rng.random((300, 3)).astype(np.float32),
                                 config, "cuda")
    with torch.no_grad():
        model.appearance_rest.normal_(0, 0.2)
    origin = torch.tensor([1.0, 2.0, 3.0], device="cuda")
    with torch.no_grad():
        fast = model.colors(origin)
    slow = model.colors(origin)  # grad enabled: the Python path
    assert slow.requires_grad and not fast.requires_grad
    torch.testing.assert_close(fast, slow.detach(), atol=2e-6, rtol=1e-5)


@gpu
def test_geometry_upload_retries_once_after_freeing_the_cache(monkeypatch):
    """OptiX allocates outside PyTorch: a failed SetGeometry frees PyTorch's cache and is tried once more."""
    from renderer.foamsurf.cuda import NativeHybridRenderer
    from renderer.foamsurf.types import GaussianScene
    renderer = NativeHybridRenderer.__new__(NativeHybridRenderer)
    calls, freed = [], []

    class FlakyOptix:
        def SetGeometry(self, *tensors):
            calls.append(len(tensors))
            if len(calls) == 1:
                raise RuntimeError("Caught an unknown exception!")

    renderer._optix = FlakyOptix()
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: freed.append(True))
    n = 4
    scene = GaussianScene(means=torch.zeros(n, 3, device="cuda"), log_scales=torch.zeros(n, 2, device="cuda"),
                          quaternions=torch.tensor([[1.0, 0, 0, 0]] * n, device="cuda"),
                          colors=torch.zeros(n, 3, device="cuda"), opacity_logits=torch.zeros(n, 1, device="cuda"),
                          raw_kappa=torch.zeros(n, 1, device="cuda"))
    renderer.set_gaussian_geometry(scene)
    assert calls == [5, 5] and freed == [True] and renderer._gaussian_geometry_is_set
    calls.clear()
    renderer._optix = type("Broken", (), {"SetGeometry": lambda self, *t: (_ for _ in ()).throw(RuntimeError("x"))})()
    with pytest.raises(RuntimeError):
        renderer.set_gaussian_geometry(scene)  # a second failure is not hidden


FX, FY = 40.0, 50.0


def camera(width=8, height=6, origin=(0.0, 0.0, 0.0)):
    return SimpleNamespace(origin=torch.tensor(origin), right=torch.tensor([1.0, 0, 0]), down=torch.tensor([0.0, 1, 0]),
                           forward=torch.tensor([0.0, 0, 1]), width=width, height=height, fov_x=1.0, fov_y=1.0,
                           intrinsics=(FX, FY, width / 2, height / 2), name="c",
                           foreground=torch.rand(height, width, 3), alpha=torch.ones(height, width, 1))


def pixel_coordinates(directions, cam):
    """Continuous pixel coordinates (x, y) of ray directions; the centre of pixel (i, j) is (i + .5, j + .5)."""
    z = directions[:, 2]
    return FX * directions[:, 0] / z + cam.width / 2, FY * directions[:, 1] / z + cam.height / 2


def test_indexed_rays_with_offsets_match_a_shifted_principal_point():
    cam = camera()
    index = torch.arange(cam.width * cam.height)
    one = lambda t: torch.as_tensor(t, dtype=torch.float32)[None]
    args = (one(cam.origin), one(cam.right), one(cam.down), one(cam.forward), one(1.0), one(1.0), cam.width, cam.height)
    intrinsics = torch.tensor([[FX, FY, cam.width / 2, cam.height / 2]])
    centre = generate_indexed_rays(index, *args, intrinsics=intrinsics)
    assert torch.equal(centre.directions, generate_indexed_rays(index, *args, intrinsics=intrinsics, pixel_offsets=None).directions)
    shifted = generate_indexed_rays(index, *args, intrinsics=intrinsics,
                                    pixel_offsets=torch.tensor([[0.25, -0.4]]).expand(len(index), 2))
    reference = generate_camera_rays(cam.origin, cam.right, cam.down, cam.forward, cam.width, cam.height, 1.0, 1.0,
                                     intrinsics=(FX, FY, cam.width / 2 - 0.25, cam.height / 2 + 0.4))
    torch.testing.assert_close(shifted.directions, reference.directions.reshape(-1, 3))


class FakeDataset:
    device = "cpu"

    def __init__(self, cam):
        self.train = [cam]

    def stacked_training_data(self, pixels=False):
        c = self.train[0]
        return dict(origins=c.origin[None], rights=c.right[None], downs=c.down[None], forwards=c.forward[None],
                    fov_x=torch.tensor([1.0]), fov_y=torch.tensor([1.0]), intrinsics=torch.tensor([c.intrinsics]))


@pytest.mark.parametrize("jitter", [False, True])
def test_sampler_rays_stay_in_their_pixel(jitter):
    torch.manual_seed(0)
    cam = camera(width=32, height=24)
    sampler = RandomCameraRaySampler(FakeDataset(cam), 0, (0.0, 0.0, 0.0), full_image=True, jitter=jitter)
    batch = sampler.next()
    x, y = pixel_coordinates(batch.rays.directions, cam)
    i = torch.arange(cam.width).repeat(cam.height).float(); j = torch.arange(cam.height).repeat_interleave(cam.width).float()
    dx, dy = x - i, y - j
    assert dx.min() >= -1e-4 and dx.max() < 1 + 1e-4 and dy.min() >= -1e-4 and dy.max() < 1 + 1e-4
    if jitter:  # uniform over the pixel: spread out, centred on the pixel centre
        assert dx.std() > 0.25 and abs(float(dx.mean()) - 0.5) < 0.05 and abs(float(dy.mean()) - 0.5) < 0.05
    else:
        torch.testing.assert_close(dx, torch.full_like(dx, 0.5), atol=1e-4, rtol=0)
    torch.testing.assert_close(batch.target, cam.foreground.reshape(-1, 3))  # the pixel value, not resampled


@pytest.mark.parametrize("jitter", [False, True])
def test_downscaled_rays_stay_in_their_pixel(jitter):
    """The resolution schedule (_rs): rays of the 1/2 image, jittered inside its (2x larger) pixels."""
    torch.manual_seed(0)
    cam = camera(width=64, height=48)
    sampler = RandomCameraRaySampler(FakeDataset(cam), 0, (0.0, 0.0, 0.0), full_image=True, jitter=jitter)
    sampler.downscale = 2
    batch = sampler.next()
    assert batch.image_size == (24, 32)
    d = batch.rays.directions
    x = FX / 2 * d[:, 0] / d[:, 2] + 16; y = FY / 2 * d[:, 1] / d[:, 2] + 12
    i = torch.arange(32).repeat(24).float(); j = torch.arange(24).repeat_interleave(32).float()
    dx, dy = x - i, y - j
    assert dx.min() >= -1e-4 and dx.max() < 1 + 1e-4 and dy.min() >= -1e-4 and dy.max() < 1 + 1e-4
    if jitter:
        assert dx.std() > 0.25 and abs(float(dy.mean()) - 0.5) < 0.05
    else:
        torch.testing.assert_close(dx, torch.full_like(dx, 0.5), atol=1e-4, rtol=0)


def gaussians(means, log_scales):
    model = SurfaceGaussianModel(np.asarray(means, np.float32), np.full((len(means), 3), 0.5, np.float32),
                                 ModelConfig(appearance="rgb"), "cpu")
    with torch.no_grad():
        model.log_scales.copy_(torch.tensor(log_scales, dtype=torch.float32))
        model.opacity_logits.fill_(float(torch.logit(torch.tensor(0.9))))
    return model


def test_mip_filter_size_from_the_nearest_camera_that_sees_the_gaussian():
    g = gaussians([[0.0, 0.0, 4.0], [0.0, 0.0, -2.0], [100.0, 0.0, 1.0]], [[-3.0, -3.0]] * 3)
    near_cam, far_cam = camera(), camera(origin=(0.0, 0.0, -6.0))
    g.update_mip_filter([near_cam, far_cam], 0.2, near=0.1)
    s = float(np.sqrt(0.2))
    # Gaussian 0: both cameras see it, the nearer one (z = 4) wins; 1: behind near_cam, far_cam at z = 4;
    # 2: outside both frames -> the largest value.
    torch.testing.assert_close(g.mip_filter, torch.tensor([s * 4 / FX, s * 4 / FX, s * 4 / FX]))
    g2 = gaussians([[0.0, 0.0, 4.0], [0.0, 0.0, 8.0]], [[-3.0, -3.0]] * 2)
    g2.update_mip_filter([near_cam], 0.2, near=0.1)
    torch.testing.assert_close(g2.mip_filter, torch.tensor([s * 4 / FX, s * 8 / FX]))


def test_mip_filter_render_scene_keeps_the_integral_and_spares_large_gaussians():
    g = gaussians([[0.0, 0.0, 4.0], [0.1, 0.0, 4.0]], [[np.log(1e-3), np.log(2e-3)], [np.log(1.0), np.log(0.5)]])
    plain = g.render_scene(None)
    g.mip_filter = torch.tensor([0.01, 0.01])
    filtered = g.render_scene(None)
    s, f = plain.log_scales.exp(), filtered.log_scales.exp()
    torch.testing.assert_close(f, (s ** 2 + 0.01 ** 2).sqrt())
    o, of = torch.sigmoid(plain.opacity_logits), torch.sigmoid(filtered.opacity_logits)
    torch.testing.assert_close(of * f.prod(-1, keepdim=True), o * s.prod(-1, keepdim=True))  # integral kept
    assert float(of[0]) < 0.9 * 0.03  # a sub-filter Gaussian becomes a faint, filter-sized blob
    torch.testing.assert_close(f[1], s[1], rtol=1e-3, atol=0)  # 100x the filter: unchanged
    torch.testing.assert_close(of[1], o[1], rtol=1e-3, atol=0)
    (filtered.log_scales.sum() + filtered.opacity_logits.sum()).backward()
    assert g.log_scales.grad.abs().sum() > 0 and g.opacity_logits.grad.abs().sum() > 0
    g.mip_filter = torch.tensor([0.01])  # stale after a population change
    with pytest.raises(RuntimeError, match="stale"):
        g.render_scene(None)


def test_mip_filter_without_compensation_only_dilates():
    # as 3DGS "classic" / WaterSplatting: a sub-pixel surfel becomes pixel-sized and keeps its opacity
    g = gaussians([[0.0, 0.0, 4.0], [0.1, 0.0, 4.0]], [[np.log(1e-3), np.log(2e-3)], [np.log(1.0), np.log(0.5)]])
    plain = g.render_scene(None)
    g.mip_filter = torch.tensor([0.01, 0.01])
    g.mip_compensate = False
    dilated = g.render_scene(None)
    torch.testing.assert_close(dilated.log_scales.exp(), (plain.log_scales.exp() ** 2 + 0.01 ** 2).sqrt())
    torch.testing.assert_close(dilated.opacity_logits, plain.opacity_logits)  # opacity kept
    assert ExperimentConfig().model.gaussian_mip_compensate is True
