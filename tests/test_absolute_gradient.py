"""Absolute (AbsGS / WaterSplatting) densification gradient: kernel buffer, screen scale, '_abs' sweep set."""
import csv
import math
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from foamsurf_model.gaussians import SurfaceGaussianModel
from train import fit_iui3, resolve_iui3
from training.config import ModelConfig

ROOT = Path(__file__).resolve().parents[1]
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")


def overrides(variant):
    script = (ROOT / "scripts/train_seathru.sbatch").read_text()
    start = script.index("variant_overrides() {")
    function = script[start:script.index("\n}\n", start) + 3]
    return subprocess.run(["bash", "-c", function + f"\nvariant_overrides {variant}"],
                          capture_output=True, text=True, check=True).stdout.splitlines()


def camera():
    right = torch.tensor([0.8, 0.6, 0.0])
    down = torch.tensor([-0.6, 0.8, 0.0])
    return SimpleNamespace(origin=torch.tensor([0.1, -0.2, -1.0]), right=right, down=down,
                           forward=torch.linalg.cross(right, down), fov_x=0.9, fov_y=0.6)


def test_world_per_viewport_is_the_screen_points_chain_rule():
    points = np.array([[0, 0, 2], [0.5, -0.3, 4], [-1, 1, 7]], dtype=np.float32)
    model = SurfaceGaussianModel(points, np.full_like(points, 0.5),
                                 ModelConfig(appearance="rgb", gaussian_initial_points=3), "cpu")
    view = camera()
    scene, screen_points = model.training_scene(view)
    weights = torch.randn(3, 3)
    (scene.means * weights).sum().backward()  # dL/dmean = weights
    scale = model.world_per_viewport(view)
    expected = torch.stack(((weights @ view.right) * scale[:, 0], (weights @ view.down) * scale[:, 1]), -1)
    torch.testing.assert_close(screen_points.grad, expected)
    depth = (torch.from_numpy(points) - view.origin) @ view.forward
    torch.testing.assert_close(scale[:, 1] / scale[:, 0], torch.full((3,), math.tan(0.3) / math.tan(0.45)))
    torch.testing.assert_close(scale[:, 0], depth * math.tan(0.45))


def test_abs_set_overrides(tmp_path):
    m = resolve_iui3(ROOT / "configs/iui3_default.yaml", tmp_path / "r", overrides=overrides("_abs6_g1200k"))
    a = m["joint"]["gaussian_adaptation"]
    assert a["absolute_gradient"] is True and a["gradient_threshold"] == 6e-4 and a["max_points"] == 1_200_000
    base = resolve_iui3(ROOT / "configs/iui3_default.yaml", tmp_path / "r", overrides=[])
    assert not base["joint"]["gaussian_adaptation"].get("absolute_gradient", False)  # off unless asked
    run = subprocess.run(["bash", "scripts/train_seathru.sbatch"], cwd=ROOT, capture_output=True, text=True,
                         env={"PATH": "/usr/bin:/bin", "FS_VARIANT_SET": "abs", "SLURM_ARRAY_TASK_ID": "24",
                              "SLURM_ARRAY_JOB_ID": "test"})
    assert "out of range" in run.stderr  # 3 variants x 4 scenes x 2 seeds = tasks 0-23


def _scene_and_rays():
    from foamsurf import GaussianScene
    from foamsurf.types import RGBVoronoiScene
    torch.manual_seed(21)
    sites = torch.randn(8, 3, device="cuda") * 2
    a = torch.rand(8, 3, device="cuda") * .4 + .01
    b = torch.rand(8, 3, device="cuda") * .3 + .02
    medium = RGBVoronoiScene(sites, a.mean(-1, keepdim=True), torch.rand(8, 3, device="cuda"), a, b)
    torch.manual_seed(7)
    count = 40
    means = torch.randn(count, 3, device="cuda") * .6 + torch.tensor([0., 0., 2.5], device="cuda")
    values = (means, torch.rand(count, 2, device="cuda") * .6 - 1.2, torch.randn(count, 4, device="cuda"),
              torch.rand(count, 3, device="cuda"), torch.randn(count, 1, device="cuda") + 1.,
              torch.randn(count, 1, device="cuda") * .3)
    gaussians = GaussianScene(*[v.contiguous().requires_grad_() for v in values])
    torch.manual_seed(3)
    directions = torch.randn(96, 3, device="cuda") * torch.tensor([.3, .3, 1.], device="cuda")
    directions[:, 2] = 1.0
    rays = torch.cat((torch.zeros_like(directions), directions), -1)
    return medium, gaussians, rays


def _renderer(medium, gaussians):
    from foamsurf import NativeHybridRenderer
    renderer = NativeHybridRenderer(max_batch_size=4096, far=5., transmittance_threshold=0.)
    n = len(medium.sites)
    ids = torch.arange(n, device="cuda")
    adjacency = ids.repeat(n, 1)
    adjacency = adjacency[adjacency != ids[:, None]].to(torch.uint32).contiguous()
    offsets = (torch.arange(n + 1, device="cuda") * (n - 1)).to(torch.uint32)
    renderer.set_voronoi_geometry(medium, adjacency, offsets)
    renderer.set_gaussian_geometry(gaussians)
    return renderer


AXES = torch.tensor([[0.8, 0.6, 0.0], [-0.6, 0.8, 0.0]])


@requires_cuda
def test_kernel_buffer_is_sum_over_rays_of_absolute_projected_mean_gradient():
    medium, gaussians, rays = _scene_and_rays()
    renderer = _renderer(medium, gaussians)
    axes = AXES.cuda()
    weights = torch.randn(len(rays), 3, device="cuda")
    buffer = torch.zeros(len(gaussians.means), 2, device="cuda")
    out = renderer.render(rays, gaussians, medium, background=(.1, .2, .3), screen_absolute=(axes, buffer))
    (plain,) = torch.autograd.grad((out.image * weights).sum(), gaussians.means)
    # Oracle: one backward per ray (only that ray's output gradient is non-zero).
    out = renderer.render(rays, gaussians, medium, background=(.1, .2, .3))
    expected = torch.zeros_like(buffer)
    for ray in range(len(rays)):
        (per_ray,) = torch.autograd.grad((out.image[ray] * weights[ray]).sum(), gaussians.means, retain_graph=True)
        expected += (per_ray @ axes.T).abs()
    assert expected.sum() > 0 and (expected > 0).sum() >= 5  # rays really hit several Gaussians
    torch.testing.assert_close(buffer, expected, atol=1e-6, rtol=1e-4)
    assert ((plain @ axes.T).abs() <= buffer + 1e-6).all()  # |sum| <= sum |.|


@requires_cuda
def test_buffer_leaves_every_gradient_unchanged_and_needs_the_fused_backend():
    medium, gaussians, rays = _scene_and_rays()
    medium = replace(medium, attenuation=medium.attenuation.clone().requires_grad_())
    renderer = _renderer(medium, gaussians)
    params = [medium.attenuation, gaussians.means, gaussians.log_scales, gaussians.quaternions,
              gaussians.colors, gaussians.opacity_logits, gaussians.raw_kappa]
    weights = torch.randn(len(rays), 3, device="cuda")
    grads = []
    for screen in (None, (AXES.cuda(), torch.zeros(len(gaussians.means), 2, device="cuda"))):
        out = renderer.render(rays, gaussians, medium, background=(.1, .2, .3), screen_absolute=screen)
        grads.append(torch.autograd.grad((out.image * weights).sum(), params))
    for plain, with_buffer in zip(*grads):
        torch.testing.assert_close(plain, with_buffer, atol=1e-6, rtol=1e-5)
    with pytest.raises(ValueError):
        renderer.render(rays, gaussians, medium, screen_absolute=(AXES.cuda(), torch.zeros(3, 2, device="cuda")))
    renderer.rgb_medium_backend = "six_pass"
    with pytest.raises(ValueError):
        renderer.render(rays, gaussians, medium,
                        screen_absolute=(AXES.cuda(), torch.zeros(len(gaussians.means), 2, device="cuda")))


@requires_cuda
def test_absolute_gradient_short_real_training(tmp_path):
    from test_medium_fix import _short_manifest
    m = _short_manifest(tmp_path, "abs", overrides("_abs8") + [
        "warmup.background=[0.0, 0.0, 0.0]", "joint.background=[0.0, 0.0, 0.0]"])
    fit_iui3(m)
    rows = list(csv.DictReader((tmp_path / "abs/joint/metrics/train.csv").open()))
    assert len(rows) == 4 and all(math.isfinite(float(r["train/loss"])) for r in rows)
