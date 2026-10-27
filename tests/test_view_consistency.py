"""Overfitting to individual photos: Gaussian-DK near-camera gradient scaling and OceanSplat
trinocular view consistency (training/view_consistency.py), the 'consist' set."""
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

from foamsurf_model.gaussian_adaptation import GaussianAdaptationController
from foamsurf_model.gaussians import SurfaceGaussianModel
from train import fit_iui3, resolve_iui3
from training import view_consistency as vc
from training.config import ExperimentConfig, GaussianAdaptationConfig, ModelConfig

ROOT = Path(__file__).resolve().parents[1]


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


# --- Gaussian-DK -------------------------------------------------------------------------------

CAMERAS = torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])  # spread 1 -> R = 1.1
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


# --- OceanSplat --------------------------------------------------------------------------------

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


class PlaneRenderer:
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
    renderer = PlaneRenderer()
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


# --- configuration and the sweep ---------------------------------------------------------------

def overrides(variant):
    script = (ROOT / "scripts/train_seathru.sbatch").read_text()
    start = script.index("variant_overrides() {")
    function = script[start:script.index("\n}\n", start) + 3]
    return subprocess.run(["bash", "-c", function + f"\nvariant_overrides {variant}"],
                          capture_output=True, text=True, check=True).stdout.splitlines()


def test_suffixes_validation_and_set(tmp_path):
    assert overrides("_gdkw") == ["joint.gaussian_adaptation.near_camera_gradient_scaling=true"]
    assert overrides("_gdk") == ["joint.gaussian_adaptation.near_camera_gradient_scaling=true",
                                 "joint.gaussian_adaptation.near_camera_steps=[4000, 7000]"]
    assert overrides("_tri") == ["joint.optimization.trinocular_weight=0.1"]
    m = resolve_iui3(ROOT / "configs/iui3_default.yaml", tmp_path / "r", overrides=overrides("_clamp05_minit_tri_gdk"))
    assert m["joint"]["optimization"]["trinocular_weight"] == 0.1
    assert m["joint"]["gaussian_adaptation"]["near_camera_steps"] == [4000, 7000]
    config = ExperimentConfig()
    config.dataset.source_path = str(tmp_path)
    config.optimization.trinocular_weight = 0.1  # scalar medium: no object/medium parts
    with pytest.raises(ValueError, match="trinocular"):
        config.validate()
    config = ExperimentConfig()
    config.dataset.source_path = str(tmp_path)
    config.gaussian_adaptation.near_camera_steps = [7000, 4000]
    with pytest.raises(ValueError, match="near_camera_steps"):
        config.validate()
    env = {"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin", "FS_VARIANT_SET": "consist",
           "SLURM_ARRAY_JOB_ID": "test", "FS_DRY_RUN": "1",
           "FS_DATASETS": os.environ.get("FS_DATASETS", "/shared/sets/datasets/SeathruNeRF_dataset")}
    run = lambda task: subprocess.run(["bash", "scripts/train_seathru.sbatch"], cwd=ROOT, capture_output=True,
                                      text=True, env={**env, "SLURM_ARRAY_TASK_ID": str(task)})
    assert "out of range" in run(32).stderr  # 4 variants x 4 scenes x 2 seeds
    assert run(31).returncode == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_short_real_training_with_both(tmp_path):
    from test_medium_fix import _short_manifest
    m = _short_manifest(tmp_path, "consist", overrides("_tri_gdk") + [
        "joint.optimization.trinocular_from=1", "warmup.background=[0.0, 0.0, 0.0]",
        "joint.background=[0.0, 0.0, 0.0]"])
    fit_iui3(m)
    rows = list(csv.DictReader((tmp_path / "consist/joint/metrics/train.csv").open()))
    values = [float(r["train/trinocular_obj"]) for r in rows if r.get("train/trinocular_obj")]
    assert values and all(np.isfinite(v) and v >= 0 for v in values)
    assert torch.load(tmp_path / "consist/joint/checkpoints/last.pt", map_location="cpu",
                      weights_only=False)["iteration"] == 4
