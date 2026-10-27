"""Full OceanSplat (training/oceansplat.py): differentiable alpha-blended depth, depth-warped
trinocular consistency, smoothness, epipolar prior, depth residual, depth-aware opacity, the
resolution schedule and the 'ocean' set."""
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

ROOT = Path(__file__).resolve().parents[1]


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


# --- warping on a synthetic textured plane z = 5 --------------------------------------------

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
    assert stray["res"].item() > 1.0  # a Gaussian 3 units in front of the surface


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


# --- configuration, schedule and the set --------------------------------------------------------

def overrides(variant):
    script = (ROOT / "scripts/train_seathru.sbatch").read_text()
    start = script.index("variant_overrides() {")
    function = script[start:script.index("\n}\n", start) + 3]
    return subprocess.run(["bash", "-c", function + f"\nvariant_overrides {variant}"],
                          capture_output=True, text=True, check=True).stdout.splitlines()


def test_resolution_schedule_matches_oceansplat():
    optimization = OptimizationConfig(training_downscale_levels=2, training_downscale_interval=6000)
    assert [training_downscale(optimization, it) for it in (0, 5999, 6000, 11999, 12000, 30000)] == [4, 4, 2, 2, 1, 1]


def test_suffixes_validation_and_set(tmp_path):
    assert overrides("_rs") == ["joint.optimization.training_downscale_levels=2",
                                "joint.optimization.training_downscale_interval=6000"]
    assert overrides("_ocean") == ["joint.optimization.ocean_tri_weight=0.1", "joint.optimization.ocean_epi_weight_start=0.4",
                                   "joint.optimization.ocean_epi_weight_end=0.2", "joint.optimization.ocean_res_weight=0.01",
                                   "joint.optimization.ocean_alpha_until=12000"]
    assert overrides("_oceane") == overrides("_ocean") + ["joint.optimization.ocean_from=4000"]
    early = resolve_iui3(ROOT / "configs/iui3_default.yaml", tmp_path / "e", overrides=overrides("_clamp05_minit_rel_oceane"))
    assert early["joint"]["optimization"]["ocean_from"] == 4000
    assert early["joint"]["optimization"]["reconstruction_loss"] == "reg_l1_reg_dssim"
    m = resolve_iui3(ROOT / "configs/iui3_default.yaml", tmp_path / "r", overrides=overrides("_clamp05_minit_rs_ocean"))
    assert m["joint"]["optimization"]["ocean_tri_weight"] == 0.1
    assert m["joint"]["optimization"]["training_downscale_levels"] == 2
    config = ExperimentConfig()
    config.dataset.source_path = str(tmp_path)
    config.optimization.ocean_res_weight = 0.01  # scalar medium
    with pytest.raises(ValueError, match="ocean"):
        config.validate()
    env = {"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin", "FS_VARIANT_SET": "ocean",
           "SLURM_ARRAY_JOB_ID": "test", "FS_DRY_RUN": "1",
           "FS_DATASETS": os.environ.get("FS_DATASETS", "/shared/sets/datasets/SeathruNeRF_dataset")}
    run = lambda task: subprocess.run(["bash", "scripts/train_seathru.sbatch"], cwd=ROOT, capture_output=True,
                                      text=True, env={**env, "SLURM_ARRAY_TASK_ID": str(task)})
    assert "out of range" in run(48).stderr  # 6 variants x 4 scenes x 2 seeds
    assert run(47).returncode == 0
    nooverlap = resolve_iui3(ROOT / "configs/iui3_default.yaml", tmp_path / "n", overrides=overrides("_clamp05_minit_rel_nooverlap"))
    assert nooverlap["rgb_surface"]["weight"] == 0.0
    assert nooverlap["joint"]["optimization"]["reconstruction_loss"] == "reg_l1_reg_dssim"
    env["FS_VARIANT_SET"] = "overlap"
    assert "out of range" in run(8).stderr  # 1 variant x 4 scenes x 2 seeds
    assert run(7).returncode == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_short_real_training_with_full_oceansplat(tmp_path):
    from test_medium_fix import _short_manifest
    m = _short_manifest(tmp_path, "ocean", overrides("_ocean") + [
        "joint.optimization.ocean_from=1", "joint.optimization.ocean_until=5", "joint.optimization.ocean_downscale=1",
        "joint.optimization.ocean_alpha_until=3", "warmup.background=[0.0, 0.0, 0.0]", "joint.background=[0.0, 0.0, 0.0]"])
    fit_iui3(m)
    rows = list(csv.DictReader((tmp_path / "ocean/joint/metrics/train.csv").open()))
    for name in ("obj", "full", "smooth", "res"):
        values = [float(r[f"ocean/{name}"]) for r in rows if r.get(f"ocean/{name}") not in (None, "", "nan")]
        assert values and all(np.isfinite(v) and v >= 0 for v in values), name
    checkpoint = torch.load(tmp_path / "ocean/joint/checkpoints/last.pt", map_location="cpu", weights_only=False)
    assert checkpoint["iteration"] == 4 and "alpha_mlp" in checkpoint
