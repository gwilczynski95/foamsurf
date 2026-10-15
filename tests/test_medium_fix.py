"""Tests for the medium/seabed priors, manifest overrides and the IUI3 default."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from train import apply_overrides, fit_iui3, resolve_iui3
from training.config import ModelConfig


ROOT = Path(__file__).resolve().parents[1]


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


def test_overrides_and_run_suffix(tmp_path):
    base = resolve_iui3(ROOT / "configs/iui3_5k.yaml", tmp_path / "base")
    from training.losses import MediumFixConfig
    assert base["medium_fix"] == MediumFixConfig().as_dict()
    assert base["medium_fix"]["chroma_weight"] == 0
    assert base["medium_fix"]["surface_coverage_weight"] == 0 and base["medium_fix"]["under_surface_weight"] == 0
    m = resolve_iui3(ROOT / "configs/iui3_5k.yaml", tmp_path / "run",
                     overrides=["medium_fix.chroma_weight=1.0", "medium_fix.surface_coverage_weight=3.0e-3",
                                "joint.optimization.medium_density_lr=0.003", "rgb_smoothing.backscatter=1.0"],
                     run_suffix="E1b")
    assert m["medium_fix"]["chroma_weight"] == pytest.approx(1.0)
    assert m["medium_fix"]["surface_coverage_weight"] == pytest.approx(0.003)
    assert m["joint"]["optimization"]["medium_density_lr"] == pytest.approx(0.003)
    assert m["rgb_smoothing"]["backscatter"] == pytest.approx(1.0)
    assert m["joint"]["experiment_name"].endswith("-E1b")
    assert m["joint"]["wandb"]["name"].endswith("-E1b")
    with pytest.raises(KeyError, match="Unknown override path"):
        apply_overrides({"a": {"b": 1}}, ["a.c=2"])
    with pytest.raises(ValueError, match="key.path=value"):
        apply_overrides({"a": 1}, ["a"])
    with pytest.raises(ValueError, match="chroma_weight"):
        resolve_iui3(ROOT / "configs/iui3_5k.yaml", tmp_path / "bad", overrides=["medium_fix.chroma_weight=-1.0"])
    with pytest.raises(KeyError, match="Unknown override path"):  # removed option
        resolve_iui3(ROOT / "configs/iui3_5k.yaml", tmp_path / "gone", overrides=["medium_fix.color_mode=global"])


def test_resume_accepts_manifest_written_before_medium_fix(tmp_path):
    m = resolve_iui3(ROOT / "configs/iui3_5k.yaml", tmp_path / "run")
    root = Path(m["output_dir"])
    root.mkdir(parents=True)
    legacy = {k: v for k, v in m.items() if k != "medium_fix"}
    (root / "pipeline_config.json").write_text(json.dumps(legacy))
    for stage in ("warmup", "joint"):
        (root / f"{stage}_done.json").write_text(json.dumps({"artifact": str(root / "pipeline_config.json")}))
    fit_iui3(m, resume=True)  # both stages done: only the manifest comparison runs
    changed = resolve_iui3(ROOT / "configs/iui3_5k.yaml", tmp_path / "run", overrides=["medium_fix.chroma_weight=1.0"])
    with pytest.raises(ValueError, match="identical configuration"):
        fit_iui3(changed, resume=True)


def _short_manifest(tmp_path, name, overrides):
    m = resolve_iui3(ROOT / "configs/iui3_5k.yaml", tmp_path / name,
                     gaussian_rgb_activation="sigmoid", medium_color_activation="sigmoid",
                     overrides=overrides)
    m["profile"]["spatial_count"] = 128
    for stage in ("warmup", "joint"):
        c = m[stage]
        c["model"]["medium_points"] = 128
        c["dataset"]["resolution"] = 16
        c["wandb"]["mode"] = "disabled"
        c["validation"].update(eval_interval=2, checkpoint_interval=2, log_interval=1,
                               max_eval_views=1, image_views=0)
    m["joint"]["optimization"]["iterations"] = 4
    m["joint"]["gaussian_adaptation"].update(densify_from=1, densify_until=6, opacity_prune_from=1,
                                             opacity_prune_ramp=0, opacity_prune_until=4,
                                             opacity_prune_interval=1)
    m["rgb_surface"].update(start=0, ramp_steps=0, rays=16)
    m["rgb_smoothing"]["ramp_steps"] = 1
    return m


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_default_medium_fix_short_real_training_matches_contract(tmp_path):
    m = _short_manifest(tmp_path, "default", [])
    fit_iui3(m)
    root = tmp_path / "default"
    checkpoint = torch.load(root / "joint/checkpoints/last.pt", map_location="cpu", weights_only=False)
    assert checkpoint["iteration"] == 4
    rows = list(csv.DictReader((root / "joint/metrics/train.csv").open()))
    assert rows and not any(key.startswith("medium_fix/") for key in rows[0])  # all priors off


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


def test_chroma_config_validation_and_overrides(tmp_path):
    from training.losses import MediumFixConfig

    config = MediumFixConfig(chroma_weight=0.1, chroma_start=10, chroma_ramp=10)
    assert config.chroma_weight_at(5) == 0 and config.chroma_weight_at(15) == pytest.approx(0.05)
    for bad in (dict(chroma_weight=-1), dict(chroma_margin_degrees=90), dict(chroma_water_alpha=0.6),
                dict(chroma_ema=1.0), dict(chroma_start=-1)):
        with pytest.raises(ValueError):
            MediumFixConfig(**bad)
    m = resolve_iui3(ROOT / "configs/iui3_5k.yaml", tmp_path / "c",
                     overrides=["medium_fix.chroma_weight=0.1", "rgb_smoothing.color=10.0"])
    assert m["medium_fix"]["chroma_weight"] == pytest.approx(0.1)
    assert m["rgb_smoothing"]["color"] == pytest.approx(10.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_backscatter_chroma_black_background_short_real_training(tmp_path):
    m = _short_manifest(tmp_path, "chroma", [
        "medium_fix.chroma_weight=0.1", "medium_fix.chroma_start=0", "medium_fix.chroma_ramp=0",
        "medium_fix.prior_rays=64", "rgb_smoothing.color=10.0",
        "warmup.background=[0.0, 0.0, 0.0]", "joint.background=[0.0, 0.0, 0.0]",
    ])
    fit_iui3(m)
    fit_iui3(m, resume=True)
    root = tmp_path / "chroma"
    rows = list(csv.DictReader((root / "joint/metrics/train.csv").open()))
    assert len(rows) == 4
    assert all(float(row["medium_fix/chroma_weight"]) == pytest.approx(0.1) for row in rows)
    state = json.loads((root / "joint/checkpoints/last.medium_fix_state.json").read_text())
    assert state["water_ema"] is None or len(state["water_ema"]) == 3
    checkpoint = torch.load(root / "joint/checkpoints/last.pt", map_location="cpu", weights_only=False)
    assert checkpoint["iteration"] == 4 and checkpoint["config"]["background"] == [0.0, 0.0, 0.0]


def test_default_config_equals_sweep_run_b0_chroma100_bk(tmp_path):
    """configs/iui3_default.yaml must stay identical to the sweep run it was chosen from."""
    output = tmp_path / "default"
    default = resolve_iui3(ROOT / "configs/iui3_default.yaml", output)
    sweep = resolve_iui3(ROOT / "configs/iui3_5k.yaml", output,
                         gaussian_rgb_activation="sigmoid", medium_color_activation="sigmoid",
                         wandb_project="foamsurf_iui3_medium_color",
                         overrides=["medium_fix.chroma_weight=1.0", "warmup.background=[0.0, 0.0, 0.0]",
                                    "joint.background=[0.0, 0.0, 0.0]"])
    assert default == sweep
    assert default["medium_fix"]["chroma_weight"] == 1.0
    assert default["joint"]["background"] == [0.0, 0.0, 0.0]
    later = resolve_iui3(ROOT / "configs/iui3_default.yaml", output, overrides=["medium_fix.chroma_weight=3.0"])
    assert later["medium_fix"]["chroma_weight"] == 3.0  # command line wins over the file


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


def test_seabed_config_validation(tmp_path):
    from training.losses import MediumFixConfig

    config = MediumFixConfig(surface_coverage_weight=2.0, under_surface_weight=1.0)
    assert config.surface_coverage_weight_at(500) == 0
    assert config.under_surface_weight_at(1500) == pytest.approx(0.5)
    for bad in (dict(surface_coverage_weight=-1), dict(under_surface_threshold=1.0),
                dict(surface_coverage_ramp=-1), dict(under_surface_offset=-0.1)):
        with pytest.raises(ValueError):
            MediumFixConfig(**bad)
    m = resolve_iui3(ROOT / "configs/iui3_default.yaml", tmp_path / "s",
                     overrides=["joint.model.gaussian_initial_kappa=4.0", "medium_fix.under_surface_weight=0.1"])
    assert m["joint"]["model"]["gaussian_initial_kappa"] == 4.0
    assert m["medium_fix"]["under_surface_weight"] == pytest.approx(0.1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_seabed_priors_short_real_training(tmp_path):
    m = _short_manifest(tmp_path, "seabed", [
        "medium_fix.chroma_weight=1.0", "medium_fix.chroma_start=0", "medium_fix.chroma_ramp=0",
        "medium_fix.prior_rays=64",
        "medium_fix.surface_coverage_weight=1.0", "medium_fix.surface_coverage_start=0",
        "medium_fix.surface_coverage_ramp=0", "medium_fix.surface_coverage_threshold=0.01",
        "medium_fix.under_surface_weight=1.0", "medium_fix.under_surface_start=0",
        "medium_fix.under_surface_ramp=0", "medium_fix.under_surface_threshold=0.99",
        "joint.model.gaussian_initial_kappa=4.0",
        "warmup.background=[0.0, 0.0, 0.0]", "joint.background=[0.0, 0.0, 0.0]",
    ])
    fit_iui3(m)
    fit_iui3(m, resume=True)
    root = tmp_path / "seabed"
    rows = list(csv.DictReader((root / "joint/metrics/train.csv").open()))
    assert len(rows) == 4
    assert all(float(row["medium_fix/surface_coverage_weight"]) == 1.0 for row in rows)
    assert all(float(row["medium_fix/under_surface_weight"]) == 1.0 for row in rows)
    assert all(0 <= float(row["medium_fix/under_surface_ray_fraction"]) <= 1 for row in rows)
    checkpoint = torch.load(root / "joint/checkpoints/initialization.pt", map_location="cpu", weights_only=False)
    kappa = 1 + torch.nn.functional.softplus(checkpoint["model"]["gaussians.raw_kappa"])
    assert torch.allclose(kappa, torch.full_like(kappa, 4.0), atol=1e-4)


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
