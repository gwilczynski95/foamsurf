"""Tests for the two supported public train.py configurations."""
import csv
import json
from pathlib import Path

import pytest
import torch

from train import fit_iui3, resolve_iui3
from training.end_to_end import EndToEndConfig


ROOT = Path(__file__).resolve().parents[1]


def test_iui3_baseline_contract(tmp_path):
    m = resolve_iui3(ROOT / "configs/iui3_5k.yaml", tmp_path / "run")
    assert m["warmup_steps"] == 0
    assert m["profile"]["spatial_count"] == 5000
    assert m["profile"]["spatial_colmap_fraction"] == 0
    assert m["joint"]["model"]["medium_points"] == 5000
    assert not m["joint"]["voronoi_adaptation"]["enabled"]
    assert m["joint"]["voronoi_adaptation"]["max_points"] == 5000
    assert m["capacity_loss"]["repulsion_weight"] == 0
    assert m["capacity_loss"]["direct_weight"] == 0
    assert m["rgb_surface"]["weight"] == pytest.approx(.03)
    assert m["rgb_smoothing"]["color"] == pytest.approx(1.0)


def test_iui3_sigmoid_configs_resolve_to_cli_equivalents(tmp_path):
    baseline = ROOT / "configs/iui3_5k.yaml"
    cases = (
        ("iui3_sigmoid_both.yaml", "sigmoid"),
        ("iui3_sigmoid_gaussians.yaml", "dc"),
    )
    for filename, medium in cases:
        output = tmp_path / filename
        config = resolve_iui3(ROOT / "configs" / filename, output)
        via_cli = resolve_iui3(
            baseline, output,
            gaussian_rgb_activation="sigmoid",
            medium_color_activation=medium,
            wandb_project="foamsurf_iui3_sigmoid",
        )
        assert config == via_cli
        assert config["joint"]["model"]["gaussian_rgb_activation"] == "sigmoid"
        assert config["medium_color_activation"] == medium


def test_iui3_variant_rejects_unexpected_fields(tmp_path):
    config = tmp_path / "variant.yaml"
    config.write_text(
        "base_config: " + str(ROOT / "configs/iui3_5k.yaml")
        + "\nunknown_setting: true\n"
    )
    with pytest.raises(ValueError, match="Unsupported IUI3 variant fields"):
        resolve_iui3(config, tmp_path / "run")


def test_synthetic_baseline_contract(tmp_path):
    c = EndToEndConfig.load(ROOT / "configs/synthetic.yaml", tmp_path / "run")
    assert c.warmup.optimization.iterations == 5000
    assert c.joint.optimization.iterations == 30000
    assert c.partition["method"] == "weighted_valley_w1_s2"
    assert c.warmup.representation_mode == "voronoi_only"
    assert c.joint.representation_mode == "hybrid"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_iui3_short_real_training(tmp_path):
    m = resolve_iui3(ROOT / "configs/iui3_5k.yaml", tmp_path / "run")
    m["profile"]["spatial_count"] = 256
    for name in ("warmup", "joint"):
        c = m[name]
        c["model"]["medium_points"] = 256
        c["dataset"]["resolution"] = 16
        c["wandb"]["mode"] = "disabled"
        c["validation"].update(eval_interval=2, checkpoint_interval=2,
            log_interval=1, max_eval_views=1, image_views=1)
    j = m["joint"]
    j["optimization"]["iterations"] = 2
    j["gaussian_adaptation"].update(densify_from=1, densify_until=6,
        opacity_prune_from=1, opacity_prune_ramp=0, opacity_prune_until=2)
    m["rgb_surface"].update(start=0, ramp_steps=0, rays=32)
    m["rgb_smoothing"]["ramp_steps"] = 1
    fit_iui3(m)
    fit_iui3(m, resume=True)
    root = tmp_path / "run"
    checkpoint = torch.load(root / "joint/checkpoints/last.pt",
                            map_location="cpu", weights_only=False)
    assert checkpoint["iteration"] == 2
    assert checkpoint["model"]["medium.sites"].shape[0] == 256
    rows = list(csv.DictReader((root / "joint/metrics/train.csv").open()))
    assert len(rows) == 2
    assert json.loads((root / "pipeline_config.json").read_text()) == m
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_synthetic_short_real_training(tmp_path):
    import subprocess
    import sys
    import yaml

    c = EndToEndConfig.load(ROOT / "configs/synthetic.yaml")
    for stage in (c.warmup, c.joint):
        stage.dataset.resolution = 32
        stage.dataset.gaussian_ground_truth_path = None
        stage.dataset.medium_ground_truth_path = None
        stage.optimization.iterations = 4
        stage.optimization.rays_per_batch = 512
        stage.validation.eval_interval = 4
        stage.validation.checkpoint_interval = 2
        stage.validation.log_interval = 1
        stage.validation.max_eval_views = 1
        stage.validation.image_views = 0
        stage.wandb.mode = "disabled"
    c.warmup.model.medium_points = 256
    c.warmup.voronoi_adaptation.densify_from = 2
    c.warmup.voronoi_adaptation.densify_until = 5
    c.warmup.voronoi_adaptation.min_points = 4
    c.warmup.voronoi_adaptation.max_points = 512
    c.warmup.voronoi_warmup.milestone_iterations = [0, 4]
    c.profile.outer_points = 8
    c.profile.population_target = 512
    c.profile.population_horizon = 8
    c.profile.population_min_interval = 2
    c.profile.error_map_chunk = 1024
    c.profile.density_warmup = 0
    c.profile.resolutions = [1]
    c.profile.resolution_steps = [0]
    c.joint.gaussian_adaptation.enabled = False
    adaptation = c.joint.gaussian_adaptation
    adaptation.densify_from = 1
    adaptation.densify_until = 4
    adaptation.opacity_prune_from = 1
    adaptation.opacity_prune_until = 4
    adaptation.size_prune_from = 1
    c.joint.model.medium_points = 32
    c.joint.voronoi_adaptation.min_points = 32
    c.warmup.save(tmp_path / "warmup.yaml")
    c.joint.save(tmp_path / "joint.yaml")
    wrapper = dict(
        output_dir=str(tmp_path / "run"),
        warmup=dict(config="warmup.yaml", profile=c.profile.as_dict()),
        joint=dict(config="joint.yaml"),
        partition=dict(method="otsu_w1", batch_size=128, image_views=1),
        handoff=dict(gaussian_count=16, pca_neighbors=3, reset_density=1e-8),
    )
    path = tmp_path / "smoke.yaml"
    path.write_text(yaml.safe_dump(wrapper))
    command = [sys.executable, str(ROOT / "train.py"), "--config", str(path)]
    subprocess.run(command, cwd=ROOT, check=True)
    root = tmp_path / "run"
    assert all((root / f"{s}_done.json").is_file()
               for s in ("warmup", "partition", "joint"))
    checkpoint = torch.load(root / "joint/checkpoints/last.pt",
                            map_location="cpu", weights_only=False)
    assert checkpoint["iteration"] == 4
    assert (root / "partition/histograms.png").is_file()
    assert (root / "handoff_report.json").is_file()

def test_iui3_sigmoid_manifest_and_zero_step_transfer(tmp_path):
    from training.colmap_joint import transfer_medium

    root = ROOT / "configs/iui3_5k.yaml"
    baseline = resolve_iui3(root, tmp_path / "baseline")
    assert "medium_color_activation" not in baseline
    assert baseline["joint"]["model"]["gaussian_rgb_activation"] == "identity"

    variant = resolve_iui3(
        root, tmp_path / "sigmoid",
        gaussian_rgb_activation="sigmoid",
        medium_color_activation="sigmoid",
        wandb_project="foamsurf_iui3_sigmoid",
    )
    assert variant["warmup_steps"] == 0
    assert variant["profile"]["spatial_colmap_fraction"] == 0
    assert variant["joint"]["model"]["gaussian_rgb_activation"] == "sigmoid"
    assert variant["medium_color_activation"] == "sigmoid"
    assert variant["joint"]["wandb"]["project"] == "foamsurf_iui3_sigmoid"
    assert variant["warmup"]["wandb"]["mode"] == "disabled"
    assert variant["joint"]["experiment_name"].endswith("-g_sigmoid-m_sigmoid")
    gaussian_only = resolve_iui3(
        root, tmp_path / "gaussian_only",
        gaussian_rgb_activation="sigmoid", medium_color_activation="dc",
    )
    medium_only = resolve_iui3(
        root, tmp_path / "medium_only",
        gaussian_rgb_activation="identity", medium_color_activation="sigmoid",
    )
    assert gaussian_only["joint"]["model"]["gaussian_rgb_activation"] == "sigmoid"
    assert gaussian_only["medium_color_activation"] == "dc"
    assert medium_only["joint"]["model"]["gaussian_rgb_activation"] == "identity"
    assert medium_only["medium_color_activation"] == "sigmoid"

    payload = {
        "warmup": {
            "profile": {"site_initialization": "spatial"},
            "raw_color": torch.zeros(2, 3),
        },
        "model_metadata": {
            "medium": {"point_count": 2, "guard_count": 0,
                       "medium_type": "rgb_backscatter"}
        },
        "model": {
            "medium.sites": torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            "medium.raw_density": torch.zeros(2, 6),
            "medium.raw_color": torch.zeros(2, 3),
            "medium.bbox_min": torch.zeros(3),
            "medium.bbox_max": torch.ones(3),
        },
    }
    medium = transfer_medium(payload, "cpu", "sigmoid")
    original = transfer_medium(payload, "cpu", "dc")
    assert torch.allclose(medium.colors, original.colors)
    assert medium.color_activation == "sigmoid"
    assert torch.allclose(medium.colors, torch.full((2, 3), 0.5))
    assert torch.all((medium.colors > 0) & (medium.colors < 1))
    medium.raw_color.data.fill_(20)
    assert bool((medium.colors <= 1).all())
    medium.raw_color.data.fill_(-20)
    assert bool((medium.colors > 0).all())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_iui3_sigmoid_short_real_training(tmp_path):
    from foamsurf_model.gaussians import SurfaceGaussianModel
    from foamsurf_model.medium import VoronoiMediumModel

    m = resolve_iui3(
        ROOT / "configs/iui3_5k.yaml", tmp_path / "sigmoid_run",
        gaussian_rgb_activation="sigmoid",
        medium_color_activation="sigmoid",
    )
    m["profile"]["spatial_count"] = 128
    for stage in ("warmup", "joint"):
        c = m[stage]
        c["model"]["medium_points"] = 128
        c["dataset"]["resolution"] = 16
        c["wandb"]["mode"] = "disabled"
        c["validation"].update(eval_interval=1, checkpoint_interval=1,
            log_interval=1, max_eval_views=1, image_views=0)
    m["joint"]["optimization"]["iterations"] = 2
    m["joint"]["gaussian_adaptation"].update(
        densify_from=1, densify_until=6,
        opacity_prune_from=1, opacity_prune_ramp=0, opacity_prune_until=2,
    )
    m["rgb_surface"].update(start=0, ramp_steps=0, rays=16)
    m["rgb_smoothing"]["ramp_steps"] = 1
    fit_iui3(m)
    fit_iui3(m, resume=True)
    root = tmp_path / "sigmoid_run"
    checkpoint = torch.load(root / "joint/checkpoints/last.pt",
                            map_location="cpu", weights_only=False)
    assert checkpoint["iteration"] == 2
    assert checkpoint["model_metadata"]["gaussians"]["rgb_activation"] == "sigmoid"
    assert checkpoint["model_metadata"]["medium"]["color_activation"] == "sigmoid"
    gaussian_state = {
        name: checkpoint["model"]["gaussians." + name]
        for name in SurfaceGaussianModel.PARAMETER_NAMES
    }
    medium_state = {
        name: checkpoint["model"]["medium." + name]
        for name in VoronoiMediumModel.PARAMETER_NAMES
    }
    gaussians = SurfaceGaussianModel.from_checkpoint(
        gaussian_state, checkpoint["model_metadata"]["gaussians"], "cpu"
    )
    medium = VoronoiMediumModel.from_checkpoint(
        medium_state, checkpoint["model_metadata"]["medium"],
        checkpoint["model"]["medium.bbox_min"],
        checkpoint["model"]["medium.bbox_max"], "cpu",
    )
    assert bool(((gaussians.colors(None) >= 0) & (gaussians.colors(None) <= 1)).all())
    assert bool(((medium.colors >= 0) & (medium.colors <= 1)).all())
    assert (root / "joint/metrics/eval.csv").is_file()
    with (root / "joint/metrics/eval.csv").open() as file:
        eval_rows = list(csv.DictReader(file))
    assert float(eval_rows[-1]["model/medium_color_over_1_fraction"]) == 0
    assert float(eval_rows[-1]["model/medium_color_max"]) <= 1.0001
    with (root / "joint/evaluation/iteration_000002/per_view.csv").open() as file:
        row = next(csv.DictReader(file))
    assert float(row["gaussian_raw_over_1_fraction"]) == 0
    assert float(row["gaussian_raw_max"]) <= 1.0001
    assert 0 <= float(row["gaussian_near_white_on_covered"]) <= 1
