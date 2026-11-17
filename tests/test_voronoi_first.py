"""The Voronoi-first pipeline for the synthetic scenes (configs/synthetic_voronoi.yaml, training/end_to_end.py,
training/native_handoff.py, evaluation/surface_partition.py): the configuration is run 508222, overrides from the
command line, and a short real run of warmup -> partition -> joint (GPU)."""
import json
import subprocess
import sys

import pytest
import torch
import yaml

from helpers import LEGO, ROOT, requires_gpu, requires_lego
from training.end_to_end import EndToEndConfig


def flat(d, prefix=""):
    out = {}
    for key, value in d.items():
        if isinstance(value, dict):
            out.update(flat(value, prefix + key + "."))
        else:
            out[prefix + key] = value
    return out


def test_configuration_is_run_508222(tmp_path):
    """Every setting of run 508222 is unchanged; fields added since then keep their (inactive) defaults."""
    expected = flat(json.loads((ROOT / "tests/data/synthetic_voronoi_508222.json").read_text()))
    resolved = flat(EndToEndConfig.load(ROOT / "configs/synthetic_voronoi.yaml", tmp_path / "run").manifest())
    cosmetic = ("output_dir", "wandb.mode")
    differences = {k: (v, resolved.get(k, "<missing>")) for k, v in expected.items()
                   if not any(c in k for c in cosmetic) and resolved.get(k, "<missing>") != v}
    assert not differences, differences
    added = {k: v for k, v in resolved.items() if k not in expected and "output_dir" not in k}
    inactive = {k: v for k, v in added.items() if k.endswith(("ocean_tri_weight", "ocean_res_weight", "camera_isp",
                                                               "water_phase", "gaussian_mip_filter", "ocean_alpha_until"))}
    assert all(not v for v in inactive.values()), inactive


def test_overrides_reach_the_stages_and_unknown_keys_fail(tmp_path):
    c = EndToEndConfig.load(ROOT / "configs/synthetic_voronoi.yaml", tmp_path / "run", overrides=[
        "both.dataset.source_path=/data/ficus_both", "joint.optimization.dssim_weight=0.3",
        "profile.population_target=4096", "handoff.gaussian_count=50", "both.seed=7"])
    assert c.warmup.dataset.source_path == c.joint.dataset.source_path == "/data/ficus_both"
    assert c.joint.optimization.dssim_weight == 0.3 and c.warmup.optimization.dssim_weight != 0.3
    assert c.profile.population_target == 4096 and c.handoff["gaussian_count"] == 50
    assert c.warmup.seed == c.joint.seed == 7
    for bad in ("joint.optimization.no_such_field=1", "handoff.no_such=1", "nowhere.x=1"):
        with pytest.raises(KeyError):
            EndToEndConfig.load(ROOT / "configs/synthetic_voronoi.yaml", tmp_path / "run", overrides=[bad])
    with pytest.raises(ValueError, match="dataset"):  # warmup and joint must train on the same scene
        EndToEndConfig.load(ROOT / "configs/synthetic_voronoi.yaml", tmp_path / "run",
                            overrides=["joint.dataset.source_path=/elsewhere"])


@requires_lego
def test_command_line_runs_the_voronoi_first_pipeline(tmp_path):
    ficus = LEGO.parent / "ficus" / "ficus_both"
    result = subprocess.run([sys.executable, "train.py", "--config", "configs/synthetic_voronoi.yaml", "--scene", str(ficus),
                             "--out", str(tmp_path / "f"), "--seed", "43", "--dry-run",
                             "--set", f"both.dataset.gaussian_ground_truth_path={ficus.parent / 'ficus_object'}"],
                            cwd=ROOT, capture_output=True, text=True, check=True)
    m = json.loads(result.stdout)
    for stage in ("warmup", "joint"):
        assert m[stage]["dataset"]["source_path"] == str(ficus) and m[stage]["seed"] == 43
        assert m[stage]["dataset"]["gaussian_ground_truth_path"] == str(ficus.parent / "ficus_object")
        assert m[stage]["wandb"]["mode"] == "disabled"


@requires_gpu
@requires_lego
def test_short_real_run_warmup_partition_joint(tmp_path):
    """The three stages on lego at 1/32 resolution for a few steps (same settings as the original smoke test)."""
    c = EndToEndConfig.load(ROOT / "configs/synthetic_voronoi.yaml")
    for stage in (c.warmup, c.joint):
        stage.dataset.source_path = str(LEGO / "lego_both")
        stage.dataset.gaussian_ground_truth_path = str(LEGO / "lego_lego")
        stage.dataset.medium_ground_truth_path = str(LEGO / "lego_clouds")
        stage.dataset.resolution = 32
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
    adaptation = c.joint.gaussian_adaptation
    adaptation.densify_from, adaptation.densify_until = 1, 4
    adaptation.opacity_prune_from, adaptation.opacity_prune_until, adaptation.size_prune_from = 1, 4, 1
    c.joint.model.medium_points = 32
    c.joint.voronoi_adaptation.min_points = 32
    for name, stage in (("warmup", c.warmup), ("joint", c.joint)):
        stage.save(tmp_path / f"{name}.yaml")
    wrapper = dict(output_dir=str(tmp_path / "run"), warmup=dict(config="warmup.yaml", profile=c.profile.as_dict()),
                   joint=dict(config="joint.yaml"),
                   partition=dict(method="otsu_w1", batch_size=128, image_views=1),
                   handoff=dict(gaussian_count=16, pca_neighbors=3, reset_density=1e-8))
    (tmp_path / "smoke.yaml").write_text(yaml.safe_dump(wrapper))
    subprocess.run([sys.executable, "train.py", "--config", str(tmp_path / "smoke.yaml"), "--out", str(tmp_path / "run"),
                    "--seed", "0"], cwd=ROOT, check=True)
    root = tmp_path / "run"
    assert all((root / f"{s}_done.json").is_file() for s in ("warmup", "partition", "joint"))
    assert (root / "partition/histograms.png").is_file() and (root / "handoff_report.json").is_file()
    checkpoint = torch.load(root / "joint/checkpoints/last.pt", map_location="cpu", weights_only=False)
    assert checkpoint["iteration"] == 4
    report = json.loads((root / "handoff_report.json").read_text())
    assert report["gaussian_count"] == 16 and report["retained_voronoi_count"] > 0
    rows = list(__import__("csv").DictReader((root / "joint/metrics/eval.csv").open()))
    assert rows and rows[-1]["lego_lego/psnr"] not in ("", None)


def test_hybrid_joint_stage_is_seathru_without_the_underwater_parts(tmp_path):
    """configs/synthetic_hybrid.yaml: warmup / partition / handoff as synthetic_voronoi.yaml; the joint stage and its
    priors are configs/seathru.yaml except the listed underwater-only parts, on the warmup's scene and seed."""
    from train import resolve_iui3
    hybrid = EndToEndConfig.load(ROOT / "configs/synthetic_hybrid.yaml", tmp_path / "h").manifest()
    plain = EndToEndConfig.load(ROOT / "configs/synthetic_voronoi.yaml", tmp_path / "p").manifest()
    for key in ("warmup", "profile", "partition", "handoff"):
        assert hybrid[key] == plain[key] if key != "warmup" else \
            {k: v for k, v in hybrid[key].items() if k != "output_dir"} == {k: v for k, v in plain[key].items() if k != "output_dir"}
    seathru = resolve_iui3(ROOT / "configs/seathru.yaml", tmp_path / "s")
    joint = flat(hybrid["joint"])
    from training.config import ExperimentConfig, drop_legacy_keys, update_config
    full = ExperimentConfig()
    update_config(full, drop_legacy_keys(seathru["joint"]))  # with the defaults the YAML leaves out
    reference = flat(full.as_dict())
    differences = {k for k in set(joint) | set(reference) if joint.get(k) != reference.get(k)}
    allowed = {"optimization.ocean_tri_weight", "optimization.ocean_epi_weight_start", "optimization.ocean_epi_weight_end",
               "optimization.ocean_res_weight", "optimization.ocean_alpha_until", "optimization.camera_isp",
               "optimization.water_phase", "optimization.water_extrapolation", "model.medium_initial_water_fit",
               "seed", "background", "output_dir", "experiment_name", "wandb.name", "wandb.mode"}
    assert {k for k in differences if not k.startswith("dataset.")} <= allowed, differences - allowed
    assert hybrid["joint"]["dataset"] == hybrid["warmup"]["dataset"] and hybrid["joint"]["seed"] == hybrid["warmup"]["seed"]
    for section in ("rgb_surface", "rgb_smoothing", "capacity_loss"):
        assert hybrid["priors"][section] == seathru[section]
    medium_fix = {k: v for k, v in hybrid["priors"]["medium_fix"].items() if seathru["medium_fix"].get(k) != v}
    # the coverage weight of the reported synthetic runs: SeaThru's until 2026-11-08 (0.003 there since)
    assert medium_fix == {"camera_water_weight": 0.0, "surface_coverage_weight": 0.03}
    c = EndToEndConfig.load(ROOT / "configs/synthetic_hybrid.yaml", tmp_path / "h",
                            overrides=["medium_fix.surface_coverage_weight=0.05"])
    assert c.priors["medium_fix"]["surface_coverage_weight"] == 0.05


@requires_gpu
@requires_lego
def test_short_real_run_with_the_current_joint_stage(tmp_path):
    """Warmup -> partition -> handoff -> the current joint stage (MediumFixTrainer, RGB medium) on lego, a few steps
    with every prior and schedule pulled to the first iterations; layer metrics evaluated."""
    import csv
    from training.end_to_end import EndToEndTrainer
    c = EndToEndConfig.load(ROOT / "configs/synthetic_hybrid.yaml", tmp_path / "run",
                            overrides=[f"both.dataset.source_path={LEGO / 'lego_both'}",
                                       f"both.dataset.gaussian_ground_truth_path={LEGO / 'lego_lego'}",
                                       f"both.dataset.medium_ground_truth_path={LEGO / 'lego_clouds'}"])
    for stage in (c.warmup, c.joint):
        stage.dataset.resolution = 32
        stage.optimization.rays_per_batch = 512
        stage.validation.eval_interval = 4
        stage.validation.checkpoint_interval = 2
        stage.validation.log_interval = 1
        stage.validation.max_eval_views = 1
        stage.validation.image_views = 0
        stage.wandb.mode = "disabled"
    c.warmup.optimization.iterations = 4
    c.warmup.model.medium_points = 256
    c.warmup.voronoi_adaptation.densify_from, c.warmup.voronoi_adaptation.densify_until = 2, 5
    c.warmup.voronoi_adaptation.min_points, c.warmup.voronoi_adaptation.max_points = 4, 512
    c.warmup.voronoi_warmup.milestone_iterations = [0, 4]
    p = c.profile
    p.outer_points, p.population_target, p.population_horizon, p.population_min_interval = 8, 512, 8, 2
    p.error_map_chunk, p.density_warmup, p.resolutions, p.resolution_steps = 1024, 0, [1], [0]
    j = c.joint
    j.optimization.iterations = 4
    j.optimization.training_downscale_levels = 0
    j.optimization.gaussian_sh_degree_interval = 1
    a = j.gaussian_adaptation
    a.densify_from, a.densify_until, a.densification_interval = 1, 6, 1
    a.opacity_prune_from, a.opacity_prune_ramp, a.opacity_prune_until, a.opacity_prune_interval = 1, 0, 4, 1
    a.min_opacity_late_from, a.opacity_reset_interval, a.size_prune_from = 6, 2, 1
    a.min_points = 1  # the tiny handoff below has 16 Gaussians
    j.disentanglement.opacity_binarization_start, j.disentanglement.opacity_binarization_ramp = 0, 0
    j.model.medium_points = 32  # the validation minimum; replaced by the handoff's retained count
    j.voronoi_adaptation.min_points = 4
    j.voronoi_adaptation.max_points = 32  # below the retained cells, as seathru.yaml's 5000 below ~300k real ones
    for prior in ("chroma", "surface_coverage", "water_clear", "camera_water"):
        c.priors["medium_fix"][f"{prior}_start"] = c.priors["medium_fix"][f"{prior}_ramp"] = 0
    c.priors["medium_fix"]["prior_rays"] = 64
    c.priors["rgb_surface"].update(start=0, ramp_steps=0, rays=16)
    c.priors["rgb_smoothing"]["ramp_steps"] = 1
    c.partition.update(method="otsu_w1", batch_size=128, image_views=1)
    c.handoff.update(gaussian_count=16, pca_neighbors=3)
    c.validate()
    EndToEndTrainer(c).fit()
    root = tmp_path / "run"
    checkpoint = torch.load(root / "joint/checkpoints/last.pt", map_location="cpu", weights_only=False)
    assert checkpoint["iteration"] == 4 and checkpoint["model"]["medium.raw_density"].shape[1] == 6  # RGB medium
    assert checkpoint["model_metadata"]["gaussians"]["rgb_scale"] == 1.5 and "camera_isp" not in checkpoint
    rows = list(csv.DictReader((root / "joint/metrics/train.csv").open()))
    assert rows and rows[-1].get("medium_fix/surface_coverage_raw") not in (None, "")
    assert rows[-1].get("surface_overlap/raw") not in (None, "")
    evaluation = list(csv.DictReader((root / "joint/metrics/eval.csv").open()))
    assert evaluation[-1]["lego_lego/psnr"] not in ("", None)
    EndToEndTrainer(c).fit(resume=True)  # finished stages are skipped
