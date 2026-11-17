"""Configurations: configs/seathru.yaml and configs/garden_fog.yaml resolve to exactly the settings of our reported
runs (snapshots in tests/data), variant chains, the train.py command line, old run configs, prior validation."""
import json
import subprocess
import sys

import pytest
import yaml

from helpers import IUI3, LEGO, ROOT, comparable, requires_lego
from train import load_iui3_source, resolve_iui3
from training.config import ExperimentConfig
from training.losses import MediumFixConfig


@pytest.mark.parametrize("config, snapshot", [
    ("configs/seathru.yaml", "seathru_resolved.json"),      # SeaThru default, sweeps 561669 / 562266 (not _cov09 562267)
    ("configs/garden_fog.yaml", "garden_fog_resolved.json"),  # foggy Garden, sweep 555613
])
def test_configs_are_the_reported_runs(config, snapshot, tmp_path):
    expected = json.loads((ROOT / "tests/data" / snapshot).read_text())
    resolved = comparable(resolve_iui3(ROOT / config, tmp_path / "run"))
    differences = {k: (expected.get(k, "<missing>"), resolved.get(k, "<missing>"))
                   for k in set(expected) | set(resolved) if expected.get(k, "<missing>") != resolved.get(k, "<missing>")}
    assert not differences, differences


def test_synthetic_config_is_seathru_without_the_colmap_water_fit(tmp_path):
    seathru = comparable(resolve_iui3(ROOT / "configs/seathru.yaml", tmp_path / "a"))
    synthetic = comparable(resolve_iui3(ROOT / "configs/synthetic.yaml", tmp_path / "b"))
    differences = {k: (seathru.get(k), synthetic.get(k)) for k in set(seathru) | set(synthetic)
                   if seathru.get(k) != synthetic.get(k)}
    assert differences == {"joint.model.medium_initial_water_fit": ("coefficients", "off"),
                           "medium_fix.surface_coverage_weight": (0.003, 0.03)}  # as the synthetic runs were trained


@requires_lego
def test_command_line_accepts_a_blender_scene(tmp_path):
    scene = LEGO / "lego_both"
    result = subprocess.run([sys.executable, "train.py", "--scene", str(scene), "--out", str(tmp_path / "a"),
                             "--config", "configs/synthetic.yaml", "--dry-run"],
                            cwd=ROOT, capture_output=True, text=True, check=True)
    m = json.loads(result.stdout)
    assert m["joint"]["dataset"]["source_path"] == str(scene) and m["joint"]["model"]["medium_initial_water_fit"] == "off"


def test_values_in_the_config_are_the_values_that_run(tmp_path):
    m = resolve_iui3(ROOT / "configs/seathru.yaml", tmp_path / "run")
    adaptation, optimization = m["joint"]["gaussian_adaptation"], m["joint"]["optimization"]
    assert optimization["iterations"] == 20000 and adaptation["opacity_reset_interval"] == 500  # no rescaling
    assert m["joint"]["model"]["gaussian_rgb_scale"] == 1.5 and optimization["water_phase"] is True
    assert m["medium_fix"]["surface_coverage_weight"] == 0.003 and m["medium_fix"]["surface_coverage_threshold"] == 0.9


def test_variant_chain_applies_overrides_base_first(tmp_path):
    (tmp_path / "base.yaml").write_text((ROOT / "configs/base.yaml").read_text())
    (tmp_path / "a.yaml").write_text(yaml.safe_dump({"base_config": "base.yaml", "gaussian_rgb_activation": "sigmoid",
                                                     "overrides": ["joint.optimization.dssim_weight=0.3",
                                                                   "medium_fix.chroma_weight=1.0"]}))
    (tmp_path / "b.yaml").write_text(yaml.safe_dump({"base_config": "a.yaml",
                                                     "overrides": ["joint.optimization.dssim_weight=0.25"]}))
    _, variant = load_iui3_source(tmp_path / "b.yaml")
    assert variant["overrides"] == ["joint.optimization.dssim_weight=0.3", "medium_fix.chroma_weight=1.0",
                                    "joint.optimization.dssim_weight=0.25"]
    m = resolve_iui3(tmp_path / "b.yaml", tmp_path / "run")
    assert m["joint"]["optimization"]["dssim_weight"] == 0.25 and m["medium_fix"]["chroma_weight"] == 1.0
    assert m["joint"]["model"]["gaussian_rgb_activation"] == "sigmoid"  # inherited from a.yaml


@pytest.mark.skipif(not IUI3.is_dir(), reason="needs the SeaThru-NeRF IUI3 scene")
def test_command_line_sets_scene_seed_and_logging(tmp_path):
    run = lambda *args: subprocess.run([sys.executable, "train.py", *args, "--dry-run"], cwd=ROOT, capture_output=True,
                                       text=True, check=True)
    m = json.loads(run("--scene", str(IUI3), "--out", str(tmp_path / "a")).stdout)
    for stage in ("warmup", "joint"):
        assert m[stage]["dataset"]["source_path"] == str(IUI3) and m[stage]["seed"] == 42
    assert m["joint"]["wandb"]["mode"] == "disabled" and m["output_dir"] == str(tmp_path / "a")
    m = json.loads(run("--scene", str(IUI3), "--out", str(tmp_path / "b"), "--seed", "43", "--wandb", "proj",
                       "--config", "configs/garden_fog.yaml", "--set", "joint.optimization.dssim_weight=0.3").stdout)
    assert m["joint"]["seed"] == 43 and m["joint"]["wandb"]["mode"] == "online" and m["joint"]["wandb"]["project"] == "proj"
    assert m["joint"]["optimization"]["dssim_weight"] == 0.3 and m["medium_fix"]["water_anchor_weight"] == 0.03
    bad = subprocess.run([sys.executable, "train.py", "--scene", str(tmp_path), "--out", str(tmp_path / "c"), "--dry-run"],
                         cwd=ROOT, capture_output=True, text=True)
    assert bad.returncode != 0 and "not a COLMAP scene" in bad.stderr


def test_configs_of_older_runs_still_load(tmp_path):
    saved = ExperimentConfig().as_dict()
    saved["dataset"]["source_path"] = str(tmp_path)
    saved["optimization"].update(water_direction=False, water_direction_lr=0.005)  # removed option, off
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(saved))
    ExperimentConfig.load(tmp_path / "config.yaml")
    saved["optimization"]["water_direction"] = True  # a run that used it cannot be loaded silently
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(saved))
    with pytest.raises(ValueError, match="water_direction"):
        ExperimentConfig.load(tmp_path / "config.yaml")


def test_prior_validation():
    MediumFixConfig(water_clear_weight=0.01, surface_coverage_weight=0.03, surface_coverage_threshold=0.9)
    with pytest.raises(ValueError, match="water_clear_threshold"):  # both priors would pull the same rays
        MediumFixConfig(water_clear_weight=0.01, surface_coverage_weight=0.03, surface_coverage_threshold=0.3)
    with pytest.raises(ValueError, match="chroma_water_alpha"):
        MediumFixConfig(chroma_water_alpha=0.5, chroma_alpha_threshold=0.2)
    with pytest.raises(ValueError, match="surface_coverage_weight"):
        MediumFixConfig(surface_coverage_weight=-1.0)
