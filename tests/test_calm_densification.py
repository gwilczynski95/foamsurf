"""Budgeted densification (max_points_ramp_until) and the 'calm' sweep set."""
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from foamsurf_model.gaussian_adaptation import GaussianAdaptationController
from foamsurf_model.gaussians import SurfaceGaussianModel
from train import resolve_iui3
from training.config import ExperimentConfig, GaussianAdaptationConfig, ModelConfig

ROOT = Path(__file__).resolve().parents[1]
COUNT = 100


def controller(**overrides):
    points = np.random.default_rng(0).normal(size=(COUNT, 3)).astype(np.float32) + np.array([0, 0, 5], np.float32)
    model = SurfaceGaussianModel(points, np.full_like(points, 0.5),
                                 ModelConfig(appearance="rgb", gaussian_initial_points=COUNT), "cpu")
    with torch.no_grad():
        model.log_scales.fill_(np.log(0.001))  # small: every candidate is a clone
        model.opacity_logits.fill_(4.0)  # opaque: no opacity pruning
    optimizer = torch.optim.Adam([{"params": [p], "name": f"gaussian_{n}", "lr": 1e-3}
                                  for n, p in model.named_primitive_parameters().items()])
    settings = dict(enabled=True, max_points=1100, min_points=1, densify_from=500, densify_until=15000,
                    densification_interval=250, opacity_prune_from=100000, size_prune_from=100000,
                    gradient_threshold=1e-3)
    adaptation = GaussianAdaptationController(model, optimizer, GaussianAdaptationConfig(**{**settings, **overrides}),
                                              scene_extent=10.0)
    adaptation.gradient_sum = torch.linspace(0.01, 1.0, COUNT)  # every Gaussian is a candidate
    adaptation.gradient_denom = torch.ones(COUNT)
    return adaptation


def test_cap_grows_linearly_from_the_initial_count():
    adaptation = controller(max_points_ramp_until=10500)
    assert adaptation.max_points_at(0) == adaptation.max_points_at(500) == COUNT
    assert adaptation.max_points_at(5500) == 600  # halfway: 100 + 1000 / 2
    assert adaptation.max_points_at(10500) == adaptation.max_points_at(20000) == 1100
    assert controller().max_points_at(750) == 1100  # no ramp: max_points from the start


def test_densification_keeps_the_highest_gradients_within_the_ramped_cap():
    adaptation = controller(max_points_ramp_until=10500)
    plan = adaptation.propose_plan(1500)  # cap 100 + 1000 * 0.1 = 200: 100 of 100 candidates fit
    assert plan.final_count == 200 and plan.clone_count == 100
    adaptation = controller(max_points_ramp_until=100500)
    plan = adaptation.propose_plan(1500)  # cap 100 + 1000 * 0.01 = 110: only the 10 strongest
    assert plan.final_count == 110 and plan.clone_count == 10
    strongest = set(range(COUNT - 10, COUNT))
    assert set(plan.clone_indices.tolist()) == strongest
    plan = controller().propose_plan(1500)  # no ramp: all 100 clones (fixed cap 1100)
    assert plan.final_count == 200


def test_initial_count_survives_a_checkpoint():
    adaptation = controller(max_points_ramp_until=10500)
    state = adaptation.state_dict()
    other = controller(max_points_ramp_until=10500)
    other.initial_count = 5
    other.load_state_dict(state)
    assert other.initial_count == COUNT


def test_ramp_validation(tmp_path):
    config = ExperimentConfig()
    config.dataset.source_path = str(tmp_path)
    config.gaussian_adaptation.max_points_ramp_until = config.gaussian_adaptation.densify_from
    with pytest.raises(ValueError, match="max_points_ramp_until"):
        config.validate()


def overrides(variant):
    script = (ROOT / "scripts/train_seathru.sbatch").read_text()
    start = script.index("variant_overrides() {")
    function = script[start:script.index("\n}\n", start) + 3]
    return subprocess.run(["bash", "-c", function + f"\nvariant_overrides {variant}"],
                          capture_output=True, text=True, check=True).stdout.splitlines()


def test_calm_set(tmp_path):
    a = resolve_iui3(ROOT / "configs/iui3_default.yaml", tmp_path / "r", overrides=overrides("_ramp_pd01"))["joint"]
    adaptation = a["gaussian_adaptation"]
    assert adaptation["max_points_ramp_until"] == 15000 and adaptation["percent_dense"] == 0.01
    assert adaptation["max_points"] == 700000 and adaptation["gradient_attenuation_compensation"] is True
    assert a["model"]["appearance"] == "sh" and a["model"]["sh_degree"] == 3  # default is SH3 since 2026-10-24
    assert a["optimization"]["gaussian_sh_degree_interval"] == 1000
    b = resolve_iui3(ROOT / "configs/iui3_default.yaml", tmp_path / "s", overrides=overrides("_nocomp"))["joint"]
    assert b["gaussian_adaptation"]["gradient_attenuation_compensation"] is False
    run = subprocess.run(["bash", "scripts/train_seathru.sbatch"], cwd=ROOT, capture_output=True, text=True,
                         env={"PATH": "/usr/bin:/bin", "FS_VARIANT_SET": "calm", "SLURM_ARRAY_TASK_ID": "40",
                              "SLURM_ARRAY_JOB_ID": "test"})
    assert "out of range" in run.stderr  # control + 4 variants x 4 scenes x 2 seeds = tasks 0-39
