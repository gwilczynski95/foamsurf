"""Gaussians alone ('gonly' set): no medium, SH colour, Mip-NeRF 360 on images_4 and SeaThru."""
import csv
import math
import subprocess
from pathlib import Path

import pytest
import torch

from train import fit_iui3, resolve_iui3

ROOT = Path(__file__).resolve().parents[1]
MIPNERF = Path("/shared/sets/datasets/mipnerf_ds")


def overrides(variant):
    script = (ROOT / "scripts/train_seathru.sbatch").read_text()
    start = script.index("variant_overrides() {")
    function = script[start:script.index("\n}\n", start) + 3]
    return subprocess.run(["bash", "-c", function + f"\nvariant_overrides {variant}"],
                          capture_output=True, text=True, check=True).stdout.splitlines()


def dry_run(task):
    """Run the sbatch for one 'gonly' task with FS_DRY_RUN=1: prints the task and validates the manifest."""
    return subprocess.run(["bash", "scripts/train_seathru.sbatch"], cwd=ROOT, capture_output=True, text=True,
                          env={"PATH": "/shared/results/gmwilczn/conda/envs/foamsurf/bin:/usr/bin:/bin",
                               "FS_VARIANT_SET": "gonly", "SLURM_ARRAY_TASK_ID": str(task),
                               "SLURM_ARRAY_JOB_ID": "test", "FS_DRY_RUN": "1"})


def test_gonly_overrides(tmp_path):
    m = resolve_iui3(ROOT / "configs/iui3_default.yaml", tmp_path / "r", overrides=overrides("_plain_sh3_g6000k_gonly"))
    joint = m["joint"]
    assert joint["representation_mode"] == "gaussian_only" and joint["dataset"]["train_target"] == "both"
    assert joint["voronoi_adaptation"]["enabled"] is False
    assert joint["model"]["appearance"] == "sh" and joint["model"]["sh_degree"] == 3
    assert joint["gaussian_adaptation"]["max_points"] == 6_000_000
    assert m["rgb_surface"]["weight"] == 0.0 and m["medium_fix"]["chroma_weight"] == 0.0


@pytest.mark.skipif(not (MIPNERF / "garden").is_dir(), reason="needs the Mip-NeRF 360 dataset")
def test_gonly_set_tasks_and_mipnerf_sources():
    garden = dry_run(0)
    assert garden.returncode == 0, garden.stderr
    assert "scene=garden variant=_plain_sh3_g6000k_gonly seed=42" in garden.stdout
    assert "joint.dataset.resolution=4" in garden.stdout and "dry run ok" in garden.stdout
    panama = dry_run(5)
    assert panama.returncode == 0, panama.stderr
    assert "scene=Panama-Undisorted" in panama.stdout and "seed=43" in panama.stdout
    assert "dataset.resolution=4" not in panama.stdout  # SeaThru at full resolution
    assert "out of range" in dry_run(6).stderr  # 3 scenes x 2 seeds = tasks 0-5


def test_medium_losses_are_refused_without_the_medium():
    source = (ROOT / "training/colmap_joint_trainer.py").read_text()
    assert source.count("need the medium (representation_mode=gaussian_only)") == 1
    assert source.count("needs the medium (representation_mode=gaussian_only)") == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_gonly_short_real_training_reports_the_gaussians_alone(tmp_path):
    from test_medium_fix import _short_manifest
    m = _short_manifest(tmp_path, "gonly", overrides("_plain_sh3_gonly") + [
        "warmup.background=[0.0, 0.0, 0.0]", "joint.background=[0.0, 0.0, 0.0]"])
    fit_iui3(m)
    rows = list(csv.DictReader((tmp_path / "gonly/joint/metrics/train.csv").open()))
    assert len(rows) == 4 and all(math.isfinite(float(r["train/loss"])) for r in rows)
    evals = list(csv.DictReader((tmp_path / "gonly/joint/metrics/eval.csv").open()))
    for row in evals:  # the main image is the Gaussians alone: no untrained medium in it
        assert math.isclose(float(row["eval/hybrid_psnr"]), float(row["eval/gaussian_only_psnr"]), abs_tol=1e-3)
