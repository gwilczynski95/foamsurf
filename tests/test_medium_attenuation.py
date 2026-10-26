"""scripts/medium_attenuation.py recovers a known attenuation; run_local.sh and the 'diag' set."""
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def sbatch_env(**extra):
    """Environment for a dry run of train_seathru.sbatch on this machine (cluster paths by default)."""
    env = {"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin", "SLURM_ARRAY_JOB_ID": "test", "FS_DRY_RUN": "1",
           "FS_DATASETS": os.environ.get("FS_DATASETS", "/shared/sets/datasets/SeathruNeRF_dataset")}
    return {**env, **extra}


SEATHRU_PRESENT = Path(sbatch_env()["FS_DATASETS"]).is_dir()
sys.path.insert(0, str(ROOT / "scripts"))

import medium_attenuation as m  # noqa: E402


def test_recovers_a_known_sigma_and_its_distance_dependence():
    rng = np.random.default_rng(0)
    grid, height, width, sigma = m.GRID, 120, 180, 0.05
    depth = rng.uniform(2.0, 12.0, grid * grid)
    depth[:5] = np.nan  # cells without COLMAP points are skipped
    per_cell_t = np.exp(-sigma * np.nan_to_num(depth, nan=1.0)).reshape(grid, grid)
    transmittance = np.kron(per_cell_t, np.ones((height // grid, width // grid)))[..., None]
    gaussians = rng.uniform(0.2, 0.8, (height, width, 3))
    backscatter = rng.uniform(0.0, 0.2, (height, width, 3))
    hybrid = transmittance * gaussians + backscatter
    alpha = np.ones((height, width))
    ts, ds = m.cell_transmittance(hybrid - backscatter, gaussians, alpha, depth)
    assert len(ts) == grid * grid - 5
    result = m.summarise([(ts, ds)])
    assert abs(result["sigma"] - sigma) < 0.01 and result["correlation"] < -0.95
    alpha[:, :] = 0.5  # nothing covered by Gaussians: no cell is used
    assert len(m.cell_transmittance(hybrid - backscatter, gaussians, alpha, depth)[0]) == 0


def test_cell_depths_from_projected_points():
    camera = SimpleNamespace(origin=torch.zeros(3), forward=torch.tensor([0.0, 0.0, 1.0]),
                             right=torch.tensor([1.0, 0.0, 0.0]), down=torch.tensor([0.0, 1.0, 0.0]),
                             intrinsics=(100.0, 100.0, 50.0, 50.0), width=100, height=100)
    points = torch.tensor([[0.0, 0.0, z] for z in (4.0, 5.0, 6.0, 7.0, 8.0)] + [[0.0, 0.0, -3.0]])
    u, r, z, inside = m.project(points, camera)
    assert inside.tolist() == [True] * 5 + [False]
    depth = m.cell_depths(u, r, z, inside, grid=2, min_points=5)
    assert np.isfinite(depth).sum() == 1 and 4.0 <= np.nanmax(depth) <= 5.0  # 20% quantile


def overrides(variant):
    script = (ROOT / "scripts/train_seathru.sbatch").read_text()
    start = script.index("variant_overrides() {")
    function = script[start:script.index("\n}\n", start) + 3]
    return subprocess.run(["bash", "-c", function + f"\nvariant_overrides {variant}"],
                          capture_output=True, text=True, check=True).stdout.splitlines()


def test_diag_set_and_leave_one_out_suffixes():
    assert overrides("_it10k") == ["joint.optimization.iterations=10000",
                                   "joint.gaussian_adaptation.opacity_prune_until=10000"]
    assert overrides("_nooverlap") == ["rgb_surface.weight=0.0"]
    assert overrides("_nochroma") == ["medium_fix.chroma_weight=0.0"]
    assert overrides("_nocover") == ["medium_fix.surface_coverage_weight=0.0"]
    assert overrides("_noclear") == ["medium_fix.water_clear_weight=0.0"]
    assert overrides("_nobin") == ["joint.disentanglement.opacity_binarization_weight=0.0"]
    assert len(overrides("_nosmooth")) == 3
    if not SEATHRU_PRESENT:
        pytest.skip("needs the SeaThru dataset (FS_DATASETS)")
    env = sbatch_env(FS_VARIANT_SET="diag")
    for task in (0, 15):
        run = subprocess.run(["bash", "scripts/train_seathru.sbatch"], cwd=ROOT, capture_output=True, text=True,
                             env={**env, "SLURM_ARRAY_TASK_ID": str(task)})
        assert run.returncode == 0 and "dry run ok" in run.stdout, run.stderr
    assert "out of range" in subprocess.run(["bash", "scripts/train_seathru.sbatch"], cwd=ROOT, capture_output=True,
                                            text=True, env={**env, "SLURM_ARRAY_TASK_ID": "16"}).stderr


def test_sbatch_paths_can_be_overridden_and_run_local_needs_the_dataset(tmp_path):
    script = (ROOT / "scripts/train_seathru.sbatch").read_text()
    for name in ("FS_ROOT", "FS_ENV", "FS_DRIVER", "FS_DATASETS", "FS_MIPNERF"):
        assert "${" + name + ":-" in script
    assert "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.9}" in script
    run = subprocess.run(["bash", "scripts/run_local.sh", "0"], cwd=ROOT, capture_output=True, text=True,
                         env={"PATH": "/usr/bin:/bin", "FS_ENV": str(Path(sys.executable).parents[1])})
    assert run.returncode != 0 and "FS_DATASETS" in run.stderr
