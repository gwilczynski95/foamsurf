"""Tuning on held-out training photos (dataset.validation_hold, the _val<N> suffix): the test views are not
used at all, every N-th training photo is evaluated instead and removed from training."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from foamsurf_data.colmap import ColmapDataset
from train import resolve_iui3
from test_oceansplat import overrides

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("FS_DATASETS", "/shared/sets/datasets/SeathruNeRF_dataset")) / "IUI3-RedSea-Undistorted"


@pytest.mark.skipif(not DATA.is_dir(), reason="needs the SeaThru dataset")
def test_validation_holdout_replaces_the_test_views():
    plain = ColmapDataset(DATA, resolution=8, device="cpu")
    held = ColmapDataset(DATA, resolution=8, device="cpu", validation_hold=8)
    train, test = [c.name for c in plain.train], {c.name for c in plain.test}
    assert [c.name for c in held.test] == train[::8]  # every 8th training photo
    assert [c.name for c in held.train] == [n for i, n in enumerate(train) if i % 8]
    assert not test & {c.name for c in held.train + held.test}  # the test views are not used at all
    with pytest.raises(ValueError):
        ColmapDataset(DATA, resolution=8, device="cpu", validation_hold=1)


def test_suffixes_and_sets(tmp_path):
    assert overrides("_val8") == ["warmup.dataset.validation_hold=8", "joint.dataset.validation_hold=8"]
    assert overrides("_dth10") == ["joint.gaussian_adaptation.gradient_threshold=10.0e-5"]
    m = resolve_iui3(ROOT / "configs/seathru_default.yaml", tmp_path / "r", overrides=overrides("_plain_dth10_val8"))
    assert m["joint"]["dataset"]["validation_hold"] == 8 and m["joint"]["gaussian_adaptation"]["gradient_threshold"] == 1e-4
    assert m["medium_fix"]["chroma_weight"] == 0.0
    env = {"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin", "SLURM_ARRAY_JOB_ID": "test", "FS_DRY_RUN": "1",
           "FS_DATASETS": os.environ.get("FS_DATASETS", "/shared/sets/datasets/SeathruNeRF_dataset")}
    run = lambda name, task: subprocess.run(["bash", "scripts/train_seathru.sbatch"], cwd=ROOT, capture_output=True,
                                            text=True, env={**env, "FS_VARIANT_SET": name, "SLURM_ARRAY_TASK_ID": str(task)})
    diag = run("garden_diag", 1)
    assert diag.returncode == 0 and "garden_clear_seed43" in diag.stdout and "out of range" in run("garden_diag", 2).stderr
    tune = run("garden_tune", 7)
    assert tune.returncode == 0 and "garden_hard_plain_dth10_val8" in tune.stdout and "out of range" in run("garden_tune", 8).stderr


def test_garden_prior_set():
    env = {"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin", "SLURM_ARRAY_JOB_ID": "test", "FS_DRY_RUN": "1",
           "FS_VARIANT_SET": "garden_prior",
           "FS_DATASETS": os.environ.get("FS_DATASETS", "/shared/sets/datasets/SeathruNeRF_dataset")}
    run = lambda task: subprocess.run(["bash", "scripts/train_seathru.sbatch"], cwd=ROOT, capture_output=True,
                                      text=True, env={**env, "SLURM_ARRAY_TASK_ID": str(task)})
    first, last = run(0), run(7)
    assert first.returncode == 0 and "medium_fix.water_clear_weight=0.0" in first.stdout
    assert last.returncode == 0 and "garden_hard_nocover_val8" in last.stdout and "out of range" in run(8).stderr


def test_noclear_sets():
    env = {"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin", "SLURM_ARRAY_JOB_ID": "test", "FS_DRY_RUN": "1",
           "FS_DATASETS": os.environ.get("FS_DATASETS", "/shared/sets/datasets/SeathruNeRF_dataset")}
    run = lambda name, task: subprocess.run(["bash", "scripts/train_seathru.sbatch"], cwd=ROOT, capture_output=True,
                                            text=True, env={**env, "FS_VARIANT_SET": name, "SLURM_ARRAY_TASK_ID": str(task)})
    seathru = run("noclear", 7)
    assert seathru.returncode == 0 and "Panama-Undisorted_noclear_seed43" in seathru.stdout
    assert "medium_fix.water_clear_weight=0.0" in seathru.stdout and "out of range" in run("noclear", 8).stderr
    foggy = run("noclear_foggy", 7)
    assert foggy.returncode == 0 and "garden_hard_noclear_seed43" in foggy.stdout and "out of range" in run("noclear_foggy", 8).stderr


def test_garden_fit_set(tmp_path):
    assert overrides("_nomip") == ["joint.model.gaussian_mip_filter=0.0"]
    m = resolve_iui3(ROOT / "configs/seathru_default.yaml", tmp_path / "r", overrides=overrides("_noclear_noocean_it40k_val8"))
    o = m["joint"]["optimization"]
    assert o["ocean_tri_weight"] == 0 and o["ocean_epi_weight_start"] == 0 and o["ocean_res_weight"] == 0
    assert o["ocean_alpha_until"] == 0 and o["iterations"] == 40000 and m["medium_fix"]["water_clear_weight"] == 0
    env = {"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin", "SLURM_ARRAY_JOB_ID": "test", "FS_DRY_RUN": "1",
           "FS_VARIANT_SET": "garden_fit",
           "FS_DATASETS": os.environ.get("FS_DATASETS", "/shared/sets/datasets/SeathruNeRF_dataset")}
    run = lambda task: subprocess.run(["bash", "scripts/train_seathru.sbatch"], cwd=ROOT, capture_output=True,
                                      text=True, env={**env, "SLURM_ARRAY_TASK_ID": str(task)})
    last = run(9)
    assert last.returncode == 0 and "garden_clear_noclear_noocean_it40k_val8" in last.stdout and "out of range" in run(10).stderr


def test_ocean_part_suffixes_and_garden_sets(tmp_path):
    assert overrides("_noalpha") == ["joint.optimization.ocean_alpha_until=0"]
    assert overrides("_notri") == ["joint.optimization.ocean_tri_weight=0.0"]
    assert overrides("_noepi") == ["joint.optimization.ocean_epi_weight_start=0.0", "joint.optimization.ocean_epi_weight_end=0.0",
                                   "joint.optimization.ocean_res_weight=0.0"]
    m = resolve_iui3(ROOT / "configs/seathru_default.yaml", tmp_path / "r", overrides=overrides("_noclear_noalpha_val8"))
    o = m["joint"]["optimization"]
    assert o["ocean_alpha_until"] == 0 and o["ocean_tri_weight"] == 0.1  # only the opacity MLP is off
    env = {"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin", "SLURM_ARRAY_JOB_ID": "test", "FS_DRY_RUN": "1",
           "FS_DATASETS": os.environ.get("FS_DATASETS", "/shared/sets/datasets/SeathruNeRF_dataset")}
    run = lambda name, task: subprocess.run(["bash", "scripts/train_seathru.sbatch"], cwd=ROOT, capture_output=True,
                                            text=True, env={**env, "FS_VARIANT_SET": name, "SLURM_ARRAY_TASK_ID": str(task)})
    final = run("garden_final", 11)
    assert final.returncode == 0 and "garden_clear_noclear_noocean_it40k_seed43" in final.stdout
    assert "out of range" in run("garden_final", 12).stderr
    parts = run("garden_ocean", 5)
    assert parts.returncode == 0 and "garden_hard_noclear_noepi_val8" in parts.stdout and "out of range" in run("garden_ocean", 6).stderr


def test_garden_anchor2_set():
    env = {"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin", "SLURM_ARRAY_JOB_ID": "test", "FS_DRY_RUN": "1",
           "FS_VARIANT_SET": "garden_anchor2",
           "FS_DATASETS": os.environ.get("FS_DATASETS", "/shared/sets/datasets/SeathruNeRF_dataset")}
    run = lambda task: subprocess.run(["bash", "scripts/train_seathru.sbatch"], cwd=ROOT, capture_output=True,
                                      text=True, env={**env, "SLURM_ARRAY_TASK_ID": str(task)})
    last = run(5)
    assert last.returncode == 0 and "garden_hard_noclear_noocean_wanc03_val8" in last.stdout
    assert "medium_fix.water_anchor_weight=0.03" in last.stdout and "ocean_tri_weight=0" in last.stdout
    assert "out of range" in run(6).stderr


def test_garden_final2_and_unify_sets():
    env = {"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin", "SLURM_ARRAY_JOB_ID": "test", "FS_DRY_RUN": "1",
           "FS_DATASETS": os.environ.get("FS_DATASETS", "/shared/sets/datasets/SeathruNeRF_dataset")}
    run = lambda name, task: subprocess.run(["bash", "scripts/train_seathru.sbatch"], cwd=ROOT, capture_output=True,
                                            text=True, env={**env, "FS_VARIANT_SET": name, "SLURM_ARRAY_TASK_ID": str(task)})
    final = run("garden_final2", 7)
    assert final.returncode == 0 and "garden_hard_noclear_noocean_wanc03_it40k_seed43" in final.stdout
    assert "medium_fix.water_anchor_weight=0.03" in final.stdout and "out of range" in run("garden_final2", 8).stderr
    unify = run("unify", 15)
    assert unify.returncode == 0 and "Panama-Undisorted_noclear_noepi_wanc03_seed43" in unify.stdout
    assert "ocean_epi_weight_start=0.0" in unify.stdout and "ocean_tri_weight=0.0" not in unify.stdout
    assert "out of range" in run("unify", 16).stderr


def test_epivis_sets():
    env = {"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin", "SLURM_ARRAY_JOB_ID": "test", "FS_DRY_RUN": "1",
           "FS_DATASETS": os.environ.get("FS_DATASETS", "/shared/sets/datasets/SeathruNeRF_dataset")}
    run = lambda name, task: subprocess.run(["bash", "scripts/train_seathru.sbatch"], cwd=ROOT, capture_output=True,
                                            text=True, env={**env, "FS_VARIANT_SET": name, "SLURM_ARRAY_TASK_ID": str(task)})
    garden = run("garden_epivis", 1)
    assert garden.returncode == 0 and "garden_hard_noclear_epivis_val8" in garden.stdout
    assert "ocean_visible=0.1" in garden.stdout and "out of range" in run("garden_epivis", 2).stderr
    seathru = run("epivis", 7)
    assert seathru.returncode == 0 and "Panama-Undisorted_noclear_epivis_seed43" in seathru.stdout
    assert "out of range" in run("epivis", 8).stderr


def test_epivis_wanc_set():
    env = {"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin", "SLURM_ARRAY_JOB_ID": "test", "FS_DRY_RUN": "1",
           "FS_VARIANT_SET": "epivis_wanc",
           "FS_DATASETS": os.environ.get("FS_DATASETS", "/shared/sets/datasets/SeathruNeRF_dataset")}
    run = lambda task: subprocess.run(["bash", "scripts/train_seathru.sbatch"], cwd=ROOT, capture_output=True,
                                      text=True, env={**env, "SLURM_ARRAY_TASK_ID": str(task)})
    first, last = run(0), run(11)
    assert first.returncode == 0 and "IUI3-RedSea-Undistorted_noclear_epivis_wanc03" in first.stdout
    assert last.returncode == 0 and "garden_hard_noclear_epivis_wanc03_seed43" in last.stdout
    assert "ocean_visible=0.1" in last.stdout and "medium_fix.water_anchor_weight=0.03" in last.stdout
    assert "out of range" in run(12).stderr


def test_fit_learning_rate_suffixes_and_set(tmp_path):
    assert overrides("_aconst") == ["joint.optimization.gaussian_appearance_lr_decay=0.0"]
    assert overrides("_shr20") == ["joint.optimization.gaussian_appearance_rest_lr_scale=0.05"]
    m = resolve_iui3(ROOT / "configs/seathru_default.yaml", tmp_path / "r", overrides=overrides("_wsapp"))
    o = m["joint"]["optimization"]
    assert o["gaussian_appearance_lr"] == 0.0025 and o["gaussian_appearance_lr_decay"] == 0.0
    assert o["gaussian_appearance_rest_lr_scale"] == 0.05
    # the densification interval is given in 30k-schedule steps: _dint150 -> 100 at the 20k default
    m = resolve_iui3(ROOT / "configs/seathru_default.yaml", tmp_path / "s", overrides=overrides("_dint150"))
    assert m["joint"]["gaussian_adaptation"]["densification_interval"] == 100
    env = {"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin", "SLURM_ARRAY_JOB_ID": "test", "FS_DRY_RUN": "1",
           "FS_VARIANT_SET": "fitlr",
           "FS_DATASETS": os.environ.get("FS_DATASETS", "/shared/sets/datasets/SeathruNeRF_dataset")}
    run = lambda task: subprocess.run(["bash", "scripts/train_seathru.sbatch"], cwd=ROOT, capture_output=True,
                                      text=True, env={**env, "SLURM_ARRAY_TASK_ID": str(task)})
    last = run(19)
    assert last.returncode == 0 and "Panama-Undisorted_dint150_seed43" in last.stdout and "out of range" in run(20).stderr


def test_adam_eps_suffix_and_set(tmp_path):
    assert overrides("_eps15") == ["joint.optimization.gaussian_adam_eps=1.0e-15"]
    m = resolve_iui3(ROOT / "configs/seathru_default.yaml", tmp_path / "r", overrides=overrides("_eps12"))
    assert m["joint"]["optimization"]["gaussian_adam_eps"] == 1e-12
    env = {"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin", "SLURM_ARRAY_JOB_ID": "test", "FS_DRY_RUN": "1",
           "FS_VARIANT_SET": "adameps",
           "FS_DATASETS": os.environ.get("FS_DATASETS", "/shared/sets/datasets/SeathruNeRF_dataset")}
    run = lambda task: subprocess.run(["bash", "scripts/train_seathru.sbatch"], cwd=ROOT, capture_output=True,
                                      text=True, env={**env, "SLURM_ARRAY_TASK_ID": str(task)})
    last = run(7)
    assert last.returncode == 0 and "Panama-Undisorted_eps12_seed43" in last.stdout and "out of range" in run(8).stderr


def test_paper_ocean_window(tmp_path):
    # the paper enables L_tri and L_epi in 4k-8k of 10k steps; _owin gives the same fractions at the 20k default
    m = resolve_iui3(ROOT / "configs/seathru_default.yaml", tmp_path / "r", overrides=overrides("_owin"))
    o = m["joint"]["optimization"]
    assert o["iterations"] == 20000 and o["ocean_from"] == 8000 and o["ocean_until"] == 16000
    env = {"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin", "SLURM_ARRAY_JOB_ID": "test", "FS_DRY_RUN": "1",
           "FS_VARIANT_SET": "owin",
           "FS_DATASETS": os.environ.get("FS_DATASETS", "/shared/sets/datasets/SeathruNeRF_dataset")}
    run = lambda task: subprocess.run(["bash", "scripts/train_seathru.sbatch"], cwd=ROOT, capture_output=True,
                                      text=True, env={**env, "SLURM_ARRAY_TASK_ID": str(task)})
    last = run(7)
    assert last.returncode == 0 and "Panama-Undisorted_owin_seed43" in last.stdout and "out of range" in run(8).stderr
