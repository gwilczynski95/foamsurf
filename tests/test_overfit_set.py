"""'overfit' set (fewer medium cells, stronger medium smoothing) and 'gonly_seathru'."""
import subprocess
from pathlib import Path

from train import resolve_iui3

ROOT = Path(__file__).resolve().parents[1]


def overrides(variant):
    script = (ROOT / "scripts/train_seathru.sbatch").read_text()
    start = script.index("variant_overrides() {")
    function = script[start:script.index("\n}\n", start) + 3]
    return subprocess.run(["bash", "-c", function + f"\nvariant_overrides {variant}"],
                          capture_output=True, text=True, check=True).stdout.splitlines()


def dry_run(variant_set, task):
    return subprocess.run(["bash", "scripts/train_seathru.sbatch"], cwd=ROOT, capture_output=True, text=True,
                          env={"PATH": "/shared/results/gmwilczn/conda/envs/foamsurf/bin:/usr/bin:/bin",
                               "FS_VARIANT_SET": variant_set, "SLURM_ARRAY_TASK_ID": str(task),
                               "SLURM_ARRAY_JOB_ID": "test", "FS_DRY_RUN": "1"})


def test_cells_and_smoothing_overrides(tmp_path):
    m = resolve_iui3(ROOT / "configs/iui3_default.yaml", tmp_path / "a", overrides=overrides("_cells50"))
    adaptation = m["joint"]["voronoi_adaptation"]
    assert m["profile"]["spatial_count"] == 50 and adaptation["max_points"] == adaptation["min_points"] == 50
    m = resolve_iui3(ROOT / "configs/iui3_default.yaml", tmp_path / "b", overrides=overrides("_ss10"))
    assert m["rgb_smoothing"]["attenuation"] == m["rgb_smoothing"]["backscatter"] == 10.0
    assert m["rgb_smoothing"]["color"] == 1.0  # colour smoothing unchanged


def test_sets_validate_and_cover_every_task():
    for variant_set, last in (("overfit", 39), ("gonly_seathru", 5)):
        for task in (0, last):
            run = dry_run(variant_set, task)
            assert run.returncode == 0 and "dry run ok" in run.stdout, run.stderr
        assert "out of range" in dry_run(variant_set, last + 1).stderr
    assert "scene=JapaneseGradens-RedSea-Undistorted" in dry_run("gonly_seathru", 5).stdout
