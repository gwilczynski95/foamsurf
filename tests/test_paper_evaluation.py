"""Optional G7 regressions against original paper artifacts (FS_PAPER_WORKSPACE)."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = Path(os.environ["FS_PAPER_WORKSPACE"]) if os.environ.get("FS_PAPER_WORKSPACE") else None
pytestmark = pytest.mark.skipif(WORKSPACE is None, reason="set FS_PAPER_WORKSPACE to original paper artifacts")
SCENES = ["IUI3-RedSea-Undistorted", "Curasao-Undistorted", "JapaneseGradens-RedSea-Undistorted", "Panama-Undisorted"]
TAG = "cull05r_bin01_wphase_rgbs15_wext"


def execute(script, arguments, output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "command.log").open("w") as log:
        result = subprocess.run([sys.executable, str(ROOT / script), *map(str, arguments), "--out", str(output)],
                                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    assert result.returncode == 0, (output / "command.log").read_text()[-10000:]
    return json.loads((output / "metrics.json").read_text())["mean"]


@pytest.mark.parametrize("scene", SCENES)
def test_seathru_metrics_and_fps(scene, tmp_path):
    run = WORKSPACE / "foamsurf/outputs/seathru/561669" / f"{scene}_{TAG}"
    measured = execute("render.py", ["--run", run, "--modes", "full", "--metrics", "--fps", 32, "--repeats", 2], tmp_path)
    reference = WORKSPACE / "foamsurf/outputs/fps_bench" / f"fps_564824_k32_trun_{run.name}.json"
    expected = json.loads(reference.read_text())[0]["view"]
    for key, digits in [("psnr", 3), ("ssim", 4), ("lpips_vgg", 4)]:
        assert abs(round(measured[key], digits) - expected[key]) <= 0.0001, (key, measured, expected)
    assert json.loads((tmp_path / "fps.json").read_text())["fps"] > 0


@pytest.mark.parametrize("scene", SCENES)
def test_separation_matches_paper(scene, tmp_path):
    run = WORKSPACE / "foamsurf/outputs/seathru/561669" / f"{scene}_{TAG}"
    measured = execute("evaluate.py", ["separation", "--run", run, "--masks",
                                      WORKSPACE / "foamsurf/configs/water_sam" / f"{scene}.npz"], tmp_path)
    with (WORKSPACE / "foamsurf_paper/results/separation_seeds.csv").open() as stream:
        expected = next(r for r in csv.DictReader(stream) if r["method"] == "foamsurf" and r["scene"] == scene and r["seed"] == "42")
    for key in ("op_seabed", "op_water"):
        assert abs(measured[key] - float(expected[key])) <= 0.0001


@pytest.mark.parametrize("level", ["easy", "hard"])
def test_restoration_matches_paper(level, tmp_path):
    run = WORKSPACE / "foamsurf/outputs/seathru/555613" / f"garden_{level}_noclear_noocean_wanc03"
    measured = execute("evaluate.py", ["restoration", "--run", run], tmp_path)
    rows = [json.loads(line) for line in (WORKSPACE / "foamsurf_paper/results/garden_disentangle_raw.jsonl").read_text().splitlines()]
    expected = next(r["ours"] for r in rows if r["run"] == run.name)
    for key, value in expected.items():
        assert abs(measured[key] - value) <= 0.0001, (key, measured[key], value)


@pytest.mark.parametrize("scene,background", [(s, b) for s in ("lego", "ficus") for b in ("black", "white")])
def test_synthetic_layers_match_paper(scene, background, tmp_path):
    with (WORKSPACE / "foamsurf_paper/results/synthetic_seeds.csv").open() as stream:
        expected = next(r for r in csv.DictReader(stream) if r["method"] == "foamsurf" and r["scene"] == scene
                        and r["background"] == background and r["seed"] == "42")
    measured = execute("evaluate.py", ["layers", "--run", expected["source"]], tmp_path)
    for key in ("object_psnr", "cloud_psnr", "gauss_on_cloud"):
        assert abs(measured[key] - float(expected[key])) <= 0.0001, (key, measured[key], expected[key])
