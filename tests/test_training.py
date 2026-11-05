"""End to end on real data (GPU): a few training steps of each configuration with every mechanism switched on
early, evaluation, checkpoint and resume, and render.py on the trained run."""
import csv
import json
import math

import numpy as np
import pytest
import torch
from PIL import Image

from helpers import FOGGY_GARDEN, IUI3, ROOT, requires_gpu, requires_iui3, short_manifest
from train import fit_iui3

pytestmark = requires_gpu


def rows(run, name):
    return list(csv.DictReader((run / "joint/metrics" / name).open()))


def finite(values):
    return [float(v) for v in values if v not in ("", None) and math.isfinite(float(v))]


@requires_iui3
def test_seathru_short_training_runs_every_mechanism_and_renders(tmp_path):
    m = short_manifest(tmp_path, "seathru")
    fit_iui3(m)
    run = tmp_path / "seathru"
    summary = json.loads((run / "joint/summary.json").read_text())
    assert summary["status"] == "completed" and summary["final_iteration"] == 4
    train = rows(run, "train.csv")
    column = lambda name: finite(r.get(name) for r in train)
    # OceanSplat, every prior, opacity reset and densification ran
    for name in ("ocean/obj", "ocean/epi", "ocean/res", "medium_fix/chroma_raw", "medium_fix/surface_coverage_raw",
                 "medium_fix/water_clear_raw", "medium_fix/camera_water_cells", "surface_overlap/raw",
                 "disentanglement/opacity_binarization_raw"):
        assert column(name), name
    # the water-at-the-cameras prior has a value whenever free-water cells lie at a camera (not always in this tiny scene)
    assert bool(column("medium_fix/camera_water_raw")) == any(c > 0 for c in column("medium_fix/camera_water_cells"))
    assert sum(column("opacity_reset/applied")) >= 1 and sum(column("adaptation/cloned")) > 0
    assert all(math.isfinite(v) for v in column("train/loss"))
    evaluation = rows(run, "eval.csv")
    assert finite(r["eval/hybrid_psnr"] for r in evaluation)
    # the camera model, the phase function and the water extension are saved with the model
    checkpoint = torch.load(run / "joint/checkpoints/last.pt", map_location="cpu", weights_only=False)
    assert checkpoint["model_metadata"]["gaussians"]["rgb_scale"] == 1.5
    assert {"camera_isp", "water_phase", "alpha_mlp"} <= set(checkpoint)
    assert (run / "water_fit.json").is_file()

    import render
    out = tmp_path / "renders"
    render.main(["--run", str(run), "--out", str(out), "--npy"])
    trainer = render.load_run(run, "last")
    view = trainer.dataset.test[0]
    for mode in render.MODES:
        image = np.load(out / mode / f"{view.name}.npy")
        assert image.shape[:2] == (view.height, view.width) and np.isfinite(image).all(), mode
        assert (out / mode / f"{view.name}.png").is_file()
    full = np.load(out / "full" / f"{view.name}.npy")
    assert 0.0 <= full.min() and full.max() <= 1.0
    alpha = np.load(out / "alpha" / f"{view.name}.npy")
    assert 0.0 <= alpha.min() and alpha.max() <= 1.0 + 1e-5
    # a pose file written from the dataset cameras renders the same image as the dataset view
    cameras = tmp_path / "cameras.json"
    render.main(["--run", str(run), "--export-cameras", str(cameras)])
    records = json.loads(cameras.read_text())
    assert {r["split"] for r in records} == {"train", "test"} and len(records) == len(trainer.dataset.train) + len(trainer.dataset.test)
    poses = tmp_path / "poses.json"
    poses.write_text(json.dumps([r for r in records if r["name"] == view.name]))
    render.main(["--run", str(run), "--poses", str(poses), "--out", str(tmp_path / "posed"), "--modes", "full,gaussians", "--npy"])
    for mode in ("full", "gaussians"):
        np.testing.assert_allclose(np.load(tmp_path / "posed" / mode / f"{view.name}.npy"),
                                   np.load(out / mode / f"{view.name}.npy"), atol=1e-5)
    png = np.asarray(Image.open(out / "gaussians" / f"{view.name}.png"))
    assert png.dtype == np.uint8 and png.shape == (view.height, view.width, 3)


@requires_iui3
def test_resume_continues_the_same_run(tmp_path):
    m = short_manifest(tmp_path, "resume", iterations=4)
    fit_iui3(m)
    run = tmp_path / "resume"
    with pytest.raises(ValueError, match="resume"):
        fit_iui3(m)  # a non-empty run directory needs --resume
    from training.config import ExperimentConfig
    from training.trainer import Trainer
    config = ExperimentConfig.load(run / "joint/config.yaml")
    config.output_dir = str(tmp_path / "resumed")
    trainer = Trainer(config, resume=run / "joint/checkpoints/last.pt")
    assert trainer.start_iteration == 5 and trainer.water_phase is not None and trainer.camera_isp is not None
    summary, _, _ = trainer.evaluator.evaluate(4)
    assert math.isfinite(summary["eval/primary_psnr"])


@pytest.mark.skipif(not (FOGGY_GARDEN / "sparse").is_dir(), reason="needs the foggy Garden scene (FS_FOGGY)")
def test_garden_fog_short_training(tmp_path):
    m = short_manifest(tmp_path, "garden", config="configs/garden_fog.yaml", scene=FOGGY_GARDEN)
    fit_iui3(m)
    run = tmp_path / "garden"
    assert json.loads((run / "joint/summary.json").read_text())["status"] == "completed"
    train = rows(run, "train.csv")
    assert finite(r.get("medium_fix/water_anchor_raw") for r in train)  # the fog attenuation anchor ran
    assert not finite(r.get("ocean/obj") for r in train)  # no OceanSplat losses
    checkpoint = torch.load(run / "joint/checkpoints/last.pt", map_location="cpu", weights_only=False)
    assert "water_phase" not in checkpoint and checkpoint["model_metadata"]["gaussians"]["rgb_scale"] == 1.0
