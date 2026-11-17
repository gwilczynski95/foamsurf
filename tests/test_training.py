"""End to end on real data (GPU): a few training steps of each configuration with every mechanism switched on
early, evaluation, checkpoint and resume, and render.py on the trained run."""
import csv
import json
import math

import numpy as np
import pytest
import torch
from PIL import Image

from helpers import FOGGY_GARDEN, IUI3, LEGO, ROOT, requires_gpu, requires_iui3, requires_lego, short_manifest
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
    # a run moved to another machine: the stored dataset path is gone, --scene points to the data
    config_file = run / "joint/config.yaml"
    config_file.write_text(config_file.read_text().replace(str(IUI3), "/nonexistent/IUI3-RedSea-Undistorted"))
    with pytest.raises(FileNotFoundError, match="--scene"):
        render.load_run(run, "last")
    render.main(["--run", str(run), "--scene", str(IUI3), "--out", str(tmp_path / "moved"), "--modes", "full", "--npy",
                 "--metrics"])
    metrics = json.loads((tmp_path / "moved/metrics.json").read_text())
    image = torch.from_numpy(np.load(tmp_path / "moved/full" / f"{view.name}.npy"))
    target = view.ground_truth(tuple(trainer.config.background)).cpu().reshape(image.shape)
    assert abs(metrics["views"][view.name]["psnr"] - float(-10 * torch.log10(((image - target) ** 2).mean()))) < 1e-4
    assert metrics["mean"]["psnr"] == pytest.approx(np.mean([v["psnr"] for v in metrics["views"].values()]))
    np.testing.assert_allclose(np.load(tmp_path / "moved" / "full" / f"{view.name}.npy"),
                               np.load(out / "full" / f"{view.name}.npy"), atol=1e-5)
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


@requires_lego
def test_synthetic_short_training_with_layer_metrics(tmp_path):
    m = short_manifest(tmp_path, "lego", config="configs/synthetic.yaml", scene=LEGO / "lego_both", overrides=[
        f"joint.dataset.gaussian_ground_truth_path={LEGO / 'lego_lego'}",
        f"joint.dataset.medium_ground_truth_path={LEGO / 'lego_clouds'}",
        "profile.spatial_colmap_fraction=0.5"])
    fit_iui3(m)
    run = tmp_path / "lego"
    assert json.loads((run / "joint/summary.json").read_text())["status"] == "completed"
    assert not (run / "water_fit.json").exists()  # no COLMAP tracks to fit the water on
    initial = torch.load(run / "joint/checkpoints/initialization.pt", map_location="cpu", weights_only=False)["model"]
    # without the fit every cell starts nearly clear, at medium_initial_density
    torch.testing.assert_close(torch.nn.functional.softplus(initial["medium.raw_density"]),
                               torch.full_like(initial["medium.raw_density"], 0.005))
    sites = initial["medium.sites"]
    assert (sites.abs() <= 1.3).all(1).float().mean() >= 0.45  # spatial_colmap_fraction: half in the object box
    evaluation = rows(run, "eval.csv")
    for name in ("lego_lego/psnr", "lego_clouds/psnr", "eval/hybrid_psnr"):
        assert finite(r.get(name) for r in evaluation), name


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
