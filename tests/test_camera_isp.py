"""The camera after the water (training/camera_isp.py, PPISP): identity at the start, per-photo exposure vs
neutral test views, shared vignetting, checkpoint round trip, pixel coordinates of the training batches,
the OceanSplat hook, the _isp suffix and a short real training run with resume."""
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from train import fit_iui3, resolve_iui3
from training import oceansplat as oc
from training.config import ExperimentConfig
from test_oceansplat import PlaneRenderer, overrides, plane_camera, plane_scene

ROOT = Path(__file__).resolve().parents[1]
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")


@gpu
def test_identity_at_start_then_per_photo_exposure_and_neutral_test_views():
    from training.camera_isp import CameraISP, pixel_grid
    isp = CameraISP(["a", "b"], iterations=100)
    rgb = torch.rand(6 * 8, 3, device="cuda") * 0.8 + 0.1
    coords = pixel_grid(6, 8, "cuda")
    for frame in ("a", "b", None):  # PPISP starts at the identity
        torch.testing.assert_close(isp.apply(rgb, coords, (6, 8), frame), rgb, atol=1e-4, rtol=0)
    with torch.no_grad():
        isp.module.exposure_params[0] = 1.0  # photo "a": +1 stop
    torch.testing.assert_close(isp.apply(rgb, coords, (6, 8), "a"), (2 * rgb).clamp(max=1), atol=1e-3, rtol=0)
    torch.testing.assert_close(isp.apply(rgb, coords, (6, 8), None), rgb, atol=1e-4, rtol=0)  # test view: neutral
    with torch.no_grad():
        isp.module.exposure_params.zero_()
        isp.module.vignetting_params[0, :, 2] = -0.5  # shared vignetting reaches test views too
    neutral = isp.apply(rgb, coords, (6, 8), None)
    assert bool((neutral <= rgb + 1e-5).all()) and float((rgb - neutral).max()) > 0.01
    rgb.requires_grad_(True)
    isp.zero_grad()
    (isp.apply(rgb, coords, (6, 8), "b").sum() + isp.regularization()).backward()
    assert rgb.grad.abs().sum() > 0 and isp.module.vignetting_params.grad.abs().sum() > 0


@gpu
def test_state_round_trip_and_frame_check():
    from training.camera_isp import CameraISP
    a = CameraISP(["x", "y", "z"], iterations=50)
    with torch.no_grad():
        a.module.exposure_params.copy_(torch.tensor([0.1, -0.2, 0.3]))
    b = CameraISP(["x", "y", "z"], iterations=50)
    b.load_state_dict(a.state_dict())
    torch.testing.assert_close(b.module.exposure_params, a.module.exposure_params)
    with pytest.raises(ValueError, match="other photos"):
        CameraISP(["x", "y"], iterations=50).load_state_dict(a.state_dict())


def test_training_batch_pixel_coordinates():
    from training.trainer import Trainer
    seen = {}
    fake = SimpleNamespace(camera_isp=SimpleNamespace(apply=lambda image, coords, size, frame: seen.update(
        coords=coords, size=size, frame=frame) or image))
    camera = SimpleNamespace(height=3, width=4, name="p")
    index = torch.tensor([0, 5, 11])  # (x, y) = (0, 0), (1, 1), (3, 2)
    batch = SimpleNamespace(camera=camera, image_size=None, rays=SimpleNamespace(pixel_indices=index))
    Trainer._apply_camera_isp(fake, torch.zeros(3, 3), batch)
    torch.testing.assert_close(seen["coords"], torch.tensor([[0.5, 0.5], [1.5, 1.5], [3.5, 2.5]]))
    assert seen["size"] == (3, 4) and seen["frame"] == "p"
    batch = SimpleNamespace(camera=camera, image_size=(2, 2), rays=SimpleNamespace(pixel_indices=None))
    Trainer._apply_camera_isp(fake, torch.zeros(4, 3), batch)  # resolution schedule: whole image at 1/2
    torch.testing.assert_close(seen["coords"], torch.tensor([[0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5]]))
    assert seen["size"] == (2, 2)
    assert Trainer._apply_camera_isp(SimpleNamespace(camera_isp=None), torch.ones(2, 3), batch).sum() == 6


def test_ocean_full_term_goes_through_the_camera_model():
    scene = plane_scene()
    camera = plane_camera()
    renderer = PlaneRenderer()
    target = renderer.render(oc.image_rays(camera, 1)).direct.reshape(48, 64, 3) + 0.05
    run = lambda model: oc.ocean_terms(renderer, lambda origin: scene, None, (0, 0, 0), camera, target, 0.6, 1, 1e-3,
                                       1.0, 1000, generator=torch.Generator().manual_seed(1), camera_model=model)
    plain, darker = run(None), run(lambda image: 0.5 * image)
    assert darker["full"].item() > plain["full"].item() + 0.1  # the photo is compared with the camera's image
    torch.testing.assert_close(darker["obj"], plain["obj"])  # renders are compared raw


def test_suffix_and_set(tmp_path):
    assert overrides("_isp") == ["joint.optimization.camera_isp=true"]
    m = resolve_iui3(ROOT / "configs/iui3_default.yaml", tmp_path / "i", overrides=overrides("_clamp05_minit_rs_rel_oceane_isp"))
    assert m["joint"]["optimization"]["camera_isp"] is True and m["joint"]["optimization"]["ocean_from"] == 4000
    env = {"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin", "FS_VARIANT_SET": "isp",
           "SLURM_ARRAY_JOB_ID": "test", "FS_DRY_RUN": "1",
           "FS_DATASETS": os.environ.get("FS_DATASETS", "/shared/sets/datasets/SeathruNeRF_dataset")}
    run = lambda task: subprocess.run(["bash", "scripts/train_seathru.sbatch"], cwd=ROOT, capture_output=True,
                                      text=True, env={**env, "SLURM_ARRAY_TASK_ID": str(task)})
    assert "out of range" in run(8).stderr  # 1 variant x 4 scenes x 2 seeds
    assert run(7).returncode == 0


@gpu
def test_short_real_training_with_camera_isp_and_resume(tmp_path):
    from test_medium_fix import _short_manifest
    from training.trainer import Trainer
    m = _short_manifest(tmp_path, "isp", overrides("_isp_ocean") + [
        "joint.optimization.ocean_from=1", "joint.optimization.ocean_until=5", "joint.optimization.ocean_downscale=1",
        "warmup.background=[0.0, 0.0, 0.0]", "joint.background=[0.0, 0.0, 0.0]"])
    fit_iui3(m)
    root = tmp_path / "isp/joint"
    checkpoint = torch.load(root / "checkpoints/last.pt", map_location="cpu", weights_only=False)
    assert checkpoint["iteration"] == 4 and "camera_isp" in checkpoint and "alpha_mlp" in checkpoint
    config = ExperimentConfig.load(root / "config.yaml")
    config.output_dir = str(tmp_path / "resumed")
    trainer = Trainer(config, resume=root / "checkpoints/last.pt")
    torch.testing.assert_close(trainer.camera_isp.module.exposure_params.cpu(),
                               checkpoint["camera_isp"]["module"]["exposure_params"])
    summary, _, _ = trainer.evaluator.evaluate(4)
    assert torch.isfinite(torch.tensor(summary["eval/primary_psnr"]))
