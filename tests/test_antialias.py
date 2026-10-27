"""Anti-aliasing: training rays jittered inside their pixel (_jit) and the Mip-Splatting 3D filter
(_mip), and the 'alias' set."""
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from foamsurf_data.rays import generate_camera_rays, generate_indexed_rays
from foamsurf_model.gaussians import SurfaceGaussianModel
from train import fit_iui3, resolve_iui3
from training.config import ExperimentConfig, ModelConfig
from training.sampler import RandomCameraRaySampler
from test_oceansplat import overrides

ROOT = Path(__file__).resolve().parents[1]
FX, FY = 40.0, 50.0


def camera(width=8, height=6, origin=(0.0, 0.0, 0.0)):
    return SimpleNamespace(origin=torch.tensor(origin), right=torch.tensor([1.0, 0, 0]), down=torch.tensor([0.0, 1, 0]),
                           forward=torch.tensor([0.0, 0, 1]), width=width, height=height, fov_x=1.0, fov_y=1.0,
                           intrinsics=(FX, FY, width / 2, height / 2), name="c",
                           foreground=torch.rand(height, width, 3), alpha=torch.ones(height, width, 1))


def pixel_coordinates(directions, cam):
    """Continuous pixel coordinates (x, y) of ray directions; the centre of pixel (i, j) is (i + .5, j + .5)."""
    z = directions[:, 2]
    return FX * directions[:, 0] / z + cam.width / 2, FY * directions[:, 1] / z + cam.height / 2


def test_indexed_rays_with_offsets_match_a_shifted_principal_point():
    cam = camera()
    index = torch.arange(cam.width * cam.height)
    one = lambda t: torch.as_tensor(t, dtype=torch.float32)[None]
    args = (one(cam.origin), one(cam.right), one(cam.down), one(cam.forward), one(1.0), one(1.0), cam.width, cam.height)
    intrinsics = torch.tensor([[FX, FY, cam.width / 2, cam.height / 2]])
    centre = generate_indexed_rays(index, *args, intrinsics=intrinsics)
    assert torch.equal(centre.directions, generate_indexed_rays(index, *args, intrinsics=intrinsics, pixel_offsets=None).directions)
    shifted = generate_indexed_rays(index, *args, intrinsics=intrinsics,
                                    pixel_offsets=torch.tensor([[0.25, -0.4]]).expand(len(index), 2))
    reference = generate_camera_rays(cam.origin, cam.right, cam.down, cam.forward, cam.width, cam.height, 1.0, 1.0,
                                     intrinsics=(FX, FY, cam.width / 2 - 0.25, cam.height / 2 + 0.4))
    torch.testing.assert_close(shifted.directions, reference.directions.reshape(-1, 3))


class FakeDataset:
    device = "cpu"

    def __init__(self, cam):
        self.train = [cam]

    def stacked_training_data(self, pixels=False):
        c = self.train[0]
        return dict(origins=c.origin[None], rights=c.right[None], downs=c.down[None], forwards=c.forward[None],
                    fov_x=torch.tensor([1.0]), fov_y=torch.tensor([1.0]), intrinsics=torch.tensor([c.intrinsics]))


@pytest.mark.parametrize("jitter", [False, True])
def test_sampler_rays_stay_in_their_pixel(jitter):
    torch.manual_seed(0)
    cam = camera(width=32, height=24)
    sampler = RandomCameraRaySampler(FakeDataset(cam), 0, (0.0, 0.0, 0.0), full_image=True, jitter=jitter)
    batch = sampler.next()
    x, y = pixel_coordinates(batch.rays.directions, cam)
    i = torch.arange(cam.width).repeat(cam.height).float(); j = torch.arange(cam.height).repeat_interleave(cam.width).float()
    dx, dy = x - i, y - j
    assert dx.min() >= -1e-4 and dx.max() < 1 + 1e-4 and dy.min() >= -1e-4 and dy.max() < 1 + 1e-4
    if jitter:  # uniform over the pixel: spread out, centred on the pixel centre
        assert dx.std() > 0.25 and abs(float(dx.mean()) - 0.5) < 0.05 and abs(float(dy.mean()) - 0.5) < 0.05
    else:
        torch.testing.assert_close(dx, torch.full_like(dx, 0.5), atol=1e-4, rtol=0)
    torch.testing.assert_close(batch.target, cam.foreground.reshape(-1, 3))  # the pixel value, not resampled


def gaussians(means, log_scales):
    model = SurfaceGaussianModel(np.asarray(means, np.float32), np.full((len(means), 3), 0.5, np.float32),
                                 ModelConfig(appearance="rgb"), "cpu")
    with torch.no_grad():
        model.log_scales.copy_(torch.tensor(log_scales, dtype=torch.float32))
        model.opacity_logits.fill_(float(torch.logit(torch.tensor(0.9))))
    return model


def test_mip_filter_size_from_the_nearest_camera_that_sees_the_gaussian():
    g = gaussians([[0.0, 0.0, 4.0], [0.0, 0.0, -2.0], [100.0, 0.0, 1.0]], [[-3.0, -3.0]] * 3)
    near_cam, far_cam = camera(), camera(origin=(0.0, 0.0, -6.0))
    g.update_mip_filter([near_cam, far_cam], 0.2, near=0.1)
    s = float(np.sqrt(0.2))
    # Gaussian 0: both cameras see it, the nearer one (z = 4) wins; 1: behind near_cam, far_cam at z = 4;
    # 2: outside both frames -> the largest value.
    torch.testing.assert_close(g.mip_filter, torch.tensor([s * 4 / FX, s * 4 / FX, s * 4 / FX]))
    g2 = gaussians([[0.0, 0.0, 4.0], [0.0, 0.0, 8.0]], [[-3.0, -3.0]] * 2)
    g2.update_mip_filter([near_cam], 0.2, near=0.1)
    torch.testing.assert_close(g2.mip_filter, torch.tensor([s * 4 / FX, s * 8 / FX]))


def test_mip_filter_render_scene_keeps_the_integral_and_spares_large_gaussians():
    g = gaussians([[0.0, 0.0, 4.0], [0.1, 0.0, 4.0]], [[np.log(1e-3), np.log(2e-3)], [np.log(1.0), np.log(0.5)]])
    plain = g.render_scene(None)
    g.mip_filter = torch.tensor([0.01, 0.01])
    filtered = g.render_scene(None)
    s, f = plain.log_scales.exp(), filtered.log_scales.exp()
    torch.testing.assert_close(f, (s ** 2 + 0.01 ** 2).sqrt())
    o, of = torch.sigmoid(plain.opacity_logits), torch.sigmoid(filtered.opacity_logits)
    torch.testing.assert_close(of * f.prod(-1, keepdim=True), o * s.prod(-1, keepdim=True))  # integral kept
    assert float(of[0]) < 0.9 * 0.03  # a sub-filter Gaussian becomes a faint, filter-sized blob
    torch.testing.assert_close(f[1], s[1], rtol=1e-3, atol=0)  # 100x the filter: unchanged
    torch.testing.assert_close(of[1], o[1], rtol=1e-3, atol=0)
    (filtered.log_scales.sum() + filtered.opacity_logits.sum()).backward()
    assert g.log_scales.grad.abs().sum() > 0 and g.opacity_logits.grad.abs().sum() > 0
    g.mip_filter = torch.tensor([0.01])  # stale after a population change
    with pytest.raises(RuntimeError, match="stale"):
        g.render_scene(None)


def test_suffixes_validation_and_set(tmp_path):
    assert overrides("_jit") == ["joint.optimization.ray_jitter=true"]
    assert overrides("_mip") == ["joint.model.gaussian_mip_filter=0.2"]
    assert overrides("_minit") == ["joint.model.medium_initial_color_from_data=true"]  # not _mip
    m = resolve_iui3(ROOT / "configs/iui3_default.yaml", tmp_path / "a", overrides=overrides("_clamp05_minit_rel_jit_mip"))
    assert m["joint"]["optimization"]["ray_jitter"] is True and m["joint"]["model"]["gaussian_mip_filter"] == 0.2
    config = ExperimentConfig()
    config.dataset.source_path = str(tmp_path)
    config.model.gaussian_mip_filter = -0.1
    with pytest.raises(ValueError, match="mip"):
        config.validate()
    config.model.gaussian_mip_filter = 0.0
    config.optimization.ray_jitter = True
    config.optimization.training_downscale_levels = 2
    with pytest.raises(ValueError, match="jitter"):
        config.validate()
    env = {"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin", "FS_VARIANT_SET": "alias",
           "SLURM_ARRAY_JOB_ID": "test", "FS_DRY_RUN": "1",
           "FS_DATASETS": os.environ.get("FS_DATASETS", "/shared/sets/datasets/SeathruNeRF_dataset")}
    run = lambda task: subprocess.run(["bash", "scripts/train_seathru.sbatch"], cwd=ROOT, capture_output=True,
                                      text=True, env={**env, "SLURM_ARRAY_TASK_ID": str(task)})
    assert "out of range" in run(24).stderr  # 3 variants x 4 scenes x 2 seeds
    assert run(23).returncode == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_short_real_training_with_jitter_and_mip_filter_and_resume(tmp_path):
    from test_medium_fix import _short_manifest
    from training.trainer import Trainer
    m = _short_manifest(tmp_path, "alias", overrides("_jit_mip"))
    fit_iui3(m)  # densification every iteration: the filter must follow the population
    root = tmp_path / "alias/joint"
    config = ExperimentConfig.load(root / "config.yaml")
    assert config.optimization.ray_jitter and config.model.gaussian_mip_filter == 0.2
    config.output_dir = str(tmp_path / "resumed")
    trainer = Trainer(config, resume=root / "checkpoints/last.pt")
    g = trainer.model.gaussians
    assert g.mip_filter is not None and g.mip_filter.shape == (g.point_count,) and bool((g.mip_filter > 0).all())
    summary, _, _ = trainer.evaluator.evaluate(4)
    assert np.isfinite(summary["eval/primary_psnr"])
