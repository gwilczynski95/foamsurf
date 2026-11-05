"""Gaussian colour ceiling (model.gaussian_rgb_scale, the _rgbs<NN> suffix): colour = scale x sigmoid. With scale 1 a
white object behind water cannot reach the photo (Curacao colour chart: Gaussians at 1.0, ~45/255 too dark); the
lifted ceiling keeps colours bounded, starts from the same colours and is 1 for older checkpoints."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from foamsurf_model.gaussians import SurfaceGaussianModel
from test_oceansplat import overrides
from train import resolve_iui3
from training.config import ExperimentConfig, ModelConfig

ROOT = Path(__file__).resolve().parents[1]
POINTS = np.array([[0, 0, 2.0], [1, 0, 3.0], [0, 1, 4.0]], dtype=np.float32)
RGB = np.array([[0.1, 0.5, 0.9], [0.7, 0.2, 0.3], [0.95, 0.99, 0.4]], dtype=np.float32)


def model(appearance, scale, device="cpu"):
    config = ModelConfig(appearance=appearance, sh_degree=3, gaussian_rgb_activation="sigmoid",
                         gaussian_rgb_scale=scale, gaussian_initial_points=len(POINTS))
    return SurfaceGaussianModel(POINTS, RGB, config, device)


@pytest.mark.parametrize("appearance", ["sh", "rgb"])
def test_starts_from_the_same_colours_and_the_ceiling_moves(appearance):
    for scale in (1.0, 1.5, 2.0):
        m = model(appearance, scale)
        torch.testing.assert_close(m.colors(torch.zeros(3)), torch.tensor(RGB), atol=1e-5, rtol=0)
        with torch.no_grad():
            m.appearance_dc.fill_(50.0)  # far into saturation
        top = m.colors(torch.tensor([2.0, -1.0, 0.0]))
        torch.testing.assert_close(top, torch.full_like(top, scale), atol=1e-4, rtol=0)  # bounded at the scale


def test_gradient_still_flows_near_one_with_a_lifted_ceiling():
    # at colour 0.99 the plain sigmoid is nearly flat; with scale 1.5 the same colour sits mid-curve
    plain, lifted = model("sh", 1.0), model("sh", 1.5)
    for m in (plain, lifted):
        m.colors(torch.zeros(3))[2, 1].backward()
    assert float(lifted.appearance_dc.grad[2, 1]) > 10 * float(plain.appearance_dc.grad[2, 1])


def test_checkpoint_keeps_the_scale_and_older_checkpoints_mean_one():
    m = model("sh", 1.5)
    with torch.no_grad():
        m.appearance_rest.normal_(0, 1.0)
    state = {k: v.detach().clone() for k, v in m.state_dict().items()}
    meta = m.checkpoint_metadata()
    assert meta["rgb_scale"] == 1.5
    loaded = SurfaceGaussianModel.from_checkpoint(state, meta, "cpu")
    origin = torch.tensor([1.0, 2.0, -3.0])
    torch.testing.assert_close(loaded.colors(origin), m.colors(origin))
    old = dict(meta); old.pop("rgb_scale")
    assert SurfaceGaussianModel.from_checkpoint(state, old, "cpu").rgb_scale == 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_cuda_kernel_matches_the_python_colours():
    m = model("sh", 1.5, "cuda")
    with torch.no_grad():
        m.appearance_rest.normal_(0, 1.0)
        m.appearance_dc.normal_(2.0, 3.0)
    origin = torch.tensor([1.0, 2.0, -3.0], device="cuda")
    with torch.no_grad():
        kernel = m.colors(origin)  # no grad on CUDA: the CUDA kernel path
    python = m.colors(origin)  # grad enabled: the PyTorch path
    torch.testing.assert_close(kernel, python.detach(), atol=1e-5, rtol=0)
    assert float(kernel.max()) > 1.0


def test_config_validation():
    config = ExperimentConfig()
    config.model.appearance, config.model.gaussian_rgb_activation = "sh", "sigmoid"
    config.model.gaussian_rgb_scale = 1.5
    config.dataset.source_path = "/tmp"
    config.validate()
    config.model.gaussian_rgb_scale = 0.8
    with pytest.raises(ValueError, match="gaussian_rgb_scale"):
        config.validate()
    config.model.gaussian_rgb_scale, config.model.gaussian_rgb_activation = 1.5, "identity"
    with pytest.raises(ValueError, match="gaussian_rgb_scale"):
        config.validate()
    assert ModelConfig().gaussian_rgb_scale == 1.0


def test_suffix_and_set(tmp_path):
    assert overrides("_rgbs15") == ["joint.model.gaussian_rgb_scale=1.5"]
    assert overrides("_rgbs20") == ["joint.model.gaussian_rgb_scale=2.0"]
    m = resolve_iui3(ROOT / "configs/seathru_default.yaml", tmp_path / "r",
                     overrides=overrides("_cull05r_bin01_wphase_rgbs15_wext"))
    assert m["joint"]["model"]["gaussian_rgb_scale"] == 1.5 and m["joint"]["model"]["gaussian_rgb_activation"] == "sigmoid"
    assert m["joint"]["optimization"]["water_extrapolation"] is True and m["joint"]["optimization"]["water_phase"] is True
    env = {"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin", "SLURM_ARRAY_JOB_ID": "test", "FS_DRY_RUN": "1",
           "FS_VARIANT_SET": "rgbs", "FS_DATASETS": os.environ.get("FS_DATASETS", "/shared/sets/datasets/SeathruNeRF_dataset")}
    run = lambda task: subprocess.run(["bash", "scripts/train_seathru.sbatch"], cwd=ROOT, capture_output=True,
                                      text=True, env={**env, "SLURM_ARRAY_TASK_ID": str(task)})
    assert run(7).returncode == 0 and "Panama-Undisorted_cull05r_bin01_wphase_rgbs15_wext_seed43" in run(7).stdout
    assert "out of range" in run(8).stderr
    cov = lambda task, **extra: subprocess.run(["bash", "scripts/train_seathru.sbatch"], cwd=ROOT, capture_output=True, text=True,
                                               env={**env, "FS_VARIANT_SET": "rgbscov", "SLURM_ARRAY_TASK_ID": str(task), **extra})
    assert "Panama-Undisorted_cull05r_bin01_wphase_rgbs15_cov09_wext_seed44" in cov(11).stdout
    assert "out of range" in cov(12).stderr
    seed44 = subprocess.run(["bash", "scripts/train_seathru.sbatch"], cwd=ROOT, capture_output=True, text=True,
                            env={**env, "FS_SEEDS": "44", "SLURM_ARRAY_TASK_ID": "3"})
    assert "Panama-Undisorted_cull05r_bin01_wphase_rgbs15_wext_seed44" in seed44.stdout
    m = resolve_iui3(ROOT / "configs/seathru_default.yaml", tmp_path / "c",
                     overrides=overrides("_cull05r_bin01_wphase_rgbs15_cov09_wext"))
    assert m["medium_fix"]["surface_coverage_threshold"] == 0.9 and m["medium_fix"]["surface_coverage_weight"] == 0.03
    assert m["joint"]["model"]["gaussian_rgb_scale"] == 1.5
