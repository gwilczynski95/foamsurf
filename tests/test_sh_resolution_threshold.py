"""sigmoid(SH) Gaussian colour and the resolution-scaled densification threshold."""
import csv
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from foamsurf_model.gaussian_adaptation import GaussianAdaptationController
from foamsurf_model.gaussians import SurfaceGaussianModel
from train import fit_iui3, resolve_iui3
from training.config import ExperimentConfig, GaussianAdaptationConfig, ModelConfig

ROOT = Path(__file__).resolve().parents[1]
POINTS = np.array([[0, 0, 2.0], [1, 0, 3.0], [0, 1, 4.0]], dtype=np.float32)
RGB = np.array([[0.1, 0.5, 0.9], [0.7, 0.2, 0.3], [0.4, 0.4, 0.4]], dtype=np.float32)


def model(appearance, degree=3):
    config = ModelConfig(appearance=appearance, sh_degree=degree, gaussian_rgb_activation="sigmoid",
                         gaussian_initial_points=len(POINTS))
    return SurfaceGaussianModel(POINTS, RGB, config, "cpu")


def test_sigmoid_sh_starts_with_the_rgb_colours_from_every_direction():
    sh, rgb = model("sh"), model("rgb")
    for origin in (torch.zeros(3), torch.tensor([5.0, -3.0, 1.0])):
        torch.testing.assert_close(sh.colors(origin), torch.tensor(RGB), atol=1e-5, rtol=0)
        torch.testing.assert_close(sh.colors(origin), rgb.colors(origin), atol=1e-5, rtol=0)
    assert sh.appearance_rest.shape == (3, 3, 15)


def test_sigmoid_sh_is_view_dependent_bounded_and_trainable():
    sh = model("sh")
    with torch.no_grad():
        sh.appearance_rest.normal_(0, 2.0)
    a, b = sh.colors(torch.zeros(3)), sh.colors(torch.tensor([3.0, 3.0, 0.0]))
    assert (a - b).abs().max() > 0.05
    assert ((a > 0) & (a < 1)).all() and ((b > 0) & (b < 1)).all()
    a.sum().backward()
    assert sh.appearance_rest.grad.abs().sum() > 0 and sh.appearance_dc.grad.abs().sum() > 0
    sh.active_sh_degree = 0  # only the DC term: the colour no longer depends on the view
    torch.testing.assert_close(sh.colors(torch.zeros(3)), sh.colors(torch.tensor([3.0, 3.0, 0.0])))


def test_config_accepts_sigmoid_sh(tmp_path):
    config = ExperimentConfig()
    config.dataset.source_path = str(tmp_path)
    config.model.appearance, config.model.gaussian_rgb_activation = "sh", "sigmoid"
    config.validate()
    config.gaussian_adaptation.gradient_threshold_reference_width = 0
    with pytest.raises(ValueError, match="reference_width"):
        config.validate()


def adaptation(**settings):
    gaussians = model("rgb")
    optimizer = torch.optim.Adam([{"params": [p], "name": f"gaussian_{n}", "lr": 1e-3}
                                  for n, p in gaussians.named_primitive_parameters().items()])
    config = GaussianAdaptationConfig(enabled=True, gradient_threshold=2e-4,
                                      **{k: v for k, v in settings.items() if k != "width"})
    return GaussianAdaptationController(gaussians, optimizer, config, 10.0, image_width=settings.get("width"))


def test_threshold_scales_with_the_image_width():
    assert adaptation().gradient_threshold == 2e-4
    assert adaptation(width=1794).gradient_threshold == 2e-4  # off by default
    assert adaptation(gradient_threshold_reference_width=1383, width=1383).gradient_threshold == pytest.approx(2e-4)
    assert adaptation(gradient_threshold_reference_width=1383, width=1794).gradient_threshold == pytest.approx(
        2e-4 * 1383 / 1794)
    with pytest.raises(ValueError, match="image width"):
        adaptation(gradient_threshold_reference_width=1383)


def test_threshold_decides_densification():
    controller = adaptation(gradient_threshold_reference_width=1000, width=2000)  # threshold 1e-4
    controller.gradient_sum[:] = torch.tensor([1.5e-4, 0.5e-4, 3e-4])
    controller.gradient_denom[:] = 1
    plan = controller.propose_plan(controller.config.densify_from + controller.config.densification_interval)
    chosen = set(plan.clone_indices.tolist()) | set(plan.split_indices.tolist())
    assert chosen == {0, 2}


def test_sh_sweep_set(tmp_path):
    script = (ROOT / "scripts/train_seathru.sbatch").read_text()
    start = script.index("variant_overrides() {")
    function = script[start:script.index("\n}\n", start) + 3]
    lines = subprocess.run(["bash", "-c", function + "\nvariant_overrides _sh3_g1200k_dres"],
                           capture_output=True, text=True, check=True).stdout.splitlines()
    m = resolve_iui3(ROOT / "configs/iui3_default.yaml", tmp_path / "r", overrides=lines)
    assert (m["joint"]["model"]["appearance"], m["joint"]["model"]["sh_degree"]) == ("sh", 3)
    assert m["joint"]["optimization"]["gaussian_sh_degree_interval"] == 1000
    adaptation_config = m["joint"]["gaussian_adaptation"]
    assert (adaptation_config["max_points"], adaptation_config["gradient_threshold_reference_width"]) == (1200000, 1383)
    run = subprocess.run(["bash", "scripts/train_seathru.sbatch"], cwd=ROOT, capture_output=True, text=True,
                         env={"PATH": "/usr/bin:/bin", "FS_VARIANT_SET": "sh", "SLURM_ARRAY_TASK_ID": "20",
                              "SLURM_ARRAY_JOB_ID": "test"})
    assert "out of range" in run.stderr  # 5 variants x 4 scenes x 1 seed = tasks 0-19


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_sigmoid_sh_resolution_threshold_short_real_training(tmp_path):
    from test_medium_fix import _short_manifest
    m = _short_manifest(tmp_path, "sh", [
        "joint.model.appearance=sh", "joint.model.sh_degree=3", "joint.optimization.gaussian_sh_degree_interval=2",
        "joint.gaussian_adaptation.gradient_threshold_reference_width=1383",
        "warmup.background=[0.0, 0.0, 0.0]", "joint.background=[0.0, 0.0, 0.0]",
    ])
    fit_iui3(m)
    rows = list(csv.DictReader((tmp_path / "sh/joint/metrics/train.csv").open()))
    assert len(rows) == 4 and int(rows[-1]["model/active_sh_degree"]) == 2
