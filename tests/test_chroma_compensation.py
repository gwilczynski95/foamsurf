"""Attenuation-compensated densification and the chroma sweep (chroma prior on all rays)."""
import csv
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from foamsurf_model.gaussian_adaptation import GaussianAdaptationController
from foamsurf_model.gaussians import SurfaceGaussianModel
from train import fit_iui3, resolve_iui3
from training.config import ExperimentConfig, GaussianAdaptationConfig, ModelConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
POINTS = np.array([[0, 0, 1], [0, 0, 10], [0, 0, 100]], dtype=np.float32)


def controller(**settings):
    model = SurfaceGaussianModel(POINTS, np.full_like(POINTS, 0.5),
                                 ModelConfig(appearance="rgb", gaussian_initial_points=len(POINTS)), "cpu")
    optimizer = torch.optim.Adam([{"params": [p], "name": f"gaussian_{n}", "lr": 1e-3}
                                  for n, p in model.named_primitive_parameters().items()])
    return GaussianAdaptationController(model, optimizer, GaussianAdaptationConfig(enabled=True, **settings), 10.0)


def test_compensation_is_inverse_mean_transmittance_capped():
    adaptation = controller(gradient_attenuation_compensation=True, gradient_compensation_max=20.0)
    adaptation.medium_attenuation = torch.tensor([0.1, 0.05, 0.02])
    factor = adaptation.attenuation_compensation(torch.zeros(3))
    for distance, value in zip((1.0, 10.0), factor[:2].tolist()):
        expected = 1 / np.mean(np.exp(-distance * np.array([0.1, 0.05, 0.02])))
        assert value == pytest.approx(expected, rel=1e-5)
    assert factor[2].item() == pytest.approx(20.0)  # 100 units of water: capped
    adaptation.medium_attenuation = None
    with pytest.raises(RuntimeError, match="medium_attenuation"):
        adaptation.attenuation_compensation(torch.zeros(3))


@pytest.mark.parametrize("compensate", [False, True])
def test_collect_statistics_scales_the_gradient(compensate):
    adaptation = controller(gradient_attenuation_compensation=compensate)
    adaptation.medium_attenuation = torch.tensor([0.05, 0.05, 0.05])
    adaptation.projected_radii = lambda camera: torch.ones(3)  # all visible
    gradient = torch.ones(3, 2)
    camera = SimpleNamespace(origin=torch.zeros(3))
    adaptation.collect_statistics(gradient, torch.zeros(3, 4), camera)
    expected = np.sqrt(2) * (np.exp(0.05 * np.array([1.0, 10.0, 100.0])).clip(max=20.0) if compensate else 1.0)
    assert np.allclose(adaptation.gradient_sum.numpy(), expected, rtol=1e-5)
    assert adaptation.gradient_denom.tolist() == [1, 1, 1]


def test_water_attenuation_is_median_over_non_guard_cells():
    from training.trainer import Trainer
    raw = torch.tensor([[0.1, 0.2, 0.3, 9, 9, 9], [0.2, 0.4, 0.6, 9, 9, 9], [0.3, 0.6, 0.9, 9, 9, 9],
                        [50.0, 50.0, 50.0, 9, 9, 9]])
    medium = SimpleNamespace(optical_coefficients=raw, is_rgb=True,
                             guard_mask=torch.tensor([False, False, False, True]))
    fake = SimpleNamespace(model=SimpleNamespace(medium=medium))
    assert torch.allclose(Trainer._water_attenuation(fake), torch.tensor([0.2, 0.4, 0.6]))


def test_compensation_config(tmp_path):
    assert GaussianAdaptationConfig().gradient_attenuation_compensation is False
    config = ExperimentConfig()
    config.dataset.source_path = str(tmp_path)
    config.gaussian_adaptation.gradient_compensation_max = 0.5
    with pytest.raises(ValueError, match="gradient_compensation_max"):
        config.validate()


def sweep(command):
    script = (ROOT / "scripts/sweep_medium_fix.sbatch").read_text()
    start = script.index("variant_overrides() {")
    function = script[start:script.index("\n}\n", start) + 3]
    names = script[script.index("CHROMA_VARIANTS=()"):script.index('case "${FS_VARIANT_SET')]
    return subprocess.run(["bash", "-c", function + names + "\n" + command],
                          capture_output=True, text=True, check=True).stdout.split("\n")


def test_chroma_sweep_names_and_overrides(tmp_path):
    variants = [v for v in sweep('printf "%s\\n" "${CHROMA_VARIANTS[@]}"') if v]
    assert len(variants) == 22 == len(set(variants)) and variants[0] == "B0_chroma100_scov3e3_cthr90_g700k_water1e2_bk"
    assert sum("_tden" in v for v in variants) == 10 and sum("_chthr02" in v for v in variants) == 18
    name = "B0_chroma1000_chthr02_scov3e3_cthr90_g700k_tden_water1e2_bk_seed43"
    lines = [line for line in sweep(f"variant_overrides {name}") if line]
    m = resolve_iui3(ROOT / "configs/iui3_5k.yaml", tmp_path / "r", gaussian_rgb_activation="sigmoid",
                     medium_color_activation="sigmoid", overrides=lines)
    fix = m["medium_fix"]
    assert fix["chroma_weight"] == pytest.approx(10.0) and fix["chroma_alpha_threshold"] == pytest.approx(0.02)
    assert fix["surface_coverage_threshold"] == pytest.approx(0.9) and fix["water_clear_weight"] == pytest.approx(0.01)
    assert m["joint"]["gaussian_adaptation"]["gradient_attenuation_compensation"] is True
    assert m["joint"]["seed"] == 43


def test_report_chroma_matrix_and_names():
    import medium_fix_report as report
    variants = [v for v in sweep('printf "%s\\n" "${CHROMA_VARIANTS[@]}"') if v]
    assert all(report.baseline_name(v.split("_seed")[0]) == report.CHROMA_BASE for v in variants[2:])
    assert "na promieniach z α > 0,02" in report.describe("B0_chroma030_chthr02_scov3e3_cthr90_g700k_water1e2_bk")
    assert "kompensowana" in report.describe("B0_chroma100_scov3e3_cthr90_g700k_tden_water1e2_bk")
    runs = [dict(name=n, variant=n.split("_seed")[0], best_psnr=29.0, path=Path("/nonexistent")) for n in variants]
    html = report.matrices(runs, runs)
    for name in variants:
        assert f"title='{name}'" in html, name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_compensated_densification_short_real_training(tmp_path):
    from test_medium_fix import _short_manifest
    m = _short_manifest(tmp_path, "tden", [
        "joint.gaussian_adaptation.gradient_attenuation_compensation=true",
        "medium_fix.chroma_weight=1.0", "medium_fix.chroma_alpha_threshold=0.02", "medium_fix.chroma_start=0",
        "medium_fix.chroma_ramp=0", "medium_fix.prior_rays=64",
        "warmup.background=[0.0, 0.0, 0.0]", "joint.background=[0.0, 0.0, 0.0]",
    ])
    fit_iui3(m)
    rows = list(csv.DictReader((tmp_path / "tden/joint/metrics/train.csv").open()))
    assert len(rows) == 4 and all(np.isfinite(float(row["train/loss"])) for row in rows)
