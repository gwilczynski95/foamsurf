"""Water clearing (push Gaussians off open-water rays), the coverage end and the water sweep."""
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from train import fit_iui3, resolve_iui3
from training.losses import MediumFixConfig, surface_coverage_loss, water_clear_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_water_clear_loss_pushes_only_rays_below_threshold():
    alpha = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.6, 0.95], requires_grad=True)
    loss, fraction = water_clear_loss(alpha, 0.3)
    assert loss.item() == pytest.approx(0.1) and fraction == pytest.approx(0.5)
    loss.backward()
    assert torch.allclose(alpha.grad, torch.tensor([1, 1, 1, 0, 0, 0]) / 3.0)
    # Mirror of the coverage loss: together with a higher coverage threshold no ray gets both.
    cover, _ = surface_coverage_loss(alpha, 0.9)
    assert cover.item() == pytest.approx(0.05)
    empty, none = water_clear_loss(torch.ones(4, requires_grad=True), 0.3)
    assert empty.item() == 0 and none == 0.0


def test_water_clear_schedule_and_coverage_end():
    config = MediumFixConfig(water_clear_weight=0.1, surface_coverage_weight=3e-3, surface_coverage_threshold=0.9,
                             surface_coverage_end=15000)
    assert config.water_clear_weight_at(3999) == 0
    assert config.water_clear_weight_at(5000) == pytest.approx(0.05)
    assert config.water_clear_weight_at(6000) == pytest.approx(0.1)
    assert config.surface_coverage_weight_at(14999) == pytest.approx(3e-3)
    assert config.surface_coverage_weight_at(15000) == 0
    assert MediumFixConfig().water_clear_weight == 0 and MediumFixConfig().surface_coverage_end is None


@pytest.mark.parametrize("bad", [
    dict(water_clear_weight=-1.0), dict(water_clear_threshold=0.0), dict(water_clear_ramp=-1),
    dict(surface_coverage_end=500), dict(surface_coverage_end=2.5e4),
    # Opposite pulls on the same rays.
    dict(water_clear_weight=0.1, surface_coverage_weight=3e-3, water_clear_threshold=0.5),
])
def test_water_clear_validation(bad):
    with pytest.raises(ValueError):
        MediumFixConfig(**bad)


def test_old_manifest_resolves_new_fields_to_inactive_defaults(tmp_path):
    from train import normalize_saved_manifest
    manifest = resolve_iui3(ROOT / "configs/iui3_default.yaml", tmp_path / "run")
    old = json.loads(json.dumps(manifest))
    for key in ("water_clear_weight", "water_clear_start", "water_clear_ramp", "water_clear_threshold",
                "surface_coverage_end"):
        del old["medium_fix"][key]
    assert normalize_saved_manifest(old) == manifest


def sweep_function():
    script = (ROOT / "scripts/sweep_medium_fix.sbatch").read_text()
    start = script.index("variant_overrides() {")
    function = script[start:script.index("\n}\n", start) + 3]
    names = script[script.index("WATER_VARIANTS=()"):script.index('case "${FS_VARIANT_SET')]
    return function + names


def test_water_sweep_names_and_overrides(tmp_path):
    command = sweep_function() + '\nprintf "%s\\n" "${WATER_VARIANTS[@]}"\n'
    variants = subprocess.run(["bash", "-c", command], capture_output=True, text=True, check=True).stdout.split()
    assert len(variants) == 14 and len(set(variants)) == 14 and variants[0] == "B0_chroma100_g700k_bk"
    assert sum(v.endswith("_seed43") for v in variants) == 7
    weights = {}
    for variant in ("B0_chroma100_g700k_water1e3_bk", "B0_chroma100_g700k_water1e0_bk",
                    "B0_chroma100_scov3e3_cthr90_g700k_water1e2_bk", "B0_chroma100_scov3e3_send15k_g700k_bk"):
        lines = subprocess.run(["bash", "-c", sweep_function() + f"\nvariant_overrides {variant}"],
                               capture_output=True, text=True, check=True).stdout.splitlines()
        manifest = resolve_iui3(ROOT / "configs/iui3_5k.yaml", tmp_path / variant, gaussian_rgb_activation="sigmoid",
                                medium_color_activation="sigmoid", overrides=lines)
        weights[variant] = manifest["medium_fix"]
        assert manifest["joint"]["gaussian_adaptation"]["max_points"] == 700000
        assert manifest["joint"]["background"] == [0.0, 0.0, 0.0]
    assert weights["B0_chroma100_g700k_water1e3_bk"]["water_clear_weight"] == pytest.approx(1e-3)
    assert weights["B0_chroma100_g700k_water1e0_bk"]["water_clear_weight"] == pytest.approx(1.0)
    assert weights["B0_chroma100_g700k_water1e0_bk"]["surface_coverage_weight"] == 0
    seabed = weights["B0_chroma100_scov3e3_cthr90_g700k_water1e2_bk"]
    assert (seabed["surface_coverage_threshold"], seabed["water_clear_weight"]) == (pytest.approx(0.9), pytest.approx(1e-2))
    assert weights["B0_chroma100_scov3e3_send15k_g700k_bk"]["surface_coverage_end"] == 15000


def test_report_names_water_variants():
    from medium_fix_report import WATER_BASE, baseline_name, describe, parse_variant
    assert parse_variant("B0_chroma100_scov3e3_cthr90_g700k_water1e2_bk")[4] == "_chroma100_scov3e3_cthr90_g700k_water1e2"
    for variant in ("B0_chroma100_g700k_water1e3_bk", "B0_chroma100_scov3e3_send15k_g700k_bk"):
        assert baseline_name(variant) == WATER_BASE == "B0_chroma100_g700k_bk"
    assert "strata wodna 1e-3" in describe("B0_chroma100_g700k_water1e3_bk")
    assert "wyłączone od 15k" in describe("B0_chroma100_scov3e3_send15k_g700k_bk")


def test_report_water_history_from_alpha_maps(tmp_path):
    import medium_fix_report as report
    views = list(report.water_regions())
    run = tmp_path / "run"
    for iteration, value in ((1000, 1.0), (1500, 0.0), (2000, 0.0)):
        folder = run / "joint/evaluation" / f"iteration_{iteration:06d}"
        folder.mkdir(parents=True)
        for index, view in enumerate(views):
            alpha = np.full(report.water_regions()[view]["water"].shape + (1,), value, dtype=np.float32)
            np.savez(folder / f"{index:04d}_{view}_gaussian_alpha.npz", alpha=alpha)
    history = report.water_history(dict(path=run))
    assert [row["iteration"] for row in history] == [1000, 2000]  # every 1000 iterations
    assert history[0] == dict(iteration=1000, water=1.0, band=1.0, seabed_band=1.0)
    assert history[1] == dict(iteration=2000, water=0.0, band=0.0, seabed_band=0.0)
    assert (run / "audit/water_history_1000.json").is_file()
    assert report.water_history(dict(path=run)) == history  # cached
    regions = report.water_regions()[views[0]]
    assert not (regions["band"] & regions["seabed_band"]).any() and (regions["band"] <= regions["water"]).all()


def test_line_chart_ticks_and_units():
    import medium_fix_report as report
    assert report.nice_step(100) == 20 and report.nice_step(3) == 0.5 and report.nice_step(0.2) == 0.05
    chart = report.line_chart("c", "t", [dict(name="a", curve=[(0, 10.0), (1000, 90.0)])], curve_from=0, unit="%")
    assert '&quot;unit&quot;: &quot;%&quot;' in chart and "<polyline" in chart


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_water_clear_short_real_training(tmp_path):
    from test_medium_fix import _short_manifest
    m = _short_manifest(tmp_path, "water", [
        "medium_fix.prior_rays=64", "medium_fix.water_clear_weight=1.0", "medium_fix.water_clear_start=0",
        "medium_fix.water_clear_ramp=0", "medium_fix.water_clear_threshold=0.99",
        "warmup.background=[0.0, 0.0, 0.0]", "joint.background=[0.0, 0.0, 0.0]",
    ])
    fit_iui3(m)
    fit_iui3(m, resume=True)
    rows = list(csv.DictReader((tmp_path / "water/joint/metrics/train.csv").open()))
    assert len(rows) == 4
    assert all(float(row["medium_fix/water_clear_weight"]) == 1.0 for row in rows)
    assert all(math.isfinite(float(row["medium_fix/water_clear_raw"])) for row in rows)
    assert all(0 <= float(row["medium_fix/water_clear_ray_fraction"]) <= 1 for row in rows)
    assert "medium_fix/surface_coverage_weight" not in rows[0]  # only enabled priors add columns
