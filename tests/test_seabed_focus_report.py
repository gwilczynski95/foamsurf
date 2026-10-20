"""Selection and drawing helpers of scripts/seabed_focus_report.py."""
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import seabed_focus_report as focus  # noqa: E402
from sam_water_mask import boundary  # noqa: E402

BASE = "B0_chroma100_scov3e3_cthr90_g700k_water1e2_bk"


def entry(psnr, water, ring0, ring1):
    values = {key: [float("nan")] * 2 for key, *_ in focus.METRICS}
    values.update(best_psnr=[psnr - 0.1, psnr + 0.1], **{"sam_water_gt_0.5": [water] * 2,
                  "sam_seabed_0_30_gt_0.5": [ring0] * 2, "sam_seabed_30_60_gt_0.5": [ring1] * 2})
    return dict(values=values, runs=[])


def test_promising_keeps_water_and_psnr_then_ranks_by_boundary():
    variants = {BASE: entry(29.2, 0.0, 0.05, 0.3),
                "a": entry(29.3, 0.001, 0.25, 0.65),   # best boundary
                "b": entry(29.3, 0.02, 0.40, 0.80),    # water too dirty
                "c": entry(28.9, 0.0, 0.40, 0.80),     # PSNR too low
                "d": entry(29.15, 0.0, 0.10, 0.40)}
    assert focus.promising(variants, BASE, 3) == ["a", "d"]
    assert math.isclose(focus.boundary_score(variants["a"]), 0.45)


def test_factors_and_short_labels():
    assert focus.factors(BASE) == ("1 (α>0,5)", False)
    name = "B0_chroma010_chthr02_scov3e3_cthr90_g700k_tden_water1e2_bk"
    assert focus.factors(name) == ("0,1 (wszystkie promienie)", True)
    assert focus.short_label(name, BASE) == "chroma 0,1 na wszystkich promieniach + kompensacja densyfikacji"
    assert focus.short_label(BASE, BASE) == "baza"
    assert focus.factors("B0_chroma100_g700k_bk") is None


def test_coverage_map_colours_regions_and_boundary_without_wrap():
    water = np.zeros((6, 8), bool)
    water[:3] = True
    alpha = np.zeros((6, 8), np.float32)
    alpha[4:, :4] = 1.0  # covered seabed on the left
    alpha[0, 0] = 1.0  # one Gaussian pixel in the water
    target = Image.fromarray(np.full((6, 8, 3), 200, np.uint8))
    picture = np.asarray(focus.coverage_map(target, alpha, water)).astype(int)
    assert (picture[3] == 255).all() and not (picture[0] == 255).all()  # boundary row only, no wrap
    assert picture[5, 0, 1] > picture[5, 0, 0]  # covered seabed: green-ish
    assert picture[5, 7, 0] > picture[5, 7, 2]  # empty seabed: orange-ish
    assert picture[0, 0, 2] > picture[0, 0, 1]  # water taken: violet-ish
    assert boundary(water).sum() == 8


def test_factor_chart_draws_points_for_psnr_and_bars_otherwise():
    variants = {BASE: entry(29.2, 0.0, 0.05, 0.3),
                "B0_chroma100_scov3e3_cthr90_g700k_tden_water1e2_bk": entry(29.3, 0.01, 0.2, 0.6)}
    bars = focus.factor_chart(variants, "sam_seabed_0_30_gt_0.5", "dno", 100, 1)
    points = focus.factor_chart(variants, "best_psnr", "psnr", 1, 2)
    assert bars.count('class="bar s') == 2 and 'class="dot' not in bars
    assert points.count('class="dot s') == 2 and 'class="bar' not in points
    assert "seedy: 29,10, 29,30" in points


def test_sam_masks_show_only_the_scene_of_the_sweep():
    import pytest
    from sam_water_mask import DATASETS
    dataset = Path(DATASETS) / "IUI3-RedSea-Undistorted"
    if not dataset.is_dir():
        pytest.skip("dataset not on this machine")
    html = focus.sam_masks(str(dataset))
    assert html.count('class="mask"') == 4 and "MTN_5894" in html and "MTN_1288" not in html
