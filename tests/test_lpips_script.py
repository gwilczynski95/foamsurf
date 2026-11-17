"""scripts/lpips_vgg.py: the frame used for LPIPS must be the evaluator's frame (PSNR check per view)."""
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("lpips_vgg", Path(__file__).resolve().parents[1] / "scripts/lpips_vgg.py")
lv = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lv)


def test_check_psnr_accepts_equal_frames_and_rejects_others():
    saved = {"r_000": 30.0, "r_001": 32.0}
    assert lv.check_psnr({"r_000": 30.004, "r_001": 31.999}, saved) == pytest.approx(0.004)
    with pytest.raises(ValueError, match="differs"):
        lv.check_psnr({"r_000": 30.5, "r_001": 32.0}, saved)
    with pytest.raises(ValueError, match="views differ"):
        lv.check_psnr({"r_000": 30.0}, saved)


def test_saved_psnr_reads_the_final_iteration(tmp_path):
    d = tmp_path / "joint/evaluation/iteration_020000"
    d.mkdir(parents=True)
    (d / "per_view.csv").write_text("iteration,view_index,view_name,hybrid_psnr\n20000,0,r_000,31.5\n")
    assert lv.saved_psnr(tmp_path, 20000) == {"r_000": 31.5}


def _load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_medium_transmittance_view_selection_and_png():
    import numpy as np
    mt = _load("medium_transmittance")
    assert mt.select_views(["r_000", "r_001", "r_002"], None) == ["r_000", "r_001", "r_002"]
    assert mt.select_views(["r_000", "r_001", "r_002"], ["r_002", "r_000"]) == ["r_000", "r_002"]
    with pytest.raises(ValueError, match="not test views"):
        mt.select_views(["r_000"], ["r_009"])
    png = np.asarray(mt.to_png(np.array([[0.0, 1.0, 1.5]])))
    assert png.tolist() == [[0, 255, 255]]
