"""Helpers of scripts/seathru_report.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import seathru_report as report  # noqa: E402


def test_split_name():
    assert report.split_name("Panama-Undisorted") == ("Panama-Undisorted", "", 42)
    assert report.split_name("IUI3-RedSea-Undistorted_view75_c20k_cs10") == (
        "IUI3-RedSea-Undistorted", "_view75_c20k_cs10", 42)
    assert report.split_name("Curasao-Undistorted_seed43") == ("Curasao-Undistorted", "", 43)


def test_fps_median_uses_the_last_evaluations(tmp_path):
    metrics = tmp_path / "joint/metrics"
    metrics.mkdir(parents=True)
    rows = ["iteration,eval/hybrid_fps"] + [f"{i},{fps}" for i, fps in enumerate([10] * 5 + [100, 90, 110] + [95] * 7)]
    (metrics / "eval.csv").write_text("\n".join(rows))
    assert report.fps_median(tmp_path) == 95.0  # the first five slow evaluations are ignored


def test_chart_bars_points_and_legend():
    categories = [("a", "A"), ("b", "B")]
    series = [("x", "seria x"), ("y", "seria y")]
    values = {("a", "x"): [1.0, 3.0], ("a", "y"): [2.0], ("b", "x"): [4.0]}
    bars = report.chart("t", categories, series, values, 1)
    assert bars.count('class="bar s') == 3 and "seria y" in bars and bars.count('class="seed"') == 2
    points = report.chart("t", categories, series, values, 1, as_points=True)
    assert points.count('class="dot s') == 3 and "seedy: 1,0, 3,0" in points
    assert report.chart("t", categories, series, {}, 1) == ""
