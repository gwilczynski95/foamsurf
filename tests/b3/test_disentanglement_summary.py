from __future__ import annotations

import csv
import json

from scripts.summarize_disentanglement_phase1 import collect


def _write_csv(path, row) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def test_summary_maps_array_index_and_preserves_selection_metrics(
    tmp_path,
) -> None:
    run = tmp_path / "dis_phase1_12345_0"
    run.mkdir()
    (run / "summary.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "best_psnr": 32.5,
                "gaussian_count": 100,
                "voronoi_count": 20,
            }
        )
    )
    _write_csv(
        run / "metrics" / "eval.csv",
        {
            "eval/hybrid_psnr": 32.0,
            "eval/gaussian_only_psnr": 30.0,
            "eval/voronoi_only_psnr": 12.0,
            "eval/hybrid_minus_gaussian_psnr": 2.0,
            "entanglement/gaussian_weight_on_cloud_only": 0.1,
            "entanglement/medium_weight_on_lego_only": 0.01,
        },
    )
    _write_csv(
        run / "metrics" / "train.csv",
        {
            "gaussian/opacity_gt_0_5_fraction": 0.4,
            "gaussian/opacity_gt_0_9_fraction": 0.2,
            "medium/density_p95": 0.3,
            "medium/density_p99": 0.6,
            "disentanglement/graph_tv_raw": 0.02,
            "disentanglement/optical_thickness_raw": 0.5,
            "disentanglement/density_cap_value": "nan",
        },
    )

    rows = collect(tmp_path)
    assert len(rows) == 1
    assert rows[0]["index"] == 0
    assert rows[0]["label"] == "control"
    assert rows[0]["hybrid_psnr"] == 32.0
    assert rows[0]["gaussian_weight_on_cloud_only"] == 0.1
