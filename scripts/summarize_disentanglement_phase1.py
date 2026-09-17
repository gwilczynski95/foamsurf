#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.disentanglement_sweeps import (
    DISENTANGLEMENT_SWEEP_SPECS,
)


def _last_csv_row(path: Path) -> dict[str, str]:
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"empty metrics file: {path}")
    return rows[-1]


def _task_index(path: Path) -> int:
    match = re.search(r"_(\d+)$", path.name)
    if match is None:
        raise ValueError(f"cannot infer array index from {path.name}")
    return int(match.group(1))


def collect(root: Path) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []
    for run in sorted(path for path in root.iterdir() if path.is_dir()):
        summary_path = run / "summary.json"
        train_path = run / "metrics" / "train.csv"
        eval_path = run / "metrics" / "eval.csv"
        if not (summary_path.is_file() and train_path.is_file() and eval_path.is_file()):
            continue
        index = _task_index(run)
        if not 0 <= index < len(DISENTANGLEMENT_SWEEP_SPECS):
            raise ValueError(f"invalid array index {index} in {run}")
        spec = DISENTANGLEMENT_SWEEP_SPECS[index]
        summary = json.loads(summary_path.read_text())
        train = _last_csv_row(train_path)
        evaluation = _last_csv_row(eval_path)
        rows.append(
            {
                "index": index,
                "label": spec.label,
                "family": spec.family,
                "status": summary["status"],
                "best_psnr": float(summary["best_psnr"]),
                "hybrid_psnr": float(evaluation["eval/hybrid_psnr"]),
                "gaussian_only_psnr": float(
                    evaluation["eval/gaussian_only_psnr"]
                ),
                "voronoi_only_psnr": float(
                    evaluation["eval/voronoi_only_psnr"]
                ),
                "hybrid_minus_gaussian_psnr": float(
                    evaluation["eval/hybrid_minus_gaussian_psnr"]
                ),
                "gaussian_weight_on_cloud_only": float(
                    evaluation["entanglement/gaussian_weight_on_cloud_only"]
                ),
                "medium_weight_on_lego_only": float(
                    evaluation["entanglement/medium_weight_on_lego_only"]
                ),
                "opacity_gt_0_5_fraction": float(
                    train["gaussian/opacity_gt_0_5_fraction"]
                ),
                "opacity_gt_0_9_fraction": float(
                    train["gaussian/opacity_gt_0_9_fraction"]
                ),
                "medium_density_p95": float(train["medium/density_p95"]),
                "medium_density_p99": float(train["medium/density_p99"]),
                "graph_tv_raw": float(
                    train["disentanglement/graph_tv_raw"]
                ),
                "optical_thickness_raw": float(
                    train["disentanglement/optical_thickness_raw"]
                ),
                "density_cap_value": float(
                    train["disentanglement/density_cap_value"]
                ),
                "gaussian_count": int(summary["gaussian_count"]),
                "voronoi_count": int(summary["voronoi_count"]),
            }
        )
    return sorted(rows, key=lambda row: int(row["index"]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="outputs/disentanglement_phase1", type=Path
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.root / "phase1_summary.csv"
    rows = collect(args.root)
    if not rows:
        raise SystemExit(f"no completed Phase 1 runs found under {args.root}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"Wrote {len(rows)}/{len(DISENTANGLEMENT_SWEEP_SPECS)} runs to "
        f"{output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
