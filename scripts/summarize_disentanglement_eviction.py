#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.disentanglement_eviction_sweeps import (
    GAUSSIAN_EVICTION_SWEEP_SPECS,
)


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"empty metrics file: {path}")
    return rows


def _at_iteration(
    rows: list[dict[str, str]], iteration: int
) -> dict[str, str]:
    eligible = [row for row in rows if int(row["iteration"]) <= iteration]
    if not eligible:
        raise ValueError(f"no metric row at or before iteration {iteration}")
    return max(eligible, key=lambda row: int(row["iteration"]))


def _task_index(path: Path) -> int:
    match = re.search(r"_(\d+)$", path.name)
    if match is None:
        raise ValueError(f"cannot infer array index from {path.name}")
    return int(match.group(1))


def _number(row: dict[str, str], name: str, kind=float):
    value = row.get(name)
    return "" if value in (None, "") else kind(float(value))


def collect(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for run in sorted(path for path in root.iterdir() if path.is_dir()):
        summary_path = run / "summary.json"
        train_path = run / "metrics" / "train.csv"
        eval_path = run / "metrics" / "eval.csv"
        if not (
            summary_path.is_file()
            and train_path.is_file()
            and eval_path.is_file()
        ):
            continue
        index = _task_index(run)
        if not 0 <= index < len(GAUSSIAN_EVICTION_SWEEP_SPECS):
            raise ValueError(f"invalid array index {index} in {run}")
        spec = GAUSSIAN_EVICTION_SWEEP_SPECS[index]
        summary = json.loads(summary_path.read_text())
        train_rows = _csv_rows(train_path)
        eval_rows = _csv_rows(eval_path)
        train15 = _at_iteration(train_rows, 15_000)
        train30 = _at_iteration(train_rows, 30_000)
        eval15 = _at_iteration(eval_rows, 15_000)
        eval30 = _at_iteration(eval_rows, 30_000)
        factors = dict(spec.factors)
        record: dict[str, Any] = {
            "index": index,
            "label": spec.label,
            "family": spec.family,
            "status": summary.get("status", "unknown"),
            "best_psnr": float(summary["best_psnr"]),
            "peak_allocated_mib": float(summary["peak_allocated_mib"]),
            "prune_target": factors.get("prune_target", ""),
            "prune_start": factors.get("prune_start", 3_000),
            "prune_ramp": factors.get("prune_ramp", 0),
            "prune_until": factors.get("prune_until", ""),
            "prune_interval": factors.get("prune_interval", 250),
            "bin_weight": factors.get("bin_weight", 0.0),
            "bin_start": factors.get("bin_start", ""),
            "bin_ramp": factors.get("bin_ramp", ""),
            "bin_policy": factors.get("bin_policy", "none"),
        }
        for suffix, train, evaluation in (
            ("15k", train15, eval15),
            ("30k", train30, eval30),
        ):
            record.update(
                {
                    f"hybrid_psnr_{suffix}": _number(
                        evaluation, "eval/hybrid_psnr"
                    ),
                    f"gaussian_layer_psnr_{suffix}": _number(
                        evaluation, "lego_lego/psnr"
                    ),
                    f"medium_layer_psnr_{suffix}": _number(
                        evaluation, "lego_clouds/psnr"
                    ),
                    f"gaussian_weight_cloud_{suffix}": _number(
                        evaluation,
                        "entanglement/gaussian_weight_on_cloud_only",
                    ),
                    f"medium_weight_lego_{suffix}": _number(
                        evaluation,
                        "entanglement/medium_weight_on_lego_only",
                    ),
                    f"gaussian_alpha_cloud_{suffix}": _number(
                        evaluation,
                        "entanglement/gaussian_alpha_on_cloud_only",
                    ),
                    f"medium_cloud_dominance_{suffix}": _number(
                        evaluation,
                        "entanglement/medium_dominance_on_cloud_only",
                    ),
                    f"gaussian_count_{suffix}": _number(
                        train, "model/gaussian_count", int
                    ),
                    f"opacity_lt_0.1_{suffix}": _number(
                        train, "gaussian/opacity_lt_0_1_fraction"
                    ),
                    f"opacity_gt_0.9_{suffix}": _number(
                        train, "gaussian/opacity_gt_0_9_fraction"
                    ),
                    f"total_pruned_{suffix}": _number(
                        train, "adaptation/total_pruned", int
                    ),
                    f"bin_weight_active_{suffix}": _number(
                        train,
                        "disentanglement/opacity_binarization_weight",
                    ),
                }
            )
        record["post15_pruned"] = (
            int(record["total_pruned_30k"])
            - int(record["total_pruned_15k"])
        )
        record["post15_gaussian_count_delta"] = (
            int(record["gaussian_count_30k"])
            - int(record["gaussian_count_15k"])
        )
        record["post15_medium_layer_psnr_delta"] = (
            float(record["medium_layer_psnr_30k"])
            - float(record["medium_layer_psnr_15k"])
        )
        record["post15_hybrid_psnr_delta"] = (
            float(record["hybrid_psnr_30k"])
            - float(record["hybrid_psnr_15k"])
        )
        records.append(record)
    return sorted(records, key=lambda row: int(row["index"]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="outputs/disentanglement_eviction", type=Path
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.root / "eviction_summary.csv"
    rows = collect(args.root)
    if not rows:
        raise SystemExit(f"no completed runs found under {args.root}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"Wrote {len(rows)}/{len(GAUSSIAN_EVICTION_SWEEP_SPECS)} runs "
        f"to {output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
