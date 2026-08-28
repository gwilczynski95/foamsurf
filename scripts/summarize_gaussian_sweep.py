#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml


FAMILIES = {"control", "graphdeco", "scale_shape", "flare_movement"}


def collect_rows(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for summary_path in sorted(root.glob("*/summary.json")):
        config_path = summary_path.parent / "config.yaml"
        if not config_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        config = yaml.safe_load(config_path.read_text()) or {}
        tags = list((config.get("wandb") or {}).get("tags") or [])
        family = next((tag for tag in tags if tag in FAMILIES), "unknown")
        label = tags[-1] if tags else summary_path.parent.name
        rows.append(
            {
                "rank": 0,
                "best_psnr": float(summary.get("best_psnr", float("-inf"))),
                "status": summary.get("status", "unknown"),
                "gaussian_count": int(summary.get("gaussian_count", -1)),
                "family": family,
                "label": label,
                "run_dir": str(summary_path.parent),
            }
        )
    rows.sort(key=lambda row: float(row["best_psnr"]), reverse=True)
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rank completed massive Gaussian sweep runs by best PSNR."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/stage1_gaussian_sweep"),
    )
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    rows = collect_rows(args.root)
    fields = (
        "rank", "best_psnr", "status", "gaussian_count",
        "family", "label", "run_dir",
    )
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    print("\t".join(fields))
    for row in rows[: args.top]:
        print("\t".join(str(row[field]) for field in fields))
    print(f"completed_runs={len(rows)} expected_runs=372")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
