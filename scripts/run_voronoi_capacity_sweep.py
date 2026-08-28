#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training import ExperimentConfig
from training.trainer import Trainer
from training.voronoi_capacity_sweeps import (
    VORONOI_CAPACITY_SWEEP_SPECS,
    build_voronoi_capacity_config,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wandb-name", required=True)
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    args = parser.parse_args()
    if not 0 <= args.index < len(VORONOI_CAPACITY_SWEEP_SPECS):
        parser.error(
            f"--index must be in [0, "
            f"{len(VORONOI_CAPACITY_SWEEP_SPECS) - 1}]"
        )

    spec = VORONOI_CAPACITY_SWEEP_SPECS[args.index]
    resolved_name = f"{args.wandb_name}-{spec.label}"
    config = build_voronoi_capacity_config(
        ExperimentConfig.load(args.config), args.index, args.output_dir,
        resolved_name, args.wandb_mode,
    )
    print(
        "FOAMSURF_VORONOI_CAPACITY_SPEC="
        + json.dumps({
            "index": args.index,
            "label": spec.label,
            "seed": spec.seed,
            "max_points": spec.max_points,
            "child_jitter_fraction": spec.child_jitter_fraction,
            "iterations": spec.iterations,
            "densify_until": spec.densify_until,
            "lr_decay_from": spec.lr_decay_from,
        }),
        flush=True,
    )
    result = Trainer(config).fit()
    print("FOAMSURF_VORONOI_CAPACITY_RESULT=" + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
