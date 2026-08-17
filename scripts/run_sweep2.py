#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

from training import ExperimentConfig
from training.sweep2 import SWEEP2_SPECS, build_sweep2_config
from training.trainer import Trainer


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one FoamSurf sweep2 task.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--index", required=True, type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wandb-name", required=True)
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    parser.add_argument("--wandb-project", default="foamsurf_sweep2")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base = ExperimentConfig.load(args.config)
    config = build_sweep2_config(
        base,
        args.index,
        args.output_dir,
        args.wandb_name,
        wandb_mode=args.wandb_mode,
        wandb_project=args.wandb_project,
        seed=args.seed,
    )
    spec = SWEEP2_SPECS[args.index]
    print(
        "FOAMSURF_SWEEP2_SPEC="
        + json.dumps(
            {
                "index": args.index,
                "label": spec.label,
                "question": spec.question,
            }
        ),
        flush=True,
    )
    summary = Trainer(config).fit()
    print("FOAMSURF_SWEEP2_RESULT=" + json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
