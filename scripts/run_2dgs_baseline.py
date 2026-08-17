#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

from training import ExperimentConfig
from training.trainer import Trainer
from training.twodgs_profile import build_twodgs_profile


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train FoamSurf with the official 2DGS B1 profile."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wandb-name", required=True)
    parser.add_argument("--wandb-project", default="foamsurf_2dgs_baseline")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=30_000)
    args = parser.parse_args()

    config = build_twodgs_profile(
        ExperimentConfig.load(args.config),
        args.output_dir,
        args.wandb_name,
        wandb_mode=args.wandb_mode,
        wandb_project=args.wandb_project,
        seed=args.seed,
        iterations=args.iterations,
    )
    Trainer(config).fit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
