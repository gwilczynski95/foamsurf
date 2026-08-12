#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from training import ExperimentConfig
from training.trainer import Trainer


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train the unregularized FoamSurf B0 baseline."
    )
    parser.add_argument(
        "--config", required=True, help="Path to a B0 YAML config"
    )
    parser.add_argument(
        "--source-path",
        help="Override dataset.source_path from the config",
    )
    parser.add_argument(
        "--output-dir",
        help="Override output_dir from the config",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        help="Override optimization.iterations",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        help="Override wandb.mode",
    )
    parser.add_argument(
        "--resume",
        help="Resume model and both optimizer states from a checkpoint",
    )
    args = parser.parse_args()

    config = ExperimentConfig.load(args.config)
    if args.source_path:
        config.dataset.source_path = args.source_path
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.iterations is not None:
        config.optimization.iterations = args.iterations
    if args.wandb_mode:
        config.wandb.mode = args.wandb_mode
    config.validate()

    summary = Trainer(config, resume=args.resume).fit()
    print("FOAMSURF_B0_RESULT=" + json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
