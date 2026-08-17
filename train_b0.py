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
        "--seed",
        type=int,
        help="Override the experiment random seed",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        help="Override validation.eval_interval",
    )
    parser.add_argument(
        "--experiment-name",
        help="Override experiment_name",
    )
    parser.add_argument(
        "--wandb-name",
        help="Override wandb.name",
    )
    parser.add_argument(
        "--min-opacity",
        type=float,
        help="Override gaussian_adaptation.min_opacity",
    )
    parser.add_argument(
        "--opacity-prune-from",
        type=int,
        help="Override gaussian_adaptation.opacity_prune_from",
    )
    parser.add_argument(
        "--rays-per-batch",
        type=int,
        help="Override optimization.rays_per_batch",
    )
    parser.add_argument(
        "--full-image-batch",
        action="store_true",
        help="Use every pixel of one camera in each optimizer step",
    )
    parser.add_argument(
        "--cameras-without-replacement",
        action="store_true",
        help="Shuffle cameras and exhaust the stack before reshuffling",
    )
    parser.add_argument(
        "--no-max-points",
        action="store_true",
        help="Disable the Gaussian densification population budget",
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
    if args.rays_per_batch is not None:
        config.optimization.rays_per_batch = args.rays_per_batch
    if args.full_image_batch:
        config.optimization.full_image_batch = True
    if args.cameras_without_replacement:
        config.optimization.cameras_without_replacement = True
    if args.no_max_points:
        config.gaussian_adaptation.max_points = None
    if args.source_path:
        config.dataset.source_path = args.source_path
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.iterations is not None:
        config.optimization.iterations = args.iterations
    if args.seed is not None:
        config.seed = args.seed
    if args.eval_interval is not None:
        config.validation.eval_interval = args.eval_interval
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    if args.wandb_name:
        config.wandb.name = args.wandb_name
    if args.min_opacity is not None:
        config.gaussian_adaptation.min_opacity = args.min_opacity
    if args.opacity_prune_from is not None:
        config.gaussian_adaptation.opacity_prune_from = (
            args.opacity_prune_from
        )
    if args.wandb_mode:
        config.wandb.mode = args.wandb_mode
    config.validate()

    summary = Trainer(config, resume=args.resume).fit()
    print("FOAMSURF_B0_RESULT=" + json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
