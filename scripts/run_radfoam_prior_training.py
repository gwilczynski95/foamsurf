#!/usr/bin/env python3
"""Run one FoamSurf training initialized by a separated RadFoam prior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training import ExperimentConfig
from training.trainer import Trainer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--wandb-name")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--eval-interval", type=int)
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled")
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="initialize, evaluate iteration zero and stop before training",
    )
    args = parser.parse_args()

    config = ExperimentConfig.load(args.config)
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.wandb_name:
        config.experiment_name = args.wandb_name
        config.wandb.name = args.wandb_name
    if args.iterations is not None:
        config.optimization.iterations = args.iterations
        config.optimization.medium_lr_decay_from = min(
            config.optimization.medium_lr_decay_from, max(0, args.iterations // 2)
        )
        config.optimization.medium_lr_decay_until = args.iterations
        adaptation = config.gaussian_adaptation
        adaptation.densify_from = min(adaptation.densify_from, args.iterations)
        adaptation.densify_until = max(
            adaptation.densify_from,
            min(adaptation.densify_until, args.iterations),
        )
        adaptation.opacity_prune_from = min(
            adaptation.opacity_prune_from, args.iterations
        )
        if adaptation.opacity_prune_until is not None:
            adaptation.opacity_prune_until = args.iterations
    if args.eval_interval is not None:
        config.validation.eval_interval = args.eval_interval
    if args.wandb_mode:
        config.wandb.mode = args.wandb_mode
    config.validate()
    if not config.radfoam_prior.enabled:
        raise ValueError("config.radfoam_prior.enabled must be true")

    trainer = Trainer(config)
    report = getattr(trainer.model, "initialization_report", None)
    if report is None:
        raise RuntimeError(
            "Trainer did not produce a RadFoam initialization report"
        )
    report_path = trainer.output_dir / "radfoam_initialization_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("FOAMSURF_RADFOAM_INITIALIZATION=" + json.dumps(report), flush=True)

    summary, images, histograms = trainer.evaluator.evaluate(0)
    trainer.logger.log_eval(summary, images, histograms)
    trainer._checkpoint(
        trainer.output_dir / "checkpoints" / "initialization.pt", 0
    )
    print("FOAMSURF_RADFOAM_ITERATION_ZERO=" + json.dumps(summary), flush=True)
    if args.prepare_only:
        return 0

    result = trainer.fit()
    print("FOAMSURF_RADFOAM_RESULT=" + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
