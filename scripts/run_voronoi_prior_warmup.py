#!/usr/bin/env python3
"""Train an adaptive native FoamSurf Voronoi prior on the full scene."""

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
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled")
    )
    args = parser.parse_args()

    config = ExperimentConfig.load(args.config)
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.wandb_name:
        config.experiment_name = args.wandb_name
        config.wandb.name = args.wandb_name
    if args.wandb_mode:
        config.wandb.mode = args.wandb_mode
    config.validate()
    if not config.voronoi_warmup.enabled:
        raise ValueError("config.voronoi_warmup.enabled must be true")

    trainer = Trainer(config)
    report = {
        "stage": "native_foamsurf_voronoi_warmup",
        "training_target": config.dataset.train_target,
        "training_source": config.dataset.source_path,
        "representation_mode": config.representation_mode,
        "initial_voronoi_count": trainer.model.medium.point_count,
        "population_is_adaptive": config.voronoi_adaptation.enabled,
        "minimum_voronoi_count": config.voronoi_adaptation.min_points,
        "maximum_voronoi_count": config.voronoi_adaptation.max_points,
        "adaptation_from": config.voronoi_adaptation.densify_from,
        "adaptation_until_exclusive": (
            config.voronoi_adaptation.densify_until
        ),
        "adaptation_interval": config.voronoi_adaptation.interval,
        "milestone_iterations": (
            config.voronoi_warmup.milestone_iterations
        ),
        "bbox_min": trainer.model.medium.bbox_min.detach().cpu().tolist(),
        "bbox_max": trainer.model.medium.bbox_max.detach().cpu().tolist(),
        "gaussians_rendered_or_optimized": False,
    }
    report_path = trainer.output_dir / "voronoi_warmup_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print("FOAMSURF_VORONOI_WARMUP=" + json.dumps(report), flush=True)

    summary, images, histograms = trainer.evaluator.evaluate(0)
    trainer.logger.log_eval(summary, images, histograms)
    checkpoint = trainer.save_voronoi_warmup_milestone(0)
    if checkpoint is None:
        raise RuntimeError("iteration-zero warmup milestone was not saved")
    print(
        "FOAMSURF_VORONOI_WARMUP_ITERATION_ZERO=" + json.dumps(summary),
        flush=True,
    )

    result = trainer.fit()
    print("FOAMSURF_VORONOI_WARMUP_RESULT=" + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
