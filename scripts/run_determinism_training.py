#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

from training import ExperimentConfig
from training.determinism import runtime_provenance, write_json
from training.sweep2 import build_sweep2_config
from training.trainer import Trainer


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one replicated FoamSurf determinism audit training."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wandb-name", required=True)
    parser.add_argument("--wandb-project", default="foamsurf_determinism_audit")
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="online"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=12_000)
    parser.add_argument("--adaptation", choices=("on", "off"), default="on")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    base = ExperimentConfig.load(args.config)
    config = build_sweep2_config(
        base,
        0,
        output_dir,
        args.wandb_name,
        wandb_mode=args.wandb_mode,
        wandb_project=args.wandb_project,
        seed=args.seed,
    )
    config.experiment_name = f"determinism-training-{args.adaptation}"
    config.optimization.iterations = args.iterations
    config.gaussian_adaptation.enabled = args.adaptation == "on"
    config.wandb.tags = [
        "determinism-audit",
        "training-replicate",
        f"adaptation-{args.adaptation}",
        f"seed-{args.seed}",
    ]
    config.validate()

    trainer = Trainer(config)
    provenance = runtime_provenance(REPOSITORY)
    write_json(output_dir / "provenance.json", provenance)
    if trainer.logger.wandb_run is not None:
        trainer.logger.wandb_run.config.update(
            {"audit_provenance": provenance}, allow_val_change=True
        )
    summary = trainer.fit()
    summary["audit_adaptation"] = args.adaptation
    summary["audit_seed"] = args.seed
    write_json(output_dir / "audit_summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
