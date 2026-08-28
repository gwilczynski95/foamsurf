#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training import ExperimentConfig
from training.standalone_sweeps import (
    GAUSSIAN_SWEEP_SPECS,
    VORONOI_SWEEP_SPECS,
    build_standalone_sweep_config,
)
from training.trainer import Trainer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("gaussian", "voronoi"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wandb-name", required=True)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    args = parser.parse_args()
    specs = GAUSSIAN_SWEEP_SPECS if args.phase == "gaussian" else VORONOI_SWEEP_SPECS
    if not 0 <= args.index < len(specs):
        parser.error(
            f"--index must be in [0, {len(specs) - 1}] for {args.phase}"
        )
    spec = specs[args.index]
    resolved_wandb_name = f"{args.wandb_name}-{spec.label}"
    config = build_standalone_sweep_config(
        ExperimentConfig.load(args.config), args.phase, args.index,
        args.output_dir, resolved_wandb_name, args.wandb_mode,
    )
    print("FOAMSURF_STANDALONE_SWEEP_SPEC=" + json.dumps({
        "phase": args.phase, "index": args.index, "family": spec.family,
        "label": spec.label, "question": spec.question,
    }), flush=True)
    result = Trainer(config).fit()
    print("FOAMSURF_STANDALONE_SWEEP_RESULT=" + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
