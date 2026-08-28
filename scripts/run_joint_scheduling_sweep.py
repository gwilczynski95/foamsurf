#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training import ExperimentConfig
from training.joint_scheduling_sweeps import (
    JOINT_SCHEDULING_SPECS,
    build_joint_scheduling_config,
)
from training.trainer import Trainer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--index", required=True, type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wandb-name", required=True)
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    args = parser.parse_args()
    if not 0 <= args.index < len(JOINT_SCHEDULING_SPECS):
        parser.error(
            f"--index must be in [0, {len(JOINT_SCHEDULING_SPECS) - 1}]"
        )

    spec = JOINT_SCHEDULING_SPECS[args.index]
    name = f"{args.wandb_name}-{spec.label}"
    config = build_joint_scheduling_config(
        ExperimentConfig.load(args.config),
        args.index,
        args.output_dir,
        name,
        args.wandb_mode,
    )
    print(
        "FOAMSURF_JOINT_SCHEDULING_SPEC="
        + json.dumps({"index": args.index, **spec.__dict__}),
        flush=True,
    )
    result = Trainer(config).fit()
    print("FOAMSURF_JOINT_SCHEDULING_RESULT=" + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
