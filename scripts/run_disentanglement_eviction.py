#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training import ExperimentConfig
from training.disentanglement_eviction_sweeps import (
    GAUSSIAN_EVICTION_SWEEP_SPECS,
    build_gaussian_eviction_config,
    gaussian_eviction_spec_payload,
)
from training.trainer import Trainer


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one experiment from the Gaussian-eviction sweep."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--index", required=True, type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wandb-name", required=True)
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    parser.add_argument(
        "--print-spec",
        action="store_true",
        help="validate and print the resolved spec without constructing Trainer",
    )
    args = parser.parse_args()
    if not 0 <= args.index < len(GAUSSIAN_EVICTION_SWEEP_SPECS):
        parser.error(
            f"--index must be in [0, "
            f"{len(GAUSSIAN_EVICTION_SWEEP_SPECS) - 1}]"
        )

    spec = GAUSSIAN_EVICTION_SWEEP_SPECS[args.index]
    name = f"{args.wandb_name}-{spec.label}"
    config = build_gaussian_eviction_config(
        ExperimentConfig.load(args.config),
        args.index,
        args.output_dir,
        name,
        args.wandb_mode,
    )
    payload = gaussian_eviction_spec_payload(args.index)
    payload["resolved_name"] = name
    print(
        "FOAMSURF_GAUSSIAN_EVICTION_SPEC=" + json.dumps(payload),
        flush=True,
    )
    if args.print_spec:
        return 0

    result = Trainer(config).fit()
    print(
        "FOAMSURF_GAUSSIAN_EVICTION_RESULT=" + json.dumps(result),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
