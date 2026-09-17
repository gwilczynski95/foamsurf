#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training import ExperimentConfig
from training.disentanglement_sweeps import (
    DISENTANGLEMENT_SWEEP_SPECS,
    build_disentanglement_config,
    disentanglement_spec_payload,
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
    if not 0 <= args.index < len(DISENTANGLEMENT_SWEEP_SPECS):
        parser.error(
            f"--index must be in [0, "
            f"{len(DISENTANGLEMENT_SWEEP_SPECS) - 1}]"
        )

    spec = DISENTANGLEMENT_SWEEP_SPECS[args.index]
    name = f"{args.wandb_name}-{spec.label}"
    config = build_disentanglement_config(
        ExperimentConfig.load(args.config),
        args.index,
        args.output_dir,
        name,
        args.wandb_mode,
    )
    print(
        "FOAMSURF_DISENTANGLEMENT_SPEC="
        + json.dumps(disentanglement_spec_payload(args.index)),
        flush=True,
    )
    result = Trainer(config).fit()
    print(
        "FOAMSURF_DISENTANGLEMENT_RESULT=" + json.dumps(result),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
