#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training import ExperimentConfig
from training.k_sweeps import (
    K_SWEEP_SIZE,
    build_k_sweep_config,
    k_sweep_case,
)
from training.trainer import Trainer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wandb-name", required=True)
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    args = parser.parse_args()
    if not 0 <= args.index < K_SWEEP_SIZE:
        parser.error(f"--index must be in [0, {K_SWEEP_SIZE - 1}]")
    k, recipe = k_sweep_case(args.index)
    resolved_name = f"{args.wandb_name}-k{k}-{recipe.label}"
    config = build_k_sweep_config(
        ExperimentConfig.load(args.config),
        args.index,
        args.output_dir,
        resolved_name,
        args.wandb_mode,
    )
    print(
        "FOAMSURF_K_SWEEP_SPEC="
        + json.dumps(
            {
                "index": args.index,
                "k": k,
                "family": recipe.family,
                "label": recipe.label,
            }
        ),
        flush=True,
    )
    result = Trainer(config).fit()
    print("FOAMSURF_K_SWEEP_RESULT=" + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
