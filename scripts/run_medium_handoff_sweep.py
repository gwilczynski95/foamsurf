#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training import ExperimentConfig
from training.medium_handoff_sweeps import (
    MEDIUM_HANDOFF_SWEEP_SPECS,
    build_medium_handoff_config,
    medium_handoff_spec_payload,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one medium-handoff disentanglement experiment."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--index", required=True, type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wandb-name", required=True)
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"),
        default="online",
    )
    parser.add_argument("--print-spec", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.index < len(MEDIUM_HANDOFF_SWEEP_SPECS):
        parser.error(
            f"--index must be in [0, {len(MEDIUM_HANDOFF_SWEEP_SPECS)-1}]"
        )

    spec = MEDIUM_HANDOFF_SWEEP_SPECS[args.index]
    name = f"{args.wandb_name}-{spec.label}"
    config = build_medium_handoff_config(
        ExperimentConfig.load(args.config), args.index, args.output_dir,
        name, args.wandb_mode,
    )
    payload = medium_handoff_spec_payload(args.index)
    payload["resolved_name"] = name
    print("FOAMSURF_MEDIUM_HANDOFF_SPEC=" + json.dumps(payload), flush=True)
    if args.print_spec:
        return 0

    # Importing Trainer loads the native CUDA topology extension. Keep it
    # lazy so --print-spec remains a genuine CPU-only validation path.
    from training.trainer import Trainer

    result = Trainer(config).fit()
    (Path(args.output_dir) / "sweep_spec.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    print("FOAMSURF_MEDIUM_HANDOFF_RESULT=" + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
