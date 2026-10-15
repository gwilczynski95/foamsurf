#!/usr/bin/env python3
"""Freeze the open-water mask of the IUI3 test views used by scripts/medium_audit.py.

Usage: python scripts/open_water_mask.py [--reference <run>] [--output configs/iui3_open_water.npz]

Open water = test pixels where the reference run's Gaussian-only alpha (best
checkpoint evaluation) is below --threshold. The default reference is the
default configuration (B0_chroma100_bk, 530646), whose Gaussians leave the
water clean. Freezing the mask lets every audit measure the same region.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from water_gaussians import alpha_maps  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path,
                        default=ROOT / "outputs/iui3_medium_fix/530646/B0_chroma100_bk")
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--output", type=Path, default=ROOT / "configs/iui3_open_water.npz")
    args = parser.parse_args()
    masks = {name: alpha < args.threshold for name, alpha in alpha_maps(args.reference).items()}
    iteration = json.loads((args.reference / "audit/best.json").read_text())["iteration"]
    meta = dict(reference=str(args.reference.resolve().relative_to(ROOT)), iteration=iteration,
                threshold=args.threshold)
    np.savez_compressed(args.output, meta=json.dumps(meta), **masks)
    for name, mask in masks.items():
        print(name, mask.shape, f"open water {mask.mean():.3f}")
    print(args.output)


if __name__ == "__main__":
    main()
