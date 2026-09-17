#!/usr/bin/env python3
"""Load a RadFoam checkpoint into FoamSurf and render class diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from foamsurf import NativeHybridRenderer
from foamsurf_data import BlenderDataset
from training.radfoam_import import RadFoamCheckpointInitializer


DEFAULT_RUN = Path(
    "/shared/results/gmwilczn/foamsurf/radfoam/output/lego_both_prior_487402"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_RUN / "checkpoints/iter_00004000.pt",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=(
            DEFAULT_RUN
            / "separation/iter_00004000/checkpoint_only/local_contrast_otsu/cell_labels.pt"
        ),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("/shared/sets/datasets/foamsurf/lego/lego_both"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--views", type=int, nargs="+", default=[0])
    parser.add_argument("--resolution", type=int, default=1)
    parser.add_argument("--fixed-density", type=float, default=0.15)
    parser.add_argument("--near", type=float, default=0.1)
    parser.add_argument("--far", type=float, default=1000.0)
    parser.add_argument("--transmittance-threshold", type=float, default=0.01)
    parser.add_argument("--max-intersections", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("the native debug renderer requires a CUDA GPU")
    device = torch.device("cuda")
    initializer = RadFoamCheckpointInitializer(
        args.checkpoint, args.labels, device=device
    )
    imported = initializer.load()
    dataset = BlenderDataset(
        args.dataset,
        resolution=args.resolution,
        device=device,
        splits=("test",),
    )
    pixel_count = dataset.width * dataset.height
    renderer = NativeHybridRenderer(
        max_batch_size=pixel_count,
        near=args.near,
        far=args.far,
        transmittance_threshold=args.transmittance_threshold,
        max_intersections=args.max_intersections,
    )
    for view_index in args.views:
        if not 0 <= view_index < len(dataset.test):
            raise ValueError(
                f"view {view_index} is outside [0, {len(dataset.test) - 1}]"
            )
        path = initializer.render_debug(
            imported,
            renderer,
            dataset.test[view_index],
            args.output / f"view_{view_index:03d}",
            fixed_density=args.fixed_density,
        )
        print(f"wrote {path}")
    print(imported.metadata)


if __name__ == "__main__":
    main()
