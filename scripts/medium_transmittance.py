#!/usr/bin/env python3
"""Transmittance of the medium alone (no surfels) on test views, from the final checkpoint: the medium-only render of
the evaluator (renderer.render_voronoi, the run's near / far range, the validation transmittance threshold).
    transmittance        the renderer's scalar transmittance at the end of the ray (white = clear, black = opaque)
    direct_transmittance per-channel exp(-integral of sigma^A) when the medium is RGB (None otherwise)
Writes OUT/<view>_medium_transmittance.png (scalar, 8-bit), OUT/<view>_medium_transmittance_rgb.png (when RGB) and
OUT/<view>_medium_transmittance.npz (float arrays).
Usage: python scripts/medium_transmittance.py RUN_DIR OUT_DIR [--views r_000 r_001 ...]   (default: every test view)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def to_png(values: np.ndarray) -> Image.Image:
    """[0, 1] floats (H x W or H x W x 3) -> 8-bit image."""
    return Image.fromarray((np.clip(values, 0.0, 1.0) * 255 + 0.5).astype(np.uint8))


def select_views(names: list[str], wanted: list[str] | None) -> list[str]:
    if not wanted:
        return names
    missing = sorted(set(wanted) - set(names))
    if missing:
        raise ValueError(f"not test views: {missing}")
    return [n for n in names if n in wanted]


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--views", nargs="*")
    args = parser.parse_args()
    from evaluation.evaluator import _rays
    from training.config import ExperimentConfig
    from training.trainer import Trainer

    run = args.run.resolve()
    config = ExperimentConfig.load(run / "joint/config.yaml")
    config.output_dir = str(ROOT / "outputs/medium_transmittance/_trainer" / run.name)
    config.wandb.mode = "disabled"
    trainer = Trainer(config, resume=run / "joint/checkpoints/last.pt")
    renderer, model = trainer.renderer, trainer.model
    renderer.transmittance_threshold = config.renderer.validation_transmittance_threshold
    trainer.topology.synchronize_renderer(renderer, model.medium)
    medium = model.medium.render_scene()
    cameras = {c.name: c for c in trainer.dataset.test}
    args.out.mkdir(parents=True, exist_ok=True)
    for name in select_views(list(cameras), args.views):
        camera = cameras[name]
        H, W = camera.height, camera.width
        out = renderer.render_voronoi(_rays(camera), medium, background=tuple(config.background))
        scalar = out.transmittance.reshape(H, W).float().cpu().numpy()
        arrays = dict(transmittance=scalar)
        to_png(scalar).save(args.out / f"{name}_medium_transmittance.png")
        if out.direct_transmittance is not None:
            rgb = out.direct_transmittance.reshape(H, W, -1).float().cpu().numpy()
            arrays["direct_transmittance"] = rgb
            to_png(rgb).save(args.out / f"{name}_medium_transmittance_rgb.png")
        np.savez_compressed(args.out / f"{name}_medium_transmittance.npz", **arrays)
        print(f"{name}: transmittance mean {scalar.mean():.3f} min {scalar.min():.3f}", flush=True)


if __name__ == "__main__":
    main()
