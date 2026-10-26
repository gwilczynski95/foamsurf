#!/usr/bin/env python3
"""How strongly does the medium attenuate the seabed, compared with WaterSplatting?

Per test view, the image is split into a GRID x GRID grid. In cells covered by Gaussians
(alpha > MIN_ALPHA) and holding COLMAP points, the medium transmittance in front of the seabed is
  ours:            T = (hybrid - backscatter) / gaussians-only    (hybrid = T * Gaussians + backscatter)
  WaterSplatting:  T = rgb_object / rgb_clear
and sigma_eff = -ln(T) / d, d = distance to the seabed (20% quantile of the COLMAP point depths in
the cell). Reported: the median sigma_eff over the cells and the Spearman correlation of T with d
(physically negative: farther seabed, more attenuation).

Usage:
  python scripts/medium_attenuation.py --runs outputs/seathru/<job>/<scene><suffix> ... \
      [--iterations 500 2000 5000 10000] [--ws <WaterSplatting run dir>] [--datasets <SeathruNeRF_dataset>]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GRID = 12
MIN_ALPHA = 0.95
MIN_POINTS = 5
# Measured on our WaterSplatting run 531560 (seed 42), with this script: reference when the WS renders
# are not on this machine.
WS_REFERENCE = {"IUI3-RedSea-Undistorted": 0.007, "JapaneseGradens-RedSea-Undistorted": 0.035,
                "Panama-Undisorted": 0.046, "Curasao-Undistorted": 0.068}


def project(points, camera):
    """Normalised image coordinates (u, r in [0, 1)), depth and the in-frame mask of points."""
    v = points - camera.origin
    z = v @ camera.forward
    fx, fy, cx, cy = camera.intrinsics
    u = fx * (v @ camera.right) / z.clamp_min(1e-6) + cx
    r = fy * (v @ camera.down) / z.clamp_min(1e-6) + cy
    inside = (z > 0) & (u >= 0) & (u < camera.width) & (r >= 0) & (r < camera.height)
    return (u / camera.width).numpy(), (r / camera.height).numpy(), z.numpy(), inside.numpy()


def cell_depths(u, r, z, inside, grid=GRID, min_points=MIN_POINTS):
    """Seabed distance per grid cell (20% quantile of the point depths), NaN without enough points."""
    cell = (np.floor(r * grid).clip(0, grid - 1) * grid + np.floor(u * grid).clip(0, grid - 1)).astype(int)
    depth = np.full(grid * grid, np.nan)
    for c in np.unique(cell[inside]):
        values = z[inside & (cell == c)]
        if len(values) >= min_points:
            depth[c] = np.quantile(values, 0.2)
    return depth


def cell_transmittance(attenuated, clear, alpha, depth, grid=GRID, min_alpha=MIN_ALPHA):
    """T per usable cell: mean(attenuated) / mean(clear); returns (T, depth) of the cells kept."""
    height, width = alpha.shape
    ts, ds = [], []
    for c in np.flatnonzero(np.isfinite(depth)):
        i, j = divmod(c, grid)
        sl = (slice(i * height // grid, (i + 1) * height // grid), slice(j * width // grid, (j + 1) * width // grid))
        if alpha[sl].mean() < min_alpha:
            continue
        ts.append(attenuated[sl].mean() / max(clear[sl].mean(), 1e-3))
        ds.append(depth[c])
    return np.array(ts), np.array(ds)


def summarise(pairs):
    """pairs: [(T, d)] per view -> median sigma_eff over all cells, mean per-view Spearman(T, d)."""
    sigmas, correlations = [], []
    for ts, ds in pairs:
        if len(ts) < 8:
            continue
        sigmas.append(-np.log(np.clip(ts, 1e-3, 1.0)) / ds)
        correlations.append(spearmanr(ds, ts).correlation)
    if not sigmas:
        return dict(sigma=float("nan"), correlation=float("nan"), cells=0)
    return dict(sigma=float(np.median(np.concatenate(sigmas))), correlation=float(np.mean(correlations)),
                cells=int(sum(len(s) for s in sigmas)))


def load(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


def scene_of(run):
    return next(s for s in WS_REFERENCE if run.name.startswith(s))


def test_views(dataset_dir):
    from foamsurf_data.colmap import ColmapDataset
    dataset = ColmapDataset(str(dataset_dir), resolution=8, device="cpu", llffhold=8)
    points = torch.as_tensor(dataset.initial_point_cloud()[0], dtype=torch.float32)
    return [(camera.name, cell_depths(*project(points, camera))) for camera in dataset.test]


def ours(run, iteration, views):
    folder = run / "joint/evaluation" / f"iteration_{iteration:06d}"
    pairs = []
    for name, depth in views:
        hybrid, gaussians, backscatter = (load(next(folder.glob(f"*_{name}_{layer}.png")))
                                          for layer in ("hybrid", "gaussian_only", "backscatter"))
        alpha = np.load(next(folder.glob(f"*_{name}_gaussian_alpha.npz")))["alpha"][..., 0]
        pairs.append(cell_transmittance(hybrid - backscatter, gaussians, alpha, depth))
    return summarise(pairs)


def watersplatting(ws_scene, views):
    pairs = []
    for name, depth in views:
        renders = ws_scene / "renders"
        alpha = np.asarray(Image.open(renders / "accumulation" / f"{name}.png"), dtype=np.float64) / 255.0
        alpha = alpha[..., 0] if alpha.ndim == 3 else alpha
        pairs.append(cell_transmittance(load(renders / "rgb_object" / f"{name}.png"),
                                        load(renders / "rgb_clear" / f"{name}.png"), alpha, depth))
    return summarise(pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--iterations", type=int, nargs="+", default=[500, 2000, 5000, 10000])
    parser.add_argument("--datasets", type=Path, default=Path("/shared/sets/datasets/SeathruNeRF_dataset"))
    parser.add_argument("--ws", type=Path, help="WaterSplatting run dir (<scene>/renders/...); else the reference")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    views_cache, ws_cache, report = {}, {}, {}
    for run in args.runs:
        scene = scene_of(run)
        if scene not in views_cache:
            views_cache[scene] = test_views(args.datasets / scene)
            ws_cache[scene] = (watersplatting(args.ws / scene, views_cache[scene])["sigma"]
                               if args.ws and (args.ws / scene / "renders/rgb_object").is_dir() else WS_REFERENCE[scene])
        rows = {}
        for iteration in args.iterations:
            if (run / "joint/evaluation" / f"iteration_{iteration:06d}").is_dir():
                rows[iteration] = ours(run, iteration, views_cache[scene])
        report[str(run)] = dict(scene=scene, ws_sigma=ws_cache[scene], iterations=rows)
        line = "  ".join(f"{it}: σ {r['sigma']:.3f} (×{r['sigma'] / ws_cache[scene]:.2f} WS), corr {r['correlation']:+.2f}"
                         for it, r in rows.items())
        print(f"{run.name:55s} WS σ {ws_cache[scene]:.3f} | {line}", flush=True)
    if args.output:
        args.output.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
