#!/usr/bin/env python3
"""Find Gaussians that cover open water and describe their clusters (CPU only).

Usage: python scripts/water_gaussians.py --run <dense run> --reference <run with clean water> --output <dir>

"Water pixels" are test pixels where the reference run's Gaussian-only alpha is
below 0.01 (Gaussian-free open water). Pixels where the dense run's alpha
exceeds 0.5 there are "invaded". Gaussians of the dense run whose screen
footprint (radius 3 x largest scale, projected) overlaps invaded pixels of any
test view are the water Gaussians (an isotropic upper bound on the planar footprint).
Clusters are connected components of the graph linking centers closer than
--cluster-radius (default: 3x the median nearest-neighbour spacing).
Writes stats.json, projections.png and an interactive 3D plot (water_gaussians_3d.html).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from foamsurf_data.colmap import ColmapDataset  # noqa: E402


def load_gaussians(run):
    model = torch.load(run / "joint/checkpoints/best.pt", map_location="cpu", weights_only=False)["model"]
    return dict(means=model["gaussians.means"].numpy().astype(np.float64),
                opacity=torch.sigmoid(model["gaussians.opacity_logits"]).flatten().numpy(),
                scale=torch.exp(model["gaussians.log_scales"]).amax(-1).numpy())


def alpha_maps(run):
    iteration = json.loads((run / "audit/best.json").read_text())["iteration"]
    folder = run / "joint/evaluation" / f"iteration_{iteration:06d}"
    return {path.name.split("_gaussian_alpha")[0].split("_", 1)[1]: np.load(path)["alpha"][..., 0]
            for path in sorted(folder.glob("*_gaussian_alpha.npz"))}


def project(points, camera):
    """Pixel (x, y) and depth of points, same convention as foamsurf_data/rays.py."""
    v = points - camera.origin.numpy()
    z = v @ camera.forward.numpy()
    fx, fy, cx, cy = camera.intrinsics
    x = fx * (v @ camera.right.numpy()) / np.maximum(z, 1e-9) + cx - 0.5
    y = fy * (v @ camera.down.numpy()) / np.maximum(z, 1e-9) + cy - 0.5
    return np.rint(x).astype(np.int64), np.rint(y).astype(np.int64), z


def clusters(points, radius):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    pairs = cKDTree(points).query_pairs(radius, output_type="ndarray")
    n = len(points)
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    return labels, sizes[labels]


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {q: float(np.quantile(values, p)) for q, p in (("p10", .1), ("p50", .5), ("p90", .9), ("p99", .99))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cluster-radius", type=float)
    parser.add_argument("--min-opacity", type=float, default=0.3)
    parser.add_argument("--plot-background", type=int, default=150000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    from scipy.spatial import cKDTree

    config = json.loads((args.run / "pipeline_config.json").read_text())["joint"]["dataset"]
    dataset = ColmapDataset(config["source_path"], resolution=config["resolution"], device="cpu",
                            llffhold=config["llffhold"])
    colmap_points = np.asarray(dataset.initial_point_cloud()[0], dtype=np.float64)
    g = load_gaussians(args.run)
    dense, reference = alpha_maps(args.run), alpha_maps(args.reference)
    visible = g["opacity"] > args.min_opacity

    water = np.zeros(len(g["means"]), dtype=bool)
    per_view = {}
    for camera in dataset.test:
        from scipy.ndimage import distance_transform_edt
        x, y, z = project(g["means"], camera)
        open_water = reference[camera.name] < 0.01
        invaded = open_water & (dense[camera.name] > 0.5)
        # Distance (pixels) from every pixel to the nearest invaded pixel.
        to_invaded = distance_transform_edt(~invaded)
        radius_px = 3 * g["scale"] * camera.intrinsics[0] / np.maximum(z, 1e-9)
        cx = np.clip(x, 0, camera.width - 1); cy = np.clip(y, 0, camera.height - 1)
        outside = np.hypot(x - cx, y - cy)  # distance of off-screen centers to the image border
        hit = (z > 0) & (to_invaded[cy, cx] + outside <= radius_px + 0.5)
        if not hasattr(main, "_invaded_images"):
            main._invaded_images = {}
        main._invaded_images[camera.name] = (reference[camera.name], dense[camera.name], invaded)
        water |= hit & visible
        per_view[camera.name] = dict(open_water_pixels=float(open_water.mean()),
                                     invaded_fraction_of_open_water=float(invaded.sum() / max(open_water.sum(), 1)),
                                     gaussians_in_invaded_pixels=int((hit & visible).sum()))

    spacing = cKDTree(g["means"]).query(g["means"], k=2)[0][:, 1]
    radius = args.cluster_radius or 3 * float(np.median(spacing))
    _, cluster_size = clusters(g["means"], radius)
    colmap_distance = cKDTree(colmap_points).query(g["means"])[0]
    cameras = np.stack([c.origin.numpy() for c in dataset.train])
    camera_distance = cKDTree(cameras).query(g["means"])[0]
    rest = visible & ~water
    groups = {"water": water, "rest_visible": rest}
    stats = dict(run=str(args.run), reference=str(args.reference), gaussians=int(len(water)),
                 visible=int(visible.sum()), water_gaussians=int(water.sum()), cluster_radius=radius,
                 median_spacing=float(np.median(spacing)), per_view=per_view)
    for name, mask in groups.items():
        stats[name] = dict(count=int(mask.sum()),
                           cluster_size=summary(cluster_size[mask]),
                           in_cluster_le_10=float((cluster_size[mask] <= 10).mean()),
                           in_cluster_le_100=float((cluster_size[mask] <= 100).mean()),
                           colmap_distance=summary(colmap_distance[mask]),
                           camera_distance=summary(camera_distance[mask]),
                           opacity=summary(g["opacity"][mask]), scale=summary(g["scale"][mask]))
    (args.output / "stats.json").write_text(json.dumps(stats, indent=2))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(main._invaded_images)
    fig, axes = plt.subplots(len(names), 3, figsize=(15, 3.4 * len(names)))
    for row, name in zip(axes, names):
        ref_alpha, dense_alpha, invaded = main._invaded_images[name]
        for ax, image, title in zip(row, (ref_alpha, dense_alpha, invaded),
                                    ("alpha Gaussów, baza ok. 290 tys.", "alpha Gaussów, ok. 800 tys.",
                                     "woda zajęta przez Gaussy (biały)")):
            ax.imshow(image, cmap="gray", vmin=0, vmax=1); ax.set_title(f"{name}: {title}", fontsize=9); ax.axis("off")
    fig.tight_layout(); fig.savefig(args.output / "invaded_water.png", dpi=80); plt.close(fig)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(0)
    background = np.flatnonzero(rest)
    background = rng.choice(background, min(args.plot_background, len(background)), replace=False)
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    for ax, (a, b) in zip(axes, ((0, 2), (0, 1), (2, 1))):
        ax.scatter(g["means"][background, a], g["means"][background, b], s=0.2, c="0.6", alpha=0.3, label="pozostałe Gaussy")
        ax.scatter(g["means"][water, a], g["means"][water, b], s=1.5, c="tab:red", alpha=0.6, label="Gaussy na wodzie")
        ax.scatter(cameras[:, a], cameras[:, b], s=25, c="tab:blue", marker="^", label="kamery")
        ax.set_xlabel("XYZ"[a]); ax.set_ylabel("XYZ"[b]); ax.set_aspect("equal", adjustable="datalim")
    axes[0].legend(markerscale=8, loc="best")
    fig.tight_layout(); fig.savefig(args.output / "projections.png", dpi=110); plt.close(fig)

    import plotly.graph_objects as go
    figure = go.Figure([
        go.Scatter3d(x=g["means"][background, 0], y=g["means"][background, 1], z=g["means"][background, 2],
                     mode="markers", name="pozostałe Gaussy (próbka)", marker=dict(size=1, color="#9a9a9a", opacity=0.35)),
        go.Scatter3d(x=g["means"][water, 0], y=g["means"][water, 1], z=g["means"][water, 2], mode="markers",
                     name="Gaussy na wodzie", marker=dict(size=1.6, color=np.log10(cluster_size[water]),
                     colorscale="Turbo", colorbar=dict(title="log10 rozmiaru skupiska"), opacity=0.8)),
        go.Scatter3d(x=cameras[:, 0], y=cameras[:, 1], z=cameras[:, 2], mode="markers", name="kamery treningowe",
                     marker=dict(size=4, color="#2a78d6", symbol="diamond")),
    ])
    figure.update_layout(title="Środki Gaussów: zasłaniające otwartą wodę vs pozostałe",
                         scene=dict(aspectmode="data"), legend=dict(itemsizing="constant"))
    figure.write_html(args.output / "water_gaussians_3d.html", include_plotlyjs="cdn")
    print(json.dumps({k: stats[k] for k in ("gaussians", "visible", "water_gaussians", "cluster_radius", "median_spacing")}))
    for name in groups:
        s = stats[name]
        print(name, "count", s["count"], "| cluster p10/p50/p90", [round(s["cluster_size"][q]) for q in ("p10", "p50", "p90")],
              "| <=10:", round(s["in_cluster_le_10"], 3), "<=100:", round(s["in_cluster_le_100"], 3),
              "| COLMAP dist p50/p90", round(s["colmap_distance"]["p50"], 2), round(s["colmap_distance"]["p90"], 2),
              "| camera dist p50", round(s["camera_distance"]["p50"], 2),
              "| opacity p50", round(s["opacity"]["p50"], 2), "| scale p50", round(s["scale"]["p50"], 3))
    print(json.dumps(per_view, indent=1))


if __name__ == "__main__":
    main()
