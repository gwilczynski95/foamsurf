"""Paint an uncoloured STL by its connected parts and export it for insert.py (examples.insertion.mesh.load_manifest).

    python scripts/export_stl.py MODEL.stl OUT_DIR [--paint paint.json]

The STL (z-up, as exported from most modelling tools) is split into connected components; each face gets one
of these parts:
    hull   components that are not thin and whose middle extent is at least --hull_size x the model's length
           (the body, tanks, fins, housings),
    pipe   elongated components (longest extent > --pipe_ratio x the middle one): rails, pipes, cables,
    detail everything else (small fittings),
    stripe faces of the largest component (the main body) whose x (in the STL's units) falls in one of
           --stripes [[x0, x1], ...] (paint bands).
OUT_DIR/mesh.npz (vertices y-up, faces, part) and OUT_DIR/manifest.json are written; --paint overrides the
materials ({"hull": {"mode": ..., "rgb": ..., "kd": ..., "eta": ...}, ...}).
"""
import argparse
import json
import struct
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

PAINT = {  # FALCON-style materials
    "hull": {"mode": "ceramic", "rgb": [0.72, 0.1, 0.05], "kd": 1.0, "eta": 1.5},
    "stripe": {"mode": "ceramic", "rgb": [0.85, 0.85, 0.82], "kd": 1.0, "eta": 1.5},
    "pipe": {"mode": "metal", "rgb": [0.75, 0.75, 0.75], "kd": 0.6, "eta": 40.0},
    "detail": {"mode": "metal", "rgb": [0.35, 0.35, 0.37], "kd": 0.3, "eta": 40.0},
}


def read_stl(path: Path):
    data = path.read_bytes()
    n = struct.unpack("<I", data[80:84])[0]
    if len(data) != 84 + 50 * n:
        raise ValueError("only binary STL is supported")
    tri = np.frombuffer(data[84:], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    corners = tri["v"].reshape(-1, 3).astype(np.float64)
    scale = np.abs(corners).max()
    _, first, inverse = np.unique(np.round(corners / (scale * 1e-7)).astype(np.int64), axis=0, return_index=True,
                                  return_inverse=True)
    return corners[first], inverse.reshape(-1, 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stl", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--paint", type=Path)
    parser.add_argument("--hull_size", type=float, default=0.03)
    parser.add_argument("--pipe_ratio", type=float, default=4.0)
    parser.add_argument("--stripes", type=json.loads, default=[])
    args = parser.parse_args()
    vertices, faces = read_stl(args.stl)
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]]))
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(vertices),) * 2)
    count, label = connected_components(graph, directed=False)
    component = label[faces[:, 0]]
    corners = vertices[faces]
    low = np.full((count, 3), np.inf)
    high = np.full((count, 3), -np.inf)
    np.minimum.at(low, component, corners.min(1))
    np.maximum.at(high, component, corners.max(1))
    extent = np.sort(high - low, axis=1)  # ascending
    thin = extent[:, 2] > args.pipe_ratio * np.maximum(extent[:, 1], 1e-9)
    names = ["hull", "stripe", "pipe", "detail"]
    length = float((vertices.max(0) - vertices.min(0)).max())
    kind = np.where(thin, 2, np.where(extent[:, 1] >= args.hull_size * length, 0, 3))
    part = kind[component]
    x = corners[:, :, 0].mean(1)
    for x0, x1 in args.stripes:
        part[(component == np.bincount(component).argmax()) & (part == 0) & (x >= x0) & (x <= x1)] = 1
    paint = dict(PAINT, **(json.loads(args.paint.read_text()) if args.paint else {}))
    v = np.stack((vertices[:, 0], vertices[:, 2], -vertices[:, 1]), -1).astype(np.float32)  # z-up -> y-up
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "mesh.npz", vertices=v, faces=faces.astype(np.int64), part=part.astype(np.int64),
                        corner_uv=np.zeros((len(faces), 3, 2), dtype=np.float32))
    manifest = {"format": "npz", "mesh": "mesh.npz", "source": str(args.stl.name),
                "parts": [{"id": n, "material": paint[n]} for n in names]}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"{len(v)} vertices, {len(faces)} faces, {count} components;",
          {n: int((part == i).sum()) for i, n in enumerate(names)})


if __name__ == "__main__":
    main()
