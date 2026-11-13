"""A procedural fish mesh for vis/render_clip.py (vis.mesh.load_manifest, "format": "npz").

    python vis/make_fish.py OUT_DIR [--kind silver|tang|clown]

The body is a laterally flattened surface of revolution along +x (tail at x = 0, nose at x = 1), with a forked
tail fin, a dorsal fin, a pair of pectoral fins and eyes. Parts and their default materials by kind:
    silver  a sardine / herring: dark blue-grey metallic back, mirror-like silver belly (reflects the scene),
    tang    a yellow tang: glossy yellow lacquer body and fins,
    clown   a clownfish: orange body with three white bands, black-edged fins.
OUT_DIR/mesh.npz (vertices y-up, faces, part, corner_uv zeros) and OUT_DIR/manifest.json are written.
"""
import argparse
import json
from pathlib import Path

import numpy as np

KINDS = {
    "silver": {"back": {"mode": "metal", "rgb": [0.32, 0.42, 0.52], "kd": 0.55, "eta": 40.0},
               "belly": {"mode": "metal", "rgb": [0.92, 0.94, 0.96], "kd": 0.85, "eta": 40.0},
               "band": {"mode": "metal", "rgb": [0.92, 0.94, 0.96], "kd": 0.85, "eta": 40.0},
               "fin": {"mode": "ceramic", "rgb": [0.55, 0.6, 0.65], "kd": 1.0, "eta": 1.5},
               "eye": {"mode": "ceramic", "rgb": [0.02, 0.02, 0.02], "kd": 1.0, "eta": 1.6}},
    "tang": {"back": {"mode": "ceramic", "rgb": [0.95, 0.75, 0.02], "kd": 1.0, "eta": 1.5},
             "belly": {"mode": "ceramic", "rgb": [0.95, 0.78, 0.05], "kd": 1.0, "eta": 1.5},
             "band": {"mode": "ceramic", "rgb": [0.95, 0.78, 0.05], "kd": 1.0, "eta": 1.5},
             "fin": {"mode": "ceramic", "rgb": [0.92, 0.7, 0.02], "kd": 1.0, "eta": 1.5},
             "eye": {"mode": "ceramic", "rgb": [0.02, 0.02, 0.02], "kd": 1.0, "eta": 1.6}},
    "clown": {"back": {"mode": "ceramic", "rgb": [0.95, 0.32, 0.02], "kd": 1.0, "eta": 1.5},
              "belly": {"mode": "ceramic", "rgb": [0.95, 0.38, 0.04], "kd": 1.0, "eta": 1.5},
              "band": {"mode": "ceramic", "rgb": [0.95, 0.95, 0.92], "kd": 1.0, "eta": 1.5},
              "fin": {"mode": "ceramic", "rgb": [0.9, 0.3, 0.02], "kd": 1.0, "eta": 1.5},
              "eye": {"mode": "ceramic", "rgb": [0.02, 0.02, 0.02], "kd": 1.0, "eta": 1.6}},
}
SHAPE = {  # max half-height, width / height, tail fork, dorsal fin height (in body lengths)
    "silver": (0.11, 0.5, 0.17, 0.06),
    "tang": (0.24, 0.32, 0.12, 0.08),
    "clown": (0.16, 0.5, 0.1, 0.07),
}
PARTS = ["back", "belly", "band", "fin", "eye"]


def body(height, aspect, rings=48, segments=32):
    """Surface of revolution along x with elliptic cross sections, closed at the nose and at the tail stalk."""
    x = np.linspace(0.0, 1.0, rings)
    profile = 0.025 + height * np.sin(np.pi * np.clip(x, 0, 1) ** 0.75) ** 0.8
    profile[-1] = 0.0  # nose
    a = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    v = []
    for xi, r in zip(x, profile):
        for ai in a:
            v.append((xi, r * np.cos(ai), aspect * r * np.sin(ai)))
    v = np.array(v)
    f = []
    for i in range(rings - 1):
        for j in range(segments):
            p, q = i * segments + j, i * segments + (j + 1) % segments
            f += [(p, q, q + segments), (p, q + segments, p + segments)]
    tail_center = len(v)
    v = np.vstack((v, [(0.0, 0.0, 0.0)]))
    f += [(tail_center, (j + 1) % segments, j) for j in range(segments)]
    return v, np.array(f), x, profile


def slab(outline, thickness):
    """A thin closed prism from a convex-ish 2D outline [(x, y)] in the z = 0 plane."""
    n = len(outline)
    o = np.array(outline, dtype=np.float64)
    v = np.vstack((np.c_[o, np.full(n, thickness / 2)], np.c_[o, np.full(n, -thickness / 2)]))
    c = o.mean(0)
    v = np.vstack((v, [(*c, thickness / 2), (*c, -thickness / 2)]))
    top, bottom = 2 * n, 2 * n + 1
    f = []
    for i in range(n):
        j = (i + 1) % n
        f += [(top, i, j), (bottom, n + j, n + i), (i, n + i, n + j), (i, n + j, j)]
    return v, np.array(f)


def ball(center, radius, k=2):
    t = (1 + 5 ** 0.5) / 2
    v = np.array([(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
                  (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)], dtype=np.float64)
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6),
         (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10),
         (8, 6, 7), (9, 8, 1)]
    for _ in range(k):
        v, mids, nf = list(v), {}, []
        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in mids:
                mids[key] = len(v)
                v.append((np.array(v[a]) + np.array(v[b])) / 2)
            return mids[key]
        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        v, f = np.array(v), nf
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    return center + radius * v, np.array(f)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path)
    parser.add_argument("--kind", choices=sorted(KINDS), default="silver")
    args = parser.parse_args()
    height, aspect, fork, dorsal = SHAPE[args.kind]
    meshes = []  # (vertices, faces, part per face)
    v, f, x, profile = body(height, aspect)
    centers = v[f].mean(1)
    part = np.where(centers[:, 1] > -0.15 * np.interp(centers[:, 0], x, profile), 0, 1)
    if args.kind == "clown":
        for lo, hi in ((0.72, 0.8), (0.42, 0.52), (0.08, 0.14)):
            part[(centers[:, 0] > lo) & (centers[:, 0] < hi)] = 2
    meshes.append((v, f, part))
    t = 0.012
    tail = [(0.04, 0.02), (-fork * 1.1, fork), (-fork * 0.6, 0.0), (-fork * 1.1, -fork), (0.04, -0.02)]
    meshes.append((*slab(tail, t), 3))
    h0 = np.interp(0.5, x, profile)
    top = [(0.32, h0 * 0.85), (0.45, h0 + dorsal), (0.7, h0 * 0.9)]
    meshes.append((*slab(top, t), 3))
    bottom = [(0.25, -h0 * 0.8), (0.3, -h0 - dorsal * 0.6), (0.45, -h0 * 0.85)]
    meshes.append((*slab(bottom, t), 3))
    for side in (1, -1):
        hp = np.interp(0.72, x, profile)
        pv, pf = slab([(0.74, 0.0), (0.6, -0.05), (0.62, 0.01)], t)
        pv[:, 2] += side * aspect * hp * 0.95
        pv[:, 1] -= hp * 0.3
        meshes.append((pv, pf, 3))
        he = np.interp(0.86, x, profile)
        ev, ef = ball(np.array([0.86, he * 0.3, side * aspect * he * 0.85]), 0.022)
        meshes.append((ev, ef, 4))
    vertices, faces, parts, offset = [], [], [], 0
    for mv, mf, mp in meshes:
        vertices.append(mv)
        faces.append(np.asarray(mf) + offset)
        parts.append(np.broadcast_to(mp, (len(mf),)))
        offset += len(mv)
    vertices = np.concatenate(vertices).astype(np.float32)
    faces = np.concatenate(faces).astype(np.int64)
    parts = np.concatenate(parts).astype(np.int64)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "mesh.npz", vertices=vertices, faces=faces, part=parts,
                        corner_uv=np.zeros((len(faces), 3, 2), dtype=np.float32))
    manifest = {"format": "npz", "mesh": "mesh.npz", "source": f"vis/make_fish.py --kind {args.kind}",
                "parts": [{"id": n, "material": KINDS[args.kind][n]} for n in PARTS]}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(args.out, len(vertices), "vertices", len(faces), "faces")


if __name__ == "__main__":
    main()
