#!/usr/bin/env python3
"""SfM points for a Blender scene from its own photos and known camera poses, so that the synthetic scenes start
from SfM points like the real ones. COLMAP extracts and matches SIFT features and triangulates the matches with
the given poses and intrinsics held fixed (no pose estimation). Run by scripts/blender_sfm.sbatch:

    python scripts/blender_sfm.py prepare SCENE OUT   # OUT/images: training photos on black, OUT/camera.json, OUT/poses.json
    (colmap feature_extractor, exhaustive_matcher on OUT/database.db)
    python scripts/blender_sfm.py model OUT           # OUT/sparse_known: the poses under the database's image ids
    (colmap point_triangulator -> OUT/sparse/0)
"""
import json
import math
import sqlite3
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def blender_to_colmap(transform_matrix):
    """(qvec (w, x, y, z), tvec) of COLMAP's world-to-camera pose for a Blender camera-to-world matrix (OpenGL axes),
    the same conversion as foamsurf_data/blender.py: right = R row 0, down = row 1, forward = row 2."""
    c2w = np.asarray(transform_matrix, dtype=np.float64).copy()
    c2w[:3, 1:3] *= -1.0  # OpenGL (up, back) -> OpenCV (down, forward)
    w2c = np.linalg.inv(c2w)
    return rotation_to_quaternion(w2c[:3, :3]), w2c[:3, 3]


def rotation_to_quaternion(R):
    """Unit quaternion (w, x, y, z), w >= 0, of a rotation matrix."""
    R = np.asarray(R, dtype=np.float64)
    trace = np.trace(R)
    if trace > 0:
        s = 2.0 * math.sqrt(trace + 1.0)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def prepare(scene, out):
    scene, out = Path(scene), Path(out)
    if out.exists():
        raise FileExistsError(f"{out} exists; remove it or choose another output directory")
    (out / "images").mkdir(parents=True)
    meta = json.loads((scene / "transforms_train.json").read_text())
    poses = {}
    for frame in meta["frames"]:
        name = Path(frame["file_path"]).name + ".png"
        with Image.open(scene / f"{frame['file_path']}.png") as image:
            rgba = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0
        rgb = rgba[..., :3] * rgba[..., 3:4]  # on black, as the training target
        Image.fromarray(np.round(rgb * 255.0).astype(np.uint8)).save(out / "images" / name)
        qvec, tvec = blender_to_colmap(frame["transform_matrix"])
        poses[name] = {"qvec": qvec.tolist(), "tvec": tvec.tolist()}
        width, height = rgba.shape[1], rgba.shape[0]
    focal = width / (2.0 * math.tan(0.5 * float(meta["camera_angle_x"])))
    camera = {"width": width, "height": height, "fx": focal, "fy": focal, "cx": width / 2.0, "cy": height / 2.0}
    (out / "camera.json").write_text(json.dumps(camera))
    (out / "poses.json").write_text(json.dumps(poses))
    print(f"{len(poses)} photos, camera {camera}")


def model(out):
    """sparse_known/{cameras,images,points3D}.txt with the image and camera ids of OUT/database.db."""
    out = Path(out)
    camera = json.loads((out / "camera.json").read_text())
    poses = json.loads((out / "poses.json").read_text())
    with sqlite3.connect(out / "database.db") as db:
        images = db.execute("SELECT image_id, name, camera_id FROM images").fetchall()
        camera_ids = {row[0] for row in db.execute("SELECT camera_id FROM cameras")}
    if camera_ids != {1} or len(images) != len(poses) or {name for _, name, _ in images} != set(poses):
        raise ValueError(f"database does not match the prepared photos: cameras {camera_ids}, {len(images)} images")
    known = out / "sparse_known"
    known.mkdir()
    (known / "cameras.txt").write_text(
        f"1 PINHOLE {camera['width']} {camera['height']} {camera['fx']!r} {camera['fy']!r} {camera['cx']!r} {camera['cy']!r}\n")
    lines = []
    for image_id, name, camera_id in sorted(images):
        q, t = poses[name]["qvec"], poses[name]["tvec"]
        lines += [f"{image_id} {q[0]!r} {q[1]!r} {q[2]!r} {q[3]!r} {t[0]!r} {t[1]!r} {t[2]!r} {camera_id} {name}", ""]
    (known / "images.txt").write_text("\n".join(lines) + "\n")
    (known / "points3D.txt").write_text("")
    print(f"sparse_known: {len(images)} images with fixed poses")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "prepare":
        prepare(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 3 and sys.argv[1] == "model":
        model(sys.argv[2])
    else:
        sys.exit(__doc__)
