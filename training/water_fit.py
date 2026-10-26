"""Water parameters from the training photos, before training (no masks).

The same COLMAP point seen from several training cameras at different distances d obeys (SeaThru,
single scattering) c = B_inf * (1 - exp(-beta_B d)) + J * exp(-beta_D d), per channel, J being the
point's own colour. A grid search over (beta_D, beta_B) with B_inf and every J solved in closed form
gives one water per scene; it initialises every Voronoi cell (attenuation = beta_D,
backscatter = beta_B, optionally colour = B_inf). For small beta_B d only B_inf * beta_B is well
determined, so when the cells keep another colour (the photos' mean), B_inf is fixed to that colour
and beta_B is fitted for it.
"""
import struct
from pathlib import Path

import numpy as np

GRID = np.geomspace(0.003, 0.5, 20)


def _unpack(file, fmt):
    size = struct.calcsize("<" + fmt)
    data = file.read(size)
    if len(data) != size:
        raise ValueError("unexpected end of COLMAP binary file")
    return struct.unpack("<" + fmt, data)


def read_tracks(sparse: Path):
    """images: id -> (name, camera id, camera centre, keypoints [(x, y, point id)]); points: id -> xyz."""
    images, points, widths = {}, {}, {}
    with open(sparse / "cameras.bin", "rb") as f:
        (count,) = _unpack(f, "Q")
        params = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8}
        for _ in range(count):
            camera_id, model_id, width, _height = _unpack(f, "iiQQ")
            _unpack(f, "d" * params[model_id])
            widths[camera_id] = width
    with open(sparse / "images.bin", "rb") as f:
        (count,) = _unpack(f, "Q")
        for _ in range(count):
            image_id, w, x, y, z, tx, ty, tz, camera_id = _unpack(f, "idddddddi")
            name = b""
            while (char := f.read(1)) != b"\x00":
                name += char
            (k,) = _unpack(f, "Q")
            keypoints = np.frombuffer(f.read(24 * k), dtype=np.dtype([("x", "<f8"), ("y", "<f8"), ("p", "<i8")]))
            rotation = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                                 [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                                 [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
            images[image_id] = (name.decode(), camera_id, -rotation.T @ np.array([tx, ty, tz]), keypoints)
    with open(sparse / "points3D.bin", "rb") as f:
        (count,) = _unpack(f, "Q")
        for _ in range(count):
            point_id, x, y, z, *_ = _unpack(f, "QdddBBBd")
            (k,) = _unpack(f, "Q")
            f.read(8 * k)
            points[point_id] = np.array([x, y, z])
    return images, points, widths


def observations(images, points, widths, photos, min_views=3, min_spread=1.3, max_points=3000, seed=0):
    """(point index, colour [3], distance) of points seen by >= min_views photos with distance
    spread >= min_spread; photos: name stem -> [H, W, 3] array in [0, 1]."""
    tracks = {}
    for name, camera_id, centre, keypoints in images.values():
        photo = photos.get(Path(name).stem)
        if photo is None:
            continue
        scale = photo.shape[1] / widths[camera_id]
        for x, y, point_id in keypoints:
            if point_id >= 0 and point_id in points:
                tracks.setdefault(int(point_id), []).append((photo, x * scale, y * scale, centre))
    chosen = []
    for point_id, track in tracks.items():
        if len(track) < min_views:
            continue
        d = [np.linalg.norm(points[point_id] - c) for *_, c in track]
        if max(d) / min(d) >= min_spread:
            chosen.append(point_id)
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.permutation(np.array(chosen, dtype=np.int64))[:max_points].tolist())
    index, colours, distances = [], [], []
    for k, point_id in enumerate(chosen):
        for photo, x, y, centre in tracks[point_id]:
            xi, yi = int(round(x)), int(round(y))
            if 1 <= xi < photo.shape[1] - 1 and 1 <= yi < photo.shape[0] - 1:
                index.append(k)
                colours.append(photo[yi - 1:yi + 2, xi - 1:xi + 2].reshape(-1, 3).mean(0))
                distances.append(np.linalg.norm(points[point_id] - centre))
    return np.array(index), np.array(colours), np.array(distances)


def _channel_error(index, colour, distance, beta_d, beta_b, b_inf=None):
    """Mean squared error of one channel for given betas, exact least squares in B_inf (unless
    given) and the J's.

    J_p = sum e (c - B h) / sum e^2 per point; substituted, the residual is a - B b with
    a = c - e S_ec / S_ee and b = h - e S_eh / S_ee, so B = sum a b / sum b^2.
    """
    direct = np.exp(-beta_d * distance)
    haze = 1 - np.exp(-beta_b * distance)
    count = index.max() + 1
    s_ee = np.maximum(np.bincount(index, direct * direct, count), 1e-12)
    a = colour - direct * (np.bincount(index, direct * colour, count) / s_ee)[index]
    b = haze - direct * (np.bincount(index, direct * haze, count) / s_ee)[index]
    if b_inf is None:
        b_inf = float((a * b).sum() / max((b * b).sum(), 1e-12))
    return float(((a - b_inf * b) ** 2).mean()), b_inf


def fit_water(index, colours, distances, grid=GRID, b_inf=None):
    """Per channel: beta_D, beta_B (grid) and B_inf (fitted, or b_inf [3] fixed); dict of 3-lists."""
    out = dict(beta_d=[], beta_b=[], b_inf=[])
    for channel in range(3):
        fixed = None if b_inf is None else float(b_inf[channel])
        errors = np.array([[_channel_error(index, colours[:, channel], distances, bd, bb, fixed)[0] for bb in grid]
                           for bd in grid])
        i, j = np.unravel_index(errors.argmin(), errors.shape)
        out["beta_d"].append(float(grid[i]))
        out["beta_b"].append(float(grid[j]))
        value = _channel_error(index, colours[:, channel], distances, grid[i], grid[j], fixed)[1]
        out["b_inf"].append(float(np.clip(value, 0.02, 0.98)))
    return out


def fit_water_from_dataset(dataset, source_path, b_inf=None):
    """Fit on the training split of a loaded COLMAP dataset (photos from camera.foreground);
    b_inf [3]: fix B_inf to this colour."""
    from foamsurf_data.colmap import colmap_model_dir
    sparse = colmap_model_dir(source_path)
    if sparse is None or not (sparse / "points3D.bin").is_file():
        raise ValueError(f"water fit needs a binary COLMAP model with points3D.bin in {source_path}")
    images, points, widths = read_tracks(sparse)
    photos = {camera.name: camera.foreground.detach().cpu().numpy() for camera in dataset.train}
    index, colours, distances = observations(images, points, widths, photos)
    if len(index) < 100:
        raise ValueError(f"water fit: only {len(index)} usable observations")
    result = fit_water(index, colours, distances, b_inf=b_inf)
    result.update(observations=int(len(index)), points=int(index.max() + 1))
    return result
