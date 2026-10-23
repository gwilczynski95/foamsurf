from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import struct

import numpy as np
from PIL import Image
import torch

from .blender import Camera


_CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


@dataclass(frozen=True)
class _Intrinsics:
    model: str
    width: int
    height: int
    params: np.ndarray


@dataclass(frozen=True)
class _Extrinsics:
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str


def _unpack(file, format_string: str):
    size = struct.calcsize("<" + format_string)
    data = file.read(size)
    if len(data) != size:
        raise ValueError("unexpected end of COLMAP binary file")
    return struct.unpack("<" + format_string, data)


def colmap_model_dir(root: str | Path) -> Path | None:
    """COLMAP model directory: ``sparse/0`` (classic layout) or flat ``sparse/``.

    Newer COLMAP exports write the model directly into ``sparse/`` (with extra
    rigs.bin/frames.bin, which are not needed here). Returns None if neither
    holds a cameras + images model.
    """
    for candidate in (Path(root) / "sparse" / "0", Path(root) / "sparse"):
        for suffix in (".bin", ".txt"):
            if (candidate / f"cameras{suffix}").is_file() and (candidate / f"images{suffix}").is_file():
                return candidate
    return None


def _read_cameras_binary(path: Path) -> dict[int, _Intrinsics]:
    cameras = {}
    with path.open("rb") as file:
        (count,) = _unpack(file, "Q")
        for _ in range(count):
            camera_id, model_id, width, height = _unpack(file, "iiQQ")
            if model_id not in _CAMERA_MODELS:
                raise ValueError(f"unsupported COLMAP camera model id {model_id}")
            model, parameter_count = _CAMERA_MODELS[model_id]
            params = np.asarray(
                _unpack(file, "d" * parameter_count), dtype=np.float64
            )
            cameras[camera_id] = _Intrinsics(
                model, int(width), int(height), params
            )
    return cameras


def _read_images_binary(path: Path) -> dict[int, _Extrinsics]:
    images = {}
    with path.open("rb") as file:
        (count,) = _unpack(file, "Q")
        for _ in range(count):
            values = _unpack(file, "idddddddi")
            image_id = int(values[0])
            name_bytes = bytearray()
            while True:
                character = file.read(1)
                if character == b"\x00":
                    break
                if not character:
                    raise ValueError("unterminated image name in images.bin")
                name_bytes.extend(character)
            (point_count,) = _unpack(file, "Q")
            file.seek(24 * point_count, 1)
            images[image_id] = _Extrinsics(
                qvec=np.asarray(values[1:5], dtype=np.float64),
                tvec=np.asarray(values[5:8], dtype=np.float64),
                camera_id=int(values[8]),
                name=name_bytes.decode("utf-8"),
            )
    return images


def _read_points_binary(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as file:
        (count,) = _unpack(file, "Q")
        xyz = np.empty((count, 3), dtype=np.float32)
        rgb = np.empty((count, 3), dtype=np.float32)
        for index in range(count):
            values = _unpack(file, "QdddBBBd")
            xyz[index] = values[1:4]
            rgb[index] = values[4:7]
            (track_length,) = _unpack(file, "Q")
            file.seek(8 * track_length, 1)
    return xyz, rgb / 255.0


def _read_cameras_text(path: Path) -> dict[int, _Intrinsics]:
    cameras = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        values = line.split()
        camera_id = int(values[0])
        cameras[camera_id] = _Intrinsics(
            values[1],
            int(values[2]),
            int(values[3]),
            np.asarray(values[4:], dtype=np.float64),
        )
    return cameras


def _read_images_text(path: Path) -> dict[int, _Extrinsics]:
    images = {}
    lines = iter(path.read_text().splitlines())
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        values = line.split()
        image_id = int(values[0])
        images[image_id] = _Extrinsics(
            qvec=np.asarray(values[1:5], dtype=np.float64),
            tvec=np.asarray(values[5:8], dtype=np.float64),
            camera_id=int(values[8]),
            name=values[9],
        )
        next(lines, None)
    return images


def _read_points_text(path: Path) -> tuple[np.ndarray, np.ndarray]:
    xyz = []
    rgb = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        values = line.split()
        xyz.append([float(value) for value in values[1:4]])
        rgb.append([int(value) for value in values[4:7]])
    return (
        np.asarray(xyz, dtype=np.float32),
        np.asarray(rgb, dtype=np.float32) / 255.0,
    )


def _rotation_from_qvec(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = qvec
    return np.asarray(
        [
            [
                1 - 2 * y * y - 2 * z * z,
                2 * x * y - 2 * w * z,
                2 * z * x + 2 * w * y,
            ],
            [
                2 * x * y + 2 * w * z,
                1 - 2 * x * x - 2 * z * z,
                2 * y * z - 2 * w * x,
            ],
            [
                2 * z * x - 2 * w * y,
                2 * y * z + 2 * w * x,
                1 - 2 * x * x - 2 * y * y,
            ],
        ],
        dtype=np.float64,
    )


def _focal_lengths(intrinsics: _Intrinsics) -> tuple[float, float]:
    if intrinsics.model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
        return float(intrinsics.params[0]), float(intrinsics.params[0])
    if intrinsics.model == "PINHOLE":
        return float(intrinsics.params[0]), float(intrinsics.params[1])
    raise ValueError(
        "COLMAP images must be undistorted; supported camera models are "
        "SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, and RADIAL"
    )


class ColmapDataset:
    def __init__(
        self,
        root: str | Path,
        resolution: int = 1,
        device: str | torch.device = "cuda",
        llffhold: int = 8,
    ) -> None:
        self.root = Path(root)
        self.resolution = int(resolution)
        self.device = torch.device(device)
        self.sampling_mode = "camera"
        self.llffhold = int(llffhold)
        if self.resolution < 1:
            raise ValueError("resolution must be positive")
        if self.llffhold < 2:
            raise ValueError("llffhold must be at least 2")

        sparse = colmap_model_dir(self.root)
        if sparse is None:
            raise ValueError(f"no COLMAP model in {self.root}/sparse/0 or {self.root}/sparse")
        if (sparse / "cameras.bin").exists():
            intrinsics = _read_cameras_binary(sparse / "cameras.bin")
            extrinsics = _read_images_binary(sparse / "images.bin")
            self._points, self._colors = _read_points_binary(
                sparse / "points3D.bin"
            )
        else:
            intrinsics = _read_cameras_text(sparse / "cameras.txt")
            extrinsics = _read_images_text(sparse / "images.txt")
            self._points, self._colors = _read_points_text(
                sparse / "points3D.txt"
            )

        image_root = self.root / f"images_{self.resolution}"
        use_prescaled = self.resolution != 1 and image_root.is_dir()
        if not use_prescaled:
            image_root = self.root / "images"

        cameras = []
        for record in sorted(extrinsics.values(), key=lambda item: item.name):
            calibration = intrinsics[record.camera_id]
            image_path = image_root / record.name
            if not image_path.is_file():
                raise FileNotFoundError(f"missing COLMAP image: {image_path}")
            with Image.open(image_path) as image:
                rgba = image.convert("RGBA")
                if not use_prescaled and self.resolution != 1:
                    size = (
                        max(1, round(rgba.width / self.resolution)),
                        max(1, round(rgba.height / self.resolution)),
                    )
                    rgba = rgba.resize(size, Image.Resampling.BILINEAR)
                pixels = torch.from_numpy(
                    np.asarray(rgba, dtype=np.float32).copy()
                ).to(self.device) / 255.0

            rotation = _rotation_from_qvec(record.qvec)
            origin = -rotation.T @ record.tvec
            basis = torch.as_tensor(
                rotation, dtype=torch.float32, device=self.device
            )
            fx, fy = _focal_lengths(calibration)
            fov_x = 2.0 * math.atan(calibration.width / (2.0 * fx))
            fov_y = 2.0 * math.atan(calibration.height / (2.0 * fy))
            alpha = pixels[..., 3:4]
            if calibration.model not in {'PINHOLE', 'SIMPLE_PINHOLE'}:
                raise ValueError('Undistort COLMAP images before training (PINHOLE required)')
            cx, cy = calibration.params[-2:]
            sx, sy = pixels.shape[1] / calibration.width, pixels.shape[0] / calibration.height
            cameras.append(
                Camera(
                    name=Path(record.name).stem,
                    origin=torch.as_tensor(
                        origin, dtype=torch.float32, device=self.device
                    ),
                    right=basis[0],
                    down=basis[1],
                    forward=basis[2],
                    intrinsics=(fx * sx, fy * sy, float(cx) * sx, float(cy) * sy),
                    fov_x=fov_x,
                    fov_y=fov_y,
                    foreground=pixels[..., :3] * alpha,
                    alpha=alpha,
                )
            )

        self.test = [
            camera for index, camera in enumerate(cameras)
            if index % self.llffhold == 0
        ]
        self.train = [
            camera for index, camera in enumerate(cameras)
            if index % self.llffhold != 0
        ]
        if not self.train or not self.test:
            raise ValueError("COLMAP LLFF split produced an empty split")
        self.width = self.train[0].width
        self.height = self.train[0].height
        for camera in self.train + self.test:
            if (camera.height, camera.width) != (self.height, self.width):
                raise ValueError("all COLMAP images must have the same resolution")

    def stacked_training_data(self, pixels: bool = True) -> dict[str, torch.Tensor]:
        """Per-camera tensors of the training split; with pixels, also a second (stacked) copy of
        every image, which the full-image sampler does not need (it reads camera.foreground)."""
        data = {
            "intrinsics": torch.tensor([c.intrinsics for c in self.train], dtype=torch.float32, device=self.device),
            "origins": torch.stack([camera.origin for camera in self.train]),
            "rights": torch.stack([camera.right for camera in self.train]),
            "downs": torch.stack([camera.down for camera in self.train]),
            "forwards": torch.stack([camera.forward for camera in self.train]),
            "fov_x": torch.tensor(
                [camera.fov_x for camera in self.train],
                dtype=torch.float32,
                device=self.device,
            ),
            "fov_y": torch.tensor(
                [camera.fov_y for camera in self.train],
                dtype=torch.float32,
                device=self.device,
            ),
        }
        if pixels:
            data["foreground"] = torch.stack([camera.foreground for camera in self.train]).reshape(-1, 3)
            data["alpha"] = torch.stack([camera.alpha for camera in self.train]).reshape(-1, 1)
        return data

    def initial_point_cloud(
        self, _synthetic_count: int = 100_000
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._points.copy(), self._colors.copy()
