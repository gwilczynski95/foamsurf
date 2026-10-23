from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .rays import RayBundle, generate_camera_rays


@dataclass
class Camera:
    name: str
    origin: torch.Tensor
    right: torch.Tensor
    down: torch.Tensor
    forward: torch.Tensor
    fov_x: float
    fov_y: float
    foreground: torch.Tensor
    alpha: torch.Tensor
    intrinsics: tuple[float, float, float, float] | None = None

    @property
    def height(self) -> int:
        return int(self.foreground.shape[0])

    @property
    def width(self) -> int:
        return int(self.foreground.shape[1])

    def rays(self) -> RayBundle:
        return generate_camera_rays(
            self.origin,
            self.right,
            self.down,
            self.forward,
            self.width,
            self.height,
            self.fov_x,
            self.fov_y,
            intrinsics=self.intrinsics,
        )

    def ground_truth(
        self, background: torch.Tensor | tuple[float, float, float]
    ) -> torch.Tensor:
        background_tensor = torch.as_tensor(
            background, device=self.foreground.device, dtype=self.foreground.dtype
        )
        return self.foreground + background_tensor * (1.0 - self.alpha)


class BlenderDataset:
    def __init__(
        self,
        root: str | Path,
        resolution: int = 1,
        device: str | torch.device = "cuda",
        splits: tuple[str, ...] = ("train", "test"),
    ) -> None:
        self.root = Path(root)
        self.resolution = int(resolution)
        self.device = torch.device(device)
        self.sampling_mode = "mixed"
        unknown_splits = set(splits) - {"train", "test"}
        if unknown_splits:
            raise ValueError(f"unknown Blender splits: {unknown_splits}")
        self.train = self._load_split("train") if "train" in splits else []
        self.test = self._load_split("test") if "test" in splits else []
        if "train" in splits and not self.train:
            raise ValueError("training split is empty")
        cameras = self.train or self.test
        if not cameras:
            raise ValueError("at least one Blender split must be loaded")
        self.width = cameras[0].width
        self.height = cameras[0].height
        self._validate_camera_shapes()

    def _load_split(self, split: str) -> list[Camera]:
        metadata = json.loads(
            (self.root / f"transforms_{split}.json").read_text()
        )
        fov_x = float(metadata["camera_angle_x"])
        cameras = []
        for frame in metadata["frames"]:
            image_path = self.root / f"{frame['file_path']}.png"
            with Image.open(image_path) as image:
                rgba = image.convert("RGBA")
                width = max(1, round(rgba.width / self.resolution))
                height = max(1, round(rgba.height / self.resolution))
                rgba = rgba.resize((width, height))
                pixels = torch.from_numpy(
                    np.asarray(rgba, dtype=np.float32).copy()
                ).to(self.device) / 255.0

            c2w = np.asarray(frame["transform_matrix"]).copy()
            c2w[:3, 1:3] *= -1.0
            w2c = np.linalg.inv(c2w)
            stored_rotation = np.transpose(w2c[:3, :3])
            translation = w2c[:3, 3]
            camera_basis = stored_rotation.transpose(1, 0)
            origin = torch.tensor(
                -stored_rotation @ translation,
                dtype=torch.float32,
                device=self.device,
            )
            basis = torch.tensor(
                camera_basis, dtype=torch.float32, device=self.device
            )
            alpha = pixels[..., 3:4]
            foreground = pixels[..., :3] * alpha
            focal = width / (2.0 * math.tan(0.5 * fov_x))
            fov_y = 2.0 * math.atan(height / (2.0 * focal))
            cameras.append(
                Camera(
                    name=Path(frame["file_path"]).name,
                    origin=origin,
                    right=basis[0],
                    down=basis[1],
                    forward=basis[2],
                    fov_x=float(np.float32(fov_x)),
                    fov_y=fov_y,
                    foreground=foreground,
                    alpha=alpha,
                )
            )
        return cameras

    def _validate_camera_shapes(self) -> None:
        expected = (self.height, self.width)
        for camera in self.train + self.test:
            if (camera.height, camera.width) != expected:
                raise ValueError("all cameras must have the same resolution")

    def stacked_training_data(self, pixels: bool = True) -> dict[str, torch.Tensor]:
        cameras = self.train
        data = {
            "origins": torch.stack([camera.origin for camera in cameras]),
            "rights": torch.stack([camera.right for camera in cameras]),
            "downs": torch.stack([camera.down for camera in cameras]),
            "forwards": torch.stack([camera.forward for camera in cameras]),
            "fov_x": torch.tensor(
                [camera.fov_x for camera in cameras],
                device=self.device,
                dtype=torch.float32,
            ),
            "fov_y": torch.tensor(
                [camera.fov_y for camera in cameras],
                device=self.device,
                dtype=torch.float32,
            ),
        }
        if pixels:
            data["foreground"] = torch.stack([camera.foreground for camera in cameras]).reshape(-1, 3)
            data["alpha"] = torch.stack([camera.alpha for camera in cameras]).reshape(-1, 1)
        return data

    @staticmethod
    def legacy_initial_point_cloud(
        count: int = 100_000,
    ) -> tuple[np.ndarray, np.ndarray]:
        xyz = np.random.random((count, 3)) * 2.6 - 1.3
        sh_values = np.random.random((count, 3)) / 255.0
        colors = sh_values * 0.28209479177387814 + 0.5
        return xyz, colors

    def initial_point_cloud(
        self, count: int = 100_000
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.legacy_initial_point_cloud(count)
