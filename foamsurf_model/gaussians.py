from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn
import torch.nn.functional as F

from foamsurf import GaussianScene
from training.config import ModelConfig
from .sh import evaluate_sh, rgb_to_sh_dc


def _inverse_softplus(value: float) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float64)
    return float((tensor + torch.log(-torch.expm1(-tensor))).item())


class SurfaceGaussianModel(nn.Module):
    """Trainable 2D Gaussian surface representation used by B0."""

    PARAMETER_NAMES = (
        "appearance",
        "opacity_logits",
        "raw_kappa",
        "means",
        "log_scales",
        "quaternions",
    )

    def __init__(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        config: ModelConfig,
        device: torch.device | str,
    ) -> None:
        super().__init__()
        self.appearance_mode = config.appearance
        self.sh_degree = config.sh_degree
        self.device = torch.device(device)

        points = np.asarray(points, dtype=np.float32)
        colors = np.asarray(colors, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Gaussian points must have shape [N, 3]")
        if colors.shape != points.shape:
            raise ValueError("Gaussian colors must have shape [N, 3]")
        if points.shape[0] == 0:
            raise ValueError("at least one Gaussian is required")

        means = torch.as_tensor(points, device=self.device)
        rgb = torch.as_tensor(colors, device=self.device)
        initial_opacity = torch.full(
            (points.shape[0], 1),
            config.gaussian_initial_opacity,
            dtype=torch.float32,
            device=self.device,
        )
        opacity_logits = torch.logit(initial_opacity)
        raw_kappa = torch.full_like(
            initial_opacity,
            _inverse_softplus(config.gaussian_initial_kappa - 1.0),
        )
        log_scales = torch.as_tensor(
            self._initial_log_scales(points),
            dtype=torch.float32,
            device=self.device,
        )
        quaternions = F.normalize(
            torch.randn(
                points.shape[0],
                4,
                dtype=torch.float32,
                device=self.device,
            ),
            dim=1,
        )

        if self.appearance_mode == "rgb":
            appearance = rgb
        else:
            coefficient_count = (self.sh_degree + 1) ** 2
            appearance = torch.zeros(
                points.shape[0],
                3,
                coefficient_count,
                dtype=torch.float32,
                device=self.device,
            )
            appearance[..., 0] = rgb_to_sh_dc(rgb)

        self.appearance = nn.Parameter(appearance)
        self.opacity_logits = nn.Parameter(opacity_logits)
        self.raw_kappa = nn.Parameter(raw_kappa)
        self.means = nn.Parameter(means)
        self.log_scales = nn.Parameter(log_scales)
        self.quaternions = nn.Parameter(quaternions)

    @staticmethod
    def _initial_log_scales(points: np.ndarray) -> np.ndarray:
        if points.shape[0] == 1:
            distance = np.full((1,), 1.0e-2, dtype=np.float32)
        else:
            tree = cKDTree(points)
            distances, _ = tree.query(points, k=2, workers=-1)
            distance = np.maximum(distances[:, 1], 1.0e-4).astype(
                np.float32
            )
        return np.log(distance)[:, None].repeat(2, axis=1)

    @property
    def point_count(self) -> int:
        return int(self.means.shape[0])

    @property
    def opacities(self) -> torch.Tensor:
        return torch.sigmoid(self.opacity_logits)

    @property
    def scales(self) -> torch.Tensor:
        return torch.exp(self.log_scales)

    @property
    def kappas(self) -> torch.Tensor:
        return 1.0 + F.softplus(self.raw_kappa)

    def colors(
        self, camera_origin: torch.Tensor | None
    ) -> torch.Tensor:
        if self.appearance_mode == "rgb":
            return self.appearance
        if camera_origin is None:
            raise ValueError("SH appearance requires a camera origin")
        directions = F.normalize(
            camera_origin.reshape(1, 3) - self.means, dim=-1
        )
        return 0.5 + evaluate_sh(
            self.sh_degree, self.appearance, directions
        )

    def render_scene(
        self, camera_origin: torch.Tensor | None
    ) -> GaussianScene:
        return GaussianScene(
            means=self.means,
            log_scales=self.log_scales,
            quaternions=F.normalize(self.quaternions, dim=-1),
            colors=self.colors(camera_origin),
            opacity_logits=self.opacity_logits,
            raw_kappa=self.raw_kappa,
        )

    def named_primitive_parameters(self) -> dict[str, nn.Parameter]:
        return {
            name: getattr(self, name) for name in self.PARAMETER_NAMES
        }

    def checkpoint_metadata(self) -> dict[str, object]:
        return {
            "appearance_mode": self.appearance_mode,
            "sh_degree": self.sh_degree,
            "point_count": self.point_count,
        }

    @classmethod
    def from_checkpoint(
        cls,
        state: Mapping[str, torch.Tensor],
        metadata: Mapping[str, object],
        device: torch.device | str,
    ) -> "SurfaceGaussianModel":
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        instance.appearance_mode = str(metadata["appearance_mode"])
        instance.sh_degree = int(metadata["sh_degree"])
        instance.device = torch.device(device)
        for name in cls.PARAMETER_NAMES:
            setattr(
                instance,
                name,
                nn.Parameter(state[name].to(instance.device)),
            )
        return instance
