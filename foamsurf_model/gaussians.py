from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn
import torch.nn.functional as F

from foamsurf import GaussianScene
from training.config import ModelConfig
from .sh import C0, evaluate_sh, rgb_to_sh_dc


def _inverse_softplus(value: float) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float64)
    return float((tensor + torch.log(-torch.expm1(-tensor))).item())


class SurfaceGaussianModel(nn.Module):
    """Trainable 2D Gaussian surface representation used by B0."""

    PARAMETER_NAMES = (
        "appearance_dc",
        "appearance_rest",
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
        self.rgb_activation = config.gaussian_rgb_activation
        self.sh_degree = config.sh_degree
        self.active_sh_degree = config.sh_degree
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
            self._initial_log_scales(
                points, config.gaussian_initial_scale_neighbors
            ),
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
            appearance_dc = (torch.logit(rgb.clamp(1.0e-5, 1.0 - 1.0e-5))
                             if self.rgb_activation == "sigmoid" else rgb)
            appearance_rest = torch.empty(
                points.shape[0], 3, 0, device=self.device
            )
        else:
            coefficient_count = (self.sh_degree + 1) ** 2
            if self.rgb_activation == "sigmoid":
                # sigmoid(C0 * dc) = rgb when the higher coefficients are zero.
                appearance_dc = (torch.logit(rgb.clamp(1.0e-5, 1.0 - 1.0e-5)) / C0)[..., None]
            else:
                appearance_dc = rgb_to_sh_dc(rgb)[..., None]
            appearance_rest = torch.zeros(
                points.shape[0], 3, coefficient_count - 1,
                dtype=torch.float32,
                device=self.device,
            )

        self.appearance_dc = nn.Parameter(appearance_dc)
        self.appearance_rest = nn.Parameter(appearance_rest)
        self.opacity_logits = nn.Parameter(opacity_logits)
        self.raw_kappa = nn.Parameter(raw_kappa)
        self.means = nn.Parameter(means)
        self.log_scales = nn.Parameter(log_scales)
        self.quaternions = nn.Parameter(quaternions)

    @staticmethod
    def _initial_log_scales(
        points: np.ndarray, neighbor_count: int = 1
    ) -> np.ndarray:
        if points.shape[0] == 1:
            distance = np.full((1,), 1.0e-2, dtype=np.float32)
        else:
            neighbor_count = min(int(neighbor_count), points.shape[0] - 1)
            tree = cKDTree(points)
            distances, _ = tree.query(
                points, k=neighbor_count + 1, workers=-1
            )
            neighbor_distances = distances[:, 1:]
            if neighbor_distances.ndim == 1:
                neighbor_distances = neighbor_distances[:, None]
            distance = np.sqrt(
                np.mean(neighbor_distances**2, axis=1)
            )
            distance = np.maximum(distance, 1.0e-4).astype(np.float32)
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

    @property
    def appearance(self) -> torch.Tensor:
        if self.appearance_mode == "rgb":
            return self.appearance_dc
        coefficient_count = (self.active_sh_degree + 1) ** 2
        return torch.cat(
            (self.appearance_dc, self.appearance_rest), dim=-1
        )[..., :coefficient_count]

    def colors(
        self, camera_origin: torch.Tensor | None
    ) -> torch.Tensor:
        if self.appearance_mode == "rgb":
            return (torch.sigmoid(self.appearance) if self.rgb_activation == "sigmoid"
                    else self.appearance)
        if camera_origin is None:
            raise ValueError("SH appearance requires a camera origin")
        directions = F.normalize(
            camera_origin.reshape(1, 3) - self.means, dim=-1
        )
        value = evaluate_sh(self.active_sh_degree, self.appearance, directions)
        # sigmoid: SH gives view-dependent logits, colours stay in (0, 1).
        return torch.sigmoid(value) if self.rgb_activation == "sigmoid" else 0.5 + value

    def render_scene(
        self,
        camera_origin: torch.Tensor | None,
        means: torch.Tensor | None = None,
    ) -> GaussianScene:
        return GaussianScene(
            means=self.means if means is None else means,
            log_scales=self.log_scales,
            quaternions=F.normalize(self.quaternions, dim=-1),
            colors=self.colors(camera_origin),
            opacity_logits=self.opacity_logits,
            raw_kappa=self.raw_kappa,
        )

    def training_scene(
        self, camera
    ) -> tuple[GaussianScene, torch.Tensor]:
        """Create the normalized-viewport gradient carrier used by 3DGS."""

        screen_points = torch.zeros(
            (self.point_count, 2),
            dtype=self.means.dtype,
            device=self.means.device,
            requires_grad=True,
        )
        world_per_viewport = self.world_per_viewport(camera)
        offsets = (
            screen_points[:, :1]
            * world_per_viewport[:, :1]
            * camera.right[None]
            + screen_points[:, 1:]
            * world_per_viewport[:, 1:]
            * camera.down[None]
        )
        return (
            self.render_scene(camera.origin, self.means + offsets),
            screen_points,
        )

    @torch.no_grad()
    def world_per_viewport(self, camera) -> torch.Tensor:
        """[N,2] world offset along camera right/down per unit of screen_points."""
        depth = ((self.means - camera.origin) * camera.forward).sum(dim=-1)
        half_fov = self.means.new_tensor((0.5 * camera.fov_x, 0.5 * camera.fov_y))
        return depth[:, None] * torch.tan(half_fov)[None]

    def named_primitive_parameters(self) -> dict[str, nn.Parameter]:
        return {
            name: getattr(self, name) for name in self.PARAMETER_NAMES
        }

    def checkpoint_metadata(self) -> dict[str, object]:
        return {
            "appearance_mode": self.appearance_mode,
            "rgb_activation": self.rgb_activation,
            "sh_degree": self.sh_degree,
            "active_sh_degree": self.active_sh_degree,
            "point_count": self.point_count,
        }

    def one_up_sh_degree(self) -> None:
        if self.active_sh_degree < self.sh_degree:
            self.active_sh_degree += 1

    @torch.no_grad()
    def clamp_scales_(self, minimum: float, maximum: float) -> None:
        if maximum < minimum:
            raise ValueError("maximum Gaussian scale must exceed minimum")
        self.log_scales.clamp_(min=np.log(minimum), max=np.log(maximum))

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
        instance.rgb_activation = str(metadata.get("rgb_activation", "identity"))
        instance.sh_degree = int(metadata["sh_degree"])
        instance.active_sh_degree = int(
            metadata.get("active_sh_degree", instance.sh_degree)
        )
        instance.device = torch.device(device)
        for name in cls.PARAMETER_NAMES:
            setattr(
                instance,
                name,
                nn.Parameter(state[name].to(instance.device)),
            )
        return instance
