from __future__ import annotations

import numpy as np
import torch
from torch import nn

from training.config import ExperimentConfig
from .gaussians import SurfaceGaussianModel
from .medium import VoronoiMediumModel, scene_bounds


class HybridSceneModel(nn.Module):
    """Container for the B0 surface and medium representations."""

    def __init__(
        self,
        gaussians: SurfaceGaussianModel,
        medium: VoronoiMediumModel,
    ) -> None:
        super().__init__()
        self.gaussians = gaussians
        self.medium = medium

    @classmethod
    def initialize(
        cls,
        dataset,
        config: ExperimentConfig,
        device: torch.device,
    ) -> "HybridSceneModel":
        if config.radfoam_prior.enabled:
            from training.radfoam_handoff import initialize_from_radfoam_prior

            initialized = initialize_from_radfoam_prior(config, device)
            initialized.model.initialization_report = initialized.report
            return initialized.model

        points, colors = dataset.initial_point_cloud(
            config.model.gaussian_initial_points
        )
        points = np.asarray(points, dtype=np.float32)
        colors = np.asarray(colors, dtype=np.float32)
        if (
            dataset.__class__.__name__ != "BlenderDataset"
            and config.model.gaussian_initial_points > 0
            and points.shape[0] > config.model.gaussian_initial_points
        ):
            selection = np.random.choice(
                points.shape[0],
                config.model.gaussian_initial_points,
                replace=False,
            )
            points = points[selection]
            colors = colors[selection]

        camera_origins = torch.stack(
            [camera.origin for camera in dataset.train + dataset.test]
        )
        lower, upper = scene_bounds(
            points, camera_origins, config.model
        )
        gaussians = SurfaceGaussianModel(
            points, colors, config.model, device
        )
        medium = VoronoiMediumModel(
            lower, upper, config.model, device
        )
        return cls(gaussians, medium)

    def checkpoint_metadata(self) -> dict[str, object]:
        return {
            "gaussians": self.gaussians.checkpoint_metadata(),
            "medium": self.medium.checkpoint_metadata(),
        }
