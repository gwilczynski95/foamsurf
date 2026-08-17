from __future__ import annotations

import numpy as np
import pytest
import torch

from foamsurf import NativeHybridRenderer
from foamsurf_model.gaussian_adaptation import (
    GaussianAdaptationController,
    GaussianTopologyPlan,
)
from foamsurf_model.gaussians import SurfaceGaussianModel
from training.config import GaussianAdaptationConfig, ModelConfig


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="native B1 equivalence requires CUDA"
)
def test_native_renderer_is_exact_across_noop_adaptation() -> None:
    points = np.array(
        [[-0.2, 0.0, 2.0], [0.2, 0.0, 2.2]], dtype=np.float32
    )
    colors = np.array(
        [[0.8, 0.2, 0.1], [0.1, 0.5, 0.9]], dtype=np.float32
    )
    model = SurfaceGaussianModel(
        points,
        colors,
        ModelConfig(appearance="rgb", gaussian_initial_points=2),
        "cuda",
    )
    optimizer = torch.optim.Adam(
        [
            {
                "params": [parameter],
                "name": f"gaussian_{name}",
                "lr": 1.0e-3,
            }
            for name, parameter in model.named_primitive_parameters().items()
        ]
    )
    controller = GaussianAdaptationController(
        model,
        optimizer,
        GaussianAdaptationConfig(enabled=True, max_points=4),
        scene_extent=4.0,
    )
    renderer = NativeHybridRenderer(max_batch_size=2, far=4.0)
    rays = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
        device="cuda",
    )
    scene = model.render_scene(None)
    renderer.set_gaussian_geometry(scene)
    before = renderer.render_gaussians(rays, scene)
    indices = torch.arange(model.point_count, device="cuda")

    event = controller.apply_plan(
        GaussianTopologyPlan(indices, indices[:0], indices[:0], 2)
    )
    scene = model.render_scene(None)
    renderer.set_gaussian_geometry(scene)
    after = renderer.render_gaussians(rays, scene)

    assert event == {"clone": 0, "split": 0, "prune": 0, "budget_skip": 0}
    torch.testing.assert_close(after.image, before.image, rtol=0, atol=0)
    torch.testing.assert_close(after.alpha, before.alpha, rtol=0, atol=0)
