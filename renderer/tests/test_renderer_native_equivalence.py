from __future__ import annotations

import math
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F

from foamsurf import GaussianScene, HybridRenderer


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="native equivalence requires CUDA"
)


def _load_flare_native():
    renderer_dir = (
        Path(__file__).resolve().parents[1] / "foamsurf" / "_optix"
    )
    module = renderer_dir / "PYOPTIXFLARERENDERER.so"
    if not module.is_file():
        pytest.skip(f"bundled OptiX renderer is not built: {module}")
    sys.path.insert(0, str(renderer_dir))
    import PYOPTIXFLARERENDERER

    return PYOPTIXFLARERENDERER


def test_gaussian_forward_and_all_gradients_match_flare_base() -> None:
    native_module = _load_flare_native()
    device = torch.device("cuda")
    dtype = torch.float32
    origins = torch.tensor(
        [[0.20, 0.03, 0.0], [0.10, -0.15, 0.0]],
        dtype=dtype,
        device=device,
    )
    directions = F.normalize(
        torch.tensor(
            [[0.02, -0.01, 1.0], [-0.01, 0.03, 1.0]],
            dtype=dtype,
            device=device,
        ),
        dim=-1,
    )
    base_values = (
        torch.tensor([[0.01, -0.02, 2.0]], dtype=dtype, device=device),
        torch.tensor([[-1.2, -1.45]], dtype=dtype, device=device),
        torch.tensor(
            [[0.98, 0.08, -0.12, 0.05]], dtype=dtype, device=device
        ),
        torch.tensor([[0.7, 0.2, 0.4]], dtype=dtype, device=device),
        torch.tensor([[1.3]], dtype=dtype, device=device),
        torch.tensor([[0.2]], dtype=dtype, device=device),
    )
    means, log_scales, quaternions, colors, opacity_logits, raw_kappa = (
        base_values
    )
    native = native_module.CPyOptiXFLARERenderer(8, 11.3449, 2)
    native.SetGeometry(
        means,
        log_scales.exp(),
        quaternions,
        opacity_logits.sigmoid(),
        1.0 + F.softplus(raw_kappa),
    )
    background = (0.1, 0.05, 0.2)
    native_image = torch.zeros_like(origins)
    depth_accumulators = torch.zeros((2, 4), dtype=dtype, device=device)
    rgba = torch.cat((colors, opacity_logits), dim=-1).contiguous()
    native.Forward_training_base(
        origins,
        directions,
        native_image,
        *background,
        means,
        log_scales,
        quaternions,
        rgba,
        raw_kappa,
        0.0,
        depth_accumulators,
        0.0,
        1.0,
    )

    rays = torch.cat((origins, directions), dim=-1)
    oracle_image = HybridRenderer(far=100.0).render_gaussians(
        rays, GaussianScene(*base_values), background=background
    ).image
    torch.testing.assert_close(oracle_image, native_image, atol=2.0e-6, rtol=2.0e-6)

    upstream = torch.tensor(
        [[0.4, -0.2, 0.7], [-0.3, 0.8, 0.1]],
        dtype=dtype,
        device=device,
    )
    native_gradients = [
        torch.zeros_like(colors),
        torch.zeros_like(opacity_logits),
        torch.zeros_like(raw_kappa),
        torch.zeros_like(means),
        torch.zeros_like(log_scales),
        torch.zeros_like(quaternions),
    ]
    native.Backward_base(
        origins,
        directions,
        *background,
        means,
        log_scales,
        quaternions,
        rgba,
        raw_kappa,
        native_image,
        upstream,
        *native_gradients,
        0.0,
        depth_accumulators,
        torch.zeros_like(depth_accumulators),
        0.0,
        0.0,
        1.0,
    )
    differentiable = [value.detach().clone().requires_grad_() for value in base_values]
    oracle = HybridRenderer(far=100.0).render_gaussians(
        rays, GaussianScene(*differentiable), background=background
    )
    (oracle.image * upstream).sum().backward()

    native_by_parameter = (
        native_gradients[3],
        native_gradients[4],
        native_gradients[5],
        native_gradients[0],
        native_gradients[1],
        native_gradients[2],
    )
    for value, expected in zip(differentiable, native_by_parameter):
        torch.testing.assert_close(
            value.grad, expected, atol=3.0e-5, rtol=3.0e-4
        )


