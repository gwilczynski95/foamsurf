"""Style transfer helpers (style_transfer.py) on the CPU: no CLIP / VGG weights needed."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from style_transfer import DIRECTION_DELTA, batched_perspective_patches, crop_positions, directional_loss


def test_directional_loss_is_finite_at_the_base_render_and_flare_like_far_from_it():
    torch.manual_seed(0)
    direction = F.normalize(torch.randn(1, 512), dim=-1)
    source = torch.randn(1, 512, dtype=torch.float16)
    render = source.clone().float().requires_grad_(True)  # first step: render == base render
    loss = directional_loss(render, source, direction)
    loss.backward()
    assert math.isclose(float(loss), 1.0, abs_tol=1e-6)
    assert torch.isfinite(render.grad).all() and float(render.grad.abs().max()) < 1.0 / DIRECTION_DELTA
    far = source.float() + 100.0 * direction  # |step| >> DIRECTION_DELTA: plain 1 - cos, here ~0
    assert float(directional_loss(far, source, direction)) < 1e-4


def test_crops_respect_the_mask_coverage():
    torch.manual_seed(0)
    mask = torch.zeros(1, 1, 300, 400)
    mask[..., 150:, :200] = 1.0  # object in the lower-left quarter
    ys, xs = crop_positions(300, 400, 64, 32, mask, coverage=0.5)
    assert len(ys) == len(xs) == 32
    for y, x in zip(ys, xs):
        assert float(mask[0, 0, y:y + 64, x:x + 64].mean()) >= 0.5
    ys, xs = crop_positions(300, 400, 64, 32, None, coverage=0.5)  # no mask: uniform, inside the image
    assert all(0 <= y <= 300 - 64 for y in ys) and all(0 <= x <= 400 - 64 for x in xs)


def test_perspective_patches_shape_and_gradient():
    torch.manual_seed(0)
    image = torch.rand(1, 3, 120, 160, requires_grad=True)
    patches = batched_perspective_patches(image, 64, 8, out_size=96)
    assert patches.shape == (8, 3, 96, 96) and torch.isfinite(patches).all()
    assert float(patches.min()) >= 0.0 and float(patches.max()) <= 1.0 + 1e-5
    patches.mean().backward()
    assert float(image.grad.abs().sum()) > 0.0
