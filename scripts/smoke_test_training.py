#!/usr/bin/env python3
"""Run one synthetic RadFoam training iteration without a dataset."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F


def stage(message: str) -> None:
    print(f"[radfoam-train-smoke] {message}", flush=True)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    project_dir = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_dir))

    import radfoam
    from radfoam_model.scene import RadFoamScene

    require(torch.cuda.is_available(), "CUDA is not available to PyTorch")
    require(torch.version.cuda == "12.4", f"Expected torch CUDA 12.4, got {torch.version.cuda}")
    device = torch.device("cuda:0")
    require(
        torch.cuda.get_device_capability(device) == (8, 9),
        f"Expected RTX 4090 capability (8, 9), got {torch.cuda.get_device_capability(device)}",
    )
    stage(f"PyTorch {torch.__version__}; GPU {torch.cuda.get_device_name(device)}")
    stage(f"native module: {Path(radfoam.__file__).resolve()}")

    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    model_args = SimpleNamespace(
        sh_degree=1,
        init_points=128,
        final_points=256,
        activation_scale=1.0,
    )
    optimizer_args = SimpleNamespace(
        points_lr_init=2.0e-4,
        points_lr_final=5.0e-6,
        density_lr_init=1.0e-1,
        density_lr_final=1.0e-2,
        attributes_lr_init=5.0e-3,
        attributes_lr_final=5.0e-4,
        sh_factor=0.1,
        freeze_points=10,
    )

    model = RadFoamScene(model_args, device=device)
    model.declare_optimizer(optimizer_args, warmup=1, max_iterations=10)
    require(model.primal_points.shape == (128, 3), "Unexpected point tensor shape")
    require(model.point_adjacency.numel() > 0, "Triangulation has no adjacency")
    stage("Delaunay triangulation/AABB/pipeline creation: OK")

    origins = torch.zeros((64, 3), device=device)
    directions = torch.randn((64, 3), device=device)
    directions[:, 2].abs_().add_(0.1)
    directions = F.normalize(directions, dim=-1)
    rays = torch.cat((origins, directions), dim=-1)

    points_before = model.primal_points.detach().clone()
    rgba, _, _, num_intersections, _ = model(rays)
    torch.cuda.synchronize()
    require(rgba.shape == (64, 4), f"Unexpected renderer output shape: {rgba.shape}")
    require(bool(torch.isfinite(rgba).all()), "Forward returned NaN or Inf")
    require(num_intersections.float().mean().item() > 0.0, "No ray intersections")
    stage(
        "native ray-tracing forward: OK; "
        f"mean intersections={num_intersections.float().mean().item():.2f}"
    )

    target = torch.rand_like(rgba)
    loss = F.smooth_l1_loss(rgba, target)
    model.optimizer.zero_grad(set_to_none=True)
    loss.backward()

    for name, parameter in model.named_parameters():
        require(parameter.grad is not None, f"Missing gradient for {name}")
        require(bool(torch.isfinite(parameter.grad).all()), f"Non-finite gradient for {name}")
    require(model.primal_points.grad.abs().max().item() > 0.0, "Point gradient is zero")
    require(model.density.grad.abs().max().item() > 0.0, "Density gradient is zero")
    stage("custom autograd trace backward: OK")

    model.optimizer.step()
    torch.cuda.synchronize()
    require(
        not torch.equal(points_before, model.primal_points.detach()),
        "Adam did not update primal points",
    )
    stage(f"Adam step: OK; synthetic loss={loss.item():.6f}")
    stage("PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(
            f"[radfoam-train-smoke] FAIL: {type(error).__name__}: {error}",
            file=sys.stderr,
            flush=True,
        )
        raise
