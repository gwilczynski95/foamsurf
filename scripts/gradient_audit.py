#!/usr/bin/env python3
"""Does the gradient of the native hybrid renderer match the renderer itself?

For every parameter group, a random direction v is drawn and the autograd directional derivative
<dL/dtheta, v> is compared with the central difference (L(theta + eps v) - L(theta - eps v)) / 2 eps
computed by the CUDA renderer itself (not by a reference implementation), for several eps.
L is a fixed random weighting of the rendered image (linear, so no kinks of its own).
Only `rows` random elements (rows of the parameter tensor) are perturbed at a time, so few
discontinuities are crossed (a Gaussian entering or leaving the 64-hit buffer, depth order swaps,
the kernel cutoff, cell boundaries). The central differences are "stable" when they agree across
eps (spread below 5%); only then is a mismatch with autograd evidence of a wrong backward.

Usage:
  python scripts/gradient_audit.py synthetic [--output audit.json]
      2000 Gaussians, 200 cells, 4096 rays, training settings (transmittance threshold 1e-4,
      far 1000); many rays fill the 64-hit buffer.
  python scripts/gradient_audit.py checkpoint --run outputs/seathru/<job>/<name> [--checkpoint last]
      the trained scene and the rays of a real test view.
"""
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EPS_SCALES = (1e-2, 3e-3, 1e-3, 3e-4)


def directional_check(loss, params, seed=0, eps_scales=EPS_SCALES, rows=None):
    """loss(params) -> float64 scalar tensor; params: name -> leaf tensor; rows: perturb only this
    many random rows of each tensor (None: all).

    Returns name -> dict(analytic, numeric per eps, relative error per eps, best relative error,
    spread of the numeric values across eps, stable).
    """
    values = {k: v.detach().clone().requires_grad_(True) for k, v in params.items()}
    total = loss(values)
    grads = torch.autograd.grad(total, list(values.values()), allow_unused=True)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    report = {}
    for (name, value), grad in zip(values.items(), grads):
        direction = torch.randn(value.shape, generator=generator, dtype=torch.float64)
        if rows is not None and rows < value.shape[0]:
            mask = torch.zeros(value.shape[0], dtype=torch.float64)
            mask[torch.randperm(value.shape[0], generator=generator)[:rows]] = 1.0
            direction = direction * mask.view(-1, *([1] * (value.dim() - 1)))
        direction = direction.to(value.device)
        analytic = 0.0 if grad is None else float((grad.double() * direction).sum())
        scale = float(value.detach().double().pow(2).mean().sqrt()) or 1.0
        numeric, errors = [], []
        for factor in eps_scales:
            eps = factor * scale
            with torch.no_grad():
                plus = {k: v.detach() for k, v in values.items()}
                minus = dict(plus)
                plus[name] = (value.detach().double() + eps * direction).to(value.dtype)
                minus[name] = (value.detach().double() - eps * direction).to(value.dtype)
                difference = float(loss(plus) - loss(minus)) / (2 * eps)
            numeric.append(difference)
            errors.append(abs(difference - analytic) / max(abs(analytic), abs(difference), 1e-12))
        middle = sorted(numeric)[len(numeric) // 2]
        spread = (max(numeric) - min(numeric)) / max(abs(middle), 1e-12)
        report[name] = dict(analytic=analytic, numeric=numeric, relative_error=errors, best=min(errors),
                            spread=spread, stable=spread < 0.05)
    return report


def weighted_image_loss(weights):
    return lambda image: (image.double() * weights).sum()


def complete_adjacency(count, device):
    ids = torch.arange(count, device=device)
    adjacency = ids.repeat(count, 1)
    adjacency = adjacency[adjacency != ids[:, None]].to(torch.uint32).contiguous()
    offsets = (torch.arange(count + 1, device=device) * (count - 1)).to(torch.uint32)
    return adjacency, offsets


SCENES = {"moderate": dict(gaussians=600, log_scale=(-3.4, -2.8)), "dense": dict(gaussians=2000, log_scale=(-2.5, -1.5))}


def synthetic_audit(seed=0, scene="moderate", cells=200, rays=4096, rows=16):
    from foamsurf import GaussianScene, NativeHybridRenderer
    from foamsurf.types import RGBVoronoiScene
    torch.manual_seed(seed)
    device = "cuda"
    gaussians, (low, high) = SCENES[scene]["gaussians"], SCENES[scene]["log_scale"]
    depth = torch.rand(gaussians, device=device) * 2.5 + 1.5
    xy = (torch.rand(gaussians, 2, device=device) * 1.2 - 0.6) * depth[:, None] * 0.35
    params = dict(
        means=torch.cat((xy, depth[:, None]), -1),
        log_scales=torch.rand(gaussians, 2, device=device) * (high - low) + low,
        quaternions=torch.randn(gaussians, 4, device=device),
        gaussian_colors=torch.rand(gaussians, 3, device=device),
        opacity_logits=torch.randn(gaussians, 1, device=device) + 0.5,
        raw_kappa=torch.randn(gaussians, 1, device=device) * 0.3,
        sites=torch.rand(cells, 3, device=device) * torch.tensor([6.0, 6.0, 7.0], device=device)
        - torch.tensor([3.0, 3.0, 1.0], device=device),
        attenuation=torch.rand(cells, 3, device=device) * 0.3 + 0.01,
        backscatter=torch.rand(cells, 3, device=device) * 0.2 + 0.01,
        medium_colors=torch.rand(cells, 3, device=device),
    )
    directions = torch.randn(rays, 3, device=device) * torch.tensor([0.12, 0.12, 0.0], device=device)
    directions[:, 2] = 1.0
    ray_tensor = torch.cat((torch.zeros_like(directions), directions), -1).contiguous()
    weights = torch.randn(rays, 3, device=device, dtype=torch.float64,
                          generator=torch.Generator(device=device).manual_seed(seed + 1))
    renderer = NativeHybridRenderer(max_batch_size=rays, near=0.0, far=1000.0, transmittance_threshold=1e-4)
    adjacency, offsets = complete_adjacency(cells, device)
    state = {}

    def render(p):
        g = GaussianScene(p["means"], p["log_scales"], p["quaternions"], p["gaussian_colors"],
                          p["opacity_logits"], p["raw_kappa"])
        m = RGBVoronoiScene(p["sites"], p["attenuation"].mean(-1, keepdim=True), p["medium_colors"],
                            p["attenuation"], p["backscatter"])
        renderer.set_voronoi_geometry(m, adjacency, offsets)
        renderer.set_gaussian_geometry(g)
        out = renderer.render(ray_tensor, g, m, background=(0.0, 0.0, 0.0))
        state["hits"] = out.hit_indices
        return out.image

    loss = weighted_image_loss(weights)
    report = directional_check(lambda p: loss(render(p)), params, seed=seed, rows=rows)
    return report, hit_saturation(state["hits"], renderer.gaussian_hit_buffer_size)


def hit_saturation(hit_indices, size):
    counts = (hit_indices >= 0).sum(0)
    return dict(buffer=size, full_fraction=float((counts >= size).float().mean()),
                mean_hits=float(counts.float().mean()))


def checkpoint_audit(run, checkpoint="last", rays=32768, view=0, seed=0, rows=16):
    from evaluation.evaluator import _rays
    from foamsurf import GaussianScene
    from training.config import ExperimentConfig
    from training.trainer import Trainer
    joint = Path(run).resolve() / "joint"
    config = ExperimentConfig.load(joint / "config.yaml")
    config.output_dir = str(Path(run).resolve() / "audit" / f"_trainer_gradient_{checkpoint}")
    config.wandb.mode = "disabled"
    trainer = Trainer(config, resume=joint / "checkpoints" / f"{checkpoint}.pt")
    model, renderer = trainer.model, trainer.renderer
    renderer.transmittance_threshold = config.renderer.training_transmittance_threshold
    camera = trainer.dataset.test[view]
    all_rays = _rays(camera).reshape(-1, 6)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    picked = torch.randperm(all_rays.shape[0], generator=generator)[:rays].to(all_rays.device)
    ray_tensor = all_rays[picked].contiguous()
    with torch.no_grad():
        g0 = model.gaussians.render_scene(camera.origin)
        m0 = model.medium.render_scene()
    trainer.topology.synchronize_renderer(renderer, model.medium)
    params = dict(means=g0.means, log_scales=g0.log_scales, quaternions=g0.quaternions, gaussian_colors=g0.colors,
                  opacity_logits=g0.opacity_logits, raw_kappa=g0.raw_kappa,
                  attenuation=m0.attenuation, backscatter=m0.backscatter, medium_colors=m0.colors)
    weights = torch.randn(rays, 3, device=ray_tensor.device, dtype=torch.float64,
                          generator=torch.Generator(device=ray_tensor.device).manual_seed(seed + 1))
    state = {}

    def render(p):
        g = GaussianScene(p["means"], p["log_scales"], p["quaternions"], p["gaussian_colors"],
                          p["opacity_logits"], p["raw_kappa"])
        m = replace(m0, densities=p["attenuation"].mean(-1, keepdim=True), attenuation=p["attenuation"],
                    backscatter=p["backscatter"], colors=p["medium_colors"])
        renderer.set_gaussian_geometry(g)
        out = renderer.render(ray_tensor, g, m, background=tuple(config.background))
        state["hits"] = out.hit_indices
        return out.image

    loss = weighted_image_loss(weights)
    report = directional_check(lambda p: loss(render(p)), params, seed=seed, rows=rows)
    return report, hit_saturation(state["hits"], config.renderer.gaussian_hit_buffer_size), camera.name


def print_report(report):
    print(f"{'parametry':16s} {'autograd':>12s} " + " ".join(f"{'FD eps=' + format(e, 'g'):>14s}" for e in EPS_SCALES)
          + f" {'najl. błąd':>10s} {'rozrzut FD':>10s}")
    for name, row in report.items():
        print(f"{name:16s} {row['analytic']:12.4e} " + " ".join(f"{v:14.4e}" for v in row["numeric"])
              + f" {100 * row['best']:9.2f}% {100 * row['spread']:9.1f}%{'' if row['stable'] else '  (FD niestabilne)'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("synthetic", "checkpoint"))
    parser.add_argument("--scene", choices=tuple(SCENES), default="moderate")
    parser.add_argument("--rows", type=int, default=16, help="perturbed rows per tensor")
    parser.add_argument("--run", type=Path)
    parser.add_argument("--checkpoint", default="last")
    parser.add_argument("--rays", type=int, default=32768)
    parser.add_argument("--view", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "synthetic":
        report, saturation = synthetic_audit(scene=args.scene, rows=args.rows)
        extra = dict(scene=args.scene)
    else:
        if args.run is None:
            parser.error("checkpoint mode needs --run")
        report, saturation, view = checkpoint_audit(args.run, args.checkpoint, args.rays, args.view, rows=args.rows)
        extra = dict(run=str(args.run), checkpoint=args.checkpoint, view=view)
    print(json.dumps(dict(saturation=saturation, **extra)))
    print_report(report)
    if args.output:
        args.output.write_text(json.dumps(dict(report=report, saturation=saturation, **extra), indent=1))
    stable = [row["best"] for row in report.values() if row["stable"]]
    worst = max(stable) if stable else float("nan")
    print(f"najgorsza grupa ze stabilnymi FD: {100 * worst:.2f}% błędu względnego; "
          f"niestabilne FD: {[n for n, r in report.items() if not r['stable']]}")


if __name__ == "__main__":
    main()
