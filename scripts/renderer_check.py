#!/usr/bin/env python3
"""Renderer equivalence and speed on a real IUI3 checkpoint.

  capture  render the test views and one training-image gradient with the
           CURRENT renderer and save them as the reference (.pt)
  compare  render the same with the renderer now installed and compare with
           a reference; exit code 1 when a tolerance is exceeded
  bench    time evaluation renders (with a breakdown) and a training
           forward/backward; --json writes the numbers

  python scripts/renderer_check.py capture --run RUN --dataset DATA --reference ref.pt
  python scripts/renderer_check.py compare --run RUN --dataset DATA --reference ref.pt
  python scripts/renderer_check.py bench   --run RUN --dataset DATA [--json bench.json]

RUN is a pipeline output directory (contains joint/config.yaml and
joint/checkpoints/<checkpoint>.pt); DATA overrides dataset.source_path stored in
the run (it was written on another machine).

Snapshot contents. Test views (validation transmittance threshold, background of
the run, exactly as evaluation/evaluator.py): hybrid RGB, backscatter RGB, alpha,
Gaussian-only alpha on black, Gaussian hit count per ray. Gradient: first
training camera, full image, training threshold,
loss = L1(hybrid, target) + 0.1 * mean(backscatter) + 0.1 * mean(Gaussian-only
alpha) (training priors differentiate the backscatter output and Gaussian-only
alpha separately), gradients of every model parameter.
"""
import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.evaluator import _rays  # noqa: E402
from training.config import ExperimentConfig  # noqa: E402

# A rewritten renderer may reorder floating-point operations, so values match
# closely but not bitwise. Gaussian hits are capped at K=64 per ray today: a
# renderer without that cap changes the rays where the cap is reached (reported
# as hit_buffer_full_fraction) and fails these tolerances there.
IMAGE_MEAN_ABS = 1.0e-5
IMAGE_OUTLIER_ABS = 1.0e-3
IMAGE_OUTLIER_FRACTION = 1.0e-4
GRADIENT_RELATIVE_L2 = 1.0e-3


def load(args):
    run = args.run.resolve()
    config = ExperimentConfig.load(run / "joint" / "config.yaml")
    if args.dataset is not None:
        config.dataset.source_path = str(args.dataset.resolve())
    work = Path(tempfile.mkdtemp(prefix="renderer_check_"))
    config.output_dir = str(work)
    config.wandb.mode = "disabled"
    from training.trainer import Trainer
    checkpoint = run / "joint" / "checkpoints" / f"{args.checkpoint}.pt"
    trainer = Trainer(config, resume=checkpoint)
    trainer.topology.synchronize_renderer(trainer.renderer, trainer.model.medium)
    return config, trainer


@torch.no_grad()
def render_views(config, trainer):
    renderer, model = trainer.renderer, trainer.model
    renderer.transmittance_threshold = config.renderer.validation_transmittance_threshold
    medium = model.medium.render_scene()
    background = tuple(config.background)
    views = {}
    for camera in trainer.dataset.test:
        scene = model.gaussians.render_scene(camera.origin)
        renderer.set_gaussian_geometry(scene)
        rays = _rays(camera)
        hybrid = renderer.render(rays, gaussians=scene, voronoi=medium, background=background)
        only = renderer.render_gaussians(rays, scene, background=(0.0, 0.0, 0.0))
        views[camera.name] = dict(
            image=hybrid.image.cpu(), alpha=hybrid.alpha.cpu(),
            backscatter=(torch.zeros_like(hybrid.image) if hybrid.backscatter is None else hybrid.backscatter).cpu(),
            gaussian_alpha=only.alpha.cpu(),
            hit_count=(hybrid.hit_indices >= 0).sum(0).to(torch.int16).reshape(hybrid.alpha.shape[:-1]).cpu(),
            hit_buffer_size=int(hybrid.hit_indices.size(0)))
    return views


def gradients(config, trainer):
    renderer, model = trainer.renderer, trainer.model
    renderer.transmittance_threshold = config.renderer.training_transmittance_threshold
    camera = trainer.dataset.train[0]
    background = tuple(config.background)
    model.zero_grad(set_to_none=True)
    scene = model.gaussians.render_scene(camera.origin)
    renderer.set_gaussian_geometry(scene)
    rays = _rays(camera)
    hybrid = renderer.render(rays, gaussians=scene, voronoi=model.medium.render_scene(), background=background)
    only = renderer.render_gaussians(rays, scene, background=(0.0, 0.0, 0.0))
    loss = (hybrid.image - camera.ground_truth(background)).abs().mean() + 0.1 * only.alpha.mean()
    if hybrid.backscatter is not None:
        loss = loss + 0.1 * hybrid.backscatter.mean()
    loss.backward()
    grads = {name: parameter.grad.detach().cpu().clone() for name, parameter in model.named_parameters()
             if parameter.grad is not None}
    model.zero_grad(set_to_none=True)
    return dict(camera=camera.name, loss=float(loss), grads=grads)


def snapshot(config, trainer, checkpoint):
    return dict(checkpoint=checkpoint, views=render_views(config, trainer), gradient=gradients(config, trainer),
                gaussians=int(trainer.model.gaussians.point_count), cells=int(trainer.model.medium.point_count))


def compare_images(reference, actual):
    """Rows per view and quantity; ok=False when a tolerance is exceeded."""
    rows = []
    for view, expected in reference["views"].items():
        got = actual["views"][view]
        full = float((expected["hit_count"] >= expected["hit_buffer_size"]).float().mean())
        for key in ("image", "backscatter", "alpha", "gaussian_alpha"):
            difference = (got[key].float() - expected[key].float()).abs()
            mean, outliers = float(difference.mean()), float((difference > IMAGE_OUTLIER_ABS).float().mean())
            mse = float(difference.square().mean())
            rows.append(dict(view=view, quantity=key, mean_abs=mean, max_abs=float(difference.max()),
                             outlier_fraction=outliers, psnr=math.inf if mse == 0 else -10 * math.log10(mse),
                             hit_buffer_full_fraction=full,
                             ok=mean <= IMAGE_MEAN_ABS and outliers <= IMAGE_OUTLIER_FRACTION))
    return rows


def compare_gradients(reference, actual):
    rows = []
    expected, got = reference["gradient"]["grads"], actual["gradient"]["grads"]
    for name in sorted(set(expected) | set(got)):
        if name not in expected or name not in got or expected[name].shape != got[name].shape:
            rows.append(dict(name=name, relative_l2=math.inf, cosine=float("nan"), ok=False))
            continue
        e, g = expected[name].double().flatten(), got[name].double().flatten()
        norm = float(e.norm())
        relative = float((g - e).norm()) / norm if norm > 0 else float((g - e).norm())
        cosine = float(torch.nn.functional.cosine_similarity(e, g, dim=0)) if norm > 0 else float("nan")
        rows.append(dict(name=name, relative_l2=relative, cosine=cosine, ok=relative <= GRADIENT_RELATIVE_L2))
    return rows


def report(image_rows, gradient_rows):
    print(f"{'view':<10} {'quantity':<15} {'mean|d|':>10} {'max|d|':>10} {'>1e-3':>9} {'PSNR':>7} {'K full':>7}")
    for row in image_rows:
        print(f"{row['view']:<10} {row['quantity']:<15} {row['mean_abs']:10.2e} {row['max_abs']:10.2e} "
              f"{row['outlier_fraction']:9.2e} {row['psnr']:7.1f} {row['hit_buffer_full_fraction']:7.4f}"
              f"{'' if row['ok'] else '  FAIL'}")
    print(f"\n{'parameter':<28} {'rel L2':>10} {'cosine':>10}")
    for row in gradient_rows:
        print(f"{row['name']:<28} {row['relative_l2']:10.2e} {row['cosine']:10.6f}{'' if row['ok'] else '  FAIL'}")
    ok = all(row["ok"] for row in image_rows + gradient_rows)
    print("\nPASS" if ok else "\nFAIL", f"(image mean|d| <= {IMAGE_MEAN_ABS}, |d| > {IMAGE_OUTLIER_ABS} on <= "
          f"{IMAGE_OUTLIER_FRACTION} of pixels, gradient relative L2 <= {GRADIENT_RELATIVE_L2})")
    return ok


def timed(function, warmup, repeats):
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(repeats):
        function()
    end.record()
    torch.cuda.synchronize()
    return begin.elapsed_time(end) / repeats


def bench(config, trainer, warmup, repeats):
    renderer, model = trainer.renderer, trainer.model
    background = tuple(config.background)
    medium = model.medium.render_scene()
    rows = []
    with torch.no_grad():
        renderer.transmittance_threshold = config.renderer.validation_transmittance_threshold
        for camera in trainer.dataset.test:
            scene = model.gaussians.render_scene(camera.origin)
            renderer.set_gaussian_geometry(scene)
            rays = _rays(camera).reshape(-1, 6).contiguous()
            origins, directions = rays[:, :3].contiguous(), torch.nn.functional.normalize(rays[:, 3:], dim=-1)
            unique, inverse = torch.unique(origins, dim=0, return_inverse=True)
            starts = renderer._nearest_sites(medium.sites, unique)[inverse].to(torch.uint32)
            near = torch.full((rays.size(0),), renderer.near, device=rays.device)
            row = dict(view=camera.name, rays=int(rays.size(0)))
            row["hybrid_ms"] = timed(lambda: renderer.render(rays, gaussians=scene, voronoi=medium,
                                                             background=background), warmup, repeats)
            row["hybrid_given_starts_ms"] = timed(lambda: renderer.render(
                rays, gaussians=scene, voronoi=medium, background=background, start_points=starts), warmup, repeats)
            row["optix_hits_ms"] = timed(lambda: renderer._optix.Render(
                origins, directions.contiguous(), near, rays.size(0)), warmup, repeats)
            row["gaussians_only_ms"] = timed(lambda: renderer.render_gaussians(rays, scene), warmup, repeats)
            row["medium_only_ms"] = timed(lambda: renderer.render_voronoi(rays, medium, start_points=starts),
                                          warmup, repeats)
            row["start_cell_lookup_ms"] = row["hybrid_ms"] - row["hybrid_given_starts_ms"]
            row["composition_ms"] = row["hybrid_given_starts_ms"] - row["optix_hits_ms"]
            rows.append(row)
    renderer.transmittance_threshold = config.renderer.training_transmittance_threshold
    camera = trainer.dataset.train[0]
    rays = _rays(camera)
    target = camera.ground_truth(background)

    def forward():
        scene = model.gaussians.render_scene(camera.origin)
        renderer.set_gaussian_geometry(scene)
        image = renderer.render(rays, gaussians=scene, voronoi=model.medium.render_scene(),
                                background=background).image
        return (image - target).abs().mean()

    def step():
        model.zero_grad(set_to_none=True)
        forward().backward()

    torch.cuda.reset_peak_memory_stats()
    training = dict(forward_ms=timed(forward, warmup, repeats), forward_backward_ms=timed(step, warmup, repeats),
                    peak_memory_gib=torch.cuda.max_memory_allocated() / 2**30)
    model.zero_grad(set_to_none=True)
    return dict(device=torch.cuda.get_device_name(), gaussians=int(model.gaussians.point_count),
                cells=int(model.medium.point_count), hit_buffer_size=renderer.gaussian_hit_buffer_size,
                views=rows, training=training)


def print_bench(result):
    print(f"{result['device']}: {result['gaussians']} Gaussians, {result['cells']} cells, K={result['hit_buffer_size']}")
    keys = ("hybrid_ms", "start_cell_lookup_ms", "optix_hits_ms", "composition_ms", "gaussians_only_ms",
            "medium_only_ms")
    print(f"{'view':<10}" + "".join(f"{key[:-3]:>22}" for key in keys))
    for row in result["views"]:
        print(f"{row['view']:<10}" + "".join(f"{row[key]:22.2f}" for key in keys))
    mean = sum(row["hybrid_ms"] for row in result["views"]) / len(result["views"])
    print(f"evaluation hybrid: {mean:.2f} ms = {1000 / mean:.1f} FPS (evaluator's eval/hybrid_fps)")
    training = result["training"]
    print(f"training image: forward {training['forward_ms']:.2f} ms, forward+backward "
          f"{training['forward_backward_ms']:.2f} ms, peak {training['peak_memory_gib']:.1f} GiB")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("capture", "compare", "bench"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, help="IUI3 dataset root on this machine")
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--reference", type=Path, help="snapshot file (capture writes, compare reads)")
    parser.add_argument("--json", type=Path, help="bench: also write the numbers here")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if args.mode in ("capture", "compare") and args.reference is None:
        parser.error("--reference is required")
    config, trainer = load(args)
    if args.mode == "bench":
        result = bench(config, trainer, args.warmup, args.repeats)
        print_bench(result)
        if args.json:
            args.json.write_text(json.dumps(result, indent=2))
        return
    current = snapshot(config, trainer, args.checkpoint)
    if args.mode == "capture":
        args.reference.parent.mkdir(parents=True, exist_ok=True)
        torch.save(current, args.reference)
        print(f"saved {args.reference}: {len(current['views'])} views, {len(current['gradient']['grads'])} gradients, "
              f"loss {current['gradient']['loss']:.6f}")
        return
    reference = torch.load(args.reference, map_location="cpu", weights_only=False)
    if (reference["gaussians"], reference["cells"]) != (current["gaussians"], current["cells"]):
        sys.exit("reference was captured from a different checkpoint")
    ok = report(compare_images(reference, current), compare_gradients(reference, current))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
