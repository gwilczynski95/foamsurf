#!/usr/bin/env python3
"""Surface/medium disentanglement audit of one IUI3 joint checkpoint.

Usage: python scripts/medium_audit.py --run outputs/<sweep>/<job>/<variant> [--checkpoint best]

Writes <run>/audit/<checkpoint>.json (or --output-dir). Metrics (test views, every STRIDE-th pixel):
- bottom coverage: Gaussian-only alpha on the lower 40% of the frame (seabed in IUI3);
- behind-surface share: fraction of backscatter produced after Gaussian T_g < 0.5;
- cell ablation: PSNR drop when Voronoi cells within 5 units of the Gaussians are
  replaced by open-water parameters (median of cells farther than 10 units);
- medium statistics near/far from the surface;
- backscatter hue angle on seabed rays vs the open-water backscatter direction;
- open water taken by Gaussians: Gaussian-only alpha on the frozen open-water mask
  (configs/iui3_open_water.npz, see scripts/open_water_mask.py);
- opaque seabed (alpha > 0.99 on the lower 40%) in the central half of the width vs the borders.
Backscatter and the ablation use an independent ray-marcher (nearest-site lookup)
validated against the native renderer; the validation error is reported.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.config import ExperimentConfig  # noqa: E402
from training.losses import backscatter_chroma_loss, water_direction  # noqa: E402
from training.surface_depth import sorted_hit_alphas as gaussian_hits  # noqa: E402
from training.trainer import Trainer  # noqa: E402

BOTTOM_FRACTION = 0.4
NEAR_DISTANCE = 5.0
FAR_DISTANCE = 10.0
OPEN_WATER = ROOT / "configs/iui3_open_water.npz"


def load_open_water(path):
    """Per-view boolean open-water masks (full resolution), or {} when absent."""
    if path is None or not Path(path).is_file():
        return {}
    with np.load(path) as data:
        return {name: data[name] for name in data.files if name != "meta"}


def coverage_metrics(alpha, bottom_fraction, open_water=None):
    """Gaussian coverage of the seabed (center vs borders) and of open water.

    alpha: [H, W] Gaussian-only alpha; open_water: [H, W] boolean or None.
    The seabed is the lower bottom_fraction of the frame; its center is the
    middle half of the width, the borders the outer quarters.
    """
    height, width = alpha.shape
    seabed = alpha[int(height * (1 - bottom_fraction)):]
    columns = torch.arange(width, device=alpha.device)
    center = (columns >= width // 4) & (columns < width - width // 4)
    opaque = (seabed > 0.99).float()
    row = {"bottom_center_alpha_gt_0.99": float(opaque[:, center].mean()),
           "bottom_border_alpha_gt_0.99": float(opaque[:, ~center].mean())}
    if open_water is not None and open_water.any():
        water = alpha[open_water]
        row["open_water_alpha_mean"] = float(water.mean())
        row["open_water_alpha_gt_0.5"] = float((water > 0.5).float().mean())
    else:
        row["open_water_alpha_mean"] = row["open_water_alpha_gt_0.5"] = float("nan")
    return row


def nearest_cell(points, sites):
    out = torch.empty(points.shape[0], dtype=torch.long, device=points.device)
    for start in range(0, points.shape[0], 60000):
        out[start:start + 60000] = torch.cdist(points[start:start + 60000], sites).argmin(1)
    return out


def psnr(prediction, target, mask=None):
    error = (prediction.clamp(0, 1) - target) ** 2
    if mask is not None:
        error = error[mask]
    return float(-10 * torch.log10(error.mean().clamp_min(1e-12)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="pipeline output (contains joint/)")
    parser.add_argument("--checkpoint", default="best", help="name in joint/checkpoints without .pt")
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, help="default: <run>/audit")
    parser.add_argument("--open-water", type=Path, default=OPEN_WATER,
                        help="frozen open-water masks of the test views")
    args = parser.parse_args()
    joint = args.run.resolve() / "joint"
    checkpoint = joint / "checkpoints" / f"{args.checkpoint}.pt"
    output = (args.output_dir or args.run / "audit").resolve()
    output.mkdir(parents=True, exist_ok=True)

    config = ExperimentConfig.load(joint / "config.yaml")
    config.output_dir = str(output / f"_trainer_{args.checkpoint}")
    config.wandb.mode = "disabled"
    trainer = Trainer(config, resume=checkpoint)
    model, renderer = trainer.model, trainer.renderer
    renderer.transmittance_threshold = config.renderer.validation_transmittance_threshold
    trainer.topology.synchronize_renderer(renderer, model.medium)
    device = torch.device(config.device)
    medium, gaussians = model.medium, model.gaussians
    background = torch.tensor(config.background, device=device)

    from scipy.spatial import cKDTree
    sites = medium.sites.detach()
    coefficients = medium.optical_coefficients.detach()
    attenuation, backscatter = coefficients[:, :3].clone(), coefficients[:, 3:].clone()
    colors = medium.colors.detach().clone()
    distance = torch.as_tensor(
        cKDTree(gaussians.means.detach().cpu().numpy()).query(sites.cpu().numpy())[0],
        device=device)
    far, near = distance > FAR_DISTANCE, distance < NEAR_DISTANCE
    water = dict(attenuation=attenuation[far].median(0).values, backscatter=backscatter[far].median(0).values,
                 color=colors[far].median(0).values)
    anomalous = near & ((backscatter.mean(-1) > 2 * water["backscatter"].mean())
                        | ((colors - water["color"]).abs().max(-1).values > 0.15))

    def replaced(mask):
        return (torch.where(mask[:, None], water["attenuation"], attenuation),
                torch.where(mask[:, None], water["backscatter"], backscatter),
                torch.where(mask[:, None], water["color"], colors))

    variants = {"original": (attenuation, backscatter, colors),
                "clean_anomalous": replaced(anomalous), "clean_near": replaced(near)}

    dense = torch.linspace(0, 80, 1601, device=device)
    tail = torch.logspace(np.log10(80.0), np.log10(float(config.renderer.far)), 65, device=device)[1:]
    edges = torch.cat((dense, tail))
    mids, deltas = 0.5 * (edges[1:] + edges[:-1]), edges[1:] - edges[:-1]

    open_water = load_open_water(args.open_water)
    views = []
    with torch.no_grad():
        for view_index, camera in enumerate(trainer.dataset.test):
            bundle = camera.rays()
            rays = torch.cat((bundle.origins, bundle.directions), -1)[::args.stride, ::args.stride].contiguous()
            height, width = rays.shape[:2]
            target = camera.ground_truth(tuple(config.background))[::args.stride, ::args.stride].reshape(-1, 3)
            bottom = torch.zeros(height, width, dtype=torch.bool, device=device)
            bottom[int(height * (1 - BOTTOM_FRACTION)):] = True
            bottom = bottom.reshape(-1)
            scene = gaussians.render_scene(camera.origin)
            renderer.set_gaussian_geometry(scene)
            reference = renderer.render(rays, gaussians=scene, voronoi=medium.render_scene(),
                                        background=tuple(config.background))
            only = renderer.render_gaussians(rays, scene, background=(0.0, 0.0, 0.0))
            gaussian_alpha = only.alpha.reshape(-1)
            flat = rays.reshape(-1, 6)
            origins, directions = flat[:, :3], F.normalize(flat[:, 3:], dim=-1)
            depth, alpha, index = gaussian_hits(gaussians, reference.hit_depths, reference.hit_indices,
                                                origins, directions, config.renderer.gaussian_radius_squared)
            surface_colors = scene.colors.detach()[index]
            survival = torch.cumprod(1 - alpha, 0)
            before = torch.cat((torch.ones_like(survival[:1]), survival[:-1]), 0)
            below = survival < 0.5
            has_surface = below.any(0)
            surface_depth = torch.where(has_surface, depth.gather(0, below.float().argmax(0)[None])[0],
                                        torch.full_like(has_surface, float("inf"), dtype=torch.float32))
            count = origins.shape[0]
            row = dict(view=view_index)
            for name, (A, B, C) in variants.items():
                direct = torch.zeros(count, 3, device=device)
                scattered = torch.zeros(count, 3, device=device)
                behind = torch.zeros(count, 3, device=device)
                for start in range(0, count, 2048):
                    part = slice(start, start + 2048)
                    n = origins[part].shape[0]
                    points = origins[part, None] + mids[None, :, None] * directions[part, None]
                    cell = nearest_cell(points.reshape(-1, 3), sites).reshape(n, -1)
                    sb, cc, sa = B[cell], C[cell], A[cell]
                    slot = torch.searchsorted(depth[:, part].T.contiguous(), mids[None].expand(n, -1).contiguous())
                    tg = torch.cat((torch.ones(n, 1, device=device), survival[:, part].T), 1).gather(1, slot)
                    tau = sb * deltas[None, :, None]
                    tb = torch.exp(-(torch.cumsum(tau, 1) - tau))
                    contribution = tg[..., None] * tb * cc * (-torch.expm1(-tau))
                    scattered[part] = contribution.sum(1)
                    behind[part] = (contribution * (mids[None] >= surface_depth[part, None])[..., None]).sum(1)
                    optical = torch.cat((torch.zeros(n, 1, 3, device=device),
                                         torch.cumsum(sa * deltas[None, :, None], 1)), 1)
                    hit = depth[:, part].T.clamp(max=edges[-1]).contiguous()
                    j = torch.searchsorted(edges, hit).clamp(1, len(edges) - 1)
                    fraction = ((hit - edges[j - 1]) / deltas[j - 1]).clamp(0, 1)[..., None]
                    at_hit = (optical.gather(1, (j - 1)[..., None].expand(-1, -1, 3)) * (1 - fraction)
                              + optical.gather(1, j[..., None].expand(-1, -1, 3)) * fraction)
                    weight = (before[:, part] * alpha[:, part]).T
                    direct[part] = ((weight[..., None] * surface_colors[:, part].transpose(0, 1)
                                     * torch.exp(-at_hit)).sum(1)
                                    + survival[-1, part, None] * torch.exp(-optical[:, -1]) * background)
                hybrid = direct + scattered
                row[f"{name}/psnr"] = psnr(hybrid, target)
                row[f"{name}/psnr_bottom"] = psnr(hybrid, target, bottom)
                row[f"{name}/psnr_water"] = psnr(hybrid, target, ~bottom)
                if name == "original":
                    row["marcher_mae_backscatter"] = float((scattered - reference.backscatter.reshape(-1, 3)).abs().mean())
                    row["marcher_mae_hybrid"] = float((hybrid - reference.image.reshape(-1, 3)).abs().mean())
                    row["renderer_psnr"] = psnr(reference.image.reshape(-1, 3), target)
                    share = behind.sum(-1) / scattered.sum(-1).clamp_min(1e-6)
                    on = has_surface & bottom
                    row["behind_share_bottom_mean"] = float(share[on].mean()) if on.any() else float("nan")
                    row["behind_share_bottom_p90"] = float(share[on].quantile(0.9)) if on.any() else float("nan")
                    row["backscatter_share_bottom"] = float(
                        (scattered.sum(-1) / hybrid.sum(-1).clamp_min(1e-6))[bottom].mean())
            for threshold in (0.5, 0.9, 0.99):
                row[f"bottom_alpha_gt_{threshold}"] = float((gaussian_alpha[bottom] > threshold).float().mean())
            row["water_alpha_lt_0.01"] = float((gaussian_alpha[~bottom] < 0.01).float().mean())
            mask = open_water.get(camera.name)
            if mask is not None:
                mask = torch.as_tensor(mask[::args.stride, ::args.stride], device=device)
                if mask.shape != (height, width):
                    raise ValueError(f"open-water mask of {camera.name} does not match the test view")
            row.update(coverage_metrics(gaussian_alpha.reshape(height, width), BOTTOM_FRACTION, mask))
            if reference.backscatter is not None:
                # Hue of the backscatter on seabed rays vs open-water rays of this view.
                # Not "backscatter": that name holds the per-cell coefficients used below.
                pixel_backscatter = reference.backscatter.reshape(-1, 3)
                water = water_direction(pixel_backscatter, gaussian_alpha, 0.01)
                seabed_alpha = torch.where(bottom, gaussian_alpha, torch.zeros_like(gaussian_alpha))
                row["backscatter_hue_angle_bottom"] = backscatter_chroma_loss(
                    pixel_backscatter, seabed_alpha, water, 0.5, 0.0)[1]
            row["gaussian_only_black_psnr"] = psnr(only.image.reshape(-1, 3), target)
            row["gaussian_only_black_psnr_bottom"] = psnr(only.image.reshape(-1, 3), target, bottom)
            views.append(row)
            print(json.dumps(row), flush=True)

    keys = [k for k in views[0] if k != "view"]
    mean = {k: float(np.nanmean([v[k] for v in views])) for k in keys}
    for name in ("clean_anomalous", "clean_near"):
        for suffix in ("", "_bottom", "_water"):
            mean[f"{name}/delta_psnr{suffix}"] = mean[f"{name}/psnr{suffix}"] - mean[f"original/psnr{suffix}"]
    opacity = gaussians.opacities.detach().flatten()
    summary = dict(
        checkpoint=str(checkpoint), iteration=int(trainer.start_iteration - 1), stride=args.stride,
        cells=dict(total=int(sites.shape[0]), near=int(near.sum()), anomalous=int(anomalous.sum()),
                   color_std_near=float(colors[distance < 2].std(0).mean()) if (distance < 2).sum() > 1 else float("nan"),
                   color_std_far=float(colors[far].std(0).mean()),
                   backscatter_median_near=float(backscatter[distance < 2].mean(-1).median()) if (distance < 2).any() else float("nan"),
                   backscatter_median_far=float(backscatter[far].mean(-1).median()),
                   backscatter_max_near=float(backscatter[distance < 2].mean(-1).max()) if (distance < 2).any() else float("nan")),
        gaussians=dict(count=int(opacity.numel()), opacity_gt_0_95=float((opacity > 0.95).float().mean()),
                       opacity_lt_0_5=float((opacity < 0.5).float().mean())),
        mean=mean, views=views)
    path = output / f"{args.checkpoint}.json"
    path.write_text(json.dumps(summary, indent=2))
    print(path)


if __name__ == "__main__":
    main()
