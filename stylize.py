"""Image/text-guided style transfer of a trained FoamSurf run, separately for the Gaussians or the Voronoi medium.

    python stylize.py --run RUN --scene DATA --target gaussians --style-image fire.jpg --out OUT
    python stylize.py --run RUN --scene DATA --target voronoi --style-image starry_night.jpg --out OUT
    python compose_style.py --gaussians OUT_G --medium OUT_V --out OUT_BOTH      # both stylized in one scene
    python render.py --run OUT_BOTH --scene DATA --views all --modes full,gaussians,voronoi

The objective is the CLIPGaussian loss as used by FlaRe (style_transfer.py): directional + patch CLIP terms against
the render of the model before stylization, VGG content, 5000 steps, one camera per step, all cameras of the dataset.

--target gaussians: the loss sees the Gaussians alone on black (render.py mode "gaussians"). Trained: the SH colour
    (DC; the higher bands with --sh-rest train) and, as in FlaRe, the geometry with small learning rates (--no-geometry
    keeps it). The medium is untouched. A background term (lambda_bg) keeps the black pixels black, i.e. the reef
    silhouette as it was (CLIPGaussian --object); crops only where the Gaussians cover the image (--patch-coverage).
    Against the hue noise CLIP leaves in bright areas (pink/cyan Gaussians next to each other): the colour stays <= 1
    (--color-cap) and its gradient is averaged over the 8 nearest Gaussians (--color-grad-smooth).
--target voronoi: the loss sees the medium alone on black (render.py mode "voronoi"). Every cell is first split into
    27 (--medium-split; children around the site with the parent's parameters): the 5000 trained cells average
    out any pattern along the rays. Trained: the cell colours, the backscatter coefficients (bright spots) and the
    sites (the Delaunay topology is rebuilt every --topology-interval steps). The attenuation of the direct light is
    never changed, nor are the Gaussians.

Defaults = the IUI3 baseline (fire on the Gaussians, starry_night on the water): 18-19 min per target on an RTX 3090
(29 cameras 1383x917, 670k Gaussians), ~13 GB of GPU memory.

OUT is a run directory (OUT/joint/config.yaml, OUT/joint/checkpoints/last.pt) that render.py loads like any trained
run, plus OUT/style/ with the loss log and previews.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from style_transfer import StyleLoss, StyleLossConfig

C0 = 0.28209479177387814


def to_bchw(flat: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return flat.reshape(height, width, 3).permute(2, 0, 1).unsqueeze(0)


def save_png(image: torch.Tensor, path: Path) -> None:
    """image [H,W,3] or [1,3,H,W] in [0,1]."""
    if image.dim() == 4:
        image = image[0].permute(1, 2, 0)
    value = image.detach().float().clamp(0, 1).cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((value * 255.0 + 0.5).astype(np.uint8)).save(path)


def tree_to(value, device):
    if isinstance(value, dict):
        return {k: tree_to(v, device) for k, v in value.items()}
    return value.to(device) if torch.is_tensor(value) else value


def subdivide_medium(trainer, factor: int, jitter: float, seed: int) -> None:
    """Every non-guard cell becomes `factor` cells: the site and factor-1 children at site + N(0, (jitter * d)^2)
    (d = distance to the nearest site), all with the parent's coefficients and colour. The medium looks almost the
    same and has factor x more cells to paint with."""
    from scipy.spatial import cKDTree
    from foamsurf_model.medium import VoronoiMediumModel
    from foamsurf_model.topology import VoronoiTopologyController
    medium = trainer.model.medium
    sites = medium.sites.detach().cpu()
    guard = medium.guard_mask.cpu()
    inner = torch.nonzero(~guard).flatten()
    distance, _ = cKDTree(sites.numpy()).query(sites.numpy(), k=2)
    radius = torch.as_tensor(distance[:, 1], dtype=torch.float32)[inner]
    generator = torch.Generator().manual_seed(seed)
    offsets = torch.randn(inner.numel(), factor - 1, 3, generator=generator) * (jitter * radius)[:, None, None]
    children = (sites[inner][:, None] + offsets).reshape(-1, 3)
    children = torch.minimum(torch.maximum(children, medium.bbox_min.cpu()), medium.bbox_max.cpu())
    parent = torch.cat((inner, inner.repeat_interleave(factor - 1), torch.nonzero(guard).flatten()))
    new_sites = torch.cat((sites[inner], children, sites[guard]))
    state = {"sites": new_sites, "raw_density": medium.raw_density.detach().cpu()[parent],
             "raw_color": medium.raw_color.detach().cpu()[parent]}
    metadata = dict(medium.checkpoint_metadata(), point_count=int(new_sites.shape[0]))
    trainer.model.medium = VoronoiMediumModel.from_checkpoint(state, metadata, medium.bbox_min, medium.bbox_max,
                                                              trainer.device)
    trainer.config.model.medium_points = int(new_sites.shape[0])
    adaptation = trainer.config.voronoi_adaptation  # config validation: medium_points <= the adaptation budget
    if adaptation.max_points is not None:
        adaptation.max_points = max(adaptation.max_points, int(new_sites.shape[0]))
    trainer.topology = VoronoiTopologyController(trainer.model.medium)
    trainer.topology.synchronize_renderer(trainer.renderer, trainer.model.medium)
    print(f"medium split x{factor}: {sites.shape[0]} -> {new_sites.shape[0]} cells", flush=True)


def build_optimizer(trainer, args):
    """Adam over the trained parameters of the target; returns (optimizer, {name: param getter}, geometry names)."""
    if args.target == "gaussians":
        g = trainer.model.gaussians
        scale = g.rgb_scale if g.rgb_activation == "sigmoid" else 1.0
        # colour step per unit of the DC coefficient at mid-range: d colour / d dc = scale * sigmoid' * C0
        slope = scale * 0.25 * C0 if g.rgb_activation == "sigmoid" else C0
        rates = {"appearance_dc": args.color_lr / slope}
        if args.sh_rest == "train":
            rates["appearance_rest"] = rates["appearance_dc"] / 20.0
        if not args.no_geometry:
            rates.update(means=args.lr_means, log_scales=args.lr_scales, quaternions=args.lr_quats,
                         opacity_logits=args.lr_opacity, raw_kappa=args.lr_kappa)
        groups = [{"params": [getattr(g, n)], "lr": lr, "name": n} for n, lr in rates.items() if lr > 0]
        getters = {n: (lambda n=n: getattr(trainer.model.gaussians, n)) for n in rates}
        geometry = [n for n in rates if n in ("means", "log_scales", "quaternions", "opacity_logits", "raw_kappa")]
        return torch.optim.Adam(groups, eps=args.adam_eps), getters, geometry
    m = trainer.model.medium
    color_slope = 0.25 if getattr(m, "color_activation", "sigmoid") == "sigmoid" else C0
    # group names of VoronoiTopologyController, so a topology update permutes the Adam state too
    groups = [{"params": [m.sites], "lr": args.medium_sites_lr, "name": "medium_sites"},
              {"params": [m.raw_density], "lr": args.medium_backscatter_lr, "name": "medium_density"},
              {"params": [m.raw_color], "lr": args.medium_color_lr / color_slope, "name": "medium_color"}]
    getters = {n: (lambda n=n: getattr(trainer.model.medium, n)) for n in ("sites", "raw_density", "raw_color")}
    return torch.optim.Adam(groups, eps=args.adam_eps), getters, []


def render_target(trainer, camera, rays, target: str, grad: bool):
    """The image the loss sees, [R,3] (and the Gaussian alpha for the gaussians target)."""
    renderer = trainer.renderer
    with torch.set_grad_enabled(grad):
        if target == "gaussians":
            scene = trainer.model.gaussians.render_scene(camera.origin)
            with torch.no_grad():
                renderer.set_gaussian_geometry(scene)
            out = renderer.render(rays, gaussians=scene, background=(0.0, 0.0, 0.0), surface=False)
            return out.image, out.alpha
        out = renderer.render(rays, voronoi=trainer.model.medium.render_scene(), background=(0.0, 0.0, 0.0),
                              surface=False)
        return out.image, None


def save_run(trainer, base_run: Path, out: Path, scene: Path | None, info: dict) -> Path:
    from training.checkpoints import checkpoint_payload, load_checkpoint, save_checkpoint
    from training.optimizer import build_optimizers
    joint = out / "joint"
    (joint / "checkpoints").mkdir(parents=True, exist_ok=True)
    for name in ("pipeline_config.json", "water_fit.json"):
        if (base_run / name).is_file():
            shutil.copy(base_run / name, out / name)
    config = trainer.config
    config.output_dir = str(joint)
    config.save(joint / "config.yaml")
    gaussian_optimizer, medium_optimizer = build_optimizers(trainer.model, config.optimization, trainer.scene_extent)
    payload = checkpoint_payload(
        trainer.model, gaussian_optimizer, medium_optimizer, config, trainer.start_iteration - 1,
        trainer.training_seconds, trainer.best_psnr, trainer.topology.revision, None, None,
        trainer.medium_frozen_until, trainer.disentanglement.state_dict(), extras=trainer._checkpoint_extras())
    base_joint = base_run / "joint" if (base_run / "joint").is_dir() else base_run
    base = load_checkpoint(base_joint / "checkpoints" / f"{info['checkpoint']}.pt")
    for key, value in base.items():  # training extras of the base run (e.g. colmap_joint), unchanged
        payload.setdefault(key, value)
    payload["style"] = dict(base.get("style") or {}, **{info["target"]: info})
    path = joint / "checkpoints" / "last.pt"
    save_checkpoint(path, payload)
    return path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="trained (or already stylized) run")
    parser.add_argument("--checkpoint", default="last", choices=("last", "best"))
    parser.add_argument("--scene", type=Path, help="dataset directory of the run")
    parser.add_argument("--target", required=True, choices=("gaussians", "voronoi"))
    parser.add_argument("--style-image", default="")
    parser.add_argument("--style-prompt", default="")
    parser.add_argument("--object-prompt", default="a Photo")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--views", default="all", choices=("all", "train"))
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--preview-views", default="", help="comma-separated camera names (default: 3 spread views)")
    parser.add_argument("--seed", type=int, default=0)
    # loss (FlaRe / CLIPGaussian defaults)
    parser.add_argument("--lambda-dir", type=float, default=5.0)
    parser.add_argument("--lambda-patch", type=float, default=90.0)
    parser.add_argument("--lambda-content", type=float, default=0.8)
    parser.add_argument("--lambda-bg", type=float, default=None, help="default 1000 for gaussians, 0 for voronoi")
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--num-crops", type=int, default=64)
    parser.add_argument("--patch-coverage", type=float, default=None,
                        help="crops only where the Gaussian alpha covers >= this fraction (default 0.5 for "
                             "gaussians; 0 = uniform crops as FlaRe)")
    parser.add_argument("--mask-threshold", type=float, default=0.02, help="Gaussian alpha of object pixels")
    parser.add_argument("--clip-model", default="ViT-B/32")
    # Gaussians (geometry lrs: FlaRe; colour step: CLIPGaussian's feature_lr 0.01 is 0.01*C0 here and FlaRe uses x2 for
    # "fire" -- x1.5 with the cap and the gradient smoothing below kept the flames of x2 at 1/4 of its hue noise)
    parser.add_argument("--color-lr", type=float, default=0.015 * C0, help="step in colour space")
    parser.add_argument("--sh-rest", default="freeze", choices=("freeze", "train"))
    parser.add_argument("--no-geometry", action="store_true")
    parser.add_argument("--lr-means", type=float, default=1e-4)
    parser.add_argument("--lr-scales", type=float, default=5e-3)
    parser.add_argument("--lr-quats", type=float, default=1e-3)
    parser.add_argument("--lr-opacity", type=float, default=1e-3)
    parser.add_argument("--lr-kappa", type=float, default=1e-4)
    parser.add_argument("--geometry-grad-clip", type=float, default=0.1, help="<= 0 disables")
    parser.add_argument("--model-reg", type=float, default=1e-4, help="MSE of the trained parameters to the base")
    parser.add_argument("--scale-reg", type=float, default=1e-3)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--color-cap", type=float, default=1.0,
                        help="every channel of the DC colour of every Gaussian stays <= cap (also clamped at the "
                             "start): the loss sees the image clamped to 1, so above it channels drift unseen up to "
                             "the sigmoid ceiling (1.5) and show up as hue shifts under the water. <= 0 disables")
    parser.add_argument("--color-grad-smooth", type=int, default=8, metavar="K",
                        help="average the colour gradient of each Gaussian with its K nearest neighbours (base "
                             "positions), against per-Gaussian colour confetti; 0 disables")
    # Voronoi (5000 trained cells are too coarse for a pattern: split x27, cell colours x4 the Gaussian FlaRe step,
    # backscatter (bright spots) and site positions trained, Delaunay rebuilt every 25 steps)
    parser.add_argument("--medium-color-lr", type=float, default=0.04 * C0, help="step in colour space")
    parser.add_argument("--medium-backscatter-lr", type=float, default=1e-2)
    parser.add_argument("--medium-sites-lr", type=float, default=1e-3)
    parser.add_argument("--medium-split", type=int, default=27)
    parser.add_argument("--medium-split-jitter", type=float, default=0.25)
    parser.add_argument("--topology-interval", type=int, default=25)
    args = parser.parse_args(argv)
    if bool(args.style_image) == bool(args.style_prompt):
        parser.error("give exactly one of --style-image and --style-prompt")
    if args.lambda_bg is None:
        args.lambda_bg = 1000.0 if args.target == "gaussians" else 0.0
    if args.patch_coverage is None:
        args.patch_coverage = 0.5 if args.target == "gaussians" else 0.0
    if args.out.exists() and any(args.out.iterdir()):
        parser.error(f"output directory is not empty: {args.out}")

    import render as foamsurf_render
    from evaluation.evaluator import _rays
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    started = time.time()
    run = args.run.resolve()
    trainer = foamsurf_render.load_run(run, args.checkpoint, args.scene)
    trainer.renderer.transmittance_threshold = trainer.config.renderer.training_transmittance_threshold
    device = trainer.device
    dataset = trainer.dataset
    cameras = list(dataset.train) + (list(dataset.test) if args.views == "all" else [])
    cameras.sort(key=lambda c: c.name)
    style_dir = args.out / "style"
    style_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(cameras)} cameras {cameras[0].width}x{cameras[0].height}, "
          f"{trainer.model.gaussians.point_count} Gaussians, {trainer.model.medium.point_count} cells, "
          f"scene extent {trainer.scene_extent:.3f}", flush=True)

    if args.target == "voronoi" and args.medium_split > 1:
        subdivide_medium(trainer, args.medium_split, args.medium_split_jitter, args.seed)

    loss_config = StyleLossConfig(style_image=str(args.style_image), style_prompt=args.style_prompt,
                                  object_prompt=args.object_prompt, clip_model=args.clip_model,
                                  lambda_dir=args.lambda_dir, lambda_patch=args.lambda_patch,
                                  lambda_content=args.lambda_content, lambda_bg=args.lambda_bg,
                                  crop_size=args.crop_size, num_crops=args.num_crops,
                                  patch_coverage=args.patch_coverage)
    loss_fn = StyleLoss(loss_config, device)

    # base renders: content reference and CLIP source of every camera (on the CPU), as FlaRe
    references = []
    with torch.no_grad():
        for camera in cameras:
            rays = _rays(camera).reshape(-1, 6)
            image, alpha = render_target(trainer, camera, rays, args.target, grad=False)
            image = to_bchw(torch.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1),
                            camera.height, camera.width)
            # object = any Gaussian coverage (FlaRe: base render differs from the background by > 0.02)
            mask = None if alpha is None else (alpha.reshape(1, 1, camera.height, camera.width)
                                               > args.mask_threshold).float()
            references.append((tree_to(loss_fn.prepare_reference(image), "cpu"),
                               None if mask is None else mask.cpu()))
    torch.cuda.empty_cache()

    names = [c.name for c in cameras]
    if args.preview_views:
        preview = [names.index(n) for n in args.preview_views.split(",")]
    else:
        preview = sorted({0, len(cameras) // 2, len(cameras) - 1})

    def save_previews(tag: str) -> None:
        for index in preview:
            images = foamsurf_render.render_view(trainer, cameras[index], ["full", args.target])
            for mode, image in images.items():
                save_png(image, style_dir / "previews" / f"{cameras[index].name}_{mode}_{tag}.png")
    save_previews("base")

    optimizer, getters, geometry = build_optimizer(trainer, args)
    # gradients only where they are used (the other representation and the frozen parameters)
    other = trainer.model.medium if args.target == "gaussians" else trainer.model.gaussians
    other.requires_grad_(False)
    if args.target == "gaussians":
        for name, parameter in trainer.model.gaussians.named_primitive_parameters().items():
            parameter.requires_grad_(name in getters)
    originals = {n: get().detach().clone() for n, get in getters.items()}
    base_log_scales = None
    if "log_scales" in getters:
        log_scales = getters["log_scales"]().detach()
        base_log_scales = (float(log_scales.min()), float(log_scales.max()))
    print("trained:", ", ".join(f"{g['name']} lr {g['lr']:.3g}" for g in optimizer.param_groups), flush=True)
    dc_cap = neighbours = None
    if args.target == "gaussians" and "appearance_dc" in getters:
        g = trainer.model.gaussians
        if args.color_cap > 0:
            if g.rgb_activation == "sigmoid":  # rgb_scale * sigmoid(C0 * dc) = cap
                limit = (math.log(args.color_cap / (g.rgb_scale - args.color_cap)) / C0
                         if args.color_cap < g.rgb_scale else math.inf)
            else:  # 0.5 + C0 * dc = cap
                limit = (args.color_cap - 0.5) / C0
            dc_cap = limit
            above = float((originals["appearance_dc"] > limit).float().mean())
            with torch.no_grad():
                getters["appearance_dc"]().clamp_(max=limit)
            print(f"colour cap {args.color_cap}: DC <= {limit:.3f}, {above:.3f} of the base values clamped", flush=True)
        if args.color_grad_smooth > 0:
            from scipy.spatial import cKDTree
            means = g.means.detach().cpu().numpy()
            _, index = cKDTree(means).query(means, k=args.color_grad_smooth + 1, workers=-1)
            neighbours = torch.as_tensor(index, device=device, dtype=torch.long)  # [N, K+1], itself first
            print(f"colour gradient averaged over {args.color_grad_smooth} neighbours", flush=True)

    log = open(style_dir / "losses.csv", "w")
    log.write("step,camera,total,directional,patch,content,background,model_reg,seconds\n")
    order = []
    for step in range(args.steps):
        if not order:  # every camera once per epoch, in random order
            order = torch.randperm(len(cameras)).tolist()
        index = order.pop()
        camera = cameras[index]
        reference, mask = references[index]
        reference = tree_to(reference, device)
        mask = None if mask is None else mask.to(device)
        optimizer.zero_grad(set_to_none=True)

        rays = _rays(camera).reshape(-1, 6)
        raw, _ = render_target(trainer, camera, rays, args.target, grad=True)
        finite = torch.isfinite(raw)
        clean = torch.where(finite, raw, torch.zeros_like(raw))
        # the loss sees the clamped image; its gradient reaches the render unclamped (FlaRe passes dL/dimage of the
        # clamped image straight to the renderer backward)
        image = to_bchw(clean + (clean.clamp(0, 1) - clean).detach(), camera.height, camera.width)
        loss, metrics = loss_fn(image, reference, mask)
        model_reg = image.new_zeros(())
        if args.model_reg > 0:
            for name, get in getters.items():
                model_reg = model_reg + F.mse_loss(get(), originals[name])
            model_reg = args.model_reg * model_reg
        (loss + model_reg).backward()

        with torch.no_grad():
            for name, get in getters.items():  # never let a NaN gradient reach Adam
                grad = get().grad
                if grad is not None:
                    grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            if neighbours is not None and getters["appearance_dc"]().grad is not None:
                grad = getters["appearance_dc"]().grad
                grad.copy_(grad[neighbours].mean(1))
            if args.target == "voronoi" and getters["raw_density"]().grad is not None:
                getters["raw_density"]().grad[:, :3] = 0.0  # attenuation of the direct light stays
            if "log_scales" in getters and args.scale_reg > 0:
                s = getters["log_scales"]()
                if s.grad is not None:
                    squared = torch.exp(s) ** 2
                    s.grad.add_((args.scale_reg / max(1, s.shape[0])) * squared
                                / torch.sqrt(squared.sum(1, keepdim=True)).clamp_min(1e-12))
        if geometry and args.geometry_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([getters[n]() for n in geometry], args.geometry_grad_clip)
        optimizer.step()
        with torch.no_grad():
            if base_log_scales is not None:
                getters["log_scales"]().clamp_(*base_log_scales)
            if dc_cap is not None:
                getters["appearance_dc"]().clamp_(max=dc_cap)
        if (args.target == "voronoi" and args.medium_sites_lr > 0
                and (step + 1) % args.topology_interval == 0):
            if trainer.topology.update(trainer.model.medium, optimizer):
                permutation = trainer.topology.last_permutation
                originals = {n: v[permutation].contiguous() for n, v in originals.items()}
            trainer.topology.synchronize_renderer(trainer.renderer, trainer.model.medium)

        seconds = time.time() - started
        log.write(f"{step},{camera.name},{metrics['total']:.6f},{metrics['directional']:.6f},{metrics['patch']:.6f},"
                  f"{metrics['content']:.6f},{metrics['background']:.6f},{float(model_reg):.3e},{seconds:.1f}\n")
        if step % 50 == 0 or step == args.steps - 1:
            log.flush()
            print(f"step {step} {camera.name} total {metrics['total']:.4f} dir {metrics['directional']:.4f} "
                  f"patch {metrics['patch']:.4f} content {metrics['content']:.4f} bg {metrics['background']:.5f} "
                  f"{seconds:.0f}s", flush=True)
        if (step + 1) % args.save_every == 0 or step == args.steps - 1:
            save_previews(f"{step + 1:05d}")
    log.close()

    info = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}
    info.update(base_run=str(run), seconds=time.time() - started, cameras=names)
    path = save_run(trainer, run, args.out.resolve(), args.scene, info)
    (style_dir / "style_config.json").write_text(json.dumps(info, indent=1))
    print(f"stylized run in {args.out} ({path}), {info['seconds'] / 60:.1f} min, "
          f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)


if __name__ == "__main__":
    main()
