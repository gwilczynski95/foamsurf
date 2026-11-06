"""How well the photos cover what a camera sees: novel views far from the diver's path show holes and smeared
surfels at the frame edges. For every pixel of a (downscaled) frame: the first Gaussian surface, and how many
training photos have that point inside their frame. A pixel is poor when its ray looks down and meets no surface
(a hole in the seabed), or when fewer than `min_views` photos saw its surface point.

    python vis/coverage.py --run RUN --scene DATA --shot vis/shots/x.json [--scale 0.25]

prints the poor fraction of every second of the shot's camera path (whole frame and a 10 % border).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@torch.no_grad()
def views_seeing(trainer, points: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    """[P] number of training photos whose frame contains the point (in front of the camera, no occlusion test)."""
    import render
    count = torch.zeros(points.shape[0], dtype=torch.long, device=points.device)
    for camera in trainer.dataset.train:
        r = render.camera_record(camera)
        v = points - camera.origin
        z = v @ camera.forward
        x = (v @ camera.right) / z.clamp_min(1e-6) * r["fx"] + r["cx"]
        y = (v @ camera.down) / z.clamp_min(1e-6) * r["fy"] + r["cy"]
        mx, my = margin * r["width"], margin * r["height"]
        count += ((z > 0) & (x >= mx) & (x < r["width"] - mx) & (y >= my) & (y < r["height"] - my)).long()
    return count


@torch.no_grad()
def poor_pixels(tracer, camera, min_views: int = 2, down_cos: float = -0.3) -> torch.Tensor:
    """[H,W] bool: hole looking down, or surface seen by fewer than min_views photos."""
    from evaluation.evaluator import _rays
    rays = _rays(camera).reshape(-1, 6).contiguous()
    only = tracer.renderer.render_gaussians(rays, tracer.gaussians, background=(0.0, 0.0, 0.0))
    depth = only.surface_depth.reshape(-1)
    hit = depth > 0
    looks_down = rays[:, 3:] @ tracer.up.to(rays) < down_cos
    poor = looks_down & ~hit
    points = rays[hit, :3] + depth[hit, None] * rays[hit, 3:]
    poor[hit] = views_seeing(tracer.trainer, points) < min_views
    return poor.reshape(camera.height, camera.width)


def border(h: int, w: int, fraction: float, device) -> torch.Tensor:
    mask = torch.ones(h, w, dtype=torch.bool, device=device)
    bh, bw = max(1, int(h * fraction)), max(1, int(w * fraction))
    mask[bh:-bh, bw:-bw] = False
    return mask


def main():
    import render
    from vis.raytrace import SceneTracer
    from vis.render_clip import camera_path, build_objects
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--scene", type=Path)
    parser.add_argument("--shot", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=0.25)
    args = parser.parse_args()
    shot = json.loads(args.shot.read_text())
    trainer = render.load_run(args.run.resolve(), "last", args.scene)
    anchor = {c.name: c for c in trainer.dataset.train + trainer.dataset.test}[shot["anchor_view"]]
    tracer = SceneTracer(trainer, anchor.origin)
    objects, _ = build_objects(tracer, anchor, shot["objects"])
    target = torch.stack([torch.as_tensor(o.center, dtype=torch.float32, device=trainer.device) for o in objects]).mean(0)
    path = dict(shot.get("path", {}), scale=args.scale)
    cameras = camera_path(trainer, anchor, target, shot.get("frames", 120), path)
    fps = shot.get("fps", 30)
    print("second  poor(frame)  poor(border 10%)")
    for second in range(0, len(cameras), fps):
        frame, edge = [], []
        for camera in cameras[second:second + fps:max(1, fps // 6)]:
            poor = poor_pixels(tracer, camera)
            frame.append(poor.float().mean())
            edge.append(poor[border(*poor.shape, 0.1, poor.device)].float().mean())
        print(f"{second // fps:6d}  {float(torch.stack(frame).mean()):11.3f}  {float(torch.stack(edge).mean()):16.3f}")


if __name__ == "__main__":
    main()
