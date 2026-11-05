"""A clip of a trained FoamSurf scene with ray-traced objects, the camera gliding along the diver's path.

    python vis/render_clip.py --run RUN --scene DATA --shot vis/shots/curacao_spheres.json --out renders/clips/NAME

Shot file (JSON):
  {"anchor_view": "MTN_1296",                 # test view used to place the objects
   "objects": [{"u": 0.40, "v": 0.72, "radius_fraction": 0.07, "lift": 1.0, "material": "mirror", "tint": [...]},
               ...],                         # resting on the surface seen at image fraction (u, v) of the anchor view
   "frames": 120, "fps": 30,                  # the camera model's exposure is that of the anchor view
   "path": {"look_at_objects": 0.7,          # 0: the photos' own viewing direction, 1: aim at the objects
            "views": null,                   # views to pass through (default: all training views, along their main axis)
            "scale": 1.0},                   # image size factor
   "trace": {...}}                           # TraceSettings overrides
The camera follows a Catmull-Rom spline through the chosen training camera positions (novel views stay close to
the photos), rolled level with the cameras' mean "up". Frames go to OUT/frames, the video to OUT.mp4, the resolved
shot (with object centres in world units) to OUT/shot_resolved.json.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import render  # noqa: E402
from vis.raytrace import SceneTracer, Sphere, TraceSettings, place_on_seabed, render_camera  # noqa: E402


def catmull_rom(points: torch.Tensor, samples: int) -> torch.Tensor:
    """Uniform Catmull-Rom spline through points [N,3] (end points repeated), samples [S,3]."""
    padded = torch.cat((points[:1], points, points[-1:]))
    segments = points.shape[0] - 1
    out = []
    for s in torch.linspace(0, segments, samples, device=points.device).tolist():
        i = min(int(s), segments - 1)
        t = s - i
        p0, p1, p2, p3 = padded[i], padded[i + 1], padded[i + 2], padded[i + 3]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    return torch.stack(out)


def smooth(values: torch.Tensor, passes: int = 3) -> torch.Tensor:
    for _ in range(passes):
        values = torch.cat((values[:1], (values[:-2] + 2 * values[1:-1] + values[2:]) / 4, values[-1:]))
    return values


def camera_path(trainer, template, target: torch.Tensor, frames: int, look_at: float, views=None, scale=1.0):
    train = list(trainer.dataset.train)
    if views:
        by_name = {c.name: c for c in train + list(trainer.dataset.test)}
        train = [by_name[v] for v in views]
    else:  # along the main axis of the camera positions (photo names zigzag)
        origins = torch.stack([c.origin for c in train])
        axis = torch.linalg.svd(origins - origins.mean(0)).Vh[0]
        train = [train[i] for i in ((origins - origins.mean(0)) @ axis).argsort().tolist()]
    origins = smooth(catmull_rom(torch.stack([c.origin for c in train]), frames), passes=frames // 10)
    own = catmull_rom(torch.stack([c.forward for c in train]), frames)
    up = -torch.stack([c.down for c in trainer.dataset.train]).mean(0)
    up = up / up.norm()
    aim = F.normalize(target[None] - origins, dim=-1)
    forward = F.normalize(smooth(F.normalize(own, dim=-1) * (1 - look_at) + aim * look_at), dim=-1)
    record = render.camera_record(template)
    width, height = int(round(record["width"] * scale / 2) * 2), int(round(record["height"] * scale / 2) * 2)
    cameras = []
    for i in range(frames):
        f = forward[i]
        right = F.normalize(torch.linalg.cross(-up, f), dim=0)
        down = torch.linalg.cross(f, right)
        cameras.append(render.camera_from_record(dict(
            name=f"frame_{i:04d}", origin=origins[i].tolist(), right=right.tolist(), down=down.tolist(),
            forward=f.tolist(), width=width, height=height, fx=record["fx"] * width / record["width"],
            fy=record["fy"] * height / record["height"], cx=record["cx"] * width / record["width"],
            cy=record["cy"] * height / record["height"]), trainer.device))
    return cameras


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--scene", type=Path)
    parser.add_argument("--shot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, help="override the shot's frame count (previews)")
    args = parser.parse_args()
    shot = json.loads(args.shot.read_text())
    trainer = render.load_run(args.run.resolve(), "last", args.scene)
    anchor = {c.name: c for c in trainer.dataset.train + trainer.dataset.test}[shot["anchor_view"]]
    tracer = SceneTracer(trainer, anchor.origin)

    objects = []
    for spec in shot["objects"]:
        if "center" in spec:
            center, radius = spec["center"], spec["radius"]
        else:
            center, radius, _ = place_on_seabed(tracer, anchor, spec["u"], spec["v"], spec["radius_fraction"],
                                                spec.get("lift", 1.0))
        objects.append(Sphere(center, radius, spec.get("material", "mirror"), tuple(spec.get("tint", (1, 1, 1))),
                              spec.get("ior", 1.5 / 1.33)))
    target = torch.tensor([o.center for o in objects], device=trainer.device).mean(0)
    path = shot.get("path", {})
    frames = args.frames or shot.get("frames", 120)
    cameras = camera_path(trainer, anchor, target, frames, path.get("look_at_objects", 0.7), path.get("views"),
                          path.get("scale", 1.0))
    settings = TraceSettings(**shot.get("trace", {}))

    out_frames = args.out / "frames"
    out_frames.mkdir(parents=True, exist_ok=True)
    resolved = dict(shot, objects=[dict(center=o.center, radius=o.radius, material=o.material, tint=list(o.tint),
                                        ior=o.ior) for o in objects], frames=frames,
                    run=str(args.run), cameras=[render.camera_record(c) for c in cameras])
    (args.out / "shot_resolved.json").write_text(json.dumps(resolved, indent=1))
    start = time.time()
    for i, camera in enumerate(cameras):
        _, full = render_camera(tracer, camera, objects, settings, isp_origin=anchor.origin)
        image = (full.cpu().numpy() * 255 + 0.5).astype(np.uint8)
        Image.fromarray(image).save(out_frames / f"{i:04d}.png")
        if i % 20 == 0:
            print(f"frame {i}/{frames}  {time.time() - start:.1f} s", flush=True)
    video = args.out.with_suffix(".mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(shot.get("fps", 30)), "-i",
                    str(out_frames / "%04d.png"), "-c:v", "libx264", "-crf", "16", "-preset", "slow",
                    "-pix_fmt", "yuv420p", str(video)], check=True)
    print(f"{frames} frames in {time.time() - start:.1f} s -> {video}")


if __name__ == "__main__":
    main()
