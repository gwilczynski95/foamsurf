"""A clip of a trained FoamSurf scene with ray-traced objects, the camera gliding along the diver's path.

    python vis/render_clip.py --run RUN --scene DATA --shot vis/shots/curacao_spheres.json --out renders/clips/NAME

Shot file (JSON):
  {"anchor_view": "MTN_1296",                 # test view used to place the objects
   "objects": [{"u": 0.40, "v": 0.72, "radius_fraction": 0.07, "material": "mirror", "tint": [...],
                "height": 0.0,                # radii above the surface (floating); or "depth_fraction": 0.5 puts
                                              # the centre on the pixel's ray, at half the surface distance;
                                              # "distance": 30 at 30 world units (also in the open water)
                "clear": true},               # lifted out of the reef if it would cut into it
               {"type": "mesh", "manifest": "$FALCON/source/fiat_bicycle/manifest.json", "u": 0.5, "v": 0.7,
                "span_fraction": 0.25, "yaw_deg": 30, "sink": 0.0, "materials": {"caroseria": {"mode": ...}}},
               {"type": "mirror_panel", "u": .., "v": .., "span_fraction": 0.2, "yaw_deg": 0, "lean_deg": 15,
                "aspect": 0.6},              # standing framed mirror (FALCON's), pane facing the anchor camera
               {"type": "bubbles", "source": "fiat126p", "local": [x, y, z],  # gas bubbles streaming from a point
                "rate": 14, "rise": 0.6, "radius": [0.012, 0.03], ...},  # of a mesh (its normalized frame), see bubbles_at
               ...],                         # resting on the surface seen at image fraction (u, v) of the anchor view
   "frames": 120, "fps": 30,                  # the camera model's exposure is that of the anchor view
   "path": {"look_at_objects": 0.7,          # 0: the photos' own viewing direction, 1: aim at the objects
            "views": null,                   # views to pass through (default: all training views, along their main axis)
            "nearest": null,                 # or the N training views nearest the anchor, along their main axis
            "scale": 1.0,                    # image size factor
            "trim": [0.0, 1.0],              # part of the spline used (fractions), e.g. to skip a poorly covered end
            "zoom": 1.0},                    # focal length factor (> 1 crops the frame edges the photos missed)
   "trace": {...}}                           # TraceSettings overrides
The camera follows a Catmull-Rom spline through the chosen training camera positions (novel views stay close to
the photos), rolled level with the cameras' mean "up". Frames go to OUT/frames, the video to OUT.mp4, the resolved
shot (with object centres in world units) to OUT/shot_resolved.json.
"""
from __future__ import annotations

import argparse
import dataclasses
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
from vis.mesh import Mesh, load_manifest, mirror_panel, place_mesh_on_seabed  # noqa: E402
from vis.raytrace import SceneTracer, Sphere, TraceSettings, TraceStats, place_on_seabed, render_camera  # noqa: E402


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


def camera_path(trainer, template, target: torch.Tensor, frames: int, path: dict):
    """Cameras of a shot's path (see the module doc for the keys of `path`)."""
    look_at, views, nearest = path.get("look_at_objects", 0.7), path.get("views"), path.get("nearest")
    scale, zoom = path.get("scale", 1.0), path.get("zoom", 1.0)
    trim = path.get("trim", (0.0, 1.0))
    train = list(trainer.dataset.train)
    if views:
        by_name = {c.name: c for c in train + list(trainer.dataset.test)}
        train = [by_name[v] for v in views]
    else:  # along the main axis of the camera positions (photo names zigzag)
        if nearest:  # the training views closest to the anchor
            distance = torch.stack([(c.origin - template.origin).norm() for c in train])
            train = [train[i] for i in distance.argsort()[:nearest].tolist()]
        origins = torch.stack([c.origin for c in train])
        axis = torch.linalg.svd(origins - origins.mean(0)).Vh[0]
        train = [train[i] for i in ((origins - origins.mean(0)) @ axis).argsort().tolist()]
    # The spline is sampled densely and the part trim = (start, end) (fractions of it) is resampled to the frames.
    dense = max(frames, int(round(frames / max(trim[1] - trim[0], 1e-3))))
    keep = slice(int(round(trim[0] * (dense - 1))), int(round(trim[1] * (dense - 1))) + 1)
    origins = catmull_rom(torch.stack([c.origin for c in train]), dense)[keep]
    own = catmull_rom(torch.stack([c.forward for c in train]), dense)[keep]
    resample = lambda x: F.interpolate(x.T[None], size=frames, mode="linear", align_corners=True)[0].T
    origins, own = resample(origins), resample(own)
    origins = smooth(origins, passes=frames // 10)
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
            forward=f.tolist(), width=width, height=height, fx=zoom * record["fx"] * width / record["width"],
            fy=zoom * record["fy"] * height / record["height"], cx=record["cx"] * width / record["width"],
            cy=record["cy"] * height / record["height"]), trainer.device))
    return cameras


def build_objects(tracer, anchor, specs):
    """Objects of a shot and their resolved form (world placement) for shot_resolved.json."""
    objects, resolved = [], []
    for spec in specs:
        kind = spec.get("type", "sphere")
        if kind == "bubbles":  # per frame, see bubbles_at
            resolved.append(dict(spec))
            continue
        if kind in ("mesh", "mirror_panel"):
            if kind == "mesh":
                vertices, faces, part, materials, ids = load_manifest(spec["manifest"], spec.get("materials"))
            else:
                vertices, faces, part, materials, ids = mirror_panel(
                    spec.get("aspect", 0.6), spec.get("border", 0.06), spec.get("thickness", 0.04),
                    spec.get("lean_deg", 15.0), spec.get("mirror"), spec.get("frame"))
            world, info = place_mesh_on_seabed(tracer, anchor, spec["u"], spec["v"], spec["span_fraction"], vertices,
                                               spec.get("yaw_deg", 0.0), spec.get("sink", 0.0))
            device = tracer.trainer.device
            objects.append(Mesh(world, torch.as_tensor(faces, device=device), torch.as_tensor(part, device=device),
                                materials, spec.get("name", "mesh"), ids))
            resolved.append(dict(spec, placement=info, materials={i: vars(m) for i, m in zip(ids, materials)}))
            continue
        if "center" in spec:
            center, radius = spec["center"], spec["radius"]
        else:
            center, radius, _, moved = place_on_seabed(
                tracer, anchor, spec["u"], spec["v"], spec["radius_fraction"], spec.get("lift", 1.0),
                spec.get("height", 0.0), spec.get("depth_fraction"), spec.get("clear", True),
                spec.get("distance"))
            if moved:
                print(f"{spec.get('material', 'sphere')} sphere lifted {moved:.1f} radii out of the surface")
        sphere = Sphere(center, radius, spec.get("material", "mirror"), tuple(spec.get("tint", (1, 1, 1))),
                        spec.get("ior", 1.5 / 1.33), spec.get("kd", 1.0))
        objects.append(sphere)
        resolved.append(dict(type="sphere", center=center, radius=radius, material=sphere.material,
                             tint=list(sphere.tint), ior=sphere.ior, kd=sphere.kd))
    return objects, resolved


def bubbles_at(spec: dict, resolved: list, up, t: float) -> list:
    """Gas bubbles of a "bubbles" spec at time t (seconds): a steady stream from `local` (a point in the normalized
    frame of the mesh named `source`), pushed out along `direction` (same frame) and rising along the world "up" at
    `rise` mesh sizes per second (larger bubbles faster), wobbling sideways (`wobble` sizes, `wobble_hz`) and drifting
    apart (`drift` sizes per second). Bubble k is born at k / rate with
    a random radius in `radius` (mesh sizes, growing by `grow` per second as the pressure drops) and lives `life`
    seconds. Deterministic for a seed, so every frame of a clip shows the same bubbles further along."""
    mesh = next(r for r in resolved if r.get("name") == spec["source"])["placement"]
    size = mesh["size"]
    origin, axes = np.array(mesh["origin"]), np.array(mesh["axes"])
    source = origin + size * np.array(spec["local"]) @ axes
    push = np.array(spec.get("direction", [0.0, 0.0, 1.0])) @ axes
    push /= np.linalg.norm(push)
    up = np.asarray(up, dtype=np.float64)
    side = np.cross(up, push); side /= np.linalg.norm(side)
    back = np.cross(side, up)
    rate, life, rise = spec.get("rate", 14.0), spec.get("life", 4.0), spec.get("rise", 0.6) * size
    speed, settle = spec.get("speed", 0.5) * size, spec.get("settle", 0.4)
    r_lo, r_hi = spec.get("radius", (0.012, 0.03))
    wobble, hz, grow = spec.get("wobble", 0.03) * size, spec.get("wobble_hz", 1.5), spec.get("grow", 0.1)
    spheres = []
    for k in range(int(math.floor((t - life) * rate)), int(math.floor(t * rate)) + 1):
        age = t - k / rate
        if not 0.0 <= age <= life:
            continue
        rng = np.random.default_rng([spec.get("seed", 0), k & 0xFFFFFFFF])
        jitter, phase, scale = rng.normal(size=3) * spec.get("spread", 0.01) * size, rng.uniform(0, 2 * math.pi, 2), rng.uniform()
        drift = rng.normal(size=2) * spec.get("drift", 0.06) * size  # the plume widens as it rises
        radius = (r_lo + (r_hi - r_lo) * scale ** 2) * size * (1 + grow * age)
        climb = rise * (0.6 + 0.4 * math.sqrt(scale))  # larger bubbles rise faster
        out = speed * settle * (1 - math.exp(-age / settle))  # the exhaust jet slows down in the water
        sway = wobble * min(age, 1.0) * (math.sin(2 * math.pi * hz * age + phase[0]) * side
                                         + math.cos(2 * math.pi * hz * age + phase[1]) * back)
        center = source + jitter + out * push + climb * age * up + sway + age * (drift[0] * side + drift[1] * back)
        spheres.append(Sphere(tuple(center.tolist()), float(radius), "glass", tuple(spec.get("tint", (1.0, 1.0, 1.0))),
                              spec.get("ior", 1 / 1.333), 1.0, shadow=False))
    return spheres


def frame_objects(objects: list, specs: list, resolved: list, up, t: float) -> list:
    """The shot's objects plus its bubbles at time t."""
    extra = []
    for spec in specs:
        if spec.get("type") == "bubbles":
            extra += bubbles_at(spec, resolved, up, t)
    return objects + extra


def save_png(path: Path, image: torch.Tensor):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((image.clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8)).save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--scene", type=Path)
    parser.add_argument("--shot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, help="override the shot's frame count (previews)")
    parser.add_argument("--still", action="store_true", help="only the anchor view: OUT/still.png (+ no shadows)")
    parser.add_argument("--part", type=int, nargs=2, metavar=("K", "N"),
                        help="render only part K of N (contiguous frames, 0-based) and no video, for parallel jobs")
    parser.add_argument("--only", type=int, nargs="+", metavar="I",
                        help="render only these frames of the path (stills at the clip's camera and time), no video")
    args = parser.parse_args()
    shot = json.loads(args.shot.read_text())
    trainer = render.load_run(args.run.resolve(), "last", args.scene)
    anchor = {c.name: c for c in trainer.dataset.train + trainer.dataset.test}[shot["anchor_view"]]
    tracer = SceneTracer(trainer, anchor.origin)

    objects, resolved_objects = build_objects(tracer, anchor, shot["objects"])
    settings = TraceSettings(**shot.get("trace", {}))
    up = tracer.up.cpu().numpy()
    fps = shot.get("fps", 30)
    if args.still:
        objects = frame_objects(objects, shot["objects"], resolved_objects, up, shot.get("still_time", 3.0))
        stats = TraceStats()
        torch.cuda.synchronize(); start = time.time()
        _, full = render_camera(tracer, anchor, objects, settings, stats)
        torch.cuda.synchronize()
        print(f"still: {time.time() - start:.2f} s, rays per depth {stats.rays_per_depth}")
        save_png(args.out / "still.png", full)
        if settings.shadows:
            _, flat = render_camera(tracer, anchor, objects, dataclasses.replace(settings, shadows=False))
            save_png(args.out / "still_noshadow.png", flat)
        save_png(args.out / "still_reference.png", render_camera(tracer, anchor, [])[1])
        return
    target = torch.stack([torch.as_tensor(o.center, dtype=torch.float32, device=trainer.device) for o in objects]).mean(0)
    path = shot.get("path", {})
    frames = args.frames or shot.get("frames", 120)
    cameras = camera_path(trainer, anchor, target, frames, path)

    out_frames = args.out / "frames"
    out_frames.mkdir(parents=True, exist_ok=True)
    resolved = dict(shot, objects=resolved_objects, frames=frames,
                    run=str(args.run), cameras=[render.camera_record(c) for c in cameras])
    (args.out / "shot_resolved.json").write_text(json.dumps(resolved, indent=1))
    start = time.time()
    first, last = 0, frames
    if args.part:
        k, n = args.part
        first, last = k * frames // n, (k + 1) * frames // n
    for i, camera in enumerate(cameras):
        if not first <= i < last or (args.only and i not in args.only):
            continue
        now = frame_objects(objects, shot["objects"], resolved_objects, up, shot.get("start_time", 3.0) + i / fps)
        _, full = render_camera(tracer, camera, now, settings, isp_origin=anchor.origin)
        save_png(out_frames / f"{i:04d}.png", full)
        if (i - first) % 20 == 0:
            print(f"frame {i}/{frames}  {time.time() - start:.1f} s", flush=True)
    if args.part or args.only:
        print(f"frames {args.only or f'{first}-{last - 1}'} in {time.time() - start:.1f} s")
        return
    video = args.out.with_suffix(".mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(shot.get("fps", 30)), "-i",
                    str(out_frames / "%04d.png"), "-c:v", "libx264", "-crf", "16", "-preset", "slow",
                    "-pix_fmt", "yuv420p", str(video)], check=True)
    print(f"{frames} frames in {time.time() - start:.1f} s -> {video}")


if __name__ == "__main__":
    main()
