"""Checks of vis/raytrace.py on a trained run:
1. no objects: the tracer equals render.py's raw and full images (bit-exact path through the limited kernel);
2. an object far behind every surface changes nothing;
3. a preview with a mirror and a glass sphere resting on the seabed in front of a test view.

    python vis/check_raytrace.py --run RUN --scene DATA --view MTN_1296 --out renders/raytrace_check
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import render  # noqa: E402
from vis.raytrace import place_on_seabed, SceneTracer, Sphere, TraceSettings, TraceStats, render_camera  # noqa: E402


def save(path: Path, image: torch.Tensor):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((image.clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8)).save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--scene", type=Path)
    parser.add_argument("--view", default=None)
    parser.add_argument("--out", type=Path, default=Path("renders/raytrace_check"))
    args = parser.parse_args()
    trainer = render.load_run(args.run.resolve(), "last", args.scene)
    cameras = {c.name: c for c in trainer.dataset.test}
    camera = cameras[args.view] if args.view else trainer.dataset.test[0]
    tracer = SceneTracer(trainer, camera.origin)

    reference = render.render_view(trainer, camera, ["raw", "full"])
    tracer.set_color_origin(camera.origin)
    raw, full = render_camera(tracer, camera, [])
    print(f"no objects: max |raw - render.py raw| = {(raw - reference['raw']).abs().max():.3e}, "
          f"max |full - render.py full| = {(full - reference['full']).abs().max():.3e}")

    behind = Sphere(center=(camera.origin + 1e4 * camera.forward).tolist(), radius=1.0)
    raw_far, _ = render_camera(tracer, camera, [behind], TraceSettings(antialias=0))
    print(f"sphere behind everything: max |raw - raw| = {(raw_far - reference['raw']).abs().max():.3e}")

    mirror_c, mirror_r, d1, _ = place_on_seabed(tracer, camera, 0.40, 0.72, 0.07)
    glass_c, glass_r, d2, _ = place_on_seabed(tracer, camera, 0.64, 0.68, 0.07)
    print(f"mirror at depth {d1:.3f} r {mirror_r:.3f}; glass at depth {d2:.3f} r {glass_r:.3f}")
    objects = [Sphere(mirror_c, mirror_r, "mirror", (0.95, 0.95, 0.95)),
               Sphere(glass_c, glass_r, "glass", (0.98, 0.99, 1.0), ior=1.5 / 1.33)]
    stats = TraceStats()
    torch.cuda.synchronize(); start = time.time()
    raw_obj, full_obj = render_camera(tracer, camera, objects, TraceSettings(max_depth=4), stats)
    torch.cuda.synchronize()
    print(f"spheres: {time.time() - start:.2f} s, rays per depth {stats.rays_per_depth}")
    save(args.out / f"{camera.name}_reference.png", reference["full"])
    save(args.out / f"{camera.name}_spheres.png", full_obj)
    _, no_shadow = render_camera(tracer, camera, objects, TraceSettings(max_depth=4, shadows=False))
    save(args.out / f"{camera.name}_spheres_noshadow.png", no_shadow)
    print(f"images in {args.out}")


if __name__ == "__main__":
    main()
