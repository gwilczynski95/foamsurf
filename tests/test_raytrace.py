"""G5: optional bit-exact tracer checks on the final SeaThru checkpoints."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = os.environ.get("FS_PAPER_WORKSPACE")
SCENES = ["IUI3-RedSea-Undistorted", "Curasao-Undistorted", "JapaneseGradens-RedSea-Undistorted", "Panama-Undisorted"]


@pytest.mark.cuda
@pytest.mark.skipif(not WORKSPACE, reason="set FS_PAPER_WORKSPACE to original paper checkpoints")
@pytest.mark.parametrize("scene", SCENES)
def test_tracer_matches_renderer(scene):
    run = Path(WORKSPACE) / "foamsurf/outputs/seathru/561669" / f"{scene}_cull05r_bin01_wphase_rgbs15_wext"
    subprocess.run([sys.executable, str(Path(__file__).resolve()), str(run)], cwd=ROOT, check=True)


def check(run):
    import torch
    from render import load_run, render_view
    from apps.raytrace.tracer import SceneTracer, render_camera
    from apps.raytrace.shading import Sphere, TraceSettings
    trainer = load_run(Path(run))
    camera = trainer.dataset.test[0]
    tracer = SceneTracer(trainer, camera.origin)
    with torch.no_grad():
        reference = render_view(trainer, camera, ["raw", "full"])
        raw, full = render_camera(tracer, camera, [])
        errors = {"raw": float((raw - reference["raw"]).abs().max()),
                  "full": float((full - reference["full"]).abs().max())}
        behind = Sphere(center=(camera.origin + 1e4 * camera.forward).tolist(), radius=1.0)
        raw_far, full_far = render_camera(tracer, camera, [behind], TraceSettings(antialias=0))
        errors.update(far_raw=float((raw_far - reference["raw"]).abs().max()),
                      far_full=float((full_far - reference["full"]).abs().max()))
    print(run, json.dumps(errors), flush=True)
    assert all(error == 0 for error in errors.values()), errors


if __name__ == "__main__":
    check(sys.argv[1])
