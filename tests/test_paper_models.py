"""Optional full-view regression against final paper checkpoints.

FS_PAPER_MODELS points to a JSON list of {"name": ..., "run": ..., "scene": ...}.
Each case loads the last checkpoint and compares all test views with final eval.csv.
"""
import csv
import gc
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from evaluation.metrics import psnr, ssim
from render import load_run, render_view

manifest = os.environ.get("FS_PAPER_MODELS")
cases = json.loads(Path(manifest).read_text()) if manifest else []


@pytest.mark.cuda
@pytest.mark.parametrize("case", cases, ids=[case["name"] for case in cases])
def test_final_checkpoint_matches_saved_evaluation(case):
    # OptiX owns allocations outside PyTorch. A separate process per model releases
    # the native context as well as tensors, so a suite cannot accumulate GPU memory.
    subprocess.run([sys.executable, str(Path(__file__).resolve()), json.dumps(case)], check=True)


def check_model(case):
    run = Path(case["run"])
    with (run / "joint/metrics/eval.csv").open() as stream:
        expected = list(csv.DictReader(stream))[-1]
    trainer = load_run(run, "last", Path(case["scene"]) if case.get("scene") else None)
    values = []
    with torch.no_grad():
        for camera in trainer.dataset.test:
            image = render_view(trainer, camera, ["full"])["full"]
            target = camera.ground_truth(tuple(trainer.config.background)).reshape_as(image)
            values.append((float(psnr(image, target)), float(ssim(image, target))))
    assert values
    measured = [sum(row[i] for row in values) / len(values) for i in range(2)]
    print(case["name"], "PSNR / SSIM", measured, flush=True)
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    assert abs(measured[0] - float(expected["eval/hybrid_psnr"])) <= 0.001
    assert abs(measured[1] - float(expected["eval/hybrid_ssim"])) <= 0.0001


if __name__ == "__main__":
    check_model(json.loads(sys.argv[1]))
