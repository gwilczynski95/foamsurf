"""Real native handoff followed by joint adaptation, executed on RTX4090 only."""
from pathlib import Path
import importlib.util
import subprocess
import sys
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_joint_adaptation_from_native_handoff(tmp_path):
    assert torch.cuda.is_available(), 'Submit scripts/test_joint_adaptation.sbatch'
    spec = importlib.util.spec_from_file_location('e2e_smoke', ROOT / 'tests/end_to_end/test_pipeline_gpu.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    wrapper = module.write_smoke_config(tmp_path)
    subprocess.run([sys.executable, str(ROOT / 'scripts/train_end_to_end.py'), '--config', str(wrapper)], cwd=ROOT, check=True)
    source = tmp_path / 'run/joint/checkpoints/initialization.pt'
    # Isolate each renderer process and check all seven paths, including controls.
    for variant in range(7):
        subprocess.run([sys.executable, str(ROOT / 'tests/joint_adaptation/gpu_worker.py'),
                        str(source), str(tmp_path / f'variant_{variant}'), str(variant)], cwd=ROOT, check=True)
