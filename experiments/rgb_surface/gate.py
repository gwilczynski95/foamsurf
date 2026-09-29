#!/usr/bin/env python3
"""Content-addressed required GPU tests plus historical RGB/scalar regressions."""
from pathlib import Path
import hashlib
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
import experiments.rgb_disentangle.gate
import warmup_gate

base_fingerprint = warmup_gate.fingerprint


def fingerprint():
    h = hashlib.sha256(base_fingerprint().encode())
    for p in sorted(Path(__file__).parent.rglob('*')):
        if p.is_file() and p.suffix in {'.py', '.yaml', '.sbatch', '.sh'}:
            h.update(str(p.relative_to(ROOT)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


warmup_gate.fingerprint = fingerprint
NEW_GROUPS = (
    ('surface_cpu', ('experiments/rgb_surface/test_cpu.py',)),
    ('surface_renderer', ('experiments/rgb_surface/test_gpu.py',)),
    ('surface_pipeline', ('experiments/rgb_surface/test_pipeline_gpu.py::test_pipeline_and_resume',)),
    ('surface_production', ('experiments/rgb_surface/test_pipeline_gpu.py::test_production',)),
)
warmup_gate.TEST_GROUPS = NEW_GROUPS + warmup_gate.TEST_GROUPS
previous_require = warmup_gate.require_certificate


def require(path):
    data = previous_require(path)
    for _, paths in NEW_GROUPS:
        for target in paths:
            if not any(target in command for command in data['commands']):
                raise RuntimeError('Missing required surface test: '+target)
    return data


warmup_gate.require_certificate = require
if __name__ == '__main__':
    warmup_gate.main()
