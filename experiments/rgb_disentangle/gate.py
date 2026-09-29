#!/usr/bin/env python3
"""Certify the new renderer/loss path AND all existing RGB/scalar regressions."""
from pathlib import Path
import hashlib
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import experiments.rgb_smooth.gate
import warmup_gate

base_fingerprint=warmup_gate.fingerprint
def fingerprint():
    h=hashlib.sha256(base_fingerprint().encode())
    for folder in ('experiments/rgb_disentangle','experiments/ray_heatmaps'):
        for p in sorted((ROOT/folder).rglob('*')):
            if p.is_file() and p.suffix in {'.py','.yaml','.sbatch','.cpp'}:
                h.update(str(p.relative_to(ROOT)).encode());h.update(p.read_bytes())
    return h.hexdigest()
warmup_gate.fingerprint=fingerprint
NEW_GROUPS=(
    ('conflict_cpu',('experiments/rgb_disentangle/test_losses.py',)),
    ('conflict_renderer',('experiments/rgb_disentangle/test_renderer_gpu.py',)),
    ('conflict_pipeline',('experiments/rgb_disentangle/test_pipeline_gpu.py::test_pipeline_resume_and_isolation',)),
    ('conflict_production',('experiments/rgb_disentangle/test_pipeline_gpu.py::test_full_resolution',)),
)
warmup_gate.TEST_GROUPS=NEW_GROUPS+warmup_gate.TEST_GROUPS
previous=warmup_gate.require_certificate
def require(path):
    data=previous(path)
    for _,targets in NEW_GROUPS:
        for target in targets:
            if not any(target in command for command in data['commands']):
                raise RuntimeError('Missing required conflict test: '+target)
    return data
warmup_gate.require_certificate=require
if __name__=='__main__':warmup_gate.main()
