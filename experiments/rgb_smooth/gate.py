#!/usr/bin/env python3
from pathlib import Path
import hashlib
import sys
ROOT=Path(__file__).resolve().parents[2]
EXT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT))
import rgb_medium_gate
import warmup_gate

base_fingerprint=warmup_gate.fingerprint
def fingerprint():
    h=hashlib.sha256(base_fingerprint().encode())
    for p in sorted(EXT.rglob('*')):
        if p.is_file() and p.suffix in {'.py','.yaml','.sbatch'}:
            h.update(str(p.relative_to(ROOT)).encode());h.update(p.read_bytes())
    return h.hexdigest()
warmup_gate.fingerprint=fingerprint
warmup_gate.TEST_GROUPS=(
    ('smooth_math',('experiments/rgb_smooth/test_loss.py',)),
    ('smooth_pipeline',('experiments/rgb_smooth/test_pipeline.py::test_pipeline_resume_and_integration',)),
    ('smooth_production',('experiments/rgb_smooth/test_pipeline.py::test_production',)),
)+warmup_gate.TEST_GROUPS
previous=warmup_gate.require_certificate
def require(path):
    c=previous(path)
    for target in ('experiments/rgb_smooth/test_loss.py',
                   'experiments/rgb_smooth/test_pipeline.py::test_pipeline_resume_and_integration',
                   'experiments/rgb_smooth/test_pipeline.py::test_production'):
        if not any(target in command for command in c['commands']):raise RuntimeError('Missing smoothing tests')
    return c
warmup_gate.require_certificate=require
if __name__=='__main__':warmup_gate.main()
