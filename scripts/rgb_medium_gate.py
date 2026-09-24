#!/usr/bin/env python3
"""Independent RGB and legacy suites, each isolated from incompatible OptiX ABIs."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import colmap_joint_gate
import warmup_gate

warmup_gate.TEST_GROUPS = (
    ('rgb_math', ('tests/rgb_medium/test_renderer.py','tests/rgb_medium/test_model.py','tests/rgb_medium/test_config.py')),
    ('rgb_pipeline', ('tests/rgb_medium/test_pipeline.py::test_rgb_pipeline_and_resume',)),
    ('rgb_production', ('tests/rgb_medium/test_pipeline.py::test_rgb_production',)),
    ('legacy_renderer', ('renderer/tests/test_native_hybrid_renderer.py','renderer/tests/test_b01_diagnostics.py')),
) + warmup_gate.TEST_GROUPS
previous=warmup_gate.require_certificate
def require(path):
    data=previous(path)
    for target in ('tests/rgb_medium/test_renderer.py',
                   'tests/rgb_medium/test_pipeline.py::test_rgb_production',
                   'tests/rgb_medium/test_pipeline.py::test_rgb_pipeline_and_resume'):
        if not any(target in command for command in data['commands']):
            raise RuntimeError('Certificate missing '+target)
    return data
warmup_gate.require_certificate=require
if __name__=='__main__':warmup_gate.main()
