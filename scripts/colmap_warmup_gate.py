#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import warmup_gate
warmup_gate.TEST_GROUPS=(
    ('colmap_cpu', ('tests/colmap_warmup/test_contract.py', 'tests/end_to_end/test_contract.py',
                    'tests/warmup_radfoam_like/test_profile.py','tests/b1/test_full_image_sampler.py')),
    ('colmap_gpu', ('tests/colmap_warmup/test_gpu.py',)),
    ('legacy_gpu', ('tests/warmup_radfoam_like/test_integration_gpu.py',
                    'tests/warmup_radfoam_like/test_organic_gpu.py','tests/end_to_end/test_pipeline_gpu.py')),
)
_require=warmup_gate.require_certificate


def require_certificate(path):
    data=_require(path)
    for target in ('tests/colmap_warmup/test_gpu.py','tests/end_to_end/test_pipeline_gpu.py'):
        if not any(target in command for command in data.get('commands',[])):
            raise RuntimeError(f'Missing tests: {target}')
    return data


warmup_gate.require_certificate=require_certificate
if __name__=='__main__': warmup_gate.main()
