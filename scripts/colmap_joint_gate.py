#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import warmup_gate
warmup_gate.TEST_GROUPS=(
    ('contracts',('tests/colmap_joint/test_contract.py','tests/colmap_joint/test_lineage.py','tests/end_to_end/test_contract.py',
                  'tests/warmup_radfoam_like/test_profile.py','tests/colmap_warmup/test_contract.py',
                  'tests/b3/test_disentanglement_losses.py')),
    ('joint_contract',('tests/joint_adaptation/test_contract.py',)),
    ('new_gpu',('tests/colmap_joint/test_gpu.py',)),
    ('production_gpu',('tests/colmap_joint/test_production_gpu.py',)),
    ('legacy_colmap',('tests/colmap_warmup/test_gpu.py::test_real_colmap_pipeline',)),
    ('legacy_restore',('tests/colmap_warmup/test_gpu.py::test_pcd_checkpoint_restores_physical_field',)),
    ('legacy_production',('tests/colmap_warmup/test_gpu.py::test_full_resolution_million_ray_growth',)),
    ('legacy_pipeline',('tests/end_to_end/test_pipeline_gpu.py',)),
)
original=warmup_gate.require_certificate
def require(path):
    data=original(path)
    for target in ('tests/colmap_joint/test_gpu.py','tests/colmap_joint/test_production_gpu.py'):
        if not any(target in c for c in data.get('commands',[])):
            raise RuntimeError('Certificate does not include '+target)
    return data
warmup_gate.require_certificate=require
if __name__=='__main__': warmup_gate.main()
