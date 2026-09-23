#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import warmup_gate

# Topology unit tests install a fake extension; never share that process with GPU tests.
warmup_gate.TEST_GROUPS = (
    ('contract', ('tests/joint_adaptation/test_contract.py', 'tests/b2/test_topology_replace_population.py')),
    ('joint_gpu', ('tests/joint_adaptation/test_gpu.py',)),
)
_require = warmup_gate.require_certificate


def require_certificate(path):
    data = _require(path)
    for target in ('tests/joint_adaptation/test_contract.py', 'tests/joint_adaptation/test_gpu.py'):
        if not any(target in command for command in data.get('commands', [])):
            raise RuntimeError(f'Certificate missing {target}')
    return data


warmup_gate.require_certificate = require_certificate
if __name__ == '__main__':
    warmup_gate.main()
