#!/usr/bin/env python3
"""Require warmup regressions and real native-to-hybrid GPU integration."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import warmup_gate

warmup_gate.TEST_GROUPS = tuple(
    (name, paths + ('tests/end_to_end', 'tests/test_surface_partition.py',
                    'tests/b3/test_radfoam_handoff.py') if name == 'warmup' else paths)
    for name, paths in warmup_gate.TEST_GROUPS
)

_require_certificate = warmup_gate.require_certificate


def require_end_to_end_certificate(path):
    data = _require_certificate(path)
    if not any('tests/end_to_end' in command for command in data.get('commands', [])):
        raise RuntimeError('Certificate did not include end-to-end integration tests')
    return data


warmup_gate.require_certificate = require_end_to_end_certificate

if __name__=='__main__':
    warmup_gate.main()
