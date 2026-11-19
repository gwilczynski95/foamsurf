"""CPU-only workflow checks: fail before expensive work and preserve paper presets."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('reproduction_preflight', ROOT / 'scripts/preflight.py')
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


def test_preflight_checks_all_commands_and_blender_images(tmp_path):
    scene = tmp_path / 'scene'
    scene.mkdir()
    for split in ('train', 'test'):
        (scene / f'transforms_{split}.json').write_text(json.dumps({'frames': [{'file_path': 'missing'}]}))
    commands = [['python', 'train.py', '--scene', str(scene), '--out', str(tmp_path / 'run')],
                ['python', 'evaluate.py', '--run', str(tmp_path / 'run'), '--masks', str(tmp_path / 'mask.npz')]]
    with pytest.raises(ValueError) as caught:
        preflight.check_inputs(commands)
    assert 'missing.png' in str(caught.value)
    assert 'mask.npz' in str(caught.value)
    assert 'last.pt' not in str(caught.value)  # Created by the planned training command.
    (scene / 'missing.png').touch()
    (tmp_path / 'mask.npz').touch()
    preflight.check_inputs(commands)
    assert not (tmp_path / 'run').exists()
