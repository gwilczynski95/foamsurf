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


def test_applications_dry_run_presets_and_check_without_cuda(tmp_path):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', FOAMSURF_PYTHON=sys.executable,
               FS_MODELS=str(tmp_path / 'missing models'), FS_ASSETS=str(tmp_path / 'assets'),
               FS_DATASETS=str(tmp_path / 'data'), FS_SYNTHETIC=str(tmp_path / 'synthetic'))
    args = ['bash', 'scripts/reproduce.sh', 'applications', '--out', str(tmp_path / 'out')]
    result = subprocess.run(args + ['--dry-run'], cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.count('stylize.py') == 6
    assert result.stdout.count('apps.style.compose_style') == 3
    assert result.stdout.count('--benchmark') == 2
    assert '--medium-backscatter-lr 0' in result.stdout
    assert '--transplant-from' in result.stdout
    assert '--shot' not in result.stdout
    demo = subprocess.run(args + ['--applications', 'teaser', '--dry-run'], cwd=ROOT, env=env,
                          capture_output=True, text=True)
    assert demo.returncode == 0, demo.stderr
    assert '--shot' in demo.stdout
    assert 'ficus_both' in result.stdout
    check = subprocess.run(args + ['--check'], cwd=ROOT, env=env, capture_output=True, text=True)
    assert check.returncode != 0
    assert 'Input preflight failed' in check.stderr
    assert 'fire.jpg' in check.stderr and 'last.pt' in check.stderr
    assert not (tmp_path / 'out').exists()
    submission = subprocess.run(args + ['--slurm', '--dry-run'], cwd=ROOT, env=env,
                                capture_output=True, text=True)
    assert submission.returncode == 0, submission.stderr
    assert submission.stdout.startswith('sbatch ')
    assert 'applications' in submission.stdout


def test_invalid_application_selection_fails_even_in_dry_run():
    env = dict(os.environ, FOAMSURF_PYTHON=sys.executable, CUDA_VISIBLE_DEVICES='')
    for extra in (['--applications', 'draining'], ['--scenes', 'typo'], ['--seeds', '42']):
        result = subprocess.run(['bash', 'scripts/reproduce.sh', 'applications', '--dry-run', *extra],
                                cwd=ROOT, env=env, capture_output=True, text=True)
        assert result.returncode != 0


def test_application_execution_manifest_records_failure_and_success(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    import reproduce_applications as apps
    commands = [['python', 'first'], ['python', 'second']]
    monkeypatch.setattr(apps, 'plan', lambda args: commands)
    monkeypatch.setattr(apps, 'check_inputs', lambda commands: None)
    monkeypatch.setattr(sys, 'argv', ['reproduce_applications.py', '--out', str(tmp_path)])
    called = []
    def run(command, check):
        called.append(command)
        if len(called) == 2:
            raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(apps.subprocess, 'run', run)
    with pytest.raises(subprocess.CalledProcessError):
        apps.main()
    manifest = tmp_path / 'applications/execution.json'
    report = json.loads(manifest.read_text())
    assert report['status'] == 'failed' and report['completed'] == 1
    assert report['commands'] == commands
    monkeypatch.setattr(apps.subprocess, 'run', lambda command, check: None)
    apps.main()
    report = json.loads(manifest.read_text())
    assert report['status'] == 'complete' and report['completed'] == 2
    assert report['insertion_metrics'] == {}
