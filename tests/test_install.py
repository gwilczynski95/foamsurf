"""Portable build environment and protection of existing Python environments."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def build_env(tmp_path, **overrides):
    sdk = tmp_path / 'optix'
    (sdk / 'include').mkdir(parents=True)
    (sdk / 'include/optix.h').touch()
    env = {k: v for k, v in os.environ.items() if k not in (
        'CONDA_ENV_PATH', 'CONDA_PREFIX', 'PYTHON', 'CUDA_HOME', 'CUDA_ARCH',
        'OPTIX_DIRECTORY', 'FOAMSURF_OPTIX_DIRECTORY')}
    env.update(FOAMSURF_PYTHON=sys.executable, OPTIX_ROOT=str(sdk), TORCH_CUDA_ARCH_LIST='8.9')
    env.update(overrides)
    script = 'source scripts/build_env.sh\n"$PYTHON" -c \'import os,json; print(json.dumps(dict(os.environ)))\''
    return subprocess.run(['bash', '-ec', script], cwd=ROOT, env=env, text=True, capture_output=True)


def test_build_uses_selected_python_and_gpu_target(tmp_path):
    result = build_env(tmp_path)
    assert result.returncode == 0, result.stderr
    env = json.loads(result.stdout)
    assert env['CONDA_ENV_PATH'] == sys.prefix
    assert env['CUDA_HOME'] == sys.prefix
    assert env['CUDA_ARCH'] == '89'
    assert env['OPTIX_DIRECTORY'] == str(tmp_path / 'optix')
    assert env['LD_LIBRARY_PATH'].split(':')[0] == env['FOAMSURF_DRIVER_LIBRARY_DIRECTORY']


@pytest.mark.parametrize('arches,target,success', [('8.6+PTX', '', True), ('8.6;8.9', '', False), ('8.6;8.9', '89', True)])
def test_optix_target_is_unambiguous(tmp_path, arches, target, success):
    result = build_env(tmp_path, TORCH_CUDA_ARCH_LIST=arches, CUDA_ARCH=target)
    assert (result.returncode == 0) == success, result.stderr
    if success:
        assert json.loads(result.stdout)['CUDA_ARCH'] == (target or '86')
    else:
        assert 'Set CUDA_ARCH' in result.stderr


def test_install_refuses_existing_environment_before_running_conda(tmp_path):
    marker = tmp_path / 'keep'
    marker.write_text('existing environment')
    result = subprocess.run(['bash', 'scripts/install.sh', '--prefix', str(tmp_path)], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 1
    assert 'already exists' in result.stderr
    assert marker.read_text() == 'existing environment'
