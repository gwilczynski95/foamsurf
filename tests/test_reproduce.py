"""Exercise emitted training commands through the real configuration parser."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]


def test_all_training_commands_resolve(tmp_path):
    data = tmp_path / 'data'
    for scene in ['IUI3-RedSea-Undistorted', 'Curasao-Undistorted', 'JapaneseGradens-RedSea-Undistorted', 'Panama-Undisorted']:
        (data / 'seathru' / scene / 'sparse').mkdir(parents=True)
    for level in ['easy', 'hard']:
        (data / 'garden' / f'garden_{level}' / 'sparse').mkdir(parents=True)
    for scene in ['lego', 'ficus']:
        p = data / 'synthetic' / scene / f'{scene}_both'
        p.mkdir(parents=True)
        (p / 'transforms_train.json').write_text('{}')
    env = dict(os.environ, FOAMSURF_PYTHON=sys.executable, FS_DATASETS=str(data / 'seathru'),
               FS_FOGGY=str(data / 'garden'), FS_SYNTHETIC=str(data / 'synthetic'))
    env.pop('FS_FOG_DEPTHS', None)
    output = subprocess.check_output(['bash', 'scripts/reproduce.sh', 'all', '--dry-run', '--seeds', '42',
                                      '--pipeline', 'all', '--out', str(tmp_path / 'runs')], cwd=ROOT, env=env, text=True)
    commands = [shlex.split(line) for line in output.splitlines()]
    training = [cmd for cmd in commands if len(cmd) > 1 and cmd[1] == 'train.py']
    configs = {cmd[cmd.index('--config') + 1] for cmd in training}
    assert {'configs/seathru.yaml', 'configs/garden_fog.yaml', 'configs/synthetic_hybrid.yaml',
            'configs/synthetic_voronoi.yaml'} <= configs
    assert len([c for c in configs if 'ablations/' in c]) == 7
    assert training and all('--resume' in cmd for cmd in training)
    assert all('best' not in ' '.join(cmd) for cmd in commands)
    def resolve(cmd):
        result = subprocess.run(cmd + ['--dry-run'], cwd=ROOT, env=env, text=True, capture_output=True)
        assert result.returncode == 0, (cmd, result.stderr[-4000:])
        assert json.loads(result.stdout)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(resolve, training))
    assert not (tmp_path / 'runs').exists()


def test_slurm_submission_preserves_arguments_and_dry_run_does_not_submit(tmp_path):
    import os
    import shlex
    import subprocess
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    fake_bin = tmp_path / 'bin'
    fake_bin.mkdir()
    marker = tmp_path / 'called'
    fake = fake_bin / 'sbatch'
    fake.write_text('#!/bin/sh\ntouch "$SUBMIT_MARKER"\n')
    fake.chmod(0o755)
    env = dict(os.environ, PATH=str(fake_bin) + ':' + os.environ['PATH'], SUBMIT_MARKER=str(marker))
    args = ['bash', 'scripts/reproduce.sh', 'seathru', '--scenes', 'IUI3', '--seeds', '42', '--out', str(tmp_path / 'with spaces'), '--slurm']
    result = subprocess.run(args + ['--dry-run'], cwd=root, env=env, text=True, capture_output=True, check=True)
    command = shlex.split(result.stdout)
    assert command[0] == 'sbatch'
    assert command[-len(args[2:-1]):] == args[2:-1]
    assert not marker.exists()
    subprocess.run(args, cwd=root, env=env, check=True, capture_output=True)
    assert marker.exists()
