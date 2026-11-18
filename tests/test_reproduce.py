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
