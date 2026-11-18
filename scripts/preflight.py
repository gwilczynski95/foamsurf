#!/usr/bin/env python3
"""Check selected reproduction inputs before training; no CUDA or third-party imports."""
import json
from pathlib import Path
import shlex
import sys


def check_inputs(commands):
    if not any(len(cmd) > 1 and cmd[1] in ('train.py', 'insert.py', 'edit.py', 'stylize.py') for cmd in commands):
        raise ValueError('No experiments match the selected scenes')
    errors = set()
    produced = set()
    def require(path, directory=False):
        path = Path(path)
        if not (path.is_dir() if directory else path.is_file()):
            errors.add(f'Missing {"directory" if directory else "file"}: {path}')
    def model(path):
        require(Path(path) / 'joint/checkpoints/last.pt')
        require(Path(path) / 'joint/config.yaml')
    def dataset(path, splits=('train', 'test')):
        path = Path(path)
        if path in produced:
            return
        if (path / 'transforms_train.json').is_file() or (path / 'transforms_test.json').is_file():
            for split in splits:
                metadata = path / f'transforms_{split}.json'
                require(metadata)
                if metadata.is_file():
                    try:
                        frames = json.loads(metadata.read_text())['frames']
                        if not frames:
                            errors.add(f'Empty camera list: {metadata}')
                        for frame in frames:
                            image = path / frame['file_path']
                            if not image.is_file():
                                require(image.with_suffix('.png'))
                    except (ValueError, KeyError, TypeError):
                        errors.add(f'Invalid Blender metadata: {metadata}')
        else:
            require(path / 'images', directory=True)
            sparse = path / 'sparse/0' if (path / 'sparse/0').is_dir() else path / 'sparse'
            for name in ('cameras', 'images', 'points3D'):
                if not any((sparse / f'{name}.{ext}').is_file() for ext in ('bin', 'txt')):
                    errors.add(f'Missing COLMAP {name}.bin/.txt: {sparse}')
    for cmd in commands:
        def option(flag):
            return cmd[cmd.index(flag) + 1] if flag in cmd else None
        entry = cmd[1] if len(cmd) > 1 else ''
        if entry == 'scripts/make_foggy_garden.py':
            for flag in ('--depth-dir', '--clear-dir', '--colmap-model'):
                require(option(flag), directory=True)
            produced.update(Path(option('--output-dir')) / f'garden_{level}' for level in ('easy', 'hard'))
        if option('--scene'):
            dataset(option('--scene'))
        if option('--run') and Path(option('--run')) not in produced:
            model(option('--run'))
        if option('--transplant-from'):
            model(option('--transplant-from'))
        for flag in ('--masks', '--style-image', '--shot'):
            if option(flag):
                require(option(flag))
        if option('--gt'):
            gt = Path(option('--gt'))
            for name in ('insert.json', 'transforms_test.json'):
                require(gt / name)
            for name in ('gt_rgba', 'masks'):
                require(gt / name, directory=True)
            if (gt / 'insert.json').is_file():
                for view in json.loads((gt / 'insert.json').read_text()).get('views', []):
                    require(gt / 'gt_rgba' / f'{view}.png')
                    require(gt / 'masks' / f'{view}.png')
        if option('--shot') and Path(option('--shot')).is_file():
            shot = Path(option('--shot'))
            spec = json.loads(shot.read_text())
            require(shot.parent / spec['camera'])
            for obj in spec.get('objects', []):
                if 'manifest' in obj:
                    manifest = Path(option('--assets') or 'assets') / obj['manifest']
                    require(manifest)
                    if manifest.is_file():
                        mesh = json.loads(manifest.read_text())
                        if mesh.get('format') == 'npz':
                            require(manifest.parent / mesh['mesh'])
                        for part in mesh.get('parts', []):
                            if 'ply' in part:
                                require(manifest.parent / part['ply'])
                            for texture in part.get('material', {}).get('maps', {}).values():
                                require(manifest.parent / texture)
        for index, arg in enumerate(cmd):
            if arg == '--set' and 'ground_truth_path=' in cmd[index + 1]:
                dataset(cmd[index + 1].split('=', 1)[1], splits=('test',))
        if entry in ('train.py', 'stylize.py') or 'apps.style.compose_style' in cmd:
            produced.add(Path(option('--out')))
    if errors:
        raise ValueError('Input preflight failed:\n' + '\n'.join(sorted(errors)))


if __name__ == '__main__':
    try:
        check_inputs([shlex.split(line) for line in sys.stdin if line.strip()])
    except (ValueError, OSError) as error:
        sys.exit(str(error))
    print('Input preflight passed.', file=sys.stderr)
