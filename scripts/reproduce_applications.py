#!/usr/bin/env python3
"""Plan and run the available paper applications without importing CUDA modules."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from preflight import check_inputs

SCENES = {'IUI3': ('iui3', 'IUI3-RedSea-Undistorted'),
          'Curasao': ('curasao', 'Curasao-Undistorted'),
          'JapaneseGradens': ('jg', 'JapaneseGradens-RedSea-Undistorted')}


def plan(args):
    python = os.environ.get('FOAMSURF_PYTHON', sys.executable)
    models = Path(os.environ.get('FS_MODELS', 'models'))
    assets = Path(os.environ.get('FS_ASSETS', 'assets'))
    data = Path(os.environ.get('FS_DATASETS', 'data/SeathruNeRF_dataset'))
    synthetic = Path(os.environ.get('FS_SYNTHETIC', 'data/synthetic'))
    out = args.out / 'applications'
    wanted = set(args.scenes.split(',')) if args.scenes else set()
    kinds = args.applications.split(',')
    unknown = set(kinds) - {'insertion', 'edits', 'style', 'teaser'}
    if unknown:
        raise ValueError(f'Unknown applications: {sorted(unknown)}')
    known = set(SCENES) | {v for pair in SCENES.values() for v in pair} | {'lego', 'ficus'}
    if wanted - known:
        raise ValueError(f'Unknown application scenes: {sorted(wanted - known)}')
    commands = []
    def add(*parts):
        commands.append([str(x) for x in (python, *parts)])
    def selected(*names):
        return not wanted or bool(wanted.intersection(names))
    if 'insertion' in kinds:
        for scene in ('lego', 'ficus'):
            if selected(scene):
                add('insert.py', '--benchmark', '--run', models / 'synthetic' / scene,
                    '--scene', synthetic / scene / f'{scene}_both', '--gt', assets / 'insertion' / scene,
                    '--out', out / 'insertion' / scene)
    if selected('IUI3', *SCENES['IUI3']):
        base = models / 'base/iui3'
        scene = data / SCENES['IUI3'][1]
        if 'edits' in kinds:
            add('edit.py', '--run', base, '--scene', scene, '--view', 'MTN_5911',
                '--edits', 'bulge,bend,haze,plume,relief,clear,murky,transplant,green',
                '--transplant-from', models / 'base/curasao', '--out', out / 'edits/iui3')
        if 'teaser' in kinds:
            add('insert.py', '--run', base, '--scene', scene, '--assets', assets,
                '--shot', 'apps/raytrace/shots/ruins/teaser.json', '--still', '--out', out / 'teaser/iui3')
    if 'style' in kinds:
        for key, (short, directory) in SCENES.items():
            if not selected(key, short, directory):
                continue
            style = out / 'style'
            for component, target in [('water', 'voronoi'), ('surface', 'gaussians')]:
                options = []
                if component == 'surface':
                    filename = {'iui3': 'fire.jpg', 'curasao': 'scream.jpg', 'jg': 'red_canna.jpg'}[short]
                    options = ['--style-image', assets / 'styles' / filename]
                elif short == 'iui3':
                    options = ['--style-image', assets / 'styles/starry_night.jpg']
                elif short == 'curasao':
                    options = ['--style-prompt', 'murky green jungle river water']
                else:
                    options = ['--style-prompt', 'bright orange fire flames', '--medium-backscatter-lr', '0']
                add('stylize.py', '--run', models / 'base' / short, '--scene', data / directory,
                    '--target', target, '--steps', '5000', '--seed', '0', *options,
                    '--out', style / f'{short}_{component}')
            add('-m', 'apps.style.compose_style', '--gaussians', style / f'{short}_surface',
                '--medium', style / f'{short}_water', '--out', style / f'{short}_both')
            add('render.py', '--run', style / f'{short}_both', '--scene', data / directory,
                '--modes', 'full,gaussians,voronoi', '--out', style / f'{short}_both/renders')
    if not commands:
        raise ValueError('No applications match the selected scenes')
    return commands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--applications', default='insertion,edits,style,teaser')
    parser.add_argument('--scenes', default='')
    parser.add_argument('--out', type=Path, default=Path('outputs/paper'))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    try:
        commands = plan(args)
        if not args.dry_run or args.check:
            check_inputs(commands)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    for command in commands:
        print(shlex.join(command), flush=True)
    if args.check or args.dry_run:
        return
    folder = args.out / 'applications'
    folder.mkdir(parents=True, exist_ok=True)
    report = {'commands': commands, 'completed': 0, 'status': 'running'}
    manifest = folder / 'execution.json'
    def save():
        manifest.write_text(json.dumps(report, indent=2) + '\n')
    save()
    try:
        for command in commands:
            subprocess.run(command, check=True)
            report['completed'] += 1
            save()
    except BaseException:
        report['status'] = 'failed'
        save()
        raise
    report['status'] = 'complete'
    report['insertion_metrics'] = {
        str(path.relative_to(folder)): json.loads(path.read_text())
        for path in sorted((folder / 'insertion').glob('*/metrics.json'))
        if any(str(path.parent) in command for command in commands)
    }
    save()
    print(f'Wrote {manifest}; visual applications do not have quantitative scores.')


if __name__ == '__main__':
    main()
