#!/usr/bin/env python3
"""Generate a partial-draining ocean animation from scratch for one or all scenes."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def commands(package, scene, out, width=960, view=None, preview=False, breaking=False, impact_strength=3., end_hold=0., dynamics=False, final_wave_scale=1., final_rise=0., rise_seconds=5., post_rise_hold=0.):
    run = package / 'models' / scene
    data = package / 'data' / scene
    if not (run / 'joint/checkpoints/last.pt').is_file() or not data.is_dir():
        raise ValueError(f'Missing checkpoint or dataset for {scene} in {package}')
    common = [sys.executable, str(ROOT / 'edit.py'), 'drain', '--run', str(run),
              '--scene', str(data), '--width', str(width)]
    if view:
        common += ['--view', view]
    base, smooth, final = out / 'calibration', out / 'smooth', out / 'animation'
    # Calibration needs every frame: these samples measure visual changes.
    return [
        common + ['--out', str(base), '--frames', '240'],
        common + ['--out', str(smooth), '--frames', '240', '--schedule', str(base / 'frame_stats.json')],
        common + ['--out', str(final), '--frames', '288', '--water-surface', '--final-rise', str(final_rise), '--rise-seconds', str(rise_seconds), '--post-rise-hold', str(post_rise_hold),
                  '--reference-levels', str(smooth / 'frame_stats.json')] + (['--preview'] if preview else []) + (['--breaking-waves', '--impact-strength', str(impact_strength), '--end-hold', str(end_hold), '--final-wave-scale', str(final_wave_scale)] if breaking or dynamics else []) + (['--surface-dynamics'] if dynamics else []),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True, help='directory containing models/ and data/')
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--scene', help='scene directory name, e.g. IUI3-RedSea-Undistorted')
    selection.add_argument('--all', action='store_true', help='all scenes with models and data in the package')
    parser.add_argument('--view', help='camera name; single scene only (default first test view)')
    parser.add_argument('--out', type=Path, default=Path('outputs/draining'))
    parser.add_argument('--width', type=int, default=960)
    parser.add_argument('--post-rise-hold', type=float, default=0.)
    parser.add_argument('--final-rise', type=float, default=0.)
    parser.add_argument('--rise-seconds', type=float, default=5.)
    parser.add_argument('--final-wave-scale', type=float, default=1.)
    parser.add_argument('--surface-dynamics', action='store_true', help='simulate reflecting surface waves on the reef')
    parser.add_argument('--impact-strength', type=float, default=3.)
    parser.add_argument('--end-hold', type=float, default=0., help='extra seconds at final water level')
    parser.add_argument('--breaking-waves', action='store_true', help='larger late waves, contact foam and spray')
    parser.add_argument('--preview', action='store_true', help='only five final frames; calibration still renders all frames')
    parser.add_argument('--dry-run', action='store_true', help='validate paths and print commands without rendering')
    args = parser.parse_args()
    if args.surface_dynamics and args.preview:
        parser.error("surface dynamics needs consecutive frames; omit --preview")
    if args.final_rise < 0 or args.rise_seconds <= 0 or args.post_rise_hold < 0:
        parser.error("final rise must be nonnegative and duration positive")
    if args.impact_strength <= 0 or args.end_hold < 0 or args.final_wave_scale <= 0:
        parser.error("impact strength must be positive and end hold nonnegative")
    if args.width < 2 or args.width % 2:
        parser.error('--width must be a positive even number >= 2')
    if args.all and args.view:
        parser.error('--view is only supported with --scene')
    package, out = args.package.resolve(), args.out.resolve()
    if args.all:
        scenes = sorted(p.name for p in (package / 'models').glob('*') if p.is_dir() and (package / 'data' / p.name).is_dir())
    else:
        scenes = [args.scene]
    if not scenes:
        parser.error('No scenes found')
    # Validate every scene before starting an expensive render.
    try:
        jobs = [(scene, commands(package, scene, out / scene, args.width, args.view, args.preview, args.breaking_waves, args.impact_strength, args.end_hold, args.surface_dynamics, args.final_wave_scale, args.final_rise, args.rise_seconds, args.post_rise_hold)) for scene in scenes]
    except ValueError as error:
        parser.error(str(error))
    if not args.dry_run and shutil.which('ffmpeg') is None:
        parser.error('ffmpeg is required on PATH')
    import shlex
    for scene, steps in jobs:
        for index, command in enumerate(steps, 1):
            print(f'{scene}: stage {index}/3\n{shlex.join(command)}', flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
