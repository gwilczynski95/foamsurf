#!/usr/bin/env python3
"""CPU report from saved evaluations; full-frame coverage is not floor coverage."""
from pathlib import Path
import argparse
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


def read_csv(path):
    if not path.is_file():
        return []
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if rows:
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def gather(root, iteration, masks=None):
    rows = []
    for directory in sorted(root.glob('run_*'), key=lambda p: int(p.name.split('_')[-1])):
        if not (directory/'pipeline_config.json').exists():
            continue
        m = json.loads((directory/'pipeline_config.json').read_text())
        v = m['variant']
        at = directory/'joint/evaluation'/f'iteration_{iteration:06d}'
        per_view, coverage = read_csv(at/'per_view.csv'), read_csv(at/'coverage.csv')
        curve = read_csv(directory/'joint/metrics/eval.csv')
        if not per_view or not coverage:
            continue
        row = dict(run=m['index'], background=v['background'], overlap=v['overlap_weight'],
                   tv=v['tv_multiplier'], seed=v['seed'], iteration=iteration,
                   complete=(directory/'joint_done.json').is_file(),
                   hybrid_psnr=float(np.mean([float(r['hybrid_psnr']) for r in per_view])),
                   best_psnr=max((float(r['eval/hybrid_psnr']) for r in curve), default=float('nan')),
                   alpha_mean=float(np.mean([float(r['alpha_mean']) for r in coverage])),
                   coverage_gt_09=float(np.mean([float(r['coverage_gt_0.9']) for r in coverage])),
                   gaussian_black_psnr=float(np.mean([float(r['gaussian_black_psnr']) for r in coverage])),
                   floor_holes_01=float('nan'), floor_coverage_09=float('nan'), floor_views=0)
        holes, opaque = [], []
        if masks:
            for path in sorted(at.glob('*_gaussian_alpha.npz')):
                name = path.name.split('_', 1)[1].removesuffix('_gaussian_alpha.npz')
                mask_path = masks/f'{name}.png'
                if not mask_path.is_file():
                    continue
                alpha = np.load(path)['alpha'][..., 0]
                mask = np.asarray(Image.open(mask_path).convert('L')) > 127
                if mask.shape != alpha.shape or not mask.any():
                    raise ValueError(f'Empty/mismatched floor mask: {mask_path}')
                holes.append(float(np.mean(alpha[mask] <= .1)))
                opaque.append(float(np.mean(alpha[mask] > .9)))
            if holes:
                row.update(floor_holes_01=float(np.mean(holes)), floor_coverage_09=float(np.mean(opaque)), floor_views=len(holes))
        rows.append(row)
    return rows


def write_report(root, destination, iteration, masks=None, view_index=0):
    rows = gather(root, iteration, masks)
    if not rows:
        raise ValueError(f'No coverage evaluations at iteration {iteration}')
    destination.mkdir(parents=True, exist_ok=True)
    write_csv(destination/'runs.csv', rows)
    metrics = [('hybrid_psnr', 'Hybrid PSNR'), ('coverage_gt_09', 'A_G > 0.9, whole image')]
    if any(r['floor_views'] for r in rows):
        metrics.append(('floor_holes_01', 'Floor holes: A_G <= 0.1'))
    overlap, tv = sorted({r['overlap'] for r in rows}), sorted({r['tv'] for r in rows})
    fig, axes = plt.subplots(2, len(metrics), figsize=(5*len(metrics), 8), squeeze=False)
    for bi, bg in enumerate(('white', 'black')):
        for mi, (key, title) in enumerate(metrics):
            grid = np.full((len(overlap), len(tv)), np.nan)
            counts = np.zeros_like(grid, dtype=int)
            for oi, ow in enumerate(overlap):
                for ti, strength in enumerate(tv):
                    values = [r[key] for r in rows if r['background'] == bg and r['overlap'] == ow and r['tv'] == strength and np.isfinite(r[key])]
                    if values:
                        grid[oi, ti], counts[oi, ti] = np.mean(values), len(values)
            ax = axes[bi, mi]
            kwargs = {'vmin': 0, 'vmax': 1} if key != 'hybrid_psnr' else {'vmin': min(r[key] for r in rows), 'vmax': max(r[key] for r in rows)}
            im = ax.imshow(grid, aspect='auto', **kwargs)
            ax.set(title=f'{bg}: {title}', xlabel='TV multiplier', ylabel='Overlap weight')
            ax.set_xticks(range(len(tv)), [f'{v:g}' for v in tv])
            ax.set_yticks(range(len(overlap)), [f'{v:g}' for v in overlap])
            for oi, ti in np.ndindex(grid.shape):
                if counts[oi, ti]:
                    ax.text(ti, oi, f'{grid[oi, ti]:.3f}\nn={counts[oi, ti]}', ha='center', va='center', color='white', fontsize=9)
            fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(destination/'factorial.png', dpi=150)
    plt.close(fig)
    deltas = []
    index = {(r['background'], r['overlap'], r['tv'], r['seed']): r for r in rows}
    for r in rows:
        for factor, baseline in [('background', 'white'), ('overlap', 0.), ('tv', 0.)]:
            if r[factor] == baseline:
                continue
            key = tuple(baseline if name == factor else r[name] for name in ('background', 'overlap', 'tv', 'seed'))
            control = index.get(key)
            if control is not None:
                deltas.append(dict(run=r['run'], control=control['run'], factor=factor, seed=r['seed'],
                                   delta_psnr=r['hybrid_psnr']-control['hybrid_psnr'],
                                   delta_coverage09=r['coverage_gt_09']-control['coverage_gt_09']))
    write_csv(destination/'paired_deltas.csv', deltas)
    gallery_links = []
    for bg in ('white', 'black'):
        for ow in overlap:
            panels = []
            for r in sorted(rows, key=lambda r: r['tv']):
                if r['background'] != bg or r['overlap'] != ow or r['seed'] != 42:
                    continue
                at = root/f'run_{r["run"]}'/'joint/evaluation'/f'iteration_{iteration:06d}'
                paths = sorted(at.glob(f'{view_index:04d}_*_coverage_panel.png'))
                if not paths:
                    continue
                img = Image.open(paths[0]).convert('RGB')
                img.thumbnail((2400, 1200))
                canvas = Image.new('RGB', (img.width, img.height+25), 'white')
                canvas.paste(img, (0, 25))
                ImageDraw.Draw(canvas).text((8, 6), f'run {r["run"]}, {bg}, O={ow:g}, TV={r["tv"]:g}, seed42, PSNR={r["hybrid_psnr"]:.2f}', fill='black')
                panels.append(canvas)
            if panels:
                gallery = Image.new('RGB', (max(p.width for p in panels), sum(p.height for p in panels)), 'white')
                y = 0
                for panel in panels:
                    gallery.paste(panel, (0, y))
                    y += panel.height
                filename = f'{bg}_O{ow:g}_view{view_index}.png'
                gallery.save(destination/filename)
                gallery_links.append(f'![{bg}, overlap={ow:g}]({filename})')
    best = max(rows, key=lambda r: r['hybrid_psnr'])
    sections = [f'# RGB surface — iteration {iteration}',
        f'Available runs: {len(rows)}/72. Compare the same iteration; cells show the actual seed count.',
        f'Best hybrid PSNR here: run {best["run"]}, {best["hybrid_psnr"]:.2f} dB. This is NOT a disentanglement ranking.',
        '![Factorial comparisons](factorial.png)',
        'Red pixels have A_G <= 0.1; open water is included. Full-frame coverage is not floor coverage. '
        'Direct/backscatter and Gaussian-only PSNR are not semantic ownership measurements.',
        'Floor metrics require fixed external evaluation masks and remain NaN without them.',
        '[Per-run data](runs.csv)' + (' · [Paired seed deltas](paired_deltas.csv)' if deltas else ''), *gallery_links]
    (destination/'RESULTS.md').write_text('\n\n'.join(sections)+'\n')
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--iteration', type=int, default=30000)
    p.add_argument('--masks', type=Path)
    p.add_argument('--view-index', type=int, default=0)
    a = p.parse_args()
    write_report(a.root, a.output or a.root/'analysis', a.iteration, a.masks, a.view_index)
