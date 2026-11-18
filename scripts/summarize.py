#!/usr/bin/env python3
"""Aggregate final-checkpoint metrics across seeds, then scenes, using sample standard deviation."""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def collect(root, target):
    rows = []
    targets = ['seathru', 'ablation', 'garden', 'synthetic'] if target == 'all' else [target]
    for category in targets:
        for run in sorted((root / category).glob('*/*/seed_*')):
            values = {}
            for path in sorted((run / 'evaluation').glob('*/metrics.json')):
                result = json.loads(path.read_text())
                if result.get('checkpoint', 'last') != 'last':
                    raise ValueError(f'not a final checkpoint: {path}')
                for key, value in result['mean'].items():
                    if isinstance(value, (int, float)):
                        if key in values:
                            raise ValueError(f'duplicate metric {key}: {path}')
                        values[key] = value
            fps = run / 'evaluation/render/fps.json'
            if fps.is_file():
                values['fps'] = json.loads(fps.read_text())['fps']
            if values:
                rows.append(dict(target=category, variant=run.parent.parent.name, scene=run.parent.name,
                                 seed=int(run.name.removeprefix('seed_')), metrics=values))
    return rows


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['target'], row['variant'], row['scene'])].append(row)
    by_variant = defaultdict(list)
    for row in rows:
        by_variant[(row['target'], row['variant'])].append(row)
    for (target, variant), members in by_variant.items():
        scenes = {r['scene'] for r in members}
        if len(scenes) < 2:
            continue
        per_seed = defaultdict(list)
        for row in members:
            per_seed[row['seed']].append(row)
        for seed, subset in per_seed.items():
            if {r['scene'] for r in subset} != scenes:
                continue
            keys = set.intersection(*(set(r['metrics']) for r in subset))
            metrics = {k: statistics.fmean(r['metrics'][k] for r in subset) for k in keys}
            groups[(target, variant, 'mean')].append(dict(seed=seed, metrics=metrics))
    result = []
    for (target, variant, scene), members in sorted(groups.items()):
        if len({r['seed'] for r in members}) != len(members):
            raise ValueError(f'duplicate seed in {target}/{variant}/{scene}')
        record = dict(target=target, variant=variant, scene=scene, seeds=len(members))
        keys = set.intersection(*(set(r['metrics']) for r in members))
        for key in sorted(keys):
            values = [r['metrics'][key] for r in members]
            record[key + '_mean'] = statistics.fmean(values)
            record[key + '_std'] = statistics.stdev(values) if len(values) > 1 else 0.0
        result.append(record)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target', choices=['seathru', 'ablation', 'garden', 'synthetic', 'all'])
    parser.add_argument('--root', type=Path, default=Path('outputs/paper'))
    args = parser.parse_args()
    rows = aggregate(collect(args.root, args.target))
    if not rows:
        parser.error('no metrics.json found under the selected output directory')
    fields = ['target', 'variant', 'scene', 'seeds'] + sorted(set().union(*(r.keys() for r in rows)) - {'target','variant','scene','seeds'})
    output = args.root / f'{args.target}_summary.csv'
    with output.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        metrics = [f"{k.removesuffix('_mean')} {v:.4f} ± {row[k.removesuffix('_mean') + '_std']:.4f}"
                   for k, v in row.items() if k.endswith('_mean')]
        print(f"{row['target']}/{row['variant']}/{row['scene']} ({row['seeds']} seeds): " + ', '.join(metrics))
    print(f'wrote {output}')


if __name__ == '__main__':
    main()
