#!/usr/bin/env python3
"""Compare rendered application panels to the supplied paper reference PNGs."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

PANELS = ('s1b_iui3_style', 's2c_iui3_edit', 's3b_curacao_style', 's4k_jg_haze')


def compare(actual, reference, min_psnr=50.):
    with Image.open(actual) as image:
        a = np.asarray(image.convert('RGB'), dtype=np.float64)
    with Image.open(reference) as image:
        b = np.asarray(image.convert('RGB'), dtype=np.float64)
    if a.shape != b.shape:
        return {'passed': False, 'error': f'Image shapes differ: {a.shape} vs {b.shape}'}
    delta = a - b
    mse = float(np.mean((delta / 255.) ** 2))
    exact = mse == 0
    psnr = None if exact else -10 * math.log10(mse)
    return {'passed': exact or psnr > min_psnr, 'exact': exact, 'psnr_db': psnr,
            'mae_u8': float(np.abs(delta).mean()), 'max_abs_u8': float(np.abs(delta).max()),
            'width': a.shape[1], 'height': a.shape[0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--renders', type=Path, required=True, help='directory with PANEL/still.png')
    parser.add_argument('--reference', type=Path, required=True, help='directory with PANEL.png')
    parser.add_argument('--out', type=Path, required=True, help='JSON report')
    parser.add_argument('--panels', nargs='+', choices=PANELS, default=PANELS)
    parser.add_argument('--min-psnr', type=float, default=50.)
    args = parser.parse_args()
    results = {}
    for panel in args.panels:
        try:
            results[panel] = compare(args.renders / panel / 'still.png', args.reference / f'{panel}.png', args.min_psnr)
        except (OSError, ValueError) as error:
            results[panel] = {'passed': False, 'error': str(error)}
    report = {'passed': all(r['passed'] for r in results.values()), 'min_psnr_db': args.min_psnr,
              'renders': str(args.renders.resolve()), 'reference': str(args.reference.resolve()), 'panels': results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
