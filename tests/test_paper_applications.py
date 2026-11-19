"""Optional pixel and table regressions against the original application packages."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = Path(os.environ['FS_PAPER_WORKSPACE']) if os.environ.get('FS_PAPER_WORKSPACE') else None
pytestmark = pytest.mark.skipif(WORKSPACE is None, reason='set FS_PAPER_WORKSPACE to original application packages')


def execute(script, arguments, output):
    output.mkdir(parents=True, exist_ok=True)
    log_path = output.parent / (output.name + '.log')
    with log_path.open('w') as log:
        result = subprocess.run([sys.executable, str(ROOT / script), *map(str, arguments), '--out', str(output)],
                                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    assert result.returncode == 0, log_path.read_text()[-12000:]


@pytest.mark.cuda
@pytest.mark.parametrize('edit', ['bulge', 'bend', 'haze', 'plume2', 'relief', 'clear', 'murky', 'transplant', 'green'])
def test_edits_match_package(edit, tmp_path):
    base = WORKSPACE / 'paper_data/haze_work/pkg'
    run = base / 'models/base/IUI3-RedSea-Undistorted'
    scene = base / 'data/IUI3-RedSea-Undistorted'
    transplant = WORKSPACE / 'foamsurf/outputs/seathru/561669/Curasao-Undistorted_cull05r_bin01_wphase_rgbs15_wext'
    execute('edit.py', ['--run', run, '--scene', scene, '--view', 'MTN_5911', '--edits', edit,
                        '--no-original', '--transplant-from', transplant], tmp_path / 'render')
    actual_name = 'plume' if edit == 'plume2' else edit
    actual = np.asarray(Image.open(tmp_path / 'render' / actual_name / 'full.png').convert('RGB'), dtype=np.float64)
    archive = WORKSPACE / 'paper_data/edits_extract/foamsurf_paper_editing_part01.zip'
    with zipfile.ZipFile(archive) as z:
        member = f'foamsurf_paper_editing_2026-11-15/renders/edits/{edit}/full/MTN_5911.png'
        expected = np.asarray(Image.open(io.BytesIO(z.read(member))).convert('RGB'), dtype=np.float64)
    assert actual.shape == expected.shape
    mse = np.mean(((actual - expected) / 255) ** 2)
    psnr = float('inf') if mse == 0 else -10 * np.log10(mse)
    assert psnr > 50, (edit, psnr)


@pytest.mark.cuda
@pytest.mark.parametrize('scene,job,package', [('lego','563888','black_comparison'),
                                             ('ficus','563889','ficus_black_comparison')])
def test_insertion_matches_paper(scene, job, package, tmp_path):
    data = WORKSPACE / f'paper_data/{scene}_insertion/{package}'
    run = WORKSPACE / f'foamsurf_slim/outputs/synthetic_hybrid/{job}/{scene}_both_seed42'
    execute('insert.py', ['--benchmark', '--run', run, '--gt', data], tmp_path / 'benchmark')
    measured = json.loads((tmp_path / 'benchmark/metrics.json').read_text())['mean']
    expected = json.loads((data / 'summary.json').read_text())['foamsurf']
    for actual_key, expected_key in [('base_psnr','baseline_psnr'),('roi_psnr','roi_mean_psnr'),
                                     ('roi_pooled_psnr','roi_pooled_psnr'),('roi_mae','roi_mean_mae')]:
        assert abs(measured[actual_key] - expected[expected_key]) <= 1e-4, (actual_key, measured, expected)
    diagnostic = json.loads((tmp_path / 'benchmark/summary.json').read_text())
    assert diagnostic['max_empty_tracer_error'] == 0
    assert diagnostic['max_outside_change'] == 0
