import json
import math

import pytest

from scripts.summarize import aggregate, collect


def test_mean_over_scenes_is_computed_per_seed_before_sample_std():
    rows = [dict(target='seathru', variant='full', scene=scene, seed=seed, metrics={'psnr':value})
            for scene, seed, value in [('a',42,10),('a',43,14),('b',42,30),('b',43,34)]]
    result = aggregate(rows)
    mean = next(r for r in result if r['scene'] == 'mean')
    assert mean['psnr_mean'] == 22
    assert mean['psnr_std'] == pytest.approx(math.sqrt(8))
    assert mean['seeds'] == 2
    with pytest.raises(ValueError, match='duplicate seed'):
        aggregate(rows + rows[:1])


def test_collect_combines_final_metrics_and_timing(tmp_path):
    run = tmp_path / 'seathru/full/IUI3/seed_42/evaluation'
    for name, values in [('render',{'psnr':31.8}), ('separation',{'separation':0.95})]:
        (run / name).mkdir(parents=True)
        (run / name / 'metrics.json').write_text(json.dumps({'mean':values,'checkpoint':'last'}))
    (run / 'render/fps.json').write_text(json.dumps({'fps':140}))
    rows = collect(tmp_path, 'seathru')
    assert rows[0]['metrics'] == {'psnr':31.8, 'separation':0.95, 'fps':140}
