"""Run in a separate pytest process: OptiX contexts outlive PyTorch fixtures."""
import csv
import pytest
import yaml
from .test_gpu import config_file
from training.colmap_joint import fit

@pytest.mark.parametrize('count',[25000,100000])
def test_production_size(tmp_path,count):
    path=config_file(tmp_path,steps=2,full=True)
    settings=yaml.safe_load(path.read_text())
    settings['counts']=[count]
    settings['profile_overrides']['population_target']=200000
    settings['joint_overrides']['voronoi_adaptation']['max_points']=500000
    path.write_text(yaml.safe_dump(settings))
    root=tmp_path/'production'
    fit(path,1,root)
    for stage in ('warmup','joint'):
        with (root/stage/'metrics/eval.csv').open() as handle:
            rows=list(csv.DictReader(handle))
        assert rows
        for row in rows:
            assert float(row['eval/responsibility/truncated_ray_fraction'])==0
