from pathlib import Path
import pytest
from training.colmap_joint import resolve

ROOT=Path(__file__).resolve().parents[2]

def test_paired_experiments(tmp_path):
    configs=[resolve(ROOT/'configs/colmap_rgb_medium.yaml',i,tmp_path/str(i)) for i in range(12)]
    assert len({c['variant']['name'] for c in configs})==12
    for i,c in enumerate(configs):
        assert c['warmup_steps']==([250,500][i//6])
        assert c['joint']['seed']==42+i%3
        assert c['joint']['model']['medium_type']==('scalar' if i%6<3 else 'rgb_backscatter')
        assert c['joint']['model']['medium_type']==c['warmup']['model']['medium_type']
        assert c['joint']['optimization']['iterations']==30000
    for i in [0,1,2,6,7,8]:
        a=configs[i]['joint'];b=configs[i+3]['joint']
        for key in a:
            if key not in ['model','output_dir','experiment_name','wandb']:assert a[key]==b[key]
    with pytest.raises(ValueError):resolve(ROOT/'configs/colmap_rgb_medium.yaml',12,tmp_path)
