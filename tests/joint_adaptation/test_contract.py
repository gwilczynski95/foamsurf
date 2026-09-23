from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
import yaml
from training.config import ExperimentConfig
from training.joint_adaptation_sweep import VARIANTS, build_variant, validate_initialization, PopulationBoundsTopology

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('index', range(7))
def test_variants(index, tmp_path):
    base = ExperimentConfig.load(ROOT / 'configs/baseline_best_2d_gaussians.yaml')
    base.optimization.iterations = 30000
    base.representation_mode = 'hybrid'
    base.dataset.train_target = 'both'
    before = deepcopy(base.as_dict())
    settings = yaml.safe_load((ROOT / 'configs/joint_adaptation_sweep.yaml').read_text())
    c = build_variant(base, index, 297912, settings, tmp_path)
    name, policy, binary = VARIANTS[index]
    a = c.voronoi_adaptation
    assert base.as_dict() == before
    assert a.enabled == (policy != 'fixed')
    assert a.min_points == 148956
    assert a.max_points == (446868 if policy in ('adaptive', 'delayed') else 297912)
    assert a.densify_from == (2500 if policy == 'delayed' else 500)
    assert a.densify_fraction == (0.02 if policy in ('adaptive', 'delayed') else 0)
    assert c.disentanglement.opacity_binarization_weight == (0.001 if binary else base.disentanglement.opacity_binarization_weight)
    if binary:
        assert c.gaussian_adaptation.opacity_prune_until == 30000
    assert c.renderer == base.renderer
    assert c.optimization == base.optimization
    c.save(tmp_path / 'config.yaml')
    assert ExperimentConfig.load(tmp_path / 'config.yaml').as_dict() == c.as_dict()


@pytest.mark.parametrize('fail', [False, True])
def test_outer_bounds_restored(fail):
    lo, hi = torch.zeros(3), torch.ones(3)
    medium = SimpleNamespace(bbox_min=lo, bbox_max=hi, sites=torch.tensor([[-4.,0,0], [5.,1,1]]))
    class Inner:
        revision = 7
        def replace_population(self, m, *args, **kwargs):
            assert m.bbox_min[0] == -4 and m.bbox_max[0] == 5
            if fail:
                raise RuntimeError('injected failure')
            return True
    proxy = PopulationBoundsTopology(Inner())
    assert proxy.revision == 7
    args = (medium, None, None, None, torch.empty(0,3))
    if fail:
        with pytest.raises(RuntimeError, match='injected'):
            proxy.replace_population(*args)
    else:
        assert proxy.replace_population(*args)
    assert medium.bbox_min is lo and medium.bbox_max is hi


def test_initialization_guard():
    payload = dict(iteration=0, config=dict(representation_mode='hybrid'),
                   gaussian_optimizer=dict(state={}), medium_optimizer=dict(state={}),
                   model_metadata=dict(medium=dict(point_count=123)))
    assert validate_initialization(payload) == 123
    payload['iteration'] = 1
    with pytest.raises(ValueError, match='iteration-zero'):
        validate_initialization(payload)
    payload['iteration'] = 0
    payload['medium_optimizer']['state'] = {0: {}}
    with pytest.raises(ValueError, match='fresh'):
        validate_initialization(payload)
