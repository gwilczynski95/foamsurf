import torch
import pytest
from training.config import ModelConfig
from foamsurf_model.medium import VoronoiMediumModel


def test_checkpoint_and_reorder():
    c=ModelConfig(medium_type='rgb_backscatter',medium_points=32,medium_guard_points=0)
    m=VoronoiMediumModel(torch.zeros(3),torch.ones(3),c,'cpu')
    assert m.raw_density.shape==(32,6)
    with torch.no_grad():m.raw_density.copy_(torch.randn(32,6))
    s=m.render_scene();s.validate()
    r=VoronoiMediumModel.from_checkpoint(m.named_primitive_parameters(),m.checkpoint_metadata(),m.bbox_min,m.bbox_max,'cpu')
    torch.testing.assert_close(s.attenuation,r.render_scene().attenuation)
    torch.testing.assert_close(s.backscatter,r.render_scene().backscatter)
    p=torch.randperm(32);m.reorder_without_optimizer(p)
    torch.testing.assert_close(m.optical_coefficients,r.optical_coefficients[p])
    bad=dict(m.checkpoint_metadata(),medium_type='scalar')
    with pytest.raises(ValueError):VoronoiMediumModel.from_checkpoint(m.named_primitive_parameters(),bad,m.bbox_min,m.bbox_max,'cpu')
