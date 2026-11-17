"""The homogeneous-medium ablation (model.medium_homogeneous).

The ablation runs of the paper were trained with this field, so their joint/config.yaml carries it. Without the
field this branch refuses to load them, and without the averaging in VoronoiMediumModel it would load them and
silently render per-cell water instead of one homogeneous water.
"""
import os
from pathlib import Path

import torch

import pytest

from foamsurf_model.medium import VoronoiMediumModel
from training.config import ExperimentConfig, ModelConfig

@pytest.fixture
def ablation_runs():
    path = os.environ.get("FS_ABLATION_RUNS")
    if not path or not Path(path).is_dir():
        pytest.skip("set FS_ABLATION_RUNS to the paper ablation runs")
    return Path(path)


def medium(points: int = 16) -> VoronoiMediumModel:
    return VoronoiMediumModel(torch.zeros(3), torch.ones(3), ModelConfig(medium_type="rgb_backscatter",
                                                                         medium_points=points), "cpu")


def test_model_config_has_the_field_and_defaults_to_off():
    assert ModelConfig().medium_homogeneous is False
    assert ModelConfig(medium_homogeneous=True).medium_homogeneous is True


def test_cells_keep_their_own_values_by_default():
    model = medium()
    with torch.no_grad():
        model.raw_density.copy_(torch.arange(model.raw_density.numel(), dtype=torch.float32).reshape_as(model.raw_density))
        model.raw_color.copy_(torch.arange(model.raw_color.numel(), dtype=torch.float32).reshape_as(model.raw_color))
    for values in (model.optical_coefficients, model.colors):
        assert values.std(0).max() > 0, "cells must differ when the ablation is off"


@pytest.mark.parametrize("field", ["optical_coefficients", "colors"])
def test_homogeneous_gives_every_cell_the_mean(field):
    model = medium()
    with torch.no_grad():
        model.raw_density.copy_(torch.arange(model.raw_density.numel(), dtype=torch.float32).reshape_as(model.raw_density))
        model.raw_color.copy_(torch.arange(model.raw_color.numel(), dtype=torch.float32).reshape_as(model.raw_color))
    per_cell = getattr(model, field)
    model.homogeneous = True
    homogeneous = getattr(model, field)
    assert homogeneous.shape == per_cell.shape
    torch.testing.assert_close(homogeneous, per_cell.mean(0, keepdim=True).expand_as(per_cell))
    assert torch.allclose(homogeneous, homogeneous[:1].expand_as(homogeneous))  # one water everywhere


@pytest.mark.parametrize("variant, homogeneous, mode", [("nomedium", False, "gaussian_only"), ("homog", True, "hybrid")])
def test_the_paper_ablation_configs_load(variant, homogeneous, mode, ablation_runs):
    path = ablation_runs / f"Panama-Undisorted_{variant}/joint/config.yaml"
    if not path.is_file():
        pytest.skip("ablation runs not available")
    config = ExperimentConfig.load(path)
    assert config.model.medium_homogeneous is homogeneous
    assert config.representation_mode == mode
