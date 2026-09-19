from copy import deepcopy
from pathlib import Path

import pytest

from training import ExperimentConfig


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "voronoi_prior_warmup.yaml"


def test_native_voronoi_warmup_contract() -> None:
    config = ExperimentConfig.load(CONFIG)

    assert config.voronoi_warmup.enabled is True
    assert config.representation_mode == "voronoi_only"
    assert config.dataset.train_target == "both"
    assert config.model.medium_points == 350_000
    assert config.model.gaussian_initial_points == 1
    assert config.model.medium_guard_points == 0
    assert config.optimization.iterations == 5_000
    assert config.voronoi_adaptation.enabled is True
    assert config.voronoi_adaptation.min_points == 100_000
    assert config.voronoi_adaptation.max_points == 500_000
    assert config.voronoi_adaptation.densify_from == 500
    assert config.voronoi_adaptation.densify_until == 5_001
    assert config.voronoi_warmup.milestone_iterations == [
        0, 500, 1_000, 2_000, 3_000, 4_000, 5_000
    ]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda c: setattr(c, "representation_mode", "hybrid"),
            "representation_mode=voronoi_only",
        ),
        (
            lambda c: setattr(c.dataset, "train_target", "clouds"),
            "dataset.train_target=both",
        ),
        (
            lambda c: setattr(c.voronoi_adaptation, "enabled", False),
            "adaptive Voronoi population",
        ),
        (
            lambda c: setattr(c.gaussian_adaptation, "enabled", True),
            "Gaussian adaptation",
        ),
        (
            lambda c: setattr(c.model, "gaussian_initial_points", 2),
            "one inactive Gaussian placeholder",
        ),
        (
            lambda c: setattr(
                c.voronoi_warmup, "milestone_iterations", [0, 1_000, 500]
            ),
            "sorted and unique",
        ),
    ],
)
def test_invalid_native_warmup_is_rejected(mutation, message: str) -> None:
    config = deepcopy(ExperimentConfig.load(CONFIG))
    mutation(config)
    with pytest.raises(ValueError, match=message):
        config.validate()


def test_full_scene_voronoi_only_remains_explicit_opt_in() -> None:
    config = ExperimentConfig.load(CONFIG)
    config.voronoi_warmup.enabled = False

    with pytest.raises(ValueError, match="requires dataset.train_target=clouds"):
        config.validate()
