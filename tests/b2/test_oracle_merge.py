from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from foamsurf_model import HybridSceneModel
from training import ExperimentConfig
from training.oracle_merge import (
    branch_tensor_digest,
    import_oracle_model,
    validate_oracle_sources,
)


def _payload(config: ExperimentConfig, count: int = 5) -> dict:
    gaussian_state = {
        "appearance_dc": torch.rand(count, 3),
        "appearance_rest": torch.empty(count, 3, 0),
        "opacity_logits": torch.rand(count, 1),
        "raw_kappa": torch.rand(count, 1),
        "means": torch.rand(count, 3),
        "log_scales": torch.rand(count, 2),
        "quaternions": torch.rand(count, 4),
    }
    medium_count = 40
    guard_mask = torch.zeros(medium_count, dtype=torch.bool)
    guard_mask[torch.tensor([1, 4, 9, 12, 18, 21, 31, 37])] = True
    medium_state = {
        "sites": torch.rand(medium_count, 3),
        "raw_density": torch.rand(medium_count, 1),
        "raw_color": torch.rand(medium_count, 3),
        "bbox_min": torch.zeros(3),
        "bbox_max": torch.ones(3),
        "guard_mask": guard_mask,
    }
    state = {
        **{f"gaussians.{name}": value for name, value in gaussian_state.items()},
        **{f"medium.{name}": value for name, value in medium_state.items()},
    }
    return {
        "config": config.as_dict(),
        "model": state,
        "model_metadata": {
            "gaussians": {
                "appearance_mode": "rgb",
                "sh_degree": 3,
                "active_sh_degree": 3,
                "point_count": count,
            },
            "medium": {"point_count": medium_count, "guard_count": 8},
        },
    }


def _configs() -> tuple[ExperimentConfig, ExperimentConfig, ExperimentConfig]:
    oracle = ExperimentConfig.load("configs/baseline_best_2d_gaussians.yaml")
    oracle.representation_mode = "hybrid"
    oracle.dataset.train_target = "both"
    gaussian = deepcopy(oracle)
    gaussian.representation_mode = "gaussian_only"
    gaussian.dataset.train_target = "lego"
    medium = deepcopy(oracle)
    medium.representation_mode = "voronoi_only"
    medium.dataset.train_target = "clouds"
    medium.renderer.far = 6.0
    medium.renderer.gaussian_hit_buffer_size = 16
    return oracle, gaussian, medium


def test_oracle_import_uses_only_selected_branches_and_preserves_guard_mask() -> None:
    oracle, gaussian_config, medium_config = _configs()
    gaussian_payload = _payload(gaussian_config, count=7)
    medium_payload = _payload(medium_config, count=3)
    expected_guard_mask = medium_payload["model"]["medium.guard_mask"].clone()

    validate_oracle_sources(gaussian_payload, medium_payload, oracle)
    model = import_oracle_model(gaussian_payload, medium_payload, "cpu")

    assert isinstance(model, HybridSceneModel)
    assert model.gaussians.point_count == 7
    assert model.medium.point_count == 40
    torch.testing.assert_close(model.medium.guard_mask, expected_guard_mask)
    assert not any(parameter.requires_grad for parameter in model.parameters())
    torch.testing.assert_close(
        model.gaussians.means,
        gaussian_payload["model"]["gaussians.means"],
    )
    torch.testing.assert_close(
        model.medium.sites,
        medium_payload["model"]["medium.sites"],
    )


def test_oracle_source_contract_rejects_wrong_mode_and_renderer() -> None:
    oracle, gaussian_config, medium_config = _configs()
    gaussian_payload = _payload(gaussian_config)
    medium_payload = _payload(medium_config)

    bad_mode = deepcopy(gaussian_payload)
    bad_mode["config"]["representation_mode"] = "hybrid"
    with pytest.raises(ValueError, match="gaussian_only"):
        validate_oracle_sources(bad_mode, medium_payload, oracle)

    wrong_renderer = deepcopy(oracle)
    wrong_renderer.renderer.far = 123.0
    with pytest.raises(ValueError, match="renderer settings"):
        validate_oracle_sources(gaussian_payload, medium_payload, wrong_renderer)


def test_branch_digest_ignores_the_inactive_checkpoint_branch() -> None:
    _, gaussian_config, _ = _configs()
    payload = _payload(gaussian_config)
    first = branch_tensor_digest(payload, "gaussians")
    payload["model"]["medium.sites"].add_(1.0)
    assert branch_tensor_digest(payload, "gaussians") == first
    payload["model"]["gaussians.means"].add_(1.0)
    assert branch_tensor_digest(payload, "gaussians") != first
