from __future__ import annotations

import torch

from training.config import (
    DatasetConfig,
    ExperimentConfig,
    ModelConfig,
    RadFoamPriorConfig,
    VoronoiAdaptationConfig,
)
from training.radfoam_handoff import initialize_from_radfoam_prior


def _write_prior(tmp_path):
    xyz = torch.tensor(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [-1.2, 0.0, 0.5],
            [1.2, 0.0, -0.5],
            [-1.5, 0.0, 1.0],
            [1.5, 0.0, -1.0],
            [0.0, 0.0, 0.75],
        ],
        dtype=torch.float32,
    )
    neighbors = (
        (1, 4),
        (0, 2),
        (1, 3),
        (2, 5),
        (0, 6),
        (3, 7),
        (4, 8),
        (5, 8),
        (6, 7),
    )
    adjacency = torch.tensor([v for row in neighbors for v in row])
    offsets = torch.arange(0, 2 * len(neighbors) + 1, 2)
    checkpoint = {
        "xyz": xyz,
        "density": torch.full((9, 1), 0.5),
        "color_dc": torch.zeros(9, 3),
        "color_sh": torch.zeros(9, 45),
        "adjacency": adjacency,
        "adjacency_offsets": offsets,
    }
    checkpoint_path = tmp_path / "checkpoint.pt"
    labels_path = tmp_path / "labels.pt"
    torch.save(checkpoint, checkpoint_path)
    torch.save(
        {
            "labels": torch.tensor([1, 1, 1, 1, 2, 2, 0, 0, 2]),
            "class_names": {
                0: "empty",
                1: "surface",
                2: "medium",
                3: "ambiguous",
            },
        },
        labels_path,
    )
    return checkpoint_path, labels_path


def _config(checkpoint_path, labels_path) -> ExperimentConfig:
    return ExperimentConfig(
        seed=7,
        dataset=DatasetConfig(source_path="unused", train_target="both"),
        model=ModelConfig(
            gaussian_initial_points=2,
            gaussian_initial_opacity=0.1,
            gaussian_initial_scale_neighbors=1,
            medium_points=5,
            medium_guard_points=0,
        ),
        radfoam_prior=RadFoamPriorConfig(
            enabled=True,
            checkpoint_path=str(checkpoint_path),
            labels_path=str(labels_path),
            gaussian_count=2,
            pca_neighbors=3,
            reset_density=1.0e-8,
        ),
        voronoi_adaptation=VoronoiAdaptationConfig(
            enabled=False,
            max_points=None,
            min_points=1,
        ),
    )


def test_handoff_removes_surface_seeds_gaussians_and_resets_cavity(tmp_path):
    checkpoint_path, labels_path = _write_prior(tmp_path)
    result = initialize_from_radfoam_prior(
        _config(checkpoint_path, labels_path), "cpu"
    )
    model = result.model

    assert model.gaussians.point_count == 2
    assert model.medium.point_count == 5
    torch.testing.assert_close(
        model.gaussians.opacities,
        torch.full((2, 1), 0.1),
    )
    torch.testing.assert_close(
        model.gaussians.quaternions.norm(dim=1), torch.ones(2)
    )
    assert bool((model.gaussians.means[:, 2] == 0.0).all())

    densities = model.medium.densities.flatten()
    torch.testing.assert_close(densities[:4], torch.full((4,), 1.0e-8))
    expected_unaffected = torch.nn.functional.softplus(
        torch.tensor(0.5), beta=10.0
    )
    torch.testing.assert_close(densities[4], expected_unaffected)
    assert result.report["removed_surface_count"] == 4
    assert result.report["affected_retained_count"] == 2
    assert result.report["reset_empty_count"] == 2
    assert result.report["retained_voronoi_count"] == 5


def test_prior_config_rejects_voronoi_population_adaptation(tmp_path):
    checkpoint_path, labels_path = _write_prior(tmp_path)
    config = _config(checkpoint_path, labels_path)
    config.voronoi_adaptation.enabled = True
    try:
        config.validate()
    except ValueError as error:
        assert "voronoi_adaptation.enabled=false" in str(error)
    else:
        raise AssertionError("enabled Voronoi adaptation should be rejected")
