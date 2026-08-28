from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from foamsurf import GaussianScene, NativeHybridRenderer, VoronoiScene
from foamsurf_model.medium import VoronoiMediumModel
from foamsurf_model.voronoi_adaptation import VoronoiAdaptationController
from training.config import (
    DatasetConfig,
    ExperimentConfig,
    ModelConfig,
    VoronoiAdaptationConfig,
)
from training.trainer import training_source


def _config(mode: str, target: str, medium_points: int = 32) -> ExperimentConfig:
    return ExperimentConfig(
        representation_mode=mode,
        dataset=DatasetConfig(
            source_path="both",
            gaussian_ground_truth_path="lego",
            medium_ground_truth_path="clouds",
            train_target=target,
        ),
        model=ModelConfig(medium_points=medium_points, medium_guard_points=0),
    )


def test_training_target_contract_and_large_sparse_initializations() -> None:
    assert training_source(_config("hybrid", "both")) == "both"
    assert training_source(_config("gaussian_only", "lego")) == "lego"
    for count in (16_384, 32_768):
        config = _config("voronoi_only", "clouds", medium_points=count)
        config.validate()
        assert training_source(config) == "clouds"
    with pytest.raises(ValueError, match="requires dataset.train_target"):
        _config("gaussian_only", "clouds").validate()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="native renderer requires CUDA")
def test_pure_native_forwards_do_not_create_inactive_gradients() -> None:
    try:
        import foamsurf._C  # noqa: F401
    except ImportError as error:
        pytest.skip(str(error))
    renderer = NativeHybridRenderer(max_batch_size=4, far=4.0)
    rays = torch.tensor([[0, 0, 0, 0, 0, 1]], dtype=torch.float32, device="cuda")
    gaussian_values = [
        torch.tensor(value, device="cuda", requires_grad=True)
        for value in (
            [[0.0, 0.0, 2.0]], [[-1.0, -1.0]], [[1.0, 0.0, 0.0, 0.0]],
            [[0.8, 0.2, 0.1]], [[1.0]], [[0.1]],
        )
    ]
    gaussians = GaussianScene(*gaussian_values)
    torch.manual_seed(7)
    medium_values = [
        (torch.randn(40, 3, device="cuda") * 1.5).requires_grad_(),
        (torch.rand(40, 1, device="cuda") * 0.2 + 0.05).requires_grad_(),
        torch.rand(40, 3, device="cuda").requires_grad_(),
    ]
    medium = VoronoiScene(*medium_values)
    indices = torch.arange(40, device="cuda")
    adjacency = indices.repeat(40, 1)
    adjacency = adjacency[adjacency != indices[:, None]].to(torch.uint32).contiguous()
    offsets = torch.arange(0, 40 * 39 + 1, 39, device="cuda").to(torch.uint32)
    renderer.set_gaussian_geometry(gaussians)
    renderer.set_voronoi_geometry(medium, adjacency, offsets)

    renderer.render(rays, gaussians=gaussians).image.sum().backward()
    assert any(value.grad is not None for value in gaussian_values)
    assert all(value.grad is None for value in medium_values)
    for value in gaussian_values:
        value.grad = None

    renderer.render(rays, voronoi=medium).image.sum().backward()
    assert all(value.grad is None for value in gaussian_values)
    assert any(value.grad is not None for value in medium_values)


def _complete_topology(count: int, device: torch.device) -> SimpleNamespace:
    indices = torch.arange(count, device=device)
    adjacency = indices.repeat(count, 1)
    adjacency = adjacency[adjacency != indices[:, None]].to(torch.uint32).contiguous()
    offsets = torch.arange(0, count * (count - 1) + 1, count - 1, device=device).to(torch.uint32)
    return SimpleNamespace(adjacency=adjacency, offsets=offsets)


def _medium_optimizer(model: VoronoiMediumModel) -> torch.optim.Adam:
    return torch.optim.Adam([
        {"params": [model.sites], "name": "medium_sites"},
        {"params": [model.raw_density], "name": "medium_density"},
        {"params": [model.raw_color], "name": "medium_color"},
    ], lr=1e-3)


def test_voronoi_plan_respects_budget_pruning_limit_and_guards() -> None:
    model = VoronoiMediumModel(
        torch.zeros(3), torch.ones(3),
        ModelConfig(medium_points=40, medium_guard_points=8), "cpu",
    )
    config = VoronoiAdaptationConfig(
        enabled=True, densify_fraction=0.25, max_points=44,
        min_points=32, min_visibility=1, contribution_threshold=1e-3,
        prune_cell_radius_fraction=100.0, max_prune_fraction=0.10,
        prune_patience=1,
    )
    controller = VoronoiAdaptationController(
        model, _medium_optimizer(model), _complete_topology(40, model.sites.device), config
    )
    statistics = torch.zeros(40, 5)
    statistics[:, 1] = 1
    statistics[:, 4] = 1
    controller.collect_statistics(statistics)
    plan = controller.propose_plan()

    assert plan.prune_count <= 4
    assert plan.original_count - plan.prune_count + plan.densify_count <= 44
    kept_guards = model.guard_mask[plan.keep_indices].sum()
    assert int(kept_guards.item()) == 8


def test_voronoi_duplicate_children_receive_safety_perturbation() -> None:
    model = VoronoiMediumModel(
        torch.zeros(3), torch.ones(3),
        ModelConfig(medium_points=40, medium_guard_points=0), "cpu",
    )
    controller = VoronoiAdaptationController(
        model, _medium_optimizer(model),
        _complete_topology(40, model.sites.device),
        VoronoiAdaptationConfig(enabled=True, child_jitter_fraction=0.0),
    )
    children = torch.stack(
        (model.sites[0], model.sites[0], model.sites[1])
    )
    keep = torch.arange(model.point_count)
    repaired, repair_count = controller._repair_duplicate_children(
        keep, children
    )
    combined = torch.cat((model.sites[keep], repaired), dim=0)

    assert repair_count >= 3
    assert torch.unique(combined, dim=0).shape[0] == combined.shape[0]
    assert torch.linalg.vector_norm(repaired - children, dim=-1).max() < 1.0e-4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="topology migration requires CUDA")
def test_voronoi_mutation_migrates_adam_state_and_rebuilds_topology() -> None:
    try:
        import foamsurf_topology  # noqa: F401
    except ImportError as error:
        pytest.skip(str(error))
    from foamsurf_model.topology import VoronoiTopologyController

    torch.manual_seed(11)
    model = VoronoiMediumModel(
        torch.full((3,), -1.0, device="cuda"),
        torch.full((3,), 1.0, device="cuda"),
        ModelConfig(medium_points=40, medium_guard_points=8), "cuda",
    )
    optimizer = _medium_optimizer(model)
    topology = VoronoiTopologyController(model, optimizer)
    (model.sites.square().mean() + model.raw_density.square().mean() + model.raw_color.square().mean()).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    controller = VoronoiAdaptationController(
        model, optimizer, topology,
        VoronoiAdaptationConfig(
            enabled=True, densify_fraction=0.10, max_points=44,
            min_points=32, contribution_threshold=1e-8,
            prune_cell_radius_fraction=0.0, prune_patience=2,
        ),
    )
    statistics = torch.zeros(40, 5, device="cuda")
    statistics[:, 0] = 1e-3
    statistics[:, 1] = 2
    statistics[:, 3] = 1
    statistics[:, 4] = 1
    controller.collect_statistics(statistics)
    plan = controller.propose_plan()
    event = controller.apply_plan(plan)

    assert event == {"densify": 3, "prune": 0}
    assert model.point_count == 43
    assert int(model.guard_mask.sum().item()) == 8
    assert topology.offsets.numel() == 44
    for group in optimizer.param_groups:
        parameter = group["params"][0]
        state = optimizer.state[parameter]
        assert state["exp_avg"].shape[0] == 43
        assert state["exp_avg_sq"].shape[0] == 43


def test_voronoi_adaptation_state_roundtrip_requires_matching_population() -> None:
    model = VoronoiMediumModel(
        torch.zeros(3), torch.ones(3),
        ModelConfig(medium_points=32, medium_guard_points=0), "cpu",
    )
    config = VoronoiAdaptationConfig(enabled=True, min_points=16)
    first = VoronoiAdaptationController(
        model, _medium_optimizer(model), _complete_topology(32, model.sites.device), config
    )
    first.low_activity_streak[3] = 2
    first.revision = 7
    first.total_densified = 11
    first.total_pruned = 4
    state = first.state_dict()
    second = VoronoiAdaptationController(
        model, _medium_optimizer(model), _complete_topology(32, model.sites.device), config
    )
    second.load_state_dict(state)
    assert second.revision == 7
    assert second.total_densified == 11
    assert second.total_pruned == 4
    torch.testing.assert_close(second.low_activity_streak, first.low_activity_streak)
