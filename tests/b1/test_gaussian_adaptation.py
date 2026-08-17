from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pytest
import torch

from foamsurf import HybridRenderer
from foamsurf_data.camera_geometry import camera_normalization
from foamsurf_model.gaussian_adaptation import (
    GaussianAdaptationController,
    GaussianTopologyPlan,
)
from foamsurf_model.gaussians import SurfaceGaussianModel
from foamsurf_model.medium import VoronoiMediumModel
from foamsurf_model.scene import HybridSceneModel
from training.config import (
    GaussianAdaptationConfig,
    ModelConfig,
    OptimizationConfig,
)
from training.optimizer import (
    build_optimizers,
    update_gaussian_learning_rate,
)


@dataclass
class _Camera:
    origin: torch.Tensor
    right: torch.Tensor
    down: torch.Tensor
    forward: torch.Tensor
    fov_x: float = math.pi / 2
    fov_y: float = math.pi / 2
    width: int = 100
    height: int = 100
    name: str = "camera"


def _camera(origin=(0.0, 0.0, 0.0)) -> _Camera:
    return _Camera(
        origin=torch.tensor(origin, dtype=torch.float32),
        right=torch.tensor([1.0, 0.0, 0.0]),
        down=torch.tensor([0.0, 1.0, 0.0]),
        forward=torch.tensor([0.0, 0.0, 1.0]),
    )


def _model(count: int = 6) -> SurfaceGaussianModel:
    points = np.stack(
        (np.linspace(-0.5, 0.5, count), np.zeros(count), np.ones(count) * 2),
        axis=-1,
    ).astype(np.float32)
    colors = np.linspace(0.1, 0.9, count * 3).reshape(count, 3).astype(np.float32)
    model = SurfaceGaussianModel(
        points,
        colors,
        ModelConfig(appearance="rgb", gaussian_initial_points=count),
        "cpu",
    )
    with torch.no_grad():
        model.quaternions.zero_()
        model.quaternions[:, 0] = 1.0
    return model


def _optimizer(model: SurfaceGaussianModel) -> torch.optim.Adam:
    return torch.optim.Adam(
        [
            {
                "params": [parameter],
                "name": f"gaussian_{name}",
                "lr": 1.0e-3,
            }
            for name, parameter in model.named_primitive_parameters().items()
        ]
    )


def _controller(model, optimizer, **values):
    defaults = dict(
        enabled=True,
        max_points=20,
        gradient_threshold=0.1,
        min_opacity=0.005,
        percent_dense=0.01,
        size_prune_from=3_000,
        screen_pruning_enabled=True,
        max_screen_size=20.0,
        max_world_scale_fraction=0.1,
    )
    defaults.update(values)
    return GaussianAdaptationController(
        model,
        optimizer,
        GaussianAdaptationConfig(**defaults),
        scene_extent=10.0,
        support_radius=3.0,
    )


def _statistics(model, visible=True):
    values = torch.zeros((model.point_count, 4))
    if visible:
        values[:, 1] = 1.0
    return values


def test_camera_normalization_matches_graphdeco_definition() -> None:
    cameras = [_camera((-2.0, 0.0, 0.0)), _camera((2.0, 0.0, 0.0))]
    center, extent = camera_normalization(cameras)
    torch.testing.assert_close(center, torch.zeros(3))
    assert extent == pytest.approx(2.2)


def test_training_scene_is_forward_noop_with_separate_screen_gradient() -> None:
    model = _model(3)
    camera = _camera()
    ordinary = model.render_scene(camera.origin)
    training, screen_points = model.training_scene(camera)
    torch.testing.assert_close(training.means, ordinary.means, rtol=0, atol=0)
    loss = training.means[:, 0].sum()
    loss.backward()
    assert screen_points.grad is not None

    torch.testing.assert_close(
        screen_points.grad[:, 0], torch.full((3,), 2.0)
    )
    assert torch.count_nonzero(screen_points.grad[:, 1]) == 0
def test_sh_color_gradient_does_not_enter_screen_gradient() -> None:
    model = _model(3)
    model.appearance_mode = "sh"
    model.sh_degree = 3
    model.active_sh_degree = 3
    model.appearance_dc = torch.nn.Parameter(
        torch.randn(3, 3, 1, dtype=model.means.dtype)
    )
    model.appearance_rest = torch.nn.Parameter(
        torch.randn(3, 3, 15, dtype=model.means.dtype)
    )
    training, screen_points = model.training_scene(_camera())
    training.colors.sum().backward()
    assert screen_points.grad is None
    assert model.means.grad is not None
    assert torch.count_nonzero(model.means.grad) > 0



def test_sh_split_preserves_dc_forward_and_checkpoint() -> None:
    points = np.array(
        [[-0.2, 0.0, 2.0], [0.2, 0.0, 2.2]], dtype=np.float32
    )
    colors = np.array(
        [[0.8, 0.2, 0.1], [0.1, 0.5, 0.9]], dtype=np.float32
    )
    model = SurfaceGaussianModel(
        points,
        colors,
        ModelConfig(appearance="sh", sh_degree=3, gaussian_initial_points=2),
        "cpu",
    )
    model.active_sh_degree = 0
    torch.testing.assert_close(
        model.colors(torch.zeros(3)), torch.from_numpy(colors)
    )
    assert model.appearance_dc.shape == (2, 3, 1)
    assert model.appearance_rest.shape == (2, 3, 15)
    state = {
        name: parameter.detach().clone()
        for name, parameter in model.named_primitive_parameters().items()
    }
    restored = SurfaceGaussianModel.from_checkpoint(
        state, model.checkpoint_metadata(), "cpu"
    )
    assert restored.active_sh_degree == 0
    torch.testing.assert_close(restored.appearance_dc, model.appearance_dc)
    torch.testing.assert_close(restored.appearance_rest, model.appearance_rest)


def test_graphdeco_optimizer_groups_and_position_schedule() -> None:
    gaussian = _model(3)
    model_config = ModelConfig(medium_points=32, medium_guard_points=0)
    medium = VoronoiMediumModel(
        torch.tensor([-1.0, -1.0, 0.0]),
        torch.tensor([1.0, 1.0, 4.0]),
        model_config,
        "cpu",
    )
    optimization = OptimizationConfig()
    torch.testing.assert_close(
        gaussian.opacities, torch.full((3, 1), 0.1)
    )
    optimizer, _ = build_optimizers(
        HybridSceneModel(gaussian, medium), optimization, scene_extent=2.5
    )
    rates = {group["name"]: group["lr"] for group in optimizer.param_groups}
    assert rates["gaussian_appearance_dc"] == pytest.approx(0.0025)
    assert rates["gaussian_appearance_rest"] == pytest.approx(0.000125)
    assert rates["gaussian_opacity_logits"] == pytest.approx(0.025)
    assert rates["gaussian_log_scales"] == pytest.approx(0.005)
    assert rates["gaussian_quaternions"] == pytest.approx(0.001)
    assert optimizer.defaults["eps"] == pytest.approx(1.0e-15)
    assert update_gaussian_learning_rate(
        optimizer, optimization, 2.5, 0
    ) == pytest.approx(0.0004)
    assert update_gaussian_learning_rate(
        optimizer, optimization, 2.5, 30_000
    ) == pytest.approx(0.000004)


def test_sh_degree_progression_starts_from_dc() -> None:
    model = _model(3)
    model.appearance_mode = "sh"
    model.sh_degree = 3
    model.active_sh_degree = 0
    model.appearance_dc = torch.nn.Parameter(torch.zeros(3, 3, 1))
    model.appearance_rest = torch.nn.Parameter(torch.zeros(3, 3, 15))
    assert model.appearance.shape[-1] == 1
    model.one_up_sh_degree()
    assert model.active_sh_degree == 1
    assert model.appearance.shape[-1] == 4


def test_projected_radius_is_in_pixel_units() -> None:
    model = _model(1)
    with torch.no_grad():
        model.log_scales.fill_(math.log(0.1))
        model.means[:] = torch.tensor([[0.0, 0.0, 2.0]])
    controller = _controller(model, _optimizer(model))
    radius = controller.projected_radii(_camera())
    assert radius.item() == pytest.approx(8.0)


def test_projected_radius_uses_exact_2dgs_tilted_homography() -> None:
    model = _model(1)
    half_angle = math.radians(60.0) / 2.0
    with torch.no_grad():
        model.means[:] = torch.tensor([[0.0, 0.0, 2.0]])
        model.log_scales[:] = torch.log(torch.tensor([[0.15, 0.02]]))
        model.quaternions[:] = torch.tensor(
            [[math.cos(half_angle), 0.0, math.sin(half_angle), 0.0]]
        )
    camera = _camera()
    camera.width = 800
    camera.height = 800
    camera.fov_x = math.radians(49.1)
    camera.fov_y = math.radians(49.1)
    controller = _controller(model, _optimizer(model))

    radius = controller.projected_radii(camera)

    # Exact value returned by the pinned 2DGS CUDA rasterizer. The previous
    # Jacobian approximation returned 99 px for this tilted surfel.
    assert radius.item() == pytest.approx(103.0)

def test_plan_uses_only_gradient_for_clone_split_and_opacity_pruning() -> None:
    model = _model()
    controller = _controller(model, _optimizer(model))
    with torch.no_grad():
        model.log_scales.fill_(math.log(0.05))
        model.log_scales[1].fill_(math.log(0.2))
        model.opacity_logits[2].fill_(torch.logit(torch.tensor(0.001)))
    gradient = torch.zeros((model.point_count, 2))
    gradient[0, 0] = 1.0
    gradient[1, 0] = 0.9
    controller.collect_statistics(
        gradient, _statistics(model), _camera()
    )
    plan = controller.propose_plan(iteration=1_000)
    assert plan.clone_indices.tolist() == [0]
    assert plan.split_indices.tolist() == [1]
    assert plan.prune_count == 1
    assert plan.final_count == 7
    assert controller.last_plan_statistics["gradient_candidates"] == 2
    assert controller.last_plan_statistics["opacity_prune"] == 1
    assert controller.last_plan_statistics["screen_prune"] == 0
    assert controller.last_plan_statistics["world_prune"] == 0


def test_opacity_pruning_has_an_independent_start_iteration() -> None:
    model = _model(3)
    controller = _controller(
        model,
        _optimizer(model),
        opacity_prune_from=3_000,
    )
    with torch.no_grad():
        model.opacity_logits[0].fill_(torch.logit(torch.tensor(0.001)))

    before = controller.propose_plan(iteration=1_000)
    assert before.prune_count == 0
    assert controller.last_plan_statistics["opacity_prune"] == 0

    after = controller.propose_plan(iteration=3_100)
    assert after.prune_count == 1
    assert controller.last_plan_statistics["opacity_prune"] == 1


def test_screen_oversized_useful_point_is_replaced_before_pruning() -> None:
    model = _model(4)
    controller = _controller(model, _optimizer(model))
    with torch.no_grad():
        model.log_scales.fill_(math.log(0.05))
    controller.gradient_sum.zero_()
    controller.gradient_sum[1] = 1.0
    controller.gradient_denom.fill_(1.0)
    controller.max_radii2d[1] = 21.0

    plan = controller.propose_plan(iteration=3_100)
    assert plan.clone_indices.tolist() == [1]
    assert plan.split_count == 0
    assert 1 not in plan.keep_indices.tolist()
    assert plan.prune_count == 1
    assert plan.final_count == model.point_count
    assert controller.last_plan_statistics["screen_prune"] == 1
    assert controller.last_plan_statistics["gradient_screen_overlap"] == 1
    assert controller.last_plan_statistics["screen_replacements"] == 1

    old = model.means.detach().clone()
    event = controller.apply_plan(plan)
    assert event == {
        "clone": 1,
        "split": 0,
        "prune": 1,
        "budget_skip": 0,
    }
    assert model.point_count == 4
    torch.testing.assert_close(model.means[-1], old[1])


def test_large_screen_oversized_point_is_split_before_parent_removal() -> None:
    model = _model(2)
    controller = _controller(model, _optimizer(model))
    with torch.no_grad():
        model.log_scales.fill_(math.log(0.2))
    controller.gradient_sum.zero_()
    controller.gradient_sum[0] = 1.0
    controller.gradient_denom.fill_(1.0)
    controller.max_radii2d[0] = 21.0

    plan = controller.propose_plan(iteration=3_100)
    assert plan.clone_count == 0
    assert plan.split_indices.tolist() == [0]
    assert 0 not in plan.keep_indices.tolist()
    assert plan.final_count == 3
    assert controller.last_plan_statistics["gradient_screen_overlap"] == 1
    assert controller.last_plan_statistics["screen_replacements"] == 1

    event = controller.apply_plan(plan)
    assert event["split"] == 1
    assert model.point_count == 3
    expected_scale = torch.full((2, 2), 0.2 / 1.6)
    torch.testing.assert_close(model.scales[-2:], expected_scale)


def test_world_oversized_point_without_viable_children_is_removed() -> None:
    model = _model(3)
    controller = _controller(model, _optimizer(model))
    with torch.no_grad():
        model.log_scales.fill_(math.log(0.05))
        model.log_scales[0].fill_(math.log(2.0))
    controller.gradient_sum.zero_()
    controller.gradient_sum[0] = 1.0
    controller.gradient_denom.fill_(1.0)

    plan = controller.propose_plan(iteration=3_100)
    assert plan.clone_count == 0
    assert plan.split_count == 0
    assert 0 not in plan.keep_indices.tolist()
    assert plan.prune_count == 1
    assert controller.last_plan_statistics["world_prune"] == 1
    assert controller.last_plan_statistics["gradient_world_overlap"] == 1
    assert controller.last_plan_statistics["screen_replacements"] == 0


def test_budget_keeps_best_candidate_and_never_overshoots() -> None:
    model = _model(4)
    controller = _controller(model, _optimizer(model), max_points=4)
    with torch.no_grad():
        model.log_scales.fill_(math.log(0.05))
        model.opacity_logits[0].fill_(torch.logit(torch.tensor(0.001)))
    controller.gradient_sum[:] = torch.tensor([0.0, 0.2, 0.3, 0.4])
    controller.gradient_denom.fill_(1.0)
    plan = controller.propose_plan(iteration=1_000)
    assert plan.densification_skipped
    assert controller.last_plan_statistics["gradient_candidates"] == 3
    assert controller.last_plan_statistics["budget_rejected"] == 2
    assert plan.clone_indices.tolist() == [3]
    assert plan.split_count == 0
    assert plan.prune_count == 1
    assert plan.final_count == 4
    event = controller.apply_plan(plan)
    assert event["budget_skip"] == 1
    assert event["clone"] == 1
    assert model.point_count == 4

    controller.gradient_sum.fill_(1.0)
    controller.gradient_denom.fill_(1.0)
    resumed = controller.propose_plan(iteration=1_100)
    assert resumed.densification_skipped
    assert resumed.clone_count == 0
    assert resumed.final_count == 4
    assert controller.last_plan_statistics["budget_rejected"] == 4


def test_pruning_releases_capacity_for_replacement_at_budget() -> None:
    model = _model(4)
    controller = _controller(model, _optimizer(model), max_points=4)
    with torch.no_grad():
        model.log_scales.fill_(math.log(0.05))
        model.opacity_logits[0].fill_(torch.logit(torch.tensor(0.001)))
    controller.gradient_sum[3] = 1.0
    controller.gradient_denom.fill_(1.0)
    plan = controller.propose_plan(iteration=1_000)
    assert plan.prune_count == 1
    assert plan.clone_indices.tolist() == [3]
    assert plan.final_count == 4
    assert not plan.densification_skipped


def test_apply_plan_maps_parameters_and_adam_moments() -> None:
    torch.manual_seed(5)
    model = _model(4)
    optimizer = _optimizer(model)
    loss = sum(parameter.square().sum() for parameter in model.parameters())
    loss.backward()
    optimizer.step()
    old_values = {
        name: parameter.detach().clone()
        for name, parameter in model.named_primitive_parameters().items()
    }
    old_moments = {
        name: optimizer.state[parameter]["exp_avg"].clone()
        for name, parameter in model.named_primitive_parameters().items()
    }
    controller = _controller(model, optimizer)
    plan = GaussianTopologyPlan(
        keep_indices=torch.tensor([0]),
        clone_indices=torch.tensor([0]),
        split_indices=torch.tensor([2]),
        original_count=4,
    )
    event = controller.apply_plan(plan)
    assert event == {"clone": 1, "split": 1, "prune": 2, "budget_skip": 0}
    assert model.point_count == 4
    for name, parameter in model.named_primitive_parameters().items():
        torch.testing.assert_close(parameter[0], old_values[name][0])
        torch.testing.assert_close(parameter[1], old_values[name][0])
        state = optimizer.state[parameter]
        torch.testing.assert_close(state["exp_avg"][0], old_moments[name][0])
        assert torch.count_nonzero(state["exp_avg"][1:]) == 0
        assert state["exp_avg_sq"].shape == parameter.shape


def test_noop_preserves_parameter_identity_and_render_exactly() -> None:
    model = _model(3)
    controller = _controller(model, _optimizer(model))
    parameters = dict(model.named_primitive_parameters())
    rays = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    renderer = HybridRenderer(far=4.0)
    before = renderer(rays, gaussians=model.render_scene(None)).image
    indices = torch.arange(model.point_count)
    event = controller.apply_plan(
        GaussianTopologyPlan(indices, indices[:0], indices[:0], model.point_count)
    )
    after = renderer(rays, gaussians=model.render_scene(None)).image
    assert event == {"clone": 0, "split": 0, "prune": 0, "budget_skip": 0}
    torch.testing.assert_close(after, before, rtol=0, atol=0)
    for name, parameter in model.named_primitive_parameters().items():
        assert parameter is parameters[name]


def test_adaptation_state_roundtrip_requires_matching_count() -> None:
    model = _model(4)
    controller = _controller(model, _optimizer(model))
    controller.gradient_denom.fill_(7)
    controller.total_cloned = 3
    state = controller.state_dict()
    restored = _controller(model, _optimizer(model))
    restored.load_state_dict(state)
    assert torch.all(restored.gradient_denom == 7)
    assert restored.total_cloned == 3

    wrong_model = _model(5)
    wrong = _controller(wrong_model, _optimizer(wrong_model))
    with pytest.raises(ValueError, match="checkpoint size mismatch"):
        wrong.load_state_dict(state)


def test_none_max_points_disables_densification_budget() -> None:
    model = _model(6)
    controller = _controller(model, _optimizer(model), max_points=None)
    with torch.no_grad():
        model.log_scales.fill_(math.log(0.05))
    controller.gradient_sum.fill_(1.0)
    controller.gradient_denom.fill_(1.0)

    plan = controller.propose_plan(iteration=1_000)

    assert not plan.densification_skipped
    assert plan.clone_count == 6
    assert plan.final_count == 12
    assert controller.last_plan_statistics["budget_rejected"] == 0
