"""Pruning of large Gaussians far from every training camera (open water)."""
import math

import numpy as np
import pytest
import torch

from foamsurf_model.gaussian_adaptation import GaussianAdaptationController
from foamsurf_model.gaussians import SurfaceGaussianModel
from training.config import ExperimentConfig, GaussianAdaptationConfig, ModelConfig

CAMERAS = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
# near & small, near & large, far & small, far & large, far & huge
POINTS = np.array([[0, 0, 5], [0, 0, 6], [0, 0, 30], [0, 0, 31], [0, 0, 32]], dtype=np.float32)
SCALES = [0.01, 0.2, 0.01, 0.2, 0.5]
PRUNE_EVENT = 1250  # opacity pruning active, densification active


def controller(far_prune_distance=15.0, far_prune_scale=0.05, **overrides):
    model = SurfaceGaussianModel(POINTS, np.full_like(POINTS, 0.5),
                                 ModelConfig(appearance="rgb", gaussian_initial_points=len(POINTS)), "cpu")
    with torch.no_grad():
        model.log_scales.copy_(torch.log(torch.tensor(SCALES))[:, None].expand_as(model.log_scales))
        model.opacity_logits.fill_(4.0)  # opaque: no opacity pruning
    optimizer = torch.optim.Adam([{"params": [p], "name": f"gaussian_{n}", "lr": 1e-3}
                                  for n, p in model.named_primitive_parameters().items()])
    settings = dict(
        enabled=True, max_points=100, min_points=1, densify_from=500, densify_until=15000,
        densification_interval=250, opacity_prune_from=1000, opacity_prune_until=30000,
        size_prune_from=100000, gradient_threshold=1.0e9,
        far_prune_distance=far_prune_distance, far_prune_scale=far_prune_scale)
    config = GaussianAdaptationConfig(**{**settings, **overrides})
    return GaussianAdaptationController(model, optimizer, config, scene_extent=100.0,
                                        camera_centers=CAMERAS)


def test_prunes_only_large_gaussians_far_from_every_camera():
    adaptation = controller()
    plan = adaptation.propose_plan(PRUNE_EVENT)
    assert plan.keep_indices.tolist() == [0, 1, 2]
    assert adaptation.last_plan_statistics["far_prune"] == 2
    assert adaptation.apply_plan(plan)["prune"] == 2
    assert adaptation.model.point_count == 3


def test_distance_is_to_the_nearest_camera():
    adaptation = controller(far_prune_distance=30.5)
    centers = adaptation.model.means
    expected = torch.cdist(centers, CAMERAS).amin(1) > 30.5
    assert adaptation.far_from_cameras(centers).tolist() == expected.tolist() == [False, False, False, True, True]


def test_disabled_by_default_and_outside_prune_events():
    assert GaussianAdaptationConfig().far_prune_distance is None
    adaptation = controller(far_prune_distance=None)
    assert adaptation.propose_plan(PRUNE_EVENT).keep_indices.tolist() == [0, 1, 2, 3, 4]
    adaptation = controller()
    # Iteration 1100 is neither an opacity-prune nor a densification event.
    assert adaptation.propose_plan(1100).keep_indices.tolist() == [0, 1, 2, 3, 4]
    # Opacity pruning continues after densification ends, and so does far pruning.
    assert adaptation.propose_plan(20000).keep_indices.tolist() == [0, 1, 2]


def test_split_children_that_would_be_far_pruned_are_not_created():
    adaptation = controller(gradient_threshold=0.0, percent_dense=0.001)
    adaptation.gradient_sum.fill_(1.0)
    adaptation.gradient_denom.fill_(1.0)
    plan = adaptation.propose_plan(PRUNE_EVENT)
    # Near Gaussians split (scale > percent_dense * extent); far large ones only vanish.
    assert set(plan.split_indices.tolist()) == {1}
    assert 3 not in plan.split_indices.tolist() and 4 not in plan.split_indices.tolist()
    assert 3 not in plan.keep_indices.tolist() and 4 not in plan.keep_indices.tolist()


def test_requires_camera_centers():
    adaptation = controller()
    with pytest.raises(ValueError, match="camera centers"):
        GaussianAdaptationController(adaptation.model, adaptation.optimizer, adaptation.config, 1.0)


@pytest.mark.parametrize("distance, scale", [(0.0, 0.05), (-1.0, 0.05), (15.0, 0.0)])
def test_config_rejects_nonpositive_thresholds(tmp_path, distance, scale):
    config = ExperimentConfig()
    config.dataset.source_path = str(tmp_path)
    config.gaussian_adaptation.far_prune_distance = distance
    config.gaussian_adaptation.far_prune_scale = scale
    with pytest.raises(ValueError, match="far_prune"):
        config.validate()
    config.gaussian_adaptation.far_prune_distance = 15.0
    config.gaussian_adaptation.far_prune_scale = 0.05
    config.validate()
    assert math.isclose(config.as_dict()["gaussian_adaptation"]["far_prune_scale"], 0.05)


def test_sweep_and_report_name_far_variants():
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts/sweep_medium_fix.sbatch").read_text()
    function = script[script.index("variant_overrides() {"):script.index("\n}\n", script.index("variant_overrides() {")) + 3]
    names = script[script.index("FAR_VARIANTS=("):script.index('FAR_VARIANTS=("${SEEDED[@]}")') + 30]
    out = subprocess.run(["bash", "-c", function + names + '\nprintf "%s\\n" "${FAR_VARIANTS[@]}"\n'
                          'variant_overrides B0_chroma100_scov5e3_g700k_far15s050_bk_seed43'],
                         capture_output=True, text=True, check=True).stdout.splitlines()
    variants, overrides = out[:24], out[24:]
    assert len(set(variants)) == 24 and sum(v.endswith("_seed43") for v in variants) == 12
    assert "B0_chroma100_scov3e3_g700k_bk" in variants
    assert "joint.gaussian_adaptation.far_prune_distance=15.0" in overrides
    assert "joint.gaussian_adaptation.far_prune_scale=0.050" in overrides
    assert "medium_fix.surface_coverage_weight=5.0e-3" in overrides and "joint.seed=43" in overrides
    sys.path.insert(0, str(root / "scripts"))
    from medium_fix_report import baseline_name, describe, parse_variant
    assert parse_variant("B0_chroma100_scov3e3_g700k_far10s030_bk")[4] == "_chroma100_scov3e3_g700k_far10s030"
    assert baseline_name("B0_chroma100_scov3e3_g700k_far10s030_bk") == "B0_chroma100_scov3e3_g700k_bk"
    assert "skali > 0,03 dalej niż 10 j." in describe("B0_chroma100_scov3e3_g700k_far10s030_bk")
