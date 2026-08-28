from __future__ import annotations

import sys
import types
from types import SimpleNamespace

topology_stub = types.ModuleType("foamsurf_topology")
topology_stub.TriangulationFailedError = RuntimeError
sys.modules.setdefault("foamsurf_topology", topology_stub)

from foamsurf_model.voronoi_adaptation import VoronoiAdaptationController
from training import ExperimentConfig


def test_offset_and_capacity_freeze_control_the_schedule() -> None:
    config = ExperimentConfig.load(
        "configs/baseline_clean_joint.yaml"
    ).voronoi_adaptation
    config.adaptation_offset = 100
    controller = VoronoiAdaptationController.__new__(
        VoronoiAdaptationController
    )
    controller.config = config
    controller.model = SimpleNamespace(point_count=4_096)

    assert controller.should_adapt(1_100)
    assert controller.should_adapt(1_600)
    assert not controller.should_adapt(1_000)
    assert not controller.should_adapt(1_500)

    config.freeze_at_capacity = True
    controller.model.point_count = config.max_points
    assert not controller.should_collect(1_700)
    assert not controller.should_adapt(2_100)
