from __future__ import annotations

import pytest
import torch


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="FoamSurf topology is a CUDA component",
)
def test_public_topology_api_is_foamsurf_named() -> None:
    import foamsurf_topology

    points = torch.randn(40, 3, device="cuda")
    topology = foamsurf_topology.Triangulation(points)

    assert foamsurf_topology.__name__ == "foamsurf_topology"
    assert topology.permutation().numel() == points.shape[0]
    assert topology.point_adjacency_offsets().numel() == points.shape[0] + 1
