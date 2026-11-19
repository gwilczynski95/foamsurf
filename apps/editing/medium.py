"""Refine medium cells for inference-time editing and style transfer."""
from __future__ import annotations

import torch


def subdivide_medium(trainer, factor: int, jitter: float, seed: int) -> None:
    """Every non-guard cell becomes `factor` cells: the site and factor-1 children at site + N(0, (jitter * d)^2)
    (d = distance to the nearest site), all with the parent's coefficients and colour. The medium looks almost the
    same and has factor x more cells to paint with."""
    from scipy.spatial import cKDTree
    from foamsurf_model.medium import VoronoiMediumModel
    from foamsurf_model.topology import VoronoiTopologyController
    medium = trainer.model.medium
    sites = medium.sites.detach().cpu()
    guard = medium.guard_mask.cpu()
    inner = torch.nonzero(~guard).flatten()
    distance, _ = cKDTree(sites.numpy()).query(sites.numpy(), k=2)
    radius = torch.as_tensor(distance[:, 1], dtype=torch.float32)[inner]
    generator = torch.Generator().manual_seed(seed)
    offsets = torch.randn(inner.numel(), factor - 1, 3, generator=generator) * (jitter * radius)[:, None, None]
    children = (sites[inner][:, None] + offsets).reshape(-1, 3)
    children = torch.minimum(torch.maximum(children, medium.bbox_min.cpu()), medium.bbox_max.cpu())
    parent = torch.cat((inner, inner.repeat_interleave(factor - 1), torch.nonzero(guard).flatten()))
    new_sites = torch.cat((sites[inner], children, sites[guard]))
    state = {"sites": new_sites, "raw_density": medium.raw_density.detach().cpu()[parent],
             "raw_color": medium.raw_color.detach().cpu()[parent]}
    metadata = dict(medium.checkpoint_metadata(), point_count=int(new_sites.shape[0]))
    trainer.model.medium = VoronoiMediumModel.from_checkpoint(state, metadata, medium.bbox_min, medium.bbox_max,
                                                              trainer.device)
    trainer.config.model.medium_points = int(new_sites.shape[0])
    adaptation = trainer.config.voronoi_adaptation  # config validation: medium_points <= the adaptation budget
    if adaptation.max_points is not None:
        adaptation.max_points = max(adaptation.max_points, int(new_sites.shape[0]))
    trainer.topology = VoronoiTopologyController(trainer.model.medium)
    trainer.topology.synchronize_renderer(trainer.renderer, trainer.model.medium)
    print(f"medium split x{factor}: {sites.shape[0]} -> {new_sites.shape[0]} cells", flush=True)
