"""Build a fresh FoamSurf scene from a separated RadFoam checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import torch

from foamsurf_model.gaussians import SurfaceGaussianModel
from foamsurf_model.medium import VoronoiMediumModel
from foamsurf_model.scene import HybridSceneModel
from training.config import ExperimentConfig
from training.radfoam_import import RadFoamCheckpointInitializer


def _inverse_softplus(value: float) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float64)
    return float((tensor + torch.log(-torch.expm1(-tensor))).item())


def _surface_orientations(
    surface_sites: np.ndarray,
    selected_sites: np.ndarray,
    neighbor_count: int,
) -> np.ndarray:
    """Estimate a tangent frame with local PCA and return wxyz quaternions."""
    if surface_sites.shape[0] < 3:
        result = np.zeros((selected_sites.shape[0], 4), dtype=np.float32)
        result[:, 0] = 1.0
        return result
    count = min(max(3, int(neighbor_count)), surface_sites.shape[0])
    tree = cKDTree(surface_sites)
    _, indices = tree.query(selected_sites, k=count, workers=-1)
    if indices.ndim == 1:
        indices = indices[:, None]
    neighbors = surface_sites[indices]
    centered = neighbors - neighbors.mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centered, centered)
    _, eigenvectors = np.linalg.eigh(covariance)

    # eigh columns are ordered from the smallest eigenvalue.  The smallest
    # direction is the surface normal; the two others span the Gaussian disc.
    normal = eigenvectors[:, :, 0]
    tangent_x = eigenvectors[:, :, 1]
    tangent_y = np.cross(normal, tangent_x)
    tangent_y /= np.linalg.norm(tangent_y, axis=1, keepdims=True).clip(1.0e-12)
    rotation = np.stack((tangent_x, tangent_y, normal), axis=-1)
    quaternion_xyzw = Rotation.from_matrix(rotation).as_quat()
    quaternion_wxyz = quaternion_xyzw[:, [3, 0, 1, 2]]
    return quaternion_wxyz.astype(np.float32)


def _cells_touching_surface(
    surface_mask: torch.Tensor,
    adjacency: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """Return retained cells whose original Delaunay star touched SURFACE."""
    surface_mask = surface_mask.to(device="cpu", dtype=torch.bool)
    adjacency = adjacency.to(device="cpu", dtype=torch.int64)
    offsets = offsets.to(device="cpu", dtype=torch.int64)
    degrees = offsets[1:] - offsets[:-1]
    rows = torch.repeat_interleave(
        torch.arange(surface_mask.numel(), dtype=torch.int64), degrees
    )
    affected = torch.zeros_like(surface_mask)
    boundary_edges = surface_mask[adjacency] & ~surface_mask[rows]
    affected[rows[boundary_edges]] = True
    return affected


@dataclass(frozen=True)
class RadFoamHandoffResult:
    model: HybridSceneModel
    report: dict[str, Any]


@torch.no_grad()
def initialize_from_radfoam_prior(
    config: ExperimentConfig,
    device: torch.device | str,
) -> RadFoamHandoffResult:
    """Remove SURFACE cells, seed Gaussians there, and retain the full medium."""
    prior = config.radfoam_prior
    if not prior.enabled:
        raise ValueError("RadFoam prior initialization is not enabled")

    imported = RadFoamCheckpointInitializer(
        prior.checkpoint_path,
        prior.labels_path,
        device=device,
    ).load()
    surface_mask = imported.class_mask("surface").detach().cpu()
    medium_mask = imported.class_mask("medium").detach().cpu()
    empty_mask = imported.class_mask("empty").detach().cpu()
    ambiguous_count = int(
        sum(
            int((imported.labels == label).sum())
            for label, name in imported.class_names.items()
            if name == "ambiguous"
        )
    )
    if ambiguous_count:
        raise ValueError(
            "RadFoam handoff requires a hard EMPTY/SURFACE/MEDIUM partition; "
            f"found {ambiguous_count} AMBIGUOUS cells"
        )
    partition = surface_mask | medium_mask | empty_mask
    if not bool(partition.all()):
        raise ValueError("RadFoam labels do not form the expected hard partition")

    surface_indices = torch.nonzero(surface_mask, as_tuple=False).flatten()
    if prior.gaussian_count > surface_indices.numel():
        raise ValueError(
            f"requested {prior.gaussian_count} Gaussians from only "
            f"{surface_indices.numel()} SURFACE cells"
        )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)
    selection = torch.randperm(
        surface_indices.numel(), generator=generator
    )[: prior.gaussian_count]
    gaussian_source_indices = surface_indices[selection]

    sites_cpu = imported.medium.sites.detach().cpu().numpy()
    colors_cpu = imported.medium.colors.detach().cpu().numpy()
    selected_sites = sites_cpu[gaussian_source_indices.numpy()]
    selected_colors = colors_cpu[gaussian_source_indices.numpy()]
    surface_sites = sites_cpu[surface_mask.numpy()]
    gaussians = SurfaceGaussianModel(
        selected_sites,
        selected_colors,
        config.model,
        device,
    )
    orientations = _surface_orientations(
        surface_sites,
        selected_sites,
        prior.pca_neighbors,
    )
    gaussians.quaternions.copy_(
        torch.from_numpy(orientations).to(device=gaussians.means.device)
    )

    affected = _cells_touching_surface(
        surface_mask,
        imported.adjacency,
        imported.offsets,
    )
    keep_mask = ~surface_mask
    retained_count = int(keep_mask.sum())
    if retained_count != config.model.medium_points:
        raise ValueError(
            f"RadFoam prior retains {retained_count} cells but "
            f"model.medium_points declares {config.model.medium_points}"
        )
    affected_kept = affected[keep_mask]
    empty_kept = empty_mask[keep_mask]
    reset_kept = affected_kept | empty_kept
    keep_device = keep_mask.to(imported.medium.sites.device)
    raw_density = imported.medium.raw_density.detach()[keep_device].clone()
    raw_density[reset_kept.to(raw_density.device)] = _inverse_softplus(
        prior.reset_density
    )
    medium = VoronoiMediumModel.from_checkpoint(
        state={
            "sites": imported.medium.sites.detach()[keep_device].clone(),
            "raw_density": raw_density,
            "raw_color": imported.medium.raw_color.detach()[keep_device].clone(),
        },
        metadata={
            "point_count": retained_count,
            "guard_count": 0,
        },
        bbox_min=imported.medium.bbox_min,
        bbox_max=imported.medium.bbox_max,
        device=device,
    )

    model = HybridSceneModel(gaussians, medium)
    report: dict[str, Any] = {
        "mode": "radfoam_surface_medium_handoff",
        "checkpoint_path": str(imported.metadata["checkpoint_path"]),
        "labels_path": str(imported.metadata["labels_path"]),
        "source_cell_count": int(imported.medium.point_count),
        "source_surface_count": int(surface_mask.sum()),
        "source_medium_count": int(medium_mask.sum()),
        "source_empty_count": int(empty_mask.sum()),
        "removed_surface_count": int(surface_mask.sum()),
        "retained_voronoi_count": retained_count,
        "retained_medium_count": int(medium_mask.sum()),
        "retained_empty_count": int(empty_mask.sum()),
        "affected_retained_count": int(affected_kept.sum()),
        "reset_empty_count": int(empty_kept.sum()),
        "reset_union_count": int(reset_kept.sum()),
        "reset_density": float(prior.reset_density),
        "gaussian_count": int(gaussians.point_count),
        "gaussian_initial_opacity": float(config.model.gaussian_initial_opacity),
        "gaussian_sampling": "uniform_without_replacement_over_surface_sites",
        "gaussian_orientation": "local_surface_site_pca",
        "pca_neighbors": int(prior.pca_neighbors),
        "voronoi_population_adaptation": False,
        "all_retained_voronoi_parameters_trainable": True,
        "source_import": imported.metadata,
    }
    return RadFoamHandoffResult(model=model, report=report)
