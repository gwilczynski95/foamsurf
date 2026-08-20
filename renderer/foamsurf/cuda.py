from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path

import torch

from .types import GaussianScene, HybridRender, VoronoiScene


@dataclass(frozen=True)
class VoronoiTopology:
    adjacency: torch.Tensor
    offsets: torch.Tensor
    site_data_ptr: int
    point_count: int


@dataclass(frozen=True)
class NativeDiagnostics:
    """Reduced B0.1 statistics; no rays-by-primitives allocation."""

    ray: torch.Tensor
    gaussian: torch.Tensor
    medium: torch.Tensor

    @property
    def gaussian_weight(self) -> torch.Tensor:
        return self.ray[..., 0:1]

    @property
    def medium_weight(self) -> torch.Tensor:
        return self.ray[..., 1:2]

    @property
    def medium_path_length(self) -> torch.Tensor:
        return self.ray[..., 2:3]

    @property
    def weighted_medium_path_length(self) -> torch.Tensor:
        return self.ray[..., 3:4]

    @property
    def gaussian_event_count(self) -> torch.Tensor:
        return self.ray[..., 4:5]

    @property
    def medium_segment_count(self) -> torch.Tensor:
        return self.ray[..., 5:6]

    @property
    def final_transmittance(self) -> torch.Tensor:
        return self.ray[..., 6:7]

    @property
    def traversal_truncated(self) -> torch.Tensor:
        return self.ray[..., 7:8]


@dataclass(frozen=True)
class NativeHybridRender:
    image: torch.Tensor
    alpha: torch.Tensor
    transmittance: torch.Tensor
    surface_depth: torch.Tensor
    surface_index: torch.Tensor
    hit_depths: torch.Tensor
    hit_indices: torch.Tensor
    num_intersections: torch.Tensor
    diagnostics: NativeDiagnostics | None = None

    def as_common(self) -> HybridRender:
        return HybridRender(
            image=self.image,
            alpha=self.alpha,
            transmittance=self.transmittance,
            surface_depth=self.surface_depth,
            surface_index=self.surface_index,
        )


class _HybridCudaFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        bindings,
        points,
        densities,
        medium_colors,
        adjacency,
        offsets,
        rays,
        start_points,
        hit_depths,
        hit_indices,
        means,
        log_scales,
        quaternions,
        gaussian_colors,
        opacity_logits,
        raw_kappa,
        background,
        near_plane,
        far_plane,
        threshold,
        max_intersections,
    ):
        tensors = (
            points,
            densities,
            medium_colors,
            adjacency,
            offsets,
            rays,
            start_points,
            hit_depths,
            hit_indices,
            means,
            log_scales,
            quaternions,
            gaussian_colors,
            opacity_logits,
            raw_kappa,
            background,
        )
        rgba, num_intersections = bindings.hybrid_forward(
            list(tensors),
            near_plane,
            far_plane,
            threshold,
            max_intersections,
        )
        ctx.bindings = bindings
        ctx.near_plane = near_plane
        ctx.far_plane = far_plane
        ctx.threshold = threshold
        ctx.max_intersections = max_intersections
        ctx.save_for_backward(*tensors, rgba)
        ctx.mark_non_differentiable(num_intersections)
        return rgba, num_intersections

    @staticmethod
    def backward(ctx, rgba_gradient, _num_intersections_gradient):
        *tensors, rgba = ctx.saved_tensors
        gradients = ctx.bindings.hybrid_backward(
            list(tensors),
            rgba,
            rgba_gradient.contiguous(),
            ctx.near_plane,
            ctx.far_plane,
            ctx.threshold,
            ctx.max_intersections,
        )
        del ctx.bindings
        return (
            None,
            gradients[0],
            gradients[1],
            gradients[2],
            None,
            None,
            None,
            None,
            None,
            None,
            gradients[3],
            gradients[4],
            gradients[5],
            gradients[6],
            gradients[7],
            gradients[8],
            None,
            None,
            None,
            None,
            None,
        )


class NativeHybridRenderer:
    """OptiX hit generation plus fused Voronoi/Gaussian CUDA compositing.

    Geometry synchronization is deliberately separate from rendering. Call
    :meth:`set_gaussian_geometry` whenever Gaussian geometry changes and
    :meth:`set_voronoi_geometry` whenever sites or triangulation change.
    Attribute-only updates do not require either call.
    """

    def __init__(
        self,
        max_batch_size: int,
        near: float = 0.0,
        far: float = 100.0,
        transmittance_threshold: float = 0.001,
        max_intersections: int = 1024,
        number_of_sides: int = 8,
        gaussian_radius_squared: float = 11.3449,
        gaussian_hit_buffer_size: int = 16,
        flare_renderer_directory: str | Path | None = None,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if far <= near:
            raise ValueError("far must exceed near")
        self.max_batch_size = int(max_batch_size)
        self.near = float(near)
        self.far = float(far)
        self.transmittance_threshold = float(transmittance_threshold)
        self.max_intersections = int(max_intersections)
        self._bindings = self._load_foamsurf_bindings()
        self._flare_renderer_directory = flare_renderer_directory
        self._number_of_sides = int(number_of_sides)
        self._gaussian_radius_squared = float(gaussian_radius_squared)
        self.gaussian_hit_buffer_size = int(gaussian_hit_buffer_size)
        if self.gaussian_hit_buffer_size not in {16, 32, 64, 128, 256}:
            raise ValueError(
                "gaussian_hit_buffer_size must be one of 16, 32, 64, 128, 256"
            )
        self._optix = None
        self._gaussian_geometry_is_set = False
        self._gaussian_signature: tuple[int, int] | None = None
        self._voronoi_topology: VoronoiTopology | None = None

    @staticmethod
    def _load_foamsurf_bindings():
        try:
            return importlib.import_module("foamsurf._C")
        except ImportError as error:
            raise RuntimeError(
                "FoamSurf CUDA bindings are not built; run ./build.sh from "
                "the standalone renderer directory"
            ) from error

    @staticmethod
    def _load_optix(directory: str | Path | None):
        if directory is None:
            directory = Path(__file__).resolve().parent / "_optix"
        directory = Path(directory)
        module = directory / "PYOPTIXFLARERENDERER.so"
        if not module.is_file():
            raise FileNotFoundError(
                f"bundled FoamSurf OptiX backend is not built: {module}; "
                "run ./build.sh from the standalone renderer directory"
            )
        spec = importlib.util.spec_from_file_location(
            "PYOPTIXFLARERENDERER", module
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load FoamSurf OptiX backend: {module}")
        native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native)
        return native

    @torch.no_grad()
    def set_gaussian_geometry(self, scene: GaussianScene) -> None:
        scene.validate()
        self._require_cuda_float32(scene.means, "Gaussian scene")
        if self._optix is None:
            directory = self._flare_renderer_directory
            if directory is None:
                directory = (
                    Path(__file__).resolve().parent
                    / f"_optix_k{self.gaussian_hit_buffer_size}"
                )
            optix_module = self._load_optix(directory)
            compiled_size = int(getattr(optix_module, "HIT_BUFFER_SIZE", -1))
            if compiled_size != self.gaussian_hit_buffer_size:
                raise RuntimeError(
                    "OptiX hit-buffer mismatch: requested "
                    f"K={self.gaussian_hit_buffer_size}, backend reports "
                    f"K={compiled_size} ({directory})"
                )
            self._optix = optix_module.CPyOptiXFLARERenderer(
                self._number_of_sides,
                self._gaussian_radius_squared,
                self.max_batch_size,
            )
        self._optix.SetGeometry(
            scene.means.contiguous(),
            scene.log_scales.exp().contiguous(),
            scene.quaternions.contiguous(),
            scene.opacity_logits.sigmoid().contiguous(),
            (1.0 + torch.nn.functional.softplus(scene.raw_kappa)).contiguous(),
        )
        self._gaussian_geometry_is_set = True
        self._gaussian_signature = (scene.means.data_ptr(), scene.means.size(0))

    @torch.no_grad()
    def set_voronoi_geometry(
        self,
        scene: VoronoiScene,
        adjacency: torch.Tensor,
        offsets: torch.Tensor,
    ) -> None:
        scene.validate()
        self._require_cuda_float32(scene.sites, "Voronoi scene")
        if adjacency.dtype != torch.uint32 or offsets.dtype != torch.uint32:
            raise ValueError("Voronoi adjacency and offsets must be uint32")
        if not adjacency.is_cuda or not offsets.is_cuda:
            raise ValueError("Voronoi adjacency and offsets must be on CUDA")
        self._voronoi_topology = VoronoiTopology(
            adjacency=adjacency.contiguous().clone(),
            offsets=offsets.contiguous().clone(),
            site_data_ptr=scene.sites.data_ptr(),
            point_count=scene.sites.size(0),
        )

    @staticmethod
    def _require_cuda_float32(tensor: torch.Tensor, name: str) -> None:
        if not tensor.is_cuda or tensor.dtype != torch.float32:
            raise ValueError(f"{name} must use CUDA float32 tensors")


    @staticmethod
    @torch.no_grad()
    def _nearest_sites(
        points: torch.Tensor,
        queries: torch.Tensor,
        chunk_size: int = 65536,
    ) -> torch.Tensor:
        """Return exact nearest-site indices without a RadFoam dependency."""
        best_distance = torch.full(
            (queries.size(0),), float("inf"), device=queries.device
        )
        best_index = torch.zeros(
            queries.size(0), dtype=torch.long, device=queries.device
        )
        for begin in range(0, points.size(0), chunk_size):
            chunk = points[begin : begin + chunk_size]
            distance = (
                queries[:, None, :] - chunk[None, :, :]
            ).square().sum(-1)
            chunk_distance, chunk_index = distance.min(dim=1)
            replace = chunk_distance < best_distance
            best_distance = torch.where(
                replace, chunk_distance, best_distance
            )
            best_index = torch.where(
                replace, chunk_index + begin, best_index
            )
        return best_index

    def _empty_medium(self, reference: torch.Tensor):
        return (
            reference.new_empty((0, 3)),
            reference.new_empty((0, 1)),
            reference.new_empty((0, 3)),
            torch.empty(0, dtype=torch.uint32, device=reference.device),
            torch.zeros(1, dtype=torch.uint32, device=reference.device),
        )

    def _empty_gaussians(self, reference: torch.Tensor):
        return (
            reference.new_empty((0, 3)),
            reference.new_empty((0, 2)),
            reference.new_empty((0, 4)),
            reference.new_empty((0, 3)),
            reference.new_empty((0, 1)),
            reference.new_empty((0, 1)),
        )

    def render(
        self,
        rays: torch.Tensor,
        gaussians: GaussianScene | None = None,
        voronoi: VoronoiScene | None = None,
        background: torch.Tensor | tuple[float, float, float] = (0.0, 0.0, 0.0),
        start_points: torch.Tensor | None = None,
        collect_diagnostics: bool = False,
        diagnostic_target: torch.Tensor | None = None,
    ) -> NativeHybridRender:
        if gaussians is None and voronoi is None:
            raise ValueError("at least one representation must be provided")
        if rays.shape[-1] != 6:
            raise ValueError("rays must have shape [..., 6]")
        self._require_cuda_float32(rays, "rays")
        leading_shape = rays.shape[:-1]
        rays = rays.reshape(-1, 6).contiguous()
        rays = torch.cat(
            (rays[:, :3], torch.nn.functional.normalize(rays[:, 3:], dim=-1)),
            dim=-1,
        ).contiguous()
        ray_count = rays.size(0)
        if ray_count > self.max_batch_size:
            raise ValueError(
                f"received {ray_count} rays, max_batch_size is {self.max_batch_size}"
            )
        background_tensor = torch.as_tensor(
            background, dtype=torch.float32, device=rays.device
        )
        if background_tensor.shape != (3,):
            raise ValueError("background must contain exactly three values")

        if voronoi is None:
            points, densities, medium_colors, adjacency, offsets = (
                self._empty_medium(rays)
            )
            start_points = torch.zeros(
                ray_count, dtype=torch.uint32, device=rays.device
            )
        else:
            voronoi.validate()
            self._require_cuda_float32(voronoi.sites, "Voronoi scene")
            if self._voronoi_topology is None:
                raise RuntimeError(
                    "Voronoi geometry is not initialized; call "
                    "set_voronoi_geometry() outside render"
                )
            if (
                self._voronoi_topology.site_data_ptr != voronoi.sites.data_ptr()
                or self._voronoi_topology.point_count != voronoi.sites.size(0)
            ):
                raise RuntimeError("Voronoi scene does not match synchronized geometry")
            points, densities, medium_colors = (
                voronoi.sites,
                voronoi.densities,
                voronoi.colors,
            )
            adjacency = self._voronoi_topology.adjacency
            offsets = self._voronoi_topology.offsets
            if start_points is None:
                with torch.no_grad():
                    origins = rays[:, :3]
                    unique, inverse = torch.unique(
                        origins, dim=0, return_inverse=True
                    )
                    nearest = self._nearest_sites(points, unique)
                    start_points = nearest[inverse].to(torch.uint32)
            else:
                start_points = start_points.reshape(-1).to(
                    device=points.device, dtype=torch.uint32
                )

        if gaussians is None:
            means, log_scales, quaternions, colors, opacity, kappa = (
                self._empty_gaussians(rays)
            )
            hit_depths = rays.new_empty((0, ray_count))
            hit_indices = torch.empty(
                (0, ray_count), dtype=torch.int32, device=rays.device
            )
        else:
            gaussians.validate()
            self._require_cuda_float32(gaussians.means, "Gaussian scene")
            if not self._gaussian_geometry_is_set:
                raise RuntimeError(
                    "Gaussian geometry is not initialized; call "
                    "set_gaussian_geometry() outside render"
                )
            if self._gaussian_signature != (
                gaussians.means.data_ptr(),
                gaussians.means.size(0),
            ):
                raise RuntimeError("Gaussian scene does not match synchronized geometry")
            means, log_scales, quaternions, colors, opacity, kappa = (
                gaussians.means,
                gaussians.log_scales,
                gaussians.quaternions,
                gaussians.colors,
                gaussians.opacity_logits,
                gaussians.raw_kappa,
            )
            minimum_depth = torch.full(
                (ray_count,), self.near, dtype=torch.float32, device=rays.device
            )
            hit_depths, hit_indices = self._optix.Render(
                rays[:, :3].contiguous(),
                rays[:, 3:].contiguous(),
                minimum_depth,
                ray_count,
            )

        rgba, num_intersections = _HybridCudaFunction.apply(
            self._bindings,
            points,
            densities,
            medium_colors,
            adjacency,
            offsets,
            rays,
            start_points,
            hit_depths,
            hit_indices,
            means,
            log_scales,
            quaternions,
            colors,
            opacity,
            kappa,
            background_tensor,
            self.near,
            self.far,
            self.transmittance_threshold,
            self.max_intersections,
        )
        diagnostics = None
        if collect_diagnostics:
            if diagnostic_target is None:
                ray_errors = rays.new_zeros((ray_count,))
            else:
                expected_shape = (*leading_shape, 3)
                if diagnostic_target.shape != expected_shape:
                    raise ValueError(
                        f"diagnostic_target must have shape {expected_shape}"
                    )
                self._require_cuda_float32(
                    diagnostic_target, "diagnostic_target"
                )
                ray_errors = (
                    rgba[:, :3] - diagnostic_target.reshape(-1, 3)
                ).abs().mean(dim=-1).detach().contiguous()
            with torch.no_grad():
                ray_stats, gaussian_stats, medium_stats = (
                    self._bindings.hybrid_diagnostics(
                        [
                            points, densities, medium_colors, adjacency, offsets,
                            rays, start_points, hit_depths, hit_indices, means,
                            log_scales, quaternions, colors, opacity, kappa,
                            background_tensor,
                            ray_errors,
                        ],
                        self.near,
                        self.far,
                        self.transmittance_threshold,
                        self.max_intersections,
                    )
                )
            diagnostics = NativeDiagnostics(
                ray=ray_stats.reshape(*leading_shape, 8),
                gaussian=gaussian_stats.reshape(-1, 4),
                medium=medium_stats.reshape(-1, 5),
            )

        if hit_depths.size(0):
            valid_hits = (
                (hit_indices >= 0)
                & (hit_depths >= self.near)
                & (hit_depths <= self.far)
            )
            masked_depths = torch.where(
                valid_hits,
                hit_depths,
                torch.full_like(hit_depths, float("inf")),
            )
            surface_depth, slots = masked_depths.min(dim=0)
            surface_index = torch.gather(hit_indices, 0, slots[None]).squeeze(0)
            surface_depth = torch.where(
                torch.isfinite(surface_depth),
                surface_depth,
                torch.full_like(surface_depth, -1.0),
            )
        else:
            surface_depth = torch.full(
                (ray_count,), -1.0, dtype=torch.float32, device=rays.device
            )
            surface_index = torch.full(
                (ray_count,), -1, dtype=torch.int32, device=rays.device
            )
        return NativeHybridRender(
            image=rgba[:, :3].reshape(*leading_shape, 3),
            alpha=rgba[:, 3:].reshape(*leading_shape, 1),
            transmittance=(1.0 - rgba[:, 3:]).reshape(*leading_shape, 1),
            surface_depth=surface_depth.reshape(*leading_shape, 1),
            surface_index=surface_index.reshape(*leading_shape, 1),
            hit_depths=hit_depths,
            hit_indices=hit_indices,
            num_intersections=num_intersections.reshape(*leading_shape, 1),
            diagnostics=diagnostics,
        )

    def render_gaussians(
        self,
        rays: torch.Tensor,
        scene: GaussianScene,
        background: torch.Tensor | tuple[float, float, float] = (0.0, 0.0, 0.0),
        collect_diagnostics: bool = False,
    ) -> NativeHybridRender:
        return self.render(
            rays,
            gaussians=scene,
            background=background,
            collect_diagnostics=collect_diagnostics,
        )

    def render_voronoi(
        self,
        rays: torch.Tensor,
        scene: VoronoiScene,
        background: torch.Tensor | tuple[float, float, float] = (0.0, 0.0, 0.0),
        start_points: torch.Tensor | None = None,
    ) -> NativeHybridRender:
        return self.render(
            rays,
            voronoi=scene,
            background=background,
            start_points=start_points,
        )
