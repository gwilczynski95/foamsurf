"""Import an inference RadFoam checkpoint as a FoamSurf Voronoi medium.

This module deliberately does not perform the later surface-cell removal or
Gaussian seeding step.  It only translates and validates a frozen RadFoam
snapshot, preserves its checkpoint topology, and provides class-mask debug
renders for inspecting a previously computed separation.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
import yaml

from foamsurf import VoronoiScene
from foamsurf_model.medium import VoronoiMediumModel


RADFOAM_DENSITY_BETA = 10.0
RADFOAM_SH_C0 = 0.28209479177387814
EXPECTED_CLASS_NAMES = {0: "empty", 1: "surface", 2: "medium", 3: "ambiguous"}


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    """Numerically stable inverse of beta=1 softplus for positive values."""
    if bool((value <= 0).any()):
        raise ValueError("activated RadFoam density must be strictly positive")
    return value + torch.log(-torch.expm1(-value))


def _logit(value: torch.Tensor) -> torch.Tensor:
    return torch.log(value) - torch.log1p(-value)


def _normalise_class_names(names: Mapping[object, object]) -> dict[int, str]:
    result = {int(key): str(value).lower() for key, value in names.items()}
    for label, name in EXPECTED_CLASS_NAMES.items():
        if label in result and result[label] != name:
            raise ValueError(
                f"label {label} must mean {name!r}, got {result[label]!r}"
            )
    required = {0, 1, 2}
    if not required.issubset(result):
        raise ValueError(
            "class_names must define EMPTY=0, SURFACE=1 and MEDIUM=2"
        )
    return result


def _validate_csr(
    adjacency: torch.Tensor, offsets: torch.Tensor, point_count: int
) -> None:
    if adjacency.ndim != 1 or offsets.ndim != 1:
        raise ValueError("RadFoam adjacency and offsets must be one-dimensional")
    if offsets.numel() != point_count + 1:
        raise ValueError(
            "RadFoam adjacency_offsets must contain point_count + 1 entries"
        )
    offsets64 = offsets.to(dtype=torch.int64, device="cpu")
    if int(offsets64[0]) != 0:
        raise ValueError("RadFoam adjacency_offsets must start at zero")
    if bool((offsets64[1:] < offsets64[:-1]).any()):
        raise ValueError("RadFoam adjacency_offsets must be non-decreasing")
    if int(offsets64[-1]) != adjacency.numel():
        raise ValueError("last adjacency offset must equal adjacency length")
    adjacency64 = adjacency.to(dtype=torch.int64, device="cpu")
    if adjacency64.numel() and (
        int(adjacency64.min()) < 0 or int(adjacency64.max()) >= point_count
    ):
        raise ValueError("RadFoam adjacency contains an invalid site index")


def _tensor_image(image: torch.Tensor) -> Image.Image:
    pixels = (
        image.detach().float().cpu().clamp(0.0, 1.0).numpy() * 255.0
    ).round().astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def _titled(image: Image.Image, title: str, header_height: int = 34) -> Image.Image:
    tile = Image.new("RGB", (image.width, image.height + header_height), "black")
    tile.paste(image, (0, header_height))
    draw = ImageDraw.Draw(tile)
    box = draw.textbbox((0, 0), title)
    width = box[2] - box[0]
    draw.text(((image.width - width) // 2, 10), title, fill="white")
    return tile


@dataclass
class ImportedRadFoamState:
    """A translated FoamSurf medium plus the exact RadFoam CSR topology."""

    medium: VoronoiMediumModel
    adjacency: torch.Tensor
    offsets: torch.Tensor
    labels: torch.Tensor
    class_names: dict[int, str]
    source_color_sh: torch.Tensor
    metadata: dict[str, object]

    def synchronize_renderer(self, renderer) -> None:
        """Install the imported checkpoint topology in a FoamSurf renderer."""
        renderer.set_voronoi_geometry(
            self.medium.render_scene(), self.adjacency, self.offsets
        )

    def class_mask(self, name: str) -> torch.Tensor:
        name = name.lower()
        matching = [label for label, value in self.class_names.items() if value == name]
        if len(matching) != 1:
            raise ValueError(f"class {name!r} is not uniquely defined")
        return self.labels == matching[0]


class RadFoamCheckpointInitializer:
    """Load RadFoam inference state for FoamSurf initialization and debugging.

    RadFoam uses ``softplus(beta=10)`` for density, while FoamSurf uses
    ``softplus(beta=1)``.  The loader converts through activated sigma so the
    physical density is preserved.  RadFoam's degree-3 directional colour
    cannot be represented by the current constant-RGB FoamSurf medium; its DC
    RGB is imported and out-of-range values are clamped before conversion to
    FoamSurf logits.  The complete source SH tensor remains available in
    :attr:`ImportedRadFoamState.source_color_sh` on CPU.
    """

    REQUIRED_CHECKPOINT_KEYS = {
        "xyz",
        "density",
        "color_dc",
        "color_sh",
        "adjacency",
        "adjacency_offsets",
    }

    def __init__(
        self,
        checkpoint_path: str | Path,
        labels_path: str | Path,
        *,
        device: str | torch.device = "cuda",
        config_path: str | Path | None = None,
        bbox_min: Sequence[float] | None = None,
        bbox_max: Sequence[float] | None = None,
        color_epsilon: float = 1.0e-6,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.labels_path = Path(labels_path).resolve()
        self.device = torch.device(device)
        self.config_path = (
            Path(config_path).resolve() if config_path is not None else None
        )
        self.explicit_bbox_min = bbox_min
        self.explicit_bbox_max = bbox_max
        if not 0.0 < color_epsilon < 0.5:
            raise ValueError("color_epsilon must lie in (0, 0.5)")
        self.color_epsilon = float(color_epsilon)

    def _bounds(self, sites: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, str]:
        if (self.explicit_bbox_min is None) != (self.explicit_bbox_max is None):
            raise ValueError("bbox_min and bbox_max must be supplied together")
        if self.explicit_bbox_min is not None:
            lower = torch.as_tensor(self.explicit_bbox_min, dtype=torch.float32)
            upper = torch.as_tensor(self.explicit_bbox_max, dtype=torch.float32)
            source = "explicit"
        else:
            config_path = self.config_path
            if config_path is None:
                candidate = self.checkpoint_path.parent.parent / "config.yaml"
                if candidate.is_file():
                    config_path = candidate
            if config_path is not None:
                config = yaml.safe_load(config_path.read_text())
                lower = torch.tensor(config["synthetic_bbox_min"], dtype=torch.float32)
                upper = torch.tensor(config["synthetic_bbox_max"], dtype=torch.float32)
                source = str(config_path)
            else:
                lower = sites.amin(dim=0)
                upper = sites.amax(dim=0)
                source = "checkpoint_site_extrema"
        if lower.shape != (3,) or upper.shape != (3,):
            raise ValueError("bounding-box limits must contain three values")
        if not bool((upper > lower).all()):
            raise ValueError("every bbox_max component must exceed bbox_min")
        return lower, upper, source

    def load(self) -> ImportedRadFoamState:
        checkpoint = torch.load(
            self.checkpoint_path, map_location="cpu", weights_only=True
        )
        if not isinstance(checkpoint, Mapping):
            raise ValueError("RadFoam checkpoint must be a mapping")
        missing = self.REQUIRED_CHECKPOINT_KEYS - set(checkpoint)
        if missing:
            raise ValueError(f"RadFoam checkpoint is missing keys: {sorted(missing)}")

        sites = checkpoint["xyz"].detach().float().contiguous()
        raw_density = checkpoint["density"].detach().float().contiguous()
        color_dc = checkpoint["color_dc"].detach().float().contiguous()
        color_sh = checkpoint["color_sh"].detach().float().contiguous()
        if sites.ndim != 2 or sites.shape[1] != 3 or sites.shape[0] == 0:
            raise ValueError("RadFoam xyz must have shape [N, 3] with N > 0")
        point_count = int(sites.shape[0])
        expected = {
            "density": (point_count, 1),
            "color_dc": (point_count, 3),
        }
        for name, shape in expected.items():
            if tuple(checkpoint[name].shape) != shape:
                raise ValueError(f"RadFoam {name} must have shape {shape}")
        if color_sh.ndim != 2 or color_sh.shape[0] != point_count:
            raise ValueError("RadFoam color_sh must have shape [N, C]")
        for name, value in (
            ("xyz", sites),
            ("density", raw_density),
            ("color_dc", color_dc),
            ("color_sh", color_sh),
        ):
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"RadFoam {name} contains non-finite values")

        adjacency = checkpoint["adjacency"].detach()
        offsets = checkpoint["adjacency_offsets"].detach()
        _validate_csr(adjacency, offsets, point_count)

        label_payload = torch.load(
            self.labels_path, map_location="cpu", weights_only=True
        )
        if not isinstance(label_payload, Mapping):
            raise ValueError("label file must contain a mapping")
        if "labels" not in label_payload or "class_names" not in label_payload:
            raise ValueError("label file must contain labels and class_names")
        labels = label_payload["labels"].detach().to(torch.int64).reshape(-1)
        if labels.numel() != point_count:
            raise ValueError(
                f"label count {labels.numel()} does not match {point_count} sites"
            )
        class_names = _normalise_class_names(label_payload["class_names"])
        unknown = set(int(value) for value in torch.unique(labels)) - set(class_names)
        if unknown:
            raise ValueError(f"labels contain undefined class ids: {sorted(unknown)}")

        sigma = F.softplus(raw_density, beta=RADFOAM_DENSITY_BETA)
        foamsurf_raw_density = _inverse_softplus(sigma)
        dc_rgb_unclamped = 0.5 + RADFOAM_SH_C0 * color_dc
        dc_rgb = dc_rgb_unclamped.clamp(
            self.color_epsilon, 1.0 - self.color_epsilon
        )
        foamsurf_raw_color = _logit(dc_rgb)
        lower, upper, bounds_source = self._bounds(sites)

        medium = VoronoiMediumModel.from_checkpoint(
            state={
                "sites": sites,
                "raw_density": foamsurf_raw_density,
                "raw_color": foamsurf_raw_color,
            },
            metadata={"point_count": point_count, "guard_count": 0},
            bbox_min=lower,
            bbox_max=upper,
            device=self.device,
        )
        labels_device = labels.to(self.device)
        adjacency_device = adjacency.to(self.device, dtype=torch.uint32).contiguous()
        offsets_device = offsets.to(self.device, dtype=torch.uint32).contiguous()
        clipped = (dc_rgb_unclamped != dc_rgb)
        counts = {
            name: int((labels == label).sum())
            for label, name in class_names.items()
        }
        metadata: dict[str, object] = {
            "checkpoint_path": str(self.checkpoint_path),
            "labels_path": str(self.labels_path),
            "point_count": point_count,
            "class_counts": counts,
            "bbox_min": [float(value) for value in lower],
            "bbox_max": [float(value) for value in upper],
            "bbox_source": bounds_source,
            "density_translation": "softplus(raw, beta=10) -> inverse_softplus(beta=1)",
            "density_max_abs_error": float(
                (medium.densities.detach().cpu() - sigma).abs().max()
            ),
            "color_translation": "RadFoam SH DC -> constant RGB -> FoamSurf logit",
            "color_clipped_scalar_count": int(clipped.sum()),
            "color_clipped_scalar_fraction": float(clipped.float().mean()),
            "color_clipped_cell_count": int(clipped.any(dim=1).sum()),
            "guard_count": 0,
        }
        return ImportedRadFoamState(
            medium=medium,
            adjacency=adjacency_device,
            offsets=offsets_device,
            labels=labels_device,
            class_names=class_names,
            source_color_sh=color_sh,
            metadata=metadata,
        )

    @torch.no_grad()
    def render_debug(
        self,
        imported: ImportedRadFoamState,
        renderer,
        camera,
        output_dir: str | Path,
        *,
        fixed_density: float = 0.15,
        bright_color: Sequence[float] = (1.0, 1.0, 1.0),
        support_far: float | None = None,
    ) -> Path:
        """Render the requested 3x3 full/class/support diagnostic panel."""
        if fixed_density <= 0.0:
            raise ValueError("fixed_density must be positive")
        bright = torch.as_tensor(
            bright_color,
            dtype=torch.float32,
            device=imported.medium.sites.device,
        )
        if bright.shape != (3,) or not bool(torch.isfinite(bright).all()):
            raise ValueError("bright_color must contain three finite values")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        imported.synchronize_renderer(renderer)
        bundle = camera.rays()
        rays = torch.cat((bundle.origins, bundle.directions), dim=-1).contiguous()
        base_scene = imported.medium.render_scene()
        black = (0.0, 0.0, 0.0)
        statistics: dict[str, dict[str, float | int]] = {}

        if support_far is None:
            lower = imported.medium.bbox_min
            upper = imported.medium.bbox_max
            corners = torch.stack(
                [
                    torch.tensor((x, y, z), device=lower.device)
                    for x in (float(lower[0]), float(upper[0]))
                    for y in (float(lower[1]), float(upper[1]))
                    for z in (float(lower[2]), float(upper[2]))
                ]
            )
            support_far = 1.05 * float(
                (corners - camera.origin.reshape(1, 3)).norm(dim=1).max()
            )
        if support_far <= renderer.near:
            raise ValueError("support_far must exceed renderer.near")

        def render(
            name: str,
            scene: VoronoiScene,
            far_override: float | None = None,
        ) -> Image.Image:
            original_far = renderer.far
            if far_override is not None:
                renderer.far = float(far_override)
            try:
                result = renderer.render_voronoi(rays, scene, background=black)
            finally:
                renderer.far = original_far
            intersections = result.num_intersections.detach().to(torch.int64)
            statistics[name] = {
                "maximum_intersections": int(intersections.max()),
                "mean_intersections": float(intersections.float().mean()),
                "fraction_at_renderer_limit": float(
                    (intersections >= renderer.max_intersections).float().mean()
                ),
                "far": float(far_override or original_far),
            }
            image = _tensor_image(result.image)
            image.save(output_dir / f"{name}.png")
            return image

        full = render("full_voronoi", base_scene)
        panels: dict[str, tuple[Image.Image, Image.Image]] = {}
        zero_density = torch.zeros_like(base_scene.densities)
        fixed = torch.full_like(base_scene.densities, float(fixed_density))
        bright_colors = bright.reshape(1, 3).expand_as(base_scene.colors)
        for class_name in ("surface", "empty", "medium"):
            mask = imported.class_mask(class_name).reshape(-1, 1)
            selected_scene = VoronoiScene(
                sites=base_scene.sites,
                densities=torch.where(mask, base_scene.densities, zero_density),
                colors=base_scene.colors,
            )
            support_scene = VoronoiScene(
                sites=base_scene.sites,
                densities=torch.where(mask, fixed, zero_density),
                colors=bright_colors,
            )
            panels[class_name] = (
                render(f"{class_name}_only", selected_scene),
                render(
                    f"{class_name}_support",
                    support_scene,
                    far_override=support_far,
                ),
            )

        target = _tensor_image(camera.ground_truth(black))
        target.save(output_dir / "ground_truth.png")
        rows = []
        for class_name in ("surface", "empty", "medium"):
            regular, support = panels[class_name]
            row = Image.new("RGB", (3 * full.width, full.height + 34), "black")
            for column, tile in enumerate(
                (
                    _titled(full, "FULL IMPORTED VORONOI"),
                    _titled(regular, f"{class_name.upper()} ONLY"),
                    _titled(support, f"{class_name.upper()} SUPPORT"),
                )
            ):
                row.paste(tile, (column * full.width, 0))
            rows.append(row)
        panel = Image.new("RGB", (rows[0].width, sum(row.height for row in rows)))
        y = 0
        for row in rows:
            panel.paste(row, (0, y))
            y += row.height
        panel_path = output_dir / "radfoam_class_debug_grid.png"
        panel.save(panel_path)

        debug_metadata = dict(imported.metadata)
        debug_metadata.update(
            {
                "camera_name": camera.name,
                "fixed_support_density": float(fixed_density),
                "support_far": float(support_far),
                "bright_support_color": [float(value) for value in bright],
                "panel_first_column": "full imported Voronoi render",
                "ground_truth_sidecar": "ground_truth.png",
                "render_statistics": statistics,
            }
        )
        (output_dir / "import_debug_metadata.json").write_text(
            json.dumps(debug_metadata, indent=2, sort_keys=True) + "\n"
        )
        return panel_path
