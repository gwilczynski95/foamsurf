from __future__ import annotations

import json

import pytest
import torch

from training.radfoam_import import (
    RADFOAM_DENSITY_BETA,
    RADFOAM_SH_C0,
    RadFoamCheckpointInitializer,
)


def _write_fixture(tmp_path, *, labels=None):
    xyz = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )
    raw_density = torch.tensor([[-2.0], [0.0], [3.0]])
    dc_rgb = torch.tensor(
        [[0.1, 0.2, 0.3], [0.5, 0.6, 0.7], [-0.2, 1.2, 0.8]]
    )
    checkpoint = {
        "xyz": xyz,
        "density": raw_density,
        "color_dc": (dc_rgb - 0.5) / RADFOAM_SH_C0,
        "color_sh": torch.arange(3 * 45, dtype=torch.float32).reshape(3, 45),
        "adjacency": torch.tensor([1, 2, 0, 2, 0, 1]),
        "adjacency_offsets": torch.tensor([0, 2, 4, 6]),
    }
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    labels_path = tmp_path / "labels.pt"
    torch.save(
        {
            "labels": torch.tensor([1, 0, 2] if labels is None else labels),
            "class_names": {0: "empty", 1: "surface", 2: "medium", 3: "ambiguous"},
        },
        labels_path,
    )
    return checkpoint_path, labels_path, checkpoint


def test_import_preserves_sites_density_topology_labels_and_source_sh(tmp_path):
    checkpoint_path, labels_path, source = _write_fixture(tmp_path)
    loaded = RadFoamCheckpointInitializer(
        checkpoint_path,
        labels_path,
        device="cpu",
        bbox_min=(-1.0, -1.0, -1.0),
        bbox_max=(2.0, 2.0, 2.0),
    ).load()

    torch.testing.assert_close(loaded.medium.sites, source["xyz"])
    expected_sigma = torch.nn.functional.softplus(
        source["density"], beta=RADFOAM_DENSITY_BETA
    )
    torch.testing.assert_close(loaded.medium.densities, expected_sigma)
    assert loaded.adjacency.dtype == torch.uint32
    assert loaded.offsets.dtype == torch.uint32
    torch.testing.assert_close(loaded.adjacency.long(), source["adjacency"])
    torch.testing.assert_close(loaded.offsets.long(), source["adjacency_offsets"])
    assert loaded.labels.tolist() == [1, 0, 2]
    torch.testing.assert_close(loaded.source_color_sh, source["color_sh"])
    assert not bool(loaded.medium.guard_mask.any())


def test_import_maps_dc_color_and_reports_clipping(tmp_path):
    checkpoint_path, labels_path, _ = _write_fixture(tmp_path)
    loaded = RadFoamCheckpointInitializer(
        checkpoint_path,
        labels_path,
        device="cpu",
        bbox_min=(-1.0, -1.0, -1.0),
        bbox_max=(2.0, 2.0, 2.0),
        color_epsilon=1.0e-4,
    ).load()

    expected = torch.tensor(
        [[0.1, 0.2, 0.3], [0.5, 0.6, 0.7], [1.0e-4, 0.9999, 0.8]]
    )
    torch.testing.assert_close(loaded.medium.colors, expected, atol=1.0e-5, rtol=0)
    assert loaded.metadata["color_clipped_scalar_count"] == 2
    assert loaded.metadata["color_clipped_cell_count"] == 1


def test_import_rejects_misaligned_labels(tmp_path):
    checkpoint_path, labels_path, _ = _write_fixture(tmp_path, labels=[1, 2])
    with pytest.raises(ValueError, match="label count"):
        RadFoamCheckpointInitializer(
            checkpoint_path,
            labels_path,
            device="cpu",
            bbox_min=(-1.0, -1.0, -1.0),
            bbox_max=(2.0, 2.0, 2.0),
        ).load()


def test_import_rejects_invalid_topology(tmp_path):
    checkpoint_path, labels_path, checkpoint = _write_fixture(tmp_path)
    checkpoint["adjacency_offsets"][-1] = 5
    torch.save(checkpoint, checkpoint_path)
    with pytest.raises(ValueError, match="adjacency offset"):
        RadFoamCheckpointInitializer(
            checkpoint_path,
            labels_path,
            device="cpu",
            bbox_min=(-1.0, -1.0, -1.0),
            bbox_max=(2.0, 2.0, 2.0),
        ).load()


def test_debug_metadata_is_json_serialisable(tmp_path):
    checkpoint_path, labels_path, _ = _write_fixture(tmp_path)
    loaded = RadFoamCheckpointInitializer(
        checkpoint_path,
        labels_path,
        device="cpu",
        bbox_min=(-1.0, -1.0, -1.0),
        bbox_max=(2.0, 2.0, 2.0),
    ).load()
    json.dumps(loaded.metadata)
