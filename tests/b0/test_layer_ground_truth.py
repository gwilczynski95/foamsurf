from __future__ import annotations

import json

import numpy as np
from PIL import Image
import pytest
import torch

from evaluation.layer_metrics import decomposition_metrics
from foamsurf_data import BlenderDataset, LayerGroundTruth
from training.config import DatasetConfig, ExperimentConfig
from training.experiment_logging import wandb_image_key


def _write_test_split(root, name: str = "r_0", offset: float = 0.0) -> None:
    root.mkdir(parents=True)
    rgba = np.zeros((16, 16, 4), dtype=np.uint8)
    rgba[..., :3] = 128
    rgba[..., 3] = 255
    Image.fromarray(rgba).save(root / f"{name}.png")
    transform = np.eye(4, dtype=np.float32)
    transform[0, 3] = offset
    metadata = {
        "camera_angle_x": 0.6911112,
        "frames": [
            {
                "file_path": f"./{name}",
                "transform_matrix": transform.tolist(),
            }
        ],
    }
    (root / "transforms_test.json").write_text(json.dumps(metadata))


def test_layer_ground_truth_loads_only_aligned_test_views_on_cpu(
    tmp_path,
) -> None:
    both = tmp_path / "both"
    lego = tmp_path / "lego"
    clouds = tmp_path / "clouds"
    for root in (both, lego, clouds):
        _write_test_split(root)
    reference = BlenderDataset(
        both, device="cpu", splits=("test",)
    )

    layers = LayerGroundTruth.load(
        reference.test, lego, clouds, resolution=1
    )

    assert len(layers.lego_lego) == 1
    assert len(layers.lego_clouds) == 1
    assert layers.lego_lego[0].foreground.device.type == "cpu"


def test_layer_ground_truth_rejects_camera_mismatch(tmp_path) -> None:
    both = tmp_path / "both"
    lego = tmp_path / "lego"
    clouds = tmp_path / "clouds"
    _write_test_split(both)
    _write_test_split(lego, offset=0.1)
    _write_test_split(clouds)
    reference = BlenderDataset(
        both, device="cpu", splits=("test",)
    )

    with pytest.raises(ValueError, match="different camera pose"):
        LayerGroundTruth.load(reference.test, lego, clouds, resolution=1)


def test_decomposition_metrics_detect_zero_cross_branch_leakage() -> None:
    lego_alpha = torch.zeros((16, 16, 1))
    cloud_alpha = torch.zeros_like(lego_alpha)
    lego_alpha[:8, :, :] = 1.0
    cloud_alpha[:, 8:, :] = 1.0
    lego_target = lego_alpha.expand(-1, -1, 3).clone()
    cloud_target = cloud_alpha.expand(-1, -1, 3).clone()

    values, masks = decomposition_metrics(
        lego_target,
        lego_alpha,
        cloud_target,
        cloud_alpha,
        lego_target,
        lego_alpha,
        cloud_target,
        cloud_alpha,
        alpha_threshold=0.01,
    )

    assert values["lego_lego/psnr"] == pytest.approx(120.0)
    assert values["lego_clouds/psnr"] == pytest.approx(120.0)
    assert values["entanglement/medium_alpha_on_lego_only"] == 0.0
    assert values["entanglement/gaussian_alpha_on_cloud_only"] == 0.0
    assert values["entanglement/gaussian_dominance_on_lego_only"] == 1.0
    assert values["entanglement/medium_dominance_on_cloud_only"] == 1.0
    assert masks["overlap"].float().mean() == pytest.approx(0.25)


def test_layer_paths_are_paired_and_wandb_sections_are_direct() -> None:
    config = ExperimentConfig(
        dataset=DatasetConfig(
            source_path="both",
            gaussian_ground_truth_path="lego",
        )
    )
    with pytest.raises(ValueError, match="must be set together"):
        config.validate()

    assert wandb_image_key("lego_lego/view_00") == "lego_lego/view_00"
    assert wandb_image_key("lego_clouds/view_00") == "lego_clouds/view_00"
    assert (
        wandb_image_key("comparison/view_00")
        == "images/comparison/view_00"
    )
