"""COLMAP photos are kept once on the GPU: the 3-channel foreground, and for an opaque photo a constant alpha
(no copy), with exactly the values of the image file."""
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from foamsurf_data.colmap import ColmapDataset

DATA = Path(os.environ.get("FS_DATASETS", "/shared/sets/datasets/SeathruNeRF_dataset")) / "IUI3-RedSea-Undistorted"
pytestmark = pytest.mark.skipif(not DATA.is_dir(), reason="needs the SeaThru dataset")


def test_photos_stored_once_with_unchanged_values():
    dataset = ColmapDataset(DATA, resolution=4, device="cpu")
    camera = dataset.train[0]
    H, W = camera.height, camera.width
    with Image.open(next((DATA / "images_4").glob(camera.name + ".*"))) as image:  # the prescaled photos
        rgba = image.convert("RGBA")
    assert rgba.size == (W, H)
    expected = torch.from_numpy(np.asarray(rgba, dtype=np.float32).copy()) / 255.0
    torch.testing.assert_close(camera.foreground, expected[..., :3] * expected[..., 3:4], atol=0, rtol=0)
    torch.testing.assert_close(camera.ground_truth((0.0, 0.0, 0.0)), expected[..., :3], atol=0, rtol=0)
    assert camera.alpha.shape == (H, W, 1) and bool((camera.alpha == 1).all())
    assert camera.alpha.untyped_storage().nbytes() == 4  # one float, broadcast
    assert camera.foreground.untyped_storage().nbytes() == H * W * 3 * 4  # no RGBA tensor kept alive
    assert camera.alpha.reshape(-1, 1)[torch.tensor([0, H * W - 1])].tolist() == [[1.0], [1.0]]
