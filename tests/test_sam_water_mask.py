"""Water/seabed masks of the SeaThru test views (scripts/sam_water_mask.py, configs/water_sam/)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import sam_water_mask as sam  # noqa: E402

MASKS = ROOT / "configs/water_sam"


def test_choose_prefers_the_proposal_agreeing_with_the_prompts():
    masks = np.zeros((3, 10, 10), bool)
    masks[0, :5] = True          # top half: agrees with both prompts
    masks[1] = True              # everything: seabed point inside
    masks[2, :, :5] = True       # left half: water point outside
    points = np.array([[7.0, 1.0], [7.0, 8.0]])  # (x, y): water at the top right, seabed at the bottom right
    labels = np.array([1, 0])
    assert sam.choose(masks, np.array([0.5, 0.99, 0.9]), points, labels) == 0
    masks[1, 5:] = False  # now also agrees: the higher score wins
    assert sam.choose(masks, np.array([0.5, 0.99, 0.9]), points, labels) == 1


def test_border_takes_the_labels_inside_and_specks_are_removed():
    mask = np.zeros((40, 40), bool)
    mask[:20] = True
    mask[:3, :] = False  # SAM's thin strip at the top edge
    mask[30, 30] = True  # a speck of water in the seabed
    cleaned = sam.clean(sam.extend_to_border(mask), min_region=0.01)
    assert cleaned[:20].all() and not cleaned[20:].any()


def test_view_without_water_prompts_has_no_water():
    class Fails:
        def set_image(self, image):
            raise AssertionError("SAM must not run without water prompts")
    mask, score = sam.segment(Fails(), np.zeros((4, 6, 3), np.uint8), dict(water=[], seabed=[]))
    assert mask.shape == (4, 6) and not mask.any() and np.isnan(score)


def test_load_masks_merges_scene_files_and_rejects_duplicates(tmp_path):
    np.savez(tmp_path / "a.npz", meta="{}", V1=np.ones((2, 2), bool))
    np.savez(tmp_path / "b.npz", meta="{}", V2=np.zeros((2, 2), bool))
    assert sorted(sam.load_masks(tmp_path)) == ["V1", "V2"]
    assert list(sam.load_masks(tmp_path / "a.npz")) == ["V1"] and sam.load_masks(tmp_path / "none") == {}
    np.savez(tmp_path / "c.npz", meta="{}", V1=np.ones((2, 2), bool))
    with pytest.raises(ValueError, match="two files"):
        sam.load_masks(tmp_path)


@pytest.mark.parametrize("scene", sorted(sam.PROMPTS))
def test_committed_masks_match_prompts_and_images(scene):
    path = MASKS / f"{scene}.npz"
    with np.load(path) as data:
        meta = json.loads(str(data["meta"]))
        masks = {name: data[name] for name in data.files if name != "meta"}
    assert sorted(masks) == sorted(sam.PROMPTS[scene]) and meta["scene"] == scene
    for view, mask in masks.items():
        assert mask.dtype == bool
        assert mask.any() == bool(sam.PROMPTS[scene][view]["water"])  # no prompts: no water
        image = Path(sam.DATASETS) / scene / "images" / f"{view}.png"
        if image.is_file():
            assert mask.shape == Image.open(image).size[::-1]


@pytest.mark.parametrize("scene", sorted(sam.PROMPTS))
def test_prompted_views_are_the_loaders_test_split(scene):
    source = Path(sam.DATASETS) / scene
    if not source.is_dir():
        pytest.skip("dataset not on this machine")
    sys.path.insert(0, str(ROOT))
    from foamsurf_data.colmap import ColmapDataset
    dataset = ColmapDataset(str(source), resolution=8, device="cpu", llffhold=8)
    assert [camera.name for camera in dataset.test] == list(sam.PROMPTS[scene])
