import copy
import hashlib

import numpy as np
import pytest
from PIL import Image

from src.comfy_pet_pipeline.pipeline import grooming_anchors as anchors


@pytest.fixture
def reviewed_guide(monkeypatch):
    rgb = np.full((512, 512, 3), (0, 255, 0), dtype=np.uint8)
    rgb[80:450, 120:390] = 144
    mask = ~np.all(rgb == (0, 255, 0), axis=2)
    monkeypatch.setattr(
        anchors, "NORMALIZED_MASK_SHA256",
        hashlib.sha256(mask.astype(np.uint8).tobytes()).hexdigest(),
    )
    metadata = {
        "source_sha256": anchors.DRIVER_SHA256,
        "sampled_frame_count": 81,
        "sampled_fps": 16.0,
        "selected_frame_index": 32,
        "raised_paw_muzzle_relation": {
            "policy": "luma", "paw_center": [262, 258],
            "shoulder_anchor": [264, 348],
        },
    }
    return rgb, metadata


def test_bright_chest_cannot_move_reviewed_foreleg(reviewed_guide):
    rgb, metadata = reviewed_guide
    second = copy.deepcopy(metadata)
    anchors.apply_reviewed_grooming_anchors(Image.fromarray(rgb), metadata)
    # Lighting and material differ while the normalized geometry is identical.
    rgb[200:360, 225:280] = 250
    before = rgb.copy()
    anchors.apply_reviewed_grooming_anchors(Image.fromarray(rgb), second)
    relation = metadata["raised_paw_muzzle_relation"]
    assert relation == second["raised_paw_muzzle_relation"]
    assert relation["paw_center"][0] < 220
    assert relation["shoulder_anchor"][0] < 220
    assert relation["semantic_contact_pixels"] >= 5
    assert relation["regions_disjoint"] is True
    assert relation["superseded_luma_anchor"]["used_for_geometry"] is False
    assert np.array_equal(rgb, before)


def test_other_driver_does_not_inherit_reviewed_coordinates(reviewed_guide):
    rgb, metadata = reviewed_guide
    metadata["source_sha256"] = "another-driver"
    original = copy.deepcopy(metadata)
    anchors.apply_reviewed_grooming_anchors(Image.fromarray(rgb), metadata)
    assert metadata == original


@pytest.mark.parametrize("field,value", [
    ("selected_frame_index", 40), ("sampled_frame_count", 41),
    ("sampled_fps", 8.0),
])
def test_changed_timeline_cannot_reuse_coordinates(reviewed_guide, field, value):
    rgb, metadata = reviewed_guide
    metadata[field] = value
    with pytest.raises(ValueError, match="sampling_changed"):
        anchors.apply_reviewed_grooming_anchors(Image.fromarray(rgb), metadata)


def test_changed_normalization_cannot_reuse_coordinates(reviewed_guide):
    rgb, metadata = reviewed_guide
    rgb[100, 200] = (0, 255, 0)
    with pytest.raises(ValueError, match="normalization_changed"):
        anchors.apply_reviewed_grooming_anchors(Image.fromarray(rgb), metadata)
