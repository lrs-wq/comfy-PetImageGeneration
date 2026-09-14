import copy
import hashlib
import json
from pathlib import Path

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
            "output_encoding": {
                "method": "compressed_blurred_quantized_neutral_anatomy",
                "gray_min": 120, "gray_max": 168, "quantum": 8,
            },
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
    assert relation["black_or_tool_semantics_preserved"] is False
    assert relation["output_encoding"]["method"] == "compressed_blurred_quantized_neutral_anatomy"
    assert relation["output_encoding"]["quantum"] == 8
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


def test_production_driver_passes_complete_pre_qwen_contract(tmp_path, monkeypatch):
    driver = Path(
        "/data/comfy-pet-pipeline/profile_assets/sleek_tabby_cat_v1/driving/"
        "20260808-user-white-model-v1/"
        "grooming_white_model_driver_512x512_81f_16fps.mp4"
    )
    if not driver.is_file():
        pytest.skip("The SHA-bound production driver is server-only media")
    import local_video_api as api

    monkeypatch.setattr(api, "OUTPUT_ROOT", tmp_path / "logs")
    guide = tmp_path / "guide.png"
    metadata = api.prepare_grooming_qwen_geometry_guide(
        "reviewed-anchor-contract-test", driver, guide, frames=81, fps=16.0,
    )
    private = metadata["private_pose_gate"]
    completion = metadata["private_completion_mask"]
    job = {"pose_reference_metadata": {"grooming": {
        "policy": api.GROOMING_QWEN_POSE_REFERENCE_POLICY,
        "generator_policy": api.GROOMING_QWEN_GEOMETRY_GUIDE_POLICY,
        "generator_metadata": metadata,
        "guide_sha256": metadata["guide_sha256"],
        "private_pose_gate_guide_path": private["guide_path"],
        "private_pose_gate_guide_sha256": private["guide_sha256"],
        "private_pose_gate_exposed_to_qwen": False,
        "private_completion_mask_path": completion["path"],
        "private_completion_mask_sha256": completion["sha256"],
        "private_completion_mask_exposed_to_qwen": False,
    }}}
    gate_path, metadata_path, completion_path = api.resolve_main_case_grooming_pose_gate_metadata(job, guide)
    assert gate_path.is_file() and completion_path.is_file()
    private_metadata = json.loads(metadata_path.read_text())
    relation = private_metadata["raised_paw_muzzle_relation"]
    assert relation["policy"] == anchors.POLICY
    assert relation["paw_center"] == [192, 242]
    # Reproduce the integration failure even when all anchor coordinates pass.
    del relation["output_encoding"]
    metadata_path.write_text(json.dumps(private_metadata))
    with pytest.raises(RuntimeError, match="lacks verified paw/muzzle geometry"):
        api.resolve_main_case_grooming_pose_gate_metadata(job, guide)
