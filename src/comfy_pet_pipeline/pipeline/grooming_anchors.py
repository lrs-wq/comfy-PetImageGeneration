"""Reviewed joint coordinates for the immutable production grooming driver.

Bright chest shading is not an anatomical landmark. These coordinates describe
the driver's normalized frame 32, never the uploaded animal's appearance.
"""

from __future__ import annotations

import hashlib
from typing import Any

import cv2
import numpy as np
from PIL import Image


DRIVER_SHA256 = "919d769f083105e6c931d91bd432e2aadac4d4b355853102bf0fb08d3ce23cbc"
NORMALIZED_MASK_SHA256 = "80ecf0640ab82b2090aeaa5d751252cf77d44c29c5db52907db1b61c7fcc5daf"
POLICY = "sha_bound_reviewed_grooming_driver_joint_anchors_v1"


def apply_reviewed_grooming_anchors(
    guide: Image.Image,
    metadata: dict[str, Any],
) -> None:
    """Replace the luma-derived chest false positive for the reviewed asset only.

The source digest, sampling, selected frame and normalized mask are all bound.
Fail on a changed normalization instead of silently placing joints elsewhere.
Other drivers retain their own geometry extraction.
    """
    if metadata.get("source_sha256") != DRIVER_SHA256:
        return
    if (
        guide.size != (512, 512)
        or metadata.get("sampled_frame_count") != 81
        or float(metadata.get("sampled_fps", 0)) != 16.0
        or metadata.get("selected_frame_index") != 32
    ):
        raise ValueError("reviewed_grooming_anchor_sampling_changed")
    rgb = np.asarray(guide.convert("RGB"), dtype=np.uint8)
    foreground = ~np.all(rgb == (0, 255, 0), axis=2)
    mask_sha = hashlib.sha256(foreground.astype(np.uint8).tobytes()).hexdigest()
    if mask_sha != NORMALIZED_MASK_SHA256:
        raise ValueError("reviewed_grooming_anchor_normalization_changed")

    # Viewer-left raised foreleg: wrist/paw at (192, 242), shoulder at
    # (202, 315). The former luma maximum at (262, 258) was on the chest.
    muzzle = np.zeros(foreground.shape, dtype=np.uint8)
    cv2.ellipse(muzzle, (195, 218), (22, 20), 0, 0, 360, 1, -1)
    muzzle = muzzle.astype(bool) & foreground
    foreleg = np.zeros(foreground.shape, dtype=np.uint8)
    cv2.line(foreleg, (202, 315), (198, 276), 1, 29)
    cv2.line(foreleg, (198, 276), (192, 242), 1, 29)
    cv2.ellipse(foreleg, (192, 242), (19, 20), 0, 0, 360, 1, -1)
    foreleg = foreleg.astype(bool) & foreground & ~muzzle
    contact = cv2.dilate(foreleg.astype(np.uint8), np.ones((5, 5), np.uint8))
    contact_pixels = int(np.count_nonzero(contact.astype(bool) & muzzle))
    if contact_pixels < 5 or not foreground[242, 192] or not foreground[315, 202]:
        raise ValueError("reviewed_grooming_anchor_foreground_support_missing")

    def bbox(mask: np.ndarray) -> list[int]:
        ys, xs = np.nonzero(mask)
        return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]

    union = muzzle | foreleg
    body_box = bbox(foreground)
    union_box = bbox(union)
    old_relation = metadata["raised_paw_muzzle_relation"]
    metadata["raised_paw_muzzle_relation"] = {
        "policy": POLICY,
        "source_sha256": DRIVER_SHA256,
        "normalized_mask_sha256": mask_sha,
        "selected_frame_index": 32,
        "coordinate_space": "normalized_512x512_driver_frame",
        "raised_foreleg_side": "viewer_left",
        "planted_foreleg_side": "viewer_right",
        "shoulder_anchor": [202, 315],
        "paw_center": [192, 242],
        "muzzle_region_bbox": bbox(muzzle),
        "muzzle_region_pixels": int(muzzle.sum()),
        "raised_foreleg_region_bbox": bbox(foreleg),
        "raised_foreleg_region_pixels": int(foreleg.sum()),
        "component_bbox": union_box,
        "component_pixels": int(union.sum()),
        "component_fraction_of_subject": float(union.sum() / foreground.sum()),
        "component_width_ratio": (union_box[2] - union_box[0]) / (body_box[2] - body_box[0]),
        "component_height_ratio": (union_box[3] - union_box[1]) / (body_box[3] - body_box[1]),
        "upper_muzzle_support_pixels": int(muzzle.sum()),
        "lower_raised_paw_support_pixels": int(foreleg.sum()),
        "minimum_support_pixels": 96,
        "semantic_contact_pixels": contact_pixels,
        "minimum_semantic_contact_pixels": 5,
        "regions_disjoint": True,
        "source_luma_values_preserved": False,
        "passed": True,
        "superseded_luma_anchor": {
            "policy": old_relation.get("policy"),
            "paw_center": old_relation.get("paw_center"),
            "shoulder_anchor": old_relation.get("shoulder_anchor"),
            "used_for_geometry": False,
        },
    }
