#!/usr/bin/env python3
"""Fail-closed, full-cycle audit for final eating videos.

This module is deliberately independent from ``local_video_api``.  It samples the
whole video, obtains a reliable alpha matte (embedded alpha or rembg), recovers
light foreground pixels that rembg can omit on a measured uniform studio
background, and then runs the existing production mouth/bowl contact evaluator.

The recovery path never edits decoded RGB and never paints or closes a contact
gap.  A pixel can receive recovered alpha only from its measured RGB distance to
the row-wise studio-background model.  Connected-component filtering may discard
pixels, but it never adds support that was absent from both rembg and that measured
distance field.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image


POLICY = "final_eating_full_cycle_video_v1"
VERSION = "1.0.0"
DEFAULT_SAMPLE_COUNT = 17

# Thresholds are normalized where possible so that the audit is not tied to
# 512x512.  The duration/frame floors are tied to the locked Main_case runtime:
# 81 frames at 16 fps.  Sixty-five frames / four seconds is the conservative
# lower tolerance for a result that still contains approach, eating, and return.
THRESHOLDS: dict[str, Any] = {
    "sample_count": DEFAULT_SAMPLE_COUNT,
    "minimum_frame_count": 65,
    "minimum_duration_seconds": 4.0,
    "alpha_threshold": 64,
    "primary_alpha_threshold": 32,
    "background_strip_fraction": 0.03125,
    "background_inlier_distance": 8.0,
    "background_inlier_fraction_min": 0.80,
    "recovery_alpha_start_distance": 7.0,
    "recovery_alpha_full_distance": 28.0,
    "recovery_component_core_distance": 18.0,
    "large_component_fraction_min": 0.0022,
    "lower_component_fraction_min": 0.0004,
    "lower_component_y_min": 0.66,
    "lower_component_x_min": 0.10,
    "lower_component_x_max": 0.90,
    "primary_foreground_fraction_min": 0.01,
    "primary_foreground_fraction_max": 0.92,
    "primary_largest_component_fraction_min": 0.50,
    "final_foreground_fraction_min": 0.03,
    "final_foreground_fraction_max": 0.85,
    "high_contrast_coverage_min": 0.985,
    "lower_high_contrast_coverage_min": 0.985,
    "recovered_canvas_fraction_max": 0.30,
    "retained_component_count_max": 5,
    "endpoint_window_samples": 2,
    "core_start_fraction": 0.25,
    "core_end_fraction": 0.75,
    "core_contact_count_min": 5,
    "core_contact_longest_run_min": 5,
    "head_lowering_fraction_min": 0.02,
    "endpoint_head_height_delta_fraction_max": 0.08,
    "foreground_margin_fraction_min": 0.02,
    "foreground_bbox_width_fraction_max": 0.96,
    "foreground_bbox_height_fraction_max": 0.96,
    "foreground_area_fraction_max": 0.72,
    "bowl_center_span_fraction_max": 0.04,
    "bowl_center_step_fraction_max": 0.03,
    "bowl_area_cv_max": 0.12,
    "bowl_width_cv_max": 0.10,
    "bowl_height_cv_max": 0.14,
    "bowl_consecutive_iou_min": 0.55,
}

THRESHOLD_RATIONALE = {
    "sampling": (
        "17 evenly spaced samples include both endpoints and divide the full "
        "Main_case action into 16 equal transitions."
    ),
    "duration": (
        "The locked standard is 81 frames at 16 fps; 65 frames and four seconds "
        "retain at least 80% of that approach/eat/return envelope."
    ),
    "contact": (
        "Five consecutive core samples cover at least four sample intervals "
        "(one quarter of the audited cycle), while first/last samples must remain "
        "non-contact."
    ),
    "head_cycle": (
        "A 2% frame-height drop must be observed from both the opening and closing "
        "head heights; the two endpoint heights may differ by at most 8%."
    ),
    "segmentation": (
        "Uniform-background recovery starts above 7 RGB-distance units and is "
        "fully opaque only at 28; at least 98.5% of high-contrast foreground must "
        "remain covered. RGB is copied bit-exact and only alpha may change."
    ),
    "framing": (
        "A 2% safety margin plus 96% bbox and 72% area ceilings reject edge contact "
        "and abnormal close-ups while retaining normal full-body framing."
    ),
    "bowl": (
        "Center movement, area/size variation and mask IoU are normalized temporal "
        "stability checks; every sample must still contain an unclipped bowl and "
        "food core."
    ),
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(contiguous)).hexdigest()


def sample_indices(frame_count: int, sample_count: int = DEFAULT_SAMPLE_COUNT) -> list[int]:
    if frame_count <= 0:
        raise ValueError("video_has_no_frames")
    if sample_count <= 0:
        raise ValueError("sample_count_must_be_positive")
    count = min(frame_count, sample_count)
    return sorted(
        {int(value) for value in np.rint(np.linspace(0, frame_count - 1, count))}
    )


def probe_video(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    capture = cv2.VideoCapture(str(resolved))
    if not capture.isOpened():
        raise ValueError(f"video_open_failed:{resolved}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    if frame_count <= 0 or width <= 0 or height <= 0 or fps <= 0:
        raise ValueError("video_metadata_invalid")
    return {
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "fps": round(fps, 6),
        "duration_seconds": round(frame_count / fps, 6),
    }


def decode_frame(path: str | Path, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"video_open_failed:{path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, bgr = capture.read()
    capture.release()
    if not ok or bgr is None:
        raise ValueError(f"video_frame_read_failed:{frame_index}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _bbox(mask: np.ndarray) -> list[int] | None:
    rows, columns = np.where(mask)
    if not len(columns):
        return None
    return [
        int(columns.min()),
        int(rows.min()),
        int(columns.max()) + 1,
        int(rows.max()) + 1,
    ]


def _geometry(mask: np.ndarray) -> dict[str, Any]:
    height, width = mask.shape
    box = _bbox(mask)
    if box is None:
        return {
            "bbox": None,
            "area": 0,
            "area_fraction": 0.0,
            "margins": None,
            "touches_edge": False,
            "bbox_width_fraction": 0.0,
            "bbox_height_fraction": 0.0,
        }
    left, top, right, bottom = box
    margins = {
        "left": left / width,
        "top": top / height,
        "right": (width - right) / width,
        "bottom": (height - bottom) / height,
    }
    return {
        "bbox": box,
        "area": int(np.count_nonzero(mask)),
        "area_fraction": round(float(np.mean(mask)), 8),
        "margins": {key: round(float(value), 8) for key, value in margins.items()},
        "touches_edge": bool(
            np.any(mask[0, :])
            or np.any(mask[-1, :])
            or np.any(mask[:, 0])
            or np.any(mask[:, -1])
        ),
        "bbox_width_fraction": round((right - left) / width, 8),
        "bbox_height_fraction": round((bottom - top) / height, 8),
    }


def _largest_component_fraction(mask: np.ndarray) -> tuple[float, int]:
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    areas = [int(stats[label, cv2.CC_STAT_AREA]) for label in range(1, count)]
    total = int(np.count_nonzero(mask))
    return (max(areas, default=0) / max(1, total), len(areas))


def validate_primary_alpha(alpha: np.ndarray, thresholds: dict[str, Any]) -> dict[str, Any]:
    hard = alpha >= int(thresholds["primary_alpha_threshold"])
    foreground_fraction = float(np.mean(hard))
    largest_fraction, component_count = _largest_component_fraction(hard)
    border = np.concatenate((hard[0, :], hard[-1, :], hard[:, 0], hard[:, -1]))
    issues: list[str] = []
    if not np.any(alpha < 250):
        issues.append("primary_alpha_is_opaque")
    if not (
        float(thresholds["primary_foreground_fraction_min"])
        <= foreground_fraction
        <= float(thresholds["primary_foreground_fraction_max"])
    ):
        issues.append("primary_foreground_fraction_invalid")
    if largest_fraction < float(thresholds["primary_largest_component_fraction_min"]):
        issues.append("primary_component_support_unreliable")
    return {
        "reliable": not issues,
        "issues": issues,
        "foreground_fraction": round(foreground_fraction, 8),
        "largest_component_fraction": round(float(largest_fraction), 8),
        "component_count": component_count,
        "border_foreground_fraction": round(float(np.mean(border)), 8),
    }


@lru_cache(maxsize=2)
def _rembg_session(model: str) -> Any:
    try:
        from rembg import new_session
    except Exception as exc:  # noqa: BLE001 - fail closed with a machine-readable cause
        raise RuntimeError(f"rembg_unavailable:{exc}") from exc
    return new_session(model)


def remove_background_rembg(
    rgb: np.ndarray,
    *,
    frame_index: int,
    model: str = "u2net",
) -> dict[str, Any]:
    del frame_index
    try:
        from rembg import remove
    except Exception as exc:  # noqa: BLE001 - fail closed with a machine-readable cause
        raise RuntimeError(f"rembg_unavailable:{exc}") from exc
    result = remove(Image.fromarray(rgb, "RGB"), session=_rembg_session(model))
    if isinstance(result, Image.Image):
        rgba = np.asarray(result.convert("RGBA"), dtype=np.uint8)
    else:
        rgba = np.asarray(result, dtype=np.uint8)
        if rgba.ndim != 3 or rgba.shape[2] != 4:
            raise RuntimeError("rembg_result_is_not_rgba")
    # Only rembg alpha is trusted.  Its returned RGB is intentionally discarded.
    rgba = np.dstack((rgb, rgba[..., 3]))
    return {"rgba": rgba, "method": f"rembg_{model}"}


def _call_background_remover(
    remover: Callable[..., Any],
    rgb: np.ndarray,
    *,
    frame_index: int,
    model: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    kwargs: dict[str, Any] = {}
    try:
        parameters = inspect.signature(remover).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "frame_index" in parameters:
        kwargs["frame_index"] = frame_index
    if "model" in parameters:
        kwargs["model"] = model
    result = remover(rgb, **kwargs)
    provenance: dict[str, Any] = {}
    if isinstance(result, dict):
        image_value = result.get("rgba", result.get("image"))
        provenance = {key: value for key, value in result.items() if key not in {"rgba", "image"}}
    else:
        image_value = result
    if isinstance(image_value, Image.Image):
        rgba = np.asarray(image_value.convert("RGBA"), dtype=np.uint8)
    else:
        rgba = np.asarray(image_value, dtype=np.uint8)
    if rgba.ndim != 3 or rgba.shape[:2] != rgb.shape[:2] or rgba.shape[2] != 4:
        raise RuntimeError("background_remover_result_invalid")
    # Enforce the bit-exact RGB contract even when an injected remover returns
    # composited or premultiplied color.
    rgba = np.dstack((rgb, rgba[..., 3]))
    return rgba, provenance


def recover_uniform_studio_foreground(
    rgb: np.ndarray,
    primary_rgba: np.ndarray,
    *,
    thresholds: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Recover alpha omissions from measured studio-background distance only."""

    policy = dict(THRESHOLDS if thresholds is None else thresholds)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("decoded_rgb_must_be_uint8_hwc3")
    if (
        primary_rgba.dtype != np.uint8
        or primary_rgba.ndim != 3
        or primary_rgba.shape[:2] != rgb.shape[:2]
        or primary_rgba.shape[2] != 4
    ):
        raise ValueError("primary_rgba_shape_or_type_invalid")
    height, width = rgb.shape[:2]
    primary_alpha = primary_rgba[..., 3]
    primary_check = validate_primary_alpha(primary_alpha, policy)

    strip_width = max(4, int(round(width * float(policy["background_strip_fraction"]))))
    edge_samples = np.concatenate((rgb[:, :strip_width], rgb[:, -strip_width:]), axis=1)
    row_background = np.median(edge_samples, axis=1).astype(np.float32)
    blur_height = max(3, int(round(height * 0.06))) | 1
    row_background = cv2.GaussianBlur(
        row_background.reshape(height, 1, 3),
        (1, blur_height),
        0,
    ).reshape(height, 3)
    distance = np.linalg.norm(
        rgb.astype(np.float32) - row_background[:, None, :],
        axis=2,
    )
    edge_distance = np.concatenate(
        (distance[:, :strip_width], distance[:, -strip_width:]), axis=1
    )
    background_inlier_fraction = float(
        np.mean(edge_distance <= float(policy["background_inlier_distance"]))
    )

    start = float(policy["recovery_alpha_start_distance"])
    full = float(policy["recovery_alpha_full_distance"])
    if full <= start:
        raise ValueError("invalid_recovery_distance_thresholds")
    recovered_alpha = np.clip((distance - start) / (full - start) * 255.0, 0, 255).astype(
        np.uint8
    )
    combined_alpha = np.maximum(primary_alpha, recovered_alpha)

    # No close, dilation, interpolation, or hand-authored contact shape is used.
    # Components are selected from pixels already supported by rembg or a strong
    # measured background difference.  The final alpha remains the unmodified max
    # of those two scalar evidence fields at selected pixels only.
    core = (primary_alpha >= int(policy["primary_alpha_threshold"])) | (
        distance >= float(policy["recovery_component_core_distance"])
    )
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        core.astype(np.uint8), connectivity=8
    )
    selected = np.zeros(core.shape, dtype=bool)
    retained: list[dict[str, Any]] = []
    large_minimum = max(32, int(round(height * width * float(policy["large_component_fraction_min"]))))
    lower_minimum = max(16, int(round(height * width * float(policy["lower_component_fraction_min"]))))
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        left = int(stats[label, cv2.CC_STAT_LEFT])
        top = int(stats[label, cv2.CC_STAT_TOP])
        component_width = int(stats[label, cv2.CC_STAT_WIDTH])
        component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        right = left + component_width
        bottom = top + component_height
        lower_central = bool(
            area >= lower_minimum
            and bottom >= height * float(policy["lower_component_y_min"])
            and right >= width * float(policy["lower_component_x_min"])
            and left <= width * float(policy["lower_component_x_max"])
        )
        reason = "large_foreground" if area >= large_minimum else "lower_prop" if lower_central else None
        if reason:
            selected |= labels == label
            retained.append(
                {
                    "area": area,
                    "bbox": [left, top, right, bottom],
                    "reason": reason,
                }
            )

    final_alpha = combined_alpha.copy()
    final_alpha[~selected] = 0
    final_rgba = np.dstack((rgb, final_alpha))
    rgb_bit_exact = bool(np.array_equal(final_rgba[..., :3], rgb))
    hard = final_alpha >= int(policy["alpha_threshold"])
    high_contrast = distance >= float(policy["recovery_alpha_full_distance"])
    lower_roi = np.zeros(hard.shape, dtype=bool)
    lower_roi[
        int(round(height * 0.55)) : max(int(round(height * 0.55)) + 1, int(round(height * 0.98))),
        int(round(width * 0.10)) : max(int(round(width * 0.10)) + 1, int(round(width * 0.90))),
    ] = True
    high_lower = high_contrast & lower_roi
    overall_coverage = float(np.count_nonzero(hard & high_contrast)) / float(
        max(1, np.count_nonzero(high_contrast))
    )
    lower_coverage = float(np.count_nonzero(hard & high_lower)) / float(
        max(1, np.count_nonzero(high_lower))
    )
    recovered = hard & (primary_alpha < int(policy["alpha_threshold"]))
    final_geometry = _geometry(hard)
    issues = list(primary_check["issues"])
    if background_inlier_fraction < float(policy["background_inlier_fraction_min"]):
        issues.append("studio_background_not_uniform_enough")
    if overall_coverage < float(policy["high_contrast_coverage_min"]):
        issues.append("high_contrast_foreground_coverage_incomplete")
    if lower_coverage < float(policy["lower_high_contrast_coverage_min"]):
        issues.append("lower_prop_high_contrast_coverage_incomplete")
    if float(np.mean(recovered)) > float(policy["recovered_canvas_fraction_max"]):
        issues.append("background_recovery_excessive")
    if len(retained) > int(policy["retained_component_count_max"]):
        issues.append("too_many_recovered_foreground_components")
    if not (
        float(policy["final_foreground_fraction_min"])
        <= float(final_geometry["area_fraction"])
        <= float(policy["final_foreground_fraction_max"])
    ):
        issues.append("recovered_foreground_fraction_invalid")
    if not rgb_bit_exact:
        issues.append("decoded_rgb_changed_by_segmentation")

    audit = {
        "reliable": not issues,
        "issues": list(dict.fromkeys(issues)),
        "method": "reliable_alpha_plus_uniform_studio_background_distance_recovery_v1",
        "primary": primary_check,
        "studio_background": {
            "model": "rowwise_median_of_outer_strips_smoothed_vertically",
            "strip_width": strip_width,
            "edge_inlier_fraction": round(background_inlier_fraction, 8),
            "edge_distance_p95": round(float(np.percentile(edge_distance, 95)), 6),
            "row_background_first": [round(float(value), 4) for value in row_background[0]],
            "row_background_middle": [
                round(float(value), 4) for value in row_background[height // 2]
            ],
            "row_background_last": [round(float(value), 4) for value in row_background[-1]],
        },
        "recovery": {
            "rgb_policy": "decoded_rgb_copied_bit_exact_only_alpha_changes",
            "rgb_bit_exact": rgb_bit_exact,
            "raw_rgb_sha256": sha256_array(rgb),
            "cutout_rgb_sha256": sha256_array(final_rgba[..., :3]),
            "primary_alpha_sha256": sha256_array(primary_alpha),
            "final_alpha_sha256": sha256_array(final_alpha),
            "contact_pixels_painted": False,
            "morphological_bridge_or_close_used": False,
            "recovered_pixel_count_ge64": int(np.count_nonzero(recovered)),
            "recovered_canvas_fraction_ge64": round(float(np.mean(recovered)), 8),
            "retained_components": retained,
            "overall_high_contrast_coverage": round(overall_coverage, 8),
            "lower_high_contrast_coverage": round(lower_coverage, 8),
        },
        "foreground": final_geometry,
    }
    return final_rgba, audit


def _default_contact_evaluator(path: Path) -> dict[str, Any]:
    from src.comfy_pet_pipeline.api.local_video import (
        eating_mouth_bowl_contact_quality,
    )

    return eating_mouth_bowl_contact_quality(path)


def _default_semantic_splitter(image: Image.Image) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
    from src.comfy_pet_pipeline.pipeline.subject_replacement import (
        make_eating_reference_identity_mask,
    )

    return make_eating_reference_identity_mask(image)


def semantic_frame_metrics(
    rgba: np.ndarray,
    *,
    splitter: Callable[[Image.Image], tuple[Image.Image, Image.Image, dict[str, Any]]],
) -> tuple[dict[str, Any], np.ndarray | None]:
    height, width = rgba.shape[:2]
    try:
        colored_image, _prop_exclusion, metadata = splitter(Image.fromarray(rgba, "RGBA"))
        colored = np.asarray(colored_image.convert("RGB"), dtype=np.uint8)
        subject = np.all(colored == (0, 0, 255), axis=2)
        bowl = np.all(colored == (255, 0, 0), axis=2)
        subject_box = _bbox(subject)
        bowl_box = _bbox(bowl)
        if subject_box is None or bowl_box is None:
            raise ValueError("semantic_subject_or_bowl_empty")
        bowl_left, bowl_top, bowl_right, bowl_bottom = bowl_box
        bowl_width = bowl_right - bowl_left
        bowl_height = bowl_bottom - bowl_top
        food_box = metadata.get("food_core_bbox")
        food_inside = bool(
            isinstance(food_box, list)
            and len(food_box) == 4
            and bowl_left <= int(food_box[0]) < int(food_box[2]) <= bowl_right
            and bowl_top <= int(food_box[1]) < int(food_box[3]) <= bowl_bottom
        )
        head_x0 = max(0, bowl_left - int(round(bowl_width * 0.25)))
        head_x1 = min(width, bowl_right + int(round(bowl_width * 0.25)))
        head_rows, _head_columns = np.where(subject[:bowl_top, head_x0:head_x1])
        head_y_p10 = float(np.percentile(head_rows, 10)) if head_rows.size else None
        subject_geometry = _geometry(subject)
        return (
            {
                "reliable": True,
                "error": None,
                "subject": subject_geometry,
                "bowl_bbox": bowl_box,
                "bowl_area": int(np.count_nonzero(bowl)),
                "bowl_center": [
                    round((bowl_left + bowl_right - 1) / 2.0, 6),
                    round((bowl_top + bowl_bottom - 1) / 2.0, 6),
                ],
                "bowl_width": bowl_width,
                "bowl_height": bowl_height,
                "bowl_clipped": bool(
                    bowl_left <= 0
                    or bowl_top <= 0
                    or bowl_right >= width
                    or bowl_bottom >= height
                ),
                "food_core_bbox": food_box,
                "food_core_pixels": metadata.get("food_core_pixels"),
                "food_inside_bowl_bbox": food_inside,
                "head_y_p10": round(head_y_p10, 6) if head_y_p10 is not None else None,
                "splitter_metadata": metadata,
            },
            bowl,
        )
    except Exception as exc:  # noqa: BLE001 - semantic failures must be auditable
        return (
            {
                "reliable": False,
                "error": f"{type(exc).__name__}:{exc}",
                "subject": None,
                "bowl_bbox": None,
                "bowl_area": 0,
                "bowl_center": None,
                "bowl_width": None,
                "bowl_height": None,
                "bowl_clipped": True,
                "food_core_bbox": None,
                "food_core_pixels": None,
                "food_inside_bowl_bbox": False,
                "head_y_p10": None,
                "splitter_metadata": None,
            },
            None,
        )


def _longest_true_run(values: list[bool]) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def _coefficient_of_variation(values: np.ndarray) -> float | None:
    if not len(values) or float(np.mean(values)) <= 0:
        return None
    return float(np.std(values) / np.mean(values))


def evaluate_full_cycle(
    records: list[dict[str, Any]],
    *,
    video: dict[str, Any],
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    policy = dict(THRESHOLDS if thresholds is None else thresholds)
    sample_count = len(records)
    issues: list[str] = []
    if sample_count != int(policy["sample_count"]):
        issues.append("not_exactly_17_full_cycle_samples")
    if int(video["frame_count"]) < int(policy["minimum_frame_count"]) or float(
        video["duration_seconds"]
    ) < float(policy["minimum_duration_seconds"]):
        issues.append("video_too_short_for_full_eating_cycle")

    unreliable_segmentation = [
        record["frame_index"]
        for record in records
        if record.get("segmentation", {}).get("reliable") is not True
    ]
    if unreliable_segmentation:
        issues.append("segmentation_unreliable_at_sampled_frames")

    semantic_failures = [
        record["frame_index"]
        for record in records
        if record.get("semantic", {}).get("reliable") is not True
    ]
    if semantic_failures:
        issues.append("bowl_or_food_not_reliably_detected_at_all_samples")

    contacts = [record.get("contact", {}).get("acceptable") is True for record in records]
    endpoint_count = min(int(policy["endpoint_window_samples"]), sample_count // 2)
    start_contact = any(contacts[:endpoint_count]) if endpoint_count else True
    end_contact = any(contacts[-endpoint_count:]) if endpoint_count else True
    if start_contact or end_contact:
        issues.append("eating_cycle_endpoints_must_be_non_contact")
    core_start = int(math.ceil((sample_count - 1) * float(policy["core_start_fraction"])))
    core_end = int(math.floor((sample_count - 1) * float(policy["core_end_fraction"])))
    core_contacts = contacts[core_start : core_end + 1]
    core_contact_count = sum(core_contacts)
    core_longest_run = _longest_true_run(core_contacts)
    center_position = sample_count // 2
    center_contact = contacts[center_position] if contacts else False
    if (
        core_contact_count < int(policy["core_contact_count_min"])
        or core_longest_run < int(policy["core_contact_longest_run_min"])
        or not center_contact
    ):
        issues.append("insufficient_continuous_core_mouth_bowl_contact")

    head_values = [record.get("semantic", {}).get("head_y_p10") for record in records]
    opening_values = [value for value in head_values[:endpoint_count] if value is not None]
    closing_values = [value for value in head_values[-endpoint_count:] if value is not None]
    core_head_values = [value for value in head_values[core_start : core_end + 1] if value is not None]
    frame_height = float(video["height"])
    head_required = frame_height * float(policy["head_lowering_fraction_min"])
    if not opening_values or not closing_values or not core_head_values:
        opening_head = closing_head = core_low_head = None
        opening_drop = closing_drop = None
        endpoint_delta = None
        issues.append("head_height_cycle_unmeasurable")
    else:
        opening_head = float(np.median(opening_values))
        closing_head = float(np.median(closing_values))
        # The lower head has the larger image y coordinate.  A high percentile is
        # robust to one noisy semantic sample without requiring every core frame.
        core_low_head = float(np.percentile(core_head_values, 80))
        opening_drop = core_low_head - opening_head
        closing_drop = core_low_head - closing_head
        endpoint_delta = abs(opening_head - closing_head)
        if (
            opening_drop < head_required
            or closing_drop < head_required
            or endpoint_delta
            > frame_height * float(policy["endpoint_head_height_delta_fraction_max"])
        ):
            issues.append("missing_head_lower_eat_raise_cycle")

    bowl_records = [
        record for record in records if record.get("semantic", {}).get("reliable") is True
    ]
    centers = np.asarray(
        [record["semantic"]["bowl_center"] for record in bowl_records], dtype=np.float64
    ) if bowl_records else np.empty((0, 2), dtype=np.float64)
    areas = np.asarray(
        [record["semantic"]["bowl_area"] for record in bowl_records], dtype=np.float64
    )
    widths = np.asarray(
        [record["semantic"]["bowl_width"] for record in bowl_records], dtype=np.float64
    )
    heights = np.asarray(
        [record["semantic"]["bowl_height"] for record in bowl_records], dtype=np.float64
    )
    if len(bowl_records) != sample_count:
        issues.append("bowl_not_detected_in_all_sampled_frames")
    bowl_complete = bool(
        len(bowl_records) == sample_count
        and all(not record["semantic"]["bowl_clipped"] for record in bowl_records)
    )
    if not bowl_complete:
        issues.append("bowl_incomplete_or_clipped")
    diagonal = math.hypot(float(video["width"]), float(video["height"]))
    if len(centers):
        center_span = float(np.linalg.norm(np.max(centers, axis=0) - np.min(centers, axis=0))) / diagonal
        center_steps = np.linalg.norm(np.diff(centers, axis=0), axis=1) / diagonal if len(centers) > 1 else np.asarray([], dtype=np.float64)
        center_step_max = float(np.max(center_steps)) if len(center_steps) else 0.0
    else:
        center_span = center_step_max = None
    area_cv = _coefficient_of_variation(areas)
    width_cv = _coefficient_of_variation(widths)
    height_cv = _coefficient_of_variation(heights)
    ious = [
        float(record["semantic"].get("bowl_iou_from_previous"))
        for record in bowl_records[1:]
        if record["semantic"].get("bowl_iou_from_previous") is not None
    ]
    iou_min = min(ious) if len(ious) == max(0, sample_count - 1) else None
    bowl_stable = bool(
        len(bowl_records) == sample_count
        and center_span is not None
        and center_span <= float(policy["bowl_center_span_fraction_max"])
        and center_step_max is not None
        and center_step_max <= float(policy["bowl_center_step_fraction_max"])
        and area_cv is not None
        and area_cv <= float(policy["bowl_area_cv_max"])
        and width_cv is not None
        and width_cv <= float(policy["bowl_width_cv_max"])
        and height_cv is not None
        and height_cv <= float(policy["bowl_height_cv_max"])
        and iou_min is not None
        and iou_min >= float(policy["bowl_consecutive_iou_min"])
    )
    if not bowl_stable:
        issues.append("bowl_temporally_unstable")

    food_inside = bool(
        len(bowl_records) == sample_count
        and all(record["semantic"]["food_inside_bowl_bbox"] for record in bowl_records)
        and all(int(record["semantic"].get("food_core_pixels") or 0) > 0 for record in bowl_records)
    )
    if not food_inside:
        issues.append("food_not_reliably_contained_inside_bowl")

    framing_geometry = [record.get("segmentation", {}).get("foreground", {}) for record in records]
    edge_frames = [
        records[index]["frame_index"]
        for index, geometry in enumerate(framing_geometry)
        if geometry.get("touches_edge") is True
    ]
    margin_frames = [
        records[index]["frame_index"]
        for index, geometry in enumerate(framing_geometry)
        if geometry.get("margins")
        and min(float(value) for value in geometry["margins"].values())
        < float(policy["foreground_margin_fraction_min"])
    ]
    closeup_frames = [
        records[index]["frame_index"]
        for index, geometry in enumerate(framing_geometry)
        if float(geometry.get("bbox_width_fraction") or 0.0)
        > float(policy["foreground_bbox_width_fraction_max"])
        or float(geometry.get("bbox_height_fraction") or 0.0)
        > float(policy["foreground_bbox_height_fraction_max"])
        or float(geometry.get("area_fraction") or 0.0)
        > float(policy["foreground_area_fraction_max"])
    ]
    if edge_frames:
        issues.append("subject_or_bowl_touches_frame_edge")
    if margin_frames:
        issues.append("foreground_safety_margin_below_2pct")
    if closeup_frames:
        issues.append("abnormal_giant_closeup")

    unique_issues = list(dict.fromkeys(issues))
    return {
        "passed": not unique_issues,
        "verdict": "accepted" if not unique_issues else "rejected_fail_closed",
        "blocking_reasons": unique_issues,
        "contact": {
            "accepted_sample_positions": [index for index, value in enumerate(contacts) if value],
            "accepted_frame_indices": [
                record["frame_index"] for record, value in zip(records, contacts, strict=True) if value
            ],
            "start_window_positions": list(range(endpoint_count)),
            "end_window_positions": list(range(max(0, sample_count - endpoint_count), sample_count)),
            "core_window_positions": list(range(core_start, core_end + 1)),
            "core_accepted_count": core_contact_count,
            "core_longest_consecutive_run": core_longest_run,
            "center_sample_contact": center_contact,
            "endpoints_non_contact": not start_contact and not end_contact,
        },
        "head_cycle": {
            "head_y_p10_by_sample": head_values,
            "opening_head_y": round(opening_head, 6) if opening_head is not None else None,
            "core_low_head_y": round(core_low_head, 6) if core_low_head is not None else None,
            "closing_head_y": round(closing_head, 6) if closing_head is not None else None,
            "opening_drop_px": round(opening_drop, 6) if opening_drop is not None else None,
            "closing_drop_px": round(closing_drop, 6) if closing_drop is not None else None,
            "endpoint_height_delta_px": round(endpoint_delta, 6) if endpoint_delta is not None else None,
            "required_drop_px": round(head_required, 6),
        },
        "bowl": {
            "detected_sample_count": len(bowl_records),
            "complete_unclipped_all_samples": bowl_complete,
            "stable": bowl_stable,
            "center_span_fraction_of_diagonal": round(center_span, 8) if center_span is not None else None,
            "center_step_max_fraction_of_diagonal": round(center_step_max, 8) if center_step_max is not None else None,
            "area_cv": round(area_cv, 8) if area_cv is not None else None,
            "width_cv": round(width_cv, 8) if width_cv is not None else None,
            "height_cv": round(height_cv, 8) if height_cv is not None else None,
            "consecutive_mask_iou_min": round(iou_min, 8) if iou_min is not None else None,
        },
        "food": {"inside_bowl_all_samples": food_inside},
        "framing": {
            "edge_contact_frame_indices": edge_frames,
            "low_margin_frame_indices": margin_frames,
            "giant_closeup_frame_indices": closeup_frames,
        },
        "fail_closed": {
            "unreliable_segmentation_frame_indices": unreliable_segmentation,
            "semantic_failure_frame_indices": semantic_failures,
        },
    }


def audit_final_eating_video(
    video_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    rembg_model: str = "u2net",
    thresholds: dict[str, Any] | None = None,
    background_remover: Callable[..., Any] = remove_background_rembg,
    contact_evaluator: Callable[[Path], dict[str, Any]] = _default_contact_evaluator,
    semantic_splitter: Callable[
        [Image.Image], tuple[Image.Image, Image.Image, dict[str, Any]]
    ] = _default_semantic_splitter,
    video_probe: Callable[[str | Path], dict[str, Any]] = probe_video,
    frame_decoder: Callable[[str | Path, int], np.ndarray] = decode_frame,
) -> dict[str, Any]:
    resolved = Path(video_path).resolve()
    if not resolved.is_file():
        raise ValueError(f"video_not_found:{resolved}")
    if sample_count != DEFAULT_SAMPLE_COUNT:
        raise ValueError("final_eating_audit_requires_exactly_17_samples")
    policy = dict(THRESHOLDS)
    if thresholds:
        policy.update(thresholds)
    metadata = video_probe(resolved)
    indices = sample_indices(int(metadata["frame_count"]), sample_count)
    artifact_root = Path(output_dir).resolve() if output_dir is not None else None
    if artifact_root is not None:
        artifact_root.mkdir(parents=True, exist_ok=True)
        (artifact_root / "cutouts").mkdir(exist_ok=True)

    records: list[dict[str, Any]] = []
    previous_bowl_mask: np.ndarray | None = None
    with tempfile.TemporaryDirectory(prefix="final-eating-video-audit-") as temporary:
        scratch = Path(temporary)
        for sample_position, frame_index in enumerate(indices):
            cutout_path = (
                artifact_root / "cutouts" / f"frame_{frame_index:05d}.png"
                if artifact_root is not None
                else scratch / f"frame_{frame_index:05d}.png"
            )
            record: dict[str, Any] = {
                "sample_position": sample_position,
                "frame_index": frame_index,
                "time_seconds": round(frame_index / float(metadata["fps"]), 6),
                "segmentation": {"reliable": False, "issues": ["not_evaluated"]},
                "contact": {
                    "acceptable": False,
                    "issues": ["not_evaluated"],
                    "method": None,
                },
                "semantic": {"reliable": False, "error": "not_evaluated"},
                "error": None,
            }
            try:
                rgb = np.asarray(frame_decoder(resolved, frame_index), dtype=np.uint8)
                if rgb.ndim != 3 or rgb.shape[2] != 3:
                    raise RuntimeError("decoded_frame_is_not_rgb")
                primary_rgba, remover_provenance = _call_background_remover(
                    background_remover,
                    rgb,
                    frame_index=frame_index,
                    model=rembg_model,
                )
                rgba, segmentation = recover_uniform_studio_foreground(
                    rgb,
                    primary_rgba,
                    thresholds=policy,
                )
                segmentation["primary_remover"] = remover_provenance
                record["segmentation"] = segmentation
                Image.fromarray(rgba, "RGBA").save(cutout_path, optimize=True)
                reloaded = np.asarray(Image.open(cutout_path).convert("RGBA"), dtype=np.uint8)
                if not np.array_equal(reloaded[..., :3], rgb):
                    raise RuntimeError("saved_cutout_rgb_not_bit_exact")
                contact = contact_evaluator(cutout_path)
                if not isinstance(contact, dict) or contact.get("acceptable") not in {
                    True,
                    False,
                }:
                    raise RuntimeError("contact_evaluator_returned_unsupported_result")
                record["contact"] = contact
                semantic, bowl_mask = semantic_frame_metrics(rgba, splitter=semantic_splitter)
                if bowl_mask is not None and previous_bowl_mask is not None:
                    union = int(np.count_nonzero(bowl_mask | previous_bowl_mask))
                    semantic["bowl_iou_from_previous"] = float(
                        np.count_nonzero(bowl_mask & previous_bowl_mask)
                    ) / float(max(1, union))
                else:
                    semantic["bowl_iou_from_previous"] = None
                if bowl_mask is not None:
                    previous_bowl_mask = bowl_mask
                record["semantic"] = semantic
            except Exception as exc:  # noqa: BLE001 - every frame fails closed
                record["error"] = f"{type(exc).__name__}:{exc}"
                if record["segmentation"].get("issues") == ["not_evaluated"]:
                    record["segmentation"] = {
                        "reliable": False,
                        "issues": ["segmentation_or_frame_processing_error"],
                    }
            records.append(record)

    cycle = evaluate_full_cycle(records, video=metadata, thresholds=policy)
    result = {
        "passed": cycle["passed"],
        "verdict": cycle["verdict"],
        "action": "eating",
        "video": {
            "path": str(resolved),
            "sha256": sha256_file(resolved),
            **metadata,
        },
        "sampling": {
            "policy": "17_evenly_spaced_full_cycle_samples_including_endpoints",
            "requested_count": sample_count,
            "actual_count": len(indices),
            "frame_indices": indices,
        },
        "cycle": cycle,
        "frames": records,
        "policy": {
            "name": POLICY,
            "version": VERSION,
            "fail_closed": True,
            "existing_production_contact_evaluator_required": True,
            "decoded_rgb_must_remain_bit_exact": True,
            "contact_pixels_may_not_be_painted": True,
            "rembg_model": rembg_model,
            "production_integrated": False,
            "thresholds": policy,
            "threshold_rationale": THRESHOLD_RATIONALE,
        },
    }
    if artifact_root is not None:
        report_path = artifact_root / "final_eating_video_audit.json"
        report_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        result["report_path"] = str(report_path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rembg-model", default="u2net")
    args = parser.parse_args()
    try:
        result = audit_final_eating_video(
            args.video,
            output_dir=args.output_dir,
            rembg_model=args.rembg_model,
        )
    except Exception as exc:  # noqa: BLE001 - CLI always emits a fail-closed JSON result
        result = {
            "passed": False,
            "verdict": "audit_error_fail_closed",
            "action": "eating",
            "video": {"path": str(args.video)},
            "error": f"{type(exc).__name__}:{exc}",
            "policy": {"name": POLICY, "version": VERSION},
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "final_eating_video_audit.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
