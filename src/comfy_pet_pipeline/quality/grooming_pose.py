#!/usr/bin/env python3
"""Fail-closed CV gate for the Main_case raised-paw grooming keyframe.

The gate is deliberately independent from ``local_video_api.py``.  It expects
an already-transparent candidate PNG and the low-frequency grooming geometry
guide.  Geometry-guide metadata is optional; when present, its source-derived
muzzle and raised-foreleg regions are registered into the candidate's alpha
bounding box.  Otherwise the frozen Main_case v3 regions are used.

This module checks pose geometry plus a localized raised-paw/planted-paw
appearance sanity check.  Subject identity and fur-length checks remain
separate concerns.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


POLICY = "main_case_grooming_raised_paw_cv_v4"
GATE_VERSION = "4.0"


@dataclass(frozen=True)
class GroomingPoseThresholds:
    """Thresholds measured in a normalized 400 x 512 subject crop."""

    alpha_threshold: int = 64
    normalized_width: int = 400
    normalized_height: int = 512
    minimum_foreground_pixels: int = 4096
    minimum_roi_foreground_pixels: int = 512
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8
    canny_low: int = 30
    canny_high: int = 60
    edge_density_min: float = 0.11
    structural_blur_sigma: float = 4.0
    structural_canny_low: int = 20
    structural_canny_high: int = 40
    structural_edge_density_min: float = 0.015
    chamfer_radius_px: float = 6.0
    chamfer_coverage_min: float = 0.62
    chamfer_mean_distance_max: float = 7.20
    continuous_edge_span_min: float = 0.24
    paw_muzzle_gap_max: float = 0.040
    paw_terminal_band_height_fraction: float = 0.11
    lab_median_delta_max: float = 35.0
    lab_histogram_distance_max: float = 0.42
    texture_p90_ratio_min: float = 0.25
    texture_p90_ratio_max: float = 4.0
    global_registration_size: int = 256
    global_registration_margin: int = 12
    global_bbox_log_aspect_delta_max: float = 0.55
    global_silhouette_iou_min: float = 0.18
    global_silhouette_iou_aspect_delta_min: float = 0.25


DEFAULT_FORELEG_ROI = (0.10, 0.43, 0.48, 0.77)
DEFAULT_MUZZLE_ROI = (0.10, 0.43, 0.30, 0.47)
DEFAULT_PLANTED_PAW_ROI = (0.34, 0.66, 0.78, 0.96)


class GroomingPoseInputError(ValueError):
    """Raised when an input cannot provide fail-closed pose evidence."""


def _largest_component(mask: np.ndarray) -> tuple[np.ndarray, list[int]]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )
    if count <= 1:
        raise GroomingPoseInputError("no_foreground_subject")
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    left = int(stats[label, cv2.CC_STAT_LEFT])
    top = int(stats[label, cv2.CC_STAT_TOP])
    width = int(stats[label, cv2.CC_STAT_WIDTH])
    height = int(stats[label, cv2.CC_STAT_HEIGHT])
    return labels == label, [left, top, left + width, top + height]


def _load_candidate(
    path: Path,
    thresholds: GroomingPoseThresholds,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise GroomingPoseInputError("candidate_image_unreadable")
    if image.ndim != 3 or image.shape[2] < 4:
        raise GroomingPoseInputError("candidate_requires_rgba_png")
    if path.suffix.lower() != ".png":
        raise GroomingPoseInputError("candidate_requires_rgba_png")
    alpha = image[:, :, 3]
    if not np.any(alpha < 250):
        raise GroomingPoseInputError("candidate_alpha_has_no_transparency")
    component, bbox = _largest_component(alpha >= thresholds.alpha_threshold)
    if int(np.count_nonzero(component)) < thresholds.minimum_foreground_pixels:
        raise GroomingPoseInputError("candidate_foreground_too_small")
    return image[:, :, :3], component, bbox


def _guide_foreground(image: np.ndarray) -> tuple[np.ndarray, str]:
    if image.ndim != 3 or image.shape[2] < 3:
        raise GroomingPoseInputError("geometry_guide_requires_rgb_or_rgba")
    if image.shape[2] >= 4 and np.any(image[:, :, 3] < 250):
        return image[:, :, 3] >= 64, "alpha"

    bgr = image[:, :, :3]
    blue = bgr[:, :, 0].astype(np.int16)
    green = bgr[:, :, 1].astype(np.int16)
    red = bgr[:, :, 2].astype(np.int16)
    green_background = (green > 80) & ((green - np.maximum(red, blue)) > 35)
    if float(np.mean(green_background)) >= 0.05:
        return ~green_background, "green_screen"

    corners = np.asarray(
        [bgr[0, 0], bgr[0, -1], bgr[-1, 0], bgr[-1, -1]],
        dtype=np.float32,
    )
    background = np.median(corners, axis=0)
    distance = np.linalg.norm(bgr.astype(np.float32) - background, axis=2)
    foreground = distance >= 24.0
    if float(np.mean(foreground)) < 0.02:
        raise GroomingPoseInputError("geometry_guide_foreground_unavailable")
    return foreground, "corner_color"


def _load_guide(path: Path) -> tuple[np.ndarray, np.ndarray, list[int], str]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise GroomingPoseInputError("geometry_guide_unreadable")
    foreground, method = _guide_foreground(image)
    component, bbox = _largest_component(foreground)
    return image[:, :, :3], component, bbox, method


def _normalized_crop(
    image: np.ndarray,
    component: np.ndarray,
    bbox: Sequence[int],
    thresholds: GroomingPoseThresholds,
) -> tuple[np.ndarray, np.ndarray]:
    left, top, right, bottom = (int(value) for value in bbox)
    if right - left < 16 or bottom - top < 16:
        raise GroomingPoseInputError("foreground_bbox_too_small")
    size = (thresholds.normalized_width, thresholds.normalized_height)
    normalized_image = cv2.resize(
        image[top:bottom, left:right],
        size,
        interpolation=cv2.INTER_AREA,
    )
    normalized_mask = cv2.resize(
        component[top:bottom, left:right].astype(np.uint8),
        size,
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    return normalized_image, normalized_mask


def _letterbox_component(
    component: np.ndarray,
    bbox: Sequence[int],
    *,
    size: int,
    margin: int,
) -> np.ndarray:
    """Center one silhouette without changing its source aspect ratio."""

    left, top, right, bottom = (int(value) for value in bbox)
    cropped = component[top:bottom, left:right].astype(np.uint8)
    if cropped.size == 0 or not np.any(cropped):
        raise GroomingPoseInputError("global_pose_foreground_unavailable")
    available = max(8, int(size) - 2 * max(0, int(margin)))
    height, width = cropped.shape
    scale = min(available / float(width), available / float(height))
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(
        cropped,
        (resized_width, resized_height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    canvas = np.zeros((int(size), int(size)), dtype=bool)
    x0 = (int(size) - resized_width) // 2
    y0 = (int(size) - resized_height) // 2
    canvas[y0 : y0 + resized_height, x0 : x0 + resized_width] = resized
    return canvas


def _global_pose_alignment(
    candidate_mask: np.ndarray,
    candidate_bbox: Sequence[int],
    guide_mask: np.ndarray,
    guide_bbox: Sequence[int],
    thresholds: GroomingPoseThresholds,
) -> dict[str, Any]:
    """Compare the full silhouettes before ROI normalization can stretch them."""

    candidate_width = max(1, int(candidate_bbox[2]) - int(candidate_bbox[0]))
    candidate_height = max(1, int(candidate_bbox[3]) - int(candidate_bbox[1]))
    guide_width = max(1, int(guide_bbox[2]) - int(guide_bbox[0]))
    guide_height = max(1, int(guide_bbox[3]) - int(guide_bbox[1]))
    candidate_aspect = candidate_width / float(candidate_height)
    guide_aspect = guide_width / float(guide_height)
    log_aspect_delta = abs(np.log(max(candidate_aspect, 1e-6) / max(guide_aspect, 1e-6)))
    candidate_registered = _letterbox_component(
        candidate_mask,
        candidate_bbox,
        size=thresholds.global_registration_size,
        margin=thresholds.global_registration_margin,
    )
    guide_registered = _letterbox_component(
        guide_mask,
        guide_bbox,
        size=thresholds.global_registration_size,
        margin=thresholds.global_registration_margin,
    )
    intersection = int(np.count_nonzero(candidate_registered & guide_registered))
    union = int(np.count_nonzero(candidate_registered | guide_registered))
    silhouette_iou = intersection / float(max(1, union))
    return {
        "policy": "aspect_preserving_full_silhouette_registration_v1",
        "candidate_bbox_aspect_ratio": round(candidate_aspect, 6),
        "guide_bbox_aspect_ratio": round(guide_aspect, 6),
        "bbox_log_aspect_delta": round(float(log_aspect_delta), 6),
        "bbox_log_aspect_delta_maximum": (
            thresholds.global_bbox_log_aspect_delta_max
        ),
        "silhouette_iou": round(float(silhouette_iou), 6),
        "silhouette_iou_minimum": thresholds.global_silhouette_iou_min,
        "silhouette_iou_aspect_delta_minimum": (
            thresholds.global_silhouette_iou_aspect_delta_min
        ),
        "registration_size": [
            thresholds.global_registration_size,
            thresholds.global_registration_size,
        ],
        "registration_margin": thresholds.global_registration_margin,
    }


def _load_metadata(
    metadata: str | Path | Mapping[str, Any] | None,
) -> tuple[dict[str, Any] | None, str | None]:
    if metadata is None:
        return None, None
    if isinstance(metadata, Mapping):
        return dict(metadata), None
    path = Path(metadata)
    if not path.is_file():
        raise GroomingPoseInputError("geometry_metadata_unreadable")
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GroomingPoseInputError("geometry_metadata_unreadable") from exc
    if not isinstance(loaded, dict):
        raise GroomingPoseInputError("geometry_metadata_requires_object")
    return loaded, str(path)


def _valid_box(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        left, top, right, bottom = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not left < right or not top < bottom:
        return None
    return left, top, right, bottom


def _clamp_roi(
    roi: Sequence[float],
    *,
    minimum_width: float = 0.08,
    minimum_height: float = 0.08,
) -> tuple[float, float, float, float] | None:
    left, right, top, bottom = (float(value) for value in roi)
    left = max(0.0, min(left, 0.98))
    right = max(0.02, min(right, 1.0))
    top = max(0.0, min(top, 0.98))
    bottom = max(0.02, min(bottom, 1.0))
    if right - left < minimum_width or bottom - top < minimum_height:
        return None
    return left, right, top, bottom


def _derive_rois(
    metadata: dict[str, Any] | None,
    guide_bbox: Sequence[int],
) -> tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    str,
    list[str],
]:
    warnings: list[str] = []
    if metadata is None:
        return (
            DEFAULT_FORELEG_ROI,
            DEFAULT_MUZZLE_ROI,
            DEFAULT_PLANTED_PAW_ROI,
            "guide_default_v1",
            warnings,
        )
    relation = metadata.get("raised_paw_muzzle_relation")
    if not isinstance(relation, dict):
        warnings.append("metadata_semantic_regions_unavailable")
        return (
            DEFAULT_FORELEG_ROI,
            DEFAULT_MUZZLE_ROI,
            DEFAULT_PLANTED_PAW_ROI,
            "guide_default_v1",
            warnings,
        )
    foreleg = _valid_box(relation.get("raised_foreleg_region_bbox"))
    muzzle = _valid_box(relation.get("muzzle_region_bbox"))
    if foreleg is None or muzzle is None:
        warnings.append("metadata_semantic_regions_invalid")
        return (
            DEFAULT_FORELEG_ROI,
            DEFAULT_MUZZLE_ROI,
            DEFAULT_PLANTED_PAW_ROI,
            "guide_default_v1",
            warnings,
        )
    guide_left, guide_top, guide_right, guide_bottom = (
        float(value) for value in guide_bbox
    )
    guide_width = max(1.0, guide_right - guide_left)
    guide_height = max(1.0, guide_bottom - guide_top)

    foreleg_roi = _clamp_roi(
        (
            max(
                DEFAULT_FORELEG_ROI[0],
                (min(foreleg[0], muzzle[0]) - guide_left) / guide_width,
            ),
            max(
                DEFAULT_FORELEG_ROI[1],
                (max(foreleg[2], muzzle[2]) - guide_left) / guide_width + 0.04,
            ),
            (foreleg[1] - guide_top) / guide_height + 0.05,
            (foreleg[3] - guide_top) / guide_height + 0.04,
        )
    )
    muzzle_roi = _clamp_roi(
        (
            max(
                DEFAULT_MUZZLE_ROI[0],
                (min(foreleg[0], muzzle[0]) - guide_left) / guide_width,
            ),
            max(
                DEFAULT_MUZZLE_ROI[1],
                (max(foreleg[2], muzzle[2]) - guide_left) / guide_width + 0.04,
            ),
            (muzzle[1] - guide_top) / guide_height - 0.03,
            (muzzle[3] - guide_top) / guide_height + 0.04,
        )
    )
    if foreleg_roi is None or muzzle_roi is None:
        warnings.append("metadata_semantic_regions_outside_guide_subject")
        return (
            DEFAULT_FORELEG_ROI,
            DEFAULT_MUZZLE_ROI,
            DEFAULT_PLANTED_PAW_ROI,
            "guide_default_v1",
            warnings,
        )
    return (
        foreleg_roi,
        muzzle_roi,
        DEFAULT_PLANTED_PAW_ROI,
        "metadata_source_derived_v1",
        warnings,
    )


def _roi_pixels(
    roi: Sequence[float],
    thresholds: GroomingPoseThresholds,
) -> tuple[int, int, int, int]:
    left, right, top, bottom = roi
    x0 = max(
        0,
        min(
            thresholds.normalized_width - 1,
            int(round(left * thresholds.normalized_width)),
        ),
    )
    x1 = max(
        x0 + 1,
        min(
            thresholds.normalized_width, int(round(right * thresholds.normalized_width))
        ),
    )
    y0 = max(
        0,
        min(
            thresholds.normalized_height - 1,
            int(round(top * thresholds.normalized_height)),
        ),
    )
    y1 = max(
        y0 + 1,
        min(
            thresholds.normalized_height,
            int(round(bottom * thresholds.normalized_height)),
        ),
    )
    return x0, y0, x1, y1


def _edge_map(
    image: np.ndarray,
    mask: np.ndarray,
    thresholds: GroomingPoseThresholds,
) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(
        clipLimit=thresholds.clahe_clip_limit,
        tileGridSize=(thresholds.clahe_grid_size, thresholds.clahe_grid_size),
    )
    enhanced = clahe.apply(gray)
    edges = cv2.Canny(
        enhanced,
        thresholds.canny_low,
        thresholds.canny_high,
    ).astype(bool)
    return edges & mask, gray


def _region(array: np.ndarray, box: Sequence[int]) -> np.ndarray:
    x0, y0, x1, y1 = (int(value) for value in box)
    return array[y0:y1, x0:x1]


def _structural_foreleg_evidence(
    gray: np.ndarray,
    mask: np.ndarray,
    foreleg_box: Sequence[int],
    thresholds: GroomingPoseThresholds,
) -> dict[str, Any]:
    """Measure low-frequency anatomy without accepting fine fur as a limb.

    The original edge and chamfer checks intentionally retain fine detail so
    they can compare a furry paw with the private Main_case guide.  A neutral
    two-paws-down cat can nevertheless fill the registered foreleg ROI with
    dense chest fur and accidentally satisfy those checks.  Blur four pixels
    before Canny here so isolated strands disappear while an organic
    shoulder-to-elbow-to-wrist boundary remains measurable.
    """

    if gray.shape != mask.shape:
        return {
            "available": False,
            "reason": "structural_foreleg_input_shape_mismatch",
            "edge_count": 0,
            "edge_density": 0.0,
        }
    roi_mask = _region(mask, foreleg_box)
    foreground_pixels = int(np.count_nonzero(roi_mask))
    if foreground_pixels < thresholds.minimum_roi_foreground_pixels:
        return {
            "available": False,
            "reason": "structural_foreleg_roi_foreground_too_small",
            "foreground_pixels": foreground_pixels,
            "edge_count": 0,
            "edge_density": 0.0,
        }
    blurred = cv2.GaussianBlur(
        gray,
        (0, 0),
        thresholds.structural_blur_sigma,
    )
    structural_edges = cv2.Canny(
        blurred,
        thresholds.structural_canny_low,
        thresholds.structural_canny_high,
    ).astype(bool)
    structural_edges &= mask
    edge_count = int(np.count_nonzero(_region(structural_edges, foreleg_box)))
    return {
        "available": True,
        "policy": "gaussian_lowpass_organic_foreleg_structure_v1",
        "blur_sigma": thresholds.structural_blur_sigma,
        "canny": [
            thresholds.structural_canny_low,
            thresholds.structural_canny_high,
        ],
        "foreground_pixels": foreground_pixels,
        "edge_count": edge_count,
        "edge_density": round(edge_count / float(foreground_pixels), 6),
    }


def _continuous_edge_component(
    edge_roi: np.ndarray,
    roi_box: Sequence[int],
    thresholds: GroomingPoseThresholds,
) -> dict[str, Any] | None:
    closed = cv2.morphologyEx(
        edge_roi.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((7, 3), dtype=np.uint8),
    )
    joined = cv2.dilate(
        closed,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    )
    count, _, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    records: list[tuple[int, int, int]] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= 20:
            records.append((label, int(stats[label, cv2.CC_STAT_HEIGHT]), area))
    if not records:
        return None
    label, height, area = max(records, key=lambda item: (item[1], item[2]))
    local_left = int(stats[label, cv2.CC_STAT_LEFT])
    local_top = int(stats[label, cv2.CC_STAT_TOP])
    width = int(stats[label, cv2.CC_STAT_WIDTH])
    x0, y0, _, _ = (int(value) for value in roi_box)
    return {
        "span_fraction": round(height / float(thresholds.normalized_height), 6),
        "area": area,
        "bbox": [
            x0 + local_left,
            y0 + local_top,
            x0 + local_left + width,
            y0 + local_top + height,
        ],
        "top_y": y0 + local_top,
    }


def _histogram_distance(first: np.ndarray, second: np.ndarray) -> dict[str, Any]:
    distances: list[float] = []
    for channel in range(3):
        first_hist, _ = np.histogram(first[:, channel], bins=32, range=(0, 256))
        second_hist, _ = np.histogram(second[:, channel], bins=32, range=(0, 256))
        first_prob = first_hist.astype(np.float64) / max(1.0, float(first_hist.sum()))
        second_prob = second_hist.astype(np.float64) / max(
            1.0, float(second_hist.sum())
        )
        coefficient = float(np.sum(np.sqrt(first_prob * second_prob)))
        distances.append(float(np.sqrt(max(0.0, 1.0 - coefficient))))
    return {
        "channels": [round(value, 6) for value in distances],
        "mean": round(float(np.mean(distances)), 6),
    }


def _terminal_paw_boxes(
    foreleg_box: Sequence[int],
    planted_box: Sequence[int],
    continuous: Mapping[str, Any] | None,
    thresholds: GroomingPoseThresholds,
) -> dict[str, Any]:
    """Localize paw tips without treating a patterned foreleg as the paw.

    The raised paw is the top terminal band of the continuous foreleg, nearest
    the muzzle.  The planted paw is the bottom terminal band of its ROI.  This
    preserves a strict paw-to-paw appearance comparison while allowing a cat's
    foreleg shaft to carry a different marking from its paw tip.
    """

    foreleg_x0, foreleg_y0, foreleg_x1, foreleg_y1 = (
        int(value) for value in foreleg_box
    )
    planted_x0, planted_y0, planted_x1, planted_y1 = (
        int(value) for value in planted_box
    )
    band_height = max(
        1,
        int(
            round(
                thresholds.paw_terminal_band_height_fraction
                * thresholds.normalized_height
            )
        ),
    )
    raised_anchor = foreleg_y0
    anchor_source = "foreleg_roi_top_fallback"
    if isinstance(continuous, Mapping):
        try:
            raised_anchor = int(round(float(continuous["top_y"])))
            anchor_source = "continuous_edge_top"
        except (KeyError, TypeError, ValueError):
            pass
    alignment_padding = max(1, int(round(band_height * 0.08)))
    raised_top = max(
        foreleg_y0,
        min(foreleg_y1 - 1, raised_anchor - alignment_padding),
    )
    raised_bottom = min(foreleg_y1, raised_top + band_height)
    planted_top = max(planted_y0, planted_y1 - band_height)
    return {
        "policy": "terminal_paw_bands_v2",
        "band_height_pixels": band_height,
        "raised_anchor_source": anchor_source,
        "raised_anchor_y": raised_anchor,
        "raised": [foreleg_x0, raised_top, foreleg_x1, raised_bottom],
        "planted": [planted_x0, planted_top, planted_x1, planted_y1],
    }


def _appearance_consistency(
    image: np.ndarray,
    mask: np.ndarray,
    gray: np.ndarray,
    raised_paw_box: Sequence[int],
    planted_paw_box: Sequence[int],
    thresholds: GroomingPoseThresholds,
) -> dict[str, Any]:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.hypot(gradient_x, gradient_y)

    samples: dict[str, dict[str, Any]] = {}
    lab_values: dict[str, np.ndarray] = {}
    for label, box in (
        ("raised", raised_paw_box),
        ("planted", planted_paw_box),
    ):
        region_mask = _region(mask, box)
        count = int(np.count_nonzero(region_mask))
        if count < thresholds.minimum_roi_foreground_pixels:
            return {
                "available": False,
                "reason": f"{label}_paw_roi_foreground_too_small",
                "raised_sample_count": (
                    count if label == "raised" else samples["raised"]["sample_count"]
                ),
                "planted_sample_count": count if label == "planted" else 0,
            }
        values = _region(lab, box)[region_mask]
        gradients = _region(gradient, box)[region_mask]
        lab_values[label] = values
        samples[label] = {
            "sample_count": count,
            "lab_median": [
                round(float(value), 6) for value in np.median(values, axis=0)
            ],
            "lab_mean": [round(float(value), 6) for value in np.mean(values, axis=0)],
            "gradient_median": round(float(np.median(gradients)), 6),
            "gradient_p90": round(float(np.percentile(gradients, 90)), 6),
        }

    raised_median = np.median(lab_values["raised"], axis=0).astype(np.float64)
    planted_median = np.median(lab_values["planted"], axis=0).astype(np.float64)
    lab_delta = float(np.linalg.norm(raised_median - planted_median))
    histogram = _histogram_distance(
        lab_values["raised"],
        lab_values["planted"],
    )
    raised_p90 = float(samples["raised"]["gradient_p90"])
    planted_p90 = float(samples["planted"]["gradient_p90"])
    texture_ratio = (raised_p90 + 1.0) / (planted_p90 + 1.0)
    return {
        "available": True,
        "raised": samples["raised"],
        "planted": samples["planted"],
        "lab_median_delta": round(lab_delta, 6),
        "lab_histogram_bhattacharyya": histogram,
        "texture_gradient_p90_ratio": round(texture_ratio, 6),
    }


def evaluate_grooming_pose(
    candidate_png: str | Path,
    geometry_guide: str | Path,
    *,
    geometry_metadata: str | Path | Mapping[str, Any] | None = None,
    thresholds: GroomingPoseThresholds | None = None,
) -> dict[str, Any]:
    """Return JSON-safe raised-paw grooming metrics and a fail-closed verdict."""

    active = thresholds or GroomingPoseThresholds()
    candidate_path = Path(candidate_png)
    guide_path = Path(geometry_guide)
    base: dict[str, Any] = {
        "policy": POLICY,
        "version": GATE_VERSION,
        "candidate": str(candidate_path),
        "geometry_guide": str(guide_path),
        "thresholds": asdict(active),
    }
    try:
        candidate_image, candidate_mask, candidate_bbox = _load_candidate(
            candidate_path,
            active,
        )
        guide_image, guide_mask, guide_bbox, guide_mask_method = _load_guide(guide_path)
        metadata, metadata_path = _load_metadata(geometry_metadata)
        foreleg_roi, muzzle_roi, planted_roi, roi_source, warnings = _derive_rois(
            metadata,
            guide_bbox,
        )
        candidate_normalized, candidate_mask_normalized = _normalized_crop(
            candidate_image,
            candidate_mask,
            candidate_bbox,
            active,
        )
        guide_normalized, guide_mask_normalized = _normalized_crop(
            guide_image,
            guide_mask,
            guide_bbox,
            active,
        )
        global_alignment = _global_pose_alignment(
            candidate_mask,
            candidate_bbox,
            guide_mask,
            guide_bbox,
            active,
        )
    except GroomingPoseInputError as exc:
        return {
            **base,
            "available": False,
            "acceptable": False,
            "issues": [str(exc)],
            "warnings": [],
            "penalty": 1000.0,
        }

    foreleg_box = _roi_pixels(foreleg_roi, active)
    muzzle_box = _roi_pixels(muzzle_roi, active)
    planted_box = _roi_pixels(planted_roi, active)
    candidate_edges, candidate_gray = _edge_map(
        candidate_normalized,
        candidate_mask_normalized,
        active,
    )
    guide_edges, _ = _edge_map(
        guide_normalized,
        guide_mask_normalized,
        active,
    )
    candidate_foreleg_edges = _region(candidate_edges, foreleg_box)
    candidate_foreleg_mask = _region(candidate_mask_normalized, foreleg_box)
    guide_foreleg_edges = _region(guide_edges, foreleg_box)

    candidate_roi_foreground = int(np.count_nonzero(candidate_foreleg_mask))
    candidate_roi_edge_count = int(np.count_nonzero(candidate_foreleg_edges))
    guide_roi_edge_count = int(np.count_nonzero(guide_foreleg_edges))
    edge_density = candidate_roi_edge_count / float(max(1, candidate_roi_foreground))
    structural = _structural_foreleg_evidence(
        candidate_gray,
        candidate_mask_normalized,
        foreleg_box,
        active,
    )

    chamfer: dict[str, Any]
    if candidate_roi_edge_count and guide_roi_edge_count:
        distance = cv2.distanceTransform(
            (~candidate_foreleg_edges).astype(np.uint8),
            cv2.DIST_L2,
            3,
        )
        guide_distances = distance[guide_foreleg_edges]
        chamfer = {
            "available": True,
            "guide_edge_count": guide_roi_edge_count,
            "coverage_within_radius": round(
                float(np.mean(guide_distances <= active.chamfer_radius_px)),
                6,
            ),
            "mean_distance_px": round(float(np.mean(guide_distances)), 6),
            "p90_distance_px": round(float(np.percentile(guide_distances, 90)), 6),
        }
    else:
        chamfer = {
            "available": False,
            "guide_edge_count": guide_roi_edge_count,
            "coverage_within_radius": 0.0,
            "mean_distance_px": None,
            "p90_distance_px": None,
        }

    continuous = _continuous_edge_component(
        candidate_foreleg_edges,
        foreleg_box,
        active,
    )
    muzzle_edges = _region(candidate_edges, muzzle_box)
    muzzle_edge_ys = np.where(muzzle_edges)[0]
    contact: dict[str, Any]
    if continuous is not None and muzzle_edge_ys.size:
        muzzle_bottom_y = int(muzzle_box[1]) + float(np.percentile(muzzle_edge_ys, 90))
        paw_top_y = float(continuous["top_y"])
        gap_pixels = max(0.0, paw_top_y - muzzle_bottom_y)
        contact = {
            "available": True,
            "paw_top_y": round(paw_top_y, 6),
            "muzzle_bottom_y": round(muzzle_bottom_y, 6),
            "gap_pixels": round(gap_pixels, 6),
            "gap_fraction_of_bbox_height": round(
                gap_pixels / float(active.normalized_height),
                6,
            ),
        }
    else:
        contact = {
            "available": False,
            "paw_top_y": continuous["top_y"] if continuous else None,
            "muzzle_bottom_y": None,
            "gap_pixels": None,
            "gap_fraction_of_bbox_height": None,
        }

    terminal_paw_sampling = _terminal_paw_boxes(
        foreleg_box,
        planted_box,
        continuous,
        active,
    )
    appearance = _appearance_consistency(
        candidate_normalized,
        candidate_mask_normalized,
        candidate_gray,
        terminal_paw_sampling["raised"],
        terminal_paw_sampling["planted"],
        active,
    )
    appearance["sampling"] = terminal_paw_sampling

    issues: list[str] = []
    global_aspect_delta = float(global_alignment["bbox_log_aspect_delta"])
    global_silhouette_iou = float(global_alignment["silhouette_iou"])
    if global_aspect_delta > active.global_bbox_log_aspect_delta_max:
        issues.append("gross_full_body_pose_aspect_mismatch")
    if (
        global_aspect_delta
        > active.global_silhouette_iou_aspect_delta_min
        and global_silhouette_iou < active.global_silhouette_iou_min
    ):
        issues.append("gross_full_body_pose_silhouette_mismatch")
    if candidate_roi_foreground < active.minimum_roi_foreground_pixels:
        issues.append("raised_foreleg_roi_foreground_too_small")
    if edge_density < active.edge_density_min:
        issues.append("raised_foreleg_edge_density_too_low")
    if not structural["available"]:
        issues.append("raised_foreleg_structural_evidence_unavailable")
    elif (
        float(structural["edge_density"])
        < active.structural_edge_density_min
    ):
        issues.append("raised_foreleg_structural_edge_density_too_low")
    if not chamfer["available"]:
        issues.append("guide_chamfer_unavailable")
    else:
        if float(chamfer["coverage_within_radius"]) < active.chamfer_coverage_min:
            issues.append("guide_chamfer_coverage_too_low")
        if float(chamfer["mean_distance_px"]) > active.chamfer_mean_distance_max:
            issues.append("guide_chamfer_distance_too_large")
    if continuous is None:
        issues.append("continuous_raised_foreleg_edge_unavailable")
    elif float(continuous["span_fraction"]) < active.continuous_edge_span_min:
        issues.append("continuous_raised_foreleg_edge_too_short")
    if not contact["available"]:
        issues.append("paw_muzzle_gap_unavailable")
    elif float(contact["gap_fraction_of_bbox_height"]) > active.paw_muzzle_gap_max:
        issues.append("raised_paw_too_far_from_muzzle")
    if not appearance["available"]:
        issues.append("raised_planted_paw_appearance_unavailable")
    else:
        if float(appearance["lab_median_delta"]) > active.lab_median_delta_max:
            issues.append("raised_paw_color_inconsistent_with_planted_paw")
        histogram_distance = float(appearance["lab_histogram_bhattacharyya"]["mean"])
        if histogram_distance > active.lab_histogram_distance_max:
            issues.append("raised_paw_material_inconsistent_with_planted_paw")
        texture_ratio = float(appearance["texture_gradient_p90_ratio"])
        if not (
            active.texture_p90_ratio_min
            <= texture_ratio
            <= active.texture_p90_ratio_max
        ):
            issues.append("raised_paw_texture_inconsistent_with_planted_paw")

    shortfalls = [
        max(
            0.0,
            global_aspect_delta
            / max(active.global_bbox_log_aspect_delta_max, 1e-6)
            - 1.0,
        ),
        (
            max(
                0.0,
                active.global_silhouette_iou_min - global_silhouette_iou,
            )
            / max(active.global_silhouette_iou_min, 1e-6)
            if global_aspect_delta
            > active.global_silhouette_iou_aspect_delta_min
            else 0.0
        ),
        max(0.0, active.edge_density_min - edge_density)
        / max(active.edge_density_min, 1e-6),
        max(
            0.0,
            active.structural_edge_density_min
            - float(structural.get("edge_density") or 0.0),
        )
        / max(active.structural_edge_density_min, 1e-6),
        max(
            0.0,
            active.chamfer_coverage_min
            - float(chamfer.get("coverage_within_radius") or 0.0),
        )
        / max(active.chamfer_coverage_min, 1e-6),
        max(
            0.0,
            float(chamfer.get("mean_distance_px") or 1e6)
            / max(active.chamfer_mean_distance_max, 1e-6)
            - 1.0,
        ),
        max(
            0.0,
            active.continuous_edge_span_min
            - float((continuous or {}).get("span_fraction", 0.0)),
        )
        / max(active.continuous_edge_span_min, 1e-6),
    ]
    penalty = 25.0 * len(issues) + 25.0 * sum(shortfalls)
    return {
        **base,
        "available": True,
        "acceptable": not issues,
        "issues": issues,
        "warnings": warnings,
        "penalty": round(float(penalty), 6),
        "metadata": {
            "path": metadata_path,
            "policy": metadata.get("policy") if metadata else None,
        },
        "registration": {
            "normalized_size": [active.normalized_width, active.normalized_height],
            "candidate_alpha_bbox": candidate_bbox,
            "guide_foreground_bbox": guide_bbox,
            "guide_mask_method": guide_mask_method,
            "roi_source": roi_source,
            "foreleg_roi": [round(value, 6) for value in foreleg_roi],
            "muzzle_roi": [round(value, 6) for value in muzzle_roi],
            "planted_paw_roi": [round(value, 6) for value in planted_roi],
            "foreleg_roi_pixels": list(foreleg_box),
            "muzzle_roi_pixels": list(muzzle_box),
            "planted_paw_roi_pixels": list(planted_box),
            "appearance_sampling_policy": terminal_paw_sampling["policy"],
            "raised_paw_terminal_box_pixels": terminal_paw_sampling["raised"],
            "planted_paw_terminal_box_pixels": terminal_paw_sampling["planted"],
        },
        "metrics": {
            "global_pose_alignment": global_alignment,
            "candidate_foreground_pixels": int(
                np.count_nonzero(candidate_mask_normalized)
            ),
            "raised_foreleg_roi_foreground_pixels": candidate_roi_foreground,
            "raised_foreleg_edge_count": candidate_roi_edge_count,
            "raised_foreleg_edge_density": round(edge_density, 6),
            "raised_foreleg_structural_evidence": structural,
            "chamfer": chamfer,
            "continuous_edge": continuous,
            "paw_muzzle_contact": contact,
            "appearance_consistency": appearance,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a transparent Main_case grooming keyframe.",
    )
    parser.add_argument("candidate", type=Path)
    parser.add_argument("geometry_guide", type=Path)
    parser.add_argument("--metadata", type=Path)
    args = parser.parse_args()
    result = evaluate_grooming_pose(
        args.candidate,
        args.geometry_guide,
        geometry_metadata=args.metadata,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["acceptable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
