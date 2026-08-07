#!/usr/bin/env python3
"""Evaluate the fur silhouette of a transparent subject animation.

The profile-comparison gate measures rear-torso and dorsal-contour expansion.
The uploaded-subject gate is pose-independent: it only enforces valid-frame
coverage and absolute normalized top-contour roughness.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageSequence

try:
    import cv2
except ImportError:  # pragma: no cover - exercised by lightweight local runtimes.
    cv2 = None


@dataclass(frozen=True)
class FurContourThresholds:
    """Thresholds for a 5-second, 100-frame eating animation."""

    alpha_threshold: int = 128
    minimum_component_fraction: float = 0.01
    minimum_valid_frames: int = 70
    minimum_valid_fraction: float = 0.90
    rear_x_start: float = 0.12
    rear_x_end: float = 0.58
    upper_height_fraction: float = 0.45
    smoothing_window_fraction: float = 0.08
    upper_area_ratio_max: float = 1.03
    upper_area_p90_ratio_max: float = 1.05
    rear_area_ratio_max: float = 1.03
    rear_area_p90_ratio_max: float = 1.06
    torso_height_ratio_max: float = 1.02
    torso_height_p90_ratio_max: float = 1.04
    dorsal_outgrowth_norm_max: float = 0.02
    roughness_ratio_max: float = 1.25
    roughness_norm_max: float = 0.016
    roughness_p90_ratio_max: float = 1.25
    roughness_p90_norm_max: float = 0.018
    severe_roughness_ratio_max: float = 1.50
    severe_roughness_norm_max: float = 0.022
    moderate_issue_count_to_fail: int = 2


def _largest_component_fallback(mask: np.ndarray) -> np.ndarray:
    """Return the largest 8-connected component without optional OpenCV."""

    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    largest: list[tuple[int, int]] = []
    for start_y, start_x in zip(*np.nonzero(mask), strict=True):
        if visited[start_y, start_x]:
            continue
        visited[start_y, start_x] = True
        pending: deque[tuple[int, int]] = deque([(int(start_y), int(start_x))])
        component: list[tuple[int, int]] = []
        while pending:
            y, x = pending.popleft()
            component.append((y, x))
            for next_y in range(max(0, y - 1), min(height, y + 2)):
                for next_x in range(max(0, x - 1), min(width, x + 2)):
                    if (
                        not visited[next_y, next_x]
                        and bool(mask[next_y, next_x])
                    ):
                        visited[next_y, next_x] = True
                        pending.append((next_y, next_x))
        if len(component) > len(largest):
            largest = component
    result = np.zeros_like(mask, dtype=bool)
    if largest:
        ys, xs = zip(*largest, strict=True)
        result[np.asarray(ys), np.asarray(xs)] = True
    return result


def largest_alpha_component(
    alpha: np.ndarray,
    *,
    alpha_threshold: int = 128,
) -> np.ndarray:
    """Return the largest connected alpha component as a boolean mask."""

    mask = np.asarray(alpha) >= alpha_threshold
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    if cv2 is None:
        return _largest_component_fallback(mask)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )
    if count <= 1:
        return np.zeros_like(mask, dtype=bool)
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest_label


def _odd_window(length: int, fraction: float) -> int:
    window = max(5, int(round(length * fraction)))
    if window % 2 == 0:
        window += 1
    maximum = length - 2 if length % 2 else length - 1
    return max(3, min(window, maximum))


def _smooth_profile(values: np.ndarray, window: int) -> np.ndarray:
    padding = window // 2
    padded = np.pad(values, (padding, padding), mode="edge")
    kernel = np.full(window, 1.0 / window, dtype=np.float64)
    return np.convolve(padded, kernel, mode="valid")


def frame_fur_signature(
    component: np.ndarray,
    *,
    thresholds: FurContourThresholds,
) -> dict[str, float] | None:
    """Measure a single rear-torso silhouette in scale-independent units."""

    rows, columns = np.where(component)
    if not len(columns):
        return None
    left = int(columns.min())
    right = int(columns.max()) + 1
    top = int(rows.min())
    bottom = int(rows.max()) + 1
    bbox_width = right - left
    bbox_height = bottom - top
    if bbox_width < 16 or bbox_height < 16:
        return None

    rear_left = left + int(round(bbox_width * thresholds.rear_x_start))
    rear_right = left + int(round(bbox_width * thresholds.rear_x_end))
    rear_left = max(left, min(rear_left, right - 2))
    rear_right = max(rear_left + 2, min(rear_right, right))

    xs: list[int] = []
    tops: list[int] = []
    bottoms: list[int] = []
    for x in range(rear_left, rear_right):
        occupied = np.flatnonzero(component[:, x])
        if occupied.size < 2:
            continue
        xs.append(x)
        tops.append(int(occupied[0]))
        bottoms.append(int(occupied[-1]))
    minimum_columns = max(8, int(round((rear_right - rear_left) * 0.65)))
    if len(xs) < minimum_columns:
        return None

    top_profile = np.asarray(tops, dtype=np.float64)
    bottom_profile = np.asarray(bottoms, dtype=np.float64)
    column_heights = bottom_profile - top_profile + 1.0
    bottom_anchor = float(np.median(bottom_profile))
    # Width is a more stable body-size anchor for this check than height:
    # abnormal dorsal growth increases the silhouette height itself, so dividing
    # by height would partially normalize away the defect we need to detect.
    # Width still makes the metrics invariant to small uniform scale changes.
    body_scale = float(max(1, bbox_width))
    area_scale = float(max(1, bbox_width * bbox_width))

    rear_mask = component[:, rear_left:rear_right]
    rear_area_norm = float(np.count_nonzero(rear_mask)) / area_scale
    upper_cutoff = bottom_anchor - thresholds.upper_height_fraction * bbox_height
    upper_rows = np.arange(component.shape[0], dtype=np.float64)[:, None]
    upper_area = np.count_nonzero(rear_mask & (upper_rows < upper_cutoff))
    upper_area_norm = float(upper_area) / area_scale

    smoothing_window = _odd_window(
        len(top_profile),
        thresholds.smoothing_window_fraction,
    )
    smoothed = _smooth_profile(top_profile, smoothing_window)
    roughness_norm = float(
        np.sqrt(np.mean(np.square(top_profile - smoothed)))
        / float(max(1, bbox_height))
    )

    return {
        "upper_area_norm": upper_area_norm,
        "rear_area_norm": rear_area_norm,
        "torso_height_norm": float(np.median(column_heights) / body_scale),
        "dorsal_height_p90_norm": float(
            np.percentile(bottom_anchor - top_profile, 90) / body_scale
        ),
        "top_contour_roughness_norm": roughness_norm,
        "bbox_width": float(bbox_width),
        "bbox_height": float(bbox_height),
    }


def _decode_alpha_frames(path: Path) -> list[np.ndarray]:
    with Image.open(path) as source:
        return [
            np.asarray(frame.convert("RGBA").getchannel("A"), dtype=np.uint8)
            for frame in ImageSequence.Iterator(source)
        ]


def _aggregate(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "median": round(float(np.median(array)), 8),
        "p10": round(float(np.percentile(array, 10)), 8),
        "p90": round(float(np.percentile(array, 90)), 8),
        "min": round(float(array.min()), 8),
        "max": round(float(array.max()), 8),
    }


def animation_fur_signature(
    path: str | Path,
    *,
    thresholds: FurContourThresholds | None = None,
) -> dict[str, Any]:
    """Return per-animation rear-contour aggregates and validity counts."""

    active = thresholds or FurContourThresholds()
    resolved = Path(path)
    alpha_frames = _decode_alpha_frames(resolved)
    frame_area = int(alpha_frames[0].size) if alpha_frames else 0
    minimum_component_area = max(
        64,
        int(round(frame_area * active.minimum_component_fraction)),
    )
    signatures: list[dict[str, float]] = []
    invalid_reasons: dict[str, int] = {}
    for alpha in alpha_frames:
        component = largest_alpha_component(
            alpha,
            alpha_threshold=active.alpha_threshold,
        )
        if int(np.count_nonzero(component)) < minimum_component_area:
            invalid_reasons["component_too_small"] = (
                invalid_reasons.get("component_too_small", 0) + 1
            )
            continue
        signature = frame_fur_signature(component, thresholds=active)
        if signature is None:
            invalid_reasons["rear_contour_unavailable"] = (
                invalid_reasons.get("rear_contour_unavailable", 0) + 1
            )
            continue
        signatures.append(signature)

    total_frames = len(alpha_frames)
    valid_frames = len(signatures)
    valid_fraction = valid_frames / float(total_frames) if total_frames else 0.0
    validity_issues: list[str] = []
    if valid_frames < active.minimum_valid_frames:
        validity_issues.append("insufficient_valid_frames")
    if valid_fraction < active.minimum_valid_fraction:
        validity_issues.append("insufficient_valid_frame_fraction")
    aggregates = (
        {
            key: _aggregate(signature[key] for signature in signatures)
            for key in signatures[0]
        }
        if signatures
        else {}
    )
    return {
        "path": str(resolved),
        "frame_count": total_frames,
        "valid_frame_count": valid_frames,
        "valid_frame_fraction": round(valid_fraction, 8),
        "invalid_reasons": invalid_reasons,
        "validity_issues": validity_issues,
        "aggregates": aggregates,
    }


def _median(signature: dict[str, Any], metric: str) -> float:
    return float(signature["aggregates"][metric]["median"])


def _p90(signature: dict[str, Any], metric: str) -> float:
    return float(signature["aggregates"][metric]["p90"])


def compare_fur_contours(
    candidate_gif: str | Path,
    profile_baseline_gif: str | Path,
    *,
    thresholds: FurContourThresholds | None = None,
) -> dict[str, Any]:
    """Compare a candidate transparent GIF with a validated profile baseline."""

    active = thresholds or FurContourThresholds()
    reference = animation_fur_signature(
        profile_baseline_gif,
        thresholds=active,
    )
    candidate = animation_fur_signature(candidate_gif, thresholds=active)
    issues = list(
        dict.fromkeys(
            [
                *(f"reference_{issue}" for issue in reference["validity_issues"]),
                *(f"candidate_{issue}" for issue in candidate["validity_issues"]),
            ]
        )
    )
    comparisons: dict[str, float] = {}
    moderate_issues: list[str] = []
    severe_issues: list[str] = []

    if not issues:
        ratio_specs = (
            ("upper_area_norm", active.upper_area_ratio_max, "upper_area_expanded"),
            ("rear_area_norm", active.rear_area_ratio_max, "rear_area_expanded"),
            (
                "torso_height_norm",
                active.torso_height_ratio_max,
                "torso_height_expanded",
            ),
        )
        for metric, maximum, issue in ratio_specs:
            reference_value = max(_median(reference, metric), 1e-9)
            ratio = _median(candidate, metric) / reference_value
            comparisons[f"{metric}_median_ratio"] = round(ratio, 8)
            if ratio > maximum:
                moderate_issues.append(issue)

        p90_ratio_specs = (
            (
                "upper_area_norm",
                active.upper_area_p90_ratio_max,
                "upper_area_p90_expanded",
            ),
            (
                "rear_area_norm",
                active.rear_area_p90_ratio_max,
                "rear_area_p90_expanded",
            ),
            (
                "torso_height_norm",
                active.torso_height_p90_ratio_max,
                "torso_height_p90_expanded",
            ),
        )
        for metric, maximum, issue in p90_ratio_specs:
            reference_value = max(_p90(reference, metric), 1e-9)
            ratio = _p90(candidate, metric) / reference_value
            comparisons[f"{metric}_p90_ratio"] = round(ratio, 8)
            if ratio > maximum:
                moderate_issues.append(issue)

        dorsal_outgrowth = _median(
            candidate,
            "dorsal_height_p90_norm",
        ) - _median(reference, "dorsal_height_p90_norm")
        comparisons["dorsal_outgrowth_norm"] = round(dorsal_outgrowth, 8)
        if dorsal_outgrowth > active.dorsal_outgrowth_norm_max:
            moderate_issues.append("dorsal_contour_expanded")

        reference_roughness = max(
            _median(reference, "top_contour_roughness_norm"),
            1e-9,
        )
        candidate_roughness = _median(
            candidate,
            "top_contour_roughness_norm",
        )
        roughness_ratio = candidate_roughness / reference_roughness
        comparisons["top_contour_roughness_median_ratio"] = round(
            roughness_ratio,
            8,
        )
        comparisons["top_contour_roughness_candidate_median"] = round(
            candidate_roughness,
            8,
        )
        candidate_roughness_p90 = _p90(
            candidate,
            "top_contour_roughness_norm",
        )
        reference_roughness_p90 = max(
            _p90(reference, "top_contour_roughness_norm"),
            1e-9,
        )
        roughness_p90_ratio = candidate_roughness_p90 / reference_roughness_p90
        comparisons["top_contour_roughness_p90_ratio"] = round(
            roughness_p90_ratio,
            8,
        )
        comparisons["top_contour_roughness_candidate_p90"] = round(
            candidate_roughness_p90,
            8,
        )
        if (
            roughness_ratio > active.severe_roughness_ratio_max
            and candidate_roughness > active.severe_roughness_norm_max
        ):
            severe_issues.append("severe_top_contour_roughness")
        elif (
            roughness_ratio > active.roughness_ratio_max
            and candidate_roughness > active.roughness_norm_max
        ):
            moderate_issues.append("top_contour_roughness")
        if (
            roughness_p90_ratio > active.roughness_p90_ratio_max
            and candidate_roughness_p90 > active.roughness_p90_norm_max
        ):
            moderate_issues.append("top_contour_roughness_p90")

        if severe_issues:
            issues.extend(severe_issues)
        elif len(set(moderate_issues)) >= active.moderate_issue_count_to_fail:
            issues.extend(moderate_issues)

    return {
        "acceptable": not issues,
        "issues": list(dict.fromkeys(issues)),
        "observed_moderate_issues": list(dict.fromkeys(moderate_issues)),
        "observed_severe_issues": list(dict.fromkeys(severe_issues)),
        "reference": reference,
        "candidate": candidate,
        "comparisons": comparisons,
        "thresholds": asdict(active),
    }


def evaluate_animation_fur_quality(
    candidate_gif: str | Path,
    *,
    thresholds: FurContourThresholds | None = None,
) -> dict[str, Any]:
    """Gate one animation by absolute contour quality, without a profile shape.

    This policy is intended for uploaded subjects whose final pose can differ
    substantially from the default profile cat.  It deliberately ignores body
    area, height, and dorsal-growth ratios and only evaluates whether enough
    frames expose a measurable subject plus the absolute normalized roughness
    of that subject's top contour.
    """

    active = thresholds or FurContourThresholds()
    candidate = animation_fur_signature(candidate_gif, thresholds=active)
    issues = list(
        dict.fromkeys(
            f"candidate_{issue}" for issue in candidate["validity_issues"]
        )
    )
    moderate_issues: list[str] = []
    severe_issues: list[str] = []
    policy = {
        "name": "uploaded_subject_absolute_roughness_v1",
        "reference_geometry_comparison": False,
        "absolute_thresholds_only": True,
        "fail_closed_on_invalid_animation": True,
        "moderate_limits": {
            "top_contour_roughness_norm_median_max": active.roughness_norm_max,
            "top_contour_roughness_norm_p90_max": (
                active.roughness_p90_norm_max
            ),
        },
        "severe_limits": {
            "top_contour_roughness_norm_median_max": (
                active.severe_roughness_norm_max
            ),
        },
        "moderate_issue_count_to_fail": active.moderate_issue_count_to_fail,
    }

    if not issues:
        median_roughness = _median(
            candidate,
            "top_contour_roughness_norm",
        )
        p90_roughness = _p90(
            candidate,
            "top_contour_roughness_norm",
        )
        if median_roughness > active.severe_roughness_norm_max:
            severe_issues.append("severe_top_contour_roughness")
        elif median_roughness > active.roughness_norm_max:
            moderate_issues.append("top_contour_roughness")
        if p90_roughness > active.roughness_p90_norm_max:
            moderate_issues.append("top_contour_roughness_p90")

        if severe_issues:
            issues.extend(severe_issues)
        elif len(set(moderate_issues)) >= active.moderate_issue_count_to_fail:
            issues.extend(moderate_issues)

    return {
        "acceptable": not issues,
        "issues": list(dict.fromkeys(issues)),
        "observed_moderate_issues": list(dict.fromkeys(moderate_issues)),
        "observed_severe_issues": list(dict.fromkeys(severe_issues)),
        "candidate": candidate,
        "thresholds": asdict(active),
        "policy": policy,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_gif", type=Path)
    parser.add_argument("profile_baseline_gif", type=Path)
    args = parser.parse_args()
    result = compare_fur_contours(
        args.candidate_gif,
        args.profile_baseline_gif,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["acceptable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
