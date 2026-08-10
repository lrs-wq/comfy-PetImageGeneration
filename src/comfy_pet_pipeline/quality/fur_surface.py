#!/usr/bin/env python3
"""Diagnostic still-image gate for overgrown and brush-like pet fur.

The existing animation gate intentionally measures only the outer alpha
contour.  This module adds a single-frame RGBA/green-screen gate that measures
body geometry, contour tightening, and coherent internal strokes.  A clean
result is only called accepted when the caller supplies an explicitly
validated profile baseline; an absolute-only clean result remains
diagnostic/unknown.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


GATE_VERSION = "1.2.0"
THRESHOLD_VERSION = "uploaded-identity-surface-20260806-v4"
POLICY = "validated_action_reference_fur_surface_v1"


@dataclass(frozen=True)
class FurSurfaceThresholds:
    """Thresholds calibrated on the sleek-tabby profile diagnostic set."""

    analysis_size: int = 512
    alpha_threshold: int = 96
    minimum_component_fraction: float = 0.015
    border_fraction: float = 0.04
    green_border_fraction_min: float = 0.45
    green_minimum: int = 60
    green_dominance: int = 18
    inner_erode_fraction: float = 0.014
    contour_open_fraction: float = 0.029
    texture_top: float = 0.08
    texture_bottom: float = 0.84
    texture_left: float = 0.06
    texture_right: float = 0.94
    face_exclusion_left: float = 0.32
    face_exclusion_right: float = 0.68
    face_exclusion_top: float = 0.25
    face_exclusion_bottom: float = 0.58
    fine_gradient_sigma: float = 1.0
    coarse_gradient_sigma: float = 5.0
    structure_tensor_sigma_multiplier: float = 2.0
    strong_gradient_percentile: float = 65.0
    coherence_min: float = 0.55
    coherent_fraction_level: float = 0.72
    radial_alignment_min: float = 0.86
    contour_tightening_loss_max: float = 0.012
    silhouette_compactness_max: float = 4.5
    # The production Main_case v2 eating/grooming references are the frozen
    # accepted baselines.  Keep a small margin above their measured maxima
    # (0.22811 coarse gradient and 0.49990 coherent fraction); relative gates
    # below still reject candidates that expand beyond either action baseline.
    coarse_gradient_p90_max: float = 0.24
    coarse_coherent_fraction_max: float = 0.52
    coherence_gradient_p90_min: float = 0.14
    radial_brush_fraction_max: float = 0.52
    hough_radial_line_fraction_max: float = 0.65
    # A directional ratio is undefined as defect evidence when Hough finds only
    # a handful of short lines.  Require radial line length covering at least
    # 0.8% of the texture ROI before the ratio can reject a candidate.
    hough_radial_line_density_min: float = 0.008
    # Frozen Main_case v2 maximum is 0.22566.  Radial-line and relative long-
    # fur checks remain independent, so this calibration does not disable the
    # brush/overgrown-fur rejection paths.
    coherent_long_line_energy_max: float = 0.24
    upper_area_ratio_max: float = 1.12
    torso_area_ratio_max: float = 1.12
    upper_width_ratio_max: float = 1.12
    torso_width_ratio_max: float = 1.12
    geometry_delta_min: float = 0.03
    contour_tightening_ratio_max: float = 2.0
    contour_tightening_candidate_min: float = 0.008
    coarse_gradient_ratio_max: float = 1.28
    coherent_fraction_ratio_max: float = 1.25
    radial_brush_ratio_max: float = 1.22
    long_line_energy_ratio_max: float = 1.20
    texture_delta_min: float = 0.015


def _native_float(value: float, digits: int = 8) -> float:
    result = float(value)
    if not np.isfinite(result):
        return 0.0
    return round(result, digits)


def _odd_kernel(size: int, fraction: float, minimum: int = 3) -> int:
    value = max(minimum, int(round(size * fraction)))
    if value % 2 == 0:
        value += 1
    return value


def _largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )
    if count <= 1:
        return np.zeros_like(mask, dtype=bool)
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == label


def _border_mask(height: int, width: int, fraction: float) -> np.ndarray:
    border = np.zeros((height, width), dtype=bool)
    thickness = max(4, int(round(min(height, width) * fraction)))
    border[:thickness, :] = True
    border[-thickness:, :] = True
    border[:, :thickness] = True
    border[:, -thickness:] = True
    return border


def _extract_foreground(
    rgba: np.ndarray,
    *,
    source_has_alpha: bool,
    thresholds: FurSurfaceThresholds,
) -> tuple[np.ndarray, str, float, list[str]]:
    """Extract the foreground from alpha, chroma green, or a fallback matte."""

    height, width = rgba.shape[:2]
    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3]
    warnings: list[str] = []

    if source_has_alpha and bool(np.any(alpha < 250)):
        mask = _largest_component(alpha >= thresholds.alpha_threshold)
        method = "alpha"
        confidence = 1.0
    else:
        red = rgb[:, :, 0].astype(np.float32)
        green = rgb[:, :, 1].astype(np.float32)
        blue = rgb[:, :, 2].astype(np.float32)
        border = _border_mask(height, width, thresholds.border_fraction)
        chroma_green = (
            (green >= thresholds.green_minimum)
            & (green - np.maximum(red, blue) >= thresholds.green_dominance)
            & (green >= red * 1.08)
            & (green >= blue * 1.03)
        )
        green_border_fraction = float(np.mean(chroma_green[border]))
        if green_border_fraction >= thresholds.green_border_fraction_min:
            raw = (~chroma_green).astype(np.uint8)
            raw = cv2.morphologyEx(
                raw,
                cv2.MORPH_CLOSE,
                np.ones((5, 5), dtype=np.uint8),
            )
            mask = _largest_component(raw > 0)
            method = "green_screen"
            confidence = green_border_fraction
        else:
            # Real regression artifacts include opaque preview frames.  Keep a
            # best-effort GrabCut fallback so they can still be rejected, but
            # never treat this fallback as sufficient evidence for acceptance.
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            grab = np.full((height, width), cv2.GC_PR_BGD, dtype=np.uint8)
            border_width = max(3, int(round(min(height, width) * 0.02)))
            grab[:border_width, :] = cv2.GC_BGD
            grab[-border_width:, :] = cv2.GC_BGD
            grab[:, :border_width] = cv2.GC_BGD
            grab[:, -border_width:] = cv2.GC_BGD
            background_median = np.median(rgb[border], axis=0)
            color_distance = np.linalg.norm(
                rgb.astype(np.float32) - background_median.astype(np.float32),
                axis=2,
            )
            grab[color_distance > 30.0] = cv2.GC_PR_FGD
            background_model = np.zeros((1, 65), dtype=np.float64)
            foreground_model = np.zeros((1, 65), dtype=np.float64)
            cv2.grabCut(
                bgr,
                grab,
                None,
                background_model,
                foreground_model,
                5,
                cv2.GC_INIT_WITH_MASK,
            )
            mask = _largest_component(
                (grab == cv2.GC_FGD) | (grab == cv2.GC_PR_FGD)
            )
            method = "opaque_background_fallback"
            confidence = max(0.0, 1.0 - green_border_fraction)
            warnings.append("non_contract_opaque_background_fallback")

    minimum_area = max(
        64,
        int(round(height * width * thresholds.minimum_component_fraction)),
    )
    if int(np.count_nonzero(mask)) < minimum_area:
        raise ValueError("foreground_component_too_small")
    return mask, method, confidence, warnings


def _canonicalize(
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    rows, columns = np.where(mask)
    top = int(rows.min())
    bottom = int(rows.max()) + 1
    left = int(columns.min())
    right = int(columns.max()) + 1
    crop_rgb = rgb[top:bottom, left:right]
    crop_mask = mask[top:bottom, left:right]
    scale = size / float(max(crop_rgb.shape[:2]))
    width = max(1, int(round(crop_rgb.shape[1] * scale)))
    height = max(1, int(round(crop_rgb.shape[0] * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized_rgb = cv2.resize(crop_rgb, (width, height), interpolation=interpolation)
    resized_mask = cv2.resize(
        crop_mask.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    ) > 0
    offset_x = (size - width) // 2
    offset_y = (size - height) // 2
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canonical_mask = np.zeros((size, size), dtype=bool)
    canvas[offset_y : offset_y + height, offset_x : offset_x + width] = (
        resized_rgb
    )
    canonical_mask[
        offset_y : offset_y + height,
        offset_x : offset_x + width,
    ] = resized_mask
    return canvas, canonical_mask, {
        "source_left": left,
        "source_top": top,
        "source_width": right - left,
        "source_height": bottom - top,
        "canonical_left": offset_x,
        "canonical_top": offset_y,
        "canonical_width": width,
        "canonical_height": height,
    }


def _band_geometry(
    mask: np.ndarray,
    *,
    top: int,
    left: int,
    width: int,
    height: int,
    start: float,
    end: float,
) -> dict[str, float]:
    first = top + int(round(height * start))
    last = top + int(round(height * end))
    band = mask[first:last, left : left + width]
    occupied_widths: list[float] = []
    for row in band:
        occupied = np.flatnonzero(row)
        if occupied.size:
            occupied_widths.append(float(occupied[-1] - occupied[0] + 1))
    width_p90 = float(np.percentile(occupied_widths, 90)) if occupied_widths else 0.0
    area = float(np.count_nonzero(band))
    return {
        "area_bbox_fraction": _native_float(area / max(1, band.size)),
        "area_height2_norm": _native_float(area / max(1, height * height)),
        "width_p90_bbox_norm": _native_float(width_p90 / max(1, width)),
        "width_p90_height_norm": _native_float(width_p90 / max(1, height)),
    }


def _contour_metrics(
    mask: np.ndarray,
    *,
    alpha: np.ndarray,
    source_has_soft_alpha: bool,
    thresholds: FurSurfaceThresholds,
) -> dict[str, float | None]:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    contour = max(contours, key=cv2.contourArea)
    perimeter = float(cv2.arcLength(contour, True))
    area = float(cv2.contourArea(contour))
    kernel_size = _odd_kernel(
        thresholds.analysis_size,
        thresholds.contour_open_fraction,
    )
    opened = cv2.morphologyEx(
        mask.astype(np.uint8),
        cv2.MORPH_OPEN,
        np.ones((kernel_size, kernel_size), dtype=np.uint8),
    ) > 0
    opened_contours, _ = cv2.findContours(
        opened.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    opened_perimeter = (
        float(cv2.arcLength(max(opened_contours, key=cv2.contourArea), True))
        if opened_contours
        else 0.0
    )
    foreground_area = max(1, int(np.count_nonzero(mask)))
    removed = int(np.count_nonzero(mask & ~opened))
    soft_fraction: float | None = None
    if source_has_soft_alpha:
        soft = (alpha > 0) & (alpha < 250)
        soft_fraction = _native_float(
            np.count_nonzero(soft) / max(1, np.count_nonzero(alpha > 0))
        )
    return {
        "silhouette_compactness": _native_float(
            perimeter * perimeter / (4.0 * np.pi * max(area, 1.0))
        ),
        "contour_perimeter_height_norm": _native_float(
            perimeter / thresholds.analysis_size
        ),
        "contour_tightening_removed_fraction": _native_float(
            removed / foreground_area
        ),
        "contour_tightening_area_ratio": _native_float(
            np.count_nonzero(opened) / foreground_area
        ),
        "contour_tightening_perimeter_ratio": _native_float(
            opened_perimeter / max(perimeter, 1e-9)
        ),
        "alpha_soft_edge_fraction": soft_fraction,
    }


def _gradient_tensor(
    gray: np.ndarray,
    sigma: float,
    tensor_sigma_multiplier: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    smoothed = cv2.GaussianBlur(gray, (0, 0), sigma)
    gradient_x = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gradient_x, gradient_y)
    tensor_sigma = max(2.0, sigma * tensor_sigma_multiplier)
    xx = cv2.GaussianBlur(gradient_x * gradient_x, (0, 0), tensor_sigma)
    yy = cv2.GaussianBlur(gradient_y * gradient_y, (0, 0), tensor_sigma)
    xy = cv2.GaussianBlur(gradient_x * gradient_y, (0, 0), tensor_sigma)
    coherence = np.sqrt((xx - yy) ** 2 + 4.0 * xy * xy) / (
        xx + yy + 1e-7
    )
    line_angle = np.arctan2(gradient_y, gradient_x) + np.pi / 2.0
    return magnitude, coherence, line_angle


def _radial_metrics(
    magnitude: np.ndarray,
    coherence: np.ndarray,
    line_angle: np.ndarray,
    roi: np.ndarray,
    *,
    bounds: dict[str, int],
    thresholds: FurSurfaceThresholds,
) -> dict[str, Any]:
    values = magnitude[roi]
    cutoff = float(np.percentile(values, thresholds.strong_gradient_percentile))
    selected = roi & (magnitude >= cutoff) & (coherence >= thresholds.coherence_min)
    if not np.any(selected):
        return {
            "fraction": 0.0,
            "energy": 0.0,
            "center_bbox_norm": [0.5, 0.42],
            "selected_pixel_count": 0,
        }
    yy, xx = np.indices(roi.shape)
    best: tuple[float, float, float, float] | None = None
    for center_x_norm in (0.38, 0.50, 0.62):
        for center_y_norm in (0.30, 0.42, 0.54):
            center_x = bounds["canonical_left"] + center_x_norm * bounds[
                "canonical_width"
            ]
            center_y = bounds["canonical_top"] + center_y_norm * bounds[
                "canonical_height"
            ]
            radial_angle = np.arctan2(yy - center_y, xx - center_x)
            alignment = np.abs(np.cos(line_angle - radial_angle))
            fraction = float(
                np.mean(alignment[selected] >= thresholds.radial_alignment_min)
            )
            energy = float(
                np.mean(
                    alignment[selected]
                    * coherence[selected]
                    * np.minimum(magnitude[selected] / (cutoff + 1e-7), 2.0)
                    / 2.0
                )
            )
            row = (fraction, energy, center_x_norm, center_y_norm)
            if best is None or row[:2] > best[:2]:
                best = row
    assert best is not None
    return {
        "fraction": _native_float(best[0]),
        "energy": _native_float(best[1]),
        "center_bbox_norm": [best[2], best[3]],
        "selected_pixel_count": int(np.count_nonzero(selected)),
    }


def _hough_line_metrics(
    gray_u8: np.ndarray,
    roi: np.ndarray,
    *,
    radial_center: list[float],
    bounds: dict[str, int],
    thresholds: FurSurfaceThresholds,
) -> dict[str, float | int]:
    blurred = cv2.GaussianBlur(gray_u8, (0, 0), 1.0)
    edges = cv2.Canny(blurred, 40, 100)
    edges[~roi] = 0
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=12,
        minLineLength=12,
        maxLineGap=4,
    )
    center_x = bounds["canonical_left"] + radial_center[0] * bounds[
        "canonical_width"
    ]
    center_y = bounds["canonical_top"] + radial_center[1] * bounds[
        "canonical_height"
    ]
    lengths: list[float] = []
    radial_lengths: list[float] = []
    if lines is not None:
        for raw in lines[:, 0]:
            x1, y1, x2, y2 = (float(value) for value in raw)
            middle_x = int(round((x1 + x2) / 2.0))
            middle_y = int(round((y1 + y2) / 2.0))
            middle_x = min(roi.shape[1] - 1, max(0, middle_x))
            middle_y = min(roi.shape[0] - 1, max(0, middle_y))
            if not roi[middle_y, middle_x]:
                continue
            length = float(np.hypot(x2 - x1, y2 - y1))
            line_angle = float(np.arctan2(y2 - y1, x2 - x1))
            radial_angle = float(
                np.arctan2(middle_y - center_y, middle_x - center_x)
            )
            lengths.append(length)
            if abs(np.cos(line_angle - radial_angle)) >= thresholds.radial_alignment_min:
                radial_lengths.append(length)
    total_length = float(sum(lengths))
    radial_total_length = float(sum(radial_lengths))
    roi_pixel_count = max(1, int(np.count_nonzero(roi)))
    return {
        "line_count": len(lengths),
        "line_density": _native_float(total_length / roi_pixel_count),
        "radial_line_density": _native_float(
            radial_total_length / roi_pixel_count
        ),
        "line_length_p90_norm": _native_float(
            (float(np.percentile(lengths, 90)) if lengths else 0.0)
            / thresholds.analysis_size
        ),
        "radial_line_fraction": _native_float(
            radial_total_length / max(total_length, 1e-9)
        ),
        "edge_pixel_fraction": _native_float(
            np.count_nonzero(edges[roi]) / max(1, np.count_nonzero(roi))
        ),
    }


def _texture_metrics(
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    bounds: dict[str, int],
    thresholds: FurSurfaceThresholds,
) -> dict[str, Any]:
    gray_u8 = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = gray_u8.astype(np.float32) / 255.0
    erode_size = _odd_kernel(
        thresholds.analysis_size,
        thresholds.inner_erode_fraction,
    )
    inner = cv2.erode(
        mask.astype(np.uint8),
        np.ones((erode_size, erode_size), dtype=np.uint8),
    ) > 0
    yy, xx = np.indices(mask.shape)
    relative_x = (xx - bounds["canonical_left"]) / max(
        1, bounds["canonical_width"]
    )
    relative_y = (yy - bounds["canonical_top"]) / max(
        1, bounds["canonical_height"]
    )
    face = (
        (relative_x >= thresholds.face_exclusion_left)
        & (relative_x <= thresholds.face_exclusion_right)
        & (relative_y >= thresholds.face_exclusion_top)
        & (relative_y <= thresholds.face_exclusion_bottom)
    )
    roi = (
        inner
        & (relative_x >= thresholds.texture_left)
        & (relative_x <= thresholds.texture_right)
        & (relative_y >= thresholds.texture_top)
        & (relative_y <= thresholds.texture_bottom)
        & ~face
    )
    if int(np.count_nonzero(roi)) < 128:
        raise ValueError("texture_roi_too_small")

    fine_magnitude, fine_coherence, fine_angle = _gradient_tensor(
        gray,
        thresholds.fine_gradient_sigma,
        thresholds.structure_tensor_sigma_multiplier,
    )
    coarse_magnitude, coarse_coherence, coarse_angle = _gradient_tensor(
        gray,
        thresholds.coarse_gradient_sigma,
        thresholds.structure_tensor_sigma_multiplier,
    )
    fine_values = fine_magnitude[roi]
    coarse_values = coarse_magnitude[roi]
    coarse_coherence_values = coarse_coherence[roi]
    fine_radial = _radial_metrics(
        fine_magnitude,
        fine_coherence,
        fine_angle,
        roi,
        bounds=bounds,
        thresholds=thresholds,
    )
    coarse_radial = _radial_metrics(
        coarse_magnitude,
        coarse_coherence,
        coarse_angle,
        roi,
        bounds=bounds,
        thresholds=thresholds,
    )
    persistence = np.abs(np.cos(fine_angle - coarse_angle))
    persistent_selection = (
        roi
        & (
            fine_magnitude
            >= np.percentile(fine_values, thresholds.strong_gradient_percentile)
        )
        & (
            coarse_magnitude
            >= np.percentile(coarse_values, thresholds.strong_gradient_percentile)
        )
    )
    persistence_fraction = (
        float(np.mean(persistence[persistent_selection] >= 0.94))
        if np.any(persistent_selection)
        else 0.0
    )
    coarse_p90 = float(np.percentile(coarse_values, 90))
    coherent_fraction = float(
        np.mean(coarse_coherence_values >= thresholds.coherent_fraction_level)
    )
    # Large-scale gradient energy weighted by tensor coherence is deliberately
    # sensitive to long, repeated brush strokes while ignoring isolated edges.
    coherent_long_line_energy = coarse_p90 * (0.5 + coherent_fraction)
    hough = _hough_line_metrics(
        gray_u8,
        roi,
        radial_center=fine_radial["center_bbox_norm"],
        bounds=bounds,
        thresholds=thresholds,
    )
    return {
        "roi_pixel_count": int(np.count_nonzero(roi)),
        "fine_gradient_p90": _native_float(np.percentile(fine_values, 90)),
        "coarse_gradient_p90": _native_float(coarse_p90),
        "coarse_to_fine_gradient_mean_ratio": _native_float(
            np.mean(coarse_values) / (np.mean(fine_values) + 1e-7)
        ),
        "structure_tensor_coherence_mean": _native_float(
            np.mean(coarse_coherence_values)
        ),
        "structure_tensor_coherent_fraction": _native_float(coherent_fraction),
        "orientation_persistence_fraction": _native_float(persistence_fraction),
        "fine_radial_brush_fraction": fine_radial["fraction"],
        "fine_radial_brush_energy": fine_radial["energy"],
        "fine_radial_center_bbox_norm": fine_radial["center_bbox_norm"],
        "coarse_radial_brush_fraction": coarse_radial["fraction"],
        "coarse_radial_brush_energy": coarse_radial["energy"],
        "coherent_long_line_energy": _native_float(coherent_long_line_energy),
        "hough": hough,
    }


def analyze_fur_surface(
    path: str | Path,
    *,
    thresholds: FurSurfaceThresholds | None = None,
) -> dict[str, Any]:
    """Return a JSON-safe fur signature for one still image."""

    active = thresholds or FurSurfaceThresholds()
    resolved = Path(path)
    with Image.open(resolved) as source:
        source_has_alpha = "A" in source.getbands()
        rgba = np.asarray(source.convert("RGBA"), dtype=np.uint8)
    alpha = rgba[:, :, 3]
    mask, method, confidence, warnings = _extract_foreground(
        rgba,
        source_has_alpha=source_has_alpha,
        thresholds=active,
    )
    rgb, canonical_mask, bounds = _canonicalize(
        rgba[:, :, :3],
        mask,
        size=active.analysis_size,
    )
    supported_contract = method in {"alpha", "green_screen"}
    upper = _band_geometry(
        canonical_mask,
        top=bounds["canonical_top"],
        left=bounds["canonical_left"],
        width=bounds["canonical_width"],
        height=bounds["canonical_height"],
        start=0.05,
        end=0.45,
    )
    torso = _band_geometry(
        canonical_mask,
        top=bounds["canonical_top"],
        left=bounds["canonical_left"],
        width=bounds["canonical_width"],
        height=bounds["canonical_height"],
        start=0.30,
        end=0.75,
    )
    geometry = {
        "bbox_width": bounds["source_width"],
        "bbox_height": bounds["source_height"],
        "bbox_aspect_ratio": _native_float(
            bounds["source_width"] / max(1, bounds["source_height"])
        ),
        "foreground_image_fraction": _native_float(np.mean(mask)),
        "upper_body": upper,
        "torso": torso,
    }
    return {
        "path": str(resolved),
        "valid": True,
        "input": {
            "width": int(rgba.shape[1]),
            "height": int(rgba.shape[0]),
            "foreground_method": method,
            "foreground_confidence": _native_float(confidence),
            "supported_contract": supported_contract,
            "contract": "transparent_alpha_or_green_screen",
        },
        "warnings": warnings,
        "metrics": {
            "geometry": geometry,
            "contour": _contour_metrics(
                canonical_mask,
                alpha=alpha,
                source_has_soft_alpha=(
                    source_has_alpha and bool(np.any((alpha > 0) & (alpha < 250)))
                ),
                thresholds=active,
            ),
            "texture": _texture_metrics(
                rgb,
                canonical_mask,
                bounds=bounds,
                thresholds=active,
            ),
        },
    }


def _absolute_issues(
    signature: dict[str, Any],
    thresholds: FurSurfaceThresholds,
) -> list[str]:
    contour = signature["metrics"]["contour"]
    texture = signature["metrics"]["texture"]
    hough = texture["hough"]
    issues: list[str] = []
    if (
        contour["contour_tightening_removed_fraction"]
        > thresholds.contour_tightening_loss_max
    ):
        issues.append("excess_contour_fringe")
    if contour["silhouette_compactness"] > thresholds.silhouette_compactness_max:
        issues.append("irregular_expanded_silhouette")
    if texture["coarse_gradient_p90"] > thresholds.coarse_gradient_p90_max:
        issues.append("coarse_fur_stroke_energy")
    if (
        texture["structure_tensor_coherent_fraction"]
        > thresholds.coarse_coherent_fraction_max
        and texture["coarse_gradient_p90"]
        > thresholds.coherence_gradient_p90_min
    ):
        issues.append("brush_texture_coherence")
    if (
        texture["fine_radial_brush_fraction"]
        > thresholds.radial_brush_fraction_max
    ):
        issues.append("radial_brush_texture")
    if (
        hough["radial_line_fraction"]
        > thresholds.hough_radial_line_fraction_max
        and hough["radial_line_density"]
        >= thresholds.hough_radial_line_density_min
    ):
        issues.append("radial_long_lines")
    if (
        texture["coherent_long_line_energy"]
        > thresholds.coherent_long_line_energy_max
    ):
        issues.append("coherent_long_fur_energy")
    return list(dict.fromkeys(issues))


def _metric(signature: dict[str, Any], dotted_path: str) -> float:
    value: Any = signature["metrics"]
    for part in dotted_path.split("."):
        value = value[part]
    return float(value)


def _relative_audit(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    thresholds: FurSurfaceThresholds,
) -> tuple[dict[str, float], list[str]]:
    comparisons: dict[str, float] = {}
    issues: list[str] = []

    geometry_specs = (
        (
            "geometry.upper_body.area_height2_norm",
            thresholds.upper_area_ratio_max,
            "upper_body_area_expanded",
        ),
        (
            "geometry.torso.area_height2_norm",
            thresholds.torso_area_ratio_max,
            "torso_area_expanded",
        ),
        (
            "geometry.upper_body.width_p90_height_norm",
            thresholds.upper_width_ratio_max,
            "upper_body_width_expanded",
        ),
        (
            "geometry.torso.width_p90_height_norm",
            thresholds.torso_width_ratio_max,
            "torso_width_expanded",
        ),
    )
    for path, maximum, issue in geometry_specs:
        candidate_value = _metric(candidate, path)
        baseline_value = _metric(baseline, path)
        ratio = candidate_value / max(baseline_value, 1e-9)
        delta = candidate_value - baseline_value
        key = path.replace(".", "_")
        comparisons[f"{key}_ratio"] = _native_float(ratio)
        comparisons[f"{key}_delta"] = _native_float(delta)
        if ratio > maximum and delta > thresholds.geometry_delta_min:
            issues.append(issue)

    contour_path = "contour.contour_tightening_removed_fraction"
    candidate_contour = _metric(candidate, contour_path)
    baseline_contour = _metric(baseline, contour_path)
    contour_ratio = candidate_contour / max(baseline_contour, 1e-9)
    comparisons["contour_tightening_removed_fraction_ratio"] = _native_float(
        contour_ratio
    )
    comparisons["contour_tightening_removed_fraction_delta"] = _native_float(
        candidate_contour - baseline_contour
    )
    if (
        contour_ratio > thresholds.contour_tightening_ratio_max
        and candidate_contour > thresholds.contour_tightening_candidate_min
    ):
        issues.append("contour_fringe_expanded_vs_profile")

    texture_specs = (
        (
            "texture.coarse_gradient_p90",
            thresholds.coarse_gradient_ratio_max,
            "coarse_strokes_expanded_vs_profile",
        ),
        (
            "texture.structure_tensor_coherent_fraction",
            thresholds.coherent_fraction_ratio_max,
            "coherent_brush_texture_expanded_vs_profile",
        ),
        (
            "texture.fine_radial_brush_fraction",
            thresholds.radial_brush_ratio_max,
            "radial_brush_texture_expanded_vs_profile",
        ),
        (
            "texture.coherent_long_line_energy",
            thresholds.long_line_energy_ratio_max,
            "long_fur_energy_expanded_vs_profile",
        ),
    )
    for path, maximum, issue in texture_specs:
        candidate_value = _metric(candidate, path)
        baseline_value = _metric(baseline, path)
        ratio = candidate_value / max(baseline_value, 1e-9)
        delta = candidate_value - baseline_value
        key = path.replace(".", "_")
        comparisons[f"{key}_ratio"] = _native_float(ratio)
        comparisons[f"{key}_delta"] = _native_float(delta)
        if ratio > maximum and delta > thresholds.texture_delta_min:
            issues.append(issue)
    return comparisons, list(dict.fromkeys(issues))


def evaluate_fur_surface(
    candidate_path: str | Path,
    profile_baseline_path: str | Path | None = None,
    *,
    baseline_is_validated: bool = False,
    accept_absolute_contract: bool = False,
    thresholds: FurSurfaceThresholds | None = None,
) -> dict[str, Any]:
    """Audit a candidate, optionally relative to a validated profile baseline.

    ``acceptable`` is deliberately tri-state.  It is ``False`` when a defect is
    found, ``True`` for an in-contract candidate compared with an explicit
    validated clean baseline, and ``None`` when the image is merely diagnostic.
    Uploaded-identity routes may opt into the calibrated absolute contract: this
    avoids comparing a user's fur texture with pixels from an unrelated frozen
    Profile while preserving the same fail-closed absolute defect checks.
    """

    active = thresholds or FurSurfaceThresholds()
    candidate = analyze_fur_surface(candidate_path, thresholds=active)
    candidate_issues = _absolute_issues(candidate, active)
    baseline: dict[str, Any] | None = None
    baseline_issues: list[str] = []
    comparisons: dict[str, float] = {}
    relative_issues: list[str] = []
    if profile_baseline_path is not None:
        baseline = analyze_fur_surface(profile_baseline_path, thresholds=active)
        baseline_issues = _absolute_issues(baseline, active)
        comparisons, relative_issues = _relative_audit(
            candidate,
            baseline,
            active,
        )
    issues = list(dict.fromkeys([*candidate_issues, *relative_issues]))
    baseline_ready = bool(
        baseline is not None
        and baseline_is_validated
        and not baseline_issues
        and baseline["input"]["supported_contract"]
    )
    candidate_in_contract = bool(candidate["input"]["supported_contract"])
    if issues:
        acceptable: bool | None = False
        verdict = "rejected"
    elif baseline_ready and candidate_in_contract:
        acceptable = True
        verdict = "accepted_against_validated_profile_baseline"
    elif accept_absolute_contract and candidate_in_contract:
        acceptable = True
        verdict = "accepted_uploaded_identity_absolute_contract"
    else:
        acceptable = None
        verdict = "diagnostic_only_not_proven_acceptable"
    limitations: list[str] = []
    if baseline is None:
        limitations.append("no_profile_baseline")
    elif not baseline_is_validated:
        limitations.append("profile_baseline_not_explicitly_validated")
    if baseline_issues:
        limitations.append("profile_baseline_fails_current_absolute_thresholds")
    if not candidate_in_contract:
        limitations.append("candidate_used_opaque_background_fallback")
    return {
        "acceptable": acceptable,
        "verdict": verdict,
        "issues": issues,
        "candidate_absolute_issues": candidate_issues,
        "relative_issues": relative_issues,
        "baseline_issues": baseline_issues,
        "candidate": candidate,
        "baseline": baseline,
        "comparisons": comparisons,
        "thresholds": asdict(active),
        "policy": {
            "name": POLICY,
            "gate_version": GATE_VERSION,
            "threshold_version": THRESHOLD_VERSION,
            "production_integrated": True,
            "validated_baseline_required_for_acceptance": not bool(
                accept_absolute_contract
            ),
            "uploaded_identity_absolute_contract_enabled": bool(
                accept_absolute_contract
            ),
            "baseline_is_validated": bool(baseline_is_validated),
            "opaque_background_can_be_rejected_but_not_accepted": True,
            "limitations": limitations,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--validated-baseline", action="store_true")
    args = parser.parse_args()
    result = evaluate_fur_surface(
        args.candidate,
        args.baseline,
        baseline_is_validated=args.validated_baseline,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    if result["acceptable"] is True:
        return 0
    if result["acceptable"] is False:
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
