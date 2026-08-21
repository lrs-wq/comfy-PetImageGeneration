from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image

GIF_ALPHA_THRESHOLD = 144
GIF_ALPHA_SMOOTHING_SIGMA = 0.45
GIF_MIN_COMPONENT_FRACTION = 0.00003


def _run(command: list[str], stage_name: str, timeout: int = 900) -> None:
    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if process.returncode != 0:
        detail = (process.stderr or process.stdout or "unknown FFmpeg error").strip()
        raise RuntimeError(f"{stage_name} failed: {detail[-4000:]}")


def _selected_video_frames(source: Path, size: int, frame_count: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(source))
    source_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if source_frame_count < frame_count:
        capture.release()
        raise RuntimeError(
            f"Animation source has only {source_frame_count} frames; at least {frame_count} are required."
        )
    selected_indices = np.rint(np.linspace(0, source_frame_count - 1, frame_count)).astype(int)
    selected_lookup = {int(source_index): output_index for output_index, source_index in enumerate(selected_indices)}
    frames: list[np.ndarray | None] = [None] * frame_count
    source_index = 0
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        output_index = selected_lookup.get(source_index)
        if output_index is not None:
            if bgr.shape[1] != size or bgr.shape[0] != size:
                bgr = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_LANCZOS4)
            frames[output_index] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        source_index += 1
    capture.release()
    if any(frame is None for frame in frames):
        missing = [index for index, frame in enumerate(frames) if frame is None]
        raise RuntimeError(f"Could not read selected animation frames: {missing}")
    return [frame for frame in frames if frame is not None]


def _subject_mask(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    red = rgb[:, :, 0].astype(np.int16)
    green = rgb[:, :, 1].astype(np.int16)
    blue = rgb[:, :, 2].astype(np.int16)
    dominance = green - np.maximum(red, blue)

    # Keep a soft transition between subject and green screen for WebP/APNG.
    soft_alpha = np.clip((92.0 - dominance.astype(np.float32)) * (255.0 / 64.0), 0, 255).astype(np.uint8)
    hard_foreground = ((green <= 55) | (dominance <= 44)).astype(np.uint8)
    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        hard_foreground,
        connectivity=8,
    )
    if component_count <= 1:
        raise RuntimeError("No foreground subject found in animation frame.")
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    largest_area = int(stats[largest_label, cv2.CC_STAT_AREA])
    if largest_area < int(rgb.shape[0] * rgb.shape[1] * 0.01):
        raise RuntimeError(f"Foreground subject is too small: {largest_area} pixels.")

    # Keep the main cat plus a compact action prop (food/water bowl) near its lower body.
    # Requiring both lower-body placement and physical proximity excludes the floating
    # spirals/marks that the image model occasionally creates around the face.
    main_component = np.where(labels == largest_label, 255, 0).astype(np.uint8)
    main_x = int(stats[largest_label, cv2.CC_STAT_LEFT])
    main_y = int(stats[largest_label, cv2.CC_STAT_TOP])
    main_width = int(stats[largest_label, cv2.CC_STAT_WIDTH])
    main_height = int(stats[largest_label, cv2.CC_STAT_HEIGHT])
    main_bottom = main_y + main_height
    proximity = cv2.dilate(main_component, np.ones((51, 51), np.uint8), iterations=1)
    kept_labels = [largest_label]
    minimum_prop_area = max(64, int(largest_area * 0.0015))
    for label in range(1, component_count):
        if label == largest_label:
            continue
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < minimum_prop_area or area > int(largest_area * 0.45):
            continue
        component_mask = labels == label
        center_x, center_y = centroids[label]
        lower_body_region = center_y >= main_y + main_height * 0.40
        horizontally_near = (
            main_x - main_width * 0.20 <= center_x <= main_x + main_width * 1.20
        )
        vertically_near = center_y <= main_bottom + max(32, int(main_height * 0.10))
        touches_expanded_subject = bool(np.any(proximity[component_mask] > 0))
        if lower_body_region and horizontally_near and vertically_near and touches_expanded_subject:
            kept_labels.append(label)

    component = np.where(np.isin(labels, kept_labels), 255, 0).astype(np.uint8)

    # Green light reflected by pale fur can satisfy the chroma threshold even when
    # the pixel is inside the cat. Preserve small background-looking islands that
    # are completely enclosed by the subject silhouette. True background gaps
    # between legs remain connected to the image border and are not filled.
    inverse_component = (component == 0).astype(np.uint8)
    hole_count, hole_labels, hole_stats, _ = cv2.connectedComponentsWithStats(
        inverse_component,
        connectivity=8,
    )
    border_labels = set(np.unique(hole_labels[0, :]))
    border_labels.update(np.unique(hole_labels[-1, :]))
    border_labels.update(np.unique(hole_labels[:, 0]))
    border_labels.update(np.unique(hole_labels[:, -1]))
    enclosed_holes = np.zeros(component.shape, dtype=np.uint8)
    maximum_hole_area = max(96, int(largest_area * 0.025))
    for label in range(1, hole_count):
        if label in border_labels:
            continue
        area = int(hole_stats[label, cv2.CC_STAT_AREA])
        if area <= maximum_hole_area:
            enclosed_holes[hole_labels == label] = 255
    component = np.maximum(component, enclosed_holes)
    support = cv2.dilate(component, np.ones((5, 5), np.uint8), iterations=1)
    soft_alpha = np.where(enclosed_holes > 0, 255, soft_alpha).astype(np.uint8)
    soft_alpha = np.where(support > 0, soft_alpha, 0).astype(np.uint8)
    soft_alpha = cv2.GaussianBlur(soft_alpha, (0, 0), sigmaX=0.55, sigmaY=0.55)
    soft_alpha[soft_alpha < 3] = 0
    soft_alpha[soft_alpha > 252] = 255

    # Remove green spill only in the narrow matte boundary.  Clamping every opaque
    # foreground pixel recolors legitimate green eyes and other identity accents.
    # Interior spill remains visible to the downstream palette gate and therefore
    # cannot be silently hidden by background removal.
    non_green_limit = np.maximum(red, blue).astype(np.uint8)
    green_channel = rgb[:, :, 1]
    subject_support = soft_alpha > 0
    interior_distance = cv2.distanceTransform(
        (component > 0).astype(np.uint8),
        cv2.DIST_L2,
        5,
    )
    matte_boundary = (soft_alpha < 248) | (interior_distance <= 6.0)
    spill = (
        subject_support
        & matte_boundary
        & (green_channel > non_green_limit)
    )
    green_channel[spill] = non_green_limit[spill]
    rgb[~subject_support] = 0
    return rgb, soft_alpha


def _anchor(alpha: np.ndarray) -> tuple[float, float, float, tuple[int, int, int, int]]:
    ys, xs = np.where(alpha >= 128)
    if not len(xs):
        raise RuntimeError("Transparent frame has no opaque subject pixels.")
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    height = y1 - y0
    body_band = (ys >= y0 + int(height * 0.35)) & (ys <= y0 + int(height * 0.82))
    anchor_x = float(np.median(xs[body_band])) if body_band.any() else float(np.median(xs))
    anchor_y = float(np.percentile(ys, 98))
    area = float(len(xs))
    return anchor_x, anchor_y, area, (x0, y0, x1, y1)


def _warp_subject(
    rgb: np.ndarray,
    alpha: np.ndarray,
    anchor_x: float,
    anchor_y: float,
    target_x: float,
    target_y: float,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.array(
        [
            [scale, 0.0, target_x - scale * anchor_x],
            [0.0, scale, target_y - scale * anchor_y],
        ],
        dtype=np.float32,
    )
    size = (rgb.shape[1], rgb.shape[0])
    stabilized_rgb = cv2.warpAffine(
        rgb,
        matrix,
        size,
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    stabilized_alpha = cv2.warpAffine(
        alpha,
        matrix,
        size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return stabilized_rgb, stabilized_alpha


def _temporal_median(alphas: list[np.ndarray]) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    for index in range(len(alphas)):
        first = max(0, index - 1)
        last = min(len(alphas), index + 2)
        result.append(np.median(np.stack(alphas[first:last], axis=0), axis=0).astype(np.uint8))
    return result


def _tracked_displacements(
    rgb_frames: list[np.ndarray],
    alpha_frames: list[np.ndarray],
    *,
    region: str,
) -> np.ndarray:
    reference_alpha = alpha_frames[0]
    ys, xs = np.where(reference_alpha >= 128)
    if not len(xs):
        raise RuntimeError("Cannot track an empty transparent subject.")
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    width = x1 - x0
    height = y1 - y0
    if region == "torso":
        rx0 = x0 + int(width * 0.14)
        rx1 = x1 - int(width * 0.14)
        ry0 = y0 + int(height * 0.34)
        ry1 = y0 + int(height * 0.82)
    elif region == "head":
        rx0 = x0 + int(width * 0.05)
        rx1 = x1 - int(width * 0.05)
        ry0 = y0
        ry1 = y0 + int(height * 0.43)
    else:
        raise ValueError(f"Unsupported tracking region {region}.")

    reference_gray = cv2.cvtColor(rgb_frames[0], cv2.COLOR_RGB2GRAY)
    feature_mask = np.zeros_like(reference_gray)
    feature_mask[ry0:ry1, rx0:rx1] = 255
    feature_mask[reference_alpha < 128] = 0
    reference_points = cv2.goodFeaturesToTrack(
        reference_gray,
        maxCorners=240,
        qualityLevel=0.01,
        minDistance=3,
        mask=feature_mask,
        blockSize=5,
    )
    if reference_points is None or len(reference_points) < 8:
        raise RuntimeError(f"Not enough visual features to stabilize the {region} region.")

    displacements: list[np.ndarray] = [np.zeros(2, dtype=np.float32)]
    for rgb in rgb_frames[1:]:
        current_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        current_points, status, _ = cv2.calcOpticalFlowPyrLK(
            reference_gray,
            current_gray,
            reference_points,
            None,
            winSize=(31, 31),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.001),
        )
        if current_points is None or status is None:
            displacements.append(displacements[-1].copy())
            continue
        valid = status[:, 0] == 1
        first = reference_points[valid].reshape(-1, 2)
        last = current_points[valid].reshape(-1, 2)
        if len(first) < 8:
            displacements.append(displacements[-1].copy())
            continue
        delta = last - first
        median = np.median(delta, axis=0)
        distances = np.linalg.norm(delta - median, axis=1)
        cutoff = float(np.percentile(distances, 70))
        robust = delta[distances <= cutoff]
        displacement = np.median(robust, axis=0) if len(robust) else median
        displacement = np.clip(displacement, -20.0, 20.0).astype(np.float32)
        displacements.append(displacement)
    return np.stack(displacements, axis=0)


def _trajectory_metrics(displacements: np.ndarray) -> dict[str, float]:
    steps = np.linalg.norm(np.diff(displacements, axis=0), axis=1)
    return {
        "x_range": float(np.ptp(displacements[:, 0])),
        "y_range": float(np.ptp(displacements[:, 1])),
        "max_step": float(np.max(steps)) if len(steps) else 0.0,
        "p95_step": float(np.percentile(steps, 95)) if len(steps) else 0.0,
    }


def _smoothed_tracking_correction(
    displacements: np.ndarray,
    *,
    window: int = 7,
    max_step: float = 1.25,
) -> np.ndarray:
    """Remove optical-flow outliers before using them as frame warps.

    A moving tongue, leg, or nearby bowl can briefly dominate the torso feature set.
    Applying that raw measurement would turn a local action into a whole-subject jump.
    Median filtering rejects the outlier and the velocity limit guarantees that any
    remaining correction evolves smoothly from frame to frame.
    """
    if len(displacements) <= 1:
        return displacements.copy()
    radius = max(1, window // 2)
    padded = np.pad(displacements, ((radius, radius), (0, 0)), mode="edge")
    filtered = np.stack(
        [np.median(padded[index : index + window], axis=0) for index in range(len(displacements))],
        axis=0,
    ).astype(np.float32)
    filtered[0] = 0.0

    limited = np.zeros_like(filtered)
    for index in range(1, len(filtered)):
        delta = filtered[index] - limited[index - 1]
        distance = float(np.linalg.norm(delta))
        if distance > max_step:
            delta *= max_step / distance
        limited[index] = limited[index - 1] + delta
    return limited


def _stabilize_visual_core(
    rgb_frames: list[np.ndarray],
    alpha_frames: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, object]]:
    before = _tracked_displacements(rgb_frames, alpha_frames, region="torso")
    correction = _smoothed_tracking_correction(before)
    stabilized_rgb: list[np.ndarray] = []
    stabilized_alpha: list[np.ndarray] = []
    size = (rgb_frames[0].shape[1], rgb_frames[0].shape[0])
    for rgb, alpha, (dx, dy) in zip(rgb_frames, alpha_frames, correction, strict=True):
        matrix = np.array([[1.0, 0.0, -float(dx)], [0.0, 1.0, -float(dy)]], dtype=np.float32)
        stabilized_rgb.append(
            cv2.warpAffine(
                rgb,
                matrix,
                size,
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
        )
        stabilized_alpha.append(
            cv2.warpAffine(
                alpha,
                matrix,
                size,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        )
    after = _tracked_displacements(stabilized_rgb, stabilized_alpha, region="torso")
    return stabilized_rgb, stabilized_alpha, {
        "region": "torso",
        "before": _trajectory_metrics(before),
        "applied_correction": _trajectory_metrics(correction),
        "correction_filter": {
            "type": "temporal_median_with_velocity_limit",
            "window": 7,
            "max_step": 1.25,
        },
        "after": _trajectory_metrics(after),
    }


def _gif_alpha(soft_alpha: np.ndarray) -> np.ndarray:
    """Create a deliberate hard matte for GIF without ordered-alpha dust.

    GIF has one transparent palette entry and cannot represent the soft alpha
    used by APNG.  Ordered dithering approximates that coverage with isolated
    opaque dots, which becomes a visible ant-line around pale fur.  Use one
    stable contour instead, then remove only tiny detached components and
    pinholes.  Fine semi-transparent hairs remain available in the APNG output.
    """
    if soft_alpha.ndim != 2:
        raise ValueError("GIF alpha must be a two-dimensional array.")
    alpha = np.asarray(soft_alpha, dtype=np.uint8)
    smoothed = cv2.GaussianBlur(
        alpha,
        (0, 0),
        sigmaX=GIF_ALPHA_SMOOTHING_SIGMA,
        sigmaY=GIF_ALPHA_SMOOTHING_SIGMA,
    )
    hard = smoothed >= GIF_ALPHA_THRESHOLD
    minimum_area = max(
        4,
        int(round(alpha.size * GIF_MIN_COMPONENT_FRACTION)),
    )

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        hard.astype(np.uint8),
        connectivity=8,
    )
    cleaned = np.zeros_like(hard)
    for label in range(1, component_count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= minimum_area:
            cleaned[labels == label] = True

    # Fill only tiny enclosed pinholes. Background regions connected to the
    # canvas border (including spaces between legs) must remain transparent.
    background_count, background_labels, background_stats, _ = (
        cv2.connectedComponentsWithStats(
            (~cleaned).astype(np.uint8),
            connectivity=8,
        )
    )
    border_labels = set(np.unique(background_labels[0, :]))
    border_labels.update(np.unique(background_labels[-1, :]))
    border_labels.update(np.unique(background_labels[:, 0]))
    border_labels.update(np.unique(background_labels[:, -1]))
    for label in range(1, background_count):
        if (
            label not in border_labels
            and int(background_stats[label, cv2.CC_STAT_AREA]) < minimum_area
        ):
            cleaned[background_labels == label] = True
    return cleaned.astype(np.uint8) * 255


def _binary_alpha_edge_metrics(alpha: np.ndarray) -> dict[str, float | int]:
    """Measure detached dust and unstable one-pixel patterns in a hard matte."""
    foreground = np.asarray(alpha) >= 128
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(
        foreground.astype(np.uint8),
        connectivity=8,
    )
    minimum_area = max(
        4,
        int(round(foreground.size * GIF_MIN_COMPONENT_FRACTION)),
    )
    tiny_components = sum(
        1
        for label in range(1, component_count)
        if int(stats[label, cv2.CC_STAT_AREA]) < minimum_area
    )
    eroded = cv2.erode(
        foreground.astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    ).astype(bool)
    boundary = foreground & ~eroded
    neighbor_count = cv2.filter2D(
        foreground.astype(np.uint8),
        ddepth=cv2.CV_16U,
        kernel=np.ones((3, 3), dtype=np.uint8),
        borderType=cv2.BORDER_CONSTANT,
    )
    sparse_boundary_pixels = int(
        np.count_nonzero(boundary & (neighbor_count <= 3))
    )
    boundary_pixels = int(np.count_nonzero(boundary))

    top_left = foreground[:-1, :-1]
    top_right = foreground[:-1, 1:]
    bottom_left = foreground[1:, :-1]
    bottom_right = foreground[1:, 1:]
    checkerboards = (
        top_left & bottom_right & ~top_right & ~bottom_left
    ) | (
        top_right & bottom_left & ~top_left & ~bottom_right
    )
    checkerboard_blocks = int(np.count_nonzero(checkerboards))
    return {
        "foreground_components": max(0, component_count - 1),
        "tiny_foreground_components": tiny_components,
        "boundary_pixels": boundary_pixels,
        "sparse_boundary_pixels": sparse_boundary_pixels,
        "sparse_boundary_fraction": float(sparse_boundary_pixels)
        / float(max(1, boundary_pixels)),
        "checkerboard_2x2_blocks": checkerboard_blocks,
        "checkerboard_fraction": float(checkerboard_blocks)
        / float(max(1, boundary_pixels)),
    }


def _encode_soft_alpha_apng(
    frame_paths: Iterable[Path],
    target: Path,
    *,
    fps: int,
) -> None:
    """Encode RGBA frame files as an animated PNG with full 8-bit alpha."""
    frames: list[Image.Image] = []
    for path in frame_paths:
        with Image.open(path) as source:
            frames.append(source.convert("RGBA"))
    if not frames:
        raise RuntimeError("Cannot encode an APNG without frames.")
    try:
        frames[0].save(
            target,
            format="PNG",
            save_all=True,
            append_images=frames[1:],
            duration=int(round(1000 / fps)),
            loop=0,
            disposal=1,
            blend=0,
            optimize=True,
        )
    finally:
        for frame in frames:
            frame.close()


def _save_rgba(path: Path, rgb: np.ndarray, alpha: np.ndarray) -> None:
    bgra = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGRA)
    bgra[:, :, 3] = alpha
    if not cv2.imwrite(str(path), bgra):
        raise RuntimeError(f"Could not write transparent animation frame {path.name}.")


def build_transparent_cutout(
    source: Path,
    target: Path,
    *,
    require_clean_background: bool = False,
    max_foreground_fraction: float = 0.85,
) -> dict[str, object]:
    bgr = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Could not read chroma-key image {source}.")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb, alpha = _subject_mask(rgb)
    foreground_fraction = float(np.count_nonzero(alpha >= 128)) / float(alpha.size)
    corners = [
        int(alpha[0, 0]),
        int(alpha[0, -1]),
        int(alpha[-1, 0]),
        int(alpha[-1, -1]),
    ]
    if require_clean_background and (
        foreground_fraction > max_foreground_fraction or any(value > 0 for value in corners)
    ):
        raise RuntimeError(
            "Chroma background removal is unreliable: the generated image did not use a clean green background."
        )

    temporary = target.with_name(f".{target.stem}.{uuid4().hex}.png")
    try:
        _save_rgba(temporary, rgb, alpha)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "size": [int(alpha.shape[1]), int(alpha.shape[0])],
        "foreground_fraction": foreground_fraction,
        "corners": corners,
        "soft_alpha": True,
    }


def _animation_metrics(alphas: list[np.ndarray]) -> dict[str, object]:
    anchors = [_anchor(alpha) for alpha in alphas]
    array = np.array([[anchor[0], anchor[1], anchor[2]] for anchor in anchors])
    bboxes = [anchor[3] for anchor in anchors]
    return {
        "anchor_x_range": float(np.ptp(array[:, 0])),
        "anchor_y_range": float(np.ptp(array[:, 1])),
        "area_range": float(np.ptp(array[:, 2])),
        "max_anchor_x_step": float(np.max(np.abs(np.diff(array[:, 0])))),
        "max_anchor_y_step": float(np.max(np.abs(np.diff(array[:, 1])))),
        "bbox_first": list(bboxes[0]),
        "bbox_last": list(bboxes[-1]),
    }


def validate_animation(
    path: Path,
    *,
    size: int | tuple[int, int],
    frame_count: int,
    duration_ms: int,
) -> dict[str, object]:
    expected_size = (size, size) if isinstance(size, int) else size
    with Image.open(path) as image:
        if image.size != expected_size:
            raise RuntimeError(f"Animation has unexpected size {image.size}.")
        if image.n_frames != frame_count:
            raise RuntimeError(f"Animation has {image.n_frames} frames, expected {frame_count}.")
        durations: list[int] = []
        corners: list[int] = []
        border_alpha_max = 0
        soft_alpha_frames = 0
        soft_alpha_pixels = 0
        edge_metrics: list[dict[str, float | int]] = []
        for frame_index in range(image.n_frames):
            image.seek(frame_index)
            durations.append(int(image.info.get("duration", 0)))
            rgba = np.asarray(image.convert("RGBA"))
            alpha = rgba[:, :, 3]
            corners.extend(
                int(alpha[y, x])
                for x, y in (
                    (0, 0),
                    (expected_size[0] - 1, 0),
                    (0, expected_size[1] - 1),
                    (expected_size[0] - 1, expected_size[1] - 1),
                )
            )
            border_alpha_max = max(
                border_alpha_max,
                int(alpha[:2, :].max()),
                int(alpha[-2:, :].max()),
                int(alpha[:, :2].max()),
                int(alpha[:, -2:].max()),
            )
            frame_soft_pixels = int(
                np.count_nonzero((alpha > 0) & (alpha < 255))
            )
            soft_alpha_pixels += frame_soft_pixels
            soft_alpha_frames += int(frame_soft_pixels > 0)
            edge_metrics.append(_binary_alpha_edge_metrics(alpha))
        actual_duration_ms = sum(durations)
        if actual_duration_ms != duration_ms:
            raise RuntimeError(f"Animation duration is {actual_duration_ms}ms, expected {duration_ms}ms.")
        if any(alpha != 0 for alpha in corners):
            raise RuntimeError("Animation has an opaque corner in at least one frame.")
        return {
            "size": list(image.size),
            "frames": image.n_frames,
            "duration_ms": actual_duration_ms,
            "frame_durations_ms": sorted(set(durations)),
            "all_corners_transparent": True,
            "full_2px_border_transparent": border_alpha_max == 0,
            "soft_alpha_frames": soft_alpha_frames,
            "soft_alpha_pixels": soft_alpha_pixels,
            "edge_quality": {
                "tiny_foreground_components_max": max(
                    int(item["tiny_foreground_components"])
                    for item in edge_metrics
                ),
                "sparse_boundary_fraction_max": max(
                    float(item["sparse_boundary_fraction"])
                    for item in edge_metrics
                ),
                "checkerboard_fraction_max": max(
                    float(item["checkerboard_fraction"])
                    for item in edge_metrics
                ),
            },
            "bytes": path.stat().st_size,
        }


def build_transparent_animations(
    source: Path,
    gif_target: Path,
    apng_target: Path | None = None,
    *,
    ffmpeg_bin: str = "ffmpeg",
    size: int = 512,
    fps: int = 20,
    frame_count: int = 100,
    duration_seconds: float = 5.0,
) -> dict[str, object]:
    frames_root = gif_target.parent / f".{gif_target.stem}.{uuid4().hex}.frames"
    gif_frames = frames_root / "gif"
    soft_frames = frames_root / "soft"
    gif_temporary = gif_target.with_name(f".{gif_target.stem}.{uuid4().hex}.gif")
    apng_temporary = (
        apng_target.with_name(f".{apng_target.stem}.{uuid4().hex}.png") if apng_target else None
    )
    gif_frames.mkdir(parents=True, exist_ok=False)
    soft_frames.mkdir(parents=True, exist_ok=False)
    try:
        source_frames = _selected_video_frames(source, size, frame_count)
        rgb_frames: list[np.ndarray] = []
        alpha_frames: list[np.ndarray] = []
        anchors: list[tuple[float, float, float, tuple[int, int, int, int]]] = []
        for frame in source_frames:
            rgb, alpha = _subject_mask(frame.copy())
            rgb_frames.append(rgb)
            alpha_frames.append(alpha)
            anchors.append(_anchor(alpha))

        before_metrics = _animation_metrics(alpha_frames)
        target_x = float(np.median([anchor[0] for anchor in anchors]))
        target_y = float(np.median([anchor[1] for anchor in anchors]))

        stabilized_rgb: list[np.ndarray] = []
        stabilized_alpha: list[np.ndarray] = []
        for rgb, alpha, (anchor_x, anchor_y, area, _) in zip(
            rgb_frames,
            alpha_frames,
            anchors,
            strict=True,
        ):
            # Per-frame scale correction creates a visible zoom pulse on large cartoon eyes and glasses.
            # Keep scale fixed and correct only the root position here; content tracking runs below.
            scale = 1.0
            warped_rgb, warped_alpha = _warp_subject(
                rgb,
                alpha,
                anchor_x,
                anchor_y,
                target_x,
                target_y,
                scale,
            )
            stabilized_rgb.append(warped_rgb)
            stabilized_alpha.append(warped_alpha)

        stabilized_alpha = _temporal_median(stabilized_alpha)
        stabilized_rgb, stabilized_alpha, visual_core_metrics = _stabilize_visual_core(
            stabilized_rgb,
            stabilized_alpha,
        )
        head_lock_metrics = {
            "mode": "disabled",
            "reference_frame_pixel_replacement": False,
            "locked_pixels": 0,
        }
        after_metrics = _animation_metrics(stabilized_alpha)

        for index, (rgb, soft_alpha) in enumerate(
            zip(stabilized_rgb, stabilized_alpha, strict=True)
        ):
            _save_rgba(soft_frames / f"frame_{index:03d}.png", rgb, soft_alpha)
            _save_rgba(gif_frames / f"frame_{index:03d}.png", rgb, _gif_alpha(soft_alpha))

        gif_filter = (
            "[0:v]format=rgba,split[gif][pal];"
            "[pal]palettegen=reserve_transparent=1:transparency_color=0x000000:stats_mode=diff[p];"
            "[gif][p]paletteuse=dither=none:alpha_threshold=128:diff_mode=rectangle[v]"
        )
        final_delay_centiseconds = int(round(100 / fps))
        _run(
            [
                ffmpeg_bin,
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                str(fps),
                "-i",
                str(gif_frames / "frame_%03d.png"),
                "-filter_complex",
                gif_filter,
                "-map",
                "[v]",
                "-frames:v",
                str(frame_count),
                "-loop",
                "0",
                "-final_delay",
                str(final_delay_centiseconds),
                str(gif_temporary),
            ],
            "Stabilized transparent GIF encoding",
        )
        gif_validation = validate_animation(
            gif_temporary,
            size=size,
            frame_count=frame_count,
            duration_ms=int(duration_seconds * 1000),
        )
        os.replace(gif_temporary, gif_target)

        apng_validation: dict[str, object] | None = None
        if apng_target and apng_temporary:
            _encode_soft_alpha_apng(
                (
                    soft_frames / f"frame_{index:03d}.png"
                    for index in range(frame_count)
                ),
                apng_temporary,
                fps=fps,
            )
            apng_validation = validate_animation(
                apng_temporary,
                size=size,
                frame_count=frame_count,
                duration_ms=int(duration_seconds * 1000),
            )
            os.replace(apng_temporary, apng_target)

        return {
            "before": before_metrics,
            "after": after_metrics,
            "visual_core_stabilization": visual_core_metrics,
            "head_lock": head_lock_metrics,
            "gif_edge": {
                "policy": "clean_hard_alpha_v1",
                "ordered_alpha_dither": False,
                "alpha_threshold": GIF_ALPHA_THRESHOLD,
                "alpha_smoothing_sigma": GIF_ALPHA_SMOOTHING_SIGMA,
                "palette_dither": "none",
            },
            "gif": gif_validation,
            "apng": apng_validation,
        }
    finally:
        gif_temporary.unlink(missing_ok=True)
        if apng_temporary:
            apng_temporary.unlink(missing_ok=True)
        shutil.rmtree(frames_root, ignore_errors=True)


def build_standard_direct_outputs(
    source: Path,
    white_mp4_target: Path,
    gif_target: Path | None = None,
    apng_target: Path | None = None,
    *,
    ffmpeg_bin: str = "ffmpeg",
    size: int = 512,
    source_fps: int = 16,
    source_frame_count: int = 81,
    gif_fps: int = 20,
    gif_frame_count: int = 100,
) -> dict[str, object]:
    """Chroma-matte an approved standard without spatially changing its action.

    The white MP4 and optional transparent animations are derived from the exact
    same per-source-frame RGB/alpha pair. Only the animation timeline is
    resampled; there is no anchoring, stabilization, scaling, optical flow, or
    generative redraw.
    """

    frames_root = white_mp4_target.parent / (
        f".{white_mp4_target.stem}.{uuid4().hex}.standard-direct.frames"
    )
    white_frames = frames_root / "white"
    rgba_frames = frames_root / "rgba"
    gif_frames = frames_root / "gif"
    mp4_temporary = white_mp4_target.with_name(
        f".{white_mp4_target.stem}.{uuid4().hex}.mp4"
    )
    gif_temporary = (
        gif_target.with_name(f".{gif_target.stem}.{uuid4().hex}.gif")
        if gif_target is not None
        else None
    )
    apng_temporary = (
        apng_target.with_name(f".{apng_target.stem}.{uuid4().hex}.png")
        if apng_target is not None
        else None
    )
    if apng_target is not None:
        apng_target.unlink(missing_ok=True)
    white_frames.mkdir(parents=True, exist_ok=False)
    rgba_frames.mkdir(parents=True, exist_ok=False)
    if gif_target is not None:
        gif_frames.mkdir(parents=True, exist_ok=False)
    try:
        source_frames = _selected_video_frames(source, size, source_frame_count)
        alpha_foreground_fractions: list[float] = []
        matte_frames: list[tuple[np.ndarray, np.ndarray]] = []
        for index, frame in enumerate(source_frames):
            rgb, alpha = _subject_mask(frame.copy())
            matte_frames.append((rgb, alpha))
            alpha_float = alpha.astype(np.float32)[:, :, None] / 255.0
            white = np.clip(
                rgb.astype(np.float32) * alpha_float
                + 255.0 * (1.0 - alpha_float),
                0,
                255,
            ).astype(np.uint8)
            Image.fromarray(white, mode="RGB").save(
                white_frames / f"frame_{index:03d}.png"
            )
            _save_rgba(rgba_frames / f"frame_{index:03d}.png", rgb, alpha)
            alpha_foreground_fractions.append(
                float(np.count_nonzero(alpha >= 128)) / float(alpha.size)
            )

        _run(
            [
                ffmpeg_bin,
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                str(source_fps),
                "-i",
                str(white_frames / "frame_%03d.png"),
                "-frames:v",
                str(source_frame_count),
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(mp4_temporary),
            ],
            "Standard-direct white-background MP4 encoding",
        )
        os.replace(mp4_temporary, white_mp4_target)

        gif_validation: dict[str, object] | None = None
        apng_validation: dict[str, object] | None = None
        apng_error: str | None = None
        sampled_indices: list[int] | None = None
        if gif_target is not None or apng_target is not None:
            sampled_indices = np.rint(
                np.linspace(0, source_frame_count - 1, gif_frame_count)
            ).astype(int).tolist()

        if (
            gif_target is not None
            and gif_temporary is not None
            and sampled_indices is not None
        ):
            for output_index, source_index in enumerate(sampled_indices):
                rgb, alpha = matte_frames[source_index]
                _save_rgba(
                    gif_frames / f"frame_{output_index:03d}.png",
                    rgb,
                    _gif_alpha(alpha),
                )
            gif_filter = (
                "[0:v]format=rgba,split[gif][pal];"
                "[pal]palettegen=reserve_transparent=1:"
                "transparency_color=0x000000:stats_mode=diff[p];"
                "[gif][p]paletteuse=dither=none:"
                "alpha_threshold=128:diff_mode=rectangle[v]"
            )
            _run(
                [
                    ffmpeg_bin,
                    "-y",
                    "-loglevel",
                    "error",
                    "-framerate",
                    str(gif_fps),
                    "-i",
                    str(gif_frames / "frame_%03d.png"),
                    "-filter_complex",
                    gif_filter,
                    "-map",
                    "[v]",
                    "-frames:v",
                    str(gif_frame_count),
                    "-loop",
                    "0",
                    "-final_delay",
                    str(int(round(100 / gif_fps))),
                    str(gif_temporary),
                ],
                "Standard-direct transparent GIF encoding",
            )
            gif_validation = validate_animation(
                gif_temporary,
                size=size,
                frame_count=gif_frame_count,
                duration_ms=int(round(gif_frame_count * 1000 / gif_fps)),
            )
            os.replace(gif_temporary, gif_target)

        if (
            apng_target is not None
            and apng_temporary is not None
            and sampled_indices is not None
        ):
            try:
                _encode_soft_alpha_apng(
                    (
                        rgba_frames / f"frame_{source_index:03d}.png"
                        for source_index in sampled_indices
                    ),
                    apng_temporary,
                    fps=gif_fps,
                )
                apng_validation = validate_animation(
                    apng_temporary,
                    size=size,
                    frame_count=gif_frame_count,
                    duration_ms=int(
                        round(gif_frame_count * 1000 / gif_fps)
                    ),
                )
                os.replace(apng_temporary, apng_target)
            except Exception as exc:
                apng_error = f"{type(exc).__name__}: {exc}"
                apng_temporary.unlink(missing_ok=True)

        return {
            "policy": "shared_per_frame_chroma_matte_no_spatial_transform_v1",
            "source_frames": source_frame_count,
            "source_fps": source_fps,
            "size": [size, size],
            "white_mp4": {
                "frames": source_frame_count,
                "fps": source_fps,
                "duration_seconds": source_frame_count / source_fps,
                "bytes": white_mp4_target.stat().st_size,
            },
            "transparent_gif": gif_validation,
            "transparent_apng": apng_validation,
            "transparent_apng_error": apng_error,
            "gif_edge": {
                "policy": "clean_hard_alpha_v1",
                "ordered_alpha_dither": False,
                "alpha_threshold": GIF_ALPHA_THRESHOLD,
                "alpha_smoothing_sigma": GIF_ALPHA_SMOOTHING_SIGMA,
                "palette_dither": "none",
            },
            "gif_sampled_source_indices": sampled_indices,
            "foreground_fraction": {
                "minimum": min(alpha_foreground_fractions),
                "maximum": max(alpha_foreground_fractions),
            },
            "foreground_rgb_redrawn": False,
            "spatial_stabilization": False,
            "spatial_scaling": False,
            "spatial_translation": False,
            "temporal_resampling": (
                "transparent_animation_nearest_source_frame"
                if sampled_indices is not None
                else False
            ),
        }
    finally:
        mp4_temporary.unlink(missing_ok=True)
        if gif_temporary is not None:
            gif_temporary.unlink(missing_ok=True)
        if apng_temporary is not None:
            apng_temporary.unlink(missing_ok=True)
        shutil.rmtree(frames_root, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build stabilized transparent GIF and soft-alpha APNG animations.")
    parser.add_argument("source", type=Path)
    parser.add_argument("gif_target", type=Path)
    parser.add_argument("--apng-target", type=Path)
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = build_transparent_animations(
        args.source,
        args.gif_target,
        args.apng_target,
        ffmpeg_bin=args.ffmpeg_bin,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
