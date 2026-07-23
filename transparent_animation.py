from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image


BAYER_8X8 = np.array(
    [
        [0, 32, 8, 40, 2, 34, 10, 42],
        [48, 16, 56, 24, 50, 18, 58, 26],
        [12, 44, 4, 36, 14, 46, 6, 38],
        [60, 28, 52, 20, 62, 30, 54, 22],
        [3, 35, 11, 43, 1, 33, 9, 41],
        [51, 19, 59, 27, 49, 17, 57, 25],
        [15, 47, 7, 39, 13, 45, 5, 37],
        [63, 31, 55, 23, 61, 29, 53, 21],
    ],
    dtype=np.float32,
)


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

    # Remove green spill from fur edges without changing neutral/cream subject colors.
    non_green_limit = np.maximum(red, blue).astype(np.uint8)
    green_channel = rgb[:, :, 1]
    subject_support = soft_alpha > 0
    spill = subject_support & (green_channel > non_green_limit)
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
    height, width = soft_alpha.shape
    threshold = np.tile(
        ((BAYER_8X8 + 0.5) * (255.0 / 64.0)).astype(np.uint8),
        (math.ceil(height / 8), math.ceil(width / 8)),
    )[:height, :width]
    return np.where(soft_alpha > threshold, 255, 0).astype(np.uint8)


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
    size: int,
    frame_count: int,
    duration_ms: int,
) -> dict[str, object]:
    with Image.open(path) as image:
        if image.size != (size, size):
            raise RuntimeError(f"Animation has unexpected size {image.size}.")
        if image.n_frames != frame_count:
            raise RuntimeError(f"Animation has {image.n_frames} frames, expected {frame_count}.")
        durations: list[int] = []
        corners: list[int] = []
        for frame_index in range(image.n_frames):
            image.seek(frame_index)
            durations.append(int(image.info.get("duration", 0)))
            rgba = image.convert("RGBA")
            corners.extend(
                rgba.getpixel(point)[3]
                for point in ((0, 0), (size - 1, 0), (0, size - 1), (size - 1, size - 1))
            )
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
            "[gif][p]paletteuse=dither=sierra2_4a:alpha_threshold=128:diff_mode=rectangle[v]"
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
            apng_frames = [
                Image.open(soft_frames / f"frame_{index:03d}.png").convert("RGBA")
                for index in range(frame_count)
            ]
            try:
                apng_frames[0].save(
                    apng_temporary,
                    format="PNG",
                    save_all=True,
                    append_images=apng_frames[1:],
                    duration=int(round(1000 / fps)),
                    loop=0,
                    disposal=1,
                    blend=0,
                    optimize=True,
                )
            finally:
                for frame in apng_frames:
                    frame.close()
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
            "gif": gif_validation,
            "apng": apng_validation,
        }
    finally:
        gif_temporary.unlink(missing_ok=True)
        if apng_temporary:
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
