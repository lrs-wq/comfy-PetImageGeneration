#!/usr/bin/env python3
"""Build appearance-neutral white-maquette driving videos.

The conversion is deterministic: it preserves every source-frame silhouette
and action timing while replacing only the cat surface with neutral white clay
shading. Eating keeps the detected bowl/food region unchanged. The generated
assets are candidates and are not activated by this script.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from src.comfy_pet_pipeline.pipeline.subject_replacement import (
    encode_video,
    make_grooming_geometry_guide,
    make_eating_prop_exclusion_masks,
    read_animation_frames,
    sha256_file,
)


POLICY = "appearance_neutral_white_maquette_driver_v5"
GREEN = np.asarray((0, 255, 0), dtype=np.float32)
DEFAULT_SOURCE_ROOT = Path(
    "/data/comfy-pet-pipeline/profile_assets/sleek_tabby_cat_v1/"
    "driving/20260731-main-case-v1"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data/comfy-pet-pipeline/profile_assets/sleek_tabby_cat_v1/"
    "driving/20260808-white-maquette-v5"
)
SOURCE_FILENAMES = {
    "eating": "eating_driver_512x512_81f_16fps.mp4",
    "grooming": "grooming_driver_512x512_81f_16fps.mp4",
}
OUTPUT_FILENAMES = {
    "eating": "eating_white_model_driver_512x512_81f_16fps.mp4",
    "grooming": "grooming_white_model_driver_512x512_81f_16fps.mp4",
}


def _binary_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, columns = np.nonzero(mask)
    if not columns.size:
        raise ValueError("white_model_driver_empty_cat_mask")
    return (
        int(columns.min()),
        int(rows.min()),
        int(columns.max()) + 1,
        int(rows.max()) + 1,
    )


def _cat_and_prop_masks(
    frames: list[Image.Image],
    *,
    action: str,
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, Any] | None]:
    foreground_masks = [
        np.asarray(frame.convert("RGBA"), dtype=np.uint8)[:, :, 3] >= 16
        for frame in frames
    ]
    if action == "eating":
        prop_images, prop_metadata = make_eating_prop_exclusion_masks(frames)
        prop_masks = [
            np.asarray(mask, dtype=np.uint8) >= 128
            for mask in prop_images
        ]
    elif action == "grooming":
        prop_masks = [np.zeros_like(mask) for mask in foreground_masks]
        prop_metadata = None
    else:
        raise ValueError(f"Unsupported white-model action: {action}")

    cat_masks = [
        foreground & ~prop
        for foreground, prop in zip(
            foreground_masks,
            prop_masks,
            strict=True,
        )
    ]
    if any(int(np.count_nonzero(mask)) < 4096 for mask in cat_masks):
        raise ValueError("white_model_driver_cat_mask_too_small")
    return cat_masks, prop_masks, prop_metadata


def _white_maquette_distance_field(cat_mask: np.ndarray) -> np.ndarray:
    """Return a source-appearance-free soft volume for one silhouette."""
    distance = cv2.distanceTransform(
        cat_mask.astype(np.uint8),
        cv2.DIST_L2,
        5,
    )
    values = distance[cat_mask]
    if values.size < 64:
        raise ValueError("white_model_driver_empty_cat_mask")
    distance_p95 = max(1.0, float(np.percentile(values, 95.0)))
    return np.sqrt(np.clip(distance / distance_p95, 0.0, 1.0))


def _normalization_inverse(
    source_bbox: list[int],
    *,
    canvas_size: tuple[int, int] = (512, 512),
    scale_fraction: float = 0.88,
) -> tuple[float, float, float]:
    """Invert the exact crop/contain transform used by the geometry audit."""
    left, top, right, bottom = source_bbox
    crop_width = right - left
    crop_height = bottom - top
    if crop_width <= 0 or crop_height <= 0:
        raise ValueError("white_model_grooming_invalid_source_bbox")
    max_width = max(1, int(canvas_size[0] * scale_fraction))
    max_height = max(1, int(canvas_size[1] * scale_fraction))
    factor = min(max_width / crop_width, max_height / crop_height)
    resized_width = max(1, int(round(crop_width * factor)))
    resized_height = max(1, int(round(crop_height * factor)))
    offset_x = (canvas_size[0] - resized_width) // 2
    offset_y = (canvas_size[1] - resized_height) // 2
    return factor, float(offset_x), float(offset_y)


def _point_from_normalized_guide(
    point: list[int],
    source_bbox: list[int],
    transform: tuple[float, float, float],
) -> np.ndarray:
    factor, offset_x, offset_y = transform
    return np.asarray(
        (
            (float(point[0]) - offset_x) / factor + source_bbox[0],
            (float(point[1]) - offset_y) / factor + source_bbox[1],
        ),
        dtype=np.float32,
    )


def _synthetic_grooming_foreleg_mask(
    cat_mask: np.ndarray,
    geometry_metadata: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build one neutral foreleg volume from verified geometry anchors only."""
    relation = geometry_metadata.get("raised_paw_muzzle_relation")
    source_bbox = geometry_metadata.get("source_union_bbox")
    if not isinstance(relation, dict) or relation.get("passed") is not True:
        raise ValueError("white_model_grooming_missing_verified_relation")
    if not isinstance(source_bbox, list) or len(source_bbox) != 4:
        raise ValueError("white_model_grooming_missing_source_bbox")
    shoulder_value = relation.get("shoulder_anchor")
    paw_value = relation.get("paw_center")
    foreleg_bbox = relation.get("raised_foreleg_region_bbox")
    if (
        not isinstance(shoulder_value, list)
        or len(shoulder_value) != 2
        or not isinstance(paw_value, list)
        or len(paw_value) != 2
        or not isinstance(foreleg_bbox, list)
        or len(foreleg_bbox) != 4
    ):
        raise ValueError("white_model_grooming_invalid_verified_relation")

    transform = _normalization_inverse(source_bbox)
    factor = transform[0]
    shoulder = _point_from_normalized_guide(
        shoulder_value,
        source_bbox,
        transform,
    )
    paw = _point_from_normalized_guide(paw_value, source_bbox, transform)
    foreleg_width_normalized = float(foreleg_bbox[2] - foreleg_bbox[0])
    line_width = int(
        round(min(42.0, max(18.0, foreleg_width_normalized * 0.62)) / factor)
    )
    line_width = max(7, line_width)
    paw_radius = int(
        round(min(24.0, max(11.0, foreleg_width_normalized * 0.38)) / factor)
    )
    paw_radius = max(5, paw_radius)
    elbow = np.asarray(
        (
            (paw[0] * 2.0 + shoulder[0]) / 3.0,
            (paw[1] + shoulder[1]) / 2.0,
        ),
        dtype=np.float32,
    )
    height, width = cat_mask.shape

    def pixel(point: np.ndarray) -> tuple[int, int]:
        bounded = np.asarray(
            (
                np.clip(np.rint(point[0]), 0, width - 1),
                np.clip(np.rint(point[1]), 0, height - 1),
            ),
            dtype=np.int32,
        )
        return int(bounded[0]), int(bounded[1])

    semantic = np.zeros(cat_mask.shape, dtype=np.uint8)
    cv2.line(
        semantic,
        pixel(shoulder),
        pixel(elbow),
        255,
        thickness=line_width,
        lineType=cv2.LINE_AA,
    )
    cv2.line(
        semantic,
        pixel(elbow),
        pixel(paw),
        255,
        thickness=line_width,
        lineType=cv2.LINE_AA,
    )
    cv2.circle(
        semantic,
        pixel(paw),
        paw_radius,
        255,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )
    semantic = (semantic >= 32) & cat_mask
    pixels = int(np.count_nonzero(semantic))
    if pixels < max(80, int(np.count_nonzero(cat_mask) * 0.008)):
        raise ValueError("white_model_grooming_foreleg_semantic_too_small")
    return semantic, {
        "selected_frame_index": int(geometry_metadata["selected_frame_index"]),
        "shoulder_anchor_source": [round(float(value), 3) for value in shoulder],
        "elbow_anchor_source": [round(float(value), 3) for value in elbow],
        "paw_center_source": [round(float(value), 3) for value in paw],
        "line_width_source_px": line_width,
        "paw_radius_source_px": paw_radius,
        "semantic_pixels_selected_frame": pixels,
        "derived_from": "verified_raised_paw_muzzle_relation_geometry_only",
        "source_rgb_or_luma_sampled_for_shape": False,
    }


def _warp_mask_to_adjacent_frame(
    source_mask: np.ndarray,
    source_gray: np.ndarray,
    target_gray: np.ndarray,
    target_cat_mask: np.ndarray,
) -> np.ndarray:
    """Move an anchor-derived semantic through one adjacent source frame."""
    target_to_source = cv2.calcOpticalFlowFarneback(
        target_gray,
        source_gray,
        None,
        0.5,
        3,
        31,
        3,
        7,
        1.5,
        cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
    )
    rows, columns = np.indices(source_mask.shape, dtype=np.float32)
    warped = cv2.remap(
        source_mask.astype(np.float32),
        columns + target_to_source[:, :, 0],
        rows + target_to_source[:, :, 1],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    warped *= target_cat_mask.astype(np.float32)
    source_pixels = int(np.count_nonzero(source_mask))
    usable_pixels = int(np.count_nonzero(warped >= 0.01))
    if usable_pixels < max(32, int(source_pixels * 0.65)):
        raise RuntimeError("white_model_grooming_foreleg_flow_lost")
    target_indices = np.flatnonzero(target_cat_mask)
    if target_indices.size < source_pixels:
        raise RuntimeError("white_model_grooming_foreleg_target_too_small")
    target_scores = warped.ravel()[target_indices]
    selected = np.argpartition(target_scores, -source_pixels)[-source_pixels:]
    tracked = np.zeros(source_mask.shape, dtype=np.uint8)
    tracked.ravel()[target_indices[selected]] = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    tracked = cv2.morphologyEx(tracked, cv2.MORPH_CLOSE, kernel)
    tracked = (tracked > 0) & target_cat_mask
    if int(np.count_nonzero(tracked)) > source_pixels:
        close_scores = warped + tracked.astype(np.float32) * 0.01
        close_indices = np.flatnonzero(tracked)
        selected = np.argpartition(
            close_scores.ravel()[close_indices],
            -source_pixels,
        )[-source_pixels:]
        bounded = np.zeros_like(tracked)
        bounded.ravel()[close_indices[selected]] = True
        tracked = bounded
    return tracked


def make_grooming_foreleg_masks(
    frames: list[Image.Image],
    cat_masks: list[np.ndarray],
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Track the verified action-only foreleg volume through every frame."""
    _guide, geometry_metadata = make_grooming_geometry_guide(
        frames,
        frames[0].size,
    )
    selected_index = int(geometry_metadata["selected_frame_index"])
    selected_mask, semantic_metadata = _synthetic_grooming_foreleg_mask(
        cat_masks[selected_index],
        geometry_metadata,
    )
    gray_frames = [
        cv2.cvtColor(
            np.asarray(frame.convert("RGB"), dtype=np.uint8),
            cv2.COLOR_RGB2GRAY,
        )
        for frame in frames
    ]
    masks: list[np.ndarray | None] = [None] * len(frames)
    masks[selected_index] = selected_mask
    for index in range(selected_index + 1, len(frames)):
        masks[index] = _warp_mask_to_adjacent_frame(
            masks[index - 1],
            gray_frames[index - 1],
            gray_frames[index],
            cat_masks[index],
        )
    for index in range(selected_index - 1, -1, -1):
        masks[index] = _warp_mask_to_adjacent_frame(
            masks[index + 1],
            gray_frames[index + 1],
            gray_frames[index],
            cat_masks[index],
        )
    finalized = [mask for mask in masks if mask is not None]
    if len(finalized) != len(frames):
        raise RuntimeError("white_model_grooming_foreleg_tracking_incomplete")
    areas = [int(np.count_nonzero(mask)) for mask in finalized]
    selected_area = max(1, areas[selected_index])
    minimum_ratio = min(areas) / float(selected_area)
    maximum_ratio = max(areas) / float(selected_area)
    if minimum_ratio < 0.45 or maximum_ratio > 1.85:
        raise RuntimeError(
            "white_model_grooming_foreleg_tracking_unstable: "
            f"area_ratio={minimum_ratio:.4f}..{maximum_ratio:.4f}"
        )
    return finalized, {
        "policy": "verified_anchor_adjacent_optical_flow_soft_depth_v1",
        "geometry_policy": geometry_metadata.get("policy"),
        "geometry_method": geometry_metadata.get("method"),
        **semantic_metadata,
        "semantic_pixels_min": min(areas),
        "semantic_pixels_max": max(areas),
        "semantic_area_ratio_min": round(minimum_ratio, 6),
        "semantic_area_ratio_max": round(maximum_ratio, 6),
        "source_rgb_role": "optical_flow_motion_tracking_only",
        "source_rgb_copied_to_output": False,
        "source_luma_copied_to_output": False,
        "semantic_discrete_color_exposed": False,
    }


def render_white_model_frame(
    frame: Image.Image,
    cat_mask: np.ndarray,
    prop_mask: np.ndarray,
    *,
    foreleg_mask: np.ndarray | None = None,
) -> tuple[Image.Image, dict[str, Any]]:
    """Render one neutral clay frame from geometry, never source appearance."""
    rgba = np.asarray(frame.convert("RGBA"), dtype=np.uint8)
    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3].astype(np.float32) / 255.0
    if cat_mask.shape != alpha.shape or prop_mask.shape != alpha.shape:
        raise ValueError("white_model_driver_mask_size_mismatch")
    if np.any(cat_mask & prop_mask):
        raise ValueError("white_model_driver_cat_prop_overlap")

    cat_pixels = int(np.count_nonzero(cat_mask))
    if cat_pixels < 64:
        raise ValueError("white_model_driver_empty_cat_mask")
    if foreleg_mask is not None:
        if foreleg_mask.shape != cat_mask.shape:
            raise ValueError("white_model_grooming_foreleg_mask_size_mismatch")
        if np.any(foreleg_mask & ~cat_mask):
            raise ValueError("white_model_grooming_foreleg_escaped_cat")

    left, top, right, bottom = _binary_bbox(cat_mask)
    rows, columns = np.indices(cat_mask.shape, dtype=np.float32)
    x_position = np.clip(
        (columns - float(left)) / float(max(1, right - left - 1)),
        0.0,
        1.0,
    )
    y_position = np.clip(
        (rows - float(top)) / float(max(1, bottom - top - 1)),
        0.0,
        1.0,
    )
    directional_light = (0.5 - x_position) * 7.0 + (0.5 - y_position) * 5.0
    geometry_depth = _white_maquette_distance_field(cat_mask)
    white_luma = 184.0 + geometry_depth * 48.0 + directional_light

    foreleg_pixels = 0
    foreleg_relief_min = 0.0
    foreleg_relief_max = 0.0
    if foreleg_mask is not None:
        foreleg_pixels = int(np.count_nonzero(foreleg_mask))
        semantic_u8 = foreleg_mask.astype(np.uint8)
        foreleg_distance = cv2.distanceTransform(
            semantic_u8,
            cv2.DIST_L2,
            5,
        )
        distance_values = foreleg_distance[foreleg_mask]
        distance_p90 = max(1.0, float(np.percentile(distance_values, 90.0)))
        core = np.sqrt(
            np.clip(foreleg_distance / distance_p90, 0.0, 1.0)
        )
        dilated = cv2.dilate(
            semantic_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        )
        eroded = cv2.erode(
            semantic_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )
        boundary = (dilated > eroded).astype(np.float32)
        foreleg_columns = np.nonzero(foreleg_mask)[1]
        foreleg_center_x = float(np.median(foreleg_columns))
        body_center_x = float(left + right - 1) * 0.5
        if body_center_x >= foreleg_center_x:
            torso_side = columns >= foreleg_center_x
        else:
            torso_side = columns <= foreleg_center_x
        shadow = cv2.GaussianBlur(
            boundary * torso_side.astype(np.float32),
            (0, 0),
            sigmaX=3.5,
            sigmaY=3.5,
        )
        shadow_peak = max(0.01, float(shadow.max()))
        shadow = np.clip(shadow / shadow_peak, 0.0, 1.0)
        relief = core * 10.0 - shadow * 10.0
        white_luma += relief
        foreleg_relief_min = float(relief[cat_mask].min())
        foreleg_relief_max = float(relief[cat_mask].max())

    white_luma = np.round(white_luma / 2.0) * 2.0
    white_luma = np.clip(white_luma, 176.0, 244.0)
    white_rgb = np.repeat(white_luma[:, :, None], 3, axis=2)

    output = np.broadcast_to(GREEN, rgb.shape).copy()
    cat_alpha = alpha * (~prop_mask).astype(np.float32)
    output = (
        white_rgb * cat_alpha[:, :, None]
        + output * (1.0 - cat_alpha[:, :, None])
    )
    prop_alpha = alpha * prop_mask.astype(np.float32)
    output = (
        rgb.astype(np.float32) * prop_alpha[:, :, None]
        + output * (1.0 - prop_alpha[:, :, None])
    )
    output_u8 = np.clip(output, 0, 255).astype(np.uint8)

    white_values = white_luma[cat_mask]
    return Image.fromarray(output_u8, mode="RGB"), {
        "cat_pixels": cat_pixels,
        "prop_pixels": int(np.count_nonzero(prop_mask)),
        "foreleg_pixels": foreleg_pixels,
        "cat_bbox": [left, top, right, bottom],
        "white_luma_std": round(float(np.std(white_values)), 6),
        "white_luma_min": int(np.floor(white_values.min())),
        "white_luma_max": int(np.ceil(white_values.max())),
        "white_luma_mean": round(float(np.mean(white_values)), 6),
        "foreleg_relief_min": round(foreleg_relief_min, 6),
        "foreleg_relief_max": round(foreleg_relief_max, 6),
        "white_rgb_channel_spread_max": 0,
        "source_rgb_sampled_for_cat_shading": False,
        "source_luma_sampled_for_cat_shading": False,
    }


def make_white_model_frames(
    frames: list[Image.Image],
    *,
    action: str,
) -> tuple[list[Image.Image], dict[str, Any]]:
    cat_masks, prop_masks, prop_metadata = _cat_and_prop_masks(
        frames,
        action=action,
    )
    if action == "grooming":
        foreleg_masks, foreleg_metadata = make_grooming_foreleg_masks(
            frames,
            cat_masks,
        )
    else:
        foreleg_masks = [None] * len(frames)
        foreleg_metadata = None
    rendered: list[Image.Image] = []
    frame_metrics: list[dict[str, Any]] = []
    for frame, cat_mask, prop_mask, foreleg_mask in zip(
        frames,
        cat_masks,
        prop_masks,
        foreleg_masks,
        strict=True,
    ):
        white_frame, metrics = render_white_model_frame(
            frame,
            cat_mask,
            prop_mask,
            foreleg_mask=foreleg_mask,
        )
        rendered.append(white_frame)
        frame_metrics.append(metrics)

    return rendered, {
        "policy": POLICY,
        "cat_surface": "neutral_achromatic_white_clay",
        "source_rgb_copied_to_cat": False,
        "source_chroma_copied_to_cat": False,
        "source_fur_texture_copied_to_cat": False,
        "source_luma_role": "none_for_cat_shading",
        "source_rgb_sampled_for_cat_shading": False,
        "source_luma_sampled_for_cat_shading": False,
        "source_luma_values_preserved": False,
        "source_marking_contrast_preserved": False,
        "high_frequency_texture_preserved": False,
        "anatomy_gray_quantum": 2,
        "distance_field_volume_shading": True,
        "background": "pure_green_00ff00",
        "prop_policy": (
            "detected_bowl_and_food_preserved"
            if action == "eating"
            else "no_prop"
        ),
        "grooming_foreleg_soft_depth": foreleg_metadata,
        "cat_pixels_min": min(item["cat_pixels"] for item in frame_metrics),
        "cat_pixels_max": max(item["cat_pixels"] for item in frame_metrics),
        "white_luma_min": min(
            item["white_luma_min"] for item in frame_metrics
        ),
        "white_luma_max": max(
            item["white_luma_max"] for item in frame_metrics
        ),
        "white_luma_mean_min": min(
            item["white_luma_mean"] for item in frame_metrics
        ),
        "white_luma_mean_max": max(
            item["white_luma_mean"] for item in frame_metrics
        ),
        "foreleg_pixels_min": min(
            item["foreleg_pixels"] for item in frame_metrics
        ),
        "foreleg_pixels_max": max(
            item["foreleg_pixels"] for item in frame_metrics
        ),
        "foreleg_relief_max": max(
            item["foreleg_relief_max"] for item in frame_metrics
        ),
        "white_rgb_channel_spread_max": 0,
        "prop_metadata": prop_metadata,
    }


def _contact_sheet(frames: list[Image.Image]) -> Image.Image:
    indices = [0, len(frames) // 4, len(frames) // 2, 3 * len(frames) // 4, len(frames) - 1]
    tiles = [
        frames[index].resize((256, 256), Image.Resampling.LANCZOS)
        for index in indices
    ]
    sheet = Image.new("RGB", (256 * len(tiles), 256), (0, 255, 0))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, (index * 256, 0))
    return sheet


def _validate_encoded_video(
    source_frames: list[Image.Image],
    target: Path,
    *,
    expected_frames: int,
) -> dict[str, Any]:
    decoded_frames, _durations = read_animation_frames(target)
    if len(decoded_frames) != expected_frames:
        raise RuntimeError(
            "white_model_driver_frame_count_mismatch: "
            f"{len(decoded_frames)} != {expected_frames}"
        )
    if any(frame.size != (512, 512) for frame in decoded_frames):
        raise RuntimeError("white_model_driver_size_mismatch")

    ious: list[float] = []
    for source, decoded in zip(source_frames, decoded_frames, strict=True):
        source_alpha = np.asarray(source.getchannel("A"), dtype=np.uint8) >= 16
        decoded_alpha = np.asarray(decoded.getchannel("A"), dtype=np.uint8) >= 16
        intersection = int(np.count_nonzero(source_alpha & decoded_alpha))
        union = max(1, int(np.count_nonzero(source_alpha | decoded_alpha)))
        ious.append(intersection / float(union))
    minimum_iou = min(ious)
    if minimum_iou < 0.98:
        raise RuntimeError(
            "white_model_driver_silhouette_changed: "
            f"minimum_iou={minimum_iou:.6f}"
        )
    return {
        "decoded_frames": len(decoded_frames),
        "size": [512, 512],
        "silhouette_iou_min": round(minimum_iou, 6),
        "silhouette_iou_mean": round(float(np.mean(ious)), 6),
    }


def build_action(
    action: str,
    source: Path,
    target: Path,
    *,
    fps: float,
    expected_frames: int,
) -> dict[str, Any]:
    frames, durations = read_animation_frames(source)
    if len(frames) != expected_frames:
        raise RuntimeError(
            f"{action} source frame count is {len(frames)}, "
            f"expected {expected_frames}"
        )
    if len(set(durations)) != 1:
        raise RuntimeError(f"{action} source timing is not constant")

    white_frames, conversion = make_white_model_frames(frames, action=action)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.stem}.tmp.mp4")
    temporary.unlink(missing_ok=True)
    try:
        encode_video(white_frames, temporary, fps, lossless_rgb=True)
        validation = _validate_encoded_video(
            frames,
            temporary,
            expected_frames=expected_frames,
        )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)

    first_frame = target.with_name(f"{action}_white_model_first_frame.png")
    contact_sheet = target.with_name(f"{action}_white_model_contact.png")
    white_frames[0].save(first_frame, format="PNG", optimize=True)
    _contact_sheet(white_frames).save(contact_sheet, format="PNG", optimize=True)
    return {
        "action": action,
        "source": str(source),
        "source_sha256": sha256_file(source),
        "output": str(target),
        "output_sha256": sha256_file(target),
        "first_frame": first_frame.name,
        "first_frame_sha256": sha256_file(first_frame),
        "contact_sheet": contact_sheet.name,
        "contact_sheet_sha256": sha256_file(contact_sheet),
        "frames": expected_frames,
        "fps": fps,
        "conversion": conversion,
        "validation": validation,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--actions",
        nargs="+",
        choices=sorted(SOURCE_FILENAMES),
        default=["eating", "grooming"],
    )
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--fps", type=float, default=16.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_root.resolve() == args.source_root.resolve():
        raise ValueError("White-model output root must differ from source root")
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(
            f"White-model asset manifest already exists: {manifest_path}"
        )

    actions: dict[str, Any] = {}
    for action in args.actions:
        source = args.source_root / SOURCE_FILENAMES[action]
        if not source.is_file():
            raise FileNotFoundError(source)
        target = args.output_root / OUTPUT_FILENAMES[action]
        actions[action] = build_action(
            action,
            source,
            target,
            fps=args.fps,
            expected_frames=args.frames,
        )

    manifest = {
        "schema_version": 1,
        "revision": "white-maquette-v5",
        "policy": POLICY,
        "role": "candidate_appearance_neutral_action_driving_standard",
        "runtime_activated": False,
        "source_root": str(args.source_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "size": [512, 512],
        "frames": args.frames,
        "fps": args.fps,
        "video_codec": "h264rgb_lossless",
        "actions": actions,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
