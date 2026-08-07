#!/usr/bin/env python3
"""Deterministic upload-only identity references for Main_case SCAIL runs."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image


POLICY = "uploaded_identity_plus_main_case_prop_direct_scail_v1"
VERSION = "1.0.0"
IDENTITY_WORKING_MAXIMUM = 1536
REFERENCE_SIZE = (1024, 1024)
IDENTITY_MAXIMUM_SIZE = (940, 984)
IDENTITY_TOP = 40
EATING_PROP_FIRST_RESIZE = (300, 146)
EATING_PROP_FINAL_RESIZE = (380, 146)
EATING_PROP_DRIVER_PAW_TRIM_WIDTH = 167


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rgba_result(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGBA")
    if isinstance(value, bytes):
        return Image.open(io.BytesIO(value)).convert("RGBA")
    return Image.fromarray(np.asarray(value)).convert("RGBA")


def _alpha_bbox(alpha: Image.Image, threshold: int = 16) -> tuple[int, int, int, int]:
    hard = alpha.point(lambda value: 255 if value >= threshold else 0)
    result = hard.getbbox()
    if result is None:
        raise ValueError("uploaded_identity_alpha_is_empty")
    return result


def find_identity_nose(
    identity: Image.Image,
) -> tuple[list[int], tuple[int, int], int]:
    """Locate the upload's warm nose so the standard bowl is placed below it."""

    alpha = np.asarray(identity.getchannel("A"), dtype=np.uint8)
    rgb = np.asarray(identity.convert("RGB"), dtype=np.uint8)
    red, green, blue = [rgb[..., index].astype(np.int16) for index in range(3)]
    rows, columns = np.indices(alpha.shape)
    pink = (
        (alpha >= 64)
        & (columns >= identity.width * 0.35)
        & (columns < identity.width * 0.65)
        & (rows >= identity.height * 0.48)
        & (rows < identity.height * 0.90)
        & (red >= 120)
        & (red - green >= 18)
        & (red - blue >= 8)
    )
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        pink.astype(np.uint8), connectivity=8
    )
    candidates: list[tuple[int, int, int, int, int, int]] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        left = int(stats[label, cv2.CC_STAT_LEFT])
        top = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area >= 20:
            candidates.append((top + height, area, left, top, width, height))
    if not candidates:
        raise ValueError("uploaded_identity_nose_detection_failed")
    _bottom, _area, left, top, width, height = max(candidates)
    box = [left, top, left + width, top + height]
    center = (left + width // 2, top + height // 2)
    return box, center, top + height


def prepare_upload_identity_layers(
    upload_path: str | Path,
    output_dir: str | Path,
    *,
    rembg_session: Any,
    remove_function: Callable[..., Any] | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    """Build a high-resolution cutout whose RGB comes only from the upload."""

    if remove_function is None:
        from rembg import remove as remove_function

    resolved_upload = Path(upload_path).resolve()
    resolved_output = Path(output_dir).resolve()
    resolved_output.mkdir(parents=True, exist_ok=True)
    with Image.open(resolved_upload) as loaded:
        source = loaded.convert("RGB")
    original_size = source.size
    scale = min(1.0, IDENTITY_WORKING_MAXIMUM / float(max(source.size)))
    working_size = (
        max(1, int(round(source.width * scale))),
        max(1, int(round(source.height * scale))),
    )
    working_rgb = source.resize(working_size, Image.Resampling.LANCZOS)
    alpha_matting = True
    try:
        removed = remove_function(
            working_rgb.convert("RGBA"),
            session=rembg_session,
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=10,
        )
    except Exception:  # noqa: BLE001 - deterministic non-matting fallback
        alpha_matting = False
        removed = remove_function(
            working_rgb.convert("RGBA"),
            session=rembg_session,
        )
    removed_rgba = _rgba_result(removed)
    if removed_rgba.size != working_size:
        raise ValueError("uploaded_identity_rembg_size_changed")
    alpha = removed_rgba.getchannel("A")
    alpha_array = np.asarray(alpha, dtype=np.uint8)
    bbox = _alpha_bbox(alpha)
    foreground_fraction = float(np.mean(alpha_array >= 16))
    if not 0.02 <= foreground_fraction <= 0.90:
        raise ValueError(
            "uploaded_identity_foreground_fraction_invalid:"
            f"{foreground_fraction:.6f}"
        )
    # The model contributes only the matte. Every visible RGB byte remains the
    # normalized user upload, preventing a hidden profile/default identity mix.
    cutout = working_rgb.convert("RGBA")
    cutout.putalpha(alpha)
    cutout_path = resolved_output / "uploaded_identity_cutout.png"
    cutout.save(cutout_path, format="PNG", optimize=True)

    crop = cutout.crop(bbox)
    factor = min(
        IDENTITY_MAXIMUM_SIZE[0] / float(crop.width),
        IDENTITY_MAXIMUM_SIZE[1] / float(crop.height),
    )
    resized_size = (
        max(1, int(round(crop.width * factor))),
        max(1, int(round(crop.height * factor))),
    )
    identity = crop.resize(resized_size, Image.Resampling.LANCZOS)
    identity_layer = Image.new("RGBA", REFERENCE_SIZE, (0, 0, 0, 0))
    identity_position = (
        (REFERENCE_SIZE[0] - identity.width) // 2,
        IDENTITY_TOP,
    )
    if identity_position[1] + identity.height > REFERENCE_SIZE[1]:
        raise ValueError("uploaded_identity_reference_overflows_canvas")
    identity_layer.alpha_composite(identity, identity_position)
    identity_path = resolved_output / "uploaded_identity_reference.png"
    identity_layer.save(identity_path, format="PNG", optimize=True)

    nose_box, nose_center_local, nose_bottom_local = find_identity_nose(identity)
    nose_center = (
        identity_position[0] + nose_center_local[0],
        identity_position[1] + nose_center_local[1],
    )
    nose_bottom = identity_position[1] + nose_bottom_local
    metadata = {
        "policy": POLICY,
        "version": VERSION,
        "upload_path": str(resolved_upload),
        "upload_sha256": sha256_file(resolved_upload),
        "upload_size": list(original_size),
        "working_size": list(working_size),
        "working_resize_scale": round(scale, 9),
        "rembg_model_role": "alpha_only",
        "alpha_matting": alpha_matting,
        "alpha_bbox_ge16": list(bbox),
        "alpha_fraction_ge16": round(foreground_fraction, 9),
        "cutout_sha256": sha256_file(cutout_path),
        "identity_reference_sha256": sha256_file(identity_path),
        "identity_reference_size": list(REFERENCE_SIZE),
        "identity_position": list(identity_position),
        "identity_resized_size": list(resized_size),
        "nose_bbox_local": nose_box,
        "nose_center_canvas": list(nose_center),
        "nose_bottom_canvas": nose_bottom,
        "identity_rgb_provenance": (
            "normalized_upload_only_rembg_alpha_only"
        ),
        "profile_or_default_cat_pixels_used": False,
    }
    return cutout_path, identity_path, metadata


def prepare_main_case_eating_prop_layer(
    driving_path: str | Path,
    output_dir: str | Path,
    *,
    nose_center: tuple[int, int],
    nose_bottom: int,
) -> tuple[Path, dict[str, Any]]:
    """Extract only the fixed Main_case frame-zero bowl/kibble pixels."""

    from scail2_subject_replacement_test import (
        make_eating_prop_exclusion_masks,
        contain_rgba,
        expand_bbox,
        read_animation_frames,
        sample_timed_frames,
        union_alpha_bbox,
    )

    resolved_driver = Path(driving_path).resolve()
    resolved_output = Path(output_dir).resolve()
    resolved_output.mkdir(parents=True, exist_ok=True)
    frames, durations = read_animation_frames(resolved_driver)
    sampled = sample_timed_frames(frames, durations, 81, 16.0)
    source_bbox = expand_bbox(
        union_alpha_bbox(sampled),
        sampled[0].size,
        margin_fraction=0.06,
    )
    normalized = [
        contain_rgba(frame.crop(source_bbox), (512, 512), 0.88)
        for frame in sampled
    ]
    prop_masks, prop_metadata = make_eating_prop_exclusion_masks(normalized)
    frame_rgb = np.asarray(normalized[0].convert("RGB"), dtype=np.uint8)
    frame_alpha = np.asarray(prop_masks[0].convert("L"), dtype=np.uint8)
    raw = Image.fromarray(np.dstack((frame_rgb, frame_alpha)), mode="RGBA")
    raw_bbox = _alpha_bbox(raw.getchannel("A"))
    raw_crop = raw.crop(raw_bbox)
    if raw_crop.width < EATING_PROP_DRIVER_PAW_TRIM_WIDTH:
        raise ValueError("main_case_eating_prop_crop_too_narrow")
    # The fixed mask includes a few right-paw pixels. The source-locked trim
    # retains the bowl/kibble only; no driver cat identity reaches the reference.
    bowl_only = raw_crop.crop(
        (0, 0, EATING_PROP_DRIVER_PAW_TRIM_WIDTH, raw_crop.height)
    )
    first_resize = bowl_only.resize(
        EATING_PROP_FIRST_RESIZE,
        Image.Resampling.LANCZOS,
    )
    final_prop = first_resize.resize(
        EATING_PROP_FINAL_RESIZE,
        Image.Resampling.LANCZOS,
    )
    prop_position = (nose_center[0] - final_prop.width // 2, nose_bottom - 6)
    if (
        prop_position[0] < 0
        or prop_position[1] < 0
        or prop_position[0] + final_prop.width > REFERENCE_SIZE[0]
        or prop_position[1] + final_prop.height > REFERENCE_SIZE[1]
    ):
        raise ValueError("main_case_eating_prop_overflows_reference_canvas")
    layer = Image.new("RGBA", REFERENCE_SIZE, (0, 0, 0, 0))
    layer.alpha_composite(final_prop, prop_position)
    layer_path = resolved_output / "main_case_eating_prop.png"
    layer.save(layer_path, format="PNG", optimize=True)
    metadata = {
        "policy": POLICY,
        "version": VERSION,
        "driving_path": str(resolved_driver),
        "driving_sha256": sha256_file(resolved_driver),
        "source_frame_index": 0,
        "source_union_bbox": list(source_bbox),
        "source_prop_bbox": list(raw_bbox),
        "driver_paw_trim_width": EATING_PROP_DRIVER_PAW_TRIM_WIDTH,
        "first_resize": list(EATING_PROP_FIRST_RESIZE),
        "final_resize": list(EATING_PROP_FINAL_RESIZE),
        "position": list(prop_position),
        "prop_layer_sha256": sha256_file(layer_path),
        "pixel_source": "main_case_frame0_bowl_and_kibble_only",
        "profile_or_default_cat_pixels_used": False,
        "runner_prop_metadata": prop_metadata,
    }
    return layer_path, metadata


def build_uploaded_identity_scail_references(
    upload_path: str | Path,
    *,
    actions: list[str],
    eating_driver_path: str | Path,
    output_dir: str | Path,
    rembg_session: Any,
    remove_function: Callable[..., Any] | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Create per-action SCAIL references and verify the runner's masks."""

    unsupported = sorted(set(actions) - {"eating", "grooming"})
    if unsupported:
        raise ValueError("unsupported_subject_actions:" + ",".join(unsupported))
    resolved_output = Path(output_dir).resolve()
    _cutout, identity_path, identity_metadata = prepare_upload_identity_layers(
        upload_path,
        resolved_output,
        rembg_session=rembg_session,
        remove_function=remove_function,
    )
    references: dict[str, str] = {}
    action_metadata: dict[str, Any] = {}
    if "grooming" in actions:
        grooming_path = resolved_output / "grooming_uploaded_identity_reference.png"
        grooming_path.write_bytes(identity_path.read_bytes())
        references["grooming"] = str(grooming_path)
        action_metadata["grooming"] = {
            "reference_sha256": sha256_file(grooming_path),
            "identity_source_sha256": identity_metadata[
                "identity_reference_sha256"
            ],
            "prop_source": None,
            "profile_or_default_cat_pixels_used": False,
        }
    if "eating" in actions:
        prop_path, prop_metadata = prepare_main_case_eating_prop_layer(
            eating_driver_path,
            resolved_output,
            nose_center=tuple(identity_metadata["nose_center_canvas"]),
            nose_bottom=int(identity_metadata["nose_bottom_canvas"]),
        )
        with Image.open(identity_path) as identity_loaded:
            eating_reference = identity_loaded.convert("RGBA")
        with Image.open(prop_path) as prop_loaded:
            eating_reference.alpha_composite(prop_loaded.convert("RGBA"))
        eating_path = resolved_output / "eating_uploaded_identity_reference.png"
        eating_reference.save(eating_path, format="PNG", optimize=True)
        references["eating"] = str(eating_path)
        action_metadata["eating"] = {
            "reference_sha256": sha256_file(eating_path),
            "identity_source_sha256": identity_metadata[
                "identity_reference_sha256"
            ],
            "prop_source_sha256": prop_metadata["prop_layer_sha256"],
            "prop": prop_metadata,
            "profile_or_default_cat_pixels_used": False,
        }

    from scail2_subject_replacement_test import make_reference_assets

    for action, reference_text in references.items():
        _normalized, mask, preflight = make_reference_assets(
            Path(reference_text),
            (512, 512),
            separate_eating_prop=action == "eating",
        )
        mask_array = np.asarray(mask.convert("RGB"), dtype=np.uint8)
        unknown = ~(
            np.all(mask_array == (0, 0, 0), axis=2)
            | np.all(mask_array == (0, 0, 255), axis=2)
            | np.all(mask_array == (255, 0, 0), axis=2)
        )
        if np.any(unknown):
            raise ValueError(f"{action}_reference_mask_contains_unknown_colors")
        if action == "eating" and preflight["reference_prop_fraction"] <= 0.0:
            raise ValueError("eating_reference_preflight_has_no_prop")
        action_metadata[action]["preflight"] = preflight
        action_metadata[action]["reference_mask_counts"] = {
            "black": int(np.count_nonzero(np.all(mask_array == (0, 0, 0), axis=2))),
            "blue_identity": int(
                np.count_nonzero(np.all(mask_array == (0, 0, 255), axis=2))
            ),
            "red_prop": int(
                np.count_nonzero(np.all(mask_array == (255, 0, 0), axis=2))
            ),
            "unknown": int(np.count_nonzero(unknown)),
        }
    manifest = {
        "policy": POLICY,
        "version": VERSION,
        "identity": identity_metadata,
        "actions": action_metadata,
        "references": references,
        "profile_or_default_cat_pixels_used": False,
    }
    manifest_path = resolved_output / "uploaded_identity_reference_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    manifest["manifest_path"] = str(manifest_path)
    manifest["manifest_sha256"] = sha256_file(manifest_path)
    return references, manifest
