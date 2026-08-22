#!/usr/bin/env python3
"""Run a minimal SCAIL-2 character-replacement test through ComfyUI.

The script prepares a short motion clip and exact SCAIL identity masks from
alpha-enabled assets, submits an API-format ComfyUI graph, and records the
result plus runtime diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageSequence

try:
    import av
except ModuleNotFoundError:  # CPU-only guide/tests do not decode video containers.
    av = None

from ..media.transparent_animation import (
    GIF_ALPHA_SMOOTHING_SIGMA,
    GIF_ALPHA_THRESHOLD,
    _binary_alpha_edge_metrics,
    _encode_soft_alpha_apng,
    _gif_alpha,
    _save_rgba,
    _subject_mask,
    validate_animation,
)

COMFY_URL = "http://127.0.0.1:8188"
COMFY_INPUT = Path("/data/comfy-pet-pipeline/comfyui/input")
COMFY_OUTPUT = Path("/data/comfy-pet-pipeline/comfyui/output")
EXPERIMENT_ROOT = Path("/data/comfy-pet-pipeline/experiments/scail2_tm_test")
SOURCE_ANIMATION = Path(
    "/data/comfy-pet-pipeline/profile_assets/sleek_tabby_cat_v1/driving/"
    "20260808-user-white-model-v1/"
    "grooming_white_model_driver_512x512_81f_16fps.mp4"
)
REFERENCE_IMAGE = Path(
    "/data/comfy-pet-pipeline/comfyui/output/fur_lora_test/"
    "20260725_direct_sleek_detailz_w010_v2/eating_pose_00001_.png"
)

UNET_NAME = "wan2.1_14B_SCAIL_2_fp8_scaled.safetensors"
DPO_LORA = "wan2.1_SCAIL_2_DPO_lora_bf16.safetensors"
DISTILL_LORA = (
    "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"
)
TEXT_ENCODER = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
VAE_NAME = "wan_2.1_vae.safetensors"
CLIP_VISION = "clip_vision_h.safetensors"
EATING_POSE_STRENGTH_MAX = 1.35
EATING_TARGET_MUZZLE_FOOD_OVERLAP_PX = 8.0
EATING_MUZZLE_FOOD_OVERLAP_MIN_PX = 6
EATING_MUZZLE_FOOD_OVERLAP_MAX_PX = 10
EATING_FULL_POSE_OVERALL_CHANGE_RATIO_MIN = 0.02
EATING_FULL_POSE_LOWER_HALF_CHANGE_RATIO_MIN = 0.01
EATING_FULL_POSE_LOWER_QUARTER_CHANGE_RATIO_MIN = 0.012
GROOMING_TEMPORAL_LOOP_POLICY = (
    "grooming_second_half_ping_pong_closed_loop_v1"
)

DEFAULT_PROMPT = (
    "A complete full-body premium stylized 3D cartoon cat matching the sleek "
    "reference as the only identity standard. Preserve the exact face shape, "
    "cheek width, eye color and spacing, pupils, nose, muzzle, ear shape, exact "
    "reference coat colors and marking placement, compact body volume, full tail, "
    "and exact smooth rounded back silhouette. Preserve the source coat-length "
    "category without amplifying it. Fur is dense, compact, uniformly directed "
    "and naturally close to the body with only short soft edge hairs. Use the "
    "neutral-white anatomy-maquette driving clip only for motion timing, joint "
    "trajectories, seated pose, raised-paw path, camera and framing. The driving "
    "maquette has no identity or appearance authority: never copy its face, "
    "eyes, white surface, body proportions, material, lighting or texture. If "
    "the supplied identity reference is genuinely white, preserve that white "
    "coat because the identity reference requires it, never because the driver "
    "does. The cat "
    "follows the supplied grooming motion, pose sequence and timing "
    "precisely, sitting and slowly licking one raised front paw and rubbing its "
    "face. Motion must never change fur length, volume, grooming direction or "
    "the clean back contour. "
    "Anatomically correct feline body and joints with exactly four legs and "
    "four paws total. There are exactly two forelegs: one foreleg rises "
    "continuously from its shoulder to the paw at the mouth, while the other "
    "foreleg ends in the only planted front paw at the bottom. Never leave a "
    "second ground paw for the raised foreleg and never create a third "
    "foreleg. Keep the two seated hind legs distinct from the front legs. "
    "The supplied reference image already contains this one-raised, one-planted "
    "foreleg topology. Animate those same two forelegs from the first frame; "
    "never preserve two planted forepaws and then add a new raised paw. "
    "Stable face and identity without freezing it, no food, no bowl, no prop, "
    "no clothes, no collar and no accessories. Static camera, centered complete "
    "uncropped body, clean uniform white studio background, smooth temporally "
    "coherent motion. Separate pale fur from the background only with natural "
    "soft tonal lighting, never with a drawn silhouette border."
)

DEFAULT_NEGATIVE_PROMPT = (
    "extra limb, extra leg, extra paw, duplicated limb, duplicated foreleg, "
    "three front legs, three forelegs, five legs, detached paw, floating paw, "
    "fused legs, malformed anatomy, retained old paw, pose ghosting, multiple "
    "cats, distorted face, changing fur markings, food, kibble, bowl, dish, "
    "copied driving-maquette face, copied driving-maquette eyes, driver identity, "
    "driver material, driver lighting, uniform maquette surface, "
    "fur volume expansion, puffed-up fur, raised guard hairs, bristling, spiky "
    "fur, needle-like fur, wire-like fur, rigid fur clumps, long isolated "
    "strands, dorsal ridge, mohawk, jagged back silhouette, changing back "
    "contour, missing back, clipped back, fur crawling, texture crawling, "
    "black contour, dark outline, ink line, comic stroke, silhouette halo, "
    "camera movement, zoom, flicker, jitter"
)

EATING_PROMPT = (
    "A complete full-body premium stylized 3D cartoon cat matching the "
    "supplied sleek reference as the only identity standard. Preserve the "
    "exact face shape, cheek width, eye color and spacing, pupils, nose, "
    "muzzle, ear shape, coat colors and marking placement, body volume, paws, "
    "tail, and the exact smooth rounded back silhouette. Preserve the source "
    "coat-length category without amplifying it. The coat is dense, compact, "
    "soft, uniformly directed and lies naturally close to the body along the "
    "spine, shoulders, flanks and chest, with only short soft edge hairs. "
    "Use the neutral-white anatomy-maquette driving clip only for low-head "
    "motion timing, joint trajectories, the fixed bowl, camera and framing. "
    "Never copy the maquette's face, eyes, white surface, body proportions, "
    "material, lighting or texture; the supplied identity reference alone "
    "controls every appearance trait. If that identity is genuinely white, "
    "keep it white because the identity reference requires it, never because "
    "the driver is white. "
    "Motion may move the joints, head, neck, ears, muzzle and chest naturally "
    "but must never change fur length, fur volume, grooming direction or the "
    "back contour. Follow the supplied eating motion, pose sequence and timing "
    "precisely. The cat repeatedly lowers its muzzle a small natural distance "
    "into exactly one shallow light-beige food bowl and raises it again. Keep "
    "the bowl stationary, unchanged and below the muzzle; all kibble stays "
    "inside the bowl below the rim. The bowl placement and horizontal "
    "coordinate follow the driving template exactly, and the projected muzzle "
    "center stays over the bowl opening during every low-head eating phase. "
    "Exactly four attached legs and four paws "
    "total. Exactly two forelegs remain continuously attached to the shoulders "
    "and end in exactly two planted front paws beside or behind the bowl "
    "without defining or shifting the bowl center; "
    "no foreleg is raised, hidden, retained twice, or merged with the bowl. "
    "Keep the recognizable face temporally stable without freezing it. Static "
    "camera, stable scale and placement, complete uncropped subject and tail, "
    "clean perfectly uniform white studio background, no floor shadow, smooth "
    "temporally coherent motion. Separate pale fur from the background only "
    "with natural soft tonal lighting, never with a drawn silhouette border."
)

EATING_NEGATIVE_PROMPT = (
    "extra limb, extra leg, extra paw, duplicated limb, duplicated foreleg, "
    "third foreleg, retained old paw, missing paw, hidden paw, fused legs, "
    "detached paw, floating paw, raised front paw, paw holding food, bowl "
    "replacing a paw, paw merged into bowl, duplicate bowl, extra bowl, "
    "changing bowl, deformed bowl, floating kibble, food outside bowl, food "
    "stuck to face, stretched mouth, distorted muzzle, melted face, changing "
    "identity, copied driving-maquette face, copied driving-maquette eyes, "
    "driver identity, driver material, driver lighting, uniform maquette surface, "
    "changing fur markings, fur crawling, texture crawling, fur "
    "volume expansion, puffed-up fur, raised fur, standing guard hairs, "
    "bristling, spiky fur, needle-like fur, wire-like fur, rigid fur clumps, "
    "long isolated strands, hanging tufts, dorsal ridge, mohawk, jagged fur "
    "silhouette, changing back contour, missing back, clipped back, multiple "
    "cats, cropped body, cropped tail, camera movement, zoom, flicker, jitter, "
    "black contour, dark outline, ink line, comic stroke, silhouette halo, "
    "green background, floor, shadow, gradient, texture, text, watermark"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE_ANIMATION)
    parser.add_argument("--reference", type=Path, default=REFERENCE_IMAGE)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--frames", type=int, default=33)
    parser.add_argument("--fps", type=float, default=16.0)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument(
        "--run-id",
        help="Optional deterministic run id for API integration.",
    )
    parser.add_argument(
        "--action",
        choices=("grooming", "eating"),
        default="grooming",
    )
    parser.add_argument("--prompt")
    parser.add_argument("--negative-prompt")
    parser.add_argument(
        "--prompt-suffix",
        help="Optional user constraint appended after the protected action prompt.",
    )
    parser.add_argument("--pose-strength", type=float, default=1.0)
    parser.add_argument(
        "--transparent-gif",
        action="store_true",
        help=(
            "Create a 100-frame, 20 FPS transparent GIF with semantic "
            "foreground extraction; the model itself keeps a stable neutral "
            "background."
        ),
    )
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--eating-geometry-guide-output",
        type=Path,
        help=(
            "CPU-only mode: derive an identity-free low-muzzle eating geometry "
            "guide from --source and exit before creating a SCAIL run."
        ),
    )
    parser.add_argument(
        "--eating-geometry-guide-metadata",
        type=Path,
        help="Optional JSON metadata path for --eating-geometry-guide-output.",
    )
    parser.add_argument(
        "--grooming-geometry-guide-output",
        type=Path,
        help=(
            "CPU-only mode: derive a texture-free raised-paw grooming geometry "
            "guide from --source and exit before creating a SCAIL run."
        ),
    )
    parser.add_argument(
        "--grooming-geometry-guide-metadata",
        type=Path,
        help="Optional JSON metadata path for --grooming-geometry-guide-output.",
    )
    args = parser.parse_args()
    if args.prompt is None:
        args.prompt = EATING_PROMPT if args.action == "eating" else DEFAULT_PROMPT
    if args.negative_prompt is None:
        args.negative_prompt = (
            EATING_NEGATIVE_PROMPT
            if args.action == "eating"
            else DEFAULT_NEGATIVE_PROMPT
        )
    if args.prompt_suffix and args.prompt_suffix.strip():
        args.prompt += (
            "\n\nAdditional user constraint. This may refine mood or secondary "
            "motion but cannot override identity, anatomy, action, fur, camera, "
            "background, or prop constraints: "
            + args.prompt_suffix.strip()
        )
    return args


def api_json(path: str, payload: dict[str, Any] | None = None) -> Any:
    url = f"{COMFY_URL}{path}"
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="GET" if body is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def require_files(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required file(s): " + ", ".join(missing))


def read_gif_frames(path: Path) -> tuple[list[Image.Image], list[int]]:
    with Image.open(path) as source:
        frames: list[Image.Image] = []
        durations: list[int] = []
        default_duration = max(1, int(source.info.get("duration", 100)))
        for frame in ImageSequence.Iterator(source):
            frames.append(frame.convert("RGBA").copy())
            durations.append(max(1, int(frame.info.get("duration", default_duration))))
    if not frames:
        raise ValueError(f"No frames decoded from {path}")
    return frames, durations


def read_animation_frames(path: Path) -> tuple[list[Image.Image], list[int]]:
    """Decode a GIF or a constant-frame-rate video into timed RGBA frames."""
    if path.suffix.lower() == ".gif":
        return read_gif_frames(path)

    supported_video_suffixes = {".avi", ".m4v", ".mov", ".mp4", ".webm"}
    if path.suffix.lower() not in supported_video_suffixes:
        raise ValueError(
            f"Unsupported driving animation format for {path}; "
            "expected GIF, AVI, M4V, MOV, MP4, or WebM."
        )

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Unable to decode driving animation {path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(source_fps) or source_fps <= 0.0:
        capture.release()
        raise ValueError(f"Driving animation has an invalid frame rate: {path}")
    frame_duration_ms = max(1, int(round(1000.0 / source_fps)))
    frames: list[Image.Image] = []
    try:
        while True:
            decoded, bgr = capture.read()
            if not decoded:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            clean_rgb, alpha = _subject_mask(rgb)
            rgba = np.dstack((clean_rgb, alpha))
            frames.append(Image.fromarray(rgba, mode="RGBA"))
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"No frames decoded from {path}")
    return frames, [frame_duration_ms] * len(frames)


def sample_timed_frames(
    frames: list[Image.Image],
    durations_ms: list[int],
    output_count: int,
    output_fps: float,
) -> list[Image.Image]:
    cumulative = np.cumsum(np.asarray(durations_ms, dtype=np.float64))
    total_ms = float(cumulative[-1])
    selected: list[Image.Image] = []
    for index in range(output_count):
        timestamp_ms = min(index * 1000.0 / output_fps, total_ms - 0.001)
        source_index = int(np.searchsorted(cumulative, timestamp_ms, side="right"))
        selected.append(frames[min(source_index, len(frames) - 1)].copy())
    return selected


def union_alpha_bbox(frames: Iterable[Image.Image], threshold: int = 8) -> tuple[int, int, int, int]:
    bbox: tuple[int, int, int, int] | None = None
    for frame in frames:
        alpha = frame.getchannel("A").point(lambda value: 255 if value > threshold else 0)
        current = alpha.getbbox()
        if not current:
            continue
        if bbox is None:
            bbox = current
        else:
            bbox = (
                min(bbox[0], current[0]),
                min(bbox[1], current[1]),
                max(bbox[2], current[2]),
                max(bbox[3], current[3]),
            )
    if bbox is None:
        raise ValueError("The source has no visible alpha subject")
    return bbox


def expand_bbox(
    bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
    margin_fraction: float = 0.05,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = bbox
    margin = int(round(max(right - left, bottom - top) * margin_fraction))
    return (
        max(0, left - margin),
        max(0, top - margin),
        min(image_size[0], right + margin),
        min(image_size[1], bottom + margin),
    )


def contain_rgba(
    image: Image.Image,
    canvas_size: tuple[int, int],
    scale_fraction: float,
) -> Image.Image:
    rgba = image.convert("RGBA")
    max_width = max(1, int(canvas_size[0] * scale_fraction))
    max_height = max(1, int(canvas_size[1] * scale_fraction))
    factor = min(max_width / rgba.width, max_height / rgba.height)
    resized = rgba.resize(
        (
            max(1, int(round(rgba.width * factor))),
            max(1, int(round(rgba.height * factor))),
        ),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    position = (
        (canvas_size[0] - resized.width) // 2,
        (canvas_size[1] - resized.height) // 2,
    )
    canvas.alpha_composite(resized, position)
    return canvas


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binary_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, columns = np.where(mask)
    if not len(columns):
        raise ValueError("eating_prop_split_unreliable: empty_mask")
    return (
        int(columns.min()),
        int(rows.min()),
        int(columns.max() + 1),
        int(rows.max() + 1),
    )


def fill_internal_holes(mask: np.ndarray) -> np.ndarray:
    binary = mask.astype(np.uint8)
    padded = np.pad(binary, 1, mode="constant", constant_values=0)
    inverse = (1 - padded).astype(np.uint8)
    flood = inverse.copy()
    flood_mask = np.zeros(
        (flood.shape[0] + 2, flood.shape[1] + 2),
        dtype=np.uint8,
    )
    cv2.floodFill(flood, flood_mask, (0, 0), 2)
    holes = (inverse != 0) & (flood != 2)
    return (padded.astype(bool) | holes)[1:-1, 1:-1]


def make_eating_prop_exclusion_masks(
    normalized_frames: list[Image.Image],
) -> tuple[list[Image.Image], dict[str, Any]]:
    """Separate a food-anchored static bowl from the animated cat."""
    if len(normalized_frames) < 9:
        raise ValueError("eating_prop_split_unreliable: fewer_than_9_frames")
    size = normalized_frames[0].size
    if any(frame.size != size for frame in normalized_frames):
        raise ValueError("eating_prop_split_unreliable: inconsistent_frame_size")

    rgba = np.stack(
        [
            np.asarray(frame.convert("RGBA"), dtype=np.uint8)
            for frame in normalized_frames
        ]
    )
    rgb = rgba[..., :3]
    foreground = rgba[..., 3] >= 16
    union = foreground.any(axis=0)
    left, top, right, bottom = binary_bbox(union)
    subject_width = right - left
    subject_height = bottom - top

    median = np.median(rgb.astype(np.int16), axis=0).astype(np.uint8)
    lower_quantile = np.percentile(rgb, 10, axis=0)
    upper_quantile = np.percentile(rgb, 90, axis=0)
    robust_span = (upper_quantile - lower_quantile).max(axis=-1)
    presence = foreground.mean(axis=0)

    # Locate the bowl by what makes an eating prop semantically distinct from
    # the cat: a wide, low band of warm food.  The former lower-right ROI
    # happened to work for the old side-mounted fixture, but on Main_case/eat.mp4
    # it selected the black-and-white cat's right tail instead of the central
    # white bowl.  Work from the temporal median so a dipping muzzle does not
    # permanently hide the food.
    row_indices, column_indices = np.indices(union.shape)
    lower_subject = (
        union
        & (presence >= 0.50)
        & (row_indices >= top + int(round(subject_height * 0.55)))
        & (row_indices < bottom)
    )
    median_red = median[..., 0].astype(np.int16)
    median_green = median[..., 1].astype(np.int16)
    median_blue = median[..., 2].astype(np.int16)
    minimum_anchor_area = max(48, int(round(subject_width * subject_height * 0.001)))
    minimum_anchor_width = max(24, int(round(subject_width * 0.10)))
    anchor_candidates: list[tuple[float, int]] = []
    warm_food = np.zeros_like(union, dtype=bool)
    labels = np.zeros_like(union, dtype=np.int32)
    anchor_red_threshold = 0
    # Prefer bright warm food/bowl pixels.  This separates a beige vessel from
    # an entirely brown driver cat.  The relaxed pass retains darker kibble for
    # white-bowl inputs where the surrounding cat is neutral.
    for red_threshold in (150, 110):
        warm_food = (
            lower_subject
            & (median_red >= red_threshold)
            & (median_red - median_green >= 18)
            & (median_green - median_blue >= 8)
        )
        warm_food = cv2.morphologyEx(
            warm_food.astype(np.uint8),
            cv2.MORPH_CLOSE,
            np.ones((5, 9), dtype=np.uint8),
            iterations=1,
        ).astype(bool)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            warm_food.astype(np.uint8),
            connectivity=8,
        )
        anchor_candidates = []
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            component_width = int(stats[label, cv2.CC_STAT_WIDTH])
            component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
            _center_x, center_y = centroids[label]
            aspect = component_width / float(max(component_height, 1))
            if (
                area < minimum_anchor_area
                or component_width < minimum_anchor_width
                or component_height > int(round(subject_height * 0.35))
                or aspect < 1.65
                or center_y < top + subject_height * 0.58
            ):
                continue
            relative_y = (center_y - top) / float(max(subject_height, 1))
            score = component_width * aspect * aspect * (1.0 + relative_y)
            anchor_candidates.append((score, label))
        if anchor_candidates:
            anchor_red_threshold = red_threshold
            break
    if not anchor_candidates:
        raise ValueError(
            "eating_prop_split_unreliable: "
            f"warm_food_anchor_candidates=0, "
            f"warm_food_pixels={int(np.count_nonzero(warm_food))}"
        )

    _score, anchor_label = max(anchor_candidates)
    food_anchor = labels == anchor_label
    anchor_left, anchor_top, anchor_right, anchor_bottom = binary_bbox(
        food_anchor
    )
    anchor_width = anchor_right - anchor_left

    # The food occupies the opening of the vessel.  Give the vessel search
    # enough lateral room for a flared white/gray bowl, while still excluding
    # the planted paws and the right-side tail.  Neutral pixels are admitted
    # only when temporally stable; the warm anchor itself remains valid for the
    # older beige-bowl drivers where food and vessel share one color.
    vessel_width = max(
        int(round(anchor_width * 1.68)),
        int(round(subject_width * 0.48)),
    )
    vessel_half_width = max(12, int(round(vessel_width / 2.0)))
    anchor_center_x = (anchor_left + anchor_right) // 2
    vessel_center_shift = int(round(anchor_width * 0.04))
    vessel_left = max(
        left,
        anchor_center_x - vessel_half_width + vessel_center_shift,
    )
    vessel_right = min(
        right,
        anchor_center_x + vessel_half_width + vessel_center_shift,
    )
    vessel_top = max(top, anchor_top - 2)
    vessel_bottom = bottom
    if (
        vessel_right - vessel_left < minimum_anchor_width
        or vessel_bottom <= vessel_top
    ):
        raise ValueError(
            "eating_prop_split_unreliable: invalid_food_anchored_vessel_roi"
        )
    vessel_roi = np.zeros_like(union, dtype=bool)
    vessel_roi[vessel_top:vessel_bottom, vessel_left:vessel_right] = True

    median_hsv = cv2.cvtColor(median, cv2.COLOR_RGB2HSV)
    neutral_bowl = (
        vessel_roi
        & union
        & (presence >= 0.88)
        & (robust_span <= 18.0)
        & (median_hsv[..., 1] <= 60)
        & (median_hsv[..., 2] >= 45)
    )
    bowl_support = neutral_bowl | food_anchor
    bowl_support = cv2.morphologyEx(
        bowl_support.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    ).astype(bool)
    support_count, support_labels, support_stats, _support_centroids = (
        cv2.connectedComponentsWithStats(
            bowl_support.astype(np.uint8),
            connectivity=8,
        )
    )
    food_touch = cv2.dilate(
        food_anchor.astype(np.uint8),
        np.ones((7, 7), dtype=np.uint8),
        iterations=1,
    ).astype(bool)
    support_candidates: list[tuple[int, int]] = []
    for label in range(1, support_count):
        component = support_labels == label
        if np.any(component & food_touch):
            support_candidates.append(
                (int(support_stats[label, cv2.CC_STAT_AREA]), label)
            )
    if not support_candidates:
        raise ValueError(
            "eating_prop_split_unreliable: food_anchor_has_no_vessel_support"
        )
    _support_area, selected_support_label = max(support_candidates)
    kept = (support_labels == selected_support_label) | food_anchor
    if int(np.count_nonzero(kept)) < 100:
        raise ValueError(
            "eating_prop_split_unreliable: "
            f"stable_core_pixels={int(np.count_nonzero(kept))}"
        )

    points_xy = np.column_stack(np.where(kept))[:, ::-1].astype(np.int32)
    hull = cv2.convexHull(points_xy)
    envelope = np.zeros_like(kept, dtype=np.uint8)
    cv2.fillConvexPoly(envelope, hull, 1)
    envelope = cv2.dilate(
        envelope,
        np.ones((5, 5), dtype=np.uint8),
        iterations=1,
    ).astype(bool)
    envelope &= vessel_roi

    exclusions: list[np.ndarray] = []
    for index in range(rgba.shape[0]):
        delta = np.abs(
            rgb[index].astype(np.int16) - median.astype(np.int16)
        ).max(axis=-1)
        prop = envelope & foreground[index] & (delta <= 18)
        prop = cv2.morphologyEx(
            prop.astype(np.uint8),
            cv2.MORPH_CLOSE,
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        )
        prop = cv2.dilate(
            prop,
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        ).astype(bool)
        prop &= envelope & foreground[index]

        exclusion = cv2.dilate(
            prop.astype(np.uint8),
            np.ones((5, 5), dtype=np.uint8),
            iterations=1,
        ).astype(bool)
        exclusion &= vessel_roi
        exclusions.append(exclusion)

    areas = np.asarray(
        [int(np.count_nonzero(mask)) for mask in exclusions],
        dtype=np.float64,
    )
    boxes = [binary_bbox(mask) for mask in exclusions]
    centers = np.asarray(
        [
            ((box[0] + box[2] - 1) / 2.0, (box[1] + box[3] - 1) / 2.0)
            for box in boxes
        ]
    )
    center_steps = (
        np.linalg.norm(np.diff(centers, axis=0), axis=1)
        if len(centers) > 1
        else np.zeros(1)
    )
    area_cv = float(areas.std() / max(areas.mean(), 1.0))
    center_step_max = float(center_steps.max())
    if area_cv > 0.05 or center_step_max > 1.5:
        raise ValueError(
            "eating_prop_split_unreliable: "
            f"area_cv={area_cv:.4f}, center_step_max={center_step_max:.3f}"
        )

    foreground_areas = np.asarray(
        [max(1, int(np.count_nonzero(mask))) for mask in foreground],
        dtype=np.float64,
    )
    prop_fractions = areas / foreground_areas
    if float(prop_fractions.min()) < 0.04 or float(prop_fractions.max()) > 0.20:
        raise ValueError(
            "eating_prop_split_unreliable: "
            f"prop_fraction={float(prop_fractions.min()):.4f}-"
            f"{float(prop_fractions.max()):.4f}"
        )

    exclusion_union = np.logical_or.reduce(exclusions)
    metadata = {
        "policy": "map_driving_bowl_as_red_scail_object",
        "method": "eating_temporal_food_anchor_bowl_v2",
        "subject_bbox": list((left, top, right, bottom)),
        "roi_bbox": [vessel_left, vessel_top, vessel_right, vessel_bottom],
        "warm_food_anchor_bbox": [
            anchor_left,
            anchor_top,
            anchor_right,
            anchor_bottom,
        ],
        "warm_food_anchor_pixels": int(np.count_nonzero(food_anchor)),
        "warm_food_anchor_candidates": len(anchor_candidates),
        "warm_food_anchor_red_threshold": anchor_red_threshold,
        "stable_core_pixels": int(np.count_nonzero(kept)),
        "stable_core_components": 1,
        "envelope_bbox": list(binary_bbox(envelope)),
        "prop_exclusion_bbox_union": list(binary_bbox(exclusion_union)),
        "prop_area_min": int(areas.min()),
        "prop_area_max": int(areas.max()),
        "prop_area_cv": area_cv,
        "prop_fraction_min": float(prop_fractions.min()),
        "prop_fraction_max": float(prop_fractions.max()),
        "prop_center_x_minmax": [
            float(centers[:, 0].min()),
            float(centers[:, 0].max()),
        ],
        "prop_center_y_minmax": [
            float(centers[:, 1].min()),
            float(centers[:, 1].max()),
        ],
        "prop_center_step_max": center_step_max,
    }
    return (
        [
            Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
            for mask in exclusions
        ],
        metadata,
    )


def make_pose_and_mask_frames(
    sampled_frames: list[Image.Image],
    canvas_size: tuple[int, int],
    pose_background: tuple[int, int, int] = (255, 255, 255),
    separate_eating_prop: bool = False,
) -> tuple[list[Image.Image], list[Image.Image], dict[str, Any]]:
    source_bbox = expand_bbox(
        union_alpha_bbox(sampled_frames),
        sampled_frames[0].size,
        margin_fraction=0.06,
    )
    normalized_frames = [
        contain_rgba(frame.crop(source_bbox), canvas_size, 0.88)
        for frame in sampled_frames
    ]
    prop_exclusions: list[Image.Image | None] = [None] * len(normalized_frames)
    prop_metadata: dict[str, Any] | None = None
    if separate_eating_prop:
        eating_exclusions, prop_metadata = make_eating_prop_exclusion_masks(
            normalized_frames
        )
        prop_exclusions = list(eating_exclusions)

    pose_frames: list[Image.Image] = []
    mask_frames: list[Image.Image] = []
    alpha_fractions: list[float] = []
    identity_fractions: list[float] = []
    prop_fractions: list[float] = []
    for normalized, prop_exclusion in zip(
        normalized_frames,
        prop_exclusions,
        strict=True,
    ):
        alpha = normalized.getchannel("A")
        hard_alpha = alpha.point(lambda value: 255 if value >= 16 else 0)
        hard_alpha = hard_alpha.filter(ImageFilter.MaxFilter(5))

        identity_alpha = hard_alpha
        if prop_exclusion is not None:
            prop_array = np.asarray(prop_exclusion, dtype=np.uint8) >= 128
            identity_array = (
                np.asarray(hard_alpha, dtype=np.uint8) >= 128
            ) & ~prop_array
            identity_alpha = Image.fromarray(
                identity_array.astype(np.uint8) * 255,
                mode="L",
            )
            prop_fractions.append(
                float(np.count_nonzero(prop_array))
                / float(canvas_size[0] * canvas_size[1])
            )

        pose_canvas = Image.new("RGB", canvas_size, pose_background)
        pose_canvas.paste(normalized.convert("RGB"), mask=alpha)
        pose_frames.append(pose_canvas)

        # SCAIL single-identity palette: replacement-video background is white,
        # while the foreground identity is pure blue.
        colored_mask = Image.new("RGB", canvas_size, (255, 255, 255))
        colored_mask.paste(
            Image.new("RGB", canvas_size, (0, 0, 255)),
            mask=identity_alpha,
        )
        if prop_exclusion is not None:
            colored_mask.paste(
                Image.new("RGB", canvas_size, (255, 0, 0)),
                mask=prop_exclusion,
            )
        mask_frames.append(colored_mask)
        alpha_fractions.append(
            float(np.count_nonzero(np.asarray(hard_alpha))) / float(canvas_size[0] * canvas_size[1])
        )
        identity_fractions.append(
            float(np.count_nonzero(np.asarray(identity_alpha)))
            / float(canvas_size[0] * canvas_size[1])
        )
    metadata = {
        "source_union_bbox": list(source_bbox),
        "subject_fraction_min": min(alpha_fractions),
        "subject_fraction_max": max(alpha_fractions),
        "subject_fraction_mean": sum(alpha_fractions) / len(alpha_fractions),
        "identity_fraction_min": min(identity_fractions),
        "identity_fraction_max": max(identity_fractions),
        "identity_fraction_mean": sum(identity_fractions) / len(identity_fractions),
        "scene_prop_separated": separate_eating_prop,
    }
    if separate_eating_prop:
        metadata.update(
            {
                "scene_prop_fraction_min": min(prop_fractions),
                "scene_prop_fraction_max": max(prop_fractions),
                "scene_prop_fraction_mean": sum(prop_fractions) / len(prop_fractions),
                "scene_prop_metadata": prop_metadata,
            }
        )
    return pose_frames, mask_frames, metadata


def make_full_eating_pose_and_mask_frames(
    sampled_frames: list[Image.Image],
    canvas_size: tuple[int, int],
) -> tuple[list[Image.Image], list[Image.Image], dict[str, Any]]:
    """Keep and audit the complete eating pose while separating its food bowl."""
    pose_frames, mask_frames, metadata = make_pose_and_mask_frames(
        sampled_frames,
        canvas_size,
        pose_background=(255, 255, 255),
        separate_eating_prop=True,
    )
    identity_bboxes: list[list[int]] = []
    identity_centers_y: list[float] = []
    identity_heights: list[int] = []
    identity_masks: list[np.ndarray] = []
    for mask in mask_frames:
        pixels = np.asarray(mask.convert("RGB"), dtype=np.uint8)
        identity = np.all(pixels == (0, 0, 255), axis=2)
        left, top, right, bottom = binary_bbox(identity)
        identity_masks.append(identity)
        identity_bboxes.append([left, top, right, bottom])
        identity_centers_y.append(round((top + bottom - 1) / 2.0, 4))
        identity_heights.append(bottom - top)

    first_identity = identity_masks[0]
    height, width = first_identity.shape
    first_left, first_top, first_right, first_bottom = identity_bboxes[0]
    lower_half_y = (first_top + first_bottom) // 2
    lower_quarter_y = (first_top + 3 * first_bottom) // 4
    canvas_area = height * width
    lower_half_area = (height - lower_half_y) * width
    lower_quarter_area = (height - lower_quarter_y) * width
    if (
        canvas_area <= 0
        or lower_half_area <= 0
        or lower_quarter_area <= 0
    ):
        raise ValueError(
            "eating_full_pose_motion_unreliable: invalid_audit_region"
        )

    overall_change_ratios: list[float] = []
    lower_half_change_ratios: list[float] = []
    lower_quarter_change_ratios: list[float] = []
    for identity in identity_masks:
        changed = np.logical_xor(first_identity, identity)
        overall_change_ratios.append(
            float(np.count_nonzero(changed)) / float(canvas_area)
        )
        lower_half_change_ratios.append(
            float(np.count_nonzero(changed[lower_half_y:, :]))
            / float(lower_half_area)
        )
        lower_quarter_change_ratios.append(
            float(np.count_nonzero(changed[lower_quarter_y:, :]))
            / float(lower_quarter_area)
        )

    overall_change_max = max(overall_change_ratios)
    lower_half_change_max = max(lower_half_change_ratios)
    lower_quarter_change_max = max(lower_quarter_change_ratios)
    motion_gate_thresholds = {
        "overall_change_ratio_min": (
            EATING_FULL_POSE_OVERALL_CHANGE_RATIO_MIN
        ),
        "lower_half_change_ratio_min": (
            EATING_FULL_POSE_LOWER_HALF_CHANGE_RATIO_MIN
        ),
        "lower_quarter_change_ratio_min": (
            EATING_FULL_POSE_LOWER_QUARTER_CHANGE_RATIO_MIN
        ),
    }
    failed_motion_checks = [
        name
        for name, value, minimum in (
            (
                "overall",
                overall_change_max,
                EATING_FULL_POSE_OVERALL_CHANGE_RATIO_MIN,
            ),
            (
                "lower_half",
                lower_half_change_max,
                EATING_FULL_POSE_LOWER_HALF_CHANGE_RATIO_MIN,
            ),
            (
                "lower_quarter",
                lower_quarter_change_max,
                EATING_FULL_POSE_LOWER_QUARTER_CHANGE_RATIO_MIN,
            ),
        )
        if value < minimum
    ]
    if failed_motion_checks:
        raise ValueError(
            "eating_full_pose_motion_unreliable: "
            f"failed={','.join(failed_motion_checks)}; "
            f"overall_change_ratio_max={overall_change_max:.6f}; "
            f"lower_half_change_ratio_max={lower_half_change_max:.6f}; "
            f"lower_quarter_change_ratio_max={lower_quarter_change_max:.6f}; "
            f"thresholds={motion_gate_thresholds}"
        )

    metadata.update(
        {
            "policy": "full_driver_pose_with_separate_bowl_v1",
            "driver_signal_only": False,
            "driver_rgb_preserved": True,
            "driver_silhouette_preserved": True,
            "pose_geometry_scope": "full_subject_silhouette_per_frame",
            "identity_bbox_by_frame": identity_bboxes,
            "identity_center_y_span": round(
                max(identity_centers_y) - min(identity_centers_y),
                4,
            ),
            "identity_height_span": int(
                max(identity_heights) - min(identity_heights)
            ),
            "identity_motion_audit": {
                "policy": "first_frame_blue_identity_mask_xor_v1",
                "reference_frame_index": 0,
                "first_frame_identity_bbox": [
                    first_left,
                    first_top,
                    first_right,
                    first_bottom,
                ],
                "overall_normalization_area_px": canvas_area,
                "lower_half_region_y_start": lower_half_y,
                "lower_half_normalization_area_px": lower_half_area,
                "lower_quarter_region_y_start": lower_quarter_y,
                "lower_quarter_normalization_area_px": lower_quarter_area,
                "overall_change_ratio_by_frame": [
                    round(value, 6) for value in overall_change_ratios
                ],
                "overall_change_ratio_max": round(
                    overall_change_max,
                    6,
                ),
                "overall_change_ratio_max_frame_index": int(
                    np.argmax(overall_change_ratios)
                ),
                "lower_half_change_ratio_by_frame": [
                    round(value, 6) for value in lower_half_change_ratios
                ],
                "lower_half_change_ratio_max": round(
                    lower_half_change_max,
                    6,
                ),
                "lower_half_change_ratio_max_frame_index": int(
                    np.argmax(lower_half_change_ratios)
                ),
                "lower_quarter_change_ratio_by_frame": [
                    round(value, 6) for value in lower_quarter_change_ratios
                ],
                "lower_quarter_change_ratio_max": round(
                    lower_quarter_change_max,
                    6,
                ),
                "lower_quarter_change_ratio_max_frame_index": int(
                    np.argmax(lower_quarter_change_ratios)
                ),
                "thresholds": motion_gate_thresholds,
                "passed": True,
            },
        }
    )
    return pose_frames, mask_frames, metadata


def _largest_binary_component(
    mask: np.ndarray,
    *,
    minimum_area: int,
    error_prefix: str = "eating_geometry_guide_unreliable",
) -> np.ndarray:
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )
    candidates = [
        (int(stats[label, cv2.CC_STAT_AREA]), label)
        for label in range(1, count)
        if int(stats[label, cv2.CC_STAT_AREA]) >= minimum_area
    ]
    if not candidates:
        raise ValueError(
            f"{error_prefix}: no_large_foreground_component"
        )
    _area, selected_label = max(candidates)
    return labels == selected_label


def _binary_external_perimeter(mask: np.ndarray) -> float:
    contours, _hierarchy = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        return 0.0
    return float(max(cv2.arcLength(contour, True) for contour in contours))


def _odd_scaled_kernel(
    canvas_size: tuple[int, int],
    *,
    value_at_512: int,
    minimum: int,
) -> int:
    value = max(
        minimum,
        int(round(min(canvas_size) * float(value_at_512) / 512.0)),
    )
    return value if value % 2 else value + 1


def make_grooming_geometry_guide(
    sampled_frames: list[Image.Image],
    canvas_size: tuple[int, int],
) -> tuple[Image.Image, dict[str, Any]]:
    """Build a texture-free Image 1 for the Main_case grooming action.

    Frame selection uses only alpha-silhouette motion.  The output uses a
    compressed, strongly blurred, quantized neutral anatomy field so the real
    TM head/shoulder/raised-paw structure remains readable without preserving
    source color or fine coat identity.  No artificial C-hook, floating muzzle
    oval, vertical flat part marker, distance gradient, exact source luma,
    whisker, or high-frequency coat texture is copied into the guide.
    """
    error_prefix = "grooming_geometry_guide_unreliable"
    if len(sampled_frames) < 9:
        raise ValueError(f"{error_prefix}: fewer_than_9_frames")
    if len({frame.size for frame in sampled_frames}) != 1:
        raise ValueError(f"{error_prefix}: inconsistent_frame_size")

    source_bbox = expand_bbox(
        union_alpha_bbox(sampled_frames),
        sampled_frames[0].size,
        margin_fraction=0.06,
    )
    normalized_frames = [
        contain_rgba(frame.crop(source_bbox), canvas_size, 0.88)
        for frame in sampled_frames
    ]
    hard_masks = [
        np.asarray(frame.getchannel("A"), dtype=np.uint8) >= 16
        for frame in normalized_frames
    ]
    canvas_area = int(canvas_size[0] * canvas_size[1])
    subject_areas = np.asarray(
        [int(np.count_nonzero(mask)) for mask in hard_masks],
        dtype=np.int64,
    )
    subject_fractions = subject_areas.astype(np.float64) / float(canvas_area)
    if (
        np.any(subject_fractions < 0.02)
        or np.any(subject_fractions > 0.80)
    ):
        raise ValueError(
            f"{error_prefix}: invalid_foreground_fraction="
            f"{float(subject_fractions.min()):.4f}.."
            f"{float(subject_fractions.max()):.4f}"
        )

    union = np.logical_or.reduce(hard_masks)
    left, top, right, bottom = binary_bbox(union)
    subject_width = right - left
    subject_height = bottom - top
    if subject_width < 32 or subject_height < 32:
        raise ValueError(f"{error_prefix}: subject_bbox_too_small")

    upper_body_bottom = min(
        canvas_size[1],
        top + int(round(subject_height * 0.72)),
    )
    upper_body_region = np.zeros_like(union, dtype=bool)
    upper_body_region[top:upper_body_bottom, left:right] = True
    first_mask = hard_masks[0]
    frame_motion: list[dict[str, Any]] = []
    for index, mask in enumerate(hard_masks):
        changed = np.logical_xor(mask, first_mask)
        overall_pixels = int(np.count_nonzero(changed))
        upper_pixels = int(
            np.count_nonzero(changed & upper_body_region)
        )
        frame_motion.append(
            {
                "index": index,
                "overall_xor_pixels": overall_pixels,
                "upper_body_xor_pixels": upper_pixels,
                "overall_xor_ratio": overall_pixels / float(canvas_area),
                "upper_body_xor_ratio": upper_pixels / float(canvas_area),
                "upper_body_motion_share": (
                    upper_pixels / float(overall_pixels)
                    if overall_pixels
                    else 0.0
                ),
            }
        )
    selected_motion = max(
        frame_motion,
        key=lambda item: (
            int(item["upper_body_xor_pixels"]),
            int(item["overall_xor_pixels"]),
            -int(item["index"]),
        ),
    )
    motion_thresholds = {
        "overall_xor_ratio_min": 0.014,
        "upper_body_xor_ratio_min": 0.012,
        "upper_body_motion_share_min": 0.55,
    }
    failed_motion_gates = [
        name
        for name, value in (
            (
                "overall",
                float(selected_motion["overall_xor_ratio"])
                < motion_thresholds["overall_xor_ratio_min"],
            ),
            (
                "upper_body",
                float(selected_motion["upper_body_xor_ratio"])
                < motion_thresholds["upper_body_xor_ratio_min"],
            ),
            (
                "upper_body_share",
                float(selected_motion["upper_body_motion_share"])
                < motion_thresholds["upper_body_motion_share_min"],
            ),
        )
        if value
    ]
    if failed_motion_gates:
        raise ValueError(
            f"{error_prefix}: insufficient_raised_paw_head_motion="
            + ",".join(failed_motion_gates)
        )

    selected_index = int(selected_motion["index"])
    raw_subject = hard_masks[selected_index]
    raw_perimeter = _binary_external_perimeter(raw_subject)
    outline_kernel_size = _odd_scaled_kernel(
        canvas_size,
        value_at_512=9,
        minimum=5,
    )
    outline_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (outline_kernel_size, outline_kernel_size),
    )
    selected_subject = cv2.morphologyEx(
        raw_subject.astype(np.uint8),
        cv2.MORPH_CLOSE,
        outline_kernel,
        iterations=1,
    )
    selected_subject = cv2.morphologyEx(
        selected_subject,
        cv2.MORPH_OPEN,
        outline_kernel,
        iterations=1,
    )
    blur_kernel_size = _odd_scaled_kernel(
        canvas_size,
        value_at_512=21,
        minimum=9,
    )
    blur_sigma = max(1.5, min(canvas_size) * 4.0 / 512.0)
    selected_subject = cv2.GaussianBlur(
        selected_subject * 255,
        (blur_kernel_size, blur_kernel_size),
        blur_sigma,
    )
    selected_subject = selected_subject >= 136
    selected_subject = fill_internal_holes(selected_subject)
    selected_subject = _largest_binary_component(
        selected_subject,
        minimum_area=max(500, int(canvas_area * 0.01)),
        error_prefix=error_prefix,
    )
    subject_pixels = int(np.count_nonzero(selected_subject))
    foreground_fraction = subject_pixels / float(canvas_area)
    if not 0.02 <= foreground_fraction <= 0.80:
        raise ValueError(
            f"{error_prefix}: cleaned_foreground_fraction="
            f"{foreground_fraction:.4f}"
        )
    cleaned_perimeter = _binary_external_perimeter(selected_subject)
    if raw_perimeter <= 0.0 or cleaned_perimeter > raw_perimeter * 1.01:
        raise ValueError(f"{error_prefix}: outline_smoothing_failed")

    # The fixed Main_case grooming driver raises the viewer-left forepaw.  Use
    # a large, heavily blurred luma field only to locate the muzzle/paw action
    # corridor.  Its original luma values are discarded.  Below, the corridor
    # is rebuilt as two flat neutral semantic regions: a rounded muzzle and one
    # continuous shoulder-to-forearm-to-rounded-paw shape.  Keeping the regions
    # distinct prevents their source contact patch from becoming a C-shaped
    # object in Qwen while retaining the real TM mouth-contact relationship.
    contact_x0 = left + int(round(subject_width * 0.10))
    contact_x1 = left + int(round(subject_width * 0.58))
    contact_y0 = top + int(round(subject_height * 0.36))
    contact_y1 = top + int(round(subject_height * 0.74))
    contact_region = np.zeros_like(union, dtype=bool)
    contact_region[
        max(0, contact_y0) : min(canvas_size[1], contact_y1),
        max(0, contact_x0) : min(canvas_size[0], contact_x1),
    ] = True
    contact_region &= selected_subject
    if int(np.count_nonzero(contact_region)) < 500:
        raise ValueError(f"{error_prefix}: contact_region_too_small")

    selected_rgb = np.asarray(
        normalized_frames[selected_index].convert("RGB"),
        dtype=np.uint8,
    )
    selected_gray = cv2.cvtColor(selected_rgb, cv2.COLOR_RGB2GRAY).astype(
        np.float32
    )
    raw_support = raw_subject.astype(np.float32)
    structure_sigma = max(2.0, min(canvas_size) * 6.0 / 512.0)
    weighted_luma = cv2.GaussianBlur(
        selected_gray * raw_support,
        (0, 0),
        sigmaX=structure_sigma,
        sigmaY=structure_sigma,
    )
    blurred_support = cv2.GaussianBlur(
        raw_support,
        (0, 0),
        sigmaX=structure_sigma,
        sigmaY=structure_sigma,
    )
    low_frequency_luma = weighted_luma / np.maximum(
        blurred_support,
        0.01,
    )
    contact_luma = low_frequency_luma[contact_region]
    luma_p10, luma_threshold, luma_p90 = np.percentile(
        contact_luma,
        (10, 65, 90),
    )
    luma_contrast = float(luma_p90 - luma_p10)
    if luma_contrast < 20.0:
        raise ValueError(
            f"{error_prefix}: low_frequency_contact_contrast="
            f"{luma_contrast:.3f}"
        )

    contact_component = (
        (low_frequency_luma >= float(luma_threshold))
        & contact_region
    ).astype(np.uint8)
    contact_close_size = _odd_scaled_kernel(
        canvas_size,
        value_at_512=9,
        minimum=5,
    )
    contact_open_size = _odd_scaled_kernel(
        canvas_size,
        value_at_512=5,
        minimum=3,
    )
    contact_component = cv2.morphologyEx(
        contact_component,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (contact_close_size, contact_close_size),
        ),
        iterations=1,
    )
    contact_component = cv2.morphologyEx(
        contact_component,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (contact_open_size, contact_open_size),
        ),
        iterations=1,
    )
    contact_component = fill_internal_holes(contact_component)
    contact_component = _largest_binary_component(
        contact_component,
        minimum_area=max(180, int(subject_pixels * 0.004)),
        error_prefix=error_prefix,
    )
    contact_component &= selected_subject
    contact_component = _largest_binary_component(
        contact_component,
        minimum_area=max(180, int(subject_pixels * 0.004)),
        error_prefix=error_prefix,
    )
    contact_pixels = int(np.count_nonzero(contact_component))
    contact_left, contact_top, contact_right, contact_bottom = binary_bbox(
        contact_component
    )
    contact_width_ratio = (
        contact_right - contact_left
    ) / float(subject_width)
    contact_height_ratio = (
        contact_bottom - contact_top
    ) / float(subject_height)
    contact_fraction = contact_pixels / float(subject_pixels)
    split_y = top + int(round(subject_height * 0.52))
    upper_muzzle_support = int(
        np.count_nonzero(
            contact_component[: min(canvas_size[1], split_y), :]
        )
    )
    lower_paw_support = int(
        np.count_nonzero(
            contact_component[max(0, split_y) :, :]
        )
    )
    minimum_contact_support = max(128, int(contact_pixels * 0.15))
    contact_failures: list[str] = []
    if not 0.025 <= contact_fraction <= 0.22:
        contact_failures.append("area")
    if not 0.12 <= contact_width_ratio <= 0.50:
        contact_failures.append("width")
    if not 0.22 <= contact_height_ratio <= 0.58:
        contact_failures.append("height")
    if contact_top > top + int(round(subject_height * 0.50)):
        contact_failures.append("muzzle_height")
    if contact_bottom < top + int(round(subject_height * 0.62)):
        contact_failures.append("paw_height")
    if upper_muzzle_support < minimum_contact_support:
        contact_failures.append("muzzle_support")
    if lower_paw_support < minimum_contact_support:
        contact_failures.append("paw_support")
    if contact_failures:
        raise ValueError(
            f"{error_prefix}: invalid_muzzle_raised_paw_relation="
            + ",".join(contact_failures)
        )

    rows = np.indices(selected_subject.shape)[0]
    semantic_muzzle_height = max(
        9,
        int(round(subject_height * 0.10)),
    )
    semantic_muzzle_y1 = min(
        contact_bottom,
        contact_top + semantic_muzzle_height,
    )
    muzzle_seed = (
        contact_component.astype(bool)
        & (rows >= contact_top)
        & (rows < semantic_muzzle_y1)
    )
    muzzle_seed_pixels = int(np.count_nonzero(muzzle_seed))
    minimum_semantic_pixels = max(96, int(subject_pixels * 0.002))
    if muzzle_seed_pixels < minimum_semantic_pixels:
        raise ValueError(f"{error_prefix}: muzzle_semantic_seed_too_small")

    muzzle_seed_left, muzzle_seed_top, muzzle_seed_right, muzzle_seed_bottom = (
        binary_bbox(muzzle_seed)
    )
    muzzle_region = np.zeros_like(selected_subject, dtype=np.uint8)
    muzzle_center = (
        (muzzle_seed_left + muzzle_seed_right - 1) // 2,
        (muzzle_seed_top + muzzle_seed_bottom - 1) // 2,
    )
    muzzle_axes = (
        max(5, (muzzle_seed_right - muzzle_seed_left) // 2),
        max(4, (muzzle_seed_bottom - muzzle_seed_top) // 2),
    )
    cv2.ellipse(
        muzzle_region,
        muzzle_center,
        muzzle_axes,
        0.0,
        0.0,
        360.0,
        1,
        thickness=-1,
    )
    muzzle_region = muzzle_region.astype(bool) & selected_subject
    muzzle_region = _largest_binary_component(
        muzzle_region,
        minimum_area=minimum_semantic_pixels,
        error_prefix=error_prefix,
    )

    # Estimate the raised foreleg axis from only the lower part of the detected
    # action corridor.  This excludes the horizontal muzzle branch that formed
    # the old hook.  A deterministic capsule then connects that shoulder anchor
    # to one rounded paw at the mouth.
    lower_axis_seed = (
        contact_component.astype(bool)
        & (rows >= max(contact_top, semantic_muzzle_y1 - 2))
    )
    lower_axis_y0 = max(
        contact_top,
        contact_bottom
        - max(
            7,
            int(round((contact_bottom - contact_top) * 0.22)),
        ),
    )
    shoulder_seed = lower_axis_seed & (rows >= lower_axis_y0)
    if int(np.count_nonzero(lower_axis_seed)) < minimum_semantic_pixels:
        raise ValueError(f"{error_prefix}: raised_foreleg_semantic_seed_too_small")
    if int(np.count_nonzero(shoulder_seed)) < max(24, minimum_semantic_pixels // 3):
        raise ValueError(f"{error_prefix}: raised_foreleg_shoulder_seed_too_small")

    _, lower_x = np.nonzero(lower_axis_seed)
    shoulder_y_values, shoulder_x_values = np.nonzero(shoulder_seed)
    paw_axis_x = int(round(float(np.median(lower_x))))
    shoulder_x = int(round(float(np.median(shoulder_x_values))))
    shoulder_y = min(
        contact_bottom - 1,
        int(round(float(np.percentile(shoulder_y_values, 90)))),
    )
    foreleg_width = max(13, int(round(subject_width * 0.115)))
    foreleg_radius = max(6, foreleg_width // 2)
    paw_radius_x = max(foreleg_radius, int(round(subject_width * 0.070)))
    paw_radius_y = max(7, int(round(subject_height * 0.050)))
    paw_center_y = min(
        contact_bottom - paw_radius_y - 1,
        max(
            muzzle_seed_bottom + paw_radius_y - 2,
            semantic_muzzle_y1 + paw_radius_y - 3,
        ),
    )
    paw_center_x = int(
        np.clip(
            paw_axis_x,
            muzzle_seed_left,
            max(muzzle_seed_left, muzzle_seed_right - 1),
        )
    )
    elbow_x = int(round((paw_center_x * 2.0 + shoulder_x) / 3.0))
    elbow_y = int(round((paw_center_y + shoulder_y) / 2.0))

    raised_foreleg_region = np.zeros_like(selected_subject, dtype=np.uint8)
    cv2.line(
        raised_foreleg_region,
        (shoulder_x, shoulder_y),
        (elbow_x, elbow_y),
        1,
        thickness=foreleg_width,
        lineType=cv2.LINE_8,
    )
    cv2.line(
        raised_foreleg_region,
        (elbow_x, elbow_y),
        (paw_center_x, paw_center_y),
        1,
        thickness=foreleg_width,
        lineType=cv2.LINE_8,
    )
    cv2.ellipse(
        raised_foreleg_region,
        (paw_center_x, paw_center_y),
        (paw_radius_x, paw_radius_y),
        0.0,
        0.0,
        360.0,
        1,
        thickness=-1,
    )
    raised_foreleg_region = (
        raised_foreleg_region.astype(bool)
        & selected_subject
        & ~muzzle_region
    )
    raised_foreleg_region = _largest_binary_component(
        raised_foreleg_region,
        minimum_area=minimum_semantic_pixels,
        error_prefix=error_prefix,
    )

    muzzle_pixels = int(np.count_nonzero(muzzle_region))
    raised_foreleg_pixels = int(np.count_nonzero(raised_foreleg_region))
    muzzle_bbox = binary_bbox(muzzle_region)
    raised_foreleg_bbox = binary_bbox(raised_foreleg_region)
    semantic_union = muzzle_region | raised_foreleg_region
    semantic_union_pixels = int(np.count_nonzero(semantic_union))
    semantic_contact_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3),
    )
    raised_foreleg_dilated = cv2.dilate(
        raised_foreleg_region.astype(np.uint8),
        semantic_contact_kernel,
        iterations=1,
    ).astype(bool)
    semantic_contact_pixels = int(
        np.count_nonzero(raised_foreleg_dilated & muzzle_region)
    )
    minimum_semantic_contact = max(3, foreleg_width // 6)
    semantic_failures: list[str] = []
    if muzzle_pixels < minimum_semantic_pixels:
        semantic_failures.append("muzzle_area")
    if raised_foreleg_pixels < minimum_semantic_pixels:
        semantic_failures.append("raised_foreleg_area")
    if semantic_contact_pixels < minimum_semantic_contact:
        semantic_failures.append("mouth_contact")
    if raised_foreleg_bbox[1] > muzzle_bbox[3]:
        semantic_failures.append("paw_height")
    if raised_foreleg_bbox[3] < contact_bottom - foreleg_radius:
        semantic_failures.append("shoulder_reach")
    if np.any(muzzle_region & raised_foreleg_region):
        semantic_failures.append("regions_overlap")
    if semantic_failures:
        raise ValueError(
            f"{error_prefix}: invalid_split_muzzle_raised_foreleg_semantics="
            + ",".join(semantic_failures)
        )

    # Compress the already heavily blurred source luminance into a narrow,
    # quantized neutral range.  This retains only low-frequency anatomy (the
    # actual head, shoulder and raised paw) and avoids the artificial separated
    # oval/capsule shapes that Qwen interpreted as a scarf, tie or brush.
    anatomy_values = low_frequency_luma[selected_subject]
    anatomy_p05, anatomy_p95 = np.percentile(anatomy_values, (5, 95))
    anatomy_span = float(anatomy_p95 - anatomy_p05)
    if anatomy_span < 16.0:
        raise ValueError(f"{error_prefix}: low_frequency_anatomy_contrast")
    normalized_anatomy = np.clip(
        (low_frequency_luma - float(anatomy_p05)) / anatomy_span,
        0.0,
        1.0,
    )
    anatomy_gray_min = 120
    anatomy_gray_max = 168
    anatomy_quantum = 8
    anatomy_gray = (
        anatomy_gray_min
        + normalized_anatomy * (anatomy_gray_max - anatomy_gray_min)
    )
    anatomy_gray = (
        np.round(anatomy_gray / anatomy_quantum) * anatomy_quantum
    ).astype(np.uint8)
    anatomy_gray = np.clip(
        anatomy_gray,
        anatomy_gray_min,
        anatomy_gray_max,
    ).astype(np.uint8)
    guide = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
    guide[:] = (0, 255, 0)
    guide[selected_subject] = np.repeat(
        anatomy_gray[selected_subject][:, None],
        3,
        axis=1,
    )

    green_background = np.all(guide == (0, 255, 0), axis=2)
    if not np.all(green_background[~selected_subject]):
        raise ValueError(f"{error_prefix}: background_not_pure_green")
    foreground_rgb = guide[selected_subject]
    if (
        np.any(foreground_rgb[:, 0] != foreground_rgb[:, 1])
        or np.any(foreground_rgb[:, 1] != foreground_rgb[:, 2])
    ):
        raise ValueError(f"{error_prefix}: non_neutral_foreground")

    metadata = {
        "policy": "driving_grooming_geometry_image1_v3",
        "method": (
            "upper_body_motion_keyframe_with_compressed_anatomy_field_v3"
        ),
        "selected_frame_index": selected_index,
        "source_union_bbox": list(source_bbox),
        "normalized_subject_union_bbox": [left, top, right, bottom],
        "subject_bbox": list(binary_bbox(selected_subject)),
        "subject_pixels": subject_pixels,
        "foreground_fraction": round(foreground_fraction, 6),
        "action_motion_audit": {
            "reference_frame_index": 0,
            "upper_body_region_bbox": [
                left,
                top,
                right,
                upper_body_bottom,
            ],
            "overall_xor_pixels": int(
                selected_motion["overall_xor_pixels"]
            ),
            "upper_body_xor_pixels": int(
                selected_motion["upper_body_xor_pixels"]
            ),
            "overall_xor_ratio": round(
                float(selected_motion["overall_xor_ratio"]),
                6,
            ),
            "upper_body_xor_ratio": round(
                float(selected_motion["upper_body_xor_ratio"]),
                6,
            ),
            "upper_body_motion_share": round(
                float(selected_motion["upper_body_motion_share"]),
                6,
            ),
            "thresholds": motion_thresholds,
            "passed": True,
        },
        "raised_paw_muzzle_relation": {
            "policy": (
                "source_derived_muzzle_foreleg_geometry_audit_v3"
            ),
            "search_region_bbox": [
                contact_x0,
                contact_y0,
                contact_x1,
                contact_y1,
            ],
            "component_bbox": [
                contact_left,
                contact_top,
                contact_right,
                contact_bottom,
            ],
            "component_pixels": semantic_union_pixels,
            "component_fraction_of_subject": round(
                semantic_union_pixels / float(subject_pixels),
                6,
            ),
            "component_width_ratio": round(contact_width_ratio, 6),
            "component_height_ratio": round(contact_height_ratio, 6),
            "upper_muzzle_support_pixels": muzzle_pixels,
            "lower_raised_paw_support_pixels": raised_foreleg_pixels,
            "minimum_support_pixels": minimum_contact_support,
            "muzzle_region_bbox": list(muzzle_bbox),
            "muzzle_region_pixels": muzzle_pixels,
            "raised_foreleg_region_bbox": list(raised_foreleg_bbox),
            "raised_foreleg_region_pixels": raised_foreleg_pixels,
            "semantic_contact_pixels": semantic_contact_pixels,
            "minimum_semantic_contact_pixels": minimum_semantic_contact,
            "regions_disjoint": True,
            "shoulder_anchor": [shoulder_x, shoulder_y],
            "paw_center": [paw_center_x, paw_center_y],
            "output_encoding": {
                "method": "compressed_blurred_quantized_neutral_anatomy",
                "gray_min": anatomy_gray_min,
                "gray_max": anatomy_gray_max,
                "quantum": anatomy_quantum,
                "source_luma_p05": round(float(anatomy_p05), 3),
                "source_luma_p95": round(float(anatomy_p95), 3),
            },
            "c_hook_component_preserved": False,
            "black_or_tool_semantics_preserved": False,
            "low_frequency_luma_p10": round(float(luma_p10), 3),
            "low_frequency_luma_threshold_p65": round(
                float(luma_threshold),
                3,
            ),
            "low_frequency_luma_p90": round(float(luma_p90), 3),
            "low_frequency_luma_contrast": round(luma_contrast, 3),
            "gaussian_sigma": round(structure_sigma, 3),
            "source_luma_values_preserved": False,
            "passed": True,
        },
        "outline_smoothing": {
            "morphology_kernel": outline_kernel_size,
            "gaussian_kernel": blur_kernel_size,
            "gaussian_sigma": round(blur_sigma, 3),
            "raw_perimeter": round(raw_perimeter, 3),
            "cleaned_perimeter": round(cleaned_perimeter, 3),
            "perimeter_ratio": round(
                cleaned_perimeter / max(raw_perimeter, 1.0),
                6,
            ),
            "high_frequency_outline_preserved": False,
        },
        "palette": {
            "background": [0, 255, 0],
            "subject_gray_range": [anatomy_gray_min, anatomy_gray_max],
            "quantum": anatomy_quantum,
            "source_rgb_preserved": False,
            "source_chroma_preserved": False,
            "source_luma_values_preserved": False,
            "low_frequency_anatomy_preserved": True,
            "source_marking_contrast_preserved": False,
            "high_frequency_texture_preserved": False,
            "distance_field_preserved": False,
            "artificial_part_markers_preserved": False,
            "black_foreground_preserved": False,
        },
    }
    return Image.fromarray(guide, mode="RGB"), metadata


def make_eating_geometry_guide(
    sampled_frames: list[Image.Image],
    canvas_size: tuple[int, int],
) -> tuple[Image.Image, dict[str, Any]]:
    """Build an identity-free Image 2 that retains only eating action geometry.

    The guide deliberately contains no source RGB pixels. A heavily blurred,
    compressed neutral-gray field describes low-frequency head/muzzle anatomy,
    while fixed semantic beige/brown fields describe the bowl and kibble. Every
    remaining pixel is pure green. This removes coat color, collar, tag,
    watermark and fur texture while retaining readable eating geometry.
    """
    if len(sampled_frames) < 9:
        raise ValueError(
            "eating_geometry_guide_unreliable: fewer_than_9_frames"
        )
    if len({frame.size for frame in sampled_frames}) != 1:
        raise ValueError(
            "eating_geometry_guide_unreliable: inconsistent_frame_size"
        )

    source_bbox = expand_bbox(
        union_alpha_bbox(sampled_frames),
        sampled_frames[0].size,
        margin_fraction=0.06,
    )
    normalized_frames = [
        contain_rgba(frame.crop(source_bbox), canvas_size, 0.88)
        for frame in sampled_frames
    ]
    prop_exclusions, prop_metadata = make_eating_prop_exclusion_masks(
        normalized_frames
    )
    prop_union = np.logical_or.reduce(
        [np.asarray(mask, dtype=np.uint8) >= 128 for mask in prop_exclusions]
    )
    prop_union = cv2.morphologyEx(
        prop_union.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((5, 5), dtype=np.uint8),
        iterations=2,
    ).astype(bool)
    prop_union = fill_internal_holes(prop_union)
    prop_union = _largest_binary_component(prop_union, minimum_area=100)
    bowl_left, bowl_top, bowl_right, bowl_bottom = binary_bbox(prop_union)
    bowl_width = bowl_right - bowl_left
    bowl_height = bowl_bottom - bowl_top
    if bowl_width < 24 or bowl_height < 8:
        raise ValueError(
            "eating_geometry_guide_unreliable: bowl_component_too_small"
        )

    # Paws normally sit beside the bowl.  Restrict the contact measurement to
    # the central opening so a planted paw cannot masquerade as a low muzzle.
    contact_x0 = bowl_left + int(round(bowl_width * 0.20))
    contact_x1 = bowl_right - int(round(bowl_width * 0.20))
    contact_y0 = max(0, bowl_top - max(48, int(round(bowl_height * 1.5))))
    contact_y1 = min(canvas_size[1], bowl_top + 3)
    head_x0 = max(0, bowl_left - int(round(bowl_width * 0.25)))
    head_x1 = min(canvas_size[0], bowl_right + int(round(bowl_width * 0.25)))
    if contact_x1 <= contact_x0 or contact_y1 <= contact_y0:
        raise ValueError(
            "eating_geometry_guide_unreliable: invalid_muzzle_contact_region"
        )
    if head_x1 <= head_x0:
        raise ValueError(
            "eating_geometry_guide_unreliable: invalid_head_lowering_region"
        )

    frame_records: list[dict[str, Any]] = []
    subject_masks: list[np.ndarray] = []
    for index, (frame, prop_exclusion) in enumerate(
        zip(normalized_frames, prop_exclusions, strict=True)
    ):
        hard_alpha = np.asarray(frame.getchannel("A"), dtype=np.uint8) >= 16
        frame_prop = np.asarray(prop_exclusion, dtype=np.uint8) >= 128
        subject = hard_alpha & ~frame_prop
        subject_masks.append(subject)
        contact_region = subject[
            contact_y0:contact_y1,
            contact_x0:contact_x1,
        ]
        rows, _columns = np.where(contact_region)
        if rows.size:
            lowest_subject_y = contact_y0 + int(rows.max())
            gap = max(0, bowl_top - lowest_subject_y - 1)
            contact_pixels = int(
                np.count_nonzero(
                    contact_region[
                        max(0, bowl_top - 5 - contact_y0) :,
                        :,
                    ]
                )
            )
        else:
            lowest_subject_y = None
            gap = canvas_size[1]
            contact_pixels = 0
        head_rows, _head_columns = np.where(
            subject[:bowl_top, head_x0:head_x1]
        )
        if head_rows.size:
            head_y_p10 = float(np.percentile(head_rows, 10))
            head_support_pixels = int(head_rows.size)
        else:
            head_y_p10 = float("-inf")
            head_support_pixels = 0
        frame_records.append(
            {
                "index": index,
                "muzzle_bowl_gap_px": gap,
                "contact_pixels": contact_pixels,
                "lowest_subject_y": lowest_subject_y,
                "head_y_p10": head_y_p10,
                "head_support_pixels": head_support_pixels,
            }
        )

    contact_candidates = [
        item
        for item in frame_records
        if int(item["muzzle_bowl_gap_px"]) <= 4
        and int(item["contact_pixels"]) >= 4
        and np.isfinite(float(item["head_y_p10"]))
    ]
    if not contact_candidates:
        raise ValueError(
            "eating_geometry_guide_unreliable: no_head_lowered_bowl_contact_frame"
        )
    head_y_values = [
        float(item["head_y_p10"])
        for item in frame_records
        if np.isfinite(float(item["head_y_p10"]))
    ]
    head_lowering_span = max(head_y_values) - min(head_y_values)
    head_lowering_span_min = max(6.0, canvas_size[1] * 0.012)
    if head_lowering_span < head_lowering_span_min:
        raise ValueError(
            "eating_geometry_guide_unreliable: insufficient_head_lowering="
            f"{head_lowering_span:.3f}"
        )
    # The bowl and planted paws touch in every Main_case frame, so contact
    # pixels alone cannot identify eating and previously selected a head-up
    # frame. The lower 10th-percentile of the central upper silhouette tracks
    # the ears/head moving down while ignoring the static paws and bowl.
    selected = max(
        contact_candidates,
        key=lambda item: (
            float(item["head_y_p10"]),
            int(item["contact_pixels"]),
            -int(item["index"]),
        ),
    )

    selected_subject = subject_masks[int(selected["index"])] & ~prop_union
    raw_subject_perimeter = _binary_external_perimeter(selected_subject)
    outline_kernel_size = _odd_scaled_kernel(
        canvas_size,
        value_at_512=9,
        minimum=5,
    )
    outline_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (outline_kernel_size, outline_kernel_size),
    )
    selected_subject = cv2.morphologyEx(
        selected_subject.astype(np.uint8),
        cv2.MORPH_CLOSE,
        outline_kernel,
        iterations=1,
    )
    selected_subject = cv2.morphologyEx(
        selected_subject,
        cv2.MORPH_OPEN,
        outline_kernel,
        iterations=1,
    )
    # Smooth away whiskers, narrow tag connections, and high-frequency coat
    # spikes while retaining ears, head, torso, paws, and the low eating pose.
    blur_kernel_size = _odd_scaled_kernel(
        canvas_size,
        value_at_512=21,
        minimum=9,
    )
    blur_sigma = max(1.5, min(canvas_size) * 4.0 / 512.0)
    selected_subject = cv2.GaussianBlur(
        selected_subject * 255,
        (blur_kernel_size, blur_kernel_size),
        blur_sigma,
    )
    selected_subject = selected_subject >= 136
    selected_subject = fill_internal_holes(selected_subject)
    selected_subject = _largest_binary_component(
        selected_subject,
        minimum_area=max(500, int(canvas_size[0] * canvas_size[1] * 0.01)),
    )
    # Keep the true source-anchored muzzle immediately adjacent to the food.
    # Removing a dilated prop mask created an artificial air gap; only the actual
    # prop pixels are excluded here. The subject remains a single neutral field.
    selected_subject &= ~prop_union
    if int(np.count_nonzero(selected_subject)) < 500:
        raise ValueError(
            "eating_geometry_guide_unreliable: cleaned_subject_too_small"
        )
    cleaned_subject_perimeter = _binary_external_perimeter(selected_subject)
    if (
        raw_subject_perimeter <= 0.0
        or cleaned_subject_perimeter > raw_subject_perimeter * 1.01
    ):
        raise ValueError(
            "eating_geometry_guide_unreliable: outline_smoothing_failed"
        )

    # Preserve only heavily blurred, compressed luminance from the selected
    # contact frame. A flat silhouette erased the ears/head/muzzle relationship
    # and let Qwen reinterpret the animal as a head-up cat looking at the bowl.
    # Weighted blurring avoids pulling pure-green background into the anatomy;
    # the narrow quantized neutral range removes source color, markings and fur
    # texture while keeping the low-frequency eating geometry readable.
    selected_rgb = np.asarray(
        normalized_frames[int(selected["index"])].convert("RGB"),
        dtype=np.uint8,
    )
    selected_luma = cv2.cvtColor(
        selected_rgb,
        cv2.COLOR_RGB2GRAY,
    ).astype(np.float32)
    anatomy_support = selected_subject.astype(np.float32)
    anatomy_sigma = max(4.0, min(canvas_size) * 10.0 / 512.0)
    anatomy_weighted_luma = cv2.GaussianBlur(
        selected_luma * anatomy_support,
        (0, 0),
        sigmaX=anatomy_sigma,
        sigmaY=anatomy_sigma,
    )
    anatomy_blurred_support = cv2.GaussianBlur(
        anatomy_support,
        (0, 0),
        sigmaX=anatomy_sigma,
        sigmaY=anatomy_sigma,
    )
    low_frequency_anatomy = anatomy_weighted_luma / np.maximum(
        anatomy_blurred_support,
        0.01,
    )
    anatomy_values = low_frequency_anatomy[selected_subject]
    anatomy_p05, anatomy_p95 = np.percentile(anatomy_values, (5, 95))
    anatomy_span = float(anatomy_p95 - anatomy_p05)
    if anatomy_span < 6.0:
        raise ValueError(
            "eating_geometry_guide_unreliable: "
            f"low_frequency_anatomy_contrast={anatomy_span:.3f}"
        )
    normalized_anatomy = np.clip(
        (low_frequency_anatomy - float(anatomy_p05)) / anatomy_span,
        0.0,
        1.0,
    )
    anatomy_gray_min = 120
    anatomy_gray_max = 168
    anatomy_quantum = 8
    anatomy_gray = (
        anatomy_gray_min
        + normalized_anatomy * (anatomy_gray_max - anatomy_gray_min)
    )
    anatomy_gray = (
        np.round(anatomy_gray / anatomy_quantum) * anatomy_quantum
    ).astype(np.uint8)
    anatomy_gray = np.clip(
        anatomy_gray,
        anatomy_gray_min,
        anatomy_gray_max,
    ).astype(np.uint8)
    anatomy_unique_values = np.unique(anatomy_gray[selected_subject])
    if anatomy_unique_values.size < 2:
        raise ValueError(
            "eating_geometry_guide_unreliable: "
            "compressed_anatomy_field_is_flat"
        )

    median_rgb = np.median(
        np.stack(
            [
                np.asarray(frame.convert("RGB"), dtype=np.float32)
                for frame in normalized_frames
            ]
        ),
        axis=0,
    ).astype(np.uint8)
    median_luma = cv2.cvtColor(median_rgb, cv2.COLOR_RGB2GRAY).astype(
        np.float32
    )
    bowl_support = prop_union.astype(np.float32)
    bowl_structure_sigma = max(1.0, min(canvas_size) * 1.5 / 512.0)
    bowl_weighted_luma = cv2.GaussianBlur(
        median_luma * bowl_support,
        (0, 0),
        sigmaX=bowl_structure_sigma,
        sigmaY=bowl_structure_sigma,
    )
    bowl_blurred_support = cv2.GaussianBlur(
        bowl_support,
        (0, 0),
        sigmaX=bowl_structure_sigma,
        sigmaY=bowl_structure_sigma,
    )
    bowl_low_frequency_luma = bowl_weighted_luma / np.maximum(
        bowl_blurred_support,
        0.01,
    )
    bowl_structure_values = bowl_low_frequency_luma[prop_union]
    bowl_luma_p10, bowl_luma_p90 = np.percentile(
        bowl_structure_values,
        (10, 90),
    )
    bowl_luma_contrast = float(bowl_luma_p90 - bowl_luma_p10)
    # Audit the real continuous muzzle tip after smoothing. No artificial ellipse,
    # point, hook, or detached semantic object is added: those shapes were read by
    # Qwen as a ball or tongue. Image 1 (the upload) owns identity; this Image 2
    # supplies only the real source-anchored contact geometry.
    contact_subject = selected_subject.copy()
    contact_subject[:contact_y0, :] = False
    contact_subject[min(canvas_size[1], bowl_top + 1) :, :] = False
    contact_subject[:, :contact_x0] = False
    contact_subject[:, contact_x1:] = False
    contact_rows, contact_columns = np.where(contact_subject)
    if contact_rows.size < 32:
        raise ValueError(
            "eating_geometry_guide_unreliable: muzzle_semantic_anchor_missing"
        )
    lowest_contact_y = int(contact_rows.max())
    muzzle_food_gap = max(0, bowl_top - lowest_contact_y - 1)
    if muzzle_food_gap > 4:
        raise ValueError(
            "eating_geometry_guide_unreliable: "
            f"continuous_muzzle_food_gap_px={muzzle_food_gap}"
        )
    contact_tip = contact_columns[contact_rows >= max(contact_y0, lowest_contact_y - 5)]
    if contact_tip.size < 8:
        raise ValueError(
            "eating_geometry_guide_unreliable: muzzle_semantic_tip_missing"
        )
    muzzle_center_x = int(round(float(np.median(contact_tip))))
    muzzle_tip_mask = (
        contact_subject
        & (np.indices(contact_subject.shape)[0] >= max(contact_y0, lowest_contact_y - 5))
    )
    muzzle_tip_bbox = binary_bbox(muzzle_tip_mask)

    # Preserve prop meaning without preserving any driver pixels. Fixed semantic
    # colors make the shallow bowl and brown kibble readable while being
    # independent of the Main_case animal identity and lighting.
    hsv = cv2.cvtColor(median_rgb, cv2.COLOR_RGB2HSV)
    row_grid = np.indices(prop_union.shape)[0]
    food_zone_bottom = min(
        canvas_size[1],
        bowl_top + max(8, int(round(bowl_height * 0.42))),
    )
    warm_food_seed = (
        prop_union
        & (row_grid >= max(0, bowl_top - 2))
        & (row_grid < food_zone_bottom)
        & (hsv[:, :, 0] >= 2)
        & (hsv[:, :, 0] <= 30)
        & (hsv[:, :, 1] >= 38)
        & (hsv[:, :, 2] >= 24)
        & (hsv[:, :, 2] <= 235)
    )
    food_core = cv2.morphologyEx(
        warm_food_seed.astype(np.uint8),
        cv2.MORPH_OPEN,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    ).astype(bool)
    minimum_food_pixels = max(24, int(round(bowl_width * bowl_height * 0.004)))
    if int(np.count_nonzero(food_core)) < minimum_food_pixels:
        upper_prop = (
            prop_union
            & (row_grid >= max(0, bowl_top - 2))
            & (row_grid < food_zone_bottom)
        )
        upper_luma = median_luma[upper_prop]
        if upper_luma.size:
            dark_cutoff = float(np.percentile(upper_luma, 42))
            food_core = upper_prop & (median_luma <= dark_cutoff)
    food_core &= prop_union
    food_core_pixels = int(np.count_nonzero(food_core))
    if food_core_pixels < minimum_food_pixels:
        raise ValueError(
            "eating_geometry_guide_unreliable: "
            f"semantic_food_core_too_small={food_core_pixels}"
        )

    bowl_semantic_rgb = np.array([214, 208, 196], dtype=np.uint8)
    bowl_rim_semantic_rgb = np.array([232, 228, 218], dtype=np.uint8)
    food_semantic_rgb = np.array([142, 86, 45], dtype=np.uint8)
    bowl_rim = (
        prop_union
        & (row_grid < bowl_top + max(5, int(round(bowl_height * 0.16))))
    )
    guide = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
    guide[:] = (0, 255, 0)
    guide[selected_subject] = np.repeat(
        anatomy_gray[selected_subject][:, None],
        3,
        axis=1,
    )
    guide[prop_union] = bowl_semantic_rgb
    guide[bowl_rim] = bowl_rim_semantic_rgb
    guide[food_core] = food_semantic_rgb

    metadata = {
        "policy": "driving_eating_geometry_image2_v6",
        "method": (
            "temporal_head_lowering_compressed_anatomy_semantic_food_fields_v7"
        ),
        "selected_frame_index": int(selected["index"]),
        "muzzle_bowl_gap_px": int(selected["muzzle_bowl_gap_px"]),
        "contact_pixels": int(selected["contact_pixels"]),
        "source_union_bbox": list(source_bbox),
        "subject_bbox": list(binary_bbox(selected_subject)),
        "bowl_bbox": [bowl_left, bowl_top, bowl_right, bowl_bottom],
        "contact_region_bbox": [contact_x0, contact_y0, contact_x1, contact_y1],
        "muzzle_contact_semantic": {
            "method": "source_anchored_continuous_neutral_muzzle_tip_v2",
            "bbox": list(muzzle_tip_bbox),
            "center_x": muzzle_center_x,
            "tip_y": lowest_contact_y,
            "muzzle_food_gap_px": muzzle_food_gap,
            "artificial_shape_added": False,
            "detached_marker_added": False,
            "source_rgb_preserved": False,
            "source_chroma_preserved": False,
            "identity_preserved": False,
            "passed": True,
        },
        "semantic_food": {
            "method": "fixed_beige_bowl_brown_kibble_from_source_mask_v1",
            "food_core_pixels": food_core_pixels,
            "minimum_food_core_pixels": minimum_food_pixels,
            "food_core_bbox": list(binary_bbox(food_core)),
            "bowl_rgb": bowl_semantic_rgb.tolist(),
            "bowl_rim_rgb": bowl_rim_semantic_rgb.tolist(),
            "food_rgb": food_semantic_rgb.tolist(),
            "source_rgb_preserved": False,
            "semantic_chroma_only": True,
            "passed": True,
        },
        "head_lowering_selection": {
            "method": "central_upper_subject_row_p10_v1",
            "search_region_bbox": [head_x0, 0, head_x1, bowl_top],
            "selected_head_y_p10": round(float(selected["head_y_p10"]), 3),
            "selected_head_support_pixels": int(
                selected["head_support_pixels"]
            ),
            "head_y_p10_min": round(min(head_y_values), 3),
            "head_y_p10_max": round(max(head_y_values), 3),
            "head_y_p10_span": round(head_lowering_span, 3),
            "head_y_p10_span_min": round(head_lowering_span_min, 3),
            "passed": True,
        },
        "subject_pixels": int(np.count_nonzero(selected_subject)),
        "bowl_pixels": int(np.count_nonzero(prop_union)),
        "outline_smoothing": {
            "morphology_kernel": outline_kernel_size,
            "gaussian_kernel": blur_kernel_size,
            "gaussian_sigma": round(blur_sigma, 3),
            "raw_perimeter": round(raw_subject_perimeter, 3),
            "cleaned_perimeter": round(cleaned_subject_perimeter, 3),
            "perimeter_ratio": round(
                cleaned_subject_perimeter / max(raw_subject_perimeter, 1.0),
                6,
            ),
            "high_frequency_outline_preserved": False,
        },
        "semantic_structure": {
            "method": "compressed_blurred_quantized_neutral_anatomy_v1",
            "subject_gray_range": [anatomy_gray_min, anatomy_gray_max],
            "subject_gray_values": anatomy_unique_values.tolist(),
            "subject_unique_rgb_count": int(anatomy_unique_values.size),
            "internal_luma_variation": True,
            "appearance_bearing_face_detail": False,
            "anatomy_gaussian_sigma": round(anatomy_sigma, 3),
            "source_luma_p05": round(float(anatomy_p05), 3),
            "source_luma_p95": round(float(anatomy_p95), 3),
            "source_luma_contrast": round(anatomy_span, 3),
            "quantum": anatomy_quantum,
            "bowl_gaussian_sigma": round(bowl_structure_sigma, 3),
            "bowl_luma_p10": round(float(bowl_luma_p10), 3),
            "bowl_luma_p90": round(float(bowl_luma_p90), 3),
            "bowl_luma_contrast": round(bowl_luma_contrast, 3),
            "source_luma_values_preserved": False,
            "source_rgb_preserved": False,
            "source_chroma_preserved": False,
            "high_frequency_texture_preserved": False,
        },
        "palette": {
            "background": [0, 255, 0],
            "subject_gray_range": [anatomy_gray_min, anatomy_gray_max],
            "subject_gray_values": anatomy_unique_values.tolist(),
            "quantum": anatomy_quantum,
            "prop_semantic_colors": {
                "bowl": bowl_semantic_rgb.tolist(),
                "bowl_rim": bowl_rim_semantic_rgb.tolist(),
                "food": food_semantic_rgb.tolist(),
            },
            "source_rgb_preserved": False,
            "source_chroma_preserved": False,
            "semantic_prop_chroma_preserved": True,
            "source_luma_values_preserved": False,
            "high_frequency_texture_preserved": False,
        },
        "prop_detection": prop_metadata,
    }
    return Image.fromarray(guide, mode="RGB"), metadata


def make_eating_geometry_guide_from_gif(
    source: Path,
    canvas_size: tuple[int, int],
    *,
    output_count: int,
    output_fps: float,
) -> tuple[Image.Image, dict[str, Any]]:
    frames, durations = read_animation_frames(source)
    sampled = sample_timed_frames(frames, durations, output_count, output_fps)
    guide, metadata = make_eating_geometry_guide(sampled, canvas_size)
    metadata.update(
        {
            "source": str(source),
            "source_sha256": sha256_file(source),
            "sampled_frame_count": output_count,
            "sampled_fps": output_fps,
            "selected_time_seconds": round(
                int(metadata["selected_frame_index"]) / output_fps,
                6,
            ),
        }
    )
    return guide, metadata


def make_grooming_geometry_guide_from_animation(
    source: Path,
    canvas_size: tuple[int, int],
    *,
    output_count: int,
    output_fps: float,
) -> tuple[Image.Image, dict[str, Any]]:
    frames, durations = read_animation_frames(source)
    sampled = sample_timed_frames(frames, durations, output_count, output_fps)
    white_maquette_source = "white_model" in source.stem.lower()
    white_maquette_selection: dict[str, Any] | None = None
    if white_maquette_source:
        midpoint = (len(sampled) - 1) / 2.0
        candidate_indices = sorted(
            {
                int(round((len(sampled) - 1) * fraction))
                for fraction in (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)
            },
            key=lambda index: (abs(index - midpoint), index),
        )
        rejected_candidates: list[dict[str, Any]] = []
        selected_source_index: int | None = None
        for candidate_index in candidate_indices:
            # The first white-maquette frame is the neutral motion reference.
            # Expose one candidate at its real timeline index so the existing
            # motion and anatomy gates stay unchanged while the old
            # whole-cycle maximum can no longer select late head-only motion
            # instead of the complete shoulder-to-paw relation.
            candidate_sequence = [sampled[0]] * len(sampled)
            candidate_sequence[candidate_index] = sampled[candidate_index]
            try:
                guide, metadata = make_grooming_geometry_guide(
                    candidate_sequence,
                    canvas_size,
                )
            except ValueError as exc:
                rejected_candidates.append(
                    {
                        "frame_index": candidate_index,
                        "reason": str(exc),
                    }
                )
                continue
            selected_source_index = candidate_index
            white_maquette_selection = {
                "policy": "midcycle_relation_gated_white_maquette_v1",
                "source_filename": source.name,
                "candidate_indices": candidate_indices,
                "selected_frame_index": candidate_index,
                "rejected_candidates": rejected_candidates,
                "reference_frame_index": 0,
                "identity_or_color_signal_used_for_candidate_ordering": False,
                "motion_and_muzzle_foreleg_relation_gates_reused": True,
            }
            break
        if selected_source_index is None:
            # Re-run the complete sequence to preserve the most informative
            # historical failure when none of the white-maquette anchors pass.
            guide, metadata = make_grooming_geometry_guide(sampled, canvas_size)
    else:
        guide, metadata = make_grooming_geometry_guide(sampled, canvas_size)
    if white_maquette_selection is not None:
        metadata["white_maquette_selection"] = white_maquette_selection
        metadata["source_white_maquette"] = True
    metadata.update(
        {
            "source": str(source),
            "source_sha256": sha256_file(source),
            "sampled_frame_count": output_count,
            "sampled_fps": output_fps,
            "selected_time_seconds": round(
                int(metadata["selected_frame_index"]) / output_fps,
                6,
            ),
        }
    )
    return guide, metadata


def make_eating_reference_aligned_pose_frames(
    sampled_frames: list[Image.Image],
    reference: Image.Image,
    reference_mask: Image.Image,
    canvas_size: tuple[int, int],
    *,
    maximum_head_lift_px: float = 18.0,
    target_muzzle_food_overlap_px: float = (
        EATING_TARGET_MUZZLE_FOOD_OVERLAP_PX
    ),
) -> tuple[list[Image.Image], list[Image.Image], dict[str, Any]]:
    """Animate the accepted eating reference in its own mouth/bowl coordinates.

    The source GIF contributes only a scalar low-head/high-head timing curve. The
    pose video is synthesized from the accepted target reference, so neither the
    driving cat's coat nor its side-view body silhouette can enter SCAIL.
    """
    if len(sampled_frames) < 9:
        raise ValueError(
            "eating_reference_motion_unreliable: fewer_than_9_frames"
        )
    if reference.size != canvas_size or reference_mask.size != canvas_size:
        raise ValueError(
            "eating_reference_motion_unreliable: reference_size_mismatch"
        )
    if maximum_head_lift_px <= 0:
        raise ValueError(
            "eating_reference_motion_unreliable: invalid_head_lift_limit"
        )
    if not (
        EATING_MUZZLE_FOOD_OVERLAP_MIN_PX
        <= target_muzzle_food_overlap_px
        <= EATING_MUZZLE_FOOD_OVERLAP_MAX_PX
    ):
        raise ValueError(
            "eating_reference_motion_unreliable: "
            "invalid_target_muzzle_food_overlap"
        )

    source_bbox = expand_bbox(
        union_alpha_bbox(sampled_frames),
        sampled_frames[0].size,
        margin_fraction=0.06,
    )
    normalized_driver = [
        contain_rgba(frame.crop(source_bbox), canvas_size, 0.88)
        for frame in sampled_frames
    ]
    driver_prop_exclusions, prop_metadata = make_eating_prop_exclusion_masks(
        normalized_driver
    )
    driver_prop_union = np.logical_or.reduce(
        [
            np.asarray(mask, dtype=np.uint8) >= 128
            for mask in driver_prop_exclusions
        ]
    )
    driver_bowl_left, driver_bowl_top, driver_bowl_right, _driver_bowl_bottom = (
        binary_bbox(driver_prop_union)
    )
    driver_bowl_width = driver_bowl_right - driver_bowl_left
    # The planted near paw in the real eating driver remains beside and partly over the
    # bowl opening in every frame.  Measuring the lowest subject pixel above the
    # bowl therefore produces a constant zero gap even while the head clearly
    # rises.  Track a robust upper-head quantile in a wider bowl-relative band
    # instead: paws occupy the bottom of this band and cannot pin its 10th-row
    # percentile, while the moving forehead and ears provide a strong signal.
    driver_head_x0 = max(
        0,
        driver_bowl_left - int(round(driver_bowl_width * 0.55)),
    )
    driver_head_x1 = min(
        canvas_size[0],
        driver_bowl_right + int(round(driver_bowl_width * 0.05)),
    )
    if driver_head_x1 - driver_head_x0 < 24:
        raise ValueError(
            "eating_reference_motion_unreliable: invalid_driver_head_band"
        )

    driver_head_y: list[float] = []
    driver_head_pixels: list[int] = []
    for frame, prop_exclusion in zip(
        normalized_driver,
        driver_prop_exclusions,
        strict=True,
    ):
        alpha = np.asarray(frame.getchannel("A"), dtype=np.uint8) >= 16
        frame_prop = np.asarray(prop_exclusion, dtype=np.uint8) >= 128
        subject = alpha & ~frame_prop
        head_band = subject[
            :driver_bowl_top,
            driver_head_x0:driver_head_x1,
        ]
        rows, _columns = np.where(head_band)
        driver_head_pixels.append(int(rows.size))
        if rows.size >= 256:
            # Larger y means the driver head is lower and closer to the bowl.
            head_y = float(np.percentile(rows, 10))
        else:
            head_y = float("nan")
        driver_head_y.append(head_y)

    finite_head_y = np.asarray(
        [value for value in driver_head_y if np.isfinite(value)],
        dtype=np.float32,
    )
    if finite_head_y.size < max(5, len(sampled_frames) // 2):
        raise ValueError(
            "eating_reference_motion_unreliable: insufficient_head_signal"
        )
    high_head_y = float(np.percentile(finite_head_y, 10))
    low_head_y = float(np.percentile(finite_head_y, 90))
    head_y_span = low_head_y - high_head_y
    if head_y_span < 4.0:
        raise ValueError(
            "eating_reference_motion_unreliable: head_motion_range_too_small"
        )
    lift_fractions = [
        float(
            np.clip(
                (low_head_y - np.clip(value, high_head_y, low_head_y))
                / head_y_span,
                0.0,
                1.0,
            )
        )
        if np.isfinite(value)
        else 1.0
        for value in driver_head_y
    ]

    reference_rgba = np.asarray(reference.convert("RGBA"), dtype=np.uint8)
    colored_reference_mask = np.asarray(
        reference_mask.convert("RGB"),
        dtype=np.uint8,
    )
    reference_subject = np.all(
        colored_reference_mask == (0, 0, 255),
        axis=2,
    )
    reference_bowl = np.all(
        colored_reference_mask == (255, 0, 0),
        axis=2,
    )
    if int(np.count_nonzero(reference_subject)) < 500:
        raise ValueError(
            "eating_reference_motion_unreliable: reference_subject_too_small"
        )
    if int(np.count_nonzero(reference_bowl)) < 100:
        raise ValueError(
            "eating_reference_motion_unreliable: reference_bowl_too_small"
        )
    subject_left, subject_top, subject_right, subject_bottom = binary_bbox(
        reference_subject
    )
    bowl_left, bowl_top, bowl_right, bowl_bottom = binary_bbox(reference_bowl)
    subject_width = subject_right - subject_left
    subject_height = subject_bottom - subject_top
    bowl_width = bowl_right - bowl_left

    # Anchor the target motion to a real face/muzzle feature, not to the lowest
    # blue pixel in the bowl's central strip.  The latter can be chest fur, a
    # planted forepaw, or a leaked food pixel.  Eating references from the
    # protected profile have a small natural warm nose/muzzle patch.  Select a
    # compact red-balanced connected component immediately above the bowl and
    # track that component through the warp.  Disconnected chest/paw pixels do
    # not participate in either the anchor or the contact validation.
    reference_rgb = reference_rgba[..., :3]
    red = reference_rgb[..., 0].astype(np.int16)
    green = reference_rgb[..., 1].astype(np.int16)
    blue = reference_rgb[..., 2].astype(np.int16)
    row_indices, column_indices = np.indices(reference_subject.shape)
    face_search_x0 = max(
        subject_left,
        bowl_left - int(round(bowl_width * 0.32)),
    )
    face_search_x1 = min(
        subject_right,
        bowl_right + int(round(bowl_width * 0.32)),
    )
    face_search_y0 = max(
        subject_top,
        bowl_top
        - int(round(max(bowl_width * 0.90, subject_height * 0.40))),
    )
    face_search = (
        (column_indices >= face_search_x0)
        & (column_indices < face_search_x1)
        & (row_indices >= face_search_y0)
        & (row_indices < bowl_top)
    )
    face_seed_candidates = (
        reference_subject
        & face_search
        & (red >= 90)
        & (red - green >= 18)
        & (red - blue >= 8)
        & (np.abs(green - blue) <= 25)
    )
    face_count, face_labels, face_stats, face_centroids = (
        cv2.connectedComponentsWithStats(
            face_seed_candidates.astype(np.uint8),
            connectivity=8,
        )
    )
    face_components: list[tuple[int, int, float, int]] = []
    minimum_face_pixels = max(
        12,
        int(round(np.count_nonzero(reference_subject) * 0.00015)),
    )
    for label in range(1, face_count):
        area = int(face_stats[label, cv2.CC_STAT_AREA])
        component_y = int(face_stats[label, cv2.CC_STAT_TOP])
        component_width = int(face_stats[label, cv2.CC_STAT_WIDTH])
        component_height = int(face_stats[label, cv2.CC_STAT_HEIGHT])
        component_bottom = component_y + component_height
        component_gap = bowl_top - component_bottom
        center_x = float(face_centroids[label][0])
        if (
            area >= minimum_face_pixels
            and 0 <= component_gap <= 8
            and 3 <= component_width <= int(round(bowl_width * 0.55))
            and 2 <= component_height <= int(round(bowl_width * 0.35))
            and bowl_left - bowl_width * 0.10
            <= center_x
            <= bowl_right + bowl_width * 0.10
        ):
            face_components.append(
                (
                    component_bottom,
                    area,
                    -abs(center_x - (bowl_left + bowl_right) / 2.0),
                    label,
                )
            )
    if not face_components:
        raise ValueError(
            "eating_reference_motion_unreliable: no_face_muzzle_anchor"
        )
    _face_bottom_rank, _face_area_rank, _face_center_rank, face_label = max(
        face_components
    )
    reference_face_seed = face_labels == face_label
    face_seed_left, face_seed_top, face_seed_right, face_seed_bottom = (
        binary_bbox(reference_face_seed)
    )
    muzzle_x = int(round(float(face_centroids[face_label][0])))
    muzzle_y = face_seed_bottom - 1
    reference_muzzle_gap = bowl_top - face_seed_bottom

    head_center_x = float(muzzle_x)
    head_center_y = float(muzzle_y - subject_height * 0.16)
    head_radius_x = max(28.0, subject_width * 0.22)
    head_radius_y = max(34.0, subject_height * 0.30)
    row_grid, column_grid = np.indices((canvas_size[1], canvas_size[0]))
    head_weight = np.exp(
        -0.5
        * (
            ((column_grid - head_center_x) / head_radius_x) ** 2
            + ((row_grid - head_center_y) / head_radius_y) ** 2
        )
    ).astype(np.float32)
    # Keep the haunches and tail fixed; only the connected head/neck/chest region
    # follows the scalar eating nod.
    head_weight[row_grid > bowl_bottom + int(round(subject_height * 0.08))] = 0.0

    # Below the muzzle, freeze both lower lateral regions on the full remap grid.
    # This keeps the two planted front paws and their surrounding silhouette
    # pixel-identical while allowing only a narrow central mouth corridor to
    # enter the food.  The bowl remains a separate static red SCAIL object.
    freeze_y0 = min(face_seed_bottom, bowl_top)
    corridor_x0 = max(subject_left, face_seed_left - 4)
    corridor_x1 = min(subject_right, face_seed_right + 4)
    lower_lateral_freeze = (row_grid >= freeze_y0) & (
        (column_grid < corridor_x0) | (column_grid >= corridor_x1)
    )
    head_weight[lower_lateral_freeze] = 0.0

    # A valid reference may already touch the rim or leave a small 1-8 px gap.
    # Merely reducing that gap to zero still leaves the cat hovering above its
    # food.  Make the accepted reference the low phase by moving the real muzzle
    # seed a bounded 6-10 px into the upper food band.  Scale by the actual
    # face-seed deformation weight, and cap the correction at the existing head
    # lift envelope so the new downward motion cannot pull harder than the
    # already accepted upward motion.
    face_seed_bottom_band = reference_face_seed & (
        row_indices >= face_seed_bottom - 2
    )
    bottom_weights = head_weight[face_seed_bottom_band]
    if not bottom_weights.size:
        raise ValueError(
            "eating_reference_motion_unreliable: empty_face_seed_bottom"
        )
    contact_weight = max(float(np.percentile(bottom_weights, 25)), 0.25)
    requested_contact_drop_control_px = (
        float(reference_muzzle_gap)
        + float(target_muzzle_food_overlap_px)
        + 0.75
    ) / contact_weight
    contact_drop_control_px = min(
        requested_contact_drop_control_px,
        float(maximum_head_lift_px),
    )

    cat_layer = reference_rgba.copy()
    cat_layer[~reference_subject] = 0
    bowl_layer = reference_rgba.copy()
    bowl_layer[~reference_bowl] = 0
    subject_mask_u8 = reference_subject.astype(np.uint8) * 255
    pose_frames: list[Image.Image] = []
    mask_frames: list[Image.Image] = []
    applied_lifts: list[float] = []
    face_bowl_gaps: list[int] = []
    face_bowl_signed_gaps: list[int] = []
    face_food_overlap_depths: list[int] = []
    face_food_overlap_pixels: list[int] = []
    vertical_controls: list[float] = []
    map_x = column_grid.astype(np.float32)
    face_seed_u8 = reference_face_seed.astype(np.uint8) * 255
    food_contact_band = (
        reference_bowl
        & (row_indices >= bowl_top)
        & (row_indices < min(bowl_bottom, bowl_top + 11))
    )
    for lift_fraction in lift_fractions:
        lift_px = float(maximum_head_lift_px * lift_fraction)
        applied_lifts.append(lift_px)
        # cv2.remap is output-to-source: positive control lifts the head, while
        # the low-phase contact correction uses a small negative control.
        vertical_control_px = lift_px - contact_drop_control_px
        vertical_controls.append(vertical_control_px)
        map_y = (row_grid + vertical_control_px * head_weight).astype(np.float32)
        warped_cat = cv2.remap(
            cat_layer,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        warped_subject = cv2.remap(
            subject_mask_u8,
            map_x,
            map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        warped_subject_bool = warped_subject >= 128
        warped_subject_bool &= ~reference_bowl
        warped_face_seed = cv2.remap(
            face_seed_u8,
            map_x,
            map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        raw_warped_face_seed = warped_face_seed >= 128
        warped_face_rows, _warped_face_columns = np.where(raw_warped_face_seed)
        if not warped_face_rows.size:
            raise ValueError(
                "eating_reference_motion_unreliable: warped_face_seed_empty"
            )
        signed_face_bowl_gap = bowl_top - int(warped_face_rows.max()) - 1
        face_bowl_signed_gaps.append(signed_face_bowl_gap)
        face_bowl_gaps.append(max(0, signed_face_bowl_gap))
        face_food_overlap_depths.append(max(0, -signed_face_bowl_gap))
        face_food_overlap_pixels.append(
            int(
                np.count_nonzero(
                    raw_warped_face_seed & food_contact_band
                )
            )
        )

        pose_rgba = np.zeros_like(reference_rgba)
        pose_rgba[:] = (255, 255, 255, 255)
        cat_alpha = warped_cat[..., 3:4].astype(np.float32) / 255.0
        pose_rgba[..., :3] = np.clip(
            warped_cat[..., :3] * cat_alpha
            + pose_rgba[..., :3] * (1.0 - cat_alpha),
            0,
            255,
        ).astype(np.uint8)
        bowl_alpha = bowl_layer[..., 3:4].astype(np.float32) / 255.0
        pose_rgba[..., :3] = np.clip(
            bowl_layer[..., :3] * bowl_alpha
            + pose_rgba[..., :3] * (1.0 - bowl_alpha),
            0,
            255,
        ).astype(np.uint8)
        pose_frames.append(Image.fromarray(pose_rgba[..., :3], mode="RGB"))

        colored_mask = np.full(
            (canvas_size[1], canvas_size[0], 3),
            255,
            dtype=np.uint8,
        )
        colored_mask[warped_subject_bool] = (0, 0, 255)
        colored_mask[reference_bowl] = (255, 0, 0)
        mask_frames.append(Image.fromarray(colored_mask, mode="RGB"))

    low_phase_index = int(np.argmin(np.asarray(applied_lifts)))
    low_phase_face_gap = face_bowl_gaps[low_phase_index]
    low_phase_signed_gap = face_bowl_signed_gaps[low_phase_index]
    low_phase_overlap_depth = face_food_overlap_depths[low_phase_index]
    low_phase_overlap_pixels = face_food_overlap_pixels[low_phase_index]
    if not (
        EATING_MUZZLE_FOOD_OVERLAP_MIN_PX
        <= low_phase_overlap_depth
        <= EATING_MUZZLE_FOOD_OVERLAP_MAX_PX
    ) or low_phase_overlap_pixels <= 0:
        raise ValueError(
            "eating_reference_motion_unreliable: "
            "unsafe_muzzle_food_overlap: "
            f"signed_gap={low_phase_signed_gap}, "
            f"overlap_depth={low_phase_overlap_depth}, "
            f"overlap_pixels={low_phase_overlap_pixels}"
        )

    metadata = {
        "policy": "reference_space_head_nod_food_overlap_v2",
        "driver_signal_only": True,
        "driver_rgb_preserved": False,
        "driver_silhouette_preserved": False,
        "source_union_bbox": list(source_bbox),
        "driver_signal_method": "bowl_relative_upper_head_row_p10_v1",
        "driver_head_band_bbox": [
            driver_head_x0,
            0,
            driver_head_x1,
            driver_bowl_top,
        ],
        "driver_head_y_p10": round(high_head_y, 4),
        "driver_head_y_p90": round(low_head_y, 4),
        "driver_head_y_span": round(head_y_span, 4),
        "driver_head_y_by_frame": [
            round(value, 4) if np.isfinite(value) else None
            for value in driver_head_y
        ],
        "driver_head_pixels_min": min(driver_head_pixels),
        "head_lift_px_min": round(min(applied_lifts), 4),
        "head_lift_px_max": round(max(applied_lifts), 4),
        "head_lift_px_by_frame": [round(value, 4) for value in applied_lifts],
        "reference_subject_bbox": [
            subject_left,
            subject_top,
            subject_right,
            subject_bottom,
        ],
        "reference_bowl_bbox": [bowl_left, bowl_top, bowl_right, bowl_bottom],
        "reference_muzzle_method": "warm_face_component_above_bowl_v1",
        "reference_face_search_bbox": [
            face_search_x0,
            face_search_y0,
            face_search_x1,
            bowl_top,
        ],
        "reference_face_seed_bbox": [
            face_seed_left,
            face_seed_top,
            face_seed_right,
            face_seed_bottom,
        ],
        "reference_face_seed_pixels": int(np.count_nonzero(reference_face_seed)),
        "reference_muzzle_anchor": [muzzle_x, muzzle_y],
        "reference_muzzle_bowl_gap_px": reference_muzzle_gap,
        "target_muzzle_food_overlap_px": float(
            target_muzzle_food_overlap_px
        ),
        "reference_contact_drop_px": round(
            reference_muzzle_gap + target_muzzle_food_overlap_px,
            4,
        ),
        "reference_contact_drop_requested_control_px": round(
            requested_contact_drop_control_px,
            4,
        ),
        "reference_contact_drop_control_px": round(contact_drop_control_px, 4),
        "reference_contact_drop_capped": (
            requested_contact_drop_control_px > maximum_head_lift_px
        ),
        "face_bowl_gap_px_by_frame": face_bowl_gaps,
        "face_bowl_signed_gap_px_by_frame": face_bowl_signed_gaps,
        "muzzle_food_overlap_depth_px_by_frame": face_food_overlap_depths,
        "muzzle_food_overlap_pixels_by_frame": face_food_overlap_pixels,
        "low_phase_face_bowl_gap_px": low_phase_face_gap,
        "low_phase_face_bowl_signed_gap_px": low_phase_signed_gap,
        "low_phase_muzzle_food_overlap_depth_px": low_phase_overlap_depth,
        "low_phase_muzzle_food_overlap_pixels": low_phase_overlap_pixels,
        "vertical_control_px_by_frame": [
            round(value, 4) for value in vertical_controls
        ],
        "vertical_control_abs_max_px": round(
            max(abs(value) for value in vertical_controls),
            4,
        ),
        "muzzle_motion_corridor_bbox": [
            corridor_x0,
            freeze_y0,
            corridor_x1,
            bowl_bottom,
        ],
        "lower_lateral_freeze_bboxes": [
            [subject_left, freeze_y0, corridor_x0, subject_bottom],
            [corridor_x1, freeze_y0, subject_right, subject_bottom],
        ],
        "head_deformation_center": [
            round(head_center_x, 4),
            round(head_center_y, 4),
        ],
        "head_deformation_radius": [
            round(head_radius_x, 4),
            round(head_radius_y, 4),
        ],
        "scene_prop_separated": True,
        "static_bowl_mask": True,
        "static_bowl_rgb": True,
        "scene_prop_metadata": prop_metadata,
    }
    return pose_frames, mask_frames, metadata


def _make_eating_reference_identity_mask_strict(
    normalized: Image.Image,
) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
    """Map the cat to blue and its bowl to red using strict visual anchors."""
    rgba = np.asarray(normalized.convert("RGBA"), dtype=np.uint8)
    rgb = rgba[..., :3]
    alpha = rgba[..., 3] >= 16
    if not np.any(alpha):
        raise ValueError("eating_reference_prop_split_unreliable: empty_alpha")

    left, top, right, bottom = binary_bbox(alpha)
    subject_width = right - left
    subject_height = bottom - top
    if subject_width < 64 or subject_height < 64:
        raise ValueError("eating_reference_prop_split_unreliable: tiny_subject")

    row_indices, column_indices = np.indices(alpha.shape)
    red = rgb[..., 0].astype(np.int16)
    green = rgb[..., 1].astype(np.int16)
    blue = rgb[..., 2].astype(np.int16)
    subject_center_x = (left + right) / 2.0

    # The full-subject bbox can be strongly biased by a long tail.  Keep the
    # search in the lower part of the reference, but leave enough horizontal
    # room for a bowl placed under an off-centre muzzle.
    zone_x0 = left + int(round(subject_width * 0.20))
    zone_x1 = left + int(round(subject_width * 0.95))
    zone_y0 = top + int(round(subject_height * 0.65))
    bowl_zone = (
        (column_indices >= zone_x0)
        & (column_indices < zone_x1)
        & (row_indices >= zone_y0)
        & (row_indices < bottom)
    )

    definite_prop = (
        alpha
        & bowl_zone
        & (red >= 45)
        & (red - green >= 28)
        & (green - blue >= 18)
        & (red - blue >= 52)
    )
    definite_pixels = int(np.count_nonzero(definite_prop))
    minimum_definite_pixels = max(
        80,
        int(round(np.count_nonzero(alpha) * 0.0015)),
    )
    if definite_pixels < minimum_definite_pixels:
        raise ValueError(
            "eating_reference_prop_split_unreliable: "
            f"warm_seed_pixels={definite_pixels}<"
            f"{minimum_definite_pixels}"
        )

    # Use the food as a high-confidence anchor.  Looking only in the bottom
    # 18 percent used to reject valid references whose kibble sits above a
    # tall, pale bowl; at that depth the warm core has already disappeared.
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        definite_prop.astype(np.uint8),
        connectivity=8,
    )
    candidates: list[int] = []
    component_bottom_tolerance = max(
        2,
        int(round(subject_height * 0.01)),
    )
    component_bottom_min = top + int(round(subject_height * 0.76))
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        center_x, center_y = centroids[label]
        component_bottom = int(stats[label, cv2.CC_STAT_TOP]) + int(
            stats[label, cv2.CC_STAT_HEIGHT]
        )
        # Real Main_case kibble is a row of several small separated pellets,
        # not one 100-pixel warm component.  Keep every credible lower-central
        # pellet and merge their union into one food anchor; tiny pink nose or
        # paw accents remain outside this lower band or fail the area floor.
        if (
            area >= 6
            and component_bottom + component_bottom_tolerance
            >= component_bottom_min
            and center_y >= zone_y0
            and abs(center_x - subject_center_x) <= subject_width * 0.36
        ):
            candidates.append(label)
    if not candidates:
        raise ValueError(
            "eating_reference_prop_split_unreliable: no_warm_food_component"
        )
    food_core = np.isin(labels, np.asarray(candidates, dtype=labels.dtype))
    food_area = int(np.count_nonzero(food_core))
    food_x, food_y, food_right, food_bottom = binary_bbox(food_core)
    food_width = food_right - food_x
    food_height = food_bottom - food_y

    # A tall warm component reaching the subject baseline is already the bowl
    # plus its food, not merely a food anchor.  Qwen often joins that orange
    # bowl to a pale paw with a few antialiased pixels.  Erode it until the
    # narrow bridge breaks, select the lower central core, then recover only
    # high-confidence warm pixels inside a tightly bounded horizontal window.
    # This keeps both paws blue instead of letting a relaxed beige search turn
    # them into the red prop identity.
    warm_component_is_bowl = (
        food_height / float(subject_height) >= 0.18
        and food_y + food_height >= top + int(round(subject_height * 0.94))
    )
    warm_component_initially_classified_as_bowl = warm_component_is_bowl
    warm_bowl_core_fallback_to_beige = False
    if warm_component_is_bowl:
        eroded_warm = cv2.erode(
            food_core.astype(np.uint8),
            np.ones((3, 3), dtype=np.uint8),
            iterations=3,
        )
        body_count, body_labels, body_stats, body_centroids = (
            cv2.connectedComponentsWithStats(eroded_warm, connectivity=8)
        )
        body_candidates: list[tuple[int, float, int]] = []
        for label in range(1, body_count):
            area = int(body_stats[label, cv2.CC_STAT_AREA])
            component_bottom = int(body_stats[label, cv2.CC_STAT_TOP]) + int(
                body_stats[label, cv2.CC_STAT_HEIGHT]
            )
            center_x = float(body_centroids[label][0])
            if (
                area >= 120
                and component_bottom >= top + int(round(subject_height * 0.90))
                and abs(center_x - subject_center_x) <= subject_width * 0.24
            ):
                body_candidates.append((area, -abs(center_x - subject_center_x), label))
        if body_candidates:
            _body_area_rank, _center_rank, body_label = max(body_candidates)
            body_area = int(body_stats[body_label, cv2.CC_STAT_AREA])
            body_x = int(body_stats[body_label, cv2.CC_STAT_LEFT])
            body_y = int(body_stats[body_label, cv2.CC_STAT_TOP])
            body_width = int(body_stats[body_label, cv2.CC_STAT_WIDTH])
            body_height = int(body_stats[body_label, cv2.CC_STAT_HEIGHT])
            prop_x0 = max(zone_x0, body_x - 6)
            prop_x1 = min(zone_x1, body_x + body_width + 6)
            bowl_body_method = "warm_bowl_eroded_core_v1"
            # Keep only the warm component that was actually selected as the
            # food/bowl object.  Using every warm pixel in the lower search zone
            # also painted a disconnected pink nose red in real Qwen references,
            # which then froze the nose into the static bowl layer.
            prop_seed = food_core
        else:
            # A thin warm rim joined to warm kibble can make the union tall
            # enough to resemble a solid warm bowl, yet leave no thick core
            # after erosion. Reuse the stricter food-anchored beige recovery
            # below; its width, prop fraction, paw support and contact checks
            # still fail closed instead of accepting the union directly.
            warm_component_is_bowl = False
            warm_bowl_core_fallback_to_beige = True
    if not warm_component_is_bowl:
        # Real profile references use a light-beige bowl, so the strict warm
        # mask normally covers the kibble rather than the bowl body. Recover
        # the lower beige component connected to that food anchor. The
        # asymmetric allowance reflects the common three-quarter eating pose.
        bowl_search_x0 = max(
            zone_x0,
            food_x - max(8, int(round(food_width * 0.10))),
        )
        bowl_search_x1 = min(
            zone_x1,
            food_x + food_width + max(8, int(round(food_width * 0.10))),
        )
        # A row of separated kibble often occupies only the central part of a
        # wide pale bowl.  Deriving the complete search width from that row
        # clipped valid Qwen bowls to 11-20% of the subject bbox.  Guarantee a
        # conservative food-centred search corridor; the connected-component,
        # prop-fraction, and two-sided paw-support checks below still prevent
        # this corridor from silently absorbing the planted paws.
        minimum_food_anchored_search_width = max(
            food_width + 16,
            int(round(subject_width * 0.32)),
        )
        food_center_x = (food_x + food_right) / 2.0
        bowl_search_x0 = max(
            zone_x0,
            min(
                bowl_search_x0,
                int(round(food_center_x - minimum_food_anchored_search_width / 2.0)),
            ),
        )
        bowl_search_x1 = min(
            zone_x1,
            max(
                bowl_search_x1,
                int(round(food_center_x + minimum_food_anchored_search_width / 2.0)),
            ),
        )
        warm_bowl_body = (
            alpha
            & (column_indices >= bowl_search_x0)
            & (column_indices < bowl_search_x1)
            & (row_indices >= max(zone_y0, food_y - 2))
            & (row_indices < bottom)
            & (red >= 45)
            & (red - green >= 6)
            & (red - blue >= 12)
            & (green - blue >= 8)
        )
        channel_max = np.maximum(np.maximum(red, green), blue)
        channel_min = np.minimum(np.minimum(red, green), blue)
        neutral_bowl_body = (
            alpha
            & (column_indices >= bowl_search_x0)
            & (column_indices < bowl_search_x1)
            & (row_indices >= max(zone_y0, food_y - 3))
            & (row_indices < bottom)
            & (channel_max >= 72)
            & ((channel_max - channel_min) <= 34)
        )
        # Preserve the neutral shaded base without reopening the lateral paws.
        bowl_base_x0 = max(
            bowl_search_x0,
            food_x - int(round(food_width * 0.05)),
        )
        bowl_base_x1 = min(
            bowl_search_x1,
            food_x + food_width + int(round(food_width * 0.02)),
        )
        neutral_bowl_base = (
            alpha
            & (column_indices >= bowl_base_x0)
            & (column_indices < bowl_base_x1)
            & (row_indices >= food_y + food_height - 2)
            & (row_indices < bottom)
            & (red >= 45)
            & (red - blue >= 12)
            & (green - blue >= 8)
        )
        bowl_body_seed = warm_bowl_body | neutral_bowl_body | neutral_bowl_base
        bowl_body_seed = cv2.morphologyEx(
            bowl_body_seed.astype(np.uint8),
            cv2.MORPH_CLOSE,
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        ).astype(bool)
        body_count, body_labels, body_stats, _body_centroids = (
            cv2.connectedComponentsWithStats(
                bowl_body_seed.astype(np.uint8),
                connectivity=8,
            )
        )
        food_support = cv2.dilate(
            food_core.astype(np.uint8),
            np.ones((5, 5), dtype=np.uint8),
            iterations=1,
        ).astype(bool)
        body_candidates_relaxed: list[tuple[int, int, int]] = []
        for label in range(1, body_count):
            component = body_labels == label
            anchor_overlap = int(np.count_nonzero(component & food_support))
            area = int(body_stats[label, cv2.CC_STAT_AREA])
            if anchor_overlap >= 20 and area >= food_area:
                body_candidates_relaxed.append((anchor_overlap, area, label))
        if not body_candidates_relaxed:
            raise ValueError(
                "eating_reference_prop_split_unreliable: no_bowl_body_component"
            )
        _anchor_overlap, body_area, body_label = max(body_candidates_relaxed)
        body_x = int(body_stats[body_label, cv2.CC_STAT_LEFT])
        body_y = int(body_stats[body_label, cv2.CC_STAT_TOP])
        body_width = int(body_stats[body_label, cv2.CC_STAT_WIDTH])
        body_height = int(body_stats[body_label, cv2.CC_STAT_HEIGHT])
        prop_x0 = max(zone_x0, body_x - 2)
        prop_x1 = min(zone_x1, body_x + body_width + 2)
        bowl_body_method = (
            "food_anchored_beige_bowl_after_eroded_warm_core_miss_v2"
            if warm_bowl_core_fallback_to_beige
            else "warm_food_core_plus_beige_bowl_v1"
        )
        prop_seed = body_labels == body_label

    safe_width_fraction = (prop_x1 - prop_x0) / float(subject_width)
    if not 0.25 <= safe_width_fraction <= 0.55:
        raise ValueError(
            "eating_reference_prop_split_unreliable: "
            f"safe_width_fraction={safe_width_fraction:.4f}"
        )

    safe_zone = bowl_zone & (column_indices >= prop_x0) & (column_indices < prop_x1)
    prop = prop_seed & safe_zone
    prop = cv2.morphologyEx(
        prop.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    )
    prop = cv2.dilate(
        prop,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    ).astype(bool)
    prop = fill_internal_holes(prop)
    prop &= safe_zone & alpha

    safety_zone = (
        (column_indices >= prop_x0)
        & (column_indices < prop_x1)
        & (row_indices >= max(0, zone_y0 - 2))
        & (row_indices < min(alpha.shape[0], bottom + 2))
    )
    prop_exclusion = cv2.dilate(
        prop.astype(np.uint8),
        np.ones((5, 5), dtype=np.uint8),
        iterations=1,
    ).astype(bool)
    prop_exclusion &= safety_zone

    hard_alpha = cv2.dilate(
        alpha.astype(np.uint8),
        np.ones((5, 5), dtype=np.uint8),
        iterations=1,
    ).astype(bool)
    identity = hard_alpha & ~prop_exclusion

    # Validate identity support immediately beside the detected bowl.  Fixed
    # percentages of the full bbox are unreliable when a long tail shifts the
    # subject centre; in that case the old "right paw" zone fell inside the
    # bowl itself.
    paw_span = max(16, int(round(subject_width * 0.20)))
    paw_y0 = top + int(round(subject_height * 0.72))
    left_paw_zone = (
        (column_indices >= max(left, prop_x0 - paw_span))
        & (column_indices < prop_x0)
        & (row_indices >= paw_y0)
        & (row_indices < bottom)
    )
    right_paw_zone = (
        (column_indices >= prop_x1)
        & (column_indices < min(right, prop_x1 + paw_span))
        & (row_indices >= paw_y0)
        & (row_indices < bottom)
    )
    left_support = int(np.count_nonzero(identity & left_paw_zone))
    right_support = int(np.count_nonzero(identity & right_paw_zone))
    if left_support < 500 or right_support < 500:
        raise ValueError(
            "eating_reference_prop_split_unreliable: "
            f"paw_support=({left_support},{right_support})"
        )

    prop_inside_subject = prop_exclusion & hard_alpha
    prop_fraction = float(np.count_nonzero(prop_inside_subject)) / float(
        max(1, np.count_nonzero(hard_alpha))
    )
    if not 0.04 <= prop_fraction <= 0.20:
        raise ValueError(
            "eating_reference_prop_split_unreliable: "
            f"prop_fraction={prop_fraction:.4f}"
        )

    colored = np.zeros((*alpha.shape, 3), dtype=np.uint8)
    colored[identity] = (0, 0, 255)
    colored[prop_exclusion] = (255, 0, 0)
    colored_image = Image.fromarray(colored, mode="RGB")
    exclusion_image = Image.fromarray(
        prop_exclusion.astype(np.uint8) * 255,
        mode="L",
    )
    metadata = {
        "policy": "eating_reference_cat_blue_bowl_red_v1",
        "split_status": "strict",
        "fallback_used": False,
        "fallback_reason": None,
        "subject_bbox": list((left, top, right, bottom)),
        "bowl_zone_bbox": list(binary_bbox(bowl_zone)),
        "warm_seed_pixels": definite_pixels,
        "minimum_warm_seed_pixels": minimum_definite_pixels,
        "food_core_bbox": [
            food_x,
            food_y,
            food_x + food_width,
            food_y + food_height,
        ],
        "food_core_pixels": food_area,
        "food_core_components": len(candidates),
        "food_component_bottom_min": component_bottom_min,
        "food_component_bottom_tolerance": component_bottom_tolerance,
        "warm_component_initially_classified_as_bowl": (
            warm_component_initially_classified_as_bowl
        ),
        "warm_bowl_core_fallback_to_beige": warm_bowl_core_fallback_to_beige,
        "deep_body_bbox": [
            body_x,
            body_y,
            body_x + body_width,
            body_y + body_height,
        ],
        "deep_body_pixels": body_area,
        "bowl_body_method": bowl_body_method,
        "food_anchored_search_width_min_fraction": 0.32,
        "safe_x_range": [prop_x0, prop_x1],
        "safe_width_fraction": safe_width_fraction,
        "paw_identity_y0": paw_y0,
        "prop_exclusion_bbox": list(binary_bbox(prop_exclusion)),
        "prop_fraction": prop_fraction,
        "left_paw_identity_support": left_support,
        "right_paw_identity_support": right_support,
    }
    return colored_image, exclusion_image, metadata


def _make_eating_reference_identity_mask_fallback(
    normalized: Image.Image,
    *,
    strict_error: ValueError,
) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
    """Build a conservative lower-centre prop mask after strict split failure.

    This fallback deliberately prefers a usable SCAIL reference over rejecting
    the complete video job because a generated bowl or food uses an unexpected
    colour.  A relaxed warm-food anchor is used when available.  Otherwise the
    mask reserves a narrow lower-centre corridor where the eating prompt and
    driver place the bowl, leaving the lateral planted-paw regions as identity.
    """

    rgba = np.asarray(normalized.convert("RGBA"), dtype=np.uint8)
    rgb = rgba[..., :3]
    alpha = rgba[..., 3] >= 16
    if not np.any(alpha):
        raise ValueError("eating_reference_prop_split_unreliable: empty_alpha")

    left, top, right, bottom = binary_bbox(alpha)
    subject_width = right - left
    subject_height = bottom - top
    if subject_width < 8 or subject_height < 8:
        raise ValueError("eating_reference_prop_split_unreliable: tiny_subject")

    row_indices, column_indices = np.indices(alpha.shape)
    red = rgb[..., 0].astype(np.int16)
    green = rgb[..., 1].astype(np.int16)
    blue = rgb[..., 2].astype(np.int16)
    channel_max = np.maximum(np.maximum(red, green), blue)
    channel_min = np.minimum(np.minimum(red, green), blue)
    subject_center_x = (left + right) / 2.0

    search_y0 = top + int(round(subject_height * 0.60))
    search_x0 = left + int(round(subject_width * 0.12))
    search_x1 = right - int(round(subject_width * 0.12))
    relaxed_warm = (
        alpha
        & (column_indices >= search_x0)
        & (column_indices < search_x1)
        & (row_indices >= search_y0)
        & (row_indices < bottom)
        & (red >= 40)
        & (red - green >= 8)
        & (red - blue >= 18)
        & ((channel_max - channel_min) >= 18)
    )
    relaxed_warm = cv2.morphologyEx(
        relaxed_warm.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((5, 5), dtype=np.uint8),
        iterations=1,
    ).astype(bool)

    anchor_method = "lower_center_geometry"
    anchor_pixels = int(np.count_nonzero(relaxed_warm))
    anchor_bbox: tuple[int, int, int, int] | None = None
    if anchor_pixels >= max(20, int(round(np.count_nonzero(alpha) * 0.0002))):
        warm_count, warm_labels, warm_stats, warm_centroids = (
            cv2.connectedComponentsWithStats(
                relaxed_warm.astype(np.uint8),
                connectivity=8,
            )
        )
        warm_candidates: list[tuple[int, float, int]] = []
        for label in range(1, warm_count):
            area = int(warm_stats[label, cv2.CC_STAT_AREA])
            center_x, center_y = warm_centroids[label]
            if (
                area >= 6
                and center_y >= search_y0
                and abs(center_x - subject_center_x) <= subject_width * 0.40
            ):
                warm_candidates.append(
                    (area, -abs(float(center_x) - subject_center_x), label)
                )
        if warm_candidates:
            selected_labels = [item[2] for item in warm_candidates]
            anchor = np.isin(
                warm_labels,
                np.asarray(selected_labels, dtype=warm_labels.dtype),
            )
            anchor_bbox = binary_bbox(anchor)
            anchor_pixels = int(np.count_nonzero(anchor))
            anchor_method = "relaxed_warm_food_geometry"

    minimum_prop_width = max(24, int(round(subject_width * 0.30)))
    maximum_prop_width = max(minimum_prop_width, int(round(subject_width * 0.48)))
    if anchor_bbox is not None:
        anchor_left, anchor_top, anchor_right, anchor_bottom = anchor_bbox
        anchor_width = anchor_right - anchor_left
        prop_width = min(
            maximum_prop_width,
            max(minimum_prop_width, anchor_width + int(round(subject_width * 0.08))),
        )
        prop_center_x = (anchor_left + anchor_right) / 2.0
        prop_y0 = max(
            search_y0,
            anchor_top - max(2, int(round(subject_height * 0.015))),
        )
    else:
        prop_width = minimum_prop_width
        prop_center_x = subject_center_x
        prop_y0 = top + int(round(subject_height * 0.70))
        anchor_bottom = prop_y0

    prop_x0 = max(left, int(round(prop_center_x - prop_width / 2.0)))
    prop_x1 = min(right, prop_x0 + prop_width)
    if prop_x1 - prop_x0 < prop_width:
        prop_x0 = max(left, prop_x1 - prop_width)
    # The food anchor normally sits at the rim while the neutral bowl body
    # continues to the subject baseline.  Keep the conservative horizontal
    # corridor narrow, but reserve it through the bottom so the bowl base does
    # not become part of the pet identity merely because its colour is neutral.
    prop_y1 = bottom

    fallback_zone = (
        (column_indices >= prop_x0)
        & (column_indices < prop_x1)
        & (row_indices >= prop_y0)
        & (row_indices < prop_y1)
    )
    hard_alpha = cv2.dilate(
        alpha.astype(np.uint8),
        np.ones((5, 5), dtype=np.uint8),
        iterations=1,
    ).astype(bool)
    prop_exclusion = fallback_zone & hard_alpha
    prop_exclusion = cv2.morphologyEx(
        prop_exclusion.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((5, 5), dtype=np.uint8),
        iterations=1,
    ).astype(bool)
    if not np.any(prop_exclusion):
        # A visible alpha subject always has a lower-central support region in
        # this action.  If antialiasing left the first corridor empty, expand
        # vertically but retain the same paw-safe horizontal bounds.
        prop_exclusion = (
            (column_indices >= prop_x0)
            & (column_indices < prop_x1)
            & (row_indices >= search_y0)
            & (row_indices < bottom)
            & hard_alpha
        )

    identity = hard_alpha & ~prop_exclusion
    if not np.any(identity):
        # Preserve the upper subject as identity even for an unusually small
        # or tightly cropped input.  This is a runtime-safety guard, not a
        # visual quality gate.
        identity = hard_alpha & (row_indices < prop_y0)
    if not np.any(identity):
        identity = hard_alpha.copy()

    prop_pixels = int(np.count_nonzero(prop_exclusion))
    identity_pixels = int(np.count_nonzero(identity))
    hard_alpha_pixels = int(np.count_nonzero(hard_alpha))
    colored = np.zeros((*alpha.shape, 3), dtype=np.uint8)
    colored[identity] = (0, 0, 255)
    colored[prop_exclusion] = (255, 0, 0)
    colored_image = Image.fromarray(colored, mode="RGB")
    exclusion_image = Image.fromarray(
        prop_exclusion.astype(np.uint8) * 255,
        mode="L",
    )
    metadata = {
        "policy": "eating_reference_cat_blue_bowl_red_v1",
        "split_status": "degraded_nonblocking_fallback",
        "fallback_used": True,
        "fallback_reason": str(strict_error)[:1000],
        "fallback_policy": "lower_center_prop_geometry_v1",
        "subject_bbox": [left, top, right, bottom],
        "bowl_zone_bbox": [prop_x0, prop_y0, prop_x1, prop_y1],
        "warm_seed_pixels": anchor_pixels,
        "minimum_warm_seed_pixels": None,
        "food_core_bbox": list(anchor_bbox) if anchor_bbox is not None else None,
        "food_core_pixels": anchor_pixels,
        "food_core_components": None,
        "warm_component_initially_classified_as_bowl": False,
        "warm_bowl_core_fallback_to_beige": False,
        "deep_body_bbox": [prop_x0, prop_y0, prop_x1, prop_y1],
        "deep_body_pixels": prop_pixels,
        "bowl_body_method": f"nonblocking_{anchor_method}_v1",
        "food_anchored_search_width_min_fraction": 0.30,
        "safe_x_range": [prop_x0, prop_x1],
        "safe_width_fraction": (prop_x1 - prop_x0) / float(subject_width),
        "paw_identity_y0": top + int(round(subject_height * 0.72)),
        "prop_exclusion_bbox": (
            list(binary_bbox(prop_exclusion)) if np.any(prop_exclusion) else None
        ),
        "prop_fraction": prop_pixels / float(max(1, hard_alpha_pixels)),
        "left_paw_identity_support": None,
        "right_paw_identity_support": None,
        "identity_pixels": identity_pixels,
    }
    return colored_image, exclusion_image, metadata


def make_eating_reference_identity_mask(
    normalized: Image.Image,
    *,
    allow_fallback: bool = True,
) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
    """Map cat/bowl identities without letting visual uncertainty stop a job."""

    try:
        return _make_eating_reference_identity_mask_strict(normalized)
    except ValueError as exc:
        if not allow_fallback:
            raise
        error = str(exc)
        if not error.startswith("eating_reference_prop_split_unreliable:"):
            raise
        if error.endswith((": empty_alpha", ": tiny_subject")):
            raise
        return _make_eating_reference_identity_mask_fallback(
            normalized,
            strict_error=exc,
        )


def make_reference_assets(
    reference_path: Path,
    canvas_size: tuple[int, int],
    separate_eating_prop: bool = False,
    contain_fraction: float = 0.84,
) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
    with Image.open(reference_path) as source:
        reference = source.convert("RGBA")
    reference_extraction = "existing_alpha"
    alpha_array = np.asarray(reference.getchannel("A"))
    if bool(np.all(alpha_array == 255)):
        rgb_array = np.asarray(reference.convert("RGB")).copy()
        corners = np.asarray(
            [
                rgb_array[0, 0],
                rgb_array[0, -1],
                rgb_array[-1, 0],
                rgb_array[-1, -1],
            ],
            dtype=np.float32,
        )
        green_dominant = bool(
            np.all(corners[:, 1] >= corners[:, 0] * 1.12)
            and np.all(corners[:, 1] >= corners[:, 2] * 1.12)
            and float(np.mean(corners[:, 1])) >= 80.0
        )
        if not green_dominant:
            raise ValueError(
                "Opaque SCAIL reference must have a green-screen background "
                f"or an existing alpha channel: {reference_path}"
            )
        extracted_rgb, extracted_alpha = _subject_mask(rgb_array)
        foreground_fraction = float(
            np.count_nonzero(extracted_alpha >= 16)
        ) / float(extracted_alpha.size)
        if not 0.02 <= foreground_fraction <= 0.85:
            raise ValueError(
                "Green-screen reference extraction produced an invalid "
                f"foreground fraction {foreground_fraction:.4f}: "
                f"{reference_path}"
            )
        reference = Image.fromarray(
            np.dstack((extracted_rgb, extracted_alpha)).astype(np.uint8),
            mode="RGBA",
        )
        reference_extraction = "chroma_key"
    alpha_bbox = reference.getchannel("A").getbbox()
    if not alpha_bbox:
        raise ValueError(f"Reference has no visible alpha subject: {reference_path}")
    cropped = reference.crop(expand_bbox(alpha_bbox, reference.size, margin_fraction=0.04))
    if not 0.50 <= contain_fraction <= 0.95:
        raise ValueError("reference contain fraction must be between 0.50 and 0.95")
    normalized = contain_rgba(cropped, canvas_size, contain_fraction)
    alpha = normalized.getchannel("A")
    hard_alpha = alpha.point(lambda value: 255 if value >= 16 else 0)
    hard_alpha = hard_alpha.filter(ImageFilter.MaxFilter(5))

    reference_prop_metadata: dict[str, Any] | None = None
    if separate_eating_prop:
        colored_mask, _prop_exclusion, reference_prop_metadata = (
            make_eating_reference_identity_mask(normalized)
        )
    else:
        # SCAIL reference-mask background is black, same identity is pure blue.
        colored_mask = Image.new("RGB", canvas_size, (0, 0, 0))
        colored_mask.paste(
            Image.new("RGB", canvas_size, (0, 0, 255)),
            mask=hard_alpha,
        )
    identity_pixels = np.all(
        np.asarray(colored_mask, dtype=np.uint8) == (0, 0, 255),
        axis=2,
    )
    prop_pixels = np.all(
        np.asarray(colored_mask, dtype=np.uint8) == (255, 0, 0),
        axis=2,
    )
    metadata = {
        "reference_extraction": reference_extraction,
        "reference_contain_fraction": round(float(contain_fraction), 6),
        "reference_alpha_bbox": list(alpha_bbox),
        "reference_subject_fraction": (
            float(np.count_nonzero(np.asarray(hard_alpha)))
            / float(canvas_size[0] * canvas_size[1])
        ),
        "reference_identity_fraction": (
            float(np.count_nonzero(identity_pixels))
            / float(canvas_size[0] * canvas_size[1])
        ),
        "reference_prop_fraction": (
            float(np.count_nonzero(prop_pixels))
            / float(canvas_size[0] * canvas_size[1])
        ),
        "scene_prop_separated": separate_eating_prop,
        "scene_prop_metadata": reference_prop_metadata,
    }
    return normalized, colored_mask, metadata


def _run_checked(command: list[str], stage: str, timeout: int = 900) -> None:
    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if process.returncode:
        detail = (process.stderr or process.stdout or "unknown error").strip()
        raise RuntimeError(f"{stage} failed: {detail[-4000:]}")


def _validate_transparent_gif(
    path: Path,
    *,
    size: tuple[int, int],
    frame_count: int,
    duration_ms: int,
    expected_foreground_masks: list[np.ndarray] | None = None,
) -> dict[str, Any]:
    with Image.open(path) as image:
        loop_count = int(image.info.get("loop", -1))
        if image.size != size:
            raise RuntimeError(f"GIF size is {image.size}, expected {size}")
        if image.n_frames != frame_count:
            raise RuntimeError(
                f"GIF has {image.n_frames} frames, expected {frame_count}"
            )
        if (
            expected_foreground_masks is not None
            and len(expected_foreground_masks) != frame_count
        ):
            raise RuntimeError(
                "Expected foreground mask count does not match GIF frames: "
                f"{len(expected_foreground_masks)} != {frame_count}"
            )
        durations: list[int] = []
        corner_alpha: list[int] = []
        border_alpha_max = 0
        encoded_foreground_retentions: list[float] = []
        encoded_foreground_ious: list[float] = []
        alpha_has_both: list[bool] = []
        edge_metrics: list[dict[str, float | int]] = []
        for index in range(image.n_frames):
            image.seek(index)
            durations.append(int(image.info.get("duration", 0)))
            rgba = np.asarray(image.convert("RGBA"))
            alpha = rgba[:, :, 3]
            corner_alpha.extend(
                int(alpha[y, x])
                for x, y in (
                    (0, 0),
                    (size[0] - 1, 0),
                    (0, size[1] - 1),
                    (size[0] - 1, size[1] - 1),
                )
            )
            border_alpha_max = max(
                border_alpha_max,
                int(alpha[:2, :].max()),
                int(alpha[-2:, :].max()),
                int(alpha[:, :2].max()),
                int(alpha[:, -2:].max()),
            )
            encoded_foreground = alpha >= 128
            alpha_has_both.append(
                bool(np.any(alpha == 0) and np.any(alpha == 255))
            )
            edge_metrics.append(_binary_alpha_edge_metrics(alpha))
            if expected_foreground_masks is not None:
                expected_foreground = expected_foreground_masks[index]
                expected_pixels = max(
                    1,
                    int(np.count_nonzero(expected_foreground)),
                )
                intersection = int(
                    np.count_nonzero(
                        encoded_foreground & expected_foreground
                    )
                )
                union = max(
                    1,
                    int(
                        np.count_nonzero(
                            encoded_foreground | expected_foreground
                        )
                    ),
                )
                encoded_foreground_retentions.append(
                    float(intersection) / float(expected_pixels)
                )
                encoded_foreground_ious.append(
                    float(intersection) / float(union)
                )
    actual_duration = sum(durations)
    if actual_duration != duration_ms:
        raise RuntimeError(
            f"GIF duration is {actual_duration}ms, expected {duration_ms}ms"
        )
    if loop_count != 0:
        raise RuntimeError(
            f"GIF loop count is {loop_count}, expected infinite loop (0)"
        )
    if any(corner_alpha):
        raise RuntimeError("GIF has an opaque canvas corner")
    if border_alpha_max:
        raise RuntimeError("GIF has an opaque pixel in its 2px canvas border")
    if not all(alpha_has_both):
        raise RuntimeError(
            "At least one GIF frame lacks transparent or opaque pixels"
        )
    encoded_foreground_retention_min = min(
        encoded_foreground_retentions,
        default=1.0,
    )
    encoded_foreground_iou_min = min(
        encoded_foreground_ious,
        default=1.0,
    )
    if encoded_foreground_retention_min < 0.995:
        raise RuntimeError(
            "GIF encoding removed foreground pixels: minimum retention="
            f"{encoded_foreground_retention_min:.4f}"
        )
    if encoded_foreground_iou_min < 0.995:
        raise RuntimeError(
            "GIF encoding changed the foreground matte: minimum IoU="
            f"{encoded_foreground_iou_min:.4f}"
        )
    return {
        "size": list(size),
        "frames": frame_count,
        "duration_ms": actual_duration,
        "frame_durations_ms": sorted(set(durations)),
        "all_corners_transparent": True,
        "full_2px_border_transparent": True,
        "all_frames_have_transparent_and_opaque_pixels": True,
        "encoded_foreground_retention_min": (
            encoded_foreground_retention_min
        ),
        "encoded_foreground_iou_min": encoded_foreground_iou_min,
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
        "loop": loop_count,
        "bytes": path.stat().st_size,
    }


def _validate_white_background_mp4(
    path: Path,
    *,
    size: tuple[int, int],
    frame_count: int,
) -> dict[str, Any]:
    decoded = 0
    corner_channel_min = 255
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            rgb = frame.to_ndarray(format="rgb24")
            if (rgb.shape[1], rgb.shape[0]) != size:
                raise RuntimeError(
                    f"White-background MP4 frame size is "
                    f"{rgb.shape[1]}x{rgb.shape[0]}, expected {size[0]}x{size[1]}"
                )
            corners = np.asarray(
                [rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1]],
                dtype=np.uint8,
            )
            corner_channel_min = min(
                corner_channel_min,
                int(corners.min()),
            )
            decoded += 1
    if decoded != frame_count:
        raise RuntimeError(
            f"White-background MP4 decoded {decoded} frames; expected {frame_count}"
        )
    if corner_channel_min < 245:
        raise RuntimeError(
            "White-background MP4 has a non-white canvas corner: minimum channel="
            f"{corner_channel_min}"
        )
    return {
        "size": list(size),
        "frames": decoded,
        "all_corner_channels_min": corner_channel_min,
        "bytes": path.stat().st_size,
    }


def _select_pose_guided_semantic_components(
    labels: np.ndarray,
    stats: np.ndarray,
    guidance: np.ndarray,
    semantic_core: np.ndarray,
    *,
    minimum_area: int = 64,
    minimum_guidance_overlap: int = 32,
    minimum_guidance_fraction: float = 0.20,
    minimum_semantic_core_pixels: int = 64,
    minimum_semantic_core_fraction: float = 0.20,
) -> tuple[list[int], list[int]]:
    """Keep guided components plus high-confidence fragments of that subject.

    The alpha passed to this helper has already been restricted to components
    accepted against the pose guide. A later cleanup can cut a thin soft-alpha
    bridge through a real paw, ear, or tail and split that accepted subject into
    a second component. Preserve such a fragment only when it still contains a
    substantial high-confidence semantic core; detached low-confidence noise
    remains excluded.
    """
    candidates: list[tuple[int, int, int, int]] = []
    guided_labels: list[int] = []
    semantic_fragment_labels: list[int] = []
    for label in range(1, int(stats.shape[0])):
        component = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        overlap = int(np.count_nonzero(component & guidance))
        core_pixels = int(np.count_nonzero(component & semantic_core))
        candidates.append((overlap, area, core_pixels, label))
        if (
            area >= minimum_area
            and overlap >= minimum_guidance_overlap
            and overlap / float(max(1, area))
            >= minimum_guidance_fraction
        ):
            guided_labels.append(label)
        if (
            area >= minimum_area
            and core_pixels >= minimum_semantic_core_pixels
            and core_pixels / float(max(1, area))
            >= minimum_semantic_core_fraction
        ):
            semantic_fragment_labels.append(label)

    selected_labels = sorted(
        set(guided_labels) | set(semantic_fragment_labels)
    )
    if not selected_labels:
        selected_labels = [max(candidates)[3]]
    recovered_labels = sorted(set(selected_labels) - set(guided_labels))
    return selected_labels, recovered_labels


def _rgb_luma(rgb: np.ndarray) -> np.ndarray:
    pixels = np.asarray(rgb, dtype=np.float32)
    return (
        pixels[:, :, 0] * 0.2126
        + pixels[:, :, 1] * 0.7152
        + pixels[:, :, 2] * 0.0722
    )


def _inward_edge_reconstruction(
    rgb: np.ndarray,
    alpha: np.ndarray,
    *,
    band_width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Propagate trustworthy interior color through the complete soft matte."""
    support = alpha > 0
    stable_foreground = alpha >= 96
    width = max(1, int(band_width))
    core = cv2.erode(
        stable_foreground.astype(np.uint8),
        np.ones((width * 2 + 1, width * 2 + 1), dtype=np.uint8),
        iterations=1,
    ).astype(bool)
    if not np.any(core):
        return rgb.astype(np.float32), core, core

    reconstructed = rgb.astype(np.float32).copy()
    known = core.copy()
    neighbor_kernel = np.ones((3, 3), dtype=np.float32)
    # Reaching the outside of the soft matte requires crossing the eroded inner
    # band and then the semi-transparent fringe. The old implementation stopped
    # at alpha 64 and therefore left the darkest fringe untouched.
    for _ in range(width * 2 + 4):
        known_float = known.astype(np.float32)
        neighbor_count = cv2.filter2D(
            known_float,
            ddepth=-1,
            kernel=neighbor_kernel,
            borderType=cv2.BORDER_CONSTANT,
        )
        frontier = (
            (cv2.dilate(known.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
            & support
            & ~known
            & (neighbor_count > 0)
        )
        if not np.any(frontier):
            break
        for channel in range(3):
            neighbor_sum = cv2.filter2D(
                reconstructed[:, :, channel] * known_float,
                ddepth=-1,
                kernel=neighbor_kernel,
                borderType=cv2.BORDER_CONSTANT,
            )
            reconstructed[:, :, channel][frontier] = (
                neighbor_sum[frontier] / neighbor_count[frontier]
            )
        known |= frontier
    return reconstructed, core, known


def _continuous_dark_edge_candidates(
    rgb: np.ndarray,
    alpha: np.ndarray,
    reconstructed: np.ndarray,
    core: np.ndarray,
    known: np.ndarray,
    *,
    minimum_reference_luma: float,
    minimum_opaque_luma_delta: float,
    minimum_component_pixels: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    band = known & ~core
    band_pixels = int(np.count_nonzero(band))
    if not band_pixels:
        return np.zeros(alpha.shape, dtype=bool), {
            "band_pixels": 0,
            "raw_candidate_pixels": 0,
            "candidate_pixels": 0,
            "candidate_fraction": 0.0,
            "low_alpha_candidate_pixels": 0,
        }

    source_luma = _rgb_luma(rgb)
    reference_luma = _rgb_luma(reconstructed)
    luma_delta = reference_luma - source_luma
    raw_candidates = (
        band
        & (reference_luma >= float(minimum_reference_luma))
        & (
            ((alpha < 96) & (luma_delta >= 8.0))
            | (
                (alpha >= 96)
                & (luma_delta >= float(minimum_opaque_luma_delta))
            )
        )
    )
    # A real coat marking normally continues into the interior. Generated
    # silhouette strokes instead form long boundary components. Select only
    # continuous components so a tiny authentic ear or tail marking survives
    # even when another part of the frame contains an outline.
    joined = cv2.morphologyEx(
        raw_candidates.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    )
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        joined,
        connectivity=8,
    )
    candidates = np.zeros(alpha.shape, dtype=bool)
    minimum_component_area = max(4, int(minimum_component_pixels))
    for label in range(1, component_count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= minimum_component_area:
            candidates |= raw_candidates & (labels == label)

    candidate_pixels = int(np.count_nonzero(candidates))
    return candidates, {
        "band_pixels": band_pixels,
        "raw_candidate_pixels": int(np.count_nonzero(raw_candidates)),
        "candidate_pixels": candidate_pixels,
        "candidate_fraction": float(candidate_pixels) / float(band_pixels),
        "low_alpha_candidate_pixels": int(
            np.count_nonzero(candidates & (alpha < 64))
        ),
    }


def _remove_eating_contact_shadow(
    rgb: np.ndarray,
    alpha: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Remove only a thin neutral shadow attached below the detected bowl."""
    unchanged = {
        "policy": "eating_bowl_contact_shadow_v1",
        "applied": False,
        "removed_pixels": 0,
        "component_count": 0,
    }
    try:
        rgba = Image.fromarray(
            np.dstack((rgb, alpha)).astype(np.uint8),
            mode="RGBA",
        )
        _colored, prop_exclusion, prop_metadata = (
            make_eating_reference_identity_mask(rgba)
        )
    except (RuntimeError, ValueError) as exc:
        return alpha, {
            **unchanged,
            "status": "prop_split_unavailable",
            "error": f"{type(exc).__name__}: {str(exc)[:1000]}",
        }

    prop = np.asarray(prop_exclusion, dtype=np.uint8) >= 128
    rows, columns = np.where(prop)
    if not rows.size:
        return alpha, {**unchanged, "status": "empty_prop"}
    x0, x1 = int(columns.min()), int(columns.max()) + 1
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    prop_width = x1 - x0
    prop_height = y1 - y0
    luma = _rgb_luma(rgb)
    chroma = (
        rgb.max(axis=2).astype(np.int16)
        - rgb.min(axis=2).astype(np.int16)
    )
    dark_neutral = (
        (alpha >= 24)
        & (luma <= 68.0)
        & (chroma <= 44)
    )
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        dark_neutral.astype(np.uint8),
        connectivity=8,
    )
    row_grid = np.indices(alpha.shape)[0]
    lower_prop = prop & (row_grid >= y0 + int(round(prop_height * 0.52)))
    removal = np.zeros(alpha.shape, dtype=bool)
    selected_components: list[dict[str, int | float]] = []
    for label in range(1, component_count):
        x, y, width, height, area = (
            int(value) for value in stats[label]
        )
        component = labels == label
        overlap = int(np.count_nonzero(component & lower_prop))
        overlap_fraction = float(overlap) / float(max(1, area))
        aspect_ratio = float(width) / float(max(1, height))
        if (
            area >= 12
            and overlap_fraction >= 0.18
            and y + height
            >= y1 - max(4, int(round(prop_height * 0.12)))
            and width >= max(12, int(round(prop_width * 0.08)))
            and aspect_ratio >= 1.7
        ):
            removal |= component
            selected_components.append(
                {
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "pixels": area,
                    "prop_overlap_pixels": overlap,
                    "aspect_ratio": round(aspect_ratio, 6),
                }
            )
    if not selected_components:
        return alpha, {
            **unchanged,
            "status": "no_contact_shadow",
            "prop_bbox": [x0, y0, x1, y1],
            "prop_split_status": prop_metadata.get("split_status"),
        }

    # Include the one-pixel antialiased rim around the dark component, but do
    # not erase the light ceramic bowl body above it.
    removal = (
        cv2.dilate(
            removal.astype(np.uint8),
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        ).astype(bool)
        & (alpha > 0)
        & (luma < 104.0)
    )
    cleaned_alpha = alpha.copy()
    cleaned_alpha[removal] = 0
    return cleaned_alpha, {
        "policy": "eating_bowl_contact_shadow_v1",
        "applied": True,
        "removed_pixels": int(np.count_nonzero(removal)),
        "component_count": len(selected_components),
        "components": selected_components,
        "prop_bbox": [x0, y0, x1, y1],
        "prop_split_status": prop_metadata.get("split_status"),
    }


def clean_transparent_edge_halo(
    rgb: np.ndarray,
    alpha: np.ndarray,
    *,
    action: str | None = None,
    band_width: int = 9,
    minimum_opaque_luma_delta: float = 24.0,
    minimum_reference_luma: float = 64.0,
    minimum_candidate_pixels: int = 64,
    minimum_candidate_fraction: float = 0.015,
    minimum_component_pixels: int = 24,
    maximum_residual_candidate_fraction: float = 0.12,
    maximum_residual_candidate_pixel_fraction: float = 0.008,
    adjust_alpha: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Remove painted dark contours without erasing authentic dark pets.

    RGB is reconstructed from foreground-side coat pixels across every nonzero
    alpha sample. For a confirmed continuous halo, the contaminated matte is
    also pulled inward by at most two pixels and lightly softened. This avoids
    turning a black outline into an equally artificial white outline.
    """
    rgb = np.asarray(rgb, dtype=np.uint8)
    alpha = np.asarray(alpha, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Transparent edge RGB must have shape HxWx3.")
    if alpha.ndim != 2 or alpha.shape != rgb.shape[:2]:
        raise ValueError("Transparent edge alpha must match RGB dimensions.")

    contact_shadow = {
        "policy": "eating_bowl_contact_shadow_v1",
        "applied": False,
        "removed_pixels": 0,
        "component_count": 0,
        "status": "not_eating",
    }
    working_alpha = alpha.copy()
    if str(action or "").strip().lower() == "eating":
        working_alpha, contact_shadow = _remove_eating_contact_shadow(
            rgb,
            working_alpha,
        )

    reconstructed, core, known = _inward_edge_reconstruction(
        rgb,
        working_alpha,
        band_width=band_width,
    )
    if int(np.count_nonzero(core)) < int(minimum_candidate_pixels):
        cleaned_rgb = rgb.copy()
        cleaned_rgb[(working_alpha == 0) & (alpha > 0)] = 0
        return cleaned_rgb, working_alpha, {
            "policy": "local_inward_color_alpha_dark_halo_v2",
            "applied": bool(contact_shadow.get("applied")),
            "rgb_applied": False,
            "alpha_applied": bool(contact_shadow.get("applied")),
            "acceptable": True,
            "penalty": 0.0,
            "band_pixels": 0,
            "candidate_pixels": 0,
            "replaced_pixels": 0,
            "candidate_fraction": 0.0,
            "low_alpha_candidate_pixels": 0,
            "alpha_adjusted_pixels": int(
                np.count_nonzero(working_alpha != alpha)
            ),
            "contact_shadow": contact_shadow,
        }

    candidates, before = _continuous_dark_edge_candidates(
        rgb,
        working_alpha,
        reconstructed,
        core,
        known,
        minimum_reference_luma=minimum_reference_luma,
        minimum_opaque_luma_delta=minimum_opaque_luma_delta,
        minimum_component_pixels=minimum_component_pixels,
    )
    candidate_pixels = int(before["candidate_pixels"])
    candidate_fraction = float(before["candidate_fraction"])
    rgb_applied = bool(
        candidate_pixels >= int(minimum_candidate_pixels)
        and candidate_fraction >= float(minimum_candidate_fraction)
    )
    cleaned_rgb = rgb.copy()
    cleaned_alpha = working_alpha.copy()
    if rgb_applied:
        blended = np.clip(
            reconstructed * 0.98 + rgb.astype(np.float32) * 0.02,
            0,
            255,
        ).astype(np.uint8)
        cleaned_rgb[candidates] = blended[candidates]
        if adjust_alpha:
            eroded_alpha = cv2.erode(
                working_alpha,
                np.ones((5, 5), dtype=np.uint8),
                iterations=1,
            )
            cleaned_alpha[candidates] = np.minimum(
                cleaned_alpha[candidates],
                eroded_alpha[candidates],
            )
            cleaned_alpha[candidates & (working_alpha < 96)] = 0
            cleaned_alpha = cv2.GaussianBlur(
                cleaned_alpha,
                (0, 0),
                sigmaX=0.55,
                sigmaY=0.55,
            )
            cleaned_alpha[cleaned_alpha < 3] = 0
            cleaned_alpha[cleaned_alpha > 252] = 255

    cleaned_rgb[(cleaned_alpha == 0) & (alpha > 0)] = 0
    after_reconstruction, after_core, after_known = _inward_edge_reconstruction(
        cleaned_rgb,
        cleaned_alpha,
        band_width=band_width,
    )
    _residual, after = _continuous_dark_edge_candidates(
        cleaned_rgb,
        cleaned_alpha,
        after_reconstruction,
        after_core,
        after_known,
        minimum_reference_luma=minimum_reference_luma,
        minimum_opaque_luma_delta=minimum_opaque_luma_delta,
        minimum_component_pixels=minimum_component_pixels,
    )
    residual_pixels = int(after["candidate_pixels"])
    residual_fraction = float(after["candidate_fraction"])
    maximum_residual_pixels = max(
        int(minimum_candidate_pixels),
        int(round(alpha.size * maximum_residual_candidate_pixel_fraction)),
    )
    acceptable = bool(
        residual_pixels <= maximum_residual_pixels
        and residual_fraction <= float(maximum_residual_candidate_fraction)
    )
    alpha_adjusted_pixels = int(np.count_nonzero(cleaned_alpha != alpha))
    return cleaned_rgb, cleaned_alpha, {
        "policy": "local_inward_color_alpha_dark_halo_v2",
        "applied": bool(rgb_applied or contact_shadow.get("applied")),
        "rgb_applied": rgb_applied,
        "alpha_applied": bool(alpha_adjusted_pixels),
        "acceptable": acceptable,
        "penalty": round(
            min(500.0, residual_fraction * 500.0 + residual_pixels / 64.0),
            6,
        ),
        "band_pixels": int(before["band_pixels"]),
        "raw_candidate_pixels": int(before["raw_candidate_pixels"]),
        "candidate_pixels": candidate_pixels,
        "replaced_pixels": candidate_pixels if rgb_applied else 0,
        "candidate_fraction": candidate_fraction,
        "low_alpha_candidate_pixels": int(
            before["low_alpha_candidate_pixels"]
        ),
        "alpha_adjusted_pixels": alpha_adjusted_pixels,
        "contact_shadow": contact_shadow,
        "after": {
            "acceptable": acceptable,
            "penalty": round(
                min(
                    500.0,
                    residual_fraction * 500.0 + residual_pixels / 64.0,
                ),
                6,
            ),
            "band_pixels": int(after["band_pixels"]),
            "candidate_pixels": residual_pixels,
            "candidate_fraction": residual_fraction,
            "low_alpha_candidate_pixels": int(
                after["low_alpha_candidate_pixels"]
            ),
            "maximum_candidate_pixels": maximum_residual_pixels,
            "maximum_candidate_fraction": float(
                maximum_residual_candidate_fraction
            ),
        },
    }


def _remove_dark_silhouette_halo(
    rgb: np.ndarray,
    alpha: np.ndarray,
    *,
    band_width: int = 9,
    minimum_luma_delta: float = 24.0,
    minimum_reference_luma: float = 64.0,
    minimum_candidate_pixels: int = 64,
    minimum_candidate_fraction: float = 0.015,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Backward-compatible RGB-only wrapper for tests and diagnostic callers."""
    cleaned_rgb, _cleaned_alpha, metrics = clean_transparent_edge_halo(
        rgb,
        alpha,
        band_width=band_width,
        minimum_opaque_luma_delta=minimum_luma_delta,
        minimum_reference_luma=minimum_reference_luma,
        minimum_candidate_pixels=minimum_candidate_pixels,
        minimum_candidate_fraction=minimum_candidate_fraction,
        adjust_alpha=False,
    )
    return cleaned_rgb, metrics


def grooming_closed_loop_source_indices(frame_count: int) -> list[int]:
    """Keep the validated grooming half and return to its contact frame."""
    if frame_count < 3 or frame_count % 2 == 0:
        raise ValueError(
            "Grooming closed-loop publication requires an odd source frame "
            f"count of at least 3, got {frame_count}."
        )
    pivot = frame_count // 2
    indices = list(range(pivot, frame_count))
    indices.extend(range(frame_count - 2, pivot - 1, -1))
    if len(indices) != frame_count or indices[0] != indices[-1]:
        raise RuntimeError("Grooming closed-loop index construction failed.")
    return indices


def _transparent_frame_transition_score(
    first: tuple[np.ndarray, np.ndarray],
    second: tuple[np.ndarray, np.ndarray],
) -> float:
    first_rgb, first_alpha = first
    second_rgb, second_alpha = second
    first_rgba = np.dstack((first_rgb, first_alpha)).astype(np.float32)
    second_rgba = np.dstack((second_rgb, second_alpha)).astype(np.float32)
    return float(np.mean(np.abs(first_rgba - second_rgba)))


def transparent_animation_loop_quality(
    frames: list[tuple[np.ndarray, np.ndarray]],
    frame_indices: list[int],
) -> dict[str, Any]:
    """Measure the published last-to-first jump against normal motion."""
    if len(frame_indices) < 2:
        raise ValueError("Animation loop quality requires at least two frames.")
    selected = [frames[index] for index in frame_indices]
    adjacent_scores = [
        _transparent_frame_transition_score(first, second)
        for first, second in zip(selected, selected[1:])
    ]
    adjacent_median = float(np.median(adjacent_scores))
    loop_score = _transparent_frame_transition_score(selected[-1], selected[0])
    permitted_score = max(adjacent_median * 1.5, 0.5)
    ratio = (
        loop_score / adjacent_median
        if adjacent_median > 1e-6
        else (0.0 if loop_score <= 1e-6 else None)
    )
    return {
        "frame_count": len(frame_indices),
        "loop_transition_score": loop_score,
        "adjacent_transition_median": adjacent_median,
        "loop_to_adjacent_ratio": ratio,
        "permitted_transition_score": permitted_score,
        "acceptable": loop_score <= permitted_score,
    }


def build_unstabilized_transparent_gif(
    source_video: Path,
    gif_target: Path,
    *,
    action: str | None = None,
    guidance_mask_video: Path | None = None,
    output_size: tuple[int, int] = (512, 512),
    source_fps: float = 16.0,
    gif_fps: int = 20,
    gif_frames: int = 100,
    white_mp4_target: Path | None = None,
    soft_apng_target: Path | None = None,
    start_frame_index: int = 0,
) -> dict[str, Any]:
    """Extract one matte and publish clean GIF plus optional soft-alpha APNG."""
    frames_root = gif_target.parent / f".{gif_target.stem}.{uuid.uuid4().hex}.frames"
    binary_dir = frames_root / "gif"
    soft_dir = frames_root / "soft"
    temporary = gif_target.with_name(
        f".{gif_target.stem}.{uuid.uuid4().hex}.gif"
    )
    white_temporary = (
        white_mp4_target.with_name(
            f".{white_mp4_target.stem}.{uuid.uuid4().hex}.mp4"
        )
        if white_mp4_target is not None
        else None
    )
    apng_temporary = (
        soft_apng_target.with_name(
            f".{soft_apng_target.stem}.{uuid.uuid4().hex}.png"
        )
        if soft_apng_target is not None
        else None
    )
    if soft_apng_target is not None:
        soft_apng_target.unlink(missing_ok=True)
    binary_dir.mkdir(parents=True, exist_ok=False)
    soft_dir.mkdir(parents=True, exist_ok=False)
    try:
        with av.open(str(source_video)) as container:
            source_frames = [
                frame.to_ndarray(format="rgb24")
                for frame in container.decode(video=0)
            ]
        if not source_frames:
            raise RuntimeError(f"No frames decoded from {source_video}")
        source_frame_count = len(source_frames)
        source_index_map = list(range(source_frame_count))
        phase_start_index = int(start_frame_index)
        if not 0 <= phase_start_index < source_frame_count:
            raise ValueError(
                "Transparent animation phase start is outside the source: "
                f"{phase_start_index} not in 0..{source_frame_count - 1}"
            )

        guidance_frames: list[np.ndarray] | None = None
        if guidance_mask_video is not None:
            with av.open(str(guidance_mask_video)) as container:
                guidance_frames = [
                    frame.to_ndarray(format="rgb24")
                    for frame in container.decode(video=0)
                ]
            if len(guidance_frames) < len(source_frames):
                raise RuntimeError(
                    "Guidance mask video is shorter than the generated video: "
                    f"{len(guidance_frames)} < {len(source_frames)}"
                )
            guidance_frames = guidance_frames[:source_frame_count]

        if phase_start_index:
            source_frames = (
                source_frames[phase_start_index:]
                + source_frames[:phase_start_index]
            )
            source_index_map = (
                source_index_map[phase_start_index:]
                + source_index_map[:phase_start_index]
            )
            if guidance_frames is not None:
                guidance_frames = (
                    guidance_frames[phase_start_index:]
                    + guidance_frames[:phase_start_index]
                )

        temporal_cycle: dict[str, Any] = {
            "policy": None,
            "action": action,
            "applied": False,
            "closed_loop": False,
            "preserves_source_frame_count": True,
            "preserves_duration": True,
        }
        if action == "grooming":
            loop_indices = grooming_closed_loop_source_indices(
                len(source_frames)
            )
            source_frames = [source_frames[index] for index in loop_indices]
            source_index_map = [
                source_index_map[index] for index in loop_indices
            ]
            if guidance_frames is not None:
                guidance_frames = [
                    guidance_frames[index] for index in loop_indices
                ]
            temporal_cycle = {
                "policy": GROOMING_TEMPORAL_LOOP_POLICY,
                "action": action,
                "applied": True,
                "playback": "validated_second_half_forward_then_reverse",
                "pivot_source_frame_index": source_index_map[0],
                "source_index_map": source_index_map,
                "closed_loop": source_index_map[0] == source_index_map[-1],
                "preserves_source_frame_count": (
                    len(source_frames) == source_frame_count
                ),
                "preserves_duration": True,
            }

        def normalized_rgb(frame: np.ndarray) -> np.ndarray:
            if (frame.shape[1], frame.shape[0]) == output_size:
                return frame
            return np.asarray(
                Image.fromarray(frame).resize(
                    output_size, Image.Resampling.LANCZOS
                )
            )

        normalized_source = [normalized_rgb(frame) for frame in source_frames]
        chroma_masks: list[tuple[np.ndarray, np.ndarray]] = []
        chroma_usable = True
        for frame in normalized_source:
            extracted_rgb, extracted_alpha = _subject_mask(frame.copy())
            corners = (
                int(extracted_alpha[0, 0]),
                int(extracted_alpha[0, -1]),
                int(extracted_alpha[-1, 0]),
                int(extracted_alpha[-1, -1]),
            )
            if any(corners):
                chroma_usable = False
                break
            chroma_masks.append((extracted_rgb, extracted_alpha))

        extraction_method = "chroma_key"
        extracted_frames: list[tuple[np.ndarray, np.ndarray]]
        semantic_metrics: list[dict[str, float]] = []
        if chroma_usable and len(chroma_masks) == len(normalized_source):
            extracted_frames = chroma_masks
        else:
            extraction_method = "rembg_u2net_guided_by_pose_mask"
            from rembg import new_session, remove

            session = new_session(
                "u2net",
                providers=["CPUExecutionProvider"],
            )
            extracted_frames = []
            for index, frame in enumerate(normalized_source):
                result = remove(
                    Image.fromarray(frame),
                    session=session,
                    alpha_matting=False,
                )
                if not isinstance(result, Image.Image):
                    raise RuntimeError(
                        f"rembg returned an unsupported result for frame {index}"
                    )
                rgba = np.asarray(result.convert("RGBA"))
                rgb = rgba[:, :, :3].copy()
                alpha = rgba[:, :, 3].copy()

                guidance: np.ndarray | None = None
                if guidance_frames is not None:
                    guidance_rgb = normalized_rgb(guidance_frames[index])
                    guidance = (
                        (guidance_rgb[:, :, 2] >= 225)
                        & (guidance_rgb[:, :, 0] <= 30)
                        & (guidance_rgb[:, :, 1] <= 30)
                    )

                hard = (alpha >= 16).astype(np.uint8)
                component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
                    hard,
                    connectivity=8,
                )
                if component_count <= 1:
                    raise RuntimeError(
                        f"Semantic extraction found no subject in frame {index}"
                    )
                candidates: list[tuple[int, int, int]] = []
                for label in range(1, component_count):
                    area = int(stats[label, cv2.CC_STAT_AREA])
                    overlap = (
                        int(np.count_nonzero((labels == label) & guidance))
                        if guidance is not None
                        else area
                    )
                    candidates.append((overlap, area, label))
                if guidance is None:
                    selected_labels = [max(candidates)[2]]
                else:
                    selected_labels = [
                        label
                        for overlap, area, label in candidates
                        if area >= 64
                        and overlap >= 32
                        and overlap / float(max(1, area)) >= 0.20
                    ]
                    if not selected_labels:
                        selected_labels = [max(candidates)[2]]
                selected = np.isin(labels, selected_labels).astype(np.uint8)
                support = cv2.dilate(
                    selected,
                    np.ones((7, 7), dtype=np.uint8),
                    iterations=1,
                )
                alpha = np.where(support > 0, alpha, 0).astype(np.uint8)
                semantic_alpha_anchored = alpha.copy()
                recovered_semantic_fragment_labels: list[int] = []
                if guidance is not None:
                    # Guidance is only a component/noise constraint. It must
                    # never become a hard crop: a replacement animal can have
                    # a taller back, longer fur, or a different silhouette than
                    # the driving animal. Preserve every high-confidence U2Net
                    # subject pixel and its soft edge; constrain only remote
                    # low-confidence pixels that are neither near guidance nor
                    # near a high-confidence semantic core.
                    guidance_u8 = guidance.astype(np.uint8) * 255
                    guidance_envelope = cv2.dilate(
                        guidance_u8,
                        np.ones((61, 61), dtype=np.uint8),
                        iterations=1,
                    )
                    semantic_core = (
                        semantic_alpha_anchored >= 160
                    ).astype(np.uint8)
                    semantic_core_support = cv2.dilate(
                        semantic_core,
                        np.ones((11, 11), dtype=np.uint8),
                        iterations=1,
                    )
                    cleanup_support = (
                        (guidance_envelope > 0)
                        | (semantic_core_support > 0)
                    )
                    alpha = np.where(
                        cleanup_support,
                        semantic_alpha_anchored,
                        0,
                    ).astype(np.uint8)

                    restricted_hard = (alpha >= 16).astype(np.uint8)
                    restricted_count, restricted_labels, restricted_stats, _ = (
                        cv2.connectedComponentsWithStats(
                            restricted_hard,
                            connectivity=8,
                        )
                    )
                    if restricted_count <= 1:
                        raise RuntimeError(
                            "Pose-guided semantic extraction removed the subject "
                            f"in frame {index}"
                        )
                    (
                        restricted_selected_labels,
                        recovered_semantic_fragment_labels,
                    ) = _select_pose_guided_semantic_components(
                        restricted_labels,
                        restricted_stats,
                        guidance,
                        semantic_alpha_anchored >= 160,
                    )
                    restricted_component = np.isin(
                        restricted_labels,
                        restricted_selected_labels,
                    ).astype(np.uint8)
                    restricted_support = cv2.dilate(
                        restricted_component,
                        np.ones((5, 5), dtype=np.uint8),
                        iterations=1,
                    )
                    alpha = np.where(
                        restricted_support > 0,
                        alpha,
                        0,
                    ).astype(np.uint8)

                    # The post-guidance matte must retain the semantic model's
                    # high-confidence subject, globally and locally. This
                    # catches accidental missing backs, tails, ears, or limbs
                    # even when overall pose-guidance coverage still looks
                    # healthy.
                    semantic_core = semantic_alpha_anchored >= 160
                    retained_foreground = alpha >= 64
                    semantic_core_pixels = max(
                        1,
                        int(np.count_nonzero(semantic_core)),
                    )
                    semantic_core_retention = float(
                        np.count_nonzero(
                            retained_foreground & semantic_core
                        )
                    ) / float(semantic_core_pixels)

                    core_y, core_x = np.nonzero(semantic_core)
                    if core_x.size:
                        core_left = int(core_x.min())
                        core_top = int(core_y.min())
                        core_right = int(core_x.max()) + 1
                        core_bottom = int(core_y.max()) + 1
                        upper_limit = core_top + max(
                            1,
                            int(round((core_bottom - core_top) * 0.60)),
                        )
                        upper_core = semantic_core.copy()
                        upper_core[upper_limit:, :] = False
                        upper_core_pixels = max(
                            1,
                            int(np.count_nonzero(upper_core)),
                        )
                        semantic_upper_core_retention = float(
                            np.count_nonzero(
                                retained_foreground & upper_core
                            )
                        ) / float(upper_core_pixels)

                        grid_retentions: list[float] = []
                        x_edges = np.linspace(
                            core_left,
                            core_right,
                            num=5,
                            dtype=int,
                        )
                        y_edges = np.linspace(
                            core_top,
                            core_bottom,
                            num=5,
                            dtype=int,
                        )
                        for grid_y in range(4):
                            for grid_x in range(4):
                                y0 = int(y_edges[grid_y])
                                y1 = int(y_edges[grid_y + 1])
                                x0 = int(x_edges[grid_x])
                                x1 = int(x_edges[grid_x + 1])
                                cell_core = semantic_core[y0:y1, x0:x1]
                                cell_pixels = int(
                                    np.count_nonzero(cell_core)
                                )
                                if cell_pixels < 64:
                                    continue
                                cell_retained = retained_foreground[
                                    y0:y1,
                                    x0:x1,
                                ]
                                grid_retentions.append(
                                    float(
                                        np.count_nonzero(
                                            cell_retained & cell_core
                                        )
                                    )
                                    / float(cell_pixels)
                                )
                        semantic_grid_core_retention_min = min(
                            grid_retentions,
                            default=semantic_core_retention,
                        )
                    else:
                        semantic_upper_core_retention = 1.0
                        semantic_grid_core_retention_min = 1.0

                    if semantic_core_retention < 0.97:
                        raise RuntimeError(
                            "Semantic cleanup removed too much of the "
                            "high-confidence subject at frame "
                            f"{index}: retention="
                            f"{semantic_core_retention:.4f}"
                        )
                    if semantic_upper_core_retention < 0.98:
                        raise RuntimeError(
                            "Semantic cleanup damaged the upper body/back at "
                            f"frame {index}: retention="
                            f"{semantic_upper_core_retention:.4f}"
                        )
                    if semantic_grid_core_retention_min < 0.93:
                        raise RuntimeError(
                            "Semantic cleanup removed a local subject region "
                            f"at frame {index}: minimum grid retention="
                            f"{semantic_grid_core_retention_min:.4f}"
                        )
                else:
                    semantic_core_retention = 1.0
                    semantic_upper_core_retention = 1.0
                    semantic_grid_core_retention_min = 1.0
                alpha[:2, :] = 0
                alpha[-2:, :] = 0
                alpha[:, :2] = 0
                alpha[:, -2:] = 0

                foreground = alpha >= 64
                foreground_fraction = float(np.count_nonzero(foreground)) / float(
                    alpha.size
                )
                guidance_coverage = 1.0
                guidance_component_coverage_min = 1.0
                area_ratio = 1.0
                if guidance is not None:
                    guidance_pixels = max(1, int(np.count_nonzero(guidance)))
                    guidance_coverage = float(
                        np.count_nonzero(foreground & guidance)
                    ) / float(guidance_pixels)
                    area_ratio = float(np.count_nonzero(foreground)) / float(
                        guidance_pixels
                    )
                    guidance_component_count, guidance_labels, guidance_stats, _ = (
                        cv2.connectedComponentsWithStats(
                            guidance.astype(np.uint8),
                            connectivity=8,
                        )
                    )
                    guidance_component_coverages: list[float] = []
                    minimum_guidance_component = max(
                        64,
                        int(guidance_pixels * 0.005),
                    )
                    for label in range(1, guidance_component_count):
                        component_area = int(
                            guidance_stats[label, cv2.CC_STAT_AREA]
                        )
                        if component_area < minimum_guidance_component:
                            continue
                        component = guidance_labels == label
                        guidance_component_coverages.append(
                            float(np.count_nonzero(foreground & component))
                            / float(component_area)
                        )
                    guidance_component_coverage_min = min(
                        guidance_component_coverages,
                        default=guidance_coverage,
                    )
                    if guidance_coverage < 0.55 or not 0.55 <= area_ratio <= 1.85:
                        raise RuntimeError(
                            "Semantic mask diverged from pose guidance at frame "
                            f"{index}: coverage={guidance_coverage:.4f}, "
                            f"area_ratio={area_ratio:.4f}"
                        )
                    if guidance_component_coverage_min < 0.45:
                        raise RuntimeError(
                            "Semantic mask missed a guided foreground component "
                            f"at frame {index}: minimum component coverage="
                            f"{guidance_component_coverage_min:.4f}"
                        )
                if not 0.02 <= foreground_fraction <= 0.85:
                    raise RuntimeError(
                        "Semantic foreground fraction is invalid at frame "
                        f"{index}: {foreground_fraction:.4f}"
                    )

                max_red_blue = np.maximum(rgb[:, :, 0], rgb[:, :, 2])
                green_spill = (alpha > 0) & (rgb[:, :, 1] > max_red_blue)
                rgb[:, :, 1][green_spill] = max_red_blue[green_spill]
                rgb[alpha == 0] = 0
                extracted_frames.append((rgb, alpha))
                semantic_metrics.append(
                    {
                        "foreground_fraction": foreground_fraction,
                        "guidance_coverage": guidance_coverage,
                        "guidance_component_coverage_min": (
                            guidance_component_coverage_min
                        ),
                        "area_ratio": area_ratio,
                        "semantic_core_retention": (
                            semantic_core_retention
                        ),
                        "semantic_upper_core_retention": (
                            semantic_upper_core_retention
                        ),
                        "semantic_grid_core_retention_min": (
                            semantic_grid_core_retention_min
                        ),
                        "semantic_fragment_recovery_count": len(
                            recovered_semantic_fragment_labels
                        ),
                    }
                )

        edge_cleanup_metrics: list[dict[str, Any]] = []
        cleaned_extracted_frames: list[tuple[np.ndarray, np.ndarray]] = []
        for rgb, alpha in extracted_frames:
            cleaned_rgb, cleaned_alpha, cleanup_metrics = (
                clean_transparent_edge_halo(
                    rgb,
                    alpha,
                    action=action,
                )
            )
            cleaned_extracted_frames.append((cleaned_rgb, cleaned_alpha))
            edge_cleanup_metrics.append(cleanup_metrics)
        extracted_frames = cleaned_extracted_frames

        sampled_indices = [
            min(
                len(source_frames) - 1,
                int(round(index * source_fps / float(gif_fps))),
            )
            for index in range(gif_frames)
        ]
        sampled_original_source_indices = [
            source_index_map[source_index]
            for source_index in sampled_indices
        ]
        sampled_loop_quality = transparent_animation_loop_quality(
            extracted_frames,
            sampled_indices,
        )
        temporal_cycle["sampled_loop_quality"] = sampled_loop_quality
        sampled_loop_acceptable = (
            bool(temporal_cycle["closed_loop"])
            and bool(temporal_cycle["preserves_source_frame_count"])
            and bool(sampled_loop_quality["acceptable"])
            if action == "grooming"
            else True
        )
        if action == "grooming" and not sampled_loop_acceptable:
            raise RuntimeError(
                "Grooming sampled temporal loop failed its continuity contract: "
                f"{json.dumps(temporal_cycle, ensure_ascii=False)}"
            )
        foreground_fractions: list[float] = []
        corner_alpha: list[int] = []
        for output_index, source_index in enumerate(sampled_indices):
            rgb, soft_alpha = extracted_frames[source_index]
            foreground_fractions.append(
                float(np.count_nonzero(soft_alpha >= 128))
                / float(soft_alpha.size)
            )
            corner_alpha.extend(
                [
                    int(soft_alpha[0, 0]),
                    int(soft_alpha[0, -1]),
                    int(soft_alpha[-1, 0]),
                    int(soft_alpha[-1, -1]),
                ]
            )
            _save_rgba(
                soft_dir / f"frame_{output_index:03d}.png",
                rgb,
                soft_alpha,
            )
            _save_rgba(
                binary_dir / f"frame_{output_index:03d}.png",
                rgb,
                _gif_alpha(soft_alpha),
            )
        if any(corner_alpha):
            raise RuntimeError(
                "Green-screen extraction left an opaque canvas corner"
            )
        if (
            min(foreground_fractions) < 0.02
            or max(foreground_fractions) > 0.85
        ):
            raise RuntimeError(
                "Green-screen foreground fraction is outside the safe range: "
                f"{min(foreground_fractions):.4f}.."
                f"{max(foreground_fractions):.4f}"
            )

        from imageio_ffmpeg import get_ffmpeg_exe

        gif_filter = (
            "[0:v]format=rgba,split[gif][pal];"
            "[pal]palettegen=reserve_transparent=1:"
            "transparency_color=0x000000:stats_mode=diff[p];"
            "[gif][p]paletteuse=dither=none:"
            "alpha_threshold=128:diff_mode=rectangle[v]"
        )
        _run_checked(
            [
                get_ffmpeg_exe(),
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                str(gif_fps),
                "-i",
                str(binary_dir / "frame_%03d.png"),
                "-filter_complex",
                gif_filter,
                "-map",
                "[v]",
                "-frames:v",
                str(gif_frames),
                "-loop",
                "0",
                "-final_delay",
                str(int(round(100 / gif_fps))),
                str(temporary),
            ],
            "Unstabilized transparent GIF encoding",
        )
        expected_foreground_masks = [
            _gif_alpha(extracted_frames[source_index][1]) >= 128
            for source_index in sampled_indices
        ]
        validation = _validate_transparent_gif(
            temporary,
            size=output_size,
            frame_count=gif_frames,
            duration_ms=int(gif_frames * 1000 / gif_fps),
            expected_foreground_masks=expected_foreground_masks,
        )
        with Image.open(temporary) as encoded_gif:
            published_frames: list[tuple[np.ndarray, np.ndarray]] = []
            for encoded_frame in ImageSequence.Iterator(encoded_gif):
                rgba = np.asarray(encoded_frame.convert("RGBA"))
                published_frames.append(
                    (rgba[:, :, :3].copy(), rgba[:, :, 3].copy())
                )
        published_loop_quality = transparent_animation_loop_quality(
            published_frames,
            list(range(len(published_frames))),
        )
        temporal_cycle["published_loop_quality"] = published_loop_quality
        temporal_cycle["acceptable"] = (
            bool(sampled_loop_acceptable)
            and bool(published_loop_quality["acceptable"])
            if action == "grooming"
            else None
        )
        if action == "grooming" and temporal_cycle["acceptable"] is not True:
            raise RuntimeError(
                "Grooming published temporal loop failed its continuity contract: "
                f"{json.dumps(temporal_cycle, ensure_ascii=False)}"
            )
        os.replace(temporary, gif_target)
        apng_validation: dict[str, Any] | None = None
        apng_error: str | None = None
        if soft_apng_target is not None and apng_temporary is not None:
            try:
                _encode_soft_alpha_apng(
                    (
                        soft_dir / f"frame_{index:03d}.png"
                        for index in range(gif_frames)
                    ),
                    apng_temporary,
                    fps=gif_fps,
                )
                apng_validation = validate_animation(
                    apng_temporary,
                    size=output_size,
                    frame_count=gif_frames,
                    duration_ms=int(gif_frames * 1000 / gif_fps),
                )
                soft_apng_target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(apng_temporary, soft_apng_target)
            except Exception as exc:
                apng_error = f"{type(exc).__name__}: {exc}"
                apng_temporary.unlink(missing_ok=True)
        white_mp4_validation: dict[str, Any] | None = None
        if white_mp4_target is not None and white_temporary is not None:
            white_frames: list[Image.Image] = []
            for rgb, soft_alpha in extracted_frames:
                alpha_float = (
                    soft_alpha.astype(np.float32)[..., None] / 255.0
                )
                composite = np.clip(
                    rgb.astype(np.float32) * alpha_float
                    + 255.0 * (1.0 - alpha_float),
                    0,
                    255,
                ).astype(np.uint8)
                white_frames.append(Image.fromarray(composite))
            encode_video(
                white_frames,
                white_temporary,
                source_fps,
                lossless_rgb=False,
            )
            white_mp4_validation = _validate_white_background_mp4(
                white_temporary,
                size=output_size,
                frame_count=len(source_frames),
            )
            white_mp4_target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(white_temporary, white_mp4_target)
        return {
            "mode": "per_frame_foreground_extraction_no_spatial_stabilization",
            "extraction_method": extraction_method,
            "silhouette_edge_cleanup": {
                "policy": "local_inward_color_alpha_dark_halo_v2",
                "action": action,
                "applied_frames": sum(
                    1
                    for item in edge_cleanup_metrics
                    if bool(item["applied"])
                ),
                "source_frames": len(edge_cleanup_metrics),
                "replaced_pixels_total": sum(
                    int(item["replaced_pixels"])
                    for item in edge_cleanup_metrics
                ),
                "replaced_pixels_max": max(
                    (
                        int(item["replaced_pixels"])
                        for item in edge_cleanup_metrics
                    ),
                    default=0,
                ),
                "candidate_fraction_max": max(
                    (
                        float(item["candidate_fraction"])
                        for item in edge_cleanup_metrics
                    ),
                    default=0.0,
                ),
                "alpha_adjusted_pixels_total": sum(
                    int(item.get("alpha_adjusted_pixels", 0))
                    for item in edge_cleanup_metrics
                ),
                "contact_shadow_removed_pixels_total": sum(
                    int(
                        (
                            item.get("contact_shadow")
                            if isinstance(item.get("contact_shadow"), dict)
                            else {}
                        ).get("removed_pixels", 0)
                    )
                    for item in edge_cleanup_metrics
                ),
                "contact_shadow_removed_pixels_max": max(
                    (
                        int(
                            (
                                item.get("contact_shadow")
                                if isinstance(
                                    item.get("contact_shadow"),
                                    dict,
                                )
                                else {}
                            ).get("removed_pixels", 0)
                        )
                        for item in edge_cleanup_metrics
                    ),
                    default=0,
                ),
                "acceptable_frames": sum(
                    1
                    for item in edge_cleanup_metrics
                    if bool(item.get("acceptable"))
                ),
                "residual_candidate_fraction_max": max(
                    (
                        float(
                            (
                                item.get("after")
                                if isinstance(item.get("after"), dict)
                                else {}
                            ).get("candidate_fraction", 0.0)
                        )
                        for item in edge_cleanup_metrics
                    ),
                    default=0.0,
                ),
            },
            "source_frames": len(source_frames),
            "source_fps": source_fps,
            "source_phase": {
                "policy": "validated_action_contact_phase_rotation_v1",
                "start_frame_index": phase_start_index,
                "rotated": bool(phase_start_index),
            },
            "temporal_cycle": temporal_cycle,
            "sampled_source_indices": sampled_indices,
            "sampled_original_source_indices": sampled_original_source_indices,
            "foreground_fraction_min": min(foreground_fractions),
            "foreground_fraction_max": max(foreground_fractions),
            "semantic_metrics": {
                "guidance_coverage_min": min(
                    (
                        item["guidance_coverage"]
                        for item in semantic_metrics
                    ),
                    default=None,
                ),
                "guidance_component_coverage_min": min(
                    (
                        item["guidance_component_coverage_min"]
                        for item in semantic_metrics
                    ),
                    default=None,
                ),
                "area_ratio_min": min(
                    (item["area_ratio"] for item in semantic_metrics),
                    default=None,
                ),
                "area_ratio_max": max(
                    (item["area_ratio"] for item in semantic_metrics),
                    default=None,
                ),
                "semantic_core_retention_min": min(
                    (
                        item["semantic_core_retention"]
                        for item in semantic_metrics
                    ),
                    default=None,
                ),
                "semantic_upper_core_retention_min": min(
                    (
                        item["semantic_upper_core_retention"]
                        for item in semantic_metrics
                    ),
                    default=None,
                ),
                "semantic_grid_core_retention_min": min(
                    (
                        item["semantic_grid_core_retention_min"]
                        for item in semantic_metrics
                    ),
                    default=None,
                ),
            },
            "gif_edge": {
                "policy": "clean_hard_alpha_v1",
                "ordered_alpha_dither": False,
                "alpha_threshold": GIF_ALPHA_THRESHOLD,
                "alpha_smoothing_sigma": GIF_ALPHA_SMOOTHING_SIGMA,
                "palette_dither": "none",
            },
            "gif": validation,
            "apng": apng_validation,
            "apng_error": apng_error,
            "white_background_mp4": white_mp4_validation,
        }
    finally:
        temporary.unlink(missing_ok=True)
        if white_temporary is not None:
            white_temporary.unlink(missing_ok=True)
        if apng_temporary is not None:
            apng_temporary.unlink(missing_ok=True)
        shutil.rmtree(frames_root, ignore_errors=True)


def encode_video(
    frames: list[Image.Image],
    output_path: Path,
    fps: float,
    lossless_rgb: bool = False,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(output_path), mode="w")
    try:
        codec = "libx264rgb" if lossless_rgb else "libx264"
        stream = container.add_stream(codec, rate=max(1, int(round(fps))))
        stream.width = frames[0].width
        stream.height = frames[0].height
        stream.pix_fmt = "rgb24" if lossless_rgb else "yuv420p"
        stream.options = (
            {"crf": "0", "preset": "medium"}
            if lossless_rgb
            else {"crf": "12", "preset": "medium"}
        )
        for image in frames:
            video_frame = av.VideoFrame.from_ndarray(np.asarray(image.convert("RGB")), format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def verify_mask_video(path: Path, expected_count: int) -> dict[str, Any]:
    blue_count = 0
    red_count = 0
    white_count = 0
    unknown_count = 0
    decoded_count = 0
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            pixels = frame.to_ndarray(format="rgb24")
            blue = (pixels[..., 2] >= 245) & (pixels[..., 0] <= 10) & (pixels[..., 1] <= 10)
            red = (pixels[..., 0] >= 245) & (pixels[..., 1] <= 10) & (pixels[..., 2] <= 10)
            white = np.all(pixels >= 245, axis=2)
            blue_count += int(np.count_nonzero(blue))
            red_count += int(np.count_nonzero(red))
            white_count += int(np.count_nonzero(white))
            unknown_count += int(
                pixels.shape[0] * pixels.shape[1]
                - np.count_nonzero(blue | red | white)
            )
            decoded_count += 1
    if decoded_count != expected_count:
        raise ValueError(
            f"Mask video decoded {decoded_count} frames; expected {expected_count}"
        )
    total = blue_count + red_count + white_count + unknown_count
    unknown_fraction = unknown_count / max(1, total)
    if unknown_fraction > 0.02:
        raise ValueError(
            f"Mask codec altered too many identity pixels: {unknown_fraction:.4%}"
        )
    return {
        "decoded_frames": decoded_count,
        "blue_fraction": blue_count / max(1, total),
        "red_fraction": red_count / max(1, total),
        "white_fraction": white_count / max(1, total),
        "unknown_fraction": unknown_fraction,
    }


def relative_input_name(path: Path) -> str:
    return path.resolve().relative_to(COMFY_INPUT.resolve()).as_posix()


def build_prompt(
    pose_video: str,
    pose_mask_video: str,
    reference_image: str,
    reference_mask: str,
    args: argparse.Namespace,
    output_prefix: str,
) -> dict[str, Any]:
    graph: dict[str, Any] = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": UNET_NAME, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["1", 0],
                "lora_name": DPO_LORA,
                "strength_model": 1.0,
            },
        },
        "3": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["2", 0],
                "lora_name": DISTILL_LORA,
                "strength_model": 0.8,
            },
        },
        "4": {
            "class_type": "ModelSamplingSD3",
            "inputs": {"model": ["3", 0], "shift": 5.0},
        },
        "5": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": TEXT_ENCODER,
                "type": "wan",
                "device": "default",
            },
        },
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_NAME}},
        "7": {
            "class_type": "CLIPVisionLoader",
            "inputs": {"clip_name": CLIP_VISION},
        },
        "8": {"class_type": "LoadVideo", "inputs": {"file": pose_video}},
        "9": {"class_type": "GetVideoComponents", "inputs": {"video": ["8", 0]}},
        "10": {"class_type": "LoadVideo", "inputs": {"file": pose_mask_video}},
        "11": {"class_type": "GetVideoComponents", "inputs": {"video": ["10", 0]}},
        "12": {"class_type": "LoadImage", "inputs": {"image": reference_image}},
        "13": {"class_type": "LoadImage", "inputs": {"image": reference_mask}},
        "14": {
            "class_type": "CLIPVisionEncode",
            "inputs": {
                "clip_vision": ["7", 0],
                "image": ["12", 0],
                "crop": "none",
            },
        },
        "15": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": args.prompt, "clip": ["5", 0]},
        },
        "16": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": args.negative_prompt, "clip": ["5", 0]},
        },
        "17": {
            "class_type": "WanSCAILToVideo",
            "inputs": {
                "positive": ["15", 0],
                "negative": ["16", 0],
                "vae": ["6", 0],
                "width": args.width,
                "height": args.height,
                "length": args.frames,
                "batch_size": 1,
                "pose_strength": args.pose_strength,
                "pose_start": 0.0,
                "pose_end": 1.0,
                "video_frame_offset": 0,
                "previous_frame_count": 5,
                "pose_video": ["9", 0],
                "pose_video_mask": ["11", 0],
                "replacement_mode": True,
                "reference_image": ["12", 0],
                "reference_image_mask": ["13", 0],
                "clip_vision_output": ["14", 0],
            },
        },
        "18": {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": "euler"},
        },
        "19": {
            "class_type": "BasicScheduler",
            "inputs": {
                "model": ["3", 0],
                "scheduler": "simple",
                "steps": args.steps,
                "denoise": 1.0,
            },
        },
        "20": {
            "class_type": "SamplerCustom",
            "inputs": {
                "model": ["4", 0],
                "add_noise": True,
                "noise_seed": args.seed,
                "cfg": 1.0,
                "positive": ["17", 0],
                "negative": ["17", 1],
                "sampler": ["18", 0],
                "sigmas": ["19", 0],
                "latent_image": ["17", 2],
            },
        },
        "21": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["20", 1], "vae": ["6", 0]},
        },
        "22": {
            "class_type": "CreateVideo",
            "inputs": {"images": ["21", 0], "fps": args.fps, "bit_depth": 8},
        },
        "23": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["22", 0],
                "filename_prefix": output_prefix,
                "format": "mp4",
                "codec": "h264",
            },
        },
        "24": {
            "class_type": "SaveAnimatedWEBP",
            "inputs": {
                "images": ["21", 0],
                "filename_prefix": output_prefix + "_preview",
                "fps": args.fps,
                "lossless": False,
                "quality": 88,
                "method": "default",
            },
        },
    }
    return graph


def extract_output_files(value: Any) -> list[dict[str, str]]:
    files: list[dict[str, str]] = []
    if isinstance(value, dict):
        if {"filename", "subfolder"}.issubset(value):
            files.append(
                {
                    "filename": str(value["filename"]),
                    "subfolder": str(value.get("subfolder", "")),
                    "type": str(value.get("type", "output")),
                }
            )
        for nested in value.values():
            files.extend(extract_output_files(nested))
    elif isinstance(value, list):
        for nested in value:
            files.extend(extract_output_files(nested))
    return files


def wait_for_prompt(prompt_id: str, timeout_seconds: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_state = ""
    while time.monotonic() < deadline:
        history = api_json(f"/history/{prompt_id}")
        if prompt_id in history:
            entry = history[prompt_id]
            status = entry.get("status", {})
            state = str(status.get("status_str", ""))
            if state and state != last_state:
                print(f"ComfyUI status: {state}", flush=True)
                last_state = state
            if status.get("completed"):
                return entry
            messages = status.get("messages", [])
            for message in messages:
                if isinstance(message, list) and message and message[0] == "execution_error":
                    raise RuntimeError(json.dumps(message[1], ensure_ascii=False, indent=2))
        time.sleep(5)
    raise TimeoutError(f"ComfyUI prompt {prompt_id} exceeded {timeout_seconds}s")


def gpu_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, timeout=15).strip()
    except Exception as error:  # diagnostics must not block the test
        return {"error": repr(error)}
    values = [value.strip() for value in output.split(",")]
    keys = ["name", "memory_total_mib", "memory_used_mib", "memory_free_mib", "gpu_util_percent"]
    return dict(zip(keys, values))


def main() -> int:
    args = parse_args()
    if args.width % 32 or args.height % 32:
        raise ValueError("SCAIL width and height must be multiples of 32")
    if args.frames < 1 or (args.frames - 1) % 4:
        raise ValueError("SCAIL frame count must be 1 + 4*n (for example 33, 65, 81)")
    if args.action == "eating" and args.pose_strength > EATING_POSE_STRENGTH_MAX:
        raise ValueError(
            "Eating pose strength exceeds the safe fur-silhouette limit "
            f"{EATING_POSE_STRENGTH_MAX:.2f}."
        )
    if args.eating_geometry_guide_metadata and not args.eating_geometry_guide_output:
        raise ValueError(
            "--eating-geometry-guide-metadata requires "
            "--eating-geometry-guide-output"
        )
    if (
        args.grooming_geometry_guide_metadata
        and not args.grooming_geometry_guide_output
    ):
        raise ValueError(
            "--grooming-geometry-guide-metadata requires "
            "--grooming-geometry-guide-output"
        )
    if (
        args.eating_geometry_guide_output
        and args.grooming_geometry_guide_output
    ):
        raise ValueError(
            "Eating and grooming geometry-guide output modes are mutually "
            "exclusive."
        )
    if args.eating_geometry_guide_output:
        if args.action != "eating":
            raise ValueError(
                "--eating-geometry-guide-output requires --action eating"
            )
        require_files([args.source])
        guide, guide_metadata = make_eating_geometry_guide_from_gif(
            args.source,
            (args.width, args.height),
            output_count=args.frames,
            output_fps=args.fps,
        )
        args.eating_geometry_guide_output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        guide.save(
            args.eating_geometry_guide_output,
            format="PNG",
            optimize=True,
        )
        if args.eating_geometry_guide_metadata:
            args.eating_geometry_guide_metadata.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            args.eating_geometry_guide_metadata.write_text(
                json.dumps(guide_metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        print(json.dumps(guide_metadata, ensure_ascii=False), flush=True)
        return 0
    if args.grooming_geometry_guide_output:
        if args.action != "grooming":
            raise ValueError(
                "--grooming-geometry-guide-output requires --action grooming"
            )
        require_files([args.source])
        guide, guide_metadata = make_grooming_geometry_guide_from_animation(
            args.source,
            (args.width, args.height),
            output_count=args.frames,
            output_fps=args.fps,
        )
        args.grooming_geometry_guide_output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        guide.save(
            args.grooming_geometry_guide_output,
            format="PNG",
            optimize=True,
        )
        if args.grooming_geometry_guide_metadata:
            args.grooming_geometry_guide_metadata.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            args.grooming_geometry_guide_metadata.write_text(
                json.dumps(guide_metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        print(json.dumps(guide_metadata, ensure_ascii=False), flush=True)
        return 0
    require_files([args.source, args.reference])
    if args.run_id and not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}",
        args.run_id,
    ):
        raise ValueError("run-id contains unsupported characters")
    run_id = args.run_id or (
        time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        + "-"
        + uuid.uuid4().hex[:8]
    )
    run_dir = EXPERIMENT_ROOT / run_id
    input_dir = COMFY_INPUT / "scail2_test" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    input_dir.mkdir(parents=True, exist_ok=False)

    frames, durations = read_animation_frames(args.source)
    sampled = sample_timed_frames(frames, durations, args.frames, args.fps)
    reference, reference_mask, reference_metadata = make_reference_assets(
        args.reference,
        (args.width, args.height),
        separate_eating_prop=args.action == "eating",
        contain_fraction=0.88 if args.action == "grooming" else 0.84,
    )
    if args.action == "eating":
        pose_frames, pose_masks, pose_metadata = (
            make_full_eating_pose_and_mask_frames(
                sampled,
                (args.width, args.height),
            )
        )
    else:
        pose_frames, pose_masks, pose_metadata = make_pose_and_mask_frames(
            sampled,
            (args.width, args.height),
            pose_background=(255, 255, 255),
            separate_eating_prop=False,
        )
    prop_metadata = pose_metadata.get("scene_prop_metadata")
    if isinstance(prop_metadata, dict):
        prop_metadata["source_sha256"] = sha256_file(args.source)

    pose_path = input_dir / "tm_pose.mp4"
    pose_mask_path = input_dir / "tm_identity_mask.mp4"
    reference_path = input_dir / "reference.png"
    reference_mask_path = input_dir / "reference_identity_mask.png"
    encode_video(pose_frames, pose_path, args.fps, lossless_rgb=False)
    encode_video(pose_masks, pose_mask_path, args.fps, lossless_rgb=True)
    reference.save(reference_path)
    reference_mask.save(reference_mask_path)
    pose_frames[0].save(run_dir / "pose_first_frame.png")
    pose_masks[0].save(run_dir / "pose_mask_first_frame.png")
    reference.save(run_dir / "reference_normalized.png")
    reference_mask.save(run_dir / "reference_mask.png")
    mask_codec_metadata = verify_mask_video(pose_mask_path, args.frames)

    output_prefix = f"scail2_test/{run_id}/subject_replacement"
    graph = build_prompt(
        relative_input_name(pose_path),
        relative_input_name(pose_mask_path),
        relative_input_name(reference_path),
        relative_input_name(reference_mask_path),
        args,
        output_prefix,
    )
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": str(args.source),
        "reference": str(args.reference),
        "width": args.width,
        "height": args.height,
        "frames": args.frames,
        "fps": args.fps,
        "steps": args.steps,
        "seed": args.seed,
        "action": args.action,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "pose_strength": args.pose_strength,
        "transparent_gif_requested": args.transparent_gif,
        "models": {
            "unet": UNET_NAME,
            "dpo_lora": {"name": DPO_LORA, "strength": 1.0},
            "distill_lora": {"name": DISTILL_LORA, "strength": 0.8},
            "text_encoder": TEXT_ENCODER,
            "vae": VAE_NAME,
            "clip_vision": CLIP_VISION,
        },
        "pose_metadata": pose_metadata,
        "scene_prop_metadata": prop_metadata,
        "reference_metadata": reference_metadata,
        "mask_codec_metadata": mask_codec_metadata,
        "gpu_before": gpu_snapshot(),
        "input_files": {
            "pose_video": str(pose_path),
            "pose_mask_video": str(pose_mask_path),
            "reference": str(reference_path),
            "reference_mask": str(reference_mask_path),
        },
        "output_prefix": output_prefix,
    }
    (run_dir / "workflow_api.json").write_text(
        json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Prepared test inputs: {run_dir}", flush=True)
    if args.prepare_only:
        return 0

    try:
        object_info = api_json("/object_info")
        missing_nodes = sorted(set(node["class_type"] for node in graph.values()) - set(object_info))
        if missing_nodes:
            raise RuntimeError("Missing ComfyUI node classes: " + ", ".join(missing_nodes))

        started = time.monotonic()
        response = api_json("/prompt", {"prompt": graph, "client_id": run_id})
        if response.get("node_errors"):
            raise RuntimeError(
                "ComfyUI graph validation failed: "
                + json.dumps(response["node_errors"], ensure_ascii=False, indent=2)
            )
        prompt_id = str(response["prompt_id"])
        manifest["prompt_id"] = prompt_id
        manifest["queue_number"] = response.get("number")
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Submitted ComfyUI prompt: {prompt_id}", flush=True)

        history = wait_for_prompt(prompt_id, args.timeout)
        elapsed = time.monotonic() - started
        output_files = extract_output_files(history.get("outputs", {}))
        transparent_gif: dict[str, Any] | None = None
        if args.transparent_gif:
            video_items = [
                item
                for item in output_files
                if item["filename"].lower().endswith(".mp4")
                and item.get("type") == "output"
            ]
            if not video_items:
                raise RuntimeError(
                    "Transparent GIF was requested but no MP4 output was found"
                )
            item = video_items[-1]
            video_path = (
                COMFY_OUTPUT
                / item.get("subfolder", "")
                / item["filename"]
            )
            gif_path = video_path.with_name(
                video_path.stem + "_transparent_512x512_5s.gif"
            )
            apng_path = video_path.with_name(
                video_path.stem + "_transparent_512x512_5s_soft.png"
            )
            gif_metrics = build_unstabilized_transparent_gif(
                video_path,
                gif_path,
                guidance_mask_video=pose_mask_path,
                output_size=(args.width, args.height),
                source_fps=args.fps,
                soft_apng_target=apng_path,
            )
            transparent_gif = {
                "path": str(gif_path),
                "soft_apng_path": str(apng_path),
                "metrics": gif_metrics,
            }
        manifest.update(
            {
                "status": "completed",
                "elapsed_seconds": round(elapsed, 3),
                "gpu_after": gpu_snapshot(),
                "outputs": output_files,
                "transparent_gif": transparent_gif,
                "history_status": history.get("status", {}),
            }
        )
        (run_dir / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
        return 0
    except Exception as error:
        manifest.update(
            {
                "status": "failed",
                "error": repr(error),
                "gpu_after": gpu_snapshot(),
            }
        )
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
