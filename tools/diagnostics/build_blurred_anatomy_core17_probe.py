#!/usr/bin/env python3
"""Build a CPU-only low-frequency-anatomy eating probe; never submits GPU work."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np
from PIL import Image, ImageDraw


PROJECT = Path("/root/comfy-pet-pipeline")
RUNNER = PROJECT / "scail2_subject_replacement_test.py"
SOURCE = Path(
    "/data/comfy-pet-pipeline/profile_assets/sleek_tabby_cat_v1/driving/"
    "20260731-main-case-v1/eating_driver_512x512_81f_16fps.mp4"
)
BASE_RUN_ID = "codex-upload-only-eating-17f-20260731-r2"
RUN_ID = "codex-upload-only-eating-blurred-anatomy-cycle17-20260731-r3"
EXPERIMENT_ROOT = Path("/data/comfy-pet-pipeline/experiments/scail2_tm_test")
INPUT_ROOT = Path("/data/comfy-pet-pipeline/comfyui/input/scail2_test")
VALIDATION_ROOT = Path(
    "/data/comfy-pet-pipeline/archive/validation/"
    "20260731-upload-only-identity-17f-r1/blurred_anatomy_cycle17_probe_r3"
)
CONTACT_SCAN = Path(
    "/data/comfy-pet-pipeline/archive/validation/"
    "20260731-upload-only-identity-17f-r1/identityfree_core17_scan/"
    "contact_scan.json"
)
SOURCE_INDICES = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
CORE_CONTACT_INDICES = {20, 25, 30, 35, 40, 45, 50, 55, 60}
REFERENCE_SOURCE_ROOT = Path(
    "/data/comfy-pet-pipeline/archive/validation/v22028-direct-builder-repro"
)
REFERENCE_SOURCE_MANIFEST = REFERENCE_SOURCE_ROOT / "uploaded_identity_reference_manifest.json"
REFERENCE_IDENTITY_SOURCE = REFERENCE_SOURCE_ROOT / "uploaded_identity_reference.png"
REFERENCE_PROP_SOURCE = REFERENCE_SOURCE_ROOT / "main_case_eating_prop.png"
REFERENCE_IDENTITY_SHA256 = "1681e73899be95574c157cb47d84a8a3b1bf0d6fd818fe18903cfc4c92ad79dd"
REFERENCE_PROP_SHA256 = "bf526f526d3f6061592e50681a1e4c999c37cd9b7731728509c9f45998889c36"
REFERENCE_IDENTITY_SCALE = 0.66

# One deliberately selected candidate, not a parameter sweep.
LOWPASS_SIGMA = 8.0
CONTRAST_GAIN = 0.52
LUMA_MIN = 82
LUMA_MAX = 174
MASK_OPEN_KERNEL = 9
MASK_CLOSE_KERNEL = 11
MASK_ADDITION_RADIUS_PX = 2.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_runner() -> Any:
    sys.path.insert(0, str(PROJECT))
    spec = importlib.util.spec_from_file_location("scail_blurred_anatomy", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load SCAIL runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask)
    if not len(xs):
        raise RuntimeError("Empty mask")
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


def decode_rgb(path: Path) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    return frames


def component_count(mask: np.ndarray, minimum_area: int = 20) -> int:
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8
    )
    return int(sum(int(stats[i, cv2.CC_STAT_AREA]) >= minimum_area for i in range(1, count)))


def perimeter(mask: np.ndarray) -> float:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    return float(sum(cv2.arcLength(contour, True) for contour in contours))


def central_gap(identity: np.ndarray, center_x: int, prop_top: int) -> int | None:
    left, right = max(0, center_x - 12), min(identity.shape[1], center_x + 13)
    ys, _xs = np.where(identity[:prop_top, left:right])
    return None if not len(ys) else int(prop_top - int(ys.max()) - 1)


def normalized_correlation(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    aa = a[mask].astype(np.float64)
    bb = b[mask].astype(np.float64)
    aa -= aa.mean()
    bb -= bb.mean()
    denominator = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    return 0.0 if denominator <= 1e-12 else float(np.dot(aa, bb) / denominator)


def quantile_abs(values: np.ndarray, q: float) -> float:
    return float(np.quantile(np.abs(values).astype(np.float64), q))


def make_board(
    originals: list[Image.Image],
    low_frequency: list[Image.Image],
    masks: list[Image.Image],
    output: Path,
) -> None:
    selected = [0, 4, 8, 12, 16]
    rows = [originals, low_frequency, masks]
    labels = [
        "Main_case original (comparison only)",
        "Selected grayscale low-frequency anatomy + exact bowl/food",
        "Topology-limited smoother mask (blue subject / red prop)",
    ]
    cell, header, label_width = 256, 42, 285
    board = Image.new(
        "RGB", (label_width + cell * len(selected), (cell + header) * 3), "white"
    )
    draw = ImageDraw.Draw(board)
    for row, (frames, label) in enumerate(zip(rows, labels, strict=True)):
        y = row * (cell + header)
        draw.text((8, y + 14), label, fill="black")
        for column, output_index in enumerate(selected):
            x = label_width + column * cell
            frame = frames[output_index].convert("RGB").resize(
                (cell, cell),
                Image.Resampling.NEAREST if row == 2 else Image.Resampling.LANCZOS,
            )
            board.paste(frame, (x, y + header))
            draw.text(
                (x + 6, y + 14),
                f"out {output_index} / src {SOURCE_INDICES[output_index]}",
                fill="black",
            )
    board.save(output, optimize=True)


def make_zoom_board(
    originals: list[Image.Image], candidates: list[Image.Image], output: Path
) -> None:
    selected = [0, 4, 8, 12, 16]
    crop = (105, 100, 395, 475)
    cell_w, cell_h, header = 232, 300, 34
    board = Image.new("RGB", (cell_w * 5, (cell_h + header) * 2), "white")
    draw = ImageDraw.Draw(board)
    for row, frames in enumerate((originals, candidates)):
        y = row * (cell_h + header)
        draw.text(
            (8, y + 10),
            "original" if row == 0 else "low-frequency candidate",
            fill="black",
        )
        for col, idx in enumerate(selected):
            frame = frames[idx].crop(crop).resize((cell_w, cell_h), Image.Resampling.LANCZOS)
            board.paste(frame, (col * cell_w, y + header))
            draw.text((col * cell_w + 5, y + 10), f"src {SOURCE_INDICES[idx]}", fill="black")
    board.save(output, optimize=True)


def build_scaled_reference(scail: Any) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
    """Scale only uploaded identity; keep the Main_case bowl layer exact."""

    if sha256_file(REFERENCE_IDENTITY_SOURCE) != REFERENCE_IDENTITY_SHA256:
        raise RuntimeError("Unexpected uploaded-only identity source hash")
    if sha256_file(REFERENCE_PROP_SOURCE) != REFERENCE_PROP_SHA256:
        raise RuntimeError("Unexpected Main_case prop source hash")
    source_manifest = json.loads(
        REFERENCE_SOURCE_MANIFEST.read_text(encoding="utf-8")
    )
    identity_metadata = source_manifest["identity"]
    with Image.open(REFERENCE_IDENTITY_SOURCE) as loaded:
        identity_source = loaded.convert("RGBA")
    with Image.open(REFERENCE_PROP_SOURCE) as loaded:
        prop_source = loaded.convert("RGBA")
    if identity_source.size != (1024, 1024) or prop_source.size != (1024, 1024):
        raise RuntimeError("Unexpected high-resolution reference canvas")

    identity_alpha = np.asarray(identity_source.getchannel("A"), dtype=np.uint8)
    identity_box = bbox(identity_alpha >= 16)
    identity_crop = identity_source.crop(tuple(identity_box))
    scaled_size = (
        max(1, int(round(identity_crop.width * REFERENCE_IDENTITY_SCALE))),
        max(1, int(round(identity_crop.height * REFERENCE_IDENTITY_SCALE))),
    )
    scaled_crop = identity_crop.resize(scaled_size, Image.Resampling.LANCZOS)

    prop_alpha = np.asarray(prop_source.getchannel("A"), dtype=np.uint8)
    prop_box = bbox(prop_alpha >= 16)
    prop_center_x = (prop_box[0] + prop_box[2] - 1) // 2
    prop_top = prop_box[1]
    nose_center_source = [int(x) for x in identity_metadata["nose_center_canvas"]]
    nose_bottom_source = int(identity_metadata["nose_bottom_canvas"])
    nose_center_local = (
        nose_center_source[0] - identity_box[0],
        nose_center_source[1] - identity_box[1],
    )
    nose_bottom_local = nose_bottom_source - identity_box[1]
    scaled_nose_center_local = (
        int(round(nose_center_local[0] * REFERENCE_IDENTITY_SCALE)),
        int(round(nose_center_local[1] * REFERENCE_IDENTITY_SCALE)),
    )
    scaled_nose_bottom_local = int(
        round(nose_bottom_local * REFERENCE_IDENTITY_SCALE)
    )
    identity_position = (
        prop_center_x - scaled_nose_center_local[0],
        prop_top - scaled_nose_bottom_local,
    )
    identity_end = (
        identity_position[0] + scaled_size[0],
        identity_position[1] + scaled_size[1],
    )
    if (
        identity_position[0] < 0
        or identity_position[1] < 0
        or identity_end[0] > 1024
        or identity_end[1] > 1024
    ):
        raise RuntimeError(f"Scaled identity overflows canvas: {identity_position}, {identity_end}")

    identity_layer = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
    identity_layer.alpha_composite(scaled_crop, identity_position)
    placed_alpha_before_occlusion = np.asarray(
        identity_layer.getchannel("A"), dtype=np.uint8
    )
    raw_overlap = (placed_alpha_before_occlusion >= 16) & (prop_alpha >= 16)
    visible_alpha = placed_alpha_before_occlusion.copy()
    visible_alpha[prop_alpha >= 16] = 0
    identity_layer.putalpha(Image.fromarray(visible_alpha, "L"))
    visible_overlap = (visible_alpha >= 16) & (prop_alpha >= 16)

    composed = identity_layer.copy()
    composed.alpha_composite(prop_source)
    identity_layer_path = VALIDATION_ROOT / "scaled_uploaded_identity_layer_1024.png"
    prop_copy_path = VALIDATION_ROOT / "exact_main_case_prop_layer_1024.png"
    composed_path = VALIDATION_ROOT / "scaled_upload_plus_main_case_prop_reference_1024.png"
    identity_layer.save(identity_layer_path, optimize=True)
    shutil.copy2(REFERENCE_PROP_SOURCE, prop_copy_path)
    composed.save(composed_path, optimize=True)
    normalized, normalized_mask, preflight = scail.make_reference_assets(
        composed_path, (512, 512), separate_eating_prop=True
    )

    normalized_mask_array = np.asarray(normalized_mask, dtype=np.uint8)
    normalized_blue = np.all(normalized_mask_array == (0, 0, 255), axis=2)
    normalized_red = np.all(normalized_mask_array == (255, 0, 0), axis=2)
    normalized_black = np.all(normalized_mask_array == (0, 0, 0), axis=2)
    normalized_unknown = ~(
        normalized_blue | normalized_red | normalized_black
    )
    normalized_ratio = float(
        np.count_nonzero(normalized_blue) / max(np.count_nonzero(normalized_red), 1)
    )
    composed_alpha = np.asarray(composed.getchannel("A"), dtype=np.uint8)
    metadata = {
        "policy": "scaled_uploaded_identity_main_case_prop_reference_v1",
        "identity_scale": REFERENCE_IDENTITY_SCALE,
        "identity_source_path": str(REFERENCE_IDENTITY_SOURCE),
        "identity_source_sha256": sha256_file(REFERENCE_IDENTITY_SOURCE),
        "identity_source_alpha_sha256": sha256_bytes(identity_alpha.tobytes()),
        "identity_rgb_provenance": "normalized_upload_only; interpolation is geometric scaling only",
        "prop_source_path": str(REFERENCE_PROP_SOURCE),
        "prop_source_sha256": sha256_file(REFERENCE_PROP_SOURCE),
        "prop_copy_sha256": sha256_file(prop_copy_path),
        "prop_pixel_source": "Main_case eating frame0 bowl/kibble only",
        "profile_or_default_pixels_used": False,
        "identity_source_bbox_ge16": identity_box,
        "identity_scaled_size": list(scaled_size),
        "identity_position": list(identity_position),
        "identity_end": list(identity_end),
        "source_nose_center": nose_center_source,
        "source_nose_bottom": nose_bottom_source,
        "scaled_nose_center": [
            identity_position[0] + scaled_nose_center_local[0],
            identity_position[1] + scaled_nose_center_local[1],
        ],
        "scaled_nose_bottom": identity_position[1] + scaled_nose_bottom_local,
        "prop_bbox_ge16": prop_box,
        "prop_center_x": prop_center_x,
        "nose_to_prop_center_x_delta": (
            identity_position[0] + scaled_nose_center_local[0] - prop_center_x
        ),
        "nose_bottom_to_prop_top_delta": (
            identity_position[1] + scaled_nose_bottom_local - prop_top
        ),
        "raw_layer_overlap_pixels_before_prop_occlusion": int(np.count_nonzero(raw_overlap)),
        "visible_identity_prop_overlap_pixels": int(np.count_nonzero(visible_overlap)),
        "identity_border_touch_pixels": int(np.count_nonzero(
            (visible_alpha >= 16)[0, :]
            | (visible_alpha >= 16)[-1, :]
            | (visible_alpha >= 16)[:, 0]
            | (visible_alpha >= 16)[:, -1]
        )),
        "prop_border_touch_pixels": int(np.count_nonzero(
            (prop_alpha >= 16)[0, :]
            | (prop_alpha >= 16)[-1, :]
            | (prop_alpha >= 16)[:, 0]
            | (prop_alpha >= 16)[:, -1]
        )),
        "composed_alpha_sha256": sha256_bytes(composed_alpha.tobytes()),
        "identity_layer_path": str(identity_layer_path),
        "identity_layer_sha256": sha256_file(identity_layer_path),
        "composed_path": str(composed_path),
        "composed_sha256": sha256_file(composed_path),
        "normalized_preflight": preflight,
        "normalized_mask_counts": {
            "black": int(np.count_nonzero(normalized_black)),
            "blue_identity": int(np.count_nonzero(normalized_blue)),
            "red_prop": int(np.count_nonzero(normalized_red)),
            "unknown": int(np.count_nonzero(normalized_unknown)),
        },
        "normalized_identity_to_prop_area_ratio": normalized_ratio,
    }
    return normalized, normalized_mask, metadata


def main() -> int:
    for path in (VALIDATION_ROOT, EXPERIMENT_ROOT / RUN_ID, INPUT_ROOT / RUN_ID):
        if path.exists():
            raise RuntimeError(f"Refusing to overwrite: {path}")
    VALIDATION_ROOT.mkdir(parents=True)
    run_dir = EXPERIMENT_ROOT / RUN_ID
    input_dir = INPUT_ROOT / RUN_ID
    run_dir.mkdir(parents=True)
    input_dir.mkdir(parents=True)
    frame_dir = VALIDATION_ROOT / "selected_frames"
    frame_dir.mkdir()

    scail = load_runner()
    saved_scan = json.loads(CONTACT_SCAN.read_text(encoding="utf-8"))
    scan_by_frame = {int(item["frame"]): item for item in saved_scan["frames"]}
    for source_index in SOURCE_INDICES:
        contact = scan_by_frame.get(source_index)
        if not contact:
            raise RuntimeError(f"Saved contact scan lacks source frame: {source_index}")
        if (
            source_index in CORE_CONTACT_INDICES
            and int(contact.get("central_identity_prop_gap_px", -1)) != 0
        ):
            raise RuntimeError(
                f"Core-contact source frame lacks saved 0px mask contact: {source_index}"
            )

    scaled_reference, scaled_reference_mask, scaled_reference_metadata = (
        build_scaled_reference(scail)
    )

    frames, durations = scail.read_animation_frames(SOURCE)
    sampled = scail.sample_timed_frames(frames, durations, 81, 16)
    source_box = scail.expand_bbox(
        scail.union_alpha_bbox(sampled), sampled[0].size, margin_fraction=0.06
    )
    normalized = [
        scail.contain_rgba(frame.crop(source_box), (512, 512), 0.88) for frame in sampled
    ]
    prop_masks, _prop_metadata = scail.make_eating_prop_exclusion_masks(normalized)

    originals: list[Image.Image] = []
    poses: list[Image.Image] = []
    masks: list[Image.Image] = []
    raw_masks: list[np.ndarray] = []
    smooth_masks: list[np.ndarray] = []
    prop_booleans: list[np.ndarray] = []
    source_rgbs: list[np.ndarray] = []
    per_frame: list[dict[str, Any]] = []

    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MASK_OPEN_KERNEL, MASK_OPEN_KERNEL)
    )
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MASK_CLOSE_KERNEL, MASK_CLOSE_KERNEL)
    )
    for output_index, source_index in enumerate(SOURCE_INDICES):
        rgba = np.asarray(normalized[source_index].convert("RGBA"), dtype=np.uint8)
        source_rgb = rgba[..., :3]
        source_gray = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        alpha = cv2.dilate(
            (rgba[..., 3] >= 16).astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1
        ).astype(bool)
        runner_prop = np.asarray(prop_masks[source_index], dtype=np.uint8) >= 128
        runner_prop_box = bbox(runner_prop)
        prop = runner_prop.copy()
        prop[:, min(runner_prop_box[2], runner_prop_box[0] + 167) :] = False
        prop &= alpha
        raw_identity = alpha & ~prop

        opened = cv2.morphologyEx(
            raw_identity.astype(np.uint8), cv2.MORPH_OPEN, open_kernel
        ).astype(bool)
        closed = cv2.morphologyEx(
            opened.astype(np.uint8), cv2.MORPH_CLOSE, close_kernel
        ).astype(bool)
        outside_distance = cv2.distanceTransform((~raw_identity).astype(np.uint8), cv2.DIST_L2, 5)
        allowed = raw_identity | (outside_distance <= MASK_ADDITION_RADIUS_PX)
        smooth = closed & allowed & ~prop

        contact = scan_by_frame[source_index]
        bowl_box = [int(value) for value in contact["bowl_bbox"]]
        contact_x = (bowl_box[0] + bowl_box[2] - 1) // 2
        prop_top = bowl_box[1]
        yy, xx = np.indices(smooth.shape)
        contact_patch = (
            raw_identity
            & (np.abs(xx - contact_x) <= 12)
            & (yy >= max(0, prop_top - 11))
            & (yy < prop_top)
        )
        smooth |= contact_patch
        smooth &= ~prop

        raw_gap = central_gap(raw_identity, contact_x, prop_top)
        smooth_gap = central_gap(smooth, contact_x, prop_top)
        if source_index in CORE_CONTACT_INDICES and (raw_gap != 0 or smooth_gap != 0):
            raise RuntimeError(f"Contact lost at source frame {source_index}")

        low = cv2.GaussianBlur(
            source_gray, (0, 0), sigmaX=LOWPASS_SIGMA, sigmaY=LOWPASS_SIGMA,
            borderType=cv2.BORDER_REFLECT101,
        )
        inner = cv2.erode(smooth.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        center = float(np.median(low[inner]))
        compressed = np.clip(
            128.0 + CONTRAST_GAIN * (low - center), LUMA_MIN, LUMA_MAX
        ).astype(np.uint8)
        pose = np.full((512, 512, 3), 255, dtype=np.uint8)
        pose[smooth] = np.repeat(compressed[..., None], 3, axis=2)[smooth]
        pose[prop] = source_rgb[prop]

        mask_rgb = np.full((512, 512, 3), 255, dtype=np.uint8)
        mask_rgb[smooth] = (0, 0, 255)
        mask_rgb[prop] = (255, 0, 0)
        pose_image = Image.fromarray(pose, "RGB")
        mask_image = Image.fromarray(mask_rgb, "RGB")
        white = Image.new("RGB", (512, 512), "white")
        white.paste(normalized[source_index].convert("RGB"), mask=normalized[source_index].getchannel("A"))
        pose_image.save(frame_dir / f"pose_{output_index:03d}_src_{source_index:03d}.png")
        mask_image.save(frame_dir / f"mask_{output_index:03d}_src_{source_index:03d}.png")
        originals.append(white)
        poses.append(pose_image)
        masks.append(mask_image)
        raw_masks.append(raw_identity)
        smooth_masks.append(smooth)
        prop_booleans.append(prop)
        source_rgbs.append(source_rgb)

        raw_hp = source_gray - cv2.GaussianBlur(source_gray, (0, 0), 3.0)
        candidate_gray = pose[..., 0].astype(np.float32)
        candidate_hp = candidate_gray - cv2.GaussianBlur(candidate_gray, (0, 0), 3.0)
        analysis_mask = cv2.erode(smooth.astype(np.uint8), np.ones((11, 11), np.uint8)).astype(bool)
        raw_hf_rms = float(np.sqrt(np.mean(np.square(raw_hp[analysis_mask]))))
        candidate_hf_rms = float(np.sqrt(np.mean(np.square(candidate_hp[analysis_mask]))))
        original_lf = cv2.GaussianBlur(source_gray, (0, 0), 12.0)
        candidate_lf = cv2.GaussianBlur(candidate_gray, (0, 0), 12.0)
        prop_diff = np.abs(pose[prop].astype(np.int16) - source_rgb[prop].astype(np.int16))
        raw_box, smooth_box = bbox(raw_identity), bbox(smooth)
        per_frame.append(
            {
                "output_index": output_index,
                "source_index": source_index,
                "phase": (
                    "descent_or_initial_head_up"
                    if source_index < 20
                    else "core_contact"
                    if source_index <= 60
                    else "raise_or_final_head_up"
                ),
                "production_contact_acceptable": contact.get("contact_acceptable"),
                "production_contact_gap_px": contact.get("contact_gap"),
                "production_central_identity_prop_gap_px": contact.get(
                    "central_identity_prop_gap_px"
                ),
                "raw_contact_gap_px": raw_gap,
                "candidate_contact_gap_px": smooth_gap,
                "contact_adjacency_pixels": int(np.count_nonzero(
                    cv2.dilate(smooth.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & prop
                )),
                "raw_identity_bbox": raw_box,
                "candidate_identity_bbox": smooth_box,
                "candidate_top_margin_px": smooth_box[1],
                "candidate_border_touch_pixels": int(np.count_nonzero(
                    smooth[0, :] | smooth[-1, :] | smooth[:, 0] | smooth[:, -1]
                )),
                "raw_area": int(np.count_nonzero(raw_identity)),
                "candidate_area": int(np.count_nonzero(smooth)),
                "area_retention_ratio": float(np.count_nonzero(smooth) / np.count_nonzero(raw_identity)),
                "mask_xor_fraction": float(np.mean(raw_identity ^ smooth)),
                "raw_component_count_ge20": component_count(raw_identity),
                "candidate_component_count_ge20": component_count(smooth),
                "raw_perimeter": perimeter(raw_identity),
                "candidate_perimeter": perimeter(smooth),
                "perimeter_retention_ratio": perimeter(smooth) / max(perimeter(raw_identity), 1e-9),
                "new_pixels_beyond_2px": int(np.count_nonzero(smooth & (outside_distance > MASK_ADDITION_RADIUS_PX))),
                "identity_chroma_max_channel_delta": int(np.max(np.ptp(pose[smooth].astype(np.int16), axis=1), initial=0)),
                "identity_luma_range": [int(compressed[smooth].min()), int(compressed[smooth].max())],
                "raw_hf_rms_sigma3": raw_hf_rms,
                "candidate_hf_rms_sigma3": candidate_hf_rms,
                "hf_rms_ratio": candidate_hf_rms / max(raw_hf_rms, 1e-9),
                "raw_hf_p95_abs": quantile_abs(raw_hp[analysis_mask], 0.95),
                "candidate_hf_p95_abs": quantile_abs(candidate_hp[analysis_mask], 0.95),
                "hf_identity_residual_correlation": normalized_correlation(raw_hp, candidate_hp, analysis_mask),
                "low_frequency_anatomy_correlation_sigma12": normalized_correlation(original_lf, candidate_lf, analysis_mask),
                "prop_rgb_max_abs_difference": int(np.max(prop_diff, initial=0)),
                "prop_area": int(np.count_nonzero(prop)),
            }
        )

    raw_motion = [float(np.mean(raw_masks[i - 1] ^ raw_masks[i])) for i in range(1, 17)]
    candidate_motion = [float(np.mean(smooth_masks[i - 1] ^ smooth_masks[i])) for i in range(1, 17)]
    motion = {
        "raw_temporal_xor_fraction": raw_motion,
        "candidate_temporal_xor_fraction": candidate_motion,
        "mean_raw": float(np.mean(raw_motion)),
        "mean_candidate": float(np.mean(candidate_motion)),
        "mean_ratio": float(np.mean(candidate_motion) / max(np.mean(raw_motion), 1e-12)),
        "pearson_correlation": float(np.corrcoef(raw_motion, candidate_motion)[0, 1]),
        "mae": float(np.mean(np.abs(np.asarray(raw_motion) - np.asarray(candidate_motion)))),
    }
    raw_top_trajectory = [int(frame["raw_identity_bbox"][1]) for frame in per_frame]
    candidate_top_trajectory = [
        int(frame["candidate_identity_bbox"][1]) for frame in per_frame
    ]
    core_positions = [
        index
        for index, source_index in enumerate(SOURCE_INDICES)
        if source_index in CORE_CONTACT_INDICES
    ]
    endpoint_positions = [0, 1, 15, 16]
    raw_core_top = float(np.median([raw_top_trajectory[i] for i in core_positions]))
    candidate_core_top = float(
        np.median([candidate_top_trajectory[i] for i in core_positions])
    )
    raw_endpoint_top = float(
        np.median([raw_top_trajectory[i] for i in endpoint_positions])
    )
    candidate_endpoint_top = float(
        np.median([candidate_top_trajectory[i] for i in endpoint_positions])
    )
    cycle_geometry = {
        "source_indices": SOURCE_INDICES,
        "raw_top_trajectory": raw_top_trajectory,
        "candidate_top_trajectory": candidate_top_trajectory,
        "top_trajectory_pearson_correlation": float(
            np.corrcoef(raw_top_trajectory, candidate_top_trajectory)[0, 1]
        ),
        "top_trajectory_mae_px": float(
            np.mean(
                np.abs(
                    np.asarray(raw_top_trajectory)
                    - np.asarray(candidate_top_trajectory)
                )
            )
        ),
        "raw_core_top_median_px": raw_core_top,
        "candidate_core_top_median_px": candidate_core_top,
        "raw_endpoint_top_median_px": raw_endpoint_top,
        "candidate_endpoint_top_median_px": candidate_endpoint_top,
        "raw_head_raise_amplitude_px": raw_core_top - raw_endpoint_top,
        "candidate_head_raise_amplitude_px": (
            candidate_core_top - candidate_endpoint_top
        ),
        "head_raise_amplitude_retention": float(
            (candidate_core_top - candidate_endpoint_top)
            / max(raw_core_top - raw_endpoint_top, 1e-9)
        ),
    }

    pose_path = input_dir / "tm_pose.mp4"
    mask_path = input_dir / "tm_identity_mask.mp4"
    scail.encode_video(poses, pose_path, 16, lossless_rgb=True)
    scail.encode_video(masks, mask_path, 16, lossless_rgb=True)
    scaled_reference.save(input_dir / "reference.png", optimize=True)
    scaled_reference_mask.save(
        input_dir / "reference_identity_mask.png", optimize=True
    )

    decoded_pose, decoded_mask = decode_rgb(pose_path), decode_rgb(mask_path)
    decode_checks: list[dict[str, Any]] = []
    for i, (pose, mask, expected_pose, expected_mask, identity, prop, source_rgb) in enumerate(
        zip(decoded_pose, decoded_mask, poses, masks, smooth_masks, prop_booleans, source_rgbs, strict=True)
    ):
        expected_pose_array = np.asarray(expected_pose)
        expected_mask_array = np.asarray(expected_mask)
        decode_checks.append(
            {
                "frame": i,
                "pose_bit_exact": bool(np.array_equal(pose, expected_pose_array)),
                "mask_bit_exact": bool(np.array_equal(mask, expected_mask_array)),
                "pose_max_abs_difference": int(np.max(np.abs(pose.astype(np.int16) - expected_pose_array.astype(np.int16)), initial=0)),
                "mask_max_abs_difference": int(np.max(np.abs(mask.astype(np.int16) - expected_mask_array.astype(np.int16)), initial=0)),
                "decoded_identity_chroma_max_channel_delta": int(np.max(np.ptp(pose[identity].astype(np.int16), axis=1), initial=0)),
                "decoded_prop_rgb_max_abs_difference": int(np.max(np.abs(pose[prop].astype(np.int16) - source_rgb[prop].astype(np.int16)), initial=0)),
            }
        )

    workflow = json.loads((EXPERIMENT_ROOT / BASE_RUN_ID / "workflow_api.json").read_text(encoding="utf-8"))
    geometry_contract = (
        "The uploaded reference supplies identity texture and facial identity only; "
        "it must never supply close-up scale, crop, camera framing, or head-to-body "
        "proportion. The Main_case low-frequency driving video exclusively controls "
        "the complete full-body geometry, head-to-body proportion, exactly four "
        "separate attached legs and four paws, the complete tail fully inside frame, "
        "and the subject-to-bowl size ratio. Follow its full cycle: head up, natural "
        "descent, sustained muzzle-over-food eating contact, and natural head raise. "
    )
    base_positive_prompt = str(workflow["15"]["inputs"]["text"]).replace(
        "body volume, paws, tail, and the exact smooth rounded back silhouette",
        "facial identity details only",
    )
    positive_prompt = geometry_contract + base_positive_prompt
    negative_addition = (
        "oversized head, giant head, close-up crop, close-up framing, reference crop "
        "scale, tail above frame, tail outside frame, tail emerging from spine, "
        "fused torso and limbs, fused limb, black jagged outline, black zipper outline, "
        "zipper edge, "
    )
    negative_prompt = negative_addition + str(workflow["16"]["inputs"]["text"])
    workflow["15"]["inputs"]["text"] = positive_prompt
    workflow["16"]["inputs"]["text"] = negative_prompt
    workflow["8"]["inputs"]["file"] = f"scail2_test/{RUN_ID}/tm_pose.mp4"
    workflow["10"]["inputs"]["file"] = f"scail2_test/{RUN_ID}/tm_identity_mask.mp4"
    workflow["12"]["inputs"]["image"] = f"scail2_test/{RUN_ID}/reference.png"
    workflow["13"]["inputs"]["image"] = f"scail2_test/{RUN_ID}/reference_identity_mask.png"
    workflow["17"]["inputs"]["length"] = 17
    workflow["17"]["inputs"]["pose_strength"] = 1.30
    workflow["19"]["inputs"]["steps"] = 6
    workflow["20"]["inputs"]["noise_seed"] = 20260726
    workflow["23"]["inputs"]["filename_prefix"] = f"scail2_test/{RUN_ID}/subject_replacement"
    workflow["24"]["inputs"]["filename_prefix"] = f"scail2_test/{RUN_ID}/subject_replacement_preview"
    workflow_path = run_dir / "workflow_api.json"
    workflow_path.write_text(json.dumps(workflow, ensure_ascii=False, indent=2), encoding="utf-8")

    make_board(originals, poses, masks, VALIDATION_ROOT / "blurred_anatomy_visual_board.png")
    make_zoom_board(originals, poses, VALIDATION_ROOT / "blurred_anatomy_face_bowl_zoom.png")
    poses[0].save(run_dir / "pose_first_frame.png")
    masks[0].save(run_dir / "pose_mask_first_frame.png")
    shutil.copy2(input_dir / "reference.png", run_dir / "reference_normalized.png")
    shutil.copy2(input_dir / "reference_identity_mask.png", run_dir / "reference_mask.png")

    reference_mask = np.asarray(Image.open(input_dir / "reference_identity_mask.png").convert("RGB"))
    reference_identity = np.all(reference_mask == (0, 0, 255), axis=2)
    reference_prop = np.all(reference_mask == (255, 0, 0), axis=2)
    reference_area_ratio = float(np.count_nonzero(reference_identity) / max(np.count_nonzero(reference_prop), 1))
    driver_area_ratios = [float(f["candidate_area"] / max(f["prop_area"], 1)) for f in per_frame]
    implied_linear_scale = float(np.sqrt(np.median(driver_area_ratios) / reference_area_ratio))
    relative_area_ratio_error = float(
        abs(reference_area_ratio - np.median(driver_area_ratios))
        / max(np.median(driver_area_ratios), 1e-9)
    )
    scale_evaluation = {
        "applied_uploaded_identity_linear_scale": REFERENCE_IDENTITY_SCALE,
        "reference_identity_area": int(np.count_nonzero(reference_identity)),
        "reference_prop_area": int(np.count_nonzero(reference_prop)),
        "reference_identity_to_prop_area_ratio": reference_area_ratio,
        "driver_identity_to_prop_area_ratio_median": float(np.median(driver_area_ratios)),
        "driver_identity_to_prop_area_ratio_range": [float(np.min(driver_area_ratios)), float(np.max(driver_area_ratios))],
        "area_ratio_implied_identity_linear_scale": implied_linear_scale,
        "relative_area_ratio_error": relative_area_ratio_error,
        "ratio_within_15_percent": relative_area_ratio_error <= 0.15,
        "recommend_further_scale_change": bool(
            implied_linear_scale < 0.90 or implied_linear_scale > 1.10
        ),
        "decision": (
            "keep_single_scaled_reference"
            if 0.90 <= implied_linear_scale <= 1.10
            else "cpu_ratio_still_mismatched_do_not_submit"
        ),
        "reference_build": scaled_reference_metadata,
    }

    with urllib.request.urlopen("http://127.0.0.1:8188/queue", timeout=3) as response:
        queue = json.load(response)
    aggregate = {
        "hf_rms_ratio_max": float(max(f["hf_rms_ratio"] for f in per_frame)),
        "hf_rms_ratio_mean": float(np.mean([f["hf_rms_ratio"] for f in per_frame])),
        "hf_identity_residual_correlation_abs_max": float(max(abs(f["hf_identity_residual_correlation"]) for f in per_frame)),
        "low_frequency_anatomy_correlation_min": float(min(f["low_frequency_anatomy_correlation_sigma12"] for f in per_frame)),
        "perimeter_retention_ratio_max": float(max(f["perimeter_retention_ratio"] for f in per_frame)),
        "perimeter_retention_ratio_mean": float(np.mean([f["perimeter_retention_ratio"] for f in per_frame])),
        "area_retention_ratio_range": [float(min(f["area_retention_ratio"] for f in per_frame)), float(max(f["area_retention_ratio"] for f in per_frame))],
        "minimum_top_margin_px": int(min(f["candidate_top_margin_px"] for f in per_frame)),
    }
    checks = {
        "all_core_source_masks_have_saved_0px_contact": all(
            f["production_central_identity_prop_gap_px"] == 0
            for f in per_frame
            if f["source_index"] in CORE_CONTACT_INDICES
        ),
        "all_core_candidate_masks_keep_0px_contact": all(
            f["candidate_contact_gap_px"] == 0
            for f in per_frame
            if f["source_index"] in CORE_CONTACT_INDICES
        ),
        "all_props_pixel_exact_before_encode": all(f["prop_rgb_max_abs_difference"] == 0 for f in per_frame),
        "all_encoded_pose_and_mask_frames_bit_exact": all(f["pose_bit_exact"] and f["mask_bit_exact"] for f in decode_checks),
        "all_encoded_props_pixel_exact": all(f["decoded_prop_rgb_max_abs_difference"] == 0 for f in decode_checks),
        "identity_is_strict_grayscale": all(f["decoded_identity_chroma_max_channel_delta"] == 0 for f in decode_checks),
        "identity_hf_rms_ratio_at_most_0_12": aggregate["hf_rms_ratio_max"] <= 0.12,
        "low_frequency_anatomy_correlation_at_least_0_90": aggregate["low_frequency_anatomy_correlation_min"] >= 0.90,
        "no_mask_pixel_added_beyond_2px": all(f["new_pixels_beyond_2px"] == 0 for f in per_frame),
        "all_candidates_are_one_coherent_subject_component": all(
            f["candidate_component_count_ge20"] == 1 for f in per_frame
        ),
        "no_candidate_border_touch": all(f["candidate_border_touch_pixels"] == 0 for f in per_frame),
        "motion_correlation_at_least_0_995": motion["pearson_correlation"] >= 0.995,
        "full_cycle_top_trajectory_correlation_at_least_0_995": (
            cycle_geometry["top_trajectory_pearson_correlation"] >= 0.995
        ),
        "full_cycle_head_raise_amplitude_at_least_50px": (
            cycle_geometry["candidate_head_raise_amplitude_px"] >= 50.0
        ),
        "full_cycle_head_raise_amplitude_retention_at_least_0_95": (
            cycle_geometry["head_raise_amplitude_retention"] >= 0.95
        ),
        "scaled_reference_ratio_within_15_percent_of_driver": (
            scale_evaluation["ratio_within_15_percent"]
        ),
        "scaled_reference_nose_centered_on_bowl": (
            scaled_reference_metadata["nose_to_prop_center_x_delta"] == 0
        ),
        "scaled_reference_nose_touches_bowl_top": (
            scaled_reference_metadata["nose_bottom_to_prop_top_delta"] == 0
        ),
        "scaled_reference_visible_identity_prop_masks_do_not_overlap": (
            scaled_reference_metadata["visible_identity_prop_overlap_pixels"] == 0
        ),
        "scaled_reference_has_no_canvas_border_touch": (
            scaled_reference_metadata["identity_border_touch_pixels"] == 0
            and scaled_reference_metadata["prop_border_touch_pixels"] == 0
        ),
        "scaled_reference_mask_palette_has_no_unknown_pixels": (
            scaled_reference_metadata["normalized_mask_counts"]["unknown"] == 0
        ),
        "queue_empty_after": not queue.get("queue_running") and not queue.get("queue_pending"),
        "gpu_submitted": False,
        "production_modified": False,
        "profile_or_default_pixels_used": False,
    }
    manifest = {
        "status": "prepared_cpu_only",
        "run_id": RUN_ID,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "policy": "main_case_grayscale_low_frequency_anatomy_full_cycle_v2",
        "source": str(SOURCE),
        "source_sha256": sha256_file(SOURCE),
        "source_indices": SOURCE_INDICES,
        "source_union_bbox": list(source_box),
        "parameters": {
            "lowpass_sigma": LOWPASS_SIGMA,
            "contrast_gain": CONTRAST_GAIN,
            "luma_clip": [LUMA_MIN, LUMA_MAX],
            "mask_open_kernel": MASK_OPEN_KERNEL,
            "mask_close_kernel": MASK_CLOSE_KERNEL,
            "mask_addition_radius_px": MASK_ADDITION_RADIUS_PX,
        },
        "provenance": {
            "driver_subject_pixels": "Main_case eating only, grayscale + strong Gaussian low-pass + contrast compression",
            "prop_pixels": "Main_case eating bowl/food exact RGB",
            "reference_pixels": "uploaded subject reference from validated direct builder",
            "profile_or_default_pixels": "none",
        },
        "per_frame": per_frame,
        "aggregate": aggregate,
        "motion": motion,
        "cycle_geometry": cycle_geometry,
        "reference_scale_evaluation": scale_evaluation,
        "conditioning_contract": {
            "reference_role": "identity_texture_and_face_only_not_scale_or_crop",
            "geometry_role": "Main_case_low_frequency_driver_exclusive",
            "positive_prompt": positive_prompt,
            "positive_prompt_sha256": sha256_bytes(
                positive_prompt.encode("utf-8")
            ),
            "negative_prompt": negative_prompt,
            "negative_prompt_sha256": sha256_bytes(
                negative_prompt.encode("utf-8")
            ),
        },
        "decode_checks": decode_checks,
        "checks": checks,
        "all_cpu_checks_passed": all(value is True or value is False and key in {"gpu_submitted", "production_modified", "profile_or_default_pixels_used"} for key, value in checks.items()),
        "workflow": {"path": str(workflow_path), "sha256": sha256_file(workflow_path), "submitted": False},
        "artifacts": {
            "pose_video": {"path": str(pose_path), "sha256": sha256_file(pose_path)},
            "mask_video": {"path": str(mask_path), "sha256": sha256_file(mask_path)},
            "reference": {"path": str(input_dir / "reference.png"), "sha256": sha256_file(input_dir / "reference.png")},
            "reference_mask": {"path": str(input_dir / "reference_identity_mask.png"), "sha256": sha256_file(input_dir / "reference_identity_mask.png")},
        },
        "queue_after": queue,
        "gpu_submitted": False,
        "production_modified": False,
    }
    (VALIDATION_ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "status": manifest["status"],
        "run_id": RUN_ID,
        "all_cpu_checks_passed": manifest["all_cpu_checks_passed"],
        "aggregate": aggregate,
        "motion": motion,
        "cycle_geometry": cycle_geometry,
        "reference_scale_evaluation": scale_evaluation,
        "checks": checks,
        "artifacts": manifest["artifacts"],
        "visual_board": str(VALIDATION_ROOT / "blurred_anatomy_visual_board.png"),
        "zoom_board": str(VALIDATION_ROOT / "blurred_anatomy_face_bowl_zoom.png"),
    }
    (VALIDATION_ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
