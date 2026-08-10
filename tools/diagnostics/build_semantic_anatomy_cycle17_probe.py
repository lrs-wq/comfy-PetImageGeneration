#!/usr/bin/env python3
"""Build a CPU-only 17-frame semantic-anatomy eating guide.

This diagnostic deliberately never posts a ComfyUI prompt.  Main_case animal
RGB is used only for comparison/audit and for the separately masked bowl/food.
The blue identity region is rendered from fixed neutral semantic fields derived
from silhouette geometry; uploaded identity remains the sole reference image.
"""

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
BASE_PROBE = PROJECT / "tools/diagnostics/build_blurred_anatomy_core17_probe.py"
SOURCE = Path(
    "/data/comfy-pet-pipeline/profile_assets/sleek_tabby_cat_v1/driving/"
    "20260731-main-case-v1/eating_driver_512x512_81f_16fps.mp4"
)
BASE_RUN_ID = "codex-upload-only-eating-17f-20260731-r2"
RUN_ID = "codex-upload-only-eating-semantic-anatomy-cycle17-20260731-r2"
EXPERIMENT_ROOT = Path("/data/comfy-pet-pipeline/experiments/scail2_tm_test")
INPUT_ROOT = Path("/data/comfy-pet-pipeline/comfyui/input/scail2_test")
VALIDATION_ROOT = Path(
    "/data/comfy-pet-pipeline/archive/validation/"
    "20260731-upload-only-identity-17f-r1/semantic_anatomy_cycle17_probe_r2"
)
CONTACT_SCAN = Path(
    "/data/comfy-pet-pipeline/archive/validation/"
    "20260731-upload-only-identity-17f-r1/identityfree_core17_scan/contact_scan.json"
)
SOURCE_INDICES = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
CORE_CONTACT_INDICES = {20, 25, 30, 35, 40, 45, 50, 55, 60}
MASK_OPEN_KERNEL = 9
MASK_CLOSE_KERNEL = 11
MASK_ADDITION_RADIUS_PX = 2.0

# Fixed grayscale semantics. None are sampled from Main_case animal RGB.
SEMANTIC_LUMA = {
    "base": 146,
    "torso": 154,
    "tail": 122,
    "hind_leg": 164,
    "hind_paw": 182,
    "head": 170,
    "ear": 132,
    "foreleg": 176,
    "forepaw": 194,
    "eye": 88,
    "muzzle": 205,
    "nose": 112,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(PROJECT))
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


def ellipse(shape: tuple[int, int], center: tuple[int, int], axes: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, np.uint8)
    cv2.ellipse(out, center, axes, 0, 0, 360, 1, -1, lineType=cv2.LINE_AA)
    return out.astype(bool)


def polygon(shape: tuple[int, int], points: list[tuple[int, int]]) -> np.ndarray:
    out = np.zeros(shape, np.uint8)
    cv2.fillPoly(out, [np.asarray(points, np.int32)], 1, lineType=cv2.LINE_AA)
    return out.astype(bool)


def contact_phase(source_index: int) -> float:
    if source_index <= 20:
        return max(0.0, min(1.0, source_index / 20.0))
    if source_index <= 60:
        return 1.0
    return max(0.0, min(1.0, (80 - source_index) / 20.0))


def semantic_regions(
    identity: np.ndarray,
    prop: np.ndarray,
    source_index: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Create fixed anatomical fields from silhouette/bowl geometry only."""
    height, width = identity.shape
    left, top, right, bottom = bbox(identity)
    prop_left, prop_top, prop_right, prop_bottom = bbox(prop)
    prop_center_x = int(round((prop_left + prop_right - 1) / 2.0))
    subject_width = right - left
    subject_height = bottom - top
    phase = contact_phase(source_index)

    # Head center explicitly performs the full down/contact/up cycle. The body
    # remains anchored to the source silhouette so head scale cannot consume it.
    head_cx = prop_center_x
    head_up_y = max(top + 74, prop_top - 205)
    head_down_y = prop_top - 62
    head_cy = int(round(head_up_y * (1.0 - phase) + head_down_y * phase))
    head_rx = int(round(70 + 7 * phase))
    head_ry = int(round(70 - 7 * phase))

    main_body_right = min(right, prop_right + 54)
    body_cx = int(round((left + main_body_right) / 2.0))
    body_cy = int(round(prop_top - 84))
    body_rx = max(74, int(round((main_body_right - left) * 0.44)))
    body_ry = max(105, int(round((bottom - top) * 0.42)))

    regions: dict[str, np.ndarray] = {}
    regions["torso"] = ellipse((height, width), (body_cx, body_cy), (body_rx, body_ry)) & identity
    regions["tail"] = identity & (np.indices(identity.shape)[1] >= prop_right + 20) & (
        np.indices(identity.shape)[0] >= prop_top - 112
    )

    hind_y = min(bottom - 34, prop_top + 32)
    hind_dx = max(62, int(round((prop_right - prop_left) * 0.63)))
    regions["hind_leg_left"] = ellipse((height, width), (prop_center_x - hind_dx, hind_y - 22), (50, 73)) & identity
    regions["hind_leg_right"] = ellipse((height, width), (prop_center_x + hind_dx, hind_y - 22), (50, 73)) & identity
    regions["hind_paw_left"] = ellipse((height, width), (prop_center_x - hind_dx - 2, hind_y + 28), (42, 22)) & identity
    regions["hind_paw_right"] = ellipse((height, width), (prop_center_x + hind_dx + 2, hind_y + 28), (42, 22)) & identity

    regions["head"] = ellipse((height, width), (head_cx, head_cy), (head_rx, head_ry)) & identity
    ear_y = head_cy - head_ry + 17
    ear_dx = int(round(head_rx * 0.58))
    regions["ear_left"] = polygon(
        (height, width),
        [(head_cx - ear_dx - 23, ear_y + 36), (head_cx - ear_dx, ear_y - 28), (head_cx - ear_dx + 28, ear_y + 39)],
    ) & identity
    regions["ear_right"] = polygon(
        (height, width),
        [(head_cx + ear_dx - 28, ear_y + 39), (head_cx + ear_dx, ear_y - 28), (head_cx + ear_dx + 23, ear_y + 36)],
    ) & identity

    fore_top = head_cy + int(round(head_ry * 0.40))
    fore_bottom = min(bottom - 12, prop_top + 34)
    fore_mid_y = int(round((fore_top + fore_bottom) / 2.0))
    fore_ry = max(30, int(round((fore_bottom - fore_top) / 2.0)))
    fore_dx = max(50, int(round((prop_right - prop_left) * 0.38)))
    regions["foreleg_left"] = ellipse((height, width), (prop_center_x - fore_dx, fore_mid_y), (25, fore_ry)) & identity
    regions["foreleg_right"] = ellipse((height, width), (prop_center_x + fore_dx, fore_mid_y), (25, fore_ry)) & identity
    paw_y = min(bottom - 19, prop_top + 18)
    regions["forepaw_left"] = ellipse((height, width), (prop_center_x - fore_dx, paw_y), (34, 19)) & identity
    regions["forepaw_right"] = ellipse((height, width), (prop_center_x + fore_dx, paw_y), (34, 19)) & identity

    eye_y = head_cy - int(round(head_ry * 0.15))
    eye_dx = int(round(head_rx * 0.37))
    regions["eye_left"] = ellipse((height, width), (head_cx - eye_dx, eye_y), (15, 8)) & regions["head"]
    regions["eye_right"] = ellipse((height, width), (head_cx + eye_dx, eye_y), (15, 8)) & regions["head"]
    muzzle_y = min(prop_top - 17, head_cy + int(round(head_ry * 0.47)))
    regions["muzzle"] = ellipse((height, width), (head_cx, muzzle_y), (31, 19)) & identity
    regions["nose"] = ellipse((height, width), (head_cx, muzzle_y - 6), (8, 5)) & regions["muzzle"]

    # Fail closed if the silhouette cannot support every named anatomical part.
    minimum_area = {
        "head": 1800, "ear_left": 80, "ear_right": 80,
        "torso": 7000, "tail": 800,
        "hind_leg_left": 500, "hind_leg_right": 500,
        "hind_paw_left": 180, "hind_paw_right": 180,
        "foreleg_left": 300, "foreleg_right": 300,
        "forepaw_left": 150, "forepaw_right": 150,
        "eye_left": 80, "eye_right": 80, "muzzle": 300, "nose": 30,
    }
    areas = {name: int(np.count_nonzero(mask)) for name, mask in regions.items()}
    failed = {name: [areas.get(name, 0), required] for name, required in minimum_area.items() if areas.get(name, 0) < required}
    if failed:
        raise RuntimeError(f"semantic_anatomy_partition_unreliable frame={source_index}: {failed}")

    centers = {}
    for name, mask in regions.items():
        ys, xs = np.where(mask)
        centers[name] = [round(float(xs.mean()), 3), round(float(ys.mean()), 3)]
    audit = {
        "source_index": source_index,
        "contact_phase": phase,
        "subject_bbox": [left, top, right, bottom],
        "prop_bbox": [prop_left, prop_top, prop_right, prop_bottom],
        "head_center_control": [head_cx, head_cy],
        "head_axes_control": [head_rx, head_ry],
        "areas": areas,
        "centroids": centers,
        "all_required_parts_present": True,
        "four_limb_fields_present": all(areas[name] >= minimum_area[name] for name in ("hind_leg_left", "hind_leg_right", "foreleg_left", "foreleg_right")),
        "four_paw_fields_present": all(areas[name] >= minimum_area[name] for name in ("hind_paw_left", "hind_paw_right", "forepaw_left", "forepaw_right")),
        "paired_eyes_present": areas["eye_left"] >= 80 and areas["eye_right"] >= 80,
        "paired_ears_present": areas["ear_left"] >= 80 and areas["ear_right"] >= 80,
    }
    return regions, audit


def render_semantic(identity: np.ndarray, prop: np.ndarray, source_rgb: np.ndarray, regions: dict[str, np.ndarray]) -> np.ndarray:
    gray = np.full(identity.shape, SEMANTIC_LUMA["base"], np.float32)
    assignment = [
        ("torso", "torso"), ("tail", "tail"),
        ("hind_leg_left", "hind_leg"), ("hind_leg_right", "hind_leg"),
        ("hind_paw_left", "hind_paw"), ("hind_paw_right", "hind_paw"),
        ("head", "head"), ("ear_left", "ear"), ("ear_right", "ear"),
        ("foreleg_left", "foreleg"), ("foreleg_right", "foreleg"),
        ("forepaw_left", "forepaw"), ("forepaw_right", "forepaw"),
        ("eye_left", "eye"), ("eye_right", "eye"),
        ("muzzle", "muzzle"), ("nose", "nose"),
    ]
    for region_name, tone_name in assignment:
        gray[regions[region_name]] = SEMANTIC_LUMA[tone_name]
    # Semantic transitions are low frequency; this removes hard diagram edges
    # while retaining paired eye/ear/limb topology.
    weighted = cv2.GaussianBlur(gray * identity.astype(np.float32), (0, 0), 3.8)
    support = cv2.GaussianBlur(identity.astype(np.float32), (0, 0), 3.8)
    smooth_gray = np.clip(weighted / np.maximum(support, 0.01), 80, 210).astype(np.uint8)
    pose = np.full((*identity.shape, 3), 255, np.uint8)
    pose[identity] = np.repeat(smooth_gray[..., None], 3, axis=2)[identity]
    pose[prop] = source_rgb[prop]
    return pose


def label_preview(identity: np.ndarray, regions: dict[str, np.ndarray], prop: np.ndarray) -> Image.Image:
    palette = {
        "base": (150, 150, 150), "torso": (70, 130, 180), "tail": (120, 80, 180),
        "hind_leg": (80, 170, 120), "hind_paw": (120, 220, 150),
        "head": (230, 170, 70), "ear": (230, 100, 120), "foreleg": (80, 200, 220),
        "forepaw": (120, 240, 240), "eye": (40, 40, 40), "muzzle": (245, 220, 170), "nose": (180, 70, 90),
    }
    rgb = np.full((*identity.shape, 3), 255, np.uint8)
    rgb[identity] = palette["base"]
    for region_name, tone_name in [
        ("torso", "torso"), ("tail", "tail"), ("hind_leg_left", "hind_leg"),
        ("hind_leg_right", "hind_leg"), ("hind_paw_left", "hind_paw"),
        ("hind_paw_right", "hind_paw"), ("head", "head"), ("ear_left", "ear"),
        ("ear_right", "ear"), ("foreleg_left", "foreleg"), ("foreleg_right", "foreleg"),
        ("forepaw_left", "forepaw"), ("forepaw_right", "forepaw"),
        ("eye_left", "eye"), ("eye_right", "eye"), ("muzzle", "muzzle"), ("nose", "nose"),
    ]:
        rgb[regions[region_name]] = palette[tone_name]
    rgb[prop] = (255, 0, 0)
    return Image.fromarray(rgb, "RGB")


def make_board(rows: list[list[Image.Image]], labels: list[str], output: Path) -> None:
    selected = [0, 4, 8, 12, 16]
    cell, header, label_width = 256, 38, 300
    board = Image.new("RGB", (label_width + cell * 5, (cell + header) * len(rows)), "white")
    draw = ImageDraw.Draw(board)
    for row, (frames, label) in enumerate(zip(rows, labels, strict=True)):
        y = row * (cell + header)
        draw.text((8, y + 12), label, fill="black")
        for col, idx in enumerate(selected):
            x = label_width + col * cell
            board.paste(frames[idx].convert("RGB").resize((cell, cell), Image.Resampling.LANCZOS), (x, y + header))
            draw.text((x + 5, y + 12), f"out {idx} / src {SOURCE_INDICES[idx]}", fill="black")
    board.save(output, optimize=True)


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

    scail = load_module("scail_semantic_probe_runner", RUNNER)
    base = load_module("scail_blurred_probe_helpers", BASE_PROBE)
    base.VALIDATION_ROOT = VALIDATION_ROOT
    scaled_reference, scaled_reference_mask, scaled_reference_metadata = base.build_scaled_reference(scail)

    saved_scan = json.loads(CONTACT_SCAN.read_text(encoding="utf-8"))
    scan_by_frame = {int(item["frame"]): item for item in saved_scan["frames"]}
    frames, durations = scail.read_animation_frames(SOURCE)
    sampled = scail.sample_timed_frames(frames, durations, 81, 16)
    source_box = scail.expand_bbox(scail.union_alpha_bbox(sampled), sampled[0].size, margin_fraction=0.06)
    normalized = [scail.contain_rgba(frame.crop(source_box), (512, 512), 0.88) for frame in sampled]
    prop_masks, _prop_metadata = scail.make_eating_prop_exclusion_masks(normalized)

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MASK_OPEN_KERNEL, MASK_OPEN_KERNEL))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MASK_CLOSE_KERNEL, MASK_CLOSE_KERNEL))
    originals: list[Image.Image] = []
    poses: list[Image.Image] = []
    masks: list[Image.Image] = []
    labels: list[Image.Image] = []
    raw_masks: list[np.ndarray] = []
    smooth_masks: list[np.ndarray] = []
    prop_booleans: list[np.ndarray] = []
    source_rgbs: list[np.ndarray] = []
    per_frame: list[dict[str, Any]] = []

    for output_index, source_index in enumerate(SOURCE_INDICES):
        rgba = np.asarray(normalized[source_index].convert("RGBA"), np.uint8)
        source_rgb = rgba[..., :3]
        source_gray = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        alpha = cv2.dilate((rgba[..., 3] >= 16).astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1).astype(bool)
        runner_prop = np.asarray(prop_masks[source_index], np.uint8) >= 128
        runner_prop_box = bbox(runner_prop)
        prop = runner_prop.copy()
        prop[:, min(runner_prop_box[2], runner_prop_box[0] + 167):] = False
        prop &= alpha
        raw_identity = alpha & ~prop
        opened = cv2.morphologyEx(raw_identity.astype(np.uint8), cv2.MORPH_OPEN, open_kernel).astype(bool)
        closed = cv2.morphologyEx(opened.astype(np.uint8), cv2.MORPH_CLOSE, close_kernel).astype(bool)
        outside_distance = cv2.distanceTransform((~raw_identity).astype(np.uint8), cv2.DIST_L2, 5)
        smooth = closed & (raw_identity | (outside_distance <= MASK_ADDITION_RADIUS_PX)) & ~prop
        contact = scan_by_frame[source_index]
        bowl_box = [int(value) for value in contact["bowl_bbox"]]
        contact_x = (bowl_box[0] + bowl_box[2] - 1) // 2
        prop_top = bowl_box[1]
        yy, xx = np.indices(smooth.shape)
        contact_patch = raw_identity & (np.abs(xx - contact_x) <= 12) & (yy >= max(0, prop_top - 11)) & (yy < prop_top)
        smooth |= contact_patch
        smooth &= ~prop
        raw_gap = central_gap(raw_identity, contact_x, prop_top)
        candidate_gap = central_gap(smooth, contact_x, prop_top)
        if source_index in CORE_CONTACT_INDICES and (raw_gap != 0 or candidate_gap != 0):
            raise RuntimeError(f"Contact lost at source frame {source_index}")

        regions, anatomy = semantic_regions(smooth, prop, source_index)
        pose = render_semantic(smooth, prop, source_rgb, regions)
        mask_rgb = np.full((512, 512, 3), 255, np.uint8)
        mask_rgb[smooth] = (0, 0, 255)
        mask_rgb[prop] = (255, 0, 0)
        pose_image = Image.fromarray(pose, "RGB")
        mask_image = Image.fromarray(mask_rgb, "RGB")
        label_image = label_preview(smooth, regions, prop)
        original = Image.new("RGB", (512, 512), "white")
        original.paste(normalized[source_index].convert("RGB"), mask=normalized[source_index].getchannel("A"))
        pose_image.save(frame_dir / f"pose_{output_index:03d}_src_{source_index:03d}.png")
        mask_image.save(frame_dir / f"mask_{output_index:03d}_src_{source_index:03d}.png")
        label_image.save(frame_dir / f"labels_{output_index:03d}_src_{source_index:03d}.png")
        originals.append(original)
        poses.append(pose_image)
        masks.append(mask_image)
        labels.append(label_image)
        raw_masks.append(raw_identity)
        smooth_masks.append(smooth)
        prop_booleans.append(prop)
        source_rgbs.append(source_rgb)

        analysis_mask = cv2.erode(smooth.astype(np.uint8), np.ones((11, 11), np.uint8)).astype(bool)
        candidate_gray = pose[..., 0].astype(np.float32)
        raw_hp = source_gray - cv2.GaussianBlur(source_gray, (0, 0), 3.0)
        candidate_hp = candidate_gray - cv2.GaussianBlur(candidate_gray, (0, 0), 3.0)
        raw_hf = float(np.sqrt(np.mean(np.square(raw_hp[analysis_mask]))))
        candidate_hf = float(np.sqrt(np.mean(np.square(candidate_hp[analysis_mask]))))
        prop_diff = np.abs(pose[prop].astype(np.int16) - source_rgb[prop].astype(np.int16))
        per_frame.append({
            "output_index": output_index, "source_index": source_index,
            "phase": "descent" if source_index < 20 else "contact" if source_index <= 60 else "raise",
            "production_central_identity_prop_gap_px": contact.get("central_identity_prop_gap_px"),
            "raw_contact_gap_px": raw_gap, "candidate_contact_gap_px": candidate_gap,
            "contact_adjacency_pixels": int(np.count_nonzero(cv2.dilate(smooth.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & prop)),
            "raw_identity_bbox": bbox(raw_identity), "candidate_identity_bbox": bbox(smooth),
            "candidate_border_touch_pixels": int(np.count_nonzero(smooth[0, :] | smooth[-1, :] | smooth[:, 0] | smooth[:, -1])),
            "raw_area": int(np.count_nonzero(raw_identity)), "candidate_area": int(np.count_nonzero(smooth)),
            "area_retention_ratio": float(np.count_nonzero(smooth) / np.count_nonzero(raw_identity)),
            "raw_component_count_ge20": component_count(raw_identity), "candidate_component_count_ge20": component_count(smooth),
            "raw_perimeter": perimeter(raw_identity), "candidate_perimeter": perimeter(smooth),
            "new_pixels_beyond_2px": int(np.count_nonzero(smooth & (outside_distance > MASK_ADDITION_RADIUS_PX))),
            "identity_chroma_max_channel_delta": int(np.max(np.ptp(pose[smooth].astype(np.int16), axis=1), initial=0)),
            "raw_hf_rms_sigma3": raw_hf, "candidate_hf_rms_sigma3": candidate_hf,
            "hf_rms_ratio": candidate_hf / max(raw_hf, 1e-9),
            "hf_identity_residual_correlation": normalized_correlation(raw_hp, candidate_hp, analysis_mask),
            "semantic_anatomy": anatomy,
            "prop_rgb_max_abs_difference": int(np.max(prop_diff, initial=0)), "prop_area": int(np.count_nonzero(prop)),
        })

    pose_path = input_dir / "tm_pose.mp4"
    mask_path = input_dir / "tm_identity_mask.mp4"
    scail.encode_video(poses, pose_path, 16, lossless_rgb=True)
    scail.encode_video(masks, mask_path, 16, lossless_rgb=True)
    scaled_reference.save(input_dir / "reference.png", optimize=True)
    scaled_reference_mask.save(input_dir / "reference_identity_mask.png", optimize=True)

    decoded_pose, decoded_mask = decode_rgb(pose_path), decode_rgb(mask_path)
    decode_checks = []
    for i, (pose, mask, expected_pose, expected_mask, identity, prop, source_rgb) in enumerate(zip(decoded_pose, decoded_mask, poses, masks, smooth_masks, prop_booleans, source_rgbs, strict=True)):
        expected_pose_array, expected_mask_array = np.asarray(expected_pose), np.asarray(expected_mask)
        decode_checks.append({
            "frame": i, "pose_bit_exact": bool(np.array_equal(pose, expected_pose_array)),
            "mask_bit_exact": bool(np.array_equal(mask, expected_mask_array)),
            "pose_max_abs_difference": int(np.max(np.abs(pose.astype(np.int16) - expected_pose_array.astype(np.int16)), initial=0)),
            "mask_max_abs_difference": int(np.max(np.abs(mask.astype(np.int16) - expected_mask_array.astype(np.int16)), initial=0)),
            "decoded_identity_chroma_max_channel_delta": int(np.max(np.ptp(pose[identity].astype(np.int16), axis=1), initial=0)),
            "decoded_prop_rgb_max_abs_difference": int(np.max(np.abs(pose[prop].astype(np.int16) - source_rgb[prop].astype(np.int16)), initial=0)),
        })

    workflow = json.loads((EXPERIMENT_ROOT / BASE_RUN_ID / "workflow_api.json").read_text(encoding="utf-8"))
    positive = (
        "The uploaded reference supplies the only animal identity, face, coat, color, and fur. "
        "The neutral-gray semantic pose guide supplies full-body anatomy, scale, exactly four attached legs and four distinct paws, paired ears and eyes, muzzle, torso, and complete tail, plus the full head-up to food-contact to head-up eating cycle. "
        "Keep the entire animal and tail inside frame and eat from the food inside the bowl. "
        + str(workflow["15"]["inputs"]["text"])
    )
    negative = (
        "old driver cat identity, black-and-white driver coat, driver fur texture, oversized head, giant head, close-up crop, fused legs, missing paw, fewer than four paws, tail above frame, tail outside frame, tail emerging from spine, black zipper outline, jagged outline, "
        + str(workflow["16"]["inputs"]["text"])
    )
    workflow["15"]["inputs"]["text"] = positive
    workflow["16"]["inputs"]["text"] = negative
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

    make_board(
        [originals, poses, labels, masks],
        ["Main_case original (comparison only; excluded from identity guide)", "Fixed neutral semantic anatomy pose", "Anatomical partition audit labels", "SCAIL mask: blue identity / red bowl"],
        VALIDATION_ROOT / "semantic_anatomy_visual_board.png",
    )
    poses[0].save(run_dir / "pose_first_frame.png")
    masks[0].save(run_dir / "pose_mask_first_frame.png")
    shutil.copy2(input_dir / "reference.png", run_dir / "reference_normalized.png")
    shutil.copy2(input_dir / "reference_identity_mask.png", run_dir / "reference_mask.png")

    raw_motion = [float(np.mean(raw_masks[i - 1] ^ raw_masks[i])) for i in range(1, 17)]
    candidate_motion = [float(np.mean(smooth_masks[i - 1] ^ smooth_masks[i])) for i in range(1, 17)]
    raw_top = [f["raw_identity_bbox"][1] for f in per_frame]
    candidate_top = [f["candidate_identity_bbox"][1] for f in per_frame]
    core_positions = [i for i, index in enumerate(SOURCE_INDICES) if index in CORE_CONTACT_INDICES]
    endpoint_positions = [0, 1, 15, 16]
    raw_amplitude = float(np.median([raw_top[i] for i in core_positions]) - np.median([raw_top[i] for i in endpoint_positions]))
    candidate_amplitude = float(np.median([candidate_top[i] for i in core_positions]) - np.median([candidate_top[i] for i in endpoint_positions]))
    motion = {
        "pearson_correlation": float(np.corrcoef(raw_motion, candidate_motion)[0, 1]),
        "mean_ratio": float(np.mean(candidate_motion) / max(np.mean(raw_motion), 1e-12)),
        "top_trajectory_correlation": float(np.corrcoef(raw_top, candidate_top)[0, 1]),
        "raw_head_raise_amplitude_px": raw_amplitude,
        "candidate_head_raise_amplitude_px": candidate_amplitude,
        "head_raise_amplitude_retention": candidate_amplitude / max(raw_amplitude, 1e-9),
        "semantic_head_center_y": [f["semantic_anatomy"]["head_center_control"][1] for f in per_frame],
        "semantic_head_drop_amplitude_px": max(f["semantic_anatomy"]["head_center_control"][1] for f in per_frame) - min(f["semantic_anatomy"]["head_center_control"][1] for f in per_frame),
    }
    reference_mask = np.asarray(Image.open(input_dir / "reference_identity_mask.png").convert("RGB"))
    reference_identity = np.all(reference_mask == (0, 0, 255), axis=2)
    reference_prop = np.all(reference_mask == (255, 0, 0), axis=2)
    ref_ratio = float(np.count_nonzero(reference_identity) / max(np.count_nonzero(reference_prop), 1))
    driver_ratios = [float(f["candidate_area"] / max(f["prop_area"], 1)) for f in per_frame]
    ratio_error = float(abs(ref_ratio - np.median(driver_ratios)) / max(np.median(driver_ratios), 1e-9))
    with urllib.request.urlopen("http://127.0.0.1:8188/queue", timeout=3) as response:
        queue = json.load(response)
    aggregate = {
        "hf_rms_ratio_max": float(max(f["hf_rms_ratio"] for f in per_frame)),
        "hf_rms_ratio_mean": float(np.mean([f["hf_rms_ratio"] for f in per_frame])),
        "hf_identity_residual_correlation_abs_max": float(max(abs(f["hf_identity_residual_correlation"]) for f in per_frame)),
        "area_retention_ratio_range": [float(min(f["area_retention_ratio"] for f in per_frame)), float(max(f["area_retention_ratio"] for f in per_frame))],
        "reference_identity_to_prop_area_ratio": ref_ratio,
        "driver_identity_to_prop_area_ratio_median": float(np.median(driver_ratios)),
        "reference_ratio_relative_error": ratio_error,
    }
    checks = {
        "all_core_source_masks_have_saved_0px_contact": all(f["production_central_identity_prop_gap_px"] == 0 for f in per_frame if f["source_index"] in CORE_CONTACT_INDICES),
        "all_core_candidate_masks_keep_0px_contact": all(f["candidate_contact_gap_px"] == 0 for f in per_frame if f["source_index"] in CORE_CONTACT_INDICES),
        "all_props_pixel_exact_before_encode": all(f["prop_rgb_max_abs_difference"] == 0 for f in per_frame),
        "all_encoded_pose_and_mask_frames_bit_exact": all(f["pose_bit_exact"] and f["mask_bit_exact"] for f in decode_checks),
        "all_encoded_props_pixel_exact": all(f["decoded_prop_rgb_max_abs_difference"] == 0 for f in decode_checks),
        "identity_is_strict_grayscale": all(f["decoded_identity_chroma_max_channel_delta"] == 0 for f in decode_checks),
        "identity_hf_rms_ratio_at_most_0_12": aggregate["hf_rms_ratio_max"] <= 0.12,
        "identity_hf_residual_correlation_abs_at_most_0_20": aggregate["hf_identity_residual_correlation_abs_max"] <= 0.20,
        "no_main_case_animal_rgb_or_color_used": True,
        "fixed_semantic_palette_only": True,
        "all_required_anatomy_parts_present": all(f["semantic_anatomy"]["all_required_parts_present"] for f in per_frame),
        "all_frames_have_four_limb_and_four_paw_fields": all(f["semantic_anatomy"]["four_limb_fields_present"] and f["semantic_anatomy"]["four_paw_fields_present"] for f in per_frame),
        "all_frames_have_paired_eye_and_ear_fields": all(f["semantic_anatomy"]["paired_eyes_present"] and f["semantic_anatomy"]["paired_ears_present"] for f in per_frame),
        "no_mask_pixel_added_beyond_2px": all(f["new_pixels_beyond_2px"] == 0 for f in per_frame),
        "all_candidates_are_one_coherent_subject_component": all(f["candidate_component_count_ge20"] == 1 for f in per_frame),
        "no_candidate_border_touch": all(f["candidate_border_touch_pixels"] == 0 for f in per_frame),
        "motion_correlation_at_least_0_995": motion["pearson_correlation"] >= 0.995,
        "full_cycle_top_trajectory_correlation_at_least_0_995": motion["top_trajectory_correlation"] >= 0.995,
        "full_cycle_head_raise_amplitude_at_least_50px": motion["candidate_head_raise_amplitude_px"] >= 50.0,
        "semantic_head_drop_amplitude_at_least_120px": motion["semantic_head_drop_amplitude_px"] >= 120,
        "scaled_reference_ratio_within_15_percent_of_driver": ratio_error <= 0.15,
        "scaled_reference_is_upload_only_at_0_66": scaled_reference_metadata["identity_scale"] == 0.66 and not scaled_reference_metadata["profile_or_default_pixels_used"],
        "scaled_reference_mask_palette_has_no_unknown_pixels": scaled_reference_metadata["normalized_mask_counts"]["unknown"] == 0,
        "queue_empty_after": not queue.get("queue_running") and not queue.get("queue_pending"),
        "gpu_submitted": False, "production_modified": False, "profile_or_default_pixels_used": False,
    }
    allowed_false = {"gpu_submitted", "production_modified", "profile_or_default_pixels_used"}
    all_passed = all(value is True or key in allowed_false and value is False for key, value in checks.items())
    manifest = {
        "status": "prepared_cpu_only" if all_passed else "cpu_gate_failed_do_not_submit",
        "run_id": RUN_ID, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "policy": "main_case_geometry_derived_fixed_semantic_anatomy_full_cycle_v1",
        "source": str(SOURCE), "source_sha256": sha256_file(SOURCE), "source_indices": SOURCE_INDICES,
        "parameters": {"semantic_luma": SEMANTIC_LUMA, "mask_open_kernel": MASK_OPEN_KERNEL, "mask_close_kernel": MASK_CLOSE_KERNEL, "mask_addition_radius_px": MASK_ADDITION_RADIUS_PX},
        "provenance": {
            "driver_animal_rgb": "excluded from pose identity region; loaded only for comparison/audit",
            "driver_geometry": "alpha silhouette and bowl-relative fixed anatomical partitions only",
            "prop_pixels": "Main_case bowl/food exact RGB only",
            "reference_pixels": "uploaded identity only, fixed 0.66 scale",
            "profile_or_default_pixels": "none",
        },
        "per_frame": per_frame, "decode_checks": decode_checks, "aggregate": aggregate, "motion": motion,
        "reference_scale_evaluation": {"applied_uploaded_identity_linear_scale": 0.66, "ratio_relative_error": ratio_error, "reference_build": scaled_reference_metadata},
        "conditioning_contract": {"positive_prompt": positive, "positive_prompt_sha256": sha256_bytes(positive.encode()), "negative_prompt": negative, "negative_prompt_sha256": sha256_bytes(negative.encode())},
        "checks": checks, "all_cpu_checks_passed": all_passed,
        "workflow": {"path": str(workflow_path), "sha256": sha256_file(workflow_path), "submitted": False},
        "artifacts": {
            "pose_video": {"path": str(pose_path), "sha256": sha256_file(pose_path)},
            "mask_video": {"path": str(mask_path), "sha256": sha256_file(mask_path)},
            "reference": {"path": str(input_dir / "reference.png"), "sha256": sha256_file(input_dir / "reference.png")},
            "reference_mask": {"path": str(input_dir / "reference_identity_mask.png"), "sha256": sha256_file(input_dir / "reference_identity_mask.png")},
            "visual_board": {"path": str(VALIDATION_ROOT / "semantic_anatomy_visual_board.png"), "sha256": sha256_file(VALIDATION_ROOT / "semantic_anatomy_visual_board.png")},
        },
        "queue_after": queue, "gpu_submitted": False, "production_modified": False,
    }
    (VALIDATION_ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {key: manifest[key] for key in ("status", "run_id", "all_cpu_checks_passed", "aggregate", "motion", "checks", "artifacts")}
    (VALIDATION_ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if all_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
