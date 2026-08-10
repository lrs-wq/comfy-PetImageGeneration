#!/usr/bin/env python3
"""Build a CPU-only compact semantic-anatomy mask candidate; never submit GPU."""

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
R2_BUILDER = PROJECT / "tools/diagnostics/build_semantic_anatomy_cycle17_probe.py"
R2_RUN_ID = "codex-upload-only-eating-semantic-anatomy-cycle17-20260731-r2"
RUN_ID = "codex-upload-only-eating-semantic-anatomy-cycle17-20260731-r3-mask-v3"
EXPERIMENT_ROOT = Path("/data/comfy-pet-pipeline/experiments/scail2_tm_test")
INPUT_ROOT = Path("/data/comfy-pet-pipeline/comfyui/input/scail2_test")
VALIDATION_ROOT = Path(
    "/data/comfy-pet-pipeline/archive/validation/"
    "20260731-upload-only-identity-17f-r1/semantic_anatomy_cycle17_probe_r3_mask_v3"
)
SOURCE_INDICES = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
CORE_CONTACT_INDICES = {20, 25, 30, 35, 40, 45, 50, 55, 60}
STRUCTURAL_PARTS = [
    "head", "ear_left", "ear_right", "torso", "tail",
    "hind_leg_left", "hind_leg_right", "hind_paw_left", "hind_paw_right",
    "foreleg_left", "foreleg_right", "forepaw_left", "forepaw_right", "muzzle",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(PROJECT))
    spec.loader.exec_module(module)
    return module


def decode_rgb(path: Path) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    return frames


def bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask)
    if not len(xs):
        raise RuntimeError("empty_mask")
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


def perimeter(mask: np.ndarray) -> float:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return float(sum(cv2.arcLength(contour, True) for contour in contours))


def roughness(mask: np.ndarray) -> float:
    smooth = cv2.morphologyEx(
        mask.astype(np.uint8),
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
    )
    smooth = cv2.morphologyEx(
        smooth,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
    ).astype(bool)
    return max(0.0, perimeter(mask) / max(perimeter(smooth), 1.0) - 1.0)


def component_count(mask: np.ndarray, minimum_area: int = 20) -> int:
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return sum(int(stats[i, cv2.CC_STAT_AREA]) >= minimum_area for i in range(1, count))


def central_gap(identity: np.ndarray, prop: np.ndarray) -> int | None:
    left, top, right, _bottom = bbox(prop)
    center = int(round((left + right - 1) / 2.0))
    ys, _xs = np.where(identity[:top, max(0, center - 12):min(identity.shape[1], center + 13)])
    return None if not len(ys) else int(top - int(ys.max()) - 1)


def temporal_motion(masks: list[np.ndarray]) -> list[float]:
    return [float(np.mean(masks[i - 1] ^ masks[i])) for i in range(1, len(masks))]


def compact_mask(
    source: np.ndarray,
    prop: np.ndarray,
    regions: dict[str, np.ndarray],
    source_index: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    union = np.logical_or.reduce([regions[name] for name in STRUCTURAL_PARTS])
    # Shave source-boundary spikes first. All candidate support remains a strict
    # subset of the source silhouette, while broad semantic parts survive.
    eroded_source = cv2.erode(
        source.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=1,
    ).astype(bool)
    compact = union & eroded_source
    compact = cv2.morphologyEx(
        compact.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (27, 27)),
    ).astype(bool)
    compact = cv2.morphologyEx(
        compact.astype(np.uint8),
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
    ).astype(bool)
    compact &= source

    # The contact bridge is copied only from real source support, only during
    # the locked core eating phase; it cannot paint a false gap closure.
    prop_left, prop_top, prop_right, _ = bbox(prop)
    center = int(round((prop_left + prop_right - 1) / 2.0))
    yy, xx = np.indices(source.shape)
    contact_support = (
        source
        & (np.abs(xx - center) <= 15)
        & (yy >= max(0, prop_top - 14))
        & (yy < prop_top)
    )
    if source_index in CORE_CONTACT_INDICES:
        compact |= contact_support
        compact = cv2.morphologyEx(
            compact.astype(np.uint8),
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
        ).astype(bool)
        compact &= source

    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(compact.astype(np.uint8), 8)
    if count <= 1:
        raise RuntimeError(f"compact_mask_empty:{source_index}")
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    compact = labels == largest

    coverage = {
        name: float(np.count_nonzero(compact & regions[name]) / max(np.count_nonzero(regions[name]), 1))
        for name in STRUCTURAL_PARTS
    }
    retained_pixels = {
        name: int(np.count_nonzero(compact & regions[name]))
        for name in STRUCTURAL_PARTS
    }
    source_perimeter = perimeter(source)
    compact_perimeter = perimeter(compact)
    source_roughness = roughness(source)
    compact_roughness = roughness(compact)
    audit = {
        "source_index": source_index,
        "source_bbox": bbox(source),
        "compact_bbox": bbox(compact),
        "source_area": int(np.count_nonzero(source)),
        "compact_area": int(np.count_nonzero(compact)),
        "area_ratio": float(np.count_nonzero(compact) / max(np.count_nonzero(source), 1)),
        "source_perimeter": source_perimeter,
        "compact_perimeter": compact_perimeter,
        "perimeter_ratio": compact_perimeter / max(source_perimeter, 1.0),
        "source_roughness": source_roughness,
        "compact_roughness": compact_roughness,
        "roughness_ratio": compact_roughness / max(source_roughness, 1e-9),
        "new_pixels_outside_source": int(np.count_nonzero(compact & ~source)),
        "component_count_ge20": component_count(compact),
        "part_coverage": coverage,
        "part_retained_pixels": retained_pixels,
        "core_contact_gap_px": central_gap(compact, prop),
        "tail_bbox_source_field": bbox(regions["tail"]),
        "tail_bbox_compact_overlap": bbox(compact & regions["tail"]),
        "tail_right_edge_retention_px": bbox(regions["tail"])[2] - bbox(compact & regions["tail"])[2],
        "border_touch_pixels": int(np.count_nonzero(compact[0, :] | compact[-1, :] | compact[:, 0] | compact[:, -1])),
    }
    return compact, audit


def make_board(
    poses: list[Image.Image],
    source_masks: list[Image.Image],
    compact_masks: list[Image.Image],
    label_frames: list[Image.Image],
    output: Path,
) -> None:
    selected = [0, 4, 8, 12, 16]
    rows = [poses, source_masks, compact_masks, label_frames]
    labels = [
        "Frozen r2 semantic pose (pixel-identical)",
        "r2 source-silhouette SCAIL mask",
        "r3 compact semantic-union SCAIL mask",
        "Anatomical partition audit labels",
    ]
    cell, header, label_width = 256, 38, 300
    board = Image.new("RGB", (label_width + cell * 5, (cell + header) * 4), "white")
    draw = ImageDraw.Draw(board)
    for row, (frames, label) in enumerate(zip(rows, labels, strict=True)):
        y = row * (cell + header)
        draw.text((8, y + 12), label, fill="black")
        for col, idx in enumerate(selected):
            x = label_width + col * cell
            resample = Image.Resampling.NEAREST if row in {1, 2, 3} else Image.Resampling.LANCZOS
            board.paste(frames[idx].convert("RGB").resize((cell, cell), resample), (x, y + header))
            draw.text((x + 5, y + 12), f"out {idx} / src {SOURCE_INDICES[idx]}", fill="black")
    board.save(output, optimize=True)


def main() -> int:
    run_dir = EXPERIMENT_ROOT / RUN_ID
    input_dir = INPUT_ROOT / RUN_ID
    for path in (VALIDATION_ROOT, run_dir, input_dir):
        if path.exists():
            raise RuntimeError(f"Refusing to overwrite: {path}")
    VALIDATION_ROOT.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    input_dir.mkdir(parents=True)
    frame_dir = VALIDATION_ROOT / "selected_frames"
    frame_dir.mkdir()

    scail = load_module("scail_r3_mask_runner", RUNNER)
    r2 = load_module("semantic_r2_builder", R2_BUILDER)
    r2_input = INPUT_ROOT / R2_RUN_ID
    r2_experiment = EXPERIMENT_ROOT / R2_RUN_ID
    r2_pose_path = r2_input / "tm_pose.mp4"
    r2_mask_path = r2_input / "tm_identity_mask.mp4"
    r2_reference = r2_input / "reference.png"
    r2_reference_mask = r2_input / "reference_identity_mask.png"
    expected = {
        r2_pose_path: "499dcbfa935c9dae5cb331a7fdbc1e5a8af0cc4f7593f2d3b4ba30b67034a179",
        r2_mask_path: "85ed2847274fc9cd352fe853f3092337c86280237eac21fde49d76aebb022f31",
        r2_reference: "6ab8019c01dedba3ad4a24169cb96df1a6982924b55e79f8063686e627d9b957",
        r2_reference_mask: "29ad86423c9cf986ed6818cff338de5510564ae7a8a5751ac6bd8b70572dc9f2",
    }
    for path, value in expected.items():
        if sha256_file(path) != value:
            raise RuntimeError(f"Frozen r2 artifact SHA mismatch: {path}")

    pose_arrays = decode_rgb(r2_pose_path)
    source_mask_arrays = decode_rgb(r2_mask_path)
    if len(pose_arrays) != 17 or len(source_mask_arrays) != 17:
        raise RuntimeError("frozen_r2_frame_count_mismatch")
    poses = [Image.fromarray(frame, "RGB") for frame in pose_arrays]
    source_mask_images = [Image.fromarray(frame, "RGB") for frame in source_mask_arrays]
    compact_mask_images: list[Image.Image] = []
    label_frames: list[Image.Image] = []
    source_subjects: list[np.ndarray] = []
    compact_subjects: list[np.ndarray] = []
    props: list[np.ndarray] = []
    per_frame: list[dict[str, Any]] = []

    for output_index, (source_index, mask_rgb) in enumerate(zip(SOURCE_INDICES, source_mask_arrays, strict=True)):
        source = np.all(mask_rgb == (0, 0, 255), axis=2)
        prop = np.all(mask_rgb == (255, 0, 0), axis=2)
        unknown = ~(source | prop | np.all(mask_rgb == (255, 255, 255), axis=2))
        if np.any(unknown):
            raise RuntimeError(f"r2_mask_unknown_palette:{output_index}")
        regions, anatomy = r2.semantic_regions(source, prop, source_index)
        compact, audit = compact_mask(source, prop, regions, source_index)
        output_mask = np.full_like(mask_rgb, 255)
        output_mask[compact] = (0, 0, 255)
        output_mask[prop] = (255, 0, 0)
        compact_image = Image.fromarray(output_mask, "RGB")
        labels = r2.label_preview(source, regions, prop)
        compact_image.save(frame_dir / f"mask_{output_index:03d}_src_{source_index:03d}.png")
        poses[output_index].save(frame_dir / f"pose_{output_index:03d}_src_{source_index:03d}.png")
        compact_mask_images.append(compact_image)
        label_frames.append(labels)
        source_subjects.append(source)
        compact_subjects.append(compact)
        props.append(prop)
        per_frame.append({**audit, "semantic_anatomy": anatomy})

    pose_path = input_dir / "tm_pose.mp4"
    mask_path = input_dir / "tm_identity_mask.mp4"
    reference_path = input_dir / "reference.png"
    reference_mask_path = input_dir / "reference_identity_mask.png"
    shutil.copy2(r2_pose_path, pose_path)
    shutil.copy2(r2_reference, reference_path)
    shutil.copy2(r2_reference_mask, reference_mask_path)
    scail.encode_video(compact_mask_images, mask_path, 16, lossless_rgb=True)
    decoded_candidate = decode_rgb(mask_path)
    decode_checks = [
        {
            "frame": i,
            "bit_exact": bool(np.array_equal(decoded, np.asarray(expected_image))),
            "max_abs_difference": int(np.max(np.abs(decoded.astype(np.int16) - np.asarray(expected_image).astype(np.int16)), initial=0)),
            "unknown_pixels": int(np.count_nonzero(~(
                np.all(decoded == (0, 0, 255), axis=2)
                | np.all(decoded == (255, 0, 0), axis=2)
                | np.all(decoded == (255, 255, 255), axis=2)
            ))),
        }
        for i, (decoded, expected_image) in enumerate(zip(decoded_candidate, compact_mask_images, strict=True))
    ]

    workflow = json.loads((r2_experiment / "workflow_api.json").read_text(encoding="utf-8"))
    frozen_contract = {
        "positive_prompt": workflow["15"]["inputs"]["text"],
        "negative_prompt": workflow["16"]["inputs"]["text"],
        "seed": workflow["20"]["inputs"]["noise_seed"],
        "steps": workflow["19"]["inputs"]["steps"],
        "pose_strength": workflow["17"]["inputs"]["pose_strength"],
        "length": workflow["17"]["inputs"]["length"],
    }
    workflow["8"]["inputs"]["file"] = f"scail2_test/{RUN_ID}/tm_pose.mp4"
    workflow["10"]["inputs"]["file"] = f"scail2_test/{RUN_ID}/tm_identity_mask.mp4"
    workflow["12"]["inputs"]["image"] = f"scail2_test/{RUN_ID}/reference.png"
    workflow["13"]["inputs"]["image"] = f"scail2_test/{RUN_ID}/reference_identity_mask.png"
    workflow["23"]["inputs"]["filename_prefix"] = f"scail2_test/{RUN_ID}/subject_replacement"
    workflow["24"]["inputs"]["filename_prefix"] = f"scail2_test/{RUN_ID}/subject_replacement_preview"
    workflow_path = run_dir / "workflow_api.json"
    workflow_path.write_text(json.dumps(workflow, ensure_ascii=False, indent=2), encoding="utf-8")

    make_board(
        poses, source_mask_images, compact_mask_images, label_frames,
        VALIDATION_ROOT / "semantic_anatomy_r3_compact_mask_board.png",
    )
    poses[0].save(run_dir / "pose_first_frame.png")
    compact_mask_images[0].save(run_dir / "pose_mask_first_frame.png")
    shutil.copy2(reference_path, run_dir / "reference_normalized.png")
    shutil.copy2(reference_mask_path, run_dir / "reference_mask.png")

    source_motion = temporal_motion(source_subjects)
    compact_motion = temporal_motion(compact_subjects)
    perimeter_ratios = [f["perimeter_ratio"] for f in per_frame]
    roughness_ratios = [f["roughness_ratio"] for f in per_frame]
    area_ratios = [f["area_ratio"] for f in per_frame]
    all_coverages = [value for f in per_frame for value in f["part_coverage"].values()]
    motion_correlation = float(np.corrcoef(source_motion, compact_motion)[0, 1])
    area_trajectory_correlation = float(np.corrcoef(
        [f["source_area"] for f in per_frame],
        [f["compact_area"] for f in per_frame],
    )[0, 1])
    semantic_head_y = [f["semantic_anatomy"]["head_center_control"][1] for f in per_frame]
    with urllib.request.urlopen("http://127.0.0.1:8188/queue", timeout=3) as response:
        queue = json.load(response)
    aggregate = {
        "perimeter_ratio_mean": float(np.mean(perimeter_ratios)),
        "perimeter_ratio_max": float(np.max(perimeter_ratios)),
        "roughness_ratio_mean": float(np.mean(roughness_ratios)),
        "roughness_ratio_max": float(np.max(roughness_ratios)),
        "area_ratio_range": [float(np.min(area_ratios)), float(np.max(area_ratios))],
        "part_coverage_min": float(np.min(all_coverages)),
        "motion_correlation": motion_correlation,
        "area_trajectory_correlation": area_trajectory_correlation,
        "semantic_head_center_y": semantic_head_y,
        "semantic_head_drop_amplitude_px": max(semantic_head_y) - min(semantic_head_y),
        "tail_right_edge_retention_max_px": int(max(f["tail_right_edge_retention_px"] for f in per_frame)),
    }
    checks = {
        "frozen_pose_video_pixel_and_file_identical_to_r2": sha256_file(pose_path) == expected[r2_pose_path],
        "frozen_reference_identical_to_r2_and_replaceable": sha256_file(reference_path) == expected[r2_reference],
        "frozen_reference_mask_identical_to_r2": sha256_file(reference_mask_path) == expected[r2_reference_mask],
        "prompt_seed_steps_pose_strength_length_unchanged": frozen_contract == {
            "positive_prompt": workflow["15"]["inputs"]["text"],
            "negative_prompt": workflow["16"]["inputs"]["text"],
            "seed": workflow["20"]["inputs"]["noise_seed"],
            "steps": workflow["19"]["inputs"]["steps"],
            "pose_strength": workflow["17"]["inputs"]["pose_strength"],
            "length": workflow["17"]["inputs"]["length"],
        },
        "all_compact_masks_are_strict_source_subsets": all(f["new_pixels_outside_source"] == 0 for f in per_frame),
        "perimeter_mean_reduced_by_at_least_10_percent": aggregate["perimeter_ratio_mean"] <= 0.90,
        "perimeter_every_frame_reduced_by_at_least_7_5_percent": aggregate["perimeter_ratio_max"] <= 0.925,
        "roughness_mean_reduced_by_at_least_90_percent": aggregate["roughness_ratio_mean"] <= 0.10,
        "compact_area_retains_45_to_84_percent": aggregate["area_ratio_range"][0] >= 0.45 and aggregate["area_ratio_range"][1] <= 0.84,
        "all_anatomy_parts_retain_at_least_45_percent": aggregate["part_coverage_min"] >= 0.45,
        "all_anatomy_part_anchors_remain_material": all(
            f["part_retained_pixels"][name] >= minimum
            for f in per_frame
            for name, minimum in {
                "head": 1500, "ear_left": 80, "ear_right": 80,
                "torso": 6000, "tail": 600,
                "hind_leg_left": 400, "hind_leg_right": 400,
                "hind_paw_left": 120, "hind_paw_right": 120,
                "foreleg_left": 250, "foreleg_right": 250,
                "forepaw_left": 100, "forepaw_right": 100,
                "muzzle": 250,
            }.items()
        ),
        "all_masks_single_connected_subject": all(f["component_count_ge20"] == 1 for f in per_frame),
        "all_core_frames_keep_0px_mouth_bowl_contact": all(f["core_contact_gap_px"] == 0 for f in per_frame if f["source_index"] in CORE_CONTACT_INDICES),
        "tail_full_right_extent_retained_within_4px": aggregate["tail_right_edge_retention_max_px"] <= 4,
        "no_subject_border_touch": all(f["border_touch_pixels"] == 0 for f in per_frame),
        "compact_area_action_trajectory_correlation_at_least_0_90": area_trajectory_correlation >= 0.90,
        "semantic_head_drop_amplitude_unchanged_at_143px": aggregate["semantic_head_drop_amplitude_px"] == 143,
        "all_encoded_mask_frames_bit_exact": all(f["bit_exact"] for f in decode_checks),
        "encoded_mask_palette_has_no_unknown_pixels": all(f["unknown_pixels"] == 0 for f in decode_checks),
        "queue_empty_after": not queue.get("queue_running") and not queue.get("queue_pending"),
        "gpu_submitted": False,
        "production_modified": False,
    }
    allowed_false = {"gpu_submitted", "production_modified"}
    all_passed = all(value is True or key in allowed_false and value is False for key, value in checks.items())
    manifest = {
        "status": "prepared_cpu_only" if all_passed else "cpu_gate_failed_do_not_submit",
        "run_id": RUN_ID,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "policy": "semantic_anatomy_union_compact_source_subset_mask_v1",
        "source_run_id": R2_RUN_ID,
        "source_indices": SOURCE_INDICES,
        "frozen_conditioning": frozen_contract,
        "reference_policy": "current_r2_reference_frozen_but_workflow_paths_are_replaceable_before_any_future_submission",
        "gate_rationale": {
            "perimeter": "At least 10% mean and 7.5% every-frame perimeter reduction is material while retaining ears, four limb/paw anchors, contact, and the complete tail.",
            "roughness": "At least 90% mean high-frequency roughness reduction directly targets long-fur spikes and zipper edges.",
            "part_retention": "Fractional coverage is paired with per-part absolute pixel floors; further erosion would remove the small forepaw/ear anchors before materially improving the already-smooth contour.",
            "motion": "The pose video is file-identical, semantic head drop is identical, and compact-mask area trajectory correlation must exceed 0.90. Raw XOR correlation remains diagnostic because removing source edge flutter is intentional.",
        },
        "per_frame": per_frame,
        "decode_checks": decode_checks,
        "aggregate": aggregate,
        "checks": checks,
        "all_cpu_checks_passed": all_passed,
        "workflow": {"path": str(workflow_path), "sha256": sha256_file(workflow_path), "submitted": False},
        "artifacts": {
            "pose_video": {"path": str(pose_path), "sha256": sha256_file(pose_path)},
            "mask_video": {"path": str(mask_path), "sha256": sha256_file(mask_path)},
            "reference": {"path": str(reference_path), "sha256": sha256_file(reference_path)},
            "reference_mask": {"path": str(reference_mask_path), "sha256": sha256_file(reference_mask_path)},
            "visual_board": {"path": str(VALIDATION_ROOT / "semantic_anatomy_r3_compact_mask_board.png"), "sha256": sha256_file(VALIDATION_ROOT / "semantic_anatomy_r3_compact_mask_board.png")},
        },
        "queue_after": queue,
        "gpu_submitted": False,
        "production_modified": False,
    }
    (VALIDATION_ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {key: manifest[key] for key in ("status", "run_id", "all_cpu_checks_passed", "aggregate", "checks", "artifacts", "workflow")}
    (VALIDATION_ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if all_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
