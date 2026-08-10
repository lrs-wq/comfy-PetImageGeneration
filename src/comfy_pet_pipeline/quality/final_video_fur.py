#!/usr/bin/env python3
"""Fail-closed sampled fur audit with uploaded-identity action authority."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image

from .fur_surface import evaluate_fur_surface


POLICY = "final_video_validated_action_reference_fur_surface_v2"
VERSION = "3.2.0"
DEFAULT_SAMPLE_COUNT = 9
SUPPORTED_ACTIONS = {"eating", "grooming"}
IDENTITY_MASK_POLICY = "scail_blue_identity_red_prop_binary_background_v1"
IDENTITY_MASK_UNKNOWN_FRACTION_MAX = 0.02
PROFILE_FULL_SCOPE = "profile_absolute_and_relative_v1"
VALIDATED_ACTION_SURFACE_SCOPE = "validated_action_reference_dual_surface_v2"
UPLOADED_IDENTITY_COMPARISON_MODE = (
    "uploaded_identity_absolute_surface_contract_v1"
)
FINAL_VIDEO_LONG_LINE_RATIO_MARGIN = 0.15
FINAL_VIDEO_LONG_LINE_DELTA_MULTIPLIER = 3.0
FINAL_VIDEO_COHERENT_BRUSH_TEMPORAL_MIN_SAMPLES = 2
FINAL_VIDEO_LOW_SIGNAL_REFERENCE_MIN_SAMPLES = 3
FINAL_VIDEO_LOW_SIGNAL_REFERENCE_MIN_LONG_BLOCKING_FRACTION = 0.50
FINAL_VIDEO_LOW_SIGNAL_REFERENCE_TEMPORAL_SPREAD_MAX = 1.20
FINAL_VIDEO_SPARSE_COARSE_LOW_SIGNAL_MIN_SAMPLES = 5
FINAL_VIDEO_SPARSE_COARSE_LOW_SIGNAL_MIN_BLOCKING_SAMPLES = 2
FINAL_VIDEO_SPARSE_COARSE_LOW_SIGNAL_MAX_BLOCKING_FRACTION = 0.50
FINAL_VIDEO_DIRECT_QWEN_SPARSE_COARSE_MIN_SAMPLES = 5
FINAL_VIDEO_DIRECT_QWEN_SPARSE_COARSE_MIN_BLOCKING_SAMPLES = 2
FINAL_VIDEO_DIRECT_QWEN_SPARSE_COARSE_MAX_BLOCKING_FRACTION = 0.50
FINAL_VIDEO_DIRECT_QWEN_SPARSE_COARSE_TEMPORAL_SPREAD_MAX = 1.20
FINAL_VIDEO_EATING_BOUNDARY_ANCHOR_MIN_SAMPLES = 5
FINAL_VIDEO_EATING_BOUNDARY_ANCHOR_MIN_BLOCKING_SAMPLES = 2
FINAL_VIDEO_EATING_BOUNDARY_ANCHOR_SPREAD_MAX = 1.20
FINAL_VIDEO_EATING_ISOLATED_LONG_FUR_MIN_SAMPLES = 5
FINAL_VIDEO_EATING_PROP_ALIGNMENT_MIN = 0.75
FINAL_VIDEO_EATING_RADIAL_BOUNDED_REFERENCE_RATIO_MARGIN = 0.10
FINAL_VIDEO_EATING_RADIAL_BOUNDED_REFERENCE_LOAD_DELTA_MULTIPLIER = 2.0
FINAL_VIDEO_EATING_RADIAL_BOUNDARY_MIN_SAMPLES = 5
FINAL_VIDEO_EATING_RADIAL_BOUNDARY_LOAD_ENERGY_SPREAD_MAX = 1.10
FINAL_VIDEO_EATING_RADIAL_BOUNDARY_COMPONENT_SPREAD_MAX = 1.20
GROOMING_MASKED_PET3D_COMPOSITE_POLICY = (
    "masked_pet3d_surface_verified_pose_edge_and_low_texture_paw_rescue_v3"
)
GROOMING_MASKED_PET3D_COMPOSITE_REQUIRED_CHECKS = {
    "global_pose_gate_contract_valid",
    "raw_issue_contract_valid",
    "edge_density_within_recovery_band",
    "chamfer_geometry_passed",
    "continuous_foreleg_passed",
    "paw_muzzle_contact_passed",
    "raised_planted_paw_color_histogram_passed",
    "raised_planted_paw_texture_contract_passed",
    "independent_fur_surface_gate_passed",
    "masked_pet3d_repair_contract_valid",
}
POSE_RELATIVE_OBSERVATIONS = {
    "upper_body_area_expanded",
    "torso_area_expanded",
    "upper_body_width_expanded",
    "torso_width_expanded",
}
SURFACE_RELATIVE_ISSUES = {
    "contour_fringe_expanded_vs_profile",
    "coarse_strokes_expanded_vs_profile",
    "coherent_brush_texture_expanded_vs_profile",
    "radial_brush_texture_expanded_vs_profile",
    "long_fur_energy_expanded_vs_profile",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_indices(frame_count: int, sample_count: int) -> list[int]:
    if frame_count <= 0:
        raise ValueError("video_has_no_frames")
    if sample_count <= 0:
        raise ValueError("sample_count_must_be_positive")
    count = min(frame_count, sample_count)
    return sorted(
        {int(value) for value in np.rint(np.linspace(0, frame_count - 1, count))}
    )


def resolve_profile_baseline(
    profile_path: str | Path,
    action: str,
) -> tuple[Path, str, dict[str, Any]]:
    if action not in SUPPORTED_ACTIONS:
        raise ValueError(f"unsupported_action:{action}")
    resolved_profile = Path(profile_path).resolve()
    profile = json.loads(resolved_profile.read_text(encoding="utf-8"))
    baseline = profile.get("validated_action_baselines", {}).get(action)
    if not isinstance(baseline, dict):
        raise ValueError(f"validated_action_baseline_missing:{action}")
    reference = str(baseline.get("reference") or "").strip()
    expected_sha256 = str(baseline.get("reference_sha256") or "").strip().lower()
    if not reference:
        raise ValueError(f"validated_action_baseline_reference_missing:{action}")
    if len(expected_sha256) != 64:
        raise ValueError(f"validated_action_baseline_sha256_missing:{action}")
    reference_path = Path(reference)
    if not reference_path.is_absolute():
        reference_path = (resolved_profile.parent / reference_path).resolve()
    if not reference_path.is_file():
        raise ValueError(f"validated_action_baseline_not_found:{reference_path}")
    actual_sha256 = sha256_file(reference_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "validated_action_baseline_sha256_mismatch:"
            f"expected={expected_sha256}:actual={actual_sha256}"
        )
    provenance = {
        "profile_path": str(resolved_profile),
        "profile_sha256": sha256_file(resolved_profile),
        "profile_id": profile.get("id"),
        "action": action,
        "run_id": baseline.get("run_id"),
        "reference": str(reference_path),
        "reference_sha256": actual_sha256,
        "baseline_policy": baseline.get("fur_surface_policy"),
        "baseline_version": baseline.get("fur_surface_version"),
    }
    return reference_path, actual_sha256, provenance


def probe_video(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    capture = cv2.VideoCapture(str(resolved))
    if not capture.isOpened():
        raise ValueError(f"video_open_failed:{resolved}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    if frame_count <= 0 or width <= 0 or height <= 0:
        raise ValueError("video_metadata_invalid")
    return {
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "fps": round(fps, 6),
        "duration_seconds": round(frame_count / fps, 6) if fps > 0 else None,
    }


def extract_frame(video_path: str | Path, frame_index: int, target: Path) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"video_open_failed:{video_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise ValueError(f"video_frame_read_failed:{frame_index}")
    if not cv2.imwrite(str(target), frame):
        raise ValueError(f"video_frame_write_failed:{frame_index}")


def validate_identity_mask_frame(
    path: str | Path,
    *,
    action: str,
    background: str = "white",
) -> dict[str, Any]:
    """Validate one decoded SCAIL blue/red/white identity-guidance frame."""

    if action not in SUPPORTED_ACTIONS:
        raise ValueError(f"unsupported_action:{action}")
    with Image.open(path) as loaded:
        pixels = np.asarray(loaded.convert("RGB"), dtype=np.uint8)
    blue = (
        (pixels[..., 2] >= 245)
        & (pixels[..., 0] <= 10)
        & (pixels[..., 1] <= 10)
    )
    red = (
        (pixels[..., 0] >= 245)
        & (pixels[..., 1] <= 10)
        & (pixels[..., 2] <= 10)
    )
    if background not in {"white", "black"}:
        raise ValueError(f"unsupported_identity_mask_background:{background}")
    background_pixels = (
        np.all(pixels >= 245, axis=2)
        if background == "white"
        else np.all(pixels <= 10, axis=2)
    )
    known = blue | red | background_pixels
    total = max(1, int(known.size))
    blue_fraction = float(np.count_nonzero(blue) / total)
    red_fraction = float(np.count_nonzero(red) / total)
    background_fraction = float(np.count_nonzero(background_pixels) / total)
    unknown_fraction = float(np.count_nonzero(~known) / total)
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(
        blue.astype(np.uint8),
        connectivity=8,
    )
    blue_pixels = max(1, int(np.count_nonzero(blue)))
    largest_blue_fraction = (
        float(np.max(stats[1:, cv2.CC_STAT_AREA]) / blue_pixels)
        if component_count > 1
        else 0.0
    )
    issues: list[str] = []
    if unknown_fraction > IDENTITY_MASK_UNKNOWN_FRACTION_MAX:
        issues.append("identity_mask_unknown_fraction_exceeded")
    if not 0.01 <= blue_fraction <= 0.90:
        issues.append("identity_mask_subject_fraction_invalid")
    if background_fraction < 0.05:
        issues.append("identity_mask_background_fraction_invalid")
    if largest_blue_fraction < 0.60:
        issues.append("identity_mask_primary_subject_component_unreliable")
    if action == "eating" and not 0.005 <= red_fraction <= 0.30:
        issues.append("identity_mask_eating_prop_fraction_invalid")
    if action == "grooming" and red_fraction > 0.02:
        issues.append("identity_mask_unexpected_grooming_prop")
    return {
        "reliable": not issues,
        "issues": issues,
        "policy": IDENTITY_MASK_POLICY,
        "background": background,
        "width": int(pixels.shape[1]),
        "height": int(pixels.shape[0]),
        "blue_identity_fraction": round(blue_fraction, 8),
        "red_prop_fraction": round(red_fraction, 8),
        "background_fraction": round(background_fraction, 8),
        "unknown_fraction": round(unknown_fraction, 8),
        "largest_blue_component_fraction": round(largest_blue_fraction, 8),
    }


def clear_red_prop_from_alpha(
    source_path: str | Path,
    identity_mask_path: str | Path,
    target_path: str | Path,
    *,
    action: str,
    mask_background: str = "white",
) -> dict[str, Any]:
    """Clear only exact red SCAIL prop pixels while preserving RGB and contour data."""

    mask_quality = validate_identity_mask_frame(
        identity_mask_path,
        action=action,
        background=mask_background,
    )
    if mask_quality["reliable"] is not True:
        raise RuntimeError(
            "identity_mask_unreliable:" + ",".join(mask_quality["issues"])
        )
    with Image.open(source_path) as loaded_source:
        rgba = np.asarray(loaded_source.convert("RGBA"), dtype=np.uint8).copy()
    with Image.open(identity_mask_path) as loaded_mask:
        mask_rgb = np.asarray(loaded_mask.convert("RGB"), dtype=np.uint8)
    if rgba.shape[:2] != mask_rgb.shape[:2]:
        raise ValueError(
            "identity_mask_image_size_mismatch:"
            f"source={rgba.shape[1]}x{rgba.shape[0]}:"
            f"mask={mask_rgb.shape[1]}x{mask_rgb.shape[0]}"
        )
    red = (
        (mask_rgb[..., 0] >= 245)
        & (mask_rgb[..., 1] <= 10)
        & (mask_rgb[..., 2] <= 10)
    )
    source_alpha = rgba[..., 3].copy()
    red_pixels = int(np.count_nonzero(red))
    red_foreground_pixels = int(np.count_nonzero(red & (source_alpha >= 96)))
    red_foreground_overlap = (
        red_foreground_pixels / max(1, red_pixels) if red_pixels else 1.0
    )
    if (
        action == "eating"
        and red_foreground_overlap < FINAL_VIDEO_EATING_PROP_ALIGNMENT_MIN
    ):
        raise RuntimeError(
            "identity_mask_prop_alignment_unreliable:"
            f"overlap={red_foreground_overlap:.6f}"
        )
    rgb_sha256 = hashlib.sha256(rgba[..., :3].tobytes()).hexdigest()
    rgba[..., 3][red] = 0
    if np.any(rgba[..., 3] > source_alpha):
        raise RuntimeError("prop_exclusion_increased_source_alpha")
    if not np.any(rgba[..., 3] >= 96):
        raise RuntimeError("prop_exclusion_removed_entire_subject")
    Image.fromarray(rgba, mode="RGBA").save(target_path)
    with Image.open(target_path) as saved:
        saved_rgba = np.asarray(saved.convert("RGBA"), dtype=np.uint8)
    saved_rgb_sha256 = hashlib.sha256(saved_rgba[..., :3].tobytes()).hexdigest()
    if saved_rgb_sha256 != rgb_sha256:
        raise RuntimeError("prop_exclusion_changed_rgb_pixels")
    return {
        "policy": "exact_scail_red_prop_alpha_exclusion_v1",
        "mask": mask_quality,
        "source_path": str(Path(source_path).resolve()),
        "source_sha256": sha256_file(source_path),
        "target_path": str(Path(target_path).resolve()),
        "target_sha256": sha256_file(target_path),
        "rgb_sha256": rgb_sha256,
        "rgb_byte_preserved": True,
        "alpha_only_decreased": True,
        "red_pixels": red_pixels,
        "red_foreground_pixels": red_foreground_pixels,
        "red_foreground_overlap": round(red_foreground_overlap, 8),
        "minimum_red_foreground_overlap": (
            FINAL_VIDEO_EATING_PROP_ALIGNMENT_MIN
            if action == "eating"
            else None
        ),
        "morphology_applied": False,
    }


def _validate_rembg_alpha(image: Image.Image) -> dict[str, Any]:
    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    alpha = rgba[:, :, 3]
    foreground = alpha >= 96
    foreground_fraction = float(np.mean(foreground))
    border = np.concatenate(
        (alpha[0, :], alpha[-1, :], alpha[:, 0], alpha[:, -1])
    )
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        foreground.astype(np.uint8), connectivity=8
    )
    largest_area = (
        int(np.max(stats[1:, cv2.CC_STAT_AREA])) if count > 1 else 0
    )
    largest_fraction = largest_area / max(1, int(np.count_nonzero(foreground)))
    issues: list[str] = []
    if not np.any(alpha < 250):
        issues.append("rembg_alpha_is_opaque")
    if not 0.01 <= foreground_fraction <= 0.92:
        issues.append("rembg_foreground_fraction_invalid")
    if float(np.mean(border >= 96)) > 0.10:
        issues.append("rembg_foreground_touches_too_much_border")
    if largest_fraction < 0.70:
        issues.append("rembg_primary_component_unreliable")
    return {
        "reliable": not issues,
        "issues": issues,
        "foreground_fraction": round(foreground_fraction, 8),
        "largest_component_fraction": round(largest_fraction, 8),
        "opaque_border_fraction": round(float(np.mean(border >= 96)), 8),
    }


@lru_cache(maxsize=2)
def _rembg_session(model: str) -> Any:
    try:
        from rembg import new_session
    except Exception as exc:  # noqa: BLE001 - surfaced as a fail-closed audit error
        raise RuntimeError(f"rembg_unavailable:{exc}") from exc
    return new_session(model)


def remove_background(
    source_path: Path,
    target_path: Path,
    *,
    model: str,
) -> dict[str, Any]:
    try:
        from rembg import remove
    except Exception as exc:  # noqa: BLE001 - surfaced as a fail-closed audit error
        raise RuntimeError(f"rembg_unavailable:{exc}") from exc
    with Image.open(source_path) as source:
        result = remove(source.convert("RGB"), session=_rembg_session(model))
    if isinstance(result, Image.Image):
        image = result.convert("RGBA")
    else:
        image = Image.fromarray(np.asarray(result)).convert("RGBA")
    reliability = _validate_rembg_alpha(image)
    if not reliability["reliable"]:
        raise RuntimeError("rembg_unreliable:" + ",".join(reliability["issues"]))
    image.save(target_path)
    return reliability


def _severity(result: dict[str, Any]) -> tuple[int, int, float]:
    acceptable = result.get("acceptable")
    verdict_rank = 2 if acceptable is False else 1 if acceptable is None else 0
    issue_count = len(result.get("issues") or [])
    texture = result.get("candidate", {}).get("metrics", {}).get("texture", {})
    long_energy = float(texture.get("coherent_long_line_energy") or 0.0)
    return verdict_rank, issue_count, long_energy


def audit_final_video(
    video_path: str | Path,
    *,
    action: str,
    baseline_path: str | Path,
    baseline_sha256: str,
    baseline_provenance: dict[str, Any],
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    rembg_model: str = "u2net",
    evaluator: Callable[..., dict[str, Any]] = evaluate_fur_surface,
    frame_extractor: Callable[[str | Path, int, Path], None] = extract_frame,
    video_probe: Callable[[str | Path], dict[str, Any]] = probe_video,
    background_remover: Callable[..., dict[str, Any]] = remove_background,
) -> dict[str, Any]:
    resolved_video = Path(video_path).resolve()
    resolved_baseline = Path(baseline_path).resolve()
    if action not in SUPPORTED_ACTIONS:
        raise ValueError(f"unsupported_action:{action}")
    if not resolved_video.is_file():
        raise ValueError(f"video_not_found:{resolved_video}")
    if not resolved_baseline.is_file():
        raise ValueError(f"baseline_not_found:{resolved_baseline}")
    actual_baseline_sha = sha256_file(resolved_baseline)
    if actual_baseline_sha != baseline_sha256.lower():
        raise ValueError(
            "baseline_sha256_mismatch:"
            f"expected={baseline_sha256}:actual={actual_baseline_sha}"
        )
    baseline_check = evaluator(
        resolved_baseline,
        resolved_baseline,
        baseline_is_validated=True,
    )
    if baseline_check.get("acceptable") is not True:
        raise ValueError("validated_baseline_fails_current_fur_surface_gate")

    metadata = video_probe(resolved_video)
    indices = sample_indices(int(metadata["frame_count"]), sample_count)
    frame_results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="final-video-fur-audit-") as temp_name:
        temp_root = Path(temp_name)
        for frame_index in indices:
            raw_path = temp_root / f"frame_{frame_index:05d}_raw.png"
            alpha_path = temp_root / f"frame_{frame_index:05d}_rembg.png"
            try:
                frame_extractor(resolved_video, frame_index, raw_path)
                evaluation = evaluator(
                    raw_path,
                    resolved_baseline,
                    baseline_is_validated=True,
                )
                segmentation = {
                    "method": evaluation["candidate"]["input"][
                        "foreground_method"
                    ],
                    "supported_contract": evaluation["candidate"]["input"][
                        "supported_contract"
                    ],
                }
                if not segmentation["supported_contract"]:
                    reliability = background_remover(
                        raw_path,
                        alpha_path,
                        model=rembg_model,
                    )
                    if (
                        not isinstance(reliability, dict)
                        or reliability.get("reliable") is not True
                    ):
                        issues = (
                            reliability.get("issues", [])
                            if isinstance(reliability, dict)
                            else []
                        )
                        detail = ",".join(str(issue) for issue in issues)
                        raise RuntimeError(
                            "background_segmentation_unreliable"
                            + (f":{detail}" if detail else "")
                        )
                    evaluation = evaluator(
                        alpha_path,
                        resolved_baseline,
                        baseline_is_validated=True,
                    )
                    segmentation = {
                        "method": f"rembg_{rembg_model}",
                        "supported_contract": evaluation["candidate"]["input"][
                            "supported_contract"
                        ],
                        "reliability": reliability,
                    }
                if (
                    not segmentation["supported_contract"]
                    or evaluation.get("acceptable") is not True
                ):
                    passed = False
                else:
                    passed = True
                frame_results.append(
                    {
                        "frame_index": frame_index,
                        "passed": passed,
                        "segmentation": segmentation,
                        "evaluation": evaluation,
                        "error": None,
                    }
                )
            except Exception as exc:  # noqa: BLE001 - each frame must fail closed
                frame_results.append(
                    {
                        "frame_index": frame_index,
                        "passed": False,
                        "segmentation": None,
                        "evaluation": None,
                        "error": f"{type(exc).__name__}:{exc}",
                    }
                )

    failed = [item for item in frame_results if not item["passed"]]
    evaluable = [item for item in frame_results if item["evaluation"] is not None]
    worst = (
        max(evaluable, key=lambda item: _severity(item["evaluation"]))
        if evaluable
        else frame_results[0]
    )
    return {
        "passed": not failed,
        "verdict": "accepted" if not failed else "rejected_fail_closed",
        "action": action,
        "video": {
            "path": str(resolved_video),
            "sha256": sha256_file(resolved_video),
            **metadata,
        },
        "baseline": {
            "path": str(resolved_baseline),
            "sha256": actual_baseline_sha,
            "provenance": baseline_provenance,
            "self_check": baseline_check,
        },
        "sampling": {
            "requested_count": sample_count,
            "actual_count": len(indices),
            "frame_indices": indices,
        },
        "frames": frame_results,
        "failed_frame_indices": [item["frame_index"] for item in failed],
        "worst_frame_index": worst["frame_index"],
        "policy": {
            "name": POLICY,
            "version": VERSION,
            "actions": sorted(SUPPORTED_ACTIONS),
            "validated_baseline_required": True,
            "unsupported_segmentation_fails_closed": True,
            "all_sampled_frames_must_pass": True,
            "rembg_model": rembg_model,
        },
    }


_audit_final_video_profile_full = audit_final_video


def _require_evaluation_list(
    evaluation: dict[str, Any],
    key: str,
) -> list[str]:
    value = evaluation.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise RuntimeError(f"fur_surface_evaluation_{key}_invalid")
    return value


def _action_reference_issue_name(issue: str) -> str:
    return issue.replace(
        "_vs_profile",
        "_vs_validated_action_reference",
    )


def _evaluation_candidate_supported(evaluation: dict[str, Any]) -> bool:
    candidate = evaluation.get("candidate")
    candidate_input = candidate.get("input") if isinstance(candidate, dict) else None
    return bool(
        isinstance(candidate, dict)
        and candidate.get("valid") is True
        and isinstance(candidate_input, dict)
        and candidate_input.get("supported_contract") is True
    )


def _evaluation_baseline_supported(evaluation: dict[str, Any]) -> bool:
    baseline = evaluation.get("baseline")
    baseline_input = baseline.get("input") if isinstance(baseline, dict) else None
    return bool(
        isinstance(baseline, dict)
        and baseline.get("valid") is True
        and isinstance(baseline_input, dict)
        and baseline_input.get("supported_contract") is True
    )


def _validate_uploaded_reference_provenance(
    provenance: dict[str, Any] | None,
    *,
    action: str,
    reference_sha256: str,
) -> dict[str, Any]:
    """Verify that the exact Qwen reference passed the upload-only gate."""

    if not isinstance(provenance, dict):
        raise ValueError("validated_action_reference_provenance_missing")
    if provenance.get("reference_kind") != "qwen_generated_upload":
        raise ValueError("validated_action_reference_is_not_qwen_upload")
    if provenance.get("pre_scail_revalidated") is not True:
        raise ValueError("validated_action_reference_not_revalidated_before_scail")
    selected_gate = provenance.get("gate")
    if not isinstance(selected_gate, dict):
        raise ValueError("validated_action_reference_gate_missing")
    gate_contract = selected_gate.get("gate")
    gate_policy = selected_gate.get("policy")
    if not isinstance(gate_contract, dict) or not isinstance(gate_policy, dict):
        raise ValueError("validated_action_reference_gate_contract_missing")
    if selected_gate.get("acceptable") is not True:
        raise ValueError("validated_action_reference_gate_not_accepted")
    if gate_policy.get("uploaded_identity_absolute_contract_enabled") is not True:
        raise ValueError("validated_action_reference_upload_contract_not_enabled")
    if gate_policy.get("validated_baseline_required_for_acceptance") is not False:
        raise ValueError("validated_action_reference_still_depends_on_profile")
    if gate_contract.get("action") != action:
        raise ValueError("validated_action_reference_gate_action_mismatch")
    if (
        gate_contract.get("candidate_comparison_mode")
        != UPLOADED_IDENTITY_COMPARISON_MODE
    ):
        raise ValueError("validated_action_reference_comparison_mode_mismatch")
    if (
        gate_contract.get("profile_pixels_used_for_candidate_appearance_validation")
        is not False
    ):
        raise ValueError("validated_action_reference_profile_pixels_were_used")
    hash_chain = {
        "configured_reference": str(
            provenance.get("configured_reference_sha256") or ""
        ).lower(),
        "selected_candidate": str(
            provenance.get("selected_candidate_sha256") or ""
        ).lower(),
        "gate_candidate": str(
            provenance.get("gate_candidate_sha256") or ""
        ).lower(),
        "selected_gate_candidate": str(
            selected_gate.get("selected_candidate_sha256") or ""
        ).lower(),
        "gate_contract_candidate": str(
            gate_contract.get("candidate_sha256") or ""
        ).lower(),
    }
    if any(value != reference_sha256 for value in hash_chain.values()):
        raise ValueError("validated_action_reference_provenance_sha256_mismatch")
    composite_recovery = provenance.get(
        "grooming_pose_gate_composite_recovery"
    )
    composite_validation: dict[str, Any] = {
        "present": False,
        "validated": False,
    }
    if composite_recovery is not None:
        if action != "grooming" or not isinstance(composite_recovery, dict):
            raise ValueError(
                "validated_action_reference_grooming_composite_scope_invalid"
            )
        checks = composite_recovery.get("checks")
        mask_sha256 = str(
            composite_recovery.get("repair_noise_mask_sha256") or ""
        ).lower()
        try:
            required_ratio = float(
                composite_recovery["required_edge_density_ratio"]
            )
            edge_density = float(composite_recovery["edge_density"])
            edge_threshold = float(
                composite_recovery["edge_density_threshold"]
            )
            edge_floor = float(composite_recovery["edge_density_floor"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "validated_action_reference_grooming_composite_metrics_invalid"
            ) from exc
        composite_values = (
            required_ratio,
            edge_density,
            edge_threshold,
            edge_floor,
        )
        if (
            composite_recovery.get("policy")
            != GROOMING_MASKED_PET3D_COMPOSITE_POLICY
            or composite_recovery.get("evaluated") is not True
            or composite_recovery.get("accepted") is not True
            or composite_recovery.get("fail_closed") is not True
            or composite_recovery.get("global_pose_threshold_changed")
            is not False
            or not isinstance(checks, dict)
            or not GROOMING_MASKED_PET3D_COMPOSITE_REQUIRED_CHECKS.issubset(
                checks
            )
            or any(value is not True for value in checks.values())
            or not re.fullmatch(r"[0-9a-f]{64}", mask_sha256)
            or not all(np.isfinite(value) for value in composite_values)
            or required_ratio <= 0.0
            or required_ratio >= 1.0
            or edge_threshold <= 0.0
            or edge_floor <= 0.0
            or edge_density < edge_floor
            or not np.isclose(
                edge_floor,
                edge_threshold * required_ratio,
                rtol=0.0,
                atol=1e-9,
            )
        ):
            raise ValueError(
                "validated_action_reference_grooming_composite_contract_invalid"
            )
        composite_validation = {
            "present": True,
            "validated": True,
            "policy": GROOMING_MASKED_PET3D_COMPOSITE_POLICY,
            "repair_noise_mask_sha256": mask_sha256,
            "required_edge_density_ratio": round(required_ratio, 8),
            "edge_density": round(edge_density, 8),
            "edge_density_floor": round(edge_floor, 8),
            "required_checks": sorted(
                GROOMING_MASKED_PET3D_COMPOSITE_REQUIRED_CHECKS
            ),
        }
    return {
        "validated": True,
        "authority": "uploaded_qwen_action_reference",
        "reference_sha256": reference_sha256,
        "hash_chain_verified": True,
        "gate_verdict": selected_gate.get("verdict"),
        "gate_policy": gate_policy.get("name"),
        "gate_version": gate_policy.get("gate_version"),
        "threshold_version": gate_policy.get("threshold_version"),
        "candidate_comparison_mode": gate_contract.get(
            "candidate_comparison_mode"
        ),
        "profile_pixels_used_for_appearance_validation": False,
        "grooming_masked_pet3d_composite_recovery": composite_validation,
    }


def _radial_relative_evidence(
    evaluation: dict[str, Any],
    *,
    relative_issues: list[str],
    absolute_issues: list[str],
) -> dict[str, Any]:
    """Require threshold-backed radial load growth, not decode-level movement."""

    issue = "radial_brush_texture_expanded_vs_profile"
    absolute_radial_issue = "radial_brush_texture" in absolute_issues
    if issue not in relative_issues:
        return {
            "relative_issue_present": False,
            "absolute_radial_issue": absolute_radial_issue,
            "load_expanded": False,
            "normalized_energy_backstop_triggered": False,
            "blocking": absolute_radial_issue,
            "blocking_issue": (
                "radial_brush_texture" if absolute_radial_issue else None
            ),
            "radial_fraction_without_load_expansion_observation": False,
        }
    candidate = evaluation.get("candidate")
    baseline = evaluation.get("baseline")
    thresholds = evaluation.get("thresholds")
    if not all(isinstance(item, dict) for item in (candidate, baseline, thresholds)):
        raise RuntimeError("radial_relative_evidence_schema_invalid")
    candidate_texture = candidate.get("metrics", {}).get("texture")
    baseline_texture = baseline.get("metrics", {}).get("texture")
    if not isinstance(candidate_texture, dict) or not isinstance(
        baseline_texture,
        dict,
    ):
        raise RuntimeError("radial_relative_texture_metrics_missing")
    try:
        candidate_fraction = float(candidate_texture["fine_radial_brush_fraction"])
        baseline_fraction = float(baseline_texture["fine_radial_brush_fraction"])
        candidate_gradient_p90 = float(candidate_texture["fine_gradient_p90"])
        baseline_gradient_p90 = float(baseline_texture["fine_gradient_p90"])
        candidate_energy = float(candidate_texture["fine_radial_brush_energy"])
        baseline_energy = float(baseline_texture["fine_radial_brush_energy"])
        energy_backstop_ratio_min = float(thresholds["radial_brush_ratio_max"])
        energy_backstop_delta_min = float(thresholds["texture_delta_min"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("radial_relative_evidence_metrics_invalid") from exc
    values = (
        candidate_fraction,
        baseline_fraction,
        candidate_gradient_p90,
        baseline_gradient_p90,
        candidate_energy,
        baseline_energy,
        energy_backstop_ratio_min,
        energy_backstop_delta_min,
    )
    if not all(np.isfinite(value) for value in values):
        raise RuntimeError("radial_relative_evidence_metrics_non_finite")
    if (
        any(value < 0.0 for value in values)
        or baseline_fraction <= 0.0
        or baseline_gradient_p90 <= 0.0
        or baseline_energy <= 0.0
        or energy_backstop_ratio_min <= 0.0
    ):
        raise RuntimeError("radial_relative_evidence_metrics_out_of_range")
    candidate_load = candidate_fraction * candidate_gradient_p90
    baseline_load = baseline_fraction * baseline_gradient_p90
    if not np.isfinite(candidate_load) or not np.isfinite(baseline_load):
        raise RuntimeError("radial_relative_load_non_finite")
    if baseline_load <= 0.0:
        raise RuntimeError("radial_relative_baseline_load_invalid")
    load_ratio = candidate_load / baseline_load
    load_delta = candidate_load - baseline_load
    raw_load_grew = bool(candidate_load > baseline_load)
    load_expanded = bool(
        load_ratio > energy_backstop_ratio_min
        and load_delta > energy_backstop_delta_min
    )
    energy_ratio = candidate_energy / baseline_energy
    energy_delta = candidate_energy - baseline_energy
    if not all(
        np.isfinite(value)
        for value in (load_ratio, load_delta, energy_ratio, energy_delta)
    ):
        raise RuntimeError("radial_relative_derived_metrics_non_finite")
    normalized_energy_backstop_triggered = bool(
        energy_ratio > energy_backstop_ratio_min
        and energy_delta > energy_backstop_delta_min
    )
    blocking = bool(
        absolute_radial_issue
        or load_expanded
        or normalized_energy_backstop_triggered
    )
    return {
        "relative_issue_present": True,
        "candidate_fraction": round(candidate_fraction, 8),
        "baseline_fraction": round(baseline_fraction, 8),
        "candidate_gradient_p90": round(candidate_gradient_p90, 8),
        "baseline_gradient_p90": round(baseline_gradient_p90, 8),
        "candidate_load_proxy": round(candidate_load, 8),
        "baseline_load_proxy": round(baseline_load, 8),
        "load_ratio": round(load_ratio, 8),
        "load_delta": round(load_delta, 8),
        "raw_load_grew": raw_load_grew,
        "load_expanded": load_expanded,
        "load_expansion_ratio_min": round(energy_backstop_ratio_min, 8),
        "load_expansion_delta_min": round(energy_backstop_delta_min, 8),
        "candidate_energy": round(candidate_energy, 8),
        "baseline_energy": round(baseline_energy, 8),
        "energy_ratio": round(energy_ratio, 8),
        "energy_delta": round(energy_delta, 8),
        "energy_backstop_ratio_min": round(energy_backstop_ratio_min, 8),
        "energy_backstop_delta_min": round(energy_backstop_delta_min, 8),
        "absolute_radial_issue": absolute_radial_issue,
        "normalized_energy_backstop_triggered": (
            normalized_energy_backstop_triggered
        ),
        "blocking": blocking,
        "blocking_issue": issue if blocking else None,
        "radial_fraction_without_load_expansion_observation": not blocking,
    }


def _coherent_brush_relative_evidence(
    evaluation: dict[str, Any],
    *,
    relative_issues: list[str],
    absolute_issues: list[str],
) -> dict[str, Any]:
    """Distinguish pose-aligned fur direction from stronger brush energy."""

    issue = "coherent_brush_texture_expanded_vs_profile"
    absolute_coherent_issue = "brush_texture_coherence" in absolute_issues
    if issue not in relative_issues:
        return {
            "relative_issue_present": False,
            "absolute_coherent_issue": absolute_coherent_issue,
            "load_expanded": False,
            "long_line_energy_backstop_triggered": False,
            "blocking": False,
            "blocking_issue": None,
            "coherence_without_load_expansion_observation": False,
        }
    candidate = evaluation.get("candidate")
    baseline = evaluation.get("baseline")
    thresholds = evaluation.get("thresholds")
    if not all(isinstance(item, dict) for item in (candidate, baseline, thresholds)):
        raise RuntimeError("coherent_brush_evidence_schema_invalid")
    candidate_texture = candidate.get("metrics", {}).get("texture")
    baseline_texture = baseline.get("metrics", {}).get("texture")
    if not isinstance(candidate_texture, dict) or not isinstance(
        baseline_texture,
        dict,
    ):
        raise RuntimeError("coherent_brush_texture_metrics_missing")
    try:
        candidate_fraction = float(
            candidate_texture["structure_tensor_coherent_fraction"]
        )
        baseline_fraction = float(
            baseline_texture["structure_tensor_coherent_fraction"]
        )
        candidate_gradient_p90 = float(
            candidate_texture["coarse_gradient_p90"]
        )
        baseline_gradient_p90 = float(
            baseline_texture["coarse_gradient_p90"]
        )
        candidate_long_energy = float(
            candidate_texture["coherent_long_line_energy"]
        )
        baseline_long_energy = float(
            baseline_texture["coherent_long_line_energy"]
        )
        load_ratio_min = float(thresholds["coherent_fraction_ratio_max"])
        long_energy_ratio_min = float(
            thresholds["long_line_energy_ratio_max"]
        )
        delta_min = float(thresholds["texture_delta_min"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("coherent_brush_evidence_metrics_invalid") from exc
    values = (
        candidate_fraction,
        baseline_fraction,
        candidate_gradient_p90,
        baseline_gradient_p90,
        candidate_long_energy,
        baseline_long_energy,
        load_ratio_min,
        long_energy_ratio_min,
        delta_min,
    )
    if not all(np.isfinite(value) for value in values):
        raise RuntimeError("coherent_brush_evidence_metrics_non_finite")
    if (
        candidate_fraction < 0.0
        or baseline_fraction <= 0.0
        or candidate_gradient_p90 < 0.0
        or baseline_gradient_p90 <= 0.0
        or candidate_long_energy < 0.0
        or baseline_long_energy <= 0.0
        or load_ratio_min <= 0.0
        or long_energy_ratio_min <= 0.0
        or delta_min < 0.0
    ):
        raise RuntimeError("coherent_brush_evidence_metrics_out_of_range")
    candidate_load = candidate_fraction * candidate_gradient_p90
    baseline_load = baseline_fraction * baseline_gradient_p90
    if baseline_load <= 0.0:
        raise RuntimeError("coherent_brush_baseline_load_invalid")
    load_ratio = candidate_load / baseline_load
    load_delta = candidate_load - baseline_load
    long_energy_ratio = candidate_long_energy / baseline_long_energy
    long_energy_delta = candidate_long_energy - baseline_long_energy
    derived_values = (
        candidate_load,
        baseline_load,
        load_ratio,
        load_delta,
        long_energy_ratio,
        long_energy_delta,
    )
    if not all(np.isfinite(value) for value in derived_values):
        raise RuntimeError("coherent_brush_evidence_derived_metrics_non_finite")
    load_expanded = bool(
        load_ratio > load_ratio_min and load_delta > delta_min
    )
    long_line_energy_backstop_triggered = bool(
        long_energy_ratio > long_energy_ratio_min
        and long_energy_delta > delta_min
    )
    blocking = bool(load_expanded or long_line_energy_backstop_triggered)
    return {
        "relative_issue_present": True,
        "candidate_coherent_fraction": round(candidate_fraction, 8),
        "baseline_coherent_fraction": round(baseline_fraction, 8),
        "candidate_coarse_gradient_p90": round(candidate_gradient_p90, 8),
        "baseline_coarse_gradient_p90": round(baseline_gradient_p90, 8),
        "candidate_load_proxy": round(candidate_load, 8),
        "baseline_load_proxy": round(baseline_load, 8),
        "load_ratio": round(load_ratio, 8),
        "load_delta": round(load_delta, 8),
        "load_expansion_ratio_min": round(load_ratio_min, 8),
        "load_expansion_delta_min": round(delta_min, 8),
        "candidate_long_line_energy": round(candidate_long_energy, 8),
        "baseline_long_line_energy": round(baseline_long_energy, 8),
        "long_line_energy_ratio": round(long_energy_ratio, 8),
        "long_line_energy_delta": round(long_energy_delta, 8),
        "long_line_energy_backstop_ratio_min": round(
            long_energy_ratio_min,
            8,
        ),
        "long_line_energy_backstop_delta_min": round(delta_min, 8),
        "absolute_coherent_issue": absolute_coherent_issue,
        "load_expanded": load_expanded,
        "long_line_energy_backstop_triggered": (
            long_line_energy_backstop_triggered
        ),
        "blocking": blocking,
        "blocking_issue": issue if blocking else None,
        "coherence_without_load_expansion_observation": not blocking,
    }


def _apply_coherent_brush_temporal_corroboration(
    frame_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Require persistence for load-only coherence while preserving backstops."""

    issue = "coherent_brush_texture_expanded_vs_profile"
    action_reference_issue = _action_reference_issue_name(issue)
    evidence_frames: list[dict[str, Any]] = []
    load_only_indices: list[int] = []
    backstop_indices: list[int] = []
    for frame in frame_results:
        decision = frame.get("decision")
        if not isinstance(decision, dict):
            continue
        evidence = decision.get("coherent_brush_relative_evidence")
        if not isinstance(evidence, dict):
            continue
        load_expanded = evidence.get("load_expanded") is True
        backstop = evidence.get("long_line_energy_backstop_triggered") is True
        if evidence.get("relative_issue_present") is True and (
            load_expanded or backstop
        ):
            evidence_frames.append(frame)
            frame_index = int(frame["frame_index"])
            if backstop:
                backstop_indices.append(frame_index)
            elif load_expanded:
                load_only_indices.append(frame_index)

    evidence_indices = [int(frame["frame_index"]) for frame in evidence_frames]
    temporally_corroborated = bool(
        len(evidence_indices)
        >= FINAL_VIDEO_COHERENT_BRUSH_TEMPORAL_MIN_SAMPLES
    )
    downgraded_indices: list[int] = []
    for frame in frame_results:
        decision = frame.get("decision")
        if not isinstance(decision, dict):
            continue
        evidence = decision.get("coherent_brush_relative_evidence")
        if not isinstance(evidence, dict):
            continue
        evidence.update(
            temporal_policy=(
                "load_only_requires_recurrence; long_line_energy_backstop_"
                "remains_single_sample_blocking"
            ),
            temporal_min_samples=(
                FINAL_VIDEO_COHERENT_BRUSH_TEMPORAL_MIN_SAMPLES
            ),
            temporal_evidence_sample_indices=evidence_indices,
            temporal_evidence_sample_count=len(evidence_indices),
            temporally_corroborated=temporally_corroborated,
        )
        load_only_blocking = bool(
            evidence.get("blocking") is True
            and evidence.get("load_expanded") is True
            and evidence.get("long_line_energy_backstop_triggered") is not True
        )
        downgraded = bool(load_only_blocking and not temporally_corroborated)
        evidence["isolated_load_only_temporal_observation"] = downgraded
        if not downgraded:
            continue

        frame_index = int(frame["frame_index"])
        downgraded_indices.append(frame_index)
        evidence["blocking"] = False
        evidence["blocking_issue"] = None
        decision["texture_relative_blocking"] = [
            item
            for item in decision.get("texture_relative_blocking", [])
            if item != issue
        ]
        decision["blocking_issues"] = [
            item
            for item in decision.get("blocking_issues", [])
            if item != issue
        ]
        decision["blocking_issues_action_reference_scope"] = [
            item
            for item in decision.get(
                "blocking_issues_action_reference_scope",
                [],
            )
            if item != action_reference_issue
        ]
        decision["passed"] = not decision["blocking_issues"]
        frame["passed"] = decision["passed"]

    return {
        "policy": (
            "coherent_brush_load_temporal_corroboration_with_single_frame_"
            "long_line_backstop_v1"
        ),
        "minimum_evidence_samples": (
            FINAL_VIDEO_COHERENT_BRUSH_TEMPORAL_MIN_SAMPLES
        ),
        "evidence_sample_indices": evidence_indices,
        "evidence_sample_count": len(evidence_indices),
        "load_only_sample_indices": load_only_indices,
        "long_line_energy_backstop_sample_indices": backstop_indices,
        "temporally_corroborated": temporally_corroborated,
        "isolated_load_only_samples_downgraded_to_observation": (
            downgraded_indices
        ),
        "long_line_energy_backstop_remains_single_sample_blocking": True,
    }


def _apply_eating_initial_radial_loop_boundary_corroboration(
    frame_results: list[dict[str, Any]],
    *,
    action: str,
) -> dict[str, Any]:
    """Corroborate one initial radial ratio hit with the clean loop endpoint.

    Radial fraction is pose-sensitive.  This recovery is intentionally narrow:
    it applies only to an eating loop whose sole failed sample is frame zero,
    where the hash-bound prop-excluded reference already carries the recorded
    ``radial_long_lines`` baseline observation.  The final loop sample and all
    interior samples must pass the original audit.  The first and final samples
    must also have tightly matching prop-excluded radial load and energy, while
    every absolute radial and contract failure remains blocking.
    """

    target_issue = "radial_brush_texture_expanded_vs_profile"
    action_reference_issue = _action_reference_issue_name(target_issue)
    report: dict[str, Any] = {
        "policy": (
            "eating_initial_radial_clean_loop_endpoint_corroboration_v1"
        ),
        "action": action,
        "minimum_samples": FINAL_VIDEO_EATING_RADIAL_BOUNDARY_MIN_SAMPLES,
        "load_energy_spread_ratio_max": (
            FINAL_VIDEO_EATING_RADIAL_BOUNDARY_LOAD_ENERGY_SPREAD_MAX
        ),
        "component_spread_ratio_max": (
            FINAL_VIDEO_EATING_RADIAL_BOUNDARY_COMPONENT_SPREAD_MAX
        ),
        "requires_single_initial_blocking_sample": True,
        "requires_clean_terminal_and_all_interior_samples": True,
        "requires_dual_surface_radial_relative_issue": True,
        "requires_hash_bound_radial_long_lines_reference_observation": True,
        "preserves_absolute_radial_fail_closed": True,
        "applied": False,
        "downgraded_sample_indices": [],
    }

    def rejected(reason: str, **details: Any) -> dict[str, Any]:
        report["reason"] = reason
        report.update(details)
        return report

    if action != "eating":
        return rejected("action_not_eating")
    if len(frame_results) < FINAL_VIDEO_EATING_RADIAL_BOUNDARY_MIN_SAMPLES:
        return rejected("insufficient_sample_count")

    failed_positions = [
        position
        for position, frame in enumerate(frame_results)
        if frame.get("passed") is not True
    ]
    if failed_positions != [0]:
        return rejected(
            "blocking_contract_is_not_single_initial_sample",
            blocking_positions=failed_positions,
        )

    first = frame_results[0]
    last = frame_results[-1]
    if first.get("error") is not None or last.get("error") is not None:
        return rejected("loop_boundary_evaluation_error_present")
    first_decision = first.get("decision")
    last_decision = last.get("decision")
    if not isinstance(first_decision, dict) or not isinstance(
        last_decision,
        dict,
    ):
        return rejected("loop_boundary_decision_contract_missing")

    if (
        set(first_decision.get("blocking_issues") or []) != {target_issue}
        or set(
            first_decision.get(
                "blocking_issues_action_reference_scope",
            )
            or []
        )
        != {action_reference_issue}
        or set(first_decision.get("texture_relative_blocking") or [])
        != {target_issue}
        or first_decision.get("contour_relative_blocking")
        or first_decision.get("unknown_relative_blocking")
    ):
        return rejected("initial_blocking_issue_contract_mismatch")
    if (
        last.get("passed") is not True
        or last_decision.get("blocking_issues")
        or last_decision.get("contour_relative_blocking")
        or last_decision.get("unknown_relative_blocking")
    ):
        return rejected("terminal_loop_sample_not_clean")

    first_radial = first_decision.get("radial_relative_evidence")
    last_radial = last_decision.get("radial_relative_evidence")
    if not isinstance(first_radial, dict) or not isinstance(last_radial, dict):
        return rejected("radial_relative_evidence_contract_missing")
    if not (
        first_radial.get("relative_issue_present") is True
        and first_radial.get("blocking") is True
        and first_radial.get("absolute_radial_issue") is False
        and (
            first_radial.get("load_expanded") is True
            or first_radial.get("normalized_energy_backstop_triggered")
            is True
        )
    ):
        return rejected("initial_radial_evidence_contract_mismatch")
    if (
        last_radial.get("relative_issue_present") is True
        or last_radial.get("blocking") is True
        or last_radial.get("absolute_radial_issue") is True
    ):
        return rejected("terminal_radial_evidence_not_clean")

    allowed_absolute = {
        "brush_texture_coherence",
        "coarse_fur_stroke_energy",
    }
    evaluations: dict[str, dict[str, Any]] = {}
    for boundary_name, frame in (("initial", first), ("terminal", last)):
        texture_evaluation = frame.get("texture_evaluation")
        contour_evaluation = frame.get("evaluation")
        if not isinstance(texture_evaluation, dict) or not isinstance(
            contour_evaluation,
            dict,
        ):
            return rejected("loop_boundary_surface_evaluation_missing")
        if not (
            _evaluation_candidate_supported(texture_evaluation)
            and _evaluation_baseline_supported(texture_evaluation)
            and _evaluation_candidate_supported(contour_evaluation)
            and _evaluation_baseline_supported(contour_evaluation)
        ):
            return rejected("loop_boundary_surface_contract_unsupported")
        try:
            texture_relative = set(
                _require_evaluation_list(texture_evaluation, "relative_issues")
            )
            contour_relative = set(
                _require_evaluation_list(contour_evaluation, "relative_issues")
            )
            texture_absolute = set(
                _require_evaluation_list(
                    texture_evaluation,
                    "candidate_absolute_issues",
                )
            )
            contour_absolute = set(
                _require_evaluation_list(
                    contour_evaluation,
                    "candidate_absolute_issues",
                )
            )
            texture_baseline_issues = set(
                _require_evaluation_list(texture_evaluation, "baseline_issues")
            )
            contour_baseline_issues = set(
                _require_evaluation_list(contour_evaluation, "baseline_issues")
            )
        except RuntimeError:
            return rejected("loop_boundary_issue_schema_invalid")
        expected_relative = {target_issue} if boundary_name == "initial" else set()
        if (
            texture_relative != expected_relative
            or contour_relative != expected_relative
        ):
            return rejected(
                "loop_boundary_relative_issue_contract_mismatch",
                boundary=boundary_name,
            )
        if (
            texture_absolute - allowed_absolute
            or contour_absolute - allowed_absolute
        ):
            return rejected(
                "loop_boundary_absolute_surface_issue_present",
                boundary=boundary_name,
            )
        if texture_baseline_issues != {"radial_long_lines"} or (
            contour_baseline_issues
        ):
            return rejected(
                "hash_bound_radial_reference_observation_contract_mismatch",
                boundary=boundary_name,
            )
        evaluations[boundary_name] = texture_evaluation

    for boundary_name, decision in (
        ("initial", first_decision),
        ("terminal", last_decision),
    ):
        baseline_observations = decision.get("baseline_absolute_observations")
        if not isinstance(baseline_observations, dict) or set(
            baseline_observations.get(
                "hash_bound_prop_excluded_reference",
            )
            or []
        ) != {"radial_long_lines"} or baseline_observations.get(
            "unmasked_action_reference"
        ):
            return rejected(
                "decision_reference_observation_contract_mismatch",
                boundary=boundary_name,
            )

    def boundary_metrics(evaluation: dict[str, Any]) -> dict[str, float]:
        texture = (
            evaluation.get("candidate", {})
            .get("metrics", {})
            .get("texture")
        )
        thresholds = evaluation.get("thresholds")
        if not isinstance(texture, dict) or not isinstance(thresholds, dict):
            raise RuntimeError("radial_boundary_metrics_missing")
        try:
            fraction = float(texture["fine_radial_brush_fraction"])
            gradient = float(texture["fine_gradient_p90"])
            energy = float(texture["fine_radial_brush_energy"])
            absolute_fraction_max = float(
                thresholds["radial_brush_fraction_max"]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("radial_boundary_metrics_invalid") from exc
        values = (fraction, gradient, energy, absolute_fraction_max)
        if (
            not all(np.isfinite(value) for value in values)
            or min(fraction, gradient, energy, absolute_fraction_max) <= 0.0
        ):
            raise RuntimeError("radial_boundary_metrics_out_of_range")
        return {
            "fraction": fraction,
            "gradient": gradient,
            "energy": energy,
            "load": fraction * gradient,
            "absolute_fraction_max": absolute_fraction_max,
        }

    try:
        initial_metrics = boundary_metrics(evaluations["initial"])
        terminal_metrics = boundary_metrics(evaluations["terminal"])
    except RuntimeError as exc:
        return rejected(str(exc))
    if abs(
        initial_metrics["absolute_fraction_max"]
        - terminal_metrics["absolute_fraction_max"]
    ) > 1e-9:
        return rejected("radial_absolute_threshold_changed_across_loop")
    if max(
        initial_metrics["fraction"],
        terminal_metrics["fraction"],
    ) > initial_metrics["absolute_fraction_max"]:
        return rejected("radial_fraction_exceeds_absolute_ceiling")

    evidence_metric_bindings = {
        "candidate_fraction": "fraction",
        "candidate_gradient_p90": "gradient",
        "candidate_energy": "energy",
        "candidate_load_proxy": "load",
    }
    for evidence_key, metric_key in evidence_metric_bindings.items():
        try:
            evidence_value = float(first_radial[evidence_key])
        except (KeyError, TypeError, ValueError):
            return rejected("initial_radial_evidence_metric_missing")
        if (
            not np.isfinite(evidence_value)
            or abs(evidence_value - initial_metrics[metric_key]) > 1e-6
        ):
            return rejected("initial_radial_evidence_metric_mismatch")

    spreads = {
        key: max(initial_metrics[key], terminal_metrics[key])
        / min(initial_metrics[key], terminal_metrics[key])
        for key in ("fraction", "gradient", "energy", "load")
    }
    evidence = {
        "boundary_frame_indices": [
            int(first["frame_index"]),
            int(last["frame_index"]),
        ],
        "blocking_frame_indices": [int(first["frame_index"])],
        "initial_radial_fraction": round(initial_metrics["fraction"], 8),
        "terminal_radial_fraction": round(terminal_metrics["fraction"], 8),
        "initial_fine_gradient_p90": round(initial_metrics["gradient"], 8),
        "terminal_fine_gradient_p90": round(terminal_metrics["gradient"], 8),
        "initial_radial_energy": round(initial_metrics["energy"], 8),
        "terminal_radial_energy": round(terminal_metrics["energy"], 8),
        "initial_radial_load": round(initial_metrics["load"], 8),
        "terminal_radial_load": round(terminal_metrics["load"], 8),
        "radial_fraction_spread_ratio": round(spreads["fraction"], 8),
        "fine_gradient_spread_ratio": round(spreads["gradient"], 8),
        "radial_energy_spread_ratio": round(spreads["energy"], 8),
        "radial_load_spread_ratio": round(spreads["load"], 8),
        "radial_fraction_absolute_max": round(
            initial_metrics["absolute_fraction_max"],
            8,
        ),
    }
    if (
        spreads["fraction"]
        > FINAL_VIDEO_EATING_RADIAL_BOUNDARY_COMPONENT_SPREAD_MAX
        or spreads["gradient"]
        > FINAL_VIDEO_EATING_RADIAL_BOUNDARY_COMPONENT_SPREAD_MAX
    ):
        return rejected("radial_boundary_components_not_stable", **evidence)
    if (
        spreads["energy"]
        > FINAL_VIDEO_EATING_RADIAL_BOUNDARY_LOAD_ENERGY_SPREAD_MAX
        or spreads["load"]
        > FINAL_VIDEO_EATING_RADIAL_BOUNDARY_LOAD_ENERGY_SPREAD_MAX
    ):
        return rejected("radial_boundary_load_or_energy_not_stable", **evidence)

    first_radial["blocking_before_eating_loop_endpoint_corroboration"] = True
    first_radial["eating_initial_radial_loop_endpoint_observation"] = True
    first_radial["blocking"] = False
    first_radial["blocking_issue"] = None
    first_decision["texture_relative_blocking"] = [
        item
        for item in first_decision.get("texture_relative_blocking", [])
        if item != target_issue
    ]
    first_decision["blocking_issues"] = [
        item
        for item in first_decision.get("blocking_issues", [])
        if item != target_issue
    ]
    first_decision["blocking_issues_action_reference_scope"] = [
        item
        for item in first_decision.get(
            "blocking_issues_action_reference_scope",
            [],
        )
        if item != action_reference_issue
    ]
    first_decision["passed"] = not first_decision["blocking_issues"]
    first["passed"] = first_decision["passed"]
    report.update(
        evidence,
        applied=True,
        reason="isolated_initial_radial_ratio_corroborated_by_clean_loop_endpoint",
        downgraded_sample_indices=[int(first["frame_index"])],
    )
    return report


def _apply_eating_initial_radial_bounded_reference_corroboration(
    frame_results: list[dict[str, Any]],
    *,
    action: str,
) -> dict[str, Any]:
    """Normalize one bounded initial radial peak when the endpoint differs.

    This is a fallback after the stricter clean-loop endpoint contract.  It
    accepts only the sole initial eating blocker, requires every later sample
    to pass, preserves the absolute radial ceiling, and bounds fraction,
    gradient, normalized energy, and gradient-weighted load against the
    hash-bound prop-excluded action reference.
    """

    target_issue = "radial_brush_texture_expanded_vs_profile"
    action_reference_issue = _action_reference_issue_name(target_issue)
    allowed_observations = {
        "coherent_brush_texture_expanded_vs_profile",
        "long_fur_energy_expanded_vs_profile",
        "contour_fringe_expanded_vs_profile",
        *POSE_RELATIVE_OBSERVATIONS,
    }
    report: dict[str, Any] = {
        "policy": "eating_initial_radial_bounded_action_reference_v1",
        "action": action,
        "minimum_samples": FINAL_VIDEO_EATING_RADIAL_BOUNDARY_MIN_SAMPLES,
        "requires_single_initial_blocking_sample": True,
        "requires_all_later_samples_original_or_normalized_pass": True,
        "requires_hash_bound_radial_long_lines_reference_observation": True,
        "reference_ratio_margin": (
            FINAL_VIDEO_EATING_RADIAL_BOUNDED_REFERENCE_RATIO_MARGIN
        ),
        "load_delta_multiplier": (
            FINAL_VIDEO_EATING_RADIAL_BOUNDED_REFERENCE_LOAD_DELTA_MULTIPLIER
        ),
        "preserves_absolute_radial_fail_closed": True,
        "applied": False,
        "downgraded_sample_indices": [],
    }

    def rejected(reason: str, **details: Any) -> dict[str, Any]:
        report["reason"] = reason
        report.update(details)
        return report

    if action != "eating":
        return rejected("action_not_eating")
    if len(frame_results) < FINAL_VIDEO_EATING_RADIAL_BOUNDARY_MIN_SAMPLES:
        return rejected("insufficient_sample_count")
    failed_positions = [
        position
        for position, frame in enumerate(frame_results)
        if frame.get("passed") is not True
    ]
    if failed_positions != [0]:
        return rejected(
            "blocking_contract_is_not_single_initial_sample",
            blocking_positions=failed_positions,
        )

    allowed_absolute = {
        "brush_texture_coherence",
        "coarse_fur_stroke_energy",
    }
    initial_frame = frame_results[0]
    initial_decision: dict[str, Any] | None = None
    initial_texture_evaluation: dict[str, Any] | None = None
    initial_texture_relative: set[str] = set()
    initial_contour_relative: set[str] = set()
    for position, frame in enumerate(frame_results):
        decision = frame.get("decision")
        texture_evaluation = frame.get("texture_evaluation")
        contour_evaluation = frame.get("evaluation")
        if not all(
            isinstance(item, dict)
            for item in (decision, texture_evaluation, contour_evaluation)
        ):
            return rejected("sample_evaluation_contract_missing")
        if frame.get("error") is not None:
            return rejected("sample_evaluation_error_present")
        if not (
            _evaluation_candidate_supported(texture_evaluation)
            and _evaluation_baseline_supported(texture_evaluation)
            and _evaluation_candidate_supported(contour_evaluation)
            and _evaluation_baseline_supported(contour_evaluation)
        ):
            return rejected("sample_surface_contract_unsupported")
        try:
            texture_relative = set(
                _require_evaluation_list(texture_evaluation, "relative_issues")
            )
            contour_relative = set(
                _require_evaluation_list(contour_evaluation, "relative_issues")
            )
            texture_absolute = set(
                _require_evaluation_list(
                    texture_evaluation,
                    "candidate_absolute_issues",
                )
            )
            contour_absolute = set(
                _require_evaluation_list(
                    contour_evaluation,
                    "candidate_absolute_issues",
                )
            )
            texture_baseline_issues = set(
                _require_evaluation_list(texture_evaluation, "baseline_issues")
            )
            contour_baseline_issues = set(
                _require_evaluation_list(contour_evaluation, "baseline_issues")
            )
        except RuntimeError:
            return rejected("sample_issue_schema_invalid")
        if (
            texture_absolute - allowed_absolute
            or contour_absolute - allowed_absolute
        ):
            return rejected("sample_absolute_surface_issue_present")
        if texture_baseline_issues != {"radial_long_lines"} or (
            contour_baseline_issues
        ):
            return rejected(
                "hash_bound_radial_reference_observation_contract_mismatch"
            )
        baseline_observations = decision.get("baseline_absolute_observations")
        if not isinstance(baseline_observations, dict) or set(
            baseline_observations.get(
                "hash_bound_prop_excluded_reference",
            )
            or []
        ) != {"radial_long_lines"} or baseline_observations.get(
            "unmasked_action_reference"
        ):
            return rejected(
                "decision_reference_observation_contract_mismatch"
            )
        radial = decision.get("radial_relative_evidence")
        coherent = decision.get("coherent_brush_relative_evidence")
        long_fur = decision.get("long_fur_relative_evidence")
        if not all(
            isinstance(item, dict) for item in (radial, coherent, long_fur)
        ):
            return rejected("sample_relative_evidence_contract_missing")
        if (
            decision.get("contour_relative_blocking")
            or decision.get("unknown_relative_blocking")
            or coherent.get("blocking") is True
            or long_fur.get("blocking") is True
            or radial.get("absolute_radial_issue") is True
        ):
            return rejected("sample_strong_surface_evidence_present")
        if position == 0:
            if (
                set(decision.get("blocking_issues") or []) != {target_issue}
                or set(
                    decision.get(
                        "blocking_issues_action_reference_scope",
                    )
                    or []
                )
                != {action_reference_issue}
                or set(decision.get("texture_relative_blocking") or [])
                != {target_issue}
                or target_issue not in texture_relative
                or radial.get("relative_issue_present") is not True
                or radial.get("blocking") is not True
                or not (
                    radial.get("load_expanded") is True
                    or radial.get("normalized_energy_backstop_triggered")
                    is True
                )
            ):
                return rejected("initial_radial_blocking_contract_mismatch")
            if (
                texture_relative - allowed_observations - {target_issue}
                or contour_relative - allowed_observations - {target_issue}
            ):
                return rejected("initial_unknown_relative_observation_present")
            initial_decision = decision
            initial_texture_evaluation = texture_evaluation
            initial_texture_relative = texture_relative
            initial_contour_relative = contour_relative
        elif (
            frame.get("passed") is not True
            or decision.get("blocking_issues")
            or radial.get("blocking") is True
        ):
            return rejected("later_sample_not_clean")

    if not isinstance(initial_decision, dict) or not isinstance(
        initial_texture_evaluation,
        dict,
    ):
        return rejected("initial_sample_contract_missing")
    candidate_texture = (
        initial_texture_evaluation.get("candidate", {})
        .get("metrics", {})
        .get("texture")
    )
    baseline_texture = (
        initial_texture_evaluation.get("baseline", {})
        .get("metrics", {})
        .get("texture")
    )
    thresholds = initial_texture_evaluation.get("thresholds")
    if not all(
        isinstance(item, dict)
        for item in (candidate_texture, baseline_texture, thresholds)
    ):
        return rejected("initial_radial_metrics_missing")
    try:
        candidate = {
            "fraction": float(candidate_texture["fine_radial_brush_fraction"]),
            "gradient": float(candidate_texture["fine_gradient_p90"]),
            "energy": float(candidate_texture["fine_radial_brush_energy"]),
        }
        baseline = {
            "fraction": float(baseline_texture["fine_radial_brush_fraction"]),
            "gradient": float(baseline_texture["fine_gradient_p90"]),
            "energy": float(baseline_texture["fine_radial_brush_energy"]),
        }
        absolute_fraction_max = float(
            thresholds["radial_brush_fraction_max"]
        )
        base_ratio_max = float(thresholds["radial_brush_ratio_max"])
        texture_delta_min = float(thresholds["texture_delta_min"])
    except (KeyError, TypeError, ValueError) as exc:
        return rejected(
            "initial_radial_metric_schema_invalid",
            metric_error=type(exc).__name__,
        )
    candidate["load"] = candidate["fraction"] * candidate["gradient"]
    baseline["load"] = baseline["fraction"] * baseline["gradient"]
    numeric_values = (
        *candidate.values(),
        *baseline.values(),
        absolute_fraction_max,
        base_ratio_max,
        texture_delta_min,
    )
    if (
        not all(np.isfinite(float(value)) for value in numeric_values)
        or min(*candidate.values(), *baseline.values()) <= 0.0
        or absolute_fraction_max <= 0.0
        or base_ratio_max <= 1.0
        or texture_delta_min < 0.0
    ):
        return rejected("initial_radial_metrics_out_of_range")
    expansion_ratios = {
        key: candidate[key] / baseline[key]
        for key in ("fraction", "gradient", "energy", "load")
    }
    ratio_ceiling = (
        base_ratio_max
        + FINAL_VIDEO_EATING_RADIAL_BOUNDED_REFERENCE_RATIO_MARGIN
    )
    load_delta = candidate["load"] - baseline["load"]
    load_delta_ceiling = (
        texture_delta_min
        * FINAL_VIDEO_EATING_RADIAL_BOUNDED_REFERENCE_LOAD_DELTA_MULTIPLIER
    )
    evidence = {
        "blocking_frame_indices": [int(initial_frame["frame_index"])],
        "initial_texture_relative_issues": sorted(initial_texture_relative),
        "initial_unmasked_relative_issues": sorted(initial_contour_relative),
        "initial_unmasked_radial_corroborated": (
            target_issue in initial_contour_relative
        ),
        "candidate_radial_fraction": round(candidate["fraction"], 8),
        "reference_radial_fraction": round(baseline["fraction"], 8),
        "candidate_fine_gradient_p90": round(candidate["gradient"], 8),
        "reference_fine_gradient_p90": round(baseline["gradient"], 8),
        "candidate_radial_energy": round(candidate["energy"], 8),
        "reference_radial_energy": round(baseline["energy"], 8),
        "candidate_radial_load": round(candidate["load"], 8),
        "reference_radial_load": round(baseline["load"], 8),
        "radial_fraction_ratio": round(expansion_ratios["fraction"], 8),
        "fine_gradient_ratio": round(expansion_ratios["gradient"], 8),
        "radial_energy_ratio": round(expansion_ratios["energy"], 8),
        "radial_load_ratio": round(expansion_ratios["load"], 8),
        "bounded_reference_ratio_ceiling": round(ratio_ceiling, 8),
        "radial_load_delta": round(load_delta, 8),
        "radial_load_delta_ceiling": round(load_delta_ceiling, 8),
        "radial_fraction_absolute_max": round(absolute_fraction_max, 8),
    }
    if candidate["fraction"] > absolute_fraction_max:
        return rejected("radial_fraction_exceeds_absolute_ceiling", **evidence)
    if (
        any(value > ratio_ceiling for value in expansion_ratios.values())
        or load_delta > load_delta_ceiling
    ):
        return rejected("initial_radial_peak_exceeds_bounded_reference", **evidence)

    radial = initial_decision["radial_relative_evidence"]
    radial["blocking_before_bounded_reference_corroboration"] = True
    radial["eating_initial_radial_bounded_reference_observation"] = True
    radial["blocking"] = False
    radial["blocking_issue"] = None
    initial_decision["texture_relative_blocking"] = [
        issue
        for issue in initial_decision.get("texture_relative_blocking", [])
        if issue != target_issue
    ]
    initial_decision["blocking_issues"] = [
        issue
        for issue in initial_decision.get("blocking_issues", [])
        if issue != target_issue
    ]
    initial_decision["blocking_issues_action_reference_scope"] = [
        issue
        for issue in initial_decision.get(
            "blocking_issues_action_reference_scope",
            [],
        )
        if issue != action_reference_issue
    ]
    initial_decision["eating_initial_radial_bounded_reference"] = {
        "applied": True,
        "policy": report["policy"],
        "ratio_ceiling": round(ratio_ceiling, 8),
        "load_delta_ceiling": round(load_delta_ceiling, 8),
    }
    initial_decision["passed"] = not initial_decision["blocking_issues"]
    initial_frame["passed"] = initial_decision["passed"]
    report.update(
        applied=True,
        reason="isolated_initial_radial_peak_within_bounded_action_reference",
        downgraded_sample_indices=[int(initial_frame["frame_index"])],
        **evidence,
    )
    return report


def _apply_eating_loop_boundary_decode_anchor(
    frame_results: list[dict[str, Any]],
    *,
    action: str,
) -> dict[str, Any]:
    """Use clean loop boundaries to distinguish an eating pose-phase crest.

    The static Qwen keyframe remains the appearance authority.  A decoded
    boundary anchor is used only as corroboration when both loop boundaries
    pass the original audit and a contiguous interior run is blocked solely by
    the same coarse/long texture pair.  Any strong growth against the decoded
    anchor, absolute ceiling, contour/radial evidence, or contract anomaly
    keeps the original failure.
    """

    target_issues = {
        "coarse_strokes_expanded_vs_profile",
        "long_fur_energy_expanded_vs_profile",
    }
    nonblocking_observation_issues = {
        "contour_fringe_expanded_vs_profile",
    }
    action_reference_issues = {
        _action_reference_issue_name(issue) for issue in target_issues
    }
    report: dict[str, Any] = {
        "policy": "eating_loop_boundary_decode_anchor_corroboration_v1",
        "action": action,
        "minimum_samples": FINAL_VIDEO_EATING_BOUNDARY_ANCHOR_MIN_SAMPLES,
        "minimum_contiguous_blocking_samples": (
            FINAL_VIDEO_EATING_BOUNDARY_ANCHOR_MIN_BLOCKING_SAMPLES
        ),
        "boundary_spread_ratio_max": (
            FINAL_VIDEO_EATING_BOUNDARY_ANCHOR_SPREAD_MAX
        ),
        "requires_clean_first_and_last_sample": True,
        "allowed_boundary_observation_issues": sorted(
            nonblocking_observation_issues
        ),
        "requires_contiguous_interior_samples": True,
        "preserves_original_ratio_and_delta_thresholds": True,
        "applied": False,
        "downgraded_sample_indices": [],
    }

    def rejected(reason: str, **details: Any) -> dict[str, Any]:
        report["reason"] = reason
        report.update(details)
        return report

    if action != "eating":
        return rejected("action_not_eating")
    if len(frame_results) < FINAL_VIDEO_EATING_BOUNDARY_ANCHOR_MIN_SAMPLES:
        return rejected("insufficient_sample_count")

    allowed_absolute = {
        "brush_texture_coherence",
        "coarse_fur_stroke_energy",
    }
    rows: list[dict[str, Any]] = []
    blocking_positions: list[int] = []
    for position, frame in enumerate(frame_results):
        decision = frame.get("decision")
        texture_evaluation = frame.get("texture_evaluation")
        contour_evaluation = frame.get("evaluation")
        if not all(
            isinstance(item, dict)
            for item in (decision, texture_evaluation, contour_evaluation)
        ):
            return rejected("sample_evaluation_contract_missing")
        if frame.get("error") is not None:
            return rejected("sample_evaluation_error_present")
        if not (
            _evaluation_candidate_supported(texture_evaluation)
            and _evaluation_baseline_supported(texture_evaluation)
            and _evaluation_candidate_supported(contour_evaluation)
            and _evaluation_baseline_supported(contour_evaluation)
        ):
            return rejected("sample_surface_contract_unsupported")
        try:
            texture_relative = set(
                _require_evaluation_list(texture_evaluation, "relative_issues")
            )
            contour_relative = set(
                _require_evaluation_list(contour_evaluation, "relative_issues")
            )
            texture_absolute = set(
                _require_evaluation_list(
                    texture_evaluation,
                    "candidate_absolute_issues",
                )
            )
            contour_absolute = set(
                _require_evaluation_list(
                    contour_evaluation,
                    "candidate_absolute_issues",
                )
            )
            texture_baseline_issues = _require_evaluation_list(
                texture_evaluation,
                "baseline_issues",
            )
            contour_baseline_issues = _require_evaluation_list(
                contour_evaluation,
                "baseline_issues",
            )
        except RuntimeError:
            return rejected("sample_issue_schema_invalid")
        if (
            texture_absolute - allowed_absolute
            or contour_absolute - allowed_absolute
            or texture_baseline_issues
            or contour_baseline_issues
        ):
            return rejected("sample_absolute_or_baseline_contract_failed")
        if (
            decision.get("contour_relative_blocking")
            or decision.get("unknown_relative_blocking")
        ):
            return rejected("sample_contour_or_unknown_issue_present")
        blocking = set(decision.get("blocking_issues") or [])
        if blocking:
            if (
                blocking != target_issues
                or set(
                    decision.get(
                        "blocking_issues_action_reference_scope"
                    )
                    or []
                )
                != action_reference_issues
                or not target_issues.issubset(texture_relative)
                or not target_issues.issubset(contour_relative)
                or (
                    (texture_relative | contour_relative)
                    - target_issues
                    - nonblocking_observation_issues
                )
            ):
                return rejected("sample_blocking_issue_contract_mismatch")
            radial = decision.get("radial_relative_evidence")
            coherent = decision.get("coherent_brush_relative_evidence")
            long_fur = decision.get("long_fur_relative_evidence")
            if not all(
                isinstance(item, dict)
                for item in (radial, coherent, long_fur)
            ):
                return rejected("sample_relative_evidence_contract_missing")
            if (
                radial.get("blocking") is True
                or coherent.get("blocking") is True
                or long_fur.get("contour_corroborated") is True
                or long_fur.get("absolute_long_issue") is True
            ):
                return rejected("sample_strong_surface_evidence_present")
            blocking_positions.append(position)
        elif frame.get("passed") is not True:
            return rejected("sample_failed_without_supported_blocking_issue")

        candidate_texture = (
            texture_evaluation.get("candidate", {})
            .get("metrics", {})
            .get("texture")
        )
        thresholds = texture_evaluation.get("thresholds")
        if not isinstance(candidate_texture, dict) or not isinstance(
            thresholds,
            dict,
        ):
            return rejected("sample_texture_metrics_missing")
        try:
            row = {
                "frame_index": int(frame["frame_index"]),
                "candidate_coarse": float(
                    candidate_texture["coarse_gradient_p90"]
                ),
                "candidate_long": float(
                    candidate_texture["coherent_long_line_energy"]
                ),
                "coarse_ratio_max": float(
                    thresholds["coarse_gradient_ratio_max"]
                ),
                "long_ratio_max": float(
                    thresholds["long_line_energy_ratio_max"]
                ),
                "texture_delta_min": float(
                    thresholds["texture_delta_min"]
                ),
                "coarse_absolute_max": float(
                    thresholds["coarse_gradient_p90_max"]
                ),
                "long_absolute_max": float(
                    thresholds["coherent_long_line_energy_max"]
                ),
                "texture_relative": texture_relative,
                "contour_relative": contour_relative,
            }
        except (KeyError, TypeError, ValueError) as exc:
            return rejected(
                "sample_texture_metric_schema_invalid",
                metric_error=type(exc).__name__,
            )
        numeric_values = (
            row["candidate_coarse"],
            row["candidate_long"],
            row["coarse_ratio_max"],
            row["long_ratio_max"],
            row["texture_delta_min"],
            row["coarse_absolute_max"],
            row["long_absolute_max"],
        )
        if (
            not all(np.isfinite(float(value)) for value in numeric_values)
            or row["candidate_coarse"] <= 0.0
            or row["candidate_long"] <= 0.0
            or row["coarse_ratio_max"] <= 1.0
            or row["long_ratio_max"] <= 1.0
            or row["texture_delta_min"] < 0.0
            or row["coarse_absolute_max"] <= 0.0
            or row["long_absolute_max"] <= 0.0
        ):
            return rejected("sample_texture_metrics_out_of_range")
        rows.append(row)

    if len(blocking_positions) < (
        FINAL_VIDEO_EATING_BOUNDARY_ANCHOR_MIN_BLOCKING_SAMPLES
    ):
        return rejected("insufficient_blocking_sample_count")
    if (
        blocking_positions[0] == 0
        or blocking_positions[-1] == len(rows) - 1
        or blocking_positions
        != list(range(blocking_positions[0], blocking_positions[-1] + 1))
    ):
        return rejected("blocking_samples_not_contiguous_interior_run")

    first = rows[0]
    last = rows[-1]
    if (
        frame_results[0].get("passed") is not True
        or frame_results[-1].get("passed") is not True
        or (
            first["texture_relative"]
            - nonblocking_observation_issues
        )
        or (
            first["contour_relative"]
            - nonblocking_observation_issues
        )
        or (
            last["texture_relative"]
            - nonblocking_observation_issues
        )
        or (
            last["contour_relative"]
            - nonblocking_observation_issues
        )
    ):
        return rejected("loop_boundary_samples_not_clean")
    stable_threshold_keys = (
        "coarse_ratio_max",
        "long_ratio_max",
        "texture_delta_min",
        "coarse_absolute_max",
        "long_absolute_max",
    )
    if any(
        max(float(row[key]) for row in rows)
        - min(float(row[key]) for row in rows)
        > 1e-9
        for key in stable_threshold_keys
    ):
        return rejected("surface_thresholds_changed_across_samples")

    boundary_coarse_spread = max(
        float(first["candidate_coarse"]),
        float(last["candidate_coarse"]),
    ) / min(
        float(first["candidate_coarse"]),
        float(last["candidate_coarse"]),
    )
    boundary_long_spread = max(
        float(first["candidate_long"]),
        float(last["candidate_long"]),
    ) / min(
        float(first["candidate_long"]),
        float(last["candidate_long"]),
    )
    anchor_coarse = max(
        float(first["candidate_coarse"]),
        float(last["candidate_coarse"]),
    )
    anchor_long = max(
        float(first["candidate_long"]),
        float(last["candidate_long"]),
    )
    target_rows = [rows[position] for position in blocking_positions]
    target_coarse = [float(row["candidate_coarse"]) for row in target_rows]
    target_long = [float(row["candidate_long"]) for row in target_rows]
    target_coarse_spread = max(target_coarse) / min(target_coarse)
    target_long_spread = max(target_long) / min(target_long)
    coarse_ratio_max = float(first["coarse_ratio_max"])
    long_strong_ratio_min = (
        float(first["long_ratio_max"])
        + FINAL_VIDEO_LONG_LINE_RATIO_MARGIN
    )
    coarse_delta_min = float(first["texture_delta_min"])
    long_strong_delta_min = (
        float(first["texture_delta_min"])
        * FINAL_VIDEO_LONG_LINE_DELTA_MULTIPLIER
    )
    peak_coarse = max(target_coarse)
    peak_long = max(target_long)
    anchor_coarse_ratio = peak_coarse / anchor_coarse
    anchor_coarse_delta = peak_coarse - anchor_coarse
    anchor_long_ratio = peak_long / anchor_long
    anchor_long_delta = peak_long - anchor_long
    evidence = {
        "boundary_frame_indices": [
            int(first["frame_index"]),
            int(last["frame_index"]),
        ],
        "blocking_frame_indices": [
            int(rows[position]["frame_index"])
            for position in blocking_positions
        ],
        "boundary_anchor_coarse_gradient_p90": round(anchor_coarse, 8),
        "boundary_anchor_coherent_long_line_energy": round(anchor_long, 8),
        "peak_blocking_coarse_gradient_p90": round(peak_coarse, 8),
        "peak_blocking_coherent_long_line_energy": round(peak_long, 8),
        "anchor_coarse_ratio": round(anchor_coarse_ratio, 8),
        "anchor_coarse_delta": round(anchor_coarse_delta, 8),
        "anchor_long_ratio": round(anchor_long_ratio, 8),
        "anchor_long_delta": round(anchor_long_delta, 8),
        "coarse_ratio_max": round(coarse_ratio_max, 8),
        "coarse_delta_min": round(coarse_delta_min, 8),
        "long_strong_ratio_min": round(long_strong_ratio_min, 8),
        "long_strong_delta_min": round(long_strong_delta_min, 8),
        "boundary_coarse_spread_ratio": round(boundary_coarse_spread, 8),
        "boundary_long_spread_ratio": round(boundary_long_spread, 8),
        "blocking_coarse_spread_ratio": round(target_coarse_spread, 8),
        "blocking_long_spread_ratio": round(target_long_spread, 8),
    }
    if (
        boundary_coarse_spread
        > FINAL_VIDEO_EATING_BOUNDARY_ANCHOR_SPREAD_MAX
        or boundary_long_spread
        > FINAL_VIDEO_EATING_BOUNDARY_ANCHOR_SPREAD_MAX
        or target_coarse_spread
        > FINAL_VIDEO_EATING_BOUNDARY_ANCHOR_SPREAD_MAX
        or target_long_spread
        > FINAL_VIDEO_EATING_BOUNDARY_ANCHOR_SPREAD_MAX
    ):
        return rejected("boundary_or_action_phase_not_temporally_stable", **evidence)
    if (
        max(float(row["candidate_coarse"]) for row in rows)
        > float(first["coarse_absolute_max"])
        or max(float(row["candidate_long"]) for row in rows)
        > float(first["long_absolute_max"])
    ):
        return rejected("candidate_texture_exceeds_absolute_ceiling", **evidence)
    if (
        anchor_coarse_ratio > coarse_ratio_max
        and anchor_coarse_delta > coarse_delta_min
    ) or (
        anchor_long_ratio > long_strong_ratio_min
        and anchor_long_delta > long_strong_delta_min
    ):
        return rejected("action_phase_remains_strong_against_boundary_anchor", **evidence)

    downgraded_indices: list[int] = []
    for position in blocking_positions:
        frame = frame_results[position]
        decision = frame["decision"]
        frame_index = int(frame["frame_index"])
        downgraded_indices.append(frame_index)
        decision["texture_relative_blocking"] = [
            issue
            for issue in decision.get("texture_relative_blocking", [])
            if issue not in target_issues
        ]
        decision["blocking_issues"] = [
            issue
            for issue in decision.get("blocking_issues", [])
            if issue not in target_issues
        ]
        decision["blocking_issues_action_reference_scope"] = [
            issue
            for issue in decision.get(
                "blocking_issues_action_reference_scope",
                [],
            )
            if issue not in action_reference_issues
        ]
        long_fur = decision["long_fur_relative_evidence"]
        long_fur["blocking_before_eating_boundary_anchor"] = bool(
            long_fur.get("blocking") is True
        )
        long_fur["blocking"] = False
        long_fur["blocking_issue"] = None
        long_fur["eating_action_phase_boundary_anchor_observation"] = True
        decision["eating_boundary_anchor_normalized_observations"] = sorted(
            target_issues
        )
        decision["eating_loop_boundary_decode_anchor"] = {
            "applied": True,
            "policy": report["policy"],
            "boundary_anchor_coarse_gradient_p90": round(anchor_coarse, 8),
            "boundary_anchor_coherent_long_line_energy": round(anchor_long, 8),
        }
        decision["passed"] = not decision["blocking_issues"]
        frame["passed"] = decision["passed"]

    report.update(
        applied=True,
        reason="contiguous_action_phase_not_strong_against_clean_boundaries",
        downgraded_sample_indices=downgraded_indices,
        **evidence,
    )
    return report


def _apply_eating_isolated_long_fur_temporal_corroboration(
    frame_results: list[dict[str, Any]],
    *,
    action: str,
) -> dict[str, Any]:
    """Downgrade only a threshold-edge, single-frame eating line peak.

    Long fur is a persistent surface change.  A sole interior strong-line hit
    with no absolute or contour-expansion evidence can instead be a pose-phase
    crest in the decoded eating motion.  An adjacent frame must carry the same
    prop-excluded texture observation.  The recovery reuses the original
    ratio/delta limits against that neighbour, records whether the unmasked
    surface agrees, and allows only one existing texture-delta margin above
    the strong action-reference threshold.  One independent initial radial
    blocker may remain for the dedicated radial contracts to validate next.
    """

    target_issue = "long_fur_energy_expanded_vs_profile"
    action_reference_issue = _action_reference_issue_name(target_issue)
    report: dict[str, Any] = {
        "policy": (
            "eating_isolated_strong_long_line_temporal_corroboration_v2"
        ),
        "action": action,
        "minimum_samples": FINAL_VIDEO_EATING_ISOLATED_LONG_FUR_MIN_SAMPLES,
        "requires_single_interior_blocking_sample": True,
        "records_blocking_frame_unmasked_long_line_corroboration": True,
        "requires_adjacent_prop_excluded_long_line_observation": True,
        "allows_one_independent_initial_radial_blocker_for_later_validation": (
            True
        ),
        "reuses_original_neighbor_ratio_and_delta_thresholds": True,
        "strong_ceiling_uses_one_existing_texture_delta_margin": True,
        "preserves_absolute_contour_radial_coherent_and_unknown_fail_closed": (
            True
        ),
        "applied": False,
        "downgraded_sample_indices": [],
    }

    def rejected(reason: str, **details: Any) -> dict[str, Any]:
        report["reason"] = reason
        report.update(details)
        return report

    if action != "eating":
        return rejected("action_not_eating")
    if len(frame_results) < FINAL_VIDEO_EATING_ISOLATED_LONG_FUR_MIN_SAMPLES:
        return rejected("insufficient_sample_count")

    allowed_absolute = {
        "brush_texture_coherence",
        "coarse_fur_stroke_energy",
    }
    rows: list[dict[str, Any]] = []
    blocking_positions: list[int] = []
    radial_companion_positions: list[int] = []
    radial_issue = "radial_brush_texture_expanded_vs_profile"
    radial_action_reference_issue = _action_reference_issue_name(radial_issue)
    for position, frame in enumerate(frame_results):
        decision = frame.get("decision")
        texture_evaluation = frame.get("texture_evaluation")
        contour_evaluation = frame.get("evaluation")
        if not all(
            isinstance(item, dict)
            for item in (decision, texture_evaluation, contour_evaluation)
        ):
            return rejected("sample_evaluation_contract_missing")
        if frame.get("error") is not None:
            return rejected("sample_evaluation_error_present")
        if not (
            _evaluation_candidate_supported(texture_evaluation)
            and _evaluation_baseline_supported(texture_evaluation)
            and _evaluation_candidate_supported(contour_evaluation)
            and _evaluation_baseline_supported(contour_evaluation)
        ):
            return rejected("sample_surface_contract_unsupported")
        try:
            texture_relative = set(
                _require_evaluation_list(texture_evaluation, "relative_issues")
            )
            contour_relative = set(
                _require_evaluation_list(contour_evaluation, "relative_issues")
            )
            texture_absolute = set(
                _require_evaluation_list(
                    texture_evaluation,
                    "candidate_absolute_issues",
                )
            )
            contour_absolute = set(
                _require_evaluation_list(
                    contour_evaluation,
                    "candidate_absolute_issues",
                )
            )
        except RuntimeError:
            return rejected("sample_issue_schema_invalid")
        if (
            texture_absolute - allowed_absolute
            or contour_absolute - allowed_absolute
        ):
            return rejected(
                "sample_absolute_long_or_other_surface_issue_present"
            )

        radial = decision.get("radial_relative_evidence")
        coherent = decision.get("coherent_brush_relative_evidence")
        long_fur = decision.get("long_fur_relative_evidence")
        if not all(
            isinstance(item, dict) for item in (radial, coherent, long_fur)
        ):
            return rejected("sample_relative_evidence_contract_missing")
        blocking = set(decision.get("blocking_issues") or [])
        action_reference_blocking = set(
            decision.get("blocking_issues_action_reference_scope") or []
        )
        is_initial_radial_companion = bool(
            position == 0
            and blocking == {radial_issue}
            and action_reference_blocking == {radial_action_reference_issue}
            and radial.get("relative_issue_present") is True
            and radial.get("blocking") is True
            and radial.get("absolute_radial_issue") is False
            and coherent.get("blocking") is not True
            and long_fur.get("blocking") is not True
        )
        if (
            decision.get("contour_relative_blocking")
            or decision.get("unknown_relative_blocking")
            or (
                radial.get("blocking") is True
                and not is_initial_radial_companion
            )
            or coherent.get("blocking") is True
            or long_fur.get("contour_corroborated") is True
            or long_fur.get("absolute_long_issue") is True
        ):
            return rejected("sample_strong_surface_evidence_present")

        if blocking:
            if is_initial_radial_companion:
                radial_companion_positions.append(position)
            else:
                if (
                    blocking != {target_issue}
                    or action_reference_blocking != {action_reference_issue}
                    or target_issue not in texture_relative
                    or (
                        texture_relative
                        - {target_issue}
                        - POSE_RELATIVE_OBSERVATIONS
                    )
                    or long_fur.get("relative_issue_present") is not True
                    or long_fur.get("strong_prop_excluded_expansion") is not True
                    or long_fur.get("blocking") is not True
                ):
                    return rejected("sample_blocking_issue_contract_mismatch")
                blocking_positions.append(position)
        elif frame.get("passed") is not True:
            return rejected("sample_failed_without_supported_blocking_issue")

        candidate_texture = (
            texture_evaluation.get("candidate", {})
            .get("metrics", {})
            .get("texture")
        )
        baseline_texture = (
            texture_evaluation.get("baseline", {})
            .get("metrics", {})
            .get("texture")
        )
        thresholds = texture_evaluation.get("thresholds")
        if not all(
            isinstance(item, dict)
            for item in (candidate_texture, baseline_texture, thresholds)
        ):
            return rejected("sample_texture_metrics_missing")
        try:
            row = {
                "frame_index": int(frame["frame_index"]),
                "candidate_long": float(
                    candidate_texture["coherent_long_line_energy"]
                ),
                "baseline_long": float(
                    baseline_texture["coherent_long_line_energy"]
                ),
                "base_ratio_min": float(
                    thresholds["long_line_energy_ratio_max"]
                ),
                "base_delta_min": float(thresholds["texture_delta_min"]),
                "absolute_max": float(
                    thresholds["coherent_long_line_energy_max"]
                ),
                "relative_present": target_issue in texture_relative,
                "strong": (
                    long_fur.get("strong_prop_excluded_expansion") is True
                ),
            }
        except (KeyError, TypeError, ValueError) as exc:
            return rejected(
                "sample_texture_metric_schema_invalid",
                metric_error=type(exc).__name__,
            )
        numeric_values = (
            row["candidate_long"],
            row["baseline_long"],
            row["base_ratio_min"],
            row["base_delta_min"],
            row["absolute_max"],
        )
        if (
            not all(np.isfinite(float(value)) for value in numeric_values)
            or row["candidate_long"] < 0.0
            or row["baseline_long"] <= 0.0
            or row["base_ratio_min"] <= 1.0
            or row["base_delta_min"] < 0.0
            or row["absolute_max"] <= 0.0
        ):
            return rejected("sample_texture_metrics_out_of_range")
        rows.append(row)

    if len(blocking_positions) != 1:
        return rejected(
            "blocking_contract_is_not_single_sample",
            blocking_positions=blocking_positions,
            radial_companion_positions=radial_companion_positions,
        )
    if radial_companion_positions not in ([], [0]):
        return rejected(
            "initial_radial_companion_contract_mismatch",
            radial_companion_positions=radial_companion_positions,
        )
    blocking_position = blocking_positions[0]
    if blocking_position in {0, len(rows) - 1}:
        return rejected("blocking_sample_is_not_interior")

    stable_keys = (
        "baseline_long",
        "base_ratio_min",
        "base_delta_min",
        "absolute_max",
    )
    if any(
        max(float(row[key]) for row in rows)
        - min(float(row[key]) for row in rows)
        > 1e-9
        for key in stable_keys
    ):
        return rejected("surface_reference_or_thresholds_changed")

    peak = rows[blocking_position]
    candidate_values = [float(row["candidate_long"]) for row in rows]
    if float(peak["candidate_long"]) < max(candidate_values) - 1e-9:
        return rejected("blocking_sample_is_not_temporal_peak")
    adjacent_positions = [blocking_position - 1, blocking_position + 1]
    supporting_positions = [
        position
        for position in adjacent_positions
        if rows[position]["relative_present"]
        and not rows[position]["strong"]
        and float(rows[position]["candidate_long"])
        <= float(peak["candidate_long"])
    ]
    if not supporting_positions:
        return rejected("adjacent_prop_excluded_observation_missing")
    support_position = max(
        supporting_positions,
        key=lambda position: float(rows[position]["candidate_long"]),
    )
    support = rows[support_position]
    peak_energy = float(peak["candidate_long"])
    support_energy = float(support["candidate_long"])
    baseline_energy = float(peak["baseline_long"])
    base_ratio_min = float(peak["base_ratio_min"])
    base_delta_min = float(peak["base_delta_min"])
    absolute_max = float(peak["absolute_max"])
    strong_ratio_min = base_ratio_min + FINAL_VIDEO_LONG_LINE_RATIO_MARGIN
    strong_delta_min = (
        base_delta_min * FINAL_VIDEO_LONG_LINE_DELTA_MULTIPLIER
    )
    neighbor_ratio = peak_energy / support_energy
    neighbor_delta = peak_energy - support_energy
    peak_ratio = peak_energy / baseline_energy
    peak_delta = peak_energy - baseline_energy
    delta_margin_ceiling = (
        baseline_energy + strong_delta_min + base_delta_min
    )
    evidence = {
        "blocking_frame_indices": [int(peak["frame_index"])],
        "blocking_frame_unmasked_corroborated": bool(
            frame_results[blocking_position]["decision"]
            ["long_fur_relative_evidence"]
            .get("unmasked_corroborated") is True
        ),
        "companion_initial_radial_frame_indices": [
            int(frame_results[position]["frame_index"])
            for position in radial_companion_positions
        ],
        "supporting_frame_indices": [
            int(rows[position]["frame_index"])
            for position in supporting_positions
        ],
        "selected_supporting_frame_index": int(support["frame_index"]),
        "candidate_long_line_energy": round(peak_energy, 8),
        "supporting_long_line_energy": round(support_energy, 8),
        "validated_reference_long_line_energy": round(baseline_energy, 8),
        "candidate_to_support_ratio": round(neighbor_ratio, 8),
        "candidate_to_support_delta": round(neighbor_delta, 8),
        "original_neighbor_ratio_max": round(base_ratio_min, 8),
        "original_neighbor_delta_min": round(base_delta_min, 8),
        "candidate_to_reference_ratio": round(peak_ratio, 8),
        "candidate_to_reference_delta": round(peak_delta, 8),
        "strong_reference_ratio_min": round(strong_ratio_min, 8),
        "strong_reference_delta_min": round(strong_delta_min, 8),
        "existing_delta_margin_ceiling": round(delta_margin_ceiling, 8),
        "absolute_long_line_energy_max": round(absolute_max, 8),
    }
    if (
        neighbor_ratio > base_ratio_min
        and neighbor_delta > base_delta_min
    ):
        return rejected("isolated_peak_remains_strong_against_neighbor", **evidence)
    if (
        peak_energy > delta_margin_ceiling
        or max(candidate_values) > absolute_max
    ):
        return rejected(
            "isolated_peak_exceeds_existing_delta_margin_or_absolute_ceiling",
            **evidence,
        )

    frame = frame_results[blocking_position]
    decision = frame["decision"]
    long_fur = decision["long_fur_relative_evidence"]
    long_fur["blocking_before_temporal_corroboration"] = True
    long_fur["blocking"] = False
    long_fur["blocking_issue"] = None
    long_fur["isolated_eating_pose_phase_observation"] = True
    decision["texture_relative_blocking"] = [
        issue
        for issue in decision.get("texture_relative_blocking", [])
        if issue != target_issue
    ]
    decision["blocking_issues"] = [
        issue
        for issue in decision.get("blocking_issues", [])
        if issue != target_issue
    ]
    decision["blocking_issues_action_reference_scope"] = [
        issue
        for issue in decision.get(
            "blocking_issues_action_reference_scope",
            [],
        )
        if issue != action_reference_issue
    ]
    decision["eating_isolated_long_fur_temporal_corroboration"] = {
        "applied": True,
        "policy": report["policy"],
        "supporting_frame_index": int(support["frame_index"]),
        "existing_delta_margin_ceiling": round(delta_margin_ceiling, 8),
    }
    decision["passed"] = not decision["blocking_issues"]
    frame["passed"] = decision["passed"]
    report.update(
        applied=True,
        reason=(
            "isolated_strong_long_line_peak_corroborated_by_adjacent_pose_phase"
        ),
        downgraded_sample_indices=[int(peak["frame_index"])],
        **evidence,
    )
    return report


def _apply_low_signal_action_reference_texture_floor(
    frame_results: list[dict[str, Any]],
    *,
    action: str,
    reference_authority: dict[str, Any],
) -> dict[str, Any]:
    """Normalize a proven grooming repair reference with too little signal.

    The masked Pet3D repair can occasionally produce a hash-bound, otherwise
    valid grooming keyframe whose texture denominator is much lower than the
    decoded SCAIL frames.  This recovery remains fail-closed: every sampled
    frame must show the same two relative findings, stay below absolute-derived
    texture ceilings, remain temporally stable, and have no contour, radial,
    baseline, segmentation, or unknown issue.
    """

    target_issues = {
        "coarse_strokes_expanded_vs_profile",
        "long_fur_energy_expanded_vs_profile",
    }
    action_reference_issues = {
        _action_reference_issue_name(issue) for issue in target_issues
    }
    report: dict[str, Any] = {
        "policy": (
            "grooming_masked_pet3d_low_signal_reference_texture_floor_v2"
        ),
        "action": action,
        "minimum_samples": FINAL_VIDEO_LOW_SIGNAL_REFERENCE_MIN_SAMPLES,
        "temporal_spread_ratio_max": (
            FINAL_VIDEO_LOW_SIGNAL_REFERENCE_TEMPORAL_SPREAD_MAX
        ),
        "required_raw_relative_issues": sorted(target_issues),
        "requires_coarse_blocking_on_all_sampled_frames": True,
        "minimum_long_blocking_fraction": (
            FINAL_VIDEO_LOW_SIGNAL_REFERENCE_MIN_LONG_BLOCKING_FRACTION
        ),
        "allows_remaining_long_samples_as_nonblocking_observations": True,
        "requires_verified_masked_pet3d_composite_recovery": True,
        "applied": False,
        "downgraded_sample_indices": [],
    }

    def rejected(reason: str, **details: Any) -> dict[str, Any]:
        report["reason"] = reason
        report.update(details)
        return report

    if action != "grooming":
        return rejected("action_not_grooming")
    composite = reference_authority.get(
        "grooming_masked_pet3d_composite_recovery"
    )
    if not isinstance(composite, dict) or composite.get("validated") is not True:
        return rejected("masked_pet3d_composite_recovery_not_verified")
    if len(frame_results) < FINAL_VIDEO_LOW_SIGNAL_REFERENCE_MIN_SAMPLES:
        return rejected("insufficient_sample_count")

    rows: list[dict[str, float | int]] = []
    long_blocking_frame_indices: list[int] = []
    long_observation_frame_indices: list[int] = []
    coarse_issue = "coarse_strokes_expanded_vs_profile"
    long_issue = "long_fur_energy_expanded_vs_profile"
    allowed_absolute = {
        "brush_texture_coherence",
        "coarse_fur_stroke_energy",
    }
    for frame in frame_results:
        decision = frame.get("decision")
        texture_evaluation = frame.get("texture_evaluation")
        contour_evaluation = frame.get("evaluation")
        if not all(
            isinstance(item, dict)
            for item in (decision, texture_evaluation, contour_evaluation)
        ):
            return rejected("sample_evaluation_contract_missing")
        if frame.get("error") is not None:
            return rejected("sample_evaluation_error_present")
        blocking = set(decision.get("blocking_issues") or [])
        if coarse_issue not in blocking or blocking - target_issues:
            return rejected(
                "sample_blocking_issue_contract_mismatch",
                blocking_issues=sorted(blocking),
            )
        expected_action_reference_blocking = {
            _action_reference_issue_name(issue) for issue in blocking
        }
        if set(
            decision.get("blocking_issues_action_reference_scope") or []
        ) != expected_action_reference_blocking:
            return rejected("sample_action_reference_issue_contract_mismatch")
        try:
            texture_relative = set(
                _require_evaluation_list(texture_evaluation, "relative_issues")
            )
            contour_relative = set(
                _require_evaluation_list(contour_evaluation, "relative_issues")
            )
            texture_absolute = set(
                _require_evaluation_list(
                    texture_evaluation,
                    "candidate_absolute_issues",
                )
            )
            contour_absolute = set(
                _require_evaluation_list(
                    contour_evaluation,
                    "candidate_absolute_issues",
                )
            )
            texture_baseline_issues = _require_evaluation_list(
                texture_evaluation,
                "baseline_issues",
            )
            contour_baseline_issues = _require_evaluation_list(
                contour_evaluation,
                "baseline_issues",
            )
        except RuntimeError:
            return rejected("sample_issue_schema_invalid")
        if texture_relative != target_issues or contour_relative != target_issues:
            return rejected("sample_relative_issue_contract_mismatch")
        if (
            texture_absolute - allowed_absolute
            or contour_absolute - allowed_absolute
            or texture_baseline_issues
            or contour_baseline_issues
        ):
            return rejected("sample_absolute_or_baseline_contract_failed")
        if (
            decision.get("contour_relative_blocking")
            or decision.get("unknown_relative_blocking")
        ):
            return rejected("sample_contour_or_unknown_issue_present")
        radial = decision.get("radial_relative_evidence")
        coherent = decision.get("coherent_brush_relative_evidence")
        long_fur = decision.get("long_fur_relative_evidence")
        if not all(isinstance(item, dict) for item in (radial, coherent, long_fur)):
            return rejected("sample_relative_evidence_contract_missing")
        if (
            radial.get("blocking") is True
            or coherent.get("blocking") is True
            or long_fur.get("contour_corroborated") is True
            or long_fur.get("absolute_long_issue") is True
        ):
            return rejected("sample_strong_surface_evidence_present")
        if long_issue in blocking:
            if long_fur.get("blocking") is not True:
                return rejected("sample_long_blocking_evidence_mismatch")
            long_blocking_frame_indices.append(int(frame["frame_index"]))
        else:
            if (
                long_fur.get("relative_issue_present") is not True
                or long_fur.get("blocking") is True
                or long_fur.get("strong_prop_excluded_expansion") is True
            ):
                return rejected("sample_long_observation_evidence_mismatch")
            long_observation_frame_indices.append(int(frame["frame_index"]))
        if not (
            _evaluation_candidate_supported(texture_evaluation)
            and _evaluation_baseline_supported(texture_evaluation)
            and _evaluation_candidate_supported(contour_evaluation)
            and _evaluation_baseline_supported(contour_evaluation)
        ):
            return rejected("sample_surface_contract_unsupported")
        candidate_texture = (
            texture_evaluation.get("candidate", {})
            .get("metrics", {})
            .get("texture")
        )
        baseline_texture = (
            texture_evaluation.get("baseline", {})
            .get("metrics", {})
            .get("texture")
        )
        thresholds = texture_evaluation.get("thresholds")
        if not all(
            isinstance(item, dict)
            for item in (candidate_texture, baseline_texture, thresholds)
        ):
            return rejected("sample_texture_metrics_missing")
        try:
            row = {
                "frame_index": int(frame["frame_index"]),
                "candidate_coarse": float(
                    candidate_texture["coarse_gradient_p90"]
                ),
                "candidate_long": float(
                    candidate_texture["coherent_long_line_energy"]
                ),
                "baseline_coarse": float(
                    baseline_texture["coarse_gradient_p90"]
                ),
                "baseline_long": float(
                    baseline_texture["coherent_long_line_energy"]
                ),
                "signal_floor": float(
                    thresholds["coherence_gradient_p90_min"]
                ),
                "coarse_ratio_max": float(
                    thresholds["coarse_gradient_ratio_max"]
                ),
                "long_ratio_max": float(
                    thresholds["long_line_energy_ratio_max"]
                ),
            }
        except (KeyError, TypeError, ValueError) as exc:
            return rejected(
                "sample_texture_metric_schema_invalid",
                metric_error=type(exc).__name__,
            )
        values = tuple(float(value) for key, value in row.items() if key != "frame_index")
        if (
            not all(np.isfinite(value) for value in values)
            or row["candidate_coarse"] <= 0.0
            or row["candidate_long"] <= 0.0
            or row["baseline_coarse"] <= 0.0
            or row["baseline_long"] <= 0.0
            or row["signal_floor"] <= 0.0
            or row["coarse_ratio_max"] <= 1.0
            or row["long_ratio_max"] <= 1.0
        ):
            return rejected("sample_texture_metrics_out_of_range")
        rows.append(row)

    long_blocking_fraction = len(long_blocking_frame_indices) / len(rows)
    if long_blocking_fraction < (
        FINAL_VIDEO_LOW_SIGNAL_REFERENCE_MIN_LONG_BLOCKING_FRACTION
    ):
        return rejected(
            "insufficient_long_blocking_fraction",
            long_blocking_frame_indices=long_blocking_frame_indices,
            long_observation_frame_indices=long_observation_frame_indices,
            long_blocking_fraction=round(long_blocking_fraction, 8),
        )

    stable_keys = (
        "baseline_coarse",
        "baseline_long",
        "signal_floor",
        "coarse_ratio_max",
        "long_ratio_max",
    )
    if any(
        max(float(row[key]) for row in rows)
        - min(float(row[key]) for row in rows)
        > 1e-9
        for key in stable_keys
    ):
        return rejected("reference_or_threshold_metrics_changed_across_samples")
    baseline_coarse = float(rows[0]["baseline_coarse"])
    baseline_long = float(rows[0]["baseline_long"])
    signal_floor = float(rows[0]["signal_floor"])
    coarse_limit = signal_floor * float(rows[0]["coarse_ratio_max"])
    long_limit = signal_floor * (
        float(rows[0]["long_ratio_max"])
        + FINAL_VIDEO_LONG_LINE_RATIO_MARGIN
    )
    candidate_coarse_values = [float(row["candidate_coarse"]) for row in rows]
    candidate_long_values = [float(row["candidate_long"]) for row in rows]
    coarse_spread = max(candidate_coarse_values) / min(candidate_coarse_values)
    long_spread = max(candidate_long_values) / min(candidate_long_values)
    evidence = {
        "coarse_blocking_frame_indices": [
            int(row["frame_index"]) for row in rows
        ],
        "long_blocking_frame_indices": long_blocking_frame_indices,
        "long_observation_frame_indices": long_observation_frame_indices,
        "long_blocking_fraction": round(long_blocking_fraction, 8),
        "baseline_coarse_gradient_p90": round(baseline_coarse, 8),
        "baseline_coherent_long_line_energy": round(baseline_long, 8),
        "effective_signal_floor": round(signal_floor, 8),
        "candidate_coarse_gradient_p90_min": round(
            min(candidate_coarse_values), 8
        ),
        "candidate_coarse_gradient_p90_max": round(
            max(candidate_coarse_values), 8
        ),
        "candidate_coherent_long_line_energy_min": round(
            min(candidate_long_values), 8
        ),
        "candidate_coherent_long_line_energy_max": round(
            max(candidate_long_values), 8
        ),
        "candidate_coarse_ceiling": round(coarse_limit, 8),
        "candidate_long_ceiling": round(long_limit, 8),
        "candidate_coarse_temporal_spread_ratio": round(coarse_spread, 8),
        "candidate_long_temporal_spread_ratio": round(long_spread, 8),
    }
    if baseline_coarse >= signal_floor or baseline_long >= signal_floor:
        return rejected("validated_action_reference_not_low_signal", **evidence)
    if (
        max(candidate_coarse_values) > coarse_limit
        or max(candidate_long_values) > long_limit
    ):
        return rejected("candidate_texture_exceeds_floor_normalized_ceiling", **evidence)
    if (
        coarse_spread > FINAL_VIDEO_LOW_SIGNAL_REFERENCE_TEMPORAL_SPREAD_MAX
        or long_spread > FINAL_VIDEO_LOW_SIGNAL_REFERENCE_TEMPORAL_SPREAD_MAX
    ):
        return rejected("candidate_texture_not_temporally_stable", **evidence)

    downgraded_indices: list[int] = []
    for frame in frame_results:
        decision = frame["decision"]
        frame_index = int(frame["frame_index"])
        downgraded_indices.append(frame_index)
        decision["texture_relative_blocking"] = [
            issue
            for issue in decision.get("texture_relative_blocking", [])
            if issue not in target_issues
        ]
        decision["blocking_issues"] = [
            issue
            for issue in decision.get("blocking_issues", [])
            if issue not in target_issues
        ]
        decision["blocking_issues_action_reference_scope"] = [
            issue
            for issue in decision.get(
                "blocking_issues_action_reference_scope",
                [],
            )
            if issue not in action_reference_issues
        ]
        long_fur = decision["long_fur_relative_evidence"]
        long_fur["blocking_before_low_signal_reference_floor"] = bool(
            long_fur.get("blocking") is True
        )
        long_fur["blocking"] = False
        long_fur["blocking_issue"] = None
        long_fur["low_signal_reference_floor_observation"] = True
        decision["texture_reference_floor_normalized_observations"] = sorted(
            target_issues
        )
        decision["low_signal_action_reference_texture_floor"] = {
            "applied": True,
            "policy": report["policy"],
            "effective_signal_floor": round(signal_floor, 8),
            "candidate_coarse_ceiling": round(coarse_limit, 8),
            "candidate_long_ceiling": round(long_limit, 8),
        }
        decision["passed"] = not decision["blocking_issues"]
        frame["passed"] = decision["passed"]

    report.update(
        applied=True,
        reason="full_sample_raw_pair_low_signal_reference_normalized",
        downgraded_sample_indices=downgraded_indices,
        composite_recovery_policy=composite.get("policy"),
        **evidence,
    )
    return report


def _apply_sparse_coarse_low_signal_action_reference_floor(
    frame_results: list[dict[str, Any]],
    *,
    action: str,
    reference_authority: dict[str, Any],
) -> dict[str, Any]:
    """Normalize sparse coarse-only hits against a proven low-signal repair.

    This is separate from the full-sample coarse+long recovery above.  It only
    handles a verified grooming Pet3D composite where every sample carries the
    same non-blocking long-line observation, while a minority of samples cross
    only the coarse ratio threshold.  The candidate remains bounded by the
    original absolute thresholds, one existing texture-delta margin, and strict
    temporal stability.  Radial, coherent, contour, unknown, baseline, or
    contract findings keep the original failure.
    """

    coarse_issue = "coarse_strokes_expanded_vs_profile"
    long_issue = "long_fur_energy_expanded_vs_profile"
    coarse_action_reference_issue = _action_reference_issue_name(coarse_issue)
    report: dict[str, Any] = {
        "policy": (
            "grooming_masked_pet3d_sparse_coarse_low_signal_reference_"
            "floor_v1"
        ),
        "action": action,
        "minimum_samples": FINAL_VIDEO_SPARSE_COARSE_LOW_SIGNAL_MIN_SAMPLES,
        "minimum_blocking_samples": (
            FINAL_VIDEO_SPARSE_COARSE_LOW_SIGNAL_MIN_BLOCKING_SAMPLES
        ),
        "maximum_blocking_fraction": (
            FINAL_VIDEO_SPARSE_COARSE_LOW_SIGNAL_MAX_BLOCKING_FRACTION
        ),
        "temporal_spread_ratio_max": (
            FINAL_VIDEO_LOW_SIGNAL_REFERENCE_TEMPORAL_SPREAD_MAX
        ),
        "requires_all_sample_long_line_observation": True,
        "requires_only_sparse_coarse_blocking": True,
        "requires_verified_masked_pet3d_composite_recovery": True,
        "coarse_ceiling_uses_one_existing_texture_delta_margin": True,
        "preserves_absolute_contour_radial_coherent_and_unknown_fail_closed": True,
        "applied": False,
        "downgraded_sample_indices": [],
    }

    def rejected(reason: str, **details: Any) -> dict[str, Any]:
        report["reason"] = reason
        report.update(details)
        return report

    if action != "grooming":
        return rejected("action_not_grooming")
    composite = reference_authority.get(
        "grooming_masked_pet3d_composite_recovery"
    )
    if not isinstance(composite, dict) or composite.get("validated") is not True:
        return rejected("masked_pet3d_composite_recovery_not_verified")
    if len(frame_results) < FINAL_VIDEO_SPARSE_COARSE_LOW_SIGNAL_MIN_SAMPLES:
        return rejected("insufficient_sample_count")

    blocking_positions = [
        position
        for position, frame in enumerate(frame_results)
        if frame.get("passed") is not True
    ]
    blocking_count = len(blocking_positions)
    blocking_fraction = blocking_count / len(frame_results)
    if blocking_count < FINAL_VIDEO_SPARSE_COARSE_LOW_SIGNAL_MIN_BLOCKING_SAMPLES:
        return rejected(
            "insufficient_sparse_coarse_blocking_samples",
            blocking_positions=blocking_positions,
        )
    if blocking_fraction > (
        FINAL_VIDEO_SPARSE_COARSE_LOW_SIGNAL_MAX_BLOCKING_FRACTION
    ):
        return rejected(
            "coarse_blocking_not_sparse",
            blocking_positions=blocking_positions,
            blocking_fraction=round(blocking_fraction, 8),
        )

    rows: list[dict[str, float | int]] = []
    allowed_absolute = {
        "brush_texture_coherence",
        "coarse_fur_stroke_energy",
    }
    for position, frame in enumerate(frame_results):
        decision = frame.get("decision")
        texture_evaluation = frame.get("texture_evaluation")
        contour_evaluation = frame.get("evaluation")
        if not all(
            isinstance(item, dict)
            for item in (decision, texture_evaluation, contour_evaluation)
        ):
            return rejected("sample_evaluation_contract_missing")
        if frame.get("error") is not None:
            return rejected("sample_evaluation_error_present")
        is_blocking = position in blocking_positions
        expected_blocking = {coarse_issue} if is_blocking else set()
        expected_action_scope = (
            {coarse_action_reference_issue} if is_blocking else set()
        )
        if set(decision.get("blocking_issues") or []) != expected_blocking:
            return rejected(
                "sample_sparse_coarse_blocking_contract_mismatch",
                frame_index=frame.get("frame_index"),
                blocking_issues=decision.get("blocking_issues") or [],
            )
        if set(
            decision.get("blocking_issues_action_reference_scope") or []
        ) != expected_action_scope:
            return rejected("sample_action_reference_issue_contract_mismatch")
        if set(decision.get("texture_relative_blocking") or []) != (
            expected_blocking
        ):
            return rejected("sample_texture_blocking_contract_mismatch")
        if (
            decision.get("contour_relative_blocking")
            or decision.get("unknown_relative_blocking")
        ):
            return rejected("sample_contour_or_unknown_issue_present")

        try:
            texture_relative = set(
                _require_evaluation_list(texture_evaluation, "relative_issues")
            )
            contour_relative = set(
                _require_evaluation_list(contour_evaluation, "relative_issues")
            )
            texture_absolute = set(
                _require_evaluation_list(
                    texture_evaluation,
                    "candidate_absolute_issues",
                )
            )
            contour_absolute = set(
                _require_evaluation_list(
                    contour_evaluation,
                    "candidate_absolute_issues",
                )
            )
            texture_baseline_issues = _require_evaluation_list(
                texture_evaluation,
                "baseline_issues",
            )
            contour_baseline_issues = _require_evaluation_list(
                contour_evaluation,
                "baseline_issues",
            )
        except RuntimeError:
            return rejected("sample_issue_schema_invalid")

        expected_surface = {long_issue} | (
            {coarse_issue} if is_blocking else set()
        )
        texture_surface = texture_relative & SURFACE_RELATIVE_ISSUES
        contour_surface = contour_relative & SURFACE_RELATIVE_ISSUES
        texture_extra = texture_relative - expected_surface
        contour_extra = contour_relative - expected_surface
        if (
            texture_surface != expected_surface
            or contour_surface != expected_surface
        ):
            return rejected(
                "sample_surface_relative_issue_contract_mismatch",
                frame_index=frame.get("frame_index"),
            )
        if (
            texture_extra - POSE_RELATIVE_OBSERVATIONS
            or contour_extra - POSE_RELATIVE_OBSERVATIONS
        ):
            return rejected(
                "sample_non_pose_relative_observation_present",
                frame_index=frame.get("frame_index"),
            )
        pose_observations = set(decision.get("pose_relative_observations") or [])
        if not (texture_extra | contour_extra).issubset(pose_observations):
            return rejected("sample_pose_observation_contract_mismatch")
        if (
            texture_absolute - allowed_absolute
            or contour_absolute - allowed_absolute
            or texture_baseline_issues
            or contour_baseline_issues
        ):
            return rejected("sample_absolute_or_baseline_contract_failed")

        radial = decision.get("radial_relative_evidence")
        coherent = decision.get("coherent_brush_relative_evidence")
        long_fur = decision.get("long_fur_relative_evidence")
        if not all(isinstance(item, dict) for item in (radial, coherent, long_fur)):
            return rejected("sample_relative_evidence_contract_missing")
        if (
            radial.get("blocking") is True
            or radial.get("absolute_radial_issue") is True
            or coherent.get("blocking") is True
            or long_fur.get("relative_issue_present") is not True
            or long_fur.get("blocking") is True
            or long_fur.get("contour_corroborated") is True
            or long_fur.get("absolute_long_issue") is True
            or long_fur.get("strong_prop_excluded_expansion") is True
        ):
            return rejected("sample_strong_surface_evidence_present")
        if not (
            _evaluation_candidate_supported(texture_evaluation)
            and _evaluation_baseline_supported(texture_evaluation)
            and _evaluation_candidate_supported(contour_evaluation)
            and _evaluation_baseline_supported(contour_evaluation)
        ):
            return rejected("sample_surface_contract_unsupported")

        candidate_texture = (
            texture_evaluation.get("candidate", {})
            .get("metrics", {})
            .get("texture")
        )
        baseline_texture = (
            texture_evaluation.get("baseline", {})
            .get("metrics", {})
            .get("texture")
        )
        thresholds = texture_evaluation.get("thresholds")
        if not all(
            isinstance(item, dict)
            for item in (candidate_texture, baseline_texture, thresholds)
        ):
            return rejected("sample_texture_metrics_missing")
        try:
            row = {
                "frame_index": int(frame["frame_index"]),
                "candidate_coarse": float(
                    candidate_texture["coarse_gradient_p90"]
                ),
                "candidate_long": float(
                    candidate_texture["coherent_long_line_energy"]
                ),
                "baseline_coarse": float(
                    baseline_texture["coarse_gradient_p90"]
                ),
                "baseline_long": float(
                    baseline_texture["coherent_long_line_energy"]
                ),
                "signal_floor": float(
                    thresholds["coherence_gradient_p90_min"]
                ),
                "coarse_ratio_max": float(
                    thresholds["coarse_gradient_ratio_max"]
                ),
                "long_ratio_max": float(
                    thresholds["long_line_energy_ratio_max"]
                ),
                "texture_delta_min": float(thresholds["texture_delta_min"]),
                "coarse_absolute_max": float(
                    thresholds["coarse_gradient_p90_max"]
                ),
                "long_absolute_max": float(
                    thresholds["coherent_long_line_energy_max"]
                ),
            }
        except (KeyError, TypeError, ValueError) as exc:
            return rejected(
                "sample_texture_metric_schema_invalid",
                metric_error=type(exc).__name__,
            )
        values = tuple(
            float(value) for key, value in row.items() if key != "frame_index"
        )
        if (
            not all(np.isfinite(value) for value in values)
            or min(
                row["candidate_coarse"],
                row["candidate_long"],
                row["baseline_coarse"],
                row["baseline_long"],
                row["signal_floor"],
                row["coarse_absolute_max"],
                row["long_absolute_max"],
            )
            <= 0.0
            or row["coarse_ratio_max"] <= 1.0
            or row["long_ratio_max"] <= 1.0
            or row["texture_delta_min"] < 0.0
        ):
            return rejected("sample_texture_metrics_out_of_range")
        rows.append(row)

    stable_keys = (
        "baseline_coarse",
        "baseline_long",
        "signal_floor",
        "coarse_ratio_max",
        "long_ratio_max",
        "texture_delta_min",
        "coarse_absolute_max",
        "long_absolute_max",
    )
    if any(
        max(float(row[key]) for row in rows)
        - min(float(row[key]) for row in rows)
        > 1e-9
        for key in stable_keys
    ):
        return rejected("reference_or_threshold_metrics_changed_across_samples")

    baseline_coarse = float(rows[0]["baseline_coarse"])
    baseline_long = float(rows[0]["baseline_long"])
    signal_floor = float(rows[0]["signal_floor"])
    texture_delta_min = float(rows[0]["texture_delta_min"])
    coarse_limit = min(
        float(rows[0]["coarse_absolute_max"]),
        signal_floor * float(rows[0]["coarse_ratio_max"])
        + texture_delta_min,
    )
    long_limit = min(
        float(rows[0]["long_absolute_max"]),
        signal_floor
        * (
            float(rows[0]["long_ratio_max"])
            + FINAL_VIDEO_LONG_LINE_RATIO_MARGIN
        ),
    )
    candidate_coarse_values = [float(row["candidate_coarse"]) for row in rows]
    candidate_long_values = [float(row["candidate_long"]) for row in rows]
    coarse_spread = max(candidate_coarse_values) / min(candidate_coarse_values)
    long_spread = max(candidate_long_values) / min(candidate_long_values)
    evidence = {
        "blocking_frame_indices": [
            int(frame_results[position]["frame_index"])
            for position in blocking_positions
        ],
        "blocking_sample_count": blocking_count,
        "blocking_sample_fraction": round(blocking_fraction, 8),
        "baseline_coarse_gradient_p90": round(baseline_coarse, 8),
        "baseline_coherent_long_line_energy": round(baseline_long, 8),
        "effective_signal_floor": round(signal_floor, 8),
        "coarse_ceiling_texture_delta_margin": round(texture_delta_min, 8),
        "candidate_coarse_gradient_p90_min": round(
            min(candidate_coarse_values),
            8,
        ),
        "candidate_coarse_gradient_p90_max": round(
            max(candidate_coarse_values),
            8,
        ),
        "candidate_coherent_long_line_energy_min": round(
            min(candidate_long_values),
            8,
        ),
        "candidate_coherent_long_line_energy_max": round(
            max(candidate_long_values),
            8,
        ),
        "candidate_coarse_ceiling": round(coarse_limit, 8),
        "candidate_long_ceiling": round(long_limit, 8),
        "candidate_coarse_temporal_spread_ratio": round(coarse_spread, 8),
        "candidate_long_temporal_spread_ratio": round(long_spread, 8),
    }
    if baseline_coarse >= signal_floor or baseline_long >= signal_floor:
        return rejected("validated_action_reference_not_low_signal", **evidence)
    if (
        max(candidate_coarse_values) > coarse_limit
        or max(candidate_long_values) > long_limit
    ):
        return rejected(
            "candidate_texture_exceeds_sparse_floor_normalized_ceiling",
            **evidence,
        )
    if (
        coarse_spread > FINAL_VIDEO_LOW_SIGNAL_REFERENCE_TEMPORAL_SPREAD_MAX
        or long_spread > FINAL_VIDEO_LOW_SIGNAL_REFERENCE_TEMPORAL_SPREAD_MAX
    ):
        return rejected("candidate_texture_not_temporally_stable", **evidence)

    downgraded_indices: list[int] = []
    for position, frame in enumerate(frame_results):
        decision = frame["decision"]
        is_blocking = position in blocking_positions
        frame_index = int(frame["frame_index"])
        if is_blocking:
            downgraded_indices.append(frame_index)
        decision["texture_relative_blocking"] = [
            issue
            for issue in decision.get("texture_relative_blocking", [])
            if issue != coarse_issue
        ]
        decision["blocking_issues"] = [
            issue
            for issue in decision.get("blocking_issues", [])
            if issue != coarse_issue
        ]
        decision["blocking_issues_action_reference_scope"] = [
            issue
            for issue in decision.get(
                "blocking_issues_action_reference_scope",
                [],
            )
            if issue != coarse_action_reference_issue
        ]
        decision["texture_reference_floor_normalized_observations"] = sorted(
            {long_issue} | ({coarse_issue} if is_blocking else set())
        )
        decision["sparse_coarse_low_signal_action_reference_floor"] = {
            "applied": True,
            "policy": report["policy"],
            "effective_signal_floor": round(signal_floor, 8),
            "candidate_coarse_ceiling": round(coarse_limit, 8),
            "candidate_long_ceiling": round(long_limit, 8),
            "coarse_blocking_before_floor": is_blocking,
        }
        decision["passed"] = not decision["blocking_issues"]
        frame["passed"] = decision["passed"]

    report.update(
        applied=True,
        reason="sparse_coarse_with_full_long_low_signal_reference_normalized",
        downgraded_sample_indices=downgraded_indices,
        composite_recovery_policy=composite.get("policy"),
        **evidence,
    )
    return report


def _apply_direct_qwen_sparse_coarse_temporal_corroboration(
    frame_results: list[dict[str, Any]],
    *,
    action: str,
    reference_authority: dict[str, Any],
) -> dict[str, Any]:
    """Normalize sparse coarse-only hits against a direct Qwen reference.

    A directly accepted grooming reference does not carry the Pet3D composite
    recovery proof used by the low-signal policies above.  It can nevertheless
    sit just over the coarse relative threshold in a minority of SCAIL frames
    when the grooming motion changes local foreshortening.  This policy is
    deliberately separate and only applies when the exact hash-bound Qwen
    reference is the authority, the reference coarse signal is not low, every
    blocking sample contains the same coarse+long dual-surface observation,
    long-line evidence remains non-strong, and the complete sample set stays
    below absolute, derived-reference, and temporal-stability ceilings.
    """

    coarse_issue = "coarse_strokes_expanded_vs_profile"
    long_issue = "long_fur_energy_expanded_vs_profile"
    coarse_action_reference_issue = _action_reference_issue_name(coarse_issue)
    report: dict[str, Any] = {
        "policy": "grooming_direct_qwen_sparse_coarse_temporal_floor_v1",
        "action": action,
        "minimum_samples": FINAL_VIDEO_DIRECT_QWEN_SPARSE_COARSE_MIN_SAMPLES,
        "minimum_blocking_samples": (
            FINAL_VIDEO_DIRECT_QWEN_SPARSE_COARSE_MIN_BLOCKING_SAMPLES
        ),
        "maximum_blocking_fraction": (
            FINAL_VIDEO_DIRECT_QWEN_SPARSE_COARSE_MAX_BLOCKING_FRACTION
        ),
        "temporal_spread_ratio_max": (
            FINAL_VIDEO_DIRECT_QWEN_SPARSE_COARSE_TEMPORAL_SPREAD_MAX
        ),
        "requires_direct_hash_bound_qwen_reference": True,
        "requires_reference_without_pet3d_composite_recovery": True,
        "requires_non_low_signal_coarse_reference": True,
        "requires_blocking_coarse_plus_non_strong_long_dual_surface_pair": True,
        "coarse_ceiling_uses_reference_ratio_plus_one_texture_delta": True,
        "long_ceiling_reuses_existing_strong_ratio_and_delta_limits": True,
        "preserves_absolute_contour_radial_coherent_and_unknown_fail_closed": True,
        "applied": False,
        "downgraded_sample_indices": [],
    }

    def rejected(reason: str, **details: Any) -> dict[str, Any]:
        report["reason"] = reason
        report.update(details)
        return report

    if action != "grooming":
        return rejected("action_not_grooming")
    if not (
        reference_authority.get("validated") is True
        and reference_authority.get("authority")
        == "uploaded_qwen_action_reference"
        and reference_authority.get("hash_chain_verified") is True
        and reference_authority.get(
            "profile_pixels_used_for_appearance_validation"
        )
        is False
    ):
        return rejected("direct_qwen_reference_authority_not_verified")
    composite = reference_authority.get(
        "grooming_masked_pet3d_composite_recovery"
    )
    if not isinstance(composite, dict) or not (
        composite.get("present") is False
        and composite.get("validated") is False
    ):
        return rejected("direct_qwen_reference_has_composite_recovery")
    if len(frame_results) < FINAL_VIDEO_DIRECT_QWEN_SPARSE_COARSE_MIN_SAMPLES:
        return rejected("insufficient_sample_count")

    blocking_positions = [
        position
        for position, frame in enumerate(frame_results)
        if frame.get("passed") is not True
    ]
    blocking_count = len(blocking_positions)
    blocking_fraction = blocking_count / len(frame_results)
    if blocking_count < (
        FINAL_VIDEO_DIRECT_QWEN_SPARSE_COARSE_MIN_BLOCKING_SAMPLES
    ):
        return rejected(
            "insufficient_sparse_coarse_blocking_samples",
            blocking_positions=blocking_positions,
        )
    if blocking_fraction > (
        FINAL_VIDEO_DIRECT_QWEN_SPARSE_COARSE_MAX_BLOCKING_FRACTION
    ):
        return rejected(
            "coarse_blocking_not_sparse",
            blocking_positions=blocking_positions,
            blocking_fraction=round(blocking_fraction, 8),
        )

    rows: list[dict[str, float | int]] = []
    long_observation_frame_indices: list[int] = []
    clean_surface_frame_indices: list[int] = []
    allowed_absolute = {
        "brush_texture_coherence",
        "coarse_fur_stroke_energy",
    }
    for position, frame in enumerate(frame_results):
        decision = frame.get("decision")
        texture_evaluation = frame.get("texture_evaluation")
        contour_evaluation = frame.get("evaluation")
        if not all(
            isinstance(item, dict)
            for item in (decision, texture_evaluation, contour_evaluation)
        ):
            return rejected("sample_evaluation_contract_missing")
        if frame.get("error") is not None:
            return rejected("sample_evaluation_error_present")
        is_blocking = position in blocking_positions
        expected_blocking = {coarse_issue} if is_blocking else set()
        expected_action_scope = (
            {coarse_action_reference_issue} if is_blocking else set()
        )
        if set(decision.get("blocking_issues") or []) != expected_blocking:
            return rejected(
                "sample_sparse_coarse_blocking_contract_mismatch",
                frame_index=frame.get("frame_index"),
                blocking_issues=decision.get("blocking_issues") or [],
            )
        if set(
            decision.get("blocking_issues_action_reference_scope") or []
        ) != expected_action_scope:
            return rejected("sample_action_reference_issue_contract_mismatch")
        if set(decision.get("texture_relative_blocking") or []) != (
            expected_blocking
        ):
            return rejected("sample_texture_blocking_contract_mismatch")
        if (
            decision.get("contour_relative_blocking")
            or decision.get("unknown_relative_blocking")
        ):
            return rejected("sample_contour_or_unknown_issue_present")

        try:
            texture_relative = set(
                _require_evaluation_list(texture_evaluation, "relative_issues")
            )
            contour_relative = set(
                _require_evaluation_list(contour_evaluation, "relative_issues")
            )
            texture_absolute = set(
                _require_evaluation_list(
                    texture_evaluation,
                    "candidate_absolute_issues",
                )
            )
            contour_absolute = set(
                _require_evaluation_list(
                    contour_evaluation,
                    "candidate_absolute_issues",
                )
            )
            texture_baseline_issues = _require_evaluation_list(
                texture_evaluation,
                "baseline_issues",
            )
            contour_baseline_issues = _require_evaluation_list(
                contour_evaluation,
                "baseline_issues",
            )
        except RuntimeError:
            return rejected("sample_issue_schema_invalid")

        texture_surface = texture_relative & SURFACE_RELATIVE_ISSUES
        contour_surface = contour_relative & SURFACE_RELATIVE_ISSUES
        if is_blocking:
            expected_surface = {coarse_issue, long_issue}
            if (
                texture_surface != expected_surface
                or contour_surface != expected_surface
            ):
                return rejected(
                    "sample_blocking_surface_pair_contract_mismatch",
                    frame_index=frame.get("frame_index"),
                    texture_surface=sorted(texture_surface),
                    contour_surface=sorted(contour_surface),
                )
        else:
            if (
                texture_surface != contour_surface
                or not texture_surface.issubset({long_issue})
            ):
                return rejected(
                    "sample_nonblocking_surface_contract_mismatch",
                    frame_index=frame.get("frame_index"),
                    texture_surface=sorted(texture_surface),
                    contour_surface=sorted(contour_surface),
                )
            expected_surface = texture_surface
        frame_index = int(frame["frame_index"])
        if long_issue in expected_surface:
            long_observation_frame_indices.append(frame_index)
        else:
            clean_surface_frame_indices.append(frame_index)

        texture_extra = texture_relative - expected_surface
        contour_extra = contour_relative - expected_surface
        if (
            texture_extra - POSE_RELATIVE_OBSERVATIONS
            or contour_extra - POSE_RELATIVE_OBSERVATIONS
        ):
            return rejected(
                "sample_non_pose_relative_observation_present",
                frame_index=frame_index,
            )
        pose_observations = set(decision.get("pose_relative_observations") or [])
        if not (texture_extra | contour_extra).issubset(pose_observations):
            return rejected("sample_pose_observation_contract_mismatch")
        if (
            texture_absolute - allowed_absolute
            or contour_absolute - allowed_absolute
            or texture_baseline_issues
            or contour_baseline_issues
        ):
            return rejected("sample_absolute_or_baseline_contract_failed")

        radial = decision.get("radial_relative_evidence")
        coherent = decision.get("coherent_brush_relative_evidence")
        long_fur = decision.get("long_fur_relative_evidence")
        if not all(isinstance(item, dict) for item in (radial, coherent, long_fur)):
            return rejected("sample_relative_evidence_contract_missing")
        long_observed = long_issue in expected_surface
        if (
            radial.get("blocking") is True
            or radial.get("absolute_radial_issue") is True
            or coherent.get("blocking") is True
            or long_fur.get("blocking") is True
            or long_fur.get("contour_corroborated") is True
            or long_fur.get("absolute_long_issue") is True
            or long_fur.get("strong_prop_excluded_expansion") is True
            or bool(long_fur.get("relative_issue_present")) != long_observed
        ):
            return rejected("sample_strong_surface_evidence_present")
        if not (
            _evaluation_candidate_supported(texture_evaluation)
            and _evaluation_baseline_supported(texture_evaluation)
            and _evaluation_candidate_supported(contour_evaluation)
            and _evaluation_baseline_supported(contour_evaluation)
        ):
            return rejected("sample_surface_contract_unsupported")

        candidate_texture = (
            texture_evaluation.get("candidate", {})
            .get("metrics", {})
            .get("texture")
        )
        baseline_texture = (
            texture_evaluation.get("baseline", {})
            .get("metrics", {})
            .get("texture")
        )
        thresholds = texture_evaluation.get("thresholds")
        if not all(
            isinstance(item, dict)
            for item in (candidate_texture, baseline_texture, thresholds)
        ):
            return rejected("sample_texture_metrics_missing")
        try:
            row = {
                "frame_index": frame_index,
                "candidate_coarse": float(
                    candidate_texture["coarse_gradient_p90"]
                ),
                "candidate_long": float(
                    candidate_texture["coherent_long_line_energy"]
                ),
                "baseline_coarse": float(
                    baseline_texture["coarse_gradient_p90"]
                ),
                "baseline_long": float(
                    baseline_texture["coherent_long_line_energy"]
                ),
                "signal_floor": float(
                    thresholds["coherence_gradient_p90_min"]
                ),
                "coarse_ratio_max": float(
                    thresholds["coarse_gradient_ratio_max"]
                ),
                "long_ratio_max": float(
                    thresholds["long_line_energy_ratio_max"]
                ),
                "texture_delta_min": float(thresholds["texture_delta_min"]),
                "coarse_absolute_max": float(
                    thresholds["coarse_gradient_p90_max"]
                ),
                "long_absolute_max": float(
                    thresholds["coherent_long_line_energy_max"]
                ),
            }
        except (KeyError, TypeError, ValueError) as exc:
            return rejected(
                "sample_texture_metric_schema_invalid",
                metric_error=type(exc).__name__,
            )
        values = tuple(
            float(value) for key, value in row.items() if key != "frame_index"
        )
        if (
            not all(np.isfinite(value) for value in values)
            or min(
                row["candidate_coarse"],
                row["candidate_long"],
                row["baseline_coarse"],
                row["baseline_long"],
                row["signal_floor"],
                row["coarse_absolute_max"],
                row["long_absolute_max"],
            )
            <= 0.0
            or row["coarse_ratio_max"] <= 1.0
            or row["long_ratio_max"] <= 1.0
            or row["texture_delta_min"] < 0.0
        ):
            return rejected("sample_texture_metrics_out_of_range")
        rows.append(row)

    stable_keys = (
        "baseline_coarse",
        "baseline_long",
        "signal_floor",
        "coarse_ratio_max",
        "long_ratio_max",
        "texture_delta_min",
        "coarse_absolute_max",
        "long_absolute_max",
    )
    if any(
        max(float(row[key]) for row in rows)
        - min(float(row[key]) for row in rows)
        > 1e-9
        for key in stable_keys
    ):
        return rejected("reference_or_threshold_metrics_changed_across_samples")

    baseline_coarse = float(rows[0]["baseline_coarse"])
    baseline_long = float(rows[0]["baseline_long"])
    signal_floor = float(rows[0]["signal_floor"])
    texture_delta_min = float(rows[0]["texture_delta_min"])
    coarse_ratio_limit = baseline_coarse * float(rows[0]["coarse_ratio_max"])
    coarse_limit = min(
        float(rows[0]["coarse_absolute_max"]),
        coarse_ratio_limit + texture_delta_min,
    )
    long_ratio_limit = baseline_long * (
        float(rows[0]["long_ratio_max"])
        + FINAL_VIDEO_LONG_LINE_RATIO_MARGIN
    )
    long_delta_limit = baseline_long + (
        texture_delta_min * FINAL_VIDEO_LONG_LINE_DELTA_MULTIPLIER
    )
    long_limit = min(
        float(rows[0]["long_absolute_max"]),
        long_ratio_limit,
        long_delta_limit,
    )
    candidate_coarse_values = [float(row["candidate_coarse"]) for row in rows]
    candidate_long_values = [float(row["candidate_long"]) for row in rows]
    coarse_spread = max(candidate_coarse_values) / min(candidate_coarse_values)
    long_spread = max(candidate_long_values) / min(candidate_long_values)
    evidence = {
        "blocking_frame_indices": [
            int(frame_results[position]["frame_index"])
            for position in blocking_positions
        ],
        "blocking_sample_count": blocking_count,
        "blocking_sample_fraction": round(blocking_fraction, 8),
        "long_observation_frame_indices": long_observation_frame_indices,
        "clean_surface_frame_indices": clean_surface_frame_indices,
        "baseline_coarse_gradient_p90": round(baseline_coarse, 8),
        "baseline_coherent_long_line_energy": round(baseline_long, 8),
        "effective_signal_floor": round(signal_floor, 8),
        "coarse_reference_ratio_limit": round(coarse_ratio_limit, 8),
        "coarse_ceiling_texture_delta_margin": round(texture_delta_min, 8),
        "candidate_coarse_gradient_p90_min": round(
            min(candidate_coarse_values), 8
        ),
        "candidate_coarse_gradient_p90_max": round(
            max(candidate_coarse_values), 8
        ),
        "candidate_coherent_long_line_energy_min": round(
            min(candidate_long_values), 8
        ),
        "candidate_coherent_long_line_energy_max": round(
            max(candidate_long_values), 8
        ),
        "candidate_coarse_ceiling": round(coarse_limit, 8),
        "candidate_long_ratio_ceiling": round(long_ratio_limit, 8),
        "candidate_long_delta_ceiling": round(long_delta_limit, 8),
        "candidate_long_ceiling": round(long_limit, 8),
        "candidate_coarse_temporal_spread_ratio": round(coarse_spread, 8),
        "candidate_long_temporal_spread_ratio": round(long_spread, 8),
    }
    if baseline_coarse < signal_floor:
        return rejected(
            "validated_action_reference_coarse_signal_below_floor",
            **evidence,
        )
    if (
        max(candidate_coarse_values) > coarse_limit
        or max(candidate_long_values) > long_limit
    ):
        return rejected(
            "candidate_texture_exceeds_direct_qwen_temporal_ceiling",
            **evidence,
        )
    if (
        coarse_spread
        > FINAL_VIDEO_DIRECT_QWEN_SPARSE_COARSE_TEMPORAL_SPREAD_MAX
        or long_spread
        > FINAL_VIDEO_DIRECT_QWEN_SPARSE_COARSE_TEMPORAL_SPREAD_MAX
    ):
        return rejected("candidate_texture_not_temporally_stable", **evidence)

    downgraded_indices: list[int] = []
    for position in blocking_positions:
        frame = frame_results[position]
        decision = frame["decision"]
        frame_index = int(frame["frame_index"])
        downgraded_indices.append(frame_index)
        decision["texture_relative_blocking"] = [
            issue
            for issue in decision.get("texture_relative_blocking", [])
            if issue != coarse_issue
        ]
        decision["blocking_issues"] = [
            issue
            for issue in decision.get("blocking_issues", [])
            if issue != coarse_issue
        ]
        decision["blocking_issues_action_reference_scope"] = [
            issue
            for issue in decision.get(
                "blocking_issues_action_reference_scope",
                [],
            )
            if issue != coarse_action_reference_issue
        ]
        decision["texture_reference_floor_normalized_observations"] = [
            coarse_issue,
            long_issue,
        ]
        decision["direct_qwen_sparse_coarse_temporal_corroboration"] = {
            "applied": True,
            "policy": report["policy"],
            "candidate_coarse_ceiling": round(coarse_limit, 8),
            "candidate_long_ceiling": round(long_limit, 8),
            "coarse_blocking_before_corroboration": True,
        }
        decision["passed"] = not decision["blocking_issues"]
        frame["passed"] = decision["passed"]

    report.update(
        applied=True,
        reason="direct_qwen_sparse_coarse_temporal_variation_normalized",
        downgraded_sample_indices=downgraded_indices,
        **evidence,
    )
    return report


def _long_fur_relative_evidence(
    evaluation: dict[str, Any],
    *,
    relative_issues: list[str],
    absolute_issues: list[str],
    unmasked_relative_issues: list[str],
    unmasked_absolute_issues: list[str],
) -> dict[str, Any]:
    """Require contour or strong texture evidence for true fur-length drift."""

    issue = "long_fur_energy_expanded_vs_profile"
    absolute_long_issue = "coherent_long_fur_energy" in absolute_issues
    relative_issue_present = issue in relative_issues
    unmasked_corroborated = issue in unmasked_relative_issues
    contour_corroboration_issues = sorted(
        (
            set(unmasked_relative_issues)
            & {"contour_fringe_expanded_vs_profile"}
        )
        | (
            set(unmasked_absolute_issues)
            & {"excess_contour_fringe", "irregular_expanded_silhouette"}
        )
    )
    contour_corroborated = bool(contour_corroboration_issues)
    if not relative_issue_present:
        return {
            "relative_issue_present": False,
            "absolute_long_issue": absolute_long_issue,
            "unmasked_corroborated": unmasked_corroborated,
            "contour_corroborated": contour_corroborated,
            "contour_corroboration_issues": contour_corroboration_issues,
            "strong_prop_excluded_expansion": False,
            "blocking": False,
            "blocking_issue": None,
            "mask_or_decode_level_observation": False,
            "texture_only_pose_or_decode_observation": False,
        }
    candidate = evaluation.get("candidate")
    baseline = evaluation.get("baseline")
    thresholds = evaluation.get("thresholds")
    if not all(isinstance(item, dict) for item in (candidate, baseline, thresholds)):
        raise RuntimeError("long_fur_relative_evidence_schema_invalid")
    candidate_texture = candidate.get("metrics", {}).get("texture")
    baseline_texture = baseline.get("metrics", {}).get("texture")
    if not isinstance(candidate_texture, dict) or not isinstance(
        baseline_texture,
        dict,
    ):
        raise RuntimeError("long_fur_relative_texture_metrics_missing")
    try:
        candidate_energy = float(
            candidate_texture["coherent_long_line_energy"]
        )
        baseline_energy = float(
            baseline_texture["coherent_long_line_energy"]
        )
        base_ratio_min = float(thresholds["long_line_energy_ratio_max"])
        base_delta_min = float(thresholds["texture_delta_min"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("long_fur_relative_evidence_metrics_invalid") from exc
    values = (
        candidate_energy,
        baseline_energy,
        base_ratio_min,
        base_delta_min,
    )
    if not all(np.isfinite(value) for value in values):
        raise RuntimeError("long_fur_relative_evidence_metrics_non_finite")
    if (
        candidate_energy < 0.0
        or baseline_energy <= 0.0
        or base_ratio_min <= 0.0
        or base_delta_min < 0.0
    ):
        raise RuntimeError("long_fur_relative_evidence_metrics_out_of_range")
    energy_ratio = candidate_energy / baseline_energy
    energy_delta = candidate_energy - baseline_energy
    strong_ratio_min = base_ratio_min + FINAL_VIDEO_LONG_LINE_RATIO_MARGIN
    strong_delta_min = (
        base_delta_min * FINAL_VIDEO_LONG_LINE_DELTA_MULTIPLIER
    )
    strong_prop_excluded_expansion = bool(
        energy_ratio > strong_ratio_min and energy_delta > strong_delta_min
    )
    blocking = bool(contour_corroborated or strong_prop_excluded_expansion)
    texture_only_observation = bool(
        not blocking
        and (
            relative_issue_present
            or absolute_long_issue
            or unmasked_corroborated
        )
    )
    return {
        "relative_issue_present": True,
        "candidate_energy": round(candidate_energy, 8),
        "baseline_energy": round(baseline_energy, 8),
        "energy_ratio": round(energy_ratio, 8),
        "energy_delta": round(energy_delta, 8),
        "base_ratio_min": round(base_ratio_min, 8),
        "base_delta_min": round(base_delta_min, 8),
        "strong_ratio_min": round(strong_ratio_min, 8),
        "strong_delta_min": round(strong_delta_min, 8),
        "absolute_long_issue": absolute_long_issue,
        "unmasked_corroborated": unmasked_corroborated,
        "contour_corroborated": contour_corroborated,
        "contour_corroboration_issues": contour_corroboration_issues,
        "strong_prop_excluded_expansion": strong_prop_excluded_expansion,
        "blocking": blocking,
        "blocking_issue": issue if blocking else None,
        "mask_or_decode_level_observation": texture_only_observation,
        "texture_only_pose_or_decode_observation": texture_only_observation,
    }


def audit_final_video(
    video_path: str | Path,
    *,
    action: str,
    baseline_path: str | Path,
    baseline_sha256: str,
    baseline_provenance: dict[str, Any],
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    rembg_model: str = "u2net",
    evaluator: Callable[..., dict[str, Any]] = evaluate_fur_surface,
    frame_extractor: Callable[[str | Path, int, Path], None] = extract_frame,
    video_probe: Callable[[str | Path], dict[str, Any]] = probe_video,
    background_remover: Callable[..., dict[str, Any]] = remove_background,
    comparison_scope: str = PROFILE_FULL_SCOPE,
    validated_action_reference_path: str | Path | None = None,
    validated_action_reference_sha256: str | None = None,
    validated_action_reference_provenance: dict[str, Any] | None = None,
    pose_identity_mask_video_path: str | Path | None = None,
    pose_identity_mask_video_sha256: str | None = None,
    reference_identity_mask_path: str | Path | None = None,
    reference_identity_mask_sha256: str | None = None,
) -> dict[str, Any]:
    """Audit final frames under the legacy gate or uploaded Qwen authority."""

    if comparison_scope == PROFILE_FULL_SCOPE:
        result = _audit_final_video_profile_full(
            video_path,
            action=action,
            baseline_path=baseline_path,
            baseline_sha256=baseline_sha256,
            baseline_provenance=baseline_provenance,
            sample_count=sample_count,
            rembg_model=rembg_model,
            evaluator=evaluator,
            frame_extractor=frame_extractor,
            video_probe=video_probe,
            background_remover=background_remover,
        )
        legacy_policy = result.setdefault("policy", {})
        legacy_policy["name"] = "final_video_sampled_fur_surface_v1"
        legacy_policy["version"] = "1.0.0"
        legacy_policy["comparison_scope"] = PROFILE_FULL_SCOPE
        legacy_policy["absolute_observations_nonblocking"] = False
        legacy_policy["geometry_relative_nonblocking"] = False
        legacy_policy["implementation_policy"] = POLICY
        legacy_policy["implementation_version"] = VERSION
        return result
    if comparison_scope != VALIDATED_ACTION_SURFACE_SCOPE:
        raise ValueError(f"unsupported_final_video_comparison_scope:{comparison_scope}")
    if action not in SUPPORTED_ACTIONS:
        raise ValueError(f"unsupported_action:{action}")

    resolved_video = Path(video_path).resolve()
    resolved_profile = Path(baseline_path).resolve()
    resolved_reference = Path(str(validated_action_reference_path or "")).resolve()
    resolved_pose_mask = Path(str(pose_identity_mask_video_path or "")).resolve()
    resolved_reference_mask = Path(str(reference_identity_mask_path or "")).resolve()
    required_files = {
        "video": resolved_video,
        "profile_baseline": resolved_profile,
        "validated_action_reference": resolved_reference,
        "pose_identity_mask_video": resolved_pose_mask,
        "reference_identity_mask": resolved_reference_mask,
    }
    for label, path in required_files.items():
        if not path.is_file():
            raise ValueError(f"{label}_not_found:{path}")
    expected_hashes = {
        "profile_baseline": str(baseline_sha256 or "").lower(),
        "validated_action_reference": str(
            validated_action_reference_sha256 or ""
        ).lower(),
        "pose_identity_mask_video": str(
            pose_identity_mask_video_sha256 or ""
        ).lower(),
        "reference_identity_mask": str(
            reference_identity_mask_sha256 or ""
        ).lower(),
    }
    actual_hashes = {
        "profile_baseline": sha256_file(resolved_profile),
        "validated_action_reference": sha256_file(resolved_reference),
        "pose_identity_mask_video": sha256_file(resolved_pose_mask),
        "reference_identity_mask": sha256_file(resolved_reference_mask),
    }
    for label, expected in expected_hashes.items():
        if len(expected) != 64 or actual_hashes[label] != expected:
            raise ValueError(
                f"{label}_sha256_mismatch:expected={expected}:"
                f"actual={actual_hashes[label]}"
            )

    uploaded_reference_authority = _validate_uploaded_reference_provenance(
        validated_action_reference_provenance,
        action=action,
        reference_sha256=actual_hashes["validated_action_reference"],
    )

    metadata = video_probe(resolved_video)
    pose_mask_metadata = video_probe(resolved_pose_mask)
    for key in ("frame_count", "width", "height"):
        if int(pose_mask_metadata.get(key) or 0) != int(metadata.get(key) or 0):
            raise ValueError(
                f"pose_identity_mask_video_{key}_mismatch:"
                f"video={metadata.get(key)}:mask={pose_mask_metadata.get(key)}"
            )
    if abs(float(pose_mask_metadata.get("fps") or 0.0) - float(metadata.get("fps") or 0.0)) > 0.01:
        raise ValueError(
            "pose_identity_mask_video_fps_mismatch:"
            f"video={metadata.get('fps')}:mask={pose_mask_metadata.get('fps')}"
        )
    indices = sample_indices(int(metadata["frame_count"]), sample_count)
    frame_results: list[dict[str, Any]] = []
    reference_texture_metadata: dict[str, Any]

    with tempfile.TemporaryDirectory(prefix="final-video-fur-audit-v22-") as temp_name:
        temp_root = Path(temp_name)
        texture_reference_path = temp_root / "validated_reference_prop_excluded.png"
        reference_texture_metadata = clear_red_prop_from_alpha(
            resolved_reference,
            resolved_reference_mask,
            texture_reference_path,
            action=action,
            mask_background="black",
        )
        reference_texture_mask = reference_texture_metadata.get("mask")
        if (
            not isinstance(reference_texture_mask, dict)
            or reference_texture_mask.get("reliable") is not True
            or reference_texture_metadata.get("source_sha256")
            != actual_hashes["validated_action_reference"]
            or reference_texture_metadata.get("rgb_byte_preserved") is not True
            or reference_texture_metadata.get("alpha_only_decreased") is not True
            or reference_texture_metadata.get("morphology_applied") is not False
        ):
            raise ValueError(
                "validated_action_texture_reference_derivation_unreliable"
            )
        for frame_index in indices:
            raw_path = temp_root / f"frame_{frame_index:05d}_raw.png"
            alpha_path = temp_root / f"frame_{frame_index:05d}_rembg.png"
            identity_mask_path = temp_root / f"frame_{frame_index:05d}_identity_mask.png"
            texture_path = temp_root / f"frame_{frame_index:05d}_prop_excluded.png"
            try:
                frame_extractor(resolved_video, frame_index, raw_path)
                frame_extractor(resolved_pose_mask, frame_index, identity_mask_path)
                identity_mask_quality = validate_identity_mask_frame(
                    identity_mask_path,
                    action=action,
                    background="white",
                )
                if identity_mask_quality["reliable"] is not True:
                    raise RuntimeError(
                        "identity_mask_unreliable:"
                        + ",".join(identity_mask_quality["issues"])
                    )
                if (
                    identity_mask_quality["width"] != int(metadata["width"])
                    or identity_mask_quality["height"] != int(metadata["height"])
                ):
                    raise RuntimeError("identity_mask_frame_dimensions_changed")
                segmentation = background_remover(
                    raw_path,
                    alpha_path,
                    model=rembg_model,
                )
                if (
                    not isinstance(segmentation, dict)
                    or segmentation.get("reliable") is not True
                ):
                    detail = ",".join(
                        str(issue)
                        for issue in (
                            segmentation.get("issues", [])
                            if isinstance(segmentation, dict)
                            else []
                        )
                    )
                    raise RuntimeError(
                        "background_segmentation_unreliable"
                        + (f":{detail}" if detail else "")
                    )
                contour_evaluation = evaluator(
                    alpha_path,
                    resolved_reference,
                    baseline_is_validated=True,
                )
                texture_exclusion = clear_red_prop_from_alpha(
                    alpha_path,
                    identity_mask_path,
                    texture_path,
                    action=action,
                    mask_background="white",
                )
                texture_evaluation = evaluator(
                    texture_path,
                    texture_reference_path,
                    baseline_is_validated=True,
                )
                contour_relative = _require_evaluation_list(
                    contour_evaluation,
                    "relative_issues",
                )
                texture_relative = _require_evaluation_list(
                    texture_evaluation,
                    "relative_issues",
                )
                contour_absolute = _require_evaluation_list(
                    contour_evaluation,
                    "candidate_absolute_issues",
                )
                texture_absolute = _require_evaluation_list(
                    texture_evaluation,
                    "candidate_absolute_issues",
                )
                contour_baseline_issues = _require_evaluation_list(
                    contour_evaluation,
                    "baseline_issues",
                )
                texture_baseline_issues = _require_evaluation_list(
                    texture_evaluation,
                    "baseline_issues",
                )
                recognized = POSE_RELATIVE_OBSERVATIONS | SURFACE_RELATIVE_ISSUES
                unknown_relative = sorted(
                    (set(contour_relative) | set(texture_relative)) - recognized
                )
                contour_blocking = sorted(
                    set(contour_relative)
                    & {"contour_fringe_expanded_vs_profile"}
                )
                radial_relative_evidence = _radial_relative_evidence(
                    texture_evaluation,
                    relative_issues=texture_relative,
                    absolute_issues=texture_absolute,
                )
                coherent_brush_relative_evidence = (
                    _coherent_brush_relative_evidence(
                        texture_evaluation,
                        relative_issues=texture_relative,
                        absolute_issues=texture_absolute,
                    )
                )
                long_fur_relative_evidence = _long_fur_relative_evidence(
                    texture_evaluation,
                    relative_issues=texture_relative,
                    absolute_issues=texture_absolute,
                    unmasked_relative_issues=contour_relative,
                    unmasked_absolute_issues=contour_absolute,
                )
                texture_blocking = sorted(
                    set(texture_relative)
                    & (
                        SURFACE_RELATIVE_ISSUES
                        - {
                            "contour_fringe_expanded_vs_profile",
                            "coherent_brush_texture_expanded_vs_profile",
                            "radial_brush_texture_expanded_vs_profile",
                            "long_fur_energy_expanded_vs_profile",
                        }
                    )
                )
                if coherent_brush_relative_evidence["blocking"]:
                    texture_blocking.append(
                        str(
                            coherent_brush_relative_evidence[
                                "blocking_issue"
                            ]
                        )
                    )
                if radial_relative_evidence["blocking"]:
                    texture_blocking.append(
                        str(radial_relative_evidence["blocking_issue"])
                    )
                if long_fur_relative_evidence["blocking"]:
                    texture_blocking.append(
                        str(long_fur_relative_evidence["blocking_issue"])
                    )
                blocking_issues = list(
                    dict.fromkeys(
                        [*contour_blocking, *texture_blocking, *unknown_relative]
                    )
                )
                contracts_supported = bool(
                    _evaluation_candidate_supported(contour_evaluation)
                    and _evaluation_candidate_supported(texture_evaluation)
                    and _evaluation_baseline_supported(contour_evaluation)
                    and _evaluation_baseline_supported(texture_evaluation)
                    and not contour_baseline_issues
                )
                if not contracts_supported:
                    blocking_issues.append("surface_evaluation_contract_unreliable")
                decision = {
                    "passed": not blocking_issues,
                    "blocking_issues": list(dict.fromkeys(blocking_issues)),
                    "blocking_issues_action_reference_scope": [
                        _action_reference_issue_name(issue)
                        for issue in dict.fromkeys(blocking_issues)
                    ],
                    "contour_relative_blocking": contour_blocking,
                    "texture_relative_blocking": texture_blocking,
                    "coherent_brush_relative_evidence": (
                        coherent_brush_relative_evidence
                    ),
                    "radial_relative_evidence": radial_relative_evidence,
                    "long_fur_relative_evidence": long_fur_relative_evidence,
                    "unknown_relative_blocking": unknown_relative,
                    "pose_relative_observations": sorted(
                        (set(contour_relative) | set(texture_relative))
                        & POSE_RELATIVE_OBSERVATIONS
                    ),
                    "absolute_still_observations": {
                        "unmasked": contour_absolute,
                        "prop_excluded": texture_absolute,
                    },
                    "baseline_absolute_observations": {
                        "unmasked_action_reference": contour_baseline_issues,
                        "hash_bound_prop_excluded_reference": (
                            texture_baseline_issues
                        ),
                    },
                    "unmasked_texture_relative_observations": sorted(
                        set(contour_relative)
                        & (
                            SURFACE_RELATIVE_ISSUES
                            - {"contour_fringe_expanded_vs_profile"}
                        )
                    ),
                    "prop_excluded_contour_relative_observations": sorted(
                        set(texture_relative)
                        & {"contour_fringe_expanded_vs_profile"}
                    ),
                }
                frame_results.append(
                    {
                        "frame_index": frame_index,
                        "passed": decision["passed"],
                        "segmentation": {
                            "method": f"rembg_{rembg_model}",
                            "reliability": segmentation,
                            "identity_mask": identity_mask_quality,
                            "texture_prop_exclusion": texture_exclusion,
                        },
                        "evaluation": contour_evaluation,
                        "texture_evaluation": texture_evaluation,
                        "decision": decision,
                        "error": None,
                    }
                )
            except Exception as exc:  # noqa: BLE001 - every sample fails closed
                frame_results.append(
                    {
                        "frame_index": frame_index,
                        "passed": False,
                        "segmentation": None,
                        "evaluation": None,
                        "texture_evaluation": None,
                        "decision": {
                            "passed": False,
                            "blocking_issues": ["frame_audit_error"],
                        },
                        "error": f"{type(exc).__name__}:{exc}",
                    }
                )

    coherent_brush_temporal_evidence = (
        _apply_coherent_brush_temporal_corroboration(frame_results)
    )
    eating_isolated_long_fur_evidence = (
        _apply_eating_isolated_long_fur_temporal_corroboration(
            frame_results,
            action=action,
        )
    )
    eating_radial_boundary_evidence = (
        _apply_eating_initial_radial_loop_boundary_corroboration(
            frame_results,
            action=action,
        )
    )
    eating_radial_bounded_reference_evidence = (
        _apply_eating_initial_radial_bounded_reference_corroboration(
            frame_results,
            action=action,
        )
    )
    eating_boundary_anchor_evidence = (
        _apply_eating_loop_boundary_decode_anchor(
            frame_results,
            action=action,
        )
    )
    low_signal_reference_evidence = (
        _apply_low_signal_action_reference_texture_floor(
            frame_results,
            action=action,
            reference_authority=uploaded_reference_authority,
        )
    )
    sparse_coarse_low_signal_reference_evidence = (
        _apply_sparse_coarse_low_signal_action_reference_floor(
            frame_results,
            action=action,
            reference_authority=uploaded_reference_authority,
        )
    )
    direct_qwen_sparse_coarse_reference_evidence = (
        _apply_direct_qwen_sparse_coarse_temporal_corroboration(
            frame_results,
            action=action,
            reference_authority=uploaded_reference_authority,
        )
    )
    failed = [item for item in frame_results if not item["passed"]]
    worst = max(
        frame_results,
        key=lambda item: (
            1 if not item["passed"] else 0,
            len(item.get("decision", {}).get("blocking_issues", [])),
            len(
                item.get("decision", {})
                .get("absolute_still_observations", {})
                .get("unmasked", [])
            ),
        ),
    )
    reference_texture_metadata = dict(reference_texture_metadata)
    reference_texture_metadata.pop("target_path", None)
    return {
        "passed": not failed,
        "verdict": "accepted" if not failed else "rejected_fail_closed",
        "action": action,
        "video": {
            "path": str(resolved_video),
            "sha256": sha256_file(resolved_video),
            **metadata,
        },
        "baseline": {
            "role": "legacy_profile_provenance_only",
            "path": str(resolved_profile),
            "sha256": actual_hashes["profile_baseline"],
            "provenance": baseline_provenance,
            "sha256_verified": True,
            "pixels_used_for_uploaded_identity_appearance_validation": False,
        },
        "validated_action_reference": {
            "path": str(resolved_reference),
            "sha256": actual_hashes["validated_action_reference"],
            "provenance": validated_action_reference_provenance or {},
            "uploaded_identity_authority_validation": (
                uploaded_reference_authority
            ),
            "texture_prop_exclusion": reference_texture_metadata,
        },
        "identity_masks": {
            "pose_video": {
                "path": str(resolved_pose_mask),
                "sha256": actual_hashes["pose_identity_mask_video"],
                "metadata": pose_mask_metadata,
                "background": "white",
            },
            "reference": {
                "path": str(resolved_reference_mask),
                "sha256": actual_hashes["reference_identity_mask"],
                "background": "black",
            },
        },
        "sampling": {
            "requested_count": sample_count,
            "actual_count": len(indices),
            "frame_indices": indices,
        },
        "frames": frame_results,
        "temporal_evidence": {
            "coherent_brush": coherent_brush_temporal_evidence,
            "eating_initial_radial_loop_boundary": (
                eating_radial_boundary_evidence
            ),
            "eating_initial_radial_bounded_reference": (
                eating_radial_bounded_reference_evidence
            ),
            "eating_loop_boundary_anchor": eating_boundary_anchor_evidence,
            "eating_isolated_long_fur_peak": (
                eating_isolated_long_fur_evidence
            ),
        },
        "reference_evidence": {
            "low_signal_texture_floor": low_signal_reference_evidence,
            "sparse_coarse_low_signal_texture_floor": (
                sparse_coarse_low_signal_reference_evidence
            ),
            "direct_qwen_sparse_coarse_temporal_floor": (
                direct_qwen_sparse_coarse_reference_evidence
            ),
        },
        "failed_frame_indices": [item["frame_index"] for item in failed],
        "worst_frame_index": worst["frame_index"],
        "policy": {
            "name": POLICY,
            "version": VERSION,
            "comparison_scope": VALIDATED_ACTION_SURFACE_SCOPE,
            "actions": sorted(SUPPORTED_ACTIONS),
            "profile_baseline_and_validated_action_reference_required": False,
            "legacy_profile_sha_provenance_required": True,
            "profile_pixels_used_for_uploaded_identity_appearance_validation": False,
            "uploaded_qwen_action_reference_is_appearance_authority": True,
            "identity_mask_policy": IDENTITY_MASK_POLICY,
            "eating_prop_alignment_minimum_overlap": (
                FINAL_VIDEO_EATING_PROP_ALIGNMENT_MIN
            ),
            "mask_hash_and_timing_fail_closed": True,
            "unsupported_segmentation_fails_closed": True,
            "all_sampled_frames_must_pass": True,
            "non_radial_absolute_observations_nonblocking": True,
            "geometry_relative_nonblocking": True,
            "unmasked_contour_relative_required": True,
            "prop_excluded_texture_relative_required": True,
            "radial_fraction_relative_requires_threshold_backed_load_expansion_or_energy_backstop": True,
            "radial_load_proxy": (
                "fine_radial_brush_fraction_x_fine_gradient_p90"
            ),
            "radial_energy_backstop_uses_existing_ratio_and_delta_thresholds": True,
            "absolute_radial_issue_remains_blocking": True,
            "eating_initial_radial_loop_endpoint_corroboration": True,
            "eating_initial_radial_requires_single_initial_blocking_sample": True,
            "eating_initial_radial_requires_clean_terminal_and_all_interior_samples": True,
            "eating_initial_radial_requires_hash_bound_radial_long_lines_reference_observation": True,
            "eating_initial_radial_boundary_minimum_samples": (
                FINAL_VIDEO_EATING_RADIAL_BOUNDARY_MIN_SAMPLES
            ),
            "eating_initial_radial_boundary_load_energy_spread_ratio_max": (
                FINAL_VIDEO_EATING_RADIAL_BOUNDARY_LOAD_ENERGY_SPREAD_MAX
            ),
            "eating_initial_radial_boundary_component_spread_ratio_max": (
                FINAL_VIDEO_EATING_RADIAL_BOUNDARY_COMPONENT_SPREAD_MAX
            ),
            "eating_initial_radial_preserves_absolute_radial_fail_closed": True,
            "eating_initial_radial_bounded_reference_corroboration": True,
            "eating_initial_radial_bounded_reference_ratio_margin": (
                FINAL_VIDEO_EATING_RADIAL_BOUNDED_REFERENCE_RATIO_MARGIN
            ),
            "eating_initial_radial_bounded_reference_load_delta_multiplier": (
                FINAL_VIDEO_EATING_RADIAL_BOUNDED_REFERENCE_LOAD_DELTA_MULTIPLIER
            ),
            "eating_initial_radial_bounded_reference_preserves_absolute_radial_fail_closed": True,
            "coherent_brush_relative_requires_gradient_weighted_load_or_long_line_energy_backstop": True,
            "coherent_brush_load_proxy": "structure_tensor_coherent_fraction_x_coarse_gradient_p90",
            "coherent_brush_load_only_temporal_min_samples": (
                FINAL_VIDEO_COHERENT_BRUSH_TEMPORAL_MIN_SAMPLES
            ),
            "coherent_brush_load_only_single_sample_is_observation": True,
            "coherent_brush_long_line_energy_backstop_remains_single_sample_blocking": True,
            "absolute_coherent_brush_issue_alone_is_observation": True,
            "long_fur_relative_requires_contour_corroboration_or_strong_expansion": True,
            "long_fur_texture_only_absolute_and_dual_surface_evidence_is_observation": True,
            "long_fur_strong_ratio_margin": FINAL_VIDEO_LONG_LINE_RATIO_MARGIN,
            "long_fur_strong_delta_multiplier": (
                FINAL_VIDEO_LONG_LINE_DELTA_MULTIPLIER
            ),
            "eating_isolated_strong_long_fur_temporal_corroboration": True,
            "eating_isolated_long_fur_requires_single_interior_blocking_sample": True,
            "eating_isolated_long_fur_records_blocking_frame_unmasked_corroboration": True,
            "eating_isolated_long_fur_requires_adjacent_prop_excluded_observation": True,
            "eating_isolated_long_fur_allows_one_initial_radial_companion_for_later_validation": True,
            "eating_isolated_long_fur_minimum_samples": (
                FINAL_VIDEO_EATING_ISOLATED_LONG_FUR_MIN_SAMPLES
            ),
            "eating_isolated_long_fur_reuses_original_neighbor_ratio_and_delta_thresholds": True,
            "eating_isolated_long_fur_strong_ceiling_uses_one_existing_texture_delta_margin": True,
            "eating_isolated_long_fur_preserves_absolute_contour_radial_coherent_and_unknown_fail_closed": True,
            "eating_loop_boundary_decode_anchor_corroboration": True,
            "eating_boundary_anchor_requires_clean_first_and_last_sample": True,
            "eating_boundary_anchor_requires_contiguous_interior_samples": True,
            "eating_boundary_anchor_minimum_samples": (
                FINAL_VIDEO_EATING_BOUNDARY_ANCHOR_MIN_SAMPLES
            ),
            "eating_boundary_anchor_minimum_blocking_samples": (
                FINAL_VIDEO_EATING_BOUNDARY_ANCHOR_MIN_BLOCKING_SAMPLES
            ),
            "eating_boundary_anchor_spread_ratio_max": (
                FINAL_VIDEO_EATING_BOUNDARY_ANCHOR_SPREAD_MAX
            ),
            "eating_boundary_anchor_preserves_original_ratio_delta_and_absolute_limits": True,
            "grooming_masked_pet3d_low_signal_reference_texture_floor": True,
            "grooming_masked_pet3d_sparse_coarse_low_signal_reference_texture_floor": True,
            "sparse_coarse_low_signal_minimum_samples": (
                FINAL_VIDEO_SPARSE_COARSE_LOW_SIGNAL_MIN_SAMPLES
            ),
            "sparse_coarse_low_signal_minimum_blocking_samples": (
                FINAL_VIDEO_SPARSE_COARSE_LOW_SIGNAL_MIN_BLOCKING_SAMPLES
            ),
            "sparse_coarse_low_signal_maximum_blocking_fraction": (
                FINAL_VIDEO_SPARSE_COARSE_LOW_SIGNAL_MAX_BLOCKING_FRACTION
            ),
            "sparse_coarse_low_signal_requires_all_sample_long_line_observation": True,
            "sparse_coarse_low_signal_coarse_ceiling_uses_one_existing_texture_delta_margin": True,
            "sparse_coarse_low_signal_preserves_absolute_contour_radial_coherent_and_unknown_fail_closed": True,
            "grooming_direct_qwen_sparse_coarse_temporal_floor": True,
            "direct_qwen_sparse_coarse_minimum_samples": (
                FINAL_VIDEO_DIRECT_QWEN_SPARSE_COARSE_MIN_SAMPLES
            ),
            "direct_qwen_sparse_coarse_minimum_blocking_samples": (
                FINAL_VIDEO_DIRECT_QWEN_SPARSE_COARSE_MIN_BLOCKING_SAMPLES
            ),
            "direct_qwen_sparse_coarse_maximum_blocking_fraction": (
                FINAL_VIDEO_DIRECT_QWEN_SPARSE_COARSE_MAX_BLOCKING_FRACTION
            ),
            "direct_qwen_sparse_coarse_temporal_spread_ratio_max": (
                FINAL_VIDEO_DIRECT_QWEN_SPARSE_COARSE_TEMPORAL_SPREAD_MAX
            ),
            "direct_qwen_sparse_coarse_requires_non_low_signal_reference": True,
            "direct_qwen_sparse_coarse_requires_blocking_dual_surface_coarse_long_pair": True,
            "direct_qwen_sparse_coarse_coarse_ceiling_uses_reference_ratio_plus_one_texture_delta": True,
            "direct_qwen_sparse_coarse_long_ceiling_reuses_existing_strong_limits": True,
            "direct_qwen_sparse_coarse_preserves_absolute_contour_radial_coherent_and_unknown_fail_closed": True,
            "low_signal_reference_texture_floor_requires_verified_composite_recovery": True,
            "low_signal_reference_texture_floor_requires_all_samples": True,
            "low_signal_reference_texture_floor_requires_coarse_blocking_all_samples": True,
            "low_signal_reference_texture_floor_minimum_long_blocking_fraction": (
                FINAL_VIDEO_LOW_SIGNAL_REFERENCE_MIN_LONG_BLOCKING_FRACTION
            ),
            "low_signal_reference_texture_floor_allows_remaining_long_observations": True,
            "low_signal_reference_texture_floor_minimum_samples": (
                FINAL_VIDEO_LOW_SIGNAL_REFERENCE_MIN_SAMPLES
            ),
            "low_signal_reference_texture_floor_temporal_spread_ratio_max": (
                FINAL_VIDEO_LOW_SIGNAL_REFERENCE_TEMPORAL_SPREAD_MAX
            ),
            "low_signal_reference_texture_floor_reuses_coherence_gradient_p90_min": True,
            "low_signal_reference_texture_floor_preserves_absolute_contour_radial_and_unknown_fail_closed": True,
            "derived_texture_reference_validated_by_hash_and_alpha_only_transform": True,
            "derived_texture_baseline_absolute_observations_nonblocking": True,
            "unmasked_action_reference_baseline_must_pass_current_absolute_contract": True,
            "legacy_evaluator_vs_profile_suffix_interpreted_as_validated_action_reference": True,
            "malformed_evaluator_contract_fails_closed": True,
            "unknown_relative_issue_fails_closed": True,
            "exact_red_prop_alpha_exclusion": True,
            "prop_exclusion_morphology": False,
            "rembg_model": rembg_model,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--action", choices=sorted(SUPPORTED_ACTIONS), required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--rembg-model", default="u2net")
    parser.add_argument(
        "--comparison-scope",
        choices=(PROFILE_FULL_SCOPE, VALIDATED_ACTION_SURFACE_SCOPE),
        default=PROFILE_FULL_SCOPE,
    )
    parser.add_argument("--validated-action-reference", type=Path)
    parser.add_argument("--validated-action-reference-sha256")
    parser.add_argument("--pose-identity-mask-video", type=Path)
    parser.add_argument("--pose-identity-mask-video-sha256")
    parser.add_argument("--reference-identity-mask", type=Path)
    parser.add_argument("--reference-identity-mask-sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        baseline_path, baseline_sha, provenance = resolve_profile_baseline(
            args.profile,
            args.action,
        )
        result = audit_final_video(
            args.video,
            action=args.action,
            baseline_path=baseline_path,
            baseline_sha256=baseline_sha,
            baseline_provenance=provenance,
            sample_count=args.samples,
            rembg_model=args.rembg_model,
            comparison_scope=args.comparison_scope,
            validated_action_reference_path=args.validated_action_reference,
            validated_action_reference_sha256=(
                args.validated_action_reference_sha256
            ),
            validated_action_reference_provenance={
                "source": "cli_explicit",
                "pre_scail_revalidated": True,
            },
            pose_identity_mask_video_path=args.pose_identity_mask_video,
            pose_identity_mask_video_sha256=(
                args.pose_identity_mask_video_sha256
            ),
            reference_identity_mask_path=args.reference_identity_mask,
            reference_identity_mask_sha256=(
                args.reference_identity_mask_sha256
            ),
        )
    except Exception as exc:  # noqa: BLE001 - CLI emits machine-readable failure
        result = {
            "passed": False,
            "verdict": "audit_error_fail_closed",
            "action": args.action,
            "video": {"path": str(args.video)},
            "error": f"{type(exc).__name__}:{exc}",
            "policy": {"name": POLICY, "version": VERSION},
        }
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if result.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
