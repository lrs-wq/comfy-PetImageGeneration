"""Subject-replacement execution and pre-SCAIL validation."""

from __future__ import annotations

# The extracted functions resolve the facade dependencies after runtime sync.
# ruff: noqa: F821

from .runtime import sync_local_video_runtime as _sync_local_video_runtime


def resolve_transparent_output_phase_start_index(
    job: dict[str, Any],
    *,
    action: str,
    source_frames: int,
) -> int:
    """Use the validated eating-contact guide frame as animation frame zero."""
    if action != "eating":
        return 0
    metadata_by_action = job.get("pose_reference_metadata")
    action_metadata = (
        metadata_by_action.get(action)
        if isinstance(metadata_by_action, dict)
        else None
    )
    generator_metadata = (
        action_metadata.get("generator_metadata")
        if isinstance(action_metadata, dict)
        else None
    )
    if (
        not isinstance(generator_metadata, dict)
        or generator_metadata.get("policy")
        != "driving_eating_geometry_image2_v7"
        or not isinstance(generator_metadata.get("head_lowering_selection"), dict)
        or generator_metadata["head_lowering_selection"].get("passed") is not True
    ):
        return 0
    try:
        selected_frame_index = int(generator_metadata["selected_frame_index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "Validated eating geometry metadata lost its selected contact frame."
        ) from exc
    if not 0 <= selected_frame_index < int(source_frames):
        raise RuntimeError(
            "Validated eating contact frame is outside the generated timeline: "
            f"{selected_frame_index} not in 0..{int(source_frames) - 1}"
        )
    return selected_frame_index


RECOVERABLE_OUTPUT_FALLBACK_POLICY = "identity_anchored_breathing_gif_v2"

TOILETING_TRANSPARENT_REFERENCE_GUIDANCE_POLICY = (
    "reference_aligned_static_subject_and_tray_mask_with_"
    "adaptive_semantic_canvas_hybrid_extraction_v5"
)
TOILETING_TRANSPARENT_DRIVER_GUIDANCE_POLICY = (
    "driver_pose_mask_for_narrow_toileting_reference_v1"
)
TOILETING_TRANSPARENT_REFERENCE_MIN_ASPECT_RATIO = 0.70
DRINKING_TRANSPARENT_FULL_DRIVER_GUIDANCE_POLICY = (
    "drinking_full_driver_subject_and_bowl_adaptive_matte_v1"
)

TOILETING_SCAIL_TEMPORAL_IDENTITY_ARTIFACT_POLICY = (
    "toileting_temporal_lateral_dark_band_and_symbol_gate_v2"
)
TOILETING_SCAIL_DARK_COMPONENT_AREA_MIN = 120
TOILETING_SCAIL_MAX_ATTEMPTS = 3
TOILETING_SCAIL_RETRY_SEED_STEP = 1
TOILETING_SCAIL_MAX_FLAGGED_FRAME_FRACTION = 0.05
TOILETING_SCAIL_MAX_FLAGGED_FRAMES_MINIMUM = 4
TOILETING_SCAIL_POSE_STRENGTH_MULTIPLIERS = (1.0, 0.92, 0.84)

DRINKING_SCAIL_TEMPORAL_ACTION_POLICY = (
    "drinking_source_driver_geometry_retention_gate_v3"
)
DRINKING_SCAIL_MAX_ATTEMPTS = 3
DRINKING_SCAIL_POSE_STRENGTH_MULTIPLIERS = (1.0, 1.18, 1.32)
DRINKING_SCAIL_ADJACENT_MAE_MIN = 0.60
DRINKING_SCAIL_FIRST_FRAME_MAE_MAX_MIN = 3.0
DRINKING_SCAIL_EXPECTED_SILHOUETTE_EXCURSION_MIN = 0.02
DRINKING_SCAIL_SILHOUETTE_RETENTION_RATIO_MIN = 0.35
DRINKING_PREPARED_SOURCE_SILHOUETTE_RETENTION_RATIO_MIN = 0.70
DRINKING_OUTPUT_SOURCE_SILHOUETTE_RETENTION_RATIO_MIN = 0.65
DRINKING_PREPARED_BOWL_AREA_RETENTION_RATIO_MIN = 0.75
DRINKING_SCAIL_BACKGROUND_DISTANCE_MIN = 24.0

EATING_SCAIL_TEMPORAL_ACTION_POLICY = "eating_visible_motion_energy_gate_v1"
EATING_SCAIL_MAX_ATTEMPTS = 2
EATING_SCAIL_POSE_STRENGTH_MULTIPLIERS = (1.0, 0.94)
EATING_SCAIL_ADJACENT_MAE_MIN = 0.55
EATING_SCAIL_FIRST_FRAME_MAE_MAX_MIN = 3.0


def select_transparent_guidance_policy(
    action: str,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Choose role-aware guidance for validated drinking/toileting props.

    A broad toileting reference contains both the uploaded animal and its tray.
    Reusing the narrower white-model pose mask for that reference compares two
    different silhouettes and can reject an otherwise correct SCAIL video. A
    full-driver drinking mask already binds the animated dog and static bowl,
    so it also enables adaptive subject-plus-prop extraction. Other actions and
    legacy narrow references keep their frame-varying driver guidance.
    """

    normalized_action = str(action or "").strip().lower()
    default = {
        "policy": (
            TOILETING_TRANSPARENT_DRIVER_GUIDANCE_POLICY
            if normalized_action == "toileting"
            else "driver_pose_mask_v1"
        ),
        "source_key": "pose_mask_video",
        "static_reference_mask": False,
        "reference_aligned_uniform_background": False,
        "reference_alpha_bbox": None,
        "reference_alpha_bbox_aspect_ratio": None,
        "minimum_reference_aspect_ratio": (
            TOILETING_TRANSPARENT_REFERENCE_MIN_ASPECT_RATIO
            if normalized_action == "toileting"
            else None
        ),
    }
    pose_metadata = manifest.get("pose_metadata")
    if (
        normalized_action == "drinking"
        and isinstance(pose_metadata, dict)
        and pose_metadata.get("primary_policy") is True
        and pose_metadata.get("driver_signal_only") is False
        and pose_metadata.get("driver_silhouette_preserved") is True
        and pose_metadata.get("full_white_driver_geometry") is True
        and pose_metadata.get("pose_geometry_scope")
        == "full_white_driver_subject_per_frame"
        and pose_metadata.get("scene_prop_authority")
        == "digest_bound_white_maquette_video"
    ):
        return {
            **default,
            "policy": DRINKING_TRANSPARENT_FULL_DRIVER_GUIDANCE_POLICY,
            "reference_aligned_uniform_background": True,
            "static_prop_required": True,
        }
    if normalized_action != "toileting":
        return default

    reference_metadata = manifest.get("reference_metadata")
    bbox = (
        reference_metadata.get("reference_alpha_bbox")
        if isinstance(reference_metadata, dict)
        else None
    )
    if not isinstance(bbox, list) or len(bbox) != 4:
        return default
    try:
        left, top, right, bottom = (int(value) for value in bbox)
    except (TypeError, ValueError):
        return default
    width = right - left
    height = bottom - top
    if width < 1 or height < 1:
        return default
    aspect_ratio = float(width) / float(height)
    plan = {
        **default,
        "reference_alpha_bbox": [left, top, right, bottom],
        "reference_alpha_bbox_aspect_ratio": aspect_ratio,
    }
    if aspect_ratio < TOILETING_TRANSPARENT_REFERENCE_MIN_ASPECT_RATIO:
        return plan
    return {
        **plan,
        "policy": TOILETING_TRANSPARENT_REFERENCE_GUIDANCE_POLICY,
        "source_key": "reference_mask",
        "static_reference_mask": True,
        "reference_aligned_uniform_background": True,
    }


def _toileting_lateral_dark_band_frame_quality(
    rgb: np.ndarray,
    *,
    frame_index: int,
) -> dict[str, Any]:
    """Find an invented high-contrast band inside the moving tail corridor.

    The small-dog toilet driver swings the tail along the subject's right side.
    SCAIL can occasionally paint a black/white arrow-like symbol across that tail
    even though the uploaded identity and the white-maquette driver are clean.
    A global palette histogram misses the narrow symbol, while a generic contour
    detector can confuse it with eyes, nose, tray grid, or an exterior key edge.
    This gate therefore examines only a deeply internal, lateral body corridor
    and requires a sizeable elongated locally-dark component.
    """

    frame = np.asarray(rgb, dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[2] != 3:
        return {
            "frame_index": int(frame_index),
            "available": False,
            "flagged": True,
            "reason": "invalid_rgb_frame",
            "components": [],
        }
    height, width = frame.shape[:2]
    if min(height, width) < 64:
        return {
            "frame_index": int(frame_index),
            "available": False,
            "flagged": True,
            "reason": "frame_too_small",
            "components": [],
        }

    border_width = max(4, min(8, min(height, width) // 32))
    border_pixels = np.concatenate(
        (
            frame[:border_width, :, :].reshape(-1, 3),
            frame[-border_width:, :, :].reshape(-1, 3),
            frame[:, :border_width, :].reshape(-1, 3),
            frame[:, -border_width:, :].reshape(-1, 3),
        ),
        axis=0,
    )
    background_rgb = np.median(border_pixels, axis=0)
    canvas_distance = np.sqrt(
        np.sum(
            (frame.astype(np.float32) - background_rgb.astype(np.float32)) ** 2,
            axis=2,
        )
    )
    foreground = (canvas_distance > 12.0).astype(np.uint8)
    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), dtype=np.uint8),
        iterations=2,
    )
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        foreground,
        connectivity=8,
    )
    if component_count <= 1:
        return {
            "frame_index": int(frame_index),
            "available": False,
            "flagged": True,
            "reason": "foreground_unavailable",
            "components": [],
        }
    largest_label = max(
        range(1, component_count),
        key=lambda label: int(stats[label, cv2.CC_STAT_AREA]),
    )
    subject = labels == largest_label
    rows, columns = np.where(subject)
    if rows.size < 256:
        return {
            "frame_index": int(frame_index),
            "available": False,
            "flagged": True,
            "reason": "foreground_too_small",
            "components": [],
        }
    left, right = int(columns.min()), int(columns.max()) + 1
    top, bottom = int(rows.min()), int(rows.max()) + 1
    subject_width = right - left
    subject_height = bottom - top

    luma = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)[:, :, 0]
    local_luma = cv2.GaussianBlur(luma, (31, 31), sigmaX=7.0)
    local_contrast = local_luma.astype(np.int16) - luma.astype(np.int16)
    interior_distance = cv2.distanceTransform(
        subject.astype(np.uint8),
        cv2.DIST_L2,
        5,
    )
    row_grid, column_grid = np.indices(subject.shape)
    lateral_corridor = (
        subject
        & (column_grid >= left + int(round(subject_width * 0.68)))
        & (column_grid < left + int(round(subject_width * 0.96)))
        & (row_grid >= top + int(round(subject_height * 0.25)))
        & (row_grid < top + int(round(subject_height * 0.78)))
        & (interior_distance >= 3.0)
    )
    dark = (
        lateral_corridor
        & (luma < 70)
        & (local_contrast >= 25)
    )
    dark = cv2.morphologyEx(
        dark.astype(np.uint8),
        cv2.MORPH_OPEN,
        np.ones((2, 2), dtype=np.uint8),
    )
    dark_count, dark_labels, dark_stats, _ = cv2.connectedComponentsWithStats(
        dark,
        connectivity=8,
    )
    selected_components: list[dict[str, int | float]] = []
    for label in range(1, dark_count):
        x, y, component_width, component_height, area = (
            int(value) for value in dark_stats[label]
        )
        long_side = max(component_width, component_height)
        short_side = max(1, min(component_width, component_height))
        elongation = float(long_side) / float(short_side)
        if (
            area < TOILETING_SCAIL_DARK_COMPONENT_AREA_MIN
            or long_side < 18
            or elongation < 2.2
        ):
            continue
        selected_components.append(
            {
                "x": x,
                "y": y,
                "width": component_width,
                "height": component_height,
                "pixels": area,
                "elongation": round(elongation, 6),
            }
        )
    return {
        "frame_index": int(frame_index),
        "available": True,
        "flagged": bool(selected_components),
        "subject_bbox": [left, top, right, bottom],
        "background_rgb": [round(float(value), 3) for value in background_rgb],
        "components": selected_components,
    }


def toileting_temporal_identity_artifact_quality(
    frames: list[np.ndarray],
) -> dict[str, Any]:
    """Require invented tail bands/symbols to be absent across the video."""

    _sync_local_video_runtime(
        globals(),
        {"toileting_temporal_identity_artifact_quality"},
    )
    frame_metrics = [
        _toileting_lateral_dark_band_frame_quality(
            frame,
            frame_index=index,
        )
        for index, frame in enumerate(frames)
    ]
    frame_count = len(frame_metrics)
    unavailable = [
        metric for metric in frame_metrics if metric.get("available") is not True
    ]
    flagged = [metric for metric in frame_metrics if metric.get("flagged") is True]
    permitted_flagged_frames = max(
        TOILETING_SCAIL_MAX_FLAGGED_FRAMES_MINIMUM,
        int(round(frame_count * TOILETING_SCAIL_MAX_FLAGGED_FRAME_FRACTION)),
    )
    acceptable = bool(
        frame_count >= 9
        and not unavailable
        and len(flagged) <= permitted_flagged_frames
    )
    issues: list[str] = []
    if frame_count < 9:
        issues.append("toileting_video_too_short_for_temporal_artifact_gate")
    if unavailable:
        issues.append("toileting_video_artifact_frame_analysis_unavailable")
    if len(flagged) > permitted_flagged_frames:
        issues.append("toileting_temporal_lateral_dark_band_or_symbol")
    peak_component_pixels = max(
        (
            int(component.get("pixels", 0))
            for metric in flagged
            for component in metric.get("components", [])
            if isinstance(component, dict)
        ),
        default=0,
    )
    return {
        "policy": TOILETING_SCAIL_TEMPORAL_IDENTITY_ARTIFACT_POLICY,
        "available": bool(frame_count >= 9 and not unavailable),
        "acceptable": acceptable,
        "issues": issues,
        "frame_count": frame_count,
        "flagged_frame_count": len(flagged),
        "flagged_frame_indices": [
            int(metric["frame_index"]) for metric in flagged
        ],
        "permitted_flagged_frames": permitted_flagged_frames,
        "dark_component_area_minimum": (
            TOILETING_SCAIL_DARK_COMPONENT_AREA_MIN
        ),
        "maximum_flagged_frame_fraction": (
            TOILETING_SCAIL_MAX_FLAGGED_FRAME_FRACTION
        ),
        "peak_component_pixels": peak_component_pixels,
        "frames": flagged,
        "penalty": round(
            float(len(unavailable) * 1000 + len(flagged) * 100)
            + float(peak_component_pixels),
            6,
        ),
    }


def evaluate_toileting_scail_video_identity_artifacts(
    video_path: Path,
) -> dict[str, Any]:
    """Decode and enforce the small-dog toileting temporal marking contract."""

    _sync_local_video_runtime(
        globals(),
        {"evaluate_toileting_scail_video_identity_artifacts"},
    )
    capture = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    quality = toileting_temporal_identity_artifact_quality(frames)
    return {
        **quality,
        "video_path": str(video_path),
        "video_sha256": file_sha256(video_path) if video_path.is_file() else None,
    }


def _drinking_subject_silhouette(frame: np.ndarray) -> np.ndarray | None:
    """Extract the largest subject component from a uniform-background frame."""

    rgb = np.asarray(frame, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return None
    height, width = rgb.shape[:2]
    border_width = max(4, min(height, width) // 64)
    border = np.concatenate(
        (
            rgb[:border_width].reshape(-1, 3),
            rgb[-border_width:].reshape(-1, 3),
            rgb[:, :border_width].reshape(-1, 3),
            rgb[:, -border_width:].reshape(-1, 3),
        ),
        axis=0,
    )
    background = np.median(border.astype(np.float32), axis=0)
    distance = np.linalg.norm(
        rgb.astype(np.float32) - background[None, None, :],
        axis=2,
    )
    foreground = (distance >= DRINKING_SCAIL_BACKGROUND_DISTANCE_MIN).astype(
        np.uint8
    )
    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    )
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        foreground,
        connectivity=8,
    )
    if component_count < 2:
        return None
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    largest_area = int(stats[largest_label, cv2.CC_STAT_AREA])
    if largest_area < max(256, int(round(height * width * 0.005))):
        return None
    return labels == largest_label


def _drinking_silhouette_excursion(
    frames: list[np.ndarray],
) -> dict[str, Any]:
    """Measure subject-outline change relative to the first animation frame."""

    silhouettes = [_drinking_subject_silhouette(frame) for frame in frames]
    available = [mask for mask in silhouettes if isinstance(mask, np.ndarray)]
    if len(available) != len(frames) or not available:
        return {
            "available": False,
            "frame_count": len(frames),
            "decoded_silhouette_count": len(available),
            "first_frame_normalized_xor_max": 0.0,
            "first_frame_normalized_xor_max_frame_index": None,
        }
    first = available[0]
    excursion_by_frame: list[float] = []
    for silhouette in available:
        normalization_area = max(
            1,
            int(np.count_nonzero(first)),
            int(np.count_nonzero(silhouette)),
        )
        excursion_by_frame.append(
            float(np.count_nonzero(np.logical_xor(first, silhouette)))
            / float(normalization_area)
        )
    maximum = max(excursion_by_frame, default=0.0)
    return {
        "available": True,
        "frame_count": len(frames),
        "decoded_silhouette_count": len(available),
        "first_frame_normalized_xor_max": round(maximum, 6),
        "first_frame_normalized_xor_max_frame_index": (
            int(np.argmax(excursion_by_frame)) if excursion_by_frame else None
        ),
    }


def drinking_scail_video_motion_quality(
    frames: list[np.ndarray],
    *,
    expected_pose_frames: list[np.ndarray] | None = None,
    source_driver_frames: list[np.ndarray] | None = None,
) -> dict[str, Any]:
    """Reject drinking that collapses either prepared or source geometry."""

    _sync_local_video_runtime(
        globals(),
        {"drinking_scail_video_motion_quality"},
    )

    normalized_frames = [
        cv2.cvtColor(np.asarray(frame, dtype=np.uint8), cv2.COLOR_RGB2GRAY).astype(
            np.float32
        )
        for frame in frames
        if isinstance(frame, np.ndarray)
        and frame.ndim == 3
        and frame.shape[2] == 3
    ]
    adjacent_mae = [
        float(np.mean(np.abs(current - previous)))
        for previous, current in zip(
            normalized_frames,
            normalized_frames[1:],
            strict=False,
        )
    ]
    first_frame_mae = [
        float(np.mean(np.abs(frame - normalized_frames[0])))
        for frame in normalized_frames
    ] if normalized_frames else []
    adjacent_mae_mean = float(np.mean(adjacent_mae)) if adjacent_mae else 0.0
    first_frame_mae_max = max(first_frame_mae, default=0.0)
    issues: list[str] = []
    if len(normalized_frames) < 9:
        issues.append("drinking_video_too_short_for_motion_gate")
    if adjacent_mae_mean < DRINKING_SCAIL_ADJACENT_MAE_MIN:
        issues.append("drinking_motion_energy_too_low")
    if first_frame_mae_max < DRINKING_SCAIL_FIRST_FRAME_MAE_MAX_MIN:
        issues.append("drinking_action_excursion_too_low")
    output_silhouette = _drinking_silhouette_excursion(frames)
    expected_silhouette = (
        _drinking_silhouette_excursion(expected_pose_frames)
        if expected_pose_frames is not None
        else None
    )
    source_driver_silhouette = (
        _drinking_silhouette_excursion(source_driver_frames)
        if source_driver_frames is not None
        else None
    )
    silhouette_retention_ratio: float | None = None
    if expected_silhouette is not None:
        expected_excursion = float(
            expected_silhouette["first_frame_normalized_xor_max"]
        )
        output_excursion = float(
            output_silhouette["first_frame_normalized_xor_max"]
        )
        if expected_silhouette.get("available") is not True:
            issues.append("drinking_expected_pose_silhouette_unavailable")
        elif (
            expected_excursion
            < DRINKING_SCAIL_EXPECTED_SILHOUETTE_EXCURSION_MIN
        ):
            issues.append("drinking_expected_pose_silhouette_excursion_too_low")
        elif output_silhouette.get("available") is not True:
            issues.append("drinking_output_silhouette_unavailable")
        else:
            silhouette_retention_ratio = output_excursion / expected_excursion
            if (
                silhouette_retention_ratio
                < DRINKING_SCAIL_SILHOUETTE_RETENTION_RATIO_MIN
            ):
                issues.append(
                    "drinking_driver_silhouette_excursion_retention_too_low"
                )
    prepared_source_retention_ratio: float | None = None
    output_source_retention_ratio: float | None = None
    if source_driver_silhouette is not None:
        source_excursion = float(
            source_driver_silhouette["first_frame_normalized_xor_max"]
        )
        prepared_silhouette = expected_silhouette or output_silhouette
        prepared_excursion = float(
            prepared_silhouette["first_frame_normalized_xor_max"]
        )
        output_excursion = float(
            output_silhouette["first_frame_normalized_xor_max"]
        )
        if source_driver_silhouette.get("available") is not True:
            issues.append("drinking_source_driver_silhouette_unavailable")
        elif (
            source_excursion
            < DRINKING_SCAIL_EXPECTED_SILHOUETTE_EXCURSION_MIN
        ):
            issues.append("drinking_source_driver_silhouette_excursion_too_low")
        elif prepared_silhouette.get("available") is not True:
            issues.append("drinking_prepared_pose_silhouette_unavailable")
        elif output_silhouette.get("available") is not True:
            issues.append("drinking_output_silhouette_unavailable")
        else:
            prepared_source_retention_ratio = (
                prepared_excursion / source_excursion
            )
            output_source_retention_ratio = output_excursion / source_excursion
            if (
                prepared_source_retention_ratio
                < DRINKING_PREPARED_SOURCE_SILHOUETTE_RETENTION_RATIO_MIN
            ):
                issues.append(
                    "drinking_prepared_pose_source_driver_geometry_retention_too_low"
                )
            if (
                output_source_retention_ratio
                < DRINKING_OUTPUT_SOURCE_SILHOUETTE_RETENTION_RATIO_MIN
            ):
                issues.append(
                    "drinking_output_source_driver_geometry_retention_too_low"
                )
    return {
        "policy": DRINKING_SCAIL_TEMPORAL_ACTION_POLICY,
        "available": len(normalized_frames) >= 9,
        "acceptable": not issues,
        "issues": issues,
        "frame_count": len(normalized_frames),
        "adjacent_grayscale_mae_mean": round(adjacent_mae_mean, 6),
        "adjacent_grayscale_mae_minimum": DRINKING_SCAIL_ADJACENT_MAE_MIN,
        "first_frame_grayscale_mae_max": round(first_frame_mae_max, 6),
        "first_frame_grayscale_mae_max_minimum": (
            DRINKING_SCAIL_FIRST_FRAME_MAE_MAX_MIN
        ),
        "output_silhouette_excursion": output_silhouette,
        "expected_pose_silhouette_excursion": expected_silhouette,
        "source_driver_silhouette_excursion": source_driver_silhouette,
        "silhouette_excursion_retention_ratio": (
            round(silhouette_retention_ratio, 6)
            if silhouette_retention_ratio is not None
            else None
        ),
        "silhouette_excursion_retention_ratio_minimum": (
            DRINKING_SCAIL_SILHOUETTE_RETENTION_RATIO_MIN
        ),
        "prepared_pose_to_source_driver_silhouette_retention_ratio": (
            round(prepared_source_retention_ratio, 6)
            if prepared_source_retention_ratio is not None
            else None
        ),
        "prepared_pose_to_source_driver_silhouette_retention_ratio_minimum": (
            DRINKING_PREPARED_SOURCE_SILHOUETTE_RETENTION_RATIO_MIN
        ),
        "output_to_source_driver_silhouette_retention_ratio": (
            round(output_source_retention_ratio, 6)
            if output_source_retention_ratio is not None
            else None
        ),
        "output_to_source_driver_silhouette_retention_ratio_minimum": (
            DRINKING_OUTPUT_SOURCE_SILHOUETTE_RETENTION_RATIO_MIN
        ),
    }


def eating_scail_video_motion_quality(
    frames: list[np.ndarray],
) -> dict[str, Any]:
    """Reject an eating result whose prepared head-nod did not survive SCAIL."""

    _sync_local_video_runtime(
        globals(),
        {"eating_scail_video_motion_quality"},
    )

    quality = drinking_scail_video_motion_quality(frames)
    adjacent = float(quality["adjacent_grayscale_mae_mean"])
    excursion = float(quality["first_frame_grayscale_mae_max"])
    issues: list[str] = []
    if int(quality["frame_count"]) < 9:
        issues.append("eating_video_too_short_for_motion_gate")
    if adjacent < EATING_SCAIL_ADJACENT_MAE_MIN:
        issues.append("eating_motion_energy_too_low")
    if excursion < EATING_SCAIL_FIRST_FRAME_MAE_MAX_MIN:
        issues.append("eating_action_excursion_too_low")
    return {
        **quality,
        "policy": EATING_SCAIL_TEMPORAL_ACTION_POLICY,
        "acceptable": not issues,
        "issues": issues,
        "adjacent_grayscale_mae_minimum": EATING_SCAIL_ADJACENT_MAE_MIN,
        "first_frame_grayscale_mae_max_minimum": (
            EATING_SCAIL_FIRST_FRAME_MAE_MAX_MIN
        ),
    }


def evaluate_drinking_scail_video_motion(
    video_path: Path,
    *,
    expected_pose_video_path: Path | None = None,
    source_driver_video_path: Path | None = None,
) -> dict[str, Any]:
    """Decode and enforce the dog drinking motion contract."""

    _sync_local_video_runtime(
        globals(),
        {"evaluate_drinking_scail_video_motion"},
    )
    capture = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    expected_pose_frames: list[np.ndarray] | None = None
    if expected_pose_video_path is not None:
        expected_capture = cv2.VideoCapture(str(expected_pose_video_path))
        expected_pose_frames = []
        try:
            while True:
                ok, bgr = expected_capture.read()
                if not ok:
                    break
                expected_pose_frames.append(
                    cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                )
        finally:
            expected_capture.release()
    source_driver_frames: list[np.ndarray] | None = None
    if source_driver_video_path is not None:
        source_capture = cv2.VideoCapture(str(source_driver_video_path))
        source_driver_frames = []
        try:
            while True:
                ok, bgr = source_capture.read()
                if not ok:
                    break
                source_driver_frames.append(
                    cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                )
        finally:
            source_capture.release()
    quality = drinking_scail_video_motion_quality(
        frames,
        expected_pose_frames=expected_pose_frames,
        source_driver_frames=source_driver_frames,
    )
    return {
        **quality,
        "video_path": str(video_path),
        "video_sha256": file_sha256(video_path) if video_path.is_file() else None,
        "expected_pose_video_path": (
            str(expected_pose_video_path)
            if expected_pose_video_path is not None
            else None
        ),
        "expected_pose_video_sha256": (
            file_sha256(expected_pose_video_path)
            if expected_pose_video_path is not None
            and expected_pose_video_path.is_file()
            else None
        ),
        "source_driver_video_path": (
            str(source_driver_video_path)
            if source_driver_video_path is not None
            else None
        ),
        "source_driver_video_sha256": (
            file_sha256(source_driver_video_path)
            if source_driver_video_path is not None
            and source_driver_video_path.is_file()
            else None
        ),
    }


def evaluate_drinking_prepared_pose_motion(
    prepared_pose_video_path: Path,
    *,
    source_driver_video_path: Path,
) -> dict[str, Any]:
    """Fail before SCAIL when preparation already collapsed source motion."""

    quality = evaluate_drinking_scail_video_motion(
        prepared_pose_video_path,
        expected_pose_video_path=prepared_pose_video_path,
        source_driver_video_path=source_driver_video_path,
    )
    return {
        **quality,
        "validation_scope": "source_driver_to_prepared_pose_before_scail",
    }


def evaluate_eating_scail_video_motion(video_path: Path) -> dict[str, Any]:
    """Decode and enforce the dog eating motion contract."""

    _sync_local_video_runtime(
        globals(),
        {"evaluate_eating_scail_video_motion"},
    )
    capture = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    quality = eating_scail_video_motion_quality(frames)
    return {
        **quality,
        "video_path": str(video_path),
        "video_sha256": file_sha256(video_path) if video_path.is_file() else None,
    }


def scail_workflow_with_retry_seed(
    workflow: dict[str, Any],
    *,
    seed: int,
) -> tuple[dict[str, Any], list[str]]:
    """Copy a prepared SCAIL workflow and replace every sampler noise seed."""

    _sync_local_video_runtime(globals(), {"scail_workflow_with_retry_seed"})
    candidate = json.loads(json.dumps(workflow))
    updated_nodes: list[str] = []
    for node_id, node in candidate.items():
        inputs = node.get("inputs") if isinstance(node, dict) else None
        if not isinstance(inputs, dict) or "noise_seed" not in inputs:
            continue
        inputs["noise_seed"] = int(seed)
        updated_nodes.append(str(node_id))
    if not updated_nodes:
        raise RuntimeError("Prepared SCAIL workflow contains no noise_seed input.")
    return candidate, updated_nodes


def scail_workflow_with_retry_controls(
    workflow: dict[str, Any],
    *,
    seed: int,
    action: str,
    attempt: int,
) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    """Vary both seed and action pose strength across quality retries."""

    candidate, updated_seed_nodes = scail_workflow_with_retry_seed(
        workflow,
        seed=seed,
    )
    normalized_action = str(action or "").strip().lower()
    multipliers = (
        TOILETING_SCAIL_POSE_STRENGTH_MULTIPLIERS
        if normalized_action == "toileting"
        else DRINKING_SCAIL_POSE_STRENGTH_MULTIPLIERS
        if normalized_action == "drinking"
        else EATING_SCAIL_POSE_STRENGTH_MULTIPLIERS
        if normalized_action == "eating"
        else (1.0,)
    )
    multiplier = float(multipliers[min(max(1, attempt), len(multipliers)) - 1])
    updated_pose_nodes: list[str] = []
    original_strengths: dict[str, float] = {}
    effective_strengths: dict[str, float] = {}
    for node_id, node in candidate.items():
        if not isinstance(node, dict) or node.get("class_type") != "WanSCAILToVideo":
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict) or "pose_strength" not in inputs:
            continue
        original = float(inputs["pose_strength"])
        effective = round(original * multiplier, 6)
        inputs["pose_strength"] = effective
        updated_pose_nodes.append(str(node_id))
        original_strengths[str(node_id)] = original
        effective_strengths[str(node_id)] = effective
    if not updated_pose_nodes:
        raise RuntimeError(
            "Prepared SCAIL workflow contains no WanSCAIL pose_strength input."
        )
    return (
        candidate,
        updated_seed_nodes,
        {
            "policy": "seed_plus_action_pose_strength_retry_v1",
            "action": normalized_action,
            "attempt": int(attempt),
            "pose_strength_multiplier": multiplier,
            "updated_pose_nodes": updated_pose_nodes,
            "original_pose_strengths": original_strengths,
            "effective_pose_strengths": effective_strengths,
        },
    )


def _recoverable_candidate_path(job_dir: Path, value: Any) -> Path | None:
    selected = str(value or "").strip()
    if not selected:
        return None
    path = Path(selected)
    return path if path.is_absolute() else job_dir / path


def _recoverable_reference_candidates(
    job_dir: Path,
    job: dict[str, Any],
    action: str,
    index: int,
) -> list[tuple[Path, str]]:
    candidates: list[tuple[Path, str]] = [
        # Terminal recovery for an uploaded subject must remain anchored to the
        # submitted identity even when action/canonical generation failed early.
        (job_dir / "input_identity_source.png", "original_upload"),
        (job_dir / "input.png", "normalized_upload"),
        (
            job_dir / f"pose_{index:02d}_qwen_reference_semantic_final.png",
            "action_semantic_reference",
        ),
        (job_dir / f"pose_{index:02d}_qwen_cutout.png", "action_qwen_cutout"),
        (
            job_dir / f"pose_{index:02d}_keyframe_transparent.png",
            "action_transparent_keyframe",
        ),
        (job_dir / "cartoon_reference_transparent.png", "canonical_transparent"),
    ]
    reference_paths = job.get("subject_reference_paths")
    if isinstance(reference_paths, dict):
        selected = _recoverable_candidate_path(job_dir, reference_paths.get(action))
        if selected is not None:
            candidates.append((selected, "persisted_action_reference"))
    effective = job.get("effective_subject_config")
    if isinstance(effective, dict):
        action_config = effective.get("actions", {}).get(action)
        if isinstance(action_config, dict):
            for field, role in (
                ("reference", "effective_action_reference"),
                ("geometry_reference", "effective_geometry_reference"),
            ):
                selected = _recoverable_candidate_path(job_dir, action_config.get(field))
                if selected is not None:
                    candidates.append((selected, role))
    unique: list[tuple[Path, str]] = []
    seen: set[str] = set()
    for path, role in candidates:
        key = str(path)
        if key not in seen:
            unique.append((path, role))
            seen.add(key)
    return unique


def _load_recoverable_reference(
    candidates: list[tuple[Path, str]],
) -> tuple[Path, str, Image.Image]:
    failures: list[str] = []
    for path, role in candidates:
        if not path.is_file():
            continue
        try:
            with Image.open(path) as loaded:
                rgba = loaded.convert("RGBA")
                rgba.load()
            if rgba.width < 2 or rgba.height < 2:
                failures.append(f"{path}: image_too_small")
                continue
            alpha_bbox = rgba.getchannel("A").getbbox()
            if alpha_bbox is None:
                failures.append(f"{path}: empty_alpha")
                continue
            return path, role, rgba
        except Exception as exc:  # noqa: BLE001 - try the next audited source
            failures.append(f"{path}: {type(exc).__name__}: {str(exc)[:200]}")
    raise RuntimeError(
        "No readable source is available for degraded output fallback: "
        + "; ".join(failures[-5:])
    )


def _normalize_recoverable_reference(
    source: Image.Image,
    target: Path,
    *,
    size: int,
) -> dict[str, Any]:
    alpha_bbox = source.getchannel("A").getbbox()
    if alpha_bbox is None:
        raise RuntimeError("Degraded output fallback source has empty alpha.")
    subject = source.crop(alpha_bbox)
    maximum_extent = max(32, int(round(size * 0.88)))
    scale = min(
        1.0,
        maximum_extent / float(subject.width),
        maximum_extent / float(subject.height),
    )
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError("Degraded output fallback source size is invalid.")
    resized_size = (
        max(1, int(round(subject.width * scale))),
        max(1, int(round(subject.height * scale))),
    )
    subject = subject.resize(resized_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    offset = ((size - subject.width) // 2, (size - subject.height) // 2)
    canvas.alpha_composite(subject, offset)
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target)
    return {
        "source_alpha_bbox": list(alpha_bbox),
        "source_size": list(source.size),
        "normalized_size": [size, size],
        "subject_size": list(subject.size),
        "subject_offset": list(offset),
        "sha256": file_sha256(target),
    }


def _inspect_recoverable_gif(
    path: Path,
    *,
    size: int,
    frame_count: int,
    fps: int,
) -> dict[str, Any]:
    expected_duration_ms = int(round(frame_count * 1000 / fps))
    with Image.open(path) as animation:
        actual_frames = int(getattr(animation, "n_frames", 1))
        actual_size = animation.size
        loop = animation.info.get("loop")
        duration_ms = 0
        transparent_pixel_seen = False
        opaque_pixel_seen = False
        for frame_index in range(actual_frames):
            animation.seek(frame_index)
            duration_ms += int(animation.info.get("duration") or 0)
            if frame_index in {0, actual_frames - 1}:
                alpha_extrema = animation.convert("RGBA").getchannel("A").getextrema()
                transparent_pixel_seen = transparent_pixel_seen or alpha_extrema[0] == 0
                opaque_pixel_seen = opaque_pixel_seen or alpha_extrema[1] >= 64
    if actual_size != (size, size):
        raise RuntimeError(
            f"Degraded fallback GIF size changed: {actual_size} != {(size, size)}"
        )
    if actual_frames != frame_count:
        raise RuntimeError(
            "Degraded fallback GIF frame count changed: "
            f"{actual_frames} != {frame_count}"
        )
    if abs(duration_ms - expected_duration_ms) > max(100, int(1000 / fps)):
        raise RuntimeError(
            "Degraded fallback GIF duration changed: "
            f"{duration_ms} != {expected_duration_ms}"
        )
    if loop != 0:
        raise RuntimeError(f"Degraded fallback GIF loop changed: {loop!r}")
    if not transparent_pixel_seen or not opaque_pixel_seen:
        raise RuntimeError(
            "Degraded fallback GIF does not contain both transparent and visible pixels."
        )
    return {
        "size": [size, size],
        "frames": actual_frames,
        "fps": fps,
        "duration_ms": duration_ms,
        "loop": loop,
        "transparent": True,
        "sha256": file_sha256(path),
    }


def _write_recoverable_breathing_gif(
    reference: Path,
    target: Path,
    *,
    size: int,
    frame_count: int,
    fps: int,
) -> dict[str, Any]:
    if frame_count < 2 or fps < 1:
        raise ValueError("Degraded output fallback animation timing is invalid.")
    with Image.open(reference) as loaded:
        rgba = loaded.convert("RGBA")
        rgba.load()
    alpha_bbox = rgba.getchannel("A").getbbox()
    if alpha_bbox is None:
        raise RuntimeError("Degraded output fallback reference has empty alpha.")
    subject = rgba.crop(alpha_bbox)
    frame_dir = target.parent / f".{target.stem}.{uuid4().hex}.frames"
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp.gif")
    frame_dir.mkdir(parents=True)
    encoding = "ffmpeg_palette"
    try:
        denominator = max(1, frame_count)
        previous_transform: tuple[tuple[int, int], int, int] | None = None
        for frame_index in range(frame_count):
            phase = 2.0 * np.pi * frame_index / denominator
            breath = (1.0 - float(np.cos(phase))) * 0.5
            frame_scale = 1.0 + 0.012 * breath
            frame_size = (
                max(1, int(round(subject.width * frame_scale))),
                max(1, int(round(subject.height * frame_scale))),
            )
            animated_subject = subject.resize(
                frame_size,
                Image.Resampling.LANCZOS,
            )
            frame = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            frame_x = (size - animated_subject.width) // 2 + int(
                round(1.5 * float(np.sin(phase)))
            )
            frame_y = (size - animated_subject.height) // 2 - int(
                round(1.5 * breath)
            )
            transform = (frame_size, frame_x, frame_y)
            if transform == previous_transform:
                for delta_x, delta_y in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    candidate_x = frame_x + delta_x
                    candidate_y = frame_y + delta_y
                    candidate = (frame_size, candidate_x, candidate_y)
                    if (
                        candidate != previous_transform
                        and candidate_x >= 0
                        and candidate_y >= 0
                        and candidate_x + animated_subject.width <= size
                        and candidate_y + animated_subject.height <= size
                    ):
                        frame_x = candidate_x
                        frame_y = candidate_y
                        transform = candidate
                        break
            frame.alpha_composite(animated_subject, (frame_x, frame_y))
            frame.save(frame_dir / f"frame_{frame_index:03d}.png")
            previous_transform = transform

        gif_filter = (
            "[0:v]format=rgba,split[gif][pal];"
            "[pal]palettegen=reserve_transparent=1:"
            "transparency_color=0x000000:stats_mode=diff[p];"
            "[gif][p]paletteuse=dither=sierra2_4a:"
            "alpha_threshold=128:diff_mode=rectangle[v]"
        )
        command = [
            str(FFMPEG_BIN),
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            str(frame_dir / "frame_%03d.png"),
            "-filter_complex",
            gif_filter,
            "-map",
            "[v]",
            "-frames:v",
            str(frame_count),
            "-loop",
            "0",
            "-final_delay",
            str(int(round(100 / fps))),
            str(temporary),
        ]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=180,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "ffmpeg degraded GIF encoding failed: "
                    + (completed.stderr or completed.stdout or "unknown")[-1000:]
                )
            _inspect_recoverable_gif(
                temporary,
                size=size,
                frame_count=frame_count,
                fps=fps,
            )
        except Exception:  # noqa: BLE001 - Pillow is the dependency-free fallback
            encoding = "pillow_multiframe"
            frames: list[Image.Image] = []
            try:
                for index in range(frame_count):
                    with Image.open(
                        frame_dir / f"frame_{index:03d}.png"
                    ) as loaded_frame:
                        frame = loaded_frame.convert("RGBA")
                        frame.load()
                    frames.append(frame)
                frames[0].save(
                    temporary,
                    format="GIF",
                    save_all=True,
                    append_images=frames[1:],
                    duration=int(round(1000 / fps)),
                    loop=0,
                    disposal=2,
                    optimize=False,
                )
            finally:
                for frame in frames:
                    frame.close()

        inspected = _inspect_recoverable_gif(
            temporary,
            size=size,
            frame_count=frame_count,
            fps=fps,
        )
        os.replace(temporary, target)
        return {
            "policy": RECOVERABLE_OUTPUT_FALLBACK_POLICY,
            "encoding": encoding,
            **inspected,
        }
    finally:
        temporary.unlink(missing_ok=True)
        shutil.rmtree(frame_dir, ignore_errors=True)


def recover_subject_replacement_action_with_fallback(
    job_id: str,
    job: dict[str, Any],
    *,
    action: str,
    pose_index: int,
    error: BaseException,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create one explicit degraded action result and keep sibling actions running."""

    _sync_local_video_runtime(
        globals(),
        {"recover_subject_replacement_action_with_fallback"},
    )
    job_dir = OUTPUT_ROOT / job_id
    source_path, source_role, source_image = _load_recoverable_reference(
        _recoverable_reference_candidates(job_dir, job, action, pose_index)
    )
    try:
        fallback_reference = (
            job_dir / f"{action}_degraded_fallback_reference.png"
        )
        normalization = _normalize_recoverable_reference(
            source_image,
            fallback_reference,
            size=TRANSPARENT_GIF_SIZE,
        )
    finally:
        source_image.close()
    effective = job.get("effective_subject_config")
    action_timing = subject_replacement_action_output_timing(
        effective,
        action,
    )
    gif_timing = action_timing["transparent_gif"]
    gif_target = job_dir / f"{action}_subject_replacement_transparent_512.gif"
    animation = _write_recoverable_breathing_gif(
        fallback_reference,
        gif_target,
        size=TRANSPARENT_GIF_SIZE,
        frame_count=int(gif_timing["frames"]),
        fps=int(gif_timing["fps"]),
    )
    action_label = SUBJECT_REPLACEMENT_ACTION_LABELS.get(action, action)
    original_error = f"{type(error).__name__}: {str(error)[:1000]}"
    append_output(
        job_id,
        {
            "type": "degraded_fallback_reference",
            "action": action,
            "action_label": action_label,
            "filename": fallback_reference.name,
            "width": TRANSPARENT_GIF_SIZE,
            "height": TRANSPARENT_GIF_SIZE,
            "transparent": True,
            "sha256": normalization["sha256"],
            "fallback_source": str(source_path),
            "fallback_source_role": source_role,
            "primary": False,
        },
    )
    append_output(
        job_id,
        {
            "type": "transparent_gif",
            "action": action,
            "action_label": action_label,
            "filename": gif_target.name,
            "width": TRANSPARENT_GIF_SIZE,
            "height": TRANSPARENT_GIF_SIZE,
            "frames": int(gif_timing["frames"]),
            "fps": int(gif_timing["fps"]),
            "duration_seconds": float(gif_timing["duration_seconds"]),
            "output_playback": action_timing,
            "transparent": True,
            "video_engine": RECOVERABLE_OUTPUT_FALLBACK_POLICY,
            "degraded": True,
            "quality_passed": False,
            "generation_succeeded": False,
            "output_exists": True,
            "sha256": animation["sha256"],
            "fallback_source": str(source_path),
            "fallback_source_role": source_role,
            "primary": True,
        },
    )
    result = {
        "action": action,
        "action_label": action_label,
        "reference": str(fallback_reference),
        "reference_sha256": normalization["sha256"],
        "video": None,
        "transparent_gif": gif_target.name,
        "transparent_metrics": animation,
        "output_playback": action_timing,
        "output_exists": True,
        "generation_succeeded": False,
        "quality_passed": False,
        "degraded": True,
        "degraded_policy": RECOVERABLE_OUTPUT_FALLBACK_POLICY,
        "fallback_source": str(source_path),
        "fallback_source_role": source_role,
        "original_error": original_error,
    }
    source = {
        "path": str(source_path),
        "role": source_role,
        "normalization": normalization,
        "animation": animation,
    }
    append_recovery_event(
        job_id,
        {
            "stage": "action_output_recovery",
            "action": action,
            "reason": original_error,
            "strategy": RECOVERABLE_OUTPUT_FALLBACK_POLICY,
            "result": "accepted_degraded_action_only",
            "source": source,
        },
    )
    record_nonblocking_job_anomaly(
        job_id,
        code=f"degraded_output_fallback_{action}",
        message=(
            f"{action_label}动作生成未通过质量门禁，已仅对该动作生成身份锚定的"
            "降级循环；其他动作继续独立生成。"
        ),
        stage="completed_with_degraded_output",
        details={
            "action": action,
            "policy": RECOVERABLE_OUTPUT_FALLBACK_POLICY,
            "exception_type": type(error).__name__,
            "error": str(error)[:1000],
            "fallback_source_role": source_role,
        },
        expose_as_public_stage=True,
    )
    return result, source


def recover_subject_replacement_job_with_fallback(
    job_id: str,
    error: BaseException,
) -> bool:
    """Return audited GIFs for every action after an otherwise fatal job error."""

    _sync_local_video_runtime(
        globals(),
        {"recover_subject_replacement_job_with_fallback"},
    )
    job = get_job(job_id)
    if (
        job.get("generation_mode") != "subject_replacement"
        or job.get("output_format") != "transparent_gif"
        or job.get("cancel_requested")
    ):
        return False
    actions = [str(action) for action in job.get("actions") or []]
    if not actions:
        return False
    effective = job.get("effective_subject_config")
    job_dir = OUTPUT_ROOT / job_id
    existing_outputs = [
        dict(output)
        for output in job.get("outputs") or []
        if isinstance(output, dict)
    ]
    existing_gifs: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for output in existing_outputs:
        action = str(output.get("action") or "")
        filename = str(output.get("filename") or "")
        if (
            output.get("type") != "transparent_gif"
            or action not in actions
            or not filename
            or Path(filename).name != filename
        ):
            continue
        output_path = job_dir / filename
        existing_action_timing = subject_replacement_action_output_timing(
            effective,
            action,
        )
        output_timing = existing_action_timing["transparent_gif"]
        try:
            metrics = _inspect_recoverable_gif(
                output_path,
                size=TRANSPARENT_GIF_SIZE,
                frame_count=int(output_timing["frames"]),
                fps=int(output_timing["fps"]),
            )
        except (OSError, RuntimeError, UnidentifiedImageError):
            LOGGER.warning(
                "Ignoring invalid existing GIF while recovering %s action %s: %s",
                job_id,
                action,
                output_path,
                exc_info=True,
            )
            continue
        existing_gifs[action] = (output, metrics)
    results_by_action = {
        str(result.get("action")): dict(result)
        for result in job.get("subject_replacement_results") or []
        if isinstance(result, dict) and str(result.get("action") or "") in actions
    }
    recovered_actions: list[str] = []
    recovery_sources: dict[str, dict[str, Any]] = {}
    for index, action in enumerate(actions, start=1):
        if action in existing_gifs:
            existing_output, existing_metrics = existing_gifs[action]
            results_by_action.setdefault(
                action,
                {
                    "action": action,
                    "action_label": SUBJECT_REPLACEMENT_ACTION_LABELS.get(
                        action,
                        action,
                    ),
                    "reference": None,
                    "video": None,
                    "transparent_gif": str(existing_output["filename"]),
                    "transparent_metrics": existing_metrics,
                    "output_playback": existing_action_timing,
                    "degraded": bool(existing_output.get("degraded")),
                    "preserved_during_terminal_recovery": True,
                },
            )
            continue
        source_path, source_role, source_image = _load_recoverable_reference(
            _recoverable_reference_candidates(job_dir, job, action, index)
        )
        try:
            fallback_reference = job_dir / f"{action}_degraded_fallback_reference.png"
            normalization = _normalize_recoverable_reference(
                source_image,
                fallback_reference,
                size=TRANSPARENT_GIF_SIZE,
            )
        finally:
            source_image.close()
        gif_target = job_dir / f"{action}_subject_replacement_transparent_512.gif"
        action_timing = subject_replacement_action_output_timing(
            effective,
            action,
        )
        gif_timing = action_timing["transparent_gif"]
        animation = _write_recoverable_breathing_gif(
            fallback_reference,
            gif_target,
            size=TRANSPARENT_GIF_SIZE,
            frame_count=int(gif_timing["frames"]),
            fps=int(gif_timing["fps"]),
        )
        action_label = SUBJECT_REPLACEMENT_ACTION_LABELS.get(action, action)
        append_output(
            job_id,
            {
                "type": "degraded_fallback_reference",
                "action": action,
                "action_label": action_label,
                "filename": fallback_reference.name,
                "width": TRANSPARENT_GIF_SIZE,
                "height": TRANSPARENT_GIF_SIZE,
                "transparent": True,
                "sha256": normalization["sha256"],
                "fallback_source": str(source_path),
                "fallback_source_role": source_role,
                "primary": False,
            },
        )
        append_output(
            job_id,
            {
                "type": "transparent_gif",
                "action": action,
                "action_label": action_label,
                "filename": gif_target.name,
                "width": TRANSPARENT_GIF_SIZE,
                "height": TRANSPARENT_GIF_SIZE,
                "frames": int(gif_timing["frames"]),
                "fps": int(gif_timing["fps"]),
                "duration_seconds": float(gif_timing["duration_seconds"]),
                "output_playback": action_timing,
                "transparent": True,
                "video_engine": RECOVERABLE_OUTPUT_FALLBACK_POLICY,
                "degraded": True,
                "output_exists": True,
                "generation_succeeded": False,
                "quality_passed": False,
                "sha256": animation["sha256"],
                "fallback_source": str(source_path),
                "fallback_source_role": source_role,
                "primary": True,
            },
        )
        results_by_action[action] = {
            "action": action,
            "action_label": action_label,
            "reference": str(fallback_reference),
            "reference_sha256": normalization["sha256"],
            "video": None,
            "transparent_gif": gif_target.name,
            "transparent_metrics": animation,
            "output_playback": action_timing,
            "degraded": True,
            "output_exists": True,
            "generation_succeeded": False,
            "quality_passed": False,
            "degraded_policy": RECOVERABLE_OUTPUT_FALLBACK_POLICY,
            "fallback_source": str(source_path),
            "fallback_source_role": source_role,
            "original_error": f"{type(error).__name__}: {str(error)[:1000]}",
        }
        recovered_actions.append(action)
        recovery_sources[action] = {
            "path": str(source_path),
            "role": source_role,
            "normalization": normalization,
            "animation": animation,
        }

    refreshed = get_job(job_id)
    produced_actions: set[str] = set()
    for output in refreshed.get("outputs") or []:
        if not isinstance(output, dict) or output.get("type") != "transparent_gif":
            continue
        action = str(output.get("action") or "")
        filename = str(output.get("filename") or "")
        if (
            action not in actions
            or not filename
            or Path(filename).name != filename
        ):
            continue
        try:
            output_timing = subject_replacement_action_output_timing(
                effective,
                action,
            )["transparent_gif"]
            _inspect_recoverable_gif(
                job_dir / filename,
                size=TRANSPARENT_GIF_SIZE,
                frame_count=int(output_timing["frames"]),
                fps=int(output_timing["fps"]),
            )
        except (OSError, RuntimeError, UnidentifiedImageError):
            continue
        produced_actions.add(action)
    missing_actions = [action for action in actions if action not in produced_actions]
    if missing_actions:
        raise RuntimeError(
            "Degraded output fallback did not produce every requested action: "
            + ", ".join(missing_actions)
        )
    append_recovery_event(
        job_id,
        {
            "stage": "terminal_output_recovery",
            "reason": f"{type(error).__name__}: {str(error)[:1000]}",
            "strategy": RECOVERABLE_OUTPUT_FALLBACK_POLICY,
            "result": "accepted",
            "actions": recovered_actions,
            "sources": recovery_sources,
        },
    )
    fallback_source_roles = {
        action: str(source.get("role") or "")
        for action, source in recovery_sources.items()
    }
    upload_anchor_roles = {"original_upload", "normalized_upload"}
    upload_identity_fallback = bool(fallback_source_roles) and all(
        role in upload_anchor_roles for role in fallback_source_roles.values()
    )
    original_error = f"{type(error).__name__}: {str(error)[:1000]}"
    public_error_summary = " ".join(original_error.split())[:320]
    recovery_message = (
        f"主生成链路发生异常（{public_error_summary}）。系统已使用原始上传图"
        "生成身份锚定的降级循环 GIF；动作幅度可能受限，请按需重试高质量生成。"
        if upload_identity_fallback
        else (
            f"主生成链路发生异常（{public_error_summary}）。系统已使用当前可用身份参考"
            "生成降级循环 GIF；动作幅度或身份一致性可能受限，请按需重试。"
        )
    )
    record_nonblocking_job_anomaly(
        job_id,
        code="degraded_output_fallback",
        message=recovery_message,
        stage="completed_with_degraded_output",
        details={
            "policy": RECOVERABLE_OUTPUT_FALLBACK_POLICY,
            "exception_type": type(error).__name__,
            "error": str(error)[:1000],
            "actions": actions,
            "recovered_actions": recovered_actions,
            "fallback_source_roles": fallback_source_roles,
            "upload_identity_fallback": upload_identity_fallback,
        },
        expose_as_public_stage=True,
    )
    update_job(
        job_id,
        status="completed",
        stage="completed_with_degraded_output",
        progress=100,
        current_action=None,
        current_action_label=None,
        current_action_index=None,
        current_pose_index=None,
        current_pose=None,
        recovery_active=False,
        recovery_attempt=None,
        quality_retry=None,
        active_prompt_id=None,
        automatic_recovery_used=True,
        recovered=True,
        degraded_recovery=True,
        degraded_output_fallback=True,
        degraded_output_fallback_policy=RECOVERABLE_OUTPUT_FALLBACK_POLICY,
        degraded_output_fallback_source_roles=fallback_source_roles,
        degraded_output_upload_identity_anchored=upload_identity_fallback,
        transparent_outputs_complete=True,
        transparent_outputs_quality_passed=False,
        degraded_output_actions=actions,
        original_generation_error=f"{type(error).__name__}: {str(error)[:4000]}",
        subject_replacement_results=[
            results_by_action[action]
            for action in actions
            if action in results_by_action
        ],
        action_quality_summary={
            action: {
                "output_exists": True,
                "generation_succeeded": bool(
                    results_by_action.get(action, {}).get(
                        "generation_succeeded",
                        not results_by_action.get(action, {}).get("degraded", False),
                    )
                ),
                "quality_passed": bool(
                    results_by_action.get(action, {}).get(
                        "quality_passed",
                        not results_by_action.get(action, {}).get("degraded", False),
                    )
                ),
                "degraded": bool(
                    results_by_action.get(action, {}).get("degraded", False)
                ),
            }
            for action in actions
            if action in results_by_action
        },
        reference_generation_status=(
            "completed"
            if job.get("reference_generation_completed")
            else "degraded_output_fallback"
        ),
        reference_generation_completed=bool(
            job.get("reference_generation_completed")
        ),
        error=nonblocking_result_error(job_id),
        finished_at=utc_now(),
    )
    return True


def run_profile_standard_direct_job(job_id: str) -> None:
    """Publish the approved Main_case standard without Qwen/SCAIL/GPU redraw."""
    _sync_local_video_runtime(globals(), {"run_profile_standard_direct_job"})

    raise_if_cancelled(job_id)
    with JOBS_LOCK:
        job = dict(JOBS[job_id])
    job_dir = OUTPUT_ROOT / job_id
    actions = list(job["actions"])
    effective = job.get("effective_subject_config")
    if not isinstance(effective, dict) or not uses_profile_standard_direct(
        job, effective
    ):
        raise RuntimeError("Profile standard-direct eligibility was not proven.")
    if (job_dir / "input.png").exists() or (
        job_dir / "input_identity_source.png"
    ).exists():
        raise RuntimeError(
            "Profile standard-direct found uploaded identity files; refusing bypass."
        )
    direct = effective["standard_direct"]
    verify_profile_standard_direct_assets(direct, actions)
    update_job(
        job_id,
        status="running",
        stage="verifying_profile_standard_direct",
        progress=2,
        started_at=utc_now(),
        error=None,
        standard_direct_policy=direct["policy"],
        reference_pipeline=direct["policy"],
        standard_direct_revision=direct["revision"],
        standard_direct_role=direct["role"],
        effective_subject_source="profile_standard_direct",
        gpu_required=False,
        gpu_submitted=False,
        gpu_bypass_reason=direct["gpu_bypass_reason"],
        qwen_used=False,
        scail_used=False,
    )
    results: list[dict[str, Any]] = []
    total_actions = len(actions)
    for index, action in enumerate(actions, start=1):
        raise_if_cancelled(job_id)
        action_label = SUBJECT_REPLACEMENT_ACTION_LABELS[action]
        action_timing = subject_replacement_action_output_timing(effective, action)
        video_timing = action_timing["video"]
        gif_timing = action_timing["transparent_gif"]
        standard = direct["actions"][action]
        runtime_path = Path(standard["runtime"])
        source_path = Path(standard["source"])
        progress_base = 4 + (index - 1) * 94 / total_actions
        update_job(
            job_id,
            stage="copying_profile_standard_direct",
            progress=progress_base,
            current_action=action,
            current_action_label=action_label,
            current_action_index=index,
        )
        runtime_contract = validate_standard_direct_runtime_video(
            runtime_path,
            expected_frames=int(effective["frames"]),
            expected_fps=float(effective["fps"]),
        )
        if runtime_contract["sha256"] != standard["runtime_sha256"]:
            raise RuntimeError(
                f"Frozen standard_direct {action} runtime digest changed."
            )
        video_target = job_dir / f"{action}_subject_replacement.mp4"
        gif_target = (
            job_dir / f"{action}_subject_replacement_transparent_512.gif"
            if job.get("output_format") == "transparent_gif"
            else None
        )
        apng_target = (
            job_dir / f"{action}_subject_replacement_transparent_512_soft.png"
            if job.get("output_format") == "transparent_gif"
            else None
        )
        update_job(
            job_id,
            stage="matting_profile_standard_direct_outputs",
            progress=progress_base + 18 / total_actions,
        )
        matte_metrics = build_standard_direct_outputs(
            runtime_path,
            video_target,
            gif_target,
            apng_target,
            ffmpeg_bin=FFMPEG_BIN,
            size=512,
            source_fps=int(video_timing["fps"]),
            source_frame_count=int(effective["frames"]),
            gif_fps=int(gif_timing["fps"]),
            gif_frame_count=int(gif_timing["frames"]),
        )
        output_contract = validate_standard_direct_runtime_video(
            video_target,
            expected_frames=int(video_timing["frames"]),
            expected_fps=float(video_timing["fps"]),
        )

        previews = write_standard_direct_preview_assets(standard, job_dir, action)
        preview_outputs = (
            (
                "standard_direct_reference",
                previews["cutout"],
                "main_case_selected_standard_identity_and_action",
            ),
            (
                "standard_direct_selected_frame",
                previews["guide"],
                "main_case_selected_standard_frame",
            ),
            (
                "standard_direct_mask",
                previews["mask"],
                "main_case_selected_standard_identity_mask",
            ),
        )
        for output_type, path, role in preview_outputs:
            append_output(
                job_id,
                {
                    "type": output_type,
                    "action": action,
                    "action_label": action_label,
                    "filename": path.name,
                    "width": 512,
                    "height": 512,
                    "sha256": file_sha256(path),
                    "role": role,
                    "primary": False,
                },
            )

        provenance = {
            "policy": direct["policy"],
            "revision": direct["revision"],
            "role": direct["role"],
            "profile_id": effective["profile_id"],
            "profile_sha256": effective["profile_sha256"],
            "action": action,
            "source": str(source_path),
            "source_sha256": standard["source_sha256"],
            "runtime": str(runtime_path),
            "runtime_sha256": standard["runtime_sha256"],
            "output_sha256": file_sha256(video_target),
            "runtime_contract": runtime_contract,
            "output_contract": output_contract,
            "output_playback": action_timing,
            "matte": matte_metrics,
            "foreground_rgb_redrawn": False,
            "spatial_transform_applied": False,
            "output_background": "white",
            "validated_run_id": standard.get("validated_run_id"),
            "deterministic": True,
            "gpu_required": False,
            "gpu_submitted": False,
            "gpu_bypass_reason": direct["gpu_bypass_reason"],
            "qwen_used": False,
            "scail_used": False,
        }
        provenance_path = job_dir / f"{action}_standard_direct_provenance.json"
        write_json_atomic(provenance_path, provenance)
        append_output(
            job_id,
            {
                "type": "standard_direct_provenance",
                "action": action,
                "action_label": action_label,
                "filename": provenance_path.name,
                "sha256": file_sha256(provenance_path),
                "primary": False,
            },
        )
        video_output = {
            "type": "subject_replacement_video",
            "action": action,
            "action_label": action_label,
            "filename": video_target.name,
            "width": 512,
            "height": 512,
            "frames": int(video_timing["frames"]),
            "fps": float(video_timing["fps"]),
            "duration_seconds": float(video_timing["duration_seconds"]),
            "output_playback": action_timing,
            "playback_contract": output_contract,
            "video_engine": "deterministic-main-case-standard-direct",
            "driving_template": runtime_path.name,
            "sha256": file_sha256(video_target),
            "source_runtime_sha256": standard["runtime_sha256"],
            "background": "white",
            "matte_policy": matte_metrics["policy"],
            "foreground_rgb_redrawn": False,
            "spatial_transform_applied": False,
            "primary": job.get("output_format") != "transparent_gif",
        }
        if job.get("output_format") != "transparent_gif":
            append_output(job_id, video_output)

        transparent_metrics: dict[str, Any] | None = None
        if job.get("output_format") == "transparent_gif":
            update_job(
                job_id,
                stage="publishing_profile_standard_transparency",
                progress=progress_base + 60 / total_actions,
            )
            if gif_target is None or not gif_target.is_file():
                raise RuntimeError(
                    f"Standard-direct {action} transparent GIF was not produced."
                )
            apng_available = bool(
                apng_target is not None and apng_target.is_file()
            )
            transparent_metrics = matte_metrics
            metrics_path = job_dir / f"{action}_transparent_metrics.json"
            write_json_atomic(metrics_path, transparent_metrics)
            append_output(job_id, video_output)
            append_output(
                job_id,
                {
                    "type": "transparent_gif",
                    "action": action,
                    "action_label": action_label,
                    "filename": gif_target.name,
                    "width": TRANSPARENT_GIF_SIZE,
                    "height": TRANSPARENT_GIF_SIZE,
                    "frames": int(gif_timing["frames"]),
                    "fps": int(gif_timing["fps"]),
                    "duration_seconds": float(gif_timing["duration_seconds"]),
                    "output_playback": action_timing,
                    "transparent": True,
                    "video_engine": "deterministic-main-case-standard-direct",
                    "driving_template": runtime_path.name,
                    "sha256": file_sha256(gif_target),
                    "source_runtime_sha256": standard["runtime_sha256"],
                    "matte_policy": matte_metrics["policy"],
                    "foreground_rgb_redrawn": False,
                    "spatial_transform_applied": False,
                    "edge_mode": "clean_hard_alpha_no_dither",
                    "primary": not apng_available,
                },
            )
            if apng_available and apng_target is not None:
                append_output(
                    job_id,
                    {
                        "type": "transparent_apng",
                        "action": action,
                        "action_label": action_label,
                        "filename": apng_target.name,
                        "width": TRANSPARENT_GIF_SIZE,
                        "height": TRANSPARENT_GIF_SIZE,
                        "frames": TRANSPARENT_GIF_FRAMES,
                        "fps": TRANSPARENT_GIF_FPS,
                        "duration_seconds": TRANSPARENT_GIF_DURATION_SECONDS,
                        "transparent": True,
                        "soft_alpha": True,
                        "video_engine": "deterministic-main-case-standard-direct",
                        "driving_template": runtime_path.name,
                        "sha256": file_sha256(apng_target),
                        "source_runtime_sha256": standard["runtime_sha256"],
                        "matte_policy": matte_metrics["policy"],
                        "foreground_rgb_redrawn": False,
                        "spatial_transform_applied": False,
                        "recommended_for_quality": True,
                        "primary": True,
                    },
                )
        results.append(
            {
                "action": action,
                "action_label": action_label,
                "policy": direct["policy"],
                "revision": direct["revision"],
                "source": str(source_path),
                "source_sha256": standard["source_sha256"],
                "runtime": str(runtime_path),
                "runtime_sha256": standard["runtime_sha256"],
                "output_sha256": file_sha256(video_target),
                "matte_policy": matte_metrics["policy"],
                "foreground_rgb_redrawn": False,
                "spatial_transform_applied": False,
                "video": video_target.name,
                "transparent_gif": (
                    f"{action}_subject_replacement_transparent_512.gif"
                    if transparent_metrics is not None
                    else None
                ),
                "transparent_apng": (
                    f"{action}_subject_replacement_transparent_512_soft.png"
                    if transparent_metrics is not None
                    and apng_target is not None
                    and apng_target.is_file()
                    else None
                ),
                "transparent_metrics": transparent_metrics,
                "output_playback": action_timing,
                "gpu_required": False,
                "gpu_submitted": False,
            }
        )
        update_job(
            job_id,
            progress=progress_base + 94 / total_actions,
            subject_replacement_results=results,
        )
    update_job(
        job_id,
        status="completed",
        stage="completed",
        progress=100,
        current_action=None,
        current_action_label=None,
        current_action_index=None,
        subject_replacement_results=results,
        finished_at=utc_now(),
    )


def revalidate_uploaded_qwen_references_before_scail(
    job_id: str,
    job: dict[str, Any],
    effective_subject_config: dict[str, Any],
    actions: list[str],
) -> None:
    """Re-prove persisted Qwen references before any SCAIL preparation or GPU call."""
    _sync_local_video_runtime(
        globals(),
        {"revalidate_uploaded_qwen_references_before_scail"},
    )

    job_dir = OUTPUT_ROOT / job_id
    verify_uploaded_identity_submission(job, job_dir)
    degraded_reference_actions = {
        str(action)
        for action, details in dict(
            job.get("qwen_reference_degraded_actions") or {}
        ).items()
        if isinstance(details, dict)
    }
    partial_reference_stage = bool(degraded_reference_actions)
    if any(action in degraded_reference_actions for action in actions):
        raise RuntimeError(
            "A degraded Qwen action was scheduled for SCAIL instead of being "
            "isolated to its identity-anchored fallback."
        )
    if (
        job.get("subject_source") != "qwen_generated_upload"
        or job.get("reference_generation_completed") is not True
        or job.get("reference_generation_status")
        not in (
            {"completed", "completed_with_degraded_actions"}
            if partial_reference_stage
            else {"completed"}
        )
        or job.get("reference_normalization_status") != "completed"
        or job.get("canonical_reference_generated") is not True
        or job.get("canonical_reference_used") is not True
        or job.get("canonical_reference_normalized") is not True
        or job.get("canonical_reference_transparent") is not True
        or job.get("qwen_action_keyframes_normalized") is not True
        or job.get("qwen_action_keyframes_transparent") is not True
        or job.get("qwen_action_reference_normalization_policy")
        != QWEN_ACTION_REFERENCE_NORMALIZATION_POLICY
        or job.get("qwen_action_reference_byte_preserved") is not True
        or (
            GENERATION_VISUAL_AUDITS_BLOCKING
            and job.get("surface_texture_gate_status")
            not in (
                {"passed", "partial_degraded"}
                if partial_reference_stage
                else {"passed"}
            )
        )
    ):
        raise RuntimeError(
            "Persisted uploaded Qwen reference stage is incomplete; refusing "
            "to enter SCAIL."
        )

    canonical_reference = Path(str(job.get("canonical_reference_path") or ""))
    canonical_reference_sha256 = (
        str(job.get("canonical_reference_sha256") or "").strip().lower()
    )
    canonical_public_path = Path(str(job.get("canonical_public_reference_path") or ""))
    canonical_public_sha256 = (
        str(job.get("canonical_public_reference_sha256") or "").strip().lower()
    )
    canonical_qwen_image1 = Path(str(job.get("canonical_qwen_image1_path") or ""))
    canonical_qwen_image1_comfy = Path(
        str(job.get("canonical_qwen_image1_comfy_path") or "")
    )
    canonical_qwen_image1_sha256 = (
        str(job.get("canonical_qwen_image1_sha256") or "").strip().lower()
    )
    canonical_qwen_image1_comfy_sha256 = (
        str(job.get("canonical_qwen_image1_comfy_sha256") or "").strip().lower()
    )
    uploaded_identity_source = job_dir / "input_identity_source.png"
    uploaded_identity_source_sha256 = (
        str(job.get("uploaded_identity_source_sha256") or "").strip().lower()
    )
    if (
        canonical_reference != job_dir / "cartoon_reference_transparent.png"
        or canonical_public_path != canonical_reference
        or not re.fullmatch(r"[0-9a-f]{64}", canonical_reference_sha256)
        or canonical_public_sha256 != canonical_reference_sha256
        or not canonical_reference.is_file()
        or file_sha256(canonical_reference) != canonical_reference_sha256
        or canonical_qwen_image1 != job_dir / "cartoon_reference.png"
        or canonical_qwen_image1_comfy
        != COMFY_INPUT / "local_video_api" / job_id / "cartoon_reference.png"
        or not re.fullmatch(r"[0-9a-f]{64}", canonical_qwen_image1_sha256)
        or canonical_qwen_image1_comfy_sha256 != canonical_qwen_image1_sha256
        or not canonical_qwen_image1.is_file()
        or not canonical_qwen_image1_comfy.is_file()
        or file_sha256(canonical_qwen_image1) != canonical_qwen_image1_sha256
        or file_sha256(canonical_qwen_image1_comfy) != canonical_qwen_image1_sha256
    ):
        raise RuntimeError(
            "Persisted Qwen full-body canonical provenance changed before SCAIL."
        )
    with Image.open(canonical_reference) as loaded_canonical:
        canonical_alpha = loaded_canonical.convert("RGBA").getchannel("A")
        canonical_alpha_minimum, canonical_alpha_maximum = canonical_alpha.getextrema()
        if (
            loaded_canonical.size != (TRANSPARENT_GIF_SIZE, TRANSPARENT_GIF_SIZE)
            or "A" not in loaded_canonical.getbands()
            or canonical_alpha_minimum == 255
            or canonical_alpha_maximum == 0
        ):
            raise RuntimeError(
                "Persisted Qwen full-body canonical is not a usable 512x512 "
                "transparent reference."
            )

    reference_paths = job.get("subject_reference_paths")
    qwen_reference_sha256s = job.get("qwen_action_keyframe_sha256s")
    raw_canonical_identity_fallbacks = job.get(
        "qwen_canonical_identity_fallbacks"
    )
    if raw_canonical_identity_fallbacks is None:
        canonical_identity_fallbacks: dict[str, Any] = {}
    elif isinstance(raw_canonical_identity_fallbacks, dict):
        canonical_identity_fallbacks = raw_canonical_identity_fallbacks
    else:
        raise RuntimeError(
            "Persisted Qwen canonical fallback provenance is invalid."
        )
    fur_surface_status = job.get("fur_surface_gate_status")
    selected_surface_metrics = job.get("fur_surface_gate_selected_metrics")
    stored_surface_quality = job.get("subject_fur_surface_quality")
    final_surface_quality = job.get("fur_surface_gate_final_quality")
    advisory_grooming_minimum_status = job.get(
        "grooming_advisory_minimum_gate_status"
    )
    advisory_grooming_minimum_selected = job.get(
        "grooming_advisory_minimum_gate_selected"
    )
    surface_provenance_complete = bool(
        isinstance(fur_surface_status, dict)
        and isinstance(selected_surface_metrics, dict)
        and isinstance(stored_surface_quality, dict)
        and isinstance(final_surface_quality, dict)
        and final_surface_quality.get("status")
        in (
            {"passed", "partial_degraded"}
            if partial_reference_stage
            else {"passed"}
        )
    )
    if (
        not isinstance(reference_paths, dict)
        or not isinstance(qwen_reference_sha256s, dict)
        or list(job.get("qwen_reference_generated_actions") or []) != actions
        or (
            GENERATION_VISUAL_AUDITS_BLOCKING
            and not surface_provenance_complete
        )
    ):
        raise RuntimeError(
            "Persisted uploaded Qwen per-action quality provenance is incomplete."
        )

    fresh_surface_quality: dict[str, Any] = {}
    for action in actions:
        if GENERATION_VISUAL_AUDITS_BLOCKING:
            require_uploaded_fur_surface_gate(
                effective_subject_config["actions"][action],
                action,
            )
        reference_path = Path(str(reference_paths.get(action) or ""))
        reference_digest = (
            file_sha256(reference_path) if reference_path.is_file() else ""
        )
        action_config = effective_subject_config["actions"][action]
        stored_canonical_fallback = canonical_identity_fallbacks.get(action)
        configured_canonical_fallback = action_config.get(
            "qwen_canonical_identity_fallback"
        )
        if stored_canonical_fallback != configured_canonical_fallback:
            raise RuntimeError(
                f"Persisted Qwen {action} canonical fallback provenance changed "
                "before SCAIL."
            )
        expected_reference_digest = (
            str(action_config.get("reference_sha256") or "").strip().lower()
        )
        persisted_qwen_digest = (
            str(qwen_reference_sha256s.get(action) or "").strip().lower()
        )
        if (
            action_config.get("reference_kind") != "qwen_generated_upload"
            or Path(str(action_config.get("reference") or "")) != reference_path
            or not re.fullmatch(r"[0-9a-f]{64}", reference_digest)
            or expected_reference_digest != reference_digest
            or persisted_qwen_digest != reference_digest
            or (
                GENERATION_VISUAL_AUDITS_BLOCKING
                and (
                    not isinstance(fur_surface_status, dict)
                    or fur_surface_status.get(action) != "passed"
                )
            )
        ):
            raise RuntimeError(
                f"Persisted Qwen {action} reference changed before SCAIL."
            )
        scail_reference_mode = action_config.get("scail_reference_mode")
        if scail_reference_mode is not None:
            if (
                action != "eating"
                or str(job.get("species") or "").strip().lower() != "dog"
                or scail_reference_mode
                != DOG_EATING_CANONICAL_FULL_DRIVER_POSE_POLICY
                or not isinstance(stored_canonical_fallback, dict)
                or stored_canonical_fallback.get("policy")
                != DOG_EATING_CANONICAL_IDENTITY_FALLBACK_POLICY
                or stored_canonical_fallback.get(
                    "action_geometry_deferred_to_scail"
                )
                is not True
                or stored_canonical_fallback.get(
                    "published_reference_path"
                )
                != str(reference_path)
                or stored_canonical_fallback.get(
                    "published_reference_sha256"
                )
                != reference_digest
            ):
                raise RuntimeError(
                    "Persisted Qwen dog-eating identity-only fallback "
                    "contract changed before SCAIL."
                )
        with Image.open(reference_path) as loaded_reference:
            alpha = loaded_reference.convert("RGBA").getchannel("A")
            alpha_minimum, alpha_maximum = alpha.getextrema()
            if (
                loaded_reference.size != (TRANSPARENT_GIF_SIZE, TRANSPARENT_GIF_SIZE)
                or "A" not in loaded_reference.getbands()
                or alpha_minimum == 255
                or alpha_maximum == 0
            ):
                raise RuntimeError(
                    f"Persisted Qwen {action} reference is not a usable "
                    "512x512 transparent cutout."
                )

        if not GENERATION_VISUAL_AUDITS_BLOCKING:
            if action == "grooming":
                selected_minimum = (
                    advisory_grooming_minimum_selected.get(action)
                    if isinstance(advisory_grooming_minimum_selected, dict)
                    else None
                )
                selected_candidate = Path(
                    str(
                        selected_minimum.get("selected_candidate_path")
                        if isinstance(selected_minimum, dict)
                        else ""
                    )
                )
                selected_candidate_sha256 = str(
                    selected_minimum.get("selected_candidate_sha256")
                    if isinstance(selected_minimum, dict)
                    else ""
                ).strip().lower()
                minimum_identity_reference = Path(
                    str(
                        selected_minimum.get("identity_reference_path")
                        if isinstance(selected_minimum, dict)
                        else ""
                    )
                )
                minimum_palette_reference = Path(
                    str(
                        selected_minimum.get(
                            "identity_palette_reference_path"
                        )
                        if isinstance(selected_minimum, dict)
                        else ""
                    )
                )
                minimum_palette_reference_sha256 = str(
                    selected_minimum.get(
                        "identity_palette_reference_sha256"
                    )
                    if isinstance(selected_minimum, dict)
                    else ""
                ).strip().lower()
                uploaded_identity_anchor = Path(
                    str(job.get("uploaded_identity_anchor_path") or "")
                )
                uploaded_identity_anchor_sha256 = str(
                    job.get("uploaded_identity_source_sha256") or ""
                ).strip().lower()
                if (
                    advisory_grooming_minimum_status
                    not in {"passed", "degraded_passed"}
                    or not isinstance(selected_minimum, dict)
                    or selected_minimum.get("policy")
                    != GROOMING_ADVISORY_MINIMUM_GATE_POLICY
                    or selected_minimum.get("version")
                    != GROOMING_ADVISORY_MINIMUM_GATE_VERSION
                    or (
                        selected_minimum.get("acceptable") is not True
                        and selected_minimum.get(
                            "degraded_fallback_accepted"
                        )
                        is not True
                    )
                    or selected_minimum.get("candidate_sha256")
                    != selected_candidate_sha256
                    or not selected_candidate.is_file()
                    or file_sha256(selected_candidate)
                    != selected_candidate_sha256
                    or selected_candidate_sha256 != reference_digest
                    or minimum_identity_reference != canonical_qwen_image1
                    or not minimum_identity_reference.is_file()
                    or file_sha256(minimum_identity_reference)
                    != canonical_qwen_image1_sha256
                    or selected_minimum.get("identity_reference_sha256")
                    != canonical_qwen_image1_sha256
                    or minimum_palette_reference != uploaded_identity_anchor
                    or not minimum_palette_reference.is_file()
                    or not re.fullmatch(
                        r"[0-9a-f]{64}",
                        minimum_palette_reference_sha256,
                    )
                    or minimum_palette_reference_sha256
                    != uploaded_identity_anchor_sha256
                    or file_sha256(minimum_palette_reference)
                    != minimum_palette_reference_sha256
                ):
                    raise RuntimeError(
                        "Persisted Qwen grooming reference lost its minimum "
                        "action/identity contract before SCAIL."
                    )
            continue

        selected_surface = selected_surface_metrics.get(action)
        selected_gate = (
            selected_surface.get("gate") if isinstance(selected_surface, dict) else None
        )
        selected_candidate = Path(
            str(
                selected_surface.get("selected_candidate_path")
                if isinstance(selected_surface, dict)
                else ""
            )
        )
        selected_candidate_sha256 = (
            str(
                selected_surface.get("selected_candidate_sha256")
                if isinstance(selected_surface, dict)
                else ""
            )
            .strip()
            .lower()
        )
        stored_surface = stored_surface_quality.get(action)
        if (
            not isinstance(selected_surface, dict)
            or selected_surface.get("acceptable") is not True
            or selected_surface.get("status") != "passed"
            or not isinstance(selected_gate, dict)
            or selected_gate.get("policy") != FUR_SURFACE_GATE_POLICY
            or selected_gate.get("gate_version") != FUR_SURFACE_GATE_VERSION
            or selected_gate.get("threshold_version") != FUR_SURFACE_THRESHOLD_VERSION
            or selected_gate.get("action") != action
            or not selected_candidate.is_file()
            or not re.fullmatch(r"[0-9a-f]{64}", selected_candidate_sha256)
            or selected_gate.get("candidate_sha256") != selected_candidate_sha256
            or file_sha256(selected_candidate) != selected_candidate_sha256
            or selected_candidate_sha256 != reference_digest
            or not isinstance(stored_surface, dict)
            or stored_surface.get("acceptable") is not True
            or stored_surface.get("pre_scail_revalidated") is not True
            or stored_surface.get("normalized_candidate_sha256") != reference_digest
        ):
            raise RuntimeError(
                f"Persisted Qwen {action} fur surface provenance changed before SCAIL."
            )
        fresh_surface = evaluate_main_case_fur_surface_candidate(
            reference_path,
            action=action,
            alpha_provenance=str(selected_gate.get("alpha_provenance") or ""),
            effective_subject_config=effective_subject_config,
        )
        if fresh_surface.get("acceptable") is not True:
            raise RuntimeError(
                f"Persisted Qwen {action} reference failed fresh fur surface "
                "validation before SCAIL: "
                + ", ".join(str(issue) for issue in fresh_surface.get("issues", []))
            )
        fresh_surface["selected_attempt"] = selected_surface.get("selected_attempt")
        fresh_surface["source_candidate_sha256"] = selected_candidate_sha256
        fresh_surface["normalized_candidate_sha256"] = reference_digest
        fresh_surface["persisted_reference_path"] = str(reference_path)
        fresh_surface["persisted_reference_sha256"] = reference_digest
        fresh_surface["byte_identical_to_selected_candidate"] = True
        fresh_surface["pre_scail_revalidated"] = True
        fresh_surface_quality[action] = fresh_surface

    update_job(
        job_id,
        subject_fur_surface_quality=fresh_surface_quality,
        generation_audit_policy=GENERATION_AUDIT_POLICY,
        visual_content_audit_mode=(
            "blocking" if GENERATION_VISUAL_AUDITS_BLOCKING else "advisory"
        ),
        qwen_reference_pre_scail_visual_audit_status=(
            "passed"
            if GENERATION_VISUAL_AUDITS_BLOCKING
            else (
                "minimum_contract_passed"
                if "grooming" in actions
                else "skipped_advisory"
            )
        ),
        qwen_reference_pre_scail_minimum_contract_status=(
            (
                "degraded_passed"
                if advisory_grooming_minimum_status == "degraded_passed"
                else "passed"
            )
            if not GENERATION_VISUAL_AUDITS_BLOCKING and "grooming" in actions
            else "not_applicable"
        ),
        qwen_reference_pre_scail_revalidated=True,
        qwen_reference_pre_scail_revalidated_at=utc_now(),
        qwen_reference_pre_scail_sha256s={
            action: qwen_reference_sha256s[action] for action in actions
        },
    )


def run_subject_replacement_job(job_id: str) -> None:
    _sync_local_video_runtime(globals(), {"run_subject_replacement_job"})
    raise_if_cancelled(job_id)
    with JOBS_LOCK:
        job = dict(JOBS[job_id])
    job_dir = OUTPUT_ROOT / job_id
    actions = list(job["actions"])
    initial_subject_source = str(job.get("subject_source") or "")
    if initial_subject_source in {
        "upload_pending_scail_reference",
        "uploaded_identity_direct_scail",
    }:
        raise RuntimeError(
            "This upload job was queued under the retired direct-SCAIL policy "
            "and cannot be resumed safely. Submit the upload again so Qwen can "
            "generate the required full-body canonical and action keyframes."
        )
    uploaded_job = initial_subject_source in {
        "upload",
        "upload_pending_qwen",
        "qwen_generated_upload",
    }
    effective_subject_config = job.get("effective_subject_config")
    profile: dict[str, Any] | None = None
    if not isinstance(effective_subject_config, dict):
        if uploaded_job:
            raise RuntimeError(
                "Uploaded identity job lacks its queue-time effective subject "
                "config. Submit the upload again instead of reading a mutable "
                "live Profile."
            )
        profile = load_subject_profile(job.get("subject_profile"))
        effective_subject_config = resolve_effective_subject_config(
            profile,
            actions,
            seed=job.get("seed"),
            scail_steps=job.get("scail_steps"),
            pose_strength=job.get("pose_strength"),
        )
    if uses_profile_standard_direct(job, effective_subject_config):
        run_profile_standard_direct_job(job_id)
        return
    expected_pose_policies = {
        str(action): str(policy)
        for action, policy in dict(job.get("expected_pose_policies") or {}).items()
        if str(action) in actions and str(policy).strip()
    }
    if uploaded_job:
        frozen_upload_validation = effective_subject_config.get("upload_validation")
        if (
            not isinstance(frozen_upload_validation, dict)
            or frozen_upload_validation.get("policy")
            != QWEN_FULL_BODY_FIRST_FRAME_PIPELINE
            or frozen_upload_validation.get("qwen_full_body_required") is not True
            or frozen_upload_validation.get("qwen_action_keyframes_required")
            is not True
            or frozen_upload_validation.get(
                "normalized_transparent_references_required"
            )
            is not True
            or (
                GENERATION_VISUAL_AUDITS_BLOCKING
                and frozen_upload_validation.get("expected_surface_gate_actions")
                != actions
            )
            or frozen_upload_validation.get("expected_pose_policies")
            != expected_pose_policies
        ):
            raise RuntimeError(
                "Uploaded identity job lacks its frozen Qwen full-body and "
                "action-keyframe validation policy."
            )
        if GENERATION_VISUAL_AUDITS_BLOCKING:
            for action in actions:
                require_uploaded_fur_surface_gate(
                    effective_subject_config["actions"][action],
                    action,
                )
    frozen_actions, upload_reference_actions = partition_subject_reference_actions(
        effective_subject_config,
        actions,
        uploaded_identity=uploaded_job,
    )
    if initial_subject_source in {
        "upload",
        "upload_pending_qwen",
    }:
        if not upload_reference_actions:
            raise RuntimeError(
                "Uploaded subject identity has no action reference work; "
                "refusing to fall back to the default profile."
            )
        prepare_uploaded_subject_qwen_references(
            job_id,
            upload_reference_actions,
        )
        job = get_job(job_id)
        effective_subject_config = job.get("effective_subject_config")
        if not isinstance(effective_subject_config, dict):
            raise RuntimeError(
                "Uploaded identity reference stage lost the effective subject config."
            )
    degraded_reference_actions = {
        str(action): dict(details)
        for action, details in dict(
            job.get("qwen_reference_degraded_actions") or {}
        ).items()
        if str(action) in actions and isinstance(details, dict)
    }
    if uploaded_job and GENERATION_VISUAL_AUDITS_BLOCKING:
        refreshed_upload_validation = effective_subject_config.get("upload_validation")
        if (
            not isinstance(refreshed_upload_validation, dict)
            or refreshed_upload_validation.get("expected_surface_gate_actions")
            != actions
        ):
            raise RuntimeError(
                "Uploaded identity reference stage lost its complete per-action "
                "fur surface contract."
            )
        for action in actions:
            if action in degraded_reference_actions:
                continue
            require_uploaded_fur_surface_gate(
                effective_subject_config["actions"][action],
                action,
            )
    scail_actions = [
        action for action in actions if action not in degraded_reference_actions
    ]

    generated_reference_paths: dict[str, str] | None = None
    if job.get("subject_source") == "qwen_generated_upload":
        raw_generated_paths = job.get("subject_reference_paths")
        if not isinstance(raw_generated_paths, dict):
            raise RuntimeError("Persisted uploaded identity references are invalid.")
        generated_reference_paths = {
            str(reference_action): str(reference_path)
            for reference_action, reference_path in raw_generated_paths.items()
            if str(reference_path).strip()
        }
        if scail_actions:
            revalidate_uploaded_qwen_references_before_scail(
                job_id,
                job,
                effective_subject_config,
                scail_actions,
            )
        job = get_job(job_id)
        refreshed_effective_subject_config = job.get("effective_subject_config")
        if not isinstance(refreshed_effective_subject_config, dict):
            raise RuntimeError(
                "Pre-SCAIL Qwen reference validation lost the effective subject config."
            )
        effective_subject_config = refreshed_effective_subject_config

    selected_references = {
        action: select_scail_subject_reference(
            job,
            profile,
            generated_reference_paths,
            action,
        )
        for action in scail_actions
    }
    effective_reference_paths = {
        action: str(selected_references[action][0]) for action in scail_actions
    }
    selected_reference_policies = {
        action: selected_references[action][1] for action in scail_actions
    }
    uploaded_reference_policies = {
        "qwen_action_identity_with_bowl_mapped_red",
        "qwen_action_specific_identity",
    }
    uploaded_applied_actions = [
        action
        for action, policy in selected_reference_policies.items()
        if policy in uploaded_reference_policies
    ]
    if uploaded_job:
        missing_uploaded_actions = [
            action
            for action in scail_actions
            if action not in uploaded_applied_actions
        ]
        if missing_uploaded_actions:
            raise RuntimeError(
                "Uploaded subject identity was not applied to every requested "
                "action; refusing to run SCAIL with the default profile: "
                + ", ".join(missing_uploaded_actions)
            )
    if frozen_actions and uploaded_applied_actions:
        effective_reference_pipeline = SCAIL_MIXED_REFERENCE_PIPELINE
    elif frozen_actions:
        effective_reference_pipeline = SCAIL_FROZEN_EATING_PIPELINE
    else:
        effective_reference_pipeline = str(
            job.get("reference_pipeline") or QWEN_FULL_BODY_FIRST_FRAME_PIPELINE
        )
    job = update_job(
        job_id,
        subject_reference_paths=effective_reference_paths,
        qwen_subject_reference_paths=(
            generated_reference_paths or {}
            if job.get("subject_source") == "qwen_generated_upload"
            else {}
        ),
        scail_reference_policy=selected_reference_policies,
        reference_pipeline=effective_reference_pipeline,
        qwen_reference_generated_actions=(
            list((generated_reference_paths or {}).keys())
            if job.get("subject_source") == "qwen_generated_upload"
            else []
        ),
        qwen_reference_applied_actions=(
            uploaded_applied_actions
            if job.get("subject_source") == "qwen_generated_upload"
            else []
        ),
        qwen_reference_bypassed_actions=list(frozen_actions),
        uploaded_identity_applied_actions=(
            uploaded_applied_actions if uploaded_job else []
        ),
        uploaded_identity_excluded_actions=[],
        uploaded_identity_reference_actions=(
            list((generated_reference_paths or {}).keys())
            if job.get("subject_source") == "qwen_generated_upload"
            else []
        ),
        effective_subject_source=(
            str(job.get("subject_source")) if uploaded_job else "profile"
        ),
    )
    if scail_actions:
        require_subject_replacement_assets(
            profile or {"action_references": effective_reference_paths},
            scail_actions,
            reference_paths=effective_reference_paths,
            effective_config=effective_subject_config,
            expected_pose_policies={
                action: policy
                for action, policy in expected_pose_policies.items()
                if action in scail_actions
            },
        )
    total_actions = len(actions)
    results: list[dict[str, Any]] = [
        dict(result)
        for result in job.get("subject_replacement_results") or []
        if isinstance(result, dict)
        and str(result.get("action") or "") in degraded_reference_actions
    ]
    final_video_fur_surface_by_action: dict[str, Any] = {}
    scail_reference_policies = dict(job.get("scail_reference_policy") or {})
    scail_scene_prop_policies = dict(job.get("scail_scene_prop_policy") or {})
    stored_transparent_guidance_policies = job.get(
        "transparent_guidance_policy"
    )
    transparent_guidance_policies = (
        dict(stored_transparent_guidance_policies)
        if isinstance(stored_transparent_guidance_policies, dict)
        else {}
    )
    scail_video_attempts_by_action: dict[str, list[dict[str, Any]]] = {}
    selected_scail_seeds: dict[str, int] = {}
    reference_generation_chain = bool(job.get("reference_generation_completed"))
    progress_base = 38.0 if reference_generation_chain else 3.0
    progress_total = 59.0 if reference_generation_chain else 94.0
    # Dog eating has the strictest prop and contact path.  Run independent dog
    # actions first so a later eating failure cannot discard an already valid
    # standing animation; terminal recovery will preserve that output and fill
    # only the failed/missing action.
    execution_actions = list(scail_actions)
    if str(job.get("species") or "").strip().lower() == "dog":
        execution_actions.sort(key=lambda action: action == "eating")

    update_job(
        job_id,
        status="running",
        stage=(
            "waiting_for_gpu"
            if execution_actions
            else "finalizing_degraded_actions"
        ),
        progress=36 if reference_generation_chain else 1,
        started_at=job.get("started_at") or utc_now(),
        error=None,
        current_action=None,
        subject_replacement_execution_order=execution_actions,
        subject_replacement_failure_isolation_policy=(
            "per_action_reference_and_output_isolation_v2"
            if degraded_reference_actions
            else (
                "independent_dog_actions_before_strict_eating_v1"
                if execution_actions != actions
                else "requested_action_order_v1"
            )
        ),
    )
    if execution_actions:
        wait_comfy_idle(timeout=21600, job_id=job_id)
        raise_if_cancelled(job_id)
        free_comfy_models()

    for index, action in enumerate(execution_actions, start=1):
        raise_if_cancelled(job_id)
        action_config = effective_subject_config["actions"][action]
        action_timing = subject_replacement_action_output_timing(
            effective_subject_config,
            action,
        )
        video_timing = action_timing["video"]
        gif_timing = action_timing["transparent_gif"]
        driving_animation_path = Path(action_config["driving_animation"])
        expected_driving_sha256 = str(
            action_config.get("driving_animation_sha256") or ""
        ).strip()
        if (
            expected_driving_sha256
            and file_sha256(driving_animation_path) != expected_driving_sha256
        ):
            raise RuntimeError(f"Frozen {action} driving animation digest changed.")
        if (
            action == "eating"
            and float(action_config["pose_strength"]) > SCAIL_EATING_POSE_STRENGTH_MAX
        ):
            raise RuntimeError(
                "Persisted eating pose_strength exceeds the safe fur-silhouette "
                f"limit {SCAIL_EATING_POSE_STRENGTH_MAX:.2f}."
            )
        action_label = SUBJECT_REPLACEMENT_ACTION_LABELS[action]
        action_start = progress_base + (index - 1) * progress_total / total_actions
        action_span = progress_total / total_actions
        reference_path, reference_policy = selected_references[action]
        if not reference_path.is_file():
            raise FileNotFoundError(
                f"SCAIL {action} reference is missing: {reference_path}"
            )
        run_id = available_scail_run_id(job_id, action)
        experiment_dir = SCAIL_EXPERIMENT_ROOT / run_id
        prepare_log = job_dir / f"{action}_prepare.log"
        update_job(
            job_id,
            stage="preparing_subject_replacement",
            progress=action_start,
            current_action=action,
            current_action_label=action_label,
            current_action_index=index,
        )
        command = [
            str(SCAIL_PYTHON),
            str(SCAIL_RUNNER),
            "--source",
            str(action_config["driving_animation"]),
            "--reference",
            str(reference_path),
            "--action",
            action,
            "--species",
            str(job.get("species") or "cat"),
            "--width",
            "512",
            "--height",
            "512",
            "--frames",
            str(effective_subject_config["frames"]),
            "--fps",
            str(effective_subject_config["fps"]),
            "--steps",
            str(effective_subject_config["steps"]),
            "--seed",
            str(action_config["effective_seed"]),
            "--pose-strength",
            str(action_config["pose_strength"]),
            "--run-id",
            run_id,
            "--prepare-only",
        ]
        canonical_full_driver_eating_fallback = bool(
            action == "eating"
            and str(job.get("species") or "").strip().lower() == "dog"
            and action_config.get("scail_reference_mode")
            == DOG_EATING_CANONICAL_FULL_DRIVER_POSE_POLICY
        )
        if canonical_full_driver_eating_fallback:
            command.append(
                "--canonical-identity-full-driver-eating-fallback"
            )
        # Subject-replacement identity, fur, action, camera, and prop rules are
        # profile-locked. User prose is retained in task metadata but is never
        # appended after the protected SCAIL prompt, where terms such as
        # "fluffy" can otherwise override the close-lying fur standard.
        prompt_suffix = str(job.get("prompt_en") or job.get("prompt") or "").strip()
        if prompt_suffix:
            update_job(
                job_id,
                user_prompt_excluded_from_scail=True,
                scail_prompt_source="protected_action_profile",
            )
        if job.get("subject_source") == "qwen_generated_upload":
            # The queue wait above can last hours, and earlier actions can also
            # keep later references waiting.  Re-prove the immutable upload,
            # canonical, per-action digest, alpha, and fur-surface contract at
            # the last possible point before this SCAIL process reads it.
            just_in_time_job = get_job(job_id)
            just_in_time_config = just_in_time_job.get("effective_subject_config")
            if not isinstance(just_in_time_config, dict):
                raise RuntimeError(
                    "Pre-SCAIL Qwen reference validation lost the effective "
                    "subject config."
                )
            revalidate_uploaded_qwen_references_before_scail(
                job_id,
                just_in_time_job,
                just_in_time_config,
                actions,
            )
            just_in_time_reference = Path(
                str(just_in_time_config["actions"][action].get("reference") or "")
            )
            if just_in_time_reference != reference_path:
                raise RuntimeError(
                    f"Persisted Qwen {action} reference selection changed before SCAIL."
                )
        run_cancelable_process(
            job_id,
            command,
            log_path=prepare_log,
            stage_name=f"{action} SCAIL input preparation",
            timeout=900,
        )
        workflow_path = experiment_dir / "workflow_api.json"
        manifest_path = experiment_dir / "manifest.json"
        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_input_files = manifest.get("input_files")
        if not isinstance(manifest_input_files, dict):
            raise RuntimeError(f"SCAIL {action} input manifest is incomplete")
        pose_identity_mask_video_path = Path(
            str(manifest_input_files.get("pose_mask_video") or "")
        )
        reference_identity_mask_path = Path(
            str(manifest_input_files.get("reference_mask") or "")
        )
        for mask_role, mask_path in (
            ("pose_mask_video", pose_identity_mask_video_path),
            ("reference_mask", reference_identity_mask_path),
        ):
            if not mask_path.is_file():
                raise FileNotFoundError(
                    f"SCAIL {action} {mask_role} is missing: {mask_path}"
                )
        pose_identity_mask_video_sha256 = file_sha256(pose_identity_mask_video_path)
        reference_identity_mask_sha256 = file_sha256(reference_identity_mask_path)
        manifest["input_files_sha256"] = {
            "pose_mask_video": pose_identity_mask_video_sha256,
            "reference_mask": reference_identity_mask_sha256,
        }
        manifest["reference_policy"] = reference_policy
        manifest["reference_source_path"] = str(reference_path)
        manifest["scail_reference_mode"] = action_config.get(
            "scail_reference_mode"
        )
        manifest["qwen_canonical_identity_fallback"] = action_config.get(
            "qwen_canonical_identity_fallback"
        )
        expected_pose_policy = expected_pose_policies.get(action)
        require_expected_scail_pose_policy(
            manifest,
            action,
            expected_pose_policy,
        )
        pose_metadata = manifest.get("pose_metadata")
        final_gate_required_actions = {
            str(value)
            for value in (
                get_job(job_id).get(
                    "action_candidate_final_gate_required_actions",
                    [],
                )
                or []
            )
        }
        dog_bowl_motion_contract = bool(
            action in {"eating", "drinking"}
            and str(job.get("species") or "").strip().lower() == "dog"
        )
        bowl_motion_contract_issues: list[str] = []
        drinking_pre_scail_motion_quality: dict[str, Any] | None = None
        drinking_bowl_area_retention_ratio: float | None = None
        if dog_bowl_motion_contract:
            if not isinstance(pose_metadata, dict):
                bowl_motion_contract_issues.append("pose_metadata_missing")
            else:
                full_driver_drinking_pose = bool(
                    action == "drinking"
                    and pose_metadata.get("policy")
                    == DOG_DRINKING_FULL_DRIVER_POSE_POLICY
                )
                if pose_metadata.get("fallback_used") is True:
                    bowl_motion_contract_issues.append(
                        "static_pose_fallback_forbidden"
                    )
                if canonical_full_driver_eating_fallback:
                    if (
                        pose_metadata.get("policy")
                        != DOG_EATING_CANONICAL_FULL_DRIVER_POSE_POLICY
                    ):
                        bowl_motion_contract_issues.append(
                            "canonical_full_driver_pose_policy_missing"
                        )
                    if pose_metadata.get("driver_input_used") is not True:
                        bowl_motion_contract_issues.append(
                            "eating_white_driver_not_used"
                        )
                    if (
                        pose_metadata.get("full_white_driver_geometry")
                        is not True
                    ):
                        bowl_motion_contract_issues.append(
                            "full_white_driver_geometry_missing"
                        )
                    if pose_metadata.get("reference_identity_only") is not True:
                        bowl_motion_contract_issues.append(
                            "canonical_reference_not_identity_only"
                        )
                    if pose_metadata.get("driver_signal_only") is not False:
                        bowl_motion_contract_issues.append(
                            "full_white_driver_silhouette_not_preserved"
                        )
                elif full_driver_drinking_pose:
                    if pose_metadata.get("primary_policy") is not True:
                        bowl_motion_contract_issues.append(
                            "drinking_full_driver_not_primary_policy"
                        )
                    if pose_metadata.get("driver_input_used") is not True:
                        bowl_motion_contract_issues.append(
                            "drinking_white_driver_not_used"
                        )
                    if (
                        pose_metadata.get("full_white_driver_geometry")
                        is not True
                    ):
                        bowl_motion_contract_issues.append(
                            "full_white_driver_geometry_missing"
                        )
                    if pose_metadata.get("driver_signal_only") is not False:
                        bowl_motion_contract_issues.append(
                            "full_white_driver_silhouette_not_preserved"
                        )
                    if pose_metadata.get("driver_silhouette_preserved") is not True:
                        bowl_motion_contract_issues.append(
                            "drinking_driver_silhouette_not_preserved"
                        )
                    if (
                        pose_metadata.get("pose_geometry_scope")
                        != "full_white_driver_subject_per_frame"
                    ):
                        bowl_motion_contract_issues.append(
                            "drinking_full_driver_geometry_scope_missing"
                        )
                    if (
                        pose_metadata.get("scene_prop_authority")
                        != "digest_bound_white_maquette_video"
                    ):
                        bowl_motion_contract_issues.append(
                            "drinking_bowl_not_sourced_from_white_driver"
                        )
                    if pose_metadata.get("static_bowl_mask") is not True:
                        bowl_motion_contract_issues.append(
                            "drinking_static_bowl_mask_missing"
                        )
                    if pose_metadata.get("static_bowl_rgb") is not True:
                        bowl_motion_contract_issues.append(
                            "drinking_static_bowl_rgb_missing"
                        )
                    try:
                        source_bowl_fraction = float(
                            pose_metadata["scene_prop_fraction"]
                        )
                        encoded_bowl_fraction = float(
                            manifest["mask_codec_metadata"]["red_fraction"]
                        )
                        drinking_bowl_area_retention_ratio = (
                            encoded_bowl_fraction / source_bowl_fraction
                        )
                    except (KeyError, TypeError, ValueError, ZeroDivisionError):
                        bowl_motion_contract_issues.append(
                            "drinking_bowl_area_retention_unavailable"
                        )
                    else:
                        if (
                            drinking_bowl_area_retention_ratio
                            < DRINKING_PREPARED_BOWL_AREA_RETENTION_RATIO_MIN
                        ):
                            bowl_motion_contract_issues.append(
                                "drinking_bowl_area_retention_too_low"
                            )
                elif action == "drinking":
                    bowl_motion_contract_issues.append(
                        "drinking_full_white_driver_primary_policy_missing"
                    )
                elif pose_metadata.get("driver_signal_only") is not True:
                    bowl_motion_contract_issues.append(
                        "white_driver_motion_signal_missing"
                    )
                if pose_metadata.get("scene_prop_separated") is not True:
                    bowl_motion_contract_issues.append(
                        "bowl_prop_not_separated"
                    )
                if (
                    action == "drinking"
                    and pose_metadata.get("driver_input_used") is not True
                ):
                    bowl_motion_contract_issues.append(
                        "drinking_white_driver_not_used"
                    )
                if action == "drinking":
                    try:
                        drinking_pre_scail_motion_quality = (
                            evaluate_drinking_prepared_pose_motion(
                                Path(manifest["input_files"]["pose_video"]),
                                source_driver_video_path=Path(manifest["source"]),
                            )
                        )
                    except Exception as exc:
                        bowl_motion_contract_issues.append(
                            "drinking_prepared_source_motion_quality_unavailable: "
                            f"{str(exc)[:500]}"
                        )
                    else:
                        if (
                            drinking_pre_scail_motion_quality.get("acceptable")
                            is not True
                        ):
                            bowl_motion_contract_issues.extend(
                                str(issue)
                                for issue in drinking_pre_scail_motion_quality.get(
                                    "issues", []
                                )
                            )
        manifest["final_action_gate"] = {
            "required_by_candidate_gate": action in final_gate_required_actions,
            "pre_scail_motion_contract_required": dog_bowl_motion_contract,
            "pre_scail_motion_contract_passed": not bowl_motion_contract_issues,
            "issues": bowl_motion_contract_issues,
            "drinking_source_to_prepared_pose_quality": (
                drinking_pre_scail_motion_quality
            ),
            "drinking_bowl_area_retention_ratio": (
                round(drinking_bowl_area_retention_ratio, 6)
                if drinking_bowl_area_retention_ratio is not None
                else None
            ),
            "drinking_bowl_area_retention_ratio_minimum": (
                DRINKING_PREPARED_BOWL_AREA_RETENTION_RATIO_MIN
                if action == "drinking"
                else None
            ),
        }
        final_video_comparison_scope = (
            select_final_video_fur_surface_comparison_scope(
                action=action,
                action_config=action_config,
                output_format=job.get("output_format"),
                reference_policy=reference_policy,
                pose_metadata=(
                    pose_metadata if isinstance(pose_metadata, dict) else None
                ),
            )
            if GENERATION_VISUAL_AUDITS_BLOCKING
            else None
        )
        write_workflow(job_dir, f"scail_{action}.json", workflow)
        write_json_atomic(job_dir / f"{action}_prepare_manifest.json", manifest)

        if bowl_motion_contract_issues:
            action_error = RuntimeError(
                f"{action}_pre_scail_action_motion_contract_failed: "
                + ", ".join(bowl_motion_contract_issues)
            )
            degraded_result, _fallback_source = (
                recover_subject_replacement_action_with_fallback(
                    job_id,
                    get_job(job_id),
                    action=action,
                    pose_index=actions.index(action) + 1,
                    error=action_error,
                )
            )
            degraded_result["final_action_gate_required"] = bool(
                action in final_gate_required_actions
            )
            degraded_result["pre_scail_action_motion_contract"] = (
                manifest["final_action_gate"]
            )
            results.append(degraded_result)
            update_job(
                job_id,
                progress=action_start + action_span,
                subject_replacement_results=results,
            )
            continue

        normalized_reference = experiment_dir / "reference_normalized.png"
        reference_target = job_dir / f"{action}_subject_reference.png"
        shutil.copy2(normalized_reference, reference_target)
        append_output(
            job_id,
            {
                "type": "subject_reference",
                "action": action,
                "action_label": action_label,
                "filename": reference_target.name,
                "width": 512,
                "height": 512,
                "transparent": True,
                "sha256": file_sha256(reference_target),
                "role": (
                    "qwen_action_keyframe_reference"
                    if job.get("subject_source") == "qwen_generated_upload"
                    else (
                        "legacy_direct_uploaded_identity_reference"
                        if job.get("subject_source") == "uploaded_identity_direct_scail"
                        else "profile_action_reference"
                    )
                ),
                "primary": False,
            },
        )
        prepared_visuals = (
            (
                "reference_mask.png",
                f"{action}_reference_identity_mask.png",
                "reference_identity_mask",
                "identity_and_prop_mapping",
            ),
            (
                "pose_first_frame.png",
                f"{action}_main_case_geometry_guide.png",
                "main_case_geometry_guide",
                "motion_pose_prop_camera_only",
            ),
            (
                "pose_mask_first_frame.png",
                f"{action}_main_case_geometry_mask.png",
                "main_case_geometry_mask",
                "driver_identity_and_prop_mapping",
            ),
        )
        for source_name, target_name, output_type, role in prepared_visuals:
            visual_source = experiment_dir / source_name
            if not visual_source.is_file():
                raise FileNotFoundError(
                    f"SCAIL {action} prepared visual is missing: {visual_source}"
                )
            visual_target = job_dir / target_name
            shutil.copy2(visual_source, visual_target)
            append_output(
                job_id,
                {
                    "type": output_type,
                    "action": action,
                    "action_label": action_label,
                    "filename": visual_target.name,
                    "width": 512,
                    "height": 512,
                    "sha256": file_sha256(visual_target),
                    "role": role,
                    "primary": False,
                },
            )
        scail_reference_policies[action] = reference_policy
        if action == "toileting":
            scail_scene_prop_policies[action] = (
                "dog_toileting_lateral_tray_red_object_mapping_v2"
            )
        elif action in {"eating", "drinking"}:
            scail_scene_prop_policies[action] = (
                "dog_white_model_bowl_scail_object_mapping_v2"
                if job.get("species") == "dog"
                else "cat_blue_bowl_red_scail_object_mapping"
            )
        else:
            scail_scene_prop_policies[action] = "no_scene_prop"
        update_job(
            job_id,
            scail_reference_policy=dict(scail_reference_policies),
            scail_scene_prop_policy=dict(scail_scene_prop_policies),
        )

        update_job(
            job_id,
            stage="generating_subject_replacement",
            progress=action_start + action_span * 0.18,
        )
        video_target = job_dir / f"{action}_subject_replacement.mp4"
        dog_toileting_temporal_gate = bool(
            action == "toileting"
            and str(job.get("species") or "").strip().lower() == "dog"
        )
        dog_drinking_motion_gate = bool(
            action == "drinking"
            and str(job.get("species") or "").strip().lower() == "dog"
        )
        dog_eating_motion_gate = bool(
            action == "eating"
            and str(job.get("species") or "").strip().lower() == "dog"
        )
        maximum_scail_attempts = (
            TOILETING_SCAIL_MAX_ATTEMPTS
            if dog_toileting_temporal_gate
            else DRINKING_SCAIL_MAX_ATTEMPTS
            if dog_drinking_motion_gate
            else EATING_SCAIL_MAX_ATTEMPTS
            if dog_eating_motion_gate
            else 1
        )
        base_scail_seed = int(action_config["effective_seed"])
        scail_attempt_records: list[dict[str, Any]] = []
        selected_scail_candidate: Path | None = None
        selected_scail_seed: int | None = None
        for scail_attempt in range(1, maximum_scail_attempts + 1):
            attempt_seed = base_scail_seed + (
                (scail_attempt - 1) * TOILETING_SCAIL_RETRY_SEED_STEP
            )
            candidate_workflow, updated_seed_nodes, retry_controls = (
                scail_workflow_with_retry_controls(
                    workflow,
                    seed=attempt_seed,
                    action=action,
                    attempt=scail_attempt,
                )
            )
            if scail_attempt > 1:
                update_job(
                    job_id,
                    stage="retrying_subject_replacement",
                    progress=action_start + action_span * 0.18,
                    current_action=action,
                    current_action_label=action_label,
                    scail_retry_action=action,
                    scail_retry_attempt=scail_attempt,
                    scail_retry_max_attempts=maximum_scail_attempts,
                    scail_retry_seed=attempt_seed,
                    scail_retry_reason=(
                        "toileting_temporal_lateral_dark_band_or_symbol"
                        if dog_toileting_temporal_gate
                        else (
                            "drinking_motion_or_driver_silhouette_"
                            "retention_too_low"
                        )
                        if dog_drinking_motion_gate
                        else "eating_visible_motion_energy_too_low"
                    ),
                )
            write_workflow(
                job_dir,
                f"scail_{action}_attempt_{scail_attempt:02d}.json",
                candidate_workflow,
            )
            entry = queue_and_wait(
                candidate_workflow,
                (
                    f"SCAIL subject replacement {action} for {job_id} "
                    f"attempt {scail_attempt}/{maximum_scail_attempts}"
                ),
                timeout=7200,
                job_id=job_id,
            )
            generated_candidate = first_output(
                entry,
                {".mp4"},
                f"SCAIL {action} subject replacement attempt {scail_attempt}",
            )
            attempt_target = (
                job_dir
                / f"{action}_subject_replacement_attempt_{scail_attempt:02d}.mp4"
                if maximum_scail_attempts > 1
                else video_target
            )
            shutil.copy2(generated_candidate, attempt_target)
            if dog_toileting_temporal_gate:
                temporal_identity_quality = (
                    evaluate_toileting_scail_video_identity_artifacts(
                        attempt_target,
                    )
                )
            elif dog_drinking_motion_gate:
                temporal_identity_quality = (
                    evaluate_drinking_scail_video_motion(
                        attempt_target,
                        expected_pose_video_path=Path(
                            manifest["input_files"]["pose_video"]
                        ),
                        source_driver_video_path=Path(manifest["source"]),
                    )
                )
            elif dog_eating_motion_gate:
                temporal_identity_quality = (
                    evaluate_eating_scail_video_motion(attempt_target)
                )
            else:
                temporal_identity_quality = {
                    "policy": "not_applicable",
                    "available": True,
                    "acceptable": True,
                    "issues": [],
                }
            attempt_record = {
                "attempt": scail_attempt,
                "seed": attempt_seed,
                "seed_step": TOILETING_SCAIL_RETRY_SEED_STEP,
                "workflow_seed_nodes": updated_seed_nodes,
                "retry_controls": retry_controls,
                "video": attempt_target.name,
                "video_sha256": file_sha256(attempt_target),
                "temporal_identity_quality": temporal_identity_quality,
                "selected": False,
            }
            scail_attempt_records.append(attempt_record)
            if temporal_identity_quality.get("acceptable") is True:
                selected_scail_candidate = attempt_target
                selected_scail_seed = attempt_seed
                attempt_record["selected"] = True
                break
            LOGGER.warning(
                "%s SCAIL %s attempt %d/%d rejected: %s",
                job_id,
                action,
                scail_attempt,
                maximum_scail_attempts,
                temporal_identity_quality.get("issues"),
            )

        scail_video_attempts_by_action[action] = scail_attempt_records
        if selected_scail_candidate is None or selected_scail_seed is None:
            update_job(
                job_id,
                scail_video_attempts=dict(scail_video_attempts_by_action),
                scail_video_temporal_identity_status={
                    selected_action: (
                        "passed"
                        if any(
                            record.get("selected") is True
                            for record in records
                        )
                        else "failed"
                    )
                    for selected_action, records in (
                        scail_video_attempts_by_action.items()
                    )
                },
            )
            action_error = RuntimeError(
                f"{action}_scail_temporal_identity_gate_failed_after_"
                f"{maximum_scail_attempts}_attempts"
            )
            degraded_result, _fallback_source = (
                recover_subject_replacement_action_with_fallback(
                    job_id,
                    get_job(job_id),
                    action=action,
                    pose_index=actions.index(action) + 1,
                    error=action_error,
                )
            )
            degraded_result["scail_attempt_count"] = len(
                scail_attempt_records
            )
            degraded_result["scail_video_attempts"] = scail_attempt_records
            degraded_result["temporal_identity_quality"] = (
                scail_attempt_records[-1].get("temporal_identity_quality")
                if scail_attempt_records
                else None
            )
            results.append(degraded_result)
            update_job(
                job_id,
                progress=action_start + action_span,
                subject_replacement_results=results,
            )
            continue
        selected_scail_seeds[action] = selected_scail_seed
        if selected_scail_candidate != video_target:
            shutil.copy2(selected_scail_candidate, video_target)
        generated_video = video_target
        update_job(
            job_id,
            scail_video_attempts=dict(scail_video_attempts_by_action),
            scail_video_temporal_identity_status={
                selected_action: (
                    "passed"
                    if any(
                        record.get("selected") is True
                        for record in records
                    )
                    else "failed"
                )
                for selected_action, records in (
                    scail_video_attempts_by_action.items()
                )
            },
            effective_action_selected_seeds=dict(selected_scail_seeds),
        )
        final_video_fur_surface: dict[str, Any] | None = None
        fur_surface_config = action_config.get("fur_surface_gate")
        if (
            GENERATION_VISUAL_AUDITS_BLOCKING
            and isinstance(fur_surface_config, dict)
            and fur_surface_config.get("enabled")
        ):
            update_job(
                job_id,
                stage="auditing_final_video_fur_surface",
                progress=action_start + action_span * 0.76,
            )
            selected_surface_metrics = job.get("fur_surface_gate_selected_metrics")
            validated_action_reference_gate = (
                selected_surface_metrics.get(action)
                if isinstance(selected_surface_metrics, dict)
                else None
            )
            final_video_fur_surface = evaluate_final_video_fur_surface(
                video_target,
                action=action,
                effective_subject_config=effective_subject_config,
                comparison_scope=final_video_comparison_scope,
                validated_action_reference_path=(
                    reference_path
                    if final_video_comparison_scope
                    == FINAL_VIDEO_VALIDATED_ACTION_SURFACE_SCOPE
                    else None
                ),
                validated_action_reference_sha256=(
                    str(action_config.get("reference_sha256") or "")
                    if final_video_comparison_scope
                    == FINAL_VIDEO_VALIDATED_ACTION_SURFACE_SCOPE
                    else None
                ),
                validated_action_reference_gate=(
                    validated_action_reference_gate
                    if final_video_comparison_scope
                    == FINAL_VIDEO_VALIDATED_ACTION_SURFACE_SCOPE
                    else None
                ),
                reference_policy=reference_policy,
                pose_identity_mask_video_path=(
                    pose_identity_mask_video_path
                    if final_video_comparison_scope
                    == FINAL_VIDEO_VALIDATED_ACTION_SURFACE_SCOPE
                    else None
                ),
                pose_identity_mask_video_sha256=(
                    pose_identity_mask_video_sha256
                    if final_video_comparison_scope
                    == FINAL_VIDEO_VALIDATED_ACTION_SURFACE_SCOPE
                    else None
                ),
                reference_identity_mask_path=(
                    reference_identity_mask_path
                    if final_video_comparison_scope
                    == FINAL_VIDEO_VALIDATED_ACTION_SURFACE_SCOPE
                    else None
                ),
                reference_identity_mask_sha256=(
                    reference_identity_mask_sha256
                    if final_video_comparison_scope
                    == FINAL_VIDEO_VALIDATED_ACTION_SURFACE_SCOPE
                    else None
                ),
                grooming_pose_gate_composite_recovery=(
                    job.get("grooming_pose_gate_composite_recovery")
                    if (
                        action == "grooming"
                        and final_video_comparison_scope
                        == FINAL_VIDEO_VALIDATED_ACTION_SURFACE_SCOPE
                    )
                    else None
                ),
            )
            final_video_fur_surface_by_action[action] = final_video_fur_surface
            audit_target = job_dir / f"{action}_final_video_fur_surface.json"
            write_json_atomic(audit_target, final_video_fur_surface)
            update_job(
                job_id,
                final_video_fur_surface_status={
                    audit_action: (
                        "passed" if audit.get("passed") is True else "failed"
                    )
                    for audit_action, audit in (
                        final_video_fur_surface_by_action.items()
                    )
                },
                final_video_fur_surface_quality=dict(final_video_fur_surface_by_action),
                fur_surface_gate_final_quality={
                    "scope": "candidate_reference_and_final_video",
                    "status": (
                        "passed"
                        if all(
                            audit.get("passed") is True
                            for audit in final_video_fur_surface_by_action.values()
                        )
                        else "failed"
                    ),
                    "policy": FUR_SURFACE_GATE_POLICY,
                    "gate_version": FUR_SURFACE_GATE_VERSION,
                    "threshold_version": FUR_SURFACE_THRESHOLD_VERSION,
                    "final_video_policy": FINAL_VIDEO_FUR_AUDIT_POLICY,
                    "final_video_version": FINAL_VIDEO_FUR_AUDIT_VERSION,
                    "actions": dict(final_video_fur_surface_by_action),
                    "final_video_frame_audit": True,
                },
            )
            if final_video_fur_surface.get("passed") is not True:
                failed_frames = final_video_fur_surface.get("failed_frame_indices")
                failed_issues = sorted(
                    {
                        str(issue)
                        for frame in final_video_fur_surface.get("frames", [])
                        if isinstance(frame, dict)
                        for issue in (
                            frame.get("decision", {}).get(
                                "blocking_issues_action_reference_scope"
                            )
                            or frame.get("decision", {}).get(
                                "blocking_issues",
                                [],
                            )
                            if isinstance(frame.get("decision"), dict)
                            else []
                        )
                    }
                )
                detail = (
                    f"frames={failed_frames}; issues={failed_issues}"
                    if isinstance(failed_frames, list)
                    else str(final_video_fur_surface.get("error") or "unknown")
                )
                action_error = RuntimeError(
                    f"{action}_final_video_fur_surface_gate_failed: {detail}"
                )
                degraded_result, _fallback_source = (
                    recover_subject_replacement_action_with_fallback(
                        job_id,
                        get_job(job_id),
                        action=action,
                        pose_index=actions.index(action) + 1,
                        error=action_error,
                    )
                )
                degraded_result.update(
                    {
                        "scail_attempt_count": len(scail_attempt_records),
                        "scail_video_attempts": scail_attempt_records,
                        "scail_generation_succeeded": True,
                        "scail_video_filename": generated_video.name,
                        "final_video_fur_surface": final_video_fur_surface,
                        "final_video_fur_surface_failed": True,
                    }
                )
                results.append(degraded_result)
                update_job(
                    job_id,
                    progress=action_start + action_span,
                    subject_replacement_results=results,
                )
                continue
            append_output(
                job_id,
                {
                    "type": "final_video_fur_surface_audit",
                    "action": action,
                    "action_label": action_label,
                    "filename": audit_target.name,
                    "primary": False,
                },
            )
        elif isinstance(fur_surface_config, dict) and fur_surface_config.get(
            "enabled"
        ):
            final_video_fur_surface = {
                "status": "skipped_advisory",
                "passed": None,
                "blocking": False,
                "action": action,
                "policy": FINAL_VIDEO_FUR_AUDIT_POLICY,
                "version": FINAL_VIDEO_FUR_AUDIT_VERSION,
                "generation_audit_policy": GENERATION_AUDIT_POLICY,
                "reason": "visual_content_audits_are_non_blocking",
            }
            final_video_fur_surface_by_action[action] = final_video_fur_surface
            update_job(
                job_id,
                final_video_fur_surface_status={
                    audit_action: str(audit.get("status") or "skipped_advisory")
                    for audit_action, audit in final_video_fur_surface_by_action.items()
                },
                final_video_fur_surface_quality=dict(
                    final_video_fur_surface_by_action
                ),
                fur_surface_gate_final_quality={
                    "scope": "candidate_reference_and_final_video",
                    "status": "skipped_advisory",
                    "blocking": False,
                    "generation_audit_policy": GENERATION_AUDIT_POLICY,
                    "actions": dict(final_video_fur_surface_by_action),
                    "final_video_frame_audit": False,
                },
            )
        video_playback_contract: dict[str, Any] | None = None
        if job.get("output_format") != "transparent_gif":
            if float(action_timing["playback_rate"]) != 1.0:
                video_playback_contract = retime_video_playback(
                    video_target,
                    video_target,
                    ffmpeg_bin=FFMPEG_BIN,
                    source_fps=float(action_timing["generation"]["fps"]),
                    output_fps=float(video_timing["fps"]),
                    frame_count=int(video_timing["frames"]),
                )
            video_contract = validate_standard_direct_runtime_video(
                video_target,
                expected_frames=int(video_timing["frames"]),
                expected_fps=float(video_timing["fps"]),
            )
            if video_playback_contract is None:
                video_playback_contract = {
                    "policy": action_timing["policy"],
                    **video_contract,
                }

        video_output = {
            "type": "subject_replacement_video",
            "action": action,
            "action_label": action_label,
            "filename": video_target.name,
            "width": 512,
            "height": 512,
            "frames": int(video_timing["frames"]),
            "fps": float(video_timing["fps"]),
            "duration_seconds": float(video_timing["duration_seconds"]),
            "output_playback": action_timing,
            "playback_contract": video_playback_contract,
            "video_engine": "scail2-14b-fp8",
            "output_exists": True,
            "generation_succeeded": True,
            "quality_passed": True,
            "degraded": False,
            "requested_effective_seed": base_scail_seed,
            "selected_effective_seed": selected_scail_seed,
            "scail_attempt_count": len(scail_attempt_records),
            "temporal_identity_quality": next(
                (
                    record.get("temporal_identity_quality")
                    for record in scail_attempt_records
                    if record.get("selected") is True
                ),
                None,
            ),
            "driving_template": Path(action_config["driving_animation"]).name,
            "final_video_fur_surface_audit": (
                f"{action}_final_video_fur_surface.json"
                if (
                    final_video_fur_surface is not None
                    and final_video_fur_surface.get("status")
                    != "skipped_advisory"
                )
                else None
            ),
            "primary": job.get("output_format") != "transparent_gif",
        }
        if job.get("output_format") != "transparent_gif":
            append_output(job_id, video_output)

        transparent_metrics: dict[str, Any] | None = None
        temporal_cycle_output: dict[str, Any] | None = None
        if job.get("output_format") == "transparent_gif":
            raise_if_cancelled(job_id)
            update_job(
                job_id,
                stage="extracting_transparent_animation",
                progress=action_start + action_span * 0.82,
            )
            transparent_guidance_plan = select_transparent_guidance_policy(
                action,
                manifest,
            )
            guidance_source_key = str(
                transparent_guidance_plan["source_key"]
            )
            guidance_path = Path(
                str(manifest["input_files"][guidance_source_key])
            )
            if transparent_guidance_plan["static_reference_mask"]:
                static_guidance_path = (
                    job_dir
                    / f"{action}_transparent_reference_guidance_mask.mp4"
                )
                static_guidance_log = (
                    job_dir
                    / f"{action}_transparent_reference_guidance_mask.log"
                )
                run_cancelable_process(
                    job_id,
                    [
                        str(SCAIL_PYTHON),
                        str(SCAIL_STATIC_MASK_RUNNER),
                        str(guidance_path),
                        str(static_guidance_path),
                        "--frames",
                        str(effective_subject_config["frames"]),
                        "--fps",
                        str(effective_subject_config["fps"]),
                    ],
                    log_path=static_guidance_log,
                    stage_name=(
                        f"{action} static reference guidance encoding"
                    ),
                    timeout=180,
                )
                if not static_guidance_path.is_file():
                    raise RuntimeError(
                        f"{action} static reference guidance produced no video."
                    )
                static_guidance_metrics = logged_json(
                    static_guidance_log
                )
                guidance_path = static_guidance_path
                transparent_guidance_plan = {
                    **transparent_guidance_plan,
                    "static_mask_video": static_guidance_path.name,
                    "static_mask_video_sha256": file_sha256(
                        static_guidance_path
                    ),
                    "static_mask_video_metrics": static_guidance_metrics,
                }
                append_output(
                    job_id,
                    {
                        "type": "transparent_guidance_mask_video",
                        "action": action,
                        "action_label": action_label,
                        "filename": static_guidance_path.name,
                        "frames": int(effective_subject_config["frames"]),
                        "fps": float(effective_subject_config["fps"]),
                        "sha256": file_sha256(static_guidance_path),
                        "policy": transparent_guidance_plan["policy"],
                        "primary": False,
                    },
                )
            transparent_guidance_policies[action] = transparent_guidance_plan
            manifest["transparent_guidance"] = transparent_guidance_plan
            write_json_atomic(
                job_dir / f"{action}_prepare_manifest.json",
                manifest,
            )
            gif_target = job_dir / f"{action}_subject_replacement_transparent_512.gif"
            apng_target = (
                job_dir
                / f"{action}_subject_replacement_transparent_512_soft.png"
            )
            transparent_log = job_dir / f"{action}_transparent.log"
            transparent_phase_start_index = (
                resolve_transparent_output_phase_start_index(
                    job,
                    action=action,
                    source_frames=int(effective_subject_config["frames"]),
                )
            )
            transparent_command = [
                str(SCAIL_PYTHON),
                str(SCAIL_TRANSPARENT_RUNNER),
                str(generated_video),
                str(guidance_path),
                str(gif_target),
                "--action",
                action,
                "--white-mp4-target",
                str(video_target),
                "--soft-apng-target",
                str(apng_target),
                "--start-frame-index",
                str(transparent_phase_start_index),
                "--source-fps",
                str(video_timing["fps"]),
                "--gif-fps",
                str(gif_timing["fps"]),
                "--gif-frames",
                str(gif_timing["frames"]),
            ]
            if transparent_guidance_plan[
                "reference_aligned_uniform_background"
            ]:
                transparent_command.append(
                    "--reference-aligned-adaptive-extraction"
                )
            update_job(
                job_id,
                transparent_output_phase_policy=(
                    "validated_action_contact_phase_rotation_v1"
                ),
                transparent_output_phase_start_index=(
                    transparent_phase_start_index
                ),
                transparent_output_phase_action=action,
                transparent_guidance_policy=dict(
                    transparent_guidance_policies
                ),
            )
            gif_target.unlink(missing_ok=True)
            apng_target.unlink(missing_ok=True)
            try:
                run_cancelable_process(
                    job_id,
                    transparent_command,
                    log_path=transparent_log,
                    stage_name=f"{action} transparent GIF extraction",
                    timeout=1800,
                )
                if not gif_target.is_file():
                    raise RuntimeError(
                        f"{action} transparent GIF extraction produced no file."
                    )
                transparent_metrics = logged_json(transparent_log)
            except JobCancelled:
                raise
            except Exception as exc:
                degraded_result, _fallback_source = (
                    recover_subject_replacement_action_with_fallback(
                        job_id,
                        get_job(job_id),
                        action=action,
                        pose_index=actions.index(action) + 1,
                        error=exc,
                    )
                )
                degraded_result.update(
                    {
                        "scail_attempt_count": len(scail_attempt_records),
                        "scail_video_attempts": scail_attempt_records,
                        "scail_generation_succeeded": True,
                        "scail_video_filename": generated_video.name,
                        "transparent_extraction_failed": True,
                        "transparent_guidance_policy": (
                            transparent_guidance_plan
                        ),
                    }
                )
                results.append(degraded_result)
                update_job(
                    job_id,
                    progress=action_start + action_span,
                    subject_replacement_results=results,
                )
                continue
            apng_available = apng_target.is_file()
            transparent_metrics["guidance_policy"] = (
                transparent_guidance_plan
            )
            if action in {"eating", "drinking"}:
                bowl_temporal_cycle = transparent_metrics.get(
                    "temporal_cycle"
                )
                bowl_published_loop = (
                    bowl_temporal_cycle.get("published_loop_quality")
                    if isinstance(bowl_temporal_cycle, dict)
                    else None
                )
                if not (
                    isinstance(bowl_published_loop, dict)
                    and bowl_published_loop.get("acceptable") is True
                ):
                    action_error = RuntimeError(
                        f"{action}_transparent_loop_quality_failed"
                    )
                    degraded_result, _fallback_source = (
                        recover_subject_replacement_action_with_fallback(
                            job_id,
                            get_job(job_id),
                            action=action,
                            pose_index=actions.index(action) + 1,
                            error=action_error,
                        )
                    )
                    degraded_result["scail_attempt_count"] = len(
                        scail_attempt_records
                    )
                    degraded_result["scail_video_attempts"] = (
                        scail_attempt_records
                    )
                    degraded_result["transparent_loop_quality"] = (
                        bowl_temporal_cycle
                    )
                    results.append(degraded_result)
                    update_job(
                        job_id,
                        progress=action_start + action_span,
                        subject_replacement_results=results,
                    )
                    continue
            if action == "grooming":
                temporal_cycle = transparent_metrics.get("temporal_cycle")
                published_loop_quality = (
                    temporal_cycle.get("published_loop_quality")
                    if isinstance(temporal_cycle, dict)
                    else None
                )
                temporal_cycle_acceptable = (
                    isinstance(temporal_cycle, dict)
                    and temporal_cycle.get("policy")
                    == GROOMING_TEMPORAL_LOOP_POLICY
                    and temporal_cycle.get("applied") is True
                    and temporal_cycle.get("closed_loop") is True
                    and temporal_cycle.get("acceptable") is True
                    and isinstance(published_loop_quality, dict)
                    and published_loop_quality.get("acceptable") is True
                )
                if not temporal_cycle_acceptable:
                    update_job(
                        job_id,
                        grooming_temporal_loop_status="failed",
                        grooming_temporal_loop_policy=(
                            GROOMING_TEMPORAL_LOOP_POLICY
                        ),
                        grooming_temporal_loop_quality=temporal_cycle,
                    )
                    action_error = RuntimeError(
                        "grooming_temporal_loop_failed: the published animation "
                        "did not preserve the closed-loop action contract"
                    )
                    degraded_result, _fallback_source = (
                        recover_subject_replacement_action_with_fallback(
                            job_id,
                            get_job(job_id),
                            action=action,
                            pose_index=actions.index(action) + 1,
                            error=action_error,
                        )
                    )
                    degraded_result.update(
                        {
                            "scail_attempt_count": len(scail_attempt_records),
                            "scail_video_attempts": scail_attempt_records,
                            "scail_generation_succeeded": True,
                            "scail_video_filename": generated_video.name,
                            "grooming_temporal_loop": temporal_cycle,
                            "grooming_temporal_loop_failed": True,
                        }
                    )
                    results.append(degraded_result)
                    update_job(
                        job_id,
                        progress=action_start + action_span,
                        subject_replacement_results=results,
                    )
                    continue
                temporal_cycle_output = {
                    "policy": GROOMING_TEMPORAL_LOOP_POLICY,
                    "playback": temporal_cycle.get("playback"),
                    "pivot_source_frame_index": temporal_cycle.get(
                        "pivot_source_frame_index"
                    ),
                    "closed_loop": True,
                    "published_loop_quality": published_loop_quality,
                }
                video_output["temporal_cycle"] = temporal_cycle_output
                update_job(
                    job_id,
                    grooming_temporal_loop_status="passed",
                    grooming_temporal_loop_policy=(
                        GROOMING_TEMPORAL_LOOP_POLICY
                    ),
                    grooming_temporal_loop_quality=temporal_cycle_output,
                )
            video_playback_contract = validate_standard_direct_runtime_video(
                video_target,
                expected_frames=int(video_timing["frames"]),
                expected_fps=float(video_timing["fps"]),
            )
            transparent_metrics["output_playback"] = action_timing
            transparent_metrics["video_playback_contract"] = (
                video_playback_contract
            )
            video_output["playback_contract"] = video_playback_contract
            fur_contour_quality: dict[str, Any] | None = None
            fur_gate = action_config.get("fur_silhouette_gate")
            if (
                GENERATION_VISUAL_AUDITS_BLOCKING
                and isinstance(fur_gate, dict)
                and fur_gate.get("enabled")
            ):
                pose_metadata = manifest.get("pose_metadata")
                full_driver_pose = isinstance(
                    pose_metadata, dict
                ) and uses_full_driver_eating_pose(pose_metadata.get("policy"))
                absolute_upload_gate = (
                    fur_gate.get("policy") == UPLOADED_ABSOLUTE_FUR_CONTOUR_POLICY
                )
                if absolute_upload_gate:
                    if action == "eating" and not full_driver_pose:
                        raise RuntimeError(
                            "eating_fur_contour_gate_failed: full-driver pose "
                            "policy changed"
                        )
                    fur_contour_quality = evaluate_animation_fur_quality(
                        gif_target,
                    )
                    fur_contour_quality["reference_policy"] = reference_policy
                    fur_contour_quality["gate_policy"] = (
                        UPLOADED_ABSOLUTE_FUR_CONTOUR_POLICY
                    )
                elif action == "eating" and full_driver_pose:
                    fur_contour_quality = evaluate_animation_fur_quality(
                        gif_target,
                    )
                    fur_contour_quality["reference_policy"] = reference_policy
                elif action == "eating":
                    baseline_config = action_config.get("fur_contour_baseline")
                    if not isinstance(baseline_config, dict):
                        raise RuntimeError(
                            "eating_fur_contour_gate_failed: frozen baseline is missing"
                        )
                    baseline_path = Path(str(baseline_config.get("path") or ""))
                    if not baseline_path.is_file():
                        raise FileNotFoundError(
                            "Frozen eating fur-contour baseline is missing: "
                            f"{baseline_path}"
                        )
                    expected_baseline_sha = str(
                        baseline_config.get("sha256") or ""
                    ).strip()
                    if (
                        expected_baseline_sha
                        and file_sha256(baseline_path) != expected_baseline_sha
                    ):
                        raise RuntimeError(
                            "eating_fur_contour_gate_failed: "
                            "frozen baseline digest changed"
                        )
                    fur_contour_quality = compare_fur_contours(
                        gif_target,
                        baseline_path,
                    )
                transparent_metrics["fur_contour"] = fur_contour_quality
            elif isinstance(fur_gate, dict) and fur_gate.get("enabled"):
                transparent_metrics["fur_contour"] = {
                    "status": "skipped_advisory",
                    "acceptable": None,
                    "blocking": False,
                    "generation_audit_policy": GENERATION_AUDIT_POLICY,
                }
                update_job(
                    job_id,
                    fur_contour_gate_status="skipped_advisory",
                    fur_contour_gate_action=action,
                    fur_contour_gate_quality=transparent_metrics["fur_contour"],
                )
            write_json_atomic(
                job_dir / f"{action}_transparent_metrics.json",
                transparent_metrics,
            )
            if (
                fur_contour_quality is not None
                and not fur_contour_quality["acceptable"]
            ):
                update_job(
                    job_id,
                    fur_contour_gate_status="failed",
                    fur_contour_gate_action=action,
                    fur_contour_gate_quality=fur_contour_quality,
                )
                action_error = RuntimeError(
                    "eating_fur_contour_gate_failed: "
                    + ", ".join(fur_contour_quality["issues"])
                )
                degraded_result, _fallback_source = (
                    recover_subject_replacement_action_with_fallback(
                        job_id,
                        get_job(job_id),
                        action=action,
                        pose_index=actions.index(action) + 1,
                        error=action_error,
                    )
                )
                degraded_result.update(
                    {
                        "scail_attempt_count": len(scail_attempt_records),
                        "scail_video_attempts": scail_attempt_records,
                        "scail_generation_succeeded": True,
                        "scail_video_filename": generated_video.name,
                        "fur_contour_quality": fur_contour_quality,
                        "fur_contour_gate_failed": True,
                    }
                )
                results.append(degraded_result)
                update_job(
                    job_id,
                    progress=action_start + action_span,
                    subject_replacement_results=results,
                )
                continue
            if fur_contour_quality is not None:
                update_job(
                    job_id,
                    fur_contour_gate_status="passed",
                    fur_contour_gate_action=action,
                    fur_contour_gate_quality=fur_contour_quality,
                )
            append_output(job_id, video_output)
            append_output(
                job_id,
                {
                    "type": "transparent_gif",
                    "action": action,
                    "action_label": action_label,
                    "filename": gif_target.name,
                    "width": TRANSPARENT_GIF_SIZE,
                    "height": TRANSPARENT_GIF_SIZE,
                    "frames": int(gif_timing["frames"]),
                    "fps": int(gif_timing["fps"]),
                    "duration_seconds": float(gif_timing["duration_seconds"]),
                    "output_playback": action_timing,
                    "transparent": True,
                    "video_engine": "scail2-14b-fp8",
                    "output_exists": True,
                    "generation_succeeded": True,
                    "quality_passed": True,
                    "degraded": False,
                    "driving_template": Path(action_config["driving_animation"]).name,
                    "edge_mode": "clean_hard_alpha_no_dither",
                    "temporal_cycle": temporal_cycle_output,
                    "primary": not apng_available,
                },
            )
            if apng_available:
                append_output(
                    job_id,
                    {
                        "type": "transparent_apng",
                        "action": action,
                        "action_label": action_label,
                        "filename": apng_target.name,
                        "width": TRANSPARENT_GIF_SIZE,
                        "height": TRANSPARENT_GIF_SIZE,
                        "frames": TRANSPARENT_GIF_FRAMES,
                        "fps": TRANSPARENT_GIF_FPS,
                        "duration_seconds": TRANSPARENT_GIF_DURATION_SECONDS,
                        "transparent": True,
                        "soft_alpha": True,
                        "video_engine": "scail2-14b-fp8",
                        "output_exists": True,
                        "generation_succeeded": True,
                        "quality_passed": True,
                        "degraded": False,
                        "driving_template": Path(
                            action_config["driving_animation"]
                        ).name,
                        "recommended_for_quality": True,
                        "temporal_cycle": temporal_cycle_output,
                        "primary": True,
                    },
                )
        results.append(
            {
                "action": action,
                "action_label": action_label,
                "reference": str(reference_path),
                "reference_sha256": file_sha256(reference_path),
                "driving_template": str(action_config["driving_animation"]),
                "requested_effective_seed": base_scail_seed,
                "effective_seed": int(selected_scail_seed),
                "scail_attempt_count": len(scail_attempt_records),
                "scail_video_attempts": scail_attempt_records,
                "temporal_identity_quality": next(
                    (
                        record.get("temporal_identity_quality")
                        for record in scail_attempt_records
                        if record.get("selected") is True
                    ),
                    None,
                ),
                "effective_pose_strength": float(action_config["pose_strength"]),
                "video": video_target.name,
                "transparent_gif": (
                    f"{action}_subject_replacement_transparent_512.gif"
                    if transparent_metrics is not None
                    else None
                ),
                "transparent_apng": (
                    f"{action}_subject_replacement_transparent_512_soft.png"
                    if transparent_metrics is not None
                    and apng_target.is_file()
                    else None
                ),
                "transparent_metrics": transparent_metrics,
                "temporal_cycle": temporal_cycle_output,
                "final_video_fur_surface": final_video_fur_surface,
                "output_playback": action_timing,
                "output_exists": True,
                "generation_succeeded": True,
                "quality_passed": True,
                "degraded": False,
            }
        )
        update_job(
            job_id,
            progress=action_start + action_span,
            subject_replacement_results=results,
        )

    raise_if_cancelled(job_id)
    free_comfy_models()
    result_order = {action: index for index, action in enumerate(actions)}
    results.sort(key=lambda result: result_order.get(str(result.get("action")), 999))
    degraded_actions = [
        str(result.get("action"))
        for result in results
        if result.get("degraded") is True
        or result.get("quality_passed") is False
    ]
    transparent_outputs_complete = bool(
        len(results) == len(actions)
        and all(str(result.get("transparent_gif") or "") for result in results)
    )
    transparent_outputs_quality_passed = bool(
        transparent_outputs_complete and not degraded_actions
    )
    update_job(
        job_id,
        status="completed",
        stage=(
            "completed_with_degraded_output"
            if degraded_actions
            else "completed"
        ),
        progress=100,
        current_action=None,
        current_action_label=None,
        current_action_index=None,
        subject_replacement_results=results,
        degraded_output_fallback=bool(degraded_actions),
        degraded_output_actions=degraded_actions,
        transparent_outputs_complete=transparent_outputs_complete,
        transparent_outputs_quality_passed=(
            transparent_outputs_quality_passed
        ),
        action_quality_summary={
            str(result.get("action")): {
                "output_exists": bool(result.get("output_exists")),
                "generation_succeeded": bool(
                    result.get("generation_succeeded")
                ),
                "quality_passed": bool(result.get("quality_passed")),
                "degraded": bool(result.get("degraded")),
            }
            for result in results
        },
        error=nonblocking_result_error(job_id),
        finished_at=utc_now(),
    )
