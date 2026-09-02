"""Top-level local-video job orchestration."""

from __future__ import annotations

# The extracted functions resolve the facade dependencies after runtime sync.
# ruff: noqa: F821

from .runtime import sync_local_video_runtime as _sync_local_video_runtime


def bounded_strict_first_frame_profiles(
    profiles: list[dict[str, Any]],
    *,
    max_attempts: int,
) -> list[dict[str, Any]]:
    """Return at most ``max_attempts`` seed-distinct first-frame profiles.

    Action-specific profile builders retain control of the initial recovery
    schedule. Once that schedule is exhausted, repeat its safest final profile
    with a new name and monotonically stronger retry index. The caller binds a
    distinct seed to every list position.
    """

    limit = max(1, int(max_attempts))
    selected = [
        {
            **profile,
            "user_loras": [dict(item) for item in profile.get("user_loras", [])],
        }
        for profile in profiles[:limit]
    ]
    if not selected:
        return selected
    safest = selected[-1]
    while len(selected) < limit:
        attempt = len(selected) + 1
        repeated = {
            **safest,
            "user_loras": [dict(item) for item in safest.get("user_loras", [])],
        }
        repeated["name"] = f"{safest['name']}_bounded_seed_retry_{attempt:02d}"
        repeated["quality_retry"] = max(
            attempt,
            int(repeated.get("quality_retry") or 0),
        )
        selected.append(repeated)
    return selected


def strict_first_frame_best_effort_selection_key(
    record: dict[str, Any],
    *,
    action: str,
) -> tuple[int | float, ...]:
    """Rank generated candidates after the bounded strict budget is exhausted."""

    chroma = record.get("chroma")
    cutout_missing = bool(
        not isinstance(chroma, dict)
        or chroma.get("acceptable") is not True
        or not record.get("cutout")
    )
    return (
        int(cutout_missing),
        *action_candidate_degraded_selection_key(record, action=action),
        float(record.get("combined_penalty") or 0.0),
    )


def merge_identity_chroma_spill_cleanup(
    chroma: dict[str, Any],
    cleanup: dict[str, Any],
    *,
    blocking: bool,
) -> dict[str, Any]:
    """Attach despill evidence and reject candidates beyond safe reconstruction."""

    merged = {**chroma, "identity_chroma_spill_cleanup": cleanup}
    cleanup_issues = [str(issue) for issue in cleanup.get("issues", [])]
    if cleanup.get("acceptable", True) is not False:
        return merged
    if blocking:
        merged["acceptable"] = False
        merged["issues"] = list(
            dict.fromkeys(
                [
                    *[str(issue) for issue in merged.get("issues", [])],
                    *cleanup_issues,
                ]
            )
        )
    else:
        merged["warnings"] = list(
            dict.fromkeys(
                [
                    *[str(issue) for issue in merged.get("warnings", [])],
                    *cleanup_issues,
                ]
            )
        )
    return merged


def action_candidate_degraded_selection_key(
    record: dict[str, Any],
    *,
    action: str,
) -> tuple[int | float, ...]:
    """Rank an exhausted retry budget without sacrificing pet identity."""
    palette = record.get("identity_palette")
    similarity = record.get("similarity")
    fur = record.get("fur_silhouette")
    quality = record.get("quality")
    chroma = record.get("chroma")
    edge = (
        chroma.get("transparent_edge_quality")
        if isinstance(chroma, dict)
        and isinstance(chroma.get("transparent_edge_quality"), dict)
        else None
    )
    attempt = int(record.get("attempt") or 0)
    palette_unacceptable = bool(
        not isinstance(palette, dict) or palette.get("acceptable") is not True
    )
    similarity_unacceptable = bool(
        not isinstance(similarity, dict)
        or similarity.get("acceptable") is not True
    )
    fur_unacceptable = bool(
        isinstance(fur, dict) and fur.get("acceptable") is not True
    )
    quality_unacceptable = bool(
        not isinstance(quality, dict) or quality.get("acceptable") is not True
    )
    edge_unacceptable = bool(
        isinstance(edge, dict) and edge.get("acceptable") is not True
    )
    edge_penalty = (
        float(edge.get("penalty", 0.0)) if isinstance(edge, dict) else 0.0
    )

    if action != "eating":
        base_key = (
            int(palette_unacceptable),
            float(palette.get("penalty", 1000.0))
            if isinstance(palette, dict)
            else 1000.0,
            int(fur_unacceptable),
            float(fur.get("penalty", 1000.0))
            if isinstance(fur, dict)
            else 0.0,
            int(edge_unacceptable),
            edge_penalty,
            int(quality_unacceptable),
            float(quality.get("penalty", 1000.0))
            if isinstance(quality, dict)
            else 1000.0,
            int(similarity_unacceptable),
            float(similarity.get("penalty", 1000.0))
            if isinstance(similarity, dict)
            else 1000.0,
            attempt,
        )
        if action != "grooming":
            return base_key
        topology = grooming_foreleg_topology_selection_evidence(record)
        anatomy_retry = record.get("grooming_advisory_action_transfer")
        anatomy_priority_retry = bool(
            isinstance(anatomy_retry, dict)
            and anatomy_retry.get("anatomy_priority_retry") is True
        )
        return (
            int(topology.get("available") is not True),
            int(
                topology.get("available") is True
                and topology.get("acceptable") is not True
            ),
            int(not anatomy_priority_retry),
            float(topology.get("penalty") or 0.0),
            *base_key,
        )

    quality_issues = {
        str(issue)
        for issue in (
            quality.get("issues", []) if isinstance(quality, dict) else []
        )
    }
    horizontal_shelf = bool(
        "eating_horizontal_foreground_shelf" in quality_issues
        or (
            isinstance(quality, dict)
            and isinstance(quality.get("eating_horizontal_shelf"), dict)
            and quality["eating_horizontal_shelf"].get("acceptable") is False
        )
    )
    eating_issues = {
        issue for issue in quality_issues if issue.startswith("eating_")
    }
    if not eating_issues:
        action_failure_severity = 0
    elif eating_issues <= {"eating_bowl_contact_unreliable"}:
        # A conservative prop splitter can be inconclusive even when the pose
        # is visibly usable. Keep it ahead of a definite missing-bowl/sniffing
        # result once identity safety has already been established.
        action_failure_severity = 1
    else:
        action_failure_severity = 2

    return (
        # These fields form the identity-safe pool. A recognizable upload must
        # always beat a structurally cleaner identity-collapsed animal.
        int(palette_unacceptable),
        int(similarity_unacceptable),
        # Within that pool, visible structure/action outrank small color deltas.
        int(horizontal_shelf),
        action_failure_severity,
        int(quality_unacceptable),
        int(edge_unacceptable),
        edge_penalty,
        int(fur_unacceptable),
        float(fur.get("penalty", 1000.0))
        if isinstance(fur, dict)
        else 0.0,
        float(quality.get("penalty", 1000.0))
        if isinstance(quality, dict)
        else 1000.0,
        float(palette.get("penalty", 1000.0))
        if isinstance(palette, dict)
        else 1000.0,
        float(similarity.get("penalty", 1000.0))
        if isinstance(similarity, dict)
        else 1000.0,
        attempt,
    )


def grooming_foreleg_topology_selection_evidence(
    record: dict[str, Any],
) -> dict[str, Any]:
    """Resolve the non-blocking topology audit used by grooming ranking."""

    minimum_gate = record.get("grooming_advisory_minimum_gate")
    pose = (
        minimum_gate.get("pose")
        if isinstance(minimum_gate, dict)
        and isinstance(minimum_gate.get("pose"), dict)
        else None
    )
    topology = (
        pose.get("topology")
        if isinstance(pose, dict)
        and isinstance(pose.get("topology"), dict)
        else None
    )
    if isinstance(topology, dict):
        return topology
    pose_gate = record.get("grooming_pose_gate")
    metrics = (
        pose_gate.get("metrics")
        if isinstance(pose_gate, dict)
        and isinstance(pose_gate.get("metrics"), dict)
        else None
    )
    topology = (
        metrics.get("foreleg_topology")
        if isinstance(metrics, dict)
        and isinstance(metrics.get("foreleg_topology"), dict)
        else None
    )
    if isinstance(topology, dict):
        return topology
    return {
        "available": False,
        "acceptable": True,
        "issues": [],
        "penalty": 0.0,
        "blocking": False,
    }


def exhausted_action_candidate_recovery_policy(
    record: dict[str, Any],
    *,
    action: str,
    approved_canonical_available: bool,
) -> str:
    """Select the safest generated candidate after the fixed attempt budget."""
    _sync_local_video_runtime(
        globals(),
        {"exhausted_action_candidate_recovery_policy"},
    )
    del action, approved_canonical_available
    palette = record.get("identity_palette")
    if isinstance(palette, dict) and palette.get("acceptable") is True:
        quality = record.get("quality")
        quality_issues = {
            str(issue)
            for issue in (
                quality.get("issues", [])
                if isinstance(quality, dict)
                else []
            )
        }
        hard_action_issues = {
            "pose_not_low_or_wide_enough_for_bowl_action",
            "eating_bowl_or_food_not_detected",
            "eating_bowl_paw_layout_invalid",
            "eating_bowl_contact_unreliable",
            "eating_muzzle_not_at_bowl",
            "drinking_muzzle_not_at_bowl",
            "toileting_tray_not_wide_beneath_subject",
        }
        if quality_issues & hard_action_issues:
            return "identity_safe_generated_candidate_requires_final_action_gate"
        return "identity_safe_generated_candidate"
    if GENERATION_COMPLETION_FIRST_VISUAL_FALLBACK_ENABLED:
        return "generated_completion_candidate"
    return "reject"


def make_toileting_canonical_tray_fallback(
    source: Path,
    target: Path,
) -> dict[str, Any]:
    """Place an identity-preserving canonical dog on a deterministic clean tray.

    This path is used only after every generated toileting candidate has failed
    identity checks.  The approved canonical remains the complete dog-identity
    authority; the generated geometry image and white model are deliberately not
    loaded.  A neutral tray is drawn behind the unchanged dog so SCAIL receives a
    reference whose wide prop footprint is already beneath, rather than behind,
    the subject.
    """
    _sync_local_video_runtime(
        globals(), {"make_toileting_canonical_tray_fallback"}
    )

    with Image.open(source) as loaded:
        canonical = loaded.convert("RGBA")
    width, height = canonical.size
    rgba = np.asarray(canonical, dtype=np.uint8)
    subject = rgba[..., 3] >= 16
    rows, columns = np.where(subject)
    if columns.size < 500:
        raise RuntimeError(
            "Toileting canonical tray fallback requires a visible canonical dog."
        )
    subject_left = int(columns.min())
    subject_top = int(rows.min())
    subject_right = int(columns.max()) + 1
    subject_bottom = int(rows.max()) + 1
    subject_width = subject_right - subject_left
    subject_height = subject_bottom - subject_top
    if subject_width < width * 0.16 or subject_height < height * 0.28:
        raise RuntimeError(
            "Toileting canonical tray fallback rejected a tiny canonical dog."
        )

    side_extension = max(int(round(width * 0.05)), int(round(subject_width * 0.40)))
    tray_left = max(int(round(width * 0.03)), subject_left - side_extension)
    tray_right = min(int(round(width * 0.97)), subject_right + side_extension)
    tray_top = max(
        subject_top + int(round(subject_height * 0.60)),
        subject_bottom - int(round(subject_height * 0.27)),
    )
    tray_bottom = min(
        height - max(8, int(round(height * 0.025))),
        subject_bottom + max(18, int(round(subject_height * 0.07))),
    )
    tray_width = tray_right - tray_left
    tray_height = tray_bottom - tray_top
    if (
        tray_width < subject_width * 1.35
        or tray_height < max(36, int(round(height * 0.08)))
        or tray_top >= subject_bottom
    ):
        raise RuntimeError(
            "Toileting canonical tray fallback could not form a broad underbody tray."
        )

    scale = 4
    tray = np.zeros((height * scale, width * scale, 4), dtype=np.uint8)

    def points(values: list[tuple[int, int]]) -> np.ndarray:
        return np.asarray(
            [[int(x * scale), int(y * scale)] for x, y in values],
            dtype=np.int32,
        )

    shoulder = max(10, int(round(tray_width * 0.035)))
    inset = max(20, int(round(tray_width * 0.075)))
    front_top = tray_top + int(round(tray_height * 0.62))
    outer = points(
        [
            (tray_left + shoulder, tray_top),
            (tray_right - shoulder, tray_top),
            (tray_right, tray_bottom),
            (tray_left, tray_bottom),
        ]
    )
    cv2.fillPoly(tray, [outer], (226, 229, 227, 255))

    grid = points(
        [
            (tray_left + inset, tray_top + 9),
            (tray_right - inset, tray_top + 9),
            (tray_right - inset - shoulder, front_top),
            (tray_left + inset + shoulder, front_top),
        ]
    )
    grid_mask = np.zeros(tray.shape[:2], dtype=np.uint8)
    cv2.fillPoly(grid_mask, [grid], 255)
    grid_layer = np.zeros_like(tray)
    grid_layer[grid_mask > 0] = (211, 216, 213, 255)
    grid_x0 = tray_left + inset
    grid_x1 = tray_right - inset
    grid_y0 = tray_top + 9
    grid_y1 = front_top
    for fraction in np.linspace(0.08, 0.92, 9):
        x = int(round(grid_x0 + (grid_x1 - grid_x0) * float(fraction)))
        cv2.line(
            grid_layer,
            (x * scale, grid_y0 * scale),
            (x * scale, grid_y1 * scale),
            (184, 190, 187, 255),
            max(2, scale),
            lineType=cv2.LINE_AA,
        )
    for fraction in np.linspace(0.10, 0.90, 6):
        y = int(round(grid_y0 + (grid_y1 - grid_y0) * float(fraction)))
        cv2.line(
            grid_layer,
            (grid_x0 * scale, y * scale),
            (grid_x1 * scale, y * scale),
            (184, 190, 187, 255),
            max(2, scale),
            lineType=cv2.LINE_AA,
        )
    tray[grid_mask > 0] = grid_layer[grid_mask > 0]

    front = points(
        [
            (tray_left, front_top - 2),
            (tray_right, front_top - 2),
            (tray_right, tray_bottom),
            (tray_left, tray_bottom),
        ]
    )
    cv2.fillPoly(tray, [front], (235, 237, 234, 255))
    cv2.polylines(
        tray,
        [outer],
        True,
        (207, 212, 209, 255),
        max(3, scale * 2),
        lineType=cv2.LINE_AA,
    )
    tray_image = Image.fromarray(tray, mode="RGBA").resize(
        (width, height),
        Image.Resampling.LANCZOS,
    )
    composite = Image.alpha_composite(tray_image, canonical)
    target.parent.mkdir(parents=True, exist_ok=True)
    composite.save(target, format="PNG", optimize=True)

    composite_alpha = np.asarray(composite.getchannel("A"), dtype=np.uint8) >= 16
    composite_rows, composite_columns = np.where(composite_alpha)
    composite_bbox = [
        int(composite_columns.min()),
        int(composite_rows.min()),
        int(composite_columns.max()) + 1,
        int(composite_rows.max()) + 1,
    ]
    composite_aspect = (composite_bbox[2] - composite_bbox[0]) / float(
        max(1, composite_bbox[3] - composite_bbox[1])
    )
    if composite_aspect < TOILETING_TRAY_MIN_REFERENCE_ASPECT_RATIO:
        raise RuntimeError(
            "Toileting canonical tray fallback remained too narrow after compositing."
        )
    return {
        "policy": TOILETING_CANONICAL_TRAY_FALLBACK_POLICY,
        "source": str(source),
        "target": str(target),
        "subject_bbox": [
            subject_left,
            subject_top,
            subject_right,
            subject_bottom,
        ],
        "tray_bbox": [tray_left, tray_top, tray_right, tray_bottom],
        "composite_bbox": composite_bbox,
        "composite_aspect_ratio": round(composite_aspect, 6),
        "tray_width_to_subject_width": round(tray_width / subject_width, 6),
        "subject_tray_vertical_overlap_px": subject_bottom - tray_top,
        "tray_drawn_behind_identity": True,
        "white_model_or_generated_identity_used": False,
    }


POSE_ATTEMPT_PROVENANCE_FIELDS = (
    "identity_input_policy",
    "image1_input_name",
    "image2_input_name",
    "image3_input_name",
    "sampler_latent_input_name",
    "sampler_noise_mask_input_name",
    "identity_input_name",
    "geometry_input_name",
)
POSE_ATTEMPT_REQUIRED_PROVENANCE_FIELDS = (
    "image1_input_name",
    "sampler_latent_input_name",
    "identity_input_name",
)


def _pose_workflow_load_image_names(record: dict[str, Any]) -> list[str]:
    workflow = record.get("workflow")
    if not isinstance(workflow, dict):
        return []
    prompt = workflow.get("prompt")
    if isinstance(prompt, dict):
        workflow = prompt
    names: list[str] = []
    for node in workflow.values():
        if not isinstance(node, dict) or node.get("class_type") != "LoadImage":
            continue
        inputs = node.get("inputs")
        image_name = inputs.get("image") if isinstance(inputs, dict) else None
        if isinstance(image_name, str) and image_name.strip():
            names.append(image_name.strip())
    return list(dict.fromkeys(names))


def pose_attempt_provenance(record: dict[str, Any]) -> dict[str, Any]:
    """Return a total, serialization-safe pose workflow provenance payload."""

    settings = record.get("settings")
    settings = settings if isinstance(settings, dict) else {}
    workflow_input_names = _pose_workflow_load_image_names(record)
    values: dict[str, str | None] = {}
    field_sources: dict[str, str] = {}
    for field in POSE_ATTEMPT_PROVENANCE_FIELDS:
        raw_value = record.get(field)
        source = "attempt_record"
        if not str(raw_value or "").strip():
            raw_value = settings.get(field)
            source = "attempt_settings"
        values[field] = (
            str(raw_value).strip() if str(raw_value or "").strip() else None
        )
        field_sources[field] = source if values[field] else "absent_optional"
    if not values["image1_input_name"] and workflow_input_names:
        values["image1_input_name"] = workflow_input_names[0]
        field_sources["image1_input_name"] = "workflow_load_image_1"
    if not values["image2_input_name"] and len(workflow_input_names) >= 2:
        values["image2_input_name"] = workflow_input_names[1]
        field_sources["image2_input_name"] = "workflow_load_image_2"
    if not values["image3_input_name"] and len(workflow_input_names) >= 3:
        values["image3_input_name"] = workflow_input_names[2]
        field_sources["image3_input_name"] = "workflow_load_image_3"
    if not values["sampler_latent_input_name"]:
        values["sampler_latent_input_name"] = values["image1_input_name"]
        field_sources["sampler_latent_input_name"] = "inferred_from_image1"
    if not values["identity_input_name"]:
        values["identity_input_name"] = values["image1_input_name"]
        field_sources["identity_input_name"] = "inferred_from_image1"
    if (
        not values["geometry_input_name"]
        and values["image2_input_name"]
    ):
        values["geometry_input_name"] = values["image2_input_name"]
        field_sources["geometry_input_name"] = "inferred_from_image2"
    required_fields = list(POSE_ATTEMPT_REQUIRED_PROVENANCE_FIELDS)
    if values["image2_input_name"] or values["image3_input_name"]:
        required_fields.append("geometry_input_name")
    missing_fields = [
        field
        for field in required_fields
        if not str(values.get(field) or "").strip()
    ]
    mapped_inputs = {
        str(value).strip()
        for field, value in values.items()
        if field.endswith("_input_name") and str(value or "").strip()
    }
    unmapped_workflow_inputs = [
        name for name in workflow_input_names if name not in mapped_inputs
    ]
    return {
        **values,
        "complete": not missing_fields and not unmapped_workflow_inputs,
        "required_fields": required_fields,
        "missing_fields": missing_fields,
        "field_sources": field_sources,
        "workflow_load_image_names": workflow_input_names,
        "unmapped_workflow_input_names": unmapped_workflow_inputs,
        "issues": [
            *missing_fields,
            *[
                f"unmapped_workflow_input:{name}"
                for name in unmapped_workflow_inputs
            ],
        ],
    }


def strict_first_frame_identity_palette_evidence(
    record: dict[str, Any],
    *,
    action: str,
) -> tuple[bool, str]:
    """Resolve the authoritative palette gate for a validated first frame."""

    _sync_local_video_runtime(
        globals(),
        {"strict_first_frame_identity_palette_evidence"},
    )
    palette = record.get("identity_palette")
    if isinstance(palette, dict):
        return (
            palette.get("acceptable") is True,
            "current_uploaded_identity_palette",
        )

    grooming_minimum = record.get("grooming_advisory_minimum_gate")
    grooming_palette = (
        grooming_minimum.get("identity_palette")
        if isinstance(grooming_minimum, dict)
        else None
    )
    if action_key(action) == "grooming" and isinstance(grooming_palette, dict):
        return (
            grooming_palette.get("acceptable") is True,
            "grooming_minimum_contract",
        )
    return False, "missing_identity_palette_evidence"


def revalidate_cached_grooming_minimum_gate(
    record: dict[str, Any],
    *,
    candidate_path: Path,
    identity_reference_palette: dict[str, Any],
    identity_palette_reference_path: Path,
    identity_reference_path: Path,
    geometry_guide: Path,
    geometry_metadata: Path,
    topology_guide: Path | None = None,
) -> dict[str, Any]:
    """Rebind and rerun a cached grooming contract against current-job inputs."""

    _sync_local_video_runtime(
        globals(),
        {"revalidate_cached_grooming_minimum_gate"},
    )
    gate = evaluate_advisory_grooming_minimum_gate(
        candidate_path,
        identity_reference_palette=identity_reference_palette,
        identity_palette_reference_path=identity_palette_reference_path,
        identity_reference_path=identity_reference_path,
        geometry_guide=geometry_guide,
        geometry_metadata=geometry_metadata,
        topology_guide=topology_guide,
        attempt=max(1, int(record.get("attempt") or 1)),
    )
    gate = json.loads(json.dumps(gate))
    gate.update(
        cache_revalidated=True,
        cache_revalidated_at=utc_now(),
        cache_source_job_id=(
            (record.get("validated_first_frame_cache") or {}).get(
                "source_job_id"
            )
            if isinstance(record.get("validated_first_frame_cache"), dict)
            else None
        ),
    )
    record["grooming_advisory_minimum_gate"] = gate
    return gate


def strict_first_frame_record_issues(
    record: dict[str, Any],
    *,
    action: str,
) -> list[str]:
    """Explain why an action reference is unsafe to publish as frame zero."""

    _sync_local_video_runtime(
        globals(),
        {"strict_first_frame_record_issues"},
    )
    issues: list[str] = []
    if record.get("reference_generation_degraded") is True:
        issues.append("reference_generation_degraded")
    if record.get("candidate_gate_degraded_fallback") is True:
        issues.append("candidate_gate_degraded_fallback")
    if isinstance(record.get("canonical_identity_fallback"), dict):
        issues.append("canonical_identity_fallback")

    for field in ("chroma", "quality", "similarity"):
        value = record.get(field)
        if not isinstance(value, dict) or value.get("acceptable") is not True:
            issues.append(f"{field}_not_acceptable")

    chroma = record.get("chroma")
    edge = (
        chroma.get("transparent_edge_quality")
        if isinstance(chroma, dict)
        else None
    )
    if not isinstance(edge, dict) or edge.get("acceptable") is not True:
        issues.append("transparent_edge_not_acceptable")

    candidate_gate = record.get("action_candidate_gate")
    if (
        not isinstance(candidate_gate, dict)
        or candidate_gate.get("acceptable") is not True
        or candidate_gate.get("degraded_fallback_accepted") is True
    ):
        issues.append("action_candidate_gate_not_strictly_acceptable")

    grooming_minimum = record.get("grooming_advisory_minimum_gate")
    if action == "grooming":
        if (
            not isinstance(grooming_minimum, dict)
            or grooming_minimum.get("acceptable") is not True
            or grooming_minimum.get("degraded_fallback_accepted") is True
        ):
            issues.append("grooming_minimum_gate_not_strictly_acceptable")
        if (
            isinstance(record.get("validated_first_frame_cache"), dict)
            and isinstance(grooming_minimum, dict)
            and str(grooming_minimum.get("candidate_sha256") or "")
            != str(record.get("keyframe_sha256") or "")
        ):
            issues.append("grooming_minimum_gate_candidate_digest_changed")
    palette_acceptable, _palette_source = (
        strict_first_frame_identity_palette_evidence(record, action=action)
    )
    if not palette_acceptable:
        issues.append("identity_palette_not_acceptable")

    for field in ("fur_silhouette", "fur_surface_gate", "grooming_pose_gate"):
        value = record.get(field)
        if isinstance(value, dict) and value.get("acceptable") is not True:
            issues.append(f"{field}_not_acceptable")
    return list(dict.fromkeys(issues))


def validated_first_frame_cache_fingerprint(
    job: dict[str, Any],
    action: str,
) -> str:
    """Build an identity/action fingerprint that excludes stochastic Qwen output."""

    _sync_local_video_runtime(
        globals(),
        {"validated_first_frame_cache_fingerprint"},
    )
    normalized_action = action_key(action)
    views = job.get("uploaded_identity_views")
    identity_views: list[dict[str, str]] = []
    if isinstance(views, dict):
        for role in sorted(views):
            view = views.get(role)
            if not isinstance(view, dict):
                continue
            identity_views.append(
                {
                    "role": str(role),
                    "input_sha256": str(view.get("input_sha256") or ""),
                    "source_sha256": str(view.get("source_sha256") or ""),
                }
            )
    pose_reference_sha256s = job.get("pose_reference_input_sha256s")
    pose_source_sha256s = job.get("pose_reference_source_sha256s")
    pose_policies = job.get("pose_identity_input_policies")
    pose_reference_policies = job.get("pose_reference_policy")
    required_digests = [
        str(job.get("subject_profile_sha256") or ""),
        str(
            pose_reference_sha256s.get(normalized_action, "")
            if isinstance(pose_reference_sha256s, dict)
            else ""
        ),
        *[
            digest
            for view in identity_views
            for digest in (view["input_sha256"], view["source_sha256"])
        ],
    ]
    if (
        not identity_views
        or any(not re.fullmatch(r"[0-9a-f]{64}", digest) for digest in required_digests)
    ):
        raise ValueError(
            "Validated first-frame cache requires digest-bound upload views, "
            "Profile, and action guide."
        )
    payload = {
        "cache_version": QWEN_VALIDATED_FIRST_FRAME_CACHE_VERSION,
        "action": normalized_action,
        "species": str(job.get("species") or "").strip().lower(),
        "identity_views": identity_views,
        "subject_profile_sha256": str(job.get("subject_profile_sha256") or ""),
        "reference_image_profile": str(job.get("reference_image_profile") or ""),
        "reference_image_profile_revision": str(
            job.get("reference_image_profile_revision") or ""
        ),
        "pose_reference_input_sha256": str(
            pose_reference_sha256s.get(normalized_action, "")
            if isinstance(pose_reference_sha256s, dict)
            else ""
        ),
        "pose_reference_source_sha256": str(
            pose_source_sha256s.get(normalized_action, "")
            if isinstance(pose_source_sha256s, dict)
            else ""
        ),
        "pose_identity_input_policy": str(
            pose_policies.get(
                normalized_action,
                job.get("pose_identity_input_policy") or "",
            )
            if isinstance(pose_policies, dict)
            else job.get("pose_identity_input_policy") or ""
        ),
        "pose_reference_policy": str(
            pose_reference_policies.get(normalized_action, "")
            if isinstance(pose_reference_policies, dict)
            else pose_reference_policies or ""
        ),
        "qwen_reference_prompt_en": str(job.get("qwen_reference_prompt_en") or ""),
        "qwen_seed": int(job.get("qwen_seed", job.get("seed", 0)) or 0),
        "qwen_steps": int(job.get("qwen_steps", 0) or 0),
        "qwen_cfg": float(job.get("qwen_cfg", 0.0) or 0.0),
        "qwen_denoise": float(job.get("qwen_denoise", 0.0) or 0.0),
        "user_loras": job.get("user_loras") or [],
        "action_candidate_gate_policy": GENERATION_ACTION_CANDIDATE_GATE_POLICY,
        "first_frame_policy": QWEN_VALIDATED_FIRST_FRAME_POLICY,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validated_first_frame_image_issues(
    path: Path,
    *,
    expected_sha256: str,
) -> list[str]:
    issues: list[str] = []
    if not path.is_file():
        return ["reference_missing"]
    actual_sha256 = file_sha256(path)
    if actual_sha256 != expected_sha256:
        issues.append("reference_digest_changed")
    try:
        with Image.open(path) as loaded:
            rgba = loaded.convert("RGBA")
            if rgba.size != (TRANSPARENT_GIF_SIZE, TRANSPARENT_GIF_SIZE):
                issues.append("reference_size_changed")
            alpha = np.asarray(rgba.getchannel("A"), dtype=np.uint8)
    except (OSError, ValueError, UnidentifiedImageError):
        return [*issues, "reference_image_unreadable"]
    foreground_fraction = float(np.count_nonzero(alpha >= 16)) / float(alpha.size)
    if foreground_fraction < 0.02 or foreground_fraction > 0.92:
        issues.append("reference_alpha_subject_fraction_invalid")
    if any(int(alpha[y, x]) != 0 for x, y in ((0, 0), (-1, 0), (0, -1), (-1, -1))):
        issues.append("reference_alpha_corner_not_clear")
    return list(dict.fromkeys(issues))


def _quality_record_as_first_frame_attempt(
    quality: dict[str, Any],
    workflow: dict[str, Any],
    *,
    reference_name: str,
    source_job_id: str,
    current_job_id: str,
    cache_fingerprint: str,
    reference_sha256: str,
) -> dict[str, Any]:
    selected_attempt = int(quality.get("selected_attempt") or 1)
    selected_attempt_record = next(
        (
            item
            for item in list(quality.get("attempts") or [])
            if isinstance(item, dict)
            and int(item.get("attempt") or -1) == selected_attempt
        ),
        {},
    )

    def remap(value: Any) -> Any:
        if isinstance(value, str):
            return value.replace(
                f"local_video_api/{source_job_id}/",
                f"local_video_api/{current_job_id}/",
            )
        if isinstance(value, list):
            return [remap(item) for item in value]
        if isinstance(value, dict):
            return {key: remap(item) for key, item in value.items()}
        return value

    settings = remap(dict(quality.get("selected_settings") or {}))
    record: dict[str, Any] = {
        "attempt": selected_attempt,
        "profile": str(quality.get("selected_profile") or "validated_cache"),
        "seed": selected_attempt_record.get("seed"),
        "identity_input_policy": quality.get("identity_input_policy"),
        "image1_input_name": remap(quality.get("image1_input_name")),
        "image2_input_name": remap(quality.get("image2_input_name")),
        "image3_input_name": remap(quality.get("image3_input_name")),
        "sampler_latent_input_name": remap(
            quality.get("sampler_latent_input_name")
        ),
        "sampler_noise_mask_input_name": remap(
            quality.get("sampler_noise_mask_input_name")
        ),
        "identity_input_name": remap(quality.get("identity_input_name")),
        "geometry_input_name": remap(quality.get("geometry_input_name")),
        "keyframe": reference_name,
        "keyframe_sha256": reference_sha256,
        "cutout": reference_name,
        "chroma": json.loads(json.dumps(quality.get("selected_chroma") or {})),
        "quality": json.loads(json.dumps(quality.get("selected_quality") or {})),
        "similarity": json.loads(
            json.dumps(quality.get("selected_similarity") or {})
        ),
        "identity_palette": json.loads(
            json.dumps(quality.get("selected_identity_palette"))
        ),
        "candidate_identity_palette": json.loads(
            json.dumps(quality.get("selected_candidate_identity_palette"))
        ),
        "candidate_upper_body_identity_palette": json.loads(
            json.dumps(
                quality.get("selected_candidate_upper_body_identity_palette")
            )
        ),
        "action_candidate_gate": json.loads(
            json.dumps(quality.get("selected_action_candidate_gate") or {})
        ),
        "canonical_identity_fallback": quality.get(
            "selected_canonical_identity_fallback"
        ),
        "fur_silhouette": json.loads(
            json.dumps(quality.get("selected_fur_silhouette"))
        ),
        "fur_surface_gate": json.loads(
            json.dumps(quality.get("selected_fur_surface_gate"))
        ),
        "grooming_pose_gate": json.loads(
            json.dumps(quality.get("selected_grooming_pose_gate"))
        ),
        "grooming_advisory_minimum_gate": json.loads(
            json.dumps(quality.get("selected_grooming_advisory_minimum_gate"))
        ),
        "color_recovery": json.loads(
            json.dumps(quality.get("selected_color_recovery"))
        ),
        "grooming_foreleg_repair": quality.get("grooming_foreleg_repair"),
        "settings": settings,
        "workflow": remap(json.loads(json.dumps(workflow))),
        "transparent": True,
        "background_removal_failed": False,
        "reference_generation_degraded": False,
        "validated_first_frame_cache": {
            "policy": QWEN_VALIDATED_FIRST_FRAME_POLICY,
            "cache_version": QWEN_VALIDATED_FIRST_FRAME_CACHE_VERSION,
            "fingerprint": cache_fingerprint,
            "source_job_id": source_job_id,
            "source_reference_sha256": reference_sha256,
            "integrity_verified": True,
            "strict_quality_record_verified": True,
        },
    }
    record["combined_penalty"] = round(
        sum(
            float(value.get("penalty", 0.0))
            for value in (
                record.get("quality"),
                record.get("similarity"),
                record.get("identity_palette"),
                record.get("fur_silhouette"),
                record.get("fur_surface_gate"),
                record.get("grooming_pose_gate"),
                record.get("grooming_advisory_minimum_gate"),
            )
            if isinstance(value, dict)
        ),
        6,
    )
    return record


def _write_validated_first_frame_cache_entry(
    *,
    fingerprint: str,
    action: str,
    source_job_id: str,
    reference_path: Path,
    quality: dict[str, Any],
    workflow: dict[str, Any],
) -> dict[str, Any]:
    reference_sha256 = file_sha256(reference_path)
    image_issues = _validated_first_frame_image_issues(
        reference_path,
        expected_sha256=reference_sha256,
    )
    record = _quality_record_as_first_frame_attempt(
        quality,
        workflow,
        reference_name="reference.png",
        source_job_id=source_job_id,
        current_job_id=source_job_id,
        cache_fingerprint=fingerprint,
        reference_sha256=reference_sha256,
    )
    quality_issues = strict_first_frame_record_issues(record, action=action)
    if image_issues or quality_issues:
        raise RuntimeError(
            "Refusing to cache an unvalidated first frame: "
            + ", ".join([*image_issues, *quality_issues])
        )
    cache_dir = VALIDATED_FIRST_FRAME_CACHE_ROOT / fingerprint
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_reference = cache_dir / "reference.png"
    temporary_reference = cache_dir / f".reference.{uuid4().hex}.tmp"
    shutil.copy2(reference_path, temporary_reference)
    if file_sha256(temporary_reference) != reference_sha256:
        temporary_reference.unlink(missing_ok=True)
        raise RuntimeError("Validated first-frame cache copy changed its digest.")
    os.replace(temporary_reference, cache_reference)
    manifest = {
        "cache_version": QWEN_VALIDATED_FIRST_FRAME_CACHE_VERSION,
        "policy": QWEN_VALIDATED_FIRST_FRAME_POLICY,
        "fingerprint": fingerprint,
        "action": action,
        "source_job_id": source_job_id,
        "source_pipeline_version": QWEN_FULL_BODY_FIRST_FRAME_PIPELINE_VERSION,
        "reference_filename": cache_reference.name,
        "reference_sha256": reference_sha256,
        "quality": quality,
        "workflow": workflow,
        "created_at": utc_now(),
    }
    write_json_atomic(cache_dir / "manifest.json", manifest)
    return {
        "fingerprint": fingerprint,
        "source_job_id": source_job_id,
        "reference_sha256": reference_sha256,
        "manifest": str(cache_dir / "manifest.json"),
    }


def _promote_historical_validated_first_frame(
    *,
    current_job_id: str,
    action: str,
    fingerprint: str,
) -> dict[str, Any] | None:
    status_paths = sorted(
        OUTPUT_ROOT.glob("*/status.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )[:QWEN_FIRST_FRAME_CACHE_HISTORY_SCAN_LIMIT]
    for source_status_path in status_paths:
        source_job_id = source_status_path.parent.name
        if source_job_id == current_job_id:
            continue
        try:
            source_job = json.loads(source_status_path.read_text(encoding="utf-8"))
            if source_job.get("status") != "completed":
                continue
            if source_job.get("reference_pipeline_version") not in (
                QWEN_VALIDATED_FIRST_FRAME_COMPATIBLE_PIPELINE_VERSIONS
            ):
                continue
            degraded = source_job.get("qwen_reference_degraded_actions")
            if isinstance(degraded, dict) and action in degraded:
                continue
            if validated_first_frame_cache_fingerprint(source_job, action) != fingerprint:
                continue
            source_actions = [
                action_key(value) for value in list(source_job.get("actions") or [])
            ]
            if action not in source_actions:
                continue
            source_pose_index = source_actions.index(action) + 1
            quality_path = (
                source_status_path.parent
                / "workflows"
                / f"qwen_pose_{source_pose_index:02d}_quality.json"
            )
            workflow_path = (
                source_status_path.parent
                / "workflows"
                / f"qwen_pose_{source_pose_index:02d}.json"
            )
            reference_paths = source_job.get("subject_reference_paths")
            source_reference = Path(
                str(
                    reference_paths.get(action, "")
                    if isinstance(reference_paths, dict)
                    else ""
                )
            )
            source_root = source_status_path.parent.resolve()
            try:
                source_reference.resolve().relative_to(source_root)
            except (OSError, ValueError):
                continue
            effective_config = source_job.get("effective_subject_config")
            effective_action = (
                effective_config.get("actions", {}).get(action)
                if isinstance(effective_config, dict)
                else None
            )
            if isinstance(effective_action, dict):
                configured_reference = Path(
                    str(effective_action.get("reference") or "")
                )
                configured_sha256 = str(
                    effective_action.get("reference_sha256") or ""
                )
                if (
                    configured_reference.resolve() != source_reference.resolve()
                    or configured_sha256 != file_sha256(source_reference)
                ):
                    continue
            if not quality_path.is_file() or not workflow_path.is_file():
                continue
            quality = json.loads(quality_path.read_text(encoding="utf-8"))
            workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
            return _write_validated_first_frame_cache_entry(
                fingerprint=fingerprint,
                action=action,
                source_job_id=source_job_id,
                reference_path=source_reference,
                quality=quality,
                workflow=workflow,
            )
        except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError):
            LOGGER.warning(
                "Ignoring invalid historical first-frame candidate %s for %s",
                source_job_id,
                action,
                exc_info=True,
            )
    return None


def load_validated_first_frame_cache(
    job_id: str,
    job: dict[str, Any],
    *,
    action: str,
    pose_index: int,
    job_dir: Path,
) -> dict[str, Any] | None:
    """Return a digest-checked, strictly gated frame from this exact input set."""

    _sync_local_video_runtime(
        globals(),
        {"load_validated_first_frame_cache"},
    )
    normalized_action = action_key(action)
    try:
        fingerprint = validated_first_frame_cache_fingerprint(
            job,
            normalized_action,
        )
    except (TypeError, ValueError):
        LOGGER.warning(
            "%s cannot use the validated first-frame cache for %s because its "
            "input digest contract is incomplete",
            job_id,
            normalized_action,
        )
        return None
    cache_dir = VALIDATED_FIRST_FRAME_CACHE_ROOT / fingerprint
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        _promote_historical_validated_first_frame(
            current_job_id=job_id,
            action=normalized_action,
            fingerprint=fingerprint,
        )
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("cache_version")
            != QWEN_VALIDATED_FIRST_FRAME_CACHE_VERSION
            or manifest.get("policy") != QWEN_VALIDATED_FIRST_FRAME_POLICY
            or manifest.get("fingerprint") != fingerprint
            or manifest.get("action") != normalized_action
        ):
            return None
        reference_sha256 = str(manifest.get("reference_sha256") or "")
        reference_filename = str(
            manifest.get("reference_filename") or "reference.png"
        )
        if Path(reference_filename).name != reference_filename:
            return None
        cache_reference = cache_dir / reference_filename
        if _validated_first_frame_image_issues(
            cache_reference,
            expected_sha256=reference_sha256,
        ):
            return None
        quality = manifest.get("quality")
        workflow = manifest.get("workflow")
        if not isinstance(quality, dict) or not isinstance(workflow, dict):
            return None
        target = job_dir / f"pose_{pose_index:02d}_validated_first_frame_cache.png"
        temporary_target = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        shutil.copy2(cache_reference, temporary_target)
        if file_sha256(temporary_target) != reference_sha256:
            temporary_target.unlink(missing_ok=True)
            return None
        os.replace(temporary_target, target)
        record = _quality_record_as_first_frame_attempt(
            quality,
            workflow,
            reference_name=target.name,
            source_job_id=str(manifest.get("source_job_id") or ""),
            current_job_id=job_id,
            cache_fingerprint=fingerprint,
            reference_sha256=reference_sha256,
        )
        if strict_first_frame_record_issues(record, action=normalized_action):
            return None
        return record
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        LOGGER.warning(
            "Ignoring invalid validated first-frame cache entry %s",
            manifest_path,
            exc_info=True,
        )
        return None


def should_reuse_validated_first_frame_cache(
    *,
    strict_first_frame_required: bool,
    first_frame_retry_wave: int,
) -> bool:
    """Keep fixed-budget generation independent from historical cache records."""

    _sync_local_video_runtime(
        globals(),
        {"should_reuse_validated_first_frame_cache"},
    )
    return bool(
        QWEN_VALIDATED_FIRST_FRAME_CACHE_REUSE_ENABLED
        and strict_first_frame_required
        and first_frame_retry_wave == 0
    )


def store_validated_first_frame_cache(
    job_id: str,
    job: dict[str, Any],
    *,
    action: str,
    pose_index: int,
    reference_path: Path,
) -> dict[str, Any]:
    """Publish a selected frame only after its persisted strict audit is complete."""

    _sync_local_video_runtime(
        globals(),
        {"store_validated_first_frame_cache"},
    )
    quality_path = (
        reference_path.parent
        / "workflows"
        / f"qwen_pose_{pose_index:02d}_quality.json"
    )
    workflow_path = (
        reference_path.parent / "workflows" / f"qwen_pose_{pose_index:02d}.json"
    )
    if not quality_path.is_file() or not workflow_path.is_file():
        raise RuntimeError("Selected first-frame audit files are incomplete.")
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    fingerprint = validated_first_frame_cache_fingerprint(job, action)
    return _write_validated_first_frame_cache_entry(
        fingerprint=fingerprint,
        action=action_key(action),
        source_job_id=job_id,
        reference_path=reference_path,
        quality=quality,
        workflow=workflow,
    )


def resolve_action_identity_palette_reference(
    *,
    uploaded_reference_job: bool,
    identity_input_policy: str,
    uploaded_identity_source: Path,
    canonical_reference: Path | None,
) -> Path | None:
    """Keep uploaded action color conditioning anchored to the raw identity."""
    _sync_local_video_runtime(
        globals(),
        {"resolve_action_identity_palette_reference"},
    )
    if uploaded_reference_job or identity_input_policy in {
        POSE_IDENTITY_INPUT_POLICY_DIRECT_UPLOAD,
        POSE_IDENTITY_INPUT_POLICY_PROFILE_GEOMETRY_FIRST,
    }:
        return uploaded_identity_source
    return canonical_reference


def run_job(job_id: str, *, force_qwen_reference: bool = False) -> None:
    _sync_local_video_runtime(globals(), {"run_job"})
    raise_if_cancelled(job_id)
    with JOBS_LOCK:
        job = dict(JOBS[job_id])
    species = str(job.get("species") or "cat").strip().lower()
    if job.get("generation_mode") == "subject_replacement" and not force_qwen_reference:
        run_subject_replacement_job(job_id)
        return
    job_dir = OUTPUT_ROOT / job_id
    poses: list[str] = list(
        job.get("qwen_reference_poses")
        if force_qwen_reference
        else (job.get("poses_en") or job["poses"])
    )
    effective_prompt = str(
        job.get("qwen_reference_prompt_en")
        if force_qwen_reference
        else (job.get("prompt_en") or job["prompt"])
    )
    profile_fur_lock = str(job.get("subject_profile_fur_lock") or "").strip()
    profile_fur_negative = str(job.get("subject_profile_fur_negative") or "").strip()
    width, height = (
        (TRANSPARENT_GIF_SIZE, TRANSPARENT_GIF_SIZE)
        if force_qwen_reference
        else RESOLUTIONS[job["resolution"]]
    )
    transparent_gif = (
        True
        if force_qwen_reference
        else job.get("output_format", "mp4") == "transparent_gif"
    )
    skip_video = True if force_qwen_reference else bool(job.get("skip_video", False))
    request_kind = (
        "video"
        if force_qwen_reference
        else str(job.get("request_kind") or "video").strip().lower()
    )
    image_request = request_kind == "image"
    suppress_builtin_style = bool(
        job.get("image_profile_lora_source")
        in {"image_profile", "image_profile_plus_user"}
    )
    user_loras = normalize_user_lora_specs(
        job.get("user_loras"),
        legacy_model_name=job.get("user_lora"),
        legacy_original_filename=job.get("user_lora_original_filename"),
        legacy_strength=(
            job["user_lora_strength"]
            if job.get("user_lora_strength") is not None
            else 1.0
        ),
    )
    canonical_user_loras, canonical_lora_scale = cap_user_lora_specs(
        user_loras,
        total_cap=float(
            job.get(
                "canonical_user_lora_total_cap",
                QWEN_CANONICAL_USER_LORA_TOTAL_CAP,
            )
        ),
        per_lora_cap=float(
            job.get(
                "canonical_user_lora_per_lora_cap",
                QWEN_CANONICAL_USER_LORA_PER_LORA_CAP,
            )
        ),
    )
    first_canonical_user_lora = (
        canonical_user_loras[0] if canonical_user_loras else None
    )
    user_lora = (
        first_canonical_user_lora["model_name"] if first_canonical_user_lora else None
    )
    user_lora_strength = (
        first_canonical_user_lora["strength"] if first_canonical_user_lora else 1.0
    )
    user_lora_mode = str(job.get("user_lora_mode") or "replace")
    qwen_steps = int(job.get("qwen_steps", 4))
    qwen_cfg = float(job.get("qwen_cfg", 1.0))
    qwen_denoise = float(job.get("qwen_denoise", 1.0))
    wan_steps = int(job.get("wan_steps", WAN_STEPS))
    wan_cfg = float(job.get("wan_cfg", WAN_CFG))
    wan_shift = float(job.get("wan_shift", WAN_SHIFT))
    # Queue-time SCAIL seeds and Qwen image-edit seeds are separate contracts.
    # Older persisted jobs have only ``seed`` and retain their historic behavior.
    base_seed = int(
        job.get("qwen_seed", job["seed"])
        if force_qwen_reference
        else job["seed"]
    )
    comfy_relative_dir = Path("local_video_api") / job_id
    comfy_input_dir = COMFY_INPUT / comfy_relative_dir
    comfy_input_dir.mkdir(parents=True, exist_ok=True)
    upload_integrity = (
        verify_uploaded_identity_submission(job, job_dir)
        if force_qwen_reference
        else None
    )
    canonical_identity_reference_input_names: list[str] = []
    comfy_identity_views: dict[str, dict[str, Any]] = {}
    if upload_integrity is not None:
        for role in upload_integrity["view_roles"]:
            view = upload_integrity["views"][role]
            comfy_view_path = comfy_input_dir / view["input_filename"]
            shutil.copy2(view["input_path"], comfy_view_path)
            if file_sha256(comfy_view_path) != view["input_sha256"]:
                raise RuntimeError(
                    f"Uploaded identity {role} ComfyUI copy failed its "
                    "submission-time digest check."
                )
            comfy_identity_views[role] = {
                "path": str(comfy_view_path),
                "sha256": view["input_sha256"],
                "qwen_input": f"image{len(comfy_identity_views) + 1}",
            }
            if role != "front":
                canonical_identity_reference_input_names.append(
                    (comfy_relative_dir / view["input_filename"]).as_posix()
                )
        source_input_path = upload_integrity["input_path"]
    else:
        source_input_path = job_dir / "input.png"
        shutil.copy2(source_input_path, comfy_input_dir / "input.png")
    if upload_integrity is not None:
        update_job(
            job_id,
            uploaded_identity_comfy_views=comfy_identity_views,
            uploaded_identity_comfy_view_count=len(comfy_identity_views),
            uploaded_identity_comfy_copy_verified=True,
        )

    update_job(
        job_id,
        status="running",
        stage="waiting_for_gpu",
        progress=1,
        started_at=job.get("started_at") or utc_now(),
        error=None,
        generation_audit_policy=GENERATION_AUDIT_POLICY,
        visual_content_audit_mode=(
            "blocking" if GENERATION_VISUAL_AUDITS_BLOCKING else "advisory"
        ),
        visual_content_audits_blocking=GENERATION_VISUAL_AUDITS_BLOCKING,
        runtime_integrity_checks_required=True,
    )
    raise_if_cancelled(job_id)
    require_models(include_video=not skip_video)
    wait_comfy_idle(job_id=job_id)
    raise_if_cancelled(job_id)
    free_comfy_models()

    recovery_profiles = qwen_recovery_profiles(
        user_lora=user_lora,
        user_lora_strength=user_lora_strength,
        user_lora_mode=user_lora_mode,
        steps=qwen_steps,
        cfg=qwen_cfg,
        denoise=qwen_denoise,
        user_loras=canonical_user_loras,
        lock_strengths=bool(
            job.get(
                "lock_canonical_retry_lora_strengths",
                job.get("lock_retry_lora_strengths", False),
            )
        ),
    )
    pose_profiles = qwen_pose_profiles(
        user_lora=(user_loras[0]["model_name"] if user_loras else None),
        user_lora_strength=(float(user_loras[0]["strength"]) if user_loras else 1.0),
        user_lora_mode=user_lora_mode,
        steps=qwen_steps,
        cfg=qwen_cfg,
        denoise=qwen_denoise,
        user_loras=user_loras,
        first_total_cap=job.get("pose_user_lora_first_total_cap"),
        first_per_lora_cap=job.get("pose_user_lora_first_per_lora_cap"),
        lock_strengths=bool(
            job.get(
                "lock_pose_retry_lora_strengths",
                job.get("lock_retry_lora_strengths", False),
            )
        ),
    )
    initial_pose_policy_default = str(
        job.get("pose_identity_input_policy") or POSE_IDENTITY_INPUT_POLICY_CANONICAL
    ).strip()
    initial_pose_policy_overrides = job.get("pose_identity_input_policies")
    if not isinstance(initial_pose_policy_overrides, dict):
        initial_pose_policy_overrides = {}
    initial_pose_policies = [
        str(
            initial_pose_policy_overrides.get(
                action_key(pose),
                initial_pose_policy_default,
            )
        ).strip()
        for pose in poses
    ]
    canonical_action_lora_lock = bool(
        force_qwen_reference
        and initial_pose_policies
        and all(
            policy == POSE_IDENTITY_INPUT_POLICY_CANONICAL
            for policy in initial_pose_policies
        )
    )
    grooming_masked_style_policy = bool(
        force_qwen_reference
        and POSE_IDENTITY_INPUT_POLICY_GROOMING_GEOMETRY_CANONICAL
        in initial_pose_policies
    )
    initial_pose_user_loras_disabled = bool(
        force_qwen_reference
        and initial_pose_policies
        and all(
            policy
            in {
                POSE_IDENTITY_INPUT_POLICY_CANONICAL,
                POSE_IDENTITY_INPUT_POLICY_PROFILE_GEOMETRY_FIRST,
                POSE_IDENTITY_INPUT_POLICY_GROOMING_GEOMETRY_CANONICAL,
            }
            for policy in initial_pose_policies
        )
    )
    initial_effective_pose_user_loras = (
        [] if initial_pose_user_loras_disabled else pose_profiles[0]["user_loras"]
    )
    initial_effective_pose_lora_scale = (
        0.0
        if initial_pose_user_loras_disabled
        else pose_profiles[0]["user_lora_strength_scale"]
    )
    update_job(
        job_id,
        requested_user_loras=user_loras,
        effective_canonical_user_loras=(
            []
            if (
                force_qwen_reference
                and job.get("main_case_raw_identity_chain")
                and job.get("bypass_generated_canonical")
            )
            else recovery_profiles[0]["user_loras"]
        ),
        canonical_user_lora_strength_scale=(
            None
            if (
                force_qwen_reference
                and job.get("main_case_raw_identity_chain")
                and job.get("bypass_generated_canonical")
            )
            else canonical_lora_scale
        ),
        initial_pose_user_loras=pose_profiles[0]["user_loras"],
        initial_pose_user_lora_strength_scale=pose_profiles[0][
            "user_lora_strength_scale"
        ],
        effective_pose_user_loras=initial_effective_pose_user_loras,
        pose_user_lora_strength_scale=initial_effective_pose_lora_scale,
        effective_pose_user_loras_scope=(
            "grooming_attempts_1_2_lightning_only; attempt_3_masked_pet3d; "
            "use effective_pose_lora_attempts for selected per-pose settings"
            if grooming_masked_style_policy
            else (
                "canonical_action_lightning_only; use effective_pose_lora_attempts "
                "for selected per-pose settings"
                if canonical_action_lora_lock
                else (
                    "geometry_first_action_user_loras_disabled; use "
                    "effective_pose_lora_attempts for selected per-pose settings"
                    if initial_pose_user_loras_disabled
                    else "initial_profile_only; use effective_pose_lora_attempts for selected per-pose settings"
                )
            )
        ),
        pose_lora_policy={
            "canonical_action_override": (
                "qwen_lightning_only_after_canonical_identity_style_lock_v1"
                if canonical_action_lora_lock
                else None
            ),
            "grooming_repair_override": (
                "qwen_lightning_plus_masked_pet3d_surface_prior_v1"
                if grooming_masked_style_policy
                else None
            ),
            "attempts": [
                {
                    "profile": profile["name"],
                    "total_strength_cap": profile["total_strength_cap"],
                    "per_lora_strength_cap": profile["per_lora_strength_cap"],
                    "strength_scale": profile["user_lora_strength_scale"],
                }
                for profile in pose_profiles
            ],
        },
        effective_pose_lora_attempts=[],
        pose_similarity_warning_count=0,
        pose_identity_palette_warning_count=0,
    )
    qwen_input_name = (comfy_relative_dir / "input.png").as_posix()
    uploaded_qwen_input_name = qwen_input_name
    pose_identity_input_policy = str(
        job.get("pose_identity_input_policy") or POSE_IDENTITY_INPUT_POLICY_CANONICAL
    ).strip()
    configured_pose_identity_input_policies = job.get("pose_identity_input_policies")
    if configured_pose_identity_input_policies is None:
        configured_pose_identity_input_policies = {}
    if not isinstance(configured_pose_identity_input_policies, dict):
        raise RuntimeError("Persisted per-action pose identity policies are invalid.")
    if pose_identity_input_policy not in {
        POSE_IDENTITY_INPUT_POLICY_CANONICAL,
        POSE_IDENTITY_INPUT_POLICY_DIRECT_UPLOAD,
        POSE_IDENTITY_INPUT_POLICY_PROFILE_GEOMETRY_FIRST,
        POSE_IDENTITY_INPUT_POLICY_GROOMING_GEOMETRY_CANONICAL,
    }:
        raise RuntimeError(
            "Persisted pose identity input policy is unsupported: "
            f"{pose_identity_input_policy}"
        )
    if (
        pose_identity_input_policy
        in {
            POSE_IDENTITY_INPUT_POLICY_DIRECT_UPLOAD,
            POSE_IDENTITY_INPUT_POLICY_PROFILE_GEOMETRY_FIRST,
            POSE_IDENTITY_INPUT_POLICY_GROOMING_GEOMETRY_CANONICAL,
        }
        and not force_qwen_reference
    ):
        raise RuntimeError(
            "The selected secondary-image pose identity policy is reserved for "
            "the frozen subject-replacement reference pipeline."
        )
    geometry_first_configured = (
        pose_identity_input_policy == POSE_IDENTITY_INPUT_POLICY_PROFILE_GEOMETRY_FIRST
    )
    grooming_geometry_canonical_configured = (
        pose_identity_input_policy
        == POSE_IDENTITY_INPUT_POLICY_GROOMING_GEOMETRY_CANONICAL
    )
    if geometry_first_configured and any(
        action_key(pose) not in {"eating", "grooming"} for pose in poses
    ):
        raise RuntimeError(
            "Main_case geometry-first identity supports only eating and grooming."
        )
    for (
        configured_action,
        configured_policy,
    ) in configured_pose_identity_input_policies.items():
        normalized_configured_policy = str(configured_policy).strip()
        if normalized_configured_policy not in {
            POSE_IDENTITY_INPUT_POLICY_CANONICAL,
            POSE_IDENTITY_INPUT_POLICY_DIRECT_UPLOAD,
            POSE_IDENTITY_INPUT_POLICY_PROFILE_GEOMETRY_FIRST,
            POSE_IDENTITY_INPUT_POLICY_GROOMING_GEOMETRY_CANONICAL,
        }:
            raise RuntimeError(
                "Persisted pose identity input policy is unsupported for "
                f"{configured_action!r}: {configured_policy!r}"
            )
        if (
            normalized_configured_policy
            in {
                POSE_IDENTITY_INPUT_POLICY_DIRECT_UPLOAD,
                POSE_IDENTITY_INPUT_POLICY_PROFILE_GEOMETRY_FIRST,
                POSE_IDENTITY_INPUT_POLICY_GROOMING_GEOMETRY_CANONICAL,
            }
            and not force_qwen_reference
        ):
            raise RuntimeError(
                "The selected per-action secondary-image identity policy is "
                "reserved for the frozen subject-replacement reference pipeline."
            )
        if (
            normalized_configured_policy
            == POSE_IDENTITY_INPUT_POLICY_PROFILE_GEOMETRY_FIRST
        ):
            geometry_first_configured = True
            if action_key(str(configured_action)) not in {"eating", "grooming"}:
                raise RuntimeError(
                    "Main_case geometry-first identity is unsupported for "
                    f"action; refusing configured action {configured_action!r}."
                )
        if (
            normalized_configured_policy
            == POSE_IDENTITY_INPUT_POLICY_GROOMING_GEOMETRY_CANONICAL
        ):
            grooming_geometry_canonical_configured = True
            if action_key(str(configured_action)) != "grooming":
                raise RuntimeError(
                    "Canonical-identity geometry-first policy is supported only "
                    f"for grooming; refusing configured action {configured_action!r}."
                )
    main_case_raw_identity_chain = bool(job.get("main_case_raw_identity_chain"))
    bypass_generated_canonical = bool(
        force_qwen_reference
        and main_case_raw_identity_chain
        and job.get("bypass_generated_canonical")
    )
    if geometry_first_configured and not bypass_generated_canonical:
        raise RuntimeError(
            "Main_case geometry-first policy requires the original uploaded "
            "identity chain and generated-canonical bypass."
        )
    if geometry_first_configured:
        if not QWEN_STYLE_LORA:
            raise RuntimeError(
                "Main_case geometry-first uploaded identity requires the dedicated "
                "QWEN_STYLE_LORA recovery model."
            )
        if "pet3d" not in Path(QWEN_STYLE_LORA).name.casefold():
            raise RuntimeError(
                "Main_case geometry-first uploaded identity requires a dedicated pet3d "
                "QWEN_STYLE_LORA recovery model."
            )
        if QWEN_EATING_GEOMETRY_FIRST_STYLE_STRENGTH <= 0.0:
            raise RuntimeError(
                "Main_case geometry-first uploaded identity requires a positive pet3d "
                "recovery strength."
            )
    if grooming_geometry_canonical_configured:
        if not QWEN_STYLE_LORA:
            raise RuntimeError(
                "Canonical-identity grooming repair requires the dedicated "
                "QWEN_STYLE_LORA surface model."
            )
        if "pet3d" not in Path(QWEN_STYLE_LORA).name.casefold():
            raise RuntimeError(
                "Canonical-identity grooming repair requires a dedicated pet3d "
                "QWEN_STYLE_LORA surface model."
            )
        if GROOMING_QWEN_FORELEG_REPAIR_STYLE_STRENGTH <= 0.0:
            raise RuntimeError(
                "Canonical-identity grooming repair requires a positive masked "
                "Pet3D surface-prior strength."
            )
    if bypass_generated_canonical:
        unsupported_bypass_actions = [
            action_key(pose)
            for pose in poses
            if str(
                configured_pose_identity_input_policies.get(
                    action_key(pose),
                    pose_identity_input_policy,
                )
            ).strip()
            != POSE_IDENTITY_INPUT_POLICY_PROFILE_GEOMETRY_FIRST
        ]
        if unsupported_bypass_actions:
            raise RuntimeError(
                "Generated canonical bypass requires every action to use the "
                "Main_case Image 1 plus uploaded Image 2 policy: "
                + ", ".join(unsupported_bypass_actions)
            )
        expected_uploaded_identity_sha256 = str(
            job.get("uploaded_identity_input_sha256") or ""
        ).strip()
        uploaded_identity_path = job_dir / "input.png"
        comfy_uploaded_identity_path = comfy_input_dir / "input.png"
        if not expected_uploaded_identity_sha256:
            raise RuntimeError(
                "Main_case uploaded identity input lacks its persisted digest."
            )
        if (
            not uploaded_identity_path.is_file()
            or file_sha256(uploaded_identity_path) != expected_uploaded_identity_sha256
            or not comfy_uploaded_identity_path.is_file()
            or file_sha256(comfy_uploaded_identity_path)
            != expected_uploaded_identity_sha256
        ):
            raise RuntimeError(
                "Main_case uploaded identity input is missing or its digest changed."
            )
    requested_identity_palette_guard = bool(
        job.get("identity_palette_guard", False)
    )
    identity_palette_guidance_enabled = requested_identity_palette_guard
    canonical_identity_palette_guard_requested = bool(
        force_qwen_reference and requested_identity_palette_guard
    )
    identity_palette_guard = bool(
        GENERATION_VISUAL_AUDITS_BLOCKING
        and requested_identity_palette_guard
    )
    update_job(
        job_id,
        requested_identity_palette_guard=requested_identity_palette_guard,
        effective_identity_palette_guard=identity_palette_guard,
        canonical_identity_palette_guard_requested=(
            canonical_identity_palette_guard_requested
        ),
        identity_palette_guidance_enabled=identity_palette_guidance_enabled,
        identity_palette_guidance_mode=(
            "prompt_and_negative_conditioning"
            if identity_palette_guidance_enabled
            else "disabled"
        ),
        identity_palette_audit_status=(
            "enabled"
            if identity_palette_guard
            else (
                "enabled_for_action_candidates"
                if force_qwen_reference
                and GENERATION_ACTION_CANDIDATE_GATES_ENABLED
                else "skipped_advisory"
            )
        ),
        action_candidate_gate_enabled=bool(
            force_qwen_reference
            and GENERATION_ACTION_CANDIDATE_GATES_ENABLED
        ),
        action_candidate_gate_policy=GENERATION_ACTION_CANDIDATE_GATE_POLICY,
    )
    identity_source_path = job_dir / "input_identity_source.png"
    if force_qwen_reference:
        if upload_integrity is None:
            raise RuntimeError("Uploaded identity integrity state is unavailable.")
        identity_source_path = upload_integrity["source_path"]
    elif not identity_source_path.is_file():
        identity_source_path = job_dir / "input.png"
    if bypass_generated_canonical:
        expected_identity_source_sha256 = str(
            job.get("uploaded_identity_source_sha256") or ""
        ).strip()
        if not expected_identity_source_sha256:
            raise RuntimeError(
                "Main_case uploaded identity source lacks its persisted digest."
            )
        if (
            not identity_source_path.is_file()
            or file_sha256(identity_source_path) != expected_identity_source_sha256
        ):
            raise RuntimeError(
                "Main_case uploaded identity source is missing or its digest changed."
            )
    source_identity_view_palettes: dict[str, dict[str, Any]] = {}
    source_identity_view_upper_body_palettes: dict[str, dict[str, Any]] = {}
    if identity_palette_guidance_enabled:
        identity_views = (
            upload_integrity["views"]
            if upload_integrity is not None
            else {
                "front": {
                    "source_path": identity_source_path,
                }
            }
        )
        cached_view_audit = job.get("uploaded_identity_view_audit")
        cached_palettes = (
            cached_view_audit.get("palettes")
            if isinstance(cached_view_audit, dict)
            and cached_view_audit.get("acceptable") is True
            else None
        )
        cached_upper_body_palettes = (
            cached_view_audit.get("upper_body_palettes")
            if isinstance(cached_view_audit, dict)
            and cached_view_audit.get("acceptable") is True
            else None
        )
        if (
            isinstance(cached_palettes, dict)
            and isinstance(cached_upper_body_palettes, dict)
            and set(cached_palettes) == set(identity_views)
            and set(cached_upper_body_palettes) == set(identity_views)
        ):
            source_identity_view_palettes = {
                role: dict(cached_palettes[role]) for role in identity_views
            }
            source_identity_view_upper_body_palettes = {
                role: dict(cached_upper_body_palettes[role]) for role in identity_views
            }
        else:
            source_identity_view_palettes = {
                role: identity_palette_signature(Path(view["source_path"]))
                for role, view in identity_views.items()
            }
            source_identity_view_upper_body_palettes = {
                role: identity_palette_signature(
                    Path(view["source_path"]),
                    exclude_bottom_fraction=(
                        BOWL_IDENTITY_PALETTE_UPPER_BODY_EXCLUSION
                    ),
                )
                for role, view in identity_views.items()
            }
    source_identity_palette = (
        aggregate_identity_palette_signatures(source_identity_view_palettes)
        if source_identity_view_palettes
        else None
    )
    source_identity_upper_body_palette = (
        aggregate_identity_palette_signatures(source_identity_view_upper_body_palettes)
        if source_identity_view_upper_body_palettes
        else None
    )
    canonical_source_upper_body_palette = canonical_upper_body_palette_reference(
        source_identity_view_upper_body_palettes,
        source_identity_upper_body_palette,
    )
    canonical_identity_palette_gate_required = bool(
        canonical_identity_palette_guard_requested
        and isinstance(source_identity_palette, dict)
        and source_identity_palette.get("available") is True
        and isinstance(canonical_source_upper_body_palette, dict)
        and canonical_source_upper_body_palette.get("available") is True
    )
    source_palette_prompt = (
        identity_palette_prompt(source_identity_palette)
        if source_identity_palette is not None
        else ""
    )
    source_palette_negative = (
        identity_palette_negative_prompt(source_identity_palette)
        if source_identity_palette is not None
        else ""
    )
    if source_identity_palette is not None:
        update_job(
            job_id,
            source_identity_palette=source_identity_palette,
            source_identity_upper_body_palette=(source_identity_upper_body_palette),
            source_identity_view_palettes=source_identity_view_palettes,
            source_identity_view_upper_body_palettes=(
                source_identity_view_upper_body_palettes
            ),
            canonical_source_upper_body_palette=(canonical_source_upper_body_palette),
            canonical_upper_body_palette_reference_policy=(
                CANONICAL_UPPER_BODY_PALETTE_REFERENCE_POLICY
            ),
            source_identity_palette_aggregation=(
                "equal_view_mean_v1"
                if len(source_identity_view_palettes) > 1
                else "single_front_view"
            ),
            identity_palette_prompt=source_palette_prompt,
            canonical_identity_palette_gate_required=(
                canonical_identity_palette_gate_required
            ),
            canonical_identity_palette_gate_mode=(
                "blocking_with_uploaded_identity_fallback"
                if canonical_identity_palette_gate_required
                else "unavailable_or_not_requested"
            ),
        )
    canonical_similarity_reference: Path | None = None
    if transparent_gif and not bypass_generated_canonical:
        update_job(
            job_id,
            stage="generating_cartoon_reference",
            progress=3,
            current_pose_index=None,
            current_pose=None,
        )
        cartoon_negative_prompt = (
            QWEN_NEGATIVE
            + ", "
            + (DOG_CARTOON_NEGATIVE if species == "dog" else CARTOON_NEGATIVE)
            + ", "
            + CHROMA_NEGATIVE
        )
        if profile_fur_negative:
            cartoon_negative_prompt += ", " + profile_fur_negative
        if source_palette_negative:
            cartoon_negative_prompt += ", " + source_palette_negative
        cartoon_attempts: list[dict[str, Any]] = []
        cartoon_workflows: list[dict[str, Any]] = []
        selected_cartoon: dict[str, Any] | None = None
        canonical_profiles = (
            recovery_profiles
            if (
                GENERATION_VISUAL_AUDITS_BLOCKING
                or canonical_identity_palette_gate_required
            )
            else recovery_profiles[:1]
        )
        for attempt, profile in enumerate(canonical_profiles, start=1):
            raise_if_cancelled(job_id)
            if attempt > 1:
                update_job(
                    job_id,
                    stage="retrying_cartoon_reference",
                    recovery_active=True,
                    recovery_attempt=attempt,
                )
            attempt_seed = base_seed + (attempt - 1) * 1009
            prompt_source = effective_prompt if profile["use_user_prompt"] else ""
            palette_lock = (
                identity_palette_prompt(
                    source_identity_palette,
                    retry=attempt > 1,
                )
                if source_identity_palette is not None
                else ""
            )
            cartoon_reference_prompt = build_cartoon_reference_prompt(
                prompt_source,
                palette_lock=palette_lock,
                fur_lock=profile_fur_lock,
                species=species,
            )
            if (
                canonical_identity_reference_input_names
                and QWEN_MULTI_VIEW_CANONICAL_PROMPT not in cartoon_reference_prompt
            ):
                cartoon_reference_prompt += "\n\n" + QWEN_MULTI_VIEW_CANONICAL_PROMPT
            cartoon_prefix = (
                f"local_video_api/{job_id}/qwen_cartoon_reference_attempt_{attempt:02d}"
            )
            cartoon_workflow = qwen_workflow(
                qwen_input_name,
                cartoon_reference_prompt,
                cartoon_negative_prompt,
                attempt_seed,
                cartoon_prefix,
                user_lora=profile["user_lora"],
                user_lora_strength=profile["user_lora_strength"],
                user_lora_mode=profile["user_lora_mode"],
                user_loras=profile["user_loras"],
                steps=profile["steps"],
                cfg=profile["cfg"],
                denoise=profile["denoise"],
                suppress_builtin_style=suppress_builtin_style,
                additional_reference_input_names=(
                    canonical_identity_reference_input_names
                ),
            )
            cartoon_workflows.append(cartoon_workflow)
            write_workflow(
                job_dir,
                f"qwen_cartoon_reference_attempt_{attempt:02d}.json",
                cartoon_workflow,
            )
            cartoon_entry = queue_and_wait(
                cartoon_workflow,
                f"{job_id} cartoon reference attempt {attempt}",
                timeout=7200,
                job_id=job_id,
            )
            cartoon_source = first_output(
                cartoon_entry,
                {".png", ".jpg", ".jpeg", ".webp"},
                "Qwen cartoon reference",
            )
            attempt_reference = job_dir / f"cartoon_reference_attempt_{attempt:02d}.png"
            with Image.open(cartoon_source) as generated:
                cartoon_size = generated.size
                generated.convert("RGB").save(
                    attempt_reference,
                    format="PNG",
                    optimize=True,
                )
            raw_candidate_identity_palette = (
                identity_palette_signature(attempt_reference)
                if source_identity_palette is not None
                else None
            )
            raw_candidate_upper_body_identity_palette = (
                identity_palette_signature(
                    attempt_reference,
                    exclude_bottom_fraction=(
                        BOWL_IDENTITY_PALETTE_UPPER_BODY_EXCLUSION
                    ),
                )
                if canonical_source_upper_body_palette is not None
                else None
            )
            attempt_cutout = (
                job_dir / f"cartoon_reference_cutout_attempt_{attempt:02d}.png"
            )
            try:
                chroma_metrics = evaluate_chroma_cutout(
                    attempt_reference,
                    attempt_cutout,
                    validate_canonical_interior=(
                        GENERATION_VISUAL_AUDITS_BLOCKING
                        or canonical_identity_palette_gate_required
                    ),
                )
            except Exception as exc:  # noqa: BLE001 - image requests may return the saved raw image
                if not image_request:
                    raise
                chroma_metrics = {
                    "acceptable": False,
                    "issues": ["background_removal_exception"],
                    "error": f"{type(exc).__name__}: {str(exc)[:1000]}",
                }
            if chroma_metrics.get("acceptable") and attempt_cutout.is_file():
                try:
                    cleanup = neutralize_uploaded_identity_chroma_spill(
                        attempt_cutout,
                        source_identity_palette,
                    )
                    chroma_metrics = merge_identity_chroma_spill_cleanup(
                        chroma_metrics,
                        cleanup,
                        blocking=bool(
                            GENERATION_VISUAL_AUDITS_BLOCKING
                            or canonical_identity_palette_gate_required
                        ),
                    )
                except Exception as exc:  # noqa: BLE001 - retry candidate fail closed
                    chroma_metrics = {
                        **chroma_metrics,
                        "acceptable": False,
                        "issues": list(
                            dict.fromkeys(
                                [
                                    *[
                                        str(issue)
                                        for issue in chroma_metrics.get(
                                            "issues",
                                            [],
                                        )
                                    ],
                                    "identity_chroma_spill_cleanup_failed",
                                ]
                            )
                        ),
                        "identity_chroma_spill_cleanup": {
                            "policy": UPLOADED_IDENTITY_CHROMA_SPILL_POLICY,
                            "applied": False,
                            "error": f"{type(exc).__name__}: {str(exc)[:1000]}",
                        },
                    }
                    if not GENERATION_VISUAL_AUDITS_BLOCKING:
                        chroma_metrics["warnings"] = list(
                            chroma_metrics.get("issues", [])
                        )
                        chroma_metrics["issues"] = []
                        chroma_metrics["acceptable"] = True
            palette_input_path = (
                attempt_cutout if attempt_cutout.is_file() else attempt_reference
            )
            candidate_identity_palette = (
                identity_palette_signature(palette_input_path)
                if source_identity_palette is not None
                else None
            )
            primary_palette_quality = (
                identity_palette_quality(
                    source_identity_palette,
                    candidate_identity_palette,
                )
                if (
                    source_identity_palette is not None
                    and candidate_identity_palette is not None
                )
                else None
            )
            candidate_upper_body_identity_palette = (
                identity_palette_signature(
                    palette_input_path,
                    exclude_bottom_fraction=(
                        BOWL_IDENTITY_PALETTE_UPPER_BODY_EXCLUSION
                    ),
                )
                if canonical_source_upper_body_palette is not None
                else None
            )
            upper_body_palette_quality = (
                identity_palette_quality(
                    canonical_source_upper_body_palette,
                    candidate_upper_body_identity_palette,
                )
                if (
                    canonical_source_upper_body_palette is not None
                    and candidate_upper_body_identity_palette is not None
                )
                else None
            )
            palette_quality = canonical_identity_palette_quality(
                primary_palette_quality,
                upper_body_palette_quality,
            )
            cartoon_quality = (
                (
                    pose_cutout_quality(attempt_cutout, "canonical character")
                    if GENERATION_VISUAL_AUDITS_BLOCKING
                    else skipped_visual_audit("canonical_subject_quality")
                )
                if chroma_metrics["acceptable"]
                else None
            )
            canonical_quality_blocking_issues = (
                semantic_quality_blocking_issues(cartoon_quality)
                if cartoon_quality is not None
                else []
            )
            if cartoon_quality is not None:
                cartoon_quality = {
                    **cartoon_quality,
                    "blocking_issues": canonical_quality_blocking_issues,
                    "acceptable_after_normalization": not (
                        canonical_quality_blocking_issues
                    ),
                    "framing_recovery_required": bool(
                        cartoon_quality.get("issues")
                        and not canonical_quality_blocking_issues
                    ),
                }
            attempt_acceptable = bool(
                chroma_metrics["acceptable"]
                and (
                    not canonical_identity_palette_gate_required
                    or (
                        isinstance(palette_quality, dict)
                        and palette_quality.get("acceptable") is True
                    )
                )
                and (
                    not GENERATION_VISUAL_AUDITS_BLOCKING
                    or (
                        cartoon_quality
                        and cartoon_quality["acceptable_after_normalization"]
                        and (
                            palette_quality is None
                            or palette_quality["acceptable"]
                        )
                    )
                )
            )
            record = {
                "attempt": attempt,
                "profile": profile["name"],
                "seed": attempt_seed,
                "keyframe": attempt_reference.name,
                "cutout": attempt_cutout.name if chroma_metrics["acceptable"] else None,
                "settings": {
                    "steps": profile["steps"],
                    "cfg": profile["cfg"],
                    "denoise": profile["denoise"],
                    "user_lora": profile["user_lora"],
                    "user_lora_strength": profile["user_lora_strength"],
                    "user_loras": profile["user_loras"],
                    "user_lora_strength_scale": round(
                        canonical_lora_scale
                        * float(profile["user_lora_strength_scale"]),
                        6,
                    ),
                    "user_lora_mode": profile["user_lora_mode"],
                    "use_user_prompt": profile["use_user_prompt"],
                },
                "chroma": chroma_metrics,
                "quality": cartoon_quality,
                "identity_palette": palette_quality,
                "identity_palette_input": palette_input_path.name,
                "raw_candidate_identity_palette": (raw_candidate_identity_palette),
                "raw_candidate_upper_body_identity_palette": (
                    raw_candidate_upper_body_identity_palette
                ),
                "candidate_identity_palette": candidate_identity_palette,
                "candidate_upper_body_identity_palette": (
                    candidate_upper_body_identity_palette
                ),
            }
            cartoon_attempts.append(record)
            if attempt_acceptable:
                selected_cartoon = {
                    **record,
                    "raw_path": attempt_reference,
                    "cutout_path": attempt_cutout,
                    "workflow": cartoon_workflow,
                    "size": cartoon_size,
                    "profile_index": attempt - 1,
                    "transparent": True,
                    "background_removal_failed": False,
                }
                if attempt > 1:
                    append_recovery_event(
                        job_id,
                        {
                            "stage": "cartoon_reference",
                            "attempt": attempt,
                            "reason": "unclean_chroma_background",
                            "strategy": "safe_qwen_retry",
                            "profile": profile["name"],
                            "result": "accepted",
                            "chroma": chroma_metrics,
                            "quality": cartoon_quality,
                            "identity_palette": palette_quality,
                        },
                    )
                break
            if palette_quality is not None and not palette_quality["acceptable"]:
                rejection_reason = "canonical_identity_palette_drift"
            elif not chroma_metrics["acceptable"]:
                rejection_reason = "unclean_chroma_background"
            else:
                rejection_reason = "canonical_subject_quality_invalid"
            if (
                image_request
                and not chroma_metrics["acceptable"]
                and (palette_quality is None or palette_quality["acceptable"])
            ):
                background_removal_error = "Generated image did not contain a clean removable green background."
                record.update(
                    {
                        "profile": f"{profile['name']}_raw_image",
                        "cutout": None,
                        "transparent": False,
                        "background_removal_failed": True,
                        "background_removal_error": background_removal_error,
                    }
                )
                selected_cartoon = {
                    **record,
                    "raw_path": attempt_reference,
                    "cutout_path": None,
                    "workflow": cartoon_workflow,
                    "size": cartoon_size,
                    "profile_index": attempt - 1,
                }
                append_background_removal_failure(
                    job_id,
                    {
                        "stage": "cartoon_reference",
                        "attempt": attempt,
                        "reason": rejection_reason,
                        "strategy": "return_raw_image",
                        "profile": profile["name"],
                        "result": "returned_opaque_original",
                        "chroma": chroma_metrics,
                        "quality": cartoon_quality,
                        "identity_palette": palette_quality,
                    },
                )
                LOGGER.warning(
                    "%s cartoon reference could not be reliably cut out; returning the raw image "
                    "for image request without changing generation settings",
                    job_id,
                )
                break
            append_recovery_event(
                job_id,
                {
                    "stage": "cartoon_reference",
                    "attempt": attempt,
                    "reason": rejection_reason,
                    "strategy": "safe_qwen_retry",
                    "profile": profile["name"],
                    "result": "rejected",
                    "chroma": chroma_metrics,
                    "quality": cartoon_quality,
                    "identity_palette": palette_quality,
                },
            )
            LOGGER.warning(
                "%s cartoon reference attempt %s (%s) failed checks: chroma=%s quality=%s identity_palette=%s",
                job_id,
                attempt,
                profile["name"],
                chroma_metrics["issues"],
                cartoon_quality["issues"] if cartoon_quality else None,
                palette_quality["issues"] if palette_quality else None,
            )
        if selected_cartoon is None:
            fallback_record = select_canonical_fallback_attempt(
                cartoon_attempts,
                identity_palette_required=bool(
                    canonical_identity_palette_gate_required
                    and source_identity_palette is not None
                    and source_identity_palette.get("available")
                ),
            )
            identity_safe_generated_attempts = [
                record
                for record in cartoon_attempts
                if isinstance(record.get("identity_palette"), dict)
                and record["identity_palette"].get("acceptable") is True
                and not bool(
                    (
                        record.get("chroma", {}).get(
                            "identity_chroma_spill_cleanup",
                            {},
                        )
                        if isinstance(record.get("chroma"), dict)
                        else {}
                    ).get("regeneration_required")
                )
            ]
            fallback_uses_uploaded_identity = bool(
                canonical_identity_palette_gate_required
                and not identity_safe_generated_attempts
                and not GENERATION_COMPLETION_FIRST_VISUAL_FALLBACK_ENABLED
            )
            fallback_source = (
                Path(identity_source_path)
                if fallback_uses_uploaded_identity
                else job_dir / fallback_record["keyframe"]
            )
            fallback_green = job_dir / "cartoon_reference_semantic_green.png"
            fallback_cutout = job_dir / "cartoon_reference_semantic_cutout_final.png"
            fallback_semantic = job_dir / "cartoon_reference_semantic_cutout_raw.png"
            generated_semantic_fallback_error: str | None = None
            completion_first_canonical_fallback = False
            try:
                chroma_metrics = semantic_background_recovery(
                    fallback_source,
                    fallback_green,
                    fallback_cutout,
                    fallback_semantic,
                    validate_canonical_interior=True,
                )
            except Exception as exc:  # noqa: BLE001 - image mode must preserve a usable raw result
                if not image_request:
                    if (
                        GENERATION_COMPLETION_FIRST_VISUAL_FALLBACK_ENABLED
                        and not fallback_uses_uploaded_identity
                    ):
                        generated_semantic_fallback_error = (
                            f"{type(exc).__name__}: {str(exc)[:1000]}"
                        )
                        chroma_metrics = semantic_background_recovery(
                            fallback_source,
                            fallback_green,
                            fallback_cutout,
                            fallback_semantic,
                            validate_canonical_interior=False,
                            enforce_visual_quality=False,
                        )
                        completion_first_canonical_fallback = True
                        append_recovery_event(
                            job_id,
                            {
                                "stage": "cartoon_reference_semantic_recovery",
                                "attempt": fallback_record["attempt"],
                                "reason": "strict_visual_canonical_matte_invalid",
                                "strategy": (
                                    "use_best_generated_semantic_matte_without_"
                                    "visual_block"
                                ),
                                "profile": fallback_record["profile"],
                                "source": fallback_source.name,
                                "result": "accepted_completion_fallback",
                                "strict_error": generated_semantic_fallback_error,
                                "policy": (
                                    GENERATION_COMPLETION_FIRST_VISUAL_FALLBACK_POLICY
                                ),
                            },
                        )
                        update_job(
                            job_id,
                            canonical_completion_fallback_used=True,
                            canonical_completion_fallback_policy=(
                                GENERATION_COMPLETION_FIRST_VISUAL_FALLBACK_POLICY
                            ),
                            canonical_completion_fallback_strict_error=(
                                generated_semantic_fallback_error
                            ),
                            canonical_completion_fallback_source=(
                                fallback_source.name
                            ),
                            canonical_completion_fallback_attempt=(
                                fallback_record["attempt"]
                            ),
                        )
                    elif fallback_uses_uploaded_identity:
                        raise
                    else:
                        generated_semantic_fallback_error = (
                            f"{type(exc).__name__}: {str(exc)[:1000]}"
                        )
                        append_recovery_event(
                            job_id,
                            {
                                "stage": "cartoon_reference_semantic_recovery",
                                "attempt": fallback_record["attempt"],
                                "reason": (
                                    "canonical_generated_semantic_matte_invalid"
                                ),
                                "strategy": (
                                    "use_uploaded_identity_after_safe_matte"
                                ),
                                "profile": fallback_record["profile"],
                                "source": fallback_source.name,
                                "result": "rejected",
                                "error": generated_semantic_fallback_error,
                            },
                        )
                        fallback_uses_uploaded_identity = True
                        fallback_source = Path(identity_source_path)
                        fallback_green = (
                            job_dir
                            / "cartoon_reference_uploaded_identity_matte_green.png"
                        )
                        fallback_cutout = (
                            job_dir
                            / "cartoon_reference_uploaded_identity_matte_cutout.png"
                        )
                        fallback_semantic = (
                            job_dir
                            / "cartoon_reference_uploaded_identity_matte_semantic.png"
                        )
                        try:
                            chroma_metrics = semantic_background_recovery(
                                fallback_source,
                                fallback_green,
                                fallback_cutout,
                                fallback_semantic,
                                validate_canonical_interior=True,
                            )
                        except Exception as uploaded_exc:
                            uploaded_error = (
                                f"{type(uploaded_exc).__name__}: "
                                f"{str(uploaded_exc)[:1000]}"
                            )
                            append_recovery_event(
                                job_id,
                                {
                                    "stage": (
                                        "cartoon_reference_semantic_recovery"
                                    ),
                                    "attempt": len(cartoon_attempts) + 1,
                                    "reason": (
                                        "canonical_uploaded_identity_matte_invalid"
                                    ),
                                    "strategy": (
                                        "use_uploaded_identity_after_safe_matte"
                                    ),
                                    "source": fallback_source.name,
                                    "result": "rejected",
                                    "error": uploaded_error,
                                },
                            )
                            raise RuntimeError(
                                "Generated canonical semantic matte failed and the "
                                "uploaded identity safe-matte recovery also failed: "
                                f"generated={generated_semantic_fallback_error}; "
                                f"uploaded={uploaded_error}"
                            ) from uploaded_exc
                else:
                    best_identity_record = fallback_record
                    best_identity_source = job_dir / best_identity_record["keyframe"]
                    with Image.open(best_identity_source) as raw_image:
                        cartoon_size = raw_image.size
                    background_removal_error = (
                        "Background removal could not preserve a complete subject; "
                        "the original generated image was returned."
                    )
                    selected_cartoon = {
                        **best_identity_record,
                        "profile": f"{best_identity_record['profile']}_raw_image",
                        "keyframe": best_identity_source.name,
                        "cutout": None,
                        "raw_path": best_identity_source,
                        "cutout_path": None,
                        "workflow": cartoon_workflows[
                            int(best_identity_record["attempt"]) - 1
                        ],
                        "size": cartoon_size,
                        "profile_index": int(best_identity_record["attempt"]) - 1,
                        "transparent": False,
                        "background_removal_failed": True,
                        "background_removal_error": background_removal_error,
                    }
                    append_background_removal_failure(
                        job_id,
                        {
                            "stage": "cartoon_reference_semantic_recovery",
                            "attempt": best_identity_record["attempt"],
                            "reason": "semantic_background_recovery_failed",
                            "strategy": "return_best_identity_raw_image",
                            "profile": best_identity_record["profile"],
                            "result": "returned_opaque_original",
                            "identity_palette": best_identity_record.get(
                                "identity_palette"
                            ),
                            "error": f"{type(exc).__name__}: {str(exc)[:1000]}",
                        },
                    )
                    LOGGER.warning(
                        "%s cartoon reference semantic recovery failed; returning "
                        "the best identity-preserving raw image for image mode",
                        job_id,
                    )
            if selected_cartoon is None:
                fallback_cleanup = (
                    neutralize_uploaded_identity_chroma_spill(
                        fallback_cutout,
                        source_identity_palette,
                    )
                    if (
                        GENERATION_VISUAL_AUDITS_BLOCKING
                        or canonical_identity_palette_gate_required
                    )
                    else {
                        "policy": UPLOADED_IDENTITY_CHROMA_SPILL_POLICY,
                        "applied": False,
                        "status": "skipped_advisory",
                    }
                )
                chroma_metrics = merge_identity_chroma_spill_cleanup(
                    chroma_metrics,
                    fallback_cleanup,
                    blocking=bool(
                        GENERATION_VISUAL_AUDITS_BLOCKING
                        or canonical_identity_palette_gate_required
                    ),
                )
                fallback_candidate_palette = (
                    identity_palette_signature(fallback_cutout)
                    if source_identity_palette is not None
                    else None
                )
                fallback_upper_body_candidate_palette = (
                    identity_palette_signature(
                        fallback_cutout,
                        exclude_bottom_fraction=(
                            BOWL_IDENTITY_PALETTE_UPPER_BODY_EXCLUSION
                        ),
                    )
                    if canonical_source_upper_body_palette is not None
                    else None
                )
                fallback_primary_quality = (
                    identity_palette_quality(
                        source_identity_palette,
                        fallback_candidate_palette,
                    )
                    if (
                        source_identity_palette is not None
                        and fallback_candidate_palette is not None
                    )
                    else None
                )
                fallback_upper_quality = (
                    identity_palette_quality(
                        canonical_source_upper_body_palette,
                        fallback_upper_body_candidate_palette,
                    )
                    if (
                        canonical_source_upper_body_palette is not None
                        and fallback_upper_body_candidate_palette is not None
                    )
                    else None
                )
                fallback_palette_quality = canonical_identity_palette_quality(
                    fallback_primary_quality,
                    fallback_upper_quality,
                )
                warm_palette_recovery_used = False
                warm_palette_recovery = (
                    attempt_warm_canonical_palette_recovery(
                        fallback_cutout,
                        Path(identity_source_path),
                        job_dir
                        / "cartoon_reference_warm_palette_recovery_cutout.png",
                        job_dir
                        / "cartoon_reference_warm_palette_recovery_green.png",
                        source_palette=source_identity_palette,
                        source_upper_body_palette=(
                            canonical_source_upper_body_palette
                        ),
                        current_quality=fallback_palette_quality,
                        upper_body_exclusion=(
                            BOWL_IDENTITY_PALETTE_UPPER_BODY_EXCLUSION
                        ),
                    )
                )
                if warm_palette_recovery.get("accepted") is True:
                    fallback_cutout = Path(
                        warm_palette_recovery["cutout_path"]
                    )
                    fallback_green = Path(warm_palette_recovery["green_path"])
                    fallback_candidate_palette = warm_palette_recovery[
                        "candidate_palette"
                    ]
                    fallback_upper_body_candidate_palette = (
                        warm_palette_recovery[
                            "upper_body_candidate_palette"
                        ]
                    )
                    fallback_palette_quality = warm_palette_recovery[
                        "post_recovery_quality"
                    ]
                    chroma_metrics = {
                        **chroma_metrics,
                        "warm_canonical_palette_recovery": (
                            warm_palette_recovery
                        ),
                    }
                    warm_palette_recovery_used = True
                    append_recovery_event(
                        job_id,
                        {
                            "stage": "cartoon_reference",
                            "attempt": fallback_record["attempt"],
                            "reason": "warm_source_recolored_achromatic",
                            "strategy": WARM_CANONICAL_PALETTE_RECOVERY_POLICY,
                            "result": (
                                "accepted_palette_pass"
                                if fallback_palette_quality.get("acceptable")
                                is True
                                else "accepted_lower_risk_best_effort"
                            ),
                            "identity_palette": fallback_palette_quality,
                        },
                    )
                    update_job(
                        job_id,
                        canonical_warm_palette_recovery=(
                            warm_palette_recovery
                        ),
                    )
                elif warm_palette_recovery.get("attempted") is True:
                    update_job(
                        job_id,
                        canonical_warm_palette_recovery=(
                            warm_palette_recovery
                        ),
                    )
                fallback_profile = (
                    "uploaded_identity_semantic_fallback"
                    if fallback_uses_uploaded_identity
                    else (
                        "generated_warm_palette_completion_fallback"
                        if warm_palette_recovery_used
                        else "generated_semantic_completion_fallback"
                        if completion_first_canonical_fallback
                        else "rembg_u2net_fallback"
                    )
                )
                fallback_recovery_reason = (
                    "strict_visual_canonical_matte_invalid"
                    if completion_first_canonical_fallback
                    else (
                        "canonical_generated_semantic_matte_invalid"
                        if generated_semantic_fallback_error is not None
                        else (
                            "canonical_identity_palette_drift_exhausted"
                            if fallback_uses_uploaded_identity
                            else "qwen_chroma_retries_exhausted"
                        )
                    )
                )
                fallback_recovery_strategy = (
                    "use_best_generated_semantic_matte_without_visual_block"
                    if completion_first_canonical_fallback
                    else (
                        "use_uploaded_identity_after_generated_matte_failure"
                        if generated_semantic_fallback_error is not None
                        else (
                            "use_original_uploaded_identity_after_safe_matte"
                            if fallback_uses_uploaded_identity
                            else "rembg_u2net_fallback"
                        )
                    )
                )
                fallback_raw_candidate_identity_palette = (
                    source_identity_palette
                    if fallback_uses_uploaded_identity
                    else fallback_record.get("raw_candidate_identity_palette")
                )
                fallback_raw_candidate_upper_body_identity_palette = (
                    canonical_source_upper_body_palette
                    if fallback_uses_uploaded_identity
                    else fallback_record.get(
                        "raw_candidate_upper_body_identity_palette"
                    )
                )
                if fallback_uses_uploaded_identity:
                    uploaded_identity_fallback_update: dict[str, Any] = {
                        "canonical_uploaded_identity_fallback_used": True,
                        "canonical_uploaded_identity_fallback_source": str(
                            fallback_source
                        ),
                    }
                    if generated_semantic_fallback_error is not None:
                        uploaded_identity_fallback_update[
                            "canonical_generated_matte_failure"
                        ] = {
                            "source": fallback_record["keyframe"],
                            "error": generated_semantic_fallback_error,
                        }
                    else:
                        uploaded_identity_fallback_update[
                            "canonical_generated_palette_failure"
                        ] = {
                            "attempts": [
                                record.get("identity_palette")
                                for record in cartoon_attempts
                            ],
                        }
                    update_job(
                        job_id,
                        **uploaded_identity_fallback_update,
                    )
                if (
                    canonical_identity_palette_gate_required
                    and not GENERATION_COMPLETION_FIRST_VISUAL_FALLBACK_ENABLED
                    and (
                        not isinstance(fallback_palette_quality, dict)
                        or fallback_palette_quality.get("acceptable") is not True
                    )
                ):
                    generated_fallback_palette_quality = fallback_palette_quality
                    uploaded_fallback_sources = list(
                        dict.fromkeys(
                            [
                                Path(identity_source_path),
                                Path(upload_integrity["input_path"])
                                if upload_integrity is not None
                                else Path(identity_source_path),
                            ]
                        )
                    )
                    uploaded_fallback_error: Exception | None = None
                    uploaded_fallback_accepted = False
                    for uploaded_fallback_index, uploaded_fallback_source in enumerate(
                        uploaded_fallback_sources,
                        start=1,
                    ):
                        uploaded_fallback_green = (
                            job_dir
                            / "cartoon_reference_uploaded_identity_"
                            f"fallback_{uploaded_fallback_index:02d}_green.png"
                        )
                        uploaded_fallback_cutout = (
                            job_dir
                            / "cartoon_reference_uploaded_identity_"
                            f"fallback_{uploaded_fallback_index:02d}_cutout.png"
                        )
                        uploaded_fallback_semantic = (
                            job_dir
                            / "cartoon_reference_uploaded_identity_"
                            f"fallback_{uploaded_fallback_index:02d}_semantic.png"
                        )
                        try:
                            uploaded_chroma_metrics = semantic_background_recovery(
                                uploaded_fallback_source,
                                uploaded_fallback_green,
                                uploaded_fallback_cutout,
                                uploaded_fallback_semantic,
                                validate_canonical_interior=True,
                            )
                            uploaded_cleanup = (
                                neutralize_uploaded_identity_chroma_spill(
                                    uploaded_fallback_cutout,
                                    source_identity_palette,
                                )
                            )
                            uploaded_chroma_metrics = (
                                merge_identity_chroma_spill_cleanup(
                                    uploaded_chroma_metrics,
                                    uploaded_cleanup,
                                    blocking=True,
                                )
                            )
                            if uploaded_chroma_metrics.get("acceptable") is not True:
                                raise RuntimeError(
                                    "Uploaded identity fallback exceeded the safe "
                                    "chroma reconstruction scope."
                                )
                            uploaded_candidate_palette = identity_palette_signature(
                                uploaded_fallback_cutout
                            )
                            uploaded_upper_candidate_palette = (
                                identity_palette_signature(
                                    uploaded_fallback_cutout,
                                    exclude_bottom_fraction=(
                                        BOWL_IDENTITY_PALETTE_UPPER_BODY_EXCLUSION
                                    ),
                                )
                            )
                            uploaded_palette_quality = (
                                canonical_identity_palette_quality(
                                    identity_palette_quality(
                                        source_identity_palette,
                                        uploaded_candidate_palette,
                                    ),
                                    identity_palette_quality(
                                        canonical_source_upper_body_palette,
                                        uploaded_upper_candidate_palette,
                                    ),
                                )
                            )
                            if (
                                not isinstance(uploaded_palette_quality, dict)
                                or uploaded_palette_quality.get("acceptable")
                                is not True
                            ):
                                raise RuntimeError(
                                    "Uploaded identity semantic fallback did not "
                                    "preserve the source palette: "
                                    + ", ".join(
                                        str(issue)
                                        for issue in (
                                            uploaded_palette_quality or {}
                                        ).get("issues", [])
                                    )
                                )
                        except Exception as exc:  # noqa: BLE001 - try normalized source next
                            uploaded_fallback_error = exc
                            append_recovery_event(
                                job_id,
                                {
                                    "stage": "cartoon_reference",
                                    "attempt": len(cartoon_attempts) + 1,
                                    "reason": "canonical_identity_palette_drift",
                                    "strategy": (
                                        "uploaded_identity_semantic_fallback"
                                    ),
                                    "source": uploaded_fallback_source.name,
                                    "result": "rejected",
                                    "error": (
                                        f"{type(exc).__name__}: {str(exc)[:1000]}"
                                    ),
                                },
                            )
                            continue
                        fallback_green = uploaded_fallback_green
                        fallback_cutout = uploaded_fallback_cutout
                        fallback_semantic = uploaded_fallback_semantic
                        chroma_metrics = uploaded_chroma_metrics
                        fallback_candidate_palette = uploaded_candidate_palette
                        fallback_upper_body_candidate_palette = (
                            uploaded_upper_candidate_palette
                        )
                        fallback_palette_quality = uploaded_palette_quality
                        fallback_profile = "uploaded_identity_semantic_fallback"
                        fallback_recovery_reason = (
                            "canonical_identity_palette_drift_exhausted"
                        )
                        fallback_recovery_strategy = (
                            "use_original_uploaded_identity_after_safe_matte"
                        )
                        fallback_raw_candidate_identity_palette = (
                            source_identity_palette
                        )
                        fallback_raw_candidate_upper_body_identity_palette = (
                            canonical_source_upper_body_palette
                        )
                        uploaded_fallback_accepted = True
                        update_job(
                            job_id,
                            canonical_uploaded_identity_fallback_used=True,
                            canonical_generated_palette_failure=(
                                generated_fallback_palette_quality
                            ),
                            canonical_uploaded_identity_fallback_source=(
                                str(uploaded_fallback_source)
                            ),
                        )
                        break
                    if not uploaded_fallback_accepted:
                        raise RuntimeError(
                            "Canonical identity palette drift remained after all "
                            "Qwen retries and the original-upload safe fallback "
                            "could not be prepared: "
                            + (
                                f"{type(uploaded_fallback_error).__name__}: "
                                f"{str(uploaded_fallback_error)[:1000]}"
                                if uploaded_fallback_error is not None
                                else "no fallback source"
                            )
                        )
                if (
                    (
                        GENERATION_VISUAL_AUDITS_BLOCKING
                        or canonical_identity_palette_gate_required
                    )
                    and not GENERATION_COMPLETION_FIRST_VISUAL_FALLBACK_ENABLED
                    and
                    fallback_palette_quality is not None
                    and fallback_palette_quality.get("acceptable") is not True
                ):
                    raise RuntimeError(
                        "Canonical identity palette remained invalid after "
                        "semantic matte and identity-aware chroma cleanup: "
                        + ", ".join(
                            str(issue)
                            for issue in fallback_palette_quality.get("issues", [])
                        )
                    )
                if (
                    fallback_palette_quality is not None
                    and fallback_palette_quality.get("acceptable") is not True
                ):
                    LOGGER.warning(
                        "%s canonical identity palette audit is advisory after "
                        "semantic recovery: %s",
                        job_id,
                        fallback_palette_quality.get("issues", []),
                    )
                with Image.open(fallback_green) as recovered_image:
                    cartoon_size = recovered_image.size
                selected_cartoon = {
                    "attempt": len(cartoon_attempts) + 1,
                    "profile": fallback_profile,
                    "seed": fallback_record["seed"],
                    "keyframe": fallback_green.name,
                    "cutout": fallback_cutout.name,
                    "settings": fallback_record["settings"],
                    "chroma": chroma_metrics,
                    "identity_palette": fallback_palette_quality,
                    "identity_palette_input": fallback_cutout.name,
                    "raw_candidate_identity_palette": (
                        fallback_raw_candidate_identity_palette
                    ),
                    "raw_candidate_upper_body_identity_palette": (
                        fallback_raw_candidate_upper_body_identity_palette
                    ),
                    "candidate_identity_palette": fallback_candidate_palette,
                    "candidate_upper_body_identity_palette": (
                        fallback_upper_body_candidate_palette
                    ),
                    "raw_path": fallback_green,
                    "cutout_path": fallback_cutout,
                    "workflow": cartoon_workflows[int(fallback_record["attempt"]) - 1],
                    "size": cartoon_size,
                    "profile_index": int(fallback_record["attempt"]) - 1,
                    "transparent": True,
                    "background_removal_failed": False,
                    "completion_first_fallback": (
                        completion_first_canonical_fallback
                        or bool(
                            isinstance(fallback_palette_quality, dict)
                            and fallback_palette_quality.get("acceptable")
                            is not True
                            and GENERATION_COMPLETION_FIRST_VISUAL_FALLBACK_ENABLED
                        )
                    ),
                }
                cartoon_attempts.append(
                    {
                        "attempt": selected_cartoon["attempt"],
                        "profile": selected_cartoon["profile"],
                        "seed": selected_cartoon["seed"],
                        "keyframe": selected_cartoon["keyframe"],
                        "cutout": selected_cartoon["cutout"],
                        "settings": selected_cartoon["settings"],
                        "chroma": selected_cartoon["chroma"],
                        "identity_palette": selected_cartoon.get("identity_palette"),
                        "identity_palette_input": selected_cartoon.get(
                            "identity_palette_input"
                        ),
                        "raw_candidate_identity_palette": selected_cartoon.get(
                            "raw_candidate_identity_palette"
                        ),
                        "raw_candidate_upper_body_identity_palette": (
                            selected_cartoon.get(
                                "raw_candidate_upper_body_identity_palette"
                            )
                        ),
                        "candidate_identity_palette": selected_cartoon.get(
                            "candidate_identity_palette"
                        ),
                        "candidate_upper_body_identity_palette": selected_cartoon.get(
                            "candidate_upper_body_identity_palette"
                        ),
                    }
                )
                append_recovery_event(
                    job_id,
                    {
                        "stage": "cartoon_reference",
                        "attempt": selected_cartoon["attempt"],
                        "reason": fallback_recovery_reason,
                        "strategy": fallback_recovery_strategy,
                        "profile": fallback_profile,
                        "result": "accepted",
                        "chroma": chroma_metrics,
                        "identity_palette": selected_cartoon.get("identity_palette"),
                    },
                )
        cartoon_reference = job_dir / "cartoon_reference.png"
        cartoon_transparent = job_dir / "cartoon_reference_transparent.png"
        cartoon_is_transparent = bool(selected_cartoon.get("transparent", True))
        if cartoon_is_transparent:
            try:
                prepare_landscape_keyframe(
                    selected_cartoon["cutout_path"],
                    cartoon_reference,
                    (width, height),
                    background_rgb=(0, 255, 0),
                )
                normalized_chroma = evaluate_chroma_cutout(
                    cartoon_reference,
                    cartoon_transparent,
                    validate_canonical_interior=(
                        GENERATION_VISUAL_AUDITS_BLOCKING
                        or canonical_identity_palette_gate_required
                    )
                    and not bool(
                        selected_cartoon.get("completion_first_fallback")
                    ),
                )
                normalized_chroma["strict_acceptable"] = bool(
                    normalized_chroma.get("acceptable")
                )
                normalized_issues = set(
                    normalized_chroma.get("issues") or []
                )
                if normalized_issues == {
                    "canonical_subject_interior_chroma_hole"
                }:
                    try:
                        normalized_interior_recovery = (
                            repair_canonical_cutout_interior_holes(
                                cartoon_transparent
                            )
                        )
                    except Exception as recovery_exc:  # noqa: BLE001 - visual recovery cannot end a valid task
                        normalized_interior_recovery = {
                            "policy": (
                                CANONICAL_NORMALIZED_INTERIOR_RECOVERY_POLICY
                            ),
                            "attempted": True,
                            "applied": False,
                            "acceptable": False,
                            "completion_blocked": False,
                            "error": (
                                f"{type(recovery_exc).__name__}: "
                                f"{str(recovery_exc)[:1000]}"
                            ),
                        }
                    normalized_chroma[
                        "canonical_normalized_interior_recovery"
                    ] = normalized_interior_recovery
                    if normalized_interior_recovery.get("applied"):
                        composite_rgba_reference_on_background(
                            cartoon_transparent,
                            cartoon_reference,
                            (0, 255, 0),
                        )
                    if normalized_interior_recovery.get("acceptable") is True:
                        normalized_chroma.update(
                            {
                                "acceptable": True,
                                "issues": [],
                                "canonical_interior_quality": (
                                    normalized_interior_recovery.get(
                                        "post_repair_quality"
                                    )
                                ),
                                "normalization_recovered": True,
                                "completion_blocked": False,
                            }
                        )
                        append_recovery_event(
                            job_id,
                            {
                                "stage": "cartoon_reference_finalization",
                                "attempt": selected_cartoon["attempt"],
                                "reason": (
                                    "scale_dependent_normalized_canonical_"
                                    "interior_hole"
                                ),
                                "strategy": (
                                    CANONICAL_NORMALIZED_INTERIOR_RECOVERY_POLICY
                                ),
                                "result": "accepted_after_candidate_only_repair",
                                "recovery": normalized_interior_recovery,
                            },
                        )
                        update_job(
                            job_id,
                            canonical_normalized_interior_recovery=(
                                normalized_interior_recovery
                            ),
                        )
                    elif GENERATION_COMPLETION_FIRST_VISUAL_FALLBACK_ENABLED:
                        runtime_issues = _validated_first_frame_image_issues(
                            cartoon_transparent,
                            expected_sha256=file_sha256(cartoon_transparent),
                        )
                        normalized_chroma[
                            "runtime_integrity_issues"
                        ] = runtime_issues
                        if not runtime_issues:
                            normalized_chroma.update(
                                {
                                    "acceptable": True,
                                    "strict_acceptable": False,
                                    "completion_first_visual_advisory": True,
                                    "completion_blocked": False,
                                }
                            )
                            selected_cartoon["completion_first_fallback"] = True
                            append_recovery_event(
                                job_id,
                                {
                                    "stage": "cartoon_reference_finalization",
                                    "attempt": selected_cartoon["attempt"],
                                    "reason": (
                                        "normalized_canonical_interior_visual_"
                                        "repair_unavailable"
                                    ),
                                    "strategy": (
                                        GENERATION_COMPLETION_FIRST_VISUAL_FALLBACK_POLICY
                                    ),
                                    "result": (
                                        "accepted_runtime_integrity_visual_advisory"
                                    ),
                                    "recovery": normalized_interior_recovery,
                                },
                            )
                            update_job(
                                job_id,
                                canonical_completion_fallback_used=True,
                                canonical_completion_fallback_policy=(
                                    GENERATION_COMPLETION_FIRST_VISUAL_FALLBACK_POLICY
                                ),
                                canonical_normalized_interior_recovery=(
                                    normalized_interior_recovery
                                ),
                            )
                if not normalized_chroma["acceptable"]:
                    raise RuntimeError(
                        "Could not normalize the recovered cartoon reference to pure green."
                    )
                cartoon_output = cartoon_transparent
            except Exception as exc:  # noqa: BLE001 - preserve the generated image for image mode
                if not image_request:
                    raise
                cartoon_is_transparent = False
                selected_cartoon.update(
                    {
                        "profile": f"{selected_cartoon['profile']}_raw_image",
                        "transparent": False,
                        "background_removal_failed": True,
                        "background_removal_error": (
                            "Background removal finalization failed; the raw image was returned."
                        ),
                    }
                )
                normalized_chroma = {
                    "acceptable": False,
                    "issues": ["background_removal_finalization_exception"],
                    "error": f"{type(exc).__name__}: {str(exc)[:1000]}",
                }
                shutil.copy2(selected_cartoon["raw_path"], cartoon_reference)
                cartoon_output = cartoon_reference
                append_background_removal_failure(
                    job_id,
                    {
                        "stage": "cartoon_reference_finalization",
                        "attempt": selected_cartoon["attempt"],
                        "reason": "background_removal_finalization_exception",
                        "strategy": "return_raw_image",
                        "profile": selected_cartoon["profile"],
                        "result": "returned_opaque_original",
                        "error": normalized_chroma["error"],
                    },
                )
        else:
            shutil.copy2(selected_cartoon["raw_path"], cartoon_reference)
            normalized_chroma = None
            cartoon_output = cartoon_reference
        canonical_palette_audit = identity_palette_audit(
            canonical_identity_palette_gate_required,
            selected_cartoon.get("identity_palette"),
        )
        write_workflow(
            job_dir,
            "qwen_cartoon_reference.json",
            selected_cartoon["workflow"],
        )
        write_json_atomic(
            job_dir / "workflows" / "uploaded_identity_anchor_quality.json",
            {
                "selected_attempt": selected_cartoon["attempt"],
                "selected_profile": selected_cartoon["profile"],
                "selected_source_chroma": selected_cartoon["chroma"],
                "selected_normalized_chroma": normalized_chroma,
                "source_identity_palette": source_identity_palette,
                "source_identity_upper_body_palette": (
                    source_identity_upper_body_palette
                ),
                "canonical_source_upper_body_palette": (
                    canonical_source_upper_body_palette
                ),
                "canonical_upper_body_palette_reference_policy": (
                    CANONICAL_UPPER_BODY_PALETTE_REFERENCE_POLICY
                ),
                "source_identity_view_palettes": source_identity_view_palettes,
                "source_identity_view_upper_body_palettes": (
                    source_identity_view_upper_body_palettes
                ),
                "source_identity_palette_aggregation": (
                    "equal_view_mean_v1"
                    if len(source_identity_view_palettes) > 1
                    else "single_front_view"
                ),
                "canonical_identity_reference_inputs": (comfy_identity_views),
                "selected_identity_palette": selected_cartoon.get("identity_palette"),
                "selected_identity_palette_input": selected_cartoon.get(
                    "identity_palette_input"
                ),
                "selected_raw_candidate_identity_palette": selected_cartoon.get(
                    "raw_candidate_identity_palette"
                ),
                "selected_raw_candidate_upper_body_identity_palette": (
                    selected_cartoon.get("raw_candidate_upper_body_identity_palette")
                ),
                "selected_candidate_identity_palette": selected_cartoon.get(
                    "candidate_identity_palette"
                ),
                "selected_candidate_upper_body_identity_palette": selected_cartoon.get(
                    "candidate_upper_body_identity_palette"
                ),
                "transparent": cartoon_is_transparent,
                "background_removal_failed": selected_cartoon.get(
                    "background_removal_failed", False
                ),
                "background_removal_error": selected_cartoon.get(
                    "background_removal_error"
                ),
                "attempts": cartoon_attempts,
            },
        )
        update_job(
            job_id,
            recovered=cartoon_is_transparent and selected_cartoon["attempt"] > 1,
            recovery_active=False,
            recovery_attempt=None,
            cartoon_reference_profile=selected_cartoon["profile"],
            effective_canonical_user_loras=selected_cartoon["settings"]["user_loras"],
            canonical_user_lora_strength_scale=selected_cartoon["settings"][
                "user_lora_strength_scale"
            ],
            effective_canonical_settings=selected_cartoon["settings"],
            effective_canonical_identity_palette=selected_cartoon.get(
                "identity_palette"
            ),
            canonical_identity_palette_input=selected_cartoon.get(
                "identity_palette_input"
            ),
            canonical_identity_palette_verdict=canonical_palette_audit[
                "identity_palette_verdict"
            ],
            canonical_completion_fallback_used=bool(
                selected_cartoon.get("completion_first_fallback")
            ),
            canonical_completion_fallback_policy=(
                GENERATION_COMPLETION_FIRST_VISUAL_FALLBACK_POLICY
                if selected_cartoon.get("completion_first_fallback")
                else None
            ),
        )
        append_output(
            job_id,
            {
                "type": "cartoon_reference",
                "filename": cartoon_output.name,
                "width": (
                    TRANSPARENT_GIF_SIZE
                    if cartoon_is_transparent
                    else selected_cartoon["size"][0]
                ),
                "height": (
                    TRANSPARENT_GIF_SIZE
                    if cartoon_is_transparent
                    else selected_cartoon["size"][1]
                ),
                "transparent": cartoon_is_transparent,
                "background_removal_failed": selected_cartoon.get(
                    "background_removal_failed", False
                ),
                "background_removal_error": selected_cartoon.get(
                    "background_removal_error"
                ),
                "requested_user_lora_preserved": bool(
                    user_loras
                    and selected_cartoon["settings"]["user_loras"] == user_loras
                    and selected_cartoon["settings"]["user_lora_mode"] == user_lora_mode
                ),
                **canonical_palette_audit,
                "visual_style": "identity_preserving_cute_3d_natural_cat",
                "source_role": "canonical_character_design",
                "recovery_profile": selected_cartoon["profile"],
            },
        )
        comfy_cartoon_reference = comfy_input_dir / cartoon_reference.name
        shutil.copy2(cartoon_reference, comfy_cartoon_reference)
        canonical_qwen_image1_sha256 = file_sha256(cartoon_reference)
        if file_sha256(comfy_cartoon_reference) != canonical_qwen_image1_sha256:
            raise RuntimeError(
                "Qwen full-body canonical ComfyUI Image 1 copy failed its digest check."
            )
        update_job(
            job_id,
            canonical_qwen_image1_path=str(cartoon_reference),
            canonical_qwen_image1_sha256=canonical_qwen_image1_sha256,
            canonical_qwen_image1_comfy_path=str(comfy_cartoon_reference),
            canonical_qwen_image1_comfy_sha256=canonical_qwen_image1_sha256,
            canonical_qwen_image1_role=("normalized_green_full_body_image1_and_latent"),
            canonical_public_reference_path=str(cartoon_output),
            canonical_public_reference_sha256=file_sha256(cartoon_output),
            canonical_public_reference_role=(
                "normalized_transparent_full_body_reference"
                if cartoon_is_transparent
                else "opaque_full_body_reference"
            ),
        )
        qwen_input_name = (comfy_relative_dir / cartoon_reference.name).as_posix()
        canonical_similarity_reference = cartoon_reference
    elif transparent_gif:
        canonical_similarity_reference = identity_source_path
        source_self_quality = (
            canonical_identity_palette_quality(
                identity_palette_quality(
                    source_identity_palette,
                    source_identity_palette,
                ),
                identity_palette_quality(
                    source_identity_upper_body_palette,
                    source_identity_upper_body_palette,
                ),
            )
            if (
                source_identity_palette is not None
                and source_identity_upper_body_palette is not None
            )
            else None
        )
        canonical_palette_audit = identity_palette_audit(
            identity_palette_guard,
            source_self_quality,
        )
        write_json_atomic(
            job_dir / "workflows" / "qwen_cartoon_reference_quality.json",
            {
                "bypassed": True,
                "reason": "main_case_driver_image1_original_upload_image2",
                "identity_source": str(identity_source_path),
                "identity_source_sha256": file_sha256(identity_source_path),
                "source_identity_palette": source_identity_palette,
                "source_identity_upper_body_palette": (
                    source_identity_upper_body_palette
                ),
                "canonical_source_upper_body_palette": (
                    canonical_source_upper_body_palette
                ),
                "canonical_upper_body_palette_reference_policy": (
                    CANONICAL_UPPER_BODY_PALETTE_REFERENCE_POLICY
                ),
                "identity_palette": source_self_quality,
            },
        )
        update_job(
            job_id,
            recovered=False,
            recovery_active=False,
            recovery_attempt=None,
            generated_canonical_bypassed=True,
            cartoon_reference_profile=None,
            effective_canonical_user_loras=[],
            canonical_user_lora_strength_scale=None,
            effective_canonical_settings={
                "bypassed": True,
                "identity_input": uploaded_qwen_input_name,
            },
            effective_canonical_identity_palette=None,
            canonical_identity_palette_verdict="bypassed",
            uploaded_identity_palette_quality=source_self_quality,
            uploaded_identity_palette_verdict=canonical_palette_audit[
                "identity_palette_verdict"
            ],
        )

    keyframes: list[Path] = []
    keyframe_transparency: list[bool] = []
    accepted_pose_similarity_references: list[dict[str, Any]] = []
    effective_pose_lora_attempts: list[dict[str, Any]] = []
    grooming_pose_gate_attempt_audits: list[dict[str, Any]] = []
    grooming_pose_gate_selected_metrics: dict[str, dict[str, Any]] = {}
    grooming_advisory_minimum_gate_attempts: list[dict[str, Any]] = []
    grooming_advisory_minimum_gate_selected: dict[str, dict[str, Any]] = {}
    fur_surface_gate_attempt_audits: list[dict[str, Any]] = []
    fur_surface_gate_selected_metrics: dict[str, dict[str, Any]] = {}
    fur_surface_gate_status_by_action: dict[str, str] = {}
    pose_effective_lora_policies: dict[str, str] = {}
    fur_surface_gate_short_circuit: dict[str, Any] | None = None
    reference_generation_degraded_actions: dict[str, dict[str, Any]] = {}
    for index, pose in enumerate(poses, start=1):
        raise_if_cancelled(job_id)
        update_job(
            job_id,
            stage="generating_cartoon_keyframes"
            if transparent_gif
            else "generating_keyframes",
            progress=round(
                (18 if transparent_gif else 5) + 17 * (index - 1) / len(poses), 1
            ),
            current_pose_index=index,
            current_pose=pose,
        )
        pose_reference_input_name = None
        configured_pose_references = job.get("pose_reference_inputs")
        if isinstance(configured_pose_references, dict):
            configured_reference = configured_pose_references.get(action_key(pose))
            if isinstance(configured_reference, str) and configured_reference.strip():
                pose_reference_input_name = configured_reference.strip()
        effective_pose_identity_input_policy = str(
            configured_pose_identity_input_policies.get(
                action_key(pose),
                pose_identity_input_policy,
            )
        ).strip()
        if (
            effective_pose_identity_input_policy
            == POSE_IDENTITY_INPUT_POLICY_PROFILE_GEOMETRY_FIRST
            and action_key(pose) not in {"eating", "grooming"}
        ):
            raise RuntimeError(
                "Main_case geometry-first identity supports only eating and grooming; "
                f"refusing pose {pose!r}."
            )
        if (
            effective_pose_identity_input_policy
            == POSE_IDENTITY_INPUT_POLICY_GROOMING_GEOMETRY_CANONICAL
            and action_key(pose) != "grooming"
        ):
            raise RuntimeError(
                "Canonical-identity geometry-first input supports only grooming; "
                f"refusing pose {pose!r}."
            )
        require_eating_mouth_contact = bool(
            force_qwen_reference and action_key(pose) == "eating"
        )
        strict_first_frame_required = bool(
            force_qwen_reference
            and job.get("qwen_first_frame_validated_required", True)
        )
        first_frame_retry_waves = {
            str(key): int(value)
            for key, value in dict(
                get_job(job_id).get("qwen_first_frame_retry_waves") or {}
            ).items()
        }
        first_frame_retry_wave = max(
            0,
            int(first_frame_retry_waves.get(action_key(pose), 0)),
        )
        pose_qwen_input_name = qwen_input_name
        pose_qwen_secondary_input_name = pose_reference_input_name
        pose_identity_input_name = qwen_input_name
        pose_geometry_input_name = pose_reference_input_name
        if force_qwen_reference and pose_reference_input_name is not None:
            pose_reference_input_sha256s = job.get("pose_reference_input_sha256s")
            expected_pose_reference_sha = (
                pose_reference_input_sha256s.get(action_key(pose))
                if isinstance(pose_reference_input_sha256s, dict)
                else None
            )
            pose_reference_path = COMFY_INPUT / pose_reference_input_name
            if not expected_pose_reference_sha:
                raise RuntimeError(
                    f"Frozen action guide lacks its persisted digest for pose {pose!r}."
                )
            if not pose_reference_path.is_file() or file_sha256(
                pose_reference_path
            ) != str(expected_pose_reference_sha):
                raise RuntimeError(
                    "Frozen action guide is missing or its digest changed for "
                    f"pose {pose!r}."
                )
        qwen_guided_identity_policy = effective_pose_identity_input_policy in {
            POSE_IDENTITY_INPUT_POLICY_CANONICAL,
            POSE_IDENTITY_INPUT_POLICY_PROFILE_GEOMETRY_FIRST,
            POSE_IDENTITY_INPUT_POLICY_GROOMING_GEOMETRY_CANONICAL,
        }
        grooming_pose_gate_required = bool(
            GENERATION_VISUAL_AUDITS_BLOCKING
            and force_qwen_reference
            and action_key(pose) == "grooming"
            and qwen_guided_identity_policy
        )
        advisory_grooming_minimum_gate_required = bool(
            not GENERATION_VISUAL_AUDITS_BLOCKING
            and force_qwen_reference
            and action_key(pose) == "grooming"
            and qwen_guided_identity_policy
        )
        pose_effective_subject_config = job.get("effective_subject_config")
        pose_action_config = (
            pose_effective_subject_config.get("actions", {}).get(action_key(pose))
            if isinstance(pose_effective_subject_config, dict)
            else None
        )
        pose_fur_surface_gate = (
            pose_action_config.get("fur_surface_gate")
            if isinstance(pose_action_config, dict)
            else None
        )
        fur_surface_gate_required = bool(
            GENERATION_VISUAL_AUDITS_BLOCKING
            and force_qwen_reference
            and action_key(pose) in {"eating", "grooming"}
            and qwen_guided_identity_policy
            and isinstance(pose_effective_subject_config, dict)
            and pose_effective_subject_config.get("profile_id") == "sleek_tabby_cat_v1"
            and isinstance(pose_fur_surface_gate, dict)
            and pose_fur_surface_gate.get("enabled") is True
        )
        if advisory_grooming_minimum_gate_required:
            update_job(
                job_id,
                grooming_pose_gate_status="minimum_contract_running",
                grooming_pose_gate_fail_closed=False,
                grooming_pose_gate_blocking=False,
                grooming_advisory_minimum_gate_policy=(
                    GROOMING_ADVISORY_MINIMUM_GATE_POLICY
                ),
                grooming_advisory_minimum_gate_version=(
                    GROOMING_ADVISORY_MINIMUM_GATE_VERSION
                ),
                grooming_advisory_minimum_gate_status="running",
                grooming_advisory_minimum_gate_blocking=False,
                grooming_advisory_minimum_gate_terminal_policy=(
                    "retry_then_identity_first_lowest_risk_degraded_pass_v1"
                ),
                grooming_advisory_minimum_gate_max_attempts=(
                    GROOMING_ADVISORY_TOTAL_ATTEMPTS
                ),
                grooming_advisory_action_draft_max_attempts=(
                    GROOMING_ADVISORY_ACTION_TRANSFER_MAX_ATTEMPTS
                ),
                grooming_advisory_identity_refine_max_attempts=(
                    GROOMING_ADVISORY_IDENTITY_REFINE_MAX_ATTEMPTS
                ),
            )
        fur_surface_rejection_streak = 0
        if fur_surface_gate_required:
            assert isinstance(pose_effective_subject_config, dict)
            # Reject mutable/missing profile inputs before spending a Qwen or
            # SCAIL GPU call. Candidate evaluation repeats this verification.
            resolve_main_case_fur_surface_baseline(
                pose_effective_subject_config,
                action_key(pose),
            )
            fur_surface_gate_status_by_action[action_key(pose)] = "running"
            update_job(
                job_id,
                fur_surface_gate_policy=FUR_SURFACE_GATE_POLICY,
                fur_surface_gate_version=FUR_SURFACE_GATE_VERSION,
                fur_surface_gate_threshold_version=(FUR_SURFACE_THRESHOLD_VERSION),
                fur_surface_gate_thresholds=fur_surface_thresholds_payload(),
                fur_surface_gate_scope=("sleek_tabby_main_case_guided_qwen_candidates"),
                fur_surface_gate_reliable_inputs=[
                    "alpha_cutout",
                    "rembg_alpha_cutout",
                ],
                fur_surface_gate_fail_closed=True,
                fur_surface_gate_max_consecutive_rejections=(
                    FUR_SURFACE_GATE_MAX_REJECTIONS
                ),
                fur_surface_gate_retry_policy=(FUR_SURFACE_GATE_RETRY_POLICY),
                fur_surface_gate_preserves_configured_eating_bowl_recovery=(True),
                fur_surface_gate_configured_eating_bowl_recovery_attempts=(
                    QWEN_BOWL_POSE_EXTRA_ATTEMPTS
                ),
                fur_surface_gate_status=dict(fur_surface_gate_status_by_action),
                fur_surface_gate_final_quality={
                    "scope": "qwen_action_reference_pre_scail",
                    "status": "running",
                    "actions": dict(fur_surface_gate_selected_metrics),
                    "final_video_frame_audit": False,
                },
            )
        grooming_pose_gate_guide_path: Path | None = None
        grooming_pose_gate_metadata_path: Path | None = None
        grooming_completion_mask_path: Path | None = None
        if (
            grooming_pose_gate_required
            or advisory_grooming_minimum_gate_required
        ):
            if pose_reference_input_name is None:
                raise RuntimeError(
                    "Main_case grooming pose gate requires its frozen geometry guide."
                )
            qwen_grooming_guide_path = COMFY_INPUT / pose_reference_input_name
            (
                grooming_pose_gate_guide_path,
                grooming_pose_gate_metadata_path,
                grooming_completion_mask_path,
            ) = resolve_main_case_grooming_pose_gate_metadata(
                job,
                qwen_grooming_guide_path,
            )
            if grooming_pose_gate_required:
                update_job(
                    job_id,
                    grooming_pose_gate_policy=GROOMING_POSE_GATE_POLICY,
                    grooming_pose_gate_version=GROOMING_POSE_GATE_VERSION,
                    grooming_pose_gate_scope=(
                        "main_case_guided_grooming_qwen_candidates"
                    ),
                    grooming_pose_gate_fail_closed=True,
                    grooming_pose_gate_input="rembg_alpha_cutout",
                    grooming_pose_gate_geometry_input=(
                        "public_open_channel_generation_plus_private_cv_gate_and_"
                        "non_qwen_completion_mask"
                    ),
                    grooming_pose_gate_status="running",
                )
        if (
            effective_pose_identity_input_policy
            == POSE_IDENTITY_INPUT_POLICY_DIRECT_UPLOAD
        ):
            if pose_reference_input_name is None:
                raise RuntimeError(
                    "Direct uploaded-photo action identity requires the frozen "
                    f"profile Image 2 for pose {pose!r}."
                )
            pose_qwen_input_name = uploaded_qwen_input_name
            pose_identity_input_name = uploaded_qwen_input_name
        elif (
            effective_pose_identity_input_policy
            == POSE_IDENTITY_INPUT_POLICY_PROFILE_GEOMETRY_FIRST
        ):
            if pose_reference_input_name is None:
                raise RuntimeError(
                    "Main_case geometry-first uploaded identity requires the frozen "
                    f"driver-derived geometry guide for pose {pose!r}."
                )
            if not QWEN_STYLE_LORA:
                raise RuntimeError(
                    "Main_case geometry-first uploaded identity requires the dedicated "
                    "QWEN_STYLE_LORA; refusing to fall back to photography models."
                )
            pose_qwen_input_name = pose_reference_input_name
            pose_qwen_secondary_input_name = uploaded_qwen_input_name
            pose_identity_input_name = pose_qwen_secondary_input_name
            pose_geometry_input_name = pose_reference_input_name
        elif (
            effective_pose_identity_input_policy
            == POSE_IDENTITY_INPUT_POLICY_GROOMING_GEOMETRY_CANONICAL
        ):
            if pose_reference_input_name is None:
                raise RuntimeError(
                    "Canonical-identity grooming geometry-first input requires "
                    "its frozen driver-derived guide."
                )
            pose_qwen_input_name = pose_reference_input_name
            pose_qwen_secondary_input_name = qwen_input_name
            pose_identity_input_name = qwen_input_name
            pose_geometry_input_name = pose_reference_input_name
        prompt = build_qwen_prompt(
            "" if transparent_gif else effective_prompt,
            pose,
            transparent_gif=transparent_gif,
            pose_reference_available=pose_reference_input_name is not None,
            profile_fur_lock=profile_fur_lock,
            pose_identity_input_policy=effective_pose_identity_input_policy,
            species=species,
        )
        canonical_action_reference = bool(
            effective_pose_identity_input_policy
            == POSE_IDENTITY_INPUT_POLICY_CANONICAL
            and pose_reference_input_name is not None
        )
        dog_eating_prompt_only_action_transfer_active = bool(
            canonical_action_reference
            and species == "dog"
            and action_key(pose) == "eating"
        )
        dog_drinking_prompt_only_action_transfer_active = bool(
            canonical_action_reference
            and species == "dog"
            and action_key(pose) == "drinking"
        )
        dog_toileting_prompt_only_action_transfer_active = bool(
            canonical_action_reference
            and species == "dog"
            and action_key(pose) == "toileting"
        )
        dog_prompt_only_action_transfer_active = bool(
            dog_eating_prompt_only_action_transfer_active
            or dog_drinking_prompt_only_action_transfer_active
            or dog_toileting_prompt_only_action_transfer_active
        )
        if dog_eating_prompt_only_action_transfer_active:
            update_job(
                job_id,
                dog_eating_first_frame_policy=(
                    DOG_EATING_PROMPT_ONLY_FIRST_FRAME_POLICY
                ),
                dog_eating_first_frame_prompt_only_action_geometry=False,
                dog_eating_first_frame_image1_role=(
                    "approved_upload_anchored_canonical_identity_sampler"
                ),
                dog_eating_first_frame_image2_role=(
                    "white_model_action_geometry_only"
                ),
                dog_eating_first_frame_white_model_loaded_by_qwen=True,
                dog_eating_first_frame_secondary_reference_loaded_by_qwen=True,
                dog_eating_first_frame_action_geometry_deferred_to_scail=False,
                dog_eating_first_frame_steps=(
                    DOG_EATING_PROMPT_ONLY_ACTION_TRANSFER_STEPS
                ),
                dog_eating_first_frame_denoise=(
                    DOG_EATING_PROMPT_ONLY_ACTION_TRANSFER_DENOISE
                ),
            )
        if dog_drinking_prompt_only_action_transfer_active:
            update_job(
                job_id,
                dog_drinking_first_frame_policy=(
                    DOG_DRINKING_PROMPT_ONLY_FIRST_FRAME_POLICY
                ),
                dog_drinking_first_frame_prompt_only_action_geometry=False,
                dog_drinking_first_frame_image1_role=(
                    "approved_upload_anchored_canonical_identity_sampler"
                ),
                dog_drinking_first_frame_image2_role=(
                    "white_model_action_geometry_only"
                ),
                dog_drinking_first_frame_white_model_loaded_by_qwen=True,
                dog_drinking_first_frame_secondary_reference_loaded_by_qwen=True,
                dog_drinking_first_frame_action_geometry_deferred_to_scail=False,
                dog_drinking_first_frame_steps=(
                    DOG_DRINKING_PROMPT_ONLY_ACTION_TRANSFER_STEPS
                ),
                dog_drinking_first_frame_denoise=(
                    DOG_DRINKING_PROMPT_ONLY_ACTION_TRANSFER_DENOISE
                ),
            )
        if dog_toileting_prompt_only_action_transfer_active:
            update_job(
                job_id,
                dog_toileting_first_frame_policy=(
                    DOG_TOILETING_PROMPT_ONLY_FIRST_FRAME_POLICY
                ),
                dog_toileting_first_frame_prompt_only_action_geometry=False,
                dog_toileting_first_frame_image1_role=(
                    "approved_upload_anchored_canonical_identity_sampler"
                ),
                dog_toileting_first_frame_image2_role=(
                    "white_model_action_geometry_only"
                ),
                dog_toileting_first_frame_white_model_loaded_by_qwen=True,
                dog_toileting_first_frame_secondary_reference_loaded_by_qwen=True,
                dog_toileting_first_frame_action_geometry_deferred_to_scail=False,
                dog_toileting_first_frame_steps=(
                    DOG_TOILETING_PROMPT_ONLY_ACTION_TRANSFER_STEPS
                ),
                dog_toileting_first_frame_denoise=(
                    DOG_TOILETING_PROMPT_ONLY_ACTION_TRANSFER_DENOISE
                ),
            )
        advisory_action_candidate_gate_required = bool(
            not GENERATION_VISUAL_AUDITS_BLOCKING
            and GENERATION_ACTION_CANDIDATE_GATES_ENABLED
            and force_qwen_reference
            and canonical_action_reference
            and action_key(pose) != "grooming"
        )
        action_candidate_audits_required = bool(
            GENERATION_VISUAL_AUDITS_BLOCKING
            or advisory_action_candidate_gate_required
        )
        if advisory_action_candidate_gate_required:
            update_job(
                job_id,
                action_candidate_gate_policy=(
                    GENERATION_ACTION_CANDIDATE_GATE_POLICY
                ),
                action_candidate_gate_status="running",
                action_candidate_gate_action=action_key(pose),
                action_candidate_gate_job_blocking=False,
                action_candidate_gate_returns_original_upload=False,
            )
        negative_prompt = QWEN_NEGATIVE
        if transparent_gif:
            active_pose_profiles = pose_generation_profiles(pose, pose_profiles)
            if not GENERATION_VISUAL_AUDITS_BLOCKING:
                if advisory_grooming_minimum_gate_required:
                    active_pose_profiles = advisory_grooming_generation_profiles(
                        pose,
                        active_pose_profiles,
                        enabled=True,
                    )
            negative_prompt += (
                ", "
                + (DOG_CARTOON_NEGATIVE if species == "dog" else CARTOON_NEGATIVE)
                + ", "
                + CHROMA_NEGATIVE
            )
            if profile_fur_negative:
                negative_prompt += ", " + profile_fur_negative
            if action_key(pose) == "running":
                negative_prompt += (
                    ", seated pose, sitting, crouched sitting, airborne leap, flying pose, extreme foreshortening, hidden hind leg, shortened hind leg, "
                    "fused legs, overlapping legs, detached paw, paw fragment, disconnected limb"
                )
            elif action_key(pose) == "eating":
                negative_prompt += (
                    ", food held in paw, paw holding kibble, raised front paw, paw-to-mouth gesture, "
                    "humanlike eating, anthropomorphic hand gesture, kibble outside bowl, "
                    "open mouth holding food, empty bowl, sniffing bowl, looking at bowl, "
                    "watching food, nose above kibble, muzzle hovering above bowl, visible air gap, "
                    "green gap between mouth and food, fully unobstructed nose"
                )
            elif action_key(pose) == "grooming":
                negative_prompt += (
                    ", neutral seated pose, both front paws planted, missing raised foreleg, "
                    "hidden raised foreleg, viewer-right raised foreleg, mirrored grooming pose, "
                    "detached rigid prop, strap, zipper, cord, pen, pencil, stick, pendant, black hanging object, "
                    "microphone, flashlight, battery, lipstick, bottle, capsule, handle, "
                    "green internal mark, gray guide material, black seam, dark seam, dark contour line, "
                    "continuous ink stroke, guide-shaped stripe, outlined foreleg, invented dark stripe, "
                    "sleeve, hoodie, hood, jacket, sweater, shirt, cuff, fabric fold, garment seam, "
                    "elephant, elephant head, elephant ears, trunk, tusk, dog head, non-cat species, hybrid animal, "
                    "pedestal, plinth, display stand, display base, circular platform, stage, floor disc"
                )
        keyframe = job_dir / f"pose_{index:02d}_keyframe.png"
        if transparent_gif:
            bowl_action = action_uses_bowl(pose)
            pose_fur_reference_path: Path | None = None
            effective_subject_config = job.get("effective_subject_config")
            if (
                action_candidate_audits_required
                and action_key(pose) == "eating"
                and isinstance(effective_subject_config, dict)
            ):
                action_config = effective_subject_config.get("actions", {}).get(
                    "eating"
                )
                fur_gate = (
                    action_config.get("fur_silhouette_gate")
                    if isinstance(action_config, dict)
                    else None
                )
                if isinstance(fur_gate, dict) and fur_gate.get("enabled"):
                    pose_fur_reference_path = Path(str(fur_gate.get("reference") or ""))
                    if not pose_fur_reference_path.is_file():
                        raise FileNotFoundError(
                            "Frozen eating fur-silhouette reference is missing: "
                            f"{pose_fur_reference_path}"
                        )
                    expected_fur_reference_sha = str(
                        fur_gate.get("reference_sha256") or ""
                    ).strip()
                    if (
                        expected_fur_reference_sha
                        and file_sha256(pose_fur_reference_path)
                        != expected_fur_reference_sha
                    ):
                        raise RuntimeError(
                            "Frozen eating fur-silhouette reference digest changed."
                        )
            pose_palette_bottom_exclusion = (
                BOWL_IDENTITY_PALETTE_BOTTOM_EXCLUSION if bowl_action else 0.0
            )
            pose_palette_reference_path = resolve_action_identity_palette_reference(
                uploaded_reference_job=bool(
                    force_qwen_reference and upload_integrity is not None
                ),
                identity_input_policy=effective_pose_identity_input_policy,
                uploaded_identity_source=identity_source_path,
                canonical_reference=canonical_similarity_reference,
            )
            grooming_identity_contract_reference_path = (
                canonical_similarity_reference or pose_palette_reference_path
            )
            pose_identity_palette_prompt_signature = (
                identity_palette_signature(
                    pose_palette_reference_path,
                    exclude_bottom_fraction=0.0,
                )
                if (
                    identity_palette_guidance_enabled
                    and pose_palette_reference_path is not None
                )
                else None
            )
            # Every generated action candidate must prove its identity palette;
            # visual-audit mode may change additional gates, never this floor.
            pose_identity_palette_gate_required = True
            pose_palette_negative = (
                identity_palette_negative_prompt(
                    pose_identity_palette_prompt_signature
                )
                if pose_identity_palette_prompt_signature is not None
                else source_palette_negative
            )
            pose_negative_prompt = negative_prompt
            if pose_palette_negative:
                pose_negative_prompt += ", " + pose_palette_negative
            pose_reference_identity_palette = (
                identity_palette_signature(
                    pose_palette_reference_path,
                    exclude_bottom_fraction=pose_palette_bottom_exclusion,
                )
                if (
                    pose_identity_palette_gate_required
                    and pose_palette_reference_path is not None
                )
                else None
            )
            pose_reference_upper_body_identity_palette = (
                identity_palette_signature(
                    pose_palette_reference_path,
                    exclude_bottom_fraction=(
                        BOWL_IDENTITY_PALETTE_UPPER_BODY_EXCLUSION
                    ),
                )
                if (
                    bowl_action
                    and pose_identity_palette_gate_required
                    and pose_palette_reference_path is not None
                )
                else None
            )
            advisory_grooming_identity_palette_reference = (
                identity_palette_signature(
                    pose_palette_reference_path,
                    exclude_bottom_fraction=0.0,
                )
                if (
                    advisory_grooming_minimum_gate_required
                    and pose_palette_reference_path is not None
                )
                else None
            )
            attempts: list[dict[str, Any]] = []
            best_attempt: dict[str, Any] | None = None
            best_action_candidate_degraded_attempt: dict[str, Any] | None = None
            best_strict_first_frame_attempt: dict[str, Any] | None = None
            best_advisory_grooming_degraded_attempt: dict[str, Any] | None = None
            best_raw_attempt: dict[str, Any] | None = None
            best_semantic_source: dict[str, Any] | None = None
            grooming_foreleg_repair_source: dict[str, Any] | None = None
            grooming_full_pose_refine_source: dict[str, Any] | None = None
            grooming_advisory_identity_refine_source: dict[str, Any] | None = None
            eating_identity_refine_source: dict[str, Any] | None = None
            eating_contact_repair_source: dict[str, Any] | None = None
            previous_pose_identity_palette_quality: dict[str, Any] | None = None
            previous_advisory_grooming_minimum_issues: list[str] = []
            canonical_grooming_completion_capable = bool(
                canonical_action_reference and action_key(pose) == "grooming"
            )
            active_pose_profiles = canonical_grooming_generation_profiles(
                pose,
                active_pose_profiles,
                enabled=(
                    GENERATION_VISUAL_AUDITS_BLOCKING
                    and canonical_grooming_completion_capable
                ),
            )
            active_pose_profiles = action_candidate_generation_profiles(
                pose,
                active_pose_profiles,
            )
            if len(active_pose_profiles) != QWEN_ACTION_CANDIDATE_ATTEMPTS:
                raise RuntimeError(
                    "Action candidate profile budget is not exactly five."
                )
            update_job(
                job_id,
                action_candidate_selection_policy=(
                    GENERATION_ACTION_CANDIDATE_GATE_POLICY
                ),
                action_candidate_attempt_budget=QWEN_ACTION_CANDIDATE_ATTEMPTS,
                action_candidate_selection_requires_identity=True,
                action_candidate_original_upload_fallback=False,
            )
            grooming_geometry_canonical_reference = bool(
                effective_pose_identity_input_policy
                == POSE_IDENTITY_INPUT_POLICY_GROOMING_GEOMETRY_CANONICAL
                and pose_reference_input_name is not None
            )
            geometry_first_action_reference = bool(
                effective_pose_identity_input_policy
                in {
                    POSE_IDENTITY_INPUT_POLICY_PROFILE_GEOMETRY_FIRST,
                    POSE_IDENTITY_INPUT_POLICY_GROOMING_GEOMETRY_CANONICAL,
                }
            )
            advisory_grooming_action_transfer_active = bool(
                not GENERATION_VISUAL_AUDITS_BLOCKING
                and canonical_grooming_completion_capable
            )
            advisory_grooming_action_canvas_audit: dict[str, Any] | None = None
            advisory_grooming_action_canvas_path: Path | None = None
            advisory_grooming_action_input_name: str | None = None
            advisory_grooming_action_geometry_guide_path: Path | None = None
            advisory_grooming_face_identity_path: Path | None = None
            advisory_grooming_face_identity_input_name: str | None = None
            advisory_grooming_face_identity_audit: dict[str, Any] | None = None
            advisory_grooming_local_action_edit_mask_path: Path | None = None
            advisory_grooming_local_action_edit_input_name: str | None = None
            advisory_grooming_local_action_edit_audit: dict[str, Any] | None = None
            advisory_grooming_white_frame0_audit: dict[str, Any] | None = None
            advisory_grooming_normalized_white_depth = False
            if advisory_grooming_action_transfer_active:
                canonical_action_identity_reference = COMFY_INPUT / qwen_input_name
                if not canonical_action_identity_reference.is_file():
                    raise RuntimeError(
                        "Advisory grooming action transfer lacks its canonical Image 1."
                    )
                if pose_reference_input_name is None:
                    raise RuntimeError(
                        "Advisory grooming action transfer lacks its appearance-free "
                        "Image 2 geometry guide."
                    )
                update_job(
                    job_id,
                    grooming_action_first_frame_input_policy=(
                        GROOMING_ADVISORY_FIRST_FRAME_INPUT_POLICY
                    ),
                    grooming_action_first_frame_open_channel_loaded_by_qwen=True,
                    grooming_action_first_frame_soft_foreleg_depth_loaded_by_qwen=False,
                    grooming_action_first_frame_normalized_white_maquette_depth_loaded_by_qwen=False,
                    grooming_action_first_frame_identity_input=(qwen_input_name),
                    grooming_action_first_frame_identity_confirmation_input=None,
                    grooming_action_first_frame_geometry_input=(
                        pose_reference_input_name
                    ),
                    grooming_action_first_frame_action_reference_role=(
                        "appearance_free_single_tone_action_geometry_image2"
                    ),
                    grooming_action_first_frame_action_reference_sha256=(
                        (job.get("pose_reference_input_sha256s") or {}).get(
                            action_key(pose)
                        )
                    ),
                    grooming_action_first_frame_action_canvas=None,
                    grooming_action_first_frame_image_roles={
                        "image1": "approved_canonical_identity_sampler",
                        "image2": "appearance_free_single_tone_action_geometry",
                    },
                    grooming_action_first_frame_negative_prompt_conditioning=False,
                    grooming_action_first_frame_sampling_schedule=(
                        "qwen_image_edit_2511_lightning_exact_four_step_v1"
                    ),
                    grooming_action_first_frame_uploaded_views_consumed_via_canonical_only=(
                        True
                    ),
                    grooming_action_first_frame_raw_action_rgb_loaded_by_qwen=False,
                    grooming_action_first_frame_source_chroma_loaded_by_qwen=False,
                    grooming_action_first_frame_source_texture_loaded_by_qwen=False,
                    grooming_action_first_frame_video_frame_loaded_by_qwen=False,
                    grooming_action_first_frame_white_maquette_loaded_by_qwen=False,
                    grooming_action_first_frame_white_maquette_frame_index=None,
                    grooming_action_first_frame_white_maquette_head_identity_removed=False,
                    grooming_action_first_frame_white_maquette_head_silhouette_removed=False,
                    grooming_action_first_frame_local_action_edit_mask_input=(
                        None
                    ),
                    grooming_action_first_frame_local_action_edit_mask=(
                        None
                    ),
                    grooming_action_first_frame_local_action_edit_mask_applied=(
                        False
                    ),
                    grooming_action_first_frame_full_canvas_action_redraw=True,
                    grooming_action_first_frame_prompt_only_action_geometry=False,
                    grooming_action_first_frame_post_qwen_pixel_composite_enabled=(
                        False
                    ),
                    grooming_action_first_frame_raw_qwen_output_required=True,
                    grooming_action_first_frame_raw_white_maquette_frame_loaded_by_qwen=(
                        False
                    ),
                    grooming_action_first_frame_canonical_face_reference=None,
                    grooming_action_first_frame_white_maquette_frame0=None,
                    grooming_action_first_frame_source_low_frequency_luma_loaded_by_qwen=(
                        False
                    ),
                    grooming_action_first_frame_normalized_low_frequency_geometry_loaded_by_qwen=True,
                    grooming_action_first_frame_identity_refine_required=False,
                    grooming_action_first_frame_identity_refine_mode=(
                        "conditional_only_for_detected_repairable_face_drift"
                    ),
                    grooming_action_first_frame_identity_safe_draft_byte_preservation_required=(
                        True
                    ),
                    grooming_action_first_frame_identity_refine_policy=(
                        GROOMING_ADVISORY_IDENTITY_REFINE_POLICY
                    ),
                    grooming_action_first_frame_identity_refine_white_maquette_loaded=(
                        False
                    ),
                )
            if canonical_action_reference:
                pose_effective_lora_policy = (
                    "qwen_lightning_only_after_canonical_identity_style_lock_v1"
                )
                pose_effective_user_loras: list[dict[str, Any]] = []
                pose_effective_lora_scale = 0.0
                pose_effective_lora_scope = (
                    "canonical_action_lightning_only; use "
                    "effective_pose_lora_attempts for selected per-pose settings"
                )
            elif grooming_geometry_canonical_reference:
                pose_effective_lora_policy = (
                    "geometry_first_user_loras_disabled_masked_pet3d_repair_v1"
                )
                pose_effective_user_loras = []
                pose_effective_lora_scale = 0.0
                pose_effective_lora_scope = (
                    "grooming_attempts_1_2_lightning_only; attempt_3 adds the "
                    "low_strength_pet3d_surface_prior_under_the_sampler_noise_mask"
                )
            elif geometry_first_action_reference:
                pose_effective_lora_policy = "geometry_first_user_loras_disabled_v1"
                pose_effective_user_loras = []
                pose_effective_lora_scale = 0.0
                pose_effective_lora_scope = (
                    "geometry_first_action_user_loras_disabled; use "
                    "effective_pose_lora_attempts for selected per-pose settings"
                )
            else:
                pose_effective_lora_policy = "profile_user_loras_enabled_v1"
                pose_effective_user_loras = active_pose_profiles[0]["user_loras"]
                pose_effective_lora_scale = active_pose_profiles[0][
                    "user_lora_strength_scale"
                ]
                pose_effective_lora_scope = (
                    "initial_profile_only; use effective_pose_lora_attempts "
                    "for selected per-pose settings"
                )
            pose_effective_lora_policies[action_key(pose)] = pose_effective_lora_policy
            update_job(
                job_id,
                pose_effective_lora_policy=pose_effective_lora_policy,
                pose_effective_lora_policies=dict(pose_effective_lora_policies),
                effective_pose_user_loras=pose_effective_user_loras,
                pose_user_lora_strength_scale=pose_effective_lora_scale,
                effective_pose_user_loras_scope=pose_effective_lora_scope,
            )
            if canonical_action_reference or grooming_geometry_canonical_reference:
                update_job(
                    job_id,
                    pose_photography_lora_policy=(
                        "disabled_after_canonical_identity_style_lock_v1"
                    ),
                    pose_builtin_style_lora_policy=(
                        "masked_pet3d_on_grooming_foreleg_repair_attempt_only_v1"
                        if grooming_geometry_canonical_reference
                        else "disabled_after_canonical_identity_style_lock_v1"
                    ),
                    pose_candidate_normalization_policy=(
                        QWEN_ACTION_REFERENCE_NORMALIZATION_POLICY
                    ),
                    pose_candidate_matte_policy=(QWEN_ACTION_CANDIDATE_MATTE_POLICY),
                )
            if should_reuse_validated_first_frame_cache(
                strict_first_frame_required=strict_first_frame_required,
                first_frame_retry_wave=first_frame_retry_wave,
            ):
                cached_attempt = load_validated_first_frame_cache(
                    job_id,
                    get_job(job_id),
                    action=action_key(pose),
                    pose_index=index,
                    job_dir=job_dir,
                )
                if cached_attempt is not None:
                    cached_path = job_dir / str(cached_attempt["keyframe"])
                    if (
                        action_key(pose) == "grooming"
                        and advisory_grooming_minimum_gate_required
                    ):
                        if (
                            advisory_grooming_identity_palette_reference is None
                            or pose_palette_reference_path is None
                            or grooming_identity_contract_reference_path is None
                            or grooming_pose_gate_guide_path is None
                            or grooming_pose_gate_metadata_path is None
                        ):
                            cached_minimum_gate = {
                                "policy": GROOMING_ADVISORY_MINIMUM_GATE_POLICY,
                                "version": GROOMING_ADVISORY_MINIMUM_GATE_VERSION,
                                "available": False,
                                "acceptable": False,
                                "issues": [
                                    "cache_revalidation_references_unavailable"
                                ],
                                "cache_revalidated": False,
                            }
                            cached_attempt[
                                "grooming_advisory_minimum_gate"
                            ] = cached_minimum_gate
                        else:
                            try:
                                cached_minimum_gate = (
                                    revalidate_cached_grooming_minimum_gate(
                                        cached_attempt,
                                        candidate_path=cached_path,
                                        identity_reference_palette=(
                                            advisory_grooming_identity_palette_reference
                                        ),
                                        identity_palette_reference_path=(
                                            pose_palette_reference_path
                                        ),
                                        identity_reference_path=(
                                            grooming_identity_contract_reference_path
                                        ),
                                        geometry_guide=grooming_pose_gate_guide_path,
                                        geometry_metadata=(
                                            grooming_pose_gate_metadata_path
                                        ),
                                        topology_guide=(
                                            qwen_grooming_guide_path
                                        ),
                                    )
                                )
                            except (OSError, ValueError, KeyError, RuntimeError) as exc:
                                LOGGER.warning(
                                    "%s cached grooming minimum-contract "
                                    "revalidation failed: %s",
                                    job_id,
                                    exc,
                                )
                                cached_minimum_gate = {
                                    "policy": (
                                        GROOMING_ADVISORY_MINIMUM_GATE_POLICY
                                    ),
                                    "version": (
                                        GROOMING_ADVISORY_MINIMUM_GATE_VERSION
                                    ),
                                    "available": False,
                                    "acceptable": False,
                                    "issues": [
                                        "cache_minimum_contract_revalidation_failed"
                                    ],
                                    "error": str(exc),
                                    "cache_revalidated": False,
                                }
                                cached_attempt[
                                    "grooming_advisory_minimum_gate"
                                ] = cached_minimum_gate
                        grooming_advisory_minimum_gate_attempts.append(
                            {
                                "pose_index": index,
                                "pose": pose,
                                "action": action_key(pose),
                                "attempt": cached_attempt.get("attempt"),
                                "seed": cached_attempt.get("seed"),
                                "gate": cached_minimum_gate,
                                "result": (
                                    "cache_revalidated_passed"
                                    if cached_minimum_gate.get("acceptable")
                                    is True
                                    else "cache_revalidated_rejected"
                                ),
                            }
                        )
                    cached_quality = pose_cutout_quality(
                        cached_path,
                        pose,
                        require_eating_mouth_contact=(
                            require_eating_mouth_contact
                        ),
                    )
                    cached_similarity = pose_similarity_quality(
                        cached_path,
                        pose=pose,
                        canonical_path=canonical_similarity_reference,
                        previous_poses=accepted_pose_similarity_references,
                    )
                    cached_attempt["quality"] = cached_quality
                    if action_key(pose) == "grooming":
                        cached_attempt["cache_revalidation_similarity_probe"] = (
                            cached_similarity
                        )
                    else:
                        cached_attempt["similarity"] = cached_similarity
                    if pose_palette_reference_path is not None:
                        cached_palette = evaluate_pose_identity_palette(
                            cached_path,
                            primary_reference=pose_reference_identity_palette,
                            upper_body_reference=(
                                pose_reference_upper_body_identity_palette
                            ),
                            bottom_exclusion=pose_palette_bottom_exclusion,
                            bowl_action=bowl_action,
                        )
                        cached_attempt["identity_palette"] = cached_palette[
                            "quality"
                        ]
                        cached_attempt["candidate_identity_palette"] = (
                            cached_palette["candidate"]
                        )
                        cached_attempt[
                            "candidate_upper_body_identity_palette"
                        ] = cached_palette["upper_body_candidate"]
                    if pose_fur_reference_path is not None:
                        cached_attempt["fur_silhouette"] = (
                            eating_fur_silhouette_quality(
                                cached_path,
                                pose_fur_reference_path,
                            )
                        )
                    cached_gate = dict(
                        cached_attempt.get("action_candidate_gate") or {}
                    )
                    (
                        cached_palette_acceptable,
                        cached_palette_source,
                    ) = strict_first_frame_identity_palette_evidence(
                        cached_attempt,
                        action=action_key(pose),
                    )
                    cached_gate.update(
                        acceptable=bool(
                            cached_quality.get("acceptable") is True
                            and cached_attempt["similarity"].get("acceptable")
                            is True
                            and cached_palette_acceptable
                            and (
                                not isinstance(
                                    cached_attempt.get("fur_silhouette"), dict
                                )
                                or cached_attempt["fur_silhouette"].get(
                                    "acceptable"
                                )
                                is True
                            )
                        ),
                        degraded_fallback_accepted=False,
                        cache_revalidated=True,
                        cache_revalidated_at=utc_now(),
                        cache_identity_palette_source=cached_palette_source,
                    )
                    cached_attempt["action_candidate_gate"] = cached_gate
                    cached_attempt["combined_penalty"] = round(
                        sum(
                            float(value.get("penalty", 0.0))
                            for value in (
                                cached_attempt.get("quality"),
                                cached_attempt.get("similarity"),
                                cached_attempt.get("identity_palette"),
                                cached_attempt.get("fur_silhouette"),
                                cached_attempt.get("fur_surface_gate"),
                                cached_attempt.get("grooming_pose_gate"),
                                cached_attempt.get(
                                    "grooming_advisory_minimum_gate"
                                ),
                            )
                            if isinstance(value, dict)
                        ),
                        6,
                    )
                    cache_issues = strict_first_frame_record_issues(
                        cached_attempt,
                        action=action_key(pose),
                    )
                    if not cache_issues:
                        best_attempt = cached_attempt
                        attempts.append(
                            {
                                key: value
                                for key, value in cached_attempt.items()
                                if key != "workflow"
                            }
                        )
                        first_frame_states = dict(
                            get_job(job_id).get("qwen_first_frame_states") or {}
                        )
                        first_frame_states[action_key(pose)] = {
                            "status": "validated_cache_hit",
                            "retry_wave": first_frame_retry_wave,
                            "source": cached_attempt[
                                "validated_first_frame_cache"
                            ],
                            "quality_revalidated": True,
                        }
                        update_job(
                            job_id,
                            qwen_first_frame_states=first_frame_states,
                            qwen_first_frame_current_action=action_key(pose),
                            qwen_first_frame_cache_status="hit",
                            qwen_first_frame_cache_fingerprint=(
                                cached_attempt["validated_first_frame_cache"][
                                    "fingerprint"
                                ]
                            ),
                        )
                        append_recovery_event(
                            job_id,
                            {
                                "stage": "cartoon_keyframe",
                                "pose_index": index,
                                "reason": "exact_input_validated_first_frame_cache_hit",
                                "strategy": QWEN_VALIDATED_FIRST_FRAME_POLICY,
                                "result": "accepted_after_current_gate_revalidation",
                                "action": action_key(pose),
                                "cache": cached_attempt[
                                    "validated_first_frame_cache"
                                ],
                            },
                        )
                    else:
                        LOGGER.warning(
                            "%s rejected cached first frame for %s after current "
                            "gate revalidation: %s",
                            job_id,
                            action_key(pose),
                            cache_issues,
                        )
            elif strict_first_frame_required:
                update_job(
                    job_id,
                    qwen_first_frame_cache_status=(
                        "disabled_for_fixed_five_candidate_budget"
                    ),
                    qwen_first_frame_cache_bypassed_action=action_key(pose),
                    qwen_first_frame_cache_bypassed_retry_wave=(
                        first_frame_retry_wave
                    ),
                )
            generation_pose_profiles = active_pose_profiles
            if strict_first_frame_required and generation_pose_profiles:
                first_frame_states = dict(
                    get_job(job_id).get("qwen_first_frame_states") or {}
                )
                first_frame_states[action_key(pose)] = {
                    "status": "generating_candidates",
                    "completed": False,
                    "quality_passed": False,
                    "published": False,
                    "retry_wave": first_frame_retry_wave,
                    "attempt_budget": len(generation_pose_profiles),
                    "attempt_limit": QWEN_FIRST_FRAME_MAX_ATTEMPTS,
                    "current_attempt": 0,
                    "original_upload_fallback_published": False,
                    "action_removed": False,
                }
                update_job(
                    job_id,
                    qwen_first_frame_states=first_frame_states,
                    qwen_first_frame_current_action=action_key(pose),
                    qwen_first_frame_cache_status="miss",
                )
            for attempt, profile in enumerate(generation_pose_profiles, start=1):
                raise_if_cancelled(job_id)
                if strict_first_frame_required:
                    first_frame_states = dict(
                        get_job(job_id).get("qwen_first_frame_states") or {}
                    )
                    action_state = dict(first_frame_states.get(action_key(pose)) or {})
                    action_state.update(
                        status="generating_candidate",
                        current_attempt=attempt,
                        attempts_completed=attempt - 1,
                        attempt_budget=len(generation_pose_profiles),
                        attempt_limit=QWEN_FIRST_FRAME_MAX_ATTEMPTS,
                    )
                    first_frame_states[action_key(pose)] = action_state
                    update_job(
                        job_id,
                        qwen_first_frame_states=first_frame_states,
                        qwen_first_frame_current_action=action_key(pose),
                        qwen_first_frame_current_attempt=attempt,
                        qwen_first_frame_attempt_limit=(
                            QWEN_FIRST_FRAME_MAX_ATTEMPTS
                        ),
                    )
                if attempt > 1:
                    update_job(
                        job_id,
                        stage="retrying_cartoon_keyframe",
                        quality_retry=attempt,
                        recovery_active=True,
                        recovery_attempt=attempt,
                    )
                attempt_seed = (
                    base_seed
                    + pose_seed_offset(pose, index)
                    + first_frame_retry_wave * QWEN_FIRST_FRAME_SEED_WAVE_STRIDE
                    + (attempt - 1) * 1009
                )
                prefix = f"local_video_api/{job_id}/qwen_pose_{index:02d}_attempt_{attempt:02d}"
                attempt_prompt = build_qwen_prompt(
                    "",
                    pose,
                    transparent_gif=True,
                    quality_retry=profile["quality_retry"],
                    pose_reference_available=pose_reference_input_name is not None,
                    profile_fur_lock=profile_fur_lock,
                    pose_identity_input_policy=(effective_pose_identity_input_policy),
                    species=species,
                )
                if pose_identity_palette_prompt_signature is not None:
                    attempt_prompt += "\n\n" + identity_palette_prompt(
                        pose_identity_palette_prompt_signature,
                        retry=attempt > 1,
                        previous_quality=previous_pose_identity_palette_quality,
                    )
                main_case_geometry_first = (
                    effective_pose_identity_input_policy
                    == POSE_IDENTITY_INPUT_POLICY_PROFILE_GEOMETRY_FIRST
                )
                grooming_geometry_canonical = (
                    effective_pose_identity_input_policy
                    == POSE_IDENTITY_INPUT_POLICY_GROOMING_GEOMETRY_CANONICAL
                )
                advisory_grooming_identity_refine_active = bool(
                    advisory_grooming_action_transfer_active
                    and isinstance(
                        grooming_advisory_identity_refine_source,
                        dict,
                    )
                    and attempt
                    > int(grooming_advisory_identity_refine_source["attempt"])
                )
                legacy_grooming_foreleg_repair_active = bool(
                    grooming_geometry_canonical
                    and attempt == GROOMING_QWEN_FORELEG_REPAIR_ATTEMPT
                    and isinstance(grooming_foreleg_repair_source, dict)
                    and not advisory_grooming_identity_refine_active
                )
                canonical_grooming_completion_active = bool(
                    canonical_grooming_completion_capable
                    and not advisory_grooming_action_transfer_active
                    and attempt == GROOMING_QWEN_FORELEG_REPAIR_ATTEMPT
                    and isinstance(grooming_foreleg_repair_source, dict)
                    and not advisory_grooming_identity_refine_active
                )
                canonical_grooming_full_pose_rescue_active = bool(
                    canonical_grooming_completion_capable
                    and not advisory_grooming_action_transfer_active
                    and attempt == GROOMING_CANONICAL_FULL_POSE_RESCUE_ATTEMPT
                    and not advisory_grooming_identity_refine_active
                )
                canonical_grooming_full_pose_refine_active = bool(
                    canonical_grooming_completion_capable
                    and not advisory_grooming_action_transfer_active
                    and attempt == GROOMING_CANONICAL_FULL_POSE_REFINE_ATTEMPT
                    and not advisory_grooming_identity_refine_active
                )
                canonical_grooming_full_pose_recovery_active = bool(
                    canonical_grooming_full_pose_rescue_active
                    or canonical_grooming_full_pose_refine_active
                )
                canonical_eating_identity_refine_active = bool(
                    canonical_action_reference
                    and action_key(pose) == "eating"
                    and attempt >= EATING_CANONICAL_IDENTITY_REFINE_ATTEMPT
                    and isinstance(eating_identity_refine_source, dict)
                )
                prior_strict_eating_candidate_exists = bool(
                    action_key(pose) == "eating"
                    and any(
                        not strict_first_frame_record_issues(
                            item,
                            action="eating",
                        )
                        for item in attempts
                        if isinstance(item, dict)
                    )
                )
                canonical_eating_contact_repair_active = bool(
                    canonical_action_reference
                    and action_key(pose) == "eating"
                    and attempt >= EATING_CONTACT_REPAIR_ATTEMPT
                    and isinstance(eating_contact_repair_source, dict)
                    and not canonical_eating_identity_refine_active
                    and not prior_strict_eating_candidate_exists
                )
                grooming_foreleg_repair_active = bool(
                    legacy_grooming_foreleg_repair_active
                    or canonical_grooming_completion_active
                )
                attempt_primary_input_name = pose_qwen_input_name
                attempt_secondary_input_name = pose_qwen_secondary_input_name
                attempt_tertiary_input_name: str | None = None
                attempt_additional_reference_input_names: list[str] | None = None
                attempt_noise_mask_input_name: str | None = None
                attempt_geometry_input_name = pose_geometry_input_name
                grooming_foreleg_repair_audit: dict[str, Any] | None = None
                grooming_full_pose_recovery_audit: dict[str, Any] | None = None
                grooming_advisory_identity_refine_audit: (
                    dict[str, Any] | None
                ) = None
                grooming_advisory_identity_refine_noise_mask_path: Path | None = None
                eating_identity_refine_audit: dict[str, Any] | None = None
                eating_contact_repair_audit: dict[str, Any] | None = None
                eating_contact_repair_canvas_path: Path | None = None
                eating_contact_repair_noise_mask_path: Path | None = None
                pose_transfer_seed_path: Path | None = None
                grooming_advisory_action_transfer_attempt: (
                    dict[str, Any] | None
                ) = None
                dog_eating_prompt_only_action_transfer_attempt: (
                    dict[str, Any] | None
                ) = None
                dog_drinking_prompt_only_action_transfer_attempt: (
                    dict[str, Any] | None
                ) = None
                dog_toileting_prompt_only_action_transfer_attempt: (
                    dict[str, Any] | None
                ) = None
                if dog_prompt_only_action_transfer_active:
                    attempt_primary_input_name = qwen_input_name
                    attempt_secondary_input_name = pose_reference_input_name
                    attempt_tertiary_input_name = None
                    attempt_additional_reference_input_names = None
                    attempt_noise_mask_input_name = None
                    attempt_geometry_input_name = pose_reference_input_name
                    if dog_eating_prompt_only_action_transfer_active:
                        attempt_prompt = (
                            build_dog_eating_prompt_only_action_transfer_prompt(
                                pose,
                                profile_fur_lock=profile_fur_lock,
                            )
                        )
                        dog_prompt_only_policy = (
                            DOG_EATING_PROMPT_ONLY_FIRST_FRAME_POLICY
                        )
                    elif dog_drinking_prompt_only_action_transfer_active:
                        attempt_prompt = (
                            build_dog_drinking_prompt_only_action_transfer_prompt(
                                pose,
                                profile_fur_lock=profile_fur_lock,
                            )
                        )
                        dog_prompt_only_policy = (
                            DOG_DRINKING_PROMPT_ONLY_FIRST_FRAME_POLICY
                        )
                    else:
                        attempt_prompt = (
                            build_dog_toileting_prompt_only_action_transfer_prompt(
                                pose,
                                profile_fur_lock=profile_fur_lock,
                            )
                        )
                        dog_prompt_only_policy = (
                            DOG_TOILETING_PROMPT_ONLY_FIRST_FRAME_POLICY
                        )
                    dog_prompt_only_action_transfer_attempt = {
                        "policy": dog_prompt_only_policy,
                        "image1_role": (
                            "approved_upload_anchored_canonical_identity_sampler"
                        ),
                        "image2_role": "white_model_action_geometry_only",
                        "image3_role": None,
                        "prompt_only_action_geometry": False,
                        "white_model_loaded_by_qwen": True,
                        "secondary_reference_loaded_by_qwen": True,
                        "foreign_pose_image_loaded_by_qwen": False,
                        "uploaded_identity_views_consumed_via_canonical_only": True,
                        "full_canvas_action_redraw": True,
                        "action_geometry_deferred_to_scail": False,
                        "sampler_noise_mask_input_name": None,
                        "post_qwen_pixel_composite_enabled": False,
                    }
                    if dog_eating_prompt_only_action_transfer_active:
                        dog_eating_prompt_only_action_transfer_attempt = (
                            dog_prompt_only_action_transfer_attempt
                        )
                    elif dog_drinking_prompt_only_action_transfer_active:
                        dog_drinking_prompt_only_action_transfer_attempt = (
                            dog_prompt_only_action_transfer_attempt
                        )
                    else:
                        dog_toileting_prompt_only_action_transfer_attempt = (
                            dog_prompt_only_action_transfer_attempt
                        )
                if advisory_grooming_action_transfer_active:
                    attempt_primary_input_name = qwen_input_name
                    if pose_reference_input_name is None:
                        raise RuntimeError(
                            "Grooming action transfer lost its appearance-free "
                            "geometry guide."
                        )
                    attempt_secondary_input_name = pose_reference_input_name
                    attempt_tertiary_input_name = None
                    attempt_additional_reference_input_names = None
                    # Keep the digest-bound canonical as Image 1/start latent and
                    # expose only the single-tone, appearance-free action guide as
                    # Image 2. Full-canvas redraw lets Qwen handle canonical inputs
                    # that are lying or otherwise far from the seated driver pose.
                    attempt_noise_mask_input_name = None
                    attempt_geometry_input_name = pose_reference_input_name
                    attempt_prompt = build_advisory_grooming_action_transfer_prompt(
                        pose,
                        profile_fur_lock=profile_fur_lock,
                        anatomy_priority_retry=(
                            attempt
                            >= GROOMING_ADVISORY_ANATOMY_FREEDOM_ATTEMPT
                        ),
                    )
                    grooming_advisory_action_transfer_attempt = {
                        "policy": GROOMING_ADVISORY_FIRST_FRAME_INPUT_POLICY,
                        "image1_role": "approved_canonical_identity_sampler",
                        "image2_role": (
                            "appearance_free_single_tone_action_geometry"
                        ),
                        "image3_role": None,
                        "uploaded_identity_views_consumed_via_canonical_only": True,
                        "prompt_only_action_geometry": False,
                        "video_first_frame_loaded_by_qwen": False,
                        "white_maquette_loaded_by_qwen": False,
                        "foreign_pose_image_loaded_by_qwen": False,
                        "raw_action_rgb_loaded_by_qwen": False,
                        "source_low_frequency_luma_loaded_by_qwen": False,
                        "normalized_low_frequency_geometry_loaded_by_qwen": True,
                        "public_open_channel_loaded_by_qwen": True,
                        "soft_foreleg_depth_loaded_by_qwen": False,
                        "white_maquette_frame_index": None,
                        "white_maquette_head_identity_removed": False,
                        "white_maquette_head_silhouette_removed": False,
                        "canonical_local_action_edit_mask": (
                            None
                        ),
                        "canonical_local_action_edit_mask_applied": False,
                        "full_canvas_action_redraw": True,
                        "anatomy_priority_retry": bool(
                            attempt
                            >= GROOMING_ADVISORY_ANATOMY_FREEDOM_ATTEMPT
                        ),
                        "guide_alignment_is_advisory": bool(
                            attempt
                            >= GROOMING_ADVISORY_ANATOMY_FREEDOM_ATTEMPT
                        ),
                        "terminal_completion_blocked_by_topology": False,
                        "full_canvas_redraw_reason": (
                            "local_latent_lock_retained_original_planted_paw_"
                            "topology"
                        ),
                        "sampler_noise_mask_input_name": (
                            attempt_noise_mask_input_name
                        ),
                        "post_qwen_pixel_composite_enabled": False,
                        "raw_qwen_output_required": True,
                        "raw_qwen_selection_byte_preserving": True,
                        "canonical_face_identity_reference": None,
                        "action_canvas": None,
                    }
                if advisory_grooming_identity_refine_active:
                    grooming_advisory_action_transfer_attempt = None
                    assert isinstance(
                        grooming_advisory_identity_refine_source,
                        dict,
                    )
                    if advisory_grooming_face_identity_input_name is None:
                        raise RuntimeError(
                            "Grooming identity restoration lost its canonical face."
                        )
                    refine_source_attempt = int(
                        grooming_advisory_identity_refine_source["attempt"]
                    )
                    refine_source_path = Path(
                        str(
                            grooming_advisory_identity_refine_source[
                                "cutout_path"
                            ]
                        )
                    )
                    expected_refine_source_sha256 = str(
                        grooming_advisory_identity_refine_source[
                            "cutout_sha256"
                        ]
                    )
                    if (
                        not refine_source_path.is_file()
                        or file_sha256(refine_source_path)
                        != expected_refine_source_sha256
                    ):
                        raise RuntimeError(
                            "Accepted grooming action draft disappeared or changed "
                            "before identity restoration."
                        )
                    refine_input_path = (
                        comfy_input_dir
                        / (
                            "grooming_advisory_identity_refine_source_"
                            f"attempt_{refine_source_attempt:02d}.png"
                        )
                    )
                    grooming_advisory_identity_refine_noise_mask_path = (
                        comfy_input_dir
                        / (
                            "grooming_advisory_identity_refine_face_noise_mask_"
                            f"attempt_{refine_source_attempt:02d}.png"
                        )
                    )
                    refine_canvas_audit = (
                        prepare_advisory_grooming_identity_refine_canvas(
                            refine_source_path,
                            refine_input_path,
                            grooming_advisory_identity_refine_noise_mask_path,
                            identity_reference_palette=(
                                advisory_grooming_identity_palette_reference
                            ),
                        )
                    )
                    refine_input_name = (
                        comfy_relative_dir / refine_input_path.name
                    ).as_posix()
                    attempt_primary_input_name = refine_input_name
                    attempt_secondary_input_name = qwen_input_name
                    attempt_tertiary_input_name = (
                        advisory_grooming_face_identity_input_name
                    )
                    attempt_additional_reference_input_names = [
                        attempt_secondary_input_name,
                        attempt_tertiary_input_name,
                    ]
                    attempt_noise_mask_input_name = (
                        comfy_relative_dir
                        / grooming_advisory_identity_refine_noise_mask_path.name
                    ).as_posix()
                    attempt_geometry_input_name = refine_input_name
                    attempt_prompt = build_advisory_grooming_identity_refine_prompt(
                        pose,
                        source_attempt=refine_source_attempt,
                        profile_fur_lock=profile_fur_lock,
                    )
                    if pose_identity_palette_prompt_signature is not None:
                        attempt_prompt += "\n\n" + identity_palette_prompt(
                            pose_identity_palette_prompt_signature,
                            retry=True,
                            previous_quality=(
                                previous_pose_identity_palette_quality
                            ),
                        )
                    grooming_advisory_identity_refine_audit = {
                        "policy": GROOMING_ADVISORY_IDENTITY_REFINE_POLICY,
                        "source_attempt": refine_source_attempt,
                        "source_path": refine_canvas_audit[
                            "sanitized_source_path"
                        ],
                        "source_sha256": refine_canvas_audit[
                            "sanitized_source_sha256"
                        ],
                        "original_source_path": str(refine_source_path),
                        "original_source_sha256": expected_refine_source_sha256,
                        "refine_canvas": refine_canvas_audit,
                        "image1_role": (
                            "accepted_grooming_pose_sampler_no_identity_authority"
                        ),
                        "image2_role": (
                            "approved_canonical_sole_whole_pet_identity_authority"
                        ),
                        "image3_role": (
                            "same_canonical_enlarged_face_identity_authority"
                        ),
                        "white_maquette_loaded": False,
                        "action_geometry_source": "accepted_pose_draft_only",
                        "sampler_noise_mask": {
                            "input_name": attempt_noise_mask_input_name,
                            "path": str(
                                grooming_advisory_identity_refine_noise_mask_path
                            ),
                            "sha256": refine_canvas_audit[
                                "noise_mask_target_sha256"
                            ],
                            "policy": refine_canvas_audit["noise_mask_policy"],
                        },
                        "full_frame_sampling": False,
                        "body_and_raised_paw_pixel_lock_required": True,
                    }
                    update_job(
                        job_id,
                        grooming_advisory_identity_refine_status="running",
                        grooming_advisory_identity_refine_policy=(
                            GROOMING_ADVISORY_IDENTITY_REFINE_POLICY
                        ),
                        grooming_advisory_identity_refine_source_attempt=(
                            refine_source_attempt
                        ),
                        grooming_advisory_identity_refine_attempt=attempt,
                        grooming_advisory_identity_refine_white_maquette_loaded=(
                            False
                        ),
                    )
                if grooming_foreleg_repair_active:
                    assert isinstance(grooming_foreleg_repair_source, dict)
                    repair_source_palette_quality = grooming_foreleg_repair_source.get(
                        "identity_palette_quality"
                    )
                    if (
                        not isinstance(repair_source_palette_quality, dict)
                        or repair_source_palette_quality.get("available") is not True
                        or repair_source_palette_quality.get("acceptable") is not True
                    ):
                        raise RuntimeError(
                            "Grooming foreleg repair source lost its required "
                            "identity-palette-safe verdict."
                        )
                    repair_source_path = Path(
                        str(grooming_foreleg_repair_source["cutout_path"])
                    )
                    if not repair_source_path.is_file():
                        raise RuntimeError(
                            "Grooming foreleg repair source disappeared before retry."
                        )
                    expected_repair_source_sha256 = str(
                        grooming_foreleg_repair_source["cutout_sha256"]
                    )
                    if file_sha256(repair_source_path) != expected_repair_source_sha256:
                        raise RuntimeError(
                            "Grooming foreleg repair source digest changed."
                        )
                    if (
                        grooming_pose_gate_guide_path is None
                        or not grooming_pose_gate_guide_path.is_file()
                    ):
                        raise RuntimeError(
                            "Grooming foreleg repair anatomy guide is unavailable."
                        )
                    repair_input_path = (
                        comfy_input_dir / "grooming_foreleg_repair_source.png"
                    )
                    if canonical_grooming_completion_active:
                        if (
                            grooming_completion_mask_path is None
                            or not grooming_completion_mask_path.is_file()
                        ):
                            raise RuntimeError(
                                "Canonical grooming completion mask is unavailable."
                            )
                        repair_canvas_audit = (
                            prepare_canonical_grooming_foreleg_completion_canvas(
                                repair_source_path,
                                grooming_completion_mask_path,
                                cartoon_transparent,
                                repair_input_path,
                            )
                        )
                    else:
                        repair_canvas_audit = prepare_grooming_foreleg_repair_canvas(
                            repair_source_path,
                            qwen_grooming_guide_path,
                            cartoon_transparent,
                            repair_input_path,
                        )
                    repair_input_sha256 = file_sha256(repair_input_path)
                    repair_noise_mask_path = Path(
                        str(repair_canvas_audit["noise_mask_target_path"])
                    )
                    if not repair_noise_mask_path.is_file():
                        raise RuntimeError(
                            "Grooming foreleg repair noise mask is unavailable."
                        )
                    repair_anatomy_path = (
                        comfy_input_dir / "grooming_foreleg_repair_anatomy_guide.png"
                    )
                    if (
                        canonical_grooming_completion_active
                        and advisory_grooming_action_canvas_path is not None
                        and advisory_grooming_action_canvas_path.is_file()
                        and isinstance(
                            advisory_grooming_action_canvas_audit,
                            dict,
                        )
                    ):
                        shutil.copy2(
                            advisory_grooming_action_canvas_path,
                            repair_anatomy_path,
                        )
                        repair_anatomy_audit = {
                            "policy": GROOMING_CANONICAL_POSE_TRANSFER_SEED_POLICY,
                            "source_path": str(
                                advisory_grooming_action_canvas_path
                            ),
                            "source_sha256": file_sha256(
                                advisory_grooming_action_canvas_path
                            ),
                            "target_path": str(repair_anatomy_path),
                            "target_sha256": file_sha256(repair_anatomy_path),
                            "role": "image3_soft_foreleg_depth_anatomy_only",
                            "public_open_channel_exposed": False,
                            "soft_foreleg_depth_exposed": True,
                            "source_rgb_or_luma_exposed": False,
                        }
                    else:
                        repair_anatomy_audit = (
                            prepare_grooming_foreleg_repair_anatomy_guide(
                                (
                                    qwen_grooming_guide_path
                                    if canonical_grooming_completion_active
                                    else grooming_pose_gate_guide_path
                                ),
                                repair_anatomy_path,
                            )
                        )
                    attempt_primary_input_name = (
                        comfy_relative_dir / repair_input_path.name
                    ).as_posix()
                    attempt_tertiary_input_name = (
                        comfy_relative_dir / repair_anatomy_path.name
                    ).as_posix()
                    attempt_noise_mask_input_name = (
                        comfy_relative_dir / repair_noise_mask_path.name
                    ).as_posix()
                    if canonical_grooming_completion_active:
                        attempt_secondary_input_name = qwen_input_name
                    if not attempt_secondary_input_name:
                        raise RuntimeError(
                            "Grooming foreleg repair canonical Image 2 is unavailable."
                        )
                    attempt_additional_reference_input_names = [
                        attempt_secondary_input_name,
                        attempt_tertiary_input_name,
                    ]
                    active_repair_policy = (
                        GROOMING_CANONICAL_FORELEG_COMPLETION_POLICY
                        if canonical_grooming_completion_active
                        else GROOMING_QWEN_FORELEG_REPAIR_POLICY
                    )
                    active_repair_style_policy = (
                        "disabled_for_canonical_masked_completion"
                        if canonical_grooming_completion_active
                        else GROOMING_QWEN_FORELEG_REPAIR_STYLE_POLICY
                    )
                    active_repair_style_lora = (
                        None
                        if canonical_grooming_completion_active
                        else QWEN_STYLE_LORA
                    )
                    active_repair_style_strength = (
                        0.0
                        if canonical_grooming_completion_active
                        else GROOMING_QWEN_FORELEG_REPAIR_STYLE_STRENGTH
                    )
                    active_composite_gate_policy = (
                        None
                        if canonical_grooming_completion_active
                        else GROOMING_QWEN_FORELEG_REPAIR_COMPOSITE_GATE_POLICY
                    )
                    attempt_prompt = (
                        build_canonical_grooming_foreleg_completion_prompt(
                            pose,
                            source_attempt=int(
                                grooming_foreleg_repair_source["attempt"]
                            ),
                            profile_fur_lock=profile_fur_lock,
                            soft_depth_anatomy=bool(
                                repair_anatomy_audit.get(
                                    "soft_foreleg_depth_exposed"
                                )
                            ),
                        )
                        if canonical_grooming_completion_active
                        else build_grooming_foreleg_repair_prompt(
                            pose,
                            source_attempt=int(
                                grooming_foreleg_repair_source["attempt"]
                            ),
                            profile_fur_lock=profile_fur_lock,
                        )
                    )
                    if pose_identity_palette_prompt_signature is not None:
                        attempt_prompt += "\n\n" + identity_palette_prompt(
                            pose_identity_palette_prompt_signature,
                            retry=True,
                            previous_quality=(
                                previous_pose_identity_palette_quality
                            ),
                        )
                    grooming_foreleg_repair_audit = {
                        "policy": active_repair_policy,
                        "production_canonical_masked_completion": (
                            canonical_grooming_completion_active
                        ),
                        "source_attempt": int(
                            grooming_foreleg_repair_source["attempt"]
                        ),
                        "source_path": str(repair_source_path),
                        "source_sha256": expected_repair_source_sha256,
                        "source_gate_issues": list(
                            grooming_foreleg_repair_source["gate_issues"]
                        ),
                        "source_fur_surface_issues": list(
                            grooming_foreleg_repair_source["fur_surface_issues"]
                        ),
                        "source_identity_palette_available": True,
                        "source_identity_palette_acceptable": True,
                        "source_identity_palette_issues": list(
                            repair_source_palette_quality.get("issues", [])
                        ),
                        "source_identity_palette_penalty": float(
                            repair_source_palette_quality.get("penalty", 0.0)
                        ),
                        "comfy_input_path": str(repair_input_path),
                        "comfy_input_sha256": repair_input_sha256,
                        "foreleg_seed": (
                            None
                            if canonical_grooming_completion_active
                            else repair_canvas_audit
                        ),
                        "completion_canvas": (
                            repair_canvas_audit
                            if canonical_grooming_completion_active
                            else None
                        ),
                        "anatomy_guide": repair_anatomy_audit,
                        "image1_role": (
                            "identity_safe_candidate_unchanged_masked_sampler"
                            if canonical_grooming_completion_active
                            else "band_limited_canonical_foreleg_seeded_candidate_sampler"
                        ),
                        "image2_role": (
                            "approved_canonical_identity_appearance_authority"
                        ),
                        "image3_role": (
                            "soft_foreleg_depth_grooming_geometry_only"
                            if canonical_grooming_completion_active
                            else "single_tone_open_channel_grooming_geometry_only"
                        ),
                        "synthetic_capsule_geometry_exposed_on_repair_attempt": (False),
                        "open_channel_geometry_exposed_on_repair_attempt": (
                            not canonical_grooming_completion_active
                        ),
                        "soft_foreleg_depth_exposed_on_repair_attempt": (
                            canonical_grooming_completion_active
                        ),
                        "private_cv_luma_exposed_on_repair_attempt": False,
                        "sampler_noise_mask_role": (
                            "expanded_context_full_denoise_muzzle_shoulder_chest_foreleg"
                        ),
                        "sampler_noise_mask_input_name": (
                            attempt_noise_mask_input_name
                        ),
                        "sampler_noise_mask_sha256": repair_canvas_audit[
                            "noise_mask_target_sha256"
                        ],
                        "localized_style_policy": (active_repair_style_policy),
                        "localized_style_lora": active_repair_style_lora,
                        "localized_style_strength": (active_repair_style_strength),
                        "localized_style_scope": (
                            "none; canonical Image 2 is the only style authority"
                            if canonical_grooming_completion_active
                            else (
                                "sampler_noise_mask_only; image2_remains_identity_"
                                "palette_marking_and_coat_length_authority"
                            )
                        ),
                        "composite_gate_policy": active_composite_gate_policy,
                        "composite_recovery_enabled": (
                            not canonical_grooming_completion_active
                        ),
                        "composite_edge_density_ratio_min": (
                            None
                            if canonical_grooming_completion_active
                            else GROOMING_QWEN_FORELEG_REPAIR_EDGE_DENSITY_RATIO_MIN
                        ),
                        "composite_planted_paw_gradient_p90_max": (
                            None
                            if canonical_grooming_completion_active
                            else GROOMING_QWEN_FORELEG_REPAIR_PLANTED_PAW_GRADIENT_P90_MAX
                        ),
                        "composite_raised_paw_gradient_p90_max": (
                            None
                            if canonical_grooming_completion_active
                            else GROOMING_QWEN_FORELEG_REPAIR_RAISED_PAW_GRADIENT_P90_MAX
                        ),
                        "canonical_completion_edge_evidence_policy": (
                            GROOMING_CANONICAL_COMPLETION_EDGE_EVIDENCE_POLICY
                            if canonical_grooming_completion_active
                            else None
                        ),
                        "canonical_completion_edge_density_ratio_min": (
                            GROOMING_CANONICAL_COMPLETION_EDGE_DENSITY_RATIO_MIN
                            if canonical_grooming_completion_active
                            else None
                        ),
                        "canonical_completion_independent_fur_surface_required": (
                            canonical_grooming_completion_active
                        ),
                        "global_pose_threshold_changed": False,
                        "main_case_rgb_appearance_exposed_on_repair_attempt": False,
                    }
                    update_job(
                        job_id,
                        grooming_foreleg_repair_active=True,
                        grooming_foreleg_repair_policy=(active_repair_policy),
                        grooming_foreleg_repair_source_attempt=(
                            grooming_foreleg_repair_audit["source_attempt"]
                        ),
                        grooming_foreleg_repair_source_sha256=(
                            expected_repair_source_sha256
                        ),
                        grooming_foreleg_repair_input_sha256=(repair_input_sha256),
                        grooming_foreleg_repair_edit_mask_sha256=(
                            repair_canvas_audit["edit_mask_sha256"]
                        ),
                        grooming_foreleg_repair_anatomy_sha256=(
                            repair_anatomy_audit["target_sha256"]
                        ),
                        grooming_foreleg_repair_noise_mask_sha256=(
                            repair_canvas_audit["noise_mask_target_sha256"]
                        ),
                        grooming_foreleg_repair_style_policy=(
                            active_repair_style_policy
                        ),
                        grooming_foreleg_repair_style_lora=(active_repair_style_lora),
                        grooming_foreleg_repair_style_strength=(
                            active_repair_style_strength
                        ),
                        grooming_foreleg_repair_style_scope=(
                            "none"
                            if canonical_grooming_completion_active
                            else "sampler_noise_mask_only"
                        ),
                        grooming_foreleg_repair_composite_gate_policy=(
                            active_composite_gate_policy
                        ),
                        grooming_foreleg_repair_composite_recovery_enabled=(
                            not canonical_grooming_completion_active
                        ),
                        grooming_foreleg_repair_composite_edge_density_ratio_min=(
                            None
                            if canonical_grooming_completion_active
                            else GROOMING_QWEN_FORELEG_REPAIR_EDGE_DENSITY_RATIO_MIN
                        ),
                        grooming_foreleg_repair_composite_planted_paw_gradient_p90_max=(
                            None
                            if canonical_grooming_completion_active
                            else GROOMING_QWEN_FORELEG_REPAIR_PLANTED_PAW_GRADIENT_P90_MAX
                        ),
                        grooming_foreleg_repair_composite_raised_paw_gradient_p90_max=(
                            None
                            if canonical_grooming_completion_active
                            else GROOMING_QWEN_FORELEG_REPAIR_RAISED_PAW_GRADIENT_P90_MAX
                        ),
                        grooming_canonical_completion_edge_evidence_policy=(
                            GROOMING_CANONICAL_COMPLETION_EDGE_EVIDENCE_POLICY
                            if canonical_grooming_completion_active
                            else None
                        ),
                        grooming_canonical_completion_edge_density_ratio_min=(
                            GROOMING_CANONICAL_COMPLETION_EDGE_DENSITY_RATIO_MIN
                            if canonical_grooming_completion_active
                            else None
                        ),
                        grooming_canonical_completion_independent_fur_surface_required=(
                            canonical_grooming_completion_active
                        ),
                        grooming_foreleg_repair_global_edge_density_min=(
                            GROOMING_QWEN_FORELEG_REPAIR_GLOBAL_EDGE_DENSITY_MIN
                        ),
                        grooming_foreleg_repair_global_pose_threshold_changed=False,
                    )
                if canonical_grooming_full_pose_recovery_active:
                    if (
                        grooming_pose_gate_guide_path is None
                        or not grooming_pose_gate_guide_path.is_file()
                    ):
                        raise RuntimeError(
                            "Canonical grooming pose-transfer validation guide is "
                            "unavailable."
                        )
                    canonical_recovery_reference_path = COMFY_INPUT / qwen_input_name
                    if not canonical_recovery_reference_path.is_file():
                        raise RuntimeError(
                            "Canonical grooming pose-transfer identity reference "
                            "is unavailable."
                        )
                    action_reference_text = (
                        str(pose_action_config.get("reference") or "").strip()
                        if isinstance(pose_action_config, dict)
                        else ""
                    )
                    expected_action_reference_sha256 = (
                        str(pose_action_config.get("reference_sha256") or "").strip()
                        if isinstance(pose_action_config, dict)
                        else ""
                    )
                    action_reference_path = (
                        Path(action_reference_text)
                        if action_reference_text
                        else Path()
                    )
                    if (
                        not action_reference_text
                        or not expected_action_reference_sha256
                        or not action_reference_path.is_file()
                        or file_sha256(action_reference_path)
                        != expected_action_reference_sha256
                    ):
                        raise RuntimeError(
                            "Canonical grooming pose-transfer action reference is "
                            "missing or its digest changed."
                        )
                    pose_transfer_seed_path = (
                        comfy_input_dir / "grooming_canonical_pose_transfer_seed.png"
                    )
                    if (
                        grooming_completion_mask_path is None
                        or not grooming_completion_mask_path.is_file()
                    ):
                        raise RuntimeError(
                            "Canonical grooming pose-transfer foreleg geometry "
                            "is unavailable."
                        )
                    pose_transfer_seed_audit = (
                        prepare_canonical_grooming_pose_transfer_seed(
                            action_reference_path,
                            canonical_recovery_reference_path,
                            pose_transfer_seed_path,
                            geometry_guide=qwen_grooming_guide_path,
                            foreleg_guide=grooming_completion_mask_path,
                        )
                    )
                    pose_transfer_seed_input_name = (
                        comfy_relative_dir / pose_transfer_seed_path.name
                    ).as_posix()
                    attempt_primary_input_name = pose_transfer_seed_input_name
                    attempt_secondary_input_name = qwen_input_name
                    attempt_tertiary_input_name = None
                    attempt_additional_reference_input_names = [
                        attempt_secondary_input_name,
                    ]
                    attempt_noise_mask_input_name = None
                    attempt_prompt = build_canonical_grooming_pose_transfer_prompt(
                        pose,
                        profile_fur_lock=profile_fur_lock,
                    )
                    recovery_mode = "canonical_palette_action_pose_transfer"
                    recovery_policy = GROOMING_CANONICAL_POSE_TRANSFER_SEED_POLICY
                    recovery_source: dict[str, Any] = {
                        "action_reference_path": str(action_reference_path),
                        "action_reference_sha256": (
                            expected_action_reference_sha256
                        ),
                        "pose_transfer_seed": pose_transfer_seed_audit,
                    }
                    recovery_fallback = bool(
                        canonical_grooming_full_pose_refine_active
                    )
                    pose_safe_candidate_used_as_sampler = False
                    recovery_source_sha256 = expected_action_reference_sha256
                    if (
                        canonical_grooming_full_pose_refine_active
                        and isinstance(grooming_full_pose_refine_source, dict)
                    ):
                        refine_source_path = Path(
                            str(grooming_full_pose_refine_source.get("cutout_path") or "")
                        )
                        expected_refine_source_sha256 = str(
                            grooming_full_pose_refine_source.get("cutout_sha256") or ""
                        )
                        refine_eligibility = grooming_full_pose_refine_source.get(
                            "eligibility"
                        )
                        expected_eligibility_sha256 = str(
                            grooming_full_pose_refine_source.get("eligibility_sha256")
                            or ""
                        )
                        source_matte = grooming_full_pose_refine_source.get(
                            "seed_bound_matte"
                        )
                        if (
                            int(grooming_full_pose_refine_source.get("attempt") or 0)
                            != GROOMING_CANONICAL_FULL_POSE_RESCUE_ATTEMPT
                            or not refine_source_path.is_file()
                            or not expected_refine_source_sha256
                            or file_sha256(refine_source_path)
                            != expected_refine_source_sha256
                            or not isinstance(refine_eligibility, dict)
                            or refine_eligibility.get("policy")
                            != GROOMING_CANONICAL_POSE_SAFE_REFINE_SOURCE_POLICY
                            or refine_eligibility.get("eligible") is not True
                            or not expected_eligibility_sha256
                            or json_value_sha256(refine_eligibility)
                            != expected_eligibility_sha256
                            or not isinstance(source_matte, dict)
                            or source_matte.get("policy")
                            != GROOMING_CANONICAL_SEED_BOUND_MATTE_POLICY
                            or source_matte.get("pose_transfer_seed_sha256")
                            != pose_transfer_seed_audit["target_sha256"]
                        ):
                            raise RuntimeError(
                                "Canonical grooming pose-safe identity-refine source "
                                "lost its digest-bound gate provenance."
                            )
                        refine_canvas_path = (
                            comfy_input_dir
                            / "grooming_full_pose_identity_refine_source.png"
                        )
                        refine_canvas_audit = (
                            prepare_canonical_grooming_full_pose_refine_canvas(
                                refine_source_path,
                                refine_canvas_path,
                            )
                        )
                        attempt_primary_input_name = (
                            comfy_relative_dir / refine_canvas_path.name
                        ).as_posix()
                        attempt_tertiary_input_name = pose_transfer_seed_input_name
                        attempt_additional_reference_input_names = [
                            attempt_secondary_input_name,
                            attempt_tertiary_input_name,
                        ]
                        attempt_prompt = (
                            build_canonical_grooming_full_pose_identity_refine_prompt(
                                pose,
                                source_attempt=int(
                                    grooming_full_pose_refine_source["attempt"]
                                ),
                                profile_fur_lock=profile_fur_lock,
                            )
                        )
                        recovery_mode = "pose_safe_identity_refine"
                        recovery_policy = GROOMING_CANONICAL_FULL_POSE_REFINE_POLICY
                        recovery_source = {
                            **grooming_full_pose_refine_source,
                            "refine_canvas": refine_canvas_audit,
                            "pose_transfer_seed": pose_transfer_seed_audit,
                        }
                        recovery_fallback = False
                        pose_safe_candidate_used_as_sampler = True
                        recovery_source_sha256 = expected_refine_source_sha256
                    if pose_identity_palette_prompt_signature is not None:
                        attempt_prompt += "\n\n" + identity_palette_prompt(
                            pose_identity_palette_prompt_signature,
                            retry=True,
                            previous_quality=(
                                previous_pose_identity_palette_quality
                            ),
                        )
                    grooming_full_pose_recovery_audit = {
                        "policy": recovery_policy,
                        "mode": recovery_mode,
                        "attempt": attempt,
                        "fallback_to_fresh_full_pose_rescue": recovery_fallback,
                        "image1_role": (
                            "seed_bound_pose_and_surface_approved_candidate_"
                            "sampler_no_identity_authority"
                            if pose_safe_candidate_used_as_sampler
                            else "canonical_palette_soft_foreleg_depth_action_"
                            "anatomy_sampler"
                        ),
                        "image2_role": (
                            "approved_canonical_sole_identity_appearance_authority"
                        ),
                        "image3_role": (
                            "digest_bound_soft_foreleg_depth_action_corroboration"
                            if pose_safe_candidate_used_as_sampler
                            else None
                        ),
                        "anatomy_seed": pose_transfer_seed_audit,
                        "source": recovery_source,
                        "sampler_noise_mask": None,
                        "full_frame_sampling": True,
                        "qwen_lightning_only": True,
                        "style_lora": None,
                        "main_case_rgb_appearance_exposed": (
                            pose_safe_candidate_used_as_sampler
                        ),
                        "main_case_raw_luma_exposed": (
                            pose_safe_candidate_used_as_sampler
                        ),
                        "main_case_low_frequency_action_geometry_exposed": False,
                        "geometry_distance_field_exposed": True,
                        "uploaded_front_identity_reference_exposed": False,
                        "uploaded_identity_views_consumed_via_canonical_only": True,
                        "public_open_channel_geometry_exposed": False,
                        "soft_foreleg_depth_geometry_exposed": True,
                        "rejected_qwen_candidate_used_as_sampler": (
                            pose_safe_candidate_used_as_sampler
                        ),
                        "candidate_rejection_scope": (
                            "identity_palette_only_after_pose_quality_and_surface_pass"
                            if pose_safe_candidate_used_as_sampler
                            else None
                        ),
                        "seed_bound_matte_required": True,
                        "pose_safe_source_policy": (
                            GROOMING_CANONICAL_POSE_SAFE_REFINE_SOURCE_POLICY
                        ),
                        "identity_gate_unchanged": True,
                        "pose_gate_unchanged": True,
                        "fur_surface_gate_unchanged": True,
                        "global_pose_threshold_changed": False,
                    }
                    update_job(
                        job_id,
                        grooming_full_pose_recovery_active=True,
                        grooming_full_pose_recovery_attempt=attempt,
                        grooming_full_pose_recovery_mode=recovery_mode,
                        grooming_full_pose_recovery_policy=recovery_policy,
                        grooming_full_pose_recovery_fallback=(recovery_fallback),
                        grooming_full_pose_recovery_anatomy_sha256=(
                            pose_transfer_seed_audit["target_sha256"]
                        ),
                        grooming_full_pose_recovery_source_sha256=(
                            recovery_source_sha256
                        ),
                        grooming_pose_transfer_seed_sha256=(
                            pose_transfer_seed_audit["target_sha256"]
                        ),
                        grooming_pose_transfer_action_reference_sha256=(
                            expected_action_reference_sha256
                        ),
                        grooming_pose_transfer_rejected_candidate_used_as_sampler=(
                            pose_safe_candidate_used_as_sampler
                        ),
                        grooming_pose_safe_identity_refine_source_sha256=(
                            recovery_source_sha256
                            if pose_safe_candidate_used_as_sampler
                            else None
                        ),
                    )
                if canonical_eating_contact_repair_active:
                    assert isinstance(eating_contact_repair_source, dict)
                    repair_source_attempt = int(
                        eating_contact_repair_source["attempt"]
                    )
                    repair_source_path = Path(
                        str(eating_contact_repair_source["cutout_path"])
                    )
                    expected_repair_source_sha256 = str(
                        eating_contact_repair_source["cutout_sha256"]
                    )
                    if (
                        not repair_source_path.is_file()
                        or file_sha256(repair_source_path)
                        != expected_repair_source_sha256
                    ):
                        raise RuntimeError(
                            "Canonical eating contact-repair source disappeared "
                            "or changed."
                        )
                    contact_geometry_input_name = attempt_secondary_input_name
                    if not contact_geometry_input_name:
                        raise RuntimeError(
                            "Canonical eating contact repair lacks its white-model "
                            "action guide."
                        )
                    eating_contact_repair_canvas_path = (
                        comfy_input_dir
                        / (
                            "eating_contact_repair_source_attempt_"
                            f"{repair_source_attempt:02d}.png"
                        )
                    )
                    eating_contact_repair_noise_mask_path = (
                        comfy_input_dir
                        / (
                            "eating_contact_repair_noise_mask_attempt_"
                            f"{repair_source_attempt:02d}.png"
                        )
                    )
                    repair_canvas_audit = (
                        prepare_canonical_eating_contact_repair_canvas(
                            repair_source_path,
                            eating_contact_repair_canvas_path,
                            eating_contact_repair_noise_mask_path,
                            contact_quality=(
                                eating_contact_repair_source["contact_quality"]
                            ),
                        )
                    )
                    attempt_primary_input_name = (
                        comfy_relative_dir
                        / eating_contact_repair_canvas_path.name
                    ).as_posix()
                    attempt_secondary_input_name = qwen_input_name
                    attempt_tertiary_input_name = contact_geometry_input_name
                    attempt_additional_reference_input_names = [
                        attempt_secondary_input_name,
                        attempt_tertiary_input_name,
                    ]
                    attempt_noise_mask_input_name = (
                        comfy_relative_dir
                        / eating_contact_repair_noise_mask_path.name
                    ).as_posix()
                    attempt_geometry_input_name = attempt_tertiary_input_name
                    attempt_prompt = build_canonical_eating_contact_repair_prompt(
                        pose,
                        source_attempt=repair_source_attempt,
                        source_gap_px=int(
                            eating_contact_repair_source["gap_px"]
                        ),
                        prealignment_shift_px=int(
                            repair_canvas_audit["prealignment_shift_px"]
                        ),
                        prealignment_expected_seed_gap_px=int(
                            repair_canvas_audit[
                                "prealignment_expected_seed_gap_px"
                            ]
                        ),
                        profile_fur_lock=profile_fur_lock,
                    )
                    eating_contact_repair_audit = {
                        "policy": EATING_CONTACT_REPAIR_POLICY,
                        "attempt": attempt,
                        "source_attempt": repair_source_attempt,
                        "source_path": str(repair_source_path),
                        "source_sha256": expected_repair_source_sha256,
                        "source_gap_px": int(
                            eating_contact_repair_source["gap_px"]
                        ),
                        "source_contact_quality": (
                            eating_contact_repair_source["contact_quality"]
                        ),
                        "source_identity_palette": (
                            eating_contact_repair_source[
                                "identity_palette_quality"
                            ]
                        ),
                        "repair_canvas": repair_canvas_audit,
                        "image1_role": (
                            "gap_measured_head_neck_prealigned_local_sampler"
                        ),
                        "image2_role": (
                            "approved_canonical_identity_appearance_corroboration"
                        ),
                        "image3_role": (
                            "white_maquette_eating_contact_geometry_only"
                        ),
                        "sampler_noise_mask": {
                            "input_name": attempt_noise_mask_input_name,
                            "path": str(
                                eating_contact_repair_noise_mask_path
                            ),
                            "sha256": repair_canvas_audit[
                                "noise_mask_target_sha256"
                            ],
                            "policy": (
                                "lower_muzzle_to_fixed_bowl_seam_v3"
                            ),
                        },
                        "full_frame_sampling": False,
                        "post_qwen_pixel_lock_required": True,
                        "upper_head_body_bowl_paws_tail_lock_required": True,
                        "identity_gate_unchanged": True,
                        "eating_contact_gate_unchanged": True,
                        "downstream_scail_prop_mask_thresholds_unchanged": True,
                    }
                    update_job(
                        job_id,
                        eating_contact_repair_active=True,
                        eating_contact_repair_policy=(
                            EATING_CONTACT_REPAIR_POLICY
                        ),
                        eating_contact_repair_attempt=attempt,
                        eating_contact_repair_source_attempt=(
                            repair_source_attempt
                        ),
                        eating_contact_repair_source_gap_px=int(
                            eating_contact_repair_source["gap_px"]
                        ),
                        eating_contact_repair_prealignment_shift_px=int(
                            repair_canvas_audit["prealignment_shift_px"]
                        ),
                        eating_contact_repair_expected_seed_gap_px=int(
                            repair_canvas_audit[
                                "prealignment_expected_seed_gap_px"
                            ]
                        ),
                        eating_contact_repair_source_sha256=(
                            expected_repair_source_sha256
                        ),
                        eating_contact_repair_noise_mask_sha256=(
                            repair_canvas_audit["noise_mask_target_sha256"]
                        ),
                    )
                elif canonical_eating_identity_refine_active:
                    assert isinstance(eating_identity_refine_source, dict)
                    refine_source_path = Path(
                        str(eating_identity_refine_source["cutout_path"])
                    )
                    expected_refine_source_sha256 = str(
                        eating_identity_refine_source["cutout_sha256"]
                    )
                    if not refine_source_path.is_file():
                        raise RuntimeError(
                            "Canonical eating identity-refine source disappeared."
                        )
                    if file_sha256(refine_source_path) != expected_refine_source_sha256:
                        raise RuntimeError(
                            "Canonical eating identity-refine source digest changed."
                        )
                    if not attempt_secondary_input_name:
                        raise RuntimeError(
                            "Canonical eating identity refine lacks its action guide."
                        )
                    eating_refine_canvas_path = (
                        comfy_input_dir / "eating_identity_refine_source.png"
                    )
                    eating_refine_canvas_audit = (
                        prepare_canonical_eating_identity_refine_canvas(
                            refine_source_path,
                            eating_refine_canvas_path,
                        )
                    )
                    attempt_primary_input_name = (
                        comfy_relative_dir / eating_refine_canvas_path.name
                    ).as_posix()
                    attempt_secondary_input_name = qwen_input_name
                    attempt_tertiary_input_name = uploaded_qwen_input_name
                    if not attempt_tertiary_input_name:
                        raise RuntimeError(
                            "Canonical eating identity refine lacks uploaded front identity."
                        )
                    attempt_additional_reference_input_names = [
                        attempt_secondary_input_name,
                        attempt_tertiary_input_name,
                    ]
                    attempt_noise_mask_input_name = None
                    attempt_prompt = build_canonical_eating_identity_refine_prompt(
                        pose,
                        source_attempt=int(eating_identity_refine_source["attempt"]),
                        profile_fur_lock=profile_fur_lock,
                    )
                    if pose_identity_palette_prompt_signature is not None:
                        attempt_prompt += "\n\n" + identity_palette_prompt(
                            pose_identity_palette_prompt_signature,
                            retry=True,
                            previous_quality=eating_identity_refine_source.get(
                                "identity_palette_quality"
                            ),
                        )
                    eating_identity_refine_audit = {
                        "policy": EATING_CANONICAL_IDENTITY_REFINE_POLICY,
                        "attempt": attempt,
                        "source_attempt": int(eating_identity_refine_source["attempt"]),
                        "source_path": str(refine_source_path),
                        "source_sha256": expected_refine_source_sha256,
                        "source_pose_quality": (
                            eating_identity_refine_source["pose_quality"]
                        ),
                        "source_identity_palette": (
                            eating_identity_refine_source["identity_palette_quality"]
                        ),
                        "source_stripe_amplification_refine_eligibility": (
                            eating_identity_refine_source.get(
                                "stripe_amplification_refine_eligibility"
                            )
                        ),
                        "source_conditionally_repairable_palette_issues": (
                            eating_identity_refine_source.get(
                                "conditionally_repairable_palette_issues",
                                [],
                            )
                        ),
                        "refine_canvas": eating_refine_canvas_audit,
                        "image1_role": (
                            "approved_eating_pose_and_bowl_sampler_no_identity_authority"
                        ),
                        "image2_role": (
                            "approved_canonical_sole_pet_identity_appearance_authority"
                        ),
                        "image3_role": (
                            "uploaded_front_real_identity_palette_corroboration"
                        ),
                        "sampler_noise_mask": None,
                        "full_frame_sampling": True,
                        "qwen_lightning_only": True,
                        "style_lora": None,
                        "identity_gate_unchanged": True,
                        "eating_pose_gate_unchanged": True,
                        "fur_surface_gate_unchanged": True,
                    }
                    update_job(
                        job_id,
                        eating_identity_refine_active=True,
                        eating_identity_refine_policy=(
                            EATING_CANONICAL_IDENTITY_REFINE_POLICY
                        ),
                        eating_identity_refine_attempt=attempt,
                        eating_identity_refine_source_attempt=(
                            eating_identity_refine_source["attempt"]
                        ),
                        eating_identity_refine_source_sha256=(
                            expected_refine_source_sha256
                        ),
                        eating_identity_refine_canvas_sha256=(
                            eating_refine_canvas_audit["target_sha256"]
                        ),
                    )
                if (
                    advisory_grooming_action_transfer_active
                    and not advisory_grooming_identity_refine_active
                ):
                    # Keep every retry on the exact prompt verified across four
                    # independent seeds. Appending issue prose previously grew
                    # the edit instruction past 1,300 words and destabilized it.
                    pass
                elif (
                    advisory_grooming_identity_refine_active
                    and previous_advisory_grooming_minimum_issues
                ):
                    attempt_prompt += (
                        "\n\nGROOMING IDENTITY RESTORATION RETRY: keep Image 1's "
                        "accepted action geometry, but correct every reported issue "
                        "using Image 2 as the sole whole-pet identity and Image 3 as "
                        "the strongest face authority. Never reintroduce a white-model "
                        "face, generic pointed head, plastic surface or extra limb. "
                        "Previous issues: "
                        + ", ".join(previous_advisory_grooming_minimum_issues)
                        + "."
                    )
                pose_user_loras_disabled = bool(
                    main_case_geometry_first
                    or grooming_geometry_canonical
                    or canonical_action_reference
                )
                effective_pose_denoise = float(profile["denoise"])
                if pose_reference_input_name is not None:
                    configured_pose_denoise = job.get("pose_reference_denoise")
                    if configured_pose_denoise is not None:
                        configured_pose_denoise_step = float(
                            job.get("pose_reference_denoise_step") or 0.0
                        )
                        effective_pose_denoise = max(
                            0.1,
                            min(
                                1.0,
                                float(configured_pose_denoise)
                                + configured_pose_denoise_step * (attempt - 1),
                            ),
                        )
                if main_case_geometry_first:
                    effective_pose_denoise = (
                        QWEN_EATING_GEOMETRY_FIRST_DENOISE_HIGH
                        if attempt in {1, 3}
                        else QWEN_EATING_GEOMETRY_FIRST_DENOISE_LOW
                    )
                elif (
                    grooming_geometry_canonical
                    or (
                        canonical_action_reference
                        and action_key(pose) == "grooming"
                    )
                ):
                    effective_pose_denoise = (
                        QWEN_GROOMING_CANONICAL_GEOMETRY_DENOISE_BY_ATTEMPT[
                            min(
                                attempt - 1,
                                len(QWEN_GROOMING_CANONICAL_GEOMETRY_DENOISE_BY_ATTEMPT)
                                - 1,
                            )
                        ]
                    )
                if advisory_grooming_action_transfer_active:
                    effective_pose_denoise = (
                        GROOMING_ADVISORY_ACTION_TRANSFER_DENOISE_BY_ATTEMPT[
                            min(
                                attempt - 1,
                                len(
                                    GROOMING_ADVISORY_ACTION_TRANSFER_DENOISE_BY_ATTEMPT
                                )
                                - 1,
                            )
                        ]
                    )
                if dog_prompt_only_action_transfer_active:
                    effective_pose_denoise = (
                        DOG_EATING_PROMPT_ONLY_ACTION_TRANSFER_DENOISE
                        if dog_eating_prompt_only_action_transfer_active
                        else (
                            DOG_DRINKING_PROMPT_ONLY_ACTION_TRANSFER_DENOISE
                            if dog_drinking_prompt_only_action_transfer_active
                            else DOG_TOILETING_PROMPT_ONLY_ACTION_TRANSFER_DENOISE
                        )
                    )
                if advisory_grooming_identity_refine_active:
                    effective_pose_denoise = (
                        GROOMING_ADVISORY_IDENTITY_REFINE_DENOISE
                    )
                if grooming_foreleg_repair_active:
                    effective_pose_denoise = GROOMING_QWEN_FORELEG_REPAIR_DENOISE
                elif canonical_grooming_full_pose_recovery_active:
                    effective_pose_denoise = (
                        GROOMING_CANONICAL_FULL_POSE_REFINE_DENOISE
                        if isinstance(grooming_full_pose_recovery_audit, dict)
                        and grooming_full_pose_recovery_audit.get("mode")
                        == "pose_safe_identity_refine"
                        else GROOMING_CANONICAL_FULL_POSE_RESCUE_DENOISE
                    )
                elif canonical_eating_contact_repair_active:
                    effective_pose_denoise = EATING_CONTACT_REPAIR_DENOISE
                elif canonical_eating_identity_refine_active:
                    effective_pose_denoise = EATING_CANONICAL_IDENTITY_REFINE_DENOISE
                effective_pose_user_lora = (
                    None if pose_user_loras_disabled else profile["user_lora"]
                )
                effective_pose_user_lora_strength = (
                    0.0 if pose_user_loras_disabled else profile["user_lora_strength"]
                )
                effective_pose_user_lora_mode = (
                    "stack" if pose_user_loras_disabled else profile["user_lora_mode"]
                )
                effective_pose_user_loras = (
                    [] if pose_user_loras_disabled else profile["user_loras"]
                )
                geometry_style_recovery = bool(
                    main_case_geometry_first
                    and action_key(pose) == "eating"
                    and attempt >= 3
                    and QWEN_EATING_GEOMETRY_FIRST_STYLE_STRENGTH > 0.0
                )
                grooming_local_style_recovery = bool(
                    legacy_grooming_foreleg_repair_active
                    and GROOMING_QWEN_FORELEG_REPAIR_STYLE_STRENGTH > 0.0
                )
                builtin_style_recovery = bool(
                    geometry_style_recovery or grooming_local_style_recovery
                )
                effective_pose_steps = (
                    max(8, int(profile["steps"]))
                    if (
                        geometry_first_action_reference
                        and action_key(pose) == "grooming"
                    )
                    else int(profile["steps"])
                )
                if advisory_grooming_action_transfer_active:
                    # The loaded Lightning LoRA is distilled for exactly four
                    # steps. Longer schedules produced neutral copies and scene
                    # drift in the same-seed grooming reproduction.
                    effective_pose_steps = GROOMING_ADVISORY_ACTION_TRANSFER_STEPS
                if dog_prompt_only_action_transfer_active:
                    effective_pose_steps = (
                        DOG_EATING_PROMPT_ONLY_ACTION_TRANSFER_STEPS
                        if dog_eating_prompt_only_action_transfer_active
                        else (
                            DOG_DRINKING_PROMPT_ONLY_ACTION_TRANSFER_STEPS
                            if dog_drinking_prompt_only_action_transfer_active
                            else DOG_TOILETING_PROMPT_ONLY_ACTION_TRANSFER_STEPS
                        )
                    )
                if advisory_grooming_identity_refine_active:
                    effective_pose_steps = max(
                        GROOMING_ADVISORY_IDENTITY_REFINE_STEPS,
                        effective_pose_steps,
                    )
                if grooming_foreleg_repair_active:
                    effective_pose_steps = max(
                        GROOMING_QWEN_FORELEG_REPAIR_STEPS,
                        effective_pose_steps,
                    )
                if canonical_grooming_full_pose_recovery_active:
                    effective_pose_steps = max(
                        GROOMING_CANONICAL_FULL_POSE_RESCUE_STEPS,
                        effective_pose_steps,
                    )
                if canonical_eating_identity_refine_active:
                    effective_pose_steps = max(
                        EATING_CANONICAL_IDENTITY_REFINE_STEPS,
                        effective_pose_steps,
                    )
                if canonical_eating_contact_repair_active:
                    effective_pose_steps = max(
                        EATING_CONTACT_REPAIR_STEPS,
                        effective_pose_steps,
                    )
                effective_suppress_builtin_style = (
                    False
                    if grooming_local_style_recovery
                    else (
                        True
                        if (canonical_action_reference or grooming_geometry_canonical)
                        else (
                            not geometry_style_recovery
                            if main_case_geometry_first
                            else suppress_builtin_style
                        )
                    )
                )
                effective_builtin_style_strength = (
                    GROOMING_QWEN_FORELEG_REPAIR_STYLE_STRENGTH
                    if grooming_local_style_recovery
                    else (
                        QWEN_EATING_GEOMETRY_FIRST_STYLE_STRENGTH
                        if geometry_style_recovery
                        else QWEN_POSE_STYLE_LORA_STRENGTH
                    )
                )
                if (
                    dog_prompt_only_action_transfer_active
                    and pose_identity_palette_prompt_signature is not None
                    and "SOURCE COAT PALETTE LOCK" not in attempt_prompt
                ):
                    # The specialized dog branches replace the generic prompt.
                    # Rebind the uploaded palette after that replacement.
                    attempt_prompt += "\n\n" + identity_palette_prompt(
                        pose_identity_palette_prompt_signature,
                        retry=attempt > 1,
                        previous_quality=previous_pose_identity_palette_quality,
                    )
                workflow = qwen_workflow(
                    attempt_primary_input_name,
                    attempt_prompt,
                    (
                        ""
                        if (
                            advisory_grooming_action_transfer_active
                            or dog_prompt_only_action_transfer_active
                        )
                        else pose_negative_prompt
                    ),
                    attempt_seed,
                    prefix,
                    user_lora=effective_pose_user_lora,
                    user_lora_strength=effective_pose_user_lora_strength,
                    user_lora_mode=effective_pose_user_lora_mode,
                    user_loras=effective_pose_user_loras,
                    steps=effective_pose_steps,
                    cfg=profile["cfg"],
                    denoise=effective_pose_denoise,
                    builtin_style_strength=effective_builtin_style_strength,
                    suppress_builtin_style=effective_suppress_builtin_style,
                    pose_reference_input_name=(
                        None
                        if attempt_additional_reference_input_names is not None
                        else attempt_secondary_input_name
                    ),
                    additional_reference_input_names=(
                        attempt_additional_reference_input_names
                    ),
                    noise_mask_input_name=attempt_noise_mask_input_name,
                )
                write_workflow(
                    job_dir,
                    f"qwen_pose_{index:02d}_attempt_{attempt:02d}.json",
                    workflow,
                )
                entry = queue_and_wait(
                    workflow,
                    f"{job_id} keyframe {index}/{len(poses)} attempt {attempt}",
                    timeout=7200,
                    job_id=job_id,
                )
                raw_qwen_source_output = first_output(
                    entry,
                    {".png", ".jpg", ".jpeg", ".webp"},
                    "Qwen keyframe",
                )
                raw_qwen_source_sha256 = file_sha256(raw_qwen_source_output)
                source_output = raw_qwen_source_output
                source_output_sha256 = file_sha256(source_output)
                raw_qwen_byte_preserved = False
                if canonical_eating_contact_repair_active:
                    if (
                        eating_contact_repair_canvas_path is None
                        or eating_contact_repair_noise_mask_path is None
                        or eating_contact_repair_audit is None
                    ):
                        raise RuntimeError(
                            "Canonical eating contact repair lost its pixel-lock "
                            "inputs."
                        )
                    locked_output = (
                        job_dir
                        / (
                            f"pose_{index:02d}_eating_contact_locked_"
                            f"attempt_{attempt:02d}.png"
                        )
                    )
                    repair_canvas_audit = eating_contact_repair_audit[
                        "repair_canvas"
                    ]
                    source_bowl_bbox = repair_canvas_audit.get(
                        "source_bowl_bbox"
                    )
                    if (
                        not isinstance(source_bowl_bbox, list)
                        or len(source_bowl_bbox) != 4
                    ):
                        raise RuntimeError(
                            "Canonical eating contact repair lost its bowl boundary."
                        )
                    pixel_lock_audit = (
                        composite_canonical_eating_contact_repair_output(
                            raw_qwen_source_output,
                            eating_contact_repair_canvas_path,
                            eating_contact_repair_noise_mask_path,
                            locked_output,
                            bowl_top=int(source_bowl_bbox[1]),
                        )
                    )
                    eating_contact_repair_audit[
                        "post_qwen_pixel_lock"
                    ] = pixel_lock_audit
                    source_output = locked_output
                    source_output_sha256 = pixel_lock_audit["target_sha256"]
                attempt_keyframe = (
                    job_dir / f"pose_{index:02d}_keyframe_attempt_{attempt:02d}.png"
                )
                if advisory_grooming_identity_refine_active:
                    raise RuntimeError(
                        "Post-Qwen grooming identity refinement is disabled: the "
                        "first frame must be an untouched raw Qwen action output."
                    )
                elif advisory_grooming_action_transfer_active:
                    if source_output.suffix.lower() != ".png":
                        raise RuntimeError(
                            "Raw Qwen grooming output must be a PNG before exact "
                            "byte-preserving selection."
                        )
                    shutil.copy2(source_output, attempt_keyframe)
                    if file_sha256(attempt_keyframe) != source_output_sha256:
                        raise RuntimeError(
                            "Raw Qwen grooming output changed while being selected."
                        )
                    raw_qwen_byte_preserved = True
                else:
                    with Image.open(source_output) as generated:
                        generated.convert("RGB").save(
                            attempt_keyframe, format="PNG", optimize=True
                        )
                candidate_identity_palette = (
                    identity_palette_signature(
                        attempt_keyframe,
                        exclude_bottom_fraction=pose_palette_bottom_exclusion,
                    )
                    if pose_reference_identity_palette is not None
                    else None
                )
                primary_palette_quality = (
                    identity_palette_quality(
                        pose_reference_identity_palette,
                        candidate_identity_palette,
                    )
                    if (
                        pose_reference_identity_palette is not None
                        and candidate_identity_palette is not None
                    )
                    else None
                )
                candidate_upper_body_identity_palette = (
                    identity_palette_signature(
                        attempt_keyframe,
                        exclude_bottom_fraction=(
                            BOWL_IDENTITY_PALETTE_UPPER_BODY_EXCLUSION
                        ),
                    )
                    if pose_reference_upper_body_identity_palette is not None
                    else None
                )
                upper_body_palette_quality = (
                    identity_palette_quality(
                        pose_reference_upper_body_identity_palette,
                        candidate_upper_body_identity_palette,
                    )
                    if (
                        pose_reference_upper_body_identity_palette is not None
                        and candidate_upper_body_identity_palette is not None
                    )
                    else None
                )
                palette_quality = (
                    bowl_aware_identity_palette_quality(
                        primary_palette_quality,
                        upper_body_palette_quality,
                    )
                    if primary_palette_quality is not None and bowl_action
                    else primary_palette_quality
                )
                attempt_cutout = (
                    job_dir / f"pose_{index:02d}_qwen_cutout_attempt_{attempt:02d}.png"
                )
                semantic_green_attempt = (
                    job_dir
                    / f"pose_{index:02d}_semantic_green_attempt_{attempt:02d}.png"
                )
                semantic_cutout_attempt = (
                    job_dir
                    / f"pose_{index:02d}_semantic_cutout_attempt_{attempt:02d}.png"
                )
                semantic_raw_attempt = (
                    job_dir / f"pose_{index:02d}_semantic_raw_attempt_{attempt:02d}.png"
                )
                chroma_metrics, attempt_cutout = normalize_qwen_action_candidate(
                    attempt_keyframe,
                    attempt_cutout,
                    semantic_green_attempt,
                    semantic_cutout_attempt,
                    semantic_raw_attempt,
                    pose=pose,
                    allow_semantic_matte=bool(
                        canonical_action_reference and not grooming_pose_gate_required
                    ),
                )
                if chroma_metrics.get("normalized_after_qwen"):
                    semantic_palette = evaluate_pose_identity_palette(
                        attempt_cutout,
                        primary_reference=pose_reference_identity_palette,
                        upper_body_reference=(
                            pose_reference_upper_body_identity_palette
                        ),
                        bottom_exclusion=pose_palette_bottom_exclusion,
                        bowl_action=bowl_action,
                    )
                    candidate_identity_palette = semantic_palette["candidate"]
                    candidate_upper_body_identity_palette = semantic_palette[
                        "upper_body_candidate"
                    ]
                    palette_quality = semantic_palette["quality"]
                grooming_pose_gate_metrics: dict[str, Any] | None = None
                if grooming_pose_gate_required:
                    if (
                        grooming_pose_gate_guide_path is None
                        or grooming_pose_gate_metadata_path is None
                    ):
                        raise RuntimeError(
                            "Main_case grooming pose gate inputs were not resolved."
                        )
                    direct_chroma_metrics = chroma_metrics
                    grooming_gate_green = (
                        job_dir
                        / f"pose_{index:02d}_grooming_gate_green_attempt_{attempt:02d}.png"
                    )
                    grooming_gate_cutout = (
                        job_dir
                        / f"pose_{index:02d}_grooming_gate_cutout_attempt_{attempt:02d}.png"
                    )
                    grooming_gate_semantic = (
                        job_dir
                        / f"pose_{index:02d}_grooming_gate_semantic_attempt_{attempt:02d}.png"
                    )
                    grooming_gate_reference = (
                        job_dir
                        / f"pose_{index:02d}_qwen_reference_attempt_{attempt:02d}.png"
                    )
                    chroma_metrics, grooming_pose_gate_metrics = (
                        evaluate_main_case_grooming_pose_candidate(
                            attempt_keyframe,
                            pose=pose,
                            green_target=grooming_gate_green,
                            cutout_target=grooming_gate_cutout,
                            semantic_target=grooming_gate_semantic,
                            normalized_cutout_target=grooming_gate_reference,
                            geometry_guide=grooming_pose_gate_guide_path,
                            geometry_metadata=grooming_pose_gate_metadata_path,
                            pose_transfer_seed=(
                                pose_transfer_seed_path
                                if canonical_grooming_full_pose_recovery_active
                                else None
                            ),
                        )
                    )
                    if (
                        canonical_action_reference
                        and attempt
                        <= GROOMING_CANONICAL_DIRECT_EDGE_EVIDENCE_MAX_ATTEMPT
                    ):
                        grooming_pose_gate_metrics = apply_canonical_grooming_direct_structural_edge_evidence_gate(
                            grooming_pose_gate_metrics,
                            attempt=attempt,
                            canonical_action_reference=(canonical_action_reference),
                            canonical_completion_active=(
                                canonical_grooming_completion_active
                            ),
                            canonical_full_pose_recovery_active=(
                                canonical_grooming_full_pose_recovery_active
                            ),
                        )
                    if canonical_grooming_completion_active:
                        grooming_pose_gate_metrics = (
                            apply_canonical_grooming_completion_edge_evidence_gate(
                                grooming_pose_gate_metrics,
                                grooming_foreleg_repair_audit,
                            )
                        )
                    chroma_metrics = {
                        **chroma_metrics,
                        "direct_chroma": direct_chroma_metrics,
                        "semantic_recovery_attempted": True,
                        "semantic_recovery_acceptable": bool(
                            chroma_metrics.get("acceptable")
                        ),
                        "normalized_after_qwen": bool(chroma_metrics.get("acceptable")),
                        "matte_policy": QWEN_ACTION_CANDIDATE_MATTE_POLICY,
                        "normalization_policy": (
                            QWEN_ACTION_REFERENCE_NORMALIZATION_POLICY
                        ),
                        "grooming_pose_gate_alpha_input": (
                            grooming_gate_reference.name
                        ),
                    }
                    attempt_cutout = (
                        grooming_gate_reference
                        if chroma_metrics.get("acceptable")
                        else grooming_gate_cutout
                    )
                    if chroma_metrics.get("acceptable"):
                        semantic_palette = evaluate_pose_identity_palette(
                            attempt_cutout,
                            primary_reference=pose_reference_identity_palette,
                            upper_body_reference=(
                                pose_reference_upper_body_identity_palette
                            ),
                            bottom_exclusion=pose_palette_bottom_exclusion,
                            bowl_action=bowl_action,
                        )
                        candidate_identity_palette = semantic_palette["candidate"]
                        candidate_upper_body_identity_palette = semantic_palette[
                            "upper_body_candidate"
                        ]
                        palette_quality = semantic_palette["quality"]
                    palette_normalization_eligibility = (
                        canonical_grooming_palette_normalization_eligibility(
                            canonical_action_identity_palette_quality(
                                palette_quality
                            ),
                            grooming_pose_gate_metrics,
                            attempt=attempt,
                        )
                    )
                    palette_normalization_audit: dict[str, Any] = {
                        "eligibility": palette_normalization_eligibility,
                        "applied": False,
                        "fail_closed": True,
                    }
                    if (
                        canonical_action_reference
                        and palette_normalization_eligibility.get("eligible") is True
                        and chroma_metrics.get("acceptable") is True
                        and attempt_cutout.is_file()
                    ):
                        palette_normalized_reference = (
                            job_dir
                            / f"pose_{index:02d}_grooming_palette_normalized_"
                            f"attempt_{attempt:02d}.png"
                        )
                        try:
                            canonical_reference_path = COMFY_INPUT / qwen_input_name
                            normalization = (
                                prepare_canonical_grooming_achromatic_palette_normalization(
                                    attempt_cutout,
                                    canonical_reference_path,
                                    palette_normalized_reference,
                                )
                            )
                            normalized_pose_gate = evaluate_grooming_pose(
                                palette_normalized_reference,
                                grooming_pose_gate_guide_path,
                                geometry_metadata=(
                                    grooming_pose_gate_metadata_path
                                ),
                            )
                            normalized_pose_gate = json.loads(
                                json.dumps(
                                    normalized_pose_gate,
                                    ensure_ascii=False,
                                    allow_nan=False,
                                )
                            )
                            normalized_pose_gate = apply_canonical_grooming_direct_structural_edge_evidence_gate(
                                normalized_pose_gate,
                                attempt=attempt,
                                canonical_action_reference=True,
                                canonical_completion_active=(
                                    canonical_grooming_completion_active
                                ),
                                canonical_full_pose_recovery_active=(
                                    canonical_grooming_full_pose_recovery_active
                                ),
                            )
                            normalized_palette = evaluate_pose_identity_palette(
                                palette_normalized_reference,
                                primary_reference=(
                                    pose_reference_identity_palette
                                ),
                                upper_body_reference=(
                                    pose_reference_upper_body_identity_palette
                                ),
                                bottom_exclusion=(pose_palette_bottom_exclusion),
                                bowl_action=False,
                            )
                            normalized_palette_quality = (
                                canonical_action_identity_palette_quality(
                                    normalized_palette["quality"]
                                )
                            )
                            revalidation_passed = bool(
                                normalized_pose_gate.get("acceptable") is True
                                and isinstance(
                                    normalized_palette_quality,
                                    dict,
                                )
                                and normalized_palette_quality.get("acceptable")
                                is True
                            )
                            palette_normalization_audit.update(
                                normalization=normalization,
                                revalidated_pose_gate=normalized_pose_gate,
                                revalidated_identity_palette=(
                                    normalized_palette_quality
                                ),
                                revalidation_passed=revalidation_passed,
                                applied=revalidation_passed,
                            )
                            if revalidation_passed:
                                attempt_cutout = palette_normalized_reference
                                grooming_gate_reference = (
                                    palette_normalized_reference
                                )
                                grooming_pose_gate_metrics = normalized_pose_gate
                                candidate_identity_palette = normalized_palette[
                                    "candidate"
                                ]
                                candidate_upper_body_identity_palette = (
                                    normalized_palette["upper_body_candidate"]
                                )
                                palette_quality = normalized_palette["quality"]
                                chroma_metrics[
                                    "pre_palette_final_reference_normalization"
                                ] = chroma_metrics.get(
                                    "final_reference_normalization"
                                )
                                chroma_metrics["final_reference_normalization"] = {
                                    "policy": (
                                        GROOMING_CANONICAL_DIRECT_ORGANIC_PAW_PALETTE_POLICY
                                    ),
                                    "source_path": normalization["source_path"],
                                    "source_sha256": normalization[
                                        "source_sha256"
                                    ],
                                    "output_path": normalization["target_path"],
                                    "output_sha256": normalization[
                                        "target_sha256"
                                    ],
                                    "output_mode": "RGBA",
                                    "output_size": [512, 512],
                                    "alpha_geometry_preserved_exactly": True,
                                }
                                chroma_metrics["final_reference_sha256"] = (
                                    file_sha256(attempt_cutout)
                                )
                                chroma_metrics[
                                    "grooming_pose_gate_alpha_input"
                                ] = attempt_cutout.name
                        except Exception as exc:  # noqa: BLE001 - retry fail closed
                            palette_normalization_audit.update(
                                applied=False,
                                revalidation_passed=False,
                                error=f"{type(exc).__name__}: {str(exc)[:1000]}",
                            )
                    chroma_metrics[
                        "grooming_achromatic_palette_normalization"
                    ] = palette_normalization_audit
                    update_job(
                        job_id,
                        grooming_achromatic_palette_normalization=(
                            palette_normalization_audit
                        ),
                    )
                    gate_audit = {
                        "pose_index": index,
                        "pose": pose,
                        "action": action_key(pose),
                        "attempt": attempt,
                        "seed": attempt_seed,
                        "source": attempt_keyframe.name,
                        "alpha_cutout": attempt_cutout.name,
                        "geometry_guide": str(grooming_pose_gate_guide_path),
                        "geometry_metadata": str(grooming_pose_gate_metadata_path),
                        "chroma": chroma_metrics,
                        "gate": grooming_pose_gate_metrics,
                        "result": (
                            "passed"
                            if grooming_pose_gate_metrics.get("acceptable")
                            else "rejected"
                        ),
                    }
                    grooming_pose_gate_attempt_audits.append(gate_audit)
                    write_json_atomic(
                        job_dir
                        / "workflows"
                        / f"qwen_pose_{index:02d}_grooming_pose_gate.json",
                        {
                            "policy": GROOMING_POSE_GATE_POLICY,
                            "version": GROOMING_POSE_GATE_VERSION,
                            "fail_closed": True,
                            "attempts": [
                                item
                                for item in grooming_pose_gate_attempt_audits
                                if item["pose_index"] == index
                            ],
                        },
                    )
                    update_job(
                        job_id,
                        grooming_pose_gate_status=(
                            "candidate_passed"
                            if grooming_pose_gate_metrics.get("acceptable")
                            else (
                                "failed"
                                if attempt == len(active_pose_profiles)
                                else "retrying"
                            )
                        ),
                        grooming_pose_gate_attempts=(grooming_pose_gate_attempt_audits),
                    )
                if (
                    canonical_action_reference
                    and chroma_metrics.get("acceptable")
                    and attempt_cutout.is_file()
                ):
                    try:
                        cleanup = neutralize_uploaded_identity_chroma_spill(
                            attempt_cutout,
                            source_identity_palette,
                        )
                        chroma_metrics = merge_identity_chroma_spill_cleanup(
                            chroma_metrics,
                            cleanup,
                            blocking=bool(action_candidate_audits_required),
                        )
                    except Exception as exc:  # noqa: BLE001 - retry fail closed
                        chroma_metrics = {
                            **chroma_metrics,
                            "acceptable": False,
                            "issues": list(
                                dict.fromkeys(
                                    [
                                        *[
                                            str(issue)
                                            for issue in chroma_metrics.get(
                                                "issues",
                                                [],
                                            )
                                        ],
                                        "identity_chroma_spill_cleanup_failed",
                                    ]
                                )
                            ),
                            "identity_chroma_spill_cleanup": {
                                "policy": UPLOADED_IDENTITY_CHROMA_SPILL_POLICY,
                                "applied": False,
                                "error": (f"{type(exc).__name__}: {str(exc)[:1000]}"),
                            },
                        }
                        if not action_candidate_audits_required:
                            chroma_metrics["warnings"] = list(
                                chroma_metrics.get("issues", [])
                            )
                            chroma_metrics["issues"] = []
                            chroma_metrics["acceptable"] = True
                    if chroma_metrics.get("acceptable"):
                        cleaned_palette = evaluate_pose_identity_palette(
                            attempt_cutout,
                            primary_reference=pose_reference_identity_palette,
                            upper_body_reference=(
                                pose_reference_upper_body_identity_palette
                            ),
                            bottom_exclusion=pose_palette_bottom_exclusion,
                            bowl_action=bowl_action,
                        )
                        candidate_identity_palette = cleaned_palette["candidate"]
                        candidate_upper_body_identity_palette = cleaned_palette[
                            "upper_body_candidate"
                        ]
                        palette_quality = cleaned_palette["quality"]
                if canonical_action_reference:
                    palette_quality = canonical_action_identity_palette_quality(
                        palette_quality
                    )
                eating_dark_feature_recovery_eligibility = (
                    light_achromatic_eating_dark_feature_recovery_eligibility(
                        palette_quality
                    )
                )
                eating_dark_feature_recovery_audit: dict[str, Any] = {
                    "eligibility": eating_dark_feature_recovery_eligibility,
                    "applied": False,
                    "identity_palette_gate_bypassed": False,
                    "fail_closed": True,
                }
                if (
                    canonical_action_reference
                    and action_key(pose) == "eating"
                    and eating_dark_feature_recovery_eligibility.get("eligible")
                    is True
                    and chroma_metrics.get("acceptable") is True
                    and attempt_cutout.is_file()
                    and canonical_similarity_reference is not None
                    and canonical_similarity_reference.is_file()
                ):
                    recovered_cutout = (
                        job_dir
                        / f"pose_{index:02d}_light_achromatic_identity_recovered_"
                        f"attempt_{attempt:02d}.png"
                    )
                    try:
                        before_recovery_pose_quality = pose_cutout_quality(
                            attempt_cutout,
                            pose,
                            require_eating_mouth_contact=(
                                require_eating_mouth_contact
                            ),
                        )
                        recovery = (
                            prepare_light_achromatic_eating_dark_feature_recovery(
                                attempt_cutout,
                                canonical_similarity_reference,
                                recovered_cutout,
                            )
                        )
                        recovered_palette = evaluate_pose_identity_palette(
                            recovered_cutout,
                            primary_reference=pose_reference_identity_palette,
                            upper_body_reference=(
                                pose_reference_upper_body_identity_palette
                            ),
                            bottom_exclusion=pose_palette_bottom_exclusion,
                            bowl_action=True,
                        )
                        recovered_palette_quality = (
                            canonical_action_identity_palette_quality(
                                recovered_palette["quality"]
                            )
                        )
                        after_recovery_pose_quality = pose_cutout_quality(
                            recovered_cutout,
                            pose,
                            require_eating_mouth_contact=(
                                require_eating_mouth_contact
                            ),
                        )
                        pose_gate_non_regressed = bool(
                            before_recovery_pose_quality.get("acceptable")
                            is not True
                            or after_recovery_pose_quality.get("acceptable")
                            is True
                        )
                        revalidation_passed = bool(
                            isinstance(recovered_palette_quality, dict)
                            and recovered_palette_quality.get("acceptable") is True
                            and pose_gate_non_regressed
                        )
                        eating_dark_feature_recovery_audit.update(
                            recovery=recovery,
                            before_identity_palette=palette_quality,
                            after_identity_palette=recovered_palette_quality,
                            before_pose_quality=before_recovery_pose_quality,
                            after_pose_quality=after_recovery_pose_quality,
                            pose_gate_non_regressed=pose_gate_non_regressed,
                            identity_palette_revalidated=bool(
                                isinstance(recovered_palette_quality, dict)
                                and recovered_palette_quality.get("acceptable")
                                is True
                            ),
                            revalidation_passed=revalidation_passed,
                            applied=revalidation_passed,
                        )
                        if revalidation_passed:
                            attempt_cutout = recovered_cutout
                            candidate_identity_palette = recovered_palette[
                                "candidate"
                            ]
                            candidate_upper_body_identity_palette = (
                                recovered_palette["upper_body_candidate"]
                            )
                            palette_quality = recovered_palette_quality
                            update_job(
                                job_id,
                                eating_light_achromatic_identity_recovery_status=(
                                    "candidate_recovered"
                                ),
                                eating_light_achromatic_identity_recovery_attempt=(
                                    attempt
                                ),
                                eating_light_achromatic_identity_recovery_policy=(
                                    EATING_LIGHT_ACHROMATIC_DARK_FEATURE_RECOVERY_POLICY
                                ),
                            )
                    except Exception as exc:  # noqa: BLE001 - retry fail closed
                        eating_dark_feature_recovery_audit.update(
                            error=f"{type(exc).__name__}: {str(exc)[:1000]}",
                            revalidation_passed=False,
                            applied=False,
                        )
                    chroma_metrics[
                        "eating_light_achromatic_dark_feature_recovery"
                    ] = eating_dark_feature_recovery_audit
                previous_pose_identity_palette_quality = (
                    json.loads(json.dumps(palette_quality))
                    if isinstance(palette_quality, dict)
                    else None
                )
                pre_surface_pose_quality: dict[str, Any] | None = None
                if (
                    canonical_action_reference
                    and chroma_metrics.get("acceptable")
                    and attempt_cutout.is_file()
                ):
                    pre_surface_pose_quality = pose_cutout_quality(
                        attempt_cutout,
                        pose,
                        require_eating_mouth_contact=(require_eating_mouth_contact),
                    )
                grooming_pose_safe_refine_surface_gate: dict[str, Any] | None = None
                grooming_pose_safe_refine_eligibility: dict[str, Any] | None = None
                if (
                    canonical_grooming_completion_capable
                    and attempt == GROOMING_CANONICAL_FULL_POSE_RESCUE_ATTEMPT
                ):
                    seed_bound_matte = chroma_metrics.get("seed_bound_matte")
                    palette_issue_set = (
                        {
                            str(issue)
                            for issue in palette_quality.get("issues", [])
                            if str(issue).strip()
                        }
                        if isinstance(palette_quality, dict)
                        else set()
                    )
                    pose_safe_surface_probe_ready = bool(
                        fur_surface_gate_required
                        and isinstance(pose_effective_subject_config, dict)
                        and chroma_metrics.get("acceptable") is True
                        and chroma_metrics.get("method")
                        == GROOMING_CANONICAL_SEED_BOUND_MATTE_ALPHA_PROVENANCE
                        and isinstance(seed_bound_matte, dict)
                        and seed_bound_matte.get("policy")
                        == GROOMING_CANONICAL_SEED_BOUND_MATTE_POLICY
                        and attempt_cutout.is_file()
                        and isinstance(grooming_pose_gate_metrics, dict)
                        and grooming_pose_gate_metrics.get("acceptable") is True
                        and isinstance(pre_surface_pose_quality, dict)
                        and pre_surface_pose_quality.get("acceptable") is True
                        and isinstance(palette_quality, dict)
                        and palette_quality.get("available") is True
                        and palette_quality.get("acceptable") is not True
                        and bool(palette_issue_set)
                        and not (
                            palette_issue_set
                            - GROOMING_CANONICAL_POSE_SAFE_IDENTITY_ONLY_ISSUES
                        )
                    )
                    if pose_safe_surface_probe_ready:
                        grooming_pose_safe_refine_surface_gate = (
                            evaluate_main_case_fur_surface_candidate(
                                attempt_cutout,
                                action="grooming",
                                alpha_provenance=(
                                    GROOMING_CANONICAL_SEED_BOUND_MATTE_ALPHA_PROVENANCE
                                ),
                                effective_subject_config=(
                                    pose_effective_subject_config
                                ),
                            )
                        )
                    grooming_pose_safe_refine_eligibility = (
                        canonical_grooming_pose_safe_identity_refine_eligibility(
                            attempt=attempt,
                            seed_bound_matte=(
                                seed_bound_matte
                                if isinstance(seed_bound_matte, dict)
                                else None
                            ),
                            pose_gate=grooming_pose_gate_metrics,
                            pose_quality=pre_surface_pose_quality,
                            identity_palette_quality=palette_quality,
                            fur_surface_gate=(
                                grooming_pose_safe_refine_surface_gate
                            ),
                        )
                    )
                    chroma_metrics["pose_safe_identity_refine_eligibility"] = (
                        grooming_pose_safe_refine_eligibility
                    )
                    update_job(
                        job_id,
                        grooming_pose_safe_identity_refine_probe={
                            "attempt": attempt,
                            "eligibility": grooming_pose_safe_refine_eligibility,
                            "independent_fur_surface_gate": (
                                grooming_pose_safe_refine_surface_gate
                            ),
                        },
                    )
                    if grooming_pose_safe_refine_eligibility.get("eligible") is True:
                        eligibility_sha256 = json_value_sha256(
                            grooming_pose_safe_refine_eligibility
                        )
                        assert isinstance(seed_bound_matte, dict)
                        grooming_full_pose_refine_source = {
                            "attempt": attempt,
                            "cutout_path": str(attempt_cutout),
                            "cutout_sha256": file_sha256(attempt_cutout),
                            "keyframe_path": str(attempt_keyframe),
                            "keyframe_sha256": file_sha256(attempt_keyframe),
                            "pose_quality": pre_surface_pose_quality,
                            "grooming_pose_gate": grooming_pose_gate_metrics,
                            "identity_palette_quality": palette_quality,
                            "identity_palette_acceptable": False,
                            "independent_fur_surface_gate": (
                                grooming_pose_safe_refine_surface_gate
                            ),
                            "seed_bound_matte": seed_bound_matte,
                            "eligibility": grooming_pose_safe_refine_eligibility,
                            "eligibility_sha256": eligibility_sha256,
                            "selection_key": [attempt],
                            "source_role": (
                                "digest_bound_seed_matte_pose_surface_approved_"
                                "identity_drift_candidate_for_attempt5_repaint"
                            ),
                        }
                        update_job(
                            job_id,
                            grooming_full_pose_refine_candidate=(
                                grooming_full_pose_refine_source
                            ),
                        )
                eating_refine_pose_issues = (
                    {
                        str(issue)
                        for issue in pre_surface_pose_quality.get("issues", [])
                        if str(issue).strip()
                    }
                    if isinstance(pre_surface_pose_quality, dict)
                    else set()
                )
                eating_refine_pose_safe = bool(
                    isinstance(pre_surface_pose_quality, dict)
                    and (
                        pre_surface_pose_quality.get("acceptable") is True
                        or (
                            bool(eating_refine_pose_issues)
                            and not (
                                eating_refine_pose_issues
                                - EATING_CANONICAL_IDENTITY_REFINE_POSE_SAFE_ISSUES
                            )
                        )
                    )
                )
                eating_refine_palette_issues = (
                    {
                        str(issue)
                        for issue in palette_quality.get("issues", [])
                        if str(issue).strip()
                    }
                    if isinstance(palette_quality, dict)
                    else set()
                )
                eating_refine_stripe_amplification_eligibility = (
                    identity_palette_stripe_amplification_refine_eligibility(
                        palette_quality
                    )
                )
                eating_refine_unrepairable_conditional_issues = sorted(
                    (
                        eating_refine_palette_issues
                        & EATING_CANONICAL_IDENTITY_REFINE_CONDITIONAL_PALETTE_ISSUES
                    )
                    if not eating_refine_stripe_amplification_eligibility.get(
                        "eligible"
                    )
                    else set()
                )
                eating_refine_unsafe_palette_issues = sorted(
                    (
                        eating_refine_palette_issues
                        & EATING_CANONICAL_IDENTITY_REFINE_UNSAFE_PALETTE_ISSUES
                    )
                    | set(eating_refine_unrepairable_conditional_issues)
                )
                if (
                    canonical_action_reference
                    and action_key(pose) == "eating"
                    # Attempt 4 can be the first candidate with strict mouth-to-food
                    # contact. Preserve it as geometry so attempt 5 can repair a
                    # conditionally safe identity drift instead of returning to a
                    # fresh full-frame pose sample.
                    and attempt < QWEN_ACTION_CANDIDATE_ATTEMPTS
                    and chroma_metrics.get("acceptable")
                    and attempt_cutout.is_file()
                    and eating_refine_pose_safe
                    and isinstance(palette_quality, dict)
                    and palette_quality.get("available") is True
                    and palette_quality.get("acceptable") is not True
                    and not eating_refine_unsafe_palette_issues
                ):
                    eating_refine_selection_key = (
                        float(palette_quality.get("penalty", 0.0)),
                        float(pre_surface_pose_quality.get("penalty", 0.0)),
                        -attempt,
                    )
                    if (
                        eating_identity_refine_source is None
                        or eating_refine_selection_key
                        < tuple(eating_identity_refine_source["selection_key"])
                    ):
                        eating_identity_refine_source = {
                            "attempt": attempt,
                            "cutout_path": str(attempt_cutout),
                            "cutout_sha256": file_sha256(attempt_cutout),
                            "keyframe_path": str(attempt_keyframe),
                            "keyframe_sha256": file_sha256(attempt_keyframe),
                            "pose_quality": pre_surface_pose_quality,
                            "pose_safe_allowed_issues": sorted(
                                eating_refine_pose_issues
                            ),
                            "unsafe_palette_issues": [],
                            "stripe_amplification_refine_eligibility": (
                                eating_refine_stripe_amplification_eligibility
                            ),
                            "conditionally_repairable_palette_issues": sorted(
                                eating_refine_palette_issues
                                & EATING_CANONICAL_IDENTITY_REFINE_CONDITIONAL_PALETTE_ISSUES
                            ),
                            "identity_palette_quality": palette_quality,
                            "selection_key": list(eating_refine_selection_key),
                            "source_role": (
                                "geometry_safe_eating_pose_and_bowl_for_identity_refine"
                            ),
                        }
                        update_job(
                            job_id,
                            eating_identity_refine_candidate=(
                                eating_identity_refine_source
                            ),
                        )
                fur_surface_gate_metrics: dict[str, Any] | None = None
                fur_surface_short_circuit_due = False
                fur_surface_retry_decision: dict[str, Any] | None = None
                if fur_surface_gate_required:
                    assert isinstance(pose_effective_subject_config, dict)
                    alpha_provenance = str(chroma_metrics.get("method") or "").strip()
                    surface_candidate_ready = bool(
                        chroma_metrics.get("acceptable")
                        and attempt_cutout.is_file()
                        and (
                            not canonical_action_reference
                            or (
                                isinstance(pre_surface_pose_quality, dict)
                                and pre_surface_pose_quality.get("acceptable") is True
                                and (
                                    not isinstance(palette_quality, dict)
                                    or palette_quality.get("acceptable") is True
                                )
                                and (
                                    not grooming_pose_gate_required
                                    or (
                                        isinstance(
                                            grooming_pose_gate_metrics,
                                            dict,
                                        )
                                        and grooming_pose_gate_metrics.get("acceptable")
                                        is True
                                    )
                                )
                            )
                        )
                    )
                    if surface_candidate_ready:
                        fur_surface_gate_metrics = (
                            evaluate_main_case_fur_surface_candidate(
                                attempt_cutout,
                                action=action_key(pose),
                                alpha_provenance=alpha_provenance,
                                effective_subject_config=(
                                    pose_effective_subject_config
                                ),
                            )
                        )
                        if (
                            action_key(pose) == "grooming"
                            and canonical_action_reference
                            and attempt
                            <= GROOMING_CANONICAL_DIRECT_EDGE_EVIDENCE_MAX_ATTEMPT
                            and fur_surface_gate_metrics.get("acceptable") is not True
                            and isinstance(grooming_pose_gate_metrics, dict)
                            and isinstance(palette_quality, dict)
                        ):
                            fur_surface_gate_metrics = apply_canonical_grooming_direct_surface_nonexpansion_gate(
                                fur_surface_gate_metrics,
                                pose_gate=grooming_pose_gate_metrics,
                                identity_palette_quality=palette_quality,
                                attempt=attempt,
                                canonical_reference_path=cartoon_output,
                                canonical_reference_sha256=(
                                    file_sha256(cartoon_output)
                                ),
                            )
                    else:
                        surface_gate_config = (
                            pose_effective_subject_config.get("actions", {})
                            .get(action_key(pose), {})
                            .get("fur_surface_gate", {})
                        )
                        baseline_config = (
                            surface_gate_config.get("baseline")
                            if isinstance(surface_gate_config, dict)
                            else None
                        )
                        fur_surface_gate_metrics = fur_surface_gate_failure(
                            action_key(pose),
                            (
                                "fur_surface_gate_not_evaluated_upstream_gate_failed"
                                if chroma_metrics.get("acceptable")
                                else "fur_surface_reliable_alpha_cutout_unavailable"
                            ),
                            baseline=(
                                baseline_config
                                if isinstance(baseline_config, dict)
                                else None
                            ),
                            candidate=attempt_cutout,
                            alpha_provenance=alpha_provenance or None,
                        )
                    if (
                        fur_surface_gate_metrics.get("evaluated") is True
                        and fur_surface_gate_metrics.get("acceptable") is not True
                    ):
                        fur_surface_rejection_streak += 1
                    elif fur_surface_gate_metrics.get("acceptable") is True:
                        fur_surface_rejection_streak = 0
                    else:
                        # Missing alpha is already rejected by the chroma path,
                        # but it is not evidence of three consecutive fur defects.
                        fur_surface_rejection_streak = 0
                    fur_surface_retry_decision = resolve_fur_surface_retry_decision(
                        action=action_key(pose),
                        rejection_streak=fur_surface_rejection_streak,
                        attempt=attempt,
                        active_profiles=active_pose_profiles,
                    )
                    fur_surface_short_circuit_due = bool(
                        fur_surface_retry_decision["short_circuit_due"]
                    )
                    surface_audit = {
                        "pose_index": index,
                        "pose": pose,
                        "action": action_key(pose),
                        "attempt": attempt,
                        "seed": attempt_seed,
                        "source": attempt_keyframe.name,
                        "alpha_cutout": (
                            attempt_cutout.name if attempt_cutout.is_file() else None
                        ),
                        "alpha_provenance": alpha_provenance or None,
                        "gate": fur_surface_gate_metrics,
                        "consecutive_rejections": (fur_surface_rejection_streak),
                        "retry_decision": fur_surface_retry_decision,
                        "result": (
                            "passed"
                            if fur_surface_gate_metrics.get("acceptable") is True
                            else (
                                "rejected"
                                if fur_surface_gate_metrics.get("evaluated") is True
                                else "blocked_upstream"
                            )
                        ),
                    }
                    fur_surface_gate_attempt_audits.append(surface_audit)
                    fur_surface_gate_status_by_action[action_key(pose)] = (
                        "candidate_passed"
                        if fur_surface_gate_metrics.get("acceptable") is True
                        else (
                            ("failed" if fur_surface_short_circuit_due else "retrying")
                            if fur_surface_gate_metrics.get("evaluated") is True
                            else "blocked_upstream"
                        )
                    )
                    write_json_atomic(
                        job_dir
                        / "workflows"
                        / f"qwen_pose_{index:02d}_fur_surface_gate.json",
                        {
                            "policy": FUR_SURFACE_GATE_POLICY,
                            "gate_version": FUR_SURFACE_GATE_VERSION,
                            "threshold_version": (FUR_SURFACE_THRESHOLD_VERSION),
                            "thresholds": fur_surface_thresholds_payload(),
                            "fail_closed": True,
                            "max_consecutive_rejections": (
                                FUR_SURFACE_GATE_MAX_REJECTIONS
                            ),
                            "retry_policy": FUR_SURFACE_GATE_RETRY_POLICY,
                            "preserves_configured_eating_bowl_recovery": True,
                            "configured_eating_bowl_recovery_attempts": (
                                QWEN_BOWL_POSE_EXTRA_ATTEMPTS
                            ),
                            "short_circuit": fur_surface_gate_short_circuit,
                            "attempts": [
                                item
                                for item in fur_surface_gate_attempt_audits
                                if item["pose_index"] == index
                            ],
                        },
                    )
                    update_job(
                        job_id,
                        fur_surface_gate_status=dict(fur_surface_gate_status_by_action),
                        fur_surface_gate_attempts=(fur_surface_gate_attempt_audits),
                    )
                if (
                    legacy_grooming_foreleg_repair_active
                    and grooming_pose_gate_required
                    and isinstance(grooming_pose_gate_metrics, dict)
                ):
                    grooming_pose_gate_metrics = (
                        apply_masked_pet3d_grooming_composite_gate(
                            grooming_pose_gate_metrics,
                            fur_surface_gate_metrics,
                            grooming_foreleg_repair_audit,
                        )
                    )
                    current_gate_audit = next(
                        (
                            item
                            for item in reversed(grooming_pose_gate_attempt_audits)
                            if item.get("pose_index") == index
                            and item.get("attempt") == attempt
                        ),
                        None,
                    )
                    if isinstance(current_gate_audit, dict):
                        current_gate_audit["gate"] = grooming_pose_gate_metrics
                        composite_recovery = grooming_pose_gate_metrics.get(
                            "composite_recovery"
                        )
                        current_gate_audit["result"] = (
                            "passed_composite"
                            if (
                                isinstance(composite_recovery, dict)
                                and composite_recovery.get("accepted") is True
                            )
                            else (
                                "passed"
                                if grooming_pose_gate_metrics.get("acceptable") is True
                                else "rejected"
                            )
                        )
                    write_json_atomic(
                        job_dir
                        / "workflows"
                        / f"qwen_pose_{index:02d}_grooming_pose_gate.json",
                        {
                            "policy": GROOMING_POSE_GATE_POLICY,
                            "version": GROOMING_POSE_GATE_VERSION,
                            "composite_recovery_policy": (
                                GROOMING_QWEN_FORELEG_REPAIR_COMPOSITE_GATE_POLICY
                            ),
                            "fail_closed": True,
                            "attempts": [
                                item
                                for item in grooming_pose_gate_attempt_audits
                                if item["pose_index"] == index
                            ],
                        },
                    )
                    update_job(
                        job_id,
                        grooming_pose_gate_status=(
                            "candidate_passed"
                            if grooming_pose_gate_metrics.get("acceptable") is True
                            else (
                                "failed"
                                if attempt == len(active_pose_profiles)
                                else "retrying"
                            )
                        ),
                        grooming_pose_gate_attempts=(grooming_pose_gate_attempt_audits),
                        grooming_pose_gate_composite_recovery=(
                            grooming_pose_gate_metrics.get("composite_recovery")
                        ),
                    )
                advisory_grooming_minimum_gate_metrics: dict[str, Any] | None = None
                if (
                    advisory_grooming_minimum_gate_required
                    and chroma_metrics.get("acceptable") is True
                    and attempt_cutout.is_file()
                ):
                    if (
                        not isinstance(
                            advisory_grooming_identity_palette_reference,
                            dict,
                        )
                        or pose_palette_reference_path is None
                        or grooming_identity_contract_reference_path is None
                        or grooming_pose_gate_guide_path is None
                        or grooming_pose_gate_metadata_path is None
                    ):
                        raise RuntimeError(
                            "Advisory grooming minimum gate lost its immutable "
                            "identity or geometry reference."
                        )
                    advisory_grooming_minimum_gate_metrics = (
                        evaluate_advisory_grooming_minimum_gate(
                            attempt_cutout,
                            identity_reference_palette=(
                                advisory_grooming_identity_palette_reference
                            ),
                            identity_palette_reference_path=(
                                pose_palette_reference_path
                            ),
                            identity_reference_path=(
                                grooming_identity_contract_reference_path
                            ),
                            geometry_guide=grooming_pose_gate_guide_path,
                            geometry_metadata=grooming_pose_gate_metadata_path,
                            topology_guide=qwen_grooming_guide_path,
                            attempt=attempt,
                        )
                    )
                    minimum_gate_issues = [
                        str(issue)
                        for issue in advisory_grooming_minimum_gate_metrics.get(
                            "issues",
                            [],
                        )
                    ]
                    action_retry_guidance = (
                        advisory_grooming_action_retry_guidance(
                            advisory_grooming_minimum_gate_metrics,
                            attempt=attempt,
                        )
                    )
                    action_retry_warnings = list(
                        action_retry_guidance["warnings"]
                    )
                    action_retry_requested = bool(
                        action_retry_guidance["requested"]
                    )
                    advisory_grooming_minimum_gate_metrics[
                        "action_retry_warnings"
                    ] = action_retry_warnings
                    advisory_grooming_minimum_gate_metrics[
                        "action_retry_requested"
                    ] = action_retry_requested
                    advisory_grooming_minimum_gate_metrics[
                        "action_retry_never_blocks_final_attempt"
                    ] = action_retry_guidance["never_blocks_final_attempt"]
                    advisory_grooming_minimum_gate_metrics[
                        "action_retry_final_attempt_fail_closed"
                    ] = action_retry_guidance["final_attempt_fail_closed"]
                    previous_advisory_grooming_minimum_issues = list(
                        dict.fromkeys(
                            [*minimum_gate_issues, *action_retry_warnings]
                        )
                    )
                    minimum_gate_audit = {
                        "pose_index": index,
                        "pose": pose,
                        "action": action_key(pose),
                        "attempt": attempt,
                        "seed": attempt_seed,
                        "gate": advisory_grooming_minimum_gate_metrics,
                        "result": (
                            "retry_requested"
                            if action_retry_requested
                            else (
                                "passed"
                                if advisory_grooming_minimum_gate_metrics.get(
                                    "acceptable"
                                )
                                is True
                                else "rejected"
                            )
                        ),
                    }
                    grooming_advisory_minimum_gate_attempts.append(
                        minimum_gate_audit
                    )
                    write_json_atomic(
                        job_dir
                        / "workflows"
                        / f"qwen_pose_{index:02d}_grooming_minimum_gate.json",
                        {
                            "policy": GROOMING_ADVISORY_MINIMUM_GATE_POLICY,
                            "version": GROOMING_ADVISORY_MINIMUM_GATE_VERSION,
                            "blocking_scope": (
                                "retry_and_candidate_ranking_only_nonterminal"
                            ),
                            "blocks_scail_on_exhaustion": False,
                            "attempts": [
                                item
                                for item in grooming_advisory_minimum_gate_attempts
                                if item["pose_index"] == index
                            ],
                        },
                    )
                    minimum_status = (
                        "candidate_retry_requested"
                        if action_retry_requested
                        else (
                            "candidate_passed"
                            if advisory_grooming_minimum_gate_metrics.get(
                                "acceptable"
                            )
                            is True
                            else (
                                "failed"
                                if attempt == len(active_pose_profiles)
                                else "retrying"
                            )
                        )
                    )
                    update_job(
                        job_id,
                        grooming_pose_gate_status=(
                            f"minimum_contract_{minimum_status}"
                        ),
                        grooming_advisory_minimum_gate_status=minimum_status,
                        grooming_advisory_minimum_gate_attempts=(
                            grooming_advisory_minimum_gate_attempts
                        ),
                    )
                record = {
                    "attempt": attempt,
                    "profile": profile["name"],
                    "seed": attempt_seed,
                    "identity_input_policy": (effective_pose_identity_input_policy),
                    "image1_input_name": attempt_primary_input_name,
                    "image2_input_name": attempt_secondary_input_name,
                    "image3_input_name": attempt_tertiary_input_name,
                    "sampler_latent_input_name": attempt_primary_input_name,
                    "sampler_noise_mask_input_name": (attempt_noise_mask_input_name),
                    "identity_input_name": pose_identity_input_name,
                    "geometry_input_name": (
                        attempt_tertiary_input_name
                        if (
                            canonical_eating_identity_refine_active
                            or canonical_eating_contact_repair_active
                        )
                        else (
                            (attempt_tertiary_input_name or attempt_primary_input_name)
                            if canonical_grooming_full_pose_recovery_active
                            else (
                                attempt_tertiary_input_name
                                if grooming_foreleg_repair_active
                                else attempt_geometry_input_name
                            )
                        )
                    ),
                    "grooming_advisory_action_transfer": (
                        grooming_advisory_action_transfer_attempt
                    ),
                    "dog_eating_prompt_only_action_transfer": (
                        dog_eating_prompt_only_action_transfer_attempt
                    ),
                    "dog_drinking_prompt_only_action_transfer": (
                        dog_drinking_prompt_only_action_transfer_attempt
                    ),
                    "dog_toileting_prompt_only_action_transfer": (
                        dog_toileting_prompt_only_action_transfer_attempt
                    ),
                    "grooming_advisory_identity_refine": (
                        grooming_advisory_identity_refine_audit
                    ),
                    "grooming_foreleg_repair": (grooming_foreleg_repair_audit),
                    "grooming_full_pose_recovery": (grooming_full_pose_recovery_audit),
                    "eating_identity_refine": eating_identity_refine_audit,
                    "eating_contact_repair": eating_contact_repair_audit,
                    "raw_qwen_source_path": str(raw_qwen_source_output),
                    "raw_qwen_source_sha256": raw_qwen_source_sha256,
                    "raw_qwen_byte_preserved": raw_qwen_byte_preserved,
                    "candidate_source_path": str(source_output),
                    "candidate_source_sha256": source_output_sha256,
                    "keyframe": attempt_keyframe.name,
                    "keyframe_sha256": file_sha256(attempt_keyframe),
                    "cutout": attempt_cutout.name
                    if chroma_metrics["acceptable"]
                    else None,
                    "chroma": chroma_metrics,
                    "quality": None,
                    "similarity": None,
                    "identity_palette": palette_quality,
                    "candidate_identity_palette": candidate_identity_palette,
                    "candidate_upper_body_identity_palette": (
                        candidate_upper_body_identity_palette
                    ),
                    "grooming_pose_gate": grooming_pose_gate_metrics,
                    "grooming_advisory_minimum_gate": (
                        advisory_grooming_minimum_gate_metrics
                    ),
                    "fur_surface_gate": fur_surface_gate_metrics,
                    "combined_penalty": None,
                    "settings": {
                        "steps": effective_pose_steps,
                        "cfg": profile["cfg"],
                        "denoise": effective_pose_denoise,
                        "user_lora": effective_pose_user_lora,
                        "user_lora_strength": effective_pose_user_lora_strength,
                        "user_loras": effective_pose_user_loras,
                        "user_lora_strength_scale": (
                            0.0
                            if pose_user_loras_disabled
                            else profile["user_lora_strength_scale"]
                        ),
                        "user_lora_mode": effective_pose_user_lora_mode,
                        "use_user_prompt": profile["use_user_prompt"],
                        "quality_retry": profile["quality_retry"],
                        "total_strength_cap": (
                            0.0
                            if pose_user_loras_disabled
                            else profile["total_strength_cap"]
                        ),
                        "per_lora_strength_cap": (
                            0.0
                            if pose_user_loras_disabled
                            else profile["per_lora_strength_cap"]
                        ),
                        "builtin_style_lora": (
                            QWEN_STYLE_LORA if builtin_style_recovery else None
                        ),
                        "builtin_style_strength": (
                            effective_builtin_style_strength
                            if builtin_style_recovery
                            else None
                        ),
                        "builtin_style_scope": (
                            "sampler_noise_mask_only"
                            if grooming_local_style_recovery
                            else ("full_sampler" if geometry_style_recovery else None)
                        ),
                        "suppress_builtin_style": (effective_suppress_builtin_style),
                        "identity_input_policy": (effective_pose_identity_input_policy),
                        "image1_input_name": attempt_primary_input_name,
                        "image2_input_name": attempt_secondary_input_name,
                        "image3_input_name": attempt_tertiary_input_name,
                        "sampler_latent_input_name": (attempt_primary_input_name),
                        "sampler_noise_mask_input_name": (
                            attempt_noise_mask_input_name
                        ),
                        "identity_input_name": pose_identity_input_name,
                        "geometry_input_name": (
                            attempt_tertiary_input_name
                            if canonical_eating_identity_refine_active
                            else (
                                (
                                    attempt_tertiary_input_name
                                    or attempt_primary_input_name
                                )
                                if canonical_grooming_full_pose_recovery_active
                                else (
                                    attempt_tertiary_input_name
                                    if grooming_foreleg_repair_active
                                    else attempt_geometry_input_name
                                )
                            )
                        ),
                        "grooming_foreleg_repair_policy": (
                            grooming_foreleg_repair_audit.get("policy")
                            if isinstance(grooming_foreleg_repair_audit, dict)
                            else None
                        ),
                        "grooming_advisory_identity_refine_policy": (
                            grooming_advisory_identity_refine_audit.get("policy")
                            if isinstance(
                                grooming_advisory_identity_refine_audit,
                                dict,
                            )
                            else None
                        ),
                        "grooming_full_pose_recovery_policy": (
                            grooming_full_pose_recovery_audit.get("policy")
                            if isinstance(
                                grooming_full_pose_recovery_audit,
                                dict,
                            )
                            else None
                        ),
                        "eating_identity_refine_policy": (
                            eating_identity_refine_audit.get("policy")
                            if isinstance(
                                eating_identity_refine_audit,
                                dict,
                            )
                            else None
                        ),
                    },
                    "workflow": workflow,
                }
                grooming_candidate_rejected = bool(
                    grooming_pose_gate_required
                    and (
                        not isinstance(grooming_pose_gate_metrics, dict)
                        or not grooming_pose_gate_metrics.get("acceptable")
                    )
                )
                fur_surface_candidate_rejected = bool(
                    fur_surface_gate_required
                    and isinstance(fur_surface_gate_metrics, dict)
                    and fur_surface_gate_metrics.get("evaluated") is True
                    and fur_surface_gate_metrics.get("acceptable") is not True
                )
                if grooming_candidate_rejected or fur_surface_candidate_rejected:
                    if chroma_metrics.get("acceptable") and attempt_cutout.is_file():
                        gate_quality = pre_surface_pose_quality or pose_cutout_quality(
                            attempt_cutout,
                            pose,
                            require_eating_mouth_contact=(require_eating_mouth_contact),
                        )
                        gate_similarity_path = attempt_cutout
                    else:
                        gate_quality = {
                            "available": False,
                            "acceptable": False,
                            "issues": ["grooming_pose_gate_alpha_cutout_unavailable"],
                            "penalty": 1000.0,
                        }
                        gate_similarity_path = attempt_keyframe
                    gate_similarity = pose_similarity_quality(
                        gate_similarity_path,
                        pose=pose,
                        canonical_path=canonical_similarity_reference,
                        previous_poses=accepted_pose_similarity_references,
                    )
                    record.update(
                        {
                            "quality": gate_quality,
                            "similarity": gate_similarity,
                            "transparent": bool(chroma_metrics.get("acceptable")),
                            "background_removal_failed": not bool(
                                chroma_metrics.get("acceptable")
                            ),
                            "background_removal_failure_reason": (
                                None
                                if chroma_metrics.get("acceptable")
                                else "candidate_gate_alpha_recovery_failed"
                            ),
                            "combined_penalty": round(
                                float(gate_quality.get("penalty", 0.0))
                                + float(gate_similarity.get("penalty", 0.0))
                                + float(
                                    grooming_pose_gate_metrics.get(
                                        "penalty",
                                        1000.0,
                                    )
                                    if isinstance(
                                        grooming_pose_gate_metrics,
                                        dict,
                                    )
                                    else (
                                        1000.0 if grooming_pose_gate_required else 0.0
                                    )
                                )
                                + (
                                    float(palette_quality.get("penalty", 0.0))
                                    if isinstance(palette_quality, dict)
                                    else 0.0
                                )
                                + (
                                    float(
                                        fur_surface_gate_metrics.get(
                                            "penalty",
                                            1000.0,
                                        )
                                    )
                                    if isinstance(
                                        fur_surface_gate_metrics,
                                        dict,
                                    )
                                    and fur_surface_gate_metrics.get("evaluated")
                                    is True
                                    else 0.0
                                ),
                                6,
                            ),
                        }
                    )
                    record["fur_silhouette"] = None
                    record["action_candidate_gate"] = {
                        "policy": GENERATION_ACTION_CANDIDATE_GATE_POLICY,
                        "required": action_candidate_audits_required,
                        "advisory_retry_gate": (
                            advisory_action_candidate_gate_required
                        ),
                        "acceptable": False,
                        "attempt": attempt,
                        "transparent_edge_quality": chroma_metrics.get(
                            "transparent_edge_quality"
                        ),
                        "grooming_pose_gate_rejected": grooming_candidate_rejected,
                        "fur_surface_gate_rejected": (
                            fur_surface_candidate_rejected
                        ),
                    }
                    if strict_first_frame_required:
                        strict_selection_key = (
                            strict_first_frame_best_effort_selection_key(
                                record,
                                action=action_key(pose),
                            )
                        )
                        record[
                            "strict_first_frame_best_effort_selection_key"
                        ] = list(strict_selection_key)
                        if (
                            best_strict_first_frame_attempt is None
                            or strict_selection_key
                            < tuple(
                                best_strict_first_frame_attempt[
                                    "strict_first_frame_best_effort_selection_key"
                                ]
                            )
                        ):
                            best_strict_first_frame_attempt = record
                    if (
                        (
                            grooming_geometry_canonical
                            or canonical_grooming_completion_capable
                        )
                        and attempt < GROOMING_QWEN_FORELEG_REPAIR_ATTEMPT
                        and chroma_metrics.get("acceptable")
                        and attempt_cutout.is_file()
                        and isinstance(palette_quality, dict)
                        and palette_quality.get("available") is True
                        and palette_quality.get("acceptable") is True
                    ):
                        repair_gate_issues = [
                            str(issue)
                            for issue in (
                                grooming_pose_gate_metrics.get("issues", [])
                                if isinstance(
                                    grooming_pose_gate_metrics,
                                    dict,
                                )
                                else []
                            )
                        ]
                        repair_fur_surface_issues = [
                            str(issue)
                            for issue in (
                                fur_surface_gate_metrics.get("issues", [])
                                if isinstance(
                                    fur_surface_gate_metrics,
                                    dict,
                                )
                                else []
                            )
                        ]
                        repair_selection_key = (
                            len(repair_gate_issues),
                            len(repair_fur_surface_issues),
                            float(gate_quality.get("penalty", 0.0)),
                            float(
                                palette_quality.get("penalty", 0.0)
                                if isinstance(palette_quality, dict)
                                else 0.0
                            ),
                        )
                        if (
                            grooming_foreleg_repair_source is None
                            or repair_selection_key
                            < tuple(grooming_foreleg_repair_source["selection_key"])
                        ):
                            grooming_foreleg_repair_source = {
                                "attempt": attempt,
                                "cutout_path": str(attempt_cutout),
                                "cutout_sha256": file_sha256(attempt_cutout),
                                "keyframe_path": str(attempt_keyframe),
                                "keyframe_sha256": file_sha256(attempt_keyframe),
                                "gate_issues": repair_gate_issues,
                                "fur_surface_issues": (repair_fur_surface_issues),
                                "identity_palette_quality": palette_quality,
                                "selection_key": list(repair_selection_key),
                            }
                            update_job(
                                job_id,
                                grooming_foreleg_repair_candidate={
                                    "policy": (
                                        GROOMING_CANONICAL_FORELEG_COMPLETION_POLICY
                                        if canonical_grooming_completion_capable
                                        else GROOMING_QWEN_FORELEG_REPAIR_POLICY
                                    ),
                                    **grooming_foreleg_repair_source,
                                },
                            )
                    attempts.append(
                        {
                            key: value
                            for key, value in record.items()
                            if key != "workflow"
                        }
                    )
                    append_recovery_event(
                        job_id,
                        {
                            "stage": "cartoon_keyframe",
                            "pose_index": index,
                            "attempt": attempt,
                            "reason": (
                                "fur_surface_gate_failed"
                                if fur_surface_candidate_rejected
                                else "grooming_pose_gate_failed"
                            ),
                            "strategy": (
                                "main_case_fur_surface_retry"
                                if fur_surface_candidate_rejected
                                else "main_case_grooming_pose_retry"
                            ),
                            "profile": profile["name"],
                            "result": "rejected",
                            "chroma": chroma_metrics,
                            "quality": gate_quality,
                            "similarity": gate_similarity,
                            "identity_palette": palette_quality,
                            "grooming_pose_gate": grooming_pose_gate_metrics,
                            "fur_surface_gate": fur_surface_gate_metrics,
                        },
                    )
                    LOGGER.warning(
                        "%s pose %s attempt %s failed Main_case candidate gate: grooming=%s fur_surface=%s",
                        job_id,
                        index,
                        attempt,
                        (
                            grooming_pose_gate_metrics.get("issues")
                            if isinstance(grooming_pose_gate_metrics, dict)
                            else ["grooming_pose_gate_result_unavailable"]
                        ),
                        (
                            fur_surface_gate_metrics.get("issues")
                            if isinstance(fur_surface_gate_metrics, dict)
                            else None
                        ),
                    )
                    if fur_surface_short_circuit_due:
                        assert isinstance(fur_surface_retry_decision, dict)
                        fur_surface_gate_short_circuit = {
                            "action": action_key(pose),
                            "pose_index": index,
                            "attempt": attempt,
                            "reason": "consecutive_fur_surface_rejections",
                            "rejections": fur_surface_rejection_streak,
                            "attempt_budget": len(active_pose_profiles),
                            "retry_decision": fur_surface_retry_decision,
                            "eating_fourth_attempt_skipped": bool(
                                action_key(pose) == "eating"
                                and len(active_pose_profiles) > attempt
                            ),
                            "blocks_scail": not strict_first_frame_required,
                            "bounded_candidate_generation_continues": (
                                strict_first_frame_required
                            ),
                        }
                        fur_surface_gate_status_by_action[action_key(pose)] = (
                            "retrying_bounded_candidates"
                            if strict_first_frame_required
                            else "failed"
                        )
                        update_job(
                            job_id,
                            fur_surface_gate_status=dict(
                                fur_surface_gate_status_by_action
                            ),
                            fur_surface_gate_attempts=(fur_surface_gate_attempt_audits),
                            fur_surface_gate_short_circuit=(
                                fur_surface_gate_short_circuit
                            ),
                            fur_surface_gate_final_quality={
                                "scope": "qwen_action_reference_pre_scail",
                                "status": (
                                    "retrying_bounded_candidates"
                                    if strict_first_frame_required
                                    else "failed"
                                ),
                                "actions": dict(fur_surface_gate_selected_metrics),
                                "failure": fur_surface_gate_short_circuit,
                                "final_video_frame_audit": False,
                            },
                        )
                        write_json_atomic(
                            job_dir
                            / "workflows"
                            / f"qwen_pose_{index:02d}_fur_surface_gate.json",
                            {
                                "policy": FUR_SURFACE_GATE_POLICY,
                                "gate_version": FUR_SURFACE_GATE_VERSION,
                                "threshold_version": (FUR_SURFACE_THRESHOLD_VERSION),
                                "thresholds": (fur_surface_thresholds_payload()),
                                "fail_closed": not strict_first_frame_required,
                                "max_consecutive_rejections": (
                                    FUR_SURFACE_GATE_MAX_REJECTIONS
                                ),
                                "retry_policy": (FUR_SURFACE_GATE_RETRY_POLICY),
                                "preserves_configured_eating_bowl_recovery": (True),
                                "configured_eating_bowl_recovery_attempts": (
                                    QWEN_BOWL_POSE_EXTRA_ATTEMPTS
                                ),
                                "short_circuit": (fur_surface_gate_short_circuit),
                                "attempts": [
                                    item
                                    for item in fur_surface_gate_attempt_audits
                                    if item["pose_index"] == index
                                ],
                            },
                        )
                        if not strict_first_frame_required:
                            raise RuntimeError(
                                "Main_case fur surface gate rejected "
                                f"{fur_surface_rejection_streak} consecutive "
                                f"{action_key(pose)} Qwen candidates after attempt "
                                f"{attempt} of {len(active_pose_profiles)}; "
                                "blocking SCAIL."
                            )
                        LOGGER.warning(
                            "%s kept bounded first-frame candidate generation active "
                            "after %s consecutive %s fur-surface rejections",
                            job_id,
                            fur_surface_rejection_streak,
                            action_key(pose),
                        )
                    continue
                if not chroma_metrics["acceptable"]:
                    raw_similarity = (
                        pose_similarity_quality(
                            attempt_keyframe,
                            pose=pose,
                            canonical_path=canonical_similarity_reference,
                            previous_poses=accepted_pose_similarity_references,
                        )
                        if GENERATION_VISUAL_AUDITS_BLOCKING
                        else skipped_visual_audit("raw_pose_similarity")
                    )
                    record.update(
                        {
                            "profile": f"{profile['name']}_raw_image",
                            "cutout": None,
                            "transparent": False,
                            "background_removal_failed": True,
                            "background_removal_error": (
                                "Generated image did not contain a clean removable green background."
                            ),
                            "background_removal_failure_reason": (
                                "unclean_chroma_background"
                            ),
                            "similarity": raw_similarity,
                            "combined_penalty": round(
                                float(raw_similarity["penalty"])
                                + (
                                    float(palette_quality["penalty"])
                                    if palette_quality is not None
                                    else 0.0
                                ),
                                6,
                            ),
                        }
                    )
                    attempts.append(
                        {
                            key: value
                            for key, value in record.items()
                            if key != "workflow"
                        }
                    )
                    semantic_recovery_not_attempted = not chroma_metrics.get(
                        "semantic_recovery_attempted",
                        False,
                    )
                    if strict_first_frame_required and semantic_recovery_not_attempted:
                        semantic_selection_key = (
                            strict_first_frame_best_effort_selection_key(
                                record,
                                action=action_key(pose),
                            )
                        )
                        record["strict_first_frame_semantic_selection_key"] = list(
                            semantic_selection_key
                        )
                        if (
                            best_semantic_source is None
                            or semantic_selection_key
                            < tuple(
                                best_semantic_source.get(
                                    "strict_first_frame_semantic_selection_key",
                                    [float("inf")],
                                )
                            )
                        ):
                            best_semantic_source = record
                    elif semantic_recovery_candidate_eligible(
                        record
                    ) and semantic_recovery_not_attempted:
                        if (
                            best_semantic_source is None
                            or record["combined_penalty"]
                            < best_semantic_source["combined_penalty"]
                        ):
                            best_semantic_source = record
                    if (
                        image_request
                        and raw_similarity["acceptable"]
                        and (palette_quality is None or palette_quality["acceptable"])
                    ):
                        if (
                            best_raw_attempt is None
                            or record["combined_penalty"]
                            < best_raw_attempt["combined_penalty"]
                        ):
                            best_raw_attempt = record
                        append_recovery_event(
                            job_id,
                            {
                                "stage": "cartoon_keyframe",
                                "pose_index": index,
                                "attempt": attempt,
                                "reason": "unclean_chroma_background",
                                "strategy": "retry_pose_before_raw_fallback",
                                "profile": profile["name"],
                                "result": "rejected",
                                "chroma": chroma_metrics,
                                "similarity": raw_similarity,
                                "identity_palette": palette_quality,
                            },
                        )
                        LOGGER.warning(
                            "%s pose %s attempt %s could not be reliably cut out; "
                            "retrying before an opaque-image fallback",
                            job_id,
                            index,
                            attempt,
                        )
                        continue
                    append_recovery_event(
                        job_id,
                        {
                            "stage": "cartoon_keyframe",
                            "pose_index": index,
                            "attempt": attempt,
                            "reason": "unclean_chroma_background",
                            "strategy": "safe_qwen_retry",
                            "profile": profile["name"],
                            "result": "rejected",
                            "chroma": chroma_metrics,
                            "similarity": raw_similarity,
                            "identity_palette": palette_quality,
                        },
                    )
                    LOGGER.warning(
                        "%s pose %s attempt %s (%s) failed chroma checks: %s",
                        job_id,
                        index,
                        attempt,
                        profile["name"],
                        chroma_metrics["issues"],
                    )
                    continue
                quality = (
                    pre_surface_pose_quality
                    or pose_cutout_quality(
                        attempt_cutout,
                        pose,
                        require_eating_mouth_contact=(
                            require_eating_mouth_contact
                        ),
                    )
                    if action_candidate_audits_required
                    else skipped_visual_audit("action_subject_quality")
                )
                similarity = (
                    pose_similarity_quality(
                        attempt_cutout,
                        pose=pose,
                        canonical_path=canonical_similarity_reference,
                        previous_poses=accepted_pose_similarity_references,
                    )
                    if action_candidate_audits_required
                    else skipped_visual_audit("action_pose_similarity")
                )
                fur_silhouette_quality = (
                    eating_fur_silhouette_quality(
                        attempt_cutout,
                        pose_fur_reference_path,
                    )
                    if (
                        action_candidate_audits_required
                        and pose_fur_reference_path is not None
                    )
                    else None
                )
                record["quality"] = quality
                record["similarity"] = similarity
                record["fur_silhouette"] = fur_silhouette_quality
                record["combined_penalty"] = round(
                    float(quality["penalty"])
                    + float(similarity["penalty"])
                    + (
                        float(palette_quality["penalty"])
                        if palette_quality is not None
                        else 0.0
                    )
                    + (
                        float(fur_silhouette_quality["penalty"])
                        if fur_silhouette_quality is not None
                        else 0.0
                    )
                    + (
                        float(fur_surface_gate_metrics.get("penalty", 1000.0))
                        if isinstance(fur_surface_gate_metrics, dict)
                        else (1000.0 if fur_surface_gate_required else 0.0)
                    )
                    + (
                        float(
                            advisory_grooming_minimum_gate_metrics.get(
                                "penalty",
                                1000.0,
                            )
                        )
                        if isinstance(
                            advisory_grooming_minimum_gate_metrics,
                            dict,
                        )
                        else (
                            1000.0
                            if advisory_grooming_minimum_gate_required
                            else 0.0
                        )
                    ),
                    6,
                )
                record["transparent"] = True
                eating_contact_quality = (
                    quality.get("eating_mouth_bowl_contact")
                    if isinstance(quality, dict)
                    else None
                )
                eating_contact_issues = {
                    str(issue)
                    for issue in quality.get("issues", [])
                    if str(issue).strip()
                }
                eating_contact_components = (
                    list(eating_contact_quality.get("component_candidates", []))
                    if isinstance(eating_contact_quality, dict)
                    and isinstance(
                        eating_contact_quality.get("component_candidates"),
                        list,
                    )
                    else []
                )
                if isinstance(eating_contact_quality, dict) and isinstance(
                    eating_contact_quality.get("selected_component"),
                    dict,
                ):
                    eating_contact_components.append(
                        eating_contact_quality["selected_component"]
                    )
                eating_contact_component_gaps = [
                    int(component["gap_px"])
                    for component in eating_contact_components
                    if isinstance(component, dict)
                    and isinstance(component.get("gap_px"), (int, float))
                    and 0
                    <= int(component["gap_px"])
                    <= EATING_CONTACT_REPAIR_SOURCE_GAP_MAX_PX
                    and not (
                        {
                            str(reason)
                            for reason in component.get(
                                "rejection_reasons",
                                [],
                            )
                            if str(reason).strip()
                        }
                        - {"gap_outside_contact_range"}
                    )
                ]
                eating_contact_gap_px = (
                    min(eating_contact_component_gaps)
                    if eating_contact_component_gaps
                    else None
                )
                if (
                    canonical_action_reference
                    and action_key(pose) == "eating"
                    and attempt < QWEN_ACTION_CANDIDATE_ATTEMPTS
                    and chroma_metrics.get("acceptable") is True
                    and attempt_cutout.is_file()
                    and eating_contact_issues
                    == EATING_CONTACT_REPAIR_SOURCE_ISSUES
                    and isinstance(eating_contact_quality, dict)
                    and eating_contact_quality.get("available") is True
                    and eating_contact_gap_px is not None
                    and eating_contact_gap_px > EATING_MUZZLE_BOWL_GAP_MAX_PX
                    and isinstance(palette_quality, dict)
                    and palette_quality.get("acceptable") is True
                    and (
                        fur_silhouette_quality is None
                        or fur_silhouette_quality.get("acceptable") is True
                    )
                ):
                    eating_contact_selection_key = (
                        int(eating_contact_gap_px),
                        float(record["combined_penalty"]),
                        -attempt,
                    )
                    if (
                        eating_contact_repair_source is None
                        or eating_contact_selection_key
                        < tuple(
                            eating_contact_repair_source["selection_key"]
                        )
                    ):
                        eating_contact_repair_source = {
                            "attempt": attempt,
                            "cutout_path": str(attempt_cutout),
                            "cutout_sha256": file_sha256(attempt_cutout),
                            "keyframe_path": str(attempt_keyframe),
                            "keyframe_sha256": file_sha256(attempt_keyframe),
                            "gap_px": int(eating_contact_gap_px),
                            "contact_quality": eating_contact_quality,
                            "identity_palette_quality": palette_quality,
                            "fur_silhouette_quality": fur_silhouette_quality,
                            "selection_key": list(
                                eating_contact_selection_key
                            ),
                            "source_role": (
                                "identity_safe_nearest_muzzle_to_bowl_candidate_"
                                "for_local_contact_repair"
                            ),
                        }
                        update_job(
                            job_id,
                            eating_contact_repair_candidate=(
                                eating_contact_repair_source
                            ),
                        )
                if (
                    image_request
                    and similarity["acceptable"]
                    and (palette_quality is None or palette_quality["acceptable"])
                    and not quality["acceptable"]
                ):
                    raw_record = {
                        **record,
                        "profile": f"{profile['name']}_raw_image",
                        "cutout": None,
                        "transparent": False,
                        "background_removal_failed": True,
                        "background_removal_error": (
                            "Background removal could not preserve a complete pose subject; "
                            "the original generated image was returned."
                        ),
                        "background_removal_failure_reason": (
                            "subject_quality_invalid"
                        ),
                    }
                    if (
                        best_raw_attempt is None
                        or raw_record["combined_penalty"]
                        < best_raw_attempt["combined_penalty"]
                    ):
                        best_raw_attempt = raw_record
                action_candidate_gate_acceptable = bool(
                    quality.get("acceptable") is True
                    and similarity.get("acceptable") is True
                    and (
                        not isinstance(
                            chroma_metrics.get("transparent_edge_quality"),
                            dict,
                        )
                        or chroma_metrics["transparent_edge_quality"].get(
                            "acceptable"
                        )
                        is True
                    )
                    and (
                        not pose_identity_palette_gate_required
                        or (
                            isinstance(palette_quality, dict)
                            and palette_quality.get("acceptable") is True
                        )
                    )
                    and (
                        fur_silhouette_quality is None
                        or fur_silhouette_quality.get("acceptable") is True
                    )
                    and (
                        not fur_surface_gate_required
                        or (
                            isinstance(fur_surface_gate_metrics, dict)
                            and fur_surface_gate_metrics.get("acceptable") is True
                        )
                    )
                    and (
                        not grooming_pose_gate_required
                        or (
                            isinstance(grooming_pose_gate_metrics, dict)
                            and grooming_pose_gate_metrics.get("acceptable") is True
                        )
                    )
                )
                record["action_candidate_gate"] = {
                    "policy": GENERATION_ACTION_CANDIDATE_GATE_POLICY,
                    "required": action_candidate_audits_required,
                    "advisory_retry_gate": (
                        advisory_action_candidate_gate_required
                    ),
                    "acceptable": action_candidate_gate_acceptable,
                    "attempt": attempt,
                    "transparent_edge_quality": chroma_metrics.get(
                        "transparent_edge_quality"
                    ),
                }
                if strict_first_frame_required:
                    strict_selection_key = (
                        strict_first_frame_best_effort_selection_key(
                            record,
                            action=action_key(pose),
                        )
                    )
                    record["strict_first_frame_best_effort_selection_key"] = list(
                        strict_selection_key
                    )
                    if (
                        best_strict_first_frame_attempt is None
                        or strict_selection_key
                        < tuple(
                            best_strict_first_frame_attempt[
                                "strict_first_frame_best_effort_selection_key"
                            ]
                        )
                    ):
                        best_strict_first_frame_attempt = record
                if advisory_action_candidate_gate_required:
                    degraded_selection_key = action_candidate_degraded_selection_key(
                        record,
                        action=action_key(pose),
                    )
                    record["action_candidate_degraded_selection_key"] = list(
                        degraded_selection_key
                    )
                    if (
                        best_action_candidate_degraded_attempt is None
                        or degraded_selection_key
                        < tuple(
                            best_action_candidate_degraded_attempt[
                                "action_candidate_degraded_selection_key"
                            ]
                        )
                    ):
                        best_action_candidate_degraded_attempt = record
                attempts.append(
                    {key: value for key, value in record.items() if key != "workflow"}
                )
                if (
                    advisory_grooming_minimum_gate_required
                    and chroma_metrics.get("acceptable") is True
                    and attempt_cutout.is_file()
                    and isinstance(
                        advisory_grooming_minimum_gate_metrics,
                        dict,
                    )
                ):
                    degraded_selection_key = (
                        advisory_grooming_degraded_candidate_selection_key(
                            record
                        )
                    )
                    record["grooming_advisory_degraded_selection_key"] = list(
                        degraded_selection_key
                    )
                    if (
                        best_advisory_grooming_degraded_attempt is None
                        or degraded_selection_key
                        < advisory_grooming_degraded_candidate_selection_key(
                            best_advisory_grooming_degraded_attempt
                        )
                    ):
                        best_advisory_grooming_degraded_attempt = record
                advisory_minimum_acceptable = bool(
                    not advisory_grooming_minimum_gate_required
                    or (
                        isinstance(
                            advisory_grooming_minimum_gate_metrics,
                            dict,
                        )
                        and advisory_grooming_minimum_gate_metrics.get("acceptable")
                        is True
                        and not advisory_grooming_minimum_gate_metrics.get(
                            "action_retry_requested",
                            False,
                        )
                    )
                )
                advisory_repair_palette_quality = (
                    advisory_grooming_minimum_gate_metrics.get(
                        "identity_palette"
                    )
                    if isinstance(
                        advisory_grooming_minimum_gate_metrics,
                        dict,
                    )
                    and isinstance(
                        advisory_grooming_minimum_gate_metrics.get(
                            "identity_palette"
                        ),
                        dict,
                    )
                    else palette_quality
                )
                if (
                    advisory_grooming_minimum_gate_required
                    and not advisory_minimum_acceptable
                    and not advisory_grooming_identity_refine_active
                    and attempt < GROOMING_QWEN_FORELEG_REPAIR_ATTEMPT
                    and chroma_metrics.get("acceptable")
                    and attempt_cutout.is_file()
                    and isinstance(advisory_repair_palette_quality, dict)
                    and advisory_repair_palette_quality.get("available") is True
                    and advisory_repair_palette_quality.get("acceptable") is True
                    and isinstance(
                        advisory_grooming_minimum_gate_metrics,
                        dict,
                    )
                    and advisory_grooming_candidate_safe_for_masked_repair(
                        advisory_grooming_minimum_gate_metrics
                    )
                ):
                    advisory_repair_issues = [
                        str(issue)
                        for issue in advisory_grooming_minimum_gate_metrics.get(
                            "issues",
                            [],
                        )
                    ]
                    advisory_repair_selection_key = (
                        len(advisory_repair_issues),
                        float(
                            advisory_grooming_minimum_gate_metrics.get(
                                "penalty",
                                1000.0,
                            )
                        ),
                        float(
                            advisory_repair_palette_quality.get("penalty", 0.0)
                        ),
                        attempt,
                    )
                    if (
                        grooming_foreleg_repair_source is None
                        or advisory_repair_selection_key
                        < tuple(grooming_foreleg_repair_source["selection_key"])
                    ):
                        grooming_foreleg_repair_source = {
                            "attempt": attempt,
                            "cutout_path": str(attempt_cutout),
                            "cutout_sha256": file_sha256(attempt_cutout),
                            "keyframe_path": str(attempt_keyframe),
                            "keyframe_sha256": file_sha256(attempt_keyframe),
                            "gate_issues": advisory_repair_issues,
                            "fur_surface_issues": [],
                            "identity_palette_quality": (
                                advisory_repair_palette_quality
                            ),
                            "selection_key": list(advisory_repair_selection_key),
                            "selection_policy": (
                                "advisory_minimum_identity_safe_candidate_for_"
                                "masked_foreleg_completion_v1"
                            ),
                            "masked_repair_unsafe_issues_absent": True,
                        }
                        update_job(
                            job_id,
                            grooming_foreleg_repair_candidate={
                                "policy": (
                                    GROOMING_CANONICAL_FORELEG_COMPLETION_POLICY
                                ),
                                **grooming_foreleg_repair_source,
                            },
                        )
                if (
                    isinstance(palette_quality, dict)
                    and palette_quality.get("acceptable") is True
                    and (
                        (
                            advisory_action_candidate_gate_required
                            and action_candidate_gate_acceptable
                        )
                        or (
                            not advisory_action_candidate_gate_required
                            and (
                                (
                                    not GENERATION_VISUAL_AUDITS_BLOCKING
                                    and advisory_minimum_acceptable
                                )
                                or (
                                    GENERATION_VISUAL_AUDITS_BLOCKING
                                    and (
                                        fur_silhouette_quality is None
                                        or fur_silhouette_quality["acceptable"]
                                    )
                                    and (
                                        not fur_surface_gate_required
                                        or (
                                            isinstance(
                                                fur_surface_gate_metrics,
                                                dict,
                                            )
                                            and fur_surface_gate_metrics.get(
                                                "acceptable"
                                            )
                                            is True
                                        )
                                    )
                                    and (
                                        not grooming_pose_gate_required
                                        or (
                                            isinstance(
                                                grooming_pose_gate_metrics,
                                                dict,
                                            )
                                            and grooming_pose_gate_metrics.get(
                                                "acceptable"
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                    and (
                        best_attempt is None
                        or record["combined_penalty"] < best_attempt["combined_penalty"]
                    )
                ):
                    best_attempt = record
                advisory_identity_refine_source_eligible = bool(
                    advisory_grooming_minimum_gate_required
                    and GROOMING_ADVISORY_IDENTITY_REFINE_MAX_ATTEMPTS > 0
                    and not advisory_grooming_identity_refine_active
                    and attempt
                    <= GROOMING_ADVISORY_ACTION_TRANSFER_MAX_ATTEMPTS
                    and (
                        advisory_minimum_acceptable
                        or attempt
                        == GROOMING_ADVISORY_ACTION_TRANSFER_MAX_ATTEMPTS
                    )
                    and chroma_metrics.get("acceptable") is True
                    and attempt_cutout.is_file()
                    and advisory_grooming_candidate_safe_for_identity_refine(
                        advisory_grooming_minimum_gate_metrics
                    )
                )
                candidate_acceptable = bool(
                    isinstance(palette_quality, dict)
                    and palette_quality.get("acceptable") is True
                    and (
                        (
                            advisory_action_candidate_gate_required
                            and action_candidate_gate_acceptable
                        )
                        or (
                            not advisory_action_candidate_gate_required
                            and not GENERATION_VISUAL_AUDITS_BLOCKING
                            and (
                                advisory_minimum_acceptable
                                or advisory_identity_refine_source_eligible
                            )
                        )
                        or (
                            GENERATION_VISUAL_AUDITS_BLOCKING
                            and action_candidate_gate_acceptable
                        )
                    )
                )
                if candidate_acceptable:
                    if (
                        advisory_grooming_minimum_gate_required
                        and not advisory_grooming_identity_refine_active
                        and advisory_identity_refine_source_eligible
                        and attempt < len(active_pose_profiles)
                        and attempt_cutout.is_file()
                    ):
                        grooming_advisory_identity_refine_source = {
                            "attempt": attempt,
                            "cutout_path": str(attempt_cutout),
                            "cutout_sha256": file_sha256(attempt_cutout),
                            "keyframe_path": str(attempt_keyframe),
                            "keyframe_sha256": file_sha256(attempt_keyframe),
                            "minimum_gate": (
                                advisory_grooming_minimum_gate_metrics
                            ),
                            "identity_palette_quality": palette_quality,
                            "minimum_gate_acceptable": (
                                advisory_minimum_acceptable
                            ),
                            "identity_refine_source_eligible": (
                                advisory_identity_refine_source_eligible
                            ),
                            "identity_refine_source_allowed_issues": sorted(
                                GROOMING_ADVISORY_IDENTITY_REFINE_SOURCE_ALLOWED_ISSUES
                            ),
                            "fallback_eligible": advisory_minimum_acceptable,
                            "selection_role": (
                                "action_safe_face_drift_candidate_for_"
                                "conditional_canonical_face_repair"
                            ),
                        }
                        update_job(
                            job_id,
                            stage="restoring_grooming_identity",
                            grooming_advisory_identity_refine_status="queued",
                            grooming_advisory_identity_refine_policy=(
                                GROOMING_ADVISORY_IDENTITY_REFINE_POLICY
                            ),
                            grooming_advisory_identity_refine_source=(
                                grooming_advisory_identity_refine_source
                            ),
                            grooming_advisory_identity_refine_required=True,
                            grooming_advisory_identity_refine_mode=(
                                "conditional_detected_face_drift_only"
                            ),
                        )
                        append_recovery_event(
                            job_id,
                            {
                                "stage": "cartoon_keyframe",
                                "pose_index": index,
                                "attempt": attempt,
                                "reason": (
                                    "grooming_action_draft_has_detected_repairable_"
                                    "face_drift"
                                ),
                                "strategy": (
                                    GROOMING_ADVISORY_IDENTITY_REFINE_POLICY
                                ),
                                "profile": profile["name"],
                                "result": "queued_identity_refine",
                                "grooming_advisory_minimum_gate": (
                                    advisory_grooming_minimum_gate_metrics
                                ),
                            },
                        )
                        continue
                    if advisory_action_candidate_gate_required:
                        update_job(
                            job_id,
                            action_candidate_gate_policy=(
                                GENERATION_ACTION_CANDIDATE_GATE_POLICY
                            ),
                            action_candidate_gate_status=(
                                "eligible_candidate_found"
                            ),
                            action_candidate_gate_action=action_key(pose),
                            action_candidate_gate_current_best_attempt=(
                                best_attempt["attempt"]
                                if best_attempt is not None
                                else attempt
                            ),
                            action_candidate_gate_degraded_fallback_used=False,
                        )
                    if advisory_grooming_identity_refine_active:
                        update_job(
                            job_id,
                            grooming_advisory_identity_refine_status="passed",
                            grooming_advisory_identity_refine_selected_attempt=(
                                attempt
                            ),
                            grooming_advisory_identity_refine_selected=(
                                grooming_advisory_identity_refine_audit
                            ),
                        )
                    if attempt > 1:
                        append_recovery_event(
                            job_id,
                            {
                                "stage": "cartoon_keyframe",
                                "pose_index": index,
                                "attempt": attempt,
                                "reason": (
                                    "candidate_added_to_best_of_five_pool"
                                ),
                                "strategy": "generate_five_then_select_best",
                                "profile": profile["name"],
                                "result": "eligible",
                                "chroma": chroma_metrics,
                                "quality": quality,
                                "similarity": similarity,
                                "identity_palette": palette_quality,
                                "fur_silhouette": fur_silhouette_quality,
                                "fur_surface_gate": fur_surface_gate_metrics,
                                "grooming_pose_gate": grooming_pose_gate_metrics,
                                "grooming_advisory_minimum_gate": (
                                    advisory_grooming_minimum_gate_metrics
                                ),
                            },
                        )
                    continue
                if (
                    advisory_grooming_minimum_gate_required
                    and not advisory_minimum_acceptable
                ):
                    rejection_reason = (
                        "grooming_minimum_action_identity_contract_failed"
                    )
                elif palette_quality is not None and not palette_quality["acceptable"]:
                    rejection_reason = "pose_identity_palette_drift"
                elif (
                    fur_silhouette_quality is not None
                    and not fur_silhouette_quality["acceptable"]
                ):
                    rejection_reason = "fur_silhouette_expansion"
                elif not quality["acceptable"]:
                    rejection_reason = "subject_quality_invalid"
                elif grooming_pose_gate_required and (
                    not isinstance(grooming_pose_gate_metrics, dict)
                    or not grooming_pose_gate_metrics.get("acceptable")
                ):
                    rejection_reason = "grooming_pose_gate_failed"
                elif (
                    fur_surface_gate_required
                    and isinstance(fur_surface_gate_metrics, dict)
                    and fur_surface_gate_metrics.get("evaluated") is True
                    and fur_surface_gate_metrics.get("acceptable") is not True
                ):
                    rejection_reason = "fur_surface_gate_failed"
                else:
                    rejection_reason = "pose_too_similar"
                append_recovery_event(
                    job_id,
                    {
                        "stage": "cartoon_keyframe",
                        "pose_index": index,
                        "attempt": attempt,
                        "reason": rejection_reason,
                        "strategy": "pose_prompt_and_lora_retry",
                        "profile": profile["name"],
                        "result": "rejected",
                        "chroma": chroma_metrics,
                        "quality": quality,
                        "similarity": similarity,
                        "identity_palette": palette_quality,
                        "fur_silhouette": fur_silhouette_quality,
                        "fur_surface_gate": fur_surface_gate_metrics,
                        "grooming_pose_gate": grooming_pose_gate_metrics,
                        "grooming_advisory_minimum_gate": (
                            advisory_grooming_minimum_gate_metrics
                        ),
                    },
                )
                LOGGER.warning(
                    "%s pose %s attempt %s failed pose checks: quality=%s similarity=%s identity_palette=%s fur_silhouette=%s fur_surface=%s minimum_gate=%s",
                    job_id,
                    index,
                    attempt,
                    quality["issues"],
                    similarity["issues"],
                    palette_quality["issues"] if palette_quality else None,
                    (
                        fur_silhouette_quality["issues"]
                        if fur_silhouette_quality
                        else None
                    ),
                    (
                        fur_surface_gate_metrics.get("issues")
                        if isinstance(fur_surface_gate_metrics, dict)
                        else None
                    ),
                    (
                        advisory_grooming_minimum_gate_metrics.get("issues")
                        if isinstance(
                            advisory_grooming_minimum_gate_metrics,
                            dict,
                        )
                        else None
                    ),
                )
                if advisory_grooming_identity_refine_active:
                    identity_refine_fallback_eligible = bool(
                        isinstance(grooming_advisory_identity_refine_source, dict)
                        and grooming_advisory_identity_refine_source.get(
                            "fallback_eligible"
                        )
                        is True
                    )
                    update_job(
                        job_id,
                        grooming_advisory_identity_refine_status=(
                            "failed_fallback_to_accepted_action_draft"
                            if identity_refine_fallback_eligible
                            else "failed_without_acceptable_action_draft_fallback"
                        ),
                        grooming_advisory_identity_refine_failed_attempt=attempt,
                        grooming_advisory_identity_refine_failure_issues=(
                            advisory_grooming_minimum_gate_metrics.get("issues", [])
                            if isinstance(
                                advisory_grooming_minimum_gate_metrics,
                                dict,
                            )
                            else ["identity_refine_candidate_unavailable"]
                        ),
                    )
                    if not identity_refine_fallback_eligible:
                        best_attempt = None
                    continue
            if len(attempts) != QWEN_ACTION_CANDIDATE_ATTEMPTS:
                raise RuntimeError(
                    "Action candidate generation did not complete all five attempts."
                )
            if strict_first_frame_required:
                first_frame_states = dict(
                    get_job(job_id).get("qwen_first_frame_states") or {}
                )
                action_state = dict(first_frame_states.get(action_key(pose)) or {})
                action_state.update(
                    status="selecting_best_of_five",
                    current_attempt=QWEN_ACTION_CANDIDATE_ATTEMPTS,
                    attempts_completed=QWEN_ACTION_CANDIDATE_ATTEMPTS,
                    attempt_budget=QWEN_ACTION_CANDIDATE_ATTEMPTS,
                    attempt_limit=QWEN_FIRST_FRAME_MAX_ATTEMPTS,
                )
                first_frame_states[action_key(pose)] = action_state
                update_job(
                    job_id,
                    qwen_first_frame_states=first_frame_states,
                    qwen_first_frame_current_action=action_key(pose),
                    qwen_first_frame_current_attempt=(
                        QWEN_ACTION_CANDIDATE_ATTEMPTS
                    ),
                )

            def fail_identity_budget_exhausted(
                additional_palette: dict[str, Any] | None = None,
            ) -> None:
                issue_counts: dict[str, int] = {}
                palette_records = [
                    item.get("identity_palette")
                    for item in attempts
                    if isinstance(item, dict)
                ]
                if isinstance(additional_palette, dict) and not any(
                    additional_palette == palette_record
                    for palette_record in palette_records
                    if isinstance(palette_record, dict)
                ):
                    palette_records.append(additional_palette)
                for palette_record in palette_records:
                    if not isinstance(palette_record, dict):
                        issue_counts["identity_palette_signature_unavailable"] = (
                            issue_counts.get(
                                "identity_palette_signature_unavailable",
                                0,
                            )
                            + 1
                        )
                        continue
                    for issue in palette_record.get("issues", []):
                        issue_name = str(issue).strip()
                        if issue_name:
                            issue_counts[issue_name] = (
                                issue_counts.get(issue_name, 0) + 1
                            )
                ranked_issues = sorted(
                    issue_counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )
                first_frame_states = dict(
                    get_job(job_id).get("qwen_first_frame_states") or {}
                )
                failed_state = dict(
                    first_frame_states.get(action_key(pose)) or {}
                )
                failed_state.update(
                    status="failed_identity_budget_exhausted",
                    completed=True,
                    quality_passed=False,
                    published=False,
                    current_attempt=QWEN_ACTION_CANDIDATE_ATTEMPTS,
                    attempts_completed=QWEN_ACTION_CANDIDATE_ATTEMPTS,
                    attempt_budget=QWEN_ACTION_CANDIDATE_ATTEMPTS,
                    attempt_limit=QWEN_FIRST_FRAME_MAX_ATTEMPTS,
                    selected_attempt=None,
                    best_effort=False,
                    best_effort_issues=[],
                    identity_issue_counts=dict(ranked_issues),
                    original_upload_fallback_published=False,
                    action_removed=False,
                )
                first_frame_states[action_key(pose)] = failed_state
                action_selection_records = get_job(job_id).get(
                    "action_candidate_selection_by_action"
                )
                action_selection_records = (
                    json.loads(json.dumps(action_selection_records))
                    if isinstance(action_selection_records, dict)
                    else {}
                )
                failure_record = {
                    "policy": GENERATION_ACTION_CANDIDATE_GATE_POLICY,
                    "source_kind": "qwen_generated_action_candidates",
                    "result": "failed_identity_budget_exhausted",
                    "attempt_budget": QWEN_ACTION_CANDIDATE_ATTEMPTS,
                    "attempts_completed": QWEN_ACTION_CANDIDATE_ATTEMPTS,
                    "selected_attempt": None,
                    "identity_palette_acceptable": False,
                    "identity_issue_counts": dict(ranked_issues),
                    "original_upload_used": False,
                    "approved_canonical_used_as_output": False,
                }
                action_selection_records[action_key(pose)] = failure_record
                write_json_atomic(
                    job_dir
                    / "workflows"
                    / f"qwen_pose_{index:02d}_identity_budget_failure.json",
                    failure_record,
                )
                update_job(
                    job_id,
                    qwen_first_frame_states=first_frame_states,
                    qwen_first_frame_best_effort_selected=False,
                    qwen_first_frame_best_effort_action=None,
                    qwen_first_frame_best_effort_attempt=None,
                    qwen_first_frame_best_effort_issues=[],
                    action_candidate_gate_status=(
                        "failed_identity_budget_exhausted"
                    ),
                    action_candidate_gate_action=action_key(pose),
                    action_candidate_gate_attempt_budget=(
                        QWEN_ACTION_CANDIDATE_ATTEMPTS
                    ),
                    action_candidate_gate_attempts_completed=(
                        QWEN_ACTION_CANDIDATE_ATTEMPTS
                    ),
                    action_candidate_gate_selected_attempt=None,
                    action_candidate_gate_returns_original_upload=False,
                    action_candidate_gate_returns_approved_canonical=False,
                    action_candidate_selection_by_action=(
                        action_selection_records
                    ),
                )
                append_recovery_event(
                    job_id,
                    {
                        "stage": "cartoon_keyframe",
                        "pose_index": index,
                        "reason": "five_candidate_identity_budget_exhausted",
                        "strategy": (
                            "fail_closed_without_original_or_canonical_output"
                        ),
                        "result": "rejected_identity_budget_exhausted",
                        "action": action_key(pose),
                        "attempt_budget": QWEN_ACTION_CANDIDATE_ATTEMPTS,
                        "attempts_completed": QWEN_ACTION_CANDIDATE_ATTEMPTS,
                        "identity_issue_counts": dict(ranked_issues),
                        "original_upload_fallback_published": False,
                        "approved_canonical_published": False,
                        "action_removed": False,
                    },
                )
                issue_summary = ", ".join(
                    f"{name}={count}" for name, count in ranked_issues
                ) or "identity_palette_signature_unavailable"
                raise RuntimeError(
                    "Best-of-five selection found no identity-safe generated "
                    f"candidate: {issue_summary}."
                )

            identity_safe_attempt_exists = any(
                isinstance(item.get("identity_palette"), dict)
                and item["identity_palette"].get("acceptable") is True
                for item in attempts
                if isinstance(item, dict)
            )
            if (
                strict_first_frame_required
                and not identity_safe_attempt_exists
                and best_semantic_source is None
                and not GENERATION_COMPLETION_FIRST_VISUAL_FALLBACK_ENABLED
            ):
                fail_identity_budget_exhausted()
            if (
                advisory_grooming_minimum_gate_required
                and best_attempt is not None
            ):
                selected_identity_refine = best_attempt.get(
                    "grooming_advisory_identity_refine"
                )
                identity_refine_was_queued = isinstance(
                    grooming_advisory_identity_refine_source,
                    dict,
                )
                update_job(
                    job_id,
                    grooming_advisory_identity_refine_status=(
                        "passed"
                        if isinstance(selected_identity_refine, dict)
                        else (
                            "fallback_to_accepted_action_draft"
                            if identity_refine_was_queued
                            else "not_required_identity_safe_draft_selected"
                        )
                    ),
                    grooming_advisory_identity_refine_completed=(
                        isinstance(selected_identity_refine, dict)
                    ),
                    grooming_advisory_identity_refine_fallback_used=(
                        identity_refine_was_queued
                        and not isinstance(selected_identity_refine, dict)
                    ),
                    grooming_advisory_identity_safe_draft_selected=(
                        not identity_refine_was_queued
                    ),
                )
            if GENERATION_VISUAL_AUDITS_BLOCKING and best_attempt is not None and (
                not best_attempt["quality"]["acceptable"]
                or not best_attempt["similarity"]["acceptable"]
                or (
                    isinstance(best_attempt.get("identity_palette"), dict)
                    and not best_attempt["identity_palette"]["acceptable"]
                )
                or (
                    fur_surface_gate_required
                    and (
                        not isinstance(
                            best_attempt.get("fur_surface_gate"),
                            dict,
                        )
                        or best_attempt["fur_surface_gate"].get("acceptable")
                        is not True
                    )
                )
                or (
                    grooming_pose_gate_required
                    and (
                        not isinstance(
                            best_attempt.get("grooming_pose_gate"),
                            dict,
                        )
                        or not best_attempt["grooming_pose_gate"].get("acceptable")
                    )
                )
            ):
                if (
                    fur_surface_gate_required
                    and isinstance(
                        best_attempt.get("fur_surface_gate"),
                        dict,
                    )
                    and best_attempt["fur_surface_gate"].get("evaluated") is True
                    and best_attempt["fur_surface_gate"].get("acceptable") is not True
                ):
                    rejection_reason = "all_keyframe_fur_surface_gate_attempts_rejected"
                elif grooming_pose_gate_required and (
                    not isinstance(
                        best_attempt.get("grooming_pose_gate"),
                        dict,
                    )
                    or not best_attempt["grooming_pose_gate"].get("acceptable")
                ):
                    rejection_reason = (
                        "all_keyframe_grooming_pose_gate_attempts_rejected"
                    )
                elif (
                    isinstance(best_attempt.get("identity_palette"), dict)
                    and not best_attempt["identity_palette"]["acceptable"]
                ):
                    rejection_reason = "all_keyframe_identity_palette_attempts_rejected"
                elif best_attempt["similarity"]["acceptable"]:
                    rejection_reason = "all_keyframe_subject_quality_attempts_rejected"
                else:
                    rejection_reason = "all_keyframe_similarity_attempts_rejected"
                recovery_strategy = (
                    "return_distinct_raw_image_or_fail"
                    if image_request
                    else (
                        "rembg_u2net_fallback"
                        if best_semantic_source is not None
                        else "fail_closed_without_background_recovery"
                    )
                )
                append_recovery_event(
                    job_id,
                    {
                        "stage": "cartoon_keyframe",
                        "pose_index": index,
                        "attempt": best_attempt["attempt"],
                        "reason": rejection_reason,
                        "strategy": recovery_strategy,
                        "profile": best_attempt["profile"],
                        "result": "rejected",
                        "chroma": best_attempt["chroma"],
                        "quality": best_attempt["quality"],
                        "similarity": best_attempt["similarity"],
                        "identity_palette": best_attempt.get("identity_palette"),
                    },
                )
                best_attempt = None
            if strict_first_frame_required and best_attempt is not None:
                strict_selection_issues = strict_first_frame_record_issues(
                    best_attempt,
                    action=action_key(pose),
                )
                if strict_selection_issues:
                    append_recovery_event(
                        job_id,
                        {
                            "stage": "cartoon_keyframe",
                            "pose_index": index,
                            "attempt": best_attempt.get("attempt"),
                            "reason": "selected_first_frame_failed_strict_publication_gate",
                            "strategy": QWEN_VALIDATED_FIRST_FRAME_POLICY,
                            "profile": best_attempt.get("profile"),
                            "result": "rejected_before_bounded_best_effort_selection",
                            "issues": strict_selection_issues,
                            "action": action_key(pose),
                        },
                    )
                    best_attempt = None
            if (
                best_attempt is None
                and strict_first_frame_required
                and best_strict_first_frame_attempt is not None
                and bool(best_strict_first_frame_attempt.get("cutout"))
                and (
                    job_dir / str(best_strict_first_frame_attempt["cutout"])
                ).is_file()
                and (
                    GENERATION_COMPLETION_FIRST_VISUAL_FALLBACK_ENABLED
                    or (
                        isinstance(
                            best_strict_first_frame_attempt.get(
                                "identity_palette"
                            ),
                            dict,
                        )
                        and best_strict_first_frame_attempt[
                            "identity_palette"
                        ].get("acceptable")
                        is True
                    )
                )
            ):
                best_attempt = best_strict_first_frame_attempt
                strict_best_effort_issues = strict_first_frame_record_issues(
                    best_attempt,
                    action=action_key(pose),
                )
                selected_candidate_gate = dict(
                    best_attempt.get("action_candidate_gate") or {}
                )
                selected_candidate_gate.update(
                    strict_acceptable=False,
                    best_effort_accepted=True,
                    degraded_fallback_accepted=True,
                    degraded_reason="bounded_first_frame_attempt_budget_exhausted",
                    degraded_selection_policy=(
                        QWEN_GENERATED_COMPLETION_SELECTION_POLICY
                    ),
                    degraded_selection_key=list(
                        best_attempt[
                            "strict_first_frame_best_effort_selection_key"
                        ]
                    ),
                    returns_original_upload=False,
                    action_removed=False,
                    blocks_job=False,
                )
                best_attempt["action_candidate_gate"] = selected_candidate_gate
                best_attempt["strict_first_frame_best_effort_selected"] = True
                best_attempt["strict_first_frame_best_effort_issues"] = (
                    strict_best_effort_issues
                )
                append_recovery_event(
                    job_id,
                    {
                        "stage": "cartoon_keyframe",
                        "pose_index": index,
                        "attempt": best_attempt["attempt"],
                        "reason": "bounded_first_frame_attempt_budget_exhausted",
                        "strategy": (
                            "select_generated_visual_completion_best_of_five"
                        ),
                        "profile": best_attempt["profile"],
                        "result": "accepted_best_effort",
                        "issues": strict_best_effort_issues,
                        "selection_key": best_attempt[
                            "strict_first_frame_best_effort_selection_key"
                        ],
                        "attempt_count": len(attempts),
                        "attempt_limit": QWEN_FIRST_FRAME_MAX_ATTEMPTS,
                        "original_upload_fallback_published": False,
                        "action_removed": False,
                    },
                )
                update_job(
                    job_id,
                    qwen_first_frame_best_effort_selected=True,
                    qwen_first_frame_best_effort_action=action_key(pose),
                    qwen_first_frame_best_effort_attempt=best_attempt["attempt"],
                    qwen_first_frame_best_effort_issues=(strict_best_effort_issues),
                    qwen_first_frame_attempt_limit=QWEN_FIRST_FRAME_MAX_ATTEMPTS,
                    visual_completion_fallback_used=True,
                    visual_completion_fallback_policy=(
                        GENERATION_COMPLETION_FIRST_VISUAL_FALLBACK_POLICY
                    ),
                )
            if (
                best_attempt is None
                and advisory_action_candidate_gate_required
                and not strict_first_frame_required
                and best_action_candidate_degraded_attempt is not None
            ):
                fallback_action = action_key(pose)
                approved_canonical_available = bool(
                    species == "dog"
                    and fallback_action
                    in {
                        "standing",
                        "eating",
                        "drinking",
                        "running",
                        "sleeping",
                        "toileting",
                    }
                    and cartoon_is_transparent
                    and cartoon_output.is_file()
                )
                exhausted_recovery = exhausted_action_candidate_recovery_policy(
                    best_action_candidate_degraded_attempt,
                    action=action_key(pose),
                    approved_canonical_available=approved_canonical_available,
                )
                if exhausted_recovery in {
                    "identity_safe_generated_candidate",
                    "identity_safe_generated_candidate_requires_final_action_gate",
                    "generated_completion_candidate",
                }:
                    best_attempt = best_action_candidate_degraded_attempt
                    requires_final_action_gate = bool(
                        exhausted_recovery
                        == "identity_safe_generated_candidate_requires_final_action_gate"
                    )
                    completion_candidate = bool(
                        exhausted_recovery == "generated_completion_candidate"
                    )
                    selected_candidate_gate = dict(
                        best_attempt.get("action_candidate_gate") or {}
                    )
                    selected_candidate_gate.update(
                        strict_acceptable=bool(
                            selected_candidate_gate.get("acceptable") is True
                        ),
                        degraded_fallback_accepted=True,
                        degraded_reason="candidate_attempt_budget_exhausted",
                        degraded_selection_policy=(
                            QWEN_GENERATED_COMPLETION_SELECTION_POLICY
                            if completion_candidate
                            else (
                                "identity_safe_structural_action_generated_"
                                "candidate_v3"
                            )
                        ),
                        degraded_selection_key=list(
                            best_attempt[
                                "action_candidate_degraded_selection_key"
                            ]
                        ),
                        returns_original_upload=False,
                        returns_approved_canonical=False,
                        blocks_job=False,
                        requires_final_action_gate=requires_final_action_gate,
                    )
                    best_attempt["action_candidate_gate"] = (
                        selected_candidate_gate
                    )
                    best_attempt["candidate_gate_degraded_fallback"] = True
                    append_recovery_event(
                        job_id,
                        {
                            "stage": "cartoon_keyframe",
                            "pose_index": index,
                            "attempt": best_attempt["attempt"],
                            "reason": "action_candidate_gate_attempts_exhausted",
                            "strategy": (
                                "select_generated_visual_completion_candidate"
                                if completion_candidate
                                else (
                                "select_identity_safe_structural_action_generated_"
                                "candidate_with_required_final_action_gate"
                                if requires_final_action_gate
                                else "select_identity_safe_structural_action_generated_"
                                "candidate"
                                )
                            ),
                            "profile": best_attempt["profile"],
                            "result": "accepted_degraded_generated_candidate",
                            "action_candidate_gate": selected_candidate_gate,
                            "identity_palette": best_attempt.get(
                                "identity_palette"
                            ),
                            "quality": best_attempt.get("quality"),
                            "similarity": best_attempt.get("similarity"),
                        },
                    )
                    update_job(
                        job_id,
                        action_candidate_gate_policy=(
                            GENERATION_ACTION_CANDIDATE_GATE_POLICY
                        ),
                        action_candidate_gate_status=(
                            "generated_completion_candidate_selected"
                            if completion_candidate
                            else (
                                "identity_safe_candidate_pending_final_action_gate"
                                if requires_final_action_gate
                                else "identity_safe_degraded_candidate_selected"
                            )
                        ),
                        action_candidate_final_gate_required_actions=sorted(
                            {
                                *(
                                    get_job(job_id).get(
                                        "action_candidate_final_gate_required_actions",
                                        [],
                                    )
                                    or []
                                ),
                                *(
                                    [action_key(pose)]
                                    if requires_final_action_gate
                                    else []
                                ),
                            }
                        ),
                        action_candidate_gate_action=action_key(pose),
                        action_candidate_gate_attempt_budget=len(
                            active_pose_profiles
                        ),
                        action_candidate_gate_selected_attempt=(
                            best_attempt["attempt"]
                        ),
                        action_candidate_gate_degraded_fallback_used=True,
                        action_candidate_gate_returns_original_upload=False,
                        action_candidate_gate_returns_approved_canonical=False,
                        visual_completion_fallback_used=completion_candidate,
                        visual_completion_fallback_policy=(
                            GENERATION_COMPLETION_FIRST_VISUAL_FALLBACK_POLICY
                            if completion_candidate
                            else None
                        ),
                    )
                elif exhausted_recovery == "upload_anchored_canonical":
                    standing_fallback = fallback_action == "standing"
                    canonical_fallback_policy = (
                        "upload_anchored_canonical_standing_reference_v1"
                        if standing_fallback
                        else (
                            DOG_EATING_CANONICAL_IDENTITY_FALLBACK_POLICY
                            if fallback_action == "eating"
                            else (
                                TOILETING_CANONICAL_TRAY_FALLBACK_POLICY
                                if fallback_action == "toileting"
                                else (
                                    "upload_anchored_canonical_dog_action_"
                                    "reference_v1"
                                )
                            )
                        )
                    )
                    canonical_fallback = (
                        job_dir
                        / f"pose_{index:02d}_upload_anchored_canonical_fallback.png"
                    )
                    canonical_identity_fallback = (
                        canonical_fallback
                        if fallback_action != "toileting"
                        else job_dir
                        / f"pose_{index:02d}_upload_anchored_canonical_identity.png"
                    )
                    canonical_identity_normalization = (
                        normalize_qwen_action_reference_rgba(
                            cartoon_output,
                            canonical_identity_fallback,
                            action=action_key(pose),
                        )
                    )
                    toileting_tray_fallback: dict[str, Any] | None = None
                    if fallback_action == "toileting":
                        tray_staging = (
                            job_dir
                            / f"pose_{index:02d}_canonical_toileting_tray_staging.png"
                        )
                        toileting_tray_fallback = (
                            make_toileting_canonical_tray_fallback(
                                canonical_identity_fallback,
                                tray_staging,
                            )
                        )
                        canonical_normalization = (
                            normalize_qwen_action_reference_rgba(
                                tray_staging,
                                canonical_fallback,
                                action=action_key(pose),
                            )
                        )
                    else:
                        canonical_normalization = canonical_identity_normalization
                    canonical_edge_quality = canonical_normalization[
                        "transparent_edge_quality"
                    ]
                    canonical_quality = pose_cutout_quality(
                        canonical_fallback,
                        pose,
                        require_eating_mouth_contact=False,
                        validate_action_geometry=fallback_action == "toileting",
                    )
                    canonical_palette = evaluate_pose_identity_palette(
                        canonical_identity_fallback,
                        primary_reference=pose_reference_identity_palette,
                        upper_body_reference=(
                            pose_reference_upper_body_identity_palette
                        ),
                        bottom_exclusion=pose_palette_bottom_exclusion,
                        bowl_action=False,
                    )
                    canonical_palette_quality = canonical_palette["quality"]
                    if (
                        canonical_edge_quality.get("acceptable") is not True
                        or canonical_quality.get("acceptable") is not True
                        or canonical_palette_quality.get("acceptable") is not True
                    ):
                        raise RuntimeError(
                            "Upload-anchored canonical "
                            f"{fallback_action} fallback failed "
                            "its alpha, subject, or identity-palette contract."
                        )
                    rejected_attempt = best_action_candidate_degraded_attempt
                    selected_candidate_gate = dict(
                        rejected_attempt.get("action_candidate_gate") or {}
                    )
                    selected_candidate_gate.update(
                        strict_acceptable=False,
                        degraded_fallback_accepted=True,
                        degraded_reason=(
                            "all_generated_candidates_failed_identity_palette"
                        ),
                        degraded_selection_policy=(
                            canonical_fallback_policy
                        ),
                        degraded_selection_key=list(
                            rejected_attempt[
                                "action_candidate_degraded_selection_key"
                            ]
                        ),
                        returns_original_upload=False,
                        returns_approved_canonical=True,
                        action_geometry_deferred_to_scail=(
                            fallback_action != "toileting"
                        ),
                        toileting_tray_geometry_prebound=(
                            fallback_action == "toileting"
                        ),
                        blocks_job=False,
                    )
                    canonical_chroma = {
                        "size": canonical_normalization["output_size"],
                        "foreground_fraction": round(
                            float(canonical_normalization["foreground_pixels"])
                            / float(
                                canonical_normalization["source_size"][0]
                                * canonical_normalization["source_size"][1]
                            ),
                            6,
                        ),
                        "corners": canonical_normalization[
                            "output_alpha_corners"
                        ],
                        "soft_alpha": True,
                        "acceptable": True,
                        "issues": [],
                        "method": (
                            "approved_upload_anchored_canonical_transparent_"
                            "reference"
                        ),
                        "normalized_after_qwen": True,
                        "matte_policy": (
                            "approved_canonical_alpha_preserving_normalization_v1"
                        ),
                        "normalization_policy": (
                            QWEN_ACTION_REFERENCE_NORMALIZATION_POLICY
                        ),
                        "final_reference_normalization": (
                            canonical_normalization
                        ),
                        "final_reference_sha256": canonical_normalization[
                            "output_sha256"
                        ],
                        "transparent_edge_quality": canonical_edge_quality,
                    }
                    canonical_similarity = {
                        "available": True,
                        "acceptable": True,
                        "warning": False,
                        "issues": [],
                        "comparisons": [],
                        "penalty": 0.0,
                        "policy": (
                            "approved_canonical_self_reuse_for_standing_"
                            "identity_recovery_v1"
                            if standing_fallback
                            else "approved_canonical_self_reuse_for_dog_action_"
                            "identity_recovery_v1"
                        ),
                        "canonical_self_reference_reuse": True,
                        "action_geometry_deferred_to_scail": (
                            fallback_action != "toileting"
                        ),
                        "toileting_tray_geometry_prebound": (
                            fallback_action == "toileting"
                        ),
                    }
                    best_attempt = {
                        **rejected_attempt,
                        "attempt": len(active_pose_profiles) + 1,
                        "profile": (
                            "upload_anchored_canonical_standing_recovery"
                            if standing_fallback
                            else f"upload_anchored_canonical_{fallback_action}_recovery"
                        ),
                        "keyframe": canonical_fallback.name,
                        "cutout": canonical_fallback.name,
                        "chroma": canonical_chroma,
                        "quality": canonical_quality,
                        "similarity": canonical_similarity,
                        "identity_palette": canonical_palette_quality,
                        "candidate_identity_palette": canonical_palette[
                            "candidate"
                        ],
                        "candidate_upper_body_identity_palette": (
                            canonical_palette["upper_body_candidate"]
                        ),
                        "fur_silhouette": None,
                        "fur_surface_gate": None,
                        "combined_penalty": round(
                            float(canonical_quality.get("penalty", 0.0))
                            + float(
                                canonical_palette_quality.get("penalty", 0.0)
                            )
                            + float(canonical_edge_quality.get("penalty", 0.0)),
                            6,
                        ),
                        "transparent": True,
                        "background_removal_failed": False,
                        "action_candidate_gate": selected_candidate_gate,
                        "candidate_gate_degraded_fallback": True,
                        "canonical_identity_fallback": {
                            "policy": (
                                canonical_fallback_policy
                            ),
                            "source": str(cartoon_output),
                            "source_sha256": file_sha256(cartoon_output),
                            "selected_path": str(canonical_fallback),
                            "selected_sha256": file_sha256(canonical_fallback),
                            "rejected_generated_attempt": rejected_attempt[
                                "attempt"
                            ],
                            "rejected_identity_palette": rejected_attempt.get(
                                "identity_palette"
                            ),
                            "identity_only_path": str(
                                canonical_identity_fallback
                            ),
                            "identity_only_sha256": file_sha256(
                                canonical_identity_fallback
                            ),
                            "action_geometry_deferred_to_scail": (
                                fallback_action != "toileting"
                            ),
                            "toileting_tray_geometry_prebound": (
                                fallback_action == "toileting"
                            ),
                            "toileting_tray_fallback": (
                                toileting_tray_fallback
                            ),
                        },
                    }
                    append_recovery_event(
                        job_id,
                        {
                            "stage": "cartoon_keyframe",
                            "pose_index": index,
                            "attempt": best_attempt["attempt"],
                            "reason": (
                                "all_generated_candidates_failed_identity_palette"
                            ),
                            "strategy": (
                                "reuse_upload_anchored_canonical_and_defer_"
                                "standing_geometry_to_scail"
                                if standing_fallback
                                else (
                                    "reuse_upload_anchored_canonical_with_"
                                    "deterministic_underbody_tray"
                                    if fallback_action == "toileting"
                                    else "reuse_upload_anchored_canonical_and_defer_"
                                    f"{fallback_action}_geometry_to_scail"
                                )
                            ),
                            "profile": best_attempt["profile"],
                            "result": "accepted_identity_safe_canonical",
                            "action_candidate_gate": selected_candidate_gate,
                            "canonical_identity_fallback": best_attempt[
                                "canonical_identity_fallback"
                            ],
                        },
                    )
                    update_job(
                        job_id,
                        action_candidate_gate_policy=(
                            GENERATION_ACTION_CANDIDATE_GATE_POLICY
                        ),
                        action_candidate_gate_status=(
                            "upload_anchored_canonical_selected"
                        ),
                        action_candidate_gate_action=action_key(pose),
                        action_candidate_gate_attempt_budget=len(
                            active_pose_profiles
                        ),
                        action_candidate_gate_selected_attempt=(
                            best_attempt["attempt"]
                        ),
                        action_candidate_gate_degraded_fallback_used=True,
                        action_candidate_gate_returns_original_upload=False,
                        action_candidate_gate_returns_approved_canonical=True,
                    )
            if (
                best_attempt is None
                and advisory_grooming_minimum_gate_required
                and not strict_first_frame_required
                and best_advisory_grooming_degraded_attempt is not None
            ):
                best_attempt = best_advisory_grooming_degraded_attempt
                selected_degraded_gate = best_attempt.get(
                    "grooming_advisory_minimum_gate"
                )
                if not isinstance(selected_degraded_gate, dict):
                    raise RuntimeError(
                        "Internal grooming recovery error: selected fallback "
                        "lost its recorded audit."
                    )
                selected_degraded_gate = json.loads(
                    json.dumps(selected_degraded_gate)
                )
                selected_degraded_gate[
                    "degraded_fallback_accepted"
                ] = True
                selected_degraded_gate["strict_acceptable"] = bool(
                    selected_degraded_gate.get("acceptable") is True
                )
                selected_degraded_gate["degraded_selection_policy"] = (
                    "gross_pose_then_identity_lowest_risk_complete_chain_v2"
                )
                selected_degraded_gate["degraded_selection_key"] = list(
                    advisory_grooming_degraded_candidate_selection_key(
                        best_attempt
                    )
                )
                selected_degraded_gate["degraded_reason"] = (
                    "strict_action_identity_attempt_budget_exhausted"
                )
                selected_degraded_gate["blocks_scail"] = False
                best_attempt["grooming_advisory_minimum_gate"] = (
                    selected_degraded_gate
                )
                append_recovery_event(
                    job_id,
                    {
                        "stage": "cartoon_keyframe",
                        "pose_index": index,
                        "attempt": best_attempt["attempt"],
                        "reason": (
                            "grooming_minimum_contract_attempts_exhausted"
                        ),
                        "strategy": (
                            "select_identity_first_lowest_risk_candidate_and_"
                            "continue_scail"
                        ),
                        "profile": best_attempt["profile"],
                        "result": "accepted_degraded",
                        "grooming_advisory_minimum_gate": (
                            selected_degraded_gate
                        ),
                    },
                )
                update_job(
                    job_id,
                    grooming_pose_gate_status=(
                        "minimum_contract_degraded_candidate_selected"
                    ),
                    grooming_advisory_minimum_gate_status=(
                        "degraded_candidate_selected"
                    ),
                    grooming_advisory_minimum_gate_attempts=(
                        grooming_advisory_minimum_gate_attempts
                    ),
                    grooming_advisory_minimum_gate_degraded_fallback_used=True,
                    grooming_advisory_minimum_gate_degraded_selected_attempt=(
                        best_attempt["attempt"]
                    ),
                )
            if best_attempt is None and image_request and best_raw_attempt is not None:
                best_attempt = best_raw_attempt
                append_background_removal_failure(
                    job_id,
                    {
                        "stage": "cartoon_keyframe",
                        "pose_index": index,
                        "attempt": best_attempt["attempt"],
                        "reason": best_attempt.get(
                            "background_removal_failure_reason",
                            "unclean_chroma_background",
                        ),
                        "strategy": "return_best_distinct_raw_image_after_retries",
                        "profile": best_attempt["profile"],
                        "result": "returned_opaque_original",
                        "chroma": best_attempt["chroma"],
                        "similarity": best_attempt["similarity"],
                        "identity_palette": best_attempt.get("identity_palette"),
                    },
                )
            if best_attempt is None and image_request:
                raise RuntimeError(
                    f"Pose {index} remained invalid after "
                    f"{len(active_pose_profiles)} attempts: "
                    + pose_attempt_failure_summary(attempts)
                )
            if (
                best_attempt is None
                and force_qwen_reference
                and not strict_first_frame_required
            ):
                failed_action = action_key(pose)
                isolated_error = RuntimeError(
                    f"Pose {index} remained invalid after "
                    f"{len(active_pose_profiles)} attempts: "
                    + pose_attempt_failure_summary(attempts)
                )
                degraded_result, degraded_source = (
                    recover_subject_replacement_action_with_fallback(
                        job_id,
                        get_job(job_id),
                        action=failed_action,
                        pose_index=index,
                        error=isolated_error,
                    )
                )
                fallback_reference = Path(str(degraded_result["reference"]))
                if (
                    not fallback_reference.is_file()
                    or fallback_reference.parent != job_dir
                ):
                    raise RuntimeError(
                        "Isolated Qwen action fallback did not produce a local "
                        "identity reference."
                    )
                refreshed_results = [
                    dict(item)
                    for item in get_job(job_id).get(
                        "subject_replacement_results",
                        [],
                    )
                    if isinstance(item, dict)
                    and str(item.get("action") or "") != failed_action
                ]
                refreshed_results.append(degraded_result)
                failure_record = {
                    "action": failed_action,
                    "pose_index": index,
                    "error": f"RuntimeError: {str(isolated_error)[:4000]}",
                    "attempt_count": len(active_pose_profiles),
                    "fallback_reference": str(fallback_reference),
                    "fallback_reference_sha256": file_sha256(fallback_reference),
                    "fallback_source": degraded_source,
                    "sibling_actions_continue": True,
                    "scail_skipped": True,
                }
                reference_generation_degraded_actions[failed_action] = failure_record
                update_job(
                    job_id,
                    qwen_reference_degraded_actions=dict(
                        reference_generation_degraded_actions
                    ),
                    qwen_reference_failure_isolation_policy=(
                        "per_action_identity_anchored_fallback_continue_siblings_v1"
                    ),
                    subject_replacement_results=refreshed_results,
                    degraded_output_actions=sorted(
                        reference_generation_degraded_actions
                    ),
                )
                append_recovery_event(
                    job_id,
                    {
                        "stage": "cartoon_keyframe",
                        "pose_index": index,
                        "reason": failure_record["error"],
                        "strategy": (
                            "isolate_failed_action_and_continue_sibling_keyframes"
                        ),
                        "result": "accepted_identity_anchored_action_fallback",
                        "action": failed_action,
                        "fallback_reference": str(fallback_reference),
                        "fallback_reference_sha256": failure_record[
                            "fallback_reference_sha256"
                        ],
                    },
                )
                fallback_template = dict(record)
                best_attempt = {
                    **fallback_template,
                    "attempt": len(active_pose_profiles) + 1,
                    "profile": "isolated_action_identity_fallback",
                    "keyframe": fallback_reference.name,
                    "cutout": fallback_reference.name,
                    "chroma": {
                        "acceptable": True,
                        "issues": [],
                        "method": "identity_anchored_action_fallback_rgba",
                    },
                    "quality": {
                        "available": True,
                        "acceptable": False,
                        "issues": ["qwen_action_reference_generation_failed"],
                        "penalty": 1000.0,
                    },
                    "similarity": {
                        "available": True,
                        "acceptable": True,
                        "warning": True,
                        "issues": ["degraded_identity_anchor_not_action_keyframe"],
                        "comparisons": [],
                        "penalty": 1000.0,
                    },
                    "identity_palette": None,
                    "fur_silhouette": None,
                    "fur_surface_gate": None,
                    "grooming_pose_gate": None,
                    "combined_penalty": 2000.0,
                    "transparent": True,
                    "background_removal_failed": False,
                    "reference_generation_degraded": True,
                    "reference_generation_failure": failure_record,
                }
            if best_attempt is None:
                if best_semantic_source is None:
                    evaluated_surface_attempts = [
                        metrics
                        for attempt_record in attempts
                        for metrics in [attempt_record.get("fur_surface_gate")]
                        if isinstance(metrics, dict)
                        and metrics.get("evaluated") is True
                    ]
                    all_evaluated_surface_attempts_rejected = bool(
                        evaluated_surface_attempts
                    ) and all(
                        metrics.get("acceptable") is not True
                        for metrics in evaluated_surface_attempts
                    )
                    if (
                        fur_surface_gate_required
                        and all_evaluated_surface_attempts_rejected
                    ):
                        fur_surface_gate_status_by_action[action_key(pose)] = "failed"
                        update_job(
                            job_id,
                            fur_surface_gate_status=dict(
                                fur_surface_gate_status_by_action
                            ),
                            fur_surface_gate_attempts=(fur_surface_gate_attempt_audits),
                            fur_surface_gate_final_quality={
                                "scope": "qwen_action_reference_pre_scail",
                                "status": "failed",
                                "actions": dict(fur_surface_gate_selected_metrics),
                                "failure": {
                                    "action": action_key(pose),
                                    "reason": "all_surface_candidates_rejected",
                                    "blocks_scail": True,
                                },
                                "final_video_frame_audit": False,
                            },
                        )
                        raise RuntimeError(
                            "Main_case fur surface gate did not approve any "
                            f"{action_key(pose)} Qwen candidate after "
                            f"{len(active_pose_profiles)} attempts: "
                            + pose_attempt_failure_summary(attempts)
                        )
                    if fur_surface_gate_required:
                        passed_surface_attempts = [
                            metrics
                            for metrics in evaluated_surface_attempts
                            if metrics.get("acceptable") is True
                        ]
                        surface_terminal_reason = (
                            "no_selectable_candidate_after_surface_gate"
                            if passed_surface_attempts
                            else "no_candidate_reached_surface_gate"
                        )
                        fur_surface_gate_status_by_action[action_key(pose)] = (
                            "passed_no_selectable_keyframe"
                            if passed_surface_attempts
                            else "not_evaluated_upstream_gate_failed"
                        )
                        update_job(
                            job_id,
                            fur_surface_gate_status=dict(
                                fur_surface_gate_status_by_action
                            ),
                            fur_surface_gate_attempts=(fur_surface_gate_attempt_audits),
                            fur_surface_gate_final_quality={
                                "scope": "qwen_action_reference_pre_scail",
                                "status": "failed",
                                "actions": dict(fur_surface_gate_selected_metrics),
                                "failure": {
                                    "action": action_key(pose),
                                    "reason": surface_terminal_reason,
                                    "surface_candidates_evaluated": len(
                                        evaluated_surface_attempts
                                    ),
                                    "surface_candidates_passed": len(
                                        passed_surface_attempts
                                    ),
                                    "blocks_scail": True,
                                },
                                "final_video_frame_audit": False,
                            },
                        )
                    if grooming_pose_gate_required:
                        evaluated_grooming_attempts = [
                            metrics
                            for attempt_record in attempts
                            for metrics in [attempt_record.get("grooming_pose_gate")]
                            if isinstance(metrics, dict) and "acceptable" in metrics
                        ]
                        all_grooming_attempts_rejected = bool(
                            evaluated_grooming_attempts
                        ) and all(
                            metrics.get("acceptable") is not True
                            for metrics in evaluated_grooming_attempts
                        )
                        passed_grooming_attempts = [
                            metrics
                            for metrics in evaluated_grooming_attempts
                            if metrics.get("acceptable") is True
                        ]
                        update_job(
                            job_id,
                            grooming_pose_gate_status=(
                                "failed"
                                if all_grooming_attempts_rejected
                                else (
                                    "passed_no_selectable_keyframe"
                                    if passed_grooming_attempts
                                    else "not_evaluated_upstream_gate_failed"
                                )
                            ),
                            grooming_pose_gate_selected_metrics=(
                                grooming_pose_gate_selected_metrics
                            ),
                            grooming_pose_gate_attempts=(
                                grooming_pose_gate_attempt_audits
                            ),
                        )
                        if all_grooming_attempts_rejected:
                            raise RuntimeError(
                                "Main_case grooming pose gate rejected every Qwen "
                                f"candidate after {len(active_pose_profiles)} attempts: "
                                + pose_attempt_failure_summary(attempts)
                            )
                        evaluated_palette_attempts = [
                            metrics
                            for attempt_record in attempts
                            for metrics in [attempt_record.get("identity_palette")]
                            if isinstance(metrics, dict) and "acceptable" in metrics
                        ]
                        if evaluated_palette_attempts and all(
                            metrics.get("acceptable") is not True
                            for metrics in evaluated_palette_attempts
                        ):
                            raise RuntimeError(
                                "Grooming Qwen identity palette gate rejected every "
                                f"candidate after {len(active_pose_profiles)} attempts: "
                                + pose_attempt_failure_summary(attempts)
                            )
                    raise RuntimeError(
                        f"Pose {index} remained invalid after "
                        f"{len(active_pose_profiles)} attempts: "
                        + pose_attempt_failure_summary(attempts)
                    )
                fallback_record = best_semantic_source
                if fallback_record["chroma"].get("acceptable", False):
                    raise RuntimeError(
                        "Internal recovery error: semantic background recovery was "
                        "selected for an already clean chroma candidate."
                    )
                fallback_source = job_dir / fallback_record["keyframe"]
                fallback_green = job_dir / f"pose_{index:02d}_semantic_green.png"
                fallback_cutout = (
                    job_dir / f"pose_{index:02d}_semantic_cutout_final.png"
                )
                fallback_semantic = (
                    job_dir / f"pose_{index:02d}_semantic_cutout_raw.png"
                )
                action_semantic_strict_error: str | None = None
                try:
                    chroma_metrics = semantic_background_recovery(
                        fallback_source,
                        fallback_green,
                        fallback_cutout,
                        fallback_semantic,
                        pose=pose,
                    )
                except Exception as exc:
                    if not GENERATION_COMPLETION_FIRST_VISUAL_FALLBACK_ENABLED:
                        raise
                    action_semantic_strict_error = (
                        f"{type(exc).__name__}: {str(exc)[:1000]}"
                    )
                    chroma_metrics = semantic_background_recovery(
                        fallback_source,
                        fallback_green,
                        fallback_cutout,
                        fallback_semantic,
                        pose=pose,
                        enforce_visual_quality=False,
                    )
                    append_recovery_event(
                        job_id,
                        {
                            "stage": "cartoon_keyframe_semantic_recovery",
                            "pose_index": index,
                            "attempt": fallback_record["attempt"],
                            "reason": "strict_visual_semantic_matte_invalid",
                            "strategy": (
                                "use_generated_semantic_matte_without_visual_block"
                            ),
                            "profile": fallback_record["profile"],
                            "source": fallback_source.name,
                            "result": "accepted_completion_fallback",
                            "strict_error": action_semantic_strict_error,
                            "policy": (
                                GENERATION_COMPLETION_FIRST_VISUAL_FALLBACK_POLICY
                            ),
                        },
                    )
                quality = (
                    pose_cutout_quality(
                        fallback_cutout,
                        pose,
                        require_eating_mouth_contact=(
                            require_eating_mouth_contact
                        ),
                    )
                    if GENERATION_VISUAL_AUDITS_BLOCKING
                    else skipped_visual_audit("semantic_subject_quality")
                )
                similarity = (
                    pose_similarity_quality(
                        fallback_cutout,
                        pose=pose,
                        canonical_path=canonical_similarity_reference,
                        previous_poses=accepted_pose_similarity_references,
                    )
                    if GENERATION_VISUAL_AUDITS_BLOCKING
                    else skipped_visual_audit("semantic_pose_similarity")
                )
                semantic_palette = evaluate_pose_identity_palette(
                    fallback_cutout,
                    primary_reference=pose_reference_identity_palette,
                    upper_body_reference=pose_reference_upper_body_identity_palette,
                    bottom_exclusion=pose_palette_bottom_exclusion,
                    bowl_action=bowl_action,
                )
                semantic_candidate_palette = semantic_palette["candidate"]
                semantic_upper_body_candidate_palette = semantic_palette[
                    "upper_body_candidate"
                ]
                semantic_palette_quality = semantic_palette["quality"]
                semantic_color_recovery: dict[str, Any] | None = None
                warm_cast_issues = {
                    "achromatic_source_recolored_with_warm_fur",
                    "new_dominant_warm_brown_or_orange",
                    "new_dominant_yellow_or_gold",
                    "new_dominant_red_or_pink",
                }
                palette_issues = set(
                    semantic_palette_quality.get("issues", [])
                    if isinstance(semantic_palette_quality, dict)
                    else []
                )
                source_upper_is_achromatic = bool(
                    isinstance(pose_reference_upper_body_identity_palette, dict)
                    and float(
                        pose_reference_upper_body_identity_palette.get(
                            "achromatic_fraction",
                            0.0,
                        )
                    )
                    >= 0.85
                    and float(
                        pose_reference_upper_body_identity_palette.get(
                            "warm_fraction",
                            1.0,
                        )
                    )
                    <= 0.12
                )
                if (
                    GENERATION_VISUAL_AUDITS_BLOCKING
                    and palette_issues
                    and palette_issues.issubset(warm_cast_issues)
                    and source_upper_is_achromatic
                    and canonical_similarity_reference is not None
                ):
                    corrected_cutout = (
                        job_dir
                        / f"pose_{index:02d}_semantic_cutout_color_corrected.png"
                    )
                    corrected_green = (
                        job_dir / f"pose_{index:02d}_semantic_green_color_corrected.png"
                    )
                    semantic_color_recovery = neutralize_achromatic_warm_cast(
                        canonical_similarity_reference,
                        fallback_cutout,
                        corrected_cutout,
                        green_target=corrected_green,
                    )
                    if semantic_color_recovery.get("applied"):
                        corrected_validation_cutout = (
                            job_dir
                            / f"pose_{index:02d}_semantic_color_chroma_validation.png"
                        )
                        corrected_chroma = evaluate_chroma_cutout(
                            corrected_green,
                            corrected_validation_cutout,
                        )
                        corrected_quality = pose_cutout_quality(
                            corrected_cutout,
                            pose,
                            require_eating_mouth_contact=(require_eating_mouth_contact),
                        )
                        corrected_similarity = pose_similarity_quality(
                            corrected_cutout,
                            pose=pose,
                            canonical_path=canonical_similarity_reference,
                            previous_poses=accepted_pose_similarity_references,
                        )
                        semantic_palette = evaluate_pose_identity_palette(
                            corrected_cutout,
                            primary_reference=pose_reference_identity_palette,
                            upper_body_reference=(
                                pose_reference_upper_body_identity_palette
                            ),
                            bottom_exclusion=pose_palette_bottom_exclusion,
                            bowl_action=bowl_action,
                        )
                        semantic_candidate_palette = semantic_palette["candidate"]
                        semantic_upper_body_candidate_palette = semantic_palette[
                            "upper_body_candidate"
                        ]
                        semantic_palette_quality = semantic_palette["quality"]
                        semantic_color_recovery.update(
                            {
                                "before_identity_palette_issues": sorted(
                                    palette_issues
                                ),
                                "after_identity_palette_issues": (
                                    list(semantic_palette_quality.get("issues", []))
                                    if isinstance(
                                        semantic_palette_quality,
                                        dict,
                                    )
                                    else []
                                ),
                                "post_chroma_acceptable": bool(
                                    corrected_chroma.get("acceptable")
                                ),
                                "post_quality_acceptable": bool(
                                    corrected_quality.get("acceptable")
                                ),
                                "post_similarity_acceptable": bool(
                                    corrected_similarity.get("acceptable")
                                ),
                            }
                        )
                        fallback_cutout = corrected_cutout
                        fallback_green = corrected_green
                        chroma_metrics = {
                            **corrected_chroma,
                            "method": "rembg_then_lab_warm_cast_recovery",
                            "semantic_cutout": fallback_semantic.name,
                            "color_recovery": semantic_color_recovery,
                        }
                        quality = corrected_quality
                        similarity = corrected_similarity
                fallback_reference = (
                    job_dir / f"pose_{index:02d}_qwen_reference_semantic_final.png"
                )
                chroma_metrics, fallback_cutout = (
                    finalize_qwen_action_reference_candidate(
                        chroma_metrics,
                        fallback_cutout,
                        fallback_reference,
                        action=action_key(pose),
                    )
                )
                if chroma_metrics.get("acceptable") is not True:
                    raise RuntimeError(
                        "Semantic fallback final 512 RGBA normalization failed: "
                        + str(
                            chroma_metrics.get(
                                "final_reference_normalization_error",
                                "unknown normalization error",
                            )
                        )
                    )
                fallback_cleanup = (
                    neutralize_uploaded_identity_chroma_spill(
                        fallback_cutout,
                        source_identity_palette,
                    )
                    if GENERATION_VISUAL_AUDITS_BLOCKING
                    else {
                        "policy": UPLOADED_IDENTITY_CHROMA_SPILL_POLICY,
                        "applied": False,
                        "status": "skipped_advisory",
                    }
                )
                chroma_metrics = merge_identity_chroma_spill_cleanup(
                    chroma_metrics,
                    fallback_cleanup,
                    blocking=bool(GENERATION_VISUAL_AUDITS_BLOCKING),
                )
                quality = (
                    pose_cutout_quality(
                        fallback_cutout,
                        pose,
                        require_eating_mouth_contact=(
                            require_eating_mouth_contact
                        ),
                    )
                    if GENERATION_VISUAL_AUDITS_BLOCKING
                    else skipped_visual_audit("semantic_subject_quality")
                )
                similarity = (
                    pose_similarity_quality(
                        fallback_cutout,
                        pose=pose,
                        canonical_path=canonical_similarity_reference,
                        previous_poses=accepted_pose_similarity_references,
                    )
                    if GENERATION_VISUAL_AUDITS_BLOCKING
                    else skipped_visual_audit("semantic_pose_similarity")
                )
                semantic_palette = evaluate_pose_identity_palette(
                    fallback_cutout,
                    primary_reference=pose_reference_identity_palette,
                    upper_body_reference=(pose_reference_upper_body_identity_palette),
                    bottom_exclusion=pose_palette_bottom_exclusion,
                    bowl_action=bowl_action,
                )
                semantic_candidate_palette = semantic_palette["candidate"]
                semantic_upper_body_candidate_palette = semantic_palette[
                    "upper_body_candidate"
                ]
                semantic_palette_quality = semantic_palette["quality"]
                semantic_fur_silhouette_quality = (
                    eating_fur_silhouette_quality(
                        fallback_cutout,
                        pose_fur_reference_path,
                    )
                    if (
                        GENERATION_VISUAL_AUDITS_BLOCKING
                        and pose_fur_reference_path is not None
                    )
                    else None
                )
                semantic_fur_surface_quality: dict[str, Any] | None = None
                if fur_surface_gate_required:
                    assert isinstance(pose_effective_subject_config, dict)
                    semantic_fur_surface_quality = (
                        evaluate_main_case_fur_surface_candidate(
                            fallback_cutout,
                            action=action_key(pose),
                            alpha_provenance=str(chroma_metrics.get("method") or ""),
                            effective_subject_config=(pose_effective_subject_config),
                        )
                    )
                    semantic_attempt = int(fallback_record["attempt"])
                    fur_surface_gate_attempt_audits.append(
                        {
                            "pose_index": index,
                            "pose": pose,
                            "action": action_key(pose),
                            "attempt": semantic_attempt,
                            "seed": fallback_record["seed"],
                            "source": fallback_source.name,
                            "alpha_cutout": fallback_cutout.name,
                            "alpha_provenance": str(chroma_metrics.get("method") or ""),
                            "semantic_recovery": True,
                            "gate": semantic_fur_surface_quality,
                            "consecutive_rejections": (
                                fur_surface_rejection_streak
                                + (
                                    1
                                    if semantic_fur_surface_quality.get("acceptable")
                                    is not True
                                    else 0
                                )
                            ),
                            "result": (
                                "passed"
                                if semantic_fur_surface_quality.get("acceptable")
                                is True
                                else "rejected"
                            ),
                        }
                    )
                    fur_surface_gate_status_by_action[action_key(pose)] = (
                        "candidate_passed"
                        if semantic_fur_surface_quality.get("acceptable") is True
                        else "failed"
                    )
                    write_json_atomic(
                        job_dir
                        / "workflows"
                        / f"qwen_pose_{index:02d}_fur_surface_gate.json",
                        {
                            "policy": FUR_SURFACE_GATE_POLICY,
                            "gate_version": FUR_SURFACE_GATE_VERSION,
                            "threshold_version": (FUR_SURFACE_THRESHOLD_VERSION),
                            "thresholds": fur_surface_thresholds_payload(),
                            "fail_closed": True,
                            "max_consecutive_rejections": (
                                FUR_SURFACE_GATE_MAX_REJECTIONS
                            ),
                            "retry_policy": FUR_SURFACE_GATE_RETRY_POLICY,
                            "preserves_configured_eating_bowl_recovery": True,
                            "configured_eating_bowl_recovery_attempts": (
                                QWEN_BOWL_POSE_EXTRA_ATTEMPTS
                            ),
                            "short_circuit": fur_surface_gate_short_circuit,
                            "attempts": [
                                item
                                for item in fur_surface_gate_attempt_audits
                                if item["pose_index"] == index
                            ],
                        },
                    )
                    update_job(
                        job_id,
                        fur_surface_gate_status=dict(fur_surface_gate_status_by_action),
                        fur_surface_gate_attempts=(fur_surface_gate_attempt_audits),
                    )
                semantic_rejection_reason = semantic_recovery_gate_failure(
                    chroma_metrics,
                    quality,
                    similarity,
                    semantic_palette_quality,
                    semantic_fur_silhouette_quality,
                    semantic_fur_surface_quality,
                    identity_palette_blocking=(
                        advisory_action_candidate_gate_required
                    ),
                )
                semantic_identity_safe = bool(
                    isinstance(semantic_palette_quality, dict)
                    and semantic_palette_quality.get("acceptable") is True
                )
                semantic_best_effort_issues: list[str] = (
                    ["strict_visual_semantic_matte_invalid"]
                    if action_semantic_strict_error is not None
                    else []
                )
                if semantic_rejection_reason is not None:
                    append_recovery_event(
                        job_id,
                        {
                            "stage": "cartoon_keyframe",
                            "pose_index": index,
                            "attempt": int(fallback_record["attempt"]),
                            "reason": semantic_rejection_reason,
                            "strategy": "rembg_u2net_fallback",
                            "profile": "rembg_u2net_fallback",
                            "result": (
                                "accepted_best_effort"
                                if strict_first_frame_required
                                and (
                                    semantic_identity_safe
                                    or GENERATION_COMPLETION_FIRST_VISUAL_FALLBACK_ENABLED
                                )
                                else "rejected"
                            ),
                            "chroma": chroma_metrics,
                            "quality": quality,
                            "similarity": similarity,
                            "identity_palette": semantic_palette_quality,
                            "fur_silhouette": (semantic_fur_silhouette_quality),
                            "fur_surface_gate": (semantic_fur_surface_quality),
                            "color_recovery": semantic_color_recovery,
                        },
                    )
                    issues = list(chroma_metrics.get("issues", []))
                    issues.extend(quality["issues"])
                    issues.extend(similarity["issues"])
                    if semantic_palette_quality is not None:
                        issues.extend(semantic_palette_quality["issues"])
                    if semantic_fur_silhouette_quality is not None:
                        issues.extend(semantic_fur_silhouette_quality["issues"])
                    if semantic_fur_surface_quality is not None:
                        issues.extend(semantic_fur_surface_quality["issues"])
                    semantic_best_effort_issues = list(
                        dict.fromkeys([*semantic_best_effort_issues, *issues])
                    )
                    if fur_surface_gate_required and not strict_first_frame_required:
                        update_job(
                            job_id,
                            fur_surface_gate_final_quality={
                                "scope": "qwen_action_reference_pre_scail",
                                "status": "failed",
                                "actions": dict(fur_surface_gate_selected_metrics),
                                "failure": {
                                    "action": action_key(pose),
                                    "reason": semantic_rejection_reason,
                                    "blocks_scail": True,
                                },
                                "final_video_frame_audit": False,
                            },
                        )
                    if not strict_first_frame_required:
                        raise RuntimeError(
                            f"Pose {index} semantic recovery failed quality checks: "
                            + ", ".join(issues)
                        )
                best_attempt = {
                    **{
                        field: fallback_record.get(field)
                        for field in POSE_ATTEMPT_PROVENANCE_FIELDS
                    },
                    "attempt": int(fallback_record["attempt"]),
                    "profile": "rembg_u2net_fallback",
                    "seed": fallback_record["seed"],
                    "keyframe": fallback_green.name,
                    "cutout": fallback_cutout.name,
                    "chroma": chroma_metrics,
                    "quality": quality,
                    "similarity": similarity,
                    "identity_palette": semantic_palette_quality,
                    "fur_silhouette": semantic_fur_silhouette_quality,
                    "fur_surface_gate": semantic_fur_surface_quality,
                    "color_recovery": semantic_color_recovery,
                    "candidate_identity_palette": semantic_candidate_palette,
                    "candidate_upper_body_identity_palette": (
                        semantic_upper_body_candidate_palette
                    ),
                    "combined_penalty": round(
                        float(quality["penalty"])
                        + float(similarity["penalty"])
                        + (
                            float(semantic_palette_quality["penalty"])
                            if semantic_palette_quality is not None
                            else 0.0
                        )
                        + (
                            float(semantic_fur_silhouette_quality["penalty"])
                            if semantic_fur_silhouette_quality is not None
                            else 0.0
                        )
                        + (
                            float(
                                semantic_fur_surface_quality.get(
                                    "penalty",
                                    1000.0,
                                )
                            )
                            if isinstance(
                                semantic_fur_surface_quality,
                                dict,
                            )
                            else (1000.0 if fur_surface_gate_required else 0.0)
                        ),
                        6,
                    ),
                    "settings": fallback_record["settings"],
                    "workflow": fallback_record["workflow"],
                    "transparent": True,
                    "background_removal_failed": False,
                }
                if semantic_best_effort_issues and (
                    semantic_identity_safe
                    or GENERATION_COMPLETION_FIRST_VISUAL_FALLBACK_ENABLED
                ):
                    best_attempt["strict_first_frame_best_effort_selected"] = True
                    best_attempt["strict_first_frame_best_effort_issues"] = (
                        semantic_best_effort_issues
                    )
                    best_attempt["strict_first_frame_best_effort_semantic_recovery"] = (
                        True
                    )
                    best_attempt["visual_completion_fallback"] = bool(
                        not semantic_identity_safe
                    )
                append_recovery_event(
                    job_id,
                    {
                        "stage": "cartoon_keyframe",
                        "pose_index": index,
                        "attempt": best_attempt["attempt"],
                        "reason": "qwen_chroma_retries_exhausted",
                        "strategy": "rembg_u2net_fallback",
                        "profile": "rembg_u2net_fallback",
                        "result": (
                            "accepted"
                            if semantic_identity_safe
                            else (
                                "accepted_visual_completion_fallback"
                                if GENERATION_COMPLETION_FIRST_VISUAL_FALLBACK_ENABLED
                                else "rejected_identity_unsafe"
                            )
                        ),
                        "chroma": chroma_metrics,
                        "quality": quality,
                        "similarity": similarity,
                        "identity_palette": semantic_palette_quality,
                        "fur_silhouette": semantic_fur_silhouette_quality,
                        "fur_surface_gate": semantic_fur_surface_quality,
                        "color_recovery": semantic_color_recovery,
                    },
                )
            selected_attempt_number = int(best_attempt.get("attempt") or 0)
            if not 1 <= selected_attempt_number <= QWEN_ACTION_CANDIDATE_ATTEMPTS:
                raise RuntimeError(
                    "Selected action candidate is outside the five-attempt budget."
                )
            selected_palette = best_attempt.get("identity_palette")
            selected_identity_palette_acceptable = bool(
                isinstance(selected_palette, dict)
                and selected_palette.get("acceptable") is True
            )
            selected_visual_completion_fallback = bool(
                best_attempt.get("strict_first_frame_best_effort_selected")
                or not selected_identity_palette_acceptable
            )
            if (
                not GENERATION_COMPLETION_FIRST_VISUAL_FALLBACK_ENABLED
                and (
                not isinstance(selected_palette, dict)
                or selected_palette.get("acceptable") is not True
                )
            ):
                fail_identity_budget_exhausted(
                    selected_palette
                    if isinstance(selected_palette, dict)
                    else None
                )
            selected_keyframe_source = job_dir / str(best_attempt["keyframe"])
            selected_cutout_source = job_dir / str(best_attempt["cutout"])
            action_selection_records = get_job(job_id).get(
                "action_candidate_selection_by_action"
            )
            if isinstance(action_selection_records, dict):
                action_selection_records = json.loads(
                    json.dumps(action_selection_records)
                )
            else:
                action_selection_records = {}
            action_selection_records[action_key(pose)] = {
                "policy": GENERATION_ACTION_CANDIDATE_GATE_POLICY,
                "source_kind": "qwen_generated_action_candidate",
                "attempt_budget": QWEN_ACTION_CANDIDATE_ATTEMPTS,
                "attempts_completed": QWEN_ACTION_CANDIDATE_ATTEMPTS,
                "selected_attempt": selected_attempt_number,
                "selected_penalty": best_attempt.get("combined_penalty"),
                "identity_palette_acceptable": (
                    selected_identity_palette_acceptable
                ),
                "result": (
                    "selected_best_of_five"
                    if not selected_visual_completion_fallback
                    else "selected_generated_completion_fallback"
                ),
                "visual_completion_fallback": (
                    selected_visual_completion_fallback
                ),
                "selected_keyframe_sha256": file_sha256(
                    selected_keyframe_source
                ),
                "selected_cutout_sha256": file_sha256(selected_cutout_source),
                "original_upload_used": False,
                "approved_canonical_used_as_output": False,
            }
            update_job(
                job_id,
                action_candidate_gate_policy=(
                    GENERATION_ACTION_CANDIDATE_GATE_POLICY
                ),
                action_candidate_gate_status=(
                    "best_of_five_selected"
                    if not selected_visual_completion_fallback
                    else "completion_first_best_of_five_selected"
                ),
                action_candidate_gate_action=action_key(pose),
                action_candidate_gate_attempt_budget=(
                    QWEN_ACTION_CANDIDATE_ATTEMPTS
                ),
                action_candidate_gate_attempts_completed=(
                    QWEN_ACTION_CANDIDATE_ATTEMPTS
                ),
                action_candidate_gate_selected_attempt=selected_attempt_number,
                action_candidate_gate_selected_penalty=(
                    best_attempt.get("combined_penalty")
                ),
                action_candidate_gate_returns_original_upload=False,
                action_candidate_gate_returns_approved_canonical=False,
                action_candidate_selection_by_action=action_selection_records,
                visual_completion_fallback_used=(
                    selected_visual_completion_fallback
                ),
                visual_completion_fallback_policy=(
                    GENERATION_COMPLETION_FIRST_VISUAL_FALLBACK_POLICY
                    if selected_visual_completion_fallback
                    else None
                ),
            )
            append_recovery_event(
                job_id,
                {
                    "stage": "cartoon_keyframe",
                    "pose_index": index,
                    "attempt": selected_attempt_number,
                    "reason": "five_candidate_budget_completed",
                    "strategy": (
                        "select_lowest_penalty_identity_safe_generated_candidate"
                        if not selected_visual_completion_fallback
                        else "select_lowest_risk_generated_completion_candidate"
                    ),
                    "profile": best_attempt["profile"],
                    "result": (
                        "selected_best_of_five"
                        if not selected_visual_completion_fallback
                        else "selected_generated_completion_fallback"
                    ),
                    "attempt_budget": QWEN_ACTION_CANDIDATE_ATTEMPTS,
                    "attempts_completed": QWEN_ACTION_CANDIDATE_ATTEMPTS,
                    "combined_penalty": best_attempt.get("combined_penalty"),
                },
            )
            shutil.copy2(job_dir / best_attempt["keyframe"], keyframe)
            keyframe_is_transparent = bool(best_attempt.get("transparent", True))
            if keyframe_is_transparent:
                keyframe_cutout = job_dir / f"pose_{index:02d}_qwen_cutout.png"
                shutil.copy2(job_dir / best_attempt["cutout"], keyframe_cutout)
                selected_keyframe = keyframe_cutout
            else:
                selected_keyframe = keyframe
            reference_generation_degraded = bool(
                best_attempt.get("reference_generation_degraded")
            )
            if fur_surface_gate_required and not reference_generation_degraded:
                selected_surface = best_attempt.get("fur_surface_gate")
                selected_digest = file_sha256(selected_keyframe)
                selected_surface_issues: list[str] = []
                if (
                    not isinstance(selected_surface, dict)
                    or selected_surface.get("acceptable") is not True
                    or selected_surface.get("status") != "passed"
                ):
                    selected_surface_issues.append(
                        "post_selection_fur_surface_gate_not_acceptable"
                    )
                else:
                    selected_gate_metadata = selected_surface.get("gate")
                    if (
                        not isinstance(selected_gate_metadata, dict)
                        or selected_gate_metadata.get("policy")
                        != FUR_SURFACE_GATE_POLICY
                        or selected_gate_metadata.get("gate_version")
                        != FUR_SURFACE_GATE_VERSION
                        or selected_gate_metadata.get("threshold_version")
                        != FUR_SURFACE_THRESHOLD_VERSION
                        or selected_gate_metadata.get("action") != action_key(pose)
                        or selected_gate_metadata.get("candidate_sha256")
                        != selected_digest
                    ):
                        selected_surface_issues.append(
                            "post_selection_fur_surface_provenance_changed"
                        )
                if selected_surface_issues:
                    fur_surface_gate_status_by_action[action_key(pose)] = "failed"
                    if not strict_first_frame_required:
                        update_job(
                            job_id,
                            fur_surface_gate_status=dict(
                                fur_surface_gate_status_by_action
                            ),
                            fur_surface_gate_attempts=(
                                fur_surface_gate_attempt_audits
                            ),
                        )
                        raise RuntimeError(
                            "Main_case fur surface gate selected an invalid "
                            f"{action_key(pose)} candidate."
                        )
                    selected_surface = (
                        json.loads(json.dumps(selected_surface))
                        if isinstance(selected_surface, dict)
                        else {}
                    )
                    selected_surface.update(
                        acceptable=False,
                        status="best_effort_selected",
                        best_effort_accepted=True,
                        strict_acceptable=False,
                        blocks_scail=False,
                        issues=list(
                            dict.fromkeys(
                                [
                                    *(selected_surface.get("issues") or []),
                                    *selected_surface_issues,
                                ]
                            )
                        ),
                        selected_candidate_path=str(selected_keyframe),
                        selected_candidate_sha256=selected_digest,
                        selected_attempt=best_attempt["attempt"],
                    )
                    best_attempt["fur_surface_gate"] = selected_surface
                    best_attempt["strict_first_frame_best_effort_selected"] = True
                    best_attempt["strict_first_frame_best_effort_issues"] = list(
                        dict.fromkeys(
                            [
                                *(
                                    best_attempt.get(
                                        "strict_first_frame_best_effort_issues"
                                    )
                                    or []
                                ),
                                *selected_surface_issues,
                            ]
                        )
                    )
                    fur_surface_gate_selected_metrics[action_key(pose)] = (
                        selected_surface
                    )
                    fur_surface_gate_status_by_action[action_key(pose)] = (
                        "best_effort_selected"
                    )
                    update_job(
                        job_id,
                        fur_surface_gate_status=dict(
                            fur_surface_gate_status_by_action
                        ),
                        fur_surface_gate_attempts=(fur_surface_gate_attempt_audits),
                        fur_surface_gate_selected_metrics=(
                            fur_surface_gate_selected_metrics
                        ),
                        fur_surface_gate_final_quality={
                            "scope": "qwen_action_reference_pre_scail",
                            "status": "best_effort_selected",
                            "policy": FUR_SURFACE_GATE_POLICY,
                            "gate_version": FUR_SURFACE_GATE_VERSION,
                            "threshold_version": FUR_SURFACE_THRESHOLD_VERSION,
                            "actions": dict(fur_surface_gate_selected_metrics),
                            "blocks_scail": False,
                            "final_video_frame_audit": False,
                        },
                    )
                else:
                    selected_surface = json.loads(json.dumps(selected_surface))
                    selected_surface["selected_candidate_path"] = str(
                        selected_keyframe
                    )
                    selected_surface["selected_candidate_sha256"] = selected_digest
                    selected_surface["selected_attempt"] = best_attempt["attempt"]
                    best_attempt["fur_surface_gate"] = selected_surface
                    fur_surface_gate_selected_metrics[action_key(pose)] = (
                        selected_surface
                    )
                    fur_surface_gate_status_by_action[action_key(pose)] = "passed"
                    update_job(
                        job_id,
                        fur_surface_gate_status=dict(
                            fur_surface_gate_status_by_action
                        ),
                        fur_surface_gate_attempts=fur_surface_gate_attempt_audits,
                        fur_surface_gate_selected_metrics=(
                            fur_surface_gate_selected_metrics
                        ),
                        fur_surface_gate_final_quality={
                            "scope": "qwen_action_reference_pre_scail",
                            "status": "running",
                            "policy": FUR_SURFACE_GATE_POLICY,
                            "gate_version": FUR_SURFACE_GATE_VERSION,
                            "threshold_version": (FUR_SURFACE_THRESHOLD_VERSION),
                            "actions": dict(fur_surface_gate_selected_metrics),
                            "final_video_frame_audit": False,
                        },
                    )
            if grooming_pose_gate_required and not reference_generation_degraded:
                selected_gate = best_attempt.get("grooming_pose_gate")
                if not isinstance(selected_gate, dict) or not selected_gate.get(
                    "acceptable"
                ):
                    if not strict_first_frame_required:
                        update_job(
                            job_id,
                            grooming_pose_gate_status="failed",
                            grooming_pose_gate_attempts=(
                                grooming_pose_gate_attempt_audits
                            ),
                        )
                        raise RuntimeError(
                            "Main_case grooming pose gate selected an invalid "
                            "candidate."
                        )
                    selected_gate = (
                        json.loads(json.dumps(selected_gate))
                        if isinstance(selected_gate, dict)
                        else {}
                    )
                    selected_gate.update(
                        acceptable=False,
                        status="best_effort_selected",
                        best_effort_accepted=True,
                        strict_acceptable=False,
                        blocks_scail=False,
                        issues=list(
                            dict.fromkeys(
                                [
                                    *(selected_gate.get("issues") or []),
                                    "post_selection_grooming_pose_gate_not_acceptable",
                                ]
                            )
                        ),
                    )
                    best_attempt["grooming_pose_gate"] = selected_gate
                    best_attempt["strict_first_frame_best_effort_selected"] = True
                    best_attempt["strict_first_frame_best_effort_issues"] = list(
                        dict.fromkeys(
                            [
                                *(
                                    best_attempt.get(
                                        "strict_first_frame_best_effort_issues"
                                    )
                                    or []
                                ),
                                "post_selection_grooming_pose_gate_not_acceptable",
                            ]
                        )
                    )
                    grooming_pose_gate_selected_metrics[action_key(pose)] = (
                        selected_gate
                    )
                    update_job(
                        job_id,
                        grooming_pose_gate_status="best_effort_selected",
                        grooming_pose_gate_attempts=(
                            grooming_pose_gate_attempt_audits
                        ),
                        grooming_pose_gate_selected_metrics=(
                            grooming_pose_gate_selected_metrics
                        ),
                    )
                else:
                    grooming_pose_gate_selected_metrics[action_key(pose)] = (
                        selected_gate
                    )
                    update_job(
                        job_id,
                        grooming_pose_gate_status="passed",
                        grooming_pose_gate_attempts=(
                            grooming_pose_gate_attempt_audits
                        ),
                        grooming_pose_gate_selected_metrics=(
                            grooming_pose_gate_selected_metrics
                        ),
                    )
            if (
                advisory_grooming_minimum_gate_required
                and not reference_generation_degraded
            ):
                selected_minimum_gate = best_attempt.get(
                    "grooming_advisory_minimum_gate"
                )
                selected_digest = file_sha256(selected_keyframe)
                selected_minimum_gate_passed = bool(
                    isinstance(selected_minimum_gate, dict)
                    and (
                        selected_minimum_gate.get("acceptable") is True
                        or selected_minimum_gate.get(
                            "degraded_fallback_accepted"
                        )
                        is True
                    )
                )
                if (
                    not isinstance(selected_minimum_gate, dict)
                    or not selected_minimum_gate_passed
                    or selected_minimum_gate.get("policy")
                    != GROOMING_ADVISORY_MINIMUM_GATE_POLICY
                    or selected_minimum_gate.get("version")
                    != GROOMING_ADVISORY_MINIMUM_GATE_VERSION
                    or selected_minimum_gate.get("candidate_sha256")
                    != selected_digest
                    or pose_palette_reference_path is None
                    or grooming_identity_contract_reference_path is None
                    or selected_minimum_gate.get("identity_reference_path")
                    != str(grooming_identity_contract_reference_path)
                    or selected_minimum_gate.get("identity_reference_sha256")
                    != file_sha256(grooming_identity_contract_reference_path)
                    or selected_minimum_gate.get(
                        "identity_palette_reference_path"
                    )
                    != str(pose_palette_reference_path)
                    or selected_minimum_gate.get(
                        "identity_palette_reference_sha256"
                    )
                    != file_sha256(pose_palette_reference_path)
                ):
                    if strict_first_frame_required:
                        invalid_minimum_gate = (
                            dict(selected_minimum_gate)
                            if isinstance(selected_minimum_gate, dict)
                            else {}
                        )
                        invalid_minimum_issues = list(
                            invalid_minimum_gate.get("issues") or []
                        )
                        invalid_minimum_issues.append(
                            "post_selection_grooming_minimum_contract_invalid"
                        )
                        invalid_minimum_gate.update(
                            policy=GROOMING_ADVISORY_MINIMUM_GATE_POLICY,
                            version=GROOMING_ADVISORY_MINIMUM_GATE_VERSION,
                            acceptable=False,
                            issues=list(dict.fromkeys(invalid_minimum_issues)),
                            candidate_sha256=selected_digest,
                            degraded_fallback_accepted=True,
                            strict_acceptable=False,
                            best_effort_accepted=True,
                            degraded_reason=(
                                "bounded_first_frame_attempt_budget_exhausted"
                            ),
                            blocks_scail=False,
                        )
                        if pose_palette_reference_path is not None:
                            invalid_minimum_gate.update(
                                identity_palette_reference_path=str(
                                    pose_palette_reference_path
                                ),
                                identity_palette_reference_sha256=file_sha256(
                                    pose_palette_reference_path
                                ),
                            )
                        if grooming_identity_contract_reference_path is not None:
                            invalid_minimum_gate.update(
                                identity_reference_path=str(
                                    grooming_identity_contract_reference_path
                                ),
                                identity_reference_sha256=file_sha256(
                                    grooming_identity_contract_reference_path
                                ),
                            )
                        selected_minimum_gate = invalid_minimum_gate
                        best_attempt[
                            "strict_first_frame_best_effort_selected"
                        ] = True
                        best_attempt[
                            "strict_first_frame_best_effort_issues"
                        ] = invalid_minimum_gate["issues"]
                        update_job(
                            job_id,
                            grooming_pose_gate_status=(
                                "minimum_contract_best_effort_selected"
                            ),
                            grooming_advisory_minimum_gate_status=(
                                "best_effort_selected"
                            ),
                            grooming_advisory_minimum_gate_attempts=(
                                grooming_advisory_minimum_gate_attempts
                            ),
                        )
                    else:
                        update_job(
                            job_id,
                            grooming_pose_gate_status="minimum_contract_failed",
                            grooming_advisory_minimum_gate_status="failed",
                            grooming_advisory_minimum_gate_attempts=(
                                grooming_advisory_minimum_gate_attempts
                            ),
                        )
                        raise RuntimeError(
                            "Advisory grooming selected candidate lost its minimum "
                            "action/identity contract."
                        )
                selected_minimum_gate = json.loads(
                    json.dumps(selected_minimum_gate)
                )
                selected_minimum_gate["selected_candidate_path"] = str(
                    selected_keyframe
                )
                selected_minimum_gate["selected_candidate_sha256"] = (
                    selected_digest
                )
                selected_minimum_gate["selected_attempt"] = best_attempt["attempt"]
                selected_minimum_gate["selection_status"] = (
                    "degraded_passed"
                    if selected_minimum_gate.get("degraded_fallback_accepted")
                    is True
                    else "passed"
                )
                best_attempt["grooming_advisory_minimum_gate"] = (
                    selected_minimum_gate
                )
                grooming_advisory_minimum_gate_selected[action_key(pose)] = (
                    selected_minimum_gate
                )
                selected_topology = grooming_foreleg_topology_selection_evidence(
                    best_attempt
                )
                selected_topology_by_action = get_job(job_id).get(
                    "grooming_foreleg_topology_selected"
                )
                selected_topology_by_action = (
                    json.loads(json.dumps(selected_topology_by_action))
                    if isinstance(selected_topology_by_action, dict)
                    else {}
                )
                selected_topology_by_action[action_key(pose)] = selected_topology
                update_job(
                    job_id,
                    grooming_pose_gate_status=(
                        "minimum_contract_degraded_passed"
                        if selected_minimum_gate.get(
                            "degraded_fallback_accepted"
                        )
                        is True
                        else "minimum_contract_passed"
                    ),
                    grooming_advisory_minimum_gate_status=(
                        "degraded_passed"
                        if selected_minimum_gate.get(
                            "degraded_fallback_accepted"
                        )
                        is True
                        else "passed"
                    ),
                    grooming_advisory_minimum_gate_attempts=(
                        grooming_advisory_minimum_gate_attempts
                    ),
                    grooming_advisory_minimum_gate_selected=(
                        grooming_advisory_minimum_gate_selected
                    ),
                    grooming_foreleg_topology_policy=(
                        selected_topology.get("policy")
                    ),
                    grooming_foreleg_topology_status=(
                        "advisory_passed"
                        if selected_topology.get("available") is True
                        and selected_topology.get("acceptable") is True
                        else "advisory_drift"
                    ),
                    grooming_foreleg_topology_selected=(
                        selected_topology_by_action
                    ),
                    grooming_foreleg_topology_blocks_completion=False,
                )
            selected_pose_provenance = pose_attempt_provenance(best_attempt)
            if not selected_pose_provenance["complete"]:
                LOGGER.warning(
                    "%s pose %s selected candidate has incomplete workflow "
                    "provenance: %s",
                    job_id,
                    index,
                    selected_pose_provenance["issues"],
                )
                record_nonblocking_job_anomaly(
                    job_id,
                    code=f"pose_{index:02d}_workflow_provenance_incomplete",
                    message=(
                        f"动作 {index} 已生成，但内部工作流来源记录不完整；"
                        "生成继续，结果需保留降级审计。"
                    ),
                    stage="pose_workflow_provenance",
                    details={
                        "pose_index": index,
                        "pose": pose,
                        "selected_attempt": best_attempt.get("attempt"),
                        "selected_profile": best_attempt.get("profile"),
                        "missing_fields": selected_pose_provenance[
                            "missing_fields"
                        ],
                        "unmapped_workflow_input_names": (
                            selected_pose_provenance[
                                "unmapped_workflow_input_names"
                            ]
                        ),
                    },
                )
            write_workflow(
                job_dir, f"qwen_pose_{index:02d}.json", best_attempt["workflow"]
            )
            selected_quality_record = {
                    "selected_attempt": best_attempt["attempt"],
                    "selected_profile": best_attempt["profile"],
                    "identity_input_policy": (effective_pose_identity_input_policy),
                    "image1_input_name": selected_pose_provenance[
                        "image1_input_name"
                    ],
                    "image2_input_name": selected_pose_provenance[
                        "image2_input_name"
                    ],
                    "image3_input_name": selected_pose_provenance[
                        "image3_input_name"
                    ],
                    "sampler_latent_input_name": selected_pose_provenance[
                        "sampler_latent_input_name"
                    ],
                    "identity_input_name": selected_pose_provenance[
                        "identity_input_name"
                    ],
                    "geometry_input_name": selected_pose_provenance[
                        "geometry_input_name"
                    ],
                    "workflow_input_provenance": selected_pose_provenance,
                    "grooming_foreleg_repair": best_attempt.get(
                        "grooming_foreleg_repair"
                    ),
                    "palette_reference_role": (
                        "uploaded_identity_source"
                        if (
                            (force_qwen_reference and upload_integrity is not None)
                            or
                            effective_pose_identity_input_policy
                            in {
                                POSE_IDENTITY_INPUT_POLICY_DIRECT_UPLOAD,
                                POSE_IDENTITY_INPUT_POLICY_PROFILE_GEOMETRY_FIRST,
                            }
                        )
                        else "approved_canonical_character"
                    ),
                    "selected_chroma": best_attempt["chroma"],
                    "selected_quality": best_attempt["quality"],
                    "selected_similarity": best_attempt["similarity"],
                    "selected_identity_palette": best_attempt.get("identity_palette"),
                    "selected_action_candidate_gate": best_attempt.get(
                        "action_candidate_gate"
                    ),
                    "selected_canonical_identity_fallback": best_attempt.get(
                        "canonical_identity_fallback"
                    ),
                    "selected_fur_silhouette": best_attempt.get("fur_silhouette"),
                    "fur_surface_gate_policy": FUR_SURFACE_GATE_POLICY,
                    "fur_surface_gate_version": FUR_SURFACE_GATE_VERSION,
                    "fur_surface_threshold_version": (FUR_SURFACE_THRESHOLD_VERSION),
                    "selected_fur_surface_gate": best_attempt.get("fur_surface_gate"),
                    "grooming_pose_gate_policy": GROOMING_POSE_GATE_POLICY,
                    "grooming_pose_gate_version": GROOMING_POSE_GATE_VERSION,
                    "selected_grooming_pose_gate": best_attempt.get(
                        "grooming_pose_gate"
                    ),
                    "selected_grooming_advisory_minimum_gate": best_attempt.get(
                        "grooming_advisory_minimum_gate"
                    ),
                    "selected_color_recovery": best_attempt.get("color_recovery"),
                    "reference_identity_palette": pose_reference_identity_palette,
                    "prompt_reference_identity_palette": (
                        pose_identity_palette_prompt_signature
                    ),
                    "reference_upper_body_identity_palette": (
                        pose_reference_upper_body_identity_palette
                    ),
                    "selected_candidate_identity_palette": best_attempt.get(
                        "candidate_identity_palette"
                    ),
                    "selected_candidate_upper_body_identity_palette": best_attempt.get(
                        "candidate_upper_body_identity_palette"
                    ),
                    "selected_settings": best_attempt["settings"],
                    "transparent": keyframe_is_transparent,
                    "background_removal_failed": best_attempt.get(
                        "background_removal_failed", False
                    ),
                    "background_removal_error": best_attempt.get(
                        "background_removal_error"
                    ),
                    "attempts": attempts,
                }

            def persist_selected_quality_record() -> None:
                selected_quality_record.update(
                    strict_first_frame_required=strict_first_frame_required,
                    strict_first_frame_best_effort_selected=bool(
                        best_attempt.get("strict_first_frame_best_effort_selected")
                    ),
                    strict_first_frame_best_effort_issues=list(
                        best_attempt.get("strict_first_frame_best_effort_issues") or []
                    ),
                    strict_first_frame_best_effort_selection_key=(
                        best_attempt.get(
                            "strict_first_frame_best_effort_selection_key"
                        )
                    ),
                    strict_first_frame_attempt_limit=(
                        QWEN_FIRST_FRAME_MAX_ATTEMPTS
                        if strict_first_frame_required
                        else None
                    ),
                    selected_reference_path=str(selected_keyframe),
                    selected_reference_sha256=file_sha256(selected_keyframe),
                    original_upload_fallback_published=False,
                    action_removed=False,
                )
                write_json_atomic(
                    job_dir
                    / "workflows"
                    / f"qwen_pose_{index:02d}_quality.json",
                    selected_quality_record,
                )

            if strict_first_frame_required:
                strict_final_issues = strict_first_frame_record_issues(
                    best_attempt,
                    action=action_key(pose),
                )
                best_effort_selected = bool(
                    best_attempt.get("strict_first_frame_best_effort_selected")
                    or strict_final_issues
                )
                if best_effort_selected:
                    best_attempt["strict_first_frame_best_effort_selected"] = True
                    best_attempt["strict_first_frame_best_effort_issues"] = list(
                        dict.fromkeys(
                            [
                                *(
                                    best_attempt.get(
                                        "strict_first_frame_best_effort_issues"
                                    )
                                    or []
                                ),
                                *strict_final_issues,
                            ]
                        )
                    )
                    append_recovery_event(
                        job_id,
                        {
                            "stage": "cartoon_keyframe",
                            "pose_index": index,
                            "attempt": best_attempt.get("attempt"),
                            "reason": "bounded_first_frame_best_effort_publication",
                            "strategy": (
                                "publish_best_generated_candidate_and_continue_scail"
                            ),
                            "profile": best_attempt.get("profile"),
                            "result": "published_best_effort",
                            "issues": best_attempt[
                                "strict_first_frame_best_effort_issues"
                            ],
                            "attempt_limit": QWEN_FIRST_FRAME_MAX_ATTEMPTS,
                            "original_upload_fallback_published": False,
                            "action_removed": False,
                        },
                    )
                # The validated cache is content-addressed from the selected audit.
                # Persist that audit before attempting to store a strict cache hit.
                persist_selected_quality_record()
                cache_info = best_attempt.get("validated_first_frame_cache")
                cache_status = (
                    "best_effort_not_cached" if best_effort_selected else "hit"
                )
                if best_effort_selected:
                    cache_info = None
                elif not isinstance(cache_info, dict):
                    try:
                        cache_info = store_validated_first_frame_cache(
                            job_id,
                            get_job(job_id),
                            action=action_key(pose),
                            pose_index=index,
                            reference_path=selected_keyframe,
                        )
                        cache_status = "stored"
                    except (OSError, ValueError, KeyError, RuntimeError):
                        LOGGER.warning(
                            "%s could not persist validated first-frame cache for %s",
                            job_id,
                            action_key(pose),
                            exc_info=True,
                        )
                        cache_info = None
                        cache_status = "store_failed"
                first_frame_states = dict(
                    get_job(job_id).get("qwen_first_frame_states") or {}
                )
                first_frame_states[action_key(pose)] = {
                    "status": (
                        "best_effort_selected"
                        if best_effort_selected
                        else "validated"
                    ),
                    "completed": True,
                    "quality_passed": not best_effort_selected,
                    "published": True,
                    "retry_wave": first_frame_retry_wave,
                    "attempt_limit": QWEN_FIRST_FRAME_MAX_ATTEMPTS,
                    "attempts_completed": min(
                        len(attempts),
                        QWEN_FIRST_FRAME_MAX_ATTEMPTS,
                    ),
                    "selected_attempt": best_attempt["attempt"],
                    "selected_profile": best_attempt["profile"],
                    "selection_policy": (
                        QWEN_GENERATED_COMPLETION_SELECTION_POLICY
                        if best_effort_selected
                        else "strict_quality_gate_first_pass_v1"
                    ),
                    "best_effort": best_effort_selected,
                    "best_effort_issues": best_attempt.get(
                        "strict_first_frame_best_effort_issues",
                        [],
                    ),
                    "reference": str(selected_keyframe),
                    "reference_sha256": file_sha256(selected_keyframe),
                    "cache_status": cache_status,
                    "cache": cache_info,
                    "original_upload_fallback_published": False,
                    "action_removed": False,
                }
                validated_action_set = set(
                    get_job(job_id).get(
                        "qwen_first_frame_validated_actions",
                        [],
                    )
                    or []
                )
                if best_effort_selected:
                    validated_action_set.discard(action_key(pose))
                else:
                    validated_action_set.add(action_key(pose))
                validated_actions = sorted(validated_action_set)
                best_effort_actions = sorted(
                    {
                        *(
                            get_job(job_id).get(
                                "qwen_first_frame_best_effort_actions",
                                [],
                            )
                            or []
                        ),
                        *([action_key(pose)] if best_effort_selected else []),
                    }
                )
                update_job(
                    job_id,
                    qwen_first_frame_states=first_frame_states,
                    qwen_first_frame_validated_actions=validated_actions,
                    qwen_first_frame_best_effort_actions=best_effort_actions,
                    qwen_first_frame_current_action=action_key(pose),
                    qwen_first_frame_current_attempt=best_attempt["attempt"],
                    qwen_first_frame_attempt_limit=QWEN_FIRST_FRAME_MAX_ATTEMPTS,
                    qwen_first_frame_cache_status=cache_status,
                    qwen_first_frame_original_fallback_disabled=True,
                    qwen_first_frame_action_removal_disabled=True,
                )
            else:
                persist_selected_quality_record()
            if not reference_generation_degraded:
                accepted_pose_similarity_references.append(
                    {
                        "label": f"pose_{index}",
                        "pose_index": index,
                        "pose": pose,
                        "path": selected_keyframe,
                    }
                )
            effective_pose_lora_attempts.append(
                {
                    "pose_index": index,
                    "pose": pose,
                    "attempt": best_attempt["attempt"],
                    "profile": best_attempt["profile"],
                    "identity_input_policy": (effective_pose_identity_input_policy),
                    "image1_input_name": selected_pose_provenance[
                        "image1_input_name"
                    ],
                    "image2_input_name": selected_pose_provenance[
                        "image2_input_name"
                    ],
                    "image3_input_name": selected_pose_provenance[
                        "image3_input_name"
                    ],
                    "sampler_latent_input_name": selected_pose_provenance[
                        "sampler_latent_input_name"
                    ],
                    "identity_input_name": selected_pose_provenance[
                        "identity_input_name"
                    ],
                    "geometry_input_name": selected_pose_provenance[
                        "geometry_input_name"
                    ],
                    "workflow_input_provenance_complete": (
                        selected_pose_provenance["complete"]
                    ),
                    "workflow_input_provenance_missing_fields": (
                        selected_pose_provenance["missing_fields"]
                    ),
                    "grooming_foreleg_repair": best_attempt.get(
                        "grooming_foreleg_repair"
                    ),
                    "user_loras": best_attempt["settings"]["user_loras"],
                    "user_lora_strength_scale": best_attempt["settings"][
                        "user_lora_strength_scale"
                    ],
                    "user_lora_mode": best_attempt["settings"]["user_lora_mode"],
                    "steps": best_attempt["settings"]["steps"],
                    "cfg": best_attempt["settings"]["cfg"],
                    "denoise": best_attempt["settings"]["denoise"],
                    "quality_retry": best_attempt["settings"]["quality_retry"],
                    "total_strength_cap": best_attempt["settings"][
                        "total_strength_cap"
                    ],
                    "per_lora_strength_cap": best_attempt["settings"][
                        "per_lora_strength_cap"
                    ],
                    "similarity": best_attempt["similarity"],
                    "identity_palette": best_attempt.get("identity_palette"),
                    "fur_silhouette": best_attempt.get("fur_silhouette"),
                    "fur_surface_gate": best_attempt.get("fur_surface_gate"),
                    "grooming_pose_gate": best_attempt.get("grooming_pose_gate"),
                    "grooming_advisory_minimum_gate": best_attempt.get(
                        "grooming_advisory_minimum_gate"
                    ),
                    "reference_generation_degraded": (
                        reference_generation_degraded
                    ),
                    "reference_generation_failure": best_attempt.get(
                        "reference_generation_failure"
                    ),
                }
            )
            if keyframe_is_transparent and best_attempt["attempt"] > 1:
                update_job(job_id, recovered=True)
            update_job(
                job_id,
                recovery_active=False,
                recovery_attempt=None,
                quality_retry=None,
                effective_pose_lora_attempts=effective_pose_lora_attempts,
                pose_similarity_warning_count=sum(
                    bool(item["similarity"].get("warning"))
                    for item in effective_pose_lora_attempts
                ),
                pose_identity_palette_warning_count=sum(
                    bool(
                        isinstance(item.get("identity_palette"), dict)
                        and item["identity_palette"].get("warning")
                    )
                    for item in effective_pose_lora_attempts
                ),
            )
            keyframes.append(selected_keyframe)
            keyframe_transparency.append(keyframe_is_transparent)
        else:
            prefix = f"local_video_api/{job_id}/qwen_pose_{index:02d}"
            workflow = qwen_workflow(
                pose_qwen_input_name,
                prompt,
                negative_prompt,
                base_seed + pose_seed_offset(pose, index),
                prefix,
                user_lora=pose_profiles[0]["user_lora"],
                user_lora_strength=pose_profiles[0]["user_lora_strength"],
                user_lora_mode=pose_profiles[0]["user_lora_mode"],
                user_loras=pose_profiles[0]["user_loras"],
                steps=pose_profiles[0]["steps"],
                cfg=pose_profiles[0]["cfg"],
                denoise=pose_profiles[0]["denoise"],
                builtin_style_strength=QWEN_POSE_STYLE_LORA_STRENGTH,
                suppress_builtin_style=suppress_builtin_style,
                pose_reference_input_name=pose_qwen_secondary_input_name,
            )
            write_workflow(job_dir, f"qwen_pose_{index:02d}.json", workflow)
            entry = queue_and_wait(
                workflow,
                f"{job_id} keyframe {index}/{len(poses)}",
                timeout=7200,
                job_id=job_id,
            )
            source_output = first_output(
                entry,
                {".png", ".jpg", ".jpeg", ".webp"},
                "Qwen keyframe",
            )
            with Image.open(source_output) as generated:
                generated.convert("RGB").save(keyframe, format="PNG", optimize=True)
            keyframes.append(keyframe)
            keyframe_transparency.append(False)
            append_output(
                job_id,
                {
                    "type": "keyframe",
                    "pose_index": index,
                    "pose": pose,
                    "filename": keyframe.name,
                },
            )

    if fur_surface_gate_status_by_action:
        expected_surface_actions = [
            action
            for action in fur_surface_gate_status_by_action
            if action not in reference_generation_degraded_actions
        ]
        missing_surface_actions = [
            action
            for action in expected_surface_actions
            if fur_surface_gate_status_by_action.get(action) != "passed"
            or action not in fur_surface_gate_selected_metrics
        ]
        if missing_surface_actions:
            raise RuntimeError(
                "Main_case fur surface final quality is incomplete before "
                "SCAIL: " + ", ".join(missing_surface_actions)
            )
        update_job(
            job_id,
            fur_surface_gate_status=dict(fur_surface_gate_status_by_action),
            fur_surface_gate_attempts=fur_surface_gate_attempt_audits,
            fur_surface_gate_selected_metrics=(fur_surface_gate_selected_metrics),
            fur_surface_gate_final_quality={
                "scope": "qwen_action_reference_pre_scail",
                "status": (
                    "partial_degraded"
                    if reference_generation_degraded_actions
                    else "passed"
                ),
                "policy": FUR_SURFACE_GATE_POLICY,
                "gate_version": FUR_SURFACE_GATE_VERSION,
                "threshold_version": FUR_SURFACE_THRESHOLD_VERSION,
                "actions": dict(fur_surface_gate_selected_metrics),
                "degraded_actions": dict(reference_generation_degraded_actions),
                "final_video_frame_audit": False,
            },
        )

    if skip_video:
        raise_if_cancelled(job_id)
        update_job(
            job_id,
            stage="finalizing_keyframes",
            progress=35 if force_qwen_reference else 92,
        )
        if transparent_gif:
            for index, (pose, keyframe, keyframe_is_transparent) in enumerate(
                zip(poses, keyframes, keyframe_transparency, strict=True),
                start=1,
            ):
                raise_if_cancelled(job_id)
                raw_keyframe = job_dir / f"pose_{index:02d}_keyframe.png"
                output_removal_error = (
                    "Background removal did not produce a reliable transparent image."
                )
                if keyframe_is_transparent:
                    try:
                        prepare_pose_input(
                            job_id,
                            job_dir,
                            index,
                            pose,
                            keyframe,
                            width,
                            height,
                            transparent=True,
                            preserve_gated_alpha=force_qwen_reference,
                        )
                    except Exception as exc:  # noqa: BLE001 - image mode returns its saved raw image
                        if not image_request:
                            raise
                        keyframe_is_transparent = False
                        output_removal_error = "Background removal finalization failed; the raw image was returned."
                        append_background_removal_failure(
                            job_id,
                            {
                                "stage": "cartoon_keyframe_finalization",
                                "pose_index": index,
                                "reason": "background_removal_finalization_exception",
                                "strategy": "return_raw_image",
                                "result": "returned_opaque_original",
                                "error": f"{type(exc).__name__}: {str(exc)[:1000]}",
                            },
                        )
                if not keyframe_is_transparent:
                    with Image.open(raw_keyframe) as raw_image:
                        raw_width, raw_height = raw_image.size
                    append_output(
                        job_id,
                        {
                            "type": "image",
                            "pose_index": index,
                            "pose": pose,
                            "filename": raw_keyframe.name,
                            "width": raw_width,
                            "height": raw_height,
                            "transparent": False,
                            "background_removal_failed": True,
                            "background_removal_error": output_removal_error,
                            "requested_user_lora_preserved": bool(
                                user_loras
                                and effective_pose_lora_attempts[index - 1][
                                    "user_loras"
                                ]
                                == user_loras
                                and effective_pose_lora_attempts[index - 1][
                                    "user_lora_mode"
                                ]
                                == user_lora_mode
                            ),
                            "effective_lora": effective_pose_lora_attempts[index - 1],
                            "source_role": "cartoon_animation_keyframe_raw",
                        },
                    )
        raise_if_cancelled(job_id)
        free_comfy_models()
        raise_if_cancelled(job_id)
        if force_qwen_reference:
            update_job(
                job_id,
                status="running",
                stage="subject_references_ready",
                progress=35,
                current_pose_index=None,
                current_pose=None,
                recovery_active=False,
                recovery_attempt=None,
            )
            return
        update_job(
            job_id,
            status="completed",
            stage="completed_keyframes_only",
            progress=100,
            current_pose_index=None,
            current_pose=None,
            finished_at=utc_now(),
        )
        return

    if transparent_gif and not all(keyframe_transparency):
        raise RuntimeError(
            "Background removal failed; transparent video generation requires clean keyframes."
        )

    raise_if_cancelled(job_id)
    update_job(job_id, stage="switching_to_video_model", progress=38)
    free_comfy_models()

    for index, (pose, keyframe) in enumerate(
        zip(poses, keyframes, strict=True), start=1
    ):
        raise_if_cancelled(job_id)
        update_job(
            job_id,
            stage="generating_videos",
            progress=round(40 + 55 * (index - 1) / len(poses), 1),
            current_pose_index=index,
            current_pose=pose,
        )
        wan_input = prepare_pose_input(
            job_id,
            job_dir,
            index,
            pose,
            keyframe,
            width,
            height,
            transparent=transparent_gif,
        )
        comfy_wan_input = comfy_input_dir / wan_input.name
        shutil.copy2(wan_input, comfy_wan_input)
        prompt = build_wan_prompt(
            effective_prompt, pose, transparent_gif=transparent_gif
        )
        negative_prompt = WAN_NEGATIVE
        if transparent_gif:
            negative_prompt += (
                ", "
                + (
                    DOG_CARTOON_WAN_NEGATIVE
                    if species == "dog"
                    else CARTOON_WAN_NEGATIVE
                )
                + ", "
                + CHROMA_NEGATIVE
            )
        source_output: Path | None = None
        video_engine = WAN_ENGINE
        video_engine_metadata: dict[str, Any] = wan_engine_metadata(
            steps=wan_steps,
            cfg=wan_cfg,
            shift=wan_shift,
        )
        video_seed = base_seed + 10_000 + pose_seed_offset(pose, index)
        if is_fun_control_pose(pose):
            update_job(
                job_id,
                stage="generating_controlled_running_video",
                progress=round(40 + 55 * (index - 1) / len(poses), 1),
            )
            fun_prompt = build_fun_control_prompt(
                effective_prompt, pose, transparent_gif
            )
            try:
                raise_if_cancelled(job_id)
                free_comfy_models()
                source_output, video_engine_metadata = run_fun_control_video(
                    job_dir=job_dir,
                    pose_index=index,
                    reference_image=wan_input,
                    prompt=fun_prompt,
                    negative_prompt=negative_prompt + ", " + FUN_CONTROL_NEGATIVE,
                    seed=video_seed,
                )
                raise_if_cancelled(job_id)
                video_engine = "wan2.2-fun-5b-control"
            except JobCancelled:
                raise
            except Exception as exc:  # noqa: BLE001 - configured production fallback
                failure = {
                    "engine": "wan2.2-fun-5b-control",
                    "fallback_enabled": FUN_CONTROL_FALLBACK_TO_WAN,
                    "error": str(exc),
                    "timestamp": utc_now(),
                }
                write_json_atomic(
                    job_dir / "workflows" / f"fun_pose_{index:02d}_failure.json",
                    failure,
                )
                if not FUN_CONTROL_FALLBACK_TO_WAN:
                    raise
                LOGGER.exception(
                    "%s controlled video %s/%s failed; falling back to ComfyUI Wan",
                    job_id,
                    index,
                    len(poses),
                )
                video_engine_metadata = wan_engine_metadata(
                    steps=wan_steps,
                    cfg=wan_cfg,
                    shift=wan_shift,
                ) | {"fallback_from": failure}

        if source_output is None:
            prefix = f"local_video_api/{job_id}/wan_pose_{index:02d}"
            workflow = wan_workflow(
                (comfy_relative_dir / wan_input.name).as_posix(),
                prompt,
                negative_prompt,
                video_seed,
                width,
                height,
                prefix,
                steps=wan_steps,
                cfg=wan_cfg,
                shift=wan_shift,
            )
            write_workflow(job_dir, f"wan_pose_{index:02d}.json", workflow)
            entry = queue_and_wait(
                workflow,
                f"{job_id} video {index}/{len(poses)}",
                timeout=21600,
                job_id=job_id,
            )
            source_output = first_output(
                entry, {".mp4", ".webm", ".mov", ".mkv"}, "Wan video"
            )
        raise_if_cancelled(job_id)
        suffix = source_output.suffix.lower()
        final_video = job_dir / f"pose_{index:02d}_video{suffix}"
        shutil.copy2(source_output, final_video)
        if transparent_gif:
            append_output(
                job_id,
                {
                    "type": "video_preview",
                    "pose_index": index,
                    "pose": pose,
                    "filename": final_video.name,
                    "width": width,
                    "height": height,
                    "frames": FRAME_COUNT,
                    "fps": FPS,
                    "duration_seconds": FRAME_COUNT / FPS,
                    "video_engine": video_engine,
                    "video_engine_metadata": video_engine_metadata,
                    "primary": False,
                },
            )
            update_job(
                job_id,
                stage="stabilizing_transparent_animation",
                progress=round(95 + 4 * (index - 1) / len(poses), 1),
            )
            final_gif = job_dir / f"pose_{index:02d}_transparent_512.gif"
            final_apng = job_dir / f"pose_{index:02d}_transparent_512_soft.png"
            raise_if_cancelled(job_id)
            metrics = convert_chroma_video_to_transparent_animations(
                final_video,
                final_gif,
                final_apng,
            )
            metrics_path = job_dir / f"pose_{index:02d}_stabilization_metrics.json"
            write_json_atomic(metrics_path, metrics)
            append_output(
                job_id,
                {
                    "type": "transparent_gif",
                    "pose_index": index,
                    "pose": pose,
                    "filename": final_gif.name,
                    "width": TRANSPARENT_GIF_SIZE,
                    "height": TRANSPARENT_GIF_SIZE,
                    "frames": TRANSPARENT_GIF_FRAMES,
                    "fps": TRANSPARENT_GIF_FPS,
                    "duration_seconds": TRANSPARENT_GIF_DURATION_SECONDS,
                    "transparent": True,
                    "visual_style": "identity_preserving_cute_3d_natural_cat",
                    "edge_mode": "clean_hard_alpha_no_dither",
                    "stabilization": metrics["after"],
                    "visual_core_stabilization": metrics["visual_core_stabilization"],
                    "head_lock": metrics["head_lock"],
                    "video_engine": video_engine,
                    "video_engine_metadata": video_engine_metadata,
                    "primary": False,
                },
            )
            append_output(
                job_id,
                {
                    "type": "transparent_apng",
                    "pose_index": index,
                    "pose": pose,
                    "filename": final_apng.name,
                    "width": TRANSPARENT_GIF_SIZE,
                    "height": TRANSPARENT_GIF_SIZE,
                    "frames": TRANSPARENT_GIF_FRAMES,
                    "fps": TRANSPARENT_GIF_FPS,
                    "duration_seconds": TRANSPARENT_GIF_DURATION_SECONDS,
                    "transparent": True,
                    "visual_style": "identity_preserving_cute_3d_studio",
                    "soft_alpha": True,
                    "stabilization": metrics["after"],
                    "visual_core_stabilization": metrics["visual_core_stabilization"],
                    "head_lock": metrics["head_lock"],
                    "video_engine": video_engine,
                    "video_engine_metadata": video_engine_metadata,
                    "recommended_for_quality": True,
                    "primary": True,
                },
            )
        else:
            append_output(
                job_id,
                {
                    "type": "video",
                    "pose_index": index,
                    "pose": pose,
                    "filename": final_video.name,
                    "width": width,
                    "height": height,
                    "frames": FRAME_COUNT,
                    "fps": FPS,
                    "duration_seconds": FRAME_COUNT / FPS,
                    "video_engine": video_engine,
                    "video_engine_metadata": video_engine_metadata,
                },
            )

    raise_if_cancelled(job_id)
    free_comfy_models()
    raise_if_cancelled(job_id)
    update_job(
        job_id,
        status="completed",
        stage="completed",
        progress=100,
        current_pose_index=None,
        current_pose=None,
        finished_at=utc_now(),
    )
