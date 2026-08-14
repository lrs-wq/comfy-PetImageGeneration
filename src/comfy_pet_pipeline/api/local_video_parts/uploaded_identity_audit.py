"""Uploaded identity persistence, verification, and view-set auditing."""

from __future__ import annotations

# The extracted functions resolve the facade dependencies after runtime sync.
# ruff: noqa: F821

from .runtime import sync_local_video_runtime as _sync_local_video_runtime


def persist_uploaded_identity_views(
    job_dir: Path,
    raw_views: dict[str, bytes],
) -> dict[str, dict[str, Any]]:
    """Normalize and freeze one or three role-addressed identity photographs."""
    _sync_local_video_runtime(globals(), {"persist_uploaded_identity_views"})

    roles = tuple(raw_views)
    if roles not in {("front",), UPLOADED_IDENTITY_VIEW_ROLES}:
        raise ValueError(
            "Uploaded identity views must be front only or front, left, and right."
        )
    records: dict[str, dict[str, Any]] = {}
    for role in roles:
        input_filename, source_filename = UPLOADED_IDENTITY_VIEW_FILES[role]
        input_path = job_dir / input_filename
        source_path = job_dir / source_filename
        normalize_uploaded_image(
            raw_views[role],
            input_path,
            identity_source_path=source_path,
        )
        records[role] = {
            "role": role,
            "input_filename": input_filename,
            "source_filename": source_filename,
            "input_sha256": file_sha256(input_path),
            "source_sha256": file_sha256(source_path),
        }
    return records


def verify_uploaded_identity_submission(
    job: dict[str, Any],
    job_dir: Path,
) -> dict[str, Any]:
    """Verify every normalized/original identity view against queue-time digests."""
    _sync_local_video_runtime(globals(), {"verify_uploaded_identity_submission"})

    persisted_views = job.get("uploaded_identity_views")
    if isinstance(persisted_views, dict) and persisted_views:
        mode = str(job.get("uploaded_identity_view_mode") or "").strip()
        expected_roles = (
            ("front",)
            if mode == "single_view"
            else UPLOADED_IDENTITY_VIEW_ROLES
            if mode == "front_left_right"
            else ()
        )
        if not expected_roles:
            raise RuntimeError("Uploaded identity view mode is invalid.")
        if list(job.get("uploaded_identity_view_roles") or []) != list(expected_roles):
            raise RuntimeError("Uploaded identity view roles changed after submission.")
        if int(job.get("uploaded_identity_view_count") or 0) != len(expected_roles):
            raise RuntimeError("Uploaded identity view count changed after submission.")
        if str(job.get("uploaded_identity_view_policy") or "") != (
            UPLOADED_IDENTITY_VIEW_POLICY
        ):
            raise RuntimeError(
                "Uploaded identity view policy changed after submission."
            )
        if set(persisted_views) != set(expected_roles):
            raise RuntimeError("Uploaded identity view set changed after submission.")

        verified_views: dict[str, dict[str, Any]] = {}
        for role in expected_roles:
            record = persisted_views.get(role)
            if not isinstance(record, dict) or record.get("role") != role:
                raise RuntimeError(f"Uploaded identity {role} record is invalid.")
            input_filename, source_filename = UPLOADED_IDENTITY_VIEW_FILES[role]
            if (
                record.get("input_filename") != input_filename
                or record.get("source_filename") != source_filename
            ):
                raise RuntimeError(
                    f"Uploaded identity {role} filenames changed after submission."
                )
            input_sha256 = str(record.get("input_sha256") or "").strip().lower()
            source_sha256 = str(record.get("source_sha256") or "").strip().lower()
            if not re.fullmatch(r"[0-9a-f]{64}", input_sha256):
                raise RuntimeError(
                    f"Uploaded identity {role} input lacks its submission-time digest."
                )
            if not re.fullmatch(r"[0-9a-f]{64}", source_sha256):
                raise RuntimeError(
                    f"Uploaded identity {role} source lacks its submission-time digest."
                )
            input_path = job_dir / input_filename
            source_path = job_dir / source_filename
            if not input_path.is_file() or file_sha256(input_path) != input_sha256:
                raise RuntimeError(
                    f"Uploaded identity {role} input is missing or changed after submission."
                )
            if not source_path.is_file() or file_sha256(source_path) != source_sha256:
                raise RuntimeError(
                    f"Uploaded identity {role} source is missing or changed after submission."
                )
            verified_views[role] = {
                **record,
                "input_path": input_path,
                "source_path": source_path,
                "input_sha256": input_sha256,
                "source_sha256": source_sha256,
            }

        front = verified_views["front"]
        legacy_input_sha256 = (
            str(job.get("uploaded_identity_input_sha256") or "").strip().lower()
        )
        legacy_source_sha256 = (
            str(job.get("uploaded_identity_source_sha256") or "").strip().lower()
        )
        if (
            legacy_input_sha256 != front["input_sha256"]
            or legacy_source_sha256 != front["source_sha256"]
        ):
            raise RuntimeError(
                "Uploaded front identity digests disagree with legacy submission fields."
            )
        return {
            "input_path": front["input_path"],
            "source_path": front["source_path"],
            "input_sha256": front["input_sha256"],
            "source_sha256": front["source_sha256"],
            "view_mode": mode,
            "view_roles": list(expected_roles),
            "views": verified_views,
        }

    # Backward compatibility for already queued v1.0 single-view jobs.
    input_path = job_dir / "input.png"
    source_path = job_dir / "input_identity_source.png"
    expected_input_sha256 = (
        str(job.get("uploaded_identity_input_sha256") or "").strip().lower()
    )
    expected_source_sha256 = (
        str(job.get("uploaded_identity_source_sha256") or "").strip().lower()
    )
    if not re.fullmatch(r"[0-9a-f]{64}", expected_input_sha256):
        raise RuntimeError("Uploaded identity input lacks its submission-time digest.")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_source_sha256):
        raise RuntimeError("Uploaded identity source lacks its submission-time digest.")
    if not input_path.is_file() or file_sha256(input_path) != expected_input_sha256:
        raise RuntimeError(
            "Uploaded identity input is missing or changed after submission."
        )
    if not source_path.is_file() or file_sha256(source_path) != expected_source_sha256:
        raise RuntimeError(
            "Uploaded identity source is missing or changed after submission."
        )
    front = {
        "role": "front",
        "input_filename": "input.png",
        "source_filename": "input_identity_source.png",
        "input_path": input_path,
        "source_path": source_path,
        "input_sha256": expected_input_sha256,
        "source_sha256": expected_source_sha256,
    }
    return {
        "input_path": input_path,
        "source_path": source_path,
        "input_sha256": expected_input_sha256,
        "source_sha256": expected_source_sha256,
        "view_mode": "single_view",
        "view_roles": ["front"],
        "views": {"front": front},
    }


def aggregate_identity_palette_signatures(
    signatures: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Create an equal-view consensus without letting a large crop dominate."""
    _sync_local_video_runtime(globals(), {"aggregate_identity_palette_signatures"})

    if len(signatures) == 1:
        return dict(next(iter(signatures.values())))
    unavailable = [
        role for role, signature in signatures.items() if not signature.get("available")
    ]
    if unavailable:
        return {
            "available": False,
            "reason": "identity_view_palette_signature_unavailable",
            "unavailable_roles": unavailable,
            "source_view_roles": list(signatures),
            "aggregation": "equal_view_mean_v1",
        }
    count = float(len(signatures))
    fractions = {
        category: round(
            sum(
                float(signature["fractions"].get(category, 0.0))
                for signature in signatures.values()
            )
            / count,
            6,
        )
        for category in IDENTITY_PALETTE_CATEGORIES
    }
    dominant = [
        category
        for category in IDENTITY_PALETTE_CATEGORIES
        if fractions[category] >= 0.10
    ]
    if not dominant:
        dominant = sorted(
            IDENTITY_PALETTE_CATEGORIES,
            key=lambda category: fractions[category],
            reverse=True,
        )[:2]

    def mean_field(name: str) -> float:
        return round(
            sum(float(signature.get(name, 0.0)) for signature in signatures.values())
            / count,
            6,
        )

    return {
        "available": True,
        "mask_method": "multi_view_equal_weight_consensus",
        "aggregation": "equal_view_mean_v1",
        "source_view_roles": list(signatures),
        "sample_count": sum(
            int(signature.get("sample_count", 0)) for signature in signatures.values()
        ),
        "fractions": fractions,
        "dominant": dominant,
        "achromatic_fraction": mean_field("achromatic_fraction"),
        "chromatic_fraction": mean_field("chromatic_fraction"),
        "warm_fraction": mean_field("warm_fraction"),
        "warm_saturation_mean": mean_field("warm_saturation_mean"),
        "warm_saturation_p90": mean_field("warm_saturation_p90"),
        "saturated_warm_fraction": mean_field("saturated_warm_fraction"),
        "red_or_pink_body_fraction": mean_field("red_or_pink_body_fraction"),
        "red_or_pink_body_share": mean_field("red_or_pink_body_share"),
        "red_or_pink_localized_to_head": all(
            bool(signature.get("red_or_pink_localized_to_head", False))
            or float(signature["fractions"].get("red_or_pink", 0.0)) <= 0.025
            for signature in signatures.values()
        ),
        "green_or_cyan_body_fraction": mean_field("green_or_cyan_body_fraction"),
        "green_or_cyan_body_share": mean_field("green_or_cyan_body_share"),
        "green_or_cyan_localized_to_head": all(
            bool(signature.get("green_or_cyan_localized_to_head", False))
            or (
                float(signature["fractions"].get("green", 0.0))
                + float(signature["fractions"].get("blue_or_cyan", 0.0))
                <= 0.025
            )
            for signature in signatures.values()
        ),
    }


def canonical_upper_body_palette_reference(
    view_upper_body_palettes: dict[str, dict[str, Any]],
    consensus_upper_body_palette: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Use the front view for a front/three-quarter canonical's local value balance."""
    _sync_local_video_runtime(globals(), {"canonical_upper_body_palette_reference"})

    front = view_upper_body_palettes.get("front")
    if isinstance(front, dict) and front.get("available"):
        selected = dict(front)
        selected["canonical_reference_role"] = "front"
        selected["canonical_reference_policy"] = (
            CANONICAL_UPPER_BODY_PALETTE_REFERENCE_POLICY
        )
        return selected
    if consensus_upper_body_palette is None:
        return None
    selected = dict(consensus_upper_body_palette)
    selected["canonical_reference_role"] = "consensus_fallback"
    selected["canonical_reference_policy"] = (
        CANONICAL_UPPER_BODY_PALETTE_REFERENCE_POLICY
    )
    return selected


def compose_uploaded_identity_view_audit(
    upload_integrity: dict[str, Any],
) -> dict[str, Any]:
    """Combine optional view-set checks with required front-view validation."""
    _sync_local_video_runtime(globals(), {"compose_uploaded_identity_view_audit"})

    views = upload_integrity.get("views")
    if not isinstance(views, dict) or not isinstance(views.get("front"), dict):
        raise RuntimeError("Uploaded identity front view integrity is invalid.")
    front_view_audit = audit_uploaded_front_identity_view(
        Path(views["front"]["input_path"])
    )
    if GENERATION_VISUAL_AUDITS_BLOCKING:
        audit = audit_uploaded_identity_view_set(upload_integrity)
        structural_acceptable = bool(audit.get("acceptable"))
        structural_issues = list(audit.get("issues") or [])
        structural_status = "passed" if structural_acceptable else "failed"
    else:
        audit = {
            "policy": UPLOADED_IDENTITY_VIEW_POLICY,
            "blocking": False,
            "fail_closed": False,
            "generation_audit_policy": GENERATION_AUDIT_POLICY,
            "view_mode": upload_integrity.get("view_mode"),
            "roles": list(upload_integrity.get("view_roles") or []),
            "view_count": len(upload_integrity.get("view_roles") or []),
            "duplicate_pairs": [],
            "view_angle_comparisons": {},
            "pairwise_palette_observations": {},
            "palettes": {},
            "upper_body_palettes": {},
        }
        structural_acceptable = True
        structural_issues = []
        structural_status = "skipped_advisory"
    front_issues = list(front_view_audit.get("issues") or [])
    front_blocking = bool(front_view_audit.get("blocking"))
    front_blocking_issues = (
        front_issues
        if front_blocking and not front_view_audit.get("acceptable")
        else []
    )
    blocking_issues = list(
        dict.fromkeys(structural_issues + front_blocking_issues)
    )
    blocking_acceptable = structural_acceptable and not front_blocking_issues
    advisory_issues = [] if front_blocking else front_issues
    overall_acceptable = blocking_acceptable and bool(
        front_view_audit.get("acceptable")
    )
    status = (
        "failed"
        if not blocking_acceptable
        else "warning_advisory"
        if advisory_issues
        else "passed"
        if GENERATION_VISUAL_AUDITS_BLOCKING
        else "passed_advisory"
    )
    audit.update(
        {
            "status": status,
            "acceptable": overall_acceptable,
            "blocking_acceptable": blocking_acceptable,
            "blocking_issues": blocking_issues,
            "advisory_issues": advisory_issues,
            "issues": list(dict.fromkeys(blocking_issues + advisory_issues)),
            "warnings": advisory_issues,
            "structural_audit_status": structural_status,
            "front_view_audit": front_view_audit,
            "front_view_audit_required": True,
            "front_view_audit_blocking": front_blocking,
        }
    )
    return audit


def audit_uploaded_identity_view_set(
    upload_integrity: dict[str, Any],
) -> dict[str, Any]:
    """Fail closed on malformed, duplicate, or grossly contradictory view sets."""
    _sync_local_video_runtime(globals(), {"audit_uploaded_identity_view_set"})

    views = upload_integrity.get("views")
    if not isinstance(views, dict) or set(views) not in (
        {"front"},
        set(UPLOADED_IDENTITY_VIEW_ROLES),
    ):
        raise RuntimeError("Uploaded identity view integrity set is invalid.")
    roles = [role for role in UPLOADED_IDENTITY_VIEW_ROLES if role in views]
    if roles == ["front"]:
        return {
            "policy": UPLOADED_IDENTITY_VIEW_POLICY,
            "view_mode": "single_view",
            "roles": roles,
            "view_count": 1,
            "acceptable": True,
            "fail_closed": True,
            "issues": [],
            "duplicate_pairs": [],
            "view_angle_comparisons": {},
            "view_angle_diversity_required": False,
            "pairwise_palette_observations": {},
            "palettes": {},
            "upper_body_palettes": {},
            "same_pet_submission_contract_required": False,
            "biometric_same_subject_guarantee": False,
        }
    palettes = {
        role: identity_palette_signature(Path(views[role]["source_path"]))
        for role in roles
    }
    upper_body_palettes = {
        role: identity_palette_signature(
            Path(views[role]["source_path"]),
            exclude_bottom_fraction=BOWL_IDENTITY_PALETTE_UPPER_BODY_EXCLUSION,
        )
        for role in roles
    }
    issues: list[str] = []
    unavailable = [
        role
        for role in roles
        if not palettes[role].get("available")
        or not upper_body_palettes[role].get("available")
    ]
    issues.extend(f"{role}_palette_signature_unavailable" for role in unavailable)
    duplicate_pairs: list[list[str]] = []
    for first_index, first_role in enumerate(roles):
        for second_role in roles[first_index + 1 :]:
            if (
                views[first_role]["source_sha256"]
                == views[second_role]["source_sha256"]
                or views[first_role]["input_sha256"]
                == views[second_role]["input_sha256"]
            ):
                duplicate_pairs.append([first_role, second_role])
                issues.append(f"duplicate_identity_views_{first_role}_{second_role}")
    view_similarity_signatures = {
        role: pose_similarity_signature(Path(views[role]["input_path"]))
        for role in roles
    }
    unavailable_similarity_roles = [
        role
        for role, signature in view_similarity_signatures.items()
        if not signature.get("available")
    ]
    issues.extend(
        f"{role}_view_angle_signature_unavailable"
        for role in unavailable_similarity_roles
    )
    view_angle_comparisons: dict[str, Any] = {}
    if not unavailable_similarity_roles:
        for first_index, first_role in enumerate(roles):
            for second_role in roles[first_index + 1 :]:
                comparison = compare_pose_signatures(
                    view_similarity_signatures[first_role],
                    view_similarity_signatures[second_role],
                )
                near_duplicate = bool(
                    comparison.get("sha256_equal")
                    or (
                        float(comparison.get("silhouette_iou", 0.0))
                        >= QWEN_POSE_HARD_DUPLICATE_IOU
                        and int(comparison.get("dhash_distance", 64))
                        <= QWEN_POSE_HARD_DUPLICATE_DHASH
                    )
                )
                comparison["near_duplicate"] = near_duplicate
                comparison["iou_threshold"] = QWEN_POSE_HARD_DUPLICATE_IOU
                comparison["dhash_threshold"] = QWEN_POSE_HARD_DUPLICATE_DHASH
                view_angle_comparisons[f"{first_role}_{second_role}"] = comparison
                if near_duplicate:
                    issues.append(
                        f"near_duplicate_identity_views_{first_role}_{second_role}"
                    )
    pairwise: dict[str, Any] = {}
    if not unavailable:
        for side_role in roles[1:]:
            front_palette = palettes["front"]
            side_palette = palettes[side_role]
            achromatic_delta = abs(
                float(front_palette["achromatic_fraction"])
                - float(side_palette["achromatic_fraction"])
            )
            warm_delta = abs(
                float(front_palette["warm_fraction"])
                - float(side_palette["warm_fraction"])
            )
            gross_palette_conflict = achromatic_delta > 0.55 and warm_delta > 0.35
            if gross_palette_conflict:
                issues.append(f"gross_palette_conflict_front_{side_role}")
            pairwise[f"front_{side_role}"] = {
                "achromatic_fraction_delta": round(achromatic_delta, 6),
                "warm_fraction_delta": round(warm_delta, 6),
                "gross_palette_conflict": gross_palette_conflict,
            }
    audit = {
        "policy": UPLOADED_IDENTITY_VIEW_POLICY,
        "view_mode": upload_integrity.get("view_mode"),
        "roles": roles,
        "view_count": len(roles),
        "acceptable": not issues,
        "fail_closed": GENERATION_VISUAL_AUDITS_BLOCKING,
        "blocking": GENERATION_VISUAL_AUDITS_BLOCKING,
        "generation_audit_policy": GENERATION_AUDIT_POLICY,
        "issues": list(dict.fromkeys(issues)),
        "duplicate_pairs": duplicate_pairs,
        "view_angle_comparisons": view_angle_comparisons,
        "view_angle_diversity_required": True,
        "pairwise_palette_observations": pairwise,
        "palettes": palettes,
        "upper_body_palettes": upper_body_palettes,
        "same_pet_submission_contract_required": len(roles) == 3,
        "biometric_same_subject_guarantee": False,
    }
    return audit
