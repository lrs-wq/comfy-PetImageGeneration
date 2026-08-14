"""Action-reference, fur-surface, pose, and silhouette quality gates."""

from __future__ import annotations

# The extracted functions resolve the facade dependencies after runtime sync.
# ruff: noqa: F821

from .runtime import sync_local_video_runtime as _sync_local_video_runtime


def finalize_qwen_action_reference_candidate(
    metrics: dict[str, Any],
    cutout: Path,
    final_target: Path | None = None,
) -> tuple[dict[str, Any], Path]:
    """Normalize one reliable matte into the exact 512 RGBA gate/SCAIL file."""
    _sync_local_video_runtime(globals(), {"finalize_qwen_action_reference_candidate"})

    final_target = final_target or cutout.with_name(f"{cutout.stem}_reference_512.png")
    try:
        normalization = normalize_qwen_action_reference_rgba(
            cutout,
            final_target,
        )
    except Exception as exc:  # noqa: BLE001 - candidate must fail closed
        final_target.unlink(missing_ok=True)
        return (
            {
                **metrics,
                "acceptable": False,
                "issues": list(
                    dict.fromkeys(
                        [
                            *[str(issue) for issue in metrics.get("issues", [])],
                            "final_reference_normalization_failed",
                        ]
                    )
                ),
                "normalized_after_qwen": False,
                "matte_policy": QWEN_ACTION_CANDIDATE_MATTE_POLICY,
                "normalization_policy": (QWEN_ACTION_REFERENCE_NORMALIZATION_POLICY),
                "final_reference_normalization_error": (
                    f"{type(exc).__name__}: {str(exc)[:1000]}"
                ),
            },
            cutout,
        )
    return (
        {
            **metrics,
            "normalized_after_qwen": True,
            "matte_policy": QWEN_ACTION_CANDIDATE_MATTE_POLICY,
            "normalization_policy": QWEN_ACTION_REFERENCE_NORMALIZATION_POLICY,
            "final_reference_normalization": normalization,
            "final_reference_sha256": normalization["output_sha256"],
        },
        final_target,
    )


def normalize_qwen_action_candidate(
    source: Path,
    direct_cutout_target: Path,
    semantic_green_target: Path,
    semantic_cutout_target: Path,
    semantic_raw_target: Path,
    *,
    pose: str,
    allow_semantic_matte: bool,
) -> tuple[dict[str, Any], Path]:
    """Normalize one Qwen action candidate before any action or SCAIL gate."""
    _sync_local_video_runtime(globals(), {"normalize_qwen_action_candidate"})

    try:
        direct_metrics = evaluate_chroma_cutout(source, direct_cutout_target)
    except Exception as exc:  # noqa: BLE001 - fail closed and preserve audit detail
        direct_metrics = {
            "acceptable": False,
            "issues": ["background_removal_exception"],
            "error": f"{type(exc).__name__}: {str(exc)[:1000]}",
            "method": "chroma_key_exception",
        }
    if direct_metrics.get("acceptable"):
        final_metrics, final_cutout = finalize_qwen_action_reference_candidate(
            direct_metrics,
            direct_cutout_target,
        )
        return (
            {
                **final_metrics,
                "direct_chroma": direct_metrics,
                "semantic_recovery_attempted": False,
                "semantic_recovery_acceptable": None,
            },
            final_cutout,
        )
    if not allow_semantic_matte:
        return (
            {
                **direct_metrics,
                "direct_chroma": direct_metrics,
                "semantic_recovery_attempted": False,
                "semantic_recovery_acceptable": None,
                "normalized_after_qwen": False,
                "matte_policy": QWEN_ACTION_CANDIDATE_MATTE_POLICY,
                "normalization_policy": QWEN_ACTION_REFERENCE_NORMALIZATION_POLICY,
            },
            direct_cutout_target,
        )

    try:
        semantic_metrics = semantic_background_recovery(
            source,
            semantic_green_target,
            semantic_cutout_target,
            semantic_raw_target,
            pose=pose,
        )
    except Exception as exc:  # noqa: BLE001 - candidate remains rejected
        return (
            {
                **direct_metrics,
                "direct_chroma": direct_metrics,
                "semantic_recovery_attempted": True,
                "semantic_recovery_acceptable": False,
                "semantic_recovery_error": (f"{type(exc).__name__}: {str(exc)[:1000]}"),
                "normalized_after_qwen": False,
                "matte_policy": QWEN_ACTION_CANDIDATE_MATTE_POLICY,
                "normalization_policy": QWEN_ACTION_REFERENCE_NORMALIZATION_POLICY,
            },
            direct_cutout_target,
        )
    final_metrics, final_cutout = finalize_qwen_action_reference_candidate(
        semantic_metrics,
        semantic_cutout_target,
    )
    return (
        {
            **final_metrics,
            "direct_chroma": direct_metrics,
            "semantic_recovery_attempted": True,
            "semantic_recovery_acceptable": bool(final_metrics.get("acceptable")),
        },
        final_cutout,
    )


def fur_surface_gate_failure(
    action: str,
    issue: str,
    *,
    baseline: dict[str, Any] | None = None,
    candidate: Path | None = None,
    alpha_provenance: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Return a JSON-safe surface verdict that can never be mistaken for pass."""
    _sync_local_video_runtime(globals(), {"fur_surface_gate_failure"})

    result: dict[str, Any] = {
        "available": False,
        "evaluated": False,
        "acceptable": False,
        "verdict": "rejected",
        "status": "rejected",
        "issues": [issue],
        "candidate_absolute_issues": [issue],
        "relative_issues": [],
        "baseline_issues": [],
        "candidate": None,
        "baseline": None,
        "comparisons": {},
        "thresholds": fur_surface_thresholds_payload(),
        "gate": {
            "policy": FUR_SURFACE_GATE_POLICY,
            "gate_version": FUR_SURFACE_GATE_VERSION,
            "threshold_version": FUR_SURFACE_THRESHOLD_VERSION,
            "action": action,
            "reliable_alpha_required": True,
            "alpha_provenance": alpha_provenance,
            "candidate_path": str(candidate) if candidate is not None else None,
            "baseline_provenance": (
                baseline.get("provenance") if isinstance(baseline, dict) else None
            ),
            "baseline_snapshot_sha256": (
                baseline.get("snapshot_sha256") if isinstance(baseline, dict) else None
            ),
        },
        "penalty": 1000.0,
        "policy": {
            "name": FUR_SURFACE_GATE_POLICY,
            "gate_version": FUR_SURFACE_GATE_VERSION,
            "threshold_version": FUR_SURFACE_THRESHOLD_VERSION,
            "production_integrated": True,
            "validated_baseline_required_for_acceptance": True,
            "opaque_background_can_be_rejected_but_not_accepted": True,
            "limitations": [issue],
        },
    }
    if error:
        result["error"] = error[:1000]
    return result


def resolve_main_case_fur_surface_baseline(
    effective_subject_config: dict[str, Any],
    action: str,
) -> tuple[Path, dict[str, Any]]:
    """Resolve a job-frozen baseline and verify every provenance boundary."""
    _sync_local_video_runtime(globals(), {"resolve_main_case_fur_surface_baseline"})

    if effective_subject_config.get("profile_id") != "sleek_tabby_cat_v1":
        raise RuntimeError("Fur surface gate received an unsupported profile")
    action_config = effective_subject_config.get("actions", {}).get(action)
    if not isinstance(action_config, dict):
        raise RuntimeError(f"Fur surface gate lacks {action} action configuration")
    gate = action_config.get("fur_surface_gate")
    if (
        not isinstance(gate, dict)
        or gate.get("enabled") is not True
        or gate.get("policy") != FUR_SURFACE_GATE_POLICY
        or gate.get("gate_version") != FUR_SURFACE_GATE_VERSION
        or gate.get("threshold_version") != FUR_SURFACE_THRESHOLD_VERSION
    ):
        raise RuntimeError(f"Fur surface gate lacks frozen {action} policy provenance")
    baseline = gate.get("baseline")
    if not isinstance(baseline, dict):
        raise RuntimeError(f"Fur surface gate lacks {action} baseline provenance")
    baseline_path = Path(str(baseline.get("path") or ""))
    if not baseline_path.is_file():
        raise RuntimeError(f"Fur surface gate {action} baseline snapshot is missing")
    digest = file_sha256(baseline_path)
    expected_digests = {
        str(baseline.get("sha256") or ""),
        str(baseline.get("snapshot_sha256") or ""),
        str(baseline.get("source_sha256") or ""),
    }
    if "" in expected_digests or expected_digests != {digest}:
        raise RuntimeError(
            f"Fur surface gate {action} baseline snapshot digest changed"
        )
    provenance = baseline.get("provenance")
    if (
        not isinstance(provenance, dict)
        or provenance.get("profile_id") != effective_subject_config.get("profile_id")
        or provenance.get("profile_sha256")
        != effective_subject_config.get("profile_sha256")
        or provenance.get("action") != action
        or provenance.get("validated_action_baseline_reference_sha256") != digest
        or baseline.get("provenance_sha256") != json_value_sha256(provenance)
    ):
        raise RuntimeError(f"Fur surface gate {action} baseline provenance changed")
    contract = baseline.get("absolute_contract")
    if not isinstance(contract, dict) or contract.get("acceptable") is not True:
        raise RuntimeError(
            f"Fur surface gate {action} baseline absolute contract is unavailable"
        )
    return baseline_path, baseline


def evaluate_main_case_fur_surface_candidate(
    cutout: Path,
    *,
    action: str,
    alpha_provenance: str,
    effective_subject_config: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate one reliable alpha/rembg candidate against its action snapshot."""
    _sync_local_video_runtime(globals(), {"evaluate_main_case_fur_surface_candidate"})

    upload_validation = effective_subject_config.get("upload_validation")
    uploaded_identity_absolute = bool(
        isinstance(upload_validation, dict)
        and upload_validation.get("uploaded_identity_appearance_authority") is True
        and effective_subject_config.get("actions", {})
        .get(action, {})
        .get("fur_surface_gate", {})
        .get("candidate_comparison_mode")
        == UPLOADED_IDENTITY_SURFACE_COMPARISON_MODE
    )
    baseline_path: Path | None = None
    baseline: dict[str, Any] | None = None
    if not uploaded_identity_absolute:
        try:
            baseline_path, baseline = resolve_main_case_fur_surface_baseline(
                effective_subject_config,
                action,
            )
        except Exception as exc:  # noqa: BLE001 - retry/fail closed
            return fur_surface_gate_failure(
                action,
                "fur_surface_baseline_provenance_invalid",
                candidate=cutout,
                alpha_provenance=alpha_provenance,
                error=f"{type(exc).__name__}: {exc}",
            )
    allowed_provenance = {
        "chroma_key",
        "rembg_u2net_fallback",
        "rembg_u2net_grooming_pose_gate",
        "rembg_then_lab_warm_cast_recovery",
        GROOMING_CANONICAL_SEED_BOUND_MATTE_ALPHA_PROVENANCE,
    }
    if alpha_provenance not in allowed_provenance or not cutout.is_file():
        return fur_surface_gate_failure(
            action,
            "fur_surface_reliable_alpha_cutout_unavailable",
            baseline=baseline,
            candidate=cutout,
            alpha_provenance=alpha_provenance,
        )
    try:
        with Image.open(cutout) as loaded:
            if "A" not in loaded.getbands():
                raise ValueError("cutout_has_no_alpha_channel")
            alpha = loaded.convert("RGBA").getchannel("A")
            alpha_minimum, alpha_maximum = alpha.getextrema()
            if alpha_minimum == 255 or alpha_maximum == 0:
                raise ValueError("cutout_alpha_is_not_a_usable_subject_matte")
        result = evaluate_fur_surface(
            cutout,
            baseline_path,
            baseline_is_validated=not uploaded_identity_absolute,
            accept_absolute_contract=uploaded_identity_absolute,
        )
        result = json.loads(json.dumps(result, ensure_ascii=False, allow_nan=False))
    except Exception as exc:  # noqa: BLE001 - malformed CV inputs fail closed
        return fur_surface_gate_failure(
            action,
            "fur_surface_gate_evaluation_failed",
            baseline=baseline,
            candidate=cutout,
            alpha_provenance=alpha_provenance,
            error=f"{type(exc).__name__}: {exc}",
        )
    result["available"] = True
    result["evaluated"] = True
    result["status"] = "passed" if result.get("acceptable") is True else "rejected"
    result["penalty"] = 0.0 if result.get("acceptable") is True else 1000.0
    result["gate"] = {
        "policy": FUR_SURFACE_GATE_POLICY,
        "gate_version": FUR_SURFACE_GATE_VERSION,
        "threshold_version": FUR_SURFACE_THRESHOLD_VERSION,
        "action": action,
        "reliable_alpha_required": True,
        "alpha_provenance": alpha_provenance,
        "candidate_path": str(cutout),
        "candidate_sha256": file_sha256(cutout),
        "candidate_comparison_mode": (
            UPLOADED_IDENTITY_SURFACE_COMPARISON_MODE
            if uploaded_identity_absolute
            else "validated_profile_relative_baseline"
        ),
        "profile_pixels_used_for_candidate_appearance_validation": False
        if uploaded_identity_absolute
        else True,
        "baseline_path": str(baseline_path) if baseline_path is not None else None,
        "baseline_snapshot_sha256": (
            baseline.get("snapshot_sha256") if isinstance(baseline, dict) else None
        ),
        "baseline_provenance": (
            baseline.get("provenance") if isinstance(baseline, dict) else None
        ),
        "baseline_provenance_sha256": (
            baseline.get("provenance_sha256") if isinstance(baseline, dict) else None
        ),
    }
    if result.get("acceptable") is not True and not result.get("issues"):
        result["issues"] = ["fur_surface_candidate_not_proven_acceptable"]
    return result


def apply_canonical_grooming_direct_surface_nonexpansion_gate(
    surface_gate: dict[str, Any],
    *,
    pose_gate: dict[str, Any] | None,
    identity_palette_quality: dict[str, Any] | None,
    attempt: int,
    canonical_reference_path: Path,
    canonical_reference_sha256: str,
) -> dict[str, Any]:
    """Recover one inherited brush issue only when canonical debt shrinks.

    This is not an alternate global fur threshold. It applies only after the
    strict direct-canonical raised-paw recovery and identity-palette pass. The
    transparent canonical is digest-bound, independently analyzed with the
    unchanged absolute contract, and every relevant candidate brush/line
    metric must remain bounded and non-expanding against that canonical.
    """
    _sync_local_video_runtime(globals(), {"apply_canonical_grooming_direct_surface_nonexpansion_gate"})

    result = dict(surface_gate)

    def number(value: Any) -> float | None:
        try:
            converted = float(value)
        except (TypeError, ValueError):
            return None
        return converted if np.isfinite(converted) else None

    def texture_metrics(audit: dict[str, Any]) -> dict[str, Any]:
        candidate = audit.get("candidate")
        candidate = candidate if isinstance(candidate, dict) else {}
        metrics = candidate.get("metrics")
        metrics = metrics if isinstance(metrics, dict) else {}
        texture = metrics.get("texture")
        return texture if isinstance(texture, dict) else {}

    def hough_metrics(texture: dict[str, Any]) -> dict[str, Any]:
        hough = texture.get("hough")
        return hough if isinstance(hough, dict) else {}

    gate = surface_gate.get("gate")
    gate = gate if isinstance(gate, dict) else {}
    thresholds = surface_gate.get("thresholds")
    thresholds = thresholds if isinstance(thresholds, dict) else {}
    candidate_record = surface_gate.get("candidate")
    candidate_record = candidate_record if isinstance(candidate_record, dict) else {}
    candidate_input = candidate_record.get("input")
    candidate_input = candidate_input if isinstance(candidate_input, dict) else {}
    candidate_texture = texture_metrics(surface_gate)
    candidate_hough = hough_metrics(candidate_texture)
    pose_gate = pose_gate if isinstance(pose_gate, dict) else {}
    pose_recovery = pose_gate.get("canonical_direct_structural_edge_evidence")
    pose_recovery = pose_recovery if isinstance(pose_recovery, dict) else {}
    identity_palette_quality = (
        identity_palette_quality if isinstance(identity_palette_quality, dict) else {}
    )

    expected_digest = str(canonical_reference_sha256 or "").strip().lower()
    canonical_result: dict[str, Any] = {}
    canonical_error: str | None = None
    canonical_digest_valid = False
    try:
        canonical_digest_valid = bool(
            canonical_reference_path.is_file()
            and re.fullmatch(r"[0-9a-f]{64}", expected_digest) is not None
            and file_sha256(canonical_reference_path) == expected_digest
        )
        if canonical_digest_valid:
            canonical_result = evaluate_fur_surface(
                canonical_reference_path,
                accept_absolute_contract=True,
            )
            canonical_result = json.loads(
                json.dumps(
                    canonical_result,
                    ensure_ascii=False,
                    allow_nan=False,
                )
            )
    except Exception as exc:  # noqa: BLE001 - recovery must fail closed
        canonical_error = f"{type(exc).__name__}: {str(exc)[:1000]}"
        canonical_result = {}

    canonical_record = canonical_result.get("candidate")
    canonical_record = canonical_record if isinstance(canonical_record, dict) else {}
    canonical_input = canonical_record.get("input")
    canonical_input = canonical_input if isinstance(canonical_input, dict) else {}
    canonical_texture = texture_metrics(canonical_result)
    canonical_hough = hough_metrics(canonical_texture)
    raw_issues = [str(issue) for issue in surface_gate.get("issues", [])]
    raw_absolute_issues = [
        str(issue) for issue in surface_gate.get("candidate_absolute_issues", [])
    ]
    attempt_number = number(attempt)
    canonical_issues = [str(issue) for issue in canonical_result.get("issues", [])]
    allowed_canonical_debt = {
        "brush_texture_coherence",
        "coherent_long_fur_energy",
    }
    default_thresholds = asdict(FurSurfaceThresholds())

    candidate_path = Path(str(gate.get("candidate_path") or ""))
    candidate_sha256 = str(gate.get("candidate_sha256") or "").lower()
    candidate_digest_valid = False
    try:
        candidate_digest_valid = bool(
            candidate_path.is_file()
            and re.fullmatch(r"[0-9a-f]{64}", candidate_sha256) is not None
            and file_sha256(candidate_path) == candidate_sha256
            and Path(str(pose_gate.get("candidate") or "")).resolve()
            == candidate_path.resolve()
        )
    except (OSError, RuntimeError, ValueError):
        candidate_digest_valid = False

    candidate_coarse = number(candidate_texture.get("coarse_gradient_p90"))
    candidate_coherence_mean = number(
        candidate_texture.get("structure_tensor_coherence_mean")
    )
    candidate_coherent_fraction = number(
        candidate_texture.get("structure_tensor_coherent_fraction")
    )
    candidate_orientation = number(
        candidate_texture.get("orientation_persistence_fraction")
    )
    candidate_fine_radial = number(candidate_texture.get("fine_radial_brush_fraction"))
    candidate_fine_radial_energy = number(
        candidate_texture.get("fine_radial_brush_energy")
    )
    candidate_coarse_radial = number(
        candidate_texture.get("coarse_radial_brush_fraction")
    )
    candidate_coarse_radial_energy = number(
        candidate_texture.get("coarse_radial_brush_energy")
    )
    candidate_long = number(candidate_texture.get("coherent_long_line_energy"))
    candidate_hough_radial = number(candidate_hough.get("radial_line_fraction"))
    coarse_gradient_max = number(thresholds.get("coarse_gradient_p90_max"))
    radial_brush_max = number(thresholds.get("radial_brush_fraction_max"))
    hough_radial_max = number(thresholds.get("hough_radial_line_fraction_max"))
    coherent_long_max = number(thresholds.get("coherent_long_line_energy_max"))
    canonical_coarse = number(canonical_texture.get("coarse_gradient_p90"))
    canonical_coherence_mean = number(
        canonical_texture.get("structure_tensor_coherence_mean")
    )
    canonical_coherent_fraction = number(
        canonical_texture.get("structure_tensor_coherent_fraction")
    )
    canonical_orientation = number(
        canonical_texture.get("orientation_persistence_fraction")
    )
    canonical_fine_radial = number(canonical_texture.get("fine_radial_brush_fraction"))
    canonical_fine_radial_energy = number(
        canonical_texture.get("fine_radial_brush_energy")
    )
    canonical_coarse_radial = number(
        canonical_texture.get("coarse_radial_brush_fraction")
    )
    canonical_coarse_radial_energy = number(
        canonical_texture.get("coarse_radial_brush_energy")
    )
    canonical_long = number(canonical_texture.get("coherent_long_line_energy"))
    canonical_hough_radial = number(canonical_hough.get("radial_line_fraction"))

    checks = {
        "surface_gate_contract_valid": bool(
            surface_gate.get("available") is True
            and surface_gate.get("evaluated") is True
            and surface_gate.get("acceptable") is not True
            and surface_gate.get("status") == "rejected"
            and gate.get("policy") == FUR_SURFACE_GATE_POLICY
            and gate.get("gate_version") == FUR_SURFACE_GATE_VERSION
            and gate.get("threshold_version") == FUR_SURFACE_THRESHOLD_VERSION
            and gate.get("action") == "grooming"
            and gate.get("candidate_comparison_mode")
            == UPLOADED_IDENTITY_SURFACE_COMPARISON_MODE
            and gate.get("profile_pixels_used_for_candidate_appearance_validation")
            is False
            and candidate_input.get("supported_contract") is True
            and candidate_input.get("foreground_method") == "alpha"
            and candidate_digest_valid
        ),
        "global_surface_thresholds_unchanged": bool(
            all(
                number(thresholds.get(name)) == number(default_thresholds.get(name))
                for name in (
                    "coarse_gradient_p90_max",
                    "coarse_coherent_fraction_max",
                    "radial_brush_fraction_max",
                    "hough_radial_line_fraction_max",
                    "coherent_long_line_energy_max",
                )
            )
        ),
        "raw_brush_issue_is_exclusive": bool(
            raw_issues == ["brush_texture_coherence"]
            and raw_absolute_issues == ["brush_texture_coherence"]
            and surface_gate.get("relative_issues") == []
            and surface_gate.get("baseline_issues") == []
        ),
        "strict_pose_recovery_bound_to_candidate": bool(
            pose_gate.get("acceptable") is True
            and pose_gate.get("verdict")
            == "accepted_canonical_direct_organic_foreleg_structural_evidence"
            and pose_recovery.get("policy")
            == GROOMING_CANONICAL_DIRECT_STRUCTURAL_EDGE_EVIDENCE_POLICY
            and pose_recovery.get("accepted") is True
            and attempt_number is not None
            and attempt_number.is_integer()
            and number(pose_recovery.get("attempt")) == attempt_number
            and 1
            <= attempt_number
            <= GROOMING_CANONICAL_DIRECT_EDGE_EVIDENCE_MAX_ATTEMPT
        ),
        "identity_palette_independently_passed": bool(
            identity_palette_quality.get("available") is True
            and identity_palette_quality.get("acceptable") is True
            and identity_palette_quality.get("issues") == []
        ),
        "canonical_reference_digest_valid": canonical_digest_valid,
        "canonical_reference_contract_reliable": bool(
            canonical_record.get("valid") is True
            and canonical_input.get("supported_contract") is True
            and canonical_input.get("foreground_method") == "alpha"
        ),
        "candidate_issue_already_present_in_bounded_canonical": bool(
            canonical_issues
            and set(canonical_issues).issubset(allowed_canonical_debt)
            and set(raw_absolute_issues).issubset(set(canonical_issues))
        ),
        "candidate_surface_absolute_bounds_passed": bool(
            candidate_coarse is not None
            and coarse_gradient_max is not None
            and candidate_coarse <= coarse_gradient_max
            and candidate_coherence_mean is not None
            and candidate_coherence_mean
            <= GROOMING_CANONICAL_DIRECT_SURFACE_COHERENCE_MEAN_MAX
            and candidate_coherent_fraction is not None
            and candidate_coherent_fraction
            <= GROOMING_CANONICAL_DIRECT_SURFACE_COHERENT_FRACTION_MAX
            and candidate_orientation is not None
            and candidate_orientation
            <= GROOMING_CANONICAL_DIRECT_SURFACE_ORIENTATION_MAX
            and candidate_fine_radial is not None
            and radial_brush_max is not None
            and candidate_fine_radial <= radial_brush_max
            and candidate_hough_radial is not None
            and hough_radial_max is not None
            and candidate_hough_radial <= hough_radial_max
            and candidate_long is not None
            and coherent_long_max is not None
            and candidate_long <= coherent_long_max
        ),
        "canonical_surface_debt_bounded": bool(
            canonical_coarse is not None
            and coarse_gradient_max is not None
            and canonical_coarse <= coarse_gradient_max
            and canonical_coherence_mean is not None
            and canonical_coherence_mean
            <= GROOMING_CANONICAL_DIRECT_SURFACE_COHERENCE_MEAN_MAX
            and canonical_coherent_fraction is not None
            and canonical_coherent_fraction
            <= GROOMING_CANONICAL_DIRECT_SURFACE_COHERENT_FRACTION_MAX
            and canonical_orientation is not None
            and canonical_orientation
            <= GROOMING_CANONICAL_DIRECT_SURFACE_ORIENTATION_MAX
            and canonical_fine_radial is not None
            and radial_brush_max is not None
            and canonical_fine_radial <= radial_brush_max
            and canonical_hough_radial is not None
            and hough_radial_max is not None
            and canonical_hough_radial <= hough_radial_max
            and canonical_long is not None
            and canonical_long
            <= GROOMING_CANONICAL_DIRECT_SURFACE_CANONICAL_LONG_ENERGY_MAX
        ),
        "canonical_surface_debt_not_expanded": bool(
            candidate_coarse is not None
            and canonical_coarse is not None
            and candidate_coarse <= canonical_coarse
            and candidate_coherence_mean is not None
            and canonical_coherence_mean is not None
            and candidate_coherence_mean
            <= canonical_coherence_mean
            + GROOMING_CANONICAL_DIRECT_SURFACE_COHERENCE_MEAN_DELTA_MAX
            and candidate_coherent_fraction is not None
            and canonical_coherent_fraction is not None
            and candidate_coherent_fraction
            <= canonical_coherent_fraction
            + GROOMING_CANONICAL_DIRECT_SURFACE_COHERENT_FRACTION_DELTA_MAX
            and candidate_orientation is not None
            and canonical_orientation is not None
            and candidate_orientation
            <= canonical_orientation
            + GROOMING_CANONICAL_DIRECT_SURFACE_ORIENTATION_DELTA_MAX
            and candidate_fine_radial is not None
            and canonical_fine_radial is not None
            and candidate_fine_radial <= canonical_fine_radial
            and candidate_fine_radial_energy is not None
            and canonical_fine_radial_energy is not None
            and candidate_fine_radial_energy <= canonical_fine_radial_energy
            and candidate_coarse_radial is not None
            and canonical_coarse_radial is not None
            and candidate_coarse_radial <= canonical_coarse_radial
            and candidate_coarse_radial_energy is not None
            and canonical_coarse_radial_energy is not None
            and candidate_coarse_radial_energy <= canonical_coarse_radial_energy
            and candidate_long is not None
            and canonical_long is not None
            and candidate_long <= canonical_long
            and candidate_hough_radial is not None
            and canonical_hough_radial is not None
            and candidate_hough_radial
            <= canonical_hough_radial
            + GROOMING_CANONICAL_DIRECT_SURFACE_HOUGH_RADIAL_DELTA_MAX
        ),
    }
    accepted = bool(all(checks.values()) and canonical_error is None)
    recovery = {
        "policy": GROOMING_CANONICAL_DIRECT_SURFACE_NONEXPANSION_POLICY,
        "evaluated": True,
        "accepted": accepted,
        "fail_closed": True,
        "attempt": (
            int(attempt_number)
            if attempt_number is not None and attempt_number.is_integer()
            else None
        ),
        "global_surface_thresholds_changed": False,
        "raw_issues": raw_issues,
        "canonical_reference": {
            "path": str(canonical_reference_path),
            "sha256": expected_digest or None,
            "issues": canonical_issues,
            "error": canonical_error,
        },
        "candidate_metrics": {
            "coarse_gradient_p90": candidate_coarse,
            "coherence_mean": candidate_coherence_mean,
            "coherent_fraction": candidate_coherent_fraction,
            "orientation_persistence_fraction": candidate_orientation,
            "fine_radial_brush_fraction": candidate_fine_radial,
            "fine_radial_brush_energy": candidate_fine_radial_energy,
            "coarse_radial_brush_fraction": candidate_coarse_radial,
            "coarse_radial_brush_energy": candidate_coarse_radial_energy,
            "coherent_long_line_energy": candidate_long,
            "hough_radial_line_fraction": candidate_hough_radial,
        },
        "canonical_metrics": {
            "coarse_gradient_p90": canonical_coarse,
            "coherence_mean": canonical_coherence_mean,
            "coherent_fraction": canonical_coherent_fraction,
            "orientation_persistence_fraction": canonical_orientation,
            "fine_radial_brush_fraction": canonical_fine_radial,
            "fine_radial_brush_energy": canonical_fine_radial_energy,
            "coarse_radial_brush_fraction": canonical_coarse_radial,
            "coarse_radial_brush_energy": canonical_coarse_radial_energy,
            "coherent_long_line_energy": canonical_long,
            "hough_radial_line_fraction": canonical_hough_radial,
        },
        "independent_final_video_fur_audit_still_required": True,
        "checks": checks,
    }
    result["canonical_direct_surface_nonexpansion"] = recovery
    if not accepted:
        return result

    result.update(
        raw_acceptable=surface_gate.get("acceptable"),
        raw_verdict=surface_gate.get("verdict"),
        raw_status=surface_gate.get("status"),
        raw_issues=raw_issues,
        raw_candidate_absolute_issues=raw_absolute_issues,
        raw_penalty=surface_gate.get("penalty"),
        acceptable=True,
        verdict="accepted_canonical_direct_surface_debt_nonexpansion",
        status="passed",
        issues=[],
        candidate_absolute_issues=[],
        penalty=0.0,
    )
    result["warnings"] = list(
        dict.fromkeys(
            [
                *[str(item) for item in surface_gate.get("warnings", [])],
                "canonical_brush_texture_debt_accepted_only_when_bounded_"
                "and_nonexpanding",
            ]
        )
    )
    result["gate"] = {
        **gate,
        "canonical_direct_surface_nonexpansion_policy": (
            GROOMING_CANONICAL_DIRECT_SURFACE_NONEXPANSION_POLICY
        ),
        "canonical_direct_surface_nonexpansion_accepted": True,
        "global_surface_thresholds_changed": False,
    }
    return result


def select_final_video_fur_surface_comparison_scope(
    *,
    action: str,
    action_config: dict[str, Any],
    output_format: str | None,
    reference_policy: str | None,
    pose_metadata: dict[str, Any] | None,
) -> str:
    """Select the final audit baseline for a persisted Qwen action reference.

    Uploaded transparent routes must compare SCAIL frames with the exact Qwen
    action keyframe that entered SCAIL. Eating additionally requires the
    separated-bowl and later GIF contour contracts. Grooming instead requires
    the explicit no-prop mask contract; its unmasked contour comparison is
    already part of the validated-action v2 audit.
    """
    _sync_local_video_runtime(globals(), {"select_final_video_fur_surface_comparison_scope"})

    uploaded_qwen_transparent = bool(
        action_config.get("reference_kind") == "qwen_generated_upload"
        and output_format == "transparent_gif"
    )
    if not uploaded_qwen_transparent:
        return FINAL_VIDEO_PROFILE_FULL_SCOPE

    expected_reference_policy = FINAL_VIDEO_VALIDATED_ACTION_REFERENCE_POLICIES.get(
        action
    )
    if expected_reference_policy is None:
        raise RuntimeError(
            "Uploaded Qwen transparent route has no validated-action final "
            f"surface policy for {action!r}."
        )
    if reference_policy != expected_reference_policy:
        raise RuntimeError(
            f"Uploaded {action} transparent route has unexpected reference "
            f"policy {reference_policy!r}; expected "
            f"{expected_reference_policy!r}."
        )
    if not isinstance(pose_metadata, dict):
        raise RuntimeError(
            f"Uploaded {action} transparent route lacks SCAIL pose metadata "
            "required by final surface audit v2."
        )

    scene_prop_separated = pose_metadata.get("scene_prop_separated")
    if action == "eating":
        post_animation_contour_gate = action_config.get("fur_silhouette_gate")
        if (
            scene_prop_separated is not True
            or not isinstance(post_animation_contour_gate, dict)
            or post_animation_contour_gate.get("enabled") is not True
        ):
            raise RuntimeError(
                "Uploaded eating transparent route lacks the separated-prop "
                "and post-animation contour contracts required by final "
                "surface audit v2."
            )
    elif action == "grooming" and scene_prop_separated is not False:
        raise RuntimeError(
            "Uploaded grooming transparent route lacks the explicit no-prop "
            "identity-mask contract required by final surface audit v2."
        )

    return FINAL_VIDEO_VALIDATED_ACTION_SURFACE_SCOPE


def grooming_pose_gate_failure(
    issue: str,
    *,
    error: str | None = None,
) -> dict[str, Any]:
    """Return a JSON-safe, fail-closed grooming-pose verdict."""
    _sync_local_video_runtime(globals(), {"grooming_pose_gate_failure"})

    result: dict[str, Any] = {
        "policy": GROOMING_POSE_GATE_POLICY,
        "version": GROOMING_POSE_GATE_VERSION,
        "available": False,
        "acceptable": False,
        "issues": [issue],
        "warnings": [],
        "penalty": 1000.0,
    }
    if error:
        result["error"] = error[:1000]
    return result


def apply_masked_pet3d_grooming_composite_gate(
    pose_gate: dict[str, Any],
    fur_surface_gate: dict[str, Any] | None,
    repair_audit: dict[str, Any] | None,
) -> dict[str, Any]:
    """Accept only scoped edge/flat-paw ratio artifacts with independent evidence.

    The global grooming-pose thresholds remain unchanged. This recovery is
    available only to the final noise-masked Pet3D foreleg repair, when the
    raw pose evaluator reports no geometry, contact, or palette defect and the
    independent fur-surface gate passes its absolute contract. A high paw-
    texture ratio can recover only when its denominator is a nearly flat
    planted paw and the raised-paw texture remains bounded.
    """
    _sync_local_video_runtime(globals(), {"apply_masked_pet3d_grooming_composite_gate"})

    result = dict(pose_gate)

    def number(value: Any) -> float | None:
        try:
            converted = float(value)
        except (TypeError, ValueError):
            return None
        return converted if np.isfinite(converted) else None

    raw_issues = [str(issue) for issue in pose_gate.get("issues", [])]
    thresholds = pose_gate.get("thresholds")
    metrics = pose_gate.get("metrics")
    thresholds = thresholds if isinstance(thresholds, dict) else {}
    metrics = metrics if isinstance(metrics, dict) else {}
    chamfer = metrics.get("chamfer")
    continuous = metrics.get("continuous_edge")
    contact = metrics.get("paw_muzzle_contact")
    appearance = metrics.get("appearance_consistency")
    chamfer = chamfer if isinstance(chamfer, dict) else {}
    continuous = continuous if isinstance(continuous, dict) else {}
    contact = contact if isinstance(contact, dict) else {}
    appearance = appearance if isinstance(appearance, dict) else {}
    histogram = appearance.get("lab_histogram_bhattacharyya")
    histogram = histogram if isinstance(histogram, dict) else {}
    raised_appearance = appearance.get("raised")
    planted_appearance = appearance.get("planted")
    raised_appearance = raised_appearance if isinstance(raised_appearance, dict) else {}
    planted_appearance = (
        planted_appearance if isinstance(planted_appearance, dict) else {}
    )

    edge_density = number(metrics.get("raised_foreleg_edge_density"))
    edge_density_min = number(thresholds.get("edge_density_min"))
    edge_density_floor = (
        edge_density_min * GROOMING_QWEN_FORELEG_REPAIR_EDGE_DENSITY_RATIO_MIN
        if edge_density_min is not None
        else None
    )
    chamfer_coverage = number(chamfer.get("coverage_within_radius"))
    chamfer_coverage_min = number(thresholds.get("chamfer_coverage_min"))
    chamfer_distance = number(chamfer.get("mean_distance_px"))
    chamfer_distance_max = number(thresholds.get("chamfer_mean_distance_max"))
    continuous_span = number(continuous.get("span_fraction"))
    continuous_span_min = number(thresholds.get("continuous_edge_span_min"))
    contact_gap = number(contact.get("gap_fraction_of_bbox_height"))
    contact_gap_max = number(thresholds.get("paw_muzzle_gap_max"))
    lab_delta = number(appearance.get("lab_median_delta"))
    lab_delta_max = number(thresholds.get("lab_median_delta_max"))
    histogram_mean = number(histogram.get("mean"))
    histogram_max = number(thresholds.get("lab_histogram_distance_max"))
    texture_ratio = number(appearance.get("texture_gradient_p90_ratio"))
    texture_ratio_min = number(thresholds.get("texture_p90_ratio_min"))
    texture_ratio_max = number(thresholds.get("texture_p90_ratio_max"))
    raised_gradient_p90 = number(raised_appearance.get("gradient_p90"))
    planted_gradient_p90 = number(planted_appearance.get("gradient_p90"))

    repair_digest = (
        str(repair_audit.get("sampler_noise_mask_sha256") or "")
        if isinstance(repair_audit, dict)
        else ""
    )
    configured_repair_model = Path(QWEN_STYLE_LORA).name.casefold()
    pose_gate_contract_valid = bool(
        pose_gate.get("policy") == GROOMING_POSE_GATE_POLICY
        and pose_gate.get("version") == GROOMING_POSE_GATE_VERSION
        and edge_density_min == GROOMING_QWEN_FORELEG_REPAIR_GLOBAL_EDGE_DENSITY_MIN
    )
    repair_contract_valid = bool(
        isinstance(repair_audit, dict)
        and repair_audit.get("policy") == GROOMING_QWEN_FORELEG_REPAIR_POLICY
        and repair_audit.get("composite_gate_policy")
        == GROOMING_QWEN_FORELEG_REPAIR_COMPOSITE_GATE_POLICY
        and number(repair_audit.get("composite_edge_density_ratio_min"))
        == GROOMING_QWEN_FORELEG_REPAIR_EDGE_DENSITY_RATIO_MIN
        and number(repair_audit.get("composite_planted_paw_gradient_p90_max"))
        == GROOMING_QWEN_FORELEG_REPAIR_PLANTED_PAW_GRADIENT_P90_MAX
        and number(repair_audit.get("composite_raised_paw_gradient_p90_max"))
        == GROOMING_QWEN_FORELEG_REPAIR_RAISED_PAW_GRADIENT_P90_MAX
        and repair_audit.get("global_pose_threshold_changed") is False
        and repair_audit.get("localized_style_policy")
        == GROOMING_QWEN_FORELEG_REPAIR_STYLE_POLICY
        and bool(QWEN_STYLE_LORA)
        and "pet3d" in configured_repair_model
        and repair_audit.get("localized_style_lora") == QWEN_STYLE_LORA
        and number(repair_audit.get("localized_style_strength"))
        == GROOMING_QWEN_FORELEG_REPAIR_STYLE_STRENGTH
        and repair_audit.get("localized_style_scope")
        == (
            "sampler_noise_mask_only; image2_remains_identity_"
            "palette_marking_and_coat_length_authority"
        )
        and re.fullmatch(r"[0-9a-f]{64}", repair_digest) is not None
    )
    fur_gate_audit = (
        fur_surface_gate.get("gate") if isinstance(fur_surface_gate, dict) else None
    )
    fur_gate_audit = fur_gate_audit if isinstance(fur_gate_audit, dict) else {}
    fur_candidate_digest = str(fur_gate_audit.get("candidate_sha256") or "")
    fur_contract_valid = bool(
        isinstance(fur_surface_gate, dict)
        and fur_surface_gate.get("available") is True
        and fur_surface_gate.get("evaluated") is True
        and fur_surface_gate.get("status") == "passed"
        and fur_surface_gate.get("acceptable") is True
        and not fur_surface_gate.get("issues")
        and fur_gate_audit.get("policy") == FUR_SURFACE_GATE_POLICY
        and fur_gate_audit.get("gate_version") == FUR_SURFACE_GATE_VERSION
        and fur_gate_audit.get("threshold_version") == FUR_SURFACE_THRESHOLD_VERSION
        and fur_gate_audit.get("action") == "grooming"
        and fur_gate_audit.get("reliable_alpha_required") is True
        and re.fullmatch(r"[0-9a-f]{64}", fur_candidate_digest) is not None
    )
    edge_issue = "raised_foreleg_edge_density_too_low"
    paw_texture_issue = "raised_paw_texture_inconsistent_with_planted_paw"
    raw_issue_contract_valid = bool(
        raw_issues == [edge_issue]
        or (len(raw_issues) == 2 and set(raw_issues) == {edge_issue, paw_texture_issue})
    )
    color_histogram_passed = bool(
        appearance.get("available") is True
        and lab_delta is not None
        and lab_delta_max is not None
        and lab_delta <= lab_delta_max
        and histogram_mean is not None
        and histogram_max is not None
        and histogram_mean <= histogram_max
    )
    standard_texture_ratio_passed = bool(
        texture_ratio is not None
        and texture_ratio_min is not None
        and texture_ratio_max is not None
        and texture_ratio_min <= texture_ratio <= texture_ratio_max
    )
    low_texture_planted_paw_rescue = bool(
        paw_texture_issue in raw_issues
        and texture_ratio is not None
        and texture_ratio_max is not None
        and texture_ratio > texture_ratio_max
        and raised_gradient_p90 is not None
        and raised_gradient_p90
        <= GROOMING_QWEN_FORELEG_REPAIR_RAISED_PAW_GRADIENT_P90_MAX
        and planted_gradient_p90 is not None
        and planted_gradient_p90
        <= GROOMING_QWEN_FORELEG_REPAIR_PLANTED_PAW_GRADIENT_P90_MAX
    )
    checks = {
        "global_pose_gate_contract_valid": pose_gate_contract_valid,
        "raw_issue_contract_valid": raw_issue_contract_valid,
        "edge_density_within_recovery_band": bool(
            edge_density is not None
            and edge_density_floor is not None
            and edge_density_min is not None
            and edge_density_floor <= edge_density < edge_density_min
        ),
        "chamfer_geometry_passed": bool(
            chamfer.get("available") is True
            and chamfer_coverage is not None
            and chamfer_coverage_min is not None
            and chamfer_coverage >= chamfer_coverage_min
            and chamfer_distance is not None
            and chamfer_distance_max is not None
            and chamfer_distance <= chamfer_distance_max
        ),
        "continuous_foreleg_passed": bool(
            continuous_span is not None
            and continuous_span_min is not None
            and continuous_span >= continuous_span_min
        ),
        "paw_muzzle_contact_passed": bool(
            contact.get("available") is True
            and contact_gap is not None
            and contact_gap_max is not None
            and contact_gap <= contact_gap_max
        ),
        "raised_planted_paw_color_histogram_passed": color_histogram_passed,
        "raised_planted_paw_texture_contract_passed": bool(
            standard_texture_ratio_passed or low_texture_planted_paw_rescue
        ),
        "independent_fur_surface_gate_passed": fur_contract_valid,
        "masked_pet3d_repair_contract_valid": repair_contract_valid,
    }
    accepted = bool(
        pose_gate.get("available") is True
        and pose_gate.get("acceptable") is not True
        and all(checks.values())
    )
    composite = {
        "policy": GROOMING_QWEN_FORELEG_REPAIR_COMPOSITE_GATE_POLICY,
        "evaluated": True,
        "accepted": accepted,
        "fail_closed": True,
        "global_pose_threshold_changed": False,
        "required_global_edge_density_min": (
            GROOMING_QWEN_FORELEG_REPAIR_GLOBAL_EDGE_DENSITY_MIN
        ),
        "required_edge_density_ratio": (
            GROOMING_QWEN_FORELEG_REPAIR_EDGE_DENSITY_RATIO_MIN
        ),
        "edge_density": edge_density,
        "edge_density_threshold": edge_density_min,
        "edge_density_floor": edge_density_floor,
        "allowed_raw_issue_sets": [
            [edge_issue],
            [edge_issue, paw_texture_issue],
        ],
        "paw_texture_evidence": {
            "texture_gradient_p90_ratio": texture_ratio,
            "standard_ratio_passed": standard_texture_ratio_passed,
            "raised_gradient_p90": raised_gradient_p90,
            "raised_gradient_p90_max": (
                GROOMING_QWEN_FORELEG_REPAIR_RAISED_PAW_GRADIENT_P90_MAX
            ),
            "planted_gradient_p90": planted_gradient_p90,
            "planted_gradient_p90_max": (
                GROOMING_QWEN_FORELEG_REPAIR_PLANTED_PAW_GRADIENT_P90_MAX
            ),
            "low_texture_planted_paw_rescue_used": (low_texture_planted_paw_rescue),
        },
        "repair_noise_mask_sha256": repair_digest or None,
        "checks": checks,
    }
    result["composite_recovery"] = composite
    if not accepted:
        return result

    result.update(
        raw_acceptable=pose_gate.get("acceptable"),
        raw_verdict=pose_gate.get("verdict"),
        raw_issues=raw_issues,
        raw_penalty=pose_gate.get("penalty"),
        acceptable=True,
        verdict="accepted_masked_pet3d_surface_composite",
        issues=[],
        penalty=0.0,
    )
    result["warnings"] = list(
        dict.fromkeys(
            [
                *[str(item) for item in pose_gate.get("warnings", [])],
                "raised_foreleg_edge_density_shortfall_accepted_only_by_"
                "masked_pet3d_surface_composite",
                *(
                    [
                        "raised_paw_texture_ratio_accepted_only_for_low_texture_"
                        "planted_paw_with_independent_fur_surface_pass"
                    ]
                    if low_texture_planted_paw_rescue
                    else []
                ),
            ]
        )
    )
    return result


def apply_canonical_grooming_direct_structural_edge_evidence_gate(
    pose_gate: dict[str, Any],
    *,
    attempt: int,
    canonical_action_reference: bool,
    canonical_completion_active: bool,
    canonical_full_pose_recovery_active: bool,
) -> dict[str, Any]:
    """Recover tightly bounded organic-paw evidence on direct attempts.

    The global raw and structural pose thresholds remain unchanged. Recovery
    either requires the historical narrow raw-edge shortfall with strict guide
    alignment, or a real-fur paw whose only failures are bounded guide offset.
    The second route requires substantially tighter raised/planted paw color,
    histogram and texture agreement. Identity-palette and fur-surface gates
    still run independently afterward.
    """
    _sync_local_video_runtime(globals(), {"apply_canonical_grooming_direct_structural_edge_evidence_gate"})

    result = dict(pose_gate)

    def number(value: Any) -> float | None:
        try:
            converted = float(value)
        except (TypeError, ValueError):
            return None
        return converted if np.isfinite(converted) else None

    thresholds = pose_gate.get("thresholds")
    metrics = pose_gate.get("metrics")
    thresholds = thresholds if isinstance(thresholds, dict) else {}
    metrics = metrics if isinstance(metrics, dict) else {}
    structural = metrics.get("raised_foreleg_structural_evidence")
    chamfer = metrics.get("chamfer")
    continuous = metrics.get("continuous_edge")
    contact = metrics.get("paw_muzzle_contact")
    appearance = metrics.get("appearance_consistency")
    structural = structural if isinstance(structural, dict) else {}
    chamfer = chamfer if isinstance(chamfer, dict) else {}
    continuous = continuous if isinstance(continuous, dict) else {}
    contact = contact if isinstance(contact, dict) else {}
    appearance = appearance if isinstance(appearance, dict) else {}
    histogram = appearance.get("lab_histogram_bhattacharyya")
    histogram = histogram if isinstance(histogram, dict) else {}

    edge_density = number(metrics.get("raised_foreleg_edge_density"))
    edge_density_min = number(thresholds.get("edge_density_min"))
    edge_density_floor = (
        edge_density_min * GROOMING_CANONICAL_DIRECT_EDGE_DENSITY_RATIO_MIN
        if edge_density_min is not None
        else None
    )
    structural_density = number(structural.get("edge_density"))
    structural_threshold = number(thresholds.get("structural_edge_density_min"))
    chamfer_coverage = number(chamfer.get("coverage_within_radius"))
    chamfer_mean = number(chamfer.get("mean_distance_px"))
    chamfer_p90 = number(chamfer.get("p90_distance_px"))
    attempt_number = number(attempt)
    raw_issues = [str(issue) for issue in pose_gate.get("issues", [])]
    checks = {
        "global_pose_gate_contract_valid": bool(
            pose_gate.get("policy") == GROOMING_POSE_GATE_POLICY
            and pose_gate.get("version") == GROOMING_POSE_GATE_VERSION
            and edge_density_min == GROOMING_QWEN_FORELEG_REPAIR_GLOBAL_EDGE_DENSITY_MIN
            and number(thresholds.get("structural_blur_sigma"))
            == GROOMING_POSE_STRUCTURAL_BLUR_SIGMA
            and structural_threshold == GROOMING_POSE_STRUCTURAL_EDGE_DENSITY_MIN
        ),
        "direct_canonical_attempt_contract_valid": bool(
            canonical_action_reference
            and attempt_number is not None
            and attempt_number.is_integer()
            and 1
            <= attempt_number
            <= GROOMING_CANONICAL_DIRECT_EDGE_EVIDENCE_MAX_ATTEMPT
            and canonical_completion_active is False
            and canonical_full_pose_recovery_active is False
            and "canonical_completion_edge_evidence" not in pose_gate
            and "composite_recovery" not in pose_gate
        ),
        "raw_edge_issue_is_exclusive": raw_issues
        == ["raised_foreleg_edge_density_too_low"],
        "raw_edge_density_within_antialias_band": bool(
            edge_density is not None
            and edge_density_floor is not None
            and edge_density_min is not None
            and edge_density_floor <= edge_density < edge_density_min
        ),
        "lowpass_organic_structure_strong": bool(
            structural.get("available") is True
            and structural.get("policy")
            == "gaussian_lowpass_organic_foreleg_structure_v1"
            and number(structural.get("blur_sigma"))
            == GROOMING_POSE_STRUCTURAL_BLUR_SIGMA
            and structural_density is not None
            and structural_density
            >= GROOMING_CANONICAL_DIRECT_STRUCTURAL_EDGE_DENSITY_MIN
            and structural_threshold is not None
            and structural_density >= structural_threshold
        ),
        "strict_organic_guide_alignment_passed": bool(
            chamfer.get("available") is True
            and chamfer_coverage is not None
            and chamfer_coverage >= GROOMING_CANONICAL_DIRECT_CHAMFER_COVERAGE_MIN
            and chamfer_mean is not None
            and chamfer_mean <= GROOMING_CANONICAL_DIRECT_CHAMFER_MEAN_DISTANCE_MAX
            and chamfer_p90 is not None
            and chamfer_p90 <= GROOMING_CANONICAL_DIRECT_CHAMFER_P90_DISTANCE_MAX
        ),
        "continuous_foreleg_passed": bool(
            number(continuous.get("span_fraction")) is not None
            and number(thresholds.get("continuous_edge_span_min")) is not None
            and number(continuous.get("span_fraction"))
            >= number(thresholds.get("continuous_edge_span_min"))
        ),
        "paw_muzzle_contact_passed": bool(
            contact.get("available") is True
            and number(contact.get("gap_fraction_of_bbox_height")) is not None
            and number(thresholds.get("paw_muzzle_gap_max")) is not None
            and number(contact.get("gap_fraction_of_bbox_height"))
            <= number(thresholds.get("paw_muzzle_gap_max"))
        ),
        "paw_appearance_passed": bool(
            appearance.get("available") is True
            and number(appearance.get("lab_median_delta")) is not None
            and number(thresholds.get("lab_median_delta_max")) is not None
            and number(appearance.get("lab_median_delta"))
            <= number(thresholds.get("lab_median_delta_max"))
            and number(histogram.get("mean")) is not None
            and number(thresholds.get("lab_histogram_distance_max")) is not None
            and number(histogram.get("mean"))
            <= number(thresholds.get("lab_histogram_distance_max"))
            and number(appearance.get("texture_gradient_p90_ratio")) is not None
            and number(thresholds.get("texture_p90_ratio_min")) is not None
            and number(thresholds.get("texture_p90_ratio_max")) is not None
            and number(thresholds.get("texture_p90_ratio_min"))
            <= number(appearance.get("texture_gradient_p90_ratio"))
            <= number(thresholds.get("texture_p90_ratio_max"))
        ),
    }
    strict_accepted = bool(
        pose_gate.get("available") is True
        and pose_gate.get("acceptable") is not True
        and all(checks.values())
    )
    offguide_checks = {
        "global_pose_gate_contract_valid": checks[
            "global_pose_gate_contract_valid"
        ],
        "direct_canonical_attempt_contract_valid": checks[
            "direct_canonical_attempt_contract_valid"
        ],
        "only_bounded_chamfer_issues": bool(
            len(raw_issues) == 2
            and set(raw_issues)
            == {
                "guide_chamfer_coverage_too_low",
                "guide_chamfer_distance_too_large",
            }
        ),
        "raw_edge_density_passed": bool(
            edge_density is not None
            and edge_density_min is not None
            and edge_density >= edge_density_min
        ),
        "lowpass_organic_structure_bounded": bool(
            structural.get("available") is True
            and structural.get("policy")
            == "gaussian_lowpass_organic_foreleg_structure_v1"
            and number(structural.get("blur_sigma"))
            == GROOMING_POSE_STRUCTURAL_BLUR_SIGMA
            and structural_density is not None
            and structural_density
            >= GROOMING_CANONICAL_DIRECT_OFFGUIDE_STRUCTURAL_EDGE_DENSITY_MIN
        ),
        "bounded_guide_offset": bool(
            chamfer.get("available") is True
            and chamfer_coverage is not None
            and chamfer_coverage
            >= GROOMING_CANONICAL_DIRECT_OFFGUIDE_CHAMFER_COVERAGE_MIN
            and chamfer_mean is not None
            and chamfer_mean
            <= GROOMING_CANONICAL_DIRECT_OFFGUIDE_CHAMFER_MEAN_DISTANCE_MAX
            and chamfer_p90 is not None
            and chamfer_p90
            <= GROOMING_CANONICAL_DIRECT_OFFGUIDE_CHAMFER_P90_DISTANCE_MAX
        ),
        "continuous_foreleg_passed": checks["continuous_foreleg_passed"],
        "paw_muzzle_contact_passed": checks["paw_muzzle_contact_passed"],
        "raised_planted_paw_fur_match": bool(
            appearance.get("available") is True
            and number(appearance.get("lab_median_delta")) is not None
            and number(appearance.get("lab_median_delta"))
            <= GROOMING_CANONICAL_DIRECT_OFFGUIDE_PAW_LAB_DELTA_MAX
            and number(histogram.get("mean")) is not None
            and number(histogram.get("mean"))
            <= GROOMING_CANONICAL_DIRECT_OFFGUIDE_PAW_HISTOGRAM_MAX
            and number(appearance.get("texture_gradient_p90_ratio")) is not None
            and GROOMING_CANONICAL_DIRECT_OFFGUIDE_PAW_TEXTURE_RATIO_MIN
            <= number(appearance.get("texture_gradient_p90_ratio"))
            <= GROOMING_CANONICAL_DIRECT_OFFGUIDE_PAW_TEXTURE_RATIO_MAX
        ),
    }
    offguide_accepted = bool(
        pose_gate.get("available") is True
        and pose_gate.get("acceptable") is not True
        and all(offguide_checks.values())
    )
    aligned_checks = {
        "global_pose_gate_contract_valid": checks[
            "global_pose_gate_contract_valid"
        ],
        "direct_canonical_attempt_contract_valid": checks[
            "direct_canonical_attempt_contract_valid"
        ],
        "raw_pose_gate_passed_without_issue": bool(
            pose_gate.get("acceptable") is True and not raw_issues
        ),
        "raw_edge_density_passed": bool(
            edge_density is not None
            and edge_density_min is not None
            and edge_density >= edge_density_min
        ),
        "lowpass_organic_structure_strong": bool(
            structural.get("available") is True
            and structural.get("policy")
            == "gaussian_lowpass_organic_foreleg_structure_v1"
            and number(structural.get("blur_sigma"))
            == GROOMING_POSE_STRUCTURAL_BLUR_SIGMA
            and structural_density is not None
            and structural_density
            >= GROOMING_CANONICAL_DIRECT_ALIGNED_STRUCTURAL_EDGE_DENSITY_MIN
        ),
        "strict_organic_guide_alignment_passed": checks[
            "strict_organic_guide_alignment_passed"
        ],
        "continuous_foreleg_passed": checks["continuous_foreleg_passed"],
        "paw_muzzle_contact_passed": checks["paw_muzzle_contact_passed"],
        "raised_planted_paw_fur_match": bool(
            appearance.get("available") is True
            and number(appearance.get("lab_median_delta")) is not None
            and number(appearance.get("lab_median_delta"))
            <= GROOMING_CANONICAL_DIRECT_ALIGNED_PAW_LAB_DELTA_MAX
            and number(histogram.get("mean")) is not None
            and number(histogram.get("mean"))
            <= GROOMING_CANONICAL_DIRECT_ALIGNED_PAW_HISTOGRAM_MAX
            and number(appearance.get("texture_gradient_p90_ratio")) is not None
            and GROOMING_CANONICAL_DIRECT_ALIGNED_PAW_TEXTURE_RATIO_MIN
            <= number(appearance.get("texture_gradient_p90_ratio"))
            <= GROOMING_CANONICAL_DIRECT_ALIGNED_PAW_TEXTURE_RATIO_MAX
        ),
    }
    aligned_accepted = bool(
        pose_gate.get("available") is True and all(aligned_checks.values())
    )
    accepted = bool(strict_accepted or offguide_accepted or aligned_accepted)
    recovery = {
        "policy": (GROOMING_CANONICAL_DIRECT_STRUCTURAL_EDGE_EVIDENCE_POLICY),
        "evaluated": True,
        "accepted": accepted,
        "fail_closed": True,
        "global_pose_threshold_changed": False,
        "attempt": (
            int(attempt_number)
            if attempt_number is not None and attempt_number.is_integer()
            else None
        ),
        "maximum_attempt": (GROOMING_CANONICAL_DIRECT_EDGE_EVIDENCE_MAX_ATTEMPT),
        "required_global_edge_density_min": edge_density_min,
        "required_edge_density_ratio": (
            GROOMING_CANONICAL_DIRECT_EDGE_DENSITY_RATIO_MIN
        ),
        "edge_density": edge_density,
        "edge_density_floor": edge_density_floor,
        "required_structural_edge_density_min": (
            GROOMING_CANONICAL_DIRECT_STRUCTURAL_EDGE_DENSITY_MIN
        ),
        "structural_edge_density": structural_density,
        "strict_chamfer": {
            "coverage": chamfer_coverage,
            "coverage_min": (GROOMING_CANONICAL_DIRECT_CHAMFER_COVERAGE_MIN),
            "mean_distance_px": chamfer_mean,
            "mean_distance_max": (GROOMING_CANONICAL_DIRECT_CHAMFER_MEAN_DISTANCE_MAX),
            "p90_distance_px": chamfer_p90,
            "p90_distance_max": (GROOMING_CANONICAL_DIRECT_CHAMFER_P90_DISTANCE_MAX),
        },
        "strict_antialias_edge_evidence_accepted": strict_accepted,
        "offguide_organic_paw_evidence": {
            "accepted": offguide_accepted,
            "structural_edge_density_min": (
                GROOMING_CANONICAL_DIRECT_OFFGUIDE_STRUCTURAL_EDGE_DENSITY_MIN
            ),
            "chamfer_coverage_min": (
                GROOMING_CANONICAL_DIRECT_OFFGUIDE_CHAMFER_COVERAGE_MIN
            ),
            "chamfer_mean_distance_max": (
                GROOMING_CANONICAL_DIRECT_OFFGUIDE_CHAMFER_MEAN_DISTANCE_MAX
            ),
            "chamfer_p90_distance_max": (
                GROOMING_CANONICAL_DIRECT_OFFGUIDE_CHAMFER_P90_DISTANCE_MAX
            ),
            "paw_lab_delta_max": (
                GROOMING_CANONICAL_DIRECT_OFFGUIDE_PAW_LAB_DELTA_MAX
            ),
            "paw_histogram_max": (
                GROOMING_CANONICAL_DIRECT_OFFGUIDE_PAW_HISTOGRAM_MAX
            ),
            "paw_texture_ratio_range": [
                GROOMING_CANONICAL_DIRECT_OFFGUIDE_PAW_TEXTURE_RATIO_MIN,
                GROOMING_CANONICAL_DIRECT_OFFGUIDE_PAW_TEXTURE_RATIO_MAX,
            ],
            "checks": offguide_checks,
        },
        "aligned_organic_paw_evidence": {
            "accepted": aligned_accepted,
            "structural_edge_density_min": (
                GROOMING_CANONICAL_DIRECT_ALIGNED_STRUCTURAL_EDGE_DENSITY_MIN
            ),
            "paw_lab_delta_max": (
                GROOMING_CANONICAL_DIRECT_ALIGNED_PAW_LAB_DELTA_MAX
            ),
            "paw_histogram_max": (
                GROOMING_CANONICAL_DIRECT_ALIGNED_PAW_HISTOGRAM_MAX
            ),
            "paw_texture_ratio_range": [
                GROOMING_CANONICAL_DIRECT_ALIGNED_PAW_TEXTURE_RATIO_MIN,
                GROOMING_CANONICAL_DIRECT_ALIGNED_PAW_TEXTURE_RATIO_MAX,
            ],
            "checks": aligned_checks,
        },
        "identity_palette_gate_still_required": True,
        "independent_fur_surface_gate_still_required": True,
        "checks": checks,
    }
    result["canonical_direct_structural_edge_evidence"] = recovery
    if not accepted:
        return result

    result.update(
        raw_acceptable=pose_gate.get("acceptable"),
        raw_verdict=pose_gate.get("verdict"),
        raw_issues=raw_issues,
        raw_penalty=pose_gate.get("penalty"),
        acceptable=True,
        verdict=(
            "accepted_canonical_direct_organic_foreleg_structural_evidence"
            if strict_accepted
            else (
                "accepted_canonical_direct_offguide_organic_paw_evidence"
                if offguide_accepted
                else "accepted_canonical_direct_aligned_organic_paw_evidence"
            )
        ),
        issues=[],
        penalty=0.0,
    )
    result["warnings"] = list(
        dict.fromkeys(
            [
                *[str(item) for item in pose_gate.get("warnings", [])],
                (
                    "raised_foreleg_raw_edge_antialias_shortfall_accepted_only_"
                    "with_strict_lowpass_structure_and_guide_alignment"
                    if strict_accepted
                    else (
                        "bounded_guide_offset_accepted_only_with_strict_"
                        "raised_planted_paw_fur_match"
                        if offguide_accepted
                        else "aligned_pose_palette_recovery_eligible_only_with_"
                        "strict_raised_planted_paw_fur_match"
                    )
                ),
            ]
        )
    )
    return result


def apply_canonical_grooming_completion_edge_evidence_gate(
    pose_gate: dict[str, Any],
    repair_audit: dict[str, Any] | None,
) -> dict[str, Any]:
    """Recover only a sub-1.5% antialias edge-count shortfall.

    The global 0.11 edge-density threshold remains unchanged. This scoped
    verdict is available only to the final digest-bound canonical masked
    completion when edge density is the sole raw issue and every independent
    geometry, contact, paw-color, histogram, and texture measurement already
    passes its original threshold. The ordinary fur-surface gate still runs
    after this decision and can independently reject the candidate.
    """
    _sync_local_video_runtime(globals(), {"apply_canonical_grooming_completion_edge_evidence_gate"})

    result = dict(pose_gate)

    def number(value: Any) -> float | None:
        try:
            converted = float(value)
        except (TypeError, ValueError):
            return None
        return converted if np.isfinite(converted) else None

    thresholds = pose_gate.get("thresholds")
    metrics = pose_gate.get("metrics")
    thresholds = thresholds if isinstance(thresholds, dict) else {}
    metrics = metrics if isinstance(metrics, dict) else {}
    chamfer = metrics.get("chamfer")
    continuous = metrics.get("continuous_edge")
    contact = metrics.get("paw_muzzle_contact")
    appearance = metrics.get("appearance_consistency")
    chamfer = chamfer if isinstance(chamfer, dict) else {}
    continuous = continuous if isinstance(continuous, dict) else {}
    contact = contact if isinstance(contact, dict) else {}
    appearance = appearance if isinstance(appearance, dict) else {}
    histogram = appearance.get("lab_histogram_bhattacharyya")
    histogram = histogram if isinstance(histogram, dict) else {}

    edge_density = number(metrics.get("raised_foreleg_edge_density"))
    edge_density_min = number(thresholds.get("edge_density_min"))
    edge_density_floor = (
        edge_density_min * GROOMING_CANONICAL_COMPLETION_EDGE_DENSITY_RATIO_MIN
        if edge_density_min is not None
        else None
    )
    completion_canvas = (
        repair_audit.get("completion_canvas")
        if isinstance(repair_audit, dict)
        else None
    )
    completion_canvas = completion_canvas if isinstance(completion_canvas, dict) else {}
    mask_digest = (
        str(repair_audit.get("sampler_noise_mask_sha256") or "")
        if isinstance(repair_audit, dict)
        else ""
    )
    raw_issues = [str(issue) for issue in pose_gate.get("issues", [])]
    checks = {
        "global_pose_gate_contract_valid": bool(
            pose_gate.get("policy") == GROOMING_POSE_GATE_POLICY
            and pose_gate.get("version") == GROOMING_POSE_GATE_VERSION
            and edge_density_min == GROOMING_QWEN_FORELEG_REPAIR_GLOBAL_EDGE_DENSITY_MIN
        ),
        "raw_edge_issue_is_exclusive": raw_issues
        == ["raised_foreleg_edge_density_too_low"],
        "edge_density_within_antialias_band": bool(
            edge_density is not None
            and edge_density_floor is not None
            and edge_density_min is not None
            and edge_density_floor <= edge_density < edge_density_min
        ),
        "chamfer_geometry_passed": bool(
            chamfer.get("available") is True
            and number(chamfer.get("coverage_within_radius")) is not None
            and number(thresholds.get("chamfer_coverage_min")) is not None
            and number(chamfer.get("coverage_within_radius"))
            >= number(thresholds.get("chamfer_coverage_min"))
            and number(chamfer.get("mean_distance_px")) is not None
            and number(thresholds.get("chamfer_mean_distance_max")) is not None
            and number(chamfer.get("mean_distance_px"))
            <= number(thresholds.get("chamfer_mean_distance_max"))
        ),
        "continuous_foreleg_passed": bool(
            number(continuous.get("span_fraction")) is not None
            and number(thresholds.get("continuous_edge_span_min")) is not None
            and number(continuous.get("span_fraction"))
            >= number(thresholds.get("continuous_edge_span_min"))
        ),
        "paw_muzzle_contact_passed": bool(
            contact.get("available") is True
            and number(contact.get("gap_fraction_of_bbox_height")) is not None
            and number(thresholds.get("paw_muzzle_gap_max")) is not None
            and number(contact.get("gap_fraction_of_bbox_height"))
            <= number(thresholds.get("paw_muzzle_gap_max"))
        ),
        "paw_appearance_passed": bool(
            appearance.get("available") is True
            and number(appearance.get("lab_median_delta")) is not None
            and number(thresholds.get("lab_median_delta_max")) is not None
            and number(appearance.get("lab_median_delta"))
            <= number(thresholds.get("lab_median_delta_max"))
            and number(histogram.get("mean")) is not None
            and number(thresholds.get("lab_histogram_distance_max")) is not None
            and number(histogram.get("mean"))
            <= number(thresholds.get("lab_histogram_distance_max"))
            and number(appearance.get("texture_gradient_p90_ratio")) is not None
            and number(thresholds.get("texture_p90_ratio_min")) is not None
            and number(thresholds.get("texture_p90_ratio_max")) is not None
            and number(thresholds.get("texture_p90_ratio_min"))
            <= number(appearance.get("texture_gradient_p90_ratio"))
            <= number(thresholds.get("texture_p90_ratio_max"))
        ),
        "canonical_masked_completion_contract_valid": bool(
            isinstance(repair_audit, dict)
            and repair_audit.get("policy")
            == GROOMING_CANONICAL_FORELEG_COMPLETION_POLICY
            and repair_audit.get("production_canonical_masked_completion") is True
            and repair_audit.get("source_identity_palette_acceptable") is True
            and repair_audit.get("localized_style_policy")
            == "disabled_for_canonical_masked_completion"
            and repair_audit.get("localized_style_lora") is None
            and number(repair_audit.get("localized_style_strength")) == 0.0
            and repair_audit.get("composite_recovery_enabled") is False
            and repair_audit.get("global_pose_threshold_changed") is False
            and repair_audit.get("main_case_rgb_appearance_exposed_on_repair_attempt")
            is False
            and re.fullmatch(r"[0-9a-f]{64}", mask_digest) is not None
            and completion_canvas.get("policy")
            == GROOMING_CANONICAL_FORELEG_COMPLETION_POLICY
            and completion_canvas.get("semantic_pixels_exposed_to_qwen") is False
            and completion_canvas.get("identity_or_geometry_seed_inserted") is False
            and completion_canvas.get(
                "source_composite_preserved_before_masked_sampling"
            )
            is True
        ),
    }
    accepted = bool(
        pose_gate.get("available") is True
        and pose_gate.get("acceptable") is not True
        and all(checks.values())
    )
    recovery = {
        "policy": GROOMING_CANONICAL_COMPLETION_EDGE_EVIDENCE_POLICY,
        "evaluated": True,
        "accepted": accepted,
        "fail_closed": True,
        "global_pose_threshold_changed": False,
        "required_global_edge_density_min": edge_density_min,
        "required_edge_density_ratio": (
            GROOMING_CANONICAL_COMPLETION_EDGE_DENSITY_RATIO_MIN
        ),
        "edge_density": edge_density,
        "edge_density_floor": edge_density_floor,
        "repair_noise_mask_sha256": mask_digest or None,
        "independent_fur_surface_gate_still_required": True,
        "checks": checks,
    }
    result["canonical_completion_edge_evidence"] = recovery
    if not accepted:
        return result

    result.update(
        raw_acceptable=pose_gate.get("acceptable"),
        raw_verdict=pose_gate.get("verdict"),
        raw_issues=raw_issues,
        raw_penalty=pose_gate.get("penalty"),
        acceptable=True,
        verdict="accepted_canonical_masked_completion_edge_evidence",
        issues=[],
        penalty=0.0,
    )
    result["warnings"] = list(
        dict.fromkeys(
            [
                *[str(item) for item in pose_gate.get("warnings", [])],
                "raised_foreleg_edge_density_antialias_shortfall_accepted_"
                "only_by_canonical_masked_completion_evidence",
            ]
        )
    )
    return result


def resolve_main_case_grooming_pose_gate_metadata(
    job: dict[str, Any],
    guide_path: Path,
) -> tuple[Path, Path, Path | None]:
    """Resolve the frozen neutral guide for CV and repair-only anatomy use."""
    _sync_local_video_runtime(globals(), {"resolve_main_case_grooming_pose_gate_metadata"})

    metadata_by_action = job.get("pose_reference_metadata")
    action_metadata = (
        metadata_by_action.get("grooming")
        if isinstance(metadata_by_action, dict)
        else None
    )
    if not isinstance(action_metadata, dict):
        raise RuntimeError(
            "Main_case grooming pose gate requires persisted geometry metadata."
        )
    generator_metadata = action_metadata.get("generator_metadata")
    if not isinstance(generator_metadata, dict):
        raise RuntimeError("Main_case grooming pose gate requires generator metadata.")
    if (
        action_metadata.get("policy") != GROOMING_QWEN_POSE_REFERENCE_POLICY
        or action_metadata.get("generator_policy")
        != GROOMING_QWEN_GEOMETRY_GUIDE_POLICY
        or generator_metadata.get("policy") != GROOMING_QWEN_GEOMETRY_GUIDE_POLICY
    ):
        raise RuntimeError(
            "Main_case grooming pose gate received unsupported geometry metadata."
        )
    guide_sha256 = file_sha256(guide_path)
    persisted_guide_sha256 = str(action_metadata.get("guide_sha256") or "").strip()
    generator_guide_sha256 = str(generator_metadata.get("guide_sha256") or "").strip()
    if (
        not persisted_guide_sha256
        or persisted_guide_sha256 != guide_sha256
        or not generator_guide_sha256
        or generator_guide_sha256 != guide_sha256
    ):
        raise RuntimeError("Main_case grooming Qwen geometry-guide digest changed.")
    metadata_text = str(generator_metadata.get("metadata_path") or "").strip()
    metadata_path = (
        Path(metadata_text)
        if metadata_text
        else guide_path.with_name(guide_path.stem + "_metadata.json")
    )
    guide_parent = guide_path.resolve().parent
    if metadata_path.resolve().parent != guide_parent or not metadata_path.is_file():
        raise RuntimeError("Main_case grooming Qwen metadata file is missing or moved.")
    try:
        metadata_file = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Main_case grooming Qwen metadata file is invalid.") from exc
    if (
        not isinstance(metadata_file, dict)
        or metadata_file.get("policy") != GROOMING_QWEN_GEOMETRY_GUIDE_POLICY
        or metadata_file.get("guide_sha256") != guide_sha256
    ):
        raise RuntimeError("Main_case grooming Qwen metadata contract is invalid.")

    qwen_metadata = metadata_file.get("qwen_guide")
    if not isinstance(qwen_metadata, dict):
        raise RuntimeError("Main_case grooming Qwen guide metadata is missing.")
    structural_negative_space = qwen_metadata.get("structural_negative_space")
    synthetic_foreleg = qwen_metadata.get("synthetic_foreleg_semantic")
    qwen_path_text = str(qwen_metadata.get("path") or "").strip()
    production_open_channel = bool(
        qwen_metadata.get("role") == "image2_action_geometry_guide"
        and qwen_metadata.get("foreground_rgb")
        == list(GROOMING_QWEN_GUIDE_FOREGROUND_RGB)
        and qwen_metadata.get("foreground_palette_rgb")
        == [list(GROOMING_QWEN_GUIDE_FOREGROUND_RGB)]
        and qwen_metadata.get("unique_foreground_rgb_count") == 1
        and qwen_metadata.get("internal_luma_variation") is False
        and qwen_metadata.get("appearance_bearing_internal_luma_variation") is False
        and qwen_metadata.get("main_case_appearance_exposed_to_qwen") is False
        and synthetic_foreleg is None
        and isinstance(structural_negative_space, dict)
        and structural_negative_space.get("method")
        == GROOMING_QWEN_NEGATIVE_SPACE_METHOD
        and structural_negative_space.get("curve_closed") is False
        and structural_negative_space.get("paw_endpoint_open") is True
        and structural_negative_space.get("shoulder_endpoint_open") is True
        and structural_negative_space.get("foreleg_remains_attached_at_shoulder")
        is True
        and structural_negative_space.get("paw_muzzle_contact_remains_connected")
        is True
        and structural_negative_space.get("source_rgb_or_luma_sampled") is False
        and structural_negative_space.get("appearance_bearing") is False
        and structural_negative_space.get("adds_foreground_color") is False
        and 300 <= int(structural_negative_space.get("pixels") or 0) <= 1800
        and 7 <= int(structural_negative_space.get("line_width") or 0) <= 11
        and isinstance(structural_negative_space.get("points"), list)
        and len(structural_negative_space.get("points")) == 6
        and qwen_metadata.get("closed_foreleg_capsule_exposed") is False
        and qwen_metadata.get("internal_green_separator_exposed") is True
        and qwen_metadata.get("raised_foreleg_validation_role")
        == "public_open_curve_geometry_plus_private_cv"
    )
    if (
        not qwen_path_text
        or Path(qwen_path_text).resolve() != guide_path.resolve()
        or qwen_metadata.get("sha256") != guide_sha256
    ):
        raise RuntimeError("Main_case grooming Qwen guide path or digest changed.")
    if not production_open_channel and (
        qwen_metadata.get("role") != "image1_action_geometry_sampler"
        or qwen_metadata.get("foreground_rgb")
        != list(GROOMING_QWEN_GUIDE_FOREGROUND_RGB)
        or qwen_metadata.get("foreground_palette_rgb")
        != [
            list(GROOMING_QWEN_GUIDE_FOREGROUND_RGB),
            list(GROOMING_QWEN_GUIDE_FORELEG_RGB),
        ]
        or qwen_metadata.get("unique_foreground_rgb_count") != 2
        or qwen_metadata.get("internal_luma_variation") is not True
        or qwen_metadata.get("appearance_bearing_internal_luma_variation") is not False
        or qwen_metadata.get("main_case_appearance_exposed_to_qwen") is not False
        or not isinstance(synthetic_foreleg, dict)
        or synthetic_foreleg.get("method") != GROOMING_QWEN_FORELEG_SEMANTIC_METHOD
        or synthetic_foreleg.get("rgb") != list(GROOMING_QWEN_GUIDE_FORELEG_RGB)
        or synthetic_foreleg.get("source_rgb_or_luma_sampled") is not False
        or synthetic_foreleg.get("appearance_bearing") is not False
        or int(synthetic_foreleg.get("pixels") or 0) < 650
        or int(synthetic_foreleg.get("pixels") or 0) > 6000
        or int(synthetic_foreleg.get("pixels_before_separator") or 0)
        < int(synthetic_foreleg.get("pixels") or 0)
        or not isinstance(structural_negative_space, dict)
        or structural_negative_space.get("method")
        != GROOMING_QWEN_NEGATIVE_SPACE_METHOD
        or structural_negative_space.get("appearance_bearing") is not False
        or structural_negative_space.get("adds_foreground_color") is not False
        or int(structural_negative_space.get("pixels") or 0) < 80
        or int(structural_negative_space.get("pixels") or 0) > 1600
    ):
        raise RuntimeError("Main_case grooming Qwen guide is not appearance-isolated.")
    with Image.open(guide_path) as loaded:
        qwen_rgb = np.asarray(loaded.convert("RGB"), dtype=np.uint8)
    qwen_foreground = ~np.all(qwen_rgb == (0, 255, 0), axis=2)
    unique_qwen_foreground = np.unique(qwen_rgb[qwen_foreground], axis=0)
    expected_foreground_palette = np.asarray(
        (
            [GROOMING_QWEN_GUIDE_FOREGROUND_RGB]
            if production_open_channel
            else [
                GROOMING_QWEN_GUIDE_FOREGROUND_RGB,
                GROOMING_QWEN_GUIDE_FORELEG_RGB,
            ]
        ),
        dtype=np.uint8,
    )
    qwen_semantic_foreleg_pixels = int(
        np.count_nonzero(
            np.all(
                qwen_rgb == np.asarray(GROOMING_QWEN_GUIDE_FORELEG_RGB, dtype=np.uint8),
                axis=2,
            )
        )
    )
    if (
        qwen_rgb.shape != (512, 512, 3)
        or not np.any(qwen_foreground)
        or unique_qwen_foreground.shape != expected_foreground_palette.shape
        or not np.array_equal(
            unique_qwen_foreground,
            expected_foreground_palette,
        )
        or (production_open_channel and qwen_semantic_foreleg_pixels != 0)
        or (
            not production_open_channel
            and qwen_semantic_foreleg_pixels
            != int(synthetic_foreleg.get("pixels") or 0)
        )
    ):
        raise RuntimeError(
            "Main_case grooming Qwen guide violates its synthetic anatomy palette."
        )

    private_gate = metadata_file.get("private_pose_gate")
    if not isinstance(private_gate, dict):
        raise RuntimeError("Main_case grooming private pose-gate metadata is missing.")
    gate_path_text = str(private_gate.get("guide_path") or "").strip()
    gate_metadata_text = str(private_gate.get("metadata_path") or "").strip()
    gate_path = Path(gate_path_text) if gate_path_text else Path()
    gate_metadata_path = Path(gate_metadata_text) if gate_metadata_text else Path()
    expected_gate_sha256 = str(private_gate.get("guide_sha256") or "").strip()
    if (
        not gate_path_text
        or not gate_metadata_text
        or gate_path.resolve().parent != guide_parent
        or gate_metadata_path.resolve().parent != guide_parent
        or not gate_path.is_file()
        or not gate_metadata_path.is_file()
        or private_gate.get("policy") != GROOMING_PRIVATE_POSE_GATE_GUIDE_POLICY
        or private_gate.get("exposed_to_qwen") is not False
        or not expected_gate_sha256
        or file_sha256(gate_path) != expected_gate_sha256
        or action_metadata.get("private_pose_gate_guide_sha256") != expected_gate_sha256
        or Path(
            str(action_metadata.get("private_pose_gate_guide_path") or "")
        ).resolve()
        != gate_path.resolve()
        or action_metadata.get("private_pose_gate_exposed_to_qwen") is not False
    ):
        raise RuntimeError(
            "Main_case grooming private pose-gate guide changed or escaped its job directory."
        )
    try:
        gate_metadata_file = json.loads(gate_metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "Main_case grooming private pose-gate metadata file is invalid."
        ) from exc
    relation = (
        gate_metadata_file.get("raised_paw_muzzle_relation")
        if isinstance(gate_metadata_file, dict)
        else None
    )
    output_encoding = (
        relation.get("output_encoding") if isinstance(relation, dict) else None
    )
    wrapper_relation = metadata_file.get("raised_paw_muzzle_relation")
    if (
        not isinstance(gate_metadata_file, dict)
        or gate_metadata_file.get("policy") != GROOMING_PRIVATE_POSE_GATE_GUIDE_POLICY
        or not isinstance(relation, dict)
        or relation.get("passed") is not True
        or relation.get("source_luma_values_preserved") is not False
        or relation.get("black_or_tool_semantics_preserved") is not False
        or not isinstance(output_encoding, dict)
        or output_encoding.get("method")
        != "compressed_blurred_quantized_neutral_anatomy"
        or not isinstance(wrapper_relation, dict)
        or wrapper_relation.get("passed") is not True
    ):
        raise RuntimeError(
            "Main_case grooming pose gate metadata lacks verified paw/muzzle geometry."
        )
    with Image.open(gate_path) as loaded:
        private_rgb = np.asarray(loaded.convert("RGB"), dtype=np.uint8)
    if private_rgb.shape != qwen_rgb.shape:
        raise RuntimeError(
            "Main_case grooming public/private guide dimensions changed."
        )
    private_foreground = ~np.all(private_rgb == (0, 255, 0), axis=2)
    removed_foreground = private_foreground & ~qwen_foreground
    expected_removed_pixels = int(structural_negative_space.get("pixels") or 0)
    if (
        not np.any(private_foreground)
        or np.any(qwen_foreground & ~private_foreground)
        or int(np.count_nonzero(removed_foreground)) != expected_removed_pixels
    ):
        raise RuntimeError(
            "Main_case grooming Qwen foreleg negative-space channel changed."
        )
    completion_mask_path: Path | None = None
    if production_open_channel:
        completion_mask = metadata_file.get("private_completion_mask")
        if not isinstance(completion_mask, dict):
            raise RuntimeError(
                "Main_case grooming private completion mask metadata is missing."
            )
        completion_path_text = str(completion_mask.get("path") or "").strip()
        completion_mask_path = (
            Path(completion_path_text) if completion_path_text else Path()
        )
        completion_sha256 = str(completion_mask.get("sha256") or "").strip()
        if (
            not completion_path_text
            or completion_mask_path.resolve().parent != guide_parent
            or not completion_mask_path.is_file()
            or completion_mask.get("policy") != GROOMING_PRIVATE_COMPLETION_MASK_POLICY
            or completion_mask.get("role") != "local_noise_mask_derivation_only"
            or completion_mask.get("exposed_to_qwen") is not False
            or completion_mask.get("loaded_as_qwen_image") is not False
            or not completion_sha256
            or file_sha256(completion_mask_path) != completion_sha256
            or action_metadata.get("private_completion_mask_path")
            != str(completion_mask_path)
            or action_metadata.get("private_completion_mask_sha256")
            != completion_sha256
            or action_metadata.get("private_completion_mask_exposed_to_qwen")
            is not False
        ):
            raise RuntimeError(
                "Main_case grooming private completion mask changed or escaped its job directory."
            )
        with Image.open(completion_mask_path) as loaded:
            completion_rgb = np.asarray(loaded.convert("RGB"), dtype=np.uint8)
        if completion_rgb.shape != qwen_rgb.shape:
            raise RuntimeError(
                "Main_case grooming private completion mask dimensions changed."
            )
        completion_background = np.all(
            completion_rgb == np.asarray((0, 255, 0), dtype=np.uint8),
            axis=2,
        )
        completion_body = np.all(
            completion_rgb
            == np.asarray(GROOMING_QWEN_GUIDE_FOREGROUND_RGB, dtype=np.uint8),
            axis=2,
        )
        completion_foreleg = np.all(
            completion_rgb
            == np.asarray(GROOMING_QWEN_GUIDE_FORELEG_RGB, dtype=np.uint8),
            axis=2,
        )
        completion_semantic_pixels = int(np.count_nonzero(completion_foreleg))
        if (
            not np.all(completion_background | completion_body | completion_foreleg)
            or completion_semantic_pixels
            != int(completion_mask.get("semantic_foreleg_pixels") or 0)
            or not 650 <= completion_semantic_pixels <= 6000
        ):
            raise RuntimeError(
                "Main_case grooming private completion mask palette changed."
            )
    return gate_path, gate_metadata_path, completion_mask_path


def evaluate_main_case_grooming_pose_candidate(
    source: Path,
    *,
    pose: str,
    green_target: Path,
    cutout_target: Path,
    semantic_target: Path,
    normalized_cutout_target: Path,
    geometry_guide: Path,
    geometry_metadata: Path,
    pose_transfer_seed: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recover an alpha subject and evaluate the Main_case grooming pose."""
    _sync_local_video_runtime(globals(), {"evaluate_main_case_grooming_pose_candidate"})

    semantic_recovery_error: str | None = None
    try:
        chroma = semantic_background_recovery(
            source,
            green_target,
            cutout_target,
            semantic_target,
            pose=pose,
        )
    except Exception as exc:  # noqa: BLE001 - retries must survive gate failures
        semantic_recovery_error = f"{type(exc).__name__}: {str(exc)[:1000]}"
        chroma = {
            "acceptable": False,
            "issues": ["grooming_pose_gate_alpha_recovery_failed"],
            "method": "rembg_u2net_grooming_pose_gate",
            "error": semantic_recovery_error,
        }
        if pose_transfer_seed is None or not semantic_target.is_file():
            return (
                chroma,
                grooming_pose_gate_failure(
                    "grooming_pose_gate_alpha_recovery_failed",
                    error=f"{type(exc).__name__}: {exc}",
                ),
            )
    if pose_transfer_seed is not None:
        semantic_recovery_chroma = dict(chroma)
        try:
            seed_bound_matte = prepare_canonical_grooming_seed_bound_matte(
                source,
                semantic_target,
                pose_transfer_seed,
                cutout_target,
                green_target=green_target,
            )
        except Exception as exc:  # noqa: BLE001 - fail closed and advance retry
            return (
                {
                    **chroma,
                    "acceptable": False,
                    "issues": list(
                        dict.fromkeys(
                            [
                                *[str(issue) for issue in chroma.get("issues", [])],
                                "grooming_seed_bound_matte_recovery_failed",
                            ]
                        )
                    ),
                    "method": GROOMING_CANONICAL_SEED_BOUND_MATTE_ALPHA_PROVENANCE,
                    "semantic_recovery": semantic_recovery_chroma,
                    "semantic_recovery_error": semantic_recovery_error,
                    "seed_bound_matte_error": (
                        f"{type(exc).__name__}: {str(exc)[:1000]}"
                    ),
                },
                grooming_pose_gate_failure(
                    "grooming_seed_bound_matte_recovery_failed",
                    error=f"{type(exc).__name__}: {exc}",
                ),
            )
        chroma = {
            **chroma,
            "acceptable": True,
            "issues": [],
            "method": GROOMING_CANONICAL_SEED_BOUND_MATTE_ALPHA_PROVENANCE,
            "semantic_recovery": semantic_recovery_chroma,
            "semantic_recovery_error": semantic_recovery_error,
            "seed_bound_matte": seed_bound_matte,
            "seed_bound_matte_policy": GROOMING_CANONICAL_SEED_BOUND_MATTE_POLICY,
        }
    chroma, normalized_cutout = finalize_qwen_action_reference_candidate(
        chroma,
        cutout_target,
        normalized_cutout_target,
    )
    if pose_transfer_seed is not None:
        chroma["matte_policy"] = GROOMING_CANONICAL_SEED_BOUND_MATTE_POLICY
    if chroma.get("acceptable") is not True:
        return (
            chroma,
            grooming_pose_gate_failure(
                "grooming_pose_gate_final_reference_normalization_failed",
                error=str(chroma.get("final_reference_normalization_error") or ""),
            ),
        )
    try:
        gate = evaluate_grooming_pose(
            normalized_cutout,
            geometry_guide,
            geometry_metadata=geometry_metadata,
        )
        # Serialize and reload at the boundary so attempt and job metadata can
        # never receive NumPy scalars, Paths, or another non-JSON value.
        gate = json.loads(json.dumps(gate, ensure_ascii=False))
    except Exception as exc:  # noqa: BLE001 - fail closed and advance retry
        gate = grooming_pose_gate_failure(
            "grooming_pose_gate_evaluation_failed",
            error=f"{type(exc).__name__}: {exc}",
        )
    return chroma, gate


def semantic_recovery_candidate_eligible(record: dict[str, Any]) -> bool:
    """Allow rembg for a distinct raw pose; audit palette again after removal."""
    _sync_local_video_runtime(globals(), {"semantic_recovery_candidate_eligible"})
    chroma = record.get("chroma")
    similarity = record.get("similarity")
    return bool(
        isinstance(chroma, dict)
        and not chroma.get("acceptable", False)
        and (
            not GENERATION_VISUAL_AUDITS_BLOCKING
            or (
                isinstance(similarity, dict)
                and similarity.get("acceptable", False)
            )
        )
    )


def pose_attempt_failure_issues(attempts: list[dict[str, Any]]) -> list[str]:
    _sync_local_video_runtime(globals(), {"pose_attempt_failure_issues"})
    issues: list[str] = []
    for failed_attempt in attempts:
        for section_name in (
            "chroma",
            "quality",
            "similarity",
            "identity_palette",
            "fur_silhouette",
            "fur_surface_gate",
            "grooming_pose_gate",
            "grooming_advisory_minimum_gate",
        ):
            section = failed_attempt.get(section_name)
            if (
                section_name == "fur_surface_gate"
                and isinstance(section, dict)
                and section.get("evaluated") is False
            ):
                continue
            if isinstance(section, dict) and isinstance(section.get("issues"), list):
                issues.extend(str(issue) for issue in section["issues"])
    return list(dict.fromkeys(issues))


def pose_attempt_failure_summary(attempts: list[dict[str, Any]]) -> str:
    """Separate persistent blockers from one-off retry failures."""
    _sync_local_video_runtime(globals(), {"pose_attempt_failure_summary"})
    if not attempts:
        return "no usable subject"
    ordered_issues = pose_attempt_failure_issues(attempts)
    issues_by_attempt: list[tuple[int, list[str]]] = []
    for fallback_index, failed_attempt in enumerate(attempts, start=1):
        attempt_issues: list[str] = []
        for section_name in (
            "chroma",
            "quality",
            "similarity",
            "identity_palette",
            "fur_silhouette",
            "fur_surface_gate",
            "grooming_pose_gate",
            "grooming_advisory_minimum_gate",
        ):
            section = failed_attempt.get(section_name)
            if (
                section_name == "fur_surface_gate"
                and isinstance(section, dict)
                and section.get("evaluated") is False
            ):
                continue
            if isinstance(section, dict) and isinstance(section.get("issues"), list):
                attempt_issues.extend(str(issue) for issue in section["issues"])
        issues_by_attempt.append(
            (
                int(failed_attempt.get("attempt") or fallback_index),
                list(dict.fromkeys(attempt_issues)),
            )
        )
    persistent = [
        issue
        for issue in ordered_issues
        if all(issue in attempt_issues for _, attempt_issues in issues_by_attempt)
    ]
    transient_parts = [
        f"attempt {attempt}: {', '.join(issue for issue in attempt_issues if issue not in persistent)}"
        for attempt, attempt_issues in issues_by_attempt
        if any(issue not in persistent for issue in attempt_issues)
    ]
    parts: list[str] = []
    if persistent:
        parts.append("persistent: " + ", ".join(persistent))
    if transient_parts:
        parts.append("transient: " + "; ".join(transient_parts))
    return "; ".join(parts) or "no usable subject"


def eating_mouth_bowl_contact_quality(path: Path) -> dict[str, Any]:
    """Require a real face/muzzle anchor immediately above the detected food bowl.

    The action reference splitter is shared with the SCAIL stage so the keyframe
    gate and final reference-space mouth motion agree about which pixels are the
    bowl, planted paws and cat identity. A low/wide silhouette alone is not an
    eating action: the warm nose/muzzle component must be centered over the bowl
    opening with at most the small gap that SCAIL can safely close at low phase.
    """
    _sync_local_video_runtime(globals(), {"eating_mouth_bowl_contact_quality"})

    try:
        from ...pipeline.subject_replacement import (
            make_eating_reference_identity_mask,
        )

        with Image.open(path) as loaded:
            reference = loaded.convert("RGBA")
        colored_mask, _prop_exclusion, prop_metadata = (
            make_eating_reference_identity_mask(
                reference,
                allow_fallback=False,
            )
        )
    except Exception as exc:  # noqa: BLE001 - convert splitter failures to gate data
        error = f"{type(exc).__name__}: {str(exc)[:1000]}"
        lowered = str(exc).lower()
        if any(
            token in lowered
            for token in (
                "warm_seed",
                "warm_food",
                "bowl_body",
                "warm_bowl",
                "prop_fraction",
            )
        ):
            issue = "eating_bowl_or_food_not_detected"
        elif "paw_support" in lowered:
            issue = "eating_bowl_paw_layout_invalid"
        else:
            issue = "eating_bowl_contact_unreliable"
        return {
            "acceptable": False,
            "issues": [issue],
            "penalty": 125.0,
            "method": EATING_MOUTH_BOWL_CONTACT_POLICY,
            "error": error,
        }

    reference_rgba = np.asarray(reference, dtype=np.uint8)
    colored = np.asarray(colored_mask.convert("RGB"), dtype=np.uint8)
    reference_subject = np.all(colored == (0, 0, 255), axis=2)
    reference_bowl = np.all(colored == (255, 0, 0), axis=2)
    if (
        int(np.count_nonzero(reference_subject)) < 500
        or int(np.count_nonzero(reference_bowl)) < 100
    ):
        return {
            "acceptable": False,
            "issues": ["eating_bowl_contact_unreliable"],
            "penalty": 125.0,
            "method": EATING_MOUTH_BOWL_CONTACT_POLICY,
        }

    subject_rows, subject_columns = np.where(reference_subject)
    bowl_rows, bowl_columns = np.where(reference_bowl)
    subject_left = int(subject_columns.min())
    subject_top = int(subject_rows.min())
    subject_right = int(subject_columns.max()) + 1
    subject_bottom = int(subject_rows.max()) + 1
    bowl_left = int(bowl_columns.min())
    bowl_top = int(bowl_rows.min())
    bowl_right = int(bowl_columns.max()) + 1
    bowl_bottom = int(bowl_rows.max()) + 1
    subject_height = subject_bottom - subject_top
    bowl_width = bowl_right - bowl_left

    rgb = reference_rgba[..., :3]
    red = rgb[..., 0].astype(np.int16)
    green = rgb[..., 1].astype(np.int16)
    blue = rgb[..., 2].astype(np.int16)
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
        bowl_top - int(round(max(bowl_width * 0.90, subject_height * 0.40))),
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
    minimum_face_pixels = max(
        12,
        int(round(np.count_nonzero(reference_subject) * 0.00015)),
    )
    maximum_face_width = int(bowl_width * EATING_MUZZLE_COMPONENT_WIDTH_BOWL_RATIO_MAX)
    maximum_face_height = int(
        bowl_width * EATING_MUZZLE_COMPONENT_HEIGHT_BOWL_RATIO_MAX
    )
    bowl_center_x = (bowl_left + bowl_right) / 2.0
    center_min = (
        bowl_left - bowl_width * EATING_MUZZLE_COMPONENT_CENTER_MARGIN_BOWL_RATIO
    )
    center_max = (
        bowl_right + bowl_width * EATING_MUZZLE_COMPONENT_CENTER_MARGIN_BOWL_RATIO
    )
    contact_thresholds = {
        "minimum_component_pixels": minimum_face_pixels,
        "gap_px_min": 0,
        "gap_px_max": EATING_MUZZLE_BOWL_GAP_MAX_PX,
        "component_width_px_min": 3,
        "component_width_px_max": maximum_face_width,
        "component_width_bowl_ratio_max": (
            EATING_MUZZLE_COMPONENT_WIDTH_BOWL_RATIO_MAX
        ),
        "component_height_px_min": 2,
        "component_height_px_max": maximum_face_height,
        "component_height_bowl_ratio_max": (
            EATING_MUZZLE_COMPONENT_HEIGHT_BOWL_RATIO_MAX
        ),
        "component_center_margin_bowl_ratio": (
            EATING_MUZZLE_COMPONENT_CENTER_MARGIN_BOWL_RATIO
        ),
    }
    face_components: list[tuple[int, int, float, int]] = []
    face_component_audits: list[dict[str, Any]] = []
    face_component_audit_by_label: dict[int, dict[str, Any]] = {}
    for label in range(1, face_count):
        area = int(face_stats[label, cv2.CC_STAT_AREA])
        component_x = int(face_stats[label, cv2.CC_STAT_LEFT])
        component_y = int(face_stats[label, cv2.CC_STAT_TOP])
        component_width = int(face_stats[label, cv2.CC_STAT_WIDTH])
        component_height = int(face_stats[label, cv2.CC_STAT_HEIGHT])
        component_bottom = component_y + component_height
        component_gap = bowl_top - component_bottom
        center_x = float(face_centroids[label][0])
        rejection_reasons: list[str] = []
        if area < minimum_face_pixels:
            rejection_reasons.append("area_below_minimum")
        if not 0 <= component_gap <= EATING_MUZZLE_BOWL_GAP_MAX_PX:
            rejection_reasons.append("gap_outside_contact_range")
        if not 3 <= component_width <= maximum_face_width:
            rejection_reasons.append("width_outside_muzzle_range")
        if not 2 <= component_height <= maximum_face_height:
            rejection_reasons.append("height_outside_muzzle_range")
        if not center_min <= center_x <= center_max:
            rejection_reasons.append("center_outside_bowl_corridor")
        component_audit = {
            "label": label,
            "bbox": [
                component_x,
                component_y,
                component_x + component_width,
                component_bottom,
            ],
            "area": area,
            "gap_px": component_gap,
            "width_bowl_ratio": round(
                component_width / max(1, bowl_width),
                6,
            ),
            "height_bowl_ratio": round(
                component_height / max(1, bowl_width),
                6,
            ),
            "center_offset_bowl_ratio": round(
                abs(center_x - bowl_center_x) / max(1, bowl_width),
                6,
            ),
            "rejection_reasons": rejection_reasons,
        }
        face_component_audits.append(component_audit)
        face_component_audit_by_label[label] = component_audit
        if not rejection_reasons:
            face_components.append(
                (
                    component_bottom,
                    area,
                    -abs(center_x - bowl_center_x),
                    label,
                )
            )
    face_component_audits.sort(
        key=lambda item: (
            abs(int(item["gap_px"])),
            len(item["rejection_reasons"]),
            -int(item["area"]),
        )
    )
    audited_components = face_component_audits[:12]
    if not face_components:
        return {
            "acceptable": False,
            "issues": ["eating_muzzle_not_at_bowl"],
            "penalty": 125.0,
            "method": EATING_MOUTH_BOWL_CONTACT_POLICY,
            "bowl_bbox": [bowl_left, bowl_top, bowl_right, bowl_bottom],
            "face_search_bbox": [
                face_search_x0,
                face_search_y0,
                face_search_x1,
                bowl_top,
            ],
            "face_seed_pixels": int(np.count_nonzero(face_seed_candidates)),
            "component_candidates": audited_components,
            "thresholds": contact_thresholds,
            "prop": prop_metadata,
        }

    _bottom_rank, _area_rank, _center_rank, face_label = max(face_components)
    face_x = int(face_stats[face_label, cv2.CC_STAT_LEFT])
    face_y = int(face_stats[face_label, cv2.CC_STAT_TOP])
    face_width = int(face_stats[face_label, cv2.CC_STAT_WIDTH])
    face_height = int(face_stats[face_label, cv2.CC_STAT_HEIGHT])
    face_bottom = face_y + face_height
    gap = bowl_top - face_bottom
    muzzle_x = int(round(float(face_centroids[face_label][0])))
    return {
        "acceptable": True,
        "issues": [],
        "penalty": round(float(max(0, gap)) * 0.5, 6),
        "method": EATING_MOUTH_BOWL_CONTACT_POLICY,
        "muzzle_anchor": [muzzle_x, face_bottom - 1],
        "muzzle_bowl_gap_px": int(gap),
        "face_seed_bbox": [
            face_x,
            face_y,
            face_x + face_width,
            face_bottom,
        ],
        "selected_component": face_component_audit_by_label[face_label],
        "component_candidates": audited_components,
        "thresholds": contact_thresholds,
        "bowl_bbox": [bowl_left, bowl_top, bowl_right, bowl_bottom],
        "prop": prop_metadata,
    }


def pose_cutout_quality(
    path: Path,
    pose: str,
    *,
    require_eating_mouth_contact: bool = False,
) -> dict[str, Any]:
    _sync_local_video_runtime(globals(), {"pose_cutout_quality"})
    rgba = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if rgba is None or rgba.ndim != 3 or rgba.shape[2] < 4:
        raise RuntimeError(f"Could not inspect transparent pose output {path.name}.")
    alpha = rgba[:, :, 3]
    foreground = (alpha >= 64).astype(np.uint8)
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(
        foreground, connectivity=8
    )
    area_records = sorted(
        (
            (label, int(stats[label, cv2.CC_STAT_AREA]))
            for label in range(1, component_count)
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    areas = [area for _, area in area_records]
    total_area = sum(areas)
    if not areas or total_area <= 0:
        return {
            "acceptable": False,
            "issues": ["no_foreground_subject"],
            "penalty": 1000.0,
            "component_count": 0,
        }
    meaningful_minimum = max(48, int(total_area * 0.001))
    meaningful_records = [
        (label, area) for label, area in area_records if area >= meaningful_minimum
    ]
    meaningful_labels = [label for label, _ in meaningful_records]
    meaningful_areas = [area for _, area in meaningful_records]
    minor_fraction = float(sum(meaningful_areas[1:])) / float(total_area)
    ys, xs = np.where(foreground > 0)
    height, width = foreground.shape
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    margins = {
        "left": bbox[0] / width,
        "top": bbox[1] / height,
        "right": (width - bbox[2]) / width,
        "bottom": (height - bbox[3]) / height,
    }
    issues: list[str] = []
    if min(margins.values()) < 0.025:
        issues.append("subject_too_close_to_canvas_edge")
    action = action_key(pose)
    if action in {"running", "lying", "lying down"} and minor_fraction > 0.003:
        issues.append("disconnected_body_or_paw_fragment")
    foreground_fraction = total_area / float(width * height)
    if foreground_fraction < 0.035:
        issues.append("subject_too_small")
    if foreground_fraction > 0.72:
        issues.append("subject_too_large")
    bbox_width = max(1, bbox[2] - bbox[0])
    bbox_height = max(1, bbox[3] - bbox[1])
    bbox_aspect_ratio = bbox_width / float(bbox_height)
    if (
        action in {"eating", "drinking"}
        and bbox_aspect_ratio < QWEN_BOWL_ACTION_MIN_ASPECT_RATIO
    ):
        issues.append("pose_not_low_or_wide_enough_for_bowl_action")
    bowl_components: list[dict[str, Any]] = []
    if action in {"eating", "drinking"}:
        for label in meaningful_labels[1:]:
            component_area = int(stats[label, cv2.CC_STAT_AREA])
            component_x = int(stats[label, cv2.CC_STAT_LEFT])
            component_y = int(stats[label, cv2.CC_STAT_TOP])
            component_width = int(stats[label, cv2.CC_STAT_WIDTH])
            component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
            component_fraction = component_area / float(total_area)
            component_aspect_ratio = component_width / float(max(1, component_height))
            component_center_y = (
                component_y + component_height / 2.0 - bbox[1]
            ) / float(bbox_height)
            if (
                0.008 <= component_fraction <= 0.30
                and component_aspect_ratio >= 0.75
                and component_width >= bbox_width * 0.10
                and component_center_y >= 0.52
            ):
                bowl_components.append(
                    {
                        "area": component_area,
                        "fraction": round(component_fraction, 6),
                        "bbox": [
                            component_x,
                            component_y,
                            component_x + component_width,
                            component_y + component_height,
                        ],
                        "aspect_ratio": round(component_aspect_ratio, 6),
                        "center_y": round(component_center_y, 6),
                    }
                )
        # A correct bowl often touches the muzzle or paws after alpha matting, so
        # connected-component separation is diagnostic only. Enforcing it as a
        # hard gate rejects validated eating references. The hard geometric gate
        # above instead requires the character-plus-bowl layout to be low and
        # wide; the validated action guide and written prompt supply semantics.
    if action in {"lying", "lying down"} and bbox_aspect_ratio < 0.90:
        issues.append("pose_not_low_or_wide_enough_for_lying")
    eating_contact = None
    if action == "eating" and require_eating_mouth_contact:
        eating_contact = eating_mouth_bowl_contact_quality(path)
        if not eating_contact["acceptable"]:
            issues.extend(eating_contact["issues"])
    lower_body_component_peak = 0
    lower_body_components: dict[str, int] = {}
    if action == "running":
        y0, y1 = bbox[1], bbox[3]
        minimum_lower_area = max(36, int(total_area * 0.0005))
        for fraction in (0.72, 0.76, 0.80, 0.84, 0.88, 0.90):
            cutoff = y0 + int((y1 - y0) * fraction)
            lower_mask = foreground.copy()
            lower_mask[:cutoff, :] = 0
            lower_count, _, lower_stats, _ = cv2.connectedComponentsWithStats(
                lower_mask,
                connectivity=8,
            )
            count = sum(
                int(lower_stats[label, cv2.CC_STAT_AREA]) >= minimum_lower_area
                for label in range(1, lower_count)
            )
            lower_body_components[f"{fraction:.2f}"] = count
            lower_body_component_peak = max(lower_body_component_peak, count)
        if lower_body_component_peak < 3:
            issues.append("fewer_than_three_distinct_grounded_paws")
    penalty = minor_fraction * 100.0
    penalty += max(0.0, 0.025 - min(margins.values())) * 200.0
    penalty += 25.0 * len(issues)
    return {
        "acceptable": not issues,
        "issues": issues,
        "penalty": round(penalty, 6),
        "component_count": len(meaningful_areas),
        "component_areas": meaningful_areas,
        "minor_component_fraction": round(minor_fraction, 6),
        "foreground_fraction": round(foreground_fraction, 6),
        "bbox": bbox,
        "bbox_aspect_ratio": round(bbox_aspect_ratio, 6),
        "margins": {key: round(value, 6) for key, value in margins.items()},
        "lower_body_component_peak": lower_body_component_peak,
        "lower_body_components": lower_body_components,
        "bowl_component_count": len(bowl_components),
        "bowl_components": bowl_components,
        "eating_mouth_bowl_contact": eating_contact,
    }


def fur_silhouette_signature(path: Path) -> dict[str, Any]:
    """Measure coat-edge expansion on a normalized eating reference."""
    _sync_local_video_runtime(globals(), {"fur_silhouette_signature"})
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 3 or image.shape[2] < 3:
        raise RuntimeError(f"Could not inspect fur silhouette {path.name}.")
    height, width = image.shape[:2]
    if image.shape[2] >= 4 and np.any(image[:, :, 3] < 250):
        foreground = image[:, :, 3] >= 128
        mask_method = "alpha"
    else:
        blue = image[:, :, 0].astype(np.int16)
        green = image[:, :, 1].astype(np.int16)
        red = image[:, :, 2].astype(np.int16)
        green_background = (green > 80) & ((green - np.maximum(red, blue)) > 35)
        green_fraction = float(np.count_nonzero(green_background)) / float(
            green_background.size
        )
        if green_fraction < 0.05:
            raise RuntimeError(
                f"Fur silhouette reference {path.name} has neither alpha nor "
                "a clean green background."
            )
        foreground = ~green_background
        mask_method = "green_screen"

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        foreground.astype(np.uint8),
        connectivity=8,
    )
    if component_count <= 1:
        raise RuntimeError(f"Fur silhouette {path.name} has no foreground subject.")
    largest_label = max(
        range(1, component_count),
        key=lambda label: int(stats[label, cv2.CC_STAT_AREA]),
    )
    subject = labels == largest_label
    ys, xs = np.where(subject)
    if xs.size < 256:
        raise RuntimeError(f"Fur silhouette {path.name} is too small to inspect.")
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    bbox_width = max(1, bbox[2] - bbox[0])
    bbox_height = max(1, bbox[3] - bbox[1])

    # This band is the back/upper-torso region in the validated eating layout.
    # Relative coordinates keep the metric stable under harmless translation and
    # small framing changes while excluding most of the bowl and front paws.
    x0 = bbox[0] + int(round(bbox_width * 0.11))
    x1 = bbox[0] + int(round(bbox_width * 0.61))
    top_values: list[float] = []
    for x in range(max(0, x0), min(width, x1)):
        column = np.flatnonzero(subject[:, x])
        if column.size:
            top_values.append(float(column[0]))
    if len(top_values) < 48:
        raise RuntimeError(
            f"Fur silhouette {path.name} has insufficient upper contour coverage."
        )
    top = np.asarray(top_values, dtype=np.float32)
    smoothing_window = max(7, int(round(bbox_width * 15.0 / 400.0)))
    if smoothing_window % 2 == 0:
        smoothing_window += 1
    smoothing_window = min(smoothing_window, max(7, len(top) // 3 * 2 + 1))
    kernel = np.ones(smoothing_window, dtype=np.float32) / float(smoothing_window)
    smoothed = np.convolve(top, kernel, mode="same")
    edge_trim = max(4, smoothing_window // 2 + 1)
    residual = top[edge_trim:-edge_trim] - smoothed[edge_trim:-edge_trim]
    if residual.size < 24:
        raise RuntimeError(
            f"Fur silhouette {path.name} has insufficient contour samples."
        )
    canvas_scale = 512.0 / float(max(height, width))
    top_roughness = float(np.sqrt(np.mean(np.square(residual)))) * canvas_scale

    upper_cutoff = min(
        height,
        bbox[1] + int(round(bbox_height * 0.72)),
    )
    upper = subject.copy()
    upper[upper_cutoff:, :] = False
    upper_area = int(np.count_nonzero(upper))
    if upper_area < 256:
        raise RuntimeError(f"Fur silhouette {path.name} has no usable upper body.")
    horizontal_transitions = int(np.count_nonzero(upper[:, 1:] != upper[:, :-1]))
    vertical_transitions = int(np.count_nonzero(upper[1:, :] != upper[:-1, :]))
    upper_perimeter_normalized = float(
        (horizontal_transitions + vertical_transitions) / np.sqrt(upper_area)
    )
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "mask_method": mask_method,
        "canvas": [width, height],
        "bbox": bbox,
        "foreground_area": int(np.count_nonzero(subject)),
        "upper_area": upper_area,
        "contour_sample_count": len(top_values),
        "smoothing_window": smoothing_window,
        "top_contour_roughness_px_512": round(top_roughness, 6),
        "upper_perimeter_normalized": round(upper_perimeter_normalized, 6),
    }


def eating_fur_silhouette_quality(
    candidate_path: Path,
    reference_path: Path,
) -> dict[str, Any]:
    """Fail closed when an eating reference grows a jagged or expanded coat."""
    _sync_local_video_runtime(globals(), {"eating_fur_silhouette_quality"})
    reference = fur_silhouette_signature(reference_path)
    candidate = fur_silhouette_signature(candidate_path)
    reference_roughness = float(reference["top_contour_roughness_px_512"])
    candidate_roughness = float(candidate["top_contour_roughness_px_512"])
    roughness_limit = max(
        reference_roughness * 1.45,
        reference_roughness + 1.50,
    )
    reference_perimeter = float(reference["upper_perimeter_normalized"])
    candidate_perimeter = float(candidate["upper_perimeter_normalized"])
    perimeter_limit = max(
        reference_perimeter * 1.20,
        reference_perimeter + 0.65,
    )
    issues: list[str] = []
    if candidate_roughness > roughness_limit:
        issues.append("fur_top_contour_too_rough")
    if candidate_perimeter > perimeter_limit:
        issues.append("fur_upper_silhouette_too_jagged")
    roughness_overage = max(0.0, candidate_roughness / roughness_limit - 1.0)
    perimeter_overage = max(0.0, candidate_perimeter / perimeter_limit - 1.0)
    return {
        "acceptable": not issues,
        "issues": issues,
        "penalty": round((roughness_overage + perimeter_overage) * 100.0, 6),
        "policy": "profile_relative_eating_contour_v1",
        "reference": reference,
        "candidate": candidate,
        "thresholds": {
            "top_contour_roughness_px_512_max": round(roughness_limit, 6),
            "upper_perimeter_normalized_max": round(perimeter_limit, 6),
        },
        "ratios": {
            "top_contour_roughness": round(
                candidate_roughness / max(reference_roughness, 1e-6),
                6,
            ),
            "upper_perimeter": round(
                candidate_perimeter / max(reference_perimeter, 1e-6),
                6,
            ),
        },
    }


def pose_similarity_signature(path: Path, size: int = 192) -> dict[str, Any]:
    """Normalize a transparent or green-screen subject for pose-only comparison."""
    _sync_local_video_runtime(globals(), {"pose_similarity_signature"})
    with Image.open(path) as loaded:
        rgba = np.asarray(loaded.convert("RGBA"))
    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3]
    if np.any(alpha < 250):
        foreground = alpha >= 64
        mask_method = "alpha"
    else:
        red = rgb[:, :, 0].astype(np.int16)
        green = rgb[:, :, 1].astype(np.int16)
        blue = rgb[:, :, 2].astype(np.int16)
        green_background = (green > 80) & ((green - np.maximum(red, blue)) > 35)
        if (
            float(np.count_nonzero(green_background)) / float(green_background.size)
            >= 0.05
        ):
            foreground = ~green_background
            mask_method = "green_screen"
        else:
            corners = np.asarray(
                [
                    rgb[0, 0],
                    rgb[0, -1],
                    rgb[-1, 0],
                    rgb[-1, -1],
                ],
                dtype=np.float32,
            )
            background = np.median(corners, axis=0)
            distance = np.linalg.norm(rgb.astype(np.float32) - background, axis=2)
            foreground = distance >= 24.0
            mask_method = "corner_color"
    foreground_u8 = foreground.astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        foreground_u8,
        connectivity=8,
    )
    areas = [int(stats[label, cv2.CC_STAT_AREA]) for label in range(1, component_count)]
    total_area = sum(areas)
    if total_area <= 0:
        return {
            "available": False,
            "reason": "no_foreground_subject",
            "path": path.name,
            "mask_method": mask_method,
        }
    meaningful_minimum = max(48, int(total_area * 0.001))
    meaningful_labels = {
        label
        for label in range(1, component_count)
        if int(stats[label, cv2.CC_STAT_AREA]) >= meaningful_minimum
    }
    foreground = np.isin(labels, list(meaningful_labels))
    ys, xs = np.where(foreground)
    if not len(xs) or not len(ys):
        return {
            "available": False,
            "reason": "no_meaningful_foreground_subject",
            "path": path.name,
            "mask_method": mask_method,
        }
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    cropped_mask = (
        foreground[bbox[1] : bbox[3], bbox[0] : bbox[2]].astype(np.uint8) * 255
    )
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cropped_gray = gray[bbox[1] : bbox[3], bbox[0] : bbox[2]]
    cropped_gray = cv2.bitwise_and(cropped_gray, cropped_gray, mask=cropped_mask)
    available_size = max(32, size - 16)
    scale = min(
        available_size / max(1, cropped_mask.shape[1]),
        available_size / max(1, cropped_mask.shape[0]),
    )
    resized_width = max(1, int(round(cropped_mask.shape[1] * scale)))
    resized_height = max(1, int(round(cropped_mask.shape[0] * scale)))
    resized_mask = cv2.resize(
        cropped_mask,
        (resized_width, resized_height),
        interpolation=cv2.INTER_NEAREST,
    )
    resized_gray = cv2.resize(
        cropped_gray,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )
    normalized_mask = np.zeros((size, size), dtype=np.uint8)
    normalized_gray = np.zeros((size, size), dtype=np.uint8)
    x_offset = (size - resized_width) // 2
    y_offset = (size - resized_height) // 2
    normalized_mask[
        y_offset : y_offset + resized_height,
        x_offset : x_offset + resized_width,
    ] = resized_mask
    normalized_gray[
        y_offset : y_offset + resized_height,
        x_offset : x_offset + resized_width,
    ] = resized_gray
    dhash_source = cv2.resize(normalized_gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (dhash_source[:, 1:] > dhash_source[:, :-1]).flatten()
    dhash = 0
    for bit in bits:
        dhash = (dhash << 1) | int(bit)
    return {
        "available": True,
        "path": path.name,
        "mask_method": mask_method,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "mask": normalized_mask >= 64,
        "dhash": dhash,
        "bbox_aspect_ratio": round(
            (bbox[2] - bbox[0]) / float(max(1, bbox[3] - bbox[1])),
            6,
        ),
    }


def compare_pose_signatures(
    candidate: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any]:
    _sync_local_video_runtime(globals(), {"compare_pose_signatures"})
    if not candidate.get("available") or not reference.get("available"):
        return {
            "available": False,
            "reason": candidate.get("reason")
            or reference.get("reason")
            or "signature_unavailable",
        }
    candidate_mask = candidate["mask"]
    reference_mask = reference["mask"]
    intersection = int(np.count_nonzero(candidate_mask & reference_mask))
    union = int(np.count_nonzero(candidate_mask | reference_mask))
    iou = intersection / float(union) if union else 0.0
    dhash_distance = int(
        (int(candidate["dhash"]) ^ int(reference["dhash"])).bit_count()
    )
    return {
        "available": True,
        "sha256_equal": candidate["sha256"] == reference["sha256"],
        "silhouette_iou": round(iou, 6),
        "dhash_distance": dhash_distance,
        "candidate_aspect_ratio": candidate["bbox_aspect_ratio"],
        "reference_aspect_ratio": reference["bbox_aspect_ratio"],
    }


def pose_similarity_quality(
    candidate_path: Path,
    *,
    pose: str,
    canonical_path: Path | None,
    previous_poses: list[dict[str, Any]],
) -> dict[str, Any]:
    """Reject a pose edit that merely reconstructs the reference or an earlier pose."""
    _sync_local_video_runtime(globals(), {"pose_similarity_quality"})
    candidate = pose_similarity_signature(candidate_path)
    if not candidate.get("available"):
        return {
            "available": False,
            "acceptable": False,
            "warning": True,
            "issues": ["candidate_similarity_signature_unavailable"],
            "comparisons": [],
            "penalty": 1000.0,
        }
    references: list[dict[str, Any]] = []
    if canonical_path and canonical_path.is_file():
        references.append(
            {
                "label": "canonical",
                "pose": "canonical",
                "pose_index": None,
                "path": canonical_path,
            }
        )
    references.extend(previous_poses)
    comparisons: list[dict[str, Any]] = []
    issues: list[str] = []
    unavailable_references: list[str] = []
    maximum_penalty = 0.0
    current_action = action_key(pose)
    for reference_info in references:
        reference_path = Path(reference_info["path"])
        reference = pose_similarity_signature(reference_path)
        comparison = compare_pose_signatures(candidate, reference)
        comparison.update(
            {
                "label": reference_info["label"],
                "pose": reference_info["pose"],
                "pose_index": reference_info.get("pose_index"),
                "filename": reference_path.name,
            }
        )
        if not comparison.get("available"):
            comparisons.append(comparison)
            unavailable_references.append(str(reference_info["label"]))
            continue
        reference_action = action_key(str(reference_info["pose"]))
        action_requires_canonical_change = bool(
            reference_info["label"] == "canonical"
            and current_action in {"eating", "drinking", "grooming"}
        )
        related_pair = (
            reference_info["label"] == "canonical" and current_action == "running"
        ) or {current_action, reference_action} == {"eating", "drinking"}
        if action_requires_canonical_change:
            iou_threshold = QWEN_ACTION_CANONICAL_IOU_THRESHOLD
            dhash_threshold = QWEN_ACTION_CANONICAL_DHASH_THRESHOLD
        elif related_pair:
            iou_threshold = QWEN_POSE_RELATED_IOU_THRESHOLD
            dhash_threshold = QWEN_POSE_RELATED_DHASH_THRESHOLD
        else:
            iou_threshold = QWEN_POSE_SIMILARITY_IOU_THRESHOLD
            dhash_threshold = QWEN_POSE_SIMILARITY_DHASH_THRESHOLD
        hard_duplicate = bool(
            comparison["sha256_equal"]
            or (
                comparison["silhouette_iou"] >= QWEN_POSE_HARD_DUPLICATE_IOU
                and comparison["dhash_distance"] <= QWEN_POSE_HARD_DUPLICATE_DHASH
            )
        )
        too_similar = bool(
            comparison["silhouette_iou"] >= iou_threshold
            and comparison["dhash_distance"] <= dhash_threshold
        )
        comparison.update(
            {
                "related_pose_threshold": related_pair,
                "action_requires_canonical_change": action_requires_canonical_change,
                "iou_threshold": iou_threshold,
                "dhash_threshold": dhash_threshold,
                "hard_duplicate": hard_duplicate,
                "too_similar": too_similar,
            }
        )
        if hard_duplicate:
            issues.append(f"hard_duplicate_of_{reference_info['label']}")
        elif too_similar:
            issues.append(f"pose_too_similar_to_{reference_info['label']}")
        maximum_penalty = max(
            maximum_penalty,
            comparison["silhouette_iou"] * 100.0
            - min(comparison["dhash_distance"], 32) * 0.5,
        )
        comparisons.append(comparison)
    for label in unavailable_references:
        issues.append(f"reference_similarity_signature_unavailable_{label}")
    if unavailable_references:
        maximum_penalty = max(maximum_penalty, 1000.0)
    return {
        "available": not unavailable_references,
        "acceptable": not issues,
        "warning": bool(issues or unavailable_references),
        "issues": list(dict.fromkeys(issues)),
        "comparisons": comparisons,
        "penalty": round(maximum_penalty, 6),
    }
