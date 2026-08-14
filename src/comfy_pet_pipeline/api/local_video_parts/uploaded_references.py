"""Uploaded identity reference generation for Qwen and SCAIL."""

from __future__ import annotations

# The extracted functions resolve the facade dependencies after runtime sync.
# ruff: noqa: F821

from .runtime import sync_local_video_runtime as _sync_local_video_runtime


def prepare_uploaded_subject_qwen_references(
    job_id: str,
    actions: list[str],
) -> dict[str, str]:
    """Create a Qwen full-body canonical and normalized action keyframes."""
    _sync_local_video_runtime(
        globals(),
        {"prepare_uploaded_subject_qwen_references"},
    )
    raise_if_cancelled(job_id)
    with JOBS_LOCK:
        job = dict(JOBS[job_id])
    job_dir = OUTPUT_ROOT / job_id
    upload_integrity = verify_uploaded_identity_submission(job, job_dir)
    identity_view_audit = compose_uploaded_identity_view_audit(upload_integrity)
    write_json_atomic(
        job_dir / "workflows" / "uploaded_identity_view_set_audit.json",
        identity_view_audit,
    )
    update_job(
        job_id,
        uploaded_identity_view_audit=identity_view_audit,
        uploaded_identity_front_view_audit=identity_view_audit["front_view_audit"],
        uploaded_identity_view_verdict=identity_view_audit["status"],
    )
    front_view_audit = identity_view_audit["front_view_audit"]
    if not front_view_audit["acceptable"]:
        front_issue = str(
            next(
                iter(front_view_audit.get("issues") or []),
                "front_view_audit_unavailable",
            )
        )
        if front_issue == FRONT_VIEW_NOT_RELIABLY_FRONTAL_STAGE:
            anomaly_code = FRONT_VIEW_NOT_RELIABLY_FRONTAL_STAGE
            anomaly_message = (
                "front_view_not_reliably_frontal: front 槽位未检测到可靠宠物正面图；"
                "任务继续运行，生成结果可能偏离输入形象。"
            )
            anomaly_stage = FRONT_VIEW_NOT_RELIABLY_FRONTAL_STAGE
            expose_as_public_stage = True
        else:
            anomaly_code = "uploaded_front_view_audit_unavailable"
            anomaly_message = (
                "上传图片正面审核异常：无法完成 front 槽位的正面审核；"
                "任务已按非阻断策略继续执行。"
            )
            anomaly_stage = "uploaded_identity_front_view_audit"
            expose_as_public_stage = False
        record_nonblocking_job_anomaly(
            job_id,
            code=anomaly_code,
            message=anomaly_message,
            stage=anomaly_stage,
            details={
                "audit_policy": UPLOADED_FRONT_VIEW_AUDIT_POLICY,
                "issues": list(front_view_audit.get("issues") or []),
                "view_role": "front",
            },
            expose_as_public_stage=expose_as_public_stage,
        )
    if (
        GENERATION_VISUAL_AUDITS_BLOCKING
        and not identity_view_audit["blocking_acceptable"]
    ):
        update_job(
            job_id,
            reference_generation_status="failed",
            reference_generation_completed=False,
        )
        raise RuntimeError(
            "Uploaded identity view set failed validation: "
            + ", ".join(identity_view_audit["issues"])
        )
    if not identity_view_audit["acceptable"]:
        LOGGER.warning(
            "%s uploaded identity view audit is advisory: %s",
            job_id,
            identity_view_audit["issues"],
        )

    image_profile = IMAGE_PROFILES[SUBJECT_REPLACEMENT_REFERENCE_IMAGE_PROFILE]
    user_loras, user_lora_mode, lora_source = merge_image_profile_loras(
        image_profile,
        [],
        requested_mode="replace",
    )
    first_lora = user_loras[0] if user_loras else None
    comfy_relative_dir = Path("local_video_api") / job_id
    comfy_input_dir = COMFY_INPUT / comfy_relative_dir
    comfy_input_dir.mkdir(parents=True, exist_ok=True)
    # The upload first produces a normalized, matted Qwen full-body canonical.
    # Eating keeps the canonical as Image 1/sampler and geometry as Image 2.
    # Grooming keeps the canonical as Image 1 and supplies an appearance-free
    # silhouette plus soft raised-foreleg depth as Image 2. Recovery retries may
    # promote that geometry to the sampler, but it never gains appearance authority.
    # SCAIL receives only normalized transparent Qwen action keyframes, never
    # the raw upload.
    effective_subject_config = job.get("effective_subject_config")
    if not isinstance(effective_subject_config, dict):
        raise RuntimeError(
            "Uploaded Qwen job lacks its queue-time effective subject config; "
            "submit the upload again instead of reading a mutable live Profile."
        )
    if "profile_fur_constraints" not in effective_subject_config:
        raise RuntimeError(
            "Persisted subject profile lacks its frozen fur constraints; "
            "refusing to re-read a mutable live profile."
        )
    profile = {
        "id": effective_subject_config.get("profile_id"),
        "fur_constraints": json.loads(
            json.dumps(effective_subject_config.get("profile_fur_constraints") or {})
        ),
    }
    upload_validation = effective_subject_config.get("upload_validation")
    if (
        not isinstance(upload_validation, dict)
        or upload_validation.get("policy") != QWEN_FULL_BODY_FIRST_FRAME_PIPELINE
        or (
            GENERATION_VISUAL_AUDITS_BLOCKING
            and upload_validation.get("expected_surface_gate_actions") != actions
        )
    ):
        raise RuntimeError(
            "Uploaded Qwen job lacks its complete queue-time per-action "
            "surface validation contract."
        )
    if GENERATION_VISUAL_AUDITS_BLOCKING:
        for action in actions:
            require_uploaded_fur_surface_gate(
                effective_subject_config["actions"][action],
                action,
            )
    pose_reference_inputs: dict[str, str] = {}
    pose_reference_source_paths: dict[str, str] = {}
    pose_reference_source_sha256s: dict[str, str] = {}
    pose_reference_input_sha256s: dict[str, str] = {}
    pose_reference_policies: dict[str, str] = {}
    pose_reference_guide_styles: dict[str, str] = {}
    pose_reference_metadata: dict[str, Any] = {}
    for action in actions:
        action_config = effective_subject_config["actions"][action]
        source_reference = Path(action_config["driving_animation"])
        if not source_reference.is_file():
            raise FileNotFoundError(
                f"Frozen {action} white-maquette driving animation is missing: "
                f"{source_reference}"
            )
        source_digest = file_sha256(source_reference)
        expected_source_digest = str(
            action_config.get("driving_animation_sha256") or ""
        ).strip()
        if expected_source_digest and source_digest != expected_source_digest:
            raise RuntimeError(
                f"Frozen {action} white-maquette driving animation digest changed."
            )
        white_maquette_driver = (
            action_config.get("driving_reference_kind")
            == "uploaded_white_maquette"
        )
        pose_guide = comfy_input_dir / f"{action}_qwen_pose_guide.png"
        if action == "eating":
            guide_metadata = prepare_eating_qwen_geometry_guide(
                job_id,
                source_reference,
                pose_guide,
                frames=int(effective_subject_config["frames"]),
                fps=float(effective_subject_config["fps"]),
            )
            pose_reference_policies[action] = (
                "main_case_eating_driver_geometry_image2_guide_v4"
            )
            pose_reference_guide_styles[action] = (
                "texture_free_low_frequency_white_maquette_geometry_with_explicit_"
                "muzzle_overlap_on_pure_green"
            )
            pose_reference_metadata[action] = {
                "policy": "main_case_eating_driver_geometry_image2_guide_v4",
                "generator_policy": guide_metadata.get("policy"),
                "generator_method": guide_metadata.get("method"),
                "image_role": "image2_action_geometry_guide",
                "identity_role": "image1_qwen_full_body_canonical",
                "palette_audit_role": "qwen_full_body_canonical",
                "source_rgb_preserved": False,
                "driver_silhouette_used": True,
                "reference_space_motion": False,
                "main_case_driver_used": not white_maquette_driver,
                "uploaded_white_maquette_driver_used": white_maquette_driver,
                "driver_appearance_authority": False,
                "source_sha256": source_digest,
                "guide_sha256": guide_metadata.get("guide_sha256"),
                "generator_metadata": guide_metadata,
            }
        elif action == "grooming":
            guide_metadata = prepare_grooming_qwen_geometry_guide(
                job_id,
                source_reference,
                pose_guide,
                frames=int(effective_subject_config["frames"]),
                fps=float(effective_subject_config["fps"]),
            )
            private_pose_gate = guide_metadata.get("private_pose_gate")
            if (
                not isinstance(private_pose_gate, dict)
                or private_pose_gate.get("exposed_to_qwen") is not False
                or not private_pose_gate.get("guide_sha256")
            ):
                raise RuntimeError(
                    "Grooming geometry preparation lacks its isolated private pose gate."
                )
            private_completion_mask = guide_metadata.get("private_completion_mask")
            if (
                not isinstance(private_completion_mask, dict)
                or private_completion_mask.get("exposed_to_qwen") is not False
                or private_completion_mask.get("loaded_as_qwen_image") is not False
                or not private_completion_mask.get("sha256")
            ):
                raise RuntimeError(
                    "Grooming geometry preparation lacks its private completion mask."
                )
            pose_reference_policies[action] = GROOMING_QWEN_POSE_REFERENCE_POLICY
            pose_reference_guide_styles[action] = GROOMING_QWEN_GUIDE_STYLE
            pose_reference_metadata[action] = {
                "policy": GROOMING_QWEN_POSE_REFERENCE_POLICY,
                "generator_policy": guide_metadata.get("policy"),
                "source_rgb_preserved": False,
                "image_role": "image2_action_geometry_guide",
                "identity_role": "image1_qwen_full_body_canonical",
                "palette_audit_role": "qwen_full_body_canonical",
                "driver_silhouette_used": True,
                "main_case_driver_used": not white_maquette_driver,
                "uploaded_white_maquette_driver_used": white_maquette_driver,
                "driver_appearance_authority": False,
                "source_sha256": source_digest,
                "guide_sha256": guide_metadata.get("guide_sha256"),
                "qwen_guide_internal_luma_variation": False,
                "qwen_guide_internal_luma_variation_role": "none",
                "qwen_guide_appearance_bearing_internal_luma_variation": False,
                "qwen_guide_synthetic_foreleg_semantic": False,
                "qwen_guide_synthetic_foreleg_semantic_method": None,
                "qwen_guide_structural_negative_space": True,
                "qwen_guide_structural_negative_space_method": (
                    GROOMING_QWEN_NEGATIVE_SPACE_METHOD
                ),
                "qwen_guide_closed_foreleg_capsule_exposed": False,
                "qwen_guide_internal_green_separator_exposed": True,
                "qwen_guide_raised_foreleg_validation_role": (
                    "public_open_curve_geometry_plus_private_cv"
                ),
                "qwen_foreleg_repair_enabled": True,
                "qwen_foreleg_repair_mode": (
                    "attempt3_local_if_identity_safe_then_attempt4_5_"
                    "seed_bound_pose_transfer_then_pose_safe_identity_refine"
                ),
                "qwen_foreleg_repair_retained_for_legacy_jobs_only": True,
                "qwen_foreleg_repair_policy": (
                    GROOMING_CANONICAL_FORELEG_COMPLETION_POLICY
                ),
                "qwen_foreleg_repair_attempt": (GROOMING_QWEN_FORELEG_REPAIR_ATTEMPT),
                "qwen_foreleg_repair_denoise": (GROOMING_QWEN_FORELEG_REPAIR_DENOISE),
                "qwen_foreleg_repair_source": (
                    "best_prior_identity_palette_safe_normalized_candidate"
                ),
                "qwen_foreleg_repair_mask_dilation_px": (
                    GROOMING_QWEN_FORELEG_REPAIR_MASK_DILATION_PX
                ),
                "qwen_foreleg_repair_anatomy_policy": (
                    GROOMING_QWEN_FORELEG_REPAIR_ANATOMY_POLICY
                ),
                "qwen_foreleg_repair_seed_source": (
                    "identity_safe_candidate_unchanged_before_masked_sampling"
                ),
                "qwen_foreleg_repair_expanded_context_full_denoise_mask": True,
                "qwen_foreleg_repair_identity_or_geometry_seed_inserted": False,
                "qwen_foreleg_repair_noise_mask_expansion_px": (
                    GROOMING_QWEN_FORELEG_REPAIR_NOISE_MASK_EXPANSION_PX
                ),
                "qwen_foreleg_repair_noise_mask_feather_sigma_px": (
                    GROOMING_QWEN_FORELEG_REPAIR_NOISE_MASK_FEATHER_SIGMA_PX
                ),
                "qwen_foreleg_repair_style_policy": (
                    "disabled_for_canonical_masked_completion"
                ),
                "qwen_foreleg_repair_style_lora": None,
                "qwen_foreleg_repair_style_strength": 0.0,
                "qwen_foreleg_repair_style_scope": "none",
                "qwen_foreleg_repair_composite_gate_policy": None,
                "qwen_foreleg_repair_composite_recovery_enabled": False,
                "qwen_foreleg_repair_composite_edge_density_ratio_min": None,
                "qwen_foreleg_repair_composite_planted_paw_gradient_p90_max": None,
                "qwen_foreleg_repair_composite_raised_paw_gradient_p90_max": None,
                "canonical_completion_edge_evidence_policy": (
                    GROOMING_CANONICAL_COMPLETION_EDGE_EVIDENCE_POLICY
                ),
                "canonical_completion_edge_density_ratio_min": (
                    GROOMING_CANONICAL_COMPLETION_EDGE_DENSITY_RATIO_MIN
                ),
                "canonical_completion_independent_fur_surface_required": True,
                "canonical_direct_structural_edge_evidence": {
                    "policy": (
                        GROOMING_CANONICAL_DIRECT_STRUCTURAL_EDGE_EVIDENCE_POLICY
                    ),
                    "maximum_attempt": (
                        GROOMING_CANONICAL_DIRECT_EDGE_EVIDENCE_MAX_ATTEMPT
                    ),
                    "raw_edge_density_ratio_min": (
                        GROOMING_CANONICAL_DIRECT_EDGE_DENSITY_RATIO_MIN
                    ),
                    "structural_edge_density_min": (
                        GROOMING_CANONICAL_DIRECT_STRUCTURAL_EDGE_DENSITY_MIN
                    ),
                    "chamfer_coverage_min": (
                        GROOMING_CANONICAL_DIRECT_CHAMFER_COVERAGE_MIN
                    ),
                    "chamfer_mean_distance_max": (
                        GROOMING_CANONICAL_DIRECT_CHAMFER_MEAN_DISTANCE_MAX
                    ),
                    "chamfer_p90_distance_max": (
                        GROOMING_CANONICAL_DIRECT_CHAMFER_P90_DISTANCE_MAX
                    ),
                    "offguide_organic_paw_evidence": {
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
                        "fail_closed": True,
                    },
                    "aligned_organic_paw_evidence": {
                        "raw_pose_gate_must_pass_without_issue": True,
                        "strict_guide_alignment_required": True,
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
                        "fail_closed": True,
                    },
                    "achromatic_palette_normalization": {
                        "policy": (
                            GROOMING_CANONICAL_DIRECT_ORGANIC_PAW_PALETTE_POLICY
                        ),
                        "operation": "foreground_lab_l_histogram_match_only",
                        "achromatic_fraction_min": (
                            GROOMING_CANONICAL_DIRECT_ORGANIC_PAW_ACHROMATIC_MIN
                        ),
                        "input_balance_l1_max": (
                            GROOMING_CANONICAL_DIRECT_ORGANIC_PAW_BALANCE_L1_MAX
                        ),
                        "source_alpha_geometry_must_remain_exact": True,
                        "source_ab_roundtrip_delta_max": 2,
                        "new_hue_or_marking_drift_allowed": False,
                        "pose_and_identity_revalidation_required": True,
                        "independent_fur_surface_gate_still_required": True,
                        "fail_closed": True,
                    },
                    "identity_palette_gate_still_required": True,
                    "independent_fur_surface_gate_still_required": True,
                    "global_pose_threshold_changed": False,
                    "fail_closed": True,
                },
                "canonical_direct_surface_nonexpansion": {
                    "policy": (GROOMING_CANONICAL_DIRECT_SURFACE_NONEXPANSION_POLICY),
                    "maximum_attempt": (
                        GROOMING_CANONICAL_DIRECT_EDGE_EVIDENCE_MAX_ATTEMPT
                    ),
                    "only_recoverable_issue": "brush_texture_coherence",
                    "requires_strict_pose_recovery": True,
                    "requires_identity_palette_pass": True,
                    "requires_digest_bound_transparent_canonical": True,
                    "coherent_fraction_max": (
                        GROOMING_CANONICAL_DIRECT_SURFACE_COHERENT_FRACTION_MAX
                    ),
                    "coherent_fraction_delta_max": (
                        GROOMING_CANONICAL_DIRECT_SURFACE_COHERENT_FRACTION_DELTA_MAX
                    ),
                    "coherence_mean_max": (
                        GROOMING_CANONICAL_DIRECT_SURFACE_COHERENCE_MEAN_MAX
                    ),
                    "coherence_mean_delta_max": (
                        GROOMING_CANONICAL_DIRECT_SURFACE_COHERENCE_MEAN_DELTA_MAX
                    ),
                    "orientation_max": (
                        GROOMING_CANONICAL_DIRECT_SURFACE_ORIENTATION_MAX
                    ),
                    "orientation_delta_max": (
                        GROOMING_CANONICAL_DIRECT_SURFACE_ORIENTATION_DELTA_MAX
                    ),
                    "hough_radial_delta_max": (
                        GROOMING_CANONICAL_DIRECT_SURFACE_HOUGH_RADIAL_DELTA_MAX
                    ),
                    "canonical_long_energy_max": (
                        GROOMING_CANONICAL_DIRECT_SURFACE_CANONICAL_LONG_ENERGY_MAX
                    ),
                    "independent_final_video_fur_audit_still_required": True,
                    "global_surface_thresholds_changed": False,
                    "fail_closed": True,
                },
                "canonical_full_pose_recovery_enabled": True,
                "canonical_full_pose_rescue_policy": (
                    GROOMING_CANONICAL_FULL_POSE_RESCUE_POLICY
                ),
                "canonical_full_pose_rescue_attempt": (
                    GROOMING_CANONICAL_FULL_POSE_RESCUE_ATTEMPT
                ),
                "canonical_full_pose_identity_refine_policy": (
                    GROOMING_CANONICAL_FULL_POSE_REFINE_POLICY
                ),
                "canonical_full_pose_identity_refine_attempt": (
                    GROOMING_CANONICAL_FULL_POSE_REFINE_ATTEMPT
                ),
                "canonical_full_pose_refine_source_policy": (
                    GROOMING_CANONICAL_POSE_SAFE_REFINE_SOURCE_POLICY
                ),
                "canonical_full_pose_refine_image3": (
                    "digest_bound_soft_foreleg_depth_action_corroboration"
                ),
                "canonical_full_pose_refine_uploaded_front_exposed": False,
                "canonical_full_pose_refine_identity_source": (
                    "approved_canonical_three_view_consensus_only"
                ),
                "canonical_pose_transfer_geometry_smooth_sigma_px": (
                    GROOMING_CANONICAL_POSE_TRANSFER_GEOMETRY_SMOOTH_SIGMA_PX
                ),
                "canonical_pose_transfer_source_alpha_geometry_preserved": False,
                "canonical_pose_transfer_action_alpha_overlap_blocking": False,
                "canonical_pose_transfer_public_geometry_guide_required": True,
                "canonical_pose_transfer_foreleg_geometry_guide_required": True,
                "canonical_pose_transfer_internal_green_channel_exposed": False,
                "canonical_pose_transfer_soft_foreleg_depth_exposed": True,
                "canonical_pose_transfer_source_rgb_exposed": False,
                "canonical_pose_transfer_source_chroma_exposed": False,
                "canonical_pose_transfer_source_raw_luma_exposed": False,
                "canonical_pose_transfer_seed_policy": (
                    GROOMING_CANONICAL_POSE_TRANSFER_SEED_POLICY
                ),
                "canonical_pose_transfer_seed_bound_matte_policy": (
                    GROOMING_CANONICAL_SEED_BOUND_MATTE_POLICY
                ),
                "canonical_pose_transfer_seed_bound_alpha_provenance": (
                    GROOMING_CANONICAL_SEED_BOUND_MATTE_ALPHA_PROVENANCE
                ),
                "canonical_pose_transfer_rejected_candidate_used_as_sampler": True,
                "canonical_pose_transfer_rejected_candidate_scope": (
                    "attempt4_identity_palette_only_after_seed_bound_matte_pose_"
                    "quality_and_independent_surface_pass"
                ),
                "canonical_full_pose_recovery_attempt_budget": (
                    GROOMING_CANONICAL_FULL_POSE_REFINE_ATTEMPT
                ),
                "canonical_full_pose_recovery_style_lora": None,
                "canonical_full_pose_recovery_global_thresholds_changed": False,
                "qwen_foreleg_repair_global_edge_density_min": (
                    GROOMING_QWEN_FORELEG_REPAIR_GLOBAL_EDGE_DENSITY_MIN
                ),
                "qwen_foreleg_repair_global_pose_threshold_changed": False,
                "qwen_foreleg_repair_synthetic_capsule_exposed": False,
                "qwen_foreleg_repair_open_channel_anatomy_image3": True,
                "qwen_foreleg_repair_private_cv_luma_image3": False,
                "main_case_appearance_exposed_to_qwen": False,
                "white_maquette_appearance_exposed_to_qwen": False,
                "private_pose_gate_role": "cv_validation_only",
                "private_pose_gate_exposed_to_qwen": False,
                "private_pose_gate_neutral_copy_exposed_on_repair_attempt": False,
                "public_open_channel_copy_exposed_on_repair_attempt": True,
                "public_open_channel_copy_role": (
                    "attempt3_local_completion_geometry_only"
                ),
                "private_pose_gate_guide_path": private_pose_gate.get("guide_path"),
                "private_pose_gate_guide_sha256": private_pose_gate.get("guide_sha256"),
                "private_completion_mask_path": private_completion_mask.get("path"),
                "private_completion_mask_sha256": private_completion_mask.get("sha256"),
                "private_completion_mask_policy": (
                    GROOMING_PRIVATE_COMPLETION_MASK_POLICY
                ),
                "private_completion_mask_exposed_to_qwen": False,
                "generator_metadata": guide_metadata,
            }
        else:
            raise ValueError(f"Unsupported subject replacement action: {action}")
        pose_reference_inputs[action] = (
            comfy_relative_dir / pose_guide.name
        ).as_posix()
        pose_reference_input_sha256s[action] = file_sha256(pose_guide)
        pose_reference_source_paths[action] = str(source_reference)
        pose_reference_source_sha256s[action] = source_digest

    qwen_reference_prompt_en = str(image_profile["default_prompt_en"])
    if upload_integrity["view_mode"] == "front_left_right":
        qwen_reference_prompt_en = (
            qwen_reference_prompt_en + " " + QWEN_MULTI_VIEW_CANONICAL_PROMPT
        )
    profile_fur_lock, profile_fur_negative = subject_profile_fur_guard(profile)
    update_job(
        job_id,
        reference_pipeline=QWEN_FULL_BODY_FIRST_FRAME_PIPELINE,
        reference_pipeline_version=QWEN_FULL_BODY_FIRST_FRAME_PIPELINE_VERSION,
        reference_generation_status="running",
        reference_image_profile=image_profile["id"],
        reference_image_profile_revision=image_profile.get("revision"),
        reference_image_profile_label=image_profile["label"],
        reference_image_profile_source_job_id=image_profile.get("source_job_id"),
        seed=int(image_profile["seed"]),
        qwen_reference_prompt_en=qwen_reference_prompt_en,
        canonical_prompt_source="uploaded_identity_qwen_full_body",
        canonical_identity_view_mode=upload_integrity["view_mode"],
        canonical_identity_view_roles=upload_integrity["view_roles"],
        canonical_identity_view_count=len(upload_integrity["view_roles"]),
        canonical_identity_view_policy=UPLOADED_IDENTITY_VIEW_POLICY,
        canonical_identity_view_node_inputs=(
            ["image1", "image2", "image3"]
            if upload_integrity["view_mode"] == "front_left_right"
            else ["image1"]
        ),
        canonical_identity_view_prompt_applied=(
            upload_integrity["view_mode"] == "front_left_right"
        ),
        main_case_raw_identity_chain=False,
        bypass_generated_canonical=False,
        generated_canonical_bypassed=False,
        qwen_reference_generation_for_subject_replacement=True,
        direct_scail_uploaded_identity_reference=False,
        user_prompt_excluded_from_reference=True,
        subject_profile_fur_lock=profile_fur_lock,
        subject_profile_fur_negative=profile_fur_negative,
        subject_profile_appearance_role="action_geometry_only",
        subject_profile_identity_terms_applied=False,
        uploaded_identity_appearance_authority=True,
        qwen_reference_poses=[
            subject_replacement_pose_prompt(action) for action in actions
        ],
        pose_reference_inputs=pose_reference_inputs,
        pose_reference_source_paths=pose_reference_source_paths,
        pose_reference_source_sha256s=pose_reference_source_sha256s,
        pose_reference_input_sha256s=pose_reference_input_sha256s,
        pose_reference_metadata=pose_reference_metadata,
        reference_pose_source="per_action_main_case_driver_geometry",
        pose_reference_policy=pose_reference_policies,
        pose_reference_guide_style="per_action_main_case_driver_geometry_v2",
        pose_reference_guide_styles=pose_reference_guide_styles,
        pose_reference_role="per_action_geometry_input_role",
        pose_identity_input_policy=POSE_IDENTITY_INPUT_POLICY_CANONICAL,
        pose_identity_input_policy_scope=(
            "eating_and_grooming_canonical_image1_geometry_image2"
        ),
        pose_identity_input_policies={
            action: POSE_IDENTITY_INPUT_POLICY_CANONICAL for action in actions
        },
        pose_identity_input_roles={
            action: "image1_qwen_full_body_canonical" for action in actions
        },
        pose_identity_geometry_roles={
            action: "image2_main_case_driver_geometry_guide" for action in actions
        },
        eating_pose_lora_policy=("canonical_identity_plus_main_case_geometry_v1"),
        eating_identity_refine_enabled=True,
        eating_identity_refine_active=False,
        eating_identity_refine_policy=(EATING_CANONICAL_IDENTITY_REFINE_POLICY),
        eating_identity_refine_attempt=(EATING_CANONICAL_IDENTITY_REFINE_ATTEMPT),
        eating_identity_refine_steps=EATING_CANONICAL_IDENTITY_REFINE_STEPS,
        eating_identity_refine_denoise=(EATING_CANONICAL_IDENTITY_REFINE_DENOISE),
        eating_identity_refine_style_lora=None,
        eating_identity_refine_global_thresholds_changed=False,
        grooming_pose_lora_policy=(
            "canonical_identity_open_channel_seed_bound_pose_safe_refine_v11"
        ),
        canonical_action_identity_palette_policy=(
            CANONICAL_ACTION_IDENTITY_PALETTE_POLICY
        ),
        canonical_action_achromatic_balance_l1_max=(
            CANONICAL_ACTION_ACHROMATIC_BALANCE_L1_MAX
        ),
        grooming_foreleg_repair_active=False,
        grooming_foreleg_repair_policy=(GROOMING_CANONICAL_FORELEG_COMPLETION_POLICY),
        grooming_foreleg_repair_attempt=(GROOMING_QWEN_FORELEG_REPAIR_ATTEMPT),
        grooming_foreleg_repair_edit_mask_sha256=None,
        grooming_foreleg_repair_anatomy_sha256=None,
        grooming_foreleg_repair_noise_mask_sha256=None,
        grooming_foreleg_repair_style_policy=(
            "disabled_for_canonical_masked_completion"
        ),
        grooming_foreleg_repair_style_lora=None,
        grooming_foreleg_repair_style_strength=0.0,
        grooming_foreleg_repair_style_scope="none",
        grooming_foreleg_repair_composite_gate_policy=None,
        grooming_foreleg_repair_composite_recovery_enabled=False,
        grooming_foreleg_repair_composite_edge_density_ratio_min=None,
        grooming_foreleg_repair_composite_planted_paw_gradient_p90_max=None,
        grooming_foreleg_repair_composite_raised_paw_gradient_p90_max=None,
        grooming_canonical_completion_edge_evidence_policy=(
            GROOMING_CANONICAL_COMPLETION_EDGE_EVIDENCE_POLICY
        ),
        grooming_canonical_completion_edge_density_ratio_min=(
            GROOMING_CANONICAL_COMPLETION_EDGE_DENSITY_RATIO_MIN
        ),
        grooming_canonical_completion_independent_fur_surface_required=True,
        grooming_canonical_direct_structural_edge_evidence_policy=(
            GROOMING_CANONICAL_DIRECT_STRUCTURAL_EDGE_EVIDENCE_POLICY
        ),
        grooming_canonical_direct_edge_density_ratio_min=(
            GROOMING_CANONICAL_DIRECT_EDGE_DENSITY_RATIO_MIN
        ),
        grooming_canonical_direct_structural_edge_density_min=(
            GROOMING_CANONICAL_DIRECT_STRUCTURAL_EDGE_DENSITY_MIN
        ),
        grooming_canonical_direct_chamfer_coverage_min=(
            GROOMING_CANONICAL_DIRECT_CHAMFER_COVERAGE_MIN
        ),
        grooming_canonical_direct_chamfer_mean_distance_max=(
            GROOMING_CANONICAL_DIRECT_CHAMFER_MEAN_DISTANCE_MAX
        ),
        grooming_canonical_direct_chamfer_p90_distance_max=(
            GROOMING_CANONICAL_DIRECT_CHAMFER_P90_DISTANCE_MAX
        ),
        grooming_canonical_direct_organic_paw_palette_normalization_policy=(
            GROOMING_CANONICAL_DIRECT_ORGANIC_PAW_PALETTE_POLICY
        ),
        grooming_canonical_direct_offguide_paw_lab_delta_max=(
            GROOMING_CANONICAL_DIRECT_OFFGUIDE_PAW_LAB_DELTA_MAX
        ),
        grooming_canonical_direct_offguide_paw_histogram_max=(
            GROOMING_CANONICAL_DIRECT_OFFGUIDE_PAW_HISTOGRAM_MAX
        ),
        grooming_canonical_direct_offguide_paw_texture_ratio_max=(
            GROOMING_CANONICAL_DIRECT_OFFGUIDE_PAW_TEXTURE_RATIO_MAX
        ),
        grooming_canonical_direct_offguide_achromatic_balance_l1_max=(
            GROOMING_CANONICAL_DIRECT_ORGANIC_PAW_BALANCE_L1_MAX
        ),
        grooming_canonical_direct_aligned_structural_edge_density_min=(
            GROOMING_CANONICAL_DIRECT_ALIGNED_STRUCTURAL_EDGE_DENSITY_MIN
        ),
        grooming_canonical_direct_aligned_paw_lab_delta_max=(
            GROOMING_CANONICAL_DIRECT_ALIGNED_PAW_LAB_DELTA_MAX
        ),
        grooming_canonical_direct_aligned_paw_histogram_max=(
            GROOMING_CANONICAL_DIRECT_ALIGNED_PAW_HISTOGRAM_MAX
        ),
        grooming_canonical_direct_aligned_paw_texture_ratio_range=[
            GROOMING_CANONICAL_DIRECT_ALIGNED_PAW_TEXTURE_RATIO_MIN,
            GROOMING_CANONICAL_DIRECT_ALIGNED_PAW_TEXTURE_RATIO_MAX,
        ],
        grooming_canonical_direct_palette_normalization_operation=(
            "foreground_lab_l_histogram_match_only"
        ),
        grooming_canonical_direct_palette_normalization_alpha_exact=True,
        grooming_canonical_direct_palette_normalization_ab_delta_max=2,
        grooming_canonical_direct_palette_normalization_revalidation_required=True,
        grooming_achromatic_palette_normalization=None,
        grooming_canonical_direct_edge_evidence_max_attempt=(
            GROOMING_CANONICAL_DIRECT_EDGE_EVIDENCE_MAX_ATTEMPT
        ),
        grooming_canonical_direct_identity_palette_required=True,
        grooming_canonical_direct_independent_fur_surface_required=True,
        grooming_canonical_direct_global_thresholds_changed=False,
        grooming_canonical_direct_surface_nonexpansion_policy=(
            GROOMING_CANONICAL_DIRECT_SURFACE_NONEXPANSION_POLICY
        ),
        grooming_canonical_direct_surface_only_recoverable_issue=(
            "brush_texture_coherence"
        ),
        grooming_canonical_direct_surface_coherent_fraction_max=(
            GROOMING_CANONICAL_DIRECT_SURFACE_COHERENT_FRACTION_MAX
        ),
        grooming_canonical_direct_surface_coherent_fraction_delta_max=(
            GROOMING_CANONICAL_DIRECT_SURFACE_COHERENT_FRACTION_DELTA_MAX
        ),
        grooming_canonical_direct_surface_requires_pose_and_identity=True,
        grooming_canonical_direct_surface_requires_canonical_digest=True,
        grooming_canonical_direct_surface_final_video_audit_required=True,
        grooming_canonical_direct_surface_global_thresholds_changed=False,
        grooming_full_pose_recovery_enabled=True,
        grooming_full_pose_recovery_active=False,
        grooming_full_pose_rescue_policy=(GROOMING_CANONICAL_FULL_POSE_RESCUE_POLICY),
        grooming_full_pose_rescue_attempt=(GROOMING_CANONICAL_FULL_POSE_RESCUE_ATTEMPT),
        grooming_full_pose_identity_refine_policy=(
            GROOMING_CANONICAL_FULL_POSE_REFINE_POLICY
        ),
        grooming_full_pose_identity_refine_attempt=(
            GROOMING_CANONICAL_FULL_POSE_REFINE_ATTEMPT
        ),
        grooming_full_pose_recovery_attempt_budget=(
            GROOMING_CANONICAL_FULL_POSE_REFINE_ATTEMPT
        ),
        grooming_full_pose_recovery_style_lora=None,
        grooming_full_pose_recovery_global_thresholds_changed=False,
        grooming_pose_transfer_seed_sha256=None,
        grooming_pose_transfer_action_reference_sha256=None,
        grooming_pose_transfer_seed_policy=(
            GROOMING_CANONICAL_POSE_TRANSFER_SEED_POLICY
        ),
        grooming_pose_transfer_seed_bound_matte_policy=(
            GROOMING_CANONICAL_SEED_BOUND_MATTE_POLICY
        ),
        grooming_pose_transfer_seed_bound_alpha_provenance=(
            GROOMING_CANONICAL_SEED_BOUND_MATTE_ALPHA_PROVENANCE
        ),
        grooming_pose_safe_identity_refine_source_policy=(
            GROOMING_CANONICAL_POSE_SAFE_REFINE_SOURCE_POLICY
        ),
        grooming_pose_safe_identity_refine_structural_edge_density_min=(
            GROOMING_CANONICAL_POSE_SAFE_STRUCTURAL_EDGE_DENSITY_MIN
        ),
        grooming_pose_safe_identity_refine_allowed_palette_issues=sorted(
            GROOMING_CANONICAL_POSE_SAFE_IDENTITY_ONLY_ISSUES
        ),
        grooming_pose_safe_identity_refine_probe=None,
        grooming_pose_transfer_rejected_candidate_used_as_sampler=False,
        grooming_foreleg_repair_global_edge_density_min=(
            GROOMING_QWEN_FORELEG_REPAIR_GLOBAL_EDGE_DENSITY_MIN
        ),
        grooming_foreleg_repair_global_pose_threshold_changed=False,
        grooming_pose_gate_composite_recovery=None,
        pose_identity_palette_reference_roles={
            action: "qwen_full_body_canonical" for action in actions
        },
        image_profile_lora_source=lora_source,
        image_profile_suppresses_builtin_style=True,
        lock_retry_lora_strengths=bool(
            image_profile.get("lock_retry_lora_strengths", False)
        ),
        # Photography LoRAs can inflate a close-up source into a long-haired
        # character. Start at the conservative pose cap and allow progressively
        # lower-strength retries; palette and identity gates still fail closed.
        lock_canonical_retry_lora_strengths=False,
        lock_pose_retry_lora_strengths=False,
        pose_reference_denoise=QWEN_SUBJECT_REPLACEMENT_POSE_DENOISE,
        pose_reference_denoise_step=(QWEN_SUBJECT_REPLACEMENT_POSE_DENOISE_STEP),
        identity_palette_guard=bool(image_profile.get("identity_palette_guard", False)),
        identity_palette_policy=image_profile.get("identity_palette_policy"),
        user_loras=user_loras,
        requested_user_loras=user_loras,
        uploaded_user_loras=[],
        image_profile_user_loras=[
            dict(item) for item in image_profile.get("user_loras", [])
        ],
        user_lora=(first_lora["model_name"] if first_lora else None),
        user_lora_original_filename=(
            first_lora.get("original_filename") if first_lora else None
        ),
        user_lora_strength=(first_lora["strength"] if first_lora else None),
        user_lora_mode=user_lora_mode,
        user_lora_count=len(user_loras),
        active_user_lora_count=sum(
            float(item["strength"]) != 0.0 for item in user_loras
        ),
        qwen_steps=int(image_profile["qwen_steps"]),
        qwen_cfg=float(image_profile["qwen_cfg"]),
        qwen_denoise=float(image_profile["qwen_denoise"]),
        canonical_user_lora_total_cap=(
            QWEN_SUBJECT_REPLACEMENT_CANONICAL_LORA_TOTAL_CAP
        ),
        canonical_user_lora_per_lora_cap=(
            QWEN_SUBJECT_REPLACEMENT_CANONICAL_LORA_TOTAL_CAP
        ),
        pose_user_lora_first_total_cap=(QWEN_SUBJECT_REPLACEMENT_POSE_LORA_TOTAL_CAP),
        pose_user_lora_first_per_lora_cap=(
            QWEN_SUBJECT_REPLACEMENT_POSE_LORA_TOTAL_CAP
        ),
    )

    # Use the existing, fully validated Qwen-only branch with the parent job id.
    # This preserves queue cancellation and keeps every output in one task directory.
    run_job(job_id, force_qwen_reference=True)
    raise_if_cancelled(job_id)

    refreshed = get_job(job_id)
    canonical_bypassed = bool(
        refreshed.get("generated_canonical_bypassed")
        or refreshed.get("bypass_generated_canonical")
    )
    if canonical_bypassed:
        raise RuntimeError(
            "Uploaded subject reference generation bypassed the required Qwen "
            "full-body canonical."
        )
    canonical_reference: Path | None = None
    action_references: dict[str, str] = {}
    for output in refreshed.get("outputs", []):
        filename = output.get("filename")
        if not isinstance(filename, str):
            continue
        candidate = job_dir / filename
        if output.get("type") == "cartoon_reference" and candidate.is_file():
            canonical_reference = candidate
        action = output.get("action")
        if (
            output.get("type") == "transparent_keyframe"
            and action in actions
            and candidate.is_file()
        ):
            action_references[str(action)] = str(candidate)

    missing_actions = [action for action in actions if action not in action_references]
    if canonical_reference is None:
        raise RuntimeError(
            "Qwen reference generation did not produce a canonical image."
        )
    if missing_actions:
        raise RuntimeError(
            "Qwen reference generation did not produce transparent action references: "
            + ", ".join(missing_actions)
        )
    canonical_public_reference = Path(
        str(refreshed.get("canonical_public_reference_path") or "")
    )
    canonical_public_reference_sha256 = (
        str(refreshed.get("canonical_public_reference_sha256") or "").strip().lower()
    )
    if (
        canonical_public_reference != canonical_reference
        or not re.fullmatch(
            r"[0-9a-f]{64}",
            canonical_public_reference_sha256,
        )
        or not canonical_public_reference.is_file()
        or file_sha256(canonical_public_reference) != canonical_public_reference_sha256
    ):
        raise RuntimeError(
            "Qwen transparent full-body canonical provenance changed after generation."
        )
    canonical_qwen_image1 = Path(str(refreshed.get("canonical_qwen_image1_path") or ""))
    canonical_qwen_image1_comfy = Path(
        str(refreshed.get("canonical_qwen_image1_comfy_path") or "")
    )
    canonical_qwen_image1_sha256 = (
        str(refreshed.get("canonical_qwen_image1_sha256") or "").strip().lower()
    )
    canonical_qwen_image1_comfy_sha256 = (
        str(refreshed.get("canonical_qwen_image1_comfy_sha256") or "").strip().lower()
    )
    expected_canonical_qwen_image1 = job_dir / "cartoon_reference.png"
    expected_canonical_qwen_image1_comfy = comfy_input_dir / "cartoon_reference.png"
    if (
        canonical_qwen_image1 != expected_canonical_qwen_image1
        or canonical_qwen_image1_comfy != expected_canonical_qwen_image1_comfy
        or not re.fullmatch(r"[0-9a-f]{64}", canonical_qwen_image1_sha256)
        or canonical_qwen_image1_sha256 != canonical_qwen_image1_comfy_sha256
        or not canonical_qwen_image1.is_file()
        or not canonical_qwen_image1_comfy.is_file()
        or file_sha256(canonical_qwen_image1) != canonical_qwen_image1_sha256
        or file_sha256(canonical_qwen_image1_comfy) != canonical_qwen_image1_sha256
    ):
        raise RuntimeError(
            "Qwen full-body Image 1 provenance changed between its job and "
            "ComfyUI copies."
        )
    with Image.open(canonical_qwen_image1) as loaded_image1:
        if loaded_image1.size != (
            TRANSPARENT_GIF_SIZE,
            TRANSPARENT_GIF_SIZE,
        ):
            raise RuntimeError(
                "Qwen full-body action Image 1 was not normalized to 512x512."
            )
    with Image.open(canonical_reference) as loaded_canonical:
        if loaded_canonical.size != (TRANSPARENT_GIF_SIZE, TRANSPARENT_GIF_SIZE):
            raise RuntimeError(
                "Qwen full-body canonical was not normalized to 512x512."
            )
        if loaded_canonical.mode not in {"LA", "PA", "RGBA"}:
            raise RuntimeError(
                "Qwen full-body canonical is opaque after normalization."
            )
        canonical_alpha = loaded_canonical.convert("RGBA").getchannel("A")
        canonical_alpha_minimum, canonical_alpha_maximum = canonical_alpha.getextrema()
        if canonical_alpha_minimum == 255 or canonical_alpha_maximum == 0:
            raise RuntimeError(
                "Qwen full-body canonical lacks a usable alpha subject after "
                "normalization."
            )
    for action, reference in action_references.items():
        with Image.open(reference) as loaded:
            if loaded.size != (TRANSPARENT_GIF_SIZE, TRANSPARENT_GIF_SIZE):
                raise RuntimeError(
                    f"Qwen {action} reference was not normalized to 512x512."
                )
            if loaded.mode not in {"LA", "PA", "RGBA"}:
                raise RuntimeError(
                    f"Qwen {action} reference is opaque; refusing to send it to SCAIL."
                )
            alpha = loaded.convert("RGBA").getchannel("A")
            alpha_minimum, alpha_maximum = alpha.getextrema()
            if alpha_minimum == 255 or alpha_maximum == 0:
                raise RuntimeError(
                    f"Qwen {action} reference lacks a usable alpha subject; "
                    "refusing to send it to SCAIL."
                )

    upload_validation = effective_subject_config.get("upload_validation")
    expected_surface_gate_actions = (
        list(upload_validation.get("expected_surface_gate_actions") or [])
        if isinstance(upload_validation, dict)
        else []
    )
    if (
        GENERATION_VISUAL_AUDITS_BLOCKING
        and expected_surface_gate_actions != actions
    ):
        raise RuntimeError(
            "Uploaded Qwen references lost their complete per-action fur surface "
            "gate contract."
        )
    if GENERATION_VISUAL_AUDITS_BLOCKING:
        for action in actions:
            require_uploaded_fur_surface_gate(
                effective_subject_config["actions"][action],
                action,
            )
    gated_surface_actions = (
        list(actions) if GENERATION_VISUAL_AUDITS_BLOCKING else []
    )
    fur_surface_quality_by_action: dict[str, Any] = {}
    selected_surface_metrics = refreshed.get("fur_surface_gate_selected_metrics")
    if (
        GENERATION_VISUAL_AUDITS_BLOCKING
        and not isinstance(selected_surface_metrics, dict)
    ):
        raise RuntimeError(
            "Qwen references lack selected fur surface metrics; refusing "
            "to enter SCAIL."
        )
    for action in gated_surface_actions:
        reference = action_references[action]
        selected_surface = selected_surface_metrics.get(action)
        if (
            not isinstance(selected_surface, dict)
            or selected_surface.get("acceptable") is not True
            or selected_surface.get("status") != "passed"
        ):
            raise RuntimeError(
                f"Qwen {action} reference lacks a passed fur surface verdict; "
                "refusing to enter SCAIL."
            )
        selected_gate = selected_surface.get("gate")
        reference_path = Path(reference)
        reference_digest = file_sha256(reference_path)
        selected_candidate_path = Path(
            str(selected_surface.get("selected_candidate_path") or "")
        )
        selected_candidate_digest = str(
            selected_surface.get("selected_candidate_sha256") or ""
        ).strip()
        if (
            not isinstance(selected_gate, dict)
            or selected_gate.get("policy") != FUR_SURFACE_GATE_POLICY
            or selected_gate.get("gate_version") != FUR_SURFACE_GATE_VERSION
            or selected_gate.get("threshold_version") != FUR_SURFACE_THRESHOLD_VERSION
            or not selected_candidate_path.is_file()
            or not selected_candidate_digest
            or selected_gate.get("candidate_sha256") != selected_candidate_digest
            or file_sha256(selected_candidate_path) != selected_candidate_digest
            or selected_candidate_digest != reference_digest
        ):
            raise RuntimeError(
                f"Qwen {action} fur surface selected provenance changed; "
                "refusing to enter SCAIL."
            )
        defensive_surface = evaluate_main_case_fur_surface_candidate(
            reference_path,
            action=action,
            alpha_provenance=str(selected_gate.get("alpha_provenance") or ""),
            effective_subject_config=effective_subject_config,
        )
        if defensive_surface.get("acceptable") is not True:
            update_job(
                job_id,
                fur_surface_gate_final_quality={
                    "scope": "qwen_action_reference_pre_scail",
                    "status": "failed",
                    "policy": FUR_SURFACE_GATE_POLICY,
                    "gate_version": FUR_SURFACE_GATE_VERSION,
                    "threshold_version": FUR_SURFACE_THRESHOLD_VERSION,
                    "failure": {
                        "action": action,
                        "reference_path": str(reference_path),
                        "reference_sha256": reference_digest,
                        "selected_candidate_path": str(selected_candidate_path),
                        "selected_candidate_sha256": selected_candidate_digest,
                        "defensive_surface": defensive_surface,
                        "blocks_scail": True,
                    },
                    "final_video_frame_audit": False,
                },
            )
            raise RuntimeError(
                f"Qwen {action} reference failed final fur surface quality "
                "before SCAIL: "
                + ", ".join(str(issue) for issue in defensive_surface.get("issues", []))
            )
        defensive_surface["selected_attempt"] = selected_surface.get("selected_attempt")
        defensive_surface["source_candidate_sha256"] = selected_candidate_digest
        defensive_surface["normalized_candidate_sha256"] = reference_digest
        defensive_surface["persisted_reference_path"] = str(reference_path)
        defensive_surface["persisted_reference_sha256"] = reference_digest
        defensive_surface["byte_identical_to_selected_candidate"] = True
        defensive_surface["pre_scail_revalidated"] = True
        fur_surface_quality_by_action[action] = defensive_surface

    fur_silhouette_quality_by_action: dict[str, Any] = {}
    for action, reference in (
        action_references.items() if GENERATION_VISUAL_AUDITS_BLOCKING else ()
    ):
        action_config = effective_subject_config["actions"][action]
        fur_gate = action_config.get("fur_silhouette_gate")
        if not isinstance(fur_gate, dict) or not fur_gate.get("enabled"):
            continue
        fur_reference = Path(str(fur_gate.get("reference") or ""))
        fur_quality = eating_fur_silhouette_quality(
            Path(reference),
            fur_reference,
        )
        fur_silhouette_quality_by_action[action] = fur_quality
        if not fur_quality["acceptable"]:
            raise RuntimeError(
                f"Qwen {action} reference failed the frozen profile fur "
                "silhouette standard: " + ", ".join(fur_quality["issues"])
            )

    effective_subject_config = json.loads(json.dumps(effective_subject_config))
    for action, reference in action_references.items():
        action_config = effective_subject_config["actions"][action]
        action_config.setdefault("profile_reference", action_config.get("reference"))
        action_config.setdefault(
            "profile_reference_sha256", action_config.get("reference_sha256")
        )
        action_config["reference"] = reference
        action_config["reference_sha256"] = file_sha256(Path(reference))
        action_config["reference_kind"] = "qwen_generated_upload"

    update_job(
        job_id,
        subject_source="qwen_generated_upload",
        effective_subject_source="qwen_generated_upload",
        subject_reference_paths=action_references,
        qwen_subject_reference_paths=action_references,
        qwen_reference_actions=list(actions),
        qwen_reference_generated_actions=list(action_references),
        canonical_reference_path=(
            str(canonical_reference) if canonical_reference is not None else None
        ),
        canonical_reference_sha256=canonical_public_reference_sha256,
        canonical_public_reference_role="normalized_transparent_full_body_reference",
        canonical_reference_generated=True,
        canonical_reference_used=True,
        canonical_reference_normalized=True,
        canonical_reference_transparent=True,
        qwen_action_keyframes_normalized=True,
        qwen_action_keyframes_transparent=True,
        qwen_action_reference_normalization_policy=(
            QWEN_ACTION_REFERENCE_NORMALIZATION_POLICY
        ),
        qwen_action_reference_byte_preserved=True,
        qwen_action_keyframe_sha256s={
            action: file_sha256(Path(reference))
            for action, reference in action_references.items()
        },
        reference_normalization_status="completed",
        uploaded_identity_anchor_path=str(upload_integrity["source_path"]),
        uploaded_identity_anchor_paths={
            role: str(view["source_path"])
            for role, view in upload_integrity["views"].items()
        },
        uploaded_identity_anchor_sha256s={
            role: view["source_sha256"]
            for role, view in upload_integrity["views"].items()
        },
        uploaded_identity_view_mode=upload_integrity["view_mode"],
        uploaded_identity_view_roles=upload_integrity["view_roles"],
        uploaded_identity_view_count=len(upload_integrity["view_roles"]),
        uploaded_identity_view_verdict=(
            "skipped_advisory"
            if identity_view_audit.get("status") == "skipped_advisory"
            else (
                "passed"
                if identity_view_audit["acceptable"]
                else "warning_advisory"
            )
        ),
        subject_fur_silhouette_quality=fur_silhouette_quality_by_action,
        subject_fur_surface_quality=fur_surface_quality_by_action,
        fur_surface_gate_status={
            action: (
                "passed"
                if GENERATION_VISUAL_AUDITS_BLOCKING
                else "skipped_advisory"
            )
            for action in actions
        },
        surface_texture_gate_status=(
            "passed"
            if GENERATION_VISUAL_AUDITS_BLOCKING
            else "skipped_advisory"
        ),
        fur_surface_gate_final_quality={
            "scope": "qwen_action_reference_pre_scail",
            "status": (
                "passed"
                if GENERATION_VISUAL_AUDITS_BLOCKING
                else "skipped_advisory"
            ),
            "blocking": GENERATION_VISUAL_AUDITS_BLOCKING,
            "generation_audit_policy": GENERATION_AUDIT_POLICY,
            "policy": FUR_SURFACE_GATE_POLICY,
            "gate_version": FUR_SURFACE_GATE_VERSION,
            "threshold_version": FUR_SURFACE_THRESHOLD_VERSION,
            "actions": fur_surface_quality_by_action,
            "pre_scail_revalidated": True,
            "final_video_frame_audit": False,
        },
        reference_generation_status="completed",
        reference_generation_completed=True,
        effective_subject_config=effective_subject_config,
        stage="subject_references_ready",
        progress=35,
    )
    return action_references


def prepare_uploaded_subject_scail_references(
    job_id: str,
    actions: list[str],
) -> dict[str, str]:
    """Reject the retired upload-to-SCAIL route."""
    _sync_local_video_runtime(
        globals(),
        {"prepare_uploaded_subject_scail_references"},
    )

    raise RuntimeError(
        "Direct uploaded-image SCAIL references are retired. Submit the "
        "upload again so Qwen can generate the full-body canonical and action "
        "keyframes first."
    )

    # Historical implementation retained below only to keep old serialized
    # jobs diagnosable during the migration window.  It is unreachable and its
    # submission digests remain immutable even under offline forensic use.
    raise_if_cancelled(job_id)
    with JOBS_LOCK:
        job = dict(JOBS[job_id])
    job_dir = OUTPUT_ROOT / job_id
    input_path = job_dir / "input.png"
    identity_source_path = job_dir / "input_identity_source.png"
    expected_input_sha256 = (
        str(job.get("uploaded_identity_input_sha256") or "").strip().lower()
    )
    expected_source_sha256 = (
        str(job.get("uploaded_identity_source_sha256") or "").strip().lower()
    )
    if len(expected_input_sha256) != 64:
        raise RuntimeError("Uploaded identity input lacks its submission-time digest.")
    if len(expected_source_sha256) != 64:
        raise RuntimeError("Uploaded identity source lacks its submission-time digest.")
    if not input_path.is_file() or file_sha256(input_path) != expected_input_sha256:
        raise RuntimeError(
            "Uploaded identity input is missing or changed after submission."
        )
    if (
        not identity_source_path.is_file()
        or file_sha256(identity_source_path) != expected_source_sha256
    ):
        raise RuntimeError(
            "Uploaded identity source is missing or changed after submission."
        )
    effective_subject_config = job.get("effective_subject_config")
    if not isinstance(effective_subject_config, dict):
        raise RuntimeError(
            "Uploaded identity reference stage lacks its frozen subject config."
        )
    if effective_subject_config.get("profile_id") != "sleek_tabby_cat_v1":
        raise RuntimeError(
            "Deterministic uploaded identity references require the sleek-tabby "
            "profile."
        )
    eating_driver = Path(str(SCAIL_ACTION_CONFIG["eating"]["driving_template"]))
    if "eating" in actions:
        eating_action_config = effective_subject_config["actions"]["eating"]
        eating_driver = Path(eating_action_config["driving_animation"])
        expected_eating_sha = str(
            eating_action_config.get("driving_animation_sha256") or ""
        ).strip()
        if (
            not eating_driver.is_file()
            or not expected_eating_sha
            or file_sha256(eating_driver) != expected_eating_sha
        ):
            raise RuntimeError(
                "Frozen Main_case eating driver is missing or its digest changed."
            )
    reference_dir = job_dir / "uploaded_identity_scail_references"
    update_job(
        job_id,
        stage="preparing_uploaded_identity_references",
        progress=4,
        reference_generation_status="running",
    )
    references, manifest = build_uploaded_identity_scail_references(
        input_path,
        actions=actions,
        eating_driver_path=eating_driver,
        output_dir=reference_dir,
        rembg_session=get_rembg_session(),
    )
    missing_actions = [action for action in actions if action not in references]
    unexpected_actions = [action for action in references if action not in actions]
    if missing_actions:
        raise RuntimeError(
            "Deterministic uploaded identity references are missing actions: "
            + ", ".join(missing_actions)
        )
    if unexpected_actions:
        raise RuntimeError(
            "Deterministic uploaded identity references contain unexpected actions: "
            + ", ".join(unexpected_actions)
        )
    identity_manifest = manifest.get("identity")
    action_manifest = manifest.get("actions")
    if (
        manifest.get("profile_or_default_cat_pixels_used") is not False
        or not isinstance(identity_manifest, dict)
        or identity_manifest.get("profile_or_default_cat_pixels_used") is not False
        or identity_manifest.get("identity_rgb_provenance")
        != "normalized_upload_only_rembg_alpha_only"
        or identity_manifest.get("upload_sha256") != expected_input_sha256
        or not isinstance(action_manifest, dict)
    ):
        raise RuntimeError(
            "Uploaded identity reference provenance is incomplete or unsafe."
        )
    effective_subject_config = json.loads(json.dumps(effective_subject_config))
    reference_sha256s: dict[str, str] = {}
    for action, reference_text in references.items():
        reference_path = Path(reference_text)
        if reference_path.resolve().parent != reference_dir.resolve():
            raise RuntimeError(
                f"Uploaded {action} reference escaped its frozen job directory."
            )
        with Image.open(reference_path) as loaded:
            if "A" not in loaded.getbands():
                raise RuntimeError(
                    f"Uploaded {action} SCAIL reference has no alpha channel."
                )
            alpha_minimum, alpha_maximum = (
                loaded.convert("RGBA").getchannel("A").getextrema()
            )
            if alpha_minimum == 255 or alpha_maximum == 0:
                raise RuntimeError(
                    f"Uploaded {action} SCAIL reference has no usable alpha."
                )
        reference_digest = file_sha256(reference_path)
        action_provenance = action_manifest.get(action)
        if (
            not isinstance(action_provenance, dict)
            or action_provenance.get("profile_or_default_cat_pixels_used") is not False
            or action_provenance.get("reference_sha256") != reference_digest
        ):
            raise RuntimeError(
                f"Uploaded {action} reference provenance failed validation."
            )
        reference_sha256s[action] = reference_digest
        action_config = effective_subject_config["actions"][action]
        action_config.setdefault("profile_reference", action_config.get("reference"))
        action_config.setdefault(
            "profile_reference_sha256", action_config.get("reference_sha256")
        )
        action_config["reference"] = str(reference_path)
        action_config["reference_sha256"] = reference_digest
        action_config["reference_kind"] = "uploaded_identity_direct_scail"
        public_reference = job_dir / f"{action}_uploaded_identity_reference_input.png"
        shutil.copy2(reference_path, public_reference)
        append_output(
            job_id,
            {
                "type": "uploaded_identity_reference_input",
                "action": action,
                "action_label": SUBJECT_REPLACEMENT_ACTION_LABELS[action],
                "filename": public_reference.name,
                "sha256": reference_digest,
                "role": (
                    "uploaded_identity_plus_main_case_bowl_prop"
                    if action == "eating"
                    else "uploaded_identity_only"
                ),
                "transparent": True,
                "primary": False,
            },
        )
    identity_cutout = reference_dir / "uploaded_identity_cutout.png"
    if not identity_cutout.is_file() or file_sha256(
        identity_cutout
    ) != identity_manifest.get("cutout_sha256"):
        raise RuntimeError("Uploaded identity cutout provenance failed validation.")
    public_identity_cutout = job_dir / "uploaded_identity_cutout.png"
    shutil.copy2(identity_cutout, public_identity_cutout)
    append_output(
        job_id,
        {
            "type": "uploaded_identity_cutout",
            "filename": public_identity_cutout.name,
            "sha256": file_sha256(public_identity_cutout),
            "role": "identity_only_rgb_from_upload_rembg_alpha_only",
            "transparent": True,
            "primary": False,
        },
    )
    manifest_path = Path(str(manifest["manifest_path"]))
    if (
        not manifest_path.is_file()
        or manifest_path.resolve().parent != reference_dir.resolve()
        or file_sha256(manifest_path) != manifest.get("manifest_sha256")
    ):
        raise RuntimeError("Uploaded identity reference manifest digest changed.")
    update_job(
        job_id,
        subject_source="uploaded_identity_direct_scail",
        effective_subject_source="uploaded_identity_direct_scail",
        subject_reference_paths=dict(references),
        qwen_subject_reference_paths={},
        reference_pipeline=UPLOADED_IDENTITY_SCAIL_REFERENCE_POLICY,
        reference_pipeline_version=UPLOADED_IDENTITY_SCAIL_REFERENCE_VERSION,
        reference_generation_status="completed",
        reference_generation_completed=True,
        # Submission digests are immutable queue-time evidence.  This retired
        # direct builder may only preserve them after the checks above; it must
        # never re-baseline changed files into trusted job state.
        uploaded_identity_input_sha256=expected_input_sha256,
        uploaded_identity_source_sha256=expected_source_sha256,
        uploaded_identity_cutout_sha256=file_sha256(identity_cutout),
        uploaded_identity_reference_sha256s=reference_sha256s,
        uploaded_identity_reference_manifest=str(manifest_path),
        uploaded_identity_reference_manifest_sha256=file_sha256(manifest_path),
        uploaded_identity_reference_provenance=manifest,
        profile_or_default_cat_pixels_used=False,
        qwen_reference_actions=[],
        qwen_reference_generated_actions=[],
        canonical_reference_generated=False,
        canonical_reference_used=False,
        effective_subject_config=effective_subject_config,
        stage="subject_references_ready",
        progress=12,
    )
    return references
