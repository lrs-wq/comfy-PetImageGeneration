"""Local-video service health payload construction."""

from __future__ import annotations

# The extracted functions resolve the facade dependencies after runtime sync.
# ruff: noqa: F821

from .runtime import sync_local_video_runtime as _sync_local_video_runtime


def health() -> dict[str, Any]:
    _sync_local_video_runtime(globals(), {"health"})
    comfy_ok = False
    comfy_error: str | None = None
    comfy_queue = {"running": 0, "pending": 0}
    try:
        response = requests.get(f"{COMFY_URL}/queue", timeout=5)
        response.raise_for_status()
        payload = response.json()
        comfy_queue = {
            "running": len(payload.get("queue_running", [])),
            "pending": len(payload.get("queue_pending", [])),
        }
        comfy_ok = True
    except Exception as exc:  # noqa: BLE001 - health endpoint should report, not crash
        comfy_error = str(exc)
    missing_models = [
        path.name for path in required_model_paths() if not path.is_file()
    ]
    profile_model_paths = [
        MODEL_ROOT / "loras" / Path(item["model_name"])
        for profile in IMAGE_PROFILES.values()
        for item in profile["user_loras"]
    ]
    missing_profile_models = [
        path.name for path in profile_model_paths if not path.is_file()
    ]
    missing_models.extend(
        name for name in missing_profile_models if name not in missing_models
    )
    fun_control_paths = [
        FUN_CONTROL_PYTHON,
        FUN_CONTROL_SCRIPT,
        FUN_CONTROL_PROJECT_ROOT / "videox_fun" / "__init__.py",
        FUN_CONTROL_MODEL_ROOT / "diffusion_pytorch_model.safetensors",
        FUN_CONTROL_MODEL_ROOT / "models_t5_umt5-xxl-enc-bf16.pth",
        FUN_CONTROL_MODEL_ROOT / "Wan2.2_VAE.pth",
        FUN_CONTROL_RUNNING_VIDEO,
    ]
    fun_control_missing = [
        str(path) for path in fun_control_paths if not path.is_file()
    ]
    ffmpeg_ok = Path(FFMPEG_BIN).is_file()
    translator_missing = prompt_translator_missing_paths()
    translator_ok = not PROMPT_TRANSLATION_ENABLED or not translator_missing
    subject_profile_items: list[dict[str, Any]] = []
    subject_replacement_missing: list[str] = []
    species_route_missing: list[str] = []
    dog_profile_health: dict[str, Any] = {"status": "unavailable"}
    classifier_paths = {
        "python": Path(
            os.environ.get(
                "PET_SPECIES_CLASSIFIER_PYTHON",
                "/opt/miniconda3/envs/yolo/bin/python",
            )
        ),
        "model": Path(
            os.environ.get("PET_SPECIES_CLASSIFIER_MODEL", "/root/yolo26n.pt")
        ),
        "runner": (
            PROJECT_ROOT / "tools" / "classification" / "classify_pet_species.py"
        ),
    }
    classifier_missing = [
        str(path) for path in classifier_paths.values() if not path.is_file()
    ]
    species_classifier_health = {
        "status": "ready" if not classifier_missing else "fallback_cat",
        "policy": "yolo_coco_cat_dog_forced_argmax_v1",
        "forced_binary": True,
        "unknown_class": False,
        "fallback_species": "cat",
        "missing": classifier_missing,
    }
    if classifier_missing:
        species_route_missing.extend(
            f"species classifier asset is missing: {path}"
            for path in classifier_missing
        )
    default_effective_subject_config: dict[str, Any] | None = None
    standard_direct_health: dict[str, Any] = {
        "enabled": False,
        "status": "unavailable",
        "policy": PROFILE_STANDARD_DIRECT_POLICY,
        "gpu_required": False,
        "gpu_submitted": False,
        "gpu_bypass_reason": PROFILE_STANDARD_DIRECT_GPU_REASON,
        "upload_identity_excluded": True,
        "actions": {},
    }
    upload_validation_health: dict[str, Any] = {
        "enabled": True,
        "status": "unavailable",
        "policy": QWEN_FULL_BODY_FIRST_FRAME_PIPELINE,
        "version": QWEN_FULL_BODY_FIRST_FRAME_PIPELINE_VERSION,
        "generation_audit_policy": GENERATION_AUDIT_POLICY,
        "visual_content_audit_mode": (
            "blocking" if GENERATION_VISUAL_AUDITS_BLOCKING else "advisory"
        ),
        "runtime_integrity_checks_required": True,
        "eating_reference_prop_split": {
            "policy": SCAIL_EATING_REFERENCE_PROP_SPLIT_POLICY,
            "strict_visual_split_first": True,
            "visual_uncertainty_blocks_job": False,
            "terminal_fallback": "lower_center_prop_geometry_v1",
            "fallback_is_audited": True,
            "empty_or_tiny_subject_remains_runtime_error": True,
        },
        "requires_standard_direct_assets": True,
        "qwen_full_body_required": True,
        "qwen_action_keyframes_required": True,
        "normalized_transparent_references_required": True,
        "identity_upload_view_counts": [1, 3],
        "three_view_roles": list(UPLOADED_IDENTITY_VIEW_ROLES),
        "three_view_policy": UPLOADED_IDENTITY_VIEW_POLICY,
        "three_view_qwen_node_inputs": ["image1", "image2", "image3"],
        "three_view_canonical_supported": True,
        "three_view_action_stage_raw_views_used": False,
        "uploaded_identity_appearance_authority": True,
        "subject_profile_appearance_role": "action_geometry_only",
        "canonical_palette_input": "final_alpha_cutout",
        "canonical_upper_body_palette_reference_policy": (
            CANONICAL_UPPER_BODY_PALETTE_REFERENCE_POLICY
        ),
        "canonical_raw_palette_retained_for_audit": True,
        "canonical_identity_palette_gate": {
            "enabled": True,
            "fail_closed_for_generated_candidates": True,
            "retry_all_canonical_profiles": True,
            "terminal_recovery": "original_uploaded_identity_safe_matte",
            "terminal_recovery_must_pass_same_palette_gate": True,
        },
        "canonical_chroma_despill": (
            "alpha_boundary_plus_source_palette_guarded_interior_v1"
        ),
        "canonical_chroma_spill_policy": (UPLOADED_IDENTITY_CHROMA_SPILL_POLICY),
        "canonical_chroma_spill_preserves_paired_eye_accents": True,
        "canonical_interior_hole_gate": GENERATION_VISUAL_AUDITS_BLOCKING,
        "surface_texture_gate_required_before_scail": (
            GENERATION_VISUAL_AUDITS_BLOCKING
        ),
        "surface_candidate_comparison_mode": (
            UPLOADED_IDENTITY_SURFACE_COMPARISON_MODE
        ),
        "expected_surface_gate_actions": (
            list(SUBJECT_REPLACEMENT_ACTIONS)
            if GENERATION_VISUAL_AUDITS_BLOCKING
            else []
        ),
    }
    fur_surface_gate_health: dict[str, Any] = {
        "enabled": GENERATION_VISUAL_AUDITS_BLOCKING,
        "policy": FUR_SURFACE_GATE_POLICY,
        "gate_version": FUR_SURFACE_GATE_VERSION,
        "threshold_version": FUR_SURFACE_THRESHOLD_VERSION,
        "thresholds": fur_surface_thresholds_payload(),
        "radial_long_line_evidence_policy": (
            "hough_radial_fraction_and_radial_line_density_v1"
        ),
        "radial_ratio_alone_can_reject": False,
        "scope": "qwen_action_references_and_all_final_scail_videos",
        "qwen_reference_candidate_gate": GENERATION_VISUAL_AUDITS_BLOCKING,
        "direct_upload_reference_candidate_gate": False,
        "direct_upload_reference_reason": (
            "the raw upload never enters SCAIL; Qwen action references and "
            "final SCAIL video frames remain fail-closed"
        ),
        "reliable_inputs": ["alpha_cutout", "rembg_alpha_cutout"],
        "fail_closed": GENERATION_VISUAL_AUDITS_BLOCKING,
        "max_consecutive_rejections": FUR_SURFACE_GATE_MAX_REJECTIONS,
        "retry_policy": FUR_SURFACE_GATE_RETRY_POLICY,
        "preserves_configured_eating_bowl_recovery": True,
        "configured_eating_bowl_recovery_attempts": (QWEN_BOWL_POSE_EXTRA_ATTEMPTS),
        "configured_eating_bowl_recovery_profile_prefix": (
            f"{QWEN_BOWL_POSE_RETRY_NAME}_"
        ),
        "blocks_scail_on_exhaustion": GENERATION_VISUAL_AUDITS_BLOCKING,
        "final_video_frame_audit": GENERATION_VISUAL_AUDITS_BLOCKING,
        "final_video_policy": FINAL_VIDEO_FUR_AUDIT_POLICY,
        "final_video_version": FINAL_VIDEO_FUR_AUDIT_VERSION,
        "final_video_sample_count": FINAL_VIDEO_FUR_SAMPLE_COUNT,
        "final_video_profile_full_scope": FINAL_VIDEO_PROFILE_FULL_SCOPE,
        "final_video_uploaded_qwen_action_scope": (
            FINAL_VIDEO_VALIDATED_ACTION_SURFACE_SCOPE
        ),
        "final_video_uploaded_qwen_actions": sorted(
            FINAL_VIDEO_VALIDATED_ACTION_REFERENCE_POLICIES
        ),
        "final_video_uploaded_qwen_reference_policies": dict(
            FINAL_VIDEO_VALIDATED_ACTION_REFERENCE_POLICIES
        ),
        "final_video_uploaded_eating_scope": (
            FINAL_VIDEO_VALIDATED_ACTION_SURFACE_SCOPE
        ),
        "final_video_uploaded_grooming_scope": (
            FINAL_VIDEO_VALIDATED_ACTION_SURFACE_SCOPE
        ),
        "final_video_profile_and_qwen_reference_sha_required": True,
        "final_video_profile_pixels_used_for_uploaded_identity_appearance_validation": False,
        "final_video_uploaded_qwen_action_reference_is_appearance_authority": True,
        "final_video_scail_identity_mask_sha_required": True,
        "final_video_eating_prop_alignment_minimum_overlap": (
            FINAL_VIDEO_EATING_PROP_ALIGNMENT_MIN
        ),
        "final_video_non_radial_absolute_observations_nonblocking_only_for_uploaded_qwen_transparent_actions": True,
        "final_video_geometry_relative_nonblocking_only_for_uploaded_qwen_transparent_actions": True,
        "final_video_unmasked_contour_relative_required": True,
        "final_video_prop_excluded_texture_relative_required": True,
        "final_video_radial_fraction_requires_threshold_backed_load_expansion_or_energy_backstop": True,
        "final_video_radial_load_proxy": (
            "fine_radial_brush_fraction_x_fine_gradient_p90"
        ),
        "final_video_radial_energy_backstop_reuses_existing_thresholds": True,
        "final_video_absolute_radial_issue_remains_blocking": True,
        "final_video_eating_initial_radial_loop_endpoint_corroboration": True,
        "final_video_eating_initial_radial_requires_single_initial_blocking_sample": True,
        "final_video_eating_initial_radial_requires_clean_terminal_and_all_interior_samples": True,
        "final_video_eating_initial_radial_requires_hash_bound_radial_long_lines_reference_observation": True,
        "final_video_eating_initial_radial_boundary_minimum_samples": (
            FINAL_VIDEO_EATING_RADIAL_BOUNDARY_MIN_SAMPLES
        ),
        "final_video_eating_initial_radial_boundary_load_energy_spread_ratio_max": (
            FINAL_VIDEO_EATING_RADIAL_BOUNDARY_LOAD_ENERGY_SPREAD_MAX
        ),
        "final_video_eating_initial_radial_boundary_component_spread_ratio_max": (
            FINAL_VIDEO_EATING_RADIAL_BOUNDARY_COMPONENT_SPREAD_MAX
        ),
        "final_video_eating_initial_radial_preserves_absolute_radial_fail_closed": True,
        "final_video_eating_initial_radial_bounded_reference_corroboration": True,
        "final_video_eating_initial_radial_bounded_reference_ratio_margin": (
            FINAL_VIDEO_EATING_RADIAL_BOUNDED_REFERENCE_RATIO_MARGIN
        ),
        "final_video_eating_initial_radial_bounded_reference_load_delta_multiplier": (
            FINAL_VIDEO_EATING_RADIAL_BOUNDED_REFERENCE_LOAD_DELTA_MULTIPLIER
        ),
        "final_video_eating_initial_radial_bounded_reference_preserves_absolute_radial_fail_closed": True,
        "final_video_coherent_brush_requires_gradient_weighted_load_or_long_line_energy_backstop": True,
        "final_video_coherent_brush_load_proxy": (
            "structure_tensor_coherent_fraction_x_coarse_gradient_p90"
        ),
        "final_video_coherent_brush_load_only_temporal_min_samples": (
            FINAL_VIDEO_COHERENT_BRUSH_TEMPORAL_MIN_SAMPLES
        ),
        "final_video_coherent_brush_load_only_single_sample_is_observation": True,
        "final_video_coherent_brush_long_line_energy_backstop_remains_single_sample_blocking": True,
        "final_video_absolute_coherent_brush_issue_alone_is_observation": True,
        "final_video_long_fur_requires_contour_corroboration_or_strong_expansion": True,
        "final_video_long_fur_texture_only_absolute_and_dual_surface_evidence_is_observation": True,
        "final_video_eating_isolated_strong_long_fur_temporal_corroboration": True,
        "final_video_eating_isolated_long_fur_requires_single_interior_blocking_sample": True,
        "final_video_eating_isolated_long_fur_records_blocking_frame_unmasked_corroboration": True,
        "final_video_eating_isolated_long_fur_requires_adjacent_prop_excluded_observation": True,
        "final_video_eating_isolated_long_fur_allows_one_initial_radial_companion_for_later_validation": True,
        "final_video_eating_isolated_long_fur_minimum_samples": (
            FINAL_VIDEO_EATING_ISOLATED_LONG_FUR_MIN_SAMPLES
        ),
        "final_video_eating_isolated_long_fur_reuses_original_neighbor_ratio_and_delta_thresholds": True,
        "final_video_eating_isolated_long_fur_strong_ceiling_uses_one_existing_texture_delta_margin": True,
        "final_video_eating_isolated_long_fur_preserves_absolute_contour_radial_coherent_and_unknown_fail_closed": True,
        "final_video_eating_loop_boundary_decode_anchor_corroboration": True,
        "final_video_eating_boundary_anchor_requires_clean_first_and_last_sample": True,
        "final_video_eating_boundary_anchor_requires_contiguous_interior_samples": True,
        "final_video_eating_boundary_anchor_minimum_samples": (
            FINAL_VIDEO_EATING_BOUNDARY_ANCHOR_MIN_SAMPLES
        ),
        "final_video_eating_boundary_anchor_minimum_blocking_samples": (
            FINAL_VIDEO_EATING_BOUNDARY_ANCHOR_MIN_BLOCKING_SAMPLES
        ),
        "final_video_eating_boundary_anchor_spread_ratio_max": (
            FINAL_VIDEO_EATING_BOUNDARY_ANCHOR_SPREAD_MAX
        ),
        "final_video_eating_boundary_anchor_preserves_original_ratio_delta_and_absolute_limits": True,
        "final_video_grooming_masked_pet3d_low_signal_reference_texture_floor": True,
        "final_video_grooming_masked_pet3d_sparse_coarse_low_signal_reference_texture_floor": True,
        "final_video_sparse_coarse_low_signal_minimum_samples": (
            FINAL_VIDEO_SPARSE_COARSE_LOW_SIGNAL_MIN_SAMPLES
        ),
        "final_video_sparse_coarse_low_signal_minimum_blocking_samples": (
            FINAL_VIDEO_SPARSE_COARSE_LOW_SIGNAL_MIN_BLOCKING_SAMPLES
        ),
        "final_video_sparse_coarse_low_signal_maximum_blocking_fraction": (
            FINAL_VIDEO_SPARSE_COARSE_LOW_SIGNAL_MAX_BLOCKING_FRACTION
        ),
        "final_video_sparse_coarse_low_signal_requires_all_sample_long_line_observation": True,
        "final_video_sparse_coarse_low_signal_coarse_ceiling_uses_one_existing_texture_delta_margin": True,
        "final_video_sparse_coarse_low_signal_preserves_absolute_contour_radial_coherent_and_unknown_fail_closed": True,
        "final_video_grooming_direct_qwen_sparse_coarse_temporal_floor": True,
        "final_video_direct_qwen_sparse_coarse_minimum_samples": (
            FINAL_VIDEO_DIRECT_QWEN_SPARSE_COARSE_MIN_SAMPLES
        ),
        "final_video_direct_qwen_sparse_coarse_minimum_blocking_samples": (
            FINAL_VIDEO_DIRECT_QWEN_SPARSE_COARSE_MIN_BLOCKING_SAMPLES
        ),
        "final_video_direct_qwen_sparse_coarse_maximum_blocking_fraction": (
            FINAL_VIDEO_DIRECT_QWEN_SPARSE_COARSE_MAX_BLOCKING_FRACTION
        ),
        "final_video_direct_qwen_sparse_coarse_temporal_spread_ratio_max": (
            FINAL_VIDEO_DIRECT_QWEN_SPARSE_COARSE_TEMPORAL_SPREAD_MAX
        ),
        "final_video_direct_qwen_sparse_coarse_requires_non_low_signal_reference": True,
        "final_video_direct_qwen_sparse_coarse_requires_blocking_dual_surface_coarse_long_pair": True,
        "final_video_direct_qwen_sparse_coarse_coarse_ceiling_uses_reference_ratio_plus_one_texture_delta": True,
        "final_video_direct_qwen_sparse_coarse_long_ceiling_reuses_existing_strong_limits": True,
        "final_video_direct_qwen_sparse_coarse_preserves_absolute_contour_radial_coherent_and_unknown_fail_closed": True,
        "final_video_low_signal_reference_requires_verified_composite_recovery": True,
        "final_video_low_signal_reference_requires_all_samples": True,
        "final_video_low_signal_reference_requires_coarse_blocking_all_samples": True,
        "final_video_low_signal_reference_minimum_long_blocking_fraction": (
            FINAL_VIDEO_LOW_SIGNAL_REFERENCE_MIN_LONG_BLOCKING_FRACTION
        ),
        "final_video_low_signal_reference_allows_remaining_long_observations": True,
        "final_video_low_signal_reference_minimum_samples": (
            FINAL_VIDEO_LOW_SIGNAL_REFERENCE_MIN_SAMPLES
        ),
        "final_video_low_signal_reference_temporal_spread_ratio_max": (
            FINAL_VIDEO_LOW_SIGNAL_REFERENCE_TEMPORAL_SPREAD_MAX
        ),
        "final_video_low_signal_reference_reuses_coherence_gradient_p90_min": True,
        "final_video_low_signal_reference_preserves_absolute_contour_radial_and_unknown_fail_closed": True,
        "final_video_derived_texture_reference_hash_and_alpha_contract_required": True,
        "final_video_derived_texture_baseline_absolute_observations_nonblocking": True,
        "final_video_unmasked_action_reference_baseline_must_pass_current_absolute_contract": True,
        "final_video_malformed_evaluator_contract_fails_closed": True,
        "final_video_unknown_relative_issue_fails_closed": True,
        "final_video_post_animation_contour_gate_required_for_uploaded_eating_v2": True,
        "final_video_grooming_no_prop_identity_mask_required_for_v2": True,
        "status": "unavailable",
        "baselines": {},
    }
    try:
        default_subject_profile = load_subject_profile(DEFAULT_SUBJECT_PROFILE)
        default_effective_subject_config = resolve_effective_subject_config(
            default_subject_profile,
            list(SUBJECT_REPLACEMENT_ACTIONS),
            seed=None,
            scail_steps=None,
            pose_strength=None,
        )
        raw_standard_direct = default_effective_subject_config.get("standard_direct")
        if isinstance(raw_standard_direct, dict):
            try:
                verify_profile_standard_direct_assets(
                    raw_standard_direct,
                    list(SUBJECT_REPLACEMENT_ACTIONS),
                )
                standard_direct_health = {
                    "enabled": True,
                    "status": "ready",
                    "policy": raw_standard_direct["policy"],
                    "role": raw_standard_direct["role"],
                    "revision": raw_standard_direct["revision"],
                    "deterministic": True,
                    "gpu_required": False,
                    "gpu_submitted": False,
                    "gpu_bypass_reason": raw_standard_direct["gpu_bypass_reason"],
                    "upload_identity_excluded": True,
                    "outputs": {
                        "mp4": {
                            "background": "white",
                            "frames": default_effective_subject_config["frames"],
                            "fps": default_effective_subject_config["fps"],
                            "matte_policy": (
                                "shared_per_frame_chroma_matte_no_spatial_transform_v1"
                            ),
                        },
                        "transparent_gif": {
                            "frames": TRANSPARENT_GIF_FRAMES,
                            "fps": TRANSPARENT_GIF_FPS,
                            "matte_policy": (
                                "shared_per_frame_chroma_matte_no_spatial_transform_v1"
                            ),
                        },
                    },
                    "actions": json.loads(json.dumps(raw_standard_direct["actions"])),
                }
            except Exception as exc:  # noqa: BLE001 - health must fail closed
                standard_direct_health.update(
                    {
                        "enabled": True,
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {str(exc)[:1000]}",
                    }
                )
                subject_replacement_missing.append(
                    f"profile standard-direct health failed: {exc}"
                )
        upload_health_config = default_effective_subject_config
        upload_pose_policies: dict[str, str] = {}
        if standard_direct_health.get("status") == "ready":
            upload_health_config, upload_pose_policies = (
                configure_uploaded_subject_validation(
                    default_effective_subject_config,
                    list(SUBJECT_REPLACEMENT_ACTIONS),
                )
            )
            require_subject_replacement_assets(
                default_subject_profile,
                list(SUBJECT_REPLACEMENT_ACTIONS),
                require_profile_references=False,
                effective_config=upload_health_config,
                expected_pose_policies=upload_pose_policies,
            )
            upload_validation_health.update(
                {
                    "status": "ready",
                    "expected_pose_policies": dict(upload_pose_policies),
                    "expected_surface_gate_actions": list(
                        upload_health_config["upload_validation"][
                            "expected_surface_gate_actions"
                        ]
                    ),
                }
            )
        else:
            upload_validation_health.update(
                {
                    "status": "failed",
                    "error": (
                        "The frozen Main_case standard-direct assets are not "
                        "ready, so uploaded Qwen jobs cannot freeze their "
                        "driver validation contract."
                    ),
                }
            )
            subject_replacement_missing.append(
                "uploaded Qwen validation requires ready Main_case standard assets"
            )
            require_subject_replacement_assets(
                default_subject_profile,
                list(SUBJECT_REPLACEMENT_ACTIONS),
                effective_config=default_effective_subject_config,
            )
        if GENERATION_VISUAL_AUDITS_BLOCKING:
            baseline_health: dict[str, Any] = {}
            for action in SUBJECT_REPLACEMENT_ACTIONS:
                action_config = upload_health_config["actions"][action]
                gate_config = action_config.get("fur_surface_gate")
                baseline_config = (
                    gate_config.get("baseline")
                    if isinstance(gate_config, dict)
                    else None
                )
                if not isinstance(baseline_config, dict):
                    baseline_health[action] = {
                        "status": "failed",
                        "issues": ["fur_surface_baseline_configuration_missing"],
                    }
                    subject_replacement_missing.append(
                        f"{action} fur surface baseline configuration is missing"
                    )
                    continue
                baseline_path = Path(str(baseline_config.get("source") or ""))
                try:
                    digest = file_sha256(baseline_path)
                    if digest != baseline_config.get("source_sha256"):
                        raise ValueError("baseline_digest_changed")
                    baseline_result = evaluate_fur_surface(
                        baseline_path,
                        baseline_path,
                        baseline_is_validated=True,
                    )
                    baseline_result = json.loads(
                        json.dumps(
                            baseline_result,
                            ensure_ascii=False,
                            allow_nan=False,
                        )
                    )
                    baseline_health[action] = {
                        "status": (
                            "passed"
                            if baseline_result.get("acceptable") is True
                            else "failed"
                        ),
                        "sha256": digest,
                        "provenance": baseline_config.get("provenance"),
                        "issues": baseline_result.get("issues", []),
                        "metrics": baseline_result["candidate"]["metrics"],
                    }
                    if baseline_result.get("acceptable") is not True:
                        subject_replacement_missing.append(
                            f"{action} fur surface baseline fails absolute contract"
                        )
                except Exception as exc:  # noqa: BLE001 - health reports exact failure
                    baseline_health[action] = {
                        "status": "failed",
                        "issues": ["fur_surface_baseline_health_failed"],
                        "error": f"{type(exc).__name__}: {str(exc)[:1000]}",
                    }
                    subject_replacement_missing.append(
                        f"{action} fur surface baseline health failed: {exc}"
                    )
            fur_surface_gate_health["baselines"] = baseline_health
            fur_surface_gate_health["status"] = (
                "passed"
                if baseline_health
                and all(
                    item.get("status") == "passed"
                    for item in baseline_health.values()
                )
                else "failed"
            )
        else:
            fur_surface_gate_health.update(
                {
                    "status": "skipped_advisory",
                    "baselines": {},
                    "blocking": False,
                    "reason": "visual_content_audits_are_non_blocking",
                }
            )
        subject_profile_items.append(subject_profile_public(default_subject_profile))
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        default_subject_profile = None
        if upload_validation_health.get("status") == "unavailable":
            upload_validation_health.update(
                {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {str(exc)[:1000]}",
                }
            )
        subject_replacement_missing.append(str(exc))
    try:
        dog_subject_profile = load_subject_profile(DOG_SUBJECT_PROFILE)
        dog_effective_config = resolve_effective_subject_config(
            dog_subject_profile,
            list(DOG_SUBJECT_REPLACEMENT_ACTIONS),
            seed=None,
            scail_steps=None,
            pose_strength=None,
        )
        dog_upload_config, dog_pose_policies = configure_uploaded_subject_validation(
            dog_effective_config,
            list(DOG_SUBJECT_REPLACEMENT_ACTIONS),
        )
        require_subject_replacement_assets(
            dog_subject_profile,
            list(DOG_SUBJECT_REPLACEMENT_ACTIONS),
            effective_config=dog_upload_config,
            expected_pose_policies=dog_pose_policies,
        )
        subject_profile_items.append(subject_profile_public(dog_subject_profile))
        dog_profile_health = {
            "status": "ready",
            "profile_id": DOG_SUBJECT_PROFILE,
            "species": "dog",
            "actions": list(DOG_SUBJECT_REPLACEMENT_ACTIONS),
            "legacy_alias": {"grooming": "standing"},
        }
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        dog_profile_health = {
            "status": "failed",
            "profile_id": DOG_SUBJECT_PROFILE,
            "error": f"{type(exc).__name__}: {str(exc)[:1000]}",
        }
        species_route_missing.append(f"dog profile health failed: {exc}")
    subject_replacement_ok = not subject_replacement_missing
    all_species_ready = subject_replacement_ok and not species_route_missing
    return {
        "status": (
            "ok"
            if (
                comfy_ok
                and not missing_models
                and API_KEY
                and ffmpeg_ok
                and translator_ok
                and all_species_ready
            )
            else "degraded"
        ),
        "version": API_VERSION,
        "generation_audit": {
            "policy": GENERATION_AUDIT_POLICY,
            "visual_content_mode": (
                "blocking" if GENERATION_VISUAL_AUDITS_BLOCKING else "advisory"
            ),
            "visual_content_blocking": GENERATION_VISUAL_AUDITS_BLOCKING,
            "runtime_integrity_checks_required": True,
            "required_checks": [
                "input_digest",
                "asset_presence",
                "image_decode",
                "normalized_512_rgba",
                "usable_alpha_subject",
                "reference_digest",
                "model_and_process_success",
                "video_and_gif_output_presence",
            ],
            "skipped_by_default": [
                "identity_view_similarity",
                "identity_palette_drift",
                "pose_similarity",
                "grooming_pose_detail",
                "fur_surface_texture",
                "fur_silhouette_and_contour",
                "final_video_frame_surface_audit",
            ],
        },
        "worker_alive": bool(WORKER_THREAD and WORKER_THREAD.is_alive()),
        "api_key_configured": bool(API_KEY),
        "job_queue": {"pending": JOB_QUEUE.qsize(), "capacity": MAX_QUEUE_SIZE},
        "comfyui": {"ok": comfy_ok, "queue": comfy_queue, "error": comfy_error},
        "models": {"ok": not missing_models, "missing": missing_models},
        "prompt_translation": {
            "enabled": PROMPT_TRANSLATION_ENABLED,
            "ok": translator_ok,
            "provider": "local_opus_mt_zh_en",
            "model": "Helsinki-NLP/opus-mt-zh-en",
            "source_language": "zh",
            "target_language": "en",
            "missing": translator_missing,
        },
        "image_profiles": {
            "default": DEFAULT_IMAGE_PROFILE or None,
            "items": [
                image_profile_public(profile) for profile in IMAGE_PROFILES.values()
            ],
        },
        "subject_replacement": {
            "default": True,
            "enabled": subject_replacement_ok,
            "ok": subject_replacement_ok,
            "all_species_ready": all_species_ready,
            "species_route_missing": species_route_missing,
            "species_classifier": species_classifier_health,
            "dog_profile": dog_profile_health,
            "dog_eating_first_frame": {
                "policy": DOG_EATING_PROMPT_ONLY_FIRST_FRAME_POLICY,
                "steps": DOG_EATING_PROMPT_ONLY_ACTION_TRANSFER_STEPS,
                "denoise": DOG_EATING_PROMPT_ONLY_ACTION_TRANSFER_DENOISE,
                "image1": "approved_upload_anchored_canonical_identity_sampler",
                "image2": "digest_bound_white_model_action_geometry_guide",
                "prompt_only_action_geometry": False,
                "white_model_loaded_by_qwen": True,
                "white_model_rgb_loaded_by_qwen": False,
                "secondary_reference_loaded_by_qwen": True,
                "negative_prompt_conditioning": False,
            },
            "dog_drinking_first_frame": {
                "policy": DOG_DRINKING_PROMPT_ONLY_FIRST_FRAME_POLICY,
                "steps": DOG_DRINKING_PROMPT_ONLY_ACTION_TRANSFER_STEPS,
                "denoise": DOG_DRINKING_PROMPT_ONLY_ACTION_TRANSFER_DENOISE,
                "image1": "approved_upload_anchored_canonical_identity_sampler",
                "image2": "digest_bound_white_model_action_geometry_guide",
                "prompt_only_action_geometry": False,
                "white_model_loaded_by_qwen": True,
                "white_model_rgb_loaded_by_qwen": False,
                "secondary_reference_loaded_by_qwen": True,
                "negative_prompt_conditioning": False,
                "action_geometry_deferred_to_scail": False,
            },
            "dog_toileting_first_frame": {
                "policy": DOG_TOILETING_PROMPT_ONLY_FIRST_FRAME_POLICY,
                "steps": DOG_TOILETING_PROMPT_ONLY_ACTION_TRANSFER_STEPS,
                "denoise": DOG_TOILETING_PROMPT_ONLY_ACTION_TRANSFER_DENOISE,
                "image1": "approved_upload_anchored_canonical_identity_sampler",
                "image2": "digest_bound_white_model_action_geometry_guide",
                "prompt_only_action_geometry": False,
                "white_model_loaded_by_qwen": True,
                "white_model_rgb_loaded_by_qwen": False,
                "secondary_reference_loaded_by_qwen": True,
                "negative_prompt_conditioning": False,
                "required_spatial_gate": TOILETING_TRAY_SPATIAL_POLICY,
                "tray_min_reference_aspect_ratio": (
                    TOILETING_TRAY_MIN_REFERENCE_ASPECT_RATIO
                ),
                "identity_safe_terminal_fallback": (
                    TOILETING_CANONICAL_TRAY_FALLBACK_POLICY
                ),
                "scail_temporal_identity_artifact_gate": {
                    "policy": (
                        TOILETING_SCAIL_TEMPORAL_IDENTITY_ARTIFACT_POLICY
                    ),
                    "scope": "internal_lateral_tail_corridor_all_frames",
                    "blocks_invented_dark_band_arrow_or_symbol": True,
                    "max_attempts": TOILETING_SCAIL_MAX_ATTEMPTS,
                    "retry_seed_step": TOILETING_SCAIL_RETRY_SEED_STEP,
                    "retry_preserves_reference_driver_and_prompt": True,
                    "required_before_transparent_extraction": True,
                    "fail_closed": True,
                },
                "transparent_guidance": {
                    "policy": (
                        TOILETING_TRANSPARENT_REFERENCE_GUIDANCE_POLICY
                    ),
                    "reference_alpha_bbox_aspect_ratio_min": (
                        TOILETING_TRANSPARENT_REFERENCE_MIN_ASPECT_RATIO
                    ),
                    "static_reference_subject_and_tray_mask": True,
                    "driver_pose_mask_used_for_scail_generation": True,
                    "reference_mask_used_only_for_transparent_extraction": True,
                    "adaptive_semantic_bounded_or_hybrid_extraction": True,
                    "bounded_canvas_selection_border_distance_p999_max": 8.0,
                    "hybrid_canvas_selection_border_distance_p999_max": 32.0,
                    "semantic_static_reference_subject_and_tray_gate": True,
                    "hybrid_alpha_union_preserves_semantic_dog_and_canvas_tray": True,
                    "conditional_second_edge_cleanup_pass": True,
                    "fail_closed": True,
                },
            },
            "dog_drinking_scail_pose": {
                "policy": DOG_DRINKING_REFERENCE_ALIGNED_POSE_POLICY,
                "pose_source": "approved_upload_anchored_drinking_reference",
                "motion_scope": "bounded_closed_loop_head_lapping_only",
                "driver_input_used": True,
                "driver_rgb_preserved": False,
                "driver_silhouette_preserved": False,
                "body_ears_paws_tail_locked_to_reference": True,
                "static_identity_safe_fallback_policy": (
                    DOG_DRINKING_STATIC_POSE_FALLBACK_POLICY
                ),
            },
            "dog_eating_scail_pose": {
                "policy": DOG_EATING_REFERENCE_ALIGNED_POSE_POLICY,
                "driver_role": "head_lift_timing_signal_only",
                "driver_rgb_preserved": False,
                "driver_silhouette_preserved": False,
                "reference_identity_geometry_preserved": True,
                "static_identity_safe_fallback_policy": (
                    DOG_EATING_STATIC_POSE_FALLBACK_POLICY
                ),
                "canonical_identity_full_driver_fallback_policy": (
                    DOG_EATING_CANONICAL_FULL_DRIVER_POSE_POLICY
                ),
                "canonical_identity_reference_prop_inference": False,
                "canonical_fallback_full_white_driver_body_and_bowl": True,
            },
            "species_routes": {
                "cat": {
                    "profile": CAT_SUBJECT_PROFILE,
                    "actions": list(CAT_SUBJECT_REPLACEMENT_ACTIONS),
                },
                "dog": {
                    "profile": DOG_SUBJECT_PROFILE,
                    "actions": list(DOG_SUBJECT_REPLACEMENT_ACTIONS),
                    "legacy_alias": {"grooming": "standing"},
                },
            },
            "engine": "scail2-14b-fp8",
            "engine_profile": (
                default_effective_subject_config["engine_profile"]
                if default_effective_subject_config
                else None
            ),
            "prompt_profile": (
                default_effective_subject_config["prompt_profile"]
                if default_effective_subject_config
                else None
            ),
            "seed_mode": (
                default_effective_subject_config["seed_mode"]
                if default_effective_subject_config
                else "base_plus_action_offset"
            ),
            "actions": [
                {
                    "id": action,
                    "label": SUBJECT_REPLACEMENT_ACTION_LABELS[action],
                    "driving_template": Path(
                        default_effective_subject_config["actions"][action][
                            "driving_animation"
                        ]
                        if default_effective_subject_config
                        else SCAIL_ACTION_CONFIG[action]["driving_template"]
                    ).name,
                    "uploaded_driving_template": (
                        Path(
                            default_effective_subject_config["actions"][action][
                                "uploaded_identity_driving"
                            ]["runtime"]
                        ).name
                        if (
                            default_effective_subject_config
                            and isinstance(
                                default_effective_subject_config["actions"][
                                    action
                                ].get("uploaded_identity_driving"),
                                dict,
                            )
                        )
                        else None
                    ),
                    "uploaded_driving_revision": (
                        default_effective_subject_config["actions"][action][
                            "uploaded_identity_driving"
                        ].get("revision")
                        if (
                            default_effective_subject_config
                            and isinstance(
                                default_effective_subject_config["actions"][
                                    action
                                ].get("uploaded_identity_driving"),
                                dict,
                            )
                        )
                        else None
                    ),
                    "pose_strength": (
                        default_effective_subject_config["actions"][action][
                            "pose_strength"
                        ]
                        if default_effective_subject_config
                        else SCAIL_ACTION_CONFIG[action]["pose_strength"]
                    ),
                    "seed_offset": (
                        default_effective_subject_config["actions"][action][
                            "seed_offset"
                        ]
                        if default_effective_subject_config
                        else SCAIL_ACTION_CONFIG[action]["seed_offset"]
                    ),
                    "effective_seed": (
                        default_effective_subject_config["actions"][action][
                            "effective_seed"
                        ]
                        if default_effective_subject_config
                        else None
                    ),
                }
                for action in SUBJECT_REPLACEMENT_ACTIONS
            ],
            "default_subject_profile": DEFAULT_SUBJECT_PROFILE,
            "subject_profiles": subject_profile_items,
            "source_frames": (
                default_effective_subject_config["frames"]
                if default_effective_subject_config
                else SCAIL_FRAMES
            ),
            "source_fps": (
                default_effective_subject_config["fps"]
                if default_effective_subject_config
                else SCAIL_FPS
            ),
            "steps": (
                default_effective_subject_config["steps"]
                if default_effective_subject_config
                else SCAIL_STEPS
            ),
            "effective_subject_config": default_effective_subject_config,
            "standard_direct": standard_direct_health,
            "upload_validation": upload_validation_health,
            "output": {
                "size": [TRANSPARENT_GIF_SIZE, TRANSPARENT_GIF_SIZE],
                "frames": TRANSPARENT_GIF_FRAMES,
                "fps": TRANSPARENT_GIF_FPS,
                "duration_seconds": TRANSPARENT_GIF_DURATION_SECONDS,
                "recoverable_fallback": {
                    "enabled": True,
                    "policy": RECOVERABLE_OUTPUT_FALLBACK_POLICY,
                    "scope": "accepted_subject_replacement_transparent_gif_jobs",
                    "per_requested_action": True,
                    "triggers": [
                        "generation_exception",
                        "service_restart_during_generation",
                        "restored_queue_capacity_exceeded",
                    ],
                    "cancelled_jobs_excluded": True,
                },
            },
            "missing": subject_replacement_missing,
        },
        "qwen_style": {
            "enabled": bool(QWEN_STYLE_LORA),
            "profile": QWEN_STYLE_PROFILE if QWEN_STYLE_LORA else "baseline",
            "lora": QWEN_STYLE_LORA or None,
            "strength": QWEN_STYLE_LORA_STRENGTH if QWEN_STYLE_LORA else None,
            "pose_strength": QWEN_POSE_STYLE_LORA_STRENGTH if QWEN_STYLE_LORA else None,
            "steps": {
                "default": IMAGE_PROFILES[DEFAULT_IMAGE_PROFILE]["qwen_steps"]
                if DEFAULT_IMAGE_PROFILE
                else 4,
                "minimum": 1,
                "maximum": 30,
            },
            "max_user_loras": MAX_USER_LORAS,
            "max_user_lora_bytes_each": MAX_LORA_UPLOAD_BYTES,
            "max_user_lora_fusion_bytes": MAX_LORA_FUSION_BYTES,
            "stage_lora_caps": {
                "canonical_total": QWEN_CANONICAL_USER_LORA_TOTAL_CAP,
                "canonical_per_lora": QWEN_CANONICAL_USER_LORA_PER_LORA_CAP,
                "pose_total": QWEN_POSE_USER_LORA_TOTAL_CAP,
                "pose_retry_total": QWEN_POSE_USER_LORA_RETRY_TOTAL_CAP,
                "pose_final_total": QWEN_POSE_USER_LORA_FINAL_TOTAL_CAP,
                "pose_per_lora": QWEN_POSE_USER_LORA_PER_LORA_CAP,
                "subject_replacement_canonical_total": (
                    QWEN_SUBJECT_REPLACEMENT_CANONICAL_LORA_TOTAL_CAP
                ),
                "subject_replacement_pose_first_total": (
                    QWEN_SUBJECT_REPLACEMENT_POSE_LORA_TOTAL_CAP
                ),
            },
            "pose_guard": {
                "prompt_isolation": True,
                "reference_pipeline": QWEN_FULL_BODY_FIRST_FRAME_PIPELINE,
                "reference_pipeline_version": (
                    QWEN_FULL_BODY_FIRST_FRAME_PIPELINE_VERSION
                ),
                "generation_audit_policy": GENERATION_AUDIT_POLICY,
                "visual_content_audit_mode": (
                    "blocking"
                    if GENERATION_VISUAL_AUDITS_BLOCKING
                    else "advisory"
                ),
                "runtime_integrity_checks_required": True,
                "main_case_raw_identity_chain": False,
                "qwen_reference_generation_for_subject_replacement": True,
                "direct_scail_uploaded_identity_reference": False,
                "raw_upload_sent_to_scail": False,
                "identity_upload_view_counts": [1, 3],
                "identity_upload_three_view_roles": list(UPLOADED_IDENTITY_VIEW_ROLES),
                "identity_upload_three_view_policy": (UPLOADED_IDENTITY_VIEW_POLICY),
                "qwen_canonical_multi_view_supported": True,
                "qwen_canonical_multi_view_node_inputs": [
                    "image1",
                    "image2",
                    "image3",
                ],
                "qwen_canonical_image1_is_sampler_latent": True,
                "qwen_action_raw_identity_views_used": (
                    not GENERATION_VISUAL_AUDITS_BLOCKING
                ),
                "qwen_action_input_contract": {
                    "eating": "canonical_image1_plus_geometry_image2",
                    "grooming": (
                        "canonical_image1_plus_geometry_image2"
                        if GENERATION_VISUAL_AUDITS_BLOCKING
                        else (
                            "canonical_image1_plus_validated_action_image2_plus_"
                            "uploaded_front_image3"
                        )
                    ),
                },
                "canonical_action_identity_palette_gate": {
                    "enabled": bool(
                        GENERATION_VISUAL_AUDITS_BLOCKING
                        or GENERATION_ACTION_CANDIDATE_GATES_ENABLED
                    ),
                    "policy": CANONICAL_ACTION_IDENTITY_PALETTE_POLICY,
                    "achromatic_balance_l1_max": (
                        CANONICAL_ACTION_ACHROMATIC_BALANCE_L1_MAX
                    ),
                    "light_identity_black_fraction_max": (
                        CANONICAL_ACTION_LIGHT_IDENTITY_BLACK_FRACTION_MAX
                    ),
                    "dark_feature_expansion_max": (
                        CANONICAL_ACTION_DARK_FEATURE_EXPANSION_MAX
                    ),
                    "dark_feature_expansion_scope": (
                        "achromatic_canonical_with_low_source_black_fraction"
                    ),
                    "scope": "eating_and_grooming_qwen_action_candidates",
                    "fail_closed": bool(
                        GENERATION_VISUAL_AUDITS_BLOCKING
                        or GENERATION_ACTION_CANDIDATE_GATES_ENABLED
                    ),
                    "job_fails_on_exhaustion": (
                        GENERATION_VISUAL_AUDITS_BLOCKING
                    ),
                },
                "canonical_action_candidate_gate": {
                    "enabled": GENERATION_ACTION_CANDIDATE_GATES_ENABLED,
                    "policy": GENERATION_ACTION_CANDIDATE_GATE_POLICY,
                    "scope": "uploaded_canonical_non_grooming_action_candidates",
                    "full_retry_budget": True,
                    "rejects_identity_palette_quality_similarity_and_silhouette_drift": True,
                    "job_fails_on_exhaustion": GENERATION_VISUAL_AUDITS_BLOCKING,
                    "advisory_terminal_recovery": (
                        "identity_first_lowest_risk_generated_candidate"
                    ),
                    "returns_original_upload": False,
                },
                "uploaded_identity_appearance_authority": True,
                "subject_profile_appearance_role": "action_geometry_only",
                "canonical_palette_input": "final_alpha_cutout",
                "canonical_upper_body_palette_reference_policy": (
                    CANONICAL_UPPER_BODY_PALETTE_REFERENCE_POLICY
                ),
                "canonical_raw_palette_retained_for_audit": True,
                "canonical_identity_palette_gate": {
                    "enabled": True,
                    "policy": IDENTITY_PALETTE_POLICY,
                    "fail_closed_for_generated_candidates": True,
                    "retry_all_canonical_profiles": True,
                    "terminal_recovery": "original_uploaded_identity_safe_matte",
                    "terminal_recovery_must_pass_same_palette_gate": True,
                    "bidirectional_drift": {
                        "policy": IDENTITY_PALETTE_BIDIRECTIONAL_DRIFT_POLICY,
                        "warm_coat_source_min": IDENTITY_WARM_COAT_SOURCE_MIN,
                        "warm_coat_loss_delta_min": (
                            IDENTITY_WARM_COAT_LOSS_DELTA_MIN
                        ),
                        "achromatic_expansion_delta_min": (
                            IDENTITY_ACHROMATIC_EXPANSION_DELTA_MIN
                        ),
                        "gray_or_dark_source_max": (
                            IDENTITY_GRAY_OR_DARK_SOURCE_MAX
                        ),
                        "gray_or_dark_expansion_delta_min": (
                            IDENTITY_GRAY_OR_DARK_EXPANSION_DELTA_MIN
                        ),
                    },
                    "spatial_marking_gate": {
                        "policy": IDENTITY_MARKING_SPATIAL_POLICY,
                        "source_stripe_area_max": (
                            IDENTITY_NEW_STRIPE_SOURCE_AREA_MAX
                        ),
                        "stripe_component_delta_min": (
                            IDENTITY_NEW_STRIPE_COMPONENT_DELTA_MIN
                        ),
                        "stripe_area_delta_min": (
                            IDENTITY_NEW_STRIPE_AREA_DELTA_MIN
                        ),
                        "dark_contrast_delta_min": (
                            IDENTITY_NEW_STRIPE_DARK_CONTRAST_DELTA_MIN
                        ),
                    },
                },
                "canonical_chroma_despill": (
                    "alpha_boundary_plus_source_palette_guarded_interior_v1"
                ),
                "canonical_chroma_spill_policy": (
                    UPLOADED_IDENTITY_CHROMA_SPILL_POLICY
                ),
                "canonical_chroma_spill_preserves_paired_eye_accents": True,
                "canonical_interior_hole_gate": True,
                "profile_pixels_used_as_identity_or_scail_reference": False,
                "profile_pixels_used_as_validation_baseline": False,
                "uploaded_identity_absolute_surface_gate_used": True,
                "surface_candidate_comparison_mode": (
                    UPLOADED_IDENTITY_SURFACE_COMPARISON_MODE
                ),
                "canonical_reference_generated": True,
                "canonical_reference_used": True,
                "canonical_reference_normalized_and_matted": True,
                "action_keyframes_normalized_and_matted": True,
                "upload_pose_photography_lora_policy": (
                    "disabled_after_canonical_identity_style_lock_v1"
                ),
                "upload_pose_builtin_style_lora_policy": (
                    "disabled_for_canonical_actions_v2"
                ),
                "upload_pose_lora_policy": (
                    "qwen_lightning_only_with_open_channel_identity_refine_v6"
                ),
                "upload_pose_candidate_normalization_policy": (
                    QWEN_ACTION_REFERENCE_NORMALIZATION_POLICY
                ),
                "upload_pose_candidate_matte_policy": (
                    QWEN_ACTION_CANDIDATE_MATTE_POLICY
                ),
                "eating_mouth_contact_gate": {
                    "policy": EATING_MOUTH_BOWL_CONTACT_POLICY,
                    "thin_warm_rim_fallback_policy": (
                        EATING_BOWL_SPLIT_FALLBACK_POLICY
                    ),
                    "fallback_preserves_width_prop_paw_and_contact_checks": True,
                    "gap_px_min": 0,
                    "gap_px_max": EATING_MUZZLE_BOWL_GAP_MAX_PX,
                    "component_width_bowl_ratio_max": (
                        EATING_MUZZLE_COMPONENT_WIDTH_BOWL_RATIO_MAX
                    ),
                    "component_height_bowl_ratio_max": (
                        EATING_MUZZLE_COMPONENT_HEIGHT_BOWL_RATIO_MAX
                    ),
                    "component_center_margin_bowl_ratio": (
                        EATING_MUZZLE_COMPONENT_CENTER_MARGIN_BOWL_RATIO
                    ),
                    "component_center_offset_bowl_ratio_max": (
                        EATING_MUZZLE_COMPONENT_CENTER_OFFSET_BOWL_RATIO_MAX
                    ),
                    "component_height_px_min": (
                        EATING_MUZZLE_COMPONENT_HEIGHT_MIN_PX
                    ),
                    "component_area_px_min": (
                        EATING_MUZZLE_COMPONENT_AREA_MIN_PX
                    ),
                    "component_audit_limit": 12,
                },
                "normalization_contract": {
                    "full_body": (
                        "prepare_landscape_keyframe_then_evaluate_chroma_cutout"
                    ),
                    "action_keyframes": (QWEN_ACTION_REFERENCE_NORMALIZATION_POLICY),
                    "action_candidate_matte": (QWEN_ACTION_CANDIDATE_MATTE_POLICY),
                    "selected_candidate_byte_copied_to_scail_reference": True,
                    "wan_green_derived_one_way_from_rgba": True,
                    "output_size": [512, 512],
                    "full_body_normalized_transparent_required": True,
                    "action_keyframes_normalized_transparent_required": True,
                },
                "scail_reference_kind": "qwen_generated_upload",
                "upload_state_transition": [
                    "upload_pending_qwen",
                    "qwen_generated_upload",
                ],
                "uploaded_identity_digest_gate": {
                    "normalized_input_sha256_required": True,
                    "original_identity_source_sha256_required": True,
                    "comfy_input_copy_sha256_verified": True,
                    "all_submitted_views_sha256_required": True,
                    "all_submitted_views_comfy_copy_sha256_verified": True,
                    "duplicate_three_view_rejection": False,
                    "duplicate_three_view_downgrade_to_single_view": True,
                    "near_duplicate_view_angle_rejection": False,
                    "near_duplicate_view_downgrade_to_single_view": True,
                    "gross_palette_conflict_rejection": (
                        GENERATION_VISUAL_AUDITS_BLOCKING
                    ),
                    "fail_closed": True,
                    "integrity_fail_closed": True,
                    "visual_content_fail_closed": (
                        GENERATION_VISUAL_AUDITS_BLOCKING
                    ),
                },
                "upload_pose_reference_source": (
                    "per_action_qwen_canonical_and_main_case_geometry_roles"
                ),
                "upload_pose_reference_policy": {
                    "eating": "main_case_eating_driver_geometry_image2_guide_v6",
                    "grooming": GROOMING_QWEN_POSE_REFERENCE_POLICY,
                },
                "geometry_generator_policy": {
                    "eating": "driving_eating_geometry_image2_v6",
                    "grooming": GROOMING_QWEN_GEOMETRY_GUIDE_POLICY,
                },
                "eating_geometry_isolation": {
                    "qwen_internal_luma_variation": True,
                    "qwen_internal_luma_variation_role": (
                        "compressed_low_frequency_action_anatomy_only"
                    ),
                    "qwen_appearance_bearing_internal_luma_variation": False,
                    "source_rgb_preserved": False,
                    "source_chroma_preserved": False,
                    "high_frequency_texture_preserved": False,
                },
                "grooming_geometry_isolation": {
                    "qwen_guide_style": GROOMING_QWEN_GUIDE_STYLE,
                    "qwen_foreground_rgb": list(GROOMING_QWEN_GUIDE_FOREGROUND_RGB),
                    "qwen_foreground_palette_rgb": [
                        list(GROOMING_QWEN_GUIDE_FOREGROUND_RGB),
                    ],
                    "qwen_internal_luma_variation": False,
                    "qwen_internal_luma_variation_role": "none",
                    "qwen_appearance_bearing_internal_luma_variation": False,
                    "qwen_synthetic_foreleg_semantic": False,
                    "qwen_synthetic_foreleg_semantic_method": None,
                    "qwen_structural_negative_space": True,
                    "qwen_structural_negative_space_method": (
                        GROOMING_QWEN_NEGATIVE_SPACE_METHOD
                    ),
                    "qwen_closed_foreleg_capsule_exposed": False,
                    "qwen_internal_green_separator_exposed": True,
                    "qwen_raised_foreleg_validation_role": (
                        "public_open_curve_geometry_plus_private_cv"
                    ),
                    "qwen_role": "image2_action_geometry_guide",
                    "canonical_role": "image1_identity_style_and_sampler",
                    "sampler_latent": "cartoon_reference.png",
                    "denoise_by_attempt": None,
                    "canonical_action_denoise_source": ("job_pose_reference_denoise"),
                    "qwen_foreleg_repair": True,
                    "qwen_foreleg_repair_retained_for_legacy_jobs_only": True,
                    "qwen_foreleg_repair_policy": (
                        GROOMING_CANONICAL_FORELEG_COMPLETION_POLICY
                    ),
                    "qwen_foreleg_repair_attempt": (
                        GROOMING_QWEN_FORELEG_REPAIR_ATTEMPT
                    ),
                    "qwen_foreleg_repair_denoise": (
                        GROOMING_QWEN_FORELEG_REPAIR_DENOISE
                    ),
                    "qwen_foreleg_repair_steps": (GROOMING_QWEN_FORELEG_REPAIR_STEPS),
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
                    "qwen_foreleg_repair_seed_lowpass_sigma_px": None,
                    "qwen_foreleg_repair_expanded_context_full_denoise_mask": (True),
                    "qwen_foreleg_repair_seed_identity_median_blend": None,
                    "qwen_foreleg_repair_identity_or_geometry_seed_inserted": False,
                    "qwen_foreleg_repair_noise_mask_expansion_px": (
                        GROOMING_QWEN_FORELEG_REPAIR_NOISE_MASK_EXPANSION_PX
                    ),
                    "qwen_foreleg_repair_noise_mask_feather_sigma_px": (
                        GROOMING_QWEN_FORELEG_REPAIR_NOISE_MASK_FEATHER_SIGMA_PX
                    ),
                    "qwen_foreleg_repair_noise_mask_node": ("SetLatentNoiseMask"),
                    "qwen_foreleg_repair_style_policy": (
                        "disabled_for_canonical_masked_completion"
                    ),
                    "qwen_foreleg_repair_style_lora": None,
                    "qwen_foreleg_repair_style_strength": 0.0,
                    "qwen_foreleg_repair_style_scope": "none",
                    "qwen_foreleg_repair_style_identity_authority": False,
                    "qwen_foreleg_repair_style_fail_closed": True,
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
                    "canonical_full_pose_recovery_attempt_budget": (
                        GROOMING_CANONICAL_FULL_POSE_REFINE_ATTEMPT
                    ),
                    "canonical_full_pose_recovery_steps": (
                        GROOMING_CANONICAL_FULL_POSE_RESCUE_STEPS
                    ),
                    "canonical_full_pose_rescue_denoise": (
                        GROOMING_CANONICAL_FULL_POSE_RESCUE_DENOISE
                    ),
                    "canonical_full_pose_identity_refine_denoise": (
                        GROOMING_CANONICAL_FULL_POSE_REFINE_DENOISE
                    ),
                    "canonical_full_pose_recovery_style_lora": None,
                    "canonical_full_pose_recovery_qwen_lightning_only": True,
                    "canonical_full_pose_recovery_global_thresholds_changed": False,
                    "canonical_full_pose_recovery_image_roles": {
                        "attempt4_image1": (
                            "canonical_palette_soft_foreleg_depth_action_"
                            "anatomy_sampler"
                        ),
                        "attempt4_image2": (
                            "approved_canonical_sole_identity_appearance_authority"
                        ),
                        "attempt4_image3": None,
                        "attempt5_image1": (
                            "attempt4_seed_bound_pose_and_surface_approved_"
                            "candidate_sampler_no_identity_authority"
                        ),
                        "attempt5_image2": (
                            "approved_canonical_sole_identity_appearance_authority"
                        ),
                        "attempt5_image3": (
                            "digest_bound_soft_foreleg_depth_action_corroboration"
                        ),
                    },
                    "canonical_full_pose_refine_source_policy": (
                        GROOMING_CANONICAL_POSE_SAFE_REFINE_SOURCE_POLICY
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
                    "canonical_pose_safe_refine_structural_edge_density_min": (
                        GROOMING_CANONICAL_POSE_SAFE_STRUCTURAL_EDGE_DENSITY_MIN
                    ),
                    "canonical_pose_safe_refine_allowed_identity_issues": sorted(
                        GROOMING_CANONICAL_POSE_SAFE_IDENTITY_ONLY_ISSUES
                    ),
                    "canonical_pose_transfer_rejected_candidate_used_as_sampler": (
                        True
                    ),
                    "canonical_pose_transfer_rejected_candidate_scope": (
                        "attempt4_identity_palette_only_after_seed_bound_matte_"
                        "pose_quality_and_independent_surface_pass"
                    ),
                    "canonical_full_pose_geometry_guide_policy": (
                        GROOMING_CANONICAL_FULL_POSE_GUIDE_POLICY
                    ),
                    "canonical_full_pose_geometry_luma_half_span": None,
                    "qwen_foreleg_repair_global_edge_density_min": (
                        GROOMING_QWEN_FORELEG_REPAIR_GLOBAL_EDGE_DENSITY_MIN
                    ),
                    "qwen_foreleg_repair_global_pose_threshold_changed": False,
                    "qwen_foreleg_repair_composite_scope": (
                        "none_for_production_canonical_masked_completion"
                    ),
                    "qwen_foreleg_repair_image_roles": {
                        "image1": ("identity_safe_candidate_unchanged_masked_sampler"),
                        "image2": "approved_canonical_identity_appearance_authority",
                        "image3": "single_tone_open_channel_grooming_geometry",
                    },
                    "qwen_foreleg_repair_synthetic_capsule_exposed": False,
                    "qwen_foreleg_repair_open_channel_anatomy_image3": True,
                    "qwen_foreleg_repair_private_cv_luma_image3": False,
                    "main_case_appearance_exposed_to_qwen": False,
                    "white_maquette_appearance_exposed_to_qwen": False,
                    "private_cv_gate_policy": (GROOMING_PRIVATE_POSE_GATE_GUIDE_POLICY),
                    "private_cv_gate_exposed_to_qwen_initial_attempts": False,
                    "normalized_private_depth_exposed_for_uploaded_white_maquette": (
                        True
                    ),
                    "private_cv_gate_neutral_copy_exposed_on_repair_attempt": False,
                    "public_open_channel_copy_exposed_on_repair_attempt": True,
                    "private_completion_mask_policy": (
                        GROOMING_PRIVATE_COMPLETION_MASK_POLICY
                    ),
                    "private_completion_mask_exposed_to_qwen": False,
                    "private_completion_mask_loaded_as_qwen_image": False,
                    "private_completion_mask_role": (
                        "local_noise_mask_derivation_only"
                    ),
                    "digest_bound": True,
                    "fail_closed": True,
                },
                "legacy_profile_action_reference_used": {
                    "eating": False,
                    "grooming": False,
                },
                "profile_action_reference_used_as_generation_input": {
                    "eating": False,
                    "grooming": (not GENERATION_VISUAL_AUDITS_BLOCKING),
                },
                "profile_action_reference_generation_scope": {
                    "eating": None,
                    "grooming": (
                        None
                        if GENERATION_VISUAL_AUDITS_BLOCKING
                        else "qwen_image2_action_geometry_non_identity_authority"
                    ),
                },
                "profile_action_reference_used_as_validation_baseline": {
                    "eating": True,
                    "grooming": True,
                },
                "pose_reference_role": "per_action_geometry_input_role",
                "pose_identity_input_policy": POSE_IDENTITY_INPUT_POLICY_CANONICAL,
                "pose_identity_input_policy_scope": (
                    "eating_and_grooming_canonical_image1_geometry_image2"
                ),
                "pose_identity_input_policies": {
                    "eating": POSE_IDENTITY_INPUT_POLICY_CANONICAL,
                    "grooming": POSE_IDENTITY_INPUT_POLICY_CANONICAL,
                },
                "pose_identity_input_role": {
                    "eating": "image1_qwen_full_body_canonical",
                    "grooming": "image1_qwen_full_body_canonical",
                },
                "pose_identity_palette_reference_role": {
                    "eating": "uploaded_identity_source",
                    "grooming": "uploaded_identity_source",
                },
                "pose_identity_palette_anchor_policy": (
                    "original_uploaded_front_source_for_every_action_attempt_v1"
                ),
                "pose_identity_geometry_role": {
                    "eating": "image2_main_case_driver_geometry_guide",
                    "grooming": (
                        "image2_main_case_driver_geometry_guide"
                        if GENERATION_VISUAL_AUDITS_BLOCKING
                        else "prompt_only_local_action_no_pose_reference_image"
                    ),
                },
                "grooming_advisory_first_frame": {
                    "enabled": (not GENERATION_VISUAL_AUDITS_BLOCKING),
                    "policy": GROOMING_ADVISORY_FIRST_FRAME_INPUT_POLICY,
                    "steps": GROOMING_ADVISORY_ACTION_TRANSFER_STEPS,
                    "denoise": GROOMING_ADVISORY_ACTION_TRANSFER_DENOISE,
                    "denoise_by_action_attempt": list(
                        GROOMING_ADVISORY_ACTION_TRANSFER_DENOISE_BY_ATTEMPT
                    ),
                    "image_roles": {
                        "image1": "approved_canonical_identity_sampler",
                    },
                    "prompt_only_action_geometry": True,
                    "video_first_frame_loaded_by_qwen": False,
                    "white_maquette_loaded_by_qwen": False,
                    "foreign_pose_image_loaded_by_qwen": False,
                    "white_maquette_head_and_ear_silhouette_removed": False,
                    "white_maquette_complete_pose_silhouette_retained": False,
                    "validated_action_rgb_loaded_by_qwen": False,
                    "validated_action_rgb_exposed_as_identity_authority": False,
                    "validated_action_chroma_loaded_by_qwen": False,
                    "validated_action_low_frequency_luma_loaded_by_qwen": False,
                    "validated_action_high_frequency_texture_loaded_by_qwen": False,
                    "public_open_channel_loaded_by_qwen": False,
                    "soft_foreleg_depth_loaded_by_qwen": False,
                    "normalized_white_maquette_low_frequency_depth_loaded_by_qwen": (
                        False
                    ),
                    "raw_white_maquette_luma_loaded_by_qwen": False,
                    "white_maquette_frame_index": None,
                    "white_maquette_head_identity_removed": False,
                    "canonical_local_action_edit_mask": {
                        "enabled": False,
                        "policy": (
                            "diagnostic_only_canonical_local_foreleg_mask_v2"
                        ),
                        "context_dilation_px": (
                            GROOMING_ADVISORY_ACTION_EDIT_MASK_CONTEXT_DILATION_PX
                        ),
                        "feather_sigma_px": (
                            GROOMING_ADVISORY_ACTION_EDIT_MASK_FEATHER_SIGMA_PX
                        ),
                        "generation_conditioning_only": True,
                        "applied_to_sampler": False,
                        "replacement": "full_canvas_prompted_action_redraw",
                    },
                    "post_qwen_pixel_composite_enabled": False,
                    "raw_qwen_output_required": True,
                    "raw_qwen_selection_byte_preserving": True,
                    "uploaded_front_identity_confirmation_loaded_by_qwen": False,
                    "same_canonical_face_identity_reference_loaded_by_qwen": False,
                    "uploaded_identity_pixels_consumed_via_canonical_only": True,
                    "uploaded_identity_palette_consumed_directly": True,
                    "appearance_free_pose_seed_loaded_by_qwen": False,
                    "identity_palette_prompt_conditioning": True,
                    "negative_prompt_conditioning": False,
                    "sampling_schedule": (
                        "qwen_image_edit_2511_lightning_exact_four_step_v1"
                    ),
                    "maximum_attempts": (
                        GROOMING_ADVISORY_TOTAL_ATTEMPTS
                    ),
                    "action_draft_maximum_attempts": (
                        GROOMING_ADVISORY_ACTION_TRANSFER_MAX_ATTEMPTS
                    ),
                    "identity_refine_maximum_attempts": (
                        GROOMING_ADVISORY_IDENTITY_REFINE_MAX_ATTEMPTS
                    ),
                    "conditional_identity_refine": {
                        "enabled": False,
                        "disabled_reason": (
                            "raw_qwen_action_output_required_no_post_generation_"
                            "pixel_composite"
                        ),
                        "policy": GROOMING_ADVISORY_IDENTITY_REFINE_POLICY,
                        "steps": GROOMING_ADVISORY_IDENTITY_REFINE_STEPS,
                        "denoise": GROOMING_ADVISORY_IDENTITY_REFINE_DENOISE,
                        "image1": "accepted_action_draft_pose_locked_sampler",
                        "image2": "approved_canonical_face_identity_authority",
                        "image3": "same_canonical_enlarged_face_identity",
                        "noise_mask": "upper_head_only_set_latent_noise_mask",
                        "head_height_fraction": (
                            GROOMING_ADVISORY_IDENTITY_REFINE_HEAD_HEIGHT_FRACTION
                        ),
                        "head_silhouette_expansion_px": (
                            GROOMING_ADVISORY_IDENTITY_REFINE_HEAD_EXPANSION_PX
                        ),
                        "mask_feather_sigma_px": (
                            GROOMING_ADVISORY_IDENTITY_REFINE_MASK_FEATHER_SIGMA_PX
                        ),
                        "deterministic_post_composite": (
                            None
                        ),
                        "body_and_raised_paw_pixel_lock": True,
                        "action_safe_source_allowed_identity_issues": sorted(
                            GROOMING_ADVISORY_IDENTITY_REFINE_SOURCE_ALLOWED_ISSUES
                        ),
                        "repairable_identity_issue_source_attempt": (
                            GROOMING_ADVISORY_ACTION_TRANSFER_MAX_ATTEMPTS
                        ),
                        "earlier_action_drafts_must_pass_minimum_contract": True,
                        "unexpected_dark_artifact_cleanup": {
                            "enabled_when_identity_black_fraction_lte": (
                                GROOMING_ADVISORY_IDENTITY_REFINE_SOURCE_BLACK_FRACTION_MAX
                            ),
                            "dark_value_max": (
                                GROOMING_ADVISORY_IDENTITY_REFINE_DARK_ARTIFACT_VALUE_MAX
                            ),
                            "minimum_component_fraction": (
                                GROOMING_ADVISORY_IDENTITY_REFINE_DARK_ARTIFACT_MIN_FRACTION
                            ),
                            "subject_support_dilation_px": (
                                GROOMING_ADVISORY_IDENTITY_REFINE_DARK_ARTIFACT_SUBJECT_SUPPORT_PX
                            ),
                        },
                        "white_maquette_loaded": False,
                        "fallback": "accepted_action_draft",
                    },
                    "minimum_contract": {
                        "policy": GROOMING_ADVISORY_MINIMUM_GATE_POLICY,
                        "version": GROOMING_ADVISORY_MINIMUM_GATE_VERSION,
                        "blocks_only": [
                            "identity_palette_replacement",
                            "new_dominant_dark_markings",
                            "severe_face_dark_marking_or_eye_hue_replacement",
                            "raised_foreleg_cv_evidence_unavailable",
                            "canonical_grooming_action_edit_evidence_unavailable",
                            "canonical_grooming_action_edit_too_weak",
                            "original_planted_forepaw_topology_retained",
                            "raised_foreleg_structural_edge_density_too_low",
                            "continuous_raised_foreleg_edge_too_short",
                            "paw_muzzle_cv_evidence_unavailable",
                            "raised_paw_too_far_from_muzzle",
                            "obvious_hard_object_instead_of_raised_paw",
                        ],
                        "advisory_only_pose_observations": [],
                        "blocking_pose_issues": sorted(
                            GROOMING_ADVISORY_BLOCKING_POSE_ISSUES
                        ),
                        "action_retry_guidance": sorted(
                            GROOMING_ADVISORY_ACTION_RETRY_WARNINGS
                        ),
                        "action_retry_guidance_before_attempt": (
                            GROOMING_ADVISORY_ACTION_RETRY_BEFORE_ATTEMPT
                        ),
                        "action_retry_guidance_never_blocks_final_attempt": True,
                        "final_attempt_fail_closed": False,
                        "terminal_selection_policy": (
                            "identity_first_lowest_risk_complete_chain_v1"
                        ),
                        "strict_then_relaxed_threshold_scale_by_attempt": list(
                            GROOMING_ADVISORY_CANONICAL_ACTION_THRESHOLD_SCALE_BY_ATTEMPT
                        ),
                        "edge_retention_relaxation_by_attempt": list(
                            GROOMING_ADVISORY_CANONICAL_EDGE_RETENTION_RELAXATION_BY_ATTEMPT
                        ),
                        "thresholds": {
                            "canonical_action_edit_mean_delta_min": (
                                GROOMING_ADVISORY_CANONICAL_ACTION_EDIT_MEAN_DELTA_MIN
                            ),
                            "canonical_action_edit_changed_fraction_min": (
                                GROOMING_ADVISORY_CANONICAL_ACTION_EDIT_CHANGED_FRACTION_MIN
                            ),
                            "canonical_planted_paw_edge_retention_max": (
                                GROOMING_ADVISORY_CANONICAL_PLANTED_PAW_EDGE_RETENTION_MAX
                            ),
                            "canonical_planted_paw_rgb_delta_min": (
                                GROOMING_ADVISORY_CANONICAL_PLANTED_PAW_RGB_DELTA_MIN
                            ),
                            "canonical_planted_paw_foreground_removed_fraction_min": (
                                GROOMING_ADVISORY_CANONICAL_PLANTED_PAW_FOREGROUND_REMOVED_FRACTION_MIN
                            ),
                            "palette_l1_max": (
                                GROOMING_ADVISORY_IDENTITY_PALETTE_L1_MAX
                            ),
                            "black_fraction_delta_max": (
                                GROOMING_ADVISORY_IDENTITY_BLACK_FRACTION_DELTA_MAX
                            ),
                            "chromatic_fraction_delta_max": (
                                GROOMING_ADVISORY_IDENTITY_CHROMATIC_FRACTION_DELTA_MAX
                            ),
                            "achromatic_balance_l1_max": (
                                GROOMING_ADVISORY_IDENTITY_ACHROMATIC_BALANCE_L1_MAX
                            ),
                            "hard_object_texture_ratio_max": (
                                GROOMING_ADVISORY_HARD_OBJECT_TEXTURE_RATIO_MAX
                            ),
                            "hard_object_raised_gradient_p90_min": (
                                GROOMING_ADVISORY_HARD_OBJECT_RAISED_GRADIENT_P90_MIN
                            ),
                        },
                        "masked_repair_unsafe_issues": sorted(
                            GROOMING_ADVISORY_MASKED_REPAIR_UNSAFE_ISSUES
                        ),
                        "hard_object_candidate_can_seed_masked_repair": False,
                        "fur_surface_audit_enabled": False,
                    },
                    "adds_visual_audit_retries": False,
                    "adds_minimum_contract_retries": True,
                },
                "eating_pose_lora_policy": (
                    "canonical_identity_plus_main_case_geometry_v1"
                ),
                "eating_identity_refine": {
                    "enabled": True,
                    "policy": EATING_CANONICAL_IDENTITY_REFINE_POLICY,
                    "attempt": EATING_CANONICAL_IDENTITY_REFINE_ATTEMPT,
                    "starts_at_attempt": EATING_CANONICAL_IDENTITY_REFINE_ATTEMPT,
                    "persists_for_remaining_attempts": True,
                    "generic_full_redraw_forbidden_after_source": True,
                    "source": ("best_prior_pose_and_bowl_safe_palette_drift_candidate"),
                    "steps": EATING_CANONICAL_IDENTITY_REFINE_STEPS,
                    "denoise": EATING_CANONICAL_IDENTITY_REFINE_DENOISE,
                    "qwen_lightning_only": True,
                    "style_lora": None,
                    "sampler_noise_mask": None,
                    "image_roles": {
                        "image1": (
                            "approved_eating_pose_and_bowl_sampler_no_identity_authority"
                        ),
                        "image2": (
                            "approved_canonical_sole_pet_identity_appearance_authority"
                        ),
                        "image3": (
                            "uploaded_front_real_identity_palette_corroboration"
                        ),
                    },
                    "identity_gate_unchanged": True,
                    "eating_pose_gate_unchanged": True,
                    "fur_surface_gate_unchanged": True,
                    "global_thresholds_changed": False,
                    "fallback_when_no_pose_safe_palette_drift_source": (
                        "existing_seed_retry"
                    ),
                },
                "grooming_pose_lora_policy": (
                    "canonical_identity_open_channel_seed_bound_pose_safe_refine_v11"
                    if GENERATION_VISUAL_AUDITS_BLOCKING
                    else (
                        "canonical_identity_prompted_local_foreleg_action_edit_"
                        "full_canvas_retry_then_degraded_minimum_contract_v8"
                    )
                ),
                "generated_canonical_bypassed": False,
                "grooming_pose_gate": {
                    "enabled": GENERATION_VISUAL_AUDITS_BLOCKING,
                    "policy": GROOMING_POSE_GATE_POLICY,
                    "version": GROOMING_POSE_GATE_VERSION,
                    "scope": "main_case_guided_grooming_qwen_candidates",
                    "input": "rembg_alpha_cutout",
                    "geometry_input": (
                        "public_open_channel_generation_plus_private_cv_validation"
                        if GENERATION_VISUAL_AUDITS_BLOCKING
                        else (
                            "canonical_prompt_full_canvas_action_redraw_plus_"
                            "nonterminal_minimum_contract"
                        )
                    ),
                    "fail_closed": GENERATION_VISUAL_AUDITS_BLOCKING,
                    "retry_on_rejection": GENERATION_VISUAL_AUDITS_BLOCKING,
                    "blocks_scail_on_exhaustion": (
                        GENERATION_VISUAL_AUDITS_BLOCKING
                    ),
                    "global_edge_density_min": (
                        GROOMING_QWEN_FORELEG_REPAIR_GLOBAL_EDGE_DENSITY_MIN
                    ),
                    "global_edge_density_min_unchanged": True,
                    "structural_edge_policy": (
                        "gaussian_lowpass_organic_foreleg_structure_v1"
                    ),
                    "structural_blur_sigma": (GROOMING_POSE_STRUCTURAL_BLUR_SIGMA),
                    "structural_edge_density_min": (
                        GROOMING_POSE_STRUCTURAL_EDGE_DENSITY_MIN
                    ),
                    "fine_fur_edges_cannot_satisfy_structural_gate": True,
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
                        "strict_chamfer_coverage_min": (
                            GROOMING_CANONICAL_DIRECT_CHAMFER_COVERAGE_MIN
                        ),
                        "strict_chamfer_mean_distance_max": (
                            GROOMING_CANONICAL_DIRECT_CHAMFER_MEAN_DISTANCE_MAX
                        ),
                        "strict_chamfer_p90_distance_max": (
                            GROOMING_CANONICAL_DIRECT_CHAMFER_P90_DISTANCE_MAX
                        ),
                        "offguide_organic_paw_evidence": {
                            "only_chamfer_issues_allowed": True,
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
                        "rejects_historical_straight_prop_candidates": True,
                        "rejects_mismatched_hanging_prop_material": True,
                        "identity_palette_gate_still_required": True,
                        "independent_fur_surface_gate_still_required": True,
                        "global_pose_threshold_changed": False,
                        "fail_closed": True,
                    },
                    "canonical_direct_surface_nonexpansion": {
                        "policy": (
                            GROOMING_CANONICAL_DIRECT_SURFACE_NONEXPANSION_POLICY
                        ),
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
                    "masked_pet3d_composite_recovery": {
                        "production_enabled": False,
                        "legacy_geometry_first_jobs_only": True,
                        "policy": (GROOMING_QWEN_FORELEG_REPAIR_COMPOSITE_GATE_POLICY),
                        "attempt": GROOMING_QWEN_FORELEG_REPAIR_ATTEMPT,
                        "edge_density_ratio_min": (
                            GROOMING_QWEN_FORELEG_REPAIR_EDGE_DENSITY_RATIO_MIN
                        ),
                        "requires_edge_density_issue": True,
                        "allows_only_optional_low_texture_planted_paw_ratio_issue": (
                            True
                        ),
                        "planted_paw_gradient_p90_max": (
                            GROOMING_QWEN_FORELEG_REPAIR_PLANTED_PAW_GRADIENT_P90_MAX
                        ),
                        "raised_paw_gradient_p90_max": (
                            GROOMING_QWEN_FORELEG_REPAIR_RAISED_PAW_GRADIENT_P90_MAX
                        ),
                        "requires_geometry_contact_color_histogram_pass": True,
                        "requires_independent_fur_surface_pass": True,
                        "requires_digest_bound_noise_masked_pet3d_repair": True,
                        "fail_closed": True,
                    },
                },
                "fur_surface_gate": fur_surface_gate_health,
                "eating_geometry_first_denoise": {
                    "enabled": False,
                    "default_policy": False,
                    "reason": "production_uploads_use_qwen_canonical_image1",
                },
                "reference_prompt_source": (
                    "uploaded_identity_qwen_full_body_profile_locked"
                ),
                "user_prompt_excluded_from_reference": True,
                "eating_pose_strength_max": SCAIL_EATING_POSE_STRENGTH_MAX,
                "sleek_eating_profile_fast_path_policy": (
                    SCAIL_FROZEN_SLEEK_EATING_POLICY
                ),
                "sleek_eating_qwen_reference_bypassed_for_upload": False,
                "sleek_eating_profile_fast_path_without_upload": True,
                "sleek_eating_uploaded_identity_required": True,
                "sleek_eating_uploaded_identity_applied": True,
                "canonical_retry_lora_strengths_locked": False,
                "pose_retry_lora_strengths_locked": False,
                "eating_motion_alignment": (FULL_DRIVER_EATING_POSE_POLICY),
                "grooming_temporal_loop": {
                    "policy": GROOMING_TEMPORAL_LOOP_POLICY,
                    "source_window": "validated_second_half_from_frame_40",
                    "playback": "forward_then_reverse",
                    "closed_loop_required": True,
                    "preserves_duration_and_frame_count": True,
                    "published_boundary_max_adjacent_ratio": 1.5,
                    "fail_closed": True,
                },
                "eating_reference_fur_gate": ("profile_relative_eating_contour_v1"),
                "eating_video_fur_gate": (FINAL_VIDEO_FUR_AUDIT_POLICY),
                "fur_constraints_snapshot_locked": True,
                "subject_replacement_pose_denoise": (
                    QWEN_SUBJECT_REPLACEMENT_POSE_DENOISE
                ),
                "subject_replacement_pose_denoise_step": (
                    QWEN_SUBJECT_REPLACEMENT_POSE_DENOISE_STEP
                ),
                "max_attempts": QWEN_POSE_MAX_ATTEMPTS,
                "running_extra_attempts": QWEN_RUNNING_POSE_EXTRA_ATTEMPTS,
                "bowl_extra_attempts": QWEN_BOWL_POSE_EXTRA_ATTEMPTS,
                "running_max_attempts": (
                    QWEN_POSE_MAX_ATTEMPTS + QWEN_RUNNING_POSE_EXTRA_ATTEMPTS
                ),
                "bowl_max_attempts": (
                    QWEN_POSE_MAX_ATTEMPTS + QWEN_BOWL_POSE_EXTRA_ATTEMPTS
                ),
                "bowl_palette_bottom_exclusion": (
                    BOWL_IDENTITY_PALETTE_BOTTOM_EXCLUSION
                ),
                "bowl_palette_upper_body_exclusion": (
                    BOWL_IDENTITY_PALETTE_UPPER_BODY_EXCLUSION
                ),
                "similarity_iou": QWEN_POSE_SIMILARITY_IOU_THRESHOLD,
                "similarity_dhash": QWEN_POSE_SIMILARITY_DHASH_THRESHOLD,
                "action_canonical_iou": QWEN_ACTION_CANONICAL_IOU_THRESHOLD,
                "action_canonical_dhash": QWEN_ACTION_CANONICAL_DHASH_THRESHOLD,
                "bowl_action_min_aspect_ratio": (QWEN_BOWL_ACTION_MIN_ASPECT_RATIO),
                "hard_duplicate_iou": QWEN_POSE_HARD_DUPLICATE_IOU,
                "hard_duplicate_dhash": QWEN_POSE_HARD_DUPLICATE_DHASH,
            },
        },
        "wan_video": {
            **wan_engine_metadata(),
            "ok": not any(not path.is_file() for path in wan_model_paths()),
            "missing": [str(path) for path in wan_model_paths() if not path.is_file()],
        },
        "wan_control": {
            "enabled": FUN_CONTROL_ENABLED,
            "engine": "wan2.2-fun-5b-control" if FUN_CONTROL_ENABLED else None,
            "actions": ["running"] if FUN_CONTROL_ENABLED else [],
            "resolution": [512, 512] if FUN_CONTROL_ENABLED else None,
            "frames": FRAME_COUNT if FUN_CONTROL_ENABLED else None,
            "fps": FPS if FUN_CONTROL_ENABLED else None,
            "steps": FUN_CONTROL_STEPS if FUN_CONTROL_ENABLED else None,
            "gpu_memory_mode": FUN_CONTROL_GPU_MEMORY_MODE
            if FUN_CONTROL_ENABLED
            else None,
            "fallback_to_wan": FUN_CONTROL_FALLBACK_TO_WAN
            if FUN_CONTROL_ENABLED
            else None,
            "model_root": str(FUN_CONTROL_MODEL_ROOT) if FUN_CONTROL_ENABLED else None,
            "project_root": str(FUN_CONTROL_PROJECT_ROOT)
            if FUN_CONTROL_ENABLED
            else None,
            "control_video": str(FUN_CONTROL_RUNNING_VIDEO)
            if FUN_CONTROL_ENABLED
            else None,
            "ok": not fun_control_missing if FUN_CONTROL_ENABLED else True,
            "missing": fun_control_missing if FUN_CONTROL_ENABLED else [],
        },
        "ffmpeg": {"ok": ffmpeg_ok, "path": FFMPEG_BIN if ffmpeg_ok else None},
    }
