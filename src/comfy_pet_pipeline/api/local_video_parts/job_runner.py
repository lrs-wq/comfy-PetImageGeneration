"""Top-level local-video job orchestration."""

from __future__ import annotations

# The extracted functions resolve the facade dependencies after runtime sync.
# ruff: noqa: F821

from .runtime import sync_local_video_runtime as _sync_local_video_runtime


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
        return (
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


def exhausted_action_candidate_recovery_policy(
    record: dict[str, Any],
    *,
    action: str,
    approved_canonical_available: bool,
) -> str:
    """Choose a successful retry recovery without re-admitting identity drift."""
    palette = record.get("identity_palette")
    if isinstance(palette, dict) and palette.get("acceptable") is True:
        return "identity_safe_generated_candidate"
    if action == "standing" and approved_canonical_available:
        return "upload_anchored_canonical"
    return "reject"


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
    base_seed = int(job["seed"])
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
                    chroma_metrics["identity_chroma_spill_cleanup"] = (
                        neutralize_uploaded_identity_chroma_spill(
                            attempt_cutout,
                            source_identity_palette,
                        )
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
            ]
            fallback_uses_uploaded_identity = bool(
                canonical_identity_palette_gate_required
                and not identity_safe_generated_attempts
            )
            fallback_source = (
                Path(identity_source_path)
                if fallback_uses_uploaded_identity
                else job_dir / fallback_record["keyframe"]
            )
            fallback_green = job_dir / "cartoon_reference_semantic_green.png"
            fallback_cutout = job_dir / "cartoon_reference_semantic_cutout_final.png"
            fallback_semantic = job_dir / "cartoon_reference_semantic_cutout_raw.png"
            try:
                chroma_metrics = semantic_background_recovery(
                    fallback_source,
                    fallback_green,
                    fallback_cutout,
                    fallback_semantic,
                )
            except Exception as exc:  # noqa: BLE001 - image mode must preserve a usable raw result
                if not image_request:
                    raise
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
            else:
                chroma_metrics["identity_chroma_spill_cleanup"] = (
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
                fallback_profile = (
                    "uploaded_identity_semantic_fallback"
                    if fallback_uses_uploaded_identity
                    else "rembg_u2net_fallback"
                )
                fallback_recovery_reason = (
                    "canonical_identity_palette_drift_exhausted"
                    if fallback_uses_uploaded_identity
                    else "qwen_chroma_retries_exhausted"
                )
                fallback_recovery_strategy = (
                    "use_original_uploaded_identity_after_safe_matte"
                    if fallback_uses_uploaded_identity
                    else "rembg_u2net_fallback"
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
                    update_job(
                        job_id,
                        canonical_uploaded_identity_fallback_used=True,
                        canonical_generated_palette_failure={
                            "attempts": [
                                record.get("identity_palette")
                                for record in cartoon_attempts
                            ],
                        },
                        canonical_uploaded_identity_fallback_source=(
                            str(fallback_source)
                        ),
                    )
                if (
                    canonical_identity_palette_gate_required
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
                            )
                            uploaded_chroma_metrics[
                                "identity_chroma_spill_cleanup"
                            ] = neutralize_uploaded_identity_chroma_spill(
                                uploaded_fallback_cutout,
                                source_identity_palette,
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
        if dog_eating_prompt_only_action_transfer_active:
            update_job(
                job_id,
                dog_eating_first_frame_policy=(
                    DOG_EATING_PROMPT_ONLY_FIRST_FRAME_POLICY
                ),
                dog_eating_first_frame_prompt_only_action_geometry=True,
                dog_eating_first_frame_image1_role=(
                    "approved_upload_anchored_canonical_identity_sampler"
                ),
                dog_eating_first_frame_white_model_loaded_by_qwen=False,
                dog_eating_first_frame_secondary_reference_loaded_by_qwen=False,
                dog_eating_first_frame_action_geometry_deferred_to_scail=True,
                dog_eating_first_frame_steps=(
                    DOG_EATING_PROMPT_ONLY_ACTION_TRANSFER_STEPS
                ),
                dog_eating_first_frame_denoise=(
                    DOG_EATING_PROMPT_ONLY_ACTION_TRANSFER_DENOISE
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
                elif species == "dog" and action_key(pose) == "standing":
                    # Standing keyframes must be retried when the pre-normalization
                    # alpha reaches an edge (most commonly a generated floor
                    # reflection).  Later matte normalization would otherwise hide
                    # the edge contact and pass that artifact into SCAIL.
                    active_pose_profiles = active_pose_profiles[
                        :QWEN_POSE_MAX_ATTEMPTS
                    ]
                elif not advisory_action_candidate_gate_required:
                    active_pose_profiles = active_pose_profiles[:1]
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
            pose_identity_palette_gate_required = bool(
                identity_palette_guard
                or advisory_action_candidate_gate_required
            )
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
            best_advisory_grooming_degraded_attempt: dict[str, Any] | None = None
            best_raw_attempt: dict[str, Any] | None = None
            best_semantic_source: dict[str, Any] | None = None
            grooming_foreleg_repair_source: dict[str, Any] | None = None
            grooming_full_pose_refine_source: dict[str, Any] | None = None
            grooming_advisory_identity_refine_source: dict[str, Any] | None = None
            eating_identity_refine_source: dict[str, Any] | None = None
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
                update_job(
                    job_id,
                    grooming_action_first_frame_input_policy=(
                        GROOMING_ADVISORY_FIRST_FRAME_INPUT_POLICY
                    ),
                    grooming_action_first_frame_open_channel_loaded_by_qwen=False,
                    grooming_action_first_frame_soft_foreleg_depth_loaded_by_qwen=False,
                    grooming_action_first_frame_normalized_white_maquette_depth_loaded_by_qwen=False,
                    grooming_action_first_frame_identity_input=(qwen_input_name),
                    grooming_action_first_frame_identity_confirmation_input=None,
                    grooming_action_first_frame_geometry_input=None,
                    grooming_action_first_frame_action_reference_role=None,
                    grooming_action_first_frame_action_reference_sha256=None,
                    grooming_action_first_frame_action_canvas=None,
                    grooming_action_first_frame_image_roles={
                        "image1": "approved_canonical_identity_sampler",
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
                    grooming_action_first_frame_prompt_only_action_geometry=True,
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
                    grooming_action_first_frame_normalized_low_frequency_geometry_loaded_by_qwen=False,
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
            for attempt, profile in enumerate(active_pose_profiles, start=1):
                if (
                    advisory_grooming_action_transfer_active
                    and attempt
                    > GROOMING_ADVISORY_ACTION_TRANSFER_MAX_ATTEMPTS
                    and not isinstance(
                        grooming_advisory_identity_refine_source,
                        dict,
                    )
                ):
                    break
                raise_if_cancelled(job_id)
                if attempt > 1:
                    update_job(
                        job_id,
                        stage="retrying_cartoon_keyframe",
                        quality_retry=attempt,
                        recovery_active=True,
                        recovery_attempt=attempt,
                    )
                attempt_seed = (
                    base_seed + pose_seed_offset(pose, index) + (attempt - 1) * 1009
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
                    and not dog_eating_prompt_only_action_transfer_active
                    and action_key(pose) == "eating"
                    and attempt >= EATING_CANONICAL_IDENTITY_REFINE_ATTEMPT
                    and isinstance(eating_identity_refine_source, dict)
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
                pose_transfer_seed_path: Path | None = None
                grooming_advisory_action_transfer_attempt: (
                    dict[str, Any] | None
                ) = None
                dog_eating_prompt_only_action_transfer_attempt: (
                    dict[str, Any] | None
                ) = None
                if dog_eating_prompt_only_action_transfer_active:
                    attempt_primary_input_name = qwen_input_name
                    attempt_secondary_input_name = None
                    attempt_tertiary_input_name = None
                    attempt_additional_reference_input_names = None
                    attempt_noise_mask_input_name = None
                    attempt_geometry_input_name = None
                    attempt_prompt = (
                        build_dog_eating_prompt_only_action_transfer_prompt(
                            pose,
                            profile_fur_lock=profile_fur_lock,
                        )
                    )
                    dog_eating_prompt_only_action_transfer_attempt = {
                        "policy": DOG_EATING_PROMPT_ONLY_FIRST_FRAME_POLICY,
                        "image1_role": (
                            "approved_upload_anchored_canonical_identity_sampler"
                        ),
                        "image2_role": None,
                        "image3_role": None,
                        "prompt_only_action_geometry": True,
                        "white_model_loaded_by_qwen": False,
                        "secondary_reference_loaded_by_qwen": False,
                        "foreign_pose_image_loaded_by_qwen": False,
                        "uploaded_identity_views_consumed_via_canonical_only": True,
                        "full_canvas_action_redraw": True,
                        "action_geometry_deferred_to_scail": True,
                        "sampler_noise_mask_input_name": None,
                        "post_qwen_pixel_composite_enabled": False,
                    }
                if advisory_grooming_action_transfer_active:
                    attempt_primary_input_name = qwen_input_name
                    attempt_secondary_input_name = None
                    attempt_tertiary_input_name = None
                    attempt_additional_reference_input_names = None
                    # The earlier local latent lock repeatedly reproduced the
                    # canonical two-paws-down topology even at denoise=1.0.  Keep
                    # the digest-bound canonical as Image 1/start latent, but let
                    # Qwen redraw the full frame so it can actually relocate the
                    # planted foreleg instead of adding a third paw around it.
                    attempt_noise_mask_input_name = None
                    attempt_geometry_input_name = None
                    attempt_prompt = build_advisory_grooming_action_transfer_prompt(
                        pose,
                        profile_fur_lock=profile_fur_lock,
                    )
                    grooming_advisory_action_transfer_attempt = {
                        "policy": GROOMING_ADVISORY_FIRST_FRAME_INPUT_POLICY,
                        "image1_role": "approved_canonical_identity_sampler",
                        "image2_role": None,
                        "image3_role": None,
                        "uploaded_identity_views_consumed_via_canonical_only": True,
                        "prompt_only_action_geometry": True,
                        "video_first_frame_loaded_by_qwen": False,
                        "white_maquette_loaded_by_qwen": False,
                        "foreign_pose_image_loaded_by_qwen": False,
                        "raw_action_rgb_loaded_by_qwen": False,
                        "source_low_frequency_luma_loaded_by_qwen": False,
                        "normalized_low_frequency_geometry_loaded_by_qwen": False,
                        "public_open_channel_loaded_by_qwen": False,
                        "soft_foreleg_depth_loaded_by_qwen": False,
                        "white_maquette_frame_index": None,
                        "white_maquette_head_identity_removed": False,
                        "white_maquette_head_silhouette_removed": False,
                        "canonical_local_action_edit_mask": (
                            None
                        ),
                        "canonical_local_action_edit_mask_applied": False,
                        "full_canvas_action_redraw": True,
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
                if canonical_eating_identity_refine_active:
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
                if dog_eating_prompt_only_action_transfer_active:
                    effective_pose_denoise = (
                        DOG_EATING_PROMPT_ONLY_ACTION_TRANSFER_DENOISE
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
                if dog_eating_prompt_only_action_transfer_active:
                    effective_pose_steps = (
                        DOG_EATING_PROMPT_ONLY_ACTION_TRANSFER_STEPS
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
                workflow = qwen_workflow(
                    attempt_primary_input_name,
                    attempt_prompt,
                    (
                        ""
                        if (
                            advisory_grooming_action_transfer_active
                            or dog_eating_prompt_only_action_transfer_active
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
                source_output = first_output(
                    entry,
                    {".png", ".jpg", ".jpeg", ".webp"},
                    "Qwen keyframe",
                )
                source_output_sha256 = file_sha256(source_output)
                raw_qwen_byte_preserved = False
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
                        chroma_metrics["identity_chroma_spill_cleanup"] = (
                            neutralize_uploaded_identity_chroma_spill(
                                attempt_cutout,
                                source_identity_palette,
                            )
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
                if (
                    canonical_action_reference
                    and not dog_eating_prompt_only_action_transfer_active
                    and action_key(pose) == "eating"
                    and attempt < EATING_CANONICAL_IDENTITY_REFINE_ATTEMPT
                    and chroma_metrics.get("acceptable")
                    and attempt_cutout.is_file()
                    and eating_refine_pose_safe
                    and isinstance(palette_quality, dict)
                    and palette_quality.get("available") is True
                    and palette_quality.get("acceptable") is not True
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
                        if canonical_eating_identity_refine_active
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
                    "grooming_advisory_identity_refine": (
                        grooming_advisory_identity_refine_audit
                    ),
                    "grooming_foreleg_repair": (grooming_foreleg_repair_audit),
                    "grooming_full_pose_recovery": (grooming_full_pose_recovery_audit),
                    "eating_identity_refine": eating_identity_refine_audit,
                    "raw_qwen_source_path": str(source_output),
                    "raw_qwen_source_sha256": source_output_sha256,
                    "raw_qwen_byte_preserved": raw_qwen_byte_preserved,
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
                            "blocks_scail": True,
                        }
                        fur_surface_gate_status_by_action[action_key(pose)] = "failed"
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
                                "status": "failed",
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
                                "fail_closed": True,
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
                        raise RuntimeError(
                            "Main_case fur surface gate rejected "
                            f"{fur_surface_rejection_streak} consecutive "
                            f"{action_key(pose)} Qwen candidates after attempt "
                            f"{attempt} of {len(active_pose_profiles)}; "
                            "blocking SCAIL."
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
                    if semantic_recovery_candidate_eligible(
                        record
                    ) and not chroma_metrics.get(
                        "semantic_recovery_attempted",
                        False,
                    ):
                        # The first subject-replacement seed is the profile's
                        # empirically validated action seed. Preserve it as the
                        # primary semantic fallback; later seeds remain available
                        # as diagnostics instead of replacing it based on a raw
                        # background-contaminated similarity score.
                        if best_semantic_source is None:
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
                                    palette_quality is None
                                    or palette_quality["acceptable"]
                                )
                                and (
                                    fur_silhouette_quality is None
                                    or fur_silhouette_quality["acceptable"]
                                )
                                and (
                                    not fur_surface_gate_required
                                    or (
                                        isinstance(fur_surface_gate_metrics, dict)
                                        and fur_surface_gate_metrics.get("acceptable")
                                        is True
                                    )
                                )
                                and (
                                    not grooming_pose_gate_required
                                    or (
                                        isinstance(grooming_pose_gate_metrics, dict)
                                        and grooming_pose_gate_metrics.get("acceptable")
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
                        best_attempt = record
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
                    best_attempt = record
                    if advisory_action_candidate_gate_required:
                        update_job(
                            job_id,
                            action_candidate_gate_policy=(
                                GENERATION_ACTION_CANDIDATE_GATE_POLICY
                            ),
                            action_candidate_gate_status="passed",
                            action_candidate_gate_action=action_key(pose),
                            action_candidate_gate_selected_attempt=attempt,
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
                                "reason": "earlier_keyframe_attempt_rejected",
                                "strategy": "pose_prompt_and_lora_retry",
                                "profile": profile["name"],
                                "result": "accepted",
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
                    break
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
                    break
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
            if (
                best_attempt is None
                and advisory_action_candidate_gate_required
                and best_action_candidate_degraded_attempt is not None
            ):
                approved_canonical_available = bool(
                    action_key(pose) == "standing"
                    and cartoon_is_transparent
                    and cartoon_output.is_file()
                )
                exhausted_recovery = exhausted_action_candidate_recovery_policy(
                    best_action_candidate_degraded_attempt,
                    action=action_key(pose),
                    approved_canonical_available=approved_canonical_available,
                )
                if exhausted_recovery == "identity_safe_generated_candidate":
                    best_attempt = best_action_candidate_degraded_attempt
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
                            "identity_safe_structural_action_generated_candidate_v3"
                        ),
                        degraded_selection_key=list(
                            best_attempt[
                                "action_candidate_degraded_selection_key"
                            ]
                        ),
                        returns_original_upload=False,
                        returns_approved_canonical=False,
                        blocks_job=False,
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
                                "select_identity_safe_structural_action_generated_"
                                "candidate"
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
                            "identity_safe_degraded_candidate_selected"
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
                    )
                elif exhausted_recovery == "upload_anchored_canonical":
                    canonical_fallback = (
                        job_dir
                        / f"pose_{index:02d}_upload_anchored_canonical_fallback.png"
                    )
                    canonical_normalization = (
                        normalize_qwen_action_reference_rgba(
                            cartoon_output,
                            canonical_fallback,
                            action=action_key(pose),
                        )
                    )
                    canonical_edge_quality = canonical_normalization[
                        "transparent_edge_quality"
                    ]
                    canonical_quality = pose_cutout_quality(
                        canonical_fallback,
                        pose,
                        require_eating_mouth_contact=False,
                    )
                    canonical_palette = evaluate_pose_identity_palette(
                        canonical_fallback,
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
                            "Upload-anchored canonical standing fallback failed "
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
                            "upload_anchored_canonical_standing_reference_v1"
                        ),
                        degraded_selection_key=list(
                            rejected_attempt[
                                "action_candidate_degraded_selection_key"
                            ]
                        ),
                        returns_original_upload=False,
                        returns_approved_canonical=True,
                        action_geometry_deferred_to_scail=True,
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
                        ),
                        "canonical_self_reference_reuse": True,
                        "action_geometry_deferred_to_scail": True,
                    }
                    best_attempt = {
                        **rejected_attempt,
                        "attempt": len(active_pose_profiles) + 1,
                        "profile": (
                            "upload_anchored_canonical_standing_recovery"
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
                                "upload_anchored_canonical_standing_reference_v1"
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
                            "action_geometry_deferred_to_scail": True,
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
                    "identity_first_lowest_risk_complete_chain_v1"
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
                chroma_metrics = semantic_background_recovery(
                    fallback_source,
                    fallback_green,
                    fallback_cutout,
                    fallback_semantic,
                    pose=pose,
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
                chroma_metrics["identity_chroma_spill_cleanup"] = (
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
                    semantic_attempt = len(attempts) + 1
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
                if semantic_rejection_reason is not None:
                    append_recovery_event(
                        job_id,
                        {
                            "stage": "cartoon_keyframe",
                            "pose_index": index,
                            "attempt": len(attempts) + 1,
                            "reason": semantic_rejection_reason,
                            "strategy": "rembg_u2net_fallback",
                            "profile": "rembg_u2net_fallback",
                            "result": "rejected",
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
                    if fur_surface_gate_required:
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
                    raise RuntimeError(
                        f"Pose {index} semantic recovery failed quality checks: "
                        + ", ".join(issues)
                    )
                best_attempt = {
                    **{
                        field: fallback_record.get(field)
                        for field in POSE_ATTEMPT_PROVENANCE_FIELDS
                    },
                    "attempt": len(attempts) + 1,
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
                attempts.append(
                    {
                        key: value
                        for key, value in best_attempt.items()
                        if key != "workflow"
                    }
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
                        "result": "accepted",
                        "chroma": chroma_metrics,
                        "quality": quality,
                        "similarity": similarity,
                        "identity_palette": semantic_palette_quality,
                        "fur_silhouette": semantic_fur_silhouette_quality,
                        "fur_surface_gate": semantic_fur_surface_quality,
                        "color_recovery": semantic_color_recovery,
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
            if fur_surface_gate_required:
                selected_surface = best_attempt.get("fur_surface_gate")
                if (
                    not isinstance(selected_surface, dict)
                    or selected_surface.get("acceptable") is not True
                    or selected_surface.get("status") != "passed"
                ):
                    fur_surface_gate_status_by_action[action_key(pose)] = "failed"
                    update_job(
                        job_id,
                        fur_surface_gate_status=dict(fur_surface_gate_status_by_action),
                        fur_surface_gate_attempts=(fur_surface_gate_attempt_audits),
                    )
                    raise RuntimeError(
                        "Main_case fur surface gate selected an invalid "
                        f"{action_key(pose)} candidate."
                    )
                selected_digest = file_sha256(selected_keyframe)
                selected_gate_metadata = selected_surface.get("gate")
                if (
                    not isinstance(selected_gate_metadata, dict)
                    or selected_gate_metadata.get("policy") != FUR_SURFACE_GATE_POLICY
                    or selected_gate_metadata.get("gate_version")
                    != FUR_SURFACE_GATE_VERSION
                    or selected_gate_metadata.get("threshold_version")
                    != FUR_SURFACE_THRESHOLD_VERSION
                    or selected_gate_metadata.get("action") != action_key(pose)
                    or selected_gate_metadata.get("candidate_sha256") != selected_digest
                ):
                    raise RuntimeError(
                        "Main_case fur surface selected-candidate provenance changed."
                    )
                selected_surface = json.loads(json.dumps(selected_surface))
                selected_surface["selected_candidate_path"] = str(selected_keyframe)
                selected_surface["selected_candidate_sha256"] = selected_digest
                selected_surface["selected_attempt"] = best_attempt["attempt"]
                best_attempt["fur_surface_gate"] = selected_surface
                fur_surface_gate_selected_metrics[action_key(pose)] = selected_surface
                fur_surface_gate_status_by_action[action_key(pose)] = "passed"
                update_job(
                    job_id,
                    fur_surface_gate_status=dict(fur_surface_gate_status_by_action),
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
            if grooming_pose_gate_required:
                selected_gate = best_attempt.get("grooming_pose_gate")
                if not isinstance(selected_gate, dict) or not selected_gate.get(
                    "acceptable"
                ):
                    update_job(
                        job_id,
                        grooming_pose_gate_status="failed",
                        grooming_pose_gate_attempts=(grooming_pose_gate_attempt_audits),
                    )
                    raise RuntimeError(
                        "Main_case grooming pose gate selected an invalid candidate."
                    )
                grooming_pose_gate_selected_metrics[action_key(pose)] = selected_gate
                update_job(
                    job_id,
                    grooming_pose_gate_status="passed",
                    grooming_pose_gate_attempts=(grooming_pose_gate_attempt_audits),
                    grooming_pose_gate_selected_metrics=(
                        grooming_pose_gate_selected_metrics
                    ),
                )
            if advisory_grooming_minimum_gate_required:
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
            write_json_atomic(
                job_dir / "workflows" / f"qwen_pose_{index:02d}_quality.json",
                {
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
                },
            )
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
        expected_surface_actions = list(fur_surface_gate_status_by_action)
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
                "status": "passed",
                "policy": FUR_SURFACE_GATE_POLICY,
                "gate_version": FUR_SURFACE_GATE_VERSION,
                "threshold_version": FUR_SURFACE_THRESHOLD_VERSION,
                "actions": dict(fur_surface_gate_selected_metrics),
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
