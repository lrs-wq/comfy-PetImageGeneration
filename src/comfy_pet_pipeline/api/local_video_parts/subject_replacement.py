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
        != "driving_eating_geometry_image2_v6"
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
        original_generation_error=f"{type(error).__name__}: {str(error)[:4000]}",
        subject_replacement_results=[
            results_by_action[action]
            for action in actions
            if action in results_by_action
        ],
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
    if (
        job.get("subject_source") != "qwen_generated_upload"
        or job.get("reference_generation_completed") is not True
        or job.get("reference_generation_status") != "completed"
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
            and job.get("surface_texture_gate_status") != "passed"
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
        and final_surface_quality.get("status") == "passed"
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
            require_uploaded_fur_surface_gate(
                effective_subject_config["actions"][action],
                action,
            )

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
        revalidate_uploaded_qwen_references_before_scail(
            job_id,
            job,
            effective_subject_config,
            actions,
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
        for action in actions
    }
    effective_reference_paths = {
        action: str(selected_references[action][0]) for action in actions
    }
    selected_reference_policies = {
        action: selected_references[action][1] for action in actions
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
            action for action in actions if action not in uploaded_applied_actions
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
    require_subject_replacement_assets(
        profile or {"action_references": effective_reference_paths},
        actions,
        reference_paths=effective_reference_paths,
        effective_config=effective_subject_config,
        expected_pose_policies=expected_pose_policies,
    )
    total_actions = len(actions)
    results: list[dict[str, Any]] = []
    final_video_fur_surface_by_action: dict[str, Any] = {}
    scail_reference_policies = dict(job.get("scail_reference_policy") or {})
    scail_scene_prop_policies = dict(job.get("scail_scene_prop_policy") or {})
    reference_generation_chain = bool(job.get("reference_generation_completed"))
    progress_base = 38.0 if reference_generation_chain else 3.0
    progress_total = 59.0 if reference_generation_chain else 94.0
    # Dog eating has the strictest prop and contact path.  Run independent dog
    # actions first so a later eating failure cannot discard an already valid
    # standing animation; terminal recovery will preserve that output and fill
    # only the failed/missing action.
    execution_actions = list(actions)
    if str(job.get("species") or "").strip().lower() == "dog":
        execution_actions.sort(key=lambda action: action == "eating")

    update_job(
        job_id,
        status="running",
        stage="waiting_for_gpu",
        progress=36 if reference_generation_chain else 1,
        started_at=job.get("started_at") or utc_now(),
        error=None,
        current_action=None,
        subject_replacement_execution_order=execution_actions,
        subject_replacement_failure_isolation_policy=(
            "independent_dog_actions_before_strict_eating_v1"
            if execution_actions != actions
            else "requested_action_order_v1"
        ),
    )
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
        expected_pose_policy = expected_pose_policies.get(action)
        require_expected_scail_pose_policy(
            manifest,
            action,
            expected_pose_policy,
        )
        pose_metadata = manifest.get("pose_metadata")
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
        scail_scene_prop_policies[action] = (
            (
                "dog_white_model_bowl_scail_object_mapping"
                if job.get("species") == "dog"
                else "cat_blue_bowl_red_scail_object_mapping"
            )
            if action == "eating"
            else "no_scene_prop"
        )
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
        entry = queue_and_wait(
            workflow,
            f"SCAIL subject replacement {action} for {job_id}",
            timeout=7200,
            job_id=job_id,
        )
        generated_video = first_output(
            entry,
            {".mp4"},
            f"SCAIL {action} subject replacement",
        )
        video_target = job_dir / f"{action}_subject_replacement.mp4"
        shutil.copy2(generated_video, video_target)
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
                raise RuntimeError(
                    f"{action}_final_video_fur_surface_gate_failed: {detail}"
                )
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
            guidance_path = Path(manifest["input_files"]["pose_mask_video"])
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
            update_job(
                job_id,
                transparent_output_phase_policy=(
                    "validated_action_contact_phase_rotation_v1"
                ),
                transparent_output_phase_start_index=(
                    transparent_phase_start_index
                ),
                transparent_output_phase_action=action,
            )
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
            apng_available = apng_target.is_file()
            transparent_metrics = logged_json(transparent_log)
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
                    raise RuntimeError(
                        "grooming_temporal_loop_failed: the published animation "
                        "did not preserve the closed-loop action contract"
                    )
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
                raise RuntimeError(
                    "eating_fur_contour_gate_failed: "
                    + ", ".join(fur_contour_quality["issues"])
                )
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
                "effective_seed": int(action_config["effective_seed"]),
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
    update_job(
        job_id,
        status="completed",
        stage="completed",
        progress=100,
        current_action=None,
        current_action_label=None,
        current_action_index=None,
        subject_replacement_results=results,
        error=nonblocking_result_error(job_id),
        finished_at=utc_now(),
    )
