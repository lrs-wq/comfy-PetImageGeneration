"""Image preparation and geometry canvases for grooming generation."""

from __future__ import annotations

# The extracted functions resolve the facade dependencies after runtime sync.
# ruff: noqa: F821

from .runtime import sync_local_video_runtime as _sync_local_video_runtime


def wait_comfy_idle(timeout: int = 3600, job_id: str | None = None) -> None:
    _sync_local_video_runtime(globals(), {"wait_comfy_idle"})
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        raise_if_cancelled(job_id)
        response = requests.get(f"{COMFY_URL}/queue", timeout=30)
        response.raise_for_status()
        payload = response.json()
        if not payload.get("queue_running") and not payload.get("queue_pending"):
            return
        time.sleep(5)
    raise TimeoutError("Timed out waiting for the ComfyUI queue to become idle.")


def free_comfy_models() -> None:
    _sync_local_video_runtime(globals(), {"free_comfy_models"})
    response = requests.post(
        f"{COMFY_URL}/free",
        json={"unload_models": True, "free_memory": True},
        timeout=60,
    )
    response.raise_for_status()


def queue_and_wait(
    workflow: dict[str, Any],
    stage_name: str,
    timeout: int,
    job_id: str | None = None,
) -> dict[str, Any]:
    _sync_local_video_runtime(globals(), {"queue_and_wait"})
    raise_if_cancelled(job_id)
    response = requests.post(
        f"{COMFY_URL}/prompt",
        json={"prompt": workflow, "client_id": str(uuid4())},
        timeout=60,
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"ComfyUI rejected {stage_name}: {response.status_code} {response.text[:2000]}"
        )
    prompt_id = response.json()["prompt_id"]
    if job_id:
        update_job(job_id, active_prompt_id=prompt_id)
    LOGGER.info("%s queued as ComfyUI prompt %s", stage_name, prompt_id)
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            if job_id and cancel_requested(job_id):
                interrupt_comfy_prompt(prompt_id)
                raise JobCancelled("Generation was cancelled by the user.")
            history_response = requests.get(
                f"{COMFY_URL}/history/{prompt_id}", timeout=30
            )
            history_response.raise_for_status()
            history = history_response.json()
            if prompt_id in history:
                entry = history[prompt_id]
                current_status = entry.get("status", {})
                if current_status.get("completed"):
                    LOGGER.info("%s completed", stage_name)
                    return entry
                messages = current_status.get("messages", [])
                terminal_errors = {"execution_error", "execution_interrupted"}
                if any(
                    message and message[0] in terminal_errors for message in messages
                ):
                    if job_id and cancel_requested(job_id):
                        raise JobCancelled("Generation was cancelled by the user.")
                    raise RuntimeError(
                        f"ComfyUI {stage_name} execution failed: {json.dumps(messages, ensure_ascii=False)[:4000]}"
                    )
            time.sleep(5)
        raise TimeoutError(f"ComfyUI {stage_name} timed out after {timeout} seconds.")
    finally:
        if job_id:
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                if job and job.get("active_prompt_id") == prompt_id:
                    job["active_prompt_id"] = None
                    job["updated_at"] = utc_now()
                    snapshot = dict(job)
                else:
                    snapshot = None
            if snapshot is not None:
                write_json_atomic(status_path(job_id), snapshot)


def output_files(entry: dict[str, Any]) -> list[Path]:
    _sync_local_video_runtime(globals(), {"output_files"})
    found: list[Path] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if "filename" in value and value.get("type", "output") == "output":
                found.append(
                    COMFY_OUTPUT / value.get("subfolder", "") / value["filename"]
                )
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(entry.get("outputs", {}))
    return list(dict.fromkeys(found))


def first_output(entry: dict[str, Any], suffixes: set[str], stage_name: str) -> Path:
    _sync_local_video_runtime(globals(), {"first_output"})
    matches = [
        path
        for path in output_files(entry)
        if path.suffix.lower() in suffixes and path.is_file()
    ]
    if not matches:
        raise RuntimeError(f"{stage_name} completed without an expected output file.")
    return matches[0]


def prepare_landscape_keyframe(
    source: Path,
    target: Path,
    size: tuple[int, int],
    background_rgb: tuple[int, int, int] = (255, 255, 255),
) -> None:
    _sync_local_video_runtime(globals(), {"prepare_landscape_keyframe"})
    with Image.open(source) as loaded:
        image = loaded.convert("RGBA")
    background_rgba = (*background_rgb, 255)
    flattened = Image.new("RGBA", image.size, background_rgba)
    flattened.alpha_composite(image)
    rgb = flattened.convert("RGB")
    if background_rgb == (0, 255, 0):
        red, green, blue = rgb.split()
        other_channels = ImageChops.lighter(red, blue)
        green_dominance = ImageChops.subtract(green, other_channels)
        strongly_green = green_dominance.point(lambda value: 255 if value > 40 else 0)
        bright_enough = green.point(lambda value: 255 if value > 80 else 0)
        green_background = ImageChops.multiply(strongly_green, bright_enough)
        foreground = ImageChops.invert(green_background)
    else:
        difference = ImageChops.difference(
            rgb, Image.new("RGB", rgb.size, background_rgb)
        ).convert("L")
        foreground = difference.point(lambda value: 255 if value > 12 else 0)
    bbox = foreground.getbbox()
    if bbox:
        padding = max(16, int(max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * 0.06))
        bbox = (
            max(0, bbox[0] - padding),
            max(0, bbox[1] - padding),
            min(image.width, bbox[2] + padding),
            min(image.height, bbox[3] + padding),
        )
        image = image.crop(bbox)
    canvas = Image.new("RGBA", size, background_rgba)
    image.thumbnail(
        (int(size[0] * 0.88), int(size[1] * 0.90)), Image.Resampling.LANCZOS
    )
    canvas.alpha_composite(
        image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2)
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(target, format="PNG", optimize=True)


def composite_rgba_reference_on_background(
    source: Path,
    target: Path,
    background_rgb: tuple[int, int, int],
) -> None:
    """Derive an opaque working image from a final RGBA reference one-way."""
    _sync_local_video_runtime(globals(), {"composite_rgba_reference_on_background"})

    with Image.open(source) as loaded:
        if "A" not in loaded.getbands():
            raise ValueError("transparent_reference_has_no_alpha")
        image = loaded.convert("RGBA")
    background = Image.new("RGBA", image.size, (*background_rgb, 255))
    background.alpha_composite(image)
    target.parent.mkdir(parents=True, exist_ok=True)
    background.convert("RGB").save(target, format="PNG", optimize=True)


def prepare_canonical_action_identity_refine_canvas(
    source: Path,
    target: Path,
    *,
    action: str,
    policy: str,
) -> dict[str, Any]:
    """Composite a normalized action cutout unchanged onto pure green."""
    _sync_local_video_runtime(globals(), {"prepare_canonical_action_identity_refine_canvas"})

    action_name = action_key(action)
    if action_name not in {"eating", "grooming"}:
        raise ValueError("canonical identity refine action must be eating or grooming")
    if not str(policy).strip():
        raise ValueError("canonical identity refine policy is required")

    with Image.open(source) as loaded:
        if "A" not in loaded.getbands():
            raise ValueError(
                f"{action_name} identity refine source has no alpha channel"
            )
        source_rgba_image = loaded.convert("RGBA")
    if source_rgba_image.size != (512, 512):
        raise ValueError(f"{action_name} identity refine source must be 512x512")
    source_rgba = np.asarray(source_rgba_image, dtype=np.uint8)
    source_foreground = source_rgba[:, :, 3] >= 64
    foreground_pixels = int(np.count_nonzero(source_foreground))
    if not 4096 <= foreground_pixels <= int(512 * 512 * 0.90):
        raise RuntimeError(
            f"{action_name} identity refine source has an invalid foreground mask."
        )
    ys, xs = np.nonzero(source_foreground)
    foreground_bbox = [
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    ]
    composite = Image.new("RGBA", (512, 512), (0, 255, 0, 255))
    composite.alpha_composite(source_rgba_image)
    output_rgb = np.asarray(composite.convert("RGB"), dtype=np.uint8)
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output_rgb, mode="RGB").save(
        target,
        format="PNG",
        optimize=True,
    )
    with Image.open(target) as loaded:
        persisted_rgb = np.asarray(loaded.convert("RGB"), dtype=np.uint8)
    if not np.array_equal(persisted_rgb, output_rgb):
        raise RuntimeError(
            f"{action_name} identity refine canvas changed during persistence."
        )
    transparent_background = source_rgba[:, :, 3] == 0
    transparent_background_is_green = bool(
        np.all(
            persisted_rgb[transparent_background]
            == np.asarray((0, 255, 0), dtype=np.uint8)
        )
    )
    if not transparent_background_is_green:
        raise RuntimeError(
            f"{action_name} identity refine canvas did not preserve pure-green background."
        )
    return {
        "policy": str(policy),
        "action": action_name,
        "source_path": str(source),
        "source_sha256": file_sha256(source),
        "source_foreground_pixels": foreground_pixels,
        "source_foreground_bbox": foreground_bbox,
        "source_rgba_pixels_modified_before_composite": False,
        "transparent_background_composited_to_pure_green": True,
        "sampler_canvas_scope": "full_frame_identity_repaint_pose_preserved",
        "target_path": str(target),
        "target_sha256": file_sha256(target),
    }


def prepare_advisory_grooming_identity_refine_canvas(
    source: Path,
    target: Path,
    noise_mask_target: Path,
    *,
    identity_reference_palette: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Lock the accepted action and expose only its upper head to Qwen."""
    _sync_local_video_runtime(globals(), {"prepare_advisory_grooming_identity_refine_canvas"})

    with Image.open(source) as loaded:
        if "A" not in loaded.getbands():
            raise ValueError("Grooming identity-refine source has no alpha channel")
        source_image = loaded.convert("RGBA")
    if source_image.size != (512, 512):
        raise ValueError("Grooming identity-refine source must be 512x512")
    original_source_rgba = np.asarray(source_image, dtype=np.uint8)
    source_rgba = original_source_rgba.copy()
    foreground = source_rgba[:, :, 3] >= 64
    foreground_pixels_before = int(np.count_nonzero(foreground))
    reference_fractions = (
        identity_reference_palette.get("fractions")
        if isinstance(identity_reference_palette, dict)
        and isinstance(identity_reference_palette.get("fractions"), dict)
        else {}
    )
    reference_black_fraction = (
        float(reference_fractions.get("black"))
        if "black" in reference_fractions
        else None
    )
    artifact_removed = np.zeros(foreground.shape, dtype=bool)
    dark_component_min_pixels = max(
        2048,
        int(
            round(
                foreground_pixels_before
                * GROOMING_ADVISORY_IDENTITY_REFINE_DARK_ARTIFACT_MIN_FRACTION
            )
        ),
    )
    if (
        reference_black_fraction is not None
        and reference_black_fraction
        <= GROOMING_ADVISORY_IDENTITY_REFINE_SOURCE_BLACK_FRACTION_MAX
    ):
        source_value = source_rgba[:, :, :3].max(axis=2)
        dark_pixels = foreground & (
            source_value
            < GROOMING_ADVISORY_IDENTITY_REFINE_DARK_ARTIFACT_VALUE_MAX
        )
        component_count, component_labels, component_stats, _ = (
            cv2.connectedComponentsWithStats(
                dark_pixels.astype(np.uint8),
                connectivity=8,
            )
        )
        artifact_core = np.zeros(foreground.shape, dtype=bool)
        for component_index in range(1, component_count):
            component_pixels = int(
                component_stats[component_index, cv2.CC_STAT_AREA]
            )
            if component_pixels >= dark_component_min_pixels:
                artifact_core |= component_labels == component_index
        if np.any(artifact_core):
            subject_support_px = int(
                GROOMING_ADVISORY_IDENTITY_REFINE_DARK_ARTIFACT_SUBJECT_SUPPORT_PX
            )
            subject_support_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (subject_support_px * 2 + 1, subject_support_px * 2 + 1),
            )
            non_dark_subject = foreground & (
                source_value
                >= GROOMING_ADVISORY_IDENTITY_REFINE_DARK_ARTIFACT_VALUE_MAX
            )
            subject_support = cv2.dilate(
                non_dark_subject.astype(np.uint8),
                subject_support_kernel,
                iterations=1,
            ).astype(bool)
            artifact_core_to_remove = artifact_core & ~subject_support
            edge_dilation = int(
                GROOMING_ADVISORY_IDENTITY_REFINE_DARK_ARTIFACT_EDGE_DILATION_PX
            )
            edge_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (edge_dilation * 2 + 1, edge_dilation * 2 + 1),
            )
            artifact_edge = cv2.dilate(
                artifact_core_to_remove.astype(np.uint8),
                edge_kernel,
                iterations=1,
            ).astype(bool)
            artifact_removed = artifact_core_to_remove | (
                artifact_edge
                & foreground
                & ~subject_support
                & (
                    source_value
                    < GROOMING_ADVISORY_IDENTITY_REFINE_DARK_ARTIFACT_EDGE_VALUE_MAX
                )
            )
            source_rgba[artifact_removed] = np.asarray(
                (0, 0, 0, 0),
                dtype=np.uint8,
            )
            foreground = source_rgba[:, :, 3] >= 64
    sanitized_source = target.with_name(f"{target.stem}_sanitized_cutout.png")
    sanitized_source.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(source_rgba, mode="RGBA").save(
        sanitized_source,
        format="PNG",
        optimize=True,
    )
    audit = prepare_canonical_action_identity_refine_canvas(
        sanitized_source,
        target,
        action="grooming",
        policy=GROOMING_ADVISORY_IDENTITY_REFINE_POLICY,
    )
    ys, xs = np.nonzero(foreground)
    x0, y0 = int(xs.min()), int(ys.min())
    x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1
    head_cutoff_y = min(
        y1 - 1,
        y0
        + int(
            round(
                (y1 - y0)
                * GROOMING_ADVISORY_IDENTITY_REFINE_HEAD_HEIGHT_FRACTION
            )
        ),
    )
    row_grid = np.indices(foreground.shape)[0]
    head_core = foreground & (row_grid <= head_cutoff_y)
    head_core_pixels = int(np.count_nonzero(head_core))
    if head_core_pixels < 1024:
        raise RuntimeError(
            "Grooming identity-refine head mask contains too few source pixels."
        )
    expansion_px = int(GROOMING_ADVISORY_IDENTITY_REFINE_HEAD_EXPANSION_PX)
    if expansion_px:
        raise RuntimeError(
            "Grooming identity-refine must not expand the accepted head silhouette."
        )
    head_edit_support = head_core.copy()
    head_edit_support[row_grid > head_cutoff_y] = False
    feather_sigma = float(
        GROOMING_ADVISORY_IDENTITY_REFINE_MASK_FEATHER_SIGMA_PX
    )
    edit_strength = cv2.GaussianBlur(
        head_edit_support.astype(np.float32),
        (0, 0),
        sigmaX=feather_sigma,
        sigmaY=feather_sigma,
    ).clip(0.0, 1.0)
    # Never expose green background or the accepted silhouette boundary. The
    # repair may correct internal face detail only; pose, head outline, ears,
    # cheeks, mouth-contact paw and every body pixel stay source-locked.
    edit_strength *= foreground.astype(np.float32)
    edit_strength[row_grid > head_cutoff_y] = 0.0
    edit_strength[edit_strength < 0.01] = 0.0
    edit_strength[edit_strength > 0.99] = 1.0
    editable_pixels = int(np.count_nonzero(edit_strength > 0.0))
    full_strength_pixels = int(np.count_nonzero(edit_strength >= 1.0))
    protected_body = foreground & (row_grid > head_cutoff_y)
    protected_body_pixels = int(np.count_nonzero(protected_body))
    if editable_pixels < 1024 or full_strength_pixels < 512:
        raise RuntimeError(
            "Grooming identity-refine noise mask lacks a usable head-edit core."
        )
    if protected_body_pixels < 2048:
        raise RuntimeError(
            "Grooming identity-refine mask did not preserve enough action pixels."
        )

    noise_mask_alpha = np.rint((1.0 - edit_strength) * 255.0).astype(np.uint8)
    noise_mask_rgba = np.zeros((512, 512, 4), dtype=np.uint8)
    noise_mask_rgba[:, :, 3] = noise_mask_alpha
    noise_mask_target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(noise_mask_rgba, mode="RGBA").save(
        noise_mask_target,
        format="PNG",
        optimize=True,
    )
    return {
        **audit,
        "original_source_path": str(source),
        "original_source_sha256": file_sha256(source),
        "sanitized_source_path": str(sanitized_source),
        "sanitized_source_sha256": file_sha256(sanitized_source),
        "source_artifact_cleanup_policy": (
            "unexpected_large_near_black_component_removal_v1"
        ),
        "identity_reference_black_fraction": reference_black_fraction,
        "identity_reference_black_fraction_max_for_cleanup": (
            GROOMING_ADVISORY_IDENTITY_REFINE_SOURCE_BLACK_FRACTION_MAX
        ),
        "dark_artifact_component_min_pixels": dark_component_min_pixels,
        "dark_artifact_subject_support_dilation_px": (
            GROOMING_ADVISORY_IDENTITY_REFINE_DARK_ARTIFACT_SUBJECT_SUPPORT_PX
        ),
        "dark_artifact_removed_pixels": int(np.count_nonzero(artifact_removed)),
        "foreground_pixels_before_cleanup": foreground_pixels_before,
        "foreground_pixels_after_cleanup": int(np.count_nonzero(foreground)),
        "sampler_canvas_scope": "upper_head_mask_only_action_pixels_locked",
        "head_height_fraction": (
            GROOMING_ADVISORY_IDENTITY_REFINE_HEAD_HEIGHT_FRACTION
        ),
        "head_cutoff_y": head_cutoff_y,
        "head_core_pixels": head_core_pixels,
        "head_silhouette_expansion_px": expansion_px,
        "head_edit_support_pixels": int(np.count_nonzero(head_edit_support)),
        "editable_background_pixels": int(
            np.count_nonzero((edit_strength > 0.0) & ~foreground)
        ),
        "editable_pixels": editable_pixels,
        "full_strength_pixels": full_strength_pixels,
        "protected_body_pixels": protected_body_pixels,
        "raised_paw_editing_forbidden": True,
        "body_pose_editing_forbidden": True,
        "noise_mask_policy": "source_foreground_upper_head_interior_feathered_v2",
        "noise_mask_feather_sigma_px": feather_sigma,
        "noise_mask_loadimage_contract": "mask_equals_one_minus_alpha",
        "noise_mask_target_path": str(noise_mask_target),
        "noise_mask_target_sha256": file_sha256(noise_mask_target),
        "edit_strength_sha256": hashlib.sha256(
            edit_strength.tobytes()
        ).hexdigest(),
    }


def finalize_advisory_grooming_identity_refine_candidate(
    source: Path,
    generated: Path,
    noise_mask: Path,
    target: Path,
) -> dict[str, Any]:
    """Composite only the refined face over the accepted action draft."""
    _sync_local_video_runtime(globals(), {"finalize_advisory_grooming_identity_refine_candidate"})

    with Image.open(source) as loaded:
        if "A" not in loaded.getbands():
            raise ValueError("Grooming identity-refine source has no alpha channel")
        source_image = loaded.convert("RGBA")
    if source_image.size != (512, 512):
        raise ValueError("Grooming identity-refine source must be 512x512")
    with Image.open(generated) as loaded:
        generated_image = loaded.convert("RGB")
    generated_original_size = list(generated_image.size)
    if generated_image.size != source_image.size:
        generated_image = generated_image.resize(
            source_image.size,
            Image.Resampling.LANCZOS,
        )
    with Image.open(noise_mask) as loaded:
        if "A" not in loaded.getbands():
            raise ValueError("Grooming identity-refine noise mask has no alpha")
        noise_mask_image = loaded.convert("RGBA")
    if noise_mask_image.size != source_image.size:
        raise ValueError("Grooming identity-refine noise mask must be 512x512")

    source_rgba = np.asarray(source_image, dtype=np.uint8)
    generated_rgb = np.asarray(generated_image, dtype=np.uint8)
    mask_rgba = np.asarray(noise_mask_image, dtype=np.uint8)
    edit_strength = 1.0 - mask_rgba[:, :, 3].astype(np.float32) / 255.0
    foreground = source_rgba[:, :, 3] >= 64
    requested_editable = edit_strength > 0.0
    generated_green = (
        (generated_rgb[:, :, 0] < 96)
        & (generated_rgb[:, :, 1] > 150)
        & (generated_rgb[:, :, 2] < 112)
        & (
            generated_rgb[:, :, 1].astype(np.int16)
            - generated_rgb[:, :, 0].astype(np.int16)
            > 64
        )
        & (
            generated_rgb[:, :, 1].astype(np.int16)
            - generated_rgb[:, :, 2].astype(np.int16)
            > 48
        )
    )
    # Never punch a green hole into the accepted head if the generative pass
    # shifts the silhouette despite the mask.
    generated_green_skipped = generated_green & requested_editable
    edit_strength[generated_green] = 0.0

    source_composite = Image.new("RGBA", (512, 512), (0, 255, 0, 255))
    source_composite.alpha_composite(source_image)
    source_rgb = np.asarray(source_composite.convert("RGB"), dtype=np.uint8)
    weights = edit_strength[:, :, None]
    output_rgb = np.rint(
        source_rgb.astype(np.float32) * (1.0 - weights)
        + generated_rgb.astype(np.float32) * weights
    ).clip(0.0, 255.0).astype(np.uint8)
    protected = edit_strength <= 0.0
    protected_changed_pixels = int(
        np.count_nonzero(np.any(output_rgb[protected] != source_rgb[protected], axis=1))
    )
    if protected_changed_pixels:
        raise RuntimeError(
            "Grooming identity refinement changed protected action pixels."
        )
    transparent_background = source_rgba[:, :, 3] == 0
    protected_background = transparent_background & (edit_strength <= 0.0)
    if not np.all(
        output_rgb[protected_background]
        == np.asarray((0, 255, 0), dtype=np.uint8)
    ):
        raise RuntimeError(
            "Grooming identity refinement changed protected green background."
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output_rgb, mode="RGB").save(
        target,
        format="PNG",
        optimize=True,
    )
    return {
        "policy": GROOMING_ADVISORY_IDENTITY_REFINE_POLICY,
        "source_path": str(source),
        "source_sha256": file_sha256(source),
        "generated_path": str(generated),
        "generated_sha256": file_sha256(generated),
        "generated_original_size": generated_original_size,
        "noise_mask_path": str(noise_mask),
        "noise_mask_sha256": file_sha256(noise_mask),
        "editable_pixels": int(np.count_nonzero(edit_strength > 0.0)),
        "head_silhouette_expansion_pixels": int(
            np.count_nonzero(transparent_background & (edit_strength > 0.0))
        ),
        "green_generated_pixels_skipped": int(
            np.count_nonzero(generated_green_skipped)
        ),
        "protected_pixels": int(np.count_nonzero(protected)),
        "protected_background_pixels": int(np.count_nonzero(protected_background)),
        "protected_action_pixels_changed": protected_changed_pixels,
        "body_and_raised_paw_pixel_lock_passed": True,
        "background_pixel_lock_passed": True,
        "target_path": str(target),
        "target_sha256": file_sha256(target),
    }


def prepare_canonical_grooming_full_pose_refine_canvas(
    source: Path,
    target: Path,
) -> dict[str, Any]:
    """Prepare the grooming pose-bearing sampler for canonical repaint."""
    _sync_local_video_runtime(globals(), {"prepare_canonical_grooming_full_pose_refine_canvas"})

    return prepare_canonical_action_identity_refine_canvas(
        source,
        target,
        action="grooming",
        policy=GROOMING_CANONICAL_FULL_POSE_REFINE_POLICY,
    )


def _largest_green_or_alpha_foreground(
    rgba: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Return the largest subject connected to neither transparent nor green border."""
    _sync_local_video_runtime(globals(), {"_largest_green_or_alpha_foreground"})

    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("foreground analysis requires an RGBA array")
    alpha = rgba[:, :, 3]
    foreground: np.ndarray | None = None
    method = "unavailable"
    if np.any(alpha < 250):
        alpha_foreground = alpha >= 64
        alpha_fraction = float(np.count_nonzero(alpha_foreground)) / float(
            alpha_foreground.size
        )
        if 0.02 <= alpha_fraction <= 0.95:
            foreground = alpha_foreground
            method = "alpha"
    if foreground is None:
        rgb = rgba[:, :, :3]
        red = rgb[:, :, 0].astype(np.int16)
        green = rgb[:, :, 1].astype(np.int16)
        blue = rgb[:, :, 2].astype(np.int16)
        green_candidate = (
            (green >= 90)
            & ((green - red) >= 45)
            & ((green - blue) >= 45)
            & (green >= np.maximum(red, blue) * 1.15)
        )
        background_count, background_labels = cv2.connectedComponents(
            green_candidate.astype(np.uint8),
            connectivity=8,
        )
        border_labels = np.unique(
            np.concatenate(
                [
                    background_labels[0, :],
                    background_labels[-1, :],
                    background_labels[:, 0],
                    background_labels[:, -1],
                ]
            )
        )
        border_labels = border_labels[
            (border_labels > 0) & (border_labels < background_count)
        ]
        if len(border_labels):
            background = np.isin(background_labels, border_labels)
            background_fraction = float(np.count_nonzero(background)) / float(
                background.size
            )
            if background_fraction >= 0.05:
                foreground = ~background
                method = "border_connected_green_screen"
    if foreground is None:
        raise RuntimeError("subject foreground could not be isolated")
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        foreground.astype(np.uint8),
        connectivity=8,
    )
    if component_count <= 1:
        raise RuntimeError("subject foreground is empty")
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    largest = labels == largest_label
    foreground_fraction = float(np.count_nonzero(largest)) / float(largest.size)
    if not 0.02 <= foreground_fraction <= 0.90:
        raise RuntimeError("subject foreground fraction is outside safe bounds")
    return largest, method


def _identity_free_grooming_paw_depth(
    paw_mask: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Build smooth paw volume from silhouette distance, never source luma."""
    _sync_local_video_runtime(globals(), {"_identity_free_grooming_paw_depth"})

    paw_mask = np.asarray(paw_mask, dtype=bool)
    if paw_mask.ndim != 2 or int(np.count_nonzero(paw_mask)) < 64:
        raise ValueError("Grooming identity-free paw mask is too small.")
    distance = cv2.distanceTransform(
        paw_mask.astype(np.uint8),
        cv2.DIST_L2,
        5,
    )
    distance_p95 = max(
        float(np.percentile(distance[paw_mask], 95.0)),
        1.0,
    )
    normalized = np.sqrt(np.clip(distance / distance_p95, 0.0, 1.0))
    # Keep the entire paw in a narrow neutral midtone range. Its silhouette and
    # smooth radial volume remain usable geometry, while no white-maquette eye,
    # mouth, toe crease, dark patch, or lighting trough can become pet markings.
    depth = 0.48 + normalized * 0.12
    return depth.astype(np.float32), distance_p95


def prepare_grooming_white_maquette_frame0_pose_inputs(
    driving_animation: Path,
    action_cutout_target: Path,
    geometry_guide_target: Path,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    """Decode frame 0 and retain body pose while erasing maquette face identity."""
    _sync_local_video_runtime(globals(), {"prepare_grooming_white_maquette_frame0_pose_inputs"})

    expected_sha256 = str(expected_sha256 or "").strip().lower()
    if (
        not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
        or not driving_animation.is_file()
        or file_sha256(driving_animation) != expected_sha256
    ):
        raise RuntimeError(
            "Grooming white-maquette driving animation is missing or its digest changed."
        )
    capture = cv2.VideoCapture(str(driving_animation))
    try:
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        readable, frame_bgr = capture.read()
    finally:
        capture.release()
    if not readable or frame_bgr is None:
        raise RuntimeError("Grooming white-maquette frame 0 could not be decoded.")
    if frame_bgr.shape[:2] != (512, 512):
        raise RuntimeError("Grooming white-maquette frame 0 must be 512x512.")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgba = np.dstack(
        [frame_rgb, np.full((512, 512), 255, dtype=np.uint8)]
    )
    subject_mask, mask_method = _largest_green_or_alpha_foreground(frame_rgba)
    subject_pixels = int(np.count_nonzero(subject_mask))
    if subject_pixels < 4096:
        raise RuntimeError("Grooming white-maquette frame 0 subject is too small.")
    ys, xs = np.nonzero(subject_mask)
    x0, y0 = int(xs.min()), int(ys.min())
    x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1

    action_rgba = np.zeros((512, 512, 4), dtype=np.uint8)
    action_rgba[subject_mask, :3] = frame_rgb[subject_mask]
    action_rgba[subject_mask, 3] = 255
    action_cutout_target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(action_rgba, mode="RGBA").save(
        action_cutout_target,
        format="PNG",
        optimize=True,
    )

    frame_luma = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    luma_values = frame_luma[subject_mask]
    luma_fill = float(np.median(luma_values))
    filled_luma = np.where(subject_mask, frame_luma, luma_fill).astype(np.float32)
    low_frequency_luma = cv2.GaussianBlur(
        filled_luma,
        (0, 0),
        sigmaX=3.5,
        sigmaY=3.5,
    )
    low_values = low_frequency_luma[subject_mask]
    luma_p05 = float(np.percentile(low_values, 5.0))
    luma_p95 = float(np.percentile(low_values, 95.0))
    luma_span = luma_p95 - luma_p05
    if luma_span < 4.0:
        raise RuntimeError("Grooming white-maquette frame 0 lacks body-depth variation.")
    normalized_luma = np.clip(
        (low_frequency_luma - luma_p05) / luma_span,
        0.0,
        1.0,
    )
    subject_distance = cv2.distanceTransform(
        subject_mask.astype(np.uint8),
        cv2.DIST_L2,
        5,
    )
    distance_p95 = max(
        float(np.percentile(subject_distance[subject_mask], 95.0)),
        1.0,
    )
    normalized_distance = np.clip(subject_distance / distance_p95, 0.0, 1.0)
    anatomy_depth = normalized_luma * 0.82 + normalized_distance * 0.18
    head_cutoff = y0 + int(round((y1 - y0) * 0.45))
    row_grid = np.indices(subject_mask.shape)[0]
    head_mask = subject_mask & (row_grid <= head_cutoff)
    head_pixels = int(np.count_nonzero(head_mask))
    if head_pixels < 1024:
        raise RuntimeError("Grooming white-maquette frame 0 head region is too small.")
    subject_width = float(x1 - x0)
    subject_height = float(y1 - y0)
    head_identity_core = np.zeros(subject_mask.shape, dtype=np.uint8)
    cv2.ellipse(
        head_identity_core,
        (
            int(round(x0 + subject_width * 0.50)),
            int(round(y0 + subject_height * 0.28)),
        ),
        (
            max(16, int(round(subject_width * 0.40))),
            max(20, int(round(subject_height * 0.28))),
        ),
        0.0,
        0.0,
        360.0,
        255,
        -1,
    )
    head_identity_core = (
        (head_identity_core > 0) & head_mask & subject_mask
    )
    head_identity_core_pixels = int(np.count_nonzero(head_identity_core))
    if head_identity_core_pixels < 1024:
        raise RuntimeError(
            "Grooming white-maquette frame 0 identity-removal core is too small."
        )
    head_distance = cv2.distanceTransform(
        head_identity_core.astype(np.uint8),
        cv2.DIST_L2,
        5,
    )
    head_distance_p95 = max(
        float(np.percentile(head_distance[head_identity_core], 95.0)),
        1.0,
    )
    identity_free_head_depth = 0.44 + 0.12 * np.clip(
        head_distance / head_distance_p95,
        0.0,
        1.0,
    )
    head_weight = cv2.GaussianBlur(
        head_identity_core.astype(np.float32),
        (0, 0),
        sigmaX=12.0,
        sigmaY=12.0,
    )
    head_weight = np.clip(head_weight, 0.0, 1.0)
    head_weight *= subject_mask.astype(np.float32)
    anatomy_depth = (
        anatomy_depth * (1.0 - head_weight)
        + identity_free_head_depth * head_weight
    )
    raised_paw_volume = np.zeros(subject_mask.shape, dtype=np.uint8)
    cv2.ellipse(
        raised_paw_volume,
        (
            int(round(x0 + subject_width * 0.22)),
            int(round(y0 + subject_height * 0.41)),
        ),
        (
            max(8, int(round(subject_width * 0.18))),
            max(12, int(round(subject_height * 0.18))),
        ),
        0.0,
        0.0,
        360.0,
        255,
        -1,
    )
    # Do not clip this volume at ``head_cutoff``.  That former hard horizontal
    # boundary was visible in Image 2 and could be interpreted by Qwen as a
    # dark chest seam.  The ellipse already scopes the lifted foreleg; the
    # true subject silhouette is the only safe geometric boundary here.
    raised_paw_volume = (raised_paw_volume > 0) & subject_mask
    identity_free_paw_depth, raised_paw_distance_p95 = (
        _identity_free_grooming_paw_depth(raised_paw_volume)
    )
    raised_paw_weight = cv2.GaussianBlur(
        raised_paw_volume.astype(np.float32),
        (0, 0),
        sigmaX=6.0,
        sigmaY=6.0,
    )
    raised_paw_weight = np.clip(raised_paw_weight, 0.0, 1.0)
    raised_paw_weight *= subject_mask.astype(np.float32)
    anatomy_depth = (
        anatomy_depth * (1.0 - raised_paw_weight)
        + identity_free_paw_depth * raised_paw_weight
    )
    guide_luma = np.rint(96.0 + anatomy_depth * 80.0).astype(np.uint8)
    guide_rgb = np.empty((512, 512, 3), dtype=np.uint8)
    guide_rgb[:, :] = (0, 255, 0)
    for channel in range(3):
        guide_rgb[:, :, channel][subject_mask] = guide_luma[subject_mask]
    geometry_guide_target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(guide_rgb, mode="RGB").save(
        geometry_guide_target,
        format="PNG",
        optimize=True,
    )
    guide_foreground = guide_rgb[subject_mask]
    guide_neutral_fraction = float(
        np.count_nonzero(np.ptp(guide_foreground.astype(np.int16), axis=1) == 0)
    ) / float(subject_pixels)
    if guide_neutral_fraction < 0.999:
        raise RuntimeError("Grooming frame 0 pose guide retained source chroma.")
    return {
        "policy": "white_maquette_frame000_body_pose_identity_free_paw_depth_v3",
        "driving_animation_path": str(driving_animation),
        "driving_animation_sha256": expected_sha256,
        "decoded_frame_index": 0,
        "decoded_frame_count": frame_count,
        "decoded_fps": round(fps, 6),
        "decoded_size": [512, 512],
        "subject_mask_method": mask_method,
        "subject_foreground_pixels": subject_pixels,
        "subject_foreground_bbox": [x0, y0, x1, y1],
        "head_region_cutoff_y": head_cutoff,
        "head_region_pixels": head_pixels,
        "head_identity_removal_core_pixels": head_identity_core_pixels,
        "head_identity_signal_removed": True,
        "head_identity_removal_method": (
            "silhouette_bounded_elliptical_distance_volume_with_radial_feather_"
            "and_identity_free_raised_paw_distance_volume"
        ),
        "head_identity_radial_feather_sigma_px": 12.0,
        "raised_paw_action_volume_preserved": True,
        "raised_paw_action_volume_source_luma_preserved": False,
        "raised_paw_action_volume_source_rgb_preserved": False,
        "raised_paw_action_volume_depth_policy": (
            "silhouette_distance_neutral_midtone_v1"
        ),
        "raised_paw_action_volume_depth_range": [0.48, 0.60],
        "raised_paw_action_volume_distance_p95": round(
            raised_paw_distance_p95,
            6,
        ),
        "raised_paw_action_volume_weight_blur_sigma_px": 6.0,
        "raised_paw_action_volume_hard_head_cutoff_applied": False,
        "raw_frame_loaded_by_qwen": False,
        "raw_frame_rgb_conditioning_exposed": False,
        "raw_frame_chroma_conditioning_exposed": False,
        "raw_frame_exact_luma_conditioning_exposed": False,
        "raw_maquette_face_conditioning_exposed": False,
        "body_low_frequency_geometry_retained": True,
        "body_luma_p05": round(luma_p05, 6),
        "body_luma_p95": round(luma_p95, 6),
        "body_luma_span": round(luma_span, 6),
        "guide_neutral_foreground_fraction": round(guide_neutral_fraction, 6),
        "action_cutout_path": str(action_cutout_target),
        "action_cutout_sha256": file_sha256(action_cutout_target),
        "action_cutout_qwen_visible": False,
        "geometry_guide_path": str(geometry_guide_target),
        "geometry_guide_sha256": file_sha256(geometry_guide_target),
        "geometry_guide_qwen_visible": True,
    }


def prepare_canonical_grooming_face_identity_reference(
    canonical_reference: Path,
    target: Path,
) -> dict[str, Any]:
    """Create an enlarged face-only reference from the already approved canonical."""
    _sync_local_video_runtime(globals(), {"prepare_canonical_grooming_face_identity_reference"})

    with Image.open(canonical_reference) as loaded:
        canonical = np.asarray(loaded.convert("RGBA"), dtype=np.uint8)
    foreground, mask_method = _largest_green_or_alpha_foreground(canonical)
    ys, xs = np.nonzero(foreground)
    x0, y0 = int(xs.min()), int(ys.min())
    x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1
    head_cutoff = y0 + int(round((y1 - y0) * 0.47))
    row_grid = np.indices(foreground.shape)[0]
    head_mask = foreground & (row_grid <= head_cutoff)
    head_ys, head_xs = np.nonzero(head_mask)
    if len(head_xs) < 512:
        raise RuntimeError("Canonical grooming face reference is too small.")
    padding = max(10, int(round(max(x1 - x0, y1 - y0) * 0.035)))
    crop_box = (
        max(0, int(head_xs.min()) - padding),
        max(0, int(head_ys.min()) - padding),
        min(canonical.shape[1], int(head_xs.max()) + 1 + padding),
        min(canonical.shape[0], int(head_ys.max()) + 1 + padding),
    )
    crop_rgba = canonical[crop_box[1] : crop_box[3], crop_box[0] : crop_box[2]].copy()
    crop_mask = head_mask[
        crop_box[1] : crop_box[3], crop_box[0] : crop_box[2]
    ]
    crop_rgba[:, :, 3] = np.where(crop_mask, 255, 0).astype(np.uint8)
    crop = Image.fromarray(crop_rgba, mode="RGBA")
    scale = min(420.0 / max(1, crop.width), 420.0 / max(1, crop.height))
    resized_size = (
        max(1, int(round(crop.width * scale))),
        max(1, int(round(crop.height * scale))),
    )
    crop = crop.resize(resized_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (512, 512), (0, 255, 0))
    paste_x = (512 - crop.width) // 2
    paste_y = max(18, (512 - crop.height) // 2)
    canvas.paste(crop.convert("RGB"), (paste_x, paste_y), crop.getchannel("A"))
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, format="PNG", optimize=True)
    return {
        "policy": "canonical_same_image_enlarged_face_identity_reference_v1",
        "canonical_reference_path": str(canonical_reference),
        "canonical_reference_sha256": file_sha256(canonical_reference),
        "canonical_mask_method": mask_method,
        "canonical_foreground_bbox": [x0, y0, x1, y1],
        "head_cutoff_y": head_cutoff,
        "source_crop_box": list(crop_box),
        "source_head_pixels": int(np.count_nonzero(head_mask)),
        "derived_from_same_canonical_identity": True,
        "adds_second_pet_identity": False,
        "pose_authority": False,
        "face_detail_authority": True,
        "target_path": str(target),
        "target_sha256": file_sha256(target),
    }


def prepare_advisory_grooming_local_action_edit_mask(
    canonical_reference: Path,
    target: Path,
) -> dict[str, Any]:
    """Let Qwen edit only the canonical viewer-left foreleg action region.

    Opaque alpha protects the canonical latent.  A curved, feathered transparent
    corridor exposes the existing viewer-left planted foreleg, its shoulder path,
    and the small paw-to-lower-muzzle contact area.  No video frame, action image,
    or foreign animal geometry contributes to this mask.
    """
    _sync_local_video_runtime(globals(), {"prepare_advisory_grooming_local_action_edit_mask"})

    with Image.open(canonical_reference) as loaded:
        canonical = np.asarray(loaded.convert("RGBA"), dtype=np.uint8)
    if canonical.shape != (512, 512, 4):
        raise ValueError("Grooming action head-lock canonical must be 512x512.")
    foreground, mask_method = _largest_green_or_alpha_foreground(canonical)
    ys, xs = np.nonzero(foreground)
    x0, y0 = int(xs.min()), int(ys.min())
    x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1
    width = float(x1 - x0)
    height = float(y1 - y0)
    shoulder = (
        int(round(x0 + width * 0.37)),
        int(round(y0 + height * 0.69)),
    )
    elbow = (
        int(round(x0 + width * 0.30)),
        int(round(y0 + height * 0.56)),
    )
    raised_paw = (
        int(round(x0 + width * 0.41)),
        int(round(y0 + height * 0.36)),
    )
    planted_paw = (
        int(round(x0 + width * 0.38)),
        int(round(y0 + height * 0.91)),
    )
    lower_muzzle = (
        int(round(x0 + width * 0.49)),
        int(round(y0 + height * 0.35)),
    )

    editable_core = np.zeros(foreground.shape, dtype=np.uint8)
    cv2.line(
        editable_core,
        shoulder,
        elbow,
        255,
        max(36, int(round(width * 0.25))),
        cv2.LINE_AA,
    )
    cv2.line(
        editable_core,
        elbow,
        raised_paw,
        255,
        max(34, int(round(width * 0.23))),
        cv2.LINE_AA,
    )
    cv2.line(
        editable_core,
        shoulder,
        planted_paw,
        255,
        max(38, int(round(width * 0.26))),
        cv2.LINE_AA,
    )
    cv2.ellipse(
        editable_core,
        raised_paw,
        (
            max(24, int(round(width * 0.16))),
            max(20, int(round(height * 0.075))),
        ),
        0,
        0,
        360,
        255,
        -1,
        cv2.LINE_AA,
    )
    cv2.ellipse(
        editable_core,
        lower_muzzle,
        (
            max(22, int(round(width * 0.14))),
            max(16, int(round(height * 0.055))),
        ),
        0,
        0,
        360,
        255,
        -1,
        cv2.LINE_AA,
    )
    cv2.ellipse(
        editable_core,
        planted_paw,
        (
            max(26, int(round(width * 0.17))),
            max(18, int(round(height * 0.060))),
        ),
        0,
        0,
        360,
        255,
        -1,
        cv2.LINE_AA,
    )

    context_dilation_px = int(
        GROOMING_ADVISORY_ACTION_EDIT_MASK_CONTEXT_DILATION_PX
    )
    subject_context = cv2.dilate(
        foreground.astype(np.uint8),
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (context_dilation_px * 2 + 1, context_dilation_px * 2 + 1),
        ),
        iterations=1,
    ).astype(bool)
    editable_support = (editable_core > 0) & subject_context
    editable_pixels = int(np.count_nonzero(editable_support))
    if editable_pixels < 4096:
        raise RuntimeError("Grooming local action edit region is too small.")

    feather_sigma = float(GROOMING_ADVISORY_ACTION_EDIT_MASK_FEATHER_SIGMA_PX)
    edit_strength = cv2.GaussianBlur(
        editable_support.astype(np.float32),
        (0, 0),
        sigmaX=feather_sigma,
        sigmaY=feather_sigma,
    ).clip(0.0, 1.0)
    edit_strength[edit_strength < 0.01] = 0.0
    edit_strength[edit_strength > 0.99] = 1.0
    protection_strength = 1.0 - edit_strength
    full_edit_pixels = int(np.count_nonzero(edit_strength >= 0.99))
    feathered_edit_pixels = int(
        np.count_nonzero((edit_strength > 0.0) & (edit_strength < 1.0))
    )
    protected_pixels = int(np.count_nonzero(protection_strength >= 0.99))
    protected_face_probe = (
        int(round(x0 + width * 0.50)),
        int(round(y0 + height * 0.15)),
    )
    if (
        full_edit_pixels < 512
        or feathered_edit_pixels < 512
        or protected_pixels < 4096
        or protection_strength[protected_face_probe[1], protected_face_probe[0]]
        < 0.95
    ):
        raise RuntimeError("Grooming local action edit mask is not usable.")

    # ComfyUI LoadImage emits MASK = 1 - alpha. Opaque alpha locks the canonical
    # image; only the curved transparent action corridor receives fresh noise.
    # This is generation conditioning only. No returned Qwen pixel is replaced.
    mask_rgba = np.zeros((512, 512, 4), dtype=np.uint8)
    mask_rgba[:, :, 3] = np.rint(protection_strength * 255.0).astype(np.uint8)
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask_rgba, mode="RGBA").save(
        target,
        format="PNG",
        optimize=True,
    )
    return {
        "policy": "canonical_local_foreleg_prompt_edit_raw_qwen_v1",
        "canonical_reference_path": str(canonical_reference),
        "canonical_reference_sha256": file_sha256(canonical_reference),
        "canonical_mask_method": mask_method,
        "canonical_foreground_bbox": [x0, y0, x1, y1],
        "shoulder_anchor": list(shoulder),
        "elbow_anchor": list(elbow),
        "raised_paw_target": list(raised_paw),
        "planted_paw_source": list(planted_paw),
        "lower_muzzle_contact": list(lower_muzzle),
        "context_dilation_px": context_dilation_px,
        "feather_sigma_px": feather_sigma,
        "editable_support_pixels": editable_pixels,
        "full_edit_pixels": full_edit_pixels,
        "feathered_edit_pixels": feathered_edit_pixels,
        "protected_pixels": protected_pixels,
        "protected_face_probe": list(protected_face_probe),
        "protected_face_probe_strength": round(
            float(
                protection_strength[
                    protected_face_probe[1],
                    protected_face_probe[0],
                ]
            ),
            6,
        ),
        "noise_mask_loadimage_contract": "mask_equals_one_minus_alpha",
        "qwen_action_edit_strength_equals_one_minus_protection": True,
        "prompt_only_action_geometry": True,
        "video_first_frame_loaded_by_qwen": False,
        "white_maquette_loaded_by_qwen": False,
        "foreign_pose_image_loaded_by_qwen": False,
        "post_qwen_pixel_composite_enabled": False,
        "raw_qwen_output_required": True,
        "target_path": str(target),
        "target_sha256": file_sha256(target),
    }


def _canonical_aligned_grooming_foreleg_geometry(
    geometry_mask: np.ndarray,
    canonical_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build an appearance-free lifted foreleg that reaches canonical muzzle space."""
    _sync_local_video_runtime(globals(), {"_canonical_aligned_grooming_foreleg_geometry"})

    geometry_mask = np.asarray(geometry_mask, dtype=bool)
    canonical_mask = np.asarray(canonical_mask, dtype=bool)
    if (
        geometry_mask.ndim != 2
        or canonical_mask.shape != geometry_mask.shape
        or int(np.count_nonzero(geometry_mask)) < 4096
        or int(np.count_nonzero(canonical_mask)) < 4096
    ):
        raise ValueError("Grooming canonical-aligned foreleg masks are invalid.")

    geometry_ys, geometry_xs = np.nonzero(geometry_mask)
    canonical_ys, canonical_xs = np.nonzero(canonical_mask)
    geometry_x0 = int(geometry_xs.min())
    geometry_y0 = int(geometry_ys.min())
    geometry_x1 = int(geometry_xs.max()) + 1
    geometry_y1 = int(geometry_ys.max()) + 1
    canonical_x0 = int(canonical_xs.min())
    canonical_y0 = int(canonical_ys.min())
    canonical_x1 = int(canonical_xs.max()) + 1
    canonical_y1 = int(canonical_ys.max()) + 1
    geometry_width = float(geometry_x1 - geometry_x0)
    geometry_height = float(geometry_y1 - geometry_y0)
    canonical_width = float(canonical_x1 - canonical_x0)
    canonical_height = float(canonical_y1 - canonical_y0)
    canonical_center_x = canonical_x0 + canonical_width * 0.50

    # The white-maquette paw sits lower than the canonical muzzle.  Retaining a
    # broad fixed ellipse therefore produced either a chest-height paw or an
    # undifferentiated rectangular blob after the maquette head was removed.
    # This tapered, texture-free chain keeps the source shoulder/body pose but
    # places the rounded paw where Image 1's lower muzzle actually is.
    shoulder = (
        int(round(geometry_x0 + geometry_width * 0.32)),
        int(round(geometry_y0 + geometry_height * 0.70)),
    )
    elbow = (
        int(round(geometry_x0 + geometry_width * 0.25)),
        int(round(geometry_y0 + geometry_height * 0.55)),
    )
    wrist = (
        int(round(canonical_center_x - canonical_width * 0.14)),
        int(round(canonical_y0 + canonical_height * 0.43)),
    )
    paw_center = (
        int(round(canonical_center_x - canonical_width * 0.08)),
        int(round(canonical_y0 + canonical_height * 0.35)),
    )
    paw_axes = (
        max(18, int(round(canonical_width * 0.11))),
        max(16, int(round(canonical_height * 0.052))),
    )

    foreleg_u8 = np.zeros(geometry_mask.shape, dtype=np.uint8)
    cv2.line(
        foreleg_u8,
        shoulder,
        elbow,
        255,
        max(22, int(round(geometry_width * 0.16))),
        cv2.LINE_AA,
    )
    cv2.line(
        foreleg_u8,
        elbow,
        wrist,
        255,
        max(20, int(round(geometry_width * 0.14))),
        cv2.LINE_AA,
    )
    cv2.line(
        foreleg_u8,
        wrist,
        paw_center,
        255,
        max(18, int(round(geometry_width * 0.12))),
        cv2.LINE_AA,
    )
    cv2.ellipse(
        foreleg_u8,
        paw_center,
        paw_axes,
        -12.0,
        0.0,
        360.0,
        255,
        -1,
        cv2.LINE_AA,
    )
    foreleg = (foreleg_u8 >= 64) & geometry_mask
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        foreleg.astype(np.uint8),
        connectivity=8,
    )
    if component_count <= 1:
        raise RuntimeError("Grooming canonical-aligned foreleg is empty.")
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    foreleg = labels == largest_label
    foreleg_pixels = int(np.count_nonzero(foreleg))
    if foreleg_pixels < 512:
        raise RuntimeError("Grooming canonical-aligned foreleg is too small.")
    foreleg_ys, foreleg_xs = np.nonzero(foreleg)
    head_cutoff_y = int(round(geometry_y0 + geometry_height * 0.46))
    pixels_above_head_cutoff = int(
        np.count_nonzero(foreleg & (np.indices(foreleg.shape)[0] < head_cutoff_y))
    )
    pixels_below_head_cutoff = foreleg_pixels - pixels_above_head_cutoff
    if pixels_above_head_cutoff < 256 or pixels_below_head_cutoff < 256:
        raise RuntimeError(
            "Grooming canonical-aligned foreleg does not bridge shoulder and muzzle."
        )
    return foreleg, {
        "policy": "canonical_muzzle_aligned_tapered_neutral_foreleg_v1",
        "geometry_foreground_bbox": [
            geometry_x0,
            geometry_y0,
            geometry_x1,
            geometry_y1,
        ],
        "canonical_foreground_bbox": [
            canonical_x0,
            canonical_y0,
            canonical_x1,
            canonical_y1,
        ],
        "shoulder_anchor": list(shoulder),
        "elbow_anchor": list(elbow),
        "wrist_anchor": list(wrist),
        "paw_center": list(paw_center),
        "paw_axes": list(paw_axes),
        "head_cutoff_y": head_cutoff_y,
        "foreground_pixels": foreleg_pixels,
        "pixels_above_head_cutoff": pixels_above_head_cutoff,
        "pixels_below_head_cutoff": pixels_below_head_cutoff,
        "foreground_bbox": [
            int(foreleg_xs.min()),
            int(foreleg_ys.min()),
            int(foreleg_xs.max()) + 1,
            int(foreleg_ys.max()) + 1,
        ],
        "source_rgb_or_luma_used": False,
    }


def prepare_canonical_grooming_pose_transfer_seed(
    action_reference: Path,
    canonical_reference: Path,
    target: Path,
    *,
    geometry_guide: Path,
    foreleg_guide: Path | None,
    preserve_normalized_geometry_luma: bool = False,
    complete_geometry_guide: bool = False,
    remove_head_identity_silhouette: bool = False,
) -> dict[str, Any]:
    """Build a grooming pose canvas from white-maquette geometry only.

    The frozen white-maquette frame is retained only as digest-bound provenance
    and an outer-alpha check. For legacy single-tone guides, Qwen-visible pixels
    remain a geometry distance field plus a synthetic foreleg lift. A verified
    white-maquette private guide may instead contribute normalized, blurred
    low-frequency anatomical depth. Its raw RGB, chroma, exact luma, material,
    lighting and texture are still discarded, while the real head/torso/limb
    depth relationships remain readable and no artificial foreleg capsule is
    added. Every visible tone is recolored with the canonical subject's global
    LAB median.
    """
    _sync_local_video_runtime(globals(), {"prepare_canonical_grooming_pose_transfer_seed"})

    if (
        not action_reference.is_file()
        or not geometry_guide.is_file()
        or (
            not complete_geometry_guide
            and (foreleg_guide is None or not foreleg_guide.is_file())
        )
    ):
        raise FileNotFoundError("Grooming pose-transfer inputs are missing.")
    if remove_head_identity_silhouette and not (
        preserve_normalized_geometry_luma and complete_geometry_guide
    ):
        raise ValueError(
            "Headless grooming conditioning requires the complete normalized "
            "white-maquette geometry guide."
        )
    with Image.open(action_reference) as loaded:
        if "A" not in loaded.getbands():
            raise ValueError("grooming action reference requires alpha")
        action_image = loaded.convert("RGBA")
    with Image.open(canonical_reference) as loaded:
        canonical_has_alpha = "A" in loaded.getbands()
        canonical_image = loaded.convert("RGBA")
    with Image.open(geometry_guide) as loaded:
        geometry_image = loaded.convert("RGB")
    foreleg_image: Image.Image | None = None
    if foreleg_guide is not None:
        with Image.open(foreleg_guide) as loaded:
            foreleg_image = loaded.convert("RGB")
    if action_image.size != (512, 512):
        raise ValueError("grooming action reference must be 512x512")
    if canonical_image.size != action_image.size:
        raise ValueError("grooming canonical reference must be 512x512")
    if geometry_image.size != action_image.size:
        raise ValueError("grooming geometry guide must be 512x512")
    if foreleg_image is not None and foreleg_image.size != action_image.size:
        raise ValueError("grooming foreleg guide must be 512x512")

    action_rgba = np.asarray(action_image, dtype=np.uint8)
    canonical_rgba = np.asarray(canonical_image, dtype=np.uint8)
    geometry_rgb = np.asarray(geometry_image, dtype=np.uint8)
    foreleg_rgb = (
        np.asarray(foreleg_image, dtype=np.uint8)
        if foreleg_image is not None
        else None
    )
    action_mask_raw = action_rgba[:, :, 3] >= 64
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        action_mask_raw.astype(np.uint8),
        connectivity=8,
    )
    if component_count <= 1:
        raise RuntimeError("grooming action reference foreground is empty")
    action_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    action_mask = labels == action_label
    action_foreground_pixels = int(np.count_nonzero(action_mask))
    if action_foreground_pixels < 4096:
        raise RuntimeError("grooming action reference foreground is too small")

    geometry_red = geometry_rgb[:, :, 0].astype(np.int16)
    geometry_green = geometry_rgb[:, :, 1].astype(np.int16)
    geometry_blue = geometry_rgb[:, :, 2].astype(np.int16)
    geometry_background = (
        (geometry_green >= 150)
        & ((geometry_green - geometry_red) >= 60)
        & ((geometry_green - geometry_blue) >= 60)
    )
    geometry_mask_raw = ~geometry_background
    geometry_count, geometry_labels, geometry_stats, _ = (
        cv2.connectedComponentsWithStats(
            geometry_mask_raw.astype(np.uint8),
            connectivity=8,
        )
    )
    if geometry_count <= 1:
        raise RuntimeError("grooming geometry guide foreground is empty")
    geometry_label = 1 + int(
        np.argmax(geometry_stats[1:, cv2.CC_STAT_AREA])
    )
    geometry_mask = geometry_labels == geometry_label
    public_geometry_mask = geometry_mask
    public_geometry_foreground_pixels = int(
        np.count_nonzero(public_geometry_mask)
    )
    if public_geometry_foreground_pixels < 4096:
        raise RuntimeError("grooming geometry guide foreground is too small")
    geometry_foreground_rgb = geometry_rgb[public_geometry_mask]
    geometry_channel_spread = np.ptp(
        geometry_foreground_rgb.astype(np.int16),
        axis=1,
    )
    geometry_neutral_fraction = float(
        np.count_nonzero(geometry_channel_spread <= 4)
    ) / float(public_geometry_foreground_pixels)
    if geometry_neutral_fraction < 0.98:
        raise RuntimeError(
            "grooming geometry guide contains appearance-bearing chroma"
        )
    guide_unique_rgb_count = int(
        np.unique(geometry_foreground_rgb, axis=0).shape[0]
    )
    if preserve_normalized_geometry_luma and guide_unique_rgb_count < 3:
        raise RuntimeError(
            "white-maquette grooming depth guide lacks anatomical luma variation"
        )

    if complete_geometry_guide:
        if foreleg_rgb is not None:
            raise RuntimeError(
                "complete grooming geometry must not load a synthetic foreleg guide"
            )
        semantic_foreleg = np.zeros(public_geometry_mask.shape, dtype=bool)
        completion_foreground = public_geometry_mask.copy()
        completion_overlap = 1.0
    else:
        if foreleg_rgb is None:
            raise RuntimeError("grooming foreleg guide is unavailable")
        foreleg_background = np.all(
            foreleg_rgb == np.asarray((0, 255, 0), dtype=np.uint8),
            axis=2,
        )
        foreleg_body = np.all(
            foreleg_rgb
            == np.asarray(GROOMING_QWEN_GUIDE_FOREGROUND_RGB, dtype=np.uint8),
            axis=2,
        )
        semantic_foreleg = np.all(
            foreleg_rgb
            == np.asarray(GROOMING_QWEN_GUIDE_FORELEG_RGB, dtype=np.uint8),
            axis=2,
        )
        if not np.all(foreleg_background | foreleg_body | semantic_foreleg):
            raise RuntimeError(
                "grooming foreleg guide contains an unsupported appearance color"
            )
        semantic_foreleg_pixels = int(np.count_nonzero(semantic_foreleg))
        if not 650 <= semantic_foreleg_pixels <= 6000:
            raise RuntimeError(
                "grooming foreleg guide lacks its bounded synthetic semantic"
            )
        completion_foreground = foreleg_body | semantic_foreleg
        completion_overlap = float(
            np.count_nonzero(public_geometry_mask & completion_foreground)
        ) / float(public_geometry_foreground_pixels)
        if completion_overlap < 0.95:
            raise RuntimeError(
                "grooming foreleg guide does not match the public outer silhouette"
            )
    semantic_foreleg_pixels = int(np.count_nonzero(semantic_foreleg))

    combined_geometry = public_geometry_mask | completion_foreground
    background_count, background_labels = cv2.connectedComponents(
        (~combined_geometry).astype(np.uint8),
        connectivity=8,
    )
    border_labels = np.unique(
        np.concatenate(
            [
                background_labels[0, :],
                background_labels[-1, :],
                background_labels[:, 0],
                background_labels[:, -1],
            ]
        )
    )
    exterior_background = np.isin(
        background_labels,
        border_labels[(border_labels > 0) & (border_labels < background_count)],
    )
    completed_geometry_raw = ~exterior_background
    completed_count, completed_labels, completed_stats, _ = (
        cv2.connectedComponentsWithStats(
            completed_geometry_raw.astype(np.uint8),
            connectivity=8,
        )
    )
    if completed_count <= 1:
        raise RuntimeError("grooming completed geometry foreground is empty")
    completed_label = 1 + int(
        np.argmax(completed_stats[1:, cv2.CC_STAT_AREA])
    )
    geometry_mask = completed_labels == completed_label
    geometry_foreground_pixels = int(np.count_nonzero(geometry_mask))
    if np.any(semantic_foreleg & ~geometry_mask):
        raise RuntimeError("grooming foreleg semantic escaped the completed silhouette")
    retained_geometry_fraction = float(
        np.count_nonzero(action_mask & geometry_mask)
    ) / float(action_foreground_pixels)
    filled_internal_channel_pixels = int(
        np.count_nonzero(geometry_mask & ~public_geometry_mask)
    )

    if canonical_has_alpha and np.any(canonical_rgba[:, :, 3] < 250):
        canonical_mask_raw = canonical_rgba[:, :, 3] >= 64
        canonical_mask_method = "alpha"
    else:
        canonical_rgb = canonical_rgba[:, :, :3]
        red = canonical_rgb[:, :, 0].astype(np.int16)
        green = canonical_rgb[:, :, 1].astype(np.int16)
        blue = canonical_rgb[:, :, 2].astype(np.int16)
        green_candidate = (
            (green >= 150)
            & ((green - red) >= 65)
            & ((green - blue) >= 65)
        )
        green_count, green_labels = cv2.connectedComponents(
            green_candidate.astype(np.uint8),
            connectivity=8,
        )
        border_labels = np.unique(
            np.concatenate(
                [
                    green_labels[0, :],
                    green_labels[-1, :],
                    green_labels[:, 0],
                    green_labels[:, -1],
                ]
            )
        )
        background = np.isin(
            green_labels,
            border_labels[(border_labels > 0) & (border_labels < green_count)],
        )
        canonical_mask_raw = ~background
        canonical_mask_method = "border_connected_green_screen"
    canonical_count, canonical_labels, canonical_stats, _ = (
        cv2.connectedComponentsWithStats(
            canonical_mask_raw.astype(np.uint8),
            connectivity=8,
        )
    )
    if canonical_count <= 1:
        raise RuntimeError("grooming canonical reference foreground is empty")
    canonical_label = 1 + int(
        np.argmax(canonical_stats[1:, cv2.CC_STAT_AREA])
    )
    canonical_mask = canonical_labels == canonical_label
    canonical_foreground_pixels = int(np.count_nonzero(canonical_mask))
    if canonical_foreground_pixels < 4096:
        raise RuntimeError("grooming canonical reference foreground is too small")

    canonical_lab = cv2.cvtColor(
        canonical_rgba[:, :, :3],
        cv2.COLOR_RGB2LAB,
    )
    geometry_distance = cv2.distanceTransform(
        geometry_mask.astype(np.uint8),
        cv2.DIST_L2,
        5,
    )
    sigma = float(
        GROOMING_CANONICAL_POSE_TRANSFER_GEOMETRY_SMOOTH_SIGMA_PX
    )
    geometry_distance = cv2.GaussianBlur(
        geometry_distance,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
    )
    geometry_distance_values = geometry_distance[geometry_mask]
    distance_p95 = float(np.percentile(geometry_distance_values, 95.0))
    if distance_p95 <= 1.0:
        raise RuntimeError("grooming geometry guide lacks usable interior depth")
    distance_depth = np.clip(geometry_distance / distance_p95, 0.0, 1.0)
    normalized_geometry_luma_p05: float | None = None
    normalized_geometry_luma_p95: float | None = None
    normalized_geometry_luma_span: float | None = None
    normalized_geometry_smooth_sigma: float | None = None
    normalized_geometry_distance_blend = 1.0
    if preserve_normalized_geometry_luma:
        geometry_luma = geometry_rgb[:, :, 0].astype(np.float32)
        geometry_luma_values = geometry_luma[public_geometry_mask]
        geometry_luma_fill = float(np.median(geometry_luma_values))
        filled_geometry_luma = np.where(
            public_geometry_mask,
            geometry_luma,
            geometry_luma_fill,
        ).astype(np.float32)
        normalized_geometry_smooth_sigma = float(
            GROOMING_WHITE_MAQUETTE_DEPTH_SMOOTH_SIGMA_PX
        )
        normalized_geometry_luma = cv2.GaussianBlur(
            filled_geometry_luma,
            (0, 0),
            sigmaX=normalized_geometry_smooth_sigma,
            sigmaY=normalized_geometry_smooth_sigma,
        )
        normalized_values = normalized_geometry_luma[public_geometry_mask]
        normalized_geometry_luma_p05 = float(
            np.percentile(normalized_values, 5.0)
        )
        normalized_geometry_luma_p95 = float(
            np.percentile(normalized_values, 95.0)
        )
        normalized_geometry_luma_span = (
            normalized_geometry_luma_p95 - normalized_geometry_luma_p05
        )
        if normalized_geometry_luma_span < 4.0:
            raise RuntimeError(
                "white-maquette grooming depth guide has insufficient luma depth"
            )
        normalized_anatomy_depth = np.clip(
            (
                normalized_geometry_luma - normalized_geometry_luma_p05
            )
            / normalized_geometry_luma_span,
            0.0,
            1.0,
        )
        normalized_geometry_distance_blend = float(
            GROOMING_WHITE_MAQUETTE_DEPTH_DISTANCE_BLEND
        )
        geometry_depth = (
            normalized_anatomy_depth
            * (1.0 - normalized_geometry_distance_blend)
            + distance_depth * normalized_geometry_distance_blend
        )
    else:
        geometry_depth = distance_depth

    canonical_values = canonical_lab[canonical_mask]
    canonical_luma = canonical_values[:, 0].astype(np.float32)
    canonical_luma_median = float(np.median(canonical_luma))
    luma_half_range = float(
        GROOMING_WHITE_MAQUETTE_FRAME0_POSE_LUMA_HALF_RANGE
        if preserve_normalized_geometry_luma and complete_geometry_guide
        else GROOMING_WHITE_MAQUETTE_DEPTH_LUMA_HALF_RANGE
        if preserve_normalized_geometry_luma
        else GROOMING_CANONICAL_POSE_TRANSFER_GEOMETRY_LUMA_HALF_RANGE
    )
    target_luma_low = max(
        8.0,
        canonical_luma_median - luma_half_range,
    )
    target_luma_high = min(
        247.0,
        canonical_luma_median + luma_half_range,
    )
    if target_luma_high - target_luma_low < 8.0:
        midpoint = (target_luma_low + target_luma_high) / 2.0
        target_luma_low = max(8.0, midpoint - 4.0)
        target_luma_high = min(247.0, midpoint + 4.0)
    remapped_luma_base = (
        target_luma_low
        + geometry_depth * (target_luma_high - target_luma_low)
    )
    foreleg_sigma = float(
        GROOMING_CANONICAL_POSE_TRANSFER_FORELEG_SMOOTH_SIGMA_PX
    )
    if semantic_foreleg_pixels:
        foreleg_depth = cv2.GaussianBlur(
            semantic_foreleg.astype(np.float32),
            (0, 0),
            sigmaX=foreleg_sigma,
            sigmaY=foreleg_sigma,
        )
        foreleg_depth_peak = float(foreleg_depth.max())
        if foreleg_depth_peak < 0.50:
            raise RuntimeError("grooming foreleg semantic lacks a stable depth core")
        foreleg_depth = np.clip(foreleg_depth / foreleg_depth_peak, 0.0, 1.0)
    else:
        foreleg_depth = np.zeros(semantic_foreleg.shape, dtype=np.float32)
        foreleg_depth_peak = 0.0
    foreleg_luma_lift = (
        0.0
        if preserve_normalized_geometry_luma or not semantic_foreleg_pixels
        else float(GROOMING_CANONICAL_POSE_TRANSFER_FORELEG_LUMA_LIFT)
    )
    remapped_luma_float = np.clip(
        remapped_luma_base + foreleg_depth * foreleg_luma_lift,
        8.0,
        247.0,
    )
    canonical_aligned_foreleg_audit: dict[str, Any] | None = None
    raised_paw_geometry = np.zeros(geometry_mask.shape, dtype=bool)
    raised_paw_geometry_retained_pixels = 0
    raised_paw_volume_distance_p95: float | None = None
    raised_paw_volume_weight_sigma_px: float | None = None
    raised_paw_volume_luma_range: list[float] | None = None
    if preserve_normalized_geometry_luma and complete_geometry_guide:
        raised_paw_geometry, canonical_aligned_foreleg_audit = (
            _canonical_aligned_grooming_foreleg_geometry(
                geometry_mask,
                canonical_mask,
            )
        )
        raised_paw_geometry_retained_pixels = int(
            np.count_nonzero(raised_paw_geometry)
        )
        raised_paw_distance = cv2.distanceTransform(
            raised_paw_geometry.astype(np.uint8),
            cv2.DIST_L2,
            5,
        )
        raised_paw_volume_distance_p95 = max(
            float(
                np.percentile(
                    raised_paw_distance[raised_paw_geometry],
                    95.0,
                )
            ),
            1.0,
        )
        raised_paw_depth = np.sqrt(
            np.clip(
                raised_paw_distance / raised_paw_volume_distance_p95,
                0.0,
                1.0,
            )
        )
        luma_span = target_luma_high - target_luma_low
        raised_paw_luma_low = target_luma_low + luma_span * 0.56
        raised_paw_luma_high = target_luma_low + luma_span * 0.72
        raised_paw_volume = (
            raised_paw_luma_low
            + raised_paw_depth
            * (raised_paw_luma_high - raised_paw_luma_low)
        )
        raised_paw_volume_weight_sigma_px = 5.0
        raised_paw_weight = cv2.GaussianBlur(
            raised_paw_geometry.astype(np.float32),
            (0, 0),
            sigmaX=raised_paw_volume_weight_sigma_px,
            sigmaY=raised_paw_volume_weight_sigma_px,
        ).clip(0.0, 1.0)
        raised_paw_weight *= geometry_mask.astype(np.float32)
        remapped_luma_float = (
            remapped_luma_float * (1.0 - raised_paw_weight)
            + raised_paw_volume * raised_paw_weight
        )
        raised_paw_volume_luma_range = [
            round(float(raised_paw_luma_low), 6),
            round(float(raised_paw_luma_high), 6),
        ]
    remapped_luma = np.rint(remapped_luma_float).astype(np.uint8)
    canonical_ab = np.rint(
        np.median(canonical_values[:, 1:3].astype(np.float32), axis=0)
    ).astype(np.uint8)
    seed_lab = np.empty((512, 512, 3), dtype=np.uint8)
    seed_lab[:, :, 0] = remapped_luma
    seed_lab[:, :, 1] = canonical_ab[0]
    seed_lab[:, :, 2] = canonical_ab[1]
    seed_rgb = cv2.cvtColor(seed_lab, cv2.COLOR_LAB2RGB)
    qwen_visible_geometry_mask = geometry_mask.copy()
    head_silhouette_removed_pixels = 0
    head_silhouette_cutoff_y: int | None = None
    if remove_head_identity_silhouette:
        geometry_ys, geometry_xs = np.nonzero(geometry_mask)
        geometry_x0 = int(geometry_xs.min())
        geometry_y0 = int(geometry_ys.min())
        geometry_x1 = int(geometry_xs.max()) + 1
        geometry_y1 = int(geometry_ys.max()) + 1
        geometry_width = float(geometry_x1 - geometry_x0)
        geometry_height = float(geometry_y1 - geometry_y0)
        head_silhouette_cutoff_y = int(
            round(geometry_y0 + geometry_height * 0.46)
        )
        row_grid = np.indices(geometry_mask.shape)[0]
        head_region = geometry_mask & (row_grid < head_silhouette_cutoff_y)
        qwen_visible_geometry_mask = geometry_mask & (
            ~head_region | raised_paw_geometry
        )
        head_silhouette_removed_pixels = int(
            np.count_nonzero(geometry_mask & ~qwen_visible_geometry_mask)
        )
        raised_paw_geometry_retained_pixels = int(
            np.count_nonzero(raised_paw_geometry)
        )
        if head_silhouette_removed_pixels < 4096:
            raise RuntimeError(
                "Grooming body-only guide did not remove enough maquette head geometry."
            )
        if raised_paw_geometry_retained_pixels < 512:
            raise RuntimeError(
                "Grooming body-only guide lost its raised-paw action geometry."
            )
        if np.count_nonzero(qwen_visible_geometry_mask) < 4096:
            raise RuntimeError("Grooming body-only guide lost its body geometry.")
    output_rgb = np.empty((512, 512, 3), dtype=np.uint8)
    output_rgb[:, :] = (0, 255, 0)
    output_rgb[qwen_visible_geometry_mask] = seed_rgb[qwen_visible_geometry_mask]

    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output_rgb, mode="RGB").save(
        target,
        format="PNG",
        optimize=True,
    )
    with Image.open(target) as loaded:
        persisted = np.asarray(loaded.convert("RGB"), dtype=np.uint8)
    if not np.all(
        persisted[~qwen_visible_geometry_mask]
        == np.asarray((0, 255, 0), dtype=np.uint8)
    ):
        raise RuntimeError("grooming pose-transfer seed background changed")
    if not np.array_equal(
        persisted[qwen_visible_geometry_mask],
        output_rgb[qwen_visible_geometry_mask],
    ):
        raise RuntimeError("grooming pose-transfer seed changed during persistence")

    ys, xs = np.nonzero(geometry_mask)
    qwen_ys, qwen_xs = np.nonzero(qwen_visible_geometry_mask)
    seed_gray = seed_lab[:, :, 0].astype(np.float32)
    seed_high_frequency_energy = float(
        np.mean(
            np.abs(cv2.Laplacian(seed_gray, cv2.CV_32F))[
                qwen_visible_geometry_mask
            ]
        )
    )
    foreground_luma = seed_lab[:, :, 0][qwen_visible_geometry_mask]
    return {
        "policy": GROOMING_CANONICAL_POSE_TRANSFER_SEED_POLICY,
        "action_reference_path": str(action_reference),
        "action_reference_sha256": file_sha256(action_reference),
        "action_reference_kind": "white_maquette_geometry_provenance_only",
        "geometry_guide_path": str(geometry_guide),
        "geometry_guide_sha256": file_sha256(geometry_guide),
        "canonical_reference_path": str(canonical_reference),
        "canonical_reference_sha256": file_sha256(canonical_reference),
        "canonical_mask_method": canonical_mask_method,
        "action_foreground_pixels": action_foreground_pixels,
        "public_geometry_foreground_pixels": public_geometry_foreground_pixels,
        "geometry_foreground_pixels": geometry_foreground_pixels,
        "geometry_foreground_bbox": [
            int(xs.min()),
            int(ys.min()),
            int(xs.max()) + 1,
            int(ys.max()) + 1,
        ],
        "geometry_retained_fraction_of_action_alpha": round(
            retained_geometry_fraction,
            6,
        ),
        "qwen_visible_geometry_foreground_pixels": int(
            np.count_nonzero(qwen_visible_geometry_mask)
        ),
        "qwen_visible_geometry_foreground_bbox": [
            int(qwen_xs.min()),
            int(qwen_ys.min()),
            int(qwen_xs.max()) + 1,
            int(qwen_ys.max()) + 1,
        ],
        "head_identity_silhouette_removed": bool(
            remove_head_identity_silhouette
        ),
        "head_identity_silhouette_cutoff_y": head_silhouette_cutoff_y,
        "head_identity_silhouette_removed_pixels": (
            head_silhouette_removed_pixels
        ),
        "raised_paw_geometry_retained_pixels": (
            raised_paw_geometry_retained_pixels
        ),
        "raised_paw_geometry_policy": (
            canonical_aligned_foreleg_audit["policy"]
            if canonical_aligned_foreleg_audit is not None
            else None
        ),
        "raised_paw_geometry_audit": canonical_aligned_foreleg_audit,
        "raised_paw_volume_source_rgb_or_luma_used": False,
        "raised_paw_volume_distance_p95": (
            round(raised_paw_volume_distance_p95, 6)
            if raised_paw_volume_distance_p95 is not None
            else None
        ),
        "raised_paw_volume_weight_sigma_px": (
            raised_paw_volume_weight_sigma_px
        ),
        "raised_paw_volume_luma_range": raised_paw_volume_luma_range,
        "public_internal_channel_pixels_filled": filled_internal_channel_pixels,
        "qwen_visible_internal_green_channel_pixels": (
            head_silhouette_removed_pixels
        ),
        "geometry_guide_unique_foreground_rgb_count": guide_unique_rgb_count,
        "geometry_guide_neutral_foreground_fraction": round(
            geometry_neutral_fraction,
            6,
        ),
        "canonical_foreground_pixels": canonical_foreground_pixels,
        "geometry_smooth_sigma_px": sigma,
        "geometry_distance_p95": round(distance_p95, 6),
        "normalized_geometry_luma_exposed": bool(
            preserve_normalized_geometry_luma
        ),
        "normalized_geometry_luma_policy": (
            "blurred_percentile_normalized_white_maquette_depth_v1"
            if preserve_normalized_geometry_luma
            else None
        ),
        "normalized_geometry_luma_p05": (
            round(normalized_geometry_luma_p05, 6)
            if normalized_geometry_luma_p05 is not None
            else None
        ),
        "normalized_geometry_luma_p95": (
            round(normalized_geometry_luma_p95, 6)
            if normalized_geometry_luma_p95 is not None
            else None
        ),
        "normalized_geometry_luma_span": (
            round(normalized_geometry_luma_span, 6)
            if normalized_geometry_luma_span is not None
            else None
        ),
        "normalized_geometry_smooth_sigma_px": (
            normalized_geometry_smooth_sigma
        ),
        "normalized_geometry_distance_blend": round(
            normalized_geometry_distance_blend,
            6,
        ),
        "complete_geometry_guide": complete_geometry_guide,
        "foreleg_guide_path": str(foreleg_guide) if foreleg_guide is not None else None,
        "foreleg_guide_sha256": (
            file_sha256(foreleg_guide) if foreleg_guide is not None else None
        ),
        "foreleg_guide_public_overlap": round(completion_overlap, 6),
        "semantic_foreleg_pixels": semantic_foreleg_pixels,
        "foreleg_smooth_sigma_px": foreleg_sigma,
        "foreleg_depth_peak": round(foreleg_depth_peak, 6),
        "foreleg_luma_lift": foreleg_luma_lift,
        "target_luma_low": round(float(target_luma_low), 6),
        "target_luma_high": round(float(target_luma_high), 6),
        "target_luma_range": int(foreground_luma.max() - foreground_luma.min()),
        "foreground_luma_min": int(foreground_luma.min()),
        "foreground_luma_max": int(foreground_luma.max()),
        "canonical_luma_median": round(canonical_luma_median, 6),
        "canonical_ab_median": canonical_ab.tolist(),
        "seed_high_frequency_energy": round(seed_high_frequency_energy, 8),
        "source_alpha_geometry_preserved": False,
        "source_outer_alpha_geometry_retained": False,
        "action_alpha_overlap_blocking": False,
        "action_reference_used_for_conditioning": False,
        "source_rgb_read_for_conditioning": False,
        "source_rgb_preserved": False,
        "source_chroma_preserved": False,
        "source_raw_luma_values_preserved": False,
        "source_low_frequency_luma_exposed": False,
        "source_luma_used_for_shading": False,
        "source_normalized_low_frequency_geometry_exposed": bool(
            preserve_normalized_geometry_luma
        ),
        "normalized_geometry_luma_used_for_shading": bool(
            preserve_normalized_geometry_luma
        ),
        "source_high_frequency_texture_exposed": False,
        "source_maquette_face_exposed": False,
        "source_maquette_eyes_exposed": False,
        "source_maquette_white_surface_exposed": False,
        "source_maquette_material_or_lighting_exposed": False,
        "geometry_distance_field_only": False,
        "geometry_and_soft_foreleg_depth_only": bool(
            not preserve_normalized_geometry_luma and semantic_foreleg_pixels
        ),
        "soft_foreleg_depth_semantic_exposed": bool(
            not preserve_normalized_geometry_luma and semantic_foreleg_pixels
        ),
        "normalized_white_maquette_depth_anatomy_only": bool(
            preserve_normalized_geometry_luma
        ),
        "semantic_foreleg_discrete_color_exposed": False,
        "public_open_channel_geometry_exposed": False,
        "canonical_spatial_pixels_copied": False,
        "canonical_palette_conditioning_only": True,
        "sampler_role": (
            "canonical_palette_headless_white_maquette_frame000_body_pose_anatomy"
            if (
                preserve_normalized_geometry_luma
                and complete_geometry_guide
                and remove_head_identity_silhouette
            )
            else "canonical_palette_white_maquette_frame000_body_pose_anatomy"
            if preserve_normalized_geometry_luma and complete_geometry_guide
            else "canonical_palette_normalized_white_maquette_depth_anatomy"
            if preserve_normalized_geometry_luma
            else "canonical_palette_soft_foreleg_depth_action_anatomy"
        ),
        "target_path": str(target),
        "target_sha256": file_sha256(target),
    }


def prepare_canonical_grooming_seed_bound_matte(
    source: Path,
    semantic_cutout: Path,
    pose_transfer_seed: Path,
    target: Path,
    *,
    green_target: Path | None = None,
) -> dict[str, Any]:
    """Recover a complete Qwen grooming matte from two independent signals.

    The semantic matte identifies pet pixels in the generated image. The
    deterministic pose-transfer seed contributes alpha geometry only where
    rembg omitted light fur. No seed RGB, luma, chroma, or texture is copied.
    Strict overlap bounds make a shifted subject, background object, or an
    unrelated semantic component fail closed before any pose gate can run.
    """
    _sync_local_video_runtime(globals(), {"prepare_canonical_grooming_seed_bound_matte"})

    source_digest = file_sha256(source)
    semantic_digest = file_sha256(semantic_cutout)
    seed_digest = file_sha256(pose_transfer_seed)
    with Image.open(source) as loaded:
        source_mode = loaded.mode
        source_image = loaded.convert("RGB")
    with Image.open(semantic_cutout) as loaded:
        semantic_has_alpha = "A" in loaded.getbands()
        semantic_image = loaded.convert("RGBA")
    with Image.open(pose_transfer_seed) as loaded:
        seed_image = loaded.convert("RGB")

    if not semantic_has_alpha:
        raise ValueError("grooming_seed_bound_semantic_cutout_has_no_alpha")
    if semantic_image.size != source_image.size:
        raise ValueError("grooming_seed_bound_source_semantic_size_mismatch")
    if source_image.width != source_image.height or not 512 <= source_image.width <= 2048:
        raise ValueError("grooming_seed_bound_source_canvas_is_not_supported_square")
    if seed_image.size != (512, 512):
        raise ValueError("grooming_seed_bound_pose_transfer_seed_must_be_512_square")

    source_rgb = np.asarray(source_image, dtype=np.uint8)
    semantic_rgba = np.asarray(semantic_image, dtype=np.uint8)
    semantic_alpha = semantic_rgba[:, :, 3]
    semantic_mask = semantic_alpha >= FurSurfaceThresholds().alpha_threshold
    seed_rgb = np.asarray(seed_image, dtype=np.uint8)
    pure_green = np.asarray((0, 255, 0), dtype=np.uint8)
    seed_background = np.all(seed_rgb == pure_green, axis=2)
    seed_mask_raw = ~seed_background
    seed_corners_are_background = bool(
        not seed_mask_raw[0, 0]
        and not seed_mask_raw[0, -1]
        and not seed_mask_raw[-1, 0]
        and not seed_mask_raw[-1, -1]
    )
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        seed_mask_raw.astype(np.uint8),
        connectivity=8,
    )
    if component_count <= 1:
        raise ValueError("grooming_seed_bound_pose_transfer_seed_foreground_is_empty")
    seed_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    seed_mask = labels == seed_label
    seed_raw_pixels = int(np.count_nonzero(seed_mask_raw))
    seed_component_pixels = int(np.count_nonzero(seed_mask))
    discarded_seed_fraction = float(
        max(0, seed_raw_pixels - seed_component_pixels) / max(1, seed_raw_pixels)
    )
    resized_seed_mask = np.asarray(
        Image.fromarray(seed_mask.astype(np.uint8) * 255, mode="L").resize(
            source_image.size,
            Image.Resampling.NEAREST,
        ),
        dtype=np.uint8,
    ) >= 128

    semantic_pixels = int(np.count_nonzero(semantic_mask))
    seed_pixels = int(np.count_nonzero(resized_seed_mask))
    intersection = semantic_mask & resized_seed_mask
    union_mask = semantic_mask | resized_seed_mask
    added_seed_mask = resized_seed_mask & ~semantic_mask
    intersection_pixels = int(np.count_nonzero(intersection))
    union_pixels = int(np.count_nonzero(union_mask))
    added_seed_pixels = int(np.count_nonzero(added_seed_mask))
    pixel_count = int(semantic_mask.size)
    intersection_over_seed = float(intersection_pixels / max(1, seed_pixels))
    intersection_over_semantic = float(
        intersection_pixels / max(1, semantic_pixels)
    )
    semantic_to_seed = float(semantic_pixels / max(1, seed_pixels))
    added_over_seed = float(added_seed_pixels / max(1, seed_pixels))
    checks = {
        "semantic_foreground_usable": bool(0.02 <= semantic_pixels / pixel_count <= 0.70),
        "seed_foreground_usable": bool(0.08 <= seed_pixels / pixel_count <= 0.70),
        "union_foreground_usable": bool(0.08 <= union_pixels / pixel_count <= 0.75),
        "seed_corners_are_exact_green_background": seed_corners_are_background,
        "seed_has_one_dominant_component": bool(discarded_seed_fraction <= 0.005),
        "intersection_over_seed_passed": bool(
            intersection_over_seed
            >= GROOMING_CANONICAL_SEED_BOUND_INTERSECTION_OVER_SEED_MIN
        ),
        "intersection_over_semantic_passed": bool(
            intersection_over_semantic
            >= GROOMING_CANONICAL_SEED_BOUND_INTERSECTION_OVER_SEMANTIC_MIN
        ),
        "semantic_to_seed_area_passed": bool(
            GROOMING_CANONICAL_SEED_BOUND_SEMANTIC_TO_SEED_MIN
            <= semantic_to_seed
            <= GROOMING_CANONICAL_SEED_BOUND_SEMANTIC_TO_SEED_MAX
        ),
        "seed_added_area_bounded": bool(
            added_over_seed <= GROOMING_CANONICAL_SEED_BOUND_ADDED_OVER_SEED_MAX
        ),
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    if failed_checks:
        raise ValueError(
            "grooming_seed_bound_matte_alignment_failed: " + ", ".join(failed_checks)
        )

    union_alpha = np.maximum(
        semantic_alpha,
        resized_seed_mask.astype(np.uint8) * 255,
    )
    output_rgba = np.empty((*source_rgb.shape[:2], 4), dtype=np.uint8)
    output_rgba[:, :, :3] = source_rgb
    output_rgba[:, :, 3] = union_alpha
    output_rgba[union_alpha == 0, :3] = 0
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output_rgba, mode="RGBA").save(
        target,
        format="PNG",
        optimize=True,
    )
    with Image.open(target) as loaded:
        persisted = np.asarray(loaded.convert("RGBA"), dtype=np.uint8)
    if not np.array_equal(persisted, output_rgba):
        raise RuntimeError("grooming_seed_bound_matte_changed_during_persistence")
    if not np.array_equal(
        persisted[:, :, :3][union_alpha > 0],
        source_rgb[union_alpha > 0],
    ):
        raise RuntimeError("grooming_seed_bound_matte_did_not_preserve_source_rgb")

    if green_target is not None:
        alpha_float = union_alpha[:, :, None].astype(np.float32) / 255.0
        green_rgb = np.empty_like(source_rgb)
        green_rgb[:, :] = pure_green
        green_composite = source_rgb.astype(np.float32) * alpha_float + green_rgb.astype(
            np.float32
        ) * (1.0 - alpha_float)
        green_target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(
            np.clip(np.rint(green_composite), 0, 255).astype(np.uint8),
            mode="RGB",
        ).save(green_target, format="PNG", optimize=True)

    def mask_bbox(mask: np.ndarray) -> list[int]:
        ys, xs = np.nonzero(mask)
        return [
            int(xs.min()),
            int(ys.min()),
            int(xs.max()) + 1,
            int(ys.max()) + 1,
        ]

    return {
        "policy": GROOMING_CANONICAL_SEED_BOUND_MATTE_POLICY,
        "acceptable": True,
        "fail_closed": True,
        "source_path": str(source),
        "source_sha256": source_digest,
        "source_mode": source_mode,
        "source_size": list(source_image.size),
        "semantic_cutout_path": str(semantic_cutout),
        "semantic_cutout_sha256": semantic_digest,
        "pose_transfer_seed_path": str(pose_transfer_seed),
        "pose_transfer_seed_sha256": seed_digest,
        "pose_transfer_seed_size": list(seed_image.size),
        "target_path": str(target),
        "target_sha256": file_sha256(target),
        "green_target_path": str(green_target) if green_target is not None else None,
        "green_target_sha256": (
            file_sha256(green_target)
            if green_target is not None and green_target.is_file()
            else None
        ),
        "alpha_threshold": FurSurfaceThresholds().alpha_threshold,
        "semantic_foreground_pixels": semantic_pixels,
        "semantic_foreground_fraction": round(semantic_pixels / pixel_count, 6),
        "semantic_foreground_bbox": mask_bbox(semantic_mask),
        "seed_foreground_pixels": seed_pixels,
        "seed_foreground_fraction": round(seed_pixels / pixel_count, 6),
        "seed_foreground_bbox": mask_bbox(resized_seed_mask),
        "seed_discarded_component_fraction": round(discarded_seed_fraction, 8),
        "intersection_pixels": intersection_pixels,
        "intersection_over_seed": round(intersection_over_seed, 6),
        "intersection_over_semantic": round(intersection_over_semantic, 6),
        "semantic_to_seed_area_ratio": round(semantic_to_seed, 6),
        "seed_added_pixels": added_seed_pixels,
        "seed_added_over_seed": round(added_over_seed, 6),
        "union_foreground_pixels": union_pixels,
        "union_foreground_fraction": round(union_pixels / pixel_count, 6),
        "union_foreground_bbox": mask_bbox(union_mask),
        "semantic_alpha_sha256": hashlib.sha256(semantic_alpha.tobytes()).hexdigest(),
        "seed_alpha_sha256": hashlib.sha256(
            (resized_seed_mask.astype(np.uint8) * 255).tobytes()
        ).hexdigest(),
        "union_alpha_sha256": hashlib.sha256(union_alpha.tobytes()).hexdigest(),
        "checks": checks,
        "source_rgb_preserved_where_alpha_nonzero": True,
        "transparent_background_rgb_zeroed": True,
        "semantic_rgb_copied": False,
        "pose_transfer_seed_rgb_copied": False,
        "pose_transfer_seed_luma_or_chroma_copied": False,
        "pose_transfer_seed_used_for_alpha_geometry_only": True,
    }


def prepare_canonical_eating_identity_refine_canvas(
    source: Path,
    target: Path,
) -> dict[str, Any]:
    """Prepare an action-safe eating sampler for canonical repaint."""
    _sync_local_video_runtime(globals(), {"prepare_canonical_eating_identity_refine_canvas"})

    return prepare_canonical_action_identity_refine_canvas(
        source,
        target,
        action="eating",
        policy=EATING_CANONICAL_IDENTITY_REFINE_POLICY,
    )


def prepare_grooming_foreleg_repair_anatomy_guide(
    source: Path,
    target: Path,
) -> dict[str, Any]:
    """Freeze the public one-gray open-channel guide as repair Image 3."""
    _sync_local_video_runtime(globals(), {"prepare_grooming_foreleg_repair_anatomy_guide"})

    with Image.open(source) as loaded:
        image = loaded.convert("RGB")
    if image.size != (512, 512):
        raise ValueError("grooming repair anatomy guide must be 512x512")
    rgb = np.asarray(image, dtype=np.uint8)
    background = np.all(
        rgb == np.asarray((0, 255, 0), dtype=np.uint8),
        axis=2,
    )
    foreground = np.all(
        rgb
        == np.asarray(GROOMING_QWEN_GUIDE_FOREGROUND_RGB, dtype=np.uint8),
        axis=2,
    )
    if not np.all(background | foreground):
        # Compatibility for queued legacy geometry-first jobs: their repair
        # Image 3 remains the already-neutral private CV map. New canonical
        # jobs never take this branch.
        red = rgb[:, :, 0].astype(np.int16)
        green = rgb[:, :, 1].astype(np.int16)
        blue = rgb[:, :, 2].astype(np.int16)
        legacy_background = (
            (green > 80)
            & ((green - np.maximum(red, blue)) > 35)
            & (green > red * 1.18)
            & (green > blue * 1.18)
        )
        legacy_foreground = ~legacy_background
        legacy_foreground_pixels = int(np.count_nonzero(legacy_foreground))
        channel_spread = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(
            np.int16
        )
        achromatic_fraction = float(
            np.count_nonzero(legacy_foreground & (channel_spread <= 3))
        ) / float(max(1, legacy_foreground_pixels))
        if (
            not 4096
            <= legacy_foreground_pixels
            <= int(512 * 512 * 0.90)
            or achromatic_fraction < 0.995
        ):
            raise RuntimeError(
                "Grooming repair anatomy guide contains appearance-bearing pixels."
            )
        source_sha256 = file_sha256(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        target_sha256 = file_sha256(target)
        if target_sha256 != source_sha256:
            raise RuntimeError(
                "Grooming repair anatomy guide digest changed during copy."
            )
        return {
            "policy": GROOMING_QWEN_FORELEG_REPAIR_ANATOMY_POLICY,
            "source_path": str(source),
            "source_sha256": source_sha256,
            "target_path": str(target),
            "target_sha256": target_sha256,
            "foreground_pixels": legacy_foreground_pixels,
            "foreground_achromatic_fraction": round(achromatic_fraction, 6),
            "internal_open_channel_pixels": None,
            "unique_foreground_rgb_count": int(
                np.unique(rgb[legacy_foreground], axis=0).shape[0]
            ),
            "original_profile_rgb_preserved": False,
            "original_profile_luma_values_preserved": False,
            "legacy_private_cv_compatibility": True,
            "role": "image3_pose_anatomy_only",
        }
    foreground_pixels = int(np.count_nonzero(foreground))
    if not 4096 <= foreground_pixels <= int(512 * 512 * 0.90):
        raise RuntimeError(
            "Grooming repair anatomy guide has an invalid foreground mask."
        )
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        background.astype(np.uint8),
        connectivity=8,
    )
    border_labels = set(
        np.unique(
            np.concatenate(
                [labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]
            )
        ).tolist()
    )
    internal_labels = [
        label
        for label in range(1, component_count)
        if label not in border_labels
    ]
    if len(internal_labels) != 1:
        raise RuntimeError(
            "Grooming repair anatomy guide lacks one open-curve interior channel."
        )
    internal_channel_pixels = int(stats[internal_labels[0], cv2.CC_STAT_AREA])
    if not 300 <= internal_channel_pixels <= 1800:
        raise RuntimeError(
            "Grooming repair anatomy guide interior channel has invalid area."
        )
    source_sha256 = file_sha256(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    target_sha256 = file_sha256(target)
    if target_sha256 != source_sha256:
        raise RuntimeError("Grooming repair anatomy guide digest changed during copy.")
    return {
        "policy": GROOMING_QWEN_FORELEG_REPAIR_ANATOMY_POLICY,
        "source_path": str(source),
        "source_sha256": source_sha256,
        "target_path": str(target),
        "target_sha256": target_sha256,
        "foreground_pixels": foreground_pixels,
        "foreground_achromatic_fraction": 1.0,
        "internal_open_channel_pixels": internal_channel_pixels,
        "unique_foreground_rgb_count": 1,
        "original_profile_rgb_preserved": False,
        "original_profile_luma_values_preserved": False,
        "role": "image3_single_tone_open_channel_action_geometry_only",
    }


def prepare_grooming_full_pose_recovery_guide(
    public_geometry_source: Path,
    private_geometry_source: Path,
    semantic_foreleg_source: Path,
    canonical_reference: Path,
    target: Path,
) -> dict[str, Any]:
    """Freeze the public one-gray open-channel guide as rescue Image 1.

    The earlier rescue exposed compressed private luma and Qwen interpreted its
    low-frequency regions as a tuxedo coat while still omitting the raised paw.
    Rescue now receives exactly the same appearance-free public geometry used
    by direct attempts: one synthetic gray plus a source-anchor-derived green
    open curve.  The private CV guide and legacy foreleg semantic are validated
    and digest-bound for auditing only; neither contributes a visible pixel.
    """
    _sync_local_video_runtime(globals(), {"prepare_grooming_full_pose_recovery_guide"})

    def load_rgb(path: Path, label: str) -> np.ndarray:
        if not path.is_file():
            raise FileNotFoundError(f"{label} is missing: {path}")
        with Image.open(path) as loaded:
            image = loaded.convert("RGB")
        if image.size != (512, 512):
            raise ValueError(f"{label} must be 512x512")
        return np.asarray(image, dtype=np.uint8)

    def green_background(rgb: np.ndarray) -> np.ndarray:
        red = rgb[:, :, 0].astype(np.int16)
        green = rgb[:, :, 1].astype(np.int16)
        blue = rgb[:, :, 2].astype(np.int16)
        return (
            (green > 80)
            & ((green - np.maximum(red, blue)) > 35)
            & (green > red * 1.18)
            & (green > blue * 1.18)
        )

    public_rgb = load_rgb(public_geometry_source, "public grooming geometry")
    private_rgb = load_rgb(private_geometry_source, "private grooming geometry")
    semantic_rgb = load_rgb(semantic_foreleg_source, "grooming foreleg semantic")
    canonical_rgb = load_rgb(canonical_reference, "approved canonical reference")

    public_background = np.all(public_rgb == (0, 255, 0), axis=2)
    public_foreground = np.all(
        public_rgb
        == np.asarray(GROOMING_QWEN_GUIDE_FOREGROUND_RGB, dtype=np.uint8),
        axis=2,
    )
    if not np.all(public_background | public_foreground):
        raise RuntimeError(
            "Grooming full-pose public guide contains appearance-bearing pixels."
        )
    foreground_pixels = int(np.count_nonzero(public_foreground))
    if not 4096 <= foreground_pixels <= int(512 * 512 * 0.90):
        raise RuntimeError("Grooming full-pose public guide foreground is invalid.")
    private_foreground = ~green_background(private_rgb)
    private_foreground_pixels = int(np.count_nonzero(private_foreground))
    removed_foreground = private_foreground & ~public_foreground
    removed_foreground_pixels = int(np.count_nonzero(removed_foreground))
    if (
        private_foreground_pixels < foreground_pixels
        or np.any(public_foreground & ~private_foreground)
        or not 300 <= removed_foreground_pixels <= 1800
    ):
        raise RuntimeError(
            "Grooming full-pose public/private open-channel contract changed."
        )
    background_components, background_labels, background_stats, _ = (
        cv2.connectedComponentsWithStats(
            public_background.astype(np.uint8),
            connectivity=8,
        )
    )
    border_labels = set(
        np.unique(
            np.concatenate(
                [
                    background_labels[0, :],
                    background_labels[-1, :],
                    background_labels[:, 0],
                    background_labels[:, -1],
                ]
            )
        ).tolist()
    )
    internal_labels = [
        label
        for label in range(1, background_components)
        if label not in border_labels
    ]
    if len(internal_labels) != 1:
        raise RuntimeError(
            "Grooming full-pose public guide lacks one interior open curve."
        )
    internal_channel_pixels = int(
        background_stats[internal_labels[0], cv2.CC_STAT_AREA]
    )
    if not 300 <= internal_channel_pixels <= 1800:
        raise RuntimeError(
            "Grooming full-pose public open channel has invalid area."
        )
    semantic_foreleg = (
        np.all(
            semantic_rgb == np.asarray(GROOMING_QWEN_GUIDE_FORELEG_RGB, dtype=np.uint8),
            axis=2,
        )
        & private_foreground
    )
    semantic_foreleg_pixels = int(np.count_nonzero(semantic_foreleg))
    if not 650 <= semantic_foreleg_pixels <= 6000:
        raise RuntimeError(
            "Grooming full-pose guide lacks a valid raised-foreleg semantic."
        )

    canonical_foreground = ~green_background(canonical_rgb)
    canonical_foreground_pixels = int(np.count_nonzero(canonical_foreground))
    # The canonical has already passed the production foreground gate.  This
    # helper only needs enough pixels for a stable median-luma estimate; keeping
    # the local minimum small also supports compact deterministic test fixtures.
    if not 256 <= canonical_foreground_pixels <= int(512 * 512 * 0.90):
        raise RuntimeError(
            "Grooming full-pose guide lacks a valid canonical foreground."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(public_geometry_source, target)
    with Image.open(target) as loaded:
        persisted = np.asarray(loaded.convert("RGB"), dtype=np.uint8)
    if (
        not np.array_equal(persisted, public_rgb)
        or file_sha256(target) != file_sha256(public_geometry_source)
    ):
        raise RuntimeError(
            "Grooming full-pose sampler changed the public open-channel guide."
        )

    return {
        "policy": GROOMING_CANONICAL_FULL_POSE_GUIDE_POLICY,
        "public_geometry_source": str(public_geometry_source),
        "public_geometry_sha256": file_sha256(public_geometry_source),
        "private_geometry_source": str(private_geometry_source),
        "private_geometry_sha256": file_sha256(private_geometry_source),
        "semantic_foreleg_source": str(semantic_foreleg_source),
        "semantic_foreleg_source_sha256": file_sha256(semantic_foreleg_source),
        "canonical_reference": str(canonical_reference),
        "canonical_reference_sha256": file_sha256(canonical_reference),
        "foreground_pixels": foreground_pixels,
        "private_foreground_pixels": private_foreground_pixels,
        "removed_foreground_pixels": removed_foreground_pixels,
        "internal_open_channel_pixels": internal_channel_pixels,
        "semantic_foreleg_pixels": semantic_foreleg_pixels,
        "semantic_foreleg_exposed_as_discrete_label": False,
        "canonical_foreground_pixels": canonical_foreground_pixels,
        "single_foreground_rgb": list(GROOMING_QWEN_GUIDE_FOREGROUND_RGB),
        "unique_foreground_rgb_count": 1,
        "open_curve_exposed": True,
        "open_curve_is_background_negative_space": True,
        "private_luma_exposed": False,
        "private_raw_luma_values_preserved": False,
        "spatial_canonical_pixels_copied": False,
        "main_case_rgb_or_luma_preserved": False,
        "target_path": str(target),
        "target_sha256": file_sha256(target),
        "role": "image1_single_tone_open_channel_action_geometry_only",
    }
