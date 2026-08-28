"""Eating and grooming geometry-guide construction."""

from __future__ import annotations

# The extracted functions resolve the facade dependencies after runtime sync.
# ruff: noqa: F821

from .runtime import sync_local_video_runtime as _sync_local_video_runtime


def available_scail_run_id(job_id: str, action: str) -> str:
    """Return a stable SCAIL run id without overwriting an earlier retry."""
    _sync_local_video_runtime(globals(), {"available_scail_run_id"})
    base = f"{job_id}-{action}"
    if not (SCAIL_EXPERIMENT_ROOT / base).exists():
        return base
    for retry_index in range(1, 1000):
        candidate = f"{base}-retry{retry_index:03d}"
        if not (SCAIL_EXPERIMENT_ROOT / candidate).exists():
            return candidate
    raise RuntimeError(
        f"No available SCAIL experiment directory remains for {job_id} {action}."
    )


def prepare_eating_qwen_geometry_guide(
    job_id: str,
    source: Path,
    target: Path,
    *,
    frames: int,
    fps: float,
) -> dict[str, Any]:
    """Derive and validate the texture-free eating Image 2 guide."""
    _sync_local_video_runtime(globals(), {"prepare_eating_qwen_geometry_guide"})
    if not source.is_file():
        raise FileNotFoundError(f"Eating driving animation is missing: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = target.with_name(target.stem + "_metadata.json")
    log_path = OUTPUT_ROOT / job_id / "eating_geometry_guide.log"
    command = [
        str(SCAIL_PYTHON),
        str(SCAIL_RUNNER),
        "--source",
        str(source),
        "--action",
        "eating",
        "--width",
        "512",
        "--height",
        "512",
        "--frames",
        str(frames),
        "--fps",
        str(fps),
        "--eating-geometry-guide-output",
        str(target),
        "--eating-geometry-guide-metadata",
        str(metadata_path),
    ]
    run_cancelable_process(
        job_id,
        command,
        log_path=log_path,
        stage_name="eating Qwen geometry-guide preparation",
        timeout=300,
    )
    if not target.is_file() or not metadata_path.is_file():
        raise RuntimeError(
            "Eating geometry guide completed without its PNG or metadata."
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Eating geometry guide metadata is invalid.") from exc
    if not isinstance(metadata, dict):
        raise RuntimeError("Eating geometry guide metadata must be an object.")
    expected_source_sha256 = file_sha256(source)
    if metadata.get("source_sha256") != expected_source_sha256:
        raise RuntimeError("Eating geometry guide source digest does not match.")
    if int(metadata.get("sampled_frame_count", -1)) != int(frames):
        raise RuntimeError("Eating geometry guide frame count does not match.")
    if abs(float(metadata.get("sampled_fps", -1.0)) - float(fps)) > 1e-6:
        raise RuntimeError("Eating geometry guide frame rate does not match.")
    if not 0 <= int(metadata.get("selected_frame_index", -1)) < int(frames):
        raise RuntimeError("Eating geometry guide selected frame is invalid.")
    if metadata.get("policy") != "driving_eating_geometry_image2_v7":
        raise RuntimeError("Eating geometry guide policy is unsupported.")
    if metadata.get("method") != (
        "temporal_head_lowering_flat_subject_semantic_food_fields_v8"
    ):
        raise RuntimeError("Eating geometry guide method is unsupported.")
    if not 0 <= int(metadata.get("muzzle_bowl_gap_px", -1)) <= 4:
        raise RuntimeError("Eating geometry guide did not reach the food bowl.")
    if int(metadata.get("contact_pixels", 0)) < 24:
        raise RuntimeError("Eating geometry guide contact region is too small.")
    muzzle_contact = metadata.get("muzzle_contact_semantic")
    if (
        not isinstance(muzzle_contact, dict)
        or muzzle_contact.get("method")
        != "source_anchored_continuous_neutral_muzzle_tip_v2"
        or not 0 <= int(muzzle_contact.get("muzzle_food_gap_px", -1)) <= 4
        or muzzle_contact.get("artificial_shape_added") is not False
        or muzzle_contact.get("detached_marker_added") is not False
        or muzzle_contact.get("source_rgb_preserved") is not False
        or muzzle_contact.get("source_chroma_preserved") is not False
        or muzzle_contact.get("identity_preserved") is not False
        or muzzle_contact.get("passed") is not True
    ):
        raise RuntimeError(
            "Eating geometry guide lacks verified muzzle-to-food overlap semantics."
        )
    semantic_food = metadata.get("semantic_food")
    if (
        not isinstance(semantic_food, dict)
        or semantic_food.get("method")
        != "fixed_beige_bowl_brown_kibble_from_source_mask_v1"
        or int(semantic_food.get("minimum_food_core_pixels", 0)) < 24
        or int(semantic_food.get("food_core_pixels", 0))
        < int(semantic_food.get("minimum_food_core_pixels", 0))
        or semantic_food.get("source_rgb_preserved") is not False
        or semantic_food.get("semantic_chroma_only") is not True
        or semantic_food.get("passed") is not True
    ):
        raise RuntimeError("Eating geometry guide lacks verified semantic food.")
    head_lowering = metadata.get("head_lowering_selection")
    if (
        not isinstance(head_lowering, dict)
        or head_lowering.get("method") != "central_upper_subject_row_p10_v1"
        or int(head_lowering.get("selected_head_support_pixels", 0)) <= 0
        or float(head_lowering.get("head_y_p10_span", -1.0))
        < float(head_lowering.get("head_y_p10_span_min", 0.0))
        or head_lowering.get("passed") is not True
    ):
        raise RuntimeError("Eating geometry guide lacks verified head lowering.")
    outline_smoothing = metadata.get("outline_smoothing")
    if (
        not isinstance(outline_smoothing, dict)
        or not 0.0 < float(outline_smoothing.get("perimeter_ratio", -1.0)) <= 1.01
        or outline_smoothing.get("high_frequency_outline_preserved") is not False
    ):
        raise RuntimeError("Eating geometry guide retained source outline texture.")
    semantic_structure = metadata.get("semantic_structure")
    if (
        not isinstance(semantic_structure, dict)
        or semantic_structure.get("method")
        != "flat_uniform_subject_silhouette_v3"
        or semantic_structure.get("subject_rgb") != [132, 132, 132]
        or int(semantic_structure.get("subject_unique_rgb_count", -1)) != 1
        or semantic_structure.get("internal_luma_variation") is not False
        or semantic_structure.get("appearance_bearing_face_detail") is not False
        or semantic_structure.get("source_luma_values_preserved") is not False
        or semantic_structure.get("source_rgb_preserved") is not False
        or semantic_structure.get("source_chroma_preserved") is not False
        or semantic_structure.get("high_frequency_texture_preserved") is not False
    ):
        raise RuntimeError(
            "Eating geometry guide retained source identity or texture structure."
        )
    palette = metadata.get("palette")
    expected_prop_colors = {
        "bowl": [214, 208, 196],
        "bowl_rim": [232, 228, 218],
        "food": [142, 86, 45],
    }
    if (
        not isinstance(palette, dict)
        or palette.get("background") != [0, 255, 0]
        or palette.get("subject_rgb") != [132, 132, 132]
        or palette.get("prop_semantic_colors") != expected_prop_colors
        or palette.get("source_rgb_preserved") is not False
        or palette.get("source_chroma_preserved") is not False
        or palette.get("semantic_prop_chroma_preserved") is not True
        or palette.get("source_luma_values_preserved") is not False
        or palette.get("high_frequency_texture_preserved") is not False
    ):
        raise RuntimeError("Eating geometry guide retained source identity colors.")

    with Image.open(target) as loaded:
        rgb = np.asarray(loaded.convert("RGB"), dtype=np.uint8)
    if rgb.shape != (512, 512, 3):
        raise RuntimeError("Eating geometry guide must be exactly 512x512 RGB.")
    green = np.all(rgb == (0, 255, 0), axis=2)
    foreground = ~green
    foreground_fraction = float(np.count_nonzero(foreground)) / float(green.size)
    if not 0.02 <= foreground_fraction <= 0.80:
        raise RuntimeError(
            "Eating geometry guide foreground fraction is invalid: "
            f"{foreground_fraction:.4f}"
        )
    foreground_rgb = rgb[foreground]
    grayscale = np.all(foreground_rgb == foreground_rgb[:, :1], axis=1)
    allowed_semantic = np.zeros(foreground_rgb.shape[0], dtype=bool)
    for color in expected_prop_colors.values():
        allowed_semantic |= np.all(
            foreground_rgb == np.asarray(color, dtype=np.uint8),
            axis=1,
        )
    if np.any(~(grayscale | allowed_semantic)):
        raise RuntimeError(
            "Eating geometry guide contains unsupported chromatic identity color."
        )
    grayscale_values = foreground_rgb[grayscale, 0]
    if (
        grayscale_values.size == 0
        or np.any(grayscale_values != 132)
        or int(np.unique(grayscale_values).size) != 1
        or int(grayscale_values.size) != int(metadata.get("subject_pixels", -1))
        or int(grayscale_values.size) < 500
    ):
        raise RuntimeError("Eating geometry guide subject field is not flat neutral gray.")
    prop_pixel_count = 0
    for name, color in expected_prop_colors.items():
        color_count = int(
            np.count_nonzero(np.all(rgb == np.asarray(color, dtype=np.uint8), axis=2))
        )
        prop_pixel_count += color_count
        if color_count == 0:
            raise RuntimeError(
                f"Eating geometry guide lacks its {name} semantic color."
            )
        if name == "food" and color_count != int(semantic_food["food_core_pixels"]):
            raise RuntimeError(
                "Eating geometry guide food pixels do not match metadata."
            )
    if (
        prop_pixel_count != int(metadata.get("bowl_pixels", -1))
        or prop_pixel_count < 100
    ):
        raise RuntimeError("Eating geometry guide prop pixels do not match metadata.")
    metadata["validated_foreground_fraction"] = round(foreground_fraction, 6)
    metadata["guide_sha256"] = file_sha256(target)
    metadata["metadata_path"] = str(metadata_path)
    return metadata


def _prepare_grooming_qwen_geometry_guide_legacy_two_tone(
    job_id: str,
    source: Path,
    target: Path,
    *,
    frames: int,
    fps: float,
) -> dict[str, Any]:
    """Build an appearance-free Qwen guide plus a private CV gate guide."""
    _sync_local_video_runtime(globals(), {"_prepare_grooming_qwen_geometry_guide_legacy_two_tone"})
    if not source.is_file():
        raise FileNotFoundError(f"Grooming driving animation is missing: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = target.with_name(target.stem + "_metadata.json")
    gate_target = target.with_name(target.stem + "_cv_gate.png")
    gate_metadata_path = gate_target.with_name(gate_target.stem + "_metadata.json")
    log_path = OUTPUT_ROOT / job_id / "grooming_geometry_guide.log"
    command = [
        str(SCAIL_PYTHON),
        str(SCAIL_RUNNER),
        "--source",
        str(source),
        "--action",
        "grooming",
        "--width",
        "512",
        "--height",
        "512",
        "--frames",
        str(frames),
        "--fps",
        str(fps),
        "--grooming-geometry-guide-output",
        str(gate_target),
        "--grooming-geometry-guide-metadata",
        str(gate_metadata_path),
    ]
    run_cancelable_process(
        job_id,
        command,
        log_path=log_path,
        stage_name="grooming Qwen geometry-guide preparation",
        timeout=300,
    )
    if not gate_target.is_file() or not gate_metadata_path.is_file():
        raise RuntimeError(
            "Grooming private pose-gate guide completed without its PNG or metadata."
        )
    try:
        gate_metadata = json.loads(gate_metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Grooming private pose-gate metadata is invalid.") from exc
    if not isinstance(gate_metadata, dict):
        raise RuntimeError("Grooming private pose-gate metadata must be an object.")
    expected_source_sha256 = file_sha256(source)
    if gate_metadata.get("source_sha256") != expected_source_sha256:
        raise RuntimeError("Grooming private pose-gate source digest does not match.")
    if gate_metadata.get("policy") != GROOMING_PRIVATE_POSE_GATE_GUIDE_POLICY:
        raise RuntimeError("Grooming private pose-gate guide policy is unsupported.")
    palette = gate_metadata.get("palette")
    if (
        not isinstance(palette, dict)
        or palette.get("source_rgb_preserved") is not False
    ):
        raise RuntimeError(
            "Grooming private pose-gate guide retained source identity colors."
        )
    relation = gate_metadata.get("raised_paw_muzzle_relation")
    if not isinstance(relation, dict) or relation.get("passed") is not True:
        raise RuntimeError(
            "Grooming private pose-gate metadata lacks verified paw/muzzle geometry."
        )
    white_maquette_selection = gate_metadata.get("white_maquette_selection")
    source_white_maquette = bool(gate_metadata.get("source_white_maquette"))
    if "white_model" in source.stem.lower() and (
        not source_white_maquette
        or not isinstance(white_maquette_selection, dict)
        or white_maquette_selection.get("policy")
        != "midcycle_relation_gated_white_maquette_v1"
        or int(white_maquette_selection.get("selected_frame_index", -1))
        != int(gate_metadata.get("selected_frame_index", -2))
    ):
        raise RuntimeError(
            "Grooming white-maquette guide lacks its relation-gated frame "
            "selection provenance."
        )

    with Image.open(gate_target) as loaded:
        gate_rgb = np.asarray(loaded.convert("RGB"), dtype=np.uint8)
    if gate_rgb.shape != (512, 512, 3):
        raise RuntimeError(
            "Grooming private pose-gate guide must be exactly 512x512 RGB."
        )
    green = np.all(gate_rgb == (0, 255, 0), axis=2)
    foreground = ~green
    foreground_fraction = float(np.count_nonzero(foreground)) / float(green.size)
    if not 0.02 <= foreground_fraction <= 0.80:
        raise RuntimeError(
            "Grooming private pose-gate guide foreground fraction is invalid: "
            f"{foreground_fraction:.4f}"
        )
    if np.any(gate_rgb[..., 0][foreground] != gate_rgb[..., 1][foreground]) or np.any(
        gate_rgb[..., 1][foreground] != gate_rgb[..., 2][foreground]
    ):
        raise RuntimeError(
            "Grooming private pose-gate guide contains non-neutral identity color."
        )

    # The runner's compressed luma field remains useful to the deterministic CV
    # gate, but even low-frequency source relief can make Qwen repaint the
    # maquette's face, surface, or shading. Qwen therefore receives a hard-coded body
    # gray plus a second hard-coded gray capsule for the raised foreleg. The
    # capsule is generated only from the verified shoulder, paw and foreleg-box
    # geometry; no source RGB/luma sample selects either value. A narrow green
    # channel on the torso side makes the internal foreleg boundary readable.
    qwen_rgb = np.empty_like(gate_rgb)
    qwen_rgb[:, :] = (0, 255, 0)
    qwen_rgb[foreground] = GROOMING_QWEN_GUIDE_FOREGROUND_RGB
    shoulder_anchor = relation.get("shoulder_anchor")
    paw_center = relation.get("paw_center")
    raised_foreleg_bbox = relation.get("raised_foreleg_region_bbox")
    if (
        not isinstance(shoulder_anchor, list)
        or len(shoulder_anchor) != 2
        or not isinstance(paw_center, list)
        or len(paw_center) != 2
        or not isinstance(raised_foreleg_bbox, list)
        or len(raised_foreleg_bbox) != 4
    ):
        raise RuntimeError(
            "Grooming private pose-gate metadata lacks foreleg channel geometry."
        )
    shoulder = np.asarray(shoulder_anchor, dtype=np.float64)
    paw = np.asarray(paw_center, dtype=np.float64)
    foreleg_bbox = np.asarray(raised_foreleg_bbox, dtype=np.float64)
    if (
        np.any(~np.isfinite(shoulder))
        or np.any(~np.isfinite(paw))
        or np.any(~np.isfinite(foreleg_bbox))
        or np.linalg.norm(paw - shoulder) < 24.0
        or np.any(shoulder < 0.0)
        or np.any(shoulder >= 512.0)
        or np.any(paw < 0.0)
        or np.any(paw >= 512.0)
    ):
        raise RuntimeError(
            "Grooming private pose-gate foreleg channel geometry is invalid."
        )
    foreground_x = np.flatnonzero(np.any(foreground, axis=0))
    if foreground_x.size == 0:
        raise RuntimeError("Grooming private pose-gate foreground is empty.")
    body_center_x = float(foreground_x.min() + foreground_x.max()) / 2.0
    foreleg_width = max(1.0, float(foreleg_bbox[2] - foreleg_bbox[0]))
    semantic_width = int(round(min(42.0, max(18.0, foreleg_width * 0.62))))
    paw_radius = int(round(min(24.0, max(11.0, foreleg_width * 0.38))))
    torso_side = 1.0 if body_center_x >= float((shoulder[0] + paw[0]) / 2.0) else -1.0
    foreleg_vector = paw - shoulder
    shoulder_px = tuple(np.clip(np.rint(shoulder), 0, 511).astype(int).tolist())
    paw_px = tuple(np.clip(np.rint(paw), 0, 511).astype(int).tolist())
    foreleg_semantic_mask = np.zeros(foreground.shape, dtype=np.uint8)
    cv2.line(
        foreleg_semantic_mask,
        shoulder_px,
        paw_px,
        255,
        thickness=semantic_width,
        lineType=cv2.LINE_8,
    )
    cv2.circle(
        foreleg_semantic_mask,
        paw_px,
        paw_radius,
        255,
        thickness=-1,
        lineType=cv2.LINE_8,
    )
    semantic_foreleg = (foreleg_semantic_mask > 0) & foreground
    raw_semantic_foreleg_pixels = int(np.count_nonzero(semantic_foreleg))
    if not 700 <= raw_semantic_foreleg_pixels <= 6000:
        raise RuntimeError(
            "Grooming Qwen synthetic foreleg semantic has invalid area: "
            f"{raw_semantic_foreleg_pixels}"
        )
    qwen_rgb[semantic_foreleg] = GROOMING_QWEN_GUIDE_FORELEG_RGB

    channel_offset = min(22.0, max(10.0, semantic_width * 0.55))
    channel_start = shoulder + foreleg_vector * 0.08
    channel_end = shoulder + foreleg_vector * 0.92
    channel_start[0] += torso_side * channel_offset
    channel_end[0] += torso_side * channel_offset
    channel_start_px = tuple(
        np.clip(np.rint(channel_start), 0, 511).astype(int).tolist()
    )
    channel_end_px = tuple(np.clip(np.rint(channel_end), 0, 511).astype(int).tolist())
    channel_mask = np.zeros(foreground.shape, dtype=np.uint8)
    cv2.line(
        channel_mask,
        channel_start_px,
        channel_end_px,
        255,
        thickness=5,
        lineType=cv2.LINE_8,
    )
    structural_negative_space = (channel_mask > 0) & foreground
    structural_negative_space_pixels = int(np.count_nonzero(structural_negative_space))
    if not 80 <= structural_negative_space_pixels <= 1600:
        raise RuntimeError(
            "Grooming Qwen foreleg negative-space channel has invalid area: "
            f"{structural_negative_space_pixels}"
        )
    qwen_rgb[structural_negative_space] = (0, 255, 0)
    semantic_foreleg_pixels = int(
        np.count_nonzero(semantic_foreleg & ~structural_negative_space)
    )
    if not 650 <= semantic_foreleg_pixels <= raw_semantic_foreleg_pixels:
        raise RuntimeError(
            "Grooming Qwen synthetic foreleg was erased by its separator."
        )
    Image.fromarray(qwen_rgb).save(target)

    with Image.open(target) as loaded:
        persisted_qwen_rgb = np.asarray(loaded.convert("RGB"), dtype=np.uint8)
    qwen_green = np.all(persisted_qwen_rgb == (0, 255, 0), axis=2)
    qwen_foreground = ~qwen_green
    private_foreground_pixels = int(np.count_nonzero(foreground))
    qwen_foreground_pixels = int(np.count_nonzero(qwen_foreground))
    qwen_foreground_fraction = float(qwen_foreground_pixels) / float(qwen_green.size)
    foreground_retained_fraction = float(qwen_foreground_pixels) / float(
        private_foreground_pixels
    )
    unique_qwen_foreground = np.unique(
        persisted_qwen_rgb[qwen_foreground],
        axis=0,
    )
    expected_foreground_palette = np.asarray(
        [
            GROOMING_QWEN_GUIDE_FOREGROUND_RGB,
            GROOMING_QWEN_GUIDE_FORELEG_RGB,
        ],
        dtype=np.uint8,
    )
    persisted_semantic_foreleg_pixels = int(
        np.count_nonzero(
            np.all(
                persisted_qwen_rgb
                == np.asarray(GROOMING_QWEN_GUIDE_FORELEG_RGB, dtype=np.uint8),
                axis=2,
            )
        )
    )
    if (
        persisted_qwen_rgb.shape != (512, 512, 3)
        or np.any(qwen_foreground & ~foreground)
        or not 0.97 <= foreground_retained_fraction < 1.0
        or unique_qwen_foreground.shape != (2, 3)
        or not np.array_equal(
            unique_qwen_foreground,
            expected_foreground_palette,
        )
        or persisted_semantic_foreleg_pixels != semantic_foreleg_pixels
    ):
        raise RuntimeError(
            "Grooming Qwen guide is not the synthetic two-tone geometry contract."
        )

    gate_guide_sha256 = file_sha256(gate_target)
    qwen_guide_sha256 = file_sha256(target)
    metadata: dict[str, Any] = {
        "policy": GROOMING_QWEN_GEOMETRY_GUIDE_POLICY,
        "method": GROOMING_QWEN_GEOMETRY_GUIDE_METHOD,
        "source_path": str(source),
        "source_sha256": expected_source_sha256,
        "sampled_frame_count": gate_metadata.get("sampled_frame_count"),
        "sampled_fps": gate_metadata.get("sampled_fps"),
        "selected_frame_index": gate_metadata.get("selected_frame_index"),
        "source_white_maquette": source_white_maquette,
        "white_maquette_selection": white_maquette_selection,
        "raised_paw_muzzle_relation": relation,
        "validated_foreground_fraction": round(qwen_foreground_fraction, 6),
        "guide_sha256": qwen_guide_sha256,
        "metadata_path": str(metadata_path),
        "qwen_guide": {
            "role": "image1_action_geometry_sampler",
            "path": str(target),
            "sha256": qwen_guide_sha256,
            "background_rgb": [0, 255, 0],
            "foreground_rgb": list(GROOMING_QWEN_GUIDE_FOREGROUND_RGB),
            "foreground_palette_rgb": [
                list(GROOMING_QWEN_GUIDE_FOREGROUND_RGB),
                list(GROOMING_QWEN_GUIDE_FORELEG_RGB),
            ],
            "foreground_pixels": qwen_foreground_pixels,
            "foreground_fraction": round(qwen_foreground_fraction, 6),
            "private_foreground_pixels": private_foreground_pixels,
            "foreground_retained_fraction": round(
                foreground_retained_fraction,
                6,
            ),
            "unique_foreground_rgb_count": 2,
            "internal_luma_variation": True,
            "internal_luma_variation_role": "synthetic_anatomy_labels_only",
            "appearance_bearing_internal_luma_variation": False,
            "source_rgb_preserved": False,
            "source_chroma_preserved": False,
            "source_luma_values_preserved": False,
            "main_case_appearance_exposed_to_qwen": False,
            "white_maquette_appearance_exposed_to_qwen": False,
            "synthetic_foreleg_semantic": {
                "method": GROOMING_QWEN_FORELEG_SEMANTIC_METHOD,
                "rgb": list(GROOMING_QWEN_GUIDE_FORELEG_RGB),
                "pixels": semantic_foreleg_pixels,
                "pixels_before_separator": raw_semantic_foreleg_pixels,
                "capsule_width": semantic_width,
                "paw_radius": paw_radius,
                "shoulder": list(shoulder_px),
                "paw_center": list(paw_px),
                "derived_from": ("verified_raised_paw_muzzle_relation_geometry_only"),
                "source_rgb_or_luma_sampled": False,
                "appearance_bearing": False,
            },
            "structural_negative_space": {
                "method": GROOMING_QWEN_NEGATIVE_SPACE_METHOD,
                "pixels": structural_negative_space_pixels,
                "line_width": 5,
                "start": list(channel_start_px),
                "end": list(channel_end_px),
                "derived_from": ("verified_raised_paw_muzzle_relation_geometry_only"),
                "appearance_bearing": False,
                "adds_foreground_color": False,
            },
        },
        "private_pose_gate": {
            "role": "cv_geometry_validation_only",
            "guide_path": str(gate_target),
            "guide_sha256": gate_guide_sha256,
            "metadata_path": str(gate_metadata_path),
            "policy": GROOMING_PRIVATE_POSE_GATE_GUIDE_POLICY,
            "exposed_to_qwen": False,
            "generator_metadata": gate_metadata,
        },
        "palette": {
            "background": [0, 255, 0],
            "subject_rgb": [
                list(GROOMING_QWEN_GUIDE_FOREGROUND_RGB),
                list(GROOMING_QWEN_GUIDE_FORELEG_RGB),
            ],
            "source_rgb_preserved": False,
            "source_chroma_preserved": False,
            "source_luma_values_preserved": False,
            "internal_luma_variation": True,
            "internal_luma_variation_role": "synthetic_anatomy_labels_only",
            "appearance_bearing_internal_luma_variation": False,
        },
    }
    write_json_atomic(metadata_path, metadata)
    return metadata


def build_grooming_open_foreleg_negative_space(
    foreground: np.ndarray,
    relation: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Carve one source-anchor-derived open curve between foreleg and torso.

    Only the verified shoulder, paw, foreleg-box geometry and the binary outer
    silhouette participate.  No Main_case RGB, luma, texture, or marking value
    is sampled.  The curve is deliberately open at both ends so the living
    foreleg remains attached at the shoulder and touches the muzzle; unlike the
    retired two-tone capsule it does not label or fill a detachable object.
    """
    _sync_local_video_runtime(globals(), {"build_grooming_open_foreleg_negative_space"})

    if foreground.shape != (512, 512) or foreground.dtype != np.bool_:
        raise RuntimeError(
            "Grooming open-channel construction requires a 512x512 binary silhouette."
        )
    shoulder_anchor = relation.get("shoulder_anchor")
    paw_center = relation.get("paw_center")
    raised_foreleg_bbox = relation.get("raised_foreleg_region_bbox")
    if (
        not isinstance(shoulder_anchor, list)
        or len(shoulder_anchor) != 2
        or not isinstance(paw_center, list)
        or len(paw_center) != 2
        or not isinstance(raised_foreleg_bbox, list)
        or len(raised_foreleg_bbox) != 4
    ):
        raise RuntimeError(
            "Grooming private pose-gate metadata lacks open-channel geometry."
        )
    shoulder = np.asarray(shoulder_anchor, dtype=np.float64)
    paw = np.asarray(paw_center, dtype=np.float64)
    foreleg_bbox = np.asarray(raised_foreleg_bbox, dtype=np.float64)
    foreleg_vector = shoulder - paw
    foreleg_length = float(np.linalg.norm(foreleg_vector))
    foreleg_width = float(foreleg_bbox[2] - foreleg_bbox[0])
    foreleg_height = float(foreleg_bbox[3] - foreleg_bbox[1])
    if (
        np.any(~np.isfinite(shoulder))
        or np.any(~np.isfinite(paw))
        or np.any(~np.isfinite(foreleg_bbox))
        or not 24.0 <= foreleg_length <= 220.0
        or not 18.0 <= foreleg_width <= 128.0
        or not 40.0 <= foreleg_height <= 240.0
        or np.any(shoulder < 0.0)
        or np.any(shoulder >= 512.0)
        or np.any(paw < 0.0)
        or np.any(paw >= 512.0)
    ):
        raise RuntimeError(
            "Grooming private pose-gate open-channel geometry is invalid."
        )
    foreground_x = np.flatnonzero(np.any(foreground, axis=0))
    if foreground_x.size == 0:
        raise RuntimeError("Grooming private pose-gate foreground is empty.")
    body_center_x = float(foreground_x.min() + foreground_x.max()) / 2.0
    anchor_center_x = float(shoulder[0] + paw[0]) / 2.0
    torso_side = 1.0 if body_center_x >= anchor_center_x else -1.0

    # The varying offset makes a shallow organic C/J curve rather than the old
    # uniform straight separator.  Endpoints stop short of paw and shoulder so
    # Qwen receives explicit mouth contact and shoulder attachment.
    samples = (
        (0.14, 0.35),
        (0.28, 0.50),
        (0.46, 0.59),
        (0.64, 0.54),
        (0.80, 0.39),
        (0.90, 0.20),
    )
    points: list[list[int]] = []
    for position, width_scale in samples:
        point = paw + foreleg_vector * position
        point[0] += torso_side * foreleg_width * width_scale
        point_px = np.clip(np.rint(point), 0, 511).astype(np.int32)
        points.append([int(point_px[0]), int(point_px[1])])
    line_width = int(round(min(11.0, max(7.0, foreleg_width * 0.20))))
    curve = np.zeros(foreground.shape, dtype=np.uint8)
    cv2.polylines(
        curve,
        [np.asarray(points, dtype=np.int32)],
        False,
        255,
        thickness=line_width,
        lineType=cv2.LINE_AA,
    )
    negative_space = (curve >= 64) & foreground
    negative_space_pixels = int(np.count_nonzero(negative_space))
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(
        negative_space.astype(np.uint8),
        connectivity=8,
    )
    if component_count != 2:
        raise RuntimeError(
            "Grooming public foreleg negative-space curve is not one component."
        )
    curve_bbox = [
        int(stats[1, cv2.CC_STAT_LEFT]),
        int(stats[1, cv2.CC_STAT_TOP]),
        int(stats[1, cv2.CC_STAT_LEFT] + stats[1, cv2.CC_STAT_WIDTH]),
        int(stats[1, cv2.CC_STAT_TOP] + stats[1, cv2.CC_STAT_HEIGHT]),
    ]
    curve_span = float(curve_bbox[3] - curve_bbox[1]) / foreleg_length
    if (
        not 300 <= negative_space_pixels <= 1800
        or not 0.45 <= curve_span <= 1.15
        or np.any(negative_space & ~foreground)
    ):
        raise RuntimeError(
            "Grooming public foreleg negative-space curve failed its geometry bounds."
        )
    return negative_space, {
        "method": GROOMING_QWEN_NEGATIVE_SPACE_METHOD,
        "pixels": negative_space_pixels,
        "line_width": line_width,
        "points": points,
        "bbox": curve_bbox,
        "span_vs_paw_shoulder": round(curve_span, 6),
        "curve_closed": False,
        "paw_endpoint_open": True,
        "shoulder_endpoint_open": True,
        "foreleg_remains_attached_at_shoulder": True,
        "paw_muzzle_contact_remains_connected": True,
        "derived_from": "verified_raised_paw_muzzle_relation_geometry_only",
        "source_rgb_or_luma_sampled": False,
        "appearance_bearing": False,
        "adds_foreground_color": False,
    }


def prepare_grooming_qwen_geometry_guide(
    job_id: str,
    source: Path,
    target: Path,
    *,
    frames: int,
    fps: float,
) -> dict[str, Any]:
    """Build a one-gray grooming guide with an organic open foreleg channel.

    The retired two-tone guide drew a closed light-gray capsule and encouraged
    Qwen to render a microphone.  The subsequent solid silhouette removed that
    artifact but also erased the internal raised-foreleg boundary, repeatedly
    yielding a neutral two-paws-down sit.  Production now exposes one synthetic
    gray only, with a narrow pure-green open curve between the source-derived
    paw/foreleg and torso.  The curve carries geometry, never appearance, and
    remains open at paw and shoulder.  Rich CV luma and the old completion
    semantic stay digest-bound private inputs.
    """
    _sync_local_video_runtime(globals(), {"prepare_grooming_qwen_geometry_guide"})

    metadata = _prepare_grooming_qwen_geometry_guide_legacy_two_tone(
        job_id,
        source,
        target,
        frames=frames,
        fps=fps,
    )
    completion_mask_path = target.with_name(target.stem + "_completion_mask.png")
    shutil.copy2(target, completion_mask_path)
    completion_mask_sha256 = file_sha256(completion_mask_path)
    with Image.open(completion_mask_path) as loaded:
        completion_rgb = np.asarray(loaded.convert("RGB"), dtype=np.uint8)
    completion_foreleg_pixels = int(
        np.count_nonzero(
            np.all(
                completion_rgb
                == np.asarray(GROOMING_QWEN_GUIDE_FORELEG_RGB, dtype=np.uint8),
                axis=2,
            )
        )
    )
    if not 650 <= completion_foreleg_pixels <= 6000:
        raise RuntimeError(
            "Grooming private completion mask lacks its foreleg semantic."
        )
    private_gate = metadata.get("private_pose_gate")
    if not isinstance(private_gate, dict):
        raise RuntimeError("Grooming geometry preparation lacks its private CV guide.")
    private_path_text = str(private_gate.get("guide_path") or "").strip()
    private_path = Path(private_path_text) if private_path_text else Path()
    if not private_path.is_file():
        raise RuntimeError("Grooming private CV guide disappeared.")
    with Image.open(private_path) as loaded:
        private_rgb = np.asarray(loaded.convert("RGB"), dtype=np.uint8)
    if private_rgb.shape != (512, 512, 3):
        raise RuntimeError("Grooming private CV guide dimensions changed.")
    private_foreground = ~np.all(private_rgb == (0, 255, 0), axis=2)
    foreground_pixels = int(np.count_nonzero(private_foreground))
    if foreground_pixels < 1024:
        raise RuntimeError("Grooming private CV guide foreground is empty.")

    relation = metadata.get("raised_paw_muzzle_relation")
    if not isinstance(relation, dict) or relation.get("passed") is not True:
        raise RuntimeError(
            "Grooming geometry preparation lacks verified paw/muzzle anchors."
        )
    structural_negative_space, structural_negative_space_metadata = (
        build_grooming_open_foreleg_negative_space(private_foreground, relation)
    )
    qwen_rgb = np.empty_like(private_rgb)
    qwen_rgb[:, :] = (0, 255, 0)
    qwen_foreground = private_foreground & ~structural_negative_space
    qwen_rgb[qwen_foreground] = GROOMING_QWEN_GUIDE_FOREGROUND_RGB
    Image.fromarray(qwen_rgb, mode="RGB").save(
        target,
        format="PNG",
        optimize=True,
    )
    guide_sha256 = file_sha256(target)
    metadata_path = target.with_name(target.stem + "_metadata.json")
    qwen_foreground_pixels = int(np.count_nonzero(qwen_foreground))
    removed_foreground_pixels = int(np.count_nonzero(structural_negative_space))
    foreground_fraction = float(qwen_foreground_pixels) / float(
        private_foreground.size
    )
    foreground_retained_fraction = float(qwen_foreground_pixels) / float(
        foreground_pixels
    )
    metadata.update(
        policy=GROOMING_QWEN_GEOMETRY_GUIDE_POLICY,
        method=GROOMING_QWEN_GEOMETRY_GUIDE_METHOD,
        validated_foreground_fraction=round(foreground_fraction, 6),
        guide_sha256=guide_sha256,
        metadata_path=str(metadata_path),
        private_completion_mask={
            "role": "local_noise_mask_derivation_only",
            "policy": GROOMING_PRIVATE_COMPLETION_MASK_POLICY,
            "path": str(completion_mask_path),
            "sha256": completion_mask_sha256,
            "exposed_to_qwen": False,
            "loaded_as_qwen_image": False,
            "semantic_foreleg_rgb": list(GROOMING_QWEN_GUIDE_FORELEG_RGB),
            "semantic_foreleg_pixels": completion_foreleg_pixels,
        },
        qwen_guide={
            "role": "image2_action_geometry_guide",
            "path": str(target),
            "sha256": guide_sha256,
            "background_rgb": [0, 255, 0],
            "foreground_rgb": list(GROOMING_QWEN_GUIDE_FOREGROUND_RGB),
            "foreground_palette_rgb": [list(GROOMING_QWEN_GUIDE_FOREGROUND_RGB)],
            "foreground_pixels": qwen_foreground_pixels,
            "foreground_fraction": round(foreground_fraction, 6),
            "private_foreground_pixels": foreground_pixels,
            "foreground_retained_fraction": round(
                foreground_retained_fraction,
                6,
            ),
            "unique_foreground_rgb_count": 1,
            "internal_luma_variation": False,
            "internal_luma_variation_role": "none",
            "appearance_bearing_internal_luma_variation": False,
            "source_rgb_preserved": False,
            "source_chroma_preserved": False,
            "source_luma_values_preserved": False,
            "main_case_appearance_exposed_to_qwen": False,
            "white_maquette_appearance_exposed_to_qwen": False,
            "synthetic_foreleg_semantic": None,
            "structural_negative_space": structural_negative_space_metadata,
            "closed_foreleg_capsule_exposed": False,
            "internal_green_separator_exposed": True,
            "raised_foreleg_validation_role": (
                "public_open_curve_geometry_plus_private_cv"
            ),
            "removed_foreground_pixels": removed_foreground_pixels,
        },
        palette={
            "background": [0, 255, 0],
            "subject_rgb": [list(GROOMING_QWEN_GUIDE_FOREGROUND_RGB)],
            "source_rgb_preserved": False,
            "source_chroma_preserved": False,
            "source_luma_values_preserved": False,
            "internal_luma_variation": False,
            "internal_luma_variation_role": "none",
            "appearance_bearing_internal_luma_variation": False,
        },
    )
    write_json_atomic(metadata_path, metadata)
    return metadata
