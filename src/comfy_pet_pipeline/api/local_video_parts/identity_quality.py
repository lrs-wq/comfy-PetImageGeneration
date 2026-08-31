"""Identity palette, chroma, and semantic recovery quality helpers."""

from __future__ import annotations

# The extracted functions resolve the facade dependencies after runtime sync.
# ruff: noqa: F821

from .runtime import sync_local_video_runtime as _sync_local_video_runtime


CANONICAL_INTERIOR_HOLE_GATE_POLICY = (
    "central_medial_overlap_component_v2"
)
CANONICAL_INTERIOR_VOID_MEDIAL_OVERLAP_FRACTION_MIN = 0.12


def require_ffmpeg() -> None:
    _sync_local_video_runtime(globals(), {"require_ffmpeg"})
    if not Path(FFMPEG_BIN).is_file():
        raise FileNotFoundError(f"FFmpeg executable was not found at {FFMPEG_BIN}.")


def run_ffmpeg(command: list[str], stage_name: str, timeout: int = 900) -> None:
    _sync_local_video_runtime(globals(), {"run_ffmpeg"})
    require_ffmpeg()
    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if process.returncode != 0:
        detail = (process.stderr or process.stdout or "unknown FFmpeg error").strip()
        raise RuntimeError(f"{stage_name} failed: {detail[-4000:]}")


def convert_chroma_image_to_transparent_png(
    source: Path,
    target: Path,
    *,
    require_clean_background: bool = False,
) -> dict[str, object]:
    _sync_local_video_runtime(globals(), {"convert_chroma_image_to_transparent_png"})
    return build_transparent_cutout(
        source,
        target,
        require_clean_background=require_clean_background,
    )


def canonical_cutout_interior_quality(path: Path) -> dict[str, Any]:
    """Reject a large chroma-carved void through a canonical pet's head/neck.

    A normal shoulder, waist, or haunch concavity can sit inside the convex hull
    and graze the broad medial corridor by a few pixels.  Treat a void as an
    interior defect only when a meaningful share of that same component crosses
    the corridor; side-only silhouette concavities must remain valid.
    """
    _sync_local_video_runtime(globals(), {"canonical_cutout_interior_quality"})

    with Image.open(path) as loaded:
        alpha = np.asarray(loaded.convert("RGBA").getchannel("A"))
    foreground = (alpha >= 64).astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        foreground,
        connectivity=8,
    )
    if component_count <= 1:
        return {
            "acceptable": False,
            "issues": ["canonical_subject_missing_after_chroma_key"],
            "foreground_pixels": 0,
        }
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    component = (labels == largest_label).astype(np.uint8)
    foreground_pixels = int(stats[largest_label, cv2.CC_STAT_AREA])
    x0 = int(stats[largest_label, cv2.CC_STAT_LEFT])
    y0 = int(stats[largest_label, cv2.CC_STAT_TOP])
    width = int(stats[largest_label, cv2.CC_STAT_WIDTH])
    height = int(stats[largest_label, cv2.CC_STAT_HEIGHT])
    contours, _ = cv2.findContours(
        component,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return {
            "acceptable": False,
            "issues": ["canonical_subject_contour_unavailable"],
            "foreground_pixels": foreground_pixels,
        }
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    hull_mask = np.zeros(component.shape, dtype=np.uint8)
    cv2.drawContours(hull_mask, [hull], -1, 1, thickness=-1)
    roi = np.zeros(component.shape, dtype=np.uint8)
    focus_x0 = x0 + int(round(width * 0.18))
    focus_x1 = x0 + int(round(width * 0.82))
    focus_y0 = y0 + int(round(height * 0.28))
    focus_y1 = y0 + int(round(height * 0.58))
    roi[focus_y0:focus_y1, focus_x0:focus_x1] = 1
    void = ((hull_mask > 0) & (component == 0) & (roi > 0)).astype(np.uint8)
    if min(void.shape) >= 24:
        void = cv2.morphologyEx(
            void,
            cv2.MORPH_OPEN,
            np.ones((3, 3), dtype=np.uint8),
        )
    void_count, void_labels, void_stats, _ = cv2.connectedComponentsWithStats(
        void,
        connectivity=8,
    )
    largest_void_area = 0
    largest_void_width = 0
    largest_void_height = 0
    largest_void_medial_overlap = 0
    largest_void_medial_overlap_fraction = 0.0
    ignored_side_concavity_count = 0
    largest_ignored_side_concavity_area = 0
    largest_ignored_side_concavity_medial_overlap_fraction = 0.0
    medial_x0 = x0 + int(round(width * 0.36))
    medial_x1 = x0 + int(round(width * 0.64))
    if void_count > 1:
        for void_label in range(1, void_count):
            medial_overlap = int(
                np.count_nonzero(
                    void_labels[focus_y0:focus_y1, medial_x0:medial_x1] == void_label
                )
            )
            # Outer silhouette concavities can legitimately appear inside the convex
            # hull.  A chroma-carved mouth/neck hole must cross the subject's medial
            # corridor, so side-only voids are ignored.
            if not medial_overlap:
                continue
            void_area = int(void_stats[void_label, cv2.CC_STAT_AREA])
            medial_overlap_fraction = medial_overlap / float(max(1, void_area))
            if (
                medial_overlap_fraction
                < CANONICAL_INTERIOR_VOID_MEDIAL_OVERLAP_FRACTION_MIN
            ):
                ignored_side_concavity_count += 1
                if void_area > largest_ignored_side_concavity_area:
                    largest_ignored_side_concavity_area = void_area
                    largest_ignored_side_concavity_medial_overlap_fraction = (
                        medial_overlap_fraction
                    )
                continue
            if void_area <= largest_void_area:
                continue
            largest_void_area = void_area
            largest_void_width = int(void_stats[void_label, cv2.CC_STAT_WIDTH])
            largest_void_height = int(void_stats[void_label, cv2.CC_STAT_HEIGHT])
            largest_void_medial_overlap = medial_overlap
            largest_void_medial_overlap_fraction = medial_overlap_fraction
    area_fraction = largest_void_area / float(max(1, foreground_pixels))
    width_ratio = largest_void_width / float(max(1, width))
    height_ratio = largest_void_height / float(max(1, height))
    invalid_void = bool(
        largest_void_area >= max(96, int(round(foreground_pixels * 0.004)))
        and width_ratio >= 0.08
        and height_ratio >= 0.015
    )
    return {
        "policy": CANONICAL_INTERIOR_HOLE_GATE_POLICY,
        "acceptable": not invalid_void,
        "issues": ["canonical_subject_interior_chroma_hole"] if invalid_void else [],
        "foreground_pixels": foreground_pixels,
        "focus_box": [focus_x0, focus_y0, focus_x1, focus_y1],
        "medial_box": [medial_x0, focus_y0, medial_x1, focus_y1],
        "largest_void_area": largest_void_area,
        "largest_void_medial_overlap": largest_void_medial_overlap,
        "largest_void_medial_overlap_fraction": round(
            largest_void_medial_overlap_fraction,
            6,
        ),
        "medial_overlap_fraction_min": (
            CANONICAL_INTERIOR_VOID_MEDIAL_OVERLAP_FRACTION_MIN
        ),
        "ignored_side_concavity_count": ignored_side_concavity_count,
        "largest_ignored_side_concavity_area": (
            largest_ignored_side_concavity_area
        ),
        "largest_ignored_side_concavity_medial_overlap_fraction": round(
            largest_ignored_side_concavity_medial_overlap_fraction,
            6,
        ),
        "largest_void_area_fraction": round(area_fraction, 6),
        "largest_void_width_ratio": round(width_ratio, 6),
        "largest_void_height_ratio": round(height_ratio, 6),
    }


def evaluate_chroma_cutout(
    source: Path,
    target: Path,
    *,
    validate_canonical_interior: bool = False,
) -> dict[str, Any]:
    _sync_local_video_runtime(globals(), {"evaluate_chroma_cutout"})
    metrics = dict(
        convert_chroma_image_to_transparent_png(
            source,
            target,
            require_clean_background=False,
        )
    )
    base_issues: list[str] = []
    if float(metrics["foreground_fraction"]) > 0.85:
        base_issues.append("foreground_fraction_too_large")
    if any(int(value) > 0 for value in metrics["corners"]):
        base_issues.append("nontransparent_canvas_corner")
    issues = list(base_issues)
    if validate_canonical_interior and not base_issues:
        interior_quality = canonical_cutout_interior_quality(target)
        metrics["canonical_interior_quality"] = interior_quality
        issues.extend(interior_quality["issues"])
    metrics["acceptable"] = not issues
    metrics["issues"] = issues
    metrics["method"] = "chroma_key"
    if base_issues:
        target.unlink(missing_ok=True)
    return metrics


def get_rembg_session() -> Any:
    _sync_local_video_runtime(globals(), {"get_rembg_session"})
    global REMBG_SESSION
    with REMBG_SESSION_LOCK:
        if REMBG_SESSION is None:
            from rembg import new_session

            REMBG_SESSION = new_session(CHROMA_FALLBACK_REMBG_MODEL)
    return REMBG_SESSION


def _identity_marking_spatial_signature(
    rgb: np.ndarray,
    foreground: np.ndarray,
) -> dict[str, Any]:
    """Measure internal elongated dark bands without treating contour shade as fur.

    Global palette histograms cannot distinguish a few narrow invented stripes from
    ordinary dark eyes, nose, or stylized shading.  This signature looks only inside
    the middle body corridor, removes the subject boundary, and counts sizeable dark
    components that are both locally contrasted and elongated.  It is deliberately
    pose-independent: the quality gate compares component count and occupied area,
    not exact pixel coordinates.
    """

    foreground = foreground.astype(bool)
    rows, columns = np.where(foreground)
    if len(rows) < 256:
        return {
            "available": False,
            "policy": IDENTITY_MARKING_SPATIAL_POLICY,
            "reason": "insufficient_foreground_for_marking_topology",
        }
    x0, x1 = int(columns.min()), int(columns.max()) + 1
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    subject_width = x1 - x0
    subject_height = y1 - y0
    row_grid = np.indices(foreground.shape)[0]
    body = (
        foreground
        & (row_grid >= y0 + int(round(subject_height * 0.28)))
        & (row_grid < y0 + int(round(subject_height * 0.88)))
    )
    boundary_clearance = max(
        2,
        int(round(min(subject_width, subject_height) * 0.012)),
    )
    distance = cv2.distanceTransform(
        foreground.astype(np.uint8),
        cv2.DIST_L2,
        5,
    )
    body &= distance >= boundary_clearance
    body_pixels = int(np.count_nonzero(body))
    if body_pixels < 256:
        return {
            "available": False,
            "policy": IDENTITY_MARKING_SPATIAL_POLICY,
            "reason": "insufficient_internal_body_for_marking_topology",
            "foreground_bbox": [x0, y0, x1, y1],
        }

    luma = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[:, :, 0]
    blur_sigma = max(
        3,
        min(15, int(round(min(subject_width, subject_height) * 0.035))),
    )
    blur_kernel = blur_sigma * 4 + 1
    if blur_kernel % 2 == 0:
        blur_kernel += 1
    local_luma = cv2.GaussianBlur(
        luma,
        (blur_kernel, blur_kernel),
        sigmaX=float(blur_sigma),
    )
    local_contrast = local_luma.astype(np.int16) - luma.astype(np.int16)
    dark = (
        body
        & (local_contrast >= IDENTITY_MARKING_LOCAL_CONTRAST_MIN)
        & (luma <= IDENTITY_MARKING_LUMA_MAX)
    )
    dark = cv2.morphologyEx(
        dark.astype(np.uint8),
        cv2.MORPH_OPEN,
        np.ones((3, 3), dtype=np.uint8),
    )
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        dark,
        connectivity=8,
    )
    minimum_component_area = max(
        10,
        int(
            round(
                body_pixels * IDENTITY_MARKING_COMPONENT_AREA_FRACTION_MIN
            )
        ),
    )
    stripe_components: list[dict[str, Any]] = []
    stripe_pixels = 0
    subject_scale = float(max(1, min(subject_width, subject_height)))
    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < minimum_component_area:
            continue
        component_rows, component_columns = np.where(labels == label)
        points = np.column_stack((component_columns, component_rows)).astype(
            np.float64
        )
        eigenvalues = np.linalg.eigvalsh(np.cov(points, rowvar=False))
        elongation = float(
            np.sqrt(
                max(float(eigenvalues[-1]), 0.0)
                / max(0.25, float(eigenvalues[0]))
            )
        )
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        length_ratio = max(width, height) / subject_scale
        thinness = min(width, height) / float(max(1, max(width, height)))
        if (
            elongation < IDENTITY_MARKING_COMPONENT_ELONGATION_MIN
            or length_ratio < IDENTITY_MARKING_COMPONENT_LENGTH_RATIO_MIN
            or thinness > IDENTITY_MARKING_COMPONENT_THINNESS_MAX
        ):
            continue
        stripe_pixels += area
        stripe_components.append(
            {
                "area": area,
                "area_fraction": round(area / float(body_pixels), 6),
                "bbox": [
                    int(stats[label, cv2.CC_STAT_LEFT]),
                    int(stats[label, cv2.CC_STAT_TOP]),
                    width,
                    height,
                ],
                "elongation": round(elongation, 6),
                "length_ratio": round(length_ratio, 6),
                "thinness": round(thinness, 6),
            }
        )
    return {
        "available": True,
        "policy": IDENTITY_MARKING_SPATIAL_POLICY,
        "foreground_bbox": [x0, y0, x1, y1],
        "body_pixels": body_pixels,
        "boundary_clearance": boundary_clearance,
        "local_contrast_min": IDENTITY_MARKING_LOCAL_CONTRAST_MIN,
        "luma_max": IDENTITY_MARKING_LUMA_MAX,
        "minimum_component_area": minimum_component_area,
        "dark_contrast_fraction": round(
            float(np.count_nonzero(dark)) / float(body_pixels),
            6,
        ),
        "stripe_component_count": len(stripe_components),
        "stripe_area_fraction": round(
            stripe_pixels / float(body_pixels),
            6,
        ),
        "stripe_components": stripe_components,
    }


def identity_palette_signature(
    path: Path,
    maximum_size: int = 512,
    *,
    exclude_bottom_fraction: float = 0.0,
) -> dict[str, Any]:
    """Summarize visible foreground colors without treating the background as coat."""
    _sync_local_video_runtime(globals(), {"identity_palette_signature"})
    with Image.open(path) as loaded:
        image = loaded.convert("RGBA")
    if max(image.size) > maximum_size:
        scale = maximum_size / float(max(image.size))
        image = image.resize(
            (
                max(1, int(round(image.width * scale))),
                max(1, int(round(image.height * scale))),
            ),
            Image.Resampling.LANCZOS,
        )
    rgba = np.asarray(image)
    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3]
    foreground: np.ndarray | None = None
    mask_method = "unknown"
    if np.any(alpha < 250):
        alpha_foreground = alpha >= 64
        alpha_fraction = float(np.count_nonzero(alpha_foreground)) / float(
            alpha_foreground.size
        )
        if 0.02 <= alpha_fraction <= 0.95:
            foreground = alpha_foreground
            mask_method = "alpha"
    if foreground is None:
        red = rgb[:, :, 0].astype(np.int16)
        green = rgb[:, :, 1].astype(np.int16)
        blue = rgb[:, :, 2].astype(np.int16)
        green_background = (
            (green > 80)
            & ((green - np.maximum(red, blue)) > 35)
            & (green > red * 1.18)
            & (green > blue * 1.18)
        )
        green_fraction = float(np.count_nonzero(green_background)) / float(
            green_background.size
        )
        border_size = max(2, int(round(min(green_background.shape) * 0.06)))
        border_mask = np.zeros(green_background.shape, dtype=bool)
        border_mask[:border_size, :] = True
        border_mask[-border_size:, :] = True
        border_mask[:, :border_size] = True
        border_mask[:, -border_size:] = True
        border_green_fraction = float(
            np.count_nonzero(green_background & border_mask)
        ) / float(max(1, np.count_nonzero(border_mask)))
        corner_height = max(2, int(round(green_background.shape[0] * 0.10)))
        corner_width = max(2, int(round(green_background.shape[1] * 0.10)))
        corner_blocks = (
            green_background[:corner_height, :corner_width],
            green_background[:corner_height, -corner_width:],
            green_background[-corner_height:, :corner_width],
            green_background[-corner_height:, -corner_width:],
        )
        green_corner_count = sum(
            float(np.count_nonzero(block)) / float(block.size) >= 0.60
            for block in corner_blocks
        )
        if (
            green_fraction >= 0.08
            and border_green_fraction >= 0.45
            and green_corner_count >= 3
        ):
            foreground = ~green_background
            mask_method = "green_screen"
    if foreground is None:
        try:
            from rembg import remove

            recovered = remove(image, session=get_rembg_session())
            if not isinstance(recovered, Image.Image):
                recovered = Image.open(io.BytesIO(recovered)).convert("RGBA")
            recovered_alpha = np.asarray(recovered.convert("RGBA"))[:, :, 3]
            semantic_foreground = recovered_alpha >= 64
            semantic_fraction = float(np.count_nonzero(semantic_foreground)) / float(
                semantic_foreground.size
            )
            if 0.02 <= semantic_fraction <= 0.95:
                foreground = semantic_foreground
                mask_method = f"rembg_{CHROMA_FALLBACK_REMBG_MODEL}"
        except Exception:  # noqa: BLE001 - palette validation safely degrades to a warning
            LOGGER.warning(
                "Could not segment %s for identity palette analysis; "
                "palette guard is unavailable for this image",
                path.name,
                exc_info=True,
            )
    if foreground is None:
        return {
            "available": False,
            "reason": "foreground_segmentation_unavailable",
            "mask_method": mask_method,
            "exclude_bottom_fraction": round(
                max(0.0, min(0.45, float(exclude_bottom_fraction))),
                6,
            ),
        }
    foreground_u8 = foreground.astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        foreground_u8,
        connectivity=8,
    )
    if component_count > 1:
        largest_label = max(
            range(1, component_count),
            key=lambda label: int(stats[label, cv2.CC_STAT_AREA]),
        )
        foreground_u8 = (labels == largest_label).astype(np.uint8)
    exclude_bottom_fraction = max(0.0, min(0.45, float(exclude_bottom_fraction)))
    if exclude_bottom_fraction > 0.0:
        foreground_ys = np.where(foreground_u8 > 0)[0]
        if len(foreground_ys):
            y_min = int(foreground_ys.min())
            y_max = int(foreground_ys.max()) + 1
        else:
            y_min = 0
            y_max = foreground_u8.shape[0]
        cutoff = y_min + int(round((y_max - y_min) * (1.0 - exclude_bottom_fraction)))
        foreground_u8[max(1, cutoff) :, :] = 0
    if min(foreground_u8.shape) >= 24:
        foreground_u8 = cv2.erode(
            foreground_u8,
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        )
    # Bowl actions exclude the lower subject band from palette statistics, but
    # the remaining head/torso region is precisely where invented stripes must
    # still be detected.  Run the spatial marking gate on that retained mask.
    marking_spatial_signature = _identity_marking_spatial_signature(
        rgb,
        foreground_u8 > 0,
    )
    marking_spatial_signature["partial_subject_palette"] = bool(
        exclude_bottom_fraction > 0.0
    )
    foreground_rows, foreground_columns = np.where(foreground_u8 > 0)
    pixels = rgb[foreground_rows, foreground_columns]
    if len(pixels) < 64:
        return {
            "available": False,
            "reason": "insufficient_foreground_pixels",
            "mask_method": mask_method,
            "foreground_fraction": round(
                len(pixels) / float(max(1, foreground_u8.size)),
                6,
            ),
        }
    if len(pixels) > 250_000:
        stride = max(1, (len(pixels) + 249_999) // 250_000)
        pixels = pixels[::stride]
        sampled_rows = foreground_rows[::stride]
    else:
        sampled_rows = foreground_rows
    hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    hue = hsv[:, 0]
    saturation = hsv[:, 1]
    value = hsv[:, 2]
    category_masks: dict[str, np.ndarray] = {}
    category_masks["black"] = value < 58
    category_masks["white"] = (value >= 190) & (saturation < 52)
    category_masks["gray"] = ~(category_masks["black"] | category_masks["white"]) & (
        saturation < 52
    )
    chromatic = ~(
        category_masks["black"] | category_masks["white"] | category_masks["gray"]
    )
    category_masks["red_or_pink"] = chromatic & ((hue < 8) | (hue >= 165))
    category_masks["warm_brown_or_orange"] = chromatic & (hue >= 8) & (hue < 28)
    category_masks["yellow_or_gold"] = chromatic & (hue >= 28) & (hue < 42)
    category_masks["green"] = chromatic & (hue >= 42) & (hue < 88)
    category_masks["blue_or_cyan"] = chromatic & (hue >= 88) & (hue < 135)
    category_masks["purple"] = chromatic & (hue >= 135) & (hue < 165)
    total = float(len(pixels))
    fractions = {
        category: round(
            float(np.count_nonzero(category_masks[category])) / total,
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
    achromatic_fraction = sum(
        fractions[category] for category in ("black", "white", "gray")
    )
    chromatic_fraction = sum(
        fractions[category] for category in IDENTITY_PALETTE_CHROMATIC
    )
    warm_fraction = sum(
        fractions[category]
        for category in (
            "warm_brown_or_orange",
            "red_or_pink",
            "yellow_or_gold",
        )
    )
    warm_mask = (
        category_masks["warm_brown_or_orange"]
        | category_masks["red_or_pink"]
        | category_masks["yellow_or_gold"]
    )
    warm_saturation = saturation[warm_mask]
    warm_saturation_mean = (
        float(np.mean(warm_saturation)) / 255.0 if len(warm_saturation) else 0.0
    )
    warm_saturation_p90 = (
        float(np.percentile(warm_saturation, 90)) / 255.0
        if len(warm_saturation)
        else 0.0
    )
    saturated_warm_fraction = float(
        np.count_nonzero(warm_mask & (saturation >= 96))
    ) / float(len(pixels))
    foreground_y0 = int(foreground_rows.min())
    foreground_y1 = int(foreground_rows.max()) + 1
    foreground_x0 = int(foreground_columns.min())
    foreground_x1 = int(foreground_columns.max()) + 1
    body_start_y = foreground_y0 + int(round((foreground_y1 - foreground_y0) * 0.45))
    red_or_pink_pixels = category_masks["red_or_pink"]
    red_or_pink_count = int(np.count_nonzero(red_or_pink_pixels))
    red_or_pink_body_count = int(
        np.count_nonzero(red_or_pink_pixels & (sampled_rows >= body_start_y))
    )
    red_or_pink_body_fraction = red_or_pink_body_count / total
    red_or_pink_body_share = red_or_pink_body_count / float(max(1, red_or_pink_count))
    green_or_cyan_pixels = category_masks["green"] | category_masks["blue_or_cyan"]
    green_or_cyan_count = int(np.count_nonzero(green_or_cyan_pixels))
    green_or_cyan_body_count = int(
        np.count_nonzero(green_or_cyan_pixels & (sampled_rows >= body_start_y))
    )
    green_or_cyan_body_fraction = green_or_cyan_body_count / total
    green_or_cyan_body_share = green_or_cyan_body_count / float(
        max(1, green_or_cyan_count)
    )
    return {
        "available": True,
        "mask_method": mask_method,
        "exclude_bottom_fraction": round(exclude_bottom_fraction, 6),
        "foreground_fraction": round(
            float(np.count_nonzero(foreground_u8)) / float(foreground_u8.size),
            6,
        ),
        "sample_count": len(pixels),
        "fractions": fractions,
        "dominant": dominant,
        "achromatic_fraction": round(achromatic_fraction, 6),
        "chromatic_fraction": round(chromatic_fraction, 6),
        "warm_fraction": round(warm_fraction, 6),
        "warm_saturation_mean": round(warm_saturation_mean, 6),
        "warm_saturation_p90": round(warm_saturation_p90, 6),
        "saturated_warm_fraction": round(saturated_warm_fraction, 6),
        "foreground_bbox": [
            foreground_x0,
            foreground_y0,
            foreground_x1,
            foreground_y1,
        ],
        "red_or_pink_body_fraction": round(red_or_pink_body_fraction, 6),
        "red_or_pink_body_share": round(red_or_pink_body_share, 6),
        "red_or_pink_localized_to_head": bool(
            red_or_pink_count > 0
            and red_or_pink_body_fraction <= 0.025
            and red_or_pink_body_share <= 0.20
        ),
        "green_or_cyan_body_fraction": round(
            green_or_cyan_body_fraction,
            6,
        ),
        "green_or_cyan_body_share": round(green_or_cyan_body_share, 6),
        "green_or_cyan_localized_to_head": bool(
            green_or_cyan_count > 0
            and green_or_cyan_body_fraction <= 0.020
            and green_or_cyan_body_share <= 0.25
        ),
        "marking_spatial_signature": marking_spatial_signature,
    }


def neutralize_uploaded_identity_chroma_spill(
    path: Path,
    source_identity_palette: dict[str, Any] | None,
) -> dict[str, Any]:
    """Remove green-screen reflection without recoloring genuine green identity.

    The generic chroma key only despills the matte boundary.  For uploaded
    identities whose source coat has no green/cyan body region, Qwen can still
    paint the screen reflection into opaque muzzle, chest, or flank pixels.  A
    paired compact eye detector protects plausible irises; remaining
    green-dominant foreground pixels have their Lab chroma reconstructed from
    nearby unaffected coat pixels while luminance and alpha stay byte identical.
    This avoids turning warm or patterned coats into gray patches.  A downstream
    localized palette gate verifies the result.
    """
    _sync_local_video_runtime(globals(), {"neutralize_uploaded_identity_chroma_spill"})

    result: dict[str, Any] = {
        "policy": UPLOADED_IDENTITY_CHROMA_SPILL_POLICY,
        "path": str(path),
        "applied": False,
        "acceptable": True,
        "issues": [],
        "regeneration_required": False,
    }
    if not isinstance(source_identity_palette, dict) or not source_identity_palette.get(
        "available"
    ):
        result["reason"] = "source_identity_palette_unavailable"
        return result
    source_fractions = source_identity_palette.get("fractions")
    if not isinstance(source_fractions, dict):
        result["reason"] = "source_identity_palette_fractions_unavailable"
        return result
    source_green_or_cyan = float(source_fractions.get("green", 0.0)) + float(
        source_fractions.get("blue_or_cyan", 0.0)
    )
    source_green_or_cyan_body = float(
        source_identity_palette.get("green_or_cyan_body_fraction", 0.0)
    )
    result["source_green_or_cyan_fraction"] = round(source_green_or_cyan, 6)
    result["source_green_or_cyan_body_fraction"] = round(
        source_green_or_cyan_body,
        6,
    )
    if source_green_or_cyan > 0.07 or source_green_or_cyan_body > 0.012:
        result["reason"] = "source_green_or_cyan_identity_requires_preservation"
        return result

    with Image.open(path) as loaded:
        rgba = np.asarray(loaded.convert("RGBA"), dtype=np.uint8).copy()
    alpha = rgba[:, :, 3]
    foreground = alpha >= 64
    foreground_count = int(np.count_nonzero(foreground))
    if foreground_count < 64:
        result["reason"] = "foreground_unavailable"
        return result
    rows, columns = np.where(foreground)
    x0, x1 = int(columns.min()), int(columns.max()) + 1
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    subject_width = x1 - x0
    subject_height = y1 - y0

    def spill_mask(rgb: np.ndarray) -> np.ndarray:
        red = rgb[:, :, 0].astype(np.int16)
        green = rgb[:, :, 1].astype(np.int16)
        blue = rgb[:, :, 2].astype(np.int16)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        return (
            foreground
            & (green >= 52)
            & (green - red >= 5)
            & (green - blue >= -4)
            & (hsv[:, :, 0] >= 34)
            & (hsv[:, :, 0] < 112)
            & (hsv[:, :, 1] >= 20)
        )

    detected = spill_mask(rgba[:, :, :3])
    detected_count = int(np.count_nonzero(detected))
    body_start_y = y0 + int(round(subject_height * 0.42))
    row_grid = np.indices(foreground.shape)[0]
    detected_body_count = int(np.count_nonzero(detected & (row_grid >= body_start_y)))
    result.update(
        {
            "foreground_bbox": [x0, y0, x1, y1],
            "foreground_pixels": foreground_count,
            "detected_pixels_before": detected_count,
            "detected_fraction_before": round(
                detected_count / float(foreground_count),
                6,
            ),
            "detected_body_fraction_before": round(
                detected_body_count / float(foreground_count),
                6,
            ),
        }
    )
    if not detected_count:
        result["reason"] = "no_green_screen_spill_detected"
        return result

    paired_eye_mask = np.zeros(foreground.shape, dtype=bool)
    eye_components_mask = cv2.morphologyEx(
        detected.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
    )
    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        eye_components_mask,
        connectivity=8,
    )
    minimum_eye_area = max(6, int(round(foreground_count * 0.00003)))
    maximum_eye_area = max(96, int(round(foreground_count * 0.025)))
    eye_candidates: list[dict[str, Any]] = []
    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        center_x, center_y = (float(value) for value in centroids[label])
        aspect = width / float(max(1, height))
        if (
            minimum_eye_area <= area <= maximum_eye_area
            and width <= max(8, int(round(subject_width * 0.18)))
            and height <= max(8, int(round(subject_height * 0.14)))
            and 0.35 <= aspect <= 2.8
            and x0 + subject_width * 0.12 <= center_x <= x1 - subject_width * 0.12
            and center_y <= y0 + subject_height * 0.66
        ):
            eye_candidates.append(
                {
                    "label": label,
                    "area": area,
                    "center": [center_x, center_y],
                }
            )
    best_pair: tuple[float, dict[str, Any], dict[str, Any]] | None = None
    for left_index, first in enumerate(eye_candidates):
        for second in eye_candidates[left_index + 1 :]:
            first_x, first_y = first["center"]
            second_x, second_y = second["center"]
            separation = abs(second_x - first_x)
            y_delta = abs(second_y - first_y)
            area_ratio = min(first["area"], second["area"]) / float(
                max(first["area"], second["area"])
            )
            midpoint = (first_x + second_x) / 2.0
            if (
                subject_width * 0.10 <= separation <= subject_width * 0.58
                and y_delta <= subject_height * 0.09
                and area_ratio >= 0.20
                and abs(midpoint - (x0 + subject_width / 2.0)) <= subject_width * 0.20
            ):
                relative_y = ((first_y + second_y) / 2.0 - y0) / float(
                    max(1, subject_height)
                )
                score = (
                    relative_y
                    + y_delta / float(max(1, subject_height))
                    + abs(midpoint - (x0 + subject_width / 2.0))
                    / float(max(1, subject_width))
                    + (1.0 - area_ratio) * 0.10
                )
                if best_pair is None or score < best_pair[0]:
                    best_pair = (score, first, second)
    protected_labels: list[int] = []
    if best_pair is not None:
        protected_labels = [best_pair[1]["label"], best_pair[2]["label"]]
        paired_eye_mask = np.isin(labels, protected_labels)
        paired_eye_mask = cv2.dilate(
            paired_eye_mask.astype(np.uint8),
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        ).astype(bool)
        paired_eye_mask &= detected

    detected_correction = detected & ~paired_eye_mask
    enclosed_spill_repair = np.zeros(foreground.shape, dtype=bool)
    enclosed_spill_components: list[dict[str, Any]] = []
    closed_correction = cv2.morphologyEx(
        detected_correction.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    component_count, component_labels, component_stats, _ = (
        cv2.connectedComponentsWithStats(
            closed_correction,
            connectivity=8,
        )
    )
    minimum_enclosed_component = max(
        32,
        int(round(foreground_count * 0.0002)),
    )
    maximum_enclosed_hole = max(
        256,
        int(round(foreground_count * 0.04)),
    )
    repair_radius = max(
        2,
        min(
            8,
            int(round(min(subject_width, subject_height) * 0.008)),
        ),
    )
    frame_height, frame_width = foreground.shape
    for label in range(1, component_count):
        component = (component_labels == label).astype(np.uint8)
        component_area = int(np.count_nonzero(component))
        if component_area < minimum_enclosed_component:
            continue
        inverse = (component == 0).astype(np.uint8)
        hole_count, hole_labels, hole_stats, _ = (
            cv2.connectedComponentsWithStats(
                inverse,
                connectivity=8,
            )
        )
        for hole_label in range(1, hole_count):
            hole_x = int(hole_stats[hole_label, cv2.CC_STAT_LEFT])
            hole_y = int(hole_stats[hole_label, cv2.CC_STAT_TOP])
            hole_width = int(hole_stats[hole_label, cv2.CC_STAT_WIDTH])
            hole_height = int(hole_stats[hole_label, cv2.CC_STAT_HEIGHT])
            if (
                hole_x == 0
                or hole_y == 0
                or hole_x + hole_width == frame_width
                or hole_y + hole_height == frame_height
            ):
                continue
            raw_hole = hole_labels == hole_label
            raw_hole_area = int(np.count_nonzero(raw_hole))
            hole = raw_hole & foreground & ~paired_eye_mask
            hole_area = int(np.count_nonzero(hole))
            if (
                hole_area < minimum_enclosed_component
                or hole_area > maximum_enclosed_hole
                or hole_area < int(round(raw_hole_area * 0.90))
                or hole_y < y0 + int(round(subject_height * 0.32))
                or hole_width > int(round(subject_width * 0.45))
                or hole_height > int(round(subject_height * 0.35))
            ):
                continue
            repair_region = cv2.dilate(
                hole.astype(np.uint8),
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (repair_radius * 2 + 1, repair_radius * 2 + 1),
                ),
                iterations=1,
            ).astype(bool)
            repair_region &= foreground & ~paired_eye_mask
            enclosed_spill_repair |= repair_region
            enclosed_spill_components.append(
                {
                    "label": label,
                    "hole_label": hole_label,
                    "bbox": [
                        hole_x,
                        hole_y,
                        hole_x + hole_width,
                        hole_y + hole_height,
                    ],
                    "detected_component_pixels": component_area,
                    "hole_pixels": hole_area,
                    "repair_radius": repair_radius,
                    "repair_pixels": int(np.count_nonzero(repair_region)),
                }
            )

    correction = detected_correction | enclosed_spill_repair
    corrected_count = int(np.count_nonzero(correction))
    enclosed_spill_pixels = int(np.count_nonzero(enclosed_spill_repair))
    enclosed_spill_added_pixels = int(
        np.count_nonzero(enclosed_spill_repair & ~detected_correction)
    )
    broad_body_spill = bool(
        detected_body_count / float(max(1, foreground_count))
        >= UPLOADED_IDENTITY_CHROMA_SPILL_BODY_REGENERATION_FRACTION_MIN
    )
    unsafe_enclosed_components: list[dict[str, Any]] = []
    for component in enclosed_spill_components:
        component_fraction = int(component["detected_component_pixels"]) / float(
            max(1, foreground_count)
        )
        hole_fraction = int(component["hole_pixels"]) / float(
            max(1, foreground_count)
        )
        if (
            component_fraction
            >= UPLOADED_IDENTITY_CHROMA_SPILL_COMPONENT_REGENERATION_FRACTION_MIN
            and hole_fraction
            >= UPLOADED_IDENTITY_CHROMA_SPILL_HOLE_REGENERATION_FRACTION_MIN
        ):
            unsafe_enclosed_components.append(
                {
                    **component,
                    "detected_component_fraction": round(component_fraction, 6),
                    "hole_fraction": round(hole_fraction, 6),
                }
            )
    regeneration_reasons = []
    if broad_body_spill:
        regeneration_reasons.append("broad_body_green_screen_spill")
    if unsafe_enclosed_components:
        regeneration_reasons.append("large_enclosed_green_screen_marking")
    regeneration_required = bool(regeneration_reasons)
    regeneration_issues = (
        ["identity_chroma_spill_exceeds_safe_reconstruction_scope"]
        if regeneration_required
        else []
    )
    alpha_hash_before = hashlib.sha256(alpha.tobytes()).hexdigest()
    if corrected_count:
        rgb = rgba[:, :, :3]
        # Reconstruct chroma in Lab while preserving luminance and therefore fur
        # texture.  A flat neutral a/b replacement removed green but produced
        # visible gray patches on warm dogs.  Seed non-donor pixels with the
        # subject's robust coat chroma, then let Telea propagate local intact coat
        # color across the detected spill region.  Alpha is never touched.
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        donor = foreground & ~correction & ~paired_eye_mask
        donor_count = int(np.count_nonzero(donor))
        if donor_count >= 64:
            # Screen bounce can tint even nominally unaffected pale coat donors.
            # Keep the yellow/blue channel local, but never propagate a green
            # Lab a-axis value back into a source identity that has no green coat.
            fallback_a = max(128, int(np.median(lab[:, :, 1][donor])))
            fallback_b = int(np.median(lab[:, :, 2][donor]))
        else:
            fallback_a = 128
            fallback_b = 128
        inpaint_radius = max(
            3,
            min(15, int(round(min(subject_width, subject_height) * 0.025))),
        )
        correction_u8 = correction.astype(np.uint8) * 255
        reconstructed_lab = lab.copy()
        for channel, fallback in ((1, fallback_a), (2, fallback_b)):
            prepared = lab[:, :, channel].copy()
            prepared[~donor] = fallback
            repaired = cv2.inpaint(
                prepared,
                correction_u8,
                float(inpaint_radius),
                cv2.INPAINT_TELEA,
            )
            if channel == 1:
                repaired = np.maximum(repaired, 128).astype(np.uint8)
            reconstructed_lab[:, :, channel][correction] = repaired[correction]
        if enclosed_spill_pixels:
            texture_mask_u8 = enclosed_spill_repair.astype(np.uint8) * 255
            repaired_luma = cv2.inpaint(
                lab[:, :, 0],
                texture_mask_u8,
                float(inpaint_radius),
                cv2.INPAINT_TELEA,
            )
            repair_distance = cv2.distanceTransform(
                texture_mask_u8,
                cv2.DIST_L2,
                5,
            )
            luma_feather_width = float(max(2, repair_radius + 1))
            luma_weight = np.clip(
                repair_distance / luma_feather_width,
                0.0,
                1.0,
            )
            blended_luma = np.clip(
                lab[:, :, 0].astype(np.float32) * (1.0 - luma_weight)
                + repaired_luma.astype(np.float32) * luma_weight,
                0,
                255,
            ).astype(np.uint8)
            reconstructed_lab[:, :, 0][enclosed_spill_repair] = blended_luma[
                enclosed_spill_repair
            ]
        reconstructed_rgb = cv2.cvtColor(reconstructed_lab, cv2.COLOR_LAB2RGB)
        rgb[correction] = reconstructed_rgb[correction]
        Image.fromarray(rgba, mode="RGBA").save(path, format="PNG", optimize=True)
    else:
        donor_count = 0
        fallback_a = 128
        fallback_b = 128
        inpaint_radius = 0

    remaining = spill_mask(rgba[:, :, :3])
    remaining_count = int(np.count_nonzero(remaining))
    remaining_body_count = int(np.count_nonzero(remaining & (row_grid >= body_start_y)))
    result.update(
        {
            "applied": bool(corrected_count),
            "acceptable": not regeneration_required,
            "issues": regeneration_issues,
            "regeneration_required": regeneration_required,
            "regeneration_reasons": regeneration_reasons,
            "regeneration_thresholds": {
                "body_fraction_min": (
                    UPLOADED_IDENTITY_CHROMA_SPILL_BODY_REGENERATION_FRACTION_MIN
                ),
                "enclosing_component_fraction_min": (
                    UPLOADED_IDENTITY_CHROMA_SPILL_COMPONENT_REGENERATION_FRACTION_MIN
                ),
                "enclosed_hole_fraction_min": (
                    UPLOADED_IDENTITY_CHROMA_SPILL_HOLE_REGENERATION_FRACTION_MIN
                ),
            },
            "unsafe_enclosed_components": unsafe_enclosed_components,
            "reason": "corrected" if corrected_count else "only_paired_eye_accents",
            "corrected_pixels": corrected_count,
            "detected_correction_pixels": int(
                np.count_nonzero(detected_correction)
            ),
            "chroma_reconstruction": "local_lab_telea_inpaint",
            "enclosed_spill_reconstruction": (
                "full_lab_telea_inpaint"
                if enclosed_spill_pixels
                else None
            ),
            "enclosed_spill_component_count": len(
                enclosed_spill_components
            ),
            "enclosed_spill_components": enclosed_spill_components,
            "enclosed_spill_pixels": enclosed_spill_pixels,
            "enclosed_spill_added_pixels": enclosed_spill_added_pixels,
            "luminance_reconstruction_pixels": enclosed_spill_pixels,
            "luminance_reconstruction_feather_width": (
                float(max(2, repair_radius + 1))
                if enclosed_spill_pixels
                else 0.0
            ),
            "chroma_donor_pixels": donor_count,
            "fallback_lab_ab": [fallback_a, fallback_b],
            "corrected_lab_a_floor": 128,
            "inpaint_radius": inpaint_radius,
            "protected_eye_pixels": int(np.count_nonzero(paired_eye_mask)),
            "protected_eye_component_labels": protected_labels,
            "detected_pixels_after": remaining_count,
            "detected_fraction_after": round(
                remaining_count / float(foreground_count),
                6,
            ),
            "detected_body_fraction_after": round(
                remaining_body_count / float(foreground_count),
                6,
            ),
            "alpha_sha256_before": alpha_hash_before,
            "alpha_sha256_after": hashlib.sha256(rgba[:, :, 3].tobytes()).hexdigest(),
            "alpha_unchanged": alpha_hash_before
            == hashlib.sha256(rgba[:, :, 3].tobytes()).hexdigest(),
        }
    )
    return result


def identity_palette_is_achromatic(signature: dict[str, Any]) -> bool:
    """Treat a small low-saturation photographic cast as lighting, not coat identity."""
    _sync_local_video_runtime(globals(), {"identity_palette_is_achromatic"})

    achromatic = float(signature.get("achromatic_fraction", 0.0))
    warm = float(signature.get("warm_fraction", 0.0))
    saturated_warm_raw = signature.get("saturated_warm_fraction")
    low_saturation_lighting = bool(
        saturated_warm_raw is not None
        and float(saturated_warm_raw) <= 0.03
        and warm <= 0.14
    )
    return bool(achromatic >= 0.72 and (warm <= 0.12 or low_saturation_lighting))


def identity_palette_stripe_amplification_refine_eligibility(
    quality: dict[str, Any] | None,
) -> dict[str, Any]:
    """Allow repainting amplified stripes only when the upload contains them.

    ``invented_stripe_pattern`` also covers a real, low-contrast source pattern
    that a generated candidate made much darker or denser.  Such a candidate is
    safe as an eating *geometry* sampler when the source contains measurable
    stripe evidence; a genuinely unstriped upload must continue to fail closed.
    """
    _sync_local_video_runtime(
        globals(),
        {"identity_palette_stripe_amplification_refine_eligibility"},
    )

    palette = quality if isinstance(quality, dict) else {}
    issues = {
        str(issue)
        for issue in palette.get("issues", [])
        if str(issue).strip()
    }
    spatial = palette.get("marking_spatial_quality")
    spatial = spatial if isinstance(spatial, dict) else {}
    source_component_count = int(
        spatial.get("source_stripe_component_count", 0) or 0
    )
    source_area_fraction = float(
        spatial.get("source_stripe_area_fraction", 0.0) or 0.0
    )
    checks = {
        "quality_available": palette.get("available") is True,
        "invented_stripe_issue_present": "invented_stripe_pattern" in issues,
        "marking_spatial_available": spatial.get("available") is True,
        "source_component_count_min": (
            source_component_count
            >= IDENTITY_STRIPE_AMPLIFICATION_SOURCE_COMPONENT_MIN
        ),
        "source_area_fraction_min": (
            source_area_fraction >= IDENTITY_STRIPE_AMPLIFICATION_SOURCE_AREA_MIN
        ),
    }
    eligible = all(checks.values())
    return {
        "policy": IDENTITY_STRIPE_AMPLIFICATION_REFINE_POLICY,
        "eligible": eligible,
        "reason": (
            "source_present_stripes_were_amplified"
            if eligible
            else "source_stripe_evidence_insufficient_for_geometry_sampler"
        ),
        "issue": "invented_stripe_pattern",
        "source_stripe_component_count": source_component_count,
        "source_stripe_area_fraction": round(source_area_fraction, 6),
        "candidate_stripe_component_count": int(
            spatial.get("candidate_stripe_component_count", 0) or 0
        ),
        "candidate_stripe_area_fraction": round(
            float(spatial.get("candidate_stripe_area_fraction", 0.0) or 0.0),
            6,
        ),
        "source_dark_contrast_fraction": round(
            float(spatial.get("source_dark_contrast_fraction", 0.0) or 0.0),
            6,
        ),
        "candidate_dark_contrast_fraction": round(
            float(spatial.get("candidate_dark_contrast_fraction", 0.0) or 0.0),
            6,
        ),
        "thresholds": {
            "source_component_count_min": (
                IDENTITY_STRIPE_AMPLIFICATION_SOURCE_COMPONENT_MIN
            ),
            "source_area_fraction_min": (
                IDENTITY_STRIPE_AMPLIFICATION_SOURCE_AREA_MIN
            ),
        },
        "checks": checks,
        "identity_gate_unchanged": True,
    }


def identity_palette_retry_guidance(
    previous_quality: dict[str, Any] | None,
) -> str:
    """Translate the previous palette verdict into bounded correction guidance."""
    _sync_local_video_runtime(globals(), {"identity_palette_retry_guidance"})

    if (
        not isinstance(previous_quality, dict)
        or previous_quality.get("acceptable") is True
    ):
        return ""
    issues = {
        str(issue)
        for issue in previous_quality.get("issues", [])
        if str(issue).strip()
    }
    guidance = [
        "Previous output recolored the pet and was rejected; restore the source "
        "palette exactly before changing pose or rendering."
    ]
    if "invented_stripe_pattern" in issues:
        eligibility = identity_palette_stripe_amplification_refine_eligibility(
            previous_quality
        )
        if eligibility["eligible"]:
            guidance.append(
                "The upload has a real but subtle stripe pattern, while the previous "
                "candidate amplified it. Preserve only the upload's faint localized "
                "bands; reduce dark-band count, occupied area, and contrast toward "
                f"the source evidence (about {eligibility['source_stripe_component_count']} "
                "components, "
                f"{eligibility['source_stripe_area_fraction'] * 100.0:.1f}% area, "
                f"{eligibility['source_dark_contrast_fraction'] * 100.0:.1f}% dark "
                "contrast). Do not copy the previous candidate's stronger pattern "
                f"({eligibility['candidate_stripe_component_count']} components, "
                f"{eligibility['candidate_stripe_area_fraction'] * 100.0:.1f}% area, "
                f"{eligibility['candidate_dark_contrast_fraction'] * 100.0:.1f}% dark "
                "contrast), and add no flank rings, torso bars, or head bands absent "
                "from the upload."
            )
        else:
            guidance.append(
                "The previous candidate invented dark stripes without sufficient "
                "source evidence. Remove those new bands completely and keep the "
                "affected coat regions in the upload's nearest visible base color."
            )
    if issues & {
        "new_gray_or_dark_markings_expanded",
        "achromatic_coat_value_balance_lost",
        "canonical_action_achromatic_value_balance_drift",
        "canonical_action_light_identity_dark_feature_expansion",
    }:
        guidance.append(
            "Shrink any enlarged dark muzzle, mask, ear, patch, or body marking back "
            "to the upload's placement and approximate area; pose shadows must remain "
            "soft lighting, not new coat markings."
        )
    if issues & {
        "achromatic_source_recolored_with_warm_fur",
        "canonical_action_warm_expansion_drift",
    } or any(issue.startswith("new_dominant_") for issue in issues):
        guidance.append(
            "Remove every new dominant hue and lighting cast; the bowl or action "
            "reference may not transfer its color into the pet."
        )
    if issues & {
        "warm_source_recolored_achromatic",
        "source_chromatic_identity_accents_lost",
    }:
        guidance.append(
            "Restore the source's genuine warm coat and localized chromatic accents "
            "instead of flattening them into gray or white."
        )
    return " " + " ".join(guidance)


def identity_palette_prompt(
    signature: dict[str, Any],
    *,
    retry: bool = False,
    previous_quality: dict[str, Any] | None = None,
) -> str:
    _sync_local_video_runtime(globals(), {"identity_palette_prompt"})
    retry_guidance = identity_palette_retry_guidance(previous_quality)
    retry_without_measured_palette = bool(
        retry
        and not (
            isinstance(previous_quality, dict)
            and previous_quality.get("acceptable") is True
        )
    )
    retry_rule = retry_guidance or (
        " Previous output recolored the pet and was rejected; restore the source "
        "palette exactly before changing pose or rendering."
        if retry_without_measured_palette
        else ""
    )
    if not signature.get("available"):
        return (
            "SOURCE COAT COLOR LOCK: copy every visible fur color and marking region "
            "from the uploaded pet. Never recolor the coat or borrow a palette from "
            f"a LoRA training example.{retry_rule}"
        )
    fractions = signature["fractions"]
    dominant = list(signature.get("dominant") or [])
    labels = [IDENTITY_PALETTE_PROMPT_LABELS[item] for item in dominant]
    palette = ", ".join(labels)
    warm_fur_family = float(
        fractions.get("warm_brown_or_orange", 0.0)
    ) + float(fractions.get("yellow_or_gold", 0.0))
    absent_chromatic = [
        IDENTITY_PALETTE_PROMPT_LABELS[item]
        for item in IDENTITY_PALETTE_CHROMATIC
        if float(fractions.get(item, 0.0)) <= 0.025
        and not (
            item in {"warm_brown_or_orange", "yellow_or_gold"}
            and warm_fur_family > 0.025
        )
    ]
    source_is_achromatic = identity_palette_is_achromatic(signature)
    if source_is_achromatic:
        achromatic_dominant = [
            item for item in dominant if item in {"black", "white", "gray"}
        ]
        achromatic_palette = (
            ", ".join(
                IDENTITY_PALETTE_PROMPT_LABELS[item] for item in achromatic_dominant
            )
            or "black, white, and neutral gray"
        )
        rule = (
            f"The visible source coat is achromatic and predominantly "
            f"{achromatic_palette}. Use neutral-white character lighting with no amber, "
            "sepia, orange, ginger, brown, or golden color cast over neutral fur. "
            "Any genuinely warm source color may remain only in the same localized "
            "marking regions and proportion; it must never spread across gray, black, "
            "or white fur. Red or pink is a small skin accent, never a fur color: keep "
            "it only in the same tiny low-saturation nose pad and small inner-ear cores "
            "visible in the original upload. Do not enlarge, saturate, or spread it into "
            "the ear panels, cheeks, forehead, or coat, and override any oversized "
            "salmon or pink ear fill inherited from an intermediate canonical image."
        )
        black_fraction = float(fractions.get("black", 0.0))
        neutral_light_fraction = float(fractions.get("gray", 0.0)) + float(
            fractions.get("white", 0.0)
        )
        rule += (
            " Preserve the source's coarse dark-marking versus neutral-light coat "
            f"balance (approximately black/dark {black_fraction * 100.0:.1f}% and "
            f"combined neutral gray-or-white {neutral_light_fraction * 100.0:.1f}% "
            "over the segmented visible coat). Photographic shadowed white may become "
            "brighter neutral white in the stylized render, but every dark marking "
            "must keep its placement and approximate area. Never erase the dark "
            "markings or expand them into large invented black patches."
        )
    else:
        rule = f"The visible source coat palette is predominantly {palette}."
    if absent_chromatic:
        rule += (
            " Do not introduce these absent dominant fur hues: "
            + ", ".join(absent_chromatic)
            + "."
        )
    return (
        "SOURCE COAT PALETTE LOCK - higher priority than style and LoRA: "
        f"{rule} Preserve the exact spatial placement and proportion of every source "
        "color region. Never add calico, tortoiseshell, tabby, stripe, spot, patch, "
        f"or warm highlight that is absent from the uploaded pet.{retry_rule}"
    )


def identity_palette_negative_prompt(signature: dict[str, Any]) -> str:
    _sync_local_video_runtime(globals(), {"identity_palette_negative_prompt"})
    negative = [
        "recolored coat",
        "new dominant fur color",
        "LoRA sample coat",
        "changed coat palette",
    ]
    if not signature.get("available"):
        return ", ".join(negative)
    fractions = signature["fractions"]
    if identity_palette_is_achromatic(signature):
        negative.extend(
            [
                "overall warm color cast",
                "amber lighting on fur",
                "sepia fur",
                "neutral gray fur turned orange",
                "neutral gray fur turned brown",
                "ginger recolor",
                "large pink inner-ear fill",
                "salmon ear panels",
                "saturated pink ears",
                "pink cheek fur",
                "red or pink coat",
                "rosy facial cast",
                "dark identity markings erased",
                "dark markings expanded into invented black patches",
                "large invented black and white patches",
            ]
        )
    warm_fur_family = float(
        fractions.get("warm_brown_or_orange", 0.0)
    ) + float(fractions.get("yellow_or_gold", 0.0))
    if warm_fur_family <= 0.025:
        negative.extend(
            [
                "orange fur",
                "ginger fur",
                "brown fur",
                "tan fur",
                "calico coat",
                "tortoiseshell coat",
            ]
        )
    if warm_fur_family <= 0.025:
        negative.append("golden fur")
    return ", ".join(negative)


def identity_palette_quality(
    source: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    _sync_local_video_runtime(globals(), {"identity_palette_quality"})
    if not source.get("available") or not candidate.get("available"):
        return {
            "available": False,
            # Identity preservation is a safety gate for uploaded subjects.
            # An unavailable signature is not evidence that the candidate
            # preserved the source palette, so fail closed and try another
            # generation instead of silently accepting it.
            "acceptable": False,
            "warning": True,
            "issues": ["identity_palette_signature_unavailable"],
            "penalty": 1000.0,
            "source": source,
            "candidate": candidate,
        }
    source_fractions = source["fractions"]
    candidate_fractions = candidate["fractions"]
    issues: list[str] = []
    warnings: list[str] = []
    deltas = {
        category: round(
            float(candidate_fractions.get(category, 0.0))
            - float(source_fractions.get(category, 0.0)),
            6,
        )
        for category in IDENTITY_PALETTE_CATEGORIES
    }
    source_green_or_cyan = float(source_fractions.get("green", 0.0)) + float(
        source_fractions.get("blue_or_cyan", 0.0)
    )
    candidate_green_or_cyan = float(candidate_fractions.get("green", 0.0)) + float(
        candidate_fractions.get("blue_or_cyan", 0.0)
    )
    source_green_or_cyan_body = float(source.get("green_or_cyan_body_fraction", 0.0))
    candidate_green_or_cyan_body = float(
        candidate.get("green_or_cyan_body_fraction", 0.0)
    )
    for category in IDENTITY_PALETTE_CHROMATIC:
        source_fraction = float(source_fractions.get(category, 0.0))
        candidate_fraction = float(candidate_fractions.get(category, 0.0))
        candidate_threshold = 0.09 if category == "warm_brown_or_orange" else 0.14
        delta_threshold = 0.07 if category == "warm_brown_or_orange" else 0.10
        if (
            source_fraction <= 0.04
            and candidate_fraction >= candidate_threshold
            and candidate_fraction - source_fraction >= delta_threshold
        ):
            if (
                category == "red_or_pink"
                and bool(candidate.get("red_or_pink_localized_to_head"))
                and float(candidate.get("red_or_pink_body_fraction", 1.0))
                <= max(
                    0.025,
                    float(source.get("red_or_pink_body_fraction", 0.0)) + 0.015,
                )
            ):
                warnings.append("localized_red_or_pink_skin_accent_expanded")
            else:
                issues.append(f"new_dominant_{category}")
    green_or_cyan_body_spill = bool(
        source_green_or_cyan <= 0.07
        and candidate_green_or_cyan - source_green_or_cyan >= 0.035
        and candidate_green_or_cyan_body
        >= max(0.020, source_green_or_cyan_body + 0.015)
        and not bool(candidate.get("green_or_cyan_localized_to_head"))
    )
    if green_or_cyan_body_spill:
        issues.append("green_or_cyan_chroma_spill_on_body")
    source_achromatic = float(source["achromatic_fraction"])
    candidate_achromatic = float(candidate["achromatic_fraction"])
    source_warm = float(source["warm_fraction"])
    candidate_warm = float(candidate["warm_fraction"])
    source_localized_skin_red = (
        float(source_fractions.get("red_or_pink", 0.0))
        if bool(source.get("red_or_pink_localized_to_head"))
        else 0.0
    )
    candidate_localized_skin_red = (
        float(candidate_fractions.get("red_or_pink", 0.0))
        if bool(candidate.get("red_or_pink_localized_to_head"))
        else 0.0
    )
    source_coat_warm = max(0.0, source_warm - source_localized_skin_red)
    candidate_coat_warm = max(
        0.0,
        candidate_warm - candidate_localized_skin_red,
    )
    source_chromatic = float(
        source.get("chromatic_fraction", max(0.0, 1.0 - source_achromatic))
    )
    candidate_chromatic = float(
        candidate.get(
            "chromatic_fraction",
            max(0.0, 1.0 - candidate_achromatic),
        )
    )
    source_saturated_warm = float(source.get("saturated_warm_fraction", 0.0))
    candidate_saturated_warm = float(candidate.get("saturated_warm_fraction", 0.0))
    source_fur_warm = float(source_fractions.get("warm_brown_or_orange", 0.0)) + float(
        source_fractions.get("yellow_or_gold", 0.0)
    )
    candidate_fur_warm = float(
        candidate_fractions.get("warm_brown_or_orange", 0.0)
    ) + float(candidate_fractions.get("yellow_or_gold", 0.0))
    source_gray_or_dark = float(source_fractions.get("black", 0.0)) + float(
        source_fractions.get("gray", 0.0)
    )
    candidate_gray_or_dark = float(candidate_fractions.get("black", 0.0)) + float(
        candidate_fractions.get("gray", 0.0)
    )
    achromatic_balance_deltas = {
        category: round(
            float(candidate_fractions.get(category, 0.0))
            - float(source_fractions.get(category, 0.0)),
            6,
        )
        for category in ("black", "gray", "white")
    }
    achromatic_balance_l1 = sum(
        abs(value) for value in achromatic_balance_deltas.values()
    )
    source_black_fraction = float(source_fractions.get("black", 0.0))
    candidate_black_fraction = float(candidate_fractions.get("black", 0.0))
    achromatic_dark_marking_lost = bool(
        source_black_fraction >= 0.08
        and candidate_black_fraction <= max(0.025, source_black_fraction * 0.35)
    )
    achromatic_dark_marking_expanded = bool(
        candidate_black_fraction - source_black_fraction >= 0.22
    )
    achromatic_value_balance_lost = bool(
        source_achromatic >= 0.72
        and candidate_achromatic >= 0.72
        and (achromatic_dark_marking_lost or achromatic_dark_marking_expanded)
    )
    if achromatic_value_balance_lost:
        issues.append("achromatic_coat_value_balance_lost")
    broad_warm_drift = bool(
        source_achromatic - candidate_achromatic >= 0.12
        and candidate_coat_warm - source_coat_warm >= 0.12
    )
    saturated_warm_drift = bool(
        candidate_saturated_warm >= 0.055
        and candidate_saturated_warm - source_saturated_warm >= 0.035
        and candidate_fur_warm - source_fur_warm >= 0.035
    )
    if identity_palette_is_achromatic(source) and (
        broad_warm_drift or saturated_warm_drift
    ):
        issues.append("achromatic_source_recolored_with_warm_fur")
    warm_coat_fraction_lost = bool(
        source_coat_warm >= IDENTITY_WARM_COAT_SOURCE_MIN
        and source_coat_warm - candidate_coat_warm
        >= IDENTITY_WARM_COAT_LOSS_DELTA_MIN
        and candidate_achromatic - source_achromatic
        >= IDENTITY_ACHROMATIC_EXPANSION_DELTA_MIN
    )
    if warm_coat_fraction_lost:
        issues.append("warm_source_recolored_achromatic")
    gray_or_dark_markings_expanded = bool(
        source_coat_warm >= IDENTITY_WARM_COAT_SOURCE_MIN
        and source_gray_or_dark <= IDENTITY_GRAY_OR_DARK_SOURCE_MAX
        and candidate_gray_or_dark - source_gray_or_dark
        >= IDENTITY_GRAY_OR_DARK_EXPANSION_DELTA_MIN
    )
    if gray_or_dark_markings_expanded:
        issues.append("new_gray_or_dark_markings_expanded")

    source_marking_spatial = source.get("marking_spatial_signature")
    candidate_marking_spatial = candidate.get("marking_spatial_signature")
    marking_spatial_available = bool(
        isinstance(source_marking_spatial, dict)
        and source_marking_spatial.get("available") is True
        and isinstance(candidate_marking_spatial, dict)
        and candidate_marking_spatial.get("available") is True
    )
    source_stripe_count = (
        int(source_marking_spatial.get("stripe_component_count", 0))
        if marking_spatial_available
        else 0
    )
    candidate_stripe_count = (
        int(candidate_marking_spatial.get("stripe_component_count", 0))
        if marking_spatial_available
        else 0
    )
    source_stripe_area = (
        float(source_marking_spatial.get("stripe_area_fraction", 0.0))
        if marking_spatial_available
        else 0.0
    )
    candidate_stripe_area = (
        float(candidate_marking_spatial.get("stripe_area_fraction", 0.0))
        if marking_spatial_available
        else 0.0
    )
    source_dark_contrast = (
        float(source_marking_spatial.get("dark_contrast_fraction", 0.0))
        if marking_spatial_available
        else 0.0
    )
    candidate_dark_contrast = (
        float(candidate_marking_spatial.get("dark_contrast_fraction", 0.0))
        if marking_spatial_available
        else 0.0
    )
    stripe_component_delta = candidate_stripe_count - source_stripe_count
    stripe_area_delta = candidate_stripe_area - source_stripe_area
    dark_contrast_delta = candidate_dark_contrast - source_dark_contrast
    source_stripe_evidence_present = bool(
        source_stripe_count >= IDENTITY_SOURCE_PRESENT_STRIPE_COMPONENT_MIN
        and source_stripe_area >= IDENTITY_SOURCE_PRESENT_STRIPE_AREA_MIN
    )
    if source_stripe_evidence_present:
        stripe_gate_mode = "source_present_amplification"
        stripe_component_delta_min = (
            IDENTITY_AMPLIFIED_STRIPE_COMPONENT_DELTA_MIN
        )
        stripe_area_delta_min = IDENTITY_AMPLIFIED_STRIPE_AREA_DELTA_MIN
        stripe_dark_contrast_delta_min = (
            IDENTITY_AMPLIFIED_STRIPE_DARK_CONTRAST_DELTA_MIN
        )
    else:
        stripe_gate_mode = "source_absent_invention"
        stripe_component_delta_min = IDENTITY_NEW_STRIPE_COMPONENT_DELTA_MIN
        stripe_area_delta_min = IDENTITY_NEW_STRIPE_AREA_DELTA_MIN
        stripe_dark_contrast_delta_min = (
            IDENTITY_NEW_STRIPE_DARK_CONTRAST_DELTA_MIN
        )
    invented_stripe_pattern = bool(
        marking_spatial_available
        and source_stripe_area <= IDENTITY_NEW_STRIPE_SOURCE_AREA_MAX
        and stripe_component_delta >= stripe_component_delta_min
        and stripe_area_delta >= stripe_area_delta_min
        and dark_contrast_delta >= stripe_dark_contrast_delta_min
    )
    if invented_stripe_pattern:
        issues.append("invented_stripe_pattern")
    marking_spatial_quality = {
        "available": marking_spatial_available,
        "policy": IDENTITY_MARKING_SPATIAL_POLICY,
        "acceptable": not invented_stripe_pattern,
        "invented_stripe_pattern": invented_stripe_pattern,
        "classification": (
            (
                "amplified_source_present_stripe_pattern"
                if source_stripe_evidence_present
                else "invented_source_absent_stripe_pattern"
            )
            if invented_stripe_pattern
            else None
        ),
        "stripe_gate_mode": stripe_gate_mode,
        "source_stripe_evidence_present": source_stripe_evidence_present,
        "source_stripe_component_count": source_stripe_count,
        "candidate_stripe_component_count": candidate_stripe_count,
        "stripe_component_delta": stripe_component_delta,
        "source_stripe_area_fraction": round(source_stripe_area, 6),
        "candidate_stripe_area_fraction": round(candidate_stripe_area, 6),
        "stripe_area_delta": round(stripe_area_delta, 6),
        "source_dark_contrast_fraction": round(source_dark_contrast, 6),
        "candidate_dark_contrast_fraction": round(candidate_dark_contrast, 6),
        "dark_contrast_delta": round(dark_contrast_delta, 6),
        "thresholds": {
            "source_stripe_area_max": IDENTITY_NEW_STRIPE_SOURCE_AREA_MAX,
            "source_present_component_min": (
                IDENTITY_SOURCE_PRESENT_STRIPE_COMPONENT_MIN
            ),
            "source_present_area_min": IDENTITY_SOURCE_PRESENT_STRIPE_AREA_MIN,
            "stripe_component_delta_min": stripe_component_delta_min,
            "stripe_area_delta_min": stripe_area_delta_min,
            "dark_contrast_delta_min": stripe_dark_contrast_delta_min,
            "source_absent": {
                "stripe_component_delta_min": (
                    IDENTITY_NEW_STRIPE_COMPONENT_DELTA_MIN
                ),
                "stripe_area_delta_min": IDENTITY_NEW_STRIPE_AREA_DELTA_MIN,
                "dark_contrast_delta_min": (
                    IDENTITY_NEW_STRIPE_DARK_CONTRAST_DELTA_MIN
                ),
            },
            "source_present_amplification": {
                "stripe_component_delta_min": (
                    IDENTITY_AMPLIFIED_STRIPE_COMPONENT_DELTA_MIN
                ),
                "stripe_area_delta_min": (
                    IDENTITY_AMPLIFIED_STRIPE_AREA_DELTA_MIN
                ),
                "dark_contrast_delta_min": (
                    IDENTITY_AMPLIFIED_STRIPE_DARK_CONTRAST_DELTA_MIN
                ),
            },
        },
    }
    chromatic_accents_lost = bool(
        source_chromatic >= 0.04
        and source_chromatic - candidate_chromatic >= 0.04
        and candidate_chromatic <= source_chromatic * 0.35
    )
    if chromatic_accents_lost:
        issues.append("source_chromatic_identity_accents_lost")
    if source_warm >= 0.18 and candidate_warm <= 0.06:
        warnings.append("source_warm_coat_color_may_be_hidden")
    l1_distance = sum(abs(value) for value in deltas.values())
    penalty = l1_distance * 50.0 + len(issues) * 100.0 + len(warnings) * 25.0
    return {
        "available": True,
        "acceptable": not issues,
        "warning": bool(issues or warnings),
        "issues": issues,
        "warnings": warnings,
        "penalty": round(penalty, 6),
        "deltas": deltas,
        "source_dominant": source.get("dominant", []),
        "candidate_dominant": candidate.get("dominant", []),
        "source_achromatic_fraction": source["achromatic_fraction"],
        "candidate_achromatic_fraction": candidate["achromatic_fraction"],
        "source_warm_fraction": source["warm_fraction"],
        "candidate_warm_fraction": candidate["warm_fraction"],
        "source_coat_warm_fraction": round(source_coat_warm, 6),
        "candidate_coat_warm_fraction": round(candidate_coat_warm, 6),
        "source_chromatic_fraction": round(source_chromatic, 6),
        "candidate_chromatic_fraction": round(candidate_chromatic, 6),
        "chromatic_accents_lost": chromatic_accents_lost,
        "source_saturated_warm_fraction": round(source_saturated_warm, 6),
        "candidate_saturated_warm_fraction": round(candidate_saturated_warm, 6),
        "source_fur_warm_fraction": round(source_fur_warm, 6),
        "candidate_fur_warm_fraction": round(candidate_fur_warm, 6),
        "source_gray_or_dark_fraction": round(source_gray_or_dark, 6),
        "candidate_gray_or_dark_fraction": round(candidate_gray_or_dark, 6),
        "warm_coat_fraction_lost": warm_coat_fraction_lost,
        "gray_or_dark_markings_expanded": gray_or_dark_markings_expanded,
        "bidirectional_drift_policy": (
            IDENTITY_PALETTE_BIDIRECTIONAL_DRIFT_POLICY
        ),
        "bidirectional_drift_thresholds": {
            "warm_coat_source_min": IDENTITY_WARM_COAT_SOURCE_MIN,
            "warm_coat_loss_delta_min": IDENTITY_WARM_COAT_LOSS_DELTA_MIN,
            "achromatic_expansion_delta_min": (
                IDENTITY_ACHROMATIC_EXPANSION_DELTA_MIN
            ),
            "gray_or_dark_source_max": IDENTITY_GRAY_OR_DARK_SOURCE_MAX,
            "gray_or_dark_expansion_delta_min": (
                IDENTITY_GRAY_OR_DARK_EXPANSION_DELTA_MIN
            ),
        },
        "marking_spatial_quality": marking_spatial_quality,
        "source_red_or_pink_body_fraction": round(
            float(source.get("red_or_pink_body_fraction", 0.0)),
            6,
        ),
        "candidate_red_or_pink_body_fraction": round(
            float(candidate.get("red_or_pink_body_fraction", 0.0)),
            6,
        ),
        "candidate_red_or_pink_body_share": round(
            float(candidate.get("red_or_pink_body_share", 0.0)),
            6,
        ),
        "candidate_red_or_pink_localized_to_head": bool(
            candidate.get("red_or_pink_localized_to_head", False)
        ),
        "source_green_or_cyan_fraction": round(source_green_or_cyan, 6),
        "candidate_green_or_cyan_fraction": round(
            candidate_green_or_cyan,
            6,
        ),
        "source_green_or_cyan_body_fraction": round(
            source_green_or_cyan_body,
            6,
        ),
        "candidate_green_or_cyan_body_fraction": round(
            candidate_green_or_cyan_body,
            6,
        ),
        "candidate_green_or_cyan_body_share": round(
            float(candidate.get("green_or_cyan_body_share", 0.0)),
            6,
        ),
        "candidate_green_or_cyan_localized_to_head": bool(
            candidate.get("green_or_cyan_localized_to_head", False)
        ),
        "green_or_cyan_body_spill": green_or_cyan_body_spill,
        "achromatic_balance_deltas": achromatic_balance_deltas,
        "achromatic_balance_l1": round(achromatic_balance_l1, 6),
        "source_black_fraction": round(source_black_fraction, 6),
        "candidate_black_fraction": round(candidate_black_fraction, 6),
        "black_fraction_delta": round(
            candidate_black_fraction - source_black_fraction,
            6,
        ),
        "achromatic_value_balance_lost": achromatic_value_balance_lost,
        "achromatic_dark_marking_lost": achromatic_dark_marking_lost,
        "achromatic_dark_marking_expanded": achromatic_dark_marking_expanded,
        "broad_warm_drift": broad_warm_drift,
        "saturated_warm_drift": saturated_warm_drift,
    }


def canonical_action_identity_palette_quality(
    quality: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Tighten achromatic value continuity for canonical-first action edits.

    The general uploaded-identity palette gate intentionally tolerates broad
    shadow/highlight redistribution while creating the initial canonical. Once
    that canonical is approved, an action edit has no such freedom: replacing
    a mostly white canonical with the synthetic gray action guide is identity
    drift even when both images remain globally achromatic.
    """
    _sync_local_video_runtime(globals(), {"canonical_action_identity_palette_quality"})

    if not isinstance(quality, dict):
        return quality
    issues = [str(issue) for issue in quality.get("issues", [])]
    warnings = [str(warning) for warning in quality.get("warnings", [])]
    source_achromatic = float(quality.get("source_achromatic_fraction", 0.0))
    candidate_achromatic = float(quality.get("candidate_achromatic_fraction", 0.0))
    achromatic_balance_l1 = float(quality.get("achromatic_balance_l1", 0.0))
    source_black = float(quality.get("source_black_fraction", 0.0))
    candidate_black = float(quality.get("candidate_black_fraction", 0.0))
    balance_drift = bool(
        quality.get("available") is True
        and source_achromatic >= 0.72
        and candidate_achromatic >= 0.72
        and achromatic_balance_l1 > CANONICAL_ACTION_ACHROMATIC_BALANCE_L1_MAX
    )
    issue = "canonical_action_achromatic_value_balance_drift"
    if balance_drift and issue not in issues:
        issues.append(issue)
    dark_feature_expansion = bool(
        quality.get("available") is True
        and source_achromatic >= 0.72
        and candidate_achromatic >= 0.72
        and source_black
        <= CANONICAL_ACTION_LIGHT_IDENTITY_BLACK_FRACTION_MAX
        and candidate_black - source_black
        > CANONICAL_ACTION_DARK_FEATURE_EXPANSION_MAX
    )
    dark_feature_issue = (
        "canonical_action_light_identity_dark_feature_expansion"
    )
    if dark_feature_expansion and dark_feature_issue not in issues:
        issues.append(dark_feature_issue)
    warm_expansion_drift = bool(
        quality.get("available") is True
        and (
            quality.get("broad_warm_drift") is True
            or quality.get("saturated_warm_drift") is True
        )
    )
    warm_expansion_issue = "canonical_action_warm_expansion_drift"
    if warm_expansion_drift and warm_expansion_issue not in issues:
        issues.append(warm_expansion_issue)
    acceptable = bool(quality.get("acceptable")) and not (
        balance_drift or dark_feature_expansion or warm_expansion_drift
    )
    return {
        **quality,
        "acceptable": acceptable,
        "warning": bool(issues or warnings),
        "issues": issues,
        "warnings": warnings,
        "penalty": round(
            float(quality.get("penalty", 0.0))
            + (100.0 if balance_drift else 0.0)
            + (100.0 if dark_feature_expansion else 0.0)
            + (100.0 if warm_expansion_drift else 0.0),
            6,
        ),
        "canonical_action_identity_palette_policy": (
            CANONICAL_ACTION_IDENTITY_PALETTE_POLICY
        ),
        "canonical_action_achromatic_balance_l1_max": (
            CANONICAL_ACTION_ACHROMATIC_BALANCE_L1_MAX
        ),
        "canonical_action_achromatic_balance_enforced": True,
        "canonical_action_achromatic_balance_drift": balance_drift,
        "canonical_action_light_identity_black_fraction_max": (
            CANONICAL_ACTION_LIGHT_IDENTITY_BLACK_FRACTION_MAX
        ),
        "canonical_action_dark_feature_expansion_max": (
            CANONICAL_ACTION_DARK_FEATURE_EXPANSION_MAX
        ),
        "canonical_action_dark_feature_expansion": dark_feature_expansion,
        "canonical_action_warm_expansion_enforced": True,
        "canonical_action_warm_expansion_drift": warm_expansion_drift,
    }


def advisory_grooming_minimum_palette_quality(
    source: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Reject unmistakable identity replacement in advisory grooming.

    This deliberately does not reuse the production identity-palette gate.  The
    advisory chain needs broad tolerance for pose-dependent lighting, but it
    must not accept a candidate whose coat distribution was taken from the
    profile action animal instead of the uploaded pet.
    """
    _sync_local_video_runtime(globals(), {"advisory_grooming_minimum_palette_quality"})

    if not source.get("available") or not candidate.get("available"):
        return {
            "available": False,
            "acceptable": False,
            "issues": ["minimum_identity_palette_signature_unavailable"],
            "warnings": [],
            "penalty": 1000.0,
            "source": source,
            "candidate": candidate,
        }
    source_fractions = source.get("fractions")
    candidate_fractions = candidate.get("fractions")
    if not isinstance(source_fractions, dict) or not isinstance(
        candidate_fractions,
        dict,
    ):
        return {
            "available": False,
            "acceptable": False,
            "issues": ["minimum_identity_palette_fractions_unavailable"],
            "warnings": [],
            "penalty": 1000.0,
            "source": source,
            "candidate": candidate,
        }
    deltas = {
        category: round(
            float(candidate_fractions.get(category, 0.0))
            - float(source_fractions.get(category, 0.0)),
            6,
        )
        for category in IDENTITY_PALETTE_CATEGORIES
    }
    l1_distance = float(sum(abs(value) for value in deltas.values()))
    black_delta = float(deltas.get("black", 0.0))
    chromatic_delta = float(candidate.get("chromatic_fraction", 0.0)) - float(
        source.get("chromatic_fraction", 0.0)
    )
    achromatic_balance_l1 = float(
        sum(
            abs(float(deltas.get(category, 0.0)))
            for category in ("black", "gray", "white")
        )
    )
    source_dominant = {str(value) for value in source.get("dominant", [])}
    candidate_dominant = {
        str(value) for value in candidate.get("dominant", [])
    }
    new_dark_dominant = bool(
        float(source_fractions.get("black", 0.0)) <= 0.05
        and "black" not in source_dominant
        and "black" in candidate_dominant
        and black_delta
        >= GROOMING_ADVISORY_IDENTITY_NEW_DARK_DOMINANT_DELTA_MIN
    )
    achromatic_balance_drift = bool(
        float(source.get("achromatic_fraction", 0.0)) >= 0.72
        and float(candidate.get("achromatic_fraction", 0.0)) >= 0.72
        and achromatic_balance_l1
        > GROOMING_ADVISORY_IDENTITY_ACHROMATIC_BALANCE_L1_MAX
    )
    issues: list[str] = []
    if l1_distance > GROOMING_ADVISORY_IDENTITY_PALETTE_L1_MAX:
        issues.append("gross_identity_palette_l1_drift")
    if abs(black_delta) > GROOMING_ADVISORY_IDENTITY_BLACK_FRACTION_DELTA_MAX:
        issues.append("gross_dark_marking_area_drift")
    if (
        abs(chromatic_delta)
        > GROOMING_ADVISORY_IDENTITY_CHROMATIC_FRACTION_DELTA_MAX
    ):
        issues.append("gross_chromatic_coat_fraction_drift")
    if achromatic_balance_drift:
        issues.append("gross_achromatic_value_balance_drift")
    if new_dark_dominant:
        issues.append("new_dominant_dark_markings")
    return {
        "available": True,
        "acceptable": not issues,
        "issues": issues,
        "warnings": [],
        "penalty": round(l1_distance * 50.0 + len(issues) * 100.0, 6),
        "deltas": deltas,
        "palette_l1_distance": round(l1_distance, 6),
        "black_fraction_delta": round(black_delta, 6),
        "chromatic_fraction_delta": round(chromatic_delta, 6),
        "achromatic_balance_l1": round(achromatic_balance_l1, 6),
        "achromatic_balance_drift": achromatic_balance_drift,
        "new_dark_dominant": new_dark_dominant,
        "thresholds": {
            "palette_l1_max": GROOMING_ADVISORY_IDENTITY_PALETTE_L1_MAX,
            "black_fraction_delta_max": (
                GROOMING_ADVISORY_IDENTITY_BLACK_FRACTION_DELTA_MAX
            ),
            "chromatic_fraction_delta_max": (
                GROOMING_ADVISORY_IDENTITY_CHROMATIC_FRACTION_DELTA_MAX
            ),
            "achromatic_balance_l1_max": (
                GROOMING_ADVISORY_IDENTITY_ACHROMATIC_BALANCE_L1_MAX
            ),
            "new_dark_dominant_delta_min": (
                GROOMING_ADVISORY_IDENTITY_NEW_DARK_DOMINANT_DELTA_MIN
            ),
        },
        "source_dominant": sorted(source_dominant),
        "candidate_dominant": sorted(candidate_dominant),
    }


def grooming_face_identity_signature(path: Path) -> dict[str, Any]:
    """Summarize only the upper head region for a broad severe-drift check."""
    _sync_local_video_runtime(globals(), {"grooming_face_identity_signature"})

    try:
        with Image.open(path) as loaded:
            image = loaded.convert("RGBA")
        if max(image.size) > 512:
            scale = 512.0 / float(max(image.size))
            image = image.resize(
                (
                    max(1, int(round(image.width * scale))),
                    max(1, int(round(image.height * scale))),
                ),
                Image.Resampling.LANCZOS,
            )
        rgba = np.asarray(image, dtype=np.uint8)
        foreground, mask_method = _largest_green_or_alpha_foreground(rgba)
    except Exception as error:  # noqa: BLE001 - optional identity check degrades safely
        return {
            "available": False,
            "reason": f"face_foreground_unavailable:{type(error).__name__}",
        }
    ys, xs = np.nonzero(foreground)
    x0, y0 = int(xs.min()), int(ys.min())
    x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1
    head_cutoff = y0 + int(round((y1 - y0) * 0.46))
    row_grid = np.indices(foreground.shape)[0]
    head_mask = foreground & (row_grid <= head_cutoff)
    if min(head_mask.shape) >= 24:
        head_mask = cv2.erode(
            head_mask.astype(np.uint8),
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        ).astype(bool)
    pixels = rgba[:, :, :3][head_mask]
    if len(pixels) < 256:
        return {
            "available": False,
            "reason": "insufficient_face_pixels",
            "mask_method": mask_method,
        }
    hsv = cv2.cvtColor(
        pixels.reshape(-1, 1, 3),
        cv2.COLOR_RGB2HSV,
    ).reshape(-1, 3)
    hue = hsv[:, 0]
    saturation = hsv[:, 1]
    value = hsv[:, 2]
    black = value < 58
    white = (value >= 190) & (saturation < 52)
    gray = ~(black | white) & (saturation < 52)
    chromatic = ~(black | white | gray)
    yellow = chromatic & (hue >= 28) & (hue < 42)
    green = chromatic & (hue >= 42) & (hue < 88)
    warm = chromatic & (hue >= 8) & (hue < 42)
    total = float(len(pixels))
    return {
        "available": True,
        "mask_method": mask_method,
        "subject_bbox": [x0, y0, x1, y1],
        "head_cutoff_y": head_cutoff,
        "head_pixels": len(pixels),
        "fractions": {
            "black": round(float(np.count_nonzero(black)) / total, 6),
            "white": round(float(np.count_nonzero(white)) / total, 6),
            "gray": round(float(np.count_nonzero(gray)) / total, 6),
            "chromatic": round(float(np.count_nonzero(chromatic)) / total, 6),
            "yellow_or_gold": round(float(np.count_nonzero(yellow)) / total, 6),
            "green": round(float(np.count_nonzero(green)) / total, 6),
            "warm_brown_or_yellow": round(
                float(np.count_nonzero(warm)) / total,
                6,
            ),
        },
    }


def advisory_grooming_face_identity_quality(
    source: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Block only unmistakable face recoloring; normal action variation remains valid."""
    _sync_local_video_runtime(globals(), {"advisory_grooming_face_identity_quality"})

    if not source.get("available") or not candidate.get("available"):
        return {
            "available": False,
            "acceptable": True,
            "issues": [],
            "warnings": ["grooming_face_identity_check_unavailable"],
            "penalty": 0.0,
            "source": source,
            "candidate": candidate,
        }
    source_fractions = source.get("fractions")
    candidate_fractions = candidate.get("fractions")
    if not isinstance(source_fractions, dict) or not isinstance(
        candidate_fractions, dict
    ):
        return {
            "available": False,
            "acceptable": True,
            "issues": [],
            "warnings": ["grooming_face_identity_fractions_unavailable"],
            "penalty": 0.0,
            "source": source,
            "candidate": candidate,
        }
    black_delta = float(candidate_fractions.get("black", 0.0)) - float(
        source_fractions.get("black", 0.0)
    )
    chromatic_delta = float(
        candidate_fractions.get("chromatic", 0.0)
    ) - float(source_fractions.get("chromatic", 0.0))
    yellow_delta = float(
        candidate_fractions.get("yellow_or_gold", 0.0)
    ) - float(source_fractions.get("yellow_or_gold", 0.0))
    source_green = float(source_fractions.get("green", 0.0))
    candidate_green = float(candidate_fractions.get("green", 0.0))
    eye_hue_replacement = bool(
        source_green >= 0.004
        and candidate_green <= max(0.002, source_green * 0.35)
        and yellow_delta >= GROOMING_ADVISORY_FACE_GREEN_TO_YELLOW_DELTA_MIN
    )
    issues: list[str] = []
    if abs(black_delta) > GROOMING_ADVISORY_FACE_DARK_FRACTION_DELTA_MAX:
        issues.append("grooming_face_dark_marking_drift")
    if chromatic_delta > GROOMING_ADVISORY_FACE_CHROMATIC_FRACTION_DELTA_MAX:
        issues.append("grooming_face_gross_chromatic_drift")
    if (
        yellow_delta > GROOMING_ADVISORY_FACE_YELLOW_FRACTION_DELTA_MAX
        or eye_hue_replacement
    ):
        issues.append("grooming_face_eye_hue_replacement")
    return {
        "available": True,
        "acceptable": not issues,
        "issues": issues,
        "warnings": [],
        "penalty": round(
            abs(black_delta) * 40.0
            + max(0.0, chromatic_delta) * 40.0
            + max(0.0, yellow_delta) * 60.0
            + len(issues) * 100.0,
            6,
        ),
        "black_fraction_delta": round(black_delta, 6),
        "chromatic_fraction_delta": round(chromatic_delta, 6),
        "yellow_fraction_delta": round(yellow_delta, 6),
        "source_green_fraction": round(source_green, 6),
        "candidate_green_fraction": round(candidate_green, 6),
        "eye_hue_replacement": eye_hue_replacement,
        "thresholds": {
            "dark_fraction_delta_max": (
                GROOMING_ADVISORY_FACE_DARK_FRACTION_DELTA_MAX
            ),
            "chromatic_fraction_delta_max": (
                GROOMING_ADVISORY_FACE_CHROMATIC_FRACTION_DELTA_MAX
            ),
            "yellow_fraction_delta_max": (
                GROOMING_ADVISORY_FACE_YELLOW_FRACTION_DELTA_MAX
            ),
            "green_to_yellow_delta_min": (
                GROOMING_ADVISORY_FACE_GREEN_TO_YELLOW_DELTA_MIN
            ),
        },
        "source": source,
        "candidate": candidate,
    }


def advisory_grooming_minimum_pose_quality(
    pose_quality: dict[str, Any],
) -> dict[str, Any]:
    """Fail closed unless the candidate has a continuous paw-to-mouth action."""
    _sync_local_video_runtime(globals(), {"advisory_grooming_minimum_pose_quality"})

    if pose_quality.get("available") is not True:
        return {
            "available": False,
            "acceptable": False,
            "issues": ["minimum_grooming_pose_evidence_unavailable"],
            "warnings": list(pose_quality.get("issues") or []),
            "penalty": 1000.0,
            "full_pose_quality": pose_quality,
        }
    measured_issues = [str(issue) for issue in pose_quality.get("issues", [])]
    metrics = pose_quality.get("metrics")
    appearance_consistency = (
        metrics.get("appearance_consistency")
        if isinstance(metrics, dict)
        and isinstance(metrics.get("appearance_consistency"), dict)
        else {}
    )
    raised_appearance = (
        appearance_consistency.get("raised")
        if isinstance(appearance_consistency.get("raised"), dict)
        else {}
    )
    texture_ratio = float(
        appearance_consistency.get("texture_gradient_p90_ratio") or 0.0
    )
    raised_gradient_p90 = float(
        raised_appearance.get("gradient_p90") or 0.0
    )
    hard_object_texture_signature = bool(
        appearance_consistency.get("available") is True
        and texture_ratio > GROOMING_ADVISORY_HARD_OBJECT_TEXTURE_RATIO_MAX
        and raised_gradient_p90
        >= GROOMING_ADVISORY_HARD_OBJECT_RAISED_GRADIENT_P90_MIN
    )
    if hard_object_texture_signature:
        measured_issues.append("raised_paw_hard_object_texture_signature")
    blocking_issues = [
        issue
        for issue in measured_issues
        if issue in GROOMING_ADVISORY_BLOCKING_POSE_ISSUES
    ]
    warnings = [issue for issue in measured_issues if issue not in blocking_issues]
    return {
        "available": True,
        "acceptable": not blocking_issues,
        "issues": blocking_issues,
        "warnings": warnings,
        "penalty": round(100.0 * len(blocking_issues), 6),
        "blocking_issue_policy": sorted(GROOMING_ADVISORY_BLOCKING_POSE_ISSUES),
        "hard_object_texture_evidence": {
            "available": bool(appearance_consistency.get("available") is True),
            "texture_gradient_p90_ratio": round(texture_ratio, 6),
            "texture_gradient_p90_ratio_max": (
                GROOMING_ADVISORY_HARD_OBJECT_TEXTURE_RATIO_MAX
            ),
            "raised_gradient_p90": round(raised_gradient_p90, 6),
            "raised_gradient_p90_min": (
                GROOMING_ADVISORY_HARD_OBJECT_RAISED_GRADIENT_P90_MIN
            ),
            "blocking": hard_object_texture_signature,
        },
        "full_pose_quality": pose_quality,
    }


def advisory_grooming_action_retry_guidance(
    gate_metrics: dict[str, Any],
    *,
    attempt: int,
) -> dict[str, Any]:
    """Request another prompt-only draft before terminal best-effort selection."""
    _sync_local_video_runtime(globals(), {"advisory_grooming_action_retry_guidance"})

    retry_issues = list(
        dict.fromkeys(
            str(issue)
            for issue in [
                *gate_metrics.get("issues", []),
                *gate_metrics.get("warnings", []),
            ]
            if str(issue) in GROOMING_ADVISORY_ACTION_RETRY_WARNINGS
        )
    )
    return {
        "warnings": retry_issues,
        "blocking": bool(retry_issues),
        "requested": bool(
            retry_issues
            and int(attempt) < GROOMING_ADVISORY_ACTION_RETRY_BEFORE_ATTEMPT
        ),
        "never_blocks_final_attempt": True,
        "final_attempt_fail_closed": False,
    }


def advisory_grooming_canonical_action_edit_quality(
    canonical_reference: Path,
    candidate: Path,
    *,
    attempt: int = 1,
) -> dict[str, Any]:
    """Reject a neutral canonical copied through the local grooming edit mask.

    The absolute pose gate can mistake the canonical chest and planted foreleg
    for a shoulder-to-mouth contour.  This independent check compares only the
    job's canonical identity image with the raw Qwen candidate.  A valid action
    draft must materially redraw the local foreleg corridor and must not retain
    the original planted-paw edge topology.  No driver frame or pose image is
    loaded here or exposed to Qwen.
    """
    _sync_local_video_runtime(globals(), {"advisory_grooming_canonical_action_edit_quality"})

    try:
        with Image.open(canonical_reference) as loaded:
            canonical = np.asarray(loaded.convert("RGBA"), dtype=np.uint8)
        with Image.open(candidate) as loaded:
            generated = np.asarray(loaded.convert("RGBA"), dtype=np.uint8)
        if canonical.shape != (512, 512, 4) or generated.shape != canonical.shape:
            raise ValueError("canonical and candidate must be 512x512 RGBA")
        canonical_foreground, canonical_method = (
            _largest_green_or_alpha_foreground(canonical)
        )
        candidate_foreground, candidate_method = (
            _largest_green_or_alpha_foreground(generated)
        )
    except (OSError, RuntimeError, ValueError) as exc:
        return {
            "available": False,
            "acceptable": False,
            "issues": ["canonical_grooming_action_edit_evidence_unavailable"],
            "warnings": [],
            "penalty": 1000.0,
            "error": str(exc),
        }

    ys, xs = np.nonzero(canonical_foreground)
    if not len(xs):
        return {
            "available": False,
            "acceptable": False,
            "issues": ["canonical_grooming_action_edit_evidence_unavailable"],
            "warnings": [],
            "penalty": 1000.0,
            "error": "canonical foreground is empty",
        }
    x0, y0 = int(xs.min()), int(ys.min())
    x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1
    width = float(x1 - x0)
    height = float(y1 - y0)
    rows, columns = np.indices(canonical_foreground.shape)

    action_corridor = (
        (columns >= x0 + width * 0.18)
        & (columns < x0 + width * 0.62)
        & (rows >= y0 + height * 0.22)
        & (rows < y0 + height * 0.96)
    )
    planted_paw_region = (
        (
            (columns - (x0 + width * 0.38))
            / max(1.0, width * 0.17)
        )
        ** 2
        + (
            (rows - (y0 + height * 0.91))
            / max(1.0, height * 0.075)
        )
        ** 2
        <= 1.0
    ) & (rows >= y0 + height * 0.82)

    rgb_delta = np.mean(
        np.abs(
            generated[:, :, :3].astype(np.float32)
            - canonical[:, :, :3].astype(np.float32)
        ),
        axis=2,
    ) / 255.0
    action_pixels = int(np.count_nonzero(action_corridor))
    action_mean_delta = float(np.mean(rgb_delta[action_corridor]))
    action_changed_fraction = float(
        np.count_nonzero(rgb_delta[action_corridor] > 0.08)
    ) / float(max(1, action_pixels))

    def structural_edges(rgba: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=2.5, sigmaY=2.5)
        return cv2.Canny(blurred, 20, 40) > 0

    canonical_edges = structural_edges(canonical) & planted_paw_region
    candidate_edges = structural_edges(generated)
    candidate_edge_support = cv2.dilate(
        candidate_edges.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    ).astype(bool)
    canonical_paw_edge_count = int(np.count_nonzero(canonical_edges))
    planted_paw_edge_retention = float(
        np.count_nonzero(canonical_edges & candidate_edge_support)
    ) / float(max(1, canonical_paw_edge_count))
    planted_paw_rgb_delta = float(np.mean(rgb_delta[planted_paw_region]))
    planted_paw_foreground_removed_fraction = float(
        np.count_nonzero(
            canonical_foreground
            & planted_paw_region
            & ~candidate_foreground
        )
    ) / float(
        max(
            1,
            np.count_nonzero(canonical_foreground & planted_paw_region),
        )
    )

    threshold_index = min(
        max(0, int(attempt) - 1),
        len(GROOMING_ADVISORY_CANONICAL_ACTION_THRESHOLD_SCALE_BY_ATTEMPT) - 1,
    )
    threshold_scale = float(
        GROOMING_ADVISORY_CANONICAL_ACTION_THRESHOLD_SCALE_BY_ATTEMPT[
            threshold_index
        ]
    )
    edge_retention_relaxation = float(
        GROOMING_ADVISORY_CANONICAL_EDGE_RETENTION_RELAXATION_BY_ATTEMPT[
            threshold_index
        ]
    )
    action_mean_delta_min = (
        GROOMING_ADVISORY_CANONICAL_ACTION_EDIT_MEAN_DELTA_MIN * threshold_scale
    )
    action_changed_fraction_min = (
        GROOMING_ADVISORY_CANONICAL_ACTION_EDIT_CHANGED_FRACTION_MIN
        * threshold_scale
    )
    planted_paw_edge_retention_max = min(
        0.95,
        GROOMING_ADVISORY_CANONICAL_PLANTED_PAW_EDGE_RETENTION_MAX
        + edge_retention_relaxation,
    )
    planted_paw_rgb_delta_min = (
        GROOMING_ADVISORY_CANONICAL_PLANTED_PAW_RGB_DELTA_MIN * threshold_scale
    )
    action_edit_too_weak = bool(
        action_mean_delta
        < action_mean_delta_min
        or action_changed_fraction
        < action_changed_fraction_min
    )
    original_planted_paw_retained = bool(
        planted_paw_foreground_removed_fraction
        < GROOMING_ADVISORY_CANONICAL_PLANTED_PAW_FOREGROUND_REMOVED_FRACTION_MIN
        and (
            planted_paw_edge_retention > planted_paw_edge_retention_max
            or planted_paw_rgb_delta < planted_paw_rgb_delta_min
        )
    )
    issues: list[str] = []
    if action_edit_too_weak:
        issues.append("canonical_grooming_action_edit_too_weak")
    if original_planted_paw_retained:
        issues.append("original_planted_forepaw_topology_retained")
    return {
        "available": True,
        "acceptable": not issues,
        "issues": issues,
        "warnings": [],
        "penalty": round(100.0 * len(issues), 6),
        "policy": "canonical_relative_grooming_action_topology_v1",
        "attempt": int(attempt),
        "threshold_relaxation": {
            "policy": "strict_then_slightly_relaxed_by_attempt_v1",
            "scale": round(threshold_scale, 6),
            "edge_retention_relaxation": round(edge_retention_relaxation, 6),
        },
        "canonical_reference_path": str(canonical_reference),
        "canonical_reference_sha256": file_sha256(canonical_reference),
        "candidate_path": str(candidate),
        "candidate_sha256": file_sha256(candidate),
        "canonical_foreground_method": canonical_method,
        "candidate_foreground_method": candidate_method,
        "canonical_foreground_bbox": [x0, y0, x1, y1],
        "metrics": {
            "action_corridor_mean_rgb_delta": round(action_mean_delta, 6),
            "action_corridor_changed_fraction": round(
                action_changed_fraction,
                6,
            ),
            "original_planted_paw_edge_count": canonical_paw_edge_count,
            "original_planted_paw_edge_retention": round(
                planted_paw_edge_retention,
                6,
            ),
            "original_planted_paw_mean_rgb_delta": round(
                planted_paw_rgb_delta,
                6,
            ),
            "original_planted_paw_foreground_removed_fraction": round(
                planted_paw_foreground_removed_fraction,
                6,
            ),
        },
        "thresholds": {
            "action_corridor_mean_rgb_delta_min": (
                round(action_mean_delta_min, 6)
            ),
            "action_corridor_changed_fraction_min": (
                round(action_changed_fraction_min, 6)
            ),
            "original_planted_paw_edge_retention_max": (
                round(planted_paw_edge_retention_max, 6)
            ),
            "original_planted_paw_mean_rgb_delta_min": (
                round(planted_paw_rgb_delta_min, 6)
            ),
            "original_planted_paw_foreground_removed_fraction_min": (
                GROOMING_ADVISORY_CANONICAL_PLANTED_PAW_FOREGROUND_REMOVED_FRACTION_MIN
            ),
        },
        "driver_frame_loaded": False,
        "video_first_frame_loaded_by_qwen": False,
        "post_qwen_pixel_composite_enabled": False,
    }


def evaluate_advisory_grooming_minimum_gate(
    candidate: Path,
    *,
    identity_reference_palette: dict[str, Any],
    identity_reference_path: Path,
    identity_palette_reference_path: Path,
    geometry_guide: Path,
    geometry_metadata: Path,
    attempt: int = 1,
) -> dict[str, Any]:
    """Require recognizable identity and one real paw-to-mouth grooming limb.

    ``identity_reference_path`` is the normalized 512x512 canonical used to
    prove that Qwen actually edited the neutral pose.  Coat color remains
    independently anchored to ``identity_palette_reference_path``, the raw
    uploaded identity image from which ``identity_reference_palette`` was
    measured.  Keeping those roles separate prevents arbitrary upload sizes
    from disabling the canonical action-edit proof.
    """
    _sync_local_video_runtime(globals(), {"evaluate_advisory_grooming_minimum_gate"})

    candidate_palette = identity_palette_signature(candidate)
    palette_quality = advisory_grooming_minimum_palette_quality(
        identity_reference_palette,
        candidate_palette,
    )
    face_source_signature = grooming_face_identity_signature(identity_reference_path)
    face_candidate_signature = grooming_face_identity_signature(candidate)
    face_quality = advisory_grooming_face_identity_quality(
        face_source_signature,
        face_candidate_signature,
    )
    canonical_action_edit_quality = (
        advisory_grooming_canonical_action_edit_quality(
            identity_reference_path,
            candidate,
            attempt=attempt,
        )
    )
    full_pose_quality = evaluate_grooming_pose(
        candidate,
        geometry_guide,
        geometry_metadata=geometry_metadata,
    )
    pose_quality = advisory_grooming_minimum_pose_quality(full_pose_quality)
    issues = list(
        dict.fromkeys(
            [
                *[str(issue) for issue in palette_quality.get("issues", [])],
                *[str(issue) for issue in face_quality.get("issues", [])],
                *[
                    str(issue)
                    for issue in canonical_action_edit_quality.get("issues", [])
                ],
                *[str(issue) for issue in pose_quality.get("issues", [])],
            ]
        )
    )
    warnings = list(
        dict.fromkeys(
            [
                *[str(issue) for issue in palette_quality.get("warnings", [])],
                *[str(issue) for issue in face_quality.get("warnings", [])],
                *[
                    str(issue)
                    for issue in canonical_action_edit_quality.get("warnings", [])
                ],
                *[str(issue) for issue in pose_quality.get("warnings", [])],
            ]
        )
    )
    return {
        "policy": GROOMING_ADVISORY_MINIMUM_GATE_POLICY,
        "version": GROOMING_ADVISORY_MINIMUM_GATE_VERSION,
        "available": bool(
            palette_quality.get("available")
            and canonical_action_edit_quality.get("available")
            and pose_quality.get("available")
        ),
        "acceptable": not issues,
        "attempt": int(attempt),
        "issues": issues,
        "warnings": warnings,
        "penalty": round(
            float(palette_quality.get("penalty", 1000.0))
            + float(face_quality.get("penalty", 0.0))
            + float(canonical_action_edit_quality.get("penalty", 1000.0))
            + float(pose_quality.get("penalty", 1000.0)),
            6,
        ),
        "candidate_path": str(candidate),
        "candidate_sha256": file_sha256(candidate),
        "identity_palette_reference_path": str(identity_palette_reference_path),
        "identity_palette_reference_sha256": file_sha256(
            identity_palette_reference_path
        ),
        "identity_reference_path": str(identity_reference_path),
        "identity_reference_sha256": file_sha256(identity_reference_path),
        "geometry_guide_path": str(geometry_guide),
        "geometry_guide_sha256": file_sha256(geometry_guide),
        "geometry_metadata_path": str(geometry_metadata),
        "geometry_metadata_sha256": file_sha256(geometry_metadata),
        "identity_palette": palette_quality,
        "face_identity": face_quality,
        "canonical_action_edit": canonical_action_edit_quality,
        "pose": pose_quality,
    }


def advisory_grooming_degraded_candidate_selection_key(
    record: dict[str, Any],
) -> tuple[int, int, int, int, float, float, int]:
    """Rank a complete raw Qwen draft when the strict action gate is exhausted.

    Identity replacement and obvious hard-object anatomy are kept ahead of the
    action score.  Within equally safe candidates, prefer fewer issues, more
    removal of the original planted-paw topology, then lower total penalty.
    """
    _sync_local_video_runtime(globals(), {"advisory_grooming_degraded_candidate_selection_key"})

    gate = record.get("grooming_advisory_minimum_gate")
    gate = gate if isinstance(gate, dict) else {}
    issues = {str(issue) for issue in gate.get("issues", [])}
    palette = gate.get("identity_palette")
    face = gate.get("face_identity")
    canonical_edit = gate.get("canonical_action_edit")
    canonical_edit = canonical_edit if isinstance(canonical_edit, dict) else {}
    metrics = canonical_edit.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    identity_unsafe = bool(
        not isinstance(palette, dict)
        or palette.get("acceptable") is not True
        or (isinstance(face, dict) and face.get("acceptable") is not True)
    )
    hard_object = "raised_paw_hard_object_texture_signature" in issues
    evidence_missing = bool(
        gate.get("available") is not True
        or canonical_edit.get("available") is not True
    )
    edge_retention = float(
        metrics.get("original_planted_paw_edge_retention") or 1.0
    )
    action_change = float(metrics.get("action_corridor_mean_rgb_delta") or 0.0)
    combined_penalty = float(record.get("combined_penalty") or 1000.0)
    return (
        int(identity_unsafe),
        int(hard_object),
        int(evidence_missing),
        len(issues),
        round(edge_retention - action_change, 6),
        combined_penalty,
        int(record.get("attempt") or 0),
    )


def apply_canonical_grooming_organic_paw_palette_evidence(
    quality: dict[str, Any] | None,
    pose_gate: dict[str, Any] | None,
    *,
    attempt: int,
) -> dict[str, Any] | None:
    """Identify bounded achromatic drift on a proven direct organic paw.

    This is an eligibility probe, not the final identity verdict. It does not
    forgive a new hue or lost/expanded dark markings. Both off-guide and raw
    pose-pass routes require strict raised/planted paw material evidence, so a
    curved beige/red hanging object cannot request palette normalization.
    """
    _sync_local_video_runtime(globals(), {"apply_canonical_grooming_organic_paw_palette_evidence"})

    if not isinstance(quality, dict):
        return quality
    result = dict(quality)

    def number(value: Any) -> float | None:
        try:
            converted = float(value)
        except (TypeError, ValueError):
            return None
        return converted if np.isfinite(converted) else None

    pose_gate = pose_gate if isinstance(pose_gate, dict) else {}
    structural = pose_gate.get("canonical_direct_structural_edge_evidence")
    structural = structural if isinstance(structural, dict) else {}
    offguide = structural.get("offguide_organic_paw_evidence")
    offguide = offguide if isinstance(offguide, dict) else {}
    aligned = structural.get("aligned_organic_paw_evidence")
    aligned = aligned if isinstance(aligned, dict) else {}
    raw_issues = [str(issue) for issue in quality.get("issues", [])]
    source_achromatic = number(quality.get("source_achromatic_fraction"))
    candidate_achromatic = number(quality.get("candidate_achromatic_fraction"))
    balance_l1 = number(quality.get("achromatic_balance_l1"))
    checks = {
        "direct_attempt_in_scope": bool(
            isinstance(attempt, int)
            and 1 <= attempt <= GROOMING_CANONICAL_DIRECT_EDGE_EVIDENCE_MAX_ATTEMPT
        ),
        "organic_paw_evidence_accepted": bool(
            structural.get("accepted") is True
            and (
                (
                    offguide.get("accepted") is True
                    and pose_gate.get("verdict")
                    == "accepted_canonical_direct_offguide_organic_paw_evidence"
                )
                or (
                    aligned.get("accepted") is True
                    and pose_gate.get("verdict")
                    == "accepted_canonical_direct_aligned_organic_paw_evidence"
                )
            )
        ),
        "only_achromatic_balance_issue": bool(
            raw_issues == ["canonical_action_achromatic_value_balance_drift"]
        ),
        "source_and_candidate_remain_achromatic": bool(
            source_achromatic is not None
            and candidate_achromatic is not None
            and source_achromatic
            >= GROOMING_CANONICAL_DIRECT_ORGANIC_PAW_ACHROMATIC_MIN
            and candidate_achromatic
            >= GROOMING_CANONICAL_DIRECT_ORGANIC_PAW_ACHROMATIC_MIN
        ),
        "bounded_value_redistribution": bool(
            balance_l1 is not None
            and balance_l1
            <= GROOMING_CANONICAL_DIRECT_ORGANIC_PAW_BALANCE_L1_MAX
        ),
        "no_warm_or_saturated_drift": bool(
            quality.get("broad_warm_drift") is False
            and quality.get("saturated_warm_drift") is False
        ),
        "dark_markings_not_lost_or_expanded": bool(
            quality.get("achromatic_dark_marking_lost") is False
            and quality.get("achromatic_dark_marking_expanded") is False
        ),
    }
    accepted = bool(
        quality.get("available") is True
        and quality.get("acceptable") is not True
        and all(checks.values())
    )
    evidence = {
        "policy": GROOMING_CANONICAL_DIRECT_ORGANIC_PAW_PALETTE_POLICY,
        "evaluated": True,
        "accepted": accepted,
        "fail_closed": True,
        "attempt": attempt,
        "achromatic_fraction_min": (
            GROOMING_CANONICAL_DIRECT_ORGANIC_PAW_ACHROMATIC_MIN
        ),
        "achromatic_balance_l1": balance_l1,
        "achromatic_balance_l1_max": (
            GROOMING_CANONICAL_DIRECT_ORGANIC_PAW_BALANCE_L1_MAX
        ),
        "new_hue_or_marking_drift_allowed": False,
        "independent_fur_surface_gate_still_required": True,
        "checks": checks,
    }
    result["canonical_direct_organic_paw_palette_evidence"] = evidence
    result["grooming_achromatic_palette_normalization_eligible"] = accepted
    if not accepted:
        return result
    result["normalization_eligibility_verdict"] = (
        "eligible_for_organic_paw_achromatic_histogram_match"
    )
    result["warnings"] = list(
        dict.fromkeys(
            [
                *[str(item) for item in quality.get("warnings", [])],
                "bounded_achromatic_value_drift_eligible_only_for_strict_"
                "direct_organic_paw_histogram_normalization",
            ]
        )
    )
    return result


def canonical_grooming_palette_normalization_eligibility(
    quality: dict[str, Any] | None,
    pose_gate: dict[str, Any] | None,
    *,
    attempt: int,
) -> dict[str, Any]:
    """Prove that a direct organic-paw candidate may be luma-normalized."""
    _sync_local_video_runtime(globals(), {"canonical_grooming_palette_normalization_eligibility"})

    probe = apply_canonical_grooming_organic_paw_palette_evidence(
        quality,
        pose_gate,
        attempt=attempt,
    )
    evidence = (
        probe.get("canonical_direct_organic_paw_palette_evidence")
        if isinstance(probe, dict)
        else None
    )
    evidence = dict(evidence) if isinstance(evidence, dict) else {
        "policy": GROOMING_CANONICAL_DIRECT_ORGANIC_PAW_PALETTE_POLICY,
        "evaluated": True,
        "accepted": False,
        "fail_closed": True,
        "attempt": attempt,
        "checks": {},
    }
    evidence.update(
        eligible=bool(evidence.get("accepted") is True),
        operation="foreground_lab_l_histogram_match_only",
        source_alpha_geometry_must_remain_exact=True,
        source_ab_chroma_must_remain_with_roundtrip_delta_lte=2,
        pose_gate_must_be_re_evaluated=True,
        identity_palette_gate_must_be_re_evaluated=True,
        fur_surface_gate_must_run_afterward=True,
    )
    return evidence


def canonical_grooming_pose_safe_identity_refine_eligibility(
    *,
    attempt: int,
    seed_bound_matte: dict[str, Any] | None,
    pose_gate: dict[str, Any] | None,
    pose_quality: dict[str, Any] | None,
    identity_palette_quality: dict[str, Any] | None,
    fur_surface_gate: dict[str, Any] | None,
) -> dict[str, Any]:
    """Admit only attempt 4's proven real grooming pose as attempt 5 sampler."""
    _sync_local_video_runtime(globals(), {"canonical_grooming_pose_safe_identity_refine_eligibility"})

    matte = seed_bound_matte if isinstance(seed_bound_matte, dict) else {}
    gate = pose_gate if isinstance(pose_gate, dict) else {}
    quality = pose_quality if isinstance(pose_quality, dict) else {}
    palette = (
        identity_palette_quality
        if isinstance(identity_palette_quality, dict)
        else {}
    )
    surface = fur_surface_gate if isinstance(fur_surface_gate, dict) else {}
    metrics = gate.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    structural = metrics.get("raised_foreleg_structural_evidence")
    structural = structural if isinstance(structural, dict) else {}
    chamfer = metrics.get("chamfer")
    chamfer = chamfer if isinstance(chamfer, dict) else {}
    matte_checks = matte.get("checks")
    matte_checks = matte_checks if isinstance(matte_checks, dict) else {}
    surface_provenance = surface.get("gate")
    surface_provenance = (
        surface_provenance if isinstance(surface_provenance, dict) else {}
    )

    def finite_number(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if np.isfinite(number) else None

    structural_density = finite_number(structural.get("edge_density"))
    chamfer_coverage = finite_number(chamfer.get("coverage_within_radius"))
    chamfer_mean = finite_number(chamfer.get("mean_distance_px"))
    chamfer_p90 = finite_number(chamfer.get("p90_distance_px"))
    palette_issues = {
        str(issue) for issue in palette.get("issues", []) if str(issue).strip()
    }
    unexpected_palette_issues = sorted(
        palette_issues - GROOMING_CANONICAL_POSE_SAFE_IDENTITY_ONLY_ISSUES
    )
    required_digests = [
        str(matte.get(name) or "")
        for name in (
            "source_sha256",
            "semantic_cutout_sha256",
            "pose_transfer_seed_sha256",
            "target_sha256",
            "union_alpha_sha256",
        )
    ]
    digest_bound = bool(
        all(
            len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
            for digest in required_digests
        )
    )
    checks = {
        "attempt_is_deterministic_full_pose_primary": bool(
            attempt == GROOMING_CANONICAL_FULL_POSE_RESCUE_ATTEMPT
        ),
        "seed_bound_matte_policy_exact": bool(
            matte.get("policy") == GROOMING_CANONICAL_SEED_BOUND_MATTE_POLICY
            and matte.get("acceptable") is True
            and matte.get("fail_closed") is True
        ),
        "seed_bound_matte_all_checks_passed": bool(
            matte_checks and all(value is True for value in matte_checks.values())
        ),
        "seed_bound_matte_digest_complete": digest_bound,
        "seed_used_for_alpha_geometry_only": bool(
            matte.get("pose_transfer_seed_used_for_alpha_geometry_only") is True
            and matte.get("pose_transfer_seed_rgb_copied") is False
            and matte.get("pose_transfer_seed_luma_or_chroma_copied") is False
        ),
        "complete_subject_quality_passed": bool(quality.get("acceptable") is True),
        "raw_pose_gate_passed": bool(gate.get("acceptable") is True),
        "organic_structural_density_passed": bool(
            structural.get("available") is True
            and structural_density is not None
            and structural_density
            >= GROOMING_CANONICAL_POSE_SAFE_STRUCTURAL_EDGE_DENSITY_MIN
        ),
        "strict_guide_alignment_passed": bool(
            chamfer.get("available") is True
            and chamfer_coverage is not None
            and chamfer_coverage >= GROOMING_CANONICAL_DIRECT_CHAMFER_COVERAGE_MIN
            and chamfer_mean is not None
            and chamfer_mean <= GROOMING_CANONICAL_DIRECT_CHAMFER_MEAN_DISTANCE_MAX
            and chamfer_p90 is not None
            and chamfer_p90 <= GROOMING_CANONICAL_DIRECT_CHAMFER_P90_DISTANCE_MAX
        ),
        "identity_gate_failed_only_for_repaintable_palette_drift": bool(
            palette.get("available") is True
            and palette.get("acceptable") is not True
            and bool(palette_issues)
            and not unexpected_palette_issues
        ),
        "independent_fur_surface_passed": bool(
            surface.get("available") is True
            and surface.get("evaluated") is True
            and surface.get("acceptable") is True
            and surface_provenance.get("alpha_provenance")
            == GROOMING_CANONICAL_SEED_BOUND_MATTE_ALPHA_PROVENANCE
        ),
    }
    return {
        "policy": GROOMING_CANONICAL_POSE_SAFE_REFINE_SOURCE_POLICY,
        "attempt": attempt,
        "evaluated": True,
        "eligible": bool(all(checks.values())),
        "checks": checks,
        "allowed_identity_only_issues": sorted(
            GROOMING_CANONICAL_POSE_SAFE_IDENTITY_ONLY_ISSUES
        ),
        "observed_identity_issues": sorted(palette_issues),
        "unexpected_identity_issues": unexpected_palette_issues,
        "observed": {
            "structural_edge_density": structural_density,
            "chamfer_coverage": chamfer_coverage,
            "chamfer_mean_distance_px": chamfer_mean,
            "chamfer_p90_distance_px": chamfer_p90,
            "pose_transfer_seed_sha256": matte.get("pose_transfer_seed_sha256"),
            "union_alpha_sha256": matte.get("union_alpha_sha256"),
        },
        "thresholds": {
            "structural_edge_density_min": (
                GROOMING_CANONICAL_POSE_SAFE_STRUCTURAL_EDGE_DENSITY_MIN
            ),
            "chamfer_coverage_min": GROOMING_CANONICAL_DIRECT_CHAMFER_COVERAGE_MIN,
            "chamfer_mean_distance_max": (
                GROOMING_CANONICAL_DIRECT_CHAMFER_MEAN_DISTANCE_MAX
            ),
            "chamfer_p90_distance_max": (
                GROOMING_CANONICAL_DIRECT_CHAMFER_P90_DISTANCE_MAX
            ),
        },
        "identity_palette_gate_bypassed": False,
        "source_is_final_candidate": False,
        "attempt5_must_revalidate_all_gates": True,
        "fail_closed": True,
    }


def prepare_canonical_grooming_achromatic_palette_normalization(
    candidate: Path,
    canonical_reference: Path,
    target: Path,
) -> dict[str, Any]:
    """Match foreground LAB-L distribution while preserving action and chroma."""
    _sync_local_video_runtime(globals(), {"prepare_canonical_grooming_achromatic_palette_normalization"})

    if not candidate.is_file() or not canonical_reference.is_file():
        raise FileNotFoundError("Grooming palette-normalization input is missing.")
    with Image.open(candidate) as loaded:
        candidate_rgba = np.asarray(loaded.convert("RGBA"), dtype=np.uint8)
    with Image.open(canonical_reference) as loaded:
        canonical_rgba = np.asarray(loaded.convert("RGBA"), dtype=np.uint8)
        canonical_has_alpha = "A" in loaded.getbands()
    if candidate_rgba.shape != (512, 512, 4) or canonical_rgba.shape != (
        512,
        512,
        4,
    ):
        raise ValueError("Grooming palette-normalization inputs must be 512x512.")

    candidate_alpha = candidate_rgba[:, :, 3]
    candidate_foreground = candidate_alpha > 96
    if not 4096 <= int(np.count_nonzero(candidate_foreground)) <= int(
        512 * 512 * 0.90
    ):
        raise ValueError("Grooming candidate alpha geometry is invalid.")

    canonical_alpha = canonical_rgba[:, :, 3]
    canonical_foreground = canonical_alpha > 96
    if not canonical_has_alpha or np.all(canonical_alpha == 255):
        canonical_rgb = canonical_rgba[:, :, :3]
        red = canonical_rgb[:, :, 0].astype(np.int16)
        green = canonical_rgb[:, :, 1].astype(np.int16)
        blue = canonical_rgb[:, :, 2].astype(np.int16)
        canonical_background = (
            (green > 80)
            & ((green - np.maximum(red, blue)) > 35)
            & (green > red * 1.18)
            & (green > blue * 1.18)
        )
        canonical_foreground = ~canonical_background
    if not 4096 <= int(np.count_nonzero(canonical_foreground)) <= int(
        512 * 512 * 0.90
    ):
        raise ValueError("Canonical palette foreground is invalid.")

    candidate_lab = cv2.cvtColor(candidate_rgba[:, :, :3], cv2.COLOR_RGB2LAB)
    canonical_lab = cv2.cvtColor(canonical_rgba[:, :, :3], cv2.COLOR_RGB2LAB)
    source_luma = candidate_lab[:, :, 0][candidate_foreground]
    canonical_luma = canonical_lab[:, :, 0][canonical_foreground]
    source_histogram = np.bincount(source_luma, minlength=256).astype(np.float64)
    canonical_histogram = np.bincount(
        canonical_luma,
        minlength=256,
    ).astype(np.float64)
    source_cdf = np.cumsum(source_histogram) / float(source_histogram.sum())
    canonical_cdf = np.cumsum(canonical_histogram) / float(
        canonical_histogram.sum()
    )
    luma_lut = np.interp(
        source_cdf,
        canonical_cdf,
        np.arange(256, dtype=np.float64),
    )
    luma_lut = np.rint(luma_lut).clip(0, 255).astype(np.uint8)
    normalized_lab = candidate_lab.copy()
    normalized_lab[:, :, 0][candidate_foreground] = luma_lut[source_luma]
    normalized_rgb = cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2RGB)
    output_rgba = candidate_rgba.copy()
    output_rgba[:, :, :3] = normalized_rgb
    output_rgba[~candidate_foreground, :3] = 0

    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output_rgba, mode="RGBA").save(target)
    with Image.open(target) as loaded:
        verified = np.asarray(loaded.convert("RGBA"), dtype=np.uint8)
    if not np.array_equal(verified[:, :, 3], candidate_alpha):
        raise RuntimeError("Grooming palette normalization changed alpha geometry.")
    verified_lab = cv2.cvtColor(verified[:, :, :3], cv2.COLOR_RGB2LAB)
    ab_delta = np.abs(
        verified_lab[:, :, 1:].astype(np.int16)
        - normalized_lab[:, :, 1:].astype(np.int16)
    )[candidate_foreground]
    maximum_ab_roundtrip_delta = int(np.max(ab_delta)) if ab_delta.size else 0
    if maximum_ab_roundtrip_delta > 2:
        raise RuntimeError("Grooming palette normalization changed LAB chroma.")

    return {
        "policy": GROOMING_CANONICAL_DIRECT_ORGANIC_PAW_PALETTE_POLICY,
        "source_path": str(candidate),
        "source_sha256": file_sha256(candidate),
        "canonical_reference_path": str(canonical_reference),
        "canonical_reference_sha256": file_sha256(canonical_reference),
        "target_path": str(target),
        "target_sha256": file_sha256(target),
        "foreground_pixels": int(np.count_nonzero(candidate_foreground)),
        "source_alpha_geometry_preserved_exactly": True,
        "source_ab_chroma_preserved_with_roundtrip_tolerance": True,
        "maximum_ab_roundtrip_delta": maximum_ab_roundtrip_delta,
        "source_spatial_luma_rank_preserved": True,
        "canonical_spatial_pixels_copied": False,
        "canonical_luma_distribution_only": True,
        "luma_lut_monotonic": bool(np.all(np.diff(luma_lut.astype(int)) >= 0)),
        "source_luma_changed_pixels": int(
            np.count_nonzero(luma_lut[source_luma] != source_luma)
        ),
        "source_luma_p10_p50_p90": [
            round(float(value), 6)
            for value in np.percentile(source_luma, [10, 50, 90])
        ],
        "canonical_luma_p10_p50_p90": [
            round(float(value), 6)
            for value in np.percentile(canonical_luma, [10, 50, 90])
        ],
        "output_luma_p10_p50_p90": [
            round(float(value), 6)
            for value in np.percentile(
                verified_lab[:, :, 0][candidate_foreground],
                [10, 50, 90],
            )
        ],
    }


def bowl_aware_identity_palette_quality(
    primary_quality: dict[str, Any],
    upper_body_quality: dict[str, Any] | None,
) -> dict[str, Any]:
    """Prevent a touching bowl and kibble from being audited as the cat's coat."""
    _sync_local_video_runtime(globals(), {"bowl_aware_identity_palette_quality"})
    if not isinstance(upper_body_quality, dict):
        return primary_quality
    if upper_body_quality.get("available") and not upper_body_quality.get("acceptable"):
        merged_issues = list(
            dict.fromkeys(
                [
                    *(primary_quality.get("issues") or []),
                    *(upper_body_quality.get("issues") or []),
                ]
            )
        )
        merged_warnings = list(
            dict.fromkeys(
                [
                    *(primary_quality.get("warnings") or []),
                    *(upper_body_quality.get("warnings") or []),
                ]
            )
        )
        return {
            **upper_body_quality,
            "acceptable": False,
            "warning": True,
            "issues": merged_issues,
            "warnings": merged_warnings,
            "penalty": round(
                max(
                    float(primary_quality.get("penalty", 0.0)),
                    float(upper_body_quality.get("penalty", 0.0)),
                ),
                6,
            ),
            "method": "bowl_upper_body_confirmation",
            "primary_quality": primary_quality,
            "upper_body_quality": upper_body_quality,
        }
    if primary_quality.get("acceptable"):
        return primary_quality
    primary_issues = set(primary_quality.get("issues") or [])
    primary_spatial = primary_quality.get("marking_spatial_quality")
    upper_spatial = upper_body_quality.get("marking_spatial_quality")
    if (
        primary_issues == {"invented_stripe_pattern"}
        and upper_body_quality.get("available") is True
        and upper_body_quality.get("acceptable") is True
        and isinstance(primary_spatial, dict)
        and primary_spatial.get("available") is True
        and isinstance(upper_spatial, dict)
        and upper_spatial.get("available") is True
    ):
        upper_source_component_count = int(
            upper_spatial.get("source_stripe_component_count", 0) or 0
        )
        upper_source_area = float(
            upper_spatial.get("source_stripe_area_fraction", 0.0) or 0.0
        )
        upper_source_dark_contrast = float(
            upper_spatial.get("source_dark_contrast_fraction", 0.0) or 0.0
        )
        upper_source_evidence_present = bool(
            upper_source_component_count
            >= IDENTITY_SOURCE_PRESENT_STRIPE_COMPONENT_MIN
            and upper_source_area >= IDENTITY_SOURCE_PRESENT_STRIPE_AREA_MIN
        )
        cross_crop_component_delta = int(
            primary_spatial.get("candidate_stripe_component_count", 0) or 0
        ) - upper_source_component_count
        cross_crop_area_delta = float(
            primary_spatial.get("candidate_stripe_area_fraction", 0.0) or 0.0
        ) - upper_source_area
        cross_crop_dark_contrast_delta = float(
            primary_spatial.get("candidate_dark_contrast_fraction", 0.0) or 0.0
        ) - upper_source_dark_contrast
        cross_crop_severe_amplification = bool(
            upper_source_evidence_present
            and cross_crop_component_delta
            >= IDENTITY_AMPLIFIED_STRIPE_COMPONENT_DELTA_MIN
            and cross_crop_area_delta >= IDENTITY_AMPLIFIED_STRIPE_AREA_DELTA_MIN
            and cross_crop_dark_contrast_delta
            >= IDENTITY_AMPLIFIED_STRIPE_DARK_CONTRAST_DELTA_MIN
        )
        cross_crop_confirmation = {
            "policy": "bowl_upper_body_source_stripe_confirmation_v1",
            "upper_source_evidence_present": upper_source_evidence_present,
            "upper_source_stripe_component_count": upper_source_component_count,
            "upper_source_stripe_area_fraction": round(upper_source_area, 6),
            "upper_source_dark_contrast_fraction": round(
                upper_source_dark_contrast,
                6,
            ),
            "primary_candidate_stripe_component_count": int(
                primary_spatial.get("candidate_stripe_component_count", 0) or 0
            ),
            "primary_candidate_stripe_area_fraction": round(
                float(
                    primary_spatial.get("candidate_stripe_area_fraction", 0.0)
                    or 0.0
                ),
                6,
            ),
            "primary_candidate_dark_contrast_fraction": round(
                float(
                    primary_spatial.get(
                        "candidate_dark_contrast_fraction",
                        0.0,
                    )
                    or 0.0
                ),
                6,
            ),
            "cross_crop_component_delta": cross_crop_component_delta,
            "cross_crop_area_delta": round(cross_crop_area_delta, 6),
            "cross_crop_dark_contrast_delta": round(
                cross_crop_dark_contrast_delta,
                6,
            ),
            "severe_amplification": cross_crop_severe_amplification,
            "thresholds": {
                "stripe_component_delta_min": (
                    IDENTITY_AMPLIFIED_STRIPE_COMPONENT_DELTA_MIN
                ),
                "stripe_area_delta_min": (
                    IDENTITY_AMPLIFIED_STRIPE_AREA_DELTA_MIN
                ),
                "dark_contrast_delta_min": (
                    IDENTITY_AMPLIFIED_STRIPE_DARK_CONTRAST_DELTA_MIN
                ),
            },
        }
        if upper_source_evidence_present and not cross_crop_severe_amplification:
            warnings = list(upper_body_quality.get("warnings") or [])
            warnings.append(
                "primary_crop_missing_source_stripe_evidence_ignored"
            )
            return {
                **upper_body_quality,
                "acceptable": True,
                "warning": True,
                "issues": [],
                "warnings": list(dict.fromkeys(warnings)),
                "penalty": round(
                    float(upper_body_quality.get("penalty", 0.0)) + 25.0,
                    6,
                ),
                "method": "bowl_upper_body_source_stripe_confirmation_v1",
                "primary_quality": primary_quality,
                "upper_body_quality": upper_body_quality,
                "cross_crop_stripe_confirmation": cross_crop_confirmation,
            }
        if upper_source_evidence_present and cross_crop_severe_amplification:
            return {
                **primary_quality,
                "method": "bowl_upper_body_source_stripe_confirmation_v1",
                "primary_quality": primary_quality,
                "upper_body_quality": upper_body_quality,
                "cross_crop_stripe_confirmation": cross_crop_confirmation,
            }
    bowl_contamination_issues = {
        "achromatic_source_recolored_with_warm_fur",
        "new_dominant_warm_brown_or_orange",
        "new_dominant_yellow_or_gold",
        "new_dominant_red_or_pink",
    }
    if not primary_issues or not primary_issues.issubset(bowl_contamination_issues):
        return primary_quality
    if not upper_body_quality.get("available"):
        return primary_quality
    if not upper_body_quality.get("acceptable"):
        return {
            **primary_quality,
            "method": "bowl_upper_body_confirmation",
            "primary_quality": primary_quality,
            "upper_body_quality": upper_body_quality,
        }
    warnings = list(upper_body_quality.get("warnings") or [])
    warnings.append("bowl_prop_palette_contamination_ignored")
    return {
        **upper_body_quality,
        "acceptable": True,
        "warning": True,
        "issues": [],
        "warnings": list(dict.fromkeys(warnings)),
        "penalty": round(float(upper_body_quality.get("penalty", 0.0)) + 25.0, 6),
        "method": "bowl_upper_body_confirmation",
        "primary_quality": primary_quality,
        "upper_body_quality": upper_body_quality,
    }


def canonical_identity_palette_quality(
    primary_quality: dict[str, Any] | None,
    upper_body_quality: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Require both full canonical and upper-body palette audits to pass."""
    _sync_local_video_runtime(globals(), {"canonical_identity_palette_quality"})
    if not isinstance(primary_quality, dict):
        return upper_body_quality
    if not isinstance(upper_body_quality, dict):
        return primary_quality
    issues = list(
        dict.fromkeys(
            [
                *(primary_quality.get("issues") or []),
                *(upper_body_quality.get("issues") or []),
            ]
        )
    )
    warnings = list(
        dict.fromkeys(
            [
                *(primary_quality.get("warnings") or []),
                *(upper_body_quality.get("warnings") or []),
            ]
        )
    )
    available = bool(
        primary_quality.get("available") and upper_body_quality.get("available")
    )
    acceptable = bool(
        primary_quality.get("acceptable") and upper_body_quality.get("acceptable")
    )
    return {
        **primary_quality,
        "available": available,
        "acceptable": acceptable,
        "warning": bool(issues or warnings),
        "issues": issues,
        "warnings": warnings,
        "penalty": round(
            max(
                float(primary_quality.get("penalty", 0.0)),
                float(upper_body_quality.get("penalty", 0.0)),
            ),
            6,
        ),
        "method": "canonical_primary_and_upper_body_v1",
        "primary_quality": primary_quality,
        "upper_body_quality": upper_body_quality,
    }


def evaluate_pose_identity_palette(
    candidate_path: Path,
    *,
    primary_reference: dict[str, Any] | None,
    upper_body_reference: dict[str, Any] | None,
    bottom_exclusion: float,
    bowl_action: bool,
) -> dict[str, Any]:
    """Evaluate a pose subject while keeping bowl colors out of the coat audit."""
    _sync_local_video_runtime(globals(), {"evaluate_pose_identity_palette"})
    candidate = (
        identity_palette_signature(
            candidate_path,
            exclude_bottom_fraction=bottom_exclusion,
        )
        if primary_reference is not None
        else None
    )
    primary_quality = (
        identity_palette_quality(primary_reference, candidate)
        if primary_reference is not None and candidate is not None
        else None
    )
    upper_candidate = (
        identity_palette_signature(
            candidate_path,
            exclude_bottom_fraction=BOWL_IDENTITY_PALETTE_UPPER_BODY_EXCLUSION,
        )
        if upper_body_reference is not None
        else None
    )
    upper_quality = (
        identity_palette_quality(upper_body_reference, upper_candidate)
        if upper_body_reference is not None and upper_candidate is not None
        else None
    )
    quality = (
        bowl_aware_identity_palette_quality(primary_quality, upper_quality)
        if primary_quality is not None and bowl_action
        else primary_quality
    )
    return {
        "candidate": candidate,
        "primary_quality": primary_quality,
        "upper_body_candidate": upper_candidate,
        "upper_body_quality": upper_quality,
        "quality": quality,
    }


def identity_palette_audit(
    guard_enabled: bool,
    quality: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return an explicit audit verdict without treating unchecked output as safe."""
    _sync_local_video_runtime(globals(), {"identity_palette_audit"})
    if not guard_enabled:
        return {
            "identity_palette_checked": False,
            "identity_palette_verdict": "disabled",
            "identity_palette_preserved": None,
        }
    if not isinstance(quality, dict) or not quality.get("available"):
        return {
            "identity_palette_checked": True,
            "identity_palette_verdict": "unavailable",
            "identity_palette_preserved": None,
        }
    preserved = bool(quality.get("acceptable"))
    return {
        "identity_palette_checked": True,
        "identity_palette_verdict": "passed" if preserved else "failed",
        "identity_palette_preserved": preserved,
    }


def skipped_visual_audit(name: str) -> dict[str, Any]:
    """Describe a deliberately skipped content audit as a non-blocking result."""
    _sync_local_video_runtime(globals(), {"skipped_visual_audit"})
    return {
        "audit": name,
        "available": False,
        "evaluated": False,
        "acceptable": True,
        "status": "skipped_advisory",
        "verdict": "not_blocking",
        "blocking": False,
        "warning": False,
        "issues": [],
        "warnings": [],
        "penalty": 0.0,
        "generation_audit_policy": GENERATION_AUDIT_POLICY,
    }


def canonical_fallback_score(record: dict[str, Any]) -> float:
    """Rank canonical retries while making an identity failure prohibitively costly."""
    _sync_local_video_runtime(globals(), {"canonical_fallback_score"})
    palette = record.get("identity_palette")
    palette_penalty = (
        float(palette.get("penalty", 0.0)) if isinstance(palette, dict) else 0.0
    )
    palette_gate = (
        0.0
        if not isinstance(palette, dict) or palette.get("acceptable", True)
        else 10_000.0
    )
    quality = record.get("quality")
    quality_penalty = (
        float(quality.get("penalty", 100.0)) if isinstance(quality, dict) else 100.0
    )
    chroma = record.get("chroma")
    chroma_penalty = (
        0.0 if isinstance(chroma, dict) and chroma.get("acceptable") else 50.0
    )
    return palette_gate + palette_penalty * 10.0 + quality_penalty + chroma_penalty


def select_canonical_fallback_attempt(
    attempts: list[dict[str, Any]],
    *,
    identity_palette_required: bool,
) -> dict[str, Any]:
    """Select the best usable retry without blocking on advisory visual drift."""
    _sync_local_video_runtime(globals(), {"select_canonical_fallback_attempt"})
    if not attempts:
        raise RuntimeError("Canonical generation produced no fallback candidate.")
    identity_safe = [
        record
        for record in attempts
        if not isinstance(record.get("identity_palette"), dict)
        or record["identity_palette"].get("acceptable", True)
    ]
    if (
        GENERATION_VISUAL_AUDITS_BLOCKING
        and identity_palette_required
        and not identity_safe
    ):
        issues = list(
            dict.fromkeys(
                issue
                for record in attempts
                for issue in (
                    record.get("identity_palette", {}).get("issues", [])
                    if isinstance(record.get("identity_palette"), dict)
                    else []
                )
            )
        )
        raise RuntimeError(
            "Canonical identity palette drift remained after "
            f"{len(attempts)} attempts: "
            + (", ".join(issues) or "no identity-safe candidate")
        )
    candidates = (
        identity_safe or attempts
        if GENERATION_VISUAL_AUDITS_BLOCKING
        else attempts
    )
    return min(candidates, key=canonical_fallback_score)


def canonical_generated_attempts_require_uploaded_fallback(
    attempts: list[dict[str, Any]],
    *,
    identity_palette_required: bool,
) -> bool:
    """Return true only when every generated canonical failed identity checks."""

    _sync_local_video_runtime(
        globals(),
        {"canonical_generated_attempts_require_uploaded_fallback"},
    )
    if not identity_palette_required:
        return False
    return not any(
        isinstance(record.get("identity_palette"), dict)
        and record["identity_palette"].get("acceptable") is True
        for record in attempts
    )


def semantic_quality_blocking_issues(quality: dict[str, Any]) -> list[str]:
    """Return defects that cannot be repaired by final canvas normalization."""
    _sync_local_video_runtime(globals(), {"semantic_quality_blocking_issues"})
    return [
        str(issue)
        for issue in quality.get("issues", [])
        if issue not in SEMANTIC_RECOVERABLE_FRAMING_ISSUES
    ]


def semantic_recovery_gate_failure(
    chroma: dict[str, Any],
    quality: dict[str, Any],
    similarity: dict[str, Any],
    palette: dict[str, Any] | None,
    fur_silhouette: dict[str, Any] | None = None,
    fur_surface_gate: dict[str, Any] | None = None,
    *,
    identity_palette_blocking: bool = False,
) -> str | None:
    """Return a technical rejection, plus visual rejection only in strict mode."""
    _sync_local_video_runtime(globals(), {"semantic_recovery_gate_failure"})
    if not bool(chroma.get("acceptable")):
        return "semantic_chroma_invalid"
    if (
        identity_palette_blocking
        and palette is not None
        and not bool(palette.get("acceptable"))
    ):
        return "semantic_identity_palette_invalid"
    if not GENERATION_VISUAL_AUDITS_BLOCKING:
        return None
    if not bool(quality.get("acceptable")):
        return "semantic_subject_quality_invalid"
    if not bool(similarity.get("acceptable")):
        return "semantic_pose_similarity_invalid"
    if palette is not None and not bool(palette.get("acceptable")):
        return "semantic_identity_palette_invalid"
    if fur_silhouette is not None and not bool(fur_silhouette.get("acceptable")):
        return "semantic_fur_silhouette_invalid"
    if fur_surface_gate is not None and fur_surface_gate.get("acceptable") is not True:
        return "semantic_fur_surface_invalid"
    return None


def neutralize_achromatic_warm_cast(
    source_path: Path,
    candidate_path: Path,
    output_path: Path,
    *,
    green_target: Path | None = None,
) -> dict[str, Any]:
    """Correct a proven global Lab warm cast without touching local accents."""
    _sync_local_video_runtime(globals(), {"neutralize_achromatic_warm_cast"})

    output_resolved = output_path.resolve()
    if output_resolved in {source_path.resolve(), candidate_path.resolve()}:
        raise ValueError("Warm-cast recovery output must be a derived file")

    def foreground_mask(rgba: np.ndarray) -> np.ndarray:
        alpha = rgba[:, :, 3]
        if np.any(alpha < 250):
            mask = alpha >= 64
        else:
            rgb = rgba[:, :, :3]
            red = rgb[:, :, 0].astype(np.int16)
            green = rgb[:, :, 1].astype(np.int16)
            blue = rgb[:, :, 2].astype(np.int16)
            background = (
                (green > 80)
                & ((green - np.maximum(red, blue)) > 35)
                & (green > red * 1.18)
                & (green > blue * 1.18)
            )
            mask = ~background
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8),
            connectivity=8,
        )
        if count <= 1:
            return mask
        largest = max(
            range(1, count),
            key=lambda label: int(stats[label, cv2.CC_STAT_AREA]),
        )
        return labels == largest

    def analysis_assets(rgba: np.ndarray) -> dict[str, Any] | None:
        mask = foreground_mask(rgba)
        ys, xs = np.where(mask)
        if not len(ys):
            return None
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        height = max(1, y1 - y0)
        rows, cols = np.indices(mask.shape)
        upper = rows < y0 + int(round(height * 0.55))
        interior = cv2.erode(
            mask.astype(np.uint8),
            np.ones((5, 5), dtype=np.uint8),
            iterations=1,
        ).astype(bool)
        rgb = rgba[:, :, :3]
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        warm_hue = (hue <= 41) | (hue >= 165)
        red_or_pink = ((hue <= 7) | (hue >= 165)) & (saturation >= 20)
        # A warm lighting cast can put large areas of neutral gray fur into the
        # red/pink hue bin. Preserve only high-chroma red/pink pixels and small
        # localized components (nose/ear accents); broad low-chroma regions stay
        # eligible for global cast recovery.
        localized_red_or_pink = np.zeros(mask.shape, dtype=bool)
        redpink_accent_seed = mask & red_or_pink & (saturation >= 52)
        component_count, component_labels, component_stats, _ = (
            cv2.connectedComponentsWithStats(
                redpink_accent_seed.astype(np.uint8),
                connectivity=8,
            )
        )
        maximum_accent_area = max(
            96,
            int(round(np.count_nonzero(mask) * 0.012)),
        )
        for component_label in range(1, component_count):
            if (
                int(component_stats[component_label, cv2.CC_STAT_AREA])
                <= maximum_accent_area
            ):
                localized_red_or_pink |= component_labels == component_label
        red_or_pink_accent = red_or_pink & localized_red_or_pink
        blue_green_accent = (hue >= 35) & (hue <= 140) & (saturation >= 20)
        high_chroma = saturation >= 96
        protected_high_chroma = high_chroma & (~warm_hue | localized_red_or_pink)
        neutral = (
            interior
            & upper
            & (value >= 24)
            & (value <= 248)
            & ~red_or_pink_accent
            & ~blue_green_accent
            & (saturation < 80)
        )
        return {
            "mask": mask,
            "interior": interior,
            "upper": upper,
            "bbox": (x0, y0, x1, y1),
            "rgb": rgb,
            "hsv": hsv,
            "lab": lab,
            "neutral": neutral,
            "red_or_pink_accent": red_or_pink_accent,
            "blue_green_accent": blue_green_accent,
            "high_chroma": protected_high_chroma,
            "rows": rows,
            "cols": cols,
        }

    with Image.open(source_path) as loaded:
        source_rgba = np.asarray(loaded.convert("RGBA"), dtype=np.uint8)
    with Image.open(candidate_path) as loaded:
        candidate_rgba = np.asarray(loaded.convert("RGBA"), dtype=np.uint8).copy()
    alpha_sha256_before = hashlib.sha256(candidate_rgba[:, :, 3].tobytes()).hexdigest()
    source = analysis_assets(source_rgba)
    candidate = analysis_assets(candidate_rgba)
    if source is None or candidate is None:
        return {"applied": False, "reason": "no_foreground"}
    source_neutral = source["neutral"]
    candidate_neutral = candidate["neutral"]
    if (
        np.count_nonzero(source_neutral) < 256
        or np.count_nonzero(candidate_neutral) < 256
    ):
        return {"applied": False, "reason": "insufficient_neutral_fur_samples"}
    source_lab = source["lab"]
    candidate_lab = candidate["lab"]
    source_ab = np.median(source_lab[:, :, 1:3][source_neutral], axis=0)
    candidate_ab = np.median(candidate_lab[:, :, 1:3][candidate_neutral], axis=0)
    cast_delta = candidate_ab.astype(np.float32) - source_ab.astype(np.float32)
    cast_magnitude = float(np.linalg.norm(cast_delta))
    # Positive b* is yellow and positive a* is red. Require a meaningful warm
    # direction before changing any pixel; cool or localized recolors fail closed.
    warm_direction = bool(cast_delta[1] >= 3.0 or cast_delta[0] >= 3.0)
    if cast_magnitude < 4.0 or not warm_direction:
        return {
            "applied": False,
            "reason": "global_warm_cast_not_proven",
            "cast_delta_ab": [round(float(value), 4) for value in cast_delta],
            "cast_magnitude": round(cast_magnitude, 4),
        }
    direction = cast_delta / max(cast_magnitude, 1e-6)
    deviations = candidate_lab[:, :, 1:3].astype(np.float32) - source_ab.astype(
        np.float32
    )
    projection = deviations[:, :, 0] * direction[0] + deviations[:, :, 1] * direction[1]
    globally_shifted = candidate_neutral & (projection >= cast_magnitude * 0.45)
    upper_foreground_count = int(
        np.count_nonzero(candidate["mask"] & candidate["upper"])
    )
    shifted_count = int(np.count_nonzero(globally_shifted))
    neutral_coverage = float(shifted_count) / float(
        max(1, np.count_nonzero(candidate_neutral))
    )
    foreground_coverage = float(shifted_count) / float(max(1, upper_foreground_count))
    x0, y0, x1, y1 = candidate["bbox"]
    upper_y1 = y0 + max(1, int(round((y1 - y0) * 0.55)))
    grid_projections: list[float] = []
    for row in range(2):
        gy0 = y0 + (upper_y1 - y0) * row // 2
        gy1 = y0 + (upper_y1 - y0) * (row + 1) // 2
        for column in range(3):
            gx0 = x0 + (x1 - x0) * column // 3
            gx1 = x0 + (x1 - x0) * (column + 1) // 3
            grid = np.zeros(candidate_neutral.shape, dtype=bool)
            grid[gy0:gy1, gx0:gx1] = True
            samples = candidate_neutral & grid
            if np.count_nonzero(samples) < 128:
                continue
            median_ab = np.median(
                candidate_lab[:, :, 1:3][samples],
                axis=0,
            )
            grid_delta = median_ab.astype(np.float32) - source_ab.astype(np.float32)
            grid_projections.append(float(np.dot(grid_delta, direction)))
    consistent_grids = sum(value >= cast_magnitude * 0.45 for value in grid_projections)
    grid_iqr = (
        float(np.percentile(grid_projections, 75) - np.percentile(grid_projections, 25))
        if len(grid_projections) >= 4
        else float("inf")
    )
    if (
        neutral_coverage < 0.15
        or foreground_coverage < 0.06
        or consistent_grids < 4
        or grid_iqr > 12.0
    ):
        return {
            "applied": False,
            "reason": "warm_shift_is_not_spatially_global",
            "cast_delta_ab": [round(float(value), 4) for value in cast_delta],
            "cast_magnitude": round(cast_magnitude, 4),
            "neutral_coverage": round(neutral_coverage, 6),
            "foreground_coverage": round(foreground_coverage, 6),
            "consistent_grids": consistent_grids,
            "grid_count": len(grid_projections),
            "grid_iqr": round(grid_iqr, 4) if np.isfinite(grid_iqr) else None,
        }
    saturation = candidate["hsv"][:, :, 1].astype(np.float32)
    hue = candidate["hsv"][:, :, 0]
    rows = candidate["rows"]
    cols = candidate["cols"]
    height = max(1, y1 - y0)
    width = max(1, x1 - x0)
    bowl_roi = (
        # Keep the actual lower-central bowl untouched. Starting this region
        # only after the vertical correction has almost faded prevents a hard
        # horizontal colour seam across the muzzle or chest.
        (rows >= y0 + int(round(height * 0.82)))
        & (cols >= x0 + int(round(width * 0.25)))
        & (cols <= x0 + int(round(width * 0.75)))
    )
    warm_low_chroma = ((hue <= 41) | (hue >= 165)) & (saturation < 168)
    protected = (
        ~candidate["interior"]
        | candidate["red_or_pink_accent"]
        | candidate["blue_green_accent"]
        | candidate["high_chroma"]
        | bowl_roi
    )
    correction_mask = candidate["mask"] & warm_low_chroma & ~protected
    saturation_weight = np.clip((168.0 - saturation) / 72.0, 0.0, 1.0)
    relative_y = (rows.astype(np.float32) - float(y0)) / float(height)
    vertical_weight = np.clip((0.85 - relative_y) / 0.30, 0.0, 1.0)
    projection_strength = np.clip(
        projection / max(cast_magnitude, 1e-6),
        0.0,
        2.0,
    )
    weight = (
        correction_mask.astype(np.float32)
        * saturation_weight
        * vertical_weight
        * projection_strength
    )
    active = weight > 0.01
    weight[~active] = 0.0
    corrected_count = int(np.count_nonzero(active))
    if corrected_count < max(128, int(upper_foreground_count * 0.10)):
        return {
            "applied": False,
            "reason": "insufficient_safe_pixels_for_correction",
            "neutral_coverage": round(neutral_coverage, 6),
            "foreground_coverage": round(foreground_coverage, 6),
            "corrected_pixels": corrected_count,
        }
    protected_hash_before = hashlib.sha256(
        candidate["rgb"][protected & candidate["mask"]].tobytes()
    ).hexdigest()
    corrected_lab = candidate_lab.astype(np.float32).copy()
    bounded_delta = np.clip(cast_delta, -24.0, 24.0)
    corrected_lab[:, :, 1] -= bounded_delta[0] * weight
    corrected_lab[:, :, 2] -= bounded_delta[1] * weight
    corrected_rgb = cv2.cvtColor(
        np.clip(np.rint(corrected_lab), 0, 255).astype(np.uint8),
        cv2.COLOR_LAB2RGB,
    )
    # Lab conversion is not perfectly lossless. Keep every pixel outside the
    # weighted correction region byte-for-byte identical to the candidate so
    # the recovery cannot introduce unrelated colour or edge changes.
    corrected_rgb[~active] = candidate["rgb"][~active]
    corrected_rgb[protected] = candidate["rgb"][protected]
    candidate_rgba[:, :, :3] = corrected_rgb
    protected_hash_after = hashlib.sha256(
        candidate_rgba[:, :, :3][protected & candidate["mask"]].tobytes()
    ).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(candidate_rgba, mode="RGBA").save(
        output_path,
        format="PNG",
        optimize=True,
    )
    if green_target is not None:
        alpha_float = candidate_rgba[:, :, 3:4].astype(np.float32) / 255.0
        green = np.empty_like(candidate_rgba[:, :, :3])
        green[:, :, :] = (0, 255, 0)
        composite = candidate_rgba[:, :, :3].astype(
            np.float32
        ) * alpha_float + green.astype(np.float32) * (1.0 - alpha_float)
        Image.fromarray(
            np.clip(np.rint(composite), 0, 255).astype(np.uint8),
            mode="RGB",
        ).save(green_target, format="PNG", optimize=True)
    return {
        "applied": True,
        "method": "robust_lab_global_warm_cast_recovery",
        "source": source_path.name,
        "candidate": candidate_path.name,
        "output": output_path.name,
        "source_median_ab": [round(float(value), 4) for value in source_ab],
        "candidate_median_ab": [round(float(value), 4) for value in candidate_ab],
        "cast_delta_ab": [round(float(value), 4) for value in cast_delta],
        "applied_delta_ab": [round(float(value), 4) for value in bounded_delta],
        "cast_magnitude": round(cast_magnitude, 4),
        "neutral_coverage": round(neutral_coverage, 6),
        "foreground_coverage": round(foreground_coverage, 6),
        "consistent_grids": consistent_grids,
        "grid_count": len(grid_projections),
        "grid_iqr": round(grid_iqr, 4),
        "corrected_pixels": corrected_count,
        "adaptive_projection_strength_p95": round(
            float(np.percentile(projection_strength[active], 95)),
            4,
        ),
        "protected_hash_before": protected_hash_before,
        "protected_hash_after": protected_hash_after,
        "alpha_sha256_before": alpha_sha256_before,
        "alpha_sha256_after": hashlib.sha256(
            candidate_rgba[:, :, 3].tobytes()
        ).hexdigest(),
    }


def semantic_background_recovery(
    source: Path,
    chroma_target: Path,
    cutout_target: Path,
    semantic_target: Path,
    pose: str = "",
    *,
    validate_canonical_interior: bool = False,
) -> dict[str, Any]:
    _sync_local_video_runtime(globals(), {"semantic_background_recovery"})
    from rembg import remove

    with Image.open(source) as loaded:
        source_size = loaded.size
        image = loaded.convert("RGBA")
    alpha_matting = True
    try:
        recovered = remove(
            image,
            session=get_rembg_session(),
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=10,
        )
    except Exception:  # noqa: BLE001 - alpha matting may fail on difficult masks
        LOGGER.warning(
            "Alpha matting failed for %s; retrying semantic removal without matting",
            source.name,
            exc_info=True,
        )
        alpha_matting = False
        recovered = remove(image, session=get_rembg_session())
    if not isinstance(recovered, Image.Image):
        recovered = Image.open(io.BytesIO(recovered)).convert("RGBA")
    else:
        recovered = recovered.convert("RGBA")
    semantic_target.parent.mkdir(parents=True, exist_ok=True)
    recovered.save(semantic_target, format="PNG", optimize=True)
    alpha = np.asarray(recovered.getchannel("A"))
    semantic_foreground_fraction = float(np.count_nonzero(alpha >= 64)) / float(
        alpha.size
    )
    if semantic_foreground_fraction < 0.02:
        raise RuntimeError(
            "Semantic background recovery did not find a usable pet subject."
        )
    semantic_quality = (
        pose_cutout_quality(semantic_target, pose)
        if GENERATION_VISUAL_AUDITS_BLOCKING
        else skipped_visual_audit("semantic_background_subject_quality")
    )
    # The raw rembg canvas can legitimately be tight even when the complete pet
    # was recovered. prepare_landscape_keyframe below recenters and scales it, so
    # framing-only issues are evaluated again on the normalized final cutout.
    blocking_semantic_issues = (
        semantic_quality_blocking_issues(semantic_quality)
        if GENERATION_VISUAL_AUDITS_BLOCKING
        else []
    )
    if blocking_semantic_issues:
        raise RuntimeError(
            "Semantic background recovery produced an incomplete or invalid pet subject: "
            + ", ".join(blocking_semantic_issues)
        )
    prepare_landscape_keyframe(
        semantic_target,
        chroma_target,
        source_size,
        background_rgb=(0, 255, 0),
    )
    metrics = evaluate_chroma_cutout(
        chroma_target,
        cutout_target,
        validate_canonical_interior=validate_canonical_interior,
    )
    metrics.update(
        {
            "method": "rembg_u2net_fallback",
            "model": CHROMA_FALLBACK_REMBG_MODEL,
            "alpha_matting": alpha_matting,
            "semantic_foreground_fraction": semantic_foreground_fraction,
            "semantic_cutout": semantic_target.name,
            "semantic_quality": semantic_quality,
            "canonical_interior_validation": validate_canonical_interior,
        }
    )
    if not metrics["acceptable"]:
        raise RuntimeError(
            "Semantic background recovery could not create a clean green canvas: "
            + ", ".join(str(issue) for issue in metrics.get("issues", []))
        )
    return metrics
