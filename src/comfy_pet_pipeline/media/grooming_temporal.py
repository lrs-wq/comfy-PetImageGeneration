"""Select intact, forward grooming cycles without reversing or blending pixels."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

GROOMING_TEMPORAL_LOOP_POLICY = "grooming_forward_natural_cycle_v2"
GROOMING_FORWARD_CYCLE_LIMITS = {
    "minimum_seconds": 0.5,
    "maximum_seconds": 3.0,
    "speed_ratio_min": 0.85,
    "speed_ratio_max": 1.15,
    "seam_to_adjacent_ratio_max": 1.5,
    "boundary_direction_cosine_min": 0.2,
    "head_motion_excursion_min": 1.5,
}


def forward_cycle_indices(start: int, end: int, repeats: int, count: int) -> list[int]:
    """Sample whole forward cycles; only an explicit cycle boundary may wrap."""
    if start < 0 or end <= start or repeats < 1 or count < 2:
        raise ValueError("Invalid forward cycle window, repetitions or output count")
    length = end - start + 1
    return [start + (i * repeats * length // count) % length for i in range(count)]


def grooming_forward_order_quality(plan: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct both export timelines instead of trusting a 'forward' label."""
    try:
        source_count = int(plan["source_frame_count"])
        count = int(plan["animation_frame_count"])
        source_fps = float(plan["source_fps"])
        animation_fps = float(plan["animation_fps"])
        if source_count < 3 or count < 2 or source_fps <= 0 or animation_fps <= 0:
            raise ValueError("Invalid timeline dimensions")
        window = plan.get("selected_source_window")
        if window is None:
            video = list(range(source_count))
            animation = [min(source_count - 1, round(i * source_fps / animation_fps))
                         for i in range(count)]
        else:
            start, end = window
            repeats = plan["repetitions"]
            if not (0 <= start < end < source_count and 1 <= repeats <= count):
                raise ValueError("Cycle escapes the original source")
            video = forward_cycle_indices(start, end, repeats, source_count)
            animation = forward_cycle_indices(start, end, repeats, count)
        acceptable = (
            plan.get("policy") == GROOMING_TEMPORAL_LOOP_POLICY
            and plan.get("video_source_indices") == video
            and plan.get("animation_source_indices") == animation
        )
    except (KeyError, ValueError, TypeError, OverflowError):
        acceptable = False
        animation = []
    return {
        "acceptable": acceptable,
        "forward_only": acceptable,
        "issues": [] if acceptable else ["grooming_source_order_invalid"],
        "cycle_wrap_count": sum(b < a for a, b in zip(animation, animation[1:])),
        "reverse_steps": 0 if acceptable else None,
    }


def _features(frames: list[tuple[np.ndarray, np.ndarray]]) -> tuple[list[np.ndarray], list[int]]:
    mask = np.logical_or.reduce([alpha >= 128 for _, alpha in frames])
    ys, xs = np.where(mask)
    if not len(xs):
        raise ValueError("Grooming timeline has no visible subject")
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0 = int(ys.min())
    y1 = min(mask.shape[0], y0 + max(1, int((ys.max() + 1 - y0) * 0.65)))
    whole, head = [], []
    for rgb, alpha in frames:
        # Premultiply only the diagnostic features; actual output pixels are untouched.
        rgba = np.dstack((rgb.astype(np.float32) * (alpha[..., None] / 255.0), alpha))
        image = Image.fromarray(np.clip(rgba, 0, 255).astype(np.uint8), "RGBA")
        whole.append(np.asarray(image.resize((64, 64), Image.Resampling.BILINEAR), dtype=np.float32))
        head.append(np.asarray(image.crop((x0, y0, x1, y1)).resize(
            (64, 64), Image.Resampling.BILINEAR), dtype=np.float32))
    return [np.stack(whole), np.stack(head)], [x0, y0, x1, y1]


def _distances(features: np.ndarray) -> np.ndarray:
    return np.stack([np.abs(features - frame).mean(axis=(1, 2, 3)) for frame in features])


def _seams(distances: np.ndarray, indices: list[int]) -> dict[str, Any]:
    a = np.asarray(indices)
    b = np.roll(a, -1)
    scores = distances[a, b]
    # At 20 FPS the source is upsampled: duplicate frames are not evidence of stillness.
    adjacent = scores[(b > a)]
    median = float(np.median(adjacent)) if adjacent.size else 0.0
    boundary = scores[(b < a)]
    maximum = float(boundary.max()) if boundary.size else float(scores[-1])
    ratio = maximum / max(median, 1e-6)
    return {"acceptable": maximum <= max(0.5, median * 1.5),
            "maximum_seam_score": round(maximum, 6),
            "adjacent_transition_median": round(median, 6),
            "maximum_seam_ratio": round(ratio, 6)}


def select_grooming_forward_timeline(
    frames: list[tuple[np.ndarray, np.ndarray]], *, source_fps: float,
    animation_fps: float, animation_frames: int,
) -> dict[str, Any]:
    """Prefer the full source, then a moving natural cycle, otherwise retain forward motion.

    A seam warning can never authorize a palindrome, a static substitute, an RGB
    cross-fade, or an exception to the output's chronological integrity check.
    """
    count = len(frames)
    plan: dict[str, Any] = {
        "policy": GROOMING_TEMPORAL_LOOP_POLICY, "source_frame_count": count,
        "source_fps": source_fps, "animation_fps": animation_fps,
        "animation_frame_count": animation_frames, "selected_source_window": None,
        "repetitions": 1, "speed_ratio": 1.0,
        "video_source_indices": list(range(count)),
        "animation_source_indices": [min(count - 1, round(i * source_fps / animation_fps))
                                     for i in range(animation_frames)],
        "selection": "full_source_forward", "thresholds": dict(GROOMING_FORWARD_CYCLE_LIMITS),
    }
    if not grooming_forward_order_quality(plan)["acceptable"]:
        raise ValueError("Invalid grooming source or output timeline")
    features, roi = _features(frames)
    distances = [_distances(x) for x in features]

    def quality(video: list[int], animation: list[int]) -> dict[str, Any]:
        return {f"{region}_{format_name}": _seams(d, indices)
                for region, d in zip(("whole", "head_paw"), distances)
                for format_name, indices in (("video", video), ("animation", animation))}

    plan["head_paw_roi"] = roi
    plan["seam_quality"] = quality(plan["video_source_indices"], plan["animation_source_indices"])
    full_safe = all(x["acceptable"] for x in plan["seam_quality"].values())
    best: tuple[float, dict[str, Any]] | None = None
    candidate_count = 0
    if not full_safe:
        minimum = max(3, int(np.ceil(source_fps * 0.5)))
        maximum = min(count, int(source_fps * 3.0))
        target_phases = animation_frames / animation_fps * source_fps
        delta = np.diff(features[1], axis=0)
        for start in range(count - minimum + 1):
            for end in range(start + minimum - 1, min(count, start + maximum)):
                length = end - start + 1
                v0, v1 = delta[start].ravel(), delta[end - 1].ravel()
                cosine = float(np.dot(v0, v1) / max(np.linalg.norm(v0) * np.linalg.norm(v1), 1e-6))
                if cosine < 0.2:
                    continue
                adjacent = distances[1][np.arange(start, end), np.arange(start + 1, end + 1)]
                excursion = float(distances[1][start, start:end + 1].max())
                if excursion < max(1.5, 2 * float(np.median(adjacent))):
                    continue
                for repeats in range(max(1, int(np.ceil(target_phases * 0.85 / length))),
                                     int(np.floor(target_phases * 1.15 / length)) + 1):
                    video = forward_cycle_indices(start, end, repeats, count)
                    animation = forward_cycle_indices(start, end, repeats, animation_frames)
                    seam = quality(video, animation)
                    candidate_count += 1
                    if not all(x["acceptable"] for x in seam.values()):
                        continue
                    speed = repeats * length / target_phases
                    score = max(x["maximum_seam_ratio"] for x in seam.values()) + 0.2 * abs(1 - speed)
                    candidate = dict(selected_source_window=[start, end], repetitions=repeats,
                                     speed_ratio=round(speed, 6), video_source_indices=video,
                                     animation_source_indices=animation, seam_quality=seam,
                                     boundary_direction_cosine=round(cosine, 6),
                                     head_motion_excursion=round(excursion, 6),
                                     selection="natural_forward_cycle")
                    if best is None or score < best[0]:
                        best = score, candidate
    if best is not None:
        plan.update(best[1])
    plan["candidate_count"] = candidate_count
    plan["closed_loop"] = full_safe or best is not None
    plan["warnings"] = [] if plan["closed_loop"] else ["no_safe_forward_cycle_preserved_source"]
    plan["temporal_order_quality"] = grooming_forward_order_quality(plan)
    return plan
