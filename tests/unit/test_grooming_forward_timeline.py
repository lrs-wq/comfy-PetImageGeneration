from __future__ import annotations

import copy

import numpy as np

from src.comfy_pet_pipeline.media.grooming_temporal import (
    GROOMING_TEMPORAL_LOOP_POLICY,
    forward_cycle_indices,
    grooming_forward_order_quality,
    select_grooming_forward_timeline,
)


def cycle_plan():
    return {
        "policy": GROOMING_TEMPORAL_LOOP_POLICY, "source_frame_count": 81,
        "source_fps": 16, "animation_fps": 20, "animation_frame_count": 100,
        "selected_source_window": [60, 71], "repetitions": 6,
        "video_source_indices": forward_cycle_indices(60, 71, 6, 81),
        "animation_source_indices": forward_cycle_indices(60, 71, 6, 100),
    }


def test_old_palindrome_is_rejected_even_with_a_forward_label():
    plan = cycle_plan()
    plan["video_source_indices"] = list(range(40, 81)) + list(range(79, 39, -1))
    plan["playback"] = "forward_only"
    plan["closed_loop"] = True
    assert not grooming_forward_order_quality(plan)["acceptable"]


def test_both_exports_must_follow_the_same_forward_cycles():
    plan = cycle_plan()
    assert grooming_forward_order_quality(plan)["acceptable"]
    assert grooming_forward_order_quality(plan)["cycle_wrap_count"] == 5
    assert grooming_forward_order_quality(plan)["reverse_steps"] == 0
    plan["animation_source_indices"] = plan["animation_source_indices"][::-1]
    assert not grooming_forward_order_quality(plan)["acceptable"]


def test_corrupt_or_out_of_bounds_cycle_cannot_be_accepted():
    for patch in ({"selected_source_window": [60, 90]}, {"repetitions": 0},
                  {"source_fps": 0}, {"video_source_indices": [60] * 81},
                  {"policy": "grooming_second_half_ping_pong_closed_loop_v1"}):
        plan = cycle_plan() | patch
        assert not grooming_forward_order_quality(plan)["acceptable"]


def test_no_safe_cycle_preserves_entire_forward_motion_with_a_seam_warning():
    # Monotonic drift has no real return phase; a palindrome would hide this.
    alpha = np.full((32, 32), 255, dtype=np.uint8)
    frames = [(np.full((32, 32, 3), i * 3, dtype=np.uint8), alpha.copy()) for i in range(81)]
    plan = select_grooming_forward_timeline(frames, source_fps=16, animation_fps=20, animation_frames=100)
    assert plan["selected_source_window"] is None
    assert plan["video_source_indices"] == list(range(81))
    assert all(b >= a for a, b in zip(plan["animation_source_indices"], plan["animation_source_indices"][1:]))
    assert not plan["closed_loop"]
    assert plan["warnings"] == ["no_safe_forward_cycle_preserved_source"]
    assert grooming_forward_order_quality(plan)["acceptable"]


def test_natural_moving_cycle_is_repeated_forward_without_touching_identity_pixels():
    frames = []
    yy, xx = np.mgrid[:64, :64]
    for i in range(81):
        # The first frames are a unique entering motion; subsequent frames cycle.
        phase = 2 * np.pi * (i - 16) / 12
        x = (8 + i) if i < 16 else (32 + 8 * np.cos(phase))
        y = 17 if i < 16 else (17 + 8 * np.sin(phase))
        blob = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 32)
        rgb = np.broadcast_to((190 - 170 * blob)[..., None], (64, 64, 3)).astype(np.uint8).copy()
        frames.append((rgb, np.full((64, 64), 255, dtype=np.uint8)))
    before = [(rgb.tobytes(), alpha.tobytes()) for rgb, alpha in frames]
    plan = select_grooming_forward_timeline(frames, source_fps=16, animation_fps=20, animation_frames=100)
    assert plan["selection"] == "natural_forward_cycle"
    assert plan["selected_source_window"][0] >= 16
    assert plan["closed_loop"]
    assert plan["boundary_direction_cosine"] >= 0.2
    assert 0.85 <= plan["speed_ratio"] <= 1.15
    assert len(plan["video_source_indices"]) == 81
    assert len(plan["animation_source_indices"]) == 100
    assert grooming_forward_order_quality(plan)["acceptable"]
    assert before == [(rgb.tobytes(), alpha.tobytes()) for rgb, alpha in frames]


def test_static_input_does_not_create_a_fake_short_moving_cycle():
    frames = [(np.full((24, 24, 3), 90, np.uint8), np.full((24, 24), 255, np.uint8)) for _ in range(81)]
    plan = select_grooming_forward_timeline(frames, source_fps=16, animation_fps=20, animation_frames=100)
    assert plan["selected_source_window"] is None
    assert plan["video_source_indices"] == list(range(81))


def test_undeclared_cycle_reset_is_rejected():
    plan = copy.deepcopy(cycle_plan())
    plan["selected_source_window"] = None
    assert not grooming_forward_order_quality(plan)["acceptable"]
