from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

import local_video_api as pipeline


WORKSPACE_ROOT = Path(__file__).resolve().parent


def _lora(name: str, strength: float) -> dict[str, object]:
    return {
        "model_name": f"user_uploads/{name}.safetensors",
        "original_filename": f"{name}.safetensors",
        "strength": strength,
    }


def _save_transparent_subject(
    path: Path,
    *,
    offset: tuple[int, int] = (0, 0),
    lying: bool = False,
) -> None:
    image = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    dx, dy = offset
    fill = (176, 116, 72, 255)
    if lying:
        draw.rounded_rectangle(
            (12 + dx, 55 + dy, 104 + dx, 88 + dy),
            radius=14,
            fill=fill,
        )
        draw.ellipse((88 + dx, 42 + dy, 120 + dx, 74 + dy), fill=fill)
        draw.rectangle((25 + dx, 82 + dy, 42 + dx, 99 + dy), fill=fill)
        draw.rectangle((75 + dx, 82 + dy, 92 + dx, 99 + dy), fill=fill)
    else:
        draw.ellipse((42 + dx, 14 + dy, 86 + dx, 56 + dy), fill=fill)
        draw.rounded_rectangle(
            (36 + dx, 46 + dy, 92 + dx, 99 + dy),
            radius=18,
            fill=fill,
        )
        draw.rectangle((42 + dx, 91 + dy, 55 + dx, 112 + dy), fill=fill)
        draw.rectangle((73 + dx, 91 + dy, 86 + dx, 112 + dy), fill=fill)
    image.save(path, format="PNG")


def _save_eating_fur_silhouette(
    path: Path,
    *,
    offset: tuple[int, int] = (0, 0),
    jagged: bool = False,
) -> None:
    image = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    dx, dy = offset
    fill = (96, 96, 96, 255)
    draw.ellipse((18 + dx, 72 + dy, 210 + dx, 216 + dy), fill=fill)
    draw.ellipse((154 + dx, 82 + dy, 232 + dx, 160 + dy), fill=fill)
    draw.rectangle((164 + dx, 142 + dy, 188 + dx, 222 + dy), fill=fill)
    draw.rectangle((198 + dx, 142 + dy, 222 + dx, 222 + dy), fill=fill)
    if jagged:
        for x in range(48, 146, 10):
            spike = 17 if (x // 10) % 2 else 11
            draw.polygon(
                [
                    (x - 5 + dx, 82 + dy),
                    (x + dx, 82 - spike + dy),
                    (x + 5 + dx, 82 + dy),
                ],
                fill=fill,
            )
    image.save(path, format="PNG")


def _save_grounded_subject(path: Path, grounded_legs: int) -> None:
    image = Image.new("RGBA", (160, 160), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    fill = (176, 116, 72, 255)
    draw.rounded_rectangle((30, 30, 130, 95), radius=20, fill=fill)
    leg_positions = (38, 65, 92, 116)
    for left in leg_positions[:grounded_legs]:
        draw.rectangle((left, 82, left + 12, 135), fill=fill)
    image.save(path, format="PNG")


def _save_low_connected_eating_subject(path: Path) -> None:
    image = Image.new("RGBA", (160, 160), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    fill = (176, 116, 72, 255)
    draw.ellipse((58, 30, 106, 72), fill=fill)
    draw.rounded_rectangle((14, 58, 146, 108), radius=20, fill=fill)
    # The bowl deliberately touches the chest/body silhouette. Real eating
    # references commonly form one connected alpha component at the mouth or paws.
    draw.ellipse((36, 98, 124, 128), fill=(222, 190, 146, 255))
    image.save(path, format="PNG")


def _save_eating_subject_with_floor_shelf(path: Path) -> None:
    image = Image.new("RGBA", (160, 160), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    coat = (176, 116, 72, 255)
    draw.ellipse((58, 24, 102, 66), fill=coat)
    draw.rounded_rectangle((43, 52, 117, 110), radius=18, fill=coat)
    draw.ellipse((55, 92, 105, 122), fill=(222, 190, 146, 255))
    # A retained carpet strip joins the lower silhouette in one row and then
    # persists as a nearly full-width slab. It is still one alpha component.
    draw.rectangle((8, 110, 152, 119), fill=(190, 184, 176, 255))
    image.save(path, format="PNG")


def _save_mouth_bowl_contact_subject(
    path: Path,
    *,
    muzzle_at_bowl: bool = True,
    include_bowl: bool = True,
) -> None:
    """Create a low/wide cat whose bowl, paws and warm muzzle are separable."""
    image = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    coat = (124, 124, 124, 255)
    # The neutral coat prevents the warm-bowl splitter from mistaking the torso
    # for food. Broad planted paws on both sides keep the scene-prop split valid.
    draw.rounded_rectangle((20, 60, 230, 190), radius=45, fill=coat)
    draw.ellipse((86, 80, 174, 170), fill=coat)
    draw.rounded_rectangle((42, 150, 92, 220), radius=16, fill=coat)
    draw.rounded_rectangle((178, 150, 228, 220), radius=16, fill=coat)
    muzzle_top = 150 if muzzle_at_bowl else 120
    draw.ellipse(
        (118, muzzle_top, 146, muzzle_top + 16),
        fill=(196, 130, 120, 255),
    )
    if include_bowl:
        draw.ellipse((94, 170, 176, 220), fill=(222, 150, 70, 255))
    image.save(path, format="PNG")


def _save_drinking_subject_with_separate_bowl(path: Path, *, gap: int) -> None:
    image = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    coat = (124, 124, 124, 255)
    draw.rounded_rectangle((20, 55, 220, 168), radius=42, fill=coat)
    draw.ellipse((92, 80, 164, 164), fill=coat)
    bowl_top = 169 + gap
    draw.rounded_rectangle(
        (96, bowl_top, 170, bowl_top + 28),
        radius=8,
        fill=(190, 190, 190, 255),
    )
    image.save(path, format="PNG")


class UserLoraCapTests(unittest.TestCase):
    def test_cap_preserves_ratios_and_obeys_total_cap(self) -> None:
        specs = [_lora("a", 0.8), _lora("b", 0.8)]

        capped, scale = pipeline.cap_user_lora_specs(
            specs,
            total_cap=1.0,
            per_lora_cap=1.0,
        )

        self.assertAlmostEqual(scale, 0.625)
        self.assertEqual([item["strength"] for item in capped], [0.5, 0.5])
        self.assertLessEqual(sum(float(item["strength"]) for item in capped), 1.0)
        self.assertEqual(specs[0]["strength"], 0.8, "input specs must not be mutated")

    def test_cap_preserves_ratios_when_per_lora_cap_is_tighter(self) -> None:
        specs = [_lora("strong", 1.5), _lora("weak", 0.5)]

        capped, scale = pipeline.cap_user_lora_specs(
            specs,
            total_cap=10.0,
            per_lora_cap=0.6,
        )

        self.assertAlmostEqual(scale, 0.4)
        self.assertAlmostEqual(float(capped[0]["strength"]), 0.6)
        self.assertAlmostEqual(float(capped[1]["strength"]), 0.2)
        self.assertAlmostEqual(
            float(capped[0]["strength"]) / float(capped[1]["strength"]),
            3.0,
        )


class WanRuntimeConfigTests(unittest.TestCase):
    def test_a14b_uses_121_frames_at_24_fps(self) -> None:
        self.assertEqual(pipeline.FRAME_COUNT, 121)
        self.assertEqual(pipeline.FPS, 24.0)

        with patch.multiple(
            pipeline,
            WAN_ENGINE="wan2.2-i2v-a14b-fp8",
            FRAME_COUNT=121,
            FPS=24.0,
        ):
            workflow = pipeline.wan_workflow(
                "input.png",
                "animate the same cat",
                "identity drift",
                20260720,
                512,
                512,
                "tests/wan121",
            )

        self.assertEqual(workflow["8"]["inputs"]["length"], 121)
        self.assertEqual(workflow["14"]["inputs"]["fps"], 24.0)


class PoseProfileTests(unittest.TestCase):
    def test_canonical_profiles_retry_exact_lora_pair_before_reducing(self) -> None:
        with patch.multiple(
            pipeline,
            QWEN_CHROMA_MAX_ATTEMPTS=3,
            QWEN_RECOVERY_USER_LORA_STRENGTH=0.25,
        ):
            profiles = pipeline.qwen_recovery_profiles(
                user_lora=None,
                user_lora_strength=1.0,
                user_lora_mode="replace",
                steps=6,
                cfg=1.0,
                denoise=1.0,
                user_loras=[_lora("style-a", 0.5), _lora("style-b", 0.6)],
            )

        self.assertEqual(
            [profile["name"] for profile in profiles],
            ["requested", "requested_seed_retry", "safe_reduced_user_lora"],
        )
        self.assertEqual(
            [item["strength"] for item in profiles[0]["user_loras"]],
            [0.5, 0.6],
        )
        self.assertEqual(
            [item["strength"] for item in profiles[1]["user_loras"]],
            [0.5, 0.6],
        )
        self.assertEqual(
            [item["model_name"] for item in profiles[2]["user_loras"]],
            [item["model_name"] for item in profiles[0]["user_loras"]],
        )
        self.assertEqual(
            [item["strength"] for item in profiles[2]["user_loras"]],
            [0.113636, 0.136364],
        )

    def test_locked_canonical_profile_keeps_exact_lora_pair_for_every_retry(
        self,
    ) -> None:
        profiles = pipeline.qwen_recovery_profiles(
            user_lora=None,
            user_lora_strength=1.0,
            user_lora_mode="replace",
            steps=6,
            cfg=1.0,
            denoise=1.0,
            user_loras=[_lora("style-a", 0.5), _lora("style-b", 0.6)],
            lock_strengths=True,
        )

        self.assertEqual(
            [profile["name"] for profile in profiles],
            ["requested", "requested_seed_retry", "requested_final_seed_retry"],
        )
        self.assertEqual(
            [
                [item["strength"] for item in profile["user_loras"]]
                for profile in profiles
            ],
            [[0.5, 0.6], [0.5, 0.6], [0.5, 0.6]],
        )
        self.assertEqual(
            [profile["user_lora_mode"] for profile in profiles],
            ["replace", "replace", "replace"],
        )

    def test_three_pose_profiles_reduce_lora_and_isolate_user_prompt(self) -> None:
        specs = [_lora("a", 0.8), _lora("b", 0.8)]
        with patch.multiple(
            pipeline,
            QWEN_POSE_MAX_ATTEMPTS=3,
            QWEN_POSE_USER_LORA_TOTAL_CAP=0.40,
            QWEN_POSE_USER_LORA_RETRY_TOTAL_CAP=0.25,
            QWEN_POSE_USER_LORA_FINAL_TOTAL_CAP=0.15,
            QWEN_POSE_USER_LORA_PER_LORA_CAP=0.35,
        ):
            profiles = pipeline.qwen_pose_profiles(
                user_lora=None,
                user_lora_strength=1.0,
                user_lora_mode="replace",
                steps=4,
                cfg=1.0,
                denoise=0.8,
                user_loras=specs,
            )

        self.assertEqual(
            [profile["name"] for profile in profiles],
            ["pose_capped", "pose_reduced_retry", "pose_minimal_retry"],
        )
        self.assertEqual(
            [
                [item["strength"] for item in profile["user_loras"]]
                for profile in profiles
            ],
            [[0.2, 0.2], [0.125, 0.125], [0.075, 0.075]],
        )
        for actual, expected in zip(
            [profile["user_lora_strength_scale"] for profile in profiles],
            [0.25, 0.15625, 0.09375],
            strict=True,
        ):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(
            [profile["total_strength_cap"] for profile in profiles],
            [0.40, 0.25, 0.15],
        )
        self.assertEqual(
            [profile["user_lora_mode"] for profile in profiles],
            ["replace", "stack", "stack"],
        )
        self.assertEqual([profile["quality_retry"] for profile in profiles], [1, 2, 3])
        self.assertEqual(
            [profile["use_user_prompt"] for profile in profiles], [False] * 3
        )
        self.assertEqual([profile["steps"] for profile in profiles], [4, 6, 8])
        self.assertEqual([profile["denoise"] for profile in profiles], [0.8, 1.0, 1.0])

    def test_locked_pose_profiles_keep_exact_lora_weights_for_all_retries(self) -> None:
        profiles = pipeline.qwen_pose_profiles(
            user_lora=None,
            user_lora_strength=1.0,
            user_lora_mode="replace",
            steps=6,
            cfg=1.0,
            denoise=1.0,
            user_loras=[_lora("style-a", 0.5), _lora("style-b", 0.6)],
            first_total_cap=1.1,
            first_per_lora_cap=0.6,
            lock_strengths=True,
        )

        self.assertEqual(
            [profile["name"] for profile in profiles],
            ["pose_capped", "pose_exact_seed_retry", "pose_exact_final_retry"],
        )
        self.assertEqual(
            [
                [item["strength"] for item in profile["user_loras"]]
                for profile in profiles
            ],
            [[0.5, 0.6], [0.5, 0.6], [0.5, 0.6]],
        )

    def test_running_and_bowl_actions_get_one_specialized_retry(self) -> None:
        profiles = pipeline.qwen_pose_profiles(
            user_lora=None,
            user_lora_strength=1.0,
            user_lora_mode="replace",
            steps=6,
            cfg=1.0,
            denoise=1.0,
            user_loras=[_lora("style-a", 0.5), _lora("style-b", 0.6)],
            first_total_cap=1.1,
            first_per_lora_cap=0.6,
        )

        with (
            patch.object(pipeline, "QWEN_RUNNING_POSE_EXTRA_ATTEMPTS", 1),
            patch.object(pipeline, "QWEN_BOWL_POSE_EXTRA_ATTEMPTS", 1),
        ):
            running = pipeline.pose_generation_profiles(
                "running | first frame: ready", profiles
            )
            eating = pipeline.pose_generation_profiles(
                "eating | first frame: bowl", profiles
            )

        self.assertEqual(len(running), 4)
        self.assertEqual(len(eating), 4)
        self.assertEqual(running[-1]["name"], "running_ground_contact_seed_retry_1")
        self.assertEqual(running[-1]["quality_retry"], 4)
        self.assertEqual(running[-1]["steps"], 10)
        self.assertEqual(eating[-1]["name"], "bowl_palette_canvas_seed_retry_1")
        self.assertEqual(eating[-1]["quality_retry"], 4)
        self.assertEqual(eating[-1]["steps"], 10)
        self.assertEqual(
            [item["model_name"] for item in running[-1]["user_loras"]],
            [item["model_name"] for item in profiles[-1]["user_loras"]],
        )
        self.assertEqual(
            [item["model_name"] for item in eating[-1]["user_loras"]],
            [item["model_name"] for item in profiles[-1]["user_loras"]],
        )
        self.assertEqual(
            [item["strength"] for item in eating[-1]["user_loras"]],
            [item["strength"] for item in profiles[-1]["user_loras"]],
        )

    def test_specialized_retry_reaches_level_four_with_short_base_profile(self) -> None:
        profiles = pipeline.qwen_pose_profiles(
            user_lora=None,
            user_lora_strength=1.0,
            user_lora_mode="replace",
            steps=6,
            cfg=1.0,
            denoise=1.0,
            user_loras=[_lora("style-a", 0.5), _lora("style-b", 0.6)],
            first_total_cap=1.1,
            first_per_lora_cap=0.6,
        )

        with patch.object(pipeline, "QWEN_BOWL_POSE_EXTRA_ATTEMPTS", 1):
            eating = pipeline.pose_generation_profiles(
                "eating | first frame: bowl",
                profiles[:1],
            )

        self.assertEqual(len(eating), 2)
        self.assertEqual(eating[-1]["quality_retry"], 4)


class PosePromptTests(unittest.TestCase):
    def test_pose_prompt_excludes_user_fur_override_and_preserves_source_coat(
        self,
    ) -> None:
        pose = "eating | first frame: natural crouch beside one food bowl"

        prompt = pipeline.build_qwen_prompt(
            "preserve very fluffy long fur",
            pose,
            transparent_gif=True,
        )

        self.assertNotIn("preserve very fluffy long fur", prompt)
        self.assertIn("Coat identity rule:", prompt)
        self.assertIn("Never amplify its apparent length or volume", prompt)
        self.assertIn(
            "never shorten, flatten, or suppress genuine long fur",
            prompt,
        )
        self.assertIn("Remove only generated artifacts", prompt)
        self.assertIn("spiky fur", pipeline.CARTOON_NEGATIVE)
        self.assertIn("jagged generated fur silhouette", pipeline.CARTOON_NEGATIVE)

    def test_transparent_prompt_excludes_conflicting_base_prompt(self) -> None:
        conflict = "SENTINEL_CONFLICT front-facing sitting, tight crop, white studio background"
        pose = "eating | first frame: natural crouch at one food bowl; video: subtle eating"

        prompt = pipeline.build_qwen_prompt(
            conflict,
            pose,
            transparent_gif=True,
        )

        self.assertNotIn(conflict, prompt)
        self.assertNotIn("SENTINEL_CONFLICT", prompt)
        self.assertIn("Required still-frame pose only", prompt)
        self.assertIn("food bowl", prompt)

    def test_retry_prompts_have_progressive_pose_correction_instructions(self) -> None:
        pose = (
            "lying down | first frame: relaxed full-body prone pose; video: breathing"
        )

        first = pipeline.build_qwen_prompt(
            "", pose, transparent_gif=True, quality_retry=1
        )
        second = pipeline.build_qwen_prompt(
            "", pose, transparent_gif=True, quality_retry=2
        )
        third = pipeline.build_qwen_prompt(
            "", pose, transparent_gif=True, quality_retry=3
        )

        self.assertNotIn("VALIDATION RETRY", first)
        self.assertIn("VALIDATION RETRY", second)
        self.assertNotIn("FINAL POSE-CORRECTION RETRY", second)
        self.assertIn("VALIDATION RETRY", third)
        self.assertIn("FINAL POSE-CORRECTION RETRY", third)

    def test_running_final_retry_prioritizes_static_ground_contact(self) -> None:
        pose = "running | first frame: static pre-run contact pose"

        prompt = pipeline.build_qwen_prompt(
            "",
            pose,
            transparent_gif=True,
            quality_retry=4,
        )

        self.assertIn("RUNNING GROUND-CONTACT RECOVERY", prompt)
        self.assertIn("All four legs and paws must be fully visible", prompt)
        self.assertIn(
            "The later video model, not this image, will create the run", prompt
        )
        self.assertNotIn("make the requested action unmistakable", prompt)

    def test_bowl_final_retry_uses_palette_and_canvas_recovery(self) -> None:
        pose = "eating | first frame: natural crouch beside one food bowl"

        prompt = pipeline.build_qwen_prompt(
            "",
            pose,
            transparent_gif=True,
            quality_retry=4,
        )

        self.assertIn("BOWL PALETTE-AND-CANVAS RECOVERY", prompt)
        self.assertIn("neutral-white character lighting", prompt)
        self.assertIn(
            "Do not place a green gap between the bowl and both front paws",
            prompt,
        )
        self.assertNotIn("narrow pure-green gap", prompt)
        self.assertIn("every canvas corner remains pure #00FF00", prompt)
        self.assertNotIn("at thumbnail size", prompt)
        self.assertNotIn("Change the torso angle", prompt)


class SemanticRecoveryTests(unittest.TestCase):
    def test_global_lab_warm_cast_recovery_preserves_accents_alpha_and_props(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.png"
            candidate = root / "candidate.png"
            corrected = root / "corrected.png"
            green = root / "corrected-green.png"
            source_image = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
            candidate_image = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
            source_draw = ImageDraw.Draw(source_image)
            candidate_draw = ImageDraw.Draw(candidate_image)
            source_draw.rectangle((24, 16, 232, 232), fill=(145, 145, 145, 255))
            candidate_draw.rectangle(
                (24, 16, 232, 232),
                fill=(160, 130, 115, 255),
            )
            pink = (195, 135, 145, 255)
            blue = (55, 120, 185, 255)
            bowl = (195, 165, 145, 255)
            candidate_draw.rectangle((118, 68, 137, 87), fill=pink)
            candidate_draw.rectangle((72, 70, 91, 89), fill=blue)
            candidate_draw.rectangle((82, 184, 174, 224), fill=bowl)
            source_image.save(source)
            candidate_image.save(candidate)
            alpha_before = candidate_image.getchannel("A").tobytes()

            result = pipeline.neutralize_achromatic_warm_cast(
                source,
                candidate,
                corrected,
                green_target=green,
            )

            self.assertTrue(result["applied"], result)
            with Image.open(corrected) as loaded:
                output = loaded.convert("RGBA")
                self.assertEqual(output.getchannel("A").tobytes(), alpha_before)
                self.assertEqual(output.getpixel((125, 75)), pink)
                self.assertEqual(output.getpixel((80, 80)), blue)
                self.assertEqual(output.getpixel((128, 205)), bowl)
                self.assertEqual(
                    output.getpixel((40, 220)),
                    candidate_image.getpixel((40, 220)),
                )
                corrected_fur = output.getpixel((60, 60))
                original_fur = candidate_image.getpixel((60, 60))
                self.assertLess(
                    abs(corrected_fur[0] - corrected_fur[2]),
                    abs(original_fur[0] - original_fur[2]),
                )
                above_fade = output.getpixel((60, 132))
                below_fade = output.getpixel((60, 136))
                self.assertLess(
                    sum(abs(a - b) for a, b in zip(above_fade[:3], below_fade[:3])),
                    18,
                )
            with Image.open(green) as loaded:
                self.assertEqual(loaded.convert("RGB").getpixel((0, 0)), (0, 255, 0))
            self.assertEqual(
                result["protected_hash_before"],
                result["protected_hash_after"],
            )
            self.assertEqual(
                result["alpha_sha256_before"],
                result["alpha_sha256_after"],
            )

    @patch.object(pipeline, "GENERATION_VISUAL_AUDITS_BLOCKING", True)
    def test_strict_semantic_recovery_gate_is_fail_closed_for_every_quality_layer(
        self,
    ) -> None:
        passed = {"acceptable": True, "issues": []}
        failed = {"acceptable": False, "issues": ["rejected"]}

        self.assertIsNone(
            pipeline.semantic_recovery_gate_failure(
                passed,
                passed,
                passed,
                passed,
            )
        )
        self.assertEqual(
            pipeline.semantic_recovery_gate_failure(
                failed,
                passed,
                passed,
                passed,
            ),
            "semantic_chroma_invalid",
        )
        self.assertEqual(
            pipeline.semantic_recovery_gate_failure(
                passed,
                failed,
                passed,
                passed,
            ),
            "semantic_subject_quality_invalid",
        )
        self.assertEqual(
            pipeline.semantic_recovery_gate_failure(
                passed,
                passed,
                failed,
                passed,
            ),
            "semantic_pose_similarity_invalid",
        )
        self.assertEqual(
            pipeline.semantic_recovery_gate_failure(
                passed,
                passed,
                passed,
                failed,
            ),
            "semantic_identity_palette_invalid",
        )
        self.assertEqual(
            pipeline.semantic_recovery_gate_failure(
                passed,
                passed,
                passed,
                passed,
                failed,
            ),
            "semantic_fur_silhouette_invalid",
        )

    def test_canvas_framing_issues_are_recoverable_before_normalization(self) -> None:
        quality = {
            "acceptable": False,
            "issues": ["subject_too_close_to_canvas_edge", "subject_too_large"],
        }

        self.assertEqual(pipeline.semantic_quality_blocking_issues(quality), [])

    def test_incomplete_subject_issue_still_blocks_semantic_recovery(self) -> None:
        quality = {
            "acceptable": False,
            "issues": [
                "subject_too_close_to_canvas_edge",
                "fewer_than_three_distinct_grounded_paws",
            ],
        }

        self.assertEqual(
            pipeline.semantic_quality_blocking_issues(quality),
            ["fewer_than_three_distinct_grounded_paws"],
        )

    def test_clean_chroma_pose_failure_is_not_semantic_recovery_candidate(self) -> None:
        record = {
            "chroma": {"acceptable": True, "issues": []},
            "quality": {
                "acceptable": False,
                "issues": ["fewer_than_three_distinct_grounded_paws"],
            },
            "similarity": {"acceptable": True, "issues": []},
            "identity_palette": {"acceptable": True, "issues": []},
        }

        self.assertFalse(pipeline.semantic_recovery_candidate_eligible(record))
        self.assertEqual(
            pipeline.pose_attempt_failure_issues([record]),
            ["fewer_than_three_distinct_grounded_paws"],
        )

    def test_unclean_chroma_distinct_identity_safe_candidate_can_use_semantic_recovery(
        self,
    ) -> None:
        record = {
            "chroma": {"acceptable": False, "issues": ["background_not_uniform"]},
            "quality": None,
            "similarity": {"acceptable": True, "issues": []},
            "identity_palette": {"acceptable": True, "issues": []},
        }

        self.assertTrue(pipeline.semantic_recovery_candidate_eligible(record))

    def test_unclean_background_defers_palette_verdict_until_after_recovery(
        self,
    ) -> None:
        record = {
            "chroma": {
                "acceptable": False,
                "issues": ["nontransparent_canvas_corner"],
            },
            "quality": None,
            "similarity": {"acceptable": True, "issues": []},
            "identity_palette": {
                "acceptable": False,
                "issues": ["new_dominant_warm_brown_or_orange"],
            },
        }

        self.assertTrue(pipeline.semantic_recovery_candidate_eligible(record))

    def test_failure_summary_separates_persistent_and_transient_issues(self) -> None:
        attempts = [
            {
                "attempt": 1,
                "chroma": {"issues": []},
                "identity_palette": {
                    "issues": ["achromatic_source_recolored_with_warm_fur"]
                },
            },
            {
                "attempt": 2,
                "chroma": {
                    "issues": [
                        "foreground_fraction_too_large",
                        "nontransparent_canvas_corner",
                    ]
                },
                "identity_palette": {
                    "issues": ["achromatic_source_recolored_with_warm_fur"]
                },
            },
            {
                "attempt": 3,
                "chroma": {"issues": []},
                "identity_palette": {
                    "issues": ["achromatic_source_recolored_with_warm_fur"]
                },
            },
        ]

        summary = pipeline.pose_attempt_failure_summary(attempts)

        self.assertIn("persistent: achromatic_source_recolored_with_warm_fur", summary)
        self.assertIn(
            "transient: attempt 2: foreground_fraction_too_large, "
            "nontransparent_canvas_corner",
            summary,
        )


class EatingFurSilhouetteGateTests(unittest.TestCase):
    def test_smooth_translated_contour_passes_profile_relative_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "reference.png"
            candidate = root / "candidate.png"
            _save_eating_fur_silhouette(reference)
            _save_eating_fur_silhouette(candidate, offset=(3, -2))

            result = pipeline.eating_fur_silhouette_quality(
                candidate,
                reference,
            )

        self.assertTrue(result["acceptable"], result)
        self.assertEqual(result["issues"], [])

    def test_jagged_expanded_contour_fails_profile_relative_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "reference.png"
            candidate = root / "candidate.png"
            _save_eating_fur_silhouette(reference)
            _save_eating_fur_silhouette(candidate, jagged=True)

            result = pipeline.eating_fur_silhouette_quality(
                candidate,
                reference,
            )

        self.assertFalse(result["acceptable"], result)
        self.assertTrue(
            {
                "fur_top_contour_too_rough",
                "fur_upper_silhouette_too_jagged",
            }
            & set(result["issues"]),
            result,
        )


class RunningGroundContactTests(unittest.TestCase):
    def test_two_grounded_legs_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            candidate = Path(temporary) / "two.png"
            _save_grounded_subject(candidate, 2)
            result = pipeline.pose_cutout_quality(
                candidate, "running | first frame: ready"
            )

        self.assertFalse(result["acceptable"])
        self.assertIn("fewer_than_three_distinct_grounded_paws", result["issues"])

    def test_identity_only_canonical_recovery_defers_running_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            candidate = Path(temporary) / "canonical.png"
            _save_grounded_subject(candidate, 2)
            result = pipeline.pose_cutout_quality(
                candidate,
                "running | first frame: ready",
                validate_action_geometry=False,
            )

        self.assertTrue(result["acceptable"], result)
        self.assertNotIn(
            "fewer_than_three_distinct_grounded_paws",
            result["issues"],
        )

    def test_three_grounded_legs_are_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            candidate = Path(temporary) / "three.png"
            _save_grounded_subject(candidate, 3)
            result = pipeline.pose_cutout_quality(
                candidate, "running | first frame: ready"
            )

        self.assertTrue(result["acceptable"], result)
        self.assertGreaterEqual(result["lower_body_component_peak"], 3)

    def test_running_variants_produce_distinct_prompts(self) -> None:
        slow = (
            "running | first frame: left-facing slow trot contact pose with the right front paw raised; "
            "video: slow trot"
        )
        fast = (
            "running | first frame: right-facing fast run contact pose with the left front paw raised; "
            "video: fast run"
        )

        slow_prompt = pipeline.build_qwen_prompt("", slow, transparent_gif=True)
        fast_prompt = pipeline.build_qwen_prompt("", fast, transparent_gif=True)

        self.assertNotEqual(slow_prompt, fast_prompt)
        self.assertIn("left-facing slow trot contact pose", slow_prompt)
        self.assertIn("right-facing fast run contact pose", fast_prompt)
        self.assertNotIn("video: slow trot", slow_prompt.lower())
        self.assertNotIn("video: fast run", fast_prompt.lower())


class BowlActionSilhouetteTests(unittest.TestCase):
    def test_neutral_vertical_sitting_pose_is_rejected_for_eating(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            candidate = Path(temporary) / "vertical-sitting.png"
            _save_transparent_subject(candidate)

            result = pipeline.pose_cutout_quality(
                candidate,
                "eating | first frame: crouched at a food bowl",
            )

        self.assertFalse(result["acceptable"], result)
        self.assertIn(
            "pose_not_low_or_wide_enough_for_bowl_action",
            result["issues"],
        )
        self.assertLess(
            result["bbox_aspect_ratio"],
            pipeline.QWEN_BOWL_ACTION_MIN_ASPECT_RATIO,
        )

    def test_low_wide_connected_bowl_pose_passes_shape_guard(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            candidate = Path(temporary) / "connected-bowl.png"
            _save_low_connected_eating_subject(candidate)

            result = pipeline.pose_cutout_quality(
                candidate,
                "eating | first frame: crouched at a food bowl",
            )

        self.assertNotIn(
            "pose_not_low_or_wide_enough_for_bowl_action",
            result["issues"],
            result,
        )
        self.assertGreaterEqual(
            result["bbox_aspect_ratio"],
            pipeline.QWEN_BOWL_ACTION_MIN_ASPECT_RATIO,
        )
        self.assertEqual(result["component_count"], 1)
        self.assertEqual(result["bowl_component_count"], 0)

    def test_connected_floor_shelf_is_rejected_for_eating(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            candidate = Path(temporary) / "floor-shelf.png"
            _save_eating_subject_with_floor_shelf(candidate)

            result = pipeline.pose_cutout_quality(
                candidate,
                "eating | first frame: crouched at a food bowl",
            )

        self.assertFalse(result["acceptable"], result)
        self.assertIn(
            "eating_horizontal_foreground_shelf",
            result["issues"],
        )
        shelf = result["eating_horizontal_shelf"]
        self.assertFalse(shelf["acceptable"])
        self.assertGreaterEqual(
            shelf["blocking_events"][0]["jump_px"],
            shelf["thresholds"]["outward_jump_px_min"],
        )
        self.assertGreaterEqual(
            shelf["blocking_events"][0]["persistent_rows"],
            shelf["thresholds"]["persistent_rows_min"],
        )

    def test_drinking_rejects_a_detached_bowl_beyond_contact_range(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            candidate = Path(temporary) / "detached-drinking-bowl.png"
            _save_drinking_subject_with_separate_bowl(candidate, gap=18)
            result = pipeline.pose_cutout_quality(
                candidate,
                "drinking | first frame: muzzle at a water bowl",
            )

        self.assertFalse(result["acceptable"], result)
        self.assertIn("drinking_muzzle_not_at_bowl", result["issues"])
        contact = result["drinking_mouth_bowl_contact"]
        self.assertTrue(contact["separate_bowl_detected"])
        self.assertGreater(
            contact["minimum_subject_bowl_gap_px"],
            pipeline.DRINKING_MOUTH_BOWL_GAP_MAX_PX,
        )

    def test_drinking_accepts_a_separate_bowl_within_contact_range(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            candidate = Path(temporary) / "contact-drinking-bowl.png"
            _save_drinking_subject_with_separate_bowl(candidate, gap=6)
            result = pipeline.pose_cutout_quality(
                candidate,
                "drinking | first frame: muzzle at a water bowl",
            )

        self.assertTrue(result["acceptable"], result)
        contact = result["drinking_mouth_bowl_contact"]
        self.assertTrue(contact["acceptable"], contact)
        self.assertLessEqual(
            contact["minimum_subject_bowl_gap_px"],
            pipeline.DRINKING_MOUTH_BOWL_GAP_MAX_PX,
        )

    def test_synthetic_low_muzzle_bowl_contact_passes_strict_guard(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            candidate = Path(temporary) / "mouth-at-bowl.png"
            _save_mouth_bowl_contact_subject(candidate)

            result = pipeline.pose_cutout_quality(
                candidate,
                "eating | first frame: crouched at a food bowl",
                require_eating_mouth_contact=True,
            )

        self.assertTrue(result["acceptable"], result)
        contact = result["eating_mouth_bowl_contact"]
        self.assertIsNotNone(contact)
        self.assertTrue(contact["acceptable"], contact)
        self.assertEqual(contact["issues"], [])
        self.assertLessEqual(contact["muzzle_bowl_gap_px"], 8)

    def test_known_real_low_muzzle_bowl_contact_passes_when_available(self) -> None:
        candidate = (
            WORKSPACE_ROOT
            / "test_outputs"
            / "bowl_position_diagnosis"
            / "reference_normalized.png"
        )
        if not candidate.is_file():
            self.skipTest("Known real low-muzzle eating reference is not available")

        result = pipeline.pose_cutout_quality(
            candidate,
            "eating | first frame: crouched at a food bowl",
            require_eating_mouth_contact=True,
        )

        self.assertTrue(result["acceptable"], result)
        contact = result["eating_mouth_bowl_contact"]
        self.assertIsNotNone(contact)
        self.assertTrue(contact["acceptable"], contact)
        self.assertLessEqual(contact["muzzle_bowl_gap_px"], 8)

    def test_strict_contact_rejects_non_eating_low_wide_layouts_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = (
                ("far-muzzle", True, False, "eating_muzzle_not_at_bowl"),
                ("no-bowl", False, True, "eating_bowl_or_food_not_detected"),
            )
            for name, include_bowl, muzzle_at_bowl, expected_issue in cases:
                with self.subTest(case=name):
                    candidate = root / f"{name}.png"
                    _save_mouth_bowl_contact_subject(
                        candidate,
                        muzzle_at_bowl=muzzle_at_bowl,
                        include_bowl=include_bowl,
                    )

                    default_result = pipeline.pose_cutout_quality(
                        candidate,
                        "eating | first frame: crouched at a food bowl",
                    )
                    strict_result = pipeline.pose_cutout_quality(
                        candidate,
                        "eating | first frame: crouched at a food bowl",
                        require_eating_mouth_contact=True,
                    )

                    self.assertTrue(default_result["acceptable"], default_result)
                    self.assertIsNone(
                        default_result["eating_mouth_bowl_contact"]
                    )
                    self.assertFalse(strict_result["acceptable"], strict_result)
                    self.assertIn(expected_issue, strict_result["issues"])
                    self.assertFalse(
                        strict_result["eating_mouth_bowl_contact"]["acceptable"]
                    )


class PoseSimilarityTests(unittest.TestCase):
    def test_unsegmentable_candidate_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            blank = root / "blank.png"
            reference = root / "reference.png"
            Image.new("RGBA", (128, 128), (0, 0, 0, 0)).save(blank)
            _save_transparent_subject(reference)

            result = pipeline.pose_similarity_quality(
                blank,
                pose="eating | first frame: crouched at a food bowl",
                canonical_path=reference,
                previous_poses=[],
            )

        self.assertFalse(result["available"])
        self.assertFalse(result["acceptable"])
        self.assertTrue(result["warning"])
        self.assertGreaterEqual(result["penalty"], 1000.0)
        self.assertIn("candidate_similarity_signature_unavailable", result["issues"])

    def test_unsegmentable_reference_does_not_disable_duplicate_guard_silently(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            blank_reference = root / "blank_reference.png"
            candidate = root / "candidate.png"
            Image.new("RGBA", (128, 128), (0, 0, 0, 0)).save(blank_reference)
            _save_transparent_subject(candidate)

            result = pipeline.pose_similarity_quality(
                candidate,
                pose="lying down | first frame: complete relaxed prone pose",
                canonical_path=blank_reference,
                previous_poses=[],
            )

        self.assertFalse(result["available"])
        self.assertFalse(result["acceptable"])
        self.assertTrue(result["warning"])
        self.assertIn(
            "reference_similarity_signature_unavailable_canonical",
            result["issues"],
        )

    def test_identical_and_translated_subjects_are_rejected_after_normalization(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "reference.png"
            translated = root / "translated.png"
            _save_transparent_subject(reference)
            _save_transparent_subject(translated, offset=(9, 6))

            identical_result = pipeline.pose_similarity_quality(
                reference,
                pose="eating | first frame: crouched at a food bowl",
                canonical_path=reference,
                previous_poses=[],
            )
            translated_result = pipeline.pose_similarity_quality(
                translated,
                pose="eating | first frame: crouched at a food bowl",
                canonical_path=reference,
                previous_poses=[],
            )

        self.assertFalse(identical_result["acceptable"])
        self.assertIn("hard_duplicate_of_canonical", identical_result["issues"])
        self.assertFalse(translated_result["acceptable"])
        self.assertTrue(
            any(
                issue in translated_result["issues"]
                for issue in (
                    "hard_duplicate_of_canonical",
                    "pose_too_similar_to_canonical",
                )
            )
        )
        translated_comparison = translated_result["comparisons"][0]
        self.assertGreaterEqual(translated_comparison["silhouette_iou"], 0.98)

    def test_clearly_different_pose_passes_similarity_guard(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "reference.png"
            lying = root / "lying.png"
            _save_transparent_subject(reference)
            _save_transparent_subject(lying, lying=True)

            result = pipeline.pose_similarity_quality(
                lying,
                pose="lying down | first frame: complete relaxed prone pose",
                canonical_path=reference,
                previous_poses=[],
            )

        self.assertTrue(result["acceptable"], result)
        self.assertEqual(result["issues"], [])
        self.assertLess(
            result["comparisons"][0]["silhouette_iou"],
            pipeline.QWEN_POSE_SIMILARITY_IOU_THRESHOLD,
        )

    def test_eating_uses_stricter_canonical_change_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical = root / "canonical.png"
            candidate = root / "eating.png"
            _save_transparent_subject(canonical)
            _save_transparent_subject(candidate)

            comparison = {
                "available": True,
                "sha256_equal": False,
                "silhouette_iou": 0.915,
                "dhash_distance": 6,
                "candidate_aspect_ratio": 0.9,
                "reference_aspect_ratio": 0.9,
            }
            with (
                patch.object(
                    pipeline,
                    "pose_similarity_signature",
                    return_value={"available": True},
                ),
                patch.object(
                    pipeline,
                    "compare_pose_signatures",
                    return_value=comparison,
                ),
            ):
                result = pipeline.pose_similarity_quality(
                    candidate,
                    pose="eating | first frame: crouched at a food bowl",
                    canonical_path=canonical,
                    previous_poses=[],
                )

        self.assertFalse(result["acceptable"], result)
        self.assertIn("pose_too_similar_to_canonical", result["issues"])
        canonical_comparison = result["comparisons"][0]
        self.assertTrue(canonical_comparison["action_requires_canonical_change"])
        self.assertEqual(canonical_comparison["iou_threshold"], 0.90)
        self.assertEqual(canonical_comparison["dhash_threshold"], 10)

    def test_known_diagnostic_duplicate_is_rejected_when_available(self) -> None:
        root = WORKSPACE_ROOT / "diagnostics" / "20260716_lora_audit_95ef994953"
        canonical = root / "cartoon_reference_transparent.png"
        problem = root / "pose_01_keyframe_transparent.png"
        if not canonical.is_file() or not problem.is_file():
            self.skipTest("Known diagnostic duplicate images are not available")

        result = pipeline.pose_similarity_quality(
            problem,
            pose=pipeline.DEFAULT_POSES[0],
            canonical_path=canonical,
            previous_poses=[],
        )

        self.assertFalse(result["acceptable"], result)
        self.assertTrue(
            any(
                issue.startswith(("hard_duplicate_of_", "pose_too_similar_to_"))
                for issue in result["issues"]
            ),
            result,
        )

    def test_existing_fun_control_four_poses_pass_when_available(self) -> None:
        root = (
            WORKSPACE_ROOT
            / "test_outputs"
            / "fun_control_full_chain"
            / "20260715T060927Z-f3d14327bc"
        )
        canonical = root / "cartoon_reference_transparent.png"
        pose_files = [
            root / f"pose_{index:02d}_qwen_cutout.png" for index in range(1, 5)
        ]
        if not canonical.is_file() or not all(path.is_file() for path in pose_files):
            self.skipTest("Existing four-pose Fun-Control outputs are not available")

        previous: list[dict[str, object]] = []
        results = []
        for index, (pose, path) in enumerate(
            zip(pipeline.DEFAULT_POSES, pose_files, strict=True),
            start=1,
        ):
            result = pipeline.pose_similarity_quality(
                path,
                pose=pose,
                canonical_path=canonical,
                previous_poses=previous,
            )
            results.append(result)
            self.assertTrue(result["acceptable"], f"pose {index}: {result}")
            previous.append(
                {
                    "label": f"pose_{index}",
                    "pose": pose,
                    "pose_index": index,
                    "path": path,
                }
            )

        self.assertEqual(len(results), 4)


if __name__ == "__main__":
    unittest.main()
