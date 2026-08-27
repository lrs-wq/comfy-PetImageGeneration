from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

import local_video_api as pipeline
from src.comfy_pet_pipeline.pipeline import subject_replacement as scail_pipeline

from tests.integration.local_video_subject_replacement_test_support import (
    SubjectReplacementQwenReferenceTestBase,
)


class SubjectReplacementQwenPromptWorkflowTests(SubjectReplacementQwenReferenceTestBase):
    def test_white_maquette_cleanup_removes_floor_but_preserves_toilet_tray(
        self,
    ) -> None:
        rgba = np.zeros((128, 128, 4), dtype=np.uint8)
        rgba[22:94, 42:88] = (220, 220, 216, 255)
        rgba[90:112, 4:124] = (59, 76, 36, 210)

        clean, metadata = scail_pipeline.white_maquette_semantic_silhouette(
            rgba,
        )

        self.assertEqual(int(clean[100, 8]), 0)
        self.assertEqual(int(clean[60, 60]), 255)
        self.assertEqual(
            metadata["policy"],
            "neutral_material_floor_shadow_rejection_v2",
        )

        tray_rgba = rgba.copy()
        tray_rgba[88:108, 16:112] = (205, 190, 145, 255)
        tray, _ = scail_pipeline.white_maquette_semantic_silhouette(
            tray_rgba,
        )
        self.assertEqual(int(tray[100, 20]), 255)

        pose_frames, _, pose_metadata = (
            scail_pipeline.make_pose_and_mask_frames(
                [Image.fromarray(tray_rgba, mode="RGBA")],
                (128, 128),
                white_maquette_cleanup=True,
            )
        )
        pose_rgb = np.asarray(pose_frames[0], dtype=np.uint8)
        colored = np.any(pose_rgb != pose_rgb[:, :, :1], axis=2)
        self.assertFalse(np.any(colored))
        self.assertFalse(
            pose_metadata["white_maquette_source_chroma_preserved"]
        )

    def test_white_maquette_cleanup_recovers_green_spill_between_body_and_legs(
        self,
    ) -> None:
        rgba = np.zeros((160, 160, 4), dtype=np.uint8)
        neutral = (226, 226, 218, 255)
        rgba[18:68, 38:122] = neutral
        rgba[92:148, 42:66] = neutral
        rgba[92:148, 94:118] = neutral
        rgba[56:112, 48:112] = (170, 202, 138, 255)
        rgba[140:156, 4:156] = (59, 76, 36, 210)

        clean, metadata = scail_pipeline.white_maquette_semantic_silhouette(
            rgba,
        )

        self.assertEqual(int(clean[82, 80]), 255)
        self.assertEqual(int(clean[148, 8]), 0)
        self.assertEqual(metadata["retained_components"], 1)
        self.assertEqual(metadata["dominant_component_fraction"], 1.0)
        self.assertGreater(metadata["green_spill_pixels_recovered"], 0)

    def test_white_maquette_cleanup_rejects_large_disconnected_body_piece(
        self,
    ) -> None:
        rgba = np.zeros((160, 160, 4), dtype=np.uint8)
        rgba[16:86, 18:78] = (226, 226, 218, 255)
        rgba[92:150, 92:144] = (226, 226, 218, 255)

        with self.assertRaisesRegex(
            ValueError,
            "split the subject into large disconnected components",
        ):
            scail_pipeline.white_maquette_semantic_silhouette(rgba)

    def test_small_dog_extended_actions_have_qwen_and_scail_contracts(self) -> None:
        expected_tokens = {
            "drinking": ("water bowl", "water surface"),
            "running": ("quadruped", "four distinct"),
            "sleeping": ("curled sleeping", "eyes peacefully closed"),
            "toileting": ("toilet tray", "grid insert"),
        }
        for action, tokens in expected_tokens.items():
            with self.subTest(action=action):
                pose = pipeline.subject_replacement_pose_prompt(
                    action,
                    species="dog",
                )
                rules = pipeline.action_first_frame_rules(
                    pose,
                    species="dog",
                )
                prompt = pipeline.build_qwen_prompt(
                    "",
                    pose,
                    transparent_gif=True,
                    pose_reference_available=True,
                    profile_fur_lock=pipeline.UPLOADED_IDENTITY_FUR_LOCK,
                    species="dog",
                )
                for token in tokens:
                    self.assertIn(token, f"{pose} {rules} {prompt}")
                with patch(
                    "sys.argv",
                    [
                        "subject_replacement",
                        "--action",
                        action,
                        "--species",
                        "dog",
                    ],
                ):
                    parsed = scail_pipeline.parse_args()
                self.assertIs(
                    parsed.prompt,
                    scail_pipeline.DOG_ACTION_PROMPTS[action],
                )
                self.assertIs(
                    parsed.negative_prompt,
                    scail_pipeline.DOG_ACTION_NEGATIVE_PROMPTS[action],
                )

    def test_dog_scail_cli_uses_action_specific_negative_prompt(self) -> None:
        with patch(
            "sys.argv",
            ["subject_replacement", "--action", "standing", "--species", "dog"],
        ):
            standing = scail_pipeline.parse_args()
        with patch(
            "sys.argv",
            ["subject_replacement", "--action", "eating", "--species", "dog"],
        ):
            eating = scail_pipeline.parse_args()

        self.assertEqual(
            standing.negative_prompt,
            scail_pipeline.DOG_STANDING_NEGATIVE_PROMPT,
        )
        self.assertIn("static tongue", standing.negative_prompt)
        self.assertNotIn("tongue out", standing.negative_prompt)
        self.assertEqual(
            eating.negative_prompt,
            scail_pipeline.DOG_EATING_NEGATIVE_PROMPT,
        )
        self.assertIn("tongue out", eating.negative_prompt)

    def test_dog_eating_driver_keeps_compressed_kibble_below_rim(self) -> None:
        profile = pipeline.load_subject_profile("dog_white_model_v1")
        driver_path = Path(
            profile["runtime"]["actions"]["eating"]["driving_animation"]
        )
        capture = cv2.VideoCapture(str(driver_path))
        above_rim_counts: list[int] = []
        food_band_counts: list[int] = []
        try:
            while True:
                decoded, frame = capture.read()
                if not decoded:
                    break
                blue, green, red = cv2.split(frame)
                red16 = red.astype(np.int16)
                green16 = green.astype(np.int16)
                blue16 = blue.astype(np.int16)
                kibble = (
                    (red >= 65)
                    & (red16 > green16 + 10)
                    & (green16 > blue16 + 3)
                    & (red16 - blue16 >= 26)
                )
                above_rim_counts.append(int(kibble[360:367, 65:205].sum()))
                food_band_counts.append(int(kibble[373:386, 77:191].sum()))
        finally:
            capture.release()

        self.assertEqual(len(above_rim_counts), 81)
        # H.264 chroma subsampling can bleed a few edge pixels across the rim;
        # a visible mound produces hundreds here in the old white model.
        self.assertLessEqual(max(above_rim_counts), 8)
        self.assertLessEqual(sum(above_rim_counts), 150)
        self.assertGreaterEqual(min(food_band_counts), 750)
        self.assertLessEqual(max(food_band_counts), 1100)

    def test_digest_bound_dog_eating_loop_passes_full_pose_motion_gate(self) -> None:
        profile = pipeline.load_subject_profile("dog_white_model_v1")
        driver_path = Path(
            profile["runtime"]["actions"]["eating"]["driving_animation"]
        )
        frames, durations = scail_pipeline.read_animation_frames(driver_path)
        sampled = scail_pipeline.sample_timed_frames(frames, durations, 81, 16.0)

        _, _, metadata = scail_pipeline.make_full_eating_pose_and_mask_frames(
            sampled,
            (512, 512),
        )

        audit = metadata["identity_motion_audit"]
        self.assertGreaterEqual(
            audit["overall_change_ratio_max"],
            scail_pipeline.EATING_FULL_POSE_OVERALL_CHANGE_RATIO_MIN,
        )
        self.assertGreaterEqual(
            audit["lower_half_change_ratio_max"],
            scail_pipeline.EATING_FULL_POSE_LOWER_HALF_CHANGE_RATIO_MIN,
        )
        self.assertGreaterEqual(
            audit["lower_quarter_change_ratio_max"],
            scail_pipeline.EATING_FULL_POSE_LOWER_QUARTER_CHANGE_RATIO_MIN,
        )
        # Flattening the mound changes a narrow band of the separated prop/muzzle
        # silhouette without changing the digest-bound down/up/down timing. Keep a
        # tight v3 fingerprint instead of treating that expected mask delta as speed.
        self.assertLess(audit["lower_quarter_change_ratio_max"], 0.015)

    def test_dog_eating_prompts_require_low_fixed_kibble_below_rim(self) -> None:
        pose = pipeline.subject_replacement_pose_prompt("eating", species="dog")
        rules = pipeline.action_first_frame_rules(pose, species="dog")
        qwen_prompt = pipeline.build_qwen_prompt(
            "",
            pose,
            transparent_gif=True,
            pose_reference_available=True,
            profile_fur_lock=pipeline.UPLOADED_IDENTITY_FUR_LOCK,
            species="dog",
        )

        for prompt in (pose, rules, qwen_prompt):
            self.assertIn("small, low, level layer", prompt)
            self.assertIn("fully below", prompt)
            self.assertIn("mound", prompt)
            self.assertIn("pyramid", prompt)
        self.assertIn(
            "low food surface and every kibble position remain fixed",
            scail_pipeline.DOG_EATING_PROMPT,
        )
        self.assertIn(
            "only the muzzle may naturally occlude",
            scail_pipeline.DOG_EATING_PROMPT,
        )
        self.assertIn("kibble mound", scail_pipeline.DOG_NEGATIVE_PROMPT)
        self.assertIn("changing food surface", scail_pipeline.DOG_NEGATIVE_PROMPT)

    def test_prompt_only_dog_eating_transfer_locks_breed_and_grounded_bowl(
        self,
    ) -> None:
        pose = pipeline.subject_replacement_pose_prompt("eating", species="dog")
        prompt = pipeline.build_dog_eating_prompt_only_action_transfer_prompt(
            pose,
            profile_fur_lock=pipeline.UPLOADED_IDENTITY_FUR_LOCK,
        )

        self.assertTrue(prompt.startswith("Edit Image 1."))
        self.assertIn("exact apparent age and breed geometry", prompt)
        self.assertIn("ear topology/size/carriage", prompt)
        self.assertIn("tail's exact thickness, shape and carriage", prompt)
        self.assertIn("ON THE GROUND", prompt)
        self.assertIn("completely INSIDE the continuous bowl rim", prompt)
        self.assertIn("no pellet, heap, ring, scatter, spill", prompt)
        self.assertIn("physically inside and overlapped", prompt)
        self.assertIn("never lifts, carries, wears, bites or holds", prompt)
        self.assertIn("folded ears to upright ears", prompt)
        self.assertIn("straight tail to a curled tail", prompt)
        self.assertIn("flat pure #00FF00 green", prompt)
        for forbidden_token in (
            "Image 2",
            "white model",
            "white maquette",
            "action guide",
            "video frame",
        ):
            self.assertNotIn(forbidden_token.lower(), prompt.lower())

        with self.assertRaisesRegex(ValueError, "supports only eating"):
            pipeline.build_dog_eating_prompt_only_action_transfer_prompt(
                pipeline.subject_replacement_pose_prompt(
                    "standing",
                    species="dog",
                )
            )

    def test_prompt_only_dog_drinking_transfer_locks_breed_and_bowl_contact(
        self,
    ) -> None:
        pose = pipeline.subject_replacement_pose_prompt("drinking", species="dog")
        prompt = pipeline.build_dog_drinking_prompt_only_action_transfer_prompt(
            pose,
            profile_fur_lock=pipeline.UPLOADED_IDENTITY_FUR_LOCK,
        )

        self.assertTrue(prompt.startswith("Edit Image 1."))
        self.assertIn("exact apparent age and breed geometry", prompt)
        self.assertIn("ear topology/size/carriage", prompt)
        self.assertIn("tail's exact thickness, shape and carriage", prompt)
        self.assertIn("ON THE GROUND", prompt)
        self.assertIn("physically reaches and slightly overlaps", prompt)
        self.assertIn("tongue tip touches the water surface", prompt)
        self.assertIn("no food, splash, spill, stream, drool", prompt)
        self.assertIn("folded ears to upright ears", prompt)
        self.assertIn("straight tail to a curled tail", prompt)
        self.assertIn("flat pure #00FF00 green", prompt)
        for forbidden_token in (
            "Image 2",
            "white model",
            "white maquette",
            "action guide",
            "video frame",
        ):
            self.assertNotIn(forbidden_token.lower(), prompt.lower())

        with self.assertRaisesRegex(ValueError, "supports only drinking"):
            pipeline.build_dog_drinking_prompt_only_action_transfer_prompt(
                pipeline.subject_replacement_pose_prompt(
                    "standing",
                    species="dog",
                )
            )

    def test_prompt_only_dog_toileting_transfer_locks_breed_and_tray_position(
        self,
    ) -> None:
        pose = pipeline.subject_replacement_pose_prompt("toileting", species="dog")
        prompt = pipeline.build_dog_toileting_prompt_only_action_transfer_prompt(
            pose,
            profile_fur_lock=pipeline.UPLOADED_IDENTITY_FUR_LOCK,
        )

        self.assertTrue(prompt.startswith("Edit Image 1."))
        self.assertIn("exact apparent age and breed geometry", prompt)
        self.assertIn("Place the tray UNDER the dog, never behind it", prompt)
        self.assertIn("front paws and the seated hindquarters", prompt)
        self.assertIn("wider than the seated dog", prompt)
        self.assertIn("folded ears to upright ears", prompt)
        self.assertIn("straight tail to a curled tail", prompt)
        self.assertIn("flat pure #00FF00 green", prompt)
        for forbidden_token in (
            "Image 2",
            "white model",
            "white maquette",
            "action guide",
            "video frame",
        ):
            self.assertNotIn(forbidden_token.lower(), prompt.lower())

        with self.assertRaisesRegex(ValueError, "supports only toileting"):
            pipeline.build_dog_toileting_prompt_only_action_transfer_prompt(
                pipeline.subject_replacement_pose_prompt(
                    "standing",
                    species="dog",
                )
            )

    def test_dog_prompts_lock_age_muzzle_chroma_and_driver_defined_idle_pose(
        self,
    ) -> None:
        canonical = pipeline.build_cartoon_reference_prompt(species="dog")
        standing_pose = pipeline.subject_replacement_pose_prompt(
            "standing",
            species="dog",
        )
        standing_rules = pipeline.action_first_frame_rules(
            standing_pose,
            species="dog",
        )
        action_prompt = pipeline.build_qwen_prompt(
            "",
            standing_pose,
            transparent_gif=True,
            pose_reference_available=True,
            profile_fur_lock=pipeline.UPLOADED_IDENTITY_FUR_LOCK,
            species="dog",
        )

        self.assertIn("muzzle length", canonical)
        self.assertIn("Preserve an adult as an adult", canonical)
        self.assertIn("front view is authoritative for the face", canonical)
        self.assertIn("seated or upright", standing_pose)
        self.assertNotIn("stable upright", standing_pose)
        self.assertIn("mouth is visibly open", standing_pose)
        self.assertIn("anatomically attached pink tongue", standing_pose)
        self.assertIn("seated-or-upright torso state", standing_rules)
        self.assertIn("explicitly requires relaxed panting", standing_rules)
        self.assertIn("readable dark mouth cavity", standing_rules)
        self.assertIn("uninterrupted green band below", standing_rules)
        self.assertIn("Never draw a floor", standing_rules)
        self.assertIn("Face and age contract", action_prompt)
        self.assertIn("Never make an adult dog look like a puppy", action_prompt)
        self.assertIn("explicitly overrides only Image 1's still mouth state", action_prompt)
        self.assertIn("Never paste the tongue onto a closed muzzle", action_prompt)
        standing_retry = pipeline.build_qwen_prompt(
            "",
            standing_pose,
            transparent_gif=True,
            quality_retry=2,
            pose_reference_available=True,
            profile_fur_lock=pipeline.UPLOADED_IDENTITY_FUR_LOCK,
            species="dog",
        )
        self.assertIn("FLOOR-ARTIFACT RETRY", standing_retry)
        self.assertIn("nothing may continue downward", standing_retry)
        self.assertIn("pure #00FF00", scail_pipeline.DOG_EATING_PROMPT)
        self.assertIn("pure #00FF00", scail_pipeline.DOG_STANDING_PROMPT)
        self.assertIn("tongue gently extends", scail_pipeline.DOG_STANDING_PROMPT)
        self.assertIn("static tongue", scail_pipeline.DOG_STANDING_NEGATIVE_PROMPT)
        self.assertNotIn("tongue out", scail_pipeline.DOG_STANDING_NEGATIVE_PROMPT)
        self.assertIn("non-green background", scail_pipeline.DOG_NEGATIVE_PROMPT)

    def test_action_matte_edge_clearance_rejects_floor_reflection_before_resize(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            clean_path = Path(temporary) / "clean.png"
            reflected_path = Path(temporary) / "reflected.png"
            clean = np.zeros((512, 512, 4), dtype=np.uint8)
            clean[70:430, 145:370, :3] = (210, 190, 150)
            clean[70:430, 145:370, 3] = 255
            reflected = clean.copy()
            reflected[430:512, 185:335, :3] = (80, 70, 55)
            reflected[430:512, 185:335, 3] = 220
            Image.fromarray(clean, mode="RGBA").save(clean_path)
            Image.fromarray(reflected, mode="RGBA").save(reflected_path)

            clean_quality = pipeline.qwen_action_edge_clearance_quality(
                clean_path,
            )
            reflected_quality = pipeline.qwen_action_edge_clearance_quality(
                reflected_path,
            )

        self.assertTrue(clean_quality["acceptable"])
        self.assertFalse(reflected_quality["acceptable"])
        self.assertIn("bottom", reflected_quality["failing_edges"])
        self.assertIn(
            "action_matte_foreground_touches_canvas_edge",
            reflected_quality["issues"],
        )

    def test_direct_uploaded_eating_prompt_assigns_image_roles(self) -> None:
        prompt = pipeline.build_qwen_prompt(
            "",
            pipeline.subject_replacement_pose_prompt("eating"),
            transparent_gif=True,
            pose_reference_available=True,
            profile_fur_lock=pipeline.UPLOADED_IDENTITY_FUR_LOCK,
            pose_identity_input_policy=(
                pipeline.POSE_IDENTITY_INPUT_POLICY_DIRECT_UPLOAD
            ),
        )

        self.assertIn(
            "Image 1 is the original uploaded pet photo and is the sole appearance source",
            prompt,
        )
        self.assertIn(
            "Image 2 is the frozen validated subject-profile action reference",
            prompt,
        )
        self.assertIn("bowl placement", prompt)
        self.assertIn("authoritative only for joint placement", prompt)
        self.assertIn("natural coat length and volume", prompt)
        self.assertIn("outer fur silhouette", prompt)
        self.assertIn("copy no fur color, marking, face identity", prompt)
        self.assertIn("Do not copy photographic rendering from Image 1", prompt)
        self.assertIn("Image 2 controls action geometry only", prompt)
        self.assertNotIn("compact-coat silhouette standard", prompt)
        self.assertNotIn(
            "Image 1 is the approved canonical target and the only identity",
            prompt,
        )

    def test_uploaded_identity_policies_fail_closed_without_secondary_image(
        self,
    ) -> None:
        for policy in (
            pipeline.POSE_IDENTITY_INPUT_POLICY_DIRECT_UPLOAD,
            pipeline.POSE_IDENTITY_INPUT_POLICY_PROFILE_GEOMETRY_FIRST,
        ):
            with self.subTest(policy=policy):
                with self.assertRaisesRegex(
                    ValueError,
                    "requires a validated secondary",
                ):
                    pipeline.build_qwen_prompt(
                        "",
                        pipeline.subject_replacement_pose_prompt("eating"),
                        transparent_gif=True,
                        pose_reference_available=False,
                        pose_identity_input_policy=policy,
                    )

        with self.assertRaisesRegex(
            ValueError,
            "requires a validated secondary",
        ):
            pipeline.build_qwen_prompt(
                "",
                pipeline.subject_replacement_pose_prompt("grooming"),
                transparent_gif=True,
                pose_reference_available=False,
                pose_identity_input_policy=(
                    pipeline.POSE_IDENTITY_INPUT_POLICY_GROOMING_GEOMETRY_CANONICAL
                ),
            )

    def test_grooming_geometry_canonical_policy_rejects_other_actions(self) -> None:
        with self.assertRaisesRegex(ValueError, "supports only grooming"):
            pipeline.build_qwen_prompt(
                "",
                pipeline.subject_replacement_pose_prompt("eating"),
                transparent_gif=True,
                pose_reference_available=True,
                pose_identity_input_policy=(
                    pipeline.POSE_IDENTITY_INPUT_POLICY_GROOMING_GEOMETRY_CANONICAL
                ),
            )

    def test_main_case_geometry_first_eating_prompt_assigns_image_roles(self) -> None:
        prompt = pipeline.build_qwen_prompt(
            "",
            pipeline.subject_replacement_pose_prompt("eating"),
            transparent_gif=True,
            pose_reference_available=True,
            profile_fur_lock=pipeline.UPLOADED_IDENTITY_FUR_LOCK,
            pose_identity_input_policy=(
                pipeline.POSE_IDENTITY_INPUT_POLICY_PROFILE_GEOMETRY_FIRST
            ),
        )

        self.assertIn(
            "Image 1 is the texture-free low-frequency geometry guide derived directly from "
            "the frozen Main_case eat.mp4 standard",
            prompt,
        )
        self.assertIn("sampler's geometry canvas", prompt)
        self.assertIn("muzzle-to-kibble contact", prompt)
        self.assertIn(
            "not authoritative for identity, body proportions, coat length", prompt
        )
        self.assertIn(
            "Image 2 is the original uploaded pet photo",
            prompt,
        )
        self.assertIn(
            "Image 1 owns only eating pose, bowl, muzzle contact and framing",
            prompt,
        )
        self.assertIn("The subject profile owns no appearance trait", prompt)
        self.assertIn(
            "sole appearance authority",
            prompt,
        )
        self.assertIn("COAT IDENTITY LOCK", prompt)
        self.assertIn("genuine tufts or ruff", prompt)
        self.assertIn("dedicated pet3d style model", prompt)
        self.assertIn(
            "separate identity-palette audit remains locked to the original "
            "uploaded file",
            prompt,
        )
        self.assertNotIn(
            "Image 1 is the original uploaded pet photo and is the sole appearance source",
            prompt,
        )
        self.assertNotIn(
            "Image 1 is the approved canonical target and the only identity",
            prompt,
        )
        self.assertNotIn("short dense smooth coat", prompt)
        self.assertNotIn("profile-controlled", prompt)

    def test_upload_locked_canonical_prompt_preserves_source_coat(self) -> None:
        prompt = pipeline.build_cartoon_reference_prompt(
            "",
            fur_lock=pipeline.UPLOADED_IDENTITY_FUR_LOCK,
        )

        self.assertIn("UPLOAD COAT SURFACE PRESERVATION", prompt)
        self.assertIn(
            "the uploaded views, not the",
            prompt,
        )
        self.assertIn("control natural coat length, volume", prompt)
        self.assertIn("radial brush strokes", prompt)
        self.assertIn("action profile is never an appearance reference", prompt)
        self.assertIn("Identity and apparent age outrank stylization", prompt)
        self.assertIn("never make it younger or more kitten-like", prompt)
        self.assertNotIn("premium cute", prompt)
        self.assertNotIn("animated-film", prompt)
        self.assertIn("Do not use chibi enlargement", prompt)
        self.assertNotIn("slightly oversized round head", prompt)

    def test_profile_geometry_first_final_bowl_retry_has_consistent_contact(
        self,
    ) -> None:
        prompt = pipeline.build_qwen_prompt(
            "",
            pipeline.subject_replacement_pose_prompt("eating"),
            transparent_gif=True,
            quality_retry=4,
            pose_reference_available=True,
            pose_identity_input_policy=(
                pipeline.POSE_IDENTITY_INPUT_POLICY_PROFILE_GEOMETRY_FIRST
            ),
        )

        self.assertIn(
            "Keep exactly one shallow bowl touching or slightly overlapping",
            prompt,
        )
        self.assertIn("Never omit the bowl or visible brown kibble", prompt)
        self.assertIn(
            "Do not place a green gap between the bowl and both front paws",
            prompt,
        )
        self.assertNotIn("narrow pure-green gap", prompt)

    def test_profile_geometry_first_grooming_prompt_assigns_image_roles(self) -> None:
        prompt = pipeline.build_qwen_prompt(
            "",
            pipeline.subject_replacement_pose_prompt("grooming"),
            transparent_gif=True,
            pose_reference_available=True,
            pose_identity_input_policy=(
                pipeline.POSE_IDENTITY_INPUT_POLICY_PROFILE_GEOMETRY_FIRST
            ),
        )

        self.assertIn("frozen Main_case TM.mp4 standard", prompt)
        self.assertIn("Image 2 is the original uploaded pet photo", prompt)
        self.assertIn(
            "fixed RGB(176,176,176) capsule represents only the continuous shoulder-to-mouth raised foreleg",
            prompt,
        )
        self.assertIn("generated only from verified geometry", prompt)
        self.assertIn("one continuous raised foreleg and rounded paw", prompt)
        self.assertIn(
            "never as a sleeve, strap, zipper, cord, pen, stick, pendant",
            prompt,
        )
        self.assertIn(
            "one foreleg rises continuously from its shoulder to one rounded paw",
            prompt,
        )
        self.assertNotIn("all four paws must be fully visible", prompt)

    def test_default_grooming_prompt_keeps_canonical_image1_role(self) -> None:
        prompt = pipeline.build_qwen_prompt(
            "",
            pipeline.subject_replacement_pose_prompt("grooming"),
            transparent_gif=True,
            pose_reference_available=True,
        )

        self.assertTrue(
            prompt.startswith(
                "CRITICAL OUTPUT CONTRACT - satisfy this before every style detail:"
            )
        )
        self.assertIn(
            "Image 1 is the approved canonical target and the only identity",
            prompt,
        )
        self.assertIn(
            "Image 2 is an appearance-free action-layout silhouette derived "
            "directly from the digest-locked neutral-white anatomy-maquette "
            "driving video",
            prompt,
        )
        self.assertIn("same artificial RGB(144,144,144)", prompt)
        self.assertIn(
            "Never translate the artificial gray into black, white, gray, brown, orange",
            prompt,
        )
        self.assertIn("Copy Image 2's seated grooming joint geometry", prompt)
        self.assertIn("Do not mirror the white-maquette limb relationship", prompt)
        self.assertIn("only the foreleg on the viewer's left rises", prompt)
        self.assertIn("near (38% width, 67% height)", prompt)
        self.assertIn("centered near (37% width, 51% height)", prompt)
        self.assertIn(
            "A natural bent furry foreleg in this region is mandatory",
            prompt,
        )
        self.assertIn(
            "no continuous black or dark stroke from muzzle through paw",
            prompt,
        )
        self.assertIn(
            "an unstriped region must remain unstriped",
            prompt,
        )
        self.assertIn(
            "the no-prop rule forbids only a detached rigid item",
            prompt,
        )
        self.assertNotIn("or other vertical item at the mouth", prompt)
        self.assertIn("natural coat-length, coat-volume, outer-silhouette", prompt)
        self.assertIn(
            "Image 2 and the white-maquette driver never control any appearance trait",
            prompt,
        )
        self.assertIn(
            "hind legs remain anatomically attached and may be naturally tucked",
            prompt,
        )
        self.assertNotIn("exactly four complete paws", prompt.lower())
        self.assertNotIn("all four paws must be fully visible", prompt.lower())
        self.assertIn(
            "Style source: always inherit the approved canonical reference in Image 1",
            prompt,
        )
        self.assertNotIn("Image 1 is the original uploaded pet photo", prompt)

    def test_grooming_action_transfer_prompt_keeps_canonical_identity(self) -> None:
        prompt = pipeline.build_advisory_grooming_action_transfer_prompt(
            pipeline.subject_replacement_pose_prompt("grooming"),
            profile_fur_lock=pipeline.UPLOADED_IDENTITY_FUR_LOCK,
        )

        self.assertLess(len(prompt.split()), 180)
        self.assertTrue(prompt.startswith("Edit Image 1."))
        self.assertIn("Move the front leg visible on the left side", prompt)
        self.assertIn("Remove its former ground paw completely", prompt)
        self.assertIn("fill the vacated outline", prompt)
        self.assertIn("Exactly two front paws total", prompt)
        self.assertIn("one rounded furry paw touching", prompt)
        self.assertIn("tiny pink tongue", prompt)
        self.assertIn("Preserve the cat's exact face", prompt)
        for forbidden_token in (
            "Image 2",
            "white maquette",
            "pose silhouette",
            "depth guide",
            "video frame",
        ):
            self.assertNotIn(forbidden_token.lower(), prompt.lower())

        refine_prompt = pipeline.build_advisory_grooming_identity_refine_prompt(
            pipeline.subject_replacement_pose_prompt("grooming"),
            source_attempt=1,
            profile_fur_lock=pipeline.UPLOADED_IDENTITY_FUR_LOCK,
        )
        self.assertIn("CONDITIONAL GROOMING IDENTITY REPAIR", refine_prompt)
        self.assertIn("MASKED UPPER-HEAD PASS", refine_prompt)
        self.assertIn("white maquette is absent from this pass", refine_prompt)
        self.assertIn("action draft from attempt 1", refine_prompt)
        self.assertIn("IMAGE 2 is the approved Qwen full-body canonical", refine_prompt)
        self.assertIn("IMAGE 3 is an enlarged crop", refine_prompt)
        self.assertIn("smooth white-model face", refine_prompt)
        self.assertIn("black oval toy nose", refine_prompt)
        self.assertIn("only pixels enabled", refine_prompt)
        self.assertIn("raised paw", refine_prompt)

        retry_anatomy = pipeline.advisory_grooming_retry_anatomy_prompt(
            ["raised_paw_hard_object_texture_signature"]
        )
        self.assertIn("one connected shoulder", retry_anatomy)
        self.assertIn("a softly bent elbow", retry_anatomy)
        self.assertIn("one rounded furry paw", retry_anatomy)
        for forbidden_token in ("microphone", "zipper", "spring", "pendant"):
            self.assertNotIn(forbidden_token, retry_anatomy.lower())
        retained_paw_retry = pipeline.advisory_grooming_retry_anatomy_prompt(
            ["original_planted_forepaw_topology_retained"]
        )
        self.assertIn("neutral two-front-paws-down topology", retained_paw_retry)
        self.assertIn("remove its old ground paw", retained_paw_retry)
        self.assertIn("pure #00FF00", retained_paw_retry)
        self.assertFalse(
            pipeline.advisory_grooming_candidate_safe_for_masked_repair(
                {"issues": ["raised_paw_hard_object_texture_signature"]}
            )
        )
        self.assertTrue(
            pipeline.advisory_grooming_candidate_safe_for_masked_repair(
                {"issues": ["raised_paw_too_far_from_muzzle"]}
            )
        )
        self.assertFalse(
            pipeline.advisory_grooming_candidate_safe_for_masked_repair(
                {"issues": ["grooming_face_eye_hue_replacement"]}
            )
        )
        self.assertTrue(
            pipeline.advisory_grooming_candidate_safe_for_identity_refine(
                {
                    "available": True,
                    "acceptable": False,
                    "issues": ["new_dominant_dark_markings"],
                    "pose": {"acceptable": True},
                }
            )
        )
        self.assertFalse(
            pipeline.advisory_grooming_candidate_safe_for_identity_refine(
                {
                    "available": True,
                    "acceptable": False,
                    "issues": ["gross_identity_palette_l1_drift"],
                    "pose": {"acceptable": True},
                }
            )
        )
        self.assertFalse(
            pipeline.advisory_grooming_candidate_safe_for_identity_refine(
                {
                    "available": True,
                    "acceptable": True,
                    "issues": [],
                    "pose": {"acceptable": True},
                }
            )
        )
        self.assertFalse(
            pipeline.advisory_grooming_candidate_safe_for_identity_refine(
                {
                    "available": True,
                    "acceptable": False,
                    "issues": ["new_dominant_dark_markings"],
                    "pose": {"acceptable": False},
                }
            )
        )
        face_retry = pipeline.advisory_grooming_retry_anatomy_prompt(
            ["grooming_face_dark_marking_drift"]
        )
        self.assertIn("enlarged Image 2 crop", face_retry)
        self.assertIn("same cheek width", face_retry)
        self.assertTrue(
            pipeline.advisory_grooming_retry_requires_palette_correction(
                ["gross_chromatic_coat_fraction_drift"]
            )
        )
        self.assertFalse(
            pipeline.advisory_grooming_retry_requires_palette_correction(
                ["raised_paw_hard_object_texture_signature"]
            )
        )

        paw_mask = np.zeros((96, 96), dtype=np.uint8)
        cv2.ellipse(paw_mask, (48, 48), (24, 30), 0, 0, 360, 1, -1)
        paw_mask = paw_mask.astype(bool)
        paw_depth, distance_p95 = pipeline._identity_free_grooming_paw_depth(
            paw_mask
        )
        self.assertGreater(distance_p95, 1.0)
        self.assertGreaterEqual(float(paw_depth[paw_mask].min()), 0.48)
        self.assertLessEqual(float(paw_depth[paw_mask].max()), 0.60)
        self.assertGreater(float(paw_depth[paw_mask].std()), 0.0)
        self.assertTrue(np.allclose(paw_depth[~paw_mask], 0.48))

        strict_pose = pipeline.advisory_grooming_minimum_pose_quality(
            {
                "available": True,
                "issues": [
                    "raised_foreleg_structural_edge_density_too_low",
                    "continuous_raised_foreleg_edge_too_short",
                    "raised_paw_too_far_from_muzzle",
                ],
                "metrics": {},
            }
        )
        self.assertFalse(strict_pose["acceptable"])
        self.assertEqual(
            strict_pose["issues"],
            [
                "raised_foreleg_structural_edge_density_too_low",
                "continuous_raised_foreleg_edge_too_short",
                "raised_paw_too_far_from_muzzle",
            ],
        )
        self.assertEqual(strict_pose["warnings"], [])

        retry_guidance = pipeline.advisory_grooming_action_retry_guidance(
            {
                "warnings": [
                    "raised_foreleg_structural_edge_density_too_low",
                    "unrelated_visual_observation",
                ]
            },
            attempt=1,
        )
        self.assertTrue(retry_guidance["requested"])
        self.assertTrue(retry_guidance["blocking"])
        self.assertFalse(retry_guidance["final_attempt_fail_closed"])
        self.assertEqual(
            retry_guidance["warnings"],
            ["raised_foreleg_structural_edge_density_too_low"],
        )
        final_guidance = pipeline.advisory_grooming_action_retry_guidance(
            {"warnings": ["raised_paw_too_far_from_muzzle"]},
            attempt=pipeline.GROOMING_ADVISORY_ACTION_RETRY_BEFORE_ATTEMPT,
        )
        self.assertFalse(final_guidance["requested"])
        self.assertTrue(final_guidance["blocking"])
        self.assertTrue(final_guidance["never_blocks_final_attempt"])
        self.assertFalse(final_guidance["final_attempt_fail_closed"])

        retry_prompt = pipeline.build_qwen_prompt(
            "",
            pipeline.subject_replacement_pose_prompt("grooming"),
            transparent_gif=True,
            quality_retry=3,
            pose_reference_available=True,
        )
        self.assertIn("GROOMING SIDE RECOVERY", retry_prompt)
        self.assertIn("do not raise the viewer's-right foreleg", retry_prompt)
        self.assertIn("FINAL GROOMING GEOMETRY CORRECTION", retry_prompt)

        soft_completion_prompt = (
            pipeline.build_canonical_grooming_foreleg_completion_prompt(
                "grooming",
                source_attempt=2,
                soft_depth_anatomy=True,
            )
        )
        self.assertIn(
            "Image 3 is the frozen normalized neutral-depth action anatomy",
            soft_completion_prompt,
        )
        self.assertIn(
            "one living cat from head through torso and limbs",
            soft_completion_prompt,
        )
        self.assertNotIn(
            "single-gray action silhouette",
            soft_completion_prompt,
        )

    def test_default_eating_action_workflow_keeps_canonical_image1(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workflow, uploaded_input, pose_input = self._capture_action_workflow(
                Path(temporary),
                action="eating",
                action_identity_policy=(pipeline.POSE_IDENTITY_INPUT_POLICY_CANONICAL),
            )

        load_images = {
            node_id: node["inputs"]["image"]
            for node_id, node in workflow.items()
            if node.get("class_type") == "LoadImage"
        }
        expected_canonical = "local_video_api/capture-eating/cartoon_reference.png"
        self.assertEqual(set(load_images.values()), {expected_canonical, pose_input})
        self.assertEqual(load_images["1"], expected_canonical)
        self.assertNotIn(uploaded_input, load_images.values())
        pose_load_node = next(
            node_id for node_id, image in load_images.items() if image == pose_input
        )
        pose_scale_node = next(
            node_id
            for node_id, node in workflow.items()
            if node.get("class_type") == "FluxKontextImageScale"
            and node["inputs"].get("image") == [pose_load_node, 0]
        )
        self.assertEqual(workflow["6"]["inputs"]["image1"], ["5", 0])
        self.assertEqual(
            workflow["6"]["inputs"]["image2"],
            [pose_scale_node, 0],
        )
        self.assertEqual(workflow["10"]["inputs"]["pixels"], ["5", 0])
        self.assertEqual(workflow["14"]["inputs"]["latent_image"], ["10", 0])
        action_prompt = workflow["6"]["inputs"]["prompt"]
        self.assertTrue(
            action_prompt.startswith(
                "CRITICAL OUTPUT CONTRACT - satisfy this before every style detail:"
            )
        )
        self.assertIn(
            "Image 2 is the texture-free action-layout guide derived directly "
            "from the digest-locked neutral-white anatomy-maquette driving video "
            "and is authoritative only for "
            "joint placement, the exact pose, the action-required prop layout, "
            "framing, subject scale, and pure-green background",
            action_prompt,
        )
        self.assertIn(
            "The lower muzzle is inside the brown kibble",
            action_prompt,
        )
        self.assertIn(
            "rear legs may be naturally tucked or partly occluded",
            action_prompt,
        )
        self.assertNotIn(
            "exactly four complete paws",
            action_prompt.lower(),
        )
        self.assertNotIn(
            "all four paws must be fully visible",
            action_prompt.lower(),
        )
        loaded_loras = {
            node["inputs"]["lora_name"]: node["inputs"]["strength_model"]
            for node in workflow.values()
            if node.get("class_type") == "LoraLoaderModelOnly"
        }
        self.assertEqual(loaded_loras, {pipeline.QWEN_LORA: 1.0})
        photography_lora_names = {
            item["model_name"]
            for item in pipeline.IMAGE_PROFILES[
                pipeline.SUBJECT_REPLACEMENT_REFERENCE_IMAGE_PROFILE
            ]["user_loras"]
        }
        self.assertTrue(photography_lora_names.isdisjoint(loaded_loras))
        self.assertNotIn(self.TEST_QWEN_STYLE_LORA, loaded_loras)
        self.assertEqual(workflow["14"]["inputs"]["model"], ["13", 0])

    def test_default_grooming_action_workflow_keeps_canonical_image1(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workflow, uploaded_input, pose_input = self._capture_action_workflow(
                Path(temporary),
                action="grooming",
                action_identity_policy=(pipeline.POSE_IDENTITY_INPUT_POLICY_CANONICAL),
            )

        load_images = {
            node_id: node["inputs"]["image"]
            for node_id, node in workflow.items()
            if node.get("class_type") == "LoadImage"
        }
        expected_canonical = "local_video_api/capture-grooming/cartoon_reference.png"
        self.assertEqual(set(load_images.values()), {expected_canonical, pose_input})
        self.assertEqual(load_images["1"], expected_canonical)
        self.assertNotIn(uploaded_input, load_images.values())
        pose_load_node = next(
            node_id for node_id, image in load_images.items() if image == pose_input
        )
        pose_scale_node = next(
            node_id
            for node_id, node in workflow.items()
            if node.get("class_type") == "FluxKontextImageScale"
            and node["inputs"].get("image") == [pose_load_node, 0]
        )
        self.assertEqual(workflow["6"]["inputs"]["image1"], ["5", 0])
        self.assertEqual(
            workflow["6"]["inputs"]["image2"],
            [pose_scale_node, 0],
        )
        self.assertEqual(workflow["10"]["inputs"]["pixels"], ["5", 0])
        self.assertEqual(workflow["14"]["inputs"]["latent_image"], ["10", 0])
        self.assertEqual(
            workflow["14"]["inputs"]["denoise"],
            pipeline.QWEN_GROOMING_CANONICAL_GEOMETRY_DENOISE_BY_ATTEMPT[0],
        )
        action_prompt = workflow["6"]["inputs"]["prompt"]
        self.assertIn(
            "Image 1 is the approved canonical target and the only identity",
            action_prompt,
        )
        self.assertIn(
            "Image 2 is an appearance-free action-layout silhouette",
            action_prompt,
        )
        self.assertIn(
            "Copy Image 2's seated grooming joint geometry",
            action_prompt,
        )
        self.assertNotIn(
            "CRITICAL GROOMING TRANSFER CONTRACT",
            action_prompt,
        )
        loaded_loras = {
            node["inputs"]["lora_name"]: node["inputs"]["strength_model"]
            for node in workflow.values()
            if node.get("class_type") == "LoraLoaderModelOnly"
        }
        self.assertEqual(loaded_loras, {pipeline.QWEN_LORA: 1.0})
        self.assertNotIn(self.TEST_QWEN_STYLE_LORA, loaded_loras)

    def test_advisory_grooming_uses_verified_single_image_four_step_edit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workflow, uploaded_input, pose_input = self._capture_action_workflow(
                Path(temporary),
                action="grooming",
                action_identity_policy=(pipeline.POSE_IDENTITY_INPUT_POLICY_CANONICAL),
                identity_palette_guard=True,
                uploaded_white_maquette=True,
                visual_audits_blocking=False,
            )

        action_prompt = workflow["6"]["inputs"]["prompt"]
        negative_prompt = workflow["7"]["inputs"]["prompt"]
        load_images = {
            node["inputs"]["image"]
            for node in workflow.values()
            if node.get("class_type") == "LoadImage"
        }
        expected_canonical = "local_video_api/capture-grooming/cartoon_reference.png"
        self.assertEqual(load_images, {expected_canonical})
        self.assertNotIn(pose_input, load_images)
        self.assertNotIn(uploaded_input, load_images)
        self.assertTrue(action_prompt.startswith("Edit Image 1."))
        self.assertIn("Remove its former ground paw completely", action_prompt)
        self.assertIn("Exactly two front paws total", action_prompt)
        self.assertLess(len(action_prompt.split()), 180)
        self.assertNotIn("image2", workflow["6"]["inputs"])
        self.assertNotIn("image2", workflow["7"]["inputs"])
        self.assertNotIn("image3", workflow["6"]["inputs"])
        self.assertNotIn("image3", workflow["7"]["inputs"])
        self.assertEqual(negative_prompt, "")
        self.assertEqual(
            workflow["14"]["inputs"]["denoise"],
            pipeline.GROOMING_ADVISORY_ACTION_TRANSFER_DENOISE,
        )
        self.assertEqual(
            workflow["14"]["inputs"]["steps"],
            4,
        )
        self.assertFalse(
            any(
                node.get("class_type") == "SetLatentNoiseMask"
                for node in workflow.values()
            )
        )
        self.assertEqual(workflow["14"]["inputs"]["latent_image"], ["10", 0])

    def test_advisory_grooming_identity_safe_draft_skips_generative_face_refine(
        self,
    ) -> None:
        minimum_gate = {
            "available": True,
            "acceptable": True,
            "issues": [],
            "pose": {"acceptable": True},
        }

        self.assertFalse(
            pipeline.advisory_grooming_candidate_safe_for_identity_refine(
                minimum_gate
            )
        )
        self.assertLess(
            pipeline.GROOMING_ADVISORY_IDENTITY_REFINE_DENOISE,
            1.0,
        )
        self.assertEqual(
            pipeline.GROOMING_ADVISORY_IDENTITY_REFINE_HEAD_EXPANSION_PX,
            0,
        )

    def test_advisory_grooming_identity_refine_requires_reported_face_drift(
        self,
    ) -> None:
        self.assertTrue(
            pipeline.advisory_grooming_candidate_safe_for_identity_refine(
                {
                    "available": True,
                    "acceptable": False,
                    "issues": ["grooming_face_dark_marking_drift"],
                    "pose": {"acceptable": True},
                }
            )
        )
        self.assertFalse(
            pipeline.advisory_grooming_candidate_safe_for_identity_refine(
                {
                    "available": True,
                    "acceptable": False,
                    "issues": ["raised_paw_too_far_from_muzzle"],
                    "pose": {"acceptable": False},
                }
            )
        )

    def test_advisory_grooming_does_not_schedule_post_action_identity_repair(
        self,
    ) -> None:
        self.assertEqual(
            pipeline.GROOMING_ADVISORY_IDENTITY_REFINE_MAX_ATTEMPTS,
            0,
        )

    def test_advisory_grooming_profile_budget_is_raw_action_only(self) -> None:
        profiles = [
            {
                "name": f"draft-{index}",
                "user_loras": [],
                "steps": 10,
                "quality_retry": index,
                "denoise": 1.0,
            }
            for index in range(1, 4)
        ]

        selected = pipeline.advisory_grooming_generation_profiles(
            "grooming",
            profiles,
            enabled=True,
        )

        self.assertEqual(
            len(selected),
            pipeline.GROOMING_ADVISORY_TOTAL_ATTEMPTS,
        )
        self.assertEqual(
            len(pipeline.GROOMING_ADVISORY_ACTION_TRANSFER_DENOISE_BY_ATTEMPT),
            pipeline.GROOMING_ADVISORY_ACTION_TRANSFER_MAX_ATTEMPTS,
        )
        self.assertEqual(
            set(pipeline.GROOMING_ADVISORY_ACTION_TRANSFER_DENOISE_BY_ATTEMPT),
            {1.0},
        )
        self.assertEqual(
            len(selected),
            pipeline.GROOMING_ADVISORY_ACTION_TRANSFER_MAX_ATTEMPTS,
        )
        self.assertNotIn(
            "grooming_conditional_identity_refine",
            {profile["name"] for profile in selected},
        )

        health_payload = pipeline.health()
        grooming_health = health_payload["qwen_style"]["pose_guard"][
            "grooming_advisory_first_frame"
        ]
        self.assertIn("conditional_identity_refine", grooming_health)
        self.assertNotIn("mandatory_identity_refine", grooming_health)
        self.assertFalse(grooming_health["conditional_identity_refine"]["enabled"])
        self.assertFalse(grooming_health["post_qwen_pixel_composite_enabled"])
        self.assertTrue(grooming_health["raw_qwen_output_required"])
        self.assertTrue(grooming_health["raw_qwen_selection_byte_preserving"])
        self.assertTrue(grooming_health["prompt_only_action_geometry"])
        self.assertFalse(grooming_health["video_first_frame_loaded_by_qwen"])
        self.assertFalse(grooming_health["white_maquette_loaded_by_qwen"])
        self.assertFalse(grooming_health["appearance_free_pose_seed_loaded_by_qwen"])
        self.assertFalse(
            grooming_health["same_canonical_face_identity_reference_loaded_by_qwen"]
        )
        self.assertTrue(grooming_health["identity_palette_prompt_conditioning"])
        self.assertFalse(grooming_health["negative_prompt_conditioning"])
        self.assertEqual(grooming_health["steps"], 4)
        self.assertEqual(
            grooming_health["image_roles"],
            {"image1": "approved_canonical_identity_sampler"},
        )
        self.assertIn("canonical_local_action_edit_mask", grooming_health)
        minimum_contract = grooming_health["minimum_contract"]
        self.assertFalse(minimum_contract["final_attempt_fail_closed"])
        self.assertTrue(
            minimum_contract["action_retry_guidance_never_blocks_final_attempt"]
        )
        self.assertEqual(
            minimum_contract["terminal_selection_policy"],
            "identity_first_lowest_risk_complete_chain_v1",
        )
        self.assertEqual(minimum_contract["advisory_only_pose_observations"], [])
        self.assertIn(
            "raised_paw_too_far_from_muzzle",
            minimum_contract["blocking_pose_issues"],
        )

    def test_advisory_grooming_attempt3_stays_canonical_prompt_only(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workflow, uploaded_input, pose_input = self._capture_action_workflow(
                Path(temporary),
                action="grooming",
                action_identity_policy=(pipeline.POSE_IDENTITY_INPUT_POLICY_CANONICAL),
                pose_attempt=3,
                advisory_grooming_minimum_acceptable=False,
                action_chroma_acceptable_attempts={1, 2},
                identity_palette_guard=True,
                uploaded_white_maquette=True,
                visual_audits_blocking=False,
            )

        load_images = {
            node["inputs"]["image"]
            for node in workflow.values()
            if node.get("class_type") == "LoadImage"
        }
        expected_prefix = "local_video_api/capture-grooming/"
        self.assertEqual(load_images, {expected_prefix + "cartoon_reference.png"})
        self.assertNotIn(uploaded_input, load_images)
        self.assertNotIn(pose_input, load_images)
        prompt = workflow["6"]["inputs"]["prompt"]
        self.assertTrue(prompt.startswith("Edit Image 1."))
        self.assertIn("Move the front leg visible on the left side", prompt)
        self.assertNotIn("Image 2", prompt)
        self.assertEqual(workflow["7"]["inputs"]["prompt"], "")
        self.assertEqual(workflow["14"]["inputs"]["steps"], 4)
        self.assertFalse(
            any(
                node.get("class_type") == "SetLatentNoiseMask"
                for node in workflow.values()
            )
        )

    def test_advisory_grooming_hard_object_attempt3_restarts_from_clean_canonical(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workflow, uploaded_input, pose_input = self._capture_action_workflow(
                Path(temporary),
                action="grooming",
                action_identity_policy=(pipeline.POSE_IDENTITY_INPUT_POLICY_CANONICAL),
                pose_attempt=3,
                advisory_grooming_minimum_acceptable=False,
                advisory_grooming_minimum_issues=[
                    "raised_paw_hard_object_texture_signature"
                ],
                action_chroma_acceptable_attempts={1, 2},
                identity_palette_guard=True,
                uploaded_white_maquette=True,
                visual_audits_blocking=False,
            )

        expected_prefix = "local_video_api/capture-grooming/"
        load_images = {
            node["inputs"]["image"]
            for node in workflow.values()
            if node.get("class_type") == "LoadImage"
        }
        self.assertEqual(load_images, {expected_prefix + "cartoon_reference.png"})
        self.assertNotIn(uploaded_input, load_images)
        self.assertNotIn(pose_input, load_images)
        self.assertFalse(
            any(
                node.get("class_type") == "SetLatentNoiseMask"
                for node in workflow.values()
            )
        )
        prompt = workflow["6"]["inputs"]["prompt"]
        self.assertTrue(prompt.startswith("Edit Image 1."))
        self.assertNotIn("GROOMING RETRY CORRECTION", prompt)
        self.assertLess(len(prompt.split()), 180)
        self.assertEqual(workflow["7"]["inputs"]["prompt"], "")
        self.assertEqual(workflow["14"]["inputs"]["steps"], 4)

    def test_advisory_grooming_minimum_palette_rejects_identity_takeover(
        self,
    ) -> None:
        source = self._identity_palette_signature_for_test(
            {
                "black": 0.02,
                "gray": 0.51,
                "white": 0.47,
                "warm_brown_or_orange": 0.0,
                "red_or_pink": 0.0,
                "yellow_or_gold": 0.0,
                "green": 0.0,
                "blue_or_cyan": 0.0,
                "purple": 0.0,
            },
            saturated_warm_fraction=0.0,
        )
        candidate = self._identity_palette_signature_for_test(
            {
                "black": 0.38,
                "gray": 0.21,
                "white": 0.39,
                "warm_brown_or_orange": 0.01,
                "red_or_pink": 0.01,
                "yellow_or_gold": 0.0,
                "green": 0.0,
                "blue_or_cyan": 0.0,
                "purple": 0.0,
            },
            saturated_warm_fraction=0.0,
        )

        quality = pipeline.advisory_grooming_minimum_palette_quality(
            source,
            candidate,
        )

        self.assertFalse(quality["acceptable"])
        self.assertIn("gross_identity_palette_l1_drift", quality["issues"])
        self.assertIn("gross_dark_marking_area_drift", quality["issues"])

    def test_advisory_grooming_minimum_palette_allows_pose_lighting_shift(
        self,
    ) -> None:
        source = self._identity_palette_signature_for_test(
            {
                "black": 0.08,
                "gray": 0.42,
                "white": 0.48,
                "warm_brown_or_orange": 0.02,
                "red_or_pink": 0.0,
                "yellow_or_gold": 0.0,
                "green": 0.0,
                "blue_or_cyan": 0.0,
                "purple": 0.0,
            },
            saturated_warm_fraction=0.0,
        )
        candidate = self._identity_palette_signature_for_test(
            {
                "black": 0.14,
                "gray": 0.49,
                "white": 0.34,
                "warm_brown_or_orange": 0.02,
                "red_or_pink": 0.01,
                "yellow_or_gold": 0.0,
                "green": 0.0,
                "blue_or_cyan": 0.0,
                "purple": 0.0,
            },
            saturated_warm_fraction=0.0,
        )

        quality = pipeline.advisory_grooming_minimum_palette_quality(
            source,
            candidate,
        )

        self.assertTrue(quality["acceptable"])
        self.assertEqual(quality["issues"], [])

    def test_advisory_grooming_face_gate_blocks_severe_stripes_and_eye_recolor(
        self,
    ) -> None:
        source = {
            "available": True,
            "fractions": {
                "black": 0.03,
                "white": 0.86,
                "gray": 0.08,
                "chromatic": 0.03,
                "yellow_or_gold": 0.0,
                "green": 0.02,
                "warm_brown_or_yellow": 0.0,
            },
        }
        severe_drift = {
            "available": True,
            "fractions": {
                "black": 0.25,
                "white": 0.60,
                "gray": 0.09,
                "chromatic": 0.06,
                "yellow_or_gold": 0.035,
                "green": 0.001,
                "warm_brown_or_yellow": 0.04,
            },
        }
        normal_lighting = {
            "available": True,
            "fractions": {
                "black": 0.08,
                "white": 0.78,
                "gray": 0.10,
                "chromatic": 0.04,
                "yellow_or_gold": 0.004,
                "green": 0.014,
                "warm_brown_or_yellow": 0.006,
            },
        }

        rejected = pipeline.advisory_grooming_face_identity_quality(
            source,
            severe_drift,
        )
        accepted = pipeline.advisory_grooming_face_identity_quality(
            source,
            normal_lighting,
        )

        self.assertFalse(rejected["acceptable"])
        self.assertIn("grooming_face_dark_marking_drift", rejected["issues"])
        self.assertIn("grooming_face_eye_hue_replacement", rejected["issues"])
        self.assertTrue(accepted["acceptable"])
        self.assertEqual(accepted["issues"], [])

    def test_advisory_grooming_minimum_palette_rejects_observed_white_to_tabby_drift(
        self,
    ) -> None:
        source = self._identity_palette_signature_for_test(
            {
                "black": 0.018,
                "gray": 0.511,
                "white": 0.470,
                "warm_brown_or_orange": 0.0,
                "red_or_pink": 0.0,
                "yellow_or_gold": 0.0,
                "green": 0.001,
                "blue_or_cyan": 0.0,
                "purple": 0.0,
            },
            saturated_warm_fraction=0.0,
        )
        candidate = self._identity_palette_signature_for_test(
            {
                "black": 0.167293,
                "gray": 0.420317,
                "white": 0.305218,
                "warm_brown_or_orange": 0.089351,
                "red_or_pink": 0.017774,
                "yellow_or_gold": 0.000407,
                "green": 0.0,
                "blue_or_cyan": 0.0,
                "purple": 0.0,
            },
            saturated_warm_fraction=0.04,
        )

        quality = pipeline.advisory_grooming_minimum_palette_quality(
            source,
            candidate,
        )

        self.assertFalse(quality["acceptable"])
        self.assertIn("gross_identity_palette_l1_drift", quality["issues"])
        self.assertIn("gross_achromatic_value_balance_drift", quality["issues"])
        self.assertIn("new_dominant_dark_markings", quality["issues"])

    def test_advisory_grooming_minimum_pose_blocks_only_action_topology(
        self,
    ) -> None:
        appearance_only = pipeline.advisory_grooming_minimum_pose_quality(
            {
                "available": True,
                "issues": [
                    "raised_paw_color_inconsistent_with_planted_paw",
                    "guide_chamfer_coverage_too_low",
                ],
            }
        )
        missing_contact = pipeline.advisory_grooming_minimum_pose_quality(
            {
                "available": True,
                "issues": [
                    "raised_paw_color_inconsistent_with_planted_paw",
                    "continuous_raised_foreleg_edge_too_short",
                    "raised_paw_too_far_from_muzzle",
                ],
            }
        )
        unavailable_contact = pipeline.advisory_grooming_minimum_pose_quality(
            {
                "available": True,
                "issues": ["paw_muzzle_gap_unavailable"],
            }
        )
        hard_object = pipeline.advisory_grooming_minimum_pose_quality(
            {
                "available": True,
                "issues": ["guide_chamfer_coverage_too_low"],
                "metrics": {
                    "appearance_consistency": {
                        "available": True,
                        "texture_gradient_p90_ratio": 3.645234,
                        "raised": {"gradient_p90": 249.285098},
                    }
                },
            }
        )
        soft_high_contrast_fur = pipeline.advisory_grooming_minimum_pose_quality(
            {
                "available": True,
                "issues": [],
                "metrics": {
                    "appearance_consistency": {
                        "available": True,
                        "texture_gradient_p90_ratio": 3.2,
                        "raised": {"gradient_p90": 160.0},
                    }
                },
            }
        )

        self.assertTrue(appearance_only["acceptable"])
        self.assertEqual(appearance_only["issues"], [])
        self.assertIn(
            "raised_paw_color_inconsistent_with_planted_paw",
            appearance_only["warnings"],
        )
        self.assertFalse(missing_contact["acceptable"])
        self.assertIn(
            "raised_paw_too_far_from_muzzle",
            missing_contact["issues"],
        )
        self.assertIn(
            "continuous_raised_foreleg_edge_too_short",
            missing_contact["issues"],
        )
        self.assertIn(
            "raised_paw_color_inconsistent_with_planted_paw",
            missing_contact["warnings"],
        )
        self.assertFalse(unavailable_contact["acceptable"])
        self.assertEqual(
            unavailable_contact["issues"],
            ["paw_muzzle_gap_unavailable"],
        )
        self.assertFalse(hard_object["acceptable"])
        self.assertEqual(
            hard_object["issues"],
            ["raised_paw_hard_object_texture_signature"],
        )
        self.assertTrue(
            hard_object["hard_object_texture_evidence"]["blocking"]
        )
        self.assertTrue(soft_high_contrast_fur["acceptable"])
        self.assertFalse(
            soft_high_contrast_fur["hard_object_texture_evidence"]["blocking"]
        )

    def test_canonical_action_edit_gate_rejects_neutral_planted_paw_copy(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical = root / "canonical.png"
            neutral_copy = root / "neutral-copy.png"
            action_edit = root / "action-edit.png"

            canonical_array = np.full(
                (512, 512, 3),
                (0, 255, 0),
                dtype=np.uint8,
            )
            cv2.ellipse(
                canonical_array,
                (256, 290),
                (112, 176),
                0,
                0,
                360,
                (216, 216, 216),
                -1,
            )
            cv2.circle(
                canonical_array,
                (256, 142),
                72,
                (216, 216, 216),
                -1,
            )
            cv2.line(
                canonical_array,
                (226, 290),
                (224, 430),
                (184, 184, 184),
                34,
            )
            cv2.ellipse(
                canonical_array,
                (224, 435),
                (30, 18),
                0,
                0,
                360,
                (184, 184, 184),
                -1,
            )
            Image.fromarray(canonical_array, mode="RGB").save(canonical)
            Image.fromarray(canonical_array, mode="RGB").save(neutral_copy)

            action_array = canonical_array.copy()
            # Remove the original viewer-left lower foreleg and ground paw.
            cv2.rectangle(
                action_array,
                (188, 300),
                (258, 464),
                (0, 255, 0),
                -1,
            )
            # Rebuild that single limb from the shoulder to the mouth.
            cv2.line(
                action_array,
                (226, 300),
                (202, 228),
                (184, 184, 184),
                36,
            )
            cv2.circle(
                action_array,
                (206, 214),
                24,
                (184, 184, 184),
                -1,
            )
            Image.fromarray(action_array, mode="RGB").save(action_edit)

            rejected = (
                pipeline.advisory_grooming_canonical_action_edit_quality(
                    canonical,
                    neutral_copy,
                )
            )
            accepted = (
                pipeline.advisory_grooming_canonical_action_edit_quality(
                    canonical,
                    action_edit,
                )
            )

            self.assertFalse(rejected["acceptable"])
            self.assertIn(
                "canonical_grooming_action_edit_too_weak",
                rejected["issues"],
            )
            self.assertIn(
                "original_planted_forepaw_topology_retained",
                rejected["issues"],
            )
            self.assertTrue(accepted["acceptable"], accepted)
            self.assertEqual(accepted["issues"], [])
            self.assertFalse(accepted["driver_frame_loaded"])
            self.assertFalse(accepted["video_first_frame_loaded_by_qwen"])
            self.assertFalse(accepted["post_qwen_pixel_composite_enabled"])

            relaxed = (
                pipeline.advisory_grooming_canonical_action_edit_quality(
                    canonical,
                    neutral_copy,
                    attempt=(
                        pipeline.GROOMING_ADVISORY_ACTION_TRANSFER_MAX_ATTEMPTS
                    ),
                )
            )
            self.assertFalse(relaxed["acceptable"])
            self.assertLess(
                relaxed["thresholds"][
                    "action_corridor_mean_rgb_delta_min"
                ],
                rejected["thresholds"][
                    "action_corridor_mean_rgb_delta_min"
                ],
            )
            self.assertIn(
                "original_planted_forepaw_topology_retained",
                relaxed["issues"],
            )

    def test_grooming_degraded_selection_prefers_identity_and_anatomy_safety(
        self,
    ) -> None:
        def candidate(
            *,
            attempt: int,
            issues: list[str],
            palette_acceptable: bool = True,
            edge_retention: float = 0.8,
            action_delta: float = 0.08,
            penalty: float = 200.0,
        ) -> dict[str, object]:
            return {
                "attempt": attempt,
                "combined_penalty": penalty,
                "grooming_advisory_minimum_gate": {
                    "available": True,
                    "issues": issues,
                    "identity_palette": {
                        "acceptable": palette_acceptable,
                    },
                    "face_identity": {"acceptable": True},
                    "canonical_action_edit": {
                        "available": True,
                        "metrics": {
                            "original_planted_paw_edge_retention": (
                                edge_retention
                            ),
                            "action_corridor_mean_rgb_delta": action_delta,
                        },
                    },
                },
            }

        identity_drift = candidate(
            attempt=1,
            issues=["gross_identity_palette_l1_drift"],
            palette_acceptable=False,
            edge_retention=0.2,
            action_delta=0.3,
            penalty=20.0,
        )
        hard_object = candidate(
            attempt=2,
            issues=["raised_paw_hard_object_texture_signature"],
            edge_retention=0.3,
            action_delta=0.2,
            penalty=30.0,
        )
        safe_residual = candidate(
            attempt=3,
            issues=["original_planted_forepaw_topology_retained"],
            edge_retention=0.72,
            action_delta=0.10,
            penalty=220.0,
        )
        self.assertLess(
            pipeline.advisory_grooming_degraded_candidate_selection_key(
                safe_residual
            ),
            pipeline.advisory_grooming_degraded_candidate_selection_key(
                hard_object
            ),
        )
        self.assertLess(
            pipeline.advisory_grooming_degraded_candidate_selection_key(
                safe_residual
            ),
            pipeline.advisory_grooming_degraded_candidate_selection_key(
                identity_drift
            ),
        )

    def test_dog_eating_workflow_never_loads_white_model_into_qwen(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workflow, _uploaded_input, pose_input = self._capture_action_workflow(
                Path(temporary),
                action="eating",
                action_identity_policy=(pipeline.POSE_IDENTITY_INPUT_POLICY_CANONICAL),
                pose_attempt=1,
                identity_palette_guard=True,
                visual_audits_blocking=False,
                species="dog",
            )

        load_images = {
            node_id: node["inputs"]["image"]
            for node_id, node in workflow.items()
            if node.get("class_type") == "LoadImage"
        }
        expected_canonical = "local_video_api/capture-eating/cartoon_reference.png"
        self.assertEqual(load_images, {"1": expected_canonical})
        self.assertNotIn(pose_input, load_images.values())
        self.assertNotIn("image2", workflow["6"]["inputs"])
        self.assertNotIn("image3", workflow["6"]["inputs"])
        self.assertTrue(
            workflow["6"]["inputs"]["prompt"].startswith("Edit Image 1.")
        )
        self.assertEqual(workflow["7"]["inputs"]["prompt"], "")
        self.assertEqual(
            workflow["14"]["inputs"]["steps"],
            pipeline.DOG_EATING_PROMPT_ONLY_ACTION_TRANSFER_STEPS,
        )
        self.assertEqual(
            workflow["14"]["inputs"]["denoise"],
            pipeline.DOG_EATING_PROMPT_ONLY_ACTION_TRANSFER_DENOISE,
        )

    def test_dog_drinking_workflow_never_loads_white_model_into_qwen(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workflow, _uploaded_input, pose_input = self._capture_action_workflow(
                Path(temporary),
                action="drinking",
                action_identity_policy=(pipeline.POSE_IDENTITY_INPUT_POLICY_CANONICAL),
                pose_attempt=1,
                identity_palette_guard=True,
                visual_audits_blocking=False,
                species="dog",
            )

        load_images = {
            node_id: node["inputs"]["image"]
            for node_id, node in workflow.items()
            if node.get("class_type") == "LoadImage"
        }
        expected_canonical = "local_video_api/capture-drinking/cartoon_reference.png"
        self.assertEqual(load_images, {"1": expected_canonical})
        self.assertNotIn(pose_input, load_images.values())
        self.assertNotIn("image2", workflow["6"]["inputs"])
        self.assertNotIn("image3", workflow["6"]["inputs"])
        self.assertTrue(
            workflow["6"]["inputs"]["prompt"].startswith("Edit Image 1.")
        )
        self.assertIn(
            "straight tail to a curled tail",
            workflow["6"]["inputs"]["prompt"],
        )
        self.assertEqual(workflow["7"]["inputs"]["prompt"], "")
        self.assertEqual(
            workflow["14"]["inputs"]["steps"],
            pipeline.DOG_DRINKING_PROMPT_ONLY_ACTION_TRANSFER_STEPS,
        )
        self.assertEqual(
            workflow["14"]["inputs"]["denoise"],
            pipeline.DOG_DRINKING_PROMPT_ONLY_ACTION_TRANSFER_DENOISE,
        )

    def test_dog_toileting_workflow_never_loads_white_model_into_qwen(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workflow, _uploaded_input, pose_input = self._capture_action_workflow(
                Path(temporary),
                action="toileting",
                action_identity_policy=(pipeline.POSE_IDENTITY_INPUT_POLICY_CANONICAL),
                pose_attempt=1,
                identity_palette_guard=True,
                visual_audits_blocking=False,
                species="dog",
            )

        load_images = {
            node_id: node["inputs"]["image"]
            for node_id, node in workflow.items()
            if node.get("class_type") == "LoadImage"
        }
        expected_canonical = "local_video_api/capture-toileting/cartoon_reference.png"
        self.assertEqual(load_images, {"1": expected_canonical})
        self.assertNotIn(pose_input, load_images.values())
        self.assertNotIn("image2", workflow["6"]["inputs"])
        self.assertNotIn("image3", workflow["6"]["inputs"])
        self.assertTrue(
            workflow["6"]["inputs"]["prompt"].startswith("Edit Image 1.")
        )
        self.assertIn(
            "Place the tray UNDER the dog, never behind it",
            workflow["6"]["inputs"]["prompt"],
        )
        self.assertEqual(workflow["7"]["inputs"]["prompt"], "")
        self.assertEqual(
            workflow["14"]["inputs"]["steps"],
            pipeline.DOG_TOILETING_PROMPT_ONLY_ACTION_TRANSFER_STEPS,
        )
        self.assertEqual(
            workflow["14"]["inputs"]["denoise"],
            pipeline.DOG_TOILETING_PROMPT_ONLY_ACTION_TRANSFER_DENOISE,
        )

    def test_production_eating_attempt4_refines_pose_safe_palette_drift(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workflow, uploaded_input, pose_input = self._capture_action_workflow(
                Path(temporary),
                action="eating",
                action_identity_policy=(pipeline.POSE_IDENTITY_INPUT_POLICY_CANONICAL),
                pose_attempt=4,
                identity_palette_guard=True,
                canonical_action_palette_drift_attempts={3},
                action_chroma_acceptable_attempts={3},
            )

        load_images = {
            node_id: node["inputs"]["image"]
            for node_id, node in workflow.items()
            if node.get("class_type") == "LoadImage"
        }
        expected_canonical = "local_video_api/capture-eating/cartoon_reference.png"
        refine_input = (
            "local_video_api/capture-eating/eating_identity_refine_source.png"
        )
        self.assertEqual(
            set(load_images.values()),
            {refine_input, expected_canonical, uploaded_input},
        )
        self.assertEqual(load_images["1"], refine_input)
        self.assertNotIn(pose_input, load_images.values())
        prompt = workflow["6"]["inputs"]["prompt"]
        self.assertIn("CRITICAL EATING IDENTITY REFINE", prompt)
        self.assertIn(
            "normalized eating candidate from attempt 3",
            prompt,
        )
        self.assertIn(
            "Image 2 is the approved Qwen full-body canonical",
            prompt,
        )
        self.assertIn(
            "Image 3 is the original uploaded front photograph of this same pet",
            prompt,
        )
        self.assertIn(
            "lower muzzle and mouth must remain physically inside the kibble",
            prompt,
        )
        self.assertIn(
            "retain their light-beige and brown colors without transferring those warm colors into the pet",
            prompt,
        )
        self.assertEqual(
            workflow["14"]["inputs"]["denoise"],
            pipeline.EATING_CANONICAL_IDENTITY_REFINE_DENOISE,
        )
        self.assertEqual(
            workflow["14"]["inputs"]["steps"],
            pipeline.EATING_CANONICAL_IDENTITY_REFINE_STEPS,
        )
        self.assertFalse(
            any(
                node.get("class_type") == "SetLatentNoiseMask"
                for node in workflow.values()
            )
        )
        loaded_loras = {
            node["inputs"]["lora_name"]: node["inputs"]["strength_model"]
            for node in workflow.values()
            if node.get("class_type") == "LoraLoaderModelOnly"
        }
        self.assertEqual(loaded_loras, {pipeline.QWEN_LORA: 1.0})
        self.assertNotIn(self.TEST_QWEN_STYLE_LORA, loaded_loras)

    def test_advisory_eating_palette_drift_retries_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workflow, _uploaded_input, _pose_input = self._capture_action_workflow(
                Path(temporary),
                action="eating",
                action_identity_policy=(pipeline.POSE_IDENTITY_INPUT_POLICY_CANONICAL),
                pose_attempt=2,
                identity_palette_guard=True,
                canonical_action_palette_drift_attempts={1},
                action_chroma_acceptable_attempts={1},
                visual_audits_blocking=False,
            )

        eating_pose = pipeline.subject_replacement_pose_prompt("eating")
        self.assertEqual(
            workflow["14"]["inputs"]["seed"],
            123 + pipeline.pose_seed_offset(eating_pose, 0) + 1009,
        )

    def test_advisory_eating_attempt5_keeps_refining_contact_uncertain_palette_drift(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workflow, uploaded_input, pose_input = self._capture_action_workflow(
                Path(temporary),
                action="eating",
                action_identity_policy=(pipeline.POSE_IDENTITY_INPUT_POLICY_CANONICAL),
                pose_attempt=5,
                identity_palette_guard=True,
                canonical_action_palette_drift_attempts={3},
                action_chroma_acceptable_attempts={3},
                eating_pose_contact_unreliable=True,
                visual_audits_blocking=False,
            )

        load_images = {
            node_id: node["inputs"]["image"]
            for node_id, node in workflow.items()
            if node.get("class_type") == "LoadImage"
        }
        refine_input = (
            "local_video_api/capture-eating/eating_identity_refine_source.png"
        )
        self.assertEqual(load_images["1"], refine_input)
        self.assertNotIn(pose_input, load_images.values())
        self.assertIn(uploaded_input, load_images.values())
        self.assertIn(
            "normalized eating candidate from attempt 3",
            workflow["6"]["inputs"]["prompt"],
        )
        self.assertEqual(
            workflow["14"]["inputs"]["denoise"],
            0.65,
        )

    def test_eating_action_workflow_uses_main_case_geometry_and_uploaded_identity(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workflow, uploaded_input, pose_input = self._capture_action_workflow(
                Path(temporary),
                action="eating",
                action_identity_policy=(
                    pipeline.POSE_IDENTITY_INPUT_POLICY_PROFILE_GEOMETRY_FIRST
                ),
                main_case_raw_identity_chain=True,
            )

        load_images = {
            node_id: node["inputs"]["image"]
            for node_id, node in workflow.items()
            if node.get("class_type") == "LoadImage"
        }
        self.assertEqual(set(load_images.values()), {uploaded_input, pose_input})
        self.assertEqual(load_images["1"], pose_input)
        identity_load_node = next(
            node_id for node_id, image in load_images.items() if image == uploaded_input
        )
        identity_scale_node = next(
            node_id
            for node_id, node in workflow.items()
            if node.get("class_type") == "FluxKontextImageScale"
            and node["inputs"].get("image") == [identity_load_node, 0]
        )
        self.assertEqual(
            workflow["6"]["inputs"]["image1"],
            workflow["7"]["inputs"]["image1"],
        )
        self.assertEqual(workflow["6"]["inputs"]["image1"], ["5", 0])
        self.assertEqual(
            workflow["6"]["inputs"]["image2"],
            workflow["7"]["inputs"]["image2"],
        )
        self.assertEqual(
            workflow["6"]["inputs"]["image2"],
            [identity_scale_node, 0],
        )
        self.assertEqual(workflow["10"]["inputs"]["pixels"], ["5", 0])
        self.assertEqual(workflow["14"]["inputs"]["latent_image"], ["10", 0])
        self.assertIn(
            "Image 1 is the texture-free low-frequency geometry guide derived directly from "
            "the frozen Main_case eat.mp4 standard",
            workflow["6"]["inputs"]["prompt"],
        )
        self.assertIn(
            "Image 2 is the original uploaded pet photo",
            workflow["6"]["inputs"]["prompt"],
        )
        loaded_loras = {
            node["inputs"]["lora_name"]: node["inputs"]["strength_model"]
            for node in workflow.values()
            if node.get("class_type") == "LoraLoaderModelOnly"
        }
        self.assertEqual(
            loaded_loras,
            {
                pipeline.QWEN_LORA: 1.0,
            },
        )
        photography_lora_names = {
            item["model_name"]
            for item in pipeline.IMAGE_PROFILES[
                pipeline.SUBJECT_REPLACEMENT_REFERENCE_IMAGE_PROFILE
            ]["user_loras"]
        }
        self.assertTrue(photography_lora_names.isdisjoint(loaded_loras))
        self.assertEqual(workflow["14"]["inputs"]["model"], ["13", 0])
        self.assertIn(uploaded_input, load_images.values())

    def test_eating_retry_loras_are_lightning_then_pet3d_only(self) -> None:
        photography_lora_names = {
            item["model_name"]
            for item in pipeline.IMAGE_PROFILES[
                pipeline.SUBJECT_REPLACEMENT_REFERENCE_IMAGE_PROFILE
            ]["user_loras"]
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for attempt in (1, 2, 3, 4, 5):
                with self.subTest(attempt=attempt):
                    workflow, _uploaded_input, _pose_input = (
                        self._capture_action_workflow(
                            root / f"attempt-{attempt}",
                            action="eating",
                            action_identity_policy=(
                                pipeline.POSE_IDENTITY_INPUT_POLICY_PROFILE_GEOMETRY_FIRST
                            ),
                            pose_attempt=attempt,
                            main_case_raw_identity_chain=True,
                        )
                    )
                    loaded_loras = {
                        node["inputs"]["lora_name"]: node["inputs"]["strength_model"]
                        for node in workflow.values()
                        if node.get("class_type") == "LoraLoaderModelOnly"
                    }
                    expected_loras = {pipeline.QWEN_LORA: 1.0}
                    if attempt >= 3:
                        expected_loras[self.TEST_QWEN_STYLE_LORA] = 0.15
                    self.assertEqual(loaded_loras, expected_loras)
                    self.assertTrue(photography_lora_names.isdisjoint(loaded_loras))
                    expected_model_node = "17" if attempt >= 3 else "13"
                    self.assertEqual(
                        workflow["14"]["inputs"]["model"],
                        [expected_model_node, 0],
                    )
                    if attempt >= 3:
                        self.assertEqual(
                            workflow["17"]["inputs"]["model"],
                            ["13", 0],
                        )

    def test_profile_geometry_first_eating_requires_pet3d_style_lora(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(
                RuntimeError,
                "requires the dedicated QWEN_STYLE_LORA",
            ):
                self._capture_action_workflow(
                    Path(temporary),
                    action="eating",
                    action_identity_policy=(
                        pipeline.POSE_IDENTITY_INPUT_POLICY_PROFILE_GEOMETRY_FIRST
                    ),
                    style_lora="",
                )

    def test_profile_geometry_first_eating_rejects_non_pet3d_style_lora(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(
                RuntimeError,
                "requires a dedicated pet3d QWEN_STYLE_LORA",
            ):
                self._capture_action_workflow(
                    Path(temporary),
                    action="eating",
                    action_identity_policy=(
                        pipeline.POSE_IDENTITY_INPUT_POLICY_PROFILE_GEOMETRY_FIRST
                    ),
                    style_lora="generic-style-test.safetensors",
                )

    def test_profile_geometry_first_eating_requires_positive_style_strength(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(
                RuntimeError,
                "requires a positive pet3d recovery strength",
            ):
                self._capture_action_workflow(
                    Path(temporary),
                    action="eating",
                    action_identity_policy=(
                        pipeline.POSE_IDENTITY_INPUT_POLICY_PROFILE_GEOMETRY_FIRST
                    ),
                    style_strength=0.0,
                )

    def test_profile_geometry_first_fails_closed_without_raw_identity_chain(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(
                RuntimeError,
                "requires the original uploaded identity chain",
            ):
                self._capture_action_workflow(
                    Path(temporary),
                    action="eating",
                    action_identity_policy=(
                        pipeline.POSE_IDENTITY_INPUT_POLICY_PROFILE_GEOMETRY_FIRST
                    ),
                    main_case_raw_identity_chain=False,
                )

    def test_profile_geometry_first_fails_closed_without_identity_digest(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(
                RuntimeError,
                "input lacks its submission-time digest",
            ):
                self._capture_action_workflow(
                    Path(temporary),
                    action="grooming",
                    action_identity_policy=(
                        pipeline.POSE_IDENTITY_INPUT_POLICY_PROFILE_GEOMETRY_FIRST
                    ),
                    persist_identity_digest=False,
                )

    def test_profile_geometry_first_fails_closed_without_source_digest(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(
                RuntimeError,
                "source lacks its submission-time digest",
            ):
                self._capture_action_workflow(
                    Path(temporary),
                    action="eating",
                    action_identity_policy=(
                        pipeline.POSE_IDENTITY_INPUT_POLICY_PROFILE_GEOMETRY_FIRST
                    ),
                    persist_identity_source_digest=False,
                )

    def test_profile_geometry_first_grooming_uses_main_case_and_upload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workflow, uploaded_input, pose_input = self._capture_action_workflow(
                Path(temporary),
                action="grooming",
                action_identity_policy=(
                    pipeline.POSE_IDENTITY_INPUT_POLICY_PROFILE_GEOMETRY_FIRST
                ),
                main_case_raw_identity_chain=True,
            )

        load_images = {
            node_id: node["inputs"]["image"]
            for node_id, node in workflow.items()
            if node.get("class_type") == "LoadImage"
        }
        self.assertEqual(set(load_images.values()), {pose_input, uploaded_input})
        self.assertEqual(load_images["1"], pose_input)
        self.assertEqual(workflow["14"]["inputs"]["steps"], 8)
        self.assertIn(
            "frozen Main_case TM.mp4 standard", workflow["6"]["inputs"]["prompt"]
        )
        self.assertIn(
            "never as a sleeve, strap, zipper, cord, pen, stick, pendant",
            workflow["6"]["inputs"]["prompt"],
        )

    def test_profile_geometry_first_grooming_never_uses_pet3d_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workflow, _uploaded_input, _pose_input = self._capture_action_workflow(
                Path(temporary),
                action="grooming",
                action_identity_policy=(
                    pipeline.POSE_IDENTITY_INPUT_POLICY_PROFILE_GEOMETRY_FIRST
                ),
                pose_attempt=3,
            )

        loaded_loras = {
            node["inputs"]["lora_name"]
            for node in workflow.values()
            if node.get("class_type") == "LoraLoaderModelOnly"
        }
        self.assertEqual(loaded_loras, {pipeline.QWEN_LORA})
        self.assertEqual(workflow["14"]["inputs"]["steps"], 8)

    def test_grooming_action_uses_geometry_image1_and_canonical_image2(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workflow, uploaded_input, pose_input = self._capture_action_workflow(
                Path(temporary),
                action="grooming",
                action_identity_policy=(
                    pipeline.POSE_IDENTITY_INPUT_POLICY_GROOMING_GEOMETRY_CANONICAL
                ),
            )

        load_images = {
            node_id: node["inputs"]["image"]
            for node_id, node in workflow.items()
            if node.get("class_type") == "LoadImage"
        }
        expected_canonical = "local_video_api/capture-grooming/cartoon_reference.png"
        self.assertEqual(set(load_images.values()), {expected_canonical, pose_input})
        self.assertEqual(load_images["1"], pose_input)
        self.assertNotIn(uploaded_input, load_images.values())
        canonical_load_node = next(
            node_id
            for node_id, image in load_images.items()
            if image == expected_canonical
        )
        canonical_scale_node = next(
            node_id
            for node_id, node in workflow.items()
            if node.get("class_type") == "FluxKontextImageScale"
            and node["inputs"].get("image") == [canonical_load_node, 0]
        )
        self.assertEqual(
            workflow["6"]["inputs"]["image1"],
            workflow["7"]["inputs"]["image1"],
        )
        self.assertEqual(workflow["6"]["inputs"]["image1"], ["5", 0])
        self.assertEqual(
            workflow["6"]["inputs"]["image2"],
            workflow["7"]["inputs"]["image2"],
        )
        self.assertEqual(
            workflow["6"]["inputs"]["image2"],
            [canonical_scale_node, 0],
        )
        self.assertEqual(workflow["10"]["inputs"]["pixels"], ["5", 0])
        self.assertEqual(workflow["14"]["inputs"]["latent_image"], ["10", 0])
        self.assertIn(
            "CRITICAL GROOMING TRANSFER CONTRACT",
            workflow["6"]["inputs"]["prompt"],
        )
        self.assertIn(
            "Image 1 is the appearance-free grooming geometry canvas",
            workflow["6"]["inputs"]["prompt"],
        )
        self.assertIn(
            "Image 2 is the approved Qwen full-body canonical",
            workflow["6"]["inputs"]["prompt"],
        )
        self.assertIn(
            "RGB(176,176,176) only for one continuous raised-foreleg capsule",
            workflow["6"]["inputs"]["prompt"],
        )
        self.assertIn(
            "zero identity, fur-color, marking, material, texture, shadow",
            workflow["6"]["inputs"]["prompt"],
        )
        self.assertIn(
            "never a sleeve, strap, zipper, cord, pen, stick, pendant",
            workflow["6"]["inputs"]["prompt"],
        )
        self.assertIn(
            "complete coat palette",
            workflow["6"]["inputs"]["prompt"],
        )
        self.assertNotIn("green eyes", workflow["6"]["inputs"]["prompt"])
        self.assertNotIn(
            "white/neutral coat palette",
            workflow["6"]["inputs"]["prompt"],
        )
        self.assertIn(
            "Any hat, hood, shirt, coat, sweater, collar, costume, accessory",
            workflow["6"]["inputs"]["prompt"],
        )
        self.assertIn(
            "neutral seated pose, both front paws planted, missing raised foreleg",
            workflow["7"]["inputs"]["prompt"],
        )
        self.assertIn(
            "strap, zipper, cord, pen, pencil, stick, pendant",
            workflow["7"]["inputs"]["prompt"],
        )
        self.assertNotIn(
            "exactly four complete paws",
            workflow["6"]["inputs"]["prompt"].lower(),
        )
        self.assertNotIn(
            "all four paws must be fully visible",
            workflow["6"]["inputs"]["prompt"].lower(),
        )
        loaded_loras = {
            node["inputs"]["lora_name"]: node["inputs"]["strength_model"]
            for node in workflow.values()
            if node.get("class_type") == "LoraLoaderModelOnly"
        }
        self.assertEqual(loaded_loras, {pipeline.QWEN_LORA: 1.0})
        photography_lora_names = {
            item["model_name"]
            for item in pipeline.IMAGE_PROFILES[
                pipeline.SUBJECT_REPLACEMENT_REFERENCE_IMAGE_PROFILE
            ]["user_loras"]
        }
        self.assertTrue(photography_lora_names.isdisjoint(loaded_loras))
        self.assertNotIn(self.TEST_QWEN_STYLE_LORA, loaded_loras)
        self.assertEqual(workflow["14"]["inputs"]["model"], ["13", 0])
        self.assertEqual(workflow["14"]["inputs"]["steps"], 8)
        self.assertEqual(
            workflow["14"]["inputs"]["denoise"],
            pipeline.QWEN_GROOMING_CANONICAL_GEOMETRY_DENOISE_BY_ATTEMPT[0],
        )


if __name__ == "__main__":
    unittest.main()
