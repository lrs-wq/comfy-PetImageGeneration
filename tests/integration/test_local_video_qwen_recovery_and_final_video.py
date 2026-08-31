from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
from PIL import Image, ImageDraw

import local_video_api as pipeline

from tests.integration.local_video_subject_replacement_test_support import (
    SubjectReplacementQwenReferenceTestBase,
    _configured_test_upload_profile,
    _passed_surface_metric,
    _passed_uploaded_front_view_audit,
    _record_fake_canonical_image1,
    _subject_profile_with_action_references,
    _write_fake_geometry_guide,
    _write_test_upload_identity,
)


def _record_fake_action_quality(
    job_dir: Path,
    *,
    action: str,
    pose_index: int,
    reference: Path,
    selected_reference: Path | None = None,
    best_effort: bool = False,
) -> None:
    selected_reference = selected_reference or reference
    workflow_dir = job_dir / "workflows"
    workflow_dir.mkdir(parents=True, exist_ok=True)
    (workflow_dir / f"qwen_pose_{pose_index:02d}_quality.json").write_text(
        json.dumps(
            {
                "selected_attempt": 1,
                "selected_profile": "test_fixture",
                "strict_first_frame_required": best_effort,
                "strict_first_frame_best_effort_selected": best_effort,
                "strict_first_frame_best_effort_issues": (
                    ["identity_palette_not_acceptable"] if best_effort else []
                ),
                "strict_first_frame_best_effort_selection_key": (
                    [0, 0, 1, 0.0] if best_effort else None
                ),
                "strict_first_frame_attempt_limit": (
                    pipeline.QWEN_FIRST_FRAME_MAX_ATTEMPTS
                    if best_effort
                    else None
                ),
                "selected_reference_path": str(selected_reference),
                "selected_reference_sha256": pipeline.file_sha256(
                    selected_reference
                ),
                "original_upload_fallback_published": False,
                "action_removed": False,
                "action": action,
            }
        ),
        encoding="utf-8",
    )


class SubjectReplacementQwenRecoveryAndFinalVideoTests(SubjectReplacementQwenReferenceTestBase):
    def test_canonical_grooming_requires_pet3d_repair_surface_lora(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(
                RuntimeError,
                "requires the dedicated QWEN_STYLE_LORA surface model",
            ):
                self._capture_action_workflow(
                    Path(temporary),
                    action="grooming",
                    action_identity_policy=(
                        pipeline.POSE_IDENTITY_INPUT_POLICY_GROOMING_GEOMETRY_CANONICAL
                    ),
                    style_lora="",
                )

    def test_canonical_grooming_rejects_non_pet3d_repair_surface_lora(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(
                RuntimeError,
                "requires a dedicated pet3d QWEN_STYLE_LORA surface model",
            ):
                self._capture_action_workflow(
                    Path(temporary),
                    action="grooming",
                    action_identity_policy=(
                        pipeline.POSE_IDENTITY_INPUT_POLICY_GROOMING_GEOMETRY_CANONICAL
                    ),
                    style_lora="generic-style-test.safetensors",
                )

    def test_grooming_third_attempt_repairs_best_identity_safe_candidate(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workflow, uploaded_input, pose_input = self._capture_action_workflow(
                Path(temporary),
                action="grooming",
                action_identity_policy=(
                    pipeline.POSE_IDENTITY_INPUT_POLICY_GROOMING_GEOMETRY_CANONICAL
                ),
                pose_attempt=3,
                grooming_gate_acceptable=False,
                identity_palette_guard=True,
            )

        load_images = {
            node_id: node["inputs"]["image"]
            for node_id, node in workflow.items()
            if node.get("class_type") == "LoadImage"
        }
        expected_canonical = "local_video_api/capture-grooming/cartoon_reference.png"
        repair_input = (
            "local_video_api/capture-grooming/grooming_foreleg_repair_source.png"
        )
        anatomy_input = (
            "local_video_api/capture-grooming/grooming_foreleg_repair_anatomy_guide.png"
        )
        noise_mask_input = (
            "local_video_api/capture-grooming/"
            "grooming_foreleg_repair_source_noise_mask.png"
        )
        self.assertEqual(
            set(load_images.values()),
            {
                repair_input,
                expected_canonical,
                anatomy_input,
                noise_mask_input,
            },
        )
        self.assertEqual(load_images["1"], repair_input)
        self.assertNotIn(pose_input, load_images.values())
        self.assertNotIn(uploaded_input, load_images.values())
        self.assertIn(
            "CRITICAL GROOMING FORELEG COMPLETION",
            workflow["6"]["inputs"]["prompt"],
        )
        self.assertIn(
            "best identity-safe near-final candidate from attempt 2",
            workflow["6"]["inputs"]["prompt"],
        )
        self.assertIn(
            "deliberately band-limited identity-matched coat-color, mid-scale fur, and shading seed",
            workflow["6"]["inputs"]["prompt"],
        )
        self.assertIn(
            "Image 3 is a frozen neutral low-frequency anatomy map",
            workflow["6"]["inputs"]["prompt"],
        )
        self.assertIn(
            "a generously expanded feathered full-denoise mask covers the seed plus ordinary surrounding muzzle, shoulder, and chest context",
            workflow["6"]["inputs"]["prompt"],
        )
        self.assertIn(
            "none of its boundary or individual sampled strand pixels may be copied literally",
            workflow["6"]["inputs"]["prompt"],
        )
        self.assertIn(
            "loaded low-strength Pet3D adapter is permitted only as a fine surface prior inside the sampler noise mask",
            workflow["6"]["inputs"]["prompt"],
        )
        self.assertNotIn("flashlight", workflow["6"]["inputs"]["prompt"])
        self.assertNotIn("cylinder", workflow["6"]["inputs"]["prompt"])
        self.assertEqual(
            workflow["14"]["inputs"]["denoise"],
            pipeline.GROOMING_QWEN_FORELEG_REPAIR_DENOISE,
        )
        self.assertEqual(
            workflow["14"]["inputs"]["steps"],
            pipeline.GROOMING_QWEN_FORELEG_REPAIR_STEPS,
        )
        mask_load_node = next(
            node_id
            for node_id, image_name in load_images.items()
            if image_name == noise_mask_input
        )
        masked_latent_nodes = {
            node_id: node
            for node_id, node in workflow.items()
            if node.get("class_type") == "SetLatentNoiseMask"
        }
        self.assertEqual(len(masked_latent_nodes), 1)
        masked_latent_id, masked_latent = next(iter(masked_latent_nodes.items()))
        self.assertEqual(masked_latent["inputs"]["samples"], ["10", 0])
        self.assertEqual(
            masked_latent["inputs"]["mask"],
            [mask_load_node, 1],
        )
        self.assertEqual(
            workflow["14"]["inputs"]["latent_image"],
            [masked_latent_id, 0],
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
                self.TEST_QWEN_STYLE_LORA: (
                    pipeline.GROOMING_QWEN_FORELEG_REPAIR_STYLE_STRENGTH
                ),
            },
        )
        self.assertEqual(workflow["17"]["inputs"]["model"], ["13", 0])
        self.assertEqual(workflow["14"]["inputs"]["model"], ["17", 0])

    def test_grooming_repair_enforces_identity_palette_when_flag_is_disabled(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workflow, uploaded_input, pose_input = self._capture_action_workflow(
                Path(temporary),
                action="grooming",
                action_identity_policy=(
                    pipeline.POSE_IDENTITY_INPUT_POLICY_GROOMING_GEOMETRY_CANONICAL
                ),
                pose_attempt=3,
                grooming_gate_acceptable=False,
                identity_palette_guard=False,
            )

        load_images = {
            node_id: node["inputs"]["image"]
            for node_id, node in workflow.items()
            if node.get("class_type") == "LoadImage"
        }
        repair_input = (
            "local_video_api/capture-grooming/grooming_foreleg_repair_source.png"
        )
        self.assertEqual(load_images["1"], repair_input)
        self.assertNotIn(pose_input, load_images.values())
        self.assertIn(
            "CRITICAL GROOMING FORELEG COMPLETION",
            workflow["6"]["inputs"]["prompt"],
        )
        self.assertNotIn(uploaded_input, load_images.values())
        self.assertTrue(
            any(
                node.get("class_type") == "SetLatentNoiseMask"
                for node in workflow.values()
            )
        )

    def test_production_grooming_attempt3_mask_completes_identity_safe_base(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workflow, uploaded_input, pose_input = self._capture_action_workflow(
                Path(temporary),
                action="grooming",
                action_identity_policy=(pipeline.POSE_IDENTITY_INPUT_POLICY_CANONICAL),
                pose_attempt=3,
                grooming_gate_acceptable=False,
                identity_palette_guard=True,
            )

        load_images = {
            node_id: node["inputs"]["image"]
            for node_id, node in workflow.items()
            if node.get("class_type") == "LoadImage"
        }
        expected_canonical = "local_video_api/capture-grooming/cartoon_reference.png"
        repair_input = (
            "local_video_api/capture-grooming/grooming_foreleg_repair_source.png"
        )
        anatomy_input = (
            "local_video_api/capture-grooming/grooming_foreleg_repair_anatomy_guide.png"
        )
        noise_mask_input = (
            "local_video_api/capture-grooming/"
            "grooming_foreleg_repair_source_noise_mask.png"
        )
        completion_mask = (
            "local_video_api/capture-grooming/"
            "grooming_qwen_pose_guide_completion_mask.png"
        )
        self.assertEqual(
            set(load_images.values()),
            {
                repair_input,
                expected_canonical,
                anatomy_input,
                noise_mask_input,
            },
        )
        self.assertEqual(load_images["1"], repair_input)
        self.assertNotIn(pose_input, load_images.values())
        self.assertNotIn(completion_mask, load_images.values())
        self.assertNotIn(uploaded_input, load_images.values())
        prompt = workflow["6"]["inputs"]["prompt"]
        self.assertIn("CRITICAL MASKED GROOMING COMPLETION", prompt)
        self.assertIn(
            "best identity-palette-safe normalized candidate from attempt 2",
            prompt,
        )
        self.assertIn(
            "Image 2 is the approved Qwen full-body canonical",
            prompt,
        )
        self.assertIn(
            "Image 3 is the frozen appearance-free single-gray action silhouette",
            prompt,
        )
        self.assertIn("viewer-left shoulder-to-elbow-to-wrist-to-paw", prompt)
        self.assertIn("Image 2 remains the sole appearance authority", prompt)
        self.assertEqual(
            workflow["14"]["inputs"]["denoise"],
            pipeline.GROOMING_QWEN_FORELEG_REPAIR_DENOISE,
        )
        self.assertEqual(
            workflow["14"]["inputs"]["steps"],
            pipeline.GROOMING_QWEN_FORELEG_REPAIR_STEPS,
        )
        masked_latent_nodes = {
            node_id: node
            for node_id, node in workflow.items()
            if node.get("class_type") == "SetLatentNoiseMask"
        }
        self.assertEqual(len(masked_latent_nodes), 1)
        masked_latent_id = next(iter(masked_latent_nodes))
        self.assertEqual(
            workflow["14"]["inputs"]["latent_image"],
            [masked_latent_id, 0],
        )
        loaded_loras = {
            node["inputs"]["lora_name"]: node["inputs"]["strength_model"]
            for node in workflow.values()
            if node.get("class_type") == "LoraLoaderModelOnly"
        }
        self.assertEqual(loaded_loras, {pipeline.QWEN_LORA: 1.0})
        self.assertNotIn(self.TEST_QWEN_STYLE_LORA, loaded_loras)
        self.assertEqual(workflow["14"]["inputs"]["model"], ["13", 0])

    def test_production_grooming_attempt4_uses_neutralized_action_pose_seed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workflow, uploaded_input, pose_input = self._capture_action_workflow(
                Path(temporary),
                action="grooming",
                action_identity_policy=(pipeline.POSE_IDENTITY_INPUT_POLICY_CANONICAL),
                pose_attempt=4,
                grooming_gate_acceptable=False,
                identity_palette_guard=True,
            )

        load_images = {
            node_id: node["inputs"]["image"]
            for node_id, node in workflow.items()
            if node.get("class_type") == "LoadImage"
        }
        expected_canonical = "local_video_api/capture-grooming/cartoon_reference.png"
        pose_transfer_seed = (
            "local_video_api/capture-grooming/"
            "grooming_canonical_pose_transfer_seed.png"
        )
        self.assertEqual(
            set(load_images.values()),
            {pose_transfer_seed, expected_canonical},
        )
        self.assertEqual(load_images["1"], pose_transfer_seed)
        self.assertNotIn(pose_input, load_images.values())
        self.assertNotIn(uploaded_input, load_images.values())
        prompt = workflow["6"]["inputs"]["prompt"]
        self.assertIn("CRITICAL ACTION-POSE IDENTITY TRANSFER", prompt)
        self.assertIn(
            "Image 1 is the sampler canvas and a deterministic high-key anatomy maquette",
            prompt,
        )
        self.assertIn(
            "Image 2 is the approved Qwen full-body canonical",
            prompt,
        )
        self.assertIn(
            "There is deliberately no third reference image",
            prompt,
        )
        self.assertIn("already consolidates the uploaded front, left, and right", prompt)
        self.assertIn("viewer-left shoulder-to-elbow-to-wrist-to-paw", prompt)
        self.assertIn("Do not replace it with a hanging item", prompt)
        self.assertEqual(
            workflow["14"]["inputs"]["denoise"],
            pipeline.GROOMING_CANONICAL_FULL_POSE_RESCUE_DENOISE,
        )
        self.assertEqual(
            workflow["14"]["inputs"]["steps"],
            pipeline.GROOMING_CANONICAL_FULL_POSE_RESCUE_STEPS,
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

    def test_production_grooming_attempt4_never_uses_rejected_pose_candidate_as_sampler(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workflow, uploaded_input, pose_input = self._capture_action_workflow(
                Path(temporary),
                action="grooming",
                action_identity_policy=(pipeline.POSE_IDENTITY_INPUT_POLICY_CANONICAL),
                pose_attempt=4,
                grooming_gate_acceptable=True,
                identity_palette_guard=True,
                canonical_action_palette_drift_attempts={1, 2, 3},
            )

        load_images = {
            node_id: node["inputs"]["image"]
            for node_id, node in workflow.items()
            if node.get("class_type") == "LoadImage"
        }
        expected_canonical = "local_video_api/capture-grooming/cartoon_reference.png"
        pose_transfer_seed = (
            "local_video_api/capture-grooming/"
            "grooming_canonical_pose_transfer_seed.png"
        )
        self.assertEqual(
            set(load_images.values()),
            {pose_transfer_seed, expected_canonical},
        )
        self.assertEqual(load_images["1"], pose_transfer_seed)
        self.assertNotIn(pose_input, load_images.values())
        self.assertNotIn(uploaded_input, load_images.values())
        self.assertFalse(
            any(
                str(value).endswith("grooming_full_pose_identity_refine_source.png")
                for value in load_images.values()
            )
        )
        prompt = workflow["6"]["inputs"]["prompt"]
        self.assertIn("CRITICAL ACTION-POSE IDENTITY TRANSFER", prompt)
        self.assertIn(
            "original RGB, chroma, raw luma, low-frequency markings, face detail, lighting, and fur texture were removed",
            prompt,
        )
        self.assertIn(
            "Image 2 is the approved Qwen full-body canonical",
            prompt,
        )
        self.assertIn(
            "There is deliberately no third reference image",
            prompt,
        )
        self.assertEqual(
            workflow["14"]["inputs"]["denoise"],
            pipeline.GROOMING_CANONICAL_FULL_POSE_REFINE_DENOISE,
        )
        self.assertFalse(
            any(
                node.get("class_type") == "SetLatentNoiseMask"
                for node in workflow.values()
            )
        )

    def test_production_grooming_attempt5_reuses_digest_bound_action_pose_seed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workflow, uploaded_input, pose_input = self._capture_action_workflow(
                Path(temporary),
                action="grooming",
                action_identity_policy=(pipeline.POSE_IDENTITY_INPUT_POLICY_CANONICAL),
                pose_attempt=5,
                grooming_gate_acceptable=True,
                identity_palette_guard=True,
                canonical_action_palette_drift_attempts=set(range(1, 9)),
                pose_safe_grooming_attempt4=True,
            )

        load_images = {
            node_id: node["inputs"]["image"]
            for node_id, node in workflow.items()
            if node.get("class_type") == "LoadImage"
        }
        expected_canonical = "local_video_api/capture-grooming/cartoon_reference.png"
        pose_transfer_seed = (
            "local_video_api/capture-grooming/"
            "grooming_canonical_pose_transfer_seed.png"
        )
        refine_source = (
            "local_video_api/capture-grooming/"
            "grooming_full_pose_identity_refine_source.png"
        )
        self.assertEqual(
            set(load_images.values()),
            {refine_source, expected_canonical, pose_transfer_seed},
        )
        self.assertEqual(load_images["1"], refine_source)
        self.assertNotIn(pose_input, load_images.values())
        self.assertNotIn(uploaded_input, load_images.values())
        prompt = workflow["6"]["inputs"]["prompt"]
        self.assertIn("CRITICAL POSE-SAFE IDENTITY REFINE", prompt)
        self.assertIn(
            "normalized grooming candidate from attempt 4",
            prompt,
        )
        self.assertIn(
            "Image 2 is the approved Qwen full-body canonical",
            prompt,
        )
        self.assertIn(
            "Image 3 is the deterministic high-key action-anatomy seed",
            prompt,
        )
        self.assertIn("passed the independent complete-subject check", prompt)
        self.assertIn("Do not return to a neutral two-paws-down sit", prompt)
        self.assertEqual(
            workflow["14"]["inputs"]["denoise"],
            pipeline.GROOMING_CANONICAL_FULL_POSE_REFINE_DENOISE,
        )
        self.assertEqual(
            workflow["14"]["inputs"]["steps"],
            pipeline.GROOMING_CANONICAL_FULL_POSE_RESCUE_STEPS,
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

    def test_grooming_repair_inputs_seed_canonical_foreleg_and_bind_anatomy(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "candidate.png"
            geometry = root / "geometry.png"
            canonical = root / "canonical.png"
            anatomy = root / "anatomy.png"
            repair = root / "repair.png"
            repair_noise_mask = root / "repair_noise_mask.png"
            frozen_anatomy = root / "frozen-anatomy.png"

            candidate_image = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
            candidate_image.paste((90, 100, 110, 255), (112, 72, 400, 440))
            candidate_image.save(candidate)
            canonical_image = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
            canonical_image.paste(
                (210, 220, 230, 255),
                (120, 72, 392, 440),
            )
            canonical_image.save(canonical)
            geometry_image = Image.new("RGB", (512, 512), (0, 255, 0))
            geometry_image.paste(
                pipeline.GROOMING_QWEN_GUIDE_FOREGROUND_RGB,
                (112, 72, 400, 440),
            )
            geometry_image.paste(
                pipeline.GROOMING_QWEN_GUIDE_FORELEG_RGB,
                (180, 220, 210, 320),
            )
            geometry_image.save(geometry)
            anatomy_image = Image.new("RGB", (512, 512), (0, 255, 0))
            anatomy_image.paste(
                pipeline.GROOMING_QWEN_GUIDE_FOREGROUND_RGB,
                (112, 72, 400, 440),
            )
            ImageDraw.Draw(anatomy_image).line(
                [(208, 245), (218, 275), (216, 310), (204, 335)],
                fill=(0, 255, 0),
                width=9,
            )
            anatomy_image.save(anatomy)

            repair_audit = pipeline.prepare_grooming_foreleg_repair_canvas(
                candidate,
                geometry,
                canonical,
                repair,
                noise_mask_target=repair_noise_mask,
            )
            anatomy_audit = pipeline.prepare_grooming_foreleg_repair_anatomy_guide(
                anatomy,
                frozen_anatomy,
            )

            with Image.open(repair) as loaded:
                repaired = loaded.convert("RGB")
            self.assertEqual(repaired.getpixel((195, 270)), (210, 220, 230))
            self.assertEqual(repaired.getpixel((130, 150)), (90, 100, 110))
            self.assertEqual(repair_audit["edit_window_pure_green_pixels"], 0)
            self.assertTrue(repair_audit["edit_window_identity_texture_seeded"])
            self.assertEqual(
                repair_audit["canonical_reference_sha256"],
                pipeline.file_sha256(canonical),
            )
            self.assertGreater(
                repair_audit["canonical_seed_direct_pixels"],
                256,
            )
            self.assertTrue(repair_audit["outside_edit_window_byte_preserved"])
            self.assertFalse(
                repair_audit["canonical_seed_high_frequency_texture_exposed"]
            )
            self.assertEqual(
                repair_audit["canonical_seed_lowpass_sigma_px"],
                pipeline.GROOMING_QWEN_FORELEG_REPAIR_SEED_LOWPASS_SIGMA_PX,
            )
            self.assertEqual(
                repair_audit["canonical_seed_identity_median_blend"],
                pipeline.GROOMING_QWEN_FORELEG_REPAIR_SEED_IDENTITY_MEDIAN_BLEND,
            )
            self.assertEqual(
                repair_audit["noise_mask_target_sha256"],
                pipeline.file_sha256(repair_noise_mask),
            )
            self.assertEqual(
                repair_audit["noise_mask_loadimage_contract"],
                "mask_equals_one_minus_alpha",
            )
            self.assertGreater(
                repair_audit["noise_mask_full_strength_pixels"],
                repair_audit["semantic_foreleg_pixels"],
            )
            self.assertEqual(
                repair_audit["noise_mask_core_expansion_px"],
                pipeline.GROOMING_QWEN_FORELEG_REPAIR_NOISE_MASK_EXPANSION_PX,
            )
            self.assertGreater(
                repair_audit["noise_mask_core_pixels"],
                repair_audit["edit_mask_pixels"],
            )
            self.assertGreater(repair_audit["noise_mask_feather_pixels"], 0)
            with Image.open(repair_noise_mask) as loaded:
                repair_mask_rgba = loaded.convert("RGBA")
            self.assertEqual(repair_mask_rgba.getpixel((195, 270))[3], 0)
            self.assertEqual(repair_mask_rgba.getpixel((20, 20))[3], 255)
            self.assertGreater(repair_audit["removed_foreground_pixels"], 256)
            self.assertRegex(repair_audit["edit_mask_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(
                anatomy_audit["policy"],
                pipeline.GROOMING_QWEN_FORELEG_REPAIR_ANATOMY_POLICY,
            )
            self.assertEqual(
                anatomy_audit["source_sha256"],
                anatomy_audit["target_sha256"],
            )
            self.assertEqual(
                anatomy_audit["role"],
                "image3_single_tone_open_channel_action_geometry_only",
            )
            self.assertGreaterEqual(
                anatomy_audit["internal_open_channel_pixels"],
                300,
            )

    def test_canonical_grooming_completion_canvas_preserves_source_pixels(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "candidate.png"
            completion_mask = root / "completion-mask.png"
            canonical = root / "canonical.png"
            repair = root / "repair.png"
            noise_mask = root / "repair-noise-mask.png"

            candidate_image = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
            candidate_image.paste((90, 100, 110, 255), (112, 72, 400, 440))
            candidate_image.save(candidate)
            canonical_image = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
            canonical_image.paste((210, 220, 230, 255), (120, 72, 392, 440))
            canonical_image.save(canonical)
            mask_image = Image.new("RGB", (512, 512), (0, 255, 0))
            mask_image.paste(
                pipeline.GROOMING_QWEN_GUIDE_FOREGROUND_RGB,
                (112, 72, 400, 440),
            )
            mask_image.paste(
                pipeline.GROOMING_QWEN_GUIDE_FORELEG_RGB,
                (180, 220, 220, 340),
            )
            mask_image.save(completion_mask)

            audit = pipeline.prepare_canonical_grooming_foreleg_completion_canvas(
                candidate,
                completion_mask,
                canonical,
                repair,
                noise_mask_target=noise_mask,
            )

            with Image.open(repair) as loaded:
                repaired = loaded.convert("RGB")
            self.assertEqual(repaired.getpixel((195, 270)), (90, 100, 110))
            self.assertEqual(repaired.getpixel((130, 150)), (90, 100, 110))
            self.assertEqual(repaired.getpixel((20, 20)), (0, 255, 0))
            self.assertEqual(
                audit["policy"],
                pipeline.GROOMING_CANONICAL_FORELEG_COMPLETION_POLICY,
            )
            self.assertFalse(audit["semantic_pixels_exposed_to_qwen"])
            self.assertFalse(audit["identity_or_geometry_seed_inserted"])
            self.assertTrue(audit["source_composite_preserved_before_masked_sampling"])
            self.assertEqual(audit["semantic_foreleg_pixels"], 4800)
            self.assertGreater(audit["intersected_source_foreground_pixels"], 256)
            self.assertEqual(
                audit["noise_mask_target_sha256"],
                pipeline.file_sha256(noise_mask),
            )
            with Image.open(noise_mask) as loaded:
                mask_rgba = loaded.convert("RGBA")
            self.assertEqual(mask_rgba.getpixel((195, 270))[3], 0)
            self.assertEqual(mask_rgba.getpixel((20, 20))[3], 255)

    def test_full_pose_recovery_guide_exposes_only_public_open_channel(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            public = root / "public.png"
            private = root / "private.png"
            semantic = root / "semantic.png"
            canonical = root / "canonical.png"
            target = root / "recovery.png"

            private_array = np.full((512, 512, 3), (0, 255, 0), dtype=np.uint8)
            private_array[80:440, 112:400] = (96, 96, 96)
            private_array[80:260, 112:256] = (176, 176, 176)
            private_array[260:360, 112:256] = (136, 136, 136)
            Image.fromarray(private_array, mode="RGB").save(private)

            public_image = Image.new("RGB", (512, 512), (0, 255, 0))
            public_image.paste(
                pipeline.GROOMING_QWEN_GUIDE_FOREGROUND_RGB,
                (112, 80, 400, 440),
            )
            ImageDraw.Draw(public_image).line(
                [(208, 245), (218, 275), (216, 310), (204, 335)],
                fill=(0, 255, 0),
                width=9,
            )
            public_image.save(public)

            semantic_array = np.full(
                (512, 512, 3),
                (0, 255, 0),
                dtype=np.uint8,
            )
            semantic_array[80:440, 112:400] = (
                pipeline.GROOMING_QWEN_GUIDE_FOREGROUND_RGB
            )
            semantic_array[220:300, 160:220] = pipeline.GROOMING_QWEN_GUIDE_FORELEG_RGB
            Image.fromarray(semantic_array, mode="RGB").save(semantic)

            canonical_array = np.full(
                (512, 512, 3),
                (0, 255, 0),
                dtype=np.uint8,
            )
            canonical_array[72:440, 120:392] = (208, 208, 208)
            Image.fromarray(canonical_array, mode="RGB").save(canonical)

            audit = pipeline.prepare_grooming_full_pose_recovery_guide(
                public,
                private,
                semantic,
                canonical,
                target,
            )

            with Image.open(target) as loaded:
                output = np.asarray(loaded.convert("RGB"), dtype=np.uint8)
            self.assertEqual(output[20, 20].tolist(), [0, 255, 0])
            self.assertEqual(output[120, 300].tolist(), [144, 144, 144])
            self.assertEqual(output[275, 218].tolist(), [0, 255, 0])
            self.assertTrue(np.array_equal(output, np.asarray(public_image)))
            self.assertEqual(
                audit["policy"],
                pipeline.GROOMING_CANONICAL_FULL_POSE_GUIDE_POLICY,
            )
            self.assertEqual(audit["semantic_foreleg_pixels"], 4800)
            self.assertFalse(audit["semantic_foreleg_exposed_as_discrete_label"])
            self.assertTrue(audit["open_curve_exposed"])
            self.assertTrue(audit["open_curve_is_background_negative_space"])
            self.assertFalse(audit["private_luma_exposed"])
            self.assertFalse(audit["private_raw_luma_values_preserved"])
            self.assertEqual(audit["unique_foreground_rgb_count"], 1)
            self.assertGreaterEqual(audit["internal_open_channel_pixels"], 300)
            self.assertFalse(audit["spatial_canonical_pixels_copied"])
            self.assertFalse(audit["main_case_rgb_or_luma_preserved"])

    def test_full_pose_refine_canvas_preserves_rgba_and_composites_green(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "rescue-cutout.png"
            target = root / "refine-canvas.png"
            source_image = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
            source_image.paste(
                (210, 220, 230, 255),
                (120, 72, 392, 440),
            )
            source_image.save(source)

            audit = pipeline.prepare_canonical_grooming_full_pose_refine_canvas(
                source,
                target,
            )

            with Image.open(target) as loaded:
                canvas = loaded.convert("RGB")
            self.assertEqual(canvas.getpixel((20, 20)), (0, 255, 0))
            self.assertEqual(canvas.getpixel((256, 256)), (210, 220, 230))
            self.assertEqual(
                audit["policy"],
                pipeline.GROOMING_CANONICAL_FULL_POSE_REFINE_POLICY,
            )
            self.assertFalse(audit["source_rgba_pixels_modified_before_composite"])
            self.assertTrue(audit["transparent_background_composited_to_pure_green"])
            self.assertEqual(audit["source_sha256"], pipeline.file_sha256(source))
            self.assertEqual(audit["target_sha256"], pipeline.file_sha256(target))
            self.assertEqual(audit["source_foreground_bbox"], [120, 72, 392, 440])

    def test_canonical_grooming_face_reference_is_derived_from_same_identity(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical = root / "canonical.png"
            target = root / "canonical-face.png"
            image = Image.new("RGB", (512, 512), (0, 255, 0))
            draw = ImageDraw.Draw(image)
            draw.ellipse((126, 58, 386, 452), fill=(232, 232, 232))
            draw.ellipse((164, 84, 348, 250), fill=(244, 244, 244))
            draw.ellipse((205, 145, 225, 170), fill=(30, 120, 45))
            draw.ellipse((285, 145, 305, 170), fill=(30, 120, 45))
            image.save(canonical)

            audit = pipeline.prepare_canonical_grooming_face_identity_reference(
                canonical,
                target,
            )

            with Image.open(target) as loaded:
                face = loaded.convert("RGB")
            self.assertEqual(face.size, (512, 512))
            self.assertEqual(face.getpixel((0, 0)), (0, 255, 0))
            self.assertNotEqual(face.getpixel((256, 256)), (0, 255, 0))
            self.assertTrue(audit["derived_from_same_canonical_identity"])
            self.assertTrue(audit["face_detail_authority"])
            self.assertFalse(audit["pose_authority"])
            self.assertEqual(
                audit["canonical_reference_sha256"],
                pipeline.file_sha256(canonical),
            )

    def test_grooming_pose_transfer_seed_keeps_alpha_geometry_but_removes_appearance(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            action = root / "grooming-action.png"
            action_inverse = root / "grooming-action-inverse.png"
            canonical = root / "canonical.png"
            geometry = root / "grooming-geometry.png"
            foreleg = root / "grooming-foreleg.png"
            target = root / "pose-transfer-seed.png"
            inverse_target = root / "pose-transfer-seed-inverse.png"

            action_mask = np.zeros((512, 512), dtype=np.uint8)
            cv2.ellipse(action_mask, (270, 300), (120, 150), 0, 0, 360, 255, -1)
            cv2.line(action_mask, (210, 330), (190, 235), 255, 42)
            cv2.circle(action_mask, (190, 225), 24, 255, -1)
            action_array = np.zeros((512, 512, 4), dtype=np.uint8)
            yy, xx = np.indices((512, 512))
            checker = (((xx // 24) + (yy // 24)) % 2) == 0
            foreground = action_mask > 0
            action_array[:, :, 3] = action_mask
            action_array[foreground & checker, :3] = (245, 40, 30)
            action_array[foreground & ~checker, :3] = (20, 50, 235)
            Image.fromarray(action_array, mode="RGBA").save(action)
            inverse_action_array = action_array.copy()
            inverse_action_array[foreground & checker, :3] = (10, 245, 220)
            inverse_action_array[foreground & ~checker, :3] = (250, 235, 15)
            Image.fromarray(inverse_action_array, mode="RGBA").save(action_inverse)

            geometry_array = np.full(
                (512, 512, 3),
                (0, 255, 0),
                dtype=np.uint8,
            )
            geometry_foreground = foreground.copy()
            geometry_foreground[240:310, 250:270] = False
            geometry_array[geometry_foreground] = (144, 144, 144)
            Image.fromarray(geometry_array, mode="RGB").save(geometry)

            foreleg_array = np.full(
                (512, 512, 3),
                (0, 255, 0),
                dtype=np.uint8,
            )
            foreleg_array[geometry_foreground] = (144, 144, 144)
            semantic_foreleg = np.zeros((512, 512), dtype=np.uint8)
            cv2.line(semantic_foreleg, (245, 320), (245, 220), 255, 26)
            cv2.circle(semantic_foreleg, (245, 215), 18, 255, -1)
            semantic_foreleg_mask = (semantic_foreleg > 0) & foreground
            foreleg_array[semantic_foreleg_mask] = (176, 176, 176)
            Image.fromarray(foreleg_array, mode="RGB").save(foreleg)

            canonical_array = np.full(
                (512, 512, 3),
                (0, 255, 0),
                dtype=np.uint8,
            )
            canonical_array[72:448, 128:392] = (216, 222, 230)
            canonical_array[150:210, 184:336] = (176, 184, 196)
            Image.fromarray(canonical_array, mode="RGB").save(canonical)

            audit = pipeline.prepare_canonical_grooming_pose_transfer_seed(
                action,
                canonical,
                target,
                geometry_guide=geometry,
                foreleg_guide=foreleg,
            )
            inverse_audit = pipeline.prepare_canonical_grooming_pose_transfer_seed(
                action_inverse,
                canonical,
                inverse_target,
                geometry_guide=geometry,
                foreleg_guide=foreleg,
            )

            with Image.open(target) as loaded:
                output = np.asarray(loaded.convert("RGB"), dtype=np.uint8)
            self.assertTrue(np.all(output[~foreground] == (0, 255, 0)))
            self.assertFalse(
                np.any(np.all(output[foreground] == (245, 40, 30), axis=1))
            )
            self.assertFalse(
                np.any(np.all(output[foreground] == (20, 50, 235), axis=1))
            )
            self.assertEqual(
                audit["policy"],
                pipeline.GROOMING_CANONICAL_POSE_TRANSFER_SEED_POLICY,
            )
            self.assertFalse(audit["source_alpha_geometry_preserved"])
            self.assertFalse(audit["action_alpha_overlap_blocking"])
            self.assertFalse(audit["action_reference_used_for_conditioning"])
            self.assertGreater(audit["public_internal_channel_pixels_filled"], 0)
            self.assertEqual(audit["qwen_visible_internal_green_channel_pixels"], 0)
            self.assertFalse(audit["source_rgb_preserved"])
            self.assertFalse(audit["source_chroma_preserved"])
            self.assertFalse(audit["source_raw_luma_values_preserved"])
            self.assertFalse(audit["source_low_frequency_luma_exposed"])
            self.assertFalse(audit["source_luma_used_for_shading"])
            self.assertFalse(audit["source_high_frequency_texture_exposed"])
            self.assertFalse(audit["geometry_distance_field_only"])
            self.assertTrue(audit["geometry_and_soft_foreleg_depth_only"])
            self.assertTrue(audit["soft_foreleg_depth_semantic_exposed"])
            self.assertFalse(audit["semantic_foreleg_discrete_color_exposed"])
            self.assertFalse(audit["public_open_channel_geometry_exposed"])
            self.assertTrue(audit["canonical_palette_conditioning_only"])
            self.assertLessEqual(audit["target_luma_range"], 24.0)
            self.assertEqual(audit["geometry_guide_unique_foreground_rgb_count"], 1)
            self.assertEqual(target.read_bytes(), inverse_target.read_bytes())
            self.assertNotEqual(
                audit["action_reference_sha256"],
                inverse_audit["action_reference_sha256"],
            )
            self.assertEqual(audit["target_sha256"], pipeline.file_sha256(target))

            depth_geometry = root / "grooming-white-depth.png"
            depth_target = root / "pose-transfer-white-depth.png"
            depth_inverse_target = root / "pose-transfer-white-depth-inverse.png"
            depth_geometry_array = np.full(
                (512, 512, 3),
                (0, 255, 0),
                dtype=np.uint8,
            )
            depth_distance = cv2.distanceTransform(
                geometry_foreground.astype(np.uint8),
                cv2.DIST_L2,
                5,
            )
            depth_values = 104.0 + 72.0 * np.clip(
                depth_distance / max(float(depth_distance.max()), 1.0),
                0.0,
                1.0,
            )
            for channel in range(3):
                depth_geometry_array[:, :, channel][geometry_foreground] = (
                    np.rint(depth_values[geometry_foreground]).astype(np.uint8)
                )
            Image.fromarray(depth_geometry_array, mode="RGB").save(depth_geometry)

            depth_audit = pipeline.prepare_canonical_grooming_pose_transfer_seed(
                action,
                canonical,
                depth_target,
                geometry_guide=depth_geometry,
                foreleg_guide=foreleg,
                preserve_normalized_geometry_luma=True,
            )
            pipeline.prepare_canonical_grooming_pose_transfer_seed(
                action_inverse,
                canonical,
                depth_inverse_target,
                geometry_guide=depth_geometry,
                foreleg_guide=foreleg,
                preserve_normalized_geometry_luma=True,
            )

            self.assertTrue(depth_audit["normalized_geometry_luma_exposed"])
            self.assertTrue(
                depth_audit["source_normalized_low_frequency_geometry_exposed"]
            )
            self.assertTrue(depth_audit["normalized_geometry_luma_used_for_shading"])
            self.assertTrue(
                depth_audit["normalized_white_maquette_depth_anatomy_only"]
            )
            self.assertFalse(depth_audit["soft_foreleg_depth_semantic_exposed"])
            self.assertEqual(depth_audit["foreleg_luma_lift"], 0.0)
            self.assertGreater(
                depth_audit["geometry_guide_unique_foreground_rgb_count"],
                2,
            )
            self.assertGreaterEqual(depth_audit["target_luma_range"], 24)
            self.assertEqual(
                depth_audit["sampler_role"],
                "canonical_palette_normalized_white_maquette_depth_anatomy",
            )
            self.assertEqual(
                depth_target.read_bytes(),
                depth_inverse_target.read_bytes(),
            )
            self.assertNotEqual(target.read_bytes(), depth_target.read_bytes())

            frame0_geometry = root / "grooming-frame000-body-pose.png"
            frame0_target = root / "pose-transfer-frame000.png"
            frame0_array = np.full(
                (512, 512, 3),
                (0, 255, 0),
                dtype=np.uint8,
            )
            frame0_distance = cv2.distanceTransform(
                foreground.astype(np.uint8),
                cv2.DIST_L2,
                5,
            )
            frame0_values = 96.0 + 80.0 * np.clip(
                frame0_distance / max(float(frame0_distance.max()), 1.0),
                0.0,
                1.0,
            )
            for channel in range(3):
                frame0_array[:, :, channel][foreground] = np.rint(
                    frame0_values[foreground]
                ).astype(np.uint8)
            Image.fromarray(frame0_array, mode="RGB").save(frame0_geometry)

            frame0_audit = pipeline.prepare_canonical_grooming_pose_transfer_seed(
                action,
                canonical,
                frame0_target,
                geometry_guide=frame0_geometry,
                foreleg_guide=None,
                preserve_normalized_geometry_luma=True,
                complete_geometry_guide=True,
                remove_head_identity_silhouette=True,
            )

            self.assertTrue(frame0_audit["complete_geometry_guide"])
            self.assertIsNone(frame0_audit["foreleg_guide_path"])
            self.assertEqual(frame0_audit["semantic_foreleg_pixels"], 0)
            self.assertEqual(frame0_audit["public_internal_channel_pixels_filled"], 0)
            self.assertEqual(
                frame0_audit["sampler_role"],
                "canonical_palette_headless_white_maquette_frame000_body_pose_anatomy",
            )
            self.assertTrue(frame0_audit["head_identity_silhouette_removed"])
            self.assertGreater(
                frame0_audit["head_identity_silhouette_removed_pixels"],
                4096,
            )
            self.assertGreater(
                frame0_audit["qwen_visible_internal_green_channel_pixels"],
                4096,
            )
            self.assertGreater(
                frame0_audit["raised_paw_geometry_retained_pixels"],
                512,
            )
            self.assertEqual(
                frame0_audit["raised_paw_geometry_policy"],
                "canonical_muzzle_aligned_tapered_neutral_foreleg_v1",
            )
            foreleg_audit = frame0_audit["raised_paw_geometry_audit"]
            self.assertLess(
                foreleg_audit["paw_center"][1],
                foreleg_audit["head_cutoff_y"],
            )
            self.assertGreater(
                foreleg_audit["shoulder_anchor"][1],
                foreleg_audit["head_cutoff_y"],
            )
            self.assertGreater(
                foreleg_audit["pixels_above_head_cutoff"],
                256,
            )
            self.assertGreater(
                foreleg_audit["pixels_below_head_cutoff"],
                256,
            )
            self.assertFalse(foreleg_audit["source_rgb_or_luma_used"])
            self.assertFalse(frame0_audit["soft_foreleg_depth_semantic_exposed"])

            full_pose_target = root / "pose-transfer-frame000-full-pose.png"
            full_pose_audit = (
                pipeline.prepare_canonical_grooming_pose_transfer_seed(
                    action,
                    canonical,
                    full_pose_target,
                    geometry_guide=frame0_geometry,
                    foreleg_guide=None,
                    preserve_normalized_geometry_luma=True,
                    complete_geometry_guide=True,
                    remove_head_identity_silhouette=False,
                )
            )
            self.assertFalse(full_pose_audit["head_identity_silhouette_removed"])
            self.assertEqual(
                full_pose_audit["head_identity_silhouette_removed_pixels"],
                0,
            )
            self.assertEqual(
                full_pose_audit["qwen_visible_internal_green_channel_pixels"],
                0,
            )
            self.assertEqual(
                full_pose_audit["qwen_visible_geometry_foreground_bbox"],
                full_pose_audit["geometry_foreground_bbox"],
            )
            self.assertEqual(
                full_pose_audit["raised_paw_geometry_policy"],
                "canonical_muzzle_aligned_tapered_neutral_foreleg_v1",
            )
            self.assertFalse(
                full_pose_audit["raised_paw_volume_source_rgb_or_luma_used"]
            )
            self.assertEqual(
                full_pose_audit["sampler_role"],
                "canonical_palette_white_maquette_frame000_body_pose_anatomy",
            )

            action_edit_target = root / "grooming-local-action-edit-mask.png"
            action_edit_audit = (
                pipeline.prepare_advisory_grooming_local_action_edit_mask(
                    canonical,
                    action_edit_target,
                )
            )
            with Image.open(action_edit_target) as loaded:
                action_edit_alpha = np.asarray(
                    loaded.convert("RGBA"),
                    dtype=np.uint8,
                )[:, :, 3]
            self.assertGreater(int(action_edit_alpha[96, 256]), 250)
            self.assertLess(int(action_edit_alpha[400, 256]), 64)
            self.assertFalse(
                action_edit_audit["post_qwen_pixel_composite_enabled"]
            )
            self.assertTrue(action_edit_audit["raw_qwen_output_required"])
            self.assertTrue(action_edit_audit["prompt_only_action_geometry"])
            self.assertFalse(action_edit_audit["video_first_frame_loaded_by_qwen"])
            self.assertEqual(
                action_edit_audit["noise_mask_loadimage_contract"],
                "mask_equals_one_minus_alpha",
            )

    def test_grooming_seed_bound_matte_unions_alpha_without_copying_seed_rgb(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "qwen-source.png"
            semantic = root / "semantic.png"
            seed = root / "pose-seed.png"
            target = root / "seed-bound.png"
            green = root / "seed-bound-green.png"

            seed_mask = np.zeros((512, 512), dtype=np.uint8)
            cv2.ellipse(seed_mask, (280, 310), (120, 165), 0, 0, 360, 255, -1)
            cv2.line(seed_mask, (220, 345), (194, 235), 255, 46)
            cv2.circle(seed_mask, (194, 224), 24, 255, -1)
            seed_array = np.full((512, 512, 3), (0, 255, 0), dtype=np.uint8)
            seed_array[seed_mask > 0] = (212, 214, 218)
            Image.fromarray(seed_array, mode="RGB").save(seed)

            source_array = np.full((1024, 1024, 3), (171, 119, 83), dtype=np.uint8)
            source_array[:, :, 0] += np.arange(1024, dtype=np.uint8)[None, :] // 16
            Image.fromarray(source_array, mode="RGB").save(source)
            resized_seed = np.asarray(
                Image.fromarray(seed_mask, mode="L").resize(
                    (1024, 1024),
                    Image.Resampling.NEAREST,
                ),
                dtype=np.uint8,
            )
            semantic_mask = (resized_seed > 0) & (
                np.indices((1024, 1024))[1] < 690
            )
            semantic_array = np.zeros((1024, 1024, 4), dtype=np.uint8)
            semantic_array[semantic_mask, :3] = (12, 34, 220)
            semantic_array[semantic_mask, 3] = 255
            Image.fromarray(semantic_array, mode="RGBA").save(semantic)

            audit = pipeline.prepare_canonical_grooming_seed_bound_matte(
                source,
                semantic,
                seed,
                target,
                green_target=green,
            )

            with Image.open(target) as loaded:
                output = np.asarray(loaded.convert("RGBA"), dtype=np.uint8)
            expected_union = resized_seed > 0
            self.assertTrue(np.array_equal(output[:, :, 3] >= 96, expected_union))
            self.assertTrue(
                np.array_equal(
                    output[:, :, :3][expected_union],
                    source_array[expected_union],
                )
            )
            self.assertEqual(
                audit["policy"],
                pipeline.GROOMING_CANONICAL_SEED_BOUND_MATTE_POLICY,
            )
            self.assertTrue(audit["acceptable"])
            self.assertTrue(all(audit["checks"].values()))
            self.assertTrue(audit["source_rgb_preserved_where_alpha_nonzero"])
            self.assertFalse(audit["pose_transfer_seed_rgb_copied"])
            self.assertTrue(audit["pose_transfer_seed_used_for_alpha_geometry_only"])
            self.assertGreater(audit["seed_added_pixels"], 0)
            self.assertEqual(audit["target_sha256"], pipeline.file_sha256(target))
            self.assertTrue(green.is_file())

    def test_grooming_seed_bound_matte_rejects_shifted_semantic_subject(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "qwen-source.png"
            semantic = root / "semantic.png"
            seed = root / "pose-seed.png"
            target = root / "seed-bound.png"
            Image.new("RGB", (1024, 1024), (180, 140, 90)).save(source)
            seed_array = np.full((512, 512, 3), (0, 255, 0), dtype=np.uint8)
            cv2.ellipse(
                seed_array,
                (360, 350),
                (90, 130),
                0,
                0,
                360,
                (210, 210, 210),
                -1,
            )
            Image.fromarray(seed_array, mode="RGB").save(seed)
            semantic_array = np.zeros((1024, 1024, 4), dtype=np.uint8)
            cv2.ellipse(
                semantic_array,
                (220, 250),
                (170, 210),
                0,
                0,
                360,
                (40, 50, 60, 255),
                -1,
            )
            Image.fromarray(semantic_array, mode="RGBA").save(semantic)

            with self.assertRaisesRegex(ValueError, "matte_alignment_failed"):
                pipeline.prepare_canonical_grooming_seed_bound_matte(
                    source,
                    semantic,
                    seed,
                    target,
                )

    def test_pose_safe_identity_refine_eligibility_requires_all_independent_gates(
        self,
    ) -> None:
        matte = {
            "policy": pipeline.GROOMING_CANONICAL_SEED_BOUND_MATTE_POLICY,
            "acceptable": True,
            "fail_closed": True,
            "source_sha256": "a" * 64,
            "semantic_cutout_sha256": "b" * 64,
            "pose_transfer_seed_sha256": "c" * 64,
            "target_sha256": "d" * 64,
            "union_alpha_sha256": "e" * 64,
            "checks": {"alignment": True, "area": True},
            "pose_transfer_seed_used_for_alpha_geometry_only": True,
            "pose_transfer_seed_rgb_copied": False,
            "pose_transfer_seed_luma_or_chroma_copied": False,
        }
        pose_gate = {
            "available": True,
            "evaluated": True,
            "acceptable": True,
            "issues": [],
            "metrics": {
                "raised_foreleg_structural_evidence": {
                    "available": True,
                    "edge_density": 0.024736,
                },
                "chamfer": {
                    "available": True,
                    "coverage_within_radius": 0.833573,
                    "mean_distance_px": 3.408131,
                    "p90_distance_px": 8.595,
                },
            },
        }
        pose_quality = {"acceptable": True, "issues": [], "penalty": 0.0}
        palette = {
            "available": True,
            "acceptable": False,
            "issues": [
                "new_dominant_warm_brown_or_orange",
                "achromatic_source_recolored_with_warm_fur",
                "canonical_action_achromatic_value_balance_drift",
            ],
        }
        surface = {
            "available": True,
            "evaluated": True,
            "acceptable": True,
            "issues": [],
            "gate": {
                "alpha_provenance": (
                    pipeline.GROOMING_CANONICAL_SEED_BOUND_MATTE_ALPHA_PROVENANCE
                )
            },
        }

        accepted = pipeline.canonical_grooming_pose_safe_identity_refine_eligibility(
            attempt=pipeline.GROOMING_CANONICAL_FULL_POSE_RESCUE_ATTEMPT,
            seed_bound_matte=matte,
            pose_gate=pose_gate,
            pose_quality=pose_quality,
            identity_palette_quality=palette,
            fur_surface_gate=surface,
        )
        self.assertTrue(accepted["eligible"])
        self.assertTrue(all(accepted["checks"].values()))
        self.assertFalse(accepted["identity_palette_gate_bypassed"])

        prop_palette = {
            **palette,
            "issues": ["raised_paw_material_inconsistent_with_planted_paw"],
        }
        rejected = pipeline.canonical_grooming_pose_safe_identity_refine_eligibility(
            attempt=pipeline.GROOMING_CANONICAL_FULL_POSE_RESCUE_ATTEMPT,
            seed_bound_matte=matte,
            pose_gate={
                **pose_gate,
                "metrics": {
                    **pose_gate["metrics"],
                    "raised_foreleg_structural_evidence": {
                        "available": True,
                        "edge_density": 0.009,
                    },
                },
            },
            pose_quality=pose_quality,
            identity_palette_quality=prop_palette,
            fur_surface_gate=surface,
        )
        self.assertFalse(rejected["eligible"])
        self.assertFalse(rejected["checks"]["organic_structural_density_passed"])
        self.assertFalse(
            rejected["checks"][
                "identity_gate_failed_only_for_repaintable_palette_drift"
            ]
        )

    def test_advisory_grooming_identity_refine_locks_body_and_raised_paw(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "accepted-grooming-action.png"
            canvas = root / "masked-refine-canvas.png"
            noise_mask = root / "masked-refine-noise.png"
            generated = root / "raw-qwen-refine.png"
            target = root / "locked-refine-result.png"

            source_image = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
            source_draw = ImageDraw.Draw(source_image)
            source_draw.ellipse(
                (112, 48, 400, 464),
                fill=(96, 104, 112, 255),
            )
            # A raised foreleg and paw deliberately sit below the head-edit cutoff.
            source_draw.line(
                [(190, 372), (178, 300), (198, 250)],
                fill=(210, 214, 218, 255),
                width=42,
            )
            source_draw.ellipse(
                (178, 232, 220, 270),
                fill=(220, 224, 228, 255),
            )
            source_draw.rectangle(
                (60, 300, 145, 440),
                fill=(4, 8, 6, 255),
            )
            source_image.save(source)
            Image.new("RGB", (1024, 1024), (210, 80, 96)).save(generated)

            canvas_audit = (
                pipeline.prepare_advisory_grooming_identity_refine_canvas(
                    source,
                    canvas,
                    noise_mask,
                    identity_reference_palette={
                        "fractions": {"black": 0.01},
                    },
                )
            )
            composite_audit = (
                pipeline.finalize_advisory_grooming_identity_refine_candidate(
                    Path(canvas_audit["sanitized_source_path"]),
                    generated,
                    noise_mask,
                    target,
                )
            )

            self.assertEqual(
                canvas_audit["sampler_canvas_scope"],
                "upper_head_mask_only_action_pixels_locked",
            )
            self.assertTrue(canvas_audit["raised_paw_editing_forbidden"])
            self.assertGreater(canvas_audit["protected_body_pixels"], 2048)
            self.assertEqual(
                canvas_audit["head_silhouette_expansion_px"],
                pipeline.GROOMING_ADVISORY_IDENTITY_REFINE_HEAD_EXPANSION_PX,
            )
            self.assertEqual(canvas_audit["editable_background_pixels"], 0)
            self.assertGreater(canvas_audit["dark_artifact_removed_pixels"], 2048)
            self.assertEqual(
                composite_audit["protected_action_pixels_changed"],
                0,
            )
            self.assertTrue(
                composite_audit["body_and_raised_paw_pixel_lock_passed"]
            )
            self.assertEqual(
                composite_audit["head_silhouette_expansion_pixels"],
                0,
            )
            with Image.open(target) as loaded:
                result = loaded.convert("RGB")
            self.assertEqual(result.getpixel((0, 0)), (0, 255, 0))
            self.assertEqual(result.getpixel((80, 350)), (0, 255, 0))
            self.assertEqual(result.getpixel((198, 250)), (220, 224, 228))
            self.assertEqual(result.getpixel((256, 380)), (96, 104, 112))
            self.assertEqual(result.getpixel((256, 100)), (210, 80, 96))

    def test_eating_identity_refine_canvas_uses_action_specific_policy(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "eating-cutout.png"
            target = root / "eating-refine.png"
            source_image = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
            source_image.paste(
                (224, 220, 210, 255),
                (96, 96, 416, 440),
            )
            source_image.save(source)

            audit = pipeline.prepare_canonical_eating_identity_refine_canvas(
                source,
                target,
            )

            self.assertEqual(audit["action"], "eating")
            self.assertEqual(
                audit["policy"],
                pipeline.EATING_CANONICAL_IDENTITY_REFINE_POLICY,
            )
            self.assertTrue(audit["transparent_background_composited_to_pure_green"])
            self.assertFalse(audit["source_rgba_pixels_modified_before_composite"])
            with Image.open(target) as loaded:
                canvas = loaded.convert("RGB")
            self.assertEqual(canvas.getpixel((20, 20)), (0, 255, 0))
            self.assertEqual(canvas.getpixel((256, 256)), (224, 220, 210))

    def test_two_image_qwen_workflow_keeps_loras_and_adds_pose_reference(self) -> None:
        loras = [
            {
                "model_name": "user_uploads/cat-photo-v3.safetensors",
                "original_filename": "宠物小猫 _ 写实摄影_v3.0.safetensors",
                "strength": 0.5,
            },
            {
                "model_name": "user_uploads/kitten-photo.safetensors",
                "original_filename": "Qwen Image-Kitten Photography_QWEN-小猫摄影.safetensors",
                "strength": 0.6,
            },
        ]
        workflow = pipeline.qwen_workflow(
            "subject.png",
            "positive",
            "negative",
            123,
            "test/output",
            user_loras=loras,
            user_lora_mode="replace",
            suppress_builtin_style=True,
            pose_reference_input_name="grooming-first-frame.png",
        )

        loaded_loras = [
            node["inputs"]
            for node in workflow.values()
            if node.get("class_type") == "LoraLoaderModelOnly"
            and node["inputs"].get("lora_name")
            in {item["model_name"] for item in loras}
        ]
        self.assertEqual(
            [(item["lora_name"], item["strength_model"]) for item in loaded_loras],
            [(loras[0]["model_name"], 0.5), (loras[1]["model_name"], 0.6)],
        )
        pose_load_nodes = [
            node_id
            for node_id, node in workflow.items()
            if node.get("class_type") == "LoadImage"
            and node["inputs"].get("image") == "grooming-first-frame.png"
        ]
        self.assertEqual(len(pose_load_nodes), 1)
        pose_scale_nodes = [
            node_id
            for node_id, node in workflow.items()
            if node.get("class_type") == "FluxKontextImageScale"
            and node["inputs"].get("image") == [pose_load_nodes[0], 0]
        ]
        self.assertEqual(len(pose_scale_nodes), 1)
        pose_link = [pose_scale_nodes[0], 0]
        canonical_link = ["5", 0]
        self.assertEqual(workflow["6"]["inputs"]["image1"], canonical_link)
        self.assertEqual(workflow["7"]["inputs"]["image1"], canonical_link)
        self.assertIn("image2", workflow["6"]["inputs"])
        self.assertEqual(
            workflow["6"]["inputs"]["image2"], workflow["7"]["inputs"]["image2"]
        )
        self.assertEqual(workflow["6"]["inputs"]["image2"], pose_link)
        self.assertNotIn("image3", workflow["6"]["inputs"])
        self.assertNotIn("image3", workflow["7"]["inputs"])
        self.assertEqual(workflow["10"]["inputs"]["pixels"], canonical_link)
        self.assertEqual(workflow["14"]["inputs"]["denoise"], 1.0)

    def test_three_view_qwen_workflow_wires_image1_image2_image3_only_for_canonical(
        self,
    ) -> None:
        workflow = pipeline.qwen_workflow(
            "pet-front.png",
            "positive",
            "negative",
            123,
            "test/three-view",
            suppress_builtin_style=True,
            additional_reference_input_names=[
                "pet-left.png",
                "pet-right.png",
            ],
        )

        scaled_by_source: dict[str, list[object]] = {}
        for node_id, node in workflow.items():
            if node.get("class_type") != "LoadImage":
                continue
            source = node["inputs"]["image"]
            scale_nodes = [
                scale_id
                for scale_id, scale_node in workflow.items()
                if scale_node.get("class_type") == "FluxKontextImageScale"
                and scale_node["inputs"].get("image") == [node_id, 0]
            ]
            self.assertEqual(len(scale_nodes), 1)
            scaled_by_source[source] = [scale_nodes[0], 0]

        self.assertEqual(workflow["6"]["inputs"]["image1"], ["5", 0])
        self.assertEqual(
            workflow["6"]["inputs"]["image2"],
            scaled_by_source["pet-left.png"],
        )
        self.assertEqual(
            workflow["6"]["inputs"]["image3"],
            scaled_by_source["pet-right.png"],
        )
        self.assertEqual(
            workflow["6"]["inputs"]["image2"],
            workflow["7"]["inputs"]["image2"],
        )
        self.assertEqual(
            workflow["6"]["inputs"]["image3"],
            workflow["7"]["inputs"]["image3"],
        )
        self.assertEqual(workflow["10"]["inputs"]["pixels"], ["5", 0])
        self.assertEqual(workflow["14"]["inputs"]["latent_image"], ["10", 0])

    def test_qwen_workflow_rejects_conflicting_or_excess_reference_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be used together"):
            pipeline.qwen_workflow(
                "front.png",
                "positive",
                "negative",
                1,
                "test/conflict",
                pose_reference_input_name="pose.png",
                additional_reference_input_names=["left.png"],
            )
        with self.assertRaisesRegex(ValueError, "at most three images"):
            pipeline.qwen_workflow(
                "front.png",
                "positive",
                "negative",
                1,
                "test/excess",
                additional_reference_input_names=[
                    "left.png",
                    "right.png",
                    "fourth.png",
                ],
            )

    def test_uploaded_qwen_stage_uses_v2_loras_and_returns_alpha_references(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_root = root / "outputs"
            comfy_input = root / "comfy-input"
            job_id = "test-qwen-reference"
            job_dir = output_root / job_id
            job_dir.mkdir(parents=True)
            upload_digests = _write_test_upload_identity(job_dir)
            profile = _subject_profile_with_action_references(root)
            with patch.object(
                pipeline,
                "GENERATION_VISUAL_AUDITS_BLOCKING",
                True,
            ):
                effective_config, expected_pose_policies = (
                    _configured_test_upload_profile(
                        profile,
                        ["eating", "grooming"],
                    )
                )
            jobs = {
                job_id: {
                    "job_id": job_id,
                    "generation_mode": "subject_replacement",
                    "subject_source": "upload_pending_qwen",
                    "prompt": "fluffy long soft fur",
                    "prompt_en": "fluffy long soft fur",
                    "seed": 20260825,
                    "scail_base_seed": 20260825,
                    "outputs": [],
                    "cancel_requested": False,
                    "effective_subject_config": effective_config,
                    "expected_pose_policies": expected_pose_policies,
                    **upload_digests,
                }
            }
            expected_loras = [
                dict(item)
                for item in pipeline.IMAGE_PROFILES[
                    pipeline.SUBJECT_REPLACEMENT_REFERENCE_IMAGE_PROFILE
                ]["user_loras"]
            ]
            eating_driver_sha256 = pipeline.file_sha256(
                Path(profile["runtime"]["actions"]["eating"]["driving_animation"])
            )
            guide_sources: dict[str, Path] = {}
            surface_evaluations: list[tuple[Path, str]] = []
            canonical_reference = job_dir / "cartoon_reference_transparent.png"
            canonical_qwen_image1 = job_dir / "cartoon_reference.png"

            def fake_eating_guide(
                target_job_id: str,
                source: Path,
                target: Path,
                *,
                frames: int,
                fps: float,
            ) -> dict[str, object]:
                self.assertEqual(target_job_id, job_id)
                guide_sources["eating"] = source
                return _write_fake_geometry_guide(
                    target_job_id,
                    source,
                    target,
                    frames=frames,
                    fps=fps,
                )

            def fake_grooming_guide(
                target_job_id: str,
                source: Path,
                target: Path,
                *,
                frames: int,
                fps: float,
            ) -> dict[str, object]:
                self.assertEqual(target_job_id, job_id)
                guide_sources["grooming"] = source
                target.parent.mkdir(parents=True, exist_ok=True)
                qwen_guide = Image.new("RGB", (512, 512), (0, 255, 0))
                qwen_guide.paste(
                    pipeline.GROOMING_QWEN_GUIDE_FOREGROUND_RGB,
                    (112, 72, 400, 440),
                )
                open_points = [
                    (208, 245),
                    (214, 260),
                    (218, 275),
                    (218, 295),
                    (216, 310),
                    (204, 335),
                ]
                ImageDraw.Draw(qwen_guide).line(
                    open_points,
                    fill=(0, 255, 0),
                    width=9,
                )
                qwen_array = np.asarray(qwen_guide, dtype=np.uint8)
                qwen_body = np.zeros((512, 512), dtype=bool)
                qwen_body[72:440, 112:400] = True
                open_pixels = int(
                    np.count_nonzero(
                        qwen_body & np.all(qwen_array == (0, 255, 0), axis=2)
                    )
                )
                qwen_guide.save(target)
                gate_target = target.with_name(target.stem + "_cv_gate.png")
                gate_guide = Image.new("RGB", (512, 512), (0, 255, 0))
                gate_guide.paste((112, 112, 112), (112, 72, 256, 440))
                gate_guide.paste((168, 168, 168), (256, 72, 400, 440))
                gate_guide.save(gate_target)
                completion_mask_target = target.with_name(
                    target.stem + "_completion_mask.png"
                )
                completion_mask = Image.new("RGB", (512, 512), (0, 255, 0))
                completion_mask.paste(
                    pipeline.GROOMING_QWEN_GUIDE_FOREGROUND_RGB,
                    (112, 72, 400, 440),
                )
                completion_mask.paste(
                    pipeline.GROOMING_QWEN_GUIDE_FORELEG_RGB,
                    (180, 220, 220, 340),
                )
                completion_mask.save(completion_mask_target)
                gate_metadata_path = gate_target.with_name(
                    gate_target.stem + "_metadata.json"
                )
                gate_metadata_path.write_text(
                    json.dumps(
                        {
                            "policy": (
                                pipeline.GROOMING_PRIVATE_POSE_GATE_GUIDE_POLICY
                            ),
                            "raised_paw_muzzle_relation": {
                                "passed": True,
                                "source_luma_values_preserved": False,
                                "black_or_tool_semantics_preserved": False,
                                "output_encoding": {
                                    "method": (
                                        "compressed_blurred_quantized_neutral_anatomy"
                                    )
                                },
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                guide_digest = pipeline.file_sha256(target)
                gate_digest = pipeline.file_sha256(gate_target)
                completion_mask_digest = pipeline.file_sha256(completion_mask_target)
                metadata_path = target.with_name(target.stem + "_metadata.json")
                metadata: dict[str, object] = {
                    "policy": pipeline.GROOMING_QWEN_GEOMETRY_GUIDE_POLICY,
                    "method": pipeline.GROOMING_QWEN_GEOMETRY_GUIDE_METHOD,
                    "source_sha256": pipeline.file_sha256(source),
                    "guide_sha256": guide_digest,
                    "metadata_path": str(metadata_path),
                    "frames": frames,
                    "fps": fps,
                    "qwen_guide": {
                        "role": "image2_action_geometry_guide",
                        "path": str(target),
                        "sha256": guide_digest,
                        "foreground_rgb": list(
                            pipeline.GROOMING_QWEN_GUIDE_FOREGROUND_RGB
                        ),
                        "foreground_palette_rgb": [
                            list(pipeline.GROOMING_QWEN_GUIDE_FOREGROUND_RGB),
                        ],
                        "unique_foreground_rgb_count": 1,
                        "internal_luma_variation": False,
                        "internal_luma_variation_role": "none",
                        "appearance_bearing_internal_luma_variation": False,
                        "main_case_appearance_exposed_to_qwen": False,
                        "synthetic_foreleg_semantic": None,
                        "structural_negative_space": {
                            "method": pipeline.GROOMING_QWEN_NEGATIVE_SPACE_METHOD,
                            "pixels": open_pixels,
                            "line_width": 9,
                            "points": [list(point) for point in open_points],
                            "curve_closed": False,
                            "paw_endpoint_open": True,
                            "shoulder_endpoint_open": True,
                            "foreleg_remains_attached_at_shoulder": True,
                            "paw_muzzle_contact_remains_connected": True,
                            "source_rgb_or_luma_sampled": False,
                            "appearance_bearing": False,
                            "adds_foreground_color": False,
                        },
                        "closed_foreleg_capsule_exposed": False,
                        "internal_green_separator_exposed": True,
                        "raised_foreleg_validation_role": (
                            "public_open_curve_geometry_plus_private_cv"
                        ),
                    },
                    "private_pose_gate": {
                        "guide_path": str(gate_target),
                        "guide_sha256": gate_digest,
                        "metadata_path": str(gate_metadata_path),
                        "policy": (pipeline.GROOMING_PRIVATE_POSE_GATE_GUIDE_POLICY),
                        "exposed_to_qwen": False,
                    },
                    "private_completion_mask": {
                        "role": "local_noise_mask_derivation_only",
                        "policy": (pipeline.GROOMING_PRIVATE_COMPLETION_MASK_POLICY),
                        "path": str(completion_mask_target),
                        "sha256": completion_mask_digest,
                        "exposed_to_qwen": False,
                        "loaded_as_qwen_image": False,
                        "semantic_foreleg_pixels": 4800,
                    },
                    "palette": {"source_rgb_preserved": False},
                    "raised_paw_muzzle_relation": {"passed": True},
                }
                metadata_path.write_text(
                    json.dumps(metadata),
                    encoding="utf-8",
                )
                return metadata

            def fake_qwen_run(
                target_job_id: str, *, force_qwen_reference: bool = False
            ) -> None:
                self.assertEqual(target_job_id, job_id)
                self.assertTrue(force_qwen_reference)
                prepared_job = jobs[job_id]
                self.assertEqual(prepared_job["seed"], 20260825)
                self.assertEqual(prepared_job["scail_base_seed"], 20260825)
                self.assertEqual(
                    prepared_job["qwen_seed"],
                    pipeline.IMAGE_PROFILES[
                        pipeline.SUBJECT_REPLACEMENT_REFERENCE_IMAGE_PROFILE
                    ]["seed"],
                )
                self.assertEqual(
                    prepared_job["pose_identity_input_policy"],
                    pipeline.POSE_IDENTITY_INPUT_POLICY_CANONICAL,
                )
                self.assertEqual(
                    prepared_job["pose_identity_input_policies"],
                    {
                        "eating": pipeline.POSE_IDENTITY_INPUT_POLICY_CANONICAL,
                        "grooming": pipeline.POSE_IDENTITY_INPUT_POLICY_CANONICAL,
                    },
                )
                self.assertFalse(prepared_job["main_case_raw_identity_chain"])
                self.assertFalse(prepared_job["bypass_generated_canonical"])
                self.assertEqual(
                    prepared_job["reference_pipeline"],
                    pipeline.QWEN_FULL_BODY_FIRST_FRAME_PIPELINE,
                )
                for action, input_name in prepared_job["pose_reference_inputs"].items():
                    guide_path = comfy_input / input_name
                    self.assertTrue(guide_path.is_file())
                    self.assertEqual(
                        prepared_job["pose_reference_input_sha256s"][action],
                        pipeline.file_sha256(guide_path),
                    )
                canonical_image = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
                canonical_image.paste((80, 80, 80, 255), (128, 64, 384, 448))
                canonical_image.save(canonical_reference)
                canonical_green = Image.new("RGB", (512, 512), (0, 255, 0))
                canonical_green.paste((80, 80, 80), (128, 64, 384, 448))
                canonical_green.save(canonical_qwen_image1)
                _record_fake_canonical_image1(job_id, job_dir, comfy_input)
                pipeline.append_output(
                    job_id,
                    {
                        "type": "cartoon_reference",
                        "filename": canonical_reference.name,
                        "transparent": True,
                    },
                )
                selected_surface_metrics: dict[str, object] = {}
                for pose_index, action in enumerate(("eating", "grooming"), start=1):
                    target = job_dir / f"{action}_transparent.png"
                    image = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
                    image.paste((80, 80, 80, 255), (128, 96, 384, 416))
                    image.save(target)
                    pipeline.append_output(
                        job_id,
                        {
                            "type": "transparent_keyframe",
                            "action": action,
                            "pose_index": pose_index,
                            "filename": target.name,
                        },
                    )
                    _record_fake_action_quality(
                        job_dir,
                        action=action,
                        pose_index=pose_index,
                        reference=target,
                    )
                    selected_candidate = job_dir / f"{action}_selected_candidate.png"
                    selected_candidate.write_bytes(target.read_bytes())
                    selected_surface_metrics[action] = _passed_surface_metric(
                        selected_candidate,
                        action,
                    )
                pipeline.update_job(
                    job_id,
                    fur_surface_gate_selected_metrics=selected_surface_metrics,
                )

            def fake_surface_evaluation(
                reference: Path,
                **_kwargs: object,
            ) -> dict[str, object]:
                surface_evaluations.append((reference, pipeline.file_sha256(reference)))
                return {
                    "acceptable": True,
                    "status": "passed",
                    "issues": [],
                }

            with (
                patch.multiple(
                    pipeline,
                    OUTPUT_ROOT=output_root,
                    COMFY_INPUT=comfy_input,
                    JOBS=jobs,
                    GENERATION_VISUAL_AUDITS_BLOCKING=True,
                ),
                patch.object(
                    pipeline,
                    "audit_uploaded_front_identity_view",
                    return_value=_passed_uploaded_front_view_audit(),
                ),
                patch.object(
                    pipeline,
                    "merge_image_profile_loras",
                    return_value=(expected_loras, "replace", "image_profile"),
                ),
                patch.object(
                    pipeline,
                    "load_subject_profile",
                    side_effect=AssertionError("live profile must not be re-read"),
                ),
                patch.object(
                    pipeline,
                    "prepare_eating_qwen_geometry_guide",
                    side_effect=fake_eating_guide,
                ),
                patch.object(
                    pipeline,
                    "prepare_grooming_qwen_geometry_guide",
                    side_effect=fake_grooming_guide,
                ),
                patch.object(
                    pipeline,
                    "evaluate_main_case_fur_surface_candidate",
                    side_effect=fake_surface_evaluation,
                ),
                patch.object(pipeline, "run_job", side_effect=fake_qwen_run),
            ):
                references = pipeline.prepare_uploaded_subject_qwen_references(
                    job_id,
                    ["eating", "grooming"],
                )
                pipeline.revalidate_uploaded_qwen_references_before_scail(
                    job_id,
                    jobs[job_id],
                    jobs[job_id]["effective_subject_config"],
                    ["eating", "grooming"],
                )
                eating_reference = Path(references["eating"])
                eating_reference_bytes = eating_reference.read_bytes()
                eating_reference.write_bytes(b"changed-after-qwen")
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Qwen eating reference changed before SCAIL",
                ):
                    pipeline.revalidate_uploaded_qwen_references_before_scail(
                        job_id,
                        jobs[job_id],
                        jobs[job_id]["effective_subject_config"],
                        ["eating", "grooming"],
                    )
                eating_reference.write_bytes(eating_reference_bytes)

            final_job = jobs[job_id]
            self.assertEqual(final_job["seed"], 20260825)
            self.assertEqual(final_job["scail_base_seed"], 20260825)
            qwen_output_types = [output["type"] for output in final_job["outputs"]]
            self.assertEqual(qwen_output_types[0], "cartoon_reference")
            self.assertEqual(
                qwen_output_types[1:],
                ["transparent_keyframe", "transparent_keyframe"],
            )
            self.assertTrue(
                all(
                    Path(reference) != job_dir / "input.png"
                    for reference in references.values()
                )
            )
            canonical_digest = pipeline.file_sha256(canonical_reference)
            canonical_qwen_digest = pipeline.file_sha256(canonical_qwen_image1)
            with Image.open(canonical_reference) as canonical_image:
                self.assertEqual(canonical_image.size, (512, 512))
                self.assertEqual(canonical_image.mode, "RGBA")
                alpha_minimum, alpha_maximum = canonical_image.getchannel(
                    "A"
                ).getextrema()
                self.assertLess(alpha_minimum, 255)
                self.assertGreater(alpha_maximum, 0)
            for reference in references.values():
                with Image.open(reference) as keyframe_image:
                    self.assertEqual(keyframe_image.size, (512, 512))
                    self.assertEqual(keyframe_image.mode, "RGBA")
                    alpha_minimum, alpha_maximum = keyframe_image.getchannel(
                        "A"
                    ).getextrema()
                    self.assertLess(alpha_minimum, 255)
                    self.assertGreater(alpha_maximum, 0)

        self.assertEqual(set(references), {"eating", "grooming"})
        self.assertEqual(final_job["subject_source"], "qwen_generated_upload")
        self.assertEqual(final_job["reference_generation_status"], "completed")
        self.assertTrue(final_job["canonical_reference_generated"])
        self.assertTrue(final_job["canonical_reference_used"])
        self.assertTrue(final_job["canonical_reference_normalized"])
        self.assertTrue(final_job["canonical_reference_transparent"])
        self.assertTrue(final_job["qwen_action_keyframes_normalized"])
        self.assertTrue(final_job["qwen_action_keyframes_transparent"])
        self.assertEqual(
            final_job["canonical_reference_path"],
            str(canonical_reference),
        )
        self.assertEqual(
            final_job["canonical_reference_sha256"],
            canonical_digest,
        )
        self.assertEqual(final_job["reference_normalization_status"], "completed")
        self.assertEqual(
            set(final_job["qwen_action_keyframe_sha256s"]),
            {"eating", "grooming"},
        )
        self.assertTrue(final_job["qwen_action_reference_byte_preserved"])
        self.assertEqual(
            final_job["qwen_action_reference_normalization_policy"],
            pipeline.QWEN_ACTION_REFERENCE_NORMALIZATION_POLICY,
        )
        self.assertEqual(len(surface_evaluations), 4)
        for evaluated_path, evaluated_digest in surface_evaluations:
            action = evaluated_path.stem.removesuffix("_transparent")
            self.assertIn(action, {"eating", "grooming"})
            self.assertEqual(
                evaluated_digest,
                final_job["qwen_action_keyframe_sha256s"][action],
            )
            self.assertEqual(
                evaluated_digest,
                final_job["subject_fur_surface_quality"][action][
                    "persisted_reference_sha256"
                ],
            )
            self.assertTrue(
                final_job["subject_fur_surface_quality"][action][
                    "byte_identical_to_selected_candidate"
                ]
            )
        self.assertEqual(
            final_job["uploaded_identity_anchor_path"],
            str(job_dir / "input_identity_source.png"),
        )
        self.assertEqual(
            final_job["uploaded_identity_input_sha256"],
            upload_digests["uploaded_identity_input_sha256"],
        )
        self.assertEqual(
            final_job["uploaded_identity_source_sha256"],
            upload_digests["uploaded_identity_source_sha256"],
        )
        self.assertEqual(
            final_job["canonical_qwen_image1_path"],
            str(canonical_qwen_image1),
        )
        self.assertEqual(
            final_job["canonical_qwen_image1_sha256"],
            canonical_qwen_digest,
        )
        self.assertEqual(
            final_job["canonical_qwen_image1_comfy_path"],
            str(comfy_input / "local_video_api" / job_id / "cartoon_reference.png"),
        )
        self.assertEqual(
            final_job["canonical_qwen_image1_comfy_sha256"],
            final_job["canonical_qwen_image1_sha256"],
        )
        self.assertEqual(final_job["surface_texture_gate_status"], "passed")
        self.assertEqual(
            set(final_job["subject_fur_surface_quality"]),
            {"eating", "grooming"},
        )
        self.assertEqual(
            final_job["reference_image_profile"], "pet-multipose-20260717-v2"
        )
        self.assertEqual(
            [
                (item["original_filename"], item["strength"])
                for item in final_job["user_loras"]
            ],
            [
                ("宠物小猫 _ 写实摄影_v3.0.safetensors", 0.5),
                ("Qwen Image-Kitten Photography_QWEN-小猫摄影.safetensors", 0.6),
            ],
        )
        self.assertEqual(final_job["qwen_steps"], 6)
        self.assertEqual(final_job["qwen_cfg"], 1.0)
        self.assertEqual(final_job["qwen_denoise"], 1.0)
        self.assertEqual(
            final_job["qwen_seed"],
            pipeline.IMAGE_PROFILES["pet-multipose-20260717-v2"]["seed"],
        )
        self.assertFalse(final_job["lock_canonical_retry_lora_strengths"])
        self.assertFalse(final_job["lock_pose_retry_lora_strengths"])
        self.assertEqual(
            final_job["canonical_user_lora_total_cap"],
            pipeline.QWEN_SUBJECT_REPLACEMENT_CANONICAL_LORA_TOTAL_CAP,
        )
        self.assertEqual(
            final_job["pose_user_lora_first_total_cap"],
            pipeline.QWEN_SUBJECT_REPLACEMENT_POSE_LORA_TOTAL_CAP,
        )
        self.assertEqual(
            final_job["pose_reference_denoise"],
            pipeline.QWEN_SUBJECT_REPLACEMENT_POSE_DENOISE,
        )
        self.assertEqual(
            final_job["pose_reference_denoise_step"],
            pipeline.QWEN_SUBJECT_REPLACEMENT_POSE_DENOISE_STEP,
        )
        self.assertEqual(
            final_job["pose_reference_inputs"],
            {
                "eating": (
                    "local_video_api/test-qwen-reference/eating_qwen_pose_guide.png"
                ),
                "grooming": (
                    "local_video_api/test-qwen-reference/grooming_qwen_pose_guide.png"
                ),
            },
        )
        self.assertEqual(
            final_job["reference_pose_source"],
            "per_action_main_case_driver_geometry",
        )
        self.assertEqual(
            final_job["pose_reference_policy"],
            {
                "eating": ("main_case_eating_driver_geometry_image2_guide_v7"),
                "grooming": pipeline.GROOMING_QWEN_POSE_REFERENCE_POLICY,
            },
        )
        self.assertEqual(
            final_job["pose_identity_input_policy"],
            pipeline.POSE_IDENTITY_INPUT_POLICY_CANONICAL,
        )
        self.assertEqual(
            final_job["pose_identity_input_policies"],
            {
                "eating": pipeline.POSE_IDENTITY_INPUT_POLICY_CANONICAL,
                "grooming": pipeline.POSE_IDENTITY_INPUT_POLICY_CANONICAL,
            },
        )
        self.assertEqual(
            final_job["pose_identity_input_roles"],
            {
                "eating": "image1_qwen_full_body_canonical",
                "grooming": "image1_qwen_full_body_canonical",
            },
        )
        self.assertEqual(
            final_job["pose_identity_geometry_roles"],
            {
                "eating": "image2_main_case_driver_geometry_guide",
                "grooming": "image2_main_case_driver_geometry_guide",
            },
        )
        self.assertEqual(
            final_job["eating_pose_lora_policy"],
            "canonical_identity_plus_main_case_geometry_v1",
        )
        self.assertTrue(final_job["eating_identity_refine_enabled"])
        self.assertFalse(final_job["eating_identity_refine_active"])
        self.assertEqual(
            final_job["eating_identity_refine_policy"],
            pipeline.EATING_CANONICAL_IDENTITY_REFINE_POLICY,
        )
        self.assertEqual(
            final_job["eating_identity_refine_attempt"],
            pipeline.EATING_CANONICAL_IDENTITY_REFINE_ATTEMPT,
        )
        self.assertEqual(
            final_job["eating_identity_refine_source_capture_through_attempt"],
            pipeline.QWEN_ACTION_CANDIDATE_ATTEMPTS - 1,
        )
        self.assertTrue(
            final_job["eating_identity_refine_attempt5_can_refine_attempt4"]
        )
        self.assertEqual(
            final_job["eating_identity_refine_denoise"],
            pipeline.EATING_CANONICAL_IDENTITY_REFINE_DENOISE,
        )
        self.assertEqual(
            final_job["canonical_action_light_identity_black_fraction_max"],
            pipeline.CANONICAL_ACTION_LIGHT_IDENTITY_BLACK_FRACTION_MAX,
        )
        self.assertEqual(
            final_job["canonical_action_dark_feature_expansion_max"],
            pipeline.CANONICAL_ACTION_DARK_FEATURE_EXPANSION_MAX,
        )
        self.assertIsNone(final_job["eating_identity_refine_style_lora"])
        self.assertTrue(final_job["eating_identity_refine_global_thresholds_changed"])
        self.assertTrue(
            final_job["eating_identity_source_absent_stripe_gate_unchanged"]
        )
        self.assertEqual(
            final_job["eating_identity_spatial_marking_policy"],
            pipeline.IDENTITY_MARKING_SPATIAL_POLICY,
        )
        self.assertEqual(
            final_job[
                "eating_identity_source_present_stripe_amplification_thresholds"
            ]["area_delta_min"],
            pipeline.IDENTITY_AMPLIFIED_STRIPE_AREA_DELTA_MIN,
        )
        self.assertEqual(
            final_job["pose_reference_source_paths"],
            {
                "eating": profile["runtime"]["actions"]["eating"]["driving_animation"],
                "grooming": profile["runtime"]["actions"]["grooming"][
                    "driving_animation"
                ],
            },
        )
        self.assertEqual(
            guide_sources,
            {
                "eating": Path(
                    profile["runtime"]["actions"]["eating"]["driving_animation"]
                ),
                "grooming": Path(
                    profile["runtime"]["actions"]["grooming"]["driving_animation"]
                ),
            },
        )
        self.assertEqual(
            final_job["pose_reference_guide_style"],
            "per_action_main_case_driver_geometry_v2",
        )
        self.assertEqual(
            final_job["pose_reference_guide_styles"],
            {
                "eating": (
                    "single_tone_neutral_action_silhouette_with_explicit_"
                    "muzzle_overlap_and_semantic_food_on_pure_green"
                ),
                "grooming": (pipeline.GROOMING_QWEN_GUIDE_STYLE),
            },
        )
        self.assertEqual(
            final_job["pose_reference_role"],
            "per_action_geometry_input_role",
        )
        eating_metadata = final_job["pose_reference_metadata"]["eating"]
        self.assertEqual(
            eating_metadata["policy"],
            "main_case_eating_driver_geometry_image2_guide_v7",
        )
        self.assertEqual(
            eating_metadata["image_role"],
            "image2_action_geometry_guide",
        )
        self.assertEqual(
            eating_metadata["identity_role"],
            "image1_qwen_full_body_canonical",
        )
        self.assertFalse(eating_metadata["source_rgb_preserved"])
        self.assertFalse(eating_metadata["qwen_guide_internal_luma_variation"])
        self.assertFalse(
            eating_metadata[
                "qwen_guide_appearance_bearing_internal_luma_variation"
            ]
        )
        self.assertTrue(eating_metadata["driver_silhouette_used"])
        self.assertFalse(eating_metadata["reference_space_motion"])
        self.assertFalse(eating_metadata["main_case_driver_used"])
        self.assertTrue(eating_metadata["uploaded_white_maquette_driver_used"])
        self.assertEqual(
            eating_metadata["palette_audit_role"],
            "qwen_full_body_canonical",
        )
        self.assertEqual(
            eating_metadata["source_sha256"],
            eating_driver_sha256,
        )
        grooming_metadata = final_job["pose_reference_metadata"]["grooming"]
        self.assertEqual(
            grooming_metadata["policy"],
            pipeline.GROOMING_QWEN_POSE_REFERENCE_POLICY,
        )
        self.assertEqual(
            grooming_metadata["generator_policy"],
            pipeline.GROOMING_QWEN_GEOMETRY_GUIDE_POLICY,
        )
        self.assertEqual(
            grooming_metadata["image_role"],
            "image2_action_geometry_guide",
        )
        self.assertEqual(
            grooming_metadata["identity_role"],
            "image1_qwen_full_body_canonical",
        )
        self.assertTrue(grooming_metadata["qwen_guide_structural_negative_space"])
        self.assertEqual(
            grooming_metadata["qwen_guide_structural_negative_space_method"],
            pipeline.GROOMING_QWEN_NEGATIVE_SPACE_METHOD,
        )
        self.assertFalse(grooming_metadata["main_case_appearance_exposed_to_qwen"])
        self.assertTrue(
            grooming_metadata["uploaded_white_maquette_driver_used"]
        )
        self.assertFalse(grooming_metadata["qwen_guide_internal_luma_variation"])
        self.assertEqual(
            grooming_metadata["qwen_guide_internal_luma_variation_role"],
            "none",
        )
        self.assertFalse(
            grooming_metadata["qwen_guide_appearance_bearing_internal_luma_variation"]
        )
        self.assertFalse(grooming_metadata["qwen_guide_synthetic_foreleg_semantic"])
        self.assertEqual(
            grooming_metadata["qwen_guide_synthetic_foreleg_semantic_method"],
            None,
        )
        self.assertFalse(grooming_metadata["qwen_guide_closed_foreleg_capsule_exposed"])
        self.assertTrue(
            grooming_metadata["qwen_guide_internal_green_separator_exposed"]
        )
        self.assertEqual(
            grooming_metadata["qwen_guide_raised_foreleg_validation_role"],
            "public_open_curve_geometry_plus_private_cv",
        )
        self.assertEqual(
            grooming_metadata["qwen_foreleg_repair_policy"],
            pipeline.GROOMING_CANONICAL_FORELEG_COMPLETION_POLICY,
        )
        self.assertEqual(
            grooming_metadata["qwen_foreleg_repair_attempt"],
            pipeline.GROOMING_QWEN_FORELEG_REPAIR_ATTEMPT,
        )
        self.assertEqual(
            grooming_metadata["qwen_foreleg_repair_denoise"],
            pipeline.GROOMING_QWEN_FORELEG_REPAIR_DENOISE,
        )
        self.assertFalse(
            grooming_metadata["qwen_foreleg_repair_synthetic_capsule_exposed"]
        )
        self.assertTrue(
            grooming_metadata["qwen_foreleg_repair_open_channel_anatomy_image3"]
        )
        self.assertFalse(
            grooming_metadata["qwen_foreleg_repair_private_cv_luma_image3"]
        )
        self.assertEqual(
            grooming_metadata["qwen_foreleg_repair_mask_dilation_px"],
            pipeline.GROOMING_QWEN_FORELEG_REPAIR_MASK_DILATION_PX,
        )
        self.assertEqual(
            grooming_metadata["qwen_foreleg_repair_seed_source"],
            "identity_safe_candidate_unchanged_before_masked_sampling",
        )
        self.assertFalse(
            grooming_metadata["qwen_foreleg_repair_identity_or_geometry_seed_inserted"]
        )
        self.assertTrue(
            grooming_metadata["qwen_foreleg_repair_expanded_context_full_denoise_mask"]
        )
        self.assertEqual(
            grooming_metadata["qwen_foreleg_repair_style_policy"],
            "disabled_for_canonical_masked_completion",
        )
        self.assertIsNone(grooming_metadata["qwen_foreleg_repair_style_lora"])
        self.assertEqual(
            grooming_metadata["qwen_foreleg_repair_style_strength"],
            0.0,
        )
        self.assertEqual(
            grooming_metadata["qwen_foreleg_repair_style_scope"],
            "none",
        )
        self.assertIsNone(
            grooming_metadata["qwen_foreleg_repair_composite_gate_policy"]
        )
        self.assertFalse(
            grooming_metadata["qwen_foreleg_repair_composite_recovery_enabled"]
        )
        self.assertIsNone(
            grooming_metadata["qwen_foreleg_repair_composite_edge_density_ratio_min"]
        )
        self.assertIsNone(
            grooming_metadata[
                "qwen_foreleg_repair_composite_planted_paw_gradient_p90_max"
            ]
        )
        self.assertIsNone(
            grooming_metadata[
                "qwen_foreleg_repair_composite_raised_paw_gradient_p90_max"
            ]
        )
        self.assertEqual(
            grooming_metadata["canonical_completion_edge_evidence_policy"],
            pipeline.GROOMING_CANONICAL_COMPLETION_EDGE_EVIDENCE_POLICY,
        )
        self.assertEqual(
            grooming_metadata["canonical_completion_edge_density_ratio_min"],
            pipeline.GROOMING_CANONICAL_COMPLETION_EDGE_DENSITY_RATIO_MIN,
        )
        self.assertTrue(
            grooming_metadata["canonical_completion_independent_fur_surface_required"]
        )
        direct_evidence = grooming_metadata["canonical_direct_structural_edge_evidence"]
        self.assertEqual(
            direct_evidence["policy"],
            pipeline.GROOMING_CANONICAL_DIRECT_STRUCTURAL_EDGE_EVIDENCE_POLICY,
        )
        self.assertEqual(direct_evidence["maximum_attempt"], 2)
        self.assertEqual(direct_evidence["raw_edge_density_ratio_min"], 0.925)
        self.assertEqual(direct_evidence["structural_edge_density_min"], 0.03)
        self.assertEqual(direct_evidence["chamfer_coverage_min"], 0.70)
        self.assertEqual(direct_evidence["chamfer_mean_distance_max"], 5.0)
        self.assertEqual(direct_evidence["chamfer_p90_distance_max"], 12.0)
        self.assertTrue(direct_evidence["identity_palette_gate_still_required"])
        self.assertTrue(direct_evidence["independent_fur_surface_gate_still_required"])
        self.assertFalse(direct_evidence["global_pose_threshold_changed"])
        self.assertTrue(direct_evidence["fail_closed"])
        surface_nonexpansion = grooming_metadata[
            "canonical_direct_surface_nonexpansion"
        ]
        self.assertEqual(
            surface_nonexpansion["policy"],
            pipeline.GROOMING_CANONICAL_DIRECT_SURFACE_NONEXPANSION_POLICY,
        )
        self.assertEqual(surface_nonexpansion["maximum_attempt"], 2)
        self.assertEqual(
            surface_nonexpansion["only_recoverable_issue"],
            "brush_texture_coherence",
        )
        self.assertTrue(surface_nonexpansion["requires_strict_pose_recovery"])
        self.assertTrue(surface_nonexpansion["requires_identity_palette_pass"])
        self.assertTrue(
            surface_nonexpansion["requires_digest_bound_transparent_canonical"]
        )
        self.assertTrue(
            surface_nonexpansion["independent_final_video_fur_audit_still_required"]
        )
        self.assertFalse(surface_nonexpansion["global_surface_thresholds_changed"])
        self.assertTrue(surface_nonexpansion["fail_closed"])
        self.assertTrue(grooming_metadata["canonical_full_pose_recovery_enabled"])
        self.assertEqual(
            grooming_metadata["canonical_full_pose_rescue_policy"],
            pipeline.GROOMING_CANONICAL_FULL_POSE_RESCUE_POLICY,
        )
        self.assertEqual(
            grooming_metadata["canonical_full_pose_rescue_attempt"],
            pipeline.GROOMING_CANONICAL_FULL_POSE_RESCUE_ATTEMPT,
        )
        self.assertEqual(
            grooming_metadata["canonical_full_pose_identity_refine_policy"],
            pipeline.GROOMING_CANONICAL_FULL_POSE_REFINE_POLICY,
        )
        self.assertEqual(
            grooming_metadata["canonical_full_pose_identity_refine_attempt"],
            pipeline.GROOMING_CANONICAL_FULL_POSE_REFINE_ATTEMPT,
        )
        self.assertEqual(
            grooming_metadata["canonical_full_pose_refine_source_policy"],
            pipeline.GROOMING_CANONICAL_POSE_SAFE_REFINE_SOURCE_POLICY,
        )
        self.assertEqual(
            grooming_metadata["canonical_full_pose_refine_image3"],
            "digest_bound_soft_foreleg_depth_action_corroboration",
        )
        self.assertFalse(
            grooming_metadata["canonical_full_pose_refine_uploaded_front_exposed"]
        )
        self.assertEqual(
            grooming_metadata["canonical_full_pose_refine_identity_source"],
            "approved_canonical_three_view_consensus_only",
        )
        self.assertFalse(
            grooming_metadata[
                "canonical_pose_transfer_source_alpha_geometry_preserved"
            ]
        )
        self.assertFalse(
            grooming_metadata[
                "canonical_pose_transfer_action_alpha_overlap_blocking"
            ]
        )
        self.assertTrue(
            grooming_metadata[
                "canonical_pose_transfer_public_geometry_guide_required"
            ]
        )
        self.assertFalse(
            grooming_metadata["canonical_pose_transfer_source_rgb_exposed"]
        )
        self.assertFalse(
            grooming_metadata["canonical_pose_transfer_source_chroma_exposed"]
        )
        self.assertFalse(
            grooming_metadata["canonical_pose_transfer_source_raw_luma_exposed"]
        )
        self.assertTrue(
            grooming_metadata[
                "canonical_pose_transfer_rejected_candidate_used_as_sampler"
            ]
        )
        self.assertEqual(
            grooming_metadata["canonical_pose_transfer_seed_policy"],
            pipeline.GROOMING_CANONICAL_POSE_TRANSFER_SEED_POLICY,
        )
        self.assertEqual(
            grooming_metadata["canonical_pose_transfer_seed_bound_matte_policy"],
            pipeline.GROOMING_CANONICAL_SEED_BOUND_MATTE_POLICY,
        )
        self.assertFalse(
            grooming_metadata["canonical_full_pose_recovery_global_thresholds_changed"]
        )
        self.assertEqual(
            grooming_metadata["qwen_foreleg_repair_global_edge_density_min"],
            0.11,
        )
        self.assertFalse(
            grooming_metadata["qwen_foreleg_repair_global_pose_threshold_changed"]
        )
        self.assertFalse(grooming_metadata["private_pose_gate_exposed_to_qwen"])
        self.assertTrue(grooming_metadata["qwen_foreleg_repair_enabled"])
        self.assertFalse(
            grooming_metadata[
                "private_pose_gate_neutral_copy_exposed_on_repair_attempt"
            ]
        )
        self.assertTrue(
            grooming_metadata["public_open_channel_copy_exposed_on_repair_attempt"]
        )
        self.assertFalse(grooming_metadata["private_completion_mask_exposed_to_qwen"])
        self.assertRegex(
            grooming_metadata["private_completion_mask_sha256"],
            r"^[0-9a-f]{64}$",
        )
        self.assertRegex(
            grooming_metadata["private_pose_gate_guide_sha256"],
            r"^[0-9a-f]{64}$",
        )
        self.assertEqual(
            final_job["grooming_pose_lora_policy"],
            "canonical_identity_open_channel_seed_bound_pose_safe_refine_v11",
        )
        self.assertFalse(final_job["grooming_foreleg_repair_active"])
        self.assertEqual(
            final_job["grooming_foreleg_repair_policy"],
            pipeline.GROOMING_CANONICAL_FORELEG_COMPLETION_POLICY,
        )
        self.assertEqual(
            final_job["grooming_foreleg_repair_style_policy"],
            "disabled_for_canonical_masked_completion",
        )
        self.assertEqual(
            final_job["grooming_foreleg_repair_style_strength"],
            0.0,
        )
        self.assertIsNone(final_job["grooming_foreleg_repair_composite_gate_policy"])
        self.assertIsNone(
            final_job["grooming_foreleg_repair_composite_edge_density_ratio_min"]
        )
        self.assertIsNone(
            final_job["grooming_foreleg_repair_composite_planted_paw_gradient_p90_max"]
        )
        self.assertIsNone(
            final_job["grooming_foreleg_repair_composite_raised_paw_gradient_p90_max"]
        )
        self.assertEqual(
            final_job["grooming_canonical_completion_edge_evidence_policy"],
            pipeline.GROOMING_CANONICAL_COMPLETION_EDGE_EVIDENCE_POLICY,
        )
        self.assertEqual(
            final_job["grooming_canonical_completion_edge_density_ratio_min"],
            pipeline.GROOMING_CANONICAL_COMPLETION_EDGE_DENSITY_RATIO_MIN,
        )
        self.assertTrue(
            final_job["grooming_canonical_completion_independent_fur_surface_required"]
        )
        self.assertEqual(
            final_job["grooming_canonical_direct_structural_edge_evidence_policy"],
            pipeline.GROOMING_CANONICAL_DIRECT_STRUCTURAL_EDGE_EVIDENCE_POLICY,
        )
        self.assertEqual(
            final_job["grooming_canonical_direct_edge_density_ratio_min"],
            0.925,
        )
        self.assertEqual(
            final_job["grooming_canonical_direct_structural_edge_density_min"],
            0.03,
        )
        self.assertEqual(
            final_job["grooming_canonical_direct_chamfer_coverage_min"],
            0.70,
        )
        self.assertEqual(
            final_job["grooming_canonical_direct_chamfer_mean_distance_max"],
            5.0,
        )
        self.assertEqual(
            final_job["grooming_canonical_direct_chamfer_p90_distance_max"],
            12.0,
        )
        self.assertEqual(
            final_job["grooming_canonical_direct_edge_evidence_max_attempt"],
            2,
        )
        self.assertTrue(
            final_job["grooming_canonical_direct_identity_palette_required"]
        )
        self.assertTrue(
            final_job["grooming_canonical_direct_independent_fur_surface_required"]
        )
        self.assertFalse(
            final_job["grooming_canonical_direct_global_thresholds_changed"]
        )
        self.assertEqual(
            final_job["grooming_canonical_direct_surface_nonexpansion_policy"],
            pipeline.GROOMING_CANONICAL_DIRECT_SURFACE_NONEXPANSION_POLICY,
        )
        self.assertEqual(
            final_job["grooming_canonical_direct_surface_only_recoverable_issue"],
            "brush_texture_coherence",
        )
        self.assertEqual(
            final_job["grooming_canonical_direct_surface_coherent_fraction_max"],
            0.57,
        )
        self.assertEqual(
            final_job["grooming_canonical_direct_surface_coherent_fraction_delta_max"],
            0.01,
        )
        self.assertTrue(
            final_job["grooming_canonical_direct_surface_requires_pose_and_identity"]
        )
        self.assertTrue(
            final_job["grooming_canonical_direct_surface_requires_canonical_digest"]
        )
        self.assertTrue(
            final_job["grooming_canonical_direct_surface_final_video_audit_required"]
        )
        self.assertFalse(
            final_job["grooming_canonical_direct_surface_global_thresholds_changed"]
        )
        self.assertTrue(final_job["grooming_full_pose_recovery_enabled"])
        self.assertFalse(final_job["grooming_full_pose_recovery_active"])
        self.assertEqual(
            final_job["grooming_full_pose_rescue_attempt"],
            pipeline.GROOMING_CANONICAL_FULL_POSE_RESCUE_ATTEMPT,
        )
        self.assertEqual(
            final_job["grooming_full_pose_identity_refine_attempt"],
            pipeline.GROOMING_CANONICAL_FULL_POSE_REFINE_ATTEMPT,
        )
        self.assertIsNone(final_job["grooming_full_pose_recovery_style_lora"])
        self.assertFalse(
            final_job["grooming_full_pose_recovery_global_thresholds_changed"]
        )
        self.assertEqual(
            final_job["grooming_foreleg_repair_global_edge_density_min"],
            0.11,
        )
        self.assertFalse(
            final_job["grooming_foreleg_repair_global_pose_threshold_changed"]
        )
        self.assertIsNone(final_job["grooming_pose_gate_composite_recovery"])
        self.assertEqual(
            final_job["canonical_prompt_source"],
            "uploaded_identity_qwen_full_body",
        )
        self.assertTrue(final_job["user_prompt_excluded_from_reference"])
        self.assertNotIn(
            "fluffy long soft fur",
            final_job["qwen_reference_prompt_en"],
        )
        self.assertIn(
            "uploaded pet view or role-labeled view set is the only authority",
            final_job["subject_profile_fur_lock"],
        )
        self.assertNotIn(
            "compact short-to-medium coat",
            final_job["subject_profile_fur_lock"],
        )
        self.assertEqual(
            final_job["subject_profile_appearance_role"],
            "action_geometry_only",
        )
        self.assertFalse(final_job["subject_profile_identity_terms_applied"])
        self.assertIn("eating", final_job["pose_reference_source_sha256s"])

    def test_uploaded_qwen_rejects_changed_identity_source_before_generation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_root = root / "outputs"
            job_id = "test-changed-upload-source"
            job_dir = output_root / job_id
            job_dir.mkdir(parents=True)
            upload_digests = _write_test_upload_identity(job_dir)
            profile = _subject_profile_with_action_references(root, ("eating",))
            effective_config, expected_pose_policies = _configured_test_upload_profile(
                profile, ["eating"]
            )
            jobs = {
                job_id: {
                    "job_id": job_id,
                    "generation_mode": "subject_replacement",
                    "subject_source": "upload_pending_qwen",
                    "outputs": [],
                    "cancel_requested": False,
                    "effective_subject_config": effective_config,
                    "expected_pose_policies": expected_pose_policies,
                    **upload_digests,
                }
            }
            (job_dir / "input_identity_source.png").write_bytes(b"changed")
            with (
                patch.multiple(
                    pipeline,
                    OUTPUT_ROOT=output_root,
                    COMFY_INPUT=root / "comfy-input",
                    JOBS=jobs,
                ),
                patch.object(
                    pipeline,
                    "prepare_eating_qwen_geometry_guide",
                ) as prepare_guide,
                patch.object(pipeline, "run_job") as run_qwen,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "identity source is missing or changed after submission",
                ):
                    pipeline.prepare_uploaded_subject_qwen_references(
                        job_id,
                        ["eating"],
                    )

            prepare_guide.assert_not_called()
            run_qwen.assert_not_called()

    def test_uploaded_qwen_excludes_frozen_profile_identity_constraints(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_root = root / "outputs"
            comfy_input = root / "comfy-input"
            job_id = "test-frozen-fur-constraints"
            job_dir = output_root / job_id
            job_dir.mkdir(parents=True)
            upload_digests = _write_test_upload_identity(job_dir)
            profile = _subject_profile_with_action_references(root, ("eating",))
            effective_config, expected_pose_policies = _configured_test_upload_profile(
                profile, ["eating"]
            )
            effective_config["profile_fur_constraints"] = {
                "positive": "frozen queue-time close coat rule",
                "negative": "frozen queue-time expanded coat ban",
            }
            jobs = {
                job_id: {
                    "job_id": job_id,
                    "generation_mode": "subject_replacement",
                    "subject_source": "upload_pending_qwen",
                    "subject_profile": "sleek_tabby_cat_v1",
                    "prompt": "",
                    "prompt_en": "",
                    "outputs": [],
                    "cancel_requested": False,
                    "effective_subject_config": effective_config,
                    "expected_pose_policies": expected_pose_policies,
                    **upload_digests,
                },
            }
            profile_loras = [
                dict(item)
                for item in pipeline.IMAGE_PROFILES[
                    pipeline.SUBJECT_REPLACEMENT_REFERENCE_IMAGE_PROFILE
                ]["user_loras"]
            ]

            def fake_qwen(
                _target_job_id: str,
                *,
                force_qwen_reference: bool = False,
            ) -> None:
                self.assertTrue(force_qwen_reference)
                canonical = job_dir / "cartoon_reference_transparent.png"
                canonical_image = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
                canonical_image.paste((80, 80, 80, 255), (128, 64, 384, 448))
                canonical_image.save(canonical)
                canonical_green = Image.new("RGB", (512, 512), (0, 255, 0))
                canonical_green.paste((80, 80, 80), (128, 64, 384, 448))
                canonical_green.save(job_dir / "cartoon_reference.png")
                _record_fake_canonical_image1(job_id, job_dir, comfy_input)
                pipeline.append_output(
                    job_id,
                    {
                        "type": "cartoon_reference",
                        "filename": canonical.name,
                        "transparent": True,
                    },
                )
                selected_action = job_dir / "pose_01_qwen_cutout.png"
                action = job_dir / "pose_01_keyframe_transparent.png"
                image = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
                image.paste((80, 80, 80, 255), (128, 96, 384, 416))
                image.save(selected_action)
                action.write_bytes(selected_action.read_bytes())
                pipeline.append_output(
                    job_id,
                    {
                        "type": "transparent_keyframe",
                        "action": "eating",
                        "pose_index": 1,
                        "filename": action.name,
                    },
                )
                _record_fake_action_quality(
                    job_dir,
                    action="eating",
                    pose_index=1,
                    reference=action,
                    selected_reference=selected_action,
                    best_effort=True,
                )
                pipeline.update_job(
                    job_id,
                    qwen_first_frame_best_effort_actions=["eating"],
                    fur_surface_gate_selected_metrics={
                        "eating": _passed_surface_metric(action, "eating")
                    },
                )

            with (
                patch.multiple(
                    pipeline,
                    OUTPUT_ROOT=output_root,
                    COMFY_INPUT=comfy_input,
                    JOBS=jobs,
                ),
                patch.object(
                    pipeline,
                    "audit_uploaded_front_identity_view",
                    return_value=_passed_uploaded_front_view_audit(),
                ),
                patch.object(
                    pipeline,
                    "merge_image_profile_loras",
                    return_value=(profile_loras, "replace", "image_profile"),
                ),
                patch.object(
                    pipeline,
                    "load_subject_profile",
                    side_effect=AssertionError("live profile must not be re-read"),
                ),
                patch.object(
                    pipeline,
                    "prepare_eating_qwen_geometry_guide",
                    side_effect=_write_fake_geometry_guide,
                ),
                patch.object(
                    pipeline,
                    "evaluate_main_case_fur_surface_candidate",
                    return_value={
                        "acceptable": True,
                        "status": "passed",
                        "issues": [],
                    },
                ),
                patch.object(pipeline, "run_job", side_effect=fake_qwen),
            ):
                pipeline.prepare_uploaded_subject_qwen_references(
                    job_id,
                    ["eating"],
                )
                pipeline.revalidate_uploaded_qwen_references_before_scail(
                    job_id,
                    jobs[job_id],
                    jobs[job_id]["effective_subject_config"],
                    ["eating"],
                )
                (job_dir / "pose_01_qwen_cutout.png").write_bytes(
                    b"tampered-selected-candidate"
                )
                with self.assertRaisesRegex(
                    RuntimeError,
                    "best-effort reference contract",
                ):
                    pipeline.revalidate_uploaded_qwen_references_before_scail(
                        job_id,
                        jobs[job_id],
                        jobs[job_id]["effective_subject_config"],
                        ["eating"],
                    )

            final_job = jobs[job_id]

        self.assertIn(
            "uploaded pet view or role-labeled view set is the only authority",
            final_job["subject_profile_fur_lock"],
        )
        self.assertNotIn(
            "frozen queue-time close coat rule",
            final_job["subject_profile_fur_lock"],
        )
        self.assertNotIn(
            "frozen queue-time expanded coat ban",
            final_job["subject_profile_fur_negative"],
        )
        self.assertEqual(
            final_job["qwen_first_frame_best_effort_references"]["eating"][
                "published_reference_sha256"
            ],
            final_job["qwen_action_keyframe_sha256s"]["eating"],
        )
        self.assertTrue(
            final_job["qwen_first_frame_best_effort_references"]["eating"][
                "byte_preserved_publish_alias"
            ]
        )
        self.assertEqual(
            final_job["qwen_first_frame_best_effort_references"]["eating"][
                "selected_candidate_sha256"
            ],
            final_job["qwen_action_keyframe_sha256s"]["eating"],
        )
        self.assertTrue(
            final_job["qwen_first_frame_best_effort_references"]["eating"][
                "continues_to_scail"
            ]
        )

    def test_uploaded_qwen_stage_fails_closed_on_opaque_action_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_root = root / "outputs"
            comfy_input = root / "comfy-input"
            job_id = "test-opaque-reference"
            job_dir = output_root / job_id
            job_dir.mkdir(parents=True)
            upload_digests = _write_test_upload_identity(job_dir)
            profile = _subject_profile_with_action_references(root, ("eating",))
            effective_config, expected_pose_policies = _configured_test_upload_profile(
                profile, ["eating"]
            )
            jobs = {
                job_id: {
                    "job_id": job_id,
                    "generation_mode": "subject_replacement",
                    "subject_source": "upload_pending_qwen",
                    "prompt": "",
                    "prompt_en": "",
                    "outputs": [],
                    "cancel_requested": False,
                    "effective_subject_config": effective_config,
                    "expected_pose_policies": expected_pose_policies,
                    **upload_digests,
                }
            }
            profile_loras = [
                dict(item)
                for item in pipeline.IMAGE_PROFILES[
                    pipeline.SUBJECT_REPLACEMENT_REFERENCE_IMAGE_PROFILE
                ]["user_loras"]
            ]

            def fake_opaque_qwen(
                _target_job_id: str,
                *,
                force_qwen_reference: bool = False,
            ) -> None:
                self.assertTrue(force_qwen_reference)
                canonical = job_dir / "cartoon_reference_transparent.png"
                canonical_image = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
                canonical_image.paste((80, 80, 80, 255), (128, 64, 384, 448))
                canonical_image.save(canonical)
                canonical_green = Image.new("RGB", (512, 512), (0, 255, 0))
                canonical_green.paste((80, 80, 80), (128, 64, 384, 448))
                canonical_green.save(job_dir / "cartoon_reference.png")
                _record_fake_canonical_image1(job_id, job_dir, comfy_input)
                pipeline.append_output(
                    job_id,
                    {
                        "type": "cartoon_reference",
                        "filename": canonical.name,
                        "transparent": True,
                    },
                )
                opaque = job_dir / "eating_opaque.png"
                Image.new("RGB", (512, 512), "white").save(opaque)
                pipeline.append_output(
                    job_id,
                    {
                        "type": "transparent_keyframe",
                        "action": "eating",
                        "pose_index": 1,
                        "filename": opaque.name,
                    },
                )
                _record_fake_action_quality(
                    job_dir,
                    action="eating",
                    pose_index=1,
                    reference=opaque,
                )

            with (
                patch.multiple(
                    pipeline,
                    OUTPUT_ROOT=output_root,
                    COMFY_INPUT=comfy_input,
                    JOBS=jobs,
                ),
                patch.object(
                    pipeline,
                    "audit_uploaded_front_identity_view",
                    return_value=_passed_uploaded_front_view_audit(),
                ),
                patch.object(
                    pipeline,
                    "merge_image_profile_loras",
                    return_value=(profile_loras, "replace", "image_profile"),
                ),
                patch.object(
                    pipeline,
                    "load_subject_profile",
                    side_effect=AssertionError("live profile must not be re-read"),
                ),
                patch.object(
                    pipeline,
                    "prepare_eating_qwen_geometry_guide",
                    side_effect=_write_fake_geometry_guide,
                ),
                patch.object(pipeline, "run_job", side_effect=fake_opaque_qwen),
            ):
                with self.assertRaisesRegex(RuntimeError, "opaque"):
                    pipeline.prepare_uploaded_subject_qwen_references(
                        job_id,
                        ["eating"],
                    )

    def test_uploaded_grooming_transparent_selects_validated_action_scope(self) -> None:
        scope = pipeline.select_final_video_fur_surface_comparison_scope(
            action="grooming",
            action_config={"reference_kind": "qwen_generated_upload"},
            output_format="transparent_gif",
            reference_policy="qwen_action_specific_identity",
            pose_metadata={"scene_prop_separated": False},
        )
        self.assertEqual(
            scope,
            pipeline.FINAL_VIDEO_VALIDATED_ACTION_SURFACE_SCOPE,
        )

    def test_uploaded_grooming_transparent_requires_no_prop_contract(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "no-prop"):
            pipeline.select_final_video_fur_surface_comparison_scope(
                action="grooming",
                action_config={"reference_kind": "qwen_generated_upload"},
                output_format="transparent_gif",
                reference_policy="qwen_action_specific_identity",
                pose_metadata={"scene_prop_separated": True},
            )

    def test_uploaded_eating_transparent_keeps_separated_prop_contract(self) -> None:
        scope = pipeline.select_final_video_fur_surface_comparison_scope(
            action="eating",
            action_config={
                "reference_kind": "qwen_generated_upload",
                "fur_silhouette_gate": {"enabled": True},
            },
            output_format="transparent_gif",
            reference_policy="qwen_action_identity_with_bowl_mapped_red",
            pose_metadata={"scene_prop_separated": True},
        )
        self.assertEqual(
            scope,
            pipeline.FINAL_VIDEO_VALIDATED_ACTION_SURFACE_SCOPE,
        )

    def test_final_video_v2_forwards_exact_qwen_and_mask_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "video.mp4"
            profile = root / "profile.png"
            reference = root / "pose_01_keyframe_transparent.png"
            pose_mask = root / "tm_identity_mask.mp4"
            reference_mask = root / "reference_identity_mask.png"
            video.write_bytes(b"video")
            pose_mask.write_bytes(b"pose-mask")
            Image.new("RGBA", (32, 32), (80, 80, 80, 255)).save(profile)
            Image.new("RGBA", (32, 32), (90, 90, 90, 255)).save(reference)
            Image.new("RGB", (32, 32), (0, 0, 0)).save(reference_mask)
            reference_sha = pipeline.file_sha256(reference)
            pose_mask_sha = pipeline.file_sha256(pose_mask)
            reference_mask_sha = pipeline.file_sha256(reference_mask)
            selected_gate = {
                "acceptable": True,
                "verdict": "accepted_uploaded_identity_absolute_contract",
                "selected_candidate_sha256": reference_sha,
                "policy": {
                    "name": "validated_action_reference_fur_surface_v1",
                    "gate_version": pipeline.FUR_SURFACE_GATE_VERSION,
                    "threshold_version": pipeline.FUR_SURFACE_THRESHOLD_VERSION,
                    "validated_baseline_required_for_acceptance": False,
                    "uploaded_identity_absolute_contract_enabled": True,
                },
                "gate": {
                    "action": "eating",
                    "candidate_sha256": reference_sha,
                    "candidate_comparison_mode": (
                        "uploaded_identity_absolute_surface_contract_v1"
                    ),
                    "profile_pixels_used_for_candidate_appearance_validation": False,
                },
            }
            effective_config = {
                "actions": {
                    "eating": {
                        "reference": str(reference),
                        "reference_sha256": reference_sha,
                        "reference_kind": "qwen_generated_upload",
                    }
                }
            }
            audit = Mock(return_value={"passed": True, "policy": {}})
            with (
                patch.object(
                    pipeline,
                    "resolve_main_case_fur_surface_baseline",
                    return_value=(
                        profile,
                        {
                            "snapshot_sha256": pipeline.file_sha256(profile),
                            "provenance": {"profile_id": "test"},
                        },
                    ),
                ),
                patch.object(pipeline, "audit_final_video", audit),
            ):
                result = pipeline.evaluate_final_video_fur_surface(
                    video,
                    action="eating",
                    effective_subject_config=effective_config,
                    comparison_scope=(
                        pipeline.FINAL_VIDEO_VALIDATED_ACTION_SURFACE_SCOPE
                    ),
                    validated_action_reference_path=reference,
                    validated_action_reference_sha256=reference_sha,
                    validated_action_reference_gate=selected_gate,
                    reference_policy="qwen_action_identity_with_bowl_mapped_red",
                    pose_identity_mask_video_path=pose_mask,
                    pose_identity_mask_video_sha256=pose_mask_sha,
                    reference_identity_mask_path=reference_mask,
                    reference_identity_mask_sha256=reference_mask_sha,
                )
            self.assertTrue(result["passed"])
            forwarded = audit.call_args.kwargs
            self.assertEqual(
                forwarded["comparison_scope"],
                pipeline.FINAL_VIDEO_VALIDATED_ACTION_SURFACE_SCOPE,
            )
            self.assertEqual(forwarded["validated_action_reference_path"], reference)
            self.assertEqual(
                forwarded["validated_action_reference_sha256"], reference_sha
            )
            self.assertEqual(
                forwarded["pose_identity_mask_video_sha256"], pose_mask_sha
            )
            self.assertEqual(
                forwarded["reference_identity_mask_sha256"], reference_mask_sha
            )
            self.assertTrue(
                forwarded["validated_action_reference_provenance"][
                    "pre_scail_revalidated"
                ]
            )

    def test_final_video_v2_accepts_grooming_qwen_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "video.mp4"
            profile = root / "profile.png"
            reference = root / "pose_01_keyframe_transparent.png"
            pose_mask = root / "tm_identity_mask.mp4"
            reference_mask = root / "reference_identity_mask.png"
            video.write_bytes(b"video")
            pose_mask.write_bytes(b"pose-mask")
            Image.new("RGBA", (32, 32), (80, 80, 80, 255)).save(profile)
            Image.new("RGBA", (32, 32), (90, 90, 90, 255)).save(reference)
            Image.new("RGB", (32, 32), (0, 0, 0)).save(reference_mask)
            reference_sha = pipeline.file_sha256(reference)
            pose_mask_sha = pipeline.file_sha256(pose_mask)
            reference_mask_sha = pipeline.file_sha256(reference_mask)
            selected_gate = {
                "acceptable": True,
                "verdict": "accepted_uploaded_identity_absolute_contract",
                "selected_candidate_sha256": reference_sha,
                "policy": {
                    "name": "validated_action_reference_fur_surface_v1",
                    "gate_version": pipeline.FUR_SURFACE_GATE_VERSION,
                    "threshold_version": pipeline.FUR_SURFACE_THRESHOLD_VERSION,
                    "validated_baseline_required_for_acceptance": False,
                    "uploaded_identity_absolute_contract_enabled": True,
                },
                "gate": {
                    "action": "grooming",
                    "candidate_sha256": reference_sha,
                    "candidate_comparison_mode": (
                        "uploaded_identity_absolute_surface_contract_v1"
                    ),
                    "profile_pixels_used_for_candidate_appearance_validation": False,
                },
            }
            composite_recovery = {
                "policy": (pipeline.GROOMING_QWEN_FORELEG_REPAIR_COMPOSITE_GATE_POLICY),
                "evaluated": True,
                "accepted": True,
                "fail_closed": True,
                "global_pose_threshold_changed": False,
                "required_edge_density_ratio": 0.70,
                "edge_density": 0.079399,
                "edge_density_threshold": 0.11,
                "edge_density_floor": 0.077,
                "repair_noise_mask_sha256": "c" * 64,
                "checks": {
                    "global_pose_gate_contract_valid": True,
                    "raw_issue_contract_valid": True,
                    "edge_density_within_recovery_band": True,
                    "chamfer_geometry_passed": True,
                    "continuous_foreleg_passed": True,
                    "paw_muzzle_contact_passed": True,
                    "raised_planted_paw_color_histogram_passed": True,
                    "raised_planted_paw_texture_contract_passed": True,
                    "independent_fur_surface_gate_passed": True,
                    "masked_pet3d_repair_contract_valid": True,
                },
            }
            effective_config = {
                "actions": {
                    "grooming": {
                        "reference": str(reference),
                        "reference_sha256": reference_sha,
                        "reference_kind": "qwen_generated_upload",
                    }
                }
            }
            audit = Mock(return_value={"passed": True, "policy": {}})
            with (
                patch.object(
                    pipeline,
                    "resolve_main_case_fur_surface_baseline",
                    return_value=(
                        profile,
                        {
                            "snapshot_sha256": pipeline.file_sha256(profile),
                            "provenance": {"profile_id": "test"},
                        },
                    ),
                ),
                patch.object(pipeline, "audit_final_video", audit),
            ):
                result = pipeline.evaluate_final_video_fur_surface(
                    video,
                    action="grooming",
                    effective_subject_config=effective_config,
                    comparison_scope=(
                        pipeline.FINAL_VIDEO_VALIDATED_ACTION_SURFACE_SCOPE
                    ),
                    validated_action_reference_path=reference,
                    validated_action_reference_sha256=reference_sha,
                    validated_action_reference_gate=selected_gate,
                    reference_policy="qwen_action_specific_identity",
                    pose_identity_mask_video_path=pose_mask,
                    pose_identity_mask_video_sha256=pose_mask_sha,
                    reference_identity_mask_path=reference_mask,
                    reference_identity_mask_sha256=reference_mask_sha,
                    grooming_pose_gate_composite_recovery=(composite_recovery),
                )
            self.assertTrue(result["passed"])
            forwarded = audit.call_args.kwargs
            self.assertEqual(forwarded["action"], "grooming")
            self.assertEqual(
                forwarded["comparison_scope"],
                pipeline.FINAL_VIDEO_VALIDATED_ACTION_SURFACE_SCOPE,
            )
            self.assertEqual(
                forwarded["validated_action_reference_provenance"]["reference_policy"],
                "qwen_action_specific_identity",
            )
            self.assertEqual(
                forwarded["validated_action_reference_provenance"][
                    "grooming_pose_gate_composite_recovery"
                ],
                composite_recovery,
            )

    def test_final_video_v2_rejects_divergent_selected_gate_sha(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "video.mp4"
            profile = root / "profile.png"
            reference = root / "reference.png"
            video.write_bytes(b"video")
            Image.new("RGBA", (16, 16), (80, 80, 80, 255)).save(profile)
            Image.new("RGBA", (16, 16), (90, 90, 90, 255)).save(reference)
            reference_sha = pipeline.file_sha256(reference)
            effective_config = {
                "actions": {
                    "eating": {
                        "reference": str(reference),
                        "reference_sha256": reference_sha,
                        "reference_kind": "qwen_generated_upload",
                    }
                }
            }
            with patch.object(
                pipeline,
                "resolve_main_case_fur_surface_baseline",
                return_value=(
                    profile,
                    {
                        "snapshot_sha256": pipeline.file_sha256(profile),
                        "provenance": {},
                    },
                ),
            ):
                result = pipeline.evaluate_final_video_fur_surface(
                    video,
                    action="eating",
                    effective_subject_config=effective_config,
                    comparison_scope=(
                        pipeline.FINAL_VIDEO_VALIDATED_ACTION_SURFACE_SCOPE
                    ),
                    validated_action_reference_path=reference,
                    validated_action_reference_sha256=reference_sha,
                    validated_action_reference_gate={
                        "selected_candidate_sha256": "0" * 64,
                        "gate": {"candidate_sha256": reference_sha},
                    },
                    reference_policy="qwen_action_identity_with_bowl_mapped_red",
                )
            self.assertFalse(result["passed"])
            self.assertIn("provenance changed", result["error"])


if __name__ == "__main__":
    unittest.main()
