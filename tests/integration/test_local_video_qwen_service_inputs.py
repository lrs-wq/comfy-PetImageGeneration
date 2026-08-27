from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image, ImageDraw

import fur_surface_gate as surface_gate
import local_video_api as pipeline

from tests.integration.local_video_subject_replacement_test_support import (
    SubjectReplacementQwenReferenceTestBase,
    _subject_profile_v2,
    _write_eating_contact_fixture,
    _write_thin_warm_rim_beige_bowl_fixture,
)


class SubjectReplacementQwenServiceAndInputTests(SubjectReplacementQwenReferenceTestBase):
    def test_health_exposes_v22032_qwen_full_body_input_contract(self) -> None:
        effective_config = {
            "engine_profile": "scail2-14b-fp8-v1",
            "prompt_profile": "sleek-tabby-scail-v1",
            "seed_mode": "base_plus_action_offset",
            "frames": 81,
            "fps": 16.0,
            "steps": 6,
            "actions": {
                "eating": {
                    "driving_animation": "eating_driver.mp4",
                    "pose_strength": 1.1,
                    "seed_offset": 0,
                    "effective_seed": 20260726,
                },
                "grooming": {
                    "driving_animation": "grooming_driver.mp4",
                    "pose_strength": 1.15,
                    "seed_offset": 2,
                    "effective_seed": 20260728,
                },
            },
        }
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "queue_running": [],
            "queue_pending": [],
        }

        with (
            patch.multiple(
                pipeline,
                API_KEY="test-key",
                API_VERSION="2.21.4",
                IMAGE_PROFILES={},
                DEFAULT_IMAGE_PROFILE="",
                PROMPT_TRANSLATION_ENABLED=False,
                QWEN_POSE_MAX_ATTEMPTS=3,
                QWEN_BOWL_POSE_EXTRA_ATTEMPTS=2,
                GENERATION_VISUAL_AUDITS_BLOCKING=True,
            ),
            patch.object(pipeline.requests, "get", return_value=response),
            patch.object(pipeline, "required_model_paths", return_value=[]),
            patch.object(
                pipeline,
                "prompt_translator_missing_paths",
                return_value=[],
            ),
            patch.object(pipeline, "load_subject_profile", return_value={}),
            patch.object(
                pipeline,
                "resolve_effective_subject_config",
                return_value=effective_config,
            ),
            patch.object(pipeline, "require_subject_replacement_assets"),
            patch.object(
                pipeline,
                "subject_profile_public",
                return_value={"id": "sleek_tabby_cat_v1"},
            ),
        ):
            payload = pipeline.health()

        self.assertEqual(payload["version"], "2.21.4")
        self.assertFalse(payload["subject_replacement"]["ok"])
        self.assertEqual(
            payload["subject_replacement"]["upload_validation"]["status"],
            "failed",
        )
        guard = payload["qwen_style"]["pose_guard"]
        self.assertEqual(
            guard["reference_pipeline"],
            pipeline.QWEN_FULL_BODY_FIRST_FRAME_PIPELINE,
        )
        self.assertEqual(
            guard["upload_pose_reference_source"],
            "per_action_qwen_canonical_and_main_case_geometry_roles",
        )
        self.assertEqual(
            guard["geometry_generator_policy"],
            {
                "eating": "driving_eating_geometry_image2_v6",
                "grooming": pipeline.GROOMING_QWEN_GEOMETRY_GUIDE_POLICY,
            },
        )
        self.assertEqual(
            guard["grooming_geometry_isolation"]["qwen_foreground_rgb"],
            list(pipeline.GROOMING_QWEN_GUIDE_FOREGROUND_RGB),
        )
        self.assertEqual(
            guard["grooming_geometry_isolation"]["qwen_foreground_palette_rgb"],
            [list(pipeline.GROOMING_QWEN_GUIDE_FOREGROUND_RGB)],
        )
        self.assertFalse(
            guard["grooming_geometry_isolation"]["qwen_internal_luma_variation"]
        )
        self.assertEqual(
            guard["grooming_geometry_isolation"]["qwen_internal_luma_variation_role"],
            "none",
        )
        self.assertFalse(
            guard["grooming_geometry_isolation"][
                "qwen_appearance_bearing_internal_luma_variation"
            ]
        )
        self.assertFalse(
            guard["grooming_geometry_isolation"]["qwen_synthetic_foreleg_semantic"]
        )
        self.assertEqual(
            guard["grooming_geometry_isolation"][
                "qwen_synthetic_foreleg_semantic_method"
            ],
            None,
        )
        self.assertTrue(guard["grooming_geometry_isolation"]["qwen_foreleg_repair"])
        self.assertTrue(
            guard["grooming_geometry_isolation"][
                "qwen_foreleg_repair_retained_for_legacy_jobs_only"
            ]
        )
        self.assertEqual(
            guard["grooming_geometry_isolation"]["qwen_foreleg_repair_policy"],
            pipeline.GROOMING_CANONICAL_FORELEG_COMPLETION_POLICY,
        )
        self.assertEqual(
            guard["grooming_geometry_isolation"]["qwen_foreleg_repair_attempt"],
            pipeline.GROOMING_QWEN_FORELEG_REPAIR_ATTEMPT,
        )
        self.assertEqual(
            guard["grooming_geometry_isolation"]["qwen_foreleg_repair_denoise"],
            pipeline.GROOMING_QWEN_FORELEG_REPAIR_DENOISE,
        )
        self.assertFalse(
            guard["grooming_geometry_isolation"][
                "qwen_foreleg_repair_synthetic_capsule_exposed"
            ]
        )
        self.assertTrue(
            guard["grooming_geometry_isolation"][
                "qwen_foreleg_repair_open_channel_anatomy_image3"
            ]
        )
        self.assertFalse(
            guard["grooming_geometry_isolation"][
                "qwen_foreleg_repair_private_cv_luma_image3"
            ]
        )
        self.assertEqual(
            guard["grooming_geometry_isolation"][
                "qwen_foreleg_repair_mask_dilation_px"
            ],
            pipeline.GROOMING_QWEN_FORELEG_REPAIR_MASK_DILATION_PX,
        )
        self.assertEqual(
            guard["grooming_geometry_isolation"]["qwen_foreleg_repair_anatomy_policy"],
            pipeline.GROOMING_QWEN_FORELEG_REPAIR_ANATOMY_POLICY,
        )
        self.assertEqual(
            guard["grooming_geometry_isolation"]["qwen_foreleg_repair_seed_source"],
            "identity_safe_candidate_unchanged_before_masked_sampling",
        )
        self.assertIsNone(
            guard["grooming_geometry_isolation"][
                "qwen_foreleg_repair_seed_lowpass_sigma_px"
            ]
        )
        self.assertIsNone(
            guard["grooming_geometry_isolation"][
                "qwen_foreleg_repair_seed_identity_median_blend"
            ]
        )
        self.assertFalse(
            guard["grooming_geometry_isolation"][
                "qwen_foreleg_repair_identity_or_geometry_seed_inserted"
            ]
        )
        self.assertEqual(
            guard["grooming_geometry_isolation"][
                "qwen_foreleg_repair_noise_mask_expansion_px"
            ],
            pipeline.GROOMING_QWEN_FORELEG_REPAIR_NOISE_MASK_EXPANSION_PX,
        )
        self.assertTrue(
            guard["grooming_geometry_isolation"][
                "qwen_foreleg_repair_expanded_context_full_denoise_mask"
            ]
        )
        self.assertEqual(
            guard["grooming_geometry_isolation"]["qwen_foreleg_repair_noise_mask_node"],
            "SetLatentNoiseMask",
        )
        self.assertEqual(
            guard["grooming_geometry_isolation"]["qwen_foreleg_repair_style_policy"],
            "disabled_for_canonical_masked_completion",
        )
        self.assertIsNone(
            guard["grooming_geometry_isolation"]["qwen_foreleg_repair_style_lora"]
        )
        self.assertEqual(
            guard["grooming_geometry_isolation"]["qwen_foreleg_repair_style_strength"],
            0.0,
        )
        self.assertEqual(
            guard["grooming_geometry_isolation"]["qwen_foreleg_repair_style_scope"],
            "none",
        )
        self.assertFalse(
            guard["grooming_geometry_isolation"][
                "qwen_foreleg_repair_style_identity_authority"
            ]
        )
        self.assertTrue(
            guard["grooming_geometry_isolation"][
                "qwen_foreleg_repair_style_fail_closed"
            ]
        )
        self.assertIsNone(
            guard["grooming_geometry_isolation"][
                "qwen_foreleg_repair_composite_gate_policy"
            ]
        )
        self.assertFalse(
            guard["grooming_geometry_isolation"][
                "qwen_foreleg_repair_composite_recovery_enabled"
            ]
        )
        self.assertIsNone(
            guard["grooming_geometry_isolation"][
                "qwen_foreleg_repair_composite_edge_density_ratio_min"
            ]
        )
        self.assertIsNone(
            guard["grooming_geometry_isolation"][
                "qwen_foreleg_repair_composite_planted_paw_gradient_p90_max"
            ]
        )
        self.assertIsNone(
            guard["grooming_geometry_isolation"][
                "qwen_foreleg_repair_composite_raised_paw_gradient_p90_max"
            ]
        )
        self.assertEqual(
            guard["grooming_geometry_isolation"][
                "canonical_completion_edge_evidence_policy"
            ],
            pipeline.GROOMING_CANONICAL_COMPLETION_EDGE_EVIDENCE_POLICY,
        )
        self.assertEqual(
            guard["grooming_geometry_isolation"][
                "canonical_completion_edge_density_ratio_min"
            ],
            pipeline.GROOMING_CANONICAL_COMPLETION_EDGE_DENSITY_RATIO_MIN,
        )
        self.assertTrue(
            guard["grooming_geometry_isolation"][
                "canonical_completion_independent_fur_surface_required"
            ]
        )
        self.assertTrue(
            guard["grooming_geometry_isolation"]["canonical_full_pose_recovery_enabled"]
        )
        self.assertEqual(
            guard["grooming_geometry_isolation"]["canonical_full_pose_rescue_attempt"],
            pipeline.GROOMING_CANONICAL_FULL_POSE_RESCUE_ATTEMPT,
        )
        self.assertEqual(
            guard["grooming_geometry_isolation"][
                "canonical_full_pose_identity_refine_attempt"
            ],
            pipeline.GROOMING_CANONICAL_FULL_POSE_REFINE_ATTEMPT,
        )
        self.assertIsNone(
            guard["grooming_geometry_isolation"][
                "canonical_full_pose_recovery_style_lora"
            ]
        )
        self.assertFalse(
            guard["grooming_geometry_isolation"][
                "canonical_full_pose_recovery_global_thresholds_changed"
            ]
        )
        self.assertEqual(
            guard["grooming_geometry_isolation"][
                "qwen_foreleg_repair_global_edge_density_min"
            ],
            0.11,
        )
        self.assertFalse(
            guard["grooming_geometry_isolation"][
                "qwen_foreleg_repair_global_pose_threshold_changed"
            ]
        )
        composite_recovery = guard["grooming_pose_gate"][
            "masked_pet3d_composite_recovery"
        ]
        self.assertFalse(composite_recovery["production_enabled"])
        self.assertTrue(composite_recovery["legacy_geometry_first_jobs_only"])
        self.assertEqual(
            composite_recovery["policy"],
            pipeline.GROOMING_QWEN_FORELEG_REPAIR_COMPOSITE_GATE_POLICY,
        )
        self.assertEqual(composite_recovery["attempt"], 3)
        self.assertEqual(composite_recovery["edge_density_ratio_min"], 0.70)
        self.assertTrue(composite_recovery["requires_edge_density_issue"])
        self.assertTrue(
            composite_recovery[
                "allows_only_optional_low_texture_planted_paw_ratio_issue"
            ]
        )
        self.assertEqual(
            composite_recovery["planted_paw_gradient_p90_max"],
            8.0,
        )
        self.assertEqual(
            composite_recovery["raised_paw_gradient_p90_max"],
            128.0,
        )
        self.assertTrue(
            composite_recovery["requires_geometry_contact_color_histogram_pass"]
        )
        self.assertTrue(composite_recovery["requires_independent_fur_surface_pass"])
        self.assertTrue(
            composite_recovery["requires_digest_bound_noise_masked_pet3d_repair"]
        )
        self.assertTrue(composite_recovery["fail_closed"])
        self.assertEqual(guard["grooming_pose_gate"]["global_edge_density_min"], 0.11)
        self.assertTrue(
            guard["grooming_pose_gate"]["global_edge_density_min_unchanged"]
        )
        self.assertEqual(
            guard["grooming_pose_gate"]["structural_edge_policy"],
            "gaussian_lowpass_organic_foreleg_structure_v1",
        )
        self.assertEqual(
            guard["grooming_pose_gate"]["structural_blur_sigma"],
            4.0,
        )
        self.assertEqual(
            guard["grooming_pose_gate"]["structural_edge_density_min"],
            0.015,
        )
        self.assertTrue(
            guard["grooming_pose_gate"]["fine_fur_edges_cannot_satisfy_structural_gate"]
        )
        direct_evidence = guard["grooming_pose_gate"][
            "canonical_direct_structural_edge_evidence"
        ]
        self.assertEqual(
            direct_evidence["policy"],
            pipeline.GROOMING_CANONICAL_DIRECT_STRUCTURAL_EDGE_EVIDENCE_POLICY,
        )
        self.assertEqual(direct_evidence["maximum_attempt"], 2)
        self.assertEqual(
            direct_evidence["raw_edge_density_ratio_min"],
            0.925,
        )
        self.assertEqual(
            direct_evidence["structural_edge_density_min"],
            0.03,
        )
        self.assertEqual(
            direct_evidence["strict_chamfer_coverage_min"],
            0.70,
        )
        self.assertEqual(
            direct_evidence["strict_chamfer_mean_distance_max"],
            5.0,
        )
        self.assertEqual(
            direct_evidence["strict_chamfer_p90_distance_max"],
            12.0,
        )
        self.assertTrue(direct_evidence["rejects_historical_straight_prop_candidates"])
        self.assertTrue(direct_evidence["identity_palette_gate_still_required"])
        self.assertTrue(direct_evidence["independent_fur_surface_gate_still_required"])
        self.assertFalse(direct_evidence["global_pose_threshold_changed"])
        self.assertTrue(direct_evidence["fail_closed"])
        surface_nonexpansion = guard["grooming_pose_gate"][
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
        self.assertEqual(surface_nonexpansion["coherent_fraction_max"], 0.57)
        self.assertEqual(
            surface_nonexpansion["coherent_fraction_delta_max"],
            0.01,
        )
        self.assertTrue(
            surface_nonexpansion["independent_final_video_fur_audit_still_required"]
        )
        self.assertFalse(surface_nonexpansion["global_surface_thresholds_changed"])
        self.assertTrue(surface_nonexpansion["fail_closed"])
        self.assertFalse(
            guard["grooming_geometry_isolation"]["main_case_appearance_exposed_to_qwen"]
        )
        self.assertFalse(
            guard["grooming_geometry_isolation"][
                "private_cv_gate_exposed_to_qwen_initial_attempts"
            ]
        )
        self.assertFalse(
            guard["grooming_geometry_isolation"][
                "private_cv_gate_neutral_copy_exposed_on_repair_attempt"
            ]
        )
        self.assertTrue(
            guard["grooming_geometry_isolation"][
                "public_open_channel_copy_exposed_on_repair_attempt"
            ]
        )
        self.assertEqual(
            guard["grooming_geometry_isolation"]["private_completion_mask_policy"],
            pipeline.GROOMING_PRIVATE_COMPLETION_MASK_POLICY,
        )
        self.assertFalse(
            guard["grooming_geometry_isolation"][
                "private_completion_mask_exposed_to_qwen"
            ]
        )
        self.assertFalse(
            guard["grooming_geometry_isolation"][
                "private_completion_mask_loaded_as_qwen_image"
            ]
        )
        self.assertEqual(
            guard["pose_identity_input_role"],
            {
                "eating": "image1_qwen_full_body_canonical",
                "grooming": "image1_qwen_full_body_canonical",
            },
        )
        self.assertFalse(guard["generated_canonical_bypassed"])
        self.assertTrue(guard["canonical_reference_generated"])
        self.assertTrue(guard["canonical_reference_used"])
        self.assertFalse(guard["raw_upload_sent_to_scail"])
        self.assertEqual(guard["identity_upload_view_counts"], [1, 3])
        self.assertEqual(
            guard["identity_upload_three_view_roles"],
            ["front", "left", "right"],
        )
        self.assertTrue(guard["qwen_canonical_multi_view_supported"])
        self.assertEqual(
            guard["qwen_canonical_multi_view_node_inputs"],
            ["image1", "image2", "image3"],
        )
        self.assertTrue(guard["qwen_canonical_image1_is_sampler_latent"])
        self.assertFalse(guard["qwen_action_raw_identity_views_used"])
        self.assertEqual(
            guard["qwen_action_input_contract"],
            {
                "eating": "canonical_image1_plus_geometry_image2",
                "grooming": "canonical_image1_plus_geometry_image2",
            },
        )
        canonical_action_palette_gate = guard["canonical_action_identity_palette_gate"]
        self.assertTrue(canonical_action_palette_gate["enabled"])
        self.assertEqual(
            canonical_action_palette_gate["policy"],
            pipeline.CANONICAL_ACTION_IDENTITY_PALETTE_POLICY,
        )
        self.assertEqual(
            canonical_action_palette_gate["achromatic_balance_l1_max"],
            pipeline.CANONICAL_ACTION_ACHROMATIC_BALANCE_L1_MAX,
        )
        self.assertEqual(
            canonical_action_palette_gate["light_identity_black_fraction_max"],
            pipeline.CANONICAL_ACTION_LIGHT_IDENTITY_BLACK_FRACTION_MAX,
        )
        self.assertEqual(
            canonical_action_palette_gate["dark_feature_expansion_max"],
            pipeline.CANONICAL_ACTION_DARK_FEATURE_EXPANSION_MAX,
        )
        self.assertTrue(canonical_action_palette_gate["fail_closed"])
        self.assertTrue(
            guard["grooming_geometry_isolation"]["qwen_structural_negative_space"]
        )
        self.assertEqual(
            guard["grooming_geometry_isolation"][
                "qwen_structural_negative_space_method"
            ],
            pipeline.GROOMING_QWEN_NEGATIVE_SPACE_METHOD,
        )
        self.assertFalse(
            guard["grooming_geometry_isolation"]["qwen_closed_foreleg_capsule_exposed"]
        )
        self.assertTrue(
            guard["grooming_geometry_isolation"][
                "qwen_internal_green_separator_exposed"
            ]
        )
        self.assertEqual(
            guard["grooming_geometry_isolation"]["qwen_raised_foreleg_validation_role"],
            "public_open_curve_geometry_plus_private_cv",
        )
        self.assertEqual(
            guard["grooming_geometry_isolation"]["qwen_role"],
            "image2_action_geometry_guide",
        )
        self.assertTrue(guard["uploaded_identity_appearance_authority"])
        self.assertEqual(
            guard["subject_profile_appearance_role"],
            "action_geometry_only",
        )
        self.assertEqual(guard["canonical_palette_input"], "final_alpha_cutout")
        self.assertEqual(
            guard["canonical_upper_body_palette_reference_policy"],
            pipeline.CANONICAL_UPPER_BODY_PALETTE_REFERENCE_POLICY,
        )
        self.assertTrue(guard["canonical_raw_palette_retained_for_audit"])
        canonical_palette_gate = guard["canonical_identity_palette_gate"]
        self.assertTrue(canonical_palette_gate["enabled"])
        self.assertTrue(
            canonical_palette_gate["fail_closed_for_generated_candidates"]
        )
        self.assertEqual(
            canonical_palette_gate["terminal_recovery"],
            "original_uploaded_identity_safe_matte",
        )
        self.assertEqual(
            guard["canonical_chroma_despill"],
            "alpha_boundary_plus_source_palette_guarded_interior_v1",
        )
        self.assertEqual(
            guard["canonical_chroma_spill_policy"],
            pipeline.UPLOADED_IDENTITY_CHROMA_SPILL_POLICY,
        )
        self.assertTrue(guard["canonical_chroma_spill_preserves_paired_eye_accents"])
        self.assertTrue(guard["canonical_interior_hole_gate"])
        self.assertEqual(
            guard["upload_pose_photography_lora_policy"],
            "disabled_after_canonical_identity_style_lock_v1",
        )
        self.assertEqual(
            guard["upload_pose_builtin_style_lora_policy"],
            "disabled_for_canonical_actions_v2",
        )
        self.assertEqual(
            guard["upload_pose_lora_policy"],
            "qwen_lightning_only_with_open_channel_identity_refine_v6",
        )
        eating_refine = guard["eating_identity_refine"]
        self.assertTrue(eating_refine["enabled"])
        self.assertEqual(
            eating_refine["policy"],
            pipeline.EATING_CANONICAL_IDENTITY_REFINE_POLICY,
        )
        self.assertEqual(
            eating_refine["attempt"],
            pipeline.EATING_CANONICAL_IDENTITY_REFINE_ATTEMPT,
        )
        self.assertEqual(
            eating_refine["starts_at_attempt"],
            pipeline.EATING_CANONICAL_IDENTITY_REFINE_ATTEMPT,
        )
        self.assertTrue(eating_refine["persists_for_remaining_attempts"])
        self.assertTrue(eating_refine["generic_full_redraw_forbidden_after_source"])
        self.assertTrue(eating_refine["qwen_lightning_only"])
        self.assertIsNone(eating_refine["style_lora"])
        self.assertIsNone(eating_refine["sampler_noise_mask"])
        self.assertFalse(eating_refine["global_thresholds_changed"])
        self.assertEqual(
            guard["upload_pose_candidate_normalization_policy"],
            pipeline.QWEN_ACTION_REFERENCE_NORMALIZATION_POLICY,
        )
        self.assertEqual(
            guard["upload_pose_candidate_matte_policy"],
            pipeline.QWEN_ACTION_CANDIDATE_MATTE_POLICY,
        )
        self.assertEqual(
            guard["eating_mouth_contact_gate"]["policy"],
            pipeline.EATING_MOUTH_BOWL_CONTACT_POLICY,
        )
        self.assertEqual(
            guard["eating_mouth_contact_gate"]["component_width_bowl_ratio_max"],
            0.70,
        )
        self.assertEqual(
            guard["eating_mouth_contact_gate"]["gap_px_max"],
            8,
        )
        self.assertEqual(
            guard["normalization_contract"]["action_keyframes"],
            pipeline.QWEN_ACTION_REFERENCE_NORMALIZATION_POLICY,
        )
        self.assertTrue(
            guard["normalization_contract"][
                "selected_candidate_byte_copied_to_scail_reference"
            ]
        )
        self.assertTrue(
            guard["normalization_contract"]["full_body_normalized_transparent_required"]
        )
        self.assertTrue(
            guard["normalization_contract"][
                "action_keyframes_normalized_transparent_required"
            ]
        )
        self.assertEqual(
            guard["pose_identity_palette_reference_role"]["eating"],
            "uploaded_identity_source",
        )
        self.assertEqual(
            guard["legacy_profile_action_reference_used"],
            {"eating": False, "grooming": False},
        )
        self.assertFalse(guard["profile_pixels_used_as_identity_or_scail_reference"])
        self.assertFalse(guard["profile_pixels_used_as_validation_baseline"])
        self.assertTrue(guard["uploaded_identity_absolute_surface_gate_used"])
        self.assertEqual(
            guard["surface_candidate_comparison_mode"],
            pipeline.UPLOADED_IDENTITY_SURFACE_COMPARISON_MODE,
        )
        self.assertEqual(guard["max_attempts"], 3)
        self.assertEqual(guard["bowl_extra_attempts"], 2)
        self.assertEqual(guard["bowl_max_attempts"], 5)

    def test_health_reports_ready_upload_surface_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = _subject_profile_v2(root)
            ffmpeg = root / "ffmpeg"
            ffmpeg.write_bytes(b"test")
            response = Mock()
            response.raise_for_status.return_value = None
            response.json.return_value = {
                "queue_running": [],
                "queue_pending": [],
            }
            surface_result = {
                "acceptable": True,
                "issues": [],
                "candidate": {"metrics": {}},
            }

            with (
                patch.multiple(
                    pipeline,
                    API_KEY="test-key",
                    API_VERSION="2.21.4",
                    IMAGE_PROFILES={},
                    DEFAULT_IMAGE_PROFILE="",
                    PROMPT_TRANSLATION_ENABLED=False,
                    QWEN_POSE_MAX_ATTEMPTS=3,
                    QWEN_BOWL_POSE_EXTRA_ATTEMPTS=2,
                    GENERATION_VISUAL_AUDITS_BLOCKING=True,
                    FFMPEG_BIN=str(ffmpeg),
                ),
                patch.object(pipeline.requests, "get", return_value=response),
                patch.object(pipeline, "required_model_paths", return_value=[]),
                patch.object(
                    pipeline,
                    "prompt_translator_missing_paths",
                    return_value=[],
                ),
                patch.object(
                    pipeline,
                    "load_subject_profile",
                    return_value=profile,
                ),
                patch.object(pipeline, "require_subject_replacement_assets"),
                patch.object(
                    pipeline,
                    "evaluate_fur_surface",
                    return_value=surface_result,
                ),
                patch.object(
                    pipeline,
                    "subject_profile_public",
                    return_value={"id": "sleek_tabby_cat_v1"},
                ),
            ):
                payload = pipeline.health()

        replacement = payload["subject_replacement"]
        self.assertTrue(replacement["ok"])
        self.assertEqual(
            replacement["upload_validation"]["status"],
            "ready",
        )
        self.assertEqual(
            replacement["upload_validation"]["expected_surface_gate_actions"],
            ["eating", "grooming"],
        )
        prop_split = replacement["upload_validation"][
            "eating_reference_prop_split"
        ]
        self.assertEqual(
            prop_split["policy"],
            pipeline.SCAIL_EATING_REFERENCE_PROP_SPLIT_POLICY,
        )
        self.assertTrue(prop_split["strict_visual_split_first"])
        self.assertFalse(prop_split["visual_uncertainty_blocks_job"])
        self.assertEqual(
            prop_split["terminal_fallback"],
            "lower_center_prop_geometry_v1",
        )
        self.assertTrue(prop_split["fallback_is_audited"])
        self.assertTrue(prop_split["empty_or_tiny_subject_remains_runtime_error"])
        self.assertEqual(
            payload["qwen_style"]["pose_guard"]["fur_surface_gate"]["status"],
            "passed",
        )
        surface_gate = payload["qwen_style"]["pose_guard"]["fur_surface_gate"]
        self.assertEqual(
            surface_gate["retry_policy"],
            pipeline.FUR_SURFACE_GATE_RETRY_POLICY,
        )
        self.assertTrue(surface_gate["preserves_configured_eating_bowl_recovery"])
        self.assertEqual(
            surface_gate["configured_eating_bowl_recovery_attempts"],
            2,
        )
        self.assertEqual(
            surface_gate["configured_eating_bowl_recovery_profile_prefix"],
            "bowl_palette_canvas_seed_retry_",
        )
        self.assertEqual(
            surface_gate["radial_long_line_evidence_policy"],
            "hough_radial_fraction_and_radial_line_density_v1",
        )
        self.assertFalse(surface_gate["radial_ratio_alone_can_reject"])
        self.assertEqual(
            surface_gate["thresholds"]["hough_radial_line_density_min"],
            0.008,
        )
        self.assertTrue(surface_gate["blocks_scail_on_exhaustion"])
        self.assertEqual(
            surface_gate["final_video_uploaded_grooming_scope"],
            pipeline.FINAL_VIDEO_VALIDATED_ACTION_SURFACE_SCOPE,
        )
        self.assertEqual(
            surface_gate["final_video_uploaded_qwen_actions"],
            ["eating", "grooming"],
        )
        self.assertEqual(
            surface_gate["final_video_uploaded_qwen_reference_policies"],
            pipeline.FINAL_VIDEO_VALIDATED_ACTION_REFERENCE_POLICIES,
        )
        self.assertTrue(
            surface_gate["final_video_grooming_no_prop_identity_mask_required_for_v2"]
        )
        self.assertFalse(
            surface_gate[
                "final_video_profile_pixels_used_for_uploaded_identity_appearance_validation"
            ]
        )
        self.assertTrue(
            surface_gate[
                "final_video_uploaded_qwen_action_reference_is_appearance_authority"
            ]
        )
        self.assertTrue(
            surface_gate[
                "final_video_derived_texture_reference_hash_and_alpha_contract_required"
            ]
        )
        self.assertTrue(
            surface_gate[
                "final_video_derived_texture_baseline_absolute_observations_nonblocking"
            ]
        )
        self.assertTrue(
            surface_gate[
                "final_video_unmasked_action_reference_baseline_must_pass_current_absolute_contract"
            ]
        )
        self.assertTrue(
            surface_gate["final_video_malformed_evaluator_contract_fails_closed"]
        )
        self.assertTrue(
            surface_gate[
                "final_video_coherent_brush_requires_gradient_weighted_load_or_long_line_energy_backstop"
            ]
        )
        self.assertEqual(
            surface_gate["final_video_coherent_brush_load_proxy"],
            "structure_tensor_coherent_fraction_x_coarse_gradient_p90",
        )
        self.assertEqual(
            surface_gate["final_video_coherent_brush_load_only_temporal_min_samples"],
            2,
        )
        self.assertTrue(
            surface_gate[
                "final_video_coherent_brush_load_only_single_sample_is_observation"
            ]
        )
        self.assertTrue(
            surface_gate[
                "final_video_coherent_brush_long_line_energy_backstop_remains_single_sample_blocking"
            ]
        )
        self.assertTrue(
            surface_gate[
                "final_video_absolute_coherent_brush_issue_alone_is_observation"
            ]
        )
        self.assertTrue(
            surface_gate[
                "final_video_long_fur_requires_contour_corroboration_or_strong_expansion"
            ]
        )
        self.assertTrue(
            surface_gate[
                "final_video_long_fur_texture_only_absolute_and_dual_surface_evidence_is_observation"
            ]
        )
        self.assertTrue(
            surface_gate[
                "final_video_eating_isolated_strong_long_fur_temporal_corroboration"
            ]
        )
        self.assertTrue(
            surface_gate[
                "final_video_eating_isolated_long_fur_requires_single_interior_blocking_sample"
            ]
        )
        self.assertTrue(
            surface_gate[
                "final_video_eating_isolated_long_fur_records_blocking_frame_unmasked_corroboration"
            ]
        )
        self.assertTrue(
            surface_gate[
                "final_video_eating_isolated_long_fur_requires_adjacent_prop_excluded_observation"
            ]
        )
        self.assertTrue(
            surface_gate[
                "final_video_eating_isolated_long_fur_allows_one_initial_radial_companion_for_later_validation"
            ]
        )
        self.assertEqual(
            surface_gate["final_video_eating_isolated_long_fur_minimum_samples"],
            5,
        )
        self.assertTrue(
            surface_gate[
                "final_video_eating_isolated_long_fur_reuses_original_neighbor_ratio_and_delta_thresholds"
            ]
        )
        self.assertTrue(
            surface_gate[
                "final_video_eating_isolated_long_fur_strong_ceiling_uses_one_existing_texture_delta_margin"
            ]
        )
        self.assertTrue(
            surface_gate[
                "final_video_eating_isolated_long_fur_preserves_absolute_contour_radial_coherent_and_unknown_fail_closed"
            ]
        )
        self.assertTrue(
            surface_gate[
                "final_video_eating_initial_radial_loop_endpoint_corroboration"
            ]
        )
        self.assertTrue(
            surface_gate[
                "final_video_eating_initial_radial_requires_single_initial_blocking_sample"
            ]
        )
        self.assertTrue(
            surface_gate[
                "final_video_eating_initial_radial_requires_clean_terminal_and_all_interior_samples"
            ]
        )
        self.assertTrue(
            surface_gate[
                "final_video_eating_initial_radial_requires_hash_bound_radial_long_lines_reference_observation"
            ]
        )
        self.assertEqual(
            surface_gate["final_video_eating_initial_radial_boundary_minimum_samples"],
            5,
        )
        self.assertEqual(
            surface_gate[
                "final_video_eating_initial_radial_boundary_load_energy_spread_ratio_max"
            ],
            1.10,
        )
        self.assertEqual(
            surface_gate[
                "final_video_eating_initial_radial_boundary_component_spread_ratio_max"
            ],
            1.20,
        )
        self.assertTrue(
            surface_gate[
                "final_video_eating_initial_radial_preserves_absolute_radial_fail_closed"
            ]
        )
        self.assertEqual(
            surface_gate["final_video_eating_prop_alignment_minimum_overlap"],
            0.75,
        )
        self.assertTrue(
            surface_gate[
                "final_video_eating_initial_radial_bounded_reference_corroboration"
            ]
        )
        self.assertEqual(
            surface_gate[
                "final_video_eating_initial_radial_bounded_reference_ratio_margin"
            ],
            0.10,
        )
        self.assertEqual(
            surface_gate[
                "final_video_eating_initial_radial_bounded_reference_load_delta_multiplier"
            ],
            2.0,
        )
        self.assertTrue(
            surface_gate[
                "final_video_eating_initial_radial_bounded_reference_preserves_absolute_radial_fail_closed"
            ]
        )
        self.assertTrue(
            surface_gate["final_video_eating_loop_boundary_decode_anchor_corroboration"]
        )
        self.assertTrue(
            surface_gate[
                "final_video_eating_boundary_anchor_requires_clean_first_and_last_sample"
            ]
        )
        self.assertTrue(
            surface_gate[
                "final_video_eating_boundary_anchor_requires_contiguous_interior_samples"
            ]
        )
        self.assertEqual(
            surface_gate["final_video_eating_boundary_anchor_minimum_samples"],
            5,
        )
        self.assertEqual(
            surface_gate["final_video_eating_boundary_anchor_minimum_blocking_samples"],
            2,
        )
        self.assertEqual(
            surface_gate["final_video_eating_boundary_anchor_spread_ratio_max"],
            1.20,
        )
        self.assertTrue(
            surface_gate[
                "final_video_grooming_masked_pet3d_low_signal_reference_texture_floor"
            ]
        )
        self.assertTrue(
            surface_gate[
                "final_video_grooming_masked_pet3d_sparse_coarse_low_signal_reference_texture_floor"
            ]
        )
        self.assertEqual(
            surface_gate["final_video_sparse_coarse_low_signal_minimum_samples"],
            5,
        )
        self.assertEqual(
            surface_gate[
                "final_video_sparse_coarse_low_signal_minimum_blocking_samples"
            ],
            2,
        )
        self.assertEqual(
            surface_gate[
                "final_video_sparse_coarse_low_signal_maximum_blocking_fraction"
            ],
            0.50,
        )
        self.assertTrue(
            surface_gate[
                "final_video_sparse_coarse_low_signal_requires_all_sample_long_line_observation"
            ]
        )
        self.assertTrue(
            surface_gate[
                "final_video_sparse_coarse_low_signal_coarse_ceiling_uses_one_existing_texture_delta_margin"
            ]
        )
        self.assertTrue(
            surface_gate[
                "final_video_sparse_coarse_low_signal_preserves_absolute_contour_radial_coherent_and_unknown_fail_closed"
            ]
        )
        self.assertTrue(
            surface_gate[
                "final_video_grooming_direct_qwen_sparse_coarse_temporal_floor"
            ]
        )
        self.assertEqual(
            surface_gate["final_video_direct_qwen_sparse_coarse_minimum_samples"],
            5,
        )
        self.assertEqual(
            surface_gate[
                "final_video_direct_qwen_sparse_coarse_minimum_blocking_samples"
            ],
            2,
        )
        self.assertEqual(
            surface_gate[
                "final_video_direct_qwen_sparse_coarse_maximum_blocking_fraction"
            ],
            0.50,
        )
        self.assertEqual(
            surface_gate[
                "final_video_direct_qwen_sparse_coarse_temporal_spread_ratio_max"
            ],
            1.20,
        )
        self.assertTrue(
            surface_gate[
                "final_video_direct_qwen_sparse_coarse_requires_non_low_signal_reference"
            ]
        )
        self.assertTrue(
            surface_gate[
                "final_video_direct_qwen_sparse_coarse_requires_blocking_dual_surface_coarse_long_pair"
            ]
        )
        self.assertTrue(
            surface_gate[
                "final_video_direct_qwen_sparse_coarse_coarse_ceiling_uses_reference_ratio_plus_one_texture_delta"
            ]
        )
        self.assertTrue(
            surface_gate[
                "final_video_direct_qwen_sparse_coarse_long_ceiling_reuses_existing_strong_limits"
            ]
        )
        self.assertTrue(
            surface_gate[
                "final_video_direct_qwen_sparse_coarse_preserves_absolute_contour_radial_coherent_and_unknown_fail_closed"
            ]
        )
        self.assertTrue(
            surface_gate[
                "final_video_low_signal_reference_requires_verified_composite_recovery"
            ]
        )
        self.assertTrue(
            surface_gate["final_video_low_signal_reference_requires_all_samples"]
        )
        self.assertTrue(
            surface_gate[
                "final_video_low_signal_reference_requires_coarse_blocking_all_samples"
            ]
        )
        self.assertEqual(
            surface_gate[
                "final_video_low_signal_reference_minimum_long_blocking_fraction"
            ],
            0.50,
        )
        self.assertTrue(
            surface_gate[
                "final_video_low_signal_reference_allows_remaining_long_observations"
            ]
        )
        self.assertEqual(
            surface_gate["final_video_low_signal_reference_minimum_samples"],
            3,
        )
        self.assertEqual(
            surface_gate["final_video_low_signal_reference_temporal_spread_ratio_max"],
            1.20,
        )
        self.assertTrue(
            surface_gate[
                "final_video_low_signal_reference_preserves_absolute_contour_radial_and_unknown_fail_closed"
            ]
        )

    def test_static_eating_geometry_recovery_exposes_only_frozen_silhouette(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "eating-reference.png"
            source = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
            draw = ImageDraw.Draw(source)
            draw.ellipse((100, 70, 410, 440), fill=(250, 240, 225, 255))
            draw.rectangle((170, 360, 350, 440), fill=(145, 82, 35, 255))
            source.save(reference)
            target = root / "eating-recovery-guide.png"
            events: list[dict[str, object]] = []

            with patch.object(
                pipeline,
                "append_recovery_event",
                side_effect=lambda _job_id, event: events.append(event),
            ):
                metadata = (
                    pipeline.prepare_static_eating_qwen_geometry_recovery_guide(
                        "guide-recovery-job",
                        {
                            "geometry_reference": str(reference),
                            "geometry_reference_sha256": pipeline.file_sha256(
                                reference
                            ),
                            "uploaded_identity_driving": {
                                "selected_frame_index": 40,
                            },
                        },
                        target,
                        ValueError("eating_prop_split_unreliable"),
                    )
                )

            pixels = np.asarray(Image.open(target).convert("RGB"), dtype=np.uint8)
            unique = {tuple(color) for color in pixels.reshape(-1, 3)}
            self.assertEqual(unique, {(0, 255, 0), (144, 144, 144)})
            self.assertEqual(
                metadata["policy"],
                "digest_bound_static_eating_geometry_recovery_v1",
            )
            self.assertTrue(metadata["temporal_geometry_recovery"])
            self.assertFalse(metadata["driver_rgb_preserved"])
            self.assertEqual(metadata["selected_frame_index"], 40)
            self.assertEqual(
                events[0]["result"],
                "continued_with_digest_bound_static_geometry",
            )

    def test_eating_qwen_geometry_guide_fails_closed_without_driver(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(
                FileNotFoundError,
                "Eating driving animation is missing",
            ):
                pipeline.prepare_eating_qwen_geometry_guide(
                    "missing-eating-driver",
                    root / "missing-driver.mp4",
                    root / "guide.png",
                    frames=81,
                    fps=16.0,
                )

    def test_grooming_qwen_geometry_guide_fails_closed_without_driver(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(
                FileNotFoundError,
                "Grooming driving animation is missing",
            ):
                pipeline.prepare_grooming_qwen_geometry_guide(
                    "missing-grooming-driver",
                    root / "missing-driver.mp4",
                    root / "guide.png",
                    frames=81,
                    fps=16.0,
                )

    def test_eating_qwen_geometry_guide_uses_versioned_runner_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "eat.mp4"
            source.write_bytes(b"main-case-eat")
            target = root / "eating-guide.png"
            captured: dict[str, object] = {}

            def fake_process(
                target_job_id: str,
                command: list[str],
                **kwargs: object,
            ) -> None:
                captured["job_id"] = target_job_id
                captured["command"] = command
                captured["kwargs"] = kwargs
                Image.new("RGB", (512, 512), (0, 255, 0)).save(target)
                with Image.open(target) as loaded:
                    image = loaded.copy()
                image.paste((120, 120, 120), (96, 80, 202, 432))
                image.paste((144, 144, 144), (202, 80, 310, 432))
                image.paste((168, 168, 168), (310, 80, 416, 432))
                image.paste((214, 208, 196), (180, 360, 330, 420))
                image.paste((232, 228, 218), (180, 360, 330, 370))
                image.paste((142, 86, 45), (220, 365, 260, 390))
                image.save(target)
                metadata_path = Path(
                    command[command.index("--eating-geometry-guide-metadata") + 1]
                )
                metadata_path.write_text(
                    json.dumps(
                        {
                            "policy": "driving_eating_geometry_image2_v6",
                            "method": (
                                "temporal_head_lowering_compressed_anatomy_"
                                "semantic_food_fields_v7"
                            ),
                            "source_sha256": pipeline.file_sha256(source),
                            "sampled_frame_count": 81,
                            "sampled_fps": 16.0,
                            "selected_frame_index": 54,
                            "muzzle_bowl_gap_px": 0,
                            "contact_pixels": 96,
                            "muzzle_contact_semantic": {
                                "method": (
                                    "source_anchored_continuous_neutral_muzzle_tip_v2"
                                ),
                                "muzzle_food_gap_px": 0,
                                "artificial_shape_added": False,
                                "detached_marker_added": False,
                                "source_rgb_preserved": False,
                                "source_chroma_preserved": False,
                                "identity_preserved": False,
                                "passed": True,
                            },
                            "semantic_food": {
                                "method": (
                                    "fixed_beige_bowl_brown_kibble_from_source_mask_v1"
                                ),
                                "food_core_pixels": 1000,
                                "minimum_food_core_pixels": 58,
                                "source_rgb_preserved": False,
                                "semantic_chroma_only": True,
                                "passed": True,
                            },
                            "head_lowering_selection": {
                                "method": "central_upper_subject_row_p10_v1",
                                "selected_head_support_pixels": 4096,
                                "head_y_p10_span": 75.0,
                                "head_y_p10_span_min": 6.144,
                                "passed": True,
                            },
                            "subject_pixels": 103640,
                            "bowl_pixels": 9000,
                            "outline_smoothing": {
                                "perimeter_ratio": 0.92,
                                "high_frequency_outline_preserved": False,
                            },
                            "semantic_structure": {
                                "method": (
                                    "compressed_blurred_quantized_neutral_anatomy_v1"
                                ),
                                "subject_gray_range": [120, 168],
                                "subject_gray_values": [120, 144, 168],
                                "subject_unique_rgb_count": 3,
                                "internal_luma_variation": True,
                                "appearance_bearing_face_detail": False,
                                "anatomy_gaussian_sigma": 10.0,
                                "source_luma_contrast": 48.0,
                                "quantum": 8,
                                "source_luma_values_preserved": False,
                                "source_rgb_preserved": False,
                                "source_chroma_preserved": False,
                                "high_frequency_texture_preserved": False,
                            },
                            "palette": {
                                "background": [0, 255, 0],
                                "subject_gray_range": [120, 168],
                                "subject_gray_values": [120, 144, 168],
                                "quantum": 8,
                                "prop_semantic_colors": {
                                    "bowl": [214, 208, 196],
                                    "bowl_rim": [232, 228, 218],
                                    "food": [142, 86, 45],
                                },
                                "source_rgb_preserved": False,
                                "source_chroma_preserved": False,
                                "semantic_prop_chroma_preserved": True,
                                "source_luma_values_preserved": False,
                                "high_frequency_texture_preserved": False,
                            },
                        }
                    ),
                    encoding="utf-8",
                )

            with (
                patch.multiple(
                    pipeline,
                    OUTPUT_ROOT=root / "outputs",
                    SCAIL_PYTHON=root / "python",
                    SCAIL_RUNNER=root / "runner.py",
                ),
                patch.object(
                    pipeline,
                    "run_cancelable_process",
                    side_effect=fake_process,
                ),
            ):
                metadata = pipeline.prepare_eating_qwen_geometry_guide(
                    "eat-guide-job",
                    source,
                    target,
                    frames=81,
                    fps=16.0,
                )

        command = captured["command"]
        self.assertEqual(captured["job_id"], "eat-guide-job")
        self.assertEqual(command[command.index("--source") + 1], str(source))
        self.assertEqual(command[command.index("--action") + 1], "eating")
        self.assertIn("--eating-geometry-guide-output", command)
        self.assertIn("--eating-geometry-guide-metadata", command)
        self.assertEqual(
            metadata["policy"],
            "driving_eating_geometry_image2_v6",
        )
        self.assertEqual(
            metadata["method"],
            "temporal_head_lowering_compressed_anatomy_semantic_food_fields_v7",
        )
        self.assertEqual(
            metadata["muzzle_contact_semantic"]["muzzle_food_gap_px"],
            0,
        )
        self.assertTrue(metadata["semantic_food"]["passed"])
        self.assertFalse(metadata["palette"]["source_rgb_preserved"])
        self.assertTrue(metadata["head_lowering_selection"]["passed"])
        self.assertTrue(metadata["semantic_structure"]["internal_luma_variation"])
        self.assertFalse(metadata["semantic_structure"]["source_luma_values_preserved"])
        self.assertRegex(metadata["guide_sha256"], r"^[0-9a-f]{64}$")

    def test_grooming_qwen_geometry_guide_uses_versioned_runner_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "TM.mp4"
            source.write_bytes(b"main-case-tm")
            target = root / "grooming-guide.png"
            captured: dict[str, object] = {}

            def fake_process(
                target_job_id: str,
                command: list[str],
                **kwargs: object,
            ) -> None:
                captured["job_id"] = target_job_id
                captured["command"] = command
                captured["kwargs"] = kwargs
                gate_target = Path(
                    command[command.index("--grooming-geometry-guide-output") + 1]
                )
                Image.new("RGB", (512, 512), (0, 255, 0)).save(gate_target)
                with Image.open(gate_target) as loaded:
                    image = loaded.copy()
                image.paste((112, 112, 112), (112, 72, 256, 440))
                image.paste((168, 168, 168), (256, 72, 400, 440))
                image.save(gate_target)
                metadata_path = Path(
                    command[command.index("--grooming-geometry-guide-metadata") + 1]
                )
                metadata_path.write_text(
                    json.dumps(
                        {
                            "policy": "driving_grooming_geometry_image1_v3",
                            "source_sha256": pipeline.file_sha256(source),
                            "palette": {"source_rgb_preserved": False},
                            "raised_paw_muzzle_relation": {
                                "passed": True,
                                "source_luma_values_preserved": False,
                                "black_or_tool_semantics_preserved": False,
                                "output_encoding": {
                                    "method": (
                                        "compressed_blurred_quantized_neutral_anatomy"
                                    )
                                },
                                "shoulder_anchor": [194, 342],
                                "paw_center": [190, 261],
                                "raised_foreleg_region_bbox": [
                                    168,
                                    242,
                                    214,
                                    362,
                                ],
                            },
                        }
                    ),
                    encoding="utf-8",
                )

            with (
                patch.multiple(
                    pipeline,
                    OUTPUT_ROOT=root / "outputs",
                    SCAIL_PYTHON=root / "python",
                    SCAIL_RUNNER=root / "runner.py",
                ),
                patch.object(
                    pipeline,
                    "run_cancelable_process",
                    side_effect=fake_process,
                ),
            ):
                metadata = pipeline.prepare_grooming_qwen_geometry_guide(
                    "groom-guide-job",
                    source,
                    target,
                    frames=81,
                    fps=16.0,
                )
                with Image.open(target) as loaded:
                    qwen_rgb = np.asarray(loaded.convert("RGB"), dtype=np.uint8)
                qwen_foreground = ~np.all(qwen_rgb == (0, 255, 0), axis=2)
                captured["qwen_unique_foreground"] = np.unique(
                    qwen_rgb[qwen_foreground], axis=0
                ).tolist()
                private_gate_path = Path(metadata["private_pose_gate"]["guide_path"])
                with Image.open(private_gate_path) as loaded:
                    private_rgb = np.asarray(loaded.convert("RGB"), dtype=np.uint8)
                private_foreground = ~np.all(private_rgb == (0, 255, 0), axis=2)
                captured["private_unique_foreground_count"] = len(
                    np.unique(private_rgb[private_foreground], axis=0)
                )
                completion_mask_path = Path(metadata["private_completion_mask"]["path"])
                with Image.open(completion_mask_path) as loaded:
                    completion_rgb = np.asarray(loaded.convert("RGB"), dtype=np.uint8)
                completion_foreground = ~np.all(completion_rgb == (0, 255, 0), axis=2)
                captured["completion_unique_foreground"] = np.unique(
                    completion_rgb[completion_foreground], axis=0
                ).tolist()
                captured["completion_foreleg_pixels"] = int(
                    np.count_nonzero(
                        np.all(
                            completion_rgb
                            == np.asarray(
                                pipeline.GROOMING_QWEN_GUIDE_FORELEG_RGB,
                                dtype=np.uint8,
                            ),
                            axis=2,
                        )
                    )
                )
                resolved_gate, resolved_metadata, resolved_completion = (
                    pipeline.resolve_main_case_grooming_pose_gate_metadata(
                        {
                            "pose_reference_metadata": {
                                "grooming": {
                                    "policy": (
                                        pipeline.GROOMING_QWEN_POSE_REFERENCE_POLICY
                                    ),
                                    "generator_policy": metadata["policy"],
                                    "guide_sha256": metadata["guide_sha256"],
                                    "private_pose_gate_guide_path": metadata[
                                        "private_pose_gate"
                                    ]["guide_path"],
                                    "private_pose_gate_guide_sha256": metadata[
                                        "private_pose_gate"
                                    ]["guide_sha256"],
                                    "private_pose_gate_exposed_to_qwen": False,
                                    "private_completion_mask_path": str(
                                        completion_mask_path
                                    ),
                                    "private_completion_mask_sha256": metadata[
                                        "private_completion_mask"
                                    ]["sha256"],
                                    "private_completion_mask_exposed_to_qwen": False,
                                    "generator_metadata": metadata,
                                }
                            }
                        },
                        target,
                    )
                )
                captured["resolved_gate"] = resolved_gate
                captured["resolved_metadata"] = resolved_metadata
                captured["resolved_completion"] = resolved_completion

        command = captured["command"]
        self.assertEqual(captured["job_id"], "groom-guide-job")
        self.assertEqual(command[command.index("--source") + 1], str(source))
        self.assertEqual(command[command.index("--action") + 1], "grooming")
        self.assertIn("--grooming-geometry-guide-output", command)
        self.assertIn("--grooming-geometry-guide-metadata", command)
        self.assertEqual(
            metadata["policy"],
            pipeline.GROOMING_QWEN_GEOMETRY_GUIDE_POLICY,
        )
        self.assertFalse(metadata["palette"]["source_rgb_preserved"])
        self.assertFalse(metadata["palette"]["internal_luma_variation"])
        self.assertEqual(
            metadata["palette"]["internal_luma_variation_role"],
            "none",
        )
        self.assertFalse(
            metadata["palette"]["appearance_bearing_internal_luma_variation"]
        )
        self.assertFalse(metadata["qwen_guide"]["main_case_appearance_exposed_to_qwen"])
        self.assertEqual(
            metadata["qwen_guide"]["foreground_rgb"],
            list(pipeline.GROOMING_QWEN_GUIDE_FOREGROUND_RGB),
        )
        self.assertEqual(
            metadata["qwen_guide"]["unique_foreground_rgb_count"],
            1,
        )
        self.assertEqual(
            metadata["qwen_guide"]["foreground_palette_rgb"],
            [list(pipeline.GROOMING_QWEN_GUIDE_FOREGROUND_RGB)],
        )
        self.assertFalse(metadata["qwen_guide"]["internal_luma_variation"])
        self.assertEqual(
            metadata["qwen_guide"]["internal_luma_variation_role"],
            "none",
        )
        self.assertFalse(
            metadata["qwen_guide"]["appearance_bearing_internal_luma_variation"]
        )
        self.assertEqual(
            metadata["qwen_guide"]["role"],
            "image2_action_geometry_guide",
        )
        structural_negative_space = metadata["qwen_guide"][
            "structural_negative_space"
        ]
        self.assertEqual(
            structural_negative_space["method"],
            pipeline.GROOMING_QWEN_NEGATIVE_SPACE_METHOD,
        )
        self.assertFalse(structural_negative_space["curve_closed"])
        self.assertTrue(structural_negative_space["paw_endpoint_open"])
        self.assertTrue(structural_negative_space["shoulder_endpoint_open"])
        self.assertFalse(structural_negative_space["source_rgb_or_luma_sampled"])
        self.assertGreaterEqual(structural_negative_space["pixels"], 300)
        self.assertLessEqual(structural_negative_space["pixels"], 1800)
        self.assertIsNone(metadata["qwen_guide"]["synthetic_foreleg_semantic"])
        self.assertFalse(metadata["qwen_guide"]["closed_foreleg_capsule_exposed"])
        self.assertTrue(metadata["qwen_guide"]["internal_green_separator_exposed"])
        self.assertEqual(
            metadata["qwen_guide"]["raised_foreleg_validation_role"],
            "public_open_curve_geometry_plus_private_cv",
        )
        self.assertEqual(
            int(
                np.count_nonzero(
                    np.all(
                        qwen_rgb
                        == np.asarray(
                            pipeline.GROOMING_QWEN_GUIDE_FORELEG_RGB,
                            dtype=np.uint8,
                        ),
                        axis=2,
                    )
                )
            ),
            0,
        )
        self.assertEqual(
            metadata["qwen_guide"]["foreground_pixels"],
            metadata["qwen_guide"]["private_foreground_pixels"]
            - structural_negative_space["pixels"],
        )
        self.assertEqual(
            int(np.count_nonzero(private_foreground & ~qwen_foreground)),
            structural_negative_space["pixels"],
        )
        self.assertFalse(np.any(qwen_foreground & ~private_foreground))
        self.assertFalse(metadata["private_pose_gate"]["exposed_to_qwen"])
        self.assertFalse(metadata["private_completion_mask"]["exposed_to_qwen"])
        self.assertFalse(metadata["private_completion_mask"]["loaded_as_qwen_image"])
        self.assertEqual(
            metadata["private_completion_mask"]["policy"],
            pipeline.GROOMING_PRIVATE_COMPLETION_MASK_POLICY,
        )
        self.assertEqual(
            captured["completion_unique_foreground"],
            [
                list(pipeline.GROOMING_QWEN_GUIDE_FOREGROUND_RGB),
                list(pipeline.GROOMING_QWEN_GUIDE_FORELEG_RGB),
            ],
        )
        self.assertGreaterEqual(captured["completion_foreleg_pixels"], 650)
        self.assertEqual(
            captured["resolved_gate"],
            Path(metadata["private_pose_gate"]["guide_path"]),
        )
        self.assertEqual(
            captured["resolved_metadata"],
            Path(metadata["private_pose_gate"]["metadata_path"]),
        )
        self.assertEqual(
            captured["resolved_completion"],
            Path(metadata["private_completion_mask"]["path"]),
        )
        self.assertEqual(
            metadata["private_pose_gate"]["policy"],
            pipeline.GROOMING_PRIVATE_POSE_GATE_GUIDE_POLICY,
        )
        self.assertNotEqual(
            Path(command[command.index("--grooming-geometry-guide-output") + 1]),
            target,
        )
        self.assertEqual(
            captured["qwen_unique_foreground"],
            [list(pipeline.GROOMING_QWEN_GUIDE_FOREGROUND_RGB)],
        )
        self.assertGreater(
            captured["private_unique_foreground_count"],
            1,
        )
        self.assertRegex(metadata["guide_sha256"], r"^[0-9a-f]{64}$")

    def test_eating_contact_gate_accepts_broad_centered_stylized_muzzle(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            candidate, colored_mask = _write_eating_contact_fixture(
                Path(temporary),
                name="broad-contact",
                muzzle_width_bowl_ratio=0.62,
                muzzle_gap_px=0,
            )
            with patch(
                "scail2_subject_replacement_test.make_eating_reference_identity_mask",
                return_value=(colored_mask, None, {"fixture": "broad-contact"}),
            ):
                result = pipeline.eating_mouth_bowl_contact_quality(candidate)

        self.assertTrue(result["acceptable"])
        self.assertEqual(result["method"], pipeline.EATING_MOUTH_BOWL_CONTACT_POLICY)
        self.assertEqual(result["muzzle_bowl_gap_px"], 0)
        self.assertAlmostEqual(
            result["selected_component"]["width_bowl_ratio"],
            0.616667,
            places=6,
        )
        self.assertEqual(
            result["thresholds"]["component_width_bowl_ratio_max"],
            0.70,
        )
        self.assertEqual(
            result["selected_component"]["rejection_reasons"],
            [],
        )

    def test_eating_prop_split_falls_back_from_thin_warm_rim_to_beige_bowl(
        self,
    ) -> None:
        from scail2_subject_replacement_test import (
            make_eating_reference_identity_mask,
        )

        with tempfile.TemporaryDirectory() as temporary:
            candidate = Path(temporary) / "thin-warm-rim-beige-bowl.png"
            _write_thin_warm_rim_beige_bowl_fixture(candidate)
            with Image.open(candidate) as loaded:
                colored, exclusion, metadata = make_eating_reference_identity_mask(
                    loaded.convert("RGBA")
                )

        colored_array = np.asarray(colored.convert("RGB"))
        red_pixels = int(np.count_nonzero(np.all(colored_array == (255, 0, 0), axis=2)))
        blue_pixels = int(
            np.count_nonzero(np.all(colored_array == (0, 0, 255), axis=2))
        )
        self.assertIsNotNone(exclusion)
        self.assertTrue(metadata["warm_component_initially_classified_as_bowl"])
        self.assertTrue(metadata["warm_bowl_core_fallback_to_beige"])
        self.assertEqual(
            metadata["bowl_body_method"],
            pipeline.EATING_BOWL_SPLIT_FALLBACK_POLICY,
        )
        self.assertGreater(red_pixels, 100)
        self.assertGreater(blue_pixels, 500)
        self.assertGreaterEqual(metadata["safe_width_fraction"], 0.25)
        self.assertLessEqual(metadata["safe_width_fraction"], 0.55)
        self.assertGreaterEqual(metadata["prop_fraction"], 0.04)
        self.assertLessEqual(metadata["prop_fraction"], 0.20)
        self.assertGreaterEqual(metadata["left_paw_identity_support"], 500)
        self.assertGreaterEqual(metadata["right_paw_identity_support"], 500)

    def test_eating_contact_gate_rejects_broad_muzzle_with_air_gap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            candidate, colored_mask = _write_eating_contact_fixture(
                Path(temporary),
                name="broad-gap",
                muzzle_width_bowl_ratio=0.62,
                muzzle_gap_px=31,
            )
            with patch(
                "scail2_subject_replacement_test.make_eating_reference_identity_mask",
                return_value=(colored_mask, None, {"fixture": "broad-gap"}),
            ):
                result = pipeline.eating_mouth_bowl_contact_quality(candidate)

        self.assertFalse(result["acceptable"])
        self.assertEqual(result["issues"], ["eating_muzzle_not_at_bowl"])
        nearest = result["component_candidates"][0]
        self.assertEqual(nearest["gap_px"], 31)
        self.assertIn("gap_outside_contact_range", nearest["rejection_reasons"])

    def test_eating_contact_gate_keeps_broad_component_upper_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            candidate, colored_mask = _write_eating_contact_fixture(
                Path(temporary),
                name="too-wide-contact",
                muzzle_width_bowl_ratio=0.705,
                muzzle_gap_px=0,
            )
            with patch(
                "scail2_subject_replacement_test.make_eating_reference_identity_mask",
                return_value=(colored_mask, None, {"fixture": "too-wide-contact"}),
            ):
                result = pipeline.eating_mouth_bowl_contact_quality(candidate)

        self.assertFalse(result["acceptable"])
        self.assertEqual(result["issues"], ["eating_muzzle_not_at_bowl"])
        nearest = result["component_candidates"][0]
        self.assertEqual(nearest["width_bowl_ratio"], 0.708333)
        self.assertIn("width_outside_muzzle_range", nearest["rejection_reasons"])

    def test_eating_contact_gate_rejects_tiny_off_center_bowl_edge_fragment(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "bowl-edge-fragment.png"
            reference = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
            reference.paste((105, 110, 115, 255), (30, 20, 226, 220))
            reference.paste((214, 208, 196, 255), (68, 170, 188, 225))
            reference.paste((205, 118, 116, 255), (72, 168, 81, 170))
            reference.save(candidate)
            colored_mask = Image.new("RGB", (256, 256), (0, 0, 0))
            colored_mask.paste((0, 0, 255), (30, 20, 226, 220))
            colored_mask.paste((255, 0, 0), (68, 170, 188, 225))
            with patch(
                "scail2_subject_replacement_test.make_eating_reference_identity_mask",
                return_value=(colored_mask, None, {"fixture": "edge-fragment"}),
            ):
                result = pipeline.eating_mouth_bowl_contact_quality(candidate)

        self.assertFalse(result["acceptable"], result)
        nearest = result["component_candidates"][0]
        self.assertLess(nearest["area"], pipeline.EATING_MUZZLE_COMPONENT_AREA_MIN_PX)
        self.assertIn("area_below_minimum", nearest["rejection_reasons"])
        self.assertIn("height_outside_muzzle_range", nearest["rejection_reasons"])
        self.assertIn(
            "center_too_far_from_bowl_center",
            nearest["rejection_reasons"],
        )

    def test_eating_degraded_selection_prefers_clean_uncertain_contact_candidate(
        self,
    ) -> None:
        identity_ok = {"acceptable": True, "penalty": 8.0}
        similarity_ok = {"acceptable": True, "penalty": 12.0}
        bad_shelf = {
            "attempt": 2,
            "identity_palette": identity_ok,
            "similarity": similarity_ok,
            "fur_silhouette": {"acceptable": False, "penalty": 70.0},
            "quality": {
                "acceptable": False,
                "penalty": 50.0,
                "issues": [
                    "eating_bowl_or_food_not_detected",
                    "eating_horizontal_foreground_shelf",
                ],
                "eating_horizontal_shelf": {"acceptable": False},
            },
        }
        clean_uncertain = {
            "attempt": 3,
            "identity_palette": {"acceptable": True, "penalty": 18.0},
            "similarity": similarity_ok,
            "fur_silhouette": {"acceptable": True, "penalty": 4.0},
            "quality": {
                "acceptable": False,
                "penalty": 25.0,
                "issues": ["eating_bowl_contact_unreliable"],
                "eating_horizontal_shelf": {"acceptable": True},
            },
        }
        identity_collapsed = {
            "attempt": 4,
            "identity_palette": {"acceptable": False, "penalty": 1.0},
            "similarity": similarity_ok,
            "fur_silhouette": {"acceptable": True, "penalty": 0.0},
            "quality": {"acceptable": True, "penalty": 0.0, "issues": []},
        }

        clean_key = pipeline.action_candidate_degraded_selection_key(
            clean_uncertain,
            action="eating",
        )
        self.assertLess(
            clean_key,
            pipeline.action_candidate_degraded_selection_key(
                bad_shelf,
                action="eating",
            ),
        )
        self.assertLess(
            clean_key,
            pipeline.action_candidate_degraded_selection_key(
                identity_collapsed,
                action="eating",
            ),
        )

    def test_eating_degraded_selection_prefers_clean_edge_after_identity(self) -> None:
        base = {
            "identity_palette": {"acceptable": True, "penalty": 2.0},
            "similarity": {"acceptable": True, "penalty": 2.0},
            "fur_silhouette": {"acceptable": True, "penalty": 0.0},
            "quality": {"acceptable": True, "penalty": 0.0, "issues": []},
        }
        black_edge = {
            **base,
            "attempt": 1,
            "chroma": {
                "transparent_edge_quality": {
                    "acceptable": False,
                    "penalty": 80.0,
                }
            },
        }
        clean_edge = {
            **base,
            "attempt": 2,
            "chroma": {
                "transparent_edge_quality": {
                    "acceptable": True,
                    "penalty": 0.0,
                }
            },
        }

        self.assertLess(
            pipeline.action_candidate_degraded_selection_key(
                clean_edge,
                action="eating",
            ),
            pipeline.action_candidate_degraded_selection_key(
                black_edge,
                action="eating",
            ),
        )

    def test_exhausted_action_candidate_never_reuses_identity_drift(self) -> None:
        safe = {"identity_palette": {"acceptable": True}}
        drifted = {"identity_palette": {"acceptable": False}}

        self.assertEqual(
            pipeline.exhausted_action_candidate_recovery_policy(
                safe,
                action="standing",
                approved_canonical_available=True,
            ),
            "identity_safe_generated_candidate",
        )
        geometry_failed = {
            "identity_palette": {"acceptable": True},
            "quality": {
                "acceptable": False,
                "issues": ["pose_not_low_or_wide_enough_for_bowl_action"],
            },
        }
        self.assertEqual(
            pipeline.exhausted_action_candidate_recovery_policy(
                geometry_failed,
                action="drinking",
                approved_canonical_available=True,
            ),
            "identity_safe_generated_candidate_requires_final_action_gate",
        )
        self.assertEqual(
            pipeline.exhausted_action_candidate_recovery_policy(
                drifted,
                action="standing",
                approved_canonical_available=True,
            ),
            "upload_anchored_canonical",
        )
        for action in ("drinking", "running", "sleeping", "toileting"):
            with self.subTest(action=action):
                self.assertEqual(
                    pipeline.exhausted_action_candidate_recovery_policy(
                        drifted,
                        action=action,
                        approved_canonical_available=True,
                    ),
                    "upload_anchored_canonical",
                )
        self.assertEqual(
            pipeline.exhausted_action_candidate_recovery_policy(
                drifted,
                action="eating",
                approved_canonical_available=True,
            ),
            "reject",
        )

    def test_eating_transparent_output_starts_at_validated_contact_phase(
        self,
    ) -> None:
        job = {
            "pose_reference_metadata": {
                "eating": {
                    "generator_metadata": {
                        "policy": "driving_eating_geometry_image2_v6",
                        "selected_frame_index": 56,
                        "head_lowering_selection": {"passed": True},
                    }
                }
            }
        }

        self.assertEqual(
            pipeline.resolve_transparent_output_phase_start_index(
                job,
                action="eating",
                source_frames=81,
            ),
            56,
        )
        self.assertEqual(
            pipeline.resolve_transparent_output_phase_start_index(
                job,
                action="grooming",
                source_frames=81,
            ),
            0,
        )

    def test_fur_surface_retry_preserves_only_configured_eating_recovery(
        self,
    ) -> None:
        eating_profiles = [
            {"name": "pose_capped"},
            {"name": "pose_reduced_retry"},
            {"name": "pose_minimal_retry"},
            {"name": "bowl_palette_canvas_seed_retry_1"},
            {"name": "bowl_palette_canvas_seed_retry_2"},
        ]
        before_first_recovery = pipeline.resolve_fur_surface_retry_decision(
            action="eating",
            rejection_streak=3,
            attempt=3,
            active_profiles=eating_profiles,
        )
        self.assertEqual(
            before_first_recovery["policy"],
            pipeline.FUR_SURFACE_GATE_RETRY_POLICY,
        )
        self.assertTrue(before_first_recovery["threshold_reached"])
        self.assertTrue(before_first_recovery["pending_eating_bowl_recovery"])
        self.assertFalse(before_first_recovery["short_circuit_due"])
        self.assertEqual(
            before_first_recovery["remaining_profiles"],
            [
                "bowl_palette_canvas_seed_retry_1",
                "bowl_palette_canvas_seed_retry_2",
            ],
        )

        between_recoveries = pipeline.resolve_fur_surface_retry_decision(
            action="eating",
            rejection_streak=4,
            attempt=4,
            active_profiles=eating_profiles,
        )
        self.assertTrue(between_recoveries["threshold_reached"])
        self.assertTrue(between_recoveries["pending_eating_bowl_recovery"])
        self.assertFalse(between_recoveries["short_circuit_due"])
        self.assertEqual(
            between_recoveries["remaining_profiles"],
            ["bowl_palette_canvas_seed_retry_2"],
        )
        self.assertEqual(between_recoveries["attempt_budget"], 5)

        after_recovery = pipeline.resolve_fur_surface_retry_decision(
            action="eating",
            rejection_streak=5,
            attempt=5,
            active_profiles=eating_profiles,
        )
        self.assertFalse(after_recovery["pending_eating_bowl_recovery"])
        self.assertTrue(after_recovery["short_circuit_due"])
        self.assertTrue(after_recovery["blocks_scail_on_exhaustion"])
        self.assertEqual(after_recovery["remaining_profiles"], [])
        self.assertEqual(after_recovery["attempt_budget"], 5)

        grooming_final = pipeline.resolve_fur_surface_retry_decision(
            action="grooming",
            rejection_streak=3,
            attempt=3,
            active_profiles=eating_profiles[:3],
        )
        self.assertFalse(grooming_final["pending_eating_bowl_recovery"])
        self.assertTrue(grooming_final["short_circuit_due"])
        self.assertTrue(grooming_final["blocks_scail_on_exhaustion"])

        grooming_before_threshold = pipeline.resolve_fur_surface_retry_decision(
            action="grooming",
            rejection_streak=2,
            attempt=2,
            active_profiles=eating_profiles[:3],
        )
        self.assertFalse(grooming_before_threshold["threshold_reached"])
        self.assertFalse(grooming_before_threshold["short_circuit_due"])

        generic_eating_retry = pipeline.resolve_fur_surface_retry_decision(
            action="eating",
            rejection_streak=3,
            attempt=3,
            active_profiles=[*eating_profiles[:3], {"name": "generic_retry"}],
        )
        self.assertFalse(generic_eating_retry["pending_eating_bowl_recovery"])
        self.assertTrue(generic_eating_retry["short_circuit_due"])

        non_adjacent_bowl_retry = pipeline.resolve_fur_surface_retry_decision(
            action="eating",
            rejection_streak=3,
            attempt=3,
            active_profiles=[
                *eating_profiles[:3],
                {"name": "generic_retry"},
                {"name": "bowl_palette_canvas_seed_retry_1"},
            ],
        )
        self.assertEqual(non_adjacent_bowl_retry["next_profile"], "generic_retry")
        self.assertFalse(non_adjacent_bowl_retry["pending_eating_bowl_recovery"])
        self.assertTrue(non_adjacent_bowl_retry["short_circuit_due"])

    def test_uploaded_surface_candidate_uses_absolute_identity_contract(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cutout = Path(temporary) / "candidate.png"
            image = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
            ImageDraw.Draw(image).ellipse(
                (24, 12, 104, 116),
                fill=(128, 128, 128, 255),
            )
            image.save(cutout)
            effective = {
                "upload_validation": {
                    "uploaded_identity_appearance_authority": True,
                },
                "actions": {
                    "eating": {
                        "fur_surface_gate": {
                            "candidate_comparison_mode": (
                                pipeline.UPLOADED_IDENTITY_SURFACE_COMPARISON_MODE
                            ),
                        }
                    }
                },
            }
            evaluated = {
                "acceptable": True,
                "verdict": "accepted_uploaded_identity_absolute_contract",
                "issues": [],
            }
            with patch.object(
                pipeline,
                "evaluate_fur_surface",
                return_value=evaluated,
            ) as evaluator:
                result = pipeline.evaluate_main_case_fur_surface_candidate(
                    cutout,
                    action="eating",
                    alpha_provenance="chroma_key",
                    effective_subject_config=effective,
                )

        args, kwargs = evaluator.call_args
        self.assertEqual(args[0], cutout)
        self.assertIsNone(args[1])
        self.assertTrue(kwargs["accept_absolute_contract"])
        self.assertFalse(kwargs["baseline_is_validated"])
        self.assertTrue(result["acceptable"])
        self.assertEqual(
            result["gate"]["candidate_comparison_mode"],
            pipeline.UPLOADED_IDENTITY_SURFACE_COMPARISON_MODE,
        )
        self.assertFalse(
            result["gate"]["profile_pixels_used_for_candidate_appearance_validation"]
        )

    def test_eating_pose_profiles_append_two_finite_minimal_seed_retries(
        self,
    ) -> None:
        profiles = [
            {
                "name": "pose_capped",
                "user_loras": [{"model_name": "first.safetensors", "strength": 0.2}],
                "quality_retry": 1,
                "steps": 8,
            },
            {
                "name": "pose_reduced_retry",
                "user_loras": [{"model_name": "second.safetensors", "strength": 0.1}],
                "quality_retry": 2,
                "steps": 9,
            },
            {
                "name": "pose_minimal_retry",
                "user_loras": [],
                "quality_retry": 3,
                "steps": 10,
            },
        ]

        with patch.object(pipeline, "QWEN_BOWL_POSE_EXTRA_ATTEMPTS", 2):
            selected = pipeline.pose_generation_profiles("eating", profiles)

        self.assertEqual(
            [profile["name"] for profile in selected],
            [
                "pose_capped",
                "pose_reduced_retry",
                "pose_minimal_retry",
                "bowl_palette_canvas_seed_retry_1",
                "bowl_palette_canvas_seed_retry_2",
            ],
        )
        self.assertEqual(
            [profile["quality_retry"] for profile in selected[-2:]],
            [4, 5],
        )
        self.assertEqual(
            [profile["steps"] for profile in selected[-2:]],
            [10, 10],
        )
        self.assertEqual(selected[-2]["user_loras"], [])
        self.assertEqual(selected[-1]["user_loras"], [])
        self.assertIsNot(selected[-2]["user_loras"], profiles[-1]["user_loras"])
        self.assertIsNot(selected[-1]["user_loras"], profiles[-1]["user_loras"])
        self.assertIsNot(selected[-1]["user_loras"], selected[-2]["user_loras"])
        self.assertEqual(profiles[-1]["name"], "pose_minimal_retry")

    def test_normalize_qwen_action_candidate_returns_direct_cutout_without_semantic_matte(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "candidate.png"
            direct_cutout = root / "direct-cutout.png"
            semantic_green = root / "semantic-green.png"
            semantic_cutout = root / "semantic-cutout.png"
            semantic_raw = root / "semantic-raw.png"
            source.write_bytes(b"candidate")
            direct_metrics = {
                "acceptable": True,
                "issues": [],
                "method": "chroma_key",
            }

            def fake_direct_evaluation(
                _source: Path,
                target: Path,
            ) -> dict[str, object]:
                image = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
                image.paste((80, 90, 100, 255), (180, 220, 844, 900))
                image.save(target)
                return direct_metrics

            with (
                patch.object(
                    pipeline,
                    "evaluate_chroma_cutout",
                    side_effect=fake_direct_evaluation,
                ) as evaluate_direct,
                patch.object(
                    pipeline,
                    "semantic_background_recovery",
                ) as recover_semantic,
            ):
                metrics, selected_cutout = pipeline.normalize_qwen_action_candidate(
                    source,
                    direct_cutout,
                    semantic_green,
                    semantic_cutout,
                    semantic_raw,
                    pose="eating",
                    allow_semantic_matte=True,
                )
            with Image.open(selected_cutout) as normalized:
                selected_size = normalized.size
                selected_mode = normalized.mode
            selected_digest = pipeline.file_sha256(selected_cutout)

        evaluate_direct.assert_called_once_with(source, direct_cutout)
        recover_semantic.assert_not_called()
        self.assertNotEqual(selected_cutout, direct_cutout)
        self.assertEqual(selected_cutout.name, "direct-cutout_reference_512.png")
        self.assertEqual(selected_size, (512, 512))
        self.assertEqual(selected_mode, "RGBA")
        self.assertTrue(metrics["acceptable"])
        self.assertEqual(metrics["direct_chroma"], direct_metrics)
        self.assertFalse(metrics["semantic_recovery_attempted"])
        self.assertIsNone(metrics["semantic_recovery_acceptable"])
        self.assertTrue(metrics["normalized_after_qwen"])
        self.assertEqual(
            metrics["normalization_policy"],
            pipeline.QWEN_ACTION_REFERENCE_NORMALIZATION_POLICY,
        )
        self.assertEqual(
            metrics["matte_policy"],
            pipeline.QWEN_ACTION_CANDIDATE_MATTE_POLICY,
        )
        self.assertEqual(
            metrics["final_reference_sha256"],
            selected_digest,
        )

    def test_normalize_qwen_action_candidate_uses_semantic_matte_after_direct_failure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "candidate.png"
            direct_cutout = root / "direct-cutout.png"
            semantic_green = root / "semantic-green.png"
            semantic_cutout = root / "semantic-cutout.png"
            semantic_raw = root / "semantic-raw.png"
            source.write_bytes(b"candidate")
            direct_metrics = {
                "acceptable": False,
                "issues": ["green_background_not_clean"],
                "method": "chroma_key",
            }
            semantic_metrics = {
                "acceptable": True,
                "issues": [],
                "method": "rembg_u2net_fallback",
            }

            def fake_semantic_recovery(
                _source: Path,
                _green: Path,
                target: Path,
                _raw: Path,
                *,
                pose: str,
            ) -> dict[str, object]:
                self.assertEqual(pose, "eating")
                image = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
                image.paste((80, 90, 100, 255), (180, 220, 844, 900))
                image.save(target)
                return semantic_metrics

            with (
                patch.object(
                    pipeline,
                    "evaluate_chroma_cutout",
                    return_value=direct_metrics,
                ) as evaluate_direct,
                patch.object(
                    pipeline,
                    "semantic_background_recovery",
                    side_effect=fake_semantic_recovery,
                ) as recover_semantic,
            ):
                metrics, selected_cutout = pipeline.normalize_qwen_action_candidate(
                    source,
                    direct_cutout,
                    semantic_green,
                    semantic_cutout,
                    semantic_raw,
                    pose="eating",
                    allow_semantic_matte=True,
                )
            with Image.open(selected_cutout) as normalized:
                selected_size = normalized.size
                selected_mode = normalized.mode

        evaluate_direct.assert_called_once_with(source, direct_cutout)
        recover_semantic.assert_called_once_with(
            source,
            semantic_green,
            semantic_cutout,
            semantic_raw,
            pose="eating",
        )
        self.assertNotEqual(selected_cutout, semantic_cutout)
        self.assertEqual(selected_cutout.name, "semantic-cutout_reference_512.png")
        self.assertEqual(selected_size, (512, 512))
        self.assertEqual(selected_mode, "RGBA")
        self.assertTrue(metrics["acceptable"])
        self.assertEqual(metrics["direct_chroma"], direct_metrics)
        self.assertTrue(metrics["semantic_recovery_attempted"])
        self.assertTrue(metrics["semantic_recovery_acceptable"])
        self.assertTrue(metrics["normalized_after_qwen"])
        self.assertEqual(
            metrics["normalization_policy"],
            pipeline.QWEN_ACTION_REFERENCE_NORMALIZATION_POLICY,
        )

    def test_normalize_qwen_action_candidate_audits_semantic_exception_as_failure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "candidate.png"
            direct_cutout = root / "direct-cutout.png"
            semantic_green = root / "semantic-green.png"
            semantic_cutout = root / "semantic-cutout.png"
            semantic_raw = root / "semantic-raw.png"
            source.write_bytes(b"candidate")
            direct_metrics = {
                "acceptable": False,
                "issues": ["green_background_not_clean"],
                "method": "chroma_key",
            }

            with (
                patch.object(
                    pipeline,
                    "evaluate_chroma_cutout",
                    return_value=direct_metrics,
                ) as evaluate_direct,
                patch.object(
                    pipeline,
                    "semantic_background_recovery",
                    side_effect=RuntimeError("semantic matte boom"),
                ) as recover_semantic,
            ):
                metrics, selected_cutout = pipeline.normalize_qwen_action_candidate(
                    source,
                    direct_cutout,
                    semantic_green,
                    semantic_cutout,
                    semantic_raw,
                    pose="grooming",
                    allow_semantic_matte=True,
                )

        evaluate_direct.assert_called_once_with(source, direct_cutout)
        recover_semantic.assert_called_once_with(
            source,
            semantic_green,
            semantic_cutout,
            semantic_raw,
            pose="grooming",
        )
        self.assertEqual(selected_cutout, direct_cutout)
        self.assertFalse(metrics["acceptable"])
        self.assertEqual(metrics["direct_chroma"], direct_metrics)
        self.assertTrue(metrics["semantic_recovery_attempted"])
        self.assertFalse(metrics["semantic_recovery_acceptable"])
        self.assertEqual(
            metrics["semantic_recovery_error"],
            "RuntimeError: semantic matte boom",
        )
        self.assertFalse(metrics["normalized_after_qwen"])
        self.assertEqual(
            metrics["normalization_policy"],
            pipeline.QWEN_ACTION_REFERENCE_NORMALIZATION_POLICY,
        )

    def test_prepare_pose_input_byte_preserves_gated_rgba_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary) / "outputs"
            job_id = "preserve-gated-alpha"
            job_dir = output_root / job_id
            job_dir.mkdir(parents=True)
            source = job_dir / "pose_01_qwen_cutout.png"
            image = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
            image.paste((80, 90, 100, 255), (96, 64, 416, 448))
            image.save(source, format="PNG", optimize=True)
            source_bytes = source.read_bytes()
            jobs = {
                job_id: {
                    "job_id": job_id,
                    "outputs": [],
                    "cancel_requested": False,
                }
            }

            with (
                patch.multiple(
                    pipeline,
                    OUTPUT_ROOT=output_root,
                    JOBS=jobs,
                ),
                patch.object(
                    pipeline,
                    "convert_chroma_image_to_transparent_png",
                ) as chroma_rekey,
            ):
                wan_input = pipeline.prepare_pose_input(
                    job_id,
                    job_dir,
                    1,
                    "eating",
                    source,
                    512,
                    512,
                    transparent=True,
                    preserve_gated_alpha=True,
                )

            published = job_dir / "pose_01_keyframe_transparent.png"
            self.assertEqual(published.read_bytes(), source_bytes)
            self.assertEqual(
                pipeline.file_sha256(published),
                pipeline.file_sha256(source),
            )
            chroma_rekey.assert_not_called()
            with Image.open(wan_input) as green:
                self.assertEqual(green.mode, "RGB")
                self.assertEqual(green.size, (512, 512))
                self.assertEqual(green.getpixel((0, 0)), (0, 255, 0))
                self.assertEqual(green.getpixel((256, 256)), (80, 90, 100))
            output = jobs[job_id]["outputs"][0]
            self.assertTrue(output["byte_preserved_from_gated_candidate"])
            self.assertEqual(
                output["normalization_policy"],
                pipeline.QWEN_ACTION_REFERENCE_NORMALIZATION_POLICY,
            )
            self.assertEqual(output["sha256"], pipeline.file_sha256(source))

    def test_action_reference_normalization_cleans_dark_soft_alpha_edge(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "candidate.png"
            target = root / "candidate_reference_512.png"
            rgba = np.zeros((256, 256, 4), dtype=np.uint8)
            rgba[28:228, 28:228, :3] = 16
            rgba[28:228, 28:228, 3] = 40
            rgba[34:222, 34:222, :3] = 20
            rgba[34:222, 34:222, 3] = 180
            rgba[42:214, 42:214, :3] = 232
            rgba[42:214, 42:214, 3] = 255
            Image.fromarray(rgba, mode="RGBA").save(source)

            metrics = pipeline.normalize_qwen_action_reference_rgba(
                source,
                target,
                action="grooming",
            )

            cleanup = metrics["transparent_edge_cleanup"]
            self.assertTrue(cleanup["applied"])
            self.assertTrue(cleanup["acceptable"])
            self.assertGreater(cleanup["low_alpha_candidate_pixels"], 64)
            self.assertEqual(
                metrics["transparent_edge_quality"],
                cleanup["after"],
            )
            with Image.open(target) as cleaned:
                cleaned_rgba = np.asarray(cleaned.convert("RGBA"), dtype=np.uint8)
            retained_soft_edge = (
                (cleaned_rgba[:, :, 3] > 0)
                & (cleaned_rgba[:, :, 3] < 255)
            )
            self.assertGreater(
                float(cleaned_rgba[:, :, :3][retained_soft_edge].mean()),
                150.0,
            )


if __name__ == "__main__":
    unittest.main()
