from __future__ import annotations

import asyncio
import io
import json
import queue
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
from PIL import Image, ImageDraw

import fur_surface_gate as surface_gate
import local_video_api as pipeline
import transparent_animation as transparency

from tests.integration.local_video_subject_replacement_test_support import (
    _subject_profile_v2,
)




def _qwen_upload_validation(
    actions: tuple[str, ...],
    expected_pose_policies: dict[str, str] | None = None,
) -> dict[str, object]:
    resolved_pose_policies = (
        dict(expected_pose_policies)
        if expected_pose_policies is not None
        else {
            action: pipeline.FULL_DRIVER_EATING_POSE_POLICY
            for action in actions
            if action == "eating"
        }
    )
    return {
        "policy": pipeline.QWEN_FULL_BODY_FIRST_FRAME_PIPELINE,
        "qwen_full_body_required": True,
        "qwen_action_keyframes_required": True,
        "normalized_transparent_references_required": True,
        "expected_surface_gate_actions": list(actions),
        "expected_pose_policies": resolved_pose_policies,
    }


def _test_fur_surface_gate(action: str, reference: Path) -> dict[str, object]:
    digest = pipeline.file_sha256(reference)
    return {
        "enabled": True,
        "policy": pipeline.FUR_SURFACE_GATE_POLICY,
        "gate_version": pipeline.FUR_SURFACE_GATE_VERSION,
        "threshold_version": pipeline.FUR_SURFACE_THRESHOLD_VERSION,
        "scope": "main_case_guided_qwen_action_candidates",
        "reliable_inputs": ["alpha_cutout", "rembg_alpha_cutout"],
        "fail_closed": True,
        "max_consecutive_rejections": (pipeline.FUR_SURFACE_GATE_MAX_REJECTIONS),
        "baseline": {
            "enabled": True,
            "policy": pipeline.FUR_SURFACE_GATE_POLICY,
            "gate_version": pipeline.FUR_SURFACE_GATE_VERSION,
            "threshold_version": pipeline.FUR_SURFACE_THRESHOLD_VERSION,
            "action": action,
            "source": str(reference),
            "source_sha256": digest,
            "path": str(reference),
            "sha256": digest,
        },
    }
























class FurSurfaceRadialEvidenceTests(unittest.TestCase):
    @staticmethod
    def _signature(
        *,
        radial_line_fraction: float,
        radial_line_density: float,
        line_density: float,
        line_count: int,
        edge_pixel_fraction: float,
    ) -> dict[str, object]:
        return {
            "metrics": {
                "contour": {
                    "contour_tightening_removed_fraction": 0.002,
                    "silhouette_compactness": 2.2,
                },
                "texture": {
                    "coarse_gradient_p90": 0.10,
                    "structure_tensor_coherent_fraction": 0.48,
                    "fine_radial_brush_fraction": 0.37,
                    "coherent_long_line_energy": 0.10,
                    "hough": {
                        "line_count": line_count,
                        "line_density": line_density,
                        "radial_line_density": radial_line_density,
                        "radial_line_fraction": radial_line_fraction,
                        "edge_pixel_fraction": edge_pixel_fraction,
                    },
                },
            }
        }

    def test_sparse_hough_ratio_is_not_absolute_long_fur_evidence(self) -> None:
        signature = self._signature(
            radial_line_fraction=0.72059702,
            radial_line_density=0.003041,
            line_density=0.00422015,
            line_count=16,
            edge_pixel_fraction=0.00649166,
        )

        issues = surface_gate._absolute_issues(
            signature,
            surface_gate.FurSurfaceThresholds(),
        )

        self.assertNotIn("radial_long_lines", issues)

    def test_loaded_hough_radial_lines_remain_fail_closed(self) -> None:
        signature = self._signature(
            radial_line_fraction=0.7647943,
            radial_line_density=0.016554,
            line_density=0.02164735,
            line_count=76,
            edge_pixel_fraction=0.03858144,
        )

        issues = surface_gate._absolute_issues(
            signature,
            surface_gate.FurSurfaceThresholds(),
        )

        self.assertIn("radial_long_lines", issues)


class SubjectReplacementParsingTests(unittest.TestCase):
    def test_default_profile_uses_visually_accepted_action_strengths(self) -> None:
        profile = pipeline.load_subject_profile("sleek_tabby_cat_v1")

        self.assertEqual(
            profile["runtime"]["actions"]["eating"]["pose_strength"],
            1.1,
        )
        self.assertEqual(
            profile["runtime"]["actions"]["grooming"]["pose_strength"],
            1.15,
        )
        self.assertTrue(
            profile["runtime"]["actions"]["eating"]["driving_animation"].endswith(
                "20260731-main-case-v1/eating_driver_512x512_81f_16fps.mp4"
            )
        )
        self.assertTrue(
            profile["runtime"]["actions"]["grooming"]["driving_animation"].endswith(
                "20260731-main-case-v1/grooming_driver_512x512_81f_16fps.mp4"
            )
        )
        self.assertRegex(
            profile["runtime"]["actions"]["eating"]["driving_animation_sha256"],
            r"^[0-9a-f]{64}$",
        )
        self.assertRegex(
            profile["runtime"]["actions"]["grooming"]["driving_animation_sha256"],
            r"^[0-9a-f]{64}$",
        )

    def test_actions_default_to_eating_and_parse_small_dog_action_vocabulary(self) -> None:
        self.assertEqual(pipeline.parse_subject_actions(None), ["eating"])
        self.assertEqual(
            pipeline.parse_subject_actions('["吃饭", "舔毛", "eating"]'),
            ["eating", "grooming"],
        )
        self.assertEqual(
            pipeline.parse_subject_actions(
                '["喝水", "running", "睡觉", "上厕所"]'
            ),
            ["drinking", "running", "sleeping", "toileting"],
        )
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            pipeline.parse_subject_actions('["dancing"]')

    def test_dog_actions_use_standing_and_map_legacy_grooming_alias(self) -> None:
        self.assertEqual(
            pipeline.parse_subject_actions('["eating", "standing"]'),
            ["eating", "standing"],
        )
        self.assertEqual(
            pipeline.normalize_subject_actions_for_species(
                ["eating", "grooming"],
                "dog",
            ),
            ["eating", "standing"],
        )
        with self.assertRaisesRegex(ValueError, "cat subject replacement"):
            pipeline.normalize_subject_actions_for_species(["standing"], "cat")
        with self.assertRaisesRegex(ValueError, "dog subject replacement"):
            pipeline.normalize_subject_actions_for_species(["running"], "dog")
        small_profile = pipeline.load_subject_profile(
            "dog_small_white_model_test_v1"
        )
        self.assertEqual(
            pipeline.normalize_subject_actions_for_species(
                ["drinking", "running", "sleeping", "toileting"],
                "dog",
                supported_actions=pipeline.subject_profile_actions(small_profile),
            ),
            ["drinking", "running", "sleeping", "toileting"],
        )

    def test_dog_profile_exposes_digest_bound_eating_and_standing_assets(self) -> None:
        profile = pipeline.load_subject_profile("dog_white_model_v1")
        runtime = pipeline.subject_profile_runtime(profile)

        self.assertEqual(profile["species"], "dog")
        self.assertEqual(
            list(pipeline.subject_profile_actions(profile)),
            ["eating", "standing"],
        )
        self.assertEqual(set(runtime["actions"]), {"eating", "standing"})
        self.assertEqual(
            profile["uploaded_identity_driving"]["revision"],
            "dog-stable-kibble-standing-pant-20260818-v4",
        )
        self.assertEqual(
            profile["uploaded_identity_driving"]["actions"]["eating"]["loop_policy"],
            "full_source_down_up_down_cycle_2x_v1",
        )
        self.assertGreaterEqual(
            profile["uploaded_identity_driving"]["actions"]["eating"]["endpoint_ssim"],
            0.95,
        )
        food_stabilization = profile["uploaded_identity_driving"]["actions"][
            "eating"
        ]["food_stabilization"]
        self.assertEqual(
            food_stabilization["policy"],
            "fixed_low_level_small_kibble_below_rim_v1",
        )
        self.assertGreater(
            food_stabilization["food_pattern_top_y"],
            food_stabilization["bowl_rim_top_y"],
        )
        self.assertFalse(food_stabilization["food_above_rim_allowed"])
        self.assertEqual(food_stabilization["frames_audited"], 81)
        self.assertEqual(food_stabilization["forbidden_food_pixels_above_rim"], 0)
        self.assertTrue(food_stabilization["passed"])
        asset_manifest = json.loads(
            Path(profile["profile_asset_manifest"]["path"]).read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            asset_manifest["policy"],
            "dog_white_model_action_specific_loop_normalization_standing_pant_v4",
        )
        manifest_food = asset_manifest["actions"]["eating"][
            "food_stabilization"
        ]
        self.assertEqual(
            manifest_food["food_pattern_sha256"],
            food_stabilization["food_pattern_sha256"],
        )
        self.assertEqual(manifest_food["food_pattern_top_y"], 373)
        self.assertEqual(manifest_food["forbidden_food_pixels_above_rim"], 0)
        self.assertTrue(manifest_food["passed"])
        manifest_mouth = asset_manifest["actions"]["standing"][
            "mouth_tongue_motion"
        ]
        self.assertEqual(
            manifest_mouth["policy"],
            "relaxed_open_mouth_attached_tongue_extension_retraction_v1",
        )
        self.assertGreaterEqual(
            manifest_mouth["tongue_tip_y_span_pixels"],
            manifest_mouth["minimum_tongue_tip_y_span_pixels"],
        )
        self.assertGreaterEqual(
            manifest_mouth["visible_mouth_cavity_dark_pixels_min"],
            manifest_mouth["minimum_visible_mouth_cavity_dark_pixels"],
        )
        self.assertEqual(manifest_mouth["frames_audited"], 81)
        self.assertTrue(manifest_mouth["passed"])
        effective = pipeline.resolve_effective_subject_config(
            profile,
            ["eating", "standing"],
            seed=None,
            scail_steps=None,
            pose_strength=None,
        )
        self.assertEqual(effective["actions"]["eating"]["output_playback_rate"], 0.5)
        self.assertEqual(
            effective["actions"]["eating"]["output_playback_policy"],
            "restore_original_eating_source_tempo_v1",
        )
        self.assertEqual(
            effective["actions"]["standing"]["output_playback_rate"], 1.0
        )
        eating_timing = pipeline.subject_replacement_action_output_timing(
            effective,
            "eating",
        )
        standing_timing = pipeline.subject_replacement_action_output_timing(
            effective,
            "standing",
        )
        self.assertEqual(
            eating_timing["generation"],
            {"frames": 81, "fps": 16.0, "duration_seconds": 5.0625},
        )
        self.assertEqual(
            eating_timing["video"],
            {"frames": 81, "fps": 8.0, "duration_seconds": 10.125},
        )
        self.assertEqual(
            eating_timing["transparent_gif"],
            {"frames": 100, "fps": 10, "duration_seconds": 10.0},
        )
        self.assertEqual(standing_timing["video"]["fps"], 16.0)
        self.assertEqual(standing_timing["transparent_gif"]["fps"], 20)
        self.assertEqual(standing_timing["transparent_gif"]["duration_seconds"], 5.0)
        public_actions = {
            action["id"]: action
            for action in pipeline.subject_profile_public(profile)["actions"]
        }
        self.assertEqual(public_actions["eating"]["output_playback_rate"], 0.5)
        self.assertEqual(public_actions["standing"]["output_playback_rate"], 1.0)
        eating_driver = effective["actions"]["eating"][
            "uploaded_identity_driving"
        ]
        self.assertEqual(
            eating_driver["loop_policy"],
            "full_source_down_up_down_cycle_2x_v1",
        )
        self.assertEqual(eating_driver["selected_frame_index"], 8)
        self.assertEqual(eating_driver["minimum_endpoint_ssim"], 0.95)
        frozen, _ = pipeline.configure_uploaded_subject_validation(
            effective,
            ["eating", "standing"],
        )
        self.assertEqual(
            frozen["actions"]["eating"]["geometry_reference_kind"],
            "uploaded_white_maquette_frame008_rgba",
        )
        self.assertEqual(
            frozen["actions"]["standing"]["geometry_reference_kind"],
            "uploaded_white_maquette_frame040_rgba",
        )
        self.assertTrue(
            runtime["actions"]["eating"]["driving_animation"].endswith(
                "20260818-standing-pant-v4/"
                "eating_white_model_driver_512x512_81f_16fps.mp4"
            )
        )
        self.assertEqual(
            runtime["actions"]["eating"]["driving_animation_sha256"],
            "b6009f9ca7f12c5233c4e6b172b50d3fe6a69e8bc2faf97e6f2593843d5c7d20",
        )
        self.assertEqual(
            runtime["actions"]["standing"]["driving_animation_sha256"],
            "9f2f951fdb46e08249e4b553dbdcd55e1940fded56e92cfe8462bd37ba40f1e0",
        )
        for action in ("eating", "standing"):
            action_runtime = runtime["actions"][action]
            self.assertTrue(Path(action_runtime["driving_animation"]).is_file())
            self.assertRegex(
                action_runtime["driving_animation_sha256"],
                r"^[0-9a-f]{64}$",
            )

    def test_profile_rejects_out_of_range_output_playback_rate(self) -> None:
        profile = pipeline.load_subject_profile("dog_white_model_v1")
        profile["runtime"]["actions"]["eating"]["output_playback_rate"] = 0.1

        with self.assertRaisesRegex(ValueError, "output_playback_rate"):
            pipeline.subject_profile_runtime(profile)

    def test_profile_exposes_action_specific_references(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            eating = root / "eat.png"
            grooming = root / "groom.png"
            eating.touch()
            grooming.touch()
            profile = {
                "id": "test_subject",
                "display_name": "测试主体",
                "preview_action": "eating",
                "action_references": {
                    "eating": str(eating),
                    "grooming": str(grooming),
                },
            }
            (root / "test_subject.json").write_text(
                json.dumps(profile, ensure_ascii=False),
                encoding="utf-8",
            )
            with patch.object(pipeline, "SUBJECT_PROFILE_DIR", root):
                loaded = pipeline.load_subject_profile("test_subject")
                public = pipeline.subject_profile_public(loaded)

        self.assertEqual(public["id"], "test_subject")
        self.assertEqual(
            [item["id"] for item in public["actions"]],
            ["eating", "grooming"],
        )
        self.assertTrue(all(item["reference_ready"] for item in public["actions"]))

    def test_scail_retry_uses_a_new_experiment_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "job-eating").mkdir()
            (root / "job-eating-retry001").mkdir()
            with patch.object(pipeline, "SCAIL_EXPERIMENT_ROOT", root):
                run_id = pipeline.available_scail_run_id("job", "eating")

        self.assertEqual(run_id, "job-eating-retry002")

    def test_uploaded_eating_scail_uses_action_pose_with_bowl_masked_out(self) -> None:
        job = {"canonical_reference_path": "/tmp/canonical-cat-only.png"}
        profile = {
            "action_references": {
                "eating": "/tmp/profile-eating.png",
                "grooming": "/tmp/profile-grooming.png",
            }
        }
        generated = {
            "eating": "/tmp/generated-eating-with-bowl.png",
            "grooming": "/tmp/generated-grooming.png",
        }

        eating_path, eating_policy = pipeline.select_scail_subject_reference(
            job,
            profile,
            generated,
            "eating",
        )
        grooming_path, grooming_policy = pipeline.select_scail_subject_reference(
            job,
            profile,
            generated,
            "grooming",
        )

        self.assertEqual(eating_path, Path("/tmp/generated-eating-with-bowl.png"))
        self.assertEqual(
            eating_policy,
            "qwen_action_identity_with_bowl_mapped_red",
        )
        self.assertEqual(grooming_path, Path("/tmp/generated-grooming.png"))
        self.assertEqual(grooming_policy, "qwen_action_specific_identity")

    def test_profile_eating_scail_uses_action_reference_with_bowl_masked_out(
        self,
    ) -> None:
        path, policy = pipeline.select_scail_subject_reference(
            {},
            {
                "action_references": {
                    "eating": "/tmp/profile-eating-with-bowl.png",
                },
            },
            None,
            "eating",
        )

        self.assertEqual(path, Path("/tmp/profile-eating-with-bowl.png"))
        self.assertEqual(
            policy,
            "subject_profile_action_identity_with_bowl_mapped_red",
        )

    def test_sleek_uploaded_eating_scail_uses_generated_upload_reference(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile_reference = root / "frozen-profile-eating.png"
            generated_reference = root / "generated-long-fur.png"
            profile_reference.write_bytes(b"profile-eating")
            generated_reference.write_bytes(b"generated-eating")
            job = {
                "effective_subject_config": {
                    "profile_id": "sleek_tabby_cat_v1",
                    "actions": {
                        "eating": {
                            "reference": str(generated_reference),
                            "reference_sha256": pipeline.file_sha256(
                                generated_reference
                            ),
                            "reference_kind": "qwen_generated_upload",
                            "profile_reference": str(profile_reference),
                            "profile_reference_sha256": pipeline.file_sha256(
                                profile_reference
                            ),
                        }
                    },
                }
            }

            path, policy = pipeline.select_scail_subject_reference(
                job,
                None,
                {"eating": str(generated_reference)},
                "eating",
            )

        self.assertEqual(path, generated_reference)
        self.assertEqual(
            policy,
            "qwen_action_identity_with_bowl_mapped_red",
        )

    def test_sleek_eating_scail_fails_closed_if_frozen_digest_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            profile_reference = Path(temporary) / "frozen-profile-eating.png"
            profile_reference.write_bytes(b"original")
            job = {
                "effective_subject_config": {
                    "profile_id": "sleek_tabby_cat_v1",
                    "actions": {
                        "eating": {
                            "reference": str(profile_reference),
                            "reference_sha256": pipeline.file_sha256(profile_reference),
                        }
                    },
                }
            }
            profile_reference.write_bytes(b"changed")

            with self.assertRaisesRegex(RuntimeError, "digest changed"):
                pipeline.select_scail_subject_reference(
                    job,
                    None,
                    None,
                    "eating",
                )

    def test_default_sleek_eating_keeps_frozen_profile_reference(
        self,
    ) -> None:
        frozen, qwen = pipeline.partition_subject_reference_actions(
            {"profile_id": "sleek_tabby_cat_v1"},
            ["eating", "grooming"],
        )

        self.assertEqual(frozen, ["eating"])
        self.assertEqual(qwen, ["grooming"])

    def test_uploaded_sleek_subject_generates_every_requested_action(
        self,
    ) -> None:
        frozen, qwen = pipeline.partition_subject_reference_actions(
            {"profile_id": "sleek_tabby_cat_v1"},
            ["eating", "grooming"],
            uploaded_identity=True,
        )

        self.assertEqual(frozen, [])
        self.assertEqual(qwen, ["eating", "grooming"])

    def test_other_profile_keeps_all_requested_qwen_actions(self) -> None:
        frozen, qwen = pipeline.partition_subject_reference_actions(
            {"profile_id": "other_cat_v1"},
            ["eating", "grooming"],
        )

        self.assertEqual(frozen, [])
        self.assertEqual(qwen, ["eating", "grooming"])


class SubjectReplacementEffectiveConfigTests(unittest.TestCase):
    def test_dog_eating_pose_policy_accepts_identity_safe_static_fallback(
        self,
    ) -> None:
        for actual_policy in (
            pipeline.DOG_EATING_REFERENCE_ALIGNED_POSE_POLICY,
            pipeline.DOG_EATING_STATIC_POSE_FALLBACK_POLICY,
        ):
            pipeline.require_expected_scail_pose_policy(
                {"pose_metadata": {"policy": actual_policy}},
                "eating",
                pipeline.DOG_EATING_REFERENCE_ALIGNED_POSE_POLICY,
            )

        with self.assertRaisesRegex(
            RuntimeError,
            "SCAIL eating pose policy changed before GPU submission",
        ):
            pipeline.require_expected_scail_pose_policy(
                {
                    "pose_metadata": {
                        "policy": pipeline.FULL_DRIVER_EATING_POSE_POLICY,
                    }
                },
                "eating",
                pipeline.DOG_EATING_REFERENCE_ALIGNED_POSE_POLICY,
            )

    def test_cat_full_driver_policy_still_rejects_dog_static_fallback(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            RuntimeError,
            "SCAIL eating pose policy changed before GPU submission",
        ):
            pipeline.require_expected_scail_pose_policy(
                {
                    "pose_metadata": {
                        "policy": pipeline.DOG_EATING_STATIC_POSE_FALLBACK_POLICY,
                    }
                },
                "eating",
                pipeline.FULL_DRIVER_EATING_POSE_POLICY,
            )

    def test_default_upload_surface_baselines_satisfy_frozen_contract(
        self,
    ) -> None:
        profile = pipeline.load_subject_profile("sleek_tabby_cat_v1")
        actions = ["eating", "grooming"]
        effective = pipeline.resolve_effective_subject_config(
            profile,
            actions,
            seed=None,
            scail_steps=None,
            pose_strength=None,
        )
        effective, _ = pipeline.configure_uploaded_subject_validation(
            effective,
            actions,
        )

        with tempfile.TemporaryDirectory() as temporary:
            snapshot = pipeline.snapshot_effective_subject_references(
                effective,
                Path(temporary),
                actions,
            )

        for action in actions:
            baseline = snapshot["actions"][action]["fur_surface_gate"]["baseline"]
            self.assertTrue(baseline["absolute_contract"]["acceptable"])
            self.assertEqual(
                baseline["absolute_contract"]["issues"],
                [],
            )

    def test_v2_profile_resolves_runtime_defaults_and_action_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            profile = _subject_profile_v2(Path(temporary))
            effective = pipeline.resolve_effective_subject_config(
                profile,
                ["eating", "grooming"],
                seed=None,
                scail_steps=None,
                pose_strength=None,
            )

        self.assertEqual(effective["profile_schema_version"], 2)
        self.assertEqual(effective["runtime_source"], "profile.runtime")
        self.assertEqual(effective["engine_profile"], "scail2-14b-fp8-v1")
        self.assertEqual(effective["prompt_profile"], "sleek-tabby-scail-v1")
        self.assertEqual(effective["model"], pipeline.SCAIL_UNET)
        self.assertEqual(effective["frames"], 81)
        self.assertEqual(effective["fps"], 16.0)
        self.assertEqual(effective["steps"], 6)
        self.assertEqual(effective["base_seed"], 20260726)
        self.assertEqual(effective["seed_mode"], "base_plus_action_offset")
        self.assertEqual(effective["actions"]["eating"]["effective_seed"], 20260726)
        self.assertEqual(effective["actions"]["grooming"]["effective_seed"], 20260728)
        self.assertEqual(effective["actions"]["eating"]["pose_strength"], 1.1)
        self.assertEqual(effective["actions"]["grooming"]["pose_strength"], 1.15)
        self.assertEqual(
            effective["actions"]["eating"]["driving_animation_sha256"],
            profile["runtime"]["actions"]["eating"]["driving_animation_sha256"],
        )
        self.assertEqual(
            effective["profile_fur_constraints"]["positive"],
            "frozen close-lying test coat",
        )
        self.assertTrue(
            effective["actions"]["eating"]["fur_silhouette_gate"]["enabled"]
        )

    def test_changed_driving_animation_digest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            profile = _subject_profile_v2(Path(temporary))
            driving = Path(profile["runtime"]["actions"]["eating"]["driving_animation"])
            driving.write_bytes(b"changed-driving-animation")

            with self.assertRaisesRegex(
                ValueError,
                "eating driving animation digest changed",
            ):
                pipeline.resolve_effective_subject_config(
                    profile,
                    ["eating"],
                    seed=None,
                    scail_steps=None,
                    pose_strength=None,
                )

    def test_eating_pose_strength_above_sleek_limit_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            profile = _subject_profile_v2(Path(temporary))
            with self.assertRaisesRegex(
                ValueError,
                "eating pose_strength must not exceed 1.35",
            ):
                pipeline.resolve_effective_subject_config(
                    profile,
                    ["eating"],
                    seed=None,
                    scail_steps=None,
                    pose_strength=2.0,
                )

    def test_eating_only_upload_uses_qwen_reference_and_never_profile_fallback(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_id = "test-sleek-eating-bypass"
            job_dir = root / job_id
            job_dir.mkdir()
            reference = root / "eating-profile.png"
            generated_reference = root / "qwen-eating.png"
            reference.write_bytes(b"frozen-eating")
            generated_reference.write_bytes(b"qwen-eating")
            original_started_at = "2026-08-03T00:00:00+00:00"
            jobs = {
                job_id: {
                    "job_id": job_id,
                    "actions": ["eating"],
                    "subject_source": "upload_pending_qwen",
                    "started_at": original_started_at,
                    "progress": 0,
                    "expected_pose_policies": {
                        "eating": pipeline.FULL_DRIVER_EATING_POSE_POLICY,
                    },
                    "effective_subject_config": {
                        "profile_id": "sleek_tabby_cat_v1",
                        "frames": 81,
                        "fps": 16.0,
                        "steps": 6,
                        "upload_validation": _qwen_upload_validation(("eating",)),
                        "actions": {
                            "eating": {
                                "reference": str(reference),
                                "reference_sha256": pipeline.file_sha256(reference),
                                "driving_animation": str(reference),
                                "pose_strength": 1.1,
                                "effective_seed": 123,
                                "fur_surface_gate": _test_fur_surface_gate(
                                    "eating",
                                    reference,
                                ),
                            }
                        },
                    },
                }
            }

            def fake_prepare(_job_id: str, qwen_actions: list[str]) -> None:
                self.assertEqual(qwen_actions, ["eating"])
                updated = json.loads(
                    json.dumps(jobs[job_id]["effective_subject_config"])
                )
                eating = updated["actions"]["eating"]
                eating["profile_reference"] = eating["reference"]
                eating["profile_reference_sha256"] = eating["reference_sha256"]
                eating["reference"] = str(generated_reference)
                eating["reference_sha256"] = pipeline.file_sha256(generated_reference)
                eating["reference_kind"] = "qwen_generated_upload"
                pipeline.update_job(
                    job_id,
                    subject_source="qwen_generated_upload",
                    subject_reference_paths={
                        "eating": str(generated_reference),
                    },
                    qwen_subject_reference_paths={
                        "eating": str(generated_reference),
                    },
                    reference_generation_completed=True,
                    stage="subject_references_ready",
                    progress=35,
                    effective_subject_config=updated,
                )

            progress_updates: list[float] = []
            started_at_updates: list[str] = []
            route_events: list[tuple[str, str]] = []
            real_update_job = pipeline.update_job

            def recording_update_job(
                target_job_id: str,
                **changes: object,
            ) -> dict[str, object]:
                if "progress" in changes:
                    progress_updates.append(float(changes["progress"]))
                if "started_at" in changes:
                    started_at_updates.append(str(changes["started_at"]))
                return real_update_job(target_job_id, **changes)

            def record_revalidation(*_args: object, **_kwargs: object) -> None:
                route_events.append(
                    ("revalidate", str(jobs[job_id].get("stage") or ""))
                )

            def record_wait(*_args: object, **_kwargs: object) -> None:
                route_events.append(("wait", str(jobs[job_id].get("stage") or "")))

            def stop_at_scail(*_args: object, **_kwargs: object) -> None:
                route_events.append(("scail", str(jobs[job_id].get("stage") or "")))
                raise RuntimeError("stop-at-scail")

            with (
                patch.multiple(
                    pipeline,
                    OUTPUT_ROOT=root,
                    JOBS=jobs,
                    SCAIL_EXPERIMENT_ROOT=root / "scail-experiments",
                ),
                patch.object(
                    pipeline,
                    "update_job",
                    side_effect=recording_update_job,
                ),
                patch.object(
                    pipeline,
                    "prepare_uploaded_subject_qwen_references",
                    side_effect=fake_prepare,
                ) as prepare_qwen,
                patch.object(
                    pipeline,
                    "revalidate_uploaded_qwen_references_before_scail",
                    side_effect=record_revalidation,
                ) as revalidate_qwen,
                patch.object(pipeline, "require_subject_replacement_assets"),
                patch.object(
                    pipeline,
                    "wait_comfy_idle",
                    side_effect=record_wait,
                ),
                patch.object(pipeline, "free_comfy_models"),
                patch.object(
                    pipeline,
                    "run_cancelable_process",
                    side_effect=stop_at_scail,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "stop-at-scail"):
                    pipeline.run_subject_replacement_job(job_id)

            prepare_qwen.assert_called_once_with(job_id, ["eating"])
            self.assertEqual(revalidate_qwen.call_count, 2)
            self.assertEqual(
                route_events,
                [
                    ("revalidate", "subject_references_ready"),
                    ("wait", "waiting_for_gpu"),
                    ("revalidate", "preparing_subject_replacement"),
                    ("scail", "preparing_subject_replacement"),
                ],
            )
            self.assertEqual(progress_updates, [35.0, 36.0, 38.0])
            self.assertTrue(
                all(
                    later >= earlier
                    for earlier, later in zip(
                        progress_updates,
                        progress_updates[1:],
                    )
                )
            )
            self.assertEqual(started_at_updates, [original_started_at])
            final_job = jobs[job_id]
            self.assertEqual(final_job["started_at"], original_started_at)
            self.assertEqual(
                final_job["subject_source"],
                "qwen_generated_upload",
            )
            self.assertEqual(
                final_job["effective_subject_source"],
                "qwen_generated_upload",
            )
            self.assertEqual(
                final_job["reference_pipeline"],
                pipeline.QWEN_FULL_BODY_FIRST_FRAME_PIPELINE,
            )
            self.assertEqual(
                final_job["subject_reference_paths"],
                {"eating": str(generated_reference)},
            )
            self.assertEqual(
                final_job["scail_reference_policy"],
                {
                    "eating": "qwen_action_identity_with_bowl_mapped_red",
                },
            )
            self.assertEqual(
                final_job["uploaded_identity_applied_actions"],
                ["eating"],
            )
            self.assertEqual(
                final_job["uploaded_identity_excluded_actions"],
                [],
            )

    def test_mixed_upload_uses_qwen_identity_for_both_actions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_id = "test-sleek-mixed-references"
            job_dir = root / job_id
            job_dir.mkdir()
            eating_reference = root / "eating-profile.png"
            grooming_reference = root / "grooming-profile.png"
            qwen_eating = root / "qwen-eating.png"
            qwen_grooming = root / "qwen-grooming.png"
            eating_reference.write_bytes(b"frozen-eating")
            grooming_reference.write_bytes(b"frozen-grooming")
            qwen_eating.write_bytes(b"qwen-eating")
            qwen_grooming.write_bytes(b"qwen-grooming")
            jobs = {
                job_id: {
                    "job_id": job_id,
                    "actions": ["eating", "grooming"],
                    "subject_source": "upload_pending_qwen",
                    "expected_pose_policies": {
                        "eating": pipeline.FULL_DRIVER_EATING_POSE_POLICY,
                    },
                    "effective_subject_config": {
                        "profile_id": "sleek_tabby_cat_v1",
                        "upload_validation": _qwen_upload_validation(
                            ("eating", "grooming")
                        ),
                        "actions": {
                            "eating": {
                                "reference": str(eating_reference),
                                "reference_sha256": pipeline.file_sha256(
                                    eating_reference
                                ),
                                "fur_surface_gate": _test_fur_surface_gate(
                                    "eating",
                                    eating_reference,
                                ),
                            },
                            "grooming": {
                                "reference": str(grooming_reference),
                                "reference_sha256": pipeline.file_sha256(
                                    grooming_reference
                                ),
                                "fur_surface_gate": _test_fur_surface_gate(
                                    "grooming",
                                    grooming_reference,
                                ),
                            },
                        },
                    },
                }
            }

            def fake_prepare(_job_id: str, qwen_actions: list[str]) -> None:
                self.assertEqual(qwen_actions, ["eating", "grooming"])
                updated = json.loads(
                    json.dumps(jobs[job_id]["effective_subject_config"])
                )
                eating = updated["actions"]["eating"]
                grooming = updated["actions"]["grooming"]
                for action, generated in (
                    (eating, qwen_eating),
                    (grooming, qwen_grooming),
                ):
                    action["profile_reference"] = action["reference"]
                    action["profile_reference_sha256"] = action["reference_sha256"]
                    action["reference"] = str(generated)
                    action["reference_sha256"] = pipeline.file_sha256(generated)
                    action["reference_kind"] = "qwen_generated_upload"
                pipeline.update_job(
                    job_id,
                    subject_source="qwen_generated_upload",
                    subject_reference_paths={
                        "eating": str(qwen_eating),
                        "grooming": str(qwen_grooming),
                    },
                    qwen_subject_reference_paths={
                        "eating": str(qwen_eating),
                        "grooming": str(qwen_grooming),
                    },
                    reference_generation_completed=True,
                    effective_subject_config=updated,
                )

            with (
                patch.multiple(pipeline, OUTPUT_ROOT=root, JOBS=jobs),
                patch.object(
                    pipeline,
                    "prepare_uploaded_subject_qwen_references",
                    side_effect=fake_prepare,
                ) as prepare_qwen,
                patch.object(
                    pipeline,
                    "revalidate_uploaded_qwen_references_before_scail",
                ),
                patch.object(
                    pipeline,
                    "require_subject_replacement_assets",
                    side_effect=RuntimeError("stop-before-gpu"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "stop-before-gpu"):
                    pipeline.run_subject_replacement_job(job_id)

            prepare_qwen.assert_called_once_with(
                job_id,
                ["eating", "grooming"],
            )
            final_job = jobs[job_id]
            self.assertEqual(
                final_job["reference_pipeline"],
                pipeline.QWEN_FULL_BODY_FIRST_FRAME_PIPELINE,
            )
            self.assertEqual(
                final_job["subject_reference_paths"],
                {
                    "eating": str(qwen_eating),
                    "grooming": str(qwen_grooming),
                },
            )
            self.assertEqual(
                final_job["qwen_reference_applied_actions"],
                ["eating", "grooming"],
            )
            self.assertEqual(
                final_job["qwen_reference_bypassed_actions"],
                [],
            )
            self.assertEqual(
                final_job["uploaded_identity_applied_actions"],
                ["eating", "grooming"],
            )
            self.assertEqual(
                final_job["uploaded_identity_excluded_actions"],
                [],
            )

    def test_uploaded_subject_fails_closed_before_gpu_without_qwen_identity(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_id = "test-upload-no-fallback"
            (root / job_id).mkdir()
            reference = root / "profile.png"
            reference.write_bytes(b"profile")
            jobs = {
                job_id: {
                    "job_id": job_id,
                    "actions": ["eating"],
                    "subject_source": "upload_pending_qwen",
                    "expected_pose_policies": {
                        "eating": pipeline.FULL_DRIVER_EATING_POSE_POLICY,
                    },
                    "effective_subject_config": {
                        "profile_id": "sleek_tabby_cat_v1",
                        "upload_validation": _qwen_upload_validation(("eating",)),
                        "actions": {
                            "eating": {
                                "reference": str(reference),
                                "reference_sha256": pipeline.file_sha256(reference),
                                "fur_surface_gate": _test_fur_surface_gate(
                                    "eating",
                                    reference,
                                ),
                            }
                        },
                    },
                }
            }

            with (
                patch.multiple(pipeline, OUTPUT_ROOT=root, JOBS=jobs),
                patch.object(
                    pipeline,
                    "prepare_uploaded_subject_qwen_references",
                ),
                patch.object(
                    pipeline,
                    "require_subject_replacement_assets",
                ) as require_assets,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Uploaded subject identity was not applied",
                ):
                    pipeline.run_subject_replacement_job(job_id)

            require_assets.assert_not_called()

    def test_upload_without_surface_gate_fails_before_qwen(self) -> None:
        job_id = "test-upload-without-surface-gate"
        jobs = {
            job_id: {
                "job_id": job_id,
                "actions": ["eating"],
                "subject_source": "upload_pending_qwen",
                "expected_pose_policies": {
                    "eating": pipeline.FULL_DRIVER_EATING_POSE_POLICY,
                },
                "effective_subject_config": {
                    "profile_id": "sleek_tabby_cat_v1",
                    "upload_validation": _qwen_upload_validation(("eating",)),
                    "actions": {"eating": {}},
                },
            }
        }
        with (
            patch.object(pipeline, "JOBS", jobs),
            patch.object(
                pipeline,
                "GENERATION_VISUAL_AUDITS_BLOCKING",
                True,
            ),
            patch.object(
                pipeline,
                "prepare_uploaded_subject_qwen_references",
            ) as prepare_qwen,
        ):
            with self.assertRaisesRegex(RuntimeError, "fur surface gate"):
                pipeline.run_subject_replacement_job(job_id)

        prepare_qwen.assert_not_called()

    def test_retired_direct_upload_states_require_resubmission(self) -> None:
        for retired_state in (
            "upload_pending_scail_reference",
            "uploaded_identity_direct_scail",
        ):
            with self.subTest(retired_state=retired_state):
                job_id = f"test-retired-{retired_state}"
                jobs = {
                    job_id: {
                        "job_id": job_id,
                        "actions": ["eating"],
                        "subject_source": retired_state,
                        "effective_subject_config": {
                            "profile_id": "sleek_tabby_cat_v1",
                            "actions": {"eating": {}},
                        },
                    }
                }
                with (
                    patch.object(pipeline, "JOBS", jobs),
                    patch.object(
                        pipeline,
                        "prepare_uploaded_subject_qwen_references",
                    ) as prepare_qwen,
                    patch.object(
                        pipeline,
                        "prepare_uploaded_subject_scail_references",
                    ) as prepare_direct,
                ):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "Submit the upload again",
                    ):
                        pipeline.run_subject_replacement_job(job_id)

                prepare_qwen.assert_not_called()
                prepare_direct.assert_not_called()

        with self.assertRaisesRegex(
            RuntimeError,
            "Direct uploaded-image SCAIL references are retired",
        ):
            pipeline.prepare_uploaded_subject_scail_references(
                "retired-direct-builder",
                ["eating"],
            )

    def test_reference_snapshot_copies_content_and_records_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = _subject_profile_v2(root)
            effective = pipeline.resolve_effective_subject_config(
                profile,
                ["eating", "grooming"],
                seed=None,
                scail_steps=None,
                pose_strength=None,
            )
            job_dir = root / "job"
            snapshot = pipeline.snapshot_effective_subject_references(
                effective,
                job_dir,
                ["eating", "grooming"],
            )
            eating_source = Path(snapshot["actions"]["eating"]["reference_source"])
            eating_snapshot = Path(snapshot["actions"]["eating"]["reference"])
            frozen_bytes = eating_snapshot.read_bytes()
            Image.new("RGB", (32, 32), "blue").save(eating_source)

            self.assertTrue(eating_snapshot.is_file())
            self.assertEqual(eating_snapshot.parent, job_dir / "profile_inputs")
            self.assertEqual(eating_snapshot.read_bytes(), frozen_bytes)
            self.assertNotEqual(eating_source.read_bytes(), frozen_bytes)
            self.assertEqual(
                snapshot["actions"]["eating"]["reference_sha256"],
                pipeline.file_sha256(eating_snapshot),
            )
            baseline_snapshot = Path(
                snapshot["actions"]["eating"]["fur_contour_baseline"]["path"]
            )
            self.assertTrue(baseline_snapshot.is_file())
            self.assertEqual(baseline_snapshot.parent, job_dir / "profile_inputs")
            self.assertEqual(
                snapshot["actions"]["eating"]["fur_contour_baseline"]["sha256"],
                pipeline.file_sha256(baseline_snapshot),
            )

    def test_worker_reference_selection_is_fixed_against_profile_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = _subject_profile_v2(root)
            effective = pipeline.resolve_effective_subject_config(
                profile,
                ["eating"],
                seed=None,
                scail_steps=None,
                pose_strength=None,
            )
            snapshot = pipeline.snapshot_effective_subject_references(
                effective,
                root / "job",
                ["eating"],
            )
            replacement = root / "changed-after-creation.png"
            Image.new("RGB", (32, 32), "red").save(replacement)
            profile["action_references"]["eating"] = str(replacement)
            profile["fur_constraints"]["positive"] = "mutated live coat rule"

            selected, _ = pipeline.select_scail_subject_reference(
                {"effective_subject_config": snapshot},
                profile,
                None,
                "eating",
            )

        self.assertEqual(selected, Path(snapshot["actions"]["eating"]["reference"]))
        self.assertNotEqual(selected, replacement)
        frozen_positive, _ = pipeline.subject_profile_fur_guard(
            {
                "id": snapshot["profile_id"],
                "fur_constraints": snapshot["profile_fur_constraints"],
            }
        )
        self.assertIn("uploaded pet", frozen_positive.lower())
        self.assertNotIn("frozen close-lying test coat", frozen_positive)
        self.assertNotIn("mutated live coat rule", frozen_positive)

    def test_uploaded_fur_guard_excludes_profile_animal_identity_terms(self) -> None:
        positive, negative = pipeline.subject_profile_fur_guard(
            {
                "id": "sleek_tabby_cat_v1",
                "fur_constraints": {
                    "positive": "exact Main_case black-and-white coat identity",
                    "negative": "old gray tabby identity, green eyes",
                },
            }
        )

        combined = f"{positive}, {negative}".lower()
        self.assertIn("uploaded identity appearance authority", combined)
        self.assertIn("action timing and geometry only", combined)
        self.assertNotIn("exact main_case black-and-white", combined)
        self.assertNotIn("old gray tabby identity", combined)
        self.assertNotIn("green eyes", combined)

    def test_uploaded_fur_guard_accepts_only_explicit_upload_extension(self) -> None:
        positive, negative = pipeline.subject_profile_fur_guard(
            {
                "fur_constraints": {
                    "positive": "profile identity must stay out",
                },
                "upload_fur_constraints": {
                    "positive": "keep source whisker length",
                    "negative": "invented whisker fan",
                },
            }
        )

        self.assertIn("keep source whisker length", positive)
        self.assertIn("invented whisker fan", negative)
        self.assertNotIn("profile identity must stay out", positive)


class TransparentGuidanceSelectionTests(unittest.TestCase):
    def test_broad_toileting_reference_uses_static_reference_mask(self) -> None:
        plan = pipeline.select_transparent_guidance_policy(
            "toileting",
            {
                "reference_metadata": {
                    "reference_alpha_bbox": [54, 56, 458, 460],
                }
            },
        )

        self.assertEqual(
            plan["policy"],
            pipeline.TOILETING_TRANSPARENT_REFERENCE_GUIDANCE_POLICY,
        )
        self.assertEqual(plan["source_key"], "reference_mask")
        self.assertTrue(plan["static_reference_mask"])
        self.assertTrue(plan["reference_aligned_uniform_background"])
        self.assertEqual(plan["reference_alpha_bbox_aspect_ratio"], 1.0)

    def test_narrow_toileting_and_other_actions_keep_driver_mask(self) -> None:
        narrow = pipeline.select_transparent_guidance_policy(
            "toileting",
            {
                "reference_metadata": {
                    "reference_alpha_bbox": [133, 50, 379, 462],
                }
            },
        )
        drinking = pipeline.select_transparent_guidance_policy(
            "drinking",
            {
                "reference_metadata": {
                    "reference_alpha_bbox": [54, 56, 458, 460],
                }
            },
        )

        self.assertEqual(
            narrow["policy"],
            pipeline.TOILETING_TRANSPARENT_DRIVER_GUIDANCE_POLICY,
        )
        self.assertEqual(narrow["source_key"], "pose_mask_video")
        self.assertFalse(narrow["static_reference_mask"])
        self.assertEqual(drinking["policy"], "driver_pose_mask_v1")
        self.assertEqual(drinking["source_key"], "pose_mask_video")

    def test_borderline_real_tray_reference_uses_static_reference_mask(self) -> None:
        plan = pipeline.select_transparent_guidance_policy(
            "toileting",
            {
                "reference_metadata": {
                    "reference_alpha_bbox": [108, 50, 404, 462],
                }
            },
        )

        self.assertAlmostEqual(
            plan["reference_alpha_bbox_aspect_ratio"],
            296 / 412,
        )
        self.assertEqual(
            plan["policy"],
            pipeline.TOILETING_TRANSPARENT_REFERENCE_GUIDANCE_POLICY,
        )
        self.assertEqual(plan["source_key"], "reference_mask")
        self.assertTrue(plan["static_reference_mask"])
        self.assertTrue(plan["reference_aligned_uniform_background"])


class ToiletingScailTemporalIdentityArtifactTests(unittest.TestCase):
    @staticmethod
    def _frame(*, invented_tail_band: bool) -> np.ndarray:
        frame = np.full((512, 512, 3), 255, dtype=np.uint8)
        coat = (226, 201, 160)
        cv2.ellipse(frame, (245, 245), (88, 170), 0, 0, 360, coat, -1)
        cv2.rectangle(frame, (91, 398), (421, 461), (205, 200, 190), -1)
        cv2.ellipse(frame, (344, 270), (52, 82), 0, 0, 360, coat, -1)
        cv2.rectangle(frame, (292, 246), (344, 294), coat, -1)
        if invented_tail_band:
            cv2.rectangle(frame, (349, 234), (357, 302), (8, 8, 8), -1)
        return frame

    def test_temporal_gate_rejects_repeated_invented_tail_band(self) -> None:
        clean = self._frame(invented_tail_band=False)
        bad = self._frame(invented_tail_band=True)

        quality = pipeline.toileting_temporal_identity_artifact_quality(
            [bad.copy() for _ in range(6)]
            + [clean.copy() for _ in range(6)]
        )

        self.assertFalse(quality["acceptable"], quality)
        self.assertEqual(quality["flagged_frame_count"], 6)
        self.assertIn(
            "toileting_temporal_lateral_dark_band_or_symbol",
            quality["issues"],
        )

    def test_temporal_gate_tolerates_sparse_natural_lateral_shadow(self) -> None:
        clean = self._frame(invented_tail_band=False)
        isolated = self._frame(invented_tail_band=True)

        quality = pipeline.toileting_temporal_identity_artifact_quality(
            [isolated.copy() for _ in range(3)]
            + [clean.copy() for _ in range(9)]
        )

        self.assertTrue(quality["acceptable"], quality)
        self.assertEqual(quality["flagged_frame_count"], 3)
        self.assertEqual(quality["permitted_flagged_frames"], 4)

    def test_retry_seed_copy_does_not_mutate_prepared_workflow(self) -> None:
        workflow = {
            "20": {
                "class_type": "SamplerCustom",
                "inputs": {"noise_seed": 20260835, "cfg": 1.0},
            },
            "23": {
                "class_type": "SaveVideo",
                "inputs": {"filename_prefix": "subject_replacement"},
            },
        }

        candidate, updated = pipeline.scail_workflow_with_retry_seed(
            workflow,
            seed=20260836,
        )

        self.assertEqual(updated, ["20"])
        self.assertEqual(candidate["20"]["inputs"]["noise_seed"], 20260836)
        self.assertEqual(workflow["20"]["inputs"]["noise_seed"], 20260835)


class CanonicalChromaIntegrityTests(unittest.TestCase):
    def test_canonical_cutout_rejects_large_central_neck_hole(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            clean_path = root / "clean.png"
            hole_path = root / "hole.png"
            clean = Image.new("RGBA", (200, 300), (0, 0, 0, 0))
            clean_draw = ImageDraw.Draw(clean)
            clean_draw.ellipse((25, 20, 175, 155), fill=(120, 120, 120, 255))
            clean_draw.rectangle((50, 110, 150, 280), fill=(120, 120, 120, 255))
            clean.save(clean_path)
            hole = clean.copy()
            ImageDraw.Draw(hole).rectangle((70, 112, 130, 130), fill=(0, 0, 0, 0))
            hole.save(hole_path)

            clean_quality = pipeline.canonical_cutout_interior_quality(clean_path)
            hole_quality = pipeline.canonical_cutout_interior_quality(hole_path)

        self.assertTrue(clean_quality["acceptable"], clean_quality)
        self.assertFalse(hole_quality["acceptable"])
        self.assertIn(
            "canonical_subject_interior_chroma_hole",
            hole_quality["issues"],
        )
        self.assertGreater(hole_quality["largest_void_area_fraction"], 0.004)

    def test_chroma_despill_preserves_green_eye_inside_subject(self) -> None:
        rgb = np.zeros((96, 96, 3), dtype=np.uint8)
        rgb[:, :] = (0, 255, 0)
        rgb[12:84, 24:72] = (120, 120, 120)
        rgb[46:51, 46:51] = (30, 160, 30)
        rgb[15:18, 30:34] = (120, 170, 120)

        cleaned, alpha = transparency._subject_mask(rgb.copy())

        self.assertGreaterEqual(int(alpha[48, 48]), 128)
        self.assertEqual(tuple(cleaned[48, 48]), (30, 160, 30))
        self.assertGreaterEqual(int(alpha[16, 31]), 128)
        self.assertLessEqual(
            int(cleaned[16, 31, 1]),
            max(int(cleaned[16, 31, 0]), int(cleaned[16, 31, 2])),
        )


class SubjectReplacementJobCreationTests(unittest.TestCase):
    def test_test_platform_dog_size_profile_selection_is_authoritative(self) -> None:
        for requested in (
            "dog_medium_white_model_test_v1",
            "dog_small_white_model_test_v1",
        ):
            with self.subTest(requested=requested):
                self.assertEqual(
                    pipeline.uploaded_subject_profile_selection("cat", requested),
                    (requested, "test_platform_explicit_dog_size_v1"),
                )

    def test_ordinary_profile_requests_keep_species_classifier_routing(self) -> None:
        self.assertEqual(
            pipeline.uploaded_subject_profile_selection(
                "dog",
                "sleek_tabby_cat_v1",
            ),
            ("dog_white_model_v1", "species_classifier_default_v1"),
        )
        self.assertEqual(
            pipeline.uploaded_subject_profile_selection(
                "cat",
                "dog_white_model_v1",
            ),
            ("sleek_tabby_cat_v1", "species_classifier_default_v1"),
        )

    def test_uploaded_dog_routes_to_standing_profile_and_legacy_alias(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary) / "outputs"
            jobs: dict[str, dict[str, object]] = {}
            job_queue: queue.Queue[str] = queue.Queue(maxsize=4)
            image_bytes = io.BytesIO()
            Image.new("RGB", (512, 512), (130, 90, 60)).save(
                image_bytes,
                format="PNG",
            )
            upload = pipeline.UploadFile(
                filename="dog.png",
                file=io.BytesIO(image_bytes.getvalue()),
            )
            classification = {
                "species": "dog",
                "scores": {"cat": 0.02, "dog": 0.91},
                "margin": 0.89,
                "forced_binary": True,
                "unknown_class": False,
                "fallback": False,
            }
            with (
                patch.multiple(
                    pipeline,
                    OUTPUT_ROOT=output_root,
                    JOBS=jobs,
                    JOB_QUEUE=job_queue,
                ),
                patch.object(
                    pipeline,
                    "classify_pet_species",
                    return_value=classification,
                ),
                patch.object(pipeline, "require_subject_replacement_assets"),
            ):
                result = asyncio.run(
                    pipeline.create_subject_replacement_api_job(
                        image=upload,
                        prompt="",
                        actions_raw='["eating", "grooming"]',
                        subject_profile_id="sleek_tabby_cat_v1",
                        output_format="transparent_gif",
                        scail_steps=6,
                        pose_strength=None,
                        seed=123,
                    )
                )

        self.assertEqual(result["species"], "dog")
        self.assertEqual(result["species_classification"], classification)
        self.assertEqual(result["subject_profile"], "dog_white_model_v1")
        self.assertEqual(result["requested_subject_profile"], "sleek_tabby_cat_v1")
        self.assertTrue(result["subject_profile_routed_by_species"])
        self.assertEqual(
            result["subject_profile_selection_policy"],
            "species_classifier_default_v1",
        )
        self.assertEqual(result["actions"], ["eating", "standing"])
        self.assertTrue(result["output_timing_is_action_specific"])
        self.assertEqual(
            result["output_timing_by_action"]["eating"]["transparent_gif"],
            {"frames": 100, "fps": 10, "duration_seconds": 10.0},
        )
        self.assertEqual(
            result["output_timing_by_action"]["standing"]["transparent_gif"],
            {"frames": 100, "fps": 20, "duration_seconds": 5.0},
        )
        self.assertEqual(
            result["reference_image_profile"],
            pipeline.DOG_SUBJECT_REPLACEMENT_REFERENCE_IMAGE_PROFILE,
        )
        self.assertEqual(
            set(result["effective_subject_config"]["actions"]),
            {"eating", "standing"},
        )

    def test_default_profile_job_needs_no_uploaded_image(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_root = root / "outputs"
            jobs: dict[str, dict[str, object]] = {}
            job_queue: queue.Queue[str] = queue.Queue(maxsize=4)
            profile = _subject_profile_v2(root)
            with (
                patch.multiple(
                    pipeline,
                    OUTPUT_ROOT=output_root,
                    JOBS=jobs,
                    JOB_QUEUE=job_queue,
                ),
                patch.object(pipeline, "load_subject_profile", return_value=profile),
                patch.object(pipeline, "require_subject_replacement_assets"),
            ):
                result = asyncio.run(
                    pipeline.create_subject_replacement_api_job(
                        image=None,
                        prompt="",
                        actions_raw='["eating", "grooming"]',
                        subject_profile_id="sleek_tabby_cat_v1",
                        output_format="transparent_gif",
                        scail_steps=6,
                        pose_strength=None,
                        seed=123,
                    )
                )
            standard_actions = result["effective_subject_config"]["standard_direct"][
                "actions"
            ]
            expected_references = {
                action: standard_actions[action]["selected_reference"]["cutout"]
                for action in ("eating", "grooming")
            }
            self.assertEqual(result["subject_reference_paths"], expected_references)
            self.assertTrue(
                all(Path(path).is_file() for path in expected_references.values())
            )
            self.assertEqual(result["standard_direct_actions"], standard_actions)

        self.assertEqual(result["generation_mode"], "subject_replacement")
        self.assertEqual(result["actions"], ["eating", "grooming"])
        self.assertEqual(result["subject_source"], "profile")
        self.assertEqual(result["requested_subject_source"], "profile")
        self.assertEqual(result["effective_subject_source"], "profile")
        self.assertEqual(result["subject_profile"], "sleek_tabby_cat_v1")
        self.assertEqual(result["frames"], 100)
        self.assertEqual(result["fps"], 20.0)
        self.assertFalse(result["output_timing_is_action_specific"])
        self.assertEqual(
            result["output_timing_by_action"]["eating"]["transparent_gif"],
            {"frames": 100, "fps": 20, "duration_seconds": 5.0},
        )
        self.assertEqual(result["seed"], 123)
        self.assertEqual(result["seed_mode"], "base_plus_action_offset")
        self.assertEqual(
            result["effective_action_seeds"], {"eating": 123, "grooming": 125}
        )
        self.assertEqual(result["source_frames"], 81)
        self.assertEqual(result["source_fps"], 16.0)
        self.assertEqual(
            result["reference_pipeline"],
            pipeline.PROFILE_STANDARD_DIRECT_POLICY,
        )
        self.assertIsNone(result["reference_pipeline_version"])
        self.assertEqual(
            result["reference_generation_status"],
            "standard_direct_ready",
        )
        self.assertEqual(
            result["generation_engine"],
            "deterministic-main-case-standard-direct",
        )
        self.assertEqual(
            result["video_engine"],
            "deterministic-main-case-standard-direct",
        )
        self.assertEqual(
            result["visual_style"],
            "main_case_profile_standard_direct",
        )
        self.assertEqual(
            result["standard_direct_policy"],
            pipeline.PROFILE_STANDARD_DIRECT_POLICY,
        )
        self.assertEqual(
            result["standard_direct_role"],
            pipeline.PROFILE_STANDARD_DIRECT_ROLE,
        )
        self.assertEqual(
            result["standard_direct_revision"],
            "test-main-case-standard-v1",
        )
        self.assertFalse(result["gpu_required"])
        self.assertFalse(result["gpu_submitted"])
        self.assertEqual(
            result["gpu_bypass_reason"],
            pipeline.PROFILE_STANDARD_DIRECT_GPU_REASON,
        )
        self.assertFalse(result["generation_controls_applied"])
        self.assertIsNone(result["scail_model"])
        self.assertIsNone(result["scail_dpo_lora"])
        self.assertIsNone(result["scail_distill_lora"])
        self.assertIsNone(result["scail_steps"])
        self.assertIsNone(result["reference_image_profile"])
        self.assertIsNone(result["reference_generation_engine"])
        self.assertIsNone(result["reference_normalization_contract"])
        self.assertFalse(result["qwen_reference_generation_for_subject_replacement"])
        self.assertFalse(result["canonical_reference_generated"])
        self.assertFalse(result["canonical_reference_used"])
        self.assertEqual(set(result["subject_reference_paths"]), {"eating", "grooming"})
        self.assertEqual(
            result["subject_profile_sha256"],
            result["effective_subject_config"]["profile_sha256"],
        )

    def test_uploaded_photo_is_never_registered_as_a_scail_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_root = root / "outputs"
            jobs: dict[str, dict[str, object]] = {}
            job_queue: queue.Queue[str] = queue.Queue(maxsize=4)
            profile = _subject_profile_v2(root)

            def normalize_upload(
                _raw: bytes,
                target: Path,
                *,
                identity_source_path: Path,
            ) -> None:
                target.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (32, 32), "white").save(target)
                Image.new("RGB", (32, 32), "white").save(identity_source_path)

            upload = pipeline.UploadFile(
                filename="ordinary-photo.jpg",
                file=io.BytesIO(b"ordinary-photo"),
            )
            with (
                patch.multiple(
                    pipeline,
                    OUTPUT_ROOT=output_root,
                    JOBS=jobs,
                    JOB_QUEUE=job_queue,
                ),
                patch.object(pipeline, "load_subject_profile", return_value=profile),
                patch.object(pipeline, "require_subject_replacement_assets"),
                patch.object(
                    pipeline, "normalize_uploaded_image", side_effect=normalize_upload
                ),
            ):
                result = asyncio.run(
                    pipeline.create_subject_replacement_api_job(
                        image=upload,
                        prompt="",
                        actions_raw='["eating", "grooming"]',
                        subject_profile_id="sleek_tabby_cat_v1",
                        output_format="transparent_gif",
                        scail_steps=6,
                        pose_strength=None,
                        seed=123,
                    )
                )
            profile_snapshots = {
                action: Path(
                    result["effective_subject_config"]["actions"][action]["reference"]
                )
                for action in ("eating", "grooming")
            }
            self.assertTrue(
                all(snapshot.is_file() for snapshot in profile_snapshots.values())
            )
            self.assertTrue(
                all(
                    snapshot.parent.name == "profile_inputs"
                    for snapshot in profile_snapshots.values()
                )
            )
            self.assertNotEqual(
                profile_snapshots["eating"],
                Path(profile["action_references"]["eating"]),
            )
            self.assertNotEqual(
                profile_snapshots["grooming"],
                Path(profile["action_references"]["grooming"]),
            )

        self.assertEqual(result["subject_source"], "upload_pending_qwen")
        self.assertEqual(result["requested_subject_source"], "upload")
        self.assertEqual(
            result["effective_subject_source"],
            "upload_pending_qwen",
        )
        self.assertEqual(result["subject_reference_paths"], {})
        self.assertEqual(
            result["reference_pipeline"],
            pipeline.QWEN_FULL_BODY_FIRST_FRAME_PIPELINE,
        )
        self.assertEqual(
            result["reference_pipeline_version"],
            pipeline.QWEN_FULL_BODY_FIRST_FRAME_PIPELINE_VERSION,
        )
        self.assertFalse(
            result["reference_normalization_contract"]["raw_upload_sent_to_scail"]
        )
        self.assertEqual(result["reference_image_profile"], "pet-multipose-20260717-v2")

    def test_three_view_upload_is_persisted_with_role_and_digest_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_root = root / "outputs"
            jobs: dict[str, dict[str, object]] = {}
            job_queue: queue.Queue[str] = queue.Queue(maxsize=4)
            profile = _subject_profile_v2(root)
            colors = {
                b"front": (64, 64, 64),
                b"left": (96, 96, 96),
                b"right": (128, 128, 128),
            }

            def normalize_upload(
                raw: bytes,
                target: Path,
                *,
                identity_source_path: Path,
            ) -> None:
                target.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (32, 32), colors[raw]).save(target)
                Image.new("RGBA", (40, 30), (*colors[raw], 255)).save(
                    identity_source_path
                )

            upload_files = {
                role: io.BytesIO(role.encode("ascii"))
                for role in ("front", "left", "right")
            }
            uploads = {
                role: pipeline.UploadFile(
                    filename=f"pet-{role}.jpg",
                    file=upload_files[role],
                )
                for role in upload_files
            }
            with (
                patch.multiple(
                    pipeline,
                    OUTPUT_ROOT=output_root,
                    JOBS=jobs,
                    JOB_QUEUE=job_queue,
                ),
                patch.object(pipeline, "load_subject_profile", return_value=profile),
                patch.object(pipeline, "require_subject_replacement_assets"),
                patch.object(
                    pipeline,
                    "normalize_uploaded_image",
                    side_effect=normalize_upload,
                ),
            ):
                result = asyncio.run(
                    pipeline.create_subject_replacement_api_job(
                        image=uploads["front"],
                        image_left=uploads["left"],
                        image_right=uploads["right"],
                        prompt="",
                        actions_raw='["grooming"]',
                        subject_profile_id="sleek_tabby_cat_v1",
                        output_format="transparent_gif",
                        scail_steps=6,
                        pose_strength=None,
                        seed=123,
                    )
                )
                job_dir = output_root / result["job_id"]
                verified = pipeline.verify_uploaded_identity_submission(
                    result,
                    job_dir,
                )

            self.assertEqual(result["uploaded_identity_view_mode"], "front_left_right")
            self.assertEqual(result["uploaded_identity_view_count"], 3)
            self.assertEqual(
                result["uploaded_identity_view_roles"],
                ["front", "left", "right"],
            )
            self.assertEqual(
                result["uploaded_identity_view_policy"],
                pipeline.UPLOADED_IDENTITY_VIEW_POLICY,
            )
            self.assertEqual(set(verified["views"]), {"front", "left", "right"})
            self.assertEqual(
                result["uploaded_identity_input_sha256"],
                result["uploaded_identity_views"]["front"]["input_sha256"],
            )
            self.assertTrue(
                result["reference_normalization_contract"][
                    "identity_views_used_only_for_qwen_canonical"
                ]
            )
            self.assertFalse(
                result["reference_normalization_contract"][
                    "action_stage_raw_identity_views_used"
                ]
            )
            self.assertTrue(all(stream.closed for stream in upload_files.values()))

    def test_partial_three_view_upload_is_rejected_and_all_files_are_closed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = _subject_profile_v2(root)
            front_file = io.BytesIO(b"front")
            left_file = io.BytesIO(b"left")
            front = pipeline.UploadFile(filename="front.jpg", file=front_file)
            left = pipeline.UploadFile(filename="left.jpg", file=left_file)
            with patch.object(
                pipeline,
                "load_subject_profile",
                return_value=profile,
            ):
                with self.assertRaises(pipeline.HTTPException) as raised:
                    asyncio.run(
                        pipeline.create_subject_replacement_api_job(
                            image=front,
                            image_left=left,
                            prompt="",
                            actions_raw='["grooming"]',
                            subject_profile_id="sleek_tabby_cat_v1",
                            output_format="transparent_gif",
                            scail_steps=6,
                            pose_strength=None,
                            seed=123,
                        )
                    )
            self.assertEqual(raised.exception.status_code, 422)
            self.assertIn("complete image", raised.exception.detail)
            self.assertTrue(front_file.closed)
            self.assertTrue(left_file.closed)

    def test_three_view_integrity_rejects_changed_side_source(self) -> None:
        def png_bytes(color: tuple[int, int, int]) -> bytes:
            buffer = io.BytesIO()
            Image.new("RGB", (32, 24), color).save(buffer, format="PNG")
            return buffer.getvalue()

        with tempfile.TemporaryDirectory() as temporary:
            job_dir = Path(temporary)
            records = pipeline.persist_uploaded_identity_views(
                job_dir,
                {
                    "front": png_bytes((40, 40, 40)),
                    "left": png_bytes((80, 80, 80)),
                    "right": png_bytes((120, 120, 120)),
                },
            )
            job = {
                "uploaded_identity_input_sha256": records["front"]["input_sha256"],
                "uploaded_identity_source_sha256": records["front"]["source_sha256"],
                "uploaded_identity_view_policy": pipeline.UPLOADED_IDENTITY_VIEW_POLICY,
                "uploaded_identity_view_mode": "front_left_right",
                "uploaded_identity_view_count": 3,
                "uploaded_identity_view_roles": ["front", "left", "right"],
                "uploaded_identity_views": records,
            }
            (job_dir / "input_left_identity_source.png").write_bytes(b"changed")
            with self.assertRaisesRegex(
                RuntimeError,
                "identity left source is missing or changed",
            ):
                pipeline.verify_uploaded_identity_submission(job, job_dir)

    def test_distinct_three_view_set_passes_real_palette_and_angle_audit(self) -> None:
        def view_bytes(role: str) -> bytes:
            image = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
            draw = ImageDraw.Draw(image)
            if role == "front":
                draw.ellipse((35, 10, 93, 118), fill=(82, 82, 82, 255))
                draw.ellipse((42, 8, 86, 50), fill=(205, 205, 205, 255))
            elif role == "left":
                draw.ellipse((8, 40, 116, 102), fill=(88, 88, 88, 255))
                draw.ellipse((82, 24, 122, 66), fill=(200, 200, 200, 255))
            else:
                draw.polygon(
                    [(14, 105), (35, 30), (92, 12), (118, 76), (100, 116)],
                    fill=(92, 92, 92, 255),
                )
                draw.ellipse((18, 18, 58, 58), fill=(198, 198, 198, 255))
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            return buffer.getvalue()

        with tempfile.TemporaryDirectory() as temporary:
            job_dir = Path(temporary)
            records = pipeline.persist_uploaded_identity_views(
                job_dir,
                {role: view_bytes(role) for role in ("front", "left", "right")},
            )
            job = {
                "uploaded_identity_input_sha256": records["front"]["input_sha256"],
                "uploaded_identity_source_sha256": records["front"]["source_sha256"],
                "uploaded_identity_view_policy": pipeline.UPLOADED_IDENTITY_VIEW_POLICY,
                "uploaded_identity_view_mode": "front_left_right",
                "uploaded_identity_view_count": 3,
                "uploaded_identity_view_roles": ["front", "left", "right"],
                "uploaded_identity_views": records,
            }
            integrity = pipeline.verify_uploaded_identity_submission(job, job_dir)
            audit = pipeline.audit_uploaded_identity_view_set(integrity)

        self.assertTrue(audit["acceptable"])
        self.assertEqual(audit["issues"], [])
        self.assertEqual(set(audit["palettes"]), {"front", "left", "right"})
        self.assertTrue(all(item["available"] for item in audit["palettes"].values()))
        self.assertTrue(
            all(
                not comparison["near_duplicate"]
                for comparison in audit["view_angle_comparisons"].values()
            )
        )




class PoseAttemptFailureReportingTests(unittest.TestCase):
    def test_unevaluated_surface_placeholder_is_not_reported_as_a_failure(self) -> None:
        attempts = [
            {
                "attempt": attempt,
                "grooming_pose_gate": {
                    "issues": ["raised_paw_color_inconsistent_with_planted_paw"]
                },
                "fur_surface_gate": {
                    "evaluated": False,
                    "acceptable": False,
                    "issues": ["fur_surface_gate_not_evaluated_upstream_gate_failed"],
                },
            }
            for attempt in range(1, 4)
        ]
        summary = pipeline.pose_attempt_failure_summary(attempts)
        self.assertEqual(
            summary,
            "persistent: raised_paw_color_inconsistent_with_planted_paw",
        )
        self.assertNotIn("fur_surface", summary)
        self.assertEqual(
            pipeline.pose_attempt_failure_issues(attempts),
            ["raised_paw_color_inconsistent_with_planted_paw"],
        )

    def test_evaluated_surface_rejection_remains_in_failure_summary(self) -> None:
        attempts = [
            {
                "attempt": 1,
                "fur_surface_gate": {
                    "evaluated": True,
                    "acceptable": False,
                    "issues": ["coherent_long_fur_energy"],
                },
            }
        ]
        summary = pipeline.pose_attempt_failure_summary(attempts)
        self.assertEqual(summary, "persistent: coherent_long_fur_energy")


class AdvisoryGenerationAuditTests(unittest.TestCase):
    def test_default_policy_keeps_only_runtime_integrity_blocking(self) -> None:
        self.assertFalse(pipeline.GENERATION_VISUAL_AUDITS_BLOCKING)
        self.assertEqual(
            pipeline.GENERATION_AUDIT_POLICY,
            "runtime_integrity_required_visual_content_advisory_v1",
        )
        audit = pipeline.skipped_visual_audit("grooming_pose_detail")
        self.assertTrue(audit["acceptable"])
        self.assertFalse(audit["blocking"])
        self.assertEqual(audit["status"], "skipped_advisory")

    def test_palette_drift_no_longer_rejects_all_canonical_candidates(self) -> None:
        attempts = [
            {
                "attempt": attempt,
                "identity_palette": {
                    "acceptable": False,
                    "issues": ["achromatic_coat_value_balance_lost"],
                    "penalty": float(attempt),
                },
                "quality": {"penalty": float(attempt)},
                "chroma": {"acceptable": True},
            }
            for attempt in range(1, 4)
        ]
        selected = pipeline.select_canonical_fallback_attempt(
            attempts,
            identity_palette_required=True,
        )
        self.assertEqual(selected["attempt"], 1)

    def test_semantic_recovery_ignores_visual_issues_but_not_bad_alpha(self) -> None:
        failed_visual = {
            "acceptable": False,
            "issues": ["visual_issue"],
        }
        self.assertIsNone(
            pipeline.semantic_recovery_gate_failure(
                {"acceptable": True},
                failed_visual,
                failed_visual,
                failed_visual,
                failed_visual,
                failed_visual,
            )
        )
        self.assertEqual(
            pipeline.semantic_recovery_gate_failure(
                {"acceptable": False},
                failed_visual,
                failed_visual,
                failed_visual,
            ),
            "semantic_chroma_invalid",
        )
        self.assertEqual(
            pipeline.semantic_recovery_gate_failure(
                {"acceptable": True},
                {"acceptable": True},
                {"acceptable": True},
                failed_visual,
                identity_palette_blocking=True,
            ),
            "semantic_identity_palette_invalid",
        )

    def test_pre_scail_revalidation_needs_alpha_and_digests_not_surface_scores(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_root = root / "outputs"
            comfy_input = root / "comfy-input"
            job_id = "advisory-pre-scail"
            job_dir = output_root / job_id
            job_dir.mkdir(parents=True)
            canonical = job_dir / "cartoon_reference_transparent.png"
            canonical_image1 = job_dir / "cartoon_reference.png"
            action_reference = job_dir / "eating_transparent.png"
            rgba = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
            rgba.paste((90, 100, 110, 255), (128, 64, 384, 448))
            rgba.save(canonical)
            rgba.save(action_reference)
            green = Image.new("RGB", (512, 512), (0, 255, 0))
            green.paste((90, 100, 110), (128, 64, 384, 448))
            green.save(canonical_image1)
            comfy_canonical = (
                comfy_input
                / "local_video_api"
                / job_id
                / canonical_image1.name
            )
            comfy_canonical.parent.mkdir(parents=True)
            comfy_canonical.write_bytes(canonical_image1.read_bytes())
            canonical_digest = pipeline.file_sha256(canonical)
            image1_digest = pipeline.file_sha256(canonical_image1)
            action_digest = pipeline.file_sha256(action_reference)
            jobs: dict[str, dict[str, object]] = {
                job_id: {
                    "job_id": job_id,
                    "subject_source": "qwen_generated_upload",
                    "reference_generation_completed": True,
                    "reference_generation_status": "completed",
                    "reference_normalization_status": "completed",
                    "canonical_reference_generated": True,
                    "canonical_reference_used": True,
                    "canonical_reference_normalized": True,
                    "canonical_reference_transparent": True,
                    "qwen_action_keyframes_normalized": True,
                    "qwen_action_keyframes_transparent": True,
                    "qwen_action_reference_normalization_policy": (
                        pipeline.QWEN_ACTION_REFERENCE_NORMALIZATION_POLICY
                    ),
                    "qwen_action_reference_byte_preserved": True,
                    "surface_texture_gate_status": "skipped_advisory",
                    "canonical_reference_path": str(canonical),
                    "canonical_reference_sha256": canonical_digest,
                    "canonical_public_reference_path": str(canonical),
                    "canonical_public_reference_sha256": canonical_digest,
                    "canonical_qwen_image1_path": str(canonical_image1),
                    "canonical_qwen_image1_comfy_path": str(comfy_canonical),
                    "canonical_qwen_image1_sha256": image1_digest,
                    "canonical_qwen_image1_comfy_sha256": image1_digest,
                    "subject_reference_paths": {"eating": str(action_reference)},
                    "qwen_action_keyframe_sha256s": {"eating": action_digest},
                    "qwen_reference_generated_actions": ["eating"],
                }
            }
            effective_config = {
                "actions": {
                    "eating": {
                        "reference_kind": "qwen_generated_upload",
                        "reference": str(action_reference),
                        "reference_sha256": action_digest,
                    }
                }
            }
            with (
                patch.multiple(
                    pipeline,
                    OUTPUT_ROOT=output_root,
                    COMFY_INPUT=comfy_input,
                    JOBS=jobs,
                    GENERATION_VISUAL_AUDITS_BLOCKING=False,
                ),
                patch.object(pipeline, "verify_uploaded_identity_submission"),
            ):
                pipeline.revalidate_uploaded_qwen_references_before_scail(
                    job_id,
                    jobs[job_id],
                    effective_config,
                    ["eating"],
                )

            self.assertTrue(jobs[job_id]["qwen_reference_pre_scail_revalidated"])
            self.assertEqual(
                jobs[job_id]["qwen_reference_pre_scail_visual_audit_status"],
                "skipped_advisory",
            )
            self.assertEqual(jobs[job_id]["subject_fur_surface_quality"], {})

    def test_advisory_grooming_pre_scail_requires_minimum_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_root = root / "outputs"
            comfy_input = root / "comfy-input"
            job_id = "advisory-grooming-pre-scail"
            job_dir = output_root / job_id
            job_dir.mkdir(parents=True)
            canonical = job_dir / "cartoon_reference_transparent.png"
            canonical_image1 = job_dir / "cartoon_reference.png"
            uploaded_identity = job_dir / "input_identity_source.png"
            action_reference = job_dir / "grooming_transparent.png"
            rgba = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
            rgba.paste((220, 220, 220, 255), (128, 64, 384, 448))
            rgba.save(canonical)
            rgba.save(action_reference)
            green = Image.new("RGB", (512, 512), (0, 255, 0))
            green.paste((220, 220, 220), (128, 64, 384, 448))
            green.save(canonical_image1)
            # Regression fixture for task 20260822T054539Z-98a9c633ea: the
            # uploaded identity is intentionally not 512x512.  It is the coat
            # palette authority, never the canonical action-edit canvas.
            uploaded_rgba = Image.new("RGBA", (514, 584), (0, 0, 0, 0))
            uploaded_rgba.paste((220, 220, 220, 255), (80, 48, 434, 552))
            uploaded_rgba.save(uploaded_identity)
            comfy_canonical = (
                comfy_input
                / "local_video_api"
                / job_id
                / canonical_image1.name
            )
            comfy_canonical.parent.mkdir(parents=True)
            comfy_canonical.write_bytes(canonical_image1.read_bytes())
            canonical_digest = pipeline.file_sha256(canonical)
            image1_digest = pipeline.file_sha256(canonical_image1)
            uploaded_identity_digest = pipeline.file_sha256(uploaded_identity)
            action_digest = pipeline.file_sha256(action_reference)
            selected_minimum = {
                "policy": pipeline.GROOMING_ADVISORY_MINIMUM_GATE_POLICY,
                "version": pipeline.GROOMING_ADVISORY_MINIMUM_GATE_VERSION,
                "acceptable": True,
                "candidate_sha256": action_digest,
                "selected_candidate_path": str(action_reference),
                "selected_candidate_sha256": action_digest,
                "identity_reference_path": str(canonical_image1),
                "identity_reference_sha256": image1_digest,
                "identity_palette_reference_path": str(uploaded_identity),
                "identity_palette_reference_sha256": uploaded_identity_digest,
            }
            jobs: dict[str, dict[str, object]] = {
                job_id: {
                    "job_id": job_id,
                    "subject_source": "qwen_generated_upload",
                    "reference_generation_completed": True,
                    "reference_generation_status": "completed",
                    "reference_normalization_status": "completed",
                    "canonical_reference_generated": True,
                    "canonical_reference_used": True,
                    "canonical_reference_normalized": True,
                    "canonical_reference_transparent": True,
                    "qwen_action_keyframes_normalized": True,
                    "qwen_action_keyframes_transparent": True,
                    "qwen_action_reference_normalization_policy": (
                        pipeline.QWEN_ACTION_REFERENCE_NORMALIZATION_POLICY
                    ),
                    "qwen_action_reference_byte_preserved": True,
                    "surface_texture_gate_status": "skipped_advisory",
                    "canonical_reference_path": str(canonical),
                    "canonical_reference_sha256": canonical_digest,
                    "canonical_public_reference_path": str(canonical),
                    "canonical_public_reference_sha256": canonical_digest,
                    "canonical_qwen_image1_path": str(canonical_image1),
                    "canonical_qwen_image1_comfy_path": str(comfy_canonical),
                    "canonical_qwen_image1_sha256": image1_digest,
                    "canonical_qwen_image1_comfy_sha256": image1_digest,
                    "uploaded_identity_anchor_path": str(uploaded_identity),
                    "uploaded_identity_source_sha256": uploaded_identity_digest,
                    "subject_reference_paths": {
                        "grooming": str(action_reference)
                    },
                    "qwen_action_keyframe_sha256s": {
                        "grooming": action_digest
                    },
                    "qwen_reference_generated_actions": ["grooming"],
                    "grooming_advisory_minimum_gate_status": "passed",
                    "grooming_advisory_minimum_gate_selected": {
                        "grooming": selected_minimum
                    },
                }
            }
            effective_config = {
                "actions": {
                    "grooming": {
                        "reference_kind": "qwen_generated_upload",
                        "reference": str(action_reference),
                        "reference_sha256": action_digest,
                    }
                }
            }
            with (
                patch.multiple(
                    pipeline,
                    OUTPUT_ROOT=output_root,
                    COMFY_INPUT=comfy_input,
                    JOBS=jobs,
                    GENERATION_VISUAL_AUDITS_BLOCKING=False,
                ),
                patch.object(pipeline, "verify_uploaded_identity_submission"),
            ):
                pipeline.revalidate_uploaded_qwen_references_before_scail(
                    job_id,
                    jobs[job_id],
                    effective_config,
                    ["grooming"],
                )
                self.assertEqual(
                    jobs[job_id][
                        "qwen_reference_pre_scail_minimum_contract_status"
                    ],
                    "passed",
                )
                selected_minimum["identity_palette_reference_sha256"] = "0" * 64
                with self.assertRaisesRegex(
                    RuntimeError,
                    "minimum action/identity contract",
                ):
                    pipeline.revalidate_uploaded_qwen_references_before_scail(
                        job_id,
                        jobs[job_id],
                        effective_config,
                        ["grooming"],
                    )
                selected_minimum["identity_palette_reference_sha256"] = (
                    uploaded_identity_digest
                )
                selected_minimum["acceptable"] = False
                selected_minimum["degraded_fallback_accepted"] = True
                jobs[job_id][
                    "grooming_advisory_minimum_gate_status"
                ] = "degraded_passed"
                pipeline.revalidate_uploaded_qwen_references_before_scail(
                    job_id,
                    jobs[job_id],
                    effective_config,
                    ["grooming"],
                )
                self.assertEqual(
                    jobs[job_id][
                        "qwen_reference_pre_scail_minimum_contract_status"
                    ],
                    "degraded_passed",
                )
                selected_minimum["selected_candidate_sha256"] = "0" * 64
                with self.assertRaisesRegex(
                    RuntimeError,
                    "minimum action/identity contract",
                ):
                    pipeline.revalidate_uploaded_qwen_references_before_scail(
                        job_id,
                        jobs[job_id],
                        effective_config,
                        ["grooming"],
                    )


if __name__ == "__main__":
    unittest.main()
