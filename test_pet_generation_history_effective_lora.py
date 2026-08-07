from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pet_generation_platform as platform


class HistoryEffectiveLoraTests(unittest.TestCase):
    def test_old_job_without_effective_fields_remains_compatible(self) -> None:
        self.assertIsNone(platform.history_effective_lora({"job_id": "legacy"}))

    def test_selected_canonical_and_per_pose_settings_are_projected(self) -> None:
        job = {
            "cartoon_reference_profile": "safe_reduced_user_lora",
            "effective_canonical_user_loras": [
                {
                    "model_name": "user_uploads/style-a.safetensors",
                    "original_filename": "style-a.safetensors",
                    "strength": 0.4,
                }
            ],
            "canonical_user_lora_strength_scale": 0.5,
            "effective_canonical_settings": {
                "user_lora_mode": "stack",
                "steps": 6,
                "cfg": 1.0,
                "denoise": 1.0,
            },
            "effective_pose_lora_attempts": [
                {
                    "pose_index": 1,
                    "pose": "running",
                    "attempt": 2,
                    "profile": "pose_reduced_retry",
                    "user_loras": [
                        {
                            "model_name": "user_uploads/style-a.safetensors",
                            "original_filename": "style-a.safetensors",
                            "strength": 0.2,
                        }
                    ],
                    "user_lora_strength_scale": 0.25,
                    "user_lora_mode": "stack",
                    "steps": 8,
                    "cfg": 1.0,
                    "denoise": 1.0,
                    "quality_retry": 2,
                    "total_strength_cap": 0.25,
                    "per_lora_strength_cap": 0.25,
                    "similarity": {
                        "available": True,
                        "warning": False,
                        "issues": [],
                        "comparisons": [{"label": "canonical"}],
                    },
                }
            ],
        }

        effective = platform.history_effective_lora(job)

        self.assertIsNotNone(effective)
        assert effective is not None
        self.assertEqual(effective["canonical"]["mode"], "stack")
        self.assertEqual(effective["canonical"]["steps"], 6)
        self.assertEqual(effective["canonical"]["loras"][0]["name"], "style-a.safetensors")
        pose = effective["poses"][0]
        self.assertEqual(pose["profile"], "pose_reduced_retry")
        self.assertEqual(pose["mode"], "stack")
        self.assertEqual(pose["steps"], 8)
        self.assertEqual(pose["quality_retry"], 2)
        self.assertAlmostEqual(pose["strength_scale"], 0.25)
        self.assertFalse(pose["similarity"]["warning"])

    def test_history_parameters_keep_v2_pose_and_palette_audit_fields(self) -> None:
        source_palette = {
            "available": True,
            "dominant": ["black", "white"],
            "warm_fraction": 0.0,
        }
        canonical_palette = {
            "available": True,
            "acceptable": True,
            "issues": [],
        }
        parameters = platform.history_parameters(
            {
                "seed": 20260715,
                "poses": ["legacy effective pose"],
                "poses_original": ["猫咪趴卧"],
                "poses_en": ["lying down | first frame: The cat lies prone."],
                "image_profile": "pet-multipose-20260717-v2",
                "image_profile_revision": "multipose-v2",
                "pose_prompt_source": "verified_profile_presets",
                "pose_preset_ids": ["lying_down"],
                "pose_seed_offsets": {"lying down": 401},
                "identity_palette_guard": True,
                "identity_palette_policy": "source_palette_guard_v1",
                "source_identity_palette": source_palette,
                "effective_canonical_identity_palette": canonical_palette,
                "pose_identity_palette_warning_count": 0,
            },
            {"parameters": {}},
        )

        self.assertEqual(parameters["poses"], ["猫咪趴卧"])
        self.assertEqual(
            parameters["poses_en"],
            ["lying down | first frame: The cat lies prone."],
        )
        self.assertEqual(
            parameters["image_profile"],
            "pet-multipose-20260717-v2",
        )
        self.assertEqual(parameters["image_profile_revision"], "multipose-v2")
        self.assertEqual(
            parameters["pose_prompt_source"],
            "verified_profile_presets",
        )
        self.assertEqual(parameters["pose_preset_ids"], ["lying_down"])
        self.assertEqual(parameters["pose_seed_offsets"], {"lying down": 401})
        self.assertTrue(parameters["identity_palette_guard"])
        self.assertEqual(
            parameters["identity_palette_policy"],
            "source_palette_guard_v1",
        )
        self.assertEqual(parameters["source_identity_palette"], source_palette)
        self.assertEqual(
            parameters["effective_canonical_identity_palette"],
            canonical_palette,
        )
        self.assertEqual(parameters["pose_identity_palette_warning_count"], 0)

    def test_failed_job_without_media_is_retained(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            job_dir = Path(temporary_directory) / "failed-job"
            job_dir.mkdir()
            item = platform.history_item(
                {
                    "job_id": "failed-job",
                    "status": "failed",
                    "stage": "failed",
                    "request_kind": "video",
                    "error": "generation failed before producing media",
                    "created_at": "2026-07-28T00:00:00+00:00",
                },
                job_dir,
                "pipeline",
                "2026-07-28T00:00:00+00:00",
            )

        self.assertIsNotNone(item)
        assert item is not None
        self.assertEqual(item["status"], "failed")
        self.assertEqual(item["error"], "generation failed before producing media")
        self.assertEqual(item["media"], [])
        self.assertEqual(item["image_count"], 0)
        self.assertEqual(item["video_count"], 0)

    def test_history_formats_uploaded_subject_reference_states(self) -> None:
        html = (
            Path(__file__)
            .with_name("pet_generation_history.html")
            .read_text(encoding="utf-8")
        )

        self.assertIn("value === 'upload_pending_qwen'", html)
        self.assertIn("本次上传（正在生成 Qwen 参考）", html)
        self.assertIn("value === 'qwen_generated_upload'", html)
        self.assertIn("本次上传（Qwen 参考已生成）", html)


if __name__ == "__main__":
    unittest.main()
