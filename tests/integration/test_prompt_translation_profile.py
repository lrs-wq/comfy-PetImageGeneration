from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

import local_video_api as pipeline

PLATFORM_PAGE = Path(__file__).resolve().parents[2] / "web" / "platform" / "index.html"


class PromptTranslationTests(unittest.TestCase):
    @staticmethod
    def prepare_translator_files(root: Path) -> tuple[Path, Path, Path]:
        python_path = root / "python"
        script_path = root / "translator.py"
        model_path = root / "model"
        model_path.mkdir()
        python_path.touch()
        script_path.touch()
        for filename in (
            "config.json",
            "source.spm",
            "target.spm",
            "vocab.json",
            "pytorch_model.bin",
            "translation_ready.json",
        ):
            (model_path / filename).touch()
        return python_path, script_path, model_path

    def test_han_detection_does_not_misclassify_english_or_hex_color(self) -> None:
        self.assertTrue(pipeline.contains_han("保留猫咪特征"))
        self.assertTrue(pipeline.contains_han("cute 3D 猫"))
        self.assertFalse(pipeline.contains_han("pure #00FF00 green background"))

    def test_reference_upload_wording_is_normalized_only_for_translation(self) -> None:
        source = "保留上传猫咪特征，并参考上传的图片中原有毛色"

        self.assertEqual(
            pipeline.normalize_chinese_prompt_for_translation(source),
            "保留参考图片中的猫咪特征，并参考图片中原有毛色",
        )

    def test_verified_baseline_translation_and_pose_normalization(self) -> None:
        prompt_en, poses_en, metadata = pipeline.translate_prompt_bundle(
            pipeline.BASELINE_IMAGE_PROMPT_ZH,
            pipeline.DEFAULT_POSES,
        )

        self.assertEqual(prompt_en, pipeline.BASELINE_IMAGE_PROMPT_EN)
        self.assertEqual(
            poses_en,
            [
                f"{preset['action_key']} | first frame: {preset['prompt_en']}"
                for preset in pipeline.MULTIPOSE_PRESETS
            ],
        )
        self.assertTrue(metadata["applied"])
        self.assertEqual(metadata["provider"], "verified_profile_translation")
        self.assertEqual(metadata["pose_applied_count"], 4)

    def test_verified_pose_presets_do_not_start_local_translator(self) -> None:
        with patch.object(pipeline.subprocess, "run") as run:
            translated, metadata = pipeline.translate_texts_to_english(
                pipeline.DEFAULT_POSES
            )

        self.assertEqual(
            translated,
            [preset["prompt_en"] for preset in pipeline.MULTIPOSE_PRESETS],
        )
        self.assertFalse(run.called)
        self.assertEqual(metadata["provider"], "verified_profile_translation")

    def test_pose_presets_have_distinct_action_semantics(self) -> None:
        prompts = {
            preset["id"]: preset["prompt_en"].lower()
            for preset in pipeline.MULTIPOSE_PRESETS
        }

        self.assertNotIn("bowl", prompts["running"])
        self.assertIn("food bowl", prompts["eating"])
        self.assertIn("kibble", prompts["eating"])
        self.assertIn("both complete front paws are visibly planted", prompts["eating"])
        self.assertIn("never grasps, lifts, carries, or holds food", prompts["eating"])
        self.assertIn("water bowl", prompts["drinking"])
        self.assertIn("water surface", prompts["drinking"])
        self.assertNotIn("bowl", prompts["lying_down"])
        self.assertIn("prone", prompts["lying_down"])

    def test_generated_prompt_layers_do_not_leak_eating_into_other_actions(
        self,
    ) -> None:
        canonical = pipeline.build_cartoon_reference_prompt(
            pipeline.BASELINE_IMAGE_PROMPT_EN
        ).lower()
        self.assertNotIn("food bowl", canonical)
        self.assertNotIn("brown kibble", canonical)
        self.assertIn("color-faithful neutral high-key lighting", canonical)
        self.assertNotIn("warm high-key lighting", canonical)
        self.assertIn("identity-faithful restrained stylized 3d", canonical)
        self.assertIn("never make it younger or more kitten-like", canonical)
        self.assertNotIn("premium cute", canonical)
        self.assertNotIn("animated-film", canonical)
        self.assertIn("natural coat-length category", canonical)
        self.assertIn("never amplify its apparent length or volume", canonical)
        self.assertIn("source-bounded outer contour", canonical)
        self.assertIn("never shorten, flatten, or suppress genuine long fur", canonical)

        for preset in pipeline.MULTIPOSE_PRESETS:
            pose = f"{preset['action_key']} | first frame: {preset['prompt_en']}"
            keyframe = pipeline.build_qwen_prompt(
                "",
                pose,
                transparent_gif=True,
            ).lower()
            video = pipeline.build_wan_prompt(
                pipeline.BASELINE_IMAGE_PROMPT_EN,
                pose,
                transparent_gif=True,
            ).lower()
            self.assertIn(preset["prompt_en"].lower(), keyframe)
            self.assertIn(preset["video_prompt_en"].lower(), video)
            self.assertIn("color-faithful neutral high-key lighting", video)
            self.assertIn("fur motion lock", video)
            self.assertIn("compact close-lying grooming", video)
            self.assertIn("never increase fur length or volume", video)
            if preset["id"] != "eating":
                self.assertNotIn("brown kibble", keyframe)
                self.assertNotIn("food bowl completely fixed", video)
            else:
                self.assertIn("never grasps, lifts, carries, or holds food", keyframe)
                self.assertIn("must not lift a paw to the mouth", video)
            if preset["id"] in {"running", "lying_down"}:
                self.assertNotIn("bowl", keyframe)
                self.assertNotIn("bowl", video)

    def test_final_pose_retry_keeps_no_prop_actions_bowl_free(self) -> None:
        for preset_id in ("running", "lying_down"):
            preset = next(
                item for item in pipeline.MULTIPOSE_PRESETS if item["id"] == preset_id
            )
            pose = f"{preset['action_key']} | first frame: {preset['prompt_en']}"
            prompt = pipeline.build_qwen_prompt(
                "",
                pose,
                transparent_gif=True,
                quality_retry=3,
            ).lower()

            self.assertNotIn("bowl", prompt)
            self.assertIn("no prop may appear", prompt)

    def test_unknown_chinese_uses_local_translator_once(self) -> None:
        source = "猫咪趴卧在地面上，轻轻抬头"
        translated = "The cat lies prone on the ground and gently raises its head."
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            python_path, script_path, model_path = self.prepare_translator_files(root)
            completed = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=json.dumps({"translations": [translated]}),
                stderr="",
            )
            with (
                patch.multiple(
                    pipeline,
                    PROMPT_TRANSLATOR_PYTHON=python_path,
                    PROMPT_TRANSLATOR_SCRIPT=script_path,
                    PROMPT_TRANSLATOR_MODEL=model_path,
                    PROMPT_TRANSLATION_ENABLED=True,
                ),
                patch.object(pipeline.subprocess, "run", return_value=completed) as run,
            ):
                output, metadata = pipeline.translate_texts_to_english([source, source])

        self.assertEqual(output, [translated, translated])
        self.assertEqual(run.call_count, 1)
        self.assertEqual(metadata["provider"], "local_opus_mt_zh_en")
        self.assertEqual(metadata["semantic_normalization_count"], 0)

    def test_translator_receives_reference_wording_not_upload_command(self) -> None:
        source = "保留上传猫咪特征"
        translated = "Preserve the identity features of the cat in the reference image."
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            python_path, script_path, model_path = self.prepare_translator_files(root)
            completed = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=json.dumps({"translations": [translated]}),
                stderr="",
            )
            with (
                patch.multiple(
                    pipeline,
                    PROMPT_TRANSLATOR_PYTHON=python_path,
                    PROMPT_TRANSLATOR_SCRIPT=script_path,
                    PROMPT_TRANSLATOR_MODEL=model_path,
                    PROMPT_TRANSLATION_ENABLED=True,
                ),
                patch.object(pipeline.subprocess, "run", return_value=completed) as run,
            ):
                output, metadata = pipeline.translate_texts_to_english([source])

        payload = json.loads(run.call_args.kwargs["input"])
        self.assertEqual(payload["texts"], ["保留参考图片中的猫咪特征"])
        self.assertEqual(output, [translated])
        self.assertEqual(metadata["semantic_normalization_count"], 1)

    def test_translator_output_with_han_fails_closed(self) -> None:
        source = "猫咪喝水"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            python_path, script_path, model_path = self.prepare_translator_files(root)
            completed = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=json.dumps({"translations": ["cat 喝水"]}),
                stderr="",
            )
            with (
                patch.multiple(
                    pipeline,
                    PROMPT_TRANSLATOR_PYTHON=python_path,
                    PROMPT_TRANSLATOR_SCRIPT=script_path,
                    PROMPT_TRANSLATOR_MODEL=model_path,
                    PROMPT_TRANSLATION_ENABLED=True,
                ),
                patch.object(pipeline.subprocess, "run", return_value=completed),
            ):
                with self.assertRaises(pipeline.HTTPException) as raised:
                    pipeline.translate_texts_to_english([source])

        self.assertEqual(raised.exception.status_code, 502)

    def test_cached_local_translation_keeps_its_audit_provider(self) -> None:
        source = "缓存测试猫咪提示词"
        translated = "A cached cat prompt."
        with pipeline._PROMPT_TRANSLATION_CACHE_LOCK:
            pipeline._PROMPT_TRANSLATION_CACHE[source] = (
                translated,
                "local_opus_mt_zh_en",
            )
        try:
            output, metadata = pipeline.translate_texts_to_english([source])
        finally:
            with pipeline._PROMPT_TRANSLATION_CACHE_LOCK:
                pipeline._PROMPT_TRANSLATION_CACHE.pop(source, None)

        self.assertEqual(output, [translated])
        self.assertEqual(metadata["provider"], "local_opus_mt_zh_en")


class ImageProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = pipeline.IMAGE_PROFILES[pipeline.BASELINE_IMAGE_PROFILE_ID]
        self.custom = [
            {
                "model_name": "user_uploads/custom.safetensors",
                "original_filename": "custom.safetensors",
                "strength": 0.7,
            }
        ]

    def test_profile_is_available_for_image_and_video_requests(self) -> None:
        image_resolved = pipeline.resolve_image_profile(
            pipeline.BASELINE_IMAGE_PROFILE_ID,
            request_kind="image",
        )
        video_resolved = pipeline.resolve_image_profile(
            pipeline.BASELINE_IMAGE_PROFILE_ID,
            request_kind="video",
        )
        self.assertEqual(
            image_resolved["source_job_id"],
            pipeline.BASELINE_IMAGE_SOURCE_JOB_ID,
        )
        self.assertIs(video_resolved, image_resolved)

    def test_profile_preserves_source_parameters_and_exposes_four_presets(self) -> None:
        self.assertEqual(
            pipeline.BASELINE_IMAGE_PROFILE_ID, "pet-multipose-20260717-v2"
        )
        self.assertEqual(self.profile["revision"], "multipose-v2")
        self.assertEqual(
            [item["strength"] for item in self.profile["user_loras"]],
            [0.5, 0.6],
        )
        self.assertEqual(
            [item["original_filename"] for item in self.profile["user_loras"]],
            [
                "宠物小猫 _ 写实摄影_v3.0.safetensors",
                "Qwen Image-Kitten Photography_QWEN-小猫摄影.safetensors",
            ],
        )
        self.assertEqual(self.profile["qwen_steps"], 6)
        self.assertEqual(self.profile["qwen_cfg"], 1.0)
        self.assertEqual(self.profile["qwen_denoise"], 1.0)
        self.assertEqual(self.profile["seed"], 20260715)
        self.assertEqual(self.profile["canonical_total_cap"], 1.1)
        self.assertEqual(self.profile["canonical_per_lora_cap"], 0.6)
        self.assertEqual(
            [preset["id"] for preset in self.profile["pose_presets"]],
            ["running", "eating", "drinking", "lying_down"],
        )
        self.assertEqual(
            self.profile["pose_seed_offsets"],
            {
                "running": 101,
                "eating": 211,
                "drinking": 307,
                "lying down": 401,
            },
        )
        self.assertEqual(
            len({preset["id"] for preset in self.profile["pose_presets"]}),
            4,
        )
        self.assertTrue(
            all(
                len(preset["prompt_zh"]) <= 300
                for preset in self.profile["pose_presets"]
            )
        )

    def test_legacy_source_profile_remains_resolvable(self) -> None:
        legacy = pipeline.resolve_image_profile(
            pipeline.LEGACY_EATING_IMAGE_PROFILE_ID,
            request_kind="image",
        )

        self.assertEqual(legacy["revision"], "eating-v1")
        self.assertEqual(
            legacy["default_prompt"],
            pipeline.BASELINE_IMAGE_SOURCE_PROMPT_ZH,
        )
        self.assertEqual(legacy["default_poses"], pipeline.LEGACY_DEFAULT_POSES)

    def test_palette_guard_is_enabled_only_for_v2_profile(self) -> None:
        current = pipeline.resolve_image_profile(
            pipeline.BASELINE_IMAGE_PROFILE_ID,
            request_kind="image",
        )
        multipose_v1 = pipeline.resolve_image_profile(
            pipeline.MULTIPOSE_V1_IMAGE_PROFILE_ID,
            request_kind="image",
        )
        legacy = pipeline.resolve_image_profile(
            pipeline.LEGACY_EATING_IMAGE_PROFILE_ID,
            request_kind="image",
        )

        self.assertTrue(current["identity_palette_guard"])
        self.assertEqual(
            current["identity_palette_policy"],
            "source_palette_guard_v2",
        )
        self.assertFalse(multipose_v1["identity_palette_guard"])
        self.assertIsNone(multipose_v1["identity_palette_policy"])
        self.assertFalse(legacy["identity_palette_guard"])
        self.assertIsNone(legacy["identity_palette_policy"])

    def test_profile_defaults_and_explicit_pose_override(self) -> None:
        self.assertEqual(
            pipeline.parse_poses(
                None,
                4,
                defaults=self.profile["default_poses"],
            ),
            pipeline.DEFAULT_POSES,
        )
        self.assertEqual(
            pipeline.parse_poses(
                '["custom pose"]',
                4,
                defaults=self.profile["default_poses"],
            ),
            ["custom pose"],
        )

    def test_public_profile_metadata_contains_versioned_pose_baseline(self) -> None:
        public = pipeline.image_profile_public(self.profile)

        self.assertEqual(public["revision"], "multipose-v2")
        self.assertEqual(public["pose_prompt_source"], "verified_profile_presets")
        self.assertTrue(public["identity_palette_guard"])
        self.assertEqual(
            public["identity_palette_policy"],
            "source_palette_guard_v2",
        )
        self.assertEqual(
            [preset["id"] for preset in public["pose_presets"]],
            ["running", "eating", "drinking", "lying_down"],
        )
        self.assertEqual(public["pose_seed_offsets"]["drinking"], 307)

    def test_pose_seed_offsets_are_action_stable_and_distinct(self) -> None:
        offsets = [
            pipeline.pose_seed_offset(
                f"{preset['action_key']} | first frame: {preset['prompt_en']}",
                index,
            )
            for index, preset in enumerate(pipeline.MULTIPOSE_PRESETS, start=1)
        ]

        self.assertEqual(offsets, [101, 211, 307, 401])
        self.assertEqual(len(set(offsets)), 4)

    def test_uploaded_loras_replace_or_stack_profile(self) -> None:
        replaced, replaced_mode, replaced_source = pipeline.merge_image_profile_loras(
            self.profile,
            self.custom,
            requested_mode="replace",
            resolve_models=False,
        )
        stacked, stacked_mode, stacked_source = pipeline.merge_image_profile_loras(
            self.profile,
            self.custom,
            requested_mode="stack",
            resolve_models=False,
        )

        self.assertEqual(replaced, self.custom)
        self.assertEqual(replaced_mode, "replace")
        self.assertEqual(replaced_source, "user_override")
        self.assertEqual(len(stacked), 3)
        self.assertEqual(stacked_mode, "stack")
        self.assertEqual(stacked_source, "image_profile_plus_user")

    def test_baseline_first_attempt_preserves_source_weights_then_reduces(self) -> None:
        profiles = pipeline.qwen_pose_profiles(
            user_lora=None,
            user_lora_strength=1.0,
            user_lora_mode="replace",
            steps=6,
            cfg=1.0,
            denoise=1.0,
            user_loras=self.profile["user_loras"],
            first_total_cap=self.profile["pose_first_total_cap"],
            first_per_lora_cap=self.profile["pose_first_per_lora_cap"],
        )

        self.assertEqual(
            [item["strength"] for item in profiles[0]["user_loras"]],
            [0.5, 0.6],
        )
        self.assertAlmostEqual(
            sum(float(item["strength"]) for item in profiles[1]["user_loras"]),
            pipeline.QWEN_POSE_USER_LORA_RETRY_TOTAL_CAP,
        )
        self.assertAlmostEqual(
            sum(float(item["strength"]) for item in profiles[2]["user_loras"]),
            pipeline.QWEN_POSE_USER_LORA_FINAL_TOTAL_CAP,
        )

    def test_profile_stack_can_suppress_unrelated_builtin_style(self) -> None:
        with patch.object(
            pipeline,
            "QWEN_STYLE_LORA",
            "unrelated-built-in-style.safetensors",
        ):
            workflow = pipeline.qwen_workflow(
                "input.png",
                "A cat.",
                "bad anatomy",
                1,
                "test/profile",
                user_lora_mode="stack",
                user_loras=self.profile["user_loras"],
                suppress_builtin_style=True,
            )

        self.assertNotIn("17", workflow)
        self.assertEqual(
            workflow["18"]["inputs"]["lora_name"],
            self.profile["user_loras"][0]["model_name"],
        )
        self.assertEqual(workflow["18"]["inputs"]["strength_model"], 0.5)
        self.assertEqual(
            workflow["19"]["inputs"]["lora_name"],
            self.profile["user_loras"][1]["model_name"],
        )
        self.assertEqual(workflow["19"]["inputs"]["strength_model"], 0.6)


class GenerationPlatformFrontendTests(unittest.TestCase):
    def test_frontend_discovers_default_subject_and_restores_image_tab(self) -> None:
        html = PLATFORM_PAGE.read_text(encoding="utf-8")

        self.assertIn('id="subjectProfile" name="subject_profile" type="hidden" value=""', html)
        self.assertIn('id="defaultSubjectName"', html)
        self.assertIn("async function loadSubjects()", html)
        self.assertIn("fetch('api/subjects', {cache: 'no-store'})", html)
        self.assertIn("item?.id === payload.default", html)
        self.assertIn("encodeURIComponent(profile.id)", html)
        self.assertIn("action.reference_ready !== false", html)
        self.assertNotIn('src="api/subjects/sleek_tabby_cat_v1/preview"', html)
        self.assertIn("上传原始宠物图片（可选）", html)
        self.assertIn('data-ui-mode="replacement"', html)
        self.assertIn('data-ui-mode="image"', html)
        self.assertIn("图片单独测试", html)
        self.assertIn('id="image" name="image" type="file"', html)
        self.assertNotIn('id="image" name="image" type="file" required', html)
        self.assertIn("$('image').required = imageMode", html)
        self.assertIn("const identityFront = $('image').files[0]", html)
        self.assertIn("if (identityFront) form.append('image', identityFront)", html)

    def test_frontend_scopes_actions_and_image_poses_by_tab(self) -> None:
        html = PLATFORM_PAGE.read_text(encoding="utf-8")

        self.assertIn('class="action" type="checkbox" value="eating" checked disabled', html)
        self.assertIn('class="action" type="checkbox" value="grooming" disabled', html)
        self.assertIn(".action:checked:not(:disabled)", html)
        self.assertIn("eating 使用 Main_case/eat.mp4", html)
        self.assertIn("grooming 使用 Main_case/TM.mp4", html)
        self.assertIn('data-pose-key="running"', html)
        self.assertIn('data-pose-key="eating"', html)
        self.assertIn('data-pose-key="drinking"', html)
        self.assertIn('data-pose-key="lying_down"', html)
        self.assertIn('id="customPoses"', html)
        self.assertIn("selectedImagePoses()", html)

    def test_frontend_submits_replacement_and_image_fields(self) -> None:
        html = PLATFORM_PAGE.read_text(encoding="utf-8")

        self.assertIn("form.append('generation_mode', 'subject_replacement')", html)
        self.assertIn("form.append('subject_profile'", html)
        self.assertIn("form.append('actions', JSON.stringify(actions))", html)
        self.assertIn("form.append('scail_steps'", html)
        self.assertIn("form.append('pose_strength'", html)
        self.assertNotIn("form.append('mode', 'video')", html)
        self.assertNotIn("form.append('request_kind', 'video')", html)
        self.assertNotIn("form.append('action_count'", html)
        self.assertIn('id="scailSteps" type="number" min="4" max="12" value="6"', html)
        self.assertIn('id="poseStrength" type="number" min="0" max="2"', html)
        self.assertIn('id="seed" type="number" min="0" value="20260726"', html)
        self.assertIn('for="seed">基础随机种子</label>', html)
        self.assertIn("preparing_subject_replacement", html)
        self.assertIn("subject_references_ready", html)
        self.assertIn("generating_subject_replacement", html)
        self.assertIn("extracting_transparent_animation", html)
        self.assertIn("job.current_action_label", html)
        self.assertIn("job.current_action_index", html)
        self.assertIn("job.action_count", html)
        self.assertIn("subject_reference: 'SCAIL 主体参考图（身份/外观）'", html)
        self.assertIn("subject_replacement_video: '主体替换 MP4'", html)
        self.assertIn("form.append('generation_mode', 'keyframe_video')", html)
        self.assertIn("form.append('mode', 'image')", html)
        self.assertIn("form.append('request_kind', 'image')", html)
        self.assertIn("form.append('skip_video', 'true')", html)
        self.assertIn("form.append('poses', JSON.stringify(poses))", html)
        self.assertIn('id="qwenSteps" type="number" min="1" max="30" value="6"', html)
        self.assertIn('id="qwenCfg" type="number"', html)
        self.assertIn('id="qwenDenoise" type="number"', html)
        self.assertNotIn('id="wanSteps"', html)
        self.assertNotIn('id="skipVideo"', html)
        self.assertIn('id="lora" class="lora-file" type="file"', html)
        self.assertNotIn('id="lora" class="lora-file" type="file" multiple', html)
        self.assertIn("loraItems.splice(index, 1)", html)
        self.assertIn("loraFilenameKey(item.file.name)", html)

    def test_frontend_preserves_active_jobs_and_media_across_transient_polls(self) -> None:
        html = PLATFORM_PAGE.read_text(encoding="utf-8")

        self.assertIn("pollFailureCount += 1", html)
        self.assertIn("Math.min(30000", html)
        self.assertIn("setBusy(true);", html)
        self.assertIn("poll(currentJob.status_url)", html)
        self.assertIn("function outputSignature(outputs)", html)
        self.assertIn("function renderOutputsIfChanged(outputs)", html)
        self.assertIn("if (signature === renderedOutputSignature) return", html)
        self.assertIn("job.subject_source === 'profile'", html)
        self.assertIn("job.subject_source === 'upload_pending_qwen'", html)
        self.assertIn("job.subject_source === 'qwen_generated_upload'", html)
        self.assertNotIn("reference_image_uploaded", html)


class IdentityPaletteGuardTests(unittest.TestCase):
    @staticmethod
    def palette(
        *,
        achromatic: float,
        warm: float,
        warm_category: float | None = None,
        saturated_warm: float = 0.0,
    ) -> dict[str, object]:
        fractions = {category: 0.0 for category in pipeline.IDENTITY_PALETTE_CATEGORIES}
        fractions["warm_brown_or_orange"] = (
            warm if warm_category is None else warm_category
        )
        fractions["achromatic_dark"] = achromatic / 2.0
        fractions["achromatic_light"] = achromatic / 2.0
        return {
            "available": True,
            "fractions": fractions,
            "achromatic_fraction": achromatic,
            "warm_fraction": warm,
            "saturated_warm_fraction": saturated_warm,
            "dominant": [],
        }

    @staticmethod
    def write_alpha_subject(
        path: Path, left: tuple[int, int, int], right: tuple[int, int, int]
    ) -> None:
        image = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.rectangle((32, 32, 127, 223), fill=(*left, 255))
        draw.rectangle((128, 32, 223, 223), fill=(*right, 255))
        image.save(path)

    def test_black_white_source_rejects_new_orange_coat(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "source.png"
            candidate_path = root / "candidate.png"
            self.write_alpha_subject(source_path, (24, 24, 24), (242, 242, 242))
            self.write_alpha_subject(candidate_path, (195, 92, 24), (242, 242, 242))

            source = pipeline.identity_palette_signature(source_path)
            candidate = pipeline.identity_palette_signature(candidate_path)
            quality = pipeline.identity_palette_quality(source, candidate)

        self.assertTrue(source["available"])
        self.assertEqual(source["mask_method"], "alpha")
        self.assertTrue(candidate["available"])
        self.assertFalse(quality["acceptable"])
        self.assertIn("new_dominant_warm_brown_or_orange", quality["issues"])

    def test_matching_black_white_palette_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "source.png"
            candidate_path = root / "candidate.png"
            self.write_alpha_subject(source_path, (24, 24, 24), (242, 242, 242))
            self.write_alpha_subject(candidate_path, (30, 30, 30), (235, 235, 235))

            source = pipeline.identity_palette_signature(source_path)
            candidate = pipeline.identity_palette_signature(candidate_path)
            quality = pipeline.identity_palette_quality(source, candidate)

        self.assertTrue(quality["available"])
        self.assertTrue(quality["acceptable"])
        self.assertEqual(quality["issues"], [])

    def test_localized_head_skin_color_is_warning_but_body_spread_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "source.png"
            head_accent_path = root / "head-accent.png"
            body_spread_path = root / "body-spread.png"
            source_image = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
            source_draw = ImageDraw.Draw(source_image)
            source_draw.rectangle((32, 24, 223, 231), fill=(128, 128, 128, 255))
            source_image.save(source_path)
            head_accent = source_image.copy()
            ImageDraw.Draw(head_accent).rectangle(
                (40, 32, 215, 92),
                fill=(212, 86, 100, 255),
            )
            head_accent.save(head_accent_path)
            body_spread = source_image.copy()
            ImageDraw.Draw(body_spread).rectangle(
                (40, 150, 215, 220),
                fill=(212, 86, 100, 255),
            )
            body_spread.save(body_spread_path)

            source = pipeline.identity_palette_signature(source_path)
            localized = pipeline.identity_palette_signature(head_accent_path)
            spread = pipeline.identity_palette_signature(body_spread_path)
            localized_quality = pipeline.identity_palette_quality(source, localized)
            spread_quality = pipeline.identity_palette_quality(source, spread)

        self.assertTrue(localized["red_or_pink_localized_to_head"])
        self.assertTrue(localized_quality["acceptable"], localized_quality)
        self.assertIn(
            "localized_red_or_pink_skin_accent_expanded",
            localized_quality["warnings"],
        )
        self.assertFalse(spread["red_or_pink_localized_to_head"])
        self.assertFalse(spread_quality["acceptable"])
        self.assertIn("new_dominant_red_or_pink", spread_quality["issues"])

    def test_paired_green_eyes_pass_but_green_body_spill_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "source.png"
            eyes_path = root / "green-eyes.png"
            spill_path = root / "green-body-spill.png"
            source_image = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
            source_draw = ImageDraw.Draw(source_image)
            source_draw.rectangle((32, 24, 223, 231), fill=(128, 128, 128, 255))
            source_image.save(source_path)
            eyes = source_image.copy()
            eyes_draw = ImageDraw.Draw(eyes)
            eyes_draw.ellipse((72, 58, 99, 82), fill=(42, 142, 62, 255))
            eyes_draw.ellipse((156, 58, 183, 82), fill=(42, 142, 62, 255))
            eyes.save(eyes_path)
            spill = source_image.copy()
            ImageDraw.Draw(spill).rectangle(
                (48, 138, 207, 218),
                fill=(42, 170, 70, 255),
            )
            spill.save(spill_path)

            source = pipeline.identity_palette_signature(source_path)
            eyes_signature = pipeline.identity_palette_signature(eyes_path)
            spill_signature = pipeline.identity_palette_signature(spill_path)
            eyes_quality = pipeline.identity_palette_quality(source, eyes_signature)
            spill_quality = pipeline.identity_palette_quality(source, spill_signature)

        self.assertTrue(eyes_signature["green_or_cyan_localized_to_head"])
        self.assertTrue(eyes_quality["acceptable"], eyes_quality)
        self.assertFalse(spill_signature["green_or_cyan_localized_to_head"])
        self.assertFalse(spill_quality["acceptable"])
        self.assertIn(
            "green_or_cyan_chroma_spill_on_body",
            spill_quality["issues"],
        )

    def test_identity_aware_despill_preserves_paired_eyes_and_alpha(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "candidate.png"
            image = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
            draw = ImageDraw.Draw(image)
            draw.rectangle((32, 24, 223, 231), fill=(132, 132, 132, 255))
            eye_color = (32, 154, 52, 255)
            draw.ellipse((72, 58, 99, 82), fill=eye_color)
            draw.ellipse((156, 58, 183, 82), fill=eye_color)
            # A low-hue, low-red-delta screen reflection like the pale green cast
            # observed on the white-dog E2E canonical.
            draw.rectangle((76, 138, 179, 208), fill=(163, 169, 125, 255))
            image.save(path)
            alpha_before = image.getchannel("A").tobytes()
            source_palette = {
                "available": True,
                "fractions": {"green": 0.0, "blue_or_cyan": 0.0},
                "green_or_cyan_body_fraction": 0.0,
            }

            result = pipeline.neutralize_uploaded_identity_chroma_spill(
                path,
                source_palette,
            )
            with Image.open(path) as corrected_image:
                corrected = corrected_image.convert("RGBA")

        self.assertTrue(result["applied"], result)
        self.assertTrue(result["alpha_unchanged"])
        self.assertGreater(result["protected_eye_pixels"], 0)
        self.assertEqual(corrected.getpixel((85, 70)), eye_color)
        chest = corrected.getpixel((128, 170))
        self.assertLessEqual(max(chest[:3]) - min(chest[:3]), 3)
        self.assertEqual(corrected.getchannel("A").tobytes(), alpha_before)

    def test_identity_aware_despill_reconstructs_warm_local_coat_chroma(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "warm-dog-candidate.png"
            image = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
            draw = ImageDraw.Draw(image)
            warm_coat = (194, 158, 96, 255)
            draw.rectangle((32, 24, 223, 231), fill=warm_coat)
            draw.rectangle((76, 138, 179, 208), fill=(82, 172, 94, 255))
            image.save(path)
            alpha_before = image.getchannel("A").tobytes()
            source_palette = {
                "available": True,
                "fractions": {"green": 0.0, "blue_or_cyan": 0.0},
                "green_or_cyan_body_fraction": 0.0,
            }

            result = pipeline.neutralize_uploaded_identity_chroma_spill(
                path,
                source_palette,
            )
            with Image.open(path) as corrected_image:
                corrected = corrected_image.convert("RGBA")

        chest = corrected.getpixel((128, 170))
        self.assertTrue(result["applied"], result)
        self.assertEqual(
            result["chroma_reconstruction"],
            "local_lab_telea_inpaint",
        )
        self.assertGreater(result["chroma_donor_pixels"], 0)
        self.assertGreater(chest[0], chest[1])
        self.assertGreater(chest[1], chest[2])
        self.assertGreater(max(chest[:3]) - min(chest[:3]), 20)
        self.assertEqual(corrected.getchannel("A").tobytes(), alpha_before)

    def test_unavailable_identity_palette_signature_fails_closed(self) -> None:
        source = {"available": True, "fractions": {}}
        candidate = {"available": False, "reason": "empty_subject_mask"}

        quality = pipeline.identity_palette_quality(source, candidate)

        self.assertFalse(quality["available"])
        self.assertFalse(quality["acceptable"])
        self.assertEqual(
            quality["issues"],
            ["identity_palette_signature_unavailable"],
        )
        self.assertGreaterEqual(quality["penalty"], 1000.0)

    def test_source_chromatic_identity_accents_cannot_become_full_grayscale(
        self,
    ) -> None:
        source = self.palette(achromatic=0.92338, warm=0.071366)
        source["chromatic_fraction"] = 0.07662
        candidate = self.palette(achromatic=0.999984, warm=0.0)
        candidate["chromatic_fraction"] = 0.000016

        quality = pipeline.identity_palette_quality(source, candidate)

        self.assertFalse(quality["acceptable"])
        self.assertTrue(quality["chromatic_accents_lost"])
        self.assertIn(
            "source_chromatic_identity_accents_lost",
            quality["issues"],
        )

    def test_warm_multicolor_reference_is_not_misclassified_as_achromatic(self) -> None:
        source = self.palette(achromatic=0.752012, warm=0.241811)
        candidate = self.palette(achromatic=0.477007, warm=0.508855)

        quality = pipeline.identity_palette_quality(source, candidate)

        self.assertTrue(quality["acceptable"], quality)
        self.assertNotIn(
            "achromatic_source_recolored_with_warm_fur",
            quality["issues"],
        )

    def test_achromatic_source_rejects_canonical_warm_drift_at_real_job_boundary(
        self,
    ) -> None:
        source = self.palette(achromatic=0.880764, warm=0.118038)
        candidate = self.palette(achromatic=0.692824, warm=0.254740)

        quality = pipeline.identity_palette_quality(source, candidate)

        self.assertFalse(quality["acceptable"])
        self.assertIn(
            "achromatic_source_recolored_with_warm_fur",
            quality["issues"],
        )

    def test_achromatic_guard_uses_relative_drop_without_absolute_threshold_cliff(
        self,
    ) -> None:
        source = self.palette(achromatic=0.880000, warm=0.118000)
        candidate = self.palette(achromatic=0.701000, warm=0.245000)

        quality = pipeline.identity_palette_quality(source, candidate)

        self.assertFalse(quality["acceptable"])
        self.assertIn(
            "achromatic_source_recolored_with_warm_fur",
            quality["issues"],
        )

    def test_achromatic_guard_rejects_new_saturated_orange_patches(self) -> None:
        source = self.palette(
            achromatic=0.880764,
            warm=0.118038,
            saturated_warm=0.011,
        )
        candidate = self.palette(
            achromatic=0.790181,
            warm=0.158354,
            saturated_warm=0.07658,
        )

        quality = pipeline.identity_palette_quality(source, candidate)

        self.assertFalse(quality["acceptable"])
        self.assertFalse(quality["broad_warm_drift"])
        self.assertTrue(quality["saturated_warm_drift"])
        self.assertIn(
            "achromatic_source_recolored_with_warm_fur",
            quality["issues"],
        )

    def test_achromatic_prompt_keeps_pink_skin_accents_small(self) -> None:
        source = self.palette(achromatic=0.91, warm=0.088)

        prompt = pipeline.identity_palette_prompt(source, retry=True)
        negative = pipeline.identity_palette_negative_prompt(source)

        self.assertIn("uploaded pet", prompt)
        self.assertNotIn("uploaded cat", prompt)
        self.assertIn("Red or pink is a small skin accent, never a fur color", prompt)
        self.assertIn("tiny low-saturation nose pad", prompt)
        self.assertIn("Do not enlarge, saturate, or spread it", prompt)
        self.assertIn("large pink inner-ear fill", negative)
        self.assertIn("saturated pink ears", negative)
        self.assertIn("rosy facial cast", negative)

    def test_low_saturation_photographic_warmth_keeps_silver_coat_achromatic(
        self,
    ) -> None:
        source = self.palette(
            achromatic=0.878335,
            warm=0.121665,
            saturated_warm=0.024,
        )

        prompt = pipeline.identity_palette_prompt(source)
        negative = pipeline.identity_palette_negative_prompt(source)

        self.assertTrue(pipeline.identity_palette_is_achromatic(source))
        self.assertIn("visible source coat is achromatic", prompt)
        self.assertIn("no amber, sepia, orange, ginger, brown, or golden", prompt)
        self.assertIn("overall warm color cast", negative)

    def test_canonical_upper_body_palette_drift_overrides_primary_pass(self) -> None:
        primary = {
            "available": True,
            "acceptable": True,
            "issues": [],
            "warnings": [],
            "penalty": 3.0,
        }
        upper_body = {
            "available": True,
            "acceptable": False,
            "issues": ["achromatic_source_recolored_with_warm_fur"],
            "warnings": [],
            "penalty": 109.0,
        }

        quality = pipeline.canonical_identity_palette_quality(primary, upper_body)

        self.assertIsNotNone(quality)
        self.assertFalse(quality["acceptable"])
        self.assertEqual(quality["method"], "canonical_primary_and_upper_body_v1")
        self.assertIn(
            "achromatic_source_recolored_with_warm_fur",
            quality["issues"],
        )

    def test_canonical_palette_requires_both_audits(self) -> None:
        primary = {
            "available": True,
            "acceptable": True,
            "issues": [],
            "warnings": [],
            "penalty": 2.0,
        }
        upper_body = {
            "available": True,
            "acceptable": True,
            "issues": [],
            "warnings": [],
            "penalty": 4.0,
        }

        quality = pipeline.canonical_identity_palette_quality(primary, upper_body)

        self.assertIsNotNone(quality)
        self.assertTrue(quality["acceptable"])
        self.assertEqual(quality["penalty"], 4.0)

    def test_bowl_palette_contamination_can_be_cleared_by_upper_body_audit(
        self,
    ) -> None:
        primary = {
            "available": True,
            "acceptable": False,
            "warning": True,
            "issues": ["achromatic_source_recolored_with_warm_fur"],
            "warnings": [],
            "penalty": 125.0,
        }
        upper_body = {
            "available": True,
            "acceptable": True,
            "warning": False,
            "issues": [],
            "warnings": [],
            "penalty": 4.0,
        }

        quality = pipeline.bowl_aware_identity_palette_quality(primary, upper_body)

        self.assertTrue(quality["acceptable"], quality)
        self.assertEqual(quality["method"], "bowl_upper_body_confirmation")
        self.assertIn("bowl_prop_palette_contamination_ignored", quality["warnings"])
        self.assertIs(quality["primary_quality"], primary)
        self.assertIs(quality["upper_body_quality"], upper_body)

    def test_bowl_upper_body_palette_drift_is_still_rejected(self) -> None:
        primary = {
            "available": True,
            "acceptable": False,
            "issues": ["new_dominant_warm_brown_or_orange"],
            "penalty": 125.0,
        }
        upper_body = {
            "available": True,
            "acceptable": False,
            "issues": ["achromatic_source_recolored_with_warm_fur"],
            "penalty": 110.0,
        }

        quality = pipeline.bowl_aware_identity_palette_quality(primary, upper_body)

        self.assertFalse(quality["acceptable"])
        self.assertEqual(quality["method"], "bowl_upper_body_confirmation")
        self.assertIn(
            "achromatic_source_recolored_with_warm_fur",
            quality["issues"],
        )
        self.assertIn("new_dominant_warm_brown_or_orange", quality["issues"])

    def test_bowl_upper_body_drift_overrides_a_diluted_primary_pass(self) -> None:
        primary = {
            "available": True,
            "acceptable": True,
            "issues": [],
            "warnings": [],
            "penalty": 2.0,
        }
        upper_body = {
            "available": True,
            "acceptable": False,
            "issues": ["new_dominant_warm_brown_or_orange"],
            "warnings": [],
            "penalty": 108.0,
        }

        quality = pipeline.bowl_aware_identity_palette_quality(primary, upper_body)

        self.assertFalse(quality["acceptable"])
        self.assertEqual(quality["method"], "bowl_upper_body_confirmation")
        self.assertIn("new_dominant_warm_brown_or_orange", quality["issues"])

    def test_fallback_selection_skips_identity_drift_candidate(self) -> None:
        drifted = {
            "id": "strong-style-but-recolored",
            "identity_palette": {
                "acceptable": False,
                "penalty": 1.0,
                "issues": ["new_dominant_warm_brown_or_orange"],
            },
            "quality": {"penalty": 0.0},
            "chroma": {"acceptable": True},
        }
        identity_safe = {
            "id": "identity-safe-retry",
            "identity_palette": {
                "acceptable": True,
                "penalty": 5.0,
                "issues": [],
            },
            "quality": {"penalty": 25.0},
            "chroma": {"acceptable": False},
        }

        selected = pipeline.select_canonical_fallback_attempt(
            [drifted, identity_safe],
            identity_palette_required=True,
        )

        self.assertIs(selected, identity_safe)

    @patch.object(pipeline, "GENERATION_VISUAL_AUDITS_BLOCKING", True)
    def test_strict_fallback_selection_fails_when_all_candidates_drift(self) -> None:
        attempts = [
            {
                "identity_palette": {
                    "acceptable": False,
                    "penalty": 1.0,
                    "issues": ["new_dominant_warm_brown_or_orange"],
                },
                "quality": {"penalty": 0.0},
                "chroma": {"acceptable": True},
            },
            {
                "identity_palette": {
                    "acceptable": False,
                    "penalty": 2.0,
                    "issues": ["achromatic_source_recolored_with_warm_fur"],
                },
                "quality": {"penalty": 0.0},
                "chroma": {"acceptable": True},
            },
        ]

        with self.assertRaisesRegex(
            RuntimeError,
            "Canonical identity palette drift remained after 2 attempts",
        ):
            pipeline.select_canonical_fallback_attempt(
                attempts,
                identity_palette_required=True,
            )


if __name__ == "__main__":
    unittest.main()
