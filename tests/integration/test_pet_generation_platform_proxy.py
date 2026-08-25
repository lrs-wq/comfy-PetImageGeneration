from __future__ import annotations

import io
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from starlette.datastructures import UploadFile

import pet_generation_platform as platform


class GenerationPlatformProxyTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def request_arguments(**overrides: object) -> dict[str, object]:
        arguments: dict[str, object] = {
            "image": None,
            "image_left": None,
            "image_right": None,
            "lora": None,
            "prompt": "",
            "mode": "video",
            "generation_mode": "subject_replacement",
            "actions": '["eating"]',
            "subject_profile": None,
            "image_profile": None,
            "transparent": "true",
            "poses": None,
            "pose_count": 4,
            "output_format": "transparent_gif",
            "skip_video": "false",
            "lora_strength": 1.0,
            "lora_strengths": None,
            "lora_mode": "replace",
            "qwen_steps": None,
            "qwen_cfg": 1.0,
            "qwen_denoise": 1.0,
            "wan_steps": platform.pipeline.WAN_STEPS,
            "wan_cfg": platform.pipeline.WAN_CFG,
            "wan_shift": platform.pipeline.WAN_SHIFT,
            "scail_steps": 6,
            "pose_strength": None,
            "seed": 20260726,
            "_": None,
        }
        arguments.update(overrides)
        return arguments

    @staticmethod
    def accepted_response(job_id: str = "job-1") -> Mock:
        response = Mock()
        response.status_code = 202
        response.text = ""
        response.json.return_value = {
            "job_id": job_id,
            "request_kind": "video",
            "outputs": [],
        }
        return response

    async def test_subject_replacement_omits_empty_profile_and_posts_in_thread(self) -> None:
        response = self.accepted_response()
        to_thread = AsyncMock(return_value=response)
        with (
            patch.object(platform, "video_headers", return_value={"X-API-Key": "test"}),
            patch.object(platform.asyncio, "to_thread", to_thread),
        ):
            result = await platform.create_job(**self.request_arguments())

        self.assertEqual(result["job_id"], "job-1")
        to_thread.assert_awaited_once()
        call = to_thread.await_args
        assert call is not None
        self.assertIs(call.args[0], platform.requests.post)
        self.assertNotIn("subject_profile", call.kwargs["data"])

    async def test_subject_replacement_forwards_explicit_profile(self) -> None:
        response = self.accepted_response("job-2")
        to_thread = AsyncMock(return_value=response)
        with (
            patch.object(platform, "video_headers", return_value={"X-API-Key": "test"}),
            patch.object(platform.asyncio, "to_thread", to_thread),
        ):
            await platform.create_job(
                **self.request_arguments(subject_profile="  configured-cat  ")
            )

        call = to_thread.await_args
        assert call is not None
        self.assertEqual(call.kwargs["data"]["subject_profile"], "configured-cat")

    async def test_subject_replacement_forwards_complete_three_view_set(self) -> None:
        response = self.accepted_response("three-view-job")
        to_thread = AsyncMock(return_value=response)
        streams = {
            role: io.BytesIO(role.encode("ascii"))
            for role in ("front", "left", "right")
        }
        uploads = {
            role: UploadFile(
                filename=f"pet-{role}.png",
                file=streams[role],
                headers={"content-type": "image/png"},
            )
            for role in streams
        }
        with (
            patch.object(platform, "video_headers", return_value={"X-API-Key": "test"}),
            patch.object(platform.asyncio, "to_thread", to_thread),
        ):
            result = await platform.create_job(
                **self.request_arguments(
                    image=uploads["front"],
                    image_left=uploads["left"],
                    image_right=uploads["right"],
                )
            )

        self.assertEqual(result["job_id"], "three-view-job")
        call = to_thread.await_args
        assert call is not None
        self.assertEqual(
            set(call.kwargs["files"]),
            {"image", "image_left", "image_right"},
        )
        self.assertEqual(call.kwargs["files"]["image"][1], b"front")
        self.assertEqual(call.kwargs["files"]["image_left"][1], b"left")
        self.assertEqual(call.kwargs["files"]["image_right"][1], b"right")
        self.assertTrue(all(stream.closed for stream in streams.values()))

    async def test_subject_replacement_rejects_partial_three_view_set(self) -> None:
        front_stream = io.BytesIO(b"front")
        left_stream = io.BytesIO(b"left")
        front = UploadFile(filename="front.png", file=front_stream)
        left = UploadFile(filename="left.png", file=left_stream)

        with self.assertRaises(platform.HTTPException) as raised:
            await platform.create_job(
                **self.request_arguments(
                    image=front,
                    image_left=left,
                )
            )

        self.assertEqual(raised.exception.status_code, 422)
        self.assertIn("complete image", raised.exception.detail)
        self.assertTrue(front_stream.closed)
        self.assertTrue(left_stream.closed)

    async def test_image_proxy_post_also_runs_in_thread(self) -> None:
        response = self.accepted_response("image-job")
        to_thread = AsyncMock(return_value=response)
        upload = UploadFile(filename="cat.png", file=io.BytesIO(b"image-bytes"))
        with (
            patch.object(platform, "video_headers", return_value={"X-API-Key": "test"}),
            patch.object(platform.pipeline, "resolve_image_profile", return_value=None),
            patch.object(platform.asyncio, "to_thread", to_thread),
        ):
            await platform.create_job(
                **self.request_arguments(
                    image=upload,
                    prompt="cartoon cat",
                    mode="image",
                    generation_mode="keyframe_video",
                    image_profile="",
                )
            )

        call = to_thread.await_args
        assert call is not None
        self.assertIs(call.args[0], platform.requests.post)
        self.assertEqual(call.kwargs["data"]["request_kind"], "image")

    def test_create_page_explains_default_direct_and_upload_identity_paths(self) -> None:
        html = platform.HOST_PAGE.read_text(encoding="utf-8")

        self.assertIn("未上传图片时的 Main_case 标准主体", html)
        self.assertIn("同时决定主体、毛发、动作和道具", html)
        self.assertIn("Main_case/eat.mp4", html)
        self.assertIn("grooming 使用 Main_case/TM.mp4", html)
        self.assertIn("上传图是身份来源", html)
        self.assertIn("standard_direct_reference", html)
        self.assertIn("Main_case 标准直出正常（上传链路降级）", html)
        self.assertIn("standardDirect.status === 'ready'", html)
        self.assertIn("上传图片后，上传图决定身份", html)
        self.assertIn("正面 + 左侧 + 右侧", html)
        self.assertIn("image_left", html)
        self.assertIn("image_right", html)
        self.assertIn("standardDirect.status === 'ready'", html)

    def test_create_page_renders_server_driving_standard_provenance(self) -> None:
        html = platform.HOST_PAGE.read_text(encoding="utf-8")

        self.assertIn("function drivingStandardsSummary(profile)", html)
        self.assertIn("profile?.driving_standards", html)
        self.assertIn("record.source_filename", html)
        self.assertIn("record.source_sha256", html)
        self.assertIn("digest.slice(0, 8)", html)
        self.assertIn("主体与动作标准 ${revision}", html)
        self.assertIn("profileActionIds.map(action => actionSummary", html)
        self.assertIn("defaultDrivingStandardsSummary = drivingStandardsSummary(profile)", html)
        self.assertIn("$('defaultSubject').hidden = true", html)
        self.assertIn(
            "$('defaultSubject').hidden = uiMode === 'image'",
            html,
        )

    def test_create_page_selects_medium_or_small_dog_test_profile(self) -> None:
        html = platform.HOST_PAGE.read_text(encoding="utf-8")

        self.assertIn('id="dogSizeProfile"', html)
        self.assertNotIn('id="dogSizeProfile" disabled', html)
        self.assertIn("dog_medium_white_model_test_v1", html)
        self.assertIn("dog_small_white_model_test_v1", html)
        self.assertIn(">中型犬<", html)
        self.assertIn(">小型犬<", html)
        self.assertIn('value="standing"', html)
        self.assertIn("standing: '站立'", html)
        for action, label in (
            ("drinking", "喝水"),
            ("running", "奔跑"),
            ("sleeping", "睡觉"),
            ("toileting", "上厕所"),
        ):
            self.assertIn(f'value="{action}"', html)
            self.assertIn(f"{action}: '{label}'", html)
        self.assertIn("function applySubjectProfile(profile, backendConfirmed = true)", html)
        self.assertIn("const selectedBeforeApply = new Set(", html)
        self.assertIn("input.checked = enabled && selectedBeforeApply.has(input.value)", html)
        self.assertIn("const requestedProfileId = $('dogSizeProfile').value", html)
        self.assertIn("const subjectProfileId = $('dogSizeProfile').value || $('subjectProfile').value", html)
        self.assertIn("if (subjectProfileId) form.append('subject_profile', subjectProfileId)", html)
        self.assertNotIn("if (!subjectsReady || !$('subjectProfile').value)", html)
        self.assertIn("本次将严格按该体型白膜生成", html)
        self.assertIn("fallbackSubjectProfiles", html)
        self.assertIn("scheduleSubjectProfileRetry()", html)
        self.assertIn("setTimeout(loadSubjects, SUBJECTS_RETRY_MS)", html)
        self.assertIn("if (!subjectsReady) loadSubjects()", html)

    def test_create_page_disables_browser_caching(self) -> None:
        page = platform.index(None)

        self.assertEqual(
            page.headers["cache-control"],
            "no-store, no-cache, must-revalidate, max-age=0",
        )
        self.assertEqual(page.headers["pragma"], "no-cache")
        self.assertEqual(page.headers["expires"], "0")

    def test_subject_response_disables_browser_caching(self) -> None:
        response = Mock()
        response.status_code = 200
        response.json.return_value = {"default": "sleek_tabby_cat_v1", "items": []}
        with (
            patch.object(platform, "video_headers", return_value={"X-API-Key": "test"}),
            patch.object(platform.requests, "get", return_value=response),
        ):
            result = platform.subjects(None)

        self.assertEqual(
            result.headers["cache-control"],
            "no-store, no-cache, must-revalidate, max-age=0",
        )
        self.assertEqual(result.headers["pragma"], "no-cache")
        self.assertEqual(result.headers["expires"], "0")


if __name__ == "__main__":
    unittest.main()
