import json
import os
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("LOCAL_VIDEO_API_KEY", "test-key")

from fastapi import HTTPException, UploadFile
from starlette.requests import Request

import simple_pet_gif_api as api


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self.headers = {}

    @property
    def ok(self):
        return 200 <= self.status_code < 400

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def make_request(host="api.example.test:8051"):
    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "server": ("api.example.test", 8051),
            "path": "/api",
            "raw_path": b"/api",
            "query_string": b"",
            "headers": [(b"host", host.encode("ascii"))],
            "client": ("127.0.0.1", 12345),
        }
    )


def upload(name, value):
    return UploadFile(filename=name, file=BytesIO(value))


async def immediate_to_thread(function, *args, **kwargs):
    return function(*args, **kwargs)


class SimplePetGifApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_front_only_upload_uses_fixed_two_action_contract(self):
        captured = {}

        def fake_post(url, **kwargs):
            captured["url"] = url
            captured["data"] = kwargs["data"]
            captured["file_keys"] = set(kwargs["files"])
            captured["front"] = kwargs["files"]["image"][1].read()
            return FakeResponse(
                202,
                {"job_id": "20260807T120000Z-test1234", "status": "queued"},
            )

        with (
            patch.object(api.asyncio, "to_thread", new=immediate_to_thread),
            patch.object(api.requests, "post", side_effect=fake_post),
        ):
            response = await api.api_gateway(
                make_request(),
                operation="upload",
                key=None,
                front=upload("front.png", b"front"),
                left=None,
                right=None,
                _auth=None,
            )

        self.assertEqual(response.status_code, 202)
        self.assertEqual(
            json.loads(response.body),
            {"key": "20260807T120000Z-test1234", "status": "queued"},
        )
        self.assertEqual(captured["file_keys"], {"image"})
        self.assertEqual(captured["front"], b"front")
        self.assertEqual(json.loads(captured["data"]["actions"]), ["eating", "grooming"])
        self.assertEqual(captured["data"]["output_format"], "transparent_gif")
        self.assertEqual(captured["data"]["resolution"], "512x512")

    async def test_three_view_upload_maps_all_images(self):
        captured = {}

        def fake_post(url, **kwargs):
            del url
            captured["file_keys"] = set(kwargs["files"])
            return FakeResponse(202, {"job_id": "three-view-key", "status": "queued"})

        with (
            patch.object(api.asyncio, "to_thread", new=immediate_to_thread),
            patch.object(api.requests, "post", side_effect=fake_post),
        ):
            await api.api_gateway(
                make_request(),
                operation="upload",
                key=None,
                front=upload("front.png", b"front"),
                left=upload("left.png", b"left"),
                right=upload("right.png", b"right"),
                _auth=None,
            )
        self.assertEqual(captured["file_keys"], {"image", "image_left", "image_right"})

    async def test_partial_side_view_is_rejected(self):
        front = upload("front.png", b"front")
        left = upload("left.png", b"left")
        with self.assertRaises(HTTPException) as caught:
            await api.api_gateway(
                make_request(),
                operation="upload",
                key=None,
                front=front,
                left=left,
                right=None,
                _auth=None,
            )
        self.assertEqual(caught.exception.status_code, 422)
        self.assertTrue(front.file.closed)
        self.assertTrue(left.file.closed)

    async def test_upload_requires_front(self):
        with self.assertRaises(HTTPException) as caught:
            await api.api_gateway(
                make_request(),
                operation="upload",
                key=None,
                front=None,
                left=None,
                right=None,
                _auth=None,
            )
        self.assertEqual(caught.exception.status_code, 422)

    async def test_query_returns_only_two_gif_urls(self):
        payload = {
            "status": "completed",
            "progress": 100,
            "stage": "completed",
            "outputs": [
                {
                    "type": "transparent_gif",
                    "action": "eating",
                    "filename": "eating.gif",
                    "primary": True,
                },
                {
                    "type": "transparent_gif",
                    "action": "grooming",
                    "filename": "grooming.gif",
                    "primary": True,
                },
                {"type": "mp4", "action": "eating", "filename": "hidden.mp4"},
                {"type": "png", "action": "grooming", "filename": "hidden.png"},
            ],
        }
        with (
            patch.object(api, "PUBLIC_BASE_URL", "http://39.106.120.253:8051"),
            patch.object(api.asyncio, "to_thread", new=immediate_to_thread),
            patch.object(api.requests, "get", return_value=FakeResponse(200, payload)),
        ):
            result = await api.api_gateway(
                make_request(),
                operation="query",
                key="task-key",
                front=None,
                left=None,
                right=None,
                _auth=None,
            )
        self.assertEqual(result["status"], "completed")
        self.assertEqual(
            result["gifs"],
            {
                "eating": "http://39.106.120.253:8051/file/task-key/eating.gif",
                "grooming": "http://39.106.120.253:8051/file/task-key/grooming.gif",
            },
        )
        self.assertNotIn("outputs", result)

    async def test_completed_job_missing_a_gif_is_failed_contract(self):
        payload = {
            "status": "completed",
            "progress": 100,
            "outputs": [
                {
                    "type": "transparent_gif",
                    "action": "eating",
                    "filename": "eating.gif",
                }
            ],
        }
        with (
            patch.object(api.asyncio, "to_thread", new=immediate_to_thread),
            patch.object(api.requests, "get", return_value=FakeResponse(200, payload)),
        ):
            result = await api.api_gateway(
                make_request(),
                operation="query",
                key="task-key",
                front=None,
                left=None,
                right=None,
                _auth=None,
            )
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["stage"], "output_contract_failed")
        self.assertIn("grooming", result["error"])

    async def test_query_preserves_nonfrontal_front_warning_while_running(self):
        payload = {
            "status": "running",
            "progress": 26.5,
            "stage": "front_view_not_reliably_frontal",
            "current_action": None,
            "error": None,
            "outputs": [],
        }
        with (
            patch.object(api.asyncio, "to_thread", new=immediate_to_thread),
            patch.object(api.requests, "get", return_value=FakeResponse(200, payload)),
        ):
            result = await api.api_gateway(
                make_request(),
                operation="query",
                key="nonfrontal-front-key",
                front=None,
                left=None,
                right=None,
                _auth=None,
            )

        self.assertEqual(result["status"], "running")
        self.assertEqual(result["progress"], 26.5)
        self.assertEqual(result["stage"], "front_view_not_reliably_frontal")
        self.assertIsNone(result["error"])
        self.assertEqual(result["gifs"], {"eating": None, "grooming": None})

    async def test_query_rejects_invalid_key(self):
        with self.assertRaises(HTTPException) as caught:
            await api.api_gateway(
                make_request(),
                operation="query",
                key="../../status.json",
                front=None,
                left=None,
                right=None,
                _auth=None,
            )
        self.assertEqual(caught.exception.status_code, 422)

    def test_documentation_covers_minimal_contract(self):
        documentation = (
            Path(__file__).resolve().parents[2]
            / "docs"
            / "api"
            / "SIMPLE_PET_GIF_API.md"
        ).read_text(encoding="utf-8")
        for required in (
            "type=upload",
            "type=query",
            "front=@",
            "left=@",
            "right=@",
            '"eating"',
            '"grooming"',
            "X-API-Key",
            "front_view_not_reliably_frontal",
        ):
            self.assertIn(required, documentation)


if __name__ == "__main__":
    unittest.main()
