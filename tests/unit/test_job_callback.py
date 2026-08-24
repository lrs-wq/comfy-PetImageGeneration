import json
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import requests

from src.comfy_pet_pipeline.api.job_callback import (
    CALLBACK_DEAD_LETTER_DIRECTORY,
    CALLBACK_OUTBOX_DIRECTORY,
    CALLBACK_STATE_FILENAME,
    JobCallbackReporter,
    build_callback_payload,
)


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {
            "code": 200,
            "message": "success",
            "data": None,
        }
        self.text = text

    def json(self):
        return self._payload


def wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class JobCallbackPayloadTests(unittest.TestCase):
    def test_queued_and_waiting_stages_report_zero_progress(self):
        queued = build_callback_payload(
            {
                "job_id": "task-queued",
                "status": "queued",
                "stage": "queued",
                "progress": 91,
            },
            "http://39.106.120.253:8051",
        )
        waiting = build_callback_payload(
            {
                "job_id": "task-waiting",
                "status": "running",
                "stage": "waiting_for_gpu",
                "progress": 1,
            },
            "http://39.106.120.253:8051",
        )

        self.assertEqual(queued["progress"], 0.0)
        self.assertEqual(waiting["progress"], 0.0)
        self.assertEqual(queued["status"], "queued")
        self.assertEqual(waiting["status"], "running")
        self.assertNotIn("gifs", queued)
        self.assertNotIn("gifs", waiting)

    def test_completed_callback_contains_primary_eating_and_grooming_gifs(self):
        payload = build_callback_payload(
            {
                "job_id": "task with spaces",
                "status": "completed",
                "stage": "completed",
                "progress": 99,
                "outputs": [
                    {
                        "type": "transparent_gif",
                        "action": "eating",
                        "filename": "eating-secondary.gif",
                        "primary": False,
                    },
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
                    {
                        "type": "video",
                        "action": "eating",
                        "filename": "internal.mp4",
                    },
                ],
            },
            "http://39.106.120.253:8051/",
        )

        self.assertEqual(payload["progress"], 100.0)
        self.assertEqual(
            payload["gifs"],
            {
                "eating": (
                    "http://39.106.120.253:8051/file/"
                    "task%20with%20spaces/eating.gif"
                ),
                "grooming": (
                    "http://39.106.120.253:8051/file/"
                    "task%20with%20spaces/grooming.gif"
                ),
            },
        )

    def test_dog_callback_maps_standing_to_legacy_grooming_field(self):
        payload = build_callback_payload(
            {
                "job_id": "dog-task",
                "species": "dog",
                "status": "completed",
                "stage": "completed",
                "outputs": [
                    {
                        "type": "transparent_gif",
                        "action": "eating",
                        "filename": "eating.gif",
                    },
                    {
                        "type": "transparent_gif",
                        "action": "standing",
                        "filename": "standing.gif",
                    },
                ],
            },
            "http://39.106.120.253:8051",
        )

        self.assertEqual(
            payload["gifs"],
            {
                "eating": "http://39.106.120.253:8051/file/dog-task/eating.gif",
                "standing": "http://39.106.120.253:8051/file/dog-task/standing.gif",
                "grooming": "http://39.106.120.253:8051/file/dog-task/standing.gif",
            },
        )


class JobCallbackReporterTests(unittest.TestCase):
    def make_reporter(self, root, post, retry_delays=(0.01,)):
        return JobCallbackReporter(
            callback_url="https://callback.example.test/api/open/gifCallback",
            api_key="callback-secret",
            public_base_url="http://39.106.120.253:8051",
            output_root=Path(root),
            request_post=post,
            retry_delays=retry_delays,
        )

    def test_same_stage_progress_refresh_is_deduplicated(self):
        with TemporaryDirectory() as temporary:
            reporter = self.make_reporter(temporary, lambda *args, **kwargs: None)
            created = {
                "job_id": "task-dedupe",
                "status": "queued",
                "stage": "queued",
                "progress": 0,
            }
            self.assertTrue(reporter.enqueue(created, force=True))
            self.assertFalse(reporter.enqueue({**created, "progress": 7}))
            self.assertTrue(
                reporter.enqueue(
                    {
                        **created,
                        "status": "running",
                        "stage": "generating_cartoon_reference",
                        "progress": 15,
                    }
                )
            )
            outbox = Path(temporary) / "task-dedupe" / CALLBACK_OUTBOX_DIRECTORY
            self.assertEqual(len(list(outbox.glob("*.json"))), 2)

    def test_worker_posts_authenticated_json_and_records_delivery(self):
        captured = []

        def fake_post(url, **kwargs):
            captured.append((url, kwargs))
            return FakeResponse()

        with TemporaryDirectory() as temporary:
            reporter = self.make_reporter(temporary, fake_post)
            reporter.start()
            try:
                reporter.enqueue(
                    {
                        "job_id": "task-delivery",
                        "status": "queued",
                        "stage": "queued",
                        "progress": 50,
                    },
                    force=True,
                )
                outbox = Path(temporary) / "task-delivery" / CALLBACK_OUTBOX_DIRECTORY
                self.assertTrue(
                    wait_until(lambda: bool(captured) and not list(outbox.glob("*.json")))
                )
            finally:
                reporter.stop()

            self.assertEqual(
                captured[0][0],
                "https://callback.example.test/api/open/gifCallback",
            )
            self.assertEqual(
                captured[0][1]["headers"]["X-API-Key"],
                "callback-secret",
            )
            self.assertEqual(
                captured[0][1]["json"],
                {
                    "key": "task-delivery",
                    "status": "queued",
                    "progress": 0.0,
                    "stage": "queued",
                },
            )
            state = json.loads(
                (
                    Path(temporary)
                    / "task-delivery"
                    / CALLBACK_STATE_FILENAME
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                state["last_delivered_signature"],
                state["last_enqueued_signature"],
            )

    def test_transient_network_failure_is_retried(self):
        attempts = []

        def flaky_post(url, **kwargs):
            attempts.append((url, kwargs))
            if len(attempts) == 1:
                raise requests.ConnectionError("temporary outage")
            return FakeResponse()

        with TemporaryDirectory() as temporary:
            reporter = self.make_reporter(temporary, flaky_post)
            reporter.start()
            try:
                reporter.enqueue(
                    {
                        "job_id": "task-retry",
                        "status": "running",
                        "stage": "generating_cartoon_keyframes",
                        "progress": 26.5,
                    },
                    force=True,
                )
                outbox = Path(temporary) / "task-retry" / CALLBACK_OUTBOX_DIRECTORY
                self.assertTrue(
                    wait_until(
                        lambda: len(attempts) == 2 and not list(outbox.glob("*.json"))
                    )
                )
            finally:
                reporter.stop()

    def test_permanent_rejection_moves_event_to_dead_letter(self):
        with TemporaryDirectory() as temporary:
            reporter = self.make_reporter(
                temporary,
                lambda *args, **kwargs: FakeResponse(
                    status_code=422,
                    payload={"code": 422, "message": "invalid payload"},
                    text="invalid payload",
                ),
            )
            reporter.start()
            try:
                reporter.enqueue(
                    {
                        "job_id": "task-rejected",
                        "status": "queued",
                        "stage": "queued",
                        "progress": 0,
                    },
                    force=True,
                )
                dead_letter = (
                    Path(temporary)
                    / "task-rejected"
                    / CALLBACK_DEAD_LETTER_DIRECTORY
                )
                self.assertTrue(wait_until(lambda: len(list(dead_letter.glob("*.json"))) == 1))
            finally:
                reporter.stop()

    def test_authentication_rejection_retains_event_for_corrected_credentials(self):
        with TemporaryDirectory() as temporary:
            reporter = self.make_reporter(
                temporary,
                lambda *args, **kwargs: FakeResponse(
                    status_code=401,
                    payload={"code": 401, "message": "invalid key"},
                    text="invalid key",
                ),
            )
            reporter.start()
            try:
                reporter.enqueue(
                    {
                        "job_id": "task-auth-rejected",
                        "status": "queued",
                        "stage": "queued",
                        "progress": 0,
                    },
                    force=True,
                )
                job_directory = Path(temporary) / "task-auth-rejected"
                outbox = job_directory / CALLBACK_OUTBOX_DIRECTORY
                state_path = job_directory / CALLBACK_STATE_FILENAME
                self.assertTrue(
                    wait_until(
                        lambda: state_path.is_file()
                        and json.loads(state_path.read_text(encoding="utf-8")).get(
                            "delivery_deferred_until_restart"
                        )
                        is True
                    )
                )
                self.assertEqual(len(list(outbox.glob("*.json"))), 1)
                self.assertFalse(
                    list(
                        (
                            job_directory / CALLBACK_DEAD_LETTER_DIRECTORY
                        ).glob("*.json")
                    )
                )
            finally:
                reporter.stop()


if __name__ == "__main__":
    unittest.main()
