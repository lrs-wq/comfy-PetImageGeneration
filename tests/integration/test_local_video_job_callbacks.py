import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import local_video_api as pipeline


class LocalVideoJobCallbackHookTests(unittest.TestCase):
    def setUp(self):
        self.job_ids = []

    def tearDown(self):
        with pipeline.JOBS_LOCK:
            for job_id in self.job_ids:
                pipeline.JOBS.pop(job_id, None)

    def test_creation_and_stage_change_use_central_callback_reporter(self):
        job_id = "callback-hook-test"
        self.job_ids.append(job_id)
        job = {
            "job_id": job_id,
            "callback_enabled": True,
            "callback_source": "pet_gif_api",
            "status": "queued",
            "stage": "queued",
            "progress": 0,
            "outputs": [],
        }

        with (
            TemporaryDirectory() as temporary,
            patch.object(pipeline, "OUTPUT_ROOT", Path(temporary)),
            patch.object(
                pipeline.JOB_CALLBACK_REPORTER,
                "enqueue",
                return_value=True,
            ) as enqueue,
        ):
            with pipeline.JOBS_LOCK:
                pipeline.JOBS[job_id] = dict(job)
            pipeline.persist_job(job_id)
            updated = pipeline.update_job(
                job_id,
                status="running",
                stage="waiting_for_gpu",
                progress=1,
            )

        self.assertEqual(enqueue.call_count, 2)
        initial_call, update_call = enqueue.call_args_list
        self.assertEqual(initial_call.args[0]["stage"], "queued")
        self.assertEqual(initial_call.kwargs, {"force": True})
        self.assertEqual(update_call.args[0], updated)
        self.assertEqual(update_call.kwargs, {"force": False})

    def test_callback_enqueue_exception_does_not_fail_job_update(self):
        job_id = "callback-hook-error-test"
        self.job_ids.append(job_id)
        with (
            TemporaryDirectory() as temporary,
            patch.object(pipeline, "OUTPUT_ROOT", Path(temporary)),
            patch.object(
                pipeline.JOB_CALLBACK_REPORTER,
                "enqueue",
                side_effect=OSError("outbox unavailable"),
            ),
        ):
            with pipeline.JOBS_LOCK:
                pipeline.JOBS[job_id] = {
                    "job_id": job_id,
                    "callback_enabled": True,
                    "callback_source": "pet_gif_api",
                    "status": "queued",
                    "stage": "queued",
                    "progress": 0,
                }
            snapshot = pipeline.update_job(
                job_id,
                status="running",
                stage="generating_cartoon_reference",
                progress=15,
            )

        self.assertEqual(snapshot["status"], "running")
        self.assertEqual(snapshot["stage"], "generating_cartoon_reference")

    def test_test_bench_job_does_not_enqueue_callback(self):
        job_id = "test-bench-no-callback"
        self.job_ids.append(job_id)
        with (
            TemporaryDirectory() as temporary,
            patch.object(pipeline, "OUTPUT_ROOT", Path(temporary)),
            patch.object(
                pipeline.JOB_CALLBACK_REPORTER,
                "enqueue",
                return_value=True,
            ) as enqueue,
        ):
            with pipeline.JOBS_LOCK:
                pipeline.JOBS[job_id] = {
                    "job_id": job_id,
                    "status": "queued",
                    "stage": "queued",
                    "progress": 0,
                }
            pipeline.persist_job(job_id)
            pipeline.update_job(
                job_id,
                status="running",
                stage="waiting_for_gpu",
                progress=1,
            )

        enqueue.assert_not_called()


if __name__ == "__main__":
    unittest.main()
