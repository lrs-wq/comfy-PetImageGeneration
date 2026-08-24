from __future__ import annotations

import json
import queue
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image, ImageDraw

import local_video_api as pipeline


TEST_GIF_SIZE = 64
TEST_GIF_FRAMES = 8
TEST_GIF_FPS = 4
TEST_GIF_DURATION_SECONDS = 2.0


def _write_transparent_identity(path: Path) -> None:
    image = Image.new("RGBA", (96, 96), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((20, 8, 76, 72), fill=(238, 224, 198, 255))
    draw.rectangle((28, 50, 68, 88), fill=(232, 218, 194, 255))
    draw.ellipse((34, 25, 41, 32), fill=(32, 28, 24, 255))
    draw.ellipse((55, 25, 62, 32), fill=(32, 28, 24, 255))
    image.save(path)


def _accepted_job(job_id: str, actions: list[str], *, status: str) -> dict[str, object]:
    return {
        "job_id": job_id,
        "status": status,
        "stage": status,
        "progress": 40 if status == "running" else 0,
        "generation_mode": "subject_replacement",
        "output_format": "transparent_gif",
        "actions": actions,
        "outputs": [],
        "subject_replacement_results": [],
        "reference_generation_status": "running" if status == "running" else "pending",
        "reference_generation_completed": False,
        "callback_enabled": False,
        "cancel_requested": False,
    }


def _fallback_patches(
    output_root: Path,
    jobs: dict[str, dict[str, object]],
    job_queue: queue.Queue[str],
    *,
    max_queue_size: int = 8,
) -> dict[str, object]:
    return {
        "OUTPUT_ROOT": output_root,
        "JOBS": jobs,
        "JOB_QUEUE": job_queue,
        "MAX_QUEUE_SIZE": max_queue_size,
        "TRANSPARENT_GIF_SIZE": TEST_GIF_SIZE,
        "TRANSPARENT_GIF_FRAMES": TEST_GIF_FRAMES,
        "TRANSPARENT_GIF_FPS": TEST_GIF_FPS,
        "TRANSPARENT_GIF_DURATION_SECONDS": TEST_GIF_DURATION_SECONDS,
        "FFMPEG_BIN": output_root / "missing-ffmpeg-for-pillow-fallback",
    }


class PoseAttemptProvenanceRegressionTests(unittest.TestCase):
    def test_incomplete_fallback_record_is_total_and_does_not_raise(self) -> None:
        provenance = pipeline.pose_attempt_provenance(
            {
                "identity_input_policy": "canonical_identity_plus_action_geometry",
            }
        )

        self.assertFalse(provenance["complete"])
        self.assertIsNone(provenance["image1_input_name"])
        self.assertIsNone(provenance["identity_input_name"])
        self.assertIn("image1_input_name", provenance["missing_fields"])
        self.assertIn("identity_input_name", provenance["missing_fields"])

    def test_single_image_grooming_workflow_has_no_false_missing_geometry(self) -> None:
        canonical = "local_video_api/job/cartoon_reference.png"
        provenance = pipeline.pose_attempt_provenance(
            {
                "identity_input_policy": "canonical_character_image1",
                "image1_input_name": Path(canonical),
                "sampler_latent_input_name": canonical,
                "identity_input_name": canonical,
                "workflow": {
                    "1": {
                        "class_type": "LoadImage",
                        "inputs": {"image": canonical},
                    }
                },
            }
        )

        self.assertTrue(provenance["complete"])
        self.assertEqual(provenance["missing_fields"], [])
        self.assertEqual(provenance["unmapped_workflow_input_names"], [])
        self.assertIsNone(provenance["image2_input_name"])
        self.assertIsNone(provenance["geometry_input_name"])

    def test_fallback_provenance_recovers_sources_from_settings_and_workflow(
        self,
    ) -> None:
        canonical = "local_video_api/job/cartoon_reference.png"
        guide = "local_video_api/job/eating_qwen_pose_guide.png"
        provenance = pipeline.pose_attempt_provenance(
            {
                "image1_input_name": None,
                "image2_input_name": None,
                "settings": {
                    "image1_input_name": canonical,
                    "image2_input_name": guide,
                    "sampler_latent_input_name": canonical,
                    "identity_input_name": canonical,
                    "geometry_input_name": guide,
                },
                "workflow": {
                    "1": {
                        "class_type": "LoadImage",
                        "inputs": {"image": canonical},
                    },
                    "17": {
                        "class_type": "LoadImage",
                        "inputs": {"image": guide},
                    },
                },
            }
        )

        self.assertTrue(provenance["complete"])
        self.assertEqual(provenance["image1_input_name"], canonical)
        self.assertEqual(provenance["image2_input_name"], guide)
        self.assertEqual(provenance["geometry_input_name"], guide)
        self.assertEqual(
            provenance["field_sources"]["image1_input_name"],
            "attempt_settings",
        )
        self.assertEqual(provenance["workflow_load_image_names"], [canonical, guide])

    def test_all_pose_workflow_sources_are_preserved(self) -> None:
        record = {
            "identity_input_policy": "canonical_identity_plus_action_geometry",
            "image1_input_name": "canonical.png",
            "image2_input_name": "eating-guide.png",
            "image3_input_name": "right-view.png",
            "sampler_latent_input_name": "sampler-canvas.png",
            "sampler_noise_mask_input_name": "noise-mask.png",
            "identity_input_name": "canonical.png",
            "geometry_input_name": "eating-guide.png",
        }

        provenance = pipeline.pose_attempt_provenance(record)

        self.assertTrue(provenance["complete"])
        self.assertEqual(provenance["missing_fields"], [])
        for field, expected in record.items():
            self.assertEqual(provenance[field], expected)


class RecoverableOutputGuaranteeTests(unittest.TestCase):
    def assert_gif_contract(
        self,
        path: Path,
        *,
        expected_duration_ms: int = 2000,
    ) -> None:
        self.assertTrue(path.is_file(), path)
        with Image.open(path) as animation:
            self.assertEqual(animation.size, (TEST_GIF_SIZE, TEST_GIF_SIZE))
            self.assertEqual(animation.n_frames, TEST_GIF_FRAMES)
            self.assertEqual(animation.info.get("loop"), 0)
            observed_duration_ms = 0
            transparent = False
            visible = False
            for index in range(animation.n_frames):
                animation.seek(index)
                observed_duration_ms += int(animation.info.get("duration") or 0)
                alpha_extrema = animation.convert("RGBA").getchannel("A").getextrema()
                transparent = transparent or alpha_extrema[0] == 0
                visible = visible or alpha_extrema[1] >= 64
            self.assertEqual(observed_duration_ms, expected_duration_ms)
            self.assertTrue(transparent)
            self.assertTrue(visible)

    def test_public_fallback_contract_is_512_square_100_frames_five_seconds(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            job_id = "public-terminal-recovery-contract"
            job_dir = output_root / job_id
            job_dir.mkdir()
            _write_transparent_identity(job_dir / "input.png")
            jobs = {
                job_id: _accepted_job(
                    job_id,
                    ["standing"],
                    status="running",
                )
            }

            with patch.multiple(
                pipeline,
                OUTPUT_ROOT=output_root,
                JOBS=jobs,
            ):
                recovered = pipeline.recover_subject_replacement_job_with_fallback(
                    job_id,
                    RuntimeError("forced public contract recovery"),
                )

            self.assertTrue(recovered)
            output = next(
                item
                for item in jobs[job_id]["outputs"]
                if item["type"] == "transparent_gif"
            )
            gif_path = job_dir / str(output["filename"])
            with Image.open(gif_path) as animation:
                self.assertEqual(animation.size, (512, 512))
                self.assertEqual(animation.n_frames, 100)
                self.assertEqual(animation.info.get("loop"), 0)
                duration_ms = 0
                for index in range(animation.n_frames):
                    animation.seek(index)
                    duration_ms += int(animation.info.get("duration") or 0)
                self.assertEqual(duration_ms, 5000)
            result = jobs[job_id]["subject_replacement_results"][0]
            self.assertIn(
                result["transparent_metrics"]["encoding"],
                {"ffmpeg_palette", "pillow_multiframe"},
            )

    def test_worker_exception_returns_every_dog_action_gif(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            job_id = "dog-worker-terminal-recovery"
            job_dir = output_root / job_id
            job_dir.mkdir()
            _write_transparent_identity(job_dir / "input.png")
            _write_transparent_identity(job_dir / "input_identity_source.png")
            profile_reference = output_root / "small-dog-profile-reference.png"
            Image.new("RGBA", (96, 96), (245, 245, 245, 255)).save(
                profile_reference
            )
            jobs = {
                job_id: _accepted_job(
                    job_id,
                    ["eating", "standing"],
                    status="running",
                )
            }
            jobs[job_id]["effective_subject_config"] = {
                "frames": 81,
                "fps": 16,
                "actions": {
                    "eating": {
                        "reference": str(profile_reference),
                        "output_playback_rate": 0.5,
                        "output_playback_policy": (
                            "restore_original_eating_source_tempo_v1"
                        ),
                    },
                    "standing": {"reference": str(profile_reference)},
                },
            }
            job_queue: queue.Queue[str] = queue.Queue(maxsize=8)
            job_queue.put_nowait(job_id)
            stop_event = Mock()
            stop_event.is_set.side_effect = [False, True]

            with (
                patch.multiple(
                    pipeline,
                    **_fallback_patches(output_root, jobs, job_queue),
                    STOP_EVENT=stop_event,
                ),
                patch.object(
                    pipeline,
                    "run_job",
                    side_effect=KeyError("image1_input_name"),
                ),
                patch.object(pipeline, "wait_comfy_idle"),
                patch.object(pipeline, "free_comfy_models"),
            ):
                pipeline.worker_loop()

            final_job = jobs[job_id]
            self.assertEqual(final_job["status"], "completed")
            self.assertEqual(final_job["stage"], "completed_with_degraded_output")
            self.assertTrue(final_job["transparent_outputs_complete"])
            self.assertEqual(
                final_job["degraded_output_fallback_policy"],
                pipeline.RECOVERABLE_OUTPUT_FALLBACK_POLICY,
            )
            gif_outputs = {
                str(output["action"]): output
                for output in final_job["outputs"]
                if output["type"] == "transparent_gif"
            }
            self.assertEqual(set(gif_outputs), {"eating", "standing"})
            for action, output in gif_outputs.items():
                self.assertTrue(output["degraded"])
                self.assertEqual(output["fallback_source_role"], "original_upload")
                self.assert_gif_contract(
                    job_dir / str(output["filename"]),
                    expected_duration_ms=4000 if action == "eating" else 2000,
                )
            self.assertEqual(gif_outputs["eating"]["fps"], 2)
            self.assertEqual(gif_outputs["eating"]["duration_seconds"], 4.0)
            self.assertEqual(gif_outputs["standing"]["fps"], 4)
            self.assertEqual(gif_outputs["standing"]["duration_seconds"], 2.0)
            self.assertEqual(
                {result["action"] for result in final_job["subject_replacement_results"]},
                {"eating", "standing"},
            )
            self.assertEqual(
                final_job["result_anomalies"][-1]["code"],
                "degraded_output_fallback",
            )
            self.assertTrue(final_job["degraded_output_upload_identity_anchored"])
            self.assertEqual(
                final_job["degraded_output_fallback_source_roles"],
                {"eating": "original_upload", "standing": "original_upload"},
            )
            self.assertIn("KeyError", final_job["error"])

    def test_running_job_is_recovered_after_service_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            job_id = "restart-terminal-recovery"
            job_dir = output_root / job_id
            job_dir.mkdir()
            _write_transparent_identity(job_dir / "input.png")
            initial = _accepted_job(job_id, ["eating"], status="running")
            (job_dir / "status.json").write_text(
                json.dumps(initial),
                encoding="utf-8",
            )
            jobs: dict[str, dict[str, object]] = {}
            job_queue: queue.Queue[str] = queue.Queue(maxsize=8)

            with patch.multiple(
                pipeline,
                **_fallback_patches(output_root, jobs, job_queue),
            ):
                pipeline.restore_jobs()

            self.assertEqual(jobs[job_id]["status"], "completed")
            self.assertEqual(job_queue.qsize(), 0)
            output = next(
                item
                for item in jobs[job_id]["outputs"]
                if item["type"] == "transparent_gif"
            )
            self.assert_gif_contract(job_dir / str(output["filename"]))

    def test_restored_queue_overflow_returns_a_gif_instead_of_failed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            job_id = "queue-overflow-terminal-recovery"
            job_dir = output_root / job_id
            job_dir.mkdir()
            _write_transparent_identity(job_dir / "input.png")
            initial = _accepted_job(job_id, ["grooming"], status="queued")
            (job_dir / "status.json").write_text(
                json.dumps(initial),
                encoding="utf-8",
            )
            jobs: dict[str, dict[str, object]] = {}
            job_queue: queue.Queue[str] = queue.Queue()

            with patch.multiple(
                pipeline,
                **_fallback_patches(
                    output_root,
                    jobs,
                    job_queue,
                    max_queue_size=0,
                ),
            ):
                pipeline.restore_jobs()

            self.assertEqual(jobs[job_id]["status"], "completed")
            self.assertNotEqual(jobs[job_id]["stage"], "failed")
            output = next(
                item
                for item in jobs[job_id]["outputs"]
                if item["type"] == "transparent_gif"
            )
            self.assert_gif_contract(job_dir / str(output["filename"]))


if __name__ == "__main__":
    unittest.main()
