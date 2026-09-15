from __future__ import annotations

import json
import queue
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image, ImageDraw, ImageSequence

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


def _mark_generated_candidates_selected(
    job_dir: Path,
    job: dict[str, object],
    actions: list[str],
) -> None:
    job["action_candidate_selection_by_action"] = {
        action: {
            "policy": pipeline.GENERATION_ACTION_CANDIDATE_GATE_POLICY,
            "source_kind": "qwen_generated_action_candidate",
            "attempt_budget": 5,
            "attempts_completed": 5,
            "selected_attempt": 1,
            "selected_penalty": 0.0,
            "identity_palette_acceptable": True,
            "selected_keyframe_sha256": pipeline.file_sha256(
                job_dir / f"pose_{index:02d}_qwen_cutout.png"
            ),
            "selected_cutout_sha256": pipeline.file_sha256(
                job_dir / f"pose_{index:02d}_qwen_cutout.png"
            ),
            "original_upload_used": False,
            "approved_canonical_used_as_output": False,
        }
        for index, action in enumerate(actions, start=1)
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
            _write_transparent_identity(job_dir / "pose_01_qwen_cutout.png")
            jobs = {
                job_id: _accepted_job(
                    job_id,
                    ["standing"],
                    status="running",
                )
            }
            _mark_generated_candidates_selected(
                job_dir,
                jobs[job_id],
                ["standing"],
            )

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
            self.assertEqual(result["fallback_source_role"], "action_qwen_cutout")
            self.assertIn(
                result["transparent_metrics"]["encoding"],
                {"ffmpeg_palette", "pillow_multiframe"},
            )

    def test_visual_completion_candidate_recovers_despite_identity_warning(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            job_id = "visual-completion-identity-warning"
            job_dir = output_root / job_id
            job_dir.mkdir()
            _write_transparent_identity(job_dir / "input.png")
            _write_transparent_identity(job_dir / "pose_01_qwen_cutout.png")
            job = _accepted_job(job_id, ["eating"], status="running")
            _mark_generated_candidates_selected(job_dir, job, ["eating"])
            selection = job["action_candidate_selection_by_action"]["eating"]
            selection.update(
                identity_palette_acceptable=False,
                visual_completion_fallback=True,
                result="selected_generated_completion_fallback",
            )
            job["visual_completion_fallback_enabled"] = True
            jobs = {job_id: job}

            with patch.multiple(
                pipeline,
                OUTPUT_ROOT=output_root,
                JOBS=jobs,
            ):
                recovered = pipeline.recover_subject_replacement_job_with_fallback(
                    job_id,
                    RuntimeError("forced downstream eating recovery"),
                )

            self.assertTrue(recovered)
            self.assertEqual(jobs[job_id]["status"], "completed")
            result = jobs[job_id]["subject_replacement_results"][0]
            self.assertEqual(result["fallback_source_role"], "action_qwen_cutout")
            self.assertFalse(
                job["action_candidate_selection_by_action"]["eating"][
                    "identity_palette_acceptable"
                ]
            )

    def test_identity_warning_without_completion_selection_stays_unrecoverable(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            job_id = "identity-warning-without-completion-contract"
            job_dir = output_root / job_id
            job_dir.mkdir()
            _write_transparent_identity(job_dir / "input.png")
            _write_transparent_identity(job_dir / "pose_01_qwen_cutout.png")
            job = _accepted_job(job_id, ["eating"], status="running")
            _mark_generated_candidates_selected(job_dir, job, ["eating"])
            job["action_candidate_selection_by_action"]["eating"][
                "identity_palette_acceptable"
            ] = False
            jobs = {job_id: job}

            with (
                patch.multiple(
                    pipeline,
                    OUTPUT_ROOT=output_root,
                    JOBS=jobs,
                ),
                self.assertRaisesRegex(
                    RuntimeError,
                    "No readable generated action candidate",
                ),
            ):
                pipeline.recover_subject_replacement_job_with_fallback(
                    job_id,
                    RuntimeError("forced ineligible recovery"),
                )

    def test_persisted_original_upload_path_is_not_a_recovery_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            job_id = "legacy-original-reference-is-forbidden"
            job_dir = output_root / job_id
            job_dir.mkdir()
            _write_transparent_identity(job_dir / "input.png")
            job = _accepted_job(job_id, ["standing"], status="running")
            job["subject_reference_paths"] = {"standing": "input.png"}
            jobs = {job_id: job}

            with (
                patch.multiple(
                    pipeline,
                    OUTPUT_ROOT=output_root,
                    JOBS=jobs,
                ),
                self.assertRaisesRegex(
                    RuntimeError,
                    "No readable generated action candidate",
                ),
            ):
                pipeline.recover_subject_replacement_job_with_fallback(
                    job_id,
                    RuntimeError("forced legacy recovery"),
                )

    def test_action_fallback_does_not_complete_or_replace_sibling_actions(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            job_id = "single-action-isolation"
            job_dir = output_root / job_id
            job_dir.mkdir()
            _write_transparent_identity(job_dir / "input_identity_source.png")
            _write_transparent_identity(job_dir / "pose_01_qwen_cutout.png")
            job = _accepted_job(
                job_id,
                ["toileting", "standing", "eating"],
                status="running",
            )
            job["effective_subject_config"] = {
                "frames": 81,
                "fps": 16,
                "actions": {
                    action: {}
                    for action in ("toileting", "standing", "eating")
                },
            }
            _mark_generated_candidates_selected(job_dir, job, ["toileting"])
            jobs = {job_id: job}

            with patch.multiple(
                pipeline,
                OUTPUT_ROOT=output_root,
                JOBS=jobs,
                TRANSPARENT_GIF_SIZE=TEST_GIF_SIZE,
                TRANSPARENT_GIF_FRAMES=TEST_GIF_FRAMES,
                TRANSPARENT_GIF_FPS=TEST_GIF_FPS,
                TRANSPARENT_GIF_DURATION_SECONDS=(
                    TEST_GIF_DURATION_SECONDS
                ),
                FFMPEG_BIN=(
                    output_root / "missing-ffmpeg-for-pillow-fallback"
                ),
            ):
                result, _source = (
                    pipeline.recover_subject_replacement_action_with_fallback(
                        job_id,
                        job,
                        action="toileting",
                        pose_index=1,
                        error=RuntimeError("forced toileting gate failure"),
                    )
                )

            self.assertEqual(jobs[job_id]["status"], "running")
            self.assertEqual(result["action"], "toileting")
            self.assertTrue(result["degraded"])
            self.assertFalse(result["generation_succeeded"])
            self.assertFalse(result["quality_passed"])
            self.assertEqual(result["fallback_source_role"], "action_qwen_cutout")
            gif_actions = {
                output["action"]
                for output in jobs[job_id]["outputs"]
                if output["type"] == "transparent_gif"
            }
            self.assertEqual(gif_actions, {"toileting"})
            self.assertNotIn("standing", gif_actions)
            self.assertNotIn("eating", gif_actions)

    def test_terminal_recovery_preserves_each_existing_actions_own_timing(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            job_id = "existing-action-timing-isolation"
            job_dir = output_root / job_id
            job_dir.mkdir()
            _write_transparent_identity(job_dir / "input_identity_source.png")
            _write_transparent_identity(job_dir / "pose_01_qwen_cutout.png")
            _write_transparent_identity(job_dir / "pose_02_qwen_cutout.png")
            job = _accepted_job(
                job_id,
                ["eating", "standing"],
                status="running",
            )
            job["effective_subject_config"] = {
                "frames": 81,
                "fps": 16,
                "actions": {
                    "eating": {
                        "output_playback_rate": 0.5,
                        "output_playback_policy": (
                            "restore_original_eating_source_tempo_v1"
                        ),
                    },
                    "standing": {},
                },
            }
            _mark_generated_candidates_selected(
                job_dir,
                job,
                ["eating", "standing"],
            )
            jobs = {job_id: job}

            with patch.multiple(
                pipeline,
                **_fallback_patches(
                    output_root,
                    jobs,
                    queue.Queue(maxsize=8),
                ),
            ):
                for index, action in enumerate(job["actions"], start=1):
                    pipeline.recover_subject_replacement_action_with_fallback(
                        job_id,
                        jobs[job_id],
                        action=action,
                        pose_index=index,
                        error=RuntimeError(f"forced {action} fallback"),
                    )
                # Exercise the branch that reconstructs results from already
                # published sibling outputs.  No prior result may carry timing
                # across actions for it.
                jobs[job_id]["subject_replacement_results"] = []
                recovered = pipeline.recover_subject_replacement_job_with_fallback(
                    job_id,
                    RuntimeError("forced terminal recovery after publication"),
                )

            self.assertTrue(recovered)
            results = {
                result["action"]: result
                for result in jobs[job_id]["subject_replacement_results"]
            }
            self.assertEqual(
                results["eating"]["output_playback"]["transparent_gif"]["fps"],
                2,
            )
            self.assertEqual(
                results["standing"]["output_playback"]["transparent_gif"]["fps"],
                4,
            )
            self.assertTrue(
                results["eating"]["preserved_during_terminal_recovery"]
            )
            self.assertTrue(
                results["standing"]["preserved_during_terminal_recovery"]
            )

    def test_worker_exception_recovers_each_action_from_generated_candidates(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            job_id = "dog-worker-terminal-recovery"
            job_dir = output_root / job_id
            job_dir.mkdir()
            _write_transparent_identity(job_dir / "input.png")
            _write_transparent_identity(job_dir / "input_identity_source.png")
            _write_transparent_identity(job_dir / "pose_01_qwen_cutout.png")
            _write_transparent_identity(job_dir / "pose_02_qwen_cutout.png")
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
            _mark_generated_candidates_selected(
                job_dir,
                jobs[job_id],
                ["eating", "standing"],
            )
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
                self.assertEqual(output["fallback_source_role"], "action_qwen_cutout")
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
            results_by_action = {
                str(result["action"]): result
                for result in final_job["subject_replacement_results"]
            }
            eating_animation = results_by_action["eating"][
                "transparent_metrics"
            ]
            self.assertEqual(
                eating_animation["motion_policy"],
                "bowl_anchored_eating_contact_cycle_v2",
            )
            self.assertFalse(eating_animation["forced_position_jitter"])
            with Image.open(
                job_dir / gif_outputs["eating"]["filename"]
            ) as encoded:
                eating_frames = [
                    np.asarray(frame.convert("RGBA"))[:, :, 3]
                    for frame in ImageSequence.Iterator(encoded)
                ]
            first_alpha = eating_frames[0]
            raised_alpha = eating_frames[len(eating_frames) // 2]
            upper_changed = int(
                np.count_nonzero(
                    first_alpha[: TEST_GIF_SIZE // 2]
                    != raised_alpha[: TEST_GIF_SIZE // 2]
                )
            )
            lower_changed = int(
                np.count_nonzero(
                    first_alpha[TEST_GIF_SIZE * 3 // 4 :]
                    != raised_alpha[TEST_GIF_SIZE * 3 // 4 :]
                )
            )
            self.assertGreater(upper_changed, 8)
            self.assertLessEqual(lower_changed, 4)
            self.assertEqual(
                final_job["result_anomalies"][-1]["code"],
                "degraded_output_fallback",
            )
            self.assertFalse(final_job["degraded_output_upload_identity_anchored"])
            self.assertTrue(final_job["degraded_output_generated_candidate_only"])
            self.assertFalse(final_job["degraded_output_original_upload_used"])
            self.assertEqual(
                final_job["degraded_output_fallback_source_roles"],
                {
                    "eating": "action_qwen_cutout",
                    "standing": "action_qwen_cutout",
                },
            )
            self.assertIn("KeyError", final_job["error"])

    def test_worker_exception_before_candidate_generation_fails_job(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            job_id = "worker-fails-before-generated-candidate"
            job_dir = output_root / job_id
            job_dir.mkdir()
            _write_transparent_identity(job_dir / "input.png")
            _write_transparent_identity(job_dir / "input_identity_source.png")
            jobs = {
                job_id: _accepted_job(
                    job_id,
                    ["eating"],
                    status="running",
                )
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
                    side_effect=RuntimeError("failed before candidate generation"),
                ),
                patch.object(pipeline, "wait_comfy_idle"),
                patch.object(pipeline, "free_comfy_models"),
            ):
                pipeline.worker_loop()

            final_job = jobs[job_id]
            self.assertEqual(final_job["status"], "failed")
            self.assertEqual(final_job["stage"], "failed")
            self.assertEqual(final_job["outputs"], [])
            self.assertNotIn("degraded_output_fallback", final_job)

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

            self.assertEqual(jobs[job_id]["status"], "queued")
            self.assertEqual(jobs[job_id]["stage"], "queued_after_service_restart")
            self.assertEqual(job_queue.qsize(), 1)
            self.assertEqual(jobs[job_id]["outputs"], [])

    def test_restored_queue_overflow_without_generated_candidate_fails(self) -> None:
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

            self.assertEqual(jobs[job_id]["status"], "failed")
            self.assertEqual(jobs[job_id]["stage"], "failed")
            self.assertEqual(jobs[job_id]["outputs"], [])


if __name__ == "__main__":
    unittest.main()
