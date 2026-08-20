from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

import local_video_api as pipeline
from src.comfy_pet_pipeline.api import uploaded_identity_front_view as front_view


class _PipelineContinued(RuntimeError):
    """Stop the test after proving execution advanced beyond the front audit."""


class UploadedFrontViewDetectorTests(unittest.TestCase):
    class _Cascade:
        def __init__(self, boxes: list[tuple[int, int, int, int]]) -> None:
            self.boxes = np.asarray(boxes, dtype=np.int32).reshape((-1, 4))

        def empty(self) -> bool:
            return False

        def detectMultiScale(self, *_args: object, **_kwargs: object) -> np.ndarray:
            return self.boxes

    def test_qualified_frontal_cat_face_detection_passes_advisory_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            image_path = Path(temporary) / "front.png"
            Image.new("RGB", (400, 500), "white").save(image_path)
            classifier = self._Cascade([(120, 60, 150, 150)])
            with patch.object(
                front_view.cv2,
                "CascadeClassifier",
                return_value=classifier,
            ):
                audit = front_view.audit_uploaded_front_identity_view(image_path)

        self.assertTrue(audit["acceptable"])
        self.assertEqual(audit["status"], "passed_advisory")
        self.assertEqual(audit["issues"], [])
        self.assertEqual(len(audit["qualified_detections"]), 2)
        self.assertFalse(audit["blocking"])

    def test_missing_frontal_cat_face_returns_nonblocking_warning(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            image_path = Path(temporary) / "front.png"
            Image.new("RGB", (400, 500), "white").save(image_path)
            classifier = self._Cascade([])
            with patch.object(
                front_view.cv2,
                "CascadeClassifier",
                return_value=classifier,
            ):
                audit = front_view.audit_uploaded_front_identity_view(image_path)

        self.assertFalse(audit["acceptable"])
        self.assertEqual(audit["status"], "warning_advisory")
        self.assertEqual(
            audit["issues"],
            [pipeline.FRONT_VIEW_NOT_RELIABLY_FRONTAL_STAGE],
        )
        self.assertFalse(audit["blocking"])

    def test_audit_internal_error_is_reported_and_generation_can_continue(self) -> None:
        with patch.object(
            front_view,
            "_audit_uploaded_front_identity_view",
            side_effect=RuntimeError("detector unavailable"),
        ):
            audit = front_view.audit_uploaded_front_identity_view(Path("front.png"))

        self.assertFalse(audit["acceptable"])
        self.assertFalse(audit["blocking"])
        self.assertEqual(audit["status"], "unavailable_advisory")
        self.assertEqual(audit["issues"], ["front_view_audit_internal_error"])


class UploadedFrontViewPipelineTests(unittest.TestCase):
    @staticmethod
    def _front_warning() -> dict[str, object]:
        return {
            "policy": pipeline.UPLOADED_FRONT_VIEW_AUDIT_POLICY,
            "role": "front",
            "blocking": False,
            "acceptable": False,
            "status": "warning_advisory",
            "issues": [pipeline.FRONT_VIEW_NOT_RELIABLY_FRONTAL_STAGE],
        }

    def test_front_warning_is_advisory_when_other_audits_are_advisory(self) -> None:
        integrity = {
            "view_mode": "single_view",
            "view_roles": ["front"],
            "views": {"front": {"input_path": Path("front.png")}},
        }
        with (
            patch.object(pipeline, "GENERATION_VISUAL_AUDITS_BLOCKING", False),
            patch.object(
                pipeline,
                "audit_uploaded_front_identity_view",
                return_value=self._front_warning(),
            ),
        ):
            audit = pipeline.compose_uploaded_identity_view_audit(integrity)

        self.assertFalse(audit["acceptable"])
        self.assertTrue(audit["blocking_acceptable"])
        self.assertEqual(audit["blocking_issues"], [])
        self.assertEqual(
            audit["advisory_issues"],
            [pipeline.FRONT_VIEW_NOT_RELIABLY_FRONTAL_STAGE],
        )
        self.assertEqual(audit["status"], "warning_advisory")
        self.assertFalse(audit["front_view_audit_blocking"])

    def test_duplicate_three_view_is_downgraded_to_front_only(self) -> None:
        integrity = {
            "view_mode": "front_left_right",
            "view_roles": ["front", "left", "right"],
            "views": {
                role: {
                    "input_path": Path(f"{role}.png"),
                    "source_path": Path(f"{role}-source.png"),
                }
                for role in ("front", "left", "right")
            },
        }
        structural_audit = {
            "policy": pipeline.UPLOADED_IDENTITY_VIEW_POLICY,
            "view_mode": "front_left_right",
            "roles": ["front", "left", "right"],
            "view_count": 3,
            "acceptable": False,
            "blocking": False,
            "fail_closed": False,
            "issues": [
                "duplicate_identity_views_front_left",
                "duplicate_identity_views_front_right",
            ],
            "duplicate_pairs": [
                ["front", "left"],
                ["front", "right"],
            ],
            "view_angle_comparisons": {},
            "palettes": {},
            "upper_body_palettes": {},
        }
        front_pass = {
            "policy": pipeline.UPLOADED_FRONT_VIEW_AUDIT_POLICY,
            "role": "front",
            "blocking": False,
            "acceptable": True,
            "status": "passed_advisory",
            "issues": [],
        }
        with (
            patch.object(pipeline, "GENERATION_VISUAL_AUDITS_BLOCKING", False),
            patch.object(
                pipeline,
                "audit_uploaded_front_identity_view",
                return_value=front_pass,
            ),
            patch.object(
                pipeline,
                "audit_uploaded_identity_view_set",
                return_value=structural_audit,
            ),
        ):
            audit = pipeline.compose_uploaded_identity_view_audit(integrity)

        self.assertTrue(audit["acceptable"])
        self.assertTrue(audit["blocking_acceptable"])
        self.assertTrue(audit["downgraded_to_single_view"])
        self.assertEqual(audit["status"], "downgraded_to_single_view")
        self.assertEqual(audit["effective_view_mode"], "single_view")
        self.assertEqual(audit["effective_view_roles"], ["front"])
        self.assertEqual(
            audit["exact_duplicate_pairs"],
            [["front", "left"], ["front", "right"]],
        )
        effective = pipeline.effective_uploaded_identity_integrity(
            integrity,
            audit,
        )
        self.assertEqual(effective["view_mode"], "single_view")
        self.assertEqual(effective["view_roles"], ["front"])
        self.assertEqual(set(effective["views"]), {"front"})
        self.assertEqual(
            effective["submitted_view_roles"],
            ["front", "left", "right"],
        )

    def test_verified_submission_applies_persisted_single_view_downgrade(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job_dir = Path(temporary)
            records: dict[str, dict[str, object]] = {}
            for index, role in enumerate(("front", "left", "right"), start=1):
                input_filename, source_filename = (
                    pipeline.UPLOADED_IDENTITY_VIEW_FILES[role]
                )
                input_path = job_dir / input_filename
                source_path = job_dir / source_filename
                Image.new("RGB", (32, 32), (index * 30, 60, 90)).save(input_path)
                Image.new("RGB", (32, 32), (index * 30, 60, 90)).save(source_path)
                records[role] = {
                    "role": role,
                    "input_filename": input_filename,
                    "source_filename": source_filename,
                    "input_sha256": pipeline.file_sha256(input_path),
                    "source_sha256": pipeline.file_sha256(source_path),
                }
            front = records["front"]
            job = {
                "uploaded_identity_view_mode": "front_left_right",
                "uploaded_identity_view_roles": ["front", "left", "right"],
                "uploaded_identity_view_count": 3,
                "uploaded_identity_view_policy": (
                    pipeline.UPLOADED_IDENTITY_VIEW_POLICY
                ),
                "uploaded_identity_views": records,
                "uploaded_identity_input_sha256": front["input_sha256"],
                "uploaded_identity_source_sha256": front["source_sha256"],
                "uploaded_identity_effective_view_mode": "single_view",
                "uploaded_identity_effective_view_roles": ["front"],
                "uploaded_identity_effective_view_count": 1,
                "uploaded_identity_view_downgraded_to_single_view": True,
            }

            verified = pipeline.verify_uploaded_identity_submission(job, job_dir)

        self.assertEqual(verified["view_mode"], "single_view")
        self.assertEqual(verified["view_roles"], ["front"])
        self.assertEqual(set(verified["views"]), {"front"})
        self.assertEqual(
            verified["submitted_view_roles"],
            ["front", "left", "right"],
        )

    def test_uploaded_action_palette_reference_always_uses_original_source(self) -> None:
        uploaded_source = Path("input_identity_source.png")
        generated_canonical = Path("cartoon_reference.png")

        selected = pipeline.resolve_action_identity_palette_reference(
            uploaded_reference_job=True,
            identity_input_policy=pipeline.POSE_IDENTITY_INPUT_POLICY_CANONICAL,
            uploaded_identity_source=uploaded_source,
            canonical_reference=generated_canonical,
        )

        self.assertEqual(selected, uploaded_source)
        self.assertNotEqual(selected, generated_canonical)

    def test_prepare_records_public_warning_stage_and_continues_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            job_id = "nonfrontal-front"
            (output_root / job_id).mkdir()
            jobs = {
                job_id: {
                    "job_id": job_id,
                    "status": "running",
                    "stage": "generating_cartoon_reference",
                }
            }
            front_warning = self._front_warning()
            identity_audit = {
                "policy": pipeline.UPLOADED_IDENTITY_VIEW_POLICY,
                "status": "warning_advisory",
                "acceptable": False,
                "blocking_acceptable": True,
                "blocking_issues": [],
                "advisory_issues": [
                    pipeline.FRONT_VIEW_NOT_RELIABLY_FRONTAL_STAGE
                ],
                "issues": [pipeline.FRONT_VIEW_NOT_RELIABLY_FRONTAL_STAGE],
                "front_view_audit": front_warning,
                "submitted_view_mode": "single_view",
                "submitted_view_roles": ["front"],
                "submitted_view_count": 1,
                "effective_view_mode": "single_view",
                "effective_view_roles": ["front"],
                "effective_view_count": 1,
                "downgraded_to_single_view": False,
                "downgrade_reason": None,
            }
            with (
                patch.multiple(pipeline, OUTPUT_ROOT=output_root, JOBS=jobs),
                patch.object(
                    pipeline,
                    "verify_uploaded_identity_submission",
                    return_value={
                        "view_mode": "single_view",
                        "view_roles": ["front"],
                        "views": {"front": {}},
                    },
                ),
                patch.object(
                    pipeline,
                    "compose_uploaded_identity_view_audit",
                    return_value=identity_audit,
                ),
                patch.object(
                    pipeline,
                    "merge_image_profile_loras",
                    side_effect=_PipelineContinued("continued after front warning"),
                ),
            ):
                with self.assertRaisesRegex(
                    _PipelineContinued,
                    "continued after front warning",
                ):
                    pipeline.prepare_uploaded_subject_qwen_references(
                        job_id,
                        ["eating", "grooming"],
                    )

                running = pipeline.public_job(jobs[job_id])
                self.assertEqual(running["status"], "running")
                self.assertEqual(
                    running["stage"],
                    pipeline.FRONT_VIEW_NOT_RELIABLY_FRONTAL_STAGE,
                )
                self.assertNotIn("public_stage_override", running)

                pipeline.update_job(
                    job_id,
                    status="completed",
                    stage="completed",
                    progress=100,
                )
                completed = pipeline.public_job(jobs[job_id])
                self.assertEqual(completed["status"], "completed")
                self.assertEqual(
                    completed["stage"],
                    pipeline.FRONT_VIEW_NOT_RELIABLY_FRONTAL_STAGE,
                )

                pipeline.update_job(job_id, status="failed", stage="failed")
                failed = pipeline.public_job(jobs[job_id])
                self.assertEqual(failed["status"], "failed")
                self.assertEqual(failed["stage"], "failed")

        anomaly = jobs[job_id]["result_anomalies"][0]
        self.assertEqual(
            anomaly["code"],
            pipeline.FRONT_VIEW_NOT_RELIABLY_FRONTAL_STAGE,
        )
        self.assertFalse(anomaly["blocking"])

    def test_audit_unavailable_does_not_override_public_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job_id = "front-audit-unavailable"
            output_root = Path(temporary)
            (output_root / job_id).mkdir()
            jobs = {
                job_id: {
                    "job_id": job_id,
                    "status": "running",
                    "stage": "generating_cartoon_reference",
                }
            }
            with patch.multiple(pipeline, OUTPUT_ROOT=output_root, JOBS=jobs):
                pipeline.record_nonblocking_job_anomaly(
                    job_id,
                    code="uploaded_front_view_audit_unavailable",
                    message="front audit unavailable",
                    stage="uploaded_identity_front_view_audit",
                )
                result = pipeline.public_job(jobs[job_id])

        self.assertEqual(result["status"], "running")
        self.assertEqual(result["stage"], "generating_cartoon_reference")
        self.assertNotIn("public_stage_override", jobs[job_id])


if __name__ == "__main__":
    unittest.main()
