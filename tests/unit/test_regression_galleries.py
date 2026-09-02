from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests
from PIL import Image

from src.comfy_pet_pipeline.regression import historical_gallery
from src.comfy_pet_pipeline.regression import live_stability_gallery as live


def image_response() -> requests.Response:
    buffer = io.BytesIO()
    Image.new("RGB", (12, 9), (120, 80, 40)).save(buffer, format="GIF")
    response = requests.Response()
    response.status_code = 200
    response._content = buffer.getvalue()
    response.url = "http://example.test/output.gif"
    return response


class HistoricalGalleryContractTest(unittest.TestCase):
    def run_with_results(
        self,
        gate_results: list[dict[str, object]],
        canary_results: list[dict[str, object]],
    ) -> dict[str, object]:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_dir = root / "report"
            with (
                patch.object(historical_gallery, "load_manifest", return_value={"name": "test"}),
                patch.object(historical_gallery, "resolve_artifacts", return_value={}),
                patch.object(historical_gallery, "run_gate_cases", return_value=gate_results),
                patch.object(
                    historical_gallery,
                    "validate_canaries",
                    return_value=canary_results,
                ),
                patch.object(historical_gallery, "render_sheet"),
                patch.object(historical_gallery, "write_html_report"),
                patch.object(historical_gallery, "update_latest_pointer"),
                patch.object(historical_gallery, "git_revision", return_value="abc123"),
            ):
                report, _ = historical_gallery.run_gallery(
                    root / "manifest.json",
                    root,
                    report_root=root / "reports",
                    output_dir=output_dir,
                )
            return report

    def test_quality_drift_is_non_blocking(self) -> None:
        report = self.run_with_results(
            [{"id": "quality", "passed": False}],
            [{"id": "input", "passed": True}],
        )
        self.assertEqual(report["summary"]["failed"], 0)
        self.assertEqual(report["summary"]["diagnostic_drifted"], 1)

    def test_input_integrity_remains_blocking(self) -> None:
        report = self.run_with_results(
            [{"id": "quality", "passed": True}],
            [{"id": "input", "passed": False}],
        )
        self.assertEqual(report["summary"]["failed"], 1)
        self.assertEqual(report["summary"]["diagnostic_drifted"], 0)


class LiveStabilityGalleryContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = {
            "canary_cases": [
                {
                    "id": "cat",
                    "actions": ["eating", "grooming"],
                    "smoke_actions": ["grooming"],
                },
                {
                    "id": "dog",
                    "actions": ["eating"],
                },
                {
                    "id": "full-only",
                    "actions": ["standing"],
                    "smoke_actions": ["standing"],
                },
            ]
        }

    def test_smoke_and_explicit_case_selection(self) -> None:
        smoke = live.select_cases(self.manifest, suite="smoke")
        self.assertEqual([item["id"] for item in smoke], ["cat", "full-only"])
        self.assertEqual(smoke[0]["selected_actions"], ["grooming"])

        explicit = live.select_cases(
            self.manifest,
            suite="smoke",
            requested_ids=["dog"],
        )
        self.assertEqual([item["id"] for item in explicit], ["dog"])
        self.assertEqual(explicit[0]["selected_actions"], ["eating"])

    def test_completion_contract_ignores_visual_quality_fields(self) -> None:
        session = Mock()
        session.get.return_value = image_response()
        case = {"id": "cat", "selected_actions": ["grooming"]}
        job = {
            "job_id": "job-1",
            "status": "completed",
            "stage": "completed",
            "actions": ["grooming"],
            "identity_quality": {"acceptable": False},
            "outputs": [
                {
                    "action": "grooming",
                    "type": "transparent_gif",
                    "filename": "grooming.gif",
                    "primary": True,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            outputs, issues = live.validate_completed_job(
                session,
                "http://example.test/api/local-video",
                {"X-API-Key": "secret"},
                case,
                job,
                Path(temporary),
            )
        self.assertEqual(issues, [])
        self.assertEqual(outputs[0]["action"], "grooming")
        self.assertTrue(outputs[0]["readable"])

    def test_missing_action_output_fails_completion_contract(self) -> None:
        case = {"id": "dog", "selected_actions": ["eating"]}
        job = {
            "job_id": "job-2",
            "status": "completed",
            "actions": ["eating"],
            "outputs": [],
        }
        with tempfile.TemporaryDirectory() as temporary:
            outputs, issues = live.validate_completed_job(
                Mock(),
                "http://example.test/api/local-video",
                {},
                case,
                job,
                Path(temporary),
            )
        self.assertEqual(outputs, [])
        self.assertIn("no primary animation output", issues[0])

    def test_visual_diagnostics_keep_topology_without_becoming_a_gate(self) -> None:
        diagnostics = live.job_visual_diagnostics(
            {
                "qwen_first_frame_best_effort_attempt": 4,
                "grooming_advisory_minimum_gate_status": "degraded_passed",
                "grooming_foreleg_topology_status": "advisory_drift",
                "grooming_foreleg_topology_policy": "separator-v1",
                "grooming_foreleg_topology_blocks_completion": False,
                "grooming_foreleg_topology_selected": {
                    "grooming": {"acceptable": False, "penalty": 42.0}
                },
                "scail_video_temporal_identity_status": {
                    "grooming": "best_effort"
                },
                "scail_video_attempts": {
                    "grooming": [
                        {"attempt": 1, "selected": False},
                        {"attempt": 2, "selected": True},
                    ]
                },
            }
        )

        self.assertEqual(diagnostics["qwen_first_frame_selected_attempt"], 4)
        self.assertEqual(
            diagnostics["grooming_foreleg_topology_status"],
            "advisory_drift",
        )
        self.assertFalse(
            diagnostics["grooming_foreleg_topology_blocks_completion"]
        )
        self.assertEqual(diagnostics["scail_grooming_selected_attempt"], 2)


if __name__ == "__main__":
    unittest.main()
