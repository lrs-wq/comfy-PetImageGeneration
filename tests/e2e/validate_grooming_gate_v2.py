#!/usr/bin/env python3
"""Offline regression for the production grooming-gate false positive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import local_video_api as api
from grooming_pose_gate import (
    GATE_VERSION,
    POLICY,
    evaluate_grooming_pose,
)


DEFAULT_JOB_ID = "20260804T032056Z-66bb7b39c6"
DEFAULT_OUTPUT_ROOT = Path("/data/comfy-pet-pipeline/outputs/local_video_api")
DEFAULT_INPUT_ROOT = Path("/data/comfy-pet-pipeline/comfyui/input/local_video_api")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", default=DEFAULT_JOB_ID)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    args = parser.parse_args()

    job_root = args.output_root / args.job_id
    input_root = args.input_root / args.job_id
    guide = input_root / "grooming_qwen_pose_guide.png"
    metadata = input_root / "grooming_qwen_pose_guide_metadata.json"
    status = json.loads((job_root / "status.json").read_text(encoding="utf-8"))

    require(POLICY == "main_case_grooming_raised_paw_cv_v2", POLICY)
    require(GATE_VERSION == "2.0", GATE_VERSION)
    results: list[dict[str, Any]] = []
    for attempt in range(1, 4):
        candidate = job_root / f"pose_01_qwen_reference_attempt_{attempt:02d}.png"
        result = evaluate_grooming_pose(
            candidate,
            guide,
            geometry_metadata=metadata,
        )
        appearance = result["metrics"]["appearance_consistency"]
        require(result["available"] is True, f"attempt {attempt}: gate unavailable")
        require(
            result["registration"]["appearance_sampling_policy"]
            == "terminal_paw_bands_v2",
            f"attempt {attempt}: wrong sampling policy",
        )
        require(
            appearance["sampling"]["band_height_pixels"] == 56,
            f"attempt {attempt}: terminal band drifted",
        )
        require(
            "raised_paw_color_inconsistent_with_planted_paw"
            not in result["issues"],
            f"attempt {attempt}: patterned foreleg still rejected as paw color",
        )
        require(
            result["acceptable"] is True,
            f"attempt {attempt}: grooming gate still rejected {result['issues']}",
        )
        results.append(
            {
                "attempt": attempt,
                "acceptable": result["acceptable"],
                "issues": result["issues"],
                "paw_muzzle_gap": result["metrics"]["paw_muzzle_contact"][
                    "gap_fraction_of_bbox_height"
                ],
                "lab_median_delta": appearance["lab_median_delta"],
                "histogram_distance": appearance[
                    "lab_histogram_bhattacharyya"
                ]["mean"],
                "texture_ratio": appearance["texture_gradient_p90_ratio"],
                "raised_terminal_box": appearance["sampling"]["raised"],
                "planted_terminal_box": appearance["sampling"]["planted"],
            }
        )

    first_gap = float(results[0]["paw_muzzle_gap"])
    require(
        0.035 < first_gap <= 0.040,
        f"regression fixture no longer exercises the contact tolerance: {first_gap}",
    )
    first_candidate = job_root / "pose_01_qwen_reference_attempt_01.png"
    surface = api.evaluate_main_case_fur_surface_candidate(
        first_candidate,
        action="grooming",
        alpha_provenance="rembg_u2net_fallback",
        effective_subject_config=status["effective_subject_config"],
    )
    require(surface.get("evaluated") is True, "surface gate was not evaluated")
    require(
        surface.get("acceptable") is True,
        f"first candidate surface gate rejected: {surface.get('issues')}",
    )

    report = {
        "passed": True,
        "policy": POLICY,
        "version": GATE_VERSION,
        "job_id": args.job_id,
        "attempts": results,
        "selected_attempt": 1,
        "selected_surface_acceptable": surface["acceptable"],
        "selected_surface_issues": surface["issues"],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
