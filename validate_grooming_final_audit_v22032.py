#!/usr/bin/env python3
"""Replay the v2.20.32 grooming final-video audit on an existing job."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import local_video_api as pipeline


DEFAULT_JOB_DIR = Path(
    "/data/comfy-pet-pipeline/outputs/local_video_api/"
    "20260804T040338Z-60eed99419"
)


def require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} is missing or invalid")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", type=Path, default=DEFAULT_JOB_DIR)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    job_dir = args.job_dir.resolve()
    job = require_dict(
        json.loads((job_dir / "status.json").read_text(encoding="utf-8")),
        "job status",
    )
    manifest = require_dict(
        json.loads(
            (job_dir / "grooming_prepare_manifest.json").read_text(
                encoding="utf-8"
            )
        ),
        "grooming prepare manifest",
    )
    effective_config = require_dict(
        job.get("effective_subject_config"),
        "effective subject config",
    )
    action_config = require_dict(
        require_dict(effective_config.get("actions"), "actions").get(
            "grooming"
        ),
        "grooming action config",
    )
    input_files = require_dict(manifest.get("input_files"), "manifest inputs")
    input_hashes = require_dict(
        manifest.get("input_files_sha256"),
        "manifest input hashes",
    )
    selected_metrics = require_dict(
        job.get("fur_surface_gate_selected_metrics"),
        "selected surface metrics",
    )
    selected_gate = require_dict(
        selected_metrics.get("grooming"),
        "selected grooming surface gate",
    )
    reference_path = Path(str(action_config.get("reference") or ""))
    pose_mask_path = Path(str(input_files.get("pose_mask_video") or ""))
    reference_mask_path = Path(str(input_files.get("reference_mask") or ""))
    reference_policy = str(manifest.get("reference_policy") or "")
    pose_metadata = require_dict(
        manifest.get("pose_metadata"),
        "pose metadata",
    )

    scope = pipeline.select_final_video_fur_surface_comparison_scope(
        action="grooming",
        action_config=action_config,
        output_format=str(job.get("output_format") or ""),
        reference_policy=reference_policy,
        pose_metadata=pose_metadata,
    )
    result = pipeline.evaluate_final_video_fur_surface(
        job_dir / "grooming_subject_replacement.mp4",
        action="grooming",
        effective_subject_config=effective_config,
        comparison_scope=scope,
        validated_action_reference_path=reference_path,
        validated_action_reference_sha256=str(
            action_config.get("reference_sha256") or ""
        ),
        validated_action_reference_gate=selected_gate,
        reference_policy=reference_policy,
        pose_identity_mask_video_path=pose_mask_path,
        pose_identity_mask_video_sha256=str(
            input_hashes.get("pose_mask_video") or ""
        ),
        reference_identity_mask_path=reference_mask_path,
        reference_identity_mask_sha256=str(
            input_hashes.get("reference_mask") or ""
        ),
    )
    output = args.output or (
        Path(__file__).resolve().parent
        / "grooming_final_video_fur_surface_v22032_replay.json"
    )
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    policy = require_dict(result.get("policy"), "audit policy")
    audited_reference = require_dict(
        result.get("validated_action_reference"),
        "audited action reference",
    )
    gate_record = require_dict(selected_gate.get("gate"), "selected gate record")
    expected_sha = str(action_config.get("reference_sha256") or "")
    sha_values = {
        expected_sha,
        str(selected_gate.get("selected_candidate_sha256") or ""),
        str(gate_record.get("candidate_sha256") or ""),
        str(audited_reference.get("sha256") or ""),
    }
    issues: list[str] = []
    if scope != pipeline.FINAL_VIDEO_VALIDATED_ACTION_SURFACE_SCOPE:
        issues.append(f"unexpected scope: {scope}")
    if result.get("passed") is not True:
        issues.append(f"audit did not pass: {result.get('verdict')}")
    if policy.get("comparison_scope") != scope:
        issues.append("audit policy scope differs from selected scope")
    if result.get("failed_frame_indices") != []:
        issues.append("one or more sampled frames failed")
    if len(sha_values) != 1 or len(expected_sha) != 64:
        issues.append("reference provenance SHA-256 values differ")

    print(
        json.dumps(
            {
                "job_id": job.get("job_id"),
                "api_version_under_test": pipeline.API_VERSION,
                "scope": scope,
                "passed": result.get("passed"),
                "sampled_frames": result.get("sampling", {}).get(
                    "frame_indices"
                ),
                "failed_frames": result.get("failed_frame_indices"),
                "reference_sha256": expected_sha,
                "output": str(output),
                "issues": issues,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if not issues else 5


if __name__ == "__main__":
    raise SystemExit(main())
