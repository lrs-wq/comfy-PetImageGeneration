#!/usr/bin/env python3
"""Run one production grooming upload and verify the complete v2.20.32 route."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import requests


BASE_URL = "http://127.0.0.1:8770/api/local-video"
ENV_PATH = Path("/etc/comfy-pet-pipeline/api.env")
SOURCE_PATH = Path(
    "/data/comfy-pet-pipeline/outputs/local_video_api/"
    "20260804T032056Z-66bb7b39c6/input.png"
)
AUDIT_DIR = Path(
    "/data/comfy-pet-pipeline/validation/e2e/"
    "v2.20.32-grooming-final-audit-v2"
)
EXPECTED_API_VERSION = "2.20.32"
EXPECTED_GROOMING_POLICY = "main_case_grooming_raised_paw_cv_v2"
EXPECTED_GROOMING_VERSION = "2.0"
EXPECTED_FINAL_VIDEO_POLICY = (
    "final_video_validated_action_reference_fur_surface_v2"
)
EXPECTED_FINAL_VIDEO_VERSION = "2.1.0"
EXPECTED_FINAL_VIDEO_SCOPE = "validated_action_reference_dual_surface_v2"
POLL_SECONDS = 15
TIMEOUT_SECONDS = 6 * 60 * 60


def emit(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=False), flush=True)


def read_api_key() -> str:
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith("LOCAL_VIDEO_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("LOCAL_VIDEO_API_KEY is unavailable")


def acceptance_issues(job: dict[str, Any]) -> list[str]:
    expected = {
        "status": "completed",
        "generation_mode": "subject_replacement",
        "canonical_reference_generated": True,
        "canonical_reference_used": True,
        "generated_canonical_bypassed": False,
        "effective_subject_source": "qwen_generated_upload",
        "qwen_reference_generated_actions": ["grooming"],
        "qwen_reference_applied_actions": ["grooming"],
        "grooming_pose_gate_policy": EXPECTED_GROOMING_POLICY,
        "grooming_pose_gate_version": EXPECTED_GROOMING_VERSION,
        "grooming_pose_gate_status": "passed",
    }
    issues = [
        f"{name}: expected {value!r}, got {job.get(name)!r}"
        for name, value in expected.items()
        if job.get(name) != value
    ]

    selected_gates = job.get("grooming_pose_gate_selected_metrics")
    selected_gate = (
        selected_gates.get("grooming")
        if isinstance(selected_gates, dict)
        else None
    )
    appearance = (
        selected_gate.get("metrics", {}).get("appearance_consistency")
        if isinstance(selected_gate, dict)
        else None
    )
    registration = (
        selected_gate.get("registration")
        if isinstance(selected_gate, dict)
        else None
    )
    if (
        not isinstance(selected_gate, dict)
        or selected_gate.get("acceptable") is not True
        or selected_gate.get("policy") != EXPECTED_GROOMING_POLICY
        or selected_gate.get("version") != EXPECTED_GROOMING_VERSION
        or not isinstance(registration, dict)
        or registration.get("appearance_sampling_policy")
        != "terminal_paw_bands_v2"
        or not isinstance(appearance, dict)
        or appearance.get("available") is not True
    ):
        issues.append("selected grooming gate does not satisfy the v2 terminal-paw contract")

    surface_selected = job.get("fur_surface_gate_selected_metrics")
    grooming_surface = (
        surface_selected.get("grooming")
        if isinstance(surface_selected, dict)
        else None
    )
    if (
        not isinstance(grooming_surface, dict)
        or grooming_surface.get("evaluated") is not True
        or grooming_surface.get("acceptable") is not True
    ):
        issues.append("selected grooming keyframe did not pass the fur-surface gate")

    final_quality = job.get("final_video_fur_surface_quality")
    grooming_final = (
        final_quality.get("grooming")
        if isinstance(final_quality, dict)
        else None
    )
    if not isinstance(grooming_final, dict) or grooming_final.get("passed") is not True:
        issues.append("grooming final-video fur-surface audit is not passed")
    else:
        policy = grooming_final.get("policy")
        expected_policy = {
            "name": EXPECTED_FINAL_VIDEO_POLICY,
            "version": EXPECTED_FINAL_VIDEO_VERSION,
            "comparison_scope": EXPECTED_FINAL_VIDEO_SCOPE,
            "non_radial_absolute_observations_nonblocking": True,
            "geometry_relative_nonblocking": True,
            "unmasked_contour_relative_required": True,
            "prop_excluded_texture_relative_required": True,
            "all_sampled_frames_must_pass": True,
        }
        if not isinstance(policy, dict):
            issues.append("grooming final-video audit policy is missing")
        else:
            for name, expected_value in expected_policy.items():
                if policy.get(name) != expected_value:
                    issues.append(
                        "grooming final-video policy "
                        f"{name}: expected {expected_value!r}, "
                        f"got {policy.get(name)!r}"
                    )
        expected_reference_sha = (
            job.get("qwen_action_keyframe_sha256s", {}).get("grooming")
            if isinstance(job.get("qwen_action_keyframe_sha256s"), dict)
            else None
        )
        selected_sha = (
            grooming_surface.get("selected_candidate_sha256")
            if isinstance(grooming_surface, dict)
            else None
        )
        gate = (
            grooming_surface.get("gate")
            if isinstance(grooming_surface, dict)
            else None
        )
        gate_sha = gate.get("candidate_sha256") if isinstance(gate, dict) else None
        audited_reference = grooming_final.get("validated_action_reference")
        audited_sha = (
            audited_reference.get("sha256")
            if isinstance(audited_reference, dict)
            else None
        )
        if not expected_reference_sha or not (
            expected_reference_sha == selected_sha == gate_sha == audited_sha
        ):
            issues.append(
                "grooming Qwen keyframe, selected gate and final audit SHA-256 "
                "values differ"
            )
        identity_masks = grooming_final.get("identity_masks")
        pose_mask = (
            identity_masks.get("pose_video")
            if isinstance(identity_masks, dict)
            else None
        )
        reference_mask = (
            identity_masks.get("reference")
            if isinstance(identity_masks, dict)
            else None
        )
        for label, mask in (
            ("pose video", pose_mask),
            ("reference", reference_mask),
        ):
            mask_sha = mask.get("sha256") if isinstance(mask, dict) else None
            if not isinstance(mask_sha, str) or len(mask_sha) != 64:
                issues.append(f"grooming final-video {label} mask SHA-256 is missing")
        if grooming_final.get("failed_frame_indices") != []:
            issues.append("grooming final-video audit contains failed sample frames")
        sampled = grooming_final.get("sampling")
        if (
            not isinstance(sampled, dict)
            or sampled.get("actual_count") != 9
        ):
            issues.append("grooming final-video audit did not inspect 9 frames")

    outputs = job.get("outputs")
    output_types = {
        str(item.get("type"))
        for item in outputs
        if isinstance(item, dict) and item.get("type")
    } if isinstance(outputs, list) else set()
    required = {
        "cartoon_reference",
        "transparent_keyframe",
        "subject_reference",
        "subject_replacement_video",
        "transparent_gif",
    }
    if required - output_types:
        issues.append("missing output types: " + ", ".join(sorted(required - output_types)))
    return issues


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--completed-job", type=Path)
    args = parser.parse_args()
    if args.completed_job:
        job = json.loads(args.completed_job.read_text(encoding="utf-8"))
        issues = acceptance_issues(job)
        emit("completed_job_check", job_id=job.get("job_id"), issues=issues)
        return 0 if not issues else 5

    api_key = read_api_key()
    headers = {"X-API-Key": api_key}
    health_response = requests.get(f"{BASE_URL}/health", headers=headers, timeout=60)
    health_response.raise_for_status()
    health = health_response.json()
    grooming_health = (
        health.get("qwen_style", {})
        .get("pose_guard", {})
        .get("grooming_pose_gate", {})
    )
    surface_health = (
        health.get("qwen_style", {})
        .get("pose_guard", {})
        .get("fur_surface_gate", {})
    )
    health_issues = []
    if health.get("status") != "ok":
        health_issues.append(f"health status is {health.get('status')!r}")
    if health.get("version") != EXPECTED_API_VERSION:
        health_issues.append(f"API version is {health.get('version')!r}")
    if grooming_health.get("policy") != EXPECTED_GROOMING_POLICY:
        health_issues.append(f"grooming policy is {grooming_health.get('policy')!r}")
    if grooming_health.get("version") != EXPECTED_GROOMING_VERSION:
        health_issues.append(f"grooming gate version is {grooming_health.get('version')!r}")
    if (
        surface_health.get("final_video_uploaded_grooming_scope")
        != EXPECTED_FINAL_VIDEO_SCOPE
    ):
        health_issues.append(
            "grooming final-video comparison scope is "
            f"{surface_health.get('final_video_uploaded_grooming_scope')!r}"
        )
    expected_reference_policies = surface_health.get(
        "final_video_uploaded_qwen_reference_policies"
    )
    if (
        not isinstance(expected_reference_policies, dict)
        or expected_reference_policies.get("grooming")
        != "qwen_action_specific_identity"
    ):
        health_issues.append(
            "grooming final-video Qwen reference policy is missing"
        )
    if (
        surface_health.get(
            "final_video_grooming_no_prop_identity_mask_required_for_v2"
        )
        is not True
    ):
        health_issues.append("grooming final-video no-prop mask contract is disabled")
    emit("health_preflight", issues=health_issues)
    if health_issues:
        return 5

    with SOURCE_PATH.open("rb") as image_handle:
        response = requests.post(
            f"{BASE_URL}/jobs",
            headers=headers,
            files={"image": (SOURCE_PATH.name, image_handle, "image/png")},
            data={
                "generation_mode": "subject_replacement",
                "actions": "grooming",
                "subject_profile": "sleek_tabby_cat_v1",
                "output_format": "transparent_gif",
                "prompt": "full-body pet identity preservation grooming route validation",
            },
            timeout=180,
        )
    if response.status_code != 202:
        emit("submit_failed", status_code=response.status_code, body=response.text[:2000])
        return 2
    job_id = str(response.json()["job_id"])
    emit("submitted", job_id=job_id)

    started = time.monotonic()
    last_observation: tuple[Any, ...] | None = None
    while True:
        if time.monotonic() - started > TIMEOUT_SECONDS:
            emit("monitor_timeout", job_id=job_id)
            return 3
        poll = requests.get(
            f"{BASE_URL}/jobs/{job_id}",
            headers=headers,
            timeout=60,
        )
        poll.raise_for_status()
        job = poll.json()
        observation = (
            job.get("status"),
            job.get("stage"),
            job.get("progress"),
            job.get("current_action"),
        )
        if observation != last_observation:
            emit("progress", job_id=job_id, observation=observation)
            last_observation = observation
        if job.get("status") in {"completed", "failed", "cancelled"}:
            AUDIT_DIR.mkdir(parents=True, exist_ok=True)
            record_path = AUDIT_DIR / f"grooming-e2e-{job_id}.json"
            record_path.write_text(
                json.dumps(job, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            issues = acceptance_issues(job)
            emit(
                "finished",
                job_id=job_id,
                status=job.get("status"),
                record_path=str(record_path),
                issues=issues,
            )
            return 0 if not issues else 5
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
