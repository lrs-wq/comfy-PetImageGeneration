#!/usr/bin/env python3
"""Run one production grooming upload and verify the complete v2.20.31 route."""

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
    "v2.20.31-grooming-gate-v2"
)
EXPECTED_API_VERSION = "2.20.31"
EXPECTED_GROOMING_POLICY = "main_case_grooming_raised_paw_cv_v2"
EXPECTED_GROOMING_VERSION = "2.0"
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
    health_issues = []
    if health.get("status") != "ok":
        health_issues.append(f"health status is {health.get('status')!r}")
    if health.get("version") != EXPECTED_API_VERSION:
        health_issues.append(f"API version is {health.get('version')!r}")
    if grooming_health.get("policy") != EXPECTED_GROOMING_POLICY:
        health_issues.append(f"grooming policy is {grooming_health.get('policy')!r}")
    if grooming_health.get("version") != EXPECTED_GROOMING_VERSION:
        health_issues.append(f"grooming gate version is {grooming_health.get('version')!r}")
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
