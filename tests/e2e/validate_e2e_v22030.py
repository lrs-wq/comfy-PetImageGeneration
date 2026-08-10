#!/usr/bin/env python3
"""Submit and monitor the single v2.20.32 production-route acceptance job."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import requests


BASE_URL = "http://127.0.0.1:8770/api/local-video"
ENV_PATH = Path("/etc/comfy-pet-pipeline/api.env")
SOURCE_PATH = Path(
    "/data/comfy-pet-pipeline/outputs/local_video_api/"
    "20260803T013932Z-b8f14adfe2/input.png"
)
AUDIT_DIR = Path(
    "/data/comfy-pet-pipeline/validation/e2e/"
    "v2.20.32-grooming-final-audit-v2"
)
POLL_SECONDS = 15
TIMEOUT_SECONDS = 6 * 60 * 60
EXPECTED_API_VERSION = "2.20.32"
EXPECTED_REFERENCE_PIPELINE = (
    "qwen_full_body_then_action_keyframes_then_scail_v1"
)
EXPECTED_REFERENCE_PIPELINE_VERSION = "1.0.0"
EXPECTED_CONTACT_POLICY = "warm_face_component_above_detected_bowl_v2"
EXPECTED_EATING_BOWL_SPLIT_FALLBACK_POLICY = (
    "food_anchored_beige_bowl_after_eroded_warm_core_miss_v2"
)
EXPECTED_ACTION_NORMALIZATION_POLICY = (
    "alpha_preserving_rgba_canvas_512_then_byte_copy_v1"
)
EXPECTED_ACTION_MATTE_POLICY = (
    "direct_chroma_then_per_candidate_semantic_matte_v1"
)
EXPECTED_GROOMING_POSE_POLICY = "main_case_grooming_raised_paw_cv_v2"
EXPECTED_GROOMING_POSE_VERSION = "2.0"
EXPECTED_SURFACE_RETRY_POLICY = (
    "preserve_configured_eating_bowl_surface_recovery_v1"
)
EXPECTED_FINAL_VIDEO_POLICY = (
    "final_video_validated_action_reference_fur_surface_v2"
)
EXPECTED_FINAL_VIDEO_SCOPE = "validated_action_reference_dual_surface_v2"


def read_api_key() -> str:
    for raw_line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() == "LOCAL_VIDEO_API_KEY":
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if value:
                return value
    raise RuntimeError(f"LOCAL_VIDEO_API_KEY is missing from {ENV_PATH}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def emit(event: str, **values: object) -> None:
    print(
        json.dumps({"event": event, **values}, ensure_ascii=False, sort_keys=True),
        flush=True,
    )


def health_preflight(headers: dict[str, str]) -> list[str]:
    try:
        response = requests.get(
            f"{BASE_URL}/health",
            headers=headers,
            timeout=60,
        )
        response.raise_for_status()
        health = response.json()
    except Exception as exc:  # noqa: BLE001 - report a bounded preflight failure
        return [f"health request failed: {type(exc).__name__}: {exc}"]

    pose_guard = health.get("qwen_style", {}).get("pose_guard", {})
    contact_gate = pose_guard.get("eating_mouth_contact_gate", {})
    grooming_gate = pose_guard.get("grooming_pose_gate", {})
    surface_gate = pose_guard.get("fur_surface_gate", {})
    observed = {
        "status": health.get("status"),
        "version": health.get("version"),
        "worker_alive": health.get("worker_alive"),
        "subject_replacement_ok": health.get("subject_replacement", {}).get(
            "ok"
        ),
        "reference_pipeline": pose_guard.get("reference_pipeline"),
        "reference_pipeline_version": pose_guard.get("reference_pipeline_version"),
        "canonical_reference_normalized_and_matted": pose_guard.get(
            "canonical_reference_normalized_and_matted"
        ),
        "action_keyframes_normalized_and_matted": pose_guard.get(
            "action_keyframes_normalized_and_matted"
        ),
        "raw_upload_sent_to_scail": pose_guard.get("raw_upload_sent_to_scail"),
        "action_normalization_policy": pose_guard.get(
            "upload_pose_candidate_normalization_policy"
        ),
        "action_matte_policy": pose_guard.get(
            "upload_pose_candidate_matte_policy"
        ),
        "selected_candidate_byte_copied": pose_guard.get(
            "normalization_contract", {}
        ).get("selected_candidate_byte_copied_to_scail_reference"),
        "wan_green_derived_one_way": pose_guard.get(
            "normalization_contract", {}
        ).get("wan_green_derived_one_way_from_rgba"),
        "contact_policy": contact_gate.get("policy"),
        "contact_thin_warm_rim_fallback_policy": contact_gate.get(
            "thin_warm_rim_fallback_policy"
        ),
        "contact_fallback_preserves_checks": contact_gate.get(
            "fallback_preserves_width_prop_paw_and_contact_checks"
        ),
        "contact_gap_px_max": contact_gate.get("gap_px_max"),
        "contact_width_bowl_ratio_max": contact_gate.get(
            "component_width_bowl_ratio_max"
        ),
        "grooming_pose_policy": grooming_gate.get("policy"),
        "grooming_pose_version": grooming_gate.get("version"),
        "surface_retry_policy": surface_gate.get("retry_policy"),
        "surface_preserves_eating_bowl_recovery": surface_gate.get(
            "preserves_configured_eating_bowl_recovery"
        ),
        "surface_configured_eating_bowl_recovery_attempts": surface_gate.get(
            "configured_eating_bowl_recovery_attempts"
        ),
        "surface_configured_eating_bowl_recovery_profile_prefix": surface_gate.get(
            "configured_eating_bowl_recovery_profile_prefix"
        ),
        "surface_blocks_scail_on_exhaustion": surface_gate.get(
            "blocks_scail_on_exhaustion"
        ),
        "pose_max_attempts": pose_guard.get("max_attempts"),
        "bowl_extra_attempts": pose_guard.get("bowl_extra_attempts"),
        "bowl_max_attempts": pose_guard.get("bowl_max_attempts"),
        "final_video_policy": surface_gate.get("final_video_policy"),
        "final_video_version": surface_gate.get("final_video_version"),
        "final_video_uploaded_eating_scope": surface_gate.get(
            "final_video_uploaded_eating_scope"
        ),
        "final_video_qwen_reference_sha_required": surface_gate.get(
            "final_video_profile_and_qwen_reference_sha_required"
        ),
        "final_video_identity_mask_sha_required": surface_gate.get(
            "final_video_scail_identity_mask_sha_required"
        ),
        "final_video_unmasked_contour_required": surface_gate.get(
            "final_video_unmasked_contour_relative_required"
        ),
        "final_video_prop_excluded_texture_required": surface_gate.get(
            "final_video_prop_excluded_texture_relative_required"
        ),
        "final_video_non_radial_absolute_observations_nonblocking": surface_gate.get(
            "final_video_non_radial_absolute_observations_nonblocking_only_for_uploaded_qwen_transparent_actions"
        ),
        "final_video_radial_load_or_backstop": surface_gate.get(
            "final_video_radial_fraction_requires_load_expansion_or_energy_backstop"
        ),
        "final_video_radial_load_proxy": surface_gate.get(
            "final_video_radial_load_proxy"
        ),
        "final_video_radial_backstop_reuses_thresholds": surface_gate.get(
            "final_video_radial_energy_backstop_reuses_existing_thresholds"
        ),
        "final_video_absolute_radial_blocking": surface_gate.get(
            "final_video_absolute_radial_issue_remains_blocking"
        ),
        "final_video_texture_baseline_fail_closed": surface_gate.get(
            "final_video_texture_baseline_contract_fails_closed"
        ),
        "final_video_unknown_relative_fails_closed": surface_gate.get(
            "final_video_unknown_relative_issue_fails_closed"
        ),
        "final_video_post_contour_required": surface_gate.get(
            "final_video_post_animation_contour_gate_required_for_uploaded_eating_v2"
        ),
    }
    expected = {
        "status": "ok",
        "version": EXPECTED_API_VERSION,
        "worker_alive": True,
        "subject_replacement_ok": True,
        "reference_pipeline": EXPECTED_REFERENCE_PIPELINE,
        "reference_pipeline_version": EXPECTED_REFERENCE_PIPELINE_VERSION,
        "canonical_reference_normalized_and_matted": True,
        "action_keyframes_normalized_and_matted": True,
        "raw_upload_sent_to_scail": False,
        "action_normalization_policy": EXPECTED_ACTION_NORMALIZATION_POLICY,
        "action_matte_policy": EXPECTED_ACTION_MATTE_POLICY,
        "selected_candidate_byte_copied": True,
        "wan_green_derived_one_way": True,
        "contact_policy": EXPECTED_CONTACT_POLICY,
        "contact_thin_warm_rim_fallback_policy": (
            EXPECTED_EATING_BOWL_SPLIT_FALLBACK_POLICY
        ),
        "contact_fallback_preserves_checks": True,
        "contact_gap_px_max": 8,
        "contact_width_bowl_ratio_max": 0.70,
        "grooming_pose_policy": EXPECTED_GROOMING_POSE_POLICY,
        "grooming_pose_version": EXPECTED_GROOMING_POSE_VERSION,
        "surface_retry_policy": EXPECTED_SURFACE_RETRY_POLICY,
        "surface_preserves_eating_bowl_recovery": True,
        "surface_configured_eating_bowl_recovery_attempts": 2,
        "surface_configured_eating_bowl_recovery_profile_prefix": (
            "bowl_palette_canvas_seed_retry_"
        ),
        "surface_blocks_scail_on_exhaustion": True,
        "pose_max_attempts": 3,
        "bowl_extra_attempts": 2,
        "bowl_max_attempts": 5,
        "final_video_policy": EXPECTED_FINAL_VIDEO_POLICY,
        "final_video_version": "2.1.0",
        "final_video_uploaded_eating_scope": EXPECTED_FINAL_VIDEO_SCOPE,
        "final_video_qwen_reference_sha_required": True,
        "final_video_identity_mask_sha_required": True,
        "final_video_unmasked_contour_required": True,
        "final_video_prop_excluded_texture_required": True,
        "final_video_non_radial_absolute_observations_nonblocking": True,
        "final_video_radial_load_or_backstop": True,
        "final_video_radial_load_proxy": (
            "fine_radial_brush_fraction_x_fine_gradient_p90"
        ),
        "final_video_radial_backstop_reuses_thresholds": True,
        "final_video_absolute_radial_blocking": True,
        "final_video_texture_baseline_fail_closed": True,
        "final_video_unknown_relative_fails_closed": True,
        "final_video_post_contour_required": True,
    }
    issues = [
        f"{name}: expected {expected_value!r}, got {observed.get(name)!r}"
        for name, expected_value in expected.items()
        if observed.get(name) != expected_value
    ]
    emit("health_preflight", observed=observed, issues=issues)
    return issues


def completed_job_acceptance_issues(job: dict[str, object]) -> list[str]:
    expected_fields = {
        "generation_mode": "subject_replacement",
        "reference_pipeline": EXPECTED_REFERENCE_PIPELINE,
        "reference_pipeline_version": EXPECTED_REFERENCE_PIPELINE_VERSION,
        "canonical_reference_generated": True,
        "canonical_reference_used": True,
        "generated_canonical_bypassed": False,
        "direct_scail_uploaded_identity_reference": False,
        "effective_subject_source": "qwen_generated_upload",
        "pose_identity_input_policy": "canonical_character_image1",
        "pose_candidate_normalization_policy": (
            EXPECTED_ACTION_NORMALIZATION_POLICY
        ),
        "pose_candidate_matte_policy": EXPECTED_ACTION_MATTE_POLICY,
        "qwen_action_reference_normalization_policy": (
            EXPECTED_ACTION_NORMALIZATION_POLICY
        ),
        "qwen_action_reference_byte_preserved": True,
        "fur_surface_gate_max_consecutive_rejections": 3,
        "fur_surface_gate_retry_policy": EXPECTED_SURFACE_RETRY_POLICY,
        "fur_surface_gate_preserves_configured_eating_bowl_recovery": True,
        "fur_surface_gate_configured_eating_bowl_recovery_attempts": 2,
        # gpu_submitted is create-time bookkeeping and the worker does not mutate
        # it. Prove real Qwen/SCAIL execution from completed engine, action,
        # reference, output, and final-video audit evidence instead.
        "gpu_required": True,
        "gpu_bypass_reason": None,
        "generation_controls_applied": True,
        "reference_generation_engine": "qwen-image-edit",
        "reference_generation_status": "completed",
        "reference_generation_completed": True,
        "qwen_reference_generated_actions": ["eating"],
        "qwen_reference_applied_actions": ["eating"],
        "qwen_reference_bypassed_actions": [],
        "qwen_reference_pre_scail_revalidated": True,
        "scail_reference_policy": {
            "eating": "qwen_action_identity_with_bowl_mapped_red"
        },
    }
    issues = [
        f"{name}: expected {expected_value!r}, got {job.get(name)!r}"
        for name, expected_value in expected_fields.items()
        if job.get(name) != expected_value
    ]
    surface_quality = job.get("fur_surface_gate_final_quality")
    if (
        not isinstance(surface_quality, dict)
        or surface_quality.get("status") != "passed"
    ):
        issues.append("fur_surface_gate_final_quality.status is not passed")

    reference_sha256s = job.get("qwen_action_keyframe_sha256s")
    selected_metrics = job.get("fur_surface_gate_selected_metrics")
    if not isinstance(reference_sha256s, dict) or not isinstance(
        selected_metrics, dict
    ):
        issues.append("Qwen action reference SHA provenance is incomplete")
    else:
        for action, reference_sha256 in reference_sha256s.items():
            selected = selected_metrics.get(action)
            selected_sha256 = (
                selected.get("selected_candidate_sha256")
                if isinstance(selected, dict)
                else None
            )
            gate = selected.get("gate") if isinstance(selected, dict) else None
            gate_sha256 = (
                gate.get("candidate_sha256") if isinstance(gate, dict) else None
            )
            if not reference_sha256 or not (
                reference_sha256 == selected_sha256 == gate_sha256
            ):
                issues.append(
                    f"{action} selected/gated/persisted SHA-256 values differ"
                )

    final_video_quality = job.get("final_video_fur_surface_quality")
    eating_audit = (
        final_video_quality.get("eating")
        if isinstance(final_video_quality, dict)
        else None
    )
    if not isinstance(eating_audit, dict) or eating_audit.get("passed") is not True:
        issues.append("eating final-video fur-surface audit is not passed")
    else:
        policy = eating_audit.get("policy")
        if not isinstance(policy, dict):
            issues.append("eating final-video audit policy is missing")
        else:
            expected_policy_fields = {
                "name": EXPECTED_FINAL_VIDEO_POLICY,
                "version": "2.1.0",
                "comparison_scope": EXPECTED_FINAL_VIDEO_SCOPE,
                "non_radial_absolute_observations_nonblocking": True,
                "geometry_relative_nonblocking": True,
                "unmasked_contour_relative_required": True,
                "prop_excluded_texture_relative_required": True,
                "radial_fraction_relative_requires_load_expansion_or_energy_backstop": True,
                "radial_load_proxy": (
                    "fine_radial_brush_fraction_x_fine_gradient_p90"
                ),
                "radial_energy_backstop_uses_existing_ratio_and_delta_thresholds": True,
                "absolute_radial_issue_remains_blocking": True,
                "texture_baseline_contract_fails_closed": True,
                "unknown_relative_issue_fails_closed": True,
                "all_sampled_frames_must_pass": True,
            }
            for name, expected_value in expected_policy_fields.items():
                if policy.get(name) != expected_value:
                    issues.append(
                        "eating final-video policy "
                        f"{name}: expected {expected_value!r}, got {policy.get(name)!r}"
                    )
        validated_reference = eating_audit.get("validated_action_reference")
        expected_eating_reference_sha = (
            reference_sha256s.get("eating")
            if isinstance(reference_sha256s, dict)
            else None
        )
        if (
            not isinstance(validated_reference, dict)
            or validated_reference.get("sha256") != expected_eating_reference_sha
            or validated_reference.get("provenance", {}).get(
                "pre_scail_revalidated"
            )
            is not True
        ):
            issues.append("final-video audit did not use the exact revalidated Qwen reference")
        identity_masks = eating_audit.get("identity_masks")
        for mask_name in ("pose_video", "reference"):
            mask = (
                identity_masks.get(mask_name)
                if isinstance(identity_masks, dict)
                else None
            )
            digest = mask.get("sha256") if isinstance(mask, dict) else None
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                issues.append(f"final-video {mask_name} SHA-256 is invalid")
        frames = eating_audit.get("frames")
        if not isinstance(frames, list) or len(frames) != 9:
            issues.append("final-video audit did not evaluate exactly 9 frames")
        else:
            for frame in frames:
                decision = frame.get("decision") if isinstance(frame, dict) else None
                segmentation = (
                    frame.get("segmentation") if isinstance(frame, dict) else None
                )
                exclusion = (
                    segmentation.get("texture_prop_exclusion")
                    if isinstance(segmentation, dict)
                    else None
                )
                radial_evidence = (
                    decision.get("radial_relative_evidence")
                    if isinstance(decision, dict)
                    else None
                )
                if (
                    frame.get("passed") is not True
                    or not isinstance(decision, dict)
                    or decision.get("blocking_issues") != []
                    or not isinstance(radial_evidence, dict)
                    or radial_evidence.get("blocking") is not False
                    or radial_evidence.get("absolute_radial_issue") is not False
                    or radial_evidence.get("load_expanded") is not False
                    or radial_evidence.get(
                        "normalized_energy_backstop_triggered"
                    )
                    is not False
                    or (
                        radial_evidence.get("relative_issue_present") is True
                        and radial_evidence.get(
                            "radial_fraction_without_load_expansion_observation"
                        )
                        is not True
                    )
                    or not isinstance(
                        radial_evidence.get("relative_issue_present"),
                        bool,
                    )
                    or not isinstance(exclusion, dict)
                    or exclusion.get("rgb_byte_preserved") is not True
                    or exclusion.get("alpha_only_decreased") is not True
                    or exclusion.get("morphology_applied") is not False
                ):
                    issues.append(
                        "final-video sampled frame contract failed at "
                        f"{frame.get('frame_index') if isinstance(frame, dict) else 'unknown'}"
                    )

    output_items = job.get("outputs")
    output_types = (
        {
            str(item.get("type"))
            for item in output_items
            if isinstance(item, dict) and item.get("type")
        }
        if isinstance(output_items, list)
        else set()
    )
    required_output_types = {
        "cartoon_reference",
        "transparent_keyframe",
        "subject_reference",
        "subject_replacement_video",
        "transparent_gif",
    }
    missing_output_types = sorted(required_output_types - output_types)
    if missing_output_types:
        issues.append("missing output types: " + ", ".join(missing_output_types))
    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--completed-job",
        type=Path,
        help=(
            "Recheck one already-completed status JSON without submitting or "
            "monitoring a new GPU job."
        ),
    )
    args = parser.parse_args(argv)
    if args.completed_job is not None:
        try:
            job = json.loads(args.completed_job.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - emit a bounded offline failure
            emit(
                "completed_job_read_failed",
                path=str(args.completed_job),
                error=f"{type(exc).__name__}: {exc}",
            )
            return 2
        if not isinstance(job, dict) or job.get("status") != "completed":
            emit(
                "completed_job_not_terminal_success",
                path=str(args.completed_job),
                status=job.get("status") if isinstance(job, dict) else None,
            )
            return 4
        acceptance_issues = completed_job_acceptance_issues(job)
        emit(
            "completed_job_recheck",
            path=str(args.completed_job),
            job_id=job.get("job_id"),
            status=job.get("status"),
            stage=job.get("stage"),
            acceptance_issues=acceptance_issues,
        )
        return 0 if not acceptance_issues else 5

    api_key = read_api_key()
    headers = {"X-API-Key": api_key}
    source_sha256 = sha256_file(SOURCE_PATH)
    emit(
        "source",
        path=str(SOURCE_PATH),
        bytes=SOURCE_PATH.stat().st_size,
        sha256=source_sha256,
    )

    preflight_issues = health_preflight(headers)
    if preflight_issues:
        emit("preflight_failed", issues=preflight_issues)
        return 5

    with SOURCE_PATH.open("rb") as image_handle:
        response = requests.post(
            f"{BASE_URL}/jobs",
            headers=headers,
            files={"image": (SOURCE_PATH.name, image_handle, "image/png")},
            data={
                "generation_mode": "subject_replacement",
                "actions": "eating",
                "subject_profile": "sleek_tabby_cat_v1",
                "output_format": "transparent_gif",
                "prompt": "full-body pet identity preservation production route validation",
            },
            timeout=180,
        )
    if response.status_code != 202:
        emit("submit_failed", status_code=response.status_code, body=response.text[:2000])
        return 2

    submitted = response.json()
    job_id = str(submitted["job_id"])
    status_url = f"{BASE_URL}/jobs/{job_id}"
    record_path = AUDIT_DIR / f"e2e-{job_id}.json"
    emit(
        "submitted",
        job_id=job_id,
        status_url=submitted.get("status_url"),
        source_sha256=source_sha256,
        record_path=str(record_path),
    )

    started = time.monotonic()
    last_observation: tuple[object, ...] | None = None
    last_emit = 0.0
    while True:
        elapsed = time.monotonic() - started
        if elapsed > TIMEOUT_SECONDS:
            emit("monitor_timeout", job_id=job_id, elapsed_seconds=round(elapsed, 1))
            return 3
        try:
            poll = requests.get(status_url, headers=headers, timeout=60)
            poll.raise_for_status()
            job = poll.json()
        except Exception as exc:  # noqa: BLE001 - monitor survives transient API failures
            emit("poll_error", job_id=job_id, error=f"{type(exc).__name__}: {exc}")
            time.sleep(POLL_SECONDS)
            continue

        observation = (
            job.get("status"),
            job.get("stage"),
            job.get("progress"),
            job.get("current_action"),
        )
        now = time.monotonic()
        if observation != last_observation or now - last_emit >= 60:
            emit(
                "progress",
                job_id=job_id,
                status=job.get("status"),
                stage=job.get("stage"),
                progress=job.get("progress"),
                current_action=job.get("current_action"),
                elapsed_seconds=round(elapsed, 1),
            )
            last_observation = observation
            last_emit = now

        if job.get("status") in {"completed", "failed", "cancelled"}:
            record_path.write_text(
                json.dumps(job, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            outputs = [
                {
                    "type": item.get("type"),
                    "action": item.get("action"),
                    "filename": item.get("filename"),
                    "primary": item.get("primary"),
                    "sha256": item.get("sha256"),
                }
                for item in job.get("outputs", [])
            ]
            acceptance_issues = (
                completed_job_acceptance_issues(job)
                if job.get("status") == "completed"
                else []
            )
            emit(
                "terminal",
                job_id=job_id,
                status=job.get("status"),
                stage=job.get("stage"),
                progress=job.get("progress"),
                error=job.get("error"),
                outputs=outputs,
                acceptance_issues=acceptance_issues,
                record_path=str(record_path),
            )
            if job.get("status") != "completed":
                return 4
            return 0 if not acceptance_issues else 5
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    sys.exit(main())
