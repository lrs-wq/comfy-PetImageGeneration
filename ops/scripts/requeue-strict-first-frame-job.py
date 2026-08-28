#!/usr/bin/env python3
"""Safely requeue a failed strict-first-frame job under the same job ID."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


JOB_ID_PATTERN = re.compile(r"^\d{8}T\d{6}Z-[0-9a-f]{10}$")
STRICT_POLICY = "strict_quality_gate_then_content_addressed_reuse_or_seed_wave_retry_v1"


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            os.environ.get("PET_PIPELINE_DATA_ROOT", "/data/comfy-pet-pipeline")
        ),
    )
    args = parser.parse_args()
    if not JOB_ID_PATTERN.fullmatch(args.job_id):
        parser.error("job_id has an invalid format")

    output_root = (args.data_root / "outputs" / "local_video_api").resolve()
    job_dir = (output_root / args.job_id).resolve()
    if job_dir.parent != output_root:
        parser.error("job path escaped the local-video output root")
    status_path = job_dir / "status.json"
    job = json.loads(status_path.read_text(encoding="utf-8"))
    if job.get("job_id") != args.job_id:
        parser.error("status.json job_id does not match the requested job")
    if job.get("status") != "failed":
        parser.error("only failed jobs may be requeued by this maintenance tool")
    if job.get("qwen_first_frame_publication_policy") != STRICT_POLICY:
        parser.error("job does not use the strict first-frame publication policy")

    previous_error = str(job.get("error") or "")
    history = list(job.get("qwen_first_frame_requeue_history") or [])
    history.append(
        {
            "requeued_at": datetime.now(timezone.utc).isoformat(),
            "previous_status": "failed",
            "previous_stage": job.get("stage"),
            "previous_error": previous_error[:4000],
            "reason": "resume_same_job_after_strict_first_frame_fix",
        }
    )
    job.update(
        status="queued",
        stage="queued_for_first_frame_quality_retry",
        error=None,
        active_prompt_id=None,
        finished_at=None,
        cancel_requested=False,
        recovery_active=True,
        qwen_first_frame_manual_requeue=True,
        qwen_first_frame_last_retry_reason=(
            "same job resumed after strict first-frame publication fix"
        ),
        qwen_first_frame_requeue_history=history,
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    write_json_atomic(status_path, job)
    print(
        json.dumps(
            {
                "job_id": args.job_id,
                "status": job["status"],
                "stage": job["stage"],
                "history_count": len(history),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
