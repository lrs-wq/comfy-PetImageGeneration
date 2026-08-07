#!/usr/bin/env python3
"""Fail unless the local video service is idle enough for a safe restart."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlopen


HEALTH_URL = "http://127.0.0.1:8770/api/local-video/health"
OUTPUT_ROOT = Path("/data/comfy-pet-pipeline/outputs/local_video_api")
ACTIVE_STATES = {"queued", "running"}


def main() -> int:
    with urlopen(HEALTH_URL, timeout=15) as response:
        health = json.load(response)

    active_jobs: list[dict[str, str]] = []
    if OUTPUT_ROOT.is_dir():
        for status_path in OUTPUT_ROOT.glob("*/status.json"):
            try:
                job = json.loads(status_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if job.get("status") in ACTIVE_STATES:
                active_jobs.append(
                    {
                        "job_id": str(job.get("job_id") or status_path.parent.name),
                        "status": str(job.get("status")),
                        "stage": str(job.get("stage") or ""),
                    }
                )

    api_pending = int(health.get("job_queue", {}).get("pending") or 0)
    comfy_queue = health.get("comfyui", {}).get("queue") or {}
    comfy_running = int(comfy_queue.get("running") or 0)
    comfy_pending = int(comfy_queue.get("pending") or 0)
    safe = not (api_pending or comfy_running or comfy_pending or active_jobs)
    subject_replacement = health.get("subject_replacement") or {}
    effective_config = subject_replacement.get("effective_subject_config") or {}
    effective_actions = effective_config.get("actions") or {}
    print(
        json.dumps(
            {
                "safe_to_restart": safe,
                "health_status": health.get("status"),
                "version": health.get("version"),
                "worker_alive": health.get("worker_alive"),
                "models_ok": (health.get("models") or {}).get("ok"),
                "subject_replacement_ok": subject_replacement.get("ok"),
                "runtime_source": effective_config.get("runtime_source"),
                "base_seed": effective_config.get("base_seed"),
                "effective_actions": {
                    action: {
                        "driving_animation": config.get("driving_animation"),
                        "pose_strength": config.get("pose_strength"),
                        "seed_offset": config.get("seed_offset"),
                        "effective_seed": config.get("effective_seed"),
                    }
                    for action, config in effective_actions.items()
                    if isinstance(config, dict)
                },
                "api_pending": api_pending,
                "comfy_running": comfy_running,
                "comfy_pending": comfy_pending,
                "active_jobs": active_jobs,
            },
            ensure_ascii=False,
        )
    )
    return 0 if safe else 1


if __name__ == "__main__":
    raise SystemExit(main())
