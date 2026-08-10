#!/usr/bin/env python3
"""Explicitly submit the frozen canonical Qwen workflow exactly once.

The CPU builder never invokes this file.  Running it requires an explicit flag,
an unchanged passing manifest, unchanged artifact hashes, and an empty ComfyUI
queue.  It posts one workflow and exits without waiting or retrying.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
import uuid
from pathlib import Path


DEFAULT_BUNDLE = Path(
    "/data/comfy-pet-pipeline/archive/validation/"
    "20260731-upload-only-identity-17f-r1/"
    "upload_identity_main_case_canonical_v1_r2"
)
COMFY_URL = "http://127.0.0.1:8188"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def get_json(path: str) -> dict:
    request = urllib.request.Request(COMFY_URL + path, method="GET")
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument(
        "--i-understand-this-submits-one-gpu-job",
        action="store_true",
        help="required explicit GPU submission acknowledgement",
    )
    args = parser.parse_args()
    if not args.i_understand_this_submits_one_gpu_job:
        raise SystemExit("refusing_without_explicit_gpu_submission_flag")

    bundle = args.bundle.resolve()
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "cpu_gate_passed_ready_for_explicit_submission":
        raise SystemExit("manifest_not_ready_for_submission")
    hard_checks = manifest.get("hard_checks")
    if not isinstance(hard_checks, dict) or not hard_checks or not all(
        value is True for value in hard_checks.values()
    ):
        raise SystemExit("cpu_hard_checks_not_all_true")
    if manifest.get("gpu_submitted") is not False:
        raise SystemExit("manifest_already_marks_gpu_submission")

    artifacts = manifest["artifacts"]
    for name in ("guide", "guide_comfy_input", "upload_comfy_input", "workflow"):
        artifact = artifacts[name]
        path = Path(artifact["path"])
        if not path.is_file() or sha256_file(path) != artifact["sha256"]:
            raise SystemExit(f"artifact_missing_or_changed:{name}")
    queue = get_json("/queue")
    if queue.get("queue_running") or queue.get("queue_pending"):
        raise SystemExit("comfy_queue_not_empty")

    workflow_path = Path(artifacts["workflow"]["path"])
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    payload = json.dumps(
        {"prompt": workflow, "client_id": str(uuid.uuid4())},
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        COMFY_URL + "/prompt",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.load(response)
    record = {
        "run_id": manifest["run_id"],
        "submitted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prompt_id": result.get("prompt_id"),
        "response": result,
        "workflow_sha256": sha256_file(workflow_path),
        "single_post_only": True,
        "retry_performed": False,
    }
    record_path = bundle / "gpu_submission_record.json"
    if record_path.exists():
        raise SystemExit("submission_record_already_exists")
    record_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
