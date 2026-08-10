#!/usr/bin/env python3
"""Submit exactly one prepared semantic-anatomy eating probe and archive it."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any


COMFY_URL = "http://127.0.0.1:8188"
COMFY_OUTPUT = Path("/data/comfy-pet-pipeline/comfyui/output")
RUN_ID = "codex-upload-only-eating-semantic-anatomy-cycle17-20260731-r2"
EXPERIMENT = Path("/data/comfy-pet-pipeline/experiments/scail2_tm_test") / RUN_ID
INPUT = Path("/data/comfy-pet-pipeline/comfyui/input/scail2_test") / RUN_ID
VALIDATION = Path(
    "/data/comfy-pet-pipeline/archive/validation/"
    "20260731-upload-only-identity-17f-r1/semantic_anatomy_cycle17_probe_r2"
)
ARCHIVE = VALIDATION / "gpu_eating_semantic_anatomy_r2_once"
EXPECTED = {
    EXPERIMENT / "workflow_api.json": "9ee5856be6a1dd1b78ede32595fbd26366f798514e71982abe8ddc3b4790fd64",
    INPUT / "tm_pose.mp4": "499dcbfa935c9dae5cb331a7fdbc1e5a8af0cc4f7593f2d3b4ba30b67034a179",
    INPUT / "tm_identity_mask.mp4": "85ed2847274fc9cd352fe853f3092337c86280237eac21fde49d76aebb022f31",
    INPUT / "reference.png": "6ab8019c01dedba3ad4a24169cb96df1a6982924b55e79f8063686e627d9b957",
    INPUT / "reference_identity_mask.png": "29ad86423c9cf986ed6818cff338de5510564ae7a8a5751ac6bd8b70572dc9f2",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def api_json(path: str, payload: dict[str, Any] | None = None) -> Any:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        COMFY_URL + path,
        data=body,
        headers={"Content-Type": "application/json"},
        method="GET" if body is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def save(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def output_items(history: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for node_id, node_output in history.get("outputs", {}).items():
        for collection, values in node_output.items():
            if not isinstance(values, list):
                continue
            for value in values:
                if isinstance(value, dict) and value.get("filename"):
                    items.append({"node_id": node_id, "collection": collection, **value})
    return items


def main() -> int:
    # Exist-ok false is the physical exactly-once guard.
    ARCHIVE.mkdir(parents=True, exist_ok=False)
    raw_root = ARCHIVE / "raw_outputs"
    raw_root.mkdir()
    report_path = ARCHIVE / "submission_report.json"

    observed = {str(path): sha256_file(path) for path in EXPECTED}
    mismatches = {
        str(path): {"expected": expected, "observed": observed[str(path)]}
        for path, expected in EXPECTED.items()
        if observed[str(path)] != expected
    }
    if mismatches:
        raise RuntimeError(f"Pre-submit SHA mismatch: {mismatches}")
    queue_before = api_json("/queue")
    if queue_before.get("queue_running") or queue_before.get("queue_pending"):
        raise RuntimeError(f"Comfy queue is not empty: {queue_before}")

    workflow_path = EXPERIMENT / "workflow_api.json"
    graph = json.loads(workflow_path.read_text(encoding="utf-8"))
    client_id = f"codex-eating-semantic-r2-{uuid.uuid4().hex[:12]}"
    started_wall = time.time()
    started_mono = time.monotonic()
    report: dict[str, Any] = {
        "probe": "upload_only_eating_semantic_anatomy_r2_17f_gpu_once",
        "prepared_run_id": RUN_ID,
        "archive": str(ARCHIVE),
        "input_sha256": observed,
        "workflow_node_count": len(graph),
        "client_id": client_id,
        "started_epoch": started_wall,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started_wall)),
        "queue_before": queue_before,
        "submission_count": 0,
        "grooming_submitted": False,
        "production_modified": False,
        "status": "preflight_passed",
    }
    save(report_path, report)

    try:
        response = api_json("/prompt", {"prompt": graph, "client_id": client_id})
        report["submission_count"] = 1
        if response.get("node_errors"):
            raise RuntimeError("Workflow validation failed: " + json.dumps(response["node_errors"], ensure_ascii=False))
        prompt_id = str(response["prompt_id"])
        report.update({
            "status": "submitted",
            "prompt_id": prompt_id,
            "queue_number": response.get("number"),
            "submission_response": response,
        })
        save(report_path, report)
        print(json.dumps({"event": "submitted", "prompt_id": prompt_id, "queue_number": response.get("number"), "archive": str(ARCHIVE)}), flush=True)

        history_payload: dict[str, Any] | None = None
        last_progress = 0.0
        while time.monotonic() - started_mono < 3600:
            payload = api_json(f"/history/{prompt_id}")
            if prompt_id in payload:
                history_payload = payload
                break
            elapsed = time.monotonic() - started_mono
            if elapsed - last_progress >= 20:
                queue = api_json("/queue")
                print(json.dumps({"event": "waiting", "prompt_id": prompt_id, "elapsed_seconds": round(elapsed, 3), "queue_running": len(queue.get("queue_running", [])), "queue_pending": len(queue.get("queue_pending", []))}), flush=True)
                last_progress = elapsed
            time.sleep(3)
        if history_payload is None:
            raise TimeoutError("Prompt did not finish within 3600 seconds")
        save(ARCHIVE / "history_response.json", history_payload)
        history = history_payload[prompt_id]
        save(ARCHIVE / "history.json", history)
        status = history.get("status", {})
        if status.get("status_str") == "error" or status.get("completed") is False:
            raise RuntimeError(f"Prompt failed: {json.dumps(status, ensure_ascii=False)}")

        archived: list[dict[str, Any]] = []
        for index, item in enumerate(output_items(history)):
            if item.get("type") != "output":
                continue
            source = COMFY_OUTPUT / item.get("subfolder", "") / item["filename"]
            if not source.is_file():
                raise RuntimeError(f"History output missing: {source}")
            target = raw_root / f"{index:02d}_{source.name}"
            shutil.copy2(source, target)
            archived.append({
                **item,
                "source_path": str(source),
                "archived_path": str(target),
                "sha256": sha256_file(target),
                "bytes": target.stat().st_size,
            })
        queue_after = api_json("/queue")
        if queue_after.get("queue_running") or queue_after.get("queue_pending"):
            raise RuntimeError(f"Queue not empty after completed history: {queue_after}")
        completed_wall = time.time()
        report.update({
            "status": "completed",
            "completed_epoch": completed_wall,
            "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(completed_wall)),
            "elapsed_seconds": round(time.monotonic() - started_mono, 3),
            "history_status": status,
            "history_response_sha256": sha256_file(ARCHIVE / "history_response.json"),
            "history_sha256": sha256_file(ARCHIVE / "history.json"),
            "outputs": archived,
            "queue_after": queue_after,
        })
        save(report_path, report)
        print(json.dumps({"event": "completed", "prompt_id": prompt_id, "queue_number": report["queue_number"], "elapsed_seconds": report["elapsed_seconds"], "archive": str(ARCHIVE), "outputs": archived, "queue_after": queue_after}, ensure_ascii=False), flush=True)
        return 0
    except Exception as error:
        report.update({"status": "failed", "error": repr(error), "elapsed_seconds": round(time.monotonic() - started_mono, 3)})
        try:
            report["queue_after"] = api_json("/queue")
        except Exception as queue_error:  # noqa: BLE001
            report["queue_after_error"] = repr(queue_error)
        save(report_path, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
