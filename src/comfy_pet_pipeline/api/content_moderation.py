"""Content Review adapter and Comfy's independent publication policy.

No visual-quality fallback is allowed to override these decisions. Audit
records contain identifiers and scores, never OCR text or original media.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import mimetypes
import os
import queue
import subprocess
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

import requests
from PIL import Image

POLICY_VERSION = "comfy-content-moderation-v1"
MEDIA_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
    ".apng",
    ".mp4",
    ".mov",
    ".webm",
    ".mkv",
    ".avi",
}
VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".mkv", ".avi"}
HOLD_STATUSES = {"pending_review", "blocked", "moderation_error"}
RULES = (
    ("CM01", "上传的全部视图及用户提示在生成前审核"),
    ("CM02", "全部交付媒体在生成完成后异步审核，含降级结果"),
    ("CM03", "机器判定 block 时停止生成或交付"),
    ("CM04", "输入 review 暂停生成；结果 review 进入人工复核，不等待复核即可交付"),
    ("CM05", "输入审核异常暂停生成；结果审核异常保留重审记录，不阻塞首次交付"),
    ("CM06", "人工复核绑定当前文件摘要和审核版本"),
    ("CM07", "已判违规或人工驳回的结果拦截后续下载和尚未发出的成功回调"),
    ("CM08", "按配置对机器通过的结果继续执行人工复核；记录异步交付边界"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def env_bool(name: str, default: bool = False) -> bool:
    return os.environ.get(name, "1" if default else "0").lower().strip() in {
        "1",
        "true",
        "yes",
        "on",
    }


def enabled() -> bool:
    return env_bool("COMFY_CONTENT_MODERATION_ENABLED")


def required(job: dict[str, Any]) -> bool:
    return job.get("content_moderation", {}).get("required") is True


def released(job: dict[str, Any]) -> bool:
    """Output auditing is asynchronous; an explicit rejection revokes delivery.

    The persisted required flag survives configuration changes. Already-sent
    callbacks and downloaded bytes cannot be recalled by this gate.
    """
    if not required(job):
        return True
    record = job["content_moderation"]
    return (
        job.get("status") == "completed"
        and not record.get("delivery_revoked")
        and record.get("input", {}).get("action") in {"allow", "manual_allow"}
        and record.get("state") != "blocked"
    )


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def output_manifest_matches(job: dict[str, Any], job_dir: Path) -> bool:
    if not required(job):
        return True
    if not released(job):
        return False
    expected = job["content_moderation"].get("delivery_files", [])
    if not expected or [item.get("filename") for item in expected] != [
        item.get("filename") for item in job.get("outputs", [])
    ]:
        return False
    try:
        root = job_dir.resolve()
        return all(
            isinstance(item.get("filename"), str)
            and Path(item["filename"]).name == item["filename"]
            and (root / item["filename"]).resolve().parent == root
            and sha256(root / item["filename"]) == item.get("sha256")
            for item in expected
        )
    except (OSError, KeyError, TypeError):
        return False


class ModerationHold(RuntimeError):
    """The worker must persist this hold without invoking generation recovery."""


class ModerationError(RuntimeError):
    pass


class ContentReviewClient:
    def __init__(self) -> None:
        self.base_url = os.environ.get("COMFY_CONTENT_REVIEW_URL", "http://127.0.0.1:8040").rstrip(
            "/"
        )
        self.timeout = max(5.0, float(os.environ.get("COMFY_CONTENT_REVIEW_TIMEOUT", "300")))

    def health(self) -> dict[str, Any]:
        with requests.Session() as session:
            session.trust_env = False
            response = session.get(f"{self.base_url}/health", timeout=(5, 15))
            response.raise_for_status()
            body = response.json()
            host = urlsplit(self.base_url).hostname
            for item in body.get("downstream", {}).values():
                url = item.get("url", "")
                if urlsplit(url).hostname != host:
                    continue
                try:
                    status = session.get(url, timeout=(5, 15))
                    status.raise_for_status()
                    item["runtime"] = status.json()
                except (requests.RequestException, ValueError):
                    item["runtime"] = {"status": "unavailable"}
            return body

    def audit(
        self,
        *,
        job_id: str,
        stage: str,
        role: str,
        path: Path | None = None,
        text: str | None = None,
    ) -> dict[str, Any]:
        content_id = f"comfy_{job_id}_{stage}_{uuid4().hex[:10]}"
        fields = {"content_id": content_id, "user_id": f"comfy:{job_id}", "scene": "general"}
        conversion = None
        with tempfile.TemporaryDirectory(prefix="comfy-content-audit-") as temp_dir:
            audit_path = path
            if path is not None and path.suffix.lower() in {".gif", ".apng", ".webp", ".png"}:
                with Image.open(path) as image:
                    if getattr(image, "n_frames", 1) > 1:
                        width, height = image.size
                        audit_path = Path(temp_dir) / "animation.mp4"
                        subprocess.run(
                            [
                                "ffmpeg",
                                "-hide_banner",
                                "-loglevel",
                                "error",
                                "-y",
                                "-f",
                                "lavfi",
                                "-i",
                                f"color=c=white:s={width}x{height}:r=30",
                                "-i",
                                str(path),
                                "-filter_complex",
                                "[0:v][1:v]overlay=shortest=1:format=auto,format=yuv420p",
                                "-an",
                                "-c:v",
                                "libx264",
                                "-preset",
                                "ultrafast",
                                "-crf",
                                "18",
                                str(audit_path),
                            ],
                            capture_output=True,
                            timeout=120,
                            check=True,
                        )
                        conversion = "animated_media_to_white_background_video"
            with requests.Session() as session:
                session.trust_env = False
                if audit_path is None:
                    response = session.post(
                        f"{self.base_url}/v1/moderation",
                        params={"brief": "false"},
                        json={**fields, "content_type": "text", "text": text},
                        timeout=(5, self.timeout),
                    )
                else:
                    fields["content_type"] = (
                        "video" if audit_path.suffix.lower() in VIDEO_SUFFIXES else "image"
                    )
                    with audit_path.open("rb") as stream:
                        response = session.post(
                            f"{self.base_url}/v1/moderation",
                            params={"brief": "false"},
                            data=fields,
                            files={
                                "file": (
                                    audit_path.name,
                                    stream,
                                    mimetypes.guess_type(audit_path.name)[0]
                                    or "application/octet-stream",
                                )
                            },
                            timeout=(5, self.timeout),
                        )
                response.raise_for_status()
                body = response.json()
        if (
            not isinstance(body, dict)
            or body.get("action") not in {"allow", "review", "block"}
            or body.get("risk_level") not in {"none", "low", "medium", "high"}
            or body.get("content_id") != content_id
            or not body.get("policy_version")
        ):
            raise ModerationError("invalid_provider_response")
        if body["action"] == "allow" and body["risk_level"] in {"medium", "high"}:
            raise ModerationError("inconsistent_provider_response")
        labels = [str(label) for label in body.get("labels", [])]
        partial = any(label.endswith(("_error", "_unavailable", "_unchecked")) for label in labels)
        if partial and body["action"] != "block":
            raise ModerationError("provider_partial_audit")
        return {
            "content_id": content_id,
            "trace_id": body.get("trace_id"),
            "stage": stage,
            "role": role,
            "action": body["action"],
            "risk_level": body["risk_level"],
            "labels": labels,
            "partial_audit": partial,
            "scores": body.get("scores", {}),
            "provider_policy_version": str(body["policy_version"]),
            "model_status": body.get("model_status", {}),
            "conversion": conversion,
            "elapsed_ms": body.get("elapsed_ms"),
            "checked_at": utc_now(),
        }


class ModerationController:
    """Coordinates audit stages with the local-video facade's persistent jobs."""

    def __init__(self, api: Any, client: ContentReviewClient | None = None) -> None:
        self.api = api
        self.client = client or ContentReviewClient()
        self.lock = threading.RLock()
        self.job_locks: dict[str, threading.RLock] = {}
        self.output_queue: queue.Queue[str] = queue.Queue()
        self.output_pending: set[str] = set()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def job_lock(self, job_id: str) -> threading.RLock:
        with self.lock:
            return self.job_locks.setdefault(job_id, threading.RLock())

    def prepare(self, job: dict[str, Any]) -> None:
        if enabled() and not required(job):
            job["content_moderation"] = {
                "required": True,
                "policy_version": POLICY_VERSION,
                "state": "input_pending",
                "require_output_review": env_bool(
                    "COMFY_CONTENT_MODERATION_REQUIRE_OUTPUT_REVIEW", True
                ),
                "delivery_mode": "async_output",
                "delivery_revoked": False,
                "created_at": utc_now(),
                "events": [],
            }

    def persist_animated_upload(self, job_dir: Path, role: str, raw: bytes) -> str | None:
        if not enabled():
            return None
        with Image.open(io.BytesIO(raw)) as source:
            if getattr(source, "n_frames", 1) < 2:
                return None
            suffix = {"GIF": "gif", "PNG": "png", "WEBP": "webp"}.get(source.format)
            if suffix is None:
                raise ValueError("Unsupported animated upload format.")
        filename = f"moderation_{role}_upload.{suffix}"
        (job_dir / filename).write_bytes(raw)
        return filename

    def raw_job(self, job_id: str) -> dict[str, Any]:
        self.api.get_job(job_id)
        with self.api.JOBS_LOCK:
            return copy.deepcopy(self.api.JOBS[job_id])

    def assets(
        self, job: dict[str, Any], stage: str
    ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        root = (self.api.OUTPUT_ROOT / job["job_id"]).resolve()
        files: list[dict[str, str]] = []
        texts: list[dict[str, str]] = []
        if stage == "input":
            views = job.get("uploaded_identity_views") or {}
            names = [
                (role, record.get("moderation_filename") or record.get("source_filename"))
                for role, record in views.items()
            ]
            if not names and job.get("moderation_input_filename"):
                names = [("front", job["moderation_input_filename"])]
            if not names and (root / "input_identity_source.png").is_file():
                names = [("front", "input_identity_source.png")]
            if not names and (root / "input.png").is_file():
                names = [("front", "input.png")]
            for role in ("prompt_original", "prompt", "poses_original", "poses"):
                value = job.get(role)
                text = (
                    "\n".join(str(item) for item in value)
                    if isinstance(value, list)
                    else str(value or "")
                )
                if text.strip() and text not in [item["text"] for item in texts]:
                    texts.append(
                        {
                            "role": role,
                            "text": text,
                            "sha256": hashlib.sha256(text.encode()).hexdigest(),
                        }
                    )
        else:
            names = [
                (str(output.get("type") or "output"), output.get("filename"))
                for output in job.get("outputs", [])
            ]
            if not names:
                raise ModerationError("missing_output_manifest")
        for role, filename in names:
            if not isinstance(filename, str) or Path(filename).name != filename:
                raise ModerationError("invalid_asset_path")
            path = root / filename
            if path.resolve().parent != root or not path.is_file():
                raise ModerationError("missing_or_unsafe_asset")
            files.append({"role": str(role), "filename": filename, "sha256": sha256(path)})
        if stage == "input" and not files and job.get("uploaded_identity_views"):
            raise ModerationError("missing_input_media")
        if stage == "output" and not any(
            Path(item["filename"]).suffix.lower() in MEDIA_SUFFIXES for item in files
        ):
            raise ModerationError("missing_output_media")
        return files, texts

    def revision(self, files: list[dict[str, str]], texts: list[dict[str, str]]) -> str:
        return digest(
            {
                "policy": POLICY_VERSION,
                "files": files,
                "texts": [{"role": item["role"], "sha256": item["sha256"]} for item in texts],
            }
        )

    def save(self, job_id: str, record: dict[str, Any], **changes: Any) -> None:
        self.api.raise_if_cancelled(job_id)
        record["updated_at"] = utc_now()
        self.api.write_json_atomic(
            self.api.OUTPUT_ROOT / job_id / "content_moderation.json", record
        )
        self.api.update_job(job_id, content_moderation=record, **changes)

    def check(self, job: dict[str, Any], stage: str) -> dict[str, Any]:
        job_id = job["job_id"]
        record = copy.deepcopy(job["content_moderation"])
        checks: list[dict[str, Any]] = []
        files: list[dict[str, str]] = []
        revision = None
        try:
            files, texts = self.assets(job, stage)
            revision = self.revision(files, texts)
            cached = record.get(stage) or {}
            if cached.get("revision") == revision and cached.get("action") in {
                "allow",
                "manual_allow",
            }:
                return record
            self.save(
                job_id,
                {**record, "state": f"{stage}_auditing"},
                **({"stage": f"content_moderation_{stage}"} if stage == "input" else {}),
            )
            root = self.api.OUTPUT_ROOT / job_id
            for item in files:
                self.api.raise_if_cancelled(job_id)
                if Path(item["filename"]).suffix.lower() not in MEDIA_SUFFIXES:
                    # Diagnostic JSON is hash-bound, not passed to a media detector.
                    continue
                result = self.client.audit(
                    job_id=job_id, stage=stage, role=item["role"], path=root / item["filename"]
                )
                checks.append({**result, "filename": item["filename"], "sha256": item["sha256"]})
            for item in texts:
                self.api.raise_if_cancelled(job_id)
                checks.append(
                    {
                        **self.client.audit(
                            job_id=job_id, stage=stage, role=item["role"], text=item["text"]
                        ),
                        "sha256": item["sha256"],
                    }
                )
            if self.revision(*self.assets(job, stage)) != revision:
                raise ModerationError("asset_changed_during_audit")
            action = (
                "block"
                if any(item["action"] == "block" for item in checks)
                else "review"
                if any(item["action"] == "review" for item in checks)
                else "allow"
            )
            if stage == "output" and action == "allow" and record.get("require_output_review"):
                action = "review"
        except self.api.JobCancelled:
            raise
        except Exception as exc:  # Includes codec/network/provider errors; never default-allow.
            action = "block" if any(item.get("action") == "block" for item in checks) else "error"
            error = str(exc) if isinstance(exc, ModerationError) else type(exc).__name__
        else:
            error = None
        record[stage] = {
            "action": action,
            "revision": revision,
            "files": files,
            "checks": checks,
            "error": error,
            "checked_at": utc_now(),
        }
        record["revision"] = revision
        record["active_stage"] = stage
        record["events"].append(
            {
                "event_id": uuid4().hex,
                "type": "machine_audit",
                "stage": stage,
                "action": action,
                "revision": revision,
                "files": files,
                "checks": checks,
                "error": error,
                "at": utc_now(),
            }
        )
        record["state"] = (
            "input_passed"
            if stage == "input" and action == "allow"
            else "released"
            if action == "allow"
            else "pending_review"
            if action == "review"
            else "blocked"
            if action == "block"
            else "moderation_error"
        )
        if stage == "output" and action == "block":
            record["delivery_revoked"] = True
        elif action == "allow":
            record["delivery_revoked"] = False
        changes: dict[str, Any] = {}
        if action != "allow" and (stage == "input" or action == "block"):
            changes = {
                "status": record["state"],
                "stage": f"content_moderation_{stage}_{record['state']}",
                "error": {
                    "review": "内容等待人工复核，暂不交付。",
                    "block": "内容审核未通过，已拦截。",
                    "error": "内容审核未完成，暂不交付，可重审。",
                }[action],
            }
        elif stage == "output" and action == "allow" and job.get("status") == "blocked":
            changes = {"status": "completed", "stage": "completed", "error": None}
        self.save(job_id, record, **changes)
        if action != "allow" and stage == "input":
            raise ModerationHold(changes["error"])
        return record

    def check_inputs(self, job_id: str) -> None:
        with self.job_lock(job_id):
            job = self.raw_job(job_id)
            if required(job):
                if job["content_moderation"].get("input", {}).get("action") in {
                    "block",
                    "manual_block",
                    "error",
                    "review",
                }:
                    raise ModerationHold("输入尚未通过审核；须通过人工复核或显式重审。")
                self.check(job, "input")

    def before_complete(self, job_id: str, changes: dict[str, Any]) -> None:
        with self.job_lock(job_id):
            job = self.raw_job(job_id)
            if not required(job):
                return
            job.update(changes)
            self.api.update_job(
                job_id,
                outputs=job.get("outputs", []),
                _content_moderation_completion=copy.deepcopy(changes),
                _content_moderation_generation_done=True,
            )
            self.check_inputs(job_id)
            record = self.raw_job(job_id)["content_moderation"]
            files, _ = self.assets(job, "output")
            record["delivery_files"] = files
            if record.get("state") != "released":
                record["state"] = "output_pending"
            self.save(job_id, record)

    def enqueue_output(self, job_id: str) -> None:
        with self.lock:
            job = self.raw_job(job_id)
            if (
                required(job)
                and job["content_moderation"].get("state") in {"output_pending", "output_auditing"}
                and job_id not in self.output_pending
            ):
                self.output_pending.add(job_id)
                self.output_queue.put_nowait(job_id)

    def check_outputs(self, job_id: str) -> None:
        with self.job_lock(job_id):
            job = self.raw_job(job_id)
            if (
                required(job)
                and not job.get("cancel_requested")
                and job["content_moderation"].get("state") in {"output_pending", "output_auditing"}
            ):
                self.check(job, "output")

    def output_worker(self) -> None:
        while not self.stop_event.is_set():
            try:
                job_id = self.output_queue.get(timeout=1)
            except queue.Empty:
                continue
            try:
                self.check_outputs(job_id)
            except self.api.JobCancelled:
                pass
            except Exception:
                self.api.LOGGER.exception("Output audit worker failed for %s", job_id)
            finally:
                with self.lock:
                    self.output_pending.discard(job_id)
                self.output_queue.task_done()

    def start(self) -> None:
        self.stop_event.clear()
        with self.api.JOBS_LOCK:
            restored = copy.deepcopy(list(self.api.JOBS.values()))
        for job in restored:
            if job.get("_content_moderation_generation_done"):
                self.enqueue_output(job["job_id"])
        self.thread = threading.Thread(
            target=self.output_worker, name="content-moderation-worker", daemon=True
        )
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)

    def decide(
        self,
        job_id: str,
        *,
        decision: str,
        reviewer: str,
        note: str,
        revision: str,
        review_origin: str = "human",
    ) -> dict[str, Any]:
        with self.job_lock(job_id):
            job = self.raw_job(job_id)
            record = copy.deepcopy(job.get("content_moderation") or {})
            if decision not in {"approve", "reject"} or not reviewer.strip() or not note.strip():
                raise ValueError("审核操作、复核人和说明不能为空。")
            if not required(job) or revision != record.get("revision"):
                raise ValueError("审核记录已变化，请刷新后重新检查。")
            if job.get("cancel_requested") or job.get("status") == "cancelled":
                raise ValueError("已取消的任务不能通过复核。")
            stage = str(record.get("active_stage") or "input")
            if decision == "approve":
                if record.get("state") != "pending_review":
                    raise ValueError("只能通过待人工复核内容，机器拒绝或审核异常须重审。")
                if self.revision(*self.assets(job, stage)) != revision:
                    raise ValueError("文件内容已变化，须重新审核。")
                record[stage]["action"] = "manual_allow"
                record["state"] = "input_passed" if stage == "input" else "released"
                record["delivery_revoked"] = False
            else:
                if (
                    record.get("state") not in {"pending_review", "released", "input_passed"}
                    or stage not in record
                ):
                    raise ValueError("当前状态不能进行人工驳回，请等待审核或重审。")
                record[stage]["action"] = "manual_block"
                record["state"] = "blocked"
                record["delivery_revoked"] = True
            record["events"].append(
                {
                    "event_id": uuid4().hex,
                    "type": "verification_review"
                    if review_origin == "automated_test"
                    else "manual_review",
                    "review_origin": review_origin,
                    "reviewer": reviewer[:120],
                    "note": note[:2000],
                    "decision": decision,
                    "stage": stage,
                    "revision": revision,
                    "at": utc_now(),
                }
            )
            if decision == "reject":
                self.save(
                    job_id,
                    record,
                    status="blocked",
                    stage="content_moderation_manual_reject",
                    error="人工审核未通过，已停止交付。",
                    finished_at=utc_now(),
                )
            elif stage == "output":
                self.save(job_id, record)
                self.api.update_job(
                    job_id,
                    **job.get(
                        "_content_moderation_completion",
                        {"status": "completed", "stage": "completed", "progress": 100},
                    ),
                )
            else:
                if self.api.JOB_QUEUE.full():
                    raise ValueError("生成队列已满，请稍后重试。")
                self.save(job_id, record, status="queued", stage="queued", error=None)
                self.api.JOB_QUEUE.put_nowait(job_id)
            return self.api.get_job(job_id)

    def retry(self, job_id: str) -> dict[str, Any]:
        with self.job_lock(job_id):
            job = self.raw_job(job_id)
            if (
                not required(job)
                or job["content_moderation"].get("state") not in HOLD_STATUSES
                or job.get("cancel_requested")
            ):
                raise ValueError("只能重审未取消的待复核、已拦截或审核异常任务。")
            record = copy.deepcopy(job["content_moderation"])
            stage = record["active_stage"]
            if self.api.JOB_QUEUE.full() and not (
                stage == "output" and job.get("_content_moderation_generation_done")
            ):
                raise ValueError("生成队列已满，请稍后重试。")
            record.pop(stage, None)
            record["state"] = f"{stage}_pending"
            record["events"].append(
                {
                    "event_id": uuid4().hex,
                    "type": "machine_reaudit_requested",
                    "stage": stage,
                    "at": utc_now(),
                }
            )
            if stage == "output" and job.get("_content_moderation_generation_done"):
                self.save(job_id, record)
                self.enqueue_output(job_id)
            else:
                self.save(
                    job_id,
                    record,
                    status="queued",
                    stage=f"content_moderation_{stage}_retry",
                    error=None,
                )
                self.api.JOB_QUEUE.put_nowait(job_id)
            return self.api.get_job(job_id)

    def file_allowed(self, job: dict[str, Any], filename: str) -> bool:
        if not required(job):
            return True
        if not released(job):
            return False
        try:
            files, _ = self.assets(job, "output")
            expected = job["content_moderation"]["delivery_files"]
            return files == expected and any(item["filename"] == filename for item in files)
        except (ModerationError, OSError):
            return False
