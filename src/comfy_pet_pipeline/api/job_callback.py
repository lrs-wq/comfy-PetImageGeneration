"""Durable, non-blocking lifecycle callbacks for GIF generation jobs."""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote
from uuid import uuid4

import requests


EXPECTED_GIF_ACTIONS = ("eating", "grooming")
TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
WAITING_STAGES = {"queued", "waiting_for_gpu"}
CALLBACK_OUTBOX_DIRECTORY = "callback_outbox"
CALLBACK_DEAD_LETTER_DIRECTORY = "callback_dead_letter"
CALLBACK_STATE_FILENAME = "callback_state.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _normalized_status(job: dict[str, Any]) -> str:
    raw_status = str(job.get("status") or "queued").strip().lower()
    aliases = {
        "pending": "queued",
        "processing": "running",
        "succeeded": "completed",
        "success": "completed",
        "error": "failed",
        "canceled": "cancelled",
    }
    status = aliases.get(raw_status, raw_status)
    if status not in {"queued", "running", *TERMINAL_STATUSES}:
        return "running"
    return status


def _normalized_progress(job: dict[str, Any], status: str, stage: str) -> float:
    if status == "queued" or stage in WAITING_STAGES:
        return 0.0
    if status == "completed":
        return 100.0
    try:
        return round(min(100.0, max(0.0, float(job.get("progress") or 0.0))), 1)
    except (TypeError, ValueError):
        return 0.0


def _gif_urls(
    job: dict[str, Any],
    public_base_url: str,
    job_id: str,
) -> dict[str, str]:
    candidates: dict[str, list[dict[str, Any]]] = {
        action: [] for action in EXPECTED_GIF_ACTIONS
    }
    for output in job.get("outputs") or []:
        if not isinstance(output, dict):
            continue
        action = str(output.get("action") or "").strip().lower()
        filename = str(output.get("filename") or "").strip()
        if (
            output.get("type") != "transparent_gif"
            or action not in candidates
            or not filename
            or Path(filename).name != filename
        ):
            continue
        candidates[action].append(output)

    gifs: dict[str, str] = {}
    for action in EXPECTED_GIF_ACTIONS:
        records = sorted(
            candidates[action],
            key=lambda item: bool(item.get("primary")),
            reverse=True,
        )
        if not records:
            continue
        filename = str(records[0]["filename"])
        gifs[action] = (
            f"{public_base_url}/file/{quote(job_id, safe='')}/"
            f"{quote(filename, safe='')}"
        )
    return gifs


def build_callback_payload(
    job: dict[str, Any],
    public_base_url: str,
) -> dict[str, Any]:
    """Build the external callback contract from an internal job snapshot."""

    job_id = str(job.get("job_id") or "").strip()
    if not job_id:
        raise ValueError("Callback job snapshot is missing job_id.")
    status = _normalized_status(job)
    stage = str(job.get("stage") or status).strip() or status
    payload: dict[str, Any] = {
        "key": job_id,
        "status": status,
        "progress": _normalized_progress(job, status, stage),
        "stage": stage,
    }
    if status == "completed":
        payload["gifs"] = _gif_urls(job, public_base_url.rstrip("/"), job_id)
    return payload


def callback_signature(job: dict[str, Any], payload: dict[str, Any]) -> str:
    """Identify lifecycle transitions while ignoring same-stage progress refreshes."""

    return json.dumps(
        [
            payload["status"],
            payload["stage"],
            str(job.get("current_action") or ""),
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )


@dataclass(frozen=True)
class CallbackDeliveryError(RuntimeError):
    message: str
    permanent: bool = False
    defer_until_restart: bool = False

    def __str__(self) -> str:
        return self.message


class JobCallbackReporter:
    """Persist callbacks before a single daemon thread delivers them in order."""

    def __init__(
        self,
        *,
        callback_url: str,
        api_key: str,
        public_base_url: str,
        output_root: Path,
        logger: logging.Logger | None = None,
        request_post: Callable[..., requests.Response] = requests.post,
        request_timeout: tuple[float, float] = (5.0, 20.0),
        retry_delays: tuple[float, ...] = (1.0, 3.0, 10.0, 30.0),
    ) -> None:
        self.callback_url = callback_url.strip()
        self.api_key = api_key.strip()
        self.public_base_url = public_base_url.strip().rstrip("/")
        self.output_root = Path(output_root)
        self.logger = logger or logging.getLogger(__name__)
        self.request_post = request_post
        self.request_timeout = request_timeout
        self.retry_delays = retry_delays or (1.0,)
        self._queue: queue.Queue[Path | None] = queue.Queue()
        self._state_lock = threading.RLock()
        self._lifecycle_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_signatures: dict[str, str] = {}

    @classmethod
    def from_environment(
        cls,
        *,
        output_root: Path,
        logger: logging.Logger | None = None,
    ) -> JobCallbackReporter:
        return cls(
            callback_url=os.getenv("PET_GIF_CALLBACK_URL", ""),
            api_key=os.getenv("PET_GIF_CALLBACK_API_KEY", ""),
            public_base_url=os.getenv("PET_GIF_PUBLIC_BASE_URL", ""),
            output_root=output_root,
            logger=logger,
        )

    @property
    def enabled(self) -> bool:
        return bool(self.callback_url and self.api_key and self.public_base_url)

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def _job_directory(self, job_id: str) -> Path:
        return self.output_root / job_id

    def _state_path(self, job_id: str) -> Path:
        return self._job_directory(job_id) / CALLBACK_STATE_FILENAME

    def _outbox_directory(self, job_id: str) -> Path:
        return self._job_directory(job_id) / CALLBACK_OUTBOX_DIRECTORY

    def _load_state(self, job_id: str) -> dict[str, Any]:
        path = self._state_path(job_id)
        if not path.is_file():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.logger.exception("Could not read callback state for %s", job_id)
            return {}
        return value if isinstance(value, dict) else {}

    def _known_signature(self, job_id: str) -> str | None:
        cached = self._last_signatures.get(job_id)
        if cached is not None:
            return cached
        state = self._load_state(job_id)
        signature = str(state.get("last_enqueued_signature") or "") or None
        if signature is not None:
            self._last_signatures[job_id] = signature
        return signature

    def enqueue(self, job: dict[str, Any], *, force: bool = False) -> bool:
        """Persist a transition before returning; network delivery is asynchronous."""

        if not self.enabled:
            return False
        payload = build_callback_payload(job, self.public_base_url)
        job_id = str(payload["key"])
        signature = callback_signature(job, payload)
        with self._state_lock:
            if not force and self._known_signature(job_id) == signature:
                return False
            event = {
                "schema_version": 1,
                "job_id": job_id,
                "signature": signature,
                "created_at": utc_now(),
                "payload": payload,
            }
            event_path = self._outbox_directory(job_id) / (
                f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
                f"-{uuid4().hex}.json"
            )
            _write_json_atomic(event_path, event)
            state = self._load_state(job_id)
            state.update(
                schema_version=1,
                last_enqueued_signature=signature,
                last_enqueued_at=event["created_at"],
                last_enqueued_payload=payload,
            )
            _write_json_atomic(self._state_path(job_id), state)
            self._last_signatures[job_id] = signature
            if self.running:
                self._queue.put(event_path)
        return True

    def start(self) -> None:
        if not self.enabled:
            self.logger.warning(
                "GIF callback reporting is disabled because its URL, API key, "
                "or public base URL is not configured."
            )
            return
        with self._lifecycle_lock:
            if self.running:
                return
            self._stop_event.clear()
            pending = sorted(
                self.output_root.glob(f"*/{CALLBACK_OUTBOX_DIRECTORY}/*.json")
            )
            for path in pending:
                self._queue.put(path)
            self._thread = threading.Thread(
                target=self._worker_loop,
                name="pet-gif-callback-reporter",
                daemon=True,
            )
            self._thread.start()
            self.logger.info(
                "GIF callback reporter started with %d pending event(s)",
                len(pending),
            )

    def stop(self, timeout: float = 5.0) -> None:
        with self._lifecycle_lock:
            thread = self._thread
            if thread is None:
                return
            self._stop_event.set()
            self._queue.put(None)
        thread.join(timeout=timeout)
        with self._lifecycle_lock:
            if self._thread is thread and not thread.is_alive():
                self._thread = None

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                event_path = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                if event_path is None:
                    return
                self._deliver_until_resolved(event_path)
            except Exception:  # noqa: BLE001 - callback failures never stop generation
                self.logger.exception("Unexpected GIF callback reporter failure")
            finally:
                self._queue.task_done()

    def _load_event(self, path: Path) -> dict[str, Any]:
        try:
            event = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CallbackDeliveryError(
                f"Could not read callback outbox event {path}: {exc}",
                permanent=True,
            ) from exc
        if not isinstance(event, dict) or not isinstance(event.get("payload"), dict):
            raise CallbackDeliveryError(
                f"Callback outbox event {path} has an invalid payload.",
                permanent=True,
            )
        return event

    def _post_event(self, event: dict[str, Any]) -> dict[str, Any] | None:
        try:
            response = self.request_post(
                self.callback_url,
                headers={
                    "Content-Type": "application/json",
                    "X-API-Key": self.api_key,
                },
                json=event["payload"],
                timeout=self.request_timeout,
            )
        except requests.RequestException as exc:
            raise CallbackDeliveryError(str(exc)) from exc
        if response.status_code >= 400:
            text = (response.text or "").strip()[:500]
            authentication_rejected = response.status_code in {401, 403}
            permanent = (
                400 <= response.status_code < 500
                and response.status_code not in {408, 429}
                and not authentication_rejected
            )
            raise CallbackDeliveryError(
                f"HTTP {response.status_code}: {text or 'callback rejected'}",
                permanent=permanent,
                defer_until_restart=authentication_rejected,
            )
        try:
            response_payload = response.json()
        except ValueError:
            response_payload = None
        if isinstance(response_payload, dict) and "code" in response_payload:
            try:
                response_code = int(response_payload["code"])
            except (TypeError, ValueError):
                response_code = 500
            if response_code != 200:
                raise CallbackDeliveryError(
                    "Callback response returned code "
                    f"{response_payload.get('code')}: "
                    f"{str(response_payload.get('message') or '')[:500]}",
                    permanent=(
                        400 <= response_code < 500
                        and response_code not in {401, 403, 408, 429}
                    ),
                    defer_until_restart=response_code in {401, 403},
                )
        return response_payload if isinstance(response_payload, dict) else None

    def _record_delivery(
        self,
        event: dict[str, Any],
        response_payload: dict[str, Any] | None,
    ) -> None:
        job_id = str(event["job_id"])
        with self._state_lock:
            state = self._load_state(job_id)
            state.update(
                last_delivered_signature=event["signature"],
                last_delivered_at=utc_now(),
                last_delivered_payload=event["payload"],
                last_response=response_payload,
                last_error=None,
                delivery_deferred_until_restart=False,
            )
            _write_json_atomic(self._state_path(job_id), state)

    def _move_to_dead_letter(
        self,
        event_path: Path,
        event: dict[str, Any] | None,
        error: CallbackDeliveryError,
    ) -> None:
        job_id = str((event or {}).get("job_id") or event_path.parents[1].name)
        dead_letter = (
            self._job_directory(job_id)
            / CALLBACK_DEAD_LETTER_DIRECTORY
            / event_path.name
        )
        dead_letter.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(event_path, dead_letter)
        except OSError:
            self.logger.exception("Could not move rejected callback to %s", dead_letter)
        with self._state_lock:
            state = self._load_state(job_id)
            state.update(last_error=str(error), last_error_at=utc_now())
            _write_json_atomic(self._state_path(job_id), state)

    def _record_deferred_error(
        self,
        event: dict[str, Any],
        error: CallbackDeliveryError,
    ) -> None:
        job_id = str(event["job_id"])
        with self._state_lock:
            state = self._load_state(job_id)
            state.update(
                last_error=str(error),
                last_error_at=utc_now(),
                delivery_deferred_until_restart=True,
            )
            _write_json_atomic(self._state_path(job_id), state)

    def _deliver_until_resolved(self, event_path: Path) -> None:
        event: dict[str, Any] | None = None
        try:
            event = self._load_event(event_path)
        except CallbackDeliveryError as exc:
            self._move_to_dead_letter(event_path, event, exc)
            return

        attempt = 0
        while not self._stop_event.is_set():
            try:
                response_payload = self._post_event(event)
            except CallbackDeliveryError as exc:
                attempt += 1
                if exc.defer_until_restart:
                    self.logger.error(
                        "GIF callback authentication rejected for %s (%s/%s); "
                        "event retained for restart after credential correction: %s",
                        event["job_id"],
                        event["payload"].get("status"),
                        event["payload"].get("stage"),
                        exc,
                    )
                    self._record_deferred_error(event, exc)
                    return
                if exc.permanent:
                    self.logger.error(
                        "GIF callback permanently rejected for %s (%s/%s): %s",
                        event["job_id"],
                        event["payload"].get("status"),
                        event["payload"].get("stage"),
                        exc,
                    )
                    self._move_to_dead_letter(event_path, event, exc)
                    return
                delay = self.retry_delays[min(attempt - 1, len(self.retry_delays) - 1)]
                self.logger.warning(
                    "GIF callback attempt %d failed for %s (%s/%s); retrying in %.1fs: %s",
                    attempt,
                    event["job_id"],
                    event["payload"].get("status"),
                    event["payload"].get("stage"),
                    delay,
                    exc,
                )
                if self._stop_event.wait(delay):
                    return
                continue

            self._record_delivery(event, response_payload)
            try:
                event_path.unlink()
            except FileNotFoundError:
                pass
            self.logger.info(
                "GIF callback delivered for %s (%s/%s)",
                event["job_id"],
                event["payload"].get("status"),
                event["payload"].get("stage"),
            )
            return
