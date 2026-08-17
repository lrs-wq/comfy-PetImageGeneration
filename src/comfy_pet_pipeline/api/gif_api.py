from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask


APP_VERSION = "1.2.0"
INTERNAL_API_URL = os.getenv(
    "LOCAL_VIDEO_INTERNAL_URL", "http://127.0.0.1:8770"
).rstrip("/")
INTERNAL_API_KEY = os.getenv("LOCAL_VIDEO_API_KEY", "").strip()
PUBLIC_API_KEY = os.getenv("PET_GIF_API_KEY", INTERNAL_API_KEY).strip()
PUBLIC_BASE_URL = os.getenv("PET_GIF_PUBLIC_BASE_URL", "").strip().rstrip("/")
INTERNAL_JOBS_URL = f"{INTERNAL_API_URL}/api/local-video/jobs"
CAT_ACTIONS = ("eating", "grooming")
DOG_ACTIONS = ("eating", "standing")
ALL_ACTIONS = ("eating", "grooming", "standing")
# Upload keeps the legacy request payload; the internal binary classifier maps
# dog grooming to the formal standing action before the job is queued.
UPLOAD_ACTIONS = CAT_ACTIONS
KEY_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,96}$")


app = FastAPI(
    title="Cat and Dog Action GIF API",
    version=APP_VERSION,
    description=(
        "Submit one pet identity for forced cat/dog routing. Cats return eating and "
        "grooming; dogs return eating and standing with a temporary grooming alias."
    ),
)


def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    if not PUBLIC_API_KEY:
        raise HTTPException(status_code=503, detail="API key is not configured.")
    if not x_api_key or not secrets.compare_digest(x_api_key, PUBLIC_API_KEY):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing X-API-Key header.",
        )


def validate_key(key: str | None) -> str:
    normalized = (key or "").strip()
    if not KEY_PATTERN.fullmatch(normalized):
        raise HTTPException(status_code=422, detail="A valid query key is required.")
    return normalized


def internal_headers() -> dict[str, str]:
    if not INTERNAL_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Generation service credentials are not configured.",
        )
    return {"X-API-Key": INTERNAL_API_KEY}


def backend_detail(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        detail = payload.get("detail") or payload.get("error")
        if detail:
            return str(detail)[:600]
    text = (response.text or "").strip()
    return text[:600] or f"HTTP {response.status_code}"


def raise_backend_error(response: requests.Response, *, query: bool = False) -> None:
    if query and response.status_code == 404:
        public_status = 404
        prefix = "Task not found"
    elif response.status_code in {400, 413, 415, 422}:
        public_status = response.status_code
        prefix = "Generation request was rejected"
    elif response.status_code == 503:
        public_status = 503
        prefix = "Generation service is unavailable"
    else:
        public_status = 502
        prefix = "Generation service request failed"
    raise HTTPException(
        status_code=public_status,
        detail=f"{prefix}: {backend_detail(response)}",
    )


def request_base_url(request: Request) -> str:
    return PUBLIC_BASE_URL or str(request.base_url).rstrip("/")


def job_species(job: dict[str, Any]) -> str:
    species = str(job.get("species") or "").strip().lower()
    if species in {"cat", "dog"}:
        return species
    for output in job.get("outputs") or []:
        if isinstance(output, dict) and str(output.get("action") or "").lower() == "standing":
            return "dog"
    return "cat"


def required_actions(job: dict[str, Any]) -> tuple[str, ...]:
    return DOG_ACTIONS if job_species(job) == "dog" else CAT_ACTIONS


def output_gifs(job: dict[str, Any], base_url: str, key: str) -> dict[str, str | None]:
    candidates: dict[str, list[dict[str, Any]]] = {
        action: [] for action in ALL_ACTIONS
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

    urls: dict[str, str | None] = {}
    for action in ALL_ACTIONS:
        records = sorted(
            candidates[action],
            key=lambda item: bool(item.get("primary")),
            reverse=True,
        )
        if not records:
            urls[action] = None
            continue
        filename = str(records[0]["filename"])
        urls[action] = (
            f"{base_url}/file/{quote(key, safe='')}/{quote(filename, safe='')}"
        )
    if job_species(job) == "dog":
        standing_url = urls["standing"]
        return {
            "eating": urls["eating"],
            "standing": standing_url,
            "grooming": standing_url,
        }
    return {"eating": urls["eating"], "grooming": urls["grooming"]}


def public_job(job: dict[str, Any], request: Request, key: str) -> dict[str, Any]:
    raw_status = str(job.get("status") or "queued").strip().lower()
    status_aliases = {
        "pending": "queued",
        "processing": "running",
        "succeeded": "completed",
        "success": "completed",
        "error": "failed",
        "canceled": "cancelled",
    }
    normalized_status = status_aliases.get(raw_status, raw_status)
    if normalized_status not in {"queued", "running", "completed", "failed", "cancelled"}:
        normalized_status = "running"

    try:
        progress = round(min(100.0, max(0.0, float(job.get("progress") or 0.0))), 1)
    except (TypeError, ValueError):
        progress = 0.0

    gifs = output_gifs(job, request_base_url(request), key)
    error = job.get("error")
    stage = str(job.get("stage") or normalized_status)
    if normalized_status == "completed":
        missing = [
            action
            for action in required_actions(job)
            if not (
                gifs.get(action)
                or (action == "standing" and gifs.get("grooming"))
            )
        ]
        if missing:
            normalized_status = "failed"
            stage = "output_contract_failed"
            error = "Generation completed without required GIF outputs: " + ", ".join(
                missing
            )

    return {
        "key": key,
        "status": normalized_status,
        "progress": progress,
        "stage": stage,
        "action": job.get("current_action"),
        "species": job_species(job),
        "error": str(error) if error else None,
        "gifs": gifs,
    }


def upload_tuple(upload: UploadFile) -> tuple[str, Any, str]:
    return (
        upload.filename or "image",
        upload.file,
        upload.content_type or "application/octet-stream",
    )


async def close_uploads(*uploads: UploadFile | None) -> None:
    for upload in uploads:
        if upload is not None:
            await upload.close()


@app.get("/health", tags=["service"])
async def health() -> dict[str, Any]:
    backend = "unavailable"
    try:
        response = await asyncio.to_thread(
            requests.get,
            f"{INTERNAL_API_URL}/api/local-video/health",
            timeout=8,
        )
        if response.ok:
            backend = "ok"
    except requests.RequestException:
        pass
    configured = bool(PUBLIC_API_KEY and INTERNAL_API_KEY)
    return {
        "status": "ok" if configured and backend == "ok" else "degraded",
        "version": APP_VERSION,
        "backend": backend,
    }


@app.post("/api", tags=["jobs"], response_model=None)
async def api_gateway(
    request: Request,
    operation: str = Form(alias="type", description="upload or query"),
    key: str | None = Form(default=None),
    front: UploadFile | None = File(default=None),
    left: UploadFile | None = File(default=None),
    right: UploadFile | None = File(default=None),
    _auth: None = Depends(require_api_key),
) -> dict[str, Any] | JSONResponse:
    del _auth
    operation = operation.strip().lower()
    try:
        if operation == "upload":
            if front is None:
                raise HTTPException(
                    status_code=422,
                    detail="front is required for type=upload.",
                )
            if (left is None) != (right is None):
                raise HTTPException(
                    status_code=422,
                    detail="left and right must be uploaded together.",
                )
            if key:
                raise HTTPException(
                    status_code=422,
                    detail="key is only valid for type=query.",
                )

            files: dict[str, tuple[str, Any, str]] = {"image": upload_tuple(front)}
            if left is not None and right is not None:
                files["image_left"] = upload_tuple(left)
                files["image_right"] = upload_tuple(right)
            data = {
                "generation_mode": "subject_replacement",
                "request_kind": "video",
                "actions": json.dumps(list(UPLOAD_ACTIONS)),
                "resolution": "512x512",
                "output_format": "transparent_gif",
                "prompt": "",
            }
            try:
                response = await asyncio.to_thread(
                    requests.post,
                    INTERNAL_JOBS_URL,
                    headers={
                        **internal_headers(),
                        "X-Pet-Gif-Callback": "1",
                    },
                    data=data,
                    files=files,
                    timeout=(10, 120),
                )
            except requests.RequestException as exc:
                raise HTTPException(
                    status_code=502,
                    detail="Generation service is unreachable.",
                ) from exc
            if not response.ok:
                raise_backend_error(response)
            try:
                payload = response.json()
            except ValueError as exc:
                raise HTTPException(
                    status_code=502,
                    detail="Generation service returned an invalid response.",
                ) from exc
            job_key = validate_key(str(payload.get("job_id") or ""))
            return JSONResponse(
                status_code=202,
                content={"key": job_key, "status": str(payload.get("status") or "queued")},
            )

        if operation == "query":
            job_key = validate_key(key)
            if any(upload is not None for upload in (front, left, right)):
                raise HTTPException(
                    status_code=422,
                    detail="Images are only valid for type=upload.",
                )
            try:
                response = await asyncio.to_thread(
                    requests.get,
                    f"{INTERNAL_JOBS_URL}/{quote(job_key, safe='')}",
                    headers=internal_headers(),
                    timeout=(10, 30),
                )
            except requests.RequestException as exc:
                raise HTTPException(
                    status_code=502,
                    detail="Generation service is unreachable.",
                ) from exc
            if not response.ok:
                raise_backend_error(response, query=True)
            try:
                job = response.json()
            except ValueError as exc:
                raise HTTPException(
                    status_code=502,
                    detail="Generation service returned an invalid response.",
                ) from exc
            if not isinstance(job, dict):
                raise HTTPException(
                    status_code=502,
                    detail="Generation service returned an invalid response.",
                )
            return public_job(job, request, job_key)

        raise HTTPException(status_code=422, detail="type must be upload or query.")
    finally:
        await close_uploads(front, left, right)


@app.get("/file/{key}/{filename}", tags=["jobs"])
def download_gif(
    key: str,
    filename: str,
    _auth: None = Depends(require_api_key),
) -> StreamingResponse:
    del _auth
    job_key = validate_key(key)
    if not filename or Path(filename).name != filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    try:
        status_response = requests.get(
            f"{INTERNAL_JOBS_URL}/{quote(job_key, safe='')}",
            headers=internal_headers(),
            timeout=(10, 30),
        )
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail="Generation service is unreachable.",
        ) from exc
    if not status_response.ok:
        raise_backend_error(status_response, query=True)
    try:
        job = status_response.json()
    except ValueError as exc:
        raise HTTPException(
            status_code=502,
            detail="Generation service returned an invalid response.",
        ) from exc

    allowed = {
        str(output.get("filename"))
        for output in job.get("outputs") or []
        if isinstance(output, dict)
        and output.get("type") == "transparent_gif"
        and str(output.get("action") or "").lower() in ALL_ACTIONS
        and Path(str(output.get("filename") or "")).name
        == str(output.get("filename") or "")
    }
    if filename not in allowed:
        raise HTTPException(status_code=404, detail="GIF output not found.")

    try:
        response = requests.get(
            f"{INTERNAL_JOBS_URL}/{quote(job_key, safe='')}/files/{quote(filename, safe='')}",
            headers=internal_headers(),
            timeout=(10, 300),
            stream=True,
        )
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail="Generation service is unreachable.",
        ) from exc
    if not response.ok:
        detail = backend_detail(response)
        response.close()
        raise HTTPException(
            status_code=404 if response.status_code == 404 else 502,
            detail=f"GIF download failed: {detail}",
        )

    headers = {
        "Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename, safe='')}",
        "Cache-Control": "private, max-age=3600",
    }
    content_length = response.headers.get("Content-Length")
    if content_length:
        headers["Content-Length"] = content_length
    return StreamingResponse(
        response.iter_content(chunk_size=1024 * 1024),
        media_type=response.headers.get("Content-Type", "image/gif"),
        headers=headers,
        background=BackgroundTask(response.close),
    )
