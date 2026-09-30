"""Authenticated history-page proxies; reviewer credentials never reach the browser."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

import requests
from fastapi import Depends, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask


class HistoryReview(BaseModel):
    decision: Literal["approve", "reject"]
    note: str = Field(min_length=1, max_length=2000)
    revision: str = Field(min_length=1, max_length=64)


def review_headers() -> dict[str, str]:
    key = os.environ.get("COMFY_CONTENT_REVIEW_ADMIN_KEY", "").strip()
    if not key:
        raise HTTPException(503, "人工复核凭证尚未配置。")
    return {"X-Review-Key": key}


def register_platform_moderation(app: Any, platform: Any) -> None:
    def csrf(x_requested_with: str | None = Header(default=None)) -> None:
        if x_requested_with != "XMLHttpRequest":
            raise HTTPException(403, "缺少复核操作请求标识。")

    def upstream(
        path: str,
        *,
        method: str = "GET",
        body: dict[str, Any] | None = None,
        privileged: bool = True,
        stream: bool = False,
    ) -> requests.Response:
        headers = review_headers() if privileged else platform.video_headers()
        try:
            response = requests.request(
                method,
                f"{platform.LOCAL_API_URL}/api/local-video/{path}",
                headers=headers,
                json=body,
                stream=stream,
                timeout=(5, 600),
            )
        except requests.RequestException as exc:
            raise HTTPException(502, f"审核服务暂不可用：{type(exc).__name__}") from exc
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", "审核请求失败。")
            except ValueError:
                detail = "审核请求失败。"
            response.close()
            raise HTTPException(response.status_code, detail)
        return response

    def json_proxy(response: requests.Response) -> JSONResponse:
        try:
            return JSONResponse(response.json(), headers=platform.NO_STORE_HEADERS)
        finally:
            response.close()

    @app.get("/api/moderation/rules")
    def rules(_: None = Depends(platform.require_auth)) -> JSONResponse:
        return json_proxy(upstream("moderation/rules", privileged=False))

    @app.get("/api/jobs/video/{job_id}/moderation")
    def record(job_id: str, _: None = Depends(platform.require_auth)) -> JSONResponse:
        return json_proxy(upstream(f"jobs/{quote(job_id, safe='')}/moderation"))

    @app.post("/api/jobs/video/{job_id}/moderation/review")
    def review(
        job_id: str,
        body: HistoryReview,
        _: None = Depends(platform.require_auth),
        _csrf: None = Depends(csrf),
    ) -> JSONResponse:
        data = {**body.model_dump(), "reviewer": f"platform:{platform.PLATFORM_USER}"}
        return json_proxy(
            upstream(f"jobs/{quote(job_id, safe='')}/moderation/review", method="POST", body=data)
        )

    @app.post("/api/jobs/video/{job_id}/moderation/retry")
    def retry(
        job_id: str, _: None = Depends(platform.require_auth), _csrf: None = Depends(csrf)
    ) -> JSONResponse:
        return json_proxy(
            upstream(f"jobs/{quote(job_id, safe='')}/moderation/retry", method="POST")
        )

    @app.get("/api/jobs/video/{job_id}/review-files/{filename}")
    def media(
        job_id: str, filename: str, _: None = Depends(platform.require_auth)
    ) -> StreamingResponse:
        if Path(filename).name != filename:
            raise HTTPException(400, "Invalid filename.")
        response = upstream(
            f"jobs/{quote(job_id, safe='')}/review-files/{quote(filename, safe='')}", stream=True
        )
        return StreamingResponse(
            platform.iter_remote(response),
            media_type=response.headers.get("Content-Type", "application/octet-stream"),
            headers={
                **platform.NO_STORE_HEADERS,
                "Content-Disposition": "inline",
                "X-Content-Type-Options": "nosniff",
            },
            background=BackgroundTask(response.close),
        )
