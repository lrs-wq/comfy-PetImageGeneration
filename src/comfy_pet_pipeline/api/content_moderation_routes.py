"""Privileged moderation APIs used by the existing generation history page."""

from __future__ import annotations

import os
import secrets
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .content_moderation import POLICY_VERSION, RULES, enabled, env_bool


class ReviewDecision(BaseModel):
    decision: Literal["approve", "reject"]
    reviewer: str = Field(min_length=1, max_length=120)
    note: str = Field(min_length=1, max_length=2000)
    revision: str = Field(min_length=1, max_length=64)
    review_origin: Literal["human", "automated_test"] = "human"


def register_moderation_routes(app: Any, api: Any) -> None:
    prefix = api.API_PREFIX

    def require_reviewer(
        x_review_key: str | None = Header(default=None, alias="X-Review-Key"),
    ) -> None:
        key = os.environ.get("COMFY_CONTENT_REVIEW_ADMIN_KEY", "").strip()
        if not key:
            raise HTTPException(503, "人工复核凭证尚未配置。")
        if not x_review_key or not secrets.compare_digest(key, x_review_key):
            raise HTTPException(401, "Invalid reviewer credential.")

    @app.get(f"{prefix}/moderation/rules")
    def rules(_: None = Depends(api.require_api_key)) -> dict[str, Any]:
        try:
            provider = api.CONTENT_MODERATION.client.health()
        except Exception as exc:
            provider = {"status": "unavailable", "error": type(exc).__name__}
        return {
            "policy_version": POLICY_VERSION,
            "enabled": enabled(),
            "delivery_mode": "async_output",
            "delivery_boundary": (
                "生成完成即下载和成功回调；异步审核拒绝后拦截后续交付，不能追回已交付内容"
            ),
            "require_output_review": env_bool(
                "COMFY_CONTENT_MODERATION_REQUIRE_OUTPUT_REVIEW", True
            ),
            "rules": [
                {"rule_id": rule_id, "description": description, "enabled": enabled()}
                for rule_id, description in RULES
            ],
            "provider": provider,
            "provider_thresholds": "upstream_runtime_thresholds_not_exposed_by_existing_api",
            "animated_media_audit": (
                "white-background MP4 through Content Review video sampling; "
                "not every-frame detection"
            ),
        }

    @app.get(f"{prefix}/jobs/{{job_id}}/moderation")
    def audit_record(job_id: str, _: None = Depends(require_reviewer)) -> dict[str, Any]:
        job = api.CONTENT_MODERATION.raw_job(job_id)
        return {
            "job_id": job_id,
            "status": job.get("status"),
            "content_moderation": job.get("content_moderation"),
            "historical_record_without_audit": not bool(job.get("content_moderation")),
        }

    @app.post(f"{prefix}/jobs/{{job_id}}/moderation/review")
    def review(
        job_id: str, body: ReviewDecision, _: None = Depends(require_reviewer)
    ) -> dict[str, Any]:
        if body.review_origin == "automated_test" and not env_bool(
            "COMFY_CONTENT_REVIEW_ALLOW_TEST_DECISIONS"
        ):
            raise HTTPException(403, "Automated test decisions are disabled.")
        try:
            return api.CONTENT_MODERATION.decide(job_id, **body.model_dump())
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post(f"{prefix}/jobs/{{job_id}}/moderation/retry")
    def retry(job_id: str, _: None = Depends(require_reviewer)) -> dict[str, Any]:
        try:
            return api.CONTENT_MODERATION.retry(job_id)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get(f"{prefix}/jobs/{{job_id}}/review-files/{{filename}}")
    def reviewer_file(
        job_id: str, filename: str, _: None = Depends(require_reviewer)
    ) -> FileResponse:
        job = api.CONTENT_MODERATION.raw_job(job_id)
        allowed = {item.get("filename") for item in job.get("outputs", [])}
        for record in (job.get("uploaded_identity_views") or {}).values():
            allowed.update(
                {
                    record.get("source_filename"),
                    record.get("input_filename"),
                    record.get("moderation_filename"),
                }
            )
        allowed.update(
            {"input.png", "input_identity_source.png", job.get("moderation_input_filename")}
        )
        root = (api.OUTPUT_ROOT / job_id).resolve()
        path = root / filename
        if (
            Path(filename).name != filename
            or filename not in allowed
            or path.resolve().parent != root
            or not path.is_file()
        ):
            raise HTTPException(404, "Review asset not found.")
        return FileResponse(
            path, headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}
        )
