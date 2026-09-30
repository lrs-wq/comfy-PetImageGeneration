"""Async delivery and later interception with simulated provider decisions."""

import copy
import json
import queue
import threading
from unittest.mock import Mock

import pytest
import requests
from fastapi.testclient import TestClient
from PIL import Image

from src.comfy_pet_pipeline.api import local_video as api
from src.comfy_pet_pipeline.api import platform
from src.comfy_pet_pipeline.api.content_moderation import (
    ContentReviewClient,
    ModerationController,
    ModerationError,
    ModerationHold,
    released,
)
from src.comfy_pet_pipeline.api.job_callback import JobCallbackReporter, build_callback_payload


@pytest.fixture
def moderated_job(monkeypatch, tmp_path):
    monkeypatch.setenv("COMFY_CONTENT_MODERATION_ENABLED", "1")
    monkeypatch.setenv("COMFY_CONTENT_MODERATION_REQUIRE_OUTPUT_REVIEW", "0")
    monkeypatch.setenv("COMFY_CONTENT_REVIEW_ADMIN_KEY", "test-review-key")
    monkeypatch.setattr(api, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(api, "JOBS", {})
    monkeypatch.setattr(api, "JOB_QUEUE", queue.Queue(maxsize=8))
    monkeypatch.setattr(api, "STOP_EVENT", threading.Event())
    monkeypatch.setattr(api, "report_job_callback", Mock())
    monkeypatch.setattr(api, "API_KEY", "test-api-key")
    client = Mock()
    client.audit.return_value = {
        "action": "allow",
        "risk_level": "none",
        "labels": [],
        "provider_policy_version": "simulated-test-policy",
        "model_status": {"test_double": True},
    }
    controller = ModerationController(api, client)
    monkeypatch.setattr(api, "CONTENT_MODERATION", controller)
    job = {
        "job_id": "test-moderation",
        "status": "queued",
        "stage": "queued",
        "outputs": [],
        "generation_mode": "subject_replacement",
        "callback_enabled": False,
        "cancel_requested": False,
        "prompt_original": "一只可爱的猫",
        "uploaded_identity_views": {"front": {"source_filename": "input_identity_source.png"}},
    }
    root = tmp_path / job["job_id"]
    root.mkdir()
    Image.new("RGB", (32, 32), "white").save(root / "input_identity_source.png")
    Image.new("RGB", (32, 32), "orange").save(root / "result.png")
    controller.prepare(job)
    api.JOBS[job["job_id"]] = job
    api.persist_job(job["job_id"])
    return job, controller, client, root


def complete(job):
    return api.update_job(
        job["job_id"],
        status="completed",
        stage="completed",
        progress=100,
        outputs=[{"filename": "result.png", "type": "image", "primary": True}],
    )


def test_worker_does_not_generate_or_recover_blocked_input(moderated_job, monkeypatch):
    job, controller, client, root = moderated_job

    def blocked(**kwargs):
        api.STOP_EVENT.set()
        return {"action": "block", "risk_level": "high", "labels": ["simulated"]}

    client.audit.side_effect = blocked
    generate, recover = Mock(), Mock()
    monkeypatch.setattr(api, "run_job", generate)
    monkeypatch.setattr(api, "try_recover_subject_replacement_output", recover)
    api.JOB_QUEUE.put(job["job_id"])
    api.worker_loop()
    assert job["status"] == "blocked"
    generate.assert_not_called()
    recover.assert_not_called()
    assert json.loads((root / "content_moderation.json").read_text())["input"]["action"] == "block"


@pytest.mark.parametrize("outcome", ["block", "review", "timeout"])
def test_async_output_delivers_immediately_then_intercepts_explicit_block(moderated_job, outcome):
    job, controller, client, root = moderated_job
    controller.check_inputs(job["job_id"])
    if outcome == "timeout":
        client.audit.side_effect = requests.Timeout("simulated upstream timeout")
    else:
        client.audit.return_value = {
            "action": outcome,
            "risk_level": "high" if outcome == "block" else "medium",
        }
    complete(job)
    # Completion, downloads and callbacks precede all output provider calls.
    assert job["status"] == "completed" and released(job)
    assert controller.file_allowed(job, "result.png")
    assert "gifs" in build_callback_payload(job, "https://example.test")
    browser = TestClient(api.app)
    path = f"/api/local-video/jobs/{job['job_id']}"
    assert (
        browser.get(f"{path}/files/result.png", headers={"X-API-Key": "test-api-key"}).status_code
        == 200
    )
    controller.check_outputs(job["job_id"])
    assert (
        job["content_moderation"]["state"]
        == {"block": "blocked", "review": "pending_review", "timeout": "moderation_error"}[outcome]
    )
    assert released(job) is (outcome != "block")
    if outcome != "block":
        assert job["stage"] == "completed"  # No duplicate completion callback for audit progress.
    assert controller.file_allowed(job, "result.png") is (outcome != "block")
    assert browser.get(
        f"{path}/files/result.png", headers={"X-API-Key": "test-api-key"}
    ).status_code == (403 if outcome == "block" else 200)
    assert bool(api.public_job(job)["outputs"]) is (outcome != "block")
    assert ("gifs" in build_callback_payload(job, "https://example.test")) is (outcome != "block")
    assert (root / "result.png").is_file()
    assert (
        browser.get(
            f"{path}/review-files/result.png", headers={"X-API-Key": "test-api-key"}
        ).status_code
        == 401
    )
    assert (
        browser.get(
            f"{path}/review-files/result.png", headers={"X-Review-Key": "test-review-key"}
        ).status_code
        == 200
    )


def test_default_human_review_and_approval_bound_to_files(moderated_job):
    job, controller, client, root = moderated_job
    job["content_moderation"]["require_output_review"] = True
    complete(job)
    controller.check_outputs(job["job_id"])
    record = job["content_moderation"]
    assert record["output"]["checks"][0]["action"] == "allow"
    result = controller.decide(
        job["job_id"],
        decision="approve",
        reviewer="test-reviewer",
        note="Synthetic test approval",
        revision=record["revision"],
    )
    assert result["status"] == "completed" and released(job)
    assert controller.file_allowed(job, "result.png")
    assert job["content_moderation"]["events"][-1]["reviewer"] == "test-reviewer"
    Image.new("RGB", (32, 32), "blue").save(root / "result.png")
    assert not controller.file_allowed(job, "result.png")


def test_approval_rejects_stale_revision_and_changed_file(moderated_job):
    job, controller, client, root = moderated_job
    job["content_moderation"]["require_output_review"] = True
    complete(job)
    controller.check_outputs(job["job_id"])
    params = dict(decision="approve", reviewer="test-reviewer", note="Synthetic test")
    with pytest.raises(ValueError, match="已变化"):
        controller.decide(job["job_id"], revision="stale", **params)
    revision = job["content_moderation"]["revision"]
    Image.new("RGB", (32, 32), "blue").save(root / "result.png")
    with pytest.raises(ValueError, match="文件内容已变化"):
        controller.decide(job["job_id"], revision=revision, **params)
    assert job["status"] == "completed"
    assert job["content_moderation"]["state"] == "pending_review"


def test_machine_block_cannot_be_manually_approved(moderated_job):
    job, controller, client, root = moderated_job
    client.audit.return_value = {"action": "block", "risk_level": "high"}
    with pytest.raises(ModerationHold):
        controller.check_inputs(job["job_id"])
    with pytest.raises(ValueError, match="只能通过待人工复核"):
        controller.decide(
            job["job_id"],
            decision="approve",
            reviewer="test",
            note="test",
            revision=job["content_moderation"]["revision"],
        )


def test_input_approval_requeues_but_cancelled_job_cannot_be_approved(moderated_job):
    job, controller, client, root = moderated_job
    client.audit.return_value = {"action": "review", "risk_level": "medium"}
    with pytest.raises(ModerationHold):
        controller.check_inputs(job["job_id"])
    revision = job["content_moderation"]["revision"]
    job["cancel_requested"] = True
    with pytest.raises(ValueError, match="已取消"):
        controller.decide(
            job["job_id"], decision="approve", reviewer="test", note="test", revision=revision
        )
    job["cancel_requested"] = False
    controller.decide(
        job["job_id"], decision="approve", reviewer="test", note="test", revision=revision
    )
    assert job["status"] == "queued"
    assert api.JOB_QUEUE.get_nowait() == job["job_id"]


def test_retry_output_reuses_generation_and_retains_audit_history(moderated_job, monkeypatch):
    job, controller, client, root = moderated_job
    job["content_moderation"]["require_output_review"] = True
    complete(job)
    controller.check_outputs(job["job_id"])
    previous_events = copy.deepcopy(job["content_moderation"]["events"])
    controller.retry(job["job_id"])
    generate = Mock()
    monkeypatch.setattr(api, "run_job", generate)
    assert job["job_id"] in list(controller.output_queue.queue)
    assert api.JOB_QUEUE.empty()
    controller.check_outputs(job["job_id"])
    generate.assert_not_called()
    assert job["content_moderation"]["state"] == "pending_review"
    assert job["content_moderation"]["events"][: len(previous_events)] == previous_events


def test_degraded_completion_and_restart_cannot_override_hold(moderated_job, monkeypatch):
    job, controller, client, root = moderated_job
    controller.check_inputs(job["job_id"])
    client.audit.return_value = {"action": "block", "risk_level": "high"}
    job["status"] = "running"
    api.persist_job(job["job_id"])

    def fallback(*args):
        complete(job)
        return True

    monkeypatch.setattr(api, "recover_subject_replacement_job_with_fallback", fallback)
    api.restore_jobs()
    assert job["job_id"] not in list(api.JOB_QUEUE.queue)
    assert api.JOBS[job["job_id"]]["status"] == "completed"
    controller.check_outputs(job["job_id"])
    assert api.JOBS[job["job_id"]]["status"] == "blocked"


def test_delayed_callback_is_revoked_when_files_change_or_record_disappears(moderated_job):
    job, controller, client, root = moderated_job
    complete(job)
    posted = Mock(return_value=Mock(status_code=200, json=lambda: {"code": 200}))
    reporter = JobCallbackReporter(
        callback_url="https://example.test/callback",
        api_key="test",
        public_base_url="https://example.test",
        output_root=root.parent,
        request_post=posted,
    )
    event = {
        "job_id": job["job_id"],
        "content_moderation_required": True,
        "payload": {
            "key": job["job_id"],
            "status": "completed",
            "gifs": {"eating": "https://example.test/file"},
        },
    }
    Image.new("RGB", (32, 32), "black").save(root / "result.png")
    reporter._post_event(event)
    assert posted.call_args.kwargs["json"]["status"] == "failed"
    assert "gifs" not in posted.call_args.kwargs["json"]
    (root / "status.json").unlink()
    event["payload"] = {"status": "completed", "gifs": {}}
    reporter._post_event(event)
    assert posted.call_args.kwargs["json"]["status"] == "failed"


def test_manual_rejection_revokes_previously_released_download(moderated_job):
    job, controller, client, root = moderated_job
    complete(job)
    controller.check_outputs(job["job_id"])
    controller.decide(
        job["job_id"],
        decision="reject",
        reviewer="test-reviewer",
        note="Synthetic rejection",
        revision=job["content_moderation"]["revision"],
    )
    assert not released(job) and job["status"] == "blocked"
    assert not controller.file_allowed(job, "result.png")


def test_required_flag_survives_switch_off_and_legacy_not_fabricated(moderated_job, monkeypatch):
    job, controller, client, root = moderated_job
    monkeypatch.setenv("COMFY_CONTENT_MODERATION_ENABLED", "0")
    assert not released(job)
    legacy = {"job_id": "legacy", "status": "completed", "outputs": []}
    controller.prepare(legacy)
    assert "content_moderation" not in legacy
    assert released(legacy)


def test_backend_reviewer_auth_and_platform_account_binding(moderated_job, monkeypatch):
    job, controller, client, root = moderated_job
    backend = TestClient(api.app)
    path = f"/api/local-video/jobs/{job['job_id']}/moderation"
    assert backend.get(path).status_code == 401
    assert backend.get(path, headers={"X-API-Key": "test-api-key"}).status_code == 401
    assert backend.get(path, headers={"X-Review-Key": "test-review-key"}).status_code == 200
    platform.app.dependency_overrides[platform.require_auth] = lambda: None
    response = Mock(status_code=200, json=lambda: {"status": "pending_review"})
    request = Mock(return_value=response)
    monkeypatch.setattr(platform.requests, "request", request)
    browser = TestClient(platform.app)
    review_path = f"/api/jobs/video/{job['job_id']}/moderation/review"
    body = {
        "decision": "reject",
        "note": "Synthetic test",
        "revision": "abc",
        "reviewer": "spoofed-account",
    }
    try:
        assert browser.post(review_path, json=body).status_code == 403
        assert (
            browser.post(
                review_path, json=body, headers={"X-Requested-With": "XMLHttpRequest"}
            ).status_code
            == 200
        )
        assert request.call_args.kwargs["json"]["reviewer"] == f"platform:{platform.PLATFORM_USER}"
        assert "test-review-key" not in browser.get("/history").text
    finally:
        platform.app.dependency_overrides.clear()


def test_history_uses_protected_previews_and_labels_absent_records(moderated_job):
    job, controller, client, root = moderated_job
    job["outputs"] = [{"filename": "result.png", "type": "image"}]
    item = platform.history_item(job, root, "pipeline", "2026-09-30")
    assert item["content_moderation"]["state"] == "input_pending"
    assert len(item["images"]) == 2
    assert all("/review-files/" in media["download_url"] for media in item["images"])
    legacy = platform.history_item({"job_id": "legacy"}, root, "legacy", "2026-09-30")
    assert legacy["content_moderation"] is None


@pytest.mark.parametrize(
    "bad",
    [
        {"action": "allow", "risk_level": "high"},
        {"action": "allow", "risk_level": "none", "labels": ["ocr_unavailable"]},
        {"action": "unexpected", "risk_level": "none"},
    ],
)
def test_provider_invalid_or_partial_response_fails_closed(monkeypatch, bad):
    session = Mock()

    def post(*args, **kwargs):
        return Mock(
            json=lambda: {
                "content_id": kwargs["json"]["content_id"],
                "policy_version": "test",
                **bad,
            }
        )

    session.post.side_effect = post
    context = Mock(__enter__=lambda _: session, __exit__=lambda *args: None)
    monkeypatch.setattr(requests, "Session", lambda: context)
    with pytest.raises(ModerationError):
        ContentReviewClient().audit(
            job_id="test", stage="input", role="text", text="safe synthetic test"
        )


def test_provider_protocol_uses_full_response_and_animation_video(monkeypatch, tmp_path):
    session = Mock()
    calls = []

    def post(*args, **kwargs):
        calls.append(kwargs)
        fields = kwargs.get("data") or kwargs["json"]
        return Mock(
            json=lambda: {
                "content_id": fields["content_id"],
                "policy_version": "test",
                "action": "allow",
                "risk_level": "none",
            }
        )

    session.post.side_effect = post
    context = Mock(__enter__=lambda _: session, __exit__=lambda *args: None)
    monkeypatch.setattr(requests, "Session", lambda: context)
    path = tmp_path / "animated.gif"
    frames = [Image.new("RGB", (32, 32), color) for color in ("white", "red", "blue")]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=100, loop=0)

    def convert(args, **kwargs):
        from pathlib import Path

        Path(args[-1]).write_bytes(b"synthetic video bytes")

    monkeypatch.setattr("src.comfy_pet_pipeline.api.content_moderation.subprocess.run", convert)
    result = ContentReviewClient().audit(job_id="test", stage="output", role="gif", path=path)
    assert calls[0]["data"]["content_type"] == "video"
    assert calls[0]["params"] == {"brief": "false"}
    assert result["conversion"] == "animated_media_to_white_background_video"


def test_output_auditing_does_not_block_other_jobs(moderated_job, monkeypatch):
    job, controller, client, root = moderated_job
    complete(job)
    entered, finish = threading.Event(), threading.Event()

    def slow_audit(**kwargs):
        entered.set()
        assert finish.wait(timeout=10)
        return {"action": "allow", "risk_level": "none"}

    client.audit.side_effect = slow_audit
    audit_thread = threading.Thread(target=controller.check_outputs, args=(job["job_id"],))
    audit_thread.start()
    assert entered.wait(timeout=2)
    second_job = {"job_id": "unrelated-legacy", "status": "running", "outputs": []}
    api.JOBS[second_job["job_id"]] = second_job
    done = threading.Event()

    def complete_other():
        api.update_job(second_job["job_id"], status="completed")
        done.set()

    thread = threading.Thread(target=complete_other)
    thread.start()
    try:
        assert done.wait(timeout=1), "Other completion waited on unrelated output auditor"
        assert controller.file_allowed(job, "result.png")
    finally:
        finish.set()
        audit_thread.join(timeout=5)
        thread.join(timeout=5)


def test_restart_resumes_async_audit_without_regeneration(moderated_job, monkeypatch):
    job, controller, client, root = moderated_job
    complete(job)
    restored = ModerationController(api, client)
    monkeypatch.setattr(api, "CONTENT_MODERATION", restored)
    generation = Mock()
    monkeypatch.setattr(api, "run_job", generation)
    restored.start()
    try:
        restored.output_queue.join()
        assert job["content_moderation"]["state"] == "released"
        generation.assert_not_called()
        assert api.JOB_QUEUE.empty()
    finally:
        restored.stop()


def test_animated_input_is_not_flattened_for_moderation(moderated_job):
    import io

    job, controller, client, root = moderated_job
    buffer = io.BytesIO()
    frames = [Image.new("RGB", (32, 32), color) for color in ("white", "blue")]
    frames[0].save(
        buffer, format="GIF", save_all=True, append_images=frames[1:], duration=500, loop=0
    )
    filename = controller.persist_animated_upload(root, "front", buffer.getvalue())
    job["uploaded_identity_views"]["front"]["moderation_filename"] = filename
    files, _ = controller.assets(job, "input")
    assert files[0]["filename"] == "moderation_front_upload.gif"
    with Image.open(root / filename) as image:
        assert image.n_frames == 2
    controller.check_inputs(job["job_id"])
    assert client.audit.call_args_list[0].kwargs["path"].name.endswith(".gif")


def test_manual_input_rejection_cannot_be_overridden_by_completion(moderated_job):
    job, controller, client, root = moderated_job
    controller.check_inputs(job["job_id"])
    controller.decide(
        job["job_id"],
        decision="reject",
        reviewer="test",
        note="Synthetic input rejection",
        revision=job["content_moderation"]["revision"],
    )
    previous_calls = client.audit.call_count
    with pytest.raises(ModerationHold):
        complete(job)
    assert client.audit.call_count == previous_calls
    assert job["status"] == "blocked" and not released(job)


def test_old_output_queue_entry_cannot_override_manual_rejection(moderated_job):
    job, controller, client, root = moderated_job
    complete(job)
    controller.check_outputs(job["job_id"])
    controller.decide(
        job["job_id"],
        decision="reject",
        reviewer="test",
        note="Synthetic output rejection",
        revision=job["content_moderation"]["revision"],
    )
    previous_calls = client.audit.call_count
    controller.check_outputs(job["job_id"])
    assert client.audit.call_count == previous_calls
    assert job["status"] == "blocked" and not released(job)


def test_public_gif_completion_does_not_wait_for_output_review(moderated_job):
    from starlette.requests import Request

    from src.comfy_pet_pipeline.api import gif_api

    job, controller, client, root = moderated_job
    controller.check_inputs(job["job_id"])
    api.update_job(
        job["job_id"],
        status="completed",
        stage="completed",
        progress=100,
        outputs=[
            {
                "filename": "result.png",
                "type": "transparent_gif",
                "action": "eating",
                "primary": True,
            },
            {
                "filename": "result.png",
                "type": "transparent_gif",
                "action": "grooming",
                "primary": True,
            },
        ],
    )
    public = api.public_job(job)
    request = Request(
        {
            "type": "http",
            "scheme": "https",
            "server": ("example.test", 443),
            "root_path": "",
            "path": "/api",
            "headers": [],
        }
    )
    result = gif_api.public_job(public, request=request, key=job["job_id"])
    assert result["status"] == "completed"
    assert result["gifs"]["eating"] and result["gifs"]["grooming"]
    assert result["content_moderation"]["state"] == "output_pending"


def test_internal_generation_reads_intermediate_outputs_while_business_api_hides_them(
    moderated_job,
):
    job, controller, client, root = moderated_job
    controller.check_inputs(job["job_id"])
    job.update(status="running", outputs=[{"filename": "result.png", "type": "cartoon_reference"}])
    assert api.get_job(job["job_id"])["outputs"][0]["filename"] == "result.png"
    browser = TestClient(api.app)
    result = browser.get(
        f"/api/local-video/jobs/{job['job_id']}", headers={"X-API-Key": "test-api-key"}
    )
    assert result.status_code == 200 and result.json()["outputs"] == []


def test_detected_block_is_preserved_when_later_asset_audit_errors(moderated_job):
    job, controller, client, root = moderated_job
    controller.check_inputs(job["job_id"])
    Image.new("RGB", (32, 32), "blue").save(root / "second.png")
    api.update_job(
        job["job_id"],
        status="completed",
        stage="completed",
        outputs=[
            {"filename": "result.png", "type": "image"},
            {"filename": "second.png", "type": "image"},
        ],
    )
    client.audit.side_effect = [
        {"action": "block", "risk_level": "high"},
        requests.Timeout("later synthetic error"),
    ]
    controller.check_outputs(job["job_id"])
    assert job["content_moderation"]["state"] == "blocked"
    assert job["content_moderation"]["output"]["error"] == "Timeout"
    assert not released(job)


def test_partial_provider_block_is_not_downgraded_to_error(monkeypatch):
    session = Mock()

    def post(*args, **kwargs):
        return Mock(
            json=lambda: {
                "content_id": kwargs["json"]["content_id"],
                "policy_version": "test",
                "action": "block",
                "risk_level": "high",
                "labels": ["fraud", "ocr_unavailable"],
            }
        )

    session.post.side_effect = post
    context = Mock(__enter__=lambda _: session, __exit__=lambda *args: None)
    monkeypatch.setattr(requests, "Session", lambda: context)
    result = ContentReviewClient().audit(
        job_id="test", stage="output", role="text", text="synthetic fixture"
    )
    assert result["action"] == "block" and result["partial_audit"] is True
