"""Run historical inputs through the live API and assert only completion.

The stability gallery deliberately has no visual-quality expectations.  A case
passes when the API reaches ``completed`` and publishes one readable primary
artifact for every requested action.  Thumbnails are diagnostic evidence only.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

import requests
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .historical_gallery import (
    DEFAULT_DATA_ROOT,
    DEFAULT_MANIFEST,
    GalleryError,
    load_manifest,
    resolve_artifacts,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_API_ROOT = "http://127.0.0.1:8770/api/local-video"
DEFAULT_ENV_FILE = Path("/etc/comfy-pet-pipeline/api.env")
DEFAULT_REPORT_ROOT_RELATIVE = Path("outputs/regression_gallery/live")
TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
ACTIVE_STATUSES = {"queued", "running"}
PRIMARY_OUTPUT_TYPES = (
    "transparent_apng",
    "transparent_gif",
    "subject_replacement_video",
)
REQUIRED_POST_CHANGE_SMOKE_ACTIONS = {
    "cat": ("eating", "grooming"),
    "dog": ("eating", "standing"),
}


class StabilityError(RuntimeError):
    """Raised when the live success contract is not met."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def read_api_key(path: Path) -> str:
    if not path.is_file():
        raise StabilityError(f"API environment file is missing: {path}")
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() == "LOCAL_VIDEO_API_KEY":
            key = value.strip().strip('"').strip("'")
            if key:
                return key
    raise StabilityError(f"LOCAL_VIDEO_API_KEY is missing from {path}")


def git_revision() -> str:
    revision = subprocess.run(
        ["git", "rev-parse", "--short=10", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return (revision or "unknown") + ("-dirty" if dirty else "")


def create_report_directory(report_root: Path) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    stem = f"{timestamp}-{git_revision()}"
    report_root.mkdir(parents=True, exist_ok=True)
    for suffix in range(100):
        candidate = report_root / (stem if suffix == 0 else f"{stem}-{suffix:02d}")
        try:
            candidate.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return candidate
    raise StabilityError(f"Could not allocate a report directory below {report_root}")


def validate_default_smoke_matrix(cases: list[dict[str, Any]]) -> None:
    expected_species = set(REQUIRED_POST_CHANGE_SMOKE_ACTIONS)
    if len(cases) != len(expected_species):
        raise StabilityError(
            "Default smoke must select exactly one cat case and one dog case."
        )

    seen_species: set[str] = set()
    for case in cases:
        species = str(case.get("species") or "").strip().lower()
        if species not in expected_species:
            raise StabilityError(
                f"Default smoke selected unsupported species {species!r}."
            )
        if species in seen_species:
            raise StabilityError(
                f"Default smoke selected more than one {species} case."
            )
        expected_actions = list(REQUIRED_POST_CHANGE_SMOKE_ACTIONS[species])
        selected_actions = [
            str(action) for action in case.get("selected_actions", [])
        ]
        if selected_actions != expected_actions:
            raise StabilityError(
                f"Default smoke {species} case must select each required action "
                f"exactly once: expected={expected_actions}, actual={selected_actions}."
            )
        seen_species.add(species)

    missing_species = expected_species - seen_species
    if missing_species:
        raise StabilityError(
            "Default smoke is missing required species: "
            + ", ".join(sorted(missing_species))
        )


def stop_after_failed_case(
    *,
    suite: str,
    requested_ids: list[str] | None,
    keep_going: bool,
) -> bool:
    mandatory_default_matrix = suite == "smoke" and not requested_ids
    return not keep_going and not mandatory_default_matrix


def select_cases(
    manifest: dict[str, Any],
    *,
    suite: str,
    requested_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    canaries = manifest.get("canary_cases")
    if not isinstance(canaries, list) or not canaries:
        raise StabilityError("Historical gallery has no generation canaries.")
    by_id = {str(item.get("id") or ""): item for item in canaries}
    unknown = [item for item in requested_ids or [] if item not in by_id]
    if unknown:
        raise StabilityError("Unknown canary case(s): " + ", ".join(unknown))

    if requested_ids:
        selected = [by_id[item] for item in requested_ids]
    elif suite == "smoke":
        selected = [item for item in canaries if item.get("smoke_actions")]
    else:
        raise StabilityError(f"Unsupported stability suite: {suite}")
    if not selected:
        raise StabilityError(f"Stability suite {suite!r} selected no cases.")

    result: list[dict[str, Any]] = []
    for item in selected:
        available_actions = [str(action) for action in item.get("actions", [])]
        actions = (
            [str(action) for action in item.get("smoke_actions", [])]
            if not requested_ids
            else available_actions
        )
        if not actions or any(action not in available_actions for action in actions):
            raise StabilityError(
                f"Canary {item.get('id')} has an invalid action selection: {actions}"
            )
        result.append(dict(item) | {"selected_actions": actions})
    if suite == "smoke" and not requested_ids:
        validate_default_smoke_matrix(result)
    return result


def checked_json(response: requests.Response, *, operation: str) -> dict[str, Any]:
    try:
        response.raise_for_status()
    except requests.RequestException as exc:
        body = response.text[:1000]
        raise StabilityError(f"{operation} failed: {exc}; response={body!r}") from exc
    try:
        payload = response.json()
    except ValueError as exc:
        raise StabilityError(f"{operation} returned invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise StabilityError(f"{operation} returned a non-object JSON payload.")
    return payload


def preflight(
    session: requests.Session,
    api_root: str,
    headers: dict[str, str],
    *,
    allow_busy: bool,
) -> dict[str, Any]:
    health = checked_json(
        session.get(f"{api_root}/health", timeout=60),
        operation="health preflight",
    )
    issues: list[str] = []
    if health.get("status") != "ok":
        issues.append(f"status={health.get('status')!r}")
    if health.get("worker_alive") is not True:
        issues.append("worker_alive is not true")
    if health.get("api_key_configured") is not True:
        issues.append("api_key_configured is not true")
    if not isinstance(health.get("comfyui"), dict) or health["comfyui"].get("ok") is not True:
        issues.append("ComfyUI is not healthy")
    if not isinstance(health.get("models"), dict) or health["models"].get("ok") is not True:
        issues.append("required models are unavailable")
    if issues:
        raise StabilityError("Live stability preflight failed: " + "; ".join(issues))

    jobs = checked_json(
        session.get(f"{api_root}/jobs", headers=headers, params={"limit": 100}, timeout=60),
        operation="active-job preflight",
    )
    active = [
        {
            "job_id": item.get("job_id"),
            "status": item.get("status"),
            "stage": item.get("stage"),
        }
        for item in jobs.get("jobs", [])
        if isinstance(item, dict) and item.get("status") in ACTIVE_STATUSES
    ]
    if active and not allow_busy:
        raise StabilityError(
            "Live API already has active jobs; rerun after they finish or pass "
            f"--allow-busy explicitly: {active}"
        )
    return {
        "status": health.get("status"),
        "version": health.get("version"),
        "worker_alive": health.get("worker_alive"),
        "comfyui_ok": health.get("comfyui", {}).get("ok"),
        "models_ok": health.get("models", {}).get("ok"),
        "active_jobs_before_run": active,
    }


def submit_case(
    session: requests.Session,
    api_root: str,
    headers: dict[str, str],
    case: dict[str, Any],
    artifacts: dict[str, Path],
) -> dict[str, Any]:
    field_by_role = {"front": "image", "left": "image_left", "right": "image_right"}
    handles: list[Any] = []
    files: dict[str, tuple[str, Any, str]] = {}
    try:
        for role, artifact_id_raw in dict(case.get("views") or {}).items():
            if role not in field_by_role:
                raise StabilityError(f"Canary {case['id']} has unsupported view role {role!r}.")
            path = artifacts.get(str(artifact_id_raw))
            if path is None:
                raise StabilityError(
                    f"Canary {case['id']} references unknown artifact {artifact_id_raw!r}."
                )
            handle = path.open("rb")
            handles.append(handle)
            files[field_by_role[role]] = (path.name, handle, "image/png")
        response = session.post(
            f"{api_root}/jobs",
            headers=headers,
            files=files,
            data={
                "generation_mode": "subject_replacement",
                "actions": json.dumps(case["selected_actions"]),
                "subject_profile": str(case.get("subject_profile") or ""),
                "output_format": "transparent_gif",
                "prompt": "",
            },
            timeout=180,
        )
    finally:
        for handle in handles:
            handle.close()
    payload = checked_json(response, operation=f"submit canary {case['id']}")
    if response.status_code != 202 or not payload.get("job_id"):
        raise StabilityError(
            f"Canary {case['id']} was not accepted as a job: status={response.status_code}"
        )
    return payload


def poll_job(
    session: requests.Session,
    api_root: str,
    headers: dict[str, str],
    job_id: str,
    *,
    timeout_seconds: int,
    poll_seconds: float,
    on_observation: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    last_observation: tuple[Any, ...] | None = None
    consecutive_errors = 0
    while True:
        elapsed = time.monotonic() - started
        if elapsed > timeout_seconds:
            raise StabilityError(
                f"Job {job_id} exceeded the {timeout_seconds}-second stability timeout."
            )
        try:
            job = checked_json(
                session.get(f"{api_root}/jobs/{job_id}", headers=headers, timeout=60),
                operation=f"poll job {job_id}",
            )
        except (requests.RequestException, StabilityError):
            consecutive_errors += 1
            if consecutive_errors >= 5:
                raise
            time.sleep(poll_seconds)
            continue
        consecutive_errors = 0
        observation = (
            job.get("status"),
            job.get("stage"),
            job.get("progress"),
            job.get("current_action"),
        )
        if observation != last_observation:
            if on_observation is not None:
                on_observation(job)
            last_observation = observation
        if job.get("status") in TERMINAL_STATUSES:
            return job
        time.sleep(poll_seconds)


def _output_for_action(job: dict[str, Any], action: str) -> dict[str, Any] | None:
    outputs = [
        item
        for item in job.get("outputs", [])
        if isinstance(item, dict) and item.get("action") == action and item.get("filename")
    ]
    primary = [item for item in outputs if item.get("primary") is True]
    if primary:
        return primary[-1]
    for output_type in PRIMARY_OUTPUT_TYPES:
        matches = [item for item in outputs if item.get("type") == output_type]
        if matches:
            return matches[-1]
    return None


def _media_thumbnail(raw: bytes, filename: str) -> tuple[Image.Image | None, dict[str, Any]]:
    suffix = Path(filename).suffix.lower()
    digest = hashlib.sha256(raw).hexdigest()
    metadata: dict[str, Any] = {"bytes": len(raw), "sha256": digest}
    if not raw:
        raise StabilityError(f"Downloaded output is empty: {filename}")
    if suffix in {".png", ".gif", ".webp", ".jpg", ".jpeg"}:
        try:
            with Image.open(io.BytesIO(raw)) as probe:
                probe.verify()
            with Image.open(io.BytesIO(raw)) as source:
                source.seek(0)
                frame = ImageOps.exif_transpose(source).convert("RGBA")
                metadata.update(
                    width=frame.width,
                    height=frame.height,
                    format=source.format,
                    frames=int(getattr(source, "n_frames", 1)),
                )
                thumbnail = Image.new("RGBA", (256, 256), (245, 245, 245, 255))
                frame.thumbnail((244, 244), Image.Resampling.LANCZOS)
                thumbnail.alpha_composite(
                    frame,
                    ((thumbnail.width - frame.width) // 2, (thumbnail.height - frame.height) // 2),
                )
                return thumbnail.convert("RGB"), metadata
        except (OSError, ValueError) as exc:
            raise StabilityError(
                f"Output cannot be decoded as an image: {filename}: {exc}"
            ) from exc
    if suffix == ".mp4":
        if len(raw) < 12 or raw[4:8] != b"ftyp":
            raise StabilityError(f"Output is not a readable MP4 container: {filename}")
        metadata["format"] = "MP4"
        return None, metadata
    raise StabilityError(f"Unsupported primary output format: {filename}")


def validate_completed_job(
    session: requests.Session,
    api_root: str,
    headers: dict[str, str],
    case: dict[str, Any],
    job: dict[str, Any],
    report_dir: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    issues: list[str] = []
    if job.get("status") != "completed":
        issues.append(
            f"terminal status is {job.get('status')!r}, expected 'completed': "
            f"{str(job.get('error') or '')[:1000]}"
        )
    requested_actions = list(case["selected_actions"])
    if list(job.get("actions") or []) != requested_actions:
        issues.append(
            f"job actions are {job.get('actions')!r}, expected {requested_actions!r}"
        )

    validated: list[dict[str, Any]] = []
    thumbnail_dir = report_dir / "thumbnails"
    thumbnail_dir.mkdir(parents=True, exist_ok=True)
    for action in requested_actions:
        output = _output_for_action(job, action)
        if output is None:
            issues.append(f"{action}: no primary animation output was published")
            continue
        filename = str(output["filename"])
        try:
            response = session.get(
                f"{api_root}/jobs/{job['job_id']}/files/{quote(filename)}",
                headers=headers,
                timeout=180,
            )
            response.raise_for_status()
            thumbnail, metadata = _media_thumbnail(response.content, filename)
        except (requests.RequestException, StabilityError) as exc:
            issues.append(f"{action}: output {filename!r} is unreadable: {exc}")
            continue
        thumbnail_name = f"{case['id']}-{action}.png"
        thumbnail_path: Path | None = None
        if thumbnail is not None:
            thumbnail_path = thumbnail_dir / thumbnail_name
            thumbnail.save(thumbnail_path, format="PNG", optimize=True)
        validated.append(
            {
                "action": action,
                "type": output.get("type"),
                "filename": filename,
                "primary": output.get("primary") is True,
                "readable": True,
                "thumbnail": (
                    str(thumbnail_path.relative_to(report_dir)) if thumbnail_path else None
                ),
                **metadata,
            }
        )
    return validated, issues


def job_visual_diagnostics(job: dict[str, Any]) -> dict[str, Any]:
    """Keep posture evidence in the gallery without turning it into assertions."""

    scail_statuses = job.get("scail_video_temporal_identity_status")
    scail_statuses = scail_statuses if isinstance(scail_statuses, dict) else {}
    scail_attempts = job.get("scail_video_attempts")
    grooming_scail_attempts = (
        scail_attempts.get("grooming")
        if isinstance(scail_attempts, dict)
        and isinstance(scail_attempts.get("grooming"), list)
        else []
    )
    selected_scail_attempt = next(
        (
            item.get("attempt")
            for item in grooming_scail_attempts
            if isinstance(item, dict) and item.get("selected") is True
        ),
        None,
    )
    return {
        "qwen_first_frame_selected_attempt": job.get(
            "qwen_first_frame_best_effort_attempt"
        )
        or job.get("action_candidate_gate_selected_attempt"),
        "grooming_minimum_gate_status": job.get(
            "grooming_advisory_minimum_gate_status"
        ),
        "grooming_foreleg_topology_status": job.get(
            "grooming_foreleg_topology_status"
        ),
        "grooming_foreleg_topology_policy": job.get(
            "grooming_foreleg_topology_policy"
        ),
        "grooming_foreleg_topology_blocks_completion": job.get(
            "grooming_foreleg_topology_blocks_completion"
        ),
        "grooming_foreleg_topology_selected": job.get(
            "grooming_foreleg_topology_selected"
        ),
        "scail_grooming_status": scail_statuses.get("grooming"),
        "scail_grooming_selected_attempt": selected_scail_attempt,
    }


def write_report(report_dir: Path, report: dict[str, Any]) -> None:
    (report_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for base in (Path("/usr/share/fonts/truetype/dejavu"), Path("/usr/share/fonts/dejavu")):
        candidate = base / name
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def render_live_sheet(report_dir: Path, results: list[dict[str, Any]]) -> None:
    cells: list[dict[str, Any]] = []
    for result in results:
        by_action = {item["action"]: item for item in result.get("outputs", [])}
        for action in result.get("actions", []):
            cells.append(
                {
                    "case": result["id"],
                    "action": action,
                    "job_id": result.get("job_id"),
                    "passed": result.get("passed") and action in by_action,
                    "thumbnail": by_action.get(action, {}).get("thumbnail"),
                }
            )
    columns = 3
    cell_width, cell_height, header = 330, 360, 70
    rows = max(1, (len(cells) + columns - 1) // columns)
    canvas = Image.new("RGB", (columns * cell_width, header + rows * cell_height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (20, 18),
        "Live stability gallery — completion only",
        font=_font(25, bold=True),
        fill=(28, 28, 28),
    )
    for index, cell in enumerate(cells):
        x = (index % columns) * cell_width
        y = header + (index // columns) * cell_height
        accent = (20, 130, 80) if cell["passed"] else (198, 40, 40)
        draw.rectangle(
            (x + 8, y + 8, x + cell_width - 8, y + cell_height - 8),
            outline=accent,
            width=4,
        )
        thumb_raw = cell.get("thumbnail")
        if thumb_raw and (report_dir / thumb_raw).is_file():
            with Image.open(report_dir / thumb_raw) as source:
                thumb = source.convert("RGB")
            canvas.paste(thumb, (x + 37, y + 24))
        draw.text(
            (x + 20, y + 292),
            str(cell["case"])[:36],
            font=_font(14, bold=True),
            fill=(30, 30, 30),
        )
        draw.text(
            (x + 20, y + 316),
            f"{cell['action']} | {'PASS' if cell['passed'] else 'FAIL'}",
            font=_font(14),
            fill=accent,
        )
        draw.text(
            (x + 20, y + 338),
            str(cell.get("job_id") or "not submitted")[:38],
            font=_font(11),
            fill=(80, 80, 80),
        )
    canvas.save(report_dir / "live-gallery.png", format="PNG", optimize=True)


def write_html(report_dir: Path, report: dict[str, Any]) -> None:
    def diagnostic_summary(item: dict[str, Any]) -> str:
        diagnostics = item.get("diagnostics")
        if not isinstance(diagnostics, dict):
            return ""
        values = [
            (
                "first-frame",
                diagnostics.get("qwen_first_frame_selected_attempt"),
            ),
            (
                "foreleg",
                diagnostics.get("grooming_foreleg_topology_status"),
            ),
            ("SCAIL", diagnostics.get("scail_grooming_status")),
        ]
        return " · ".join(
            f"{label}={value}"
            for label, value in values
            if value is not None and value != ""
        )

    rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(item['id']))}</td>"
        f"<td>{html.escape(', '.join(item.get('actions', [])))}</td>"
        f"<td>{html.escape(str(item.get('job_id') or ''))}</td>"
        f"<td class=\"{'pass' if item.get('passed') else 'fail'}\">"
        f"{'PASS' if item.get('passed') else 'FAIL'}</td>"
        f"<td>{html.escape(diagnostic_summary(item))}</td>"
        f"<td>{html.escape('; '.join(item.get('issues', [])))}</td>"
        "</tr>"
        for item in report.get("cases", [])
    )
    document = f"""<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><title>Live stability gallery</title>
<style>
body{{font:15px system-ui,sans-serif;margin:32px;color:#202124}}img{{max-width:100%}}
table{{border-collapse:collapse;width:100%;margin-top:24px}}
th,td{{border:1px solid #ddd;padding:8px;text-align:left}}
.pass{{color:#087f4f;font-weight:700}}.fail{{color:#c62828;font-weight:700}}
code{{background:#f2f3f5;padding:2px 5px}}
</style>
<h1>Live stability gallery</h1>
<p>Revision <code>{html.escape(str(report['git_revision']))}</code> · suite
<code>{html.escape(str(report['suite']))}</code> ·
<strong>{report['summary']['passed']}/{report['summary']['total']} passed</strong></p>
<p>Only completion and readable per-action outputs are asserted. Visual quality is not graded.</p>
<img src="live-gallery.png" alt="Live stability gallery">
<table><thead><tr><th>Case</th><th>Actions</th><th>New job</th>
<th>Result</th><th>Diagnostic (non-blocking)</th><th>Issue</th></tr></thead>
<tbody>{rows}</tbody></table>
</html>"""
    (report_dir / "index.html").write_text(document, encoding="utf-8")


def run_live_gallery(
    *,
    manifest_path: Path,
    data_root: Path,
    api_root: str,
    env_file: Path,
    suite: str,
    requested_ids: list[str] | None,
    report_root: Path,
    output_dir: Path | None,
    timeout_seconds: int,
    poll_seconds: float,
    allow_busy: bool,
    keep_going: bool,
    resume_job_id: str | None = None,
    session: requests.Session | None = None,
) -> tuple[dict[str, Any], Path]:
    manifest = load_manifest(manifest_path)
    artifacts = resolve_artifacts(manifest, data_root)
    cases = select_cases(manifest, suite=suite, requested_ids=requested_ids)
    if resume_job_id and len(cases) != 1:
        raise StabilityError("--resume-job requires exactly one --case selection.")
    report_dir = output_dir or create_report_directory(report_root)
    report_dir.mkdir(parents=True, exist_ok=True)
    owned_session = session is None
    client = session or requests.Session()
    headers = {"X-API-Key": read_api_key(env_file)}
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "live-stability-gallery",
        "git_revision": git_revision(),
        "created_at": utc_now(),
        "finished_at": None,
        "suite": suite,
        "quality_assertions": False,
        "success_contract": [
            "terminal status is completed",
            "every requested action publishes a primary animation",
            "every selected animation is downloadable and format-readable",
        ],
        "cases": [],
        "summary": {"passed": 0, "failed": 0, "total": len(cases)},
    }
    try:
        report["preflight"] = preflight(
            client,
            api_root,
            headers,
            allow_busy=allow_busy or bool(resume_job_id),
        )
        write_report(report_dir, report)
        for case in cases:
            result: dict[str, Any] = {
                "id": case["id"],
                "label": case.get("label"),
                "source_job_id": case.get("source_job_id"),
                "species": case.get("species"),
                "actions": list(case["selected_actions"]),
                "job_id": None,
                "status": "not_submitted",
                "stage": None,
                "outputs": [],
                "diagnostics": {},
                "issues": [],
                "passed": False,
            }
            report["cases"].append(result)
            write_report(report_dir, report)
            try:
                submitted = (
                    {
                        "job_id": resume_job_id,
                        "status": "resuming",
                        "stage": "resuming",
                    }
                    if resume_job_id
                    else submit_case(client, api_root, headers, case, artifacts)
                )
                result.update(
                    job_id=submitted["job_id"],
                    status=submitted.get("status"),
                    stage=submitted.get("stage"),
                    resumed=bool(resume_job_id),
                )

                def observe(job: dict[str, Any]) -> None:
                    result.update(
                        status=job.get("status"),
                        stage=job.get("stage"),
                        progress=job.get("progress"),
                    )
                    print(
                        f"case={case['id']} job={job.get('job_id')} "
                        f"status={job.get('status')} stage={job.get('stage')} "
                        f"progress={job.get('progress')}",
                        flush=True,
                    )
                    write_report(report_dir, report)

                job = poll_job(
                    client,
                    api_root,
                    headers,
                    str(submitted["job_id"]),
                    timeout_seconds=timeout_seconds,
                    poll_seconds=poll_seconds,
                    on_observation=observe,
                )
                result.update(
                    status=job.get("status"),
                    stage=job.get("stage"),
                    progress=job.get("progress"),
                    diagnostics=job_visual_diagnostics(job),
                )
                outputs, issues = validate_completed_job(
                    client, api_root, headers, case, job, report_dir
                )
                result["outputs"] = outputs
                result["issues"] = issues
                result["passed"] = not issues
            except (OSError, requests.RequestException, StabilityError) as exc:
                result["issues"].append(str(exc))
                result["passed"] = False
            write_report(report_dir, report)
            if not result["passed"] and stop_after_failed_case(
                suite=suite,
                requested_ids=requested_ids,
                keep_going=keep_going,
            ):
                break
    finally:
        if owned_session:
            client.close()
    passed = sum(bool(item.get("passed")) for item in report["cases"])
    attempted = len(report["cases"])
    report["summary"] = {
        "passed": passed,
        "failed": attempted - passed,
        "not_run": len(cases) - attempted,
        "total": len(cases),
    }
    report["finished_at"] = utc_now()
    write_report(report_dir, report)
    render_live_sheet(report_dir, report["cases"])
    write_html(report_dir, report)
    return report, report_dir


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Run historical inputs through the live completion-only stability gallery."
    )
    result.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    result.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("PET_PIPELINE_DATA_ROOT", DEFAULT_DATA_ROOT)),
    )
    result.add_argument("--api-root", default=DEFAULT_API_ROOT)
    result.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    result.add_argument("--suite", choices=("smoke",), default="smoke")
    result.add_argument("--case", action="append", dest="case_ids")
    result.add_argument(
        "--resume-job",
        help="Resume polling one existing job; requires exactly one --case.",
    )
    result.add_argument("--report-root", type=Path)
    result.add_argument("--output-dir", type=Path)
    result.add_argument("--timeout-seconds", type=int, default=21600)
    result.add_argument("--poll-seconds", type=float, default=20.0)
    result.add_argument("--allow-busy", action="store_true")
    result.add_argument("--keep-going", action="store_true")
    result.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify corpus and print the selected cases without submitting jobs.",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    manifest_path = args.manifest.resolve()
    data_root = args.data_root.resolve()
    try:
        manifest = load_manifest(manifest_path)
        resolve_artifacts(manifest, data_root)
        selected = select_cases(manifest, suite=args.suite, requested_ids=args.case_ids)
        if args.dry_run:
            for case in selected:
                print(
                    json.dumps(
                        {
                            "id": case["id"],
                            "species": case.get("species"),
                            "actions": case["selected_actions"],
                            "source_job_id": case.get("source_job_id"),
                        },
                        ensure_ascii=False,
                    )
                )
            return 0
        report_root = (
            args.report_root.resolve()
            if args.report_root
            else data_root / DEFAULT_REPORT_ROOT_RELATIVE
        )
        report, report_dir = run_live_gallery(
            manifest_path=manifest_path,
            data_root=data_root,
            api_root=args.api_root.rstrip("/"),
            env_file=args.env_file.resolve(),
            suite=args.suite,
            requested_ids=args.case_ids,
            report_root=report_root,
            output_dir=args.output_dir.resolve() if args.output_dir else None,
            timeout_seconds=args.timeout_seconds,
            poll_seconds=args.poll_seconds,
            allow_busy=args.allow_busy,
            keep_going=args.keep_going,
            resume_job_id=args.resume_job,
        )
    except (GalleryError, OSError, requests.RequestException, StabilityError, ValueError) as exc:
        print(f"live_stability_gallery=failed: {exc}", file=sys.stderr)
        return 1
    summary = report["summary"]
    passed = summary["failed"] == 0 and summary["not_run"] == 0
    print("live_stability_gallery=" + ("passed" if passed else "failed"))
    print(
        f"cases={summary['passed']}/{summary['total']} "
        f"failed={summary['failed']} not_run={summary['not_run']}"
    )
    print(f"report_dir={report_dir}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
