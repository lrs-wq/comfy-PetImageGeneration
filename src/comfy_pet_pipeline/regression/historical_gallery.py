"""Verify a SHA-pinned historical corpus and render optional diagnostics.

The repository keeps only provenance and expectations.  Media stays below the
runtime data root and is copied into an immutable, versioned corpus on first
materialization.  Quality-result drift is reported but deliberately does not
fail this command; only corpus/input integrity is a blocking contract.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = PROJECT_ROOT / "tests/regression/historical_gallery.json"
DEFAULT_DATA_ROOT = Path("/data/comfy-pet-pipeline")
DEFAULT_REPORT_ROOT_RELATIVE = Path("outputs/regression_gallery")
SHEET_COLUMNS = 4
SHEET_CELL_WIDTH = 330
SHEET_CELL_HEIGHT = 390
SHEET_HEADER_HEIGHT = 64
SHEET_THUMBNAIL_SIZE = (286, 286)


class GalleryError(RuntimeError):
    """Raised for invalid corpus data or a blocking integrity mismatch."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GalleryError(f"Gallery manifest must be a JSON object: {path}")
    if int(value.get("schema_version", 0)) != 1:
        raise GalleryError("Unsupported historical gallery schema version.")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise GalleryError("Historical gallery manifest has no artifacts.")
    artifact_ids = [str(item.get("id") or "") for item in artifacts]
    if any(not artifact_id for artifact_id in artifact_ids):
        raise GalleryError("Every historical gallery artifact requires an id.")
    if len(set(artifact_ids)) != len(artifact_ids):
        raise GalleryError("Historical gallery artifact ids must be unique.")
    return value


def relative_data_path(data_root: Path, raw: str, *, field: str) -> Path:
    relative = Path(str(raw))
    if relative.is_absolute() or ".." in relative.parts:
        raise GalleryError(f"{field} must stay below the data root: {raw}")
    resolved_root = data_root.resolve()
    resolved = (resolved_root / relative).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise GalleryError(f"{field} escapes the data root: {raw}")
    return resolved


def artifact_index(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["id"]): item for item in manifest["artifacts"]}


def corpus_path(
    manifest: dict[str, Any],
    artifact: dict[str, Any],
    data_root: Path,
) -> Path:
    corpus_root = Path(str(manifest.get("corpus_root") or ""))
    if corpus_root.is_absolute() or ".." in corpus_root.parts:
        raise GalleryError("corpus_root must be relative to the data root.")
    corpus_file = Path(str(artifact.get("file") or ""))
    if corpus_file.is_absolute() or ".." in corpus_file.parts:
        raise GalleryError(f"Invalid corpus artifact path: {corpus_file}")
    return relative_data_path(
        data_root,
        str(corpus_root / corpus_file),
        field=f"artifact {artifact['id']} file",
    )


def verify_digest(path: Path, expected: str, *, label: str) -> None:
    if not path.is_file():
        raise GalleryError(f"Missing {label}: {path}")
    actual = file_sha256(path)
    if actual != expected:
        raise GalleryError(
            f"SHA-256 mismatch for {label}: expected {expected}, got {actual}"
        )


def materialize_corpus(
    manifest: dict[str, Any],
    data_root: Path,
) -> list[Path]:
    """Copy missing SHA-pinned artifacts from history into the v1 corpus.

    Existing files are never overwritten.  A digest mismatch fails closed so a
    curated baseline cannot be silently replaced by later output.
    """

    materialized: list[Path] = []
    for artifact in manifest["artifacts"]:
        artifact_id = str(artifact["id"])
        expected = str(artifact.get("sha256") or "")
        if len(expected) != 64:
            raise GalleryError(f"Artifact {artifact_id} has no valid SHA-256.")
        source = relative_data_path(
            data_root,
            str(artifact.get("source") or ""),
            field=f"artifact {artifact_id} source",
        )
        target = corpus_path(manifest, artifact, data_root)
        verify_digest(source, expected, label=f"historical source {artifact_id}")
        if target.exists():
            verify_digest(target, expected, label=f"corpus artifact {artifact_id}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
        if temporary.exists():
            temporary.unlink()
        shutil.copy2(source, temporary)
        verify_digest(temporary, expected, label=f"copied artifact {artifact_id}")
        os.replace(temporary, target)
        materialized.append(target)
    return materialized


def resolve_artifacts(
    manifest: dict[str, Any],
    data_root: Path,
) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    missing: list[str] = []
    for artifact in manifest["artifacts"]:
        artifact_id = str(artifact["id"])
        path = corpus_path(manifest, artifact, data_root)
        if not path.is_file():
            missing.append(artifact_id)
            continue
        verify_digest(
            path,
            str(artifact["sha256"]),
            label=f"corpus artifact {artifact_id}",
        )
        resolved[artifact_id] = path
    if missing:
        raise GalleryError(
            "Historical gallery corpus is incomplete: "
            + ", ".join(missing)
            + ". Run with --materialize once on the server."
        )
    return resolved


def _issues_match(actual: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    mismatches: list[str] = []
    actual_issues = [str(item) for item in actual.get("issues", [])]
    if "issues" in expected and actual_issues != list(expected["issues"]):
        mismatches.append(
            f"issues expected {expected['issues']!r}, got {actual_issues!r}"
        )
    for issue in expected.get("issues_contains", []):
        if issue not in actual_issues:
            mismatches.append(f"missing expected issue {issue!r}")
    return mismatches


def _simple_expectations(
    actual: dict[str, Any],
    expected: dict[str, Any],
    *,
    keys: tuple[str, ...] = ("available", "acceptable", "policy", "method"),
) -> list[str]:
    mismatches = _issues_match(actual, expected)
    for key in keys:
        if key in expected and actual.get(key) != expected[key]:
            mismatches.append(
                f"{key} expected {expected[key]!r}, got {actual.get(key)!r}"
            )
    return mismatches


def _probe_gap(probe: dict[str, Any]) -> int | None:
    for key in ("muzzle_bowl_gap_px", "nose_food_gap_px"):
        if probe.get(key) is not None:
            return int(probe[key])
    selected = probe.get("selected_component")
    if isinstance(selected, dict) and selected.get("gap_px") is not None:
        return int(selected["gap_px"])
    candidate_gaps = [
        int(candidate["gap_px"])
        for candidate in probe.get("component_candidates", [])
        if isinstance(candidate, dict) and candidate.get("gap_px") is not None
    ]
    return min(candidate_gaps) if candidate_gaps else None


def _contact_summary(probe: dict[str, Any]) -> dict[str, Any]:
    return {
        "available": bool(probe.get("available")),
        "acceptable": bool(probe.get("acceptable")),
        "gap_px": _probe_gap(probe),
        "issues": [str(item) for item in probe.get("issues", [])],
        "policy": probe.get("policy"),
        "evidence_policy": probe.get("evidence_policy"),
    }


def _contact_probe_expectations(
    name: str,
    actual: dict[str, Any],
    expected: dict[str, Any],
) -> list[str]:
    mismatches = _simple_expectations(actual, expected)
    if "gap_px" in expected and actual.get("gap_px") != expected["gap_px"]:
        mismatches.append(
            f"gap_px expected {expected['gap_px']!r}, got {actual.get('gap_px')!r}"
        )
    return [f"{name}: {message}" for message in mismatches]


def run_canonical_check(
    pipeline: Any,
    image_path: Path,
    expected: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    raw = pipeline.canonical_cutout_interior_quality(image_path)
    actual = {
        "acceptable": bool(raw.get("acceptable")),
        "issues": [str(item) for item in raw.get("issues", [])],
        "largest_void_area": int(raw.get("largest_void_area", 0) or 0),
        "largest_void_medial_overlap_fraction": float(
            raw.get("largest_void_medial_overlap_fraction", 0.0) or 0.0
        ),
        "ignored_side_concavity_count": int(
            raw.get("ignored_side_concavity_count", 0) or 0
        ),
        "policy": raw.get("policy"),
    }
    mismatches = _simple_expectations(
        actual,
        expected,
        keys=("acceptable", "policy"),
    )
    minimum_ignored = expected.get("ignored_side_concavity_count_min")
    if minimum_ignored is not None and actual["ignored_side_concavity_count"] < int(
        minimum_ignored
    ):
        mismatches.append(
            "ignored_side_concavity_count expected at least "
            f"{minimum_ignored}, got {actual['ignored_side_concavity_count']}"
        )
    minimum_overlap = expected.get("largest_void_medial_overlap_fraction_min")
    if minimum_overlap is not None and actual[
        "largest_void_medial_overlap_fraction"
    ] < float(minimum_overlap):
        mismatches.append(
            "largest_void_medial_overlap_fraction expected at least "
            f"{minimum_overlap}, got "
            f"{actual['largest_void_medial_overlap_fraction']}"
        )
    return actual, mismatches


def run_identity_check(
    pipeline: Any,
    image_path: Path,
    reference_path: Path,
    expected: dict[str, Any],
    *,
    bowl_action: bool,
    reference_cache: dict[tuple[Path, float], dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    def signature(path: Path, exclusion: float) -> dict[str, Any]:
        key = (path, exclusion)
        if key not in reference_cache:
            reference_cache[key] = pipeline.identity_palette_signature(
                path,
                exclude_bottom_fraction=exclusion,
            )
        return reference_cache[key]

    primary_reference = signature(reference_path, 0.0)
    upper_reference = signature(
        reference_path,
        pipeline.BOWL_IDENTITY_PALETTE_UPPER_BODY_EXCLUSION,
    )
    result = pipeline.evaluate_pose_identity_palette(
        image_path,
        primary_reference=primary_reference,
        upper_body_reference=upper_reference,
        bottom_exclusion=(
            pipeline.BOWL_IDENTITY_PALETTE_BOTTOM_EXCLUSION
            if bowl_action
            else 0.0
        ),
        bowl_action=bowl_action,
    )
    quality = result["quality"]
    spatial = quality.get("marking_spatial_quality") or {}
    actual = {
        "available": bool(quality.get("available")),
        "acceptable": bool(quality.get("acceptable")),
        "issues": [str(item) for item in quality.get("issues", [])],
        "warnings": [str(item) for item in quality.get("warnings", [])],
        "method": quality.get("method"),
        "stripe_gate_mode": spatial.get("stripe_gate_mode"),
        "stripe_component_delta": spatial.get("stripe_component_delta"),
        "stripe_area_delta": spatial.get("stripe_area_delta"),
        "dark_contrast_delta": spatial.get("dark_contrast_delta"),
    }
    mismatches = _simple_expectations(
        actual,
        expected,
        keys=("available", "acceptable", "method"),
    )
    if (
        "stripe_gate_mode" in expected
        and actual["stripe_gate_mode"] != expected["stripe_gate_mode"]
    ):
        mismatches.append(
            "stripe_gate_mode expected "
            f"{expected['stripe_gate_mode']!r}, got {actual['stripe_gate_mode']!r}"
        )
    return actual, mismatches


def run_contact_check(
    pipeline: Any,
    image_path: Path,
    expected: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    neutral_raw = pipeline.eating_neutral_muzzle_bowl_contact_quality(image_path)
    nose_raw = pipeline.eating_nose_kibble_contact_quality(
        image_path,
        neutral_contact=neutral_raw,
    )
    neutral = _contact_summary(neutral_raw)
    nose = _contact_summary(nose_raw)
    acceptable = bool(neutral["acceptable"] or nose["acceptable"])
    evidence = (
        "neutral_muzzle_bowl"
        if neutral["acceptable"]
        else "nose_kibble"
        if nose["acceptable"]
        else None
    )
    actual = {
        "acceptable": acceptable,
        "evidence": evidence,
        "neutral": neutral,
        "nose": nose,
    }
    mismatches: list[str] = []
    for key in ("acceptable", "evidence"):
        if key in expected and actual[key] != expected[key]:
            mismatches.append(
                f"{key} expected {expected[key]!r}, got {actual[key]!r}"
            )
    if isinstance(expected.get("neutral"), dict):
        mismatches.extend(
            _contact_probe_expectations(
                "neutral",
                neutral,
                expected["neutral"],
            )
        )
    if isinstance(expected.get("nose"), dict):
        mismatches.extend(
            _contact_probe_expectations(
                "nose",
                nose,
                expected["nose"],
            )
        )
    return actual, mismatches


def run_gate_cases(
    manifest: dict[str, Any],
    artifacts: dict[str, Path],
) -> list[dict[str, Any]]:
    from src.comfy_pet_pipeline.api import local_video as pipeline

    results: list[dict[str, Any]] = []
    reference_cache: dict[tuple[Path, float], dict[str, Any]] = {}
    for case in manifest.get("gate_cases", []):
        case_id = str(case["id"])
        artifact_id = str(case["artifact"])
        image_path = artifacts.get(artifact_id)
        if image_path is None:
            raise GalleryError(
                f"Gate case {case_id} references unknown artifact {artifact_id}."
            )
        check_results: list[dict[str, Any]] = []
        case_mismatches: list[str] = []
        for check in case.get("checks", []):
            check_type = str(check.get("type") or "")
            expected = check.get("expected") or {}
            reference_path: Path | None = None
            if check_type == "identity_palette":
                reference_id = str(check.get("reference") or "")
                reference_path = artifacts.get(reference_id)
                if reference_path is None:
                    raise GalleryError(
                        f"Gate case {case_id} references unknown identity "
                        f"artifact {reference_id}."
                    )
            if check_type not in {
                "canonical_interior",
                "identity_palette",
                "eating_contact",
            }:
                raise GalleryError(
                    f"Gate case {case_id} has unsupported check type {check_type!r}."
                )
            try:
                if check_type == "canonical_interior":
                    actual, mismatches = run_canonical_check(
                        pipeline,
                        image_path,
                        expected,
                    )
                elif check_type == "identity_palette":
                    assert reference_path is not None
                    actual, mismatches = run_identity_check(
                        pipeline,
                        image_path,
                        reference_path,
                        expected,
                        bowl_action=bool(check.get("bowl_action")),
                        reference_cache=reference_cache,
                    )
                else:
                    actual, mismatches = run_contact_check(
                        pipeline,
                        image_path,
                        expected,
                    )
            except Exception as exc:  # noqa: BLE001 - diagnostics never block stability
                actual = {
                    "diagnostic_error": f"{type(exc).__name__}: {str(exc)[:1000]}"
                }
                mismatches = [f"diagnostic raised {actual['diagnostic_error']}"]
            prefixed = [f"{check_type}: {message}" for message in mismatches]
            case_mismatches.extend(prefixed)
            check_results.append(
                {
                    "type": check_type,
                    "expected": expected,
                    "actual": actual,
                    "baseline_match": not mismatches,
                    "passed": not mismatches,
                    "mismatches": mismatches,
                }
            )
        results.append(
            {
                "id": case_id,
                "label": str(case.get("label") or case_id),
                "category": str(case.get("category") or "historical"),
                "source_job_id": case.get("source_job_id"),
                "artifact": artifact_id,
                "image_path": str(image_path),
                "baseline_match": not case_mismatches,
                "passed": not case_mismatches,
                "mismatches": case_mismatches,
                "checks": check_results,
            }
        )
    return results


def validate_canaries(
    manifest: dict[str, Any],
    artifacts: dict[str, Path],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for canary in manifest.get("canary_cases", []):
        canary_id = str(canary["id"])
        view_ids = canary.get("views") or {}
        if not isinstance(view_ids, dict) or "front" not in view_ids:
            raise GalleryError(f"Canary {canary_id} requires a front view.")
        view_paths: dict[str, str] = {}
        view_hashes: list[str] = []
        mismatches: list[str] = []
        for role, artifact_id_raw in view_ids.items():
            artifact_id = str(artifact_id_raw)
            path = artifacts.get(artifact_id)
            if path is None:
                raise GalleryError(
                    f"Canary {canary_id} references unknown artifact {artifact_id}."
                )
            view_paths[str(role)] = str(path)
            view_hashes.append(file_sha256(path))
        expected_view_count = int(canary.get("expected_view_count", len(view_paths)))
        if len(view_paths) != expected_view_count:
            mismatches.append(
                f"view count expected {expected_view_count}, got {len(view_paths)}"
            )
        unique_view_count = len(set(view_hashes))
        expected_unique = int(
            canary.get("expected_unique_view_count", unique_view_count)
        )
        if unique_view_count != expected_unique:
            mismatches.append(
                f"unique view count expected {expected_unique}, got {unique_view_count}"
            )
        results.append(
            {
                "id": canary_id,
                "label": str(canary.get("label") or canary_id),
                "source_job_id": canary.get("source_job_id"),
                "species": str(canary["species"]),
                "actions": [str(action) for action in canary.get("actions", [])],
                "subject_profile": str(canary.get("subject_profile") or ""),
                "views": view_paths,
                "view_count": len(view_paths),
                "unique_view_count": unique_view_count,
                "passed": not mismatches,
                "mismatches": mismatches,
            }
        )
    return results


def _load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path(
            "/usr/share/fonts/truetype/dejavu/"
            + ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
        ),
        Path(
            "/usr/share/fonts/dejavu/"
            + ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
        ),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _thumbnail(path: Path) -> Image.Image:
    with Image.open(path) as source:
        rgba = ImageOps.exif_transpose(source).convert("RGBA")
        checker = Image.new("RGBA", rgba.size, (245, 245, 245, 255))
        if rgba.getextrema()[3][0] < 255:
            draw = ImageDraw.Draw(checker)
            square = max(8, min(rgba.size) // 24)
            for y in range(0, rgba.height, square):
                for x in range(0, rgba.width, square):
                    if (x // square + y // square) % 2:
                        draw.rectangle(
                            (x, y, x + square - 1, y + square - 1),
                            fill=(220, 220, 220, 255),
                        )
        checker.alpha_composite(rgba)
        rgb = checker.convert("RGB")
    rgb.thumbnail(SHEET_THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
    frame = Image.new("RGB", SHEET_THUMBNAIL_SIZE, (250, 250, 250))
    left = (frame.width - rgb.width) // 2
    top = (frame.height - rgb.height) // 2
    frame.paste(rgb, (left, top))
    return frame


def render_sheet(
    items: list[dict[str, Any]],
    destination: Path,
    *,
    title: str,
    canary: bool = False,
) -> None:
    rows = max(1, (len(items) + SHEET_COLUMNS - 1) // SHEET_COLUMNS)
    sheet = Image.new(
        "RGB",
        (
            SHEET_COLUMNS * SHEET_CELL_WIDTH,
            SHEET_HEADER_HEIGHT + rows * SHEET_CELL_HEIGHT,
        ),
        (27, 29, 35),
    )
    draw = ImageDraw.Draw(sheet)
    title_font = _load_font(26, bold=True)
    label_font = _load_font(15, bold=True)
    detail_font = _load_font(12)
    draw.text((20, 16), title, font=title_font, fill=(245, 245, 245))
    for index, item in enumerate(items):
        column = index % SHEET_COLUMNS
        row = index // SHEET_COLUMNS
        cell_x = column * SHEET_CELL_WIDTH
        cell_y = SHEET_HEADER_HEIGHT + row * SHEET_CELL_HEIGHT
        passed = bool(item.get("passed"))
        accent = (41, 171, 103) if passed else (224, 72, 72)
        draw.rounded_rectangle(
            (
                cell_x + 8,
                cell_y + 8,
                cell_x + SHEET_CELL_WIDTH - 8,
                cell_y + SHEET_CELL_HEIGHT - 8,
            ),
            radius=12,
            fill=(248, 248, 248),
            outline=accent,
            width=5,
        )
        image_path = (
            Path(item["views"]["front"])
            if canary
            else Path(item["image_path"])
        )
        thumb = _thumbnail(image_path)
        sheet.paste(thumb, (cell_x + 22, cell_y + 22))
        draw.text(
            (cell_x + 22, cell_y + 317),
            str(item["label"])[:38],
            font=label_font,
            fill=(25, 25, 25),
        )
        if canary:
            details = (
                f"{item['species']} | {item['view_count']} views / "
                f"{item['unique_view_count']} unique"
            )
            actions = ",".join(item.get("actions", []))
            second = actions[:48]
        else:
            details = f"{item['category']} | {'MATCH' if passed else 'DRIFT'}"
            second = (
                item["mismatches"][0][:48]
                if item.get("mismatches")
                else str(item.get("source_job_id") or "historical replay")[:48]
            )
        draw.text(
            (cell_x + 22, cell_y + 342),
            details,
            font=detail_font,
            fill=(60, 60, 60),
        )
        draw.text(
            (cell_x + 22, cell_y + 360),
            second,
            font=detail_font,
            fill=accent if not passed else (85, 85, 85),
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, format="PNG", optimize=True)


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short=10", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or "unknown"


def create_report_directory(report_root: Path) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    stem = f"{timestamp}-{git_revision()}"
    for suffix in range(100):
        candidate = report_root / (stem if suffix == 0 else f"{stem}-{suffix:02d}")
        try:
            candidate.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return candidate
    raise GalleryError(f"Could not allocate a report directory below {report_root}")


def write_html_report(
    destination: Path,
    report: dict[str, Any],
) -> None:
    gate_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(item['id']))}</td>"
        f"<td>{html.escape(str(item['label']))}</td>"
        f"<td class=\"{'pass' if item['passed'] else 'fail'}\">"
        f"{'MATCH' if item['passed'] else 'DRIFT'}</td>"
        f"<td>{html.escape('; '.join(item.get('mismatches', [])))}</td>"
        "</tr>"
        for item in report["gate_cases"]
    )
    canary_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(item['id']))}</td>"
        f"<td>{html.escape(str(item['species']))}</td>"
        f"<td>{html.escape(', '.join(item['actions']))}</td>"
        f"<td>{item['view_count']} / {item['unique_view_count']}</td>"
        "</tr>"
        for item in report["canary_cases"]
    )
    document = f"""<!doctype html>
<html lang="zh-CN">
<meta charset="utf-8">
<title>Historical regression gallery</title>
<style>
body {{ font: 15px system-ui, sans-serif; margin: 32px; color: #202124; }}
img {{ display: block; max-width: 100%; margin: 12px 0 28px; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 28px; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
.pass {{ color: #087f4f; font-weight: 700; }}
.fail {{ color: #c62828; font-weight: 700; }}
code {{ background: #f2f3f5; padding: 2px 5px; }}
</style>
<h1>Historical stability gallery</h1>
<p>Revision <code>{html.escape(str(report['git_revision']))}</code> ·
{html.escape(str(report['created_at']))} ·
<strong>{report['summary']['passed']}/{report['summary']['total']}
input contracts passed</strong></p>
<p>Quality differences are diagnostic only and never fail this command.
Live completion is checked separately by <code>run-live-stability-gallery.py</code>.</p>
<h2>Non-blocking quality diagnostics</h2>
<img src="gate-gallery.png" alt="Non-blocking quality diagnostic gallery">
<table><thead><tr><th>Case</th><th>Label</th><th>Diagnostic</th><th>Difference</th></tr></thead>
<tbody>{gate_rows}</tbody></table>
<h2>Generation canary inputs</h2>
<img src="canary-inputs.png" alt="Generation canary input gallery">
<table><thead><tr><th>Canary</th><th>Species</th><th>Actions</th>
<th>Views / unique</th></tr></thead>
<tbody>{canary_rows}</tbody></table>
</html>
"""
    destination.write_text(document, encoding="utf-8")


def update_latest_pointer(report_root: Path, report_dir: Path, passed: bool) -> None:
    pointer = {
        "report_dir": str(report_dir),
        "passed": passed,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    temporary = report_root / f".latest.json.tmp-{os.getpid()}"
    temporary.write_text(
        json.dumps(pointer, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, report_root / "latest.json")


def run_gallery(
    manifest_path: Path,
    data_root: Path,
    *,
    report_root: Path | None = None,
    output_dir: Path | None = None,
) -> tuple[dict[str, Any], Path]:
    manifest = load_manifest(manifest_path)
    artifacts = resolve_artifacts(manifest, data_root)
    gate_results = run_gate_cases(manifest, artifacts)
    canary_results = validate_canaries(manifest, artifacts)
    passed_count = sum(bool(item["passed"]) for item in canary_results)
    diagnostic_matches = sum(bool(item["passed"]) for item in gate_results)
    report = {
        "schema_version": 1,
        "gallery": str(manifest.get("name") or "historical-regression-gallery"),
        "manifest": str(manifest_path),
        "data_root": str(data_root),
        "git_revision": git_revision(),
        "created_at": datetime.now(UTC).isoformat(),
        "summary": {
            "passed": passed_count,
            "failed": len(canary_results) - passed_count,
            "total": len(canary_results),
            "diagnostic_matched": diagnostic_matches,
            "diagnostic_drifted": len(gate_results) - diagnostic_matches,
            "diagnostic_total": len(gate_results),
            "canary_cases": len(canary_results),
        },
        "gate_cases": gate_results,
        "canary_cases": canary_results,
    }
    effective_report_root = report_root or relative_data_path(
        data_root,
        str(DEFAULT_REPORT_ROOT_RELATIVE),
        field="report root",
    )
    report_dir = output_dir or create_report_directory(effective_report_root)
    report_dir.mkdir(parents=True, exist_ok=True)
    render_sheet(
        gate_results,
        report_dir / "gate-gallery.png",
        title="Historical quality diagnostics (non-blocking)",
    )
    render_sheet(
        canary_results,
        report_dir / "canary-inputs.png",
        title="Historical generation canary inputs",
        canary=True,
    )
    (report_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_html_report(report_dir / "index.html", report)
    effective_report_root.mkdir(parents=True, exist_ok=True)
    update_latest_pointer(
        effective_report_root,
        report_dir,
        passed=passed_count == len(canary_results),
    )
    return report, report_dir


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Verify the SHA-pinned inputs and render non-blocking diagnostics.",
    )
    result.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    result.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("PET_PIPELINE_DATA_ROOT", DEFAULT_DATA_ROOT)),
    )
    result.add_argument("--report-root", type=Path)
    result.add_argument("--output-dir", type=Path)
    result.add_argument(
        "--materialize",
        action="store_true",
        help="Copy missing SHA-pinned files from historical jobs into the corpus.",
    )
    result.add_argument(
        "--materialize-only",
        action="store_true",
        help="Materialize and verify the corpus without executing gate replay.",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    manifest_path = args.manifest.resolve()
    data_root = args.data_root.resolve()
    try:
        manifest = load_manifest(manifest_path)
        if args.materialize or args.materialize_only:
            copied = materialize_corpus(manifest, data_root)
            print(f"materialized={len(copied)}")
            print(f"corpus={data_root / manifest['corpus_root']}")
        if args.materialize_only:
            resolve_artifacts(manifest, data_root)
            print(f"verified_artifacts={len(manifest['artifacts'])}")
            return 0
        report_root = args.report_root.resolve() if args.report_root else None
        output_dir = args.output_dir.resolve() if args.output_dir else None
        report, report_dir = run_gallery(
            manifest_path,
            data_root,
            report_root=report_root,
            output_dir=output_dir,
        )
    except (GalleryError, OSError, ValueError) as exc:
        print(f"historical_gallery=failed: {exc}", file=sys.stderr)
        return 1
    summary = report["summary"]
    print(
        "historical_gallery="
        + ("passed" if summary["failed"] == 0 else "failed")
    )
    print(
        f"input_contracts={summary['passed']}/{summary['total']} "
        f"diagnostics={summary['diagnostic_matched']}/"
        f"{summary['diagnostic_total']} matched"
    )
    print(f"report_dir={report_dir}")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
