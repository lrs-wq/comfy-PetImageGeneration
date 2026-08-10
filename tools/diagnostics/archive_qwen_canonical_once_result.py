#!/usr/bin/env python3
"""Archive and CPU-audit the one approved canonical Qwen result.

This script never posts to ComfyUI.  It refuses to overwrite its audit folder,
copies the immutable raw result and history, builds a transparent cutout, and
writes fail-closed identity/composition/fur gate evidence for later review.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


PROJECT = Path("/root/comfy-pet-pipeline")
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.comfy_pet_pipeline.api import local_video as api  # noqa: E402
from src.comfy_pet_pipeline.quality.fur_surface import (  # noqa: E402
    evaluate_fur_surface,
)


PROMPT_ID = "97f60b6f-f1fc-4434-9759-529c9927056b"
QUEUE_NUMBER = 197
RUN_ID = "codex-upload-identity-main-case-canonical-v1-20260731-r2"
BUNDLE = Path(
    "/data/comfy-pet-pipeline/archive/validation/"
    "20260731-upload-only-identity-17f-r1/"
    "upload_identity_main_case_canonical_v1_r2"
)
AUDIT = BUNDLE / "gpu_qwen_canonical_once_audit_r1"
RAW = Path(
    "/data/comfy-pet-pipeline/comfyui/output/scail2_test/"
    f"{RUN_ID}/canonical_00001_.png"
)
UPLOAD = Path(
    "/data/comfy-pet-pipeline/outputs/local_video_api/"
    "20260731T050352Z-ec6245d4d4/input_identity_source.png"
)
EATING_BASELINE = Path(
    "/data/comfy-pet-pipeline/outputs/local_video_api/"
    "20260731T050352Z-ec6245d4d4/profile_inputs/eating_reference.png"
)
GROOMING_BASELINE = Path(
    "/data/comfy-pet-pipeline/outputs/local_video_api/"
    "20260728T064333Z-c64261b1f9/profile_inputs/grooming_reference.png"
)
EXPECTED = {
    RAW: "3738a79a623196612909e415379d9b35a5ea5337fa83ece5918cd985c3dbfd2a",
    UPLOAD: "d8c377e76cf25460d756d18149017b73328b9fbd3bcd7370b35cd334f85a2a31",
    BUNDLE / "qwen_canonical_one_shot_workflow_api.json": "311705f556c3432a14dc9639e6d1bd39bc0e0cf2443a66f754a836a2f845ad88",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def get_json(endpoint: str) -> dict[str, Any]:
    with urllib.request.urlopen("http://127.0.0.1:8188" + endpoint, timeout=10) as response:
        return json.load(response)


def json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def save_board(paths: list[tuple[str, Path]], output: Path) -> None:
    cell, header = 300, 54
    board = Image.new("RGB", (cell * len(paths), cell + header), "white")
    draw = ImageDraw.Draw(board)
    for index, (label, path) in enumerate(paths):
        with Image.open(path) as loaded:
            image = loaded.convert("RGBA")
            if "A" in image.getbands():
                background = Image.new("RGBA", image.size, (238, 238, 238, 255))
                background.alpha_composite(image)
                thumb = background.convert("RGB")
            else:
                thumb = image.convert("RGB")
        thumb.thumbnail((cell - 16, cell - 16), Image.Resampling.LANCZOS)
        x = index * cell + (cell - thumb.width) // 2
        y = header + (cell - thumb.height) // 2
        board.paste(thumb, (x, y))
        draw.text((index * cell + 8, 17), label, fill="black")
    board.save(output, format="PNG", optimize=True)


def main() -> int:
    if AUDIT.exists():
        raise RuntimeError(f"refusing_to_overwrite:{AUDIT}")
    for path, digest in EXPECTED.items():
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"frozen_artifact_missing_or_changed:{path}")
    for path in (EATING_BASELINE, GROOMING_BASELINE):
        if not path.is_file():
            raise RuntimeError(f"validated_fur_baseline_missing:{path}")
    AUDIT.mkdir(parents=True)

    history_response = get_json("/history/" + PROMPT_ID)
    if list(history_response) != [PROMPT_ID]:
        raise RuntimeError("history_prompt_id_mismatch")
    history = history_response[PROMPT_ID]
    prompt = history.get("prompt") or []
    if int(prompt[0]) != QUEUE_NUMBER or str(prompt[1]) != PROMPT_ID:
        raise RuntimeError("history_queue_identity_mismatch")
    status = history.get("status") or {}
    if status.get("status_str") != "success" or status.get("completed") is not True:
        raise RuntimeError("qwen_prompt_not_successful")
    output_record = history.get("outputs", {}).get("16", {}).get("images", [])[0]
    if (
        output_record.get("filename") != RAW.name
        or output_record.get("subfolder") != f"scail2_test/{RUN_ID}"
        or output_record.get("type") != "output"
    ):
        raise RuntimeError("history_output_does_not_match_frozen_raw_png")

    raw_copy = AUDIT / "canonical_qwen_raw.png"
    shutil.copy2(RAW, raw_copy)
    history_path = AUDIT / "history.json"
    history_path.write_text(
        json.dumps(history, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    history_response_path = AUDIT / "history_response.json"
    history_response_path.write_text(
        json.dumps(history_response, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    for filename in (
        "manifest.json",
        "summary.json",
        "qwen_canonical_one_shot_workflow_api.json",
        "gpu_submission_record.json",
        "canonical_main_case_semantic_guide.png",
        "canonical_input_roles_visual_board.png",
    ):
        source = BUNDLE / filename
        if source.is_file():
            shutil.copy2(source, AUDIT / filename)

    chroma_cutout = AUDIT / "canonical_chroma_cutout.png"
    try:
        chroma = json_safe(api.evaluate_chroma_cutout(raw_copy, chroma_cutout))
    except Exception as exc:  # noqa: BLE001 - preserve fail-closed evidence
        chroma = {
            "acceptable": False,
            "issues": ["chroma_evaluation_exception"],
            "error": f"{type(exc).__name__}: {exc}",
        }
    cutout = chroma_cutout
    alpha_provenance = "chroma_key"
    recovery = None
    if chroma.get("acceptable") is not True:
        recovery_green = AUDIT / "canonical_recovered_green.png"
        recovery_cutout = AUDIT / "canonical_recovered_cutout.png"
        recovery_semantic = AUDIT / "canonical_recovered_semantic_raw.png"
        try:
            recovery = json_safe(
                api.semantic_background_recovery(
                    raw_copy,
                    recovery_green,
                    recovery_cutout,
                    recovery_semantic,
                    pose="canonical character",
                )
            )
        except Exception as exc:  # noqa: BLE001
            recovery = {
                "acceptable": False,
                "issues": ["semantic_background_recovery_exception"],
                "error": f"{type(exc).__name__}: {exc}",
            }
        if recovery.get("acceptable") is True:
            cutout = recovery_cutout
            alpha_provenance = "rembg_u2net_fallback"

    reliable_cutout = bool(
        cutout.is_file()
        and (
            chroma.get("acceptable") is True
            or isinstance(recovery, dict)
            and recovery.get("acceptable") is True
        )
    )
    identity = None
    composition = None
    fur: dict[str, Any] = {}
    if reliable_cutout:
        source_full = api.identity_palette_signature(UPLOAD)
        candidate_full = api.identity_palette_signature(cutout)
        source_upper = api.identity_palette_signature(UPLOAD, exclude_bottom_fraction=0.45)
        candidate_upper = api.identity_palette_signature(cutout, exclude_bottom_fraction=0.45)
        identity = json_safe(
            api.canonical_identity_palette_quality(
                api.identity_palette_quality(source_full, candidate_full),
                api.identity_palette_quality(source_upper, candidate_upper),
            )
        )
        composition = json_safe(api.pose_cutout_quality(cutout, "canonical character"))
        for action, baseline in (
            ("eating", EATING_BASELINE),
            ("grooming", GROOMING_BASELINE),
        ):
            try:
                result = evaluate_fur_surface(
                    cutout,
                    baseline,
                    baseline_is_validated=True,
                )
                fur[action] = json_safe(result)
            except Exception as exc:  # noqa: BLE001
                fur[action] = {
                    "acceptable": False,
                    "issues": ["fur_surface_gate_exception"],
                    "error": f"{type(exc).__name__}: {exc}",
                }

    queue_after = get_json("/queue")
    messages = status.get("messages") or []
    execution_start = next(
        int(item[1]["timestamp"]) for item in messages if item[0] == "execution_start"
    )
    execution_success = next(
        int(item[1]["timestamp"]) for item in messages if item[0] == "execution_success"
    )
    machine_gates = {
        "history_execution_success": True,
        "raw_png_sha256_exact": sha256_file(raw_copy) == EXPECTED[RAW],
        "single_qwen_prompt_only": int(prompt[0]) == QUEUE_NUMBER,
        "workflow_has_no_scail_nodes": "SCAIL" not in json.dumps(prompt[2], ensure_ascii=False),
        "queue_empty_after_archive": not queue_after.get("queue_running") and not queue_after.get("queue_pending"),
        "reliable_transparent_cutout_available": reliable_cutout,
        "identity_palette_gate_passed": isinstance(identity, dict) and identity.get("acceptable") is True,
        "canonical_composition_gate_passed": isinstance(composition, dict) and composition.get("acceptable") is True,
        "eating_fur_surface_gate_passed": fur.get("eating", {}).get("acceptable") is True,
        "grooming_fur_surface_gate_passed": fur.get("grooming", {}).get("acceptable") is True,
    }
    machine_all_passed = all(value is True for value in machine_gates.values())
    gate_manifest = {
        "status": "machine_gates_passed_manual_review_required" if machine_all_passed else "machine_gate_failed_do_not_scail",
        "prompt_id": PROMPT_ID,
        "queue_number": QUEUE_NUMBER,
        "run_id": RUN_ID,
        "raw_png": {"path": str(raw_copy), "sha256": sha256_file(raw_copy), "bytes": raw_copy.stat().st_size},
        "alpha_provenance": alpha_provenance if reliable_cutout else None,
        "cutout": {"path": str(cutout), "sha256": sha256_file(cutout) if reliable_cutout else None},
        "chroma": chroma,
        "semantic_recovery": recovery,
        "identity_palette": identity,
        "composition": composition,
        "fur_surface": fur,
        "machine_gates": machine_gates,
        "machine_all_passed": machine_all_passed,
        "manual_review_required": True,
        "scail_submitted": False,
        "production_modified": False,
    }
    gate_manifest_path = AUDIT / "canonical_pre_scail_gate_manifest.json"
    gate_manifest_path.write_text(
        json.dumps(gate_manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    board_items = [
        ("uploaded identity", UPLOAD),
        ("semantic geometry", BUNDLE / "canonical_main_case_semantic_guide.png"),
        ("Qwen raw", raw_copy),
    ]
    if reliable_cutout:
        board_items.append((f"cutout: {alpha_provenance}", cutout))
    board_items.extend(
        [("eating fur baseline", EATING_BASELINE), ("grooming fur baseline", GROOMING_BASELINE)]
    )
    board_path = AUDIT / "canonical_gpu_audit_board.png"
    save_board(board_items, board_path)

    report = {
        "status": gate_manifest["status"],
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prompt_id": PROMPT_ID,
        "queue_number": QUEUE_NUMBER,
        "execution_seconds": round((execution_success - execution_start) / 1000.0, 3),
        "history_status": status.get("status_str"),
        "output_record": output_record,
        "raw_png_sha256": sha256_file(raw_copy),
        "history_sha256": sha256_file(history_path),
        "history_response_sha256": sha256_file(history_response_path),
        "gate_manifest_sha256": sha256_file(gate_manifest_path),
        "audit_board_sha256": sha256_file(board_path),
        "queue_after": queue_after,
        "scail_submitted": False,
        "production_modified": False,
    }
    report_path = AUDIT / "GPU_QWEN_CANONICAL_REPORT.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    artifact_hashes = {
        path.name: sha256_file(path)
        for path in sorted(AUDIT.iterdir())
        if path.is_file()
    }
    hashes_path = AUDIT / "artifact_sha256.json"
    hashes_path.write_text(
        json.dumps(artifact_hashes, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        **report,
        "report_sha256": sha256_file(report_path),
        "artifact_sha256_manifest": str(hashes_path),
        "artifact_sha256_manifest_sha256": sha256_file(hashes_path),
        "machine_gates": machine_gates,
        "failed_machine_gates": [name for name, passed in machine_gates.items() if not passed],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
