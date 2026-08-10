#!/usr/bin/env python3
"""Build deterministic Main_case default-subject profile assets (CPU only).

The script never submits ComfyUI work and never writes outside ``--output-dir``.
It freezes both original source-video digests and the normalized 81-frame driver
digests before extracting the selected reference frames.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.comfy_pet_pipeline.pipeline.subject_replacement import (  # noqa: E402
    make_eating_reference_identity_mask,
)
from src.comfy_pet_pipeline.media.transparent_animation import (  # noqa: E402
    _subject_mask,
)


PROFILE_ID = "sleek_tabby_cat_v1"
REVISION = "main-case-20260801-v2"
ASSET_VALIDATION_RUN_ID = "codex-main-case-profile-assets-20260801-v2"
SOURCE_SHA256 = {
    "eating": "ea893acad01481dcd2c96e7b115478d1e82ad3785ad8d641296aecb7a25986b7",
    "grooming": "b03d7e5ebb5aa878d1f57fff389dfa241d7ff8d7d273df1fd264bb479311aedd",
}
RUNTIME_SHA256 = {
    "eating": "07cf90c34ecca9742cf10c7a0290274ec1b9e5c1a008d73335836aec53d49b46",
    "grooming": "2fc8ed9a398706fb2bf88b43f340be1a555d7dc638073a43e8f89f5c15ec0993",
}
SELECTED_FRAME = {"eating": 40, "grooming": 40}
SERVER_ASSET_ROOT = (
    "/data/comfy-pet-pipeline/profile_assets/sleek_tabby_cat_v1/"
    "identity/main-case-20260801-v2"
)
SERVER_GENERATOR_PATH = (
    "/root/comfy-pet-pipeline/tools/diagnostics/build_maincase_profile_assets.py"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def decode_frame(path: Path, index: int) -> tuple[np.ndarray, dict[str, Any]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Unable to decode {path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_count != 81 or abs(fps - 16.0) > 0.001 or (width, height) != (512, 512):
        capture.release()
        raise RuntimeError(
            "Normalized driver contract changed: "
            f"frames={frame_count}, fps={fps}, size={width}x{height}"
        )
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, bgr = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Unable to decode frame {index} from {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), {
        "frames": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
    }


def bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask)
    if not len(xs):
        raise RuntimeError("Reference alpha is empty")
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


def save_png(image: Image.Image, path: Path) -> None:
    image.save(path, format="PNG", optimize=False, compress_level=9)


def make_preview(cutout: Image.Image) -> Image.Image:
    background = Image.new("RGB", cutout.size, (238, 243, 240))
    background.paste(cutout.convert("RGB"), mask=cutout.getchannel("A"))
    return background


def make_audit_board(records: dict[str, dict[str, Any]], output: Path) -> None:
    cell = 384
    header = 42
    board = Image.new("RGB", (cell * 3, (cell + header) * 2), "white")
    draw = ImageDraw.Draw(board)
    for row, action in enumerate(("eating", "grooming")):
        record = records[action]
        y = row * (cell + header)
        labels = (
            f"{action} preview / frame {record['selected_frame_index']}",
            f"{action} transparent cutout",
            f"{action} identity mask",
        )
        paths = (
            Path(record["local_preview"]),
            Path(record["local_cutout"]),
            Path(record["local_identity_mask"]),
        )
        for column, (label, path) in enumerate(zip(labels, paths, strict=True)):
            x = column * cell
            draw.text((x + 8, y + 14), label, fill="black")
            with Image.open(path) as loaded:
                image = loaded.convert("RGBA")
            if column == 1:
                checker = Image.new("RGB", image.size, (225, 225, 225))
                checker_draw = ImageDraw.Draw(checker)
                tile = 32
                for yy in range(0, image.height, tile):
                    for xx in range(0, image.width, tile):
                        if (xx // tile + yy // tile) % 2:
                            checker_draw.rectangle(
                                (xx, yy, xx + tile - 1, yy + tile - 1),
                                fill=(250, 250, 250),
                            )
                checker.paste(image.convert("RGB"), mask=image.getchannel("A"))
                shown = checker
            else:
                shown = image.convert("RGB")
            shown = shown.resize((cell, cell), Image.Resampling.LANCZOS)
            board.paste(shown, (x, y + header))
    save_png(board, output)


def build_action(
    action: str,
    source: Path,
    runtime: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if sha256_file(source) != SOURCE_SHA256[action]:
        raise RuntimeError(f"{action} Main_case source SHA-256 changed")
    if sha256_file(runtime) != RUNTIME_SHA256[action]:
        raise RuntimeError(f"{action} normalized 81-frame SHA-256 changed")
    selected_index = SELECTED_FRAME[action]
    rgb, video_contract = decode_frame(runtime, selected_index)
    clean_rgb, alpha = _subject_mask(rgb.copy())
    alpha_support = alpha >= 16
    alpha_bbox = bbox(alpha_support)
    border_contact = {
        "top": int(np.count_nonzero(alpha_support[0, :])),
        "bottom": int(np.count_nonzero(alpha_support[-1, :])),
        "left": int(np.count_nonzero(alpha_support[:, 0])),
        "right": int(np.count_nonzero(alpha_support[:, -1])),
    }
    corners_opaque = bool(
        alpha_support[0, 0]
        or alpha_support[0, -1]
        or alpha_support[-1, 0]
        or alpha_support[-1, -1]
    )
    if corners_opaque or any(border_contact[key] for key in ("top", "bottom", "left")):
        raise RuntimeError(f"{action} cutout has unexpected canvas-border foreground")
    if action == "grooming" and border_contact["right"]:
        raise RuntimeError("Grooming cutout unexpectedly touches the right border")
    if action == "eating":
        right_rows = np.where(alpha_support[:, -1])[0]
        if not len(right_rows) or int(right_rows.min()) < 350:
            raise RuntimeError("Eating right-border contact is not the expected lower tail")
    if np.count_nonzero(alpha_support) < 20_000:
        raise RuntimeError(f"{action} cutout is unexpectedly small")

    cutout_array = np.dstack((clean_rgb, alpha))
    cutout = Image.fromarray(cutout_array, mode="RGBA")
    preview = make_preview(cutout)
    if action == "eating":
        identity_mask, _prop_mask, prop_metadata = make_eating_reference_identity_mask(
            cutout
        )
        mask_policy = "blue_maincase_cat_red_bowl_v1"
    else:
        mask_array = np.zeros((512, 512, 3), dtype=np.uint8)
        mask_array[alpha_support] = (0, 0, 255)
        identity_mask = Image.fromarray(mask_array, mode="RGB")
        prop_metadata = None
        mask_policy = "blue_maincase_cat_no_prop_v1"

    suffix = f"frame{selected_index:03d}"
    preview_path = output_dir / f"{action}_preview_{suffix}.png"
    cutout_path = output_dir / f"{action}_reference_{suffix}_cutout.png"
    mask_path = output_dir / f"{action}_reference_{suffix}_identity_mask.png"
    save_png(preview, preview_path)
    save_png(cutout, cutout_path)
    save_png(identity_mask.convert("RGB"), mask_path)

    mask_pixels = np.asarray(identity_mask.convert("RGB"), dtype=np.uint8)
    allowed = (
        np.all(mask_pixels == (0, 0, 0), axis=2)
        | np.all(mask_pixels == (0, 0, 255), axis=2)
        | np.all(mask_pixels == (255, 0, 0), axis=2)
    )
    if not np.all(allowed):
        raise RuntimeError(f"{action} identity mask contains unknown colors")
    counts = {
        "black": int(np.count_nonzero(np.all(mask_pixels == (0, 0, 0), axis=2))),
        "blue_identity": int(
            np.count_nonzero(np.all(mask_pixels == (0, 0, 255), axis=2))
        ),
        "red_prop": int(
            np.count_nonzero(np.all(mask_pixels == (255, 0, 0), axis=2))
        ),
    }
    if counts["blue_identity"] < 20_000:
        raise RuntimeError(f"{action} identity mask is unexpectedly small")
    if action == "eating" and counts["red_prop"] < 500:
        raise RuntimeError("Eating reference mask did not preserve the bowl")
    if action == "grooming" and counts["red_prop"] != 0:
        raise RuntimeError("Grooming reference unexpectedly contains a prop")

    server_files = {
        "preview": f"{SERVER_ASSET_ROOT}/{preview_path.name}",
        "cutout": f"{SERVER_ASSET_ROOT}/{cutout_path.name}",
        "identity_mask": f"{SERVER_ASSET_ROOT}/{mask_path.name}",
    }
    return {
        "source_filename": source.name,
        "source_sha256": SOURCE_SHA256[action],
        "runtime_filename": runtime.name,
        "runtime_sha256": RUNTIME_SHA256[action],
        "runtime_contract": video_contract,
        "selected_frame_index": selected_index,
        "selected_frame_timestamp_seconds": selected_index / 16.0,
        "selection_policy": (
            "cycle_midpoint_stable_muzzle_inside_bowl_opening"
            if action == "eating"
            else "cycle_midpoint_visible_raised_paw_lick_contact"
        ),
        "segmentation_policy": "largest_connected_green_screen_subject_soft_alpha_v1",
        "alpha_bbox_xyxy": alpha_bbox,
        "alpha_pixels_ge_16": int(np.count_nonzero(alpha_support)),
        "border_contact_pixels": border_contact,
        "border_contact_policy": (
            "original_maincase_lower_tail_may_touch_right_border"
            if action == "eating"
            else "no_border_contact"
        ),
        "identity_mask_policy": mask_policy,
        "identity_mask_counts": counts,
        "prop_split": prop_metadata,
        "preview": server_files["preview"],
        "preview_sha256": sha256_file(preview_path),
        "cutout": server_files["cutout"],
        "cutout_sha256": sha256_file(cutout_path),
        "identity_mask": server_files["identity_mask"],
        "identity_mask_sha256": sha256_file(mask_path),
        "local_preview": str(preview_path.resolve()),
        "local_cutout": str(cutout_path.resolve()),
        "local_identity_mask": str(mask_path.resolve()),
        "validation": {
            "scope": "deterministic_cpu_profile_asset",
            "status": "passed",
            "full_gpu_generation": "not_applicable_profile_standard_direct_no_redraw",
            "end_to_end_api_validation": "pending",
        },
    }


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[1]
    default_output = (
        project
        / "subject_profiles"
        / "assets"
        / PROFILE_ID
        / REVISION
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=project)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    driving_dir = output_dir / "driving"
    inputs = {
        "eating": {
            "source": project / "Main_case" / "eat.mp4",
            "runtime": driving_dir / "eating_driver_512x512_81f_16fps.mp4",
        },
        "grooming": {
            "source": project / "Main_case" / "TM.mp4",
            "runtime": driving_dir / "grooming_driver_512x512_81f_16fps.mp4",
        },
    }
    records = {
        action: build_action(action, values["source"], values["runtime"], output_dir)
        for action, values in inputs.items()
    }
    board_path = output_dir / "reference_asset_audit_board.png"
    make_audit_board(records, board_path)

    clean_records = {
        action: {
            key: value
            for key, value in record.items()
            if not key.startswith("local_")
        }
        for action, record in records.items()
    }
    manifest = {
        "schema_version": 1,
        "profile_id": PROFILE_ID,
        "revision": REVISION,
        "role": "default_subject_identity_geometry_fur_and_motion_standard",
        "asset_validation_run_id": ASSET_VALIDATION_RUN_ID,
        "generator": SERVER_GENERATOR_PATH,
        "generator_sha256": sha256_file(Path(__file__).resolve()),
        "server_asset_root": SERVER_ASSET_ROOT,
        "actions": clean_records,
        "audit_board": f"{SERVER_ASSET_ROOT}/{board_path.name}",
        "audit_board_sha256": sha256_file(board_path),
        "checks": {
            "source_hashes_frozen": True,
            "normalized_81f_hashes_frozen": True,
            "selected_frames_from_normalized_drivers": True,
            "watermark_excluded_by_connected_component": True,
            "transparent_cutouts_have_empty_corners": True,
            "eating_right_border_contact_is_original_lower_tail": True,
            "identity_masks_use_only_black_blue_red": True,
            "old_gray_profile_reference_hashes_present": False,
            "full_gpu_generation_validation": (
                "not_applicable_profile_standard_direct_no_redraw"
            ),
            "end_to_end_api_validation": "pending",
        },
    }
    manifest["content_sha256"] = canonical_json_sha256(manifest)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"manifest_file_sha256={sha256_file(manifest_path)}")


if __name__ == "__main__":
    main()
