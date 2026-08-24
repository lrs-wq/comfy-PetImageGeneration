#!/usr/bin/env python3
"""Build the test-only small-dog eating and standing white-model assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_ROOT = Path(
    "/data/comfy-pet-pipeline/profile_assets/dog_white_model_catalog/"
    "20260824-v1/small_dog/source"
)
DEFAULT_ASSET_ROOT = Path(
    "/data/comfy-pet-pipeline/profile_assets/dog_white_model_catalog/"
    "20260824-v1/small_dog/runtime-v1"
)
SOURCE_FILENAMES = {
    "eating": "eating.mp4",
    "standing": "standing_head_turn.mp4",
}
EXPECTED_SOURCE_SHA256 = {
    "eating": "f1bc84b241d42fec559d8f9fa22d2c5a7d77ac9943ae66a29813f6f6c3b058c9",
    "standing": "1de916b209409aed0cdcbfccc4db77e7d32d298c6c88d35625ae50b59fbac06f",
}
RUNTIME_FILENAMES = {
    action: f"small_dog_{action}_driver_512x512_81f_16fps.mp4"
    for action in SOURCE_FILENAMES
}
REFERENCE_FILENAMES = {
    action: f"small_dog_{action}_reference_frame040_cutout.png"
    for action in SOURCE_FILENAMES
}
OUTPUT_SIZE = 512
OUTPUT_FRAMES = 81
OUTPUT_FPS = 16
SOURCE_DURATION_SECONDS = 8.0
PTS_SCALE = 0.625
REFERENCE_FRAME_INDEX = 40
MINIMUM_ENDPOINT_SSIM = 0.90
CHROMA_KEY = "0x55A74E"
CHROMA_SIMILARITY = {
    "eating": 0.12,
    "standing": 0.21,
}
CHROMA_BLEND = 0.05
POLICY = "small_dog_test_platform_square_full_cycle_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def probe_video(ffprobe: str, path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_read_frames,pix_fmt",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    stream = (payload.get("streams") or [None])[0]
    if not isinstance(stream, dict):
        raise RuntimeError(f"Expected one video stream in {path}")
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frames": int(stream["nb_read_frames"]),
        "fps": float(Fraction(str(stream["avg_frame_rate"]))),
        "pixel_format": str(stream.get("pix_fmt") or ""),
        "duration_seconds": float(payload.get("format", {}).get("duration") or 0),
    }


def assert_runtime_contract(observed: dict[str, Any], path: Path) -> None:
    expected = {
        "width": OUTPUT_SIZE,
        "height": OUTPUT_SIZE,
        "frames": OUTPUT_FRAMES,
        "fps": float(OUTPUT_FPS),
    }
    for key, value in expected.items():
        if abs(float(observed[key]) - float(value)) > 0.001:
            raise RuntimeError(
                f"{path} has invalid {key}: {observed[key]!r}, expected {value!r}"
            )


def build_runtime(ffmpeg: str, source: Path, target: Path) -> None:
    temporary = target.with_name(f".{target.stem}.tmp{target.suffix}")
    temporary.unlink(missing_ok=True)
    filters = (
        "crop=min(iw\\,ih):min(iw\\,ih):(iw-ow)/2:(ih-oh)/2,"
        f"scale={OUTPUT_SIZE}:{OUTPUT_SIZE}:flags=lanczos,"
        f"setpts={PTS_SCALE:.9f}*PTS,fps={OUTPUT_FPS},"
        "tpad=stop_mode=clone:stop=1"
    )
    run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-t",
            str(SOURCE_DURATION_SECONDS),
            "-vf",
            filters,
            "-frames:v",
            str(OUTPUT_FRAMES),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(temporary),
        ]
    )
    os.replace(temporary, target)


def build_reference(
    ffmpeg: str,
    runtime: Path,
    target: Path,
    *,
    action: str,
) -> None:
    temporary = target.with_name(f".{target.stem}.tmp{target.suffix}")
    temporary.unlink(missing_ok=True)
    run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(runtime),
            "-vf",
            (
                f"select=eq(n\\,{REFERENCE_FRAME_INDEX}),"
                f"colorkey={CHROMA_KEY}:{CHROMA_SIMILARITY[action]}:{CHROMA_BLEND},"
                "format=rgba"
            ),
            "-frames:v",
            "1",
            str(temporary),
        ]
    )
    os.replace(temporary, target)


def probe_reference(ffmpeg: str, ffprobe: str, path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,pix_fmt",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = (json.loads(completed.stdout).get("streams") or [None])[0]
    if not isinstance(stream, dict):
        raise RuntimeError(f"Expected one image stream in {path}")
    observed = {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "pixel_format": str(stream.get("pix_fmt") or ""),
    }
    if (
        observed["width"] != OUTPUT_SIZE
        or observed["height"] != OUTPUT_SIZE
        or "a" not in observed["pixel_format"]
    ):
        raise RuntimeError(f"{path} is not a 512x512 alpha reference: {observed}")
    alpha = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(path),
            "-vf",
            "alphaextract",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ],
        check=True,
        capture_output=True,
    ).stdout
    expected_pixels = OUTPUT_SIZE * OUTPUT_SIZE
    if len(alpha) != expected_pixels:
        raise RuntimeError(
            f"{path} alpha plane has {len(alpha)} bytes; expected {expected_pixels}"
        )
    alpha_minimum = min(alpha)
    alpha_maximum = max(alpha)
    visible_fraction = sum(value > 0 for value in alpha) / expected_pixels
    opaque_fraction = alpha.count(255) / expected_pixels
    if (
        alpha_minimum != 0
        or alpha_maximum != 255
        or not 0.15 <= visible_fraction <= 0.65
        or opaque_fraction <= 0.10
    ):
        raise RuntimeError(
            f"{path} has an unusable alpha matte: min={alpha_minimum}, "
            f"max={alpha_maximum}, visible={visible_fraction:.6f}, "
            f"opaque={opaque_fraction:.6f}"
        )
    observed.update(
        {
            "alpha_minimum": alpha_minimum,
            "alpha_maximum": alpha_maximum,
            "visible_fraction": visible_fraction,
            "opaque_fraction": opaque_fraction,
        }
    )
    return observed


def probe_endpoint_ssim(ffmpeg: str, runtime: Path) -> float:
    with tempfile.TemporaryDirectory(prefix="small-dog-loop-") as temporary:
        root = Path(temporary)
        first = root / "first.png"
        last = root / "last.png"
        for frame_index, target in ((0, first), (OUTPUT_FRAMES - 1, last)):
            run(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(runtime),
                    "-vf",
                    f"select=eq(n\\,{frame_index})",
                    "-frames:v",
                    "1",
                    str(target),
                ]
            )
        completed = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-i",
                str(first),
                "-i",
                str(last),
                "-lavfi",
                "ssim",
                "-f",
                "null",
                "-",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    match = re.search(r"All:([0-9.]+)", completed.stderr)
    if match is None:
        raise RuntimeError(f"Could not parse endpoint SSIM for {runtime}")
    return float(match.group(1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    asset_root = args.asset_root.expanduser().resolve()
    asset_root.mkdir(parents=True, exist_ok=True)
    actions: dict[str, dict[str, Any]] = {}
    for action in ("eating", "standing"):
        source = source_root / SOURCE_FILENAMES[action]
        if not source.is_file():
            raise FileNotFoundError(f"Missing small-dog source: {source}")
        source_digest = sha256_file(source)
        if source_digest != EXPECTED_SOURCE_SHA256[action]:
            raise RuntimeError(
                f"Unexpected SHA-256 for {source}: {source_digest}; "
                f"expected {EXPECTED_SOURCE_SHA256[action]}"
            )
        runtime = asset_root / RUNTIME_FILENAMES[action]
        reference = asset_root / REFERENCE_FILENAMES[action]
        build_runtime(args.ffmpeg, source, runtime)
        observed_runtime = probe_video(args.ffprobe, runtime)
        assert_runtime_contract(observed_runtime, runtime)
        endpoint_ssim = probe_endpoint_ssim(args.ffmpeg, runtime)
        if endpoint_ssim < MINIMUM_ENDPOINT_SSIM:
            raise RuntimeError(
                f"{runtime} endpoint SSIM {endpoint_ssim:.6f} is below "
                f"{MINIMUM_ENDPOINT_SSIM:.6f}"
            )
        build_reference(args.ffmpeg, runtime, reference, action=action)
        observed_reference = probe_reference(args.ffmpeg, args.ffprobe, reference)
        actions[action] = {
            "source": str(source),
            "source_sha256": source_digest,
            "runtime": str(runtime),
            "runtime_sha256": sha256_file(runtime),
            "geometry_reference": str(reference),
            "geometry_reference_sha256": sha256_file(reference),
            "selected_frame_index": REFERENCE_FRAME_INDEX,
            "selected_frame_timestamp_seconds": (
                REFERENCE_FRAME_INDEX / OUTPUT_FPS
            ),
            "source_duration_seconds": SOURCE_DURATION_SECONDS,
            "pts_scale": PTS_SCALE,
            "loop_policy": "full_source_cycle_speed_normalized_v1",
            "endpoint_ssim": endpoint_ssim,
            "minimum_endpoint_ssim": MINIMUM_ENDPOINT_SSIM,
            "observed_runtime": observed_runtime,
            "observed_reference": observed_reference,
        }

    manifest = {
        "schema_version": 1,
        "policy": POLICY,
        "category": "small_dog",
        "scope": "test_platform_only",
        "normalization": {
            "crop": "center_square_576x576_from_736x576",
            "size": [OUTPUT_SIZE, OUTPUT_SIZE],
            "frames": OUTPUT_FRAMES,
            "fps": OUTPUT_FPS,
            "audio_removed": True,
            "source_duration_seconds": SOURCE_DURATION_SECONDS,
            "pts_scale": PTS_SCALE,
            "reference_frame_index": REFERENCE_FRAME_INDEX,
            "chroma_key": CHROMA_KEY,
            "chroma_similarity": CHROMA_SIMILARITY,
            "chroma_blend": CHROMA_BLEND,
        },
        "actions": actions,
    }
    manifest_path = asset_root / "manifest.json"
    temporary = manifest_path.with_name(".manifest.tmp.json")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, manifest_path)
    print(json.dumps({**manifest, "manifest": str(manifest_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
