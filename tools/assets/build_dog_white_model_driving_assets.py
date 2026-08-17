#!/usr/bin/env python3
"""Build digest-bound dog white-model drivers from the supplied source videos.

The script intentionally uses only the Python standard library plus ffmpeg and
ffprobe so it can run from the server's global Python environment.  It keeps
the source motion, normalizes both actions to the SCAIL 512-square contract,
and extracts an RGBA geometry reference from frame 40.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any


DEFAULT_ASSET_ROOT = Path(
    "/data/comfy-pet-pipeline/profile_assets/dog_white_model_v1/"
    "driving/20260817-user-white-model-v1"
)
SOURCE_FILENAMES = {
    "eating": "eat.mp4",
    "standing": "stand.mp4",
}
RUNTIME_FILENAMES = {
    "eating": "eating_white_model_driver_512x512_81f_16fps.mp4",
    "standing": "standing_white_model_driver_512x512_81f_16fps.mp4",
}
REFERENCE_FILENAMES = {
    "eating": "eating_white_model_reference_frame040_cutout.png",
    "standing": "standing_white_model_reference_frame040_cutout.png",
}
SOURCE_START_SECONDS = 5.0175
SOURCE_DURATION_SECONDS = 5.0625
OUTPUT_SIZE = 512
OUTPUT_FRAMES = 81
OUTPUT_FPS = 16
REFERENCE_FRAME_INDEX = 40
POLICY = "dog_white_model_source_normalization_v1"


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
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    streams = payload.get("streams") or []
    if len(streams) != 1:
        raise RuntimeError(f"Expected one video stream in {path}")
    stream = streams[0]
    fps = float(Fraction(str(stream["avg_frame_rate"])))
    observed = {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frames": int(stream["nb_read_frames"]),
        "fps": fps,
        "pix_fmt": str(stream.get("pix_fmt") or ""),
    }
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
    return observed


def build_runtime(ffmpeg: str, source: Path, target: Path) -> None:
    temporary = target.with_name(f".{target.stem}.tmp{target.suffix}")
    temporary.unlink(missing_ok=True)
    video_filter = (
        "crop=720:720:(in_w-720)/2:(in_h-720)/2,"
        f"scale={OUTPUT_SIZE}:{OUTPUT_SIZE}:flags=lanczos,"
        f"fps={OUTPUT_FPS},tpad=stop_mode=clone:stop=1"
    )
    run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            str(SOURCE_START_SECONDS),
            "-i",
            str(source),
            "-t",
            str(SOURCE_DURATION_SECONDS),
            "-vf",
            video_filter,
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


def build_reference(ffmpeg: str, runtime: Path, target: Path) -> None:
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
                "chromakey=0x00FF00:0.10:0.03,format=rgba"
            ),
            "-frames:v",
            "1",
            str(temporary),
        ]
    )
    os.replace(temporary, target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    asset_root = args.asset_root.expanduser().resolve()
    source_root = asset_root / "source"
    asset_root.mkdir(parents=True, exist_ok=True)
    actions: dict[str, dict[str, Any]] = {}
    for action in ("eating", "standing"):
        source = source_root / SOURCE_FILENAMES[action]
        if not source.is_file():
            raise FileNotFoundError(f"Missing dog white-model source: {source}")
        runtime = asset_root / RUNTIME_FILENAMES[action]
        reference = asset_root / REFERENCE_FILENAMES[action]
        build_runtime(args.ffmpeg, source, runtime)
        observed = probe_video(args.ffprobe, runtime)
        build_reference(args.ffmpeg, runtime, reference)
        actions[action] = {
            "source": str(source),
            "source_sha256": sha256_file(source),
            "runtime": str(runtime),
            "runtime_sha256": sha256_file(runtime),
            "geometry_reference": str(reference),
            "geometry_reference_sha256": sha256_file(reference),
            "selected_frame_index": REFERENCE_FRAME_INDEX,
            "selected_frame_timestamp_seconds": REFERENCE_FRAME_INDEX / OUTPUT_FPS,
            "observed_runtime": observed,
        }

    manifest = {
        "schema_version": 1,
        "policy": POLICY,
        "normalization": {
            "source_segment_start_seconds": SOURCE_START_SECONDS,
            "source_segment_duration_seconds": SOURCE_DURATION_SECONDS,
            "crop": "center_720x720_from_960x720",
            "size": [OUTPUT_SIZE, OUTPUT_SIZE],
            "frames": OUTPUT_FRAMES,
            "fps": OUTPUT_FPS,
            "audio_removed": True,
            "last_frame_padding": "clone_source_endpoint_to_81_frames",
        },
        "actions": actions,
    }
    manifest_path = asset_root / "manifest.json"
    temporary_manifest = manifest_path.with_name(".manifest.tmp.json")
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, manifest_path)
    print(json.dumps({**manifest, "manifest": str(manifest_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
