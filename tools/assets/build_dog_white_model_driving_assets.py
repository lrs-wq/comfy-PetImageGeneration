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
import re
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any


DEFAULT_ASSET_ROOT = Path(
    "/data/comfy-pet-pipeline/profile_assets/dog_white_model_v1/"
    "driving/20260817-user-white-model-v2"
)
DEFAULT_SOURCE_ROOT = Path(
    "/data/comfy-pet-pipeline/profile_assets/dog_white_model_v1/"
    "driving/20260817-user-white-model-v1/source"
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
    "eating": "eating_white_model_reference_frame008_cutout.png",
    "standing": "standing_white_model_reference_frame040_cutout.png",
}
ACTION_NORMALIZATION = {
    # The supplied eating clip is one complete down -> up -> down cycle.  The
    # previous v1 build kept only its second half (up -> down), which guaranteed
    # a visible GIF loop jump.  Preserve the full cycle and play it at 2x speed
    # so the public 5-second/81-frame contract remains unchanged.
    "eating": {
        "source_start_seconds": 0.0,
        "source_duration_seconds": 10.041667,
        "pts_scale": 0.5,
        "reference_frame_index": 8,
        "minimum_endpoint_ssim": 0.95,
        "loop_policy": "full_source_down_up_down_cycle_2x_v1",
    },
    # Keep the already stable tail-motion window byte-for-byte in timing terms.
    "standing": {
        "source_start_seconds": 5.0175,
        "source_duration_seconds": 5.0625,
        "pts_scale": 1.0,
        "reference_frame_index": 40,
        "minimum_endpoint_ssim": None,
        "loop_policy": "existing_second_half_idle_window_v1",
    },
}
OUTPUT_SIZE = 512
OUTPUT_FRAMES = 81
OUTPUT_FPS = 16
POLICY = "dog_white_model_action_specific_loop_normalization_v2"


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


def build_runtime(
    ffmpeg: str,
    source: Path,
    target: Path,
    normalization: dict[str, Any],
) -> None:
    temporary = target.with_name(f".{target.stem}.tmp{target.suffix}")
    temporary.unlink(missing_ok=True)
    filters = [
        "crop=720:720:(in_w-720)/2:(in_h-720)/2",
        f"scale={OUTPUT_SIZE}:{OUTPUT_SIZE}:flags=lanczos",
    ]
    pts_scale = float(normalization["pts_scale"])
    if abs(pts_scale - 1.0) > 1e-9:
        filters.append(f"setpts={pts_scale:.9f}*PTS")
    filters.extend(
        [
            f"fps={OUTPUT_FPS}",
            "tpad=stop_mode=clone:stop=1",
        ]
    )
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    source_start = float(normalization["source_start_seconds"])
    if source_start > 0.0:
        command.extend(["-ss", str(source_start)])
    command.extend(["-i", str(source)])
    source_duration = float(normalization["source_duration_seconds"])
    if source_duration > 0.0:
        command.extend(["-t", str(source_duration)])
    command.extend(
        [
            "-vf",
            ",".join(filters),
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
    run(command)
    os.replace(temporary, target)


def build_reference(
    ffmpeg: str,
    runtime: Path,
    target: Path,
    frame_index: int,
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
                f"select=eq(n\\,{frame_index}),"
                "chromakey=0x00FF00:0.10:0.03,format=rgba"
            ),
            "-frames:v",
            "1",
            str(temporary),
        ]
    )
    os.replace(temporary, target)


def probe_endpoint_ssim(ffmpeg: str, runtime: Path) -> float:
    """Measure first/last visual continuity of the normalized runtime."""

    with tempfile.TemporaryDirectory(prefix="dog-white-model-loop-") as temporary:
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
                    "-fps_mode",
                    "vfr",
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
        raise RuntimeError("Could not parse normalized driver endpoint SSIM")
    return float(match.group(1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    asset_root = args.asset_root.expanduser().resolve()
    source_root = args.source_root.expanduser().resolve()
    asset_root.mkdir(parents=True, exist_ok=True)
    actions: dict[str, dict[str, Any]] = {}
    for action in ("eating", "standing"):
        normalization = dict(ACTION_NORMALIZATION[action])
        source = source_root / SOURCE_FILENAMES[action]
        if not source.is_file():
            raise FileNotFoundError(f"Missing dog white-model source: {source}")
        runtime = asset_root / RUNTIME_FILENAMES[action]
        reference = asset_root / REFERENCE_FILENAMES[action]
        build_runtime(args.ffmpeg, source, runtime, normalization)
        observed = probe_video(args.ffprobe, runtime)
        reference_frame_index = int(normalization["reference_frame_index"])
        build_reference(
            args.ffmpeg,
            runtime,
            reference,
            reference_frame_index,
        )
        endpoint_ssim = probe_endpoint_ssim(args.ffmpeg, runtime)
        minimum_endpoint_ssim = normalization["minimum_endpoint_ssim"]
        if (
            minimum_endpoint_ssim is not None
            and endpoint_ssim < float(minimum_endpoint_ssim)
        ):
            raise RuntimeError(
                f"{runtime} endpoint SSIM {endpoint_ssim:.6f} is below "
                f"{float(minimum_endpoint_ssim):.6f}"
            )
        actions[action] = {
            "source": str(source),
            "source_sha256": sha256_file(source),
            "runtime": str(runtime),
            "runtime_sha256": sha256_file(runtime),
            "geometry_reference": str(reference),
            "geometry_reference_sha256": sha256_file(reference),
            "source_start_seconds": normalization["source_start_seconds"],
            "source_duration_seconds": normalization["source_duration_seconds"],
            "pts_scale": normalization["pts_scale"],
            "loop_policy": normalization["loop_policy"],
            "endpoint_ssim": endpoint_ssim,
            "minimum_endpoint_ssim": minimum_endpoint_ssim,
            "selected_frame_index": reference_frame_index,
            "selected_frame_timestamp_seconds": reference_frame_index / OUTPUT_FPS,
            "observed_runtime": observed,
        }

    manifest = {
        "schema_version": 1,
        "policy": POLICY,
        "normalization": {
            "source_segment_policy": "action_specific",
            "crop": "center_720x720_from_960x720",
            "size": [OUTPUT_SIZE, OUTPUT_SIZE],
            "frames": OUTPUT_FRAMES,
            "fps": OUTPUT_FPS,
            "audio_removed": True,
            "last_frame_padding": "clone_source_endpoint_to_81_frames",
            "actions": ACTION_NORMALIZATION,
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
