#!/usr/bin/env python3
"""Build digest-bound dog white-model drivers from the supplied source videos.

The script intentionally uses only the Python standard library plus ffmpeg and
ffprobe so it can run from the server's global Python environment.  It keeps
the source motion, normalizes both actions to the SCAIL 512-square contract,
stabilizes the eating bowl to a low level kibble layer, and extracts RGBA
geometry references from action-specific frames.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any


DEFAULT_ASSET_ROOT = Path(
    "/data/comfy-pet-pipeline/profile_assets/dog_white_model_v1/"
    "driving/20260818-stable-kibble-v3"
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
EXPECTED_SOURCE_SHA256 = {
    "eating": "2a4a2571aa1c8fd3f9e65c8f84c5d69205cedc366947d60b1f9144296bd3d7f4",
    "standing": "7af21bd55744a11f6106aa735a6dbb18e8113671c881af2bb4951855ab9e0913",
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
POLICY = "dog_white_model_action_specific_loop_normalization_stable_kibble_v3"
EATING_KIBBLE_STABILIZATION = {
    "policy": "fixed_low_level_small_kibble_below_rim_v1",
    "source_locked_size": [512, 512],
    "bowl_roi": [65, 360, 205, 394],
    "opening_center": [133.5, 378.0],
    "opening_radius": [57.5, 9.5],
    "back_rim_radius": [61.0, 12.0],
    "food_center": [133.5, 379.0],
    "food_radius": [53.0, 6.5],
    "food_pattern_seed": 20260818,
    "food_pellet_count": 65,
    "food_surface_top_y": 373,
    "bowl_rim_top_y": 366,
    "food_above_rim_allowed": False,
    "source_mound_removed": True,
    "food_motion": "fixed_pattern_with_only_natural_muzzle_occlusion",
}


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


def _read_ppm(path: Path) -> tuple[bytes, bytearray]:
    raw = path.read_bytes()
    marker = b"\n255\n"
    try:
        header, payload = raw.split(marker, 1)
    except ValueError as exc:
        raise RuntimeError(f"Unsupported ffmpeg PPM header in {path}") from exc
    tokens = header.split()
    if tokens != [b"P6", b"512", b"512"]:
        raise RuntimeError(f"Unexpected ffmpeg PPM dimensions in {path}: {tokens!r}")
    expected_bytes = OUTPUT_SIZE * OUTPUT_SIZE * 3
    if len(payload) != expected_bytes:
        raise RuntimeError(
            f"Unexpected ffmpeg PPM payload in {path}: "
            f"{len(payload)} bytes, expected {expected_bytes}"
        )
    return header + marker, bytearray(payload)


def _is_source_kibble(r: int, g: int, b: int) -> bool:
    return r >= 65 and r > g + 10 and g > b + 3 and r - b >= 26


def _is_chroma_green(r: int, g: int, b: int) -> bool:
    return g >= 150 and g > r + 70 and g > b + 70


def _stable_kibble_pattern() -> dict[tuple[int, int], tuple[int, int, int]]:
    """Create one deterministic, low, sparse layer of small dry-food pellets."""

    center_x, center_y = EATING_KIBBLE_STABILIZATION["food_center"]
    radius_x, radius_y = EATING_KIBBLE_STABILIZATION["food_radius"]
    food_pixels: dict[tuple[int, int], tuple[int, int, int]] = {}
    for y in range(370, 388):
        for x in range(77, 191):
            distance = (
                ((x - center_x) / radius_x) ** 2
                + ((y - center_y) / radius_y) ** 2
            )
            if distance <= 1.0:
                food_pixels[(x, y)] = (78, 46, 26)

    random_source = random.Random(
        int(EATING_KIBBLE_STABILIZATION["food_pattern_seed"])
    )
    centers: list[tuple[float, float, float, float, int]] = []
    requested_pellets = int(EATING_KIBBLE_STABILIZATION["food_pellet_count"])
    attempts = 0
    while len(centers) < requested_pellets:
        attempts += 1
        if attempts > requested_pellets * 1000:
            raise RuntimeError("Could not place the stable low kibble pattern")
        x = random_source.uniform(82, 186)
        y = random_source.uniform(373.5, 384.5)
        if ((x - center_x) / 50.5) ** 2 + ((y - center_y) / 5.4) ** 2 > 1.0:
            continue
        if any(
            ((x - old_x) / 3.0) ** 2 + ((y - old_y) / 1.55) ** 2 < 1.0
            for old_x, old_y, *_rest in centers
        ):
            continue
        centers.append(
            (
                x,
                y,
                random_source.uniform(1.45, 1.9),
                random_source.uniform(0.8, 1.1),
                random_source.randint(-10, 12),
            )
        )

    for pellet_x, pellet_y, pellet_rx, pellet_ry, jitter in centers:
        for y in range(
            int(pellet_y - pellet_ry - 1),
            int(pellet_y + pellet_ry + 2),
        ):
            for x in range(
                int(pellet_x - pellet_rx - 1),
                int(pellet_x + pellet_rx + 2),
            ):
                distance = (
                    ((x - pellet_x) / pellet_rx) ** 2
                    + ((y - pellet_y) / pellet_ry) ** 2
                )
                if distance > 1.0 or (x, y) not in food_pixels:
                    continue
                red = max(0, min(255, 132 + jitter - int(10 * distance)))
                green = 78 + jitter // 2
                blue = 40 + jitter // 3
                if x <= pellet_x and y <= pellet_y:
                    red = min(255, red + 9)
                    green = min(255, green + 6)
                    blue = min(255, blue + 3)
                food_pixels[(x, y)] = (red, green, blue)
    return food_pixels


def _stabilize_eating_ppm(
    path: Path,
    food_pixels: dict[tuple[int, int], tuple[int, int, int]],
) -> dict[str, int]:
    """Remove the source mound and redraw a fixed food plane under the rim."""

    prefix, pixels = _read_ppm(path)
    left, top, right, bottom = EATING_KIBBLE_STABILIZATION["bowl_roi"]
    center_x, center_y = EATING_KIBBLE_STABILIZATION["opening_center"]
    opening_rx, opening_ry = EATING_KIBBLE_STABILIZATION["opening_radius"]
    rim_rx, rim_ry = EATING_KIBBLE_STABILIZATION["back_rim_radius"]
    source_kibble_pixels = 0
    source_mound_pixels_removed = 0
    rendered_food_pixels = 0
    forbidden_food_pixels_above_rim = 0
    for y in range(top, bottom):
        for x in range(left, right):
            offset = (y * OUTPUT_SIZE + x) * 3
            red, green, blue = pixels[offset : offset + 3]
            source_kibble = _is_source_kibble(red, green, blue)
            chroma_green = _is_chroma_green(red, green, blue)
            if source_kibble:
                source_kibble_pixels += 1
            if not (source_kibble or chroma_green):
                # White/gray/black dog pixels are deliberately left untouched, so
                # the muzzle naturally occludes the otherwise fixed food layer.
                continue

            outer_distance = (
                ((x - center_x) / rim_rx) ** 2
                + ((y - center_y) / rim_ry) ** 2
            )
            opening = (
                ((x - center_x) / opening_rx) ** 2
                + ((y - center_y) / opening_ry) ** 2
                <= 1.0
            )
            back_rim = y <= center_y and 0.78 <= outer_distance <= 1.08
            if back_rim:
                shade = max(150, min(215, 196 + (y - 370) * 2))
                replacement = (shade, shade, shade)
            elif (x, y) in food_pixels:
                replacement = food_pixels[(x, y)]
                rendered_food_pixels += 1
            elif opening:
                replacement = (74, 70, 64)
            elif source_kibble:
                replacement = (0, 255, 0)
                source_mound_pixels_removed += 1
            else:
                continue
            pixels[offset : offset + 3] = bytes(replacement)

    rim_top = int(EATING_KIBBLE_STABILIZATION["bowl_rim_top_y"])
    for y in range(top, rim_top + 1):
        for x in range(left, right):
            offset = (y * OUTPUT_SIZE + x) * 3
            red, green, blue = pixels[offset : offset + 3]
            if _is_source_kibble(red, green, blue):
                forbidden_food_pixels_above_rim += 1

    path.write_bytes(prefix + bytes(pixels))
    return {
        "source_kibble_pixels": source_kibble_pixels,
        "source_mound_pixels_removed": source_mound_pixels_removed,
        "rendered_food_pixels": rendered_food_pixels,
        "forbidden_food_pixels_above_rim": forbidden_food_pixels_above_rim,
    }


def stabilize_eating_runtime(ffmpeg: str, runtime: Path) -> dict[str, Any]:
    """Apply the source-locked bowl correction using only stdlib and ffmpeg."""

    temporary = runtime.with_name(f".{runtime.stem}.stable.tmp{runtime.suffix}")
    temporary.unlink(missing_ok=True)
    food_pixels = _stable_kibble_pattern()
    with tempfile.TemporaryDirectory(prefix="dog-white-model-stable-kibble-") as root:
        frames_root = Path(root)
        run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(runtime),
                str(frames_root / "frame-%03d.ppm"),
            ]
        )
        frame_paths = sorted(frames_root.glob("frame-*.ppm"))
        if len(frame_paths) != OUTPUT_FRAMES:
            raise RuntimeError(
                f"Expected {OUTPUT_FRAMES} normalized eating frames, "
                f"found {len(frame_paths)}"
            )
        frame_audits = [
            _stabilize_eating_ppm(frame_path, food_pixels)
            for frame_path in frame_paths
        ]
        run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-framerate",
                str(OUTPUT_FPS),
                "-i",
                str(frames_root / "frame-%03d.ppm"),
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
    os.replace(temporary, runtime)
    food_rows = [y for _x, y in food_pixels]
    food_top = min(food_rows)
    food_bottom = max(food_rows) + 1
    rim_top = int(EATING_KIBBLE_STABILIZATION["bowl_rim_top_y"])
    if food_top <= rim_top:
        raise RuntimeError(
            f"Stable food starts at y={food_top}, not below rim y={rim_top}"
        )
    forbidden_food_pixels_above_rim = max(
        item["forbidden_food_pixels_above_rim"] for item in frame_audits
    )
    if forbidden_food_pixels_above_rim:
        raise RuntimeError(
            "Stable eating runtime contains food-colored pixels above the bowl "
            f"rim in at least one frame: {forbidden_food_pixels_above_rim}"
        )
    pattern_bytes = b"".join(
        x.to_bytes(2, "big") + y.to_bytes(2, "big") + bytes(rgb)
        for (x, y), rgb in sorted(food_pixels.items())
    )
    return {
        **EATING_KIBBLE_STABILIZATION,
        "food_pattern_sha256": hashlib.sha256(pattern_bytes).hexdigest(),
        "food_pattern_pixels": len(food_pixels),
        "food_pattern_top_y": food_top,
        "food_pattern_bottom_y_exclusive": food_bottom,
        "source_kibble_pixels_min": min(
            item["source_kibble_pixels"] for item in frame_audits
        ),
        "source_kibble_pixels_max": max(
            item["source_kibble_pixels"] for item in frame_audits
        ),
        "source_mound_pixels_removed_min": min(
            item["source_mound_pixels_removed"] for item in frame_audits
        ),
        "source_mound_pixels_removed_max": max(
            item["source_mound_pixels_removed"] for item in frame_audits
        ),
        "rendered_food_pixels_min": min(
            item["rendered_food_pixels"] for item in frame_audits
        ),
        "rendered_food_pixels_max": max(
            item["rendered_food_pixels"] for item in frame_audits
        ),
        "frames_audited": len(frame_audits),
        "forbidden_food_pixels_above_rim": forbidden_food_pixels_above_rim,
        "passed": True,
    }


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
        source_sha256 = sha256_file(source)
        expected_source_sha256 = EXPECTED_SOURCE_SHA256[action]
        if source_sha256 != expected_source_sha256:
            raise RuntimeError(
                f"{source} has unexpected SHA-256 {source_sha256}; "
                f"expected {expected_source_sha256}. The fixed 512px bowl "
                "stabilization is source-geometry locked."
            )
        runtime = asset_root / RUNTIME_FILENAMES[action]
        reference = asset_root / REFERENCE_FILENAMES[action]
        build_runtime(args.ffmpeg, source, runtime, normalization)
        eating_food_stabilization = None
        if action == "eating":
            eating_food_stabilization = stabilize_eating_runtime(
                args.ffmpeg,
                runtime,
            )
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
        action_manifest = {
            "source": str(source),
            "source_sha256": source_sha256,
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
        if eating_food_stabilization is not None:
            action_manifest["food_stabilization"] = eating_food_stabilization
        actions[action] = action_manifest

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
            "eating_food_stabilization": EATING_KIBBLE_STABILIZATION,
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
