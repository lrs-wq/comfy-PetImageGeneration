#!/usr/bin/env python3
"""Build the unstabilized transparent GIF from an existing SCAIL video."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.comfy_pet_pipeline.pipeline.subject_replacement import (  # noqa: E402
    build_unstabilized_transparent_gif,
)

SUPPORTED_ACTIONS = (
    "eating",
    "grooming",
    "drinking",
    "running",
    "sleeping",
    "standing",
    "toileting",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("guidance_mask_video", type=Path)
    parser.add_argument("gif_target", type=Path)
    parser.add_argument("--action", choices=SUPPORTED_ACTIONS)
    parser.add_argument("--white-mp4-target", type=Path)
    parser.add_argument("--soft-apng-target", type=Path)
    parser.add_argument("--start-frame-index", type=int, default=0)
    parser.add_argument("--source-fps", type=float, default=16.0)
    parser.add_argument("--gif-fps", type=int, default=20)
    parser.add_argument("--gif-frames", type=int, default=100)
    parser.add_argument(
        "--reference-aligned-uniform-background",
        action="store_true",
        help=(
            "Extract a complete subject+prop matte from a uniform canvas using "
            "the supplied immutable reference mask."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metrics = build_unstabilized_transparent_gif(
        args.video,
        args.gif_target,
        action=args.action,
        guidance_mask_video=args.guidance_mask_video,
        reference_aligned_uniform_background=(
            args.reference_aligned_uniform_background
        ),
        source_fps=args.source_fps,
        gif_fps=args.gif_fps,
        gif_frames=args.gif_frames,
        white_mp4_target=args.white_mp4_target,
        soft_apng_target=args.soft_apng_target,
        start_frame_index=args.start_frame_index,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
