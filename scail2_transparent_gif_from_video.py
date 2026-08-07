#!/usr/bin/env python3
"""Build the unstabilized transparent GIF from an existing SCAIL video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scail2_subject_replacement_test import (
    build_unstabilized_transparent_gif,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("guidance_mask_video", type=Path)
    parser.add_argument("gif_target", type=Path)
    parser.add_argument("--white-mp4-target", type=Path)
    args = parser.parse_args()
    metrics = build_unstabilized_transparent_gif(
        args.video,
        args.gif_target,
        guidance_mask_video=args.guidance_mask_video,
        white_mp4_target=args.white_mp4_target,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
