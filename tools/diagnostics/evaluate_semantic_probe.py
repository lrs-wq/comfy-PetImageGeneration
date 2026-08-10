#!/usr/bin/env python3
"""Evaluate a Qwen action probe through the production recovery gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.comfy_pet_pipeline.api import local_video as pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--identity-source", type=Path)
    parser.add_argument(
        "--pose",
        choices=("eating", "grooming"),
        default="eating",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    green = args.output_dir / "semantic_green.png"
    cutout = args.output_dir / "semantic_cutout.png"
    raw_cutout = args.output_dir / "semantic_cutout_raw.png"
    direct_cutout = args.output_dir / "direct_cutout.png"
    direct_chroma = pipeline.evaluate_chroma_cutout(args.source, direct_cutout)
    chroma = pipeline.semantic_background_recovery(
        args.source,
        green,
        cutout,
        raw_cutout,
        pose=args.pose,
    )
    quality = pipeline.pose_cutout_quality(
        cutout,
        args.pose,
        require_eating_mouth_contact=args.pose == "eating",
    )
    fur = (
        pipeline.eating_fur_silhouette_quality(cutout, args.reference)
        if args.pose == "eating" and args.reference is not None
        else None
    )
    palette_reference = None
    upper_palette_reference = None
    direct_palette = None
    semantic_palette = None
    if args.identity_source is not None:
        bottom_exclusion = (
            pipeline.BOWL_IDENTITY_PALETTE_BOTTOM_EXCLUSION
            if args.pose == "eating"
            else 0.0
        )
        palette_reference = pipeline.identity_palette_signature(
            args.identity_source,
            exclude_bottom_fraction=bottom_exclusion,
        )
        upper_palette_reference = (
            pipeline.identity_palette_signature(
                args.identity_source,
                exclude_bottom_fraction=(
                    pipeline.BOWL_IDENTITY_PALETTE_UPPER_BODY_EXCLUSION
                ),
            )
            if args.pose == "eating"
            else None
        )
        if direct_chroma.get("acceptable"):
            direct_palette = pipeline.evaluate_pose_identity_palette(
                direct_cutout,
                primary_reference=palette_reference,
                upper_body_reference=upper_palette_reference,
                bottom_exclusion=(
                    bottom_exclusion
                ),
                bowl_action=args.pose == "eating",
            )
        semantic_palette = pipeline.evaluate_pose_identity_palette(
            cutout,
            primary_reference=palette_reference,
            upper_body_reference=upper_palette_reference,
            bottom_exclusion=bottom_exclusion,
            bowl_action=args.pose == "eating",
        )
    print(
        json.dumps(
            {
                "api_version": pipeline.API_VERSION,
                "chroma": chroma,
                "direct_chroma": direct_chroma,
                "quality": quality,
                "fur": fur,
                "palette_reference": palette_reference,
                "upper_palette_reference": upper_palette_reference,
                "direct_palette": direct_palette,
                "semantic_palette": semantic_palette,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
