#!/usr/bin/env python3
"""Probe Main_case TM Image 1 geometry against uploaded Image 2 identity."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.comfy_pet_pipeline.api import local_video as pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quality-retry", type=int, default=3)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--denoise", type=float, default=1.0)
    parser.add_argument("--no-style", action="store_true")
    parser.add_argument("--geometry-first", action="store_true")
    args = parser.parse_args()

    relative_root = Path("local_video_api") / args.job_id
    identity_name = (relative_root / "input.png").as_posix()
    guide_name = (relative_root / "grooming_qwen_pose_guide.png").as_posix()
    identity_path = pipeline.COMFY_INPUT / identity_name
    guide_path = pipeline.COMFY_INPUT / guide_name
    if not identity_path.is_file():
        raise FileNotFoundError(identity_path)
    if not guide_path.is_file():
        raise FileNotFoundError(guide_path)

    prompt = pipeline.build_qwen_prompt(
        "",
        pipeline.subject_replacement_pose_prompt("grooming"),
        transparent_gif=True,
        quality_retry=args.quality_retry,
        pose_reference_available=True,
        profile_fur_lock=(
            "dense compact uniformly directed coat lying naturally close to the "
            "body; exact smooth rounded back contour; short soft edge hairs"
        ),
        pose_identity_input_policy=(
            pipeline.POSE_IDENTITY_INPUT_POLICY_PROFILE_GEOMETRY_FIRST
            if args.geometry_first
            else pipeline.POSE_IDENTITY_INPUT_POLICY_DIRECT_UPLOAD
        ),
    )
    if args.geometry_first:
        prompt += """

MAIN_CASE IMAGE ROLE OVERRIDE — non-negotiable: Image 1 is the texture-free geometry guide derived directly from the frozen Main_case TM.mp4 standard and is the sampler geometry canvas. It controls only seated grooming posture, raised forepaw-to-mouth relation, compact full-body layout, camera and tight coat silhouette. It contains no identity, black paw, accessory, or style. Image 2 is the original uploaded close-up and controls only the exact face, eyes, ears, nose, muzzle, fur base colors and markings; do not copy its crop, background, pose, photographic lighting, apparent fur length, cheek volume, or infer a mane, ruff, long body coat or expanded volume. Render the exact identity from Image 2 in Image 1's complete Main_case grooming geometry. The written premium stylized 3D target and profile coat surface control rendering and grooming. The raised shape at the mouth must be the cat's own correctly colored foreleg and paw, never black, detached, glove-like, forked, or an object."""
    else:
        prompt += """

MAIN_CASE IMAGE 2 OVERRIDE — non-negotiable: Image 2 is the texture-free geometry guide derived directly from the frozen Main_case TM.mp4 standard. It controls only seated grooming posture, raised forepaw-to-mouth relation, compact full-body layout, camera and tight coat silhouette. It contains no identity or style. Image 1 is the original uploaded close-up and controls only face, eyes, ears, nose, muzzle, fur base colors and markings; do not infer a mane, ruff, long body coat or expanded volume from the crop. The written premium stylized 3D target and profile coat surface control rendering and grooming."""
    palette = pipeline.identity_palette_signature(identity_path)
    prompt += "\n\n" + pipeline.identity_palette_prompt(
        palette,
        retry=args.quality_retry > 1,
    )
    negative = (
        pipeline.QWEN_NEGATIVE
        + ", "
        + pipeline.CARTOON_NEGATIVE
        + ", "
        + pipeline.CHROMA_NEGATIVE
        + ", "
        + pipeline.SLEEK_PROFILE_FUR_NEGATIVE
        + ", "
        + pipeline.identity_palette_negative_prompt(palette)
    )
    workflow = pipeline.qwen_workflow(
        guide_name if args.geometry_first else identity_name,
        prompt,
        negative,
        args.seed,
        f"codex_probe/{args.job_id}_grooming_raw_identity",
        user_loras=[],
        steps=args.steps,
        cfg=1.0,
        denoise=args.denoise,
        builtin_style_strength=0.15,
        suppress_builtin_style=args.no_style,
        pose_reference_input_name=(
            identity_name if args.geometry_first else guide_name
        ),
    )
    entry = pipeline.queue_and_wait(
        workflow,
        "grooming raw-identity Main_case probe",
        timeout=1200,
    )
    source = pipeline.first_output(
        entry,
        {".png", ".jpg", ".jpeg", ".webp"},
        "grooming raw-identity Main_case probe",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
