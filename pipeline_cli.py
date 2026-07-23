from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent


def data_root() -> Path:
    return Path(os.environ.get("PET_PIPELINE_DATA_ROOT", PROJECT_ROOT / "outputs")).expanduser().resolve()


def default_input() -> Path:
    return data_root() / "inputs" / "1.jpg"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified pet image and video pipeline.")
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pose = subparsers.add_parser("pose", help="Generate transparent cartoon poses.")
    pose.add_argument("--image", type=Path)
    pose.add_argument("--output-dir", type=Path)
    pose.add_argument("--count", type=int, default=6)

    views = subparsers.add_parser("three-view", help="Generate front/side/back views.")
    views.add_argument("--image", type=Path)
    views.add_argument("--output-dir", type=Path)
    views.add_argument("--reference-dir", type=Path)

    eat = subparsers.add_parser("eat-video", help="Generate an eating video from one pose.")
    eat.add_argument("--pose-image", type=Path, required=True)
    eat.add_argument("--output-dir", type=Path)
    eat.add_argument("--image-model", default="qwen-image-edit")
    eat.add_argument("--video-model", default="wan2.2-i2v-flash")
    eat.add_argument("--resolution", default="720P")
    eat.add_argument("--duration", type=int, default=5)
    eat.add_argument("--seed", type=int, default=20260622)
    eat.add_argument("--poll-interval", type=int, default=15)
    eat.add_argument("--timeout-seconds", type=int, default=900)

    actions = subparsers.add_parser("two-actions", help="Generate poses plus eating and running videos.")
    actions.add_argument("--image", type=Path)
    actions.add_argument("--output-dir", type=Path)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    load_dotenv(args.env_file)
    root = data_root()
    image = (getattr(args, "image", None) or default_input()).resolve()
    key_file = Path(os.environ.get("DASHSCOPE_API_KEY_FILE", PROJECT_ROOT / "apikey.txt"))

    if args.command == "pose":
        from pet_pose_generator import generate_pose_sheet

        result = generate_pose_sheet(
            image_path=image,
            key_file=key_file,
            output_dir=(args.output_dir or root / "outputs" / "poses").resolve(),
            output_count=args.count,
        )
        print(result.metadata_path)
    elif args.command == "three-view":
        from pet_three_view_generator import generate_three_view_sheet

        result = generate_three_view_sheet(
            image_path=image,
            key_file=key_file,
            output_dir=(args.output_dir or root / "outputs" / "three-view").resolve(),
            reference_dir=args.reference_dir.resolve() if args.reference_dir else None,
        )
        print(result.metadata_path)
    elif args.command == "eat-video":
        from pet_video_i2v import generate_eating_video

        result = generate_eating_video(
            pose_image_path=args.pose_image.resolve(),
            key_file=key_file,
            output_dir=(args.output_dir or root / "outputs" / "eat-video").resolve(),
            image_model=args.image_model,
            video_model=args.video_model,
            resolution=args.resolution,
            duration=args.duration,
            seed=args.seed,
            endpoint="https://dashscope.aliyuncs.com/api/v1",
            poll_interval=args.poll_interval,
            timeout_seconds=args.timeout_seconds,
        )
        print(result.metadata_path)
    else:
        from pet_two_action_pipeline import generate_two_action_videos

        result = generate_two_action_videos(
            image_path=image,
            key_file=key_file,
            output_dir=(args.output_dir or root / "outputs" / "two-actions").resolve(),
        )
        print(result.metadata_path)


if __name__ == "__main__":
    main()
