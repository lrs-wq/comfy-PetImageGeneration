from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from pet_pose_generator import PoseGenerationResult, generate_pose_sheet
from pet_video_i2v import (
    EatingVideoResult,
    RunningVideoResult,
    generate_eating_video,
    generate_running_video,
)


@dataclass(frozen=True)
class TwoActionPipelineResult:
    pose_sheet_path: Path
    pose_metadata_path: Path
    transparent_pose_paths: list[Path]
    action_pose_path: Path
    eating_result: EatingVideoResult
    running_result: RunningVideoResult
    metadata_path: Path


def generate_two_action_videos(
    *,
    image_path: Path,
    key_file: Path,
    output_dir: Path,
    image_model: str = "qwen-image-edit",
    video_model: str = "wan2.2-i2v-flash",
    pose_size: str = "1024*1024",
    resolution: str = "720P",
    duration: int = 5,
    seed: int = 20260617,
    action_pose_index: int = 2,
    endpoint: str = "https://dashscope.aliyuncs.com/api/v1",
    poll_interval: int = 15,
    timeout_seconds: int = 900,
) -> TwoActionPipelineResult:
    image_path = image_path.resolve()
    key_file = key_file.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pose_result: PoseGenerationResult = generate_pose_sheet(
        image_path=image_path,
        key_file=key_file,
        output_dir=output_dir / "poses",
        output_count=6,
        size=pose_size,
        model=image_model,
        seed=seed,
        endpoint=endpoint,
    )
    if not 1 <= action_pose_index <= len(pose_result.transparent_paths):
        raise ValueError(
            f"action_pose_index must be between 1 and {len(pose_result.transparent_paths)}, "
            f"got {action_pose_index}."
        )
    action_pose_path = pose_result.transparent_paths[action_pose_index - 1]

    eating_result = generate_eating_video(
        pose_image_path=action_pose_path,
        key_file=key_file,
        output_dir=output_dir / "videos" / "eat",
        image_model=image_model,
        video_model=video_model,
        resolution=resolution,
        duration=duration,
        seed=seed + 100,
        endpoint=endpoint,
        poll_interval=poll_interval,
        timeout_seconds=timeout_seconds,
    )
    running_result = generate_running_video(
        pose_image_path=action_pose_path,
        key_file=key_file,
        output_dir=output_dir / "videos" / "run",
        image_model=image_model,
        video_model=video_model,
        resolution=resolution,
        duration=duration,
        seed=seed + 200,
        endpoint=endpoint,
        poll_interval=poll_interval,
        timeout_seconds=timeout_seconds,
    )

    metadata_path = output_dir / "pipeline_metadata.json"
    metadata = {
        "input_image": str(image_path),
        "output_dir": str(output_dir),
        "image_model": image_model,
        "video_model": video_model,
        "pose_size": pose_size,
        "resolution": resolution,
        "duration": duration,
        "seed": seed,
        "endpoint": endpoint,
        "action_pose_index": action_pose_index,
        "pose_sheet": str(pose_result.sheet_path),
        "pose_metadata": str(pose_result.metadata_path),
        "transparent_poses": [str(path) for path in pose_result.transparent_paths],
        "action_pose": str(action_pose_path),
        "eat": {
            "keyframe": str(eating_result.keyframe_path),
            "raw_video": str(eating_result.raw_video_path),
            "final_video": str(eating_result.final_video_path),
            "metadata": str(eating_result.metadata_path),
        },
        "run": {
            "keyframe": str(running_result.keyframe_path),
            "raw_video": str(running_result.raw_video_path),
            "final_video": str(running_result.final_video_path),
            "metadata": str(running_result.metadata_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    return TwoActionPipelineResult(
        pose_sheet_path=pose_result.sheet_path,
        pose_metadata_path=pose_result.metadata_path,
        transparent_pose_paths=pose_result.transparent_paths,
        action_pose_path=action_pose_path,
        eating_result=eating_result,
        running_result=running_result,
        metadata_path=metadata_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate six cat pose cutouts and two action videos, eating and running, from one image."
    )
    parser.add_argument("--image", type=Path, default=Path("1.jpg"))
    parser.add_argument("--key-file", type=Path, default=Path("apikey.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cat_two_action_pipeline_wan22_side_eat"))
    parser.add_argument("--image-model", default="qwen-image-edit")
    parser.add_argument("--video-model", default="wan2.2-i2v-flash")
    parser.add_argument("--pose-size", default="1024*1024")
    parser.add_argument("--resolution", default="720P")
    parser.add_argument("--duration", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--action-pose-index", type=int, default=2)
    parser.add_argument("--endpoint", default="https://dashscope.aliyuncs.com/api/v1")
    parser.add_argument("--poll-interval", type=int, default=15)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = generate_two_action_videos(
        image_path=args.image,
        key_file=args.key_file,
        output_dir=args.output_dir,
        image_model=args.image_model,
        video_model=args.video_model,
        pose_size=args.pose_size,
        resolution=args.resolution,
        duration=args.duration,
        seed=args.seed,
        action_pose_index=args.action_pose_index,
        endpoint=args.endpoint,
        poll_interval=args.poll_interval,
        timeout_seconds=args.timeout_seconds,
    )
    print(f"pose_sheet={result.pose_sheet_path}")
    print(f"action_pose={result.action_pose_path}")
    print(f"eat_video={result.eating_result.final_video_path}")
    print(f"run_video={result.running_result.final_video_path}")
    print(f"metadata={result.metadata_path}")


if __name__ == "__main__":
    main()
