from __future__ import annotations

import json
from pathlib import Path

from pet_pose_generator import read_api_key
from pet_video_i2v import (
    EATING_VIDEO_NEGATIVE_PROMPT,
    EATING_VIDEO_PROMPT,
    generate_running_video,
    postprocess_video,
    submit_and_wait_video,
)


def main() -> None:
    root = Path("outputs/cat_two_action_pipeline_wan22_side_eat").resolve()
    key_file = Path("apikey.txt").resolve()
    api_key = read_api_key(key_file)
    action_pose = root / "poses" / "transparent" / "pose_02_transparent.png"
    eat_keyframe = root / "videos" / "eat" / "keyframes" / "eat_keyframe_000.png"
    image_model = "qwen-image-edit"
    video_model = "wan2.2-i2v-flash"
    resolution = "720P"
    duration = 5
    seed = 20260617
    endpoint = "https://dashscope.aliyuncs.com/api/v1"

    eat_output = root / "videos" / "eat"
    raw_eat, eat_response = submit_and_wait_video(
        keyframe_path=eat_keyframe,
        api_key=api_key,
        output_dir=eat_output,
        model=video_model,
        resolution=resolution,
        duration=duration,
        seed=seed + 200,
        endpoint=endpoint,
        poll_interval=15,
        timeout_seconds=1200,
        prompt=EATING_VIDEO_PROMPT,
        negative_prompt=EATING_VIDEO_NEGATIVE_PROMPT,
        action_name="eat",
    )
    final_eat = postprocess_video(raw_eat, eat_output / "final" / "eat_5s_30fps_16x9.mp4")

    eat_metadata = eat_output / "metadata" / "eat_video_metadata.json"
    eat_metadata.parent.mkdir(parents=True, exist_ok=True)
    eat_metadata.write_text(
        json.dumps(
            {
                "pose_image": str(action_pose),
                "keyframe": str(eat_keyframe),
                "raw_video": str(raw_eat),
                "final_video": str(final_eat),
                "image_model": image_model,
                "video_model": video_model,
                "resolution": resolution,
                "duration": duration,
                "seed": seed,
                "endpoint": endpoint,
                "video_response": eat_response,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    run_result = generate_running_video(
        pose_image_path=action_pose,
        key_file=key_file,
        output_dir=root / "videos" / "run",
        image_model=image_model,
        video_model=video_model,
        resolution=resolution,
        duration=duration,
        seed=seed + 300,
        endpoint=endpoint,
        poll_interval=15,
        timeout_seconds=1200,
    )

    pose_metadata = root / "poses" / "generation_metadata.json"
    transparent_poses = sorted((root / "poses" / "transparent").glob("pose_*_transparent.png"))
    metadata = {
        "input_image": str(Path("1.jpg").resolve()),
        "output_dir": str(root),
        "image_model": image_model,
        "video_model": video_model,
        "pose_size": "1024*1024",
        "resolution": resolution,
        "duration": duration,
        "seed": seed,
        "endpoint": endpoint,
        "action_pose_index": 2,
        "pose_sheet": str(root / "poses" / "cat_6_poses_transparent_sheet.png"),
        "pose_metadata": str(pose_metadata),
        "transparent_poses": [str(path) for path in transparent_poses],
        "action_pose": str(action_pose),
        "eat": {
            "keyframe": str(eat_keyframe),
            "raw_video": str(raw_eat),
            "final_video": str(final_eat),
            "metadata": str(eat_metadata),
        },
        "run": {
            "keyframe": str(run_result.keyframe_path),
            "raw_video": str(run_result.raw_video_path),
            "final_video": str(run_result.final_video_path),
            "metadata": str(run_result.metadata_path),
        },
    }
    metadata_path = root / "pipeline_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"pose_sheet={metadata['pose_sheet']}")
    print(f"eat_video={final_eat}")
    print(f"run_video={run_result.final_video_path}")
    print(f"metadata={metadata_path}")


if __name__ == "__main__":
    main()
