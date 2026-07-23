from pathlib import Path

from pet_pose_generator import generate_pose_sheet
from pet_video_i2v import generate_eating_video


def main() -> None:
    root = Path("outputs/open_api_probe")
    pose = generate_pose_sheet(
        image_path=Path("1.jpg").resolve(),
        key_file=Path("apikey.txt").resolve(),
        output_dir=(root / "image_qwen_image_edit").resolve(),
        output_count=1,
        model="qwen-image-edit",
        seed=20260618,
    )
    video = generate_eating_video(
        pose_image_path=pose.transparent_paths[0],
        key_file=Path("apikey.txt").resolve(),
        output_dir=(root / "video_wanx21_i2v_turbo").resolve(),
        image_model="qwen-image-edit",
        video_model="wanx2.1-i2v-turbo",
        resolution="480P",
        duration=5,
        seed=20260618,
        endpoint="https://dashscope.aliyuncs.com/api/v1",
        poll_interval=20,
        timeout_seconds=1200,
    )
    print(f"image={pose.transparent_paths[0]}")
    print(f"image_sheet={pose.sheet_path}")
    print(f"video={video.final_video_path}")
    print(f"metadata={video.metadata_path}")


if __name__ == "__main__":
    main()
