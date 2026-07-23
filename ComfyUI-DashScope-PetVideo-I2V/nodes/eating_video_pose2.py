from pathlib import Path
import os
import sys

for parent in Path(__file__).resolve().parents:
    if (parent / "pet_video_i2v.py").exists():
        sys.path.insert(0, str(parent))
        break

from pet_video_i2v import generate_eating_video


DATA_ROOT = os.environ.get("PET_PIPELINE_DATA_ROOT", "D:/gogelila/comfy")
PROJECT_ROOT = os.environ.get("PET_PIPELINE_PROJECT_ROOT", "D:/gogelila/comfy")


class DashScopePose2EatingVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_image_path": ("STRING", {"default": f"{DATA_ROOT}/outputs/cat_poses/transparent/pose_02_transparent.png"}),
                "api_key_file": ("STRING", {"default": f"{PROJECT_ROOT}/apikey.txt"}),
                "output_dir": ("STRING", {"default": f"{DATA_ROOT}/outputs/eat_video_pose2"}),
                "image_model": ("STRING", {"default": "qwen-image-edit-plus"}),
                "video_model": ("STRING", {"default": "wan2.7-i2v"}),
                "resolution": (["720P", "1080P"], {"default": "720P"}),
                "duration": ("INT", {"default": 5, "min": 2, "max": 15}),
                "seed": ("INT", {"default": 20260622, "min": 0, "max": 2147483647}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("keyframe_path", "raw_video_path", "final_video_path", "metadata_path")
    FUNCTION = "generate"
    CATEGORY = "DashScope/PetVideoI2V"

    def generate(self, pose_image_path, api_key_file, output_dir, image_model, video_model, resolution, duration, seed):
        result = generate_eating_video(
            pose_image_path=Path(pose_image_path),
            key_file=Path(api_key_file),
            output_dir=Path(output_dir),
            image_model=image_model,
            video_model=video_model,
            resolution=resolution,
            duration=duration,
            seed=seed,
            endpoint="https://dashscope.aliyuncs.com/api/v1",
            poll_interval=15,
            timeout_seconds=900,
        )
        return (
            str(result.keyframe_path),
            str(result.raw_video_path),
            str(result.final_video_path),
            str(result.metadata_path),
        )


NODE_CLASS_MAPPINGS = {
    "DashScopePose2EatingVideo": DashScopePose2EatingVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DashScopePose2EatingVideo": "DashScope Pose2 Eating Video",
}
