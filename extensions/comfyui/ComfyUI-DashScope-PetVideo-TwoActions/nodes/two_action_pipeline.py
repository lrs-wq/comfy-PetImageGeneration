from pathlib import Path
import os
import sys

import numpy as np
from PIL import Image

for parent in Path(__file__).resolve().parents:
    if (parent / "src" / "comfy_pet_pipeline").is_dir():
        sys.path.insert(0, str(parent))
        break

from src.comfy_pet_pipeline.legacy.two_action_pipeline import (
    generate_two_action_videos,
)


DATA_ROOT = os.environ.get("PET_PIPELINE_DATA_ROOT", "D:/gogelila/comfy")
PROJECT_ROOT = os.environ.get("PET_PIPELINE_PROJECT_ROOT", "D:/gogelila/comfy")


def save_uploaded_image(image, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    array = image[0].detach().cpu().numpy() if hasattr(image[0], "detach") else np.asarray(image[0])
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    path = output_dir / "uploaded_reference.png"
    Image.fromarray(array).save(path)
    return path


class DashScopePetTwoActionVideos:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_path": ("STRING", {"default": f"{DATA_ROOT}/inputs/1.jpg"}),
                "api_key_file": ("STRING", {"default": f"{PROJECT_ROOT}/apikey.txt"}),
                "output_dir": ("STRING", {"default": f"{DATA_ROOT}/outputs/cat_two_action_pipeline"}),
                "image_model": ("STRING", {"default": "qwen-image-edit"}),
                "video_model": ("STRING", {"default": "wan2.2-i2v-flash"}),
                "pose_size": ("STRING", {"default": "1024*1024"}),
                "resolution": (["480P", "720P", "1080P"], {"default": "720P"}),
                "duration": ("INT", {"default": 5, "min": 2, "max": 15}),
                "seed": ("INT", {"default": 20260617, "min": 0, "max": 2147483647}),
                "action_pose_index": ("INT", {"default": 2, "min": 1, "max": 6}),
            },
            "optional": {
                "reference_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "pose_sheet_path",
        "pose_metadata_path",
        "action_pose_path",
        "eat_video_path",
        "run_video_path",
        "pipeline_metadata_path",
    )
    FUNCTION = "generate"
    CATEGORY = "DashScope/PetVideo"

    def generate(
        self,
        image_path,
        api_key_file,
        output_dir,
        image_model,
        video_model,
        pose_size,
        resolution,
        duration,
        seed,
        action_pose_index,
        reference_image=None,
    ):
        output_path = Path(output_dir)
        if reference_image is not None:
            source_image = save_uploaded_image(reference_image, output_path / "input")
        else:
            source_image = Path(image_path)

        result = generate_two_action_videos(
            image_path=source_image,
            key_file=Path(api_key_file),
            output_dir=output_path,
            image_model=image_model,
            video_model=video_model,
            pose_size=pose_size,
            resolution=resolution,
            duration=duration,
            seed=seed,
            action_pose_index=action_pose_index,
            endpoint="https://dashscope.aliyuncs.com/api/v1",
            poll_interval=15,
            timeout_seconds=900,
        )
        return (
            str(result.pose_sheet_path),
            str(result.pose_metadata_path),
            str(result.action_pose_path),
            str(result.eating_result.final_video_path),
            str(result.running_result.final_video_path),
            str(result.metadata_path),
        )


NODE_CLASS_MAPPINGS = {
    "DashScopePetTwoActionVideos": DashScopePetTwoActionVideos,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DashScopePetTwoActionVideos": "DashScope Cat Two Action Videos",
}
