from pathlib import Path
import os
import sys

import numpy as np
import torch
from PIL import Image

for parent in Path(__file__).resolve().parents:
    if (parent / "pet_pose_generator.py").exists():
        sys.path.insert(0, str(parent))
        break

from pet_pose_generator import DEFAULT_NEGATIVE_PROMPT, DEFAULT_PROMPT, generate_pose_sheet


DATA_ROOT = os.environ.get("PET_PIPELINE_DATA_ROOT", "D:/gogelila/comfy")
PROJECT_ROOT = os.environ.get("PET_PIPELINE_PROJECT_ROOT", "D:/gogelila/comfy")


class DashScopeCatPoseSheet:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_path": ("STRING", {"default": f"{DATA_ROOT}/inputs/1.jpg"}),
                "api_key_file": ("STRING", {"default": f"{PROJECT_ROOT}/apikey.txt"}),
                "output_dir": ("STRING", {"default": f"{DATA_ROOT}/outputs/cat_poses"}),
                "model": ("STRING", {"default": "qwen-image-edit-plus"}),
                "size": ("STRING", {"default": "1024*1024"}),
                "seed": ("INT", {"default": 20260617, "min": 0, "max": 2147483647}),
            },
            "optional": {
                "prompt": ("STRING", {"default": DEFAULT_PROMPT, "multiline": True}),
                "negative_prompt": ("STRING", {"default": DEFAULT_NEGATIVE_PROMPT, "multiline": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("pose_sheet_preview", "alpha_mask", "transparent_sheet_path")
    FUNCTION = "generate"
    CATEGORY = "DashScope/PetVideo"

    def generate(
        self,
        image_path,
        api_key_file,
        output_dir,
        model,
        size,
        seed,
        prompt=DEFAULT_PROMPT,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
    ):
        result = generate_pose_sheet(
            image_path=Path(image_path),
            key_file=Path(api_key_file),
            output_dir=Path(output_dir),
            output_count=6,
            size=size,
            model=model,
            seed=seed,
            prompt=prompt,
            negative_prompt=negative_prompt,
        )
        image = Image.open(result.sheet_path).convert("RGBA")
        array = np.asarray(image).astype(np.float32) / 255.0
        rgb = torch.from_numpy(array[..., :3]).unsqueeze(0)
        mask = torch.from_numpy(1.0 - array[..., 3]).unsqueeze(0)
        return (rgb, mask, str(result.sheet_path))


NODE_CLASS_MAPPINGS = {
    "DashScopeCatPoseSheet": DashScopeCatPoseSheet,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DashScopeCatPoseSheet": "DashScope Cat 6 Pose Sheet",
}
