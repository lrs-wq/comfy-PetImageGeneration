from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path("/root/comfy-pet-pipeline")
sys.path.insert(0, str(PROJECT_ROOT))

import pet_pose_generator  # noqa: E402
import pet_two_action_pipeline  # noqa: E402
import pet_video_i2v  # noqa: E402


assert Path(os.environ["PET_PIPELINE_DATA_ROOT"]) == Path("/data/comfy-pet-pipeline")
assert pet_pose_generator.read_api_key()
print("pipeline_imports=ok")
