from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import queue
import re
import secrets
import shutil
import subprocess
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import requests
import cv2
import numpy as np
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse
from PIL import Image, ImageChops, UnidentifiedImageError

from transparent_animation import build_transparent_animations, build_transparent_cutout


LOGGER = logging.getLogger("local_video_api")
logging.basicConfig(
    level=os.environ.get("LOCAL_VIDEO_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

API_PREFIX = "/api/local-video"
API_VERSION = "2.18.4"
DATA_ROOT = Path(
    os.environ.get("PET_PIPELINE_DATA_ROOT", "/data/comfy-pet-pipeline")
).resolve()
PROJECT_ROOT = Path(__file__).resolve().parent
COMFY_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188").rstrip("/")
COMFY_INPUT = DATA_ROOT / "comfyui" / "input"
COMFY_OUTPUT = DATA_ROOT / "comfyui" / "output"
MODEL_ROOT = DATA_ROOT / "models" / "comfyui"
OUTPUT_ROOT = DATA_ROOT / "outputs" / "local_video_api"
USER_LORA_DIR = MODEL_ROOT / "loras" / "user_uploads"

API_KEY = os.environ.get("LOCAL_VIDEO_API_KEY", "").strip()
PROMPT_TRANSLATION_ENABLED = os.environ.get(
    "PROMPT_TRANSLATION_ENABLED", "1"
).strip().lower() not in {"0", "false", "no", "off"}
PROMPT_TRANSLATOR_PYTHON = Path(
    os.environ.get(
        "PROMPT_TRANSLATOR_PYTHON",
        "/opt/miniconda3/envs/comfyui/bin/python",
    )
).resolve()
PROMPT_TRANSLATOR_MODEL = Path(
    os.environ.get(
        "PROMPT_TRANSLATOR_MODEL",
        str(DATA_ROOT / "models" / "translation" / "opus-mt-zh-en"),
    )
).resolve()
PROMPT_TRANSLATOR_SCRIPT = PROJECT_ROOT / "prompt_translator.py"
PROMPT_TRANSLATION_TIMEOUT = max(
    10,
    min(300, int(os.environ.get("PROMPT_TRANSLATION_TIMEOUT", "120"))),
)
PROMPT_TRANSLATION_THREADS = max(
    1,
    min(16, int(os.environ.get("PROMPT_TRANSLATION_THREADS", "4"))),
)
MAX_UPLOAD_BYTES = int(
    os.environ.get("LOCAL_VIDEO_MAX_UPLOAD_BYTES", str(15 * 1024 * 1024))
)
QWEN_POSE_MAX_ATTEMPTS = max(
    1, min(3, int(os.environ.get("QWEN_POSE_MAX_ATTEMPTS", "3")))
)
QWEN_RUNNING_POSE_EXTRA_ATTEMPTS = max(
    0,
    min(2, int(os.environ.get("QWEN_RUNNING_POSE_EXTRA_ATTEMPTS", "1"))),
)
QWEN_BOWL_POSE_EXTRA_ATTEMPTS = max(
    0,
    min(2, int(os.environ.get("QWEN_BOWL_POSE_EXTRA_ATTEMPTS", "1"))),
)
QWEN_CHROMA_MAX_ATTEMPTS = max(
    1,
    min(3, int(os.environ.get("QWEN_CHROMA_MAX_ATTEMPTS", "3"))),
)
QWEN_RECOVERY_USER_LORA_STRENGTH = float(
    os.environ.get("QWEN_RECOVERY_USER_LORA_STRENGTH", "0.25")
)
QWEN_CANONICAL_USER_LORA_TOTAL_CAP = float(
    os.environ.get("QWEN_CANONICAL_USER_LORA_TOTAL_CAP", "1.0")
)
QWEN_CANONICAL_USER_LORA_PER_LORA_CAP = float(
    os.environ.get("QWEN_CANONICAL_USER_LORA_PER_LORA_CAP", "1.0")
)
QWEN_POSE_USER_LORA_TOTAL_CAP = float(
    os.environ.get("QWEN_POSE_USER_LORA_TOTAL_CAP", "0.40")
)
QWEN_POSE_USER_LORA_RETRY_TOTAL_CAP = float(
    os.environ.get("QWEN_POSE_USER_LORA_RETRY_TOTAL_CAP", "0.25")
)
QWEN_POSE_USER_LORA_FINAL_TOTAL_CAP = float(
    os.environ.get("QWEN_POSE_USER_LORA_FINAL_TOTAL_CAP", "0.15")
)
QWEN_POSE_USER_LORA_PER_LORA_CAP = float(
    os.environ.get("QWEN_POSE_USER_LORA_PER_LORA_CAP", "0.35")
)
BOWL_IDENTITY_PALETTE_BOTTOM_EXCLUSION = max(
    0.0,
    min(
        0.45,
        float(os.environ.get("BOWL_IDENTITY_PALETTE_BOTTOM_EXCLUSION", "0.22")),
    ),
)
BOWL_IDENTITY_PALETTE_UPPER_BODY_EXCLUSION = max(
    BOWL_IDENTITY_PALETTE_BOTTOM_EXCLUSION,
    min(
        0.45,
        float(
            os.environ.get(
                "BOWL_IDENTITY_PALETTE_UPPER_BODY_EXCLUSION",
                "0.45",
            )
        ),
    ),
)
QWEN_POSE_SIMILARITY_IOU_THRESHOLD = float(
    os.environ.get("QWEN_POSE_SIMILARITY_IOU_THRESHOLD", "0.94")
)
QWEN_POSE_SIMILARITY_DHASH_THRESHOLD = int(
    os.environ.get("QWEN_POSE_SIMILARITY_DHASH_THRESHOLD", "6")
)
QWEN_POSE_RELATED_IOU_THRESHOLD = float(
    os.environ.get("QWEN_POSE_RELATED_IOU_THRESHOLD", "0.975")
)
QWEN_POSE_RELATED_DHASH_THRESHOLD = int(
    os.environ.get("QWEN_POSE_RELATED_DHASH_THRESHOLD", "3")
)
QWEN_POSE_HARD_DUPLICATE_IOU = float(
    os.environ.get("QWEN_POSE_HARD_DUPLICATE_IOU", "0.985")
)
QWEN_POSE_HARD_DUPLICATE_DHASH = int(
    os.environ.get("QWEN_POSE_HARD_DUPLICATE_DHASH", "2")
)
CHROMA_FALLBACK_REMBG_MODEL = (
    os.environ.get("CHROMA_FALLBACK_REMBG_MODEL", "u2net").strip() or "u2net"
)
MAX_LORA_UPLOAD_BYTES = int(
    os.environ.get("LOCAL_VIDEO_MAX_LORA_UPLOAD_BYTES", str(2 * 1024 * 1024 * 1024))
)
MAX_USER_LORAS = max(1, min(8, int(os.environ.get("LOCAL_VIDEO_MAX_USER_LORAS", "4"))))
MAX_LORA_FUSION_BYTES = int(
    os.environ.get("LOCAL_VIDEO_MAX_LORA_FUSION_BYTES", str(4 * 1024 * 1024 * 1024))
)
MAX_QUEUE_SIZE = int(os.environ.get("LOCAL_VIDEO_MAX_QUEUE_SIZE", "8"))
MAX_POSES = int(os.environ.get("LOCAL_VIDEO_MAX_POSES", "6"))
TRANSPARENT_GIF_SIZE = 512
TRANSPARENT_GIF_FPS = 20
TRANSPARENT_GIF_FRAMES = 100
TRANSPARENT_GIF_DURATION_SECONDS = 5.0
FFMPEG_BIN = os.environ.get(
    "FFMPEG_BIN", shutil.which("ffmpeg") or "/usr/local/bin/ffmpeg"
)

QWEN_MODEL = "qwen_image_edit_2511_fp8mixed.safetensors"
QWEN_CLIP = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
QWEN_VAE = "qwen_image_vae.safetensors"
QWEN_LORA = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
QWEN_STYLE_LORA = os.environ.get("QWEN_STYLE_LORA", "").strip()
QWEN_STYLE_LORA_STRENGTH = float(os.environ.get("QWEN_STYLE_LORA_STRENGTH", "0.65"))
QWEN_POSE_STYLE_LORA_STRENGTH = float(
    os.environ.get("QWEN_POSE_STYLE_LORA_STRENGTH", "0.25")
)
QWEN_STYLE_PROFILE = (
    os.environ.get("QWEN_STYLE_PROFILE", "baseline").strip() or "baseline"
)
WAN_ENGINE = os.environ.get("WAN_ENGINE", "wan2.2-ti2v-5b").strip().lower()
if WAN_ENGINE not in {"wan2.2-ti2v-5b", "wan2.2-i2v-a14b-fp8"}:
    raise ValueError("WAN_ENGINE must be wan2.2-ti2v-5b or wan2.2-i2v-a14b-fp8")
WAN_MODEL = os.environ.get("WAN_MODEL", "wan2.2_ti2v_5B_fp16.safetensors").strip()
WAN_HIGH_NOISE_MODEL = os.environ.get(
    "WAN_HIGH_NOISE_MODEL", "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
).strip()
WAN_LOW_NOISE_MODEL = os.environ.get(
    "WAN_LOW_NOISE_MODEL", "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
).strip()
WAN_CLIP = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
WAN_VAE = os.environ.get(
    "WAN_VAE",
    "wan_2.1_vae.safetensors"
    if WAN_ENGINE == "wan2.2-i2v-a14b-fp8"
    else "wan2.2_vae.safetensors",
).strip()
FRAME_COUNT = int(os.environ.get("WAN_FRAME_COUNT", "121"))
FPS = float(os.environ.get("WAN_FPS", "24"))
WAN_STEPS = int(os.environ.get("WAN_STEPS", "20"))
WAN_SPLIT_STEP = int(os.environ.get("WAN_SPLIT_STEP", str(WAN_STEPS // 2)))
WAN_CFG = float(
    os.environ.get("WAN_CFG", "3.5" if WAN_ENGINE == "wan2.2-i2v-a14b-fp8" else "4.0")
)
WAN_SHIFT = float(
    os.environ.get("WAN_SHIFT", "5.0" if WAN_ENGINE == "wan2.2-i2v-a14b-fp8" else "8.0")
)

if FRAME_COUNT < 1 or FRAME_COUNT % 4 != 1:
    raise ValueError("WAN_FRAME_COUNT must be positive and equal to 4n+1")
if WAN_STEPS < 2 or not 1 <= WAN_SPLIT_STEP < WAN_STEPS:
    raise ValueError("WAN_SPLIT_STEP must be between 1 and WAN_STEPS - 1")

FUN_CONTROL_ENABLED = os.environ.get(
    "FUN_CONTROL_ENABLED", "false"
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
FUN_CONTROL_FALLBACK_TO_WAN = os.environ.get(
    "FUN_CONTROL_FALLBACK_TO_WAN", "true"
).strip().lower() in {"1", "true", "yes", "on"}
FUN_CONTROL_PYTHON = (
    Path(
        os.environ.get(
            "FUN_CONTROL_PYTHON",
            "/data/comfy-pet-pipeline/experiments/videoxfun-venv/bin/python",
        )
    )
    .expanduser()
    .absolute()
)
FUN_CONTROL_SCRIPT = Path(
    os.environ.get(
        "FUN_CONTROL_SCRIPT",
        "/data/comfy-pet-pipeline/experiments/wan22_fun_5b_control_test/wan22_fun_5b_control_ref_test.py",
    )
).resolve()
FUN_CONTROL_PROJECT_ROOT = Path(
    os.environ.get(
        "FUN_CONTROL_PROJECT_ROOT",
        "/data/comfy-pet-pipeline/experiments/VideoX-Fun",
    )
).resolve()
FUN_CONTROL_MODEL_ROOT = Path(
    os.environ.get(
        "FUN_CONTROL_MODEL_ROOT",
        "/data/comfy-pet-pipeline/models/videoxfun/Wan2.2-Fun-5B-Control",
    )
).resolve()
FUN_CONTROL_RUNNING_VIDEO = Path(
    os.environ.get(
        "FUN_CONTROL_RUNNING_VIDEO",
        "/data/comfy-pet-pipeline/experiments/wan22_fun_5b_control_test/muybridge_cat_canny_512_121f_24fps.mp4",
    )
).resolve()
FUN_CONTROL_STEPS = int(os.environ.get("FUN_CONTROL_STEPS", "30"))
FUN_CONTROL_TIMEOUT = int(os.environ.get("FUN_CONTROL_TIMEOUT", "7200"))
FUN_CONTROL_GPU_MEMORY_MODE = os.environ.get(
    "FUN_CONTROL_GPU_MEMORY_MODE", "model_cpu_offload"
).strip()

if not 0.0 <= QWEN_STYLE_LORA_STRENGTH <= 2.0:
    raise ValueError("QWEN_STYLE_LORA_STRENGTH must be between 0.0 and 2.0")
if not 0.0 <= QWEN_POSE_STYLE_LORA_STRENGTH <= 2.0:
    raise ValueError("QWEN_POSE_STYLE_LORA_STRENGTH must be between 0.0 and 2.0")
if not 0.0 <= QWEN_RECOVERY_USER_LORA_STRENGTH <= 2.0:
    raise ValueError("QWEN_RECOVERY_USER_LORA_STRENGTH must be between 0.0 and 2.0")
for name, value in {
    "QWEN_CANONICAL_USER_LORA_TOTAL_CAP": QWEN_CANONICAL_USER_LORA_TOTAL_CAP,
    "QWEN_CANONICAL_USER_LORA_PER_LORA_CAP": QWEN_CANONICAL_USER_LORA_PER_LORA_CAP,
    "QWEN_POSE_USER_LORA_TOTAL_CAP": QWEN_POSE_USER_LORA_TOTAL_CAP,
    "QWEN_POSE_USER_LORA_RETRY_TOTAL_CAP": QWEN_POSE_USER_LORA_RETRY_TOTAL_CAP,
    "QWEN_POSE_USER_LORA_FINAL_TOTAL_CAP": QWEN_POSE_USER_LORA_FINAL_TOTAL_CAP,
    "QWEN_POSE_USER_LORA_PER_LORA_CAP": QWEN_POSE_USER_LORA_PER_LORA_CAP,
}.items():
    if not 0.0 <= value <= 2.0:
        raise ValueError(f"{name} must be between 0.0 and 2.0")
for name, value in {
    "QWEN_POSE_SIMILARITY_IOU_THRESHOLD": QWEN_POSE_SIMILARITY_IOU_THRESHOLD,
    "QWEN_POSE_RELATED_IOU_THRESHOLD": QWEN_POSE_RELATED_IOU_THRESHOLD,
    "QWEN_POSE_HARD_DUPLICATE_IOU": QWEN_POSE_HARD_DUPLICATE_IOU,
}.items():
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0")
for name, value in {
    "QWEN_POSE_SIMILARITY_DHASH_THRESHOLD": QWEN_POSE_SIMILARITY_DHASH_THRESHOLD,
    "QWEN_POSE_RELATED_DHASH_THRESHOLD": QWEN_POSE_RELATED_DHASH_THRESHOLD,
    "QWEN_POSE_HARD_DUPLICATE_DHASH": QWEN_POSE_HARD_DUPLICATE_DHASH,
}.items():
    if not 0 <= value <= 64:
        raise ValueError(f"{name} must be between 0 and 64")

LEGACY_EATING_IMAGE_PROFILE_ID = "20260717T094459Z-f738613526"
MULTIPOSE_V1_IMAGE_PROFILE_ID = "pet-multipose-20260717-v1"
BASELINE_IMAGE_PROFILE_ID = "pet-multipose-20260717-v2"
BASELINE_IMAGE_SOURCE_JOB_ID = LEGACY_EATING_IMAGE_PROFILE_ID
BASELINE_IMAGE_PROFILE_REVISION = "multipose-v2"
BASELINE_IMAGE_SOURCE_PROMPT_ZH = (
    "俯身低头，嘴巴凑近浅米色陶瓷猫粮碗，碗内装有棕色颗粒猫粮，正在进食的自然动态"
)
BASELINE_IMAGE_SOURCE_PROMPT_EN = (
    "A natural eating action: the cat leans forward with its head lowered, "
    "bringing its mouth close to a shallow light-beige ceramic food bowl "
    "filled with brown kibble."
)
BASELINE_IMAGE_PROMPT_ZH = (
    "以参考图片中的猫咪为唯一身份标准，完整保留毛色和花纹位置、脸型、"
    "眼睛颜色与间距、耳形、鼻色、毛长、体型比例和尾巴特征；生成高品质"
    "萌系3D卡通角色，毛发细腻柔软，光照明亮柔和，保持自然无衣服无配饰"
    "的完整全身形象。不继承或固定参考图中的姿态、肢体坐标和构图，具体"
    "动作只由所选姿态决定。"
)
BASELINE_IMAGE_PROMPT_EN = (
    "Use the cat in the reference image as the only identity standard. Preserve "
    "its exact fur colors and marking placement, face shape, eye color and spacing, "
    "ear shape, nose color, fur length, body proportions, and tail features. Render "
    "it as a high-quality cute stylized 3D character with finely groomed soft fur "
    "and bright gentle lighting, showing its complete natural unclothed body with "
    "no accessories. Do not inherit or lock the reference pose, limb coordinates, "
    "or composition; the selected pose alone controls the action."
)
MULTIPOSE_PRESETS: tuple[dict[str, str], ...] = (
    {
        "id": "running",
        "action_key": "running",
        "label": "奔跑",
        "prompt_zh": (
            "三分之四侧面完整全身视角，呈稳定的起跑准备站姿；身体略微前倾，"
            "头部自然抬起，四条腿完整可见并前后错开，四只完整爪全部清晰压在"
            "同一地面基线上且没有悬空缝隙，远侧腿不能被身体遮住或与近侧腿"
            "重叠；尾巴从根部到末端完整可见。这是供后续动画使用的静态落地"
            "准备姿态，不是奔跑瞬间、跳跃或四脚腾空"
        ),
        "prompt_en": (
            "A complete full-body cat at a three-quarter side angle in a stable static "
            "pre-run standing contact pose. The torso leans slightly forward and the "
            "head is naturally raised. Exactly four complete legs are fully visible, "
            "anatomically attached, and staggered front-to-back so the far-side legs do "
            "not hide behind or merge with the near-side legs. All four complete paws "
            "are visibly planted and flattened on the same horizontal baseline with no "
            "gap below them. The tail is visible continuously from base to tip. This is "
            "a calm grounded ready stance for later animation, not an illustrated "
            "running instant, jump, leap, flight, or gallop suspension. No prop."
        ),
        "video_prompt_en": (
            "Animate a clear moderate-speed feline run-in-place cycle. Use a readable "
            "quadruped gait: the hind legs gather and push in sequence, the spine "
            "flexes and extends slightly, the forelegs reach and land in sequence, "
            "and each paw returns beneath the torso. Keep all four legs distinct and "
            "attached; a weight-bearing paw must not slide. Allow a small coordinated "
            "body rise and fall and a gentle tail counter-sway. Keep the global body "
            "center near the same position, with no forward travel or long airborne "
            "suspension, and finish near the starting contact phase for a clean loop."
        ),
    },
    {
        "id": "eating",
        "action_key": "eating",
        "label": "吃饭",
        "prompt_zh": BASELINE_IMAGE_SOURCE_PROMPT_ZH,
        "prompt_en": (
            "A complete full-body cat in a natural low quadruped feeding crouch. Both "
            "forelegs extend down from the shoulders and both complete front paws are "
            "visibly planted on the same baseline. Exactly one small shallow light-beige "
            "matte ceramic food bowl is centered directly below the lowered muzzle and "
            "slightly ahead of the planted paws. Keep a narrow visible green gap between "
            "the bowl and every paw, so it never touches, overlaps, hides, or replaces a "
            "limb. The nose is immediately above the bowl opening in a calm pre-bite "
            "feeding posture. The mouth is closed or hidden naturally behind the bowl "
            "rim. All brown kibble remains below the rim inside the bowl: no kibble at "
            "the mouth, above the rim, or in a paw. The cat never raises a foreleg and "
            "never grasps, lifts, carries, or holds food. The bowl is the only prop. Keep "
            "the eyes, face silhouette, all paws, body, and full tail readable."
        ),
        "video_prompt_en": (
            "Keep the food bowl completely fixed. Animate a restrained natural eating "
            "loop by lowering the muzzle a few millimeters into the bowl and raising it "
            "back, with tiny coordinated neck and lower-jaw movement plus "
            "subtle breathing through the chest and body. Move the head and face as "
            "one anatomical structure; do not stretch the mouth or reshape the cheeks, "
            "eyes, or nose. Keep both front paws planted and motionless beside the bowl; "
            "the cat must not lift a paw to the mouth or hold food. Kibble must not fly, "
            "appear, or disappear."
        ),
    },
    {
        "id": "drinking",
        "action_key": "drinking",
        "label": "喝水",
        "prompt_zh": (
            "俯身低头，嘴巴凑近浅米色陶瓷水碗，碗内装有平静清水，一小段短小"
            "粉色三角形舌尖从下唇伸出并与水面相交接触，舌尖长度不超过头部"
            "宽度的二十分之一，正在喝水的自然动态"
        ),
        "prompt_en": (
            "A complete full-body cat in a natural low crouch, leaning forward with "
            "the head lowered toward exactly one small shallow light-beige matte "
            "ceramic water bowl containing still clear water. The bowl sits immediately "
            "beside the front paws with a narrow visible green gap and never touches, "
            "overlaps, hides, or replaces a paw. Exactly one tiny solid-pink triangular "
            "tongue tip emerges from the lower lip and visibly intersects the water "
            "surface with no air gap; its visible length is no more than one twentieth "
            "of the head width. Keep the eyes, face silhouette, paws, body, and full "
            "tail readable; no long tongue, splash, drool, or water strand."
        ),
        "video_prompt_en": (
            "Keep the water bowl completely fixed. Animate restrained lapping: one "
            "short pink tongue tip extends and retracts naturally with extremely small "
            "coordinated muzzle and neck movement, subtle breathing, and soft localized "
            "ripples on the water. Preserve the face as one coherent anatomical "
            "structure. No splash, drool, transparent or elongated tongue, or "
            "continuous water strand."
        ),
    },
    {
        "id": "lying_down",
        "action_key": "lying down",
        "label": "趴卧",
        "prompt_zh": (
            "身体自然降低并完整趴卧，胸腹贴近地面，前爪平行向前伸出，后腿自然"
            "收拢，头部轻轻抬起，尾巴沿身体一侧完整弯曲，安静趴卧的自然状态"
        ),
        "prompt_en": (
            "A complete full-body cat in a relaxed low sphinx-like prone pose. The "
            "chest and abdomen are close to the invisible baseline, both front paws "
            "extend forward in parallel, the hind legs fold naturally beside the body, "
            "the head is gently raised, and the tail curves continuously along one side "
            "from base to tip. No prop and no cropped belly, haunch, paw, or tail."
        ),
        "video_prompt_en": (
            "Maintain the relaxed prone pose. Animate slow subtle breathing through "
            "the chest and abdomen, a small naturally coordinated head-and-ear response, "
            "and one gentle tail-tip movement. Let the whole body respond softly to "
            "breathing instead of freezing the upper body; prevent global sliding or "
            "scale drift."
        ),
    },
)
MULTIPOSE_PRESET_BY_ACTION = {
    preset["action_key"]: preset for preset in MULTIPOSE_PRESETS
}
MULTIPOSE_PRESET_BY_PROMPT_ZH = {
    preset["prompt_zh"]: preset for preset in MULTIPOSE_PRESETS
}
MULTIPOSE_SEED_OFFSETS = {
    "running": 101,
    "eating": 211,
    "drinking": 307,
    "lying down": 401,
}
DEFAULT_POSES = [preset["prompt_zh"] for preset in MULTIPOSE_PRESETS]
LEGACY_DEFAULT_POSES = [
    "running | first frame: grounded three-quarter side-view start pose, three paws on one baseline, near front paw slightly raised, four complete separated legs and paws, full tail; video: smooth run-in-place cycle with rhythmic legs and tail counter-sway",
    "eating | first frame: complete full-body natural crouch with muzzle at one small shallow food bowl touching the front paws; video: restrained eating rhythm while the bowl stays fixed",
    "drinking | first frame: complete full-body natural crouch with muzzle at one small shallow water bowl touching the front paws; video: subtle lapping rhythm and small water ripples while the bowl stays fixed",
    "lying down | first frame: complete full-body relaxed sphinx-like prone pose with front paws forward and full tail visible; video: subtle breathing and one slow tail-tip movement",
]
BASELINE_IMAGE_USER_LORAS = [
    {
        "model_name": "user_uploads/70e0a15b036a79af-v3.0.safetensors",
        "original_filename": "宠物小猫 _ 写实摄影_v3.0.safetensors",
        "strength": 0.5,
    },
    {
        "model_name": (
            "user_uploads/"
            "a5c10317e202894c-Qwen-Image-Kitten-Photography_QWEN.safetensors"
        ),
        "original_filename": (
            "Qwen Image-Kitten Photography_QWEN-小猫摄影.safetensors"
        ),
        "strength": 0.6,
    },
]
BASELINE_IMAGE_GENERATION_SETTINGS = {
    "user_lora_mode": "replace",
    "qwen_steps": 6,
    "qwen_cfg": 1.0,
    "qwen_denoise": 1.0,
    "seed": 20260715,
    # Preserve the source task on attempt 1; retries still use the global
    # reduced/final caps and the duplicate-pose gate.
    "canonical_total_cap": 1.1,
    "canonical_per_lora_cap": 0.6,
    "pose_first_total_cap": 1.1,
    "pose_first_per_lora_cap": 0.6,
    "lock_retry_lora_strengths": True,
}

IMAGE_PROFILES: dict[str, dict[str, Any]] = {
    BASELINE_IMAGE_PROFILE_ID: {
        "id": BASELINE_IMAGE_PROFILE_ID,
        "revision": BASELINE_IMAGE_PROFILE_REVISION,
        "label": "20260717 小猫四姿态统一基准 v2",
        "source_job_id": BASELINE_IMAGE_SOURCE_JOB_ID,
        "source_prompt": BASELINE_IMAGE_SOURCE_PROMPT_ZH,
        "source_prompt_en": BASELINE_IMAGE_SOURCE_PROMPT_EN,
        "default_prompt": BASELINE_IMAGE_PROMPT_ZH,
        "default_prompt_en": BASELINE_IMAGE_PROMPT_EN,
        "pose_prompt_source": "verified_profile_presets",
        "pose_presets": [dict(preset) for preset in MULTIPOSE_PRESETS],
        "pose_seed_offsets": dict(MULTIPOSE_SEED_OFFSETS),
        "default_poses": list(DEFAULT_POSES),
        "identity_palette_guard": True,
        "identity_palette_policy": "source_palette_guard_v1",
        "user_loras": [dict(item) for item in BASELINE_IMAGE_USER_LORAS],
        **BASELINE_IMAGE_GENERATION_SETTINGS,
    },
    MULTIPOSE_V1_IMAGE_PROFILE_ID: {
        "id": MULTIPOSE_V1_IMAGE_PROFILE_ID,
        "revision": "multipose-v1",
        "label": "20260717 小猫四姿态统一基准 v1（历史兼容）",
        "source_job_id": BASELINE_IMAGE_SOURCE_JOB_ID,
        "source_prompt": BASELINE_IMAGE_SOURCE_PROMPT_ZH,
        "source_prompt_en": BASELINE_IMAGE_SOURCE_PROMPT_EN,
        "default_prompt": BASELINE_IMAGE_PROMPT_ZH,
        "default_prompt_en": BASELINE_IMAGE_PROMPT_EN,
        "pose_prompt_source": "verified_profile_presets",
        "pose_presets": [dict(preset) for preset in MULTIPOSE_PRESETS],
        "pose_seed_offsets": dict(MULTIPOSE_SEED_OFFSETS),
        "default_poses": list(DEFAULT_POSES),
        "identity_palette_guard": False,
        "identity_palette_policy": None,
        "user_loras": [dict(item) for item in BASELINE_IMAGE_USER_LORAS],
        **BASELINE_IMAGE_GENERATION_SETTINGS,
    },
    LEGACY_EATING_IMAGE_PROFILE_ID: {
        "id": LEGACY_EATING_IMAGE_PROFILE_ID,
        "revision": "eating-v1",
        "label": "20260717 写实小猫进食基准（历史兼容）",
        "source_job_id": BASELINE_IMAGE_SOURCE_JOB_ID,
        "source_prompt": BASELINE_IMAGE_SOURCE_PROMPT_ZH,
        "source_prompt_en": BASELINE_IMAGE_SOURCE_PROMPT_EN,
        "default_prompt": BASELINE_IMAGE_SOURCE_PROMPT_ZH,
        "default_prompt_en": BASELINE_IMAGE_SOURCE_PROMPT_EN,
        "pose_prompt_source": "legacy_global_defaults",
        "pose_presets": [],
        "pose_seed_offsets": {},
        "default_poses": list(LEGACY_DEFAULT_POSES),
        "identity_palette_guard": False,
        "identity_palette_policy": None,
        "user_loras": [dict(item) for item in BASELINE_IMAGE_USER_LORAS],
        **BASELINE_IMAGE_GENERATION_SETTINGS,
    },
}
DEFAULT_IMAGE_PROFILE = os.environ.get(
    "DEFAULT_IMAGE_PROFILE", BASELINE_IMAGE_PROFILE_ID
).strip()
if DEFAULT_IMAGE_PROFILE and DEFAULT_IMAGE_PROFILE not in IMAGE_PROFILES:
    raise ValueError(f"Unknown DEFAULT_IMAGE_PROFILE: {DEFAULT_IMAGE_PROFILE}")

_HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_PROMPT_TRANSLATION_CACHE: dict[str, tuple[str, str]] = {
    BASELINE_IMAGE_PROMPT_ZH: (
        BASELINE_IMAGE_PROMPT_EN,
        "verified_profile_translation",
    ),
    **{
        preset["prompt_zh"]: (
            preset["prompt_en"],
            "verified_profile_translation",
        )
        for preset in MULTIPOSE_PRESETS
    },
}
_PROMPT_TRANSLATION_CACHE_LOCK = threading.RLock()
RESOLUTIONS = {
    "480p": (832, 480),
    "720p": (1280, 704),
    "512x512": (512, 512),
}
OUTPUT_FORMATS = {"mp4", "transparent_gif"}

CHROMA_BACKGROUND_PROMPT = """The entire background must be a perfectly flat, uniform pure #00FF00 chroma-key green field for later background removal. There must be no floor plane, cast shadow, contact shadow, gradient, texture, reflection, vignette, or lighting variation in the green background. Keep generous green padding around the complete subject, and do not use green anywhere on the subject."""
CHROMA_NEGATIVE = (
    "white background, gray background, black background, dark background, non-green background, scenery, floor, floor line, cast shadow, "
    "contact shadow, reflection, background gradient, background texture, green subject, green fur, green clothing"
)

CARTOON_NEGATIVE = (
    "flat 2D drawing, hand-drawn line art, anime, comic, watercolor, low-poly, hard plastic toy, flat fur, "
    "generic cat, different breed, changed fur pattern, changed marking placement, changed face shape, "
    "changed eye color, changed ear shape, changed muzzle, changed body proportions, changed tail, "
    "invented stripes, invented spots, decorative markings, clothes, clothing, shirt, overalls, suspenders, bow tie, "
    "collar, harness, glasses, eyewear, hat, shoes, jewelry, accessory, costume, misaligned eyes, crossed eyes, "
    "deformed muzzle, white sticker border, white outer stroke, halo, unrequested object, duplicate bowl, extra bowl, "
    "floating mark, stray line, water strand from mouth, drool, falling water droplet, wink, blinking, closed eye, "
    "squint, spiral, swirl, motion symbol, "
    "extreme caricature, extreme chibi distortion"
)

QWEN_NEGATIVE = (
    "different identity, changed face, changed fur, changed clothing, duplicate subject, extra subject, "
    "extra animal, extra person, deformed anatomy, extra limbs, missing limbs, cropped body, leaving frame, "
    "text, watermark, logo, busy background, low quality, blurry"
)
WAN_NEGATIVE = (
    "identity change, face change, clothing change, color change, scene cut, camera movement, zoom, pan, "
    "rotation, duplicate subject, extra subject, extra limbs, missing limbs, deformed anatomy, cropped body, "
    "leaving frame, walking toward camera, moving closer, subject drift, scale change, text, watermark, "
    "violent motion, severe motion blur, flicker, frame jitter"
)
CARTOON_WAN_NEGATIVE = (
    "flat 2D style, style drift, face distortion, warped face, melting face, changing head shape, asymmetric face, "
    "eye size change, eye position change, pupil drift, crossed eyes, lazy eye, extra eye, disappearing eye, "
    "nose drift, muzzle deformation, mouth morphing, wide open mouth, changing expression, clothes appearing, "
    "shirt, overalls, suspenders, bow tie, collar, harness, glasses, eyewear, hat, shoes, accessories appearing, "
    "blink, wink, closed eye, eyelid motion, squint, fur crawling, texture crawling, color flicker, "
    "changed fur markings, changed eye color, generic cat"
)

JOB_QUEUE: queue.Queue[str] = queue.Queue(maxsize=MAX_QUEUE_SIZE)
JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.RLock()
STOP_EVENT = threading.Event()
WORKER_THREAD: threading.Thread | None = None
REMBG_SESSION_LOCK = threading.Lock()
REMBG_SESSION: Any | None = None

TERMINAL_JOB_STATUSES = {"completed", "failed", "cancelled"}


class JobCancelled(RuntimeError):
    """Raised when a user requests cancellation of a generation job."""


Image.MAX_IMAGE_PIXELS = 40_000_000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def link(output_node: str, slot: int = 0) -> list[Any]:
    return [output_node, slot]


def wan_model_paths() -> list[Path]:
    paths = [MODEL_ROOT / "text_encoders" / WAN_CLIP, MODEL_ROOT / "vae" / WAN_VAE]
    if WAN_ENGINE == "wan2.2-i2v-a14b-fp8":
        paths.extend(
            [
                MODEL_ROOT / "diffusion_models" / WAN_HIGH_NOISE_MODEL,
                MODEL_ROOT / "diffusion_models" / WAN_LOW_NOISE_MODEL,
            ]
        )
    else:
        paths.append(MODEL_ROOT / "diffusion_models" / WAN_MODEL)
    return paths


def wan_engine_metadata(
    steps: int = WAN_STEPS,
    cfg: float = WAN_CFG,
    shift: float = WAN_SHIFT,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "engine": WAN_ENGINE,
        "frames": FRAME_COUNT,
        "fps": FPS,
        "steps": steps,
        "cfg": cfg,
        "shift": shift,
        "vae": WAN_VAE,
        "text_encoder": WAN_CLIP,
    }
    if WAN_ENGINE == "wan2.2-i2v-a14b-fp8":
        metadata.update(
            {
                "high_noise_model": WAN_HIGH_NOISE_MODEL,
                "low_noise_model": WAN_LOW_NOISE_MODEL,
                "split_step": max(
                    1, min(steps - 1, round(steps * WAN_SPLIT_STEP / WAN_STEPS))
                ),
                "precision": "fp8_scaled",
            }
        )
    else:
        metadata.update({"model": WAN_MODEL, "precision": "fp16"})
    return metadata


def required_model_paths(include_video: bool = True) -> list[Path]:
    paths = [
        MODEL_ROOT / "diffusion_models" / QWEN_MODEL,
        MODEL_ROOT / "text_encoders" / QWEN_CLIP,
        MODEL_ROOT / "vae" / QWEN_VAE,
        MODEL_ROOT / "loras" / QWEN_LORA,
    ]
    if include_video:
        paths.extend(wan_model_paths())
    if QWEN_STYLE_LORA:
        paths.append(MODEL_ROOT / "loras" / QWEN_STYLE_LORA)
    if include_video and FUN_CONTROL_ENABLED:
        paths.extend(
            [
                FUN_CONTROL_PYTHON,
                FUN_CONTROL_SCRIPT,
                FUN_CONTROL_PROJECT_ROOT / "videox_fun" / "__init__.py",
                FUN_CONTROL_MODEL_ROOT / "diffusion_pytorch_model.safetensors",
                FUN_CONTROL_MODEL_ROOT / "models_t5_umt5-xxl-enc-bf16.pth",
                FUN_CONTROL_MODEL_ROOT / "Wan2.2_VAE.pth",
                FUN_CONTROL_RUNNING_VIDEO,
            ]
        )
    return paths


def require_models(include_video: bool = True) -> None:
    missing = [
        str(path)
        for path in required_model_paths(include_video=include_video)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Required local model files are missing: " + ", ".join(missing)
        )


def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    if not API_KEY:
        raise HTTPException(status_code=503, detail="Server API key is not configured.")
    if not x_api_key or not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(
            status_code=401, detail="Invalid or missing X-API-Key header."
        )


def status_path(job_id: str) -> Path:
    return OUTPUT_ROOT / job_id / "status.json"


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def public_job(job: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(job))
    result.pop("active_prompt_id", None)
    result["status_url"] = f"{API_PREFIX}/jobs/{job['job_id']}"
    result["cancel_url"] = f"{API_PREFIX}/jobs/{job['job_id']}/cancel"
    for output in result.get("outputs", []):
        output["download_url"] = (
            f"{API_PREFIX}/jobs/{job['job_id']}/files/{output['filename']}"
        )
    return result


def persist_job(job_id: str) -> None:
    with JOBS_LOCK:
        job = dict(JOBS[job_id])
    write_json_atomic(status_path(job_id), job)


def update_job(job_id: str, **changes: Any) -> dict[str, Any]:
    with JOBS_LOCK:
        job = JOBS[job_id]
        job.update(changes)
        job["updated_at"] = utc_now()
        snapshot = dict(job)
    write_json_atomic(status_path(job_id), snapshot)
    return snapshot


def append_output(job_id: str, output: dict[str, Any]) -> None:
    with JOBS_LOCK:
        outputs = list(JOBS[job_id].get("outputs", []))
        outputs.append(output)
        JOBS[job_id]["outputs"] = outputs
        JOBS[job_id]["updated_at"] = utc_now()
        snapshot = dict(JOBS[job_id])
    write_json_atomic(status_path(job_id), snapshot)


def append_recovery_event(job_id: str, event: dict[str, Any]) -> None:
    with JOBS_LOCK:
        history = list(JOBS[job_id].get("recovery_history", []))
        history.append(event)
        JOBS[job_id]["recovery_history"] = history
        JOBS[job_id]["automatic_recovery_used"] = True
        if event.get("strategy") == "rembg_u2net_fallback":
            JOBS[job_id]["degraded_recovery"] = True
        JOBS[job_id]["updated_at"] = utc_now()
        snapshot = dict(JOBS[job_id])
    write_json_atomic(status_path(job_id), snapshot)


def append_background_removal_failure(job_id: str, event: dict[str, Any]) -> None:
    """Record an opaque image fallback without claiming that recovery succeeded."""
    with JOBS_LOCK:
        failures = list(JOBS[job_id].get("background_removal_failures", []))
        failures.append(event)
        JOBS[job_id]["background_removal_failures"] = failures
        JOBS[job_id]["background_removal_failed"] = True
        JOBS[job_id]["transparent_outputs_complete"] = False
        JOBS[job_id]["updated_at"] = utc_now()
        snapshot = dict(JOBS[job_id])
    write_json_atomic(status_path(job_id), snapshot)


def get_job(job_id: str) -> dict[str, Any]:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        path = status_path(job_id)
        if path.is_file():
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
                with JOBS_LOCK:
                    JOBS[job_id] = job
            except (OSError, json.JSONDecodeError):
                job = None
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return public_job(job)


def cancel_requested(job_id: str) -> bool:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        return bool(
            job and (job.get("cancel_requested") or job.get("status") == "cancelled")
        )


def raise_if_cancelled(job_id: str | None) -> None:
    if job_id and cancel_requested(job_id):
        raise JobCancelled("Generation was cancelled by the user.")


def interrupt_comfy_prompt(prompt_id: str | None) -> None:
    if not prompt_id:
        return
    errors: list[str] = []
    try:
        response = requests.post(
            f"{COMFY_URL}/queue",
            json={"delete": [prompt_id]},
            timeout=30,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        errors.append(f"queue delete: {exc}")
    try:
        response = requests.post(
            f"{COMFY_URL}/interrupt",
            json={"prompt_id": prompt_id},
            timeout=30,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        errors.append(f"interrupt: {exc}")
    if errors:
        LOGGER.warning(
            "Could not fully interrupt ComfyUI prompt %s: %s",
            prompt_id,
            "; ".join(errors),
        )


def mark_job_cancelled(job_id: str) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        progress = job.get("progress", 0) if job else 0
    update_job(
        job_id,
        status="cancelled",
        stage="cancelled",
        progress=progress,
        cancel_requested=True,
        active_prompt_id=None,
        current_pose_index=None,
        current_pose=None,
        error=None,
        finished_at=utc_now(),
    )


def parse_poses(
    raw: str | None,
    pose_count: int,
    *,
    defaults: list[str] | None = None,
) -> list[str]:
    if not 1 <= pose_count <= MAX_POSES:
        raise HTTPException(
            status_code=422, detail=f"pose_count must be between 1 and {MAX_POSES}."
        )
    if raw and raw.strip():
        text = raw.strip()
        try:
            if text.startswith("["):
                value = json.loads(text)
                if not isinstance(value, list) or not all(
                    isinstance(item, str) for item in value
                ):
                    raise ValueError("poses JSON must be an array of strings")
                poses = value
            else:
                separator = "|" if "|" in text else "\n"
                poses = text.split(separator)
        except (json.JSONDecodeError, ValueError) as exc:
            raise HTTPException(
                status_code=422, detail=f"Invalid poses: {exc}"
            ) from exc
        poses = [item.strip() for item in poses if item.strip()]
    else:
        pose_defaults = list(defaults) if defaults is not None else DEFAULT_POSES
        poses = pose_defaults[:pose_count]
        while len(poses) < pose_count:
            poses.append(f"distinct natural pose number {len(poses) + 1}")
    if not poses:
        raise HTTPException(status_code=422, detail="At least one pose is required.")
    if len(poses) > MAX_POSES:
        raise HTTPException(
            status_code=422, detail=f"At most {MAX_POSES} poses are allowed per job."
        )
    if any(len(item) > 300 for item in poses):
        raise HTTPException(
            status_code=422, detail="Each pose must be at most 300 characters."
        )
    return poses


def contains_han(text: str) -> bool:
    return bool(_HAN_RE.search(text))


def normalize_chinese_prompt_for_translation(text: str) -> str:
    """Clarify reference-image wording that generic MT often reads as an upload command."""
    replacements = (
        ("参考上传的图片中的", "参考图片中的"),
        ("参考上传图片中的", "参考图片中的"),
        ("参考上传的图片中", "参考图片中"),
        ("参考上传图片中", "参考图片中"),
        ("上传的图片中的", "参考图片中的"),
        ("上传图片中的", "参考图片中的"),
        ("上传的图片中", "参考图片中"),
        ("上传图片中", "参考图片中"),
        ("上传的猫咪", "参考图片中的猫咪"),
        ("上传猫咪", "参考图片中的猫咪"),
        ("上传的宠物", "参考图片中的宠物"),
        ("上传宠物", "参考图片中的宠物"),
        ("上传的图片", "参考图片"),
        ("上传图片", "参考图片"),
    )
    normalized = text
    for source, target in replacements:
        normalized = normalized.replace(source, target)
    return normalized


def prompt_translator_missing_paths() -> list[str]:
    required = [
        PROMPT_TRANSLATOR_PYTHON,
        PROMPT_TRANSLATOR_SCRIPT,
        PROMPT_TRANSLATOR_MODEL / "config.json",
        PROMPT_TRANSLATOR_MODEL / "source.spm",
        PROMPT_TRANSLATOR_MODEL / "target.spm",
        PROMPT_TRANSLATOR_MODEL / "vocab.json",
        PROMPT_TRANSLATOR_MODEL / "translation_ready.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    weights = [
        PROMPT_TRANSLATOR_MODEL / "pytorch_model.bin",
        PROMPT_TRANSLATOR_MODEL / "model.safetensors",
    ]
    if not any(path.is_file() for path in weights):
        missing.append(
            str(PROMPT_TRANSLATOR_MODEL / "{pytorch_model.bin|model.safetensors}")
        )
    return missing


def translate_texts_to_english(
    texts: list[str],
) -> tuple[list[str], dict[str, Any]]:
    """Translate only Han-containing strings and fail closed if translation is unavailable."""
    unique_pending: list[str] = []
    normalized_pending: list[str] = []
    methods: dict[str, str] = {}
    resolved: dict[str, str] = {}
    with _PROMPT_TRANSLATION_CACHE_LOCK:
        cached = dict(_PROMPT_TRANSLATION_CACHE)
    for text in texts:
        if not contains_han(text):
            resolved[text] = text
            methods[text] = "identity"
        elif text in cached:
            resolved[text], methods[text] = cached[text]
        elif text not in unique_pending:
            unique_pending.append(text)

    if unique_pending:
        normalized_pending = [
            normalize_chinese_prompt_for_translation(text) for text in unique_pending
        ]
        if not PROMPT_TRANSLATION_ENABLED:
            raise HTTPException(
                status_code=503,
                detail="Chinese prompt translation is disabled on this server.",
            )
        missing = prompt_translator_missing_paths()
        if missing:
            raise HTTPException(
                status_code=503,
                detail=(
                    "The local Chinese-to-English prompt translator is not ready: "
                    + ", ".join(missing)
                ),
            )
        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": "",
                "TOKENIZERS_PARALLELISM": "false",
                "OMP_NUM_THREADS": str(PROMPT_TRANSLATION_THREADS),
                "MKL_NUM_THREADS": str(PROMPT_TRANSLATION_THREADS),
            }
        )
        try:
            process = subprocess.run(
                [
                    str(PROMPT_TRANSLATOR_PYTHON),
                    str(PROMPT_TRANSLATOR_SCRIPT),
                    "--model",
                    str(PROMPT_TRANSLATOR_MODEL),
                ],
                input=json.dumps({"texts": normalized_pending}, ensure_ascii=False),
                capture_output=True,
                text=True,
                timeout=PROMPT_TRANSLATION_TIMEOUT,
                check=False,
                env=environment,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Local prompt translation failed to start: {type(exc).__name__}.",
            ) from exc
        if process.returncode != 0:
            LOGGER.error(
                "Local prompt translator exited with %s: %s",
                process.returncode,
                (process.stderr or process.stdout or "unknown error")[-2000:],
            )
            raise HTTPException(
                status_code=502,
                detail="Local Chinese-to-English prompt translation failed.",
            )
        try:
            payload = json.loads(process.stdout)
            translated = payload["translations"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise HTTPException(
                status_code=502,
                detail="Local prompt translator returned an invalid response.",
            ) from exc
        if not isinstance(translated, list) or len(translated) != len(unique_pending):
            raise HTTPException(
                status_code=502,
                detail="Local prompt translator returned an incomplete response.",
            )
        for source, target in zip(unique_pending, translated, strict=True):
            normalized = str(target).strip()
            if not normalized or len(normalized) > 4000 or contains_han(normalized):
                raise HTTPException(
                    status_code=502,
                    detail="Local prompt translation did not produce valid English text.",
                )
            resolved[source] = normalized
            methods[source] = "local_opus_mt_zh_en"
            with _PROMPT_TRANSLATION_CACHE_LOCK:
                _PROMPT_TRANSLATION_CACHE[source] = (
                    normalized,
                    "local_opus_mt_zh_en",
                )

    ordered = [resolved[text] for text in texts]
    applied_methods = {methods[text] for text in texts if methods[text] != "identity"}
    provider = (
        next(iter(applied_methods))
        if len(applied_methods) == 1
        else ("mixed" if applied_methods else "identity")
    )
    return ordered, {
        "applied": bool(applied_methods),
        "provider": provider,
        "model": (
            "Helsinki-NLP/opus-mt-zh-en"
            if "local_opus_mt_zh_en" in applied_methods
            else None
        ),
        "source_language": "zh",
        "target_language": "en",
        "translated_item_count": sum(methods[text] != "identity" for text in texts),
        "semantic_normalization_count": sum(
            source != normalized
            for source, normalized in zip(
                unique_pending,
                normalized_pending,
                strict=True,
            )
        ),
        "methods": [methods[text] for text in texts],
    }


def infer_pose_action(source: str, translated: str) -> str | None:
    combined = f"{source}\n{translated}".lower()
    if re.search(r"喝水|饮水|舔水|\bdrinking\b|\bdrink(?:ing)?\s+water\b", combined):
        return "drinking"
    if re.search(
        r"进食|吃饭|吃东西|猫粮|食盆|\beating\b|\bfood bowl\b|\bkibble\b", combined
    ):
        return "eating"
    if re.search(r"奔跑|跑步|快跑|\brunning\b|\brun(?:ning)?\b", combined):
        return "running"
    if re.search(r"趴卧|趴下|躺下|卧姿|\blying down\b|\bprone\b", combined):
        return "lying down"
    return None


def normalize_translated_pose(source: str, translated: str) -> str:
    translated = translated.strip()
    first_key = translated.split("|", 1)[0].strip().lower()
    if first_key in {"running", "eating", "drinking", "lying", "lying down"}:
        return translated
    inferred_action = infer_pose_action(source, translated)
    if not inferred_action:
        return translated
    return f"{inferred_action} | first frame: {translated}"


def translate_prompt_bundle(
    prompt: str,
    poses: list[str],
) -> tuple[str, list[str], dict[str, Any]]:
    translated, metadata = translate_texts_to_english([prompt, *poses])
    prompt_en = translated[0]
    poses_en = [
        normalize_translated_pose(source, target)
        for source, target in zip(poses, translated[1:], strict=True)
    ]
    metadata.update(
        {
            "prompt_applied": contains_han(prompt),
            "pose_applied_count": sum(contains_han(pose) for pose in poses),
        }
    )
    return prompt_en, poses_en, metadata


def normalize_uploaded_image(
    raw: bytes,
    output_path: Path,
    *,
    identity_source_path: Path | None = None,
) -> None:
    try:
        with Image.open(io.BytesIO(raw)) as probe:
            probe.verify()
        with Image.open(io.BytesIO(raw)) as source:
            rgba = source.convert("RGBA")
            if identity_source_path is not None:
                identity_source_path.parent.mkdir(parents=True, exist_ok=True)
                rgba.save(identity_source_path, format="PNG", optimize=True)
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            background.alpha_composite(rgba)
            rgb = background.convert("RGB")
            if max(rgb.size) > 4096:
                rgb.thumbnail((4096, 4096), Image.Resampling.LANCZOS)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            rgb.save(output_path, format="PNG", optimize=True)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(
            status_code=422, detail="Uploaded file is not a valid supported image."
        ) from exc


def validate_safetensors_file(path: Path) -> int:
    size = path.stat().st_size
    if size < 16:
        raise HTTPException(
            status_code=422, detail="Uploaded LoRA is not a valid safetensors file."
        )
    with path.open("rb") as handle:
        header_size = int.from_bytes(handle.read(8), byteorder="little", signed=False)
        if header_size <= 2 or header_size > min(size - 8, 64 * 1024 * 1024):
            raise HTTPException(
                status_code=422,
                detail="Uploaded LoRA has an invalid safetensors header.",
            )
        try:
            header = json.loads(handle.read(header_size).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HTTPException(
                status_code=422,
                detail="Uploaded LoRA has an unreadable safetensors header.",
            ) from exc
    if not isinstance(header, dict):
        raise HTTPException(
            status_code=422, detail="Uploaded LoRA has an invalid tensor index."
        )
    data_size = size - 8 - header_size
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    if not tensors:
        raise HTTPException(
            status_code=422, detail="Uploaded LoRA contains no tensors."
        )
    lora_key_seen = False
    for key, value in tensors.items():
        if any(marker in key.lower() for marker in ("lora", "lokr", "loha", "dora")):
            lora_key_seen = True
        if not isinstance(value, dict):
            raise HTTPException(
                status_code=422, detail="Uploaded LoRA has an invalid tensor entry."
            )
        offsets = value.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(item, int) for item in offsets)
            or offsets[0] < 0
            or offsets[0] > offsets[1]
            or offsets[1] > data_size
        ):
            raise HTTPException(
                status_code=422, detail="Uploaded LoRA has invalid tensor offsets."
            )
    if not lora_key_seen:
        raise HTTPException(
            status_code=422,
            detail="Uploaded safetensors file does not appear to contain LoRA weights.",
        )
    return len(tensors)


async def store_user_lora(upload: UploadFile | None) -> dict[str, Any] | None:
    if upload is None or not upload.filename:
        return None
    original_name = upload.filename.replace("\\", "/").rsplit("/", 1)[-1]
    if not original_name.lower().endswith(".safetensors"):
        await upload.close()
        raise HTTPException(
            status_code=422, detail="User LoRA must be a .safetensors file."
        )
    USER_LORA_DIR.mkdir(parents=True, exist_ok=True)
    temporary = USER_LORA_DIR / f".upload-{uuid4().hex}.tmp"
    digest = hashlib.sha256()
    total = 0
    try:
        with temporary.open("wb") as target:
            while chunk := await upload.read(4 * 1024 * 1024):
                total += len(chunk)
                if total > MAX_LORA_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"User LoRA exceeds {MAX_LORA_UPLOAD_BYTES} bytes.",
                    )
                digest.update(chunk)
                target.write(chunk)
        if total == 0:
            raise HTTPException(status_code=422, detail="Uploaded LoRA is empty.")
        tensor_count = validate_safetensors_file(temporary)
        sha256 = digest.hexdigest()
        safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(original_name).stem).strip(
            "-._"
        )
        safe_stem = (safe_stem or "uploaded-lora")[:48]
        stored_name = f"{sha256[:16]}-{safe_stem}.safetensors"
        existing_matches = sorted(USER_LORA_DIR.glob(f"{sha256[:16]}-*.safetensors"))
        final_path = (
            existing_matches[0] if existing_matches else USER_LORA_DIR / stored_name
        )
        stored_name = final_path.name
        if final_path.exists():
            temporary.unlink()
        else:
            os.replace(temporary, final_path)
            final_path.chmod(0o640)
        return {
            "model_name": f"user_uploads/{stored_name}",
            "filename": stored_name,
            "original_filename": original_name,
            "sha256": sha256,
            "size_bytes": total,
            "tensor_count": tensor_count,
        }
    finally:
        await upload.close()
        temporary.unlink(missing_ok=True)


def resolve_user_lora(model_name: str | None) -> str | None:
    if not model_name or not model_name.strip():
        return None
    normalized = model_name.strip().replace("\\", "/")
    relative = Path(normalized)
    if (
        relative.is_absolute()
        or len(relative.parts) != 2
        or relative.parts[0] != "user_uploads"
        or relative.suffix.lower() != ".safetensors"
    ):
        raise HTTPException(status_code=422, detail="Invalid user LoRA model name.")
    candidate = (MODEL_ROOT / "loras" / relative).resolve()
    try:
        candidate.relative_to(USER_LORA_DIR.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Invalid user LoRA path.") from exc
    if not candidate.is_file():
        raise HTTPException(status_code=422, detail="Uploaded user LoRA is missing.")
    return relative.as_posix()


def normalize_user_lora_specs(
    value: Any,
    *,
    legacy_model_name: str | None = None,
    legacy_original_filename: str | None = None,
    legacy_strength: float = 1.0,
    resolve_models: bool = True,
) -> list[dict[str, Any]]:
    """Normalize the new multi-LoRA payload while retaining legacy single-LoRA support."""
    if isinstance(value, str):
        if not value.strip():
            value = None
        else:
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise HTTPException(
                    status_code=422,
                    detail="user_loras must be a valid JSON array.",
                ) from exc
    if value is None:
        raw_items: list[Any] = []
        if legacy_model_name and legacy_model_name.strip():
            raw_items.append(
                {
                    "model_name": legacy_model_name,
                    "original_filename": legacy_original_filename,
                    "strength": legacy_strength,
                }
            )
    elif isinstance(value, list):
        raw_items = value
    else:
        raise HTTPException(status_code=422, detail="user_loras must be a JSON array.")
    if len(raw_items) > MAX_USER_LORAS:
        raise HTTPException(
            status_code=422,
            detail=f"At most {MAX_USER_LORAS} user LoRAs may be fused in one job.",
        )

    normalized: list[dict[str, Any]] = []
    seen_models: set[str] = set()
    resolved_total_size = 0
    for index, item in enumerate(raw_items, start=1):
        if not isinstance(item, dict):
            raise HTTPException(
                status_code=422,
                detail=f"user_loras[{index - 1}] must be an object.",
            )
        model_name = item.get("model_name") or item.get("user_lora") or item.get("name")
        if not isinstance(model_name, str) or not model_name.strip():
            raise HTTPException(
                status_code=422,
                detail=f"user_loras[{index - 1}].model_name is required.",
            )
        if resolve_models:
            resolved_name = resolve_user_lora(model_name)
            if resolved_name is None:
                raise HTTPException(
                    status_code=422,
                    detail=f"user_loras[{index - 1}].model_name is invalid.",
                )
            resolved_total_size += (
                (MODEL_ROOT / "loras" / Path(resolved_name)).stat().st_size
            )
            if resolved_total_size > MAX_LORA_FUSION_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        "Combined user LoRA files exceed "
                        f"{MAX_LORA_FUSION_BYTES} bytes."
                    ),
                )
        else:
            resolved_name = model_name.strip().replace("\\", "/")
        if resolved_name in seen_models:
            raise HTTPException(
                status_code=422,
                detail=f"Duplicate user LoRA is not allowed: {resolved_name}.",
            )
        seen_models.add(resolved_name)
        try:
            strength = float(item.get("strength", 1.0))
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=422,
                detail=f"user_loras[{index - 1}].strength must be a number.",
            ) from exc
        if not 0.0 <= strength <= 2.0:
            raise HTTPException(
                status_code=422,
                detail=f"user_loras[{index - 1}].strength must be between 0.0 and 2.0.",
            )
        original_filename = item.get("original_filename")
        if original_filename is not None:
            original_filename = (
                str(original_filename).replace("\\", "/").rsplit("/", 1)[-1][:255]
            )
        normalized.append(
            {
                "model_name": resolved_name,
                "original_filename": original_filename,
                "strength": strength,
            }
        )
    return normalized


def resolve_image_profile(
    profile_id: str | None,
    *,
    request_kind: str,
) -> dict[str, Any] | None:
    normalized = (profile_id or "").strip()
    if not normalized:
        return None
    if request_kind not in {"image", "video"}:
        raise HTTPException(
            status_code=422,
            detail="image_profile requires an image or video request.",
        )
    profile = IMAGE_PROFILES.get(normalized)
    if profile is None:
        raise HTTPException(status_code=422, detail="Unknown image_profile.")
    return profile


def image_profile_public(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": profile["id"],
        "revision": profile.get("revision"),
        "label": profile["label"],
        "source_job_id": profile["source_job_id"],
        "source_prompt": profile.get("source_prompt"),
        "source_prompt_en": profile.get("source_prompt_en"),
        "default_prompt": profile["default_prompt"],
        "default_prompt_en": profile["default_prompt_en"],
        "pose_prompt_source": profile.get("pose_prompt_source"),
        "pose_presets": [
            {
                "id": preset["id"],
                "action_key": preset["action_key"],
                "label": preset["label"],
                "prompt_zh": preset["prompt_zh"],
                "prompt_en": preset["prompt_en"],
                "video_prompt_en": preset["video_prompt_en"],
            }
            for preset in profile.get("pose_presets", [])
        ],
        "pose_seed_offsets": dict(profile.get("pose_seed_offsets", {})),
        "identity_palette_guard": bool(profile.get("identity_palette_guard", False)),
        "identity_palette_policy": profile.get("identity_palette_policy"),
        "lock_retry_lora_strengths": bool(
            profile.get("lock_retry_lora_strengths", False)
        ),
        "user_loras": [
            {
                "original_filename": item["original_filename"],
                "strength": item["strength"],
                "available": (
                    MODEL_ROOT / "loras" / Path(item["model_name"])
                ).is_file(),
            }
            for item in profile["user_loras"]
        ],
        "user_lora_mode": profile["user_lora_mode"],
        "qwen_steps": profile["qwen_steps"],
        "qwen_cfg": profile["qwen_cfg"],
        "qwen_denoise": profile["qwen_denoise"],
        "seed": profile["seed"],
    }


def merge_image_profile_loras(
    profile: dict[str, Any] | None,
    requested: list[dict[str, Any]],
    *,
    requested_mode: str,
    resolve_models: bool = True,
) -> tuple[list[dict[str, Any]], str, str]:
    """Apply baseline LoRAs, allowing uploaded LoRAs to replace or stack them."""
    if profile is None:
        return [dict(item) for item in requested], requested_mode, "request"
    profile_loras = normalize_user_lora_specs(
        profile["user_loras"],
        resolve_models=resolve_models,
    )
    if not requested:
        return profile_loras, str(profile["user_lora_mode"]), "image_profile"
    if requested_mode == "replace":
        return [dict(item) for item in requested], requested_mode, "user_override"
    combined = [*profile_loras, *requested]
    normalized = normalize_user_lora_specs(
        combined,
        resolve_models=resolve_models,
    )
    return normalized, requested_mode, "image_profile_plus_user"


def cap_user_lora_specs(
    specs: list[dict[str, Any]],
    *,
    total_cap: float,
    per_lora_cap: float,
) -> tuple[list[dict[str, Any]], float]:
    """Scale all LoRAs proportionally so stage-specific caps cannot lock the pose."""
    total_strength = sum(float(item["strength"]) for item in specs)
    maximum_strength = max((float(item["strength"]) for item in specs), default=0.0)
    if total_strength <= 0.0 or maximum_strength <= 0.0:
        return [dict(item) for item in specs], 1.0
    scale = min(
        1.0,
        total_cap / total_strength if total_cap > 0.0 else 0.0,
        per_lora_cap / maximum_strength if per_lora_cap > 0.0 else 0.0,
    )
    scaled = [
        {**item, "strength": round(float(item["strength"]) * scale, 6)}
        for item in specs
    ]
    return scaled, scale


def reduced_user_lora_specs(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Preserve blend ratios while capping the combined recovery influence."""
    reduced, _ = cap_user_lora_specs(
        specs,
        total_cap=QWEN_RECOVERY_USER_LORA_STRENGTH,
        per_lora_cap=2.0,
    )
    return reduced


def action_key(pose: str) -> str:
    return pose.split("|", 1)[0].strip().lower()


def action_uses_bowl(pose: str) -> bool:
    return action_key(pose) in {"eating", "drinking"}


def pose_seed_offset(pose: str, index: int) -> int:
    action = action_key(pose)
    canonical_action = "lying down" if action == "lying" else action
    return MULTIPOSE_SEED_OFFSETS.get(canonical_action, max(1, index))


def qwen_first_frame_spec(pose: str) -> str:
    action = action_key(pose)
    detail = pose.split("|", 1)[1].strip() if "|" in pose else ""
    detail = re.split(r";\s*video\s*:", detail, maxsplit=1, flags=re.IGNORECASE)[
        0
    ].strip()
    detail = re.sub(r"^first\s+frame\s*:\s*", "", detail, flags=re.IGNORECASE)
    if action == "running":
        safety_spec = MULTIPOSE_PRESET_BY_ACTION["running"]["prompt_en"]
        if detail.casefold() == safety_spec.casefold():
            return safety_spec
        return (
            f"{safety_spec}; requested first-frame detail: {detail}"
            if detail
            else safety_spec
        )
    if not detail:
        detail = pose.strip()
    return detail or pose.strip()


def action_first_frame_rules(pose: str) -> str:
    action = action_key(pose)
    if action == "running":
        return (
            "No prop. This still image is a calm static pre-run standing pose for later animation, not an illustrated "
            "running instant, jump, leap, gallop suspension, or flying pose. Use a three-quarter side view with the "
            "camera pulled back. The torso is upright and visibly supported above the legs. Plant and visibly flatten "
            "all four complete paws on the same horizontal baseline with no gap below them. Show exactly four long "
            "legs, each continuously attached to the torso, spatially separated, staggered front-to-back, and ending "
            "in one complete paw. Offset the far-side paws horizontally so they remain readable rather than overlapping "
            "the near-side paws. Each leg must be at least one quarter of the torso height. "
            "Keep both hind legs substantial and fully formed; do not hide, shorten, fuse, overlap, fragment, or "
            "detach any leg or paw. The head turns only slightly toward camera so both eyes and identity markings remain "
            "readable. At least three paws must make unambiguous ground contact in the final silhouette; any airborne "
            "body pose invalidates the image."
        )
    if action == "eating":
        return (
            "Include exactly one small shallow matte food bowl immediately beside the front paws, with a narrow visible "
            "green gap so the bowl remains a separate foreground component. Center the bowl directly below the lowered "
            "muzzle and slightly ahead of both front paws. Both forelegs extend down from the shoulders and both complete "
            "front paws are visibly planted on the same baseline. Use a calm static pre-bite feeding posture: the nose "
            "is immediately above the bowl opening and the mouth is closed or naturally hidden by the rim. Keep every "
            "piece of kibble below the rim inside the bowl; no food may appear at the mouth, above the rim, or in a paw. "
            "The cat never raises a foreleg and never grasps, lifts, carries, or holds food. The bowl is the only prop. "
            "Do not let the bowl touch, overlap, hide, or replace a paw, and keep the eyes, face silhouette, paws, and "
            "body readable."
        )
    if action == "drinking":
        return (
            "Include exactly one small shallow matte water bowl immediately beside the front paws, with a narrow visible "
            "green gap so the bowl remains a separate foreground component. The bowl is the only prop. Do not let it "
            "touch, overlap, hide, or replace a paw. Show a recognizable active-drinking keyframe: the "
            "muzzle is lowered naturally and exactly one tiny solid-pink triangular cat tongue tip emerges from the "
            "lower lip and visibly intersects the still water with no air gap. Its visible length must not exceed one "
            "twentieth of the head width. The tongue must look anatomical and remain visibly pink, never transparent, "
            "blue, elongated, or detached. Keep both eyes, the "
            "face silhouette, paws, and body readable."
        )
    if action in {"lying", "lying down"}:
        return "No prop. Use a relaxed low full-body pose; do not crop the paws, belly, haunches, or tail."
    return "Do not add a prop unless the action specification explicitly requires one."


def action_video_rules(pose: str) -> str:
    action = action_key(pose)
    canonical_action = "lying down" if action == "lying" else action
    preset = MULTIPOSE_PRESET_BY_ACTION.get(canonical_action)
    if preset:
        return preset["video_prompt_en"]
    return "Use only small action-appropriate body motion while the torso root, head design, and camera remain stable."


def build_cartoon_reference_prompt(
    base_prompt: str = "",
    *,
    palette_lock: str = "",
) -> str:
    creative_direction = (
        base_prompt.strip() or "premium cute stylized 3D animated-film character"
    )
    palette_section = f"\n\n{palette_lock.strip()}" if palette_lock.strip() else ""
    return f"""Use case: identity-preserving stylized 3D character design.
BACKGROUND RULE — non-negotiable: every pixel outside the character must be one perfectly flat uniform pure #00FF00 green. No black, gray, white, studio, or environmental background; no floor and no shadow.

The uploaded photo is the only identity reference. Create a canonical premium cute 3D character render of this exact same cat. Keep the cat recognizable and change only the rendering style, never the cat's identity or natural appearance.

User creative direction: {creative_direction}
Apply this direction only to rendering quality, fur finish, expression, and lighting on the character. It must not override the identity, full-body anatomy, composition, clothing, or background rules below.
Ignore every pose, action, camera angle, framing, crop, background, and subject-placement instruction found in the user creative direction. The canonical pose, composition, and background rules below are authoritative.

Identity lock: preserve the exact fur base colors, visible markings and their placement, head and face silhouette, cheek width, eye color, ear shape, nose color, muzzle, fur length, breed traits, and body proportions from the uploaded cat. Preserve natural asymmetry. Do not turn lighting, shadows, blur, or fur texture into new markings. Do not invent stripes or spots.{palette_section}

Target style from the supplied example: high-end stylized 3D animated-film character, slightly oversized round head, softly rounded cheeks, large glossy round eyes with clean catchlights, tiny natural nose, compact closed mouth, soft finely groomed realistic fur, gentle subsurface softness, polished color-faithful neutral high-key lighting on the character only, cute and intelligent expression. Lighting must never tint or replace the source fur palette. Apply this visual language while deriving every fur color and facial trait from the uploaded cat. The result should look like a premium character render rather than a real animal photograph, flat drawing, or plastic toy.

Natural cat only: the character is completely unclothed and wears no glasses, collar, bow tie, shirt, overalls, suspenders, harness, hat, shoes, jewelry, costume, or other accessory. Show the natural fur across the neck, chest, torso, and legs. Do not import the example image's wardrobe or eyewear.

Composition lock: this is a pulled-back full-body character sheet, never a portrait or close-up. Show one complete cat from both ear tips through the torso to every paw and the full tail tip. Use a neutral compact standing pose at a slight three-quarter front angle. Show exactly four anatomically attached legs and four complete paws with visible separation; show the tail continuously from base to tip. Nothing may be hidden behind the torso, fused, shortened, cropped, fragmented, or outside the canvas. Keep at least 12% clear green margin on every side and let the cat occupy no more than 72% of the canvas width and height. Both eyes are fully open, equally sized, fully visible and level, pupils centered, face neutral and symmetrical, mouth closed. The fur silhouette must meet green directly with no white border or halo.

{CHROMA_BACKGROUND_PROMPT} The pure green background rule overrides all aesthetic lighting language. Keep the cat at a useful readable scale. No floor, shadow, text, logo, or watermark."""


def build_qwen_prompt(
    base_prompt: str,
    pose: str,
    transparent_gif: bool = False,
    quality_retry: int = 0,
) -> str:
    background_prompt = (
        CHROMA_BACKGROUND_PROMPT
        if transparent_gif
        else "Use a clean simple studio-like background with soft even lighting."
    )
    if transparent_gif:
        first_frame_rules = action_first_frame_rules(pose)
        first_frame_spec = qwen_first_frame_spec(pose)
        bowl_action = action_uses_bowl(pose)
        retry_instruction = ""
        if quality_retry >= 2:
            retry_instruction = """
VALIDATION RETRY: the previous result failed one or more identity, pose, scale, or green-background checks. Recreate the action-specific full-body keyframe while preserving the exact character identity, source coat palette, pulled-back framing, and pure-green canvas rules."""
        if quality_retry >= 3:
            if bowl_action:
                retry_instruction += """
FINAL BOWL-ACTION RETRY: keep a fixed pulled-back camera and do not enlarge the cat or bowl. Express the action only through a natural low torso, lowered head and muzzle position beside the small bowl. Keep at least 12% pure-green clearance on every edge, keep the complete cat and bowl within 72% of the canvas width and height, and keep all four canvas corners exactly pure green."""
            elif action_key(pose) == "running":
                retry_instruction += """
FINAL POSE-CORRECTION RETRY: prioritize a complete grounded pre-run stance over visible motion. Keep all four legs and paws separated and readable, with at least three unambiguous ground contacts. Do not make the action dynamic and do not lift the body."""
            else:
                retry_instruction += """
FINAL POSE-CORRECTION RETRY: make the requested action unmistakable at thumbnail size. Change the torso angle, leg placement, head height, and tail placement enough to be visibly distinct from a neutral sitting or standing reference. No prop may appear."""
        if quality_retry >= 4 and action_key(pose) == "running":
            retry_instruction += """
RUNNING GROUND-CONTACT RECOVERY: ignore the motion implied by the action name in this still frame. Render a calm static four-paw standing ready pose. All four legs and paws must be fully visible, separated, and grounded; do not lift a paw, hide a far-side leg, or depict forward motion. The later video model, not this image, will create the run."""
        if quality_retry >= 4 and bowl_action:
            retry_instruction += """
BOWL PALETTE-AND-CANVAS RECOVERY: use neutral-white character lighting with no warm grading or amber bounce on the fur. Copy every fur color and marking region from the approved character exactly. Keep the small bowl visually separated from both front paws by a narrow pure-green gap. Pull back enough that the complete cat and bowl have generous green padding and every canvas corner remains pure #00FF00."""
        subject_contact_prompt = (
            "Fur and the single action-required bowl must touch green directly."
            if bowl_action
            else "Fur must touch green directly. No prop may appear."
        )
        return f"""Use case: identity-preserving pose edit for a stylized 3D character.
The input is the approved canonical character reference. Repose this exact character only; do not redesign the face, natural fur, or rendering style.

Style source: inherit the approved canonical reference's exact rendering style, fur finish, expression design, lighting, and identity. Do not reuse any user-authored pose, action, camera, framing, crop, background, or subject-placement instruction from the canonical design stage.

Character lock: copy the exact same head silhouette, rounded cheeks, eye size, eye color and spacing, pupils, catchlights, ears, nose, closed muzzle, fur colors and marking placement, fur length, body proportions, soft groomed 3D fur, and character lighting. Both eyes must remain fully open, equally sized, level, and recognizable whenever the action angle allows. The cat is completely unclothed: no glasses, collar, bow tie, shirt, overalls, suspenders, harness, hat, shoes, jewelry, costume, or accessory. Do not add markings, unrequested props, floating marks, motion symbols, a white border, or a halo. Where the reference does not show an area, continue the nearest dominant body fur color conservatively with no new pattern.

Required still-frame pose only: {first_frame_spec}
First-frame action rules: {first_frame_rules}
{retry_instruction}
This is a clean static character keyframe that will be animated later. Do not illustrate the later motion in this image. Show only the balanced readable starting posture; do not add speed lines, duplicated limbs, motion blur, airborne action, or floating motion symbols. Do not blink, wink, or squint.
Only the "Required still-frame pose" and "First-frame action rules" control body posture, camera angle, framing, and action props. Every conflicting instruction is void.

Anatomy checklist — all items are mandatory: exactly one continuous cat silhouette; exactly four legs attached naturally to the torso; exactly four complete paws; two complete ears; one tail visible continuously from base to tip. No limb may be hidden, fused, shortened, duplicated, detached, reduced to a fragment, or strongly foreshortened. Show the complete full-body cat with at least 12% green clearance on every canvas edge. Keep the subject at no more than 72% of canvas width and height. Keep the identity-bearing face readable while respecting the requested action angle. Keep the exact premium cute 3D character style from the reference. {background_prompt} Stable centered composition. {subject_contact_prompt} No floor, shadow, text, logo, or watermark."""

    return f"""Use the input image as the exact identity and appearance reference.
Create one polished first frame for a short image-to-video animation.

Creative direction: {base_prompt}
Required still-frame pose: {qwen_first_frame_spec(pose)}

Preserve the main subject's exact identity, face, distinctive colors and markings, hairstyle or fur, clothing, accessories, body proportions, and visual style. Keep the same number of main subjects as the input. Place the complete subject in the required pose, with anatomically correct limbs and the full body visible. {background_prompt} Use a stable centered composition, no text, and no watermark. The result must be a clean, coherent first frame suitable for animation."""


def build_wan_prompt(base_prompt: str, pose: str, transparent_gif: bool = False) -> str:
    background_prompt = (
        "Keep the background perfectly uniform pure #00FF00 chroma-key green in every frame, with no shadow, gradient, texture, flicker, color shift, or green spill on the subject."
        if transparent_gif
        else "Preserve the background from the first frame."
    )
    if transparent_gif:
        motion_rules = action_video_rules(pose)
        subject_position_prompt = (
            "Keep the complete character and the single action-required bowl centered "
            "at the same screen position, scale, and perspective."
            if action_uses_bowl(pose)
            else "Keep the complete character centered at the same screen position, "
            "scale, and perspective. No prop may appear."
        )
        return f"""Animate this exact character keyframe into one smooth five-second stylized 3D character loop.

FACE CONSISTENCY: preserve the recognizable head silhouette, cheek width, eye color and spacing, nose, muzzle, ears, and facial fur markings while allowing natural coordinated head, ear, muzzle, and expression motion required by the action. Do not hold any facial pixel or screen coordinate fixed. Avoid melting, independent feature drift, or sudden proportion changes.

STYLE AND IDENTITY LOCK: preserve this exact cat's fur colors and marking placement, soft groomed fur, premium stylized 3D rendering, color-faithful neutral high-key lighting, body proportions, paws, and full tail. Lighting must never introduce an amber, orange, brown, or sepia cast that is absent from the approved keyframe. The cat remains completely unclothed with natural fur: no glasses, collar, bow tie, shirt, overalls, suspenders, harness, hat, shoes, jewelry, costume, or accessory may appear. Never switch to a real photograph, flat 2D drawing, plastic toy, or a different character design.

User motion direction: {base_prompt}
Starting pose and action: {pose}
The user direction controls mood and secondary motion only and cannot override the identity, natural unclothed appearance, action, or style constraints.

Required action motion: {motion_rules}
Motion safety: allow the head, ears, face, neck, chest, legs, and tail to move naturally and coherently for the action. Keep only the torso root, background, and camera trajectory stable. Motion must be smooth and periodic. Never create duplicate limbs, rubbery anatomy, facial melting, fur texture crawling, sudden pose changes, or random whole-frame movement.

{background_prompt} {subject_position_prompt} Fixed camera, no translation across the canvas, zoom, scene cut, new subject, or body part leaving the frame."""

    return f"""Create a smooth five-second animation from this exact first frame.
Creative direction: {base_prompt}
Starting pose and action: {pose}
Preserve the exact subject identity, face, colors and markings, clothing, accessories, body proportions, and visual style from the first frame. {background_prompt} Use natural controlled body, head, facial, hair or fur movement appropriate to the prompt. Animate in place: keep the complete subject centered at the same screen position, scale, and perspective throughout. Unless the creative direction explicitly asks for travel, do not walk toward the camera, move closer, or leave the frame. Fixed stable camera, coherent motion, no scene cuts, no new subjects, and no sudden appearance changes."""


def is_fun_control_pose(pose: str) -> bool:
    if not FUN_CONTROL_ENABLED:
        return False
    action = pose.split("|", 1)[0].strip().lower()
    return action in {"running", "run", "奔跑", "跑步"} or action.startswith("running ")


def build_fun_control_prompt(base_prompt: str, pose: str, transparent_gif: bool) -> str:
    background = (
        "The entire background remains perfectly uniform pure #00FF00 chroma-key green in every frame, without a floor, horizon, shadow, gradient, texture, reflection, flicker, or green spill on the cat."
        if transparent_gif
        else "Preserve the clean background from the supplied reference keyframe."
    )
    return f"""Animate the exact full-body stylized 3D cartoon cat from the reference image.

User direction: {base_prompt}
Required action: {pose}

Follow the supplied side-view quadruped control video precisely. Preserve the same fur colors, facial markings, face shape, eye color, ears, muzzle, body proportions, paws, and tail as the reference image. Use anatomically correct feline locomotion: four distinct legs and four distinct paws, coherent shoulder and hip motion, correct paw contacts, natural spine compression and extension, and natural tail counterbalance. The control sequence determines the leg positions and contact timing. Never invent an independent gait or random paw motion.

Keep the recognizable face structurally stable while allowing natural head motion. The cat remains completely unclothed, with no collar, glasses, harness, clothing, shoes, or accessories. Keep the subject centered at a stable scale with a fixed side-view camera. {background} High temporal consistency, soft groomed fur, clean edges, no text, and no watermark."""


FUN_CONTROL_NEGATIVE = (
    "deformed face, changing identity, changing fur pattern, extra legs, missing legs, three legs, five legs, "
    "duplicated paws, fused paws, merged limbs, broken joints, twisted ankles, impossible gait, human arms, "
    "human hands, bipedal motion, random paw waving, sliding feet, foot skating, floating paws, disconnected limbs, "
    "body morphing, size drift, camera shake, camera motion, zoom, crop, close-up, clothes, collar, glasses, harness, "
    "accessories, cluttered background, floor, horizon, shadows on background, text, watermark, blur, flicker, jitter, low quality"
)


def qwen_recovery_profiles(
    *,
    user_lora: str | None,
    user_lora_strength: float,
    user_lora_mode: str,
    steps: int,
    cfg: float,
    denoise: float,
    user_loras: list[dict[str, Any]] | None = None,
    lock_strengths: bool = False,
) -> list[dict[str, Any]]:
    effective_user_loras = normalize_user_lora_specs(
        user_loras,
        legacy_model_name=user_lora,
        legacy_strength=user_lora_strength,
        resolve_models=False,
    )
    reduced_user_loras = reduced_user_lora_specs(effective_user_loras)
    requested_total_strength = sum(
        float(item["strength"]) for item in effective_user_loras
    )
    reduced_total_strength = sum(float(item["strength"]) for item in reduced_user_loras)
    reduced_scale = (
        reduced_total_strength / requested_total_strength
        if requested_total_strength > 0.0
        else 1.0
    )
    final_user_loras = (
        [dict(item) for item in effective_user_loras]
        if lock_strengths
        else reduced_user_loras
    )
    final_scale = 1.0 if lock_strengths else reduced_scale
    final_name = (
        "requested_final_seed_retry" if lock_strengths else "safe_reduced_user_lora"
    )
    final_mode = user_lora_mode if lock_strengths else "stack"

    def legacy_fields(specs: list[dict[str, Any]]) -> dict[str, Any]:
        first = specs[0] if specs else None
        return {
            "user_lora": first["model_name"] if first else None,
            "user_lora_strength": first["strength"] if first else 0.0,
        }

    profiles = [
        {
            "name": "requested",
            "user_loras": [dict(item) for item in effective_user_loras],
            **legacy_fields(effective_user_loras),
            "user_lora_strength_scale": 1.0,
            "user_lora_mode": user_lora_mode,
            "steps": steps,
            "cfg": cfg,
            "denoise": denoise,
            "use_user_prompt": True,
        },
        {
            "name": "requested_seed_retry",
            "user_loras": [dict(item) for item in effective_user_loras],
            **legacy_fields(effective_user_loras),
            "user_lora_strength_scale": 1.0,
            "user_lora_mode": user_lora_mode,
            "steps": max(6, steps),
            "cfg": cfg,
            "denoise": 1.0,
            "use_user_prompt": True,
        },
        {
            "name": final_name,
            "user_loras": final_user_loras,
            **legacy_fields(final_user_loras),
            "user_lora_strength_scale": final_scale,
            "user_lora_mode": final_mode,
            "steps": max(8, min(10, steps)),
            "cfg": 1.0,
            "denoise": 1.0,
            "use_user_prompt": True,
        },
    ]
    return profiles[:QWEN_CHROMA_MAX_ATTEMPTS]


def qwen_pose_profiles(
    *,
    user_lora: str | None,
    user_lora_strength: float,
    user_lora_mode: str,
    steps: int,
    cfg: float,
    denoise: float,
    user_loras: list[dict[str, Any]] | None = None,
    first_total_cap: float | None = None,
    first_per_lora_cap: float | None = None,
    lock_strengths: bool = False,
) -> list[dict[str, Any]]:
    """Build progressively less pose-locking profiles for every image/video request."""
    requested = normalize_user_lora_specs(
        user_loras,
        legacy_model_name=user_lora,
        legacy_strength=user_lora_strength,
        resolve_models=False,
    )
    requested_total = sum(float(item["strength"]) for item in requested)
    effective_first_total_cap = (
        QWEN_POSE_USER_LORA_TOTAL_CAP
        if first_total_cap is None
        else float(first_total_cap)
    )
    effective_first_per_lora_cap = (
        QWEN_POSE_USER_LORA_PER_LORA_CAP
        if first_per_lora_cap is None
        else float(first_per_lora_cap)
    )
    if lock_strengths:
        configurations = [
            (
                "pose_capped",
                effective_first_total_cap,
                effective_first_per_lora_cap,
                user_lora_mode,
                steps,
            ),
            (
                "pose_exact_seed_retry",
                effective_first_total_cap,
                effective_first_per_lora_cap,
                user_lora_mode,
                max(6, min(8, steps)),
            ),
            (
                "pose_exact_final_retry",
                effective_first_total_cap,
                effective_first_per_lora_cap,
                user_lora_mode,
                max(8, min(10, steps)),
            ),
        ]
    else:
        configurations = [
            (
                "pose_capped",
                effective_first_total_cap,
                effective_first_per_lora_cap,
                user_lora_mode,
                steps,
            ),
            (
                "pose_reduced_retry",
                QWEN_POSE_USER_LORA_RETRY_TOTAL_CAP,
                min(
                    QWEN_POSE_USER_LORA_PER_LORA_CAP,
                    QWEN_POSE_USER_LORA_RETRY_TOTAL_CAP,
                ),
                "stack",
                max(6, min(8, steps)),
            ),
            (
                "pose_minimal_retry",
                QWEN_POSE_USER_LORA_FINAL_TOTAL_CAP,
                min(
                    QWEN_POSE_USER_LORA_PER_LORA_CAP,
                    QWEN_POSE_USER_LORA_FINAL_TOTAL_CAP,
                ),
                "stack",
                max(8, min(10, steps)),
            ),
        ]
    profiles: list[dict[str, Any]] = []
    for quality_retry, (
        name,
        total_cap,
        per_lora_cap,
        mode,
        effective_steps,
    ) in enumerate(
        configurations,
        start=1,
    ):
        effective_loras, scale = cap_user_lora_specs(
            requested,
            total_cap=total_cap,
            per_lora_cap=per_lora_cap,
        )
        first = effective_loras[0] if effective_loras else None
        profiles.append(
            {
                "name": name,
                "user_loras": effective_loras,
                "user_lora": first["model_name"] if first else None,
                "user_lora_strength": first["strength"] if first else 0.0,
                "user_lora_strength_scale": (scale if requested_total > 0.0 else 1.0),
                "user_lora_mode": mode,
                "steps": effective_steps,
                "cfg": max(1.0, cfg),
                "denoise": 1.0 if quality_retry > 1 else denoise,
                "use_user_prompt": False,
                "quality_retry": quality_retry,
                "total_strength_cap": total_cap,
                "per_lora_strength_cap": per_lora_cap,
            }
        )
    return profiles[:QWEN_POSE_MAX_ATTEMPTS]


def pose_generation_profiles(
    pose: str,
    profiles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Add action-specific seed retries for difficult running and bowl keyframes."""
    selected = [
        {
            **profile,
            "user_loras": [dict(item) for item in profile.get("user_loras", [])],
        }
        for profile in profiles
    ]
    if not selected:
        return selected
    action = action_key(pose)
    if action == "running":
        extra_attempts = QWEN_RUNNING_POSE_EXTRA_ATTEMPTS
        retry_name = "running_ground_contact_seed_retry"
    elif action_uses_bowl(pose):
        extra_attempts = QWEN_BOWL_POSE_EXTRA_ATTEMPTS
        retry_name = "bowl_palette_canvas_seed_retry"
    else:
        return selected
    retry_source = selected[0] if action_uses_bowl(pose) else selected[-1]
    for extra_index in range(extra_attempts):
        repeated = {
            **retry_source,
            "user_loras": [dict(item) for item in retry_source.get("user_loras", [])],
        }
        repeated["name"] = f"{retry_name}_{extra_index + 1}"
        repeated["quality_retry"] = max(
            4,
            int(selected[-1]["quality_retry"]) + 1,
        )
        repeated["steps"] = max(10, int(repeated["steps"]))
        selected.append(repeated)
    return selected


def pose_recovery_profiles(
    profiles: list[dict[str, Any]],
    minimum_profile_index: int,
) -> list[dict[str, Any]]:
    selected = [dict(profile) for profile in profiles[minimum_profile_index:]]
    if not selected:
        selected = [dict(profiles[-1])]
    while len(selected) < QWEN_POSE_MAX_ATTEMPTS:
        repeated = dict(selected[-1])
        repeated["name"] = f"{repeated['name']}_seed_retry_{len(selected) + 1}"
        selected.append(repeated)
    return selected[:QWEN_POSE_MAX_ATTEMPTS]


def qwen_workflow(
    input_name: str,
    prompt: str,
    negative_prompt: str,
    seed: int,
    prefix: str,
    user_lora: str | None = None,
    user_lora_strength: float = 1.0,
    user_lora_mode: str = "replace",
    steps: int = 4,
    cfg: float = 1.0,
    denoise: float = 1.0,
    builtin_style_strength: float | None = None,
    user_loras: list[dict[str, Any]] | None = None,
    suppress_builtin_style: bool = False,
) -> dict[str, Any]:
    effective_user_loras = normalize_user_lora_specs(
        user_loras,
        legacy_model_name=user_lora,
        legacy_strength=user_lora_strength,
        resolve_models=False,
    )
    active_user_loras = [
        item for item in effective_user_loras if float(item["strength"]) != 0.0
    ]
    effective_style_strength = (
        QWEN_STYLE_LORA_STRENGTH
        if builtin_style_strength is None
        else float(builtin_style_strength)
    )
    if not 0.0 <= effective_style_strength <= 2.0:
        raise ValueError("builtin_style_strength must be between 0.0 and 2.0")
    workflow: dict[str, Any] = {
        "1": {"class_type": "LoadImage", "inputs": {"image": input_name}},
        "2": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": QWEN_MODEL, "weight_dtype": "default"},
        },
        "3": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": QWEN_CLIP,
                "type": "qwen_image",
                "device": "default",
            },
        },
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": QWEN_VAE}},
        "5": {"class_type": "FluxKontextImageScale", "inputs": {"image": link("1")}},
        "6": {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {
                "clip": link("3"),
                "vae": link("4"),
                "image1": link("5"),
                "prompt": prompt,
            },
        },
        "7": {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {
                "clip": link("3"),
                "vae": link("4"),
                "image1": link("5"),
                "prompt": negative_prompt,
            },
        },
        "8": {
            "class_type": "FluxKontextMultiReferenceLatentMethod",
            "inputs": {
                "conditioning": link("6"),
                "reference_latents_method": "index_timestep_zero",
            },
        },
        "9": {
            "class_type": "FluxKontextMultiReferenceLatentMethod",
            "inputs": {
                "conditioning": link("7"),
                "reference_latents_method": "index_timestep_zero",
            },
        },
        "10": {
            "class_type": "VAEEncode",
            "inputs": {"pixels": link("5"), "vae": link("4")},
        },
        "11": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {"model": link("2"), "shift": 3.1},
        },
        "12": {
            "class_type": "CFGNorm",
            "inputs": {"model": link("11"), "strength": 1.0},
        },
        "13": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": link("12"),
                "lora_name": QWEN_LORA,
                "strength_model": 1.0,
            },
        },
        "14": {
            "class_type": "KSampler",
            "inputs": {
                "model": link("13"),
                "positive": link("8"),
                "negative": link("9"),
                "latent_image": link("10"),
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": denoise,
            },
        },
        "15": {
            "class_type": "VAEDecode",
            "inputs": {"samples": link("14"), "vae": link("4")},
        },
        "16": {
            "class_type": "SaveImage",
            "inputs": {"images": link("15"), "filename_prefix": prefix},
        },
    }
    model_node = "13"
    apply_builtin_style = (
        bool(QWEN_STYLE_LORA)
        and not suppress_builtin_style
        and not (active_user_loras and user_lora_mode == "replace")
    )
    if apply_builtin_style:
        workflow["17"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": link("13"),
                "lora_name": QWEN_STYLE_LORA,
                "strength_model": effective_style_strength,
            },
        }
        model_node = "17"
    for lora_index, lora_spec in enumerate(active_user_loras, start=18):
        node_id = str(lora_index)
        workflow[node_id] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": link(model_node),
                "lora_name": lora_spec["model_name"],
                "strength_model": lora_spec["strength"],
            },
        }
        model_node = node_id
    workflow["14"]["inputs"]["model"] = link(model_node)
    return workflow


def wan_workflow(
    input_name: str,
    prompt: str,
    negative_prompt: str,
    seed: int,
    width: int,
    height: int,
    prefix: str,
    steps: int = WAN_STEPS,
    cfg: float = WAN_CFG,
    shift: float = WAN_SHIFT,
) -> dict[str, Any]:
    split_step = max(1, min(steps - 1, round(steps * WAN_SPLIT_STEP / WAN_STEPS)))
    if WAN_ENGINE == "wan2.2-i2v-a14b-fp8":
        return {
            "1": {
                "class_type": "UNETLoader",
                "inputs": {
                    "unet_name": WAN_HIGH_NOISE_MODEL,
                    "weight_dtype": "default",
                },
            },
            "2": {
                "class_type": "UNETLoader",
                "inputs": {
                    "unet_name": WAN_LOW_NOISE_MODEL,
                    "weight_dtype": "default",
                },
            },
            "3": {
                "class_type": "CLIPLoader",
                "inputs": {"clip_name": WAN_CLIP, "type": "wan", "device": "default"},
            },
            "4": {"class_type": "VAELoader", "inputs": {"vae_name": WAN_VAE}},
            "5": {"class_type": "LoadImage", "inputs": {"image": input_name}},
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": link("3"), "text": prompt},
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": link("3"), "text": negative_prompt},
            },
            "8": {
                "class_type": "WanImageToVideo",
                "inputs": {
                    "positive": link("6"),
                    "negative": link("7"),
                    "vae": link("4"),
                    "start_image": link("5"),
                    "width": width,
                    "height": height,
                    "length": FRAME_COUNT,
                    "batch_size": 1,
                },
            },
            "9": {
                "class_type": "ModelSamplingSD3",
                "inputs": {"model": link("1"), "shift": shift},
            },
            "10": {
                "class_type": "KSamplerAdvanced",
                "inputs": {
                    "model": link("9"),
                    "add_noise": "enable",
                    "noise_seed": seed,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "positive": link("8", 0),
                    "negative": link("8", 1),
                    "latent_image": link("8", 2),
                    "start_at_step": 0,
                    "end_at_step": split_step,
                    "return_with_leftover_noise": "enable",
                },
            },
            "11": {
                "class_type": "ModelSamplingSD3",
                "inputs": {"model": link("2"), "shift": shift},
            },
            "12": {
                "class_type": "KSamplerAdvanced",
                "inputs": {
                    "model": link("11"),
                    "add_noise": "disable",
                    "noise_seed": 0,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "positive": link("8", 0),
                    "negative": link("8", 1),
                    "latent_image": link("10"),
                    "start_at_step": split_step,
                    "end_at_step": steps,
                    "return_with_leftover_noise": "disable",
                },
            },
            "13": {
                "class_type": "VAEDecode",
                "inputs": {"samples": link("12"), "vae": link("4")},
            },
            "14": {
                "class_type": "CreateVideo",
                "inputs": {"images": link("13"), "fps": FPS},
            },
            "15": {
                "class_type": "SaveVideo",
                "inputs": {
                    "video": link("14"),
                    "filename_prefix": prefix,
                    "format": "auto",
                    "codec": "auto",
                },
            },
        }
    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": WAN_MODEL, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": WAN_CLIP, "type": "wan", "device": "default"},
        },
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": WAN_VAE}},
        "4": {"class_type": "LoadImage", "inputs": {"image": input_name}},
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": link("2"), "text": prompt},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": link("2"), "text": negative_prompt},
        },
        "7": {
            "class_type": "Wan22ImageToVideoLatent",
            "inputs": {
                "vae": link("3"),
                "start_image": link("4"),
                "width": width,
                "height": height,
                "length": FRAME_COUNT,
                "batch_size": 1,
            },
        },
        "8": {
            "class_type": "ModelSamplingSD3",
            "inputs": {"model": link("1"), "shift": shift},
        },
        "9": {
            "class_type": "KSampler",
            "inputs": {
                "model": link("8"),
                "positive": link("5"),
                "negative": link("6"),
                "latent_image": link("7"),
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "uni_pc",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        "10": {
            "class_type": "VAEDecode",
            "inputs": {"samples": link("9"), "vae": link("3")},
        },
        "11": {
            "class_type": "CreateVideo",
            "inputs": {"images": link("10"), "fps": FPS},
        },
        "12": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": link("11"),
                "filename_prefix": prefix,
                "format": "auto",
                "codec": "auto",
            },
        },
    }


def wait_comfy_idle(timeout: int = 3600, job_id: str | None = None) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        raise_if_cancelled(job_id)
        response = requests.get(f"{COMFY_URL}/queue", timeout=30)
        response.raise_for_status()
        payload = response.json()
        if not payload.get("queue_running") and not payload.get("queue_pending"):
            return
        time.sleep(5)
    raise TimeoutError("Timed out waiting for the ComfyUI queue to become idle.")


def free_comfy_models() -> None:
    response = requests.post(
        f"{COMFY_URL}/free",
        json={"unload_models": True, "free_memory": True},
        timeout=60,
    )
    response.raise_for_status()


def queue_and_wait(
    workflow: dict[str, Any],
    stage_name: str,
    timeout: int,
    job_id: str | None = None,
) -> dict[str, Any]:
    raise_if_cancelled(job_id)
    response = requests.post(
        f"{COMFY_URL}/prompt",
        json={"prompt": workflow, "client_id": str(uuid4())},
        timeout=60,
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"ComfyUI rejected {stage_name}: {response.status_code} {response.text[:2000]}"
        )
    prompt_id = response.json()["prompt_id"]
    if job_id:
        update_job(job_id, active_prompt_id=prompt_id)
    LOGGER.info("%s queued as ComfyUI prompt %s", stage_name, prompt_id)
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            if job_id and cancel_requested(job_id):
                interrupt_comfy_prompt(prompt_id)
                raise JobCancelled("Generation was cancelled by the user.")
            history_response = requests.get(
                f"{COMFY_URL}/history/{prompt_id}", timeout=30
            )
            history_response.raise_for_status()
            history = history_response.json()
            if prompt_id in history:
                entry = history[prompt_id]
                current_status = entry.get("status", {})
                if current_status.get("completed"):
                    LOGGER.info("%s completed", stage_name)
                    return entry
                messages = current_status.get("messages", [])
                terminal_errors = {"execution_error", "execution_interrupted"}
                if any(
                    message and message[0] in terminal_errors for message in messages
                ):
                    if job_id and cancel_requested(job_id):
                        raise JobCancelled("Generation was cancelled by the user.")
                    raise RuntimeError(
                        f"ComfyUI {stage_name} execution failed: {json.dumps(messages, ensure_ascii=False)[:4000]}"
                    )
            time.sleep(5)
        raise TimeoutError(f"ComfyUI {stage_name} timed out after {timeout} seconds.")
    finally:
        if job_id:
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                if job and job.get("active_prompt_id") == prompt_id:
                    job["active_prompt_id"] = None
                    job["updated_at"] = utc_now()
                    snapshot = dict(job)
                else:
                    snapshot = None
            if snapshot is not None:
                write_json_atomic(status_path(job_id), snapshot)


def output_files(entry: dict[str, Any]) -> list[Path]:
    found: list[Path] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if "filename" in value and value.get("type", "output") == "output":
                found.append(
                    COMFY_OUTPUT / value.get("subfolder", "") / value["filename"]
                )
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(entry.get("outputs", {}))
    return list(dict.fromkeys(found))


def first_output(entry: dict[str, Any], suffixes: set[str], stage_name: str) -> Path:
    matches = [
        path
        for path in output_files(entry)
        if path.suffix.lower() in suffixes and path.is_file()
    ]
    if not matches:
        raise RuntimeError(f"{stage_name} completed without an expected output file.")
    return matches[0]


def prepare_landscape_keyframe(
    source: Path,
    target: Path,
    size: tuple[int, int],
    background_rgb: tuple[int, int, int] = (255, 255, 255),
) -> None:
    with Image.open(source) as loaded:
        image = loaded.convert("RGBA")
    background_rgba = (*background_rgb, 255)
    flattened = Image.new("RGBA", image.size, background_rgba)
    flattened.alpha_composite(image)
    rgb = flattened.convert("RGB")
    if background_rgb == (0, 255, 0):
        red, green, blue = rgb.split()
        other_channels = ImageChops.lighter(red, blue)
        green_dominance = ImageChops.subtract(green, other_channels)
        strongly_green = green_dominance.point(lambda value: 255 if value > 40 else 0)
        bright_enough = green.point(lambda value: 255 if value > 80 else 0)
        green_background = ImageChops.multiply(strongly_green, bright_enough)
        foreground = ImageChops.invert(green_background)
    else:
        difference = ImageChops.difference(
            rgb, Image.new("RGB", rgb.size, background_rgb)
        ).convert("L")
        foreground = difference.point(lambda value: 255 if value > 12 else 0)
    bbox = foreground.getbbox()
    if bbox:
        padding = max(16, int(max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * 0.06))
        bbox = (
            max(0, bbox[0] - padding),
            max(0, bbox[1] - padding),
            min(image.width, bbox[2] + padding),
            min(image.height, bbox[3] + padding),
        )
        image = image.crop(bbox)
    canvas = Image.new("RGBA", size, background_rgba)
    image.thumbnail(
        (int(size[0] * 0.88), int(size[1] * 0.90)), Image.Resampling.LANCZOS
    )
    canvas.alpha_composite(
        image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2)
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(target, format="PNG", optimize=True)


def require_ffmpeg() -> None:
    if not Path(FFMPEG_BIN).is_file():
        raise FileNotFoundError(f"FFmpeg executable was not found at {FFMPEG_BIN}.")


def run_ffmpeg(command: list[str], stage_name: str, timeout: int = 900) -> None:
    require_ffmpeg()
    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if process.returncode != 0:
        detail = (process.stderr or process.stdout or "unknown FFmpeg error").strip()
        raise RuntimeError(f"{stage_name} failed: {detail[-4000:]}")


def convert_chroma_image_to_transparent_png(
    source: Path,
    target: Path,
    *,
    require_clean_background: bool = False,
) -> dict[str, object]:
    return build_transparent_cutout(
        source,
        target,
        require_clean_background=require_clean_background,
    )


def evaluate_chroma_cutout(source: Path, target: Path) -> dict[str, Any]:
    metrics = dict(
        convert_chroma_image_to_transparent_png(
            source,
            target,
            require_clean_background=False,
        )
    )
    issues: list[str] = []
    if float(metrics["foreground_fraction"]) > 0.85:
        issues.append("foreground_fraction_too_large")
    if any(int(value) > 0 for value in metrics["corners"]):
        issues.append("nontransparent_canvas_corner")
    metrics["acceptable"] = not issues
    metrics["issues"] = issues
    metrics["method"] = "chroma_key"
    if issues:
        target.unlink(missing_ok=True)
    return metrics


def get_rembg_session() -> Any:
    global REMBG_SESSION
    with REMBG_SESSION_LOCK:
        if REMBG_SESSION is None:
            from rembg import new_session

            REMBG_SESSION = new_session(CHROMA_FALLBACK_REMBG_MODEL)
    return REMBG_SESSION


IDENTITY_PALETTE_CATEGORIES = (
    "black",
    "white",
    "gray",
    "warm_brown_or_orange",
    "red_or_pink",
    "yellow_or_gold",
    "green",
    "blue_or_cyan",
    "purple",
)
IDENTITY_PALETTE_PROMPT_LABELS = {
    "black": "black",
    "white": "white",
    "gray": "neutral gray",
    "warm_brown_or_orange": "brown/orange/ginger/tan",
    "red_or_pink": "red/pink",
    "yellow_or_gold": "yellow/gold",
    "green": "green",
    "blue_or_cyan": "blue/cyan",
    "purple": "purple",
}
IDENTITY_PALETTE_CHROMATIC = (
    "warm_brown_or_orange",
    "red_or_pink",
    "yellow_or_gold",
    "green",
    "blue_or_cyan",
    "purple",
)


def identity_palette_signature(
    path: Path,
    maximum_size: int = 512,
    *,
    exclude_bottom_fraction: float = 0.0,
) -> dict[str, Any]:
    """Summarize visible foreground colors without treating the background as coat."""
    with Image.open(path) as loaded:
        image = loaded.convert("RGBA")
    if max(image.size) > maximum_size:
        scale = maximum_size / float(max(image.size))
        image = image.resize(
            (
                max(1, int(round(image.width * scale))),
                max(1, int(round(image.height * scale))),
            ),
            Image.Resampling.LANCZOS,
        )
    rgba = np.asarray(image)
    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3]
    foreground: np.ndarray | None = None
    mask_method = "unknown"
    if np.any(alpha < 250):
        alpha_foreground = alpha >= 64
        alpha_fraction = float(np.count_nonzero(alpha_foreground)) / float(
            alpha_foreground.size
        )
        if 0.02 <= alpha_fraction <= 0.95:
            foreground = alpha_foreground
            mask_method = "alpha"
    if foreground is None:
        red = rgb[:, :, 0].astype(np.int16)
        green = rgb[:, :, 1].astype(np.int16)
        blue = rgb[:, :, 2].astype(np.int16)
        green_background = (
            (green > 80)
            & ((green - np.maximum(red, blue)) > 35)
            & (green > red * 1.18)
            & (green > blue * 1.18)
        )
        green_fraction = float(np.count_nonzero(green_background)) / float(
            green_background.size
        )
        border_size = max(2, int(round(min(green_background.shape) * 0.06)))
        border_mask = np.zeros(green_background.shape, dtype=bool)
        border_mask[:border_size, :] = True
        border_mask[-border_size:, :] = True
        border_mask[:, :border_size] = True
        border_mask[:, -border_size:] = True
        border_green_fraction = float(
            np.count_nonzero(green_background & border_mask)
        ) / float(max(1, np.count_nonzero(border_mask)))
        corner_height = max(2, int(round(green_background.shape[0] * 0.10)))
        corner_width = max(2, int(round(green_background.shape[1] * 0.10)))
        corner_blocks = (
            green_background[:corner_height, :corner_width],
            green_background[:corner_height, -corner_width:],
            green_background[-corner_height:, :corner_width],
            green_background[-corner_height:, -corner_width:],
        )
        green_corner_count = sum(
            float(np.count_nonzero(block)) / float(block.size) >= 0.60
            for block in corner_blocks
        )
        if (
            green_fraction >= 0.08
            and border_green_fraction >= 0.45
            and green_corner_count >= 3
        ):
            foreground = ~green_background
            mask_method = "green_screen"
    if foreground is None:
        try:
            from rembg import remove

            recovered = remove(image, session=get_rembg_session())
            if not isinstance(recovered, Image.Image):
                recovered = Image.open(io.BytesIO(recovered)).convert("RGBA")
            recovered_alpha = np.asarray(recovered.convert("RGBA"))[:, :, 3]
            semantic_foreground = recovered_alpha >= 64
            semantic_fraction = float(np.count_nonzero(semantic_foreground)) / float(
                semantic_foreground.size
            )
            if 0.02 <= semantic_fraction <= 0.95:
                foreground = semantic_foreground
                mask_method = f"rembg_{CHROMA_FALLBACK_REMBG_MODEL}"
        except Exception:  # noqa: BLE001 - palette validation safely degrades to a warning
            LOGGER.warning(
                "Could not segment %s for identity palette analysis; "
                "palette guard is unavailable for this image",
                path.name,
                exc_info=True,
            )
    if foreground is None:
        return {
            "available": False,
            "reason": "foreground_segmentation_unavailable",
            "mask_method": mask_method,
            "exclude_bottom_fraction": round(
                max(0.0, min(0.45, float(exclude_bottom_fraction))),
                6,
            ),
        }
    foreground_u8 = foreground.astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        foreground_u8,
        connectivity=8,
    )
    if component_count > 1:
        largest_label = max(
            range(1, component_count),
            key=lambda label: int(stats[label, cv2.CC_STAT_AREA]),
        )
        foreground_u8 = (labels == largest_label).astype(np.uint8)
    exclude_bottom_fraction = max(0.0, min(0.45, float(exclude_bottom_fraction)))
    if exclude_bottom_fraction > 0.0:
        foreground_ys = np.where(foreground_u8 > 0)[0]
        if len(foreground_ys):
            y_min = int(foreground_ys.min())
            y_max = int(foreground_ys.max()) + 1
        else:
            y_min = 0
            y_max = foreground_u8.shape[0]
        cutoff = y_min + int(round((y_max - y_min) * (1.0 - exclude_bottom_fraction)))
        foreground_u8[max(1, cutoff) :, :] = 0
    if min(foreground_u8.shape) >= 24:
        foreground_u8 = cv2.erode(
            foreground_u8,
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        )
    pixels = rgb[foreground_u8 > 0]
    if len(pixels) < 64:
        return {
            "available": False,
            "reason": "insufficient_foreground_pixels",
            "mask_method": mask_method,
            "foreground_fraction": round(
                len(pixels) / float(max(1, foreground_u8.size)),
                6,
            ),
        }
    if len(pixels) > 250_000:
        stride = max(1, len(pixels) // 250_000)
        pixels = pixels[::stride]
    hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    hue = hsv[:, 0]
    saturation = hsv[:, 1]
    value = hsv[:, 2]
    category_masks: dict[str, np.ndarray] = {}
    category_masks["black"] = value < 58
    category_masks["white"] = (value >= 190) & (saturation < 52)
    category_masks["gray"] = ~(category_masks["black"] | category_masks["white"]) & (
        saturation < 52
    )
    chromatic = ~(
        category_masks["black"] | category_masks["white"] | category_masks["gray"]
    )
    category_masks["red_or_pink"] = chromatic & ((hue < 8) | (hue >= 165))
    category_masks["warm_brown_or_orange"] = chromatic & (hue >= 8) & (hue < 28)
    category_masks["yellow_or_gold"] = chromatic & (hue >= 28) & (hue < 42)
    category_masks["green"] = chromatic & (hue >= 42) & (hue < 88)
    category_masks["blue_or_cyan"] = chromatic & (hue >= 88) & (hue < 135)
    category_masks["purple"] = chromatic & (hue >= 135) & (hue < 165)
    total = float(len(pixels))
    fractions = {
        category: round(
            float(np.count_nonzero(category_masks[category])) / total,
            6,
        )
        for category in IDENTITY_PALETTE_CATEGORIES
    }
    dominant = [
        category
        for category in IDENTITY_PALETTE_CATEGORIES
        if fractions[category] >= 0.10
    ]
    if not dominant:
        dominant = sorted(
            IDENTITY_PALETTE_CATEGORIES,
            key=lambda category: fractions[category],
            reverse=True,
        )[:2]
    achromatic_fraction = sum(
        fractions[category] for category in ("black", "white", "gray")
    )
    chromatic_fraction = sum(
        fractions[category] for category in IDENTITY_PALETTE_CHROMATIC
    )
    warm_fraction = sum(
        fractions[category]
        for category in (
            "warm_brown_or_orange",
            "red_or_pink",
            "yellow_or_gold",
        )
    )
    warm_mask = (
        category_masks["warm_brown_or_orange"]
        | category_masks["red_or_pink"]
        | category_masks["yellow_or_gold"]
    )
    warm_saturation = saturation[warm_mask]
    warm_saturation_mean = (
        float(np.mean(warm_saturation)) / 255.0 if len(warm_saturation) else 0.0
    )
    warm_saturation_p90 = (
        float(np.percentile(warm_saturation, 90)) / 255.0
        if len(warm_saturation)
        else 0.0
    )
    saturated_warm_fraction = float(
        np.count_nonzero(warm_mask & (saturation >= 96))
    ) / float(len(pixels))
    return {
        "available": True,
        "mask_method": mask_method,
        "exclude_bottom_fraction": round(exclude_bottom_fraction, 6),
        "foreground_fraction": round(
            float(np.count_nonzero(foreground_u8)) / float(foreground_u8.size),
            6,
        ),
        "sample_count": len(pixels),
        "fractions": fractions,
        "dominant": dominant,
        "achromatic_fraction": round(achromatic_fraction, 6),
        "chromatic_fraction": round(chromatic_fraction, 6),
        "warm_fraction": round(warm_fraction, 6),
        "warm_saturation_mean": round(warm_saturation_mean, 6),
        "warm_saturation_p90": round(warm_saturation_p90, 6),
        "saturated_warm_fraction": round(saturated_warm_fraction, 6),
    }


def identity_palette_prompt(
    signature: dict[str, Any],
    *,
    retry: bool = False,
) -> str:
    if not signature.get("available"):
        return (
            "SOURCE COAT COLOR LOCK: copy every visible fur color and marking region "
            "from the uploaded cat. Never recolor the coat or borrow a palette from "
            "a LoRA training example."
        )
    fractions = signature["fractions"]
    dominant = list(signature.get("dominant") or [])
    labels = [IDENTITY_PALETTE_PROMPT_LABELS[item] for item in dominant]
    palette = ", ".join(labels)
    absent_chromatic = [
        IDENTITY_PALETTE_PROMPT_LABELS[item]
        for item in IDENTITY_PALETTE_CHROMATIC
        if float(fractions.get(item, 0.0)) <= 0.025
    ]
    source_is_achromatic = (
        float(signature["achromatic_fraction"]) >= 0.72
        and float(signature.get("warm_fraction", 0.0)) <= 0.12
    )
    if source_is_achromatic:
        achromatic_dominant = [
            item for item in dominant if item in {"black", "white", "gray"}
        ]
        achromatic_palette = (
            ", ".join(
                IDENTITY_PALETTE_PROMPT_LABELS[item] for item in achromatic_dominant
            )
            or "black, white, and neutral gray"
        )
        rule = (
            f"The visible source coat is achromatic and predominantly "
            f"{achromatic_palette}. Use neutral-white character lighting with no amber, "
            "sepia, orange, ginger, brown, or golden color cast over neutral fur. "
            "Any genuinely warm source color may remain only in the same localized "
            "marking regions and proportion; it must never spread across gray, black, "
            "or white fur."
        )
    else:
        rule = f"The visible source coat palette is predominantly {palette}."
    if absent_chromatic:
        rule += (
            " Do not introduce these absent dominant fur hues: "
            + ", ".join(absent_chromatic)
            + "."
        )
    retry_rule = (
        " Previous output recolored the cat and was rejected; restore the source "
        "palette exactly before changing pose or rendering."
        if retry
        else ""
    )
    return (
        "SOURCE COAT PALETTE LOCK - higher priority than style and LoRA: "
        f"{rule} Preserve the exact spatial placement and proportion of every source "
        "color region. Never add calico, tortoiseshell, tabby, stripe, spot, patch, "
        f"or warm highlight that is absent from the uploaded cat.{retry_rule}"
    )


def identity_palette_negative_prompt(signature: dict[str, Any]) -> str:
    negative = [
        "recolored coat",
        "new dominant fur color",
        "LoRA sample coat",
        "changed coat palette",
    ]
    if not signature.get("available"):
        return ", ".join(negative)
    fractions = signature["fractions"]
    if (
        float(signature.get("achromatic_fraction", 0.0)) >= 0.72
        and float(signature.get("warm_fraction", 0.0)) <= 0.12
    ):
        negative.extend(
            [
                "overall warm color cast",
                "amber lighting on fur",
                "sepia fur",
                "neutral gray fur turned orange",
                "neutral gray fur turned brown",
                "ginger recolor",
            ]
        )
    if float(fractions.get("warm_brown_or_orange", 0.0)) <= 0.025:
        negative.extend(
            [
                "orange fur",
                "ginger fur",
                "brown fur",
                "tan fur",
                "calico coat",
                "tortoiseshell coat",
            ]
        )
    if float(fractions.get("yellow_or_gold", 0.0)) <= 0.025:
        negative.append("golden fur")
    return ", ".join(negative)


def identity_palette_quality(
    source: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    if not source.get("available") or not candidate.get("available"):
        return {
            "available": False,
            "acceptable": True,
            "warning": True,
            "issues": ["identity_palette_signature_unavailable"],
            "penalty": 0.0,
            "source": source,
            "candidate": candidate,
        }
    source_fractions = source["fractions"]
    candidate_fractions = candidate["fractions"]
    issues: list[str] = []
    warnings: list[str] = []
    deltas = {
        category: round(
            float(candidate_fractions.get(category, 0.0))
            - float(source_fractions.get(category, 0.0)),
            6,
        )
        for category in IDENTITY_PALETTE_CATEGORIES
    }
    for category in IDENTITY_PALETTE_CHROMATIC:
        source_fraction = float(source_fractions.get(category, 0.0))
        candidate_fraction = float(candidate_fractions.get(category, 0.0))
        candidate_threshold = 0.09 if category == "warm_brown_or_orange" else 0.14
        delta_threshold = 0.07 if category == "warm_brown_or_orange" else 0.10
        if (
            source_fraction <= 0.04
            and candidate_fraction >= candidate_threshold
            and candidate_fraction - source_fraction >= delta_threshold
        ):
            issues.append(f"new_dominant_{category}")
    source_achromatic = float(source["achromatic_fraction"])
    candidate_achromatic = float(candidate["achromatic_fraction"])
    source_warm = float(source["warm_fraction"])
    candidate_warm = float(candidate["warm_fraction"])
    source_saturated_warm = float(source.get("saturated_warm_fraction", 0.0))
    candidate_saturated_warm = float(candidate.get("saturated_warm_fraction", 0.0))
    broad_warm_drift = bool(
        source_achromatic - candidate_achromatic >= 0.12
        and candidate_warm - source_warm >= 0.12
    )
    saturated_warm_drift = bool(
        candidate_saturated_warm >= 0.055
        and candidate_saturated_warm - source_saturated_warm >= 0.035
    )
    if (
        source_achromatic >= 0.72
        and source_warm <= 0.12
        and (broad_warm_drift or saturated_warm_drift)
    ):
        issues.append("achromatic_source_recolored_with_warm_fur")
    if source_warm >= 0.18 and candidate_warm <= 0.06:
        warnings.append("source_warm_coat_color_may_be_hidden")
    l1_distance = sum(abs(value) for value in deltas.values())
    penalty = l1_distance * 50.0 + len(issues) * 100.0 + len(warnings) * 25.0
    return {
        "available": True,
        "acceptable": not issues,
        "warning": bool(issues or warnings),
        "issues": issues,
        "warnings": warnings,
        "penalty": round(penalty, 6),
        "deltas": deltas,
        "source_dominant": source.get("dominant", []),
        "candidate_dominant": candidate.get("dominant", []),
        "source_achromatic_fraction": source["achromatic_fraction"],
        "candidate_achromatic_fraction": candidate["achromatic_fraction"],
        "source_warm_fraction": source["warm_fraction"],
        "candidate_warm_fraction": candidate["warm_fraction"],
        "source_saturated_warm_fraction": round(source_saturated_warm, 6),
        "candidate_saturated_warm_fraction": round(candidate_saturated_warm, 6),
        "broad_warm_drift": broad_warm_drift,
        "saturated_warm_drift": saturated_warm_drift,
    }


def bowl_aware_identity_palette_quality(
    primary_quality: dict[str, Any],
    upper_body_quality: dict[str, Any] | None,
) -> dict[str, Any]:
    """Prevent a touching bowl and kibble from being audited as the cat's coat."""
    if not isinstance(upper_body_quality, dict):
        return primary_quality
    if upper_body_quality.get("available") and not upper_body_quality.get("acceptable"):
        merged_issues = list(
            dict.fromkeys(
                [
                    *(primary_quality.get("issues") or []),
                    *(upper_body_quality.get("issues") or []),
                ]
            )
        )
        merged_warnings = list(
            dict.fromkeys(
                [
                    *(primary_quality.get("warnings") or []),
                    *(upper_body_quality.get("warnings") or []),
                ]
            )
        )
        return {
            **upper_body_quality,
            "acceptable": False,
            "warning": True,
            "issues": merged_issues,
            "warnings": merged_warnings,
            "penalty": round(
                max(
                    float(primary_quality.get("penalty", 0.0)),
                    float(upper_body_quality.get("penalty", 0.0)),
                ),
                6,
            ),
            "method": "bowl_upper_body_confirmation",
            "primary_quality": primary_quality,
            "upper_body_quality": upper_body_quality,
        }
    if primary_quality.get("acceptable"):
        return primary_quality
    bowl_contamination_issues = {
        "achromatic_source_recolored_with_warm_fur",
        "new_dominant_warm_brown_or_orange",
        "new_dominant_yellow_or_gold",
        "new_dominant_red_or_pink",
    }
    primary_issues = set(primary_quality.get("issues") or [])
    if not primary_issues or not primary_issues.issubset(bowl_contamination_issues):
        return primary_quality
    if not upper_body_quality.get("available"):
        return primary_quality
    if not upper_body_quality.get("acceptable"):
        return {
            **primary_quality,
            "method": "bowl_upper_body_confirmation",
            "primary_quality": primary_quality,
            "upper_body_quality": upper_body_quality,
        }
    warnings = list(upper_body_quality.get("warnings") or [])
    warnings.append("bowl_prop_palette_contamination_ignored")
    return {
        **upper_body_quality,
        "acceptable": True,
        "warning": True,
        "issues": [],
        "warnings": list(dict.fromkeys(warnings)),
        "penalty": round(float(upper_body_quality.get("penalty", 0.0)) + 25.0, 6),
        "method": "bowl_upper_body_confirmation",
        "primary_quality": primary_quality,
        "upper_body_quality": upper_body_quality,
    }


def identity_palette_audit(
    guard_enabled: bool,
    quality: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return an explicit audit verdict without treating unchecked output as safe."""
    if not guard_enabled:
        return {
            "identity_palette_checked": False,
            "identity_palette_verdict": "disabled",
            "identity_palette_preserved": None,
        }
    if not isinstance(quality, dict) or not quality.get("available"):
        return {
            "identity_palette_checked": True,
            "identity_palette_verdict": "unavailable",
            "identity_palette_preserved": None,
        }
    preserved = bool(quality.get("acceptable"))
    return {
        "identity_palette_checked": True,
        "identity_palette_verdict": "passed" if preserved else "failed",
        "identity_palette_preserved": preserved,
    }


def canonical_fallback_score(record: dict[str, Any]) -> float:
    """Rank canonical retries while making an identity failure prohibitively costly."""
    palette = record.get("identity_palette")
    palette_penalty = (
        float(palette.get("penalty", 0.0)) if isinstance(palette, dict) else 0.0
    )
    palette_gate = (
        0.0
        if not isinstance(palette, dict) or palette.get("acceptable", True)
        else 10_000.0
    )
    quality = record.get("quality")
    quality_penalty = (
        float(quality.get("penalty", 100.0)) if isinstance(quality, dict) else 100.0
    )
    chroma = record.get("chroma")
    chroma_penalty = (
        0.0 if isinstance(chroma, dict) and chroma.get("acceptable") else 50.0
    )
    return palette_gate + palette_penalty * 10.0 + quality_penalty + chroma_penalty


def select_canonical_fallback_attempt(
    attempts: list[dict[str, Any]],
    *,
    identity_palette_required: bool,
) -> dict[str, Any]:
    """Select the best usable retry and fail closed when every coat palette drifted."""
    if not attempts:
        raise RuntimeError("Canonical generation produced no fallback candidate.")
    identity_safe = [
        record
        for record in attempts
        if not isinstance(record.get("identity_palette"), dict)
        or record["identity_palette"].get("acceptable", True)
    ]
    if identity_palette_required and not identity_safe:
        issues = list(
            dict.fromkeys(
                issue
                for record in attempts
                for issue in (
                    record.get("identity_palette", {}).get("issues", [])
                    if isinstance(record.get("identity_palette"), dict)
                    else []
                )
            )
        )
        raise RuntimeError(
            "Canonical identity palette drift remained after "
            f"{len(attempts)} attempts: "
            + (", ".join(issues) or "no identity-safe candidate")
        )
    return min(
        identity_safe or attempts,
        key=canonical_fallback_score,
    )


def semantic_background_recovery(
    source: Path,
    chroma_target: Path,
    cutout_target: Path,
    semantic_target: Path,
    pose: str = "",
) -> dict[str, Any]:
    from rembg import remove

    with Image.open(source) as loaded:
        source_size = loaded.size
        image = loaded.convert("RGBA")
    alpha_matting = True
    try:
        recovered = remove(
            image,
            session=get_rembg_session(),
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=10,
        )
    except Exception:  # noqa: BLE001 - alpha matting may fail on difficult masks
        LOGGER.warning(
            "Alpha matting failed for %s; retrying semantic removal without matting",
            source.name,
            exc_info=True,
        )
        alpha_matting = False
        recovered = remove(image, session=get_rembg_session())
    if not isinstance(recovered, Image.Image):
        recovered = Image.open(io.BytesIO(recovered)).convert("RGBA")
    else:
        recovered = recovered.convert("RGBA")
    semantic_target.parent.mkdir(parents=True, exist_ok=True)
    recovered.save(semantic_target, format="PNG", optimize=True)
    alpha = np.asarray(recovered.getchannel("A"))
    semantic_foreground_fraction = float(np.count_nonzero(alpha >= 64)) / float(
        alpha.size
    )
    if semantic_foreground_fraction < 0.02:
        raise RuntimeError(
            "Semantic background recovery did not find a usable pet subject."
        )
    semantic_quality = pose_cutout_quality(semantic_target, pose)
    if not semantic_quality["acceptable"]:
        raise RuntimeError(
            "Semantic background recovery produced an incomplete or invalid pet subject: "
            + ", ".join(semantic_quality["issues"])
        )
    prepare_landscape_keyframe(
        semantic_target,
        chroma_target,
        source_size,
        background_rgb=(0, 255, 0),
    )
    metrics = evaluate_chroma_cutout(chroma_target, cutout_target)
    metrics.update(
        {
            "method": "rembg_u2net_fallback",
            "model": CHROMA_FALLBACK_REMBG_MODEL,
            "alpha_matting": alpha_matting,
            "semantic_foreground_fraction": semantic_foreground_fraction,
            "semantic_cutout": semantic_target.name,
            "semantic_quality": semantic_quality,
        }
    )
    if not metrics["acceptable"]:
        raise RuntimeError(
            "Semantic background recovery could not create a clean green canvas."
        )
    return metrics


def semantic_recovery_candidate_eligible(record: dict[str, Any]) -> bool:
    """Only use rembg when chroma failed; it cannot repair anatomy or pose quality."""
    chroma = record.get("chroma")
    similarity = record.get("similarity")
    palette = record.get("identity_palette")
    return bool(
        isinstance(chroma, dict)
        and not chroma.get("acceptable", False)
        and isinstance(similarity, dict)
        and similarity.get("acceptable", False)
        and (
            palette is None
            or (isinstance(palette, dict) and palette.get("acceptable", False))
        )
    )


def pose_attempt_failure_issues(attempts: list[dict[str, Any]]) -> list[str]:
    issues: list[str] = []
    for failed_attempt in attempts:
        for section_name in ("chroma", "quality", "similarity", "identity_palette"):
            section = failed_attempt.get(section_name)
            if isinstance(section, dict) and isinstance(section.get("issues"), list):
                issues.extend(str(issue) for issue in section["issues"])
    return list(dict.fromkeys(issues))


def pose_attempt_failure_summary(attempts: list[dict[str, Any]]) -> str:
    """Separate persistent blockers from one-off retry failures."""
    if not attempts:
        return "no usable subject"
    ordered_issues = pose_attempt_failure_issues(attempts)
    issues_by_attempt: list[tuple[int, list[str]]] = []
    for fallback_index, failed_attempt in enumerate(attempts, start=1):
        attempt_issues: list[str] = []
        for section_name in ("chroma", "quality", "similarity", "identity_palette"):
            section = failed_attempt.get(section_name)
            if isinstance(section, dict) and isinstance(section.get("issues"), list):
                attempt_issues.extend(str(issue) for issue in section["issues"])
        issues_by_attempt.append(
            (
                int(failed_attempt.get("attempt") or fallback_index),
                list(dict.fromkeys(attempt_issues)),
            )
        )
    persistent = [
        issue
        for issue in ordered_issues
        if all(issue in attempt_issues for _, attempt_issues in issues_by_attempt)
    ]
    transient_parts = [
        f"attempt {attempt}: {', '.join(issue for issue in attempt_issues if issue not in persistent)}"
        for attempt, attempt_issues in issues_by_attempt
        if any(issue not in persistent for issue in attempt_issues)
    ]
    parts: list[str] = []
    if persistent:
        parts.append("persistent: " + ", ".join(persistent))
    if transient_parts:
        parts.append("transient: " + "; ".join(transient_parts))
    return "; ".join(parts) or "no usable subject"


def pose_cutout_quality(path: Path, pose: str) -> dict[str, Any]:
    rgba = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if rgba is None or rgba.ndim != 3 or rgba.shape[2] < 4:
        raise RuntimeError(f"Could not inspect transparent pose output {path.name}.")
    alpha = rgba[:, :, 3]
    foreground = (alpha >= 64).astype(np.uint8)
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(
        foreground, connectivity=8
    )
    areas = sorted(
        (int(stats[label, cv2.CC_STAT_AREA]) for label in range(1, component_count)),
        reverse=True,
    )
    total_area = sum(areas)
    if not areas or total_area <= 0:
        return {
            "acceptable": False,
            "issues": ["no_foreground_subject"],
            "penalty": 1000.0,
            "component_count": 0,
        }
    meaningful_minimum = max(48, int(total_area * 0.001))
    meaningful_areas = [area for area in areas if area >= meaningful_minimum]
    minor_fraction = float(sum(meaningful_areas[1:])) / float(total_area)
    ys, xs = np.where(foreground > 0)
    height, width = foreground.shape
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    margins = {
        "left": bbox[0] / width,
        "top": bbox[1] / height,
        "right": (width - bbox[2]) / width,
        "bottom": (height - bbox[3]) / height,
    }
    issues: list[str] = []
    if min(margins.values()) < 0.025:
        issues.append("subject_too_close_to_canvas_edge")
    action = action_key(pose)
    if action in {"running", "lying", "lying down"} and minor_fraction > 0.003:
        issues.append("disconnected_body_or_paw_fragment")
    foreground_fraction = total_area / float(width * height)
    if foreground_fraction < 0.035:
        issues.append("subject_too_small")
    if foreground_fraction > 0.72:
        issues.append("subject_too_large")
    bbox_width = max(1, bbox[2] - bbox[0])
    bbox_height = max(1, bbox[3] - bbox[1])
    bbox_aspect_ratio = bbox_width / float(bbox_height)
    if action in {"eating", "drinking"} and bbox_aspect_ratio < 0.78:
        issues.append("pose_not_low_or_wide_enough_for_bowl_action")
    if action in {"lying", "lying down"} and bbox_aspect_ratio < 0.90:
        issues.append("pose_not_low_or_wide_enough_for_lying")
    lower_body_component_peak = 0
    lower_body_components: dict[str, int] = {}
    if action == "running":
        y0, y1 = bbox[1], bbox[3]
        minimum_lower_area = max(36, int(total_area * 0.0005))
        for fraction in (0.72, 0.76, 0.80, 0.84, 0.88, 0.90):
            cutoff = y0 + int((y1 - y0) * fraction)
            lower_mask = foreground.copy()
            lower_mask[:cutoff, :] = 0
            lower_count, _, lower_stats, _ = cv2.connectedComponentsWithStats(
                lower_mask,
                connectivity=8,
            )
            count = sum(
                int(lower_stats[label, cv2.CC_STAT_AREA]) >= minimum_lower_area
                for label in range(1, lower_count)
            )
            lower_body_components[f"{fraction:.2f}"] = count
            lower_body_component_peak = max(lower_body_component_peak, count)
        if lower_body_component_peak < 3:
            issues.append("fewer_than_three_distinct_grounded_paws")
    penalty = minor_fraction * 100.0
    penalty += max(0.0, 0.025 - min(margins.values())) * 200.0
    penalty += 25.0 * len(issues)
    return {
        "acceptable": not issues,
        "issues": issues,
        "penalty": round(penalty, 6),
        "component_count": len(meaningful_areas),
        "component_areas": meaningful_areas,
        "minor_component_fraction": round(minor_fraction, 6),
        "foreground_fraction": round(foreground_fraction, 6),
        "bbox": bbox,
        "bbox_aspect_ratio": round(bbox_aspect_ratio, 6),
        "margins": {key: round(value, 6) for key, value in margins.items()},
        "lower_body_component_peak": lower_body_component_peak,
        "lower_body_components": lower_body_components,
    }


def pose_similarity_signature(path: Path, size: int = 192) -> dict[str, Any]:
    """Normalize a transparent or green-screen subject for pose-only comparison."""
    with Image.open(path) as loaded:
        rgba = np.asarray(loaded.convert("RGBA"))
    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3]
    if np.any(alpha < 250):
        foreground = alpha >= 64
        mask_method = "alpha"
    else:
        red = rgb[:, :, 0].astype(np.int16)
        green = rgb[:, :, 1].astype(np.int16)
        blue = rgb[:, :, 2].astype(np.int16)
        green_background = (green > 80) & ((green - np.maximum(red, blue)) > 35)
        if (
            float(np.count_nonzero(green_background)) / float(green_background.size)
            >= 0.05
        ):
            foreground = ~green_background
            mask_method = "green_screen"
        else:
            corners = np.asarray(
                [
                    rgb[0, 0],
                    rgb[0, -1],
                    rgb[-1, 0],
                    rgb[-1, -1],
                ],
                dtype=np.float32,
            )
            background = np.median(corners, axis=0)
            distance = np.linalg.norm(rgb.astype(np.float32) - background, axis=2)
            foreground = distance >= 24.0
            mask_method = "corner_color"
    foreground_u8 = foreground.astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        foreground_u8,
        connectivity=8,
    )
    areas = [int(stats[label, cv2.CC_STAT_AREA]) for label in range(1, component_count)]
    total_area = sum(areas)
    if total_area <= 0:
        return {
            "available": False,
            "reason": "no_foreground_subject",
            "path": path.name,
            "mask_method": mask_method,
        }
    meaningful_minimum = max(48, int(total_area * 0.001))
    meaningful_labels = {
        label
        for label in range(1, component_count)
        if int(stats[label, cv2.CC_STAT_AREA]) >= meaningful_minimum
    }
    foreground = np.isin(labels, list(meaningful_labels))
    ys, xs = np.where(foreground)
    if not len(xs) or not len(ys):
        return {
            "available": False,
            "reason": "no_meaningful_foreground_subject",
            "path": path.name,
            "mask_method": mask_method,
        }
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    cropped_mask = (
        foreground[bbox[1] : bbox[3], bbox[0] : bbox[2]].astype(np.uint8) * 255
    )
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cropped_gray = gray[bbox[1] : bbox[3], bbox[0] : bbox[2]]
    cropped_gray = cv2.bitwise_and(cropped_gray, cropped_gray, mask=cropped_mask)
    available_size = max(32, size - 16)
    scale = min(
        available_size / max(1, cropped_mask.shape[1]),
        available_size / max(1, cropped_mask.shape[0]),
    )
    resized_width = max(1, int(round(cropped_mask.shape[1] * scale)))
    resized_height = max(1, int(round(cropped_mask.shape[0] * scale)))
    resized_mask = cv2.resize(
        cropped_mask,
        (resized_width, resized_height),
        interpolation=cv2.INTER_NEAREST,
    )
    resized_gray = cv2.resize(
        cropped_gray,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )
    normalized_mask = np.zeros((size, size), dtype=np.uint8)
    normalized_gray = np.zeros((size, size), dtype=np.uint8)
    x_offset = (size - resized_width) // 2
    y_offset = (size - resized_height) // 2
    normalized_mask[
        y_offset : y_offset + resized_height,
        x_offset : x_offset + resized_width,
    ] = resized_mask
    normalized_gray[
        y_offset : y_offset + resized_height,
        x_offset : x_offset + resized_width,
    ] = resized_gray
    dhash_source = cv2.resize(normalized_gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (dhash_source[:, 1:] > dhash_source[:, :-1]).flatten()
    dhash = 0
    for bit in bits:
        dhash = (dhash << 1) | int(bit)
    return {
        "available": True,
        "path": path.name,
        "mask_method": mask_method,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "mask": normalized_mask >= 64,
        "dhash": dhash,
        "bbox_aspect_ratio": round(
            (bbox[2] - bbox[0]) / float(max(1, bbox[3] - bbox[1])),
            6,
        ),
    }


def compare_pose_signatures(
    candidate: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any]:
    if not candidate.get("available") or not reference.get("available"):
        return {
            "available": False,
            "reason": candidate.get("reason")
            or reference.get("reason")
            or "signature_unavailable",
        }
    candidate_mask = candidate["mask"]
    reference_mask = reference["mask"]
    intersection = int(np.count_nonzero(candidate_mask & reference_mask))
    union = int(np.count_nonzero(candidate_mask | reference_mask))
    iou = intersection / float(union) if union else 0.0
    dhash_distance = int(
        (int(candidate["dhash"]) ^ int(reference["dhash"])).bit_count()
    )
    return {
        "available": True,
        "sha256_equal": candidate["sha256"] == reference["sha256"],
        "silhouette_iou": round(iou, 6),
        "dhash_distance": dhash_distance,
        "candidate_aspect_ratio": candidate["bbox_aspect_ratio"],
        "reference_aspect_ratio": reference["bbox_aspect_ratio"],
    }


def pose_similarity_quality(
    candidate_path: Path,
    *,
    pose: str,
    canonical_path: Path | None,
    previous_poses: list[dict[str, Any]],
) -> dict[str, Any]:
    """Reject a pose edit that merely reconstructs the reference or an earlier pose."""
    candidate = pose_similarity_signature(candidate_path)
    if not candidate.get("available"):
        return {
            "available": False,
            "acceptable": False,
            "warning": True,
            "issues": ["candidate_similarity_signature_unavailable"],
            "comparisons": [],
            "penalty": 1000.0,
        }
    references: list[dict[str, Any]] = []
    if canonical_path and canonical_path.is_file():
        references.append(
            {
                "label": "canonical",
                "pose": "canonical",
                "pose_index": None,
                "path": canonical_path,
            }
        )
    references.extend(previous_poses)
    comparisons: list[dict[str, Any]] = []
    issues: list[str] = []
    unavailable_references: list[str] = []
    maximum_penalty = 0.0
    current_action = action_key(pose)
    for reference_info in references:
        reference_path = Path(reference_info["path"])
        reference = pose_similarity_signature(reference_path)
        comparison = compare_pose_signatures(candidate, reference)
        comparison.update(
            {
                "label": reference_info["label"],
                "pose": reference_info["pose"],
                "pose_index": reference_info.get("pose_index"),
                "filename": reference_path.name,
            }
        )
        if not comparison.get("available"):
            comparisons.append(comparison)
            unavailable_references.append(str(reference_info["label"]))
            continue
        reference_action = action_key(str(reference_info["pose"]))
        related_pair = (
            reference_info["label"] == "canonical" and current_action == "running"
        ) or {current_action, reference_action} == {"eating", "drinking"}
        iou_threshold = (
            QWEN_POSE_RELATED_IOU_THRESHOLD
            if related_pair
            else QWEN_POSE_SIMILARITY_IOU_THRESHOLD
        )
        dhash_threshold = (
            QWEN_POSE_RELATED_DHASH_THRESHOLD
            if related_pair
            else QWEN_POSE_SIMILARITY_DHASH_THRESHOLD
        )
        hard_duplicate = bool(
            comparison["sha256_equal"]
            or (
                comparison["silhouette_iou"] >= QWEN_POSE_HARD_DUPLICATE_IOU
                and comparison["dhash_distance"] <= QWEN_POSE_HARD_DUPLICATE_DHASH
            )
        )
        too_similar = bool(
            comparison["silhouette_iou"] >= iou_threshold
            and comparison["dhash_distance"] <= dhash_threshold
        )
        comparison.update(
            {
                "related_pose_threshold": related_pair,
                "iou_threshold": iou_threshold,
                "dhash_threshold": dhash_threshold,
                "hard_duplicate": hard_duplicate,
                "too_similar": too_similar,
            }
        )
        if hard_duplicate:
            issues.append(f"hard_duplicate_of_{reference_info['label']}")
        elif too_similar:
            issues.append(f"pose_too_similar_to_{reference_info['label']}")
        maximum_penalty = max(
            maximum_penalty,
            comparison["silhouette_iou"] * 100.0
            - min(comparison["dhash_distance"], 32) * 0.5,
        )
        comparisons.append(comparison)
    for label in unavailable_references:
        issues.append(f"reference_similarity_signature_unavailable_{label}")
    if unavailable_references:
        maximum_penalty = max(maximum_penalty, 1000.0)
    return {
        "available": not unavailable_references,
        "acceptable": not issues,
        "warning": bool(issues or unavailable_references),
        "issues": list(dict.fromkeys(issues)),
        "comparisons": comparisons,
        "penalty": round(maximum_penalty, 6),
    }


def convert_chroma_video_to_transparent_animations(
    source: Path,
    gif_target: Path,
    apng_target: Path,
) -> dict[str, Any]:
    return build_transparent_animations(
        source,
        gif_target,
        apng_target,
        ffmpeg_bin=FFMPEG_BIN,
        size=TRANSPARENT_GIF_SIZE,
        fps=TRANSPARENT_GIF_FPS,
        frame_count=TRANSPARENT_GIF_FRAMES,
        duration_seconds=TRANSPARENT_GIF_DURATION_SECONDS,
    )


def run_fun_control_video(
    *,
    job_dir: Path,
    pose_index: int,
    reference_image: Path,
    prompt: str,
    negative_prompt: str,
    seed: int,
) -> tuple[Path, dict[str, Any]]:
    run_id = uuid4().hex
    output_dir = job_dir / "fun_control" / f"pose_{pose_index:02d}" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    log_path = job_dir / f"pose_{pose_index:02d}_fun_control.log"
    metadata_path = job_dir / "workflows" / f"fun_pose_{pose_index:02d}.json"
    metadata = {
        "engine": "wan2.2-fun-5b-control",
        "model_root": str(FUN_CONTROL_MODEL_ROOT),
        "control_video": str(FUN_CONTROL_RUNNING_VIDEO),
        "reference_image": reference_image.name,
        "resolution": [512, 512],
        "frames": FRAME_COUNT,
        "fps": FPS,
        "steps": FUN_CONTROL_STEPS,
        "guidance_scale": 6.0,
        "seed": seed,
        "gpu_memory_mode": FUN_CONTROL_GPU_MEMORY_MODE,
        "teacache": False,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
    }
    write_json_atomic(metadata_path, metadata)

    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH", "").strip()
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(FUN_CONTROL_PROJECT_ROOT), existing_pythonpath) if part
    )
    environment.update(
        {
            "PYTHONUNBUFFERED": "1",
            "VIDEO_LENGTH": str(FRAME_COUNT),
            "INFERENCE_STEPS": str(FUN_CONTROL_STEPS),
            "SAVE_PATH": str(output_dir),
            "GPU_MEMORY_MODE": FUN_CONTROL_GPU_MEMORY_MODE,
            "ENABLE_TEACACHE": "0",
            "CONTROL_VIDEO": str(FUN_CONTROL_RUNNING_VIDEO),
            "REF_IMAGE": str(reference_image),
            "PROMPT": prompt,
            "NEGATIVE_PROMPT": negative_prompt,
            "GUIDANCE_SCALE": "6.0",
            "SEED": str(seed),
        }
    )
    started = time.monotonic()
    process = subprocess.run(
        [str(FUN_CONTROL_PYTHON), str(FUN_CONTROL_SCRIPT)],
        cwd=str(FUN_CONTROL_PROJECT_ROOT),
        env=environment,
        capture_output=True,
        text=True,
        timeout=FUN_CONTROL_TIMEOUT,
        check=False,
    )
    elapsed = round(time.monotonic() - started, 3)
    log_path.write_text(
        (process.stdout or "") + "\n--- STDERR ---\n" + (process.stderr or ""),
        encoding="utf-8",
    )
    if process.returncode != 0:
        detail = (
            process.stderr or process.stdout or "unknown VideoX-Fun error"
        ).strip()
        raise RuntimeError(
            f"Wan2.2-Fun-5B-Control exited with {process.returncode}: {detail[-4000:]}"
        )
    videos = sorted(output_dir.glob("*.mp4"), key=lambda path: path.stat().st_mtime)
    if not videos:
        raise FileNotFoundError("Wan2.2-Fun-5B-Control did not produce an MP4 output.")
    metadata.update({"elapsed_seconds": elapsed, "log": log_path.name})
    write_json_atomic(metadata_path, metadata)
    return videos[-1], metadata


def write_workflow(job_dir: Path, filename: str, workflow: dict[str, Any]) -> None:
    write_json_atomic(job_dir / "workflows" / filename, workflow)


def prepare_pose_input(
    job_id: str,
    job_dir: Path,
    pose_index: int,
    pose: str,
    keyframe: Path,
    width: int,
    height: int,
    transparent: bool,
) -> Path:
    wan_input = job_dir / f"pose_{pose_index:02d}_wan_input.png"
    background_rgb = (0, 255, 0) if transparent else (255, 255, 255)
    prepare_landscape_keyframe(
        keyframe, wan_input, (width, height), background_rgb=background_rgb
    )
    if transparent:
        transparent_keyframe = (
            job_dir / f"pose_{pose_index:02d}_keyframe_transparent.png"
        )
        convert_chroma_image_to_transparent_png(
            wan_input,
            transparent_keyframe,
            require_clean_background=True,
        )
        append_output(
            job_id,
            {
                "type": "transparent_keyframe",
                "pose_index": pose_index,
                "pose": pose,
                "filename": transparent_keyframe.name,
                "width": TRANSPARENT_GIF_SIZE,
                "height": TRANSPARENT_GIF_SIZE,
                "transparent": True,
                "visual_style": "identity_preserving_cute_3d_studio",
                "source_role": "cartoon_animation_keyframe",
            },
        )
    return wan_input


def run_job(job_id: str) -> None:
    raise_if_cancelled(job_id)
    with JOBS_LOCK:
        job = dict(JOBS[job_id])
    job_dir = OUTPUT_ROOT / job_id
    poses: list[str] = list(job.get("poses_en") or job["poses"])
    effective_prompt = str(job.get("prompt_en") or job["prompt"])
    width, height = RESOLUTIONS[job["resolution"]]
    transparent_gif = job.get("output_format", "mp4") == "transparent_gif"
    skip_video = bool(job.get("skip_video", False))
    request_kind = str(job.get("request_kind") or "video").strip().lower()
    image_request = request_kind == "image"
    suppress_builtin_style = bool(
        job.get("image_profile_lora_source")
        in {"image_profile", "image_profile_plus_user"}
    )
    user_loras = normalize_user_lora_specs(
        job.get("user_loras"),
        legacy_model_name=job.get("user_lora"),
        legacy_original_filename=job.get("user_lora_original_filename"),
        legacy_strength=(
            job["user_lora_strength"]
            if job.get("user_lora_strength") is not None
            else 1.0
        ),
    )
    canonical_user_loras, canonical_lora_scale = cap_user_lora_specs(
        user_loras,
        total_cap=float(
            job.get(
                "canonical_user_lora_total_cap",
                QWEN_CANONICAL_USER_LORA_TOTAL_CAP,
            )
        ),
        per_lora_cap=float(
            job.get(
                "canonical_user_lora_per_lora_cap",
                QWEN_CANONICAL_USER_LORA_PER_LORA_CAP,
            )
        ),
    )
    first_canonical_user_lora = (
        canonical_user_loras[0] if canonical_user_loras else None
    )
    user_lora = (
        first_canonical_user_lora["model_name"] if first_canonical_user_lora else None
    )
    user_lora_strength = (
        first_canonical_user_lora["strength"] if first_canonical_user_lora else 1.0
    )
    user_lora_mode = str(job.get("user_lora_mode") or "replace")
    qwen_steps = int(job.get("qwen_steps", 4))
    qwen_cfg = float(job.get("qwen_cfg", 1.0))
    qwen_denoise = float(job.get("qwen_denoise", 1.0))
    wan_steps = int(job.get("wan_steps", WAN_STEPS))
    wan_cfg = float(job.get("wan_cfg", WAN_CFG))
    wan_shift = float(job.get("wan_shift", WAN_SHIFT))
    base_seed = int(job["seed"])
    comfy_relative_dir = Path("local_video_api") / job_id
    comfy_input_dir = COMFY_INPUT / comfy_relative_dir
    comfy_input_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(job_dir / "input.png", comfy_input_dir / "input.png")

    update_job(
        job_id,
        status="running",
        stage="waiting_for_gpu",
        progress=1,
        started_at=utc_now(),
        error=None,
    )
    raise_if_cancelled(job_id)
    require_models(include_video=not skip_video)
    wait_comfy_idle(job_id=job_id)
    raise_if_cancelled(job_id)
    free_comfy_models()

    recovery_profiles = qwen_recovery_profiles(
        user_lora=user_lora,
        user_lora_strength=user_lora_strength,
        user_lora_mode=user_lora_mode,
        steps=qwen_steps,
        cfg=qwen_cfg,
        denoise=qwen_denoise,
        user_loras=canonical_user_loras,
        lock_strengths=bool(job.get("lock_retry_lora_strengths", False)),
    )
    pose_profiles = qwen_pose_profiles(
        user_lora=(user_loras[0]["model_name"] if user_loras else None),
        user_lora_strength=(float(user_loras[0]["strength"]) if user_loras else 1.0),
        user_lora_mode=user_lora_mode,
        steps=qwen_steps,
        cfg=qwen_cfg,
        denoise=qwen_denoise,
        user_loras=user_loras,
        first_total_cap=job.get("pose_user_lora_first_total_cap"),
        first_per_lora_cap=job.get("pose_user_lora_first_per_lora_cap"),
        lock_strengths=bool(job.get("lock_retry_lora_strengths", False)),
    )
    update_job(
        job_id,
        requested_user_loras=user_loras,
        effective_canonical_user_loras=recovery_profiles[0]["user_loras"],
        canonical_user_lora_strength_scale=canonical_lora_scale,
        initial_pose_user_loras=pose_profiles[0]["user_loras"],
        initial_pose_user_lora_strength_scale=pose_profiles[0][
            "user_lora_strength_scale"
        ],
        effective_pose_user_loras=pose_profiles[0]["user_loras"],
        pose_user_lora_strength_scale=pose_profiles[0]["user_lora_strength_scale"],
        effective_pose_user_loras_scope="initial_profile_only; use effective_pose_lora_attempts for selected per-pose settings",
        pose_lora_policy={
            "attempts": [
                {
                    "profile": profile["name"],
                    "total_strength_cap": profile["total_strength_cap"],
                    "per_lora_strength_cap": profile["per_lora_strength_cap"],
                    "strength_scale": profile["user_lora_strength_scale"],
                }
                for profile in pose_profiles
            ]
        },
        effective_pose_lora_attempts=[],
        pose_similarity_warning_count=0,
        pose_identity_palette_warning_count=0,
    )
    qwen_input_name = (comfy_relative_dir / "input.png").as_posix()
    identity_palette_guard = bool(job.get("identity_palette_guard", False))
    identity_source_path = job_dir / "input_identity_source.png"
    if not identity_source_path.is_file():
        identity_source_path = job_dir / "input.png"
    source_identity_palette = (
        identity_palette_signature(identity_source_path)
        if identity_palette_guard
        else None
    )
    source_palette_prompt = (
        identity_palette_prompt(source_identity_palette)
        if source_identity_palette is not None
        else ""
    )
    source_palette_negative = (
        identity_palette_negative_prompt(source_identity_palette)
        if source_identity_palette is not None
        else ""
    )
    if source_identity_palette is not None:
        update_job(
            job_id,
            source_identity_palette=source_identity_palette,
            identity_palette_prompt=source_palette_prompt,
        )
    canonical_similarity_reference: Path | None = None
    if transparent_gif:
        update_job(
            job_id,
            stage="generating_cartoon_reference",
            progress=3,
            current_pose_index=None,
            current_pose=None,
        )
        cartoon_negative_prompt = (
            QWEN_NEGATIVE + ", " + CARTOON_NEGATIVE + ", " + CHROMA_NEGATIVE
        )
        if source_palette_negative:
            cartoon_negative_prompt += ", " + source_palette_negative
        cartoon_attempts: list[dict[str, Any]] = []
        cartoon_workflows: list[dict[str, Any]] = []
        selected_cartoon: dict[str, Any] | None = None
        canonical_profiles = recovery_profiles
        for attempt, profile in enumerate(canonical_profiles, start=1):
            raise_if_cancelled(job_id)
            if attempt > 1:
                update_job(
                    job_id,
                    stage="retrying_cartoon_reference",
                    recovery_active=True,
                    recovery_attempt=attempt,
                )
            attempt_seed = base_seed + (attempt - 1) * 1009
            prompt_source = effective_prompt if profile["use_user_prompt"] else ""
            palette_lock = (
                identity_palette_prompt(
                    source_identity_palette,
                    retry=attempt > 1,
                )
                if source_identity_palette is not None
                else ""
            )
            cartoon_reference_prompt = build_cartoon_reference_prompt(
                prompt_source,
                palette_lock=palette_lock,
            )
            cartoon_prefix = (
                f"local_video_api/{job_id}/qwen_cartoon_reference_attempt_{attempt:02d}"
            )
            cartoon_workflow = qwen_workflow(
                qwen_input_name,
                cartoon_reference_prompt,
                cartoon_negative_prompt,
                attempt_seed,
                cartoon_prefix,
                user_lora=profile["user_lora"],
                user_lora_strength=profile["user_lora_strength"],
                user_lora_mode=profile["user_lora_mode"],
                user_loras=profile["user_loras"],
                steps=profile["steps"],
                cfg=profile["cfg"],
                denoise=profile["denoise"],
                suppress_builtin_style=suppress_builtin_style,
            )
            cartoon_workflows.append(cartoon_workflow)
            write_workflow(
                job_dir,
                f"qwen_cartoon_reference_attempt_{attempt:02d}.json",
                cartoon_workflow,
            )
            cartoon_entry = queue_and_wait(
                cartoon_workflow,
                f"{job_id} cartoon reference attempt {attempt}",
                timeout=7200,
                job_id=job_id,
            )
            cartoon_source = first_output(
                cartoon_entry,
                {".png", ".jpg", ".jpeg", ".webp"},
                "Qwen cartoon reference",
            )
            attempt_reference = job_dir / f"cartoon_reference_attempt_{attempt:02d}.png"
            with Image.open(cartoon_source) as generated:
                cartoon_size = generated.size
                generated.convert("RGB").save(
                    attempt_reference,
                    format="PNG",
                    optimize=True,
                )
            candidate_identity_palette = (
                identity_palette_signature(attempt_reference)
                if source_identity_palette is not None
                else None
            )
            palette_quality = (
                identity_palette_quality(
                    source_identity_palette,
                    candidate_identity_palette,
                )
                if (
                    source_identity_palette is not None
                    and candidate_identity_palette is not None
                )
                else None
            )
            attempt_cutout = (
                job_dir / f"cartoon_reference_cutout_attempt_{attempt:02d}.png"
            )
            try:
                chroma_metrics = evaluate_chroma_cutout(
                    attempt_reference, attempt_cutout
                )
            except Exception as exc:  # noqa: BLE001 - image requests may return the saved raw image
                if not image_request:
                    raise
                chroma_metrics = {
                    "acceptable": False,
                    "issues": ["background_removal_exception"],
                    "error": f"{type(exc).__name__}: {str(exc)[:1000]}",
                }
            cartoon_quality = (
                pose_cutout_quality(attempt_cutout, "canonical character")
                if chroma_metrics["acceptable"]
                else None
            )
            attempt_acceptable = bool(
                chroma_metrics["acceptable"]
                and cartoon_quality
                and cartoon_quality["acceptable"]
                and (palette_quality is None or palette_quality["acceptable"])
            )
            record = {
                "attempt": attempt,
                "profile": profile["name"],
                "seed": attempt_seed,
                "keyframe": attempt_reference.name,
                "cutout": attempt_cutout.name if chroma_metrics["acceptable"] else None,
                "settings": {
                    "steps": profile["steps"],
                    "cfg": profile["cfg"],
                    "denoise": profile["denoise"],
                    "user_lora": profile["user_lora"],
                    "user_lora_strength": profile["user_lora_strength"],
                    "user_loras": profile["user_loras"],
                    "user_lora_strength_scale": round(
                        canonical_lora_scale
                        * float(profile["user_lora_strength_scale"]),
                        6,
                    ),
                    "user_lora_mode": profile["user_lora_mode"],
                    "use_user_prompt": profile["use_user_prompt"],
                },
                "chroma": chroma_metrics,
                "quality": cartoon_quality,
                "identity_palette": palette_quality,
                "candidate_identity_palette": candidate_identity_palette,
            }
            cartoon_attempts.append(record)
            if attempt_acceptable:
                selected_cartoon = {
                    **record,
                    "raw_path": attempt_reference,
                    "cutout_path": attempt_cutout,
                    "workflow": cartoon_workflow,
                    "size": cartoon_size,
                    "profile_index": attempt - 1,
                    "transparent": True,
                    "background_removal_failed": False,
                }
                if attempt > 1:
                    append_recovery_event(
                        job_id,
                        {
                            "stage": "cartoon_reference",
                            "attempt": attempt,
                            "reason": "unclean_chroma_background",
                            "strategy": "safe_qwen_retry",
                            "profile": profile["name"],
                            "result": "accepted",
                            "chroma": chroma_metrics,
                            "quality": cartoon_quality,
                            "identity_palette": palette_quality,
                        },
                    )
                break
            if palette_quality is not None and not palette_quality["acceptable"]:
                rejection_reason = "canonical_identity_palette_drift"
            elif not chroma_metrics["acceptable"]:
                rejection_reason = "unclean_chroma_background"
            else:
                rejection_reason = "canonical_subject_quality_invalid"
            if (
                image_request
                and not chroma_metrics["acceptable"]
                and (palette_quality is None or palette_quality["acceptable"])
            ):
                background_removal_error = "Generated image did not contain a clean removable green background."
                record.update(
                    {
                        "profile": f"{profile['name']}_raw_image",
                        "cutout": None,
                        "transparent": False,
                        "background_removal_failed": True,
                        "background_removal_error": background_removal_error,
                    }
                )
                selected_cartoon = {
                    **record,
                    "raw_path": attempt_reference,
                    "cutout_path": None,
                    "workflow": cartoon_workflow,
                    "size": cartoon_size,
                    "profile_index": attempt - 1,
                }
                append_background_removal_failure(
                    job_id,
                    {
                        "stage": "cartoon_reference",
                        "attempt": attempt,
                        "reason": rejection_reason,
                        "strategy": "return_raw_image",
                        "profile": profile["name"],
                        "result": "returned_opaque_original",
                        "chroma": chroma_metrics,
                        "quality": cartoon_quality,
                        "identity_palette": palette_quality,
                    },
                )
                LOGGER.warning(
                    "%s cartoon reference could not be reliably cut out; returning the raw image "
                    "for image request without changing generation settings",
                    job_id,
                )
                break
            append_recovery_event(
                job_id,
                {
                    "stage": "cartoon_reference",
                    "attempt": attempt,
                    "reason": rejection_reason,
                    "strategy": "safe_qwen_retry",
                    "profile": profile["name"],
                    "result": "rejected",
                    "chroma": chroma_metrics,
                    "quality": cartoon_quality,
                    "identity_palette": palette_quality,
                },
            )
            LOGGER.warning(
                "%s cartoon reference attempt %s (%s) failed checks: chroma=%s quality=%s identity_palette=%s",
                job_id,
                attempt,
                profile["name"],
                chroma_metrics["issues"],
                cartoon_quality["issues"] if cartoon_quality else None,
                palette_quality["issues"] if palette_quality else None,
            )
        if selected_cartoon is None:
            fallback_record = select_canonical_fallback_attempt(
                cartoon_attempts,
                identity_palette_required=bool(
                    identity_palette_guard
                    and source_identity_palette is not None
                    and source_identity_palette.get("available")
                ),
            )
            fallback_source = job_dir / fallback_record["keyframe"]
            fallback_green = job_dir / "cartoon_reference_semantic_green.png"
            fallback_cutout = job_dir / "cartoon_reference_semantic_cutout_final.png"
            fallback_semantic = job_dir / "cartoon_reference_semantic_cutout_raw.png"
            try:
                chroma_metrics = semantic_background_recovery(
                    fallback_source,
                    fallback_green,
                    fallback_cutout,
                    fallback_semantic,
                )
            except Exception as exc:  # noqa: BLE001 - image mode must preserve a usable raw result
                if not image_request:
                    raise
                best_identity_record = fallback_record
                best_identity_source = job_dir / best_identity_record["keyframe"]
                with Image.open(best_identity_source) as raw_image:
                    cartoon_size = raw_image.size
                background_removal_error = (
                    "Background removal could not preserve a complete subject; "
                    "the original generated image was returned."
                )
                selected_cartoon = {
                    **best_identity_record,
                    "profile": f"{best_identity_record['profile']}_raw_image",
                    "keyframe": best_identity_source.name,
                    "cutout": None,
                    "raw_path": best_identity_source,
                    "cutout_path": None,
                    "workflow": cartoon_workflows[
                        int(best_identity_record["attempt"]) - 1
                    ],
                    "size": cartoon_size,
                    "profile_index": int(best_identity_record["attempt"]) - 1,
                    "transparent": False,
                    "background_removal_failed": True,
                    "background_removal_error": background_removal_error,
                }
                append_background_removal_failure(
                    job_id,
                    {
                        "stage": "cartoon_reference_semantic_recovery",
                        "attempt": best_identity_record["attempt"],
                        "reason": "semantic_background_recovery_failed",
                        "strategy": "return_best_identity_raw_image",
                        "profile": best_identity_record["profile"],
                        "result": "returned_opaque_original",
                        "identity_palette": best_identity_record.get(
                            "identity_palette"
                        ),
                        "error": f"{type(exc).__name__}: {str(exc)[:1000]}",
                    },
                )
                LOGGER.warning(
                    "%s cartoon reference semantic recovery failed; returning "
                    "the best identity-preserving raw image for image mode",
                    job_id,
                )
            else:
                with Image.open(fallback_green) as recovered_image:
                    cartoon_size = recovered_image.size
                selected_cartoon = {
                    "attempt": len(cartoon_attempts) + 1,
                    "profile": "rembg_u2net_fallback",
                    "seed": fallback_record["seed"],
                    "keyframe": fallback_green.name,
                    "cutout": fallback_cutout.name,
                    "settings": fallback_record["settings"],
                    "chroma": chroma_metrics,
                    "identity_palette": fallback_record.get("identity_palette"),
                    "candidate_identity_palette": fallback_record.get(
                        "candidate_identity_palette"
                    ),
                    "raw_path": fallback_green,
                    "cutout_path": fallback_cutout,
                    "workflow": cartoon_workflows[int(fallback_record["attempt"]) - 1],
                    "size": cartoon_size,
                    "profile_index": int(fallback_record["attempt"]) - 1,
                    "transparent": True,
                    "background_removal_failed": False,
                }
                cartoon_attempts.append(
                    {
                        "attempt": selected_cartoon["attempt"],
                        "profile": selected_cartoon["profile"],
                        "seed": selected_cartoon["seed"],
                        "keyframe": selected_cartoon["keyframe"],
                        "cutout": selected_cartoon["cutout"],
                        "settings": selected_cartoon["settings"],
                        "chroma": selected_cartoon["chroma"],
                        "identity_palette": selected_cartoon.get("identity_palette"),
                        "candidate_identity_palette": selected_cartoon.get(
                            "candidate_identity_palette"
                        ),
                    }
                )
                append_recovery_event(
                    job_id,
                    {
                        "stage": "cartoon_reference",
                        "attempt": selected_cartoon["attempt"],
                        "reason": "qwen_chroma_retries_exhausted",
                        "strategy": "rembg_u2net_fallback",
                        "profile": "rembg_u2net_fallback",
                        "result": "accepted",
                        "chroma": chroma_metrics,
                        "identity_palette": selected_cartoon.get("identity_palette"),
                    },
                )
        cartoon_reference = job_dir / "cartoon_reference.png"
        cartoon_transparent = job_dir / "cartoon_reference_transparent.png"
        cartoon_is_transparent = bool(selected_cartoon.get("transparent", True))
        if cartoon_is_transparent:
            try:
                prepare_landscape_keyframe(
                    selected_cartoon["cutout_path"],
                    cartoon_reference,
                    tuple(selected_cartoon["size"]),
                    background_rgb=(0, 255, 0),
                )
                normalized_chroma = evaluate_chroma_cutout(
                    cartoon_reference,
                    cartoon_transparent,
                )
                if not normalized_chroma["acceptable"]:
                    raise RuntimeError(
                        "Could not normalize the recovered cartoon reference to pure green."
                    )
                cartoon_output = cartoon_transparent
            except Exception as exc:  # noqa: BLE001 - preserve the generated image for image mode
                if not image_request:
                    raise
                cartoon_is_transparent = False
                selected_cartoon.update(
                    {
                        "profile": f"{selected_cartoon['profile']}_raw_image",
                        "transparent": False,
                        "background_removal_failed": True,
                        "background_removal_error": (
                            "Background removal finalization failed; the raw image was returned."
                        ),
                    }
                )
                normalized_chroma = {
                    "acceptable": False,
                    "issues": ["background_removal_finalization_exception"],
                    "error": f"{type(exc).__name__}: {str(exc)[:1000]}",
                }
                shutil.copy2(selected_cartoon["raw_path"], cartoon_reference)
                cartoon_output = cartoon_reference
                append_background_removal_failure(
                    job_id,
                    {
                        "stage": "cartoon_reference_finalization",
                        "attempt": selected_cartoon["attempt"],
                        "reason": "background_removal_finalization_exception",
                        "strategy": "return_raw_image",
                        "profile": selected_cartoon["profile"],
                        "result": "returned_opaque_original",
                        "error": normalized_chroma["error"],
                    },
                )
        else:
            shutil.copy2(selected_cartoon["raw_path"], cartoon_reference)
            normalized_chroma = None
            cartoon_output = cartoon_reference
        canonical_palette_audit = identity_palette_audit(
            identity_palette_guard,
            selected_cartoon.get("identity_palette"),
        )
        write_workflow(
            job_dir,
            "qwen_cartoon_reference.json",
            selected_cartoon["workflow"],
        )
        write_json_atomic(
            job_dir / "workflows" / "qwen_cartoon_reference_quality.json",
            {
                "selected_attempt": selected_cartoon["attempt"],
                "selected_profile": selected_cartoon["profile"],
                "selected_source_chroma": selected_cartoon["chroma"],
                "selected_normalized_chroma": normalized_chroma,
                "source_identity_palette": source_identity_palette,
                "selected_identity_palette": selected_cartoon.get("identity_palette"),
                "selected_candidate_identity_palette": selected_cartoon.get(
                    "candidate_identity_palette"
                ),
                "transparent": cartoon_is_transparent,
                "background_removal_failed": selected_cartoon.get(
                    "background_removal_failed", False
                ),
                "background_removal_error": selected_cartoon.get(
                    "background_removal_error"
                ),
                "attempts": cartoon_attempts,
            },
        )
        update_job(
            job_id,
            recovered=cartoon_is_transparent and selected_cartoon["attempt"] > 1,
            recovery_active=False,
            recovery_attempt=None,
            cartoon_reference_profile=selected_cartoon["profile"],
            effective_canonical_user_loras=selected_cartoon["settings"]["user_loras"],
            canonical_user_lora_strength_scale=selected_cartoon["settings"][
                "user_lora_strength_scale"
            ],
            effective_canonical_settings=selected_cartoon["settings"],
            effective_canonical_identity_palette=selected_cartoon.get(
                "identity_palette"
            ),
            canonical_identity_palette_verdict=canonical_palette_audit[
                "identity_palette_verdict"
            ],
        )
        append_output(
            job_id,
            {
                "type": "cartoon_reference",
                "filename": cartoon_output.name,
                "width": selected_cartoon["size"][0],
                "height": selected_cartoon["size"][1],
                "transparent": cartoon_is_transparent,
                "background_removal_failed": selected_cartoon.get(
                    "background_removal_failed", False
                ),
                "background_removal_error": selected_cartoon.get(
                    "background_removal_error"
                ),
                "requested_user_lora_preserved": bool(
                    user_loras
                    and selected_cartoon["settings"]["user_loras"] == user_loras
                    and selected_cartoon["settings"]["user_lora_mode"] == user_lora_mode
                ),
                **canonical_palette_audit,
                "visual_style": "identity_preserving_cute_3d_natural_cat",
                "source_role": "canonical_character_design",
                "recovery_profile": selected_cartoon["profile"],
            },
        )
        comfy_cartoon_reference = comfy_input_dir / cartoon_reference.name
        shutil.copy2(cartoon_reference, comfy_cartoon_reference)
        qwen_input_name = (comfy_relative_dir / cartoon_reference.name).as_posix()
        canonical_similarity_reference = cartoon_reference

    keyframes: list[Path] = []
    keyframe_transparency: list[bool] = []
    accepted_pose_similarity_references: list[dict[str, Any]] = []
    effective_pose_lora_attempts: list[dict[str, Any]] = []
    for index, pose in enumerate(poses, start=1):
        raise_if_cancelled(job_id)
        update_job(
            job_id,
            stage="generating_cartoon_keyframes"
            if transparent_gif
            else "generating_keyframes",
            progress=round(
                (18 if transparent_gif else 5) + 17 * (index - 1) / len(poses), 1
            ),
            current_pose_index=index,
            current_pose=pose,
        )
        prompt = build_qwen_prompt(
            "" if transparent_gif else effective_prompt,
            pose,
            transparent_gif=transparent_gif,
        )
        negative_prompt = QWEN_NEGATIVE
        if transparent_gif:
            active_pose_profiles = pose_generation_profiles(pose, pose_profiles)
            negative_prompt += ", " + CARTOON_NEGATIVE + ", " + CHROMA_NEGATIVE
            if source_palette_negative:
                negative_prompt += ", " + source_palette_negative
            if action_key(pose) == "running":
                negative_prompt += (
                    ", seated pose, sitting, crouched sitting, airborne leap, flying pose, extreme foreshortening, hidden hind leg, shortened hind leg, "
                    "fused legs, overlapping legs, detached paw, paw fragment, disconnected limb"
                )
            elif action_key(pose) == "eating":
                negative_prompt += (
                    ", food held in paw, paw holding kibble, raised front paw, paw-to-mouth gesture, "
                    "humanlike eating, anthropomorphic hand gesture, kibble outside bowl, kibble above bowl rim, "
                    "kibble at mouth, open mouth holding food"
                )
        keyframe = job_dir / f"pose_{index:02d}_keyframe.png"
        if transparent_gif:
            bowl_action = action_uses_bowl(pose)
            pose_palette_bottom_exclusion = (
                BOWL_IDENTITY_PALETTE_BOTTOM_EXCLUSION if bowl_action else 0.0
            )
            pose_reference_identity_palette = (
                identity_palette_signature(
                    canonical_similarity_reference,
                    exclude_bottom_fraction=pose_palette_bottom_exclusion,
                )
                if (
                    identity_palette_guard
                    and canonical_similarity_reference is not None
                )
                else None
            )
            pose_reference_upper_body_identity_palette = (
                identity_palette_signature(
                    canonical_similarity_reference,
                    exclude_bottom_fraction=(
                        BOWL_IDENTITY_PALETTE_UPPER_BODY_EXCLUSION
                    ),
                )
                if (
                    bowl_action
                    and identity_palette_guard
                    and canonical_similarity_reference is not None
                )
                else None
            )
            pose_identity_palette_prompt_signature = (
                identity_palette_signature(
                    canonical_similarity_reference,
                    exclude_bottom_fraction=0.0,
                )
                if (
                    bowl_action
                    and identity_palette_guard
                    and canonical_similarity_reference is not None
                )
                else pose_reference_identity_palette
            )
            attempts: list[dict[str, Any]] = []
            best_attempt: dict[str, Any] | None = None
            best_raw_attempt: dict[str, Any] | None = None
            best_semantic_source: dict[str, Any] | None = None
            best_semantic_penalty: float | None = None
            for attempt, profile in enumerate(active_pose_profiles, start=1):
                raise_if_cancelled(job_id)
                if attempt > 1:
                    update_job(
                        job_id,
                        stage="retrying_cartoon_keyframe",
                        quality_retry=attempt,
                        recovery_active=True,
                        recovery_attempt=attempt,
                    )
                attempt_seed = (
                    base_seed + pose_seed_offset(pose, index) + (attempt - 1) * 1009
                )
                prefix = f"local_video_api/{job_id}/qwen_pose_{index:02d}_attempt_{attempt:02d}"
                attempt_prompt = build_qwen_prompt(
                    "",
                    pose,
                    transparent_gif=True,
                    quality_retry=profile["quality_retry"],
                )
                if pose_identity_palette_prompt_signature is not None:
                    attempt_prompt += "\n\n" + identity_palette_prompt(
                        pose_identity_palette_prompt_signature,
                        retry=attempt > 1,
                    )
                workflow = qwen_workflow(
                    qwen_input_name,
                    attempt_prompt,
                    negative_prompt,
                    attempt_seed,
                    prefix,
                    user_lora=profile["user_lora"],
                    user_lora_strength=profile["user_lora_strength"],
                    user_lora_mode=profile["user_lora_mode"],
                    user_loras=profile["user_loras"],
                    steps=profile["steps"],
                    cfg=profile["cfg"],
                    denoise=profile["denoise"],
                    builtin_style_strength=QWEN_POSE_STYLE_LORA_STRENGTH,
                    suppress_builtin_style=suppress_builtin_style,
                )
                write_workflow(
                    job_dir,
                    f"qwen_pose_{index:02d}_attempt_{attempt:02d}.json",
                    workflow,
                )
                entry = queue_and_wait(
                    workflow,
                    f"{job_id} keyframe {index}/{len(poses)} attempt {attempt}",
                    timeout=7200,
                    job_id=job_id,
                )
                source_output = first_output(
                    entry,
                    {".png", ".jpg", ".jpeg", ".webp"},
                    "Qwen keyframe",
                )
                attempt_keyframe = (
                    job_dir / f"pose_{index:02d}_keyframe_attempt_{attempt:02d}.png"
                )
                with Image.open(source_output) as generated:
                    generated.convert("RGB").save(
                        attempt_keyframe, format="PNG", optimize=True
                    )
                candidate_identity_palette = (
                    identity_palette_signature(
                        attempt_keyframe,
                        exclude_bottom_fraction=pose_palette_bottom_exclusion,
                    )
                    if pose_reference_identity_palette is not None
                    else None
                )
                primary_palette_quality = (
                    identity_palette_quality(
                        pose_reference_identity_palette,
                        candidate_identity_palette,
                    )
                    if (
                        pose_reference_identity_palette is not None
                        and candidate_identity_palette is not None
                    )
                    else None
                )
                candidate_upper_body_identity_palette = (
                    identity_palette_signature(
                        attempt_keyframe,
                        exclude_bottom_fraction=(
                            BOWL_IDENTITY_PALETTE_UPPER_BODY_EXCLUSION
                        ),
                    )
                    if pose_reference_upper_body_identity_palette is not None
                    else None
                )
                upper_body_palette_quality = (
                    identity_palette_quality(
                        pose_reference_upper_body_identity_palette,
                        candidate_upper_body_identity_palette,
                    )
                    if (
                        pose_reference_upper_body_identity_palette is not None
                        and candidate_upper_body_identity_palette is not None
                    )
                    else None
                )
                palette_quality = (
                    bowl_aware_identity_palette_quality(
                        primary_palette_quality,
                        upper_body_palette_quality,
                    )
                    if primary_palette_quality is not None and bowl_action
                    else primary_palette_quality
                )
                attempt_cutout = (
                    job_dir / f"pose_{index:02d}_qwen_cutout_attempt_{attempt:02d}.png"
                )
                try:
                    chroma_metrics = evaluate_chroma_cutout(
                        attempt_keyframe,
                        attempt_cutout,
                    )
                except Exception as exc:  # noqa: BLE001 - image requests may return raw output
                    if not image_request:
                        raise
                    chroma_metrics = {
                        "acceptable": False,
                        "issues": ["background_removal_exception"],
                        "error": f"{type(exc).__name__}: {str(exc)[:1000]}",
                    }
                record = {
                    "attempt": attempt,
                    "profile": profile["name"],
                    "seed": attempt_seed,
                    "keyframe": attempt_keyframe.name,
                    "cutout": attempt_cutout.name
                    if chroma_metrics["acceptable"]
                    else None,
                    "chroma": chroma_metrics,
                    "quality": None,
                    "similarity": None,
                    "identity_palette": palette_quality,
                    "candidate_identity_palette": candidate_identity_palette,
                    "candidate_upper_body_identity_palette": (
                        candidate_upper_body_identity_palette
                    ),
                    "combined_penalty": None,
                    "settings": {
                        "steps": profile["steps"],
                        "cfg": profile["cfg"],
                        "denoise": profile["denoise"],
                        "user_lora": profile["user_lora"],
                        "user_lora_strength": profile["user_lora_strength"],
                        "user_loras": profile["user_loras"],
                        "user_lora_strength_scale": profile["user_lora_strength_scale"],
                        "user_lora_mode": profile["user_lora_mode"],
                        "use_user_prompt": profile["use_user_prompt"],
                        "quality_retry": profile["quality_retry"],
                        "total_strength_cap": profile["total_strength_cap"],
                        "per_lora_strength_cap": profile["per_lora_strength_cap"],
                    },
                    "workflow": workflow,
                }
                if not chroma_metrics["acceptable"]:
                    raw_similarity = pose_similarity_quality(
                        attempt_keyframe,
                        pose=pose,
                        canonical_path=canonical_similarity_reference,
                        previous_poses=accepted_pose_similarity_references,
                    )
                    record.update(
                        {
                            "profile": f"{profile['name']}_raw_image",
                            "cutout": None,
                            "transparent": False,
                            "background_removal_failed": True,
                            "background_removal_error": (
                                "Generated image did not contain a clean removable green background."
                            ),
                            "background_removal_failure_reason": (
                                "unclean_chroma_background"
                            ),
                            "similarity": raw_similarity,
                            "combined_penalty": round(
                                float(raw_similarity["penalty"])
                                + (
                                    float(palette_quality["penalty"])
                                    if palette_quality is not None
                                    else 0.0
                                ),
                                6,
                            ),
                        }
                    )
                    attempts.append(
                        {
                            key: value
                            for key, value in record.items()
                            if key != "workflow"
                        }
                    )
                    if semantic_recovery_candidate_eligible(record):
                        semantic_penalty = 100.0 + float(raw_similarity["penalty"])
                        if (
                            best_semantic_source is None
                            or best_semantic_penalty is None
                            or semantic_penalty < best_semantic_penalty
                        ):
                            best_semantic_source = record
                            best_semantic_penalty = semantic_penalty
                    if (
                        image_request
                        and raw_similarity["acceptable"]
                        and (palette_quality is None or palette_quality["acceptable"])
                    ):
                        if (
                            best_raw_attempt is None
                            or record["combined_penalty"]
                            < best_raw_attempt["combined_penalty"]
                        ):
                            best_raw_attempt = record
                        append_recovery_event(
                            job_id,
                            {
                                "stage": "cartoon_keyframe",
                                "pose_index": index,
                                "attempt": attempt,
                                "reason": "unclean_chroma_background",
                                "strategy": "retry_pose_before_raw_fallback",
                                "profile": profile["name"],
                                "result": "rejected",
                                "chroma": chroma_metrics,
                                "similarity": raw_similarity,
                                "identity_palette": palette_quality,
                            },
                        )
                        LOGGER.warning(
                            "%s pose %s attempt %s could not be reliably cut out; "
                            "retrying before an opaque-image fallback",
                            job_id,
                            index,
                            attempt,
                        )
                        continue
                    append_recovery_event(
                        job_id,
                        {
                            "stage": "cartoon_keyframe",
                            "pose_index": index,
                            "attempt": attempt,
                            "reason": "unclean_chroma_background",
                            "strategy": "safe_qwen_retry",
                            "profile": profile["name"],
                            "result": "rejected",
                            "chroma": chroma_metrics,
                            "similarity": raw_similarity,
                            "identity_palette": palette_quality,
                        },
                    )
                    LOGGER.warning(
                        "%s pose %s attempt %s (%s) failed chroma checks: %s",
                        job_id,
                        index,
                        attempt,
                        profile["name"],
                        chroma_metrics["issues"],
                    )
                    continue
                quality = pose_cutout_quality(attempt_cutout, pose)
                similarity = pose_similarity_quality(
                    attempt_cutout,
                    pose=pose,
                    canonical_path=canonical_similarity_reference,
                    previous_poses=accepted_pose_similarity_references,
                )
                record["quality"] = quality
                record["similarity"] = similarity
                record["combined_penalty"] = round(
                    float(quality["penalty"])
                    + float(similarity["penalty"])
                    + (
                        float(palette_quality["penalty"])
                        if palette_quality is not None
                        else 0.0
                    ),
                    6,
                )
                record["transparent"] = True
                if (
                    image_request
                    and similarity["acceptable"]
                    and (palette_quality is None or palette_quality["acceptable"])
                    and not quality["acceptable"]
                ):
                    raw_record = {
                        **record,
                        "profile": f"{profile['name']}_raw_image",
                        "cutout": None,
                        "transparent": False,
                        "background_removal_failed": True,
                        "background_removal_error": (
                            "Background removal could not preserve a complete pose subject; "
                            "the original generated image was returned."
                        ),
                        "background_removal_failure_reason": (
                            "subject_quality_invalid"
                        ),
                    }
                    if (
                        best_raw_attempt is None
                        or raw_record["combined_penalty"]
                        < best_raw_attempt["combined_penalty"]
                    ):
                        best_raw_attempt = raw_record
                attempts.append(
                    {key: value for key, value in record.items() if key != "workflow"}
                )
                if (palette_quality is None or palette_quality["acceptable"]) and (
                    best_attempt is None
                    or record["combined_penalty"] < best_attempt["combined_penalty"]
                ):
                    best_attempt = record
                candidate_acceptable = bool(
                    quality["acceptable"]
                    and similarity["acceptable"]
                    and (palette_quality is None or palette_quality["acceptable"])
                )
                if candidate_acceptable:
                    best_attempt = record
                    if attempt > 1:
                        append_recovery_event(
                            job_id,
                            {
                                "stage": "cartoon_keyframe",
                                "pose_index": index,
                                "attempt": attempt,
                                "reason": "earlier_keyframe_attempt_rejected",
                                "strategy": "pose_prompt_and_lora_retry",
                                "profile": profile["name"],
                                "result": "accepted",
                                "chroma": chroma_metrics,
                                "quality": quality,
                                "similarity": similarity,
                                "identity_palette": palette_quality,
                            },
                        )
                    break
                if palette_quality is not None and not palette_quality["acceptable"]:
                    rejection_reason = "pose_identity_palette_drift"
                elif not quality["acceptable"]:
                    rejection_reason = "subject_quality_invalid"
                else:
                    rejection_reason = "pose_too_similar"
                append_recovery_event(
                    job_id,
                    {
                        "stage": "cartoon_keyframe",
                        "pose_index": index,
                        "attempt": attempt,
                        "reason": rejection_reason,
                        "strategy": "pose_prompt_and_lora_retry",
                        "profile": profile["name"],
                        "result": "rejected",
                        "chroma": chroma_metrics,
                        "quality": quality,
                        "similarity": similarity,
                        "identity_palette": palette_quality,
                    },
                )
                LOGGER.warning(
                    "%s pose %s attempt %s failed pose checks: quality=%s similarity=%s identity_palette=%s",
                    job_id,
                    index,
                    attempt,
                    quality["issues"],
                    similarity["issues"],
                    palette_quality["issues"] if palette_quality else None,
                )
            if best_attempt is not None and (
                not best_attempt["quality"]["acceptable"]
                or not best_attempt["similarity"]["acceptable"]
                or (
                    isinstance(best_attempt.get("identity_palette"), dict)
                    and not best_attempt["identity_palette"]["acceptable"]
                )
            ):
                if (
                    isinstance(best_attempt.get("identity_palette"), dict)
                    and not best_attempt["identity_palette"]["acceptable"]
                ):
                    rejection_reason = "all_keyframe_identity_palette_attempts_rejected"
                elif best_attempt["similarity"]["acceptable"]:
                    rejection_reason = "all_keyframe_subject_quality_attempts_rejected"
                else:
                    rejection_reason = "all_keyframe_similarity_attempts_rejected"
                recovery_strategy = (
                    "return_distinct_raw_image_or_fail"
                    if image_request
                    else (
                        "rembg_u2net_fallback"
                        if best_semantic_source is not None
                        else "fail_closed_without_background_recovery"
                    )
                )
                append_recovery_event(
                    job_id,
                    {
                        "stage": "cartoon_keyframe",
                        "pose_index": index,
                        "attempt": best_attempt["attempt"],
                        "reason": rejection_reason,
                        "strategy": recovery_strategy,
                        "profile": best_attempt["profile"],
                        "result": "rejected",
                        "chroma": best_attempt["chroma"],
                        "quality": best_attempt["quality"],
                        "similarity": best_attempt["similarity"],
                        "identity_palette": best_attempt.get("identity_palette"),
                    },
                )
                best_attempt = None
            if best_attempt is None and image_request and best_raw_attempt is not None:
                best_attempt = best_raw_attempt
                append_background_removal_failure(
                    job_id,
                    {
                        "stage": "cartoon_keyframe",
                        "pose_index": index,
                        "attempt": best_attempt["attempt"],
                        "reason": best_attempt.get(
                            "background_removal_failure_reason",
                            "unclean_chroma_background",
                        ),
                        "strategy": "return_best_distinct_raw_image_after_retries",
                        "profile": best_attempt["profile"],
                        "result": "returned_opaque_original",
                        "chroma": best_attempt["chroma"],
                        "similarity": best_attempt["similarity"],
                        "identity_palette": best_attempt.get("identity_palette"),
                    },
                )
            if best_attempt is None and image_request:
                raise RuntimeError(
                    f"Pose {index} remained invalid after "
                    f"{len(active_pose_profiles)} attempts: "
                    + pose_attempt_failure_summary(attempts)
                )
            if best_attempt is None:
                if best_semantic_source is None:
                    raise RuntimeError(
                        f"Pose {index} remained invalid after "
                        f"{len(active_pose_profiles)} attempts: "
                        + pose_attempt_failure_summary(attempts)
                    )
                fallback_record = best_semantic_source
                if fallback_record["chroma"].get("acceptable", False):
                    raise RuntimeError(
                        "Internal recovery error: semantic background recovery was "
                        "selected for an already clean chroma candidate."
                    )
                fallback_source = job_dir / fallback_record["keyframe"]
                fallback_green = job_dir / f"pose_{index:02d}_semantic_green.png"
                fallback_cutout = (
                    job_dir / f"pose_{index:02d}_semantic_cutout_final.png"
                )
                fallback_semantic = (
                    job_dir / f"pose_{index:02d}_semantic_cutout_raw.png"
                )
                chroma_metrics = semantic_background_recovery(
                    fallback_source,
                    fallback_green,
                    fallback_cutout,
                    fallback_semantic,
                    pose=pose,
                )
                quality = pose_cutout_quality(fallback_cutout, pose)
                similarity = pose_similarity_quality(
                    fallback_cutout,
                    pose=pose,
                    canonical_path=canonical_similarity_reference,
                    previous_poses=accepted_pose_similarity_references,
                )
                semantic_candidate_palette = (
                    identity_palette_signature(
                        fallback_green,
                        exclude_bottom_fraction=pose_palette_bottom_exclusion,
                    )
                    if pose_reference_identity_palette is not None
                    else None
                )
                semantic_primary_palette_quality = (
                    identity_palette_quality(
                        pose_reference_identity_palette,
                        semantic_candidate_palette,
                    )
                    if (
                        pose_reference_identity_palette is not None
                        and semantic_candidate_palette is not None
                    )
                    else None
                )
                semantic_upper_body_candidate_palette = (
                    identity_palette_signature(
                        fallback_green,
                        exclude_bottom_fraction=(
                            BOWL_IDENTITY_PALETTE_UPPER_BODY_EXCLUSION
                        ),
                    )
                    if pose_reference_upper_body_identity_palette is not None
                    else None
                )
                semantic_upper_body_palette_quality = (
                    identity_palette_quality(
                        pose_reference_upper_body_identity_palette,
                        semantic_upper_body_candidate_palette,
                    )
                    if (
                        pose_reference_upper_body_identity_palette is not None
                        and semantic_upper_body_candidate_palette is not None
                    )
                    else None
                )
                semantic_palette_quality = (
                    bowl_aware_identity_palette_quality(
                        semantic_primary_palette_quality,
                        semantic_upper_body_palette_quality,
                    )
                    if semantic_primary_palette_quality is not None and bowl_action
                    else semantic_primary_palette_quality
                )
                if (
                    not quality["acceptable"]
                    or not similarity["acceptable"]
                    or (
                        semantic_palette_quality is not None
                        and not semantic_palette_quality["acceptable"]
                    )
                ):
                    if (
                        semantic_palette_quality is not None
                        and not semantic_palette_quality["acceptable"]
                    ):
                        semantic_rejection_reason = "semantic_identity_palette_invalid"
                    elif not quality["acceptable"]:
                        semantic_rejection_reason = "semantic_subject_quality_invalid"
                    else:
                        semantic_rejection_reason = "semantic_pose_similarity_invalid"
                    append_recovery_event(
                        job_id,
                        {
                            "stage": "cartoon_keyframe",
                            "pose_index": index,
                            "attempt": len(attempts) + 1,
                            "reason": semantic_rejection_reason,
                            "strategy": "rembg_u2net_fallback",
                            "profile": "rembg_u2net_fallback",
                            "result": "rejected",
                            "chroma": chroma_metrics,
                            "quality": quality,
                            "similarity": similarity,
                            "identity_palette": semantic_palette_quality,
                        },
                    )
                    issues = list(quality["issues"])
                    issues.extend(similarity["issues"])
                    if semantic_palette_quality is not None:
                        issues.extend(semantic_palette_quality["issues"])
                    raise RuntimeError(
                        f"Pose {index} semantic recovery failed quality checks: "
                        + ", ".join(issues)
                    )
                best_attempt = {
                    "attempt": len(attempts) + 1,
                    "profile": "rembg_u2net_fallback",
                    "seed": fallback_record["seed"],
                    "keyframe": fallback_green.name,
                    "cutout": fallback_cutout.name,
                    "chroma": chroma_metrics,
                    "quality": quality,
                    "similarity": similarity,
                    "identity_palette": semantic_palette_quality,
                    "candidate_identity_palette": semantic_candidate_palette,
                    "candidate_upper_body_identity_palette": (
                        semantic_upper_body_candidate_palette
                    ),
                    "combined_penalty": round(
                        float(quality["penalty"])
                        + float(similarity["penalty"])
                        + (
                            float(semantic_palette_quality["penalty"])
                            if semantic_palette_quality is not None
                            else 0.0
                        ),
                        6,
                    ),
                    "settings": fallback_record["settings"],
                    "workflow": fallback_record["workflow"],
                    "transparent": True,
                    "background_removal_failed": False,
                }
                attempts.append(
                    {
                        key: value
                        for key, value in best_attempt.items()
                        if key != "workflow"
                    }
                )
                append_recovery_event(
                    job_id,
                    {
                        "stage": "cartoon_keyframe",
                        "pose_index": index,
                        "attempt": best_attempt["attempt"],
                        "reason": "qwen_chroma_retries_exhausted",
                        "strategy": "rembg_u2net_fallback",
                        "profile": "rembg_u2net_fallback",
                        "result": "accepted",
                        "chroma": chroma_metrics,
                        "quality": quality,
                        "similarity": similarity,
                        "identity_palette": semantic_palette_quality,
                    },
                )
            shutil.copy2(job_dir / best_attempt["keyframe"], keyframe)
            keyframe_is_transparent = bool(best_attempt.get("transparent", True))
            if keyframe_is_transparent:
                keyframe_cutout = job_dir / f"pose_{index:02d}_qwen_cutout.png"
                shutil.copy2(job_dir / best_attempt["cutout"], keyframe_cutout)
                selected_keyframe = keyframe_cutout
            else:
                selected_keyframe = keyframe
            write_workflow(
                job_dir, f"qwen_pose_{index:02d}.json", best_attempt["workflow"]
            )
            write_json_atomic(
                job_dir / "workflows" / f"qwen_pose_{index:02d}_quality.json",
                {
                    "selected_attempt": best_attempt["attempt"],
                    "selected_profile": best_attempt["profile"],
                    "selected_chroma": best_attempt["chroma"],
                    "selected_quality": best_attempt["quality"],
                    "selected_similarity": best_attempt["similarity"],
                    "selected_identity_palette": best_attempt.get("identity_palette"),
                    "reference_identity_palette": pose_reference_identity_palette,
                    "prompt_reference_identity_palette": (
                        pose_identity_palette_prompt_signature
                    ),
                    "reference_upper_body_identity_palette": (
                        pose_reference_upper_body_identity_palette
                    ),
                    "selected_candidate_identity_palette": best_attempt.get(
                        "candidate_identity_palette"
                    ),
                    "selected_candidate_upper_body_identity_palette": best_attempt.get(
                        "candidate_upper_body_identity_palette"
                    ),
                    "selected_settings": best_attempt["settings"],
                    "transparent": keyframe_is_transparent,
                    "background_removal_failed": best_attempt.get(
                        "background_removal_failed", False
                    ),
                    "background_removal_error": best_attempt.get(
                        "background_removal_error"
                    ),
                    "attempts": attempts,
                },
            )
            accepted_pose_similarity_references.append(
                {
                    "label": f"pose_{index}",
                    "pose_index": index,
                    "pose": pose,
                    "path": selected_keyframe,
                }
            )
            effective_pose_lora_attempts.append(
                {
                    "pose_index": index,
                    "pose": pose,
                    "attempt": best_attempt["attempt"],
                    "profile": best_attempt["profile"],
                    "user_loras": best_attempt["settings"]["user_loras"],
                    "user_lora_strength_scale": best_attempt["settings"][
                        "user_lora_strength_scale"
                    ],
                    "user_lora_mode": best_attempt["settings"]["user_lora_mode"],
                    "steps": best_attempt["settings"]["steps"],
                    "cfg": best_attempt["settings"]["cfg"],
                    "denoise": best_attempt["settings"]["denoise"],
                    "quality_retry": best_attempt["settings"]["quality_retry"],
                    "total_strength_cap": best_attempt["settings"][
                        "total_strength_cap"
                    ],
                    "per_lora_strength_cap": best_attempt["settings"][
                        "per_lora_strength_cap"
                    ],
                    "similarity": best_attempt["similarity"],
                    "identity_palette": best_attempt.get("identity_palette"),
                }
            )
            if keyframe_is_transparent and best_attempt["attempt"] > 1:
                update_job(job_id, recovered=True)
            update_job(
                job_id,
                recovery_active=False,
                recovery_attempt=None,
                quality_retry=None,
                effective_pose_lora_attempts=effective_pose_lora_attempts,
                pose_similarity_warning_count=sum(
                    bool(item["similarity"].get("warning"))
                    for item in effective_pose_lora_attempts
                ),
                pose_identity_palette_warning_count=sum(
                    bool(
                        isinstance(item.get("identity_palette"), dict)
                        and item["identity_palette"].get("warning")
                    )
                    for item in effective_pose_lora_attempts
                ),
            )
            keyframes.append(selected_keyframe)
            keyframe_transparency.append(keyframe_is_transparent)
        else:
            prefix = f"local_video_api/{job_id}/qwen_pose_{index:02d}"
            workflow = qwen_workflow(
                qwen_input_name,
                prompt,
                negative_prompt,
                base_seed + pose_seed_offset(pose, index),
                prefix,
                user_lora=pose_profiles[0]["user_lora"],
                user_lora_strength=pose_profiles[0]["user_lora_strength"],
                user_lora_mode=pose_profiles[0]["user_lora_mode"],
                user_loras=pose_profiles[0]["user_loras"],
                steps=pose_profiles[0]["steps"],
                cfg=pose_profiles[0]["cfg"],
                denoise=pose_profiles[0]["denoise"],
                builtin_style_strength=QWEN_POSE_STYLE_LORA_STRENGTH,
                suppress_builtin_style=suppress_builtin_style,
            )
            write_workflow(job_dir, f"qwen_pose_{index:02d}.json", workflow)
            entry = queue_and_wait(
                workflow,
                f"{job_id} keyframe {index}/{len(poses)}",
                timeout=7200,
                job_id=job_id,
            )
            source_output = first_output(
                entry,
                {".png", ".jpg", ".jpeg", ".webp"},
                "Qwen keyframe",
            )
            with Image.open(source_output) as generated:
                generated.convert("RGB").save(keyframe, format="PNG", optimize=True)
            keyframes.append(keyframe)
            keyframe_transparency.append(False)
            append_output(
                job_id,
                {
                    "type": "keyframe",
                    "pose_index": index,
                    "pose": pose,
                    "filename": keyframe.name,
                },
            )

    if skip_video:
        raise_if_cancelled(job_id)
        update_job(job_id, stage="finalizing_keyframes", progress=92)
        if transparent_gif:
            for index, (pose, keyframe, keyframe_is_transparent) in enumerate(
                zip(poses, keyframes, keyframe_transparency, strict=True),
                start=1,
            ):
                raise_if_cancelled(job_id)
                raw_keyframe = job_dir / f"pose_{index:02d}_keyframe.png"
                output_removal_error = (
                    "Background removal did not produce a reliable transparent image."
                )
                if keyframe_is_transparent:
                    try:
                        prepare_pose_input(
                            job_id,
                            job_dir,
                            index,
                            pose,
                            keyframe,
                            width,
                            height,
                            transparent=True,
                        )
                    except Exception as exc:  # noqa: BLE001 - image mode returns its saved raw image
                        if not image_request:
                            raise
                        keyframe_is_transparent = False
                        output_removal_error = "Background removal finalization failed; the raw image was returned."
                        append_background_removal_failure(
                            job_id,
                            {
                                "stage": "cartoon_keyframe_finalization",
                                "pose_index": index,
                                "reason": "background_removal_finalization_exception",
                                "strategy": "return_raw_image",
                                "result": "returned_opaque_original",
                                "error": f"{type(exc).__name__}: {str(exc)[:1000]}",
                            },
                        )
                if not keyframe_is_transparent:
                    with Image.open(raw_keyframe) as raw_image:
                        raw_width, raw_height = raw_image.size
                    append_output(
                        job_id,
                        {
                            "type": "image",
                            "pose_index": index,
                            "pose": pose,
                            "filename": raw_keyframe.name,
                            "width": raw_width,
                            "height": raw_height,
                            "transparent": False,
                            "background_removal_failed": True,
                            "background_removal_error": output_removal_error,
                            "requested_user_lora_preserved": bool(
                                user_loras
                                and effective_pose_lora_attempts[index - 1][
                                    "user_loras"
                                ]
                                == user_loras
                                and effective_pose_lora_attempts[index - 1][
                                    "user_lora_mode"
                                ]
                                == user_lora_mode
                            ),
                            "effective_lora": effective_pose_lora_attempts[index - 1],
                            "source_role": "cartoon_animation_keyframe_raw",
                        },
                    )
        raise_if_cancelled(job_id)
        free_comfy_models()
        raise_if_cancelled(job_id)
        update_job(
            job_id,
            status="completed",
            stage="completed_keyframes_only",
            progress=100,
            current_pose_index=None,
            current_pose=None,
            finished_at=utc_now(),
        )
        return

    if transparent_gif and not all(keyframe_transparency):
        raise RuntimeError(
            "Background removal failed; transparent video generation requires clean keyframes."
        )

    raise_if_cancelled(job_id)
    update_job(job_id, stage="switching_to_video_model", progress=38)
    free_comfy_models()

    for index, (pose, keyframe) in enumerate(
        zip(poses, keyframes, strict=True), start=1
    ):
        raise_if_cancelled(job_id)
        update_job(
            job_id,
            stage="generating_videos",
            progress=round(40 + 55 * (index - 1) / len(poses), 1),
            current_pose_index=index,
            current_pose=pose,
        )
        wan_input = prepare_pose_input(
            job_id,
            job_dir,
            index,
            pose,
            keyframe,
            width,
            height,
            transparent=transparent_gif,
        )
        comfy_wan_input = comfy_input_dir / wan_input.name
        shutil.copy2(wan_input, comfy_wan_input)
        prompt = build_wan_prompt(
            effective_prompt, pose, transparent_gif=transparent_gif
        )
        negative_prompt = WAN_NEGATIVE
        if transparent_gif:
            negative_prompt += ", " + CARTOON_WAN_NEGATIVE + ", " + CHROMA_NEGATIVE
        source_output: Path | None = None
        video_engine = WAN_ENGINE
        video_engine_metadata: dict[str, Any] = wan_engine_metadata(
            steps=wan_steps,
            cfg=wan_cfg,
            shift=wan_shift,
        )
        video_seed = base_seed + 10_000 + pose_seed_offset(pose, index)
        if is_fun_control_pose(pose):
            update_job(
                job_id,
                stage="generating_controlled_running_video",
                progress=round(40 + 55 * (index - 1) / len(poses), 1),
            )
            fun_prompt = build_fun_control_prompt(
                effective_prompt, pose, transparent_gif
            )
            try:
                raise_if_cancelled(job_id)
                free_comfy_models()
                source_output, video_engine_metadata = run_fun_control_video(
                    job_dir=job_dir,
                    pose_index=index,
                    reference_image=wan_input,
                    prompt=fun_prompt,
                    negative_prompt=negative_prompt + ", " + FUN_CONTROL_NEGATIVE,
                    seed=video_seed,
                )
                raise_if_cancelled(job_id)
                video_engine = "wan2.2-fun-5b-control"
            except JobCancelled:
                raise
            except Exception as exc:  # noqa: BLE001 - configured production fallback
                failure = {
                    "engine": "wan2.2-fun-5b-control",
                    "fallback_enabled": FUN_CONTROL_FALLBACK_TO_WAN,
                    "error": str(exc),
                    "timestamp": utc_now(),
                }
                write_json_atomic(
                    job_dir / "workflows" / f"fun_pose_{index:02d}_failure.json",
                    failure,
                )
                if not FUN_CONTROL_FALLBACK_TO_WAN:
                    raise
                LOGGER.exception(
                    "%s controlled video %s/%s failed; falling back to ComfyUI Wan",
                    job_id,
                    index,
                    len(poses),
                )
                video_engine_metadata = wan_engine_metadata(
                    steps=wan_steps,
                    cfg=wan_cfg,
                    shift=wan_shift,
                ) | {"fallback_from": failure}

        if source_output is None:
            prefix = f"local_video_api/{job_id}/wan_pose_{index:02d}"
            workflow = wan_workflow(
                (comfy_relative_dir / wan_input.name).as_posix(),
                prompt,
                negative_prompt,
                video_seed,
                width,
                height,
                prefix,
                steps=wan_steps,
                cfg=wan_cfg,
                shift=wan_shift,
            )
            write_workflow(job_dir, f"wan_pose_{index:02d}.json", workflow)
            entry = queue_and_wait(
                workflow,
                f"{job_id} video {index}/{len(poses)}",
                timeout=21600,
                job_id=job_id,
            )
            source_output = first_output(
                entry, {".mp4", ".webm", ".mov", ".mkv"}, "Wan video"
            )
        raise_if_cancelled(job_id)
        suffix = source_output.suffix.lower()
        final_video = job_dir / f"pose_{index:02d}_video{suffix}"
        shutil.copy2(source_output, final_video)
        if transparent_gif:
            append_output(
                job_id,
                {
                    "type": "video_preview",
                    "pose_index": index,
                    "pose": pose,
                    "filename": final_video.name,
                    "width": width,
                    "height": height,
                    "frames": FRAME_COUNT,
                    "fps": FPS,
                    "duration_seconds": FRAME_COUNT / FPS,
                    "video_engine": video_engine,
                    "video_engine_metadata": video_engine_metadata,
                    "primary": False,
                },
            )
            update_job(
                job_id,
                stage="stabilizing_transparent_animation",
                progress=round(95 + 4 * (index - 1) / len(poses), 1),
            )
            final_gif = job_dir / f"pose_{index:02d}_transparent_512.gif"
            final_apng = job_dir / f"pose_{index:02d}_transparent_512_soft.png"
            raise_if_cancelled(job_id)
            metrics = convert_chroma_video_to_transparent_animations(
                final_video,
                final_gif,
                final_apng,
            )
            metrics_path = job_dir / f"pose_{index:02d}_stabilization_metrics.json"
            write_json_atomic(metrics_path, metrics)
            append_output(
                job_id,
                {
                    "type": "transparent_gif",
                    "pose_index": index,
                    "pose": pose,
                    "filename": final_gif.name,
                    "width": TRANSPARENT_GIF_SIZE,
                    "height": TRANSPARENT_GIF_SIZE,
                    "frames": TRANSPARENT_GIF_FRAMES,
                    "fps": TRANSPARENT_GIF_FPS,
                    "duration_seconds": TRANSPARENT_GIF_DURATION_SECONDS,
                    "transparent": True,
                    "visual_style": "identity_preserving_cute_3d_natural_cat",
                    "edge_mode": "fixed_ordered_dither",
                    "stabilization": metrics["after"],
                    "visual_core_stabilization": metrics["visual_core_stabilization"],
                    "head_lock": metrics["head_lock"],
                    "video_engine": video_engine,
                    "video_engine_metadata": video_engine_metadata,
                    "primary": True,
                },
            )
            append_output(
                job_id,
                {
                    "type": "transparent_apng",
                    "pose_index": index,
                    "pose": pose,
                    "filename": final_apng.name,
                    "width": TRANSPARENT_GIF_SIZE,
                    "height": TRANSPARENT_GIF_SIZE,
                    "frames": TRANSPARENT_GIF_FRAMES,
                    "fps": TRANSPARENT_GIF_FPS,
                    "duration_seconds": TRANSPARENT_GIF_DURATION_SECONDS,
                    "transparent": True,
                    "visual_style": "identity_preserving_cute_3d_studio",
                    "soft_alpha": True,
                    "stabilization": metrics["after"],
                    "visual_core_stabilization": metrics["visual_core_stabilization"],
                    "head_lock": metrics["head_lock"],
                    "video_engine": video_engine,
                    "video_engine_metadata": video_engine_metadata,
                    "recommended_for_quality": True,
                    "primary": False,
                },
            )
        else:
            append_output(
                job_id,
                {
                    "type": "video",
                    "pose_index": index,
                    "pose": pose,
                    "filename": final_video.name,
                    "width": width,
                    "height": height,
                    "frames": FRAME_COUNT,
                    "fps": FPS,
                    "duration_seconds": FRAME_COUNT / FPS,
                    "video_engine": video_engine,
                    "video_engine_metadata": video_engine_metadata,
                },
            )

    raise_if_cancelled(job_id)
    free_comfy_models()
    raise_if_cancelled(job_id)
    update_job(
        job_id,
        status="completed",
        stage="completed",
        progress=100,
        current_pose_index=None,
        current_pose=None,
        finished_at=utc_now(),
    )


def worker_loop() -> None:
    LOGGER.info("Local video worker started")
    while not STOP_EVENT.is_set():
        try:
            job_id = JOB_QUEUE.get(timeout=1)
        except queue.Empty:
            continue
        try:
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                already_cancelled = bool(
                    job
                    and (
                        job.get("status") == "cancelled" or job.get("cancel_requested")
                    )
                )
            if already_cancelled:
                mark_job_cancelled(job_id)
                LOGGER.info("Skipped cancelled job %s", job_id)
                continue
            LOGGER.info("Starting job %s", job_id)
            run_job(job_id)
            LOGGER.info("Completed job %s", job_id)
        except JobCancelled:
            LOGGER.info("Job %s was cancelled by the user", job_id)
            mark_job_cancelled(job_id)
            try:
                wait_comfy_idle(timeout=120)
                free_comfy_models()
            except Exception:  # noqa: BLE001 - best effort cleanup
                LOGGER.exception(
                    "GPU cleanup after cancelled job %s also failed", job_id
                )
        except Exception as exc:  # noqa: BLE001 - worker must persist all failures
            LOGGER.error("Job %s failed: %s\n%s", job_id, exc, traceback.format_exc())
            update_job(
                job_id,
                status="failed",
                stage="failed",
                error=str(exc)[:4000],
                recovery_active=False,
                recovery_attempt=None,
                quality_retry=None,
                active_prompt_id=None,
                current_pose_index=None,
                current_pose=None,
                finished_at=utc_now(),
            )
            try:
                wait_comfy_idle(timeout=120)
                free_comfy_models()
            except Exception:  # noqa: BLE001 - best effort cleanup
                LOGGER.exception("GPU cleanup after failed job %s also failed", job_id)
        finally:
            JOB_QUEUE.task_done()
    LOGGER.info("Local video worker stopped")


def restore_jobs() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    queued: list[str] = []
    for path in sorted(OUTPUT_ROOT.glob("*/status.json")):
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
            job_id = str(job["job_id"])
            if (
                job.get("cancel_requested")
                and job.get("status") not in TERMINAL_JOB_STATUSES
            ):
                job.update(
                    status="cancelled",
                    stage="cancelled",
                    error=None,
                    active_prompt_id=None,
                    finished_at=utc_now(),
                    updated_at=utc_now(),
                )
                write_json_atomic(path, job)
            elif job.get("status") == "running":
                job.update(
                    status="failed",
                    stage="failed",
                    error="API service restarted while this job was running; submit it again.",
                    finished_at=utc_now(),
                    updated_at=utc_now(),
                )
                write_json_atomic(path, job)
            elif job.get("status") == "queued":
                queued.append(job_id)
            JOBS[job_id] = job
        except (OSError, KeyError, json.JSONDecodeError):
            LOGGER.exception("Could not restore job status from %s", path)
    for job_id in queued[:MAX_QUEUE_SIZE]:
        JOB_QUEUE.put_nowait(job_id)
    for job_id in queued[MAX_QUEUE_SIZE:]:
        update_job(
            job_id,
            status="failed",
            stage="failed",
            error="Queue capacity was exceeded while restoring the service.",
            finished_at=utc_now(),
        )


app = FastAPI(
    title="Local Multi-Pose Video API",
    version=API_VERSION,
    description=(
        "Upload one image and a prompt to generate multiple local-model 5-second pose videos, "
        "including an optional 512x512 transparent GIF output mode."
    ),
    docs_url=f"{API_PREFIX}/docs",
    redoc_url=None,
    openapi_url=f"{API_PREFIX}/openapi.json",
)


@app.on_event("startup")
def startup() -> None:
    global WORKER_THREAD
    if not API_KEY:
        LOGGER.warning(
            "LOCAL_VIDEO_API_KEY is not configured; protected endpoints will return 503"
        )
    restore_jobs()
    STOP_EVENT.clear()
    WORKER_THREAD = threading.Thread(
        target=worker_loop, name="local-video-worker", daemon=True
    )
    WORKER_THREAD.start()


@app.on_event("shutdown")
def shutdown() -> None:
    STOP_EVENT.set()
    if WORKER_THREAD and WORKER_THREAD.is_alive():
        WORKER_THREAD.join(timeout=5)


@app.get(f"{API_PREFIX}/health")
def health() -> dict[str, Any]:
    comfy_ok = False
    comfy_error: str | None = None
    comfy_queue = {"running": 0, "pending": 0}
    try:
        response = requests.get(f"{COMFY_URL}/queue", timeout=5)
        response.raise_for_status()
        payload = response.json()
        comfy_queue = {
            "running": len(payload.get("queue_running", [])),
            "pending": len(payload.get("queue_pending", [])),
        }
        comfy_ok = True
    except Exception as exc:  # noqa: BLE001 - health endpoint should report, not crash
        comfy_error = str(exc)
    missing_models = [
        path.name for path in required_model_paths() if not path.is_file()
    ]
    profile_model_paths = [
        MODEL_ROOT / "loras" / Path(item["model_name"])
        for profile in IMAGE_PROFILES.values()
        for item in profile["user_loras"]
    ]
    missing_profile_models = [
        path.name for path in profile_model_paths if not path.is_file()
    ]
    missing_models.extend(
        name for name in missing_profile_models if name not in missing_models
    )
    fun_control_paths = [
        FUN_CONTROL_PYTHON,
        FUN_CONTROL_SCRIPT,
        FUN_CONTROL_PROJECT_ROOT / "videox_fun" / "__init__.py",
        FUN_CONTROL_MODEL_ROOT / "diffusion_pytorch_model.safetensors",
        FUN_CONTROL_MODEL_ROOT / "models_t5_umt5-xxl-enc-bf16.pth",
        FUN_CONTROL_MODEL_ROOT / "Wan2.2_VAE.pth",
        FUN_CONTROL_RUNNING_VIDEO,
    ]
    fun_control_missing = [
        str(path) for path in fun_control_paths if not path.is_file()
    ]
    ffmpeg_ok = Path(FFMPEG_BIN).is_file()
    translator_missing = prompt_translator_missing_paths()
    translator_ok = not PROMPT_TRANSLATION_ENABLED or not translator_missing
    return {
        "status": (
            "ok"
            if (
                comfy_ok
                and not missing_models
                and API_KEY
                and ffmpeg_ok
                and translator_ok
            )
            else "degraded"
        ),
        "version": API_VERSION,
        "worker_alive": bool(WORKER_THREAD and WORKER_THREAD.is_alive()),
        "api_key_configured": bool(API_KEY),
        "job_queue": {"pending": JOB_QUEUE.qsize(), "capacity": MAX_QUEUE_SIZE},
        "comfyui": {"ok": comfy_ok, "queue": comfy_queue, "error": comfy_error},
        "models": {"ok": not missing_models, "missing": missing_models},
        "prompt_translation": {
            "enabled": PROMPT_TRANSLATION_ENABLED,
            "ok": translator_ok,
            "provider": "local_opus_mt_zh_en",
            "model": "Helsinki-NLP/opus-mt-zh-en",
            "source_language": "zh",
            "target_language": "en",
            "missing": translator_missing,
        },
        "image_profiles": {
            "default": DEFAULT_IMAGE_PROFILE or None,
            "items": [
                image_profile_public(profile) for profile in IMAGE_PROFILES.values()
            ],
        },
        "qwen_style": {
            "enabled": bool(QWEN_STYLE_LORA),
            "profile": QWEN_STYLE_PROFILE if QWEN_STYLE_LORA else "baseline",
            "lora": QWEN_STYLE_LORA or None,
            "strength": QWEN_STYLE_LORA_STRENGTH if QWEN_STYLE_LORA else None,
            "pose_strength": QWEN_POSE_STYLE_LORA_STRENGTH if QWEN_STYLE_LORA else None,
            "steps": {
                "default": IMAGE_PROFILES[DEFAULT_IMAGE_PROFILE]["qwen_steps"]
                if DEFAULT_IMAGE_PROFILE
                else 4,
                "minimum": 1,
                "maximum": 30,
            },
            "max_user_loras": MAX_USER_LORAS,
            "max_user_lora_bytes_each": MAX_LORA_UPLOAD_BYTES,
            "max_user_lora_fusion_bytes": MAX_LORA_FUSION_BYTES,
            "stage_lora_caps": {
                "canonical_total": QWEN_CANONICAL_USER_LORA_TOTAL_CAP,
                "canonical_per_lora": QWEN_CANONICAL_USER_LORA_PER_LORA_CAP,
                "pose_total": QWEN_POSE_USER_LORA_TOTAL_CAP,
                "pose_retry_total": QWEN_POSE_USER_LORA_RETRY_TOTAL_CAP,
                "pose_final_total": QWEN_POSE_USER_LORA_FINAL_TOTAL_CAP,
                "pose_per_lora": QWEN_POSE_USER_LORA_PER_LORA_CAP,
            },
            "pose_guard": {
                "prompt_isolation": True,
                "max_attempts": QWEN_POSE_MAX_ATTEMPTS,
                "running_extra_attempts": QWEN_RUNNING_POSE_EXTRA_ATTEMPTS,
                "bowl_extra_attempts": QWEN_BOWL_POSE_EXTRA_ATTEMPTS,
                "running_max_attempts": (
                    QWEN_POSE_MAX_ATTEMPTS + QWEN_RUNNING_POSE_EXTRA_ATTEMPTS
                ),
                "bowl_max_attempts": (
                    QWEN_POSE_MAX_ATTEMPTS + QWEN_BOWL_POSE_EXTRA_ATTEMPTS
                ),
                "bowl_palette_bottom_exclusion": (
                    BOWL_IDENTITY_PALETTE_BOTTOM_EXCLUSION
                ),
                "bowl_palette_upper_body_exclusion": (
                    BOWL_IDENTITY_PALETTE_UPPER_BODY_EXCLUSION
                ),
                "similarity_iou": QWEN_POSE_SIMILARITY_IOU_THRESHOLD,
                "similarity_dhash": QWEN_POSE_SIMILARITY_DHASH_THRESHOLD,
                "hard_duplicate_iou": QWEN_POSE_HARD_DUPLICATE_IOU,
                "hard_duplicate_dhash": QWEN_POSE_HARD_DUPLICATE_DHASH,
            },
        },
        "wan_video": {
            **wan_engine_metadata(),
            "ok": not any(not path.is_file() for path in wan_model_paths()),
            "missing": [str(path) for path in wan_model_paths() if not path.is_file()],
        },
        "wan_control": {
            "enabled": FUN_CONTROL_ENABLED,
            "engine": "wan2.2-fun-5b-control" if FUN_CONTROL_ENABLED else None,
            "actions": ["running"] if FUN_CONTROL_ENABLED else [],
            "resolution": [512, 512] if FUN_CONTROL_ENABLED else None,
            "frames": FRAME_COUNT if FUN_CONTROL_ENABLED else None,
            "fps": FPS if FUN_CONTROL_ENABLED else None,
            "steps": FUN_CONTROL_STEPS if FUN_CONTROL_ENABLED else None,
            "gpu_memory_mode": FUN_CONTROL_GPU_MEMORY_MODE
            if FUN_CONTROL_ENABLED
            else None,
            "fallback_to_wan": FUN_CONTROL_FALLBACK_TO_WAN
            if FUN_CONTROL_ENABLED
            else None,
            "model_root": str(FUN_CONTROL_MODEL_ROOT) if FUN_CONTROL_ENABLED else None,
            "project_root": str(FUN_CONTROL_PROJECT_ROOT)
            if FUN_CONTROL_ENABLED
            else None,
            "control_video": str(FUN_CONTROL_RUNNING_VIDEO)
            if FUN_CONTROL_ENABLED
            else None,
            "ok": not fun_control_missing if FUN_CONTROL_ENABLED else True,
            "missing": fun_control_missing if FUN_CONTROL_ENABLED else [],
        },
        "ffmpeg": {"ok": ffmpeg_ok, "path": FFMPEG_BIN if ffmpeg_ok else None},
    }


@app.post(f"{API_PREFIX}/jobs", status_code=status.HTTP_202_ACCEPTED)
async def create_job(
    image: UploadFile = File(..., description="Reference image (PNG/JPEG/WebP)."),
    prompt: str = Form(..., min_length=1, max_length=2000),
    poses: str | None = Form(
        default=None,
        description="Optional JSON string array, newline-separated list, or pipe-separated list.",
    ),
    pose_count: int = Form(
        default=4,
        description="Number of outputs. Defaults to running, eating, drinking, and lying down.",
    ),
    resolution: str = Form(default="720p"),
    output_format: str = Form(
        default="mp4",
        description="mp4 or transparent_gif; transparent_gif forces a 512x512 final output.",
    ),
    skip_video: bool = Form(
        default=False,
        description="Generate the selected pose keyframes but skip Wan video inference.",
    ),
    request_kind: str = Form(
        default="video", description="Calling UI mode: image or video."
    ),
    image_profile: str | None = Form(
        default=None,
        description="Optional named image baseline profile.",
    ),
    user_loras: str | None = Form(
        default=None,
        description="Optional JSON array of user LoRA model_name, original_filename, and strength.",
    ),
    user_lora: str | None = Form(
        default=None, description="Validated user_uploads/*.safetensors name."
    ),
    user_lora_original_filename: str | None = Form(default=None),
    user_lora_strength: float = Form(default=1.0, ge=0.0, le=2.0),
    user_lora_mode: str = Form(
        default="replace", description="replace or stack the built-in style LoRA."
    ),
    qwen_steps: int | None = Form(default=None, ge=1, le=30),
    qwen_cfg: float | None = Form(default=None, ge=0.1, le=5.0),
    qwen_denoise: float | None = Form(default=None, ge=0.1, le=1.0),
    wan_steps: int = Form(default=WAN_STEPS, ge=4, le=40),
    wan_cfg: float = Form(default=WAN_CFG, ge=1.0, le=8.0),
    wan_shift: float = Form(default=WAN_SHIFT, ge=1.0, le=12.0),
    seed: int | None = Form(default=None, ge=0, le=2**63 - 1),
    _: None = Depends(require_api_key),
) -> dict[str, Any]:
    request_kind = request_kind.strip().lower()
    if request_kind not in {"image", "video"}:
        raise HTTPException(
            status_code=422, detail="request_kind must be image or video."
        )
    if request_kind == "image":
        output_format = "transparent_gif"
        skip_video = True
        resolution = "512x512"
    if image_profile is None:
        image_profile = DEFAULT_IMAGE_PROFILE or None
    profile = resolve_image_profile(image_profile, request_kind=request_kind)
    qwen_steps = int(
        qwen_steps
        if qwen_steps is not None
        else (profile["qwen_steps"] if profile else 4)
    )
    qwen_cfg = float(
        qwen_cfg if qwen_cfg is not None else (profile["qwen_cfg"] if profile else 1.0)
    )
    qwen_denoise = float(
        qwen_denoise
        if qwen_denoise is not None
        else (profile["qwen_denoise"] if profile else 1.0)
    )
    seed = int(seed if seed is not None else (profile["seed"] if profile else 20260713))
    if output_format not in OUTPUT_FORMATS:
        raise HTTPException(
            status_code=422,
            detail=f"output_format must be one of {sorted(OUTPUT_FORMATS)}.",
        )
    user_lora_mode = user_lora_mode.strip().lower()
    if user_lora_mode not in {"replace", "stack"}:
        raise HTTPException(
            status_code=422, detail="user_lora_mode must be replace or stack."
        )
    if user_loras and user_loras.strip() and user_lora and user_lora.strip():
        raise HTTPException(
            status_code=422,
            detail="Use either user_loras or the legacy user_lora fields, not both.",
        )
    uploaded_user_lora_specs = normalize_user_lora_specs(
        user_loras,
        legacy_model_name=user_lora,
        legacy_original_filename=user_lora_original_filename,
        legacy_strength=user_lora_strength,
    )
    user_lora_specs, user_lora_mode, image_profile_lora_source = (
        merge_image_profile_loras(
            profile,
            uploaded_user_lora_specs,
            requested_mode=user_lora_mode,
        )
    )
    first_user_lora = user_lora_specs[0] if user_lora_specs else None
    active_user_lora_specs = [
        item for item in user_lora_specs if float(item["strength"]) != 0.0
    ]
    user_lora = first_user_lora["model_name"] if first_user_lora else None
    user_lora_original_filename = (
        first_user_lora["original_filename"] if first_user_lora else None
    )
    user_lora_strength = first_user_lora["strength"] if first_user_lora else 1.0
    if output_format == "transparent_gif":
        resolution = "512x512"
    elif resolution not in RESOLUTIONS:
        raise HTTPException(
            status_code=422, detail=f"resolution must be one of {sorted(RESOLUTIONS)}."
        )
    prompt = prompt.strip()
    if not prompt:
        raise HTTPException(status_code=422, detail="prompt cannot be blank.")
    selected_poses = parse_poses(
        poses,
        pose_count,
        defaults=profile.get("default_poses") if profile else None,
    )
    if JOB_QUEUE.full():
        raise HTTPException(
            status_code=503, detail="Generation queue is full; try again later."
        )
    prompt_en, selected_poses_en, translation = translate_prompt_bundle(
        prompt,
        selected_poses,
    )
    selected_pose_preset_ids = [
        (
            MULTIPOSE_PRESET_BY_PROMPT_ZH[pose]["id"]
            if pose in MULTIPOSE_PRESET_BY_PROMPT_ZH
            else None
        )
        for pose in selected_poses
    ]
    if all(preset_id is not None for preset_id in selected_pose_preset_ids):
        pose_prompt_source = (
            str(profile.get("pose_prompt_source"))
            if profile and profile.get("pose_prompt_source")
            else "default_multipose_presets"
        )
    elif any(preset_id is not None for preset_id in selected_pose_preset_ids):
        pose_prompt_source = "mixed_presets_and_request"
    else:
        pose_prompt_source = "request"

    raw = await image.read(MAX_UPLOAD_BYTES + 1)
    await image.close()
    if not raw:
        raise HTTPException(status_code=422, detail="Uploaded image is empty.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413, detail=f"Image exceeds {MAX_UPLOAD_BYTES} bytes."
        )

    job_id = (
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:10]}"
    )
    job_dir = OUTPUT_ROOT / job_id
    try:
        normalize_uploaded_image(
            raw,
            job_dir / "input.png",
            identity_source_path=job_dir / "input_identity_source.png",
        )
        final_frames = (
            None
            if skip_video
            else (
                TRANSPARENT_GIF_FRAMES
                if output_format == "transparent_gif"
                else FRAME_COUNT
            )
        )
        final_fps = (
            None
            if skip_video
            else (TRANSPARENT_GIF_FPS if output_format == "transparent_gif" else FPS)
        )
        final_duration = (
            None
            if skip_video
            else (
                TRANSPARENT_GIF_DURATION_SECONDS
                if output_format == "transparent_gif"
                else FRAME_COUNT / FPS
            )
        )
        profile_suppresses_builtin_style = bool(
            profile
            and image_profile_lora_source
            in {"image_profile", "image_profile_plus_user"}
        )
        builtin_style_applied = (
            bool(QWEN_STYLE_LORA)
            and not profile_suppresses_builtin_style
            and not (active_user_lora_specs and user_lora_mode == "replace")
        )
        job = {
            "job_id": job_id,
            "status": "queued",
            "stage": "queued",
            "progress": 0,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "started_at": None,
            "finished_at": None,
            "prompt": prompt,
            "prompt_original": prompt,
            "prompt_en": prompt_en,
            "prompt_translation": translation,
            "poses": selected_poses,
            "poses_original": selected_poses,
            "poses_en": selected_poses_en,
            "pose_count": len(selected_poses),
            "image_profile": profile["id"] if profile else None,
            "image_profile_revision": (profile.get("revision") if profile else None),
            "image_profile_label": profile["label"] if profile else None,
            "image_profile_source_job_id": profile["source_job_id"]
            if profile
            else None,
            "pose_prompt_source": pose_prompt_source,
            "pose_preset_ids": selected_pose_preset_ids,
            "pose_seed_offsets": (
                dict(profile.get("pose_seed_offsets", {}))
                if profile
                else dict(MULTIPOSE_SEED_OFFSETS)
            ),
            "identity_palette_guard": bool(
                profile.get("identity_palette_guard", False)
                if profile
                else output_format == "transparent_gif"
            ),
            "identity_palette_policy": (
                profile.get("identity_palette_policy")
                if profile
                else (
                    "source_palette_guard_v1"
                    if output_format == "transparent_gif"
                    else None
                )
            ),
            "image_profile_lora_source": image_profile_lora_source,
            "image_profile_suppresses_builtin_style": (
                profile_suppresses_builtin_style
            ),
            "lock_retry_lora_strengths": bool(
                profile
                and profile.get("lock_retry_lora_strengths", False)
                and image_profile_lora_source == "image_profile"
            ),
            "output_format": output_format,
            "skip_video": skip_video,
            "request_kind": request_kind,
            "generation_mode": "keyframes_only" if skip_video else "video",
            "visual_style": "identity_preserving_cute_3d_natural_cat"
            if output_format == "transparent_gif"
            else "reference",
            "qwen_style_profile": (
                f"image_profile:{profile['id']}"
                if profile and image_profile_lora_source != "user_override"
                else (
                    QWEN_STYLE_PROFILE
                    if builtin_style_applied
                    else (
                        "user_lora_mix"
                        if len(active_user_lora_specs) > 1
                        else ("user_lora" if active_user_lora_specs else "baseline")
                    )
                )
            ),
            "qwen_style_lora": QWEN_STYLE_LORA if builtin_style_applied else None,
            "qwen_style_lora_strength": QWEN_STYLE_LORA_STRENGTH
            if builtin_style_applied
            else None,
            "qwen_pose_style_lora_strength": (
                QWEN_POSE_STYLE_LORA_STRENGTH if builtin_style_applied else None
            ),
            "user_lora": user_lora,
            "user_lora_original_filename": user_lora_original_filename
            if user_lora
            else None,
            "user_lora_strength": user_lora_strength if user_lora else None,
            "user_loras": user_lora_specs,
            "requested_user_loras": user_lora_specs,
            "uploaded_user_loras": uploaded_user_lora_specs,
            "image_profile_user_loras": profile["user_loras"] if profile else [],
            "user_lora_count": len(user_lora_specs),
            "active_user_lora_count": len(active_user_lora_specs),
            "user_lora_mode": user_lora_mode if user_lora_specs else None,
            "qwen_steps": qwen_steps,
            "qwen_cfg": qwen_cfg,
            "qwen_denoise": qwen_denoise,
            "wan_steps": wan_steps,
            "wan_cfg": wan_cfg,
            "wan_shift": wan_shift,
            "resolution": resolution,
            "width": RESOLUTIONS[resolution][0],
            "height": RESOLUTIONS[resolution][1],
            "frames": final_frames,
            "fps": final_fps,
            "duration_seconds": final_duration,
            "seed": seed,
            "canonical_user_lora_total_cap": (
                profile["canonical_total_cap"]
                if profile and image_profile_lora_source != "user_override"
                else QWEN_CANONICAL_USER_LORA_TOTAL_CAP
            ),
            "canonical_user_lora_per_lora_cap": (
                profile["canonical_per_lora_cap"]
                if profile and image_profile_lora_source != "user_override"
                else QWEN_CANONICAL_USER_LORA_PER_LORA_CAP
            ),
            "pose_user_lora_first_total_cap": (
                profile["pose_first_total_cap"]
                if profile and image_profile_lora_source != "user_override"
                else QWEN_POSE_USER_LORA_TOTAL_CAP
            ),
            "pose_user_lora_first_per_lora_cap": (
                profile["pose_first_per_lora_cap"]
                if profile and image_profile_lora_source != "user_override"
                else QWEN_POSE_USER_LORA_PER_LORA_CAP
            ),
            "current_pose_index": None,
            "current_pose": None,
            "outputs": [],
            "automatic_recovery_used": False,
            "recovered": False,
            "degraded_recovery": False,
            "recovery_active": False,
            "recovery_attempt": None,
            "recovery_history": [],
            "background_removal_failed": False,
            "background_removal_failures": [],
            "transparent_outputs_complete": True,
            "error": None,
            "cancel_requested": False,
            "active_prompt_id": None,
        }
        with JOBS_LOCK:
            JOBS[job_id] = job
        persist_job(job_id)
        JOB_QUEUE.put_nowait(job_id)
    except Exception:
        with JOBS_LOCK:
            JOBS.pop(job_id, None)
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    return public_job(job)


@app.get(f"{API_PREFIX}/jobs")
def list_jobs(
    limit: int = Query(default=20, ge=1, le=100),
    _: None = Depends(require_api_key),
) -> dict[str, Any]:
    with JOBS_LOCK:
        jobs = sorted(
            JOBS.values(), key=lambda item: item.get("created_at", ""), reverse=True
        )[:limit]
    return {"jobs": [public_job(job) for job in jobs], "count": len(jobs)}


@app.get(f"{API_PREFIX}/jobs/{{job_id}}")
def job_status(job_id: str, _: None = Depends(require_api_key)) -> dict[str, Any]:
    return get_job(job_id)


@app.post(f"{API_PREFIX}/jobs/{{job_id}}/cancel")
def cancel_job(job_id: str, _: None = Depends(require_api_key)) -> dict[str, Any]:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        current_status = str(job.get("status", ""))
        if current_status in TERMINAL_JOB_STATUSES:
            return public_job(job)
        active_prompt_id = job.get("active_prompt_id")
        if current_status == "queued":
            job.update(
                status="cancelled",
                stage="cancelled",
                cancel_requested=True,
                active_prompt_id=None,
                error=None,
                finished_at=utc_now(),
                updated_at=utc_now(),
            )
        else:
            job.update(
                stage="cancel_requested",
                cancel_requested=True,
                updated_at=utc_now(),
            )
        snapshot = dict(job)
    write_json_atomic(status_path(job_id), snapshot)
    interrupt_comfy_prompt(active_prompt_id)
    return public_job(snapshot)


@app.get(f"{API_PREFIX}/jobs/{{job_id}}/files/{{filename}}")
def download_file(
    job_id: str, filename: str, _: None = Depends(require_api_key)
) -> FileResponse:
    if Path(filename).name != filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    job = get_job(job_id)
    allowed = {item["filename"] for item in job.get("outputs", [])}
    if filename not in allowed:
        raise HTTPException(
            status_code=404, detail="Output file not found for this job."
        )
    path = OUTPUT_ROOT / job_id / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Output file is missing on disk.")
    return FileResponse(path, filename=filename)
