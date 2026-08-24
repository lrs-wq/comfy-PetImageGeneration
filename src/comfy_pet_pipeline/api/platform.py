from __future__ import annotations

import asyncio
import base64
import binascii
import copy
import json
import logging
import mimetypes
import os
import queue
import secrets
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import quote
from uuid import uuid4

import requests
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from PIL import Image
from starlette.background import BackgroundTask

from . import local_video as pipeline


LOGGER = logging.getLogger("pet_generation_platform")
logging.basicConfig(
    level=os.environ.get("PET_PLATFORM_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

VERSION = "1.13.0"
PROJECT_ROOT = Path(
    os.environ.get(
        "PET_PIPELINE_PROJECT_ROOT",
        str(Path(__file__).resolve().parents[3]),
    )
).resolve()
HOST_PAGE = PROJECT_ROOT / "web" / "platform" / "index.html"
HISTORY_PAGE = PROJECT_ROOT / "web" / "platform" / "history.html"
OUTPUT_ROOT = Path(
    os.environ.get(
        "PET_PLATFORM_OUTPUT_ROOT",
        "/data/comfy-pet-pipeline/outputs/test_platform",
    )
).resolve()
PLATFORM_USER = os.environ.get("PET_PLATFORM_USER", "admin").strip()
PLATFORM_PASSWORD = os.environ.get("PET_PLATFORM_PASSWORD", "").strip()
LOCAL_API_URL = os.environ.get("LOCAL_VIDEO_INTERNAL_URL", "http://127.0.0.1:8770").rstrip("/")
LOCAL_API_KEY = os.environ.get("LOCAL_VIDEO_API_KEY", "").strip()
PUBLIC_BASE_PATH = (
    "/" + os.environ.get("PET_PLATFORM_BASE_PATH", "pet-generation").strip("/")
).rstrip("/")
MAX_QUEUE = int(os.environ.get("PET_PLATFORM_IMAGE_QUEUE", "4"))
HISTORY_IMAGE_OUTPUT_TYPES = frozenset(
    {
        "cartoon_reference",
        "transparent_keyframe",
        "keyframe",
        "image",
        "transparent_image",
        "transparent_gif",
        "transparent_apng",
    }
)
HISTORY_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif", ".apng"})
HISTORY_VIDEO_OUTPUT_TYPES = frozenset(
    {
        "video",
        "video_preview",
        "subject_replacement_video",
    }
)
HISTORY_VIDEO_SUFFIXES = frozenset({".mp4", ".webm", ".mov", ".mkv"})

IMAGE_JOBS: dict[str, dict[str, Any]] = {}
IMAGE_JOBS_LOCK = threading.RLock()
IMAGE_QUEUE: queue.Queue[str] = queue.Queue(maxsize=MAX_QUEUE)
STOP_EVENT = threading.Event()
WORKER: threading.Thread | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_auth(authorization: str | None = Header(default=None)) -> None:
    if not PLATFORM_PASSWORD:
        raise HTTPException(status_code=503, detail="Platform password is not configured.")
    if not authorization or not authorization.startswith("Basic "):
        raise HTTPException(
            status_code=401,
            detail="Authentication required.",
            headers={"WWW-Authenticate": 'Basic realm="Pet Generation Test Platform"'},
        )
    try:
        decoded = base64.b64decode(authorization[6:], validate=True).decode("utf-8")
        username, password = decoded.split(":", 1)
    except (binascii.Error, ValueError, UnicodeDecodeError):
        username, password = "", ""
    if not (
        secrets.compare_digest(username, PLATFORM_USER)
        and secrets.compare_digest(password, PLATFORM_PASSWORD)
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password.",
            headers={"WWW-Authenticate": 'Basic realm="Pet Generation Test Platform"'},
        )


def status_path(job_id: str) -> Path:
    return OUTPUT_ROOT / job_id / "status.json"


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def persist_image_job(job_id: str) -> None:
    with IMAGE_JOBS_LOCK:
        payload = dict(IMAGE_JOBS[job_id])
    write_json_atomic(status_path(job_id), payload)


def update_image_job(job_id: str, **changes: Any) -> None:
    with IMAGE_JOBS_LOCK:
        IMAGE_JOBS[job_id].update(changes)
        IMAGE_JOBS[job_id]["updated_at"] = utc_now()
    persist_image_job(job_id)


def public_image_job(job: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(job)
    result["kind"] = "image"
    result["status_url"] = f"{PUBLIC_BASE_PATH}/api/jobs/image/{job['job_id']}"
    for output in result.get("outputs", []):
        output["download_url"] = (
            f"{PUBLIC_BASE_PATH}/api/jobs/image/{job['job_id']}/files/{output['filename']}"
        )
    return result


def restore_image_jobs() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for path in OUTPUT_ROOT.glob("*/status.json"):
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
            job_id = str(job["job_id"])
            if job.get("status") in {"queued", "running"}:
                job.update(
                    status="failed",
                    stage="interrupted",
                    error="The test platform restarted before this image job completed.",
                    finished_at=utc_now(),
                    updated_at=utc_now(),
                )
                write_json_atomic(path, job)
            IMAGE_JOBS[job_id] = job
        except Exception:
            LOGGER.exception("Unable to restore image job from %s", path)


def image_positive_prompt(user_prompt: str, transparent: bool) -> str:
    background = (
        pipeline.CHROMA_BACKGROUND_PROMPT
        if transparent
        else "Use a clean, simple, softly lit studio background without text or clutter."
    )
    return f"""Use the uploaded photograph as the only identity reference.

User request: {user_prompt}

Create one polished premium stylized 3D cartoon image of this exact same cat. Follow the user's requested pose, action and mood while strictly preserving the original cat's fur base colors, marking placement, face shape, cheek width, eye color, ears, nose, muzzle, natural coat-length category, breed traits, body proportions and tail. Preserve natural asymmetry and do not invent stripes or spots.

Visual style: high-end animated-film character render, softly rounded face, large clean glossy eyes with natural catchlights, tiny natural nose, compact mouth, soft finely groomed fur, warm high-key character lighting, appealing but recognizable. {pipeline.SLEEK_GROOMING_PROMPT} The grooming rule overrides any request or reference feature that would amplify fluffy volume, standing guard hairs, long isolated tufts or a jagged fur silhouette; it does not override identity colors or markings. The cat remains completely unclothed and wears no glasses, collar, bow tie, shirt, harness, hat, shoes, jewelry or other accessory unless the user explicitly requests a prop. Show exactly one complete subject with anatomically correct limbs and full tail, generous padding, no crop, no duplicate animal, no text, logo or watermark.

{background}"""


def run_image_job(job_id: str) -> None:
    with IMAGE_JOBS_LOCK:
        job = dict(IMAGE_JOBS[job_id])
    job_dir = OUTPUT_ROOT / job_id
    transparent = bool(job["transparent"])
    try:
        update_image_job(
            job_id,
            status="running",
            stage="waiting_for_gpu",
            progress=5,
            started_at=utc_now(),
        )
        pipeline.wait_comfy_idle(timeout=21600)
        pipeline.free_comfy_models()

        comfy_relative = Path("test_platform") / job_id
        comfy_input_dir = pipeline.COMFY_INPUT / comfy_relative
        comfy_input_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(job_dir / "input.png", comfy_input_dir / "input.png")
        input_name = (comfy_relative / "input.png").as_posix()
        positive = image_positive_prompt(job["prompt"], transparent)
        negative = pipeline.QWEN_NEGATIVE + ", " + pipeline.CARTOON_NEGATIVE
        if transparent:
            negative += ", " + pipeline.CHROMA_NEGATIVE
        prefix = f"test_platform/{job_id}/qwen_image"
        workflow = pipeline.qwen_workflow(
            input_name,
            positive,
            negative,
            int(job["seed"]),
            prefix,
        )
        write_json_atomic(job_dir / "workflow.json", workflow)
        update_image_job(job_id, stage="generating_image", progress=25)
        entry = pipeline.queue_and_wait(workflow, f"platform image {job_id}", timeout=7200)
        source = pipeline.first_output(
            entry,
            {".png", ".jpg", ".jpeg", ".webp"},
            "Platform Qwen image",
        )
        update_image_job(job_id, stage="processing_image", progress=85)
        raw_target = job_dir / "generated_raw.png"
        with Image.open(source) as loaded:
            loaded.convert("RGB").save(raw_target, format="PNG", optimize=True)
        if transparent:
            target = job_dir / "generated_transparent.png"
            pipeline.convert_chroma_image_to_transparent_png(
                raw_target,
                target,
                require_clean_background=True,
            )
        else:
            target = job_dir / "generated.png"
            shutil.copy2(raw_target, target)
        with Image.open(target) as loaded:
            width, height = loaded.size
        output = {
            "type": "transparent_image" if transparent else "image",
            "filename": target.name,
            "width": width,
            "height": height,
            "transparent": transparent,
            "model": pipeline.QWEN_MODEL,
            "style_lora": pipeline.QWEN_STYLE_LORA or None,
            "style_lora_strength": (
                pipeline.QWEN_STYLE_LORA_STRENGTH if pipeline.QWEN_STYLE_LORA else None
            ),
        }
        update_image_job(
            job_id,
            status="completed",
            stage="completed",
            progress=100,
            outputs=[output],
            finished_at=utc_now(),
            error=None,
        )
        LOGGER.info("Completed platform image job %s", job_id)
    except Exception as exc:
        LOGGER.exception("Platform image job %s failed", job_id)
        update_image_job(
            job_id,
            status="failed",
            stage="failed",
            error=f"{type(exc).__name__}: {exc}",
            finished_at=utc_now(),
        )


def image_worker() -> None:
    while not STOP_EVENT.is_set():
        try:
            job_id = IMAGE_QUEUE.get(timeout=1)
        except queue.Empty:
            continue
        try:
            run_image_job(job_id)
        finally:
            IMAGE_QUEUE.task_done()


def video_headers() -> dict[str, str]:
    if not LOCAL_API_KEY:
        raise HTTPException(status_code=503, detail="Local video API key is missing.")
    return {"X-API-Key": LOCAL_API_KEY}


def rewrite_video_job(job: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(job)
    result["kind"] = result.get("request_kind", "video")
    result["status_url"] = f"{PUBLIC_BASE_PATH}/api/jobs/video/{job['job_id']}"
    result["cancel_url"] = f"{PUBLIC_BASE_PATH}/api/jobs/video/{job['job_id']}/cancel"
    for output in result.get("outputs", []):
        output["download_url"] = (
            f"{PUBLIC_BASE_PATH}/api/jobs/video/{job['job_id']}/files/{output['filename']}"
        )
    return result


def history_filename(value: Any) -> str | None:
    filename = str(value or "").replace("\\", "/").rsplit("/", 1)[-1].strip()
    if not filename or Path(filename).name != filename:
        return None
    return filename[:255]


def history_number(value: Any) -> int | float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if number.is_integer() else number


def legacy_workflow_history(job_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"parameters": {}, "loras": []}
    path = job_dir / "workflow.json"
    if not path.is_file():
        return result
    try:
        workflow = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        LOGGER.warning("Unable to read legacy workflow metadata from %s", path)
        return result
    if not isinstance(workflow, dict):
        return result
    seen_loras: set[tuple[str, int | float | None]] = set()
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        class_type = str(node.get("class_type") or "")
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        if class_type == "KSampler":
            for source, target in (
                ("seed", "seed"),
                ("steps", "qwen_steps"),
                ("cfg", "qwen_cfg"),
                ("denoise", "qwen_denoise"),
                ("sampler_name", "qwen_sampler"),
                ("scheduler", "qwen_scheduler"),
            ):
                if inputs.get(source) is not None:
                    result["parameters"][target] = inputs[source]
        elif class_type == "UNETLoader" and inputs.get("unet_name"):
            result["parameters"]["qwen_model"] = history_filename(inputs["unet_name"])
        elif class_type.startswith("LoraLoader") and inputs.get("lora_name"):
            name = history_filename(inputs["lora_name"])
            strength = history_number(inputs.get("strength_model"))
            if name and (name, strength) not in seen_loras:
                seen_loras.add((name, strength))
                result["loras"].append(
                    {
                        "name": name,
                        "strength": strength,
                        "active": strength is None or strength != 0,
                        "role": "workflow",
                    }
                )
    return result


def history_user_loras(job: dict[str, Any]) -> list[dict[str, Any]]:
    raw_loras = job.get("requested_user_loras") or job.get("user_loras")
    if isinstance(raw_loras, str):
        try:
            raw_loras = json.loads(raw_loras)
        except json.JSONDecodeError:
            raw_loras = None
    if not isinstance(raw_loras, list):
        raw_loras = []
    if not raw_loras and job.get("user_lora"):
        raw_loras = [
            {
                "model_name": job.get("user_lora"),
                "original_filename": job.get("user_lora_original_filename"),
                "strength": job.get("user_lora_strength"),
            }
        ]
    result: list[dict[str, Any]] = []
    for index, item in enumerate(raw_loras):
        if not isinstance(item, dict):
            continue
        name = history_filename(item.get("original_filename")) or history_filename(
            item.get("model_name")
        )
        if not name:
            continue
        strength = history_number(item.get("strength"))
        result.append(
            {
                "name": name,
                "strength": strength,
                "active": strength is None or strength != 0,
                "order": index + 1,
                "role": "user",
            }
        )
    return result


def history_builtin_lora(job: dict[str, Any]) -> dict[str, Any] | None:
    name = history_filename(job.get("qwen_style_lora"))
    if not name:
        for output in job.get("outputs", []):
            if isinstance(output, dict):
                name = history_filename(output.get("style_lora"))
                if name:
                    break
    if not name:
        return None
    strength = history_number(job.get("qwen_style_lora_strength"))
    if strength is None:
        for output in job.get("outputs", []):
            if isinstance(output, dict) and output.get("style_lora_strength") is not None:
                strength = history_number(output.get("style_lora_strength"))
                break
    return {
        "name": name,
        "strength": strength,
        "pose_strength": history_number(job.get("qwen_pose_style_lora_strength")),
        "active": strength is None or strength != 0,
        "role": "builtin",
    }


def history_effective_lora_specs(raw_loras: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_loras, list):
        return []
    result: list[dict[str, Any]] = []
    for index, item in enumerate(raw_loras):
        if not isinstance(item, dict):
            continue
        name = history_filename(item.get("original_filename")) or history_filename(
            item.get("model_name")
        )
        if not name:
            continue
        strength = history_number(item.get("strength"))
        result.append(
            {
                "name": name,
                "strength": strength,
                "active": strength is None or strength != 0,
                "order": index + 1,
                "role": "effective",
            }
        )
    return result


def history_effective_lora(job: dict[str, Any]) -> dict[str, Any] | None:
    canonical_loras = history_effective_lora_specs(
        job.get("effective_canonical_user_loras")
    )
    canonical_scale = history_number(job.get("canonical_user_lora_strength_scale"))
    canonical_settings = job.get("effective_canonical_settings")
    if not isinstance(canonical_settings, dict):
        canonical_settings = {}
    canonical = None
    if canonical_loras or canonical_scale is not None:
        canonical = {
            "profile": job.get("cartoon_reference_profile"),
            "strength_scale": canonical_scale,
            "mode": canonical_settings.get("user_lora_mode"),
            "steps": canonical_settings.get("steps"),
            "cfg": canonical_settings.get("cfg"),
            "denoise": canonical_settings.get("denoise"),
            "loras": canonical_loras,
        }
    poses: list[dict[str, Any]] = []
    raw_pose_attempts = job.get("effective_pose_lora_attempts")
    if isinstance(raw_pose_attempts, list):
        for item in raw_pose_attempts:
            if not isinstance(item, dict):
                continue
            similarity = item.get("similarity")
            if not isinstance(similarity, dict):
                similarity = None
            poses.append(
                {
                    "pose_index": item.get("pose_index"),
                    "pose": item.get("pose"),
                    "attempt": item.get("attempt"),
                    "profile": item.get("profile"),
                    "strength_scale": history_number(
                        item.get("user_lora_strength_scale")
                    ),
                    "mode": item.get("user_lora_mode"),
                    "steps": item.get("steps"),
                    "cfg": item.get("cfg"),
                    "denoise": item.get("denoise"),
                    "quality_retry": item.get("quality_retry"),
                    "total_strength_cap": history_number(
                        item.get("total_strength_cap")
                    ),
                    "per_lora_strength_cap": history_number(
                        item.get("per_lora_strength_cap")
                    ),
                    "loras": history_effective_lora_specs(item.get("user_loras")),
                    "similarity": (
                        {
                            "available": similarity.get("available"),
                            "warning": bool(similarity.get("warning")),
                            "issues": similarity.get("issues")
                            if isinstance(similarity.get("issues"), list)
                            else [],
                            "comparisons": similarity.get("comparisons")
                            if isinstance(similarity.get("comparisons"), list)
                            else [],
                        }
                        if similarity
                        else None
                    ),
                }
            )
    if canonical is None and not poses:
        return None
    return {"canonical": canonical, "poses": poses}


def history_images(
    job: dict[str, Any],
    job_dir: Path,
    source: str,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for output in job.get("outputs", []):
        if not isinstance(output, dict):
            continue
        filename = history_filename(output.get("filename"))
        if not filename or filename in seen:
            continue
        output_type = str(output.get("type") or "image")
        suffix = Path(filename).suffix.lower()
        if (
            output_type not in HISTORY_IMAGE_OUTPUT_TYPES
            and suffix not in HISTORY_IMAGE_SUFFIXES
        ):
            continue
        if output_type in {"video", "video_preview"}:
            continue
        seen.add(filename)
        route_kind = "image" if source == "legacy" else "video"
        item = {
            "type": output_type,
            "media_kind": "image",
            "filename": filename,
            "download_url": (
                f"{PUBLIC_BASE_PATH}/api/jobs/{route_kind}/{job['job_id']}/files/{filename}"
            ),
            "available": (job_dir / filename).is_file(),
            "pose_index": output.get("pose_index"),
            "pose": output.get("pose"),
            "action": output.get("action"),
            "width": output.get("width"),
            "height": output.get("height"),
            "frames": output.get("frames"),
            "fps": output.get("fps"),
            "duration_seconds": output.get("duration_seconds"),
            "transparent": output.get("transparent"),
            "primary": output.get("primary"),
            "generation_engine": output.get("generation_engine"),
        }
        result.append(item)
    return result


def history_videos(
    job: dict[str, Any],
    job_dir: Path,
    source: str,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for output in job.get("outputs", []):
        if not isinstance(output, dict):
            continue
        filename = history_filename(output.get("filename"))
        if not filename or filename in seen:
            continue
        output_type = str(output.get("type") or "video")
        suffix = Path(filename).suffix.lower()
        if (
            output_type not in HISTORY_VIDEO_OUTPUT_TYPES
            and suffix not in HISTORY_VIDEO_SUFFIXES
        ):
            continue
        seen.add(filename)
        route_kind = "image" if source == "legacy" else "video"
        result.append(
            {
                "type": output_type,
                "media_kind": "video",
                "filename": filename,
                "download_url": (
                    f"{PUBLIC_BASE_PATH}/api/jobs/{route_kind}/{job['job_id']}/files/{filename}"
                ),
                "available": (job_dir / filename).is_file(),
                "pose_index": output.get("pose_index"),
                "pose": output.get("pose"),
                "action": output.get("action"),
                "width": output.get("width"),
                "height": output.get("height"),
                "frames": output.get("frames"),
                "fps": output.get("fps"),
                "duration_seconds": output.get("duration_seconds"),
                "primary": output.get("primary"),
                "video_engine": output.get("video_engine"),
                "generation_engine": output.get("generation_engine"),
            }
        )
    return result


def history_parameters(
    job: dict[str, Any],
    legacy_metadata: dict[str, Any],
) -> dict[str, Any]:
    parameters = {
        "seed": job.get("seed"),
        "pose_count": job.get("pose_count"),
        "poses": job.get("poses_original") or job.get("poses"),
        "poses_en": job.get("poses_en"),
        "image_profile": job.get("image_profile"),
        "image_profile_revision": job.get("image_profile_revision"),
        "pose_prompt_source": job.get("pose_prompt_source"),
        "pose_preset_ids": job.get("pose_preset_ids"),
        "pose_seed_offsets": job.get("pose_seed_offsets"),
        "identity_palette_guard": job.get("identity_palette_guard"),
        "identity_palette_policy": job.get("identity_palette_policy"),
        "source_identity_palette": job.get("source_identity_palette"),
        "effective_canonical_identity_palette": job.get(
            "effective_canonical_identity_palette"
        ),
        "canonical_identity_palette_verdict": job.get(
            "canonical_identity_palette_verdict"
        ),
        "pose_identity_palette_warning_count": job.get(
            "pose_identity_palette_warning_count"
        ),
        "resolution": job.get("resolution"),
        "width": job.get("width"),
        "height": job.get("height"),
        "output_format": job.get("output_format"),
        "skip_video": job.get("skip_video"),
        "generation_mode": job.get("generation_mode"),
        "actions": job.get("actions"),
        "subject_profile": job.get("subject_profile"),
        "requested_subject_source": job.get("requested_subject_source"),
        "effective_subject_source": job.get("effective_subject_source"),
        "driving_templates": job.get("driving_templates"),
        "generation_engine": job.get("generation_engine"),
        "scail_steps": job.get("scail_steps"),
        "pose_strength": job.get("pose_strength"),
        "subject_source": job.get("subject_source"),
        "qwen_steps": job.get("qwen_steps"),
        "qwen_cfg": job.get("qwen_cfg"),
        "qwen_denoise": job.get("qwen_denoise"),
        "qwen_sampler": job.get("qwen_sampler"),
        "qwen_scheduler": job.get("qwen_scheduler"),
        "qwen_style_profile": job.get("qwen_style_profile"),
        "qwen_model": job.get("qwen_model"),
        "wan_steps": job.get("wan_steps"),
        "wan_cfg": job.get("wan_cfg"),
        "wan_shift": job.get("wan_shift"),
        "frames": job.get("frames"),
        "fps": job.get("fps"),
        "duration_seconds": job.get("duration_seconds"),
    }
    for key, value in legacy_metadata.get("parameters", {}).items():
        if parameters.get(key) is None:
            parameters[key] = value
    if not parameters.get("resolution") and parameters.get("width") and parameters.get("height"):
        parameters["resolution"] = f"{parameters['width']}x{parameters['height']}"
    return parameters


def history_item(
    job: dict[str, Any],
    job_dir: Path,
    source: str,
    fallback_created_at: str,
) -> dict[str, Any] | None:
    job_id = str(job.get("job_id") or job_dir.name)
    if not job_id:
        return None
    job = dict(job)
    job["job_id"] = job_id
    legacy_metadata = legacy_workflow_history(job_dir) if source == "legacy" else {
        "parameters": {},
        "loras": [],
    }
    images = history_images(job, job_dir, source)
    videos = history_videos(job, job_dir, source)
    user_loras = history_user_loras(job)
    builtin_lora = history_builtin_lora(job)
    effective_lora = history_effective_lora(job)
    workflow_loras = legacy_metadata.get("loras", [])
    request_kind = str(
        job.get("request_kind")
        or ("image" if source == "legacy" else job.get("kind") or "video")
    ).lower()
    if request_kind not in {"image", "video"}:
        request_kind = "video"
    created_at = str(job.get("created_at") or fallback_created_at)
    parameters = history_parameters(job, legacy_metadata)
    prompt_original = str(
        job.get("prompt_original")
        or job.get("prompt")
        or ""
    )
    prompt_en = str(
        job.get("prompt_en")
        or (
            job.get("prompt")
            if not job.get("prompt_original")
            else ""
        )
        or ""
    )
    raw_poses = job.get("poses_original") or job.get("poses")
    raw_poses_en = job.get("poses_en")
    pose_search = (
        " ".join(str(pose) for pose in raw_poses)
        if isinstance(raw_poses, list)
        else str(raw_poses or "")
    )
    pose_search_en = (
        " ".join(str(pose) for pose in raw_poses_en)
        if isinstance(raw_poses_en, list)
        else str(raw_poses_en or "")
    )
    search_parts = [
        job_id,
        prompt_original,
        prompt_en,
        pose_search,
        pose_search_en,
        " ".join(str(action) for action in job.get("actions", []))
        if isinstance(job.get("actions"), list)
        else str(job.get("actions") or ""),
        str(job.get("subject_profile") or ""),
        str(job.get("generation_engine") or ""),
        " ".join(item["name"] for item in user_loras),
        " ".join(item["name"] for item in workflow_loras),
        builtin_lora["name"] if builtin_lora else "",
    ]
    return {
        "job_id": job_id,
        "source": source,
        "status": str(job.get("status") or "unknown"),
        "stage": str(job.get("stage") or ""),
        "created_at": created_at,
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "updated_at": job.get("updated_at"),
        "request_kind": request_kind,
        "prompt": prompt_original,
        "prompt_original": prompt_original,
        "prompt_en": prompt_en,
        "prompt_translation": (
            job.get("prompt_translation")
            if isinstance(job.get("prompt_translation"), dict)
            else None
        ),
        "image_profile": job.get("image_profile"),
        "image_profile_revision": job.get("image_profile_revision"),
        "image_profile_label": job.get("image_profile_label"),
        "pose_prompt_source": job.get("pose_prompt_source"),
        "pose_preset_ids": job.get("pose_preset_ids"),
        "pose_seed_offsets": job.get("pose_seed_offsets"),
        "identity_palette_guard": job.get("identity_palette_guard"),
        "identity_palette_policy": job.get("identity_palette_policy"),
        "source_identity_palette": job.get("source_identity_palette"),
        "effective_canonical_identity_palette": job.get(
            "effective_canonical_identity_palette"
        ),
        "canonical_identity_palette_verdict": job.get(
            "canonical_identity_palette_verdict"
        ),
        "pose_identity_palette_warning_count": job.get(
            "pose_identity_palette_warning_count"
        ),
        "user_loras": user_loras,
        "workflow_loras": workflow_loras,
        "builtin_lora": builtin_lora,
        "effective_lora": effective_lora,
        "similarity_warning_count": int(
            history_number(job.get("pose_similarity_warning_count")) or 0
        ),
        "user_lora_mode": job.get("user_lora_mode"),
        "has_active_user_lora": any(item["active"] for item in user_loras),
        "parameters": parameters,
        "images": images,
        "videos": videos,
        "media": [*images, *videos],
        "thumbnail": (
            next((item for item in images if item["available"]), images[0])
            if images
            else None
        ),
        "image_count": len(images),
        "available_image_count": sum(bool(item["available"]) for item in images),
        "video_count": len(videos),
        "available_video_count": sum(bool(item["available"]) for item in videos),
        "automatic_recovery_used": bool(job.get("automatic_recovery_used")),
        "background_removal_failed": bool(job.get("background_removal_failed")),
        "error": job.get("error"),
        "_search": " ".join(search_parts).casefold(),
    }


def load_history_items() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    sources = (
        (pipeline.OUTPUT_ROOT, "pipeline"),
        (OUTPUT_ROOT, "legacy"),
    )
    for root, source in sources:
        if not root.is_dir():
            continue
        for path in root.glob("*/status.json"):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
                fallback_created_at = datetime.fromtimestamp(
                    path.stat().st_mtime,
                    tz=timezone.utc,
                ).isoformat()
                item = history_item(job, path.parent, source, fallback_created_at)
                if item:
                    items.append(item)
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                LOGGER.exception("Unable to read history entry from %s", path)
    items.sort(
        key=lambda item: (item["created_at"], item["job_id"], item["source"]),
        reverse=True,
    )
    return items


def encode_history_cursor(item: dict[str, Any]) -> str:
    raw = json.dumps(
        [item["created_at"], item["job_id"], item["source"]],
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_history_cursor(cursor: str) -> tuple[str, str, str]:
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        value = json.loads(base64.urlsafe_b64decode(padded).decode("utf-8"))
        if not isinstance(value, list) or len(value) != 3:
            raise ValueError
        return str(value[0]), str(value[1]), str(value[2])
    except (ValueError, binascii.Error, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail="Invalid history cursor.") from exc


app = FastAPI(
    title="Pet Image and Video Test Platform",
    version=VERSION,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


@app.on_event("startup")
def startup() -> None:
    global WORKER
    restore_image_jobs()
    STOP_EVENT.clear()
    WORKER = threading.Thread(target=image_worker, name="platform-image-worker", daemon=True)
    WORKER.start()
    if not PLATFORM_PASSWORD:
        LOGGER.error("PET_PLATFORM_PASSWORD is not configured")


@app.on_event("shutdown")
def shutdown() -> None:
    STOP_EVENT.set()
    if WORKER and WORKER.is_alive():
        WORKER.join(timeout=5)


@app.get("/", response_class=HTMLResponse)
def index(_: None = Depends(require_auth)) -> HTMLResponse:
    if not HOST_PAGE.is_file():
        raise HTTPException(status_code=500, detail="Platform HTML is missing.")
    return HTMLResponse(HOST_PAGE.read_text(encoding="utf-8"))


@app.get("/history", response_class=HTMLResponse)
def history_page(_: None = Depends(require_auth)) -> HTMLResponse:
    if not HISTORY_PAGE.is_file():
        raise HTTPException(status_code=500, detail="History HTML is missing.")
    return HTMLResponse(HISTORY_PAGE.read_text(encoding="utf-8"))


@app.get("/api/health")
def health(_: None = Depends(require_auth)) -> dict[str, Any]:
    pipeline_health: dict[str, Any]
    try:
        response = requests.get(f"{LOCAL_API_URL}/api/local-video/health", timeout=10)
        response.raise_for_status()
        pipeline_health = response.json()
    except Exception as exc:
        pipeline_health = {"status": "unavailable", "error": str(exc)}
    return {
        "status": "ok" if pipeline_health.get("status") == "ok" else "degraded",
        "version": VERSION,
        "image_worker_alive": bool(WORKER and WORKER.is_alive()),
        "image_queue": {"pending": IMAGE_QUEUE.qsize(), "capacity": MAX_QUEUE},
        "pipeline": pipeline_health,
    }


@app.get("/api/subjects")
def subjects(_: None = Depends(require_auth)) -> Any:
    try:
        response = requests.get(
            f"{LOCAL_API_URL}/api/local-video/subjects",
            headers=video_headers(),
            timeout=30,
        )
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Generation API unavailable: {exc}",
        ) from exc
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text[:1000])
    try:
        return response.json()
    except requests.JSONDecodeError as exc:
        raise HTTPException(
            status_code=502,
            detail="Generation API returned an invalid subject profile response.",
        ) from exc


@app.get("/api/subjects/{profile_id}/preview")
def subject_preview(
    profile_id: str,
    _: None = Depends(require_auth),
) -> StreamingResponse:
    if (
        not profile_id
        or profile_id in {".", ".."}
        or Path(profile_id).name != profile_id
    ):
        raise HTTPException(status_code=400, detail="Invalid subject profile id.")
    encoded_profile_id = quote(profile_id, safe="")
    try:
        response = requests.get(
            f"{LOCAL_API_URL}/api/local-video/subjects/{encoded_profile_id}/preview",
            headers=video_headers(),
            stream=True,
            timeout=120,
        )
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Generation API unavailable: {exc}",
        ) from exc
    if response.status_code != 200:
        detail = response.text[:1000]
        response.close()
        raise HTTPException(status_code=response.status_code, detail=detail)
    media_type = response.headers.get("Content-Type", "image/png").split(";", 1)[0]
    return StreamingResponse(
        iter_remote(response),
        media_type=media_type,
        headers={
            "Content-Disposition": "inline",
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        },
        background=BackgroundTask(response.close),
    )


@app.get("/api/history")
def history(
    limit: int = Query(default=12, ge=1, le=48),
    cursor: str | None = Query(default=None, max_length=2048),
    q: str = Query(default="", max_length=200),
    kind: str = Query(default="all"),
    state: str = Query(default="all"),
    lora: str = Query(default="all"),
    _: None = Depends(require_auth),
) -> dict[str, Any]:
    kind = kind.strip().lower()
    state = state.strip().lower()
    lora = lora.strip().lower()
    if kind not in {"all", "image", "video"}:
        raise HTTPException(status_code=422, detail="kind must be all, image, or video.")
    if state not in {"all", "completed", "failed", "cancelled", "active"}:
        raise HTTPException(
            status_code=422,
            detail="state must be all, completed, failed, cancelled, or active.",
        )
    if lora not in {"all", "used", "none"}:
        raise HTTPException(status_code=422, detail="lora must be all, used, or none.")

    items = load_history_items()
    if kind != "all":
        items = [item for item in items if item["request_kind"] == kind]
    if state == "active":
        items = [item for item in items if item["status"] in {"queued", "running"}]
    elif state != "all":
        items = [item for item in items if item["status"] == state]
    if lora == "used":
        items = [item for item in items if item["has_active_user_lora"]]
    elif lora == "none":
        items = [item for item in items if not item["has_active_user_lora"]]
    query = q.strip().casefold()
    if query:
        items = [item for item in items if query in item["_search"]]

    total = len(items)
    if cursor:
        cursor_key = decode_history_cursor(cursor)
        items = [
            item
            for item in items
            if (item["created_at"], item["job_id"], item["source"]) < cursor_key
        ]
    has_more = len(items) > limit
    page = items[:limit]
    public_items: list[dict[str, Any]] = []
    for item in page:
        public_item = copy.deepcopy(item)
        public_item.pop("_search", None)
        public_items.append(public_item)
    return {
        "items": public_items,
        "count": len(public_items),
        "total": total,
        "limit": limit,
        "next_cursor": encode_history_cursor(page[-1]) if has_more and page else None,
        "has_more": has_more,
    }


@app.post("/api/jobs", status_code=status.HTTP_202_ACCEPTED)
async def create_job(
    image: UploadFile | None = File(default=None),
    image_left: UploadFile | None = File(default=None),
    image_right: UploadFile | None = File(default=None),
    lora: list[UploadFile] | None = File(default=None),
    prompt: str = Form(default=""),
    mode: str = Form(default="video"),
    generation_mode: str = Form(default="subject_replacement"),
    actions: str = Form(default='["eating"]'),
    subject_profile: str | None = Form(default=None),
    image_profile: str | None = Form(default=None),
    transparent: str = Form(default="true"),
    poses: str | None = Form(default=None),
    pose_count: int = Form(default=4),
    output_format: str = Form(default="transparent_gif"),
    skip_video: str = Form(default="false"),
    lora_strength: float = Form(default=1.0, ge=0.0, le=2.0),
    lora_strengths: str | None = Form(
        default=None,
        description="Optional JSON array matching the repeated lora file parts.",
    ),
    lora_mode: str = Form(default="replace"),
    qwen_steps: int | None = Form(default=None, ge=1, le=30),
    qwen_cfg: float = Form(default=1.0, ge=0.1, le=5.0),
    qwen_denoise: float = Form(default=1.0, ge=0.1, le=1.0),
    wan_steps: int = Form(default=pipeline.WAN_STEPS, ge=4, le=40),
    wan_cfg: float = Form(default=pipeline.WAN_CFG, ge=1.0, le=8.0),
    wan_shift: float = Form(default=pipeline.WAN_SHIFT, ge=1.0, le=12.0),
    scail_steps: int = Form(default=6, ge=4, le=12),
    pose_strength: float | None = Form(default=None, ge=0.0, le=2.0),
    seed: int = Form(default=20260726),
    _: None = Depends(require_auth),
) -> dict[str, Any]:
    generation_mode = generation_mode.strip().lower()
    if generation_mode not in {"subject_replacement", "keyframe_video"}:
        for upload in [image, image_left, image_right, *(lora or [])]:
            if upload is not None:
                await upload.close()
        raise HTTPException(
            status_code=422,
            detail="generation_mode must be subject_replacement or keyframe_video.",
        )
    prompt = prompt.strip()
    if not 0 <= seed <= 2**63 - 1:
        raise HTTPException(status_code=422, detail="Seed is out of range.")

    if generation_mode == "subject_replacement":
        for upload in lora or []:
            await upload.close()
        subject_files: dict[str, tuple[str, bytes, str]] | None = None
        identity_uploads = {
            "image": image,
            "image_left": image_left,
            "image_right": image_right,
        }
        present_parts = tuple(
            name
            for name, upload in identity_uploads.items()
            if upload is not None and bool(upload.filename)
        )
        try:
            if present_parts not in {
                (),
                ("image",),
                ("image", "image_left", "image_right"),
            }:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "Identity upload must contain either image only, or "
                        "the complete image + image_left + image_right set."
                    ),
                )
            if present_parts:
                subject_files = {}
                for part_name in present_parts:
                    upload = identity_uploads[part_name]
                    if upload is None:
                        raise RuntimeError("Resolved upload unexpectedly disappeared.")
                    raw = await upload.read(pipeline.MAX_UPLOAD_BYTES + 1)
                    if not raw:
                        raise HTTPException(
                            status_code=422,
                            detail=f"Uploaded {part_name} is empty.",
                        )
                    if len(raw) > pipeline.MAX_UPLOAD_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail=f"Uploaded {part_name} is too large.",
                        )
                    subject_files[part_name] = (
                        upload.filename or part_name,
                        raw,
                        upload.content_type or "application/octet-stream",
                    )
        finally:
            for upload in identity_uploads.values():
                if upload is not None:
                    await upload.close()

        data = {
            "generation_mode": "subject_replacement",
            "request_kind": "video",
            "prompt": prompt,
            "actions": actions.strip(),
            "resolution": "512x512",
            "output_format": output_format,
            "scail_steps": str(scail_steps),
            "seed": str(seed),
        }
        normalized_subject_profile = (subject_profile or "").strip()
        if normalized_subject_profile:
            data["subject_profile"] = normalized_subject_profile
        if pose_strength is not None:
            data["pose_strength"] = str(pose_strength)
        try:
            response = await asyncio.to_thread(
                requests.post,
                f"{LOCAL_API_URL}/api/local-video/jobs",
                headers=video_headers(),
                files=subject_files,
                data=data,
                timeout=120,
            )
        except requests.RequestException as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Generation API unavailable: {exc}",
            ) from exc
        if response.status_code != 202:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Generation API rejected the job: {response.text[:1000]}",
            )
        return rewrite_video_job(response.json())

    if any(
        upload is not None and bool(upload.filename)
        for upload in (image_left, image_right)
    ):
        for upload in [image, image_left, image_right, *(lora or [])]:
            if upload is not None:
                await upload.close()
        raise HTTPException(
            status_code=422,
            detail=(
                "image_left and image_right are supported only for "
                "generation_mode=subject_replacement."
            ),
        )
    for upload in (image_left, image_right):
        if upload is not None:
            await upload.close()

    mode = mode.strip().lower()
    if mode not in {"image", "video"}:
        raise HTTPException(status_code=422, detail="mode must be image or video.")
    if not prompt:
        raise HTTPException(status_code=422, detail="Prompt cannot be blank.")
    if image is None or not image.filename:
        if image is not None:
            await image.close()
        for upload in lora or []:
            await upload.close()
        raise HTTPException(
            status_code=422,
            detail="An uploaded image is required for keyframe_video generation.",
        )
    lora_mode = lora_mode.strip().lower()
    if lora_mode not in {"replace", "stack"}:
        raise HTTPException(status_code=422, detail="lora_mode must be replace or stack.")
    if image_profile is None:
        image_profile = pipeline.DEFAULT_IMAGE_PROFILE or None
    profile = pipeline.resolve_image_profile(
        image_profile,
        request_kind=mode,
    )
    lora_uploads = [upload for upload in (lora or []) if upload.filename]
    profile_lora_count = (
        len(profile["user_loras"])
        if profile and lora_mode == "stack" and lora_uploads
        else 0
    )
    available_upload_slots = pipeline.MAX_USER_LORAS - profile_lora_count
    if len(lora_uploads) > available_upload_slots:
        for upload in lora_uploads:
            await upload.close()
        raise HTTPException(
            status_code=422,
            detail=(
                f"At most {available_upload_slots} uploaded LoRAs may be used with "
                "the selected generation profile and fusion mode."
            ),
        )
    known_total_size = sum(
        int(upload.size)
        for upload in lora_uploads
        if getattr(upload, "size", None) is not None
    )
    if known_total_size > pipeline.MAX_LORA_FUSION_BYTES:
        for upload in lora_uploads:
            await upload.close()
        raise HTTPException(
            status_code=413,
            detail=(
                "Combined user LoRA files exceed "
                f"{pipeline.MAX_LORA_FUSION_BYTES} bytes."
            ),
        )
    if lora_strengths and lora_strengths.strip():
        try:
            parsed_strengths = json.loads(lora_strengths)
        except json.JSONDecodeError as exc:
            for upload in lora_uploads:
                await upload.close()
            raise HTTPException(
                status_code=422,
                detail="lora_strengths must be a valid JSON array.",
            ) from exc
        if not isinstance(parsed_strengths, list) or len(parsed_strengths) != len(lora_uploads):
            for upload in lora_uploads:
                await upload.close()
            raise HTTPException(
                status_code=422,
                detail="lora_strengths must contain one value for each uploaded LoRA.",
            )
        normalized_strengths: list[float] = []
        for index, value in enumerate(parsed_strengths):
            try:
                strength = float(value)
            except (TypeError, ValueError) as exc:
                for upload in lora_uploads:
                    await upload.close()
                raise HTTPException(
                    status_code=422,
                    detail=f"lora_strengths[{index}] must be a number.",
                ) from exc
            if not 0.0 <= strength <= 2.0:
                for upload in lora_uploads:
                    await upload.close()
                raise HTTPException(
                    status_code=422,
                    detail=f"lora_strengths[{index}] must be between 0.0 and 2.0.",
                )
            normalized_strengths.append(strength)
    else:
        normalized_strengths = [float(lora_strength)] * len(lora_uploads)
    raw = await image.read(pipeline.MAX_UPLOAD_BYTES + 1)
    filename = image.filename or "upload"
    content_type = image.content_type or "application/octet-stream"
    await image.close()
    if not raw:
        raise HTTPException(status_code=422, detail="Uploaded image is empty.")
    if len(raw) > pipeline.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Uploaded image is too large.")

    stored_user_loras: list[dict[str, Any]] = []
    stored_total_size = 0
    stored_sha256: set[str] = set()
    try:
        for upload, strength in zip(lora_uploads, normalized_strengths, strict=True):
            stored = await pipeline.store_user_lora(upload)
            if stored:
                stored_total_size += int(stored["size_bytes"])
                if stored_total_size > pipeline.MAX_LORA_FUSION_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "Combined user LoRA files exceed "
                            f"{pipeline.MAX_LORA_FUSION_BYTES} bytes."
                        ),
                    )
                if stored["sha256"] in stored_sha256:
                    raise HTTPException(
                        status_code=422,
                        detail="The same LoRA file cannot be fused more than once.",
                    )
                stored_sha256.add(stored["sha256"])
                stored_user_loras.append(
                    {
                        "model_name": stored["model_name"],
                        "original_filename": stored["original_filename"],
                        "strength": strength,
                    }
                )
    finally:
        for upload in lora_uploads:
            await upload.close()
    model_names = [item["model_name"] for item in stored_user_loras]
    if len(model_names) != len(set(model_names)):
        raise HTTPException(
            status_code=422,
            detail="The same LoRA file cannot be fused more than once.",
        )
    data: dict[str, str] = {
        "generation_mode": "keyframe_video",
        "prompt": prompt,
        "request_kind": mode,
        "pose_count": str(max(1, min(pose_count, pipeline.MAX_POSES))),
        "resolution": "512x512",
        "output_format": "transparent_gif" if mode == "image" else output_format,
        "skip_video": (
            "true"
            if mode == "image" or skip_video.strip().lower() in {"1", "true", "yes", "on"}
            else "false"
        ),
        "seed": str(seed),
        "qwen_cfg": str(qwen_cfg),
        "qwen_denoise": str(qwen_denoise),
        "wan_steps": str(wan_steps),
        "wan_cfg": str(wan_cfg),
        "wan_shift": str(wan_shift),
    }
    if poses and poses.strip():
        data["poses"] = poses.strip()
    if qwen_steps is not None:
        data["qwen_steps"] = str(qwen_steps)
    data["image_profile"] = image_profile or ""
    if stored_user_loras:
        data.update(
            {
                "user_loras": json.dumps(stored_user_loras, ensure_ascii=False),
                "user_lora_mode": lora_mode,
            }
        )
    try:
        response = await asyncio.to_thread(
            requests.post,
            f"{LOCAL_API_URL}/api/local-video/jobs",
            headers=video_headers(),
            files={"image": (filename, raw, content_type)},
            data=data,
            timeout=120,
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Generation API unavailable: {exc}") from exc
    if response.status_code != 202:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Generation API rejected the job: {response.text[:1000]}",
        )
    return rewrite_video_job(response.json())


@app.get("/api/jobs/image/{job_id}")
def image_job(job_id: str, _: None = Depends(require_auth)) -> dict[str, Any]:
    with IMAGE_JOBS_LOCK:
        job = IMAGE_JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Image job not found.")
        return public_image_job(job)


@app.get("/api/jobs/video/{job_id}")
def video_job(job_id: str, _: None = Depends(require_auth)) -> dict[str, Any]:
    try:
        response = requests.get(
            f"{LOCAL_API_URL}/api/local-video/jobs/{job_id}",
            headers=video_headers(),
            timeout=30,
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Video API unavailable: {exc}") from exc
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text[:1000])
    return rewrite_video_job(response.json())


@app.post("/api/jobs/video/{job_id}/cancel")
def cancel_video_job(job_id: str, _: None = Depends(require_auth)) -> dict[str, Any]:
    try:
        response = requests.post(
            f"{LOCAL_API_URL}/api/local-video/jobs/{job_id}/cancel",
            headers=video_headers(),
            timeout=60,
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Video API unavailable: {exc}") from exc
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text[:1000])
    return rewrite_video_job(response.json())


@app.get("/api/jobs/image/{job_id}/files/{filename}")
def image_file(job_id: str, filename: str, _: None = Depends(require_auth)) -> FileResponse:
    if Path(filename).name != filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    with IMAGE_JOBS_LOCK:
        job = IMAGE_JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Image job not found.")
        allowed = {item["filename"] for item in job.get("outputs", [])}
    if filename not in allowed:
        raise HTTPException(status_code=404, detail="Output is not available.")
    path = OUTPUT_ROOT / job_id / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Output file is missing.")
    media_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type, headers={"Content-Disposition": "inline"})


def iter_remote(response: requests.Response) -> Iterator[bytes]:
    for chunk in response.iter_content(chunk_size=1024 * 1024):
        if chunk:
            yield chunk


@app.get("/api/jobs/video/{job_id}/files/{filename}")
def video_file(
    job_id: str,
    filename: str,
    _: None = Depends(require_auth),
) -> StreamingResponse:
    if Path(filename).name != filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    try:
        response = requests.get(
            f"{LOCAL_API_URL}/api/local-video/jobs/{job_id}/files/{filename}",
            headers=video_headers(),
            stream=True,
            timeout=120,
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Video API unavailable: {exc}") from exc
    if response.status_code != 200:
        detail = response.text[:1000]
        response.close()
        raise HTTPException(status_code=response.status_code, detail=detail)
    media_type = mimetypes.guess_type(filename)[0] or response.headers.get(
        "Content-Type", "application/octet-stream"
    )
    return StreamingResponse(
        iter_remote(response),
        media_type=media_type,
        headers={"Content-Disposition": "inline"},
        background=BackgroundTask(response.close),
    )
