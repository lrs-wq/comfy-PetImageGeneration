from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import requests
from PIL import Image, ImageChops


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path("/data/comfy-pet-pipeline")
COMFY_URL = "http://127.0.0.1:8188"
COMFY_INPUT = DATA_ROOT / "comfyui" / "input"
COMFY_OUTPUT = DATA_ROOT / "comfyui" / "output"
MODEL_ROOT = DATA_ROOT / "models" / "comfyui"
TRIAL_NAME = "local_model_trial_20260713"
TRIAL_OUTPUT = DATA_ROOT / "outputs" / TRIAL_NAME

QWEN_MODEL = "qwen_image_edit_2511_fp8mixed.safetensors"
QWEN_CLIP = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
QWEN_VAE = "qwen_image_vae.safetensors"
QWEN_LORA = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"

WAN_MODEL = "wan2.2_ti2v_5B_fp16.safetensors"
WAN_CLIP = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
WAN_VAE = "wan2.2_vae.safetensors"

QWEN_PROMPT = """Use the input photograph as the exact identity reference for this cat.
Create one polished 3D plush cartoon image of the same cat eating dry cat food from a small blue ceramic bowl.
Preserve the cat's distinctive identity: fluffy long cream-white fur, gray-brown tabby forehead and ears, soft gray eye mask, white muzzle and chest, pink nose, dark glossy eyes, and the same facial proportions.
Show exactly one complete cat in a three-quarter side view, full body visible. The cat's head is lowered naturally toward the visible brown kibble in the bowl, ready to chew. Keep all four legs anatomically correct and the tail visible.
Clean pure white studio background, soft even light, stable centered composition, no text, no watermark, no other animals or people. Make this a clean first frame suitable for image-to-video animation."""

QWEN_NEGATIVE = """multiple cats, extra animal, human, hand, text, watermark, logo, room, furniture, cropped body, cropped ears, cropped tail, missing bowl, empty bowl, water bowl, deformed face, extra legs, blurry, low quality"""

WAN_PROMPT = """A cute 3D plush cartoon cat eats dry cat food from a small blue ceramic bowl. The same cat gently lowers and raises its head in small natural motions and softly chews the food. Its ears twitch slightly, whiskers move subtly, and the tail sways gently. Preserve the exact cat identity, fur markings, face, body proportions, bowl, and plush cartoon style from the first frame. Fixed camera, clean white studio background, full body remains centered and visible, smooth subtle motion, no scene cuts and no camera movement."""

WAN_NEGATIVE = """scene cut, camera movement, zoom, pan, rotation, extra cat, extra animal, human, hand, text, watermark, changing fur color, changing face, deformed body, deformed legs, extra legs, missing legs, cropped body, leaving frame, complex background, missing bowl, water bowl, violent motion, flicker"""


def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required local model files are missing:\n" + "\n".join(missing))


def link(output_node: str, slot: int = 0) -> list[Any]:
    return [output_node, slot]


def qwen_workflow(input_name: str, seed: int) -> dict[str, Any]:
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": input_name}},
        "2": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": QWEN_MODEL, "weight_dtype": "default"},
        },
        "3": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": QWEN_CLIP, "type": "qwen_image", "device": "default"},
        },
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": QWEN_VAE}},
        "5": {"class_type": "FluxKontextImageScale", "inputs": {"image": link("1")}},
        "6": {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {
                "clip": link("3"),
                "vae": link("4"),
                "image1": link("5"),
                "prompt": QWEN_PROMPT,
            },
        },
        "7": {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {
                "clip": link("3"),
                "vae": link("4"),
                "image1": link("5"),
                "prompt": QWEN_NEGATIVE,
            },
        },
        "8": {
            "class_type": "FluxKontextMultiReferenceLatentMethod",
            "inputs": {"conditioning": link("6"), "reference_latents_method": "index_timestep_zero"},
        },
        "9": {
            "class_type": "FluxKontextMultiReferenceLatentMethod",
            "inputs": {"conditioning": link("7"), "reference_latents_method": "index_timestep_zero"},
        },
        "10": {"class_type": "VAEEncode", "inputs": {"pixels": link("5"), "vae": link("4")}},
        "11": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {"model": link("2"), "shift": 3.1},
        },
        "12": {"class_type": "CFGNorm", "inputs": {"model": link("11"), "strength": 1.0}},
        "13": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": link("12"), "lora_name": QWEN_LORA, "strength_model": 1.0},
        },
        "14": {
            "class_type": "KSampler",
            "inputs": {
                "model": link("13"),
                "positive": link("8"),
                "negative": link("9"),
                "latent_image": link("10"),
                "seed": seed,
                "steps": 4,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        "15": {"class_type": "VAEDecode", "inputs": {"samples": link("14"), "vae": link("4")}},
        "16": {
            "class_type": "SaveImage",
            "inputs": {"images": link("15"), "filename_prefix": f"{TRIAL_NAME}/qwen_local_eating_keyframe"},
        },
    }


def wan_workflow(input_name: str, seed: int, width: int, height: int, length: int) -> dict[str, Any]:
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
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": link("2"), "text": WAN_PROMPT}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": link("2"), "text": WAN_NEGATIVE}},
        "7": {
            "class_type": "Wan22ImageToVideoLatent",
            "inputs": {
                "vae": link("3"),
                "start_image": link("4"),
                "width": width,
                "height": height,
                "length": length,
                "batch_size": 1,
            },
        },
        "8": {"class_type": "ModelSamplingSD3", "inputs": {"model": link("1"), "shift": 8.0}},
        "9": {
            "class_type": "KSampler",
            "inputs": {
                "model": link("8"),
                "positive": link("5"),
                "negative": link("6"),
                "latent_image": link("7"),
                "seed": seed,
                "steps": 20,
                "cfg": 5.0,
                "sampler_name": "uni_pc",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        "10": {"class_type": "VAEDecode", "inputs": {"samples": link("9"), "vae": link("3")}},
        "11": {"class_type": "CreateVideo", "inputs": {"images": link("10"), "fps": 24.0}},
        "12": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": link("11"),
                "filename_prefix": f"{TRIAL_NAME}/wan22_local_eating_{width}x{height}_{length}f",
                "format": "auto",
                "codec": "auto",
            },
        },
    }


def queue_and_wait(workflow: dict[str, Any], stage: str, timeout: int) -> dict[str, Any]:
    client_id = str(uuid4())
    response = requests.post(
        f"{COMFY_URL}/prompt",
        json={"prompt": workflow, "client_id": client_id},
        timeout=60,
    )
    if response.status_code != 200:
        raise RuntimeError(f"ComfyUI rejected {stage} workflow: {response.status_code} {response.text}")
    prompt_id = response.json()["prompt_id"]
    print(f"{stage}: queued prompt_id={prompt_id}", flush=True)
    deadline = time.monotonic() + timeout
    last_report = 0.0
    while time.monotonic() < deadline:
        history_response = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=30)
        history_response.raise_for_status()
        history = history_response.json()
        if prompt_id in history:
            entry = history[prompt_id]
            status = entry.get("status", {})
            if status.get("completed"):
                print(f"{stage}: completed prompt_id={prompt_id}", flush=True)
                return entry
            messages = status.get("messages", [])
            if any(message and message[0] == "execution_error" for message in messages):
                raise RuntimeError(f"ComfyUI {stage} execution failed: {json.dumps(messages, ensure_ascii=False)}")
        now = time.monotonic()
        if now - last_report >= 30:
            queue = requests.get(f"{COMFY_URL}/queue", timeout=30).json()
            print(
                f"{stage}: running elapsed={int(timeout - (deadline - now))}s "
                f"queue_running={len(queue.get('queue_running', []))} "
                f"queue_pending={len(queue.get('queue_pending', []))}",
                flush=True,
            )
            last_report = now
        time.sleep(5)
    raise TimeoutError(f"ComfyUI {stage} timed out after {timeout} seconds")


def free_comfy_models() -> None:
    response = requests.post(
        f"{COMFY_URL}/free",
        json={"unload_models": True, "free_memory": True},
        timeout=60,
    )
    response.raise_for_status()
    print("comfyui: unloaded cached models", flush=True)


def output_files(entry: dict[str, Any]) -> list[Path]:
    found: list[Path] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if "filename" in value:
                output_type = value.get("type", "output")
                if output_type == "output":
                    found.append(COMFY_OUTPUT / value.get("subfolder", "") / value["filename"])
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(entry.get("outputs", {}))
    return list(dict.fromkeys(found))


def write_workflow(name: str, workflow: dict[str, Any]) -> None:
    TRIAL_OUTPUT.mkdir(parents=True, exist_ok=True)
    (TRIAL_OUTPUT / name).write_text(
        json.dumps(workflow, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def prepare_landscape_keyframe(source: Path, target: Path, size: tuple[int, int] = (832, 480)) -> None:
    image = Image.open(source).convert("RGBA")
    flattened = Image.new("RGBA", image.size, (255, 255, 255, 255))
    flattened.alpha_composite(image)
    rgb = flattened.convert("RGB")
    difference = ImageChops.difference(rgb, Image.new("RGB", rgb.size, (255, 255, 255))).convert("L")
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
    canvas = Image.new("RGBA", size, (255, 255, 255, 255))
    image.thumbnail((int(size[0] * 0.88), int(size[1] * 0.90)), Image.Resampling.LANCZOS)
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.alpha_composite(image, (x, y))
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(target)


def prepare_existing_video_input(width: int = 832, height: int = 480) -> Path:
    final_image = TRIAL_OUTPUT / "qwen_local_eating_keyframe.png"
    if not final_image.is_file():
        raise FileNotFoundError(f"Run the image stage first; missing {final_image}")
    landscape = TRIAL_OUTPUT / f"wan_input_{width}x{height}.png"
    prepare_landscape_keyframe(final_image, landscape, size=(width, height))
    comfy_landscape = COMFY_INPUT / TRIAL_NAME / landscape.name
    comfy_landscape.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(landscape, comfy_landscape)
    print(f"qwen-image: wan_input={landscape}", flush=True)
    return landscape


def run_image(seed: int) -> Path:
    require_files(
        [
            MODEL_ROOT / "diffusion_models" / QWEN_MODEL,
            MODEL_ROOT / "text_encoders" / QWEN_CLIP,
            MODEL_ROOT / "vae" / QWEN_VAE,
            MODEL_ROOT / "loras" / QWEN_LORA,
        ]
    )
    source = DATA_ROOT / "inputs" / "1.jpg"
    input_relative = Path(TRIAL_NAME) / "original_cat.jpg"
    input_path = COMFY_INPUT / input_relative
    input_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, input_path)
    workflow = qwen_workflow(input_relative.as_posix(), seed)
    write_workflow("qwen_workflow_api.json", workflow)
    free_comfy_models()
    entry = queue_and_wait(workflow, "qwen-image", timeout=7200)
    images = [path for path in output_files(entry) if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}]
    if not images:
        raise RuntimeError(f"Qwen workflow completed without an image output: {json.dumps(entry, ensure_ascii=False)}")
    final_image = TRIAL_OUTPUT / "qwen_local_eating_keyframe.png"
    shutil.copy2(images[0], final_image)
    landscape = prepare_existing_video_input()
    print(f"qwen-image: output={final_image}", flush=True)
    free_comfy_models()
    return landscape


def run_video(seed: int, width: int = 832, height: int = 480, length: int = 49) -> Path:
    require_files(
        [
            MODEL_ROOT / "diffusion_models" / WAN_MODEL,
            MODEL_ROOT / "text_encoders" / WAN_CLIP,
            MODEL_ROOT / "vae" / WAN_VAE,
        ]
    )
    landscape = TRIAL_OUTPUT / f"wan_input_{width}x{height}.png"
    if not landscape.is_file():
        prepare_existing_video_input(width, height)
    input_relative = Path(TRIAL_NAME) / landscape.name
    comfy_landscape = COMFY_INPUT / input_relative
    comfy_landscape.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(landscape, comfy_landscape)
    workflow = wan_workflow(input_relative.as_posix(), seed, width, height, length)
    write_workflow(f"wan_workflow_api_{width}x{height}_{length}f.json", workflow)
    free_comfy_models()
    entry = queue_and_wait(workflow, "wan-video", timeout=21600)
    videos = [path for path in output_files(entry) if path.suffix.lower() in {".mp4", ".webm", ".mov", ".mkv"}]
    if not videos:
        raise RuntimeError(f"Wan workflow completed without a video output: {json.dumps(entry, ensure_ascii=False)}")
    final_video = TRIAL_OUTPUT / f"wan22_local_eating_{width}x{height}_{length}f.mp4"
    shutil.copy2(videos[0], final_video)
    print(f"wan-video: output={final_video}", flush=True)
    free_comfy_models()
    return final_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local Qwen image-edit and Wan2.2 I2V smoke tests through ComfyUI.")
    parser.add_argument("--stage", choices=("image", "prepare", "video", "free", "all"), default="all")
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--length", type=int, default=49)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.width % 32 or args.height % 32:
        raise ValueError("Wan width and height must be multiples of 32")
    if args.length < 1 or (args.length - 1) % 4:
        raise ValueError("Wan frame length must equal 4n+1")
    TRIAL_OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.stage in {"image", "all"}:
        run_image(args.seed)
    if args.stage == "prepare":
        prepare_existing_video_input(args.width, args.height)
    if args.stage in {"video", "all"}:
        run_video(args.seed + 1, args.width, args.height, args.length)
    if args.stage == "free":
        free_comfy_models()


if __name__ == "__main__":
    main()
