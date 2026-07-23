from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Literal
from uuid import uuid4

import dashscope
import requests
from dashscope import MultiModalConversation
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageStat


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
DATA_ROOT = Path(os.environ.get("PET_PIPELINE_DATA_ROOT", ROOT)).expanduser().resolve()
ASSET_ROOT = Path(os.environ.get("PET_DRESSUP_ASSET_ROOT", DATA_ROOT / "assets")).expanduser().resolve()
OUTPUT_ROOT = Path(
    os.environ.get("PET_DRESSUP_OUTPUT_ROOT", DATA_ROOT / "outputs" / "refined")
).expanduser().resolve()
KEY_FILE = Path(os.environ.get("DASHSCOPE_API_KEY_FILE", WORKSPACE / "apikey.txt")).expanduser().resolve()
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


class RefineRequest(BaseModel):
    imageDataUrl: str = Field(..., description="Canvas PNG data URL from the browser.")
    outfit: list[str] = Field(default_factory=list)
    background: Literal["transparent", "studio", "mint", "peach"] = "transparent"
    model: str = "qwen-image-edit-plus"
    size: str = "1024*1536"
    seed: int = 20260703


class RefineResponse(BaseModel):
    jobId: str
    inputUrl: str
    rawUrl: str
    refinedUrl: str
    metadataUrl: str


app = FastAPI(title="Pet Dressup MVP")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def decode_data_url(data_url: str) -> bytes:
    if "," not in data_url:
        raise HTTPException(status_code=400, detail="imageDataUrl must be a data URL.")
    header, payload = data_url.split(",", 1)
    if not header.startswith("data:image/"):
        raise HTTPException(status_code=400, detail="Only image data URLs are supported.")
    try:
        return base64.b64decode(payload)
    except Exception as exc:  # pragma: no cover - defensive boundary
        raise HTTPException(status_code=400, detail="Invalid base64 image data.") from exc


def normalize_input_image(raw_bytes: bytes, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(".upload")
    temp_path.write_bytes(raw_bytes)
    try:
        image = Image.open(temp_path).convert("RGBA")
        image.save(output_path)
    finally:
        temp_path.unlink(missing_ok=True)


def read_api_key() -> str:
    key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if key:
        return key
    if KEY_FILE.exists():
        return KEY_FILE.read_text(encoding="utf-8").strip()
    raise RuntimeError("DASHSCOPE_API_KEY is not set and apikey.txt was not found.")


def image_to_data_url(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def call_qwen_image_edit(
    *,
    image_path: Path,
    api_key: str,
    prompt: str,
    negative_prompt: str,
    output_count: int,
    size: str,
    model: str,
    seed: int,
    endpoint: str,
):
    dashscope.base_http_api_url = endpoint
    messages = [
        {
            "role": "user",
            "content": [
                {"image": image_to_data_url(image_path)},
                {"text": prompt},
            ],
        }
    ]
    request_kwargs = {
        "api_key": api_key,
        "model": model,
        "messages": messages,
        "stream": False,
        "n": output_count,
        "watermark": False,
        "negative_prompt": negative_prompt,
        "prompt_extend": True,
        "seed": seed,
    }
    if model != "qwen-image-edit":
        request_kwargs["size"] = size
    response = MultiModalConversation.call(**request_kwargs)
    if getattr(response, "status_code", None) != 200:
        raise RuntimeError(
            f"DashScope image generation failed: status={getattr(response, 'status_code', None)} "
            f"code={getattr(response, 'code', None)} message={getattr(response, 'message', None)}"
        )
    return response


def extract_image_urls(response) -> list[str]:
    output = getattr(response, "output", None)
    choices = getattr(output, "choices", None) if output is not None else None
    if not choices:
        raise RuntimeError(f"DashScope response has no image choices: {response}")
    urls: list[str] = []
    for choice in choices:
        message = getattr(choice, "message", None) or choice.get("message", {})
        content = getattr(message, "content", None) or message.get("content", [])
        for item in content:
            url = item.get("image") if isinstance(item, dict) else None
            if url:
                urls.append(url)
    if not urls:
        raise RuntimeError(f"DashScope image results do not contain URLs: {choices}")
    return urls


def download_image(url: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, timeout=120) as response:
        response.raise_for_status()
        path.write_bytes(response.content)
    return path


def estimate_corner_color(image: Image.Image, patch_size: int = 28) -> tuple[int, int, int]:
    rgb = image.convert("RGB")
    width, height = rgb.size
    boxes = [
        (0, 0, patch_size, patch_size),
        (width - patch_size, 0, width, patch_size),
        (0, height - patch_size, patch_size, height),
        (width - patch_size, height - patch_size, width, height),
    ]
    channels = [[], [], []]
    for box in boxes:
        mean = ImageStat.Stat(rgb.crop(box)).mean
        for index, value in enumerate(mean):
            channels[index].append(value)
    return tuple(int(sum(values) / len(values)) for values in channels)


def remove_connected_flat_background(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    rgb = rgba.convert("RGB")
    bg_color = estimate_corner_color(rgba)
    diff = ImageChops.difference(rgb, Image.new("RGB", rgb.size, bg_color)).convert("L")
    near_background = diff.point(lambda value: 255 if value < 36 else 0)

    for point in [
        (0, 0),
        (near_background.width - 1, 0),
        (0, near_background.height - 1),
        (near_background.width - 1, near_background.height - 1),
    ]:
        if near_background.getpixel(point) == 255:
            ImageDraw.floodfill(near_background, point, 128, thresh=0)

    connected = near_background.point(lambda value: 255 if value == 128 else 0)
    connected = connected.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(1.1))
    alpha = ImageChops.subtract(rgba.getchannel("A"), connected)
    rgba.putalpha(alpha)
    return rgba


def build_refine_prompt(outfit: list[str], background: str) -> str:
    outfit_text = ", ".join(outfit) if outfit else "no extra outfit"
    background_line = (
        "Keep a clean transparent or plain studio-like background."
        if background == "transparent"
        else "Keep the simple background feeling, but do not add scenery or furniture."
    )
    return f"""Use the input image as the exact composition reference.
Create a polished 2D/2.5D cute pet avatar final render of the same cat wearing the same outfit: {outfit_text}.

Important:
- Preserve the cat identity, face, markings, body proportions, pose, gaze direction, and overall silhouette.
- Preserve the selected outfit design, colors, approximate size, and placement from the input image.
- Redraw the outfit so it naturally fits the pet body instead of looking like a flat sticker.
- Add believable fur occlusion around the collar, chest, paws, ears, and clothing edges.
- Match lighting, softness, outline thickness, shadow direction, and material style across pet and clothing.
- Smooth harsh cutout edges, make cloth folds follow the rounded body, and keep paws visibly in front where appropriate.
- Output exactly one full-body pet only.
- {background_line}

Do not change the outfit into a different costume. Do not add humans, extra animals, text, watermark, logo, room, furniture, or props."""


NEGATIVE_PROMPT = """extra animals, second pet, human, hand, text, watermark, logo, room, furniture, busy background, different costume, changed face, changed markings, cropped ears, cropped paws, cropped body, deformed body, extra legs, low quality, blurry"""


def make_transparent(raw_path: Path, refined_path: Path) -> None:
    try:
        from rembg import remove
    except Exception:
        remove_connected_flat_background(Image.open(raw_path)).save(refined_path)
        return

    image = Image.open(raw_path).convert("RGBA")
    cutout = remove(image).convert("RGBA")
    bbox = cutout.getbbox()
    if bbox:
        canvas = Image.new("RGBA", image.size, (255, 255, 255, 0))
        cropped = cutout.crop(bbox)
        cropped.thumbnail((image.width - 80, image.height - 80), Image.Resampling.LANCZOS)
        x = (image.width - cropped.width) // 2
        y = (image.height - cropped.height) // 2
        canvas.alpha_composite(cropped, (x, y))
        cutout = canvas
    cutout.save(refined_path)


@app.post("/api/refine-outfit", response_model=RefineResponse)
def refine_outfit(payload: RefineRequest) -> RefineResponse:
    job_id = f"{int(time.time())}-{uuid4().hex[:8]}"
    job_dir = OUTPUT_ROOT / job_id
    input_path = job_dir / "input_composite.png"
    raw_path = job_dir / "raw_model_output.png"
    refined_path = job_dir / "refined.png"
    metadata_path = job_dir / "metadata.json"

    normalize_input_image(decode_data_url(payload.imageDataUrl), input_path)

    try:
        api_key = read_api_key()
        response = call_qwen_image_edit(
            image_path=input_path,
            api_key=api_key,
            prompt=build_refine_prompt(payload.outfit, payload.background),
            negative_prompt=NEGATIVE_PROMPT,
            output_count=1,
            size=payload.size,
            model=payload.model,
            seed=payload.seed,
            endpoint="https://dashscope.aliyuncs.com/api/v1",
        )
        urls = extract_image_urls(response)
        download_image(urls[0], raw_path)
        if payload.background == "transparent":
            make_transparent(raw_path, refined_path)
        else:
            Image.open(raw_path).convert("RGBA").save(refined_path)

        metadata = {
            "job_id": job_id,
            "model": payload.model,
            "size": payload.size,
            "seed": payload.seed,
            "outfit": payload.outfit,
            "background": payload.background,
            "request_id": getattr(response, "request_id", None),
            "input": str(input_path),
            "raw": str(raw_path),
            "refined": str(refined_path),
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Relative URLs keep the application functional behind an Nginx path prefix.
    base = f"refined_outputs/{job_id}"
    return RefineResponse(
        jobId=job_id,
        inputUrl=f"{base}/input_composite.png",
        rawUrl=f"{base}/raw_model_output.png",
        refinedUrl=f"{base}/refined.png",
        metadataUrl=f"{base}/metadata.json",
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "index.html")


app.mount("/assets", StaticFiles(directory=ASSET_ROOT), name="assets")
app.mount("/data", StaticFiles(directory=ROOT / "data"), name="data")
app.mount("/refined_outputs", StaticFiles(directory=OUTPUT_ROOT), name="refined_outputs")
app.mount("/", StaticFiles(directory=ROOT, html=True), name="static")
