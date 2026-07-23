from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import dashscope
import requests
from dashscope import MultiModalConversation
from PIL import Image
from rembg import remove


DEFAULT_PROMPT = """Use the input photo only as identity reference for the cat.
Create one cute 3D plush cartoon character image of the same cat.
Preserve the identity: fluffy long fur, mostly white body and chest, gray-brown tabby forehead and ears, soft gray facial mask, pink nose, large glossy eyes.
Exactly one complete cat only, centered, full body visible, no props, no text, no watermark.
Do not create a multi-pose sheet. Do not place multiple cats in one image.
Transparent background if supported; otherwise use pure white background for clean cutout.
High quality, cute, soft plush toy style."""

DEFAULT_NEGATIVE_PROMPT = """multiple cats, multi-pose sheet, collage, grid, background, scenery, room, furniture, floor, shadow, human, hand, extra animal, text, watermark, logo, cropped body, cropped tail, cropped ears, deformed legs, extra legs, extra tail, extra head, blurry, low quality"""

POSES = [
    "sitting front view, both front paws visible, cute calm expression",
    "standing side view, tail naturally extended, full body profile",
    "lying down on belly, front paws stretched forward",
    "stretching playfully, back arched, one front paw reaching forward",
    "sitting with one paw raised as if waving, pink paw pad visible",
    "curled sleeping pose, tail wrapped near body, peaceful expression",
]


@dataclass(frozen=True)
class PoseGenerationResult:
    raw_paths: list[Path]
    transparent_paths: list[Path]
    sheet_path: Path
    metadata_path: Path


def read_api_key(key_file: Path | None = None) -> str:
    key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if key:
        return key
    if key_file and key_file.exists():
        return key_file.read_text(encoding="utf-8").strip()
    raise RuntimeError("DASHSCOPE_API_KEY is not set and apikey.txt was not found.")


def image_to_data_url(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
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


def download_images(urls: Iterable[str], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, url in enumerate(urls, start=1):
        path = output_dir / f"pose_{index:02d}_raw.png"
        with requests.get(url, timeout=120) as response:
            response.raise_for_status()
            path.write_bytes(response.content)
        paths.append(path)
    return paths


def download_image(url: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, timeout=120) as response:
        response.raise_for_status()
        path.write_bytes(response.content)
    return path


def remove_backgrounds(raw_paths: Iterable[Path], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    transparent_paths: list[Path] = []
    for index, raw_path in enumerate(raw_paths, start=1):
        image = Image.open(raw_path).convert("RGBA")
        cutout = remove(image).convert("RGBA")
        bbox = cutout.getbbox()
        if bbox:
            cutout = cutout.crop(bbox)
        out_path = output_dir / f"pose_{index:02d}_transparent.png"
        cutout.save(out_path)
        transparent_paths.append(out_path)
    return transparent_paths


def build_transparent_sheet(image_paths: list[Path], sheet_path: Path, columns: int = 3, cell_size: int = 768) -> Path:
    rows = (len(image_paths) + columns - 1) // columns
    sheet = Image.new("RGBA", (columns * cell_size, rows * cell_size), (255, 255, 255, 0))
    for index, image_path in enumerate(image_paths):
        image = Image.open(image_path).convert("RGBA")
        image.thumbnail((cell_size - 80, cell_size - 80), Image.Resampling.LANCZOS)
        col = index % columns
        row = index // columns
        x = col * cell_size + (cell_size - image.width) // 2
        y = row * cell_size + (cell_size - image.height) // 2
        sheet.alpha_composite(image, (x, y))
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(sheet_path)
    return sheet_path


def generate_pose_sheet(
    *,
    image_path: Path,
    key_file: Path | None,
    output_dir: Path,
    output_count: int = 6,
    size: str = "1024*1024",
    model: str = "qwen-image-edit-plus",
    seed: int = 20260617,
    endpoint: str = "https://dashscope.aliyuncs.com/api/v1",
    prompt: str = DEFAULT_PROMPT,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
) -> PoseGenerationResult:
    api_key = read_api_key(key_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    transparent_dir = output_dir / "transparent"

    raw_paths: list[Path] = []
    request_ids: list[str | None] = []
    poses = POSES[:output_count]
    for index, pose in enumerate(poses, start=1):
        pose_prompt = (
            f"{prompt}\n"
            f"Pose for this image: {pose}.\n"
            "Strict requirement: output exactly one cat only, not a collage, not a contact sheet, not multiple variations."
        )
        response = call_qwen_image_edit(
            image_path=image_path,
            api_key=api_key,
            prompt=pose_prompt,
            negative_prompt=negative_prompt,
            output_count=1,
            size=size,
            model=model,
            seed=seed + index,
            endpoint=endpoint,
        )
        request_ids.append(getattr(response, "request_id", None))
        urls = extract_image_urls(response)
        raw_paths.append(download_image(urls[0], raw_dir / f"pose_{index:02d}_raw.png"))
    transparent_paths = remove_backgrounds(raw_paths, transparent_dir)
    sheet_path = build_transparent_sheet(
        transparent_paths,
        output_dir / "cat_6_poses_transparent_sheet.png",
    )
    metadata_path = output_dir / "generation_metadata.json"
    metadata = {
        "model": model,
        "size": size,
        "seed": seed,
        "endpoint": endpoint,
        "input_image": str(image_path),
        "raw_images": [str(path) for path in raw_paths],
        "transparent_images": [str(path) for path in transparent_paths],
        "sheet": str(sheet_path),
        "request_ids": request_ids,
        "poses": poses,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return PoseGenerationResult(raw_paths, transparent_paths, sheet_path, metadata_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate six transparent cartoon cat poses from a reference photo.")
    parser.add_argument("--image", type=Path, default=Path("1.jpg"))
    parser.add_argument("--key-file", type=Path, default=Path("apikey.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cat_poses"))
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--size", default="1024*1024")
    parser.add_argument("--model", default="qwen-image-edit-plus")
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--endpoint", default="https://dashscope.aliyuncs.com/api/v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = generate_pose_sheet(
        image_path=args.image.resolve(),
        key_file=args.key_file.resolve(),
        output_dir=args.output_dir.resolve(),
        output_count=args.count,
        size=args.size,
        model=args.model,
        seed=args.seed,
        endpoint=args.endpoint,
    )
    print(f"sheet={result.sheet_path}")
    print(f"metadata={result.metadata_path}")


if __name__ == "__main__":
    main()
