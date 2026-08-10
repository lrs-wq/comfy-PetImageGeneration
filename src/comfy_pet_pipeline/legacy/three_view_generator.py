from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import dashscope
from PIL import Image
from dashscope import MultiModalConversation

from .pose_generator import (
    DEFAULT_NEGATIVE_PROMPT,
    call_qwen_image_edit,
    download_image,
    extract_image_urls,
    image_to_data_url,
    read_api_key,
    remove_backgrounds,
)


THREE_VIEW_PROMPT = """Use the input photo only as identity reference for the cat.
Create one cute 3D plush cartoon character image of the same cat.
Preserve the identity: fluffy long fur, mostly white body and chest, gray-brown tabby forehead and ears, soft gray facial mask, pink nose, large glossy eyes.
Strict character modeling reference view: orthographic camera, no perspective, no tilted camera, neutral standing pose, full body visible from ears to paws and tail, all paws grounded, centered, clean silhouette.
Transparent background if supported; otherwise use pure white background for clean cutout.
Exactly one complete cat only, no props, no text, no watermark.
High quality, cute, soft plush toy style."""


THREE_VIEW_NEGATIVE_PROMPT = (
    DEFAULT_NEGATIVE_PROMPT
    + ", perspective view, three quarter view, angled view, tilted head, dynamic pose, sitting pose, lying pose, "
    + "cropped paws, cropped ears, cropped tail, asymmetrical camera, labels, annotations, ruler, turntable, "
    + "multiple views in one image"
)


VIEWS = [
    {
        "name": "front",
        "description": (
            "STRICT FRONT ORTHOGRAPHIC VIEW. The cat faces directly forward, body centered and symmetrical, "
            "front paws visible and evenly placed, head not turned, same neutral standing pose."
        ),
    },
    {
        "name": "side",
        "description": (
            "STRICT LEFT SIDE ORTHOGRAPHIC PROFILE VIEW. The cat is exactly side-on, head and body both in perfect profile, "
            "no front-facing eyes, no three-quarter angle, four paws on the same ground line, tail visible in profile."
        ),
    },
    {
        "name": "back",
        "description": (
            "STRICT BACK ORTHOGRAPHIC VIEW. The cat faces completely away from the viewer, back of head, ears, body and tail visible, "
            "no face, no eyes, no nose, no mouth, same neutral standing pose."
        ),
    },
]


@dataclass(frozen=True)
class ThreeViewResult:
    raw_paths: list[Path]
    transparent_paths: list[Path]
    sheet_path: Path
    metadata_path: Path


def collect_reference_images(
    *,
    primary_image: Path,
    reference_dir: Path | None,
    reference_frames: str,
    max_reference_images: int,
) -> list[Path]:
    images = [primary_image]
    if reference_dir and reference_dir.exists():
        if reference_frames.strip():
            wanted = [item.strip() for item in reference_frames.split(",") if item.strip()]
            for name in wanted:
                path = reference_dir / name
                if not path.suffix:
                    path = path.with_suffix(".jpg")
                if path.exists() and path not in images:
                    images.append(path)
        else:
            candidates = sorted(
                [
                    *reference_dir.glob("*.jpg"),
                    *reference_dir.glob("*.jpeg"),
                    *reference_dir.glob("*.png"),
                ]
            )
            if candidates:
                slots = max(1, max_reference_images - 1)
                if len(candidates) <= slots:
                    sampled = candidates
                else:
                    sampled = [candidates[round(i * (len(candidates) - 1) / (slots - 1))] for i in range(slots)]
                for path in sampled:
                    if path not in images:
                        images.append(path)
    return images[:max_reference_images]


def call_qwen_image_edit_multi_reference(
    *,
    image_paths: list[Path],
    api_key: str,
    prompt: str,
    negative_prompt: str,
    output_count: int,
    size: str,
    model: str,
    seed: int,
    endpoint: str,
):
    if len(image_paths) == 1:
        return call_qwen_image_edit(
            image_path=image_paths[0],
            api_key=api_key,
            prompt=prompt,
            negative_prompt=negative_prompt,
            output_count=output_count,
            size=size,
            model=model,
            seed=seed,
            endpoint=endpoint,
        )
    dashscope.base_http_api_url = endpoint
    content = [{"image": image_to_data_url(path)} for path in image_paths]
    content.append({"text": prompt})
    request_kwargs = {
        "api_key": api_key,
        "model": model,
        "messages": [{"role": "user", "content": content}],
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


def build_three_view_sheet(image_paths: list[Path], sheet_path: Path, cell_size: int = 768) -> Path:
    sheet = Image.new("RGBA", (cell_size * len(image_paths), cell_size), (255, 255, 255, 0))
    for index, image_path in enumerate(image_paths):
        image = Image.open(image_path).convert("RGBA")
        bbox = image.getbbox()
        if bbox:
            image = image.crop(bbox)
        target_height = cell_size - 96
        scale = target_height / image.height
        target_width = min(cell_size - 96, int(image.width * scale))
        target_height = int(image.height * (target_width / image.width))
        image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        x = index * cell_size + (cell_size - image.width) // 2
        y = cell_size - image.height - 48
        sheet.alpha_composite(image, (x, y))
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(sheet_path)
    return sheet_path


def generate_three_view_sheet(
    *,
    image_path: Path,
    key_file: Path | None,
    output_dir: Path,
    reference_dir: Path | None = None,
    reference_frames: str = "",
    max_reference_images: int = 6,
    size: str = "1024*1024",
    model: str = "qwen-image-edit-plus",
    seed: int = 20260703,
    endpoint: str = "https://dashscope.aliyuncs.com/api/v1",
    prompt: str = THREE_VIEW_PROMPT,
    negative_prompt: str = THREE_VIEW_NEGATIVE_PROMPT,
) -> ThreeViewResult:
    api_key = read_api_key(key_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    transparent_dir = output_dir / "transparent"
    reference_images = collect_reference_images(
        primary_image=image_path,
        reference_dir=reference_dir,
        reference_frames=reference_frames,
        max_reference_images=max_reference_images,
    )

    raw_paths: list[Path] = []
    request_ids: list[str | None] = []
    for index, view in enumerate(VIEWS, start=1):
        view_prompt = (
            "The attached reference images are multiple frames of the same real cat. "
            "Use them only to lock identity, fur distribution, forehead tabby markings, gray-brown mask, white chest, body color and tail character.\n"
            f"{prompt}\n"
            f"View requirement for this image: {view['description']}\n"
            "Strict requirement: output exactly one cat only, not a collage, not a contact sheet, not multiple variations."
        )
        response = call_qwen_image_edit_multi_reference(
            image_paths=reference_images,
            api_key=api_key,
            prompt=view_prompt,
            negative_prompt=negative_prompt,
            output_count=1,
            size=size,
            model=model,
            seed=seed + index,
            endpoint=endpoint,
        )
        request_ids.append(getattr(response, "request_id", None))
        url = extract_image_urls(response)[0]
        raw_paths.append(download_image(url, raw_dir / f"{view['name']}_raw.png"))

    transparent_paths = remove_backgrounds(raw_paths, transparent_dir)
    renamed_transparent_paths: list[Path] = []
    for view, path in zip(VIEWS, transparent_paths):
        renamed = transparent_dir / f"{view['name']}_transparent.png"
        path.replace(renamed)
        renamed_transparent_paths.append(renamed)

    sheet_path = build_three_view_sheet(
        renamed_transparent_paths,
        output_dir / "cat_strict_three_view_transparent_sheet.png",
    )
    metadata_path = output_dir / "three_view_generation_metadata.json"
    metadata = {
        "model": model,
        "size": size,
        "seed": seed,
        "endpoint": endpoint,
        "input_image": str(image_path),
        "reference_images": [str(path) for path in reference_images],
        "raw_images": [str(path) for path in raw_paths],
        "transparent_images": [str(path) for path in renamed_transparent_paths],
        "sheet": str(sheet_path),
        "request_ids": request_ids,
        "views": VIEWS,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return ThreeViewResult(raw_paths, renamed_transparent_paths, sheet_path, metadata_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate strict transparent cat front/side/back model views.")
    parser.add_argument("--image", type=Path, default=Path("1.jpg"))
    parser.add_argument("--key-file", type=Path, default=Path("apikey.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cat_poses_comfy/three_view"))
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--reference-frames", default="")
    parser.add_argument("--max-reference-images", type=int, default=6)
    parser.add_argument("--size", default="1024*1024")
    parser.add_argument("--model", default="qwen-image-edit-plus")
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--endpoint", default="https://dashscope.aliyuncs.com/api/v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = generate_three_view_sheet(
        image_path=args.image.resolve(),
        key_file=args.key_file.resolve(),
        output_dir=args.output_dir.resolve(),
        reference_dir=args.reference_dir.resolve() if args.reference_dir else None,
        reference_frames=args.reference_frames,
        max_reference_images=args.max_reference_images,
        size=args.size,
        model=args.model,
        seed=args.seed,
        endpoint=args.endpoint,
    )
    print(f"sheet={result.sheet_path}")
    print(f"metadata={result.metadata_path}")


if __name__ == "__main__":
    main()
