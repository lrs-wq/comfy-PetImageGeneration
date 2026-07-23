from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "outputs" / "cat_poses_comfy" / "transparent" / "pose_02_transparent.png"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "cat_poses_clean_alpha" / "transparent"


def largest_component(mask: np.ndarray) -> np.ndarray:
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if labels_count <= 1:
        return mask.astype(bool)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return labels == largest


def clean_mask(alpha: np.ndarray) -> np.ndarray:
    mask = alpha > 12
    mask = largest_component(mask)

    # Remove 1-3 px alpha whisker strokes and loose semi-transparent edge dust.
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_u8 = (mask.astype(np.uint8) * 255)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, open_kernel, iterations=1)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    mask_u8 = cv2.medianBlur(mask_u8, 5)
    mask = largest_component(mask_u8 > 127)

    # Shrink the final silhouette slightly to remove black premultiplied edge pixels.
    recover_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_u8 = cv2.erode(mask.astype(np.uint8) * 255, recover_kernel, iterations=2)
    return largest_component(mask_u8 > 127)


def repair_rgb(rgb: np.ndarray, original_alpha: np.ndarray, clean_alpha: np.ndarray) -> np.ndarray:
    clean_mask_u8 = clean_alpha.astype(np.uint8) * 255
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # Low-alpha edge pixels often carry a black matte. Replace them from nearby solid pixels.
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    dist = cv2.distanceTransform(clean_mask_u8, cv2.DIST_L2, 5)
    boundary_band = clean_alpha & (dist < 8)
    dark_boundary = boundary_band & (gray < 95)
    bad_edge = (((original_alpha < 230) & clean_alpha) | dark_boundary).astype(np.uint8) * 255
    bad_edge = cv2.dilate(bad_edge, np.ones((3, 3), np.uint8), iterations=1)
    repaired = cv2.inpaint(rgb_bgr, bad_edge, 3, cv2.INPAINT_TELEA)
    repaired = cv2.cvtColor(repaired, cv2.COLOR_BGR2RGB)

    # Keep transparent pixels neutral so later resampling cannot pull black into the edge.
    repaired[~clean_alpha] = (255, 255, 255)
    return repaired


def crop_and_center(image: Image.Image, size: int) -> Image.Image:
    bbox = image.getbbox()
    if bbox:
        image = image.crop(bbox)
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    canvas.alpha_composite(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def composite_on_white(image: Image.Image) -> Image.Image:
    bg = Image.new("RGB", image.size, "white")
    bg.paste(image.convert("RGB"), mask=image.getchannel("A"))
    return bg


def save_debug(original: Image.Image, cleaned: Image.Image, output_path: Path) -> None:
    thumb_w = 512
    original_view = composite_on_white(original)
    cleaned_view = composite_on_white(cleaned)
    original_view.thumbnail((thumb_w, thumb_w), Image.Resampling.LANCZOS)
    cleaned_view.thumbnail((thumb_w, thumb_w), Image.Resampling.LANCZOS)
    sheet = Image.new("RGB", (thumb_w * 2, thumb_w + 46), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((16, 16), "original alpha", fill=(30, 30, 30))
    draw.text((thumb_w + 16, 16), "clean alpha", fill=(30, 30, 30))
    sheet.paste(original_view, (0, 46))
    sheet.paste(cleaned_view, (thumb_w, 46))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=94)


def clean_image(input_path: Path, output_path: Path, size: int) -> None:
    source = crop_and_center(Image.open(input_path).convert("RGBA"), size)
    arr = np.asarray(source, dtype=np.uint8)
    rgb = arr[..., :3]
    alpha = arr[..., 3]
    alpha_f = alpha[..., None].astype(np.float32) / 255.0
    rgb = np.clip(rgb.astype(np.float32) * alpha_f + 255.0 * (1.0 - alpha_f), 0, 255).astype(np.uint8)

    mask = clean_mask(alpha)
    repaired_rgb = repair_rgb(rgb.copy(), alpha, mask)
    out = np.zeros((size, size, 4), dtype=np.uint8)
    out[..., :3] = repaired_rgb
    out[..., 3] = mask.astype(np.uint8) * 255

    cleaned = Image.fromarray(out, "RGBA")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.save(output_path)
    save_debug(source, cleaned, output_path.with_name(output_path.stem + "_alpha_compare.jpg"))

    print(f"source={input_path}")
    print(f"output={output_path}")
    print(f"source_alpha_pixels={(alpha > 0).sum()}")
    print(f"clean_alpha_pixels={mask.sum()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--name", default="pose_02_clean_alpha")
    parser.add_argument("--size", type=int, default=1024)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    clean_image(args.input, args.output_dir / f"{args.name}.png", args.size)
