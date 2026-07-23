from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image, ImageDraw

from improve_hunyuan2mini_vertex_texture_cat import save_preview


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = ROOT / "outputs" / "cat_3d_hunyuan2mini_turbo_hunyuan_texgen" / "pose_02_transparent"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "cat_3d_hunyuan2mini_turbo_hunyuan_texgen_refined" / "pose_02_transparent"


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Unable to load mesh from {path}")
    return mesh


def clean_tiny_face_components(mesh: trimesh.Trimesh, max_faces: int, max_diag: float) -> trimesh.Trimesh:
    if max_faces <= 0:
        return mesh.copy()

    face_adjacency = mesh.face_adjacency
    components = trimesh.graph.connected_components(
        face_adjacency,
        nodes=np.arange(len(mesh.faces)),
        min_len=1,
    )
    keep_faces = np.ones(len(mesh.faces), dtype=bool)
    removed_faces = 0
    for component in components:
        if len(component) > max_faces:
            continue
        verts_idx = np.unique(mesh.faces[component].reshape(-1))
        verts = mesh.vertices[verts_idx]
        diag = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
        if diag <= max_diag:
            keep_faces[component] = False
            removed_faces += len(component)

    cleaned = mesh.copy()
    cleaned.update_faces(keep_faces)
    cleaned.remove_unreferenced_vertices()
    cleaned.fix_normals()
    print(f"removed_faces={removed_faces} remaining_faces={len(cleaned.faces)}")
    return cleaned


def texture_image(mesh: trimesh.Trimesh, fallback_path: Path) -> Image.Image:
    material = getattr(mesh.visual, "material", None)
    image = getattr(material, "image", None)
    if image is None:
        image = Image.open(fallback_path)
    return image.convert("RGB")


def denoise_texture(image: Image.Image) -> Image.Image:
    arr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    denoised = cv2.fastNlMeansDenoisingColored(arr, None, 3, 3, 7, 21)
    bilateral = cv2.bilateralFilter(denoised, 7, 24, 24)

    median = cv2.medianBlur(arr, 5)
    diff = cv2.cvtColor(cv2.absdiff(arr, median), cv2.COLOR_BGR2GRAY)
    impulse_mask = (diff > 36).astype(np.uint8) * 255
    impulse_mask = cv2.morphologyEx(impulse_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    impulse_mask = cv2.dilate(impulse_mask, np.ones((3, 3), np.uint8), iterations=1)
    cleaned = arr.copy()
    cleaned[impulse_mask > 0] = bilateral[impulse_mask > 0]

    # A light final pass removes UV island pepper noise while keeping broad stripes.
    cleaned = cv2.addWeighted(cleaned, 0.62, bilateral, 0.38, 0)
    cleaned = suppress_dark_speckles(cleaned)
    return Image.fromarray(cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB))


def suppress_dark_speckles(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    local = cv2.medianBlur(gray, 13)
    dark_outliers = ((local.astype(np.int16) - gray.astype(np.int16)) > 16) & (gray < 125)
    dark_outliers = dark_outliers.astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark_outliers, 8)
    keep = np.zeros_like(dark_outliers)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area <= 4000:
            keep[labels == label] = 255
    if not np.any(keep):
        return bgr
    keep = cv2.dilate(keep, np.ones((2, 2), np.uint8), iterations=1)
    median_rgb = cv2.medianBlur(bgr, 11)
    inpainted = cv2.inpaint(bgr, keep, 3, cv2.INPAINT_TELEA)
    repaired = bgr.copy()
    repaired[keep > 0] = cv2.addWeighted(inpainted, 0.65, median_rgb, 0.35, 0)[keep > 0]
    return repaired


def enhance_texture_fur(image: Image.Image, strength: float) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    light = lab[..., 0]
    chroma = np.linalg.norm(lab[..., 1:3] - 128.0, axis=-1)
    fur_mask = smoothstep(48, 135, light) * (1.0 - smoothstep(38, 80, chroma))

    rng = np.random.default_rng(17)
    noise = rng.normal(0.0, 1.0, size=light.shape).astype(np.float32)
    fiber_a = cv2.GaussianBlur(noise, (1, 19), 0)
    fiber_b = cv2.GaussianBlur(noise, (17, 1), 0)
    fiber = fiber_a * 0.65 + fiber_b * 0.25
    fiber = cv2.GaussianBlur(fiber, (0, 0), 0.7)
    fiber = fiber / max(float(np.std(fiber)), 1e-6)

    broad = cv2.GaussianBlur(noise, (0, 0), 18)
    broad = broad / max(float(np.std(broad)), 1e-6)

    lab[..., 0] = np.clip(light + (fiber * 2.2 + broad * 1.3) * strength * fur_mask, 0, 255)

    # Very soft warm/cool color variation gives fur direction without pepper speckles.
    lab[..., 1] = np.clip(lab[..., 1] + broad * 0.9 * strength * fur_mask, 0, 255)
    lab[..., 2] = np.clip(lab[..., 2] + fiber * 0.35 * strength * fur_mask, 0, 255)
    enhanced = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    enhanced = cv2.bilateralFilter(enhanced, 5, 12, 12)
    return Image.fromarray(enhanced)


def sample_texture(uv: np.ndarray, image: Image.Image) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32) / 255.0
    h, w = arr.shape[:2]
    u = np.clip(uv[:, 0], 0.0, 1.0) * (w - 1)
    v = (1.0 - np.clip(uv[:, 1], 0.0, 1.0)) * (h - 1)
    x0 = np.floor(u).astype(np.int32)
    y0 = np.floor(v).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    wx = (u - x0)[:, None]
    wy = (v - y0)[:, None]
    top = arr[y0, x0] * (1.0 - wx) + arr[y0, x1] * wx
    bottom = arr[y1, x0] * (1.0 - wx) + arr[y1, x1] * wx
    return top * (1.0 - wy) + bottom * wy


def smoothstep(edge0: float, edge1: float, value: np.ndarray) -> np.ndarray:
    t = np.clip((value - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def normalize_vertices(vertices: np.ndarray) -> np.ndarray:
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    return (vertices - mins) / span * 2.0 - 1.0


def soft_fur_enhance(vertices: np.ndarray, base_rgb: np.ndarray, strength: float) -> np.ndarray:
    centered = normalize_vertices(vertices)
    x, y, z = centered[:, 0], centered[:, 1], centered[:, 2]
    brightness = base_rgb.mean(axis=1)
    saturation = base_rgb.max(axis=1) - base_rgb.min(axis=1)
    fur_mask = smoothstep(0.18, 0.55, brightness) * (1.0 - smoothstep(0.55, 0.95, saturation))

    vertical_fiber = np.sin((y * 42.0 + x * 4.5 + z * 2.0) * math.pi)
    secondary_fiber = np.sin((y * 76.0 - x * 5.0 + z * 5.5) * math.pi)
    fiber = 1.0 + (vertical_fiber * 0.018 + secondary_fiber * 0.010) * strength * fur_mask
    colored = np.clip(base_rgb * fiber[:, None], 0.0, 1.0)

    body_zone = smoothstep(-0.55, 0.2, y) * smoothstep(0.08, 0.75, np.abs(x) + np.maximum(z, 0.0) * 0.25)
    head_zone = smoothstep(0.0, 0.62, y) * (1.0 - smoothstep(0.52, 0.95, np.abs(x)))
    stripe_wave = np.sin((x * 5.6 + z * 3.4 + y * 0.85) * math.pi)
    stripes = smoothstep(0.58, 0.97, stripe_wave) * (0.45 * body_zone + 0.58 * head_zone) * fur_mask
    stripe_strength = np.clip(stripes * 0.17 * strength, 0.0, 0.16)
    tabby_tint = np.array([0.38, 0.31, 0.24], dtype=np.float32)
    colored = colored * (1.0 - stripe_strength[:, None]) + tabby_tint * stripe_strength[:, None]

    warm_variation = np.sin((x * 2.0 - y * 1.5 + z * 2.8) * math.pi) * 0.5 + 0.5
    colored += np.array([0.018, 0.013, 0.006], dtype=np.float32) * (warm_variation * fur_mask * strength)[:, None]
    return np.clip(colored, 0.0, 1.0)


def add_vertex_fur(mesh: trimesh.Trimesh, texture: Image.Image, strength: float) -> trimesh.Trimesh:
    uv = getattr(mesh.visual, "uv", None)
    if uv is None:
        base = np.asarray(mesh.visual.to_color().vertex_colors[:, :3], dtype=np.float32) / 255.0
    else:
        base = sample_texture(np.asarray(uv), texture)

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    enhanced = soft_fur_enhance(vertices, np.clip(base, 0.0, 1.0), strength)

    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    if normals.shape == vertices.shape:
        centered = vertices - vertices.mean(axis=0, keepdims=True)
        wave = np.sin(centered @ np.array([31.7, 79.3, 47.1], dtype=np.float32))
        displacement = wave * float(np.ptp(vertices, axis=0).max()) * 0.0012 * strength
        vertices = vertices + normals * displacement[:, None]

    colors = np.zeros((len(vertices), 4), dtype=np.uint8)
    colors[:, :3] = np.clip(enhanced * 255.0, 0, 255).astype(np.uint8)
    colors[:, 3] = 255
    refined = trimesh.Trimesh(vertices=vertices, faces=np.asarray(mesh.faces), vertex_colors=colors, process=False)
    refined.fix_normals()
    return refined


def save_textured_mesh(mesh: trimesh.Trimesh, texture: Image.Image, output_path: Path) -> None:
    uv = getattr(mesh.visual, "uv", None)
    if uv is None:
        mesh.export(output_path)
        return
    material = trimesh.visual.texture.SimpleMaterial(image=texture, diffuse=(255, 255, 255))
    textured = mesh.copy()
    textured.visual = trimesh.visual.TextureVisuals(uv=uv, image=texture, material=material)
    textured.export(output_path)


def textured_preview_mesh(mesh: trimesh.Trimesh, texture: Image.Image) -> trimesh.Trimesh:
    uv = getattr(mesh.visual, "uv", None)
    if uv is None:
        return mesh.copy()
    material = trimesh.visual.texture.SimpleMaterial(image=texture, diffuse=(255, 255, 255))
    textured = mesh.copy()
    textured.visual = trimesh.visual.TextureVisuals(uv=uv, image=texture, material=material)
    preview = textured.copy()
    preview.visual = preview.visual.to_color()
    return preview


def save_texture_contact(original: Image.Image, refined: Image.Image, output_path: Path) -> None:
    size = 512
    a = original.copy()
    b = refined.copy()
    a.thumbnail((size, size), Image.Resampling.LANCZOS)
    b.thumbnail((size, size), Image.Resampling.LANCZOS)
    sheet = Image.new("RGB", (size * 2, size + 42), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((16, 14), "original atlas", fill=(30, 30, 30))
    draw.text((size + 16, 14), "denoised atlas", fill=(30, 30, 30))
    sheet.paste(a, (0, 42))
    sheet.paste(b, (size, 42))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=94)


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    name = args.name
    mesh_path = args.input_dir / f"{name}_hunyuan2mini_turbo_hunyuan_texgen.glb"
    texture_path = args.input_dir / f"{name}_hunyuan2mini_turbo_hunyuan_texgen_texture.png"
    mesh = load_mesh(mesh_path)
    texture = texture_image(mesh, texture_path)

    denoised = denoise_texture(texture)
    enhanced_texture = enhance_texture_fur(denoised, args.texture_fur_strength)
    denoised_path = args.output_dir / f"{name}_hunyuan_texgen_denoised_texture.png"
    enhanced_texture_path = args.output_dir / f"{name}_hunyuan_texgen_fur_texture.png"
    denoised.save(denoised_path)
    enhanced_texture.save(enhanced_texture_path)
    save_texture_contact(texture, enhanced_texture, args.output_dir / f"{name}_texture_compare.jpg")

    cleaned = clean_tiny_face_components(mesh, args.max_faces, args.max_diag)

    glb_path = args.output_dir / f"{name}_hunyuan_texgen_uv_fur_clean.glb"
    obj_path = args.output_dir / f"{name}_hunyuan_texgen_uv_fur_clean.obj"
    preview_path = args.output_dir / f"{name}_hunyuan_texgen_uv_fur_clean_preview.jpg"
    save_textured_mesh(cleaned, enhanced_texture, glb_path)
    save_textured_mesh(cleaned, enhanced_texture, obj_path)
    save_preview(textured_preview_mesh(cleaned, enhanced_texture), preview_path)

    vertex_refined = add_vertex_fur(cleaned, enhanced_texture, args.fur_strength)
    vertex_glb_path = args.output_dir / f"{name}_hunyuan_texgen_vertex_fur_clean.glb"
    vertex_refined.export(vertex_glb_path)
    print(f"saved_glb={glb_path}")
    print(f"saved_obj={obj_path}")
    print(f"saved_texture={enhanced_texture_path}")
    print(f"saved_preview={preview_path}")
    print(f"saved_vertex_glb={vertex_glb_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="pose_02_transparent")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-faces", type=int, default=4)
    parser.add_argument("--max-diag", type=float, default=0.05)
    parser.add_argument("--fur-strength", type=float, default=0.55)
    parser.add_argument("--texture-fur-strength", type=float, default=0.85)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
