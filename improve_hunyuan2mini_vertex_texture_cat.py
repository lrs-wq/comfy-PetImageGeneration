from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = ROOT / "outputs" / "cat_poses_comfy" / "transparent"
DEFAULT_MESH_DIR = ROOT / "outputs" / "cat_3d_hunyuan2mini_turbo"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "cat_3d_hunyuan2mini_turbo_textured"


def prepare_rgba_image(image_path: Path, size: int = 1024) -> Image.Image:
    image = Image.open(image_path).convert("RGBA")
    bbox = image.getbbox()
    if bbox:
        image = image.crop(bbox)
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    canvas.alpha_composite(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def fill_transparent_pixels(rgba: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(rgba, dtype=np.uint8)
    rgb = arr[..., :3]
    alpha = arr[..., 3]
    transparent = alpha < 16
    if not transparent.any():
        return rgb.astype(np.float32) / 255.0, alpha.astype(np.float32) / 255.0

    try:
        from scipy import ndimage

        _, indices = ndimage.distance_transform_edt(transparent, return_indices=True)
        filled = rgb[indices[0], indices[1]]
    except Exception:
        filled = rgb.copy()
        fallback = rgb[~transparent].mean(axis=0) if (~transparent).any() else np.array([226, 216, 198])
        filled[transparent] = fallback.astype(np.uint8)
    return filled.astype(np.float32) / 255.0, alpha.astype(np.float32) / 255.0


def bilinear_sample(image: np.ndarray, uv: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    x = np.clip(uv[:, 0], 0.0, 1.0) * (w - 1)
    y = np.clip(uv[:, 1], 0.0, 1.0) * (h - 1)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    wx = (x - x0)[:, None]
    wy = (y - y0)[:, None]
    top = image[y0, x0] * (1.0 - wx) + image[y0, x1] * wx
    bottom = image[y1, x0] * (1.0 - wx) + image[y1, x1] * wx
    return top * (1.0 - wy) + bottom * wy


def normalize_vertices(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    normalized = (vertices - mins) / span
    centered = normalized * 2.0 - 1.0
    return normalized, centered, span


def hash_noise(points: np.ndarray, scale: float, seed: float) -> np.ndarray:
    p = points * scale
    value = np.sin(p @ np.array([12.9898, 78.233, 37.719], dtype=np.float32) + seed) * 43758.5453
    return value - np.floor(value)


def smoothstep(edge0: float, edge1: float, value: np.ndarray) -> np.ndarray:
    t = np.clip((value - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def enhance_fur_and_tabby(vertices: np.ndarray, base_rgb: np.ndarray, strength: float) -> np.ndarray:
    _, centered, _ = normalize_vertices(vertices)
    x, y, z = centered[:, 0], centered[:, 1], centered[:, 2]
    brightness = base_rgb.mean(axis=1)
    saturation_proxy = base_rgb.max(axis=1) - base_rgb.min(axis=1)
    fur_mask = smoothstep(0.22, 0.55, brightness) * (1.0 - smoothstep(0.45, 0.9, saturation_proxy))

    fine = (
        0.55 * hash_noise(centered, 42.0, 1.7)
        + 0.30 * hash_noise(centered, 95.0, 8.9)
        + 0.15 * hash_noise(centered, 180.0, 4.3)
    )
    fiber = 0.90 + (fine - 0.5) * 0.28 * strength * fur_mask
    colored = np.clip(base_rgb * fiber[:, None], 0.0, 1.0)

    body_zone = smoothstep(-0.35, 0.15, y) * smoothstep(-0.15, 0.65, np.abs(x) + np.maximum(z, 0.0) * 0.35)
    head_zone = smoothstep(0.05, 0.48, y)
    stripe_wave = np.sin((x * 9.0 + z * 5.5 + y * 2.8) * math.pi + hash_noise(centered, 12.0, 2.2) * 1.8)
    stripes = smoothstep(0.52, 0.96, stripe_wave) * (0.45 * body_zone + 0.65 * head_zone) * fur_mask

    forehead_wave = np.sin((np.abs(x) * 23.0 + y * 13.0) * math.pi)
    forehead = smoothstep(0.20, 0.72, y) * (1.0 - smoothstep(0.18, 0.55, np.abs(x))) * fur_mask
    stripes += smoothstep(0.35, 0.98, forehead_wave) * forehead * 0.55
    stripes = np.clip(stripes * strength, 0.0, 1.0)

    tabby_tint = np.array([0.26, 0.22, 0.18], dtype=np.float32)
    colored = colored * (1.0 - stripes[:, None] * 0.42) + tabby_tint * stripes[:, None] * 0.42

    warm_guard_hair = hash_noise(centered, 72.0, 12.5)
    guard = smoothstep(0.62, 1.0, warm_guard_hair) * fur_mask * strength * 0.12
    colored = np.clip(colored + np.array([0.08, 0.06, 0.03], dtype=np.float32) * guard[:, None], 0.0, 1.0)
    return colored


def project_vertex_colors(mesh: trimesh.Trimesh, rgba: Image.Image, strength: float) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    normalized, _, _ = normalize_vertices(vertices)
    uv = np.column_stack([normalized[:, 0], 1.0 - normalized[:, 1]])
    filled_rgb, _ = fill_transparent_pixels(rgba)
    base = bilinear_sample(filled_rgb, uv)
    base = np.clip(base * 1.04, 0.0, 1.0)
    enhanced = enhance_fur_and_tabby(vertices, base, strength)
    rgba_colors = np.zeros((len(vertices), 4), dtype=np.uint8)
    rgba_colors[:, :3] = np.clip(enhanced * 255.0, 0, 255).astype(np.uint8)
    rgba_colors[:, 3] = 255
    return rgba_colors


def add_micro_fur_displacement(mesh: trimesh.Trimesh, strength: float) -> trimesh.Trimesh:
    vertices = np.asarray(mesh.vertices, dtype=np.float32).copy()
    _, centered, span = normalize_vertices(vertices)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    if normals.shape != vertices.shape:
        return mesh
    fine = hash_noise(centered, 120.0, 9.4) - 0.5
    medium = hash_noise(centered, 36.0, 14.0) - 0.5
    displacement = (fine * 0.45 + medium * 0.55) * float(span.max()) * 0.010 * strength
    vertices = vertices + normals * displacement[:, None]
    return trimesh.Trimesh(vertices=vertices, faces=np.asarray(mesh.faces), process=False)


def rotate_y(vertices: np.ndarray, degrees: float) -> np.ndarray:
    radians = math.radians(degrees)
    c = math.cos(radians)
    s = math.sin(radians)
    rot = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)
    return vertices @ rot.T


def render_mesh_view(mesh: trimesh.Trimesh, degrees: float, size: int = 512) -> Image.Image:
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    center = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
    vertices = vertices - center
    scale = np.linalg.norm(vertices, axis=1).max()
    if scale > 0:
        vertices = vertices / scale
    vertices = rotate_y(vertices, degrees)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertex_colors = np.asarray(mesh.visual.vertex_colors[:, :3], dtype=np.float32) / 255.0

    face_vertices = vertices[faces]
    normals = np.cross(face_vertices[:, 1] - face_vertices[:, 0], face_vertices[:, 2] - face_vertices[:, 0])
    normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-6)
    light = np.array([0.25, -0.45, 0.86], dtype=np.float32)
    light = light / np.linalg.norm(light)
    brightness = np.clip((normals @ light) * 0.55 + 0.58, 0.22, 1.08)
    face_colors = vertex_colors[faces].mean(axis=1)
    face_colors = np.clip(face_colors * brightness[:, None], 0.0, 1.0)

    z_order = np.argsort(face_vertices[:, :, 2].mean(axis=1))
    xy = face_vertices[:, :, :2]
    xy[:, :, 1] *= -1.0
    xy = xy * (size * 0.38) + (size * 0.5)

    image = Image.new("RGB", (size, size), (250, 250, 250))
    draw = ImageDraw.Draw(image)
    for idx in z_order:
        pts = [tuple(p) for p in xy[idx]]
        color = tuple(np.clip(face_colors[idx] * 255.0, 0, 255).astype(np.uint8))
        draw.polygon(pts, fill=color)
    return image


def save_preview(mesh: trimesh.Trimesh, output_path: Path) -> None:
    views = [0, 45, 90, 135, 180, 225, 270, 315]
    tile = 512
    sheet = Image.new("RGB", (tile * 4, tile * 2), "white")
    for i, angle in enumerate(views):
        sheet.paste(render_mesh_view(mesh, angle, tile), ((i % 4) * tile, (i // 4) * tile))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Unable to load mesh from {path}")
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    return mesh


def process_one(name: str, args: argparse.Namespace) -> Path:
    image_path = args.input_dir / f"{name}.png"
    mesh_path = args.mesh_dir / name / f"{name}_hunyuan2mini_turbo.obj"
    if not image_path.exists():
        image_path = args.mesh_dir / name / "input_prepared_rgba_1024.png"
    if not mesh_path.exists():
        raise FileNotFoundError(mesh_path)

    output_dir = args.output_dir / name
    output_dir.mkdir(parents=True, exist_ok=True)

    image = prepare_rgba_image(image_path, args.image_size)
    mesh = load_mesh(mesh_path)
    mesh = add_micro_fur_displacement(mesh, args.texture_strength)
    vertex_colors = project_vertex_colors(mesh, image, args.texture_strength)
    textured = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.faces),
        vertex_colors=vertex_colors,
        process=False,
    )
    textured.fix_normals()

    glb_path = output_dir / f"{name}_hunyuan2mini_turbo_textured.glb"
    ply_path = output_dir / f"{name}_hunyuan2mini_turbo_textured.ply"
    obj_path = output_dir / f"{name}_hunyuan2mini_turbo_textured.obj"
    preview_path = output_dir / f"{name}_hunyuan2mini_turbo_textured_preview.jpg"

    textured.export(glb_path)
    textured.export(ply_path)
    textured.export(obj_path)
    save_preview(textured, preview_path)
    print(f"{name}: vertices={len(textured.vertices)} faces={len(textured.faces)} preview={preview_path}")
    return preview_path


def save_contact_sheet(previews: list[Path], output_path: Path) -> None:
    thumbs = []
    for preview in previews:
        image = Image.open(preview).convert("RGB")
        image.thumbnail((420, 210), Image.Resampling.LANCZOS)
        thumbs.append((preview.parent.name, image.copy()))
    cols = 2
    rows = math.ceil(len(thumbs) / cols)
    cell_w, cell_h = 460, 270
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (label, image) in enumerate(thumbs):
        x = (i % cols) * cell_w
        y = (i // cols) * cell_h
        draw.text((x + 20, y + 16), label, fill=(30, 30, 30))
        sheet.paste(image, (x + 20, y + 48))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=94)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--mesh-dir", type=Path, default=DEFAULT_MESH_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--name", help="Process a single pose folder name, for example pose_02_transparent.")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--texture-strength", type=float, default=0.85)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.name:
        names = [args.name]
    else:
        names = sorted(p.name for p in args.mesh_dir.glob("pose_*_transparent") if p.is_dir())
    previews = [process_one(name, args) for name in names]
    if len(previews) > 1:
        save_contact_sheet(previews, args.output_dir / "cat_3d_hunyuan2mini_turbo_textured_contact_sheet.jpg")


if __name__ == "__main__":
    main()
