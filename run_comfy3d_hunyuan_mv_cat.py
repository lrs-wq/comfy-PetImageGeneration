from __future__ import annotations

import argparse
import gc
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
COMFY3D_ROOT = ROOT / "ComfyUI" / "custom_nodes" / "ComfyUI-3D-Pack"
HUNYUAN_MV_MODEL_DIR = (
    COMFY3D_ROOT
    / "Checkpoints"
    / "Diffusers"
    / "tencent"
    / "Hunyuan3D-2mv"
    / "hunyuan3d-dit-v2-mv-turbo"
)
DEFAULT_SHEET = (
    ROOT
    / "outputs"
    / "cat_3d_hunyuan2mini_turbo"
    / "pose_02_transparent"
    / "pose_02_transparent_hunyuan2mini_turbo_preview.jpg"
)
DEFAULT_VIEWS_DIR = ROOT / "outputs" / "cat_3d_multiview_constraints" / "pose_02_hunyuan_views"
DEFAULT_OUTPUT = ROOT / "outputs" / "cat_3d_hunyuan2mv_turbo"


def crop_views_from_sheet(sheet_path: Path, output_dir: Path) -> dict[str, Path]:
    sheet = Image.open(sheet_path).convert("RGB")
    tile_w = sheet.width // 4
    tile_h = sheet.height // 2
    # The source sheet uses angles [0, 45, 90, 135, 180, 225, 270, 315].
    tile_map = {
        "front": 1,
        "left": 0,
        "back": 5,
        "right": 7,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, idx in tile_map.items():
        x = (idx % 4) * tile_w
        y = (idx // 4) * tile_h
        crop = sheet.crop((x, y, x + tile_w, y + tile_h))
        rgba = white_to_alpha(crop)
        path = output_dir / f"{name}.png"
        rgba.save(path)
        result[name] = path
    return result


def white_to_alpha(image: Image.Image) -> Image.Image:
    rgb = image.convert("RGB")
    arr = np.asarray(rgb).copy()
    alpha = np.where(np.all(arr > 244, axis=-1), 0, 255).astype(np.uint8)
    rgba = np.dstack([arr, alpha])
    return Image.fromarray(rgba, "RGBA")


def prepare_rgba_image(image_path: Path, size: int = 768) -> Image.Image:
    image = Image.open(image_path).convert("RGBA")
    bbox = image.getbbox()
    if bbox:
        image = image.crop(bbox)
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    left = (size - image.width) // 2
    top = (size - image.height) // 2
    canvas.alpha_composite(image, (left, top))
    return canvas


def load_pipeline(device: str):
    sys.path.insert(0, str(COMFY3D_ROOT))
    sys.path.insert(0, str(COMFY3D_ROOT / "Gen_3D_Modules"))
    from Hunyuan3D_V2.hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    ckpt_path = HUNYUAN_MV_MODEL_DIR / "model.fp16.safetensors"
    config_path = HUNYUAN_MV_MODEL_DIR / "config.yaml"
    if not ckpt_path.exists() or not config_path.exists():
        raise FileNotFoundError(
            "Hunyuan3D-2mv-Turbo weights are missing. Expected model.fp16.safetensors and config.yaml."
        )

    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
        ckpt_path=str(ckpt_path),
        config_path=str(config_path),
        device=device,
        dtype=torch.float16,
        use_safetensors=True,
        from_pretrained_kwargs={
            "model_path": "tencent/Hunyuan3D-2mv",
            "subfolder": "hunyuan3d-dit-v2-mv-turbo",
            "use_safetensors": True,
        },
    )
    return pipe.to(device, torch.float16)


def maybe_postprocess(mesh: trimesh.Trimesh, enabled: bool) -> trimesh.Trimesh:
    if not enabled:
        return mesh
    sys.path.insert(0, str(COMFY3D_ROOT / "Gen_3D_Modules"))
    from Hunyuan3D_V2.hy3dgen.shapegen import DegenerateFaceRemover, FaceReducer, FloaterRemover

    for fn in (FloaterRemover(), DegenerateFaceRemover(), FaceReducer()):
        mesh = fn(mesh)
    return mesh


def normalize_vertices(vertices: np.ndarray) -> np.ndarray:
    center = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
    vertices = vertices - center
    scale = np.linalg.norm(vertices, axis=1).max()
    if scale > 0:
        vertices = vertices / scale
    return vertices


def rotate_y(vertices: np.ndarray, degrees: float) -> np.ndarray:
    radians = math.radians(degrees)
    c = math.cos(radians)
    s = math.sin(radians)
    rot = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)
    return vertices @ rot.T


def render_mesh_view(mesh: trimesh.Trimesh, degrees: float, size: int = 512) -> Image.Image:
    vertices = normalize_vertices(np.asarray(mesh.vertices, dtype=np.float32))
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertices = rotate_y(vertices, degrees)

    light = np.array([0.25, -0.45, 0.86], dtype=np.float32)
    light = light / np.linalg.norm(light)
    face_vertices = vertices[faces]
    normals = np.cross(face_vertices[:, 1] - face_vertices[:, 0], face_vertices[:, 2] - face_vertices[:, 0])
    normal_norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(normal_norm, 1e-6)
    brightness = np.clip((normals @ light) * 0.55 + 0.45, 0.18, 1.0)

    z_order = np.argsort(face_vertices[:, :, 2].mean(axis=1))
    xy = face_vertices[:, :, :2]
    xy[:, :, 1] *= -1.0
    xy = xy * (size * 0.38) + (size * 0.5)

    image = Image.new("RGB", (size, size), (250, 250, 250))
    draw = ImageDraw.Draw(image)
    for idx in z_order:
        pts = [tuple(p) for p in xy[idx]]
        shade = int(210 * float(brightness[idx]) + 35)
        color = (shade, shade, min(255, shade + 8))
        draw.polygon(pts, fill=color)
    return image


def save_preview(mesh: trimesh.Trimesh, output_path: Path) -> None:
    views = [0, 45, 90, 135, 180, 225, 270, 315]
    tile = 512
    sheet = Image.new("RGB", (tile * 4, tile * 2), "white")
    for i, angle in enumerate(views):
        image = render_mesh_view(mesh, angle, tile)
        sheet.paste(image, ((i % 4) * tile, (i // 4) * tile))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def build_views(args: argparse.Namespace) -> dict[str, Image.Image]:
    if args.sheet:
        paths = crop_views_from_sheet(args.sheet, args.views_dir)
    else:
        paths = {name: args.views_dir / f"{name}.png" for name in ("front", "left", "back", "right")}

    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing multiview images: " + ", ".join(missing))

    return {name: prepare_rgba_image(path) for name, path in paths.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sheet", type=Path, default=DEFAULT_SHEET)
    parser.add_argument("--views-dir", type=Path, default=DEFAULT_VIEWS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--octree-resolution", type=int, default=160)
    parser.add_argument("--num-chunks", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--postprocess", action="store_true")
    args = parser.parse_args()

    if args.device != "cuda":
        raise RuntimeError("Hunyuan3D-2mv should be run on CUDA for this workflow.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this Python environment.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={args.device}")
    print(f"torch={torch.__version__}")
    print(f"gpu={torch.cuda.get_device_name(0)}")
    print(f"model_dir={HUNYUAN_MV_MODEL_DIR}")

    views = build_views(args)
    prepared_dir = args.output_dir / "prepared_views"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    for name, image in views.items():
        path = prepared_dir / f"{name}.png"
        image.save(path)
        print(f"{name}={path}")

    pipe = load_pipeline(args.device)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    start = time.time()
    try:
        with torch.inference_mode():
            mesh = pipe(
                image=views,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
                octree_resolution=args.octree_resolution,
                num_chunks=args.num_chunks,
                output_type="trimesh",
                enable_pbar=True,
            )[0]
        elapsed = time.time() - start
        mesh = maybe_postprocess(mesh, args.postprocess)
        mesh.remove_unreferenced_vertices()
        mesh.fix_normals()

        obj_path = args.output_dir / "cat_pose_02_hunyuan2mv_turbo.obj"
        glb_path = args.output_dir / "cat_pose_02_hunyuan2mv_turbo.glb"
        preview_path = args.output_dir / "cat_pose_02_hunyuan2mv_turbo_preview.jpg"
        mesh.export(obj_path)
        mesh.export(glb_path)
        save_preview(mesh, preview_path)

        print(f"obj={obj_path}")
        print(f"glb={glb_path}")
        print(f"preview={preview_path}")
        print(f"vertices={len(mesh.vertices)} faces={len(mesh.faces)} elapsed_sec={elapsed:.1f}")
    finally:
        del pipe
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
