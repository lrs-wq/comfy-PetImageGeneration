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
HUNYUAN_MODEL_DIR = (
    COMFY3D_ROOT
    / "Checkpoints"
    / "Diffusers"
    / "tencent"
    / "Hunyuan3D-2mini"
    / "hunyuan3d-dit-v2-mini-turbo"
)
DEFAULT_INPUT = ROOT / "outputs" / "cat_poses_comfy" / "transparent" / "pose_02_transparent.png"
DEFAULT_OUTPUT = ROOT / "outputs" / "cat_3d_hunyuan2mini_turbo"


def prepare_rgba_image(image_path: Path, size: int = 1024) -> Image.Image:
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

    ckpt_path = HUNYUAN_MODEL_DIR / "model.fp16.safetensors"
    config_path = HUNYUAN_MODEL_DIR / "config.yaml"
    if not ckpt_path.exists() or not config_path.exists():
        raise FileNotFoundError(
            "Hunyuan3D-2mini-Turbo weights are missing. Expected model.fp16.safetensors and config.yaml."
        )

    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
        ckpt_path=str(ckpt_path),
        config_path=str(config_path),
        device=device,
        dtype=torch.float16,
        use_safetensors=True,
        from_pretrained_kwargs={
            "model_path": "tencent/Hunyuan3D-2mini",
            "subfolder": "hunyuan3d-dit-v2-mini-turbo",
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


def run_one(args: argparse.Namespace, pipe, image_path: Path) -> None:
    name = image_path.stem
    output_dir = args.output_dir / name
    output_dir.mkdir(parents=True, exist_ok=True)

    prepared = prepare_rgba_image(image_path)
    prepared_path = output_dir / "input_prepared_rgba_1024.png"
    prepared.save(prepared_path)

    print(f"input={image_path}")
    print(f"prepared={prepared_path}")
    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    start = time.time()
    with torch.inference_mode():
        mesh = pipe(
            image=prepared,
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

    obj_path = output_dir / f"{name}_hunyuan2mini_turbo.obj"
    glb_path = output_dir / f"{name}_hunyuan2mini_turbo.glb"
    preview_path = output_dir / f"{name}_hunyuan2mini_turbo_preview.jpg"
    mesh.export(obj_path)
    mesh.export(glb_path)
    save_preview(mesh, preview_path)

    print(f"obj={obj_path}")
    print(f"glb={glb_path}")
    print(f"preview={preview_path}")
    print(f"vertices={len(mesh.vertices)} faces={len(mesh.faces)} elapsed_sec={elapsed:.1f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--octree-resolution", type=int, default=192)
    parser.add_argument("--num-chunks", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--postprocess", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device != "cuda":
        raise RuntimeError("Hunyuan3D should be run on CUDA for this workflow.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this Python environment.")

    if args.input_dir:
        images = sorted(args.input_dir.glob("*.png"))
    else:
        images = [args.input]
    if not images:
        raise FileNotFoundError("No PNG input images found.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={args.device}")
    print(f"torch={torch.__version__}")
    print(f"gpu={torch.cuda.get_device_name(0)}")
    print(f"model_dir={HUNYUAN_MODEL_DIR}")

    pipe = load_pipeline(args.device)
    try:
        for image_path in images:
            run_one(args, pipe, image_path)
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        del pipe
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
