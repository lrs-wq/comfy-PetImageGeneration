from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import torch
import trimesh
from PIL import Image

from improve_hunyuan2mini_vertex_texture_cat import save_preview


ROOT = Path(__file__).resolve().parent
COMFY3D_ROOT = ROOT / "ComfyUI" / "custom_nodes" / "ComfyUI-3D-Pack"
HUNYUAN_MODULES = COMFY3D_ROOT / "Gen_3D_Modules"
DEFAULT_MODEL_DIR = COMFY3D_ROOT / "Checkpoints" / "Diffusers" / "tencent" / "Hunyuan3D-2"
DEFAULT_IMAGE_DIR = ROOT / "outputs" / "cat_poses_comfy" / "transparent"
DEFAULT_MESH_DIR = ROOT / "outputs" / "cat_3d_hunyuan2mini_turbo"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "cat_3d_hunyuan2mini_turbo_hunyuan_texgen"


def add_runtime_paths() -> None:
    sys.path.insert(0, str(HUNYUAN_MODULES))
    sys.path.insert(
        0,
        str(
            COMFY3D_ROOT
            / "Gen_3D_Modules"
            / "Hunyuan3D_V2"
            / "hy3dgen"
            / "texgen"
            / "custom_rasterizer"
        ),
    )
    for dll_dir in [
        Path(sys.executable).resolve().parent / "Library" / "bin",
        Path(sys.executable).resolve().parent / "Lib" / "site-packages" / "torch" / "lib",
    ]:
        if dll_dir.exists() and hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(dll_dir))


def prepare_reference_image(image_path: Path, size: int) -> Image.Image:
    image = Image.open(image_path).convert("RGBA")
    bbox = image.getbbox()
    if bbox:
        image = image.crop(bbox)
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    canvas.alpha_composite(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def load_pose_mesh(mesh_path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Unable to load mesh: {mesh_path}")
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    return mesh


def texture_visual_to_vertex_colors(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    preview_mesh = mesh.copy()
    try:
        preview_mesh.visual = mesh.visual.to_color()
    except Exception:
        return preview_mesh
    return preview_mesh


def save_texture_image(mesh: trimesh.Trimesh, output_path: Path) -> None:
    material = getattr(mesh.visual, "material", None)
    image = getattr(material, "image", None)
    if image is None:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def build_texgen_pipeline(args: argparse.Namespace):
    add_runtime_paths()
    from Hunyuan3D_V2.hy3dgen.texgen.pipelines import Hunyuan3DPaintPipeline, Hunyuan3DTexGenConfig

    delight_path = args.model_dir / "hunyuan3d-delight-v2-0"
    paint_path = args.model_dir / args.subfolder
    if not delight_path.exists() or not paint_path.exists():
        raise FileNotFoundError(f"Missing TexGen weights under {args.model_dir}")

    config = Hunyuan3DTexGenConfig(str(delight_path), str(paint_path), args.subfolder)
    config.render_size = args.render_size
    config.texture_size = args.texture_size
    if args.offload:
        config.device = "cpu"

    pipe = Hunyuan3DPaintPipeline(config)
    pipe.render.set_default_render_resolution(args.render_size)
    pipe.render.set_default_texture_resolution(args.texture_size)
    pipe.render.to("cuda")

    if args.offload:
        pipe.enable_model_cpu_offload(device="cuda")
    else:
        pipe.to("cuda", torch.float16)
    return pipe


def run_one(args: argparse.Namespace) -> Path:
    name = args.name
    image_path = args.image_dir / f"{name}.png"
    mesh_path = args.mesh_dir / name / f"{name}_hunyuan2mini_turbo.obj"
    if not image_path.exists():
        image_path = args.mesh_dir / name / "input_prepared_rgba_1024.png"
    if not mesh_path.exists():
        raise FileNotFoundError(mesh_path)

    output_dir = args.output_dir / name
    output_dir.mkdir(parents=True, exist_ok=True)
    ref_path = output_dir / f"{name}_reference_rgba.png"
    ref = prepare_reference_image(image_path, args.image_size)
    ref.save(ref_path)

    torch.cuda.empty_cache()
    gc.collect()

    pipe = build_texgen_pipeline(args)
    mesh = load_pose_mesh(mesh_path)
    textured = pipe(mesh, [ref])

    glb_path = output_dir / f"{name}_hunyuan2mini_turbo_hunyuan_texgen.glb"
    obj_path = output_dir / f"{name}_hunyuan2mini_turbo_hunyuan_texgen.obj"
    texture_path = output_dir / f"{name}_hunyuan2mini_turbo_hunyuan_texgen_texture.png"
    preview_path = output_dir / f"{name}_hunyuan2mini_turbo_hunyuan_texgen_preview.jpg"

    textured.export(glb_path)
    textured.export(obj_path)
    save_texture_image(textured, texture_path)
    save_preview(texture_visual_to_vertex_colors(textured), preview_path)

    print(f"saved_glb={glb_path}")
    print(f"saved_obj={obj_path}")
    print(f"saved_texture={texture_path if texture_path.exists() else 'none'}")
    print(f"saved_preview={preview_path}")
    return preview_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="pose_02_transparent")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--subfolder", default="hunyuan3d-paint-v2-0-turbo")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--mesh-dir", type=Path, default=DEFAULT_MESH_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--render-size", type=int, default=1024)
    parser.add_argument("--texture-size", type=int, default=1024)
    parser.add_argument("--offload", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    run_one(parse_args())
