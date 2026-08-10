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
import yaml
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
COMFY3D_ROOT = ROOT / "ComfyUI" / "custom_nodes" / "ComfyUI-3D-Pack"
DEFAULT_CONFIG = COMFY3D_ROOT / "Configs" / "StableFast3D_config.yaml"
DEFAULT_WEIGHT = COMFY3D_ROOT / "Checkpoints" / "StableFast3D" / "model.safetensors"
LOCAL_DINO = COMFY3D_ROOT / "Checkpoints" / "DINO" / "facebook" / "dinov2-large"
LOCAL_CLIP = (
    COMFY3D_ROOT
    / "Checkpoints"
    / "CLIP"
    / "laion"
    / "CLIP-ViT-B-32-laion2B-s34B-b79K"
    / "open_clip_pytorch_model.bin"
)
DEFAULT_INPUT = ROOT / "outputs" / "cat_poses_comfy" / "transparent" / "pose_02_transparent.png"
DEFAULT_OUTPUT = ROOT / "outputs" / "cat_3d_stablefast3d"


def prepare_rgba_image(image_path: Path, size: int = 512) -> Image.Image:
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


def write_runtime_config(output_dir: Path) -> Path:
    with DEFAULT_CONFIG.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["image_tokenizer"]["pretrained_model_name_or_path"] = str(LOCAL_DINO)
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_config = output_dir / "StableFast3D_config.local.yaml"
    with runtime_config.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return runtime_config


def load_model(config_path: Path, weight_path: Path, device: str, geometry_only: bool):
    sys.path.insert(0, str(COMFY3D_ROOT))
    sys.path.insert(0, str(COMFY3D_ROOT / "Gen_3D_Modules"))
    patch_open_clip_downloads()
    import StableFast3D.sf3d.system as sf3d_system

    if geometry_only:
        class DummyTextureBaker(torch.nn.Module):
            pass

        sf3d_system.TextureBaker = DummyTextureBaker

    SF3D = sf3d_system.SF3D

    if not weight_path.exists():
        raise FileNotFoundError(f"StableFast3D weight not found: {weight_path}")
    if not (LOCAL_DINO / "config.json").exists() or not (LOCAL_DINO / "model.safetensors").exists():
        raise FileNotFoundError(f"Local DINOv2-large files not found under: {LOCAL_DINO}")
    if not LOCAL_CLIP.exists():
        raise FileNotFoundError(f"Local OpenCLIP file not found: {LOCAL_CLIP}")

    model = SF3D.from_pretrained(config_path=str(config_path), weight_path=str(weight_path))
    model.eval().to(device)
    return model


def patch_open_clip_downloads() -> None:
    import open_clip.factory as open_clip_factory
    import open_clip.pretrained as open_clip_pretrained

    original = open_clip_pretrained.download_pretrained_from_hf

    def local_download_pretrained_from_hf(
        model_id: str,
        filename: str | None = None,
        revision: str | None = None,
        cache_dir: str | None = None,
    ):
        if model_id == "laion/CLIP-ViT-B-32-laion2B-s34B-b79K":
            return str(LOCAL_CLIP)
        return original(model_id, filename=filename, revision=revision, cache_dir=cache_dir)

    open_clip_pretrained.download_pretrained_from_hf = local_download_pretrained_from_hf
    open_clip_factory.download_pretrained_from_hf = local_download_pretrained_from_hf


def as_trimesh(mesh: trimesh.Trimesh | trimesh.Scene) -> trimesh.Trimesh:
    if isinstance(mesh, trimesh.Scene):
        parts = [geom for geom in mesh.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        if not parts:
            return trimesh.Trimesh()
        return trimesh.util.concatenate(parts)
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
    sheet.save(output_path, quality=92)


def run_geometry_only(model, image: Image.Image, device: str, remesh: str) -> trimesh.Trimesh:
    sys.path.insert(0, str(COMFY3D_ROOT / "Gen_3D_Modules"))
    from StableFast3D.sf3d.utils import create_intrinsic_from_fov_deg, default_cond_c2w

    if image.mode != "RGBA":
        raise ValueError("Image must be in RGBA mode")

    img_cond = (
        torch.from_numpy(
            np.asarray(image.resize((model.cfg.cond_image_size, model.cfg.cond_image_size))).astype(np.float32)
            / 255.0
        )
        .float()
        .clip(0, 1)
        .to(device)
    )
    mask_cond = img_cond[:, :, -1:]
    rgb_cond = torch.lerp(
        torch.tensor(model.cfg.background_color, device=device)[None, None, :],
        img_cond[:, :, :3],
        mask_cond,
    )
    c2w_cond = default_cond_c2w(model.cfg.default_distance).to(device)
    intrinsic, intrinsic_normed_cond = create_intrinsic_from_fov_deg(
        model.cfg.default_fovy_deg,
        model.cfg.cond_image_size,
        model.cfg.cond_image_size,
    )
    batch = {
        "rgb_cond": rgb_cond,
        "mask_cond": mask_cond,
        "c2w_cond": c2w_cond.unsqueeze(0),
        "intrinsic_cond": intrinsic.to(device).unsqueeze(0),
        "intrinsic_normed_cond": intrinsic_normed_cond.to(device).unsqueeze(0),
    }
    batch["rgb_cond"] = model.image_processor(batch["rgb_cond"], model.cfg.cond_image_size)
    batch["mask_cond"] = model.image_processor(batch["mask_cond"], model.cfg.cond_image_size)

    scene_codes, _ = model.get_scene_codes(batch)
    mesh = model.triplane_to_meshes(scene_codes)[0]
    if remesh == "triangle":
        mesh = mesh.triangle_remesh()
    elif remesh == "quad":
        mesh = mesh.quad_remesh()

    vertices = mesh.v_pos.detach().float().cpu().numpy()
    vertices = vertices[:, [1, 2, 0]]
    faces = mesh.t_pos_idx.detach().long().cpu().numpy()
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def run_one(args: argparse.Namespace, model, image_path: Path, runtime_config: Path) -> None:
    name = image_path.stem
    output_dir = args.output_dir / name
    output_dir.mkdir(parents=True, exist_ok=True)

    prepared = prepare_rgba_image(image_path)
    prepared_path = output_dir / "input_prepared_rgba_512.png"
    prepared.save(prepared_path)

    print(f"input={image_path}")
    print(f"prepared={prepared_path}")
    print(f"runtime_config={runtime_config}")

    remesh = args.remesh.lower()
    start = time.time()
    with torch.inference_mode():
        with torch.autocast(device_type=args.device, dtype=torch.float16):
            if args.with_texture:
                mesh, _ = model.run_image(
                    prepared,
                    bake_resolution=args.texture_resolution,
                    remesh=remesh,
                    estimate_illumination=False,
                )
            else:
                mesh = run_geometry_only(model, prepared, args.device, remesh)
    elapsed = time.time() - start

    mesh = as_trimesh(mesh)
    mesh.remove_unreferenced_vertices()
    if len(mesh.vertices) and len(mesh.faces):
        mesh.fix_normals()

    obj_path = output_dir / f"{name}_stablefast3d.obj"
    glb_path = output_dir / f"{name}_stablefast3d.glb"
    preview_path = output_dir / f"{name}_stablefast3d_preview.jpg"
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
    parser.add_argument("--texture-resolution", type=int, default=512)
    parser.add_argument("--remesh", choices=["none", "triangle", "quad"], default="none")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--weight", type=Path, default=DEFAULT_WEIGHT)
    parser.add_argument("--with-texture", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device != "cuda":
        raise RuntimeError("StableFast3D should be run on CUDA for this workflow.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this Python environment.")

    if args.input_dir:
        images = sorted(args.input_dir.glob("*.png"))
    else:
        images = [args.input]
    if not images:
        raise FileNotFoundError("No PNG input images found.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runtime_config = write_runtime_config(args.output_dir)

    print(f"device={args.device}")
    print(f"torch={torch.__version__}")
    print(f"gpu={torch.cuda.get_device_name(0)}")
    print(f"weight={args.weight}")
    print(f"local_dino={LOCAL_DINO}")

    model = load_model(runtime_config, args.weight, args.device, geometry_only=not args.with_texture)
    try:
        for image_path in images:
            run_one(args, model, image_path, runtime_config)
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
