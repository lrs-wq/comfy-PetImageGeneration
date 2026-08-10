from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import torch
import trimesh
from huggingface_hub import hf_hub_download
from PIL import Image


ROOT = Path(__file__).resolve().parent
COMFY3D_ROOT = ROOT / "ComfyUI" / "custom_nodes" / "ComfyUI-3D-Pack"
TRIPOSR_CONFIG = COMFY3D_ROOT / "Configs" / "TripoSR_config.yaml"
TRIPOSR_CKPT_DIR = COMFY3D_ROOT / "Checkpoints" / "TripoSR"
DINO_CONFIG = COMFY3D_ROOT / "Checkpoints" / "DINO" / "facebook" / "dino-vitb16" / "config.json"
DEFAULT_INPUT = ROOT / "outputs" / "cat_poses_comfy" / "transparent" / "pose_02_transparent.png"
DEFAULT_OUTPUT = ROOT / "outputs" / "cat_3d_triposr"


def install_light_mesh_utils_patch() -> None:
    """Provide only the axis helper TripoSR needs, avoiding heavy mesh imports."""
    mesh_pkg = types.ModuleType("mesh_processer")
    mesh_utils = types.ModuleType("mesh_processer.mesh_utils")

    def switch_vector_axis(vector3s, target_axis):
        return_numpy = not isinstance(vector3s, torch.Tensor)
        if not isinstance(vector3s, torch.Tensor):
            vector3s = torch.as_tensor(vector3s)
        axes = []
        for axis in target_axis:
            sign = -1 if isinstance(axis, str) and axis.startswith("-") else 1
            key = axis[-1] if isinstance(axis, str) else axis
            if key in ("x", "0", 0):
                idx = 0
            elif key in ("y", "1", 1):
                idx = 1
            elif key in ("z", "2", 2):
                idx = 2
            else:
                idx = int(key)
            axes.append(vector3s[..., idx] * sign)
        switched = torch.stack(axes, dim=-1)
        return switched.detach().cpu().numpy() if return_numpy else switched

    mesh_utils.switch_vector_axis = switch_vector_axis
    sys.modules.setdefault("mesh_processer", mesh_pkg)
    sys.modules["mesh_processer.mesh_utils"] = mesh_utils


def prepare_rgb_image(image_path: Path, size: int = 512) -> Image.Image:
    image = Image.open(image_path).convert("RGBA")
    bbox = image.getbbox()
    if bbox:
        image = image.crop(bbox)
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    left = (size - image.width) // 2
    top = (size - image.height) // 2
    canvas.alpha_composite(image, (left, top))
    white = Image.new("RGBA", canvas.size, (255, 255, 255, 255))
    white.alpha_composite(canvas)
    return white.convert("RGB")


def load_model(device: str):
    install_light_mesh_utils_patch()
    sys.path.insert(0, str(COMFY3D_ROOT))
    sys.path.insert(0, str(COMFY3D_ROOT / "Gen_3D_Modules"))

    from TripoSR.system import TSR
    import TripoSR.models.tokenizers.image as image_tokenizer

    original_hf_hub_download = image_tokenizer.hf_hub_download

    def local_hf_hub_download(repo_id: str, filename: str, *args, **kwargs):
        if repo_id == "facebook/dino-vitb16" and filename == "config.json":
            return str(DINO_CONFIG)
        return original_hf_hub_download(repo_id, filename, *args, **kwargs)

    image_tokenizer.hf_hub_download = local_hf_hub_download

    TRIPOSR_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = TRIPOSR_CKPT_DIR / "model.ckpt"
    if not ckpt_path.exists():
        downloaded = hf_hub_download(
            repo_id="stabilityai/TripoSR",
            filename="model.ckpt",
            local_dir=str(TRIPOSR_CKPT_DIR),
            local_dir_use_symlinks=False,
        )
        ckpt_path = Path(downloaded)

    model = TSR.from_pretrained(str(ckpt_path), str(TRIPOSR_CONFIG))
    model.eval().to(device)
    return model


def export_glb(mesh: trimesh.Trimesh, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=25.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepared = prepare_rgb_image(args.input)
    prepared_path = args.output_dir / "input_prepared_white_512.png"
    prepared.save(prepared_path)

    print(f"device={args.device}")
    print(f"input={args.input}")
    print(f"prepared={prepared_path}")

    model = load_model(args.device)
    with torch.no_grad():
        scene_codes = model([prepared], device=args.device)
        meshes = model.extract_mesh(
            scene_codes,
            resolution=args.resolution,
            threshold=args.threshold,
        )

    mesh = meshes[0]
    obj_path = args.output_dir / f"{args.input.stem}_triposr.obj"
    glb_path = args.output_dir / f"{args.input.stem}_triposr.glb"
    export_glb(mesh, obj_path)
    export_glb(mesh, glb_path)

    preview_images = model.render(scene_codes, n_views=8, return_type="pil")
    preview_w = prepared.width * 4
    preview_h = prepared.height * 2
    sheet = Image.new("RGB", (preview_w, preview_h), "white")
    for i, img in enumerate(preview_images[0]):
        img = img.resize((prepared.width, prepared.height), Image.Resampling.LANCZOS)
        sheet.paste(img, ((i % 4) * prepared.width, (i // 4) * prepared.height))
    preview_path = args.output_dir / f"{args.input.stem}_triposr_preview.jpg"
    sheet.save(preview_path, quality=92)

    print(f"obj={obj_path}")
    print(f"glb={glb_path}")
    print(f"preview={preview_path}")
    print(f"vertices={len(mesh.vertices)} faces={len(mesh.faces)}")


if __name__ == "__main__":
    main()
