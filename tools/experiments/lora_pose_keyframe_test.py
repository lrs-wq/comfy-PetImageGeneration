from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.comfy_pet_pipeline.api import local_video as api  # noqa: E402
from tools.experiments.lora_style_ab_test import (  # noqa: E402
    DEFAULT_LORA,
    require_files,
    workflow_for_strength,
)


DEFAULT_REFERENCE = api.DATA_ROOT / "outputs" / "lora_style_ab_20260715" / "strength_0_65.png"
DEFAULT_OUTPUT = api.DATA_ROOT / "outputs" / "lora_pose_keyframes_20260715"
BASE_PROMPT = (
    "Preserve every identifying feature of the uploaded cat. Use smooth, coherent and natural motion. "
    "Keep the camera, subject root, scale and background stable. Preserve the same character identity "
    "and premium cute 3D rendering throughout the complete five-second animation."
)


def run(args: argparse.Namespace) -> None:
    reference = args.reference.resolve()
    output_dir = args.output.resolve()
    lora_path = api.MODEL_ROOT / "loras" / args.lora
    require_files(api.required_model_paths() + [lora_path, reference])

    output_dir.mkdir(parents=True, exist_ok=True)
    comfy_dir = api.COMFY_INPUT / "lora_pose_keyframes_20260715"
    comfy_dir.mkdir(parents=True, exist_ok=True)
    comfy_reference = comfy_dir / "reference.png"
    shutil.copy2(reference, comfy_reference)
    input_name = comfy_reference.relative_to(api.COMFY_INPUT).as_posix()

    negative = api.QWEN_NEGATIVE + ", " + api.CARTOON_NEGATIVE + ", " + api.CHROMA_NEGATIVE
    manifest = {
        "reference": str(reference),
        "lora": args.lora,
        "strength": args.strength,
        "seed": args.seed,
        "outputs": [],
    }

    api.wait_comfy_idle()
    api.free_comfy_models()
    try:
        for index, pose in enumerate(api.DEFAULT_POSES, start=1):
            prompt = api.build_qwen_prompt(BASE_PROMPT, pose, transparent_gif=True)
            workflow = workflow_for_strength(
                input_name,
                prompt,
                negative,
                args.seed + index,
                f"lora_pose_keyframes_20260715/pose_{index:02d}",
                args.lora,
                args.strength,
            )
            entry = api.queue_and_wait(workflow, f"pose {index}/4", timeout=7200)
            generated = api.first_output(
                entry, {".png", ".jpg", ".jpeg", ".webp"}, f"Qwen pose {index}"
            )
            target = output_dir / f"pose_{index:02d}.png"
            cutout = output_dir / f"pose_{index:02d}_transparent.png"
            shutil.copy2(generated, target)
            api.convert_chroma_image_to_transparent_png(target, cutout, require_clean_background=True)
            manifest["outputs"].append(
                {"index": index, "pose": pose, "image": str(target), "transparent": str(cutout)}
            )
            print(f"pose_{index:02d}: {target}", flush=True)
    finally:
        api.free_comfy_models()

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate four Qwen pose keyframes with the style LoRA.")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lora", default=DEFAULT_LORA)
    parser.add_argument("--strength", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
