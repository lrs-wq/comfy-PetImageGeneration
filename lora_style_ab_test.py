from __future__ import annotations

import argparse
import json
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

import local_video_api as api


DEFAULT_LORA = "pet3d_cat_qwen_image_r16.safetensors"
DEFAULT_INPUT = api.DATA_ROOT / "inputs" / "1.jpg"
DEFAULT_OUTPUT = api.DATA_ROOT / "outputs" / "lora_style_ab_20260715"


def link(node_id: str, slot: int = 0) -> list[Any]:
    return [node_id, slot]


def workflow_for_strength(
    input_name: str,
    prompt: str,
    negative_prompt: str,
    seed: int,
    prefix: str,
    lora_name: str,
    strength: float | None,
) -> dict[str, Any]:
    workflow = deepcopy(api.qwen_workflow(input_name, prompt, negative_prompt, seed, prefix))
    if strength is not None:
        workflow["17"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": link("13"),
                "lora_name": lora_name,
                "strength_model": strength,
            },
        }
        workflow["14"]["inputs"]["model"] = link("17")
    return workflow


def first_image(entry: dict[str, Any]) -> Path:
    return api.first_output(entry, {".png", ".jpg", ".jpeg", ".webp"}, "Qwen LoRA A/B")


def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required files are missing:\n" + "\n".join(missing))


def run(args: argparse.Namespace) -> None:
    source = args.input.resolve()
    output_dir = args.output.resolve()
    lora_path = api.MODEL_ROOT / "loras" / args.lora
    require_files(api.required_model_paths() + [lora_path, source])

    output_dir.mkdir(parents=True, exist_ok=True)
    comfy_dir = api.COMFY_INPUT / "lora_style_ab_20260715"
    comfy_dir.mkdir(parents=True, exist_ok=True)
    comfy_input = comfy_dir / "input.png"
    api.normalize_uploaded_image(source.read_bytes(), comfy_input)
    input_name = comfy_input.relative_to(api.COMFY_INPUT).as_posix()

    prompt = api.build_cartoon_reference_prompt()
    if args.trigger.strip():
        prompt += "\n\nStyle trigger: " + args.trigger.strip()
    negative = api.QWEN_NEGATIVE + ", " + api.CARTOON_NEGATIVE + ", " + api.CHROMA_NEGATIVE

    variants: list[tuple[str, float | None]] = [("baseline", None)]
    variants.extend((f"strength_{value:.2f}".replace(".", "_"), value) for value in args.strengths)

    manifest: dict[str, Any] = {
        "input": str(source),
        "seed": args.seed,
        "lora": args.lora,
        "trigger": args.trigger,
        "variants": [],
    }

    api.wait_comfy_idle()
    api.free_comfy_models()
    try:
        for name, strength in variants:
            prefix = f"lora_style_ab_20260715/{name}"
            workflow = workflow_for_strength(
                input_name,
                prompt,
                negative,
                args.seed,
                prefix,
                args.lora,
                strength,
            )
            (output_dir / f"workflow_{name}.json").write_text(
                json.dumps(workflow, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            entry = api.queue_and_wait(workflow, name, timeout=7200)
            generated = first_image(entry)
            target = output_dir / f"{name}.png"
            shutil.copy2(generated, target)
            manifest["variants"].append(
                {
                    "name": name,
                    "strength": strength,
                    "output": str(target),
                    "source_output": str(generated),
                }
            )
            print(f"{name}: {target}", flush=True)
    finally:
        api.free_comfy_models()

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed-seed Qwen style LoRA A/B images.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lora", default=DEFAULT_LORA)
    parser.add_argument("--strengths", nargs="+", type=float, default=[0.65, 0.85])
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--trigger", default="")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
