from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import local_video_api as api
from lora_style_ab_test import DEFAULT_LORA, require_files, workflow_for_strength


TEST_NAME = "running_cycle_test_20260715"
TEST_ROOT = api.DATA_ROOT / "outputs" / TEST_NAME
REFERENCE = api.DATA_ROOT / "outputs" / "lora_style_ab_20260715" / "strength_0_65.png"
RUNNING_POSE = (
    "trot preparation | first frame only: complete full-body strict side profile in a natural neutral standing "
    "pose immediately before movement begins. All four paws are planted flat on one level ground line. The two "
    "front legs are vertical and separated; the two hind legs are naturally angled and separated. All four legs "
    "and all four paws are unobstructed, anatomically attached, and fully visible. The torso is horizontal and "
    "level, the full tail extends backward, and there is no stride, running motion, airborne paw, jump, or motion "
    "effect in this still image"
)
BASE_PROMPT = (
    "Preserve every identifying feature and the exact premium cute 3D rendering. The action is a controlled "
    "anatomically correct quadruped trot cycle, not a jump or gallop. Keep the camera, torso root, scale, side-view "
    "orientation, ground contact line and background stable."
)
WAN_PROMPT = f"""Animate this exact character into exactly one seamless two-beat trot-in-place cycle.

ANATOMICAL GAIT SEQUENCE: start in grounded contact phase A. The near front paw and far hind paw support the body while the opposite diagonal pair swings forward only a short distance. Then transition through a brief four-paw passing phase into grounded contact phase B, where the far front paw and near hind paw support the body. Return smoothly to the exact starting contact phase A at the final frame.

Each paw must stay attached to its original leg and follow one smooth low forward-and-back arc. Every stance paw remains planted without sliding until toe-off. All paws touch the same ground line. Never lift both front paws together, never lift both hind paws together, and never use a bunny-hop pattern. Keep the torso nearly level with only tiny natural vertical movement. Do not sit, stand still, jump, gallop, turn, move forward, change scale, or change perspective.

Preserve the exact face, large glossy eyes, ears, nose, muzzle, fur colors, marking placement, long groomed fur, body proportions, full tail, and premium cute 3D style. Keep the head and ears naturally coordinated with the trot without freezing any body region. Fixed side-view camera and fixed torso root.

{api.CHROMA_BACKGROUND_PROMPT} Keep the complete cat centered and fully inside the square frame in every frame."""
WAN_NEGATIVE = (
    api.WAN_NEGATIVE
    + ", "
    + api.CARTOON_WAN_NEGATIVE
    + ", "
    + api.CHROMA_NEGATIVE
    + ", bunny hop, hopping, floating, levitation, both front paws airborne, both hind paws airborne, same-side legs "
    "moving together, random gait, foot sliding, skating, leg swapping, detached paw, disappearing paw, extra paw, "
    "merged legs, crossed legs, rubber legs, sudden standing pose, sudden sitting pose, body turn, perspective change"
)


def link(node_id: str, slot: int = 0) -> list[Any]:
    return [node_id, slot]


def wan_cycle_workflow(input_name: str, seed: int, prefix: str, length: int = 61) -> dict[str, Any]:
    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": api.WAN_MODEL, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": api.WAN_CLIP, "type": "wan", "device": "default"},
        },
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": api.WAN_VAE}},
        "4": {"class_type": "LoadImage", "inputs": {"image": input_name}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": link("2"), "text": WAN_PROMPT}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": link("2"), "text": WAN_NEGATIVE}},
        "7": {
            "class_type": "Wan22ImageToVideoLatent",
            "inputs": {
                "vae": link("3"),
                "start_image": link("4"),
                "width": 512,
                "height": 512,
                "length": length,
                "batch_size": 1,
            },
        },
        "8": {"class_type": "ModelSamplingSD3", "inputs": {"model": link("1"), "shift": 8.0}},
        "9": {
            "class_type": "KSampler",
            "inputs": {
                "model": link("8"),
                "positive": link("5"),
                "negative": link("6"),
                "latent_image": link("7"),
                "seed": seed,
                "steps": 20,
                "cfg": 4.0,
                "sampler_name": "uni_pc",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        "10": {"class_type": "VAEDecode", "inputs": {"samples": link("9"), "vae": link("3")}},
        "11": {"class_type": "CreateVideo", "inputs": {"images": link("10"), "fps": api.FPS}},
        "12": {
            "class_type": "SaveVideo",
            "inputs": {"video": link("11"), "filename_prefix": prefix, "format": "auto", "codec": "auto"},
        },
    }


def run_keyframe(args: argparse.Namespace) -> Path:
    require_files(api.required_model_paths() + [api.MODEL_ROOT / "loras" / args.lora, REFERENCE])
    TEST_ROOT.mkdir(parents=True, exist_ok=True)
    comfy_dir = api.COMFY_INPUT / TEST_NAME
    comfy_dir.mkdir(parents=True, exist_ok=True)
    comfy_reference = comfy_dir / "reference.png"
    shutil.copy2(REFERENCE, comfy_reference)
    input_name = comfy_reference.relative_to(api.COMFY_INPUT).as_posix()
    prompt = api.build_qwen_prompt(BASE_PROMPT, RUNNING_POSE, transparent_gif=True)
    negative = (
        api.QWEN_NEGATIVE
        + ", "
        + api.CARTOON_NEGATIVE
        + ", "
        + api.CHROMA_NEGATIVE
        + ", airborne jump, both front paws lifted, both hind paws lifted, merged legs, crossed legs, hidden paw"
    )
    workflow = workflow_for_strength(
        input_name,
        prompt,
        negative,
        args.seed,
        f"{TEST_NAME}/grounded_keyframe",
        args.lora,
        args.strength,
    )
    (TEST_ROOT / "qwen_grounded_workflow.json").write_text(
        json.dumps(workflow, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    api.wait_comfy_idle()
    api.free_comfy_models()
    entry = api.queue_and_wait(workflow, "grounded running keyframe", timeout=7200)
    generated = api.first_output(entry, {".png", ".jpg", ".jpeg", ".webp"}, "grounded keyframe")
    raw = TEST_ROOT / "grounded_keyframe.png"
    cutout = TEST_ROOT / "grounded_keyframe_transparent.png"
    shutil.copy2(generated, raw)
    api.convert_chroma_image_to_transparent_png(raw, cutout, require_clean_background=True)
    wan_input = TEST_ROOT / "grounded_wan_input.png"
    api.prepare_landscape_keyframe(cutout, wan_input, (512, 512), background_rgb=(0, 255, 0))
    shutil.copy2(wan_input, comfy_dir / wan_input.name)
    api.free_comfy_models()
    print(f"keyframe={cutout}", flush=True)
    return wan_input


def make_two_cycle_video(source: Path, target: Path) -> None:
    filter_graph = (
        "[0:v]trim=start_frame=0:end_frame=60,setpts=PTS-STARTPTS,split=3[a][b][c];"
        "[a][b]concat=n=2:v=1:a=0[twice];"
        "[c]trim=start_frame=0:end_frame=1,setpts=PTS-STARTPTS[first];"
        "[twice][first]concat=n=2:v=1:a=0[out]"
    )
    process = subprocess.run(
        [
            api.FFMPEG_BIN,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-filter_complex",
            filter_graph,
            "-map",
            "[out]",
            "-frames:v",
            "121",
            "-r",
            "24",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(target),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode:
        raise RuntimeError(process.stderr[-4000:])


def run_video(args: argparse.Namespace) -> None:
    wan_input = TEST_ROOT / "grounded_wan_input.png"
    require_files(api.required_model_paths() + [wan_input])
    comfy_input = api.COMFY_INPUT / TEST_NAME / wan_input.name
    if not comfy_input.is_file():
        comfy_input.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(wan_input, comfy_input)
    workflow = wan_cycle_workflow(
        comfy_input.relative_to(api.COMFY_INPUT).as_posix(),
        args.seed + 10_000,
        f"{TEST_NAME}/wan_native_61f",
    )
    (TEST_ROOT / "wan_cycle_workflow.json").write_text(
        json.dumps(workflow, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    api.wait_comfy_idle()
    api.free_comfy_models()
    entry = api.queue_and_wait(workflow, "closed 61-frame trot cycle", timeout=21600)
    generated = api.first_output(entry, {".mp4", ".webm", ".mov", ".mkv"}, "trot cycle")
    cycle = TEST_ROOT / "native_trot_61f.mp4"
    looped = TEST_ROOT / "native_trot_two_cycles_121f.mp4"
    shutil.copy2(generated, cycle)
    make_two_cycle_video(cycle, looped)
    gif = TEST_ROOT / "native_trot_two_cycles_512.gif"
    apng = TEST_ROOT / "native_trot_two_cycles_512_soft.png"
    metrics = api.convert_chroma_video_to_transparent_animations(looped, gif, apng)
    (TEST_ROOT / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    api.free_comfy_models()
    print(f"cycle={cycle}", flush=True)
    print(f"looped={looped}", flush=True)
    print(f"gif={gif}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test a grounded closed quadruped trot cycle.")
    parser.add_argument("--stage", choices=("keyframe", "video", "all"), default="all")
    parser.add_argument("--lora", default=DEFAULT_LORA)
    parser.add_argument("--strength", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=20260716)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage in {"keyframe", "all"}:
        run_keyframe(args)
    if args.stage in {"video", "all"}:
        run_video(args)


if __name__ == "__main__":
    main()
