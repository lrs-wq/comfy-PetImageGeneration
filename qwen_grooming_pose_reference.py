#!/usr/bin/env python3
"""Create a pose-compatible identity reference for the SCAIL grooming test."""

from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

from PIL import Image

from transparent_animation import build_transparent_cutout

from scail2_subject_replacement_test import (
    COMFY_INPUT,
    EXPERIMENT_ROOT,
    api_json,
    extract_output_files,
    make_pose_and_mask_frames,
    read_gif_frames,
    wait_for_prompt,
)


SOURCE_WORKFLOW = Path(
    "/data/comfy-pet-pipeline/outputs/local_video_api/"
    "20260720T074158Z-b08c8dee83/workflows/qwen_pose_01.json"
)
SOURCE_GIF = Path("/root/comfy-pet-pipeline/cat/TM.gif")
CANONICAL_IMAGE = Path(
    "/data/comfy-pet-pipeline/comfyui/output/fur_lora_test/"
    "20260725_direct_sleek_detailz_w010_v2/eating_pose_00001_.png"
)

POSITIVE_PROMPT = """Use case: identity-preserving two-image pose transfer for a stylized 3D cat.

IMAGE ROLES — mandatory:
- Image 1 is the only identity, fur, face, body-proportion, and rendering-style reference.
- Image 2 is the only body-pose and camera-layout reference.
- Copy no identity, fur color, markings, collar, accessory, or character design from Image 2.

Create the exact same cat from Image 1 in the exact seated grooming pose from Image 2. Preserve Image 1's head silhouette, rounded cheeks, eye size, deep green eye color and spacing, ears, pink nose, muzzle, gray-white tabby colors and exact marking placement, compact body proportions, full tail, dense close-lying coat, and smooth rounded back silhouette. Preserve the source coat-length category without amplifying it. Fur must remain short, dense, uniformly directed and naturally close to the body, with only short soft edge hairs. Keep the premium cute 3D animated-film rendering style. The cat remains completely unclothed with no collar and no accessory.

ANATOMY AND POSE — non-negotiable:
Show exactly four attached legs total and exactly four paws total. There are exactly two forelegs total. One foreleg rises continuously and visibly from its shoulder to the single paw touching the mouth. The other foreleg descends continuously from its shoulder to exactly one planted front paw at the bottom. The raised foreleg has no second or leftover paw on the ground. At the bottom center there is exactly one front paw, not two. The two hind legs remain seated naturally at the sides and must not resemble extra front legs. Match Image 2's raised-paw grooming silhouette, head angle, body lean, framing, and scale. Do not blend Image 1's original two-front-paws-down pose with Image 2.

The cat is licking the raised front paw with a tiny visible pink tongue. Show one continuous cat silhouette, both ears, one tail from base to tip, and the complete uncropped body. No duplicated limb, no detached paw, no overlapping ghost limb, no extra subject, no food, and no bowl or other prop.

BACKGROUND — non-negotiable:
Every pixel outside the character must be a perfectly flat uniform pure #00FF00 green. No floor, cast shadow, contact shadow, reflection, texture, gradient, halo, text, logo, or watermark. Keep at least 12% clear green margin on every canvas edge."""

NEGATIVE_PROMPT = """different identity, changed face, changed fur colors, changed markings, changed eye color, changed ear shape, collar, necklace, clothing, accessory, food, kibble, bowl, dish, duplicate cat, extra subject, extra limb, extra leg, extra paw, duplicated foreleg, third front leg, three forelegs, five legs, two planted front paws, leftover ground paw from the raised leg, detached paw, floating paw, fused limbs, hidden shoulder connection, malformed anatomy, pose blend, ghost limb, cropped body, missing tail, fur volume expansion, puffed-up fur, raised guard hairs, bristling, spiky fur, needle-like fur, wire-like fur, rigid fur clumps, long isolated strands, dorsal ridge, mohawk, jagged back silhouette, changing back contour, fur crawling, texture crawling, plastic toy, flat 2D drawing, real photograph, white background, gray background, black background, non-green background, floor, shadow, reflection, background gradient, text, watermark"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow", type=Path, default=SOURCE_WORKFLOW)
    parser.add_argument("--source", type=Path, default=SOURCE_GIF)
    parser.add_argument("--canonical", type=Path, default=CANONICAL_IMAGE)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--timeout", type=int, default=1800)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.workflow.is_file():
        raise FileNotFoundError(args.workflow)
    if not args.source.is_file():
        raise FileNotFoundError(args.source)
    if not args.canonical.is_file():
        raise FileNotFoundError(args.canonical)

    run_id = (
        time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        + "-"
        + uuid.uuid4().hex[:8]
    )
    run_dir = EXPERIMENT_ROOT / "pose_reference" / run_id
    input_dir = COMFY_INPUT / "scail2_pose_fix" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    input_dir.mkdir(parents=True, exist_ok=False)

    source_frames, _ = read_gif_frames(args.source)
    normalized, _, source_metadata = make_pose_and_mask_frames(
        [source_frames[0]], (512, 512)
    )
    source_pose_path = input_dir / "source_grooming_pose.png"
    normalized[0].save(source_pose_path)

    canonical_path = input_dir / "canonical_subject.png"
    with Image.open(args.canonical) as canonical_source:
        canonical_source.convert("RGB").save(
            canonical_path,
            format="PNG",
            optimize=True,
        )

    graph = json.loads(args.workflow.read_text(encoding="utf-8"))
    graph["1"]["inputs"]["image"] = (
        canonical_path.resolve().relative_to(COMFY_INPUT.resolve()).as_posix()
    )
    graph["6"]["inputs"]["prompt"] = POSITIVE_PROMPT
    graph["7"]["inputs"]["prompt"] = NEGATIVE_PROMPT
    graph["14"]["inputs"]["seed"] = args.seed
    graph["16"]["inputs"]["filename_prefix"] = (
        f"scail2_pose_fix/{run_id}/grooming_reference_green"
    )
    graph["20"] = {
        "class_type": "LoadImage",
        "inputs": {
            "image": source_pose_path.resolve()
            .relative_to(COMFY_INPUT.resolve())
            .as_posix()
        },
    }
    graph["21"] = {
        "class_type": "FluxKontextImageScale",
        "inputs": {"image": ["20", 0]},
    }
    graph["6"]["inputs"]["image2"] = ["21", 0]
    graph["7"]["inputs"]["image2"] = ["21", 0]

    manifest = {
        "run_id": run_id,
        "source_workflow": str(args.workflow),
        "canonical_source": str(args.canonical),
        "canonical_input": str(canonical_path),
        "source_pose": str(source_pose_path),
        "seed": args.seed,
        "positive_prompt": POSITIVE_PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "source_metadata": source_metadata,
    }
    (run_dir / "workflow_api.json").write_text(
        json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    response = api_json("/prompt", {"prompt": graph, "client_id": run_id})
    if response.get("node_errors"):
        raise RuntimeError(
            json.dumps(response["node_errors"], ensure_ascii=False, indent=2)
        )
    prompt_id = str(response["prompt_id"])
    manifest["prompt_id"] = prompt_id
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Submitted Qwen pose-reference prompt: {prompt_id}", flush=True)

    started = time.monotonic()
    history = wait_for_prompt(prompt_id, args.timeout)
    outputs = extract_output_files(history.get("outputs", {}))
    images = [
        item
        for item in outputs
        if item["filename"].lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    ]
    if not images:
        raise RuntimeError(
            "Qwen workflow completed without a saved image: "
            + json.dumps(history.get("outputs", {}), ensure_ascii=False)
        )
    saved = images[-1]
    if saved.get("type") != "output":
        raise RuntimeError(f"Unexpected Qwen output type: {saved}")
    green_path = (
        Path("/data/comfy-pet-pipeline/comfyui/output")
        / saved.get("subfolder", "")
        / saved["filename"]
    )
    transparent_path = run_dir / "grooming_reference_transparent.png"
    cutout_metrics = build_transparent_cutout(
        green_path,
        transparent_path,
        require_clean_background=False,
    )
    manifest.update(
        {
            "status": "completed",
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "green_output": str(green_path),
            "transparent_output": str(transparent_path),
            "cutout_metrics": cutout_metrics,
            "outputs": outputs,
        }
    )
    (run_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
