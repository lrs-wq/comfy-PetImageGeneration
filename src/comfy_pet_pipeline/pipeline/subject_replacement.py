#!/usr/bin/env python3
"""Run a minimal SCAIL-2 character-replacement test through ComfyUI.

The script prepares a short motion clip and exact SCAIL identity masks from
alpha-enabled assets, submits an API-format ComfyUI graph, and records the
result plus runtime diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageSequence

try:
    import av
except ModuleNotFoundError:  # CPU-only guide/tests do not decode video containers.
    av = None

from ..media.transparent_animation import (
    GIF_ALPHA_SMOOTHING_SIGMA,
    GIF_ALPHA_THRESHOLD,
    _binary_alpha_edge_metrics,
    _encode_soft_alpha_apng,
    _gif_alpha,
    _save_rgba,
    _subject_mask,
    validate_animation,
)

COMFY_URL = "http://127.0.0.1:8188"
COMFY_INPUT = Path("/data/comfy-pet-pipeline/comfyui/input")
COMFY_OUTPUT = Path("/data/comfy-pet-pipeline/comfyui/output")
EXPERIMENT_ROOT = Path("/data/comfy-pet-pipeline/experiments/scail2_tm_test")
SOURCE_ANIMATION = Path(
    "/data/comfy-pet-pipeline/profile_assets/sleek_tabby_cat_v1/driving/"
    "20260808-user-white-model-v1/"
    "grooming_white_model_driver_512x512_81f_16fps.mp4"
)
REFERENCE_IMAGE = Path(
    "/data/comfy-pet-pipeline/comfyui/output/fur_lora_test/"
    "20260725_direct_sleek_detailz_w010_v2/eating_pose_00001_.png"
)

UNET_NAME = "wan2.1_14B_SCAIL_2_fp8_scaled.safetensors"
DPO_LORA = "wan2.1_SCAIL_2_DPO_lora_bf16.safetensors"
DISTILL_LORA = (
    "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"
)
TEXT_ENCODER = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
VAE_NAME = "wan_2.1_vae.safetensors"
CLIP_VISION = "clip_vision_h.safetensors"
EATING_POSE_STRENGTH_MAX = 1.35
EATING_TARGET_MUZZLE_FOOD_OVERLAP_PX = 8.0
EATING_MUZZLE_FOOD_OVERLAP_MIN_PX = 6
EATING_MUZZLE_FOOD_OVERLAP_MAX_PX = 10
EATING_FULL_POSE_OVERALL_CHANGE_RATIO_MIN = 0.02
EATING_FULL_POSE_LOWER_HALF_CHANGE_RATIO_MIN = 0.01
# The digest-bound dog loop keeps its paws planted while the head completes a
# down-up-down eating cycle. Its audited lower-quarter XOR is 0.011581, so a
# 1.1% floor preserves the anti-static gate without rejecting that motion.
EATING_FULL_POSE_LOWER_QUARTER_CHANGE_RATIO_MIN = 0.011
DRINKING_REFERENCE_ALIGNED_POSE_POLICY = (
    "reference_space_white_driver_head_path_static_bowl_v3"
)
DRINKING_FULL_WHITE_DRIVER_POSE_POLICY = (
    "full_white_driver_drinking_pose_static_bowl_v2"
)
# Kept as an import-compatible alias for callers that previously treated the
# complete driver as a fallback.  It is now the only high-quality dog drinking
# geometry policy.
DRINKING_FULL_WHITE_DRIVER_FALLBACK_POLICY = (
    DRINKING_FULL_WHITE_DRIVER_POSE_POLICY
)
DRINKING_FULL_DRIVER_CHANGE_RATIO_MIN = 0.01
DRINKING_REFERENCE_HEAD_LIFT_PX = 36.0
DRINKING_REFERENCE_HEAD_LIFT_MIN_PX = 14.0
DRINKING_REFERENCE_LAP_CYCLES = 3
EATING_REFERENCE_HEAD_LIFT_PX = 36.0
GROOMING_TEMPORAL_LOOP_POLICY = (
    "grooming_second_half_ping_pong_closed_loop_v1"
)
ACTION_TAIL_GEOMETRY_SAFE_LOOP_POLICY = (
    "drinking_eating_terminal_phase_snap_to_first_v2"
)
DOG_EATING_CANONICAL_FULL_DRIVER_POSE_POLICY = (
    "canonical_identity_full_white_driver_eating_fallback_v1"
)
DOG_EATING_REFERENCE_FULL_DRIVER_RECOVERY_POSE_POLICY = (
    "reference_identity_full_white_driver_eating_recovery_v1"
)

DEFAULT_PROMPT = (
    "A complete full-body premium stylized 3D cartoon cat matching the sleek "
    "reference as the only identity standard. Preserve the exact face shape, "
    "cheek width, eye color and spacing, pupils, nose, muzzle, ear shape, exact "
    "reference coat colors and marking placement, compact body volume, full tail, "
    "and exact smooth rounded back silhouette. Preserve the source coat-length "
    "category without amplifying it. Fur is dense, compact, uniformly directed "
    "and naturally close to the body with only short soft edge hairs. Use the "
    "neutral-white anatomy-maquette driving clip only for motion timing, joint "
    "trajectories, seated pose, raised-paw path, camera and framing. The driving "
    "maquette has no identity or appearance authority: never copy its face, "
    "eyes, white surface, body proportions, material, lighting or texture. If "
    "the supplied identity reference is genuinely white, preserve that white "
    "coat because the identity reference requires it, never because the driver "
    "does. The cat "
    "follows the supplied grooming motion, pose sequence and timing "
    "precisely, sitting and slowly licking one raised front paw and rubbing its "
    "face. Motion must never change fur length, volume, grooming direction or "
    "the clean back contour. "
    "Anatomically correct feline body and joints with exactly four legs and "
    "four paws total. There are exactly two forelegs: one foreleg rises "
    "continuously from its shoulder to the paw at the mouth, while the other "
    "foreleg ends in the only planted front paw at the bottom. Never leave a "
    "second ground paw for the raised foreleg and never create a third "
    "foreleg. Keep the two seated hind legs distinct from the front legs. "
    "The supplied reference image already contains this one-raised, one-planted "
    "foreleg topology. Animate those same two forelegs from the first frame; "
    "never preserve two planted forepaws and then add a new raised paw. "
    "Stable face and identity without freezing it, no food, no bowl, no prop, "
    "no clothes, no collar and no accessories. Static camera, centered complete "
    "uncropped body, clean uniform white studio background, smooth temporally "
    "coherent motion. Separate pale fur from the background only with natural "
    "soft tonal lighting, never with a drawn silhouette border."
)

DEFAULT_NEGATIVE_PROMPT = (
    "extra limb, extra leg, extra paw, duplicated limb, duplicated foreleg, "
    "three front legs, three forelegs, five legs, detached paw, floating paw, "
    "fused legs, malformed anatomy, retained old paw, pose ghosting, multiple "
    "cats, distorted face, changing fur markings, food, kibble, bowl, dish, "
    "copied driving-maquette face, copied driving-maquette eyes, driver identity, "
    "driver material, driver lighting, uniform maquette surface, "
    "fur volume expansion, puffed-up fur, raised guard hairs, bristling, spiky "
    "fur, needle-like fur, wire-like fur, rigid fur clumps, long isolated "
    "strands, dorsal ridge, mohawk, jagged back silhouette, changing back "
    "contour, missing back, clipped back, fur crawling, texture crawling, "
    "black contour, dark outline, ink line, comic stroke, silhouette halo, "
    "camera movement, zoom, flicker, jitter"
)

EATING_PROMPT = (
    "A complete full-body premium stylized 3D cartoon cat matching the "
    "supplied sleek reference as the only identity standard. Preserve the "
    "exact face shape, cheek width, eye color and spacing, pupils, nose, "
    "muzzle, ear shape, coat colors and marking placement, body volume, paws, "
    "tail, and the exact smooth rounded back silhouette. Preserve the source "
    "coat-length category without amplifying it. The coat is dense, compact, "
    "soft, uniformly directed and lies naturally close to the body along the "
    "spine, shoulders, flanks and chest, with only short soft edge hairs. "
    "Use the neutral-white anatomy-maquette driving clip only for low-head "
    "motion timing, joint trajectories, the fixed bowl, camera and framing. "
    "Never copy the maquette's face, eyes, white surface, body proportions, "
    "material, lighting or texture; the supplied identity reference alone "
    "controls every appearance trait. If that identity is genuinely white, "
    "keep it white because the identity reference requires it, never because "
    "the driver is white. "
    "Motion may move the joints, head, neck, ears, muzzle and chest naturally "
    "but must never change fur length, fur volume, grooming direction or the "
    "back contour. Follow the supplied eating motion, pose sequence and timing "
    "precisely. The cat repeatedly lowers its muzzle a small natural distance "
    "into exactly one shallow light-beige food bowl and raises it again. Keep "
    "the bowl stationary, unchanged and below the muzzle; all kibble stays "
    "inside the bowl below the rim. The bowl placement and horizontal "
    "coordinate follow the driving template exactly, and the projected muzzle "
    "center stays over the bowl opening during every low-head eating phase. "
    "Exactly four attached legs and four paws "
    "total. Exactly two forelegs remain continuously attached to the shoulders "
    "and end in exactly two planted front paws beside or behind the bowl "
    "without defining or shifting the bowl center; "
    "no foreleg is raised, hidden, retained twice, or merged with the bowl. "
    "Keep the recognizable face temporally stable without freezing it. Static "
    "camera, stable scale and placement, complete uncropped subject and tail, "
    "clean perfectly uniform white studio background, no floor shadow, smooth "
    "temporally coherent motion. Separate pale fur from the background only "
    "with natural soft tonal lighting, never with a drawn silhouette border."
)

EATING_NEGATIVE_PROMPT = (
    "extra limb, extra leg, extra paw, duplicated limb, duplicated foreleg, "
    "third foreleg, retained old paw, missing paw, hidden paw, fused legs, "
    "detached paw, floating paw, raised front paw, paw holding food, bowl "
    "replacing a paw, paw merged into bowl, duplicate bowl, extra bowl, "
    "changing bowl, deformed bowl, floating kibble, food outside bowl, food "
    "stuck to face, stretched mouth, distorted muzzle, melted face, changing "
    "identity, copied driving-maquette face, copied driving-maquette eyes, "
    "driver identity, driver material, driver lighting, uniform maquette surface, "
    "changing fur markings, fur crawling, texture crawling, fur "
    "volume expansion, puffed-up fur, raised fur, standing guard hairs, "
    "bristling, spiky fur, needle-like fur, wire-like fur, rigid fur clumps, "
    "long isolated strands, hanging tufts, dorsal ridge, mohawk, jagged fur "
    "silhouette, changing back contour, missing back, clipped back, multiple "
    "cats, cropped body, cropped tail, camera movement, zoom, flicker, jitter, "
    "black contour, dark outline, ink line, comic stroke, silhouette halo, "
    "green background, floor, shadow, gradient, texture, text, watermark"
)

DOG_EATING_PROMPT = (
    "A complete full-body restrained stylized 3D dog matching the supplied "
    "reference as the only identity and appearance authority. Preserve the exact "
    "breed traits, apparent age, skull width, forehead slope, cheek width, exact "
    "muzzle length, width and taper, eye size, color and spacing, "
    "ear shape and carriage, nose, coat colors, localized markings, natural coat "
    "length and volume, body proportions, paws, and complete tail. Use the "
    "neutral-white dog anatomy driver only for eating motion timing, joint "
    "trajectories, the fixed shallow kibble bowl, camera and framing. Never copy "
    "the driver's white coat, face, eyes, proportions, smooth material, or lighting. "
    "Follow the supplied eating sequence precisely: lower the dog's muzzle naturally "
    "into exactly one fixed shallow bowl and raise it again. The bowl contains only a "
    "small, low, level layer of ordinary brown dry kibble fully below the rim. Never "
    "form a heap, mound, cone, pyramid, fused mass, oversized pile, overflow, or food "
    "outside the bowl. The bowl, rim, low food surface and every kibble position remain "
    "fixed and unchanged in every frame; only the muzzle may naturally occlude them. "
    "Keep all four legs anatomically attached and stable. Keep identity, coat markings, "
    "scale, camera and background temporally stable with the complete dog uncropped. "
    "Preserve an adult as an adult: never shorten or round the muzzle, enlarge the "
    "head or eyes, or add a puppy-like panting smile. The entire background remains "
    "perfectly uniform pure #00FF00 chroma-key green in every frame, with no floor, "
    "horizon, shadow, reflection, gradient, texture, flicker, or green spill on fur."
)

DOG_STANDING_PROMPT = (
    "A complete full-body restrained stylized 3D dog matching the supplied "
    "reference as the only identity and appearance authority. Preserve the exact "
    "breed traits, apparent age, face, muzzle, eyes, ears, nose, coat colors and "
    "markings, natural coat length and volume, body proportions, paws, and complete "
    "tail. Use the neutral-white dog anatomy driver only for the standing action's "
    "joint trajectories, driver-defined seated-or-upright idle geometry, gentle tail "
    "motion, relaxed panting mouth motion, camera, scale and framing. The standing "
    "idle has a visibly open natural mouth with a readable dark mouth cavity and "
    "an anatomically attached pink tongue. Follow the driver so the tongue gently "
    "extends, retracts and softly bobs in a periodic cycle; never freeze it, paste "
    "it onto the muzzle, or show it passing through closed lips. Never copy the "
    "driver's white coat, face, eyes, body "
    "proportions, material, or lighting. Keep one complete unclothed dog, stable "
    "identity, coherent anatomy and a static camera. No prop, collar, harness, "
    "clothing, floor, shadow, text, logo, or extra subject. Preserve the exact muzzle "
    "length, head-to-body ratio, eye size and apparent age while allowing only this "
    "action-required open-mouth panting motion; never make an adult look like a puppy. "
    "The entire background remains "
    "perfectly uniform pure #00FF00 chroma-key green in every frame, with no floor, "
    "horizon, shadow, reflection, gradient, texture, flicker, or green spill on fur."
)

DOG_DRINKING_PROMPT = (
    "A complete full-body restrained stylized 3D dog matching the supplied "
    "reference as the only identity and appearance authority. Preserve the exact "
    "breed traits, apparent age, face and muzzle geometry, eyes, ears, nose, coat "
    "colors and localized markings, natural coat length and volume, body "
    "proportions, paws, and complete tail. The prepared pose video is synthesized "
    "from this same approved reference and controls only a small head-lapping motion; "
    "its body, ears, paws and tail stay locked to the reference geometry. Never "
    "invent a curled tail when the reference tail is straight, change folded ears to "
    "upright ears, shorten the muzzle, or import any neutral-white driver morphology. "
    "Keep the bowl and still water fixed. "
    "Animate one restrained natural lapping cycle with only a tiny anatomically "
    "attached pink tongue tip touching the water, subtle muzzle and neck movement, "
    "and no splash, drool or long tongue. Keep all four legs attached and stable. "
    "The entire background remains perfectly uniform pure #00FF00 chroma-key green "
    "with no floor, horizon, shadow, reflection, gradient, texture, flicker, or "
    "green spill on fur."
)

DOG_RUNNING_PROMPT = (
    "A complete full-body restrained stylized 3D dog matching the supplied "
    "reference as the only identity and appearance authority. Preserve the exact "
    "breed traits, apparent age, face, muzzle, eyes, ears, nose, coat colors and "
    "localized markings, natural coat length and volume, body proportions, paws, "
    "and complete tail. Use the neutral-white dog anatomy driver only for the "
    "quadruped running-in-place gait, joint trajectories, camera, scale and framing. "
    "Never copy the driver's white coat, face, proportions, material, or lighting. "
    "Follow the supplied gait precisely with exactly four distinct anatomically "
    "attached legs and complete paws, coherent spine flexion, stable identity, and "
    "the tail continuously attached from base to tip. Keep the global body center "
    "stable with no forward travel, scene cut, fused legs, missing paw or extra limb. "
    "No prop. The entire background remains perfectly uniform pure #00FF00 green "
    "with no floor, horizon, shadow, reflection, gradient, texture or flicker."
)

DOG_SLEEPING_PROMPT = (
    "A complete full-body restrained stylized 3D dog matching the supplied "
    "reference as the only identity and appearance authority. Preserve the exact "
    "breed traits, apparent age, face and muzzle geometry, eyes, ears, nose, coat "
    "colors and localized markings, natural coat length and volume, body "
    "proportions, paws, and complete tail. Use the neutral-white dog anatomy driver "
    "only for its compact curled sleeping geometry, slow breathing, subtle head and "
    "ear motion, one natural restrained yawn, camera, scale and framing. Never copy "
    "the driver's white coat, face, proportions, material, or lighting. Keep all "
    "legs anatomically attached and naturally tucked, eyes peacefully closed outside "
    "the brief yawn, and the complete tail curled continuously beside the body. No "
    "bed, cushion, blanket or other prop. The entire background remains perfectly "
    "uniform pure #00FF00 green with no floor, shadow, reflection, gradient, texture "
    "or flicker."
)

DOG_TOILETING_PROMPT = (
    "A complete full-body restrained stylized 3D dog matching the supplied "
    "reference as the only identity and appearance authority. Preserve the exact "
    "breed traits, apparent age, face, muzzle, eyes, ears, nose, coat colors and "
    "localized markings, natural coat length and volume, body proportions, paws, "
    "and complete tail. Use the neutral-white dog anatomy driver only for the stable "
    "seated toilet action, joint trajectories, one fixed clean rectangular pet toilet "
    "tray with grid insert, camera, scale and framing. Never copy the driver's white "
    "coat, face, proportions, material, or lighting. Keep the tray fixed, complete "
    "and unchanged beneath the dog; show no urine, feces, mess, litter or extra prop. "
    "Keep all four legs anatomically attached, identity stable and the complete tail "
    "visible. Render natural softly shaded fur and tray edges without any black, "
    "inked, comic, cel-shaded, or contour-stroke outline. Keep the tray rim pale "
    "neutral gray with no dark border. The entire background outside the dog and "
    "tray remains perfectly "
    "uniform pure #00FF00 green with no floor continuation, shadow, reflection, "
    "gradient, texture or flicker."
)

DOG_NEGATIVE_PROMPT = (
    "cat, feline, kitten, different dog, different breed, generic dog, copied "
    "white-model identity, white-model face, white-model coat, changed face, changed "
    "muzzle, shortened muzzle, shortened snout, rounded baby face, puppy when source "
    "is adult, younger dog, enlarged head, oversized eyes, enlarged irises, panting "
    "smile, tongue out, changed ears, changed eye color, changed coat color, changed markings, "
    "changed coat length, changed body proportions, extra limb, missing limb, "
    "duplicated limb, detached paw, fused legs, kibble heap, kibble mound, food cone, "
    "food pyramid, oversized kibble, overfilled bowl, overflowing food, fused food "
    "mass, food outside bowl, moving food, changing food surface, growing food, "
    "shrinking food, disappearing food, extra tail, duplicate subject, "
    "collar, harness, clothing, accessory, camera movement, zoom, scene cut, flicker, "
    "white background, gray background, black background, non-green background, floor, "
    "floor line, horizon, shadow, reflection, background gradient, background texture, "
    "green bounce light, green color spill, green rim light, cyan rim light, text, logo, watermark"
)
DOG_EATING_NEGATIVE_PROMPT = DOG_NEGATIVE_PROMPT
DOG_STANDING_NEGATIVE_PROMPT = DOG_NEGATIVE_PROMPT.replace(
    "panting smile, tongue out, ",
    "",
) + (
    ", closed lips around visible tongue, tongue through closed mouth, no mouth cavity, "
    "static tongue, frozen tongue, pasted tongue, detached tongue, rigid tongue, "
    "melting tongue, disappearing tongue"
)
DOG_DRINKING_NEGATIVE_PROMPT = DOG_NEGATIVE_PROMPT.replace(
    "panting smile, tongue out, ",
    "panting smile, ",
) + (
    ", long tongue, detached tongue, transparent tongue, splash, drool, water "
    "strand, moving bowl, changing bowl, duplicate bowl, food in water bowl, "
    "curled tail when reference tail is straight, upright ears when reference ears "
    "are folded, copied white-model silhouette"
)
DOG_RUNNING_NEGATIVE_PROMPT = DOG_NEGATIVE_PROMPT + (
    ", two-legged run, bipedal pose, fused legs, hidden far-side leg, missing paw, "
    "detached paw, body translation, forward travel, long airborne suspension"
)
DOG_SLEEPING_NEGATIVE_PROMPT = DOG_NEGATIVE_PROMPT + (
    ", open alert eyes, standing pose, sitting upright, walking, running, bed, "
    "cushion, blanket, detached tucked paw, missing tail"
)
DOG_TOILETING_NEGATIVE_PROMPT = DOG_NEGATIVE_PROMPT + (
    ", urine, feces, excrement, mess, litter, dirty tray, duplicate tray, litter box, "
    "moving tray, changing tray, tray fused into body, black outline, dark outline, "
    "ink outline, comic outline, contour stroke, cel-shaded border, thick border, "
    "dark tray rim"
)

DOG_ACTION_PROMPTS = {
    "eating": DOG_EATING_PROMPT,
    "drinking": DOG_DRINKING_PROMPT,
    "running": DOG_RUNNING_PROMPT,
    "sleeping": DOG_SLEEPING_PROMPT,
    "standing": DOG_STANDING_PROMPT,
    "toileting": DOG_TOILETING_PROMPT,
}
DOG_ACTION_NEGATIVE_PROMPTS = {
    "eating": DOG_EATING_NEGATIVE_PROMPT,
    "drinking": DOG_DRINKING_NEGATIVE_PROMPT,
    "running": DOG_RUNNING_NEGATIVE_PROMPT,
    "sleeping": DOG_SLEEPING_NEGATIVE_PROMPT,
    "standing": DOG_STANDING_NEGATIVE_PROMPT,
    "toileting": DOG_TOILETING_NEGATIVE_PROMPT,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE_ANIMATION)
    parser.add_argument("--reference", type=Path, default=REFERENCE_IMAGE)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--frames", type=int, default=33)
    parser.add_argument("--fps", type=float, default=16.0)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument(
        "--run-id",
        help="Optional deterministic run id for API integration.",
    )
    parser.add_argument(
        "--action",
        choices=(
            "grooming",
            "eating",
            "drinking",
            "running",
            "sleeping",
            "standing",
            "toileting",
        ),
        default="grooming",
    )
    parser.add_argument("--species", choices=("cat", "dog"), default="cat")
    parser.add_argument("--prompt")
    parser.add_argument("--negative-prompt")
    parser.add_argument(
        "--prompt-suffix",
        help="Optional user constraint appended after the protected action prompt.",
    )
    parser.add_argument("--pose-strength", type=float, default=1.0)
    parser.add_argument(
        "--transparent-gif",
        action="store_true",
        help=(
            "Create a 100-frame, 20 FPS transparent GIF with semantic "
            "foreground extraction; the model itself keeps a stable neutral "
            "background."
        ),
    )
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--canonical-identity-full-driver-eating-fallback",
        action="store_true",
        help=(
            "Dog-eating recovery only: keep --reference as an identity-only "
            "mask and take the complete animated body and bowl geometry from "
            "the digest-bound white-maquette --source."
        ),
    )
    parser.add_argument(
        "--eating-geometry-guide-output",
        type=Path,
        help=(
            "CPU-only mode: derive an identity-free low-muzzle eating geometry "
            "guide from --source and exit before creating a SCAIL run."
        ),
    )
    parser.add_argument(
        "--eating-geometry-guide-metadata",
        type=Path,
        help="Optional JSON metadata path for --eating-geometry-guide-output.",
    )
    parser.add_argument(
        "--grooming-geometry-guide-output",
        type=Path,
        help=(
            "CPU-only mode: derive a texture-free raised-paw grooming geometry "
            "guide from --source and exit before creating a SCAIL run."
        ),
    )
    parser.add_argument(
        "--grooming-geometry-guide-metadata",
        type=Path,
        help="Optional JSON metadata path for --grooming-geometry-guide-output.",
    )
    args = parser.parse_args()
    if args.canonical_identity_full_driver_eating_fallback and not (
        args.species == "dog" and args.action == "eating"
    ):
        parser.error(
            "--canonical-identity-full-driver-eating-fallback requires "
            "--species dog --action eating"
        )
    if args.prompt is None:
        if args.species == "dog":
            args.prompt = DOG_ACTION_PROMPTS.get(args.action, DOG_STANDING_PROMPT)
        else:
            args.prompt = EATING_PROMPT if args.action == "eating" else DEFAULT_PROMPT
    if args.negative_prompt is None:
        if args.species == "dog":
            args.negative_prompt = DOG_ACTION_NEGATIVE_PROMPTS.get(
                args.action,
                DOG_STANDING_NEGATIVE_PROMPT,
            )
        else:
            args.negative_prompt = (
                EATING_NEGATIVE_PROMPT
                if args.action == "eating"
                else DEFAULT_NEGATIVE_PROMPT
            )
    if args.prompt_suffix and args.prompt_suffix.strip():
        args.prompt += (
            "\n\nAdditional user constraint. This may refine mood or secondary "
            "motion but cannot override identity, anatomy, action, fur, camera, "
            "background, or prop constraints: "
            + args.prompt_suffix.strip()
        )
    return args


def api_json(path: str, payload: dict[str, Any] | None = None) -> Any:
    url = f"{COMFY_URL}{path}"
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="GET" if body is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def require_files(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required file(s): " + ", ".join(missing))


def read_gif_frames(path: Path) -> tuple[list[Image.Image], list[int]]:
    with Image.open(path) as source:
        frames: list[Image.Image] = []
        durations: list[int] = []
        default_duration = max(1, int(source.info.get("duration", 100)))
        for frame in ImageSequence.Iterator(source):
            frames.append(frame.convert("RGBA").copy())
            durations.append(max(1, int(frame.info.get("duration", default_duration))))
    if not frames:
        raise ValueError(f"No frames decoded from {path}")
    return frames, durations


def read_animation_frames(
    path: Path,
    *,
    extract_green_screen: bool = True,
) -> tuple[list[Image.Image], list[int]]:
    """Decode a GIF or a constant-frame-rate video into timed RGBA frames."""
    if path.suffix.lower() == ".gif":
        return read_gif_frames(path)

    supported_video_suffixes = {".avi", ".m4v", ".mov", ".mp4", ".webm"}
    if path.suffix.lower() not in supported_video_suffixes:
        raise ValueError(
            f"Unsupported driving animation format for {path}; "
            "expected GIF, AVI, M4V, MOV, MP4, or WebM."
        )

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Unable to decode driving animation {path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(source_fps) or source_fps <= 0.0:
        capture.release()
        raise ValueError(f"Driving animation has an invalid frame rate: {path}")
    frame_duration_ms = max(1, int(round(1000.0 / source_fps)))
    frames: list[Image.Image] = []
    try:
        while True:
            decoded, bgr = capture.read()
            if not decoded:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if extract_green_screen:
                clean_rgb, alpha = _subject_mask(rgb)
            else:
                # The white-maquette semantic pass needs the original green
                # direction to distinguish a connected floor shadow from the
                # neutral dog and action prop.  The generic extractor removes
                # that evidence while de-spilling its matte boundary.
                clean_rgb = rgb
                alpha = np.full(rgb.shape[:2], 255, dtype=np.uint8)
            rgba = np.dstack((clean_rgb, alpha))
            frames.append(Image.fromarray(rgba, mode="RGBA"))
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"No frames decoded from {path}")
    return frames, [frame_duration_ms] * len(frames)


def sample_timed_frames(
    frames: list[Image.Image],
    durations_ms: list[int],
    output_count: int,
    output_fps: float,
) -> list[Image.Image]:
    cumulative = np.cumsum(np.asarray(durations_ms, dtype=np.float64))
    total_ms = float(cumulative[-1])
    selected: list[Image.Image] = []
    for index in range(output_count):
        timestamp_ms = min(index * 1000.0 / output_fps, total_ms - 0.001)
        source_index = int(np.searchsorted(cumulative, timestamp_ms, side="right"))
        selected.append(frames[min(source_index, len(frames) - 1)].copy())
    return selected


def union_alpha_bbox(frames: Iterable[Image.Image], threshold: int = 8) -> tuple[int, int, int, int]:
    bbox: tuple[int, int, int, int] | None = None
    for frame in frames:
        alpha = frame.getchannel("A").point(lambda value: 255 if value > threshold else 0)
        current = alpha.getbbox()
        if not current:
            continue
        if bbox is None:
            bbox = current
        else:
            bbox = (
                min(bbox[0], current[0]),
                min(bbox[1], current[1]),
                max(bbox[2], current[2]),
                max(bbox[3], current[3]),
            )
    if bbox is None:
        raise ValueError("The source has no visible alpha subject")
    return bbox


def expand_bbox(
    bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
    margin_fraction: float = 0.05,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = bbox
    margin = int(round(max(right - left, bottom - top) * margin_fraction))
    return (
        max(0, left - margin),
        max(0, top - margin),
        min(image_size[0], right + margin),
        min(image_size[1], bottom + margin),
    )


def contain_rgba(
    image: Image.Image,
    canvas_size: tuple[int, int],
    scale_fraction: float,
) -> Image.Image:
    rgba = image.convert("RGBA")
    max_width = max(1, int(canvas_size[0] * scale_fraction))
    max_height = max(1, int(canvas_size[1] * scale_fraction))
    factor = min(max_width / rgba.width, max_height / rgba.height)
    resized = rgba.resize(
        (
            max(1, int(round(rgba.width * factor))),
            max(1, int(round(rgba.height * factor))),
        ),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    position = (
        (canvas_size[0] - resized.width) // 2,
        (canvas_size[1] - resized.height) // 2,
    )
    canvas.alpha_composite(resized, position)
    return canvas


def white_maquette_semantic_silhouette(
    rgba: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Keep neutral model/action geometry while removing green floor shadows."""
    pixels = np.asarray(rgba, dtype=np.uint8)
    if pixels.ndim != 3 or pixels.shape[2] != 4:
        raise ValueError("White-maquette cleanup requires an RGBA image")
    rgb = pixels[:, :, :3]
    alpha = pixels[:, :, 3]
    red = rgb[:, :, 0].astype(np.int16)
    green = rgb[:, :, 1].astype(np.int16)
    blue = rgb[:, :, 2].astype(np.int16)
    channel_span = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(
        np.int16
    )
    luma = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    green_shadow = ((green - red) >= 12) & ((green - blue) >= 24)
    # White fur next to a green screen can keep enough reflected green to
    # satisfy the shadow rule even though its red/blue channels remain much
    # brighter and less saturated than the background.  Preserve that narrow
    # high-luma band so chest and foreleg geometry are not cut apart.  The
    # stricter red/blue and span limits continue to reject the green canvas and
    # its darker floor shadow.
    recoverable_green_spill = (
        green_shadow
        & (red >= 145)
        & (blue >= 110)
        & (luma >= 145)
        & (channel_span <= 88)
    )
    seed = (
        (alpha >= 16)
        & (luma >= 55)
        & (
            ((channel_span <= 72) & ~green_shadow)
            | recoverable_green_spill
        )
    ).astype(np.uint8)
    # Thin neutral/green contact shadows can survive chroma extraction and
    # connect every paw to a fake floor shelf.  A vertical opening removes
    # those shelves while retaining legs, tails and substantial action props
    # such as a toilet tray.  Skipping this opening for horizontal props also
    # preserves the floor whenever it touches the prop.
    seed = cv2.morphologyEx(
        seed,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 9)),
        iterations=1,
    )
    seed = cv2.morphologyEx(
        seed,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=2,
    )
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        seed,
        connectivity=8,
    )
    areas = [
        int(stats[label, cv2.CC_STAT_AREA])
        for label in range(1, component_count)
    ]
    largest_area = max(areas, default=0)
    minimum_area = max(64, int(largest_area * 0.004))
    semantic_seed = np.zeros_like(seed)
    retained_components = 0
    retained_areas: list[int] = []
    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area < minimum_area:
            continue
        if width >= height * 6 and height <= 32:
            continue
        semantic_seed[labels == label] = 255
        retained_components += 1
        retained_areas.append(area)
    retained_area = sum(retained_areas)
    dominant_component_fraction = (
        max(retained_areas, default=0) / retained_area
        if retained_area
        else 0.0
    )
    if retained_components > 0 and dominant_component_fraction < 0.93:
        raise ValueError(
            "White-maquette cleanup split the subject into large disconnected "
            "components: dominant_component_fraction="
            f"{dominant_component_fraction:.6f}, required=0.930000, "
            f"components={retained_components}"
        )
    contours, _ = cv2.findContours(
        semantic_seed,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    silhouette = np.zeros_like(semantic_seed)
    cv2.drawContours(silhouette, contours, -1, 255, thickness=cv2.FILLED)
    visible_fraction = float(np.count_nonzero(silhouette)) / float(
        silhouette.size
    )
    if retained_components < 1 or not 0.03 <= visible_fraction <= 0.75:
        raise ValueError(
            "White-maquette cleanup produced an unusable silhouette: "
            f"components={retained_components}, "
            f"visible_fraction={visible_fraction:.6f}"
        )
    return silhouette, {
        "policy": "neutral_material_floor_shadow_rejection_v2",
        "retained_components": retained_components,
        "dominant_component_fraction": round(
            dominant_component_fraction,
            6,
        ),
        "visible_fraction": round(visible_fraction, 6),
        "green_spill_pixels_recovered": int(
            np.count_nonzero((alpha >= 16) & recoverable_green_spill)
        ),
        "green_shadow_pixels_rejected": int(
            np.count_nonzero(
                (alpha >= 16)
                & green_shadow
                & ~recoverable_green_spill
            )
        ),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binary_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, columns = np.where(mask)
    if not len(columns):
        raise ValueError("eating_prop_split_unreliable: empty_mask")
    return (
        int(columns.min()),
        int(rows.min()),
        int(columns.max() + 1),
        int(rows.max() + 1),
    )


def fill_internal_holes(mask: np.ndarray) -> np.ndarray:
    binary = mask.astype(np.uint8)
    padded = np.pad(binary, 1, mode="constant", constant_values=0)
    inverse = (1 - padded).astype(np.uint8)
    flood = inverse.copy()
    flood_mask = np.zeros(
        (flood.shape[0] + 2, flood.shape[1] + 2),
        dtype=np.uint8,
    )
    cv2.floodFill(flood, flood_mask, (0, 0), 2)
    holes = (inverse != 0) & (flood != 2)
    return (padded.astype(bool) | holes)[1:-1, 1:-1]


def evaluate_eating_prop_temporal_stability(
    areas: np.ndarray,
    centers: np.ndarray,
) -> dict[str, Any]:
    """Accept tiny stationary bbox jitter without weakening moving-prop rejection."""
    numeric_areas = np.asarray(areas, dtype=np.float64)
    numeric_centers = np.asarray(centers, dtype=np.float64)
    if (
        numeric_areas.ndim != 1
        or numeric_areas.size < 2
        or numeric_centers.shape != (numeric_areas.size, 2)
        or not np.all(np.isfinite(numeric_areas))
        or not np.all(np.isfinite(numeric_centers))
        or np.any(numeric_areas <= 0)
    ):
        raise ValueError("eating_prop_split_unreliable: invalid_temporal_metrics")

    center_steps = np.linalg.norm(np.diff(numeric_centers, axis=0), axis=1)
    area_cv = float(numeric_areas.std() / max(numeric_areas.mean(), 1.0))
    center_step_max = float(center_steps.max())
    center_step_p95 = float(np.percentile(center_steps, 95))
    center_x_span = float(np.ptp(numeric_centers[:, 0]))
    center_y_span = float(np.ptp(numeric_centers[:, 1]))

    strict_step_passed = center_step_max <= 1.5
    # A fixed bowl can gain or lose one or two boundary pixels as the muzzle
    # occludes its rim.  The small-dog driver exhibits this rasterisation pattern:
    # individual bbox steps reach sqrt(5) pixels, while the centre remains inside
    # a 2.5 x 3 pixel box for the whole 81-frame cycle.  Accept that bounded,
    # stationary jitter only; a drifting or genuinely moving prop still fails.
    bounded_stationary_jitter_passed = bool(
        center_step_max <= 2.5
        and center_step_p95 <= 2.0
        and center_x_span <= 3.0
        and center_y_span <= 3.0
    )
    accepted = bool(
        area_cv <= 0.05
        and (strict_step_passed or bounded_stationary_jitter_passed)
    )
    return {
        "policy": "strict_step_or_bounded_stationary_jitter_v1",
        "accepted": accepted,
        "area_cv": area_cv,
        "center_step_max": center_step_max,
        "center_step_p95": center_step_p95,
        "center_x_span": center_x_span,
        "center_y_span": center_y_span,
        "strict_step_passed": strict_step_passed,
        "bounded_stationary_jitter_passed": bounded_stationary_jitter_passed,
        "thresholds": {
            "area_cv_max": 0.05,
            "strict_center_step_max": 1.5,
            "bounded_jitter_center_step_max": 2.5,
            "bounded_jitter_center_step_p95_max": 2.0,
            "bounded_jitter_center_x_span_max": 3.0,
            "bounded_jitter_center_y_span_max": 3.0,
        },
    }


def make_eating_prop_exclusion_masks(
    normalized_frames: list[Image.Image],
) -> tuple[list[Image.Image], dict[str, Any]]:
    """Separate a food-anchored static bowl from the animated cat."""
    if len(normalized_frames) < 9:
        raise ValueError("eating_prop_split_unreliable: fewer_than_9_frames")
    size = normalized_frames[0].size
    if any(frame.size != size for frame in normalized_frames):
        raise ValueError("eating_prop_split_unreliable: inconsistent_frame_size")

    rgba = np.stack(
        [
            np.asarray(frame.convert("RGBA"), dtype=np.uint8)
            for frame in normalized_frames
        ]
    )
    rgb = rgba[..., :3]
    foreground = rgba[..., 3] >= 16
    union = foreground.any(axis=0)
    left, top, right, bottom = binary_bbox(union)
    subject_width = right - left
    subject_height = bottom - top

    median = np.median(rgb.astype(np.int16), axis=0).astype(np.uint8)
    lower_quantile = np.percentile(rgb, 10, axis=0)
    upper_quantile = np.percentile(rgb, 90, axis=0)
    robust_span = (upper_quantile - lower_quantile).max(axis=-1)
    presence = foreground.mean(axis=0)

    # Locate the bowl by what makes an eating prop semantically distinct from
    # the cat: a wide, low band of warm food.  The former lower-right ROI
    # happened to work for the old side-mounted fixture, but on Main_case/eat.mp4
    # it selected the black-and-white cat's right tail instead of the central
    # white bowl.  Work from the temporal median so a dipping muzzle does not
    # permanently hide the food.
    row_indices, column_indices = np.indices(union.shape)
    lower_subject = (
        union
        & (presence >= 0.50)
        & (row_indices >= top + int(round(subject_height * 0.55)))
        & (row_indices < bottom)
    )
    median_red = median[..., 0].astype(np.int16)
    median_green = median[..., 1].astype(np.int16)
    median_blue = median[..., 2].astype(np.int16)
    minimum_anchor_area = max(48, int(round(subject_width * subject_height * 0.001)))
    minimum_anchor_width = max(24, int(round(subject_width * 0.10)))
    anchor_candidates: list[tuple[float, int]] = []
    warm_food = np.zeros_like(union, dtype=bool)
    labels = np.zeros_like(union, dtype=np.int32)
    anchor_red_threshold = 0
    # Prefer bright warm food/bowl pixels.  This separates a beige vessel from
    # an entirely brown driver cat.  The relaxed pass retains darker kibble for
    # white-bowl inputs where the surrounding cat is neutral.
    for red_threshold in (150, 110):
        warm_food = (
            lower_subject
            & (median_red >= red_threshold)
            & (median_red - median_green >= 18)
            & (median_green - median_blue >= 8)
        )
        warm_food = cv2.morphologyEx(
            warm_food.astype(np.uint8),
            cv2.MORPH_CLOSE,
            np.ones((5, 9), dtype=np.uint8),
            iterations=1,
        ).astype(bool)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            warm_food.astype(np.uint8),
            connectivity=8,
        )
        anchor_candidates = []
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            component_width = int(stats[label, cv2.CC_STAT_WIDTH])
            component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
            _center_x, center_y = centroids[label]
            aspect = component_width / float(max(component_height, 1))
            if (
                area < minimum_anchor_area
                or component_width < minimum_anchor_width
                or component_height > int(round(subject_height * 0.35))
                or aspect < 1.65
                or center_y < top + subject_height * 0.58
            ):
                continue
            relative_y = (center_y - top) / float(max(subject_height, 1))
            score = component_width * aspect * aspect * (1.0 + relative_y)
            anchor_candidates.append((score, label))
        if anchor_candidates:
            anchor_red_threshold = red_threshold
            break
    if not anchor_candidates:
        raise ValueError(
            "eating_prop_split_unreliable: "
            f"warm_food_anchor_candidates=0, "
            f"warm_food_pixels={int(np.count_nonzero(warm_food))}"
        )

    _score, anchor_label = max(anchor_candidates)
    food_anchor = labels == anchor_label
    anchor_left, anchor_top, anchor_right, anchor_bottom = binary_bbox(
        food_anchor
    )
    anchor_width = anchor_right - anchor_left

    # The food occupies the opening of the vessel.  Give the vessel search
    # enough lateral room for a flared white/gray bowl, while still excluding
    # the planted paws and the right-side tail.  Neutral pixels are admitted
    # only when temporally stable; the warm anchor itself remains valid for the
    # older beige-bowl drivers where food and vessel share one color.
    vessel_width = max(
        int(round(anchor_width * 1.68)),
        int(round(subject_width * 0.48)),
    )
    vessel_half_width = max(12, int(round(vessel_width / 2.0)))
    anchor_center_x = (anchor_left + anchor_right) // 2
    vessel_center_shift = int(round(anchor_width * 0.04))
    vessel_left = max(
        left,
        anchor_center_x - vessel_half_width + vessel_center_shift,
    )
    vessel_right = min(
        right,
        anchor_center_x + vessel_half_width + vessel_center_shift,
    )
    vessel_top = max(top, anchor_top - 2)
    vessel_bottom = bottom
    if (
        vessel_right - vessel_left < minimum_anchor_width
        or vessel_bottom <= vessel_top
    ):
        raise ValueError(
            "eating_prop_split_unreliable: invalid_food_anchored_vessel_roi"
        )
    vessel_roi = np.zeros_like(union, dtype=bool)
    vessel_roi[vessel_top:vessel_bottom, vessel_left:vessel_right] = True

    median_hsv = cv2.cvtColor(median, cv2.COLOR_RGB2HSV)
    neutral_bowl = (
        vessel_roi
        & union
        & (presence >= 0.88)
        & (robust_span <= 18.0)
        & (median_hsv[..., 1] <= 60)
        & (median_hsv[..., 2] >= 45)
    )
    bowl_support = neutral_bowl | food_anchor
    bowl_support = cv2.morphologyEx(
        bowl_support.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    ).astype(bool)
    support_count, support_labels, support_stats, _support_centroids = (
        cv2.connectedComponentsWithStats(
            bowl_support.astype(np.uint8),
            connectivity=8,
        )
    )
    food_touch = cv2.dilate(
        food_anchor.astype(np.uint8),
        np.ones((7, 7), dtype=np.uint8),
        iterations=1,
    ).astype(bool)
    support_candidates: list[tuple[int, int]] = []
    for label in range(1, support_count):
        component = support_labels == label
        if np.any(component & food_touch):
            support_candidates.append(
                (int(support_stats[label, cv2.CC_STAT_AREA]), label)
            )
    if not support_candidates:
        raise ValueError(
            "eating_prop_split_unreliable: food_anchor_has_no_vessel_support"
        )
    _support_area, selected_support_label = max(support_candidates)
    kept = (support_labels == selected_support_label) | food_anchor
    if int(np.count_nonzero(kept)) < 100:
        raise ValueError(
            "eating_prop_split_unreliable: "
            f"stable_core_pixels={int(np.count_nonzero(kept))}"
        )

    points_xy = np.column_stack(np.where(kept))[:, ::-1].astype(np.int32)
    hull = cv2.convexHull(points_xy)
    envelope = np.zeros_like(kept, dtype=np.uint8)
    cv2.fillConvexPoly(envelope, hull, 1)
    envelope = cv2.dilate(
        envelope,
        np.ones((5, 5), dtype=np.uint8),
        iterations=1,
    ).astype(bool)
    envelope &= vessel_roi

    exclusions: list[np.ndarray] = []
    for index in range(rgba.shape[0]):
        delta = np.abs(
            rgb[index].astype(np.int16) - median.astype(np.int16)
        ).max(axis=-1)
        prop = envelope & foreground[index] & (delta <= 18)
        prop = cv2.morphologyEx(
            prop.astype(np.uint8),
            cv2.MORPH_CLOSE,
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        )
        prop = cv2.dilate(
            prop,
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        ).astype(bool)
        prop &= envelope & foreground[index]

        exclusion = cv2.dilate(
            prop.astype(np.uint8),
            np.ones((5, 5), dtype=np.uint8),
            iterations=1,
        ).astype(bool)
        exclusion &= vessel_roi
        exclusions.append(exclusion)

    areas = np.asarray(
        [int(np.count_nonzero(mask)) for mask in exclusions],
        dtype=np.float64,
    )
    boxes = [binary_bbox(mask) for mask in exclusions]
    centers = np.asarray(
        [
            ((box[0] + box[2] - 1) / 2.0, (box[1] + box[3] - 1) / 2.0)
            for box in boxes
        ]
    )
    temporal_stability = evaluate_eating_prop_temporal_stability(areas, centers)
    area_cv = float(temporal_stability["area_cv"])
    center_step_max = float(temporal_stability["center_step_max"])
    if not temporal_stability["accepted"]:
        raise ValueError(
            "eating_prop_split_unreliable: "
            f"area_cv={area_cv:.4f}, center_step_max={center_step_max:.3f}, "
            "center_step_p95="
            f"{float(temporal_stability['center_step_p95']):.3f}, "
            "center_span="
            f"{float(temporal_stability['center_x_span']):.3f}x"
            f"{float(temporal_stability['center_y_span']):.3f}"
        )

    foreground_areas = np.asarray(
        [max(1, int(np.count_nonzero(mask))) for mask in foreground],
        dtype=np.float64,
    )
    prop_fractions = areas / foreground_areas
    if float(prop_fractions.min()) < 0.04 or float(prop_fractions.max()) > 0.20:
        raise ValueError(
            "eating_prop_split_unreliable: "
            f"prop_fraction={float(prop_fractions.min()):.4f}-"
            f"{float(prop_fractions.max()):.4f}"
        )

    exclusion_union = np.logical_or.reduce(exclusions)
    metadata = {
        "policy": "map_driving_bowl_as_red_scail_object",
        "method": "eating_temporal_food_anchor_bowl_v2",
        "subject_bbox": list((left, top, right, bottom)),
        "roi_bbox": [vessel_left, vessel_top, vessel_right, vessel_bottom],
        "warm_food_anchor_bbox": [
            anchor_left,
            anchor_top,
            anchor_right,
            anchor_bottom,
        ],
        "warm_food_anchor_pixels": int(np.count_nonzero(food_anchor)),
        "warm_food_anchor_candidates": len(anchor_candidates),
        "warm_food_anchor_red_threshold": anchor_red_threshold,
        "stable_core_pixels": int(np.count_nonzero(kept)),
        "stable_core_components": 1,
        "envelope_bbox": list(binary_bbox(envelope)),
        "prop_exclusion_bbox_union": list(binary_bbox(exclusion_union)),
        "prop_area_min": int(areas.min()),
        "prop_area_max": int(areas.max()),
        "prop_area_cv": area_cv,
        "prop_temporal_stability": temporal_stability,
        "prop_fraction_min": float(prop_fractions.min()),
        "prop_fraction_max": float(prop_fractions.max()),
        "prop_center_x_minmax": [
            float(centers[:, 0].min()),
            float(centers[:, 0].max()),
        ],
        "prop_center_y_minmax": [
            float(centers[:, 1].min()),
            float(centers[:, 1].max()),
        ],
        "prop_center_step_max": center_step_max,
        "prop_center_step_p95": float(temporal_stability["center_step_p95"]),
        "prop_center_x_span": float(temporal_stability["center_x_span"]),
        "prop_center_y_span": float(temporal_stability["center_y_span"]),
        "prop_bounded_stationary_jitter_accepted": bool(
            temporal_stability["bounded_stationary_jitter_passed"]
            and not temporal_stability["strict_step_passed"]
        ),
    }
    return (
        [
            Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
            for mask in exclusions
        ],
        metadata,
    )


def make_toileting_prop_exclusion_masks(
    normalized_frames: list[Image.Image],
) -> tuple[list[Image.Image], dict[str, Any]]:
    """Split the wide lower toilet tray from the animal identity.

    Lateral tray geometry, the static dark grid, and the lower tray base are
    marked as the red SCAIL prop.  Light animal paws remain blue, while detached
    lower floor/shadow fragments outside the stable tray bounds are discarded.
    """

    if not normalized_frames:
        raise ValueError("toileting_prop_split_unreliable: no_frames")
    size = normalized_frames[0].size
    if any(frame.size != size for frame in normalized_frames):
        raise ValueError(
            "toileting_prop_split_unreliable: inconsistent_frame_size"
        )
    alpha_masks = [
        np.asarray(frame.getchannel("A"), dtype=np.uint8) >= 16
        for frame in normalized_frames
    ]
    union = np.logical_or.reduce(alpha_masks)
    left, top, right, bottom = binary_bbox(union)
    subject_width = right - left
    subject_height = bottom - top
    if subject_width < 32 or subject_height < 48:
        raise ValueError("toileting_prop_split_unreliable: tiny_subject")

    row_widths = np.asarray(
        [
            int(columns.max() - columns.min() + 1)
            if (columns := np.where(union[row])[0]).size
            else 0
            for row in range(union.shape[0])
        ],
        dtype=np.int32,
    )
    body_y0 = top + int(round(subject_height * 0.16))
    body_y1 = top + int(round(subject_height * 0.58))
    body_width_samples = row_widths[body_y0:body_y1]
    body_width_samples = body_width_samples[body_width_samples > 0]
    if body_width_samples.size < 8:
        raise ValueError(
            "toileting_prop_split_unreliable: insufficient_body_rows"
        )
    body_width_p80 = float(np.percentile(body_width_samples, 80))
    tray_width_threshold = max(
        int(round(body_width_p80 * 1.16)),
        int(round(size[0] * 0.26)),
    )
    search_y0 = top + int(round(subject_height * 0.54))
    wide_rows = row_widths >= tray_width_threshold
    tray_top: int | None = None
    for row in range(search_y0, bottom):
        stop = min(bottom, row + 7)
        if int(np.count_nonzero(wide_rows[row:stop])) >= min(4, stop - row):
            tray_top = row
            break
    if tray_top is None:
        widest_row = int(search_y0 + np.argmax(row_widths[search_y0:bottom]))
        if row_widths[widest_row] < max(48, int(round(body_width_p80 * 1.08))):
            raise ValueError(
                "toileting_prop_split_unreliable: no_wide_lower_tray"
            )
        tray_top = widest_row

    lower_wide_rows = np.flatnonzero(
        wide_rows & (np.arange(union.shape[0]) >= tray_top)
    )
    if not lower_wide_rows.size:
        raise ValueError(
            "toileting_prop_split_unreliable: no_stable_tray_bottom"
        )
    tray_main_bottom = min(bottom, int(lower_wide_rows[-1]) + 3)
    tray_band_height = max(1, tray_main_bottom - tray_top)
    column_support = np.count_nonzero(
        union[tray_top:tray_main_bottom],
        axis=0,
    )
    supported_columns = column_support >= max(
        4,
        int(round(tray_band_height * 0.18)),
    )
    column_transitions = np.diff(
        np.pad(supported_columns.astype(np.int8), (1, 1))
    )
    column_starts = np.flatnonzero(column_transitions == 1)
    column_ends = np.flatnonzero(column_transitions == -1)
    column_runs = [
        (int(end - start), int(start), int(end))
        for start, end in zip(column_starts, column_ends, strict=True)
    ]
    if not column_runs:
        raise ValueError(
            "toileting_prop_split_unreliable: no_stable_tray_columns"
        )
    _tray_run_width, tray_main_left, tray_main_right = max(column_runs)
    tray_horizontal_margin = max(3, int(round(subject_width * 0.015)))
    tray_main_left = max(left, tray_main_left - tray_horizontal_margin)
    tray_main_right = min(right, tray_main_right + tray_horizontal_margin)

    row_indices, column_indices = np.indices(union.shape)
    valid_lower_geometry = (
        (row_indices < tray_top)
        | (
            (column_indices >= tray_main_left)
            & (column_indices < tray_main_right)
            & (row_indices < tray_main_bottom)
        )
    )
    rgb_frames = np.stack(
        [
            np.asarray(frame.convert("RGBA"), dtype=np.uint8)[..., :3]
            for frame in normalized_frames
        ],
        axis=0,
    )
    median_rgb = np.median(rgb_frames, axis=0).astype(np.uint8)
    median_luma = cv2.cvtColor(median_rgb, cv2.COLOR_RGB2GRAY)
    median_red = median_rgb[..., 0].astype(np.int16)
    median_green = median_rgb[..., 1].astype(np.int16)
    median_blue = median_rgb[..., 2].astype(np.int16)
    median_span = (
        median_rgb.max(axis=2).astype(np.int16)
        - median_rgb.min(axis=2).astype(np.int16)
    )
    foreground_span_p90 = float(np.percentile(median_span[union], 90))
    white_maquette_material = foreground_span_p90 <= 4.0
    grid_dark_luma_max = 112
    if white_maquette_material:
        grid_material_candidate = union & (
            median_luma <= grid_dark_luma_max
        )
        identity_colour_support = np.zeros_like(union)
        grid_material_policy = "white_maquette_deep_grid_slots_v1"
    else:
        # Generated identity references preserve warm coat chroma while their
        # toilet trays are neutral gray/white.  Keep the full warm dog, including
        # paws and tail, out of the prop mask even where it overlaps the grid.
        identity_colour_support = (
            union
            & (median_red - median_blue >= 15)
            & (median_red - median_green >= 6)
        )
        grid_material_candidate = (
            union
            & (median_span <= 12)
            & ~identity_colour_support
        )
        grid_material_policy = "neutral_tray_minus_warm_identity_v1"
    grid_dark_row_counts = np.asarray(
        [
            int(
                np.count_nonzero(
                    grid_material_candidate[
                        row,
                        tray_main_left:tray_main_right,
                    ]
                )
            )
            for row in range(union.shape[0])
        ],
        dtype=np.int32,
    )
    grid_dark_row_required = max(12, int(round(body_width_p80 * 0.08)))
    grid_rows = grid_dark_row_counts >= grid_dark_row_required
    tray_surface_top = tray_top
    tray_base_top = tray_main_bottom
    grid_seen = False
    for row in range(tray_top, tray_main_bottom):
        if int(np.count_nonzero(grid_rows[row : min(row + 5, tray_main_bottom)])) >= 3:
            if not grid_seen:
                tray_surface_top = row
            grid_seen = True
        if not grid_seen:
            continue
        stop = min(tray_main_bottom, row + 7)
        if stop - row >= 4 and int(np.count_nonzero(grid_rows[row:stop])) <= 1:
            tray_base_top = row
            break
    if tray_base_top <= tray_top:
        tray_base_top = tray_main_bottom

    core_y0 = top + int(round(subject_height * 0.30))
    exclusions: list[np.ndarray] = []
    prop_areas: list[int] = []
    prop_fractions: list[float] = []
    core_widths: list[int] = []
    grid_prop_areas: list[int] = []
    base_prop_areas: list[int] = []
    artifact_pixels_removed: list[int] = []
    for alpha, rgb in zip(alpha_masks, rgb_frames, strict=True):
        cleaned_alpha = alpha & valid_lower_geometry
        artifact_pixels_removed.append(
            int(np.count_nonzero(alpha & ~cleaned_alpha))
        )
        core_columns = np.any(
            cleaned_alpha[core_y0:tray_top, :],
            axis=0,
        ).astype(
            np.uint8
        )
        core_columns = cv2.dilate(
            core_columns.reshape(1, -1),
            np.ones((1, 9), dtype=np.uint8),
            iterations=1,
        ).reshape(-1).astype(bool)
        core_widths.append(int(np.count_nonzero(core_columns)))
        projected_identity = np.broadcast_to(core_columns, cleaned_alpha.shape)
        lateral_prop = (
            cleaned_alpha
            & (row_indices >= tray_top)
            & ~projected_identity
            & ~identity_colour_support
        )
        lateral_prop = cv2.morphologyEx(
            lateral_prop.astype(np.uint8),
            cv2.MORPH_CLOSE,
            np.ones((5, 9), dtype=np.uint8),
            iterations=1,
        ).astype(bool)
        frame_luma = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        grid_prop = (
            cleaned_alpha
            & (row_indices >= tray_surface_top)
            & (row_indices < tray_base_top)
            & grid_material_candidate
            & (
                (frame_luma <= grid_dark_luma_max + 18)
                if white_maquette_material
                else True
            )
        )
        grid_prop = cv2.dilate(
            grid_prop.astype(np.uint8),
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        ).astype(bool)
        grid_prop &= cleaned_alpha & (row_indices < tray_base_top)
        base_prop = (
            cleaned_alpha
            & (row_indices >= tray_base_top)
            & ~identity_colour_support
        )
        prop = lateral_prop | grid_prop | base_prop
        prop &= cleaned_alpha
        prop_area = int(np.count_nonzero(prop))
        foreground_area = int(np.count_nonzero(cleaned_alpha))
        if prop_area < 64:
            raise ValueError(
                "toileting_prop_split_unreliable: tray_prop_too_small"
            )
        prop_fraction = prop_area / float(max(1, foreground_area))
        if not 0.005 <= prop_fraction <= 0.50:
            raise ValueError(
                "toileting_prop_split_unreliable: invalid_prop_fraction="
                f"{prop_fraction:.6f}"
            )
        exclusions.append(prop)
        prop_areas.append(prop_area)
        prop_fractions.append(prop_fraction)
        grid_prop_areas.append(int(np.count_nonzero(grid_prop)))
        base_prop_areas.append(int(np.count_nonzero(base_prop)))

    return (
        [
            Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
            for mask in exclusions
        ],
        {
            "policy": "toileting_complete_tray_red_scail_prop_v3",
            "method": (
                "lateral_geometry_plus_material_grid_plus_lower_base_v3"
            ),
            "subject_bbox": [left, top, right, bottom],
            "body_width_p80": round(body_width_p80, 4),
            "tray_width_threshold": tray_width_threshold,
            "tray_top_y": tray_top,
            "tray_main_bbox": [
                tray_main_left,
                tray_top,
                tray_main_right,
                tray_main_bottom,
            ],
            "tray_base_top_y": tray_base_top,
            "tray_surface_top_y": tray_surface_top,
            "grid_dark_luma_max": grid_dark_luma_max,
            "grid_dark_row_required": grid_dark_row_required,
            "grid_material_policy": grid_material_policy,
            "foreground_material_span_p90": round(
                foreground_span_p90,
                4,
            ),
            "white_maquette_material": white_maquette_material,
            "identity_core_y0": core_y0,
            "identity_core_width_min": min(core_widths),
            "prop_area_min": min(prop_areas),
            "prop_area_max": max(prop_areas),
            "grid_prop_area_min": min(grid_prop_areas),
            "grid_prop_area_max": max(grid_prop_areas),
            "base_prop_area_min": min(base_prop_areas),
            "base_prop_area_max": max(base_prop_areas),
            "prop_fraction_min": min(prop_fractions),
            "prop_fraction_max": max(prop_fractions),
            "artifact_pixels_removed_min": min(artifact_pixels_removed),
            "artifact_pixels_removed_max": max(artifact_pixels_removed),
            "detached_floor_artifact_cleanup": True,
            "dark_tray_edge_normalization": True,
        },
    )


def normalize_toileting_prop_rgb(
    frame: Image.Image,
    prop_mask: Image.Image,
    prop_metadata: dict[str, Any] | None = None,
) -> Image.Image:
    """Remove floor artifacts and neutralize the complete tray prop."""

    rgba = np.asarray(frame.convert("RGBA"), dtype=np.uint8).copy()
    prop = np.asarray(prop_mask.convert("L"), dtype=np.uint8) >= 128
    if prop_metadata is not None:
        tray_main_bbox = prop_metadata.get("tray_main_bbox")
        if (
            isinstance(tray_main_bbox, list)
            and len(tray_main_bbox) == 4
        ):
            tray_left, tray_top, tray_right, tray_bottom = (
                int(value) for value in tray_main_bbox
            )
            rows, columns = np.indices(prop.shape)
            allowed = (
                (rows < tray_top)
                | (
                    (columns >= tray_left)
                    & (columns < tray_right)
                    & (rows < tray_bottom)
                )
            )
            artifact = (rgba[..., 3] >= 16) & ~allowed
            rgba[artifact] = 0
            prop &= allowed
    if not np.any(prop):
        return Image.fromarray(rgba, mode="RGBA")
    luma = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2GRAY).astype(np.float32)
    neutral = np.clip(175.0 + luma * 0.25, 188.0, 236.0).astype(np.uint8)
    rgba[prop, 0] = neutral[prop]
    rgba[prop, 1] = neutral[prop]
    rgba[prop, 2] = neutral[prop]
    return Image.fromarray(rgba, mode="RGBA")


def make_pose_and_mask_frames(
    sampled_frames: list[Image.Image],
    canvas_size: tuple[int, int],
    pose_background: tuple[int, int, int] = (255, 255, 255),
    separate_eating_prop: bool = False,
    separate_toileting_prop: bool = False,
    white_maquette_cleanup: bool = False,
) -> tuple[list[Image.Image], list[Image.Image], dict[str, Any]]:
    if separate_eating_prop and separate_toileting_prop:
        raise ValueError("Only one action prop split may be requested")
    cleanup_records: list[dict[str, Any]] = []
    if white_maquette_cleanup:
        cleaned_frames: list[Image.Image] = []
        for frame in sampled_frames:
            rgba = np.asarray(frame.convert("RGBA"), dtype=np.uint8).copy()
            silhouette, cleanup = white_maquette_semantic_silhouette(rgba)
            neutral_luma = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2GRAY)
            rgba[:, :, :3] = np.repeat(neutral_luma[:, :, None], 3, axis=2)
            rgba[:, :, 3] = silhouette
            rgba[silhouette == 0, :3] = 0
            cleaned_frames.append(Image.fromarray(rgba, mode="RGBA"))
            cleanup_records.append(cleanup)
        sampled_frames = cleaned_frames
    source_bbox = expand_bbox(
        union_alpha_bbox(sampled_frames),
        sampled_frames[0].size,
        margin_fraction=0.06,
    )
    normalized_frames = [
        contain_rgba(frame.crop(source_bbox), canvas_size, 0.88)
        for frame in sampled_frames
    ]
    prop_exclusions: list[Image.Image | None] = [None] * len(normalized_frames)
    prop_metadata: dict[str, Any] | None = None
    if separate_eating_prop:
        eating_exclusions, prop_metadata = make_eating_prop_exclusion_masks(
            normalized_frames
        )
        prop_exclusions = list(eating_exclusions)
    elif separate_toileting_prop:
        toileting_exclusions, prop_metadata = (
            make_toileting_prop_exclusion_masks(normalized_frames)
        )
        prop_exclusions = list(toileting_exclusions)
        normalized_frames = [
            normalize_toileting_prop_rgb(
                frame,
                exclusion,
                prop_metadata,
            )
            for frame, exclusion in zip(
                normalized_frames,
                toileting_exclusions,
                strict=True,
            )
        ]

    pose_frames: list[Image.Image] = []
    mask_frames: list[Image.Image] = []
    alpha_fractions: list[float] = []
    identity_fractions: list[float] = []
    prop_fractions: list[float] = []
    for normalized, prop_exclusion in zip(
        normalized_frames,
        prop_exclusions,
        strict=True,
    ):
        alpha = normalized.getchannel("A")
        hard_alpha = alpha.point(lambda value: 255 if value >= 16 else 0)
        hard_alpha = hard_alpha.filter(ImageFilter.MaxFilter(5))

        identity_alpha = hard_alpha
        if prop_exclusion is not None:
            prop_array = np.asarray(prop_exclusion, dtype=np.uint8) >= 128
            identity_array = (
                np.asarray(hard_alpha, dtype=np.uint8) >= 128
            ) & ~prop_array
            identity_alpha = Image.fromarray(
                identity_array.astype(np.uint8) * 255,
                mode="L",
            )
            prop_fractions.append(
                float(np.count_nonzero(prop_array))
                / float(canvas_size[0] * canvas_size[1])
            )

        pose_canvas = Image.new("RGB", canvas_size, pose_background)
        pose_canvas.paste(normalized.convert("RGB"), mask=alpha)
        pose_frames.append(pose_canvas)

        # SCAIL single-identity palette: replacement-video background is white,
        # while the foreground identity is pure blue.
        colored_mask = Image.new("RGB", canvas_size, (255, 255, 255))
        colored_mask.paste(
            Image.new("RGB", canvas_size, (0, 0, 255)),
            mask=identity_alpha,
        )
        if prop_exclusion is not None:
            colored_mask.paste(
                Image.new("RGB", canvas_size, (255, 0, 0)),
                mask=prop_exclusion,
            )
        mask_frames.append(colored_mask)
        alpha_fractions.append(
            float(np.count_nonzero(np.asarray(hard_alpha))) / float(canvas_size[0] * canvas_size[1])
        )
        identity_fractions.append(
            float(np.count_nonzero(np.asarray(identity_alpha)))
            / float(canvas_size[0] * canvas_size[1])
        )
    metadata = {
        "source_union_bbox": list(source_bbox),
        "subject_fraction_min": min(alpha_fractions),
        "subject_fraction_max": max(alpha_fractions),
        "subject_fraction_mean": sum(alpha_fractions) / len(alpha_fractions),
        "identity_fraction_min": min(identity_fractions),
        "identity_fraction_max": max(identity_fractions),
        "identity_fraction_mean": sum(identity_fractions) / len(identity_fractions),
        "scene_prop_separated": (
            separate_eating_prop or separate_toileting_prop
        ),
        "white_maquette_cleanup": white_maquette_cleanup,
    }
    if cleanup_records:
        metadata["white_maquette_cleanup_policy"] = cleanup_records[0]["policy"]
        metadata["white_maquette_visible_fraction_min"] = min(
            float(record["visible_fraction"]) for record in cleanup_records
        )
        metadata["white_maquette_visible_fraction_max"] = max(
            float(record["visible_fraction"]) for record in cleanup_records
        )
        metadata["white_maquette_source_chroma_preserved"] = False
    if separate_eating_prop or separate_toileting_prop:
        metadata.update(
            {
                "scene_prop_fraction_min": min(prop_fractions),
                "scene_prop_fraction_max": max(prop_fractions),
                "scene_prop_fraction_mean": sum(prop_fractions) / len(prop_fractions),
                "scene_prop_metadata": prop_metadata,
            }
        )
    return pose_frames, mask_frames, metadata


def make_full_eating_pose_and_mask_frames(
    sampled_frames: list[Image.Image],
    canvas_size: tuple[int, int],
) -> tuple[list[Image.Image], list[Image.Image], dict[str, Any]]:
    """Keep and audit the complete eating pose while separating its food bowl."""
    pose_frames, mask_frames, metadata = make_pose_and_mask_frames(
        sampled_frames,
        canvas_size,
        pose_background=(255, 255, 255),
        separate_eating_prop=True,
    )
    identity_bboxes: list[list[int]] = []
    identity_centers_y: list[float] = []
    identity_heights: list[int] = []
    identity_masks: list[np.ndarray] = []
    for mask in mask_frames:
        pixels = np.asarray(mask.convert("RGB"), dtype=np.uint8)
        identity = np.all(pixels == (0, 0, 255), axis=2)
        left, top, right, bottom = binary_bbox(identity)
        identity_masks.append(identity)
        identity_bboxes.append([left, top, right, bottom])
        identity_centers_y.append(round((top + bottom - 1) / 2.0, 4))
        identity_heights.append(bottom - top)

    first_identity = identity_masks[0]
    height, width = first_identity.shape
    first_left, first_top, first_right, first_bottom = identity_bboxes[0]
    lower_half_y = (first_top + first_bottom) // 2
    lower_quarter_y = (first_top + 3 * first_bottom) // 4
    canvas_area = height * width
    lower_half_area = (height - lower_half_y) * width
    lower_quarter_area = (height - lower_quarter_y) * width
    if (
        canvas_area <= 0
        or lower_half_area <= 0
        or lower_quarter_area <= 0
    ):
        raise ValueError(
            "eating_full_pose_motion_unreliable: invalid_audit_region"
        )

    overall_change_ratios: list[float] = []
    lower_half_change_ratios: list[float] = []
    lower_quarter_change_ratios: list[float] = []
    for identity in identity_masks:
        changed = np.logical_xor(first_identity, identity)
        overall_change_ratios.append(
            float(np.count_nonzero(changed)) / float(canvas_area)
        )
        lower_half_change_ratios.append(
            float(np.count_nonzero(changed[lower_half_y:, :]))
            / float(lower_half_area)
        )
        lower_quarter_change_ratios.append(
            float(np.count_nonzero(changed[lower_quarter_y:, :]))
            / float(lower_quarter_area)
        )

    overall_change_max = max(overall_change_ratios)
    lower_half_change_max = max(lower_half_change_ratios)
    lower_quarter_change_max = max(lower_quarter_change_ratios)
    motion_gate_thresholds = {
        "overall_change_ratio_min": (
            EATING_FULL_POSE_OVERALL_CHANGE_RATIO_MIN
        ),
        "lower_half_change_ratio_min": (
            EATING_FULL_POSE_LOWER_HALF_CHANGE_RATIO_MIN
        ),
        "lower_quarter_change_ratio_min": (
            EATING_FULL_POSE_LOWER_QUARTER_CHANGE_RATIO_MIN
        ),
    }
    failed_motion_checks = [
        name
        for name, value, minimum in (
            (
                "overall",
                overall_change_max,
                EATING_FULL_POSE_OVERALL_CHANGE_RATIO_MIN,
            ),
            (
                "lower_half",
                lower_half_change_max,
                EATING_FULL_POSE_LOWER_HALF_CHANGE_RATIO_MIN,
            ),
            (
                "lower_quarter",
                lower_quarter_change_max,
                EATING_FULL_POSE_LOWER_QUARTER_CHANGE_RATIO_MIN,
            ),
        )
        if value < minimum
    ]
    if failed_motion_checks:
        raise ValueError(
            "eating_full_pose_motion_unreliable: "
            f"failed={','.join(failed_motion_checks)}; "
            f"overall_change_ratio_max={overall_change_max:.6f}; "
            f"lower_half_change_ratio_max={lower_half_change_max:.6f}; "
            f"lower_quarter_change_ratio_max={lower_quarter_change_max:.6f}; "
            f"thresholds={motion_gate_thresholds}"
        )

    metadata.update(
        {
            "policy": "full_driver_pose_with_separate_bowl_v1",
            "driver_signal_only": False,
            "driver_rgb_preserved": True,
            "driver_silhouette_preserved": True,
            "pose_geometry_scope": "full_subject_silhouette_per_frame",
            "identity_bbox_by_frame": identity_bboxes,
            "identity_center_y_span": round(
                max(identity_centers_y) - min(identity_centers_y),
                4,
            ),
            "identity_height_span": int(
                max(identity_heights) - min(identity_heights)
            ),
            "identity_motion_audit": {
                "policy": "first_frame_blue_identity_mask_xor_v1",
                "reference_frame_index": 0,
                "first_frame_identity_bbox": [
                    first_left,
                    first_top,
                    first_right,
                    first_bottom,
                ],
                "overall_normalization_area_px": canvas_area,
                "lower_half_region_y_start": lower_half_y,
                "lower_half_normalization_area_px": lower_half_area,
                "lower_quarter_region_y_start": lower_quarter_y,
                "lower_quarter_normalization_area_px": lower_quarter_area,
                "overall_change_ratio_by_frame": [
                    round(value, 6) for value in overall_change_ratios
                ],
                "overall_change_ratio_max": round(
                    overall_change_max,
                    6,
                ),
                "overall_change_ratio_max_frame_index": int(
                    np.argmax(overall_change_ratios)
                ),
                "lower_half_change_ratio_by_frame": [
                    round(value, 6) for value in lower_half_change_ratios
                ],
                "lower_half_change_ratio_max": round(
                    lower_half_change_max,
                    6,
                ),
                "lower_half_change_ratio_max_frame_index": int(
                    np.argmax(lower_half_change_ratios)
                ),
                "lower_quarter_change_ratio_by_frame": [
                    round(value, 6) for value in lower_quarter_change_ratios
                ],
                "lower_quarter_change_ratio_max": round(
                    lower_quarter_change_max,
                    6,
                ),
                "lower_quarter_change_ratio_max_frame_index": int(
                    np.argmax(lower_quarter_change_ratios)
                ),
                "thresholds": motion_gate_thresholds,
                "passed": True,
            },
        }
    )
    return pose_frames, mask_frames, metadata


def make_dog_eating_canonical_full_driver_pose_frames(
    sampled_frames: list[Image.Image],
    canvas_size: tuple[int, int],
) -> tuple[list[Image.Image], list[Image.Image], dict[str, Any]]:
    """Bind approved identity to the complete digest-bound eating driver.

    This recovery deliberately does not inspect the canonical RGB image for a
    bowl. The reference mask remains identity-only, while the white-maquette
    video supplies every animated body and prop pixel through the normal full
    eating motion gate.
    """

    pose_frames, pose_masks, pose_metadata = (
        make_full_eating_pose_and_mask_frames(
            sampled_frames,
            canvas_size,
        )
    )
    pose_metadata.update(
        {
            "policy": DOG_EATING_CANONICAL_FULL_DRIVER_POSE_POLICY,
            "fallback_used": False,
            "canonical_identity_reference": True,
            "reference_identity_only": True,
            "reference_scene_prop_inference_disabled": True,
            "full_white_driver_geometry": True,
            "driver_input_used": True,
            "identity_authority": "approved_upload_canonical",
            "action_geometry_authority": (
                "digest_bound_white_maquette_video"
            ),
            "scene_prop_authority": "digest_bound_white_maquette_video",
        }
    )
    return pose_frames, pose_masks, pose_metadata


def make_dog_eating_reference_full_driver_recovery_pose_frames(
    sampled_frames: list[Image.Image],
    canvas_size: tuple[int, int],
    *,
    fallback_reason: str,
) -> tuple[list[Image.Image], list[Image.Image], dict[str, Any]]:
    """Recover an unusable reference-space nod with real driver motion.

    The selected Qwen eating keyframe remains SCAIL's identity reference. The
    digest-bound white-maquette clip supplies the complete animated body and
    bowl geometry, so a missing color-specific muzzle anchor can never turn the
    action into a repeated static first frame.
    """

    pose_frames, pose_masks, pose_metadata = (
        make_full_eating_pose_and_mask_frames(
            sampled_frames,
            canvas_size,
        )
    )
    pose_metadata.update(
        {
            "policy": DOG_EATING_REFERENCE_FULL_DRIVER_RECOVERY_POSE_POLICY,
            "fallback_used": True,
            "fallback_kind": "animated_full_driver_recovery",
            "fallback_reason": fallback_reason[:1000],
            "reference_identity_only": True,
            "reference_scene_prop_inference_disabled": True,
            "full_white_driver_geometry": True,
            "driver_input_used": True,
            "identity_authority": "selected_qwen_eating_reference",
            "action_geometry_authority": (
                "digest_bound_white_maquette_video"
            ),
            "scene_prop_authority": "digest_bound_white_maquette_video",
            "static_pose_fallback_forbidden": True,
        }
    )
    return pose_frames, pose_masks, pose_metadata


def _largest_binary_component(
    mask: np.ndarray,
    *,
    minimum_area: int,
    error_prefix: str = "eating_geometry_guide_unreliable",
) -> np.ndarray:
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )
    candidates = [
        (int(stats[label, cv2.CC_STAT_AREA]), label)
        for label in range(1, count)
        if int(stats[label, cv2.CC_STAT_AREA]) >= minimum_area
    ]
    if not candidates:
        raise ValueError(
            f"{error_prefix}: no_large_foreground_component"
        )
    _area, selected_label = max(candidates)
    return labels == selected_label


def _binary_external_perimeter(mask: np.ndarray) -> float:
    contours, _hierarchy = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        return 0.0
    return float(max(cv2.arcLength(contour, True) for contour in contours))


def _odd_scaled_kernel(
    canvas_size: tuple[int, int],
    *,
    value_at_512: int,
    minimum: int,
) -> int:
    value = max(
        minimum,
        int(round(min(canvas_size) * float(value_at_512) / 512.0)),
    )
    return value if value % 2 else value + 1


def make_grooming_geometry_guide(
    sampled_frames: list[Image.Image],
    canvas_size: tuple[int, int],
) -> tuple[Image.Image, dict[str, Any]]:
    """Build a texture-free Image 1 for the Main_case grooming action.

    Frame selection uses only alpha-silhouette motion.  The output uses a
    compressed, strongly blurred, quantized neutral anatomy field so the real
    TM head/shoulder/raised-paw structure remains readable without preserving
    source color or fine coat identity.  No artificial C-hook, floating muzzle
    oval, vertical flat part marker, distance gradient, exact source luma,
    whisker, or high-frequency coat texture is copied into the guide.
    """
    error_prefix = "grooming_geometry_guide_unreliable"
    if len(sampled_frames) < 9:
        raise ValueError(f"{error_prefix}: fewer_than_9_frames")
    if len({frame.size for frame in sampled_frames}) != 1:
        raise ValueError(f"{error_prefix}: inconsistent_frame_size")

    source_bbox = expand_bbox(
        union_alpha_bbox(sampled_frames),
        sampled_frames[0].size,
        margin_fraction=0.06,
    )
    normalized_frames = [
        contain_rgba(frame.crop(source_bbox), canvas_size, 0.88)
        for frame in sampled_frames
    ]
    hard_masks = [
        np.asarray(frame.getchannel("A"), dtype=np.uint8) >= 16
        for frame in normalized_frames
    ]
    canvas_area = int(canvas_size[0] * canvas_size[1])
    subject_areas = np.asarray(
        [int(np.count_nonzero(mask)) for mask in hard_masks],
        dtype=np.int64,
    )
    subject_fractions = subject_areas.astype(np.float64) / float(canvas_area)
    if (
        np.any(subject_fractions < 0.02)
        or np.any(subject_fractions > 0.80)
    ):
        raise ValueError(
            f"{error_prefix}: invalid_foreground_fraction="
            f"{float(subject_fractions.min()):.4f}.."
            f"{float(subject_fractions.max()):.4f}"
        )

    union = np.logical_or.reduce(hard_masks)
    left, top, right, bottom = binary_bbox(union)
    subject_width = right - left
    subject_height = bottom - top
    if subject_width < 32 or subject_height < 32:
        raise ValueError(f"{error_prefix}: subject_bbox_too_small")

    upper_body_bottom = min(
        canvas_size[1],
        top + int(round(subject_height * 0.72)),
    )
    upper_body_region = np.zeros_like(union, dtype=bool)
    upper_body_region[top:upper_body_bottom, left:right] = True
    first_mask = hard_masks[0]
    frame_motion: list[dict[str, Any]] = []
    for index, mask in enumerate(hard_masks):
        changed = np.logical_xor(mask, first_mask)
        overall_pixels = int(np.count_nonzero(changed))
        upper_pixels = int(
            np.count_nonzero(changed & upper_body_region)
        )
        frame_motion.append(
            {
                "index": index,
                "overall_xor_pixels": overall_pixels,
                "upper_body_xor_pixels": upper_pixels,
                "overall_xor_ratio": overall_pixels / float(canvas_area),
                "upper_body_xor_ratio": upper_pixels / float(canvas_area),
                "upper_body_motion_share": (
                    upper_pixels / float(overall_pixels)
                    if overall_pixels
                    else 0.0
                ),
            }
        )
    selected_motion = max(
        frame_motion,
        key=lambda item: (
            int(item["upper_body_xor_pixels"]),
            int(item["overall_xor_pixels"]),
            -int(item["index"]),
        ),
    )
    motion_thresholds = {
        "overall_xor_ratio_min": 0.014,
        "upper_body_xor_ratio_min": 0.012,
        "upper_body_motion_share_min": 0.55,
    }
    failed_motion_gates = [
        name
        for name, value in (
            (
                "overall",
                float(selected_motion["overall_xor_ratio"])
                < motion_thresholds["overall_xor_ratio_min"],
            ),
            (
                "upper_body",
                float(selected_motion["upper_body_xor_ratio"])
                < motion_thresholds["upper_body_xor_ratio_min"],
            ),
            (
                "upper_body_share",
                float(selected_motion["upper_body_motion_share"])
                < motion_thresholds["upper_body_motion_share_min"],
            ),
        )
        if value
    ]
    if failed_motion_gates:
        raise ValueError(
            f"{error_prefix}: insufficient_raised_paw_head_motion="
            + ",".join(failed_motion_gates)
        )

    selected_index = int(selected_motion["index"])
    raw_subject = hard_masks[selected_index]
    raw_perimeter = _binary_external_perimeter(raw_subject)
    outline_kernel_size = _odd_scaled_kernel(
        canvas_size,
        value_at_512=9,
        minimum=5,
    )
    outline_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (outline_kernel_size, outline_kernel_size),
    )
    selected_subject = cv2.morphologyEx(
        raw_subject.astype(np.uint8),
        cv2.MORPH_CLOSE,
        outline_kernel,
        iterations=1,
    )
    selected_subject = cv2.morphologyEx(
        selected_subject,
        cv2.MORPH_OPEN,
        outline_kernel,
        iterations=1,
    )
    blur_kernel_size = _odd_scaled_kernel(
        canvas_size,
        value_at_512=21,
        minimum=9,
    )
    blur_sigma = max(1.5, min(canvas_size) * 4.0 / 512.0)
    selected_subject = cv2.GaussianBlur(
        selected_subject * 255,
        (blur_kernel_size, blur_kernel_size),
        blur_sigma,
    )
    selected_subject = selected_subject >= 136
    selected_subject = fill_internal_holes(selected_subject)
    selected_subject = _largest_binary_component(
        selected_subject,
        minimum_area=max(500, int(canvas_area * 0.01)),
        error_prefix=error_prefix,
    )
    subject_pixels = int(np.count_nonzero(selected_subject))
    foreground_fraction = subject_pixels / float(canvas_area)
    if not 0.02 <= foreground_fraction <= 0.80:
        raise ValueError(
            f"{error_prefix}: cleaned_foreground_fraction="
            f"{foreground_fraction:.4f}"
        )
    cleaned_perimeter = _binary_external_perimeter(selected_subject)
    if raw_perimeter <= 0.0 or cleaned_perimeter > raw_perimeter * 1.01:
        raise ValueError(f"{error_prefix}: outline_smoothing_failed")

    # The fixed Main_case grooming driver raises the viewer-left forepaw.  Use
    # a large, heavily blurred luma field only to locate the muzzle/paw action
    # corridor.  Its original luma values are discarded.  Below, the corridor
    # is rebuilt as two flat neutral semantic regions: a rounded muzzle and one
    # continuous shoulder-to-forearm-to-rounded-paw shape.  Keeping the regions
    # distinct prevents their source contact patch from becoming a C-shaped
    # object in Qwen while retaining the real TM mouth-contact relationship.
    contact_x0 = left + int(round(subject_width * 0.10))
    contact_x1 = left + int(round(subject_width * 0.58))
    contact_y0 = top + int(round(subject_height * 0.36))
    contact_y1 = top + int(round(subject_height * 0.74))
    contact_region = np.zeros_like(union, dtype=bool)
    contact_region[
        max(0, contact_y0) : min(canvas_size[1], contact_y1),
        max(0, contact_x0) : min(canvas_size[0], contact_x1),
    ] = True
    contact_region &= selected_subject
    if int(np.count_nonzero(contact_region)) < 500:
        raise ValueError(f"{error_prefix}: contact_region_too_small")

    selected_rgb = np.asarray(
        normalized_frames[selected_index].convert("RGB"),
        dtype=np.uint8,
    )
    selected_gray = cv2.cvtColor(selected_rgb, cv2.COLOR_RGB2GRAY).astype(
        np.float32
    )
    raw_support = raw_subject.astype(np.float32)
    structure_sigma = max(2.0, min(canvas_size) * 6.0 / 512.0)
    weighted_luma = cv2.GaussianBlur(
        selected_gray * raw_support,
        (0, 0),
        sigmaX=structure_sigma,
        sigmaY=structure_sigma,
    )
    blurred_support = cv2.GaussianBlur(
        raw_support,
        (0, 0),
        sigmaX=structure_sigma,
        sigmaY=structure_sigma,
    )
    low_frequency_luma = weighted_luma / np.maximum(
        blurred_support,
        0.01,
    )
    contact_luma = low_frequency_luma[contact_region]
    luma_p10, luma_threshold, luma_p90 = np.percentile(
        contact_luma,
        (10, 65, 90),
    )
    luma_contrast = float(luma_p90 - luma_p10)
    if luma_contrast < 20.0:
        raise ValueError(
            f"{error_prefix}: low_frequency_contact_contrast="
            f"{luma_contrast:.3f}"
        )

    contact_component = (
        (low_frequency_luma >= float(luma_threshold))
        & contact_region
    ).astype(np.uint8)
    contact_close_size = _odd_scaled_kernel(
        canvas_size,
        value_at_512=9,
        minimum=5,
    )
    contact_open_size = _odd_scaled_kernel(
        canvas_size,
        value_at_512=5,
        minimum=3,
    )
    contact_component = cv2.morphologyEx(
        contact_component,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (contact_close_size, contact_close_size),
        ),
        iterations=1,
    )
    contact_component = cv2.morphologyEx(
        contact_component,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (contact_open_size, contact_open_size),
        ),
        iterations=1,
    )
    contact_component = fill_internal_holes(contact_component)
    contact_component = _largest_binary_component(
        contact_component,
        minimum_area=max(180, int(subject_pixels * 0.004)),
        error_prefix=error_prefix,
    )
    contact_component &= selected_subject
    contact_component = _largest_binary_component(
        contact_component,
        minimum_area=max(180, int(subject_pixels * 0.004)),
        error_prefix=error_prefix,
    )
    contact_pixels = int(np.count_nonzero(contact_component))
    contact_left, contact_top, contact_right, contact_bottom = binary_bbox(
        contact_component
    )
    contact_width_ratio = (
        contact_right - contact_left
    ) / float(subject_width)
    contact_height_ratio = (
        contact_bottom - contact_top
    ) / float(subject_height)
    contact_fraction = contact_pixels / float(subject_pixels)
    split_y = top + int(round(subject_height * 0.52))
    upper_muzzle_support = int(
        np.count_nonzero(
            contact_component[: min(canvas_size[1], split_y), :]
        )
    )
    lower_paw_support = int(
        np.count_nonzero(
            contact_component[max(0, split_y) :, :]
        )
    )
    minimum_contact_support = max(128, int(contact_pixels * 0.15))
    contact_failures: list[str] = []
    if not 0.025 <= contact_fraction <= 0.22:
        contact_failures.append("area")
    if not 0.12 <= contact_width_ratio <= 0.50:
        contact_failures.append("width")
    if not 0.22 <= contact_height_ratio <= 0.58:
        contact_failures.append("height")
    if contact_top > top + int(round(subject_height * 0.50)):
        contact_failures.append("muzzle_height")
    if contact_bottom < top + int(round(subject_height * 0.62)):
        contact_failures.append("paw_height")
    if upper_muzzle_support < minimum_contact_support:
        contact_failures.append("muzzle_support")
    if lower_paw_support < minimum_contact_support:
        contact_failures.append("paw_support")
    if contact_failures:
        raise ValueError(
            f"{error_prefix}: invalid_muzzle_raised_paw_relation="
            + ",".join(contact_failures)
        )

    rows = np.indices(selected_subject.shape)[0]
    semantic_muzzle_height = max(
        9,
        int(round(subject_height * 0.10)),
    )
    semantic_muzzle_y1 = min(
        contact_bottom,
        contact_top + semantic_muzzle_height,
    )
    muzzle_seed = (
        contact_component.astype(bool)
        & (rows >= contact_top)
        & (rows < semantic_muzzle_y1)
    )
    muzzle_seed_pixels = int(np.count_nonzero(muzzle_seed))
    minimum_semantic_pixels = max(96, int(subject_pixels * 0.002))
    if muzzle_seed_pixels < minimum_semantic_pixels:
        raise ValueError(f"{error_prefix}: muzzle_semantic_seed_too_small")

    muzzle_seed_left, muzzle_seed_top, muzzle_seed_right, muzzle_seed_bottom = (
        binary_bbox(muzzle_seed)
    )
    muzzle_region = np.zeros_like(selected_subject, dtype=np.uint8)
    muzzle_center = (
        (muzzle_seed_left + muzzle_seed_right - 1) // 2,
        (muzzle_seed_top + muzzle_seed_bottom - 1) // 2,
    )
    muzzle_axes = (
        max(5, (muzzle_seed_right - muzzle_seed_left) // 2),
        max(4, (muzzle_seed_bottom - muzzle_seed_top) // 2),
    )
    cv2.ellipse(
        muzzle_region,
        muzzle_center,
        muzzle_axes,
        0.0,
        0.0,
        360.0,
        1,
        thickness=-1,
    )
    muzzle_region = muzzle_region.astype(bool) & selected_subject
    muzzle_region = _largest_binary_component(
        muzzle_region,
        minimum_area=minimum_semantic_pixels,
        error_prefix=error_prefix,
    )

    # Estimate the raised foreleg axis from only the lower part of the detected
    # action corridor.  This excludes the horizontal muzzle branch that formed
    # the old hook.  A deterministic capsule then connects that shoulder anchor
    # to one rounded paw at the mouth.
    lower_axis_seed = (
        contact_component.astype(bool)
        & (rows >= max(contact_top, semantic_muzzle_y1 - 2))
    )
    lower_axis_y0 = max(
        contact_top,
        contact_bottom
        - max(
            7,
            int(round((contact_bottom - contact_top) * 0.22)),
        ),
    )
    shoulder_seed = lower_axis_seed & (rows >= lower_axis_y0)
    if int(np.count_nonzero(lower_axis_seed)) < minimum_semantic_pixels:
        raise ValueError(f"{error_prefix}: raised_foreleg_semantic_seed_too_small")
    if int(np.count_nonzero(shoulder_seed)) < max(24, minimum_semantic_pixels // 3):
        raise ValueError(f"{error_prefix}: raised_foreleg_shoulder_seed_too_small")

    _, lower_x = np.nonzero(lower_axis_seed)
    shoulder_y_values, shoulder_x_values = np.nonzero(shoulder_seed)
    paw_axis_x = int(round(float(np.median(lower_x))))
    shoulder_x = int(round(float(np.median(shoulder_x_values))))
    shoulder_y = min(
        contact_bottom - 1,
        int(round(float(np.percentile(shoulder_y_values, 90)))),
    )
    foreleg_width = max(13, int(round(subject_width * 0.115)))
    foreleg_radius = max(6, foreleg_width // 2)
    paw_radius_x = max(foreleg_radius, int(round(subject_width * 0.070)))
    paw_radius_y = max(7, int(round(subject_height * 0.050)))
    paw_center_y = min(
        contact_bottom - paw_radius_y - 1,
        max(
            muzzle_seed_bottom + paw_radius_y - 2,
            semantic_muzzle_y1 + paw_radius_y - 3,
        ),
    )
    paw_center_x = int(
        np.clip(
            paw_axis_x,
            muzzle_seed_left,
            max(muzzle_seed_left, muzzle_seed_right - 1),
        )
    )
    elbow_x = int(round((paw_center_x * 2.0 + shoulder_x) / 3.0))
    elbow_y = int(round((paw_center_y + shoulder_y) / 2.0))

    raised_foreleg_region = np.zeros_like(selected_subject, dtype=np.uint8)
    cv2.line(
        raised_foreleg_region,
        (shoulder_x, shoulder_y),
        (elbow_x, elbow_y),
        1,
        thickness=foreleg_width,
        lineType=cv2.LINE_8,
    )
    cv2.line(
        raised_foreleg_region,
        (elbow_x, elbow_y),
        (paw_center_x, paw_center_y),
        1,
        thickness=foreleg_width,
        lineType=cv2.LINE_8,
    )
    cv2.ellipse(
        raised_foreleg_region,
        (paw_center_x, paw_center_y),
        (paw_radius_x, paw_radius_y),
        0.0,
        0.0,
        360.0,
        1,
        thickness=-1,
    )
    raised_foreleg_region = (
        raised_foreleg_region.astype(bool)
        & selected_subject
        & ~muzzle_region
    )
    raised_foreleg_region = _largest_binary_component(
        raised_foreleg_region,
        minimum_area=minimum_semantic_pixels,
        error_prefix=error_prefix,
    )

    muzzle_pixels = int(np.count_nonzero(muzzle_region))
    raised_foreleg_pixels = int(np.count_nonzero(raised_foreleg_region))
    muzzle_bbox = binary_bbox(muzzle_region)
    raised_foreleg_bbox = binary_bbox(raised_foreleg_region)
    semantic_union = muzzle_region | raised_foreleg_region
    semantic_union_pixels = int(np.count_nonzero(semantic_union))
    semantic_contact_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3),
    )
    raised_foreleg_dilated = cv2.dilate(
        raised_foreleg_region.astype(np.uint8),
        semantic_contact_kernel,
        iterations=1,
    ).astype(bool)
    semantic_contact_pixels = int(
        np.count_nonzero(raised_foreleg_dilated & muzzle_region)
    )
    minimum_semantic_contact = max(3, foreleg_width // 6)
    semantic_failures: list[str] = []
    if muzzle_pixels < minimum_semantic_pixels:
        semantic_failures.append("muzzle_area")
    if raised_foreleg_pixels < minimum_semantic_pixels:
        semantic_failures.append("raised_foreleg_area")
    if semantic_contact_pixels < minimum_semantic_contact:
        semantic_failures.append("mouth_contact")
    if raised_foreleg_bbox[1] > muzzle_bbox[3]:
        semantic_failures.append("paw_height")
    if raised_foreleg_bbox[3] < contact_bottom - foreleg_radius:
        semantic_failures.append("shoulder_reach")
    if np.any(muzzle_region & raised_foreleg_region):
        semantic_failures.append("regions_overlap")
    if semantic_failures:
        raise ValueError(
            f"{error_prefix}: invalid_split_muzzle_raised_foreleg_semantics="
            + ",".join(semantic_failures)
        )

    # Compress the already heavily blurred source luminance into a narrow,
    # quantized neutral range.  This retains only low-frequency anatomy (the
    # actual head, shoulder and raised paw) and avoids the artificial separated
    # oval/capsule shapes that Qwen interpreted as a scarf, tie or brush.
    anatomy_values = low_frequency_luma[selected_subject]
    anatomy_p05, anatomy_p95 = np.percentile(anatomy_values, (5, 95))
    anatomy_span = float(anatomy_p95 - anatomy_p05)
    if anatomy_span < 16.0:
        raise ValueError(f"{error_prefix}: low_frequency_anatomy_contrast")
    normalized_anatomy = np.clip(
        (low_frequency_luma - float(anatomy_p05)) / anatomy_span,
        0.0,
        1.0,
    )
    anatomy_gray_min = 120
    anatomy_gray_max = 168
    anatomy_quantum = 8
    anatomy_gray = (
        anatomy_gray_min
        + normalized_anatomy * (anatomy_gray_max - anatomy_gray_min)
    )
    anatomy_gray = (
        np.round(anatomy_gray / anatomy_quantum) * anatomy_quantum
    ).astype(np.uint8)
    anatomy_gray = np.clip(
        anatomy_gray,
        anatomy_gray_min,
        anatomy_gray_max,
    ).astype(np.uint8)
    guide = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
    guide[:] = (0, 255, 0)
    guide[selected_subject] = np.repeat(
        anatomy_gray[selected_subject][:, None],
        3,
        axis=1,
    )

    green_background = np.all(guide == (0, 255, 0), axis=2)
    if not np.all(green_background[~selected_subject]):
        raise ValueError(f"{error_prefix}: background_not_pure_green")
    foreground_rgb = guide[selected_subject]
    if (
        np.any(foreground_rgb[:, 0] != foreground_rgb[:, 1])
        or np.any(foreground_rgb[:, 1] != foreground_rgb[:, 2])
    ):
        raise ValueError(f"{error_prefix}: non_neutral_foreground")

    metadata = {
        "policy": "driving_grooming_geometry_image1_v3",
        "method": (
            "upper_body_motion_keyframe_with_compressed_anatomy_field_v3"
        ),
        "selected_frame_index": selected_index,
        "source_union_bbox": list(source_bbox),
        "normalized_subject_union_bbox": [left, top, right, bottom],
        "subject_bbox": list(binary_bbox(selected_subject)),
        "subject_pixels": subject_pixels,
        "foreground_fraction": round(foreground_fraction, 6),
        "action_motion_audit": {
            "reference_frame_index": 0,
            "upper_body_region_bbox": [
                left,
                top,
                right,
                upper_body_bottom,
            ],
            "overall_xor_pixels": int(
                selected_motion["overall_xor_pixels"]
            ),
            "upper_body_xor_pixels": int(
                selected_motion["upper_body_xor_pixels"]
            ),
            "overall_xor_ratio": round(
                float(selected_motion["overall_xor_ratio"]),
                6,
            ),
            "upper_body_xor_ratio": round(
                float(selected_motion["upper_body_xor_ratio"]),
                6,
            ),
            "upper_body_motion_share": round(
                float(selected_motion["upper_body_motion_share"]),
                6,
            ),
            "thresholds": motion_thresholds,
            "passed": True,
        },
        "raised_paw_muzzle_relation": {
            "policy": (
                "source_derived_muzzle_foreleg_geometry_audit_v3"
            ),
            "search_region_bbox": [
                contact_x0,
                contact_y0,
                contact_x1,
                contact_y1,
            ],
            "component_bbox": [
                contact_left,
                contact_top,
                contact_right,
                contact_bottom,
            ],
            "component_pixels": semantic_union_pixels,
            "component_fraction_of_subject": round(
                semantic_union_pixels / float(subject_pixels),
                6,
            ),
            "component_width_ratio": round(contact_width_ratio, 6),
            "component_height_ratio": round(contact_height_ratio, 6),
            "upper_muzzle_support_pixels": muzzle_pixels,
            "lower_raised_paw_support_pixels": raised_foreleg_pixels,
            "minimum_support_pixels": minimum_contact_support,
            "muzzle_region_bbox": list(muzzle_bbox),
            "muzzle_region_pixels": muzzle_pixels,
            "raised_foreleg_region_bbox": list(raised_foreleg_bbox),
            "raised_foreleg_region_pixels": raised_foreleg_pixels,
            "semantic_contact_pixels": semantic_contact_pixels,
            "minimum_semantic_contact_pixels": minimum_semantic_contact,
            "regions_disjoint": True,
            "shoulder_anchor": [shoulder_x, shoulder_y],
            "paw_center": [paw_center_x, paw_center_y],
            "output_encoding": {
                "method": "compressed_blurred_quantized_neutral_anatomy",
                "gray_min": anatomy_gray_min,
                "gray_max": anatomy_gray_max,
                "quantum": anatomy_quantum,
                "source_luma_p05": round(float(anatomy_p05), 3),
                "source_luma_p95": round(float(anatomy_p95), 3),
            },
            "c_hook_component_preserved": False,
            "black_or_tool_semantics_preserved": False,
            "low_frequency_luma_p10": round(float(luma_p10), 3),
            "low_frequency_luma_threshold_p65": round(
                float(luma_threshold),
                3,
            ),
            "low_frequency_luma_p90": round(float(luma_p90), 3),
            "low_frequency_luma_contrast": round(luma_contrast, 3),
            "gaussian_sigma": round(structure_sigma, 3),
            "source_luma_values_preserved": False,
            "passed": True,
        },
        "outline_smoothing": {
            "morphology_kernel": outline_kernel_size,
            "gaussian_kernel": blur_kernel_size,
            "gaussian_sigma": round(blur_sigma, 3),
            "raw_perimeter": round(raw_perimeter, 3),
            "cleaned_perimeter": round(cleaned_perimeter, 3),
            "perimeter_ratio": round(
                cleaned_perimeter / max(raw_perimeter, 1.0),
                6,
            ),
            "high_frequency_outline_preserved": False,
        },
        "palette": {
            "background": [0, 255, 0],
            "subject_gray_range": [anatomy_gray_min, anatomy_gray_max],
            "quantum": anatomy_quantum,
            "source_rgb_preserved": False,
            "source_chroma_preserved": False,
            "source_luma_values_preserved": False,
            "low_frequency_anatomy_preserved": True,
            "source_marking_contrast_preserved": False,
            "high_frequency_texture_preserved": False,
            "distance_field_preserved": False,
            "artificial_part_markers_preserved": False,
            "black_foreground_preserved": False,
        },
    }
    return Image.fromarray(guide, mode="RGB"), metadata


def make_eating_geometry_guide(
    sampled_frames: list[Image.Image],
    canvas_size: tuple[int, int],
) -> tuple[Image.Image, dict[str, Any]]:
    """Build an identity-free Image 2 that retains only eating action geometry.

    The guide deliberately contains no source RGB or luma pixels. A flat
    neutral-gray field describes only the cleaned eating silhouette, while fixed
    semantic beige/brown fields describe the bowl and kibble. Every remaining
    pixel is pure green. Keeping the subject single-tone prevents low-frequency
    white-maquette shading from being reinterpreted as coat markings or facial
    identity by the image editor.
    """
    if len(sampled_frames) < 9:
        raise ValueError(
            "eating_geometry_guide_unreliable: fewer_than_9_frames"
        )
    if len({frame.size for frame in sampled_frames}) != 1:
        raise ValueError(
            "eating_geometry_guide_unreliable: inconsistent_frame_size"
        )

    source_bbox = expand_bbox(
        union_alpha_bbox(sampled_frames),
        sampled_frames[0].size,
        margin_fraction=0.06,
    )
    normalized_frames = [
        contain_rgba(frame.crop(source_bbox), canvas_size, 0.88)
        for frame in sampled_frames
    ]
    prop_exclusions, prop_metadata = make_eating_prop_exclusion_masks(
        normalized_frames
    )
    prop_union = np.logical_or.reduce(
        [np.asarray(mask, dtype=np.uint8) >= 128 for mask in prop_exclusions]
    )
    prop_union = cv2.morphologyEx(
        prop_union.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((5, 5), dtype=np.uint8),
        iterations=2,
    ).astype(bool)
    prop_union = fill_internal_holes(prop_union)
    prop_union = _largest_binary_component(prop_union, minimum_area=100)
    bowl_left, bowl_top, bowl_right, bowl_bottom = binary_bbox(prop_union)
    bowl_width = bowl_right - bowl_left
    bowl_height = bowl_bottom - bowl_top
    if bowl_width < 24 or bowl_height < 8:
        raise ValueError(
            "eating_geometry_guide_unreliable: bowl_component_too_small"
        )

    # Paws normally sit beside the bowl.  Restrict the contact measurement to
    # the central opening so a planted paw cannot masquerade as a low muzzle.
    contact_x0 = bowl_left + int(round(bowl_width * 0.20))
    contact_x1 = bowl_right - int(round(bowl_width * 0.20))
    contact_y0 = max(0, bowl_top - max(48, int(round(bowl_height * 1.5))))
    contact_y1 = min(canvas_size[1], bowl_top + 3)
    head_x0 = max(0, bowl_left - int(round(bowl_width * 0.25)))
    head_x1 = min(canvas_size[0], bowl_right + int(round(bowl_width * 0.25)))
    if contact_x1 <= contact_x0 or contact_y1 <= contact_y0:
        raise ValueError(
            "eating_geometry_guide_unreliable: invalid_muzzle_contact_region"
        )
    if head_x1 <= head_x0:
        raise ValueError(
            "eating_geometry_guide_unreliable: invalid_head_lowering_region"
        )

    frame_records: list[dict[str, Any]] = []
    subject_masks: list[np.ndarray] = []
    for index, (frame, prop_exclusion) in enumerate(
        zip(normalized_frames, prop_exclusions, strict=True)
    ):
        hard_alpha = np.asarray(frame.getchannel("A"), dtype=np.uint8) >= 16
        frame_prop = np.asarray(prop_exclusion, dtype=np.uint8) >= 128
        subject = hard_alpha & ~frame_prop
        subject_masks.append(subject)
        contact_region = subject[
            contact_y0:contact_y1,
            contact_x0:contact_x1,
        ]
        rows, _columns = np.where(contact_region)
        if rows.size:
            lowest_subject_y = contact_y0 + int(rows.max())
            gap = max(0, bowl_top - lowest_subject_y - 1)
            contact_pixels = int(
                np.count_nonzero(
                    contact_region[
                        max(0, bowl_top - 5 - contact_y0) :,
                        :,
                    ]
                )
            )
        else:
            lowest_subject_y = None
            gap = canvas_size[1]
            contact_pixels = 0
        head_rows, _head_columns = np.where(
            subject[:bowl_top, head_x0:head_x1]
        )
        if head_rows.size:
            head_y_p10 = float(np.percentile(head_rows, 10))
            head_support_pixels = int(head_rows.size)
        else:
            head_y_p10 = float("-inf")
            head_support_pixels = 0
        frame_records.append(
            {
                "index": index,
                "muzzle_bowl_gap_px": gap,
                "contact_pixels": contact_pixels,
                "lowest_subject_y": lowest_subject_y,
                "head_y_p10": head_y_p10,
                "head_support_pixels": head_support_pixels,
            }
        )

    contact_candidates = [
        item
        for item in frame_records
        if int(item["muzzle_bowl_gap_px"]) <= 4
        and int(item["contact_pixels"]) >= 4
        and np.isfinite(float(item["head_y_p10"]))
    ]
    if not contact_candidates:
        raise ValueError(
            "eating_geometry_guide_unreliable: no_head_lowered_bowl_contact_frame"
        )
    head_y_values = [
        float(item["head_y_p10"])
        for item in frame_records
        if np.isfinite(float(item["head_y_p10"]))
    ]
    head_lowering_span = max(head_y_values) - min(head_y_values)
    head_lowering_span_min = max(6.0, canvas_size[1] * 0.012)
    if head_lowering_span < head_lowering_span_min:
        raise ValueError(
            "eating_geometry_guide_unreliable: insufficient_head_lowering="
            f"{head_lowering_span:.3f}"
        )
    # The bowl and planted paws touch in every Main_case frame, so contact
    # pixels alone cannot identify eating and previously selected a head-up
    # frame. The lower 10th-percentile of the central upper silhouette tracks
    # the ears/head moving down while ignoring the static paws and bowl.
    selected = max(
        contact_candidates,
        key=lambda item: (
            float(item["head_y_p10"]),
            int(item["contact_pixels"]),
            -int(item["index"]),
        ),
    )

    selected_subject = subject_masks[int(selected["index"])] & ~prop_union
    raw_subject_perimeter = _binary_external_perimeter(selected_subject)
    outline_kernel_size = _odd_scaled_kernel(
        canvas_size,
        value_at_512=9,
        minimum=5,
    )
    outline_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (outline_kernel_size, outline_kernel_size),
    )
    selected_subject = cv2.morphologyEx(
        selected_subject.astype(np.uint8),
        cv2.MORPH_CLOSE,
        outline_kernel,
        iterations=1,
    )
    selected_subject = cv2.morphologyEx(
        selected_subject,
        cv2.MORPH_OPEN,
        outline_kernel,
        iterations=1,
    )
    # Smooth away whiskers, narrow tag connections, and high-frequency coat
    # spikes while retaining ears, head, torso, paws, and the low eating pose.
    blur_kernel_size = _odd_scaled_kernel(
        canvas_size,
        value_at_512=21,
        minimum=9,
    )
    blur_sigma = max(1.5, min(canvas_size) * 4.0 / 512.0)
    selected_subject = cv2.GaussianBlur(
        selected_subject * 255,
        (blur_kernel_size, blur_kernel_size),
        blur_sigma,
    )
    selected_subject = selected_subject >= 136
    selected_subject = fill_internal_holes(selected_subject)
    selected_subject = _largest_binary_component(
        selected_subject,
        minimum_area=max(500, int(canvas_size[0] * canvas_size[1] * 0.01)),
    )
    # Keep the true source-anchored muzzle immediately adjacent to the food.
    # Removing a dilated prop mask created an artificial air gap; only the actual
    # prop pixels are excluded here. The subject remains a single neutral field.
    selected_subject &= ~prop_union
    if int(np.count_nonzero(selected_subject)) < 500:
        raise ValueError(
            "eating_geometry_guide_unreliable: cleaned_subject_too_small"
        )
    cleaned_subject_perimeter = _binary_external_perimeter(selected_subject)
    if (
        raw_subject_perimeter <= 0.0
        or cleaned_subject_perimeter > raw_subject_perimeter * 1.01
    ):
        raise ValueError(
            "eating_geometry_guide_unreliable: outline_smoothing_failed"
        )

    # The silhouette, selected low-head frame and source-anchored food contact
    # carry the complete geometry contract. Never preserve even blurred source
    # luma here: on the white maquette it forms broad curved bands which Qwen can
    # faithfully but incorrectly render as tabby stripes.
    subject_semantic_rgb = np.array([132, 132, 132], dtype=np.uint8)

    median_rgb = np.median(
        np.stack(
            [
                np.asarray(frame.convert("RGB"), dtype=np.float32)
                for frame in normalized_frames
            ]
        ),
        axis=0,
    ).astype(np.uint8)
    median_luma = cv2.cvtColor(median_rgb, cv2.COLOR_RGB2GRAY).astype(
        np.float32
    )
    bowl_support = prop_union.astype(np.float32)
    bowl_structure_sigma = max(1.0, min(canvas_size) * 1.5 / 512.0)
    bowl_weighted_luma = cv2.GaussianBlur(
        median_luma * bowl_support,
        (0, 0),
        sigmaX=bowl_structure_sigma,
        sigmaY=bowl_structure_sigma,
    )
    bowl_blurred_support = cv2.GaussianBlur(
        bowl_support,
        (0, 0),
        sigmaX=bowl_structure_sigma,
        sigmaY=bowl_structure_sigma,
    )
    bowl_low_frequency_luma = bowl_weighted_luma / np.maximum(
        bowl_blurred_support,
        0.01,
    )
    bowl_structure_values = bowl_low_frequency_luma[prop_union]
    bowl_luma_p10, bowl_luma_p90 = np.percentile(
        bowl_structure_values,
        (10, 90),
    )
    bowl_luma_contrast = float(bowl_luma_p90 - bowl_luma_p10)
    # Audit the real continuous muzzle tip after smoothing. No artificial ellipse,
    # point, hook, or detached semantic object is added: those shapes were read by
    # Qwen as a ball or tongue. Image 1 (the upload) owns identity; this Image 2
    # supplies only the real source-anchored contact geometry.
    contact_subject = selected_subject.copy()
    contact_subject[:contact_y0, :] = False
    contact_subject[min(canvas_size[1], bowl_top + 1) :, :] = False
    contact_subject[:, :contact_x0] = False
    contact_subject[:, contact_x1:] = False
    contact_rows, contact_columns = np.where(contact_subject)
    if contact_rows.size < 32:
        raise ValueError(
            "eating_geometry_guide_unreliable: muzzle_semantic_anchor_missing"
        )
    lowest_contact_y = int(contact_rows.max())
    muzzle_food_gap = max(0, bowl_top - lowest_contact_y - 1)
    if muzzle_food_gap > 4:
        raise ValueError(
            "eating_geometry_guide_unreliable: "
            f"continuous_muzzle_food_gap_px={muzzle_food_gap}"
        )
    contact_tip = contact_columns[contact_rows >= max(contact_y0, lowest_contact_y - 5)]
    if contact_tip.size < 8:
        raise ValueError(
            "eating_geometry_guide_unreliable: muzzle_semantic_tip_missing"
        )
    muzzle_center_x = int(round(float(np.median(contact_tip))))
    muzzle_tip_mask = (
        contact_subject
        & (np.indices(contact_subject.shape)[0] >= max(contact_y0, lowest_contact_y - 5))
    )
    muzzle_tip_bbox = binary_bbox(muzzle_tip_mask)

    # Preserve prop meaning without preserving any driver pixels. Fixed semantic
    # colors make the shallow bowl and brown kibble readable while being
    # independent of the Main_case animal identity and lighting.
    hsv = cv2.cvtColor(median_rgb, cv2.COLOR_RGB2HSV)
    row_grid = np.indices(prop_union.shape)[0]
    food_zone_bottom = min(
        canvas_size[1],
        bowl_top + max(8, int(round(bowl_height * 0.42))),
    )
    warm_food_seed = (
        prop_union
        & (row_grid >= max(0, bowl_top - 2))
        & (row_grid < food_zone_bottom)
        & (hsv[:, :, 0] >= 2)
        & (hsv[:, :, 0] <= 30)
        & (hsv[:, :, 1] >= 38)
        & (hsv[:, :, 2] >= 24)
        & (hsv[:, :, 2] <= 235)
    )
    food_core = cv2.morphologyEx(
        warm_food_seed.astype(np.uint8),
        cv2.MORPH_OPEN,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    ).astype(bool)
    minimum_food_pixels = max(24, int(round(bowl_width * bowl_height * 0.004)))
    if int(np.count_nonzero(food_core)) < minimum_food_pixels:
        upper_prop = (
            prop_union
            & (row_grid >= max(0, bowl_top - 2))
            & (row_grid < food_zone_bottom)
        )
        upper_luma = median_luma[upper_prop]
        if upper_luma.size:
            dark_cutoff = float(np.percentile(upper_luma, 42))
            food_core = upper_prop & (median_luma <= dark_cutoff)
    food_core &= prop_union
    food_core_pixels = int(np.count_nonzero(food_core))
    if food_core_pixels < minimum_food_pixels:
        raise ValueError(
            "eating_geometry_guide_unreliable: "
            f"semantic_food_core_too_small={food_core_pixels}"
        )

    bowl_semantic_rgb = np.array([214, 208, 196], dtype=np.uint8)
    bowl_rim_semantic_rgb = np.array([232, 228, 218], dtype=np.uint8)
    food_semantic_rgb = np.array([142, 86, 45], dtype=np.uint8)
    bowl_rim = (
        prop_union
        & (row_grid < bowl_top + max(5, int(round(bowl_height * 0.16))))
    )
    guide = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
    guide[:] = (0, 255, 0)
    guide[selected_subject] = subject_semantic_rgb
    guide[prop_union] = bowl_semantic_rgb
    guide[bowl_rim] = bowl_rim_semantic_rgb
    guide[food_core] = food_semantic_rgb

    metadata = {
        "policy": "driving_eating_geometry_image2_v7",
        "method": (
            "temporal_head_lowering_flat_subject_semantic_food_fields_v8"
        ),
        "selected_frame_index": int(selected["index"]),
        "muzzle_bowl_gap_px": int(selected["muzzle_bowl_gap_px"]),
        "contact_pixels": int(selected["contact_pixels"]),
        "source_union_bbox": list(source_bbox),
        "subject_bbox": list(binary_bbox(selected_subject)),
        "bowl_bbox": [bowl_left, bowl_top, bowl_right, bowl_bottom],
        "contact_region_bbox": [contact_x0, contact_y0, contact_x1, contact_y1],
        "muzzle_contact_semantic": {
            "method": "source_anchored_continuous_neutral_muzzle_tip_v2",
            "bbox": list(muzzle_tip_bbox),
            "center_x": muzzle_center_x,
            "tip_y": lowest_contact_y,
            "muzzle_food_gap_px": muzzle_food_gap,
            "artificial_shape_added": False,
            "detached_marker_added": False,
            "source_rgb_preserved": False,
            "source_chroma_preserved": False,
            "identity_preserved": False,
            "passed": True,
        },
        "semantic_food": {
            "method": "fixed_beige_bowl_brown_kibble_from_source_mask_v1",
            "food_core_pixels": food_core_pixels,
            "minimum_food_core_pixels": minimum_food_pixels,
            "food_core_bbox": list(binary_bbox(food_core)),
            "bowl_rgb": bowl_semantic_rgb.tolist(),
            "bowl_rim_rgb": bowl_rim_semantic_rgb.tolist(),
            "food_rgb": food_semantic_rgb.tolist(),
            "source_rgb_preserved": False,
            "semantic_chroma_only": True,
            "passed": True,
        },
        "head_lowering_selection": {
            "method": "central_upper_subject_row_p10_v1",
            "search_region_bbox": [head_x0, 0, head_x1, bowl_top],
            "selected_head_y_p10": round(float(selected["head_y_p10"]), 3),
            "selected_head_support_pixels": int(
                selected["head_support_pixels"]
            ),
            "head_y_p10_min": round(min(head_y_values), 3),
            "head_y_p10_max": round(max(head_y_values), 3),
            "head_y_p10_span": round(head_lowering_span, 3),
            "head_y_p10_span_min": round(head_lowering_span_min, 3),
            "passed": True,
        },
        "subject_pixels": int(np.count_nonzero(selected_subject)),
        "bowl_pixels": int(np.count_nonzero(prop_union)),
        "outline_smoothing": {
            "morphology_kernel": outline_kernel_size,
            "gaussian_kernel": blur_kernel_size,
            "gaussian_sigma": round(blur_sigma, 3),
            "raw_perimeter": round(raw_subject_perimeter, 3),
            "cleaned_perimeter": round(cleaned_subject_perimeter, 3),
            "perimeter_ratio": round(
                cleaned_subject_perimeter / max(raw_subject_perimeter, 1.0),
                6,
            ),
            "high_frequency_outline_preserved": False,
        },
        "semantic_structure": {
            "method": "flat_uniform_subject_silhouette_v3",
            "subject_rgb": subject_semantic_rgb.tolist(),
            "subject_unique_rgb_count": 1,
            "internal_luma_variation": False,
            "appearance_bearing_face_detail": False,
            "bowl_gaussian_sigma": round(bowl_structure_sigma, 3),
            "bowl_luma_p10": round(float(bowl_luma_p10), 3),
            "bowl_luma_p90": round(float(bowl_luma_p90), 3),
            "bowl_luma_contrast": round(bowl_luma_contrast, 3),
            "source_luma_values_preserved": False,
            "source_rgb_preserved": False,
            "source_chroma_preserved": False,
            "high_frequency_texture_preserved": False,
        },
        "palette": {
            "background": [0, 255, 0],
            "subject_rgb": subject_semantic_rgb.tolist(),
            "prop_semantic_colors": {
                "bowl": bowl_semantic_rgb.tolist(),
                "bowl_rim": bowl_rim_semantic_rgb.tolist(),
                "food": food_semantic_rgb.tolist(),
            },
            "source_rgb_preserved": False,
            "source_chroma_preserved": False,
            "semantic_prop_chroma_preserved": True,
            "source_luma_values_preserved": False,
            "high_frequency_texture_preserved": False,
        },
        "prop_detection": prop_metadata,
    }
    return Image.fromarray(guide, mode="RGB"), metadata


def make_eating_geometry_guide_from_gif(
    source: Path,
    canvas_size: tuple[int, int],
    *,
    output_count: int,
    output_fps: float,
) -> tuple[Image.Image, dict[str, Any]]:
    frames, durations = read_animation_frames(source)
    sampled = sample_timed_frames(frames, durations, output_count, output_fps)
    guide, metadata = make_eating_geometry_guide(sampled, canvas_size)
    metadata.update(
        {
            "source": str(source),
            "source_sha256": sha256_file(source),
            "sampled_frame_count": output_count,
            "sampled_fps": output_fps,
            "selected_time_seconds": round(
                int(metadata["selected_frame_index"]) / output_fps,
                6,
            ),
        }
    )
    return guide, metadata


def make_grooming_geometry_guide_from_animation(
    source: Path,
    canvas_size: tuple[int, int],
    *,
    output_count: int,
    output_fps: float,
) -> tuple[Image.Image, dict[str, Any]]:
    frames, durations = read_animation_frames(source)
    sampled = sample_timed_frames(frames, durations, output_count, output_fps)
    white_maquette_source = "white_model" in source.stem.lower()
    white_maquette_selection: dict[str, Any] | None = None
    if white_maquette_source:
        midpoint = (len(sampled) - 1) / 2.0
        candidate_indices = sorted(
            {
                int(round((len(sampled) - 1) * fraction))
                for fraction in (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)
            },
            key=lambda index: (abs(index - midpoint), index),
        )
        rejected_candidates: list[dict[str, Any]] = []
        selected_source_index: int | None = None
        for candidate_index in candidate_indices:
            # The first white-maquette frame is the neutral motion reference.
            # Expose one candidate at its real timeline index so the existing
            # motion and anatomy gates stay unchanged while the old
            # whole-cycle maximum can no longer select late head-only motion
            # instead of the complete shoulder-to-paw relation.
            candidate_sequence = [sampled[0]] * len(sampled)
            candidate_sequence[candidate_index] = sampled[candidate_index]
            try:
                guide, metadata = make_grooming_geometry_guide(
                    candidate_sequence,
                    canvas_size,
                )
            except ValueError as exc:
                rejected_candidates.append(
                    {
                        "frame_index": candidate_index,
                        "reason": str(exc),
                    }
                )
                continue
            selected_source_index = candidate_index
            white_maquette_selection = {
                "policy": "midcycle_relation_gated_white_maquette_v1",
                "source_filename": source.name,
                "candidate_indices": candidate_indices,
                "selected_frame_index": candidate_index,
                "rejected_candidates": rejected_candidates,
                "reference_frame_index": 0,
                "identity_or_color_signal_used_for_candidate_ordering": False,
                "motion_and_muzzle_foreleg_relation_gates_reused": True,
            }
            break
        if selected_source_index is None:
            # Re-run the complete sequence to preserve the most informative
            # historical failure when none of the white-maquette anchors pass.
            guide, metadata = make_grooming_geometry_guide(sampled, canvas_size)
    else:
        guide, metadata = make_grooming_geometry_guide(sampled, canvas_size)
    if white_maquette_selection is not None:
        metadata["white_maquette_selection"] = white_maquette_selection
        metadata["source_white_maquette"] = True
    metadata.update(
        {
            "source": str(source),
            "source_sha256": sha256_file(source),
            "sampled_frame_count": output_count,
            "sampled_fps": output_fps,
            "selected_time_seconds": round(
                int(metadata["selected_frame_index"]) / output_fps,
                6,
            ),
        }
    )
    return guide, metadata


def make_eating_reference_aligned_pose_frames(
    sampled_frames: list[Image.Image],
    reference: Image.Image,
    reference_mask: Image.Image,
    canvas_size: tuple[int, int],
    *,
    maximum_head_lift_px: float = EATING_REFERENCE_HEAD_LIFT_PX,
    target_muzzle_food_overlap_px: float = (
        EATING_TARGET_MUZZLE_FOOD_OVERLAP_PX
    ),
) -> tuple[list[Image.Image], list[Image.Image], dict[str, Any]]:
    """Animate the accepted eating reference in its own mouth/bowl coordinates.

    The source GIF contributes only a scalar low-head/high-head timing curve. The
    pose video is synthesized from the accepted target reference, so neither the
    driving cat's coat nor its side-view body silhouette can enter SCAIL.
    """
    if len(sampled_frames) < 9:
        raise ValueError(
            "eating_reference_motion_unreliable: fewer_than_9_frames"
        )
    if reference.size != canvas_size or reference_mask.size != canvas_size:
        raise ValueError(
            "eating_reference_motion_unreliable: reference_size_mismatch"
        )
    if maximum_head_lift_px <= 0:
        raise ValueError(
            "eating_reference_motion_unreliable: invalid_head_lift_limit"
        )
    if not (
        EATING_MUZZLE_FOOD_OVERLAP_MIN_PX
        <= target_muzzle_food_overlap_px
        <= EATING_MUZZLE_FOOD_OVERLAP_MAX_PX
    ):
        raise ValueError(
            "eating_reference_motion_unreliable: "
            "invalid_target_muzzle_food_overlap"
        )

    source_bbox = expand_bbox(
        union_alpha_bbox(sampled_frames),
        sampled_frames[0].size,
        margin_fraction=0.06,
    )
    normalized_driver = [
        contain_rgba(frame.crop(source_bbox), canvas_size, 0.88)
        for frame in sampled_frames
    ]
    driver_prop_exclusions, prop_metadata = make_eating_prop_exclusion_masks(
        normalized_driver
    )
    driver_prop_union = np.logical_or.reduce(
        [
            np.asarray(mask, dtype=np.uint8) >= 128
            for mask in driver_prop_exclusions
        ]
    )
    driver_bowl_left, driver_bowl_top, driver_bowl_right, _driver_bowl_bottom = (
        binary_bbox(driver_prop_union)
    )
    driver_bowl_width = driver_bowl_right - driver_bowl_left
    # The planted near paw in the real eating driver remains beside and partly over the
    # bowl opening in every frame.  Measuring the lowest subject pixel above the
    # bowl therefore produces a constant zero gap even while the head clearly
    # rises.  Track a robust upper-head quantile in a wider bowl-relative band
    # instead: paws occupy the bottom of this band and cannot pin its 10th-row
    # percentile, while the moving forehead and ears provide a strong signal.
    driver_head_x0 = max(
        0,
        driver_bowl_left - int(round(driver_bowl_width * 0.55)),
    )
    driver_head_x1 = min(
        canvas_size[0],
        driver_bowl_right + int(round(driver_bowl_width * 0.05)),
    )
    if driver_head_x1 - driver_head_x0 < 24:
        raise ValueError(
            "eating_reference_motion_unreliable: invalid_driver_head_band"
        )

    driver_head_y: list[float] = []
    driver_head_pixels: list[int] = []
    for frame, prop_exclusion in zip(
        normalized_driver,
        driver_prop_exclusions,
        strict=True,
    ):
        alpha = np.asarray(frame.getchannel("A"), dtype=np.uint8) >= 16
        frame_prop = np.asarray(prop_exclusion, dtype=np.uint8) >= 128
        subject = alpha & ~frame_prop
        head_band = subject[
            :driver_bowl_top,
            driver_head_x0:driver_head_x1,
        ]
        rows, _columns = np.where(head_band)
        driver_head_pixels.append(int(rows.size))
        if rows.size >= 256:
            # Larger y means the driver head is lower and closer to the bowl.
            head_y = float(np.percentile(rows, 10))
        else:
            head_y = float("nan")
        driver_head_y.append(head_y)

    finite_head_y = np.asarray(
        [value for value in driver_head_y if np.isfinite(value)],
        dtype=np.float32,
    )
    if finite_head_y.size < max(5, len(sampled_frames) // 2):
        raise ValueError(
            "eating_reference_motion_unreliable: insufficient_head_signal"
        )
    high_head_y = float(np.percentile(finite_head_y, 10))
    low_head_y = float(np.percentile(finite_head_y, 90))
    head_y_span = low_head_y - high_head_y
    if head_y_span < 4.0:
        raise ValueError(
            "eating_reference_motion_unreliable: head_motion_range_too_small"
        )
    lift_fractions = [
        float(
            np.clip(
                (low_head_y - np.clip(value, high_head_y, low_head_y))
                / head_y_span,
                0.0,
                1.0,
            )
        )
        if np.isfinite(value)
        else 1.0
        for value in driver_head_y
    ]

    reference_rgba = np.asarray(reference.convert("RGBA"), dtype=np.uint8)
    colored_reference_mask = np.asarray(
        reference_mask.convert("RGB"),
        dtype=np.uint8,
    )
    reference_subject = np.all(
        colored_reference_mask == (0, 0, 255),
        axis=2,
    )
    reference_bowl = np.all(
        colored_reference_mask == (255, 0, 0),
        axis=2,
    )
    if int(np.count_nonzero(reference_subject)) < 500:
        raise ValueError(
            "eating_reference_motion_unreliable: reference_subject_too_small"
        )
    if int(np.count_nonzero(reference_bowl)) < 100:
        raise ValueError(
            "eating_reference_motion_unreliable: reference_bowl_too_small"
        )
    subject_left, subject_top, subject_right, subject_bottom = binary_bbox(
        reference_subject
    )
    bowl_left, bowl_top, bowl_right, bowl_bottom = binary_bbox(reference_bowl)
    subject_width = subject_right - subject_left
    subject_height = subject_bottom - subject_top
    bowl_width = bowl_right - bowl_left

    # Anchor the target motion to a real face/muzzle feature, not to the lowest
    # blue pixel in the bowl's central strip.  The latter can be chest fur, a
    # planted forepaw, or a leaked food pixel.  Eating references from the
    # protected profile have a small natural warm nose/muzzle patch.  Select a
    # compact red-balanced connected component immediately above the bowl and
    # track that component through the warp.  Disconnected chest/paw pixels do
    # not participate in either the anchor or the contact validation.
    reference_rgb = reference_rgba[..., :3]
    red = reference_rgb[..., 0].astype(np.int16)
    green = reference_rgb[..., 1].astype(np.int16)
    blue = reference_rgb[..., 2].astype(np.int16)
    row_indices, column_indices = np.indices(reference_subject.shape)
    face_search_x0 = max(
        subject_left,
        bowl_left - int(round(bowl_width * 0.32)),
    )
    face_search_x1 = min(
        subject_right,
        bowl_right + int(round(bowl_width * 0.32)),
    )
    face_search_y0 = max(
        subject_top,
        bowl_top
        - int(round(max(bowl_width * 0.90, subject_height * 0.40))),
    )
    face_search = (
        (column_indices >= face_search_x0)
        & (column_indices < face_search_x1)
        & (row_indices >= face_search_y0)
        & (row_indices < bowl_top)
    )
    face_seed_candidates = (
        reference_subject
        & face_search
        & (red >= 90)
        & (red - green >= 18)
        & (red - blue >= 8)
        & (np.abs(green - blue) <= 25)
    )
    face_count, face_labels, face_stats, face_centroids = (
        cv2.connectedComponentsWithStats(
            face_seed_candidates.astype(np.uint8),
            connectivity=8,
        )
    )
    face_components: list[tuple[int, int, float, int]] = []
    minimum_face_pixels = max(
        12,
        int(round(np.count_nonzero(reference_subject) * 0.00015)),
    )
    for label in range(1, face_count):
        area = int(face_stats[label, cv2.CC_STAT_AREA])
        component_y = int(face_stats[label, cv2.CC_STAT_TOP])
        component_width = int(face_stats[label, cv2.CC_STAT_WIDTH])
        component_height = int(face_stats[label, cv2.CC_STAT_HEIGHT])
        component_bottom = component_y + component_height
        component_gap = bowl_top - component_bottom
        center_x = float(face_centroids[label][0])
        if (
            area >= minimum_face_pixels
            and 0 <= component_gap <= 8
            and 3 <= component_width <= int(round(bowl_width * 0.55))
            and 2 <= component_height <= int(round(bowl_width * 0.35))
            and bowl_left - bowl_width * 0.10
            <= center_x
            <= bowl_right + bowl_width * 0.10
        ):
            face_components.append(
                (
                    component_bottom,
                    area,
                    -abs(center_x - (bowl_left + bowl_right) / 2.0),
                    label,
                )
            )
    if not face_components:
        raise ValueError(
            "eating_reference_motion_unreliable: no_face_muzzle_anchor"
        )
    _face_bottom_rank, _face_area_rank, _face_center_rank, face_label = max(
        face_components
    )
    reference_face_seed = face_labels == face_label
    face_seed_left, face_seed_top, face_seed_right, face_seed_bottom = (
        binary_bbox(reference_face_seed)
    )
    muzzle_x = int(round(float(face_centroids[face_label][0])))
    muzzle_y = face_seed_bottom - 1
    reference_muzzle_gap = bowl_top - face_seed_bottom

    head_center_x = float(muzzle_x)
    head_center_y = float(muzzle_y - subject_height * 0.16)
    head_radius_x = max(28.0, subject_width * 0.22)
    head_radius_y = max(34.0, subject_height * 0.30)
    row_grid, column_grid = np.indices((canvas_size[1], canvas_size[0]))
    head_weight = np.exp(
        -0.5
        * (
            ((column_grid - head_center_x) / head_radius_x) ** 2
            + ((row_grid - head_center_y) / head_radius_y) ** 2
        )
    ).astype(np.float32)
    # Keep the haunches and tail fixed; only the connected head/neck/chest region
    # follows the scalar eating nod.
    head_weight[row_grid > bowl_bottom + int(round(subject_height * 0.08))] = 0.0

    # Below the muzzle, freeze both lower lateral regions on the full remap grid.
    # This keeps the two planted front paws and their surrounding silhouette
    # pixel-identical while allowing only a narrow central mouth corridor to
    # enter the food.  The bowl remains a separate static red SCAIL object.
    freeze_y0 = min(face_seed_bottom, bowl_top)
    corridor_x0 = max(subject_left, face_seed_left - 4)
    corridor_x1 = min(subject_right, face_seed_right + 4)
    lower_lateral_freeze = (row_grid >= freeze_y0) & (
        (column_grid < corridor_x0) | (column_grid >= corridor_x1)
    )
    head_weight[lower_lateral_freeze] = 0.0

    # A valid reference may already touch the rim or leave a small 1-8 px gap.
    # Merely reducing that gap to zero still leaves the cat hovering above its
    # food.  Make the accepted reference the low phase by moving the real muzzle
    # seed a bounded 6-10 px into the upper food band.  Scale by the actual
    # face-seed deformation weight, and cap the correction at the existing head
    # lift envelope so the new downward motion cannot pull harder than the
    # already accepted upward motion.
    face_seed_bottom_band = reference_face_seed & (
        row_indices >= face_seed_bottom - 2
    )
    bottom_weights = head_weight[face_seed_bottom_band]
    if not bottom_weights.size:
        raise ValueError(
            "eating_reference_motion_unreliable: empty_face_seed_bottom"
        )
    contact_weight = max(float(np.percentile(bottom_weights, 25)), 0.25)
    requested_contact_drop_control_px = (
        float(reference_muzzle_gap)
        + float(target_muzzle_food_overlap_px)
        + 0.75
    ) / contact_weight
    contact_drop_control_px = min(
        requested_contact_drop_control_px,
        float(maximum_head_lift_px),
    )

    cat_layer = reference_rgba.copy()
    cat_layer[~reference_subject] = 0
    bowl_layer = reference_rgba.copy()
    bowl_layer[~reference_bowl] = 0
    subject_mask_u8 = reference_subject.astype(np.uint8) * 255
    pose_frames: list[Image.Image] = []
    mask_frames: list[Image.Image] = []
    applied_lifts: list[float] = []
    face_bowl_gaps: list[int] = []
    face_bowl_signed_gaps: list[int] = []
    face_food_overlap_depths: list[int] = []
    face_food_overlap_pixels: list[int] = []
    vertical_controls: list[float] = []
    map_x = column_grid.astype(np.float32)
    face_seed_u8 = reference_face_seed.astype(np.uint8) * 255
    food_contact_band = (
        reference_bowl
        & (row_indices >= bowl_top)
        & (row_indices < min(bowl_bottom, bowl_top + 11))
    )
    for lift_fraction in lift_fractions:
        lift_px = float(maximum_head_lift_px * lift_fraction)
        applied_lifts.append(lift_px)
        # cv2.remap is output-to-source: positive control lifts the head, while
        # the low-phase contact correction uses a small negative control.
        vertical_control_px = lift_px - contact_drop_control_px
        vertical_controls.append(vertical_control_px)
        map_y = (row_grid + vertical_control_px * head_weight).astype(np.float32)
        warped_cat = cv2.remap(
            cat_layer,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        warped_subject = cv2.remap(
            subject_mask_u8,
            map_x,
            map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        warped_subject_bool = warped_subject >= 128
        warped_subject_bool &= ~reference_bowl
        warped_face_seed = cv2.remap(
            face_seed_u8,
            map_x,
            map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        raw_warped_face_seed = warped_face_seed >= 128
        warped_face_rows, _warped_face_columns = np.where(raw_warped_face_seed)
        if not warped_face_rows.size:
            raise ValueError(
                "eating_reference_motion_unreliable: warped_face_seed_empty"
            )
        signed_face_bowl_gap = bowl_top - int(warped_face_rows.max()) - 1
        face_bowl_signed_gaps.append(signed_face_bowl_gap)
        face_bowl_gaps.append(max(0, signed_face_bowl_gap))
        face_food_overlap_depths.append(max(0, -signed_face_bowl_gap))
        face_food_overlap_pixels.append(
            int(
                np.count_nonzero(
                    raw_warped_face_seed & food_contact_band
                )
            )
        )

        pose_rgba = np.zeros_like(reference_rgba)
        pose_rgba[:] = (255, 255, 255, 255)
        cat_alpha = warped_cat[..., 3:4].astype(np.float32) / 255.0
        pose_rgba[..., :3] = np.clip(
            warped_cat[..., :3] * cat_alpha
            + pose_rgba[..., :3] * (1.0 - cat_alpha),
            0,
            255,
        ).astype(np.uint8)
        bowl_alpha = bowl_layer[..., 3:4].astype(np.float32) / 255.0
        pose_rgba[..., :3] = np.clip(
            bowl_layer[..., :3] * bowl_alpha
            + pose_rgba[..., :3] * (1.0 - bowl_alpha),
            0,
            255,
        ).astype(np.uint8)
        pose_frames.append(Image.fromarray(pose_rgba[..., :3], mode="RGB"))

        colored_mask = np.full(
            (canvas_size[1], canvas_size[0], 3),
            255,
            dtype=np.uint8,
        )
        colored_mask[warped_subject_bool] = (0, 0, 255)
        colored_mask[reference_bowl] = (255, 0, 0)
        mask_frames.append(Image.fromarray(colored_mask, mode="RGB"))

    low_phase_index = int(np.argmin(np.asarray(applied_lifts)))
    low_phase_face_gap = face_bowl_gaps[low_phase_index]
    low_phase_signed_gap = face_bowl_signed_gaps[low_phase_index]
    low_phase_overlap_depth = face_food_overlap_depths[low_phase_index]
    low_phase_overlap_pixels = face_food_overlap_pixels[low_phase_index]
    if not (
        EATING_MUZZLE_FOOD_OVERLAP_MIN_PX
        <= low_phase_overlap_depth
        <= EATING_MUZZLE_FOOD_OVERLAP_MAX_PX
    ) or low_phase_overlap_pixels <= 0:
        raise ValueError(
            "eating_reference_motion_unreliable: "
            "unsafe_muzzle_food_overlap: "
            f"signed_gap={low_phase_signed_gap}, "
            f"overlap_depth={low_phase_overlap_depth}, "
            f"overlap_pixels={low_phase_overlap_pixels}"
        )

    metadata = {
        "policy": "reference_space_head_nod_food_overlap_v2",
        "driver_signal_only": True,
        "driver_rgb_preserved": False,
        "driver_silhouette_preserved": False,
        "source_union_bbox": list(source_bbox),
        "driver_signal_method": "bowl_relative_upper_head_row_p10_v1",
        "driver_head_band_bbox": [
            driver_head_x0,
            0,
            driver_head_x1,
            driver_bowl_top,
        ],
        "driver_head_y_p10": round(high_head_y, 4),
        "driver_head_y_p90": round(low_head_y, 4),
        "driver_head_y_span": round(head_y_span, 4),
        "driver_head_y_by_frame": [
            round(value, 4) if np.isfinite(value) else None
            for value in driver_head_y
        ],
        "driver_head_pixels_min": min(driver_head_pixels),
        "head_lift_px_min": round(min(applied_lifts), 4),
        "head_lift_px_max": round(max(applied_lifts), 4),
        "head_lift_px_by_frame": [round(value, 4) for value in applied_lifts],
        "reference_subject_bbox": [
            subject_left,
            subject_top,
            subject_right,
            subject_bottom,
        ],
        "reference_bowl_bbox": [bowl_left, bowl_top, bowl_right, bowl_bottom],
        "reference_muzzle_method": "warm_face_component_above_bowl_v1",
        "reference_face_search_bbox": [
            face_search_x0,
            face_search_y0,
            face_search_x1,
            bowl_top,
        ],
        "reference_face_seed_bbox": [
            face_seed_left,
            face_seed_top,
            face_seed_right,
            face_seed_bottom,
        ],
        "reference_face_seed_pixels": int(np.count_nonzero(reference_face_seed)),
        "reference_muzzle_anchor": [muzzle_x, muzzle_y],
        "reference_muzzle_bowl_gap_px": reference_muzzle_gap,
        "target_muzzle_food_overlap_px": float(
            target_muzzle_food_overlap_px
        ),
        "reference_contact_drop_px": round(
            reference_muzzle_gap + target_muzzle_food_overlap_px,
            4,
        ),
        "reference_contact_drop_requested_control_px": round(
            requested_contact_drop_control_px,
            4,
        ),
        "reference_contact_drop_control_px": round(contact_drop_control_px, 4),
        "reference_contact_drop_capped": (
            requested_contact_drop_control_px > maximum_head_lift_px
        ),
        "face_bowl_gap_px_by_frame": face_bowl_gaps,
        "face_bowl_signed_gap_px_by_frame": face_bowl_signed_gaps,
        "muzzle_food_overlap_depth_px_by_frame": face_food_overlap_depths,
        "muzzle_food_overlap_pixels_by_frame": face_food_overlap_pixels,
        "low_phase_face_bowl_gap_px": low_phase_face_gap,
        "low_phase_face_bowl_signed_gap_px": low_phase_signed_gap,
        "low_phase_muzzle_food_overlap_depth_px": low_phase_overlap_depth,
        "low_phase_muzzle_food_overlap_pixels": low_phase_overlap_pixels,
        "vertical_control_px_by_frame": [
            round(value, 4) for value in vertical_controls
        ],
        "vertical_control_abs_max_px": round(
            max(abs(value) for value in vertical_controls),
            4,
        ),
        "muzzle_motion_corridor_bbox": [
            corridor_x0,
            freeze_y0,
            corridor_x1,
            bowl_bottom,
        ],
        "lower_lateral_freeze_bboxes": [
            [subject_left, freeze_y0, corridor_x0, subject_bottom],
            [corridor_x1, freeze_y0, subject_right, subject_bottom],
        ],
        "head_deformation_center": [
            round(head_center_x, 4),
            round(head_center_y, 4),
        ],
        "head_deformation_radius": [
            round(head_radius_x, 4),
            round(head_radius_y, 4),
        ],
        "scene_prop_separated": True,
        "static_bowl_mask": True,
        "static_bowl_rgb": True,
        "scene_prop_metadata": prop_metadata,
    }
    return pose_frames, mask_frames, metadata


def make_static_reference_eating_pose_frames(
    reference: Image.Image,
    reference_mask: Image.Image,
    canvas_size: tuple[int, int],
    *,
    frame_count: int,
    fallback_reason: str,
) -> tuple[list[Image.Image], list[Image.Image], dict[str, Any]]:
    """Return an identity-safe eating pose when bounded head motion is unavailable."""
    if reference.size != canvas_size or reference_mask.size != canvas_size:
        raise ValueError(
            "eating_static_reference_fallback_unreliable: reference_size_mismatch"
        )
    if frame_count < 1:
        raise ValueError(
            "eating_static_reference_fallback_unreliable: invalid_frame_count"
        )

    rgba = np.asarray(reference.convert("RGBA"), dtype=np.uint8)
    reference_mask_rgb = np.asarray(
        reference_mask.convert("RGB"),
        dtype=np.uint8,
    )
    identity = np.all(reference_mask_rgb == (0, 0, 255), axis=2)
    prop = np.all(reference_mask_rgb == (255, 0, 0), axis=2)
    if int(np.count_nonzero(identity)) < 500:
        raise ValueError(
            "eating_static_reference_fallback_unreliable: reference_subject_too_small"
        )
    if int(np.count_nonzero(prop)) < 100:
        raise ValueError(
            "eating_static_reference_fallback_unreliable: reference_bowl_too_small"
        )

    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    pose_rgb = np.full((*rgba.shape[:2], 3), 255, dtype=np.uint8)
    pose_rgb = np.clip(
        rgba[..., :3] * alpha + pose_rgb * (1.0 - alpha),
        0,
        255,
    ).astype(np.uint8)
    colored_mask = np.full((*rgba.shape[:2], 3), 255, dtype=np.uint8)
    colored_mask[identity] = (0, 0, 255)
    colored_mask[prop] = (255, 0, 0)
    pose = Image.fromarray(pose_rgb, mode="RGB")
    mask = Image.fromarray(colored_mask, mode="RGB")
    identity_bbox = list(binary_bbox(identity))
    prop_bbox = list(binary_bbox(prop))
    return (
        [pose.copy() for _ in range(frame_count)],
        [mask.copy() for _ in range(frame_count)],
        {
            "policy": "static_reference_eating_pose_identity_fallback_v1",
            "fallback_used": True,
            "fallback_reason": fallback_reason[:1000],
            "driver_signal_only": False,
            "driver_rgb_preserved": False,
            "driver_silhouette_preserved": False,
            "reference_identity_preserved": True,
            "pose_geometry_scope": "static_approved_eating_reference",
            "reference_subject_bbox": identity_bbox,
            "reference_bowl_bbox": prop_bbox,
            "identity_fraction": float(np.count_nonzero(identity))
            / float(identity.size),
            "scene_prop_fraction": float(np.count_nonzero(prop))
            / float(prop.size),
            "scene_prop_separated": True,
            "static_bowl_mask": True,
            "static_bowl_rgb": True,
            "scene_prop_metadata": {
                "policy": "reference_mask_static_bowl_fallback_v1",
                "fallback_reason": fallback_reason[:1000],
            },
        },
    )


def make_drinking_reference_aligned_pose_frames(
    sampled_frames: list[Image.Image],
    reference: Image.Image,
    reference_mask: Image.Image,
    canvas_size: tuple[int, int],
    *,
    frame_count: int,
    maximum_head_lift_px: float = DRINKING_REFERENCE_HEAD_LIFT_PX,
    lap_cycles: int = DRINKING_REFERENCE_LAP_CYCLES,
) -> tuple[list[Image.Image], list[Image.Image], dict[str, Any]]:
    """Retarget the white driver's head path onto the accepted dog reference.

    The small-dog drinking driver has identity-bearing upright ears and a tightly
    curled tail. Its RGB and silhouette must therefore never enter SCAIL as identity.
    This path extracts only the upper-body motion curve, retargets that curve inside
    the accepted reference's head/neck region, and keeps the reference bowl static.
    """
    if reference.size != canvas_size or reference_mask.size != canvas_size:
        raise ValueError(
            "drinking_reference_motion_unreliable: reference_size_mismatch"
        )
    if frame_count < 9:
        raise ValueError(
            "drinking_reference_motion_unreliable: fewer_than_9_frames"
        )
    if maximum_head_lift_px < DRINKING_REFERENCE_HEAD_LIFT_MIN_PX or maximum_head_lift_px > 48:
        raise ValueError(
            "drinking_reference_motion_unreliable: invalid_head_lift_limit"
        )
    if lap_cycles < 1 or lap_cycles > 6:
        raise ValueError(
            "drinking_reference_motion_unreliable: invalid_lap_cycle_count"
        )
    if len(sampled_frames) != frame_count:
        raise ValueError(
            "drinking_reference_motion_unreliable: driver_frame_count_mismatch"
        )

    cleaned_driver_frames: list[Image.Image] = []
    driver_cleanup_records: list[dict[str, Any]] = []
    for frame in sampled_frames:
        driver_rgba = np.asarray(frame.convert("RGBA"), dtype=np.uint8).copy()
        driver_silhouette, cleanup = white_maquette_semantic_silhouette(
            driver_rgba
        )
        driver_luma = cv2.cvtColor(driver_rgba[..., :3], cv2.COLOR_RGB2GRAY)
        driver_rgba[..., :3] = np.repeat(driver_luma[..., None], 3, axis=2)
        driver_rgba[..., 3] = driver_silhouette
        driver_rgba[driver_silhouette == 0, :3] = 0
        cleaned_driver_frames.append(Image.fromarray(driver_rgba, mode="RGBA"))
        driver_cleanup_records.append(cleanup)
    driver_source_bbox = expand_bbox(
        union_alpha_bbox(cleaned_driver_frames),
        cleaned_driver_frames[0].size,
        margin_fraction=0.06,
    )
    normalized_driver = [
        contain_rgba(frame.crop(driver_source_bbox), canvas_size, 0.88)
        for frame in cleaned_driver_frames
    ]
    driver_masks = [
        np.asarray(frame.getchannel("A"), dtype=np.uint8) >= 16
        for frame in normalized_driver
    ]
    driver_union = np.logical_or.reduce(driver_masks)
    driver_left, driver_top, driver_right, driver_bottom = binary_bbox(
        driver_union
    )
    driver_width = driver_right - driver_left
    driver_height = driver_bottom - driver_top
    driver_rows, driver_columns = np.indices(driver_union.shape)
    driver_upper_limit = driver_top + int(round(driver_height * 0.70))
    driver_upper_center_x: list[float] = []
    driver_upper_center_y: list[float] = []
    driver_upper_pixels: list[int] = []
    for driver_mask in driver_masks:
        driver_upper = driver_mask & (driver_rows < driver_upper_limit)
        upper_y, upper_x = np.where(driver_upper)
        driver_upper_pixels.append(int(upper_y.size))
        if upper_y.size < 256:
            raise ValueError(
                "drinking_reference_motion_unreliable: insufficient_driver_head_signal"
            )
        driver_upper_center_x.append(float(np.mean(upper_x)))
        driver_upper_center_y.append(float(np.mean(upper_y)))
    driver_high_y = float(np.percentile(driver_upper_center_y, 10))
    driver_low_y = float(np.percentile(driver_upper_center_y, 90))
    driver_head_y_span = driver_low_y - driver_high_y
    if driver_head_y_span < 4.0:
        raise ValueError(
            "drinking_reference_motion_unreliable: driver_head_motion_range_too_small"
        )
    driver_lift_fractions = [
        float(
            np.clip(
                (driver_low_y - center_y) / driver_head_y_span,
                0.0,
                1.0,
            )
        )
        for center_y in driver_upper_center_y
    ]
    low_phase_x_values = [
        center_x
        for center_x, lift_fraction in zip(
            driver_upper_center_x,
            driver_lift_fractions,
            strict=True,
        )
        if lift_fraction <= 0.20
    ]
    driver_low_phase_x = float(
        np.median(low_phase_x_values or driver_upper_center_x)
    )

    reference_rgba = np.asarray(reference.convert("RGBA"), dtype=np.uint8).copy()
    reference_mask_rgb = np.asarray(
        reference_mask.convert("RGB"),
        dtype=np.uint8,
    )
    reference_identity = np.all(
        reference_mask_rgb == (0, 0, 255),
        axis=2,
    )
    reference_prop = np.all(
        reference_mask_rgb == (255, 0, 0),
        axis=2,
    )
    identity_pixels = int(np.count_nonzero(reference_identity))
    if identity_pixels < 500:
        raise ValueError(
            "drinking_reference_motion_unreliable: reference_subject_too_small"
        )
    prop_pixels = int(np.count_nonzero(reference_prop))
    if prop_pixels < 64:
        raise ValueError(
            "drinking_reference_motion_unreliable: reference_bowl_not_separated"
        )

    subject_left, subject_top, subject_right, subject_bottom = binary_bbox(
        reference_identity
    )
    subject_width = subject_right - subject_left
    subject_height = subject_bottom - subject_top
    scaled_driver_head_lift_px = (
        driver_head_y_span
        * subject_height
        / float(max(1, driver_height))
        * 2.35
    )
    effective_head_lift_px = float(
        np.clip(
            scaled_driver_head_lift_px,
            DRINKING_REFERENCE_HEAD_LIFT_MIN_PX,
            maximum_head_lift_px,
        )
    )
    driver_horizontal_scale = subject_width / float(max(1, driver_width))
    row_indices, column_indices = np.indices(reference_identity.shape)
    reference_rgb = reference_rgba[..., :3]
    red = reference_rgb[..., 0].astype(np.int16)
    green = reference_rgb[..., 1].astype(np.int16)
    blue = reference_rgb[..., 2].astype(np.int16)

    # Prefer a visible tongue as the mouth anchor, but inspect the original
    # alpha subject rather than the already split blue identity.  A tongue that
    # physically touches the water can otherwise be absorbed into the red bowl
    # component by a generic prop splitter.
    reference_alpha = reference_rgba[..., 3] >= 16
    tongue_candidates = (
        reference_alpha
        & (row_indices >= subject_top + int(round(subject_height * 0.35)))
        & (red >= 110)
        & (red - green >= 25)
        & (blue - green >= 5)
    )
    tongue_count, tongue_labels, tongue_stats, tongue_centroids = (
        cv2.connectedComponentsWithStats(
            tongue_candidates.astype(np.uint8),
            connectivity=8,
        )
    )
    minimum_tongue_pixels = max(6, int(round(identity_pixels * 0.00005)))
    maximum_tongue_pixels = max(120, int(round(identity_pixels * 0.04)))
    tongue_components: list[tuple[int, int, float, int]] = []
    for label in range(1, tongue_count):
        area = int(tongue_stats[label, cv2.CC_STAT_AREA])
        component_top = int(tongue_stats[label, cv2.CC_STAT_TOP])
        component_height = int(tongue_stats[label, cv2.CC_STAT_HEIGHT])
        component_bottom = component_top + component_height
        center_x = float(tongue_centroids[label][0])
        if (
            minimum_tongue_pixels <= area <= maximum_tongue_pixels
            and subject_left <= center_x < subject_right
        ):
            tongue_components.append(
                (
                    area,
                    component_bottom,
                    -abs(center_x - (subject_left + subject_right) / 2.0),
                    label,
                )
            )
    tongue_anchor_detected = bool(tongue_components)
    if tongue_anchor_detected:
        _area_rank, _bottom_rank, _center_rank, tongue_label = max(
            tongue_components
        )
        reference_motion_anchor = tongue_labels == tongue_label
        # The tongue belongs to the dog even when the prop splitter connected
        # it to the water surface.  Reclassify only the colour-qualified tongue
        # component; the surrounding bowl remains immutable red geometry.
        reference_identity = reference_identity | reference_motion_anchor
        reference_prop = reference_prop & ~reference_motion_anchor
        motion_anchor_policy = "pink_tongue_component_from_reference_alpha_v2"
    else:
        # Some valid drinking keyframes hide the tongue or render it in a coat-
        # coloured material.  Find the nearest lower identity pixels to the
        # separated bowl and use that compact mouth/contact patch as the motion
        # anchor.  This keeps action generation available without inventing a
        # driver silhouette or moving the bowl.
        prop_left, prop_top, prop_right, _prop_bottom = binary_bbox(
            reference_prop
        )
        distance_to_prop = cv2.distanceTransform(
            (~reference_prop).astype(np.uint8),
            cv2.DIST_L2,
            3,
        )
        contact_distance_max = max(12.0, subject_height * 0.08)
        contact_corridor = (
            reference_identity
            & (column_indices >= prop_left - int(round(subject_width * 0.12)))
            & (column_indices < prop_right + int(round(subject_width * 0.12)))
            & (row_indices >= subject_top + int(round(subject_height * 0.30)))
            & (row_indices < prop_top + int(round(subject_height * 0.12)))
            & (distance_to_prop <= contact_distance_max)
        )
        if not np.any(contact_corridor):
            raise ValueError(
                "drinking_reference_motion_unreliable: "
                "no_mouth_to_bowl_contact_anchor"
            )
        nearest_distance = float(np.min(distance_to_prop[contact_corridor]))
        reference_motion_anchor = contact_corridor & (
            distance_to_prop <= nearest_distance + 3.0
        )
        if int(np.count_nonzero(reference_motion_anchor)) < 3:
            contact_y, contact_x = np.where(contact_corridor)
            lowest_index = int(np.argmax(contact_y))
            anchor_x = int(contact_x[lowest_index])
            anchor_y = int(contact_y[lowest_index])
            anchor_disk = (
                (column_indices - anchor_x) ** 2
                + (row_indices - anchor_y) ** 2
                <= 8**2
            )
            reference_motion_anchor = anchor_disk & reference_identity
        motion_anchor_policy = "nearest_identity_to_static_bowl_contact_v1"

    anchor_left, anchor_top, anchor_right, anchor_bottom = binary_bbox(
        reference_motion_anchor
    )
    anchor_y, anchor_x = np.where(reference_motion_anchor)
    tongue_center_x = float(np.mean(anchor_x))
    tongue_center_y = float(np.mean(anchor_y))

    head_center_x = tongue_center_x
    head_center_y = max(
        float(subject_top),
        float(anchor_top) - subject_height * 0.16,
    )
    head_radius_x = max(28.0, subject_width * 0.22)
    head_radius_y = max(34.0, subject_height * 0.25)
    head_weight = np.exp(
        -0.5
        * (
            ((column_indices - head_center_x) / head_radius_x) ** 2
            + ((row_indices - head_center_y) / head_radius_y) ** 2
        )
    ).astype(np.float32)
    corridor_half_width = max(34.0, subject_width * 0.28)
    head_weight[
        (column_indices < head_center_x - corridor_half_width)
        | (column_indices > head_center_x + corridor_half_width)
    ] = 0.0
    lower_static_y = min(canvas_size[1], anchor_bottom + 2)
    attachment_half_width = max(18.0, head_radius_x * 0.70)
    attachment_corridor = (
        (column_indices >= head_center_x - attachment_half_width)
        & (column_indices <= head_center_x + attachment_half_width)
        & (row_indices >= head_center_y - head_radius_y * 0.15)
        & (row_indices < lower_static_y)
    )
    attachment_weight = np.exp(
        -0.5 * ((column_indices - head_center_x) / head_radius_x) ** 2
    ).astype(np.float32)
    head_weight[attachment_corridor] = np.maximum(
        head_weight[attachment_corridor],
        attachment_weight[attachment_corridor],
    )
    head_weight[row_indices >= lower_static_y] = 0.0
    head_weight[head_weight < 0.08] = 0.0
    motion_region = head_weight > 0.0
    if int(np.count_nonzero(reference_identity & motion_region)) < 200:
        raise ValueError(
            "drinking_reference_motion_unreliable: head_motion_region_too_small"
        )

    identity_layer = reference_rgba.copy()
    identity_layer[~reference_identity] = 0
    prop_layer = reference_rgba.copy()
    prop_layer[~reference_prop] = 0
    identity_mask_u8 = reference_identity.astype(np.uint8) * 255
    motion_anchor_mask_u8 = reference_motion_anchor.astype(np.uint8) * 255
    pose_frames: list[Image.Image] = []
    mask_frames: list[Image.Image] = []
    head_lifts: list[float] = []
    head_horizontal_shifts: list[float] = []
    tongue_center_y_by_frame: list[float] = []
    motion_anchor_main_component_coverages: list[float] = []

    for index, driver_lift_fraction in enumerate(driver_lift_fractions):
        phase = (2.0 * np.pi * lap_cycles * index) / float(frame_count - 1)
        lap_px = float(
            2.0
            * (1.0 - driver_lift_fraction)
            * (0.5 - 0.5 * np.cos(phase))
        )
        lift_px = float(
            effective_head_lift_px * driver_lift_fraction + lap_px
        )
        horizontal_shift_px = float(
            np.clip(
                (driver_upper_center_x[index] - driver_low_phase_x)
                * driver_horizontal_scale,
                -10.0,
                10.0,
            )
        )
        if index == frame_count - 1:
            lift_px = head_lifts[0]
            horizontal_shift_px = head_horizontal_shifts[0]
        head_lifts.append(lift_px)
        head_horizontal_shifts.append(horizontal_shift_px)
        map_x_frame = (
            column_indices - horizontal_shift_px * head_weight
        ).astype(np.float32)
        map_y = (row_indices + lift_px * head_weight).astype(np.float32)
        warped_identity_layer = cv2.remap(
            identity_layer,
            map_x_frame,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        warped_identity = cv2.remap(
            identity_mask_u8,
            map_x_frame,
            map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ) >= 128
        warped_motion_anchor = cv2.remap(
            motion_anchor_mask_u8,
            map_x_frame,
            map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ) >= 128
        anchor_rows, _anchor_columns = np.where(warped_motion_anchor)
        if not anchor_rows.size:
            raise ValueError(
                "drinking_reference_motion_unreliable: "
                "warped_mouth_anchor_empty"
            )
        component_count, component_labels, component_stats, _ = (
            cv2.connectedComponentsWithStats(
                warped_identity.astype(np.uint8),
                connectivity=8,
            )
        )
        if component_count < 2:
            raise ValueError(
                "drinking_reference_motion_unreliable: "
                "warped_identity_empty"
            )
        main_component_label = 1 + int(
            np.argmax(component_stats[1:, cv2.CC_STAT_AREA])
        )
        anchor_main_coverage = float(
            np.count_nonzero(
                warped_motion_anchor
                & (component_labels == main_component_label)
            )
        ) / float(max(1, np.count_nonzero(warped_motion_anchor)))
        motion_anchor_main_component_coverages.append(anchor_main_coverage)
        if anchor_main_coverage < 0.80:
            raise ValueError(
                "drinking_reference_motion_unreliable: "
                "mouth_anchor_detached_from_subject"
            )
        tongue_center_y_by_frame.append(float(np.mean(anchor_rows)))

        pose_rgb = np.full((*reference_identity.shape, 3), 255, dtype=np.uint8)
        prop_alpha = prop_layer[..., 3:4].astype(np.float32) / 255.0
        pose_rgb = np.clip(
            prop_layer[..., :3] * prop_alpha
            + pose_rgb * (1.0 - prop_alpha),
            0,
            255,
        ).astype(np.uint8)
        warped_alpha = (
            warped_identity_layer[..., 3:4].astype(np.float32) / 255.0
        )
        pose_rgb = np.clip(
            warped_identity_layer[..., :3] * warped_alpha
            + pose_rgb * (1.0 - warped_alpha),
            0,
            255,
        ).astype(np.uint8)
        pose_frames.append(Image.fromarray(pose_rgb, mode="RGB"))

        colored_mask = np.full((*reference_identity.shape, 3), 255, dtype=np.uint8)
        colored_mask[reference_prop] = (255, 0, 0)
        colored_mask[warped_identity] = (0, 0, 255)
        mask_frames.append(Image.fromarray(colored_mask, mode="RGB"))

    tongue_motion_span = max(tongue_center_y_by_frame) - min(
        tongue_center_y_by_frame
    )
    if tongue_motion_span < 2.0:
        raise ValueError(
            "drinking_reference_motion_unreliable: tongue_motion_too_small"
        )
    if not (
        np.array_equal(np.asarray(pose_frames[0]), np.asarray(pose_frames[-1]))
        and np.array_equal(np.asarray(mask_frames[0]), np.asarray(mask_frames[-1]))
    ):
        raise ValueError(
            "drinking_reference_motion_unreliable: loop_endpoints_not_closed"
        )

    return (
        pose_frames,
        mask_frames,
        {
            "policy": DRINKING_REFERENCE_ALIGNED_POSE_POLICY,
            "fallback_used": False,
            "driver_input_used": True,
            "driver_signal_only": True,
            "driver_rgb_preserved": False,
            "driver_silhouette_preserved": False,
            "reference_identity_preserved": True,
            "pose_geometry_scope": (
                "reference_locked_identity_plus_white_driver_head_path"
            ),
            "driver_source_bbox": list(driver_source_bbox),
            "driver_subject_bbox": [
                driver_left,
                driver_top,
                driver_right,
                driver_bottom,
            ],
            "driver_signal_method": "upper_subject_centroid_xy_p10_p90_v1",
            "driver_upper_limit_y": driver_upper_limit,
            "driver_upper_pixels_min": min(driver_upper_pixels),
            "driver_upper_center_x_by_frame": [
                round(value, 4) for value in driver_upper_center_x
            ],
            "driver_upper_center_y_by_frame": [
                round(value, 4) for value in driver_upper_center_y
            ],
            "driver_head_y_p10": round(driver_high_y, 4),
            "driver_head_y_p90": round(driver_low_y, 4),
            "driver_head_y_span": round(driver_head_y_span, 4),
            "driver_lift_fraction_by_frame": [
                round(value, 4) for value in driver_lift_fractions
            ],
            "driver_cleanup_policy": driver_cleanup_records[0]["policy"],
            "scaled_driver_head_lift_px": round(
                scaled_driver_head_lift_px,
                4,
            ),
            "effective_head_lift_px": round(effective_head_lift_px, 4),
            "reference_subject_bbox": [
                subject_left,
                subject_top,
                subject_right,
                subject_bottom,
            ],
            "reference_tongue_bbox": [
                anchor_left,
                anchor_top,
                anchor_right,
                anchor_bottom,
            ] if tongue_anchor_detected else None,
            "reference_motion_anchor_bbox": [
                anchor_left,
                anchor_top,
                anchor_right,
                anchor_bottom,
            ],
            "reference_tongue_pixels": (
                int(np.count_nonzero(reference_motion_anchor))
                if tongue_anchor_detected
                else 0
            ),
            "reference_motion_anchor_pixels": int(
                np.count_nonzero(reference_motion_anchor)
            ),
            "tongue_anchor_detected": tongue_anchor_detected,
            "motion_anchor_policy": motion_anchor_policy,
            "reference_tongue_center": [
                round(tongue_center_x, 4),
                round(tongue_center_y, 4),
            ],
            "head_deformation_center": [
                round(head_center_x, 4),
                round(head_center_y, 4),
            ],
            "head_deformation_radius": [
                round(head_radius_x, 4),
                round(head_radius_y, 4),
            ],
            "attachment_corridor_half_width": round(
                attachment_half_width,
                4,
            ),
            "head_motion_region_bbox": list(binary_bbox(motion_region)),
            "head_lift_px_by_frame": [round(value, 4) for value in head_lifts],
            "head_lift_px_max": round(max(head_lifts), 4),
            "head_horizontal_shift_px_by_frame": [
                round(value, 4) for value in head_horizontal_shifts
            ],
            "head_horizontal_shift_span_px": round(
                max(head_horizontal_shifts) - min(head_horizontal_shifts),
                4,
            ),
            "lap_cycles": lap_cycles,
            "tongue_center_y_by_frame": [
                round(value, 4) for value in tongue_center_y_by_frame
            ],
            "tongue_motion_span_px": round(tongue_motion_span, 4),
            "motion_anchor_main_component_coverage_min": round(
                min(motion_anchor_main_component_coverages),
                6,
            ),
            "motion_anchor_connected_all_frames": True,
            "lower_foreground_static_y_start": lower_static_y,
            "body_ears_paws_tail_outside_head_motion_locked": True,
            "closed_loop_endpoints": True,
            "scene_prop_separated": True,
            "scene_prop_fraction": prop_pixels / float(reference_prop.size),
            "static_bowl_mask": True,
            "static_bowl_rgb": True,
        },
    )


def make_static_reference_drinking_pose_frames(
    reference: Image.Image,
    reference_mask: Image.Image,
    canvas_size: tuple[int, int],
    *,
    frame_count: int,
    fallback_reason: str,
) -> tuple[list[Image.Image], list[Image.Image], dict[str, Any]]:
    """Keep the approved drinking identity when a bounded head lap is unavailable."""
    if reference.size != canvas_size or reference_mask.size != canvas_size:
        raise ValueError(
            "drinking_static_reference_fallback_unreliable: reference_size_mismatch"
        )
    if frame_count < 1:
        raise ValueError(
            "drinking_static_reference_fallback_unreliable: invalid_frame_count"
        )
    rgba = np.asarray(reference.convert("RGBA"), dtype=np.uint8).copy()
    mask_rgb = np.asarray(reference_mask.convert("RGB"), dtype=np.uint8)
    identity = np.all(mask_rgb == (0, 0, 255), axis=2)
    prop = np.all(mask_rgb == (255, 0, 0), axis=2)
    if int(np.count_nonzero(identity)) < 500:
        raise ValueError(
            "drinking_static_reference_fallback_unreliable: reference_subject_too_small"
        )
    if int(np.count_nonzero(prop)) < 64:
        raise ValueError(
            "drinking_static_reference_fallback_unreliable: reference_bowl_too_small"
        )
    rgba[~(identity | prop)] = 0
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    pose_rgb = np.full((*identity.shape, 3), 255, dtype=np.uint8)
    pose_rgb = np.clip(
        rgba[..., :3] * alpha + pose_rgb * (1.0 - alpha),
        0,
        255,
    ).astype(np.uint8)
    colored_mask = np.full((*identity.shape, 3), 255, dtype=np.uint8)
    colored_mask[identity] = (0, 0, 255)
    colored_mask[prop] = (255, 0, 0)
    pose = Image.fromarray(pose_rgb, mode="RGB")
    mask = Image.fromarray(colored_mask, mode="RGB")
    return (
        [pose.copy() for _ in range(frame_count)],
        [mask.copy() for _ in range(frame_count)],
        {
            "policy": "static_reference_drinking_pose_identity_fallback_v1",
            "fallback_used": True,
            "fallback_reason": fallback_reason[:1000],
            "driver_input_used": False,
            "driver_signal_only": False,
            "driver_rgb_preserved": False,
            "driver_silhouette_preserved": False,
            "reference_identity_preserved": True,
            "pose_geometry_scope": "static_approved_drinking_reference",
            "reference_subject_bbox": list(binary_bbox(identity)),
            "body_ears_paws_tail_outside_head_motion_locked": True,
            "scene_prop_separated": True,
            "static_bowl_mask": True,
            "static_bowl_rgb": True,
        },
    )


def make_full_drinking_pose_and_mask_frames(
    sampled_frames: list[Image.Image],
    canvas_size: tuple[int, int],
    *,
    reference_retarget_failure: str | None = None,
) -> tuple[list[Image.Image], list[Image.Image], dict[str, Any]]:
    """Use the complete white driver as dog drinking's geometry authority.

    The approved upload remains SCAIL's identity reference.  Only action geometry
    comes from the digest-bound white maquette: the dog silhouette varies on every
    frame, while a temporal vote freezes the bowl mask and bowl RGB for the loop.
    A reference-space head-only retarget is not a high-quality alternative because
    it removes the approach, body shift, contact, and release phases before GPU
    submission.
    """

    frame_count = len(sampled_frames)
    if frame_count < 9:
        raise ValueError(
            "drinking_full_driver_fallback_unreliable: fewer_than_9_frames"
        )

    cleaned_frames: list[Image.Image] = []
    cleanup_records: list[dict[str, Any]] = []
    for frame in sampled_frames:
        driver_rgba = np.asarray(frame.convert("RGBA"), dtype=np.uint8).copy()
        driver_silhouette, cleanup = white_maquette_semantic_silhouette(
            driver_rgba
        )
        driver_luma = cv2.cvtColor(driver_rgba[..., :3], cv2.COLOR_RGB2GRAY)
        driver_rgba[..., :3] = np.repeat(driver_luma[..., None], 3, axis=2)
        driver_rgba[..., 3] = driver_silhouette
        driver_rgba[driver_silhouette == 0, :3] = 0
        cleaned_frames.append(Image.fromarray(driver_rgba, mode="RGBA"))
        cleanup_records.append(cleanup)

    source_bbox = expand_bbox(
        union_alpha_bbox(cleaned_frames),
        cleaned_frames[0].size,
        margin_fraction=0.06,
    )
    normalized_frames = [
        contain_rgba(frame.crop(source_bbox), canvas_size, 0.88)
        for frame in cleaned_frames
    ]

    detected_prop_masks: list[np.ndarray] = []
    detected_frame_indices: list[int] = []
    detection_records: list[dict[str, Any]] = []
    detection_errors: list[dict[str, Any]] = []
    for index, normalized in enumerate(normalized_frames):
        try:
            colored_mask, _prop_exclusion, metadata = (
                make_drinking_reference_identity_mask(normalized)
            )
        except ValueError as exc:
            if not str(exc).startswith(
                "drinking_reference_prop_split_unreliable:"
            ):
                raise
            detection_errors.append(
                {"frame_index": index, "error": str(exc)[:500]}
            )
            continue
        colored_rgb = np.asarray(colored_mask.convert("RGB"), dtype=np.uint8)
        prop = np.all(colored_rgb == (255, 0, 0), axis=2)
        if int(np.count_nonzero(prop)) < 64:
            detection_errors.append(
                {
                    "frame_index": index,
                    "error": "detected_prop_smaller_than_64_pixels",
                }
            )
            continue
        detected_prop_masks.append(prop)
        detected_frame_indices.append(index)
        detection_records.append(metadata)

    minimum_detected_frames = max(6, int(np.ceil(frame_count * 0.60)))
    if len(detected_prop_masks) < minimum_detected_frames:
        raise ValueError(
            "drinking_full_driver_fallback_unreliable: "
            f"bowl_detected_frames={len(detected_prop_masks)}; "
            f"minimum={minimum_detected_frames}"
        )

    prop_stack = np.stack(detected_prop_masks, axis=0)
    temporal_vote_minimum = max(
        3,
        int(np.ceil(len(detected_prop_masks) * 0.25)),
    )
    static_prop = np.count_nonzero(prop_stack, axis=0) >= temporal_vote_minimum
    static_prop = cv2.morphologyEx(
        static_prop.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((7, 7), dtype=np.uint8),
        iterations=1,
    ).astype(bool)
    static_prop = _largest_binary_component(
        static_prop,
        minimum_area=64,
        error_prefix="drinking_full_driver_fallback_unreliable",
    )
    static_prop_pixels = int(np.count_nonzero(static_prop))
    canvas_area = canvas_size[0] * canvas_size[1]
    static_prop_fraction = static_prop_pixels / float(canvas_area)
    if not 0.001 <= static_prop_fraction <= 0.20:
        raise ValueError(
            "drinking_full_driver_fallback_unreliable: "
            f"static_bowl_fraction={static_prop_fraction:.6f}"
        )

    prop_y, prop_x = np.where(static_prop)
    detected_rgb = np.stack(
        [
            np.asarray(normalized_frames[index].convert("RGB"), dtype=np.uint8)
            for index in detected_frame_indices
        ],
        axis=0,
    )
    point_rgb = detected_rgb[:, prop_y, prop_x, :]
    point_observed = prop_stack[:, prop_y, prop_x]
    static_point_rgb = np.median(point_rgb, axis=0)
    observed_points = np.any(point_observed, axis=0)
    if np.any(observed_points):
        observed_rgb = np.where(
            point_observed[:, observed_points, None],
            point_rgb[:, observed_points, :].astype(np.float32),
            np.nan,
        )
        static_point_rgb[observed_points] = np.nanmedian(
            observed_rgb,
            axis=0,
        )
    static_prop_rgb = np.full(
        (canvas_size[1], canvas_size[0], 3),
        255,
        dtype=np.uint8,
    )
    static_prop_rgb[prop_y, prop_x] = np.clip(
        np.rint(static_point_rgb),
        0,
        255,
    ).astype(np.uint8)

    pose_frames: list[Image.Image] = []
    mask_frames: list[Image.Image] = []
    identity_masks: list[np.ndarray] = []
    identity_bboxes: list[list[int]] = []
    for normalized in normalized_frames:
        rgba = np.asarray(normalized.convert("RGBA"), dtype=np.uint8)
        hard_alpha = cv2.dilate(
            (rgba[..., 3] >= 16).astype(np.uint8),
            np.ones((5, 5), dtype=np.uint8),
            iterations=1,
        ).astype(bool)
        identity = hard_alpha & ~static_prop
        identity = _largest_binary_component(
            identity,
            minimum_area=500,
            error_prefix="drinking_full_driver_fallback_unreliable",
        )
        identity_masks.append(identity)
        identity_bboxes.append(list(binary_bbox(identity)))

        pose_rgb = np.full(
            (canvas_size[1], canvas_size[0], 3),
            255,
            dtype=np.uint8,
        )
        pose_rgb[identity] = rgba[..., :3][identity]
        pose_rgb[static_prop] = static_prop_rgb[static_prop]
        pose_frames.append(Image.fromarray(pose_rgb, mode="RGB"))

        colored_mask = np.full_like(pose_rgb, 255)
        colored_mask[identity] = (0, 0, 255)
        colored_mask[static_prop] = (255, 0, 0)
        mask_frames.append(Image.fromarray(colored_mask, mode="RGB"))

    first_identity = identity_masks[0]
    identity_change_ratios = [
        float(np.count_nonzero(np.logical_xor(first_identity, identity)))
        / float(canvas_area)
        for identity in identity_masks
    ]
    identity_change_ratio_max = max(identity_change_ratios)
    if identity_change_ratio_max < DRINKING_FULL_DRIVER_CHANGE_RATIO_MIN:
        raise ValueError(
            "drinking_full_driver_fallback_unreliable: "
            f"identity_change_ratio_max={identity_change_ratio_max:.6f}; "
            f"minimum={DRINKING_FULL_DRIVER_CHANGE_RATIO_MIN:.6f}"
        )

    return (
        pose_frames,
        mask_frames,
        {
            "policy": DRINKING_FULL_WHITE_DRIVER_POSE_POLICY,
            "primary_policy": reference_retarget_failure is None,
            "fallback_used": False,
            "reference_retarget_failure": (
                reference_retarget_failure[:1000]
                if reference_retarget_failure
                else None
            ),
            "driver_input_used": True,
            "driver_signal_only": False,
            "driver_rgb_preserved": False,
            "driver_silhouette_preserved": True,
            "full_white_driver_geometry": True,
            "reference_identity_preserved": False,
            "pose_geometry_scope": "full_white_driver_subject_per_frame",
            "identity_authority": "approved_upload_drinking_reference",
            "action_geometry_authority": "digest_bound_white_maquette_video",
            "scene_prop_authority": "digest_bound_white_maquette_video",
            "source_union_bbox": list(source_bbox),
            "identity_bbox_by_frame": identity_bboxes,
            "identity_motion_audit": {
                "policy": "first_frame_blue_identity_mask_xor_v1",
                "reference_frame_index": 0,
                "change_ratio_by_frame": [
                    round(value, 6) for value in identity_change_ratios
                ],
                "change_ratio_max": round(identity_change_ratio_max, 6),
                "change_ratio_max_frame_index": int(
                    np.argmax(identity_change_ratios)
                ),
                "change_ratio_minimum": DRINKING_FULL_DRIVER_CHANGE_RATIO_MIN,
                "passed": True,
            },
            "driver_cleanup_policy": cleanup_records[0]["policy"],
            "driver_cleanup_visible_fraction_min": min(
                float(record["visible_fraction"])
                for record in cleanup_records
            ),
            "bowl_detection_policy": detection_records[0]["policy"],
            "bowl_detected_frames": len(detected_prop_masks),
            "bowl_detection_minimum_frames": minimum_detected_frames,
            "bowl_detection_failed_frames": detection_errors,
            "bowl_temporal_vote_minimum_frames": temporal_vote_minimum,
            "scene_prop_separated": True,
            "scene_prop_fraction": static_prop_fraction,
            "scene_prop_bbox": list(binary_bbox(static_prop)),
            "static_bowl_mask": True,
            "static_bowl_rgb": True,
            "closed_loop_endpoints": bool(
                np.array_equal(
                    np.asarray(pose_frames[0]),
                    np.asarray(pose_frames[-1]),
                )
                and np.array_equal(
                    np.asarray(mask_frames[0]),
                    np.asarray(mask_frames[-1]),
                )
            ),
        },
    )


def _make_eating_reference_identity_mask_strict(
    normalized: Image.Image,
) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
    """Map the cat to blue and its bowl to red using strict visual anchors."""
    rgba = np.asarray(normalized.convert("RGBA"), dtype=np.uint8)
    rgb = rgba[..., :3]
    alpha = rgba[..., 3] >= 16
    if not np.any(alpha):
        raise ValueError("eating_reference_prop_split_unreliable: empty_alpha")

    left, top, right, bottom = binary_bbox(alpha)
    subject_width = right - left
    subject_height = bottom - top
    if subject_width < 64 or subject_height < 64:
        raise ValueError("eating_reference_prop_split_unreliable: tiny_subject")

    row_indices, column_indices = np.indices(alpha.shape)
    red = rgb[..., 0].astype(np.int16)
    green = rgb[..., 1].astype(np.int16)
    blue = rgb[..., 2].astype(np.int16)
    subject_center_x = (left + right) / 2.0

    # The full-subject bbox can be strongly biased by a long tail.  Keep the
    # search in the lower part of the reference, but leave enough horizontal
    # room for a bowl placed under an off-centre muzzle.
    zone_x0 = left + int(round(subject_width * 0.02))
    zone_x1 = left + int(round(subject_width * 0.98))
    zone_y0 = top + int(round(subject_height * 0.65))
    bowl_zone = (
        (column_indices >= zone_x0)
        & (column_indices < zone_x1)
        & (row_indices >= zone_y0)
        & (row_indices < bottom)
    )

    definite_prop = (
        alpha
        & bowl_zone
        & (red >= 45)
        & (red - green >= 28)
        & (green - blue >= 18)
        & (red - blue >= 52)
    )
    definite_pixels = int(np.count_nonzero(definite_prop))
    minimum_definite_pixels = max(
        80,
        int(round(np.count_nonzero(alpha) * 0.0015)),
    )
    if definite_pixels < minimum_definite_pixels:
        raise ValueError(
            "eating_reference_prop_split_unreliable: "
            f"warm_seed_pixels={definite_pixels}<"
            f"{minimum_definite_pixels}"
        )

    # Use the food as a high-confidence anchor.  Looking only in the bottom
    # 18 percent used to reject valid references whose kibble sits above a
    # tall, pale bowl; at that depth the warm core has already disappeared.
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        definite_prop.astype(np.uint8),
        connectivity=8,
    )
    preliminary_candidates: list[int] = []
    candidate_bottoms: dict[int, int] = {}
    candidate_shapes: dict[int, tuple[int, int, int]] = {}
    component_bottom_tolerance = max(
        2,
        int(round(subject_height * 0.01)),
    )
    component_bottom_min = top + int(round(subject_height * 0.76))
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        center_x, center_y = centroids[label]
        component_bottom = int(stats[label, cv2.CC_STAT_TOP]) + int(
            stats[label, cv2.CC_STAT_HEIGHT]
        )
        # Real Main_case kibble is a row of several small separated pellets,
        # not one 100-pixel warm component.  Keep every credible lower-central
        # pellet and merge their union into one food anchor; tiny pink nose or
        # paw accents remain outside this lower band or fail the area floor.
        if (
            area >= 6
            and component_bottom + component_bottom_tolerance
            >= component_bottom_min
            and center_y >= zone_y0
            and abs(center_x - subject_center_x) <= subject_width * 0.36
        ):
            preliminary_candidates.append(label)
            candidate_bottoms[label] = component_bottom
            candidate_shapes[label] = (
                int(stats[label, cv2.CC_STAT_WIDTH]),
                int(stats[label, cv2.CC_STAT_HEIGHT]),
                area,
            )
    if not preliminary_candidates:
        raise ValueError(
            "eating_reference_prop_split_unreliable: no_warm_food_component"
        )
    # A warm dog/cat muzzle can physically touch the food and therefore pass
    # every lower-central test above.  It is usually a tall component ending
    # above the actual kibble row.  Keep only components in a narrow band of
    # the deepest credible warm component before constructing the food union;
    # this prevents the touching muzzle (and nearby warm paws) from becoming a
    # red static SCAIL prop while still allowing separated kibble pellets.
    horizontal_food_candidates = [
        label
        for label in preliminary_candidates
        if candidate_shapes[label][2] >= 20
        and candidate_shapes[label][0]
        >= max(12, int(round(candidate_shapes[label][1] * 1.4)))
    ]
    food_candidate_pool = (
        horizontal_food_candidates
        if horizontal_food_candidates
        else preliminary_candidates
    )
    food_candidate_selection = (
        "deepest_horizontal_warm_food_components_v1"
        if horizontal_food_candidates
        else "deepest_warm_food_components_v1"
    )
    deepest_candidate_bottom = max(
        candidate_bottoms[label] for label in food_candidate_pool
    )
    food_bottom_band_tolerance = max(
        6,
        int(round(subject_height * 0.04)),
    )
    candidates = [
        label
        for label in food_candidate_pool
        if candidate_bottoms[label]
        >= deepest_candidate_bottom - food_bottom_band_tolerance
    ]
    food_core = np.isin(labels, np.asarray(candidates, dtype=labels.dtype))
    food_area = int(np.count_nonzero(food_core))
    food_x, food_y, food_right, food_bottom = binary_bbox(food_core)
    food_width = food_right - food_x
    food_height = food_bottom - food_y

    # A tall warm component reaching the subject baseline is already the bowl
    # plus its food, not merely a food anchor.  Qwen often joins that orange
    # bowl to a pale paw with a few antialiased pixels.  Erode it until the
    # narrow bridge breaks, select the lower central core, then recover only
    # high-confidence warm pixels inside a tightly bounded horizontal window.
    # This keeps both paws blue instead of letting a relaxed beige search turn
    # them into the red prop identity.
    warm_component_is_bowl = (
        food_height / float(subject_height) >= 0.18
        and food_y + food_height >= top + int(round(subject_height * 0.94))
    )
    warm_component_initially_classified_as_bowl = warm_component_is_bowl
    warm_bowl_core_fallback_to_beige = False
    if warm_component_is_bowl:
        eroded_warm = cv2.erode(
            food_core.astype(np.uint8),
            np.ones((3, 3), dtype=np.uint8),
            iterations=3,
        )
        body_count, body_labels, body_stats, body_centroids = (
            cv2.connectedComponentsWithStats(eroded_warm, connectivity=8)
        )
        body_candidates: list[tuple[int, float, int]] = []
        for label in range(1, body_count):
            area = int(body_stats[label, cv2.CC_STAT_AREA])
            component_bottom = int(body_stats[label, cv2.CC_STAT_TOP]) + int(
                body_stats[label, cv2.CC_STAT_HEIGHT]
            )
            center_x = float(body_centroids[label][0])
            if (
                area >= 120
                and component_bottom >= top + int(round(subject_height * 0.90))
                and abs(center_x - subject_center_x) <= subject_width * 0.24
            ):
                body_candidates.append((area, -abs(center_x - subject_center_x), label))
        if body_candidates:
            _body_area_rank, _center_rank, body_label = max(body_candidates)
            body_area = int(body_stats[body_label, cv2.CC_STAT_AREA])
            body_x = int(body_stats[body_label, cv2.CC_STAT_LEFT])
            body_y = int(body_stats[body_label, cv2.CC_STAT_TOP])
            body_width = int(body_stats[body_label, cv2.CC_STAT_WIDTH])
            body_height = int(body_stats[body_label, cv2.CC_STAT_HEIGHT])
            prop_x0 = max(zone_x0, body_x - 6)
            prop_x1 = min(zone_x1, body_x + body_width + 6)
            bowl_body_method = "warm_bowl_eroded_core_v1"
            # Keep only the warm component that was actually selected as the
            # food/bowl object.  Using every warm pixel in the lower search zone
            # also painted a disconnected pink nose red in real Qwen references,
            # which then froze the nose into the static bowl layer.
            prop_seed = food_core
        else:
            # A thin warm rim joined to warm kibble can make the union tall
            # enough to resemble a solid warm bowl, yet leave no thick core
            # after erosion. Reuse the stricter food-anchored beige recovery
            # below; its width, prop fraction, paw support and contact checks
            # still fail closed instead of accepting the union directly.
            warm_component_is_bowl = False
            warm_bowl_core_fallback_to_beige = True
    if not warm_component_is_bowl:
        # Real profile references use a light-beige bowl, so the strict warm
        # mask normally covers the kibble rather than the bowl body. Recover
        # the lower beige component connected to that food anchor. The
        # asymmetric allowance reflects the common three-quarter eating pose.
        bowl_search_x0 = max(
            zone_x0,
            food_x - max(8, int(round(food_width * 0.10))),
        )
        bowl_search_x1 = min(
            zone_x1,
            food_x + food_width + max(8, int(round(food_width * 0.10))),
        )
        # A row of separated kibble often occupies only the central part of a
        # wide pale bowl.  Deriving the complete search width from that row
        # clipped valid Qwen bowls to 11-20% of the subject bbox.  Guarantee a
        # conservative food-centred search corridor; the connected-component,
        # prop-fraction, and two-sided paw-support checks below still prevent
        # this corridor from silently absorbing the planted paws.
        minimum_food_anchored_search_width = max(
            food_width + 16,
            int(round(subject_width * 0.32)),
        )
        food_center_x = (food_x + food_right) / 2.0
        bowl_search_x0 = max(
            zone_x0,
            min(
                bowl_search_x0,
                int(round(food_center_x - minimum_food_anchored_search_width / 2.0)),
            ),
        )
        bowl_search_x1 = min(
            zone_x1,
            max(
                bowl_search_x1,
                int(round(food_center_x + minimum_food_anchored_search_width / 2.0)),
            ),
        )
        warm_bowl_body = (
            alpha
            & (column_indices >= bowl_search_x0)
            & (column_indices < bowl_search_x1)
            & (row_indices >= max(zone_y0, food_y - 2))
            & (row_indices < bottom)
            & (red >= 45)
            & (red - green >= 6)
            & (red - blue >= 12)
            & (green - blue >= 8)
        )
        channel_max = np.maximum(np.maximum(red, green), blue)
        channel_min = np.minimum(np.minimum(red, green), blue)
        neutral_bowl_body = (
            alpha
            & (column_indices >= bowl_search_x0)
            & (column_indices < bowl_search_x1)
            & (row_indices >= max(zone_y0, food_y - 3))
            & (row_indices < bottom)
            & (channel_max >= 72)
            & ((channel_max - channel_min) <= 34)
        )
        # Preserve the neutral shaded base without reopening the lateral paws.
        bowl_base_x0 = max(
            bowl_search_x0,
            food_x - int(round(food_width * 0.05)),
        )
        bowl_base_x1 = min(
            bowl_search_x1,
            food_x + food_width + int(round(food_width * 0.02)),
        )
        neutral_bowl_base = (
            alpha
            & (column_indices >= bowl_base_x0)
            & (column_indices < bowl_base_x1)
            & (row_indices >= food_y + food_height - 2)
            & (row_indices < bottom)
            & (red >= 45)
            & (red - blue >= 12)
            & (green - blue >= 8)
        )
        bowl_body_seed = warm_bowl_body | neutral_bowl_body | neutral_bowl_base
        bowl_body_seed = cv2.morphologyEx(
            bowl_body_seed.astype(np.uint8),
            cv2.MORPH_CLOSE,
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        ).astype(bool)
        body_count, body_labels, body_stats, _body_centroids = (
            cv2.connectedComponentsWithStats(
                bowl_body_seed.astype(np.uint8),
                connectivity=8,
            )
        )
        food_support = cv2.dilate(
            food_core.astype(np.uint8),
            np.ones((5, 5), dtype=np.uint8),
            iterations=1,
        ).astype(bool)
        body_candidates_relaxed: list[tuple[int, int, int]] = []
        for label in range(1, body_count):
            component = body_labels == label
            anchor_overlap = int(np.count_nonzero(component & food_support))
            area = int(body_stats[label, cv2.CC_STAT_AREA])
            if anchor_overlap >= 20 and area >= food_area:
                body_candidates_relaxed.append((anchor_overlap, area, label))
        if not body_candidates_relaxed:
            raise ValueError(
                "eating_reference_prop_split_unreliable: no_bowl_body_component"
            )
        _anchor_overlap, body_area, body_label = max(body_candidates_relaxed)
        body_x = int(body_stats[body_label, cv2.CC_STAT_LEFT])
        body_y = int(body_stats[body_label, cv2.CC_STAT_TOP])
        body_width = int(body_stats[body_label, cv2.CC_STAT_WIDTH])
        body_height = int(body_stats[body_label, cv2.CC_STAT_HEIGHT])
        prop_x0 = max(zone_x0, body_x - 2)
        prop_x1 = min(zone_x1, body_x + body_width + 2)
        bowl_body_method = (
            "food_anchored_beige_bowl_after_eroded_warm_core_miss_v2"
            if warm_bowl_core_fallback_to_beige
            else "warm_food_core_plus_beige_bowl_v1"
        )
        prop_seed = body_labels == body_label

    safe_width_fraction = (prop_x1 - prop_x0) / float(subject_width)
    if not 0.25 <= safe_width_fraction <= 0.55:
        raise ValueError(
            "eating_reference_prop_split_unreliable: "
            f"safe_width_fraction={safe_width_fraction:.4f}"
        )

    safe_zone = bowl_zone & (column_indices >= prop_x0) & (column_indices < prop_x1)
    prop = prop_seed & safe_zone
    prop = cv2.morphologyEx(
        prop.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    )
    prop = cv2.dilate(
        prop,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    ).astype(bool)
    prop = fill_internal_holes(prop)
    prop &= safe_zone & alpha

    safety_zone = (
        (column_indices >= prop_x0)
        & (column_indices < prop_x1)
        & (row_indices >= max(0, zone_y0 - 2))
        & (row_indices < min(alpha.shape[0], bottom + 2))
    )
    prop_exclusion = cv2.dilate(
        prop.astype(np.uint8),
        np.ones((5, 5), dtype=np.uint8),
        iterations=1,
    ).astype(bool)
    prop_exclusion &= safety_zone

    hard_alpha = cv2.dilate(
        alpha.astype(np.uint8),
        np.ones((5, 5), dtype=np.uint8),
        iterations=1,
    ).astype(bool)
    identity = hard_alpha & ~prop_exclusion

    # Validate identity support immediately beside the detected bowl.  Fixed
    # percentages of the full bbox are unreliable when a long tail shifts the
    # subject centre; in that case the old "right paw" zone fell inside the
    # bowl itself.
    paw_span = max(16, int(round(subject_width * 0.20)))
    paw_y0 = top + int(round(subject_height * 0.72))
    left_paw_zone = (
        (column_indices >= max(left, prop_x0 - paw_span))
        & (column_indices < prop_x0)
        & (row_indices >= paw_y0)
        & (row_indices < bottom)
    )
    right_paw_zone = (
        (column_indices >= prop_x1)
        & (column_indices < min(right, prop_x1 + paw_span))
        & (row_indices >= paw_y0)
        & (row_indices < bottom)
    )
    left_support = int(np.count_nonzero(identity & left_paw_zone))
    right_support = int(np.count_nonzero(identity & right_paw_zone))
    minimum_paw_support = 500
    edge_margin = max(8, int(round(subject_width * 0.08)))
    prop_near_left_edge = prop_x0 - left <= edge_margin
    prop_near_right_edge = right - prop_x1 <= edge_margin
    left_support_accepted = left_support >= minimum_paw_support or (
        prop_near_left_edge and right_support >= minimum_paw_support
    )
    right_support_accepted = right_support >= minimum_paw_support or (
        prop_near_right_edge and left_support >= minimum_paw_support
    )
    if not left_support_accepted or not right_support_accepted:
        raise ValueError(
            "eating_reference_prop_split_unreliable: "
            f"paw_support=({left_support},{right_support})"
        )

    prop_inside_subject = prop_exclusion & hard_alpha
    prop_fraction = float(np.count_nonzero(prop_inside_subject)) / float(
        max(1, np.count_nonzero(hard_alpha))
    )
    if not 0.04 <= prop_fraction <= 0.20:
        raise ValueError(
            "eating_reference_prop_split_unreliable: "
            f"prop_fraction={prop_fraction:.4f}"
        )

    colored = np.zeros((*alpha.shape, 3), dtype=np.uint8)
    colored[identity] = (0, 0, 255)
    colored[prop_exclusion] = (255, 0, 0)
    colored_image = Image.fromarray(colored, mode="RGB")
    exclusion_image = Image.fromarray(
        prop_exclusion.astype(np.uint8) * 255,
        mode="L",
    )
    metadata = {
        "policy": "eating_reference_cat_blue_bowl_red_v1",
        "split_status": "strict",
        "fallback_used": False,
        "fallback_reason": None,
        "subject_bbox": list((left, top, right, bottom)),
        "bowl_zone_bbox": list(binary_bbox(bowl_zone)),
        "warm_seed_pixels": definite_pixels,
        "minimum_warm_seed_pixels": minimum_definite_pixels,
        "food_core_bbox": [
            food_x,
            food_y,
            food_x + food_width,
            food_y + food_height,
        ],
        "food_core_pixels": food_area,
        "food_core_components": len(candidates),
        "food_preliminary_components": len(preliminary_candidates),
        "food_horizontal_components": len(horizontal_food_candidates),
        "food_candidate_selection": food_candidate_selection,
        "food_deepest_component_bottom": deepest_candidate_bottom,
        "food_bottom_band_tolerance": food_bottom_band_tolerance,
        "food_component_bottom_min": component_bottom_min,
        "food_component_bottom_tolerance": component_bottom_tolerance,
        "warm_component_initially_classified_as_bowl": (
            warm_component_initially_classified_as_bowl
        ),
        "warm_bowl_core_fallback_to_beige": warm_bowl_core_fallback_to_beige,
        "deep_body_bbox": [
            body_x,
            body_y,
            body_x + body_width,
            body_y + body_height,
        ],
        "deep_body_pixels": body_area,
        "bowl_body_method": bowl_body_method,
        "food_anchored_search_width_min_fraction": 0.32,
        "safe_x_range": [prop_x0, prop_x1],
        "safe_width_fraction": safe_width_fraction,
        "paw_identity_y0": paw_y0,
        "prop_exclusion_bbox": list(binary_bbox(prop_exclusion)),
        "prop_fraction": prop_fraction,
        "left_paw_identity_support": left_support,
        "right_paw_identity_support": right_support,
        "minimum_paw_identity_support": minimum_paw_support,
        "prop_edge_margin": edge_margin,
        "prop_near_left_edge": prop_near_left_edge,
        "prop_near_right_edge": prop_near_right_edge,
        "left_paw_identity_support_accepted": left_support_accepted,
        "right_paw_identity_support_accepted": right_support_accepted,
    }
    return colored_image, exclusion_image, metadata


def _make_eating_reference_identity_mask_fallback(
    normalized: Image.Image,
    *,
    strict_error: ValueError,
) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
    """Build a conservative lower-centre prop mask after strict split failure.

    This fallback deliberately prefers a usable SCAIL reference over rejecting
    the complete video job because a generated bowl or food uses an unexpected
    colour.  A relaxed warm-food anchor is used when available.  Otherwise the
    mask reserves a narrow lower-centre corridor where the eating prompt and
    driver place the bowl, leaving the lateral planted-paw regions as identity.
    """

    rgba = np.asarray(normalized.convert("RGBA"), dtype=np.uint8)
    rgb = rgba[..., :3]
    alpha = rgba[..., 3] >= 16
    if not np.any(alpha):
        raise ValueError("eating_reference_prop_split_unreliable: empty_alpha")

    left, top, right, bottom = binary_bbox(alpha)
    subject_width = right - left
    subject_height = bottom - top
    if subject_width < 8 or subject_height < 8:
        raise ValueError("eating_reference_prop_split_unreliable: tiny_subject")

    row_indices, column_indices = np.indices(alpha.shape)
    red = rgb[..., 0].astype(np.int16)
    green = rgb[..., 1].astype(np.int16)
    blue = rgb[..., 2].astype(np.int16)
    channel_max = np.maximum(np.maximum(red, green), blue)
    channel_min = np.minimum(np.minimum(red, green), blue)
    subject_center_x = (left + right) / 2.0

    search_y0 = top + int(round(subject_height * 0.60))
    search_x0 = left + int(round(subject_width * 0.12))
    search_x1 = right - int(round(subject_width * 0.12))
    relaxed_warm = (
        alpha
        & (column_indices >= search_x0)
        & (column_indices < search_x1)
        & (row_indices >= search_y0)
        & (row_indices < bottom)
        & (red >= 40)
        & (red - green >= 8)
        & (red - blue >= 18)
        & ((channel_max - channel_min) >= 18)
    )
    relaxed_warm = cv2.morphologyEx(
        relaxed_warm.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((5, 5), dtype=np.uint8),
        iterations=1,
    ).astype(bool)

    anchor_method = "lower_center_geometry"
    anchor_pixels = int(np.count_nonzero(relaxed_warm))
    anchor_bbox: tuple[int, int, int, int] | None = None
    if anchor_pixels >= max(20, int(round(np.count_nonzero(alpha) * 0.0002))):
        warm_count, warm_labels, warm_stats, warm_centroids = (
            cv2.connectedComponentsWithStats(
                relaxed_warm.astype(np.uint8),
                connectivity=8,
            )
        )
        warm_candidates: list[tuple[int, float, int]] = []
        for label in range(1, warm_count):
            area = int(warm_stats[label, cv2.CC_STAT_AREA])
            center_x, center_y = warm_centroids[label]
            if (
                area >= 6
                and center_y >= search_y0
                and abs(center_x - subject_center_x) <= subject_width * 0.40
            ):
                warm_candidates.append(
                    (area, -abs(float(center_x) - subject_center_x), label)
                )
        if warm_candidates:
            selected_labels = [item[2] for item in warm_candidates]
            anchor = np.isin(
                warm_labels,
                np.asarray(selected_labels, dtype=warm_labels.dtype),
            )
            anchor_bbox = binary_bbox(anchor)
            anchor_pixels = int(np.count_nonzero(anchor))
            anchor_method = "relaxed_warm_food_geometry"

    minimum_prop_width = max(24, int(round(subject_width * 0.30)))
    maximum_prop_width = max(minimum_prop_width, int(round(subject_width * 0.48)))
    if anchor_bbox is not None:
        anchor_left, anchor_top, anchor_right, anchor_bottom = anchor_bbox
        anchor_width = anchor_right - anchor_left
        prop_width = min(
            maximum_prop_width,
            max(minimum_prop_width, anchor_width + int(round(subject_width * 0.08))),
        )
        prop_center_x = (anchor_left + anchor_right) / 2.0
        prop_y0 = max(
            search_y0,
            anchor_top - max(2, int(round(subject_height * 0.015))),
        )
    else:
        prop_width = minimum_prop_width
        prop_center_x = subject_center_x
        prop_y0 = top + int(round(subject_height * 0.70))
        anchor_bottom = prop_y0

    prop_x0 = max(left, int(round(prop_center_x - prop_width / 2.0)))
    prop_x1 = min(right, prop_x0 + prop_width)
    if prop_x1 - prop_x0 < prop_width:
        prop_x0 = max(left, prop_x1 - prop_width)
    # The food anchor normally sits at the rim while the neutral bowl body
    # continues to the subject baseline.  Keep the conservative horizontal
    # corridor narrow, but reserve it through the bottom so the bowl base does
    # not become part of the pet identity merely because its colour is neutral.
    prop_y1 = bottom

    fallback_zone = (
        (column_indices >= prop_x0)
        & (column_indices < prop_x1)
        & (row_indices >= prop_y0)
        & (row_indices < prop_y1)
    )
    hard_alpha = cv2.dilate(
        alpha.astype(np.uint8),
        np.ones((5, 5), dtype=np.uint8),
        iterations=1,
    ).astype(bool)
    prop_exclusion = fallback_zone & hard_alpha
    prop_exclusion = cv2.morphologyEx(
        prop_exclusion.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((5, 5), dtype=np.uint8),
        iterations=1,
    ).astype(bool)
    if not np.any(prop_exclusion):
        # A visible alpha subject always has a lower-central support region in
        # this action.  If antialiasing left the first corridor empty, expand
        # vertically but retain the same paw-safe horizontal bounds.
        prop_exclusion = (
            (column_indices >= prop_x0)
            & (column_indices < prop_x1)
            & (row_indices >= search_y0)
            & (row_indices < bottom)
            & hard_alpha
        )

    identity = hard_alpha & ~prop_exclusion
    if not np.any(identity):
        # Preserve the upper subject as identity even for an unusually small
        # or tightly cropped input.  This is a runtime-safety guard, not a
        # visual quality gate.
        identity = hard_alpha & (row_indices < prop_y0)
    if not np.any(identity):
        identity = hard_alpha.copy()

    prop_pixels = int(np.count_nonzero(prop_exclusion))
    identity_pixels = int(np.count_nonzero(identity))
    hard_alpha_pixels = int(np.count_nonzero(hard_alpha))
    colored = np.zeros((*alpha.shape, 3), dtype=np.uint8)
    colored[identity] = (0, 0, 255)
    colored[prop_exclusion] = (255, 0, 0)
    colored_image = Image.fromarray(colored, mode="RGB")
    exclusion_image = Image.fromarray(
        prop_exclusion.astype(np.uint8) * 255,
        mode="L",
    )
    metadata = {
        "policy": "eating_reference_cat_blue_bowl_red_v1",
        "split_status": "degraded_nonblocking_fallback",
        "fallback_used": True,
        "fallback_reason": str(strict_error)[:1000],
        "fallback_policy": "lower_center_prop_geometry_v1",
        "subject_bbox": [left, top, right, bottom],
        "bowl_zone_bbox": [prop_x0, prop_y0, prop_x1, prop_y1],
        "warm_seed_pixels": anchor_pixels,
        "minimum_warm_seed_pixels": None,
        "food_core_bbox": list(anchor_bbox) if anchor_bbox is not None else None,
        "food_core_pixels": anchor_pixels,
        "food_core_components": None,
        "warm_component_initially_classified_as_bowl": False,
        "warm_bowl_core_fallback_to_beige": False,
        "deep_body_bbox": [prop_x0, prop_y0, prop_x1, prop_y1],
        "deep_body_pixels": prop_pixels,
        "bowl_body_method": f"nonblocking_{anchor_method}_v1",
        "food_anchored_search_width_min_fraction": 0.30,
        "safe_x_range": [prop_x0, prop_x1],
        "safe_width_fraction": (prop_x1 - prop_x0) / float(subject_width),
        "paw_identity_y0": top + int(round(subject_height * 0.72)),
        "prop_exclusion_bbox": (
            list(binary_bbox(prop_exclusion)) if np.any(prop_exclusion) else None
        ),
        "prop_fraction": prop_pixels / float(max(1, hard_alpha_pixels)),
        "left_paw_identity_support": None,
        "right_paw_identity_support": None,
        "identity_pixels": identity_pixels,
    }
    return colored_image, exclusion_image, metadata


def make_eating_reference_identity_mask(
    normalized: Image.Image,
    *,
    allow_fallback: bool = True,
) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
    """Map cat/bowl identities without letting visual uncertainty stop a job."""

    try:
        return _make_eating_reference_identity_mask_strict(normalized)
    except ValueError as exc:
        if not allow_fallback:
            raise
        error = str(exc)
        if not error.startswith("eating_reference_prop_split_unreliable:"):
            raise
        if error.endswith((": empty_alpha", ": tiny_subject")):
            raise
        return _make_eating_reference_identity_mask_fallback(
            normalized,
            strict_error=exc,
        )


def make_drinking_reference_identity_mask(
    normalized: Image.Image,
) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
    """Keep the complete dog blue while reserving only its bowl as red.

    Drinking references often connect a pink tongue and a warm muzzle directly
    to the water surface.  The eating splitter intentionally follows connected
    food geometry and can therefore classify that contact corridor as a prop.
    This refinement tracks sustained lower-body alpha runs and selects the
    widest track close to the subject baseline as the bowl body.
    """

    rgba = np.asarray(normalized.convert("RGBA"), dtype=np.uint8)
    rgb = rgba[..., :3]
    alpha = rgba[..., 3] >= 16
    if not np.any(alpha):
        raise ValueError("drinking_reference_prop_split_unreliable: empty_alpha")

    left, top, right, bottom = binary_bbox(alpha)
    subject_width = right - left
    subject_height = bottom - top
    if subject_width < 64 or subject_height < 64:
        raise ValueError("drinking_reference_prop_split_unreliable: tiny_subject")

    search_x0 = left
    search_x1 = right
    search_y0 = top + int(round(subject_height * 0.60))

    row_runs: list[tuple[int, int, int, int]] = []
    for row in range(search_y0, bottom):
        row_alpha = alpha[row, search_x0:search_x1].astype(np.int8)
        transitions = np.diff(np.pad(row_alpha, (1, 1)))
        starts = np.flatnonzero(transitions == 1) + search_x0
        ends = np.flatnonzero(transitions == -1) + search_x0
        for run_start, run_end in zip(starts, ends, strict=True):
            row_runs.append(
                (
                    int(run_end - run_start),
                    row,
                    int(run_start),
                    int(run_end),
                )
            )
    if not row_runs:
        raise ValueError(
            "drinking_reference_prop_split_unreliable: no_bowl_row_runs"
        )

    wide_threshold = max(
        24,
        int(round(subject_width * 0.18)),
    )
    wide_runs = sorted(
        (run for run in row_runs if run[0] >= wide_threshold),
        key=lambda run: run[1],
    )
    if not wide_runs:
        raise ValueError(
            "drinking_reference_prop_split_unreliable: no_wide_bowl_rows"
        )

    groups: list[list[tuple[int, int, int, int]]] = []
    for run in wide_runs:
        best_group: list[tuple[int, int, int, int]] | None = None
        best_overlap = -1
        for group in groups:
            previous = group[-1]
            row_gap = run[1] - previous[1]
            if not 0 <= row_gap <= 2:
                continue
            overlap = max(
                0,
                min(run[3], previous[3]) - max(run[2], previous[2]),
            )
            minimum_overlap = int(round(min(run[0], previous[0]) * 0.25))
            if overlap >= minimum_overlap and overlap > best_overlap:
                best_group = group
                best_overlap = overlap
        if best_group is None:
            groups.append([run])
        else:
            best_group.append(run)
    minimum_group_rows = max(4, int(round(subject_height * 0.015)))
    sustained_groups = [
        group for group in groups if len(group) >= minimum_group_rows
    ]
    if not sustained_groups:
        raise ValueError(
            "drinking_reference_prop_split_unreliable: "
            "no_sustained_bowl_rows"
        )
    lowest_group_bottom = max(group[-1][1] for group in sustained_groups)
    bottom_tolerance = max(4, int(round(subject_height * 0.06)))
    baseline_groups = [
        group
        for group in sustained_groups
        if group[-1][1] >= lowest_group_bottom - bottom_tolerance
    ]
    bowl_group = max(
        baseline_groups,
        key=lambda group: (
            max(run[0] for run in group),
            group[-1][1],
            len(group),
        ),
    )
    baseline_span = max(12, int(round(subject_height * 0.12)))
    baseline_runs = [
        run
        for run in bowl_group
        if run[1] >= bowl_group[-1][1] - baseline_span
    ]
    widest_run = max(run[0] for run in baseline_runs)
    bowl_left = max(left, min(run[2] for run in baseline_runs) - 3)
    bowl_right = min(right, max(run[3] for run in baseline_runs) + 3)
    bowl_bottom = min(bottom, bowl_group[-1][1] + 3)
    bowl_width_fraction = (bowl_right - bowl_left) / float(subject_width)
    if not 0.18 <= bowl_width_fraction <= 0.60:
        raise ValueError(
            "drinking_reference_prop_split_unreliable: "
            f"bowl_width_fraction={bowl_width_fraction:.4f}"
        )

    red = rgb[..., 0].astype(np.int16)
    green = rgb[..., 1].astype(np.int16)
    blue = rgb[..., 2].astype(np.int16)
    channel_max = np.maximum(np.maximum(red, green), blue)
    channel_min = np.minimum(np.minimum(red, green), blue)
    luma = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    row_indices, column_indices = np.indices(alpha.shape)
    bowl_material = (
        alpha
        & (column_indices >= bowl_left)
        & (column_indices < bowl_right)
        & (row_indices >= search_y0)
        & (
            (((channel_max - channel_min) <= 48) & (luma <= 225))
            | (blue >= red + 5)
        )
    )
    material_row_threshold = max(
        6,
        int(round((bowl_right - bowl_left) * 0.60)),
    )
    material_rows = [
        row
        for row in range(search_y0, bottom)
        if int(np.count_nonzero(bowl_material[row])) >= material_row_threshold
    ]
    material_groups: list[list[int]] = []
    for row in material_rows:
        if not material_groups or row - material_groups[-1][-1] > 2:
            material_groups.append([row])
        else:
            material_groups[-1].append(row)
    sustained_material_groups = [
        group for group in material_groups if len(group) >= 3
    ]
    if sustained_material_groups:
        bowl_material_group = max(
            sustained_material_groups,
            key=lambda group: (group[-1], len(group)),
        )
        bowl_top = max(top, bowl_material_group[0] - 4)
        bowl_top_policy = "lower_neutral_or_cool_material_rows_v1"
    else:
        bowl_top = max(
            search_y0,
            bowl_group[-1][1] - max(20, int(round(subject_height * 0.20))),
        )
        bowl_top_policy = "baseline_geometry_height_fallback_v1"

    hard_alpha = cv2.dilate(
        alpha.astype(np.uint8),
        np.ones((5, 5), dtype=np.uint8),
        iterations=1,
    ).astype(bool)
    bowl_zone = (
        (column_indices >= bowl_left)
        & (column_indices < bowl_right)
        & (row_indices >= bowl_top)
        & (row_indices < bottom + 2)
    )
    prop_exclusion = hard_alpha & bowl_zone

    tongue_seed = (
        alpha
        & (column_indices >= bowl_left)
        & (column_indices < bowl_right)
        & (row_indices >= top + int(round(subject_height * 0.35)))
        & (red >= 110)
        & (red - green >= 25)
        & (blue - green >= 5)
    )
    tongue_count, tongue_labels, tongue_stats, tongue_centroids = (
        cv2.connectedComponentsWithStats(
            tongue_seed.astype(np.uint8),
            connectivity=8,
        )
    )
    tongue_candidates: list[tuple[int, int, float, int]] = []
    hard_pixels = int(np.count_nonzero(hard_alpha))
    for label in range(1, tongue_count):
        area = int(tongue_stats[label, cv2.CC_STAT_AREA])
        component_top = int(tongue_stats[label, cv2.CC_STAT_TOP])
        component_height = int(tongue_stats[label, cv2.CC_STAT_HEIGHT])
        component_bottom = component_top + component_height
        center_x = float(tongue_centroids[label][0])
        if (
            6 <= area <= max(120, int(round(hard_pixels * 0.04)))
            and bowl_left <= center_x < bowl_right
        ):
            tongue_candidates.append(
                (
                    area,
                    component_bottom,
                    -abs(center_x - (bowl_left + bowl_right - 1) / 2.0),
                    label,
                )
            )
    tongue_support = np.zeros_like(alpha)
    tongue_bbox: list[int] | None = None
    if tongue_candidates:
        _area_rank, _bottom_rank, _center_rank, tongue_label = max(
            tongue_candidates
        )
        tongue_core = tongue_labels == tongue_label
        tongue_neighbourhood = cv2.dilate(
            tongue_core.astype(np.uint8),
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        ).astype(bool)
        tongue_colour_support = (
            alpha
            & (red >= 90)
            & (red - green >= 15)
            & (blue - green >= -5)
        )
        tongue_support = tongue_neighbourhood & tongue_colour_support
        tongue_bbox = list(binary_bbox(tongue_support))
        prop_exclusion &= ~tongue_support

    identity = hard_alpha & ~prop_exclusion
    prop_pixels = int(np.count_nonzero(prop_exclusion))
    identity_pixels = int(np.count_nonzero(identity))
    if prop_pixels < 64 or identity_pixels < 500:
        raise ValueError(
            "drinking_reference_prop_split_unreliable: "
            f"invalid_pixel_counts=({identity_pixels},{prop_pixels})"
        )
    prop_fraction = prop_pixels / float(max(1, hard_pixels))
    if not 0.015 <= prop_fraction <= 0.24:
        raise ValueError(
            "drinking_reference_prop_split_unreliable: "
            f"prop_fraction={prop_fraction:.4f}"
        )

    colored = np.zeros((*alpha.shape, 3), dtype=np.uint8)
    colored[identity] = (0, 0, 255)
    colored[prop_exclusion] = (255, 0, 0)
    metadata = {
        "policy": "drinking_reference_dog_blue_bowl_red_v2",
        "split_status": "strict_geometry_refinement",
        "fallback_used": False,
        "subject_bbox": [left, top, right, bottom],
        "bowl_search_bbox": [search_x0, search_y0, search_x1, bottom],
        "bowl_zone_bbox": [bowl_left, bowl_top, bowl_right, bottom],
        "bowl_detected_bottom": bowl_bottom,
        "bowl_top_policy": bowl_top_policy,
        "bowl_material_row_threshold": material_row_threshold,
        "bowl_widest_row_px": widest_run,
        "bowl_wide_row_threshold_px": wide_threshold,
        "bowl_sustained_row_count": len(bowl_group),
        "bowl_width_fraction": bowl_width_fraction,
        "prop_exclusion_bbox": list(binary_bbox(prop_exclusion)),
        "prop_fraction": prop_fraction,
        "identity_pixels": identity_pixels,
        "tongue_reclassified_as_identity": bool(np.any(tongue_support)),
        "tongue_bbox": tongue_bbox,
    }
    return (
        Image.fromarray(colored, mode="RGB"),
        Image.fromarray(prop_exclusion.astype(np.uint8) * 255, mode="L"),
        metadata,
    )


def make_reference_assets(
    reference_path: Path,
    canvas_size: tuple[int, int],
    separate_eating_prop: bool = False,
    separate_drinking_prop: bool = False,
    separate_toileting_prop: bool = False,
    contain_fraction: float = 0.84,
) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
    if sum(
        (
            separate_eating_prop,
            separate_drinking_prop,
            separate_toileting_prop,
        )
    ) > 1:
        raise ValueError("Only one reference prop split may be requested")
    with Image.open(reference_path) as source:
        reference = source.convert("RGBA")
    reference_extraction = "existing_alpha"
    alpha_array = np.asarray(reference.getchannel("A"))
    if bool(np.all(alpha_array == 255)):
        rgb_array = np.asarray(reference.convert("RGB")).copy()
        corners = np.asarray(
            [
                rgb_array[0, 0],
                rgb_array[0, -1],
                rgb_array[-1, 0],
                rgb_array[-1, -1],
            ],
            dtype=np.float32,
        )
        green_dominant = bool(
            np.all(corners[:, 1] >= corners[:, 0] * 1.12)
            and np.all(corners[:, 1] >= corners[:, 2] * 1.12)
            and float(np.mean(corners[:, 1])) >= 80.0
        )
        if not green_dominant:
            raise ValueError(
                "Opaque SCAIL reference must have a green-screen background "
                f"or an existing alpha channel: {reference_path}"
            )
        extracted_rgb, extracted_alpha = _subject_mask(rgb_array)
        foreground_fraction = float(
            np.count_nonzero(extracted_alpha >= 16)
        ) / float(extracted_alpha.size)
        if not 0.02 <= foreground_fraction <= 0.85:
            raise ValueError(
                "Green-screen reference extraction produced an invalid "
                f"foreground fraction {foreground_fraction:.4f}: "
                f"{reference_path}"
            )
        reference = Image.fromarray(
            np.dstack((extracted_rgb, extracted_alpha)).astype(np.uint8),
            mode="RGBA",
        )
        reference_extraction = "chroma_key"
    alpha_bbox = reference.getchannel("A").getbbox()
    if not alpha_bbox:
        raise ValueError(f"Reference has no visible alpha subject: {reference_path}")
    cropped = reference.crop(expand_bbox(alpha_bbox, reference.size, margin_fraction=0.04))
    if not 0.50 <= contain_fraction <= 0.95:
        raise ValueError("reference contain fraction must be between 0.50 and 0.95")
    normalized = contain_rgba(cropped, canvas_size, contain_fraction)
    alpha = normalized.getchannel("A")
    hard_alpha = alpha.point(lambda value: 255 if value >= 16 else 0)
    hard_alpha = hard_alpha.filter(ImageFilter.MaxFilter(5))

    reference_prop_metadata: dict[str, Any] | None = None
    if separate_eating_prop:
        colored_mask, _prop_exclusion, reference_prop_metadata = (
            make_eating_reference_identity_mask(normalized)
        )
    elif separate_drinking_prop:
        colored_mask, _prop_exclusion, reference_prop_metadata = (
            make_drinking_reference_identity_mask(normalized)
        )
    elif separate_toileting_prop:
        toileting_exclusions, reference_prop_metadata = (
            make_toileting_prop_exclusion_masks([normalized])
        )
        prop_exclusion = toileting_exclusions[0]
        normalized = normalize_toileting_prop_rgb(
            normalized,
            prop_exclusion,
            reference_prop_metadata,
        )
        normalized_alpha = normalized.getchannel("A")
        hard_alpha = normalized_alpha.point(
            lambda value: 255 if value >= 16 else 0
        )
        hard_alpha = hard_alpha.filter(ImageFilter.MaxFilter(5))
        prop = np.asarray(prop_exclusion, dtype=np.uint8) >= 128
        identity = (np.asarray(hard_alpha, dtype=np.uint8) >= 128) & ~prop
        colored_pixels = np.zeros((*identity.shape, 3), dtype=np.uint8)
        colored_pixels[identity] = (0, 0, 255)
        colored_pixels[prop] = (255, 0, 0)
        colored_mask = Image.fromarray(colored_pixels, mode="RGB")
    else:
        # SCAIL reference-mask background is black, same identity is pure blue.
        colored_mask = Image.new("RGB", canvas_size, (0, 0, 0))
        colored_mask.paste(
            Image.new("RGB", canvas_size, (0, 0, 255)),
            mask=hard_alpha,
        )
    identity_pixels = np.all(
        np.asarray(colored_mask, dtype=np.uint8) == (0, 0, 255),
        axis=2,
    )
    prop_pixels = np.all(
        np.asarray(colored_mask, dtype=np.uint8) == (255, 0, 0),
        axis=2,
    )
    metadata = {
        "reference_extraction": reference_extraction,
        "reference_contain_fraction": round(float(contain_fraction), 6),
        "reference_alpha_bbox": list(alpha_bbox),
        "reference_subject_fraction": (
            float(np.count_nonzero(np.asarray(hard_alpha)))
            / float(canvas_size[0] * canvas_size[1])
        ),
        "reference_identity_fraction": (
            float(np.count_nonzero(identity_pixels))
            / float(canvas_size[0] * canvas_size[1])
        ),
        "reference_prop_fraction": (
            float(np.count_nonzero(prop_pixels))
            / float(canvas_size[0] * canvas_size[1])
        ),
        "scene_prop_separated": (
            separate_eating_prop
            or separate_drinking_prop
            or separate_toileting_prop
        ),
        "scene_prop_metadata": reference_prop_metadata,
    }
    return normalized, colored_mask, metadata


def _run_checked(command: list[str], stage: str, timeout: int = 900) -> None:
    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if process.returncode:
        detail = (process.stderr or process.stdout or "unknown error").strip()
        raise RuntimeError(f"{stage} failed: {detail[-4000:]}")


def _validate_transparent_gif(
    path: Path,
    *,
    size: tuple[int, int],
    frame_count: int,
    duration_ms: int,
    expected_foreground_masks: list[np.ndarray] | None = None,
) -> dict[str, Any]:
    with Image.open(path) as image:
        loop_count = int(image.info.get("loop", -1))
        if image.size != size:
            raise RuntimeError(f"GIF size is {image.size}, expected {size}")
        if image.n_frames != frame_count:
            raise RuntimeError(
                f"GIF has {image.n_frames} frames, expected {frame_count}"
            )
        if (
            expected_foreground_masks is not None
            and len(expected_foreground_masks) != frame_count
        ):
            raise RuntimeError(
                "Expected foreground mask count does not match GIF frames: "
                f"{len(expected_foreground_masks)} != {frame_count}"
            )
        durations: list[int] = []
        corner_alpha: list[int] = []
        border_alpha_max = 0
        encoded_foreground_retentions: list[float] = []
        encoded_foreground_ious: list[float] = []
        alpha_has_both: list[bool] = []
        edge_metrics: list[dict[str, float | int]] = []
        for index in range(image.n_frames):
            image.seek(index)
            durations.append(int(image.info.get("duration", 0)))
            rgba = np.asarray(image.convert("RGBA"))
            alpha = rgba[:, :, 3]
            corner_alpha.extend(
                int(alpha[y, x])
                for x, y in (
                    (0, 0),
                    (size[0] - 1, 0),
                    (0, size[1] - 1),
                    (size[0] - 1, size[1] - 1),
                )
            )
            border_alpha_max = max(
                border_alpha_max,
                int(alpha[:2, :].max()),
                int(alpha[-2:, :].max()),
                int(alpha[:, :2].max()),
                int(alpha[:, -2:].max()),
            )
            encoded_foreground = alpha >= 128
            alpha_has_both.append(
                bool(np.any(alpha == 0) and np.any(alpha == 255))
            )
            edge_metrics.append(_binary_alpha_edge_metrics(alpha))
            if expected_foreground_masks is not None:
                expected_foreground = expected_foreground_masks[index]
                expected_pixels = max(
                    1,
                    int(np.count_nonzero(expected_foreground)),
                )
                intersection = int(
                    np.count_nonzero(
                        encoded_foreground & expected_foreground
                    )
                )
                union = max(
                    1,
                    int(
                        np.count_nonzero(
                            encoded_foreground | expected_foreground
                        )
                    ),
                )
                encoded_foreground_retentions.append(
                    float(intersection) / float(expected_pixels)
                )
                encoded_foreground_ious.append(
                    float(intersection) / float(union)
                )
    actual_duration = sum(durations)
    if actual_duration != duration_ms:
        raise RuntimeError(
            f"GIF duration is {actual_duration}ms, expected {duration_ms}ms"
        )
    if loop_count != 0:
        raise RuntimeError(
            f"GIF loop count is {loop_count}, expected infinite loop (0)"
        )
    if any(corner_alpha):
        raise RuntimeError("GIF has an opaque canvas corner")
    if border_alpha_max:
        raise RuntimeError("GIF has an opaque pixel in its 2px canvas border")
    if not all(alpha_has_both):
        raise RuntimeError(
            "At least one GIF frame lacks transparent or opaque pixels"
        )
    encoded_foreground_retention_min = min(
        encoded_foreground_retentions,
        default=1.0,
    )
    encoded_foreground_iou_min = min(
        encoded_foreground_ious,
        default=1.0,
    )
    if encoded_foreground_retention_min < 0.995:
        raise RuntimeError(
            "GIF encoding removed foreground pixels: minimum retention="
            f"{encoded_foreground_retention_min:.4f}"
        )
    if encoded_foreground_iou_min < 0.995:
        raise RuntimeError(
            "GIF encoding changed the foreground matte: minimum IoU="
            f"{encoded_foreground_iou_min:.4f}"
        )
    return {
        "size": list(size),
        "frames": frame_count,
        "duration_ms": actual_duration,
        "frame_durations_ms": sorted(set(durations)),
        "all_corners_transparent": True,
        "full_2px_border_transparent": True,
        "all_frames_have_transparent_and_opaque_pixels": True,
        "encoded_foreground_retention_min": (
            encoded_foreground_retention_min
        ),
        "encoded_foreground_iou_min": encoded_foreground_iou_min,
        "edge_quality": {
            "tiny_foreground_components_max": max(
                int(item["tiny_foreground_components"])
                for item in edge_metrics
            ),
            "sparse_boundary_fraction_max": max(
                float(item["sparse_boundary_fraction"])
                for item in edge_metrics
            ),
            "checkerboard_fraction_max": max(
                float(item["checkerboard_fraction"])
                for item in edge_metrics
            ),
        },
        "loop": loop_count,
        "bytes": path.stat().st_size,
    }


def _validate_white_background_mp4(
    path: Path,
    *,
    size: tuple[int, int],
    frame_count: int,
) -> dict[str, Any]:
    decoded = 0
    corner_channel_min = 255
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            rgb = frame.to_ndarray(format="rgb24")
            if (rgb.shape[1], rgb.shape[0]) != size:
                raise RuntimeError(
                    f"White-background MP4 frame size is "
                    f"{rgb.shape[1]}x{rgb.shape[0]}, expected {size[0]}x{size[1]}"
                )
            corners = np.asarray(
                [rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1]],
                dtype=np.uint8,
            )
            corner_channel_min = min(
                corner_channel_min,
                int(corners.min()),
            )
            decoded += 1
    if decoded != frame_count:
        raise RuntimeError(
            f"White-background MP4 decoded {decoded} frames; expected {frame_count}"
        )
    if corner_channel_min < 245:
        raise RuntimeError(
            "White-background MP4 has a non-white canvas corner: minimum channel="
            f"{corner_channel_min}"
        )
    return {
        "size": list(size),
        "frames": decoded,
        "all_corner_channels_min": corner_channel_min,
        "bytes": path.stat().st_size,
    }


def _select_pose_guided_semantic_components(
    labels: np.ndarray,
    stats: np.ndarray,
    guidance: np.ndarray,
    semantic_core: np.ndarray,
    *,
    minimum_area: int = 64,
    minimum_guidance_overlap: int = 32,
    minimum_guidance_fraction: float = 0.20,
    minimum_semantic_core_pixels: int = 64,
    minimum_semantic_core_fraction: float = 0.20,
) -> tuple[list[int], list[int]]:
    """Keep guided components plus high-confidence fragments of that subject.

    The alpha passed to this helper has already been restricted to components
    accepted against the pose guide. A later cleanup can cut a thin soft-alpha
    bridge through a real paw, ear, or tail and split that accepted subject into
    a second component. Preserve such a fragment only when it still contains a
    substantial high-confidence semantic core; detached low-confidence noise
    remains excluded.
    """
    candidates: list[tuple[int, int, int, int]] = []
    guided_labels: list[int] = []
    semantic_fragment_labels: list[int] = []
    for label in range(1, int(stats.shape[0])):
        component = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        overlap = int(np.count_nonzero(component & guidance))
        core_pixels = int(np.count_nonzero(component & semantic_core))
        candidates.append((overlap, area, core_pixels, label))
        if (
            area >= minimum_area
            and overlap >= minimum_guidance_overlap
            and overlap / float(max(1, area))
            >= minimum_guidance_fraction
        ):
            guided_labels.append(label)
        if (
            area >= minimum_area
            and core_pixels >= minimum_semantic_core_pixels
            and core_pixels / float(max(1, area))
            >= minimum_semantic_core_fraction
        ):
            semantic_fragment_labels.append(label)

    selected_labels = sorted(
        set(guided_labels) | set(semantic_fragment_labels)
    )
    if not selected_labels:
        selected_labels = [max(candidates)[3]]
    recovered_labels = sorted(set(selected_labels) - set(guided_labels))
    return selected_labels, recovered_labels


def _rgb_luma(rgb: np.ndarray) -> np.ndarray:
    pixels = np.asarray(rgb, dtype=np.float32)
    return (
        pixels[:, :, 0] * 0.2126
        + pixels[:, :, 1] * 0.7152
        + pixels[:, :, 2] * 0.0722
    )


def _inward_edge_reconstruction(
    rgb: np.ndarray,
    alpha: np.ndarray,
    *,
    band_width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Propagate trustworthy interior color through the complete soft matte."""
    support = alpha > 0
    stable_foreground = alpha >= 96
    width = max(1, int(band_width))
    core = cv2.erode(
        stable_foreground.astype(np.uint8),
        np.ones((width * 2 + 1, width * 2 + 1), dtype=np.uint8),
        iterations=1,
    ).astype(bool)
    if not np.any(core):
        return rgb.astype(np.float32), core, core

    reconstructed = rgb.astype(np.float32).copy()
    known = core.copy()
    neighbor_kernel = np.ones((3, 3), dtype=np.float32)
    # Reaching the outside of the soft matte requires crossing the eroded inner
    # band and then the semi-transparent fringe. The old implementation stopped
    # at alpha 64 and therefore left the darkest fringe untouched.
    for _ in range(width * 2 + 4):
        known_float = known.astype(np.float32)
        neighbor_count = cv2.filter2D(
            known_float,
            ddepth=-1,
            kernel=neighbor_kernel,
            borderType=cv2.BORDER_CONSTANT,
        )
        frontier = (
            (cv2.dilate(known.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
            & support
            & ~known
            & (neighbor_count > 0)
        )
        if not np.any(frontier):
            break
        for channel in range(3):
            neighbor_sum = cv2.filter2D(
                reconstructed[:, :, channel] * known_float,
                ddepth=-1,
                kernel=neighbor_kernel,
                borderType=cv2.BORDER_CONSTANT,
            )
            reconstructed[:, :, channel][frontier] = (
                neighbor_sum[frontier] / neighbor_count[frontier]
            )
        known |= frontier
    return reconstructed, core, known


def _continuous_dark_edge_candidates(
    rgb: np.ndarray,
    alpha: np.ndarray,
    reconstructed: np.ndarray,
    core: np.ndarray,
    known: np.ndarray,
    *,
    minimum_reference_luma: float,
    minimum_opaque_luma_delta: float,
    minimum_component_pixels: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    band = known & ~core
    band_pixels = int(np.count_nonzero(band))
    if not band_pixels:
        return np.zeros(alpha.shape, dtype=bool), {
            "band_pixels": 0,
            "raw_candidate_pixels": 0,
            "candidate_pixels": 0,
            "candidate_fraction": 0.0,
            "low_alpha_candidate_pixels": 0,
        }

    source_luma = _rgb_luma(rgb)
    reference_luma = _rgb_luma(reconstructed)
    luma_delta = reference_luma - source_luma
    raw_candidates = (
        band
        & (reference_luma >= float(minimum_reference_luma))
        & (
            ((alpha < 96) & (luma_delta >= 8.0))
            | (
                (alpha >= 96)
                & (luma_delta >= float(minimum_opaque_luma_delta))
            )
        )
    )
    # A real coat marking normally continues into the interior. Generated
    # silhouette strokes instead form long boundary components. Select only
    # continuous components so a tiny authentic ear or tail marking survives
    # even when another part of the frame contains an outline.
    joined = cv2.morphologyEx(
        raw_candidates.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    )
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        joined,
        connectivity=8,
    )
    candidates = np.zeros(alpha.shape, dtype=bool)
    minimum_component_area = max(4, int(minimum_component_pixels))
    for label in range(1, component_count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= minimum_component_area:
            candidates |= raw_candidates & (labels == label)

    candidate_pixels = int(np.count_nonzero(candidates))
    return candidates, {
        "band_pixels": band_pixels,
        "raw_candidate_pixels": int(np.count_nonzero(raw_candidates)),
        "candidate_pixels": candidate_pixels,
        "candidate_fraction": float(candidate_pixels) / float(band_pixels),
        "low_alpha_candidate_pixels": int(
            np.count_nonzero(candidates & (alpha < 64))
        ),
    }


def _remove_eating_contact_shadow(
    rgb: np.ndarray,
    alpha: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Remove only a thin neutral shadow attached below the detected bowl."""
    unchanged = {
        "policy": "eating_bowl_contact_shadow_v1",
        "applied": False,
        "removed_pixels": 0,
        "component_count": 0,
    }
    try:
        rgba = Image.fromarray(
            np.dstack((rgb, alpha)).astype(np.uint8),
            mode="RGBA",
        )
        _colored, prop_exclusion, prop_metadata = (
            make_eating_reference_identity_mask(rgba)
        )
    except (RuntimeError, ValueError) as exc:
        return alpha, {
            **unchanged,
            "status": "prop_split_unavailable",
            "error": f"{type(exc).__name__}: {str(exc)[:1000]}",
        }

    prop = np.asarray(prop_exclusion, dtype=np.uint8) >= 128
    rows, columns = np.where(prop)
    if not rows.size:
        return alpha, {**unchanged, "status": "empty_prop"}
    x0, x1 = int(columns.min()), int(columns.max()) + 1
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    prop_width = x1 - x0
    prop_height = y1 - y0
    luma = _rgb_luma(rgb)
    chroma = (
        rgb.max(axis=2).astype(np.int16)
        - rgb.min(axis=2).astype(np.int16)
    )
    dark_neutral = (
        (alpha >= 24)
        & (luma <= 68.0)
        & (chroma <= 44)
    )
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        dark_neutral.astype(np.uint8),
        connectivity=8,
    )
    row_grid = np.indices(alpha.shape)[0]
    lower_prop = prop & (row_grid >= y0 + int(round(prop_height * 0.52)))
    removal = np.zeros(alpha.shape, dtype=bool)
    selected_components: list[dict[str, int | float]] = []
    for label in range(1, component_count):
        x, y, width, height, area = (
            int(value) for value in stats[label]
        )
        component = labels == label
        overlap = int(np.count_nonzero(component & lower_prop))
        overlap_fraction = float(overlap) / float(max(1, area))
        aspect_ratio = float(width) / float(max(1, height))
        if (
            area >= 12
            and overlap_fraction >= 0.18
            and y + height
            >= y1 - max(4, int(round(prop_height * 0.12)))
            and width >= max(12, int(round(prop_width * 0.08)))
            and aspect_ratio >= 1.7
        ):
            removal |= component
            selected_components.append(
                {
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "pixels": area,
                    "prop_overlap_pixels": overlap,
                    "aspect_ratio": round(aspect_ratio, 6),
                }
            )
    if not selected_components:
        return alpha, {
            **unchanged,
            "status": "no_contact_shadow",
            "prop_bbox": [x0, y0, x1, y1],
            "prop_split_status": prop_metadata.get("split_status"),
        }

    # Include the one-pixel antialiased rim around the dark component, but do
    # not erase the light ceramic bowl body above it.
    removal = (
        cv2.dilate(
            removal.astype(np.uint8),
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        ).astype(bool)
        & (alpha > 0)
        & (luma < 104.0)
    )
    cleaned_alpha = alpha.copy()
    cleaned_alpha[removal] = 0
    return cleaned_alpha, {
        "policy": "eating_bowl_contact_shadow_v1",
        "applied": True,
        "removed_pixels": int(np.count_nonzero(removal)),
        "component_count": len(selected_components),
        "components": selected_components,
        "prop_bbox": [x0, y0, x1, y1],
        "prop_split_status": prop_metadata.get("split_status"),
    }


def clean_transparent_edge_halo(
    rgb: np.ndarray,
    alpha: np.ndarray,
    *,
    action: str | None = None,
    band_width: int = 9,
    minimum_opaque_luma_delta: float = 24.0,
    minimum_reference_luma: float = 64.0,
    minimum_candidate_pixels: int = 64,
    minimum_candidate_fraction: float = 0.015,
    minimum_component_pixels: int = 24,
    maximum_residual_candidate_fraction: float = 0.12,
    maximum_residual_candidate_pixel_fraction: float = 0.008,
    adjust_alpha: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Remove painted dark contours without erasing authentic dark pets.

    RGB is reconstructed from foreground-side coat pixels across every nonzero
    alpha sample. For a confirmed continuous halo, the contaminated matte is
    also pulled inward by at most two pixels and lightly softened. This avoids
    turning a black outline into an equally artificial white outline.
    """
    rgb = np.asarray(rgb, dtype=np.uint8)
    alpha = np.asarray(alpha, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Transparent edge RGB must have shape HxWx3.")
    if alpha.ndim != 2 or alpha.shape != rgb.shape[:2]:
        raise ValueError("Transparent edge alpha must match RGB dimensions.")

    contact_shadow = {
        "policy": "eating_bowl_contact_shadow_v1",
        "applied": False,
        "removed_pixels": 0,
        "component_count": 0,
        "status": "not_eating",
    }
    working_alpha = alpha.copy()
    if str(action or "").strip().lower() == "eating":
        working_alpha, contact_shadow = _remove_eating_contact_shadow(
            rgb,
            working_alpha,
        )

    reconstructed, core, known = _inward_edge_reconstruction(
        rgb,
        working_alpha,
        band_width=band_width,
    )
    if int(np.count_nonzero(core)) < int(minimum_candidate_pixels):
        cleaned_rgb = rgb.copy()
        cleaned_rgb[(working_alpha == 0) & (alpha > 0)] = 0
        return cleaned_rgb, working_alpha, {
            "policy": "local_inward_color_alpha_dark_halo_v2",
            "applied": bool(contact_shadow.get("applied")),
            "rgb_applied": False,
            "alpha_applied": bool(contact_shadow.get("applied")),
            "acceptable": True,
            "penalty": 0.0,
            "band_pixels": 0,
            "candidate_pixels": 0,
            "replaced_pixels": 0,
            "candidate_fraction": 0.0,
            "low_alpha_candidate_pixels": 0,
            "alpha_adjusted_pixels": int(
                np.count_nonzero(working_alpha != alpha)
            ),
            "contact_shadow": contact_shadow,
        }

    candidates, before = _continuous_dark_edge_candidates(
        rgb,
        working_alpha,
        reconstructed,
        core,
        known,
        minimum_reference_luma=minimum_reference_luma,
        minimum_opaque_luma_delta=minimum_opaque_luma_delta,
        minimum_component_pixels=minimum_component_pixels,
    )
    candidate_pixels = int(before["candidate_pixels"])
    candidate_fraction = float(before["candidate_fraction"])
    rgb_applied = bool(
        candidate_pixels >= int(minimum_candidate_pixels)
        and candidate_fraction >= float(minimum_candidate_fraction)
    )
    cleaned_rgb = rgb.copy()
    cleaned_alpha = working_alpha.copy()
    if rgb_applied:
        blended = np.clip(
            reconstructed * 0.98 + rgb.astype(np.float32) * 0.02,
            0,
            255,
        ).astype(np.uint8)
        cleaned_rgb[candidates] = blended[candidates]
        if adjust_alpha:
            eroded_alpha = cv2.erode(
                working_alpha,
                np.ones((5, 5), dtype=np.uint8),
                iterations=1,
            )
            cleaned_alpha[candidates] = np.minimum(
                cleaned_alpha[candidates],
                eroded_alpha[candidates],
            )
            cleaned_alpha[candidates & (working_alpha < 96)] = 0
            cleaned_alpha = cv2.GaussianBlur(
                cleaned_alpha,
                (0, 0),
                sigmaX=0.55,
                sigmaY=0.55,
            )
            cleaned_alpha[cleaned_alpha < 3] = 0
            cleaned_alpha[cleaned_alpha > 252] = 255

    cleaned_rgb[(cleaned_alpha == 0) & (alpha > 0)] = 0
    after_reconstruction, after_core, after_known = _inward_edge_reconstruction(
        cleaned_rgb,
        cleaned_alpha,
        band_width=band_width,
    )
    _residual, after = _continuous_dark_edge_candidates(
        cleaned_rgb,
        cleaned_alpha,
        after_reconstruction,
        after_core,
        after_known,
        minimum_reference_luma=minimum_reference_luma,
        minimum_opaque_luma_delta=minimum_opaque_luma_delta,
        minimum_component_pixels=minimum_component_pixels,
    )
    residual_pixels = int(after["candidate_pixels"])
    residual_fraction = float(after["candidate_fraction"])
    maximum_residual_pixels = max(
        int(minimum_candidate_pixels),
        int(round(alpha.size * maximum_residual_candidate_pixel_fraction)),
    )
    acceptable = bool(
        residual_pixels <= maximum_residual_pixels
        and residual_fraction <= float(maximum_residual_candidate_fraction)
    )
    alpha_adjusted_pixels = int(np.count_nonzero(cleaned_alpha != alpha))
    return cleaned_rgb, cleaned_alpha, {
        "policy": "local_inward_color_alpha_dark_halo_v2",
        "applied": bool(rgb_applied or contact_shadow.get("applied")),
        "rgb_applied": rgb_applied,
        "alpha_applied": bool(alpha_adjusted_pixels),
        "acceptable": acceptable,
        "penalty": round(
            min(500.0, residual_fraction * 500.0 + residual_pixels / 64.0),
            6,
        ),
        "band_pixels": int(before["band_pixels"]),
        "raw_candidate_pixels": int(before["raw_candidate_pixels"]),
        "candidate_pixels": candidate_pixels,
        "replaced_pixels": candidate_pixels if rgb_applied else 0,
        "candidate_fraction": candidate_fraction,
        "low_alpha_candidate_pixels": int(
            before["low_alpha_candidate_pixels"]
        ),
        "alpha_adjusted_pixels": alpha_adjusted_pixels,
        "contact_shadow": contact_shadow,
        "after": {
            "acceptable": acceptable,
            "penalty": round(
                min(
                    500.0,
                    residual_fraction * 500.0 + residual_pixels / 64.0,
                ),
                6,
            ),
            "band_pixels": int(after["band_pixels"]),
            "candidate_pixels": residual_pixels,
            "candidate_fraction": residual_fraction,
            "low_alpha_candidate_pixels": int(
                after["low_alpha_candidate_pixels"]
            ),
            "maximum_candidate_pixels": maximum_residual_pixels,
            "maximum_candidate_fraction": float(
                maximum_residual_candidate_fraction
            ),
        },
    }


def repair_guided_alpha_interior_holes(
    rgb: np.ndarray,
    alpha: np.ndarray,
    guidance: np.ndarray | None,
    *,
    minimum_hole_pixels: int = 8,
    maximum_hole_fraction: float = 0.015,
    minimum_guidance_core_overlap_fraction: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fill only enclosed matte holes supported by the pose-guidance core.

    Exterior background, including the gap between planted legs or between a
    tail and torso, remains connected to a canvas border and is never eligible.
    Guidance is eroded before use so a loose driver silhouette cannot grow the
    replacement animal's outer boundary.
    """

    rgb = np.asarray(rgb, dtype=np.uint8)
    alpha = np.asarray(alpha, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Guided alpha repair RGB must have shape HxWx3.")
    if alpha.ndim != 2 or alpha.shape != rgb.shape[:2]:
        raise ValueError("Guided alpha repair alpha must match RGB dimensions.")
    if guidance is None:
        return rgb.copy(), alpha.copy(), {
            "policy": "guided_enclosed_alpha_interior_repair_v1",
            "applied": False,
            "status": "guidance_unavailable",
            "repaired_pixels": 0,
            "repaired_components": 0,
            "eligible_components": [],
        }
    guidance = np.asarray(guidance, dtype=bool)
    if guidance.shape != alpha.shape:
        raise ValueError("Guided alpha repair guidance size changed.")
    guidance_u8 = guidance.astype(np.uint8)
    guidance_core = cv2.erode(
        guidance_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    ).astype(bool)
    if int(np.count_nonzero(guidance_core)) < 64:
        guidance_core = guidance

    transparent = (alpha < 64).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        transparent,
        connectivity=8,
    )
    border_labels = set(
        int(value)
        for value in np.unique(
            np.concatenate(
                (
                    labels[0, :],
                    labels[-1, :],
                    labels[:, 0],
                    labels[:, -1],
                )
            )
        )
    )
    maximum_hole_pixels = max(
        int(minimum_hole_pixels),
        int(round(alpha.size * float(maximum_hole_fraction))),
    )
    repair_mask = np.zeros(alpha.shape, dtype=bool)
    eligible_components: list[dict[str, Any]] = []
    for label in range(1, count):
        if label in border_labels:
            continue
        area = int(stats[label, cv2.CC_STAT_AREA])
        if not int(minimum_hole_pixels) <= area <= maximum_hole_pixels:
            continue
        component = labels == label
        guidance_overlap = int(
            np.count_nonzero(component & guidance_core)
        )
        guidance_overlap_fraction = guidance_overlap / float(max(1, area))
        if (
            guidance_overlap < max(4, int(round(area * 0.05)))
            or guidance_overlap_fraction
            < float(minimum_guidance_core_overlap_fraction)
        ):
            continue
        repair_mask |= component
        eligible_components.append(
            {
                "x": int(stats[label, cv2.CC_STAT_LEFT]),
                "y": int(stats[label, cv2.CC_STAT_TOP]),
                "width": int(stats[label, cv2.CC_STAT_WIDTH]),
                "height": int(stats[label, cv2.CC_STAT_HEIGHT]),
                "pixels": area,
                "guidance_core_overlap_pixels": guidance_overlap,
                "guidance_core_overlap_fraction": round(
                    guidance_overlap_fraction,
                    6,
                ),
            }
        )

    repaired_pixels = int(np.count_nonzero(repair_mask))
    if not repaired_pixels:
        return rgb.copy(), alpha.copy(), {
            "policy": "guided_enclosed_alpha_interior_repair_v1",
            "applied": False,
            "status": "no_supported_enclosed_hole",
            "repaired_pixels": 0,
            "repaired_components": 0,
            "eligible_components": [],
            "maximum_hole_pixels": maximum_hole_pixels,
            "guidance_core_pixels": int(np.count_nonzero(guidance_core)),
        }

    repaired_rgb = cv2.inpaint(
        rgb,
        repair_mask.astype(np.uint8) * 255,
        3.0,
        cv2.INPAINT_TELEA,
    )
    repaired_alpha = alpha.copy()
    component_ring = (
        cv2.dilate(
            repair_mask.astype(np.uint8),
            np.ones((5, 5), dtype=np.uint8),
            iterations=1,
        ).astype(bool)
        & ~repair_mask
        & (alpha >= 64)
    )
    ring_alpha = alpha[component_ring]
    fill_alpha = int(
        max(
            160,
            round(float(np.percentile(ring_alpha, 25)))
            if ring_alpha.size
            else 255,
        )
    )
    repaired_alpha[repair_mask] = np.maximum(
        repaired_alpha[repair_mask],
        fill_alpha,
    )
    return repaired_rgb, repaired_alpha, {
        "policy": "guided_enclosed_alpha_interior_repair_v1",
        "applied": True,
        "status": "repaired",
        "repaired_pixels": repaired_pixels,
        "repaired_components": len(eligible_components),
        "eligible_components": eligible_components,
        "fill_alpha": fill_alpha,
        "maximum_hole_pixels": maximum_hole_pixels,
        "guidance_core_pixels": int(np.count_nonzero(guidance_core)),
        "exterior_background_preserved": True,
    }


def select_safe_transparent_edge_cleanup(
    original_rgb: np.ndarray,
    original_alpha: np.ndarray,
    cleaned_rgb: np.ndarray,
    cleaned_alpha: np.ndarray,
    metrics: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Rollback a halo cleanup whose own residual audit is unacceptable."""

    selected_metrics = dict(metrics)
    if bool(metrics.get("acceptable")):
        selected_metrics["rolled_back"] = False
        selected_metrics["published_result"] = "cleaned"
        return cleaned_rgb, cleaned_alpha, selected_metrics
    selected_metrics.update(
        {
            "rolled_back": True,
            "published_result": "pre_cleanup_matte",
            "attempted_alpha_adjusted_pixels": int(
                metrics.get("alpha_adjusted_pixels", 0)
            ),
            "alpha_adjusted_pixels": 0,
            "published_alpha_adjusted_pixels": 0,
            "completion_blocked": False,
        }
    )
    return original_rgb.copy(), original_alpha.copy(), selected_metrics


def _remove_dark_silhouette_halo(
    rgb: np.ndarray,
    alpha: np.ndarray,
    *,
    band_width: int = 9,
    minimum_luma_delta: float = 24.0,
    minimum_reference_luma: float = 64.0,
    minimum_candidate_pixels: int = 64,
    minimum_candidate_fraction: float = 0.015,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Backward-compatible RGB-only wrapper for tests and diagnostic callers."""
    cleaned_rgb, _cleaned_alpha, metrics = clean_transparent_edge_halo(
        rgb,
        alpha,
        band_width=band_width,
        minimum_opaque_luma_delta=minimum_luma_delta,
        minimum_reference_luma=minimum_reference_luma,
        minimum_candidate_pixels=minimum_candidate_pixels,
        minimum_candidate_fraction=minimum_candidate_fraction,
        adjust_alpha=False,
    )
    return cleaned_rgb, metrics


def grooming_closed_loop_source_indices(frame_count: int) -> list[int]:
    """Keep the validated grooming half and return to its contact frame."""
    if frame_count < 3 or frame_count % 2 == 0:
        raise ValueError(
            "Grooming closed-loop publication requires an odd source frame "
            f"count of at least 3, got {frame_count}."
        )
    pivot = frame_count // 2
    indices = list(range(pivot, frame_count))
    indices.extend(range(frame_count - 2, pivot - 1, -1))
    if len(indices) != frame_count or indices[0] != indices[-1]:
        raise RuntimeError("Grooming closed-loop index construction failed.")
    return indices


def _transparent_frame_transition_score(
    first: tuple[np.ndarray, np.ndarray],
    second: tuple[np.ndarray, np.ndarray],
) -> float:
    first_rgb, first_alpha = first
    second_rgb, second_alpha = second
    first_rgba = np.dstack((first_rgb, first_alpha)).astype(np.float32)
    second_rgba = np.dstack((second_rgb, second_alpha)).astype(np.float32)
    return float(np.mean(np.abs(first_rgba - second_rgba)))


def transparent_animation_loop_quality(
    frames: list[tuple[np.ndarray, np.ndarray]],
    frame_indices: list[int],
) -> dict[str, Any]:
    """Measure the published last-to-first jump against normal motion."""
    if len(frame_indices) < 2:
        raise ValueError("Animation loop quality requires at least two frames.")
    selected = [frames[index] for index in frame_indices]
    adjacent_scores = [
        _transparent_frame_transition_score(first, second)
        for first, second in zip(selected, selected[1:])
    ]
    adjacent_median = float(np.median(adjacent_scores))
    loop_score = _transparent_frame_transition_score(selected[-1], selected[0])
    permitted_score = max(adjacent_median * 1.5, 0.5)
    ratio = (
        loop_score / adjacent_median
        if adjacent_median > 1e-6
        else (0.0 if loop_score <= 1e-6 else None)
    )
    return {
        "frame_count": len(frame_indices),
        "loop_transition_score": loop_score,
        "adjacent_transition_median": adjacent_median,
        "loop_to_adjacent_ratio": ratio,
        "permitted_transition_score": permitted_score,
        "acceptable": loop_score <= permitted_score,
    }


def close_transparent_action_tail_loop(
    frames: list[tuple[np.ndarray, np.ndarray]],
    *,
    target_frame_index: int,
    maximum_tail_window_frames: int = 9,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    """Close the loop without cross-dissolving two displaced silhouettes."""

    if len(frames) < 3:
        raise ValueError("Action tail closure requires at least three frames.")
    if not 1 <= target_frame_index < len(frames):
        raise ValueError(
            "Action tail closure target must be inside the source frame range."
        )
    if maximum_tail_window_frames < 1:
        raise ValueError("Action tail closure requires a non-empty tail window.")
    tail_window_count = min(maximum_tail_window_frames, target_frame_index + 1)
    tail_window_start = target_frame_index - tail_window_count + 1
    first_rgb, first_alpha = frames[0]
    closed = [(rgb.copy(), alpha.copy()) for rgb, alpha in frames]
    tail_to_first_transition_scores = [
        _transparent_frame_transition_score(frames[index], frames[0])
        for index in range(tail_window_start, target_frame_index + 1)
    ]
    pre_terminal_transition_score = _transparent_frame_transition_score(
        frames[target_frame_index - 1],
        frames[0],
    )
    # The source drivers already return toward their opening phase.  Only snap
    # the terminal publication frame to frame zero.  A cosine RGB/alpha blend
    # between spatially separated animals creates a visible double silhouette,
    # which is worse than the small residual terminal phase correction.
    closed[target_frame_index] = (first_rgb.copy(), first_alpha.copy())
    for frame_index in range(target_frame_index + 1, len(closed)):
        closed[frame_index] = (first_rgb.copy(), first_alpha.copy())
    return closed, {
        "policy": ACTION_TAIL_GEOMETRY_SAFE_LOOP_POLICY,
        "tail_window_start_frame_index": tail_window_start,
        "tail_window_frame_count": tail_window_count,
        "target_frame_index": target_frame_index,
        "blend_frame_count": 0,
        "cross_dissolve_applied": False,
        "single_subject_silhouette_preserved": True,
        "tail_to_first_transition_score_min": round(
            min(tail_to_first_transition_scores),
            6,
        ),
        "tail_to_first_transition_score_at_target": round(
            tail_to_first_transition_scores[-1],
            6,
        ),
        "pre_terminal_transition_score": round(
            pre_terminal_transition_score,
            6,
        ),
        "terminal_frame_matches_first": True,
        "preserves_source_frame_count": len(closed) == len(frames),
        "preserves_duration": True,
    }


def transparent_guidance_component_coverage_minimum(action: str | None) -> float:
    """Return the per-component overlap floor for pose-guided extraction."""
    return 0.70 if str(action or "").strip().lower() == "toileting" else 0.45


def transparent_guidance_area_ratio_minimum(action: str | None) -> float:
    """Require the toilet tray footprint to survive final matte extraction."""
    return 0.75 if str(action or "").strip().lower() == "toileting" else 0.55


def transparent_guidance_component_area_fraction_minimum(
    action: str | None,
) -> float:
    """Return the minimum meaningful pose-guidance component area."""
    return 0.005


def transparent_guidance_dynamic_component_area_fraction_maximum(
    action: str | None,
) -> float:
    """Return the largest component eligible for bounded limb proximity."""
    return (
        0.10
        if str(action or "").strip().lower() in {"drinking", "running"}
        else 0.0
    )


def transparent_guidance_dynamic_component_radius(action: str | None) -> int:
    """Return the pixel radius allowed for a detached limb phase shift."""
    return (
        12
        if str(action or "").strip().lower() in {"drinking", "running"}
        else 0
    )


REFERENCE_ALIGNED_UNIFORM_BACKGROUND_POLICY = (
    "reference_guided_bounded_canvas_subject_and_prop_matte_v3"
)
REFERENCE_ALIGNED_ADAPTIVE_EXTRACTION_POLICY = (
    "reference_guided_adaptive_semantic_bounded_hybrid_matte_v5"
)
REFERENCE_ALIGNED_SEMANTIC_EXTRACTION_STRATEGY = (
    "semantic_u2net_static_reference_subject_and_tray_gate"
)
REFERENCE_ALIGNED_BOUNDED_CANVAS_EXTRACTION_STRATEGY = (
    "bounded_canvas_color_distance"
)
REFERENCE_ALIGNED_HYBRID_EXTRACTION_STRATEGY = (
    "hybrid_semantic_u2net_plus_bounded_canvas_static_reference_gate"
)
REFERENCE_ALIGNED_HYBRID_EXTRACTION_POLICY = (
    "reference_guided_hybrid_semantic_plus_bounded_canvas_matte_v2"
)
REFERENCE_ALIGNED_BOUNDED_CANVAS_BORDER_DISTANCE_P999_MAX = 8.0
REFERENCE_ALIGNED_HYBRID_CANVAS_BORDER_DISTANCE_P999_MAX = 32.0
DRINKING_STATIC_BOWL_ALPHA_RETENTION_POLICY = (
    "drinking_static_bowl_guidance_alpha_retention_gate_v1"
)
DRINKING_STATIC_BOWL_ALPHA_RETENTION_MIN = 0.90
DRINKING_STATIC_BOWL_RECOVERY_DILATION_RADIUS = 6


def _canvas_border_color_distance_p999(
    rgb: np.ndarray,
    *,
    exclusion_mask: np.ndarray | None = None,
) -> float:
    """Measure canvas-border variation while excluding guided foreground.

    A valid moving tail or paw can briefly touch the image border.  Those few
    foreground pixels must not make an otherwise pure, uniform canvas look
    non-uniform and force the pale animal through a semantic matte.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Canvas border measurement requires an RGB frame")
    height, width = rgb.shape[:2]
    border_width = max(4, min(8, min(height, width) // 32))
    pixels = rgb.astype(np.float32)
    border_pixels = np.concatenate(
        (
            pixels[:border_width, :, :].reshape(-1, 3),
            pixels[-border_width:, :, :].reshape(-1, 3),
            pixels[:, :border_width, :].reshape(-1, 3),
            pixels[:, -border_width:, :].reshape(-1, 3),
        ),
        axis=0,
    )
    if exclusion_mask is not None:
        exclusion_mask = np.asarray(exclusion_mask, dtype=bool)
        if exclusion_mask.shape != (height, width):
            raise ValueError(
                "Canvas border exclusion mask must match RGB dimensions"
            )
        border_excluded = np.concatenate(
            (
                exclusion_mask[:border_width, :].reshape(-1),
                exclusion_mask[-border_width:, :].reshape(-1),
                exclusion_mask[:, :border_width].reshape(-1),
                exclusion_mask[:, -border_width:].reshape(-1),
            )
        )
        retained_border = border_pixels[~border_excluded]
        if retained_border.shape[0] >= max(64, border_pixels.shape[0] // 4):
            border_pixels = retained_border
    background_rgb = np.median(border_pixels, axis=0)
    border_distance = np.sqrt(
        np.sum((border_pixels - background_rgb) ** 2, axis=1)
    )
    distance_p99 = float(np.percentile(border_distance, 99.0))
    distance_p999 = float(np.percentile(border_distance, 99.9))
    # Isolated codec specks outside the pose envelope must not route every
    # frame through U2Net.  A genuinely varying canvas also raises p99, while
    # sub-percent contamination is bounded to an eight-unit outlier allowance.
    return min(distance_p999, distance_p99 + 8.0)


def reference_aligned_transparent_extraction_strategy(
    canvas_border_distance_p999_max: float,
) -> str:
    """Choose canvas extraction only when every frame is genuinely flat."""
    if (
        float(canvas_border_distance_p999_max)
        <= REFERENCE_ALIGNED_BOUNDED_CANVAS_BORDER_DISTANCE_P999_MAX
    ):
        return REFERENCE_ALIGNED_BOUNDED_CANVAS_EXTRACTION_STRATEGY
    if (
        float(canvas_border_distance_p999_max)
        <= REFERENCE_ALIGNED_HYBRID_CANVAS_BORDER_DISTANCE_P999_MAX
    ):
        return REFERENCE_ALIGNED_HYBRID_EXTRACTION_STRATEGY
    return REFERENCE_ALIGNED_SEMANTIC_EXTRACTION_STRATEGY


def _merge_reference_aligned_static_prop(
    rgb: np.ndarray,
    alpha: np.ndarray,
    bounded_rgb: np.ndarray,
    bounded_alpha: np.ndarray,
    static_prop_guidance: np.ndarray,
    *,
    dilation_radius: int = DRINKING_STATIC_BOWL_RECOVERY_DILATION_RADIUS,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Recover only the validated static prop around its guidance footprint."""

    if rgb.shape != bounded_rgb.shape or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Static prop recovery requires matching RGB frames")
    if (
        alpha.shape != bounded_alpha.shape
        or alpha.shape != static_prop_guidance.shape
        or alpha.shape != rgb.shape[:2]
    ):
        raise ValueError("Static prop recovery requires matching alpha masks")
    if dilation_radius < 0:
        raise ValueError("Static prop recovery dilation radius cannot be negative")

    static_prop = static_prop_guidance.astype(bool)
    static_prop_pixels = int(np.count_nonzero(static_prop))
    if static_prop_pixels < 64:
        raise RuntimeError("Static prop guidance is empty or too small")
    if dilation_radius:
        diameter = dilation_radius * 2 + 1
        envelope = cv2.dilate(
            static_prop.astype(np.uint8),
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (diameter, diameter),
            ),
            iterations=1,
        ).astype(bool)
    else:
        envelope = static_prop

    recovered_support = envelope & (bounded_alpha > alpha)
    output_rgb = rgb.copy()
    output_alpha = alpha.copy()
    output_rgb[recovered_support] = bounded_rgb[recovered_support]
    output_alpha[envelope] = np.maximum(
        output_alpha[envelope],
        bounded_alpha[envelope],
    )
    output_rgb[output_alpha == 0] = 0

    retained_pixels = int(
        np.count_nonzero((output_alpha >= 128) & static_prop)
    )
    input_retained_pixels = int(
        np.count_nonzero((alpha >= 128) & static_prop)
    )
    return output_rgb, output_alpha, {
        "static_prop_guidance_pixels": float(static_prop_pixels),
        "static_prop_input_retention_ratio": (
            input_retained_pixels / float(static_prop_pixels)
        ),
        "static_prop_output_retention_ratio": (
            retained_pixels / float(static_prop_pixels)
        ),
        "static_prop_recovered_pixels": float(
            np.count_nonzero(recovered_support)
        ),
        "static_prop_recovery_envelope_pixels": float(
            np.count_nonzero(envelope)
        ),
        "static_prop_recovery_dilation_radius": float(dilation_radius),
    }


def _reference_guided_uniform_background_mask(
    rgb: np.ndarray,
    guidance: np.ndarray,
    *,
    frame_index: int,
    guidance_component_coverage_required_min: float,
    guidance_area_ratio_required_min: float,
    guidance_component_area_fraction_required_min: float,
    static_prop_guidance: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Extract a complete subject+prop matte from a low-variation canvas.

    U2Net reliably isolates an animal, but it can omit a broad rectangular pet
    toilet tray. SCAIL canvases are generated on a nearly uniform background,
    sometimes with a bounded cyan/gray gradient. For a reference-aligned
    toileting guide we estimate the canvas from its clean border, adapt the
    alpha threshold to that measured variation, and retain every connected
    foreground component overlapping the immutable dog+tray reference mask.
    This is intentionally opt-in; callers must first prove that the reference
    already contains the complete tray.
    """

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Uniform-background extraction requires an RGB frame")
    if guidance.shape != rgb.shape[:2]:
        raise ValueError(
            "Uniform-background guidance dimensions do not match the source frame"
        )
    guidance = guidance.astype(bool)
    guidance_pixels = int(np.count_nonzero(guidance))
    if guidance_pixels < 64:
        raise RuntimeError(
            "Reference-aligned uniform-background guidance is empty or too small"
        )
    if static_prop_guidance is None:
        static_prop = np.zeros_like(guidance)
    else:
        if static_prop_guidance.shape != guidance.shape:
            raise ValueError(
                "Static prop guidance dimensions do not match the source frame"
            )
        static_prop = static_prop_guidance.astype(bool)
        if int(np.count_nonzero(static_prop)) < 64:
            raise RuntimeError(
                "Reference-aligned static prop guidance is empty or too small"
            )

    height, width = rgb.shape[:2]
    border_width = max(4, min(8, min(height, width) // 32))
    pixels = rgb.astype(np.float32)
    border_pixels = np.concatenate(
        (
            pixels[:border_width, :, :].reshape(-1, 3),
            pixels[-border_width:, :, :].reshape(-1, 3),
            pixels[:, :border_width, :].reshape(-1, 3),
            pixels[:, -border_width:, :].reshape(-1, 3),
        ),
        axis=0,
    )
    background_rgb = np.median(border_pixels, axis=0)
    guidance_exclusion = cv2.dilate(
        guidance.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61)),
        iterations=1,
    ).astype(bool)
    border_distance_p999 = _canvas_border_color_distance_p999(
        rgb,
        exclusion_mask=guidance_exclusion,
    )
    if border_distance_p999 > 32.0:
        raise RuntimeError(
            "Reference-aligned toileting canvas varies too much at frame "
            f"{frame_index}: border_distance_p999={border_distance_p999:.4f}"
        )

    distance = np.sqrt(
        np.sum((pixels - background_rgb.reshape(1, 1, 3)) ** 2, axis=2)
    )
    # A truly flat canvas keeps the original 6..20 soft transition. Generated
    # cyan/gray canvases can have a smooth border drift near 21 RGB-distance
    # units; placing the zero-alpha boundary three units above the measured
    # p99.9 removes that canvas without weakening dog+tray coverage.
    alpha_low = max(6.0, border_distance_p999 + 3.0)
    alpha_high = max(20.0, alpha_low + 12.0)
    alpha = np.clip(
        (distance - alpha_low) * (255.0 / (alpha_high - alpha_low)),
        0.0,
        255.0,
    ).astype(np.uint8)
    static_prop_recovered = static_prop & (alpha < 64)
    if np.any(static_prop):
        # The validated toileting reference already separates the immutable
        # tray as the red SCAIL object.  A pale generated tray can be nearly the
        # same colour as a cyan/gray canvas and is therefore invisible to a
        # distance-only matte.  Preserve that fixed prop role explicitly while
        # leaving the animal to semantic/canvas extraction.
        alpha[static_prop] = 255

    hard = (alpha >= 16).astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        hard,
        connectivity=8,
    )
    selected_labels: list[int] = []
    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        component = labels == label
        overlap = int(np.count_nonzero(component & guidance))
        if area >= 64 and overlap >= 32 and overlap / float(area) >= 0.10:
            selected_labels.append(label)
    if not selected_labels:
        raise RuntimeError(
            "Uniform-background extraction found no reference-aligned subject "
            f"at frame {frame_index}"
        )
    selected = np.isin(labels, selected_labels).astype(np.uint8)
    support = cv2.dilate(
        selected,
        np.ones((5, 5), dtype=np.uint8),
        iterations=1,
    )
    alpha = np.where(support > 0, alpha, 0).astype(np.uint8)
    alpha[:2, :] = 0
    alpha[-2:, :] = 0
    alpha[:, :2] = 0
    alpha[:, -2:] = 0

    foreground = alpha >= 64
    foreground_pixels = int(np.count_nonzero(foreground))
    foreground_fraction = float(foreground_pixels) / float(alpha.size)
    guidance_coverage = float(np.count_nonzero(foreground & guidance)) / float(
        guidance_pixels
    )
    area_ratio = float(foreground_pixels) / float(guidance_pixels)

    guidance_component_count, guidance_labels, guidance_stats, _ = (
        cv2.connectedComponentsWithStats(
            guidance.astype(np.uint8),
            connectivity=8,
        )
    )
    minimum_guidance_component = max(
        64,
        int(
            guidance_pixels
            * guidance_component_area_fraction_required_min
        ),
    )
    guidance_component_coverages: list[float] = []
    ignored_guidance_component_count = 0
    ignored_guidance_component_pixels = 0
    for label in range(1, guidance_component_count):
        component_area = int(guidance_stats[label, cv2.CC_STAT_AREA])
        if component_area < minimum_guidance_component:
            ignored_guidance_component_count += 1
            ignored_guidance_component_pixels += component_area
            continue
        component = guidance_labels == label
        guidance_component_coverages.append(
            float(np.count_nonzero(foreground & component))
            / float(component_area)
        )
    guidance_component_coverage_min = min(
        guidance_component_coverages,
        default=guidance_coverage,
    )
    ignored_guidance_component_fraction = float(
        ignored_guidance_component_pixels
    ) / float(guidance_pixels)

    if (
        guidance_coverage < 0.55
        or not guidance_area_ratio_required_min <= area_ratio <= 1.85
    ):
        raise RuntimeError(
            "Uniform-background mask diverged from reference guidance at frame "
            f"{frame_index}: coverage={guidance_coverage:.4f}, "
            f"area_ratio={area_ratio:.4f}, required_area_ratio_min="
            f"{guidance_area_ratio_required_min:.4f}"
        )
    if (
        guidance_component_coverage_min
        < guidance_component_coverage_required_min
    ):
        raise RuntimeError(
            "Uniform-background mask missed a reference foreground component at "
            f"frame {frame_index}: minimum component coverage="
            f"{guidance_component_coverage_min:.4f}, required="
            f"{guidance_component_coverage_required_min:.4f}"
        )
    if ignored_guidance_component_fraction > 0.02:
        raise RuntimeError(
            "Reference guidance contains too much ignored fragment area at frame "
            f"{frame_index}: ignored_fraction="
            f"{ignored_guidance_component_fraction:.4f}"
        )
    if not 0.02 <= foreground_fraction <= 0.85:
        raise RuntimeError(
            "Uniform-background foreground fraction is invalid at frame "
            f"{frame_index}: {foreground_fraction:.4f}"
        )

    # Remove the estimated canvas color from soft edge pixels. Limiting the
    # correction keeps pale fur stable while preventing a cyan/gray fringe when
    # the transparent animation is shown on a dark background.
    output_rgb = pixels.copy()
    alpha_float = alpha.astype(np.float32) / 255.0
    soft_edge = (alpha > 0) & (alpha < 250)
    denominator = np.maximum(alpha_float, 0.08)[..., None]
    unblended = (
        pixels
        - background_rgb.reshape(1, 1, 3)
        * (1.0 - alpha_float[..., None])
    ) / denominator
    bounded_delta = np.clip(unblended - pixels, -48.0, 48.0)
    output_rgb[soft_edge] = np.clip(
        pixels[soft_edge] + bounded_delta[soft_edge],
        0.0,
        255.0,
    )
    output_rgb = output_rgb.astype(np.uint8)
    if np.any(static_prop_recovered):
        neutral_prop = cv2.cvtColor(pixels.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        output_rgb[static_prop_recovered] = np.repeat(
            neutral_prop[static_prop_recovered, None],
            3,
            axis=1,
        )
    output_rgb[alpha == 0] = 0

    return output_rgb, alpha, {
        "foreground_fraction": foreground_fraction,
        "guidance_coverage": guidance_coverage,
        "guidance_component_coverage_min": (
            guidance_component_coverage_min
        ),
        "guidance_component_coverage_required_min": (
            guidance_component_coverage_required_min
        ),
        "guidance_component_direct_coverage_min": (
            guidance_component_coverage_min
        ),
        "guidance_component_area_fraction_required_min": (
            guidance_component_area_fraction_required_min
        ),
        "guidance_dynamic_component_area_fraction_max": 0.0,
        "guidance_dynamic_component_radius": 0.0,
        "guidance_dynamic_component_count": 0.0,
        "guidance_dynamic_component_proximity_coverage_min": 1.0,
        "ignored_guidance_component_count": float(
            ignored_guidance_component_count
        ),
        "ignored_guidance_component_fraction": (
            ignored_guidance_component_fraction
        ),
        "area_ratio": area_ratio,
        "semantic_core_retention": 1.0,
        "semantic_upper_core_retention": 1.0,
        "semantic_grid_core_retention_min": 1.0,
        "semantic_fragment_recovery_count": 0.0,
        "background_red": float(background_rgb[0]),
        "background_green": float(background_rgb[1]),
        "background_blue": float(background_rgb[2]),
        "background_border_distance_p999": border_distance_p999,
        "background_alpha_low": alpha_low,
        "background_alpha_high": alpha_high,
        "static_prop_guidance_pixels": float(np.count_nonzero(static_prop)),
        "static_prop_recovered_pixels": float(
            np.count_nonzero(static_prop_recovered)
        ),
        "static_prop_guidance_preserved": bool(np.any(static_prop)),
    }


def build_unstabilized_transparent_gif(
    source_video: Path,
    gif_target: Path,
    *,
    action: str | None = None,
    guidance_mask_video: Path | None = None,
    reference_aligned_uniform_background: bool = False,
    output_size: tuple[int, int] = (512, 512),
    source_fps: float = 16.0,
    gif_fps: int = 20,
    gif_frames: int = 100,
    white_mp4_target: Path | None = None,
    soft_apng_target: Path | None = None,
    start_frame_index: int = 0,
) -> dict[str, Any]:
    """Extract one matte and publish clean GIF plus optional soft-alpha APNG."""
    if source_fps <= 0 or gif_fps < 1 or gif_frames < 1:
        raise ValueError(
            "Transparent output timing requires positive FPS and frame counts"
        )
    if 100 % gif_fps != 0:
        raise ValueError(
            "Transparent GIF FPS must map exactly to GIF centisecond delays"
        )
    normalized_action = str(action or "").strip().lower()
    # A running driver's airborne paw or a drinking driver's separated foreleg
    # can be a small detached guidance component. SCAIL preserves the quadruped
    # but may shift that limb with gait phase or transferred body proportions.
    # Keep every meaningful component in the 45% coverage gate, but permit a
    # twelve-pixel proximity match only for non-primary drinking/running limb
    # components occupying at most 10% of the guidance subject.
    # Overall guidance coverage, area ratio, semantic-core retention, and local
    # grid retention remain unchanged and fail closed.
    guidance_component_coverage_required_min = (
        transparent_guidance_component_coverage_minimum(normalized_action)
    )
    guidance_area_ratio_required_min = transparent_guidance_area_ratio_minimum(
        normalized_action
    )
    guidance_component_area_fraction_required_min = (
        transparent_guidance_component_area_fraction_minimum(normalized_action)
    )
    guidance_dynamic_component_area_fraction_max = (
        transparent_guidance_dynamic_component_area_fraction_maximum(
            normalized_action
        )
    )
    guidance_dynamic_component_radius_px = (
        transparent_guidance_dynamic_component_radius(normalized_action)
    )
    frames_root = gif_target.parent / f".{gif_target.stem}.{uuid.uuid4().hex}.frames"
    binary_dir = frames_root / "gif"
    soft_dir = frames_root / "soft"
    temporary = gif_target.with_name(
        f".{gif_target.stem}.{uuid.uuid4().hex}.gif"
    )
    white_temporary = (
        white_mp4_target.with_name(
            f".{white_mp4_target.stem}.{uuid.uuid4().hex}.mp4"
        )
        if white_mp4_target is not None
        else None
    )
    apng_temporary = (
        soft_apng_target.with_name(
            f".{soft_apng_target.stem}.{uuid.uuid4().hex}.png"
        )
        if soft_apng_target is not None
        else None
    )
    if soft_apng_target is not None:
        soft_apng_target.unlink(missing_ok=True)
    binary_dir.mkdir(parents=True, exist_ok=False)
    soft_dir.mkdir(parents=True, exist_ok=False)
    try:
        with av.open(str(source_video)) as container:
            source_frames = [
                frame.to_ndarray(format="rgb24")
                for frame in container.decode(video=0)
            ]
        if not source_frames:
            raise RuntimeError(f"No frames decoded from {source_video}")
        source_frame_count = len(source_frames)
        source_index_map = list(range(source_frame_count))
        phase_start_index = int(start_frame_index)
        if not 0 <= phase_start_index < source_frame_count:
            raise ValueError(
                "Transparent animation phase start is outside the source: "
                f"{phase_start_index} not in 0..{source_frame_count - 1}"
            )

        guidance_frames: list[np.ndarray] | None = None
        if guidance_mask_video is not None:
            with av.open(str(guidance_mask_video)) as container:
                guidance_frames = [
                    frame.to_ndarray(format="rgb24")
                    for frame in container.decode(video=0)
                ]
            if len(guidance_frames) < len(source_frames):
                raise RuntimeError(
                    "Guidance mask video is shorter than the generated video: "
                    f"{len(guidance_frames)} < {len(source_frames)}"
                )
            guidance_frames = guidance_frames[:source_frame_count]

        if phase_start_index:
            source_frames = (
                source_frames[phase_start_index:]
                + source_frames[:phase_start_index]
            )
            source_index_map = (
                source_index_map[phase_start_index:]
                + source_index_map[:phase_start_index]
            )
            if guidance_frames is not None:
                guidance_frames = (
                    guidance_frames[phase_start_index:]
                    + guidance_frames[:phase_start_index]
                )

        temporal_cycle: dict[str, Any] = {
            "policy": None,
            "action": action,
            "applied": False,
            "closed_loop": False,
            "preserves_source_frame_count": True,
            "preserves_duration": True,
        }
        if action == "grooming":
            loop_indices = grooming_closed_loop_source_indices(
                len(source_frames)
            )
            source_frames = [source_frames[index] for index in loop_indices]
            source_index_map = [
                source_index_map[index] for index in loop_indices
            ]
            if guidance_frames is not None:
                guidance_frames = [
                    guidance_frames[index] for index in loop_indices
                ]
            temporal_cycle = {
                "policy": GROOMING_TEMPORAL_LOOP_POLICY,
                "action": action,
                "applied": True,
                "playback": "validated_second_half_forward_then_reverse",
                "pivot_source_frame_index": source_index_map[0],
                "source_index_map": source_index_map,
                "closed_loop": source_index_map[0] == source_index_map[-1],
                "preserves_source_frame_count": (
                    len(source_frames) == source_frame_count
                ),
                "preserves_duration": True,
            }

        def normalized_rgb(frame: np.ndarray) -> np.ndarray:
            if (frame.shape[1], frame.shape[0]) == output_size:
                return frame
            return np.asarray(
                Image.fromarray(frame).resize(
                    output_size, Image.Resampling.LANCZOS
                )
            )

        normalized_source = [normalized_rgb(frame) for frame in source_frames]
        def normalized_guidance_roles(
            index: int,
        ) -> tuple[np.ndarray | None, np.ndarray | None]:
            if guidance_frames is None:
                return None, None
            guidance_rgb = normalized_rgb(guidance_frames[index])
            identity = (
                (guidance_rgb[:, :, 2] >= 225)
                & (guidance_rgb[:, :, 0] <= 30)
                & (guidance_rgb[:, :, 1] <= 30)
            )
            static_prop = (
                (guidance_rgb[:, :, 0] >= 225)
                & (guidance_rgb[:, :, 1] <= 30)
                & (guidance_rgb[:, :, 2] <= 30)
            )
            if not reference_aligned_uniform_background:
                return identity, None
            if int(np.count_nonzero(static_prop)) < 64:
                return identity, None
            return identity | static_prop, static_prop

        def normalized_guidance_mask(index: int) -> np.ndarray | None:
            guidance, _static_prop = normalized_guidance_roles(index)
            return guidance

        reference_aligned_border_distance_p999_values: list[float] = []
        reference_aligned_extraction_strategy: str | None = None
        drinking_static_prop_recovery_required = bool(
            normalized_action == "drinking"
            and reference_aligned_uniform_background
        )
        if reference_aligned_uniform_background:
            if guidance_frames is None:
                raise RuntimeError(
                    "Reference-aligned adaptive extraction requires a "
                    "guidance mask video"
                )
            for index, frame in enumerate(normalized_source):
                guidance = normalized_guidance_mask(index)
                if guidance is None:
                    raise RuntimeError(
                        "Reference-aligned adaptive extraction requires pose "
                        "guidance in every frame"
                    )
                exclusion = cv2.dilate(
                    guidance.astype(np.uint8),
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61)),
                    iterations=1,
                ).astype(bool)
                reference_aligned_border_distance_p999_values.append(
                    _canvas_border_color_distance_p999(
                        frame,
                        exclusion_mask=exclusion,
                    )
                )
            reference_aligned_extraction_strategy = (
                reference_aligned_transparent_extraction_strategy(
                    max(reference_aligned_border_distance_p999_values)
                )
            )
            if drinking_static_prop_recovery_required:
                if (
                    reference_aligned_extraction_strategy
                    == REFERENCE_ALIGNED_SEMANTIC_EXTRACTION_STRATEGY
                ):
                    raise RuntimeError(
                        "Drinking static bowl recovery requires a bounded "
                        "uniform canvas"
                    )
                # Keep U2Net's cleaner animal edge, then recover only the bowl
                # from the bounded-canvas matte. Applying the bounded matte to
                # the whole pale animal creates a noisy silhouette.
                reference_aligned_extraction_strategy = (
                    REFERENCE_ALIGNED_HYBRID_EXTRACTION_STRATEGY
                )

        chroma_masks: list[tuple[np.ndarray, np.ndarray]] = []
        chroma_usable = True
        for frame in normalized_source:
            extracted_rgb, extracted_alpha = _subject_mask(frame.copy())
            corners = (
                int(extracted_alpha[0, 0]),
                int(extracted_alpha[0, -1]),
                int(extracted_alpha[-1, 0]),
                int(extracted_alpha[-1, -1]),
            )
            if any(corners):
                chroma_usable = False
                break
            chroma_masks.append((extracted_rgb, extracted_alpha))

        reference_aligned_bounded_frames: (
            list[tuple[np.ndarray, np.ndarray]] | None
        ) = None
        reference_aligned_bounded_metrics: list[dict[str, float]] = []
        static_prop_recovery_metrics: list[dict[str, float]] = []
        if (
            not chroma_usable
            and reference_aligned_extraction_strategy
            == REFERENCE_ALIGNED_HYBRID_EXTRACTION_STRATEGY
        ):
            reference_aligned_bounded_frames = []
            for index, frame in enumerate(normalized_source):
                guidance, static_prop_guidance = normalized_guidance_roles(
                    index
                )
                if guidance is None:
                    raise RuntimeError(
                        "Reference-aligned hybrid extraction requires a "
                        "guidance mask video"
                    )
                bounded_rgb, bounded_alpha, bounded_metrics = (
                    _reference_guided_uniform_background_mask(
                        frame.copy(),
                        guidance,
                        frame_index=index,
                        guidance_component_coverage_required_min=(
                            guidance_component_coverage_required_min
                        ),
                        guidance_area_ratio_required_min=(
                            guidance_area_ratio_required_min
                        ),
                        guidance_component_area_fraction_required_min=(
                            guidance_component_area_fraction_required_min
                        ),
                        static_prop_guidance=static_prop_guidance,
                    )
                )
                reference_aligned_bounded_frames.append(
                    (bounded_rgb, bounded_alpha)
                )
                reference_aligned_bounded_metrics.append(bounded_metrics)

        extraction_method = "chroma_key"
        extracted_frames: list[tuple[np.ndarray, np.ndarray]]
        semantic_metrics: list[dict[str, float]] = []
        if chroma_usable and len(chroma_masks) == len(normalized_source):
            extracted_frames = chroma_masks
        elif (
            reference_aligned_uniform_background
            and reference_aligned_extraction_strategy
            == REFERENCE_ALIGNED_BOUNDED_CANVAS_EXTRACTION_STRATEGY
        ):
            extraction_method = REFERENCE_ALIGNED_UNIFORM_BACKGROUND_POLICY
            extracted_frames = []
            for index, frame in enumerate(normalized_source):
                guidance, static_prop_guidance = normalized_guidance_roles(
                    index
                )
                if guidance is None:
                    raise RuntimeError(
                        "Reference-aligned bounded-canvas extraction requires "
                        "a guidance mask video"
                    )
                extracted_rgb, extracted_alpha, frame_metrics = (
                    _reference_guided_uniform_background_mask(
                        frame.copy(),
                        guidance,
                        frame_index=index,
                        guidance_component_coverage_required_min=(
                            guidance_component_coverage_required_min
                        ),
                        guidance_area_ratio_required_min=(
                            guidance_area_ratio_required_min
                        ),
                        guidance_component_area_fraction_required_min=(
                            guidance_component_area_fraction_required_min
                        ),
                        static_prop_guidance=static_prop_guidance,
                    )
                )
                extracted_frames.append((extracted_rgb, extracted_alpha))
                semantic_metrics.append(frame_metrics)
        else:
            extraction_method = (
                REFERENCE_ALIGNED_HYBRID_EXTRACTION_POLICY
                if reference_aligned_extraction_strategy
                == REFERENCE_ALIGNED_HYBRID_EXTRACTION_STRATEGY
                else "rembg_u2net_guided_by_pose_mask"
            )
            from rembg import new_session, remove

            session = new_session(
                "u2net",
                providers=["CPUExecutionProvider"],
            )
            extracted_frames = []
            for index, frame in enumerate(normalized_source):
                result = remove(
                    Image.fromarray(frame),
                    session=session,
                    alpha_matting=False,
                )
                if not isinstance(result, Image.Image):
                    raise RuntimeError(
                        f"rembg returned an unsupported result for frame {index}"
                    )
                rgba = np.asarray(result.convert("RGBA"))
                rgb = rgba[:, :, :3].copy()
                alpha = rgba[:, :, 3].copy()

                guidance: np.ndarray | None = None
                if guidance_frames is not None:
                    guidance = normalized_guidance_mask(index)

                hard = (alpha >= 16).astype(np.uint8)
                component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
                    hard,
                    connectivity=8,
                )
                if component_count <= 1:
                    raise RuntimeError(
                        f"Semantic extraction found no subject in frame {index}"
                    )
                candidates: list[tuple[int, int, int]] = []
                for label in range(1, component_count):
                    area = int(stats[label, cv2.CC_STAT_AREA])
                    overlap = (
                        int(np.count_nonzero((labels == label) & guidance))
                        if guidance is not None
                        else area
                    )
                    candidates.append((overlap, area, label))
                if guidance is None:
                    selected_labels = [max(candidates)[2]]
                else:
                    selected_labels = [
                        label
                        for overlap, area, label in candidates
                        if area >= 64
                        and overlap >= 32
                        and overlap / float(max(1, area)) >= 0.20
                    ]
                    if not selected_labels:
                        selected_labels = [max(candidates)[2]]
                selected = np.isin(labels, selected_labels).astype(np.uint8)
                support = cv2.dilate(
                    selected,
                    np.ones((7, 7), dtype=np.uint8),
                    iterations=1,
                )
                alpha = np.where(support > 0, alpha, 0).astype(np.uint8)
                semantic_alpha_anchored = alpha.copy()
                recovered_semantic_fragment_labels: list[int] = []
                if guidance is not None:
                    # Guidance is only a component/noise constraint. It must
                    # never become a hard crop: a replacement animal can have
                    # a taller back, longer fur, or a different silhouette than
                    # the driving animal. Preserve every high-confidence U2Net
                    # subject pixel and its soft edge; constrain only remote
                    # low-confidence pixels that are neither near guidance nor
                    # near a high-confidence semantic core.
                    guidance_u8 = guidance.astype(np.uint8) * 255
                    guidance_envelope = cv2.dilate(
                        guidance_u8,
                        np.ones((61, 61), dtype=np.uint8),
                        iterations=1,
                    )
                    semantic_core = (
                        semantic_alpha_anchored >= 160
                    ).astype(np.uint8)
                    semantic_core_support = cv2.dilate(
                        semantic_core,
                        np.ones((11, 11), dtype=np.uint8),
                        iterations=1,
                    )
                    cleanup_support = (
                        (guidance_envelope > 0)
                        | (semantic_core_support > 0)
                    )
                    alpha = np.where(
                        cleanup_support,
                        semantic_alpha_anchored,
                        0,
                    ).astype(np.uint8)

                    restricted_hard = (alpha >= 16).astype(np.uint8)
                    restricted_count, restricted_labels, restricted_stats, _ = (
                        cv2.connectedComponentsWithStats(
                            restricted_hard,
                            connectivity=8,
                        )
                    )
                    if restricted_count <= 1:
                        raise RuntimeError(
                            "Pose-guided semantic extraction removed the subject "
                            f"in frame {index}"
                        )
                    (
                        restricted_selected_labels,
                        recovered_semantic_fragment_labels,
                    ) = _select_pose_guided_semantic_components(
                        restricted_labels,
                        restricted_stats,
                        guidance,
                        semantic_alpha_anchored >= 160,
                    )
                    restricted_component = np.isin(
                        restricted_labels,
                        restricted_selected_labels,
                    ).astype(np.uint8)
                    restricted_support = cv2.dilate(
                        restricted_component,
                        np.ones((5, 5), dtype=np.uint8),
                        iterations=1,
                    )
                    alpha = np.where(
                        restricted_support > 0,
                        alpha,
                        0,
                    ).astype(np.uint8)

                    # The post-guidance matte must retain the semantic model's
                    # high-confidence subject, globally and locally. This
                    # catches accidental missing backs, tails, ears, or limbs
                    # even when overall pose-guidance coverage still looks
                    # healthy.
                    semantic_core = semantic_alpha_anchored >= 160
                    retained_foreground = alpha >= 64
                    semantic_core_pixels = max(
                        1,
                        int(np.count_nonzero(semantic_core)),
                    )
                    semantic_core_retention = float(
                        np.count_nonzero(
                            retained_foreground & semantic_core
                        )
                    ) / float(semantic_core_pixels)

                    core_y, core_x = np.nonzero(semantic_core)
                    if core_x.size:
                        core_left = int(core_x.min())
                        core_top = int(core_y.min())
                        core_right = int(core_x.max()) + 1
                        core_bottom = int(core_y.max()) + 1
                        upper_limit = core_top + max(
                            1,
                            int(round((core_bottom - core_top) * 0.60)),
                        )
                        upper_core = semantic_core.copy()
                        upper_core[upper_limit:, :] = False
                        upper_core_pixels = max(
                            1,
                            int(np.count_nonzero(upper_core)),
                        )
                        semantic_upper_core_retention = float(
                            np.count_nonzero(
                                retained_foreground & upper_core
                            )
                        ) / float(upper_core_pixels)

                        grid_retentions: list[float] = []
                        x_edges = np.linspace(
                            core_left,
                            core_right,
                            num=5,
                            dtype=int,
                        )
                        y_edges = np.linspace(
                            core_top,
                            core_bottom,
                            num=5,
                            dtype=int,
                        )
                        for grid_y in range(4):
                            for grid_x in range(4):
                                y0 = int(y_edges[grid_y])
                                y1 = int(y_edges[grid_y + 1])
                                x0 = int(x_edges[grid_x])
                                x1 = int(x_edges[grid_x + 1])
                                cell_core = semantic_core[y0:y1, x0:x1]
                                cell_pixels = int(
                                    np.count_nonzero(cell_core)
                                )
                                if cell_pixels < 64:
                                    continue
                                cell_retained = retained_foreground[
                                    y0:y1,
                                    x0:x1,
                                ]
                                grid_retentions.append(
                                    float(
                                        np.count_nonzero(
                                            cell_retained & cell_core
                                        )
                                    )
                                    / float(cell_pixels)
                                )
                        semantic_grid_core_retention_min = min(
                            grid_retentions,
                            default=semantic_core_retention,
                        )
                    else:
                        semantic_upper_core_retention = 1.0
                        semantic_grid_core_retention_min = 1.0

                    if semantic_core_retention < 0.97:
                        raise RuntimeError(
                            "Semantic cleanup removed too much of the "
                            "high-confidence subject at frame "
                            f"{index}: retention="
                            f"{semantic_core_retention:.4f}"
                        )
                    if semantic_upper_core_retention < 0.98:
                        raise RuntimeError(
                            "Semantic cleanup damaged the upper body/back at "
                            f"frame {index}: retention="
                            f"{semantic_upper_core_retention:.4f}"
                        )
                    if semantic_grid_core_retention_min < 0.93:
                        raise RuntimeError(
                            "Semantic cleanup removed a local subject region "
                            f"at frame {index}: minimum grid retention="
                            f"{semantic_grid_core_retention_min:.4f}"
                        )
                else:
                    semantic_core_retention = 1.0
                    semantic_upper_core_retention = 1.0
                    semantic_grid_core_retention_min = 1.0
                if reference_aligned_bounded_frames is not None:
                    bounded_rgb, bounded_alpha = (
                        reference_aligned_bounded_frames[index]
                    )
                    if drinking_static_prop_recovery_required:
                        _guidance, static_prop_guidance = (
                            normalized_guidance_roles(index)
                        )
                        if static_prop_guidance is None:
                            raise RuntimeError(
                                "Drinking transparent extraction requires "
                                "static bowl guidance"
                            )
                        rgb, alpha, recovery_metrics = (
                            _merge_reference_aligned_static_prop(
                                rgb,
                                alpha,
                                bounded_rgb,
                                bounded_alpha,
                                static_prop_guidance,
                            )
                        )
                        static_prop_recovery_metrics.append(
                            recovery_metrics
                        )
                    else:
                        bounded_additions = bounded_alpha > alpha
                        rgb[bounded_additions] = bounded_rgb[
                            bounded_additions
                        ]
                        alpha = np.maximum(alpha, bounded_alpha).astype(
                            np.uint8
                        )
                alpha[:2, :] = 0
                alpha[-2:, :] = 0
                alpha[:, :2] = 0
                alpha[:, -2:] = 0

                foreground = alpha >= 64
                foreground_fraction = float(np.count_nonzero(foreground)) / float(
                    alpha.size
                )
                guidance_coverage = 1.0
                guidance_component_coverage_min = 1.0
                guidance_component_direct_coverage_min = 1.0
                guidance_dynamic_component_count = 0
                guidance_dynamic_component_proximity_coverage_min = 1.0
                ignored_guidance_component_count = 0
                ignored_guidance_component_fraction = 0.0
                area_ratio = 1.0
                if guidance is not None:
                    guidance_pixels = max(1, int(np.count_nonzero(guidance)))
                    guidance_coverage = float(
                        np.count_nonzero(foreground & guidance)
                    ) / float(guidance_pixels)
                    area_ratio = float(np.count_nonzero(foreground)) / float(
                        guidance_pixels
                    )
                    guidance_component_count, guidance_labels, guidance_stats, _ = (
                        cv2.connectedComponentsWithStats(
                            guidance.astype(np.uint8),
                            connectivity=8,
                        )
                    )
                    guidance_component_coverages: list[float] = []
                    guidance_component_direct_coverages: list[float] = []
                    guidance_dynamic_component_proximity_coverages: list[
                        float
                    ] = []
                    minimum_guidance_component = max(
                        64,
                        int(
                            guidance_pixels
                            * guidance_component_area_fraction_required_min
                        ),
                    )
                    ignored_guidance_component_pixels = 0
                    largest_guidance_component_label = (
                        1
                        + int(
                            np.argmax(
                                guidance_stats[
                                    1:guidance_component_count,
                                    cv2.CC_STAT_AREA,
                                ]
                            )
                        )
                        if guidance_component_count > 1
                        else None
                    )
                    proximity_foreground = (
                        cv2.dilate(
                            foreground.astype(np.uint8),
                            cv2.getStructuringElement(
                                cv2.MORPH_ELLIPSE,
                                (
                                    guidance_dynamic_component_radius_px * 2 + 1,
                                    guidance_dynamic_component_radius_px * 2 + 1,
                                ),
                            ),
                            iterations=1,
                        )
                        if guidance_dynamic_component_radius_px > 0
                        else foreground.astype(np.uint8)
                    )
                    for label in range(1, guidance_component_count):
                        component_area = int(
                            guidance_stats[label, cv2.CC_STAT_AREA]
                        )
                        if component_area < minimum_guidance_component:
                            ignored_guidance_component_count += 1
                            ignored_guidance_component_pixels += component_area
                            continue
                        component = guidance_labels == label
                        direct_coverage = float(
                            np.count_nonzero(foreground & component)
                        ) / float(component_area)
                        effective_coverage = direct_coverage
                        component_area_fraction = float(component_area) / float(
                            guidance_pixels
                        )
                        if (
                            guidance_dynamic_component_area_fraction_max > 0
                            and label != largest_guidance_component_label
                            and component_area_fraction
                            <= guidance_dynamic_component_area_fraction_max
                        ):
                            proximity_coverage = float(
                                np.count_nonzero(
                                    (proximity_foreground > 0) & component
                                )
                            ) / float(component_area)
                            effective_coverage = max(
                                direct_coverage,
                                proximity_coverage,
                            )
                            guidance_dynamic_component_count += 1
                            guidance_dynamic_component_proximity_coverages.append(
                                proximity_coverage
                            )
                        guidance_component_direct_coverages.append(direct_coverage)
                        guidance_component_coverages.append(effective_coverage)
                    guidance_component_coverage_min = min(
                        guidance_component_coverages,
                        default=guidance_coverage,
                    )
                    guidance_component_direct_coverage_min = min(
                        guidance_component_direct_coverages,
                        default=guidance_coverage,
                    )
                    guidance_dynamic_component_proximity_coverage_min = min(
                        guidance_dynamic_component_proximity_coverages,
                        default=1.0,
                    )
                    ignored_guidance_component_fraction = float(
                        ignored_guidance_component_pixels
                    ) / float(guidance_pixels)
                    if (
                        guidance_coverage < 0.55
                        or not guidance_area_ratio_required_min
                        <= area_ratio
                        <= 1.85
                    ):
                        raise RuntimeError(
                            "Semantic mask diverged from pose guidance at frame "
                            f"{index}: coverage={guidance_coverage:.4f}, "
                            f"area_ratio={area_ratio:.4f}, "
                            f"required_area_ratio_min="
                            f"{guidance_area_ratio_required_min:.4f}, "
                            f"action={normalized_action or 'unspecified'}"
                        )
                    if (
                        guidance_component_coverage_min
                        < guidance_component_coverage_required_min
                    ):
                        raise RuntimeError(
                            "Semantic mask missed a guided foreground component "
                            f"at frame {index}: minimum component coverage="
                            f"{guidance_component_coverage_min:.4f}, required="
                            f"{guidance_component_coverage_required_min:.4f}, "
                            f"action={normalized_action or 'unspecified'}"
                        )
                    if ignored_guidance_component_fraction > 0.02:
                        raise RuntimeError(
                            "Pose guidance contains too much ignored fragment area "
                            f"at frame {index}: ignored_fraction="
                            f"{ignored_guidance_component_fraction:.4f}, "
                            "maximum=0.0200"
                        )
                if not 0.02 <= foreground_fraction <= 0.85:
                    raise RuntimeError(
                        "Semantic foreground fraction is invalid at frame "
                        f"{index}: {foreground_fraction:.4f}"
                    )

                max_red_blue = np.maximum(rgb[:, :, 0], rgb[:, :, 2])
                green_spill = (alpha > 0) & (rgb[:, :, 1] > max_red_blue)
                rgb[:, :, 1][green_spill] = max_red_blue[green_spill]
                rgb[alpha == 0] = 0
                extracted_frames.append((rgb, alpha))
                semantic_metrics.append(
                    {
                        "foreground_fraction": foreground_fraction,
                        "guidance_coverage": guidance_coverage,
                        "guidance_component_coverage_min": (
                            guidance_component_coverage_min
                        ),
                        "guidance_component_coverage_required_min": (
                            guidance_component_coverage_required_min
                        ),
                        "guidance_component_direct_coverage_min": (
                            guidance_component_direct_coverage_min
                        ),
                        "guidance_component_area_fraction_required_min": (
                            guidance_component_area_fraction_required_min
                        ),
                        "guidance_dynamic_component_area_fraction_max": (
                            guidance_dynamic_component_area_fraction_max
                        ),
                        "guidance_dynamic_component_radius": float(
                            guidance_dynamic_component_radius_px
                        ),
                        "guidance_dynamic_component_count": float(
                            guidance_dynamic_component_count
                        ),
                        "guidance_dynamic_component_proximity_coverage_min": (
                            guidance_dynamic_component_proximity_coverage_min
                        ),
                        "ignored_guidance_component_count": float(
                            ignored_guidance_component_count
                        ),
                        "ignored_guidance_component_fraction": (
                            ignored_guidance_component_fraction
                        ),
                        "area_ratio": area_ratio,
                        "semantic_core_retention": (
                            semantic_core_retention
                        ),
                        "semantic_upper_core_retention": (
                            semantic_upper_core_retention
                        ),
                        "semantic_grid_core_retention_min": (
                            semantic_grid_core_retention_min
                        ),
                        "semantic_fragment_recovery_count": len(
                            recovered_semantic_fragment_labels
                        ),
                    }
                )

        interior_repair_metrics: list[dict[str, Any]] = []
        pre_cleanup_repaired_frames: list[tuple[np.ndarray, np.ndarray]] = []
        for frame_index, (rgb, alpha) in enumerate(extracted_frames):
            guidance = normalized_guidance_mask(frame_index)
            repaired_rgb, repaired_alpha, repair_metrics = (
                repair_guided_alpha_interior_holes(
                    rgb,
                    alpha,
                    guidance,
                )
            )
            repair_metrics["frame_index"] = frame_index
            repair_metrics["stage"] = "post_extraction_pre_edge_cleanup"
            interior_repair_metrics.append(repair_metrics)
            pre_cleanup_repaired_frames.append(
                (repaired_rgb, repaired_alpha)
            )
        extracted_frames = pre_cleanup_repaired_frames

        edge_cleanup_metrics: list[dict[str, Any]] = []
        cleaned_extracted_frames: list[tuple[np.ndarray, np.ndarray]] = []
        for frame_index, (rgb, alpha) in enumerate(extracted_frames):
            cleaned_rgb, cleaned_alpha, cleanup_metrics = (
                clean_transparent_edge_halo(
                    rgb,
                    alpha,
                    action=action,
                )
            )
            cleanup_metrics["cleanup_pass_count"] = 1
            cleanup_metrics["second_pass_selected"] = False
            if (
                reference_aligned_uniform_background
                and extraction_method
                in {
                    "rembg_u2net_guided_by_pose_mask",
                    REFERENCE_ALIGNED_HYBRID_EXTRACTION_POLICY,
                }
                and (
                    not bool(cleanup_metrics.get("acceptable"))
                    or float(cleanup_metrics.get("penalty", 0.0)) > 50.0
                )
            ):
                second_rgb, second_alpha, second_metrics = (
                    clean_transparent_edge_halo(
                        cleaned_rgb,
                        cleaned_alpha,
                        action=action,
                    )
                )
                if float(second_metrics.get("penalty", 500.0)) < float(
                    cleanup_metrics.get("penalty", 500.0)
                ):
                    first_metrics = cleanup_metrics
                    cleaned_rgb = second_rgb
                    cleaned_alpha = second_alpha
                    cleanup_metrics = {
                        **second_metrics,
                        "applied": bool(first_metrics.get("applied"))
                        or bool(second_metrics.get("applied")),
                        "rgb_applied": bool(first_metrics.get("rgb_applied"))
                        or bool(second_metrics.get("rgb_applied")),
                        "alpha_applied": bool(
                            first_metrics.get("alpha_applied")
                        )
                        or bool(second_metrics.get("alpha_applied")),
                        "replaced_pixels": int(
                            first_metrics.get("replaced_pixels", 0)
                        )
                        + int(second_metrics.get("replaced_pixels", 0)),
                        "alpha_adjusted_pixels": int(
                            np.count_nonzero(cleaned_alpha != alpha)
                        ),
                        "candidate_pixels": int(
                            first_metrics.get("candidate_pixels", 0)
                        ),
                        "candidate_fraction": float(
                            first_metrics.get("candidate_fraction", 0.0)
                        ),
                        "contact_shadow": first_metrics.get(
                            "contact_shadow",
                            second_metrics.get("contact_shadow"),
                        ),
                        "cleanup_pass_count": 2,
                        "second_pass_selected": True,
                        "first_pass_penalty": float(
                            first_metrics.get("penalty", 0.0)
                        ),
                    }
            cleaned_rgb, cleaned_alpha, cleanup_metrics = (
                select_safe_transparent_edge_cleanup(
                    rgb,
                    alpha,
                    cleaned_rgb,
                    cleaned_alpha,
                    cleanup_metrics,
                )
            )
            guidance = normalized_guidance_mask(frame_index)
            cleaned_rgb, cleaned_alpha, post_repair_metrics = (
                repair_guided_alpha_interior_holes(
                    cleaned_rgb,
                    cleaned_alpha,
                    guidance,
                )
            )
            post_repair_metrics["frame_index"] = frame_index
            post_repair_metrics["stage"] = "post_edge_cleanup"
            interior_repair_metrics.append(post_repair_metrics)
            cleaned_extracted_frames.append((cleaned_rgb, cleaned_alpha))
            edge_cleanup_metrics.append(cleanup_metrics)
        extracted_frames = cleaned_extracted_frames

        drinking_static_prop_alpha_retentions: list[float] = []
        if drinking_static_prop_recovery_required:
            for index, (_rgb, alpha) in enumerate(extracted_frames):
                _guidance, static_prop_guidance = normalized_guidance_roles(
                    index
                )
                if static_prop_guidance is None:
                    raise RuntimeError(
                        "Drinking transparent extraction requires static bowl "
                        "guidance in every frame"
                    )
                static_prop_pixels = int(
                    np.count_nonzero(static_prop_guidance)
                )
                if static_prop_pixels < 64:
                    raise RuntimeError(
                        "Drinking transparent extraction static bowl guidance "
                        f"is empty at frame {index}"
                    )
                retention = float(
                    np.count_nonzero(
                        (alpha >= 128) & static_prop_guidance
                    )
                ) / float(static_prop_pixels)
                drinking_static_prop_alpha_retentions.append(retention)
                if retention < DRINKING_STATIC_BOWL_ALPHA_RETENTION_MIN:
                    raise RuntimeError(
                        "Drinking transparent extraction lost the static bowl "
                        f"at frame {index}: alpha_retention={retention:.4f}, "
                        "required="
                        f"{DRINKING_STATIC_BOWL_ALPHA_RETENTION_MIN:.4f}"
                    )

        sampled_indices = [
            min(
                len(source_frames) - 1,
                int(round(index * source_fps / float(gif_fps))),
            )
            for index in range(gif_frames)
        ]
        if action in {"drinking", "eating"}:
            extracted_frames, action_loop_metadata = (
                close_transparent_action_tail_loop(
                    extracted_frames,
                    target_frame_index=sampled_indices[-1],
                )
            )
            temporal_cycle = {
                **temporal_cycle,
                **action_loop_metadata,
                "action": action,
                "applied": True,
                "playback": "original_motion_with_geometry_safe_terminal_snap",
                "closed_loop": True,
            }
        sampled_original_source_indices = [
            source_index_map[source_index]
            for source_index in sampled_indices
        ]
        sampled_loop_quality = transparent_animation_loop_quality(
            extracted_frames,
            sampled_indices,
        )
        temporal_cycle["sampled_loop_quality"] = sampled_loop_quality
        loop_required = action in {"grooming", "drinking", "eating"}
        sampled_loop_acceptable = (
            bool(temporal_cycle["closed_loop"])
            and bool(temporal_cycle["preserves_source_frame_count"])
            and bool(sampled_loop_quality["acceptable"])
            if loop_required
            else True
        )
        if loop_required and not sampled_loop_acceptable:
            raise RuntimeError(
                "Action sampled temporal loop failed its continuity contract: "
                f"{json.dumps(temporal_cycle, ensure_ascii=False)}"
            )
        foreground_fractions: list[float] = []
        corner_alpha: list[int] = []
        for output_index, source_index in enumerate(sampled_indices):
            rgb, soft_alpha = extracted_frames[source_index]
            foreground_fractions.append(
                float(np.count_nonzero(soft_alpha >= 128))
                / float(soft_alpha.size)
            )
            corner_alpha.extend(
                [
                    int(soft_alpha[0, 0]),
                    int(soft_alpha[0, -1]),
                    int(soft_alpha[-1, 0]),
                    int(soft_alpha[-1, -1]),
                ]
            )
            _save_rgba(
                soft_dir / f"frame_{output_index:03d}.png",
                rgb,
                soft_alpha,
            )
            _save_rgba(
                binary_dir / f"frame_{output_index:03d}.png",
                rgb,
                _gif_alpha(soft_alpha),
            )
        if any(corner_alpha):
            raise RuntimeError(
                "Green-screen extraction left an opaque canvas corner"
            )
        if (
            min(foreground_fractions) < 0.02
            or max(foreground_fractions) > 0.85
        ):
            raise RuntimeError(
                "Green-screen foreground fraction is outside the safe range: "
                f"{min(foreground_fractions):.4f}.."
                f"{max(foreground_fractions):.4f}"
            )

        from imageio_ffmpeg import get_ffmpeg_exe

        gif_filter = (
            "[0:v]format=rgba,split[gif][pal];"
            "[pal]palettegen=reserve_transparent=1:"
            "transparency_color=0x000000:stats_mode=diff[p];"
            "[gif][p]paletteuse=dither=none:"
            "alpha_threshold=128:diff_mode=rectangle[v]"
        )
        _run_checked(
            [
                get_ffmpeg_exe(),
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                str(gif_fps),
                "-i",
                str(binary_dir / "frame_%03d.png"),
                "-filter_complex",
                gif_filter,
                "-map",
                "[v]",
                "-frames:v",
                str(gif_frames),
                "-loop",
                "0",
                "-final_delay",
                str(int(round(100 / gif_fps))),
                str(temporary),
            ],
            "Unstabilized transparent GIF encoding",
        )
        expected_foreground_masks = [
            _gif_alpha(extracted_frames[source_index][1]) >= 128
            for source_index in sampled_indices
        ]
        validation = _validate_transparent_gif(
            temporary,
            size=output_size,
            frame_count=gif_frames,
            duration_ms=int(gif_frames * 1000 / gif_fps),
            expected_foreground_masks=expected_foreground_masks,
        )
        with Image.open(temporary) as encoded_gif:
            published_frames: list[tuple[np.ndarray, np.ndarray]] = []
            for encoded_frame in ImageSequence.Iterator(encoded_gif):
                rgba = np.asarray(encoded_frame.convert("RGBA"))
                published_frames.append(
                    (rgba[:, :, :3].copy(), rgba[:, :, 3].copy())
                )
        published_loop_quality = transparent_animation_loop_quality(
            published_frames,
            list(range(len(published_frames))),
        )
        temporal_cycle["published_loop_quality"] = published_loop_quality
        temporal_cycle["acceptable"] = (
            bool(sampled_loop_acceptable)
            and bool(published_loop_quality["acceptable"])
            if loop_required
            else None
        )
        if loop_required and temporal_cycle["acceptable"] is not True:
            raise RuntimeError(
                "Action published temporal loop failed its continuity contract: "
                f"{json.dumps(temporal_cycle, ensure_ascii=False)}"
            )
        os.replace(temporary, gif_target)
        apng_validation: dict[str, Any] | None = None
        apng_error: str | None = None
        if soft_apng_target is not None and apng_temporary is not None:
            try:
                _encode_soft_alpha_apng(
                    (
                        soft_dir / f"frame_{index:03d}.png"
                        for index in range(gif_frames)
                    ),
                    apng_temporary,
                    fps=gif_fps,
                )
                apng_validation = validate_animation(
                    apng_temporary,
                    size=output_size,
                    frame_count=gif_frames,
                    duration_ms=int(gif_frames * 1000 / gif_fps),
                )
                soft_apng_target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(apng_temporary, soft_apng_target)
            except Exception as exc:
                apng_error = f"{type(exc).__name__}: {exc}"
                apng_temporary.unlink(missing_ok=True)
        white_mp4_validation: dict[str, Any] | None = None
        if white_mp4_target is not None and white_temporary is not None:
            white_frames: list[Image.Image] = []
            for rgb, soft_alpha in extracted_frames:
                alpha_float = (
                    soft_alpha.astype(np.float32)[..., None] / 255.0
                )
                composite = np.clip(
                    rgb.astype(np.float32) * alpha_float
                    + 255.0 * (1.0 - alpha_float),
                    0,
                    255,
                ).astype(np.uint8)
                white_frames.append(Image.fromarray(composite))
            encode_video(
                white_frames,
                white_temporary,
                source_fps,
                lossless_rgb=False,
            )
            white_mp4_validation = _validate_white_background_mp4(
                white_temporary,
                size=output_size,
                frame_count=len(source_frames),
            )
            white_mp4_validation.update(
                fps=source_fps,
                duration_seconds=len(source_frames) / source_fps,
            )
            white_mp4_target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(white_temporary, white_mp4_target)
        return {
            "mode": "per_frame_foreground_extraction_no_spatial_stabilization",
            "extraction_method": extraction_method,
            "reference_aligned_uniform_background": {
                "requested": bool(reference_aligned_uniform_background),
                "applied": (
                    extraction_method
                    == REFERENCE_ALIGNED_UNIFORM_BACKGROUND_POLICY
                ),
                "policy": REFERENCE_ALIGNED_UNIFORM_BACKGROUND_POLICY,
                "border_distance_p999_max": max(
                    reference_aligned_border_distance_p999_values,
                    default=None,
                ),
            },
            "reference_aligned_extraction": {
                "requested": bool(reference_aligned_uniform_background),
                "policy": REFERENCE_ALIGNED_ADAPTIVE_EXTRACTION_POLICY,
                "strategy": reference_aligned_extraction_strategy,
                "extraction_method": extraction_method,
                "canvas_border_distance_p999_min": min(
                    reference_aligned_border_distance_p999_values,
                    default=None,
                ),
                "canvas_border_distance_p999_max": max(
                    reference_aligned_border_distance_p999_values,
                    default=None,
                ),
                "bounded_canvas_selection_max": (
                    REFERENCE_ALIGNED_BOUNDED_CANVAS_BORDER_DISTANCE_P999_MAX
                ),
                "hybrid_canvas_selection_max": (
                    REFERENCE_ALIGNED_HYBRID_CANVAS_BORDER_DISTANCE_P999_MAX
                ),
                "bounded_guidance_coverage_min": min(
                    (
                        item.get("guidance_coverage", 1.0)
                        for item in reference_aligned_bounded_metrics
                    ),
                    default=None,
                ),
                "bounded_area_ratio_min": min(
                    (
                        item.get("area_ratio", 1.0)
                        for item in reference_aligned_bounded_metrics
                    ),
                    default=None,
                ),
            },
            "drinking_static_bowl_retention": {
                "policy": DRINKING_STATIC_BOWL_ALPHA_RETENTION_POLICY,
                "required": drinking_static_prop_recovery_required,
                "acceptable": (
                    min(
                        drinking_static_prop_alpha_retentions,
                        default=0.0,
                    )
                    >= DRINKING_STATIC_BOWL_ALPHA_RETENTION_MIN
                    if drinking_static_prop_recovery_required
                    else None
                ),
                "alpha_retention_min": min(
                    drinking_static_prop_alpha_retentions,
                    default=None,
                ),
                "alpha_retention_required_min": (
                    DRINKING_STATIC_BOWL_ALPHA_RETENTION_MIN
                ),
                "input_alpha_retention_min": min(
                    (
                        item["static_prop_input_retention_ratio"]
                        for item in static_prop_recovery_metrics
                    ),
                    default=None,
                ),
                "recovered_pixels_total": int(
                    sum(
                        item["static_prop_recovered_pixels"]
                        for item in static_prop_recovery_metrics
                    )
                ),
                "recovery_dilation_radius": (
                    DRINKING_STATIC_BOWL_RECOVERY_DILATION_RADIUS
                ),
            },
            "interior_alpha_repair": {
                "policy": "guided_enclosed_alpha_interior_repair_v1",
                "visual_advisory_only": True,
                "completion_blocked": False,
                "passes_per_source_frame": 2,
                "applied_passes": sum(
                    1
                    for item in interior_repair_metrics
                    if bool(item.get("applied"))
                ),
                "repaired_pixels_total": sum(
                    int(item.get("repaired_pixels", 0))
                    for item in interior_repair_metrics
                ),
                "repaired_pixels_max": max(
                    (
                        int(item.get("repaired_pixels", 0))
                        for item in interior_repair_metrics
                    ),
                    default=0,
                ),
                "repaired_components_total": sum(
                    int(item.get("repaired_components", 0))
                    for item in interior_repair_metrics
                ),
                "exterior_background_preserved": True,
            },
            "silhouette_edge_cleanup": {
                "policy": "local_inward_color_alpha_dark_halo_v2",
                "action": action,
                "applied_frames": sum(
                    1
                    for item in edge_cleanup_metrics
                    if bool(item["applied"])
                ),
                "source_frames": len(edge_cleanup_metrics),
                "replaced_pixels_total": sum(
                    int(item["replaced_pixels"])
                    for item in edge_cleanup_metrics
                ),
                "replaced_pixels_max": max(
                    (
                        int(item["replaced_pixels"])
                        for item in edge_cleanup_metrics
                    ),
                    default=0,
                ),
                "candidate_fraction_max": max(
                    (
                        float(item["candidate_fraction"])
                        for item in edge_cleanup_metrics
                    ),
                    default=0.0,
                ),
                "alpha_adjusted_pixels_total": sum(
                    int(item.get("alpha_adjusted_pixels", 0))
                    for item in edge_cleanup_metrics
                ),
                "contact_shadow_removed_pixels_total": sum(
                    int(
                        (
                            item.get("contact_shadow")
                            if isinstance(item.get("contact_shadow"), dict)
                            else {}
                        ).get("removed_pixels", 0)
                    )
                    for item in edge_cleanup_metrics
                ),
                "contact_shadow_removed_pixels_max": max(
                    (
                        int(
                            (
                                item.get("contact_shadow")
                                if isinstance(
                                    item.get("contact_shadow"),
                                    dict,
                                )
                                else {}
                            ).get("removed_pixels", 0)
                        )
                        for item in edge_cleanup_metrics
                    ),
                    default=0,
                ),
                "acceptable_frames": sum(
                    1
                    for item in edge_cleanup_metrics
                    if bool(item.get("acceptable"))
                ),
                "second_pass_frames": sum(
                    1
                    for item in edge_cleanup_metrics
                    if bool(item.get("second_pass_selected"))
                ),
                "rolled_back_frames": sum(
                    1
                    for item in edge_cleanup_metrics
                    if bool(item.get("rolled_back"))
                ),
                "unacceptable_cleanup_published_frames": 0,
                "completion_blocked_by_cleanup": False,
                "residual_candidate_fraction_max": max(
                    (
                        float(
                            (
                                item.get("after")
                                if isinstance(item.get("after"), dict)
                                else {}
                            ).get("candidate_fraction", 0.0)
                        )
                        for item in edge_cleanup_metrics
                    ),
                    default=0.0,
                ),
            },
            "source_frames": len(source_frames),
            "source_fps": source_fps,
            "source_phase": {
                "policy": "validated_action_contact_phase_rotation_v1",
                "start_frame_index": phase_start_index,
                "rotated": bool(phase_start_index),
            },
            "temporal_cycle": temporal_cycle,
            "output_playback": {
                "policy": "post_generation_pts_only_playback_rate_v1",
                "video_fps": source_fps,
                "gif_fps": gif_fps,
                "gif_frames": gif_frames,
                "gif_duration_seconds": gif_frames / float(gif_fps),
            },
            "sampled_source_indices": sampled_indices,
            "sampled_original_source_indices": sampled_original_source_indices,
            "foreground_fraction_min": min(foreground_fractions),
            "foreground_fraction_max": max(foreground_fractions),
            "semantic_metrics": {
                "guidance_component_coverage_policy": (
                    "dynamic_drinking_running_limb_proximity_v1"
                    if normalized_action in {"drinking", "running"}
                    else (
                        "reference_aligned_subject_and_tray_component_v2"
                        if reference_aligned_uniform_background
                        else "static_action_component_v1"
                    )
                ),
                "guidance_component_coverage_required_min": (
                    guidance_component_coverage_required_min
                ),
                "guidance_component_area_fraction_required_min": (
                    guidance_component_area_fraction_required_min
                ),
                "guidance_dynamic_component_area_fraction_max": (
                    guidance_dynamic_component_area_fraction_max
                ),
                "guidance_dynamic_component_radius": (
                    guidance_dynamic_component_radius_px
                ),
                "guidance_component_direct_coverage_min": min(
                    (
                        item["guidance_component_direct_coverage_min"]
                        for item in semantic_metrics
                    ),
                    default=None,
                ),
                "guidance_dynamic_component_count_max": max(
                    (
                        item["guidance_dynamic_component_count"]
                        for item in semantic_metrics
                    ),
                    default=None,
                ),
                "guidance_dynamic_component_proximity_coverage_min": min(
                    (
                        item[
                            "guidance_dynamic_component_proximity_coverage_min"
                        ]
                        for item in semantic_metrics
                        if item["guidance_dynamic_component_count"] > 0
                    ),
                    default=None,
                ),
                "ignored_guidance_component_fraction_max": max(
                    (
                        item["ignored_guidance_component_fraction"]
                        for item in semantic_metrics
                    ),
                    default=None,
                ),
                "ignored_guidance_component_count_max": max(
                    (
                        item["ignored_guidance_component_count"]
                        for item in semantic_metrics
                    ),
                    default=None,
                ),
                "guidance_coverage_min": min(
                    (
                        item["guidance_coverage"]
                        for item in semantic_metrics
                    ),
                    default=None,
                ),
                "guidance_component_coverage_min": min(
                    (
                        item["guidance_component_coverage_min"]
                        for item in semantic_metrics
                    ),
                    default=None,
                ),
                "area_ratio_min": min(
                    (item["area_ratio"] for item in semantic_metrics),
                    default=None,
                ),
                "area_ratio_max": max(
                    (item["area_ratio"] for item in semantic_metrics),
                    default=None,
                ),
                "area_ratio_required_min": guidance_area_ratio_required_min,
                "toileting_tray_retention_required": (
                    normalized_action == "toileting"
                ),
                "drinking_bowl_retention_required": (
                    drinking_static_prop_recovery_required
                ),
                "drinking_bowl_alpha_retention_min": min(
                    drinking_static_prop_alpha_retentions,
                    default=None,
                ),
                "drinking_bowl_alpha_retention_required_min": (
                    DRINKING_STATIC_BOWL_ALPHA_RETENTION_MIN
                ),
                "semantic_core_retention_min": min(
                    (
                        item["semantic_core_retention"]
                        for item in semantic_metrics
                    ),
                    default=None,
                ),
                "semantic_upper_core_retention_min": min(
                    (
                        item["semantic_upper_core_retention"]
                        for item in semantic_metrics
                    ),
                    default=None,
                ),
                "semantic_grid_core_retention_min": min(
                    (
                        item["semantic_grid_core_retention_min"]
                        for item in semantic_metrics
                    ),
                    default=None,
                ),
            },
            "gif_edge": {
                "policy": "clean_hard_alpha_v1",
                "ordered_alpha_dither": False,
                "alpha_threshold": GIF_ALPHA_THRESHOLD,
                "alpha_smoothing_sigma": GIF_ALPHA_SMOOTHING_SIGMA,
                "palette_dither": "none",
            },
            "gif": validation,
            "apng": apng_validation,
            "apng_error": apng_error,
            "white_background_mp4": white_mp4_validation,
        }
    finally:
        temporary.unlink(missing_ok=True)
        if white_temporary is not None:
            white_temporary.unlink(missing_ok=True)
        if apng_temporary is not None:
            apng_temporary.unlink(missing_ok=True)
        shutil.rmtree(frames_root, ignore_errors=True)


def encode_video(
    frames: list[Image.Image],
    output_path: Path,
    fps: float,
    lossless_rgb: bool = False,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(output_path), mode="w")
    try:
        codec = "libx264rgb" if lossless_rgb else "libx264"
        stream = container.add_stream(codec, rate=max(1, int(round(fps))))
        stream.width = frames[0].width
        stream.height = frames[0].height
        stream.pix_fmt = "rgb24" if lossless_rgb else "yuv420p"
        stream.options = (
            {"crf": "0", "preset": "medium"}
            if lossless_rgb
            else {"crf": "12", "preset": "medium"}
        )
        for image in frames:
            video_frame = av.VideoFrame.from_ndarray(np.asarray(image.convert("RGB")), format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def encode_static_mask_video(
    mask_path: Path,
    output_path: Path,
    *,
    frame_count: int,
    fps: float,
) -> dict[str, Any]:
    """Repeat one immutable RGB identity mask in a lossless video stream."""

    if av is None:
        raise RuntimeError("PyAV is required to encode a static mask video")
    if frame_count < 1 or fps <= 0:
        raise ValueError("Static mask video timing must be positive")
    if not mask_path.is_file():
        raise FileNotFoundError(f"Static mask source is missing: {mask_path}")
    with Image.open(mask_path) as loaded:
        source_mask = np.asarray(loaded.convert("RGB")).copy()
    blue = (
        (source_mask[:, :, 2] >= 225)
        & (source_mask[:, :, 0] <= 30)
        & (source_mask[:, :, 1] <= 30)
    )
    red = (
        (source_mask[:, :, 0] >= 225)
        & (source_mask[:, :, 1] <= 30)
        & (source_mask[:, :, 2] <= 30)
    )
    mask = np.full(source_mask.shape, 255, dtype=np.uint8)
    mask[blue] = (0, 0, 255)
    mask[red] = (255, 0, 0)
    if mask.shape[:2] != (512, 512):
        raise ValueError(
            "Static subject guidance mask must be 512x512, got "
            f"{mask.shape[1]}x{mask.shape[0]}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        f".{output_path.stem}.{uuid.uuid4().hex}.mp4"
    )
    try:
        container = av.open(str(temporary), mode="w")
        try:
            stream = container.add_stream(
                "libx264rgb",
                rate=max(1, int(round(fps))),
            )
            stream.width = int(mask.shape[1])
            stream.height = int(mask.shape[0])
            stream.pix_fmt = "rgb24"
            stream.options = {"crf": "0", "preset": "medium"}
            for _ in range(frame_count):
                video_frame = av.VideoFrame.from_ndarray(mask, format="rgb24")
                for packet in stream.encode(video_frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        finally:
            container.close()
        verification = verify_mask_video(temporary, frame_count)
        if float(verification.get("blue_fraction") or 0.0) < 0.02:
            raise ValueError("Static subject guidance mask has no usable blue subject")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, output_path)
        return {
            "policy": "lossless_rgb_static_reference_mask_video_v1",
            "source": str(mask_path),
            "source_sha256": sha256_file(mask_path),
            "normalization": "blue_red_roles_on_pure_white_background_v1",
            "output": str(output_path),
            "output_sha256": sha256_file(output_path),
            "frames": frame_count,
            "fps": fps,
            **verification,
        }
    finally:
        temporary.unlink(missing_ok=True)


def verify_mask_video(path: Path, expected_count: int) -> dict[str, Any]:
    blue_count = 0
    red_count = 0
    white_count = 0
    unknown_count = 0
    decoded_count = 0
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            pixels = frame.to_ndarray(format="rgb24")
            blue = (pixels[..., 2] >= 245) & (pixels[..., 0] <= 10) & (pixels[..., 1] <= 10)
            red = (pixels[..., 0] >= 245) & (pixels[..., 1] <= 10) & (pixels[..., 2] <= 10)
            white = np.all(pixels >= 245, axis=2)
            blue_count += int(np.count_nonzero(blue))
            red_count += int(np.count_nonzero(red))
            white_count += int(np.count_nonzero(white))
            unknown_count += int(
                pixels.shape[0] * pixels.shape[1]
                - np.count_nonzero(blue | red | white)
            )
            decoded_count += 1
    if decoded_count != expected_count:
        raise ValueError(
            f"Mask video decoded {decoded_count} frames; expected {expected_count}"
        )
    total = blue_count + red_count + white_count + unknown_count
    unknown_fraction = unknown_count / max(1, total)
    if unknown_fraction > 0.02:
        raise ValueError(
            f"Mask codec altered too many identity pixels: {unknown_fraction:.4%}"
        )
    return {
        "decoded_frames": decoded_count,
        "blue_fraction": blue_count / max(1, total),
        "red_fraction": red_count / max(1, total),
        "white_fraction": white_count / max(1, total),
        "unknown_fraction": unknown_fraction,
    }


def relative_input_name(path: Path) -> str:
    return path.resolve().relative_to(COMFY_INPUT.resolve()).as_posix()


def build_prompt(
    pose_video: str,
    pose_mask_video: str,
    reference_image: str,
    reference_mask: str,
    args: argparse.Namespace,
    output_prefix: str,
) -> dict[str, Any]:
    graph: dict[str, Any] = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": UNET_NAME, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["1", 0],
                "lora_name": DPO_LORA,
                "strength_model": 1.0,
            },
        },
        "3": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["2", 0],
                "lora_name": DISTILL_LORA,
                "strength_model": 0.8,
            },
        },
        "4": {
            "class_type": "ModelSamplingSD3",
            "inputs": {"model": ["3", 0], "shift": 5.0},
        },
        "5": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": TEXT_ENCODER,
                "type": "wan",
                "device": "default",
            },
        },
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_NAME}},
        "7": {
            "class_type": "CLIPVisionLoader",
            "inputs": {"clip_name": CLIP_VISION},
        },
        "8": {"class_type": "LoadVideo", "inputs": {"file": pose_video}},
        "9": {"class_type": "GetVideoComponents", "inputs": {"video": ["8", 0]}},
        "10": {"class_type": "LoadVideo", "inputs": {"file": pose_mask_video}},
        "11": {"class_type": "GetVideoComponents", "inputs": {"video": ["10", 0]}},
        "12": {"class_type": "LoadImage", "inputs": {"image": reference_image}},
        "13": {"class_type": "LoadImage", "inputs": {"image": reference_mask}},
        "14": {
            "class_type": "CLIPVisionEncode",
            "inputs": {
                "clip_vision": ["7", 0],
                "image": ["12", 0],
                "crop": "none",
            },
        },
        "15": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": args.prompt, "clip": ["5", 0]},
        },
        "16": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": args.negative_prompt, "clip": ["5", 0]},
        },
        "17": {
            "class_type": "WanSCAILToVideo",
            "inputs": {
                "positive": ["15", 0],
                "negative": ["16", 0],
                "vae": ["6", 0],
                "width": args.width,
                "height": args.height,
                "length": args.frames,
                "batch_size": 1,
                "pose_strength": args.pose_strength,
                "pose_start": 0.0,
                "pose_end": 1.0,
                "video_frame_offset": 0,
                "previous_frame_count": 5,
                "pose_video": ["9", 0],
                "pose_video_mask": ["11", 0],
                "replacement_mode": True,
                "reference_image": ["12", 0],
                "reference_image_mask": ["13", 0],
                "clip_vision_output": ["14", 0],
            },
        },
        "18": {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": "euler"},
        },
        "19": {
            "class_type": "BasicScheduler",
            "inputs": {
                "model": ["3", 0],
                "scheduler": "simple",
                "steps": args.steps,
                "denoise": 1.0,
            },
        },
        "20": {
            "class_type": "SamplerCustom",
            "inputs": {
                "model": ["4", 0],
                "add_noise": True,
                "noise_seed": args.seed,
                "cfg": 1.0,
                "positive": ["17", 0],
                "negative": ["17", 1],
                "sampler": ["18", 0],
                "sigmas": ["19", 0],
                "latent_image": ["17", 2],
            },
        },
        "21": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["20", 1], "vae": ["6", 0]},
        },
        "22": {
            "class_type": "CreateVideo",
            "inputs": {"images": ["21", 0], "fps": args.fps, "bit_depth": 8},
        },
        "23": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["22", 0],
                "filename_prefix": output_prefix,
                "format": "mp4",
                "codec": "h264",
            },
        },
        "24": {
            "class_type": "SaveAnimatedWEBP",
            "inputs": {
                "images": ["21", 0],
                "filename_prefix": output_prefix + "_preview",
                "fps": args.fps,
                "lossless": False,
                "quality": 88,
                "method": "default",
            },
        },
    }
    return graph


def extract_output_files(value: Any) -> list[dict[str, str]]:
    files: list[dict[str, str]] = []
    if isinstance(value, dict):
        if {"filename", "subfolder"}.issubset(value):
            files.append(
                {
                    "filename": str(value["filename"]),
                    "subfolder": str(value.get("subfolder", "")),
                    "type": str(value.get("type", "output")),
                }
            )
        for nested in value.values():
            files.extend(extract_output_files(nested))
    elif isinstance(value, list):
        for nested in value:
            files.extend(extract_output_files(nested))
    return files


def wait_for_prompt(prompt_id: str, timeout_seconds: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_state = ""
    while time.monotonic() < deadline:
        history = api_json(f"/history/{prompt_id}")
        if prompt_id in history:
            entry = history[prompt_id]
            status = entry.get("status", {})
            state = str(status.get("status_str", ""))
            if state and state != last_state:
                print(f"ComfyUI status: {state}", flush=True)
                last_state = state
            if status.get("completed"):
                return entry
            messages = status.get("messages", [])
            for message in messages:
                if isinstance(message, list) and message and message[0] == "execution_error":
                    raise RuntimeError(json.dumps(message[1], ensure_ascii=False, indent=2))
        time.sleep(5)
    raise TimeoutError(f"ComfyUI prompt {prompt_id} exceeded {timeout_seconds}s")


def gpu_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, timeout=15).strip()
    except Exception as error:  # diagnostics must not block the test
        return {"error": repr(error)}
    values = [value.strip() for value in output.split(",")]
    keys = ["name", "memory_total_mib", "memory_used_mib", "memory_free_mib", "gpu_util_percent"]
    return dict(zip(keys, values))


def main() -> int:
    args = parse_args()
    supported_actions = {
        "cat": {"eating", "grooming"},
        "dog": {
            "eating",
            "drinking",
            "running",
            "sleeping",
            "standing",
            "toileting",
        },
    }
    if args.action not in supported_actions[args.species]:
        raise ValueError(
            f"{args.species} does not support the {args.action} action"
        )
    if args.width % 32 or args.height % 32:
        raise ValueError("SCAIL width and height must be multiples of 32")
    if args.frames < 1 or (args.frames - 1) % 4:
        raise ValueError("SCAIL frame count must be 1 + 4*n (for example 33, 65, 81)")
    if args.action == "eating" and args.pose_strength > EATING_POSE_STRENGTH_MAX:
        raise ValueError(
            "Eating pose strength exceeds the safe fur-silhouette limit "
            f"{EATING_POSE_STRENGTH_MAX:.2f}."
        )
    if args.eating_geometry_guide_metadata and not args.eating_geometry_guide_output:
        raise ValueError(
            "--eating-geometry-guide-metadata requires "
            "--eating-geometry-guide-output"
        )
    if (
        args.grooming_geometry_guide_metadata
        and not args.grooming_geometry_guide_output
    ):
        raise ValueError(
            "--grooming-geometry-guide-metadata requires "
            "--grooming-geometry-guide-output"
        )
    if (
        args.eating_geometry_guide_output
        and args.grooming_geometry_guide_output
    ):
        raise ValueError(
            "Eating and grooming geometry-guide output modes are mutually "
            "exclusive."
        )
    if args.eating_geometry_guide_output:
        if args.action != "eating":
            raise ValueError(
                "--eating-geometry-guide-output requires --action eating"
            )
        require_files([args.source])
        guide, guide_metadata = make_eating_geometry_guide_from_gif(
            args.source,
            (args.width, args.height),
            output_count=args.frames,
            output_fps=args.fps,
        )
        args.eating_geometry_guide_output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        guide.save(
            args.eating_geometry_guide_output,
            format="PNG",
            optimize=True,
        )
        if args.eating_geometry_guide_metadata:
            args.eating_geometry_guide_metadata.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            args.eating_geometry_guide_metadata.write_text(
                json.dumps(guide_metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        print(json.dumps(guide_metadata, ensure_ascii=False), flush=True)
        return 0
    if args.grooming_geometry_guide_output:
        if args.action != "grooming":
            raise ValueError(
                "--grooming-geometry-guide-output requires --action grooming"
            )
        require_files([args.source])
        guide, guide_metadata = make_grooming_geometry_guide_from_animation(
            args.source,
            (args.width, args.height),
            output_count=args.frames,
            output_fps=args.fps,
        )
        args.grooming_geometry_guide_output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        guide.save(
            args.grooming_geometry_guide_output,
            format="PNG",
            optimize=True,
        )
        if args.grooming_geometry_guide_metadata:
            args.grooming_geometry_guide_metadata.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            args.grooming_geometry_guide_metadata.write_text(
                json.dumps(guide_metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        print(json.dumps(guide_metadata, ensure_ascii=False), flush=True)
        return 0
    require_files([args.source, args.reference])
    if args.run_id and not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}",
        args.run_id,
    ):
        raise ValueError("run-id contains unsupported characters")
    run_id = args.run_id or (
        time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        + "-"
        + uuid.uuid4().hex[:8]
    )
    run_dir = EXPERIMENT_ROOT / run_id
    input_dir = COMFY_INPUT / "scail2_test" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    input_dir.mkdir(parents=True, exist_ok=False)

    white_maquette_cleanup = args.species == "dog" and args.action != "eating"
    frames, durations = read_animation_frames(
        args.source,
        extract_green_screen=not white_maquette_cleanup,
    )
    sampled = sample_timed_frames(frames, durations, args.frames, args.fps)
    reference, reference_mask, reference_metadata = make_reference_assets(
        args.reference,
        (args.width, args.height),
        separate_eating_prop=(
            args.action == "eating"
            and not args.canonical_identity_full_driver_eating_fallback
        ),
        separate_drinking_prop=args.action == "drinking",
        separate_toileting_prop=args.action == "toileting",
        contain_fraction=(
            0.88
            if args.action in {"grooming", "sleeping", "standing", "toileting"}
            else 0.84
        ),
    )
    if args.action == "eating":
        if args.species == "dog":
            if args.canonical_identity_full_driver_eating_fallback:
                pose_frames, pose_masks, pose_metadata = (
                    make_dog_eating_canonical_full_driver_pose_frames(
                        sampled,
                        (args.width, args.height),
                    )
                )
            else:
                try:
                    pose_frames, pose_masks, pose_metadata = (
                        make_eating_reference_aligned_pose_frames(
                            sampled,
                            reference,
                            reference_mask,
                            (args.width, args.height),
                        )
                    )
                except ValueError as exc:
                    if not str(exc).startswith(
                        "eating_reference_motion_unreliable:"
                    ):
                        raise
                    pose_frames, pose_masks, pose_metadata = (
                        make_dog_eating_reference_full_driver_recovery_pose_frames(
                            sampled,
                            (args.width, args.height),
                            fallback_reason=str(exc),
                        )
                    )
        else:
            pose_frames, pose_masks, pose_metadata = (
                make_full_eating_pose_and_mask_frames(
                    sampled,
                    (args.width, args.height),
                )
            )
    elif args.action == "drinking" and args.species == "dog":
        try:
            pose_frames, pose_masks, pose_metadata = (
                make_full_drinking_pose_and_mask_frames(
                    sampled,
                    (args.width, args.height),
                )
            )
        except ValueError as exc:
            if not str(exc).startswith(
                "drinking_full_driver_fallback_unreliable:"
            ):
                raise
            pose_frames, pose_masks, pose_metadata = (
                make_static_reference_drinking_pose_frames(
                    reference,
                    reference_mask,
                    (args.width, args.height),
                    frame_count=args.frames,
                    fallback_reason=str(exc),
                )
            )
    else:
        pose_frames, pose_masks, pose_metadata = make_pose_and_mask_frames(
            sampled,
            (args.width, args.height),
            pose_background=(255, 255, 255),
            separate_eating_prop=False,
            separate_toileting_prop=args.action == "toileting",
            white_maquette_cleanup=white_maquette_cleanup,
        )
    prop_metadata = pose_metadata.get("scene_prop_metadata")
    if isinstance(prop_metadata, dict):
        prop_metadata["source_sha256"] = sha256_file(args.source)

    pose_path = input_dir / "tm_pose.mp4"
    pose_mask_path = input_dir / "tm_identity_mask.mp4"
    reference_path = input_dir / "reference.png"
    reference_mask_path = input_dir / "reference_identity_mask.png"
    encode_video(pose_frames, pose_path, args.fps, lossless_rgb=False)
    encode_video(pose_masks, pose_mask_path, args.fps, lossless_rgb=True)
    reference.save(reference_path)
    reference_mask.save(reference_mask_path)
    pose_frames[0].save(run_dir / "pose_first_frame.png")
    pose_masks[0].save(run_dir / "pose_mask_first_frame.png")
    reference.save(run_dir / "reference_normalized.png")
    reference_mask.save(run_dir / "reference_mask.png")
    mask_codec_metadata = verify_mask_video(pose_mask_path, args.frames)

    output_prefix = f"scail2_test/{run_id}/subject_replacement"
    graph = build_prompt(
        relative_input_name(pose_path),
        relative_input_name(pose_mask_path),
        relative_input_name(reference_path),
        relative_input_name(reference_mask_path),
        args,
        output_prefix,
    )
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": str(args.source),
        "reference": str(args.reference),
        "width": args.width,
        "height": args.height,
        "frames": args.frames,
        "fps": args.fps,
        "steps": args.steps,
        "seed": args.seed,
        "action": args.action,
        "species": args.species,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "pose_strength": args.pose_strength,
        "transparent_gif_requested": args.transparent_gif,
        "models": {
            "unet": UNET_NAME,
            "dpo_lora": {"name": DPO_LORA, "strength": 1.0},
            "distill_lora": {"name": DISTILL_LORA, "strength": 0.8},
            "text_encoder": TEXT_ENCODER,
            "vae": VAE_NAME,
            "clip_vision": CLIP_VISION,
        },
        "pose_metadata": pose_metadata,
        "scene_prop_metadata": prop_metadata,
        "reference_metadata": reference_metadata,
        "mask_codec_metadata": mask_codec_metadata,
        "gpu_before": gpu_snapshot(),
        "input_files": {
            "pose_video": str(pose_path),
            "pose_mask_video": str(pose_mask_path),
            "reference": str(reference_path),
            "reference_mask": str(reference_mask_path),
        },
        "output_prefix": output_prefix,
    }
    (run_dir / "workflow_api.json").write_text(
        json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Prepared test inputs: {run_dir}", flush=True)
    if args.prepare_only:
        return 0

    try:
        object_info = api_json("/object_info")
        missing_nodes = sorted(set(node["class_type"] for node in graph.values()) - set(object_info))
        if missing_nodes:
            raise RuntimeError("Missing ComfyUI node classes: " + ", ".join(missing_nodes))

        started = time.monotonic()
        response = api_json("/prompt", {"prompt": graph, "client_id": run_id})
        if response.get("node_errors"):
            raise RuntimeError(
                "ComfyUI graph validation failed: "
                + json.dumps(response["node_errors"], ensure_ascii=False, indent=2)
            )
        prompt_id = str(response["prompt_id"])
        manifest["prompt_id"] = prompt_id
        manifest["queue_number"] = response.get("number")
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Submitted ComfyUI prompt: {prompt_id}", flush=True)

        history = wait_for_prompt(prompt_id, args.timeout)
        elapsed = time.monotonic() - started
        output_files = extract_output_files(history.get("outputs", {}))
        transparent_gif: dict[str, Any] | None = None
        if args.transparent_gif:
            video_items = [
                item
                for item in output_files
                if item["filename"].lower().endswith(".mp4")
                and item.get("type") == "output"
            ]
            if not video_items:
                raise RuntimeError(
                    "Transparent GIF was requested but no MP4 output was found"
                )
            item = video_items[-1]
            video_path = (
                COMFY_OUTPUT
                / item.get("subfolder", "")
                / item["filename"]
            )
            gif_path = video_path.with_name(
                video_path.stem + "_transparent_512x512_5s.gif"
            )
            apng_path = video_path.with_name(
                video_path.stem + "_transparent_512x512_5s_soft.png"
            )
            gif_metrics = build_unstabilized_transparent_gif(
                video_path,
                gif_path,
                action=args.action,
                guidance_mask_video=pose_mask_path,
                output_size=(args.width, args.height),
                source_fps=args.fps,
                soft_apng_target=apng_path,
            )
            transparent_gif = {
                "path": str(gif_path),
                "soft_apng_path": str(apng_path),
                "metrics": gif_metrics,
            }
        manifest.update(
            {
                "status": "completed",
                "elapsed_seconds": round(elapsed, 3),
                "gpu_after": gpu_snapshot(),
                "outputs": output_files,
                "transparent_gif": transparent_gif,
                "history_status": history.get("status", {}),
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
    except Exception as error:
        manifest.update(
            {
                "status": "failed",
                "error": repr(error),
                "gpu_after": gpu_snapshot(),
            }
        )
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
