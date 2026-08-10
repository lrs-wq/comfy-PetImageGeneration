#!/usr/bin/env python3
"""Build a CPU-only, one-shot Qwen canonical-reference validation bundle.

This builder never calls ComfyUI's /prompt endpoint.  It reads only the frozen
Main_case driver alpha silhouettes, creates a fixed semantic full-body guide,
and writes a workflow whose Image 1 is the uploaded identity and whose Image 2
and sampler latent are the semantic guide.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

PROJECT_IMPORT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_IMPORT_ROOT))

from src.comfy_pet_pipeline.pipeline import subject_replacement as scail  # noqa: E402


RUN_ID = "codex-upload-identity-main-case-canonical-v1-20260731-r2"
POLICY = "upload_identity_main_case_canonical_v1"
POLICY_VERSION = "1.0.0"
EXPECTED_UPLOAD_SHA256 = (
    "beafff6ceddb78ec7440cbd7c5c8b925c6ede12ea53903ac9d5f822e2b9892ac"
)
EXPECTED_UPLOAD_SOURCE_SHA256 = (
    "d8c377e76cf25460d756d18149017b73328b9fbd3bcd7370b35cd334f85a2a31"
)
EXPECTED_EAT_SOURCE_SHA256 = (
    "ea893acad01481dcd2c96e7b115478d1e82ad3785ad8d641296aecb7a25986b7"
)
EXPECTED_EAT_RUNTIME_SHA256 = (
    "07cf90c34ecca9742cf10c7a0290274ec1b9e5c1a008d73335836aec53d49b46"
)
EXPECTED_TM_SOURCE_SHA256 = (
    "b03d7e5ebb5aa878d1f57fff389dfa241d7ff8d7d273df1fd264bb479311aedd"
)
EXPECTED_TM_RUNTIME_SHA256 = (
    "2fc8ed9a398706fb2bf88b43f340be1a555d7dc638073a43e8f89f5c15ec0993"
)

QWEN_MODEL = "qwen_image_edit_2511_fp8mixed.safetensors"
QWEN_CLIP = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
QWEN_VAE = "qwen_image_vae.safetensors"
QWEN_LIGHTNING_LORA = (
    "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
)
QWEN_PET3D_LORA = "pet3d_cat_qwen_image_r16.safetensors"
QWEN_PET3D_STRENGTH = 0.15
QWEN_SEED = 20260726
QWEN_STEPS = 6

CANVAS = 512
GREEN = (0, 255, 0)
FIXED_LUMA = {
    "base": 144,
    "torso": 152,
    "tail": 128,
    "hind_leg": 168,
    "hind_paw": 184,
    "head": 176,
    "ear": 136,
    "foreleg": 172,
    "forepaw": 192,
    "eye": 88,
    "muzzle": 208,
    "nose": 112,
}

POSITIVE_PROMPT = """Create one neutral canonical full-body premium stylized 3D cat on an exactly uniform pure #00FF00 background.

IMAGE ROLES — non-negotiable. Image 1 is the original uploaded pet and is the only identity reference. Preserve from Image 1 the exact recognizable face and head shape, eye color and spacing, ear shape, nose and muzzle, fur base colors, and visible marking placement. Image 1 does not control crop, framing, body scale, pose, coat volume, or background. Never extrapolate its close-up crop into a giant head, portrait, mane, ruff, plume, or enlarged fur silhouette.

Image 2 is a fixed neutral grayscale semantic guide derived only from the frozen Main_case eat.mp4 and TM.mp4 alpha geometry. It contains no source-cat RGB, luminance, texture, markings, identity, or rendering style. Image 2 alone controls the pulled-back full-body scale, neutral compact seated anatomy, exactly four attached legs, exactly four complete distinct paws, two ears, torso placement, complete tail from base to tip, canvas margins, and composition. Copy no gray color, identity, face, fur, marking, or material from Image 2.

Render the exact pet identity from Image 1 in Image 2's complete full-body geometry. Use a compact smooth short-to-medium coat lying tightly and tangentially along the cheeks, neck, chest, shoulders, spine, flanks, haunches, legs, and tail. The outer coat is one clean continuously groomed surface with only short soft edge hairs. Show the full body, all four paws, both ear tips, and the entire tail with generous green clearance. The cat is neutral and action-free: no bowl, food, raised paw, licking, prop, clothing, collar, glasses, accessory, floor, shadow, text, logo, or watermark. Do not invent body markings not visible in Image 1; extend the nearest verified dominant coat color conservatively into unobserved body regions."""

NEGATIVE_PROMPT = """old driver cat identity, profile cat, default cat, historical reference cat, driver RGB, driver fur color, gray target cat, grayscale output, portrait, close-up, oversized head, giant head, cropped body, cropped ears, cropped paw, cropped tail, missing leg, extra leg, fused leg, hidden paw, extra paw, detached paw, duplicated limb, tail outside frame, tail emerging from spine, long-haired breed, cheek ruff, neck ruff, chest mane, chest bib, plume tail, expanded fur silhouette, radial fur, outward strands, brush bristles, black zipper outline, jagged fur, halo, white border, action pose, eating pose, grooming pose, raised paw, bowl, food, prop, clothes, collar, glasses, accessory, floor, shadow, background gradient, non-green background, multiple cats, person, text, logo, watermark"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def link(node: str, slot: int = 0) -> list[Any]:
    return [node, slot]


def bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask)
    if not len(xs):
        raise RuntimeError("empty_mask")
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


def largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        raise RuntimeError("driver_subject_component_missing")
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == label


def normalized_action_masks(
    path: Path,
    *,
    action: str,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    frames, durations = scail.read_animation_frames(path)
    sampled = scail.sample_timed_frames(frames, durations, 81, 16.0)
    union = scail.expand_bbox(
        scail.union_alpha_bbox(sampled), sampled[0].size, margin_fraction=0.06
    )
    normalized = [
        scail.contain_rgba(frame.crop(union), (CANVAS, CANVAS), 0.88)
        for frame in sampled
    ]
    if action == "eating":
        prop_masks, prop_metadata = scail.make_eating_prop_exclusion_masks(normalized)
    else:
        prop_masks = [Image.new("L", (CANVAS, CANVAS), 0) for _ in normalized]
        prop_metadata = None
    masks: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    for index, (frame, prop_image) in enumerate(zip(normalized, prop_masks, strict=True)):
        # RGB is deliberately never read here.  Only the alpha matte created by
        # the runner and, for eating, the independent prop matte enter geometry.
        alpha = np.asarray(frame.getchannel("A"), dtype=np.uint8) >= 16
        prop = np.asarray(prop_image, dtype=np.uint8) >= 128
        subject = largest_component(alpha & ~prop)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        subject = cv2.morphologyEx(
            subject.astype(np.uint8), cv2.MORPH_OPEN, kernel_open
        )
        subject = cv2.morphologyEx(subject, cv2.MORPH_CLOSE, kernel_close)
        subject = largest_component(subject > 0)
        masks.append(subject)
        box = bbox(subject)
        records.append(
            {
                "frame": index,
                "bbox": box,
                "width": box[2] - box[0],
                "height": box[3] - box[1],
                "center_x": round((box[0] + box[2]) / 2.0, 6),
                "bottom": box[3],
                "area": int(np.count_nonzero(subject)),
            }
        )
    values = np.asarray(
        [
            [r["bbox"][0], r["bbox"][1], r["bbox"][2], r["bbox"][3]]
            for r in records
        ],
        dtype=np.float64,
    )
    summary = {
        "action": action,
        "source_path": str(path.resolve()),
        "source_sha256": sha256_file(path),
        "frame_count": len(masks),
        "union_alpha_bbox_before_normalization": list(union),
        "median_bbox": [round(float(v), 6) for v in np.median(values, axis=0)],
        "median_width": round(float(np.median(values[:, 2] - values[:, 0])), 6),
        "median_height": round(float(np.median(values[:, 3] - values[:, 1])), 6),
        "median_center_x": round(float(np.median((values[:, 0] + values[:, 2]) / 2)), 6),
        "median_bottom": round(float(np.median(values[:, 3])), 6),
        "alpha_only_geometry": True,
        "source_rgb_read_by_guide_renderer": False,
        "source_luma_read_by_guide_renderer": False,
        "source_texture_read_by_guide_renderer": False,
        "prop_metadata_present": prop_metadata is not None,
    }
    return masks, summary


def ellipse(
    shape: tuple[int, int], center: tuple[int, int], axes: tuple[int, int]
) -> np.ndarray:
    output = np.zeros(shape, dtype=np.uint8)
    cv2.ellipse(output, center, axes, 0, 0, 360, 1, -1, lineType=cv2.LINE_AA)
    return output.astype(bool)


def polygon(shape: tuple[int, int], points: list[tuple[int, int]]) -> np.ndarray:
    output = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(output, [np.asarray(points, np.int32)], 1, lineType=cv2.LINE_AA)
    return output.astype(bool)


def build_fixed_semantic_guide(
    eating: dict[str, Any], grooming: dict[str, Any]
) -> tuple[Image.Image, Image.Image, dict[str, Any], dict[str, np.ndarray]]:
    width = int(round(np.median([eating["median_width"], grooming["median_width"]])))
    height = int(round(np.median([eating["median_height"], grooming["median_height"]])))
    width = min(width, int(CANVAS * 0.70))
    height = min(height, int(CANVAS * 0.70))
    center_x = int(
        round(np.median([eating["median_center_x"], grooming["median_center_x"]]))
    )
    bottom = min(
        int(round(np.median([eating["median_bottom"], grooming["median_bottom"]]))),
        int(CANVAS * 0.88),
    )
    top = bottom - height
    left = center_x - width // 2
    right = left + width

    def px(fraction: float) -> int:
        return int(round(left + width * fraction))

    def py(fraction: float) -> int:
        return int(round(top + height * fraction))

    shape = (CANVAS, CANVAS)
    regions: dict[str, np.ndarray] = {}
    regions["torso"] = ellipse(
        shape,
        (px(0.50), py(0.57)),
        (int(width * 0.31), int(height * 0.34)),
    )
    regions["head"] = ellipse(
        shape,
        (px(0.50), py(0.24)),
        (int(width * 0.225), int(height * 0.205)),
    )
    regions["ear_left"] = polygon(
        shape,
        [(px(0.29), py(0.17)), (px(0.34), py(0.00)), (px(0.43), py(0.15))],
    )
    regions["ear_right"] = polygon(
        shape,
        [(px(0.57), py(0.15)), (px(0.66), py(0.00)), (px(0.71), py(0.17))],
    )
    regions["hind_leg_left"] = ellipse(
        shape, (px(0.27), py(0.73)), (int(width * 0.17), int(height * 0.20))
    )
    regions["hind_leg_right"] = ellipse(
        shape, (px(0.73), py(0.73)), (int(width * 0.17), int(height * 0.20))
    )
    regions["foreleg_left"] = ellipse(
        shape, (px(0.42), py(0.73)), (int(width * 0.065), int(height * 0.23))
    )
    regions["foreleg_right"] = ellipse(
        shape, (px(0.58), py(0.73)), (int(width * 0.065), int(height * 0.23))
    )
    regions["hind_paw_left"] = ellipse(
        shape, (px(0.22), py(0.90)), (int(width * 0.13), int(height * 0.06))
    )
    regions["hind_paw_right"] = ellipse(
        shape, (px(0.78), py(0.90)), (int(width * 0.13), int(height * 0.06))
    )
    regions["forepaw_left"] = ellipse(
        shape, (px(0.40), py(0.94)), (int(width * 0.09), int(height * 0.052))
    )
    regions["forepaw_right"] = ellipse(
        shape, (px(0.60), py(0.94)), (int(width * 0.09), int(height * 0.052))
    )
    tail_line = np.zeros(shape, dtype=np.uint8)
    tail_points = np.asarray(
        [
            [px(0.76), py(0.62)],
            [px(0.91), py(0.66)],
            [px(0.98), py(0.75)],
            [px(0.94), py(0.88)],
        ],
        dtype=np.int32,
    )
    cv2.polylines(
        tail_line,
        [tail_points],
        isClosed=False,
        color=1,
        thickness=max(18, int(width * 0.105)),
        lineType=cv2.LINE_AA,
    )
    regions["tail"] = tail_line.astype(bool)
    regions["muzzle"] = ellipse(
        shape, (px(0.50), py(0.30)), (int(width * 0.095), int(height * 0.055))
    )
    regions["eye_left"] = ellipse(
        shape, (px(0.43), py(0.22)), (int(width * 0.035), int(height * 0.021))
    )
    regions["eye_right"] = ellipse(
        shape, (px(0.57), py(0.22)), (int(width * 0.035), int(height * 0.021))
    )
    regions["nose"] = ellipse(
        shape, (px(0.50), py(0.285)), (int(width * 0.024), int(height * 0.016))
    )

    silhouette = np.zeros(shape, dtype=bool)
    for value in regions.values():
        silhouette |= value
    silhouette = cv2.morphologyEx(
        silhouette.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    ).astype(bool)
    silhouette = largest_component(silhouette)

    gray = np.full(shape, FIXED_LUMA["base"], dtype=np.float32)
    assignments = [
        ("tail", "tail"),
        ("torso", "torso"),
        ("hind_leg_left", "hind_leg"),
        ("hind_leg_right", "hind_leg"),
        ("hind_paw_left", "hind_paw"),
        ("hind_paw_right", "hind_paw"),
        ("foreleg_left", "foreleg"),
        ("foreleg_right", "foreleg"),
        ("forepaw_left", "forepaw"),
        ("forepaw_right", "forepaw"),
        ("head", "head"),
        ("ear_left", "ear"),
        ("ear_right", "ear"),
        ("muzzle", "muzzle"),
        ("eye_left", "eye"),
        ("eye_right", "eye"),
        ("nose", "nose"),
    ]
    for region_name, luma_name in assignments:
        gray[regions[region_name]] = FIXED_LUMA[luma_name]
    weighted = cv2.GaussianBlur(gray * silhouette, (0, 0), 4.0)
    support = cv2.GaussianBlur(silhouette.astype(np.float32), (0, 0), 4.0)
    smooth = np.clip(weighted / np.maximum(support, 0.01), 80, 216)
    smooth = (np.round(smooth / 8.0) * 8.0).astype(np.uint8)
    guide_rgb = np.zeros((CANVAS, CANVAS, 3), dtype=np.uint8)
    guide_rgb[:] = GREEN
    guide_rgb[silhouette] = np.repeat(smooth[..., None], 3, axis=2)[silhouette]

    label_palette = {
        "torso": (80, 140, 190),
        "head": (230, 170, 70),
        "ear": (230, 90, 120),
        "hind_leg": (80, 170, 120),
        "hind_paw": (120, 220, 150),
        "foreleg": (80, 200, 220),
        "forepaw": (120, 240, 240),
        "tail": (130, 90, 190),
        "eye": (40, 40, 40),
        "muzzle": (245, 220, 170),
        "nose": (180, 70, 90),
    }
    label_rgb = np.full((CANVAS, CANVAS, 3), 255, dtype=np.uint8)
    for region_name, luma_name in assignments:
        label_rgb[regions[region_name]] = label_palette[luma_name]

    region_areas = {name: int(np.count_nonzero(value)) for name, value in regions.items()}
    region_centroids: dict[str, list[float]] = {}
    for name, value in regions.items():
        ys, xs = np.where(value)
        region_centroids[name] = [round(float(xs.mean()), 6), round(float(ys.mean()), 6)]
    guide_box = bbox(silhouette)
    geometry = {
        "derivation": "median_eating_and_grooming_alpha_scale_plus_fixed_neutral_anatomical_primitives",
        "derived_target_bbox_before_primitive_render": [left, top, right, bottom],
        "guide_subject_bbox": guide_box,
        "guide_width_fraction": round((guide_box[2] - guide_box[0]) / CANVAS, 6),
        "guide_height_fraction": round((guide_box[3] - guide_box[1]) / CANVAS, 6),
        "guide_margins": {
            "left": round(guide_box[0] / CANVAS, 6),
            "top": round(guide_box[1] / CANVAS, 6),
            "right": round((CANVAS - guide_box[2]) / CANVAS, 6),
            "bottom": round((CANVAS - guide_box[3]) / CANVAS, 6),
        },
        "region_areas": region_areas,
        "region_centroids": region_centroids,
        "four_leg_fields": [
            "hind_leg_left",
            "hind_leg_right",
            "foreleg_left",
            "foreleg_right",
        ],
        "four_paw_fields": [
            "hind_paw_left",
            "hind_paw_right",
            "forepaw_left",
            "forepaw_right",
        ],
        "fixed_luma": FIXED_LUMA,
        "gaussian_sigma": 4.0,
        "luma_quantum": 8,
    }
    return (
        Image.fromarray(guide_rgb, mode="RGB"),
        Image.fromarray(label_rgb, mode="RGB"),
        geometry,
        regions,
    )


def qwen_workflow(upload_name: str, guide_name: str) -> dict[str, Any]:
    workflow: dict[str, Any] = {
        "1": {"class_type": "LoadImage", "inputs": {"image": upload_name}},
        "2": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": QWEN_MODEL, "weight_dtype": "default"},
        },
        "3": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": QWEN_CLIP, "type": "qwen_image", "device": "default"},
        },
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": QWEN_VAE}},
        "5": {"class_type": "FluxKontextImageScale", "inputs": {"image": link("1")}},
        "6": {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {
                "clip": link("3"),
                "vae": link("4"),
                "image1": link("5"),
                "image2": link("21"),
                "prompt": POSITIVE_PROMPT,
            },
        },
        "7": {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {
                "clip": link("3"),
                "vae": link("4"),
                "image1": link("5"),
                "image2": link("21"),
                "prompt": NEGATIVE_PROMPT,
            },
        },
        "8": {
            "class_type": "FluxKontextMultiReferenceLatentMethod",
            "inputs": {"conditioning": link("6"), "reference_latents_method": "index_timestep_zero"},
        },
        "9": {
            "class_type": "FluxKontextMultiReferenceLatentMethod",
            "inputs": {"conditioning": link("7"), "reference_latents_method": "index_timestep_zero"},
        },
        # Geometry owns the sampler canvas.  This is intentionally Image 2,
        # unlike the project's historical qwen_workflow default.
        "10": {"class_type": "VAEEncode", "inputs": {"pixels": link("21"), "vae": link("4")}},
        "11": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": link("2"), "shift": 3.1}},
        "12": {"class_type": "CFGNorm", "inputs": {"model": link("11"), "strength": 1.0}},
        "13": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": link("12"), "lora_name": QWEN_LIGHTNING_LORA, "strength_model": 1.0},
        },
        "14": {
            "class_type": "KSampler",
            "inputs": {
                "model": link("17"),
                "positive": link("8"),
                "negative": link("9"),
                "latent_image": link("10"),
                "seed": QWEN_SEED,
                "steps": QWEN_STEPS,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        "15": {"class_type": "VAEDecode", "inputs": {"samples": link("14"), "vae": link("4")}},
        "16": {
            "class_type": "SaveImage",
            "inputs": {"images": link("15"), "filename_prefix": f"scail2_test/{RUN_ID}/canonical"},
        },
        "17": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": link("13"), "lora_name": QWEN_PET3D_LORA, "strength_model": QWEN_PET3D_STRENGTH},
        },
        "20": {"class_type": "LoadImage", "inputs": {"image": guide_name}},
        "21": {"class_type": "FluxKontextImageScale", "inputs": {"image": link("20")}},
    }
    return workflow


def alpha_preview(mask: np.ndarray) -> Image.Image:
    rgb = np.full((CANVAS, CANVAS, 3), 255, dtype=np.uint8)
    rgb[mask] = (80, 80, 80)
    return Image.fromarray(rgb, mode="RGB")


def make_board(
    upload: Path,
    eating_mask: np.ndarray,
    grooming_mask: np.ndarray,
    guide: Image.Image,
    labels: Image.Image,
    output: Path,
) -> None:
    cell = 320
    header = 58
    titles = [
        "Image 1: uploaded identity only",
        "eat.mp4 alpha geometry only",
        "TM.mp4 alpha geometry only",
        "Image 2: fixed semantic guide",
        "guide topology audit labels",
    ]
    with Image.open(upload) as loaded:
        images = [loaded.convert("RGB"), alpha_preview(eating_mask), alpha_preview(grooming_mask), guide, labels]
    board = Image.new("RGB", (cell * len(images), cell + header), "white")
    draw = ImageDraw.Draw(board)
    for index, (title, item) in enumerate(zip(titles, images, strict=True)):
        thumb = item.copy()
        thumb.thumbnail((cell - 16, cell - 16), Image.Resampling.LANCZOS)
        x = index * cell + (cell - thumb.width) // 2
        y = header + (cell - thumb.height) // 2
        board.paste(thumb, (x, y))
        draw.text((index * cell + 8, 18), title, fill="black")
    board.save(output, format="PNG", optimize=True)


def queue_snapshot() -> dict[str, Any]:
    try:
        with urllib.request.urlopen("http://127.0.0.1:8188/queue", timeout=3) as response:
            return json.load(response)
    except Exception as exc:  # noqa: BLE001 - read-only evidence only
        return {"unavailable": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("/root/comfy-pet-pipeline"))
    parser.add_argument("--asset-root", type=Path, default=Path("/data/comfy-pet-pipeline/profile_assets/sleek_tabby_cat_v1/driving/20260731-main-case-v1"))
    parser.add_argument("--job-root", type=Path, default=Path("/data/comfy-pet-pipeline/outputs/local_video_api/20260731T050352Z-ec6245d4d4"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/data/comfy-pet-pipeline/archive/validation/"
            "20260731-upload-only-identity-17f-r1/"
            "upload_identity_main_case_canonical_v1_r2"
        ),
    )
    parser.add_argument("--comfy-input", type=Path, default=Path("/data/comfy-pet-pipeline/comfyui/input"))
    args = parser.parse_args()

    output = args.output.resolve()
    input_dir = args.comfy_input.resolve() / "scail2_test" / RUN_ID
    if output.exists() or input_dir.exists():
        raise RuntimeError(f"refusing_to_overwrite:{output}:{input_dir}")
    output.mkdir(parents=True)
    input_dir.mkdir(parents=True)

    upload = args.job_root / "input.png"
    upload_source = args.job_root / "input_identity_source.png"
    eat_source = args.asset_root / "source" / "eat.mp4"
    eat_runtime = args.asset_root / "eating_driver_512x512_81f_16fps.mp4"
    tm_source = args.asset_root / "source" / "TM.mp4"
    tm_runtime = args.asset_root / "grooming_driver_512x512_81f_16fps.mp4"
    submit_script = args.project / "diagnostics" / "submit_upload_identity_main_case_canonical_once.py"
    builder_script = Path(__file__).resolve()

    expected = {
        upload: EXPECTED_UPLOAD_SHA256,
        upload_source: EXPECTED_UPLOAD_SOURCE_SHA256,
        eat_source: EXPECTED_EAT_SOURCE_SHA256,
        eat_runtime: EXPECTED_EAT_RUNTIME_SHA256,
        tm_source: EXPECTED_TM_SOURCE_SHA256,
        tm_runtime: EXPECTED_TM_RUNTIME_SHA256,
    }
    for path, digest in expected.items():
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"frozen_input_missing_or_changed:{path}")
    if not submit_script.is_file():
        raise RuntimeError(f"one_shot_submission_script_missing:{submit_script}")

    eating_masks, eating_metrics = normalized_action_masks(eat_runtime, action="eating")
    grooming_masks, grooming_metrics = normalized_action_masks(tm_runtime, action="grooming")
    guide, labels, geometry, regions = build_fixed_semantic_guide(eating_metrics, grooming_metrics)

    copied_upload = input_dir / "upload_identity.png"
    guide_path = input_dir / "canonical_main_case_semantic_guide.png"
    shutil.copy2(upload, copied_upload)
    guide.save(guide_path, format="PNG", optimize=True)
    output_guide = output / guide_path.name
    shutil.copy2(guide_path, output_guide)
    label_path = output / "canonical_main_case_semantic_topology.png"
    labels.save(label_path, format="PNG", optimize=True)
    board_path = output / "canonical_input_roles_visual_board.png"
    make_board(upload, eating_masks[0], grooming_masks[76], guide, labels, board_path)

    upload_name = f"scail2_test/{RUN_ID}/{copied_upload.name}"
    guide_name = f"scail2_test/{RUN_ID}/{guide_path.name}"
    workflow = qwen_workflow(upload_name, guide_name)
    workflow_path = output / "qwen_canonical_one_shot_workflow_api.json"
    workflow_path.write_text(
        json.dumps(workflow, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    guide_array = np.asarray(guide.convert("RGB"), dtype=np.uint8)
    green = np.all(guide_array == GREEN, axis=2)
    foreground = ~green
    foreground_values = guide_array[foreground]
    component_count, _labels, component_stats, _ = cv2.connectedComponentsWithStats(
        foreground.astype(np.uint8), connectivity=8
    )
    meaningful_components = sum(
        int(component_stats[index, cv2.CC_STAT_AREA]) >= 64
        for index in range(1, component_count)
    )
    required_regions = [
        "head", "ear_left", "ear_right", "torso", "tail",
        "hind_leg_left", "hind_leg_right", "foreleg_left", "foreleg_right",
        "hind_paw_left", "hind_paw_right", "forepaw_left", "forepaw_right",
        "eye_left", "eye_right", "muzzle", "nose",
    ]
    required_minimums = {
        "head": 2500, "ear_left": 180, "ear_right": 180, "torso": 12000,
        "tail": 1200, "hind_leg_left": 1500, "hind_leg_right": 1500,
        "foreleg_left": 800, "foreleg_right": 800, "hind_paw_left": 500,
        "hind_paw_right": 500, "forepaw_left": 350, "forepaw_right": 350,
        "eye_left": 60, "eye_right": 60, "muzzle": 500, "nose": 30,
    }
    role_nodes = workflow["6"]["inputs"]
    negative_role_nodes = workflow["7"]["inputs"]
    lora_nodes = [
        node for node in workflow.values() if node.get("class_type") == "LoraLoaderModelOnly"
    ]
    lora_config = {
        str(node["inputs"]["lora_name"]): float(node["inputs"]["strength_model"])
        for node in lora_nodes
    }
    load_images = [
        node["inputs"]["image"]
        for node in workflow.values()
        if node.get("class_type") == "LoadImage"
    ]
    prompt_text = (POSITIVE_PROMPT + "\n" + NEGATIVE_PROMPT).casefold()
    forbidden_photography_lora_tokens = ["photography", "摄影", "v3.0"]
    all_region_areas_pass = all(
        int(np.count_nonzero(regions[name])) >= required_minimums[name]
        for name in required_regions
    )
    hard_checks = {
        "target_job_upload_sha256_exact": sha256_file(copied_upload) == EXPECTED_UPLOAD_SHA256,
        "target_job_identity_source_sha256_exact": sha256_file(upload_source) == EXPECTED_UPLOAD_SOURCE_SHA256,
        "eat_source_sha256_exact": sha256_file(eat_source) == EXPECTED_EAT_SOURCE_SHA256,
        "eat_runtime_sha256_exact": sha256_file(eat_runtime) == EXPECTED_EAT_RUNTIME_SHA256,
        "tm_source_sha256_exact": sha256_file(tm_source) == EXPECTED_TM_SOURCE_SHA256,
        "tm_runtime_sha256_exact": sha256_file(tm_runtime) == EXPECTED_TM_RUNTIME_SHA256,
        "both_action_geometry_metrics_alpha_only": eating_metrics["alpha_only_geometry"] and grooming_metrics["alpha_only_geometry"],
        "guide_renderer_did_not_read_source_rgb": not eating_metrics["source_rgb_read_by_guide_renderer"] and not grooming_metrics["source_rgb_read_by_guide_renderer"],
        "guide_renderer_did_not_read_source_luma": not eating_metrics["source_luma_read_by_guide_renderer"] and not grooming_metrics["source_luma_read_by_guide_renderer"],
        "guide_renderer_did_not_read_source_texture": not eating_metrics["source_texture_read_by_guide_renderer"] and not grooming_metrics["source_texture_read_by_guide_renderer"],
        "guide_foreground_is_strict_grayscale": bool(np.all(foreground_values[:, 0] == foreground_values[:, 1]) and np.all(foreground_values[:, 1] == foreground_values[:, 2])),
        "guide_background_is_exact_pure_green": bool(np.all(guide_array[green] == GREEN)),
        "guide_has_background_and_foreground": bool(np.any(green) and np.any(foreground)),
        "guide_is_one_coherent_subject_component": meaningful_components == 1,
        "guide_has_all_required_semantic_parts": all_region_areas_pass,
        "guide_has_exactly_four_leg_fields": len(geometry["four_leg_fields"]) == 4,
        "guide_has_exactly_four_paw_fields": len(geometry["four_paw_fields"]) == 4,
        "guide_complete_tail_field_present": geometry["region_areas"]["tail"] >= required_minimums["tail"],
        "guide_width_at_most_72_percent": geometry["guide_width_fraction"] <= 0.72,
        "guide_height_at_most_72_percent": geometry["guide_height_fraction"] <= 0.72,
        "guide_minimum_margin_at_least_10_percent": min(geometry["guide_margins"].values()) >= 0.10,
        "workflow_image1_loads_only_target_upload": workflow["1"]["inputs"]["image"] == upload_name and role_nodes["image1"] == link("5") and negative_role_nodes["image1"] == link("5"),
        "workflow_image2_loads_only_semantic_guide": workflow["20"]["inputs"]["image"] == guide_name and role_nodes["image2"] == link("21") and negative_role_nodes["image2"] == link("21"),
        "workflow_vae_latent_is_image2": workflow["10"]["inputs"]["pixels"] == link("21") and workflow["14"]["inputs"]["latent_image"] == link("10"),
        "workflow_has_no_profile_or_default_reference_load": load_images == [upload_name, guide_name],
        "workflow_has_only_lightning_and_pet3d_lora": lora_config == {QWEN_LIGHTNING_LORA: 1.0, QWEN_PET3D_LORA: QWEN_PET3D_STRENGTH},
        "workflow_has_no_photography_lora": not any(any(token in name.casefold() for token in forbidden_photography_lora_tokens) for name in lora_config),
        "workflow_pet3d_strength_exact_0_15": lora_config.get(QWEN_PET3D_LORA) == 0.15,
        "workflow_seed_exact_20260726": workflow["14"]["inputs"]["seed"] == QWEN_SEED,
        "workflow_steps_exact_6": workflow["14"]["inputs"]["steps"] == QWEN_STEPS,
        "workflow_is_one_output_one_sampler": sum(node.get("class_type") == "KSampler" for node in workflow.values()) == 1 and sum(node.get("class_type") == "SaveImage" for node in workflow.values()) == 1,
        "prompt_declares_upload_only_identity": "image 1 is the original uploaded pet and is the only identity reference" in prompt_text,
        "prompt_forbids_old_profile_default_identity": all(token in prompt_text for token in ["old driver cat identity", "profile cat", "default cat"]),
        "one_shot_script_requires_explicit_flag": "--i-understand-this-submits-one-gpu-job" in submit_script.read_text(encoding="utf-8"),
        "one_shot_script_posts_prompt_once": submit_script.read_text(encoding="utf-8").count('Request(') == 2 and submit_script.read_text(encoding="utf-8").count('"/prompt"') == 1,
        "builder_script_is_frozen_and_readable": builder_script.is_file() and sha256_file(builder_script) != "",
    }
    all_cpu_hard_checks_passed = all(value is True for value in hard_checks.values())
    queue = queue_snapshot()
    manifest = {
        "status": "cpu_gate_passed_ready_for_explicit_submission" if all_cpu_hard_checks_passed else "cpu_gate_failed_do_not_submit",
        "run_id": RUN_ID,
        "policy": POLICY,
        "policy_version": POLICY_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu_submitted": False,
        "production_modified": False,
        "all_cpu_hard_checks_passed": all_cpu_hard_checks_passed,
        "hard_checks": hard_checks,
        "provenance": {
            "image1": {"role": "uploaded_identity_only", "path": str(copied_upload), "sha256": sha256_file(copied_upload)},
            "image2": {"role": "main_case_alpha_scale_plus_fixed_semantic_topology_only", "path": str(guide_path), "sha256": sha256_file(guide_path)},
            "profile_or_default_reference_pixels_used": False,
            "main_case_source_rgb_used_in_guide": False,
            "main_case_source_luma_used_in_guide": False,
            "main_case_source_texture_used_in_guide": False,
            "main_case_geometry_inputs": {"eating": eating_metrics, "grooming": grooming_metrics},
        },
        "geometry": geometry,
        "workflow_contract": {
            "image1_node": "1->5->6/7.image1",
            "image2_node": "20->21->6/7.image2",
            "latent_node": "20->21->10->14.latent_image",
            "model": QWEN_MODEL,
            "clip": QWEN_CLIP,
            "vae": QWEN_VAE,
            "loras": lora_config,
            "seed": QWEN_SEED,
            "steps": QWEN_STEPS,
            "cfg": 1.0,
            "denoise": 1.0,
            "sampler": "euler",
            "scheduler": "simple",
            "positive_prompt_sha256": sha256_bytes(POSITIVE_PROMPT.encode("utf-8")),
            "negative_prompt_sha256": sha256_bytes(NEGATIVE_PROMPT.encode("utf-8")),
        },
        "artifacts": {
            "guide": {"path": str(output_guide), "sha256": sha256_file(output_guide)},
            "guide_comfy_input": {"path": str(guide_path), "sha256": sha256_file(guide_path)},
            "upload_comfy_input": {"path": str(copied_upload), "sha256": sha256_file(copied_upload)},
            "topology_labels": {"path": str(label_path), "sha256": sha256_file(label_path)},
            "visual_board": {"path": str(board_path), "sha256": sha256_file(board_path)},
            "workflow": {"path": str(workflow_path), "sha256": sha256_file(workflow_path), "submitted": False},
            "one_shot_submission_script": {"path": str(submit_script), "sha256": sha256_file(submit_script), "executed": False},
            "cpu_builder_script": {"path": str(builder_script), "sha256": sha256_file(builder_script)},
        },
        "queue_snapshot_after_cpu_build": queue,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    summary = {
        "status": manifest["status"],
        "run_id": RUN_ID,
        "all_cpu_hard_checks_passed": all_cpu_hard_checks_passed,
        "failed_checks": [name for name, passed in hard_checks.items() if not passed],
        "guide_sha256": sha256_file(output_guide),
        "workflow_sha256": sha256_file(workflow_path),
        "manifest_sha256": sha256_file(manifest_path),
        "visual_board_sha256": sha256_file(board_path),
        "one_shot_script_sha256": sha256_file(submit_script),
        "gpu_submitted": False,
        "production_modified": False,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if all_cpu_hard_checks_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
