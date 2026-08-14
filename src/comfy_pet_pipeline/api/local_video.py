from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import queue
import re
import secrets
import shutil
import subprocess
import threading
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

import requests
import cv2
import numpy as np
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse
from PIL import Image, ImageChops, UnidentifiedImageError

from ..quality.fur_contour import (
    compare_fur_contours,
    evaluate_animation_fur_quality,
)
from ..quality.fur_surface import (
    GATE_VERSION as FUR_SURFACE_GATE_VERSION,
    POLICY as FUR_SURFACE_GATE_POLICY,
    THRESHOLD_VERSION as FUR_SURFACE_THRESHOLD_VERSION,
    FurSurfaceThresholds,
    evaluate_fur_surface,
)
from ..quality.final_video_fur import (
    DEFAULT_SAMPLE_COUNT as FINAL_VIDEO_FUR_SAMPLE_COUNT,
    FINAL_VIDEO_COHERENT_BRUSH_TEMPORAL_MIN_SAMPLES,
    FINAL_VIDEO_DIRECT_QWEN_SPARSE_COARSE_MAX_BLOCKING_FRACTION,
    FINAL_VIDEO_DIRECT_QWEN_SPARSE_COARSE_MIN_BLOCKING_SAMPLES,
    FINAL_VIDEO_DIRECT_QWEN_SPARSE_COARSE_MIN_SAMPLES,
    FINAL_VIDEO_DIRECT_QWEN_SPARSE_COARSE_TEMPORAL_SPREAD_MAX,
    FINAL_VIDEO_EATING_BOUNDARY_ANCHOR_MIN_BLOCKING_SAMPLES,
    FINAL_VIDEO_EATING_BOUNDARY_ANCHOR_MIN_SAMPLES,
    FINAL_VIDEO_EATING_BOUNDARY_ANCHOR_SPREAD_MAX,
    FINAL_VIDEO_EATING_ISOLATED_LONG_FUR_MIN_SAMPLES,
    FINAL_VIDEO_EATING_PROP_ALIGNMENT_MIN,
    FINAL_VIDEO_EATING_RADIAL_BOUNDED_REFERENCE_LOAD_DELTA_MULTIPLIER,
    FINAL_VIDEO_EATING_RADIAL_BOUNDED_REFERENCE_RATIO_MARGIN,
    FINAL_VIDEO_EATING_RADIAL_BOUNDARY_COMPONENT_SPREAD_MAX,
    FINAL_VIDEO_EATING_RADIAL_BOUNDARY_LOAD_ENERGY_SPREAD_MAX,
    FINAL_VIDEO_EATING_RADIAL_BOUNDARY_MIN_SAMPLES,
    FINAL_VIDEO_LOW_SIGNAL_REFERENCE_MIN_SAMPLES,
    FINAL_VIDEO_LOW_SIGNAL_REFERENCE_MIN_LONG_BLOCKING_FRACTION,
    FINAL_VIDEO_LOW_SIGNAL_REFERENCE_TEMPORAL_SPREAD_MAX,
    FINAL_VIDEO_SPARSE_COARSE_LOW_SIGNAL_MAX_BLOCKING_FRACTION,
    FINAL_VIDEO_SPARSE_COARSE_LOW_SIGNAL_MIN_BLOCKING_SAMPLES,
    FINAL_VIDEO_SPARSE_COARSE_LOW_SIGNAL_MIN_SAMPLES,
    POLICY as FINAL_VIDEO_FUR_AUDIT_POLICY,
    PROFILE_FULL_SCOPE as FINAL_VIDEO_PROFILE_FULL_SCOPE,
    VALIDATED_ACTION_SURFACE_SCOPE as FINAL_VIDEO_VALIDATED_ACTION_SURFACE_SCOPE,
    VERSION as FINAL_VIDEO_FUR_AUDIT_VERSION,
    audit_final_video,
)
from ..quality.grooming_pose import (
    GATE_VERSION as GROOMING_POSE_GATE_VERSION,
    POLICY as GROOMING_POSE_GATE_POLICY,
    GroomingPoseThresholds,
    evaluate_grooming_pose,
)
from ..media.transparent_animation import (
    build_standard_direct_outputs,
    build_transparent_animations,
    build_transparent_cutout,
)
from ..identity.uploaded_identity import (
    POLICY as UPLOADED_IDENTITY_SCAIL_REFERENCE_POLICY,
    VERSION as UPLOADED_IDENTITY_SCAIL_REFERENCE_VERSION,
    build_uploaded_identity_scail_references,
)
from .job_callback import JobCallbackReporter
from .local_video_parts import (
    generation_quality as _local_video_generation_quality,
    geometry_guides as _local_video_geometry_guides,
    grooming_image_preparation as _local_video_grooming_image_preparation,
    health as _local_video_health,
    identity_quality as _local_video_identity_quality,
    job_runner as _local_video_job_runner,
    prompting_workflows as _local_video_prompting_workflows,
    subject_replacement as _local_video_subject_replacement,
    uploaded_identity_audit as _local_video_uploaded_identity_audit,
    uploaded_references as _local_video_uploaded_references,
)
from .uploaded_identity_front_view import (
    FRONT_VIEW_NOT_RELIABLY_FRONTAL_STAGE,
    UPLOADED_FRONT_VIEW_AUDIT_POLICY,
    audit_uploaded_front_identity_view,
)


LOGGER = logging.getLogger("local_video_api")
logging.basicConfig(
    level=os.environ.get("LOCAL_VIDEO_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

API_PREFIX = "/api/local-video"
API_VERSION = "2.24.19"
DATA_ROOT = Path(
    os.environ.get("PET_PIPELINE_DATA_ROOT", "/data/comfy-pet-pipeline")
).resolve()
PROJECT_ROOT = Path(
    os.environ.get(
        "PET_PIPELINE_PROJECT_ROOT",
        str(Path(__file__).resolve().parents[3]),
    )
).resolve()
COMFY_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188").rstrip("/")
COMFY_INPUT = DATA_ROOT / "comfyui" / "input"
COMFY_OUTPUT = DATA_ROOT / "comfyui" / "output"
MODEL_ROOT = DATA_ROOT / "models" / "comfyui"
OUTPUT_ROOT = DATA_ROOT / "outputs" / "local_video_api"
USER_LORA_DIR = MODEL_ROOT / "loras" / "user_uploads"
JOB_CALLBACK_REPORTER = JobCallbackReporter.from_environment(
    output_root=OUTPUT_ROOT,
    logger=LOGGER,
)

API_KEY = os.environ.get("LOCAL_VIDEO_API_KEY", "").strip()
PROMPT_TRANSLATION_ENABLED = os.environ.get(
    "PROMPT_TRANSLATION_ENABLED", "1"
).strip().lower() not in {"0", "false", "no", "off"}
PROMPT_TRANSLATOR_PYTHON = Path(
    os.environ.get(
        "PROMPT_TRANSLATOR_PYTHON",
        "/opt/miniconda3/envs/comfyui/bin/python",
    )
).resolve()
PROMPT_TRANSLATOR_MODEL = Path(
    os.environ.get(
        "PROMPT_TRANSLATOR_MODEL",
        str(DATA_ROOT / "models" / "translation" / "opus-mt-zh-en"),
    )
).resolve()
PROMPT_TRANSLATOR_SCRIPT = PROJECT_ROOT / "tools" / "prompting" / "prompt_translator.py"
PROMPT_TRANSLATION_TIMEOUT = max(
    10,
    min(300, int(os.environ.get("PROMPT_TRANSLATION_TIMEOUT", "120"))),
)
PROMPT_TRANSLATION_THREADS = max(
    1,
    min(16, int(os.environ.get("PROMPT_TRANSLATION_THREADS", "4"))),
)
MAX_UPLOAD_BYTES = int(
    os.environ.get("LOCAL_VIDEO_MAX_UPLOAD_BYTES", str(15 * 1024 * 1024))
)
QWEN_POSE_MAX_ATTEMPTS = max(
    1, min(3, int(os.environ.get("QWEN_POSE_MAX_ATTEMPTS", "3")))
)
GENERATION_AUDIT_POLICY = "runtime_integrity_required_visual_content_advisory_v1"
GENERATION_VISUAL_AUDIT_MODE = os.environ.get(
    "LOCAL_VIDEO_VISUAL_AUDIT_MODE",
    "advisory",
).strip().lower()
GENERATION_VISUAL_AUDITS_BLOCKING = GENERATION_VISUAL_AUDIT_MODE in {
    "blocking",
    "strict",
    "enforced",
    "1",
    "true",
    "yes",
    "on",
}
FUR_SURFACE_GATE_MAX_REJECTIONS = 3
FUR_SURFACE_GATE_RETRY_POLICY = "preserve_configured_eating_bowl_surface_recovery_v1"
QWEN_BOWL_POSE_RETRY_NAME = "bowl_palette_canvas_seed_retry"
QWEN_RUNNING_POSE_EXTRA_ATTEMPTS = max(
    0,
    min(2, int(os.environ.get("QWEN_RUNNING_POSE_EXTRA_ATTEMPTS", "1"))),
)
QWEN_BOWL_POSE_EXTRA_ATTEMPTS = max(
    0,
    min(2, int(os.environ.get("QWEN_BOWL_POSE_EXTRA_ATTEMPTS", "2"))),
)
QWEN_CHROMA_MAX_ATTEMPTS = max(
    1,
    min(3, int(os.environ.get("QWEN_CHROMA_MAX_ATTEMPTS", "3"))),
)
QWEN_RECOVERY_USER_LORA_STRENGTH = float(
    os.environ.get("QWEN_RECOVERY_USER_LORA_STRENGTH", "0.25")
)
QWEN_CANONICAL_USER_LORA_TOTAL_CAP = float(
    os.environ.get("QWEN_CANONICAL_USER_LORA_TOTAL_CAP", "1.0")
)
QWEN_CANONICAL_USER_LORA_PER_LORA_CAP = float(
    os.environ.get("QWEN_CANONICAL_USER_LORA_PER_LORA_CAP", "1.0")
)
QWEN_POSE_USER_LORA_TOTAL_CAP = float(
    os.environ.get("QWEN_POSE_USER_LORA_TOTAL_CAP", "0.40")
)
QWEN_POSE_USER_LORA_RETRY_TOTAL_CAP = float(
    os.environ.get("QWEN_POSE_USER_LORA_RETRY_TOTAL_CAP", "0.25")
)
QWEN_POSE_USER_LORA_FINAL_TOTAL_CAP = float(
    os.environ.get("QWEN_POSE_USER_LORA_FINAL_TOTAL_CAP", "0.15")
)
QWEN_POSE_USER_LORA_PER_LORA_CAP = float(
    os.environ.get("QWEN_POSE_USER_LORA_PER_LORA_CAP", "0.35")
)
QWEN_SUBJECT_REPLACEMENT_CANONICAL_LORA_TOTAL_CAP = float(
    os.environ.get(
        "QWEN_SUBJECT_REPLACEMENT_CANONICAL_LORA_TOTAL_CAP",
        "0.15",
    )
)
QWEN_SUBJECT_REPLACEMENT_POSE_LORA_TOTAL_CAP = float(
    os.environ.get(
        "QWEN_SUBJECT_REPLACEMENT_POSE_LORA_TOTAL_CAP",
        "0.25",
    )
)
POSE_IDENTITY_INPUT_POLICY_CANONICAL = "canonical_character_image1"
POSE_IDENTITY_INPUT_POLICY_DIRECT_UPLOAD = "uploaded_photo_direct_profile_action_image2"
POSE_IDENTITY_INPUT_POLICY_MAIN_CASE_GEOMETRY_FIRST = (
    "main_case_driver_geometry_image1_uploaded_identity_image2_v2"
)
# Backward-compatible import name for the retired geometry-first diagnostic
# branch. The production upload path always generates a Qwen canonical first
# and uses that canonical as Image 1 for both supported actions. The dedicated
# grooming geometry-first policy below remains readable only for old jobs and
# regression diagnostics.
POSE_IDENTITY_INPUT_POLICY_PROFILE_GEOMETRY_FIRST = (
    POSE_IDENTITY_INPUT_POLICY_MAIN_CASE_GEOMETRY_FIRST
)
POSE_IDENTITY_INPUT_POLICY_GROOMING_GEOMETRY_CANONICAL = (
    "main_case_grooming_geometry_image1_canonical_identity_image2_v1"
)
SCAIL_FROZEN_SLEEK_EATING_POLICY = (
    "frozen_profile_action_identity_with_bowl_mapped_red_v1"
)
SCAIL_FROZEN_EATING_PIPELINE = "frozen_profile_eating_then_scail_v1"
SCAIL_MIXED_REFERENCE_PIPELINE = (
    "frozen_profile_eating_plus_qwen_grooming_then_scail_v1"
)
PROFILE_STANDARD_DIRECT_POLICY = "main_case_profile_standard_direct_v2"
PROFILE_STANDARD_DIRECT_ROLE = (
    "authoritative_black_white_main_case_subject_identity_and_action_standard"
)
PROFILE_STANDARD_DIRECT_GPU_REASON = (
    "The frozen Main_case runtime is already the requested profile identity and "
    "action standard; generative redraw would change the approved subject."
)
QWEN_FULL_BODY_FIRST_FRAME_PIPELINE = (
    "qwen_full_body_then_action_keyframes_then_scail_v1"
)
QWEN_FULL_BODY_FIRST_FRAME_PIPELINE_VERSION = "1.40.15"
GROOMING_QWEN_POSE_REFERENCE_POLICY = (
    "canonical_only_prompted_grooming_full_canvas_action_redraw_raw_qwen_v40"
)
GROOMING_QWEN_GEOMETRY_GUIDE_POLICY = "driving_grooming_geometry_image2_v8"
GROOMING_PRIVATE_POSE_GATE_GUIDE_POLICY = "driving_grooming_geometry_image1_v3"
GROOMING_QWEN_GEOMETRY_GUIDE_METHOD = (
    "single_tone_organic_open_channel_main_case_silhouette_v5"
)
GROOMING_QWEN_GUIDE_FOREGROUND_RGB = (144, 144, 144)
GROOMING_QWEN_GUIDE_FORELEG_RGB = (176, 176, 176)
GROOMING_QWEN_GUIDE_STYLE = (
    "single_tone_achromatic_open_foreleg_channel_plus_private_cv_v5"
)
CANONICAL_ACTION_IDENTITY_PALETTE_POLICY = (
    "canonical_action_identity_palette_continuity_v1"
)
# Historical accepted canonical-first action candidates top out at 0.358322,
# while every observed grooming guide-copy failure starts at 0.489248. Keep a
# fixed fail-closed gap between those populations instead of allowing a gray
# guide to replace a light canonical subject.
CANONICAL_ACTION_ACHROMATIC_BALANCE_L1_MAX = 0.40
GROOMING_QWEN_NEGATIVE_SPACE_METHOD = (
    "verified_foreleg_anchor_organic_open_curve_negative_space_v3"
)
GROOMING_QWEN_FORELEG_SEMANTIC_METHOD = "verified_shoulder_to_paw_achromatic_capsule_v1"
QWEN_GROOMING_CANONICAL_GEOMETRY_DENOISE_BY_ATTEMPT = (0.84, 0.82, 0.86)
GROOMING_QWEN_FORELEG_REPAIR_POLICY = (
    "canonical_foreleg_seed_expanded_context_local_pet3d_composite_gate_v9"
)
GROOMING_CANONICAL_FORELEG_COMPLETION_POLICY = (
    "canonical_identity_base_masked_private_anatomy_foreleg_completion_v2"
)
GROOMING_PRIVATE_COMPLETION_MASK_POLICY = (
    "verified_foreleg_completion_mask_not_qwen_input_v1"
)
GROOMING_CANONICAL_COMPLETION_EDGE_EVIDENCE_POLICY = (
    "canonical_masked_completion_antialias_edge_evidence_v1"
)
# The v29 production candidate missed the unchanged 0.11 global threshold by
# 23 Canny pixels out of a 16,820-pixel ROI while every independent geometry,
# contact, color, histogram, and texture check passed. Accept only this narrow
# quantization band on the digest-bound canonical masked-completion attempt.
GROOMING_CANONICAL_COMPLETION_EDGE_DENSITY_RATIO_MIN = 0.985
GROOMING_CANONICAL_DIRECT_STRUCTURAL_EDGE_EVIDENCE_POLICY = (
    "canonical_direct_organic_foreleg_structural_edge_evidence_v3"
)
# Direct canonical attempts 1-2 can land just below the unchanged 0.11 raw
# Canny threshold when an antialiased light paw is otherwise correctly formed.
# Recover only the narrow population that also has strong low-pass organic
# structure and substantially exceeds the ordinary guide-alignment contract.
# The strict chamfer bounds keep historical straight prop/cylinder false
# positives (coverage <= 0.632565, mean distance >= 6.789175) rejected.
GROOMING_CANONICAL_DIRECT_EDGE_DENSITY_RATIO_MIN = 0.925
GROOMING_CANONICAL_DIRECT_STRUCTURAL_EDGE_DENSITY_MIN = 0.03
GROOMING_CANONICAL_DIRECT_CHAMFER_COVERAGE_MIN = 0.70
GROOMING_CANONICAL_DIRECT_CHAMFER_MEAN_DISTANCE_MAX = 5.0
GROOMING_CANONICAL_DIRECT_CHAMFER_P90_DISTANCE_MAX = 12.0
GROOMING_CANONICAL_DIRECT_EDGE_EVIDENCE_MAX_ATTEMPT = 2
GROOMING_CANONICAL_DIRECT_OFFGUIDE_STRUCTURAL_EDGE_DENSITY_MIN = 0.025
GROOMING_CANONICAL_DIRECT_OFFGUIDE_CHAMFER_COVERAGE_MIN = 0.56
GROOMING_CANONICAL_DIRECT_OFFGUIDE_CHAMFER_MEAN_DISTANCE_MAX = 11.0
GROOMING_CANONICAL_DIRECT_OFFGUIDE_CHAMFER_P90_DISTANCE_MAX = 35.0
GROOMING_CANONICAL_DIRECT_OFFGUIDE_PAW_LAB_DELTA_MAX = 8.0
GROOMING_CANONICAL_DIRECT_OFFGUIDE_PAW_HISTOGRAM_MAX = 0.15
GROOMING_CANONICAL_DIRECT_OFFGUIDE_PAW_TEXTURE_RATIO_MIN = 0.75
GROOMING_CANONICAL_DIRECT_OFFGUIDE_PAW_TEXTURE_RATIO_MAX = 2.0
GROOMING_CANONICAL_DIRECT_ALIGNED_STRUCTURAL_EDGE_DENSITY_MIN = 0.03
GROOMING_CANONICAL_DIRECT_ALIGNED_PAW_LAB_DELTA_MAX = 16.0
GROOMING_CANONICAL_DIRECT_ALIGNED_PAW_HISTOGRAM_MAX = 0.10
GROOMING_CANONICAL_DIRECT_ALIGNED_PAW_TEXTURE_RATIO_MIN = 0.75
GROOMING_CANONICAL_DIRECT_ALIGNED_PAW_TEXTURE_RATIO_MAX = 1.5
GROOMING_CANONICAL_DIRECT_ORGANIC_PAW_PALETTE_POLICY = (
    "canonical_direct_organic_paw_achromatic_histogram_match_v1"
)
GROOMING_CANONICAL_DIRECT_ORGANIC_PAW_ACHROMATIC_MIN = 0.95
GROOMING_CANONICAL_DIRECT_ORGANIC_PAW_BALANCE_L1_MAX = 0.90
GROOMING_CANONICAL_DIRECT_SURFACE_NONEXPANSION_POLICY = (
    "canonical_direct_identity_surface_debt_nonexpansion_v1"
)
# A canonical generated from the uploaded identity can carry a borderline
# directional-fur signal of its own. A direct action candidate may inherit that
# single issue only when every measured brush/line energy stays bounded and is
# non-expanding against the digest-bound transparent canonical.
GROOMING_CANONICAL_DIRECT_SURFACE_COHERENT_FRACTION_MAX = 0.57
GROOMING_CANONICAL_DIRECT_SURFACE_COHERENT_FRACTION_DELTA_MAX = 0.01
GROOMING_CANONICAL_DIRECT_SURFACE_COHERENCE_MEAN_MAX = 0.73
GROOMING_CANONICAL_DIRECT_SURFACE_COHERENCE_MEAN_DELTA_MAX = 0.02
GROOMING_CANONICAL_DIRECT_SURFACE_ORIENTATION_MAX = 0.72
GROOMING_CANONICAL_DIRECT_SURFACE_ORIENTATION_DELTA_MAX = 0.12
GROOMING_CANONICAL_DIRECT_SURFACE_HOUGH_RADIAL_DELTA_MAX = 0.03
GROOMING_CANONICAL_DIRECT_SURFACE_CANONICAL_LONG_ENERGY_MAX = 0.25
GROOMING_QWEN_FORELEG_REPAIR_ATTEMPT = 3
GROOMING_QWEN_FORELEG_REPAIR_DENOISE = 1.0
GROOMING_QWEN_FORELEG_REPAIR_STEPS = 12
GROOMING_CANONICAL_FULL_POSE_RESCUE_POLICY = (
    "canonical_soft_foreleg_depth_full_pose_rescue_v5"
)
GROOMING_CANONICAL_FULL_POSE_REFINE_POLICY = (
    "canonical_seed_bound_pose_safe_identity_refine_v6"
)
GROOMING_CANONICAL_POSE_TRANSFER_SEED_POLICY = (
    "canonical_palette_white_maquette_frame000_full_pose_transfer_seed_v9"
)
GROOMING_ADVISORY_FIRST_FRAME_INPUT_POLICY = (
    "canonical_image1_only_four_step_direct_action_prompt_"
    "no_secondary_or_video_frame_reference_"
    "raw_qwen_output_without_postprocess_v24"
)
GROOMING_ADVISORY_ACTION_TRANSFER_STEPS = 4
GROOMING_ADVISORY_ACTION_TRANSFER_DENOISE_BY_ATTEMPT = (
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
)
GROOMING_ADVISORY_ACTION_TRANSFER_DENOISE = (
    GROOMING_ADVISORY_ACTION_TRANSFER_DENOISE_BY_ATTEMPT[0]
)
GROOMING_ADVISORY_ACTION_TRANSFER_MAX_ATTEMPTS = len(
    GROOMING_ADVISORY_ACTION_TRANSFER_DENOISE_BY_ATTEMPT
)
GROOMING_ADVISORY_IDENTITY_REFINE_POLICY = (
    "pose_locked_draft_image1_canonical_image2_face_image3_conditional_masked_v7"
)
GROOMING_ADVISORY_IDENTITY_REFINE_STEPS = 12
GROOMING_ADVISORY_IDENTITY_REFINE_DENOISE = 0.86
GROOMING_ADVISORY_IDENTITY_REFINE_MAX_ATTEMPTS = 0
GROOMING_ADVISORY_IDENTITY_REFINE_HEAD_HEIGHT_FRACTION = 0.42
GROOMING_ADVISORY_IDENTITY_REFINE_HEAD_EXPANSION_PX = 0
GROOMING_ADVISORY_IDENTITY_REFINE_MASK_FEATHER_SIGMA_PX = 4.0
GROOMING_ADVISORY_IDENTITY_REFINE_SOURCE_ALLOWED_ISSUES = frozenset(
    {
        "new_dominant_dark_markings",
        "grooming_face_dark_marking_drift",
        "grooming_face_gross_chromatic_drift",
        "grooming_face_eye_hue_replacement",
    }
)
GROOMING_ADVISORY_IDENTITY_REFINE_SOURCE_BLACK_FRACTION_MAX = 0.05
GROOMING_ADVISORY_IDENTITY_REFINE_DARK_ARTIFACT_VALUE_MAX = 70
GROOMING_ADVISORY_IDENTITY_REFINE_DARK_ARTIFACT_MIN_FRACTION = 0.035
GROOMING_ADVISORY_IDENTITY_REFINE_DARK_ARTIFACT_EDGE_VALUE_MAX = 112
GROOMING_ADVISORY_IDENTITY_REFINE_DARK_ARTIFACT_EDGE_DILATION_PX = 2
GROOMING_ADVISORY_IDENTITY_REFINE_DARK_ARTIFACT_SUBJECT_SUPPORT_PX = 8
GROOMING_ADVISORY_TOTAL_ATTEMPTS = (
    GROOMING_ADVISORY_ACTION_TRANSFER_MAX_ATTEMPTS
    + GROOMING_ADVISORY_IDENTITY_REFINE_MAX_ATTEMPTS
)
GROOMING_ADVISORY_MINIMUM_GATE_POLICY = (
    "grooming_action_identity_minimum_contract_v9"
)
GROOMING_ADVISORY_MINIMUM_GATE_VERSION = "9.0"
GROOMING_ADVISORY_IDENTITY_PALETTE_L1_MAX = 0.48
GROOMING_ADVISORY_IDENTITY_BLACK_FRACTION_DELTA_MAX = 0.20
GROOMING_ADVISORY_IDENTITY_CHROMATIC_FRACTION_DELTA_MAX = 0.14
GROOMING_ADVISORY_IDENTITY_ACHROMATIC_BALANCE_L1_MAX = 0.38
GROOMING_ADVISORY_IDENTITY_NEW_DARK_DOMINANT_DELTA_MIN = 0.10
GROOMING_ADVISORY_FACE_DARK_FRACTION_DELTA_MAX = 0.14
GROOMING_ADVISORY_FACE_CHROMATIC_FRACTION_DELTA_MAX = 0.18
GROOMING_ADVISORY_FACE_YELLOW_FRACTION_DELTA_MAX = 0.025
GROOMING_ADVISORY_FACE_GREEN_TO_YELLOW_DELTA_MIN = 0.012
GROOMING_ADVISORY_HARD_OBJECT_TEXTURE_RATIO_MAX = 3.0
GROOMING_ADVISORY_HARD_OBJECT_RAISED_GRADIENT_P90_MIN = 200.0
GROOMING_ADVISORY_CANONICAL_ACTION_EDIT_MEAN_DELTA_MIN = 0.08
GROOMING_ADVISORY_CANONICAL_ACTION_EDIT_CHANGED_FRACTION_MIN = 0.30
GROOMING_ADVISORY_CANONICAL_PLANTED_PAW_EDGE_RETENTION_MAX = 0.70
GROOMING_ADVISORY_CANONICAL_PLANTED_PAW_RGB_DELTA_MIN = 0.12
GROOMING_ADVISORY_CANONICAL_PLANTED_PAW_FOREGROUND_REMOVED_FRACTION_MIN = 0.18
GROOMING_ADVISORY_CANONICAL_ACTION_THRESHOLD_SCALE_BY_ATTEMPT = (
    1.0,
    1.0,
    0.95,
    0.90,
    0.85,
)
GROOMING_ADVISORY_CANONICAL_EDGE_RETENTION_RELAXATION_BY_ATTEMPT = (
    0.0,
    0.0,
    0.03,
    0.06,
    0.08,
)
GROOMING_ADVISORY_BLOCKING_POSE_ISSUES = frozenset(
    {
        "canonical_grooming_action_edit_evidence_unavailable",
        "canonical_grooming_action_edit_too_weak",
        "original_planted_forepaw_topology_retained",
        "continuous_raised_foreleg_edge_unavailable",
        "raised_foreleg_structural_edge_density_too_low",
        "continuous_raised_foreleg_edge_too_short",
        "paw_muzzle_gap_unavailable",
        "raised_paw_too_far_from_muzzle",
        "raised_paw_hard_object_texture_signature",
    }
)
GROOMING_ADVISORY_ACTION_RETRY_WARNINGS = frozenset(
    {
        "canonical_grooming_action_edit_too_weak",
        "original_planted_forepaw_topology_retained",
        "raised_foreleg_structural_edge_density_too_low",
        "continuous_raised_foreleg_edge_too_short",
        "raised_paw_too_far_from_muzzle",
    }
)
GROOMING_ADVISORY_ACTION_RETRY_BEFORE_ATTEMPT = (
    GROOMING_ADVISORY_ACTION_TRANSFER_MAX_ATTEMPTS
)
GROOMING_ADVISORY_ACTION_EDIT_MASK_FEATHER_SIGMA_PX = 9.0
GROOMING_ADVISORY_ACTION_EDIT_MASK_CONTEXT_DILATION_PX = 12
GROOMING_ADVISORY_MASKED_REPAIR_UNSAFE_ISSUES = frozenset(
    {
        "raised_paw_hard_object_texture_signature",
        "grooming_face_dark_marking_drift",
        "grooming_face_gross_chromatic_drift",
        "grooming_face_eye_hue_replacement",
    }
)
GROOMING_ADVISORY_RETRY_ANATOMY_POLICY = (
    "previous_candidate_issue_driven_organic_foreleg_retry_v1"
)
GROOMING_CANONICAL_SEED_BOUND_MATTE_POLICY = (
    "canonical_pose_transfer_seed_bound_matte_union_v1"
)
GROOMING_CANONICAL_SEED_BOUND_MATTE_ALPHA_PROVENANCE = (
    "rembg_u2net_plus_digest_bound_pose_seed"
)
GROOMING_CANONICAL_POSE_SAFE_REFINE_SOURCE_POLICY = (
    "canonical_seed_bound_pose_safe_identity_refine_source_v1"
)
GROOMING_CANONICAL_SEED_BOUND_INTERSECTION_OVER_SEED_MIN = 0.50
GROOMING_CANONICAL_SEED_BOUND_INTERSECTION_OVER_SEMANTIC_MIN = 0.85
GROOMING_CANONICAL_SEED_BOUND_SEMANTIC_TO_SEED_MIN = 0.45
GROOMING_CANONICAL_SEED_BOUND_SEMANTIC_TO_SEED_MAX = 1.15
GROOMING_CANONICAL_SEED_BOUND_ADDED_OVER_SEED_MAX = 0.45
GROOMING_CANONICAL_POSE_SAFE_STRUCTURAL_EDGE_DENSITY_MIN = 0.02
GROOMING_CANONICAL_POSE_SAFE_IDENTITY_ONLY_ISSUES = frozenset(
    {
        "new_dominant_warm_brown_or_orange",
        "achromatic_source_recolored_with_warm_fur",
        "achromatic_coat_value_balance_lost",
        "canonical_action_achromatic_value_balance_drift",
    }
)
GROOMING_CANONICAL_FULL_POSE_GUIDE_POLICY = (
    "single_tone_open_channel_organic_grooming_geometry_v3"
)
GROOMING_CANONICAL_FULL_POSE_GUIDE_LUMA_HALF_SPAN = 12
GROOMING_CANONICAL_FULL_POSE_RESCUE_ATTEMPT = 4
GROOMING_CANONICAL_FULL_POSE_REFINE_ATTEMPT = 5
GROOMING_CANONICAL_FULL_POSE_RESCUE_STEPS = 12
GROOMING_CANONICAL_FULL_POSE_RESCUE_DENOISE = 1.0
GROOMING_CANONICAL_FULL_POSE_REFINE_DENOISE = 1.0
GROOMING_CANONICAL_POSE_TRANSFER_GEOMETRY_SMOOTH_SIGMA_PX = 4.0
GROOMING_CANONICAL_POSE_TRANSFER_GEOMETRY_LUMA_HALF_RANGE = 6.0
GROOMING_CANONICAL_POSE_TRANSFER_FORELEG_SMOOTH_SIGMA_PX = 5.0
GROOMING_CANONICAL_POSE_TRANSFER_FORELEG_LUMA_LIFT = 12.0
GROOMING_WHITE_MAQUETTE_DEPTH_SMOOTH_SIGMA_PX = 2.5
GROOMING_WHITE_MAQUETTE_DEPTH_LUMA_HALF_RANGE = 18.0
GROOMING_WHITE_MAQUETTE_FRAME0_POSE_LUMA_HALF_RANGE = 32.0
GROOMING_WHITE_MAQUETTE_DEPTH_DISTANCE_BLEND = 0.18
EATING_CANONICAL_IDENTITY_REFINE_POLICY = (
    "canonical_eating_pose_uploaded_front_identity_refine_v2"
)
EATING_CANONICAL_IDENTITY_REFINE_ATTEMPT = 5
EATING_CANONICAL_IDENTITY_REFINE_STEPS = 12
EATING_CANONICAL_IDENTITY_REFINE_DENOISE = 1.0
GROOMING_QWEN_FORELEG_REPAIR_MASK_DILATION_PX = 18
GROOMING_QWEN_FORELEG_REPAIR_SEED_LOWPASS_SIGMA_PX = 2.5
GROOMING_QWEN_FORELEG_REPAIR_SEED_IDENTITY_MEDIAN_BLEND = 0.12
GROOMING_QWEN_FORELEG_REPAIR_NOISE_MASK_EXPANSION_PX = 32
GROOMING_QWEN_FORELEG_REPAIR_NOISE_MASK_FEATHER_SIGMA_PX = 8.0
GROOMING_QWEN_FORELEG_REPAIR_STYLE_STRENGTH = float(
    os.environ.get("GROOMING_QWEN_FORELEG_REPAIR_STYLE_STRENGTH", "0.22")
)
GROOMING_QWEN_FORELEG_REPAIR_STYLE_POLICY = (
    "expanded_noise_mask_local_pet3d_surface_prior_v1"
)
GROOMING_QWEN_FORELEG_REPAIR_COMPOSITE_GATE_POLICY = (
    "masked_pet3d_surface_verified_pose_edge_and_low_texture_paw_rescue_v3"
)
GROOMING_QWEN_FORELEG_REPAIR_EDGE_DENSITY_RATIO_MIN = 0.70
GROOMING_QWEN_FORELEG_REPAIR_GLOBAL_EDGE_DENSITY_MIN = 0.11
GROOMING_POSE_DEFAULT_THRESHOLDS = GroomingPoseThresholds()
GROOMING_POSE_STRUCTURAL_BLUR_SIGMA = (
    GROOMING_POSE_DEFAULT_THRESHOLDS.structural_blur_sigma
)
GROOMING_POSE_STRUCTURAL_EDGE_DENSITY_MIN = (
    GROOMING_POSE_DEFAULT_THRESHOLDS.structural_edge_density_min
)
GROOMING_QWEN_FORELEG_REPAIR_PLANTED_PAW_GRADIENT_P90_MAX = 8.0
GROOMING_QWEN_FORELEG_REPAIR_RAISED_PAW_GRADIENT_P90_MAX = 128.0
GROOMING_QWEN_FORELEG_REPAIR_ANATOMY_POLICY = (
    "frozen_single_tone_open_channel_grooming_anatomy_image3_v2"
)
UPLOADED_IDENTITY_VIEW_POLICY = "front_left_right_qwen_canonical_identity_v1"
CANONICAL_UPPER_BODY_PALETTE_REFERENCE_POLICY = (
    "front_view_for_front_three_quarter_canonical_v1"
)
UPLOADED_IDENTITY_VIEW_ROLES = ("front", "left", "right")
UPLOADED_IDENTITY_VIEW_FILES = {
    "front": ("input.png", "input_identity_source.png"),
    "left": ("input_left.png", "input_left_identity_source.png"),
    "right": ("input_right.png", "input_right_identity_source.png"),
}
QWEN_MULTI_VIEW_CANONICAL_PROMPT = (
    "MULTI-VIEW IDENTITY CONTRACT: Images 1, 2, and 3 are front, left-side, "
    "and right-side photographs of the same single pet. Generate exactly one "
    "pet. Use Image 1 as the primary face and frontal-identity authority; use "
    "Images 2 and 3 to preserve side-specific coat markings, body proportions, "
    "tail, legs, and asymmetry. Reconcile the three observations into one "
    "coherent full-body identity. Do not average markings, mirror one side onto "
    "the other, combine multiple animals, or place reference panels in output."
)
FINAL_VIDEO_VALIDATED_ACTION_REFERENCE_POLICIES = {
    "eating": "qwen_action_identity_with_bowl_mapped_red",
    "grooming": "qwen_action_specific_identity",
}
EATING_MOUTH_BOWL_CONTACT_POLICY = "warm_face_component_above_detected_bowl_v3"
EATING_BOWL_SPLIT_FALLBACK_POLICY = (
    "food_anchored_beige_bowl_after_eroded_warm_core_miss_v2"
)
EATING_MUZZLE_BOWL_GAP_MAX_PX = 8
EATING_MUZZLE_COMPONENT_WIDTH_BOWL_RATIO_MAX = 0.70
EATING_MUZZLE_COMPONENT_HEIGHT_BOWL_RATIO_MAX = 0.35
EATING_MUZZLE_COMPONENT_CENTER_MARGIN_BOWL_RATIO = 0.10
FULL_DRIVER_EATING_POSE_POLICY = "full_driver_pose_with_separate_bowl_v1"
UPLOADED_ABSOLUTE_FUR_CONTOUR_POLICY = (
    "uploaded_subject_absolute_animation_roughness_v1"
)
UPLOADED_IDENTITY_SURFACE_COMPARISON_MODE = (
    "uploaded_identity_absolute_surface_contract_v1"
)
UPLOADED_IDENTITY_CHROMA_SPILL_POLICY = (
    "source_palette_guarded_green_spill_neutralization_v1"
)
BOWL_IDENTITY_PALETTE_BOTTOM_EXCLUSION = max(
    0.0,
    min(
        0.45,
        float(os.environ.get("BOWL_IDENTITY_PALETTE_BOTTOM_EXCLUSION", "0.22")),
    ),
)
BOWL_IDENTITY_PALETTE_UPPER_BODY_EXCLUSION = max(
    BOWL_IDENTITY_PALETTE_BOTTOM_EXCLUSION,
    min(
        0.45,
        float(
            os.environ.get(
                "BOWL_IDENTITY_PALETTE_UPPER_BODY_EXCLUSION",
                "0.45",
            )
        ),
    ),
)
QWEN_POSE_SIMILARITY_IOU_THRESHOLD = float(
    os.environ.get("QWEN_POSE_SIMILARITY_IOU_THRESHOLD", "0.94")
)
QWEN_POSE_SIMILARITY_DHASH_THRESHOLD = int(
    os.environ.get("QWEN_POSE_SIMILARITY_DHASH_THRESHOLD", "6")
)
QWEN_POSE_RELATED_IOU_THRESHOLD = float(
    os.environ.get("QWEN_POSE_RELATED_IOU_THRESHOLD", "0.975")
)
QWEN_POSE_RELATED_DHASH_THRESHOLD = int(
    os.environ.get("QWEN_POSE_RELATED_DHASH_THRESHOLD", "3")
)
QWEN_ACTION_CANONICAL_IOU_THRESHOLD = float(
    os.environ.get("QWEN_ACTION_CANONICAL_IOU_THRESHOLD", "0.90")
)
QWEN_ACTION_CANONICAL_DHASH_THRESHOLD = int(
    os.environ.get("QWEN_ACTION_CANONICAL_DHASH_THRESHOLD", "10")
)
QWEN_BOWL_ACTION_MIN_ASPECT_RATIO = float(
    os.environ.get("QWEN_BOWL_ACTION_MIN_ASPECT_RATIO", "1.05")
)
QWEN_SUBJECT_REPLACEMENT_POSE_DENOISE = max(
    0.1,
    min(
        1.0,
        float(os.environ.get("QWEN_SUBJECT_REPLACEMENT_POSE_DENOISE", "1.0")),
    ),
)
QWEN_SUBJECT_REPLACEMENT_POSE_DENOISE_STEP = max(
    0.0,
    min(
        0.25,
        float(
            os.environ.get(
                "QWEN_SUBJECT_REPLACEMENT_POSE_DENOISE_STEP",
                "0.0",
            )
        ),
    ),
)
QWEN_EATING_GEOMETRY_FIRST_DENOISE_HIGH = max(
    0.1,
    min(
        1.0,
        float(
            os.environ.get(
                "QWEN_EATING_GEOMETRY_FIRST_DENOISE_HIGH",
                "1.0",
            )
        ),
    ),
)
QWEN_EATING_GEOMETRY_FIRST_DENOISE_LOW = max(
    0.1,
    min(
        1.0,
        float(
            os.environ.get(
                "QWEN_EATING_GEOMETRY_FIRST_DENOISE_LOW",
                "0.90",
            )
        ),
    ),
)
QWEN_EATING_GEOMETRY_FIRST_STYLE_STRENGTH = max(
    0.0,
    min(
        0.25,
        float(
            os.environ.get(
                "QWEN_EATING_GEOMETRY_FIRST_STYLE_STRENGTH",
                "0.15",
            )
        ),
    ),
)
QWEN_POSE_HARD_DUPLICATE_IOU = float(
    os.environ.get("QWEN_POSE_HARD_DUPLICATE_IOU", "0.985")
)
QWEN_POSE_HARD_DUPLICATE_DHASH = int(
    os.environ.get("QWEN_POSE_HARD_DUPLICATE_DHASH", "2")
)
CHROMA_FALLBACK_REMBG_MODEL = (
    os.environ.get("CHROMA_FALLBACK_REMBG_MODEL", "u2net").strip() or "u2net"
)
MAX_LORA_UPLOAD_BYTES = int(
    os.environ.get("LOCAL_VIDEO_MAX_LORA_UPLOAD_BYTES", str(2 * 1024 * 1024 * 1024))
)
MAX_USER_LORAS = max(1, min(8, int(os.environ.get("LOCAL_VIDEO_MAX_USER_LORAS", "4"))))
MAX_LORA_FUSION_BYTES = int(
    os.environ.get("LOCAL_VIDEO_MAX_LORA_FUSION_BYTES", str(4 * 1024 * 1024 * 1024))
)
MAX_QUEUE_SIZE = int(os.environ.get("LOCAL_VIDEO_MAX_QUEUE_SIZE", "8"))
MAX_POSES = int(os.environ.get("LOCAL_VIDEO_MAX_POSES", "6"))
TRANSPARENT_GIF_SIZE = 512
TRANSPARENT_GIF_FPS = 20
TRANSPARENT_GIF_FRAMES = 100
TRANSPARENT_GIF_DURATION_SECONDS = 5.0
QWEN_ACTION_CANDIDATE_MATTE_POLICY = (
    "direct_chroma_then_per_candidate_semantic_matte_v1"
)
QWEN_ACTION_REFERENCE_NORMALIZATION_POLICY = (
    "alpha_preserving_rgba_canvas_512_then_byte_copy_v1"
)
FFMPEG_BIN = os.environ.get(
    "FFMPEG_BIN", shutil.which("ffmpeg") or "/usr/local/bin/ffmpeg"
)

SUBJECT_PROFILE_DIR = PROJECT_ROOT / "profiles"
DEFAULT_SUBJECT_PROFILE = (
    os.environ.get("DEFAULT_SUBJECT_PROFILE", "sleek_tabby_cat_v1").strip()
    or "sleek_tabby_cat_v1"
)
SUBJECT_REPLACEMENT_ACTIONS = ("eating", "grooming")
SUBJECT_REPLACEMENT_ACTION_LABELS = {
    "eating": "吃饭",
    "grooming": "舔毛",
}
SUBJECT_REPLACEMENT_REFERENCE_IMAGE_PROFILE = "pet-multipose-20260717-v2"
SCAIL_PYTHON = Path(
    os.environ.get(
        "SCAIL_PYTHON",
        "/opt/miniconda3/envs/comfyui/bin/python",
    )
).resolve()
SCAIL_RUNNER = PROJECT_ROOT / "scail2_subject_replacement_test.py"
SCAIL_TRANSPARENT_RUNNER = PROJECT_ROOT / "scail2_transparent_gif_from_video.py"
SCAIL_EXPERIMENT_ROOT = DATA_ROOT / "experiments" / "scail2_tm_test"
SCAIL_UNET = "wan2.1_14B_SCAIL_2_fp8_scaled.safetensors"
SCAIL_DPO_LORA = "wan2.1_SCAIL_2_DPO_lora_bf16.safetensors"
SCAIL_DISTILL_LORA = "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"
SCAIL_FRAMES = 81
SCAIL_FPS = 16.0
SCAIL_STEPS = 6
SCAIL_EATING_POSE_STRENGTH_MAX = 1.35
SCAIL_EATING_REFERENCE_PROP_SPLIT_POLICY = (
    "strict_visual_first_then_nonblocking_lower_center_geometry_v1"
)
SCAIL_DRIVING_TEMPLATE_ROOT = (
    DATA_ROOT
    / "profile_assets"
    / "sleek_tabby_cat_v1"
    / "driving"
    / "20260731-main-case-v1"
)
SCAIL_ACTION_CONFIG = {
    "eating": {
        "driving_template": (
            SCAIL_DRIVING_TEMPLATE_ROOT / "eating_driver_512x512_81f_16fps.mp4"
        ),
        "pose_strength": 1.10,
        "seed_offset": 0,
    },
    "grooming": {
        "driving_template": (
            SCAIL_DRIVING_TEMPLATE_ROOT / "grooming_driver_512x512_81f_16fps.mp4"
        ),
        "pose_strength": 1.15,
        "seed_offset": 2,
    },
}

SUBJECT_REPLACEMENT_GROOMING_POSE_PROMPT = (
    "grooming | A complete full-body cat in a stable seated grooming pose at a "
    "three-quarter front angle. Exactly one foreleg rises continuously and visibly "
    "from its shoulder to one front paw held at the mouth. The other foreleg descends "
    "continuously from its shoulder to exactly one planted front paw. The raised "
    "foreleg has no duplicate or leftover paw on the ground. Both hind legs remain "
    "naturally seated at the sides and may be partly tucked behind the crouched torso. "
    "A single tiny pink tongue touches the raised paw. Show exactly four anatomically "
    "attached legs, both complete front paws, both ears, the complete body, and one "
    "tail continuously from base to tip; the attached hind paws may be naturally "
    "partly occluded by the seated pose. No food, "
    "bowl, collar, clothing, accessory, detached paw, ghost limb, or extra subject."
)

QWEN_MODEL = "qwen_image_edit_2511_fp8mixed.safetensors"
QWEN_CLIP = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
QWEN_VAE = "qwen_image_vae.safetensors"
QWEN_LORA = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
QWEN_STYLE_LORA = os.environ.get("QWEN_STYLE_LORA", "").strip()
QWEN_STYLE_LORA_STRENGTH = float(os.environ.get("QWEN_STYLE_LORA_STRENGTH", "0.65"))
QWEN_POSE_STYLE_LORA_STRENGTH = float(
    os.environ.get("QWEN_POSE_STYLE_LORA_STRENGTH", "0.25")
)
QWEN_STYLE_PROFILE = (
    os.environ.get("QWEN_STYLE_PROFILE", "baseline").strip() or "baseline"
)
WAN_ENGINE = os.environ.get("WAN_ENGINE", "wan2.2-ti2v-5b").strip().lower()
if WAN_ENGINE not in {"wan2.2-ti2v-5b", "wan2.2-i2v-a14b-fp8"}:
    raise ValueError("WAN_ENGINE must be wan2.2-ti2v-5b or wan2.2-i2v-a14b-fp8")
WAN_MODEL = os.environ.get("WAN_MODEL", "wan2.2_ti2v_5B_fp16.safetensors").strip()
WAN_HIGH_NOISE_MODEL = os.environ.get(
    "WAN_HIGH_NOISE_MODEL", "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
).strip()
WAN_LOW_NOISE_MODEL = os.environ.get(
    "WAN_LOW_NOISE_MODEL", "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
).strip()
WAN_CLIP = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
WAN_VAE = os.environ.get(
    "WAN_VAE",
    "wan_2.1_vae.safetensors"
    if WAN_ENGINE == "wan2.2-i2v-a14b-fp8"
    else "wan2.2_vae.safetensors",
).strip()
FRAME_COUNT = int(os.environ.get("WAN_FRAME_COUNT", "121"))
FPS = float(os.environ.get("WAN_FPS", "24"))
WAN_STEPS = int(os.environ.get("WAN_STEPS", "20"))
WAN_SPLIT_STEP = int(os.environ.get("WAN_SPLIT_STEP", str(WAN_STEPS // 2)))
WAN_CFG = float(
    os.environ.get("WAN_CFG", "3.5" if WAN_ENGINE == "wan2.2-i2v-a14b-fp8" else "4.0")
)
WAN_SHIFT = float(
    os.environ.get("WAN_SHIFT", "5.0" if WAN_ENGINE == "wan2.2-i2v-a14b-fp8" else "8.0")
)

if FRAME_COUNT < 1 or FRAME_COUNT % 4 != 1:
    raise ValueError("WAN_FRAME_COUNT must be positive and equal to 4n+1")
if WAN_STEPS < 2 or not 1 <= WAN_SPLIT_STEP < WAN_STEPS:
    raise ValueError("WAN_SPLIT_STEP must be between 1 and WAN_STEPS - 1")

FUN_CONTROL_ENABLED = os.environ.get(
    "FUN_CONTROL_ENABLED", "false"
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
FUN_CONTROL_FALLBACK_TO_WAN = os.environ.get(
    "FUN_CONTROL_FALLBACK_TO_WAN", "true"
).strip().lower() in {"1", "true", "yes", "on"}
FUN_CONTROL_PYTHON = (
    Path(
        os.environ.get(
            "FUN_CONTROL_PYTHON",
            "/data/comfy-pet-pipeline/experiments/videoxfun-venv/bin/python",
        )
    )
    .expanduser()
    .absolute()
)
FUN_CONTROL_SCRIPT = Path(
    os.environ.get(
        "FUN_CONTROL_SCRIPT",
        "/data/comfy-pet-pipeline/experiments/wan22_fun_5b_control_test/wan22_fun_5b_control_ref_test.py",
    )
).resolve()
FUN_CONTROL_PROJECT_ROOT = Path(
    os.environ.get(
        "FUN_CONTROL_PROJECT_ROOT",
        "/data/comfy-pet-pipeline/experiments/VideoX-Fun",
    )
).resolve()
FUN_CONTROL_MODEL_ROOT = Path(
    os.environ.get(
        "FUN_CONTROL_MODEL_ROOT",
        "/data/comfy-pet-pipeline/models/videoxfun/Wan2.2-Fun-5B-Control",
    )
).resolve()
FUN_CONTROL_RUNNING_VIDEO = Path(
    os.environ.get(
        "FUN_CONTROL_RUNNING_VIDEO",
        "/data/comfy-pet-pipeline/experiments/wan22_fun_5b_control_test/muybridge_cat_canny_512_121f_24fps.mp4",
    )
).resolve()
FUN_CONTROL_STEPS = int(os.environ.get("FUN_CONTROL_STEPS", "30"))
FUN_CONTROL_TIMEOUT = int(os.environ.get("FUN_CONTROL_TIMEOUT", "7200"))
FUN_CONTROL_GPU_MEMORY_MODE = os.environ.get(
    "FUN_CONTROL_GPU_MEMORY_MODE", "model_cpu_offload"
).strip()

if not 0.0 <= QWEN_STYLE_LORA_STRENGTH <= 2.0:
    raise ValueError("QWEN_STYLE_LORA_STRENGTH must be between 0.0 and 2.0")
if not 0.0 <= QWEN_POSE_STYLE_LORA_STRENGTH <= 2.0:
    raise ValueError("QWEN_POSE_STYLE_LORA_STRENGTH must be between 0.0 and 2.0")
if not 0.0 <= GROOMING_QWEN_FORELEG_REPAIR_STYLE_STRENGTH <= 2.0:
    raise ValueError(
        "GROOMING_QWEN_FORELEG_REPAIR_STYLE_STRENGTH must be between 0.0 and 2.0"
    )
if not 0.0 <= QWEN_RECOVERY_USER_LORA_STRENGTH <= 2.0:
    raise ValueError("QWEN_RECOVERY_USER_LORA_STRENGTH must be between 0.0 and 2.0")
for name, value in {
    "QWEN_CANONICAL_USER_LORA_TOTAL_CAP": QWEN_CANONICAL_USER_LORA_TOTAL_CAP,
    "QWEN_CANONICAL_USER_LORA_PER_LORA_CAP": QWEN_CANONICAL_USER_LORA_PER_LORA_CAP,
    "QWEN_POSE_USER_LORA_TOTAL_CAP": QWEN_POSE_USER_LORA_TOTAL_CAP,
    "QWEN_POSE_USER_LORA_RETRY_TOTAL_CAP": QWEN_POSE_USER_LORA_RETRY_TOTAL_CAP,
    "QWEN_POSE_USER_LORA_FINAL_TOTAL_CAP": QWEN_POSE_USER_LORA_FINAL_TOTAL_CAP,
    "QWEN_POSE_USER_LORA_PER_LORA_CAP": QWEN_POSE_USER_LORA_PER_LORA_CAP,
}.items():
    if not 0.0 <= value <= 2.0:
        raise ValueError(f"{name} must be between 0.0 and 2.0")
for name, value in {
    "QWEN_POSE_SIMILARITY_IOU_THRESHOLD": QWEN_POSE_SIMILARITY_IOU_THRESHOLD,
    "QWEN_POSE_RELATED_IOU_THRESHOLD": QWEN_POSE_RELATED_IOU_THRESHOLD,
    "QWEN_POSE_HARD_DUPLICATE_IOU": QWEN_POSE_HARD_DUPLICATE_IOU,
}.items():
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0")
for name, value in {
    "QWEN_POSE_SIMILARITY_DHASH_THRESHOLD": QWEN_POSE_SIMILARITY_DHASH_THRESHOLD,
    "QWEN_POSE_RELATED_DHASH_THRESHOLD": QWEN_POSE_RELATED_DHASH_THRESHOLD,
    "QWEN_POSE_HARD_DUPLICATE_DHASH": QWEN_POSE_HARD_DUPLICATE_DHASH,
}.items():
    if not 0 <= value <= 64:
        raise ValueError(f"{name} must be between 0 and 64")

LEGACY_EATING_IMAGE_PROFILE_ID = "20260717T094459Z-f738613526"
MULTIPOSE_V1_IMAGE_PROFILE_ID = "pet-multipose-20260717-v1"
BASELINE_IMAGE_PROFILE_ID = "pet-multipose-20260717-v2"
BASELINE_IMAGE_SOURCE_JOB_ID = LEGACY_EATING_IMAGE_PROFILE_ID
BASELINE_IMAGE_PROFILE_REVISION = "multipose-v2"
BASELINE_IMAGE_SOURCE_PROMPT_ZH = (
    "俯身低头，嘴巴凑近浅米色陶瓷猫粮碗，碗内装有棕色颗粒猫粮，正在进食的自然动态"
)
BASELINE_IMAGE_SOURCE_PROMPT_EN = (
    "A natural eating action: the cat leans forward with its head lowered, "
    "bringing its mouth close to a shallow light-beige ceramic food bowl "
    "filled with brown kibble."
)
BASELINE_IMAGE_PROMPT_ZH = (
    "以参考图片中的猫咪为唯一身份标准，完整保留毛色和花纹位置、脸型、"
    "眼睛颜色与间距、耳形、鼻色、毛长、体型比例和尾巴特征；生成高品质"
    "萌系3D卡通角色，毛发细腻柔软，光照明亮柔和，保持自然无衣服无配饰"
    "的完整全身形象。不继承或固定参考图中的姿态、肢体坐标和构图，具体"
    "动作只由所选姿态决定。"
)
BASELINE_IMAGE_PROMPT_EN = (
    "Use the cat in the reference image as the only identity standard. Preserve "
    "its exact fur colors and marking placement, face shape, eye color and spacing, "
    "ear shape, nose color, natural coat-length category, body proportions, and tail "
    "features. Render it as an identity-faithful restrained stylized 3D digital double "
    "with the same apparent age, head-to-body ratio, and eye-to-head ratio, using compact, "
    "finely groomed soft fur lying naturally along the body surface "
    "and bright gentle lighting, showing its complete natural unclothed body with "
    "no accessories. Do not inherit or lock the reference pose, limb coordinates, "
    "or composition; the selected pose alone controls the action."
)
MULTIPOSE_PRESETS: tuple[dict[str, str], ...] = (
    {
        "id": "running",
        "action_key": "running",
        "label": "奔跑",
        "prompt_zh": (
            "三分之四侧面完整全身视角，呈稳定的起跑准备站姿；身体略微前倾，"
            "头部自然抬起，四条腿完整可见并前后错开，四只完整爪全部清晰压在"
            "同一地面基线上且没有悬空缝隙，远侧腿不能被身体遮住或与近侧腿"
            "重叠；尾巴从根部到末端完整可见。这是供后续动画使用的静态落地"
            "准备姿态，不是奔跑瞬间、跳跃或四脚腾空"
        ),
        "prompt_en": (
            "A complete full-body cat at a three-quarter side angle in a stable static "
            "pre-run standing contact pose. The torso leans slightly forward and the "
            "head is naturally raised. Exactly four complete legs are fully visible, "
            "anatomically attached, and staggered front-to-back so the far-side legs do "
            "not hide behind or merge with the near-side legs. All four complete paws "
            "are visibly planted and flattened on the same horizontal baseline with no "
            "gap below them. The tail is visible continuously from base to tip. This is "
            "a calm grounded ready stance for later animation, not an illustrated "
            "running instant, jump, leap, flight, or gallop suspension. No prop."
        ),
        "video_prompt_en": (
            "Animate a clear moderate-speed feline run-in-place cycle. Use a readable "
            "quadruped gait: the hind legs gather and push in sequence, the spine "
            "flexes and extends slightly, the forelegs reach and land in sequence, "
            "and each paw returns beneath the torso. Keep all four legs distinct and "
            "attached; a weight-bearing paw must not slide. Allow a small coordinated "
            "body rise and fall and a gentle tail counter-sway. Keep the global body "
            "center near the same position, with no forward travel or long airborne "
            "suspension, and finish near the starting contact phase for a clean loop."
        ),
    },
    {
        "id": "eating",
        "action_key": "eating",
        "label": "吃饭",
        "prompt_zh": BASELINE_IMAGE_SOURCE_PROMPT_ZH,
        "prompt_en": (
            "A complete full-body cat in a natural low crouch, leaning forward with "
            "the head fully lowered so the lower muzzle physically enters and overlaps "
            "the clearly visible brown kibble by 6 to 10 output pixels in exactly one "
            "small shallow light-beige matte ceramic food bowl. The nose and lower "
            "muzzle may be partially hidden by kibble and the near bowl rim; this "
            "occlusion is required evidence of eating, not a framing defect. "
            "The bowl touches or slightly overlaps the front paws and is the only prop. "
            "Both complete front paws are visibly planted; the cat never grasps, lifts, "
            "carries, or holds food. Keep the paws, body, ears, upper face markings, and "
            "full tail readable, but never raise the head merely to show the eyes, nose, "
            "or complete face. Do not let the bowl hide or replace any limb."
        ),
        "video_prompt_en": (
            "Keep the food bowl completely fixed. Animate a restrained natural eating "
            "loop by lowering the muzzle a few millimeters into the bowl and raising it "
            "back, with tiny coordinated neck and lower-jaw movement plus "
            "subtle breathing through the chest and body. Move the head and face as "
            "one anatomical structure; do not stretch the mouth or reshape the cheeks, "
            "eyes, or nose. Keep both front paws planted and motionless beside the bowl; "
            "the cat must not lift a paw to the mouth or hold food. Kibble must not fly, "
            "appear, or disappear."
        ),
    },
    {
        "id": "drinking",
        "action_key": "drinking",
        "label": "喝水",
        "prompt_zh": (
            "俯身低头，嘴巴凑近浅米色陶瓷水碗，碗内装有平静清水，一小段短小"
            "粉色三角形舌尖从下唇伸出并与水面相交接触，舌尖长度不超过头部"
            "宽度的二十分之一，正在喝水的自然动态"
        ),
        "prompt_en": (
            "A complete full-body cat in a natural low crouch, leaning forward with "
            "the head lowered toward exactly one small shallow light-beige matte "
            "ceramic water bowl containing still clear water. The bowl sits immediately "
            "beside the front paws with a narrow visible green gap and never touches, "
            "overlaps, hides, or replaces a paw. Exactly one tiny solid-pink triangular "
            "tongue tip emerges from the lower lip and visibly intersects the water "
            "surface with no air gap; its visible length is no more than one twentieth "
            "of the head width. Keep the eyes, face silhouette, paws, body, and full "
            "tail readable; no long tongue, splash, drool, or water strand."
        ),
        "video_prompt_en": (
            "Keep the water bowl completely fixed. Animate restrained lapping: one "
            "short pink tongue tip extends and retracts naturally with extremely small "
            "coordinated muzzle and neck movement, subtle breathing, and soft localized "
            "ripples on the water. Preserve the face as one coherent anatomical "
            "structure. No splash, drool, transparent or elongated tongue, or "
            "continuous water strand."
        ),
    },
    {
        "id": "lying_down",
        "action_key": "lying down",
        "label": "趴卧",
        "prompt_zh": (
            "身体自然降低并完整趴卧，胸腹贴近地面，前爪平行向前伸出，后腿自然"
            "收拢，头部轻轻抬起，尾巴沿身体一侧完整弯曲，安静趴卧的自然状态"
        ),
        "prompt_en": (
            "A complete full-body cat in a relaxed low sphinx-like prone pose. The "
            "chest and abdomen are close to the invisible baseline, both front paws "
            "extend forward in parallel, the hind legs fold naturally beside the body, "
            "the head is gently raised, and the tail curves continuously along one side "
            "from base to tip. No prop and no cropped belly, haunch, paw, or tail."
        ),
        "video_prompt_en": (
            "Maintain the relaxed prone pose. Animate slow subtle breathing through "
            "the chest and abdomen, a small naturally coordinated head-and-ear response, "
            "and one gentle tail-tip movement. Let the whole body respond softly to "
            "breathing instead of freezing the upper body; prevent global sliding or "
            "scale drift."
        ),
    },
)
MULTIPOSE_PRESET_BY_ACTION = {
    preset["action_key"]: preset for preset in MULTIPOSE_PRESETS
}
MULTIPOSE_PRESET_BY_PROMPT_ZH = {
    preset["prompt_zh"]: preset for preset in MULTIPOSE_PRESETS
}
MULTIPOSE_SEED_OFFSETS = {
    "running": 101,
    "eating": 211,
    "drinking": 307,
    "lying down": 401,
}
DEFAULT_POSES = [preset["prompt_zh"] for preset in MULTIPOSE_PRESETS]
LEGACY_DEFAULT_POSES = [
    "running | first frame: grounded three-quarter side-view start pose, three paws on one baseline, near front paw slightly raised, four complete separated legs and paws, full tail; video: smooth run-in-place cycle with rhythmic legs and tail counter-sway",
    "eating | first frame: complete full-body natural crouch with muzzle at one small shallow food bowl touching the front paws; video: restrained eating rhythm while the bowl stays fixed",
    "drinking | first frame: complete full-body natural crouch with muzzle at one small shallow water bowl touching the front paws; video: subtle lapping rhythm and small water ripples while the bowl stays fixed",
    "lying down | first frame: complete full-body relaxed sphinx-like prone pose with front paws forward and full tail visible; video: subtle breathing and one slow tail-tip movement",
]
BASELINE_IMAGE_USER_LORAS = [
    {
        "model_name": "user_uploads/70e0a15b036a79af-v3.0.safetensors",
        "original_filename": "宠物小猫 _ 写实摄影_v3.0.safetensors",
        "strength": 0.5,
    },
    {
        "model_name": (
            "user_uploads/"
            "a5c10317e202894c-Qwen-Image-Kitten-Photography_QWEN.safetensors"
        ),
        "original_filename": (
            "Qwen Image-Kitten Photography_QWEN-小猫摄影.safetensors"
        ),
        "strength": 0.6,
    },
]
BASELINE_IMAGE_GENERATION_SETTINGS = {
    "user_lora_mode": "replace",
    "qwen_steps": 6,
    "qwen_cfg": 1.0,
    "qwen_denoise": 1.0,
    "seed": 20260715,
    # Preserve the source task on attempt 1; retries still use the global
    # reduced/final caps and the duplicate-pose gate.
    "canonical_total_cap": 1.1,
    "canonical_per_lora_cap": 0.6,
    "pose_first_total_cap": 1.1,
    "pose_first_per_lora_cap": 0.6,
    "lock_retry_lora_strengths": True,
}

IMAGE_PROFILES: dict[str, dict[str, Any]] = {
    BASELINE_IMAGE_PROFILE_ID: {
        "id": BASELINE_IMAGE_PROFILE_ID,
        "revision": BASELINE_IMAGE_PROFILE_REVISION,
        "label": "20260717 小猫四姿态统一基准 v2",
        "source_job_id": BASELINE_IMAGE_SOURCE_JOB_ID,
        "source_prompt": BASELINE_IMAGE_SOURCE_PROMPT_ZH,
        "source_prompt_en": BASELINE_IMAGE_SOURCE_PROMPT_EN,
        "default_prompt": BASELINE_IMAGE_PROMPT_ZH,
        "default_prompt_en": BASELINE_IMAGE_PROMPT_EN,
        "pose_prompt_source": "verified_profile_presets",
        "pose_presets": [dict(preset) for preset in MULTIPOSE_PRESETS],
        "pose_seed_offsets": dict(MULTIPOSE_SEED_OFFSETS),
        "default_poses": list(DEFAULT_POSES),
        "identity_palette_guard": True,
        "identity_palette_policy": "source_palette_guard_v2",
        "user_loras": [dict(item) for item in BASELINE_IMAGE_USER_LORAS],
        **BASELINE_IMAGE_GENERATION_SETTINGS,
    },
    MULTIPOSE_V1_IMAGE_PROFILE_ID: {
        "id": MULTIPOSE_V1_IMAGE_PROFILE_ID,
        "revision": "multipose-v1",
        "label": "20260717 小猫四姿态统一基准 v1（历史兼容）",
        "source_job_id": BASELINE_IMAGE_SOURCE_JOB_ID,
        "source_prompt": BASELINE_IMAGE_SOURCE_PROMPT_ZH,
        "source_prompt_en": BASELINE_IMAGE_SOURCE_PROMPT_EN,
        "default_prompt": BASELINE_IMAGE_PROMPT_ZH,
        "default_prompt_en": BASELINE_IMAGE_PROMPT_EN,
        "pose_prompt_source": "verified_profile_presets",
        "pose_presets": [dict(preset) for preset in MULTIPOSE_PRESETS],
        "pose_seed_offsets": dict(MULTIPOSE_SEED_OFFSETS),
        "default_poses": list(DEFAULT_POSES),
        "identity_palette_guard": False,
        "identity_palette_policy": None,
        "user_loras": [dict(item) for item in BASELINE_IMAGE_USER_LORAS],
        **BASELINE_IMAGE_GENERATION_SETTINGS,
    },
    LEGACY_EATING_IMAGE_PROFILE_ID: {
        "id": LEGACY_EATING_IMAGE_PROFILE_ID,
        "revision": "eating-v1",
        "label": "20260717 写实小猫进食基准（历史兼容）",
        "source_job_id": BASELINE_IMAGE_SOURCE_JOB_ID,
        "source_prompt": BASELINE_IMAGE_SOURCE_PROMPT_ZH,
        "source_prompt_en": BASELINE_IMAGE_SOURCE_PROMPT_EN,
        "default_prompt": BASELINE_IMAGE_SOURCE_PROMPT_ZH,
        "default_prompt_en": BASELINE_IMAGE_SOURCE_PROMPT_EN,
        "pose_prompt_source": "legacy_global_defaults",
        "pose_presets": [],
        "pose_seed_offsets": {},
        "default_poses": list(LEGACY_DEFAULT_POSES),
        "identity_palette_guard": False,
        "identity_palette_policy": None,
        "user_loras": [dict(item) for item in BASELINE_IMAGE_USER_LORAS],
        **BASELINE_IMAGE_GENERATION_SETTINGS,
    },
}
DEFAULT_IMAGE_PROFILE = os.environ.get(
    "DEFAULT_IMAGE_PROFILE", BASELINE_IMAGE_PROFILE_ID
).strip()
if DEFAULT_IMAGE_PROFILE and DEFAULT_IMAGE_PROFILE not in IMAGE_PROFILES:
    raise ValueError(f"Unknown DEFAULT_IMAGE_PROFILE: {DEFAULT_IMAGE_PROFILE}")

_HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_PROMPT_TRANSLATION_CACHE: dict[str, tuple[str, str]] = {
    BASELINE_IMAGE_PROMPT_ZH: (
        BASELINE_IMAGE_PROMPT_EN,
        "verified_profile_translation",
    ),
    **{
        preset["prompt_zh"]: (
            preset["prompt_en"],
            "verified_profile_translation",
        )
        for preset in MULTIPOSE_PRESETS
    },
}
_PROMPT_TRANSLATION_CACHE_LOCK = threading.RLock()
RESOLUTIONS = {
    "480p": (832, 480),
    "720p": (1280, 704),
    "512x512": (512, 512),
}
OUTPUT_FORMATS = {"mp4", "transparent_gif"}

CHROMA_BACKGROUND_PROMPT = """The entire background must be a perfectly flat, uniform pure #00FF00 chroma-key green field for later background removal. Treat it as a digitally inserted color plate, never as a physical green studio: it emits no light and creates no green or cyan bounce, rim light, reflection, translucency, eye catchlight, or color spill on the character. There must be no floor plane, cast shadow, contact shadow, gradient, texture, reflection, vignette, or lighting variation in the green background. Light the character independently with neutral white light. Keep generous green padding around the complete subject, and do not use green anywhere on the subject except a genuinely green eye color visible in the uploaded identity views."""
CHROMA_NEGATIVE = (
    "white background, gray background, black background, dark background, non-green background, scenery, floor, floor line, cast shadow, "
    "contact shadow, reflection, background gradient, background texture, green subject, green fur, green clothing, "
    "green bounce light, green color spill, green rim light, cyan rim light, chroma reflection on fur"
)

SLEEK_GROOMING_PROMPT = (
    "Preserve the source coat colors, markings, exact natural coat-length category, "
    "visible volume, and source-bounded outer contour. Never amplify its apparent "
    "length or volume, and never shorten, flatten, or suppress genuine long fur, "
    "cheek or neck ruff, chest fur, leg feathering, or plume tail when the uploaded "
    "identity shows those traits. Keep source-matched strand direction and coherent "
    "animation-ready grooming. Remove only generated artifacts such as isolated "
    "rigid clumps, a new dorsal ridge, a mohawk, radial brush-bristle rays, repeated "
    "fan-shaped streaks, detached filament bundles, or a synthetic wispy halo."
)

UPLOADED_IDENTITY_FUR_LOCK = (
    "UPLOADED IDENTITY APPEARANCE AUTHORITY - non-negotiable: the uploaded pet "
    "view or role-labeled view set is the only authority for coat length, coat "
    "volume, fur texture, colors, markings, face, eyes, ears, nose, body shape, "
    "and tail identity. The selected subject profile supplies action timing and "
    "geometry only; never copy that profile animal's identity, palette, eye "
    "color, breed, markings, or coat-length standard into the uploaded pet. "
    "Preserve the uploaded pet's natural short, medium, or long coat category and "
    "its source-bounded silhouette. Groom only enough to keep coherent strands "
    "and clean animation-ready edges without shortening, lengthening, puffing, "
    "or redesigning the coat."
)
UPLOADED_IDENTITY_FUR_NEGATIVE = (
    "borrowed profile animal appearance, Main_case appearance, profile-specific "
    "coat recoloring, profile-specific eye recoloring, profile-specific markings, "
    "profile-specific breed traits, changed natural coat length, shortened source "
    "coat, invented coat length, amplified fur volume, puffed-up fur, raised guard "
    "hairs, bristling, spiky fur, rigid clumps, dorsal ridge, mohawk, jagged back "
    "silhouette, changing back contour, fur crawling, texture crawling"
)

CARTOON_NEGATIVE = (
    "flat 2D drawing, hand-drawn line art, anime, comic, watercolor, low-poly, hard plastic toy, flat fur, "
    "artificially puffed-up fur, raised guard hairs, bristled hair, spiky fur, needle-like fur, "
    "wire-like fur, rigid fur clumps, detached strand bundles, repeated radial fur strokes, "
    "brush-bristle rays, synthetic wispy fur halo, dorsal ridge, mohawk, changed coat length, "
    "changed coat volume, invented mane or ruff or plume, jagged generated fur silhouette, "
    "generic cat, different breed, changed fur pattern, changed marking placement, changed face shape, "
    "changed eye color, changed ear shape, changed muzzle, changed body proportions, changed tail, "
    "invented stripes, invented spots, decorative markings, clothes, clothing, shirt, overalls, suspenders, bow tie, "
    "collar, harness, glasses, eyewear, hat, shoes, jewelry, accessory, costume, misaligned eyes, crossed eyes, "
    "deformed muzzle, white sticker border, white outer stroke, halo, unrequested object, duplicate bowl, extra bowl, "
    "floating mark, stray line, water strand from mouth, drool, falling water droplet, wink, blinking, closed eye, "
    "squint, spiral, swirl, motion symbol, "
    "extreme caricature, extreme chibi distortion, oversized head, oversized eyes, enlarged irises, "
    "oversized ears, disproportionately enlarged inner-ear panels, altered facial proportions"
)

QWEN_NEGATIVE = (
    "different identity, changed face, changed fur, changed clothing, duplicate subject, extra subject, "
    "extra animal, extra person, deformed anatomy, extra limbs, missing limbs, cropped body, leaving frame, "
    "text, watermark, logo, busy background, low quality, blurry"
)
WAN_NEGATIVE = (
    "identity change, face change, clothing change, color change, scene cut, camera movement, zoom, pan, "
    "rotation, duplicate subject, extra subject, extra limbs, missing limbs, deformed anatomy, cropped body, "
    "leaving frame, walking toward camera, moving closer, subject drift, scale change, text, watermark, "
    "violent motion, severe motion blur, flicker, frame jitter"
)
CARTOON_WAN_NEGATIVE = (
    "flat 2D style, style drift, face distortion, warped face, melting face, changing head shape, asymmetric face, "
    "eye size change, eye position change, pupil drift, crossed eyes, lazy eye, extra eye, disappearing eye, "
    "nose drift, muzzle deformation, mouth morphing, wide open mouth, changing expression, clothes appearing, "
    "shirt, overalls, suspenders, bow tie, collar, harness, glasses, eyewear, hat, shoes, accessories appearing, "
    "blink, wink, closed eye, eyelid motion, squint, fur crawling, texture crawling, color flicker, "
    "fur volume expansion, puffed-up fur, raised fur, bristling, standing guard hairs, spiky fur, "
    "needle-like fur, wire-like fur, rigid fur clumps, long isolated strands, dorsal ridge, mohawk, "
    "jagged fur silhouette, changing back contour, "
    "changed fur markings, changed eye color, generic cat"
)

JOB_QUEUE: queue.Queue[str] = queue.Queue(maxsize=MAX_QUEUE_SIZE)
JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.RLock()
STOP_EVENT = threading.Event()
WORKER_THREAD: threading.Thread | None = None
REMBG_SESSION_LOCK = threading.Lock()
REMBG_SESSION: Any | None = None

TERMINAL_JOB_STATUSES = {"completed", "failed", "cancelled"}


class JobCancelled(RuntimeError):
    """Raised when a user requests cancellation of a generation job."""


Image.MAX_IMAGE_PIXELS = 40_000_000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def link(output_node: str, slot: int = 0) -> list[Any]:
    return [output_node, slot]


def wan_model_paths() -> list[Path]:
    paths = [MODEL_ROOT / "text_encoders" / WAN_CLIP, MODEL_ROOT / "vae" / WAN_VAE]
    if WAN_ENGINE == "wan2.2-i2v-a14b-fp8":
        paths.extend(
            [
                MODEL_ROOT / "diffusion_models" / WAN_HIGH_NOISE_MODEL,
                MODEL_ROOT / "diffusion_models" / WAN_LOW_NOISE_MODEL,
            ]
        )
    else:
        paths.append(MODEL_ROOT / "diffusion_models" / WAN_MODEL)
    return paths


def wan_engine_metadata(
    steps: int = WAN_STEPS,
    cfg: float = WAN_CFG,
    shift: float = WAN_SHIFT,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "engine": WAN_ENGINE,
        "frames": FRAME_COUNT,
        "fps": FPS,
        "steps": steps,
        "cfg": cfg,
        "shift": shift,
        "vae": WAN_VAE,
        "text_encoder": WAN_CLIP,
    }
    if WAN_ENGINE == "wan2.2-i2v-a14b-fp8":
        metadata.update(
            {
                "high_noise_model": WAN_HIGH_NOISE_MODEL,
                "low_noise_model": WAN_LOW_NOISE_MODEL,
                "split_step": max(
                    1, min(steps - 1, round(steps * WAN_SPLIT_STEP / WAN_STEPS))
                ),
                "precision": "fp8_scaled",
            }
        )
    else:
        metadata.update({"model": WAN_MODEL, "precision": "fp16"})
    return metadata


def required_model_paths(include_video: bool = True) -> list[Path]:
    paths = [
        MODEL_ROOT / "diffusion_models" / QWEN_MODEL,
        MODEL_ROOT / "text_encoders" / QWEN_CLIP,
        MODEL_ROOT / "vae" / QWEN_VAE,
        MODEL_ROOT / "loras" / QWEN_LORA,
    ]
    if include_video:
        paths.extend(wan_model_paths())
    if QWEN_STYLE_LORA:
        paths.append(MODEL_ROOT / "loras" / QWEN_STYLE_LORA)
    if include_video and FUN_CONTROL_ENABLED:
        paths.extend(
            [
                FUN_CONTROL_PYTHON,
                FUN_CONTROL_SCRIPT,
                FUN_CONTROL_PROJECT_ROOT / "videox_fun" / "__init__.py",
                FUN_CONTROL_MODEL_ROOT / "diffusion_pytorch_model.safetensors",
                FUN_CONTROL_MODEL_ROOT / "models_t5_umt5-xxl-enc-bf16.pth",
                FUN_CONTROL_MODEL_ROOT / "Wan2.2_VAE.pth",
                FUN_CONTROL_RUNNING_VIDEO,
            ]
        )
    return paths


def require_models(include_video: bool = True) -> None:
    missing = [
        str(path)
        for path in required_model_paths(include_video=include_video)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Required local model files are missing: " + ", ".join(missing)
        )


def scail_model_paths() -> list[Path]:
    return [
        MODEL_ROOT / "diffusion_models" / SCAIL_UNET,
        MODEL_ROOT / "loras" / SCAIL_DPO_LORA,
        MODEL_ROOT / "loras" / SCAIL_DISTILL_LORA,
        MODEL_ROOT / "text_encoders" / WAN_CLIP,
        MODEL_ROOT / "vae" / "wan_2.1_vae.safetensors",
        MODEL_ROOT / "clip_vision" / "clip_vision_h.safetensors",
        SCAIL_PYTHON,
        SCAIL_RUNNER,
        SCAIL_TRANSPARENT_RUNNER,
    ]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def subject_profile_sha256(profile: dict[str, Any]) -> str:
    profile_path = Path(str(profile.get("_path") or ""))
    if profile_path.is_file():
        return file_sha256(profile_path)
    serializable = {key: value for key, value in profile.items() if key != "_path"}
    payload = json.dumps(
        serializable,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def json_value_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def fur_surface_thresholds_payload() -> dict[str, Any]:
    return json.loads(json.dumps(asdict(FurSurfaceThresholds())))


def resolve_profile_fur_surface_baseline(
    profile: dict[str, Any],
    action: str,
    action_reference: str,
) -> dict[str, Any] | None:
    """Freeze profile-declared action-baseline provenance and its digest."""

    if profile.get("id") != "sleek_tabby_cat_v1":
        return None
    validated_actions = profile.get("validated_action_baselines")
    if not isinstance(validated_actions, dict):
        raise ValueError(
            "Sleek-tabby fur surface gate requires validated_action_baselines"
        )
    record = validated_actions.get(action)
    if not isinstance(record, dict):
        raise ValueError(
            f"Sleek-tabby fur surface gate lacks the {action} validated baseline"
        )
    baseline_reference = str(record.get("reference") or "").strip()
    if not baseline_reference:
        raise ValueError(
            f"Sleek-tabby {action} validated baseline lacks reference provenance"
        )
    if Path(baseline_reference).resolve() != Path(action_reference).resolve():
        raise ValueError(
            f"Sleek-tabby {action} validated baseline reference does not match "
            "action_references"
        )
    expected_sha256 = str(record.get("reference_sha256") or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError(
            f"Sleek-tabby {action} validated baseline requires reference_sha256"
        )
    baseline_path = Path(baseline_reference)
    actual_sha256 = file_sha256(baseline_path) if baseline_path.is_file() else None
    if actual_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(
            f"Sleek-tabby {action} validated baseline reference digest changed"
        )
    run_id = str(record.get("run_id") or "").strip()
    if not run_id:
        raise ValueError(
            f"Sleek-tabby {action} validated baseline lacks run_id provenance"
        )
    profile_digest = subject_profile_sha256(profile)
    record_digest = json_value_sha256(record)
    provenance = {
        "profile_id": str(profile["id"]),
        "profile_sha256": profile_digest,
        "action": action,
        "validated_action_baseline_run_id": run_id,
        "validated_action_baseline_reference": baseline_reference,
        "validated_action_baseline_reference_sha256": expected_sha256,
        "validated_action_baseline_record_sha256": record_digest,
    }
    return {
        "enabled": True,
        "policy": FUR_SURFACE_GATE_POLICY,
        "gate_version": FUR_SURFACE_GATE_VERSION,
        "threshold_version": FUR_SURFACE_THRESHOLD_VERSION,
        "thresholds": fur_surface_thresholds_payload(),
        "action": action,
        "source": baseline_reference,
        "source_sha256": expected_sha256,
        "path": baseline_reference,
        "sha256": expected_sha256,
        "snapshot_sha256": None,
        "provenance": provenance,
        "provenance_sha256": json_value_sha256(provenance),
    }


def _runtime_path(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Subject profile runtime {field} must be a non-empty path")
    normalized = value.strip()
    path = Path(normalized)
    if not path.is_absolute() and not normalized.startswith("/"):
        path = PROJECT_ROOT / path
    return str(path)


def subject_profile_runtime(profile: dict[str, Any]) -> dict[str, Any]:
    runtime = profile.get("runtime")
    if runtime is None:
        return {
            "schema_version": int(profile.get("schema_version") or 1),
            "source": "legacy_defaults",
            "engine_profile": "scail2-14b-fp8-v1",
            "prompt_profile": "sleek-tabby-scail-v1",
            "default_base_seed": 20260726,
            "steps": SCAIL_STEPS,
            "frames": SCAIL_FRAMES,
            "fps": SCAIL_FPS,
            "actions": {
                action: {
                    "driving_animation": str(
                        SCAIL_ACTION_CONFIG[action]["driving_template"]
                    ),
                    "pose_strength": float(
                        SCAIL_ACTION_CONFIG[action]["pose_strength"]
                    ),
                    "seed_offset": int(SCAIL_ACTION_CONFIG[action]["seed_offset"]),
                }
                for action in SUBJECT_REPLACEMENT_ACTIONS
            },
        }
    if profile.get("schema_version") != 2 or not isinstance(runtime, dict):
        raise ValueError("Subject profile runtime requires schema_version 2")

    engine_profile = runtime.get("engine_profile")
    prompt_profile = runtime.get("prompt_profile")
    if not isinstance(engine_profile, str) or not engine_profile.strip():
        raise ValueError("Subject profile runtime lacks engine_profile")
    if not isinstance(prompt_profile, str) or not prompt_profile.strip():
        raise ValueError("Subject profile runtime lacks prompt_profile")
    try:
        default_base_seed = int(runtime["default_base_seed"])
        steps = int(runtime["steps"])
        frames = int(runtime["frames"])
        fps = float(runtime["fps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Subject profile runtime numeric defaults are invalid"
        ) from exc
    if default_base_seed < 0:
        raise ValueError(
            "Subject profile runtime default_base_seed must be non-negative"
        )
    if not 4 <= steps <= 12:
        raise ValueError("Subject profile runtime steps must be between 4 and 12")
    if frames < 1 or (frames - 1) % 4:
        raise ValueError("Subject profile runtime frames must be 1 + 4*n")
    if fps <= 0:
        raise ValueError("Subject profile runtime fps must be positive")

    raw_actions = runtime.get("actions")
    if not isinstance(raw_actions, dict):
        raise ValueError("Subject profile runtime lacks actions")
    resolved_actions: dict[str, dict[str, Any]] = {}
    for action in SUBJECT_REPLACEMENT_ACTIONS:
        raw_action = raw_actions.get(action)
        if not isinstance(raw_action, dict):
            raise ValueError(f"Subject profile runtime lacks the {action} action")
        try:
            action_pose_strength = float(raw_action["pose_strength"])
            action_seed_offset = int(raw_action["seed_offset"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Subject profile runtime {action} parameters are invalid"
            ) from exc
        if not 0.0 <= action_pose_strength <= 2.0:
            raise ValueError(
                f"Subject profile runtime {action} pose_strength is out of range"
            )
        driving_animation_sha256 = (
            str(raw_action.get("driving_animation_sha256") or "").strip().lower()
        )
        if driving_animation_sha256 and not re.fullmatch(
            r"[0-9a-f]{64}",
            driving_animation_sha256,
        ):
            raise ValueError(
                f"Subject profile runtime {action} driving_animation_sha256 "
                "must be a 64-character lowercase hexadecimal digest"
            )
        resolved_actions[action] = {
            "driving_animation": _runtime_path(
                raw_action.get("driving_animation"),
                field=f"actions.{action}.driving_animation",
            ),
            "driving_animation_sha256": driving_animation_sha256 or None,
            "pose_strength": action_pose_strength,
            "seed_offset": action_seed_offset,
        }
    return {
        "schema_version": 2,
        "source": "profile.runtime",
        "engine_profile": engine_profile.strip(),
        "prompt_profile": prompt_profile.strip(),
        "default_base_seed": default_base_seed,
        "steps": steps,
        "frames": frames,
        "fps": fps,
        "actions": resolved_actions,
    }


def profile_standard_direct_config(
    profile: dict[str, Any],
    runtime: dict[str, Any],
    actions: list[str],
) -> dict[str, Any] | None:
    """Freeze the optional no-redraw Main_case profile contract.

    This is deliberately profile data, not an implicit filename convention.  A
    profile without the explicit opt-in continues through the regular uploaded
    identity/SCAIL path.
    """

    raw_direct = profile.get("standard_direct")
    if not isinstance(raw_direct, dict) or raw_direct.get("enabled") is not True:
        return None
    if profile.get("id") != "sleek_tabby_cat_v1":
        raise ValueError("standard_direct is only approved for sleek_tabby_cat_v1")
    if raw_direct.get("policy") != PROFILE_STANDARD_DIRECT_POLICY:
        raise ValueError("Subject profile standard_direct policy is unsupported")
    if raw_direct.get("role") != PROFILE_STANDARD_DIRECT_ROLE:
        raise ValueError("Subject profile standard_direct role is unsupported")

    driving_standards = profile.get("driving_standards")
    if not isinstance(driving_standards, dict):
        raise ValueError("Subject profile standard_direct lacks driving_standards")
    revision = str(driving_standards.get("revision") or "").strip()
    if not revision or revision != str(raw_direct.get("revision") or "").strip():
        raise ValueError("Subject profile standard_direct revision mismatch")
    standard_actions = driving_standards.get("actions")
    if not isinstance(standard_actions, dict):
        raise ValueError("Subject profile standard_direct lacks action standards")
    reference_assets = profile.get("reference_assets")
    if (
        not isinstance(reference_assets, dict)
        or str(reference_assets.get("revision") or "").strip() != revision
        or not isinstance(reference_assets.get("actions"), dict)
    ):
        raise ValueError(
            "Subject profile standard_direct lacks matching reference_assets"
        )

    resolved_actions: dict[str, Any] = {}
    for action in actions:
        standard = standard_actions.get(action)
        runtime_action = runtime.get("actions", {}).get(action)
        if not isinstance(standard, dict) or not isinstance(runtime_action, dict):
            raise ValueError(f"Subject profile standard_direct lacks {action}")
        source_path = _runtime_path(
            standard.get("source"),
            field=f"driving_standards.actions.{action}.source",
        )
        runtime_path = _runtime_path(
            standard.get("runtime"),
            field=f"driving_standards.actions.{action}.runtime",
        )
        source_sha256 = str(standard.get("source_sha256") or "").strip().lower()
        runtime_sha256 = str(standard.get("runtime_sha256") or "").strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", source_sha256):
            raise ValueError(
                f"Subject profile standard_direct {action} source SHA is invalid"
            )
        if not re.fullmatch(r"[0-9a-f]{64}", runtime_sha256):
            raise ValueError(
                f"Subject profile standard_direct {action} runtime SHA is invalid"
            )
        if runtime_path != str(runtime_action.get("driving_animation") or ""):
            raise ValueError(
                f"Subject profile standard_direct {action} runtime path mismatch"
            )
        if (
            runtime_sha256
            != str(runtime_action.get("driving_animation_sha256") or "").strip().lower()
        ):
            raise ValueError(
                f"Subject profile standard_direct {action} runtime SHA mismatch"
            )
        reference_asset = reference_assets["actions"].get(action)
        if not isinstance(reference_asset, dict):
            raise ValueError(
                f"Subject profile standard_direct lacks {action} reference assets"
            )
        try:
            selected_frame_index = int(reference_asset["selected_frame_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Subject profile standard_direct {action} selected frame is invalid"
            ) from exc
        if selected_frame_index < 0 or selected_frame_index >= int(runtime["frames"]):
            raise ValueError(
                f"Subject profile standard_direct {action} selected frame is out of range"
            )
        selected_assets: dict[str, Any] = {
            "selected_frame_index": selected_frame_index,
            "selected_frame_timestamp_seconds": reference_asset.get(
                "selected_frame_timestamp_seconds"
            ),
            "identity_mask_policy": reference_asset.get("identity_mask_policy"),
        }
        for asset_kind in ("preview", "cutout", "identity_mask"):
            asset_path = _runtime_path(
                reference_asset.get(asset_kind),
                field=f"reference_assets.actions.{action}.{asset_kind}",
            )
            asset_sha = (
                str(reference_asset.get(f"{asset_kind}_sha256") or "").strip().lower()
            )
            if not re.fullmatch(r"[0-9a-f]{64}", asset_sha):
                raise ValueError(
                    f"Subject profile standard_direct {action} {asset_kind} SHA is invalid"
                )
            selected_assets[asset_kind] = asset_path
            selected_assets[f"{asset_kind}_sha256"] = asset_sha
        resolved_actions[action] = {
            "source": source_path,
            "source_sha256": source_sha256,
            "runtime": runtime_path,
            "runtime_sha256": runtime_sha256,
            "validated_run_id": standard.get("validated_run_id"),
            "selected_reference": selected_assets,
        }
    return {
        "enabled": True,
        "policy": PROFILE_STANDARD_DIRECT_POLICY,
        "role": PROFILE_STANDARD_DIRECT_ROLE,
        "revision": revision,
        "deterministic": True,
        "gpu_required": False,
        "gpu_bypass_reason": PROFILE_STANDARD_DIRECT_GPU_REASON,
        "actions": resolved_actions,
    }


def verify_profile_standard_direct_assets(
    standard_direct: dict[str, Any],
    actions: list[str],
) -> None:
    """Fail closed if either the original or normalized frozen standard changed."""

    for action in actions:
        action_config = standard_direct.get("actions", {}).get(action)
        if not isinstance(action_config, dict):
            raise FileNotFoundError(
                f"Frozen standard_direct {action} config is missing"
            )
        for kind in ("source", "runtime"):
            path = Path(str(action_config.get(kind) or ""))
            expected = str(action_config.get(f"{kind}_sha256") or "").strip()
            if not path.is_file():
                raise FileNotFoundError(
                    f"Frozen standard_direct {action} {kind} is missing: {path}"
                )
            if not expected or file_sha256(path) != expected:
                raise RuntimeError(
                    f"Frozen standard_direct {action} {kind} digest changed."
                )
        selected = action_config.get("selected_reference")
        if not isinstance(selected, dict):
            raise FileNotFoundError(
                f"Frozen standard_direct {action} selected reference is missing"
            )
        for kind in ("preview", "cutout", "identity_mask"):
            path = Path(str(selected.get(kind) or ""))
            expected = str(selected.get(f"{kind}_sha256") or "").strip()
            if not path.is_file():
                raise FileNotFoundError(
                    f"Frozen standard_direct {action} {kind} is missing: {path}"
                )
            if not expected or file_sha256(path) != expected:
                raise RuntimeError(
                    f"Frozen standard_direct {action} {kind} digest changed."
                )


def uses_full_driver_eating_pose(policy: object) -> bool:
    """Return true only for the eating runner policy that needs no legacy GIF."""

    return policy == FULL_DRIVER_EATING_POSE_POLICY


def require_expected_scail_pose_policy(
    manifest: dict[str, Any],
    action: str,
    expected_policy: str | None,
) -> None:
    """Fail before GPU submission if a prepared driver policy drifted."""

    if not expected_policy:
        return
    pose_metadata = manifest.get("pose_metadata")
    actual_policy = (
        str(pose_metadata.get("policy") or "")
        if isinstance(pose_metadata, dict)
        else ""
    )
    if actual_policy != expected_policy:
        raise RuntimeError(
            f"SCAIL {action} pose policy changed before GPU submission: "
            f"expected {expected_policy}, got {actual_policy or 'missing'}"
        )


def require_uploaded_fur_surface_gate(
    action_config: dict[str, Any],
    action: str,
) -> dict[str, Any]:
    """Return the frozen per-action surface gate or fail the upload contract."""

    gate = action_config.get("fur_surface_gate")
    if (
        not isinstance(gate, dict)
        or gate.get("enabled") is not True
        or gate.get("policy") != FUR_SURFACE_GATE_POLICY
        or gate.get("gate_version") != FUR_SURFACE_GATE_VERSION
        or gate.get("threshold_version") != FUR_SURFACE_THRESHOLD_VERSION
        or gate.get("fail_closed") is not True
    ):
        raise RuntimeError(
            f"Uploaded {action} reference lacks its enabled, version-locked "
            "fur surface gate."
        )
    baseline = gate.get("baseline")
    if (
        not isinstance(baseline, dict)
        or baseline.get("enabled") is not True
        or baseline.get("policy") != FUR_SURFACE_GATE_POLICY
        or baseline.get("gate_version") != FUR_SURFACE_GATE_VERSION
        or baseline.get("threshold_version") != FUR_SURFACE_THRESHOLD_VERSION
        or baseline.get("action") != action
    ):
        raise RuntimeError(
            f"Uploaded {action} reference lacks its version-locked Profile "
            "fur surface baseline."
        )
    baseline_path = Path(str(baseline.get("path") or baseline.get("source") or ""))
    baseline_sha256 = (
        str(baseline.get("sha256") or baseline.get("source_sha256") or "")
        .strip()
        .lower()
    )
    if (
        not re.fullmatch(r"[0-9a-f]{64}", baseline_sha256)
        or not baseline_path.is_file()
        or file_sha256(baseline_path) != baseline_sha256
    ):
        raise RuntimeError(
            f"Uploaded {action} Profile fur surface baseline is missing or "
            "its digest changed."
        )
    return gate


def configure_uploaded_subject_validation(
    effective_config: dict[str, Any],
    actions: list[str],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Freeze validation rules for uploaded Qwen references and SCAIL motion.

    Every upload must first produce a normalized Qwen full-body canonical and
    normalized transparent action keyframes.  A digest-locked neutral-white
    anatomy-maquette runtime supplies only action geometry to Qwen and motion
    timing to SCAIL; it never supplies pet identity, coat, color, markings,
    face, eyes, material, or style.
    """

    if effective_config.get("profile_id") != "sleek_tabby_cat_v1":
        raise RuntimeError(
            "Uploaded full-driver validation requires sleek_tabby_cat_v1."
        )
    standard_direct = effective_config.get("standard_direct")
    if (
        not isinstance(standard_direct, dict)
        or standard_direct.get("enabled") is not True
        or standard_direct.get("policy") != PROFILE_STANDARD_DIRECT_POLICY
    ):
        raise RuntimeError(
            "Uploaded full-driver validation requires the frozen Main_case standard."
        )
    verify_profile_standard_direct_assets(standard_direct, actions)

    frozen = json.loads(json.dumps(effective_config))
    expected_pose_policies: dict[str, str] = {}
    expected_surface_gate_actions: list[str] = []
    for action in actions:
        action_config = frozen["actions"][action]
        direct_action = standard_direct.get("actions", {}).get(action)
        if not isinstance(direct_action, dict):
            raise RuntimeError(
                f"Uploaded full-driver validation lacks {action} standard metadata."
            )
        runtime_path = str(action_config.get("driving_animation") or "")
        runtime_sha256 = str(
            action_config.get("driving_animation_sha256") or ""
        ).strip()
        if (
            runtime_path != str(direct_action.get("runtime") or "")
            or not runtime_sha256
            or runtime_sha256 != str(direct_action.get("runtime_sha256") or "")
        ):
            raise RuntimeError(
                f"Uploaded {action} driver is not the frozen Main_case runtime."
            )
        uploaded_driver = action_config.get("uploaded_identity_driving")
        if not isinstance(uploaded_driver, dict):
            raise RuntimeError(
                f"Uploaded {action} lacks its white-maquette driving contract."
            )
        uploaded_revision = str(uploaded_driver.get("revision") or "").strip()
        uploaded_role = str(uploaded_driver.get("role") or "").strip()
        uploaded_runtime_path = Path(
            str(uploaded_driver.get("runtime") or "")
        )
        uploaded_runtime_sha256 = str(
            uploaded_driver.get("runtime_sha256") or ""
        ).strip().lower()
        uploaded_source_path = Path(
            str(uploaded_driver.get("source") or "")
        )
        uploaded_source_sha256 = str(
            uploaded_driver.get("source_sha256") or ""
        ).strip().lower()
        geometry_reference_path = Path(
            str(uploaded_driver.get("geometry_reference") or "")
        )
        geometry_reference_sha256 = str(
            uploaded_driver.get("geometry_reference_sha256") or ""
        ).strip().lower()
        for field_name, path, expected_sha256 in (
            ("source", uploaded_source_path, uploaded_source_sha256),
            ("runtime", uploaded_runtime_path, uploaded_runtime_sha256),
            (
                "geometry_reference",
                geometry_reference_path,
                geometry_reference_sha256,
            ),
        ):
            if (
                not uploaded_revision
                or not uploaded_role
                or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
                or not path.is_file()
                or file_sha256(path) != expected_sha256
            ):
                raise RuntimeError(
                    "Uploaded white-maquette "
                    f"{action} {field_name} is missing or its digest changed."
                )
        with Image.open(geometry_reference_path) as loaded:
            if loaded.size != (512, 512) or "A" not in loaded.getbands():
                raise RuntimeError(
                    f"Uploaded white-maquette {action} geometry reference must "
                    "be a 512x512 RGBA image."
                )
            alpha_minimum, alpha_maximum = loaded.getchannel("A").getextrema()
            if alpha_minimum == 255 or alpha_maximum == 0:
                raise RuntimeError(
                    f"Uploaded white-maquette {action} geometry reference lacks "
                    "a usable alpha matte."
                )
        action_config["profile_driving_animation"] = runtime_path
        action_config["profile_driving_animation_sha256"] = runtime_sha256
        action_config["driving_animation"] = str(uploaded_runtime_path)
        action_config["driving_animation_sha256"] = uploaded_runtime_sha256
        action_config["driving_reference_kind"] = "uploaded_white_maquette"
        action_config["driving_reference_revision"] = uploaded_revision
        action_config["driving_reference_role"] = uploaded_role
        action_config["geometry_reference"] = str(geometry_reference_path)
        action_config["geometry_reference_sha256"] = (
            geometry_reference_sha256
        )
        action_config["geometry_reference_kind"] = (
            "uploaded_white_maquette_frame040_rgba"
        )
        if GENERATION_VISUAL_AUDITS_BLOCKING:
            require_uploaded_fur_surface_gate(action_config, action)
            action_config["fur_surface_gate"]["candidate_comparison_mode"] = (
                UPLOADED_IDENTITY_SURFACE_COMPARISON_MODE
            )
            expected_surface_gate_actions.append(action)

        if action == "eating":
            expected_pose_policies[action] = FULL_DRIVER_EATING_POSE_POLICY
            fur_silhouette_gate = action_config.get("fur_silhouette_gate")
            if isinstance(fur_silhouette_gate, dict):
                fur_silhouette_gate["required_pose_policy"] = (
                    FULL_DRIVER_EATING_POSE_POLICY
                )

    frozen["prompt_profile"] = "uploaded-white-maquette-v1"
    frozen["upload_validation"] = {
        "policy": QWEN_FULL_BODY_FIRST_FRAME_PIPELINE,
        "scope": "qwen_full_body_and_action_keyframes_then_scail",
        "qwen_full_body_required": True,
        "qwen_action_keyframes_required": True,
        "normalized_transparent_references_required": True,
        "identity_upload_view_counts": [1, 3],
        "three_view_roles": list(UPLOADED_IDENTITY_VIEW_ROLES),
        "three_view_policy": UPLOADED_IDENTITY_VIEW_POLICY,
        "three_view_qwen_node_inputs": ["image1", "image2", "image3"],
        "three_view_canonical_supported": True,
        "three_view_action_stage_raw_views_used": False,
        "uploaded_identity_appearance_authority": True,
        "subject_profile_appearance_role": "action_geometry_only",
        "driving_reference_kind": "uploaded_white_maquette",
        "driving_reference_revision": frozen.get(
            "uploaded_identity_driving_revision"
        ),
        "driving_reference_role": frozen.get("uploaded_identity_driving_role"),
        "driving_reference_appearance_authority": False,
        "driving_reference_geometry_and_timing_only": True,
        "white_maquette_face_eye_coat_and_material_ignored": True,
        "canonical_palette_input": "final_alpha_cutout",
        "canonical_upper_body_palette_reference_policy": (
            CANONICAL_UPPER_BODY_PALETTE_REFERENCE_POLICY
        ),
        "canonical_raw_palette_retained_for_audit": True,
        "canonical_chroma_despill": (
            "alpha_boundary_plus_source_palette_guarded_interior_v1"
        ),
        "canonical_chroma_spill_policy": (UPLOADED_IDENTITY_CHROMA_SPILL_POLICY),
        "canonical_chroma_spill_preserves_paired_eye_accents": True,
        "canonical_interior_hole_gate": True,
        "generation_audit_policy": GENERATION_AUDIT_POLICY,
        "visual_content_audit_mode": (
            "blocking" if GENERATION_VISUAL_AUDITS_BLOCKING else "advisory"
        ),
        "runtime_integrity_checks_required": True,
        "visual_content_audits_blocking": GENERATION_VISUAL_AUDITS_BLOCKING,
        "surface_texture_gate_required_before_scail": (
            GENERATION_VISUAL_AUDITS_BLOCKING
        ),
        "profile_relative_surface_gate_used": False,
        "uploaded_identity_absolute_surface_gate_used": True,
        "surface_candidate_comparison_mode": (
            UPLOADED_IDENTITY_SURFACE_COMPARISON_MODE
        ),
        "surface_texture_gate": (
            "required_before_scail"
            if GENERATION_VISUAL_AUDITS_BLOCKING
            else "skipped_advisory"
        ),
        "absolute_animation_contour_gate": False,
        "actions": list(actions),
        "expected_surface_gate_actions": expected_surface_gate_actions,
        "expected_pose_policies": dict(expected_pose_policies),
    }
    return frozen, expected_pose_policies


def resolve_effective_subject_config(
    profile: dict[str, Any],
    actions: list[str],
    *,
    seed: int | None,
    scail_steps: int | None,
    pose_strength: float | None,
) -> dict[str, Any]:
    runtime = subject_profile_runtime(profile)
    base_seed = int(seed if seed is not None else runtime["default_base_seed"])
    if base_seed < 0:
        raise ValueError("seed must be non-negative")
    steps = int(scail_steps if scail_steps is not None else runtime["steps"])
    if not 4 <= steps <= 12:
        raise ValueError("scail_steps must be between 4 and 12")
    if pose_strength is not None and not 0.0 <= float(pose_strength) <= 2.0:
        raise ValueError("pose_strength must be between 0.0 and 2.0")

    action_references = profile["action_references"]
    raw_fur_constraints = profile.get("fur_constraints")
    fur_constraints = (
        json.loads(json.dumps(raw_fur_constraints))
        if isinstance(raw_fur_constraints, dict)
        else {}
    )
    raw_validated_baseline = profile.get("validated_replacement_baseline")
    validated_baseline = (
        json.loads(json.dumps(raw_validated_baseline))
        if isinstance(raw_validated_baseline, dict)
        else {}
    )
    raw_uploaded_identity_driving = profile.get("uploaded_identity_driving")
    uploaded_identity_driving_actions: dict[str, Any] = {}
    uploaded_identity_driving_revision: str | None = None
    uploaded_identity_driving_role: str | None = None
    if isinstance(raw_uploaded_identity_driving, dict):
        uploaded_identity_driving_revision = str(
            raw_uploaded_identity_driving.get("revision") or ""
        ).strip()
        uploaded_identity_driving_role = str(
            raw_uploaded_identity_driving.get("role") or ""
        ).strip()
        raw_uploaded_actions = raw_uploaded_identity_driving.get("actions")
        if (
            not uploaded_identity_driving_revision
            or not uploaded_identity_driving_role
            or not isinstance(raw_uploaded_actions, dict)
        ):
            raise ValueError(
                "Subject profile uploaded_identity_driving metadata is incomplete"
            )
        uploaded_identity_driving_actions = raw_uploaded_actions
    elif profile.get("id") == "sleek_tabby_cat_v1":
        raise ValueError(
            "Sleek-tabby uploads require digest-bound white-maquette driving assets"
        )
    resolved_actions: dict[str, dict[str, Any]] = {}
    for action in actions:
        action_runtime = runtime["actions"][action]
        reference = str(action_references[action])
        reference_path = Path(reference)
        driving_animation = str(action_runtime["driving_animation"])
        driving_animation_path = Path(driving_animation)
        expected_driving_sha256 = str(
            action_runtime.get("driving_animation_sha256") or ""
        ).strip()
        actual_driving_sha256 = (
            file_sha256(driving_animation_path)
            if driving_animation_path.is_file()
            else None
        )
        if (
            expected_driving_sha256
            and actual_driving_sha256 is not None
            and actual_driving_sha256 != expected_driving_sha256
        ):
            raise ValueError(
                f"Subject profile {action} driving animation digest changed"
            )
        effective_pose_strength = float(
            pose_strength
            if pose_strength is not None
            else action_runtime["pose_strength"]
        )
        if (
            action == "eating"
            and effective_pose_strength > SCAIL_EATING_POSE_STRENGTH_MAX
        ):
            raise ValueError(
                "eating pose_strength must not exceed "
                f"{SCAIL_EATING_POSE_STRENGTH_MAX:.2f}; higher values import "
                "the driving animal's fur silhouette"
            )
        resolved_actions[action] = {
            "reference": reference,
            "reference_source": reference,
            "reference_sha256": (
                file_sha256(reference_path) if reference_path.is_file() else None
            ),
            "driving_animation": driving_animation,
            "driving_animation_sha256": (
                expected_driving_sha256 or actual_driving_sha256
            ),
            "pose_strength": effective_pose_strength,
            "seed_offset": int(action_runtime["seed_offset"]),
            "effective_seed": base_seed + int(action_runtime["seed_offset"]),
        }
        uploaded_action = uploaded_identity_driving_actions.get(action)
        if uploaded_identity_driving_actions and not isinstance(uploaded_action, dict):
            raise ValueError(
                f"Subject profile uploaded_identity_driving lacks {action}"
            )
        if isinstance(uploaded_action, dict):
            uploaded_source = _runtime_path(
                uploaded_action.get("source"),
                field=f"uploaded_identity_driving.actions.{action}.source",
            )
            uploaded_runtime = _runtime_path(
                uploaded_action.get("runtime"),
                field=f"uploaded_identity_driving.actions.{action}.runtime",
            )
            uploaded_geometry_reference = _runtime_path(
                uploaded_action.get("geometry_reference"),
                field=(
                    "uploaded_identity_driving.actions."
                    f"{action}.geometry_reference"
                ),
            )
            uploaded_source_sha256 = str(
                uploaded_action.get("source_sha256") or ""
            ).strip().lower()
            uploaded_runtime_sha256 = str(
                uploaded_action.get("runtime_sha256") or ""
            ).strip().lower()
            uploaded_geometry_reference_sha256 = str(
                uploaded_action.get("geometry_reference_sha256") or ""
            ).strip().lower()
            for field_name, path_text, expected_sha256 in (
                ("source", uploaded_source, uploaded_source_sha256),
                ("runtime", uploaded_runtime, uploaded_runtime_sha256),
                (
                    "geometry_reference",
                    uploaded_geometry_reference,
                    uploaded_geometry_reference_sha256,
                ),
            ):
                if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
                    raise ValueError(
                        "Subject profile uploaded white-maquette "
                        f"{action} {field_name} requires a SHA-256 digest"
                    )
                path = Path(path_text)
                if path.is_file() and file_sha256(path) != expected_sha256:
                    raise ValueError(
                        "Subject profile uploaded white-maquette "
                        f"{action} {field_name} digest changed"
                    )
            resolved_actions[action]["uploaded_identity_driving"] = {
                "revision": uploaded_identity_driving_revision,
                "role": uploaded_identity_driving_role,
                "source": uploaded_source,
                "source_sha256": uploaded_source_sha256,
                "runtime": uploaded_runtime,
                "runtime_sha256": uploaded_runtime_sha256,
                "geometry_reference": uploaded_geometry_reference,
                "geometry_reference_sha256": (
                    uploaded_geometry_reference_sha256
                ),
                "selected_frame_index": int(
                    uploaded_action.get("selected_frame_index", 40)
                ),
                "selected_frame_timestamp_seconds": float(
                    uploaded_action.get("selected_frame_timestamp_seconds", 2.5)
                ),
                "action_contract": str(
                    uploaded_action.get("action_contract") or ""
                ).strip(),
            }
        fur_surface_baseline = resolve_profile_fur_surface_baseline(
            profile,
            action,
            reference,
        )
        if fur_surface_baseline is not None:
            resolved_actions[action]["fur_surface_gate"] = {
                "enabled": True,
                "policy": FUR_SURFACE_GATE_POLICY,
                "gate_version": FUR_SURFACE_GATE_VERSION,
                "threshold_version": FUR_SURFACE_THRESHOLD_VERSION,
                "scope": "main_case_guided_qwen_action_candidates",
                "reliable_inputs": ["alpha_cutout", "rembg_alpha_cutout"],
                "fail_closed": True,
                "max_consecutive_rejections": (FUR_SURFACE_GATE_MAX_REJECTIONS),
                "retry_policy": FUR_SURFACE_GATE_RETRY_POLICY,
                "preserves_configured_eating_bowl_recovery": True,
                "configured_eating_bowl_recovery_attempts": (
                    QWEN_BOWL_POSE_EXTRA_ATTEMPTS
                ),
                "baseline": fur_surface_baseline,
            }
        if (
            action == "eating"
            and profile.get("id") == "sleek_tabby_cat_v1"
            and fur_constraints
        ):
            resolved_actions[action]["fur_silhouette_gate"] = {
                "enabled": True,
                "reference": reference,
                "reference_source": reference,
                "reference_sha256": (
                    file_sha256(reference_path) if reference_path.is_file() else None
                ),
                "policy": "profile_relative_eating_contour_v1",
            }
            baseline_path_text = str(
                validated_baseline.get("transparent_gif") or ""
            ).strip()
            if baseline_path_text:
                baseline_path = Path(baseline_path_text)
                resolved_actions[action]["fur_contour_baseline"] = {
                    "path": baseline_path_text,
                    "source": baseline_path_text,
                    "sha256": (
                        file_sha256(baseline_path) if baseline_path.is_file() else None
                    ),
                    "run_id": validated_baseline.get("run_id"),
                    "policy": "validated_profile_eating_video_v2",
                }
    standard_direct = profile_standard_direct_config(profile, runtime, actions)
    return {
        "schema_version": 1,
        "profile_schema_version": runtime["schema_version"],
        "profile_id": profile["id"],
        "profile_path": str(profile.get("_path") or ""),
        "profile_sha256": subject_profile_sha256(profile),
        "profile_fur_constraints": fur_constraints,
        "runtime_source": runtime["source"],
        "engine_profile": runtime["engine_profile"],
        "prompt_profile": runtime["prompt_profile"],
        "uploaded_identity_driving_revision": uploaded_identity_driving_revision,
        "uploaded_identity_driving_role": uploaded_identity_driving_role,
        "seed_mode": "base_plus_action_offset",
        "base_seed": base_seed,
        "model": SCAIL_UNET,
        "frames": int(runtime["frames"]),
        "fps": float(runtime["fps"]),
        "steps": steps,
        "actions": resolved_actions,
        "standard_direct": standard_direct,
    }


def snapshot_effective_subject_references(
    effective_config: dict[str, Any],
    job_dir: Path,
    actions: list[str],
) -> dict[str, Any]:
    snapshot = json.loads(json.dumps(effective_config))
    reference_dir = job_dir / "profile_inputs"
    reference_dir.mkdir(parents=True, exist_ok=True)
    for action in actions:
        action_config = snapshot["actions"][action]
        source = Path(action_config["reference_source"])
        if not source.is_file():
            raise FileNotFoundError(
                f"Subject profile {action} reference is missing: {source}"
            )
        suffix = source.suffix if source.suffix else ".png"
        target = reference_dir / f"{action}_reference{suffix}"
        shutil.copy2(source, target)
        digest = file_sha256(target)
        source_digest = file_sha256(source)
        if digest != source_digest:
            raise OSError(f"Subject profile {action} reference snapshot is corrupt")
        action_config["reference"] = str(target)
        action_config["reference_sha256"] = digest
        fur_surface_gate = action_config.get("fur_surface_gate")
        if isinstance(fur_surface_gate, dict) and fur_surface_gate.get("enabled"):
            baseline = fur_surface_gate.get("baseline")
            if not isinstance(baseline, dict):
                raise ValueError(
                    f"Subject profile {action} fur surface baseline is missing"
                )
            baseline_source = Path(str(baseline.get("source") or ""))
            expected_source_sha256 = str(baseline.get("source_sha256") or "").strip()
            if baseline_source.resolve() != source.resolve():
                raise ValueError(
                    f"Subject profile {action} fur surface baseline source "
                    "does not match its action reference"
                )
            if not expected_source_sha256 or source_digest != expected_source_sha256:
                raise ValueError(
                    f"Subject profile {action} fur surface baseline digest changed"
                )
            provenance = baseline.get("provenance")
            if (
                not isinstance(provenance, dict)
                or provenance.get("profile_id") != snapshot.get("profile_id")
                or provenance.get("profile_sha256") != snapshot.get("profile_sha256")
                or provenance.get("action") != action
                or baseline.get("provenance_sha256") != json_value_sha256(provenance)
            ):
                raise ValueError(
                    f"Subject profile {action} fur surface baseline provenance changed"
                )
            baseline["path"] = str(target)
            baseline["sha256"] = digest
            baseline["snapshot_sha256"] = digest
            try:
                baseline_contract = evaluate_fur_surface(
                    target,
                    target,
                    baseline_is_validated=True,
                )
            except Exception as exc:  # noqa: BLE001 - profile snapshot must fail closed
                raise ValueError(
                    f"Subject profile {action} fur surface baseline could not be audited"
                ) from exc
            if baseline_contract.get("acceptable") is not True:
                raise ValueError(
                    f"Subject profile {action} fur surface baseline fails its "
                    "absolute contract: "
                    + ", ".join(
                        str(issue)
                        for issue in baseline_contract.get(
                            "baseline_issues",
                            baseline_contract.get("issues", []),
                        )
                    )
                )
            baseline["absolute_contract"] = {
                "acceptable": True,
                "verdict": baseline_contract.get("verdict"),
                "issues": [],
                "input": baseline_contract["candidate"]["input"],
                "metrics": baseline_contract["candidate"]["metrics"],
                "policy": baseline_contract["policy"],
            }
        fur_gate = action_config.get("fur_silhouette_gate")
        if isinstance(fur_gate, dict) and fur_gate.get("enabled"):
            fur_gate["reference"] = str(target)
            fur_gate["reference_sha256"] = digest
        fur_baseline = action_config.get("fur_contour_baseline")
        if isinstance(fur_baseline, dict):
            baseline_source = Path(str(fur_baseline.get("source") or ""))
            if not baseline_source.is_file():
                raise FileNotFoundError(
                    f"Subject profile eating fur baseline is missing: {baseline_source}"
                )
            baseline_target = reference_dir / f"{action}_fur_baseline.gif"
            shutil.copy2(baseline_source, baseline_target)
            baseline_digest = file_sha256(baseline_target)
            source_baseline_digest = file_sha256(baseline_source)
            if baseline_digest != source_baseline_digest:
                raise OSError("Subject profile eating fur baseline snapshot is corrupt")
            fur_baseline["path"] = str(baseline_target)
            fur_baseline["sha256"] = baseline_digest
    return snapshot


def load_subject_profile(profile_id: str | None) -> dict[str, Any]:
    normalized = (profile_id or DEFAULT_SUBJECT_PROFILE).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", normalized):
        raise ValueError("subject_profile contains unsupported characters")
    path = SUBJECT_PROFILE_DIR / f"{normalized}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Subject profile does not exist: {normalized}")
    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Subject profile is invalid: {normalized}") from exc
    if not isinstance(profile, dict) or profile.get("id") != normalized:
        raise ValueError(f"Subject profile id mismatch: {normalized}")
    action_references = profile.get("action_references")
    if not isinstance(action_references, dict):
        raise ValueError(f"Subject profile lacks action_references: {normalized}")
    for action in SUBJECT_REPLACEMENT_ACTIONS:
        reference = action_references.get(action)
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError(
                f"Subject profile lacks the {action} reference: {normalized}"
            )
    profile["_path"] = str(path)
    subject_profile_runtime(profile)
    return profile


def subject_profile_fur_guard(profile: dict[str, Any]) -> tuple[str, str]:
    """Return upload-safe fur rules without leaking the action profile identity."""

    # ``fur_constraints`` describes the frozen profile animal and remains valid for
    # deterministic profile rendering.  Uploaded replacement jobs must never inherit
    # those identity-bearing colors, markings, eyes, breed, or coat-length words.
    # Profiles may opt into a separate upload-only extension whose contract is
    # explicitly appearance-neutral.
    constraints = profile.get("upload_fur_constraints")
    positive_extension = ""
    negative_extension = ""
    if isinstance(constraints, dict):
        positive_extension = str(constraints.get("positive") or "").strip()
        negative_extension = str(constraints.get("negative") or "").strip()
    positive = " ".join(
        part for part in (UPLOADED_IDENTITY_FUR_LOCK, positive_extension) if part
    )
    negative = ", ".join(
        part for part in (UPLOADED_IDENTITY_FUR_NEGATIVE, negative_extension) if part
    )
    return positive, negative


def subject_profile_public(profile: dict[str, Any]) -> dict[str, Any]:
    action_references = profile.get("action_references", {})
    driving_standards = profile.get("driving_standards")
    raw_standard_direct = profile.get("standard_direct")
    public_driving_standards: dict[str, Any] | None = None
    reference_assets = profile.get("reference_assets")
    public_reference_assets: dict[str, Any] | None = None
    if isinstance(reference_assets, dict):
        raw_reference_actions = reference_assets.get("actions")
        reference_actions: dict[str, Any] = {}
        if isinstance(raw_reference_actions, dict):
            for action in SUBJECT_REPLACEMENT_ACTIONS:
                record = raw_reference_actions.get(action)
                if not isinstance(record, dict):
                    continue
                reference_actions[action] = {
                    "selected_frame_index": record.get("selected_frame_index"),
                    "selected_frame_timestamp_seconds": record.get(
                        "selected_frame_timestamp_seconds"
                    ),
                    "selection_reason": record.get("selection_reason"),
                    "preview_filename": Path(str(record.get("preview") or "")).name,
                    "preview_sha256": record.get("preview_sha256"),
                    "cutout_filename": Path(str(record.get("cutout") or "")).name,
                    "cutout_sha256": record.get("cutout_sha256"),
                    "identity_mask_filename": Path(
                        str(record.get("identity_mask") or "")
                    ).name,
                    "identity_mask_sha256": record.get("identity_mask_sha256"),
                    "identity_mask_policy": record.get("identity_mask_policy"),
                    "validation_run_id": record.get("validation_run_id"),
                }
        public_reference_assets = {
            "revision": reference_assets.get("revision"),
            "actions": reference_actions,
            "audit_board_filename": Path(
                str(reference_assets.get("audit_board") or "")
            ).name,
            "audit_board_sha256": reference_assets.get("audit_board_sha256"),
        }
    asset_manifest = profile.get("profile_asset_manifest")
    public_asset_manifest: dict[str, Any] | None = None
    if isinstance(asset_manifest, dict):
        public_asset_manifest = {
            "filename": Path(str(asset_manifest.get("path") or "")).name,
            "sha256": asset_manifest.get("sha256"),
            "content_sha256": asset_manifest.get("content_sha256"),
            "asset_validation_run_id": asset_manifest.get("asset_validation_run_id"),
            "validation_scope": asset_manifest.get("validation_scope"),
            "full_gpu_generation_validation": asset_manifest.get(
                "full_gpu_generation_validation"
            ),
            "end_to_end_api_validation": asset_manifest.get(
                "end_to_end_api_validation"
            ),
        }
    if isinstance(driving_standards, dict):
        raw_actions = driving_standards.get("actions")
        standard_actions: dict[str, Any] = {}
        if isinstance(raw_actions, dict):
            for action in SUBJECT_REPLACEMENT_ACTIONS:
                record = raw_actions.get(action)
                if not isinstance(record, dict):
                    continue
                public_standard_action = {
                    "source_filename": Path(str(record.get("source") or "")).name,
                    "source_sha256": record.get("source_sha256"),
                    "runtime_filename": Path(str(record.get("runtime") or "")).name,
                    "runtime_sha256": record.get("runtime_sha256"),
                    "validated_run_id": record.get("validated_run_id"),
                }
                if record.get("selected_frame_index") is not None:
                    public_standard_action.update(
                        {
                            "selected_frame_index": record.get("selected_frame_index"),
                            "preview_filename": Path(
                                str(record.get("preview") or "")
                            ).name,
                            "preview_sha256": record.get("preview_sha256"),
                            "reference_filename": Path(
                                str(record.get("reference") or "")
                            ).name,
                            "reference_sha256": record.get("reference_sha256"),
                            "identity_mask_filename": Path(
                                str(record.get("identity_mask") or "")
                            ).name,
                            "identity_mask_sha256": record.get("identity_mask_sha256"),
                        }
                    )
                standard_actions[action] = public_standard_action
        public_driving_standards = {
            "revision": driving_standards.get("revision"),
            "role": driving_standards.get("role"),
            "actions": standard_actions,
        }
    public_standard_direct_actions: dict[str, Any] = {}
    if isinstance(raw_standard_direct, dict):
        raw_direct_actions = raw_standard_direct.get("actions")
        if isinstance(raw_direct_actions, dict):
            for action in SUBJECT_REPLACEMENT_ACTIONS:
                record = raw_direct_actions.get(action)
                if not isinstance(record, dict):
                    continue
                public_standard_direct_actions[action] = {
                    "selected_frame_index": record.get("selected_frame_index"),
                    "preview_filename": Path(str(record.get("preview") or "")).name,
                    "preview_sha256": record.get("preview_sha256"),
                    "reference_filename": Path(str(record.get("reference") or "")).name,
                    "reference_sha256": record.get("reference_sha256"),
                    "identity_mask_filename": Path(
                        str(record.get("identity_mask") or "")
                    ).name,
                    "identity_mask_sha256": record.get("identity_mask_sha256"),
                }
    return {
        "id": profile["id"],
        "display_name": profile.get("display_name") or profile["id"],
        "status": profile.get("status"),
        "preview_action": profile.get("preview_action", "eating"),
        "driving_standards": public_driving_standards,
        "reference_assets": public_reference_assets,
        "profile_asset_manifest": public_asset_manifest,
        "standard_direct": (
            {
                "enabled": raw_standard_direct.get("enabled") is True,
                "policy": raw_standard_direct.get("policy"),
                "role": raw_standard_direct.get("role"),
                "revision": raw_standard_direct.get("revision"),
                "deterministic": raw_standard_direct.get("deterministic") is True,
                "gpu_required": False,
                "gpu_submitted": False,
                "gpu_bypass_reason": PROFILE_STANDARD_DIRECT_GPU_REASON,
                "gpu_validation": raw_standard_direct.get("gpu_validation"),
                "end_to_end_api_validation": raw_standard_direct.get(
                    "end_to_end_api_validation"
                ),
                "upload_identity_excluded": True,
                "actions": public_standard_direct_actions,
            }
            if isinstance(raw_standard_direct, dict)
            else None
        ),
        "actions": [
            {
                "id": action,
                "label": SUBJECT_REPLACEMENT_ACTION_LABELS[action],
                "reference_filename": Path(
                    str(action_references.get(action) or "")
                ).name,
                "reference_sha256": (
                    profile.get("validated_action_baselines", {})
                    .get(action, {})
                    .get("reference_sha256")
                    if isinstance(profile.get("validated_action_baselines"), dict)
                    else None
                ),
                "reference_ready": bool(
                    isinstance(action_references.get(action), str)
                    and Path(action_references[action]).is_file()
                ),
            }
            for action in SUBJECT_REPLACEMENT_ACTIONS
        ],
    }


def parse_subject_actions(raw: str | None) -> list[str]:
    if not raw or not raw.strip():
        return ["eating"]
    text = raw.strip()
    try:
        values = (
            json.loads(text) if text.startswith("[") else re.split(r"[|,\n]+", text)
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            "actions must be a JSON string array or a separated list"
        ) from exc
    if not isinstance(values, list) or not all(
        isinstance(item, str) for item in values
    ):
        raise ValueError("actions must be an array of strings")
    aliases = {
        "eat": "eating",
        "eating": "eating",
        "吃饭": "eating",
        "进食": "eating",
        "groom": "grooming",
        "grooming": "grooming",
        "licking": "grooming",
        "舔毛": "grooming",
    }
    actions: list[str] = []
    for value in values:
        key = aliases.get(value.strip().lower()) or aliases.get(value.strip())
        if key is None:
            raise ValueError(
                "Only eating and grooming subject-replacement actions are available."
            )
        if key not in actions:
            actions.append(key)
    if not actions:
        raise ValueError("At least one subject-replacement action is required.")
    return actions


def require_subject_replacement_assets(
    profile: dict[str, Any],
    actions: list[str],
    *,
    reference_paths: dict[str, str] | None = None,
    require_profile_references: bool = True,
    profile_reference_actions: list[str] | None = None,
    effective_config: dict[str, Any] | None = None,
    expected_pose_policies: dict[str, str] | None = None,
) -> None:
    paths = scail_model_paths()
    paths.extend(
        Path(effective_config["actions"][action]["driving_animation"])
        if effective_config is not None
        else SCAIL_ACTION_CONFIG[action]["driving_template"]
        for action in actions
    )
    if effective_config is not None:
        for action in actions:
            geometry_reference = str(
                effective_config["actions"][action].get("geometry_reference") or ""
            ).strip()
            if geometry_reference:
                paths.append(Path(geometry_reference))
    if effective_config is not None and GENERATION_VISUAL_AUDITS_BLOCKING:
        for action in actions:
            action_config = effective_config["actions"][action]
            surface_gate = action_config.get("fur_surface_gate")
            if isinstance(surface_gate, dict) and surface_gate.get("enabled"):
                surface_baseline = surface_gate.get("baseline")
                if not isinstance(surface_baseline, dict):
                    raise FileNotFoundError(
                        f"Required {action} fur-surface baseline configuration "
                        "is missing"
                    )
                paths.append(Path(str(surface_baseline.get("path") or "")))
            fur_gate = action_config.get("fur_silhouette_gate")
            if not isinstance(fur_gate, dict) or not fur_gate.get("enabled"):
                continue
            gate_policy = str(fur_gate.get("policy") or "")
            expected_pose_policy = str((expected_pose_policies or {}).get(action) or "")
            uploaded_absolute_gate = gate_policy == UPLOADED_ABSOLUTE_FUR_CONTOUR_POLICY
            if uploaded_absolute_gate:
                if action == "eating" and not uses_full_driver_eating_pose(
                    expected_pose_policy
                ):
                    raise FileNotFoundError(
                        "Uploaded eating fur gate requires the exact full-driver "
                        "pose policy"
                    )
            else:
                paths.append(Path(str(fur_gate.get("reference") or "")))
            fur_baseline = action_config.get("fur_contour_baseline")
            if (
                action == "eating"
                and not uploaded_absolute_gate
                and not uses_full_driver_eating_pose(expected_pose_policy)
            ):
                if not isinstance(fur_baseline, dict):
                    raise FileNotFoundError(
                        "Required eating fur-contour baseline configuration is missing"
                    )
                paths.append(Path(str(fur_baseline.get("path") or "")))
    if reference_paths is not None:
        paths.extend(Path(reference_paths[action]) for action in actions)
    elif require_profile_references:
        paths.extend(
            Path(profile["action_references"][action])
            for action in (profile_reference_actions or actions)
        )
    missing = [str(path) for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(
            "Required subject-replacement files are missing: " + ", ".join(missing)
        )


def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    if not API_KEY:
        raise HTTPException(status_code=503, detail="Server API key is not configured.")
    if not x_api_key or not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(
            status_code=401, detail="Invalid or missing X-API-Key header."
        )


def status_path(job_id: str) -> Path:
    return OUTPUT_ROOT / job_id / "status.json"


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def public_job(job: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(job))
    result.pop("active_prompt_id", None)
    public_stage_override = str(
        result.pop("public_stage_override", None) or ""
    ).strip()
    public_status = str(result.get("status") or "").strip().lower()
    if public_stage_override and public_status in {"queued", "running", "completed"}:
        result["stage"] = public_stage_override
    result["status_url"] = f"{API_PREFIX}/jobs/{job['job_id']}"
    result["cancel_url"] = f"{API_PREFIX}/jobs/{job['job_id']}/cancel"
    for output in result.get("outputs", []):
        output["download_url"] = (
            f"{API_PREFIX}/jobs/{job['job_id']}/files/{output['filename']}"
        )
    return result


def persist_job(job_id: str) -> None:
    with JOBS_LOCK:
        job = dict(JOBS[job_id])
    write_json_atomic(status_path(job_id), job)
    report_job_callback(job, force=True)


def report_job_callback(job: dict[str, Any], *, force: bool = False) -> bool:
    """Queue a callback without ever allowing notification failure to fail a job."""

    if not bool(job.get("callback_enabled")):
        return False
    try:
        return JOB_CALLBACK_REPORTER.enqueue(job, force=force)
    except Exception:  # noqa: BLE001 - generation must not depend on callback uptime
        LOGGER.exception(
            "Could not enqueue GIF callback for job %s",
            job.get("job_id"),
        )
        return False


def update_job(job_id: str, **changes: Any) -> dict[str, Any]:
    with JOBS_LOCK:
        job = JOBS[job_id]
        job.update(changes)
        job["updated_at"] = utc_now()
        snapshot = dict(job)
    write_json_atomic(status_path(job_id), snapshot)
    report_job_callback(snapshot)
    return snapshot


def record_nonblocking_job_anomaly(
    job_id: str,
    *,
    code: str,
    message: str,
    stage: str,
    details: dict[str, Any] | None = None,
    expose_as_public_stage: bool = False,
) -> dict[str, Any]:
    """Persist a result anomaly without changing the job's execution status."""

    anomaly = {
        "code": code,
        "message": message,
        "stage": stage,
        "blocking": False,
        "created_at": utc_now(),
        "details": dict(details or {}),
    }
    with JOBS_LOCK:
        job = JOBS[job_id]
        anomalies = [
            dict(item)
            for item in list(job.get("result_anomalies") or [])
            if isinstance(item, dict) and item.get("code") != code
        ]
        anomalies.append(anomaly)
        job["result_anomalies"] = anomalies
        job["result_has_anomalies"] = True
        job["result_anomaly_count"] = len(anomalies)
        if expose_as_public_stage:
            job["public_stage_override"] = code
        job["updated_at"] = utc_now()
        snapshot = dict(job)
    write_json_atomic(status_path(job_id), snapshot)
    return anomaly


def nonblocking_result_error(job_id: str) -> str | None:
    """Return the user-facing error text for a successfully completed job."""

    with JOBS_LOCK:
        anomalies = list(JOBS[job_id].get("result_anomalies") or [])
    messages = [
        str(item.get("message") or "").strip()
        for item in anomalies
        if isinstance(item, dict) and str(item.get("message") or "").strip()
    ]
    return "\n".join(dict.fromkeys(messages)) or None


def append_output(job_id: str, output: dict[str, Any]) -> None:
    with JOBS_LOCK:
        outputs = list(JOBS[job_id].get("outputs", []))
        outputs.append(output)
        JOBS[job_id]["outputs"] = outputs
        JOBS[job_id]["updated_at"] = utc_now()
        snapshot = dict(JOBS[job_id])
    write_json_atomic(status_path(job_id), snapshot)


def append_recovery_event(job_id: str, event: dict[str, Any]) -> None:
    with JOBS_LOCK:
        history = list(JOBS[job_id].get("recovery_history", []))
        history.append(event)
        JOBS[job_id]["recovery_history"] = history
        JOBS[job_id]["automatic_recovery_used"] = True
        if event.get("strategy") == "rembg_u2net_fallback":
            JOBS[job_id]["degraded_recovery"] = True
        JOBS[job_id]["updated_at"] = utc_now()
        snapshot = dict(JOBS[job_id])
    write_json_atomic(status_path(job_id), snapshot)


def append_background_removal_failure(job_id: str, event: dict[str, Any]) -> None:
    """Record an opaque image fallback without claiming that recovery succeeded."""
    with JOBS_LOCK:
        failures = list(JOBS[job_id].get("background_removal_failures", []))
        failures.append(event)
        JOBS[job_id]["background_removal_failures"] = failures
        JOBS[job_id]["background_removal_failed"] = True
        JOBS[job_id]["transparent_outputs_complete"] = False
        JOBS[job_id]["updated_at"] = utc_now()
        snapshot = dict(JOBS[job_id])
    write_json_atomic(status_path(job_id), snapshot)


def get_job(job_id: str) -> dict[str, Any]:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        path = status_path(job_id)
        if path.is_file():
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
                with JOBS_LOCK:
                    JOBS[job_id] = job
            except (OSError, json.JSONDecodeError):
                job = None
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return public_job(job)


def cancel_requested(job_id: str) -> bool:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        return bool(
            job and (job.get("cancel_requested") or job.get("status") == "cancelled")
        )


def raise_if_cancelled(job_id: str | None) -> None:
    if job_id and cancel_requested(job_id):
        raise JobCancelled("Generation was cancelled by the user.")


def interrupt_comfy_prompt(prompt_id: str | None) -> None:
    if not prompt_id:
        return
    errors: list[str] = []
    try:
        response = requests.post(
            f"{COMFY_URL}/queue",
            json={"delete": [prompt_id]},
            timeout=30,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        errors.append(f"queue delete: {exc}")
    try:
        response = requests.post(
            f"{COMFY_URL}/interrupt",
            json={"prompt_id": prompt_id},
            timeout=30,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        errors.append(f"interrupt: {exc}")
    if errors:
        LOGGER.warning(
            "Could not fully interrupt ComfyUI prompt %s: %s",
            prompt_id,
            "; ".join(errors),
        )


def mark_job_cancelled(job_id: str) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        progress = job.get("progress", 0) if job else 0
    update_job(
        job_id,
        status="cancelled",
        stage="cancelled",
        progress=progress,
        cancel_requested=True,
        active_prompt_id=None,
        current_pose_index=None,
        current_pose=None,
        error=None,
        finished_at=utc_now(),
    )


def parse_poses(
    raw: str | None,
    pose_count: int,
    *,
    defaults: list[str] | None = None,
) -> list[str]:
    if not 1 <= pose_count <= MAX_POSES:
        raise HTTPException(
            status_code=422, detail=f"pose_count must be between 1 and {MAX_POSES}."
        )
    if raw and raw.strip():
        text = raw.strip()
        try:
            if text.startswith("["):
                value = json.loads(text)
                if not isinstance(value, list) or not all(
                    isinstance(item, str) for item in value
                ):
                    raise ValueError("poses JSON must be an array of strings")
                poses = value
            else:
                separator = "|" if "|" in text else "\n"
                poses = text.split(separator)
        except (json.JSONDecodeError, ValueError) as exc:
            raise HTTPException(
                status_code=422, detail=f"Invalid poses: {exc}"
            ) from exc
        poses = [item.strip() for item in poses if item.strip()]
    else:
        pose_defaults = list(defaults) if defaults is not None else DEFAULT_POSES
        poses = pose_defaults[:pose_count]
        while len(poses) < pose_count:
            poses.append(f"distinct natural pose number {len(poses) + 1}")
    if not poses:
        raise HTTPException(status_code=422, detail="At least one pose is required.")
    if len(poses) > MAX_POSES:
        raise HTTPException(
            status_code=422, detail=f"At most {MAX_POSES} poses are allowed per job."
        )
    if any(len(item) > 300 for item in poses):
        raise HTTPException(
            status_code=422, detail="Each pose must be at most 300 characters."
        )
    return poses


def contains_han(text: str) -> bool:
    return bool(_HAN_RE.search(text))


def normalize_chinese_prompt_for_translation(text: str) -> str:
    """Clarify reference-image wording that generic MT often reads as an upload command."""
    replacements = (
        ("参考上传的图片中的", "参考图片中的"),
        ("参考上传图片中的", "参考图片中的"),
        ("参考上传的图片中", "参考图片中"),
        ("参考上传图片中", "参考图片中"),
        ("上传的图片中的", "参考图片中的"),
        ("上传图片中的", "参考图片中的"),
        ("上传的图片中", "参考图片中"),
        ("上传图片中", "参考图片中"),
        ("上传的猫咪", "参考图片中的猫咪"),
        ("上传猫咪", "参考图片中的猫咪"),
        ("上传的宠物", "参考图片中的宠物"),
        ("上传宠物", "参考图片中的宠物"),
        ("上传的图片", "参考图片"),
        ("上传图片", "参考图片"),
    )
    normalized = text
    for source, target in replacements:
        normalized = normalized.replace(source, target)
    return normalized


def prompt_translator_missing_paths() -> list[str]:
    required = [
        PROMPT_TRANSLATOR_PYTHON,
        PROMPT_TRANSLATOR_SCRIPT,
        PROMPT_TRANSLATOR_MODEL / "config.json",
        PROMPT_TRANSLATOR_MODEL / "source.spm",
        PROMPT_TRANSLATOR_MODEL / "target.spm",
        PROMPT_TRANSLATOR_MODEL / "vocab.json",
        PROMPT_TRANSLATOR_MODEL / "translation_ready.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    weights = [
        PROMPT_TRANSLATOR_MODEL / "pytorch_model.bin",
        PROMPT_TRANSLATOR_MODEL / "model.safetensors",
    ]
    if not any(path.is_file() for path in weights):
        missing.append(
            str(PROMPT_TRANSLATOR_MODEL / "{pytorch_model.bin|model.safetensors}")
        )
    return missing


def translate_texts_to_english(
    texts: list[str],
) -> tuple[list[str], dict[str, Any]]:
    """Translate only Han-containing strings and fail closed if translation is unavailable."""
    unique_pending: list[str] = []
    normalized_pending: list[str] = []
    methods: dict[str, str] = {}
    resolved: dict[str, str] = {}
    with _PROMPT_TRANSLATION_CACHE_LOCK:
        cached = dict(_PROMPT_TRANSLATION_CACHE)
    for text in texts:
        if not contains_han(text):
            resolved[text] = text
            methods[text] = "identity"
        elif text in cached:
            resolved[text], methods[text] = cached[text]
        elif text not in unique_pending:
            unique_pending.append(text)

    if unique_pending:
        normalized_pending = [
            normalize_chinese_prompt_for_translation(text) for text in unique_pending
        ]
        if not PROMPT_TRANSLATION_ENABLED:
            raise HTTPException(
                status_code=503,
                detail="Chinese prompt translation is disabled on this server.",
            )
        missing = prompt_translator_missing_paths()
        if missing:
            raise HTTPException(
                status_code=503,
                detail=(
                    "The local Chinese-to-English prompt translator is not ready: "
                    + ", ".join(missing)
                ),
            )
        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": "",
                "TOKENIZERS_PARALLELISM": "false",
                "OMP_NUM_THREADS": str(PROMPT_TRANSLATION_THREADS),
                "MKL_NUM_THREADS": str(PROMPT_TRANSLATION_THREADS),
            }
        )
        try:
            process = subprocess.run(
                [
                    str(PROMPT_TRANSLATOR_PYTHON),
                    str(PROMPT_TRANSLATOR_SCRIPT),
                    "--model",
                    str(PROMPT_TRANSLATOR_MODEL),
                ],
                input=json.dumps({"texts": normalized_pending}, ensure_ascii=False),
                capture_output=True,
                text=True,
                timeout=PROMPT_TRANSLATION_TIMEOUT,
                check=False,
                env=environment,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Local prompt translation failed to start: {type(exc).__name__}.",
            ) from exc
        if process.returncode != 0:
            LOGGER.error(
                "Local prompt translator exited with %s: %s",
                process.returncode,
                (process.stderr or process.stdout or "unknown error")[-2000:],
            )
            raise HTTPException(
                status_code=502,
                detail="Local Chinese-to-English prompt translation failed.",
            )
        try:
            payload = json.loads(process.stdout)
            translated = payload["translations"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise HTTPException(
                status_code=502,
                detail="Local prompt translator returned an invalid response.",
            ) from exc
        if not isinstance(translated, list) or len(translated) != len(unique_pending):
            raise HTTPException(
                status_code=502,
                detail="Local prompt translator returned an incomplete response.",
            )
        for source, target in zip(unique_pending, translated, strict=True):
            normalized = str(target).strip()
            if not normalized or len(normalized) > 4000 or contains_han(normalized):
                raise HTTPException(
                    status_code=502,
                    detail="Local prompt translation did not produce valid English text.",
                )
            resolved[source] = normalized
            methods[source] = "local_opus_mt_zh_en"
            with _PROMPT_TRANSLATION_CACHE_LOCK:
                _PROMPT_TRANSLATION_CACHE[source] = (
                    normalized,
                    "local_opus_mt_zh_en",
                )

    ordered = [resolved[text] for text in texts]
    applied_methods = {methods[text] for text in texts if methods[text] != "identity"}
    provider = (
        next(iter(applied_methods))
        if len(applied_methods) == 1
        else ("mixed" if applied_methods else "identity")
    )
    return ordered, {
        "applied": bool(applied_methods),
        "provider": provider,
        "model": (
            "Helsinki-NLP/opus-mt-zh-en"
            if "local_opus_mt_zh_en" in applied_methods
            else None
        ),
        "source_language": "zh",
        "target_language": "en",
        "translated_item_count": sum(methods[text] != "identity" for text in texts),
        "semantic_normalization_count": sum(
            source != normalized
            for source, normalized in zip(
                unique_pending,
                normalized_pending,
                strict=True,
            )
        ),
        "methods": [methods[text] for text in texts],
    }


def infer_pose_action(source: str, translated: str) -> str | None:
    combined = f"{source}\n{translated}".lower()
    if re.search(r"喝水|饮水|舔水|\bdrinking\b|\bdrink(?:ing)?\s+water\b", combined):
        return "drinking"
    if re.search(
        r"进食|吃饭|吃东西|猫粮|食盆|\beating\b|\bfood bowl\b|\bkibble\b", combined
    ):
        return "eating"
    if re.search(r"奔跑|跑步|快跑|\brunning\b|\brun(?:ning)?\b", combined):
        return "running"
    if re.search(r"趴卧|趴下|躺下|卧姿|\blying down\b|\bprone\b", combined):
        return "lying down"
    return None


def normalize_translated_pose(source: str, translated: str) -> str:
    translated = translated.strip()
    first_key = translated.split("|", 1)[0].strip().lower()
    if first_key in {"running", "eating", "drinking", "lying", "lying down"}:
        return translated
    inferred_action = infer_pose_action(source, translated)
    if not inferred_action:
        return translated
    return f"{inferred_action} | first frame: {translated}"


def translate_prompt_bundle(
    prompt: str,
    poses: list[str],
) -> tuple[str, list[str], dict[str, Any]]:
    translated, metadata = translate_texts_to_english([prompt, *poses])
    prompt_en = translated[0]
    poses_en = [
        normalize_translated_pose(source, target)
        for source, target in zip(poses, translated[1:], strict=True)
    ]
    metadata.update(
        {
            "prompt_applied": contains_han(prompt),
            "pose_applied_count": sum(contains_han(pose) for pose in poses),
        }
    )
    return prompt_en, poses_en, metadata


def normalize_uploaded_image(
    raw: bytes,
    output_path: Path,
    *,
    identity_source_path: Path | None = None,
) -> None:
    try:
        with Image.open(io.BytesIO(raw)) as probe:
            probe.verify()
        with Image.open(io.BytesIO(raw)) as source:
            rgba = source.convert("RGBA")
            if identity_source_path is not None:
                identity_source_path.parent.mkdir(parents=True, exist_ok=True)
                rgba.save(identity_source_path, format="PNG", optimize=True)
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            background.alpha_composite(rgba)
            rgb = background.convert("RGB")
            if max(rgb.size) > 4096:
                rgb.thumbnail((4096, 4096), Image.Resampling.LANCZOS)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            rgb.save(output_path, format="PNG", optimize=True)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(
            status_code=422, detail="Uploaded file is not a valid supported image."
        ) from exc


def validate_safetensors_file(path: Path) -> int:
    size = path.stat().st_size
    if size < 16:
        raise HTTPException(
            status_code=422, detail="Uploaded LoRA is not a valid safetensors file."
        )
    with path.open("rb") as handle:
        header_size = int.from_bytes(handle.read(8), byteorder="little", signed=False)
        if header_size <= 2 or header_size > min(size - 8, 64 * 1024 * 1024):
            raise HTTPException(
                status_code=422,
                detail="Uploaded LoRA has an invalid safetensors header.",
            )
        try:
            header = json.loads(handle.read(header_size).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HTTPException(
                status_code=422,
                detail="Uploaded LoRA has an unreadable safetensors header.",
            ) from exc
    if not isinstance(header, dict):
        raise HTTPException(
            status_code=422, detail="Uploaded LoRA has an invalid tensor index."
        )
    data_size = size - 8 - header_size
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    if not tensors:
        raise HTTPException(
            status_code=422, detail="Uploaded LoRA contains no tensors."
        )
    lora_key_seen = False
    for key, value in tensors.items():
        if any(marker in key.lower() for marker in ("lora", "lokr", "loha", "dora")):
            lora_key_seen = True
        if not isinstance(value, dict):
            raise HTTPException(
                status_code=422, detail="Uploaded LoRA has an invalid tensor entry."
            )
        offsets = value.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(item, int) for item in offsets)
            or offsets[0] < 0
            or offsets[0] > offsets[1]
            or offsets[1] > data_size
        ):
            raise HTTPException(
                status_code=422, detail="Uploaded LoRA has invalid tensor offsets."
            )
    if not lora_key_seen:
        raise HTTPException(
            status_code=422,
            detail="Uploaded safetensors file does not appear to contain LoRA weights.",
        )
    return len(tensors)


async def store_user_lora(upload: UploadFile | None) -> dict[str, Any] | None:
    if upload is None or not upload.filename:
        return None
    original_name = upload.filename.replace("\\", "/").rsplit("/", 1)[-1]
    if not original_name.lower().endswith(".safetensors"):
        await upload.close()
        raise HTTPException(
            status_code=422, detail="User LoRA must be a .safetensors file."
        )
    USER_LORA_DIR.mkdir(parents=True, exist_ok=True)
    temporary = USER_LORA_DIR / f".upload-{uuid4().hex}.tmp"
    digest = hashlib.sha256()
    total = 0
    try:
        with temporary.open("wb") as target:
            while chunk := await upload.read(4 * 1024 * 1024):
                total += len(chunk)
                if total > MAX_LORA_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"User LoRA exceeds {MAX_LORA_UPLOAD_BYTES} bytes.",
                    )
                digest.update(chunk)
                target.write(chunk)
        if total == 0:
            raise HTTPException(status_code=422, detail="Uploaded LoRA is empty.")
        tensor_count = validate_safetensors_file(temporary)
        sha256 = digest.hexdigest()
        safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(original_name).stem).strip(
            "-._"
        )
        safe_stem = (safe_stem or "uploaded-lora")[:48]
        stored_name = f"{sha256[:16]}-{safe_stem}.safetensors"
        existing_matches = sorted(USER_LORA_DIR.glob(f"{sha256[:16]}-*.safetensors"))
        final_path = (
            existing_matches[0] if existing_matches else USER_LORA_DIR / stored_name
        )
        stored_name = final_path.name
        if final_path.exists():
            temporary.unlink()
        else:
            os.replace(temporary, final_path)
            final_path.chmod(0o640)
        return {
            "model_name": f"user_uploads/{stored_name}",
            "filename": stored_name,
            "original_filename": original_name,
            "sha256": sha256,
            "size_bytes": total,
            "tensor_count": tensor_count,
        }
    finally:
        await upload.close()
        temporary.unlink(missing_ok=True)


def resolve_user_lora(model_name: str | None) -> str | None:
    if not model_name or not model_name.strip():
        return None
    normalized = model_name.strip().replace("\\", "/")
    relative = Path(normalized)
    if (
        relative.is_absolute()
        or len(relative.parts) != 2
        or relative.parts[0] != "user_uploads"
        or relative.suffix.lower() != ".safetensors"
    ):
        raise HTTPException(status_code=422, detail="Invalid user LoRA model name.")
    candidate = (MODEL_ROOT / "loras" / relative).resolve()
    try:
        candidate.relative_to(USER_LORA_DIR.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Invalid user LoRA path.") from exc
    if not candidate.is_file():
        raise HTTPException(status_code=422, detail="Uploaded user LoRA is missing.")
    return relative.as_posix()


def normalize_user_lora_specs(
    value: Any,
    *,
    legacy_model_name: str | None = None,
    legacy_original_filename: str | None = None,
    legacy_strength: float = 1.0,
    resolve_models: bool = True,
) -> list[dict[str, Any]]:
    """Normalize the new multi-LoRA payload while retaining legacy single-LoRA support."""
    if isinstance(value, str):
        if not value.strip():
            value = None
        else:
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise HTTPException(
                    status_code=422,
                    detail="user_loras must be a valid JSON array.",
                ) from exc
    if value is None:
        raw_items: list[Any] = []
        if legacy_model_name and legacy_model_name.strip():
            raw_items.append(
                {
                    "model_name": legacy_model_name,
                    "original_filename": legacy_original_filename,
                    "strength": legacy_strength,
                }
            )
    elif isinstance(value, list):
        raw_items = value
    else:
        raise HTTPException(status_code=422, detail="user_loras must be a JSON array.")
    if len(raw_items) > MAX_USER_LORAS:
        raise HTTPException(
            status_code=422,
            detail=f"At most {MAX_USER_LORAS} user LoRAs may be fused in one job.",
        )

    normalized: list[dict[str, Any]] = []
    seen_models: set[str] = set()
    resolved_total_size = 0
    for index, item in enumerate(raw_items, start=1):
        if not isinstance(item, dict):
            raise HTTPException(
                status_code=422,
                detail=f"user_loras[{index - 1}] must be an object.",
            )
        model_name = item.get("model_name") or item.get("user_lora") or item.get("name")
        if not isinstance(model_name, str) or not model_name.strip():
            raise HTTPException(
                status_code=422,
                detail=f"user_loras[{index - 1}].model_name is required.",
            )
        if resolve_models:
            resolved_name = resolve_user_lora(model_name)
            if resolved_name is None:
                raise HTTPException(
                    status_code=422,
                    detail=f"user_loras[{index - 1}].model_name is invalid.",
                )
            resolved_total_size += (
                (MODEL_ROOT / "loras" / Path(resolved_name)).stat().st_size
            )
            if resolved_total_size > MAX_LORA_FUSION_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        "Combined user LoRA files exceed "
                        f"{MAX_LORA_FUSION_BYTES} bytes."
                    ),
                )
        else:
            resolved_name = model_name.strip().replace("\\", "/")
        if resolved_name in seen_models:
            raise HTTPException(
                status_code=422,
                detail=f"Duplicate user LoRA is not allowed: {resolved_name}.",
            )
        seen_models.add(resolved_name)
        try:
            strength = float(item.get("strength", 1.0))
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=422,
                detail=f"user_loras[{index - 1}].strength must be a number.",
            ) from exc
        if not 0.0 <= strength <= 2.0:
            raise HTTPException(
                status_code=422,
                detail=f"user_loras[{index - 1}].strength must be between 0.0 and 2.0.",
            )
        original_filename = item.get("original_filename")
        if original_filename is not None:
            original_filename = (
                str(original_filename).replace("\\", "/").rsplit("/", 1)[-1][:255]
            )
        normalized.append(
            {
                "model_name": resolved_name,
                "original_filename": original_filename,
                "strength": strength,
            }
        )
    return normalized


def resolve_image_profile(
    profile_id: str | None,
    *,
    request_kind: str,
) -> dict[str, Any] | None:
    normalized = (profile_id or "").strip()
    if not normalized:
        return None
    if request_kind not in {"image", "video"}:
        raise HTTPException(
            status_code=422,
            detail="image_profile requires an image or video request.",
        )
    profile = IMAGE_PROFILES.get(normalized)
    if profile is None:
        raise HTTPException(status_code=422, detail="Unknown image_profile.")
    return profile


def image_profile_public(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": profile["id"],
        "revision": profile.get("revision"),
        "label": profile["label"],
        "source_job_id": profile["source_job_id"],
        "source_prompt": profile.get("source_prompt"),
        "source_prompt_en": profile.get("source_prompt_en"),
        "default_prompt": profile["default_prompt"],
        "default_prompt_en": profile["default_prompt_en"],
        "pose_prompt_source": profile.get("pose_prompt_source"),
        "pose_presets": [
            {
                "id": preset["id"],
                "action_key": preset["action_key"],
                "label": preset["label"],
                "prompt_zh": preset["prompt_zh"],
                "prompt_en": preset["prompt_en"],
                "video_prompt_en": preset["video_prompt_en"],
            }
            for preset in profile.get("pose_presets", [])
        ],
        "pose_seed_offsets": dict(profile.get("pose_seed_offsets", {})),
        "identity_palette_guard": bool(profile.get("identity_palette_guard", False)),
        "identity_palette_policy": profile.get("identity_palette_policy"),
        "lock_retry_lora_strengths": bool(
            profile.get("lock_retry_lora_strengths", False)
        ),
        "user_loras": [
            {
                "original_filename": item["original_filename"],
                "strength": item["strength"],
                "available": (
                    MODEL_ROOT / "loras" / Path(item["model_name"])
                ).is_file(),
            }
            for item in profile["user_loras"]
        ],
        "user_lora_mode": profile["user_lora_mode"],
        "qwen_steps": profile["qwen_steps"],
        "qwen_cfg": profile["qwen_cfg"],
        "qwen_denoise": profile["qwen_denoise"],
        "seed": profile["seed"],
    }


merge_image_profile_loras = _local_video_prompting_workflows.merge_image_profile_loras


cap_user_lora_specs = _local_video_prompting_workflows.cap_user_lora_specs


reduced_user_lora_specs = _local_video_prompting_workflows.reduced_user_lora_specs


action_key = _local_video_prompting_workflows.action_key


subject_replacement_pose_prompt = (
    _local_video_prompting_workflows.subject_replacement_pose_prompt
)


action_uses_bowl = _local_video_prompting_workflows.action_uses_bowl


pose_seed_offset = _local_video_prompting_workflows.pose_seed_offset


qwen_first_frame_spec = _local_video_prompting_workflows.qwen_first_frame_spec


action_first_frame_rules = _local_video_prompting_workflows.action_first_frame_rules


action_video_rules = _local_video_prompting_workflows.action_video_rules


build_cartoon_reference_prompt = _local_video_prompting_workflows.build_cartoon_reference_prompt


def build_qwen_prompt(
    base_prompt: str,
    pose: str,
    transparent_gif: bool = False,
    quality_retry: int = 0,
    pose_reference_available: bool = False,
    profile_fur_lock: str = "",
    pose_identity_input_policy: str = POSE_IDENTITY_INPUT_POLICY_CANONICAL,
) -> str:
    if pose_identity_input_policy not in {
        POSE_IDENTITY_INPUT_POLICY_CANONICAL,
        POSE_IDENTITY_INPUT_POLICY_DIRECT_UPLOAD,
        POSE_IDENTITY_INPUT_POLICY_PROFILE_GEOMETRY_FIRST,
        POSE_IDENTITY_INPUT_POLICY_GROOMING_GEOMETRY_CANONICAL,
    }:
        raise ValueError(
            f"Unsupported pose identity input policy: {pose_identity_input_policy}"
        )
    direct_uploaded_identity = (
        pose_identity_input_policy == POSE_IDENTITY_INPUT_POLICY_DIRECT_UPLOAD
    )
    main_case_geometry_first = (
        pose_identity_input_policy == POSE_IDENTITY_INPUT_POLICY_PROFILE_GEOMETRY_FIRST
    )
    grooming_geometry_canonical = (
        pose_identity_input_policy
        == POSE_IDENTITY_INPUT_POLICY_GROOMING_GEOMETRY_CANONICAL
    )
    if main_case_geometry_first and action_key(pose) not in {"eating", "grooming"}:
        raise ValueError(
            "Main_case geometry-first identity supports only eating and grooming."
        )
    if grooming_geometry_canonical and action_key(pose) != "grooming":
        raise ValueError(
            "Canonical-identity geometry-first input supports only grooming."
        )
    secondary_identity_required = (
        direct_uploaded_identity
        or main_case_geometry_first
        or grooming_geometry_canonical
    )
    if secondary_identity_required and transparent_gif and not pose_reference_available:
        raise ValueError(
            "The selected pose identity policy requires a validated secondary "
            "action or identity image."
        )
    background_prompt = (
        CHROMA_BACKGROUND_PROMPT
        if transparent_gif
        else "Use a clean simple studio-like background with soft even lighting."
    )
    profile_fur_section = (
        "\n\nUploaded identity coat preservation — non-negotiable: "
        + profile_fur_lock.strip()
        if profile_fur_lock.strip()
        else ""
    )
    if transparent_gif:
        first_frame_rules = action_first_frame_rules(pose)
        first_frame_spec = qwen_first_frame_spec(pose)
        bowl_action = action_uses_bowl(pose)
        retry_instruction = ""
        if quality_retry >= 2:
            retry_instruction = """
VALIDATION RETRY: the previous result failed one or more identity, pose, scale, or green-background checks. Recreate the action-specific full-body keyframe while preserving the exact character identity, source coat palette, pulled-back framing, and pure-green canvas rules."""
        if quality_retry >= 3:
            if bowl_action:
                retry_instruction += """
FINAL BOWL-ACTION RETRY: keep a fixed pulled-back camera and do not enlarge the cat or bowl. Express the action through a natural low torso and a lowered head with the lower muzzle inside and overlapping the visible kibble. Keep at least 12% pure-green clearance on every edge, keep the complete cat and bowl within 72% of the canvas width and height, and keep all four canvas corners exactly pure green."""
            elif action_key(pose) == "running":
                retry_instruction += """
FINAL POSE-CORRECTION RETRY: prioritize a complete grounded pre-run stance over visible motion. Keep all four legs and paws separated and readable, with at least three unambiguous ground contacts. Do not make the action dynamic and do not lift the body."""
            else:
                retry_instruction += """
FINAL POSE-CORRECTION RETRY: make the requested action unmistakable at thumbnail size. Change the torso angle, leg placement, head height, and tail placement enough to be visibly distinct from a neutral sitting or standing reference. No prop may appear."""
        if quality_retry >= 4 and action_key(pose) == "running":
            retry_instruction += """
RUNNING GROUND-CONTACT RECOVERY: ignore the motion implied by the action name in this still frame. Render a calm static four-paw standing ready pose. All four legs and paws must be fully visible, separated, and grounded; do not lift a paw, hide a far-side leg, or depict forward motion. The later video model, not this image, will create the run."""
        if quality_retry >= 4 and bowl_action:
            bowl_geometry_image = "Image 1" if main_case_geometry_first else "Image 2"
            bowl_identity_source = (
                "the original uploaded pet identity in Image 2"
                if main_case_geometry_first
                else (
                    "the uploaded pet photo"
                    if direct_uploaded_identity
                    else "the approved character"
                )
            )
            retry_instruction += f"""
BOWL PALETTE-AND-CANVAS RECOVERY: use neutral-white character lighting with no warm grading or amber bounce on the fur. Copy every fur color and marking region from {bowl_identity_source} exactly. Keep exactly one shallow bowl touching or slightly overlapping the two planted front paws, exactly as shown by {bowl_geometry_image}. Never omit the bowl or visible brown kibble. Do not place a green gap between the bowl and both front paws. Pull back enough that the complete cat and bowl have generous green padding and every canvas corner remains pure #00FF00."""
        subject_contact_prompt = (
            "Fur and the single action-required bowl must touch green directly."
            if bowl_action
            else "Fur must touch green directly. No prop may appear."
        )
        if pose_reference_available and grooming_geometry_canonical:
            grooming_fur_lock = profile_fur_lock.strip() or (
                "Preserve Image 2's exact natural coat length, visible volume, "
                "strand direction, genuine tufts, and source-bounded silhouette."
            )
            return f"""CRITICAL GROOMING TRANSFER CONTRACT - satisfy this before every other detail:
Output exactly one completely unclothed natural cat. Any hat, hood, shirt, coat, sweater, collar, costume, accessory, fabric, button, seam, or garment is invalid.

IMAGE ROLES - fixed and non-negotiable:
Image 1 is the appearance-free grooming geometry canvas and the sampler layout. It contains pure #00FF00 background, artificial RGB(144,144,144) for the body geometry, and artificial RGB(176,176,176) only for one continuous raised-foreleg capsule running from the shoulder to the paw at the mouth. A narrow green negative-space channel marks the torso-side boundary of that foreleg. Both grays are fixed synthetic anatomy labels: they have zero identity, fur-color, marking, material, texture, shadow, lighting, clothing, or style meaning. Image 1 controls only the exact seated pose, raised-foreleg path, paw-to-mouth contact, planted-paw placement, subject bounds, camera, and green canvas.
Image 2 is the approved Qwen full-body canonical and the sole authority for identity and appearance: exact face, age, eye color, ears, nose, muzzle, complete coat palette, localized markings, body proportions, natural coat length and volume, strand direction, outer fur silhouette, tail, and restrained stylized 3D material. Image 2 does not control pose.

TASK: repaint both artificial gray anatomy labels of Image 1 as the exact natural-fur cat from Image 2, articulated around Image 1's grooming geometry. The RGB(176) capsule must become the cat's own continuous raised foreleg and correctly colored paw at the mouth, never a sleeve, strap, zipper, cord, pen, stick, pendant, or detached object. The output must contain no gray guide material, green internal line, or garment. Close the synthetic green separator with natural fur while preserving the readable foreleg/torso boundary. Do not return Image 2's neutral seated pose.

POSE CONTRACT: {first_frame_spec}
{first_frame_rules}
The raised foreleg must remain visibly continuous from shoulder to one correctly colored rounded paw touching the mouth. Its paw must use the same canonical paw material and corresponding fur palette as the planted paw; the synthetic label color must never survive. Exactly one other front paw is planted. A tiny pink tongue touches the raised paw. Keep both attached hind legs and the full tail. No duplicate ground paw, ghost limb, detached paw, food, bowl, object, or extra subject.

IDENTITY CONTRACT: copy Image 2 only. Preserve its exact dominant/supporting coat-color balance and every localized marking. Never invent any hue, marking, or clothing color absent from Image 2; do not globally warm, cool, darken, lighten, or recolor the coat. Use neutral-white character lighting. {grooming_fur_lock}

OUTPUT CONTRACT: exactly one clean static 512x512 first frame; complete cat within 72% of canvas width and height; at least 12% clear green margin on every edge. Every pixel outside the cat must be flat pure #00FF00. No floor, floor line, shadow, reflection, scenery, text, logo, or watermark. {subject_contact_prompt}
{retry_instruction}
Return the corrected grooming keyframe only."""
        pose_reference_rules = ""
        pose_edit_intro = (
            "The input is the approved canonical character reference. Repose this exact character only; "
            "do not redesign the face, natural fur, or rendering style."
        )
        style_source_rules = (
            "Style source: always inherit the approved canonical reference in Image 1, "
            "including its exact rendering style, fur finish, expression design, lighting, "
            "colors, and identity. Image 2 never controls identity or style."
        )
        character_lock_source = "the approved canonical character in Image 1"
        identity_image_label = "Image 1"
        grooming_conflict_source = "the canonical reference"
        final_style_rule = (
            "Keep the exact identity-faithful restrained stylized 3D rendering from "
            "the reference."
        )
        main_case_coat_surface_rule = ""
        if pose_reference_available:
            if main_case_geometry_first:
                main_case_action = action_key(pose)
                if main_case_action == "eating":
                    pose_reference_rules = """
IMAGE ROLES — non-negotiable: Image 1 is the texture-free low-frequency geometry guide derived directly from the frozen Main_case eat.mp4 standard and is the sampler's geometry canvas. It is authoritative only for joint placement, the complete full-body eating pose, low torso position, exact low head height, muzzle-to-kibble contact, bowl placement, camera layout, framing, and pure-green background. Its small light-neutral rounded field at the food is an identity-free muzzle-contact semantic: render the uploaded cat's lower muzzle there and keep it visibly overlapping the kibble; do not interpret that field as a ball, tongue, egg, accessory, marking, or separate object. Image 1 and the subject profile are not authoritative for identity, body proportions, coat length, coat volume, strand direction, edge fur, chest or cheek grooming, tail fur, outer fur silhouette, color, markings, eyes, or breed. Copy no appearance or lighting hue from Image 1.

Image 2 is the original uploaded pet photo and is the sole appearance authority: preserve its exact face and head shape, eye size, color and spacing, ear shape, nose and muzzle, fur base colors, marking placement, visible body proportions, natural coat length and volume, strand direction, outer fur silhouette, and tail identity. It is not authoritative for pose, crop, camera, background, or photographic lighting. Where the crop does not show an area, extend the nearest visible proportions, coat behavior, and dominant color conservatively without borrowing anything from Main_case. Ignore Image 2's portrait head height and put the same pet into Image 1's validated low eating geometry. The muzzle must stay at the visible kibble instead of returning to portrait height. The separate identity-palette audit remains locked to the original uploaded file. Follow the written action rules whenever either image conflicts with them."""
                    action_geometry_label = (
                        "eating pose, bowl, muzzle contact and framing"
                    )
                else:
                    pose_reference_rules = """
IMAGE ROLES — non-negotiable: Image 1 is the appearance-free synthetic-anatomy geometry guide derived from the frozen Main_case TM.mp4 standard and is the sampler's geometry canvas. It is authoritative only for joint placement, the stable seated grooming posture, one continuous raised foreleg and rounded paw touching the mouth, one planted front paw, naturally tucked hind legs, camera layout, framing, and pure-green background. Its fixed RGB(144,144,144) label represents the body and its fixed RGB(176,176,176) capsule represents only the continuous shoulder-to-mouth raised foreleg; the narrow green channel marks that foreleg's torso-side boundary. These labels are generated only from verified geometry and have zero color, marking, material, texture, shading, lighting, clothing, or identity meaning. Image 1 and the subject profile are not authoritative for identity, body proportions, coat length, coat volume, strand direction, edge fur, outer fur silhouette, color, markings, eyes, breed, material, accessory, or black object. Copy no appearance or lighting hue from Image 1, and never leave either artificial gray or the internal green channel in the result.

Image 2 is the original uploaded pet photo and is the sole appearance authority: preserve its exact face and head shape, eye size, color and spacing, ear shape, nose and muzzle, fur base colors, marking placement, visible body proportions, natural coat length and volume, strand direction, outer fur silhouette, and tail identity. It is not authoritative for pose, crop, camera, background, or photographic lighting. Where the crop does not show an area, extend the nearest visible proportions, coat behavior, and dominant color conservatively without borrowing anything from Main_case. Render Image 1's RGB(176) capsule as the cat's own correctly colored continuous foreleg ending in one rounded paw at the mouth, never as a sleeve, strap, zipper, cord, pen, stick, pendant, other object, or detached shape. Put Image 2's exact identity and appearance into Image 1's validated Main_case grooming geometry. The separate identity-palette audit remains locked to the original uploaded file. Follow the written action rules whenever either image conflicts with them."""
                    action_geometry_label = (
                        "seated grooming pose, raised-paw contact and framing"
                    )
                pose_edit_intro = (
                    "Keep the frozen Main_case action geometry from "
                    "Image 1, but render the exact original uploaded pet identity from "
                    "Image 2. Image 1 owns only "
                    + action_geometry_label
                    + "; Image 2 alone owns identity, face, eyes, colors, markings, body "
                    "proportions, coat length and volume, outer fur silhouette, strand "
                    "direction, grooming, and tail appearance. The subject profile owns "
                    "no appearance trait."
                )
                style_source_rules = (
                    "Target style comes from this written instruction, not from either "
                    "image: restrained semi-realistic stylized 3D digital-double material with "
                    "color-faithful neutral high-key lighting and the uploaded pet's exact "
                    "natural coat surface. On recovery attempts the dedicated pet3d style model "
                    "may reinforce only the overall 3D material and lighting. Image 1 "
                    "never controls identity or color, and Image 2's photographic finish "
                    "never controls rendering style while its visible coat geometry and "
                    "strand direction remain identity traits."
                )
                character_lock_source = "the original uploaded pet in Image 2"
                identity_image_label = "Image 2"
                grooming_conflict_source = (
                    "Image 2's close-up crop, photographic finish, or model inference"
                )
                final_style_rule = (
                    "Render the required restrained identity-faithful stylized 3D material "
                    "while using only "
                    "the uploaded pet's source-matched natural coat surface; do not preserve "
                    "Image 2's photographic finish, invent coat features, or output a "
                    "grayscale guide."
                )
                main_case_coat_surface_rule = """
COAT IDENTITY LOCK — non-negotiable: preserve Image 2's visible natural coat length, volume, direction, edge behavior, genuine tufts or ruff, colors, and markings. Repose that coat around Image 1's joints without copying Image 1's tight outer outline. Remove only newly generated brush-bristle rays, repeated fan-shaped streaks, detached filament bundles, synthetic halo, or amplified volume absent from Image 2."""
            elif direct_uploaded_identity:
                pose_reference_rules = """
IMAGE ROLES — non-negotiable: Image 1 is the original uploaded pet photo and is the sole appearance source for identity, face shape, fur base colors, marking placement, eye size and color, ear shape, nose, muzzle, visible body proportions, natural coat length and volume, strand direction, outer fur silhouette, and tail identity. Ignore only its photographic rendering, pose, crop, camera angle, and background. Extend unseen appearance conservatively from visible source traits.

Image 2 is the frozen validated subject-profile action reference. It is authoritative only for joint placement, the complete full-body action pose, camera layout, framing, and action-required bowl placement. Its depicted animal is not the target identity: copy no fur color, marking, face identity, body proportion, coat length or volume, outer silhouette, tail appearance, accessory, or lighting hue from Image 2. Transfer the exact pet appearance from Image 1 into Image 2's validated action geometry. Follow the written action rules whenever either image conflicts with them."""
                pose_edit_intro = (
                    "Convert the exact pet from the original uploaded photo in Image 1 directly into "
                    "the validated full-body action geometry in Image 2. Preserve Image 1's identity, "
                    "face, colors, markings, body proportions, and natural coat while replacing only "
                    "its pose, crop, photographic rendering, and background."
                )
                style_source_rules = (
                    "Target rendering style — required by text: restrained semi-realistic "
                    "stylized 3D digital double, polished without juvenile exaggeration, with color-faithful "
                    "neutral high-key character lighting and source-matched natural fur. "
                    "Do not copy photographic rendering from Image 1 or character identity and "
                    "lighting from Image 2. Image 1 controls pet identity; Image 2 controls action "
                    "geometry only."
                )
                character_lock_source = "the original uploaded pet in Image 1"
                identity_image_label = "Image 1"
                grooming_conflict_source = (
                    "the uploaded photo, incomplete crop, or model inference"
                )
                final_style_rule = (
                    "Render the required restrained identity-faithful stylized 3D target style; "
                    "do not preserve Image 1's photographic style."
                )
            else:
                pose_reference_rules = """
IMAGE ROLES — non-negotiable: Image 1 is the approved canonical target and the only identity, face, fur-color, marking, body-proportion, natural coat-length, coat-volume, outer-silhouette, tail, and rendering-style reference. Image 2 is an appearance-free single-tone action silhouette derived from the digest-locked neutral-white anatomy-maquette driving video. Preserve from Image 2 only its joint placement, body pose, camera layout, and action-required prop placement. The source maquette's face, eyes, white surface, body material, and lighting were removed before Image 2 was built and have no target role. Every non-green Image 2 subject pixel has one identical synthetic gray value with zero fur-color, marking, material, texture, shading, lighting, face, eye, breed, or identity meaning. For grooming only, the thin pure-green open curved channel inside the gray silhouette is background negative space marking the torso-side boundary of the raised viewer-left foreleg: it is open at the paw and shoulder, must guide the organic limb, and must never be rendered as a line, marking, seam, hole, or object. Repose the exact character from Image 1 into Image 2's action layout. Never reconstruct a black, white, brown, gray, warm, cool, dark, or light fur region from Image 2. Copy no identity, body proportion, fur, coat surface, outer silhouette, markings, face, eyes, accessory, lighting color, or character design from Image 2. Follow the written action rules whenever either image conflicts with them."""
                pose_edit_intro = (
                    "Keep the exact approved canonical character from Image 1, but repose it using only the "
                    "action geometry and prop layout from Image 2. Preserve Image 1's identity, colors, "
                    "markings, proportions, and rendering style."
                )
        canonical_reference_first = bool(
            pose_reference_available
            and not direct_uploaded_identity
            and not main_case_geometry_first
        )
        if canonical_reference_first:
            compact_retry = ""
            if quality_retry >= 2:
                compact_retry = (
                    "RETRY CORRECTION: the previous candidate failed the action, "
                    "identity, framing, or matte gate. Follow Image 2's layout more "
                    "literally; do not return a studio portrait or neutral sitting pose."
                )
            if quality_retry >= 3:
                compact_retry += (
                    " Remove every floor, shadow, background gradient, costume, and "
                    "extra prop. Keep the complete action silhouette smaller than "
                    "Image 1 and inside the Image 2 bounding box."
                )
            if quality_retry >= 4 and action_key(pose) == "eating":
                compact_retry += (
                    " FINAL EATING CORRECTION: keep the bowl exactly in Image 2's "
                    "location. Put the lower muzzle inside and overlapping the visible "
                    "brown kibble; never omit the bowl, place the nose above it, or "
                    "return a sniffing pose."
                )
            if action_key(pose) == "eating":
                compact_geometry_role = (
                    "Image 2 is the texture-free action-layout guide derived directly "
                    "from the digest-locked neutral-white anatomy-maquette driving "
                    "video and is authoritative only "
                    "for joint placement, the exact pose, the action-required prop "
                    "layout, framing, subject scale, and pure-green background. Its "
                    "neutral subject geometry and semantic bowl colors are not an "
                    "identity or coat-appearance reference. Ignore the source "
                    "maquette's face, eyes, white surface, proportions, material, and "
                    "lighting completely."
                )
                compact_action_contract = (
                    "Copy Image 2's low eating joint geometry, subject bounding region, "
                    "head height, bowl position, front-paw placement, tail layout, and "
                    "camera exactly while preserving Image 1's body proportions and natural "
                    "coat outline around the reposed body. The lower muzzle is inside the brown kibble and is "
                    "partly occluded by the food and near rim. Keep exactly one shallow "
                    "light-beige bowl. Both complete front paws remain planted beside or "
                    "slightly touching the bowl; rear legs may be naturally tucked or "
                    "partly occluded by the crouched torso. Never raise the head to show "
                    "the full face."
                )
                compact_anatomy = (
                    "Use one continuous cat silhouette, two complete ears, two complete "
                    "front legs and paws, an anatomically connected crouched torso and "
                    "hindquarters, and one continuous tail. No extra, detached, fused, "
                    "or humanlike limb."
                )
            else:
                compact_geometry_role = (
                    "Image 2 is an appearance-free action-layout silhouette derived "
                    "directly from the digest-locked neutral-white anatomy-maquette "
                    "driving video and is "
                    "authoritative only for joint placement, the exact pose, framing, "
                    "subject scale, and pure-green background. Every non-green subject "
                    "pixel in Image 2 is the same artificial RGB(144,144,144), so it "
                    "carries no fur color, marking, material, texture, shading, or "
                    "lighting information. The source maquette's face, eyes, white "
                    "surface, body material, and proportions have no authority. Repose "
                    "the exact Image 1 character around "
                    "Image 2's joints without copying Image 2's body proportions, coat "
                    "surface, outer fur silhouette, or tail appearance. Never translate "
                    "the artificial gray into black, white, gray, brown, orange, warm, "
                    "cool, dark, or light fur regions. Do not redesign or reinterpret "
                    "the action layout. Its one thin pure-green open curved channel is "
                    "background negative space between the raised viewer-left foreleg "
                    "and chest. It is open at paw and shoulder: use it to read the "
                    "organic bent limb, but never render the channel as a colored line, "
                    "coat marking, seam, hole, prop, or hard object. Repaint the channel "
                    "completely with Image 1's local fur palette. It must leave no continuous "
                    "black or dark stroke from muzzle through paw, wrist, foreleg, or chest; "
                    "show anatomical separation only with soft low-contrast fur shading."
                )
                compact_action_contract = (
                    "Copy Image 2's seated grooming joint geometry, subject bounding region, "
                    "camera, raised foreleg, paw-to-mouth contact, planted front paw, "
                    "tucked hindquarters, and tail layout exactly while preserving Image 1's "
                    "body proportions and natural coat outline. Do not mirror the white-maquette "
                    "limb relationship: on the 512x512 output, only the foreleg on the "
                    "viewer's left rises from its visible shoulder near (38% width, 67% "
                    "height), bends through a readable organic elbow and wrist, and ends in "
                    "one rounded furry paw centered near (37% width, 51% height) touching the "
                    "viewer's-left underside of the muzzle. The viewer's-right front paw is "
                    "the single planted front paw. Image 2 contains one single-tone cat "
                    "silhouette, one open green torso-side negative-space curve, and no "
                    "prop: reconstruct the viewer-left shoulder-to-elbow-to-wrist anatomy "
                    "from that curve and this written coordinate contract "
                    "and render the raised paw with the exact same canonical fur palette and "
                    "soft paw material as Image 1's planted paw. A natural bent furry foreleg "
                    "in this region is mandatory and is not forbidden by the no-prop rule; "
                    "the no-prop rule forbids only a detached rigid item. Keep the limb visibly "
                    "attached to shoulder and chest, with an irregular soft-fur contour, never "
                    "a straight, hard, uniform-width cylinder or separate object."
                )
                if quality_retry >= 2:
                    compact_retry += (
                        " GROOMING SIDE RECOVERY: do not keep both front paws down and do "
                        "not raise the viewer's-right foreleg. Raise only the viewer's-left "
                        "foreleg from the shoulder coordinates above until its furry paw "
                        "physically touches the muzzle; preserve one viewer's-right planted "
                        "front paw. The organic limb is required, not an optional prop."
                    )
                if quality_retry >= 3:
                    compact_retry += (
                        " FINAL GROOMING GEOMETRY CORRECTION: the viewer-left raised limb "
                        "must be unmistakable at thumbnail size, visibly connected through "
                        "shoulder, elbow and wrist, and must not be mirrored, omitted, merged "
                        "into the torso, or replaced by a rigid object."
                    )
                compact_anatomy = (
                    "Use one continuous cat silhouette, two complete ears, one foreleg "
                    "connected from shoulder to the paw at the mouth, one planted front "
                    "paw, naturally tucked attached hind legs, and one continuous tail. "
                    "No extra, detached, fused, or humanlike limb."
                )
            compact_foreground = (
                "the cat and exactly one shallow food bowl"
                if action_key(pose) == "eating"
                else "the cat"
            )
            compact_fur_lock = profile_fur_lock.strip() or (
                "Preserve Image 1's exact natural coat length, visible volume, strand "
                "direction, genuine tufts or ruff, and source-bounded outer silhouette."
            )
            return f"""CRITICAL OUTPUT CONTRACT - satisfy this before every style detail:
Image 1 is the approved canonical target and the only identity, face, fur-color, marking, proportion, natural coat-length, coat-volume, outer-silhouette, tail, and rendering-style reference.
{compact_geometry_role}

{compact_action_contract}

OUTPUT RESTRICTIONS - all are mandatory: exactly one cat; no hat, clothes, collar, costume, accessory, text, logo, extra object, floor, floor line, reflection, cast shadow, contact shadow, scenery, or background variation. Every pixel outside {compact_foreground} must be flat uniform pure #00FF00 green. Keep at least 12% green clearance on all four edges and keep the complete foreground within 72% of canvas width and height.

Identity and style: copy Image 1's exact apparent age, face, eye size and color, ears, nose, fur colors, marking placement, body proportions, natural coat length and volume, strand direction, outer fur silhouette, tail identity, and restrained identity-faithful stylized 3D material. Preserve Image 1's light-versus-dark coat-area balance exactly: an unstriped region must remain unstriped, and no new dark band, stripe, outline, or connected marking may be invented from guide geometry. Style source: always inherit the approved canonical reference in Image 1. Image 2 and the white-maquette driver never control any appearance trait. The cat is completely unclothed.

Action contract: {first_frame_spec}
{first_frame_rules}
{compact_anatomy}

Coat contract: {compact_fur_lock}
Keep every source-authentic short, medium, or long coat trait from Image 1. Remove only newly generated rigid clumps, repeated radial brush rays, detached strands, synthetic halo, or amplified volume absent from Image 1; never shorten genuine long fur or suppress a genuine ruff or plume.

{compact_retry}
Return one clean static first frame only. No motion blur, duplicated limbs, motion symbols, text, logo, or watermark. {background_prompt}"""
        if action_key(pose) == "grooming":
            anatomy_checklist = (
                "Anatomy checklist — all items are mandatory: exactly one continuous cat "
                "silhouette; one foreleg rises continuously from its shoulder to one rounded "
                "paw touching the mouth; the other foreleg ends in one planted paw; both hind "
                "legs remain anatomically attached and may be naturally tucked in the seated "
                "pose. Show no duplicate ground paw, detached shape, extra limb, fused limb, "
                "human hand, or object at the mouth. Two complete ears and one continuous tail."
            )
        elif action_uses_bowl(pose):
            anatomy_checklist = (
                "Anatomy checklist — all items are mandatory: exactly one continuous cat "
                "silhouette; both complete front legs and paws remain planted beside the "
                "bowl; the crouched rear legs stay anatomically attached and may be "
                "naturally tucked or partly occluded by the torso; two complete ears; one "
                "tail visible continuously from base to tip. No limb may be fused, "
                "duplicated, detached, or reduced to a fragment."
            )
        else:
            anatomy_checklist = (
                "Anatomy checklist — all items are mandatory: exactly one continuous cat "
                "silhouette with anatomically attached legs and paws appropriate to the "
                "required pose, two complete ears, and one tail continuous from base to tip. "
                "No limb may be fused, duplicated, detached, hidden contrary to the action "
                "rules, or reduced to a fragment. No prop may appear."
            )
        if action_key(pose) == "eating":
            identity_visibility_rule = (
                "Keep both eyes anatomically coherent when visible, but the frozen "
                "eating angle may naturally foreshorten or partially hide them. Never "
                "raise the head to display the eyes, nose, mouth, or complete face; "
                "the upper head silhouette and markings carry identity while the lower "
                "muzzle remains inside the kibble."
            )
            final_identity_readability_rule = (
                "Keep the ears and upper-head identity markings readable while the "
                "nose and lower muzzle are visibly occluded by the food and near bowl "
                "rim. A fully unobstructed nose above the kibble is an invalid sniffing "
                "pose and must not be returned."
            )
        else:
            identity_visibility_rule = (
                "Both eyes must remain fully open, equally sized, level, and "
                "recognizable whenever the action angle allows."
            )
            final_identity_readability_rule = (
                "Keep the identity-bearing face readable while respecting the "
                "requested action angle."
            )
        return f"""Use case: identity-preserving pose edit for a stylized 3D character.
{pose_edit_intro}
{pose_reference_rules}

{style_source_rules} Do not reuse any user-authored pose, action, camera, framing, crop, background, or subject-placement instruction from the canonical design stage.

Character lock: copy the exact same head silhouette, cheek width, eye size, eye color and spacing, pupils, catchlights, ears, nose, closed muzzle, fur colors and marking placement, natural coat length and volume, body proportions, outer fur silhouette, and tail identity from {character_lock_source}. The validated profile controls action geometry only and never fills an appearance gap. Where the identity source is cropped or ambiguous, extend its nearest visible proportions, coat behavior, and dominant color conservatively without borrowing Main_case appearance. {identity_visibility_rule} The cat is completely unclothed: no glasses, collar, bow tie, shirt, overalls, suspenders, harness, hat, shoes, jewelry, costume, or accessory. Do not add markings, unrequested props, floating marks, motion symbols, a white border, or a halo. Where {identity_image_label} does not show an identity area, continue the nearest dominant body fur color conservatively with no new pattern.

Coat identity rule: {SLEEK_GROOMING_PROMPT} Treat {grooming_conflict_source} only as a possible source of generated artifacts; never suppress a genuine coat trait present in the uploaded identity.{profile_fur_section}
{main_case_coat_surface_rule}

Required still-frame pose only: {first_frame_spec}
First-frame action rules: {first_frame_rules}
{retry_instruction}
This is a clean static character keyframe that will be animated later. Do not illustrate the later motion in this image. Show only the balanced readable starting posture; do not add speed lines, duplicated limbs, motion blur, airborne action, or floating motion symbols. Do not blink, wink, or squint.
Only the "Required still-frame pose" and "First-frame action rules" control body posture, camera angle, framing, and action props. Every conflicting instruction is void.

{anatomy_checklist} Show the complete full-body cat with at least 12% green clearance on every canvas edge. Keep the subject at no more than 72% of canvas width and height. {final_identity_readability_rule} {final_style_rule} {background_prompt} Stable centered composition. {subject_contact_prompt} No floor, shadow, text, logo, or watermark."""

    return f"""Use the input image as the exact identity and appearance reference.
Create one polished first frame for a short image-to-video animation.

Creative direction: {base_prompt}
Required still-frame pose: {qwen_first_frame_spec(pose)}

Preserve the main subject's exact identity, face, distinctive colors and markings, hairstyle or fur, clothing, accessories, body proportions, and visual style. Keep the same number of main subjects as the input. Place the complete subject in the required pose, with anatomically correct limbs and the full body visible. {background_prompt} Use a stable centered composition, no text, and no watermark. The result must be a clean, coherent first frame suitable for animation."""


build_grooming_foreleg_repair_prompt = (
    _local_video_prompting_workflows.build_grooming_foreleg_repair_prompt
)


build_canonical_grooming_foreleg_completion_prompt = (
    _local_video_prompting_workflows.build_canonical_grooming_foreleg_completion_prompt
)


build_canonical_grooming_full_pose_rescue_prompt = (
    _local_video_prompting_workflows.build_canonical_grooming_full_pose_rescue_prompt
)


build_canonical_grooming_pose_transfer_prompt = (
    _local_video_prompting_workflows.build_canonical_grooming_pose_transfer_prompt
)


build_advisory_grooming_action_transfer_prompt = (
    _local_video_prompting_workflows.build_advisory_grooming_action_transfer_prompt
)


build_advisory_grooming_identity_refine_prompt = (
    _local_video_prompting_workflows.build_advisory_grooming_identity_refine_prompt
)


advisory_grooming_retry_anatomy_prompt = (
    _local_video_prompting_workflows.advisory_grooming_retry_anatomy_prompt
)


advisory_grooming_retry_requires_palette_correction = (
    _local_video_prompting_workflows.advisory_grooming_retry_requires_palette_correction
)


advisory_grooming_candidate_safe_for_masked_repair = (
    _local_video_prompting_workflows.advisory_grooming_candidate_safe_for_masked_repair
)


advisory_grooming_candidate_safe_for_identity_refine = (
    _local_video_prompting_workflows.advisory_grooming_candidate_safe_for_identity_refine
)


build_canonical_grooming_full_pose_identity_refine_prompt = (
    _local_video_prompting_workflows.build_canonical_grooming_full_pose_identity_refine_prompt
)


build_canonical_eating_identity_refine_prompt = (
    _local_video_prompting_workflows.build_canonical_eating_identity_refine_prompt
)


build_wan_prompt = _local_video_prompting_workflows.build_wan_prompt


is_fun_control_pose = _local_video_prompting_workflows.is_fun_control_pose


build_fun_control_prompt = _local_video_prompting_workflows.build_fun_control_prompt


FUN_CONTROL_NEGATIVE = (
    "deformed face, changing identity, changing fur pattern, extra legs, missing legs, three legs, five legs, "
    "duplicated paws, fused paws, merged limbs, broken joints, twisted ankles, impossible gait, human arms, "
    "human hands, bipedal motion, random paw waving, sliding feet, foot skating, floating paws, disconnected limbs, "
    "body morphing, size drift, camera shake, camera motion, zoom, crop, close-up, clothes, collar, glasses, harness, "
    "accessories, cluttered background, floor, horizon, shadows on background, text, watermark, blur, flicker, jitter, low quality"
)


qwen_recovery_profiles = _local_video_prompting_workflows.qwen_recovery_profiles


qwen_pose_profiles = _local_video_prompting_workflows.qwen_pose_profiles


pose_generation_profiles = _local_video_prompting_workflows.pose_generation_profiles


canonical_grooming_generation_profiles = (
    _local_video_prompting_workflows.canonical_grooming_generation_profiles
)


advisory_grooming_generation_profiles = (
    _local_video_prompting_workflows.advisory_grooming_generation_profiles
)


resolve_fur_surface_retry_decision = (
    _local_video_prompting_workflows.resolve_fur_surface_retry_decision
)


pose_recovery_profiles = _local_video_prompting_workflows.pose_recovery_profiles


qwen_workflow = _local_video_prompting_workflows.qwen_workflow


def wan_workflow(
    input_name: str,
    prompt: str,
    negative_prompt: str,
    seed: int,
    width: int,
    height: int,
    prefix: str,
    steps: int = WAN_STEPS,
    cfg: float = WAN_CFG,
    shift: float = WAN_SHIFT,
) -> dict[str, Any]:
    split_step = max(1, min(steps - 1, round(steps * WAN_SPLIT_STEP / WAN_STEPS)))
    if WAN_ENGINE == "wan2.2-i2v-a14b-fp8":
        return {
            "1": {
                "class_type": "UNETLoader",
                "inputs": {
                    "unet_name": WAN_HIGH_NOISE_MODEL,
                    "weight_dtype": "default",
                },
            },
            "2": {
                "class_type": "UNETLoader",
                "inputs": {
                    "unet_name": WAN_LOW_NOISE_MODEL,
                    "weight_dtype": "default",
                },
            },
            "3": {
                "class_type": "CLIPLoader",
                "inputs": {"clip_name": WAN_CLIP, "type": "wan", "device": "default"},
            },
            "4": {"class_type": "VAELoader", "inputs": {"vae_name": WAN_VAE}},
            "5": {"class_type": "LoadImage", "inputs": {"image": input_name}},
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": link("3"), "text": prompt},
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": link("3"), "text": negative_prompt},
            },
            "8": {
                "class_type": "WanImageToVideo",
                "inputs": {
                    "positive": link("6"),
                    "negative": link("7"),
                    "vae": link("4"),
                    "start_image": link("5"),
                    "width": width,
                    "height": height,
                    "length": FRAME_COUNT,
                    "batch_size": 1,
                },
            },
            "9": {
                "class_type": "ModelSamplingSD3",
                "inputs": {"model": link("1"), "shift": shift},
            },
            "10": {
                "class_type": "KSamplerAdvanced",
                "inputs": {
                    "model": link("9"),
                    "add_noise": "enable",
                    "noise_seed": seed,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "positive": link("8", 0),
                    "negative": link("8", 1),
                    "latent_image": link("8", 2),
                    "start_at_step": 0,
                    "end_at_step": split_step,
                    "return_with_leftover_noise": "enable",
                },
            },
            "11": {
                "class_type": "ModelSamplingSD3",
                "inputs": {"model": link("2"), "shift": shift},
            },
            "12": {
                "class_type": "KSamplerAdvanced",
                "inputs": {
                    "model": link("11"),
                    "add_noise": "disable",
                    "noise_seed": 0,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "positive": link("8", 0),
                    "negative": link("8", 1),
                    "latent_image": link("10"),
                    "start_at_step": split_step,
                    "end_at_step": steps,
                    "return_with_leftover_noise": "disable",
                },
            },
            "13": {
                "class_type": "VAEDecode",
                "inputs": {"samples": link("12"), "vae": link("4")},
            },
            "14": {
                "class_type": "CreateVideo",
                "inputs": {"images": link("13"), "fps": FPS},
            },
            "15": {
                "class_type": "SaveVideo",
                "inputs": {
                    "video": link("14"),
                    "filename_prefix": prefix,
                    "format": "auto",
                    "codec": "auto",
                },
            },
        }
    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": WAN_MODEL, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": WAN_CLIP, "type": "wan", "device": "default"},
        },
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": WAN_VAE}},
        "4": {"class_type": "LoadImage", "inputs": {"image": input_name}},
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": link("2"), "text": prompt},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": link("2"), "text": negative_prompt},
        },
        "7": {
            "class_type": "Wan22ImageToVideoLatent",
            "inputs": {
                "vae": link("3"),
                "start_image": link("4"),
                "width": width,
                "height": height,
                "length": FRAME_COUNT,
                "batch_size": 1,
            },
        },
        "8": {
            "class_type": "ModelSamplingSD3",
            "inputs": {"model": link("1"), "shift": shift},
        },
        "9": {
            "class_type": "KSampler",
            "inputs": {
                "model": link("8"),
                "positive": link("5"),
                "negative": link("6"),
                "latent_image": link("7"),
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "uni_pc",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        "10": {
            "class_type": "VAEDecode",
            "inputs": {"samples": link("9"), "vae": link("3")},
        },
        "11": {
            "class_type": "CreateVideo",
            "inputs": {"images": link("10"), "fps": FPS},
        },
        "12": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": link("11"),
                "filename_prefix": prefix,
                "format": "auto",
                "codec": "auto",
            },
        },
    }


wait_comfy_idle = _local_video_grooming_image_preparation.wait_comfy_idle


free_comfy_models = _local_video_grooming_image_preparation.free_comfy_models


queue_and_wait = _local_video_grooming_image_preparation.queue_and_wait


output_files = _local_video_grooming_image_preparation.output_files


first_output = _local_video_grooming_image_preparation.first_output


prepare_landscape_keyframe = _local_video_grooming_image_preparation.prepare_landscape_keyframe


def normalize_qwen_action_reference_rgba(
    source: Path,
    target: Path,
    *,
    size: tuple[int, int] = (TRANSPARENT_GIF_SIZE, TRANSPARENT_GIF_SIZE),
) -> dict[str, Any]:
    """Create the one final RGBA action reference used by every later gate.

    The reliable matte is never flattened onto green and reconstructed.  RGB is
    resized in premultiplied-alpha space while the already-approved soft alpha
    samples are downscaled with nearest-neighbour resampling.  This avoids a
    second chroma-key erosion while retaining the existing 512px framing.
    """

    source_digest = file_sha256(source)
    with Image.open(source) as loaded:
        source_mode = loaded.mode
        source_size = loaded.size
        source_has_alpha = "A" in loaded.getbands()
        image = loaded.convert("RGBA")
    if not source_has_alpha:
        raise ValueError("qwen_action_reference_source_has_no_alpha")

    alpha = image.getchannel("A")
    alpha_minimum, alpha_maximum = alpha.getextrema()
    if alpha_minimum == 255 or alpha_maximum == 0:
        raise ValueError("qwen_action_reference_source_alpha_is_not_usable")

    alpha_array = np.asarray(alpha, dtype=np.uint8)
    alpha_threshold = FurSurfaceThresholds().alpha_threshold
    foreground = np.where(alpha_array >= alpha_threshold, 255, 0).astype(np.uint8)
    foreground_image = Image.fromarray(foreground, mode="L")
    bbox = foreground_image.getbbox()
    foreground_pixels = int(np.count_nonzero(foreground))
    minimum_foreground_pixels = max(64, int(round(alpha_array.size * 0.002)))
    if bbox is None or foreground_pixels < minimum_foreground_pixels:
        raise ValueError("qwen_action_reference_foreground_is_not_usable")

    padding = max(
        16,
        int(max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * 0.06),
    )
    padded_bbox = (
        max(0, bbox[0] - padding),
        max(0, bbox[1] - padding),
        min(image.width, bbox[2] + padding),
        min(image.height, bbox[3] + padding),
    )
    cropped = image.crop(padded_bbox)
    maximum_width = max(1, int(size[0] * 0.88))
    maximum_height = max(1, int(size[1] * 0.90))
    scale = min(
        maximum_width / max(1, cropped.width),
        maximum_height / max(1, cropped.height),
        1.0,
    )
    resized_size = (
        max(1, int(round(cropped.width * scale))),
        max(1, int(round(cropped.height * scale))),
    )
    if resized_size != cropped.size:
        resized = (
            cropped.convert("RGBa")
            .resize(resized_size, Image.Resampling.LANCZOS)
            .convert("RGBA")
        )
        preserved_alpha = cropped.getchannel("A").resize(
            resized_size,
            Image.Resampling.NEAREST,
        )
        resized.putalpha(preserved_alpha)
    else:
        resized = cropped

    canvas = Image.new("RGBA", size, (0, 0, 0, 0))
    offset = ((size[0] - resized.width) // 2, (size[1] - resized.height) // 2)
    canvas.alpha_composite(resized, offset)
    output_alpha = canvas.getchannel("A")
    output_alpha_minimum, output_alpha_maximum = output_alpha.getextrema()
    output_corners = [
        output_alpha.getpixel((0, 0)),
        output_alpha.getpixel((size[0] - 1, 0)),
        output_alpha.getpixel((0, size[1] - 1)),
        output_alpha.getpixel((size[0] - 1, size[1] - 1)),
    ]
    if output_alpha_minimum == 255 or output_alpha_maximum == 0 or any(output_corners):
        raise ValueError("qwen_action_reference_output_alpha_is_not_usable")

    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, format="PNG", optimize=True)
    target_digest = file_sha256(target)
    return {
        "policy": QWEN_ACTION_REFERENCE_NORMALIZATION_POLICY,
        "source_path": str(source),
        "source_sha256": source_digest,
        "source_mode": source_mode,
        "source_size": list(source_size),
        "alpha_threshold": alpha_threshold,
        "foreground_pixels": foreground_pixels,
        "foreground_bbox": list(bbox),
        "padded_bbox": list(padded_bbox),
        "resized_size": list(resized_size),
        "output_path": str(target),
        "output_sha256": target_digest,
        "output_mode": "RGBA",
        "output_size": list(size),
        "alpha_resampling": "nearest_preserve_approved_matte_samples",
        "rgb_resampling": "premultiplied_alpha_lanczos",
        "output_alpha_extrema": [
            output_alpha_minimum,
            output_alpha_maximum,
        ],
        "output_alpha_corners": output_corners,
    }


composite_rgba_reference_on_background = (
    _local_video_grooming_image_preparation.composite_rgba_reference_on_background
)


def prepare_grooming_foreleg_repair_canvas(
    source: Path,
    geometry_guide: Path,
    canonical_reference: Path,
    target: Path,
    *,
    noise_mask_target: Path | None = None,
    mask_dilation_px: int = GROOMING_QWEN_FORELEG_REPAIR_MASK_DILATION_PX,
) -> dict[str, Any]:
    """Seed and context-mask a full-denoise identity-matched foreleg redraw."""

    if not 1 <= int(mask_dilation_px) <= 64:
        raise ValueError("grooming repair mask dilation must be between 1 and 64")
    with Image.open(source) as loaded:
        if "A" not in loaded.getbands():
            raise ValueError("grooming repair source has no alpha channel")
        source_rgba_image = loaded.convert("RGBA")
    with Image.open(geometry_guide) as loaded:
        guide_rgb_image = loaded.convert("RGB")
    with Image.open(canonical_reference) as loaded:
        if "A" not in loaded.getbands():
            raise ValueError("grooming repair canonical reference has no alpha")
        canonical_rgba_image = loaded.convert("RGBA")
    if source_rgba_image.size != (512, 512):
        raise ValueError("grooming repair source must be normalized to 512x512")
    if guide_rgb_image.size != source_rgba_image.size:
        raise ValueError("grooming repair geometry guide size does not match source")
    if canonical_rgba_image.size != source_rgba_image.size:
        raise ValueError("grooming repair canonical reference size does not match")

    source_rgba = np.asarray(source_rgba_image, dtype=np.uint8).copy()
    canonical_rgba = np.asarray(canonical_rgba_image, dtype=np.uint8)
    guide_rgb = np.asarray(guide_rgb_image, dtype=np.uint8)
    semantic_foreleg = np.all(
        guide_rgb == np.asarray(GROOMING_QWEN_GUIDE_FORELEG_RGB, dtype=np.uint8),
        axis=2,
    )
    semantic_foreleg_pixels = int(np.count_nonzero(semantic_foreleg))
    if not 650 <= semantic_foreleg_pixels <= 6000:
        raise RuntimeError(
            "Grooming repair geometry guide lacks its bounded foreleg semantic."
        )
    dilation = int(mask_dilation_px)
    kernel_size = dilation * 2 + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    edit_mask = cv2.dilate(
        semantic_foreleg.astype(np.uint8),
        kernel,
        iterations=1,
    ).astype(bool)
    edit_pixels = int(np.count_nonzero(edit_mask))
    source_foreground = source_rgba[:, :, 3] >= 64
    removed_foreground_pixels = int(np.count_nonzero(source_foreground & edit_mask))
    if removed_foreground_pixels < 256:
        raise RuntimeError(
            "Grooming repair edit window did not intersect enough candidate pixels."
        )

    original_composite = Image.new("RGBA", (512, 512), (0, 255, 0, 255))
    original_composite.alpha_composite(source_rgba_image)
    ys, xs = np.nonzero(edit_mask)
    edit_bbox = [
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    ]
    canonical_foreground = canonical_rgba[:, :, 3] >= 64
    canonical_ys, canonical_xs = np.nonzero(canonical_foreground)
    if canonical_xs.size == 0:
        raise RuntimeError(
            "Grooming repair canonical reference has no reliable foreground."
        )
    canonical_bbox = [
        int(canonical_xs.min()),
        int(canonical_ys.min()),
        int(canonical_xs.max()) + 1,
        int(canonical_ys.max()) + 1,
    ]
    canonical_width = canonical_bbox[2] - canonical_bbox[0]
    canonical_height = canonical_bbox[3] - canonical_bbox[1]
    seed_source_bbox = [
        int(round(canonical_bbox[0] + canonical_width * 0.18)),
        int(round(canonical_bbox[1] + canonical_height * 0.48)),
        int(round(canonical_bbox[0] + canonical_width * 0.52)),
        canonical_bbox[3],
    ]
    if (
        seed_source_bbox[2] - seed_source_bbox[0] < 24
        or seed_source_bbox[3] - seed_source_bbox[1] < 64
    ):
        raise RuntimeError(
            "Grooming repair canonical planted-foreleg crop is too small."
        )
    edit_width = edit_bbox[2] - edit_bbox[0]
    edit_height = edit_bbox[3] - edit_bbox[1]
    canonical_crop = canonical_rgba_image.crop(tuple(seed_source_bbox))
    canonical_crop_array = np.asarray(canonical_crop, dtype=np.uint8)
    crop_foreground_rgb = canonical_crop_array[:, :, :3][
        canonical_crop_array[:, :, 3] >= 64
    ]
    if crop_foreground_rgb.size == 0:
        raise RuntimeError(
            "Grooming repair canonical foreleg crop has no identity texture."
        )
    fallback_rgb = np.rint(
        np.median(crop_foreground_rgb.astype(np.float32), axis=0)
    ).astype(np.uint8)

    canonical_seed = canonical_crop.resize(
        (edit_width, edit_height),
        Image.Resampling.LANCZOS,
    ).rotate(180)
    canonical_seed_rgba_raw = np.asarray(canonical_seed, dtype=np.uint8)
    seed_alpha_float = canonical_seed_rgba_raw[:, :, 3].astype(np.float32) / 255.0
    seed_premultiplied = (
        canonical_seed_rgba_raw[:, :, :3].astype(np.float32)
        * seed_alpha_float[:, :, None]
    )
    lowpass_sigma = float(GROOMING_QWEN_FORELEG_REPAIR_SEED_LOWPASS_SIGMA_PX)
    blurred_alpha = cv2.GaussianBlur(
        seed_alpha_float,
        (0, 0),
        sigmaX=lowpass_sigma,
        sigmaY=lowpass_sigma,
    )
    blurred_premultiplied = cv2.GaussianBlur(
        seed_premultiplied,
        (0, 0),
        sigmaX=lowpass_sigma,
        sigmaY=lowpass_sigma,
    )
    blurred_seed_rgb = np.divide(
        blurred_premultiplied,
        np.maximum(blurred_alpha[:, :, None], 1e-3),
    )
    # Image 1 should communicate only the pet's bounded coat palette and
    # coarse shading. High-frequency planted-leg strands otherwise survive
    # img2img sampling as a pasted bristle patch instead of being rerendered.
    identity_median_blend = float(
        GROOMING_QWEN_FORELEG_REPAIR_SEED_IDENTITY_MEDIAN_BLEND
    )
    lowpass_seed_rgb = (
        np.rint(
            blurred_seed_rgb * (1.0 - identity_median_blend)
            + fallback_rgb.astype(np.float32)[None, None, :] * identity_median_blend
        )
        .clip(0, 255)
        .astype(np.uint8)
    )
    canonical_seed_rgba = canonical_seed_rgba_raw.copy()
    canonical_seed_rgba[:, :, :3] = lowpass_seed_rgb
    seed_foreground_local = canonical_seed_rgba_raw[:, :, 3] >= 64
    raw_seed_gray = cv2.cvtColor(
        canonical_seed_rgba_raw[:, :, :3],
        cv2.COLOR_RGB2GRAY,
    ).astype(np.float32)
    lowpass_seed_gray = cv2.cvtColor(
        lowpass_seed_rgb,
        cv2.COLOR_RGB2GRAY,
    ).astype(np.float32)
    raw_seed_high_frequency_energy = float(
        np.mean(np.abs(cv2.Laplacian(raw_seed_gray, cv2.CV_32F))[seed_foreground_local])
    )
    lowpass_seed_high_frequency_energy = float(
        np.mean(
            np.abs(cv2.Laplacian(lowpass_seed_gray, cv2.CV_32F))[seed_foreground_local]
        )
    )
    seed_canvas = np.zeros_like(canonical_rgba)
    seed_canvas[
        edit_bbox[1] : edit_bbox[3],
        edit_bbox[0] : edit_bbox[2],
    ] = canonical_seed_rgba
    seed_foreground = seed_canvas[:, :, 3] >= 64
    seed_direct_mask = edit_mask & seed_foreground
    seed_direct_pixels = int(np.count_nonzero(seed_direct_mask))
    if seed_direct_pixels < max(256, int(round(edit_pixels * 0.45))):
        raise RuntimeError(
            "Grooming repair canonical foreleg seed has insufficient coverage."
        )
    original_rgb = np.asarray(
        original_composite.convert("RGB"),
        dtype=np.uint8,
    )
    output_rgb = original_rgb.copy()
    output_rgb[seed_direct_mask] = seed_canvas[:, :, :3][seed_direct_mask]
    canonical_local_mask = edit_mask & ~seed_direct_mask & canonical_foreground
    output_rgb[canonical_local_mask] = canonical_rgba[:, :, :3][canonical_local_mask]
    median_fallback_mask = edit_mask & ~seed_direct_mask & ~canonical_local_mask
    output_rgb[median_fallback_mask] = fallback_rgb
    pure_green = np.asarray((0, 255, 0), dtype=np.uint8)
    edit_window_pure_green_pixels = int(
        np.count_nonzero(np.all(output_rgb[edit_mask] == pure_green, axis=1))
    )
    outside_edit_window_byte_preserved = bool(
        np.array_equal(output_rgb[~edit_mask], original_rgb[~edit_mask])
    )
    if edit_window_pure_green_pixels != 0 or not outside_edit_window_byte_preserved:
        raise RuntimeError(
            "Grooming repair canvas violated its canonical texture-seed contract."
        )

    if noise_mask_target is None:
        noise_mask_target = target.with_name(f"{target.stem}_noise_mask.png")
    noise_mask_expansion = int(GROOMING_QWEN_FORELEG_REPAIR_NOISE_MASK_EXPANSION_PX)
    noise_mask_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (noise_mask_expansion * 2 + 1, noise_mask_expansion * 2 + 1),
    )
    noise_mask_core = cv2.dilate(
        edit_mask.astype(np.uint8),
        noise_mask_kernel,
        iterations=1,
    ).astype(bool)
    noise_mask_sigma = float(GROOMING_QWEN_FORELEG_REPAIR_NOISE_MASK_FEATHER_SIGMA_PX)
    noise_mask = cv2.GaussianBlur(
        noise_mask_core.astype(np.float32),
        (0, 0),
        sigmaX=noise_mask_sigma,
        sigmaY=noise_mask_sigma,
    ).clip(0.0, 1.0)
    noise_mask[noise_mask < 0.01] = 0.0
    noise_mask[noise_mask > 0.99] = 1.0
    noise_mask_alpha = np.rint((1.0 - noise_mask) * 255.0).astype(np.uint8)
    noise_mask_rgba = np.zeros((512, 512, 4), dtype=np.uint8)
    noise_mask_rgba[:, :, 3] = noise_mask_alpha
    noise_mask_nonzero_pixels = int(np.count_nonzero(noise_mask > 0.0))
    noise_mask_full_strength_pixels = int(np.count_nonzero(noise_mask >= 1.0))
    noise_mask_feather_pixels = int(
        np.count_nonzero((noise_mask > 0.0) & (noise_mask < 1.0))
    )
    if noise_mask_full_strength_pixels < semantic_foreleg_pixels:
        raise RuntimeError(
            "Grooming repair noise mask lacks a full-denoise foreleg core."
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output_rgb, mode="RGB").save(
        target,
        format="PNG",
        optimize=True,
    )
    noise_mask_target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(noise_mask_rgba, mode="RGBA").save(
        noise_mask_target,
        format="PNG",
        optimize=True,
    )
    return {
        "policy": "canonical_planted_foreleg_band_limited_seed_v4",
        "source_path": str(source),
        "source_sha256": file_sha256(source),
        "geometry_guide_path": str(geometry_guide),
        "geometry_guide_sha256": file_sha256(geometry_guide),
        "canonical_reference_path": str(canonical_reference),
        "canonical_reference_sha256": file_sha256(canonical_reference),
        "canonical_foreground_bbox": canonical_bbox,
        "canonical_seed_source_bbox": seed_source_bbox,
        "canonical_seed_rotation_degrees": 180,
        "canonical_seed_lowpass_sigma_px": lowpass_sigma,
        "canonical_seed_identity_median_blend": identity_median_blend,
        "canonical_seed_high_frequency_texture_exposed": False,
        "canonical_seed_raw_high_frequency_energy": round(
            raw_seed_high_frequency_energy,
            8,
        ),
        "canonical_seed_lowpass_high_frequency_energy": round(
            lowpass_seed_high_frequency_energy,
            8,
        ),
        "canonical_seed_direct_pixels": seed_direct_pixels,
        "canonical_local_fallback_pixels": int(np.count_nonzero(canonical_local_mask)),
        "canonical_median_fallback_pixels": int(np.count_nonzero(median_fallback_mask)),
        "canonical_median_fallback_rgb": fallback_rgb.tolist(),
        "semantic_foreleg_rgb": list(GROOMING_QWEN_GUIDE_FORELEG_RGB),
        "semantic_foreleg_pixels": semantic_foreleg_pixels,
        "mask_dilation_px": dilation,
        "edit_mask_pixels": edit_pixels,
        "edit_mask_bbox": edit_bbox,
        "edit_mask_sha256": hashlib.sha256(edit_mask.tobytes()).hexdigest(),
        "removed_foreground_pixels": removed_foreground_pixels,
        "edit_window_pure_green_pixels": edit_window_pure_green_pixels,
        "edit_window_identity_texture_seeded": True,
        "outside_edit_window_byte_preserved": (outside_edit_window_byte_preserved),
        "noise_mask_policy": "expanded_context_feathered_full_denoise_v2",
        "noise_mask_core_expansion_px": noise_mask_expansion,
        "noise_mask_core_pixels": int(np.count_nonzero(noise_mask_core)),
        "noise_mask_feather_sigma_px": noise_mask_sigma,
        "noise_mask_nonzero_pixels": noise_mask_nonzero_pixels,
        "noise_mask_full_strength_pixels": noise_mask_full_strength_pixels,
        "noise_mask_feather_pixels": noise_mask_feather_pixels,
        "noise_mask_loadimage_contract": "mask_equals_one_minus_alpha",
        "noise_mask_target_path": str(noise_mask_target),
        "noise_mask_target_sha256": file_sha256(noise_mask_target),
        "target_path": str(target),
        "target_sha256": file_sha256(target),
    }


def prepare_canonical_grooming_foreleg_completion_canvas(
    source: Path,
    completion_mask: Path,
    canonical_reference: Path,
    target: Path,
    *,
    noise_mask_target: Path | None = None,
    mask_dilation_px: int = GROOMING_QWEN_FORELEG_REPAIR_MASK_DILATION_PX,
) -> dict[str, Any]:
    """Keep an identity-safe base intact and expose only a local noise mask."""

    if not 1 <= int(mask_dilation_px) <= 64:
        raise ValueError("grooming completion mask dilation must be between 1 and 64")
    with Image.open(source) as loaded:
        if "A" not in loaded.getbands():
            raise ValueError("grooming completion source has no alpha channel")
        source_rgba_image = loaded.convert("RGBA")
    with Image.open(completion_mask) as loaded:
        completion_rgb_image = loaded.convert("RGB")
    with Image.open(canonical_reference) as loaded:
        if "A" not in loaded.getbands():
            raise ValueError("grooming completion canonical reference has no alpha")
        canonical_rgba_image = loaded.convert("RGBA")
    if source_rgba_image.size != (512, 512):
        raise ValueError("grooming completion source must be normalized to 512x512")
    if completion_rgb_image.size != source_rgba_image.size:
        raise ValueError("grooming completion mask size does not match source")
    if canonical_rgba_image.size != source_rgba_image.size:
        raise ValueError("grooming completion canonical reference size does not match")

    source_rgba = np.asarray(source_rgba_image, dtype=np.uint8)
    completion_rgb = np.asarray(completion_rgb_image, dtype=np.uint8)
    semantic_foreleg = np.all(
        completion_rgb == np.asarray(GROOMING_QWEN_GUIDE_FORELEG_RGB, dtype=np.uint8),
        axis=2,
    )
    semantic_foreleg_pixels = int(np.count_nonzero(semantic_foreleg))
    if not 650 <= semantic_foreleg_pixels <= 6000:
        raise RuntimeError(
            "Grooming completion mask lacks its bounded foreleg semantic."
        )
    completion_background = np.all(
        completion_rgb == np.asarray((0, 255, 0), dtype=np.uint8),
        axis=2,
    )
    completion_foreground = ~completion_background
    allowed_foreground = (
        np.all(
            completion_rgb
            == np.asarray(GROOMING_QWEN_GUIDE_FOREGROUND_RGB, dtype=np.uint8),
            axis=2,
        )
        | semantic_foreleg
    )
    if not np.array_equal(completion_foreground, allowed_foreground):
        raise RuntimeError(
            "Grooming completion mask contains an unsupported semantic color."
        )

    dilation = int(mask_dilation_px)
    edit_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (dilation * 2 + 1, dilation * 2 + 1),
    )
    edit_mask = cv2.dilate(
        semantic_foreleg.astype(np.uint8),
        edit_kernel,
        iterations=1,
    ).astype(bool)
    edit_pixels = int(np.count_nonzero(edit_mask))
    source_foreground = source_rgba[:, :, 3] >= 64
    intersected_source_pixels = int(np.count_nonzero(source_foreground & edit_mask))
    if intersected_source_pixels < 256:
        raise RuntimeError(
            "Grooming completion edit window did not intersect enough candidate pixels."
        )
    ys, xs = np.nonzero(edit_mask)
    edit_bbox = [
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    ]

    original_composite = Image.new("RGBA", (512, 512), (0, 255, 0, 255))
    original_composite.alpha_composite(source_rgba_image)
    output_rgb = np.asarray(
        original_composite.convert("RGB"),
        dtype=np.uint8,
    ).copy()

    if noise_mask_target is None:
        noise_mask_target = target.with_name(f"{target.stem}_noise_mask.png")
    noise_mask_expansion = int(GROOMING_QWEN_FORELEG_REPAIR_NOISE_MASK_EXPANSION_PX)
    noise_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (noise_mask_expansion * 2 + 1, noise_mask_expansion * 2 + 1),
    )
    noise_mask_core = cv2.dilate(
        edit_mask.astype(np.uint8),
        noise_kernel,
        iterations=1,
    ).astype(bool)
    noise_mask_sigma = float(GROOMING_QWEN_FORELEG_REPAIR_NOISE_MASK_FEATHER_SIGMA_PX)
    noise_mask = cv2.GaussianBlur(
        noise_mask_core.astype(np.float32),
        (0, 0),
        sigmaX=noise_mask_sigma,
        sigmaY=noise_mask_sigma,
    ).clip(0.0, 1.0)
    noise_mask[noise_mask < 0.01] = 0.0
    noise_mask[noise_mask > 0.99] = 1.0
    noise_mask_alpha = np.rint((1.0 - noise_mask) * 255.0).astype(np.uint8)
    noise_mask_rgba = np.zeros((512, 512, 4), dtype=np.uint8)
    noise_mask_rgba[:, :, 3] = noise_mask_alpha
    noise_mask_nonzero_pixels = int(np.count_nonzero(noise_mask > 0.0))
    noise_mask_full_strength_pixels = int(np.count_nonzero(noise_mask >= 1.0))
    noise_mask_feather_pixels = int(
        np.count_nonzero((noise_mask > 0.0) & (noise_mask < 1.0))
    )
    if noise_mask_full_strength_pixels < semantic_foreleg_pixels:
        raise RuntimeError(
            "Grooming completion noise mask lacks a full-denoise foreleg core."
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output_rgb, mode="RGB").save(
        target,
        format="PNG",
        optimize=True,
    )
    noise_mask_target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(noise_mask_rgba, mode="RGBA").save(
        noise_mask_target,
        format="PNG",
        optimize=True,
    )
    with Image.open(target) as loaded:
        persisted_rgb = np.asarray(loaded.convert("RGB"), dtype=np.uint8)
    source_composite_preserved = bool(np.array_equal(persisted_rgb, output_rgb))
    if not source_composite_preserved:
        raise RuntimeError(
            "Grooming completion canvas changed the identity-safe source before sampling."
        )
    return {
        "policy": GROOMING_CANONICAL_FORELEG_COMPLETION_POLICY,
        "source_path": str(source),
        "source_sha256": file_sha256(source),
        "completion_mask_path": str(completion_mask),
        "completion_mask_sha256": file_sha256(completion_mask),
        "canonical_reference_path": str(canonical_reference),
        "canonical_reference_sha256": file_sha256(canonical_reference),
        "semantic_foreleg_rgb": list(GROOMING_QWEN_GUIDE_FORELEG_RGB),
        "semantic_foreleg_pixels": semantic_foreleg_pixels,
        "semantic_pixels_exposed_to_qwen": False,
        "mask_dilation_px": dilation,
        "edit_mask_pixels": edit_pixels,
        "edit_mask_bbox": edit_bbox,
        "edit_mask_sha256": hashlib.sha256(edit_mask.tobytes()).hexdigest(),
        "intersected_source_foreground_pixels": intersected_source_pixels,
        "identity_or_geometry_seed_inserted": False,
        "source_composite_preserved_before_masked_sampling": (
            source_composite_preserved
        ),
        "noise_mask_policy": "expanded_context_feathered_full_denoise_v2",
        "noise_mask_core_expansion_px": noise_mask_expansion,
        "noise_mask_core_pixels": int(np.count_nonzero(noise_mask_core)),
        "noise_mask_feather_sigma_px": noise_mask_sigma,
        "noise_mask_nonzero_pixels": noise_mask_nonzero_pixels,
        "noise_mask_full_strength_pixels": noise_mask_full_strength_pixels,
        "noise_mask_feather_pixels": noise_mask_feather_pixels,
        "noise_mask_loadimage_contract": "mask_equals_one_minus_alpha",
        "noise_mask_target_path": str(noise_mask_target),
        "noise_mask_target_sha256": file_sha256(noise_mask_target),
        "target_path": str(target),
        "target_sha256": file_sha256(target),
    }


prepare_canonical_action_identity_refine_canvas = (
    _local_video_grooming_image_preparation.prepare_canonical_action_identity_refine_canvas
)


prepare_advisory_grooming_identity_refine_canvas = (
    _local_video_grooming_image_preparation.prepare_advisory_grooming_identity_refine_canvas
)


finalize_advisory_grooming_identity_refine_candidate = (
    _local_video_grooming_image_preparation.finalize_advisory_grooming_identity_refine_candidate
)


prepare_canonical_grooming_full_pose_refine_canvas = (
    _local_video_grooming_image_preparation.prepare_canonical_grooming_full_pose_refine_canvas
)


_largest_green_or_alpha_foreground = (
    _local_video_grooming_image_preparation._largest_green_or_alpha_foreground
)


_identity_free_grooming_paw_depth = (
    _local_video_grooming_image_preparation._identity_free_grooming_paw_depth
)


prepare_grooming_white_maquette_frame0_pose_inputs = (
    _local_video_grooming_image_preparation.prepare_grooming_white_maquette_frame0_pose_inputs
)


prepare_canonical_grooming_face_identity_reference = (
    _local_video_grooming_image_preparation.prepare_canonical_grooming_face_identity_reference
)


prepare_advisory_grooming_local_action_edit_mask = (
    _local_video_grooming_image_preparation.prepare_advisory_grooming_local_action_edit_mask
)


_canonical_aligned_grooming_foreleg_geometry = (
    _local_video_grooming_image_preparation._canonical_aligned_grooming_foreleg_geometry
)


prepare_canonical_grooming_pose_transfer_seed = (
    _local_video_grooming_image_preparation.prepare_canonical_grooming_pose_transfer_seed
)


prepare_canonical_grooming_seed_bound_matte = (
    _local_video_grooming_image_preparation.prepare_canonical_grooming_seed_bound_matte
)


prepare_canonical_eating_identity_refine_canvas = (
    _local_video_grooming_image_preparation.prepare_canonical_eating_identity_refine_canvas
)


prepare_grooming_foreleg_repair_anatomy_guide = (
    _local_video_grooming_image_preparation.prepare_grooming_foreleg_repair_anatomy_guide
)


prepare_grooming_full_pose_recovery_guide = (
    _local_video_grooming_image_preparation.prepare_grooming_full_pose_recovery_guide
)


require_ffmpeg = _local_video_identity_quality.require_ffmpeg


run_ffmpeg = _local_video_identity_quality.run_ffmpeg


convert_chroma_image_to_transparent_png = (
    _local_video_identity_quality.convert_chroma_image_to_transparent_png
)


canonical_cutout_interior_quality = (
    _local_video_identity_quality.canonical_cutout_interior_quality
)


evaluate_chroma_cutout = _local_video_identity_quality.evaluate_chroma_cutout


get_rembg_session = _local_video_identity_quality.get_rembg_session


IDENTITY_PALETTE_CATEGORIES = (
    "black",
    "white",
    "gray",
    "warm_brown_or_orange",
    "red_or_pink",
    "yellow_or_gold",
    "green",
    "blue_or_cyan",
    "purple",
)
IDENTITY_PALETTE_PROMPT_LABELS = {
    "black": "black",
    "white": "white",
    "gray": "neutral gray",
    "warm_brown_or_orange": "brown/orange/ginger/tan",
    "red_or_pink": "red/pink",
    "yellow_or_gold": "yellow/gold",
    "green": "green",
    "blue_or_cyan": "blue/cyan",
    "purple": "purple",
}
IDENTITY_PALETTE_CHROMATIC = (
    "warm_brown_or_orange",
    "red_or_pink",
    "yellow_or_gold",
    "green",
    "blue_or_cyan",
    "purple",
)


identity_palette_signature = _local_video_identity_quality.identity_palette_signature


neutralize_uploaded_identity_chroma_spill = (
    _local_video_identity_quality.neutralize_uploaded_identity_chroma_spill
)


identity_palette_is_achromatic = _local_video_identity_quality.identity_palette_is_achromatic


identity_palette_prompt = _local_video_identity_quality.identity_palette_prompt


identity_palette_negative_prompt = (
    _local_video_identity_quality.identity_palette_negative_prompt
)


identity_palette_quality = _local_video_identity_quality.identity_palette_quality


canonical_action_identity_palette_quality = (
    _local_video_identity_quality.canonical_action_identity_palette_quality
)


advisory_grooming_minimum_palette_quality = (
    _local_video_identity_quality.advisory_grooming_minimum_palette_quality
)


grooming_face_identity_signature = (
    _local_video_identity_quality.grooming_face_identity_signature
)


advisory_grooming_face_identity_quality = (
    _local_video_identity_quality.advisory_grooming_face_identity_quality
)


advisory_grooming_minimum_pose_quality = (
    _local_video_identity_quality.advisory_grooming_minimum_pose_quality
)


advisory_grooming_action_retry_guidance = (
    _local_video_identity_quality.advisory_grooming_action_retry_guidance
)


advisory_grooming_canonical_action_edit_quality = (
    _local_video_identity_quality.advisory_grooming_canonical_action_edit_quality
)


evaluate_advisory_grooming_minimum_gate = (
    _local_video_identity_quality.evaluate_advisory_grooming_minimum_gate
)


advisory_grooming_degraded_candidate_selection_key = (
    _local_video_identity_quality.advisory_grooming_degraded_candidate_selection_key
)


apply_canonical_grooming_organic_paw_palette_evidence = (
    _local_video_identity_quality.apply_canonical_grooming_organic_paw_palette_evidence
)


canonical_grooming_palette_normalization_eligibility = (
    _local_video_identity_quality.canonical_grooming_palette_normalization_eligibility
)


canonical_grooming_pose_safe_identity_refine_eligibility = (
    _local_video_identity_quality.canonical_grooming_pose_safe_identity_refine_eligibility
)


prepare_canonical_grooming_achromatic_palette_normalization = (
    _local_video_identity_quality.prepare_canonical_grooming_achromatic_palette_normalization
)


bowl_aware_identity_palette_quality = (
    _local_video_identity_quality.bowl_aware_identity_palette_quality
)


canonical_identity_palette_quality = (
    _local_video_identity_quality.canonical_identity_palette_quality
)


evaluate_pose_identity_palette = _local_video_identity_quality.evaluate_pose_identity_palette


identity_palette_audit = _local_video_identity_quality.identity_palette_audit


skipped_visual_audit = _local_video_identity_quality.skipped_visual_audit


canonical_fallback_score = _local_video_identity_quality.canonical_fallback_score


select_canonical_fallback_attempt = (
    _local_video_identity_quality.select_canonical_fallback_attempt
)


SEMANTIC_RECOVERABLE_FRAMING_ISSUES = frozenset(
    {
        "subject_too_close_to_canvas_edge",
        "subject_too_large",
    }
)


semantic_quality_blocking_issues = (
    _local_video_identity_quality.semantic_quality_blocking_issues
)


semantic_recovery_gate_failure = _local_video_identity_quality.semantic_recovery_gate_failure


neutralize_achromatic_warm_cast = _local_video_identity_quality.neutralize_achromatic_warm_cast


semantic_background_recovery = _local_video_identity_quality.semantic_background_recovery


finalize_qwen_action_reference_candidate = (
    _local_video_generation_quality.finalize_qwen_action_reference_candidate
)


normalize_qwen_action_candidate = (
    _local_video_generation_quality.normalize_qwen_action_candidate
)


fur_surface_gate_failure = _local_video_generation_quality.fur_surface_gate_failure


resolve_main_case_fur_surface_baseline = (
    _local_video_generation_quality.resolve_main_case_fur_surface_baseline
)


evaluate_main_case_fur_surface_candidate = (
    _local_video_generation_quality.evaluate_main_case_fur_surface_candidate
)


apply_canonical_grooming_direct_surface_nonexpansion_gate = (
    _local_video_generation_quality.apply_canonical_grooming_direct_surface_nonexpansion_gate
)


select_final_video_fur_surface_comparison_scope = (
    _local_video_generation_quality.select_final_video_fur_surface_comparison_scope
)


def evaluate_final_video_fur_surface(
    video_path: Path,
    *,
    action: str,
    effective_subject_config: dict[str, Any],
    comparison_scope: str = FINAL_VIDEO_PROFILE_FULL_SCOPE,
    validated_action_reference_path: Path | None = None,
    validated_action_reference_sha256: str | None = None,
    validated_action_reference_gate: dict[str, Any] | None = None,
    reference_policy: str | None = None,
    pose_identity_mask_video_path: Path | None = None,
    pose_identity_mask_video_sha256: str | None = None,
    reference_identity_mask_path: Path | None = None,
    reference_identity_mask_sha256: str | None = None,
    grooming_pose_gate_composite_recovery: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit MP4 samples against the exact accepted uploaded action reference."""

    try:
        baseline_path, baseline = resolve_main_case_fur_surface_baseline(
            effective_subject_config,
            action,
        )
        action_config = effective_subject_config.get("actions", {}).get(action)
        if not isinstance(action_config, dict):
            raise RuntimeError(f"Final fur audit lacks {action} action config")
        reference_provenance: dict[str, Any] | None = None
        if comparison_scope == FINAL_VIDEO_VALIDATED_ACTION_SURFACE_SCOPE:
            expected_reference_policy = (
                FINAL_VIDEO_VALIDATED_ACTION_REFERENCE_POLICIES.get(action)
            )
            if expected_reference_policy is None:
                raise RuntimeError(
                    "Validated-action final surface v2 does not support "
                    f"action {action!r}"
                )
            if action_config.get("reference_kind") != "qwen_generated_upload":
                raise RuntimeError(
                    "Validated-action final surface v2 requires a Qwen upload reference"
                )
            if reference_policy != expected_reference_policy:
                raise RuntimeError(
                    "Validated-action final surface v2 reference policy changed "
                    f"for {action}: expected {expected_reference_policy!r}, "
                    f"got {reference_policy!r}"
                )
            if validated_action_reference_path is None:
                raise RuntimeError("Validated Qwen action reference path is missing")
            configured_reference_path = Path(str(action_config.get("reference") or ""))
            configured_reference_sha = (
                str(action_config.get("reference_sha256") or "").strip().lower()
            )
            expected_reference_sha = (
                str(validated_action_reference_sha256 or "").strip().lower()
            )
            gate = (
                validated_action_reference_gate.get("gate")
                if isinstance(validated_action_reference_gate, dict)
                else None
            )
            gate_sha = (
                str(gate.get("candidate_sha256") or "").strip().lower()
                if isinstance(gate, dict)
                else ""
            )
            selected_sha = (
                str(
                    validated_action_reference_gate.get("selected_candidate_sha256")
                    or ""
                )
                .strip()
                .lower()
                if isinstance(validated_action_reference_gate, dict)
                else ""
            )
            actual_reference_sha = (
                file_sha256(validated_action_reference_path)
                if validated_action_reference_path.is_file()
                else ""
            )
            if (
                configured_reference_path != validated_action_reference_path
                or not re.fullmatch(r"[0-9a-f]{64}", expected_reference_sha)
                or not (
                    configured_reference_sha
                    == expected_reference_sha
                    == actual_reference_sha
                    == gate_sha
                    == selected_sha
                )
            ):
                raise RuntimeError(
                    "Validated Qwen action reference provenance changed after SCAIL"
                )
            reference_provenance = {
                "reference_kind": action_config.get("reference_kind"),
                "reference_policy": reference_policy,
                "configured_reference": str(configured_reference_path),
                "configured_reference_sha256": configured_reference_sha,
                "selected_candidate_sha256": selected_sha,
                "gate_candidate_sha256": gate_sha,
                "pre_scail_revalidated": True,
                "gate": json.loads(
                    json.dumps(
                        validated_action_reference_gate,
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                ),
            }
            if grooming_pose_gate_composite_recovery is not None:
                if action != "grooming" or not isinstance(
                    grooming_pose_gate_composite_recovery,
                    dict,
                ):
                    raise RuntimeError(
                        "Grooming pose composite recovery provenance has an "
                        "invalid action scope or schema."
                    )
                reference_provenance["grooming_pose_gate_composite_recovery"] = (
                    json.loads(
                        json.dumps(
                            grooming_pose_gate_composite_recovery,
                            ensure_ascii=False,
                            allow_nan=False,
                        )
                    )
                )
        result = audit_final_video(
            video_path,
            action=action,
            baseline_path=baseline_path,
            baseline_sha256=str(baseline["snapshot_sha256"]),
            baseline_provenance=json.loads(json.dumps(baseline["provenance"])),
            sample_count=FINAL_VIDEO_FUR_SAMPLE_COUNT,
            rembg_model=CHROMA_FALLBACK_REMBG_MODEL,
            comparison_scope=comparison_scope,
            validated_action_reference_path=validated_action_reference_path,
            validated_action_reference_sha256=(validated_action_reference_sha256),
            validated_action_reference_provenance=reference_provenance,
            pose_identity_mask_video_path=pose_identity_mask_video_path,
            pose_identity_mask_video_sha256=pose_identity_mask_video_sha256,
            reference_identity_mask_path=reference_identity_mask_path,
            reference_identity_mask_sha256=reference_identity_mask_sha256,
        )
        result = json.loads(json.dumps(result, ensure_ascii=False, allow_nan=False))
    except Exception as exc:  # noqa: BLE001 - final output must fail closed
        return {
            "passed": False,
            "verdict": "audit_error_fail_closed",
            "action": action,
            "video": {
                "path": str(video_path),
                "sha256": file_sha256(video_path) if video_path.is_file() else None,
            },
            "error": f"{type(exc).__name__}: {str(exc)[:1000]}",
            "policy": {
                "name": FINAL_VIDEO_FUR_AUDIT_POLICY,
                "version": FINAL_VIDEO_FUR_AUDIT_VERSION,
                "sample_count": FINAL_VIDEO_FUR_SAMPLE_COUNT,
                "comparison_scope": comparison_scope,
                "fail_closed": True,
            },
        }
    result["production_integrated"] = True
    return result


grooming_pose_gate_failure = _local_video_generation_quality.grooming_pose_gate_failure


apply_masked_pet3d_grooming_composite_gate = (
    _local_video_generation_quality.apply_masked_pet3d_grooming_composite_gate
)


apply_canonical_grooming_direct_structural_edge_evidence_gate = (
    _local_video_generation_quality.apply_canonical_grooming_direct_structural_edge_evidence_gate
)


apply_canonical_grooming_completion_edge_evidence_gate = (
    _local_video_generation_quality.apply_canonical_grooming_completion_edge_evidence_gate
)


resolve_main_case_grooming_pose_gate_metadata = (
    _local_video_generation_quality.resolve_main_case_grooming_pose_gate_metadata
)


evaluate_main_case_grooming_pose_candidate = (
    _local_video_generation_quality.evaluate_main_case_grooming_pose_candidate
)


semantic_recovery_candidate_eligible = (
    _local_video_generation_quality.semantic_recovery_candidate_eligible
)


pose_attempt_failure_issues = _local_video_generation_quality.pose_attempt_failure_issues


pose_attempt_failure_summary = _local_video_generation_quality.pose_attempt_failure_summary


eating_mouth_bowl_contact_quality = (
    _local_video_generation_quality.eating_mouth_bowl_contact_quality
)


pose_cutout_quality = _local_video_generation_quality.pose_cutout_quality


fur_silhouette_signature = _local_video_generation_quality.fur_silhouette_signature


eating_fur_silhouette_quality = _local_video_generation_quality.eating_fur_silhouette_quality


pose_similarity_signature = _local_video_generation_quality.pose_similarity_signature


compare_pose_signatures = _local_video_generation_quality.compare_pose_signatures


pose_similarity_quality = _local_video_generation_quality.pose_similarity_quality


def convert_chroma_video_to_transparent_animations(
    source: Path,
    gif_target: Path,
    apng_target: Path,
) -> dict[str, Any]:
    return build_transparent_animations(
        source,
        gif_target,
        apng_target,
        ffmpeg_bin=FFMPEG_BIN,
        size=TRANSPARENT_GIF_SIZE,
        fps=TRANSPARENT_GIF_FPS,
        frame_count=TRANSPARENT_GIF_FRAMES,
        duration_seconds=TRANSPARENT_GIF_DURATION_SECONDS,
    )


def run_fun_control_video(
    *,
    job_dir: Path,
    pose_index: int,
    reference_image: Path,
    prompt: str,
    negative_prompt: str,
    seed: int,
) -> tuple[Path, dict[str, Any]]:
    run_id = uuid4().hex
    output_dir = job_dir / "fun_control" / f"pose_{pose_index:02d}" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    log_path = job_dir / f"pose_{pose_index:02d}_fun_control.log"
    metadata_path = job_dir / "workflows" / f"fun_pose_{pose_index:02d}.json"
    metadata = {
        "engine": "wan2.2-fun-5b-control",
        "model_root": str(FUN_CONTROL_MODEL_ROOT),
        "control_video": str(FUN_CONTROL_RUNNING_VIDEO),
        "reference_image": reference_image.name,
        "resolution": [512, 512],
        "frames": FRAME_COUNT,
        "fps": FPS,
        "steps": FUN_CONTROL_STEPS,
        "guidance_scale": 6.0,
        "seed": seed,
        "gpu_memory_mode": FUN_CONTROL_GPU_MEMORY_MODE,
        "teacache": False,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
    }
    write_json_atomic(metadata_path, metadata)

    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH", "").strip()
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(FUN_CONTROL_PROJECT_ROOT), existing_pythonpath) if part
    )
    environment.update(
        {
            "PYTHONUNBUFFERED": "1",
            "VIDEO_LENGTH": str(FRAME_COUNT),
            "INFERENCE_STEPS": str(FUN_CONTROL_STEPS),
            "SAVE_PATH": str(output_dir),
            "GPU_MEMORY_MODE": FUN_CONTROL_GPU_MEMORY_MODE,
            "ENABLE_TEACACHE": "0",
            "CONTROL_VIDEO": str(FUN_CONTROL_RUNNING_VIDEO),
            "REF_IMAGE": str(reference_image),
            "PROMPT": prompt,
            "NEGATIVE_PROMPT": negative_prompt,
            "GUIDANCE_SCALE": "6.0",
            "SEED": str(seed),
        }
    )
    started = time.monotonic()
    process = subprocess.run(
        [str(FUN_CONTROL_PYTHON), str(FUN_CONTROL_SCRIPT)],
        cwd=str(FUN_CONTROL_PROJECT_ROOT),
        env=environment,
        capture_output=True,
        text=True,
        timeout=FUN_CONTROL_TIMEOUT,
        check=False,
    )
    elapsed = round(time.monotonic() - started, 3)
    log_path.write_text(
        (process.stdout or "") + "\n--- STDERR ---\n" + (process.stderr or ""),
        encoding="utf-8",
    )
    if process.returncode != 0:
        detail = (
            process.stderr or process.stdout or "unknown VideoX-Fun error"
        ).strip()
        raise RuntimeError(
            f"Wan2.2-Fun-5B-Control exited with {process.returncode}: {detail[-4000:]}"
        )
    videos = sorted(output_dir.glob("*.mp4"), key=lambda path: path.stat().st_mtime)
    if not videos:
        raise FileNotFoundError("Wan2.2-Fun-5B-Control did not produce an MP4 output.")
    metadata.update({"elapsed_seconds": elapsed, "log": log_path.name})
    write_json_atomic(metadata_path, metadata)
    return videos[-1], metadata


def write_workflow(job_dir: Path, filename: str, workflow: dict[str, Any]) -> None:
    write_json_atomic(job_dir / "workflows" / filename, workflow)


def prepare_pose_input(
    job_id: str,
    job_dir: Path,
    pose_index: int,
    pose: str,
    keyframe: Path,
    width: int,
    height: int,
    transparent: bool,
    preserve_gated_alpha: bool = False,
) -> Path:
    wan_input = job_dir / f"pose_{pose_index:02d}_wan_input.png"
    transparent_keyframe: Path | None = None
    reference_digest: str | None = None
    if transparent and preserve_gated_alpha:
        if (width, height) != (TRANSPARENT_GIF_SIZE, TRANSPARENT_GIF_SIZE):
            raise RuntimeError(
                "Gated Qwen action reference target is not the required 512x512."
            )
        with Image.open(keyframe) as loaded:
            if loaded.size != (width, height) or "A" not in loaded.getbands():
                raise RuntimeError(
                    "Gated Qwen action reference is not a 512x512 RGBA image."
                )
            alpha = loaded.convert("RGBA").getchannel("A")
            alpha_minimum, alpha_maximum = alpha.getextrema()
            if alpha_minimum == 255 or alpha_maximum == 0:
                raise RuntimeError(
                    "Gated Qwen action reference lacks a usable alpha matte."
                )
        transparent_keyframe = (
            job_dir / f"pose_{pose_index:02d}_keyframe_transparent.png"
        )
        shutil.copy2(keyframe, transparent_keyframe)
        reference_digest = file_sha256(keyframe)
        if file_sha256(transparent_keyframe) != reference_digest:
            raise RuntimeError(
                "Gated Qwen action reference changed during byte-preserving publish."
            )
        composite_rgba_reference_on_background(
            transparent_keyframe,
            wan_input,
            (0, 255, 0),
        )
    else:
        background_rgb = (0, 255, 0) if transparent else (255, 255, 255)
        prepare_landscape_keyframe(
            keyframe, wan_input, (width, height), background_rgb=background_rgb
        )
    if transparent:
        if transparent_keyframe is None:
            transparent_keyframe = (
                job_dir / f"pose_{pose_index:02d}_keyframe_transparent.png"
            )
            convert_chroma_image_to_transparent_png(
                wan_input,
                transparent_keyframe,
                require_clean_background=True,
            )
            reference_digest = file_sha256(transparent_keyframe)
        append_output(
            job_id,
            {
                "type": "transparent_keyframe",
                "pose_index": pose_index,
                "pose": pose,
                "action": (
                    action_key(pose)
                    if action_key(pose) in SUBJECT_REPLACEMENT_ACTIONS
                    else None
                ),
                "action_label": SUBJECT_REPLACEMENT_ACTION_LABELS.get(action_key(pose)),
                "filename": transparent_keyframe.name,
                "width": TRANSPARENT_GIF_SIZE,
                "height": TRANSPARENT_GIF_SIZE,
                "transparent": True,
                "visual_style": "identity_preserving_cute_3d_studio",
                "source_role": "cartoon_animation_keyframe",
                "normalization_policy": (
                    QWEN_ACTION_REFERENCE_NORMALIZATION_POLICY
                    if preserve_gated_alpha
                    else "landscape_green_then_chroma_key_legacy"
                ),
                "byte_preserved_from_gated_candidate": preserve_gated_alpha,
                "sha256": reference_digest,
            },
        )
    return wan_input


def run_cancelable_process(
    job_id: str,
    command: list[str],
    *,
    log_path: Path,
    stage_name: str,
    timeout: int,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        while process.poll() is None:
            if cancel_requested(job_id):
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
                raise JobCancelled("Generation was cancelled by the user.")
            if time.monotonic() - started > timeout:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
                raise TimeoutError(f"{stage_name} timed out after {timeout} seconds.")
            time.sleep(1)
    if process.returncode:
        detail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"{stage_name} failed: {detail.strip()}")


def logged_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    start = text.find("{")
    if start < 0:
        raise RuntimeError(f"No JSON result was written to {path}")
    try:
        value = json.loads(text[start:])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON result in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Unexpected JSON result in {path}")
    return value


available_scail_run_id = _local_video_geometry_guides.available_scail_run_id


prepare_eating_qwen_geometry_guide = (
    _local_video_geometry_guides.prepare_eating_qwen_geometry_guide
)


_prepare_grooming_qwen_geometry_guide_legacy_two_tone = (
    _local_video_geometry_guides._prepare_grooming_qwen_geometry_guide_legacy_two_tone
)


build_grooming_open_foreleg_negative_space = (
    _local_video_geometry_guides.build_grooming_open_foreleg_negative_space
)


prepare_grooming_qwen_geometry_guide = (
    _local_video_geometry_guides.prepare_grooming_qwen_geometry_guide
)


persist_uploaded_identity_views = (
    _local_video_uploaded_identity_audit.persist_uploaded_identity_views
)


verify_uploaded_identity_submission = (
    _local_video_uploaded_identity_audit.verify_uploaded_identity_submission
)


aggregate_identity_palette_signatures = (
    _local_video_uploaded_identity_audit.aggregate_identity_palette_signatures
)


canonical_upper_body_palette_reference = (
    _local_video_uploaded_identity_audit.canonical_upper_body_palette_reference
)


compose_uploaded_identity_view_audit = (
    _local_video_uploaded_identity_audit.compose_uploaded_identity_view_audit
)


audit_uploaded_identity_view_set = (
    _local_video_uploaded_identity_audit.audit_uploaded_identity_view_set
)


prepare_uploaded_subject_qwen_references = (
    _local_video_uploaded_references.prepare_uploaded_subject_qwen_references
)


prepare_uploaded_subject_scail_references = (
    _local_video_uploaded_references.prepare_uploaded_subject_scail_references
)


def uses_frozen_profile_scail_reference(
    effective_subject_config: dict[str, Any],
    action: str,
) -> bool:
    return (
        action == "eating"
        and effective_subject_config.get("profile_id") == "sleek_tabby_cat_v1"
    )


def partition_subject_reference_actions(
    effective_subject_config: dict[str, Any],
    actions: list[str],
    *,
    uploaded_identity: bool = False,
) -> tuple[list[str], list[str]]:
    if uploaded_identity:
        return [], list(actions)
    frozen_actions = [
        action
        for action in actions
        if uses_frozen_profile_scail_reference(
            effective_subject_config,
            action,
        )
    ]
    qwen_actions = [action for action in actions if action not in frozen_actions]
    return frozen_actions, qwen_actions


def select_scail_subject_reference(
    job: dict[str, Any],
    profile: dict[str, Any] | None,
    generated_reference_paths: dict[str, str] | None,
    action: str,
) -> tuple[Path, str]:
    """Choose a frozen action reference without importing unsafe fur geometry."""
    if generated_reference_paths is not None:
        generated_reference = str(generated_reference_paths.get(action) or "").strip()
        if not generated_reference:
            raise RuntimeError(
                f"Uploaded identity stage did not produce the required {action} reference."
            )
        direct_upload = job.get("subject_source") == "uploaded_identity_direct_scail"
        return (
            Path(generated_reference),
            (
                (
                    "uploaded_identity_with_main_case_bowl_direct_scail_v1"
                    if action == "eating"
                    else "uploaded_identity_direct_scail_v1"
                )
                if direct_upload
                else (
                    "qwen_action_identity_with_bowl_mapped_red"
                    if action == "eating"
                    else "qwen_action_specific_identity"
                )
            ),
        )

    effective_subject_config = job.get("effective_subject_config")
    if isinstance(
        effective_subject_config, dict
    ) and uses_frozen_profile_scail_reference(
        effective_subject_config,
        action,
    ):
        action_config = effective_subject_config["actions"][action]
        reference_text = str(action_config.get("profile_reference") or "").strip()
        expected_sha256 = str(
            action_config.get("profile_reference_sha256") or ""
        ).strip()
        if not reference_text:
            if action_config.get("reference_kind") == "qwen_generated_upload":
                raise RuntimeError(
                    "Frozen sleek eating profile reference is missing; "
                    "refusing to reuse the Qwen eating keyframe."
                )
            reference_text = str(action_config.get("reference") or "").strip()
            expected_sha256 = str(action_config.get("reference_sha256") or "").strip()
        if not reference_text or not expected_sha256:
            raise RuntimeError(
                "Frozen sleek eating profile reference metadata is incomplete."
            )
        reference_path = Path(reference_text)
        if not reference_path.is_file():
            raise FileNotFoundError(
                f"Frozen sleek eating profile reference is missing: {reference_path}"
            )
        if file_sha256(reference_path) != expected_sha256:
            raise RuntimeError("Frozen sleek eating profile reference digest changed.")
        return reference_path, SCAIL_FROZEN_SLEEK_EATING_POLICY

    if isinstance(effective_subject_config, dict):
        reference_path = Path(effective_subject_config["actions"][action]["reference"])
    elif profile is not None:
        reference_path = Path(profile["action_references"][action])
    else:
        raise ValueError(f"No subject reference is available for {action}")
    return (
        reference_path,
        (
            "subject_profile_action_identity_with_bowl_mapped_red"
            if action == "eating"
            else "subject_profile_action_reference"
        ),
    )


def uses_profile_standard_direct(
    job: dict[str, Any],
    effective_subject_config: dict[str, Any],
) -> bool:
    """Return true only for an explicit no-upload profile request."""

    direct = effective_subject_config.get("standard_direct")
    return bool(
        isinstance(direct, dict)
        and direct.get("enabled") is True
        and job.get("subject_source") == "profile"
        and job.get("requested_subject_source") in {None, "profile"}
        and not job.get("uploaded_identity_input_sha256")
        and not job.get("uploaded_identity_source_sha256")
    )


def validate_standard_direct_runtime_video(
    path: Path,
    *,
    expected_frames: int,
    expected_fps: float,
) -> dict[str, Any]:
    """Decode the frozen runtime and enforce the public 512-square contract."""

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open frozen standard runtime: {path}")
    decoded = 0
    try:
        observed_fps = float(capture.get(cv2.CAP_PROP_FPS))
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape[:2] != (512, 512):
                raise RuntimeError(
                    "Frozen standard runtime frame size changed: "
                    f"{frame.shape[1]}x{frame.shape[0]}"
                )
            decoded += 1
    finally:
        capture.release()
    if decoded != expected_frames:
        raise RuntimeError(
            f"Frozen standard runtime has {decoded} frames, expected {expected_frames}."
        )
    if abs(observed_fps - expected_fps) > 0.01:
        raise RuntimeError(
            f"Frozen standard runtime is {observed_fps:.6f} fps, "
            f"expected {expected_fps:.6f}."
        )
    return {
        "width": 512,
        "height": 512,
        "frames": decoded,
        "fps": observed_fps,
        "duration_seconds": decoded / observed_fps,
        "sha256": file_sha256(path),
    }


def write_standard_direct_preview_assets(
    action_standard: dict[str, Any],
    job_dir: Path,
    action: str,
) -> dict[str, Path]:
    """Copy the frozen selected (frame 40) standard preview/reference/mask."""

    selected = action_standard.get("selected_reference")
    if not isinstance(selected, dict):
        raise RuntimeError(f"Frozen {action} selected standard assets are missing")
    frame_index = int(selected["selected_frame_index"])
    guide = job_dir / f"{action}_main_case_standard_selected_frame{frame_index:03d}.png"
    cutout = job_dir / (
        f"{action}_main_case_standard_selected_reference_frame{frame_index:03d}.png"
    )
    mask = job_dir / (
        f"{action}_main_case_standard_selected_identity_mask_frame{frame_index:03d}.png"
    )
    for kind, target in (
        ("preview", guide),
        ("cutout", cutout),
        ("identity_mask", mask),
    ):
        source = Path(str(selected[kind]))
        expected = str(selected[f"{kind}_sha256"])
        if not source.is_file() or file_sha256(source) != expected:
            raise RuntimeError(
                f"Frozen standard_direct {action} {kind} digest changed."
            )
        shutil.copy2(source, target)
        if file_sha256(target) != expected:
            raise RuntimeError(f"Copied standard_direct {action} {kind} is corrupt.")
    return {"guide": guide, "cutout": cutout, "mask": mask}


run_profile_standard_direct_job = (
    _local_video_subject_replacement.run_profile_standard_direct_job
)


revalidate_uploaded_qwen_references_before_scail = (
    _local_video_subject_replacement.revalidate_uploaded_qwen_references_before_scail
)


run_subject_replacement_job = (
    _local_video_subject_replacement.run_subject_replacement_job
)


run_job = _local_video_job_runner.run_job


def worker_loop() -> None:
    LOGGER.info("Local video worker started")
    while not STOP_EVENT.is_set():
        try:
            job_id = JOB_QUEUE.get(timeout=1)
        except queue.Empty:
            continue
        try:
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                already_cancelled = bool(
                    job
                    and (
                        job.get("status") == "cancelled" or job.get("cancel_requested")
                    )
                )
            if already_cancelled:
                mark_job_cancelled(job_id)
                LOGGER.info("Skipped cancelled job %s", job_id)
                continue
            LOGGER.info("Starting job %s", job_id)
            run_job(job_id)
            LOGGER.info("Completed job %s", job_id)
        except JobCancelled:
            LOGGER.info("Job %s was cancelled by the user", job_id)
            mark_job_cancelled(job_id)
            try:
                wait_comfy_idle(timeout=120)
                free_comfy_models()
            except Exception:  # noqa: BLE001 - best effort cleanup
                LOGGER.exception(
                    "GPU cleanup after cancelled job %s also failed", job_id
                )
        except Exception as exc:  # noqa: BLE001 - worker must persist all failures
            LOGGER.error("Job %s failed: %s\n%s", job_id, exc, traceback.format_exc())
            with JOBS_LOCK:
                failed_job = dict(JOBS.get(job_id) or {})
            reference_failure: dict[str, Any] = {}
            if failed_job.get("reference_generation_status") == "running":
                reference_failure = {
                    "reference_generation_status": "failed",
                    "reference_generation_completed": False,
                }
            update_job(
                job_id,
                status="failed",
                stage="failed",
                error=str(exc)[:4000],
                recovery_active=False,
                recovery_attempt=None,
                quality_retry=None,
                active_prompt_id=None,
                current_pose_index=None,
                current_pose=None,
                finished_at=utc_now(),
                **reference_failure,
            )
            try:
                wait_comfy_idle(timeout=120)
                free_comfy_models()
            except Exception:  # noqa: BLE001 - best effort cleanup
                LOGGER.exception("GPU cleanup after failed job %s also failed", job_id)
        finally:
            JOB_QUEUE.task_done()
    LOGGER.info("Local video worker stopped")


def restore_jobs() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    queued: list[str] = []
    for path in sorted(OUTPUT_ROOT.glob("*/status.json")):
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
            job_id = str(job["job_id"])
            callback_changed = False
            if (
                job.get("cancel_requested")
                and job.get("status") not in TERMINAL_JOB_STATUSES
            ):
                job.update(
                    status="cancelled",
                    stage="cancelled",
                    error=None,
                    active_prompt_id=None,
                    finished_at=utc_now(),
                    updated_at=utc_now(),
                )
                write_json_atomic(path, job)
                callback_changed = True
            elif job.get("status") == "running":
                if job.get("reference_generation_status") == "running":
                    job.update(
                        reference_generation_status="failed",
                        reference_generation_completed=False,
                    )
                job.update(
                    status="failed",
                    stage="failed",
                    error="API service restarted while this job was running; submit it again.",
                    finished_at=utc_now(),
                    updated_at=utc_now(),
                )
                write_json_atomic(path, job)
                callback_changed = True
            elif job.get("status") == "queued":
                queued.append(job_id)
            JOBS[job_id] = job
            if callback_changed:
                report_job_callback(job)
        except (OSError, KeyError, json.JSONDecodeError):
            LOGGER.exception("Could not restore job status from %s", path)
    for job_id in queued[:MAX_QUEUE_SIZE]:
        JOB_QUEUE.put_nowait(job_id)
    for job_id in queued[MAX_QUEUE_SIZE:]:
        update_job(
            job_id,
            status="failed",
            stage="failed",
            error="Queue capacity was exceeded while restoring the service.",
            finished_at=utc_now(),
        )


app = FastAPI(
    title="Local Pet Subject-Replacement API",
    version=API_VERSION,
    description=(
        "Generate local-model pet subject-replacement videos from the eating and grooming "
        "driving templates. The legacy Qwen-to-Wan keyframe workflow remains available only "
        "when generation_mode=keyframe_video is requested explicitly."
    ),
    docs_url=f"{API_PREFIX}/docs",
    redoc_url=None,
    openapi_url=f"{API_PREFIX}/openapi.json",
)


@app.on_event("startup")
def startup() -> None:
    global WORKER_THREAD
    if not API_KEY:
        LOGGER.warning(
            "LOCAL_VIDEO_API_KEY is not configured; protected endpoints will return 503"
        )
    JOB_CALLBACK_REPORTER.start()
    restore_jobs()
    STOP_EVENT.clear()
    WORKER_THREAD = threading.Thread(
        target=worker_loop, name="local-video-worker", daemon=True
    )
    WORKER_THREAD.start()


@app.on_event("shutdown")
def shutdown() -> None:
    STOP_EVENT.set()
    if WORKER_THREAD and WORKER_THREAD.is_alive():
        WORKER_THREAD.join(timeout=5)
    JOB_CALLBACK_REPORTER.stop(timeout=5)


@app.get(f"{API_PREFIX}/health")
def health() -> dict[str, Any]:
    return _local_video_health.health()


@app.get(f"{API_PREFIX}/subjects")
def list_subject_profiles(
    _: None = Depends(require_api_key),
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for path in sorted(SUBJECT_PROFILE_DIR.glob("*.json")):
        try:
            items.append(subject_profile_public(load_subject_profile(path.stem)))
        except (ValueError, FileNotFoundError):
            LOGGER.exception("Ignoring invalid subject profile %s", path)
    return {
        "default": DEFAULT_SUBJECT_PROFILE,
        "items": items,
        "actions": [
            {"id": action, "label": SUBJECT_REPLACEMENT_ACTION_LABELS[action]}
            for action in SUBJECT_REPLACEMENT_ACTIONS
        ],
    }


@app.get(f"{API_PREFIX}/subjects/{{profile_id}}/preview")
def subject_profile_preview(
    profile_id: str,
    action: str | None = Query(default=None),
    _: None = Depends(require_api_key),
) -> FileResponse:
    try:
        profile = load_subject_profile(profile_id)
        selected_action = parse_subject_actions(
            action or str(profile.get("preview_action") or "eating")
        )[0]
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    preview = Path(profile["action_references"][selected_action])
    if not preview.is_file():
        raise HTTPException(status_code=404, detail="Subject preview is unavailable.")
    return FileResponse(
        preview,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


async def close_uploaded_identity_uploads(
    *uploads: UploadFile | None,
) -> None:
    for upload in uploads:
        if upload is not None:
            await upload.close()


async def read_uploaded_identity_views(
    image: UploadFile | None,
    image_left: UploadFile | None,
    image_right: UploadFile | None,
) -> dict[str, bytes]:
    uploads = {
        "front": image,
        "left": image_left,
        "right": image_right,
    }
    present_roles = tuple(
        role
        for role in UPLOADED_IDENTITY_VIEW_ROLES
        if uploads[role] is not None and bool(uploads[role].filename)
    )
    try:
        if present_roles not in {
            (),
            ("front",),
            UPLOADED_IDENTITY_VIEW_ROLES,
        }:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Identity upload must contain either image only, or the "
                    "complete image + image_left + image_right set."
                ),
            )
        raw_views: dict[str, bytes] = {}
        for role in present_roles:
            upload = uploads[role]
            if upload is None:
                raise RuntimeError("Resolved upload unexpectedly disappeared.")
            raw = await upload.read(MAX_UPLOAD_BYTES + 1)
            if not raw:
                raise HTTPException(
                    status_code=422,
                    detail=f"Uploaded identity {role} image is empty.",
                )
            if len(raw) > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=(f"Identity {role} image exceeds {MAX_UPLOAD_BYTES} bytes."),
                )
            raw_views[role] = raw
        return raw_views
    finally:
        await close_uploaded_identity_uploads(image, image_left, image_right)


async def create_subject_replacement_api_job(
    *,
    image: UploadFile | None,
    prompt: str,
    actions_raw: str | None,
    subject_profile_id: str | None,
    output_format: str,
    scail_steps: int | None,
    pose_strength: float | None,
    seed: int | None,
    image_left: UploadFile | None = None,
    image_right: UploadFile | None = None,
    callback_enabled: bool = False,
) -> dict[str, Any]:
    try:
        actions = parse_subject_actions(actions_raw)
        profile = load_subject_profile(subject_profile_id)
        effective_subject_config = resolve_effective_subject_config(
            profile,
            actions,
            seed=seed,
            scail_steps=scail_steps,
            pose_strength=pose_strength,
        )
    except (ValueError, FileNotFoundError) as exc:
        await close_uploaded_identity_uploads(image, image_left, image_right)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if output_format not in {"mp4", "transparent_gif"}:
        await close_uploaded_identity_uploads(image, image_left, image_right)
        raise HTTPException(
            status_code=422,
            detail="Subject replacement output_format must be mp4 or transparent_gif.",
        )
    if JOB_QUEUE.full():
        await close_uploaded_identity_uploads(image, image_left, image_right)
        raise HTTPException(
            status_code=503,
            detail="Generation queue is full; try again later.",
        )

    raw_views = await read_uploaded_identity_views(
        image,
        image_left,
        image_right,
    )
    raw = raw_views.get("front")

    prompt = prompt.strip()
    if prompt:
        translated, translation = translate_texts_to_english([prompt])
        prompt_en = translated[0]
        translation.update(
            {
                "prompt_applied": contains_han(prompt),
                "pose_applied_count": 0,
                "applied": contains_han(prompt),
            }
        )
    else:
        prompt_en = ""
        translation = {
            "enabled": PROMPT_TRANSLATION_ENABLED,
            "applied": False,
            "prompt_applied": False,
            "pose_applied_count": 0,
        }

    expected_pose_policies: dict[str, str] = {}
    try:
        # Profile stills remain immutable downstream contour/fur standards only.
        # Qwen pose geometry is generated from the frozen Main_case driving
        # videos, never from these historical action stills.
        raw_standard_direct = effective_subject_config.get("standard_direct")
        standard_direct_request = bool(
            raw is None
            and isinstance(raw_standard_direct, dict)
            and raw_standard_direct.get("enabled") is True
        )
        if raw is not None:
            effective_subject_config, expected_pose_policies = (
                configure_uploaded_subject_validation(
                    effective_subject_config,
                    actions,
                )
            )
        profile_reference_actions = [] if standard_direct_request else list(actions)
        if standard_direct_request:
            verify_profile_standard_direct_assets(raw_standard_direct, actions)
        else:
            require_subject_replacement_assets(
                profile,
                actions,
                require_profile_references=bool(profile_reference_actions),
                profile_reference_actions=profile_reference_actions,
                effective_config=effective_subject_config,
                expected_pose_policies=expected_pose_policies,
            )
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    resolved_seed = int(effective_subject_config["base_seed"])
    job_id = (
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:10]}"
    )
    job_dir = OUTPUT_ROOT / job_id
    try:
        subject_source = "profile"
        uploaded_identity_input_sha256: str | None = None
        uploaded_identity_source_sha256: str | None = None
        uploaded_identity_views: dict[str, dict[str, Any]] = {}
        uploaded_identity_view_mode: str | None = None
        if raw is not None:
            uploaded_identity_views = persist_uploaded_identity_views(
                job_dir,
                raw_views,
            )
            front_view = uploaded_identity_views["front"]
            uploaded_identity_input_sha256 = front_view["input_sha256"]
            uploaded_identity_source_sha256 = front_view["source_sha256"]
            uploaded_identity_view_mode = (
                "front_left_right"
                if len(uploaded_identity_views) == 3
                else "single_view"
            )
            subject_source = "upload_pending_qwen"
        if profile_reference_actions:
            effective_subject_config = snapshot_effective_subject_references(
                effective_subject_config,
                job_dir,
                profile_reference_actions,
            )
        effective_actions = effective_subject_config["actions"]
        final_frames = (
            TRANSPARENT_GIF_FRAMES
            if output_format == "transparent_gif"
            else int(effective_subject_config["frames"])
        )
        final_fps = (
            TRANSPARENT_GIF_FPS
            if output_format == "transparent_gif"
            else float(effective_subject_config["fps"])
        )
        final_duration = (
            TRANSPARENT_GIF_DURATION_SECONDS
            if output_format == "transparent_gif"
            else float(effective_subject_config["frames"])
            / float(effective_subject_config["fps"])
        )
        job = {
            "job_id": job_id,
            "callback_enabled": callback_enabled,
            "callback_source": "pet_gif_api" if callback_enabled else None,
            "status": "queued",
            "stage": "queued",
            "progress": 0,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "started_at": None,
            "finished_at": None,
            "generation_mode": "subject_replacement",
            "request_kind": "video",
            "prompt": prompt,
            "prompt_original": prompt,
            "prompt_en": prompt_en,
            "prompt_translation": translation,
            "actions": actions,
            "action_labels": [
                SUBJECT_REPLACEMENT_ACTION_LABELS[item] for item in actions
            ],
            "action_count": len(actions),
            "poses": actions,
            "poses_original": actions,
            "poses_en": actions,
            "pose_count": len(actions),
            "subject_profile": profile["id"],
            "subject_profile_label": profile.get("display_name") or profile["id"],
            "subject_profile_path": profile["_path"],
            "subject_profile_sha256": effective_subject_config["profile_sha256"],
            "subject_source": subject_source,
            "requested_subject_source": ("upload" if raw is not None else "profile"),
            "effective_subject_source": subject_source,
            "uploaded_identity_input_sha256": (uploaded_identity_input_sha256),
            "uploaded_identity_source_sha256": (uploaded_identity_source_sha256),
            "uploaded_identity_view_policy": (
                UPLOADED_IDENTITY_VIEW_POLICY if raw is not None else None
            ),
            "uploaded_identity_view_mode": uploaded_identity_view_mode,
            "uploaded_identity_view_count": len(uploaded_identity_views),
            "uploaded_identity_view_roles": list(uploaded_identity_views),
            "uploaded_identity_views": uploaded_identity_views,
            "subject_reference_paths": (
                {}
                if raw is not None
                else {
                    action: str(
                        effective_subject_config["standard_direct"]["actions"][action][
                            "selected_reference"
                        ]["cutout"]
                        if standard_direct_request
                        else effective_actions[action]["reference"]
                    )
                    for action in actions
                }
            ),
            "reference_pipeline": (
                QWEN_FULL_BODY_FIRST_FRAME_PIPELINE
                if raw is not None
                else (
                    PROFILE_STANDARD_DIRECT_POLICY
                    if standard_direct_request
                    else "validated_subject_profile_fast_path"
                )
            ),
            "reference_pipeline_version": (
                QWEN_FULL_BODY_FIRST_FRAME_PIPELINE_VERSION if raw is not None else None
            ),
            "generation_audit_policy": GENERATION_AUDIT_POLICY,
            "visual_content_audit_mode": (
                "blocking" if GENERATION_VISUAL_AUDITS_BLOCKING else "advisory"
            ),
            "visual_content_audits_blocking": (
                GENERATION_VISUAL_AUDITS_BLOCKING
            ),
            "runtime_integrity_checks_required": True,
            "expected_pose_policies": dict(expected_pose_policies),
            "expected_surface_gate_actions": (
                list(actions)
                if raw is not None and GENERATION_VISUAL_AUDITS_BLOCKING
                else []
            ),
            "fur_contour_validation_policy": None,
            "upload_validation_policy": (
                QWEN_FULL_BODY_FIRST_FRAME_PIPELINE if raw is not None else None
            ),
            "surface_texture_gate_status": (
                (
                    "pending_qwen_reference"
                    if GENERATION_VISUAL_AUDITS_BLOCKING
                    else "skipped_advisory"
                )
                if raw is not None
                else None
            ),
            "reference_generation_status": (
                "pending"
                if raw is not None
                else (
                    "standard_direct_ready"
                    if standard_direct_request
                    else "profile_ready"
                )
            ),
            "reference_image_profile": (
                SUBJECT_REPLACEMENT_REFERENCE_IMAGE_PROFILE if raw is not None else None
            ),
            "reference_normalization_contract": (
                {
                    "full_body": "landscape_normalize_then_alpha_matte",
                    "action_keyframes": (QWEN_ACTION_REFERENCE_NORMALIZATION_POLICY),
                    "action_candidate_matte": (QWEN_ACTION_CANDIDATE_MATTE_POLICY),
                    "size": [512, 512],
                    "identity_view_mode": uploaded_identity_view_mode,
                    "identity_view_roles": list(uploaded_identity_views),
                    "identity_view_policy": UPLOADED_IDENTITY_VIEW_POLICY,
                    "identity_views_used_only_for_qwen_canonical": True,
                    "action_stage_raw_identity_views_used": False,
                    "action_stage_identity_input": (
                        "normalized_qwen_full_body_canonical_image1"
                    ),
                    "action_stage_geometry_input": ("main_case_action_guide_image2"),
                    "raw_upload_sent_to_scail": False,
                    "selected_candidate_byte_copied_to_scail_reference": True,
                    "wan_green_derived_one_way_from_rgba": True,
                }
                if raw is not None
                else None
            ),
            "driving_templates": {
                action: str(effective_actions[action]["driving_animation"])
                for action in actions
            },
            "seed_mode": effective_subject_config["seed_mode"],
            "effective_action_seeds": {
                action: int(effective_actions[action]["effective_seed"])
                for action in actions
            },
            "effective_action_pose_strengths": {
                action: float(effective_actions[action]["pose_strength"])
                for action in actions
            },
            "effective_subject_config": effective_subject_config,
            "output_format": output_format,
            "skip_video": False,
            "generation_engine": (
                "deterministic-main-case-standard-direct"
                if standard_direct_request
                else (
                    "qwen-image-edit-plus-scail2-14b-fp8"
                    if raw is not None
                    else "scail2-14b-fp8"
                )
            ),
            "reference_generation_engine": (
                "qwen-image-edit" if raw is not None else None
            ),
            "video_engine": (
                "deterministic-main-case-standard-direct"
                if standard_direct_request
                else "scail2-14b-fp8"
            ),
            "scail_model": (
                None if standard_direct_request else effective_subject_config["model"]
            ),
            "scail_dpo_lora": (
                None
                if standard_direct_request
                else {"name": SCAIL_DPO_LORA, "strength": 1.0}
            ),
            "scail_distill_lora": (
                None
                if standard_direct_request
                else {"name": SCAIL_DISTILL_LORA, "strength": 0.8}
            ),
            "scail_steps": (
                None
                if standard_direct_request
                else int(effective_subject_config["steps"])
            ),
            "pose_strength": pose_strength,
            "source_frames": int(effective_subject_config["frames"]),
            "source_fps": float(effective_subject_config["fps"]),
            "resolution": "512x512",
            "width": 512,
            "height": 512,
            "frames": final_frames,
            "fps": final_fps,
            "duration_seconds": final_duration,
            "seed": resolved_seed,
            "visual_style": (
                "uploaded_identity_qwen_full_body_action_keyframes_scail"
                if raw is not None
                else (
                    "main_case_profile_standard_direct"
                    if standard_direct_request
                    else "sleek_tabby_subject_replacement"
                )
            ),
            "standard_direct_policy": (
                PROFILE_STANDARD_DIRECT_POLICY if standard_direct_request else None
            ),
            "standard_direct_revision": (
                effective_subject_config["standard_direct"]["revision"]
                if standard_direct_request
                else None
            ),
            "standard_direct_role": (
                effective_subject_config["standard_direct"]["role"]
                if standard_direct_request
                else None
            ),
            "standard_direct_actions": (
                json.loads(
                    json.dumps(effective_subject_config["standard_direct"]["actions"])
                )
                if standard_direct_request
                else {}
            ),
            "gpu_required": False if standard_direct_request else True,
            "gpu_submitted": False,
            "gpu_bypass_reason": (
                effective_subject_config["standard_direct"]["gpu_bypass_reason"]
                if standard_direct_request
                else None
            ),
            "generation_controls_applied": not standard_direct_request,
            "qwen_reference_generation_for_subject_replacement": bool(raw is not None),
            "direct_scail_uploaded_identity_reference": False,
            "canonical_reference_generated": False,
            "canonical_reference_used": False,
            "current_action": None,
            "current_action_label": None,
            "current_action_index": None,
            "outputs": [],
            "subject_replacement_results": [],
            "error": None,
            "cancel_requested": False,
            "active_prompt_id": None,
        }
        with JOBS_LOCK:
            JOBS[job_id] = job
        persist_job(job_id)
        JOB_QUEUE.put_nowait(job_id)
    except Exception:
        with JOBS_LOCK:
            JOBS.pop(job_id, None)
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    return public_job(job)


@app.post(f"{API_PREFIX}/jobs", status_code=status.HTTP_202_ACCEPTED)
async def create_job(
    image: UploadFile | None = File(
        default=None,
        description=(
            "Optional front/primary subject image (PNG/JPEG/WebP). Use with "
            "image_left and image_right for three-view canonical generation."
        ),
    ),
    image_left: UploadFile | None = File(
        default=None,
        description="Optional left-side view; requires image and image_right.",
    ),
    image_right: UploadFile | None = File(
        default=None,
        description="Optional right-side view; requires image and image_left.",
    ),
    prompt: str = Form(default="", max_length=2000),
    generation_mode: str = Form(
        default="subject_replacement",
        description="subject_replacement (default) or keyframe_video.",
    ),
    actions: str | None = Form(
        default=None,
        description="Subject replacement actions: eating and/or grooming.",
    ),
    subject_profile: str | None = Form(default=DEFAULT_SUBJECT_PROFILE),
    scail_steps: int | None = Form(default=None, ge=4, le=12),
    pose_strength: float | None = Form(default=None, ge=0.0, le=2.0),
    poses: str | None = Form(
        default=None,
        description="Optional JSON string array, newline-separated list, or pipe-separated list.",
    ),
    pose_count: int = Form(
        default=4,
        description="Number of outputs. Defaults to running, eating, drinking, and lying down.",
    ),
    resolution: str = Form(default="720p"),
    output_format: str = Form(
        default="mp4",
        description="mp4 or transparent_gif; transparent_gif forces a 512x512 final output.",
    ),
    skip_video: bool = Form(
        default=False,
        description="Generate the selected pose keyframes but skip Wan video inference.",
    ),
    request_kind: str = Form(
        default="video", description="Calling UI mode: image or video."
    ),
    image_profile: str | None = Form(
        default=None,
        description="Optional named image baseline profile.",
    ),
    user_loras: str | None = Form(
        default=None,
        description="Optional JSON array of user LoRA model_name, original_filename, and strength.",
    ),
    user_lora: str | None = Form(
        default=None, description="Validated user_uploads/*.safetensors name."
    ),
    user_lora_original_filename: str | None = Form(default=None),
    user_lora_strength: float = Form(default=1.0, ge=0.0, le=2.0),
    user_lora_mode: str = Form(
        default="replace", description="replace or stack the built-in style LoRA."
    ),
    qwen_steps: int | None = Form(default=None, ge=1, le=30),
    qwen_cfg: float | None = Form(default=None, ge=0.1, le=5.0),
    qwen_denoise: float | None = Form(default=None, ge=0.1, le=1.0),
    wan_steps: int = Form(default=WAN_STEPS, ge=4, le=40),
    wan_cfg: float = Form(default=WAN_CFG, ge=1.0, le=8.0),
    wan_shift: float = Form(default=WAN_SHIFT, ge=1.0, le=12.0),
    seed: int | None = Form(default=None, ge=0, le=2**63 - 1),
    x_pet_gif_callback: str | None = Header(
        default=None,
        alias="X-Pet-Gif-Callback",
        include_in_schema=False,
    ),
    _: None = Depends(require_api_key),
) -> dict[str, Any]:
    callback_enabled = secrets.compare_digest(
        (x_pet_gif_callback or "").strip(),
        "1",
    )
    normalized_generation_mode = generation_mode.strip().lower().replace("-", "_")
    generation_mode_aliases = {
        "subject_replacement": "subject_replacement",
        "replacement": "subject_replacement",
        "scail": "subject_replacement",
        "keyframe_video": "keyframe_video",
        "legacy": "keyframe_video",
    }
    normalized_generation_mode = generation_mode_aliases.get(
        normalized_generation_mode,
        normalized_generation_mode,
    )
    if normalized_generation_mode not in {"subject_replacement", "keyframe_video"}:
        await close_uploaded_identity_uploads(image, image_left, image_right)
        raise HTTPException(
            status_code=422,
            detail="generation_mode must be subject_replacement or keyframe_video.",
        )
    if normalized_generation_mode == "subject_replacement":
        return await create_subject_replacement_api_job(
            image=image,
            image_left=image_left,
            image_right=image_right,
            prompt=prompt,
            actions_raw=actions,
            subject_profile_id=subject_profile,
            output_format=output_format,
            scail_steps=scail_steps,
            pose_strength=pose_strength,
            seed=seed,
            callback_enabled=callback_enabled,
        )
    if image_left is not None or image_right is not None:
        await close_uploaded_identity_uploads(image, image_left, image_right)
        raise HTTPException(
            status_code=422,
            detail=(
                "image_left and image_right are supported only for "
                "generation_mode=subject_replacement."
            ),
        )
    if image is None:
        raise HTTPException(
            status_code=422,
            detail="An image is required for the legacy keyframe_video workflow.",
        )
    request_kind = request_kind.strip().lower()
    if request_kind not in {"image", "video"}:
        raise HTTPException(
            status_code=422, detail="request_kind must be image or video."
        )
    if request_kind == "image":
        output_format = "transparent_gif"
        skip_video = True
        resolution = "512x512"
    if image_profile is None:
        image_profile = DEFAULT_IMAGE_PROFILE or None
    profile = resolve_image_profile(image_profile, request_kind=request_kind)
    qwen_steps = int(
        qwen_steps
        if qwen_steps is not None
        else (profile["qwen_steps"] if profile else 4)
    )
    qwen_cfg = float(
        qwen_cfg if qwen_cfg is not None else (profile["qwen_cfg"] if profile else 1.0)
    )
    qwen_denoise = float(
        qwen_denoise
        if qwen_denoise is not None
        else (profile["qwen_denoise"] if profile else 1.0)
    )
    seed = int(seed if seed is not None else (profile["seed"] if profile else 20260713))
    if output_format not in OUTPUT_FORMATS:
        raise HTTPException(
            status_code=422,
            detail=f"output_format must be one of {sorted(OUTPUT_FORMATS)}.",
        )
    user_lora_mode = user_lora_mode.strip().lower()
    if user_lora_mode not in {"replace", "stack"}:
        raise HTTPException(
            status_code=422, detail="user_lora_mode must be replace or stack."
        )
    if user_loras and user_loras.strip() and user_lora and user_lora.strip():
        raise HTTPException(
            status_code=422,
            detail="Use either user_loras or the legacy user_lora fields, not both.",
        )
    uploaded_user_lora_specs = normalize_user_lora_specs(
        user_loras,
        legacy_model_name=user_lora,
        legacy_original_filename=user_lora_original_filename,
        legacy_strength=user_lora_strength,
    )
    user_lora_specs, user_lora_mode, image_profile_lora_source = (
        merge_image_profile_loras(
            profile,
            uploaded_user_lora_specs,
            requested_mode=user_lora_mode,
        )
    )
    first_user_lora = user_lora_specs[0] if user_lora_specs else None
    active_user_lora_specs = [
        item for item in user_lora_specs if float(item["strength"]) != 0.0
    ]
    user_lora = first_user_lora["model_name"] if first_user_lora else None
    user_lora_original_filename = (
        first_user_lora["original_filename"] if first_user_lora else None
    )
    user_lora_strength = first_user_lora["strength"] if first_user_lora else 1.0
    if output_format == "transparent_gif":
        resolution = "512x512"
    elif resolution not in RESOLUTIONS:
        raise HTTPException(
            status_code=422, detail=f"resolution must be one of {sorted(RESOLUTIONS)}."
        )
    prompt = prompt.strip()
    if not prompt:
        raise HTTPException(status_code=422, detail="prompt cannot be blank.")
    selected_poses = parse_poses(
        poses,
        pose_count,
        defaults=profile.get("default_poses") if profile else None,
    )
    if JOB_QUEUE.full():
        raise HTTPException(
            status_code=503, detail="Generation queue is full; try again later."
        )
    prompt_en, selected_poses_en, translation = translate_prompt_bundle(
        prompt,
        selected_poses,
    )
    selected_pose_preset_ids = [
        (
            MULTIPOSE_PRESET_BY_PROMPT_ZH[pose]["id"]
            if pose in MULTIPOSE_PRESET_BY_PROMPT_ZH
            else None
        )
        for pose in selected_poses
    ]
    if all(preset_id is not None for preset_id in selected_pose_preset_ids):
        pose_prompt_source = (
            str(profile.get("pose_prompt_source"))
            if profile and profile.get("pose_prompt_source")
            else "default_multipose_presets"
        )
    elif any(preset_id is not None for preset_id in selected_pose_preset_ids):
        pose_prompt_source = "mixed_presets_and_request"
    else:
        pose_prompt_source = "request"

    raw = await image.read(MAX_UPLOAD_BYTES + 1)
    await image.close()
    if not raw:
        raise HTTPException(status_code=422, detail="Uploaded image is empty.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413, detail=f"Image exceeds {MAX_UPLOAD_BYTES} bytes."
        )

    job_id = (
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:10]}"
    )
    job_dir = OUTPUT_ROOT / job_id
    try:
        normalize_uploaded_image(
            raw,
            job_dir / "input.png",
            identity_source_path=job_dir / "input_identity_source.png",
        )
        final_frames = (
            None
            if skip_video
            else (
                TRANSPARENT_GIF_FRAMES
                if output_format == "transparent_gif"
                else FRAME_COUNT
            )
        )
        final_fps = (
            None
            if skip_video
            else (TRANSPARENT_GIF_FPS if output_format == "transparent_gif" else FPS)
        )
        final_duration = (
            None
            if skip_video
            else (
                TRANSPARENT_GIF_DURATION_SECONDS
                if output_format == "transparent_gif"
                else FRAME_COUNT / FPS
            )
        )
        profile_suppresses_builtin_style = bool(
            profile
            and image_profile_lora_source
            in {"image_profile", "image_profile_plus_user"}
        )
        builtin_style_applied = (
            bool(QWEN_STYLE_LORA)
            and not profile_suppresses_builtin_style
            and not (active_user_lora_specs and user_lora_mode == "replace")
        )
        job = {
            "job_id": job_id,
            "callback_enabled": callback_enabled,
            "callback_source": "pet_gif_api" if callback_enabled else None,
            "status": "queued",
            "stage": "queued",
            "progress": 0,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "started_at": None,
            "finished_at": None,
            "prompt": prompt,
            "prompt_original": prompt,
            "prompt_en": prompt_en,
            "prompt_translation": translation,
            "poses": selected_poses,
            "poses_original": selected_poses,
            "poses_en": selected_poses_en,
            "pose_count": len(selected_poses),
            "image_profile": profile["id"] if profile else None,
            "image_profile_revision": (profile.get("revision") if profile else None),
            "image_profile_label": profile["label"] if profile else None,
            "image_profile_source_job_id": profile["source_job_id"]
            if profile
            else None,
            "pose_prompt_source": pose_prompt_source,
            "pose_preset_ids": selected_pose_preset_ids,
            "pose_seed_offsets": (
                dict(profile.get("pose_seed_offsets", {}))
                if profile
                else dict(MULTIPOSE_SEED_OFFSETS)
            ),
            "identity_palette_guard": bool(
                profile.get("identity_palette_guard", False)
                if profile
                else output_format == "transparent_gif"
            ),
            "identity_palette_policy": (
                profile.get("identity_palette_policy")
                if profile
                else (
                    "source_palette_guard_v2"
                    if output_format == "transparent_gif"
                    else None
                )
            ),
            "image_profile_lora_source": image_profile_lora_source,
            "image_profile_suppresses_builtin_style": (
                profile_suppresses_builtin_style
            ),
            "lock_retry_lora_strengths": bool(
                profile
                and profile.get("lock_retry_lora_strengths", False)
                and image_profile_lora_source == "image_profile"
            ),
            "output_format": output_format,
            "skip_video": skip_video,
            "request_kind": request_kind,
            "generation_mode": "keyframes_only" if skip_video else "video",
            "visual_style": "identity_preserving_cute_3d_natural_cat"
            if output_format == "transparent_gif"
            else "reference",
            "qwen_style_profile": (
                f"image_profile:{profile['id']}"
                if profile and image_profile_lora_source != "user_override"
                else (
                    QWEN_STYLE_PROFILE
                    if builtin_style_applied
                    else (
                        "user_lora_mix"
                        if len(active_user_lora_specs) > 1
                        else ("user_lora" if active_user_lora_specs else "baseline")
                    )
                )
            ),
            "qwen_style_lora": QWEN_STYLE_LORA if builtin_style_applied else None,
            "qwen_style_lora_strength": QWEN_STYLE_LORA_STRENGTH
            if builtin_style_applied
            else None,
            "qwen_pose_style_lora_strength": (
                QWEN_POSE_STYLE_LORA_STRENGTH if builtin_style_applied else None
            ),
            "user_lora": user_lora,
            "user_lora_original_filename": user_lora_original_filename
            if user_lora
            else None,
            "user_lora_strength": user_lora_strength if user_lora else None,
            "user_loras": user_lora_specs,
            "requested_user_loras": user_lora_specs,
            "uploaded_user_loras": uploaded_user_lora_specs,
            "image_profile_user_loras": profile["user_loras"] if profile else [],
            "user_lora_count": len(user_lora_specs),
            "active_user_lora_count": len(active_user_lora_specs),
            "user_lora_mode": user_lora_mode if user_lora_specs else None,
            "qwen_steps": qwen_steps,
            "qwen_cfg": qwen_cfg,
            "qwen_denoise": qwen_denoise,
            "wan_steps": wan_steps,
            "wan_cfg": wan_cfg,
            "wan_shift": wan_shift,
            "resolution": resolution,
            "width": RESOLUTIONS[resolution][0],
            "height": RESOLUTIONS[resolution][1],
            "frames": final_frames,
            "fps": final_fps,
            "duration_seconds": final_duration,
            "seed": seed,
            "canonical_user_lora_total_cap": (
                profile["canonical_total_cap"]
                if profile and image_profile_lora_source != "user_override"
                else QWEN_CANONICAL_USER_LORA_TOTAL_CAP
            ),
            "canonical_user_lora_per_lora_cap": (
                profile["canonical_per_lora_cap"]
                if profile and image_profile_lora_source != "user_override"
                else QWEN_CANONICAL_USER_LORA_PER_LORA_CAP
            ),
            "pose_user_lora_first_total_cap": (
                profile["pose_first_total_cap"]
                if profile and image_profile_lora_source != "user_override"
                else QWEN_POSE_USER_LORA_TOTAL_CAP
            ),
            "pose_user_lora_first_per_lora_cap": (
                profile["pose_first_per_lora_cap"]
                if profile and image_profile_lora_source != "user_override"
                else QWEN_POSE_USER_LORA_PER_LORA_CAP
            ),
            "current_pose_index": None,
            "current_pose": None,
            "outputs": [],
            "automatic_recovery_used": False,
            "recovered": False,
            "degraded_recovery": False,
            "recovery_active": False,
            "recovery_attempt": None,
            "recovery_history": [],
            "background_removal_failed": False,
            "background_removal_failures": [],
            "transparent_outputs_complete": True,
            "error": None,
            "cancel_requested": False,
            "active_prompt_id": None,
        }
        with JOBS_LOCK:
            JOBS[job_id] = job
        persist_job(job_id)
        JOB_QUEUE.put_nowait(job_id)
    except Exception:
        with JOBS_LOCK:
            JOBS.pop(job_id, None)
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    return public_job(job)


@app.get(f"{API_PREFIX}/jobs")
def list_jobs(
    limit: int = Query(default=20, ge=1, le=100),
    _: None = Depends(require_api_key),
) -> dict[str, Any]:
    with JOBS_LOCK:
        jobs = sorted(
            JOBS.values(), key=lambda item: item.get("created_at", ""), reverse=True
        )[:limit]
    return {"jobs": [public_job(job) for job in jobs], "count": len(jobs)}


@app.get(f"{API_PREFIX}/jobs/{{job_id}}")
def job_status(job_id: str, _: None = Depends(require_api_key)) -> dict[str, Any]:
    return get_job(job_id)


@app.post(f"{API_PREFIX}/jobs/{{job_id}}/cancel")
def cancel_job(job_id: str, _: None = Depends(require_api_key)) -> dict[str, Any]:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        current_status = str(job.get("status", ""))
        if current_status in TERMINAL_JOB_STATUSES:
            return public_job(job)
        active_prompt_id = job.get("active_prompt_id")
        if current_status == "queued":
            job.update(
                status="cancelled",
                stage="cancelled",
                cancel_requested=True,
                active_prompt_id=None,
                error=None,
                finished_at=utc_now(),
                updated_at=utc_now(),
            )
        else:
            job.update(
                stage="cancel_requested",
                cancel_requested=True,
                updated_at=utc_now(),
            )
        snapshot = dict(job)
    write_json_atomic(status_path(job_id), snapshot)
    report_job_callback(snapshot)
    interrupt_comfy_prompt(active_prompt_id)
    return public_job(snapshot)


@app.get(f"{API_PREFIX}/jobs/{{job_id}}/files/{{filename}}")
def download_file(
    job_id: str, filename: str, _: None = Depends(require_api_key)
) -> FileResponse:
    if Path(filename).name != filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    job = get_job(job_id)
    allowed = {item["filename"] for item in job.get("outputs", [])}
    if filename not in allowed:
        raise HTTPException(
            status_code=404, detail="Output file not found for this job."
        )
    path = OUTPUT_ROOT / job_id / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Output file is missing on disk.")
    return FileResponse(path, filename=filename)
