from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image, ImageDraw

import local_video_api as pipeline


class _PoseWorkflowCaptured(RuntimeError):
    """Stop a mocked Qwen run immediately after its first action workflow."""


def _write_test_upload_identity(job_dir: Path) -> dict[str, str]:
    input_path = job_dir / "input.png"
    source_path = job_dir / "input_identity_source.png"
    Image.new("RGB", (32, 32), "white").save(input_path)
    Image.new("RGB", (40, 30), (224, 224, 224)).save(source_path)
    return {
        "uploaded_identity_input_sha256": pipeline.file_sha256(input_path),
        "uploaded_identity_source_sha256": pipeline.file_sha256(source_path),
    }


def _passed_uploaded_front_view_audit() -> dict[str, object]:
    return {
        "policy": pipeline.UPLOADED_FRONT_VIEW_AUDIT_POLICY,
        "role": "front",
        "blocking": False,
        "acceptable": True,
        "status": "passed_advisory",
        "issues": [],
        "qualified_detections": [{"qualified": True}],
    }


def _write_eating_contact_fixture(
    root: Path,
    *,
    name: str,
    muzzle_width_bowl_ratio: float,
    muzzle_gap_px: int,
) -> tuple[Path, Image.Image]:
    size = (256, 256)
    bowl_left, bowl_top, bowl_right, bowl_bottom = 68, 170, 188, 225
    bowl_width = bowl_right - bowl_left
    muzzle_width = int(round(bowl_width * muzzle_width_bowl_ratio))
    muzzle_left = (bowl_left + bowl_right - muzzle_width) // 2
    muzzle_bottom = bowl_top - muzzle_gap_px
    muzzle_top = muzzle_bottom - 25

    reference = Image.new("RGBA", size, (0, 0, 0, 0))
    reference.paste((105, 110, 115, 255), (30, 20, 226, 220))
    reference.paste(
        (214, 208, 196, 255),
        (bowl_left, bowl_top, bowl_right, bowl_bottom),
    )
    reference.paste(
        (205, 118, 116, 255),
        (
            muzzle_left,
            muzzle_top,
            muzzle_left + muzzle_width,
            muzzle_bottom,
        ),
    )
    candidate = root / f"{name}.png"
    reference.save(candidate)

    colored_mask = Image.new("RGB", size, (0, 0, 0))
    colored_mask.paste((0, 0, 255), (30, 20, 226, 220))
    colored_mask.paste(
        (255, 0, 0),
        (bowl_left, bowl_top, bowl_right, bowl_bottom),
    )
    return candidate, colored_mask


def _write_thin_warm_rim_beige_bowl_fixture(path: Path) -> None:
    """Create a beige bowl whose thin warm rim joins the warm kibble union."""

    image = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((55, 76, 456, 435), fill=(32, 32, 36, 255))
    draw.rectangle((72, 330, 132, 435), fill=(235, 235, 235, 255))
    draw.rectangle((294, 330, 356, 435), fill=(235, 235, 235, 255))
    draw.rectangle((135, 350, 289, 414), fill=(190, 180, 160, 255))
    warm = (180, 120, 70, 255)
    draw.rectangle((145, 343, 278, 351), fill=warm)
    draw.rectangle((135, 343, 139, 415), fill=warm)
    draw.rectangle((135, 411, 273, 415), fill=warm)
    image.save(path)


def _configured_test_upload_profile(
    profile: dict[str, object],
    actions: list[str],
) -> tuple[dict[str, object], dict[str, str]]:
    effective = pipeline.resolve_effective_subject_config(
        profile,
        actions,
        seed=None,
        scail_steps=None,
        pose_strength=None,
    )
    return pipeline.configure_uploaded_subject_validation(effective, actions)


def _passed_surface_metric(
    reference: Path,
    action: str,
) -> dict[str, object]:
    digest = pipeline.file_sha256(reference)
    return {
        "acceptable": True,
        "status": "passed",
        "selected_attempt": 1,
        "selected_candidate_path": str(reference),
        "selected_candidate_sha256": digest,
        "gate": {
            "policy": pipeline.FUR_SURFACE_GATE_POLICY,
            "gate_version": pipeline.FUR_SURFACE_GATE_VERSION,
            "threshold_version": pipeline.FUR_SURFACE_THRESHOLD_VERSION,
            "action": action,
            "alpha_provenance": "chroma_key",
            "candidate_sha256": digest,
        },
    }


def _record_fake_canonical_image1(
    job_id: str,
    job_dir: Path,
    comfy_input: Path,
) -> None:
    canonical = job_dir / "cartoon_reference.png"
    comfy_canonical = comfy_input / "local_video_api" / job_id / canonical.name
    comfy_canonical.parent.mkdir(parents=True, exist_ok=True)
    comfy_canonical.write_bytes(canonical.read_bytes())
    digest = pipeline.file_sha256(canonical)
    public_canonical = job_dir / "cartoon_reference_transparent.png"
    public_digest = pipeline.file_sha256(public_canonical)
    pipeline.update_job(
        job_id,
        canonical_qwen_image1_path=str(canonical),
        canonical_qwen_image1_sha256=digest,
        canonical_qwen_image1_comfy_path=str(comfy_canonical),
        canonical_qwen_image1_comfy_sha256=digest,
        canonical_qwen_image1_role=("normalized_green_full_body_image1_and_latent"),
        canonical_public_reference_path=str(public_canonical),
        canonical_public_reference_sha256=public_digest,
        canonical_public_reference_role=("normalized_transparent_full_body_reference"),
    )


def _write_fake_geometry_guide(
    _job_id: str,
    source: Path,
    target: Path,
    *,
    frames: int,
    fps: float,
) -> dict[str, object]:
    target.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (32, 32), (0, 255, 0))
    image.paste((96, 96, 96), (6, 6, 26, 26))
    image.save(target)
    return {
        "policy": "test_texture_free_geometry",
        "source_sha256": pipeline.file_sha256(source),
        "guide_sha256": pipeline.file_sha256(target),
        "palette": {"source_rgb_preserved": False},
        "frames": frames,
        "fps": fps,
    }


def _test_uploaded_identity_driving(
    driving_animations: dict[str, str],
    references: dict[str, str],
    actions: tuple[str, ...],
) -> dict[str, object]:
    uploaded_actions: dict[str, object] = {}
    for action in actions:
        geometry_reference = Path(references[action]).with_name(
            f"{action}-white-maquette-reference.png"
        )
        geometry_image = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
        geometry_image.paste((160, 160, 160, 255), (112, 72, 400, 456))
        geometry_image.save(geometry_reference)
        uploaded_actions[action] = {
            "source": driving_animations[action],
            "source_sha256": pipeline.file_sha256(
                Path(driving_animations[action])
            ),
            "runtime": driving_animations[action],
            "runtime_sha256": pipeline.file_sha256(
                Path(driving_animations[action])
            ),
            "geometry_reference": str(geometry_reference),
            "geometry_reference_sha256": pipeline.file_sha256(
                geometry_reference
            ),
            "selected_frame_index": 0,
            "selected_frame_timestamp_seconds": 0.0,
            "action_contract": f"test_{action}_action_contract",
        }
    return {
        "revision": "test-white-maquette-v1",
        "role": "identity_neutral_white_maquette_action_geometry_and_motion",
        "actions": uploaded_actions,
    }


def _subject_profile_with_action_references(
    root: Path,
    actions: tuple[str, ...] = ("eating", "grooming"),
) -> dict[str, object]:
    references: dict[str, str] = {}
    for action in actions:
        reference = root / f"{action}-validated-reference.png"
        reference_image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        reference_image.paste((80, 80, 80, 255), (12, 10, 52, 56))
        reference_image.save(reference)
        references[action] = str(reference)
    driving_animations: dict[str, str] = {}
    for action in ("eating", "grooming"):
        driving = root / f"{action}-driving.gif"
        Image.new("RGB", (32, 32), "white").save(driving, format="GIF")
        driving_animations[action] = str(driving)
    identity_masks: dict[str, str] = {}
    for action in actions:
        identity_mask = root / f"{action}-validated-identity-mask.png"
        mask_image = Image.new("L", (64, 64), 0)
        mask_image.paste(255, (12, 10, 52, 56))
        mask_image.save(identity_mask)
        identity_masks[action] = str(identity_mask)
    return {
        "schema_version": 2,
        "id": "sleek_tabby_cat_v1",
        "action_references": references,
        "uploaded_identity_driving": _test_uploaded_identity_driving(
            driving_animations,
            references,
            actions,
        ),
        "validated_action_baselines": {
            action: {
                "run_id": f"test-{action}-surface-baseline",
                "reference": references[action],
                "reference_sha256": pipeline.file_sha256(Path(references[action])),
            }
            for action in actions
        },
        "standard_direct": {
            "enabled": True,
            "policy": pipeline.PROFILE_STANDARD_DIRECT_POLICY,
            "role": pipeline.PROFILE_STANDARD_DIRECT_ROLE,
            "revision": "test-main-case-standard-v1",
        },
        "driving_standards": {
            "revision": "test-main-case-standard-v1",
            "actions": {
                action: {
                    "source": driving_animations[action],
                    "source_sha256": pipeline.file_sha256(
                        Path(driving_animations[action])
                    ),
                    "runtime": driving_animations[action],
                    "runtime_sha256": pipeline.file_sha256(
                        Path(driving_animations[action])
                    ),
                    "validated_run_id": f"test-{action}-standard-direct",
                }
                for action in actions
            },
        },
        "reference_assets": {
            "revision": "test-main-case-standard-v1",
            "actions": {
                action: {
                    "selected_frame_index": 0,
                    "selected_frame_timestamp_seconds": 0.0,
                    "identity_mask_policy": "test_identity_mask",
                    "preview": references[action],
                    "preview_sha256": pipeline.file_sha256(Path(references[action])),
                    "cutout": references[action],
                    "cutout_sha256": pipeline.file_sha256(Path(references[action])),
                    "identity_mask": identity_masks[action],
                    "identity_mask_sha256": pipeline.file_sha256(
                        Path(identity_masks[action])
                    ),
                }
                for action in actions
            },
        },
        "runtime": {
            "engine_profile": "scail2-14b-fp8-v1",
            "prompt_profile": "sleek-tabby-scail-v1",
            "default_base_seed": 20260726,
            "steps": 6,
            "frames": 81,
            "fps": 16,
            "actions": {
                "eating": {
                    "driving_animation": driving_animations["eating"],
                    "driving_animation_sha256": pipeline.file_sha256(
                        Path(driving_animations["eating"])
                    ),
                    "pose_strength": 1.1,
                    "seed_offset": 0,
                },
                "grooming": {
                    "driving_animation": driving_animations["grooming"],
                    "driving_animation_sha256": pipeline.file_sha256(
                        Path(driving_animations["grooming"])
                    ),
                    "pose_strength": 1.15,
                    "seed_offset": 2,
                },
            },
        },
    }


def _subject_profile_v2(root: Path) -> dict[str, object]:
    references: dict[str, str] = {}
    driving_animations: dict[str, str] = {}
    identity_masks: dict[str, str] = {}
    for action in ("eating", "grooming"):
        reference = root / f"{action}-profile-reference.png"
        reference_image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        reference_image.paste((80, 80, 80, 255), (12, 10, 52, 56))
        reference_image.save(reference)
        references[action] = str(reference)
        driving = root / f"{action}-driving.gif"
        driving.write_bytes(b"GIF89a")
        driving_animations[action] = str(driving)
        identity_mask = root / f"{action}-identity-mask.png"
        mask_image = Image.new("L", (64, 64), 0)
        mask_image.paste(255, (12, 10, 52, 56))
        mask_image.save(identity_mask)
        identity_masks[action] = str(identity_mask)
    baseline = root / "eating-fur-baseline.gif"
    baseline_frame = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    baseline_frame.paste((80, 80, 80, 255), (6, 8, 26, 28))
    baseline_frame.save(baseline, format="GIF")
    profile: dict[str, object] = {
        "schema_version": 2,
        "id": "sleek_tabby_cat_v1",
        "display_name": "Sleek tabby",
        "action_references": references,
        "uploaded_identity_driving": _test_uploaded_identity_driving(
            driving_animations,
            references,
            ("eating", "grooming"),
        ),
        "fur_constraints": {
            "positive": "frozen close-lying test coat",
            "negative": "frozen expanded test coat",
        },
        "runtime": {
            "engine_profile": "scail2-14b-fp8-v1",
            "prompt_profile": "sleek-tabby-scail-v1",
            "default_base_seed": 20260726,
            "steps": 6,
            "frames": 81,
            "fps": 16,
            "actions": {
                "eating": {
                    "driving_animation": driving_animations["eating"],
                    "driving_animation_sha256": pipeline.file_sha256(
                        Path(driving_animations["eating"])
                    ),
                    "pose_strength": 1.1,
                    "seed_offset": 0,
                },
                "grooming": {
                    "driving_animation": driving_animations["grooming"],
                    "driving_animation_sha256": pipeline.file_sha256(
                        Path(driving_animations["grooming"])
                    ),
                    "pose_strength": 1.15,
                    "seed_offset": 2,
                },
            },
        },
        "validated_replacement_baseline": {
            "run_id": "test-eating-baseline",
            "transparent_gif": str(baseline),
        },
        "validated_action_baselines": {
            action: {
                "run_id": f"test-{action}-surface-baseline",
                "reference": references[action],
                "reference_sha256": pipeline.file_sha256(Path(references[action])),
            }
            for action in ("eating", "grooming")
        },
        "standard_direct": {
            "enabled": True,
            "policy": pipeline.PROFILE_STANDARD_DIRECT_POLICY,
            "role": pipeline.PROFILE_STANDARD_DIRECT_ROLE,
            "revision": "test-main-case-standard-v1",
        },
        "driving_standards": {
            "revision": "test-main-case-standard-v1",
            "actions": {
                action: {
                    "source": driving_animations[action],
                    "source_sha256": pipeline.file_sha256(
                        Path(driving_animations[action])
                    ),
                    "runtime": driving_animations[action],
                    "runtime_sha256": pipeline.file_sha256(
                        Path(driving_animations[action])
                    ),
                    "validated_run_id": f"test-{action}-standard-direct",
                }
                for action in ("eating", "grooming")
            },
        },
        "reference_assets": {
            "revision": "test-main-case-standard-v1",
            "actions": {
                action: {
                    "selected_frame_index": 0,
                    "selected_frame_timestamp_seconds": 0.0,
                    "identity_mask_policy": "test_identity_mask",
                    "preview": references[action],
                    "preview_sha256": pipeline.file_sha256(Path(references[action])),
                    "cutout": references[action],
                    "cutout_sha256": pipeline.file_sha256(Path(references[action])),
                    "identity_mask": identity_masks[action],
                    "identity_mask_sha256": pipeline.file_sha256(
                        Path(identity_masks[action])
                    ),
                }
                for action in ("eating", "grooming")
            },
        },
    }
    profile_path = root / "sleek_tabby_cat_v1.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    profile["_path"] = str(profile_path)
    return profile


class SubjectReplacementQwenReferenceTestBase(unittest.TestCase):
    TEST_QWEN_STYLE_LORA = "pet3d-cat-test.safetensors"

    @staticmethod
    def _masked_pet3d_composite_pose_gate() -> dict[str, object]:
        return {
            "policy": pipeline.GROOMING_POSE_GATE_POLICY,
            "version": pipeline.GROOMING_POSE_GATE_VERSION,
            "available": True,
            "acceptable": False,
            "verdict": "rejected",
            "issues": ["raised_foreleg_edge_density_too_low"],
            "warnings": [],
            "penalty": 1000.0,
            "thresholds": {
                "edge_density_min": 0.11,
                "structural_blur_sigma": 4.0,
                "structural_edge_density_min": 0.015,
                "chamfer_coverage_min": 0.62,
                "chamfer_mean_distance_max": 7.2,
                "continuous_edge_span_min": 0.24,
                "paw_muzzle_gap_max": 0.04,
                "lab_median_delta_max": 35.0,
                "lab_histogram_distance_max": 0.42,
                "texture_p90_ratio_min": 0.25,
                "texture_p90_ratio_max": 4.0,
            },
            "metrics": {
                "raised_foreleg_edge_density": 0.087542,
                "raised_foreleg_structural_evidence": {
                    "available": True,
                    "policy": ("gaussian_lowpass_organic_foreleg_structure_v1"),
                    "blur_sigma": 4.0,
                    "edge_density": 0.034458,
                },
                "chamfer": {
                    "available": True,
                    "coverage_within_radius": 0.729827,
                    "mean_distance_px": 4.554365,
                    "p90_distance_px": 10.83082,
                },
                "continuous_edge": {"span_fraction": 0.292969},
                "paw_muzzle_contact": {
                    "available": True,
                    "gap_fraction_of_bbox_height": 0.02,
                },
                "appearance_consistency": {
                    "available": True,
                    "raised": {"gradient_p90": 109.114616},
                    "planted": {"gradient_p90": 71.021126},
                    "lab_median_delta": 9.055385,
                    "lab_histogram_bhattacharyya": {"mean": 0.25},
                    "texture_gradient_p90_ratio": 1.528921,
                },
            },
        }

    def _masked_pet3d_composite_repair_audit(self) -> dict[str, object]:
        return {
            "policy": pipeline.GROOMING_QWEN_FORELEG_REPAIR_POLICY,
            "composite_gate_policy": (
                pipeline.GROOMING_QWEN_FORELEG_REPAIR_COMPOSITE_GATE_POLICY
            ),
            "composite_edge_density_ratio_min": (
                pipeline.GROOMING_QWEN_FORELEG_REPAIR_EDGE_DENSITY_RATIO_MIN
            ),
            "composite_planted_paw_gradient_p90_max": (
                pipeline.GROOMING_QWEN_FORELEG_REPAIR_PLANTED_PAW_GRADIENT_P90_MAX
            ),
            "composite_raised_paw_gradient_p90_max": (
                pipeline.GROOMING_QWEN_FORELEG_REPAIR_RAISED_PAW_GRADIENT_P90_MAX
            ),
            "global_pose_threshold_changed": False,
            "localized_style_policy": (
                pipeline.GROOMING_QWEN_FORELEG_REPAIR_STYLE_POLICY
            ),
            "localized_style_lora": self.TEST_QWEN_STYLE_LORA,
            "localized_style_strength": (
                pipeline.GROOMING_QWEN_FORELEG_REPAIR_STYLE_STRENGTH
            ),
            "localized_style_scope": (
                "sampler_noise_mask_only; image2_remains_identity_"
                "palette_marking_and_coat_length_authority"
            ),
            "sampler_noise_mask_sha256": "a" * 64,
        }

    @staticmethod
    def _masked_pet3d_composite_fur_gate() -> dict[str, object]:
        return {
            "available": True,
            "evaluated": True,
            "status": "passed",
            "acceptable": True,
            "issues": [],
            "gate": {
                "policy": pipeline.FUR_SURFACE_GATE_POLICY,
                "gate_version": pipeline.FUR_SURFACE_GATE_VERSION,
                "threshold_version": pipeline.FUR_SURFACE_THRESHOLD_VERSION,
                "action": "grooming",
                "reliable_alpha_required": True,
                "candidate_sha256": "b" * 64,
            },
        }

    @staticmethod
    def _canonical_completion_repair_audit() -> dict[str, object]:
        return {
            "policy": pipeline.GROOMING_CANONICAL_FORELEG_COMPLETION_POLICY,
            "production_canonical_masked_completion": True,
            "source_identity_palette_acceptable": True,
            "localized_style_policy": ("disabled_for_canonical_masked_completion"),
            "localized_style_lora": None,
            "localized_style_strength": 0.0,
            "composite_recovery_enabled": False,
            "global_pose_threshold_changed": False,
            "main_case_rgb_appearance_exposed_on_repair_attempt": False,
            "sampler_noise_mask_sha256": "c" * 64,
            "completion_canvas": {
                "policy": (pipeline.GROOMING_CANONICAL_FORELEG_COMPLETION_POLICY),
                "semantic_pixels_exposed_to_qwen": False,
                "identity_or_geometry_seed_inserted": False,
                "source_composite_preserved_before_masked_sampling": True,
            },
        }

    @staticmethod
    def _canonical_direct_surface_gate(
        candidate_path: Path,
    ) -> dict[str, object]:
        return {
            "available": True,
            "evaluated": True,
            "acceptable": False,
            "verdict": "rejected",
            "status": "rejected",
            "issues": ["brush_texture_coherence"],
            "candidate_absolute_issues": ["brush_texture_coherence"],
            "relative_issues": [],
            "baseline_issues": [],
            "penalty": 1000.0,
            "thresholds": {
                "coarse_gradient_p90_max": 0.24,
                "coarse_coherent_fraction_max": 0.52,
                "radial_brush_fraction_max": 0.52,
                "hough_radial_line_fraction_max": 0.65,
                "coherent_long_line_energy_max": 0.24,
            },
            "candidate": {
                "valid": True,
                "input": {
                    "supported_contract": True,
                    "foreground_method": "alpha",
                },
                "metrics": {
                    "texture": {
                        "coarse_gradient_p90": 0.18903524,
                        "structure_tensor_coherence_mean": 0.71565646,
                        "structure_tensor_coherent_fraction": 0.55975808,
                        "orientation_persistence_fraction": 0.69283441,
                        "fine_radial_brush_fraction": 0.34900594,
                        "fine_radial_brush_energy": 0.51005183,
                        "coarse_radial_brush_fraction": 0.31607198,
                        "coarse_radial_brush_energy": 0.40949999,
                        "coherent_long_line_energy": 0.20033163,
                        "hough": {
                            "radial_line_fraction": 0.46527208,
                        },
                    },
                },
            },
            "gate": {
                "policy": pipeline.FUR_SURFACE_GATE_POLICY,
                "gate_version": pipeline.FUR_SURFACE_GATE_VERSION,
                "threshold_version": pipeline.FUR_SURFACE_THRESHOLD_VERSION,
                "action": "grooming",
                "candidate_path": str(candidate_path),
                "candidate_sha256": pipeline.file_sha256(candidate_path),
                "candidate_comparison_mode": (
                    pipeline.UPLOADED_IDENTITY_SURFACE_COMPARISON_MODE
                ),
                "profile_pixels_used_for_candidate_appearance_validation": False,
            },
        }

    @staticmethod
    def _canonical_surface_debt_audit() -> dict[str, object]:
        return {
            "acceptable": False,
            "issues": [
                "brush_texture_coherence",
                "coherent_long_fur_energy",
            ],
            "candidate": {
                "valid": True,
                "input": {
                    "supported_contract": True,
                    "foreground_method": "alpha",
                },
                "metrics": {
                    "texture": {
                        "coarse_gradient_p90": 0.23142637,
                        "structure_tensor_coherence_mean": 0.70045859,
                        "structure_tensor_coherent_fraction": 0.55510023,
                        "orientation_persistence_fraction": 0.5892125,
                        "fine_radial_brush_fraction": 0.47366194,
                        "fine_radial_brush_energy": 0.52617555,
                        "coarse_radial_brush_fraction": 0.54219812,
                        "coarse_radial_brush_energy": 0.55551475,
                        "coherent_long_line_energy": 0.24417802,
                        "hough": {
                            "radial_line_fraction": 0.46029041,
                        },
                    },
                },
            },
        }

    @staticmethod
    def _identity_palette_signature_for_test(
        fractions: dict[str, float],
        *,
        saturated_warm_fraction: float,
    ) -> dict[str, object]:
        achromatic = sum(
            fractions.get(category, 0.0) for category in ("black", "gray", "white")
        )
        warm = sum(
            fractions.get(category, 0.0)
            for category in (
                "warm_brown_or_orange",
                "red_or_pink",
                "yellow_or_gold",
            )
        )
        return {
            "available": True,
            "fractions": fractions,
            "dominant": [
                category for category, fraction in fractions.items() if fraction >= 0.10
            ],
            "achromatic_fraction": achromatic,
            "chromatic_fraction": 1.0 - achromatic,
            "warm_fraction": warm,
            "saturated_warm_fraction": saturated_warm_fraction,
        }

    def _capture_action_workflow(
        self,
        root: Path,
        *,
        action: str,
        action_identity_policy: str,
        style_lora: str = TEST_QWEN_STYLE_LORA,
        style_strength: float = 0.15,
        pose_attempt: int = 1,
        grooming_gate_acceptable: bool = True,
        advisory_grooming_minimum_acceptable: bool = True,
        advisory_grooming_minimum_acceptable_attempts: set[int] | None = None,
        advisory_grooming_minimum_issues: list[str] | None = None,
        identity_palette_guard: bool = False,
        canonical_action_palette_drift_attempts: set[int] | None = None,
        action_chroma_acceptable_attempts: set[int] | None = None,
        eating_pose_contact_unreliable: bool = False,
        pose_safe_grooming_attempt4: bool = False,
        uploaded_white_maquette: bool = False,
        main_case_raw_identity_chain: bool | None = None,
        persist_identity_digest: bool = True,
        persist_identity_source_digest: bool = True,
        visual_audits_blocking: bool = True,
    ) -> tuple[dict[str, object], str, str]:
        if pose_attempt not in {1, 2, 3, 4, 5, 6}:
            raise ValueError("pose_attempt must be between 1 and 6")
        if main_case_raw_identity_chain is None:
            main_case_raw_identity_chain = (
                action_identity_policy
                == pipeline.POSE_IDENTITY_INPUT_POLICY_PROFILE_GEOMETRY_FIRST
            )
        output_root = root / "outputs"
        comfy_input = root / "comfy-input"
        job_id = f"capture-{action}"
        job_dir = output_root / job_id
        job_dir.mkdir(parents=True)
        Image.new("RGB", (32, 32), "white").save(job_dir / "input.png")
        Image.new("RGB", (32, 32), "white").save(job_dir / "input_identity_source.png")

        relative_dir = Path("local_video_api") / job_id
        pose_guide_name = f"{action}_qwen_pose_guide.png"
        pose_guide = comfy_input / relative_dir / pose_guide_name
        pose_guide.parent.mkdir(parents=True)
        production_canonical_grooming = bool(
            action == "grooming"
            and action_identity_policy == pipeline.POSE_IDENTITY_INPUT_POLICY_CANONICAL
        )
        grooming_completion_mask_path: Path | None = None
        grooming_completion_mask_digest: str | None = None
        grooming_open_channel_pixels: int | None = None
        grooming_open_channel_points = [
            [208, 245],
            [214, 260],
            [218, 275],
            [218, 295],
            [216, 310],
            [204, 335],
        ]
        if action == "grooming":
            qwen_guide_image = Image.new("RGB", (512, 512), (0, 255, 0))
            qwen_guide_image.paste(
                pipeline.GROOMING_QWEN_GUIDE_FOREGROUND_RGB,
                (112, 72, 400, 440),
            )
            qwen_guide_image.paste(
                pipeline.GROOMING_QWEN_GUIDE_FORELEG_RGB,
                (180, 220, 220, 340),
            )
            qwen_guide_image.paste((0, 255, 0), (200, 250, 204, 300))
            if production_canonical_grooming:
                grooming_completion_mask_path = pose_guide.with_name(
                    pose_guide.stem + "_completion_mask.png"
                )
                qwen_guide_image.save(grooming_completion_mask_path)
                grooming_completion_mask_digest = pipeline.file_sha256(
                    grooming_completion_mask_path
                )
                qwen_guide_image = Image.new("RGB", (512, 512), (0, 255, 0))
                qwen_guide_image.paste(
                    pipeline.GROOMING_QWEN_GUIDE_FOREGROUND_RGB,
                    (112, 72, 400, 440),
                )
                ImageDraw.Draw(qwen_guide_image).line(
                    [tuple(point) for point in grooming_open_channel_points],
                    fill=(0, 255, 0),
                    width=9,
                )
            qwen_guide_image.save(pose_guide)
            if production_canonical_grooming:
                qwen_array = np.asarray(qwen_guide_image, dtype=np.uint8)
                private_fixture_foreground = np.zeros((512, 512), dtype=bool)
                private_fixture_foreground[72:440, 112:400] = True
                grooming_open_channel_pixels = int(
                    np.count_nonzero(
                        private_fixture_foreground
                        & np.all(qwen_array == (0, 255, 0), axis=2)
                    )
                )
        else:
            Image.new("RGB", (32, 32), (0, 255, 0)).save(pose_guide)
        pose_guide_digest = pipeline.file_sha256(pose_guide)
        grooming_metadata_path = pose_guide.with_name(
            pose_guide.stem + "_metadata.json"
        )
        grooming_generator_metadata: dict[str, object] | None = None
        grooming_private_gate_digest: str | None = None
        if action == "grooming":
            private_gate_path = pose_guide.with_name(pose_guide.stem + "_cv_gate.png")
            private_gate_image = Image.new("RGB", (512, 512), (0, 255, 0))
            private_gate_image.paste((96, 96, 96), (112, 72, 208, 440))
            private_gate_image.paste((136, 136, 136), (208, 72, 304, 440))
            private_gate_image.paste((176, 176, 176), (304, 72, 400, 440))
            private_gate_image.save(private_gate_path)
            grooming_private_gate_digest = pipeline.file_sha256(private_gate_path)
            private_metadata_path = private_gate_path.with_name(
                private_gate_path.stem + "_metadata.json"
            )
            private_metadata_path.write_text(
                json.dumps(
                    {
                        "policy": (pipeline.GROOMING_PRIVATE_POSE_GATE_GUIDE_POLICY),
                        "raised_paw_muzzle_relation": {
                            "passed": True,
                            "source_luma_values_preserved": False,
                            "black_or_tool_semantics_preserved": False,
                            "output_encoding": {
                                "method": (
                                    "compressed_blurred_quantized_neutral_anatomy"
                                )
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            grooming_generator_metadata = {
                "policy": pipeline.GROOMING_QWEN_GEOMETRY_GUIDE_POLICY,
                "guide_sha256": pose_guide_digest,
                "metadata_path": str(grooming_metadata_path),
                "raised_paw_muzzle_relation": {"passed": True},
                "qwen_guide": {
                    "role": (
                        "image2_action_geometry_guide"
                        if production_canonical_grooming
                        else "image1_action_geometry_sampler"
                    ),
                    "path": str(pose_guide),
                    "sha256": pose_guide_digest,
                    "foreground_rgb": list(pipeline.GROOMING_QWEN_GUIDE_FOREGROUND_RGB),
                    "foreground_palette_rgb": (
                        [list(pipeline.GROOMING_QWEN_GUIDE_FOREGROUND_RGB)]
                        if production_canonical_grooming
                        else [
                            list(pipeline.GROOMING_QWEN_GUIDE_FOREGROUND_RGB),
                            list(pipeline.GROOMING_QWEN_GUIDE_FORELEG_RGB),
                        ]
                    ),
                    "unique_foreground_rgb_count": (
                        1 if production_canonical_grooming else 2
                    ),
                    "internal_luma_variation": (
                        False if production_canonical_grooming else True
                    ),
                    "internal_luma_variation_role": (
                        "none"
                        if production_canonical_grooming
                        else "synthetic_anatomy_labels_only"
                    ),
                    "appearance_bearing_internal_luma_variation": False,
                    "main_case_appearance_exposed_to_qwen": False,
                    "synthetic_foreleg_semantic": (
                        None
                        if production_canonical_grooming
                        else {
                            "method": (pipeline.GROOMING_QWEN_FORELEG_SEMANTIC_METHOD),
                            "rgb": list(pipeline.GROOMING_QWEN_GUIDE_FORELEG_RGB),
                            "pixels": 4600,
                            "pixels_before_separator": 4800,
                            "capsule_width": 28,
                            "paw_radius": 17,
                            "shoulder": [200, 330],
                            "paw_center": [200, 240],
                            "derived_from": (
                                "verified_raised_paw_muzzle_relation_geometry_only"
                            ),
                            "source_rgb_or_luma_sampled": False,
                            "appearance_bearing": False,
                        }
                    ),
                    "structural_negative_space": (
                        {
                            "method": (pipeline.GROOMING_QWEN_NEGATIVE_SPACE_METHOD),
                            "pixels": grooming_open_channel_pixels,
                            "line_width": 9,
                            "points": grooming_open_channel_points,
                            "bbox": [199, 240, 224, 341],
                            "span_vs_paw_shoulder": 0.9,
                            "curve_closed": False,
                            "paw_endpoint_open": True,
                            "shoulder_endpoint_open": True,
                            "foreleg_remains_attached_at_shoulder": True,
                            "paw_muzzle_contact_remains_connected": True,
                            "derived_from": (
                                "verified_raised_paw_muzzle_relation_geometry_only"
                            ),
                            "source_rgb_or_luma_sampled": False,
                            "appearance_bearing": False,
                            "adds_foreground_color": False,
                        }
                        if production_canonical_grooming
                        else {
                            "method": (pipeline.GROOMING_QWEN_NEGATIVE_SPACE_METHOD),
                            "pixels": 200,
                            "appearance_bearing": False,
                            "adds_foreground_color": False,
                        }
                    ),
                    "closed_foreleg_capsule_exposed": False,
                    "internal_green_separator_exposed": (
                        production_canonical_grooming
                    ),
                    "raised_foreleg_validation_role": (
                        "public_open_curve_geometry_plus_private_cv"
                        if production_canonical_grooming
                        else "private_cv_only"
                    ),
                },
                "private_pose_gate": {
                    "guide_path": str(private_gate_path),
                    "guide_sha256": grooming_private_gate_digest,
                    "metadata_path": str(private_metadata_path),
                    "policy": (pipeline.GROOMING_PRIVATE_POSE_GATE_GUIDE_POLICY),
                    "exposed_to_qwen": False,
                },
            }
            if production_canonical_grooming:
                assert grooming_completion_mask_path is not None
                assert grooming_completion_mask_digest is not None
                grooming_generator_metadata["private_completion_mask"] = {
                    "role": "local_noise_mask_derivation_only",
                    "policy": pipeline.GROOMING_PRIVATE_COMPLETION_MASK_POLICY,
                    "path": str(grooming_completion_mask_path),
                    "sha256": grooming_completion_mask_digest,
                    "exposed_to_qwen": False,
                    "loaded_as_qwen_image": False,
                    "semantic_foreleg_pixels": 4600,
                }
            grooming_metadata_path.write_text(
                json.dumps(grooming_generator_metadata),
                encoding="utf-8",
            )
        canonical_source = root / "generated-canonical.png"
        Image.new("RGB", (32, 32), (0, 255, 0)).save(canonical_source)
        action_reference = root / f"{action}-action-reference.png"
        action_reference_image = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
        action_reference_draw = ImageDraw.Draw(action_reference_image)
        action_reference_draw.ellipse(
            (130, 72, 410, 462),
            fill=(82, 82, 82, 255),
        )
        action_reference_draw.line(
            [(205, 350), (184, 286), (196, 232)],
            fill=(224, 224, 224, 255),
            width=44,
        )
        action_reference_draw.ellipse(
            (174, 210, 218, 252),
            fill=(236, 236, 236, 255),
        )
        action_reference_draw.rectangle(
            (250, 276, 292, 446),
            fill=(190, 190, 190, 255),
        )
        action_reference_image.save(action_reference)
        white_maquette_driver = root / "grooming-white-maquette-driver.mp4"
        white_maquette_driver.write_bytes(b"fixture-white-maquette-frame-zero")
        original_started_at = "2026-08-03T00:00:00+00:00"

        photography_loras = [
            dict(item)
            for item in pipeline.IMAGE_PROFILES[
                pipeline.SUBJECT_REPLACEMENT_REFERENCE_IMAGE_PROFILE
            ]["user_loras"]
        ]
        model_root = root / "models"
        user_lora_dir = model_root / "loras" / "user_uploads"
        for item in photography_loras:
            lora_path = model_root / "loras" / Path(item["model_name"])
            lora_path.parent.mkdir(parents=True, exist_ok=True)
            lora_path.write_bytes(b"test-lora")
        jobs: dict[str, dict[str, object]] = {
            job_id: {
                "job_id": job_id,
                "generation_mode": "subject_replacement",
                "qwen_reference_poses": [
                    pipeline.subject_replacement_pose_prompt(action)
                ],
                "qwen_reference_prompt_en": "profile-locked reference prompt",
                "prompt": "",
                "prompt_en": "",
                "poses": [action],
                "poses_en": [action],
                "outputs": [],
                "cancel_requested": False,
                "started_at": original_started_at,
                "seed": 123,
                "user_loras": photography_loras,
                "user_lora_mode": "replace",
                "image_profile_lora_source": "image_profile",
                "qwen_steps": 4,
                "qwen_cfg": 1.0,
                "qwen_denoise": 1.0,
                "pose_user_lora_first_total_cap": (
                    pipeline.QWEN_SUBJECT_REPLACEMENT_POSE_LORA_TOTAL_CAP
                ),
                "pose_user_lora_first_per_lora_cap": (
                    pipeline.QWEN_SUBJECT_REPLACEMENT_POSE_LORA_TOTAL_CAP
                ),
                "pose_reference_inputs": {
                    action: (relative_dir / pose_guide_name).as_posix(),
                },
                "pose_reference_input_sha256s": {
                    action: pose_guide_digest,
                },
                "pose_reference_metadata": (
                    {
                        "grooming": {
                            "policy": (pipeline.GROOMING_QWEN_POSE_REFERENCE_POLICY),
                            "generator_policy": (
                                pipeline.GROOMING_QWEN_GEOMETRY_GUIDE_POLICY
                            ),
                            "guide_sha256": pose_guide_digest,
                            "private_pose_gate_guide_sha256": (
                                grooming_private_gate_digest
                            ),
                            "private_pose_gate_guide_path": str(private_gate_path),
                            "private_pose_gate_exposed_to_qwen": False,
                            **(
                                {
                                    "private_completion_mask_path": str(
                                        grooming_completion_mask_path
                                    ),
                                    "private_completion_mask_sha256": (
                                        grooming_completion_mask_digest
                                    ),
                                    "private_completion_mask_exposed_to_qwen": False,
                                }
                                if production_canonical_grooming
                                else {}
                            ),
                            "generator_metadata": grooming_generator_metadata,
                        }
                    }
                    if action == "grooming"
                    else {}
                ),
                # Keep the scalar policy as the backward-compatible default;
                # only the per-action map may opt eating into Candidate D.
                "pose_identity_input_policy": (
                    pipeline.POSE_IDENTITY_INPUT_POLICY_CANONICAL
                ),
                "pose_identity_input_policies": {
                    action: action_identity_policy,
                },
                "main_case_raw_identity_chain": main_case_raw_identity_chain,
                "bypass_generated_canonical": main_case_raw_identity_chain,
                "uploaded_identity_input_sha256": pipeline.file_sha256(
                    job_dir / "input.png"
                ),
                "uploaded_identity_source_sha256": pipeline.file_sha256(
                    job_dir / "input_identity_source.png"
                ),
                "identity_palette_guard": identity_palette_guard,
                "effective_subject_config": {
                    "profile_id": (
                        "sleek_tabby_cat_v1"
                        if pose_safe_grooming_attempt4
                        else "capture_fixture"
                    ),
                    "actions": {
                        action: {
                            "reference": str(action_reference),
                            "reference_sha256": pipeline.file_sha256(
                                action_reference
                            ),
                            **(
                                {
                                    "driving_reference_kind": (
                                        "uploaded_white_maquette"
                                    ),
                                    "driving_animation": str(
                                        white_maquette_driver
                                    ),
                                    "driving_animation_sha256": (
                                        pipeline.file_sha256(white_maquette_driver)
                                    ),
                                }
                                if uploaded_white_maquette
                                else {}
                            ),
                            **(
                                {"fur_surface_gate": {"enabled": True}}
                                if pose_safe_grooming_attempt4
                                else {}
                            ),
                        }
                    },
                },
                "subject_profile_fur_lock": pipeline.UPLOADED_IDENTITY_FUR_LOCK,
                "subject_profile_fur_negative": (
                    pipeline.UPLOADED_IDENTITY_FUR_NEGATIVE
                ),
            }
        }
        if not persist_identity_digest:
            jobs[job_id].pop("uploaded_identity_input_sha256", None)
        if not persist_identity_source_digest:
            jobs[job_id].pop("uploaded_identity_source_sha256", None)
        captured: dict[str, object] = {}
        queue_calls = 0
        canonical_action_palette_calls = 0
        identity_palette_signature = self._identity_palette_signature_for_test(
            {
                "black": 0.25,
                "gray": 0.25,
                "white": 0.50,
                "warm_brown_or_orange": 0.0,
                "red_or_pink": 0.0,
                "yellow_or_gold": 0.0,
                "green": 0.0,
                "blue_or_cyan": 0.0,
                "purple": 0.0,
            },
            saturated_warm_fraction=0.0,
        )
        identity_palette_quality = {
            "available": True,
            "acceptable": True,
            "warning": False,
            "issues": [],
            "warnings": [],
            "penalty": 0.0,
        }

        def fake_canonical_action_palette_quality(
            quality: dict[str, object] | None,
        ) -> dict[str, object] | None:
            nonlocal canonical_action_palette_calls
            canonical_action_palette_calls += 1
            if (
                canonical_action_palette_drift_attempts
                and canonical_action_palette_calls
                in canonical_action_palette_drift_attempts
            ):
                return {
                    "available": True,
                    "acceptable": False,
                    "warning": False,
                    "issues": [
                        "new_dominant_warm_brown_or_orange",
                        "achromatic_source_recolored_with_warm_fur",
                    ],
                    "warnings": [],
                    "penalty": 100.0,
                }
            return quality

        def fake_queue_and_wait(
            workflow: dict[str, object],
            *_args: object,
            **_kwargs: object,
        ) -> dict[str, object]:
            nonlocal queue_calls
            queue_calls += 1
            pose_queue_call = (
                queue_calls if main_case_raw_identity_chain else queue_calls - 1
            )
            if not main_case_raw_identity_chain and queue_calls == 1:
                return {}
            if pose_queue_call == pose_attempt:
                captured["workflow"] = workflow
                raise _PoseWorkflowCaptured
            return {}

        def fake_evaluate_chroma(
            _source: Path,
            target: Path,
            **_kwargs: object,
        ) -> dict[str, object]:
            target.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(_source) as loaded:
                source_size = loaded.size
            forced_acceptable_attempt = next(
                (
                    attempt
                    for attempt in (action_chroma_acceptable_attempts or set())
                    if f"attempt_{attempt:02d}" in target.name
                ),
                None,
            )
            output_size = (
                (512, 512) if forced_acceptable_attempt is not None else source_size
            )
            image = Image.new("RGBA", output_size, (0, 0, 0, 0))
            if output_size == (512, 512):
                image.paste((90, 100, 110, 255), (120, 72, 392, 440))
            else:
                image.paste((90, 100, 110, 255), (8, 7, 25, 27))
            image.save(target)
            if forced_acceptable_attempt is not None:
                return {
                    "acceptable": True,
                    "issues": [],
                    "method": "forced_direct_chroma_for_workflow_capture",
                }
            if target.name.startswith("pose_"):
                return {
                    "acceptable": False,
                    "issues": ["forced_retry_before_capture"],
                }
            return {"acceptable": True, "issues": []}

        def fake_prepare_landscape(
            _source: Path,
            target: Path,
            size: tuple[int, int],
            background_rgb: tuple[int, int, int] = (255, 255, 255),
        ) -> None:
            target.parent.mkdir(parents=True, exist_ok=True)
            image = Image.new("RGB", size, background_rgb)
            if size == (512, 512):
                image.paste((190, 200, 210), (120, 72, 392, 440))
                image.paste((90, 100, 110), (180, 120, 332, 190))
            else:
                image.paste((90, 100, 110), (8, 7, 25, 27))
            image.save(target)

        def fake_prepare_white_maquette_frame0(
            driving_animation: Path,
            action_cutout_target: Path,
            geometry_guide_target: Path,
            *,
            expected_sha256: str,
        ) -> dict[str, object]:
            self.assertEqual(driving_animation, white_maquette_driver)
            self.assertEqual(
                expected_sha256,
                pipeline.file_sha256(white_maquette_driver),
            )
            with Image.open(action_reference) as loaded:
                action_rgba = np.asarray(loaded.convert("RGBA"), dtype=np.uint8)
            Image.fromarray(action_rgba, mode="RGBA").save(action_cutout_target)
            foreground = action_rgba[:, :, 3] >= 64
            distance = cv2.distanceTransform(
                foreground.astype(np.uint8),
                cv2.DIST_L2,
                5,
            )
            distance /= max(float(distance.max()), 1.0)
            guide = np.full((512, 512, 3), (0, 255, 0), dtype=np.uint8)
            guide_luma = np.rint(96.0 + distance * 80.0).astype(np.uint8)
            for channel in range(3):
                guide[:, :, channel][foreground] = guide_luma[foreground]
            Image.fromarray(guide, mode="RGB").save(geometry_guide_target)
            return {
                "policy": "fixture_white_maquette_frame000",
                "decoded_frame_index": 0,
                "head_identity_signal_removed": True,
                "raw_frame_loaded_by_qwen": False,
                "action_cutout_path": str(action_cutout_target),
                "action_cutout_sha256": pipeline.file_sha256(action_cutout_target),
                "geometry_guide_path": str(geometry_guide_target),
                "geometry_guide_sha256": pipeline.file_sha256(
                    geometry_guide_target
                ),
            }

        def fake_grooming_candidate(
            source: Path,
            *,
            green_target: Path,
            cutout_target: Path,
            semantic_target: Path,
            normalized_cutout_target: Path,
            **_kwargs: object,
        ) -> tuple[dict[str, object], dict[str, object]]:
            green_target.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (32, 32), (0, 255, 0)).save(green_target)
            cutout = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
            cutout.paste((90, 100, 110, 255), (8, 7, 25, 27))
            cutout.save(cutout_target)
            cutout.save(semantic_target)
            normalized = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
            normalized.paste((90, 100, 110, 255), (120, 96, 392, 432))
            normalized.save(normalized_cutout_target)
            rejected_issues = (
                [
                    "raised_foreleg_edge_density_too_low",
                    "guide_chamfer_coverage_too_low",
                    "guide_chamfer_distance_too_large",
                    "raised_paw_material_inconsistent_with_planted_paw",
                ]
                if "attempt_01" in source.name
                else [
                    "raised_foreleg_edge_density_too_low",
                    "guide_chamfer_distance_too_large",
                ]
            )
            attempt_number = next(
                (
                    value
                    for value in range(1, 7)
                    if f"attempt_{value:02d}" in source.name
                ),
                0,
            )
            chroma: dict[str, object] = {
                "acceptable": True,
                "issues": [],
                "method": "rembg_u2net_fallback",
            }
            if pose_safe_grooming_attempt4 and attempt_number == 4:
                pose_transfer_seed = _kwargs.get("pose_transfer_seed")
                assert isinstance(pose_transfer_seed, Path)
                chroma.update(
                    method=(
                        pipeline.GROOMING_CANONICAL_SEED_BOUND_MATTE_ALPHA_PROVENANCE
                    ),
                    seed_bound_matte={
                        "policy": pipeline.GROOMING_CANONICAL_SEED_BOUND_MATTE_POLICY,
                        "acceptable": True,
                        "fail_closed": True,
                        "source_sha256": "a" * 64,
                        "semantic_cutout_sha256": "b" * 64,
                        "pose_transfer_seed_sha256": pipeline.file_sha256(
                            pose_transfer_seed
                        ),
                        "target_sha256": "c" * 64,
                        "union_alpha_sha256": "d" * 64,
                        "checks": {"fixture_alignment": True},
                        "pose_transfer_seed_used_for_alpha_geometry_only": True,
                        "pose_transfer_seed_rgb_copied": False,
                        "pose_transfer_seed_luma_or_chroma_copied": False,
                    },
                )
            gate: dict[str, object] = {
                    "available": True,
                    "evaluated": True,
                    "acceptable": grooming_gate_acceptable,
                    "issues": ([] if grooming_gate_acceptable else rejected_issues),
                    "penalty": (0.0 if grooming_gate_acceptable else 1000.0),
                }
            if pose_safe_grooming_attempt4 and attempt_number == 4:
                gate["metrics"] = {
                    "raised_foreleg_structural_evidence": {
                        "available": True,
                        "edge_density": 0.024736,
                    },
                    "chamfer": {
                        "available": True,
                        "coverage_within_radius": 0.833573,
                        "mean_distance_px": 3.408131,
                        "p90_distance_px": 8.595,
                    },
                }
            return chroma, gate

        def fake_fur_surface_candidate(
            _cutout: Path,
            *,
            alpha_provenance: str,
            **_kwargs: object,
        ) -> dict[str, object]:
            return {
                "available": True,
                "evaluated": True,
                "acceptable": True,
                "issues": [],
                "penalty": 0.0,
                "gate": {"alpha_provenance": alpha_provenance},
            }

        def fake_advisory_grooming_minimum_gate(
            candidate: Path,
            *,
            identity_reference_path: Path,
            identity_palette_reference_path: Path,
            **_kwargs: object,
        ) -> dict[str, object]:
            self.assertEqual(
                identity_reference_path,
                job_dir / "cartoon_reference.png",
                "the grooming action-edit proof must use the normalized canonical",
            )
            self.assertEqual(
                identity_palette_reference_path,
                job_dir / "input_identity_source.png",
                "grooming coat color must remain anchored to the raw upload",
            )
            attempt_number = next(
                (
                    value
                    for value in range(1, 7)
                    if f"attempt_{value:02d}" in candidate.name
                ),
                0,
            )
            acceptable = (
                attempt_number in advisory_grooming_minimum_acceptable_attempts
                if advisory_grooming_minimum_acceptable_attempts is not None
                else advisory_grooming_minimum_acceptable
            )
            return {
                "policy": pipeline.GROOMING_ADVISORY_MINIMUM_GATE_POLICY,
                "version": pipeline.GROOMING_ADVISORY_MINIMUM_GATE_VERSION,
                "available": True,
                "acceptable": acceptable,
                "issues": (
                    []
                    if acceptable
                    else (
                        advisory_grooming_minimum_issues
                        or [
                            "continuous_raised_foreleg_edge_too_short",
                            "raised_paw_too_far_from_muzzle",
                        ]
                    )
                ),
                "warnings": [],
                "penalty": (
                    0.0 if acceptable else 200.0
                ),
                "candidate_path": str(candidate),
                "candidate_sha256": pipeline.file_sha256(candidate),
                "identity_palette_reference_path": str(
                    identity_palette_reference_path
                ),
                "identity_palette_reference_sha256": pipeline.file_sha256(
                    identity_palette_reference_path
                ),
                "identity_reference_path": str(identity_reference_path),
                "identity_reference_sha256": pipeline.file_sha256(
                    identity_reference_path
                ),
                "identity_palette_reference_path": str(
                    identity_palette_reference_path
                ),
                "identity_palette_reference_sha256": pipeline.file_sha256(
                    identity_palette_reference_path
                ),
                "identity_palette": identity_palette_quality,
            }

        with (
            patch.multiple(
                pipeline,
                OUTPUT_ROOT=output_root,
                COMFY_INPUT=comfy_input,
                JOBS=jobs,
                QWEN_STYLE_LORA=style_lora,
                QWEN_POSE_MAX_ATTEMPTS=3,
                QWEN_BOWL_POSE_EXTRA_ATTEMPTS=2,
                GENERATION_VISUAL_AUDITS_BLOCKING=visual_audits_blocking,
                GENERATION_ACTION_CANDIDATE_GATES_ENABLED=True,
                QWEN_EATING_GEOMETRY_FIRST_STYLE_STRENGTH=style_strength,
                MODEL_ROOT=model_root,
                USER_LORA_DIR=user_lora_dir,
            ),
            patch.object(pipeline, "require_models"),
            patch.object(pipeline, "wait_comfy_idle"),
            patch.object(pipeline, "free_comfy_models"),
            patch.object(
                pipeline,
                "queue_and_wait",
                side_effect=fake_queue_and_wait,
            ),
            patch.object(
                pipeline,
                "first_output",
                return_value=canonical_source,
            ),
            patch.object(
                pipeline,
                "evaluate_chroma_cutout",
                side_effect=fake_evaluate_chroma,
            ),
            patch.object(
                pipeline,
                "pose_cutout_quality",
                return_value={
                    "available": True,
                    "acceptable": (not eating_pose_contact_unreliable),
                    "issues": (
                        ["eating_bowl_contact_unreliable"]
                        if eating_pose_contact_unreliable
                        else []
                    ),
                    "penalty": (24.0 if eating_pose_contact_unreliable else 0.0),
                },
            ),
            patch.object(
                pipeline,
                "evaluate_main_case_grooming_pose_candidate",
                side_effect=fake_grooming_candidate,
            ),
            patch.object(
                pipeline,
                "pose_similarity_quality",
                return_value={
                    "available": True,
                    "acceptable": False,
                    "warning": False,
                    "issues": ["forced_retry_before_capture"],
                    "penalty": 1000.0,
                    "comparisons": [],
                },
            ),
            patch.object(
                pipeline,
                "identity_palette_signature",
                return_value=identity_palette_signature,
            ),
            patch.object(
                pipeline,
                "identity_palette_quality",
                return_value=identity_palette_quality,
            ),
            patch.object(
                pipeline,
                "canonical_action_identity_palette_quality",
                side_effect=fake_canonical_action_palette_quality,
            ),
            patch.object(
                pipeline,
                "prepare_landscape_keyframe",
                side_effect=fake_prepare_landscape,
            ),
            patch.object(
                pipeline,
                "prepare_grooming_white_maquette_frame0_pose_inputs",
                side_effect=fake_prepare_white_maquette_frame0,
            ),
            patch.object(
                pipeline,
                "resolve_main_case_fur_surface_baseline",
                return_value=(action_reference, {}),
            ),
            patch.object(
                pipeline,
                "evaluate_main_case_fur_surface_candidate",
                side_effect=fake_fur_surface_candidate,
            ),
            patch.object(
                pipeline,
                "evaluate_advisory_grooming_minimum_gate",
                side_effect=fake_advisory_grooming_minimum_gate,
            ),
        ):
            try:
                with self.assertRaises(_PoseWorkflowCaptured):
                    pipeline.run_job(job_id, force_qwen_reference=True)
            except RuntimeError:
                self.assertEqual(
                    queue_calls,
                    0,
                    "geometry-first policy errors must fail before any Qwen queue",
                )
                raise

        copied_uploaded_input = comfy_input / relative_dir / "input.png"
        self.assertTrue(copied_uploaded_input.is_file())
        self.assertEqual(
            pipeline.file_sha256(copied_uploaded_input),
            jobs[job_id]["uploaded_identity_input_sha256"],
        )
        self.assertEqual(jobs[job_id]["started_at"], original_started_at)
        if action_identity_policy == pipeline.POSE_IDENTITY_INPUT_POLICY_CANONICAL:
            self.assertEqual(
                jobs[job_id]["pose_photography_lora_policy"],
                "disabled_after_canonical_identity_style_lock_v1",
            )
            self.assertEqual(
                jobs[job_id]["pose_builtin_style_lora_policy"],
                "disabled_after_canonical_identity_style_lock_v1",
            )
            self.assertEqual(
                jobs[job_id]["pose_effective_lora_policy"],
                "qwen_lightning_only_after_canonical_identity_style_lock_v1",
            )
            self.assertEqual(
                jobs[job_id]["pose_effective_lora_policies"][action],
                "qwen_lightning_only_after_canonical_identity_style_lock_v1",
            )
            self.assertIsInstance(jobs[job_id]["pose_lora_policy"], dict)
            self.assertEqual(
                jobs[job_id]["pose_lora_policy"]["canonical_action_override"],
                "qwen_lightning_only_after_canonical_identity_style_lock_v1",
            )
            self.assertEqual(jobs[job_id]["effective_pose_user_loras"], [])
            self.assertEqual(jobs[job_id]["pose_user_lora_strength_scale"], 0.0)
            self.assertEqual(
                jobs[job_id]["pose_candidate_normalization_policy"],
                pipeline.QWEN_ACTION_REFERENCE_NORMALIZATION_POLICY,
            )
            self.assertEqual(
                jobs[job_id]["pose_candidate_matte_policy"],
                pipeline.QWEN_ACTION_CANDIDATE_MATTE_POLICY,
            )
        workflow = captured.get("workflow")
        self.assertIsInstance(workflow, dict)
        uploaded_input = (relative_dir / "input.png").as_posix()
        pose_input = (relative_dir / pose_guide_name).as_posix()
        return workflow, uploaded_input, pose_input
