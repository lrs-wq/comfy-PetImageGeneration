from __future__ import annotations

import asyncio
import io
import json
import queue
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
from PIL import Image, ImageDraw

import fur_surface_gate as surface_gate
import local_video_api as pipeline
import transparent_animation as transparency


class _PoseWorkflowCaptured(RuntimeError):
    """Stop a mocked Qwen run immediately after its first action workflow."""


def _qwen_upload_validation(
    actions: tuple[str, ...],
    expected_pose_policies: dict[str, str] | None = None,
) -> dict[str, object]:
    resolved_pose_policies = (
        dict(expected_pose_policies)
        if expected_pose_policies is not None
        else {
            action: pipeline.FULL_DRIVER_EATING_POSE_POLICY
            for action in actions
            if action == "eating"
        }
    )
    return {
        "policy": pipeline.QWEN_FULL_BODY_FIRST_FRAME_PIPELINE,
        "qwen_full_body_required": True,
        "qwen_action_keyframes_required": True,
        "normalized_transparent_references_required": True,
        "expected_surface_gate_actions": list(actions),
        "expected_pose_policies": resolved_pose_policies,
    }


def _test_fur_surface_gate(action: str, reference: Path) -> dict[str, object]:
    digest = pipeline.file_sha256(reference)
    return {
        "enabled": True,
        "policy": pipeline.FUR_SURFACE_GATE_POLICY,
        "gate_version": pipeline.FUR_SURFACE_GATE_VERSION,
        "threshold_version": pipeline.FUR_SURFACE_THRESHOLD_VERSION,
        "scope": "main_case_guided_qwen_action_candidates",
        "reliable_inputs": ["alpha_cutout", "rembg_alpha_cutout"],
        "fail_closed": True,
        "max_consecutive_rejections": (pipeline.FUR_SURFACE_GATE_MAX_REJECTIONS),
        "baseline": {
            "enabled": True,
            "policy": pipeline.FUR_SURFACE_GATE_POLICY,
            "gate_version": pipeline.FUR_SURFACE_GATE_VERSION,
            "threshold_version": pipeline.FUR_SURFACE_THRESHOLD_VERSION,
            "action": action,
            "source": str(reference),
            "source_sha256": digest,
            "path": str(reference),
            "sha256": digest,
        },
    }


def _write_test_upload_identity(job_dir: Path) -> dict[str, str]:
    input_path = job_dir / "input.png"
    source_path = job_dir / "input_identity_source.png"
    Image.new("RGB", (32, 32), "white").save(input_path)
    Image.new("RGB", (40, 30), (224, 224, 224)).save(source_path)
    return {
        "uploaded_identity_input_sha256": pipeline.file_sha256(input_path),
        "uploaded_identity_source_sha256": pipeline.file_sha256(source_path),
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


class FurSurfaceRadialEvidenceTests(unittest.TestCase):
    @staticmethod
    def _signature(
        *,
        radial_line_fraction: float,
        radial_line_density: float,
        line_density: float,
        line_count: int,
        edge_pixel_fraction: float,
    ) -> dict[str, object]:
        return {
            "metrics": {
                "contour": {
                    "contour_tightening_removed_fraction": 0.002,
                    "silhouette_compactness": 2.2,
                },
                "texture": {
                    "coarse_gradient_p90": 0.10,
                    "structure_tensor_coherent_fraction": 0.48,
                    "fine_radial_brush_fraction": 0.37,
                    "coherent_long_line_energy": 0.10,
                    "hough": {
                        "line_count": line_count,
                        "line_density": line_density,
                        "radial_line_density": radial_line_density,
                        "radial_line_fraction": radial_line_fraction,
                        "edge_pixel_fraction": edge_pixel_fraction,
                    },
                },
            }
        }

    def test_sparse_hough_ratio_is_not_absolute_long_fur_evidence(self) -> None:
        signature = self._signature(
            radial_line_fraction=0.72059702,
            radial_line_density=0.003041,
            line_density=0.00422015,
            line_count=16,
            edge_pixel_fraction=0.00649166,
        )

        issues = surface_gate._absolute_issues(
            signature,
            surface_gate.FurSurfaceThresholds(),
        )

        self.assertNotIn("radial_long_lines", issues)

    def test_loaded_hough_radial_lines_remain_fail_closed(self) -> None:
        signature = self._signature(
            radial_line_fraction=0.7647943,
            radial_line_density=0.016554,
            line_density=0.02164735,
            line_count=76,
            edge_pixel_fraction=0.03858144,
        )

        issues = surface_gate._absolute_issues(
            signature,
            surface_gate.FurSurfaceThresholds(),
        )

        self.assertIn("radial_long_lines", issues)


class SubjectReplacementParsingTests(unittest.TestCase):
    def test_default_profile_uses_visually_accepted_action_strengths(self) -> None:
        profile = pipeline.load_subject_profile("sleek_tabby_cat_v1")

        self.assertEqual(
            profile["runtime"]["actions"]["eating"]["pose_strength"],
            1.1,
        )
        self.assertEqual(
            profile["runtime"]["actions"]["grooming"]["pose_strength"],
            1.15,
        )
        self.assertTrue(
            profile["runtime"]["actions"]["eating"]["driving_animation"].endswith(
                "20260731-main-case-v1/eating_driver_512x512_81f_16fps.mp4"
            )
        )
        self.assertTrue(
            profile["runtime"]["actions"]["grooming"]["driving_animation"].endswith(
                "20260731-main-case-v1/grooming_driver_512x512_81f_16fps.mp4"
            )
        )
        self.assertRegex(
            profile["runtime"]["actions"]["eating"]["driving_animation_sha256"],
            r"^[0-9a-f]{64}$",
        )
        self.assertRegex(
            profile["runtime"]["actions"]["grooming"]["driving_animation_sha256"],
            r"^[0-9a-f]{64}$",
        )

    def test_actions_default_to_eating_and_accept_only_open_actions(self) -> None:
        self.assertEqual(pipeline.parse_subject_actions(None), ["eating"])
        self.assertEqual(
            pipeline.parse_subject_actions('["吃饭", "舔毛", "eating"]'),
            ["eating", "grooming"],
        )
        with self.assertRaisesRegex(ValueError, "Only eating and grooming"):
            pipeline.parse_subject_actions('["running"]')

    def test_profile_exposes_action_specific_references(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            eating = root / "eat.png"
            grooming = root / "groom.png"
            eating.touch()
            grooming.touch()
            profile = {
                "id": "test_subject",
                "display_name": "测试主体",
                "preview_action": "eating",
                "action_references": {
                    "eating": str(eating),
                    "grooming": str(grooming),
                },
            }
            (root / "test_subject.json").write_text(
                json.dumps(profile, ensure_ascii=False),
                encoding="utf-8",
            )
            with patch.object(pipeline, "SUBJECT_PROFILE_DIR", root):
                loaded = pipeline.load_subject_profile("test_subject")
                public = pipeline.subject_profile_public(loaded)

        self.assertEqual(public["id"], "test_subject")
        self.assertEqual(
            [item["id"] for item in public["actions"]],
            ["eating", "grooming"],
        )
        self.assertTrue(all(item["reference_ready"] for item in public["actions"]))

    def test_scail_retry_uses_a_new_experiment_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "job-eating").mkdir()
            (root / "job-eating-retry001").mkdir()
            with patch.object(pipeline, "SCAIL_EXPERIMENT_ROOT", root):
                run_id = pipeline.available_scail_run_id("job", "eating")

        self.assertEqual(run_id, "job-eating-retry002")

    def test_uploaded_eating_scail_uses_action_pose_with_bowl_masked_out(self) -> None:
        job = {"canonical_reference_path": "/tmp/canonical-cat-only.png"}
        profile = {
            "action_references": {
                "eating": "/tmp/profile-eating.png",
                "grooming": "/tmp/profile-grooming.png",
            }
        }
        generated = {
            "eating": "/tmp/generated-eating-with-bowl.png",
            "grooming": "/tmp/generated-grooming.png",
        }

        eating_path, eating_policy = pipeline.select_scail_subject_reference(
            job,
            profile,
            generated,
            "eating",
        )
        grooming_path, grooming_policy = pipeline.select_scail_subject_reference(
            job,
            profile,
            generated,
            "grooming",
        )

        self.assertEqual(eating_path, Path("/tmp/generated-eating-with-bowl.png"))
        self.assertEqual(
            eating_policy,
            "qwen_action_identity_with_bowl_mapped_red",
        )
        self.assertEqual(grooming_path, Path("/tmp/generated-grooming.png"))
        self.assertEqual(grooming_policy, "qwen_action_specific_identity")

    def test_profile_eating_scail_uses_action_reference_with_bowl_masked_out(
        self,
    ) -> None:
        path, policy = pipeline.select_scail_subject_reference(
            {},
            {
                "action_references": {
                    "eating": "/tmp/profile-eating-with-bowl.png",
                },
            },
            None,
            "eating",
        )

        self.assertEqual(path, Path("/tmp/profile-eating-with-bowl.png"))
        self.assertEqual(
            policy,
            "subject_profile_action_identity_with_bowl_mapped_red",
        )

    def test_sleek_uploaded_eating_scail_uses_generated_upload_reference(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile_reference = root / "frozen-profile-eating.png"
            generated_reference = root / "generated-long-fur.png"
            profile_reference.write_bytes(b"profile-eating")
            generated_reference.write_bytes(b"generated-eating")
            job = {
                "effective_subject_config": {
                    "profile_id": "sleek_tabby_cat_v1",
                    "actions": {
                        "eating": {
                            "reference": str(generated_reference),
                            "reference_sha256": pipeline.file_sha256(
                                generated_reference
                            ),
                            "reference_kind": "qwen_generated_upload",
                            "profile_reference": str(profile_reference),
                            "profile_reference_sha256": pipeline.file_sha256(
                                profile_reference
                            ),
                        }
                    },
                }
            }

            path, policy = pipeline.select_scail_subject_reference(
                job,
                None,
                {"eating": str(generated_reference)},
                "eating",
            )

        self.assertEqual(path, generated_reference)
        self.assertEqual(
            policy,
            "qwen_action_identity_with_bowl_mapped_red",
        )

    def test_sleek_eating_scail_fails_closed_if_frozen_digest_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            profile_reference = Path(temporary) / "frozen-profile-eating.png"
            profile_reference.write_bytes(b"original")
            job = {
                "effective_subject_config": {
                    "profile_id": "sleek_tabby_cat_v1",
                    "actions": {
                        "eating": {
                            "reference": str(profile_reference),
                            "reference_sha256": pipeline.file_sha256(profile_reference),
                        }
                    },
                }
            }
            profile_reference.write_bytes(b"changed")

            with self.assertRaisesRegex(RuntimeError, "digest changed"):
                pipeline.select_scail_subject_reference(
                    job,
                    None,
                    None,
                    "eating",
                )

    def test_default_sleek_eating_keeps_frozen_profile_reference(
        self,
    ) -> None:
        frozen, qwen = pipeline.partition_subject_reference_actions(
            {"profile_id": "sleek_tabby_cat_v1"},
            ["eating", "grooming"],
        )

        self.assertEqual(frozen, ["eating"])
        self.assertEqual(qwen, ["grooming"])

    def test_uploaded_sleek_subject_generates_every_requested_action(
        self,
    ) -> None:
        frozen, qwen = pipeline.partition_subject_reference_actions(
            {"profile_id": "sleek_tabby_cat_v1"},
            ["eating", "grooming"],
            uploaded_identity=True,
        )

        self.assertEqual(frozen, [])
        self.assertEqual(qwen, ["eating", "grooming"])

    def test_other_profile_keeps_all_requested_qwen_actions(self) -> None:
        frozen, qwen = pipeline.partition_subject_reference_actions(
            {"profile_id": "other_cat_v1"},
            ["eating", "grooming"],
        )

        self.assertEqual(frozen, [])
        self.assertEqual(qwen, ["eating", "grooming"])


class SubjectReplacementEffectiveConfigTests(unittest.TestCase):
    def test_default_upload_surface_baselines_satisfy_frozen_contract(
        self,
    ) -> None:
        profile = pipeline.load_subject_profile("sleek_tabby_cat_v1")
        actions = ["eating", "grooming"]
        effective = pipeline.resolve_effective_subject_config(
            profile,
            actions,
            seed=None,
            scail_steps=None,
            pose_strength=None,
        )
        effective, _ = pipeline.configure_uploaded_subject_validation(
            effective,
            actions,
        )

        with tempfile.TemporaryDirectory() as temporary:
            snapshot = pipeline.snapshot_effective_subject_references(
                effective,
                Path(temporary),
                actions,
            )

        for action in actions:
            baseline = snapshot["actions"][action]["fur_surface_gate"]["baseline"]
            self.assertTrue(baseline["absolute_contract"]["acceptable"])
            self.assertEqual(
                baseline["absolute_contract"]["issues"],
                [],
            )

    def test_v2_profile_resolves_runtime_defaults_and_action_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            profile = _subject_profile_v2(Path(temporary))
            effective = pipeline.resolve_effective_subject_config(
                profile,
                ["eating", "grooming"],
                seed=None,
                scail_steps=None,
                pose_strength=None,
            )

        self.assertEqual(effective["profile_schema_version"], 2)
        self.assertEqual(effective["runtime_source"], "profile.runtime")
        self.assertEqual(effective["engine_profile"], "scail2-14b-fp8-v1")
        self.assertEqual(effective["prompt_profile"], "sleek-tabby-scail-v1")
        self.assertEqual(effective["model"], pipeline.SCAIL_UNET)
        self.assertEqual(effective["frames"], 81)
        self.assertEqual(effective["fps"], 16.0)
        self.assertEqual(effective["steps"], 6)
        self.assertEqual(effective["base_seed"], 20260726)
        self.assertEqual(effective["seed_mode"], "base_plus_action_offset")
        self.assertEqual(effective["actions"]["eating"]["effective_seed"], 20260726)
        self.assertEqual(effective["actions"]["grooming"]["effective_seed"], 20260728)
        self.assertEqual(effective["actions"]["eating"]["pose_strength"], 1.1)
        self.assertEqual(effective["actions"]["grooming"]["pose_strength"], 1.15)
        self.assertEqual(
            effective["actions"]["eating"]["driving_animation_sha256"],
            profile["runtime"]["actions"]["eating"]["driving_animation_sha256"],
        )
        self.assertEqual(
            effective["profile_fur_constraints"]["positive"],
            "frozen close-lying test coat",
        )
        self.assertTrue(
            effective["actions"]["eating"]["fur_silhouette_gate"]["enabled"]
        )

    def test_changed_driving_animation_digest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            profile = _subject_profile_v2(Path(temporary))
            driving = Path(profile["runtime"]["actions"]["eating"]["driving_animation"])
            driving.write_bytes(b"changed-driving-animation")

            with self.assertRaisesRegex(
                ValueError,
                "eating driving animation digest changed",
            ):
                pipeline.resolve_effective_subject_config(
                    profile,
                    ["eating"],
                    seed=None,
                    scail_steps=None,
                    pose_strength=None,
                )

    def test_eating_pose_strength_above_sleek_limit_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            profile = _subject_profile_v2(Path(temporary))
            with self.assertRaisesRegex(
                ValueError,
                "eating pose_strength must not exceed 1.35",
            ):
                pipeline.resolve_effective_subject_config(
                    profile,
                    ["eating"],
                    seed=None,
                    scail_steps=None,
                    pose_strength=2.0,
                )

    def test_eating_only_upload_uses_qwen_reference_and_never_profile_fallback(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_id = "test-sleek-eating-bypass"
            job_dir = root / job_id
            job_dir.mkdir()
            reference = root / "eating-profile.png"
            generated_reference = root / "qwen-eating.png"
            reference.write_bytes(b"frozen-eating")
            generated_reference.write_bytes(b"qwen-eating")
            original_started_at = "2026-08-03T00:00:00+00:00"
            jobs = {
                job_id: {
                    "job_id": job_id,
                    "actions": ["eating"],
                    "subject_source": "upload_pending_qwen",
                    "started_at": original_started_at,
                    "progress": 0,
                    "expected_pose_policies": {
                        "eating": pipeline.FULL_DRIVER_EATING_POSE_POLICY,
                    },
                    "effective_subject_config": {
                        "profile_id": "sleek_tabby_cat_v1",
                        "frames": 81,
                        "fps": 16.0,
                        "steps": 6,
                        "upload_validation": _qwen_upload_validation(("eating",)),
                        "actions": {
                            "eating": {
                                "reference": str(reference),
                                "reference_sha256": pipeline.file_sha256(reference),
                                "driving_animation": str(reference),
                                "pose_strength": 1.1,
                                "effective_seed": 123,
                                "fur_surface_gate": _test_fur_surface_gate(
                                    "eating",
                                    reference,
                                ),
                            }
                        },
                    },
                }
            }

            def fake_prepare(_job_id: str, qwen_actions: list[str]) -> None:
                self.assertEqual(qwen_actions, ["eating"])
                updated = json.loads(
                    json.dumps(jobs[job_id]["effective_subject_config"])
                )
                eating = updated["actions"]["eating"]
                eating["profile_reference"] = eating["reference"]
                eating["profile_reference_sha256"] = eating["reference_sha256"]
                eating["reference"] = str(generated_reference)
                eating["reference_sha256"] = pipeline.file_sha256(generated_reference)
                eating["reference_kind"] = "qwen_generated_upload"
                pipeline.update_job(
                    job_id,
                    subject_source="qwen_generated_upload",
                    subject_reference_paths={
                        "eating": str(generated_reference),
                    },
                    qwen_subject_reference_paths={
                        "eating": str(generated_reference),
                    },
                    reference_generation_completed=True,
                    stage="subject_references_ready",
                    progress=35,
                    effective_subject_config=updated,
                )

            progress_updates: list[float] = []
            started_at_updates: list[str] = []
            route_events: list[tuple[str, str]] = []
            real_update_job = pipeline.update_job

            def recording_update_job(
                target_job_id: str,
                **changes: object,
            ) -> dict[str, object]:
                if "progress" in changes:
                    progress_updates.append(float(changes["progress"]))
                if "started_at" in changes:
                    started_at_updates.append(str(changes["started_at"]))
                return real_update_job(target_job_id, **changes)

            def record_revalidation(*_args: object, **_kwargs: object) -> None:
                route_events.append(
                    ("revalidate", str(jobs[job_id].get("stage") or ""))
                )

            def record_wait(*_args: object, **_kwargs: object) -> None:
                route_events.append(("wait", str(jobs[job_id].get("stage") or "")))

            def stop_at_scail(*_args: object, **_kwargs: object) -> None:
                route_events.append(("scail", str(jobs[job_id].get("stage") or "")))
                raise RuntimeError("stop-at-scail")

            with (
                patch.multiple(
                    pipeline,
                    OUTPUT_ROOT=root,
                    JOBS=jobs,
                    SCAIL_EXPERIMENT_ROOT=root / "scail-experiments",
                ),
                patch.object(
                    pipeline,
                    "update_job",
                    side_effect=recording_update_job,
                ),
                patch.object(
                    pipeline,
                    "prepare_uploaded_subject_qwen_references",
                    side_effect=fake_prepare,
                ) as prepare_qwen,
                patch.object(
                    pipeline,
                    "revalidate_uploaded_qwen_references_before_scail",
                    side_effect=record_revalidation,
                ) as revalidate_qwen,
                patch.object(pipeline, "require_subject_replacement_assets"),
                patch.object(
                    pipeline,
                    "wait_comfy_idle",
                    side_effect=record_wait,
                ),
                patch.object(pipeline, "free_comfy_models"),
                patch.object(
                    pipeline,
                    "run_cancelable_process",
                    side_effect=stop_at_scail,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "stop-at-scail"):
                    pipeline.run_subject_replacement_job(job_id)

            prepare_qwen.assert_called_once_with(job_id, ["eating"])
            self.assertEqual(revalidate_qwen.call_count, 2)
            self.assertEqual(
                route_events,
                [
                    ("revalidate", "subject_references_ready"),
                    ("wait", "waiting_for_gpu"),
                    ("revalidate", "preparing_subject_replacement"),
                    ("scail", "preparing_subject_replacement"),
                ],
            )
            self.assertEqual(progress_updates, [35.0, 36.0, 38.0])
            self.assertTrue(
                all(
                    later >= earlier
                    for earlier, later in zip(
                        progress_updates,
                        progress_updates[1:],
                    )
                )
            )
            self.assertEqual(started_at_updates, [original_started_at])
            final_job = jobs[job_id]
            self.assertEqual(final_job["started_at"], original_started_at)
            self.assertEqual(
                final_job["subject_source"],
                "qwen_generated_upload",
            )
            self.assertEqual(
                final_job["effective_subject_source"],
                "qwen_generated_upload",
            )
            self.assertEqual(
                final_job["reference_pipeline"],
                pipeline.QWEN_FULL_BODY_FIRST_FRAME_PIPELINE,
            )
            self.assertEqual(
                final_job["subject_reference_paths"],
                {"eating": str(generated_reference)},
            )
            self.assertEqual(
                final_job["scail_reference_policy"],
                {
                    "eating": "qwen_action_identity_with_bowl_mapped_red",
                },
            )
            self.assertEqual(
                final_job["uploaded_identity_applied_actions"],
                ["eating"],
            )
            self.assertEqual(
                final_job["uploaded_identity_excluded_actions"],
                [],
            )

    def test_mixed_upload_uses_qwen_identity_for_both_actions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_id = "test-sleek-mixed-references"
            job_dir = root / job_id
            job_dir.mkdir()
            eating_reference = root / "eating-profile.png"
            grooming_reference = root / "grooming-profile.png"
            qwen_eating = root / "qwen-eating.png"
            qwen_grooming = root / "qwen-grooming.png"
            eating_reference.write_bytes(b"frozen-eating")
            grooming_reference.write_bytes(b"frozen-grooming")
            qwen_eating.write_bytes(b"qwen-eating")
            qwen_grooming.write_bytes(b"qwen-grooming")
            jobs = {
                job_id: {
                    "job_id": job_id,
                    "actions": ["eating", "grooming"],
                    "subject_source": "upload_pending_qwen",
                    "expected_pose_policies": {
                        "eating": pipeline.FULL_DRIVER_EATING_POSE_POLICY,
                    },
                    "effective_subject_config": {
                        "profile_id": "sleek_tabby_cat_v1",
                        "upload_validation": _qwen_upload_validation(
                            ("eating", "grooming")
                        ),
                        "actions": {
                            "eating": {
                                "reference": str(eating_reference),
                                "reference_sha256": pipeline.file_sha256(
                                    eating_reference
                                ),
                                "fur_surface_gate": _test_fur_surface_gate(
                                    "eating",
                                    eating_reference,
                                ),
                            },
                            "grooming": {
                                "reference": str(grooming_reference),
                                "reference_sha256": pipeline.file_sha256(
                                    grooming_reference
                                ),
                                "fur_surface_gate": _test_fur_surface_gate(
                                    "grooming",
                                    grooming_reference,
                                ),
                            },
                        },
                    },
                }
            }

            def fake_prepare(_job_id: str, qwen_actions: list[str]) -> None:
                self.assertEqual(qwen_actions, ["eating", "grooming"])
                updated = json.loads(
                    json.dumps(jobs[job_id]["effective_subject_config"])
                )
                eating = updated["actions"]["eating"]
                grooming = updated["actions"]["grooming"]
                for action, generated in (
                    (eating, qwen_eating),
                    (grooming, qwen_grooming),
                ):
                    action["profile_reference"] = action["reference"]
                    action["profile_reference_sha256"] = action["reference_sha256"]
                    action["reference"] = str(generated)
                    action["reference_sha256"] = pipeline.file_sha256(generated)
                    action["reference_kind"] = "qwen_generated_upload"
                pipeline.update_job(
                    job_id,
                    subject_source="qwen_generated_upload",
                    subject_reference_paths={
                        "eating": str(qwen_eating),
                        "grooming": str(qwen_grooming),
                    },
                    qwen_subject_reference_paths={
                        "eating": str(qwen_eating),
                        "grooming": str(qwen_grooming),
                    },
                    reference_generation_completed=True,
                    effective_subject_config=updated,
                )

            with (
                patch.multiple(pipeline, OUTPUT_ROOT=root, JOBS=jobs),
                patch.object(
                    pipeline,
                    "prepare_uploaded_subject_qwen_references",
                    side_effect=fake_prepare,
                ) as prepare_qwen,
                patch.object(
                    pipeline,
                    "revalidate_uploaded_qwen_references_before_scail",
                ),
                patch.object(
                    pipeline,
                    "require_subject_replacement_assets",
                    side_effect=RuntimeError("stop-before-gpu"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "stop-before-gpu"):
                    pipeline.run_subject_replacement_job(job_id)

            prepare_qwen.assert_called_once_with(
                job_id,
                ["eating", "grooming"],
            )
            final_job = jobs[job_id]
            self.assertEqual(
                final_job["reference_pipeline"],
                pipeline.QWEN_FULL_BODY_FIRST_FRAME_PIPELINE,
            )
            self.assertEqual(
                final_job["subject_reference_paths"],
                {
                    "eating": str(qwen_eating),
                    "grooming": str(qwen_grooming),
                },
            )
            self.assertEqual(
                final_job["qwen_reference_applied_actions"],
                ["eating", "grooming"],
            )
            self.assertEqual(
                final_job["qwen_reference_bypassed_actions"],
                [],
            )
            self.assertEqual(
                final_job["uploaded_identity_applied_actions"],
                ["eating", "grooming"],
            )
            self.assertEqual(
                final_job["uploaded_identity_excluded_actions"],
                [],
            )

    def test_uploaded_subject_fails_closed_before_gpu_without_qwen_identity(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_id = "test-upload-no-fallback"
            (root / job_id).mkdir()
            reference = root / "profile.png"
            reference.write_bytes(b"profile")
            jobs = {
                job_id: {
                    "job_id": job_id,
                    "actions": ["eating"],
                    "subject_source": "upload_pending_qwen",
                    "expected_pose_policies": {
                        "eating": pipeline.FULL_DRIVER_EATING_POSE_POLICY,
                    },
                    "effective_subject_config": {
                        "profile_id": "sleek_tabby_cat_v1",
                        "upload_validation": _qwen_upload_validation(("eating",)),
                        "actions": {
                            "eating": {
                                "reference": str(reference),
                                "reference_sha256": pipeline.file_sha256(reference),
                                "fur_surface_gate": _test_fur_surface_gate(
                                    "eating",
                                    reference,
                                ),
                            }
                        },
                    },
                }
            }

            with (
                patch.multiple(pipeline, OUTPUT_ROOT=root, JOBS=jobs),
                patch.object(
                    pipeline,
                    "prepare_uploaded_subject_qwen_references",
                ),
                patch.object(
                    pipeline,
                    "require_subject_replacement_assets",
                ) as require_assets,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Uploaded subject identity was not applied",
                ):
                    pipeline.run_subject_replacement_job(job_id)

            require_assets.assert_not_called()

    def test_upload_without_surface_gate_fails_before_qwen(self) -> None:
        job_id = "test-upload-without-surface-gate"
        jobs = {
            job_id: {
                "job_id": job_id,
                "actions": ["eating"],
                "subject_source": "upload_pending_qwen",
                "expected_pose_policies": {
                    "eating": pipeline.FULL_DRIVER_EATING_POSE_POLICY,
                },
                "effective_subject_config": {
                    "profile_id": "sleek_tabby_cat_v1",
                    "upload_validation": _qwen_upload_validation(("eating",)),
                    "actions": {"eating": {}},
                },
            }
        }
        with (
            patch.object(pipeline, "JOBS", jobs),
            patch.object(
                pipeline,
                "GENERATION_VISUAL_AUDITS_BLOCKING",
                True,
            ),
            patch.object(
                pipeline,
                "prepare_uploaded_subject_qwen_references",
            ) as prepare_qwen,
        ):
            with self.assertRaisesRegex(RuntimeError, "fur surface gate"):
                pipeline.run_subject_replacement_job(job_id)

        prepare_qwen.assert_not_called()

    def test_retired_direct_upload_states_require_resubmission(self) -> None:
        for retired_state in (
            "upload_pending_scail_reference",
            "uploaded_identity_direct_scail",
        ):
            with self.subTest(retired_state=retired_state):
                job_id = f"test-retired-{retired_state}"
                jobs = {
                    job_id: {
                        "job_id": job_id,
                        "actions": ["eating"],
                        "subject_source": retired_state,
                        "effective_subject_config": {
                            "profile_id": "sleek_tabby_cat_v1",
                            "actions": {"eating": {}},
                        },
                    }
                }
                with (
                    patch.object(pipeline, "JOBS", jobs),
                    patch.object(
                        pipeline,
                        "prepare_uploaded_subject_qwen_references",
                    ) as prepare_qwen,
                    patch.object(
                        pipeline,
                        "prepare_uploaded_subject_scail_references",
                    ) as prepare_direct,
                ):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "Submit the upload again",
                    ):
                        pipeline.run_subject_replacement_job(job_id)

                prepare_qwen.assert_not_called()
                prepare_direct.assert_not_called()

        with self.assertRaisesRegex(
            RuntimeError,
            "Direct uploaded-image SCAIL references are retired",
        ):
            pipeline.prepare_uploaded_subject_scail_references(
                "retired-direct-builder",
                ["eating"],
            )

    def test_reference_snapshot_copies_content_and_records_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = _subject_profile_v2(root)
            effective = pipeline.resolve_effective_subject_config(
                profile,
                ["eating", "grooming"],
                seed=None,
                scail_steps=None,
                pose_strength=None,
            )
            job_dir = root / "job"
            snapshot = pipeline.snapshot_effective_subject_references(
                effective,
                job_dir,
                ["eating", "grooming"],
            )
            eating_source = Path(snapshot["actions"]["eating"]["reference_source"])
            eating_snapshot = Path(snapshot["actions"]["eating"]["reference"])
            frozen_bytes = eating_snapshot.read_bytes()
            Image.new("RGB", (32, 32), "blue").save(eating_source)

            self.assertTrue(eating_snapshot.is_file())
            self.assertEqual(eating_snapshot.parent, job_dir / "profile_inputs")
            self.assertEqual(eating_snapshot.read_bytes(), frozen_bytes)
            self.assertNotEqual(eating_source.read_bytes(), frozen_bytes)
            self.assertEqual(
                snapshot["actions"]["eating"]["reference_sha256"],
                pipeline.file_sha256(eating_snapshot),
            )
            baseline_snapshot = Path(
                snapshot["actions"]["eating"]["fur_contour_baseline"]["path"]
            )
            self.assertTrue(baseline_snapshot.is_file())
            self.assertEqual(baseline_snapshot.parent, job_dir / "profile_inputs")
            self.assertEqual(
                snapshot["actions"]["eating"]["fur_contour_baseline"]["sha256"],
                pipeline.file_sha256(baseline_snapshot),
            )

    def test_worker_reference_selection_is_fixed_against_profile_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = _subject_profile_v2(root)
            effective = pipeline.resolve_effective_subject_config(
                profile,
                ["eating"],
                seed=None,
                scail_steps=None,
                pose_strength=None,
            )
            snapshot = pipeline.snapshot_effective_subject_references(
                effective,
                root / "job",
                ["eating"],
            )
            replacement = root / "changed-after-creation.png"
            Image.new("RGB", (32, 32), "red").save(replacement)
            profile["action_references"]["eating"] = str(replacement)
            profile["fur_constraints"]["positive"] = "mutated live coat rule"

            selected, _ = pipeline.select_scail_subject_reference(
                {"effective_subject_config": snapshot},
                profile,
                None,
                "eating",
            )

        self.assertEqual(selected, Path(snapshot["actions"]["eating"]["reference"]))
        self.assertNotEqual(selected, replacement)
        frozen_positive, _ = pipeline.subject_profile_fur_guard(
            {
                "id": snapshot["profile_id"],
                "fur_constraints": snapshot["profile_fur_constraints"],
            }
        )
        self.assertIn("uploaded pet", frozen_positive.lower())
        self.assertNotIn("frozen close-lying test coat", frozen_positive)
        self.assertNotIn("mutated live coat rule", frozen_positive)

    def test_uploaded_fur_guard_excludes_profile_animal_identity_terms(self) -> None:
        positive, negative = pipeline.subject_profile_fur_guard(
            {
                "id": "sleek_tabby_cat_v1",
                "fur_constraints": {
                    "positive": "exact Main_case black-and-white coat identity",
                    "negative": "old gray tabby identity, green eyes",
                },
            }
        )

        combined = f"{positive}, {negative}".lower()
        self.assertIn("uploaded identity appearance authority", combined)
        self.assertIn("action timing and geometry only", combined)
        self.assertNotIn("exact main_case black-and-white", combined)
        self.assertNotIn("old gray tabby identity", combined)
        self.assertNotIn("green eyes", combined)

    def test_uploaded_fur_guard_accepts_only_explicit_upload_extension(self) -> None:
        positive, negative = pipeline.subject_profile_fur_guard(
            {
                "fur_constraints": {
                    "positive": "profile identity must stay out",
                },
                "upload_fur_constraints": {
                    "positive": "keep source whisker length",
                    "negative": "invented whisker fan",
                },
            }
        )

        self.assertIn("keep source whisker length", positive)
        self.assertIn("invented whisker fan", negative)
        self.assertNotIn("profile identity must stay out", positive)


class CanonicalChromaIntegrityTests(unittest.TestCase):
    def test_canonical_cutout_rejects_large_central_neck_hole(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            clean_path = root / "clean.png"
            hole_path = root / "hole.png"
            clean = Image.new("RGBA", (200, 300), (0, 0, 0, 0))
            clean_draw = ImageDraw.Draw(clean)
            clean_draw.ellipse((25, 20, 175, 155), fill=(120, 120, 120, 255))
            clean_draw.rectangle((50, 110, 150, 280), fill=(120, 120, 120, 255))
            clean.save(clean_path)
            hole = clean.copy()
            ImageDraw.Draw(hole).rectangle((70, 112, 130, 130), fill=(0, 0, 0, 0))
            hole.save(hole_path)

            clean_quality = pipeline.canonical_cutout_interior_quality(clean_path)
            hole_quality = pipeline.canonical_cutout_interior_quality(hole_path)

        self.assertTrue(clean_quality["acceptable"], clean_quality)
        self.assertFalse(hole_quality["acceptable"])
        self.assertIn(
            "canonical_subject_interior_chroma_hole",
            hole_quality["issues"],
        )
        self.assertGreater(hole_quality["largest_void_area_fraction"], 0.004)

    def test_chroma_despill_preserves_green_eye_inside_subject(self) -> None:
        rgb = np.zeros((96, 96, 3), dtype=np.uint8)
        rgb[:, :] = (0, 255, 0)
        rgb[12:84, 24:72] = (120, 120, 120)
        rgb[46:51, 46:51] = (30, 160, 30)
        rgb[15:18, 30:34] = (120, 170, 120)

        cleaned, alpha = transparency._subject_mask(rgb.copy())

        self.assertGreaterEqual(int(alpha[48, 48]), 128)
        self.assertEqual(tuple(cleaned[48, 48]), (30, 160, 30))
        self.assertGreaterEqual(int(alpha[16, 31]), 128)
        self.assertLessEqual(
            int(cleaned[16, 31, 1]),
            max(int(cleaned[16, 31, 0]), int(cleaned[16, 31, 2])),
        )


class SubjectReplacementJobCreationTests(unittest.TestCase):
    def test_default_profile_job_needs_no_uploaded_image(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_root = root / "outputs"
            jobs: dict[str, dict[str, object]] = {}
            job_queue: queue.Queue[str] = queue.Queue(maxsize=4)
            profile = _subject_profile_v2(root)
            with (
                patch.multiple(
                    pipeline,
                    OUTPUT_ROOT=output_root,
                    JOBS=jobs,
                    JOB_QUEUE=job_queue,
                ),
                patch.object(pipeline, "load_subject_profile", return_value=profile),
                patch.object(pipeline, "require_subject_replacement_assets"),
            ):
                result = asyncio.run(
                    pipeline.create_subject_replacement_api_job(
                        image=None,
                        prompt="",
                        actions_raw='["eating", "grooming"]',
                        subject_profile_id="sleek_tabby_cat_v1",
                        output_format="transparent_gif",
                        scail_steps=6,
                        pose_strength=None,
                        seed=123,
                    )
                )
            standard_actions = result["effective_subject_config"]["standard_direct"][
                "actions"
            ]
            expected_references = {
                action: standard_actions[action]["selected_reference"]["cutout"]
                for action in ("eating", "grooming")
            }
            self.assertEqual(result["subject_reference_paths"], expected_references)
            self.assertTrue(
                all(Path(path).is_file() for path in expected_references.values())
            )
            self.assertEqual(result["standard_direct_actions"], standard_actions)

        self.assertEqual(result["generation_mode"], "subject_replacement")
        self.assertEqual(result["actions"], ["eating", "grooming"])
        self.assertEqual(result["subject_source"], "profile")
        self.assertEqual(result["requested_subject_source"], "profile")
        self.assertEqual(result["effective_subject_source"], "profile")
        self.assertEqual(result["subject_profile"], "sleek_tabby_cat_v1")
        self.assertEqual(result["frames"], 100)
        self.assertEqual(result["fps"], 20.0)
        self.assertEqual(result["seed"], 123)
        self.assertEqual(result["seed_mode"], "base_plus_action_offset")
        self.assertEqual(
            result["effective_action_seeds"], {"eating": 123, "grooming": 125}
        )
        self.assertEqual(result["source_frames"], 81)
        self.assertEqual(result["source_fps"], 16.0)
        self.assertEqual(
            result["reference_pipeline"],
            pipeline.PROFILE_STANDARD_DIRECT_POLICY,
        )
        self.assertIsNone(result["reference_pipeline_version"])
        self.assertEqual(
            result["reference_generation_status"],
            "standard_direct_ready",
        )
        self.assertEqual(
            result["generation_engine"],
            "deterministic-main-case-standard-direct",
        )
        self.assertEqual(
            result["video_engine"],
            "deterministic-main-case-standard-direct",
        )
        self.assertEqual(
            result["visual_style"],
            "main_case_profile_standard_direct",
        )
        self.assertEqual(
            result["standard_direct_policy"],
            pipeline.PROFILE_STANDARD_DIRECT_POLICY,
        )
        self.assertEqual(
            result["standard_direct_role"],
            pipeline.PROFILE_STANDARD_DIRECT_ROLE,
        )
        self.assertEqual(
            result["standard_direct_revision"],
            "test-main-case-standard-v1",
        )
        self.assertFalse(result["gpu_required"])
        self.assertFalse(result["gpu_submitted"])
        self.assertEqual(
            result["gpu_bypass_reason"],
            pipeline.PROFILE_STANDARD_DIRECT_GPU_REASON,
        )
        self.assertFalse(result["generation_controls_applied"])
        self.assertIsNone(result["scail_model"])
        self.assertIsNone(result["scail_dpo_lora"])
        self.assertIsNone(result["scail_distill_lora"])
        self.assertIsNone(result["scail_steps"])
        self.assertIsNone(result["reference_image_profile"])
        self.assertIsNone(result["reference_generation_engine"])
        self.assertIsNone(result["reference_normalization_contract"])
        self.assertFalse(result["qwen_reference_generation_for_subject_replacement"])
        self.assertFalse(result["canonical_reference_generated"])
        self.assertFalse(result["canonical_reference_used"])
        self.assertEqual(set(result["subject_reference_paths"]), {"eating", "grooming"})
        self.assertEqual(
            result["subject_profile_sha256"],
            result["effective_subject_config"]["profile_sha256"],
        )

    def test_uploaded_photo_is_never_registered_as_a_scail_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_root = root / "outputs"
            jobs: dict[str, dict[str, object]] = {}
            job_queue: queue.Queue[str] = queue.Queue(maxsize=4)
            profile = _subject_profile_v2(root)

            def normalize_upload(
                _raw: bytes,
                target: Path,
                *,
                identity_source_path: Path,
            ) -> None:
                target.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (32, 32), "white").save(target)
                Image.new("RGB", (32, 32), "white").save(identity_source_path)

            upload = pipeline.UploadFile(
                filename="ordinary-photo.jpg",
                file=io.BytesIO(b"ordinary-photo"),
            )
            with (
                patch.multiple(
                    pipeline,
                    OUTPUT_ROOT=output_root,
                    JOBS=jobs,
                    JOB_QUEUE=job_queue,
                ),
                patch.object(pipeline, "load_subject_profile", return_value=profile),
                patch.object(pipeline, "require_subject_replacement_assets"),
                patch.object(
                    pipeline, "normalize_uploaded_image", side_effect=normalize_upload
                ),
            ):
                result = asyncio.run(
                    pipeline.create_subject_replacement_api_job(
                        image=upload,
                        prompt="",
                        actions_raw='["eating", "grooming"]',
                        subject_profile_id="sleek_tabby_cat_v1",
                        output_format="transparent_gif",
                        scail_steps=6,
                        pose_strength=None,
                        seed=123,
                    )
                )
            profile_snapshots = {
                action: Path(
                    result["effective_subject_config"]["actions"][action]["reference"]
                )
                for action in ("eating", "grooming")
            }
            self.assertTrue(
                all(snapshot.is_file() for snapshot in profile_snapshots.values())
            )
            self.assertTrue(
                all(
                    snapshot.parent.name == "profile_inputs"
                    for snapshot in profile_snapshots.values()
                )
            )
            self.assertNotEqual(
                profile_snapshots["eating"],
                Path(profile["action_references"]["eating"]),
            )
            self.assertNotEqual(
                profile_snapshots["grooming"],
                Path(profile["action_references"]["grooming"]),
            )

        self.assertEqual(result["subject_source"], "upload_pending_qwen")
        self.assertEqual(result["requested_subject_source"], "upload")
        self.assertEqual(
            result["effective_subject_source"],
            "upload_pending_qwen",
        )
        self.assertEqual(result["subject_reference_paths"], {})
        self.assertEqual(
            result["reference_pipeline"],
            pipeline.QWEN_FULL_BODY_FIRST_FRAME_PIPELINE,
        )
        self.assertEqual(
            result["reference_pipeline_version"],
            pipeline.QWEN_FULL_BODY_FIRST_FRAME_PIPELINE_VERSION,
        )
        self.assertFalse(
            result["reference_normalization_contract"]["raw_upload_sent_to_scail"]
        )
        self.assertEqual(result["reference_image_profile"], "pet-multipose-20260717-v2")

    def test_three_view_upload_is_persisted_with_role_and_digest_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_root = root / "outputs"
            jobs: dict[str, dict[str, object]] = {}
            job_queue: queue.Queue[str] = queue.Queue(maxsize=4)
            profile = _subject_profile_v2(root)
            colors = {
                b"front": (64, 64, 64),
                b"left": (96, 96, 96),
                b"right": (128, 128, 128),
            }

            def normalize_upload(
                raw: bytes,
                target: Path,
                *,
                identity_source_path: Path,
            ) -> None:
                target.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (32, 32), colors[raw]).save(target)
                Image.new("RGBA", (40, 30), (*colors[raw], 255)).save(
                    identity_source_path
                )

            upload_files = {
                role: io.BytesIO(role.encode("ascii"))
                for role in ("front", "left", "right")
            }
            uploads = {
                role: pipeline.UploadFile(
                    filename=f"pet-{role}.jpg",
                    file=upload_files[role],
                )
                for role in upload_files
            }
            with (
                patch.multiple(
                    pipeline,
                    OUTPUT_ROOT=output_root,
                    JOBS=jobs,
                    JOB_QUEUE=job_queue,
                ),
                patch.object(pipeline, "load_subject_profile", return_value=profile),
                patch.object(pipeline, "require_subject_replacement_assets"),
                patch.object(
                    pipeline,
                    "normalize_uploaded_image",
                    side_effect=normalize_upload,
                ),
            ):
                result = asyncio.run(
                    pipeline.create_subject_replacement_api_job(
                        image=uploads["front"],
                        image_left=uploads["left"],
                        image_right=uploads["right"],
                        prompt="",
                        actions_raw='["grooming"]',
                        subject_profile_id="sleek_tabby_cat_v1",
                        output_format="transparent_gif",
                        scail_steps=6,
                        pose_strength=None,
                        seed=123,
                    )
                )
                job_dir = output_root / result["job_id"]
                verified = pipeline.verify_uploaded_identity_submission(
                    result,
                    job_dir,
                )

            self.assertEqual(result["uploaded_identity_view_mode"], "front_left_right")
            self.assertEqual(result["uploaded_identity_view_count"], 3)
            self.assertEqual(
                result["uploaded_identity_view_roles"],
                ["front", "left", "right"],
            )
            self.assertEqual(
                result["uploaded_identity_view_policy"],
                pipeline.UPLOADED_IDENTITY_VIEW_POLICY,
            )
            self.assertEqual(set(verified["views"]), {"front", "left", "right"})
            self.assertEqual(
                result["uploaded_identity_input_sha256"],
                result["uploaded_identity_views"]["front"]["input_sha256"],
            )
            self.assertTrue(
                result["reference_normalization_contract"][
                    "identity_views_used_only_for_qwen_canonical"
                ]
            )
            self.assertFalse(
                result["reference_normalization_contract"][
                    "action_stage_raw_identity_views_used"
                ]
            )
            self.assertTrue(all(stream.closed for stream in upload_files.values()))

    def test_partial_three_view_upload_is_rejected_and_all_files_are_closed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = _subject_profile_v2(root)
            front_file = io.BytesIO(b"front")
            left_file = io.BytesIO(b"left")
            front = pipeline.UploadFile(filename="front.jpg", file=front_file)
            left = pipeline.UploadFile(filename="left.jpg", file=left_file)
            with patch.object(
                pipeline,
                "load_subject_profile",
                return_value=profile,
            ):
                with self.assertRaises(pipeline.HTTPException) as raised:
                    asyncio.run(
                        pipeline.create_subject_replacement_api_job(
                            image=front,
                            image_left=left,
                            prompt="",
                            actions_raw='["grooming"]',
                            subject_profile_id="sleek_tabby_cat_v1",
                            output_format="transparent_gif",
                            scail_steps=6,
                            pose_strength=None,
                            seed=123,
                        )
                    )
            self.assertEqual(raised.exception.status_code, 422)
            self.assertIn("complete image", raised.exception.detail)
            self.assertTrue(front_file.closed)
            self.assertTrue(left_file.closed)

    def test_three_view_integrity_rejects_changed_side_source(self) -> None:
        def png_bytes(color: tuple[int, int, int]) -> bytes:
            buffer = io.BytesIO()
            Image.new("RGB", (32, 24), color).save(buffer, format="PNG")
            return buffer.getvalue()

        with tempfile.TemporaryDirectory() as temporary:
            job_dir = Path(temporary)
            records = pipeline.persist_uploaded_identity_views(
                job_dir,
                {
                    "front": png_bytes((40, 40, 40)),
                    "left": png_bytes((80, 80, 80)),
                    "right": png_bytes((120, 120, 120)),
                },
            )
            job = {
                "uploaded_identity_input_sha256": records["front"]["input_sha256"],
                "uploaded_identity_source_sha256": records["front"]["source_sha256"],
                "uploaded_identity_view_policy": pipeline.UPLOADED_IDENTITY_VIEW_POLICY,
                "uploaded_identity_view_mode": "front_left_right",
                "uploaded_identity_view_count": 3,
                "uploaded_identity_view_roles": ["front", "left", "right"],
                "uploaded_identity_views": records,
            }
            (job_dir / "input_left_identity_source.png").write_bytes(b"changed")
            with self.assertRaisesRegex(
                RuntimeError,
                "identity left source is missing or changed",
            ):
                pipeline.verify_uploaded_identity_submission(job, job_dir)

    def test_distinct_three_view_set_passes_real_palette_and_angle_audit(self) -> None:
        def view_bytes(role: str) -> bytes:
            image = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
            draw = ImageDraw.Draw(image)
            if role == "front":
                draw.ellipse((35, 10, 93, 118), fill=(82, 82, 82, 255))
                draw.ellipse((42, 8, 86, 50), fill=(205, 205, 205, 255))
            elif role == "left":
                draw.ellipse((8, 40, 116, 102), fill=(88, 88, 88, 255))
                draw.ellipse((82, 24, 122, 66), fill=(200, 200, 200, 255))
            else:
                draw.polygon(
                    [(14, 105), (35, 30), (92, 12), (118, 76), (100, 116)],
                    fill=(92, 92, 92, 255),
                )
                draw.ellipse((18, 18, 58, 58), fill=(198, 198, 198, 255))
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            return buffer.getvalue()

        with tempfile.TemporaryDirectory() as temporary:
            job_dir = Path(temporary)
            records = pipeline.persist_uploaded_identity_views(
                job_dir,
                {role: view_bytes(role) for role in ("front", "left", "right")},
            )
            job = {
                "uploaded_identity_input_sha256": records["front"]["input_sha256"],
                "uploaded_identity_source_sha256": records["front"]["source_sha256"],
                "uploaded_identity_view_policy": pipeline.UPLOADED_IDENTITY_VIEW_POLICY,
                "uploaded_identity_view_mode": "front_left_right",
                "uploaded_identity_view_count": 3,
                "uploaded_identity_view_roles": ["front", "left", "right"],
                "uploaded_identity_views": records,
            }
            integrity = pipeline.verify_uploaded_identity_submission(job, job_dir)
            audit = pipeline.audit_uploaded_identity_view_set(integrity)

        self.assertTrue(audit["acceptable"])
        self.assertEqual(audit["issues"], [])
        self.assertEqual(set(audit["palettes"]), {"front", "left", "right"})
        self.assertTrue(all(item["available"] for item in audit["palettes"].values()))
        self.assertTrue(
            all(
                not comparison["near_duplicate"]
                for comparison in audit["view_angle_comparisons"].values()
            )
        )


class SubjectReplacementQwenReferenceTests(unittest.TestCase):
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

    def test_canonical_completion_accepts_only_antialias_edge_shortfall(
        self,
    ) -> None:
        raw_gate = self._masked_pet3d_composite_pose_gate()
        raw_gate["metrics"]["raised_foreleg_edge_density"] = 0.108621
        result = pipeline.apply_canonical_grooming_completion_edge_evidence_gate(
            raw_gate,
            self._canonical_completion_repair_audit(),
        )

        self.assertFalse(raw_gate["acceptable"])
        self.assertTrue(result["acceptable"])
        self.assertEqual(
            result["verdict"],
            "accepted_canonical_masked_completion_edge_evidence",
        )
        recovery = result["canonical_completion_edge_evidence"]
        self.assertTrue(recovery["accepted"])
        self.assertFalse(recovery["global_pose_threshold_changed"])
        self.assertEqual(recovery["required_global_edge_density_min"], 0.11)
        self.assertAlmostEqual(recovery["edge_density_floor"], 0.10835)
        self.assertTrue(recovery["independent_fur_surface_gate_still_required"])
        self.assertTrue(all(recovery["checks"].values()))

    def test_canonical_direct_accepts_only_strict_organic_edge_evidence(
        self,
    ) -> None:
        raw_gate = self._masked_pet3d_composite_pose_gate()
        raw_gate["metrics"]["raised_foreleg_edge_density"] = 0.102601
        result = pipeline.apply_canonical_grooming_direct_structural_edge_evidence_gate(
            raw_gate,
            attempt=2,
            canonical_action_reference=True,
            canonical_completion_active=False,
            canonical_full_pose_recovery_active=False,
        )

        self.assertFalse(raw_gate["acceptable"])
        self.assertTrue(result["acceptable"])
        self.assertEqual(
            result["verdict"],
            "accepted_canonical_direct_organic_foreleg_structural_evidence",
        )
        recovery = result["canonical_direct_structural_edge_evidence"]
        self.assertTrue(recovery["accepted"])
        self.assertFalse(recovery["global_pose_threshold_changed"])
        self.assertEqual(recovery["required_global_edge_density_min"], 0.11)
        self.assertAlmostEqual(recovery["edge_density_floor"], 0.10175)
        self.assertEqual(
            recovery["required_structural_edge_density_min"],
            0.03,
        )
        self.assertTrue(recovery["identity_palette_gate_still_required"])
        self.assertTrue(recovery["independent_fur_surface_gate_still_required"])
        self.assertTrue(all(recovery["checks"].values()))

    def test_canonical_direct_accepts_bounded_offguide_real_fur_paw(
        self,
    ) -> None:
        raw_gate = self._masked_pet3d_composite_pose_gate()
        raw_gate["issues"] = [
            "guide_chamfer_coverage_too_low",
            "guide_chamfer_distance_too_large",
        ]
        raw_gate["metrics"]["raised_foreleg_edge_density"] = 0.124191
        raw_gate["metrics"]["raised_foreleg_structural_evidence"][
            "edge_density"
        ] = 0.027961
        raw_gate["metrics"]["chamfer"].update(
            coverage_within_radius=0.597262,
            mean_distance_px=10.209558,
            p90_distance_px=34.748634,
        )
        raw_gate["metrics"]["continuous_edge"]["span_fraction"] = 0.240234
        raw_gate["metrics"]["paw_muzzle_contact"][
            "gap_fraction_of_bbox_height"
        ] = 0.025391
        raw_gate["metrics"]["appearance_consistency"].update(
            lab_median_delta=4.0,
            lab_histogram_bhattacharyya={"mean": 0.122506},
            texture_gradient_p90_ratio=1.709433,
        )

        pose_result = (
            pipeline.apply_canonical_grooming_direct_structural_edge_evidence_gate(
                raw_gate,
                attempt=1,
                canonical_action_reference=True,
                canonical_completion_active=False,
                canonical_full_pose_recovery_active=False,
            )
        )
        self.assertTrue(pose_result["acceptable"])
        self.assertEqual(
            pose_result["verdict"],
            "accepted_canonical_direct_offguide_organic_paw_evidence",
        )
        offguide = pose_result["canonical_direct_structural_edge_evidence"][
            "offguide_organic_paw_evidence"
        ]
        self.assertTrue(offguide["accepted"])
        self.assertTrue(all(offguide["checks"].values()))

        palette_quality = {
            "available": True,
            "acceptable": False,
            "issues": ["canonical_action_achromatic_value_balance_drift"],
            "warnings": [],
            "penalty": 135.1123,
            "source_achromatic_fraction": 0.984794,
            "candidate_achromatic_fraction": 0.999041,
            "achromatic_balance_l1": 0.686885,
            "broad_warm_drift": False,
            "saturated_warm_drift": False,
            "achromatic_dark_marking_lost": False,
            "achromatic_dark_marking_expanded": False,
        }
        palette_probe = (
            pipeline.apply_canonical_grooming_organic_paw_palette_evidence(
                palette_quality,
                pose_result,
                attempt=1,
            )
        )
        self.assertFalse(palette_probe["acceptable"])
        self.assertEqual(
            palette_probe["issues"],
            ["canonical_action_achromatic_value_balance_drift"],
        )
        eligibility = pipeline.canonical_grooming_palette_normalization_eligibility(
            palette_quality,
            pose_result,
            attempt=1,
        )
        self.assertTrue(eligibility["eligible"])
        self.assertTrue(all(eligibility["checks"].values()))
        self.assertEqual(
            eligibility["operation"],
            "foreground_lab_l_histogram_match_only",
        )
        self.assertTrue(eligibility["pose_gate_must_be_re_evaluated"])
        self.assertTrue(eligibility["identity_palette_gate_must_be_re_evaluated"])
        self.assertTrue(eligibility["fur_surface_gate_must_run_afterward"])

    def test_canonical_direct_aligned_real_paw_is_palette_normalization_eligible(
        self,
    ) -> None:
        raw_gate = self._masked_pet3d_composite_pose_gate()
        raw_gate.update(
            acceptable=True,
            verdict="passed",
            issues=[],
            penalty=0.0,
        )
        raw_gate["metrics"]["raised_foreleg_edge_density"] = 0.181210
        raw_gate["metrics"]["raised_foreleg_structural_evidence"][
            "edge_density"
        ] = 0.052462
        raw_gate["metrics"]["chamfer"].update(
            coverage_within_radius=0.972622,
            mean_distance_px=1.734,
            p90_distance_px=3.82,
        )
        raw_gate["metrics"]["continuous_edge"]["span_fraction"] = 0.292969
        raw_gate["metrics"]["paw_muzzle_contact"][
            "gap_fraction_of_bbox_height"
        ] = 0.027344
        raw_gate["metrics"]["appearance_consistency"].update(
            lab_median_delta=15.0,
            lab_histogram_bhattacharyya={"mean": 0.070902},
            texture_gradient_p90_ratio=1.190325,
        )

        pose_result = (
            pipeline.apply_canonical_grooming_direct_structural_edge_evidence_gate(
                raw_gate,
                attempt=1,
                canonical_action_reference=True,
                canonical_completion_active=False,
                canonical_full_pose_recovery_active=False,
            )
        )
        self.assertTrue(pose_result["acceptable"])
        self.assertEqual(
            pose_result["verdict"],
            "accepted_canonical_direct_aligned_organic_paw_evidence",
        )
        aligned = pose_result["canonical_direct_structural_edge_evidence"][
            "aligned_organic_paw_evidence"
        ]
        self.assertTrue(aligned["accepted"])
        self.assertTrue(all(aligned["checks"].values()))

        eligibility = pipeline.canonical_grooming_palette_normalization_eligibility(
            {
                "available": True,
                "acceptable": False,
                "issues": ["canonical_action_achromatic_value_balance_drift"],
                "warnings": [],
                "penalty": 176.9294,
                "source_achromatic_fraction": 0.984348,
                "candidate_achromatic_fraction": 0.997349,
                "achromatic_balance_l1": 0.884647,
                "broad_warm_drift": False,
                "saturated_warm_drift": False,
                "achromatic_dark_marking_lost": False,
                "achromatic_dark_marking_expanded": False,
            },
            pose_result,
            attempt=1,
        )
        self.assertTrue(eligibility["eligible"])
        self.assertEqual(eligibility["achromatic_balance_l1"], 0.884647)

    def test_offguide_palette_evidence_rejects_hanging_prop_metrics(
        self,
    ) -> None:
        raw_gate = self._masked_pet3d_composite_pose_gate()
        raw_gate["issues"] = [
            "guide_chamfer_coverage_too_low",
            "guide_chamfer_distance_too_large",
        ]
        raw_gate["metrics"]["raised_foreleg_edge_density"] = 0.136218
        raw_gate["metrics"]["raised_foreleg_structural_evidence"][
            "edge_density"
        ] = 0.028803
        raw_gate["metrics"]["chamfer"].update(
            coverage_within_radius=0.791787,
            mean_distance_px=4.409295,
            p90_distance_px=13.98372,
        )
        raw_gate["metrics"]["appearance_consistency"].update(
            lab_median_delta=15.532225,
            lab_histogram_bhattacharyya={"mean": 0.127734},
            texture_gradient_p90_ratio=2.250538,
        )
        pose_result = (
            pipeline.apply_canonical_grooming_direct_structural_edge_evidence_gate(
                raw_gate,
                attempt=1,
                canonical_action_reference=True,
                canonical_completion_active=False,
                canonical_full_pose_recovery_active=False,
            )
        )
        self.assertFalse(pose_result["acceptable"])
        self.assertFalse(
            pose_result["canonical_direct_structural_edge_evidence"][
                "offguide_organic_paw_evidence"
            ]["accepted"]
        )

    def test_aligned_palette_evidence_rejects_hanging_prop_metrics(
        self,
    ) -> None:
        raw_gate = self._masked_pet3d_composite_pose_gate()
        raw_gate.update(
            acceptable=True,
            verdict="passed",
            issues=[],
            penalty=0.0,
        )
        raw_gate["metrics"]["raised_foreleg_edge_density"] = 0.136218
        raw_gate["metrics"]["raised_foreleg_structural_evidence"][
            "edge_density"
        ] = 0.028803
        raw_gate["metrics"]["chamfer"].update(
            coverage_within_radius=0.791787,
            mean_distance_px=4.409295,
            p90_distance_px=13.98372,
        )
        raw_gate["metrics"]["appearance_consistency"].update(
            lab_median_delta=15.532225,
            lab_histogram_bhattacharyya={"mean": 0.127734},
            texture_gradient_p90_ratio=2.250538,
        )
        pose_result = (
            pipeline.apply_canonical_grooming_direct_structural_edge_evidence_gate(
                raw_gate,
                attempt=1,
                canonical_action_reference=True,
                canonical_completion_active=False,
                canonical_full_pose_recovery_active=False,
            )
        )
        aligned = pose_result["canonical_direct_structural_edge_evidence"][
            "aligned_organic_paw_evidence"
        ]
        self.assertFalse(aligned["accepted"])
        self.assertFalse(aligned["checks"]["lowpass_organic_structure_strong"])
        self.assertFalse(aligned["checks"]["strict_organic_guide_alignment_passed"])
        self.assertFalse(aligned["checks"]["raised_planted_paw_fur_match"])
        eligibility = pipeline.canonical_grooming_palette_normalization_eligibility(
            {
                "available": True,
                "acceptable": False,
                "issues": ["canonical_action_achromatic_value_balance_drift"],
                "warnings": [],
                "penalty": 100.0,
                "source_achromatic_fraction": 0.99,
                "candidate_achromatic_fraction": 0.99,
                "achromatic_balance_l1": 0.50,
                "broad_warm_drift": False,
                "saturated_warm_drift": False,
                "achromatic_dark_marking_lost": False,
                "achromatic_dark_marking_expanded": False,
            },
            pose_result,
            attempt=1,
        )
        self.assertFalse(eligibility["eligible"])

    def test_grooming_palette_normalization_preserves_geometry_and_chroma(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate_path = root / "candidate.png"
            canonical_path = root / "canonical.png"
            target_path = root / "normalized.png"

            yy, xx = np.mgrid[:512, :512]
            candidate_foreground = (
                ((xx - 256.0) / 145.0) ** 2
                + ((yy - 270.0) / 190.0) ** 2
                <= 1.0
            )
            canonical_foreground = (
                ((xx - 250.0) / 170.0) ** 2
                + ((yy - 255.0) / 175.0) ** 2
                <= 1.0
            )
            candidate_luma = np.clip(45 + (yy * 90 // 511), 0, 255).astype(
                np.uint8
            )
            canonical_luma = np.clip(150 + (xx * 95 // 511), 0, 255).astype(
                np.uint8
            )
            candidate_rgba = np.zeros((512, 512, 4), dtype=np.uint8)
            candidate_rgba[candidate_foreground, :3] = np.repeat(
                candidate_luma[:, :, None],
                3,
                axis=2,
            )[candidate_foreground]
            candidate_rgba[candidate_foreground, 3] = 255
            canonical_rgba = np.zeros((512, 512, 4), dtype=np.uint8)
            canonical_rgba[canonical_foreground, :3] = np.repeat(
                canonical_luma[:, :, None],
                3,
                axis=2,
            )[canonical_foreground]
            canonical_rgba[canonical_foreground, 3] = 255
            Image.fromarray(candidate_rgba, mode="RGBA").save(candidate_path)
            Image.fromarray(canonical_rgba, mode="RGBA").save(canonical_path)

            audit = (
                pipeline.prepare_canonical_grooming_achromatic_palette_normalization(
                    candidate_path,
                    canonical_path,
                    target_path,
                )
            )
            with Image.open(target_path) as loaded:
                normalized_rgba = np.asarray(loaded.convert("RGBA"), dtype=np.uint8)

            self.assertTrue(
                np.array_equal(normalized_rgba[:, :, 3], candidate_rgba[:, :, 3])
            )
            self.assertTrue(audit["source_alpha_geometry_preserved_exactly"])
            self.assertLessEqual(audit["maximum_ab_roundtrip_delta"], 2)
            self.assertTrue(audit["luma_lut_monotonic"])
            self.assertGreater(audit["source_luma_changed_pixels"], 0)
            source_quantiles = np.asarray(audit["source_luma_p10_p50_p90"])
            canonical_quantiles = np.asarray(audit["canonical_luma_p10_p50_p90"])
            output_quantiles = np.asarray(audit["output_luma_p10_p50_p90"])
            self.assertLess(
                float(np.abs(output_quantiles - canonical_quantiles).sum()),
                float(np.abs(source_quantiles - canonical_quantiles).sum()),
            )
            self.assertNotEqual(audit["source_sha256"], audit["target_sha256"])
            self.assertFalse(audit["canonical_spatial_pixels_copied"])

    def test_canonical_direct_structural_evidence_fails_closed(
        self,
    ) -> None:
        for case in (
            "edge_below_band",
            "weak_structural_density",
            "weak_chamfer_coverage",
            "large_chamfer_mean",
            "large_chamfer_p90",
            "extra_pose_issue",
            "repair_attempt",
            "completion_active",
            "full_pose_recovery_active",
            "global_threshold_changed",
            "historical_straight_prop",
        ):
            with self.subTest(case=case):
                raw_gate = self._masked_pet3d_composite_pose_gate()
                raw_gate["metrics"]["raised_foreleg_edge_density"] = 0.102601
                attempt = 2
                completion_active = False
                full_pose_recovery_active = False
                if case == "edge_below_band":
                    raw_gate["metrics"]["raised_foreleg_edge_density"] = 0.101
                elif case == "weak_structural_density":
                    raw_gate["metrics"]["raised_foreleg_structural_evidence"][
                        "edge_density"
                    ] = 0.029
                elif case == "weak_chamfer_coverage":
                    raw_gate["metrics"]["chamfer"]["coverage_within_radius"] = 0.699
                elif case == "large_chamfer_mean":
                    raw_gate["metrics"]["chamfer"]["mean_distance_px"] = 5.001
                elif case == "large_chamfer_p90":
                    raw_gate["metrics"]["chamfer"]["p90_distance_px"] = 12.001
                elif case == "extra_pose_issue":
                    raw_gate["issues"].append("guide_chamfer_coverage_too_low")
                elif case == "repair_attempt":
                    attempt = 3
                elif case == "completion_active":
                    completion_active = True
                elif case == "full_pose_recovery_active":
                    full_pose_recovery_active = True
                elif case == "global_threshold_changed":
                    raw_gate["thresholds"]["edge_density_min"] = 0.10
                elif case == "historical_straight_prop":
                    raw_gate["metrics"]["raised_foreleg_edge_density"] = 0.103003
                    raw_gate["metrics"]["raised_foreleg_structural_evidence"][
                        "edge_density"
                    ] = 0.035563
                    raw_gate["metrics"]["chamfer"].update(
                        coverage_within_radius=0.632565,
                        mean_distance_px=6.996345,
                        p90_distance_px=19.7,
                    )

                result = pipeline.apply_canonical_grooming_direct_structural_edge_evidence_gate(
                    raw_gate,
                    attempt=attempt,
                    canonical_action_reference=True,
                    canonical_completion_active=completion_active,
                    canonical_full_pose_recovery_active=(full_pose_recovery_active),
                )

                self.assertFalse(result["acceptable"])
                self.assertFalse(
                    result["canonical_direct_structural_edge_evidence"]["accepted"]
                )

    def test_canonical_direct_surface_accepts_only_nonexpanding_debt(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "candidate.png"
            canonical = root / "canonical.png"
            candidate.write_bytes(b"candidate")
            canonical.write_bytes(b"canonical")
            raw_pose = self._masked_pet3d_composite_pose_gate()
            raw_pose["candidate"] = str(candidate)
            raw_pose["metrics"]["raised_foreleg_edge_density"] = 0.102601
            pose_gate = (
                pipeline.apply_canonical_grooming_direct_structural_edge_evidence_gate(
                    raw_pose,
                    attempt=2,
                    canonical_action_reference=True,
                    canonical_completion_active=False,
                    canonical_full_pose_recovery_active=False,
                )
            )
            surface = self._canonical_direct_surface_gate(candidate)
            with patch.object(
                pipeline,
                "evaluate_fur_surface",
                return_value=self._canonical_surface_debt_audit(),
            ):
                result = (
                    pipeline.apply_canonical_grooming_direct_surface_nonexpansion_gate(
                        surface,
                        pose_gate=pose_gate,
                        identity_palette_quality={
                            "available": True,
                            "acceptable": True,
                            "issues": [],
                        },
                        attempt=2,
                        canonical_reference_path=canonical,
                        canonical_reference_sha256=(pipeline.file_sha256(canonical)),
                    )
                )

        self.assertFalse(surface["acceptable"])
        self.assertTrue(result["acceptable"])
        self.assertEqual(
            result["verdict"],
            "accepted_canonical_direct_surface_debt_nonexpansion",
        )
        self.assertEqual(result["issues"], [])
        self.assertEqual(result["candidate_absolute_issues"], [])
        recovery = result["canonical_direct_surface_nonexpansion"]
        self.assertTrue(recovery["accepted"])
        self.assertFalse(recovery["global_surface_thresholds_changed"])
        self.assertTrue(recovery["independent_final_video_fur_audit_still_required"])
        self.assertTrue(all(recovery["checks"].values()))

    def test_canonical_direct_surface_nonexpansion_fails_closed(
        self,
    ) -> None:
        for case in (
            "extra_surface_issue",
            "pose_not_recovered",
            "identity_failed",
            "digest_changed",
            "canonical_debt_absent",
            "coherent_fraction_expanded",
            "hough_radial_expanded",
            "canonical_debt_unbounded",
            "global_threshold_changed",
            "repair_attempt",
        ):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                candidate = root / "candidate.png"
                canonical = root / "canonical.png"
                candidate.write_bytes(b"candidate")
                canonical.write_bytes(b"canonical")
                raw_pose = self._masked_pet3d_composite_pose_gate()
                raw_pose["candidate"] = str(candidate)
                raw_pose["metrics"]["raised_foreleg_edge_density"] = 0.102601
                pose_gate = pipeline.apply_canonical_grooming_direct_structural_edge_evidence_gate(
                    raw_pose,
                    attempt=2,
                    canonical_action_reference=True,
                    canonical_completion_active=False,
                    canonical_full_pose_recovery_active=False,
                )
                surface = self._canonical_direct_surface_gate(candidate)
                palette = {
                    "available": True,
                    "acceptable": True,
                    "issues": [],
                }
                canonical_audit = self._canonical_surface_debt_audit()
                digest = pipeline.file_sha256(canonical)
                attempt = 2
                if case == "extra_surface_issue":
                    surface["issues"].append("coherent_long_fur_energy")
                    surface["candidate_absolute_issues"].append(
                        "coherent_long_fur_energy"
                    )
                elif case == "pose_not_recovered":
                    pose_gate = raw_pose
                elif case == "identity_failed":
                    palette.update(
                        acceptable=False,
                        issues=["achromatic_coat_value_balance_lost"],
                    )
                elif case == "digest_changed":
                    digest = "0" * 64
                elif case == "canonical_debt_absent":
                    canonical_audit["issues"] = []
                elif case == "coherent_fraction_expanded":
                    surface["candidate"]["metrics"]["texture"][
                        "structure_tensor_coherent_fraction"
                    ] = 0.57
                elif case == "hough_radial_expanded":
                    surface["candidate"]["metrics"]["texture"]["hough"][
                        "radial_line_fraction"
                    ] = 0.50
                elif case == "canonical_debt_unbounded":
                    canonical_audit["candidate"]["metrics"]["texture"][
                        "coherent_long_line_energy"
                    ] = 0.251
                elif case == "global_threshold_changed":
                    surface["thresholds"]["coarse_gradient_p90_max"] = 0.25
                elif case == "repair_attempt":
                    attempt = 3

                with patch.object(
                    pipeline,
                    "evaluate_fur_surface",
                    return_value=canonical_audit,
                ):
                    result = pipeline.apply_canonical_grooming_direct_surface_nonexpansion_gate(
                        surface,
                        pose_gate=pose_gate,
                        identity_palette_quality=palette,
                        attempt=attempt,
                        canonical_reference_path=canonical,
                        canonical_reference_sha256=digest,
                    )

                self.assertFalse(result["acceptable"])
                self.assertFalse(
                    result["canonical_direct_surface_nonexpansion"]["accepted"]
                )

    def test_canonical_completion_edge_evidence_fails_closed(
        self,
    ) -> None:
        for case in (
            "edge_below_band",
            "extra_pose_issue",
            "global_threshold_changed",
            "style_lora_enabled",
            "identity_seed_inserted",
        ):
            with self.subTest(case=case):
                raw_gate = self._masked_pet3d_composite_pose_gate()
                raw_gate["metrics"]["raised_foreleg_edge_density"] = 0.108621
                repair_audit = self._canonical_completion_repair_audit()
                if case == "edge_below_band":
                    raw_gate["metrics"]["raised_foreleg_edge_density"] = 0.108
                elif case == "extra_pose_issue":
                    raw_gate["issues"].append("guide_chamfer_coverage_too_low")
                elif case == "global_threshold_changed":
                    raw_gate["thresholds"]["edge_density_min"] = 0.10
                elif case == "style_lora_enabled":
                    repair_audit["localized_style_lora"] = self.TEST_QWEN_STYLE_LORA
                elif case == "identity_seed_inserted":
                    repair_audit["completion_canvas"][
                        "identity_or_geometry_seed_inserted"
                    ] = True

                result = (
                    pipeline.apply_canonical_grooming_completion_edge_evidence_gate(
                        raw_gate,
                        repair_audit,
                    )
                )

                self.assertFalse(result["acceptable"])
                self.assertFalse(
                    result["canonical_completion_edge_evidence"]["accepted"]
                )

    def test_masked_pet3d_composite_accepts_only_surface_verified_edge_shortfall(
        self,
    ) -> None:
        raw_gate = self._masked_pet3d_composite_pose_gate()
        fur_gate = self._masked_pet3d_composite_fur_gate()
        with patch.object(
            pipeline,
            "QWEN_STYLE_LORA",
            self.TEST_QWEN_STYLE_LORA,
        ):
            result = pipeline.apply_masked_pet3d_grooming_composite_gate(
                raw_gate,
                fur_gate,
                self._masked_pet3d_composite_repair_audit(),
            )

        self.assertFalse(raw_gate["acceptable"])
        self.assertTrue(result["acceptable"])
        self.assertEqual(
            result["verdict"],
            "accepted_masked_pet3d_surface_composite",
        )
        self.assertEqual(
            result["raw_issues"],
            ["raised_foreleg_edge_density_too_low"],
        )
        recovery = result["composite_recovery"]
        self.assertTrue(recovery["accepted"])
        self.assertFalse(recovery["global_pose_threshold_changed"])
        self.assertEqual(recovery["required_global_edge_density_min"], 0.11)
        self.assertTrue(all(recovery["checks"].values()))

    def test_masked_pet3d_composite_accepts_low_texture_planted_paw_ratio_artifact(
        self,
    ) -> None:
        raw_gate = self._masked_pet3d_composite_pose_gate()
        raw_gate["issues"].append("raised_paw_texture_inconsistent_with_planted_paw")
        raw_gate["metrics"]["raised_foreleg_edge_density"] = 0.082643
        appearance = raw_gate["metrics"]["appearance_consistency"]
        appearance.update(
            lab_median_delta=0.0,
            lab_histogram_bhattacharyya={"mean": 0.320373},
            texture_gradient_p90_ratio=18.029094,
            raised={"gradient_p90": 93.52005},
            planted={"gradient_p90": 4.24264},
        )

        with patch.object(
            pipeline,
            "QWEN_STYLE_LORA",
            self.TEST_QWEN_STYLE_LORA,
        ):
            result = pipeline.apply_masked_pet3d_grooming_composite_gate(
                raw_gate,
                self._masked_pet3d_composite_fur_gate(),
                self._masked_pet3d_composite_repair_audit(),
            )

        self.assertTrue(result["acceptable"])
        recovery = result["composite_recovery"]
        self.assertTrue(recovery["accepted"])
        self.assertTrue(
            recovery["paw_texture_evidence"]["low_texture_planted_paw_rescue_used"]
        )
        self.assertEqual(
            result["raw_issues"],
            [
                "raised_foreleg_edge_density_too_low",
                "raised_paw_texture_inconsistent_with_planted_paw",
            ],
        )

    def test_masked_pet3d_composite_accepts_surface_verified_seventy_percent_edge_floor(
        self,
    ) -> None:
        raw_gate = self._masked_pet3d_composite_pose_gate()
        raw_gate["metrics"]["raised_foreleg_edge_density"] = 0.077316

        with patch.object(
            pipeline,
            "QWEN_STYLE_LORA",
            self.TEST_QWEN_STYLE_LORA,
        ):
            result = pipeline.apply_masked_pet3d_grooming_composite_gate(
                raw_gate,
                self._masked_pet3d_composite_fur_gate(),
                self._masked_pet3d_composite_repair_audit(),
            )

        self.assertTrue(result["acceptable"])
        recovery = result["composite_recovery"]
        self.assertTrue(recovery["accepted"])
        self.assertAlmostEqual(recovery["edge_density_floor"], 0.077)
        self.assertTrue(recovery["checks"]["edge_density_within_recovery_band"])

    def test_masked_pet3d_composite_fails_closed_outside_exact_contract(
        self,
    ) -> None:
        cases = (
            "additional_pose_issue",
            "fur_surface_rejected",
            "edge_density_below_recovery_band",
            "global_edge_threshold_changed",
            "repair_model_mismatch",
            "raised_paw_texture_unbounded",
            "planted_paw_not_low_texture",
        )
        for case in cases:
            with self.subTest(case=case):
                pose_gate = self._masked_pet3d_composite_pose_gate()
                fur_gate = self._masked_pet3d_composite_fur_gate()
                repair_audit = self._masked_pet3d_composite_repair_audit()
                if case == "additional_pose_issue":
                    pose_gate["issues"].append(
                        "raised_paw_color_inconsistent_with_planted_paw"
                    )
                elif case == "fur_surface_rejected":
                    pose_gate["metrics"]["raised_foreleg_edge_density"] = 0.077179
                    fur_gate.update(
                        acceptable=False,
                        status="rejected",
                        issues=[
                            "brush_texture_coherence",
                            "radial_long_lines",
                        ],
                    )
                elif case == "edge_density_below_recovery_band":
                    pose_gate["metrics"]["raised_foreleg_edge_density"] = 0.076
                elif case == "global_edge_threshold_changed":
                    pose_gate["thresholds"]["edge_density_min"] = 0.10
                elif case == "repair_model_mismatch":
                    repair_audit["localized_style_lora"] = (
                        "photography-style.safetensors"
                    )
                elif case in {
                    "raised_paw_texture_unbounded",
                    "planted_paw_not_low_texture",
                }:
                    pose_gate["issues"].append(
                        "raised_paw_texture_inconsistent_with_planted_paw"
                    )
                    appearance = pose_gate["metrics"]["appearance_consistency"]
                    appearance["texture_gradient_p90_ratio"] = 18.0
                    appearance["planted"]["gradient_p90"] = (
                        4.0 if case == "raised_paw_texture_unbounded" else 12.0
                    )
                    appearance["raised"]["gradient_p90"] = (
                        160.0 if case == "raised_paw_texture_unbounded" else 93.0
                    )

                with patch.object(
                    pipeline,
                    "QWEN_STYLE_LORA",
                    self.TEST_QWEN_STYLE_LORA,
                ):
                    result = pipeline.apply_masked_pet3d_grooming_composite_gate(
                        pose_gate,
                        fur_gate,
                        repair_audit,
                    )

                self.assertFalse(result["acceptable"])
                self.assertFalse(result["composite_recovery"]["accepted"])

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

    def test_achromatic_palette_gate_allows_stylized_shadow_to_white_rebin(
        self,
    ) -> None:
        source = self._identity_palette_signature_for_test(
            {
                "black": 0.148978,
                "white": 0.014032,
                "gray": 0.747331,
                "warm_brown_or_orange": 0.083535,
                "red_or_pink": 0.004411,
                "yellow_or_gold": 0.000117,
                "green": 0.0,
                "blue_or_cyan": 0.001596,
                "purple": 0.0,
            },
            saturated_warm_fraction=0.0023,
        )
        candidate = self._identity_palette_signature_for_test(
            {
                "black": 0.308152,
                "white": 0.253738,
                "gray": 0.267391,
                "warm_brown_or_orange": 0.048198,
                "red_or_pink": 0.069917,
                "yellow_or_gold": 0.007358,
                "green": 0.034427,
                "blue_or_cyan": 0.01082,
                "purple": 0.0,
            },
            saturated_warm_fraction=0.083097,
        )

        quality = pipeline.identity_palette_quality(source, candidate)

        self.assertTrue(quality["acceptable"])
        self.assertGreater(quality["achromatic_balance_l1"], 0.85)
        self.assertFalse(quality["achromatic_value_balance_lost"])
        self.assertFalse(quality["saturated_warm_drift"])

    def test_canonical_action_palette_gate_keeps_valid_value_redistribution(
        self,
    ) -> None:
        source = self._identity_palette_signature_for_test(
            {
                "black": 0.04,
                "white": 0.52,
                "gray": 0.42,
                "warm_brown_or_orange": 0.0,
                "red_or_pink": 0.0,
                "yellow_or_gold": 0.0,
                "green": 0.01,
                "blue_or_cyan": 0.01,
                "purple": 0.0,
            },
            saturated_warm_fraction=0.0,
        )
        candidate = self._identity_palette_signature_for_test(
            {
                "black": 0.04,
                "white": 0.36,
                "gray": 0.58,
                "warm_brown_or_orange": 0.0,
                "red_or_pink": 0.0,
                "yellow_or_gold": 0.0,
                "green": 0.01,
                "blue_or_cyan": 0.01,
                "purple": 0.0,
            },
            saturated_warm_fraction=0.0,
        )

        quality = pipeline.canonical_action_identity_palette_quality(
            pipeline.identity_palette_quality(source, candidate)
        )

        self.assertTrue(quality["acceptable"])
        self.assertLessEqual(
            quality["achromatic_balance_l1"],
            pipeline.CANONICAL_ACTION_ACHROMATIC_BALANCE_L1_MAX,
        )
        self.assertFalse(quality["canonical_action_achromatic_balance_drift"])

    def test_canonical_action_palette_gate_rejects_gray_geometry_copy(
        self,
    ) -> None:
        source = self._identity_palette_signature_for_test(
            {
                "black": 0.04,
                "white": 0.52,
                "gray": 0.42,
                "warm_brown_or_orange": 0.0,
                "red_or_pink": 0.0,
                "yellow_or_gold": 0.0,
                "green": 0.01,
                "blue_or_cyan": 0.01,
                "purple": 0.0,
            },
            saturated_warm_fraction=0.0,
        )
        candidate = self._identity_palette_signature_for_test(
            {
                "black": 0.01,
                "white": 0.26,
                "gray": 0.71,
                "warm_brown_or_orange": 0.0,
                "red_or_pink": 0.0,
                "yellow_or_gold": 0.0,
                "green": 0.01,
                "blue_or_cyan": 0.01,
                "purple": 0.0,
            },
            saturated_warm_fraction=0.0,
        )

        general_quality = pipeline.identity_palette_quality(source, candidate)
        quality = pipeline.canonical_action_identity_palette_quality(general_quality)

        self.assertTrue(general_quality["acceptable"])
        self.assertFalse(quality["acceptable"])
        self.assertIn(
            "canonical_action_achromatic_value_balance_drift",
            quality["issues"],
        )
        self.assertGreater(
            quality["achromatic_balance_l1"],
            pipeline.CANONICAL_ACTION_ACHROMATIC_BALANCE_L1_MAX,
        )

    def test_multi_view_palette_consensus_uses_equal_view_weight(self) -> None:
        front = self._identity_palette_signature_for_test(
            {
                "black": 0.6,
                "white": 0.4,
                "gray": 0.0,
                "warm_brown_or_orange": 0.0,
                "red_or_pink": 0.0,
                "yellow_or_gold": 0.0,
                "green": 0.0,
                "blue_or_cyan": 0.0,
                "purple": 0.0,
            },
            saturated_warm_fraction=0.0,
        )
        side = self._identity_palette_signature_for_test(
            {
                "black": 0.3,
                "white": 0.7,
                "gray": 0.0,
                "warm_brown_or_orange": 0.0,
                "red_or_pink": 0.0,
                "yellow_or_gold": 0.0,
                "green": 0.0,
                "blue_or_cyan": 0.0,
                "purple": 0.0,
            },
            saturated_warm_fraction=0.0,
        )
        consensus = pipeline.aggregate_identity_palette_signatures(
            {"front": front, "left": side, "right": side}
        )
        self.assertTrue(consensus["available"])
        self.assertEqual(consensus["aggregation"], "equal_view_mean_v1")
        self.assertAlmostEqual(consensus["fractions"]["black"], 0.4)
        self.assertAlmostEqual(consensus["fractions"]["white"], 0.6)

    def test_frontal_canonical_upper_body_palette_uses_front_view(self) -> None:
        front = {"available": True, "fractions": {"black": 0.057331}}
        left = {"available": True, "fractions": {"black": 0.711047}}
        right = {"available": True, "fractions": {"black": 0.407536}}
        consensus = {"available": True, "fractions": {"black": 0.391971}}

        selected = pipeline.canonical_upper_body_palette_reference(
            {"front": front, "left": left, "right": right},
            consensus,
        )

        self.assertIsNotNone(selected)
        self.assertAlmostEqual(selected["fractions"]["black"], 0.057331)
        self.assertEqual(selected["canonical_reference_role"], "front")
        self.assertEqual(
            selected["canonical_reference_policy"],
            pipeline.CANONICAL_UPPER_BODY_PALETTE_REFERENCE_POLICY,
        )

    def test_multi_view_source_audit_reports_duplicate_views_as_advisory(self) -> None:
        signature = self._identity_palette_signature_for_test(
            {
                "black": 0.5,
                "white": 0.5,
                "gray": 0.0,
                "warm_brown_or_orange": 0.0,
                "red_or_pink": 0.0,
                "yellow_or_gold": 0.0,
                "green": 0.0,
                "blue_or_cyan": 0.0,
                "purple": 0.0,
            },
            saturated_warm_fraction=0.0,
        )
        integrity = {
            "view_mode": "front_left_right",
            "views": {
                "front": {
                    "source_path": Path("front.png"),
                    "input_path": Path("front-input.png"),
                    "source_sha256": "a" * 64,
                    "input_sha256": "b" * 64,
                },
                "left": {
                    "source_path": Path("left.png"),
                    "input_path": Path("left-input.png"),
                    "source_sha256": "a" * 64,
                    "input_sha256": "c" * 64,
                },
                "right": {
                    "source_path": Path("right.png"),
                    "input_path": Path("right-input.png"),
                    "source_sha256": "d" * 64,
                    "input_sha256": "e" * 64,
                },
            },
        }
        similarity_signature = {
            "available": True,
            "mask": np.ones((8, 8), dtype=bool),
            "dhash": 1,
            "sha256": "f" * 64,
            "bbox_aspect_ratio": 1.0,
        }
        with (
            patch.object(
                pipeline,
                "identity_palette_signature",
                return_value=signature,
            ),
            patch.object(
                pipeline,
                "pose_similarity_signature",
                return_value=similarity_signature,
            ),
        ):
            audit = pipeline.audit_uploaded_identity_view_set(integrity)
        self.assertFalse(audit["acceptable"])
        self.assertIn("duplicate_identity_views_front_left", audit["issues"])
        self.assertFalse(audit["fail_closed"])
        self.assertFalse(audit["blocking"])

    def test_achromatic_palette_gate_rejects_new_warm_fur(self) -> None:
        source = self._identity_palette_signature_for_test(
            {
                "black": 0.15,
                "white": 0.05,
                "gray": 0.70,
                "warm_brown_or_orange": 0.08,
                "red_or_pink": 0.005,
                "yellow_or_gold": 0.005,
                "green": 0.005,
                "blue_or_cyan": 0.005,
                "purple": 0.0,
            },
            saturated_warm_fraction=0.005,
        )
        candidate = self._identity_palette_signature_for_test(
            {
                "black": 0.15,
                "white": 0.05,
                "gray": 0.59,
                "warm_brown_or_orange": 0.16,
                "red_or_pink": 0.005,
                "yellow_or_gold": 0.005,
                "green": 0.01,
                "blue_or_cyan": 0.03,
                "purple": 0.0,
            },
            saturated_warm_fraction=0.12,
        )

        quality = pipeline.identity_palette_quality(source, candidate)

        self.assertFalse(quality["acceptable"])
        self.assertIn(
            "achromatic_source_recolored_with_warm_fur",
            quality["issues"],
        )
        self.assertTrue(quality["saturated_warm_drift"])

    def test_achromatic_palette_gate_rejects_lost_or_expanded_dark_markings(
        self,
    ) -> None:
        source_fractions = {
            "black": 0.15,
            "white": 0.05,
            "gray": 0.70,
            "warm_brown_or_orange": 0.08,
            "red_or_pink": 0.005,
            "yellow_or_gold": 0.005,
            "green": 0.005,
            "blue_or_cyan": 0.005,
            "purple": 0.0,
        }
        source = self._identity_palette_signature_for_test(
            source_fractions,
            saturated_warm_fraction=0.005,
        )
        candidates = {
            "lost": {
                **source_fractions,
                "black": 0.03,
                "gray": 0.82,
            },
            "expanded": {
                **source_fractions,
                "black": 0.40,
                "gray": 0.45,
            },
        }

        for expected_flag, fractions in candidates.items():
            with self.subTest(expected_flag=expected_flag):
                candidate = self._identity_palette_signature_for_test(
                    fractions,
                    saturated_warm_fraction=0.005,
                )
                quality = pipeline.identity_palette_quality(source, candidate)
                self.assertFalse(quality["acceptable"])
                self.assertIn(
                    "achromatic_coat_value_balance_lost",
                    quality["issues"],
                )
                self.assertTrue(quality[f"achromatic_dark_marking_{expected_flag}"])

    def test_health_exposes_v22032_qwen_full_body_input_contract(self) -> None:
        effective_config = {
            "engine_profile": "scail2-14b-fp8-v1",
            "prompt_profile": "sleek-tabby-scail-v1",
            "seed_mode": "base_plus_action_offset",
            "frames": 81,
            "fps": 16.0,
            "steps": 6,
            "actions": {
                "eating": {
                    "driving_animation": "eating_driver.mp4",
                    "pose_strength": 1.1,
                    "seed_offset": 0,
                    "effective_seed": 20260726,
                },
                "grooming": {
                    "driving_animation": "grooming_driver.mp4",
                    "pose_strength": 1.15,
                    "seed_offset": 2,
                    "effective_seed": 20260728,
                },
            },
        }
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "queue_running": [],
            "queue_pending": [],
        }

        with (
            patch.multiple(
                pipeline,
                API_KEY="test-key",
                API_VERSION="2.21.4",
                IMAGE_PROFILES={},
                DEFAULT_IMAGE_PROFILE="",
                PROMPT_TRANSLATION_ENABLED=False,
                QWEN_POSE_MAX_ATTEMPTS=3,
                QWEN_BOWL_POSE_EXTRA_ATTEMPTS=2,
                GENERATION_VISUAL_AUDITS_BLOCKING=True,
            ),
            patch.object(pipeline.requests, "get", return_value=response),
            patch.object(pipeline, "required_model_paths", return_value=[]),
            patch.object(
                pipeline,
                "prompt_translator_missing_paths",
                return_value=[],
            ),
            patch.object(pipeline, "load_subject_profile", return_value={}),
            patch.object(
                pipeline,
                "resolve_effective_subject_config",
                return_value=effective_config,
            ),
            patch.object(pipeline, "require_subject_replacement_assets"),
            patch.object(
                pipeline,
                "subject_profile_public",
                return_value={"id": "sleek_tabby_cat_v1"},
            ),
        ):
            payload = pipeline.health()

        self.assertEqual(payload["version"], "2.21.4")
        self.assertFalse(payload["subject_replacement"]["ok"])
        self.assertEqual(
            payload["subject_replacement"]["upload_validation"]["status"],
            "failed",
        )
        guard = payload["qwen_style"]["pose_guard"]
        self.assertEqual(
            guard["reference_pipeline"],
            pipeline.QWEN_FULL_BODY_FIRST_FRAME_PIPELINE,
        )
        self.assertEqual(
            guard["upload_pose_reference_source"],
            "per_action_qwen_canonical_and_main_case_geometry_roles",
        )
        self.assertEqual(
            guard["geometry_generator_policy"],
            {
                "eating": "driving_eating_geometry_image2_v4",
                "grooming": pipeline.GROOMING_QWEN_GEOMETRY_GUIDE_POLICY,
            },
        )
        self.assertEqual(
            guard["grooming_geometry_isolation"]["qwen_foreground_rgb"],
            list(pipeline.GROOMING_QWEN_GUIDE_FOREGROUND_RGB),
        )
        self.assertEqual(
            guard["grooming_geometry_isolation"]["qwen_foreground_palette_rgb"],
            [list(pipeline.GROOMING_QWEN_GUIDE_FOREGROUND_RGB)],
        )
        self.assertFalse(
            guard["grooming_geometry_isolation"]["qwen_internal_luma_variation"]
        )
        self.assertEqual(
            guard["grooming_geometry_isolation"]["qwen_internal_luma_variation_role"],
            "none",
        )
        self.assertFalse(
            guard["grooming_geometry_isolation"][
                "qwen_appearance_bearing_internal_luma_variation"
            ]
        )
        self.assertFalse(
            guard["grooming_geometry_isolation"]["qwen_synthetic_foreleg_semantic"]
        )
        self.assertEqual(
            guard["grooming_geometry_isolation"][
                "qwen_synthetic_foreleg_semantic_method"
            ],
            None,
        )
        self.assertTrue(guard["grooming_geometry_isolation"]["qwen_foreleg_repair"])
        self.assertTrue(
            guard["grooming_geometry_isolation"][
                "qwen_foreleg_repair_retained_for_legacy_jobs_only"
            ]
        )
        self.assertEqual(
            guard["grooming_geometry_isolation"]["qwen_foreleg_repair_policy"],
            pipeline.GROOMING_CANONICAL_FORELEG_COMPLETION_POLICY,
        )
        self.assertEqual(
            guard["grooming_geometry_isolation"]["qwen_foreleg_repair_attempt"],
            pipeline.GROOMING_QWEN_FORELEG_REPAIR_ATTEMPT,
        )
        self.assertEqual(
            guard["grooming_geometry_isolation"]["qwen_foreleg_repair_denoise"],
            pipeline.GROOMING_QWEN_FORELEG_REPAIR_DENOISE,
        )
        self.assertFalse(
            guard["grooming_geometry_isolation"][
                "qwen_foreleg_repair_synthetic_capsule_exposed"
            ]
        )
        self.assertTrue(
            guard["grooming_geometry_isolation"][
                "qwen_foreleg_repair_open_channel_anatomy_image3"
            ]
        )
        self.assertFalse(
            guard["grooming_geometry_isolation"][
                "qwen_foreleg_repair_private_cv_luma_image3"
            ]
        )
        self.assertEqual(
            guard["grooming_geometry_isolation"][
                "qwen_foreleg_repair_mask_dilation_px"
            ],
            pipeline.GROOMING_QWEN_FORELEG_REPAIR_MASK_DILATION_PX,
        )
        self.assertEqual(
            guard["grooming_geometry_isolation"]["qwen_foreleg_repair_anatomy_policy"],
            pipeline.GROOMING_QWEN_FORELEG_REPAIR_ANATOMY_POLICY,
        )
        self.assertEqual(
            guard["grooming_geometry_isolation"]["qwen_foreleg_repair_seed_source"],
            "identity_safe_candidate_unchanged_before_masked_sampling",
        )
        self.assertIsNone(
            guard["grooming_geometry_isolation"][
                "qwen_foreleg_repair_seed_lowpass_sigma_px"
            ]
        )
        self.assertIsNone(
            guard["grooming_geometry_isolation"][
                "qwen_foreleg_repair_seed_identity_median_blend"
            ]
        )
        self.assertFalse(
            guard["grooming_geometry_isolation"][
                "qwen_foreleg_repair_identity_or_geometry_seed_inserted"
            ]
        )
        self.assertEqual(
            guard["grooming_geometry_isolation"][
                "qwen_foreleg_repair_noise_mask_expansion_px"
            ],
            pipeline.GROOMING_QWEN_FORELEG_REPAIR_NOISE_MASK_EXPANSION_PX,
        )
        self.assertTrue(
            guard["grooming_geometry_isolation"][
                "qwen_foreleg_repair_expanded_context_full_denoise_mask"
            ]
        )
        self.assertEqual(
            guard["grooming_geometry_isolation"]["qwen_foreleg_repair_noise_mask_node"],
            "SetLatentNoiseMask",
        )
        self.assertEqual(
            guard["grooming_geometry_isolation"]["qwen_foreleg_repair_style_policy"],
            "disabled_for_canonical_masked_completion",
        )
        self.assertIsNone(
            guard["grooming_geometry_isolation"]["qwen_foreleg_repair_style_lora"]
        )
        self.assertEqual(
            guard["grooming_geometry_isolation"]["qwen_foreleg_repair_style_strength"],
            0.0,
        )
        self.assertEqual(
            guard["grooming_geometry_isolation"]["qwen_foreleg_repair_style_scope"],
            "none",
        )
        self.assertFalse(
            guard["grooming_geometry_isolation"][
                "qwen_foreleg_repair_style_identity_authority"
            ]
        )
        self.assertTrue(
            guard["grooming_geometry_isolation"][
                "qwen_foreleg_repair_style_fail_closed"
            ]
        )
        self.assertIsNone(
            guard["grooming_geometry_isolation"][
                "qwen_foreleg_repair_composite_gate_policy"
            ]
        )
        self.assertFalse(
            guard["grooming_geometry_isolation"][
                "qwen_foreleg_repair_composite_recovery_enabled"
            ]
        )
        self.assertIsNone(
            guard["grooming_geometry_isolation"][
                "qwen_foreleg_repair_composite_edge_density_ratio_min"
            ]
        )
        self.assertIsNone(
            guard["grooming_geometry_isolation"][
                "qwen_foreleg_repair_composite_planted_paw_gradient_p90_max"
            ]
        )
        self.assertIsNone(
            guard["grooming_geometry_isolation"][
                "qwen_foreleg_repair_composite_raised_paw_gradient_p90_max"
            ]
        )
        self.assertEqual(
            guard["grooming_geometry_isolation"][
                "canonical_completion_edge_evidence_policy"
            ],
            pipeline.GROOMING_CANONICAL_COMPLETION_EDGE_EVIDENCE_POLICY,
        )
        self.assertEqual(
            guard["grooming_geometry_isolation"][
                "canonical_completion_edge_density_ratio_min"
            ],
            pipeline.GROOMING_CANONICAL_COMPLETION_EDGE_DENSITY_RATIO_MIN,
        )
        self.assertTrue(
            guard["grooming_geometry_isolation"][
                "canonical_completion_independent_fur_surface_required"
            ]
        )
        self.assertTrue(
            guard["grooming_geometry_isolation"]["canonical_full_pose_recovery_enabled"]
        )
        self.assertEqual(
            guard["grooming_geometry_isolation"]["canonical_full_pose_rescue_attempt"],
            pipeline.GROOMING_CANONICAL_FULL_POSE_RESCUE_ATTEMPT,
        )
        self.assertEqual(
            guard["grooming_geometry_isolation"][
                "canonical_full_pose_identity_refine_attempt"
            ],
            pipeline.GROOMING_CANONICAL_FULL_POSE_REFINE_ATTEMPT,
        )
        self.assertIsNone(
            guard["grooming_geometry_isolation"][
                "canonical_full_pose_recovery_style_lora"
            ]
        )
        self.assertFalse(
            guard["grooming_geometry_isolation"][
                "canonical_full_pose_recovery_global_thresholds_changed"
            ]
        )
        self.assertEqual(
            guard["grooming_geometry_isolation"][
                "qwen_foreleg_repair_global_edge_density_min"
            ],
            0.11,
        )
        self.assertFalse(
            guard["grooming_geometry_isolation"][
                "qwen_foreleg_repair_global_pose_threshold_changed"
            ]
        )
        composite_recovery = guard["grooming_pose_gate"][
            "masked_pet3d_composite_recovery"
        ]
        self.assertFalse(composite_recovery["production_enabled"])
        self.assertTrue(composite_recovery["legacy_geometry_first_jobs_only"])
        self.assertEqual(
            composite_recovery["policy"],
            pipeline.GROOMING_QWEN_FORELEG_REPAIR_COMPOSITE_GATE_POLICY,
        )
        self.assertEqual(composite_recovery["attempt"], 3)
        self.assertEqual(composite_recovery["edge_density_ratio_min"], 0.70)
        self.assertTrue(composite_recovery["requires_edge_density_issue"])
        self.assertTrue(
            composite_recovery[
                "allows_only_optional_low_texture_planted_paw_ratio_issue"
            ]
        )
        self.assertEqual(
            composite_recovery["planted_paw_gradient_p90_max"],
            8.0,
        )
        self.assertEqual(
            composite_recovery["raised_paw_gradient_p90_max"],
            128.0,
        )
        self.assertTrue(
            composite_recovery["requires_geometry_contact_color_histogram_pass"]
        )
        self.assertTrue(composite_recovery["requires_independent_fur_surface_pass"])
        self.assertTrue(
            composite_recovery["requires_digest_bound_noise_masked_pet3d_repair"]
        )
        self.assertTrue(composite_recovery["fail_closed"])
        self.assertEqual(guard["grooming_pose_gate"]["global_edge_density_min"], 0.11)
        self.assertTrue(
            guard["grooming_pose_gate"]["global_edge_density_min_unchanged"]
        )
        self.assertEqual(
            guard["grooming_pose_gate"]["structural_edge_policy"],
            "gaussian_lowpass_organic_foreleg_structure_v1",
        )
        self.assertEqual(
            guard["grooming_pose_gate"]["structural_blur_sigma"],
            4.0,
        )
        self.assertEqual(
            guard["grooming_pose_gate"]["structural_edge_density_min"],
            0.015,
        )
        self.assertTrue(
            guard["grooming_pose_gate"]["fine_fur_edges_cannot_satisfy_structural_gate"]
        )
        direct_evidence = guard["grooming_pose_gate"][
            "canonical_direct_structural_edge_evidence"
        ]
        self.assertEqual(
            direct_evidence["policy"],
            pipeline.GROOMING_CANONICAL_DIRECT_STRUCTURAL_EDGE_EVIDENCE_POLICY,
        )
        self.assertEqual(direct_evidence["maximum_attempt"], 2)
        self.assertEqual(
            direct_evidence["raw_edge_density_ratio_min"],
            0.925,
        )
        self.assertEqual(
            direct_evidence["structural_edge_density_min"],
            0.03,
        )
        self.assertEqual(
            direct_evidence["strict_chamfer_coverage_min"],
            0.70,
        )
        self.assertEqual(
            direct_evidence["strict_chamfer_mean_distance_max"],
            5.0,
        )
        self.assertEqual(
            direct_evidence["strict_chamfer_p90_distance_max"],
            12.0,
        )
        self.assertTrue(direct_evidence["rejects_historical_straight_prop_candidates"])
        self.assertTrue(direct_evidence["identity_palette_gate_still_required"])
        self.assertTrue(direct_evidence["independent_fur_surface_gate_still_required"])
        self.assertFalse(direct_evidence["global_pose_threshold_changed"])
        self.assertTrue(direct_evidence["fail_closed"])
        surface_nonexpansion = guard["grooming_pose_gate"][
            "canonical_direct_surface_nonexpansion"
        ]
        self.assertEqual(
            surface_nonexpansion["policy"],
            pipeline.GROOMING_CANONICAL_DIRECT_SURFACE_NONEXPANSION_POLICY,
        )
        self.assertEqual(surface_nonexpansion["maximum_attempt"], 2)
        self.assertEqual(
            surface_nonexpansion["only_recoverable_issue"],
            "brush_texture_coherence",
        )
        self.assertTrue(surface_nonexpansion["requires_strict_pose_recovery"])
        self.assertTrue(surface_nonexpansion["requires_identity_palette_pass"])
        self.assertTrue(
            surface_nonexpansion["requires_digest_bound_transparent_canonical"]
        )
        self.assertEqual(surface_nonexpansion["coherent_fraction_max"], 0.57)
        self.assertEqual(
            surface_nonexpansion["coherent_fraction_delta_max"],
            0.01,
        )
        self.assertTrue(
            surface_nonexpansion["independent_final_video_fur_audit_still_required"]
        )
        self.assertFalse(surface_nonexpansion["global_surface_thresholds_changed"])
        self.assertTrue(surface_nonexpansion["fail_closed"])
        self.assertFalse(
            guard["grooming_geometry_isolation"]["main_case_appearance_exposed_to_qwen"]
        )
        self.assertFalse(
            guard["grooming_geometry_isolation"][
                "private_cv_gate_exposed_to_qwen_initial_attempts"
            ]
        )
        self.assertFalse(
            guard["grooming_geometry_isolation"][
                "private_cv_gate_neutral_copy_exposed_on_repair_attempt"
            ]
        )
        self.assertTrue(
            guard["grooming_geometry_isolation"][
                "public_open_channel_copy_exposed_on_repair_attempt"
            ]
        )
        self.assertEqual(
            guard["grooming_geometry_isolation"]["private_completion_mask_policy"],
            pipeline.GROOMING_PRIVATE_COMPLETION_MASK_POLICY,
        )
        self.assertFalse(
            guard["grooming_geometry_isolation"][
                "private_completion_mask_exposed_to_qwen"
            ]
        )
        self.assertFalse(
            guard["grooming_geometry_isolation"][
                "private_completion_mask_loaded_as_qwen_image"
            ]
        )
        self.assertEqual(
            guard["pose_identity_input_role"],
            {
                "eating": "image1_qwen_full_body_canonical",
                "grooming": "image1_qwen_full_body_canonical",
            },
        )
        self.assertFalse(guard["generated_canonical_bypassed"])
        self.assertTrue(guard["canonical_reference_generated"])
        self.assertTrue(guard["canonical_reference_used"])
        self.assertFalse(guard["raw_upload_sent_to_scail"])
        self.assertEqual(guard["identity_upload_view_counts"], [1, 3])
        self.assertEqual(
            guard["identity_upload_three_view_roles"],
            ["front", "left", "right"],
        )
        self.assertTrue(guard["qwen_canonical_multi_view_supported"])
        self.assertEqual(
            guard["qwen_canonical_multi_view_node_inputs"],
            ["image1", "image2", "image3"],
        )
        self.assertTrue(guard["qwen_canonical_image1_is_sampler_latent"])
        self.assertFalse(guard["qwen_action_raw_identity_views_used"])
        self.assertEqual(
            guard["qwen_action_input_contract"],
            {
                "eating": "canonical_image1_plus_geometry_image2",
                "grooming": "canonical_image1_plus_geometry_image2",
            },
        )
        canonical_action_palette_gate = guard["canonical_action_identity_palette_gate"]
        self.assertTrue(canonical_action_palette_gate["enabled"])
        self.assertEqual(
            canonical_action_palette_gate["policy"],
            pipeline.CANONICAL_ACTION_IDENTITY_PALETTE_POLICY,
        )
        self.assertEqual(
            canonical_action_palette_gate["achromatic_balance_l1_max"],
            pipeline.CANONICAL_ACTION_ACHROMATIC_BALANCE_L1_MAX,
        )
        self.assertTrue(canonical_action_palette_gate["fail_closed"])
        self.assertTrue(
            guard["grooming_geometry_isolation"]["qwen_structural_negative_space"]
        )
        self.assertEqual(
            guard["grooming_geometry_isolation"][
                "qwen_structural_negative_space_method"
            ],
            pipeline.GROOMING_QWEN_NEGATIVE_SPACE_METHOD,
        )
        self.assertFalse(
            guard["grooming_geometry_isolation"]["qwen_closed_foreleg_capsule_exposed"]
        )
        self.assertTrue(
            guard["grooming_geometry_isolation"][
                "qwen_internal_green_separator_exposed"
            ]
        )
        self.assertEqual(
            guard["grooming_geometry_isolation"]["qwen_raised_foreleg_validation_role"],
            "public_open_curve_geometry_plus_private_cv",
        )
        self.assertEqual(
            guard["grooming_geometry_isolation"]["qwen_role"],
            "image2_action_geometry_guide",
        )
        self.assertTrue(guard["uploaded_identity_appearance_authority"])
        self.assertEqual(
            guard["subject_profile_appearance_role"],
            "action_geometry_only",
        )
        self.assertEqual(guard["canonical_palette_input"], "final_alpha_cutout")
        self.assertEqual(
            guard["canonical_upper_body_palette_reference_policy"],
            pipeline.CANONICAL_UPPER_BODY_PALETTE_REFERENCE_POLICY,
        )
        self.assertTrue(guard["canonical_raw_palette_retained_for_audit"])
        self.assertEqual(
            guard["canonical_chroma_despill"],
            "alpha_boundary_plus_source_palette_guarded_interior_v1",
        )
        self.assertEqual(
            guard["canonical_chroma_spill_policy"],
            pipeline.UPLOADED_IDENTITY_CHROMA_SPILL_POLICY,
        )
        self.assertTrue(guard["canonical_chroma_spill_preserves_paired_eye_accents"])
        self.assertTrue(guard["canonical_interior_hole_gate"])
        self.assertEqual(
            guard["upload_pose_photography_lora_policy"],
            "disabled_after_canonical_identity_style_lock_v1",
        )
        self.assertEqual(
            guard["upload_pose_builtin_style_lora_policy"],
            "disabled_for_canonical_actions_v2",
        )
        self.assertEqual(
            guard["upload_pose_lora_policy"],
            "qwen_lightning_only_with_open_channel_identity_refine_v6",
        )
        eating_refine = guard["eating_identity_refine"]
        self.assertTrue(eating_refine["enabled"])
        self.assertEqual(
            eating_refine["policy"],
            pipeline.EATING_CANONICAL_IDENTITY_REFINE_POLICY,
        )
        self.assertEqual(
            eating_refine["attempt"],
            pipeline.EATING_CANONICAL_IDENTITY_REFINE_ATTEMPT,
        )
        self.assertTrue(eating_refine["qwen_lightning_only"])
        self.assertIsNone(eating_refine["style_lora"])
        self.assertIsNone(eating_refine["sampler_noise_mask"])
        self.assertFalse(eating_refine["global_thresholds_changed"])
        self.assertEqual(
            guard["upload_pose_candidate_normalization_policy"],
            pipeline.QWEN_ACTION_REFERENCE_NORMALIZATION_POLICY,
        )
        self.assertEqual(
            guard["upload_pose_candidate_matte_policy"],
            pipeline.QWEN_ACTION_CANDIDATE_MATTE_POLICY,
        )
        self.assertEqual(
            guard["eating_mouth_contact_gate"]["policy"],
            pipeline.EATING_MOUTH_BOWL_CONTACT_POLICY,
        )
        self.assertEqual(
            guard["eating_mouth_contact_gate"]["component_width_bowl_ratio_max"],
            0.70,
        )
        self.assertEqual(
            guard["eating_mouth_contact_gate"]["gap_px_max"],
            8,
        )
        self.assertEqual(
            guard["normalization_contract"]["action_keyframes"],
            pipeline.QWEN_ACTION_REFERENCE_NORMALIZATION_POLICY,
        )
        self.assertTrue(
            guard["normalization_contract"][
                "selected_candidate_byte_copied_to_scail_reference"
            ]
        )
        self.assertTrue(
            guard["normalization_contract"]["full_body_normalized_transparent_required"]
        )
        self.assertTrue(
            guard["normalization_contract"][
                "action_keyframes_normalized_transparent_required"
            ]
        )
        self.assertEqual(
            guard["pose_identity_palette_reference_role"]["eating"],
            "qwen_full_body_canonical",
        )
        self.assertEqual(
            guard["legacy_profile_action_reference_used"],
            {"eating": False, "grooming": False},
        )
        self.assertFalse(guard["profile_pixels_used_as_identity_or_scail_reference"])
        self.assertFalse(guard["profile_pixels_used_as_validation_baseline"])
        self.assertTrue(guard["uploaded_identity_absolute_surface_gate_used"])
        self.assertEqual(
            guard["surface_candidate_comparison_mode"],
            pipeline.UPLOADED_IDENTITY_SURFACE_COMPARISON_MODE,
        )
        self.assertEqual(guard["max_attempts"], 3)
        self.assertEqual(guard["bowl_extra_attempts"], 2)
        self.assertEqual(guard["bowl_max_attempts"], 5)

    def test_health_reports_ready_upload_surface_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = _subject_profile_v2(root)
            ffmpeg = root / "ffmpeg"
            ffmpeg.write_bytes(b"test")
            response = Mock()
            response.raise_for_status.return_value = None
            response.json.return_value = {
                "queue_running": [],
                "queue_pending": [],
            }
            surface_result = {
                "acceptable": True,
                "issues": [],
                "candidate": {"metrics": {}},
            }

            with (
                patch.multiple(
                    pipeline,
                    API_KEY="test-key",
                    API_VERSION="2.21.4",
                    IMAGE_PROFILES={},
                    DEFAULT_IMAGE_PROFILE="",
                    PROMPT_TRANSLATION_ENABLED=False,
                    QWEN_POSE_MAX_ATTEMPTS=3,
                    QWEN_BOWL_POSE_EXTRA_ATTEMPTS=2,
                    GENERATION_VISUAL_AUDITS_BLOCKING=True,
                    FFMPEG_BIN=str(ffmpeg),
                ),
                patch.object(pipeline.requests, "get", return_value=response),
                patch.object(pipeline, "required_model_paths", return_value=[]),
                patch.object(
                    pipeline,
                    "prompt_translator_missing_paths",
                    return_value=[],
                ),
                patch.object(
                    pipeline,
                    "load_subject_profile",
                    return_value=profile,
                ),
                patch.object(pipeline, "require_subject_replacement_assets"),
                patch.object(
                    pipeline,
                    "evaluate_fur_surface",
                    return_value=surface_result,
                ),
                patch.object(
                    pipeline,
                    "subject_profile_public",
                    return_value={"id": "sleek_tabby_cat_v1"},
                ),
            ):
                payload = pipeline.health()

        replacement = payload["subject_replacement"]
        self.assertTrue(replacement["ok"])
        self.assertEqual(
            replacement["upload_validation"]["status"],
            "ready",
        )
        self.assertEqual(
            replacement["upload_validation"]["expected_surface_gate_actions"],
            ["eating", "grooming"],
        )
        self.assertEqual(
            payload["qwen_style"]["pose_guard"]["fur_surface_gate"]["status"],
            "passed",
        )
        surface_gate = payload["qwen_style"]["pose_guard"]["fur_surface_gate"]
        self.assertEqual(
            surface_gate["retry_policy"],
            pipeline.FUR_SURFACE_GATE_RETRY_POLICY,
        )
        self.assertTrue(surface_gate["preserves_configured_eating_bowl_recovery"])
        self.assertEqual(
            surface_gate["configured_eating_bowl_recovery_attempts"],
            2,
        )
        self.assertEqual(
            surface_gate["configured_eating_bowl_recovery_profile_prefix"],
            "bowl_palette_canvas_seed_retry_",
        )
        self.assertEqual(
            surface_gate["radial_long_line_evidence_policy"],
            "hough_radial_fraction_and_radial_line_density_v1",
        )
        self.assertFalse(surface_gate["radial_ratio_alone_can_reject"])
        self.assertEqual(
            surface_gate["thresholds"]["hough_radial_line_density_min"],
            0.008,
        )
        self.assertTrue(surface_gate["blocks_scail_on_exhaustion"])
        self.assertEqual(
            surface_gate["final_video_uploaded_grooming_scope"],
            pipeline.FINAL_VIDEO_VALIDATED_ACTION_SURFACE_SCOPE,
        )
        self.assertEqual(
            surface_gate["final_video_uploaded_qwen_actions"],
            ["eating", "grooming"],
        )
        self.assertEqual(
            surface_gate["final_video_uploaded_qwen_reference_policies"],
            pipeline.FINAL_VIDEO_VALIDATED_ACTION_REFERENCE_POLICIES,
        )
        self.assertTrue(
            surface_gate["final_video_grooming_no_prop_identity_mask_required_for_v2"]
        )
        self.assertFalse(
            surface_gate[
                "final_video_profile_pixels_used_for_uploaded_identity_appearance_validation"
            ]
        )
        self.assertTrue(
            surface_gate[
                "final_video_uploaded_qwen_action_reference_is_appearance_authority"
            ]
        )
        self.assertTrue(
            surface_gate[
                "final_video_derived_texture_reference_hash_and_alpha_contract_required"
            ]
        )
        self.assertTrue(
            surface_gate[
                "final_video_derived_texture_baseline_absolute_observations_nonblocking"
            ]
        )
        self.assertTrue(
            surface_gate[
                "final_video_unmasked_action_reference_baseline_must_pass_current_absolute_contract"
            ]
        )
        self.assertTrue(
            surface_gate["final_video_malformed_evaluator_contract_fails_closed"]
        )
        self.assertTrue(
            surface_gate[
                "final_video_coherent_brush_requires_gradient_weighted_load_or_long_line_energy_backstop"
            ]
        )
        self.assertEqual(
            surface_gate["final_video_coherent_brush_load_proxy"],
            "structure_tensor_coherent_fraction_x_coarse_gradient_p90",
        )
        self.assertEqual(
            surface_gate["final_video_coherent_brush_load_only_temporal_min_samples"],
            2,
        )
        self.assertTrue(
            surface_gate[
                "final_video_coherent_brush_load_only_single_sample_is_observation"
            ]
        )
        self.assertTrue(
            surface_gate[
                "final_video_coherent_brush_long_line_energy_backstop_remains_single_sample_blocking"
            ]
        )
        self.assertTrue(
            surface_gate[
                "final_video_absolute_coherent_brush_issue_alone_is_observation"
            ]
        )
        self.assertTrue(
            surface_gate[
                "final_video_long_fur_requires_contour_corroboration_or_strong_expansion"
            ]
        )
        self.assertTrue(
            surface_gate[
                "final_video_long_fur_texture_only_absolute_and_dual_surface_evidence_is_observation"
            ]
        )
        self.assertTrue(
            surface_gate[
                "final_video_eating_isolated_strong_long_fur_temporal_corroboration"
            ]
        )
        self.assertTrue(
            surface_gate[
                "final_video_eating_isolated_long_fur_requires_single_interior_blocking_sample"
            ]
        )
        self.assertTrue(
            surface_gate[
                "final_video_eating_isolated_long_fur_records_blocking_frame_unmasked_corroboration"
            ]
        )
        self.assertTrue(
            surface_gate[
                "final_video_eating_isolated_long_fur_requires_adjacent_prop_excluded_observation"
            ]
        )
        self.assertTrue(
            surface_gate[
                "final_video_eating_isolated_long_fur_allows_one_initial_radial_companion_for_later_validation"
            ]
        )
        self.assertEqual(
            surface_gate["final_video_eating_isolated_long_fur_minimum_samples"],
            5,
        )
        self.assertTrue(
            surface_gate[
                "final_video_eating_isolated_long_fur_reuses_original_neighbor_ratio_and_delta_thresholds"
            ]
        )
        self.assertTrue(
            surface_gate[
                "final_video_eating_isolated_long_fur_strong_ceiling_uses_one_existing_texture_delta_margin"
            ]
        )
        self.assertTrue(
            surface_gate[
                "final_video_eating_isolated_long_fur_preserves_absolute_contour_radial_coherent_and_unknown_fail_closed"
            ]
        )
        self.assertTrue(
            surface_gate[
                "final_video_eating_initial_radial_loop_endpoint_corroboration"
            ]
        )
        self.assertTrue(
            surface_gate[
                "final_video_eating_initial_radial_requires_single_initial_blocking_sample"
            ]
        )
        self.assertTrue(
            surface_gate[
                "final_video_eating_initial_radial_requires_clean_terminal_and_all_interior_samples"
            ]
        )
        self.assertTrue(
            surface_gate[
                "final_video_eating_initial_radial_requires_hash_bound_radial_long_lines_reference_observation"
            ]
        )
        self.assertEqual(
            surface_gate["final_video_eating_initial_radial_boundary_minimum_samples"],
            5,
        )
        self.assertEqual(
            surface_gate[
                "final_video_eating_initial_radial_boundary_load_energy_spread_ratio_max"
            ],
            1.10,
        )
        self.assertEqual(
            surface_gate[
                "final_video_eating_initial_radial_boundary_component_spread_ratio_max"
            ],
            1.20,
        )
        self.assertTrue(
            surface_gate[
                "final_video_eating_initial_radial_preserves_absolute_radial_fail_closed"
            ]
        )
        self.assertEqual(
            surface_gate["final_video_eating_prop_alignment_minimum_overlap"],
            0.75,
        )
        self.assertTrue(
            surface_gate[
                "final_video_eating_initial_radial_bounded_reference_corroboration"
            ]
        )
        self.assertEqual(
            surface_gate[
                "final_video_eating_initial_radial_bounded_reference_ratio_margin"
            ],
            0.10,
        )
        self.assertEqual(
            surface_gate[
                "final_video_eating_initial_radial_bounded_reference_load_delta_multiplier"
            ],
            2.0,
        )
        self.assertTrue(
            surface_gate[
                "final_video_eating_initial_radial_bounded_reference_preserves_absolute_radial_fail_closed"
            ]
        )
        self.assertTrue(
            surface_gate["final_video_eating_loop_boundary_decode_anchor_corroboration"]
        )
        self.assertTrue(
            surface_gate[
                "final_video_eating_boundary_anchor_requires_clean_first_and_last_sample"
            ]
        )
        self.assertTrue(
            surface_gate[
                "final_video_eating_boundary_anchor_requires_contiguous_interior_samples"
            ]
        )
        self.assertEqual(
            surface_gate["final_video_eating_boundary_anchor_minimum_samples"],
            5,
        )
        self.assertEqual(
            surface_gate["final_video_eating_boundary_anchor_minimum_blocking_samples"],
            2,
        )
        self.assertEqual(
            surface_gate["final_video_eating_boundary_anchor_spread_ratio_max"],
            1.20,
        )
        self.assertTrue(
            surface_gate[
                "final_video_grooming_masked_pet3d_low_signal_reference_texture_floor"
            ]
        )
        self.assertTrue(
            surface_gate[
                "final_video_grooming_masked_pet3d_sparse_coarse_low_signal_reference_texture_floor"
            ]
        )
        self.assertEqual(
            surface_gate["final_video_sparse_coarse_low_signal_minimum_samples"],
            5,
        )
        self.assertEqual(
            surface_gate[
                "final_video_sparse_coarse_low_signal_minimum_blocking_samples"
            ],
            2,
        )
        self.assertEqual(
            surface_gate[
                "final_video_sparse_coarse_low_signal_maximum_blocking_fraction"
            ],
            0.50,
        )
        self.assertTrue(
            surface_gate[
                "final_video_sparse_coarse_low_signal_requires_all_sample_long_line_observation"
            ]
        )
        self.assertTrue(
            surface_gate[
                "final_video_sparse_coarse_low_signal_coarse_ceiling_uses_one_existing_texture_delta_margin"
            ]
        )
        self.assertTrue(
            surface_gate[
                "final_video_sparse_coarse_low_signal_preserves_absolute_contour_radial_coherent_and_unknown_fail_closed"
            ]
        )
        self.assertTrue(
            surface_gate[
                "final_video_grooming_direct_qwen_sparse_coarse_temporal_floor"
            ]
        )
        self.assertEqual(
            surface_gate["final_video_direct_qwen_sparse_coarse_minimum_samples"],
            5,
        )
        self.assertEqual(
            surface_gate[
                "final_video_direct_qwen_sparse_coarse_minimum_blocking_samples"
            ],
            2,
        )
        self.assertEqual(
            surface_gate[
                "final_video_direct_qwen_sparse_coarse_maximum_blocking_fraction"
            ],
            0.50,
        )
        self.assertEqual(
            surface_gate[
                "final_video_direct_qwen_sparse_coarse_temporal_spread_ratio_max"
            ],
            1.20,
        )
        self.assertTrue(
            surface_gate[
                "final_video_direct_qwen_sparse_coarse_requires_non_low_signal_reference"
            ]
        )
        self.assertTrue(
            surface_gate[
                "final_video_direct_qwen_sparse_coarse_requires_blocking_dual_surface_coarse_long_pair"
            ]
        )
        self.assertTrue(
            surface_gate[
                "final_video_direct_qwen_sparse_coarse_coarse_ceiling_uses_reference_ratio_plus_one_texture_delta"
            ]
        )
        self.assertTrue(
            surface_gate[
                "final_video_direct_qwen_sparse_coarse_long_ceiling_reuses_existing_strong_limits"
            ]
        )
        self.assertTrue(
            surface_gate[
                "final_video_direct_qwen_sparse_coarse_preserves_absolute_contour_radial_coherent_and_unknown_fail_closed"
            ]
        )
        self.assertTrue(
            surface_gate[
                "final_video_low_signal_reference_requires_verified_composite_recovery"
            ]
        )
        self.assertTrue(
            surface_gate["final_video_low_signal_reference_requires_all_samples"]
        )
        self.assertTrue(
            surface_gate[
                "final_video_low_signal_reference_requires_coarse_blocking_all_samples"
            ]
        )
        self.assertEqual(
            surface_gate[
                "final_video_low_signal_reference_minimum_long_blocking_fraction"
            ],
            0.50,
        )
        self.assertTrue(
            surface_gate[
                "final_video_low_signal_reference_allows_remaining_long_observations"
            ]
        )
        self.assertEqual(
            surface_gate["final_video_low_signal_reference_minimum_samples"],
            3,
        )
        self.assertEqual(
            surface_gate["final_video_low_signal_reference_temporal_spread_ratio_max"],
            1.20,
        )
        self.assertTrue(
            surface_gate[
                "final_video_low_signal_reference_preserves_absolute_contour_radial_and_unknown_fail_closed"
            ]
        )

    def test_eating_qwen_geometry_guide_fails_closed_without_driver(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(
                FileNotFoundError,
                "Eating driving animation is missing",
            ):
                pipeline.prepare_eating_qwen_geometry_guide(
                    "missing-eating-driver",
                    root / "missing-driver.mp4",
                    root / "guide.png",
                    frames=81,
                    fps=16.0,
                )

    def test_grooming_qwen_geometry_guide_fails_closed_without_driver(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(
                FileNotFoundError,
                "Grooming driving animation is missing",
            ):
                pipeline.prepare_grooming_qwen_geometry_guide(
                    "missing-grooming-driver",
                    root / "missing-driver.mp4",
                    root / "guide.png",
                    frames=81,
                    fps=16.0,
                )

    def test_eating_qwen_geometry_guide_uses_versioned_runner_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "eat.mp4"
            source.write_bytes(b"main-case-eat")
            target = root / "eating-guide.png"
            captured: dict[str, object] = {}

            def fake_process(
                target_job_id: str,
                command: list[str],
                **kwargs: object,
            ) -> None:
                captured["job_id"] = target_job_id
                captured["command"] = command
                captured["kwargs"] = kwargs
                Image.new("RGB", (512, 512), (0, 255, 0)).save(target)
                with Image.open(target) as loaded:
                    image = loaded.copy()
                image.paste((96, 96, 96), (96, 80, 416, 432))
                image.paste((214, 208, 196), (180, 360, 330, 420))
                image.paste((232, 228, 218), (180, 360, 330, 370))
                image.paste((142, 86, 45), (220, 365, 260, 390))
                image.save(target)
                metadata_path = Path(
                    command[command.index("--eating-geometry-guide-metadata") + 1]
                )
                metadata_path.write_text(
                    json.dumps(
                        {
                            "policy": "driving_eating_geometry_image2_v4",
                            "method": (
                                "temporal_head_lowering_continuous_muzzle_"
                                "semantic_food_fields_v5"
                            ),
                            "source_sha256": pipeline.file_sha256(source),
                            "sampled_frame_count": 81,
                            "sampled_fps": 16.0,
                            "selected_frame_index": 54,
                            "muzzle_bowl_gap_px": 0,
                            "contact_pixels": 96,
                            "muzzle_contact_semantic": {
                                "method": (
                                    "source_anchored_continuous_neutral_muzzle_tip_v2"
                                ),
                                "muzzle_food_gap_px": 0,
                                "artificial_shape_added": False,
                                "detached_marker_added": False,
                                "source_rgb_preserved": False,
                                "source_chroma_preserved": False,
                                "identity_preserved": False,
                                "passed": True,
                            },
                            "semantic_food": {
                                "method": (
                                    "fixed_beige_bowl_brown_kibble_from_source_mask_v1"
                                ),
                                "food_core_pixels": 1000,
                                "minimum_food_core_pixels": 58,
                                "source_rgb_preserved": False,
                                "semantic_chroma_only": True,
                                "passed": True,
                            },
                            "head_lowering_selection": {
                                "method": "central_upper_subject_row_p10_v1",
                                "selected_head_support_pixels": 4096,
                                "head_y_p10_span": 75.0,
                                "head_y_p10_span_min": 6.144,
                                "passed": True,
                            },
                            "subject_pixels": 103640,
                            "bowl_pixels": 9000,
                            "outline_smoothing": {
                                "perimeter_ratio": 0.92,
                                "high_frequency_outline_preserved": False,
                            },
                            "semantic_structure": {
                                "method": (
                                    "percentile_normalized_low_frequency_luma_v1"
                                ),
                                "source_luma_values_preserved": False,
                                "source_rgb_preserved": False,
                                "source_chroma_preserved": False,
                                "high_frequency_texture_preserved": False,
                            },
                            "palette": {
                                "background": [0, 255, 0],
                                "subject_gray_range": [96, 174],
                                "prop_semantic_colors": {
                                    "bowl": [214, 208, 196],
                                    "bowl_rim": [232, 228, 218],
                                    "food": [142, 86, 45],
                                },
                                "source_rgb_preserved": False,
                                "source_chroma_preserved": False,
                                "semantic_prop_chroma_preserved": True,
                                "source_luma_values_preserved": False,
                                "high_frequency_texture_preserved": False,
                            },
                        }
                    ),
                    encoding="utf-8",
                )

            with (
                patch.multiple(
                    pipeline,
                    OUTPUT_ROOT=root / "outputs",
                    SCAIL_PYTHON=root / "python",
                    SCAIL_RUNNER=root / "runner.py",
                ),
                patch.object(
                    pipeline,
                    "run_cancelable_process",
                    side_effect=fake_process,
                ),
            ):
                metadata = pipeline.prepare_eating_qwen_geometry_guide(
                    "eat-guide-job",
                    source,
                    target,
                    frames=81,
                    fps=16.0,
                )

        command = captured["command"]
        self.assertEqual(captured["job_id"], "eat-guide-job")
        self.assertEqual(command[command.index("--source") + 1], str(source))
        self.assertEqual(command[command.index("--action") + 1], "eating")
        self.assertIn("--eating-geometry-guide-output", command)
        self.assertIn("--eating-geometry-guide-metadata", command)
        self.assertEqual(
            metadata["policy"],
            "driving_eating_geometry_image2_v4",
        )
        self.assertEqual(
            metadata["method"],
            "temporal_head_lowering_continuous_muzzle_semantic_food_fields_v5",
        )
        self.assertEqual(
            metadata["muzzle_contact_semantic"]["muzzle_food_gap_px"],
            0,
        )
        self.assertTrue(metadata["semantic_food"]["passed"])
        self.assertFalse(metadata["palette"]["source_rgb_preserved"])
        self.assertTrue(metadata["head_lowering_selection"]["passed"])
        self.assertFalse(metadata["semantic_structure"]["source_luma_values_preserved"])
        self.assertRegex(metadata["guide_sha256"], r"^[0-9a-f]{64}$")

    def test_grooming_qwen_geometry_guide_uses_versioned_runner_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "TM.mp4"
            source.write_bytes(b"main-case-tm")
            target = root / "grooming-guide.png"
            captured: dict[str, object] = {}

            def fake_process(
                target_job_id: str,
                command: list[str],
                **kwargs: object,
            ) -> None:
                captured["job_id"] = target_job_id
                captured["command"] = command
                captured["kwargs"] = kwargs
                gate_target = Path(
                    command[command.index("--grooming-geometry-guide-output") + 1]
                )
                Image.new("RGB", (512, 512), (0, 255, 0)).save(gate_target)
                with Image.open(gate_target) as loaded:
                    image = loaded.copy()
                image.paste((112, 112, 112), (112, 72, 256, 440))
                image.paste((168, 168, 168), (256, 72, 400, 440))
                image.save(gate_target)
                metadata_path = Path(
                    command[command.index("--grooming-geometry-guide-metadata") + 1]
                )
                metadata_path.write_text(
                    json.dumps(
                        {
                            "policy": "driving_grooming_geometry_image1_v3",
                            "source_sha256": pipeline.file_sha256(source),
                            "palette": {"source_rgb_preserved": False},
                            "raised_paw_muzzle_relation": {
                                "passed": True,
                                "source_luma_values_preserved": False,
                                "black_or_tool_semantics_preserved": False,
                                "output_encoding": {
                                    "method": (
                                        "compressed_blurred_quantized_neutral_anatomy"
                                    )
                                },
                                "shoulder_anchor": [194, 342],
                                "paw_center": [190, 261],
                                "raised_foreleg_region_bbox": [
                                    168,
                                    242,
                                    214,
                                    362,
                                ],
                            },
                        }
                    ),
                    encoding="utf-8",
                )

            with (
                patch.multiple(
                    pipeline,
                    OUTPUT_ROOT=root / "outputs",
                    SCAIL_PYTHON=root / "python",
                    SCAIL_RUNNER=root / "runner.py",
                ),
                patch.object(
                    pipeline,
                    "run_cancelable_process",
                    side_effect=fake_process,
                ),
            ):
                metadata = pipeline.prepare_grooming_qwen_geometry_guide(
                    "groom-guide-job",
                    source,
                    target,
                    frames=81,
                    fps=16.0,
                )
                with Image.open(target) as loaded:
                    qwen_rgb = np.asarray(loaded.convert("RGB"), dtype=np.uint8)
                qwen_foreground = ~np.all(qwen_rgb == (0, 255, 0), axis=2)
                captured["qwen_unique_foreground"] = np.unique(
                    qwen_rgb[qwen_foreground], axis=0
                ).tolist()
                private_gate_path = Path(metadata["private_pose_gate"]["guide_path"])
                with Image.open(private_gate_path) as loaded:
                    private_rgb = np.asarray(loaded.convert("RGB"), dtype=np.uint8)
                private_foreground = ~np.all(private_rgb == (0, 255, 0), axis=2)
                captured["private_unique_foreground_count"] = len(
                    np.unique(private_rgb[private_foreground], axis=0)
                )
                completion_mask_path = Path(metadata["private_completion_mask"]["path"])
                with Image.open(completion_mask_path) as loaded:
                    completion_rgb = np.asarray(loaded.convert("RGB"), dtype=np.uint8)
                completion_foreground = ~np.all(completion_rgb == (0, 255, 0), axis=2)
                captured["completion_unique_foreground"] = np.unique(
                    completion_rgb[completion_foreground], axis=0
                ).tolist()
                captured["completion_foreleg_pixels"] = int(
                    np.count_nonzero(
                        np.all(
                            completion_rgb
                            == np.asarray(
                                pipeline.GROOMING_QWEN_GUIDE_FORELEG_RGB,
                                dtype=np.uint8,
                            ),
                            axis=2,
                        )
                    )
                )
                resolved_gate, resolved_metadata, resolved_completion = (
                    pipeline.resolve_main_case_grooming_pose_gate_metadata(
                        {
                            "pose_reference_metadata": {
                                "grooming": {
                                    "policy": (
                                        pipeline.GROOMING_QWEN_POSE_REFERENCE_POLICY
                                    ),
                                    "generator_policy": metadata["policy"],
                                    "guide_sha256": metadata["guide_sha256"],
                                    "private_pose_gate_guide_path": metadata[
                                        "private_pose_gate"
                                    ]["guide_path"],
                                    "private_pose_gate_guide_sha256": metadata[
                                        "private_pose_gate"
                                    ]["guide_sha256"],
                                    "private_pose_gate_exposed_to_qwen": False,
                                    "private_completion_mask_path": str(
                                        completion_mask_path
                                    ),
                                    "private_completion_mask_sha256": metadata[
                                        "private_completion_mask"
                                    ]["sha256"],
                                    "private_completion_mask_exposed_to_qwen": False,
                                    "generator_metadata": metadata,
                                }
                            }
                        },
                        target,
                    )
                )
                captured["resolved_gate"] = resolved_gate
                captured["resolved_metadata"] = resolved_metadata
                captured["resolved_completion"] = resolved_completion

        command = captured["command"]
        self.assertEqual(captured["job_id"], "groom-guide-job")
        self.assertEqual(command[command.index("--source") + 1], str(source))
        self.assertEqual(command[command.index("--action") + 1], "grooming")
        self.assertIn("--grooming-geometry-guide-output", command)
        self.assertIn("--grooming-geometry-guide-metadata", command)
        self.assertEqual(
            metadata["policy"],
            pipeline.GROOMING_QWEN_GEOMETRY_GUIDE_POLICY,
        )
        self.assertFalse(metadata["palette"]["source_rgb_preserved"])
        self.assertFalse(metadata["palette"]["internal_luma_variation"])
        self.assertEqual(
            metadata["palette"]["internal_luma_variation_role"],
            "none",
        )
        self.assertFalse(
            metadata["palette"]["appearance_bearing_internal_luma_variation"]
        )
        self.assertFalse(metadata["qwen_guide"]["main_case_appearance_exposed_to_qwen"])
        self.assertEqual(
            metadata["qwen_guide"]["foreground_rgb"],
            list(pipeline.GROOMING_QWEN_GUIDE_FOREGROUND_RGB),
        )
        self.assertEqual(
            metadata["qwen_guide"]["unique_foreground_rgb_count"],
            1,
        )
        self.assertEqual(
            metadata["qwen_guide"]["foreground_palette_rgb"],
            [list(pipeline.GROOMING_QWEN_GUIDE_FOREGROUND_RGB)],
        )
        self.assertFalse(metadata["qwen_guide"]["internal_luma_variation"])
        self.assertEqual(
            metadata["qwen_guide"]["internal_luma_variation_role"],
            "none",
        )
        self.assertFalse(
            metadata["qwen_guide"]["appearance_bearing_internal_luma_variation"]
        )
        self.assertEqual(
            metadata["qwen_guide"]["role"],
            "image2_action_geometry_guide",
        )
        structural_negative_space = metadata["qwen_guide"][
            "structural_negative_space"
        ]
        self.assertEqual(
            structural_negative_space["method"],
            pipeline.GROOMING_QWEN_NEGATIVE_SPACE_METHOD,
        )
        self.assertFalse(structural_negative_space["curve_closed"])
        self.assertTrue(structural_negative_space["paw_endpoint_open"])
        self.assertTrue(structural_negative_space["shoulder_endpoint_open"])
        self.assertFalse(structural_negative_space["source_rgb_or_luma_sampled"])
        self.assertGreaterEqual(structural_negative_space["pixels"], 300)
        self.assertLessEqual(structural_negative_space["pixels"], 1800)
        self.assertIsNone(metadata["qwen_guide"]["synthetic_foreleg_semantic"])
        self.assertFalse(metadata["qwen_guide"]["closed_foreleg_capsule_exposed"])
        self.assertTrue(metadata["qwen_guide"]["internal_green_separator_exposed"])
        self.assertEqual(
            metadata["qwen_guide"]["raised_foreleg_validation_role"],
            "public_open_curve_geometry_plus_private_cv",
        )
        self.assertEqual(
            int(
                np.count_nonzero(
                    np.all(
                        qwen_rgb
                        == np.asarray(
                            pipeline.GROOMING_QWEN_GUIDE_FORELEG_RGB,
                            dtype=np.uint8,
                        ),
                        axis=2,
                    )
                )
            ),
            0,
        )
        self.assertEqual(
            metadata["qwen_guide"]["foreground_pixels"],
            metadata["qwen_guide"]["private_foreground_pixels"]
            - structural_negative_space["pixels"],
        )
        self.assertEqual(
            int(np.count_nonzero(private_foreground & ~qwen_foreground)),
            structural_negative_space["pixels"],
        )
        self.assertFalse(np.any(qwen_foreground & ~private_foreground))
        self.assertFalse(metadata["private_pose_gate"]["exposed_to_qwen"])
        self.assertFalse(metadata["private_completion_mask"]["exposed_to_qwen"])
        self.assertFalse(metadata["private_completion_mask"]["loaded_as_qwen_image"])
        self.assertEqual(
            metadata["private_completion_mask"]["policy"],
            pipeline.GROOMING_PRIVATE_COMPLETION_MASK_POLICY,
        )
        self.assertEqual(
            captured["completion_unique_foreground"],
            [
                list(pipeline.GROOMING_QWEN_GUIDE_FOREGROUND_RGB),
                list(pipeline.GROOMING_QWEN_GUIDE_FORELEG_RGB),
            ],
        )
        self.assertGreaterEqual(captured["completion_foreleg_pixels"], 650)
        self.assertEqual(
            captured["resolved_gate"],
            Path(metadata["private_pose_gate"]["guide_path"]),
        )
        self.assertEqual(
            captured["resolved_metadata"],
            Path(metadata["private_pose_gate"]["metadata_path"]),
        )
        self.assertEqual(
            captured["resolved_completion"],
            Path(metadata["private_completion_mask"]["path"]),
        )
        self.assertEqual(
            metadata["private_pose_gate"]["policy"],
            pipeline.GROOMING_PRIVATE_POSE_GATE_GUIDE_POLICY,
        )
        self.assertNotEqual(
            Path(command[command.index("--grooming-geometry-guide-output") + 1]),
            target,
        )
        self.assertEqual(
            captured["qwen_unique_foreground"],
            [list(pipeline.GROOMING_QWEN_GUIDE_FOREGROUND_RGB)],
        )
        self.assertGreater(
            captured["private_unique_foreground_count"],
            1,
        )
        self.assertRegex(metadata["guide_sha256"], r"^[0-9a-f]{64}$")

    def test_eating_contact_gate_accepts_broad_centered_stylized_muzzle(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            candidate, colored_mask = _write_eating_contact_fixture(
                Path(temporary),
                name="broad-contact",
                muzzle_width_bowl_ratio=0.62,
                muzzle_gap_px=0,
            )
            with patch(
                "scail2_subject_replacement_test.make_eating_reference_identity_mask",
                return_value=(colored_mask, None, {"fixture": "broad-contact"}),
            ):
                result = pipeline.eating_mouth_bowl_contact_quality(candidate)

        self.assertTrue(result["acceptable"])
        self.assertEqual(result["method"], pipeline.EATING_MOUTH_BOWL_CONTACT_POLICY)
        self.assertEqual(result["muzzle_bowl_gap_px"], 0)
        self.assertAlmostEqual(
            result["selected_component"]["width_bowl_ratio"],
            0.616667,
            places=6,
        )
        self.assertEqual(
            result["thresholds"]["component_width_bowl_ratio_max"],
            0.70,
        )
        self.assertEqual(
            result["selected_component"]["rejection_reasons"],
            [],
        )

    def test_eating_prop_split_falls_back_from_thin_warm_rim_to_beige_bowl(
        self,
    ) -> None:
        from scail2_subject_replacement_test import (
            make_eating_reference_identity_mask,
        )

        with tempfile.TemporaryDirectory() as temporary:
            candidate = Path(temporary) / "thin-warm-rim-beige-bowl.png"
            _write_thin_warm_rim_beige_bowl_fixture(candidate)
            with Image.open(candidate) as loaded:
                colored, exclusion, metadata = make_eating_reference_identity_mask(
                    loaded.convert("RGBA")
                )

        colored_array = np.asarray(colored.convert("RGB"))
        red_pixels = int(np.count_nonzero(np.all(colored_array == (255, 0, 0), axis=2)))
        blue_pixels = int(
            np.count_nonzero(np.all(colored_array == (0, 0, 255), axis=2))
        )
        self.assertIsNotNone(exclusion)
        self.assertTrue(metadata["warm_component_initially_classified_as_bowl"])
        self.assertTrue(metadata["warm_bowl_core_fallback_to_beige"])
        self.assertEqual(
            metadata["bowl_body_method"],
            pipeline.EATING_BOWL_SPLIT_FALLBACK_POLICY,
        )
        self.assertGreater(red_pixels, 100)
        self.assertGreater(blue_pixels, 500)
        self.assertGreaterEqual(metadata["safe_width_fraction"], 0.25)
        self.assertLessEqual(metadata["safe_width_fraction"], 0.55)
        self.assertGreaterEqual(metadata["prop_fraction"], 0.04)
        self.assertLessEqual(metadata["prop_fraction"], 0.20)
        self.assertGreaterEqual(metadata["left_paw_identity_support"], 500)
        self.assertGreaterEqual(metadata["right_paw_identity_support"], 500)

    def test_eating_contact_gate_rejects_broad_muzzle_with_air_gap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            candidate, colored_mask = _write_eating_contact_fixture(
                Path(temporary),
                name="broad-gap",
                muzzle_width_bowl_ratio=0.62,
                muzzle_gap_px=31,
            )
            with patch(
                "scail2_subject_replacement_test.make_eating_reference_identity_mask",
                return_value=(colored_mask, None, {"fixture": "broad-gap"}),
            ):
                result = pipeline.eating_mouth_bowl_contact_quality(candidate)

        self.assertFalse(result["acceptable"])
        self.assertEqual(result["issues"], ["eating_muzzle_not_at_bowl"])
        nearest = result["component_candidates"][0]
        self.assertEqual(nearest["gap_px"], 31)
        self.assertIn("gap_outside_contact_range", nearest["rejection_reasons"])

    def test_eating_contact_gate_keeps_broad_component_upper_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            candidate, colored_mask = _write_eating_contact_fixture(
                Path(temporary),
                name="too-wide-contact",
                muzzle_width_bowl_ratio=0.705,
                muzzle_gap_px=0,
            )
            with patch(
                "scail2_subject_replacement_test.make_eating_reference_identity_mask",
                return_value=(colored_mask, None, {"fixture": "too-wide-contact"}),
            ):
                result = pipeline.eating_mouth_bowl_contact_quality(candidate)

        self.assertFalse(result["acceptable"])
        self.assertEqual(result["issues"], ["eating_muzzle_not_at_bowl"])
        nearest = result["component_candidates"][0]
        self.assertEqual(nearest["width_bowl_ratio"], 0.708333)
        self.assertIn("width_outside_muzzle_range", nearest["rejection_reasons"])

    def test_fur_surface_retry_preserves_only_configured_eating_recovery(
        self,
    ) -> None:
        eating_profiles = [
            {"name": "pose_capped"},
            {"name": "pose_reduced_retry"},
            {"name": "pose_minimal_retry"},
            {"name": "bowl_palette_canvas_seed_retry_1"},
            {"name": "bowl_palette_canvas_seed_retry_2"},
        ]
        before_first_recovery = pipeline.resolve_fur_surface_retry_decision(
            action="eating",
            rejection_streak=3,
            attempt=3,
            active_profiles=eating_profiles,
        )
        self.assertEqual(
            before_first_recovery["policy"],
            pipeline.FUR_SURFACE_GATE_RETRY_POLICY,
        )
        self.assertTrue(before_first_recovery["threshold_reached"])
        self.assertTrue(before_first_recovery["pending_eating_bowl_recovery"])
        self.assertFalse(before_first_recovery["short_circuit_due"])
        self.assertEqual(
            before_first_recovery["remaining_profiles"],
            [
                "bowl_palette_canvas_seed_retry_1",
                "bowl_palette_canvas_seed_retry_2",
            ],
        )

        between_recoveries = pipeline.resolve_fur_surface_retry_decision(
            action="eating",
            rejection_streak=4,
            attempt=4,
            active_profiles=eating_profiles,
        )
        self.assertTrue(between_recoveries["threshold_reached"])
        self.assertTrue(between_recoveries["pending_eating_bowl_recovery"])
        self.assertFalse(between_recoveries["short_circuit_due"])
        self.assertEqual(
            between_recoveries["remaining_profiles"],
            ["bowl_palette_canvas_seed_retry_2"],
        )
        self.assertEqual(between_recoveries["attempt_budget"], 5)

        after_recovery = pipeline.resolve_fur_surface_retry_decision(
            action="eating",
            rejection_streak=5,
            attempt=5,
            active_profiles=eating_profiles,
        )
        self.assertFalse(after_recovery["pending_eating_bowl_recovery"])
        self.assertTrue(after_recovery["short_circuit_due"])
        self.assertTrue(after_recovery["blocks_scail_on_exhaustion"])
        self.assertEqual(after_recovery["remaining_profiles"], [])
        self.assertEqual(after_recovery["attempt_budget"], 5)

        grooming_final = pipeline.resolve_fur_surface_retry_decision(
            action="grooming",
            rejection_streak=3,
            attempt=3,
            active_profiles=eating_profiles[:3],
        )
        self.assertFalse(grooming_final["pending_eating_bowl_recovery"])
        self.assertTrue(grooming_final["short_circuit_due"])
        self.assertTrue(grooming_final["blocks_scail_on_exhaustion"])

        grooming_before_threshold = pipeline.resolve_fur_surface_retry_decision(
            action="grooming",
            rejection_streak=2,
            attempt=2,
            active_profiles=eating_profiles[:3],
        )
        self.assertFalse(grooming_before_threshold["threshold_reached"])
        self.assertFalse(grooming_before_threshold["short_circuit_due"])

        generic_eating_retry = pipeline.resolve_fur_surface_retry_decision(
            action="eating",
            rejection_streak=3,
            attempt=3,
            active_profiles=[*eating_profiles[:3], {"name": "generic_retry"}],
        )
        self.assertFalse(generic_eating_retry["pending_eating_bowl_recovery"])
        self.assertTrue(generic_eating_retry["short_circuit_due"])

        non_adjacent_bowl_retry = pipeline.resolve_fur_surface_retry_decision(
            action="eating",
            rejection_streak=3,
            attempt=3,
            active_profiles=[
                *eating_profiles[:3],
                {"name": "generic_retry"},
                {"name": "bowl_palette_canvas_seed_retry_1"},
            ],
        )
        self.assertEqual(non_adjacent_bowl_retry["next_profile"], "generic_retry")
        self.assertFalse(non_adjacent_bowl_retry["pending_eating_bowl_recovery"])
        self.assertTrue(non_adjacent_bowl_retry["short_circuit_due"])

    def test_uploaded_surface_candidate_uses_absolute_identity_contract(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cutout = Path(temporary) / "candidate.png"
            image = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
            ImageDraw.Draw(image).ellipse(
                (24, 12, 104, 116),
                fill=(128, 128, 128, 255),
            )
            image.save(cutout)
            effective = {
                "upload_validation": {
                    "uploaded_identity_appearance_authority": True,
                },
                "actions": {
                    "eating": {
                        "fur_surface_gate": {
                            "candidate_comparison_mode": (
                                pipeline.UPLOADED_IDENTITY_SURFACE_COMPARISON_MODE
                            ),
                        }
                    }
                },
            }
            evaluated = {
                "acceptable": True,
                "verdict": "accepted_uploaded_identity_absolute_contract",
                "issues": [],
            }
            with patch.object(
                pipeline,
                "evaluate_fur_surface",
                return_value=evaluated,
            ) as evaluator:
                result = pipeline.evaluate_main_case_fur_surface_candidate(
                    cutout,
                    action="eating",
                    alpha_provenance="chroma_key",
                    effective_subject_config=effective,
                )

        args, kwargs = evaluator.call_args
        self.assertEqual(args[0], cutout)
        self.assertIsNone(args[1])
        self.assertTrue(kwargs["accept_absolute_contract"])
        self.assertFalse(kwargs["baseline_is_validated"])
        self.assertTrue(result["acceptable"])
        self.assertEqual(
            result["gate"]["candidate_comparison_mode"],
            pipeline.UPLOADED_IDENTITY_SURFACE_COMPARISON_MODE,
        )
        self.assertFalse(
            result["gate"]["profile_pixels_used_for_candidate_appearance_validation"]
        )

    def test_eating_pose_profiles_append_two_finite_minimal_seed_retries(
        self,
    ) -> None:
        profiles = [
            {
                "name": "pose_capped",
                "user_loras": [{"model_name": "first.safetensors", "strength": 0.2}],
                "quality_retry": 1,
                "steps": 8,
            },
            {
                "name": "pose_reduced_retry",
                "user_loras": [{"model_name": "second.safetensors", "strength": 0.1}],
                "quality_retry": 2,
                "steps": 9,
            },
            {
                "name": "pose_minimal_retry",
                "user_loras": [],
                "quality_retry": 3,
                "steps": 10,
            },
        ]

        with patch.object(pipeline, "QWEN_BOWL_POSE_EXTRA_ATTEMPTS", 2):
            selected = pipeline.pose_generation_profiles("eating", profiles)

        self.assertEqual(
            [profile["name"] for profile in selected],
            [
                "pose_capped",
                "pose_reduced_retry",
                "pose_minimal_retry",
                "bowl_palette_canvas_seed_retry_1",
                "bowl_palette_canvas_seed_retry_2",
            ],
        )
        self.assertEqual(
            [profile["quality_retry"] for profile in selected[-2:]],
            [4, 5],
        )
        self.assertEqual(
            [profile["steps"] for profile in selected[-2:]],
            [10, 10],
        )
        self.assertEqual(selected[-2]["user_loras"], [])
        self.assertEqual(selected[-1]["user_loras"], [])
        self.assertIsNot(selected[-2]["user_loras"], profiles[-1]["user_loras"])
        self.assertIsNot(selected[-1]["user_loras"], profiles[-1]["user_loras"])
        self.assertIsNot(selected[-1]["user_loras"], selected[-2]["user_loras"])
        self.assertEqual(profiles[-1]["name"], "pose_minimal_retry")

    def test_normalize_qwen_action_candidate_returns_direct_cutout_without_semantic_matte(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "candidate.png"
            direct_cutout = root / "direct-cutout.png"
            semantic_green = root / "semantic-green.png"
            semantic_cutout = root / "semantic-cutout.png"
            semantic_raw = root / "semantic-raw.png"
            source.write_bytes(b"candidate")
            direct_metrics = {
                "acceptable": True,
                "issues": [],
                "method": "chroma_key",
            }

            def fake_direct_evaluation(
                _source: Path,
                target: Path,
            ) -> dict[str, object]:
                image = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
                image.paste((80, 90, 100, 255), (180, 220, 844, 900))
                image.save(target)
                return direct_metrics

            with (
                patch.object(
                    pipeline,
                    "evaluate_chroma_cutout",
                    side_effect=fake_direct_evaluation,
                ) as evaluate_direct,
                patch.object(
                    pipeline,
                    "semantic_background_recovery",
                ) as recover_semantic,
            ):
                metrics, selected_cutout = pipeline.normalize_qwen_action_candidate(
                    source,
                    direct_cutout,
                    semantic_green,
                    semantic_cutout,
                    semantic_raw,
                    pose="eating",
                    allow_semantic_matte=True,
                )
            with Image.open(selected_cutout) as normalized:
                selected_size = normalized.size
                selected_mode = normalized.mode
            selected_digest = pipeline.file_sha256(selected_cutout)

        evaluate_direct.assert_called_once_with(source, direct_cutout)
        recover_semantic.assert_not_called()
        self.assertNotEqual(selected_cutout, direct_cutout)
        self.assertEqual(selected_cutout.name, "direct-cutout_reference_512.png")
        self.assertEqual(selected_size, (512, 512))
        self.assertEqual(selected_mode, "RGBA")
        self.assertTrue(metrics["acceptable"])
        self.assertEqual(metrics["direct_chroma"], direct_metrics)
        self.assertFalse(metrics["semantic_recovery_attempted"])
        self.assertIsNone(metrics["semantic_recovery_acceptable"])
        self.assertTrue(metrics["normalized_after_qwen"])
        self.assertEqual(
            metrics["normalization_policy"],
            pipeline.QWEN_ACTION_REFERENCE_NORMALIZATION_POLICY,
        )
        self.assertEqual(
            metrics["matte_policy"],
            pipeline.QWEN_ACTION_CANDIDATE_MATTE_POLICY,
        )
        self.assertEqual(
            metrics["final_reference_sha256"],
            selected_digest,
        )

    def test_normalize_qwen_action_candidate_uses_semantic_matte_after_direct_failure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "candidate.png"
            direct_cutout = root / "direct-cutout.png"
            semantic_green = root / "semantic-green.png"
            semantic_cutout = root / "semantic-cutout.png"
            semantic_raw = root / "semantic-raw.png"
            source.write_bytes(b"candidate")
            direct_metrics = {
                "acceptable": False,
                "issues": ["green_background_not_clean"],
                "method": "chroma_key",
            }
            semantic_metrics = {
                "acceptable": True,
                "issues": [],
                "method": "rembg_u2net_fallback",
            }

            def fake_semantic_recovery(
                _source: Path,
                _green: Path,
                target: Path,
                _raw: Path,
                *,
                pose: str,
            ) -> dict[str, object]:
                self.assertEqual(pose, "eating")
                image = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
                image.paste((80, 90, 100, 255), (180, 220, 844, 900))
                image.save(target)
                return semantic_metrics

            with (
                patch.object(
                    pipeline,
                    "evaluate_chroma_cutout",
                    return_value=direct_metrics,
                ) as evaluate_direct,
                patch.object(
                    pipeline,
                    "semantic_background_recovery",
                    side_effect=fake_semantic_recovery,
                ) as recover_semantic,
            ):
                metrics, selected_cutout = pipeline.normalize_qwen_action_candidate(
                    source,
                    direct_cutout,
                    semantic_green,
                    semantic_cutout,
                    semantic_raw,
                    pose="eating",
                    allow_semantic_matte=True,
                )
            with Image.open(selected_cutout) as normalized:
                selected_size = normalized.size
                selected_mode = normalized.mode

        evaluate_direct.assert_called_once_with(source, direct_cutout)
        recover_semantic.assert_called_once_with(
            source,
            semantic_green,
            semantic_cutout,
            semantic_raw,
            pose="eating",
        )
        self.assertNotEqual(selected_cutout, semantic_cutout)
        self.assertEqual(selected_cutout.name, "semantic-cutout_reference_512.png")
        self.assertEqual(selected_size, (512, 512))
        self.assertEqual(selected_mode, "RGBA")
        self.assertTrue(metrics["acceptable"])
        self.assertEqual(metrics["direct_chroma"], direct_metrics)
        self.assertTrue(metrics["semantic_recovery_attempted"])
        self.assertTrue(metrics["semantic_recovery_acceptable"])
        self.assertTrue(metrics["normalized_after_qwen"])
        self.assertEqual(
            metrics["normalization_policy"],
            pipeline.QWEN_ACTION_REFERENCE_NORMALIZATION_POLICY,
        )

    def test_normalize_qwen_action_candidate_audits_semantic_exception_as_failure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "candidate.png"
            direct_cutout = root / "direct-cutout.png"
            semantic_green = root / "semantic-green.png"
            semantic_cutout = root / "semantic-cutout.png"
            semantic_raw = root / "semantic-raw.png"
            source.write_bytes(b"candidate")
            direct_metrics = {
                "acceptable": False,
                "issues": ["green_background_not_clean"],
                "method": "chroma_key",
            }

            with (
                patch.object(
                    pipeline,
                    "evaluate_chroma_cutout",
                    return_value=direct_metrics,
                ) as evaluate_direct,
                patch.object(
                    pipeline,
                    "semantic_background_recovery",
                    side_effect=RuntimeError("semantic matte boom"),
                ) as recover_semantic,
            ):
                metrics, selected_cutout = pipeline.normalize_qwen_action_candidate(
                    source,
                    direct_cutout,
                    semantic_green,
                    semantic_cutout,
                    semantic_raw,
                    pose="grooming",
                    allow_semantic_matte=True,
                )

        evaluate_direct.assert_called_once_with(source, direct_cutout)
        recover_semantic.assert_called_once_with(
            source,
            semantic_green,
            semantic_cutout,
            semantic_raw,
            pose="grooming",
        )
        self.assertEqual(selected_cutout, direct_cutout)
        self.assertFalse(metrics["acceptable"])
        self.assertEqual(metrics["direct_chroma"], direct_metrics)
        self.assertTrue(metrics["semantic_recovery_attempted"])
        self.assertFalse(metrics["semantic_recovery_acceptable"])
        self.assertEqual(
            metrics["semantic_recovery_error"],
            "RuntimeError: semantic matte boom",
        )
        self.assertFalse(metrics["normalized_after_qwen"])
        self.assertEqual(
            metrics["normalization_policy"],
            pipeline.QWEN_ACTION_REFERENCE_NORMALIZATION_POLICY,
        )

    def test_prepare_pose_input_byte_preserves_gated_rgba_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary) / "outputs"
            job_id = "preserve-gated-alpha"
            job_dir = output_root / job_id
            job_dir.mkdir(parents=True)
            source = job_dir / "pose_01_qwen_cutout.png"
            image = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
            image.paste((80, 90, 100, 255), (96, 64, 416, 448))
            image.save(source, format="PNG", optimize=True)
            source_bytes = source.read_bytes()
            jobs = {
                job_id: {
                    "job_id": job_id,
                    "outputs": [],
                    "cancel_requested": False,
                }
            }

            with (
                patch.multiple(
                    pipeline,
                    OUTPUT_ROOT=output_root,
                    JOBS=jobs,
                ),
                patch.object(
                    pipeline,
                    "convert_chroma_image_to_transparent_png",
                ) as chroma_rekey,
            ):
                wan_input = pipeline.prepare_pose_input(
                    job_id,
                    job_dir,
                    1,
                    "eating",
                    source,
                    512,
                    512,
                    transparent=True,
                    preserve_gated_alpha=True,
                )

            published = job_dir / "pose_01_keyframe_transparent.png"
            self.assertEqual(published.read_bytes(), source_bytes)
            self.assertEqual(
                pipeline.file_sha256(published),
                pipeline.file_sha256(source),
            )
            chroma_rekey.assert_not_called()
            with Image.open(wan_input) as green:
                self.assertEqual(green.mode, "RGB")
                self.assertEqual(green.size, (512, 512))
                self.assertEqual(green.getpixel((0, 0)), (0, 255, 0))
                self.assertEqual(green.getpixel((256, 256)), (80, 90, 100))
            output = jobs[job_id]["outputs"][0]
            self.assertTrue(output["byte_preserved_from_gated_candidate"])
            self.assertEqual(
                output["normalization_policy"],
                pipeline.QWEN_ACTION_REFERENCE_NORMALIZATION_POLICY,
            )
            self.assertEqual(output["sha256"], pipeline.file_sha256(source))

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
            **_kwargs: object,
        ) -> dict[str, object]:
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
                "identity_reference_path": str(identity_reference_path),
                "identity_reference_sha256": pipeline.file_sha256(
                    identity_reference_path
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
                    "acceptable": True,
                    "issues": [],
                    "penalty": 0.0,
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

    def test_direct_uploaded_eating_prompt_assigns_image_roles(self) -> None:
        prompt = pipeline.build_qwen_prompt(
            "",
            pipeline.subject_replacement_pose_prompt("eating"),
            transparent_gif=True,
            pose_reference_available=True,
            profile_fur_lock=pipeline.UPLOADED_IDENTITY_FUR_LOCK,
            pose_identity_input_policy=(
                pipeline.POSE_IDENTITY_INPUT_POLICY_DIRECT_UPLOAD
            ),
        )

        self.assertIn(
            "Image 1 is the original uploaded pet photo and is the sole appearance source",
            prompt,
        )
        self.assertIn(
            "Image 2 is the frozen validated subject-profile action reference",
            prompt,
        )
        self.assertIn("bowl placement", prompt)
        self.assertIn("authoritative only for joint placement", prompt)
        self.assertIn("natural coat length and volume", prompt)
        self.assertIn("outer fur silhouette", prompt)
        self.assertIn("copy no fur color, marking, face identity", prompt)
        self.assertIn("Do not copy photographic rendering from Image 1", prompt)
        self.assertIn("Image 2 controls action geometry only", prompt)
        self.assertNotIn("compact-coat silhouette standard", prompt)
        self.assertNotIn(
            "Image 1 is the approved canonical target and the only identity",
            prompt,
        )

    def test_uploaded_identity_policies_fail_closed_without_secondary_image(
        self,
    ) -> None:
        for policy in (
            pipeline.POSE_IDENTITY_INPUT_POLICY_DIRECT_UPLOAD,
            pipeline.POSE_IDENTITY_INPUT_POLICY_PROFILE_GEOMETRY_FIRST,
        ):
            with self.subTest(policy=policy):
                with self.assertRaisesRegex(
                    ValueError,
                    "requires a validated secondary",
                ):
                    pipeline.build_qwen_prompt(
                        "",
                        pipeline.subject_replacement_pose_prompt("eating"),
                        transparent_gif=True,
                        pose_reference_available=False,
                        pose_identity_input_policy=policy,
                    )

        with self.assertRaisesRegex(
            ValueError,
            "requires a validated secondary",
        ):
            pipeline.build_qwen_prompt(
                "",
                pipeline.subject_replacement_pose_prompt("grooming"),
                transparent_gif=True,
                pose_reference_available=False,
                pose_identity_input_policy=(
                    pipeline.POSE_IDENTITY_INPUT_POLICY_GROOMING_GEOMETRY_CANONICAL
                ),
            )

    def test_grooming_geometry_canonical_policy_rejects_other_actions(self) -> None:
        with self.assertRaisesRegex(ValueError, "supports only grooming"):
            pipeline.build_qwen_prompt(
                "",
                pipeline.subject_replacement_pose_prompt("eating"),
                transparent_gif=True,
                pose_reference_available=True,
                pose_identity_input_policy=(
                    pipeline.POSE_IDENTITY_INPUT_POLICY_GROOMING_GEOMETRY_CANONICAL
                ),
            )

    def test_main_case_geometry_first_eating_prompt_assigns_image_roles(self) -> None:
        prompt = pipeline.build_qwen_prompt(
            "",
            pipeline.subject_replacement_pose_prompt("eating"),
            transparent_gif=True,
            pose_reference_available=True,
            profile_fur_lock=pipeline.UPLOADED_IDENTITY_FUR_LOCK,
            pose_identity_input_policy=(
                pipeline.POSE_IDENTITY_INPUT_POLICY_PROFILE_GEOMETRY_FIRST
            ),
        )

        self.assertIn(
            "Image 1 is the texture-free low-frequency geometry guide derived directly from "
            "the frozen Main_case eat.mp4 standard",
            prompt,
        )
        self.assertIn("sampler's geometry canvas", prompt)
        self.assertIn("muzzle-to-kibble contact", prompt)
        self.assertIn(
            "not authoritative for identity, body proportions, coat length", prompt
        )
        self.assertIn(
            "Image 2 is the original uploaded pet photo",
            prompt,
        )
        self.assertIn(
            "Image 1 owns only eating pose, bowl, muzzle contact and framing",
            prompt,
        )
        self.assertIn("The subject profile owns no appearance trait", prompt)
        self.assertIn(
            "sole appearance authority",
            prompt,
        )
        self.assertIn("COAT IDENTITY LOCK", prompt)
        self.assertIn("genuine tufts or ruff", prompt)
        self.assertIn("dedicated pet3d style model", prompt)
        self.assertIn(
            "separate identity-palette audit remains locked to the original "
            "uploaded file",
            prompt,
        )
        self.assertNotIn(
            "Image 1 is the original uploaded pet photo and is the sole appearance source",
            prompt,
        )
        self.assertNotIn(
            "Image 1 is the approved canonical target and the only identity",
            prompt,
        )
        self.assertNotIn("short dense smooth coat", prompt)
        self.assertNotIn("profile-controlled", prompt)

    def test_upload_locked_canonical_prompt_preserves_source_coat(self) -> None:
        prompt = pipeline.build_cartoon_reference_prompt(
            "",
            fur_lock=pipeline.UPLOADED_IDENTITY_FUR_LOCK,
        )

        self.assertIn("UPLOAD COAT SURFACE PRESERVATION", prompt)
        self.assertIn(
            "the uploaded views, not the",
            prompt,
        )
        self.assertIn("control natural coat length, volume", prompt)
        self.assertIn("radial brush strokes", prompt)
        self.assertIn("action profile is never an appearance reference", prompt)
        self.assertIn("Identity and apparent age outrank stylization", prompt)
        self.assertIn("never make it younger or more kitten-like", prompt)
        self.assertNotIn("premium cute", prompt)
        self.assertNotIn("animated-film", prompt)
        self.assertIn("Do not use chibi enlargement", prompt)
        self.assertNotIn("slightly oversized round head", prompt)

    def test_profile_geometry_first_final_bowl_retry_has_consistent_contact(
        self,
    ) -> None:
        prompt = pipeline.build_qwen_prompt(
            "",
            pipeline.subject_replacement_pose_prompt("eating"),
            transparent_gif=True,
            quality_retry=4,
            pose_reference_available=True,
            pose_identity_input_policy=(
                pipeline.POSE_IDENTITY_INPUT_POLICY_PROFILE_GEOMETRY_FIRST
            ),
        )

        self.assertIn(
            "Keep exactly one shallow bowl touching or slightly overlapping",
            prompt,
        )
        self.assertIn("Never omit the bowl or visible brown kibble", prompt)
        self.assertIn(
            "Do not place a green gap between the bowl and both front paws",
            prompt,
        )
        self.assertNotIn("narrow pure-green gap", prompt)

    def test_profile_geometry_first_grooming_prompt_assigns_image_roles(self) -> None:
        prompt = pipeline.build_qwen_prompt(
            "",
            pipeline.subject_replacement_pose_prompt("grooming"),
            transparent_gif=True,
            pose_reference_available=True,
            pose_identity_input_policy=(
                pipeline.POSE_IDENTITY_INPUT_POLICY_PROFILE_GEOMETRY_FIRST
            ),
        )

        self.assertIn("frozen Main_case TM.mp4 standard", prompt)
        self.assertIn("Image 2 is the original uploaded pet photo", prompt)
        self.assertIn(
            "fixed RGB(176,176,176) capsule represents only the continuous shoulder-to-mouth raised foreleg",
            prompt,
        )
        self.assertIn("generated only from verified geometry", prompt)
        self.assertIn("one continuous raised foreleg and rounded paw", prompt)
        self.assertIn(
            "never as a sleeve, strap, zipper, cord, pen, stick, pendant",
            prompt,
        )
        self.assertIn(
            "one foreleg rises continuously from its shoulder to one rounded paw",
            prompt,
        )
        self.assertNotIn("all four paws must be fully visible", prompt)

    def test_default_grooming_prompt_keeps_canonical_image1_role(self) -> None:
        prompt = pipeline.build_qwen_prompt(
            "",
            pipeline.subject_replacement_pose_prompt("grooming"),
            transparent_gif=True,
            pose_reference_available=True,
        )

        self.assertTrue(
            prompt.startswith(
                "CRITICAL OUTPUT CONTRACT - satisfy this before every style detail:"
            )
        )
        self.assertIn(
            "Image 1 is the approved canonical target and the only identity",
            prompt,
        )
        self.assertIn(
            "Image 2 is an appearance-free action-layout silhouette derived "
            "directly from the digest-locked neutral-white anatomy-maquette "
            "driving video",
            prompt,
        )
        self.assertIn("same artificial RGB(144,144,144)", prompt)
        self.assertIn(
            "Never translate the artificial gray into black, white, gray, brown, orange",
            prompt,
        )
        self.assertIn("Copy Image 2's seated grooming joint geometry", prompt)
        self.assertIn("Do not mirror the white-maquette limb relationship", prompt)
        self.assertIn("only the foreleg on the viewer's left rises", prompt)
        self.assertIn("near (38% width, 67% height)", prompt)
        self.assertIn("centered near (37% width, 51% height)", prompt)
        self.assertIn(
            "A natural bent furry foreleg in this region is mandatory",
            prompt,
        )
        self.assertIn(
            "no continuous black or dark stroke from muzzle through paw",
            prompt,
        )
        self.assertIn(
            "an unstriped region must remain unstriped",
            prompt,
        )
        self.assertIn(
            "the no-prop rule forbids only a detached rigid item",
            prompt,
        )
        self.assertNotIn("or other vertical item at the mouth", prompt)
        self.assertIn("natural coat-length, coat-volume, outer-silhouette", prompt)
        self.assertIn(
            "Image 2 and the white-maquette driver never control any appearance trait",
            prompt,
        )
        self.assertIn(
            "hind legs remain anatomically attached and may be naturally tucked",
            prompt,
        )
        self.assertNotIn("exactly four complete paws", prompt.lower())
        self.assertNotIn("all four paws must be fully visible", prompt.lower())
        self.assertIn(
            "Style source: always inherit the approved canonical reference in Image 1",
            prompt,
        )
        self.assertNotIn("Image 1 is the original uploaded pet photo", prompt)

    def test_grooming_action_transfer_prompt_keeps_canonical_identity(self) -> None:
        prompt = pipeline.build_advisory_grooming_action_transfer_prompt(
            pipeline.subject_replacement_pose_prompt("grooming"),
            profile_fur_lock=pipeline.UPLOADED_IDENTITY_FUR_LOCK,
        )

        self.assertIn(
            "IMAGE 1 is the approved Qwen full-body canonical",
            prompt,
        )
        self.assertIn(
            "IMAGE 2 is a deliberately HEADLESS neutral body-and-raised-foreleg",
            prompt,
        )
        self.assertIn("derived only from frame 0", prompt)
        self.assertIn("complete maquette head and both maquette ears", prompt)
        self.assertIn("green opening above the neck", prompt)
        self.assertIn("IMAGE 3 is a deterministic enlarged head crop", prompt)
        self.assertIn("exact recognizable face", prompt)
        self.assertIn(
            "Image 2 deliberately stops at the neck",
            prompt,
        )
        self.assertIn(
            "The raised viewer-left foreleg replaces its former planted position",
            prompt,
        )
        for forbidden_token in ("microphone", "zipper", "spring", "pendant", "cable"):
            self.assertNotIn(forbidden_token, prompt.lower())
        self.assertIn("an unstriped region remains unstriped", prompt)
        self.assertIn("never an elephant", prompt)
        self.assertIn("no pedestal, plinth, display base, circular platform", prompt)

        refine_prompt = pipeline.build_advisory_grooming_identity_refine_prompt(
            pipeline.subject_replacement_pose_prompt("grooming"),
            source_attempt=1,
            profile_fur_lock=pipeline.UPLOADED_IDENTITY_FUR_LOCK,
        )
        self.assertIn("CONDITIONAL GROOMING IDENTITY REPAIR", refine_prompt)
        self.assertIn("MASKED UPPER-HEAD PASS", refine_prompt)
        self.assertIn("white maquette is absent from this pass", refine_prompt)
        self.assertIn("action draft from attempt 1", refine_prompt)
        self.assertIn("IMAGE 2 is the approved Qwen full-body canonical", refine_prompt)
        self.assertIn("IMAGE 3 is an enlarged crop", refine_prompt)
        self.assertIn("smooth white-model face", refine_prompt)
        self.assertIn("black oval toy nose", refine_prompt)
        self.assertIn("only pixels enabled", refine_prompt)
        self.assertIn("raised paw", refine_prompt)

        retry_anatomy = pipeline.advisory_grooming_retry_anatomy_prompt(
            ["raised_paw_hard_object_texture_signature"]
        )
        self.assertIn("one connected shoulder", retry_anatomy)
        self.assertIn("a softly bent elbow", retry_anatomy)
        self.assertIn("one rounded furry paw", retry_anatomy)
        for forbidden_token in ("microphone", "zipper", "spring", "pendant"):
            self.assertNotIn(forbidden_token, retry_anatomy.lower())
        self.assertFalse(
            pipeline.advisory_grooming_candidate_safe_for_masked_repair(
                {"issues": ["raised_paw_hard_object_texture_signature"]}
            )
        )
        self.assertTrue(
            pipeline.advisory_grooming_candidate_safe_for_masked_repair(
                {"issues": ["raised_paw_too_far_from_muzzle"]}
            )
        )
        self.assertFalse(
            pipeline.advisory_grooming_candidate_safe_for_masked_repair(
                {"issues": ["grooming_face_eye_hue_replacement"]}
            )
        )
        self.assertTrue(
            pipeline.advisory_grooming_candidate_safe_for_identity_refine(
                {
                    "available": True,
                    "acceptable": False,
                    "issues": ["new_dominant_dark_markings"],
                    "pose": {"acceptable": True},
                }
            )
        )
        self.assertFalse(
            pipeline.advisory_grooming_candidate_safe_for_identity_refine(
                {
                    "available": True,
                    "acceptable": False,
                    "issues": ["gross_identity_palette_l1_drift"],
                    "pose": {"acceptable": True},
                }
            )
        )
        self.assertFalse(
            pipeline.advisory_grooming_candidate_safe_for_identity_refine(
                {
                    "available": True,
                    "acceptable": True,
                    "issues": [],
                    "pose": {"acceptable": True},
                }
            )
        )
        self.assertFalse(
            pipeline.advisory_grooming_candidate_safe_for_identity_refine(
                {
                    "available": True,
                    "acceptable": False,
                    "issues": ["new_dominant_dark_markings"],
                    "pose": {"acceptable": False},
                }
            )
        )
        face_retry = pipeline.advisory_grooming_retry_anatomy_prompt(
            ["grooming_face_dark_marking_drift"]
        )
        self.assertIn("enlarged Image 3 crop", face_retry)
        self.assertIn("same cheek width", face_retry)
        self.assertTrue(
            pipeline.advisory_grooming_retry_requires_palette_correction(
                ["gross_chromatic_coat_fraction_drift"]
            )
        )
        self.assertFalse(
            pipeline.advisory_grooming_retry_requires_palette_correction(
                ["raised_paw_hard_object_texture_signature"]
            )
        )

        retry_prompt = pipeline.build_qwen_prompt(
            "",
            pipeline.subject_replacement_pose_prompt("grooming"),
            transparent_gif=True,
            quality_retry=3,
            pose_reference_available=True,
        )
        self.assertIn("GROOMING SIDE RECOVERY", retry_prompt)
        self.assertIn("do not raise the viewer's-right foreleg", retry_prompt)
        self.assertIn("FINAL GROOMING GEOMETRY CORRECTION", retry_prompt)

        soft_completion_prompt = (
            pipeline.build_canonical_grooming_foreleg_completion_prompt(
                "grooming",
                source_attempt=2,
                soft_depth_anatomy=True,
            )
        )
        self.assertIn(
            "Image 3 is the frozen normalized neutral-depth action anatomy",
            soft_completion_prompt,
        )
        self.assertIn(
            "one living cat from head through torso and limbs",
            soft_completion_prompt,
        )
        self.assertNotIn(
            "single-gray action silhouette",
            soft_completion_prompt,
        )

    def test_default_eating_action_workflow_keeps_canonical_image1(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workflow, uploaded_input, pose_input = self._capture_action_workflow(
                Path(temporary),
                action="eating",
                action_identity_policy=(pipeline.POSE_IDENTITY_INPUT_POLICY_CANONICAL),
            )

        load_images = {
            node_id: node["inputs"]["image"]
            for node_id, node in workflow.items()
            if node.get("class_type") == "LoadImage"
        }
        expected_canonical = "local_video_api/capture-eating/cartoon_reference.png"
        self.assertEqual(set(load_images.values()), {expected_canonical, pose_input})
        self.assertEqual(load_images["1"], expected_canonical)
        self.assertNotIn(uploaded_input, load_images.values())
        pose_load_node = next(
            node_id for node_id, image in load_images.items() if image == pose_input
        )
        pose_scale_node = next(
            node_id
            for node_id, node in workflow.items()
            if node.get("class_type") == "FluxKontextImageScale"
            and node["inputs"].get("image") == [pose_load_node, 0]
        )
        self.assertEqual(workflow["6"]["inputs"]["image1"], ["5", 0])
        self.assertEqual(
            workflow["6"]["inputs"]["image2"],
            [pose_scale_node, 0],
        )
        self.assertEqual(workflow["10"]["inputs"]["pixels"], ["5", 0])
        self.assertEqual(workflow["14"]["inputs"]["latent_image"], ["10", 0])
        action_prompt = workflow["6"]["inputs"]["prompt"]
        self.assertTrue(
            action_prompt.startswith(
                "CRITICAL OUTPUT CONTRACT - satisfy this before every style detail:"
            )
        )
        self.assertIn(
            "Image 2 is the texture-free action-layout guide derived directly "
            "from the digest-locked neutral-white anatomy-maquette driving video "
            "and is authoritative only for "
            "joint placement, the exact pose, the action-required prop layout, "
            "framing, subject scale, and pure-green background",
            action_prompt,
        )
        self.assertIn(
            "The lower muzzle is inside the brown kibble",
            action_prompt,
        )
        self.assertIn(
            "rear legs may be naturally tucked or partly occluded",
            action_prompt,
        )
        self.assertNotIn(
            "exactly four complete paws",
            action_prompt.lower(),
        )
        self.assertNotIn(
            "all four paws must be fully visible",
            action_prompt.lower(),
        )
        loaded_loras = {
            node["inputs"]["lora_name"]: node["inputs"]["strength_model"]
            for node in workflow.values()
            if node.get("class_type") == "LoraLoaderModelOnly"
        }
        self.assertEqual(loaded_loras, {pipeline.QWEN_LORA: 1.0})
        photography_lora_names = {
            item["model_name"]
            for item in pipeline.IMAGE_PROFILES[
                pipeline.SUBJECT_REPLACEMENT_REFERENCE_IMAGE_PROFILE
            ]["user_loras"]
        }
        self.assertTrue(photography_lora_names.isdisjoint(loaded_loras))
        self.assertNotIn(self.TEST_QWEN_STYLE_LORA, loaded_loras)
        self.assertEqual(workflow["14"]["inputs"]["model"], ["13", 0])

    def test_default_grooming_action_workflow_keeps_canonical_image1(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workflow, uploaded_input, pose_input = self._capture_action_workflow(
                Path(temporary),
                action="grooming",
                action_identity_policy=(pipeline.POSE_IDENTITY_INPUT_POLICY_CANONICAL),
            )

        load_images = {
            node_id: node["inputs"]["image"]
            for node_id, node in workflow.items()
            if node.get("class_type") == "LoadImage"
        }
        expected_canonical = "local_video_api/capture-grooming/cartoon_reference.png"
        self.assertEqual(set(load_images.values()), {expected_canonical, pose_input})
        self.assertEqual(load_images["1"], expected_canonical)
        self.assertNotIn(uploaded_input, load_images.values())
        pose_load_node = next(
            node_id for node_id, image in load_images.items() if image == pose_input
        )
        pose_scale_node = next(
            node_id
            for node_id, node in workflow.items()
            if node.get("class_type") == "FluxKontextImageScale"
            and node["inputs"].get("image") == [pose_load_node, 0]
        )
        self.assertEqual(workflow["6"]["inputs"]["image1"], ["5", 0])
        self.assertEqual(
            workflow["6"]["inputs"]["image2"],
            [pose_scale_node, 0],
        )
        self.assertEqual(workflow["10"]["inputs"]["pixels"], ["5", 0])
        self.assertEqual(workflow["14"]["inputs"]["latent_image"], ["10", 0])
        self.assertEqual(
            workflow["14"]["inputs"]["denoise"],
            pipeline.QWEN_GROOMING_CANONICAL_GEOMETRY_DENOISE_BY_ATTEMPT[0],
        )
        action_prompt = workflow["6"]["inputs"]["prompt"]
        self.assertIn(
            "Image 1 is the approved canonical target and the only identity",
            action_prompt,
        )
        self.assertIn(
            "Image 2 is an appearance-free action-layout silhouette",
            action_prompt,
        )
        self.assertIn(
            "Copy Image 2's seated grooming joint geometry",
            action_prompt,
        )
        self.assertNotIn(
            "CRITICAL GROOMING TRANSFER CONTRACT",
            action_prompt,
        )
        loaded_loras = {
            node["inputs"]["lora_name"]: node["inputs"]["strength_model"]
            for node in workflow.values()
            if node.get("class_type") == "LoraLoaderModelOnly"
        }
        self.assertEqual(loaded_loras, {pipeline.QWEN_LORA: 1.0})
        self.assertNotIn(self.TEST_QWEN_STYLE_LORA, loaded_loras)

    def test_advisory_grooming_keeps_palette_generation_guidance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workflow, uploaded_input, pose_input = self._capture_action_workflow(
                Path(temporary),
                action="grooming",
                action_identity_policy=(pipeline.POSE_IDENTITY_INPUT_POLICY_CANONICAL),
                identity_palette_guard=True,
                uploaded_white_maquette=True,
                visual_audits_blocking=False,
            )

        action_prompt = workflow["6"]["inputs"]["prompt"]
        negative_prompt = workflow["7"]["inputs"]["prompt"]
        load_images = {
            node["inputs"]["image"]
            for node in workflow.values()
            if node.get("class_type") == "LoadImage"
        }
        expected_canonical = "local_video_api/capture-grooming/cartoon_reference.png"
        expected_action_reference = (
            "local_video_api/capture-grooming/"
            "grooming_white_maquette_frame000_body_pose_action_reference.png"
        )
        expected_face_reference = (
            "local_video_api/capture-grooming/"
            "grooming_canonical_face_identity_reference.png"
        )
        self.assertEqual(
            load_images,
            {
                expected_canonical,
                expected_action_reference,
                expected_face_reference,
            },
        )
        self.assertNotIn(pose_input, load_images)
        self.assertNotIn(uploaded_input, load_images)
        self.assertIn(
            "CRITICAL GROOMING FIRST-FRAME IDENTITY TRANSFER",
            action_prompt,
        )
        self.assertIn("derived only from frame 0", action_prompt)
        self.assertIn("IMAGE 3 is a deterministic enlarged head crop", action_prompt)
        self.assertIn(
            "IMAGE 2 is a deliberately HEADLESS neutral body-and-raised-foreleg",
            action_prompt,
        )
        self.assertIn("complete maquette head and both maquette ears", action_prompt)
        self.assertIn("green opening above the neck", action_prompt)
        self.assertIn("image3", workflow["6"]["inputs"])
        self.assertIn("image3", workflow["7"]["inputs"])
        self.assertIn("visible source coat is achromatic", action_prompt)
        self.assertIn("black seam, dark seam, dark contour line", negative_prompt)
        self.assertEqual(
            workflow["14"]["inputs"]["denoise"],
            pipeline.GROOMING_ADVISORY_ACTION_TRANSFER_DENOISE,
        )
        self.assertEqual(
            workflow["14"]["inputs"]["steps"],
            pipeline.GROOMING_ADVISORY_ACTION_TRANSFER_STEPS,
        )

    def test_advisory_grooming_identity_safe_draft_skips_generative_face_refine(
        self,
    ) -> None:
        minimum_gate = {
            "available": True,
            "acceptable": True,
            "issues": [],
            "pose": {"acceptable": True},
        }

        self.assertFalse(
            pipeline.advisory_grooming_candidate_safe_for_identity_refine(
                minimum_gate
            )
        )
        self.assertLess(
            pipeline.GROOMING_ADVISORY_IDENTITY_REFINE_DENOISE,
            1.0,
        )
        self.assertEqual(
            pipeline.GROOMING_ADVISORY_IDENTITY_REFINE_HEAD_EXPANSION_PX,
            0,
        )

    def test_advisory_grooming_identity_refine_requires_reported_face_drift(
        self,
    ) -> None:
        self.assertTrue(
            pipeline.advisory_grooming_candidate_safe_for_identity_refine(
                {
                    "available": True,
                    "acceptable": False,
                    "issues": ["grooming_face_dark_marking_drift"],
                    "pose": {"acceptable": True},
                }
            )
        )
        self.assertFalse(
            pipeline.advisory_grooming_candidate_safe_for_identity_refine(
                {
                    "available": True,
                    "acceptable": False,
                    "issues": ["raised_paw_too_far_from_muzzle"],
                    "pose": {"acceptable": False},
                }
            )
        )

    def test_advisory_grooming_repairs_final_action_safe_dark_identity_drift(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workflow, _, _ = self._capture_action_workflow(
                Path(temporary),
                action="grooming",
                action_identity_policy=(pipeline.POSE_IDENTITY_INPUT_POLICY_CANONICAL),
                pose_attempt=6,
                advisory_grooming_minimum_acceptable=False,
                advisory_grooming_minimum_issues=["new_dominant_dark_markings"],
                action_chroma_acceptable_attempts={1, 2, 3, 4, 5},
                identity_palette_guard=True,
                uploaded_white_maquette=True,
                visual_audits_blocking=False,
            )

        self.assertIn(
            "MASKED UPPER-HEAD PASS",
            workflow["6"]["inputs"]["prompt"],
        )
        self.assertEqual(
            workflow["14"]["inputs"]["denoise"],
            pipeline.GROOMING_ADVISORY_IDENTITY_REFINE_DENOISE,
        )
        load_images = {
            node["inputs"]["image"]
            for node in workflow.values()
            if node.get("class_type") == "LoadImage"
        }
        self.assertTrue(
            any(
                image.endswith(
                    "grooming_advisory_identity_refine_source_attempt_05.png"
                )
                for image in load_images
            )
        )
        self.assertTrue(
            any(
                node.get("class_type") == "SetLatentNoiseMask"
                for node in workflow.values()
            )
        )

    def test_advisory_grooming_profile_budget_includes_identity_refine(self) -> None:
        profiles = [
            {
                "name": f"draft-{index}",
                "user_loras": [],
                "steps": 10,
                "quality_retry": index,
                "denoise": 1.0,
            }
            for index in range(1, 4)
        ]

        selected = pipeline.advisory_grooming_generation_profiles(
            "grooming",
            profiles,
            enabled=True,
        )

        self.assertEqual(
            len(selected),
            pipeline.GROOMING_ADVISORY_TOTAL_ATTEMPTS,
        )
        self.assertEqual(
            len(pipeline.GROOMING_ADVISORY_ACTION_TRANSFER_DENOISE_BY_ATTEMPT),
            pipeline.GROOMING_ADVISORY_ACTION_TRANSFER_MAX_ATTEMPTS,
        )
        self.assertLess(
            pipeline.GROOMING_ADVISORY_ACTION_TRANSFER_DENOISE_BY_ATTEMPT[0],
            1.0,
        )
        self.assertIn(
            1.0,
            pipeline.GROOMING_ADVISORY_ACTION_TRANSFER_DENOISE_BY_ATTEMPT,
        )
        self.assertEqual(selected[-1]["name"], "grooming_conditional_identity_refine")
        self.assertEqual(
            selected[-1]["denoise"],
            pipeline.GROOMING_ADVISORY_IDENTITY_REFINE_DENOISE,
        )

        health_payload = pipeline.health()
        grooming_health = health_payload["qwen_style"]["pose_guard"][
            "grooming_advisory_first_frame"
        ]
        self.assertIn("conditional_identity_refine", grooming_health)
        self.assertNotIn("mandatory_identity_refine", grooming_health)

    def test_advisory_grooming_attempt3_uses_masked_foreleg_completion(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workflow, uploaded_input, pose_input = self._capture_action_workflow(
                Path(temporary),
                action="grooming",
                action_identity_policy=(pipeline.POSE_IDENTITY_INPUT_POLICY_CANONICAL),
                pose_attempt=3,
                advisory_grooming_minimum_acceptable=False,
                action_chroma_acceptable_attempts={1, 2},
                identity_palette_guard=True,
                uploaded_white_maquette=True,
                visual_audits_blocking=False,
            )

        load_images = {
            node["inputs"]["image"]
            for node in workflow.values()
            if node.get("class_type") == "LoadImage"
        }
        expected_prefix = "local_video_api/capture-grooming/"
        self.assertEqual(
            load_images,
            {
                expected_prefix + "cartoon_reference.png",
                expected_prefix + "grooming_foreleg_repair_source.png",
                expected_prefix + "grooming_foreleg_repair_anatomy_guide.png",
                expected_prefix
                + "grooming_foreleg_repair_source_noise_mask.png",
            },
        )
        self.assertNotIn(uploaded_input, load_images)
        self.assertNotIn(pose_input, load_images)
        prompt = workflow["6"]["inputs"]["prompt"]
        self.assertIn("CRITICAL MASKED GROOMING COMPLETION", prompt)
        self.assertIn(
            "Image 3 is the frozen normalized neutral-depth action anatomy",
            prompt,
        )
        self.assertIn(
            "best identity-palette-safe normalized candidate from attempt 1",
            prompt,
        )
        self.assertTrue(
            any(
                node.get("class_type") == "SetLatentNoiseMask"
                for node in workflow.values()
            )
        )

    def test_advisory_grooming_hard_object_attempt3_restarts_from_clean_canonical(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workflow, uploaded_input, pose_input = self._capture_action_workflow(
                Path(temporary),
                action="grooming",
                action_identity_policy=(pipeline.POSE_IDENTITY_INPUT_POLICY_CANONICAL),
                pose_attempt=3,
                advisory_grooming_minimum_acceptable=False,
                advisory_grooming_minimum_issues=[
                    "raised_paw_hard_object_texture_signature"
                ],
                action_chroma_acceptable_attempts={1, 2},
                identity_palette_guard=True,
                uploaded_white_maquette=True,
                visual_audits_blocking=False,
            )

        expected_prefix = "local_video_api/capture-grooming/"
        load_images = {
            node["inputs"]["image"]
            for node in workflow.values()
            if node.get("class_type") == "LoadImage"
        }
        self.assertEqual(
            load_images,
            {
                expected_prefix + "cartoon_reference.png",
                expected_prefix
                + "grooming_white_maquette_frame000_body_pose_action_reference.png",
                expected_prefix + "grooming_canonical_face_identity_reference.png",
            },
        )
        self.assertNotIn(uploaded_input, load_images)
        self.assertNotIn(pose_input, load_images)
        self.assertFalse(
            any(
                node.get("class_type") == "SetLatentNoiseMask"
                for node in workflow.values()
            )
        )
        prompt = workflow["6"]["inputs"]["prompt"]
        self.assertIn("CRITICAL GROOMING FIRST-FRAME IDENTITY TRANSFER", prompt)
        self.assertIn("GROOMING RETRY CORRECTION", prompt)
        self.assertNotIn("Previous output recolored the cat", prompt)

    def test_advisory_grooming_minimum_palette_rejects_identity_takeover(
        self,
    ) -> None:
        source = self._identity_palette_signature_for_test(
            {
                "black": 0.02,
                "gray": 0.51,
                "white": 0.47,
                "warm_brown_or_orange": 0.0,
                "red_or_pink": 0.0,
                "yellow_or_gold": 0.0,
                "green": 0.0,
                "blue_or_cyan": 0.0,
                "purple": 0.0,
            },
            saturated_warm_fraction=0.0,
        )
        candidate = self._identity_palette_signature_for_test(
            {
                "black": 0.38,
                "gray": 0.21,
                "white": 0.39,
                "warm_brown_or_orange": 0.01,
                "red_or_pink": 0.01,
                "yellow_or_gold": 0.0,
                "green": 0.0,
                "blue_or_cyan": 0.0,
                "purple": 0.0,
            },
            saturated_warm_fraction=0.0,
        )

        quality = pipeline.advisory_grooming_minimum_palette_quality(
            source,
            candidate,
        )

        self.assertFalse(quality["acceptable"])
        self.assertIn("gross_identity_palette_l1_drift", quality["issues"])
        self.assertIn("gross_dark_marking_area_drift", quality["issues"])

    def test_advisory_grooming_minimum_palette_allows_pose_lighting_shift(
        self,
    ) -> None:
        source = self._identity_palette_signature_for_test(
            {
                "black": 0.08,
                "gray": 0.42,
                "white": 0.48,
                "warm_brown_or_orange": 0.02,
                "red_or_pink": 0.0,
                "yellow_or_gold": 0.0,
                "green": 0.0,
                "blue_or_cyan": 0.0,
                "purple": 0.0,
            },
            saturated_warm_fraction=0.0,
        )
        candidate = self._identity_palette_signature_for_test(
            {
                "black": 0.14,
                "gray": 0.49,
                "white": 0.34,
                "warm_brown_or_orange": 0.02,
                "red_or_pink": 0.01,
                "yellow_or_gold": 0.0,
                "green": 0.0,
                "blue_or_cyan": 0.0,
                "purple": 0.0,
            },
            saturated_warm_fraction=0.0,
        )

        quality = pipeline.advisory_grooming_minimum_palette_quality(
            source,
            candidate,
        )

        self.assertTrue(quality["acceptable"])
        self.assertEqual(quality["issues"], [])

    def test_advisory_grooming_face_gate_blocks_severe_stripes_and_eye_recolor(
        self,
    ) -> None:
        source = {
            "available": True,
            "fractions": {
                "black": 0.03,
                "white": 0.86,
                "gray": 0.08,
                "chromatic": 0.03,
                "yellow_or_gold": 0.0,
                "green": 0.02,
                "warm_brown_or_yellow": 0.0,
            },
        }
        severe_drift = {
            "available": True,
            "fractions": {
                "black": 0.25,
                "white": 0.60,
                "gray": 0.09,
                "chromatic": 0.06,
                "yellow_or_gold": 0.035,
                "green": 0.001,
                "warm_brown_or_yellow": 0.04,
            },
        }
        normal_lighting = {
            "available": True,
            "fractions": {
                "black": 0.08,
                "white": 0.78,
                "gray": 0.10,
                "chromatic": 0.04,
                "yellow_or_gold": 0.004,
                "green": 0.014,
                "warm_brown_or_yellow": 0.006,
            },
        }

        rejected = pipeline.advisory_grooming_face_identity_quality(
            source,
            severe_drift,
        )
        accepted = pipeline.advisory_grooming_face_identity_quality(
            source,
            normal_lighting,
        )

        self.assertFalse(rejected["acceptable"])
        self.assertIn("grooming_face_dark_marking_drift", rejected["issues"])
        self.assertIn("grooming_face_eye_hue_replacement", rejected["issues"])
        self.assertTrue(accepted["acceptable"])
        self.assertEqual(accepted["issues"], [])

    def test_advisory_grooming_minimum_palette_rejects_observed_white_to_tabby_drift(
        self,
    ) -> None:
        source = self._identity_palette_signature_for_test(
            {
                "black": 0.018,
                "gray": 0.511,
                "white": 0.470,
                "warm_brown_or_orange": 0.0,
                "red_or_pink": 0.0,
                "yellow_or_gold": 0.0,
                "green": 0.001,
                "blue_or_cyan": 0.0,
                "purple": 0.0,
            },
            saturated_warm_fraction=0.0,
        )
        candidate = self._identity_palette_signature_for_test(
            {
                "black": 0.167293,
                "gray": 0.420317,
                "white": 0.305218,
                "warm_brown_or_orange": 0.089351,
                "red_or_pink": 0.017774,
                "yellow_or_gold": 0.000407,
                "green": 0.0,
                "blue_or_cyan": 0.0,
                "purple": 0.0,
            },
            saturated_warm_fraction=0.04,
        )

        quality = pipeline.advisory_grooming_minimum_palette_quality(
            source,
            candidate,
        )

        self.assertFalse(quality["acceptable"])
        self.assertIn("gross_identity_palette_l1_drift", quality["issues"])
        self.assertIn("gross_achromatic_value_balance_drift", quality["issues"])
        self.assertIn("new_dominant_dark_markings", quality["issues"])

    def test_advisory_grooming_minimum_pose_blocks_only_action_topology(
        self,
    ) -> None:
        appearance_only = pipeline.advisory_grooming_minimum_pose_quality(
            {
                "available": True,
                "issues": [
                    "raised_paw_color_inconsistent_with_planted_paw",
                    "guide_chamfer_coverage_too_low",
                ],
            }
        )
        missing_contact = pipeline.advisory_grooming_minimum_pose_quality(
            {
                "available": True,
                "issues": [
                    "raised_paw_color_inconsistent_with_planted_paw",
                    "continuous_raised_foreleg_edge_too_short",
                    "raised_paw_too_far_from_muzzle",
                ],
            }
        )
        unavailable_contact = pipeline.advisory_grooming_minimum_pose_quality(
            {
                "available": True,
                "issues": ["paw_muzzle_gap_unavailable"],
            }
        )
        hard_object = pipeline.advisory_grooming_minimum_pose_quality(
            {
                "available": True,
                "issues": ["guide_chamfer_coverage_too_low"],
                "metrics": {
                    "appearance_consistency": {
                        "available": True,
                        "texture_gradient_p90_ratio": 3.645234,
                        "raised": {"gradient_p90": 249.285098},
                    }
                },
            }
        )
        soft_high_contrast_fur = pipeline.advisory_grooming_minimum_pose_quality(
            {
                "available": True,
                "issues": [],
                "metrics": {
                    "appearance_consistency": {
                        "available": True,
                        "texture_gradient_p90_ratio": 3.2,
                        "raised": {"gradient_p90": 160.0},
                    }
                },
            }
        )

        self.assertTrue(appearance_only["acceptable"])
        self.assertEqual(appearance_only["issues"], [])
        self.assertIn(
            "raised_paw_color_inconsistent_with_planted_paw",
            appearance_only["warnings"],
        )
        self.assertTrue(missing_contact["acceptable"])
        self.assertEqual(missing_contact["issues"], [])
        self.assertIn(
            "raised_paw_too_far_from_muzzle",
            missing_contact["warnings"],
        )
        self.assertIn(
            "continuous_raised_foreleg_edge_too_short",
            missing_contact["warnings"],
        )
        self.assertFalse(unavailable_contact["acceptable"])
        self.assertEqual(
            unavailable_contact["issues"],
            ["paw_muzzle_gap_unavailable"],
        )
        self.assertFalse(hard_object["acceptable"])
        self.assertEqual(
            hard_object["issues"],
            ["raised_paw_hard_object_texture_signature"],
        )
        self.assertTrue(
            hard_object["hard_object_texture_evidence"]["blocking"]
        )
        self.assertTrue(soft_high_contrast_fur["acceptable"])
        self.assertFalse(
            soft_high_contrast_fur["hard_object_texture_evidence"]["blocking"]
        )

    def test_production_eating_attempt5_refines_pose_safe_palette_drift(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workflow, uploaded_input, pose_input = self._capture_action_workflow(
                Path(temporary),
                action="eating",
                action_identity_policy=(pipeline.POSE_IDENTITY_INPUT_POLICY_CANONICAL),
                pose_attempt=5,
                identity_palette_guard=True,
                canonical_action_palette_drift_attempts={4},
                action_chroma_acceptable_attempts={4},
            )

        load_images = {
            node_id: node["inputs"]["image"]
            for node_id, node in workflow.items()
            if node.get("class_type") == "LoadImage"
        }
        expected_canonical = "local_video_api/capture-eating/cartoon_reference.png"
        refine_input = (
            "local_video_api/capture-eating/eating_identity_refine_source.png"
        )
        self.assertEqual(
            set(load_images.values()),
            {refine_input, expected_canonical, uploaded_input},
        )
        self.assertEqual(load_images["1"], refine_input)
        self.assertNotIn(pose_input, load_images.values())
        prompt = workflow["6"]["inputs"]["prompt"]
        self.assertIn("CRITICAL EATING IDENTITY REFINE", prompt)
        self.assertIn(
            "normalized eating candidate from attempt 4",
            prompt,
        )
        self.assertIn(
            "Image 2 is the approved Qwen full-body canonical",
            prompt,
        )
        self.assertIn(
            "Image 3 is the original uploaded front photograph of this same pet",
            prompt,
        )
        self.assertIn(
            "lower muzzle and mouth must remain physically inside the kibble",
            prompt,
        )
        self.assertIn(
            "retain their light-beige and brown colors without transferring those warm colors into the cat",
            prompt,
        )
        self.assertEqual(
            workflow["14"]["inputs"]["denoise"],
            pipeline.EATING_CANONICAL_IDENTITY_REFINE_DENOISE,
        )
        self.assertEqual(
            workflow["14"]["inputs"]["steps"],
            pipeline.EATING_CANONICAL_IDENTITY_REFINE_STEPS,
        )
        self.assertFalse(
            any(
                node.get("class_type") == "SetLatentNoiseMask"
                for node in workflow.values()
            )
        )
        loaded_loras = {
            node["inputs"]["lora_name"]: node["inputs"]["strength_model"]
            for node in workflow.values()
            if node.get("class_type") == "LoraLoaderModelOnly"
        }
        self.assertEqual(loaded_loras, {pipeline.QWEN_LORA: 1.0})
        self.assertNotIn(self.TEST_QWEN_STYLE_LORA, loaded_loras)

    def test_eating_action_workflow_uses_main_case_geometry_and_uploaded_identity(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workflow, uploaded_input, pose_input = self._capture_action_workflow(
                Path(temporary),
                action="eating",
                action_identity_policy=(
                    pipeline.POSE_IDENTITY_INPUT_POLICY_PROFILE_GEOMETRY_FIRST
                ),
                main_case_raw_identity_chain=True,
            )

        load_images = {
            node_id: node["inputs"]["image"]
            for node_id, node in workflow.items()
            if node.get("class_type") == "LoadImage"
        }
        self.assertEqual(set(load_images.values()), {uploaded_input, pose_input})
        self.assertEqual(load_images["1"], pose_input)
        identity_load_node = next(
            node_id for node_id, image in load_images.items() if image == uploaded_input
        )
        identity_scale_node = next(
            node_id
            for node_id, node in workflow.items()
            if node.get("class_type") == "FluxKontextImageScale"
            and node["inputs"].get("image") == [identity_load_node, 0]
        )
        self.assertEqual(
            workflow["6"]["inputs"]["image1"],
            workflow["7"]["inputs"]["image1"],
        )
        self.assertEqual(workflow["6"]["inputs"]["image1"], ["5", 0])
        self.assertEqual(
            workflow["6"]["inputs"]["image2"],
            workflow["7"]["inputs"]["image2"],
        )
        self.assertEqual(
            workflow["6"]["inputs"]["image2"],
            [identity_scale_node, 0],
        )
        self.assertEqual(workflow["10"]["inputs"]["pixels"], ["5", 0])
        self.assertEqual(workflow["14"]["inputs"]["latent_image"], ["10", 0])
        self.assertIn(
            "Image 1 is the texture-free low-frequency geometry guide derived directly from "
            "the frozen Main_case eat.mp4 standard",
            workflow["6"]["inputs"]["prompt"],
        )
        self.assertIn(
            "Image 2 is the original uploaded pet photo",
            workflow["6"]["inputs"]["prompt"],
        )
        loaded_loras = {
            node["inputs"]["lora_name"]: node["inputs"]["strength_model"]
            for node in workflow.values()
            if node.get("class_type") == "LoraLoaderModelOnly"
        }
        self.assertEqual(
            loaded_loras,
            {
                pipeline.QWEN_LORA: 1.0,
            },
        )
        photography_lora_names = {
            item["model_name"]
            for item in pipeline.IMAGE_PROFILES[
                pipeline.SUBJECT_REPLACEMENT_REFERENCE_IMAGE_PROFILE
            ]["user_loras"]
        }
        self.assertTrue(photography_lora_names.isdisjoint(loaded_loras))
        self.assertEqual(workflow["14"]["inputs"]["model"], ["13", 0])
        self.assertIn(uploaded_input, load_images.values())

    def test_eating_retry_loras_are_lightning_then_pet3d_only(self) -> None:
        photography_lora_names = {
            item["model_name"]
            for item in pipeline.IMAGE_PROFILES[
                pipeline.SUBJECT_REPLACEMENT_REFERENCE_IMAGE_PROFILE
            ]["user_loras"]
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for attempt in (1, 2, 3, 4, 5):
                with self.subTest(attempt=attempt):
                    workflow, _uploaded_input, _pose_input = (
                        self._capture_action_workflow(
                            root / f"attempt-{attempt}",
                            action="eating",
                            action_identity_policy=(
                                pipeline.POSE_IDENTITY_INPUT_POLICY_PROFILE_GEOMETRY_FIRST
                            ),
                            pose_attempt=attempt,
                            main_case_raw_identity_chain=True,
                        )
                    )
                    loaded_loras = {
                        node["inputs"]["lora_name"]: node["inputs"]["strength_model"]
                        for node in workflow.values()
                        if node.get("class_type") == "LoraLoaderModelOnly"
                    }
                    expected_loras = {pipeline.QWEN_LORA: 1.0}
                    if attempt >= 3:
                        expected_loras[self.TEST_QWEN_STYLE_LORA] = 0.15
                    self.assertEqual(loaded_loras, expected_loras)
                    self.assertTrue(photography_lora_names.isdisjoint(loaded_loras))
                    expected_model_node = "17" if attempt >= 3 else "13"
                    self.assertEqual(
                        workflow["14"]["inputs"]["model"],
                        [expected_model_node, 0],
                    )
                    if attempt >= 3:
                        self.assertEqual(
                            workflow["17"]["inputs"]["model"],
                            ["13", 0],
                        )

    def test_profile_geometry_first_eating_requires_pet3d_style_lora(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(
                RuntimeError,
                "requires the dedicated QWEN_STYLE_LORA",
            ):
                self._capture_action_workflow(
                    Path(temporary),
                    action="eating",
                    action_identity_policy=(
                        pipeline.POSE_IDENTITY_INPUT_POLICY_PROFILE_GEOMETRY_FIRST
                    ),
                    style_lora="",
                )

    def test_profile_geometry_first_eating_rejects_non_pet3d_style_lora(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(
                RuntimeError,
                "requires a dedicated pet3d QWEN_STYLE_LORA",
            ):
                self._capture_action_workflow(
                    Path(temporary),
                    action="eating",
                    action_identity_policy=(
                        pipeline.POSE_IDENTITY_INPUT_POLICY_PROFILE_GEOMETRY_FIRST
                    ),
                    style_lora="generic-style-test.safetensors",
                )

    def test_profile_geometry_first_eating_requires_positive_style_strength(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(
                RuntimeError,
                "requires a positive pet3d recovery strength",
            ):
                self._capture_action_workflow(
                    Path(temporary),
                    action="eating",
                    action_identity_policy=(
                        pipeline.POSE_IDENTITY_INPUT_POLICY_PROFILE_GEOMETRY_FIRST
                    ),
                    style_strength=0.0,
                )

    def test_profile_geometry_first_fails_closed_without_raw_identity_chain(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(
                RuntimeError,
                "requires the original uploaded identity chain",
            ):
                self._capture_action_workflow(
                    Path(temporary),
                    action="eating",
                    action_identity_policy=(
                        pipeline.POSE_IDENTITY_INPUT_POLICY_PROFILE_GEOMETRY_FIRST
                    ),
                    main_case_raw_identity_chain=False,
                )

    def test_profile_geometry_first_fails_closed_without_identity_digest(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(
                RuntimeError,
                "input lacks its submission-time digest",
            ):
                self._capture_action_workflow(
                    Path(temporary),
                    action="grooming",
                    action_identity_policy=(
                        pipeline.POSE_IDENTITY_INPUT_POLICY_PROFILE_GEOMETRY_FIRST
                    ),
                    persist_identity_digest=False,
                )

    def test_profile_geometry_first_fails_closed_without_source_digest(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(
                RuntimeError,
                "source lacks its submission-time digest",
            ):
                self._capture_action_workflow(
                    Path(temporary),
                    action="eating",
                    action_identity_policy=(
                        pipeline.POSE_IDENTITY_INPUT_POLICY_PROFILE_GEOMETRY_FIRST
                    ),
                    persist_identity_source_digest=False,
                )

    def test_profile_geometry_first_grooming_uses_main_case_and_upload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workflow, uploaded_input, pose_input = self._capture_action_workflow(
                Path(temporary),
                action="grooming",
                action_identity_policy=(
                    pipeline.POSE_IDENTITY_INPUT_POLICY_PROFILE_GEOMETRY_FIRST
                ),
                main_case_raw_identity_chain=True,
            )

        load_images = {
            node_id: node["inputs"]["image"]
            for node_id, node in workflow.items()
            if node.get("class_type") == "LoadImage"
        }
        self.assertEqual(set(load_images.values()), {pose_input, uploaded_input})
        self.assertEqual(load_images["1"], pose_input)
        self.assertEqual(workflow["14"]["inputs"]["steps"], 8)
        self.assertIn(
            "frozen Main_case TM.mp4 standard", workflow["6"]["inputs"]["prompt"]
        )
        self.assertIn(
            "never as a sleeve, strap, zipper, cord, pen, stick, pendant",
            workflow["6"]["inputs"]["prompt"],
        )

    def test_profile_geometry_first_grooming_never_uses_pet3d_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workflow, _uploaded_input, _pose_input = self._capture_action_workflow(
                Path(temporary),
                action="grooming",
                action_identity_policy=(
                    pipeline.POSE_IDENTITY_INPUT_POLICY_PROFILE_GEOMETRY_FIRST
                ),
                pose_attempt=3,
            )

        loaded_loras = {
            node["inputs"]["lora_name"]
            for node in workflow.values()
            if node.get("class_type") == "LoraLoaderModelOnly"
        }
        self.assertEqual(loaded_loras, {pipeline.QWEN_LORA})
        self.assertEqual(workflow["14"]["inputs"]["steps"], 8)

    def test_grooming_action_uses_geometry_image1_and_canonical_image2(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workflow, uploaded_input, pose_input = self._capture_action_workflow(
                Path(temporary),
                action="grooming",
                action_identity_policy=(
                    pipeline.POSE_IDENTITY_INPUT_POLICY_GROOMING_GEOMETRY_CANONICAL
                ),
            )

        load_images = {
            node_id: node["inputs"]["image"]
            for node_id, node in workflow.items()
            if node.get("class_type") == "LoadImage"
        }
        expected_canonical = "local_video_api/capture-grooming/cartoon_reference.png"
        self.assertEqual(set(load_images.values()), {expected_canonical, pose_input})
        self.assertEqual(load_images["1"], pose_input)
        self.assertNotIn(uploaded_input, load_images.values())
        canonical_load_node = next(
            node_id
            for node_id, image in load_images.items()
            if image == expected_canonical
        )
        canonical_scale_node = next(
            node_id
            for node_id, node in workflow.items()
            if node.get("class_type") == "FluxKontextImageScale"
            and node["inputs"].get("image") == [canonical_load_node, 0]
        )
        self.assertEqual(
            workflow["6"]["inputs"]["image1"],
            workflow["7"]["inputs"]["image1"],
        )
        self.assertEqual(workflow["6"]["inputs"]["image1"], ["5", 0])
        self.assertEqual(
            workflow["6"]["inputs"]["image2"],
            workflow["7"]["inputs"]["image2"],
        )
        self.assertEqual(
            workflow["6"]["inputs"]["image2"],
            [canonical_scale_node, 0],
        )
        self.assertEqual(workflow["10"]["inputs"]["pixels"], ["5", 0])
        self.assertEqual(workflow["14"]["inputs"]["latent_image"], ["10", 0])
        self.assertIn(
            "CRITICAL GROOMING TRANSFER CONTRACT",
            workflow["6"]["inputs"]["prompt"],
        )
        self.assertIn(
            "Image 1 is the appearance-free grooming geometry canvas",
            workflow["6"]["inputs"]["prompt"],
        )
        self.assertIn(
            "Image 2 is the approved Qwen full-body canonical",
            workflow["6"]["inputs"]["prompt"],
        )
        self.assertIn(
            "RGB(176,176,176) only for one continuous raised-foreleg capsule",
            workflow["6"]["inputs"]["prompt"],
        )
        self.assertIn(
            "zero identity, fur-color, marking, material, texture, shadow",
            workflow["6"]["inputs"]["prompt"],
        )
        self.assertIn(
            "never a sleeve, strap, zipper, cord, pen, stick, pendant",
            workflow["6"]["inputs"]["prompt"],
        )
        self.assertIn(
            "complete coat palette",
            workflow["6"]["inputs"]["prompt"],
        )
        self.assertNotIn("green eyes", workflow["6"]["inputs"]["prompt"])
        self.assertNotIn(
            "white/neutral coat palette",
            workflow["6"]["inputs"]["prompt"],
        )
        self.assertIn(
            "Any hat, hood, shirt, coat, sweater, collar, costume, accessory",
            workflow["6"]["inputs"]["prompt"],
        )
        self.assertIn(
            "neutral seated pose, both front paws planted, missing raised foreleg",
            workflow["7"]["inputs"]["prompt"],
        )
        self.assertIn(
            "strap, zipper, cord, pen, pencil, stick, pendant",
            workflow["7"]["inputs"]["prompt"],
        )
        self.assertNotIn(
            "exactly four complete paws",
            workflow["6"]["inputs"]["prompt"].lower(),
        )
        self.assertNotIn(
            "all four paws must be fully visible",
            workflow["6"]["inputs"]["prompt"].lower(),
        )
        loaded_loras = {
            node["inputs"]["lora_name"]: node["inputs"]["strength_model"]
            for node in workflow.values()
            if node.get("class_type") == "LoraLoaderModelOnly"
        }
        self.assertEqual(loaded_loras, {pipeline.QWEN_LORA: 1.0})
        photography_lora_names = {
            item["model_name"]
            for item in pipeline.IMAGE_PROFILES[
                pipeline.SUBJECT_REPLACEMENT_REFERENCE_IMAGE_PROFILE
            ]["user_loras"]
        }
        self.assertTrue(photography_lora_names.isdisjoint(loaded_loras))
        self.assertNotIn(self.TEST_QWEN_STYLE_LORA, loaded_loras)
        self.assertEqual(workflow["14"]["inputs"]["model"], ["13", 0])
        self.assertEqual(workflow["14"]["inputs"]["steps"], 8)
        self.assertEqual(
            workflow["14"]["inputs"]["denoise"],
            pipeline.QWEN_GROOMING_CANONICAL_GEOMETRY_DENOISE_BY_ATTEMPT[0],
        )

    def test_canonical_grooming_requires_pet3d_repair_surface_lora(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(
                RuntimeError,
                "requires the dedicated QWEN_STYLE_LORA surface model",
            ):
                self._capture_action_workflow(
                    Path(temporary),
                    action="grooming",
                    action_identity_policy=(
                        pipeline.POSE_IDENTITY_INPUT_POLICY_GROOMING_GEOMETRY_CANONICAL
                    ),
                    style_lora="",
                )

    def test_canonical_grooming_rejects_non_pet3d_repair_surface_lora(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(
                RuntimeError,
                "requires a dedicated pet3d QWEN_STYLE_LORA surface model",
            ):
                self._capture_action_workflow(
                    Path(temporary),
                    action="grooming",
                    action_identity_policy=(
                        pipeline.POSE_IDENTITY_INPUT_POLICY_GROOMING_GEOMETRY_CANONICAL
                    ),
                    style_lora="generic-style-test.safetensors",
                )

    def test_grooming_third_attempt_repairs_best_identity_safe_candidate(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workflow, uploaded_input, pose_input = self._capture_action_workflow(
                Path(temporary),
                action="grooming",
                action_identity_policy=(
                    pipeline.POSE_IDENTITY_INPUT_POLICY_GROOMING_GEOMETRY_CANONICAL
                ),
                pose_attempt=3,
                grooming_gate_acceptable=False,
                identity_palette_guard=True,
            )

        load_images = {
            node_id: node["inputs"]["image"]
            for node_id, node in workflow.items()
            if node.get("class_type") == "LoadImage"
        }
        expected_canonical = "local_video_api/capture-grooming/cartoon_reference.png"
        repair_input = (
            "local_video_api/capture-grooming/grooming_foreleg_repair_source.png"
        )
        anatomy_input = (
            "local_video_api/capture-grooming/grooming_foreleg_repair_anatomy_guide.png"
        )
        noise_mask_input = (
            "local_video_api/capture-grooming/"
            "grooming_foreleg_repair_source_noise_mask.png"
        )
        self.assertEqual(
            set(load_images.values()),
            {
                repair_input,
                expected_canonical,
                anatomy_input,
                noise_mask_input,
            },
        )
        self.assertEqual(load_images["1"], repair_input)
        self.assertNotIn(pose_input, load_images.values())
        self.assertNotIn(uploaded_input, load_images.values())
        self.assertIn(
            "CRITICAL GROOMING FORELEG COMPLETION",
            workflow["6"]["inputs"]["prompt"],
        )
        self.assertIn(
            "best identity-safe near-final candidate from attempt 2",
            workflow["6"]["inputs"]["prompt"],
        )
        self.assertIn(
            "deliberately band-limited identity-matched coat-color, mid-scale fur, and shading seed",
            workflow["6"]["inputs"]["prompt"],
        )
        self.assertIn(
            "Image 3 is a frozen neutral low-frequency anatomy map",
            workflow["6"]["inputs"]["prompt"],
        )
        self.assertIn(
            "a generously expanded feathered full-denoise mask covers the seed plus ordinary surrounding muzzle, shoulder, and chest context",
            workflow["6"]["inputs"]["prompt"],
        )
        self.assertIn(
            "none of its boundary or individual sampled strand pixels may be copied literally",
            workflow["6"]["inputs"]["prompt"],
        )
        self.assertIn(
            "loaded low-strength Pet3D adapter is permitted only as a fine surface prior inside the sampler noise mask",
            workflow["6"]["inputs"]["prompt"],
        )
        self.assertNotIn("flashlight", workflow["6"]["inputs"]["prompt"])
        self.assertNotIn("cylinder", workflow["6"]["inputs"]["prompt"])
        self.assertEqual(
            workflow["14"]["inputs"]["denoise"],
            pipeline.GROOMING_QWEN_FORELEG_REPAIR_DENOISE,
        )
        self.assertEqual(
            workflow["14"]["inputs"]["steps"],
            pipeline.GROOMING_QWEN_FORELEG_REPAIR_STEPS,
        )
        mask_load_node = next(
            node_id
            for node_id, image_name in load_images.items()
            if image_name == noise_mask_input
        )
        masked_latent_nodes = {
            node_id: node
            for node_id, node in workflow.items()
            if node.get("class_type") == "SetLatentNoiseMask"
        }
        self.assertEqual(len(masked_latent_nodes), 1)
        masked_latent_id, masked_latent = next(iter(masked_latent_nodes.items()))
        self.assertEqual(masked_latent["inputs"]["samples"], ["10", 0])
        self.assertEqual(
            masked_latent["inputs"]["mask"],
            [mask_load_node, 1],
        )
        self.assertEqual(
            workflow["14"]["inputs"]["latent_image"],
            [masked_latent_id, 0],
        )
        loaded_loras = {
            node["inputs"]["lora_name"]: node["inputs"]["strength_model"]
            for node in workflow.values()
            if node.get("class_type") == "LoraLoaderModelOnly"
        }
        self.assertEqual(
            loaded_loras,
            {
                pipeline.QWEN_LORA: 1.0,
                self.TEST_QWEN_STYLE_LORA: (
                    pipeline.GROOMING_QWEN_FORELEG_REPAIR_STYLE_STRENGTH
                ),
            },
        )
        self.assertEqual(workflow["17"]["inputs"]["model"], ["13", 0])
        self.assertEqual(workflow["14"]["inputs"]["model"], ["17", 0])

    def test_grooming_repair_fails_closed_without_identity_palette_verdict(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workflow, uploaded_input, pose_input = self._capture_action_workflow(
                Path(temporary),
                action="grooming",
                action_identity_policy=(
                    pipeline.POSE_IDENTITY_INPUT_POLICY_GROOMING_GEOMETRY_CANONICAL
                ),
                pose_attempt=3,
                grooming_gate_acceptable=False,
                identity_palette_guard=False,
            )

        load_images = {
            node_id: node["inputs"]["image"]
            for node_id, node in workflow.items()
            if node.get("class_type") == "LoadImage"
        }
        self.assertEqual(load_images["1"], pose_input)
        self.assertFalse(
            any(
                str(value).endswith("grooming_foreleg_repair_source.png")
                for value in load_images.values()
            )
        )
        self.assertNotIn(
            "CRITICAL GROOMING FORELEG COMPLETION",
            workflow["6"]["inputs"]["prompt"],
        )
        self.assertNotIn(uploaded_input, load_images.values())
        self.assertFalse(
            any(
                node.get("class_type") == "SetLatentNoiseMask"
                for node in workflow.values()
            )
        )

    def test_production_grooming_attempt3_mask_completes_identity_safe_base(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workflow, uploaded_input, pose_input = self._capture_action_workflow(
                Path(temporary),
                action="grooming",
                action_identity_policy=(pipeline.POSE_IDENTITY_INPUT_POLICY_CANONICAL),
                pose_attempt=3,
                grooming_gate_acceptable=False,
                identity_palette_guard=True,
            )

        load_images = {
            node_id: node["inputs"]["image"]
            for node_id, node in workflow.items()
            if node.get("class_type") == "LoadImage"
        }
        expected_canonical = "local_video_api/capture-grooming/cartoon_reference.png"
        repair_input = (
            "local_video_api/capture-grooming/grooming_foreleg_repair_source.png"
        )
        anatomy_input = (
            "local_video_api/capture-grooming/grooming_foreleg_repair_anatomy_guide.png"
        )
        noise_mask_input = (
            "local_video_api/capture-grooming/"
            "grooming_foreleg_repair_source_noise_mask.png"
        )
        completion_mask = (
            "local_video_api/capture-grooming/"
            "grooming_qwen_pose_guide_completion_mask.png"
        )
        self.assertEqual(
            set(load_images.values()),
            {
                repair_input,
                expected_canonical,
                anatomy_input,
                noise_mask_input,
            },
        )
        self.assertEqual(load_images["1"], repair_input)
        self.assertNotIn(pose_input, load_images.values())
        self.assertNotIn(completion_mask, load_images.values())
        self.assertNotIn(uploaded_input, load_images.values())
        prompt = workflow["6"]["inputs"]["prompt"]
        self.assertIn("CRITICAL MASKED GROOMING COMPLETION", prompt)
        self.assertIn(
            "best identity-palette-safe normalized candidate from attempt 2",
            prompt,
        )
        self.assertIn(
            "Image 2 is the approved Qwen full-body canonical",
            prompt,
        )
        self.assertIn(
            "Image 3 is the frozen appearance-free single-gray action silhouette",
            prompt,
        )
        self.assertIn("viewer-left shoulder-to-elbow-to-wrist-to-paw", prompt)
        self.assertIn("Image 2 remains the sole appearance authority", prompt)
        self.assertEqual(
            workflow["14"]["inputs"]["denoise"],
            pipeline.GROOMING_QWEN_FORELEG_REPAIR_DENOISE,
        )
        self.assertEqual(
            workflow["14"]["inputs"]["steps"],
            pipeline.GROOMING_QWEN_FORELEG_REPAIR_STEPS,
        )
        masked_latent_nodes = {
            node_id: node
            for node_id, node in workflow.items()
            if node.get("class_type") == "SetLatentNoiseMask"
        }
        self.assertEqual(len(masked_latent_nodes), 1)
        masked_latent_id = next(iter(masked_latent_nodes))
        self.assertEqual(
            workflow["14"]["inputs"]["latent_image"],
            [masked_latent_id, 0],
        )
        loaded_loras = {
            node["inputs"]["lora_name"]: node["inputs"]["strength_model"]
            for node in workflow.values()
            if node.get("class_type") == "LoraLoaderModelOnly"
        }
        self.assertEqual(loaded_loras, {pipeline.QWEN_LORA: 1.0})
        self.assertNotIn(self.TEST_QWEN_STYLE_LORA, loaded_loras)
        self.assertEqual(workflow["14"]["inputs"]["model"], ["13", 0])

    def test_production_grooming_attempt4_uses_neutralized_action_pose_seed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workflow, uploaded_input, pose_input = self._capture_action_workflow(
                Path(temporary),
                action="grooming",
                action_identity_policy=(pipeline.POSE_IDENTITY_INPUT_POLICY_CANONICAL),
                pose_attempt=4,
                grooming_gate_acceptable=False,
                identity_palette_guard=True,
            )

        load_images = {
            node_id: node["inputs"]["image"]
            for node_id, node in workflow.items()
            if node.get("class_type") == "LoadImage"
        }
        expected_canonical = "local_video_api/capture-grooming/cartoon_reference.png"
        pose_transfer_seed = (
            "local_video_api/capture-grooming/"
            "grooming_canonical_pose_transfer_seed.png"
        )
        self.assertEqual(
            set(load_images.values()),
            {pose_transfer_seed, expected_canonical},
        )
        self.assertEqual(load_images["1"], pose_transfer_seed)
        self.assertNotIn(pose_input, load_images.values())
        self.assertNotIn(uploaded_input, load_images.values())
        prompt = workflow["6"]["inputs"]["prompt"]
        self.assertIn("CRITICAL ACTION-POSE IDENTITY TRANSFER", prompt)
        self.assertIn(
            "Image 1 is the sampler canvas and a deterministic high-key anatomy maquette",
            prompt,
        )
        self.assertIn(
            "Image 2 is the approved Qwen full-body canonical",
            prompt,
        )
        self.assertIn(
            "There is deliberately no third reference image",
            prompt,
        )
        self.assertIn("already consolidates the uploaded front, left, and right", prompt)
        self.assertIn("viewer-left shoulder-to-elbow-to-wrist-to-paw", prompt)
        self.assertIn("Do not replace it with a hanging item", prompt)
        self.assertEqual(
            workflow["14"]["inputs"]["denoise"],
            pipeline.GROOMING_CANONICAL_FULL_POSE_RESCUE_DENOISE,
        )
        self.assertEqual(
            workflow["14"]["inputs"]["steps"],
            pipeline.GROOMING_CANONICAL_FULL_POSE_RESCUE_STEPS,
        )
        self.assertFalse(
            any(
                node.get("class_type") == "SetLatentNoiseMask"
                for node in workflow.values()
            )
        )
        loaded_loras = {
            node["inputs"]["lora_name"]: node["inputs"]["strength_model"]
            for node in workflow.values()
            if node.get("class_type") == "LoraLoaderModelOnly"
        }
        self.assertEqual(loaded_loras, {pipeline.QWEN_LORA: 1.0})
        self.assertNotIn(self.TEST_QWEN_STYLE_LORA, loaded_loras)

    def test_production_grooming_attempt4_never_uses_rejected_pose_candidate_as_sampler(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workflow, uploaded_input, pose_input = self._capture_action_workflow(
                Path(temporary),
                action="grooming",
                action_identity_policy=(pipeline.POSE_IDENTITY_INPUT_POLICY_CANONICAL),
                pose_attempt=4,
                grooming_gate_acceptable=True,
                identity_palette_guard=True,
                canonical_action_palette_drift_attempts={1, 2, 3},
            )

        load_images = {
            node_id: node["inputs"]["image"]
            for node_id, node in workflow.items()
            if node.get("class_type") == "LoadImage"
        }
        expected_canonical = "local_video_api/capture-grooming/cartoon_reference.png"
        pose_transfer_seed = (
            "local_video_api/capture-grooming/"
            "grooming_canonical_pose_transfer_seed.png"
        )
        self.assertEqual(
            set(load_images.values()),
            {pose_transfer_seed, expected_canonical},
        )
        self.assertEqual(load_images["1"], pose_transfer_seed)
        self.assertNotIn(pose_input, load_images.values())
        self.assertNotIn(uploaded_input, load_images.values())
        self.assertFalse(
            any(
                str(value).endswith("grooming_full_pose_identity_refine_source.png")
                for value in load_images.values()
            )
        )
        prompt = workflow["6"]["inputs"]["prompt"]
        self.assertIn("CRITICAL ACTION-POSE IDENTITY TRANSFER", prompt)
        self.assertIn(
            "original RGB, chroma, raw luma, low-frequency markings, face detail, lighting, and fur texture were removed",
            prompt,
        )
        self.assertIn(
            "Image 2 is the approved Qwen full-body canonical",
            prompt,
        )
        self.assertIn(
            "There is deliberately no third reference image",
            prompt,
        )
        self.assertEqual(
            workflow["14"]["inputs"]["denoise"],
            pipeline.GROOMING_CANONICAL_FULL_POSE_REFINE_DENOISE,
        )
        self.assertFalse(
            any(
                node.get("class_type") == "SetLatentNoiseMask"
                for node in workflow.values()
            )
        )

    def test_production_grooming_attempt5_reuses_digest_bound_action_pose_seed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workflow, uploaded_input, pose_input = self._capture_action_workflow(
                Path(temporary),
                action="grooming",
                action_identity_policy=(pipeline.POSE_IDENTITY_INPUT_POLICY_CANONICAL),
                pose_attempt=5,
                grooming_gate_acceptable=True,
                identity_palette_guard=True,
                canonical_action_palette_drift_attempts=set(range(1, 9)),
                pose_safe_grooming_attempt4=True,
            )

        load_images = {
            node_id: node["inputs"]["image"]
            for node_id, node in workflow.items()
            if node.get("class_type") == "LoadImage"
        }
        expected_canonical = "local_video_api/capture-grooming/cartoon_reference.png"
        pose_transfer_seed = (
            "local_video_api/capture-grooming/"
            "grooming_canonical_pose_transfer_seed.png"
        )
        refine_source = (
            "local_video_api/capture-grooming/"
            "grooming_full_pose_identity_refine_source.png"
        )
        self.assertEqual(
            set(load_images.values()),
            {refine_source, expected_canonical, pose_transfer_seed},
        )
        self.assertEqual(load_images["1"], refine_source)
        self.assertNotIn(pose_input, load_images.values())
        self.assertNotIn(uploaded_input, load_images.values())
        prompt = workflow["6"]["inputs"]["prompt"]
        self.assertIn("CRITICAL POSE-SAFE IDENTITY REFINE", prompt)
        self.assertIn(
            "normalized grooming candidate from attempt 4",
            prompt,
        )
        self.assertIn(
            "Image 2 is the approved Qwen full-body canonical",
            prompt,
        )
        self.assertIn(
            "Image 3 is the deterministic high-key action-anatomy seed",
            prompt,
        )
        self.assertIn("passed the independent complete-subject check", prompt)
        self.assertIn("Do not return to a neutral two-paws-down sit", prompt)
        self.assertEqual(
            workflow["14"]["inputs"]["denoise"],
            pipeline.GROOMING_CANONICAL_FULL_POSE_REFINE_DENOISE,
        )
        self.assertEqual(
            workflow["14"]["inputs"]["steps"],
            pipeline.GROOMING_CANONICAL_FULL_POSE_RESCUE_STEPS,
        )
        self.assertFalse(
            any(
                node.get("class_type") == "SetLatentNoiseMask"
                for node in workflow.values()
            )
        )
        loaded_loras = {
            node["inputs"]["lora_name"]: node["inputs"]["strength_model"]
            for node in workflow.values()
            if node.get("class_type") == "LoraLoaderModelOnly"
        }
        self.assertEqual(loaded_loras, {pipeline.QWEN_LORA: 1.0})
        self.assertNotIn(self.TEST_QWEN_STYLE_LORA, loaded_loras)

    def test_grooming_repair_inputs_seed_canonical_foreleg_and_bind_anatomy(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "candidate.png"
            geometry = root / "geometry.png"
            canonical = root / "canonical.png"
            anatomy = root / "anatomy.png"
            repair = root / "repair.png"
            repair_noise_mask = root / "repair_noise_mask.png"
            frozen_anatomy = root / "frozen-anatomy.png"

            candidate_image = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
            candidate_image.paste((90, 100, 110, 255), (112, 72, 400, 440))
            candidate_image.save(candidate)
            canonical_image = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
            canonical_image.paste(
                (210, 220, 230, 255),
                (120, 72, 392, 440),
            )
            canonical_image.save(canonical)
            geometry_image = Image.new("RGB", (512, 512), (0, 255, 0))
            geometry_image.paste(
                pipeline.GROOMING_QWEN_GUIDE_FOREGROUND_RGB,
                (112, 72, 400, 440),
            )
            geometry_image.paste(
                pipeline.GROOMING_QWEN_GUIDE_FORELEG_RGB,
                (180, 220, 210, 320),
            )
            geometry_image.save(geometry)
            anatomy_image = Image.new("RGB", (512, 512), (0, 255, 0))
            anatomy_image.paste(
                pipeline.GROOMING_QWEN_GUIDE_FOREGROUND_RGB,
                (112, 72, 400, 440),
            )
            ImageDraw.Draw(anatomy_image).line(
                [(208, 245), (218, 275), (216, 310), (204, 335)],
                fill=(0, 255, 0),
                width=9,
            )
            anatomy_image.save(anatomy)

            repair_audit = pipeline.prepare_grooming_foreleg_repair_canvas(
                candidate,
                geometry,
                canonical,
                repair,
                noise_mask_target=repair_noise_mask,
            )
            anatomy_audit = pipeline.prepare_grooming_foreleg_repair_anatomy_guide(
                anatomy,
                frozen_anatomy,
            )

            with Image.open(repair) as loaded:
                repaired = loaded.convert("RGB")
            self.assertEqual(repaired.getpixel((195, 270)), (210, 220, 230))
            self.assertEqual(repaired.getpixel((130, 150)), (90, 100, 110))
            self.assertEqual(repair_audit["edit_window_pure_green_pixels"], 0)
            self.assertTrue(repair_audit["edit_window_identity_texture_seeded"])
            self.assertEqual(
                repair_audit["canonical_reference_sha256"],
                pipeline.file_sha256(canonical),
            )
            self.assertGreater(
                repair_audit["canonical_seed_direct_pixels"],
                256,
            )
            self.assertTrue(repair_audit["outside_edit_window_byte_preserved"])
            self.assertFalse(
                repair_audit["canonical_seed_high_frequency_texture_exposed"]
            )
            self.assertEqual(
                repair_audit["canonical_seed_lowpass_sigma_px"],
                pipeline.GROOMING_QWEN_FORELEG_REPAIR_SEED_LOWPASS_SIGMA_PX,
            )
            self.assertEqual(
                repair_audit["canonical_seed_identity_median_blend"],
                pipeline.GROOMING_QWEN_FORELEG_REPAIR_SEED_IDENTITY_MEDIAN_BLEND,
            )
            self.assertEqual(
                repair_audit["noise_mask_target_sha256"],
                pipeline.file_sha256(repair_noise_mask),
            )
            self.assertEqual(
                repair_audit["noise_mask_loadimage_contract"],
                "mask_equals_one_minus_alpha",
            )
            self.assertGreater(
                repair_audit["noise_mask_full_strength_pixels"],
                repair_audit["semantic_foreleg_pixels"],
            )
            self.assertEqual(
                repair_audit["noise_mask_core_expansion_px"],
                pipeline.GROOMING_QWEN_FORELEG_REPAIR_NOISE_MASK_EXPANSION_PX,
            )
            self.assertGreater(
                repair_audit["noise_mask_core_pixels"],
                repair_audit["edit_mask_pixels"],
            )
            self.assertGreater(repair_audit["noise_mask_feather_pixels"], 0)
            with Image.open(repair_noise_mask) as loaded:
                repair_mask_rgba = loaded.convert("RGBA")
            self.assertEqual(repair_mask_rgba.getpixel((195, 270))[3], 0)
            self.assertEqual(repair_mask_rgba.getpixel((20, 20))[3], 255)
            self.assertGreater(repair_audit["removed_foreground_pixels"], 256)
            self.assertRegex(repair_audit["edit_mask_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(
                anatomy_audit["policy"],
                pipeline.GROOMING_QWEN_FORELEG_REPAIR_ANATOMY_POLICY,
            )
            self.assertEqual(
                anatomy_audit["source_sha256"],
                anatomy_audit["target_sha256"],
            )
            self.assertEqual(
                anatomy_audit["role"],
                "image3_single_tone_open_channel_action_geometry_only",
            )
            self.assertGreaterEqual(
                anatomy_audit["internal_open_channel_pixels"],
                300,
            )

    def test_canonical_grooming_completion_canvas_preserves_source_pixels(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "candidate.png"
            completion_mask = root / "completion-mask.png"
            canonical = root / "canonical.png"
            repair = root / "repair.png"
            noise_mask = root / "repair-noise-mask.png"

            candidate_image = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
            candidate_image.paste((90, 100, 110, 255), (112, 72, 400, 440))
            candidate_image.save(candidate)
            canonical_image = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
            canonical_image.paste((210, 220, 230, 255), (120, 72, 392, 440))
            canonical_image.save(canonical)
            mask_image = Image.new("RGB", (512, 512), (0, 255, 0))
            mask_image.paste(
                pipeline.GROOMING_QWEN_GUIDE_FOREGROUND_RGB,
                (112, 72, 400, 440),
            )
            mask_image.paste(
                pipeline.GROOMING_QWEN_GUIDE_FORELEG_RGB,
                (180, 220, 220, 340),
            )
            mask_image.save(completion_mask)

            audit = pipeline.prepare_canonical_grooming_foreleg_completion_canvas(
                candidate,
                completion_mask,
                canonical,
                repair,
                noise_mask_target=noise_mask,
            )

            with Image.open(repair) as loaded:
                repaired = loaded.convert("RGB")
            self.assertEqual(repaired.getpixel((195, 270)), (90, 100, 110))
            self.assertEqual(repaired.getpixel((130, 150)), (90, 100, 110))
            self.assertEqual(repaired.getpixel((20, 20)), (0, 255, 0))
            self.assertEqual(
                audit["policy"],
                pipeline.GROOMING_CANONICAL_FORELEG_COMPLETION_POLICY,
            )
            self.assertFalse(audit["semantic_pixels_exposed_to_qwen"])
            self.assertFalse(audit["identity_or_geometry_seed_inserted"])
            self.assertTrue(audit["source_composite_preserved_before_masked_sampling"])
            self.assertEqual(audit["semantic_foreleg_pixels"], 4800)
            self.assertGreater(audit["intersected_source_foreground_pixels"], 256)
            self.assertEqual(
                audit["noise_mask_target_sha256"],
                pipeline.file_sha256(noise_mask),
            )
            with Image.open(noise_mask) as loaded:
                mask_rgba = loaded.convert("RGBA")
            self.assertEqual(mask_rgba.getpixel((195, 270))[3], 0)
            self.assertEqual(mask_rgba.getpixel((20, 20))[3], 255)

    def test_full_pose_recovery_guide_exposes_only_public_open_channel(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            public = root / "public.png"
            private = root / "private.png"
            semantic = root / "semantic.png"
            canonical = root / "canonical.png"
            target = root / "recovery.png"

            private_array = np.full((512, 512, 3), (0, 255, 0), dtype=np.uint8)
            private_array[80:440, 112:400] = (96, 96, 96)
            private_array[80:260, 112:256] = (176, 176, 176)
            private_array[260:360, 112:256] = (136, 136, 136)
            Image.fromarray(private_array, mode="RGB").save(private)

            public_image = Image.new("RGB", (512, 512), (0, 255, 0))
            public_image.paste(
                pipeline.GROOMING_QWEN_GUIDE_FOREGROUND_RGB,
                (112, 80, 400, 440),
            )
            ImageDraw.Draw(public_image).line(
                [(208, 245), (218, 275), (216, 310), (204, 335)],
                fill=(0, 255, 0),
                width=9,
            )
            public_image.save(public)

            semantic_array = np.full(
                (512, 512, 3),
                (0, 255, 0),
                dtype=np.uint8,
            )
            semantic_array[80:440, 112:400] = (
                pipeline.GROOMING_QWEN_GUIDE_FOREGROUND_RGB
            )
            semantic_array[220:300, 160:220] = pipeline.GROOMING_QWEN_GUIDE_FORELEG_RGB
            Image.fromarray(semantic_array, mode="RGB").save(semantic)

            canonical_array = np.full(
                (512, 512, 3),
                (0, 255, 0),
                dtype=np.uint8,
            )
            canonical_array[72:440, 120:392] = (208, 208, 208)
            Image.fromarray(canonical_array, mode="RGB").save(canonical)

            audit = pipeline.prepare_grooming_full_pose_recovery_guide(
                public,
                private,
                semantic,
                canonical,
                target,
            )

            with Image.open(target) as loaded:
                output = np.asarray(loaded.convert("RGB"), dtype=np.uint8)
            self.assertEqual(output[20, 20].tolist(), [0, 255, 0])
            self.assertEqual(output[120, 300].tolist(), [144, 144, 144])
            self.assertEqual(output[275, 218].tolist(), [0, 255, 0])
            self.assertTrue(np.array_equal(output, np.asarray(public_image)))
            self.assertEqual(
                audit["policy"],
                pipeline.GROOMING_CANONICAL_FULL_POSE_GUIDE_POLICY,
            )
            self.assertEqual(audit["semantic_foreleg_pixels"], 4800)
            self.assertFalse(audit["semantic_foreleg_exposed_as_discrete_label"])
            self.assertTrue(audit["open_curve_exposed"])
            self.assertTrue(audit["open_curve_is_background_negative_space"])
            self.assertFalse(audit["private_luma_exposed"])
            self.assertFalse(audit["private_raw_luma_values_preserved"])
            self.assertEqual(audit["unique_foreground_rgb_count"], 1)
            self.assertGreaterEqual(audit["internal_open_channel_pixels"], 300)
            self.assertFalse(audit["spatial_canonical_pixels_copied"])
            self.assertFalse(audit["main_case_rgb_or_luma_preserved"])

    def test_full_pose_refine_canvas_preserves_rgba_and_composites_green(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "rescue-cutout.png"
            target = root / "refine-canvas.png"
            source_image = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
            source_image.paste(
                (210, 220, 230, 255),
                (120, 72, 392, 440),
            )
            source_image.save(source)

            audit = pipeline.prepare_canonical_grooming_full_pose_refine_canvas(
                source,
                target,
            )

            with Image.open(target) as loaded:
                canvas = loaded.convert("RGB")
            self.assertEqual(canvas.getpixel((20, 20)), (0, 255, 0))
            self.assertEqual(canvas.getpixel((256, 256)), (210, 220, 230))
            self.assertEqual(
                audit["policy"],
                pipeline.GROOMING_CANONICAL_FULL_POSE_REFINE_POLICY,
            )
            self.assertFalse(audit["source_rgba_pixels_modified_before_composite"])
            self.assertTrue(audit["transparent_background_composited_to_pure_green"])
            self.assertEqual(audit["source_sha256"], pipeline.file_sha256(source))
            self.assertEqual(audit["target_sha256"], pipeline.file_sha256(target))
            self.assertEqual(audit["source_foreground_bbox"], [120, 72, 392, 440])

    def test_canonical_grooming_face_reference_is_derived_from_same_identity(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical = root / "canonical.png"
            target = root / "canonical-face.png"
            image = Image.new("RGB", (512, 512), (0, 255, 0))
            draw = ImageDraw.Draw(image)
            draw.ellipse((126, 58, 386, 452), fill=(232, 232, 232))
            draw.ellipse((164, 84, 348, 250), fill=(244, 244, 244))
            draw.ellipse((205, 145, 225, 170), fill=(30, 120, 45))
            draw.ellipse((285, 145, 305, 170), fill=(30, 120, 45))
            image.save(canonical)

            audit = pipeline.prepare_canonical_grooming_face_identity_reference(
                canonical,
                target,
            )

            with Image.open(target) as loaded:
                face = loaded.convert("RGB")
            self.assertEqual(face.size, (512, 512))
            self.assertEqual(face.getpixel((0, 0)), (0, 255, 0))
            self.assertNotEqual(face.getpixel((256, 256)), (0, 255, 0))
            self.assertTrue(audit["derived_from_same_canonical_identity"])
            self.assertTrue(audit["face_detail_authority"])
            self.assertFalse(audit["pose_authority"])
            self.assertEqual(
                audit["canonical_reference_sha256"],
                pipeline.file_sha256(canonical),
            )

    def test_grooming_pose_transfer_seed_keeps_alpha_geometry_but_removes_appearance(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            action = root / "grooming-action.png"
            action_inverse = root / "grooming-action-inverse.png"
            canonical = root / "canonical.png"
            geometry = root / "grooming-geometry.png"
            foreleg = root / "grooming-foreleg.png"
            target = root / "pose-transfer-seed.png"
            inverse_target = root / "pose-transfer-seed-inverse.png"

            action_mask = np.zeros((512, 512), dtype=np.uint8)
            cv2.ellipse(action_mask, (270, 300), (120, 150), 0, 0, 360, 255, -1)
            cv2.line(action_mask, (210, 330), (190, 235), 255, 42)
            cv2.circle(action_mask, (190, 225), 24, 255, -1)
            action_array = np.zeros((512, 512, 4), dtype=np.uint8)
            yy, xx = np.indices((512, 512))
            checker = (((xx // 24) + (yy // 24)) % 2) == 0
            foreground = action_mask > 0
            action_array[:, :, 3] = action_mask
            action_array[foreground & checker, :3] = (245, 40, 30)
            action_array[foreground & ~checker, :3] = (20, 50, 235)
            Image.fromarray(action_array, mode="RGBA").save(action)
            inverse_action_array = action_array.copy()
            inverse_action_array[foreground & checker, :3] = (10, 245, 220)
            inverse_action_array[foreground & ~checker, :3] = (250, 235, 15)
            Image.fromarray(inverse_action_array, mode="RGBA").save(action_inverse)

            geometry_array = np.full(
                (512, 512, 3),
                (0, 255, 0),
                dtype=np.uint8,
            )
            geometry_foreground = foreground.copy()
            geometry_foreground[240:310, 250:270] = False
            geometry_array[geometry_foreground] = (144, 144, 144)
            Image.fromarray(geometry_array, mode="RGB").save(geometry)

            foreleg_array = np.full(
                (512, 512, 3),
                (0, 255, 0),
                dtype=np.uint8,
            )
            foreleg_array[geometry_foreground] = (144, 144, 144)
            semantic_foreleg = np.zeros((512, 512), dtype=np.uint8)
            cv2.line(semantic_foreleg, (245, 320), (245, 220), 255, 26)
            cv2.circle(semantic_foreleg, (245, 215), 18, 255, -1)
            semantic_foreleg_mask = (semantic_foreleg > 0) & foreground
            foreleg_array[semantic_foreleg_mask] = (176, 176, 176)
            Image.fromarray(foreleg_array, mode="RGB").save(foreleg)

            canonical_array = np.full(
                (512, 512, 3),
                (0, 255, 0),
                dtype=np.uint8,
            )
            canonical_array[72:448, 128:392] = (216, 222, 230)
            canonical_array[150:210, 184:336] = (176, 184, 196)
            Image.fromarray(canonical_array, mode="RGB").save(canonical)

            audit = pipeline.prepare_canonical_grooming_pose_transfer_seed(
                action,
                canonical,
                target,
                geometry_guide=geometry,
                foreleg_guide=foreleg,
            )
            inverse_audit = pipeline.prepare_canonical_grooming_pose_transfer_seed(
                action_inverse,
                canonical,
                inverse_target,
                geometry_guide=geometry,
                foreleg_guide=foreleg,
            )

            with Image.open(target) as loaded:
                output = np.asarray(loaded.convert("RGB"), dtype=np.uint8)
            self.assertTrue(np.all(output[~foreground] == (0, 255, 0)))
            self.assertFalse(
                np.any(np.all(output[foreground] == (245, 40, 30), axis=1))
            )
            self.assertFalse(
                np.any(np.all(output[foreground] == (20, 50, 235), axis=1))
            )
            self.assertEqual(
                audit["policy"],
                pipeline.GROOMING_CANONICAL_POSE_TRANSFER_SEED_POLICY,
            )
            self.assertFalse(audit["source_alpha_geometry_preserved"])
            self.assertFalse(audit["action_alpha_overlap_blocking"])
            self.assertFalse(audit["action_reference_used_for_conditioning"])
            self.assertGreater(audit["public_internal_channel_pixels_filled"], 0)
            self.assertEqual(audit["qwen_visible_internal_green_channel_pixels"], 0)
            self.assertFalse(audit["source_rgb_preserved"])
            self.assertFalse(audit["source_chroma_preserved"])
            self.assertFalse(audit["source_raw_luma_values_preserved"])
            self.assertFalse(audit["source_low_frequency_luma_exposed"])
            self.assertFalse(audit["source_luma_used_for_shading"])
            self.assertFalse(audit["source_high_frequency_texture_exposed"])
            self.assertFalse(audit["geometry_distance_field_only"])
            self.assertTrue(audit["geometry_and_soft_foreleg_depth_only"])
            self.assertTrue(audit["soft_foreleg_depth_semantic_exposed"])
            self.assertFalse(audit["semantic_foreleg_discrete_color_exposed"])
            self.assertFalse(audit["public_open_channel_geometry_exposed"])
            self.assertTrue(audit["canonical_palette_conditioning_only"])
            self.assertLessEqual(audit["target_luma_range"], 24.0)
            self.assertEqual(audit["geometry_guide_unique_foreground_rgb_count"], 1)
            self.assertEqual(target.read_bytes(), inverse_target.read_bytes())
            self.assertNotEqual(
                audit["action_reference_sha256"],
                inverse_audit["action_reference_sha256"],
            )
            self.assertEqual(audit["target_sha256"], pipeline.file_sha256(target))

            depth_geometry = root / "grooming-white-depth.png"
            depth_target = root / "pose-transfer-white-depth.png"
            depth_inverse_target = root / "pose-transfer-white-depth-inverse.png"
            depth_geometry_array = np.full(
                (512, 512, 3),
                (0, 255, 0),
                dtype=np.uint8,
            )
            depth_distance = cv2.distanceTransform(
                geometry_foreground.astype(np.uint8),
                cv2.DIST_L2,
                5,
            )
            depth_values = 104.0 + 72.0 * np.clip(
                depth_distance / max(float(depth_distance.max()), 1.0),
                0.0,
                1.0,
            )
            for channel in range(3):
                depth_geometry_array[:, :, channel][geometry_foreground] = (
                    np.rint(depth_values[geometry_foreground]).astype(np.uint8)
                )
            Image.fromarray(depth_geometry_array, mode="RGB").save(depth_geometry)

            depth_audit = pipeline.prepare_canonical_grooming_pose_transfer_seed(
                action,
                canonical,
                depth_target,
                geometry_guide=depth_geometry,
                foreleg_guide=foreleg,
                preserve_normalized_geometry_luma=True,
            )
            pipeline.prepare_canonical_grooming_pose_transfer_seed(
                action_inverse,
                canonical,
                depth_inverse_target,
                geometry_guide=depth_geometry,
                foreleg_guide=foreleg,
                preserve_normalized_geometry_luma=True,
            )

            self.assertTrue(depth_audit["normalized_geometry_luma_exposed"])
            self.assertTrue(
                depth_audit["source_normalized_low_frequency_geometry_exposed"]
            )
            self.assertTrue(depth_audit["normalized_geometry_luma_used_for_shading"])
            self.assertTrue(
                depth_audit["normalized_white_maquette_depth_anatomy_only"]
            )
            self.assertFalse(depth_audit["soft_foreleg_depth_semantic_exposed"])
            self.assertEqual(depth_audit["foreleg_luma_lift"], 0.0)
            self.assertGreater(
                depth_audit["geometry_guide_unique_foreground_rgb_count"],
                2,
            )
            self.assertGreaterEqual(depth_audit["target_luma_range"], 24)
            self.assertEqual(
                depth_audit["sampler_role"],
                "canonical_palette_normalized_white_maquette_depth_anatomy",
            )
            self.assertEqual(
                depth_target.read_bytes(),
                depth_inverse_target.read_bytes(),
            )
            self.assertNotEqual(target.read_bytes(), depth_target.read_bytes())

            frame0_geometry = root / "grooming-frame000-body-pose.png"
            frame0_target = root / "pose-transfer-frame000.png"
            frame0_array = np.full(
                (512, 512, 3),
                (0, 255, 0),
                dtype=np.uint8,
            )
            frame0_distance = cv2.distanceTransform(
                foreground.astype(np.uint8),
                cv2.DIST_L2,
                5,
            )
            frame0_values = 96.0 + 80.0 * np.clip(
                frame0_distance / max(float(frame0_distance.max()), 1.0),
                0.0,
                1.0,
            )
            for channel in range(3):
                frame0_array[:, :, channel][foreground] = np.rint(
                    frame0_values[foreground]
                ).astype(np.uint8)
            Image.fromarray(frame0_array, mode="RGB").save(frame0_geometry)

            frame0_audit = pipeline.prepare_canonical_grooming_pose_transfer_seed(
                action,
                canonical,
                frame0_target,
                geometry_guide=frame0_geometry,
                foreleg_guide=None,
                preserve_normalized_geometry_luma=True,
                complete_geometry_guide=True,
                remove_head_identity_silhouette=True,
            )

            self.assertTrue(frame0_audit["complete_geometry_guide"])
            self.assertIsNone(frame0_audit["foreleg_guide_path"])
            self.assertEqual(frame0_audit["semantic_foreleg_pixels"], 0)
            self.assertEqual(frame0_audit["public_internal_channel_pixels_filled"], 0)
            self.assertEqual(
                frame0_audit["sampler_role"],
                "canonical_palette_headless_white_maquette_frame000_body_pose_anatomy",
            )
            self.assertTrue(frame0_audit["head_identity_silhouette_removed"])
            self.assertGreater(
                frame0_audit["head_identity_silhouette_removed_pixels"],
                4096,
            )
            self.assertGreater(
                frame0_audit["qwen_visible_internal_green_channel_pixels"],
                4096,
            )
            self.assertGreater(
                frame0_audit["raised_paw_geometry_retained_pixels"],
                512,
            )
            self.assertFalse(frame0_audit["soft_foreleg_depth_semantic_exposed"])

    def test_grooming_seed_bound_matte_unions_alpha_without_copying_seed_rgb(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "qwen-source.png"
            semantic = root / "semantic.png"
            seed = root / "pose-seed.png"
            target = root / "seed-bound.png"
            green = root / "seed-bound-green.png"

            seed_mask = np.zeros((512, 512), dtype=np.uint8)
            cv2.ellipse(seed_mask, (280, 310), (120, 165), 0, 0, 360, 255, -1)
            cv2.line(seed_mask, (220, 345), (194, 235), 255, 46)
            cv2.circle(seed_mask, (194, 224), 24, 255, -1)
            seed_array = np.full((512, 512, 3), (0, 255, 0), dtype=np.uint8)
            seed_array[seed_mask > 0] = (212, 214, 218)
            Image.fromarray(seed_array, mode="RGB").save(seed)

            source_array = np.full((1024, 1024, 3), (171, 119, 83), dtype=np.uint8)
            source_array[:, :, 0] += np.arange(1024, dtype=np.uint8)[None, :] // 16
            Image.fromarray(source_array, mode="RGB").save(source)
            resized_seed = np.asarray(
                Image.fromarray(seed_mask, mode="L").resize(
                    (1024, 1024),
                    Image.Resampling.NEAREST,
                ),
                dtype=np.uint8,
            )
            semantic_mask = (resized_seed > 0) & (
                np.indices((1024, 1024))[1] < 690
            )
            semantic_array = np.zeros((1024, 1024, 4), dtype=np.uint8)
            semantic_array[semantic_mask, :3] = (12, 34, 220)
            semantic_array[semantic_mask, 3] = 255
            Image.fromarray(semantic_array, mode="RGBA").save(semantic)

            audit = pipeline.prepare_canonical_grooming_seed_bound_matte(
                source,
                semantic,
                seed,
                target,
                green_target=green,
            )

            with Image.open(target) as loaded:
                output = np.asarray(loaded.convert("RGBA"), dtype=np.uint8)
            expected_union = resized_seed > 0
            self.assertTrue(np.array_equal(output[:, :, 3] >= 96, expected_union))
            self.assertTrue(
                np.array_equal(
                    output[:, :, :3][expected_union],
                    source_array[expected_union],
                )
            )
            self.assertEqual(
                audit["policy"],
                pipeline.GROOMING_CANONICAL_SEED_BOUND_MATTE_POLICY,
            )
            self.assertTrue(audit["acceptable"])
            self.assertTrue(all(audit["checks"].values()))
            self.assertTrue(audit["source_rgb_preserved_where_alpha_nonzero"])
            self.assertFalse(audit["pose_transfer_seed_rgb_copied"])
            self.assertTrue(audit["pose_transfer_seed_used_for_alpha_geometry_only"])
            self.assertGreater(audit["seed_added_pixels"], 0)
            self.assertEqual(audit["target_sha256"], pipeline.file_sha256(target))
            self.assertTrue(green.is_file())

    def test_grooming_seed_bound_matte_rejects_shifted_semantic_subject(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "qwen-source.png"
            semantic = root / "semantic.png"
            seed = root / "pose-seed.png"
            target = root / "seed-bound.png"
            Image.new("RGB", (1024, 1024), (180, 140, 90)).save(source)
            seed_array = np.full((512, 512, 3), (0, 255, 0), dtype=np.uint8)
            cv2.ellipse(
                seed_array,
                (360, 350),
                (90, 130),
                0,
                0,
                360,
                (210, 210, 210),
                -1,
            )
            Image.fromarray(seed_array, mode="RGB").save(seed)
            semantic_array = np.zeros((1024, 1024, 4), dtype=np.uint8)
            cv2.ellipse(
                semantic_array,
                (220, 250),
                (170, 210),
                0,
                0,
                360,
                (40, 50, 60, 255),
                -1,
            )
            Image.fromarray(semantic_array, mode="RGBA").save(semantic)

            with self.assertRaisesRegex(ValueError, "matte_alignment_failed"):
                pipeline.prepare_canonical_grooming_seed_bound_matte(
                    source,
                    semantic,
                    seed,
                    target,
                )

    def test_pose_safe_identity_refine_eligibility_requires_all_independent_gates(
        self,
    ) -> None:
        matte = {
            "policy": pipeline.GROOMING_CANONICAL_SEED_BOUND_MATTE_POLICY,
            "acceptable": True,
            "fail_closed": True,
            "source_sha256": "a" * 64,
            "semantic_cutout_sha256": "b" * 64,
            "pose_transfer_seed_sha256": "c" * 64,
            "target_sha256": "d" * 64,
            "union_alpha_sha256": "e" * 64,
            "checks": {"alignment": True, "area": True},
            "pose_transfer_seed_used_for_alpha_geometry_only": True,
            "pose_transfer_seed_rgb_copied": False,
            "pose_transfer_seed_luma_or_chroma_copied": False,
        }
        pose_gate = {
            "available": True,
            "evaluated": True,
            "acceptable": True,
            "issues": [],
            "metrics": {
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
            },
        }
        pose_quality = {"acceptable": True, "issues": [], "penalty": 0.0}
        palette = {
            "available": True,
            "acceptable": False,
            "issues": [
                "new_dominant_warm_brown_or_orange",
                "achromatic_source_recolored_with_warm_fur",
                "canonical_action_achromatic_value_balance_drift",
            ],
        }
        surface = {
            "available": True,
            "evaluated": True,
            "acceptable": True,
            "issues": [],
            "gate": {
                "alpha_provenance": (
                    pipeline.GROOMING_CANONICAL_SEED_BOUND_MATTE_ALPHA_PROVENANCE
                )
            },
        }

        accepted = pipeline.canonical_grooming_pose_safe_identity_refine_eligibility(
            attempt=pipeline.GROOMING_CANONICAL_FULL_POSE_RESCUE_ATTEMPT,
            seed_bound_matte=matte,
            pose_gate=pose_gate,
            pose_quality=pose_quality,
            identity_palette_quality=palette,
            fur_surface_gate=surface,
        )
        self.assertTrue(accepted["eligible"])
        self.assertTrue(all(accepted["checks"].values()))
        self.assertFalse(accepted["identity_palette_gate_bypassed"])

        prop_palette = {
            **palette,
            "issues": ["raised_paw_material_inconsistent_with_planted_paw"],
        }
        rejected = pipeline.canonical_grooming_pose_safe_identity_refine_eligibility(
            attempt=pipeline.GROOMING_CANONICAL_FULL_POSE_RESCUE_ATTEMPT,
            seed_bound_matte=matte,
            pose_gate={
                **pose_gate,
                "metrics": {
                    **pose_gate["metrics"],
                    "raised_foreleg_structural_evidence": {
                        "available": True,
                        "edge_density": 0.009,
                    },
                },
            },
            pose_quality=pose_quality,
            identity_palette_quality=prop_palette,
            fur_surface_gate=surface,
        )
        self.assertFalse(rejected["eligible"])
        self.assertFalse(rejected["checks"]["organic_structural_density_passed"])
        self.assertFalse(
            rejected["checks"][
                "identity_gate_failed_only_for_repaintable_palette_drift"
            ]
        )

    def test_advisory_grooming_identity_refine_locks_body_and_raised_paw(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "accepted-grooming-action.png"
            canvas = root / "masked-refine-canvas.png"
            noise_mask = root / "masked-refine-noise.png"
            generated = root / "raw-qwen-refine.png"
            target = root / "locked-refine-result.png"

            source_image = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
            source_draw = ImageDraw.Draw(source_image)
            source_draw.ellipse(
                (112, 48, 400, 464),
                fill=(96, 104, 112, 255),
            )
            # A raised foreleg and paw deliberately sit below the head-edit cutoff.
            source_draw.line(
                [(190, 372), (178, 300), (198, 250)],
                fill=(210, 214, 218, 255),
                width=42,
            )
            source_draw.ellipse(
                (178, 232, 220, 270),
                fill=(220, 224, 228, 255),
            )
            source_draw.rectangle(
                (60, 300, 145, 440),
                fill=(4, 8, 6, 255),
            )
            source_image.save(source)
            Image.new("RGB", (1024, 1024), (210, 80, 96)).save(generated)

            canvas_audit = (
                pipeline.prepare_advisory_grooming_identity_refine_canvas(
                    source,
                    canvas,
                    noise_mask,
                    identity_reference_palette={
                        "fractions": {"black": 0.01},
                    },
                )
            )
            composite_audit = (
                pipeline.finalize_advisory_grooming_identity_refine_candidate(
                    Path(canvas_audit["sanitized_source_path"]),
                    generated,
                    noise_mask,
                    target,
                )
            )

            self.assertEqual(
                canvas_audit["sampler_canvas_scope"],
                "upper_head_mask_only_action_pixels_locked",
            )
            self.assertTrue(canvas_audit["raised_paw_editing_forbidden"])
            self.assertGreater(canvas_audit["protected_body_pixels"], 2048)
            self.assertEqual(
                canvas_audit["head_silhouette_expansion_px"],
                pipeline.GROOMING_ADVISORY_IDENTITY_REFINE_HEAD_EXPANSION_PX,
            )
            self.assertEqual(canvas_audit["editable_background_pixels"], 0)
            self.assertGreater(canvas_audit["dark_artifact_removed_pixels"], 2048)
            self.assertEqual(
                composite_audit["protected_action_pixels_changed"],
                0,
            )
            self.assertTrue(
                composite_audit["body_and_raised_paw_pixel_lock_passed"]
            )
            self.assertEqual(
                composite_audit["head_silhouette_expansion_pixels"],
                0,
            )
            with Image.open(target) as loaded:
                result = loaded.convert("RGB")
            self.assertEqual(result.getpixel((0, 0)), (0, 255, 0))
            self.assertEqual(result.getpixel((80, 350)), (0, 255, 0))
            self.assertEqual(result.getpixel((198, 250)), (220, 224, 228))
            self.assertEqual(result.getpixel((256, 380)), (96, 104, 112))
            self.assertEqual(result.getpixel((256, 100)), (210, 80, 96))

    def test_eating_identity_refine_canvas_uses_action_specific_policy(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "eating-cutout.png"
            target = root / "eating-refine.png"
            source_image = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
            source_image.paste(
                (224, 220, 210, 255),
                (96, 96, 416, 440),
            )
            source_image.save(source)

            audit = pipeline.prepare_canonical_eating_identity_refine_canvas(
                source,
                target,
            )

            self.assertEqual(audit["action"], "eating")
            self.assertEqual(
                audit["policy"],
                pipeline.EATING_CANONICAL_IDENTITY_REFINE_POLICY,
            )
            self.assertTrue(audit["transparent_background_composited_to_pure_green"])
            self.assertFalse(audit["source_rgba_pixels_modified_before_composite"])
            with Image.open(target) as loaded:
                canvas = loaded.convert("RGB")
            self.assertEqual(canvas.getpixel((20, 20)), (0, 255, 0))
            self.assertEqual(canvas.getpixel((256, 256)), (224, 220, 210))

    def test_two_image_qwen_workflow_keeps_loras_and_adds_pose_reference(self) -> None:
        loras = [
            {
                "model_name": "user_uploads/cat-photo-v3.safetensors",
                "original_filename": "宠物小猫 _ 写实摄影_v3.0.safetensors",
                "strength": 0.5,
            },
            {
                "model_name": "user_uploads/kitten-photo.safetensors",
                "original_filename": "Qwen Image-Kitten Photography_QWEN-小猫摄影.safetensors",
                "strength": 0.6,
            },
        ]
        workflow = pipeline.qwen_workflow(
            "subject.png",
            "positive",
            "negative",
            123,
            "test/output",
            user_loras=loras,
            user_lora_mode="replace",
            suppress_builtin_style=True,
            pose_reference_input_name="grooming-first-frame.png",
        )

        loaded_loras = [
            node["inputs"]
            for node in workflow.values()
            if node.get("class_type") == "LoraLoaderModelOnly"
            and node["inputs"].get("lora_name")
            in {item["model_name"] for item in loras}
        ]
        self.assertEqual(
            [(item["lora_name"], item["strength_model"]) for item in loaded_loras],
            [(loras[0]["model_name"], 0.5), (loras[1]["model_name"], 0.6)],
        )
        pose_load_nodes = [
            node_id
            for node_id, node in workflow.items()
            if node.get("class_type") == "LoadImage"
            and node["inputs"].get("image") == "grooming-first-frame.png"
        ]
        self.assertEqual(len(pose_load_nodes), 1)
        pose_scale_nodes = [
            node_id
            for node_id, node in workflow.items()
            if node.get("class_type") == "FluxKontextImageScale"
            and node["inputs"].get("image") == [pose_load_nodes[0], 0]
        ]
        self.assertEqual(len(pose_scale_nodes), 1)
        pose_link = [pose_scale_nodes[0], 0]
        canonical_link = ["5", 0]
        self.assertEqual(workflow["6"]["inputs"]["image1"], canonical_link)
        self.assertEqual(workflow["7"]["inputs"]["image1"], canonical_link)
        self.assertIn("image2", workflow["6"]["inputs"])
        self.assertEqual(
            workflow["6"]["inputs"]["image2"], workflow["7"]["inputs"]["image2"]
        )
        self.assertEqual(workflow["6"]["inputs"]["image2"], pose_link)
        self.assertNotIn("image3", workflow["6"]["inputs"])
        self.assertNotIn("image3", workflow["7"]["inputs"])
        self.assertEqual(workflow["10"]["inputs"]["pixels"], canonical_link)
        self.assertEqual(workflow["14"]["inputs"]["denoise"], 1.0)

    def test_three_view_qwen_workflow_wires_image1_image2_image3_only_for_canonical(
        self,
    ) -> None:
        workflow = pipeline.qwen_workflow(
            "pet-front.png",
            "positive",
            "negative",
            123,
            "test/three-view",
            suppress_builtin_style=True,
            additional_reference_input_names=[
                "pet-left.png",
                "pet-right.png",
            ],
        )

        scaled_by_source: dict[str, list[object]] = {}
        for node_id, node in workflow.items():
            if node.get("class_type") != "LoadImage":
                continue
            source = node["inputs"]["image"]
            scale_nodes = [
                scale_id
                for scale_id, scale_node in workflow.items()
                if scale_node.get("class_type") == "FluxKontextImageScale"
                and scale_node["inputs"].get("image") == [node_id, 0]
            ]
            self.assertEqual(len(scale_nodes), 1)
            scaled_by_source[source] = [scale_nodes[0], 0]

        self.assertEqual(workflow["6"]["inputs"]["image1"], ["5", 0])
        self.assertEqual(
            workflow["6"]["inputs"]["image2"],
            scaled_by_source["pet-left.png"],
        )
        self.assertEqual(
            workflow["6"]["inputs"]["image3"],
            scaled_by_source["pet-right.png"],
        )
        self.assertEqual(
            workflow["6"]["inputs"]["image2"],
            workflow["7"]["inputs"]["image2"],
        )
        self.assertEqual(
            workflow["6"]["inputs"]["image3"],
            workflow["7"]["inputs"]["image3"],
        )
        self.assertEqual(workflow["10"]["inputs"]["pixels"], ["5", 0])
        self.assertEqual(workflow["14"]["inputs"]["latent_image"], ["10", 0])

    def test_qwen_workflow_rejects_conflicting_or_excess_reference_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be used together"):
            pipeline.qwen_workflow(
                "front.png",
                "positive",
                "negative",
                1,
                "test/conflict",
                pose_reference_input_name="pose.png",
                additional_reference_input_names=["left.png"],
            )
        with self.assertRaisesRegex(ValueError, "at most three images"):
            pipeline.qwen_workflow(
                "front.png",
                "positive",
                "negative",
                1,
                "test/excess",
                additional_reference_input_names=[
                    "left.png",
                    "right.png",
                    "fourth.png",
                ],
            )

    def test_uploaded_qwen_stage_uses_v2_loras_and_returns_alpha_references(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_root = root / "outputs"
            comfy_input = root / "comfy-input"
            job_id = "test-qwen-reference"
            job_dir = output_root / job_id
            job_dir.mkdir(parents=True)
            upload_digests = _write_test_upload_identity(job_dir)
            profile = _subject_profile_with_action_references(root)
            with patch.object(
                pipeline,
                "GENERATION_VISUAL_AUDITS_BLOCKING",
                True,
            ):
                effective_config, expected_pose_policies = (
                    _configured_test_upload_profile(
                        profile,
                        ["eating", "grooming"],
                    )
                )
            jobs = {
                job_id: {
                    "job_id": job_id,
                    "generation_mode": "subject_replacement",
                    "subject_source": "upload_pending_qwen",
                    "prompt": "fluffy long soft fur",
                    "prompt_en": "fluffy long soft fur",
                    "outputs": [],
                    "cancel_requested": False,
                    "effective_subject_config": effective_config,
                    "expected_pose_policies": expected_pose_policies,
                    **upload_digests,
                }
            }
            expected_loras = [
                dict(item)
                for item in pipeline.IMAGE_PROFILES[
                    pipeline.SUBJECT_REPLACEMENT_REFERENCE_IMAGE_PROFILE
                ]["user_loras"]
            ]
            eating_driver_sha256 = pipeline.file_sha256(
                Path(profile["runtime"]["actions"]["eating"]["driving_animation"])
            )
            guide_sources: dict[str, Path] = {}
            surface_evaluations: list[tuple[Path, str]] = []
            canonical_reference = job_dir / "cartoon_reference_transparent.png"
            canonical_qwen_image1 = job_dir / "cartoon_reference.png"

            def fake_eating_guide(
                target_job_id: str,
                source: Path,
                target: Path,
                *,
                frames: int,
                fps: float,
            ) -> dict[str, object]:
                self.assertEqual(target_job_id, job_id)
                guide_sources["eating"] = source
                return _write_fake_geometry_guide(
                    target_job_id,
                    source,
                    target,
                    frames=frames,
                    fps=fps,
                )

            def fake_grooming_guide(
                target_job_id: str,
                source: Path,
                target: Path,
                *,
                frames: int,
                fps: float,
            ) -> dict[str, object]:
                self.assertEqual(target_job_id, job_id)
                guide_sources["grooming"] = source
                target.parent.mkdir(parents=True, exist_ok=True)
                qwen_guide = Image.new("RGB", (512, 512), (0, 255, 0))
                qwen_guide.paste(
                    pipeline.GROOMING_QWEN_GUIDE_FOREGROUND_RGB,
                    (112, 72, 400, 440),
                )
                open_points = [
                    (208, 245),
                    (214, 260),
                    (218, 275),
                    (218, 295),
                    (216, 310),
                    (204, 335),
                ]
                ImageDraw.Draw(qwen_guide).line(
                    open_points,
                    fill=(0, 255, 0),
                    width=9,
                )
                qwen_array = np.asarray(qwen_guide, dtype=np.uint8)
                qwen_body = np.zeros((512, 512), dtype=bool)
                qwen_body[72:440, 112:400] = True
                open_pixels = int(
                    np.count_nonzero(
                        qwen_body & np.all(qwen_array == (0, 255, 0), axis=2)
                    )
                )
                qwen_guide.save(target)
                gate_target = target.with_name(target.stem + "_cv_gate.png")
                gate_guide = Image.new("RGB", (512, 512), (0, 255, 0))
                gate_guide.paste((112, 112, 112), (112, 72, 256, 440))
                gate_guide.paste((168, 168, 168), (256, 72, 400, 440))
                gate_guide.save(gate_target)
                completion_mask_target = target.with_name(
                    target.stem + "_completion_mask.png"
                )
                completion_mask = Image.new("RGB", (512, 512), (0, 255, 0))
                completion_mask.paste(
                    pipeline.GROOMING_QWEN_GUIDE_FOREGROUND_RGB,
                    (112, 72, 400, 440),
                )
                completion_mask.paste(
                    pipeline.GROOMING_QWEN_GUIDE_FORELEG_RGB,
                    (180, 220, 220, 340),
                )
                completion_mask.save(completion_mask_target)
                gate_metadata_path = gate_target.with_name(
                    gate_target.stem + "_metadata.json"
                )
                gate_metadata_path.write_text(
                    json.dumps(
                        {
                            "policy": (
                                pipeline.GROOMING_PRIVATE_POSE_GATE_GUIDE_POLICY
                            ),
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
                guide_digest = pipeline.file_sha256(target)
                gate_digest = pipeline.file_sha256(gate_target)
                completion_mask_digest = pipeline.file_sha256(completion_mask_target)
                metadata_path = target.with_name(target.stem + "_metadata.json")
                metadata: dict[str, object] = {
                    "policy": pipeline.GROOMING_QWEN_GEOMETRY_GUIDE_POLICY,
                    "method": pipeline.GROOMING_QWEN_GEOMETRY_GUIDE_METHOD,
                    "source_sha256": pipeline.file_sha256(source),
                    "guide_sha256": guide_digest,
                    "metadata_path": str(metadata_path),
                    "frames": frames,
                    "fps": fps,
                    "qwen_guide": {
                        "role": "image2_action_geometry_guide",
                        "path": str(target),
                        "sha256": guide_digest,
                        "foreground_rgb": list(
                            pipeline.GROOMING_QWEN_GUIDE_FOREGROUND_RGB
                        ),
                        "foreground_palette_rgb": [
                            list(pipeline.GROOMING_QWEN_GUIDE_FOREGROUND_RGB),
                        ],
                        "unique_foreground_rgb_count": 1,
                        "internal_luma_variation": False,
                        "internal_luma_variation_role": "none",
                        "appearance_bearing_internal_luma_variation": False,
                        "main_case_appearance_exposed_to_qwen": False,
                        "synthetic_foreleg_semantic": None,
                        "structural_negative_space": {
                            "method": pipeline.GROOMING_QWEN_NEGATIVE_SPACE_METHOD,
                            "pixels": open_pixels,
                            "line_width": 9,
                            "points": [list(point) for point in open_points],
                            "curve_closed": False,
                            "paw_endpoint_open": True,
                            "shoulder_endpoint_open": True,
                            "foreleg_remains_attached_at_shoulder": True,
                            "paw_muzzle_contact_remains_connected": True,
                            "source_rgb_or_luma_sampled": False,
                            "appearance_bearing": False,
                            "adds_foreground_color": False,
                        },
                        "closed_foreleg_capsule_exposed": False,
                        "internal_green_separator_exposed": True,
                        "raised_foreleg_validation_role": (
                            "public_open_curve_geometry_plus_private_cv"
                        ),
                    },
                    "private_pose_gate": {
                        "guide_path": str(gate_target),
                        "guide_sha256": gate_digest,
                        "metadata_path": str(gate_metadata_path),
                        "policy": (pipeline.GROOMING_PRIVATE_POSE_GATE_GUIDE_POLICY),
                        "exposed_to_qwen": False,
                    },
                    "private_completion_mask": {
                        "role": "local_noise_mask_derivation_only",
                        "policy": (pipeline.GROOMING_PRIVATE_COMPLETION_MASK_POLICY),
                        "path": str(completion_mask_target),
                        "sha256": completion_mask_digest,
                        "exposed_to_qwen": False,
                        "loaded_as_qwen_image": False,
                        "semantic_foreleg_pixels": 4800,
                    },
                    "palette": {"source_rgb_preserved": False},
                    "raised_paw_muzzle_relation": {"passed": True},
                }
                metadata_path.write_text(
                    json.dumps(metadata),
                    encoding="utf-8",
                )
                return metadata

            def fake_qwen_run(
                target_job_id: str, *, force_qwen_reference: bool = False
            ) -> None:
                self.assertEqual(target_job_id, job_id)
                self.assertTrue(force_qwen_reference)
                prepared_job = jobs[job_id]
                self.assertEqual(
                    prepared_job["pose_identity_input_policy"],
                    pipeline.POSE_IDENTITY_INPUT_POLICY_CANONICAL,
                )
                self.assertEqual(
                    prepared_job["pose_identity_input_policies"],
                    {
                        "eating": pipeline.POSE_IDENTITY_INPUT_POLICY_CANONICAL,
                        "grooming": pipeline.POSE_IDENTITY_INPUT_POLICY_CANONICAL,
                    },
                )
                self.assertFalse(prepared_job["main_case_raw_identity_chain"])
                self.assertFalse(prepared_job["bypass_generated_canonical"])
                self.assertEqual(
                    prepared_job["reference_pipeline"],
                    pipeline.QWEN_FULL_BODY_FIRST_FRAME_PIPELINE,
                )
                for action, input_name in prepared_job["pose_reference_inputs"].items():
                    guide_path = comfy_input / input_name
                    self.assertTrue(guide_path.is_file())
                    self.assertEqual(
                        prepared_job["pose_reference_input_sha256s"][action],
                        pipeline.file_sha256(guide_path),
                    )
                canonical_image = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
                canonical_image.paste((80, 80, 80, 255), (128, 64, 384, 448))
                canonical_image.save(canonical_reference)
                canonical_green = Image.new("RGB", (512, 512), (0, 255, 0))
                canonical_green.paste((80, 80, 80), (128, 64, 384, 448))
                canonical_green.save(canonical_qwen_image1)
                _record_fake_canonical_image1(job_id, job_dir, comfy_input)
                pipeline.append_output(
                    job_id,
                    {
                        "type": "cartoon_reference",
                        "filename": canonical_reference.name,
                        "transparent": True,
                    },
                )
                selected_surface_metrics: dict[str, object] = {}
                for action in ("eating", "grooming"):
                    target = job_dir / f"{action}_transparent.png"
                    image = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
                    image.paste((80, 80, 80, 255), (128, 96, 384, 416))
                    image.save(target)
                    pipeline.append_output(
                        job_id,
                        {
                            "type": "transparent_keyframe",
                            "action": action,
                            "filename": target.name,
                        },
                    )
                    selected_candidate = job_dir / f"{action}_selected_candidate.png"
                    selected_candidate.write_bytes(target.read_bytes())
                    selected_surface_metrics[action] = _passed_surface_metric(
                        selected_candidate,
                        action,
                    )
                pipeline.update_job(
                    job_id,
                    fur_surface_gate_selected_metrics=selected_surface_metrics,
                )

            def fake_surface_evaluation(
                reference: Path,
                **_kwargs: object,
            ) -> dict[str, object]:
                surface_evaluations.append((reference, pipeline.file_sha256(reference)))
                return {
                    "acceptable": True,
                    "status": "passed",
                    "issues": [],
                }

            with (
                patch.multiple(
                    pipeline,
                    OUTPUT_ROOT=output_root,
                    COMFY_INPUT=comfy_input,
                    JOBS=jobs,
                    GENERATION_VISUAL_AUDITS_BLOCKING=True,
                ),
                patch.object(
                    pipeline,
                    "merge_image_profile_loras",
                    return_value=(expected_loras, "replace", "image_profile"),
                ),
                patch.object(
                    pipeline,
                    "load_subject_profile",
                    side_effect=AssertionError("live profile must not be re-read"),
                ),
                patch.object(
                    pipeline,
                    "prepare_eating_qwen_geometry_guide",
                    side_effect=fake_eating_guide,
                ),
                patch.object(
                    pipeline,
                    "prepare_grooming_qwen_geometry_guide",
                    side_effect=fake_grooming_guide,
                ),
                patch.object(
                    pipeline,
                    "evaluate_main_case_fur_surface_candidate",
                    side_effect=fake_surface_evaluation,
                ),
                patch.object(pipeline, "run_job", side_effect=fake_qwen_run),
            ):
                references = pipeline.prepare_uploaded_subject_qwen_references(
                    job_id,
                    ["eating", "grooming"],
                )
                pipeline.revalidate_uploaded_qwen_references_before_scail(
                    job_id,
                    jobs[job_id],
                    jobs[job_id]["effective_subject_config"],
                    ["eating", "grooming"],
                )
                eating_reference = Path(references["eating"])
                eating_reference_bytes = eating_reference.read_bytes()
                eating_reference.write_bytes(b"changed-after-qwen")
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Qwen eating reference changed before SCAIL",
                ):
                    pipeline.revalidate_uploaded_qwen_references_before_scail(
                        job_id,
                        jobs[job_id],
                        jobs[job_id]["effective_subject_config"],
                        ["eating", "grooming"],
                    )
                eating_reference.write_bytes(eating_reference_bytes)

            final_job = jobs[job_id]
            qwen_output_types = [output["type"] for output in final_job["outputs"]]
            self.assertEqual(qwen_output_types[0], "cartoon_reference")
            self.assertEqual(
                qwen_output_types[1:],
                ["transparent_keyframe", "transparent_keyframe"],
            )
            self.assertTrue(
                all(
                    Path(reference) != job_dir / "input.png"
                    for reference in references.values()
                )
            )
            canonical_digest = pipeline.file_sha256(canonical_reference)
            canonical_qwen_digest = pipeline.file_sha256(canonical_qwen_image1)
            with Image.open(canonical_reference) as canonical_image:
                self.assertEqual(canonical_image.size, (512, 512))
                self.assertEqual(canonical_image.mode, "RGBA")
                alpha_minimum, alpha_maximum = canonical_image.getchannel(
                    "A"
                ).getextrema()
                self.assertLess(alpha_minimum, 255)
                self.assertGreater(alpha_maximum, 0)
            for reference in references.values():
                with Image.open(reference) as keyframe_image:
                    self.assertEqual(keyframe_image.size, (512, 512))
                    self.assertEqual(keyframe_image.mode, "RGBA")
                    alpha_minimum, alpha_maximum = keyframe_image.getchannel(
                        "A"
                    ).getextrema()
                    self.assertLess(alpha_minimum, 255)
                    self.assertGreater(alpha_maximum, 0)

        self.assertEqual(set(references), {"eating", "grooming"})
        self.assertEqual(final_job["subject_source"], "qwen_generated_upload")
        self.assertEqual(final_job["reference_generation_status"], "completed")
        self.assertTrue(final_job["canonical_reference_generated"])
        self.assertTrue(final_job["canonical_reference_used"])
        self.assertTrue(final_job["canonical_reference_normalized"])
        self.assertTrue(final_job["canonical_reference_transparent"])
        self.assertTrue(final_job["qwen_action_keyframes_normalized"])
        self.assertTrue(final_job["qwen_action_keyframes_transparent"])
        self.assertEqual(
            final_job["canonical_reference_path"],
            str(canonical_reference),
        )
        self.assertEqual(
            final_job["canonical_reference_sha256"],
            canonical_digest,
        )
        self.assertEqual(final_job["reference_normalization_status"], "completed")
        self.assertEqual(
            set(final_job["qwen_action_keyframe_sha256s"]),
            {"eating", "grooming"},
        )
        self.assertTrue(final_job["qwen_action_reference_byte_preserved"])
        self.assertEqual(
            final_job["qwen_action_reference_normalization_policy"],
            pipeline.QWEN_ACTION_REFERENCE_NORMALIZATION_POLICY,
        )
        self.assertEqual(len(surface_evaluations), 4)
        for evaluated_path, evaluated_digest in surface_evaluations:
            action = evaluated_path.stem.removesuffix("_transparent")
            self.assertIn(action, {"eating", "grooming"})
            self.assertEqual(
                evaluated_digest,
                final_job["qwen_action_keyframe_sha256s"][action],
            )
            self.assertEqual(
                evaluated_digest,
                final_job["subject_fur_surface_quality"][action][
                    "persisted_reference_sha256"
                ],
            )
            self.assertTrue(
                final_job["subject_fur_surface_quality"][action][
                    "byte_identical_to_selected_candidate"
                ]
            )
        self.assertEqual(
            final_job["uploaded_identity_anchor_path"],
            str(job_dir / "input_identity_source.png"),
        )
        self.assertEqual(
            final_job["uploaded_identity_input_sha256"],
            upload_digests["uploaded_identity_input_sha256"],
        )
        self.assertEqual(
            final_job["uploaded_identity_source_sha256"],
            upload_digests["uploaded_identity_source_sha256"],
        )
        self.assertEqual(
            final_job["canonical_qwen_image1_path"],
            str(canonical_qwen_image1),
        )
        self.assertEqual(
            final_job["canonical_qwen_image1_sha256"],
            canonical_qwen_digest,
        )
        self.assertEqual(
            final_job["canonical_qwen_image1_comfy_path"],
            str(comfy_input / "local_video_api" / job_id / "cartoon_reference.png"),
        )
        self.assertEqual(
            final_job["canonical_qwen_image1_comfy_sha256"],
            final_job["canonical_qwen_image1_sha256"],
        )
        self.assertEqual(final_job["surface_texture_gate_status"], "passed")
        self.assertEqual(
            set(final_job["subject_fur_surface_quality"]),
            {"eating", "grooming"},
        )
        self.assertEqual(
            final_job["reference_image_profile"], "pet-multipose-20260717-v2"
        )
        self.assertEqual(
            [
                (item["original_filename"], item["strength"])
                for item in final_job["user_loras"]
            ],
            [
                ("宠物小猫 _ 写实摄影_v3.0.safetensors", 0.5),
                ("Qwen Image-Kitten Photography_QWEN-小猫摄影.safetensors", 0.6),
            ],
        )
        self.assertEqual(final_job["qwen_steps"], 6)
        self.assertEqual(final_job["qwen_cfg"], 1.0)
        self.assertEqual(final_job["qwen_denoise"], 1.0)
        self.assertEqual(
            final_job["seed"],
            pipeline.IMAGE_PROFILES["pet-multipose-20260717-v2"]["seed"],
        )
        self.assertFalse(final_job["lock_canonical_retry_lora_strengths"])
        self.assertFalse(final_job["lock_pose_retry_lora_strengths"])
        self.assertEqual(
            final_job["canonical_user_lora_total_cap"],
            pipeline.QWEN_SUBJECT_REPLACEMENT_CANONICAL_LORA_TOTAL_CAP,
        )
        self.assertEqual(
            final_job["pose_user_lora_first_total_cap"],
            pipeline.QWEN_SUBJECT_REPLACEMENT_POSE_LORA_TOTAL_CAP,
        )
        self.assertEqual(
            final_job["pose_reference_denoise"],
            pipeline.QWEN_SUBJECT_REPLACEMENT_POSE_DENOISE,
        )
        self.assertEqual(
            final_job["pose_reference_denoise_step"],
            pipeline.QWEN_SUBJECT_REPLACEMENT_POSE_DENOISE_STEP,
        )
        self.assertEqual(
            final_job["pose_reference_inputs"],
            {
                "eating": (
                    "local_video_api/test-qwen-reference/eating_qwen_pose_guide.png"
                ),
                "grooming": (
                    "local_video_api/test-qwen-reference/grooming_qwen_pose_guide.png"
                ),
            },
        )
        self.assertEqual(
            final_job["reference_pose_source"],
            "per_action_main_case_driver_geometry",
        )
        self.assertEqual(
            final_job["pose_reference_policy"],
            {
                "eating": ("main_case_eating_driver_geometry_image2_guide_v4"),
                "grooming": pipeline.GROOMING_QWEN_POSE_REFERENCE_POLICY,
            },
        )
        self.assertEqual(
            final_job["pose_identity_input_policy"],
            pipeline.POSE_IDENTITY_INPUT_POLICY_CANONICAL,
        )
        self.assertEqual(
            final_job["pose_identity_input_policies"],
            {
                "eating": pipeline.POSE_IDENTITY_INPUT_POLICY_CANONICAL,
                "grooming": pipeline.POSE_IDENTITY_INPUT_POLICY_CANONICAL,
            },
        )
        self.assertEqual(
            final_job["pose_identity_input_roles"],
            {
                "eating": "image1_qwen_full_body_canonical",
                "grooming": "image1_qwen_full_body_canonical",
            },
        )
        self.assertEqual(
            final_job["pose_identity_geometry_roles"],
            {
                "eating": "image2_main_case_driver_geometry_guide",
                "grooming": "image2_main_case_driver_geometry_guide",
            },
        )
        self.assertEqual(
            final_job["eating_pose_lora_policy"],
            "canonical_identity_plus_main_case_geometry_v1",
        )
        self.assertTrue(final_job["eating_identity_refine_enabled"])
        self.assertFalse(final_job["eating_identity_refine_active"])
        self.assertEqual(
            final_job["eating_identity_refine_policy"],
            pipeline.EATING_CANONICAL_IDENTITY_REFINE_POLICY,
        )
        self.assertEqual(
            final_job["eating_identity_refine_attempt"],
            pipeline.EATING_CANONICAL_IDENTITY_REFINE_ATTEMPT,
        )
        self.assertEqual(
            final_job["eating_identity_refine_denoise"],
            pipeline.EATING_CANONICAL_IDENTITY_REFINE_DENOISE,
        )
        self.assertIsNone(final_job["eating_identity_refine_style_lora"])
        self.assertFalse(final_job["eating_identity_refine_global_thresholds_changed"])
        self.assertEqual(
            final_job["pose_reference_source_paths"],
            {
                "eating": profile["runtime"]["actions"]["eating"]["driving_animation"],
                "grooming": profile["runtime"]["actions"]["grooming"][
                    "driving_animation"
                ],
            },
        )
        self.assertEqual(
            guide_sources,
            {
                "eating": Path(
                    profile["runtime"]["actions"]["eating"]["driving_animation"]
                ),
                "grooming": Path(
                    profile["runtime"]["actions"]["grooming"]["driving_animation"]
                ),
            },
        )
        self.assertEqual(
            final_job["pose_reference_guide_style"],
            "per_action_main_case_driver_geometry_v2",
        )
        self.assertEqual(
            final_job["pose_reference_guide_styles"],
            {
                "eating": (
                    "texture_free_low_frequency_white_maquette_geometry_with_explicit_"
                    "muzzle_overlap_on_pure_green"
                ),
                "grooming": (pipeline.GROOMING_QWEN_GUIDE_STYLE),
            },
        )
        self.assertEqual(
            final_job["pose_reference_role"],
            "per_action_geometry_input_role",
        )
        eating_metadata = final_job["pose_reference_metadata"]["eating"]
        self.assertEqual(
            eating_metadata["policy"],
            "main_case_eating_driver_geometry_image2_guide_v4",
        )
        self.assertEqual(
            eating_metadata["image_role"],
            "image2_action_geometry_guide",
        )
        self.assertEqual(
            eating_metadata["identity_role"],
            "image1_qwen_full_body_canonical",
        )
        self.assertFalse(eating_metadata["source_rgb_preserved"])
        self.assertTrue(eating_metadata["driver_silhouette_used"])
        self.assertFalse(eating_metadata["reference_space_motion"])
        self.assertFalse(eating_metadata["main_case_driver_used"])
        self.assertTrue(eating_metadata["uploaded_white_maquette_driver_used"])
        self.assertEqual(
            eating_metadata["palette_audit_role"],
            "qwen_full_body_canonical",
        )
        self.assertEqual(
            eating_metadata["source_sha256"],
            eating_driver_sha256,
        )
        grooming_metadata = final_job["pose_reference_metadata"]["grooming"]
        self.assertEqual(
            grooming_metadata["policy"],
            pipeline.GROOMING_QWEN_POSE_REFERENCE_POLICY,
        )
        self.assertEqual(
            grooming_metadata["generator_policy"],
            pipeline.GROOMING_QWEN_GEOMETRY_GUIDE_POLICY,
        )
        self.assertEqual(
            grooming_metadata["image_role"],
            "image2_action_geometry_guide",
        )
        self.assertEqual(
            grooming_metadata["identity_role"],
            "image1_qwen_full_body_canonical",
        )
        self.assertTrue(grooming_metadata["qwen_guide_structural_negative_space"])
        self.assertEqual(
            grooming_metadata["qwen_guide_structural_negative_space_method"],
            pipeline.GROOMING_QWEN_NEGATIVE_SPACE_METHOD,
        )
        self.assertFalse(grooming_metadata["main_case_appearance_exposed_to_qwen"])
        self.assertTrue(
            grooming_metadata["uploaded_white_maquette_driver_used"]
        )
        self.assertFalse(grooming_metadata["qwen_guide_internal_luma_variation"])
        self.assertEqual(
            grooming_metadata["qwen_guide_internal_luma_variation_role"],
            "none",
        )
        self.assertFalse(
            grooming_metadata["qwen_guide_appearance_bearing_internal_luma_variation"]
        )
        self.assertFalse(grooming_metadata["qwen_guide_synthetic_foreleg_semantic"])
        self.assertEqual(
            grooming_metadata["qwen_guide_synthetic_foreleg_semantic_method"],
            None,
        )
        self.assertFalse(grooming_metadata["qwen_guide_closed_foreleg_capsule_exposed"])
        self.assertTrue(
            grooming_metadata["qwen_guide_internal_green_separator_exposed"]
        )
        self.assertEqual(
            grooming_metadata["qwen_guide_raised_foreleg_validation_role"],
            "public_open_curve_geometry_plus_private_cv",
        )
        self.assertEqual(
            grooming_metadata["qwen_foreleg_repair_policy"],
            pipeline.GROOMING_CANONICAL_FORELEG_COMPLETION_POLICY,
        )
        self.assertEqual(
            grooming_metadata["qwen_foreleg_repair_attempt"],
            pipeline.GROOMING_QWEN_FORELEG_REPAIR_ATTEMPT,
        )
        self.assertEqual(
            grooming_metadata["qwen_foreleg_repair_denoise"],
            pipeline.GROOMING_QWEN_FORELEG_REPAIR_DENOISE,
        )
        self.assertFalse(
            grooming_metadata["qwen_foreleg_repair_synthetic_capsule_exposed"]
        )
        self.assertTrue(
            grooming_metadata["qwen_foreleg_repair_open_channel_anatomy_image3"]
        )
        self.assertFalse(
            grooming_metadata["qwen_foreleg_repair_private_cv_luma_image3"]
        )
        self.assertEqual(
            grooming_metadata["qwen_foreleg_repair_mask_dilation_px"],
            pipeline.GROOMING_QWEN_FORELEG_REPAIR_MASK_DILATION_PX,
        )
        self.assertEqual(
            grooming_metadata["qwen_foreleg_repair_seed_source"],
            "identity_safe_candidate_unchanged_before_masked_sampling",
        )
        self.assertFalse(
            grooming_metadata["qwen_foreleg_repair_identity_or_geometry_seed_inserted"]
        )
        self.assertTrue(
            grooming_metadata["qwen_foreleg_repair_expanded_context_full_denoise_mask"]
        )
        self.assertEqual(
            grooming_metadata["qwen_foreleg_repair_style_policy"],
            "disabled_for_canonical_masked_completion",
        )
        self.assertIsNone(grooming_metadata["qwen_foreleg_repair_style_lora"])
        self.assertEqual(
            grooming_metadata["qwen_foreleg_repair_style_strength"],
            0.0,
        )
        self.assertEqual(
            grooming_metadata["qwen_foreleg_repair_style_scope"],
            "none",
        )
        self.assertIsNone(
            grooming_metadata["qwen_foreleg_repair_composite_gate_policy"]
        )
        self.assertFalse(
            grooming_metadata["qwen_foreleg_repair_composite_recovery_enabled"]
        )
        self.assertIsNone(
            grooming_metadata["qwen_foreleg_repair_composite_edge_density_ratio_min"]
        )
        self.assertIsNone(
            grooming_metadata[
                "qwen_foreleg_repair_composite_planted_paw_gradient_p90_max"
            ]
        )
        self.assertIsNone(
            grooming_metadata[
                "qwen_foreleg_repair_composite_raised_paw_gradient_p90_max"
            ]
        )
        self.assertEqual(
            grooming_metadata["canonical_completion_edge_evidence_policy"],
            pipeline.GROOMING_CANONICAL_COMPLETION_EDGE_EVIDENCE_POLICY,
        )
        self.assertEqual(
            grooming_metadata["canonical_completion_edge_density_ratio_min"],
            pipeline.GROOMING_CANONICAL_COMPLETION_EDGE_DENSITY_RATIO_MIN,
        )
        self.assertTrue(
            grooming_metadata["canonical_completion_independent_fur_surface_required"]
        )
        direct_evidence = grooming_metadata["canonical_direct_structural_edge_evidence"]
        self.assertEqual(
            direct_evidence["policy"],
            pipeline.GROOMING_CANONICAL_DIRECT_STRUCTURAL_EDGE_EVIDENCE_POLICY,
        )
        self.assertEqual(direct_evidence["maximum_attempt"], 2)
        self.assertEqual(direct_evidence["raw_edge_density_ratio_min"], 0.925)
        self.assertEqual(direct_evidence["structural_edge_density_min"], 0.03)
        self.assertEqual(direct_evidence["chamfer_coverage_min"], 0.70)
        self.assertEqual(direct_evidence["chamfer_mean_distance_max"], 5.0)
        self.assertEqual(direct_evidence["chamfer_p90_distance_max"], 12.0)
        self.assertTrue(direct_evidence["identity_palette_gate_still_required"])
        self.assertTrue(direct_evidence["independent_fur_surface_gate_still_required"])
        self.assertFalse(direct_evidence["global_pose_threshold_changed"])
        self.assertTrue(direct_evidence["fail_closed"])
        surface_nonexpansion = grooming_metadata[
            "canonical_direct_surface_nonexpansion"
        ]
        self.assertEqual(
            surface_nonexpansion["policy"],
            pipeline.GROOMING_CANONICAL_DIRECT_SURFACE_NONEXPANSION_POLICY,
        )
        self.assertEqual(surface_nonexpansion["maximum_attempt"], 2)
        self.assertEqual(
            surface_nonexpansion["only_recoverable_issue"],
            "brush_texture_coherence",
        )
        self.assertTrue(surface_nonexpansion["requires_strict_pose_recovery"])
        self.assertTrue(surface_nonexpansion["requires_identity_palette_pass"])
        self.assertTrue(
            surface_nonexpansion["requires_digest_bound_transparent_canonical"]
        )
        self.assertTrue(
            surface_nonexpansion["independent_final_video_fur_audit_still_required"]
        )
        self.assertFalse(surface_nonexpansion["global_surface_thresholds_changed"])
        self.assertTrue(surface_nonexpansion["fail_closed"])
        self.assertTrue(grooming_metadata["canonical_full_pose_recovery_enabled"])
        self.assertEqual(
            grooming_metadata["canonical_full_pose_rescue_policy"],
            pipeline.GROOMING_CANONICAL_FULL_POSE_RESCUE_POLICY,
        )
        self.assertEqual(
            grooming_metadata["canonical_full_pose_rescue_attempt"],
            pipeline.GROOMING_CANONICAL_FULL_POSE_RESCUE_ATTEMPT,
        )
        self.assertEqual(
            grooming_metadata["canonical_full_pose_identity_refine_policy"],
            pipeline.GROOMING_CANONICAL_FULL_POSE_REFINE_POLICY,
        )
        self.assertEqual(
            grooming_metadata["canonical_full_pose_identity_refine_attempt"],
            pipeline.GROOMING_CANONICAL_FULL_POSE_REFINE_ATTEMPT,
        )
        self.assertEqual(
            grooming_metadata["canonical_full_pose_refine_source_policy"],
            pipeline.GROOMING_CANONICAL_POSE_SAFE_REFINE_SOURCE_POLICY,
        )
        self.assertEqual(
            grooming_metadata["canonical_full_pose_refine_image3"],
            "digest_bound_soft_foreleg_depth_action_corroboration",
        )
        self.assertFalse(
            grooming_metadata["canonical_full_pose_refine_uploaded_front_exposed"]
        )
        self.assertEqual(
            grooming_metadata["canonical_full_pose_refine_identity_source"],
            "approved_canonical_three_view_consensus_only",
        )
        self.assertFalse(
            grooming_metadata[
                "canonical_pose_transfer_source_alpha_geometry_preserved"
            ]
        )
        self.assertFalse(
            grooming_metadata[
                "canonical_pose_transfer_action_alpha_overlap_blocking"
            ]
        )
        self.assertTrue(
            grooming_metadata[
                "canonical_pose_transfer_public_geometry_guide_required"
            ]
        )
        self.assertFalse(
            grooming_metadata["canonical_pose_transfer_source_rgb_exposed"]
        )
        self.assertFalse(
            grooming_metadata["canonical_pose_transfer_source_chroma_exposed"]
        )
        self.assertFalse(
            grooming_metadata["canonical_pose_transfer_source_raw_luma_exposed"]
        )
        self.assertTrue(
            grooming_metadata[
                "canonical_pose_transfer_rejected_candidate_used_as_sampler"
            ]
        )
        self.assertEqual(
            grooming_metadata["canonical_pose_transfer_seed_policy"],
            pipeline.GROOMING_CANONICAL_POSE_TRANSFER_SEED_POLICY,
        )
        self.assertEqual(
            grooming_metadata["canonical_pose_transfer_seed_bound_matte_policy"],
            pipeline.GROOMING_CANONICAL_SEED_BOUND_MATTE_POLICY,
        )
        self.assertFalse(
            grooming_metadata["canonical_full_pose_recovery_global_thresholds_changed"]
        )
        self.assertEqual(
            grooming_metadata["qwen_foreleg_repair_global_edge_density_min"],
            0.11,
        )
        self.assertFalse(
            grooming_metadata["qwen_foreleg_repair_global_pose_threshold_changed"]
        )
        self.assertFalse(grooming_metadata["private_pose_gate_exposed_to_qwen"])
        self.assertTrue(grooming_metadata["qwen_foreleg_repair_enabled"])
        self.assertFalse(
            grooming_metadata[
                "private_pose_gate_neutral_copy_exposed_on_repair_attempt"
            ]
        )
        self.assertTrue(
            grooming_metadata["public_open_channel_copy_exposed_on_repair_attempt"]
        )
        self.assertFalse(grooming_metadata["private_completion_mask_exposed_to_qwen"])
        self.assertRegex(
            grooming_metadata["private_completion_mask_sha256"],
            r"^[0-9a-f]{64}$",
        )
        self.assertRegex(
            grooming_metadata["private_pose_gate_guide_sha256"],
            r"^[0-9a-f]{64}$",
        )
        self.assertEqual(
            final_job["grooming_pose_lora_policy"],
            "canonical_identity_open_channel_seed_bound_pose_safe_refine_v11",
        )
        self.assertFalse(final_job["grooming_foreleg_repair_active"])
        self.assertEqual(
            final_job["grooming_foreleg_repair_policy"],
            pipeline.GROOMING_CANONICAL_FORELEG_COMPLETION_POLICY,
        )
        self.assertEqual(
            final_job["grooming_foreleg_repair_style_policy"],
            "disabled_for_canonical_masked_completion",
        )
        self.assertEqual(
            final_job["grooming_foreleg_repair_style_strength"],
            0.0,
        )
        self.assertIsNone(final_job["grooming_foreleg_repair_composite_gate_policy"])
        self.assertIsNone(
            final_job["grooming_foreleg_repair_composite_edge_density_ratio_min"]
        )
        self.assertIsNone(
            final_job["grooming_foreleg_repair_composite_planted_paw_gradient_p90_max"]
        )
        self.assertIsNone(
            final_job["grooming_foreleg_repair_composite_raised_paw_gradient_p90_max"]
        )
        self.assertEqual(
            final_job["grooming_canonical_completion_edge_evidence_policy"],
            pipeline.GROOMING_CANONICAL_COMPLETION_EDGE_EVIDENCE_POLICY,
        )
        self.assertEqual(
            final_job["grooming_canonical_completion_edge_density_ratio_min"],
            pipeline.GROOMING_CANONICAL_COMPLETION_EDGE_DENSITY_RATIO_MIN,
        )
        self.assertTrue(
            final_job["grooming_canonical_completion_independent_fur_surface_required"]
        )
        self.assertEqual(
            final_job["grooming_canonical_direct_structural_edge_evidence_policy"],
            pipeline.GROOMING_CANONICAL_DIRECT_STRUCTURAL_EDGE_EVIDENCE_POLICY,
        )
        self.assertEqual(
            final_job["grooming_canonical_direct_edge_density_ratio_min"],
            0.925,
        )
        self.assertEqual(
            final_job["grooming_canonical_direct_structural_edge_density_min"],
            0.03,
        )
        self.assertEqual(
            final_job["grooming_canonical_direct_chamfer_coverage_min"],
            0.70,
        )
        self.assertEqual(
            final_job["grooming_canonical_direct_chamfer_mean_distance_max"],
            5.0,
        )
        self.assertEqual(
            final_job["grooming_canonical_direct_chamfer_p90_distance_max"],
            12.0,
        )
        self.assertEqual(
            final_job["grooming_canonical_direct_edge_evidence_max_attempt"],
            2,
        )
        self.assertTrue(
            final_job["grooming_canonical_direct_identity_palette_required"]
        )
        self.assertTrue(
            final_job["grooming_canonical_direct_independent_fur_surface_required"]
        )
        self.assertFalse(
            final_job["grooming_canonical_direct_global_thresholds_changed"]
        )
        self.assertEqual(
            final_job["grooming_canonical_direct_surface_nonexpansion_policy"],
            pipeline.GROOMING_CANONICAL_DIRECT_SURFACE_NONEXPANSION_POLICY,
        )
        self.assertEqual(
            final_job["grooming_canonical_direct_surface_only_recoverable_issue"],
            "brush_texture_coherence",
        )
        self.assertEqual(
            final_job["grooming_canonical_direct_surface_coherent_fraction_max"],
            0.57,
        )
        self.assertEqual(
            final_job["grooming_canonical_direct_surface_coherent_fraction_delta_max"],
            0.01,
        )
        self.assertTrue(
            final_job["grooming_canonical_direct_surface_requires_pose_and_identity"]
        )
        self.assertTrue(
            final_job["grooming_canonical_direct_surface_requires_canonical_digest"]
        )
        self.assertTrue(
            final_job["grooming_canonical_direct_surface_final_video_audit_required"]
        )
        self.assertFalse(
            final_job["grooming_canonical_direct_surface_global_thresholds_changed"]
        )
        self.assertTrue(final_job["grooming_full_pose_recovery_enabled"])
        self.assertFalse(final_job["grooming_full_pose_recovery_active"])
        self.assertEqual(
            final_job["grooming_full_pose_rescue_attempt"],
            pipeline.GROOMING_CANONICAL_FULL_POSE_RESCUE_ATTEMPT,
        )
        self.assertEqual(
            final_job["grooming_full_pose_identity_refine_attempt"],
            pipeline.GROOMING_CANONICAL_FULL_POSE_REFINE_ATTEMPT,
        )
        self.assertIsNone(final_job["grooming_full_pose_recovery_style_lora"])
        self.assertFalse(
            final_job["grooming_full_pose_recovery_global_thresholds_changed"]
        )
        self.assertEqual(
            final_job["grooming_foreleg_repair_global_edge_density_min"],
            0.11,
        )
        self.assertFalse(
            final_job["grooming_foreleg_repair_global_pose_threshold_changed"]
        )
        self.assertIsNone(final_job["grooming_pose_gate_composite_recovery"])
        self.assertEqual(
            final_job["canonical_prompt_source"],
            "uploaded_identity_qwen_full_body",
        )
        self.assertTrue(final_job["user_prompt_excluded_from_reference"])
        self.assertNotIn(
            "fluffy long soft fur",
            final_job["qwen_reference_prompt_en"],
        )
        self.assertIn(
            "uploaded pet view or role-labeled view set is the only authority",
            final_job["subject_profile_fur_lock"],
        )
        self.assertNotIn(
            "compact short-to-medium coat",
            final_job["subject_profile_fur_lock"],
        )
        self.assertEqual(
            final_job["subject_profile_appearance_role"],
            "action_geometry_only",
        )
        self.assertFalse(final_job["subject_profile_identity_terms_applied"])
        self.assertIn("eating", final_job["pose_reference_source_sha256s"])

    def test_uploaded_qwen_rejects_changed_identity_source_before_generation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_root = root / "outputs"
            job_id = "test-changed-upload-source"
            job_dir = output_root / job_id
            job_dir.mkdir(parents=True)
            upload_digests = _write_test_upload_identity(job_dir)
            profile = _subject_profile_with_action_references(root, ("eating",))
            effective_config, expected_pose_policies = _configured_test_upload_profile(
                profile, ["eating"]
            )
            jobs = {
                job_id: {
                    "job_id": job_id,
                    "generation_mode": "subject_replacement",
                    "subject_source": "upload_pending_qwen",
                    "outputs": [],
                    "cancel_requested": False,
                    "effective_subject_config": effective_config,
                    "expected_pose_policies": expected_pose_policies,
                    **upload_digests,
                }
            }
            (job_dir / "input_identity_source.png").write_bytes(b"changed")
            with (
                patch.multiple(
                    pipeline,
                    OUTPUT_ROOT=output_root,
                    COMFY_INPUT=root / "comfy-input",
                    JOBS=jobs,
                ),
                patch.object(
                    pipeline,
                    "prepare_eating_qwen_geometry_guide",
                ) as prepare_guide,
                patch.object(pipeline, "run_job") as run_qwen,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "identity source is missing or changed after submission",
                ):
                    pipeline.prepare_uploaded_subject_qwen_references(
                        job_id,
                        ["eating"],
                    )

            prepare_guide.assert_not_called()
            run_qwen.assert_not_called()

    def test_uploaded_qwen_excludes_frozen_profile_identity_constraints(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_root = root / "outputs"
            comfy_input = root / "comfy-input"
            job_id = "test-frozen-fur-constraints"
            job_dir = output_root / job_id
            job_dir.mkdir(parents=True)
            upload_digests = _write_test_upload_identity(job_dir)
            profile = _subject_profile_with_action_references(root, ("eating",))
            effective_config, expected_pose_policies = _configured_test_upload_profile(
                profile, ["eating"]
            )
            effective_config["profile_fur_constraints"] = {
                "positive": "frozen queue-time close coat rule",
                "negative": "frozen queue-time expanded coat ban",
            }
            jobs = {
                job_id: {
                    "job_id": job_id,
                    "generation_mode": "subject_replacement",
                    "subject_source": "upload_pending_qwen",
                    "subject_profile": "sleek_tabby_cat_v1",
                    "prompt": "",
                    "prompt_en": "",
                    "outputs": [],
                    "cancel_requested": False,
                    "effective_subject_config": effective_config,
                    "expected_pose_policies": expected_pose_policies,
                    **upload_digests,
                },
            }
            profile_loras = [
                dict(item)
                for item in pipeline.IMAGE_PROFILES[
                    pipeline.SUBJECT_REPLACEMENT_REFERENCE_IMAGE_PROFILE
                ]["user_loras"]
            ]

            def fake_qwen(
                _target_job_id: str,
                *,
                force_qwen_reference: bool = False,
            ) -> None:
                self.assertTrue(force_qwen_reference)
                canonical = job_dir / "cartoon_reference_transparent.png"
                canonical_image = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
                canonical_image.paste((80, 80, 80, 255), (128, 64, 384, 448))
                canonical_image.save(canonical)
                canonical_green = Image.new("RGB", (512, 512), (0, 255, 0))
                canonical_green.paste((80, 80, 80), (128, 64, 384, 448))
                canonical_green.save(job_dir / "cartoon_reference.png")
                _record_fake_canonical_image1(job_id, job_dir, comfy_input)
                pipeline.append_output(
                    job_id,
                    {
                        "type": "cartoon_reference",
                        "filename": canonical.name,
                        "transparent": True,
                    },
                )
                action = job_dir / "eating_transparent.png"
                image = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
                image.paste((80, 80, 80, 255), (128, 96, 384, 416))
                image.save(action)
                pipeline.append_output(
                    job_id,
                    {
                        "type": "transparent_keyframe",
                        "action": "eating",
                        "filename": action.name,
                    },
                )
                pipeline.update_job(
                    job_id,
                    fur_surface_gate_selected_metrics={
                        "eating": _passed_surface_metric(action, "eating")
                    },
                )

            with (
                patch.multiple(
                    pipeline,
                    OUTPUT_ROOT=output_root,
                    COMFY_INPUT=comfy_input,
                    JOBS=jobs,
                ),
                patch.object(
                    pipeline,
                    "merge_image_profile_loras",
                    return_value=(profile_loras, "replace", "image_profile"),
                ),
                patch.object(
                    pipeline,
                    "load_subject_profile",
                    side_effect=AssertionError("live profile must not be re-read"),
                ),
                patch.object(
                    pipeline,
                    "prepare_eating_qwen_geometry_guide",
                    side_effect=_write_fake_geometry_guide,
                ),
                patch.object(
                    pipeline,
                    "evaluate_main_case_fur_surface_candidate",
                    return_value={
                        "acceptable": True,
                        "status": "passed",
                        "issues": [],
                    },
                ),
                patch.object(pipeline, "run_job", side_effect=fake_qwen),
            ):
                pipeline.prepare_uploaded_subject_qwen_references(
                    job_id,
                    ["eating"],
                )

            final_job = jobs[job_id]

        self.assertIn(
            "uploaded pet view or role-labeled view set is the only authority",
            final_job["subject_profile_fur_lock"],
        )
        self.assertNotIn(
            "frozen queue-time close coat rule",
            final_job["subject_profile_fur_lock"],
        )
        self.assertNotIn(
            "frozen queue-time expanded coat ban",
            final_job["subject_profile_fur_negative"],
        )

    def test_uploaded_qwen_stage_fails_closed_on_opaque_action_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_root = root / "outputs"
            comfy_input = root / "comfy-input"
            job_id = "test-opaque-reference"
            job_dir = output_root / job_id
            job_dir.mkdir(parents=True)
            upload_digests = _write_test_upload_identity(job_dir)
            profile = _subject_profile_with_action_references(root, ("eating",))
            effective_config, expected_pose_policies = _configured_test_upload_profile(
                profile, ["eating"]
            )
            jobs = {
                job_id: {
                    "job_id": job_id,
                    "generation_mode": "subject_replacement",
                    "subject_source": "upload_pending_qwen",
                    "prompt": "",
                    "prompt_en": "",
                    "outputs": [],
                    "cancel_requested": False,
                    "effective_subject_config": effective_config,
                    "expected_pose_policies": expected_pose_policies,
                    **upload_digests,
                }
            }
            profile_loras = [
                dict(item)
                for item in pipeline.IMAGE_PROFILES[
                    pipeline.SUBJECT_REPLACEMENT_REFERENCE_IMAGE_PROFILE
                ]["user_loras"]
            ]

            def fake_opaque_qwen(
                _target_job_id: str,
                *,
                force_qwen_reference: bool = False,
            ) -> None:
                self.assertTrue(force_qwen_reference)
                canonical = job_dir / "cartoon_reference_transparent.png"
                canonical_image = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
                canonical_image.paste((80, 80, 80, 255), (128, 64, 384, 448))
                canonical_image.save(canonical)
                canonical_green = Image.new("RGB", (512, 512), (0, 255, 0))
                canonical_green.paste((80, 80, 80), (128, 64, 384, 448))
                canonical_green.save(job_dir / "cartoon_reference.png")
                _record_fake_canonical_image1(job_id, job_dir, comfy_input)
                pipeline.append_output(
                    job_id,
                    {
                        "type": "cartoon_reference",
                        "filename": canonical.name,
                        "transparent": True,
                    },
                )
                opaque = job_dir / "eating_opaque.png"
                Image.new("RGB", (512, 512), "white").save(opaque)
                pipeline.append_output(
                    job_id,
                    {
                        "type": "transparent_keyframe",
                        "action": "eating",
                        "filename": opaque.name,
                    },
                )

            with (
                patch.multiple(
                    pipeline,
                    OUTPUT_ROOT=output_root,
                    COMFY_INPUT=comfy_input,
                    JOBS=jobs,
                ),
                patch.object(
                    pipeline,
                    "merge_image_profile_loras",
                    return_value=(profile_loras, "replace", "image_profile"),
                ),
                patch.object(
                    pipeline,
                    "load_subject_profile",
                    side_effect=AssertionError("live profile must not be re-read"),
                ),
                patch.object(
                    pipeline,
                    "prepare_eating_qwen_geometry_guide",
                    side_effect=_write_fake_geometry_guide,
                ),
                patch.object(pipeline, "run_job", side_effect=fake_opaque_qwen),
            ):
                with self.assertRaisesRegex(RuntimeError, "opaque"):
                    pipeline.prepare_uploaded_subject_qwen_references(
                        job_id,
                        ["eating"],
                    )

    def test_uploaded_grooming_transparent_selects_validated_action_scope(self) -> None:
        scope = pipeline.select_final_video_fur_surface_comparison_scope(
            action="grooming",
            action_config={"reference_kind": "qwen_generated_upload"},
            output_format="transparent_gif",
            reference_policy="qwen_action_specific_identity",
            pose_metadata={"scene_prop_separated": False},
        )
        self.assertEqual(
            scope,
            pipeline.FINAL_VIDEO_VALIDATED_ACTION_SURFACE_SCOPE,
        )

    def test_uploaded_grooming_transparent_requires_no_prop_contract(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "no-prop"):
            pipeline.select_final_video_fur_surface_comparison_scope(
                action="grooming",
                action_config={"reference_kind": "qwen_generated_upload"},
                output_format="transparent_gif",
                reference_policy="qwen_action_specific_identity",
                pose_metadata={"scene_prop_separated": True},
            )

    def test_uploaded_eating_transparent_keeps_separated_prop_contract(self) -> None:
        scope = pipeline.select_final_video_fur_surface_comparison_scope(
            action="eating",
            action_config={
                "reference_kind": "qwen_generated_upload",
                "fur_silhouette_gate": {"enabled": True},
            },
            output_format="transparent_gif",
            reference_policy="qwen_action_identity_with_bowl_mapped_red",
            pose_metadata={"scene_prop_separated": True},
        )
        self.assertEqual(
            scope,
            pipeline.FINAL_VIDEO_VALIDATED_ACTION_SURFACE_SCOPE,
        )

    def test_final_video_v2_forwards_exact_qwen_and_mask_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "video.mp4"
            profile = root / "profile.png"
            reference = root / "pose_01_keyframe_transparent.png"
            pose_mask = root / "tm_identity_mask.mp4"
            reference_mask = root / "reference_identity_mask.png"
            video.write_bytes(b"video")
            pose_mask.write_bytes(b"pose-mask")
            Image.new("RGBA", (32, 32), (80, 80, 80, 255)).save(profile)
            Image.new("RGBA", (32, 32), (90, 90, 90, 255)).save(reference)
            Image.new("RGB", (32, 32), (0, 0, 0)).save(reference_mask)
            reference_sha = pipeline.file_sha256(reference)
            pose_mask_sha = pipeline.file_sha256(pose_mask)
            reference_mask_sha = pipeline.file_sha256(reference_mask)
            selected_gate = {
                "acceptable": True,
                "verdict": "accepted_uploaded_identity_absolute_contract",
                "selected_candidate_sha256": reference_sha,
                "policy": {
                    "name": "validated_action_reference_fur_surface_v1",
                    "gate_version": pipeline.FUR_SURFACE_GATE_VERSION,
                    "threshold_version": pipeline.FUR_SURFACE_THRESHOLD_VERSION,
                    "validated_baseline_required_for_acceptance": False,
                    "uploaded_identity_absolute_contract_enabled": True,
                },
                "gate": {
                    "action": "eating",
                    "candidate_sha256": reference_sha,
                    "candidate_comparison_mode": (
                        "uploaded_identity_absolute_surface_contract_v1"
                    ),
                    "profile_pixels_used_for_candidate_appearance_validation": False,
                },
            }
            effective_config = {
                "actions": {
                    "eating": {
                        "reference": str(reference),
                        "reference_sha256": reference_sha,
                        "reference_kind": "qwen_generated_upload",
                    }
                }
            }
            audit = Mock(return_value={"passed": True, "policy": {}})
            with (
                patch.object(
                    pipeline,
                    "resolve_main_case_fur_surface_baseline",
                    return_value=(
                        profile,
                        {
                            "snapshot_sha256": pipeline.file_sha256(profile),
                            "provenance": {"profile_id": "test"},
                        },
                    ),
                ),
                patch.object(pipeline, "audit_final_video", audit),
            ):
                result = pipeline.evaluate_final_video_fur_surface(
                    video,
                    action="eating",
                    effective_subject_config=effective_config,
                    comparison_scope=(
                        pipeline.FINAL_VIDEO_VALIDATED_ACTION_SURFACE_SCOPE
                    ),
                    validated_action_reference_path=reference,
                    validated_action_reference_sha256=reference_sha,
                    validated_action_reference_gate=selected_gate,
                    reference_policy="qwen_action_identity_with_bowl_mapped_red",
                    pose_identity_mask_video_path=pose_mask,
                    pose_identity_mask_video_sha256=pose_mask_sha,
                    reference_identity_mask_path=reference_mask,
                    reference_identity_mask_sha256=reference_mask_sha,
                )
            self.assertTrue(result["passed"])
            forwarded = audit.call_args.kwargs
            self.assertEqual(
                forwarded["comparison_scope"],
                pipeline.FINAL_VIDEO_VALIDATED_ACTION_SURFACE_SCOPE,
            )
            self.assertEqual(forwarded["validated_action_reference_path"], reference)
            self.assertEqual(
                forwarded["validated_action_reference_sha256"], reference_sha
            )
            self.assertEqual(
                forwarded["pose_identity_mask_video_sha256"], pose_mask_sha
            )
            self.assertEqual(
                forwarded["reference_identity_mask_sha256"], reference_mask_sha
            )
            self.assertTrue(
                forwarded["validated_action_reference_provenance"][
                    "pre_scail_revalidated"
                ]
            )

    def test_final_video_v2_accepts_grooming_qwen_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "video.mp4"
            profile = root / "profile.png"
            reference = root / "pose_01_keyframe_transparent.png"
            pose_mask = root / "tm_identity_mask.mp4"
            reference_mask = root / "reference_identity_mask.png"
            video.write_bytes(b"video")
            pose_mask.write_bytes(b"pose-mask")
            Image.new("RGBA", (32, 32), (80, 80, 80, 255)).save(profile)
            Image.new("RGBA", (32, 32), (90, 90, 90, 255)).save(reference)
            Image.new("RGB", (32, 32), (0, 0, 0)).save(reference_mask)
            reference_sha = pipeline.file_sha256(reference)
            pose_mask_sha = pipeline.file_sha256(pose_mask)
            reference_mask_sha = pipeline.file_sha256(reference_mask)
            selected_gate = {
                "acceptable": True,
                "verdict": "accepted_uploaded_identity_absolute_contract",
                "selected_candidate_sha256": reference_sha,
                "policy": {
                    "name": "validated_action_reference_fur_surface_v1",
                    "gate_version": pipeline.FUR_SURFACE_GATE_VERSION,
                    "threshold_version": pipeline.FUR_SURFACE_THRESHOLD_VERSION,
                    "validated_baseline_required_for_acceptance": False,
                    "uploaded_identity_absolute_contract_enabled": True,
                },
                "gate": {
                    "action": "grooming",
                    "candidate_sha256": reference_sha,
                    "candidate_comparison_mode": (
                        "uploaded_identity_absolute_surface_contract_v1"
                    ),
                    "profile_pixels_used_for_candidate_appearance_validation": False,
                },
            }
            composite_recovery = {
                "policy": (pipeline.GROOMING_QWEN_FORELEG_REPAIR_COMPOSITE_GATE_POLICY),
                "evaluated": True,
                "accepted": True,
                "fail_closed": True,
                "global_pose_threshold_changed": False,
                "required_edge_density_ratio": 0.70,
                "edge_density": 0.079399,
                "edge_density_threshold": 0.11,
                "edge_density_floor": 0.077,
                "repair_noise_mask_sha256": "c" * 64,
                "checks": {
                    "global_pose_gate_contract_valid": True,
                    "raw_issue_contract_valid": True,
                    "edge_density_within_recovery_band": True,
                    "chamfer_geometry_passed": True,
                    "continuous_foreleg_passed": True,
                    "paw_muzzle_contact_passed": True,
                    "raised_planted_paw_color_histogram_passed": True,
                    "raised_planted_paw_texture_contract_passed": True,
                    "independent_fur_surface_gate_passed": True,
                    "masked_pet3d_repair_contract_valid": True,
                },
            }
            effective_config = {
                "actions": {
                    "grooming": {
                        "reference": str(reference),
                        "reference_sha256": reference_sha,
                        "reference_kind": "qwen_generated_upload",
                    }
                }
            }
            audit = Mock(return_value={"passed": True, "policy": {}})
            with (
                patch.object(
                    pipeline,
                    "resolve_main_case_fur_surface_baseline",
                    return_value=(
                        profile,
                        {
                            "snapshot_sha256": pipeline.file_sha256(profile),
                            "provenance": {"profile_id": "test"},
                        },
                    ),
                ),
                patch.object(pipeline, "audit_final_video", audit),
            ):
                result = pipeline.evaluate_final_video_fur_surface(
                    video,
                    action="grooming",
                    effective_subject_config=effective_config,
                    comparison_scope=(
                        pipeline.FINAL_VIDEO_VALIDATED_ACTION_SURFACE_SCOPE
                    ),
                    validated_action_reference_path=reference,
                    validated_action_reference_sha256=reference_sha,
                    validated_action_reference_gate=selected_gate,
                    reference_policy="qwen_action_specific_identity",
                    pose_identity_mask_video_path=pose_mask,
                    pose_identity_mask_video_sha256=pose_mask_sha,
                    reference_identity_mask_path=reference_mask,
                    reference_identity_mask_sha256=reference_mask_sha,
                    grooming_pose_gate_composite_recovery=(composite_recovery),
                )
            self.assertTrue(result["passed"])
            forwarded = audit.call_args.kwargs
            self.assertEqual(forwarded["action"], "grooming")
            self.assertEqual(
                forwarded["comparison_scope"],
                pipeline.FINAL_VIDEO_VALIDATED_ACTION_SURFACE_SCOPE,
            )
            self.assertEqual(
                forwarded["validated_action_reference_provenance"]["reference_policy"],
                "qwen_action_specific_identity",
            )
            self.assertEqual(
                forwarded["validated_action_reference_provenance"][
                    "grooming_pose_gate_composite_recovery"
                ],
                composite_recovery,
            )

    def test_final_video_v2_rejects_divergent_selected_gate_sha(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "video.mp4"
            profile = root / "profile.png"
            reference = root / "reference.png"
            video.write_bytes(b"video")
            Image.new("RGBA", (16, 16), (80, 80, 80, 255)).save(profile)
            Image.new("RGBA", (16, 16), (90, 90, 90, 255)).save(reference)
            reference_sha = pipeline.file_sha256(reference)
            effective_config = {
                "actions": {
                    "eating": {
                        "reference": str(reference),
                        "reference_sha256": reference_sha,
                        "reference_kind": "qwen_generated_upload",
                    }
                }
            }
            with patch.object(
                pipeline,
                "resolve_main_case_fur_surface_baseline",
                return_value=(
                    profile,
                    {
                        "snapshot_sha256": pipeline.file_sha256(profile),
                        "provenance": {},
                    },
                ),
            ):
                result = pipeline.evaluate_final_video_fur_surface(
                    video,
                    action="eating",
                    effective_subject_config=effective_config,
                    comparison_scope=(
                        pipeline.FINAL_VIDEO_VALIDATED_ACTION_SURFACE_SCOPE
                    ),
                    validated_action_reference_path=reference,
                    validated_action_reference_sha256=reference_sha,
                    validated_action_reference_gate={
                        "selected_candidate_sha256": "0" * 64,
                        "gate": {"candidate_sha256": reference_sha},
                    },
                    reference_policy="qwen_action_identity_with_bowl_mapped_red",
                )
            self.assertFalse(result["passed"])
            self.assertIn("provenance changed", result["error"])


class PoseAttemptFailureReportingTests(unittest.TestCase):
    def test_unevaluated_surface_placeholder_is_not_reported_as_a_failure(self) -> None:
        attempts = [
            {
                "attempt": attempt,
                "grooming_pose_gate": {
                    "issues": ["raised_paw_color_inconsistent_with_planted_paw"]
                },
                "fur_surface_gate": {
                    "evaluated": False,
                    "acceptable": False,
                    "issues": ["fur_surface_gate_not_evaluated_upstream_gate_failed"],
                },
            }
            for attempt in range(1, 4)
        ]
        summary = pipeline.pose_attempt_failure_summary(attempts)
        self.assertEqual(
            summary,
            "persistent: raised_paw_color_inconsistent_with_planted_paw",
        )
        self.assertNotIn("fur_surface", summary)
        self.assertEqual(
            pipeline.pose_attempt_failure_issues(attempts),
            ["raised_paw_color_inconsistent_with_planted_paw"],
        )

    def test_evaluated_surface_rejection_remains_in_failure_summary(self) -> None:
        attempts = [
            {
                "attempt": 1,
                "fur_surface_gate": {
                    "evaluated": True,
                    "acceptable": False,
                    "issues": ["coherent_long_fur_energy"],
                },
            }
        ]
        summary = pipeline.pose_attempt_failure_summary(attempts)
        self.assertEqual(summary, "persistent: coherent_long_fur_energy")


class AdvisoryGenerationAuditTests(unittest.TestCase):
    def test_default_policy_keeps_only_runtime_integrity_blocking(self) -> None:
        self.assertFalse(pipeline.GENERATION_VISUAL_AUDITS_BLOCKING)
        self.assertEqual(
            pipeline.GENERATION_AUDIT_POLICY,
            "runtime_integrity_required_visual_content_advisory_v1",
        )
        audit = pipeline.skipped_visual_audit("grooming_pose_detail")
        self.assertTrue(audit["acceptable"])
        self.assertFalse(audit["blocking"])
        self.assertEqual(audit["status"], "skipped_advisory")

    def test_palette_drift_no_longer_rejects_all_canonical_candidates(self) -> None:
        attempts = [
            {
                "attempt": attempt,
                "identity_palette": {
                    "acceptable": False,
                    "issues": ["achromatic_coat_value_balance_lost"],
                    "penalty": float(attempt),
                },
                "quality": {"penalty": float(attempt)},
                "chroma": {"acceptable": True},
            }
            for attempt in range(1, 4)
        ]
        selected = pipeline.select_canonical_fallback_attempt(
            attempts,
            identity_palette_required=True,
        )
        self.assertEqual(selected["attempt"], 1)

    def test_semantic_recovery_ignores_visual_issues_but_not_bad_alpha(self) -> None:
        failed_visual = {
            "acceptable": False,
            "issues": ["visual_issue"],
        }
        self.assertIsNone(
            pipeline.semantic_recovery_gate_failure(
                {"acceptable": True},
                failed_visual,
                failed_visual,
                failed_visual,
                failed_visual,
                failed_visual,
            )
        )
        self.assertEqual(
            pipeline.semantic_recovery_gate_failure(
                {"acceptable": False},
                failed_visual,
                failed_visual,
                failed_visual,
            ),
            "semantic_chroma_invalid",
        )

    def test_pre_scail_revalidation_needs_alpha_and_digests_not_surface_scores(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_root = root / "outputs"
            comfy_input = root / "comfy-input"
            job_id = "advisory-pre-scail"
            job_dir = output_root / job_id
            job_dir.mkdir(parents=True)
            canonical = job_dir / "cartoon_reference_transparent.png"
            canonical_image1 = job_dir / "cartoon_reference.png"
            action_reference = job_dir / "eating_transparent.png"
            rgba = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
            rgba.paste((90, 100, 110, 255), (128, 64, 384, 448))
            rgba.save(canonical)
            rgba.save(action_reference)
            green = Image.new("RGB", (512, 512), (0, 255, 0))
            green.paste((90, 100, 110), (128, 64, 384, 448))
            green.save(canonical_image1)
            comfy_canonical = (
                comfy_input
                / "local_video_api"
                / job_id
                / canonical_image1.name
            )
            comfy_canonical.parent.mkdir(parents=True)
            comfy_canonical.write_bytes(canonical_image1.read_bytes())
            canonical_digest = pipeline.file_sha256(canonical)
            image1_digest = pipeline.file_sha256(canonical_image1)
            action_digest = pipeline.file_sha256(action_reference)
            jobs: dict[str, dict[str, object]] = {
                job_id: {
                    "job_id": job_id,
                    "subject_source": "qwen_generated_upload",
                    "reference_generation_completed": True,
                    "reference_generation_status": "completed",
                    "reference_normalization_status": "completed",
                    "canonical_reference_generated": True,
                    "canonical_reference_used": True,
                    "canonical_reference_normalized": True,
                    "canonical_reference_transparent": True,
                    "qwen_action_keyframes_normalized": True,
                    "qwen_action_keyframes_transparent": True,
                    "qwen_action_reference_normalization_policy": (
                        pipeline.QWEN_ACTION_REFERENCE_NORMALIZATION_POLICY
                    ),
                    "qwen_action_reference_byte_preserved": True,
                    "surface_texture_gate_status": "skipped_advisory",
                    "canonical_reference_path": str(canonical),
                    "canonical_reference_sha256": canonical_digest,
                    "canonical_public_reference_path": str(canonical),
                    "canonical_public_reference_sha256": canonical_digest,
                    "canonical_qwen_image1_path": str(canonical_image1),
                    "canonical_qwen_image1_comfy_path": str(comfy_canonical),
                    "canonical_qwen_image1_sha256": image1_digest,
                    "canonical_qwen_image1_comfy_sha256": image1_digest,
                    "subject_reference_paths": {"eating": str(action_reference)},
                    "qwen_action_keyframe_sha256s": {"eating": action_digest},
                    "qwen_reference_generated_actions": ["eating"],
                }
            }
            effective_config = {
                "actions": {
                    "eating": {
                        "reference_kind": "qwen_generated_upload",
                        "reference": str(action_reference),
                        "reference_sha256": action_digest,
                    }
                }
            }
            with (
                patch.multiple(
                    pipeline,
                    OUTPUT_ROOT=output_root,
                    COMFY_INPUT=comfy_input,
                    JOBS=jobs,
                    GENERATION_VISUAL_AUDITS_BLOCKING=False,
                ),
                patch.object(pipeline, "verify_uploaded_identity_submission"),
            ):
                pipeline.revalidate_uploaded_qwen_references_before_scail(
                    job_id,
                    jobs[job_id],
                    effective_config,
                    ["eating"],
                )

            self.assertTrue(jobs[job_id]["qwen_reference_pre_scail_revalidated"])
            self.assertEqual(
                jobs[job_id]["qwen_reference_pre_scail_visual_audit_status"],
                "skipped_advisory",
            )
            self.assertEqual(jobs[job_id]["subject_fur_surface_quality"], {})

    def test_advisory_grooming_pre_scail_requires_minimum_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_root = root / "outputs"
            comfy_input = root / "comfy-input"
            job_id = "advisory-grooming-pre-scail"
            job_dir = output_root / job_id
            job_dir.mkdir(parents=True)
            canonical = job_dir / "cartoon_reference_transparent.png"
            canonical_image1 = job_dir / "cartoon_reference.png"
            action_reference = job_dir / "grooming_transparent.png"
            rgba = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
            rgba.paste((220, 220, 220, 255), (128, 64, 384, 448))
            rgba.save(canonical)
            rgba.save(action_reference)
            green = Image.new("RGB", (512, 512), (0, 255, 0))
            green.paste((220, 220, 220), (128, 64, 384, 448))
            green.save(canonical_image1)
            comfy_canonical = (
                comfy_input
                / "local_video_api"
                / job_id
                / canonical_image1.name
            )
            comfy_canonical.parent.mkdir(parents=True)
            comfy_canonical.write_bytes(canonical_image1.read_bytes())
            canonical_digest = pipeline.file_sha256(canonical)
            image1_digest = pipeline.file_sha256(canonical_image1)
            action_digest = pipeline.file_sha256(action_reference)
            selected_minimum = {
                "policy": pipeline.GROOMING_ADVISORY_MINIMUM_GATE_POLICY,
                "version": pipeline.GROOMING_ADVISORY_MINIMUM_GATE_VERSION,
                "acceptable": True,
                "candidate_sha256": action_digest,
                "selected_candidate_path": str(action_reference),
                "selected_candidate_sha256": action_digest,
                "identity_reference_path": str(canonical_image1),
                "identity_reference_sha256": image1_digest,
            }
            jobs: dict[str, dict[str, object]] = {
                job_id: {
                    "job_id": job_id,
                    "subject_source": "qwen_generated_upload",
                    "reference_generation_completed": True,
                    "reference_generation_status": "completed",
                    "reference_normalization_status": "completed",
                    "canonical_reference_generated": True,
                    "canonical_reference_used": True,
                    "canonical_reference_normalized": True,
                    "canonical_reference_transparent": True,
                    "qwen_action_keyframes_normalized": True,
                    "qwen_action_keyframes_transparent": True,
                    "qwen_action_reference_normalization_policy": (
                        pipeline.QWEN_ACTION_REFERENCE_NORMALIZATION_POLICY
                    ),
                    "qwen_action_reference_byte_preserved": True,
                    "surface_texture_gate_status": "skipped_advisory",
                    "canonical_reference_path": str(canonical),
                    "canonical_reference_sha256": canonical_digest,
                    "canonical_public_reference_path": str(canonical),
                    "canonical_public_reference_sha256": canonical_digest,
                    "canonical_qwen_image1_path": str(canonical_image1),
                    "canonical_qwen_image1_comfy_path": str(comfy_canonical),
                    "canonical_qwen_image1_sha256": image1_digest,
                    "canonical_qwen_image1_comfy_sha256": image1_digest,
                    "subject_reference_paths": {
                        "grooming": str(action_reference)
                    },
                    "qwen_action_keyframe_sha256s": {
                        "grooming": action_digest
                    },
                    "qwen_reference_generated_actions": ["grooming"],
                    "grooming_advisory_minimum_gate_status": "passed",
                    "grooming_advisory_minimum_gate_selected": {
                        "grooming": selected_minimum
                    },
                }
            }
            effective_config = {
                "actions": {
                    "grooming": {
                        "reference_kind": "qwen_generated_upload",
                        "reference": str(action_reference),
                        "reference_sha256": action_digest,
                    }
                }
            }
            with (
                patch.multiple(
                    pipeline,
                    OUTPUT_ROOT=output_root,
                    COMFY_INPUT=comfy_input,
                    JOBS=jobs,
                    GENERATION_VISUAL_AUDITS_BLOCKING=False,
                ),
                patch.object(pipeline, "verify_uploaded_identity_submission"),
            ):
                pipeline.revalidate_uploaded_qwen_references_before_scail(
                    job_id,
                    jobs[job_id],
                    effective_config,
                    ["grooming"],
                )
                self.assertEqual(
                    jobs[job_id][
                        "qwen_reference_pre_scail_minimum_contract_status"
                    ],
                    "passed",
                )
                selected_minimum["selected_candidate_sha256"] = "0" * 64
                with self.assertRaisesRegex(
                    RuntimeError,
                    "minimum action/identity contract",
                ):
                    pipeline.revalidate_uploaded_qwen_references_before_scail(
                        job_id,
                        jobs[job_id],
                        effective_config,
                        ["grooming"],
                    )


if __name__ == "__main__":
    unittest.main()
