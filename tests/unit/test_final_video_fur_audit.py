from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from final_video_fur_audit import (
    VALIDATED_ACTION_SURFACE_SCOPE,
    audit_final_video,
    clear_red_prop_from_alpha,
    sha256_file,
    validate_identity_mask_frame,
)


SIZE = 64


def save_reference(path: Path) -> None:
    image = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((8, 5, 55, 58), fill=(90, 100, 120, 255))
    draw.rectangle((18, 44, 46, 58), fill=(210, 150, 90, 255))
    image.save(path)


def save_identity_mask(
    path: Path,
    *,
    background: str,
    reliable: bool = True,
    include_prop: bool = True,
) -> None:
    color = (255, 255, 255) if background == "white" else (0, 0, 0)
    image = Image.new("RGB", (SIZE, SIZE), color)
    draw = ImageDraw.Draw(image)
    if reliable:
        draw.ellipse((8, 5, 55, 58), fill=(0, 0, 255))
        draw.rectangle(
            (18, 44, 46, 58),
            fill=(255, 0, 0) if include_prop else (0, 0, 255),
        )
    else:
        draw.rectangle((0, 0, SIZE - 1, SIZE - 1), fill=(0, 180, 0))
    image.save(path)


def grooming_composite_recovery() -> dict[str, Any]:
    return {
        "policy": (
            "masked_pet3d_surface_verified_pose_edge_and_low_texture_"
            "paw_rescue_v3"
        ),
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


def evaluation(
    *,
    acceptable: bool,
    relative: list[str] | None = None,
    absolute: list[str] | None = None,
    candidate_radial_fraction: float = 0.30,
    baseline_radial_fraction: float = 0.25,
    candidate_gradient_p90: float | None = 0.30,
    baseline_gradient_p90: float | None = 0.40,
    candidate_coarse_gradient_p90: float = 0.20,
    baseline_coarse_gradient_p90: float = 0.20,
    candidate_coherent_fraction: float = 0.50,
    baseline_coherent_fraction: float = 0.50,
    candidate_radial_energy: float = 0.46,
    baseline_radial_energy: float = 0.45,
    candidate_long_energy: float = 0.1,
    baseline_long_energy: float = 0.1,
    baseline_issues: list[str] | None = None,
) -> dict[str, Any]:
    relative_issues = list(relative or [])
    absolute_issues = list(absolute or [])
    return {
        "acceptable": acceptable,
        "verdict": "accepted" if acceptable else "rejected",
        "issues": list(dict.fromkeys([*absolute_issues, *relative_issues])),
        "candidate_absolute_issues": absolute_issues,
        "relative_issues": relative_issues,
        "baseline_issues": list(baseline_issues or []),
        "candidate": {
            "valid": True,
            "input": {
                "supported_contract": True,
                "foreground_method": "alpha",
            },
            "metrics": {
                "texture": {
                    "coherent_long_line_energy": candidate_long_energy,
                    "coarse_gradient_p90": candidate_coarse_gradient_p90,
                    "structure_tensor_coherent_fraction": (
                        candidate_coherent_fraction
                    ),
                    "fine_gradient_p90": candidate_gradient_p90,
                    "fine_radial_brush_fraction": candidate_radial_fraction,
                    "fine_radial_brush_energy": candidate_radial_energy,
                }
            },
        },
        "baseline": {
            "valid": True,
            "input": {
                "supported_contract": True,
                "foreground_method": "alpha",
            },
            "metrics": {
                "texture": {
                    "coherent_long_line_energy": baseline_long_energy,
                    "coarse_gradient_p90": baseline_coarse_gradient_p90,
                    "structure_tensor_coherent_fraction": (
                        baseline_coherent_fraction
                    ),
                    "fine_gradient_p90": baseline_gradient_p90,
                    "fine_radial_brush_fraction": baseline_radial_fraction,
                    "fine_radial_brush_energy": baseline_radial_energy,
                }
            }
        },
        "comparisons": {},
        "thresholds": {
            "coherence_gradient_p90_min": 0.14,
            "coarse_gradient_p90_max": 0.24,
            "coherent_long_line_energy_max": 0.24,
            "coarse_gradient_ratio_max": 1.28,
            "coherent_fraction_ratio_max": 1.25,
            "radial_brush_fraction_max": 0.52,
            "radial_brush_ratio_max": 1.22,
            "long_line_energy_ratio_max": 1.20,
            "texture_delta_min": 0.015,
        },
    }


class FinalVideoFurAuditV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.video = self.root / "video.mp4"
        self.pose_mask_video = self.root / "pose_identity_mask.mp4"
        self.profile = self.root / "profile.png"
        self.reference = self.root / "validated_action_reference.png"
        self.reference_mask = self.root / "reference_identity_mask.png"
        self.video.write_bytes(b"synthetic-video")
        self.pose_mask_video.write_bytes(b"synthetic-mask-video")
        save_reference(self.profile)
        save_reference(self.reference)
        save_identity_mask(self.reference_mask, background="black")
        self.mask_reliable = True
        self.texture_relative: list[str] = []
        self.texture_absolute: list[str] = []
        self.texture_issue_frame_indices: set[int] | None = None
        self.contour_issue_frame_indices: set[int] | None = None
        self.frame_texture_relative_overrides: dict[int, list[str]] = {}
        self.frame_contour_relative_overrides: dict[int, list[str]] = {}
        self.texture_baseline_issues: list[str] = []
        self.texture_baseline_issues_malformed: object | None = None
        self.grooming_composite_recovery: dict[str, Any] | None = None
        self.contour_relative: list[str] = []
        self.contour_baseline_issues: list[str] = []
        self.action = "eating"
        self.candidate_radial_fraction = 0.32891488
        self.baseline_radial_fraction = 0.26393672
        self.candidate_gradient_p90: float | None = 0.27266930
        self.baseline_gradient_p90: float | None = 0.39799012
        self.candidate_coarse_gradient_p90 = 0.20
        self.baseline_coarse_gradient_p90 = 0.20
        self.candidate_coherent_fraction = 0.50
        self.baseline_coherent_fraction = 0.50
        self.candidate_radial_energy = 0.46993961
        self.baseline_radial_energy = 0.44856318
        self.candidate_long_energy = 0.1
        self.baseline_long_energy = 0.1
        self.frame_candidate_coarse_gradient_p90: dict[int, float] = {}
        self.frame_candidate_long_energy: dict[int, float] = {}
        self.frame_candidate_radial_fraction: dict[int, float] = {}
        self.frame_candidate_gradient_p90: dict[int, float] = {}
        self.frame_candidate_radial_energy: dict[int, float] = {}
        self.frame_count = 3
        self.sample_count = 3
        self.frame_evaluator_calls = 0
        self.texture_baseline_validation_flags: list[bool] = []

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def probe(self, _path: str | Path) -> dict[str, Any]:
        return {
            "frame_count": self.frame_count,
            "width": SIZE,
            "height": SIZE,
            "fps": 16.0,
            "duration_seconds": 0.1875,
        }

    def extract(self, source: str | Path, _index: int, target: Path) -> None:
        if Path(source) == self.pose_mask_video:
            save_identity_mask(
                target,
                background="white",
                reliable=self.mask_reliable,
                include_prop=self.action == "eating",
            )
        else:
            Image.new("RGB", (SIZE, SIZE), (255, 255, 255)).save(target)

    def remove_background(
        self,
        _source: Path,
        target: Path,
        *,
        model: str,
    ) -> dict[str, Any]:
        self.assertEqual(model, "u2net")
        save_reference(target)
        return {"reliable": True, "issues": [], "method": "synthetic_rembg"}

    def evaluator(
        self,
        candidate: str | Path,
        _baseline: str | Path,
        *,
        baseline_is_validated: bool,
    ) -> dict[str, Any]:
        name = Path(candidate).name
        if name in {self.profile.name, self.reference.name}:
            self.assertTrue(baseline_is_validated)
            return evaluation(acceptable=True)
        self.frame_evaluator_calls += 1
        if name.endswith("_prop_excluded.png"):
            self.texture_baseline_validation_flags.append(
                baseline_is_validated
            )
            frame_index = int(name.split("_", 2)[1])
            texture_issue_active = bool(
                self.texture_issue_frame_indices is None
                or frame_index in self.texture_issue_frame_indices
            )
            active_relative = self.frame_texture_relative_overrides.get(
                frame_index,
                self.texture_relative if texture_issue_active else [],
            )
            active_absolute = (
                self.texture_absolute if texture_issue_active else []
            )
            result = evaluation(
                acceptable=not active_relative,
                relative=active_relative,
                absolute=["coarse_fur_stroke_energy", *active_absolute],
                candidate_radial_fraction=(
                    self.frame_candidate_radial_fraction.get(
                        frame_index,
                        self.candidate_radial_fraction,
                    )
                ),
                baseline_radial_fraction=self.baseline_radial_fraction,
                candidate_gradient_p90=(
                    self.frame_candidate_gradient_p90.get(
                        frame_index,
                        self.candidate_gradient_p90,
                    )
                ),
                baseline_gradient_p90=self.baseline_gradient_p90,
                candidate_coarse_gradient_p90=(
                    self.frame_candidate_coarse_gradient_p90.get(
                        frame_index,
                        self.candidate_coarse_gradient_p90,
                    )
                ),
                baseline_coarse_gradient_p90=(
                    self.baseline_coarse_gradient_p90
                ),
                candidate_coherent_fraction=(
                    self.candidate_coherent_fraction
                ),
                baseline_coherent_fraction=self.baseline_coherent_fraction,
                candidate_radial_energy=(
                    self.frame_candidate_radial_energy.get(
                        frame_index,
                        self.candidate_radial_energy,
                    )
                ),
                baseline_radial_energy=self.baseline_radial_energy,
                candidate_long_energy=self.frame_candidate_long_energy.get(
                    frame_index,
                    self.candidate_long_energy,
                ),
                baseline_long_energy=self.baseline_long_energy,
                baseline_issues=self.texture_baseline_issues,
            )
            if self.texture_baseline_issues_malformed is not None:
                result["baseline_issues"] = self.texture_baseline_issues_malformed
            return result
        frame_index = int(name.split("_", 2)[1])
        contour_issue_active = bool(
            self.contour_issue_frame_indices is None
            or frame_index in self.contour_issue_frame_indices
        )
        active_contour_relative = self.frame_contour_relative_overrides.get(
            frame_index,
            self.contour_relative if contour_issue_active else [],
        )
        return evaluation(
            acceptable=not active_contour_relative,
            relative=active_contour_relative,
            absolute=["brush_texture_coherence"],
            baseline_issues=self.contour_baseline_issues,
        )

    def audit(self) -> dict[str, Any]:
        reference_sha = sha256_file(self.reference)
        selected_gate = {
            "acceptable": True,
            "verdict": "accepted_uploaded_identity_absolute_contract",
            "selected_candidate_sha256": reference_sha,
            "policy": {
                "name": "validated_action_reference_fur_surface_v1",
                "gate_version": "1.1.0",
                "threshold_version": "uploaded-identity-surface-test-v1",
                "validated_baseline_required_for_acceptance": False,
                "uploaded_identity_absolute_contract_enabled": True,
            },
            "gate": {
                "action": self.action,
                "candidate_sha256": reference_sha,
                "candidate_comparison_mode": (
                    "uploaded_identity_absolute_surface_contract_v1"
                ),
                "profile_pixels_used_for_candidate_appearance_validation": False,
            },
        }
        provenance: dict[str, Any] = {
            "reference_kind": "qwen_generated_upload",
            "configured_reference_sha256": reference_sha,
            "selected_candidate_sha256": reference_sha,
            "gate_candidate_sha256": reference_sha,
            "pre_scail_revalidated": True,
            "gate": selected_gate,
        }
        if self.grooming_composite_recovery is not None:
            provenance["grooming_pose_gate_composite_recovery"] = (
                self.grooming_composite_recovery
            )
        return audit_final_video(
            self.video,
            action=self.action,
            baseline_path=self.profile,
            baseline_sha256=sha256_file(self.profile),
            baseline_provenance={"profile_id": "synthetic"},
            sample_count=self.sample_count,
            evaluator=self.evaluator,
            frame_extractor=self.extract,
            video_probe=self.probe,
            background_remover=self.remove_background,
            comparison_scope=VALIDATED_ACTION_SURFACE_SCOPE,
            validated_action_reference_path=self.reference,
            validated_action_reference_sha256=reference_sha,
            validated_action_reference_provenance=provenance,
            pose_identity_mask_video_path=self.pose_mask_video,
            pose_identity_mask_video_sha256=sha256_file(self.pose_mask_video),
            reference_identity_mask_path=self.reference_mask,
            reference_identity_mask_sha256=sha256_file(self.reference_mask),
        )

    def configure_low_signal_grooming_reference_case(self) -> None:
        self.action = "grooming"
        save_identity_mask(
            self.reference_mask,
            background="black",
            include_prop=False,
        )
        self.grooming_composite_recovery = grooming_composite_recovery()
        issues = [
            "coarse_strokes_expanded_vs_profile",
            "long_fur_energy_expanded_vs_profile",
        ]
        self.texture_relative = list(issues)
        self.contour_relative = list(issues)
        self.candidate_coarse_gradient_p90 = 0.1727
        self.baseline_coarse_gradient_p90 = 0.09798744
        self.candidate_long_energy = 0.1785
        self.baseline_long_energy = 0.09695754

    def configure_dense_low_signal_mixed_long_grooming_case(self) -> None:
        self.configure_low_signal_grooming_reference_case()
        self.frame_count = 9
        self.sample_count = 9
        self.baseline_coarse_gradient_p90 = 0.09756645
        self.baseline_long_energy = 0.09681457
        self.frame_candidate_coarse_gradient_p90 = dict(
            enumerate(
                (
                    0.15005128,
                    0.15951843,
                    0.14670277,
                    0.15925690,
                    0.15417358,
                    0.15398050,
                    0.15171522,
                    0.14201814,
                    0.13896633,
                )
            )
        )
        self.frame_candidate_long_energy = dict(
            enumerate(
                (
                    0.15194075,
                    0.15824529,
                    0.14590363,
                    0.15727952,
                    0.15712477,
                    0.15727899,
                    0.15302389,
                    0.14488540,
                    0.14086080,
                )
            )
        )

    def configure_sparse_coarse_low_signal_grooming_case(self) -> None:
        self.action = "grooming"
        self.frame_count = 9
        self.sample_count = 9
        save_identity_mask(
            self.reference_mask,
            background="black",
            include_prop=False,
        )
        self.grooming_composite_recovery = grooming_composite_recovery()
        coarse = "coarse_strokes_expanded_vs_profile"
        long_fur = "long_fur_energy_expanded_vs_profile"
        self.texture_relative = [long_fur]
        self.contour_relative = [long_fur]
        self.frame_texture_relative_overrides = {
            0: [coarse, long_fur],
            1: [coarse, long_fur],
            4: [coarse, long_fur],
            7: ["upper_body_width_expanded", long_fur],
        }
        self.frame_contour_relative_overrides = {
            0: [coarse, long_fur],
            1: [coarse, long_fur],
            4: [coarse, long_fur],
            7: ["upper_body_width_expanded", long_fur],
        }
        self.baseline_coarse_gradient_p90 = 0.13765330
        self.baseline_long_energy = 0.12114103
        self.candidate_coarse_gradient_p90 = 0.18
        coarse_values = (
            0.18623375,
            0.18237696,
            0.16988126,
            0.17434274,
            0.17750587,
            0.17129120,
            0.17154991,
            0.16697542,
            0.16733317,
        )
        long_values = (
            0.16332443,
            0.16127349,
            0.15414427,
            0.16197878,
            0.16587823,
            0.16131856,
            0.16005403,
            0.15278894,
            0.15336036,
        )
        self.frame_candidate_coarse_gradient_p90 = dict(
            enumerate(coarse_values)
        )
        self.frame_candidate_long_energy = dict(enumerate(long_values))

    def configure_direct_qwen_sparse_coarse_grooming_case(self) -> None:
        self.action = "grooming"
        self.frame_count = 9
        self.sample_count = 9
        save_identity_mask(
            self.reference_mask,
            background="black",
            include_prop=False,
        )
        self.grooming_composite_recovery = None
        coarse = "coarse_strokes_expanded_vs_profile"
        long_fur = "long_fur_energy_expanded_vs_profile"
        self.texture_relative = []
        self.contour_relative = []
        self.frame_texture_relative_overrides = {
            0: [coarse, long_fur],
            1: [coarse, long_fur],
            3: [coarse, long_fur],
            4: [long_fur],
            5: [long_fur],
            6: [long_fur],
        }
        self.frame_contour_relative_overrides = {
            index: list(issues)
            for index, issues in self.frame_texture_relative_overrides.items()
        }
        self.baseline_coarse_gradient_p90 = 0.17014021
        self.baseline_long_energy = 0.13854357
        coarse_values = (
            0.22031633,
            0.21882705,
            0.20227633,
            0.21932246,
            0.21122834,
            0.20873569,
            0.20967547,
            0.19623622,
            0.19380313,
        )
        long_values = (
            0.17212403,
            0.16953813,
            0.16412911,
            0.18205735,
            0.17847169,
            0.17629612,
            0.17607052,
            0.16392789,
            0.16184081,
        )
        self.frame_candidate_coarse_gradient_p90 = dict(
            enumerate(coarse_values)
        )
        self.frame_candidate_long_energy = dict(enumerate(long_values))

    def configure_eating_action_phase_boundary_case(self) -> None:
        self.action = "eating"
        self.frame_count = 9
        self.sample_count = 9
        issues = [
            "coarse_strokes_expanded_vs_profile",
            "long_fur_energy_expanded_vs_profile",
        ]
        self.texture_relative = list(issues)
        self.contour_relative = list(issues)
        self.texture_issue_frame_indices = {2, 3, 4}
        self.contour_issue_frame_indices = {2, 3, 4}
        fringe = "contour_fringe_expanded_vs_profile"
        self.frame_texture_relative_overrides = {
            0: [fringe],
            3: [fringe, *issues],
            4: [fringe, *issues],
            5: [fringe, "long_fur_energy_expanded_vs_profile"],
            6: [fringe, "long_fur_energy_expanded_vs_profile"],
            7: [fringe],
            8: [fringe],
        }
        self.frame_contour_relative_overrides = {
            5: ["long_fur_energy_expanded_vs_profile"],
            6: ["long_fur_energy_expanded_vs_profile"],
        }
        self.baseline_coarse_gradient_p90 = 0.13521676
        self.baseline_long_energy = 0.14182064
        coarse_values = (
            0.1470,
            0.1414,
            0.1825,
            0.1760,
            0.1735,
            0.1671,
            0.1706,
            0.1636,
            0.1425,
        )
        long_values = (
            0.1669,
            0.1511,
            0.2067,
            0.1975,
            0.1919,
            0.1858,
            0.1820,
            0.1643,
            0.1483,
        )
        self.frame_candidate_coarse_gradient_p90 = dict(
            enumerate(coarse_values)
        )
        self.frame_candidate_long_energy = dict(enumerate(long_values))

    def configure_eating_isolated_strong_long_fur_case(self) -> None:
        self.action = "eating"
        self.frame_count = 9
        self.sample_count = 9
        issue = "long_fur_energy_expanded_vs_profile"
        self.texture_relative = [issue]
        self.contour_relative = [issue]
        self.texture_issue_frame_indices = {4, 5, 6}
        self.contour_issue_frame_indices = {6}
        self.baseline_long_energy = 0.16165724
        long_values = (
            0.18323322,
            0.17043386,
            0.18905176,
            0.19024942,
            0.19847216,
            0.20257684,
            0.21946399,
            0.17700907,
            0.14534439,
        )
        self.frame_candidate_long_energy = dict(enumerate(long_values))

    def configure_eating_initial_radial_boundary_case(self) -> None:
        self.action = "eating"
        self.frame_count = 9
        self.sample_count = 9
        issue = "radial_brush_texture_expanded_vs_profile"
        self.texture_relative = [issue]
        self.contour_relative = [issue]
        self.texture_issue_frame_indices = {0}
        self.contour_issue_frame_indices = {0}
        self.texture_baseline_issues = ["radial_long_lines"]
        self.baseline_radial_fraction = 0.30378251
        self.baseline_gradient_p90 = 0.24672212
        self.baseline_radial_energy = 0.44572905
        self.candidate_radial_fraction = 0.33
        self.candidate_gradient_p90 = 0.28
        self.candidate_radial_energy = 0.48
        self.frame_candidate_radial_fraction = {
            0: 0.39288796,
            8: 0.34573543,
        }
        self.frame_candidate_gradient_p90 = {
            0: 0.25779910,
            8: 0.28810423,
        }
        self.frame_candidate_radial_energy = {
            0: 0.54802108,
            8: 0.52142485,
        }

    def configure_eating_combined_isolated_surface_peaks(self) -> None:
        self.configure_eating_isolated_strong_long_fur_case()
        radial = "radial_brush_texture_expanded_vs_profile"
        coherent = "coherent_brush_texture_expanded_vs_profile"
        self.frame_texture_relative_overrides[0] = [coherent, radial]
        self.frame_contour_relative_overrides[0] = [coherent]
        self.contour_issue_frame_indices = set()
        self.texture_baseline_issues = ["radial_long_lines"]
        self.baseline_radial_fraction = 0.30753138
        self.baseline_gradient_p90 = 0.25119658
        self.baseline_radial_energy = 0.45802651
        self.candidate_radial_fraction = 0.29612326
        self.candidate_gradient_p90 = 0.28
        self.candidate_radial_energy = 0.47561448
        self.frame_candidate_radial_fraction[0] = 0.39026493
        self.frame_candidate_gradient_p90[0] = 0.25624280
        self.frame_candidate_radial_energy[0] = 0.56187447
        self.baseline_long_energy = 0.16751640
        self.frame_candidate_long_energy.update(
            {
                4: 0.22054789,
                5: 0.22323310,
                6: 0.22733869,
                7: 0.20798456,
                8: 0.21580394,
            }
        )

    def test_grooming_no_prop_masks_use_the_validated_action_scope(self) -> None:
        self.action = "grooming"
        save_identity_mask(
            self.reference_mask,
            background="black",
            include_prop=False,
        )
        result = self.audit()
        self.assertTrue(result["passed"])
        self.assertEqual(result["action"], "grooming")
        self.assertEqual(
            result["policy"]["comparison_scope"],
            VALIDATED_ACTION_SURFACE_SCOPE,
        )
        self.assertEqual(
            result["validated_action_reference"]["texture_prop_exclusion"][
                "red_pixels"
            ],
            0,
        )

    def test_exact_red_prop_exclusion_only_decreases_alpha_and_preserves_rgb(self) -> None:
        target = self.root / "prop_excluded.png"
        result = clear_red_prop_from_alpha(
            self.reference,
            self.reference_mask,
            target,
            action="eating",
            mask_background="black",
        )
        source = np.asarray(Image.open(self.reference).convert("RGBA"))
        cleared = np.asarray(Image.open(target).convert("RGBA"))
        mask = np.asarray(Image.open(self.reference_mask).convert("RGB"))
        red = (mask[..., 0] == 255) & (mask[..., 1] == 0) & (mask[..., 2] == 0)
        self.assertTrue(np.array_equal(source[..., :3], cleared[..., :3]))
        self.assertTrue(np.all(cleared[..., 3] <= source[..., 3]))
        self.assertTrue(np.all(cleared[..., 3][red] == 0))
        self.assertTrue(result["rgb_byte_preserved"])
        self.assertTrue(result["alpha_only_decreased"])
        self.assertFalse(result["morphology_applied"])
        self.assertEqual(result["minimum_red_foreground_overlap"], 0.75)

    def test_eating_prop_alignment_accepts_verified_bowl_shadow_margin(
        self,
    ) -> None:
        source = self.root / "bowl_shadow_margin.png"
        target = self.root / "bowl_shadow_margin_cleared.png"
        rgba = np.asarray(Image.open(self.reference).convert("RGBA")).copy()
        rgba[56:59, 18:47, 3] = 0
        Image.fromarray(rgba, mode="RGBA").save(source)

        result = clear_red_prop_from_alpha(
            source,
            self.reference_mask,
            target,
            action="eating",
            mask_background="black",
        )

        self.assertGreaterEqual(result["red_foreground_overlap"], 0.75)
        self.assertLess(result["red_foreground_overlap"], 0.81)

    def test_eating_prop_alignment_below_hard_floor_fails_closed(self) -> None:
        source = self.root / "misaligned_bowl.png"
        target = self.root / "misaligned_bowl_cleared.png"
        rgba = np.asarray(Image.open(self.reference).convert("RGBA")).copy()
        rgba[55:59, 18:47, 3] = 0
        Image.fromarray(rgba, mode="RGBA").save(source)

        with self.assertRaisesRegex(
            RuntimeError,
            "identity_mask_prop_alignment_unreliable",
        ):
            clear_red_prop_from_alpha(
                source,
                self.reference_mask,
                target,
                action="eating",
                mask_background="black",
            )

    def test_derived_texture_reference_uses_validated_hash_bound_contract(self) -> None:
        result = self.audit()
        self.assertTrue(result["passed"])
        self.assertEqual(
            self.texture_baseline_validation_flags,
            [True, True, True],
        )
        authority = result["validated_action_reference"][
            "uploaded_identity_authority_validation"
        ]
        self.assertTrue(authority["hash_chain_verified"])
        self.assertFalse(
            result["baseline"][
                "pixels_used_for_uploaded_identity_appearance_validation"
            ]
        )

    def test_pose_geometry_and_non_radial_absolute_observations_do_not_block_v2(self) -> None:
        self.contour_relative = [
            "upper_body_area_expanded",
            "coarse_strokes_expanded_vs_profile",
        ]
        self.texture_relative = ["torso_width_expanded"]
        result = self.audit()
        self.assertTrue(result["passed"])
        self.assertEqual(result["failed_frame_indices"], [])
        for frame in result["frames"]:
            self.assertTrue(frame["decision"]["passed"])
            self.assertTrue(frame["decision"]["pose_relative_observations"])
            self.assertTrue(frame["decision"]["absolute_still_observations"])

    def test_prop_excluded_texture_relative_issue_fails_entire_video(self) -> None:
        self.texture_relative = ["long_fur_energy_expanded_vs_profile"]
        self.candidate_long_energy = 0.21
        self.baseline_long_energy = 0.10
        result = self.audit()
        self.assertFalse(result["passed"])
        self.assertEqual(result["failed_frame_indices"], [0, 1, 2])
        for frame in result["frames"]:
            self.assertIn(
                "long_fur_energy_expanded_vs_profile",
                frame["decision"]["blocking_issues"],
            )
            self.assertIn(
                "long_fur_energy_expanded_vs_validated_action_reference",
                frame["decision"][
                    "blocking_issues_action_reference_scope"
                ],
            )

    def test_full_sample_low_signal_grooming_composite_is_normalized(self) -> None:
        self.configure_low_signal_grooming_reference_case()

        result = self.audit()

        self.assertTrue(result["passed"])
        self.assertEqual(result["failed_frame_indices"], [])
        evidence = result["reference_evidence"][
            "low_signal_texture_floor"
        ]
        self.assertTrue(evidence["applied"])
        self.assertEqual(evidence["downgraded_sample_indices"], [0, 1, 2])
        self.assertAlmostEqual(
            evidence["baseline_coherent_long_line_energy"],
            0.09695754,
        )
        self.assertAlmostEqual(evidence["candidate_coarse_ceiling"], 0.1792)
        self.assertAlmostEqual(evidence["candidate_long_ceiling"], 0.189)
        for frame in result["frames"]:
            decision = frame["decision"]
            self.assertTrue(decision["passed"])
            self.assertEqual(decision["blocking_issues"], [])
            self.assertEqual(
                decision["texture_reference_floor_normalized_observations"],
                [
                    "coarse_strokes_expanded_vs_profile",
                    "long_fur_energy_expanded_vs_profile",
                ],
            )
            long_fur = decision["long_fur_relative_evidence"]
            self.assertTrue(
                long_fur["blocking_before_low_signal_reference_floor"]
            )
            self.assertTrue(
                long_fur["low_signal_reference_floor_observation"]
            )
            self.assertFalse(long_fur["blocking"])

    def test_low_signal_texture_floor_requires_verified_composite(self) -> None:
        self.configure_low_signal_grooming_reference_case()
        self.grooming_composite_recovery = None

        result = self.audit()

        self.assertFalse(result["passed"])
        evidence = result["reference_evidence"][
            "low_signal_texture_floor"
        ]
        self.assertFalse(evidence["applied"])
        self.assertEqual(
            evidence["reason"],
            "masked_pet3d_composite_recovery_not_verified",
        )

    def test_dense_low_signal_allows_terminal_long_observation(self) -> None:
        self.configure_dense_low_signal_mixed_long_grooming_case()

        result = self.audit()

        self.assertTrue(result["passed"])
        evidence = result["reference_evidence"][
            "low_signal_texture_floor"
        ]
        self.assertTrue(evidence["applied"])
        self.assertEqual(evidence["long_blocking_frame_indices"], list(range(8)))
        self.assertEqual(evidence["long_observation_frame_indices"], [8])
        self.assertAlmostEqual(evidence["long_blocking_fraction"], 8 / 9)
        terminal_long = result["frames"][8]["decision"][
            "long_fur_relative_evidence"
        ]
        self.assertFalse(
            terminal_long["blocking_before_low_signal_reference_floor"]
        )
        self.assertTrue(terminal_long["low_signal_reference_floor_observation"])

    def test_dense_low_signal_requires_half_long_blocking_samples(self) -> None:
        self.configure_dense_low_signal_mixed_long_grooming_case()
        for index in range(4, 9):
            self.frame_candidate_long_energy[index] = 0.14

        result = self.audit()

        self.assertFalse(result["passed"])
        evidence = result["reference_evidence"][
            "low_signal_texture_floor"
        ]
        self.assertFalse(evidence["applied"])
        self.assertEqual(
            evidence["reason"],
            "insufficient_long_blocking_fraction",
        )
        self.assertAlmostEqual(evidence["long_blocking_fraction"], 4 / 9)

    def test_low_signal_texture_floor_keeps_absolute_ceiling_fail_closed(
        self,
    ) -> None:
        self.configure_low_signal_grooming_reference_case()
        self.candidate_coarse_gradient_p90 = 0.1800

        result = self.audit()

        self.assertFalse(result["passed"])
        evidence = result["reference_evidence"][
            "low_signal_texture_floor"
        ]
        self.assertFalse(evidence["applied"])
        self.assertEqual(
            evidence["reason"],
            "candidate_texture_exceeds_floor_normalized_ceiling",
        )

    def test_low_signal_texture_floor_keeps_contour_expansion_fail_closed(
        self,
    ) -> None:
        self.configure_low_signal_grooming_reference_case()
        self.contour_relative.append("contour_fringe_expanded_vs_profile")

        result = self.audit()

        self.assertFalse(result["passed"])
        evidence = result["reference_evidence"][
            "low_signal_texture_floor"
        ]
        self.assertFalse(evidence["applied"])
        self.assertIn(
            "contour_fringe_expanded_vs_profile",
            result["frames"][0]["decision"]["blocking_issues"],
        )

    def test_sparse_coarse_low_signal_grooming_composite_is_normalized(
        self,
    ) -> None:
        self.configure_sparse_coarse_low_signal_grooming_case()

        result = self.audit()

        self.assertTrue(result["passed"])
        self.assertEqual(result["failed_frame_indices"], [])
        evidence = result["reference_evidence"][
            "sparse_coarse_low_signal_texture_floor"
        ]
        self.assertTrue(evidence["applied"])
        self.assertEqual(evidence["blocking_frame_indices"], [0, 1, 4])
        self.assertEqual(evidence["downgraded_sample_indices"], [0, 1, 4])
        self.assertAlmostEqual(evidence["blocking_sample_fraction"], 1 / 3)
        self.assertAlmostEqual(evidence["candidate_coarse_ceiling"], 0.1942)
        self.assertAlmostEqual(evidence["candidate_long_ceiling"], 0.189)
        self.assertLess(
            evidence["candidate_coarse_temporal_spread_ratio"],
            1.20,
        )
        self.assertLess(
            evidence["candidate_long_temporal_spread_ratio"],
            1.20,
        )
        for frame_index in (0, 1, 4):
            decision = result["frames"][frame_index]["decision"]
            self.assertTrue(decision["passed"])
            self.assertEqual(decision["blocking_issues"], [])
            self.assertTrue(
                decision[
                    "sparse_coarse_low_signal_action_reference_floor"
                ]["coarse_blocking_before_floor"]
            )

    def test_sparse_coarse_low_signal_keeps_ceiling_failure_closed(
        self,
    ) -> None:
        self.configure_sparse_coarse_low_signal_grooming_case()
        self.frame_candidate_coarse_gradient_p90[0] = 0.195

        result = self.audit()

        self.assertFalse(result["passed"])
        evidence = result["reference_evidence"][
            "sparse_coarse_low_signal_texture_floor"
        ]
        self.assertFalse(evidence["applied"])
        self.assertEqual(
            evidence["reason"],
            "candidate_texture_exceeds_sparse_floor_normalized_ceiling",
        )

    def test_sparse_coarse_low_signal_keeps_dense_coarse_failure_closed(
        self,
    ) -> None:
        self.configure_sparse_coarse_low_signal_grooming_case()
        pair = [
            "coarse_strokes_expanded_vs_profile",
            "long_fur_energy_expanded_vs_profile",
        ]
        self.frame_texture_relative_overrides = {
            index: list(pair) for index in range(9)
        }
        self.frame_contour_relative_overrides = {
            index: list(pair) for index in range(9)
        }

        result = self.audit()

        self.assertFalse(result["passed"])
        evidence = result["reference_evidence"][
            "sparse_coarse_low_signal_texture_floor"
        ]
        self.assertFalse(evidence["applied"])
        self.assertEqual(evidence["reason"], "coarse_blocking_not_sparse")

    def test_sparse_coarse_low_signal_requires_full_long_observation(
        self,
    ) -> None:
        self.configure_sparse_coarse_low_signal_grooming_case()
        self.frame_texture_relative_overrides[8] = []
        self.frame_contour_relative_overrides[8] = []

        result = self.audit()

        self.assertFalse(result["passed"])
        evidence = result["reference_evidence"][
            "sparse_coarse_low_signal_texture_floor"
        ]
        self.assertFalse(evidence["applied"])
        self.assertEqual(
            evidence["reason"],
            "sample_surface_relative_issue_contract_mismatch",
        )

    def test_direct_qwen_sparse_coarse_grooming_is_temporally_normalized(
        self,
    ) -> None:
        self.configure_direct_qwen_sparse_coarse_grooming_case()

        result = self.audit()

        self.assertTrue(result["passed"])
        self.assertEqual(result["failed_frame_indices"], [])
        evidence = result["reference_evidence"][
            "direct_qwen_sparse_coarse_temporal_floor"
        ]
        self.assertTrue(evidence["applied"])
        self.assertEqual(evidence["blocking_frame_indices"], [0, 1, 3])
        self.assertEqual(evidence["downgraded_sample_indices"], [0, 1, 3])
        self.assertEqual(
            evidence["long_observation_frame_indices"],
            [0, 1, 3, 4, 5, 6],
        )
        self.assertEqual(evidence["clean_surface_frame_indices"], [2, 7, 8])
        self.assertAlmostEqual(evidence["blocking_sample_fraction"], 1 / 3)
        self.assertAlmostEqual(evidence["candidate_coarse_ceiling"], 0.23277947)
        self.assertAlmostEqual(evidence["candidate_long_ceiling"], 0.18354357)
        self.assertLess(
            evidence["candidate_coarse_temporal_spread_ratio"],
            1.20,
        )
        self.assertLess(
            evidence["candidate_long_temporal_spread_ratio"],
            1.20,
        )
        for frame_index in (0, 1, 3):
            decision = result["frames"][frame_index]["decision"]
            self.assertTrue(decision["passed"])
            self.assertEqual(decision["blocking_issues"], [])
            self.assertTrue(
                decision[
                    "direct_qwen_sparse_coarse_temporal_corroboration"
                ]["coarse_blocking_before_corroboration"]
            )

    def test_direct_qwen_sparse_coarse_keeps_derived_ceiling_fail_closed(
        self,
    ) -> None:
        self.configure_direct_qwen_sparse_coarse_grooming_case()
        self.frame_candidate_coarse_gradient_p90[0] = 0.233

        result = self.audit()

        self.assertFalse(result["passed"])
        evidence = result["reference_evidence"][
            "direct_qwen_sparse_coarse_temporal_floor"
        ]
        self.assertFalse(evidence["applied"])
        self.assertEqual(
            evidence["reason"],
            "candidate_texture_exceeds_direct_qwen_temporal_ceiling",
        )

    def test_direct_qwen_sparse_coarse_keeps_dense_failure_closed(
        self,
    ) -> None:
        self.configure_direct_qwen_sparse_coarse_grooming_case()
        pair = [
            "coarse_strokes_expanded_vs_profile",
            "long_fur_energy_expanded_vs_profile",
        ]
        self.frame_texture_relative_overrides = {
            index: list(pair) for index in range(5)
        }
        self.frame_contour_relative_overrides = {
            index: list(pair) for index in range(5)
        }

        result = self.audit()

        self.assertFalse(result["passed"])
        evidence = result["reference_evidence"][
            "direct_qwen_sparse_coarse_temporal_floor"
        ]
        self.assertFalse(evidence["applied"])
        self.assertEqual(evidence["reason"], "coarse_blocking_not_sparse")

    def test_direct_qwen_sparse_coarse_requires_long_pair_on_blockers(
        self,
    ) -> None:
        self.configure_direct_qwen_sparse_coarse_grooming_case()
        self.frame_texture_relative_overrides[0] = [
            "coarse_strokes_expanded_vs_profile"
        ]
        self.frame_contour_relative_overrides[0] = [
            "coarse_strokes_expanded_vs_profile"
        ]

        result = self.audit()

        self.assertFalse(result["passed"])
        evidence = result["reference_evidence"][
            "direct_qwen_sparse_coarse_temporal_floor"
        ]
        self.assertFalse(evidence["applied"])
        self.assertEqual(
            evidence["reason"],
            "sample_blocking_surface_pair_contract_mismatch",
        )

    def test_direct_qwen_sparse_coarse_does_not_replace_composite_policy(
        self,
    ) -> None:
        self.configure_direct_qwen_sparse_coarse_grooming_case()
        self.grooming_composite_recovery = grooming_composite_recovery()

        result = self.audit()

        self.assertFalse(result["passed"])
        evidence = result["reference_evidence"][
            "direct_qwen_sparse_coarse_temporal_floor"
        ]
        self.assertFalse(evidence["applied"])
        self.assertEqual(
            evidence["reason"],
            "direct_qwen_reference_has_composite_recovery",
        )

    def test_eating_contiguous_action_phase_uses_clean_boundary_anchor(
        self,
    ) -> None:
        self.configure_eating_action_phase_boundary_case()

        result = self.audit()

        self.assertTrue(result["passed"])
        self.assertEqual(result["failed_frame_indices"], [])
        evidence = result["temporal_evidence"][
            "eating_loop_boundary_anchor"
        ]
        self.assertTrue(evidence["applied"])
        self.assertEqual(evidence["boundary_frame_indices"], [0, 8])
        self.assertEqual(evidence["blocking_frame_indices"], [2, 3, 4])
        self.assertEqual(evidence["downgraded_sample_indices"], [2, 3, 4])
        self.assertLess(evidence["anchor_coarse_ratio"], 1.28)
        self.assertLess(evidence["anchor_long_ratio"], 1.35)
        self.assertLess(evidence["anchor_long_delta"], 0.045)
        for frame_index in (2, 3, 4):
            decision = result["frames"][frame_index]["decision"]
            self.assertTrue(decision["passed"])
            self.assertEqual(decision["blocking_issues"], [])
            self.assertTrue(
                decision["long_fur_relative_evidence"][
                    "eating_action_phase_boundary_anchor_observation"
                ]
            )

    def test_eating_isolated_initial_radial_hit_uses_clean_loop_endpoint(
        self,
    ) -> None:
        self.configure_eating_initial_radial_boundary_case()

        result = self.audit()

        self.assertTrue(result["passed"])
        self.assertEqual(result["failed_frame_indices"], [])
        evidence = result["temporal_evidence"][
            "eating_initial_radial_loop_boundary"
        ]
        self.assertTrue(evidence["applied"])
        self.assertEqual(evidence["boundary_frame_indices"], [0, 8])
        self.assertEqual(evidence["blocking_frame_indices"], [0])
        self.assertEqual(evidence["downgraded_sample_indices"], [0])
        self.assertLess(evidence["radial_load_spread_ratio"], 1.10)
        self.assertLess(evidence["radial_energy_spread_ratio"], 1.10)
        self.assertLess(evidence["radial_fraction_spread_ratio"], 1.20)
        radial = result["frames"][0]["decision"][
            "radial_relative_evidence"
        ]
        self.assertTrue(
            radial["blocking_before_eating_loop_endpoint_corroboration"]
        )
        self.assertTrue(
            radial["eating_initial_radial_loop_endpoint_observation"]
        )
        self.assertFalse(radial["blocking"])

    def test_eating_initial_radial_keeps_unstable_loop_endpoint_fail_closed(
        self,
    ) -> None:
        self.configure_eating_initial_radial_boundary_case()
        self.frame_candidate_radial_energy[8] = 0.45

        result = self.audit()

        self.assertFalse(result["passed"])
        evidence = result["temporal_evidence"][
            "eating_initial_radial_loop_boundary"
        ]
        self.assertFalse(evidence["applied"])
        self.assertEqual(
            evidence["reason"],
            "radial_boundary_load_or_energy_not_stable",
        )
        self.assertEqual(result["failed_frame_indices"], [0])

    def test_eating_initial_radial_preserves_absolute_radial_failure(
        self,
    ) -> None:
        self.configure_eating_initial_radial_boundary_case()
        self.texture_absolute = ["radial_brush_texture"]

        result = self.audit()

        self.assertFalse(result["passed"])
        evidence = result["temporal_evidence"][
            "eating_initial_radial_loop_boundary"
        ]
        self.assertFalse(evidence["applied"])
        self.assertEqual(
            evidence["reason"],
            "initial_radial_evidence_contract_mismatch",
        )
        self.assertEqual(result["failed_frame_indices"], [0])

    def test_eating_recurrent_radial_hits_remain_fail_closed(self) -> None:
        self.configure_eating_initial_radial_boundary_case()
        self.texture_issue_frame_indices = {0, 1}
        self.contour_issue_frame_indices = {0, 1}

        result = self.audit()

        self.assertFalse(result["passed"])
        evidence = result["temporal_evidence"][
            "eating_initial_radial_loop_boundary"
        ]
        self.assertFalse(evidence["applied"])
        self.assertEqual(
            evidence["reason"],
            "blocking_contract_is_not_single_initial_sample",
        )
        self.assertEqual(result["failed_frame_indices"], [0, 1])

    def test_eating_boundary_anchor_keeps_strong_anchor_growth_fail_closed(
        self,
    ) -> None:
        self.configure_eating_action_phase_boundary_case()
        self.frame_candidate_long_energy[2] = 0.23

        result = self.audit()

        self.assertFalse(result["passed"])
        evidence = result["temporal_evidence"][
            "eating_loop_boundary_anchor"
        ]
        self.assertFalse(evidence["applied"])
        self.assertEqual(
            evidence["reason"],
            "action_phase_remains_strong_against_boundary_anchor",
        )

    def test_eating_boundary_anchor_requires_contiguous_interior_run(
        self,
    ) -> None:
        self.configure_eating_action_phase_boundary_case()
        self.texture_issue_frame_indices = {2, 4}
        self.contour_issue_frame_indices = {2, 4}
        self.frame_texture_relative_overrides.pop(3, None)

        result = self.audit()

        self.assertFalse(result["passed"])
        evidence = result["temporal_evidence"][
            "eating_loop_boundary_anchor"
        ]
        self.assertFalse(evidence["applied"])
        self.assertEqual(
            evidence["reason"],
            "blocking_samples_not_contiguous_interior_run",
        )

    def test_eating_isolated_strong_long_peak_uses_adjacent_pose_phase(
        self,
    ) -> None:
        self.configure_eating_isolated_strong_long_fur_case()

        result = self.audit()

        self.assertTrue(result["passed"])
        evidence = result["temporal_evidence"][
            "eating_isolated_long_fur_peak"
        ]
        self.assertTrue(evidence["applied"])
        self.assertEqual(evidence["downgraded_sample_indices"], [6])
        self.assertEqual(evidence["selected_supporting_frame_index"], 5)
        long_fur = result["frames"][6]["decision"][
            "long_fur_relative_evidence"
        ]
        self.assertTrue(long_fur["blocking_before_temporal_corroboration"])
        self.assertTrue(long_fur["isolated_eating_pose_phase_observation"])
        self.assertFalse(long_fur["blocking"])

    def test_eating_recurrent_strong_long_peak_remains_fail_closed(
        self,
    ) -> None:
        self.configure_eating_isolated_strong_long_fur_case()
        self.frame_candidate_long_energy[5] = 0.21946399
        self.contour_issue_frame_indices = {5, 6}

        result = self.audit()

        self.assertFalse(result["passed"])
        evidence = result["temporal_evidence"][
            "eating_isolated_long_fur_peak"
        ]
        self.assertFalse(evidence["applied"])
        self.assertEqual(
            evidence["reason"],
            "blocking_contract_is_not_single_sample",
        )
        self.assertEqual(result["failed_frame_indices"], [5, 6])

    def test_eating_isolated_long_peak_preserves_delta_margin_ceiling(
        self,
    ) -> None:
        self.configure_eating_isolated_strong_long_fur_case()
        self.frame_candidate_long_energy[6] = 0.223

        result = self.audit()

        self.assertFalse(result["passed"])
        evidence = result["temporal_evidence"][
            "eating_isolated_long_fur_peak"
        ]
        self.assertFalse(evidence["applied"])
        self.assertEqual(
            evidence["reason"],
            "isolated_peak_exceeds_existing_delta_margin_or_absolute_ceiling",
        )

    def test_eating_isolated_long_peak_preserves_absolute_long_failure(
        self,
    ) -> None:
        self.configure_eating_isolated_strong_long_fur_case()
        self.texture_absolute = ["coherent_long_fur_energy"]

        result = self.audit()

        self.assertFalse(result["passed"])
        evidence = result["temporal_evidence"][
            "eating_isolated_long_fur_peak"
        ]
        self.assertFalse(evidence["applied"])
        self.assertEqual(
            evidence["reason"],
            "sample_absolute_long_or_other_surface_issue_present",
        )

    def test_eating_isolated_long_peak_allows_prop_excluded_only_peak(
        self,
    ) -> None:
        self.configure_eating_isolated_strong_long_fur_case()
        self.contour_issue_frame_indices = set()

        result = self.audit()

        self.assertTrue(result["passed"])
        evidence = result["temporal_evidence"][
            "eating_isolated_long_fur_peak"
        ]
        self.assertTrue(evidence["applied"])
        self.assertFalse(evidence["blocking_frame_unmasked_corroborated"])

    def test_eating_independent_long_and_initial_radial_peaks_are_bounded(
        self,
    ) -> None:
        self.configure_eating_combined_isolated_surface_peaks()

        result = self.audit()

        self.assertTrue(result["passed"])
        long_evidence = result["temporal_evidence"][
            "eating_isolated_long_fur_peak"
        ]
        self.assertTrue(long_evidence["applied"])
        self.assertEqual(
            long_evidence["companion_initial_radial_frame_indices"],
            [0],
        )
        radial_evidence = result["temporal_evidence"][
            "eating_initial_radial_bounded_reference"
        ]
        self.assertTrue(radial_evidence["applied"])
        self.assertEqual(radial_evidence["downgraded_sample_indices"], [0])
        self.assertFalse(
            radial_evidence["initial_unmasked_radial_corroborated"]
        )

    def test_eating_initial_radial_bounded_reference_ceiling_fails_closed(
        self,
    ) -> None:
        self.configure_eating_combined_isolated_surface_peaks()
        self.frame_candidate_radial_fraction[0] = 0.42

        result = self.audit()

        self.assertFalse(result["passed"])
        radial_evidence = result["temporal_evidence"][
            "eating_initial_radial_bounded_reference"
        ]
        self.assertFalse(radial_evidence["applied"])
        self.assertEqual(
            radial_evidence["reason"],
            "initial_radial_peak_exceeds_bounded_reference",
        )

    def test_mask_level_long_line_growth_is_an_observation_only(self) -> None:
        self.texture_relative = ["long_fur_energy_expanded_vs_profile"]
        self.candidate_long_energy = 0.18590406
        self.baseline_long_energy = 0.14705206
        result = self.audit()
        self.assertTrue(result["passed"])
        for frame in result["frames"]:
            evidence = frame["decision"]["long_fur_relative_evidence"]
            self.assertAlmostEqual(evidence["energy_ratio"], 1.2642, places=3)
            self.assertTrue(evidence["mask_or_decode_level_observation"])
            self.assertFalse(evidence["blocking"])

    def test_dual_texture_long_line_growth_without_contour_is_observation(self) -> None:
        issue = "long_fur_energy_expanded_vs_profile"
        self.texture_relative = [issue]
        self.contour_relative = [issue]
        self.candidate_long_energy = 0.18590406
        self.baseline_long_energy = 0.14705206
        result = self.audit()
        self.assertTrue(result["passed"])
        for frame in result["frames"]:
            evidence = frame["decision"]["long_fur_relative_evidence"]
            self.assertTrue(evidence["unmasked_corroborated"])
            self.assertFalse(evidence["contour_corroborated"])
            self.assertTrue(evidence["texture_only_pose_or_decode_observation"])
            self.assertNotIn(issue, frame["decision"]["blocking_issues"])

    def test_absolute_long_line_issue_without_contour_is_observation(self) -> None:
        issue = "long_fur_energy_expanded_vs_profile"
        self.texture_relative = [issue]
        self.texture_absolute = ["coherent_long_fur_energy"]
        self.candidate_long_energy = 0.18590406
        self.baseline_long_energy = 0.14705206
        result = self.audit()
        self.assertTrue(result["passed"])
        for frame in result["frames"]:
            evidence = frame["decision"]["long_fur_relative_evidence"]
            self.assertTrue(evidence["absolute_long_issue"])
            self.assertFalse(evidence["contour_corroborated"])
            self.assertNotIn(issue, frame["decision"]["blocking_issues"])

    def test_long_line_growth_with_contour_expansion_fails_closed(self) -> None:
        issue = "long_fur_energy_expanded_vs_profile"
        self.texture_relative = [issue]
        self.contour_relative = [
            issue,
            "contour_fringe_expanded_vs_profile",
        ]
        self.candidate_long_energy = 0.18590406
        self.baseline_long_energy = 0.14705206
        result = self.audit()
        self.assertFalse(result["passed"])
        for frame in result["frames"]:
            evidence = frame["decision"]["long_fur_relative_evidence"]
            self.assertTrue(evidence["contour_corroborated"])
            self.assertIn(issue, frame["decision"]["blocking_issues"])

    def test_pose_concentrated_long_line_regression_is_observation(self) -> None:
        issue = "long_fur_energy_expanded_vs_profile"
        self.texture_relative = [issue]
        self.contour_relative = [issue]
        self.texture_absolute = ["coherent_long_fur_energy"]
        self.candidate_long_energy = 0.24243586
        self.baseline_long_energy = 0.18181148
        result = self.audit()
        self.assertTrue(result["passed"])
        for frame in result["frames"]:
            evidence = frame["decision"]["long_fur_relative_evidence"]
            self.assertAlmostEqual(evidence["energy_ratio"], 1.33345, places=4)
            self.assertTrue(evidence["absolute_long_issue"])
            self.assertTrue(evidence["unmasked_corroborated"])
            self.assertFalse(evidence["strong_prop_excluded_expansion"])
            self.assertFalse(evidence["contour_corroborated"])
            self.assertFalse(evidence["blocking"])

    def test_pose_aligned_coherence_without_load_growth_is_observation(self) -> None:
        issue = "coherent_brush_texture_expanded_vs_profile"
        self.texture_relative = [issue]
        self.texture_absolute = ["brush_texture_coherence"]
        self.candidate_coherent_fraction = 0.60256863
        self.baseline_coherent_fraction = 0.45498824
        self.candidate_coarse_gradient_p90 = 0.16834296
        self.baseline_coarse_gradient_p90 = 0.19041804
        self.candidate_long_energy = 0.18556867
        self.baseline_long_energy = 0.18181148
        result = self.audit()
        self.assertTrue(result["passed"])
        for frame in result["frames"]:
            evidence = frame["decision"][
                "coherent_brush_relative_evidence"
            ]
            self.assertTrue(evidence["relative_issue_present"])
            self.assertTrue(evidence["absolute_coherent_issue"])
            self.assertFalse(evidence["load_expanded"])
            self.assertFalse(
                evidence["long_line_energy_backstop_triggered"]
            )
            self.assertTrue(
                evidence["coherence_without_load_expansion_observation"]
            )

    def test_coherent_brush_load_growth_fails_closed(self) -> None:
        issue = "coherent_brush_texture_expanded_vs_profile"
        self.texture_relative = [issue]
        self.texture_absolute = ["brush_texture_coherence"]
        self.candidate_coherent_fraction = 0.70
        self.baseline_coherent_fraction = 0.45
        self.candidate_coarse_gradient_p90 = 0.30
        self.baseline_coarse_gradient_p90 = 0.20
        self.candidate_long_energy = 0.36
        self.baseline_long_energy = 0.19
        result = self.audit()
        self.assertFalse(result["passed"])
        for frame in result["frames"]:
            evidence = frame["decision"][
                "coherent_brush_relative_evidence"
            ]
            self.assertTrue(evidence["load_expanded"])
            self.assertTrue(evidence["blocking"])
            self.assertIn(issue, frame["decision"]["blocking_issues"])

    def test_isolated_coherent_load_growth_without_long_line_backstop_is_observation(
        self,
    ) -> None:
        issue = "coherent_brush_texture_expanded_vs_profile"
        self.texture_relative = [issue]
        self.texture_absolute = ["brush_texture_coherence"]
        self.texture_issue_frame_indices = {0}
        self.candidate_coherent_fraction = 0.65455146
        self.baseline_coherent_fraction = 0.48205546
        self.candidate_coarse_gradient_p90 = 0.14793241
        self.baseline_coarse_gradient_p90 = 0.15628128
        self.candidate_long_energy = 0.17079558
        self.baseline_long_energy = 0.15347688

        result = self.audit()

        self.assertTrue(result["passed"])
        self.assertEqual(result["failed_frame_indices"], [])
        temporal = result["temporal_evidence"]["coherent_brush"]
        self.assertEqual(temporal["evidence_sample_indices"], [0])
        self.assertEqual(
            temporal[
                "isolated_load_only_samples_downgraded_to_observation"
            ],
            [0],
        )
        evidence = result["frames"][0]["decision"][
            "coherent_brush_relative_evidence"
        ]
        self.assertTrue(evidence["load_expanded"])
        self.assertFalse(evidence["long_line_energy_backstop_triggered"])
        self.assertFalse(evidence["temporally_corroborated"])
        self.assertTrue(evidence["isolated_load_only_temporal_observation"])
        self.assertFalse(evidence["blocking"])

    def test_recurrent_coherent_load_growth_without_long_line_backstop_fails_closed(
        self,
    ) -> None:
        issue = "coherent_brush_texture_expanded_vs_profile"
        self.texture_relative = [issue]
        self.texture_absolute = ["brush_texture_coherence"]
        self.texture_issue_frame_indices = {0, 1}
        self.candidate_coherent_fraction = 0.65455146
        self.baseline_coherent_fraction = 0.48205546
        self.candidate_coarse_gradient_p90 = 0.14793241
        self.baseline_coarse_gradient_p90 = 0.15628128
        self.candidate_long_energy = 0.17079558
        self.baseline_long_energy = 0.15347688

        result = self.audit()

        self.assertFalse(result["passed"])
        self.assertEqual(result["failed_frame_indices"], [0, 1])
        temporal = result["temporal_evidence"]["coherent_brush"]
        self.assertEqual(temporal["evidence_sample_indices"], [0, 1])
        self.assertTrue(temporal["temporally_corroborated"])
        self.assertEqual(
            temporal[
                "isolated_load_only_samples_downgraded_to_observation"
            ],
            [],
        )

    def test_single_sample_coherent_long_line_backstop_remains_fail_closed(
        self,
    ) -> None:
        issue = "coherent_brush_texture_expanded_vs_profile"
        self.texture_relative = [issue]
        self.texture_absolute = ["brush_texture_coherence"]
        self.texture_issue_frame_indices = {0}
        self.candidate_coherent_fraction = 0.65455146
        self.baseline_coherent_fraction = 0.48205546
        self.candidate_coarse_gradient_p90 = 0.14793241
        self.baseline_coarse_gradient_p90 = 0.15628128
        self.candidate_long_energy = 0.21
        self.baseline_long_energy = 0.15

        result = self.audit()

        self.assertFalse(result["passed"])
        self.assertEqual(result["failed_frame_indices"], [0])
        evidence = result["frames"][0]["decision"][
            "coherent_brush_relative_evidence"
        ]
        self.assertTrue(evidence["long_line_energy_backstop_triggered"])
        self.assertFalse(evidence["temporally_corroborated"])
        self.assertFalse(evidence["isolated_load_only_temporal_observation"])
        self.assertTrue(evidence["blocking"])

    def test_pose_sensitive_radial_fraction_without_load_growth_is_observation_only(self) -> None:
        self.texture_relative = ["radial_brush_texture_expanded_vs_profile"]
        result = self.audit()
        self.assertTrue(result["passed"])
        for frame in result["frames"]:
            evidence = frame["decision"]["radial_relative_evidence"]
            self.assertTrue(evidence["relative_issue_present"])
            self.assertAlmostEqual(evidence["load_ratio"], 0.8538, places=3)
            self.assertFalse(evidence["load_expanded"])
            self.assertFalse(evidence["normalized_energy_backstop_triggered"])
            self.assertTrue(
                evidence["radial_fraction_without_load_expansion_observation"]
            )
            self.assertNotIn(
                "radial_brush_texture_expanded_vs_profile",
                frame["decision"]["blocking_issues"],
            )

    def test_historical_radial_bad_sample_still_fails_on_load_growth(self) -> None:
        self.texture_relative = ["radial_brush_texture_expanded_vs_profile"]
        self.candidate_radial_fraction = 0.38184591
        self.baseline_radial_fraction = 0.30365508
        self.candidate_gradient_p90 = 0.48414860
        self.baseline_gradient_p90 = 0.38332589
        self.candidate_radial_energy = 0.48441005
        self.baseline_radial_energy = 0.44840063
        result = self.audit()
        self.assertFalse(result["passed"])
        for frame in result["frames"]:
            evidence = frame["decision"]["radial_relative_evidence"]
            self.assertTrue(evidence["load_expanded"])
            self.assertFalse(evidence["normalized_energy_backstop_triggered"])
            self.assertGreater(evidence["load_ratio"], 1.48)
            self.assertIn(
                "radial_brush_texture_expanded_vs_profile",
                frame["decision"]["blocking_issues"],
            )

    def test_decode_level_radial_load_growth_below_ratio_is_observation(self) -> None:
        self.texture_relative = ["radial_brush_texture_expanded_vs_profile"]
        self.candidate_radial_fraction = 0.43528125
        self.baseline_radial_fraction = 0.35073403
        self.candidate_gradient_p90 = 0.23371542
        self.baseline_gradient_p90 = 0.23898765
        result = self.audit()
        self.assertTrue(result["passed"])
        for frame in result["frames"]:
            evidence = frame["decision"]["radial_relative_evidence"]
            self.assertTrue(evidence["raw_load_grew"])
            self.assertAlmostEqual(evidence["load_ratio"], 1.21368, places=4)
            self.assertFalse(evidence["load_expanded"])
            self.assertFalse(evidence["blocking"])

    def test_radial_energy_backstop_fails_when_load_did_not_grow(self) -> None:
        self.texture_relative = ["radial_brush_texture_expanded_vs_profile"]
        self.candidate_radial_energy = 0.58
        self.baseline_radial_energy = 0.45
        result = self.audit()
        self.assertFalse(result["passed"])
        for frame in result["frames"]:
            evidence = frame["decision"]["radial_relative_evidence"]
            self.assertFalse(evidence["load_expanded"])
            self.assertTrue(evidence["normalized_energy_backstop_triggered"])
            self.assertIn(
                "radial_brush_texture_expanded_vs_profile",
                frame["decision"]["blocking_issues"],
            )

    def test_absolute_radial_issue_blocks_relative_fraction_growth(self) -> None:
        self.texture_relative = ["radial_brush_texture_expanded_vs_profile"]
        self.texture_absolute = ["radial_brush_texture"]
        result = self.audit()
        self.assertFalse(result["passed"])
        for frame in result["frames"]:
            evidence = frame["decision"]["radial_relative_evidence"]
            self.assertTrue(evidence["absolute_radial_issue"])
            self.assertTrue(evidence["blocking"])

    def test_absolute_radial_issue_blocks_without_relative_radial_issue(self) -> None:
        self.texture_absolute = ["radial_brush_texture"]
        result = self.audit()
        self.assertFalse(result["passed"])
        for frame in result["frames"]:
            evidence = frame["decision"]["radial_relative_evidence"]
            self.assertFalse(evidence["relative_issue_present"])
            self.assertTrue(evidence["absolute_radial_issue"])
            self.assertIn(
                "radial_brush_texture",
                frame["decision"]["blocking_issues"],
            )

    def test_missing_radial_load_metric_fails_closed(self) -> None:
        self.texture_relative = ["radial_brush_texture_expanded_vs_profile"]
        self.candidate_gradient_p90 = None
        result = self.audit()
        self.assertFalse(result["passed"])
        for frame in result["frames"]:
            self.assertEqual(
                frame["decision"]["blocking_issues"],
                ["frame_audit_error"],
            )
            self.assertIn("radial_relative_evidence_metrics_invalid", frame["error"])

    def test_non_finite_radial_load_metric_fails_closed(self) -> None:
        self.texture_relative = ["radial_brush_texture_expanded_vs_profile"]
        self.candidate_gradient_p90 = float("nan")
        result = self.audit()
        self.assertFalse(result["passed"])
        for frame in result["frames"]:
            self.assertIn("radial_relative_evidence_metrics_non_finite", frame["error"])

    def test_zero_radial_baseline_load_fails_closed(self) -> None:
        self.texture_relative = ["radial_brush_texture_expanded_vs_profile"]
        self.baseline_gradient_p90 = 0.0
        result = self.audit()
        self.assertFalse(result["passed"])
        for frame in result["frames"]:
            self.assertIn("radial_relative_evidence_metrics_out_of_range", frame["error"])

    def test_derived_texture_baseline_absolute_issue_is_observation(self) -> None:
        self.texture_baseline_issues = ["radial_long_lines"]
        result = self.audit()
        self.assertTrue(result["passed"])
        for frame in result["frames"]:
            self.assertEqual(
                frame["decision"]["baseline_absolute_observations"][
                    "hash_bound_prop_excluded_reference"
                ],
                ["radial_long_lines"],
            )

    def test_unmasked_action_reference_baseline_issue_fails_closed(self) -> None:
        self.contour_baseline_issues = ["radial_brush_texture"]
        result = self.audit()
        self.assertFalse(result["passed"])
        for frame in result["frames"]:
            self.assertIn(
                "surface_evaluation_contract_unreliable",
                frame["decision"]["blocking_issues"],
            )

    def test_malformed_texture_baseline_issues_contract_fails_closed(self) -> None:
        self.texture_baseline_issues_malformed = "not-a-list"
        result = self.audit()
        self.assertFalse(result["passed"])
        for frame in result["frames"]:
            self.assertEqual(
                frame["decision"]["blocking_issues"],
                ["frame_audit_error"],
            )
            self.assertIn(
                "fur_surface_evaluation_baseline_issues_invalid",
                frame["error"],
            )

    def test_unmasked_contour_and_unknown_relative_issues_fail_closed(self) -> None:
        self.contour_relative = [
            "contour_fringe_expanded_vs_profile",
            "unexpected_future_relative_issue",
        ]
        result = self.audit()
        self.assertFalse(result["passed"])
        for frame in result["frames"]:
            self.assertIn(
                "contour_fringe_expanded_vs_profile",
                frame["decision"]["blocking_issues"],
            )
            self.assertIn(
                "unexpected_future_relative_issue",
                frame["decision"]["blocking_issues"],
            )

    def test_unreliable_identity_mask_fails_every_frame_before_frame_evaluation(self) -> None:
        self.mask_reliable = False
        result = self.audit()
        self.assertFalse(result["passed"])
        self.assertEqual(result["failed_frame_indices"], [0, 1, 2])
        self.assertEqual(self.frame_evaluator_calls, 0)
        for frame in result["frames"]:
            self.assertIn("identity_mask_unreliable", frame["error"])

    def test_static_black_and_pose_white_identity_mask_contracts_are_distinct(self) -> None:
        pose_frame = self.root / "pose_frame.png"
        save_identity_mask(pose_frame, background="white")
        static_quality = validate_identity_mask_frame(
            self.reference_mask,
            action="eating",
            background="black",
        )
        pose_quality = validate_identity_mask_frame(
            pose_frame,
            action="eating",
            background="white",
        )
        self.assertTrue(static_quality["reliable"])
        self.assertTrue(pose_quality["reliable"])
        self.assertEqual(static_quality["background"], "black")
        self.assertEqual(pose_quality["background"], "white")

    def test_mask_sha_mismatch_stops_before_any_frame_work(self) -> None:
        with self.assertRaisesRegex(ValueError, "pose_identity_mask_video_sha256_mismatch"):
            audit_final_video(
                self.video,
                action="eating",
                baseline_path=self.profile,
                baseline_sha256=sha256_file(self.profile),
                baseline_provenance={},
                comparison_scope=VALIDATED_ACTION_SURFACE_SCOPE,
                validated_action_reference_path=self.reference,
                validated_action_reference_sha256=sha256_file(self.reference),
                pose_identity_mask_video_path=self.pose_mask_video,
                pose_identity_mask_video_sha256=hashlib.sha256(b"changed").hexdigest(),
                reference_identity_mask_path=self.reference_mask,
                reference_identity_mask_sha256=sha256_file(self.reference_mask),
                evaluator=self.evaluator,
                frame_extractor=self.extract,
                video_probe=self.probe,
                background_remover=self.remove_background,
            )
        self.assertEqual(self.frame_evaluator_calls, 0)


if __name__ == "__main__":
    unittest.main()
