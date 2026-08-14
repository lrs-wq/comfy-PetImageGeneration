from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

import local_video_api as pipeline

from tests.integration.local_video_subject_replacement_test_support import (
    SubjectReplacementQwenReferenceTestBase,
)


class SubjectReplacementQwenQualityGateTests(SubjectReplacementQwenReferenceTestBase):
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


if __name__ == "__main__":
    unittest.main()
