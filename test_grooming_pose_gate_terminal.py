from __future__ import annotations

import unittest

import cv2
import numpy as np

import grooming_pose_gate as gate


class GroomingTerminalPawSamplingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.thresholds = gate.GroomingPoseThresholds()
        self.foreleg = (40, 200, 160, 360)
        self.planted = (190, 380, 310, 480)
        self.sampling = gate._terminal_paw_boxes(
            self.foreleg,
            self.planted,
            {"top_y": 210},
            self.thresholds,
        )

    def make_patterned_foreleg_fixture(
        self,
        *,
        raised_paw_value: int = 230,
        planted_paw_value: int = 230,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        image = np.zeros(
            (
                self.thresholds.normalized_height,
                self.thresholds.normalized_width,
                3,
            ),
            dtype=np.uint8,
        )
        mask = np.zeros(image.shape[:2], dtype=bool)
        for box in (self.foreleg, self.planted):
            x0, y0, x1, y1 = box
            mask[y0:y1, x0:x1] = True
            image[y0:y1, x0:x1] = 24

        raised_x0, raised_y0, raised_x1, raised_y1 = self.sampling["raised"]
        image[raised_y0:raised_y1, raised_x0:raised_x1] = raised_paw_value
        planted_x0, planted_y0, planted_x1, planted_y1 = self.sampling["planted"]
        image[planted_y0:planted_y1, planted_x0:planted_x1] = planted_paw_value
        return image, mask, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def test_policy_uses_terminal_paw_bands_and_small_contact_tolerance(self) -> None:
        self.assertEqual(gate.POLICY, "main_case_grooming_raised_paw_cv_v3")
        self.assertEqual(gate.GATE_VERSION, "3.0")
        self.assertEqual(self.sampling["policy"], "terminal_paw_bands_v2")
        self.assertEqual(self.sampling["band_height_pixels"], 56)
        self.assertEqual(self.sampling["raised"], [40, 206, 160, 262])
        self.assertEqual(self.sampling["planted"], [190, 424, 310, 480])
        self.assertEqual(self.thresholds.paw_muzzle_gap_max, 0.040)
        self.assertEqual(self.thresholds.structural_blur_sigma, 4.0)
        self.assertEqual(self.thresholds.structural_edge_density_min, 0.015)

    def test_lowpass_structure_rejects_fur_texture_without_a_limb(self) -> None:
        gray = np.full(
            (
                self.thresholds.normalized_height,
                self.thresholds.normalized_width,
            ),
            190,
            dtype=np.uint8,
        )
        mask = np.ones(gray.shape, dtype=bool)
        x0, y0, x1, y1 = self.foreleg
        for x in range(x0, x1, 2):
            gray[y0:y1, x : x + 1] = 120

        evidence = gate._structural_foreleg_evidence(
            gray,
            mask,
            self.foreleg,
            self.thresholds,
        )

        self.assertTrue(evidence["available"])
        self.assertLess(
            evidence["edge_density"],
            self.thresholds.structural_edge_density_min,
        )

    def test_lowpass_structure_preserves_an_organic_foreleg_boundary(self) -> None:
        gray = np.full(
            (
                self.thresholds.normalized_height,
                self.thresholds.normalized_width,
            ),
            190,
            dtype=np.uint8,
        )
        mask = np.ones(gray.shape, dtype=bool)
        points = np.asarray(
            [[70, 350], [78, 320], [88, 285], [104, 240]],
            dtype=np.int32,
        )
        cv2.polylines(
            gray,
            [points],
            False,
            110,
            32,
            lineType=cv2.LINE_AA,
        )
        cv2.circle(gray, (108, 230), 22, 110, -1, lineType=cv2.LINE_AA)

        evidence = gate._structural_foreleg_evidence(
            gray,
            mask,
            self.foreleg,
            self.thresholds,
        )

        self.assertTrue(evidence["available"])
        self.assertGreaterEqual(
            evidence["edge_density"],
            self.thresholds.structural_edge_density_min,
        )

    def test_patterned_foreleg_is_not_mistaken_for_mismatched_paw(self) -> None:
        image, mask, gray = self.make_patterned_foreleg_fixture()
        whole_limb = gate._appearance_consistency(
            image,
            mask,
            gray,
            self.foreleg,
            self.planted,
            self.thresholds,
        )
        terminal_paws = gate._appearance_consistency(
            image,
            mask,
            gray,
            self.sampling["raised"],
            self.sampling["planted"],
            self.thresholds,
        )
        self.assertGreater(
            whole_limb["lab_median_delta"],
            self.thresholds.lab_median_delta_max,
        )
        self.assertLessEqual(
            terminal_paws["lab_median_delta"],
            self.thresholds.lab_median_delta_max,
        )
        self.assertLessEqual(
            terminal_paws["lab_histogram_bhattacharyya"]["mean"],
            self.thresholds.lab_histogram_distance_max,
        )

    def test_true_terminal_paw_color_mismatch_still_fails(self) -> None:
        image, mask, gray = self.make_patterned_foreleg_fixture(
            raised_paw_value=24,
            planted_paw_value=230,
        )
        appearance = gate._appearance_consistency(
            image,
            mask,
            gray,
            self.sampling["raised"],
            self.sampling["planted"],
            self.thresholds,
        )
        self.assertTrue(appearance["available"])
        self.assertGreater(
            appearance["lab_median_delta"],
            self.thresholds.lab_median_delta_max,
        )

    def test_terminal_sampling_fails_closed_when_a_paw_is_missing(self) -> None:
        image, mask, gray = self.make_patterned_foreleg_fixture()
        planted_x0, planted_y0, planted_x1, planted_y1 = self.sampling["planted"]
        mask[planted_y0:planted_y1, planted_x0:planted_x1] = False
        appearance = gate._appearance_consistency(
            image,
            mask,
            gray,
            self.sampling["raised"],
            self.sampling["planted"],
            self.thresholds,
        )
        self.assertFalse(appearance["available"])
        self.assertEqual(
            appearance["reason"],
            "planted_paw_roi_foreground_too_small",
        )


if __name__ == "__main__":
    unittest.main()
