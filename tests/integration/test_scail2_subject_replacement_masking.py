from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image, ImageDraw

import scail2_subject_replacement_test as scail


class EatingScenePropMaskTests(unittest.TestCase):
    def test_stationary_small_driver_bbox_jitter_is_accepted(self) -> None:
        areas = np.full(81, 5100.0, dtype=np.float64)
        centers = np.tile(np.asarray([[120.0, 394.5]]), (81, 1))
        centers[28:33, 0] = (118.0, 118.0, 118.5, 119.0, 119.0)
        centers[33:37] = (
            (120.0, 396.5),
            (120.0, 396.0),
            (120.5, 394.5),
            (119.5, 394.5),
        )

        stability = scail.evaluate_eating_prop_temporal_stability(areas, centers)

        self.assertTrue(stability["accepted"])
        self.assertFalse(stability["strict_step_passed"])
        self.assertTrue(stability["bounded_stationary_jitter_passed"])
        self.assertAlmostEqual(stability["center_step_max"], np.sqrt(5.0))
        self.assertLessEqual(stability["center_x_span"], 3.0)
        self.assertLessEqual(stability["center_y_span"], 3.0)

    def test_moving_bowl_is_not_hidden_by_jitter_tolerance(self) -> None:
        areas = np.full(20, 5100.0, dtype=np.float64)
        centers = np.column_stack(
            (np.arange(20, dtype=np.float64) * 2.0, np.full(20, 394.5))
        )

        stability = scail.evaluate_eating_prop_temporal_stability(areas, centers)

        self.assertFalse(stability["accepted"])
        self.assertFalse(stability["strict_step_passed"])
        self.assertFalse(stability["bounded_stationary_jitter_passed"])
        self.assertGreater(stability["center_x_span"], 3.0)

    @staticmethod
    def _normalized_frames() -> list[Image.Image]:
        frames: list[Image.Image] = []
        for index in range(9):
            frame = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
            draw = ImageDraw.Draw(frame)
            draw.ellipse((55, 140, 315, 380), fill=(90, 100, 110, 255))
            draw.rectangle((320, 336, 442, 378), fill=(220, 185, 135, 255))
            if index in {0, 1}:
                draw.ellipse((365, 326, 385, 344), fill=(20, 30, 210, 255))
            frames.append(frame)
        return frames

    @staticmethod
    def _source_frames() -> list[Image.Image]:
        frames: list[Image.Image] = []
        for index in range(9):
            frame = Image.new("RGBA", (640, 360), (0, 0, 0, 0))
            draw = ImageDraw.Draw(frame)
            draw.ellipse((80, 80, 390, 320), fill=(90, 100, 110, 255))
            draw.rectangle((395, 270, 540, 325), fill=(220, 185, 135, 255))
            if index in {0, 1}:
                draw.ellipse((445, 258, 465, 278), fill=(20, 30, 210, 255))
            frames.append(frame)
        return frames

    @staticmethod
    def _white_bowl_frames(*, with_bowl: bool = True) -> list[Image.Image]:
        frames: list[Image.Image] = []
        for index in range(13):
            frame = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
            draw = ImageDraw.Draw(frame)
            draw.ellipse((55, 80, 382, 438), fill=(38, 42, 48, 255))
            draw.ellipse((76, 310, 150, 440), fill=(218, 218, 214, 255))
            draw.ellipse((300, 310, 370, 440), fill=(215, 215, 212, 255))
            # A large, static right-side tail reproduces the old false match.
            draw.ellipse((350, 324, 456, 455), fill=(76, 78, 82, 255))
            if with_bowl:
                draw.rounded_rectangle(
                    (150, 388, 320, 455),
                    radius=20,
                    fill=(210, 208, 202, 255),
                )
                draw.ellipse((165, 370, 305, 398), fill=(151, 91, 38, 255))
                if index in {0, 1}:
                    # The dipping muzzle temporarily hides the food and rim.
                    draw.ellipse((220, 342, 270, 397), fill=(42, 45, 50, 255))
            else:
                # A tiny warm nose is not wide enough to qualify as food.
                draw.ellipse((230, 338, 246, 350), fill=(165, 92, 52, 255))
            frames.append(frame)
        return frames

    @staticmethod
    def _geometry_frames(
        *,
        low_muzzle: bool = True,
        static_contact_paw: bool = False,
        full_body_motion: bool = False,
    ) -> list[Image.Image]:
        frames: list[Image.Image] = []
        for index in range(13):
            frame = Image.new("RGBA", (640, 360), (0, 0, 0, 0))
            draw = ImageDraw.Draw(frame)
            low_phase = index in {2, 9}
            if full_body_motion and low_phase:
                body_box = (48, 92, 405, 334)
                left_paw_box = (72, 272, 184, 342)
                right_paw_box = (286, 280, 378, 342)
            else:
                body_box = (65, 75, 390, 318)
                left_paw_box = (90, 265, 170, 325)
                right_paw_box = (300, 270, 365, 325)
            draw.ellipse(body_box, fill=(115, 82, 52, 255))
            draw.ellipse(left_paw_box, fill=(145, 105, 70, 255))
            draw.ellipse(right_paw_box, fill=(145, 105, 70, 255))
            head_box = (
                (335, 188, 475, 288)
                if low_muzzle and low_phase
                else (335, 112, 475, 230)
            )
            draw.ellipse(head_box, fill=(125, 90, 58, 255))
            # Source-only identity signals that must not survive in Image 2.
            draw.line((342, 225, 456, 225), fill=(225, 25, 25, 255), width=7)
            draw.polygon(
                ((395, 226), (405, 249), (415, 226)),
                fill=(235, 45, 25, 255),
            )
            if static_contact_paw:
                # This planted paw reaches the bowl rim in every frame. The old
                # lowest-pixel gap signal was therefore always zero and missed
                # the otherwise obvious vertical head motion.
                draw.ellipse((420, 238, 462, 286), fill=(145, 105, 70, 255))
            for offset in (0, 18, 36):
                draw.line(
                    (110 + offset, 105, 145 + offset, 300),
                    fill=(45, 30, 20, 255),
                    width=5,
                )
            draw.rectangle((400, 270, 540, 325), fill=(215, 185, 140, 255))
            frames.append(frame)
        return frames

    @staticmethod
    def _grooming_geometry_frames(
        *,
        motion: bool = True,
        contact_structure: bool = True,
    ) -> list[Image.Image]:
        frames: list[Image.Image] = []
        for index in range(13):
            frame = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
            draw = ImageDraw.Draw(frame)
            action_phase = motion and index in {4, 10}
            if contact_structure:
                body_color = (30, 90, 220, 255)
                head_color = (210, 35, 70, 255)
                tail_color = (105, 40, 180, 255)
                contact_color = (245, 245, 225, 255)
                planted_color = (230, 170, 45, 255)
            else:
                body_color = (110, 110, 110, 255)
                head_color = body_color
                tail_color = body_color
                contact_color = body_color
                planted_color = body_color

            draw.ellipse((145, 135, 425, 460), fill=body_color)
            draw.ellipse((360, 335, 475, 455), fill=tail_color)
            draw.ellipse((205, 395, 295, 470), fill=planted_color)
            if action_phase:
                head_box = (80, 90, 310, 310)
                muzzle_box = (112, 185, 240, 245)
                paw_box = (140, 212, 218, 370)
            else:
                head_box = (105, 60, 335, 280)
                muzzle_box = (137, 165, 265, 225)
                paw_box = (158, 220, 236, 378)
            draw.ellipse(head_box, fill=head_color)
            draw.polygon(
                (
                    (head_box[0] + 28, head_box[1] + 35),
                    (head_box[0] + 58, head_box[1] - 18),
                    (head_box[0] + 92, head_box[1] + 42),
                ),
                fill=head_color,
            )
            draw.ellipse(muzzle_box, fill=contact_color)
            draw.ellipse(paw_box, fill=contact_color)

            if contact_structure:
                # High-frequency source-only coat marks must not be copied to
                # the guide. They are deliberately saturated and narrow.
                for offset in range(0, 90, 12):
                    draw.line(
                        (285 + offset // 3, 190, 315 + offset // 3, 365),
                        fill=(20, 235, 80, 255),
                        width=4,
                    )
                draw.line(
                    (112, 140, 286, 245),
                    fill=(255, 35, 220, 255),
                    width=5,
                )
            frames.append(frame)
        return frames

    @staticmethod
    def _aligned_reference(
        *,
        muzzle_gap_px: int = 6,
    ) -> tuple[Image.Image, Image.Image]:
        reference = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
        draw = ImageDraw.Draw(reference)
        draw.ellipse((55, 130, 455, 405), fill=(92, 102, 112, 255))
        draw.ellipse((175, 175, 350, 338), fill=(112, 122, 132, 255))
        draw.ellipse((85, 335, 175, 405), fill=(122, 132, 142, 255))
        draw.ellipse((340, 335, 430, 405), fill=(122, 132, 142, 255))
        # Neutral chest fur and a planted paw deliberately intrude into the
        # bowl-centre strip and reach its rim.  A lowest-blue-pixel heuristic
        # would select these instead of the mouth.
        draw.ellipse((260, 292, 315, 342), fill=(105, 112, 120, 255))
        draw.ellipse((272, 311, 318, 350), fill=(118, 124, 132, 255))
        # The real profile has a small natural warm nose/muzzle.  The fixture
        # can either leave a small gap or exactly touch the food boundary.
        muzzle_bottom = 329 - muzzle_gap_px
        draw.ellipse(
            (230, muzzle_bottom - 24, 270, muzzle_bottom),
            fill=(220, 120, 125, 255),
        )
        draw.rounded_rectangle(
            (218, 330, 350, 405),
            radius=22,
            fill=(215, 190, 155, 255),
        )

        colored = Image.new("RGB", (512, 512), (0, 0, 0))
        mask_draw = ImageDraw.Draw(colored)
        mask_draw.ellipse((55, 130, 455, 405), fill=(0, 0, 255))
        mask_draw.ellipse((175, 175, 350, 338), fill=(0, 0, 255))
        mask_draw.ellipse((85, 335, 175, 405), fill=(0, 0, 255))
        mask_draw.ellipse((340, 335, 430, 405), fill=(0, 0, 255))
        mask_draw.rounded_rectangle(
            (218, 330, 350, 405),
            radius=22,
            fill=(255, 0, 0),
        )
        return reference, colored

    @staticmethod
    def _drinking_reference() -> tuple[Image.Image, Image.Image]:
        reference = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
        draw = ImageDraw.Draw(reference)
        coat = (232, 202, 148, 255)
        ear = (205, 158, 104, 255)
        draw.ellipse((165, 145, 405, 345), fill=coat)
        draw.ellipse((70, 225, 235, 385), fill=coat)
        draw.ellipse((55, 300, 175, 402), fill=(238, 210, 165, 255))
        draw.ellipse((70, 240, 115, 345), fill=ear)
        draw.ellipse((188, 238, 232, 350), fill=ear)
        draw.rounded_rectangle((180, 290, 230, 410), radius=18, fill=coat)
        draw.rounded_rectangle((300, 285, 350, 410), radius=18, fill=coat)
        # A straight, low tail is deliberately far outside the head edit region.
        draw.polygon(
            ((380, 225), (472, 245), (472, 270), (380, 265)),
            fill=coat,
        )
        draw.rounded_rectangle(
            (55, 382, 185, 442),
            radius=20,
            fill=(170, 175, 180, 255),
        )
        draw.ellipse((66, 375, 174, 408), fill=(105, 155, 190, 255))
        draw.rounded_rectangle(
            (112, 350, 132, 401),
            radius=8,
            fill=(228, 92, 145, 255),
        )

        colored = Image.new("RGB", (512, 512), (0, 0, 0))
        mask_draw = ImageDraw.Draw(colored)
        mask_draw.ellipse((165, 145, 405, 345), fill=(0, 0, 255))
        mask_draw.ellipse((70, 225, 235, 385), fill=(0, 0, 255))
        mask_draw.ellipse((55, 300, 175, 402), fill=(0, 0, 255))
        mask_draw.ellipse((70, 240, 115, 345), fill=(0, 0, 255))
        mask_draw.ellipse((188, 238, 232, 350), fill=(0, 0, 255))
        mask_draw.rounded_rectangle(
            (180, 290, 230, 410), radius=18, fill=(0, 0, 255)
        )
        mask_draw.rounded_rectangle(
            (300, 285, 350, 410), radius=18, fill=(0, 0, 255)
        )
        mask_draw.polygon(
            ((380, 225), (472, 245), (472, 270), (380, 265)),
            fill=(0, 0, 255),
        )
        mask_draw.rounded_rectangle(
            (55, 382, 185, 442), radius=20, fill=(255, 0, 0)
        )
        mask_draw.ellipse((66, 375, 174, 408), fill=(255, 0, 0))
        mask_draw.rounded_rectangle(
            (112, 350, 132, 401), radius=8, fill=(0, 0, 255)
        )
        return reference, colored

    @staticmethod
    def _drinking_driver_frames() -> list[Image.Image]:
        frames: list[Image.Image] = []
        for index in range(81):
            phase = 2.0 * np.pi * index / 80.0
            lift = int(round(52.0 * (0.5 - 0.5 * np.cos(phase))))
            frame = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
            draw = ImageDraw.Draw(frame)
            neutral = (220, 222, 219, 255)
            draw.ellipse((165, 145, 405, 345), fill=neutral)
            draw.rounded_rectangle(
                (180, 290, 230, 410), radius=18, fill=neutral
            )
            draw.rounded_rectangle(
                (300, 285, 350, 410), radius=18, fill=neutral
            )
            draw.ellipse(
                (70, 225 - lift, 235, 385 - lift),
                fill=neutral,
            )
            draw.polygon(
                ((380, 225), (472, 245), (472, 270), (380, 265)),
                fill=neutral,
            )
            frames.append(frame)
        return frames

    def test_temporal_prop_mask_keeps_cat_occlusion_out_of_bowl_layer(self) -> None:
        frames = self._normalized_frames()
        exclusions, metadata = scail.make_eating_prop_exclusion_masks(frames)
        occluded = np.asarray(exclusions[0], dtype=np.uint8)
        clear = np.asarray(exclusions[4], dtype=np.uint8)

        self.assertEqual(
            metadata["policy"],
            "map_driving_bowl_as_red_scail_object",
        )
        self.assertEqual(
            metadata["method"],
            "eating_temporal_food_anchor_bowl_v2",
        )
        self.assertEqual(int(occluded[335, 375]), 0)
        self.assertEqual(int(clear[360, 400]), 255)
        self.assertLess(metadata["prop_area_cv"], 0.05)
        self.assertLessEqual(metadata["prop_center_step_max"], 1.5)

    def test_white_bowl_uses_brown_food_anchor_instead_of_right_tail(
        self,
    ) -> None:
        exclusions, metadata = scail.make_eating_prop_exclusion_masks(
            self._white_bowl_frames()
        )
        occluded = np.asarray(exclusions[0], dtype=np.uint8)
        clear = np.asarray(exclusions[4], dtype=np.uint8)
        left, top, right, bottom = metadata["prop_exclusion_bbox_union"]

        self.assertLess(left, 165)
        self.assertGreater(right, 305)
        self.assertGreaterEqual(top, 360)
        self.assertGreater(bottom, 445)
        self.assertEqual(int(clear[430, 235]), 255)
        self.assertEqual(int(clear[400, 410]), 0)
        self.assertEqual(int(occluded[382, 245]), 0)
        self.assertGreaterEqual(metadata["prop_fraction_min"], 0.04)
        self.assertLessEqual(metadata["prop_fraction_max"], 0.20)

    def test_temporal_prop_mask_fails_closed_without_food_or_bowl(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "eating_prop_split_unreliable",
        ):
            scail.make_eating_prop_exclusion_masks(
                self._white_bowl_frames(with_bowl=False)
            )

    def test_pose_identity_excludes_bowl_but_keeps_total_pose_pixels(self) -> None:
        source_frames = self._source_frames()

        pose_frames, masks, metadata = scail.make_pose_and_mask_frames(
            source_frames,
            (512, 512),
            separate_eating_prop=True,
        )
        mask = np.asarray(masks[4].convert("RGB"), dtype=np.uint8)
        blue = (mask[..., 0] == 0) & (mask[..., 1] == 0) & (mask[..., 2] == 255)
        red = (mask[..., 0] == 255) & (mask[..., 1] == 0) & (mask[..., 2] == 0)
        white = np.all(mask == 255, axis=2)

        self.assertEqual(pose_frames[0].size, (512, 512))
        self.assertTrue(metadata["scene_prop_separated"])
        self.assertGreater(metadata["scene_prop_fraction_min"], 0.0)
        self.assertLess(
            metadata["identity_fraction_mean"],
            metadata["subject_fraction_mean"],
        )
        self.assertTrue(np.any(blue))
        self.assertTrue(np.any(red))
        self.assertTrue(np.any(white))

    def test_full_eating_pose_preserves_driver_silhouette_and_separates_bowl(
        self,
    ) -> None:
        poses, masks, metadata = scail.make_full_eating_pose_and_mask_frames(
            self._geometry_frames(full_body_motion=True),
            (512, 512),
        )

        self.assertEqual(
            metadata["policy"],
            "full_driver_pose_with_separate_bowl_v1",
        )
        self.assertFalse(metadata["driver_signal_only"])
        self.assertTrue(metadata["driver_rgb_preserved"])
        self.assertTrue(metadata["driver_silhouette_preserved"])
        self.assertEqual(
            metadata["pose_geometry_scope"],
            "full_subject_silhouette_per_frame",
        )
        self.assertTrue(metadata["scene_prop_separated"])
        self.assertEqual(len(metadata["identity_bbox_by_frame"]), len(poses))
        self.assertFalse(
            np.array_equal(
                np.asarray(poses[0].convert("RGB")),
                np.asarray(poses[2].convert("RGB")),
            )
        )

        upright_mask = np.asarray(masks[0].convert("RGB"), dtype=np.uint8)
        low_mask = np.asarray(masks[2].convert("RGB"), dtype=np.uint8)
        upright_blue = np.all(upright_mask == (0, 0, 255), axis=2)
        blue = np.all(low_mask == (0, 0, 255), axis=2)
        red = np.all(low_mask == (255, 0, 0), axis=2)
        white = np.all(low_mask == (255, 255, 255), axis=2)
        self.assertFalse(np.array_equal(upright_blue, blue))
        self.assertTrue(np.any(blue))
        self.assertTrue(np.any(red))
        self.assertTrue(np.all(blue | red | white))
        self.assertFalse(np.any(blue & red))
        motion_audit = metadata["identity_motion_audit"]
        self.assertEqual(
            motion_audit["policy"],
            "first_frame_blue_identity_mask_xor_v1",
        )
        self.assertTrue(motion_audit["passed"])
        self.assertGreaterEqual(
            motion_audit["overall_change_ratio_max"],
            motion_audit["thresholds"]["overall_change_ratio_min"],
        )
        self.assertGreaterEqual(
            motion_audit["lower_half_change_ratio_max"],
            motion_audit["thresholds"]["lower_half_change_ratio_min"],
        )
        self.assertGreaterEqual(
            motion_audit["lower_quarter_change_ratio_max"],
            motion_audit["thresholds"]["lower_quarter_change_ratio_min"],
        )
        self.assertEqual(
            len(motion_audit["overall_change_ratio_by_frame"]),
            len(poses),
        )

    def test_full_eating_pose_rejects_head_only_motion(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "eating_full_pose_motion_unreliable.*lower_quarter",
        ):
            scail.make_full_eating_pose_and_mask_frames(
                self._geometry_frames(),
                (512, 512),
            )

    def test_full_eating_pose_rejects_almost_static_motion(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "eating_full_pose_motion_unreliable.*overall",
        ):
            scail.make_full_eating_pose_and_mask_frames(
                self._geometry_frames(low_muzzle=False),
                (512, 512),
            )

    def test_non_eating_pose_mask_keeps_original_single_identity_policy(self) -> None:
        source_frames = self._source_frames()

        _, masks, metadata = scail.make_pose_and_mask_frames(
            source_frames,
            (512, 512),
            separate_eating_prop=False,
        )
        mask = np.asarray(masks[0].convert("RGB"), dtype=np.uint8)
        unknown = ~(
            np.all(mask == (255, 255, 255), axis=2)
            | np.all(mask == (0, 0, 255), axis=2)
        )

        self.assertFalse(metadata["scene_prop_separated"])
        self.assertFalse(np.any(unknown))

    def test_toileting_pose_separates_and_lightens_lateral_tray(self) -> None:
        frames: list[Image.Image] = []
        for _index in range(13):
            frame = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
            draw = ImageDraw.Draw(frame)
            draw.rounded_rectangle(
                (62, 365, 450, 466),
                radius=18,
                fill=(62, 68, 65, 255),
            )
            draw.ellipse((150, 72, 365, 420), fill=(225, 205, 170, 255))
            draw.rounded_rectangle(
                (185, 300, 240, 445),
                radius=20,
                fill=(225, 205, 170, 255),
            )
            draw.rounded_rectangle(
                (275, 300, 330, 445),
                radius=20,
                fill=(225, 205, 170, 255),
            )
            frames.append(frame)

        poses, masks, metadata = scail.make_pose_and_mask_frames(
            frames,
            (512, 512),
            separate_toileting_prop=True,
        )

        self.assertTrue(metadata["scene_prop_separated"])
        self.assertEqual(
            metadata["scene_prop_metadata"]["policy"],
            "toileting_complete_tray_red_scail_prop_v3",
        )
        mask = np.asarray(masks[0].convert("RGB"), dtype=np.uint8)
        pose = np.asarray(poses[0].convert("RGB"), dtype=np.uint8)
        blue = np.all(mask == (0, 0, 255), axis=2)
        red = np.all(mask == (255, 0, 0), axis=2)
        white = np.all(mask == (255, 255, 255), axis=2)
        self.assertTrue(np.any(blue))
        self.assertTrue(np.any(red))
        self.assertTrue(np.all(blue | red | white))
        self.assertTrue(red[430, 256])
        self.assertTrue(blue[400, 210])
        self.assertTrue(
            np.array_equal(pose[400, 210], np.asarray((225, 205, 170)))
        )
        self.assertGreater(
            metadata["scene_prop_metadata"]["grid_prop_area_min"],
            0,
        )
        self.assertGreaterEqual(int(pose[red].min()), 188)

    def test_toileting_white_paws_remain_identity_over_dark_grid(self) -> None:
        frames: list[Image.Image] = []
        for _index in range(9):
            frame = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
            draw = ImageDraw.Draw(frame)
            draw.rounded_rectangle(
                (62, 365, 450, 466),
                radius=18,
                fill=(72, 72, 72, 255),
            )
            draw.ellipse((150, 72, 365, 420), fill=(225, 225, 225, 255))
            draw.rounded_rectangle(
                (185, 300, 240, 445),
                radius=20,
                fill=(225, 225, 225, 255),
            )
            draw.rounded_rectangle(
                (275, 300, 330, 445),
                radius=20,
                fill=(225, 225, 225, 255),
            )
            frames.append(frame)

        _poses, masks, metadata = scail.make_pose_and_mask_frames(
            frames,
            (512, 512),
            separate_toileting_prop=True,
        )
        mask = np.asarray(masks[0].convert("RGB"), dtype=np.uint8)
        blue = np.all(mask == (0, 0, 255), axis=2)
        red = np.all(mask == (255, 0, 0), axis=2)

        self.assertTrue(metadata["scene_prop_metadata"]["white_maquette_material"])
        self.assertEqual(
            metadata["scene_prop_metadata"]["grid_material_policy"],
            "white_maquette_deep_grid_slots_v1",
        )
        self.assertTrue(blue[400, 210])
        self.assertTrue(red[430, 256])

    def test_eating_geometry_guide_selects_low_muzzle_and_removes_identity(self) -> None:
        guide, metadata = scail.make_eating_geometry_guide(
            self._geometry_frames(),
            (512, 512),
        )
        pixels = np.asarray(guide.convert("RGB"), dtype=np.uint8)
        green = np.all(pixels == (0, 255, 0), axis=2)
        foreground = pixels[~green]

        self.assertEqual(metadata["selected_frame_index"], 2)
        self.assertLessEqual(metadata["muzzle_bowl_gap_px"], 4)
        self.assertGreaterEqual(metadata["contact_pixels"], 4)
        self.assertTrue(metadata["head_lowering_selection"]["passed"])
        self.assertGreaterEqual(
            metadata["head_lowering_selection"]["head_y_p10_span"],
            metadata["head_lowering_selection"]["head_y_p10_span_min"],
        )
        self.assertEqual(
            metadata["policy"],
            "driving_eating_geometry_image2_v6",
        )
        self.assertFalse(metadata["palette"]["source_rgb_preserved"])
        self.assertFalse(
            metadata["palette"]["high_frequency_texture_preserved"]
        )
        structure = metadata["semantic_structure"]
        self.assertEqual(
            structure["method"],
            "compressed_blurred_quantized_neutral_anatomy_v1",
        )
        self.assertEqual(structure["subject_gray_range"], [120, 168])
        self.assertGreaterEqual(structure["subject_unique_rgb_count"], 2)
        self.assertLessEqual(structure["subject_unique_rgb_count"], 7)
        self.assertTrue(structure["internal_luma_variation"])
        self.assertFalse(structure["appearance_bearing_face_detail"])
        self.assertFalse(structure["source_luma_values_preserved"])
        self.assertLessEqual(
            metadata["outline_smoothing"]["perimeter_ratio"],
            1.01,
        )
        self.assertFalse(
            metadata["outline_smoothing"]["high_frequency_outline_preserved"]
        )
        semantic_colors = np.asarray(
            [
                metadata["semantic_food"]["bowl_rgb"],
                metadata["semantic_food"]["bowl_rim_rgb"],
                metadata["semantic_food"]["food_rgb"],
            ],
            dtype=np.uint8,
        )
        semantic_prop = np.any(
            np.all(foreground[:, None, :] == semantic_colors[None, :, :], axis=2),
            axis=1,
        )
        subject_foreground = foreground[~semantic_prop]
        self.assertTrue(np.all(subject_foreground[:, 0] == subject_foreground[:, 1]))
        self.assertTrue(np.all(subject_foreground[:, 1] == subject_foreground[:, 2]))
        subject_gray_values = np.unique(subject_foreground[:, 0]).tolist()
        self.assertEqual(
            subject_gray_values,
            structure["subject_gray_values"],
        )
        self.assertGreaterEqual(min(subject_gray_values), 120)
        self.assertLessEqual(max(subject_gray_values), 168)
        self.assertTrue(all(value % 8 == 0 for value in subject_gray_values))
        self.assertTrue(metadata["semantic_food"]["semantic_chroma_only"])
        self.assertTrue(
            all(
                np.any(np.all(pixels == color, axis=2))
                for color in semantic_colors
            )
        )
        self.assertFalse(np.any(np.all(pixels == (225, 25, 25), axis=2)))
        self.assertFalse(np.any(np.all(pixels == (115, 82, 52), axis=2)))
        self.assertGreater(metadata["subject_pixels"], 500)
        self.assertGreater(metadata["bowl_pixels"], 100)

    def test_eating_geometry_guide_fails_closed_without_muzzle_contact(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "eating_geometry_guide_unreliable",
        ):
            scail.make_eating_geometry_guide(
                self._geometry_frames(low_muzzle=False),
                (512, 512),
            )

    def test_eating_geometry_guide_ignores_static_paw_bowl_contact(self) -> None:
        _guide, metadata = scail.make_eating_geometry_guide(
            self._geometry_frames(static_contact_paw=True),
            (512, 512),
        )

        self.assertEqual(metadata["selected_frame_index"], 2)
        self.assertGreater(
            metadata["head_lowering_selection"]["head_y_p10_span"],
            metadata["head_lowering_selection"]["head_y_p10_span_min"],
        )

    def test_eating_geometry_guide_requires_temporal_prop_evidence(self) -> None:
        with self.assertRaisesRegex(ValueError, "fewer_than_9_frames"):
            scail.make_eating_geometry_guide(
                self._geometry_frames()[:8],
                (512, 512),
            )

    def test_grooming_geometry_guide_keeps_contact_but_removes_identity(
        self,
    ) -> None:
        guide, metadata = scail.make_grooming_geometry_guide(
            self._grooming_geometry_frames(),
            (512, 512),
        )
        pixels = np.asarray(guide.convert("RGB"), dtype=np.uint8)
        green = np.all(pixels == (0, 255, 0), axis=2)
        foreground = pixels[~green]
        gray = pixels[:, :, 0]

        self.assertEqual(metadata["selected_frame_index"], 4)
        self.assertEqual(
            metadata["policy"],
            "driving_grooming_geometry_image1_v3",
        )
        self.assertTrue(metadata["action_motion_audit"]["passed"])
        self.assertGreaterEqual(
            metadata["action_motion_audit"]["upper_body_xor_ratio"],
            metadata["action_motion_audit"]["thresholds"][
                "upper_body_xor_ratio_min"
            ],
        )
        relation = metadata["raised_paw_muzzle_relation"]
        self.assertTrue(relation["passed"])
        self.assertEqual(
            relation["policy"],
            "source_derived_muzzle_foreleg_geometry_audit_v3",
        )
        self.assertGreaterEqual(
            relation["upper_muzzle_support_pixels"],
            relation["minimum_support_pixels"],
        )
        self.assertGreaterEqual(
            relation["lower_raised_paw_support_pixels"],
            relation["minimum_support_pixels"],
        )
        self.assertTrue(relation["regions_disjoint"])
        self.assertFalse(relation["c_hook_component_preserved"])
        self.assertFalse(relation["black_or_tool_semantics_preserved"])
        self.assertGreaterEqual(
            relation["semantic_contact_pixels"],
            relation["minimum_semantic_contact_pixels"],
        )
        encoding = relation["output_encoding"]
        self.assertEqual(
            encoding["method"],
            "compressed_blurred_quantized_neutral_anatomy",
        )
        foreground_values = np.unique(gray[~green])
        self.assertGreaterEqual(len(foreground_values), 4)
        self.assertGreaterEqual(int(foreground_values.min()), 120)
        self.assertLessEqual(int(foreground_values.max()), 168)
        self.assertTrue(np.all(foreground_values % 8 == 0))
        self.assertFalse(metadata["palette"]["source_rgb_preserved"])
        self.assertFalse(
            metadata["palette"]["high_frequency_texture_preserved"]
        )
        self.assertFalse(metadata["palette"]["distance_field_preserved"])
        self.assertTrue(metadata["palette"]["low_frequency_anatomy_preserved"])
        self.assertFalse(
            metadata["palette"]["source_marking_contrast_preserved"]
        )
        self.assertFalse(
            metadata["palette"]["artificial_part_markers_preserved"]
        )
        self.assertFalse(metadata["palette"]["black_foreground_preserved"])
        self.assertGreaterEqual(int(foreground.min()), 120)
        self.assertTrue(np.all(foreground[:, 0] == foreground[:, 1]))
        self.assertTrue(np.all(foreground[:, 1] == foreground[:, 2]))
        self.assertFalse(np.any(np.all(pixels == (30, 90, 220), axis=2)))
        self.assertFalse(np.any(np.all(pixels == (210, 35, 70), axis=2)))
        self.assertFalse(np.any(np.all(pixels == (20, 235, 80), axis=2)))
        self.assertGreater(metadata["subject_pixels"], 500)
        self.assertLessEqual(
            metadata["outline_smoothing"]["perimeter_ratio"],
            1.01,
        )

    def test_grooming_geometry_guide_fails_closed_without_action_motion(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "grooming_geometry_guide_unreliable:"
            " insufficient_raised_paw_head_motion",
        ):
            scail.make_grooming_geometry_guide(
                self._grooming_geometry_frames(motion=False),
                (512, 512),
            )

    def test_grooming_geometry_guide_fails_without_paw_muzzle_structure(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "grooming_geometry_guide_unreliable:"
            " low_frequency_contact_contrast",
        ):
            scail.make_grooming_geometry_guide(
                self._grooming_geometry_frames(contact_structure=False),
                (512, 512),
            )

    def test_grooming_geometry_guide_requires_temporal_evidence(self) -> None:
        with self.assertRaisesRegex(ValueError, "fewer_than_9_frames"):
            scail.make_grooming_geometry_guide(
                self._grooming_geometry_frames()[:8],
                (512, 512),
            )

    def test_reference_aligned_eating_motion_uses_only_driver_timing(self) -> None:
        reference, reference_mask = self._aligned_reference()
        poses, masks, metadata = scail.make_eating_reference_aligned_pose_frames(
            self._geometry_frames(static_contact_paw=True),
            reference,
            reference_mask,
            (512, 512),
        )

        self.assertEqual(
            metadata["policy"],
            "reference_space_head_nod_food_overlap_v2",
        )
        self.assertTrue(metadata["driver_signal_only"])
        self.assertFalse(metadata["driver_rgb_preserved"])
        self.assertFalse(metadata["driver_silhouette_preserved"])
        self.assertEqual(
            metadata["driver_signal_method"],
            "bowl_relative_upper_head_row_p10_v1",
        )
        self.assertGreaterEqual(metadata["driver_head_y_span"], 4.0)
        self.assertLessEqual(metadata["reference_muzzle_bowl_gap_px"], 8)
        self.assertEqual(
            metadata["reference_muzzle_method"],
            "warm_face_component_above_bowl_v1",
        )
        self.assertEqual(metadata["reference_muzzle_bowl_gap_px"], 6)
        self.assertEqual(metadata["target_muzzle_food_overlap_px"], 8.0)
        self.assertEqual(metadata["reference_contact_drop_px"], 14.0)
        self.assertGreater(metadata["reference_contact_drop_control_px"], 6.0)
        self.assertEqual(metadata["low_phase_face_bowl_gap_px"], 0)
        self.assertLessEqual(
            metadata["low_phase_face_bowl_signed_gap_px"],
            -6,
        )
        self.assertGreaterEqual(
            metadata["low_phase_face_bowl_signed_gap_px"],
            -10,
        )
        self.assertGreaterEqual(
            metadata["low_phase_muzzle_food_overlap_depth_px"],
            6,
        )
        self.assertLessEqual(
            metadata["low_phase_muzzle_food_overlap_depth_px"],
            10,
        )
        self.assertGreater(
            metadata["low_phase_muzzle_food_overlap_pixels"],
            0,
        )
        self.assertLessEqual(metadata["vertical_control_abs_max_px"], 18.0)
        self.assertEqual(metadata["reference_face_seed_bbox"], [230, 299, 271, 324])
        self.assertGreaterEqual(metadata["head_lift_px_max"], 12.0)
        self.assertEqual(len(poses), 13)
        self.assertEqual(len(masks), 13)

        lift_values = metadata["head_lift_px_by_frame"]
        low_index = int(np.argmin(lift_values))
        high_index = int(np.argmax(lift_values))
        low_mask = np.asarray(masks[low_index].convert("RGB"), dtype=np.uint8)
        high_mask = np.asarray(masks[high_index].convert("RGB"), dtype=np.uint8)
        low_blue = np.all(low_mask == (0, 0, 255), axis=2)
        high_blue = np.all(high_mask == (0, 0, 255), axis=2)
        low_red = np.all(low_mask == (255, 0, 0), axis=2)
        high_red = np.all(high_mask == (255, 0, 0), axis=2)

        self.assertTrue(np.array_equal(low_red, high_red))
        reference_red = low_red.copy()
        first_pose = np.asarray(poses[0].convert("RGB"), dtype=np.uint8)
        first_bowl_rgb = first_pose[reference_red]
        for pose, mask in zip(poses, masks, strict=True):
            mask_pixels = np.asarray(mask.convert("RGB"), dtype=np.uint8)
            red = np.all(mask_pixels == (255, 0, 0), axis=2)
            blue = np.all(mask_pixels == (0, 0, 255), axis=2)
            white = np.all(mask_pixels == (255, 255, 255), axis=2)
            self.assertTrue(np.array_equal(red, reference_red))
            self.assertFalse(np.any(red & blue))
            self.assertTrue(np.all(red | blue | white))
            pose_pixels = np.asarray(pose.convert("RGB"), dtype=np.uint8)
            self.assertTrue(
                np.array_equal(pose_pixels[reference_red], first_bowl_rgb)
            )
        self.assertTrue(metadata["static_bowl_mask"])
        self.assertTrue(metadata["static_bowl_rgb"])
        self.assertLess(np.where(high_blue)[0].min(), np.where(low_blue)[0].min())
        allowed = (
            np.all(high_mask == (255, 255, 255), axis=2)
            | high_blue
            | high_red
        )
        self.assertTrue(np.all(allowed))
        high_pose = np.asarray(poses[high_index].convert("RGB"), dtype=np.uint8)
        self.assertFalse(np.any(np.all(high_pose == (225, 25, 25), axis=2)))

    def test_reference_aligned_drinking_locks_tail_and_body_geometry(self) -> None:
        reference, reference_mask = self._drinking_reference()
        poses, masks, metadata = scail.make_drinking_reference_aligned_pose_frames(
            self._drinking_driver_frames(),
            reference,
            reference_mask,
            (512, 512),
            frame_count=81,
        )

        self.assertEqual(
            metadata["policy"],
            "reference_space_white_driver_head_path_static_bowl_v3",
        )
        self.assertTrue(metadata["driver_input_used"])
        self.assertTrue(metadata["driver_signal_only"])
        self.assertFalse(metadata["driver_rgb_preserved"])
        self.assertFalse(metadata["driver_silhouette_preserved"])
        self.assertTrue(metadata["reference_identity_preserved"])
        self.assertTrue(
            metadata["body_ears_paws_tail_outside_head_motion_locked"]
        )
        self.assertTrue(metadata["closed_loop_endpoints"])
        self.assertEqual(metadata["lap_cycles"], 3)
        self.assertGreaterEqual(metadata["head_lift_px_max"], 14.0)
        self.assertGreater(metadata["driver_head_y_span"], 4.0)
        self.assertGreaterEqual(metadata["tongue_motion_span_px"], 2.0)
        self.assertEqual(len(poses), 81)
        self.assertEqual(len(masks), 81)

        first_pose = np.asarray(poses[0].convert("RGB"), dtype=np.uint8)
        first_mask = np.asarray(masks[0].convert("RGB"), dtype=np.uint8)
        self.assertTrue(
            np.array_equal(first_pose, np.asarray(poses[-1].convert("RGB")))
        )
        self.assertTrue(
            np.array_equal(first_mask, np.asarray(masks[-1].convert("RGB")))
        )
        # Straight tail and lower bowl/body regions remain byte-identical while
        # only the tongue/head corridor moves.
        for pose, mask in zip(poses[1:], masks[1:], strict=True):
            pose_pixels = np.asarray(pose.convert("RGB"), dtype=np.uint8)
            mask_pixels = np.asarray(mask.convert("RGB"), dtype=np.uint8)
            self.assertTrue(
                np.array_equal(
                    pose_pixels[215:285, 370:480],
                    first_pose[215:285, 370:480],
                )
            )
            self.assertTrue(
                np.array_equal(
                    mask_pixels[215:285, 370:480],
                    first_mask[215:285, 370:480],
                )
            )
            self.assertTrue(
                np.array_equal(
                    pose_pixels[410:450, 45:195],
                    first_pose[410:450, 45:195],
                )
            )
            blue = np.all(mask_pixels == (0, 0, 255), axis=2)
            red = np.all(mask_pixels == (255, 0, 0), axis=2)
            white = np.all(mask_pixels == (255, 255, 255), axis=2)
            self.assertTrue(np.all(blue | red | white))

        tongue_values = metadata["tongue_center_y_by_frame"]
        self.assertLess(min(tongue_values), max(tongue_values))

    def test_drinking_reference_split_keeps_contact_tongue_out_of_bowl(
        self,
    ) -> None:
        reference, _reference_mask = self._drinking_reference()

        colored, exclusion, metadata = (
            scail.make_drinking_reference_identity_mask(reference)
        )
        mask = np.asarray(colored.convert("RGB"), dtype=np.uint8)
        excluded = np.asarray(exclusion.convert("L"), dtype=np.uint8)

        self.assertEqual(
            metadata["policy"],
            "drinking_reference_dog_blue_bowl_red_v2",
        )
        self.assertTrue(metadata["tongue_reclassified_as_identity"])
        self.assertEqual(tuple(mask[370, 122]), (0, 0, 255))
        self.assertEqual(int(excluded[370, 122]), 0)
        self.assertEqual(tuple(mask[430, 100]), (255, 0, 0))
        self.assertEqual(int(excluded[430, 100]), 255)
        self.assertEqual(tuple(mask[250, 300]), (0, 0, 255))

    def test_drinking_motion_uses_contact_anchor_without_visible_tongue(
        self,
    ) -> None:
        reference, reference_mask = self._drinking_reference()
        reference_pixels = np.asarray(reference.convert("RGBA"), dtype=np.uint8).copy()
        reference_pixels[350:402, 112:133, :3] = (232, 202, 148)
        reference_without_pink = Image.fromarray(reference_pixels, mode="RGBA")

        poses, masks, metadata = scail.make_drinking_reference_aligned_pose_frames(
            self._drinking_driver_frames(),
            reference_without_pink,
            reference_mask,
            (512, 512),
            frame_count=81,
        )

        self.assertFalse(metadata["tongue_anchor_detected"])
        self.assertEqual(
            metadata["motion_anchor_policy"],
            "nearest_identity_to_static_bowl_contact_v1",
        )
        self.assertTrue(metadata["driver_input_used"])
        self.assertGreaterEqual(metadata["head_lift_px_max"], 14.0)
        self.assertGreaterEqual(metadata["tongue_motion_span_px"], 2.0)
        self.assertEqual(len(poses), 81)
        self.assertEqual(len(masks), 81)

    def test_static_drinking_fallback_keeps_reference_identity(self) -> None:
        reference, reference_mask = self._drinking_reference()
        poses, masks, metadata = scail.make_static_reference_drinking_pose_frames(
            reference,
            reference_mask,
            (512, 512),
            frame_count=13,
            fallback_reason=(
                "drinking_reference_motion_unreliable: no_pink_tongue_anchor"
            ),
        )

        self.assertTrue(metadata["fallback_used"])
        self.assertFalse(metadata["driver_input_used"])
        self.assertFalse(metadata["driver_silhouette_preserved"])
        self.assertEqual(len(poses), 13)
        self.assertEqual(len(masks), 13)
        for pose, mask in zip(poses[1:], masks[1:], strict=True):
            self.assertTrue(np.array_equal(np.asarray(pose), np.asarray(poses[0])))
            self.assertTrue(np.array_equal(np.asarray(mask), np.asarray(masks[0])))

    def test_reference_aligned_eating_motion_pushes_touching_muzzle_into_food(
        self,
    ) -> None:
        reference, reference_mask = self._aligned_reference(muzzle_gap_px=0)
        poses, masks, metadata = scail.make_eating_reference_aligned_pose_frames(
            self._geometry_frames(static_contact_paw=True),
            reference,
            reference_mask,
            (512, 512),
        )

        self.assertEqual(metadata["reference_muzzle_bowl_gap_px"], 0)
        self.assertEqual(metadata["reference_contact_drop_px"], 8.0)
        self.assertGreaterEqual(
            metadata["low_phase_muzzle_food_overlap_depth_px"],
            6,
        )
        self.assertLessEqual(
            metadata["low_phase_muzzle_food_overlap_depth_px"],
            10,
        )
        self.assertGreater(
            metadata["low_phase_muzzle_food_overlap_pixels"],
            0,
        )
        self.assertLessEqual(metadata["vertical_control_abs_max_px"], 18.0)

        first_pose = np.asarray(poses[0].convert("RGB"), dtype=np.uint8)
        first_mask = np.asarray(masks[0].convert("RGB"), dtype=np.uint8)
        for left, top, right, bottom in metadata[
            "lower_lateral_freeze_bboxes"
        ]:
            expected_pose = first_pose[top:bottom, left:right]
            expected_mask = first_mask[top:bottom, left:right]
            for pose, mask in zip(poses[1:], masks[1:], strict=True):
                pose_pixels = np.asarray(pose.convert("RGB"), dtype=np.uint8)
                mask_pixels = np.asarray(mask.convert("RGB"), dtype=np.uint8)
                self.assertTrue(
                    np.array_equal(
                        pose_pixels[top:bottom, left:right],
                        expected_pose,
                    )
                )
                self.assertTrue(
                    np.array_equal(
                        mask_pixels[top:bottom, left:right],
                        expected_mask,
                    )
                )

    def test_eating_reference_mask_excludes_bowl_and_preserves_paws(self) -> None:
        reference = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
        draw = ImageDraw.Draw(reference)
        draw.ellipse((57, 121, 455, 391), fill=(95, 105, 115, 255))
        # A disconnected warm muzzle lies inside the lower prop-search zone.
        # It must remain identity-blue instead of joining the static red bowl.
        draw.ellipse((235, 300, 252, 312), fill=(220, 145, 100, 255))
        draw.rectangle((179, 348, 314, 389), fill=(205, 125, 55, 255))
        draw.ellipse((188, 315, 305, 360), fill=(175, 90, 35, 255))

        colored, exclusion, metadata = scail.make_eating_reference_identity_mask(
            reference
        )
        mask = np.asarray(colored, dtype=np.uint8)
        excluded = np.asarray(exclusion, dtype=np.uint8)

        self.assertEqual(
            metadata["policy"],
            "eating_reference_cat_blue_bowl_red_v1",
        )
        self.assertEqual(
            metadata["bowl_body_method"],
            "warm_bowl_eroded_core_v1",
        )
        self.assertEqual(tuple(mask[370, 245]), (255, 0, 0))
        self.assertEqual(int(excluded[370, 245]), 255)
        self.assertEqual(tuple(mask[360, 150]), (0, 0, 255))
        self.assertEqual(tuple(mask[360, 380]), (0, 0, 255))
        self.assertEqual(tuple(mask[306, 243]), (0, 0, 255))
        self.assertGreater(metadata["left_paw_identity_support"], 100)
        self.assertGreater(metadata["right_paw_identity_support"], 100)

    def test_eating_reference_mask_accepts_food_above_a_pale_bowl(self) -> None:
        reference = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
        draw = ImageDraw.Draw(reference)
        draw.ellipse((57, 112, 455, 399), fill=(95, 105, 115, 255))
        draw.ellipse((218, 344, 280, 398), fill=(225, 225, 220, 255))
        draw.ellipse((390, 344, 450, 398), fill=(225, 225, 220, 255))
        draw.rounded_rectangle(
            (268, 338, 397, 398),
            radius=26,
            fill=(211, 194, 168, 255),
        )
        draw.ellipse((277, 326, 382, 345), fill=(160, 100, 45, 255))

        colored, exclusion, metadata = scail.make_eating_reference_identity_mask(
            reference
        )
        mask = np.asarray(colored, dtype=np.uint8)
        excluded = np.asarray(exclusion, dtype=np.uint8)

        self.assertEqual(
            metadata["bowl_body_method"],
            "warm_food_core_plus_beige_bowl_v1",
        )
        self.assertLess(
            metadata["food_core_bbox"][3],
            112 + round((399 - 112) * 0.82),
        )
        self.assertEqual(tuple(mask[382, 330]), (255, 0, 0))
        self.assertEqual(int(excluded[382, 330]), 255)
        self.assertEqual(tuple(mask[375, 235]), (0, 0, 255))
        self.assertEqual(tuple(mask[375, 430]), (0, 0, 255))
        self.assertEqual(tuple(mask[375, 260]), (0, 0, 255))
        self.assertEqual(tuple(mask[375, 405]), (0, 0, 255))
        self.assertEqual(int(excluded[375, 260]), 0)
        self.assertEqual(int(excluded[375, 405]), 0)
        self.assertGreater(metadata["left_paw_identity_support"], 500)
        self.assertGreater(metadata["right_paw_identity_support"], 500)

    def test_golden_dog_muzzle_above_kibble_remains_identity_blue(self) -> None:
        reference = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
        draw = ImageDraw.Draw(reference)
        draw.ellipse((57, 90, 455, 399), fill=(225, 210, 175, 255))
        draw.ellipse((105, 345, 170, 420), fill=(228, 218, 194, 255))
        draw.ellipse((350, 345, 425, 420), fill=(228, 218, 194, 255))
        # A deep warm muzzle ends only ten pixels above the food. It must not
        # be unioned into the static red bowl identity merely because this is a
        # golden dog eating with physical mouth-to-food contact.
        draw.rounded_rectangle(
            (205, 320, 270, 382),
            radius=22,
            fill=(170, 115, 70, 255),
        )
        draw.rounded_rectangle(
            (150, 390, 315, 450),
            radius=20,
            fill=(205, 194, 173, 255),
        )
        draw.ellipse((164, 391, 301, 414), fill=(145, 80, 35, 255))

        colored, exclusion, metadata = scail.make_eating_reference_identity_mask(
            reference
        )
        mask = np.asarray(colored, dtype=np.uint8)
        excluded = np.asarray(exclusion, dtype=np.uint8)

        self.assertEqual(metadata["split_status"], "strict")
        self.assertGreater(
            metadata["food_preliminary_components"],
            metadata["food_core_components"],
        )
        self.assertGreaterEqual(metadata["food_bottom_band_tolerance"], 6)
        self.assertEqual(tuple(mask[360, 235]), (0, 0, 255))
        self.assertEqual(int(excluded[360, 235]), 0)
        self.assertEqual(tuple(mask[430, 230]), (255, 0, 0))
        self.assertEqual(int(excluded[430, 230]), 255)

    def test_off_center_dog_bowl_keeps_full_prop_and_reference_motion(
        self,
    ) -> None:
        reference = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
        draw = ImageDraw.Draw(reference)
        draw.ellipse((150, 55, 420, 360), fill=(225, 210, 175, 255))
        draw.ellipse((90, 200, 270, 395), fill=(220, 205, 175, 255))
        draw.ellipse((245, 330, 310, 455), fill=(228, 218, 194, 255))
        draw.ellipse((350, 300, 420, 455), fill=(228, 218, 194, 255))
        # The natural warm muzzle touches a bowl placed at the extreme left of
        # the complete dog bbox. There is no paw on the bowl's outer side, so a
        # symmetric two-sided paw requirement would incorrectly reject it.
        draw.ellipse((135, 350, 195, 378), fill=(140, 120, 115, 255))
        draw.rounded_rectangle(
            (90, 380, 237, 457),
            radius=22,
            fill=(205, 194, 173, 255),
        )
        draw.ellipse((103, 380, 225, 410), fill=(145, 80, 35, 255))

        colored, exclusion, metadata = scail.make_eating_reference_identity_mask(
            reference
        )
        mask = np.asarray(colored, dtype=np.uint8)
        excluded = np.asarray(exclusion, dtype=np.uint8)

        self.assertEqual(metadata["split_status"], "strict")
        self.assertTrue(metadata["prop_near_left_edge"])
        self.assertLess(
            metadata["left_paw_identity_support"],
            metadata["minimum_paw_identity_support"],
        )
        self.assertTrue(metadata["left_paw_identity_support_accepted"])
        self.assertTrue(metadata["right_paw_identity_support_accepted"])
        self.assertLessEqual(metadata["prop_exclusion_bbox"][0], 100)
        self.assertGreaterEqual(metadata["prop_exclusion_bbox"][2], 235)
        self.assertEqual(tuple(mask[365, 165]), (0, 0, 255))
        self.assertEqual(tuple(mask[430, 120]), (255, 0, 0))
        self.assertEqual(int(excluded[430, 120]), 255)
        self.assertEqual(tuple(mask[410, 275]), (0, 0, 255))

        _poses, _masks, motion_metadata = (
            scail.make_eating_reference_aligned_pose_frames(
                self._geometry_frames(static_contact_paw=True),
                reference,
                colored,
                (512, 512),
            )
        )
        self.assertEqual(
            motion_metadata["policy"],
            "reference_space_head_nod_food_overlap_v2",
        )
        self.assertGreaterEqual(
            motion_metadata["low_phase_muzzle_food_overlap_depth_px"],
            6,
        )

    def test_static_dog_eating_pose_fallback_preserves_reference_geometry(
        self,
    ) -> None:
        reference, reference_mask = self._aligned_reference()
        poses, masks, metadata = scail.make_static_reference_eating_pose_frames(
            reference,
            reference_mask,
            (512, 512),
            frame_count=13,
            fallback_reason=(
                "eating_reference_motion_unreliable: no_face_muzzle_anchor"
            ),
        )

        self.assertEqual(len(poses), 13)
        self.assertEqual(len(masks), 13)
        self.assertEqual(
            metadata["policy"],
            "static_reference_eating_pose_identity_fallback_v1",
        )
        self.assertTrue(metadata["reference_identity_preserved"])
        self.assertFalse(metadata["driver_silhouette_preserved"])
        first_pose = np.asarray(poses[0], dtype=np.uint8)
        first_mask = np.asarray(masks[0], dtype=np.uint8)
        for pose, mask in zip(poses[1:], masks[1:], strict=True):
            self.assertTrue(np.array_equal(np.asarray(pose), first_pose))
            self.assertTrue(np.array_equal(np.asarray(mask), first_mask))

    def test_narrow_kibble_row_keeps_wide_pale_bowl_search_corridor(self) -> None:
        reference = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
        draw = ImageDraw.Draw(reference)
        draw.ellipse((57, 108, 455, 405), fill=(96, 106, 116, 255))
        draw.ellipse((145, 348, 207, 405), fill=(224, 224, 218, 255))
        draw.ellipse((365, 348, 430, 405), fill=(224, 224, 218, 255))
        draw.rounded_rectangle(
            (205, 342, 365, 405),
            radius=22,
            fill=(218, 210, 198, 255),
        )
        # The visible food occupies only the centre 42 px of a 160 px bowl.
        for x in (264, 276, 288, 300):
            draw.ellipse((x, 331, x + 10, 343), fill=(162, 94, 38, 255))

        colored, exclusion, metadata = scail.make_eating_reference_identity_mask(
            reference
        )
        mask = np.asarray(colored, dtype=np.uint8)
        excluded = np.asarray(exclusion, dtype=np.uint8)

        self.assertEqual(
            metadata["bowl_body_method"],
            "warm_food_core_plus_beige_bowl_v1",
        )
        self.assertEqual(
            metadata["food_anchored_search_width_min_fraction"],
            0.32,
        )
        self.assertGreaterEqual(metadata["safe_width_fraction"], 0.25)
        self.assertLessEqual(metadata["safe_width_fraction"], 0.55)
        self.assertEqual(tuple(mask[385, 225]), (255, 0, 0))
        self.assertEqual(tuple(mask[385, 345]), (255, 0, 0))
        self.assertEqual(int(excluded[385, 225]), 255)
        self.assertEqual(int(excluded[385, 345]), 255)
        self.assertEqual(tuple(mask[380, 175]), (0, 0, 255))
        self.assertEqual(tuple(mask[380, 400]), (0, 0, 255))

    def test_kibble_one_pixel_above_old_bottom_cutoff_remains_strict(self) -> None:
        reference = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
        draw = ImageDraw.Draw(reference)
        draw.ellipse((57, 127, 456, 385), fill=(224, 224, 220, 255))
        draw.ellipse((72, 312, 145, 385), fill=(230, 230, 226, 255))
        draw.ellipse((330, 312, 409, 385), fill=(230, 230, 226, 255))
        draw.rectangle((154, 307, 282, 385), fill=(218, 216, 210, 255))
        # This reproduces job 20260812T085706Z-296d1f3e5d after reference
        # normalization: the warm food bbox ends at y=322 while the rounded
        # 76% cutoff is y=323.  Scale rounding must not turn real food into a
        # hard no_warm_food_component failure.
        draw.rectangle((166, 310, 268, 321), fill=(166, 94, 35, 255))

        colored, exclusion, metadata = scail.make_eating_reference_identity_mask(
            reference
        )
        mask = np.asarray(colored, dtype=np.uint8)
        excluded = np.asarray(exclusion, dtype=np.uint8)

        self.assertEqual(metadata["split_status"], "strict")
        self.assertFalse(metadata["fallback_used"])
        self.assertGreaterEqual(metadata["food_component_bottom_tolerance"], 2)
        self.assertEqual(tuple(mask[360, 220]), (255, 0, 0))
        self.assertEqual(int(excluded[360, 220]), 255)
        self.assertEqual(tuple(mask[360, 105]), (0, 0, 255))
        self.assertEqual(tuple(mask[360, 370]), (0, 0, 255))

    def test_neutral_food_and_bowl_use_nonblocking_geometry_fallback(self) -> None:
        reference = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
        draw = ImageDraw.Draw(reference)
        draw.ellipse((57, 112, 455, 399), fill=(92, 102, 112, 255))
        draw.ellipse((145, 342, 207, 399), fill=(220, 220, 216, 255))
        draw.ellipse((365, 342, 430, 399), fill=(220, 220, 216, 255))
        # A fully achromatic bowl and food intentionally provide no warm
        # colour anchor.  Visual uncertainty must be recorded and degraded,
        # never raised as an exception that terminates the video job.
        draw.rectangle((205, 342, 365, 399), fill=(170, 170, 170, 255))
        draw.ellipse((235, 330, 335, 349), fill=(120, 120, 120, 255))

        colored, exclusion, metadata = scail.make_eating_reference_identity_mask(
            reference
        )
        mask = np.asarray(colored, dtype=np.uint8)
        excluded = np.asarray(exclusion, dtype=np.uint8)

        self.assertEqual(
            metadata["split_status"],
            "degraded_nonblocking_fallback",
        )
        self.assertTrue(metadata["fallback_used"])
        self.assertIn("warm_seed_pixels", metadata["fallback_reason"])
        self.assertEqual(
            metadata["fallback_policy"],
            "lower_center_prop_geometry_v1",
        )
        self.assertGreater(int(np.count_nonzero(excluded)), 0)
        self.assertEqual(tuple(mask[370, 285]), (255, 0, 0))
        self.assertEqual(tuple(mask[370, 175]), (0, 0, 255))
        self.assertEqual(tuple(mask[370, 400]), (0, 0, 255))

    def test_empty_reference_remains_a_runtime_integrity_error(self) -> None:
        reference = Image.new("RGBA", (512, 512), (0, 0, 0, 0))

        with self.assertRaisesRegex(ValueError, "empty_alpha"):
            scail.make_eating_reference_identity_mask(reference)

    def test_strict_scoring_mode_can_reject_visual_uncertainty(self) -> None:
        reference = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
        ImageDraw.Draw(reference).ellipse(
            (57, 112, 455, 399),
            fill=(128, 128, 128, 255),
        )

        with self.assertRaisesRegex(ValueError, "warm_seed_pixels"):
            scail.make_eating_reference_identity_mask(
                reference,
                allow_fallback=False,
            )

    def test_eating_model_reference_maps_bowl_as_second_object(self) -> None:
        reference = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
        draw = ImageDraw.Draw(reference)
        draw.ellipse((57, 121, 455, 391), fill=(95, 105, 115, 255))
        draw.rectangle((179, 348, 314, 389), fill=(205, 125, 55, 255))
        draw.ellipse((188, 315, 305, 360), fill=(175, 90, 35, 255))

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "eating-reference.png"
            reference.save(path)
            model_reference, _mask, metadata = scail.make_reference_assets(
                path,
                (512, 512),
                separate_eating_prop=True,
            )

        model_rgba = np.asarray(model_reference.convert("RGBA"), dtype=np.uint8)
        self.assertTrue(metadata["scene_prop_separated"])
        self.assertGreater(
            metadata["reference_prop_fraction"],
            0,
        )
        self.assertTrue(np.any(model_rgba[..., 3] > 0))

    def test_grooming_prompt_uses_reference_palette_and_existing_forelegs(
        self,
    ) -> None:
        self.assertIn("exact reference coat colors", scail.DEFAULT_PROMPT)
        self.assertNotIn("gray-white coat colors", scail.DEFAULT_PROMPT)
        self.assertIn(
            "already contains this one-raised, one-planted foreleg topology",
            scail.DEFAULT_PROMPT,
        )
        self.assertIn(
            "never preserve two planted forepaws and then add a new raised paw",
            scail.DEFAULT_PROMPT,
        )

    def test_toileting_prompt_blocks_dark_contour_outlines(self) -> None:
        self.assertIn("without any black", scail.DOG_TOILETING_PROMPT)
        self.assertIn("black outline", scail.DOG_TOILETING_NEGATIVE_PROMPT)
        self.assertIn("dark tray rim", scail.DOG_TOILETING_NEGATIVE_PROMPT)

    def test_grooming_reference_can_match_driver_contain_fraction(self) -> None:
        reference = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
        ImageDraw.Draw(reference).ellipse(
            (140, 72, 372, 462),
            fill=(180, 180, 180, 255),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "grooming-reference.png"
            reference.save(path)
            _default_reference, _default_mask, default_metadata = (
                scail.make_reference_assets(
                    path,
                    (512, 512),
                )
            )
            _aligned_reference, _aligned_mask, aligned_metadata = (
                scail.make_reference_assets(
                    path,
                    (512, 512),
                    contain_fraction=0.88,
                )
            )

        self.assertEqual(default_metadata["reference_contain_fraction"], 0.84)
        self.assertEqual(aligned_metadata["reference_contain_fraction"], 0.88)
        self.assertGreater(
            aligned_metadata["reference_subject_fraction"],
            default_metadata["reference_subject_fraction"],
        )


class SemanticCleanupComponentTests(unittest.TestCase):
    @staticmethod
    def _outlined_subject() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rgb = np.full((64, 64, 3), 255, dtype=np.uint8)
        alpha = np.zeros((64, 64), dtype=np.uint8)
        alpha[8:56, 8:56] = 255
        rgb[8:56, 8:56] = 232
        outline = np.zeros((64, 64), dtype=bool)
        outline[8:10, 8:56] = True
        outline[54:56, 8:56] = True
        outline[8:56, 8:10] = True
        outline[8:56, 54:56] = True
        rgb[outline] = 18
        return rgb, alpha, outline

    def test_continuous_dark_silhouette_halo_uses_inward_coat_color(
        self,
    ) -> None:
        rgb, alpha, outline = self._outlined_subject()

        cleaned, metrics = scail._remove_dark_silhouette_halo(rgb, alpha)

        self.assertTrue(metrics["applied"])
        self.assertGreater(metrics["replaced_pixels"], 64)
        self.assertGreater(float(cleaned[outline].mean()), 180.0)
        self.assertGreater(float(cleaned[outline].mean()), float(rgb[outline].mean()))

    def test_dark_subject_and_small_edge_marking_are_preserved(self) -> None:
        dark_rgb = np.full((64, 64, 3), 255, dtype=np.uint8)
        dark_alpha = np.zeros((64, 64), dtype=np.uint8)
        dark_alpha[8:56, 8:56] = 255
        dark_rgb[8:56, 8:56] = 24

        dark_cleaned, dark_metrics = scail._remove_dark_silhouette_halo(
            dark_rgb,
            dark_alpha,
        )

        self.assertFalse(dark_metrics["applied"])
        self.assertTrue(np.array_equal(dark_cleaned, dark_rgb))

        marked_rgb = np.full((64, 64, 3), 255, dtype=np.uint8)
        marked_alpha = np.zeros((64, 64), dtype=np.uint8)
        marked_alpha[8:56, 8:56] = 255
        marked_rgb[8:56, 8:56] = 232
        marked_rgb[8:10, 28:34] = 18

        marked_cleaned, marked_metrics = (
            scail._remove_dark_silhouette_halo(
                marked_rgb,
                marked_alpha,
            )
        )

        self.assertFalse(marked_metrics["applied"])
        self.assertTrue(np.array_equal(marked_cleaned, marked_rgb))

    def test_low_alpha_black_fringe_is_recolored_and_pulled_inward(self) -> None:
        rgb = np.zeros((72, 72, 3), dtype=np.uint8)
        alpha = np.zeros((72, 72), dtype=np.uint8)
        rgb[8:64, 8:64] = 18
        alpha[8:64, 8:64] = 40
        rgb[10:62, 10:62] = 20
        alpha[10:62, 10:62] = 180
        rgb[13:59, 13:59] = 232
        alpha[13:59, 13:59] = 255

        cleaned_rgb, cleaned_alpha, metrics = (
            scail.clean_transparent_edge_halo(rgb, alpha)
        )

        self.assertTrue(metrics["applied"])
        self.assertTrue(metrics["rgb_applied"])
        self.assertTrue(metrics["alpha_applied"])
        self.assertGreater(metrics["low_alpha_candidate_pixels"], 64)
        self.assertGreater(metrics["alpha_adjusted_pixels"], 64)
        self.assertLess(int(cleaned_alpha[8, 36]), int(alpha[8, 36]))
        retained_edge = (cleaned_alpha > 0) & (alpha < 255)
        self.assertGreater(float(cleaned_rgb[retained_edge].mean()), 150.0)
        self.assertTrue(metrics["acceptable"])

    def test_eating_contact_shadow_is_removed_without_erasing_dark_tail(self) -> None:
        rgb = np.full((128, 128, 3), 238, dtype=np.uint8)
        alpha = np.zeros((128, 128), dtype=np.uint8)
        alpha[18:105, 14:116] = 255
        rgb[102:109, 34:94] = 18
        alpha[102:109, 34:94] = 255
        rgb[88:112, 98:122] = 24
        alpha[88:112, 98:122] = 255
        prop = Image.new("L", (128, 128), 0)
        ImageDraw.Draw(prop).rectangle((40, 70, 88, 104), fill=255)
        metadata = {"split_status": "strict"}

        with patch.object(
            scail,
            "make_eating_reference_identity_mask",
            return_value=(Image.new("RGB", (128, 128)), prop, metadata),
        ):
            cleaned_alpha, metrics = scail._remove_eating_contact_shadow(
                rgb,
                alpha,
            )

        self.assertTrue(metrics["applied"])
        self.assertGreater(metrics["removed_pixels"], 100)
        self.assertEqual(int(cleaned_alpha[105, 50]), 0)
        self.assertEqual(int(cleaned_alpha[100, 110]), 255)

    def test_high_confidence_subject_fragment_survives_guidance_cleanup(
        self,
    ) -> None:
        hard = np.zeros((32, 48), dtype=np.uint8)
        hard[4:26, 3:15] = 1
        hard[8:20, 20:28] = 1
        hard[3:15, 35:43] = 1
        _count, labels, stats, _centroids = (
            cv2.connectedComponentsWithStats(hard, connectivity=8)
        )
        primary_label = int(labels[8, 8])
        paw_fragment_label = int(labels[10, 22])
        noise_label = int(labels[5, 37])

        guidance = np.zeros_like(hard, dtype=bool)
        guidance[5:25, 4:14] = True
        semantic_core = np.zeros_like(hard, dtype=bool)
        semantic_core[4:26, 3:15] = True
        semantic_core[8:20, 20:28] = True
        semantic_core[3:5, 35:40] = True

        selected, recovered = (
            scail._select_pose_guided_semantic_components(
                labels,
                stats,
                guidance,
                semantic_core,
            )
        )

        self.assertIn(primary_label, selected)
        self.assertIn(paw_fragment_label, selected)
        self.assertIn(paw_fragment_label, recovered)
        self.assertNotIn(noise_label, selected)
        self.assertNotIn(noise_label, recovered)

    def test_low_confidence_detached_noise_is_not_recovered(self) -> None:
        hard = np.zeros((32, 48), dtype=np.uint8)
        hard[4:26, 3:15] = 1
        hard[8:20, 20:28] = 1
        _count, labels, stats, _centroids = (
            cv2.connectedComponentsWithStats(hard, connectivity=8)
        )
        primary_label = int(labels[8, 8])
        noise_label = int(labels[10, 22])

        guidance = np.zeros_like(hard, dtype=bool)
        guidance[5:25, 4:14] = True
        semantic_core = np.zeros_like(hard, dtype=bool)
        semantic_core[4:26, 3:15] = True
        semantic_core[8:12, 20:24] = True

        selected, recovered = (
            scail._select_pose_guided_semantic_components(
                labels,
                stats,
                guidance,
                semantic_core,
            )
        )

        self.assertEqual(selected, [primary_label])
        self.assertEqual(recovered, [])
        self.assertNotIn(noise_label, selected)


class DrivingAnimationDecodeTests(unittest.TestCase):
    def test_grooming_loop_keeps_action_half_and_closes_at_contact(self) -> None:
        self.assertEqual(
            scail.grooming_closed_loop_source_indices(9),
            [4, 5, 6, 7, 8, 7, 6, 5, 4],
        )

    def test_grooming_loop_rejects_non_scail_frame_counts(self) -> None:
        for frame_count in (0, 2, 8):
            with self.subTest(frame_count=frame_count), self.assertRaisesRegex(
                ValueError,
                "odd source frame count",
            ):
                scail.grooming_closed_loop_source_indices(frame_count)

    def test_closed_loop_quality_rejects_large_boundary_jump(self) -> None:
        alpha = np.full((4, 4), 255, dtype=np.uint8)
        frames = [
            (np.full((4, 4, 3), value, dtype=np.uint8), alpha)
            for value in (0, 5, 10, 200)
        ]
        open_quality = scail.transparent_animation_loop_quality(
            frames,
            [0, 1, 2, 3],
        )
        closed_quality = scail.transparent_animation_loop_quality(
            frames,
            [1, 2, 1],
        )

        self.assertFalse(open_quality["acceptable"])
        self.assertTrue(closed_quality["acceptable"])
        self.assertEqual(closed_quality["loop_transition_score"], 0.0)

    def test_action_tail_cosine_closure_preserves_length_and_closes(self) -> None:
        alpha = np.full((4, 4), 255, dtype=np.uint8)
        frames = [
            (np.full((4, 4, 3), value, dtype=np.uint8), alpha)
            for value in (0, 15, 30, 60, 120, 220)
        ]

        closed, metadata = scail.close_transparent_action_tail_loop(
            frames,
            target_frame_index=4,
            maximum_blend_frames=3,
        )

        self.assertEqual(len(closed), len(frames))
        self.assertTrue(np.array_equal(closed[4][0], closed[0][0]))
        self.assertTrue(np.array_equal(closed[5][0], closed[0][0]))
        self.assertGreater(int(closed[2][0][0, 0, 0]), 0)
        self.assertLess(int(closed[3][0][0, 0, 0]), 60)
        self.assertEqual(
            metadata["policy"],
            "drinking_eating_tail_cosine_blend_to_first_v1",
        )
        quality = scail.transparent_animation_loop_quality(
            closed,
            [0, 1, 2, 3, 4],
        )
        self.assertTrue(quality["acceptable"])

    def test_video_driver_decodes_all_frames_with_constant_timing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "driver.avi"
            writer = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                10.0,
                (32, 24),
            )
            self.assertTrue(writer.isOpened())
            try:
                for value in (30, 90, 150):
                    bgr = np.zeros((24, 32, 3), dtype=np.uint8)
                    bgr[..., 1] = 230
                    bgr[6:20, 9:24] = value
                    bgr[0:3, 0:3] = 240
                    writer.write(bgr)
            finally:
                writer.release()

            frames, durations = scail.read_animation_frames(path)
            raw_frames, raw_durations = scail.read_animation_frames(
                path,
                extract_green_screen=False,
            )

        self.assertEqual(len(frames), 3)
        self.assertEqual(durations, [100, 100, 100])
        self.assertEqual(raw_durations, durations)
        self.assertTrue(all(frame.mode == "RGBA" for frame in frames))
        self.assertTrue(all(frame.size == (32, 24) for frame in frames))
        first_rgba = np.asarray(frames[0], dtype=np.uint8)
        self.assertEqual(int(first_rgba[1, 1, 3]), 0)
        self.assertGreater(int(first_rgba[12, 16, 3]), 250)
        first_raw_rgba = np.asarray(raw_frames[0], dtype=np.uint8)
        self.assertEqual(int(first_raw_rgba[10, 2, 3]), 255)
        self.assertGreater(
            int(first_raw_rgba[10, 2, 1]),
            int(first_raw_rgba[10, 2, 0]),
        )
        decoded_means = [
            float(np.asarray(frame.convert("RGB"), dtype=np.uint8).mean())
            for frame in frames
        ]
        self.assertLess(decoded_means[0], decoded_means[1])
        self.assertLess(decoded_means[1], decoded_means[2])

    def test_unknown_driver_format_is_rejected_before_decode(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Unsupported driving animation format",
        ):
            scail.read_animation_frames(Path("driver.txt"))


if __name__ == "__main__":
    unittest.main()
