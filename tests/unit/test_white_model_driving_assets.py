from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from tools.assets import build_white_model_driving_assets as white_model


class WhiteModelDrivingAssetTests(unittest.TestCase):
    @staticmethod
    def _frame() -> tuple[Image.Image, np.ndarray, np.ndarray]:
        rgba = np.zeros((64, 64, 4), dtype=np.uint8)
        rgba[:, :, :3] = (0, 255, 0)
        cat = np.zeros((64, 64), dtype=bool)
        cat[8:56, 6:42] = True
        prop = np.zeros((64, 64), dtype=bool)
        prop[42:58, 42:62] = True
        rgba[cat, :3] = (25, 18, 20)
        rgba[cat, 3] = 255
        rgba[12:50, 24:42, :3] = (242, 230, 218)
        rgba[prop, :3] = (174, 112, 55)
        rgba[prop, 3] = 255
        return Image.fromarray(rgba, mode="RGBA"), cat, prop

    def test_cat_becomes_neutral_white_while_prop_is_preserved(self) -> None:
        frame, cat, prop = self._frame()

        rendered, metrics = white_model.render_white_model_frame(
            frame,
            cat,
            prop,
        )

        pixels = np.asarray(rendered, dtype=np.uint8)
        cat_pixels = pixels[cat]
        self.assertGreaterEqual(int(cat_pixels.min()), 170)
        self.assertTrue(
            np.array_equal(cat_pixels[:, 0], cat_pixels[:, 1])
        )
        self.assertTrue(
            np.array_equal(cat_pixels[:, 1], cat_pixels[:, 2])
        )
        self.assertTrue(np.all(pixels[0, 0] == (0, 255, 0)))
        self.assertTrue(np.all(pixels[prop] == (174, 112, 55)))
        self.assertEqual(metrics["white_rgb_channel_spread_max"], 0)

    def test_source_cat_appearance_cannot_change_white_model(self) -> None:
        frame, cat, prop = self._frame()
        changed = np.asarray(frame, dtype=np.uint8).copy()
        changed[cat, :3] = (205, 34, 146)
        changed[12:28, 6:24, :3] = (8, 231, 73)
        changed_frame = Image.fromarray(changed, mode="RGBA")

        rendered, metrics = white_model.render_white_model_frame(
            frame,
            cat,
            prop,
        )
        changed_rendered, _changed_metrics = (
            white_model.render_white_model_frame(
                changed_frame,
                cat,
                prop,
            )
        )

        self.assertTrue(
            np.array_equal(
                np.asarray(rendered, dtype=np.uint8),
                np.asarray(changed_rendered, dtype=np.uint8),
            )
        )
        self.assertLessEqual(metrics["white_luma_max"], 246)
        self.assertGreaterEqual(metrics["white_luma_min"], 176)
        self.assertFalse(metrics["source_rgb_sampled_for_cat_shading"])
        self.assertFalse(metrics["source_luma_sampled_for_cat_shading"])

    def test_foreleg_is_encoded_as_neutral_soft_depth(self) -> None:
        frame, cat, prop = self._frame()
        foreleg = np.zeros_like(cat)
        foreleg[16:48, 14:25] = True

        plain, _plain_metrics = white_model.render_white_model_frame(
            frame,
            cat,
            prop,
        )
        rendered, metrics = white_model.render_white_model_frame(
            frame,
            cat,
            prop,
            foreleg_mask=foreleg,
        )

        plain_pixels = np.asarray(plain, dtype=np.uint8)
        pixels = np.asarray(rendered, dtype=np.uint8)
        self.assertGreater(
            int(pixels[28:36, 17:22, 0].mean()),
            int(plain_pixels[28:36, 17:22, 0].mean()),
        )
        self.assertTrue(
            np.array_equal(pixels[cat, 0], pixels[cat, 2])
        )
        self.assertEqual(metrics["foreleg_pixels"], int(foreleg.sum()))
        self.assertGreater(metrics["foreleg_relief_max"], 7.0)

    def test_cat_and_prop_masks_are_required_to_be_disjoint(self) -> None:
        frame, cat, prop = self._frame()
        prop = prop.copy()
        prop[20:22, 20:22] = True

        with self.assertRaisesRegex(
            ValueError,
            "white_model_driver_cat_prop_overlap",
        ):
            white_model.render_white_model_frame(
                frame,
                cat,
                prop,
            )


if __name__ == "__main__":
    unittest.main()
