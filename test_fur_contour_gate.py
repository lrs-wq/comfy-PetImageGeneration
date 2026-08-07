from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from fur_contour_gate import (
    FurContourThresholds,
    compare_fur_contours,
    evaluate_animation_fur_quality,
)


FRAME_COUNT = 80
SIZE = 128


def synthetic_cat_frame(
    frame_index: int,
    *,
    translate: tuple[int, int] = (0, 0),
    scale: float = 1.0,
    back_raise: int = 0,
    back_widen: int = 0,
    jagged: bool = False,
    visible: bool = True,
) -> Image.Image:
    image = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    if not visible:
        return image
    layer = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    head_motion = int(round(math.sin(frame_index * math.pi / 10.0) * 5.0))
    frame_shade = 80 + frame_index % 100

    # Tail and head keep the overall action extent stable.  The rear body is the
    # portion intentionally measured by the gate.
    draw.ellipse((12, 72, 38, 98), fill=(70, 70, 70, 255))
    draw.ellipse(
        (
            26 - back_widen,
            43 - back_raise,
            91 + back_widen,
            101,
        ),
        fill=(frame_shade, frame_shade, frame_shade, 255),
    )
    draw.ellipse(
        (76, 54 + head_motion, 112, 90 + head_motion),
        fill=(130, 130, 130, 255),
    )
    draw.rectangle((76, 78, 88, 105), fill=(105, 105, 105, 255))
    draw.rectangle((91, 78, 103, 105), fill=(105, 105, 105, 255))
    if jagged:
        for x in range(32, 83, 7):
            height = 10 if (x // 7) % 2 else 6
            draw.polygon(
                [(x - 3, 49), (x, 49 - height), (x + 3, 49)],
                fill=(100, 100, 100, 255),
            )

    if scale != 1.0:
        resized = layer.resize(
            (
                max(1, int(round(SIZE * scale))),
                max(1, int(round(SIZE * scale))),
            ),
            Image.Resampling.NEAREST,
        )
        scaled = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
        scaled.alpha_composite(
            resized,
            (
                (SIZE - resized.width) // 2,
                (SIZE - resized.height) // 2,
            ),
        )
        layer = scaled
    image.alpha_composite(layer, translate)
    return image


def save_gif(path: Path, frames: list[Image.Image]) -> None:
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=50,
        loop=0,
        disposal=2,
        optimize=False,
    )


class FurContourGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.reference = self.root / "reference.gif"
        save_gif(
            self.reference,
            [synthetic_cat_frame(index) for index in range(FRAME_COUNT)],
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def compare(self, name: str, frames: list[Image.Image]) -> dict:
        candidate = self.root / name
        save_gif(candidate, frames)
        return compare_fur_contours(candidate, self.reference)

    def test_translation_and_slight_scaling_with_head_motion_pass(self) -> None:
        result = self.compare(
            "translated_scaled.gif",
            [
                synthetic_cat_frame(
                    index + 2,
                    translate=(2, -1),
                    scale=1.02,
                )
                for index in range(FRAME_COUNT)
            ],
        )
        self.assertTrue(result["acceptable"], result)
        self.assertEqual(result["issues"], [])

    def test_raised_back_fails(self) -> None:
        result = self.compare(
            "raised_back.gif",
            [
                synthetic_cat_frame(index, back_raise=9)
                for index in range(FRAME_COUNT)
            ],
        )
        self.assertFalse(result["acceptable"], result)
        self.assertIn(
            "dorsal_contour_expanded",
            result["observed_moderate_issues"],
        )

    def test_widened_rear_body_fails(self) -> None:
        result = self.compare(
            "widened_back.gif",
            [
                synthetic_cat_frame(index, back_widen=9)
                for index in range(FRAME_COUNT)
            ],
        )
        self.assertFalse(result["acceptable"], result)
        self.assertIn("rear_area_expanded", result["observed_moderate_issues"])

    def test_jagged_dorsal_contour_fails(self) -> None:
        result = self.compare(
            "jagged_back.gif",
            [
                synthetic_cat_frame(index, jagged=True)
                for index in range(FRAME_COUNT)
            ],
        )
        self.assertFalse(result["acceptable"], result)
        self.assertTrue(
            {
                "top_contour_roughness",
                "severe_top_contour_roughness",
            }
            & set(
                result["observed_moderate_issues"]
                + result["observed_severe_issues"]
            ),
            result,
        )

    def test_intermitent_p90_expansion_fails_even_when_median_is_clean(self) -> None:
        frames = []
        for index in range(FRAME_COUNT):
            expanded = index >= int(FRAME_COUNT * 0.80)
            frames.append(
                synthetic_cat_frame(
                    index,
                    back_raise=9 if expanded else 0,
                    jagged=expanded,
                )
            )
        result = self.compare("intermittent_p90_expansion.gif", frames)

        self.assertFalse(result["acceptable"], result)
        self.assertTrue(
            {
                "torso_height_p90_expanded",
                "top_contour_roughness_p90",
            }
            & set(result["observed_moderate_issues"]),
            result,
        )

    def test_insufficient_valid_frames_fails_closed(self) -> None:
        result = self.compare(
            "too_short.gif",
            [synthetic_cat_frame(index) for index in range(12)],
        )
        self.assertFalse(result["acceptable"], result)
        self.assertIn("candidate_insufficient_valid_frames", result["issues"])

    def evaluate_uploaded(
        self,
        name: str,
        frames: list[Image.Image],
        *,
        thresholds: FurContourThresholds | None = None,
    ) -> dict:
        candidate = self.root / name
        save_gif(candidate, frames)
        return evaluate_animation_fur_quality(
            candidate,
            thresholds=thresholds,
        )

    def test_uploaded_subject_smooth_contour_passes_without_reference(self) -> None:
        result = self.evaluate_uploaded(
            "uploaded_smooth.gif",
            [
                synthetic_cat_frame(
                    index,
                    translate=(3, -2),
                    scale=1.08,
                )
                for index in range(FRAME_COUNT)
            ],
        )

        self.assertTrue(result["acceptable"], result)
        self.assertEqual(result["issues"], [])
        self.assertFalse(result["policy"]["reference_geometry_comparison"])
        self.assertNotIn("reference", result)

    def test_uploaded_subject_rough_contour_fails_absolute_gate(self) -> None:
        result = self.evaluate_uploaded(
            "uploaded_jagged.gif",
            [
                synthetic_cat_frame(index, jagged=True)
                for index in range(FRAME_COUNT)
            ],
        )

        self.assertFalse(result["acceptable"], result)
        self.assertIn(
            "severe_top_contour_roughness",
            result["observed_severe_issues"],
        )
        roughness = result["candidate"]["aggregates"][
            "top_contour_roughness_norm"
        ]["median"]
        self.assertGreater(
            roughness,
            result["thresholds"]["severe_roughness_norm_max"],
        )

    def test_uploaded_subject_moderate_absolute_limits_fail_together(self) -> None:
        result = self.evaluate_uploaded(
            "uploaded_moderate_jagged.gif",
            [
                synthetic_cat_frame(index, jagged=True)
                for index in range(FRAME_COUNT)
            ],
            thresholds=FurContourThresholds(
                severe_roughness_norm_max=1.0,
            ),
        )

        self.assertFalse(result["acceptable"], result)
        self.assertEqual(result["observed_severe_issues"], [])
        self.assertEqual(
            set(result["observed_moderate_issues"]),
            {
                "top_contour_roughness",
                "top_contour_roughness_p90",
            },
        )
        self.assertEqual(
            set(result["issues"]),
            {
                "top_contour_roughness",
                "top_contour_roughness_p90",
            },
        )

    def test_uploaded_subject_invalid_frames_fail_closed(self) -> None:
        result = self.evaluate_uploaded(
            "uploaded_invalid.gif",
            [
                synthetic_cat_frame(index, visible=index < 8)
                for index in range(FRAME_COUNT)
            ],
        )

        self.assertFalse(result["acceptable"], result)
        self.assertIn("candidate_insufficient_valid_frames", result["issues"])
        self.assertIn(
            "candidate_insufficient_valid_frame_fraction",
            result["issues"],
        )


if __name__ == "__main__":
    unittest.main()
