from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.assets import build_dog_white_model_driving_assets as builder


class DogWhiteModelAssetBuilderTests(unittest.TestCase):
    @staticmethod
    def write_standing_frame(path: Path) -> None:
        pixels = bytearray([0, 255, 0] * (builder.OUTPUT_SIZE**2))

        def paint(x: int, y: int, rgb: tuple[int, int, int]) -> None:
            offset = (y * builder.OUTPUT_SIZE + x) * 3
            pixels[offset : offset + 3] = bytes(rgb)

        for y in range(150, 205):
            for x in range(154, 206):
                paint(x, y, (230, 230, 230))
        for y in range(158, 184):
            for x in range(163, 198):
                if ((x - 180) / 18) ** 2 + ((y - 171) / 14) ** 2 <= 1:
                    paint(x, y, (55, 43, 48))
        for y in range(163, 196):
            half_width = max(4, round(14 - (y - 163) * 0.22))
            for x in range(180 - half_width, 181 + half_width):
                paint(x, y, (205, 105, 125))
        path.write_bytes(b"P6\n512 512\n255\n" + bytes(pixels))

    def runtime_command(self, action: str) -> list[str]:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.mp4"
            target = root / "runtime.mp4"
            with (
                patch.object(builder, "run") as run,
                patch.object(builder.os, "replace"),
            ):
                builder.build_runtime(
                    "ffmpeg",
                    source,
                    target,
                    dict(builder.ACTION_NORMALIZATION[action]),
                )
        return list(run.call_args.args[0])

    def test_eating_uses_complete_down_up_down_cycle_at_double_speed(self) -> None:
        command = self.runtime_command("eating")
        video_filter = command[command.index("-vf") + 1]

        self.assertNotIn("-ss", command)
        self.assertEqual(command[command.index("-t") + 1], "10.041667")
        self.assertIn("setpts=0.500000000*PTS", video_filter)
        self.assertEqual(
            builder.REFERENCE_FILENAMES["eating"],
            "eating_white_model_reference_frame008_cutout.png",
        )
        self.assertEqual(
            builder.ACTION_NORMALIZATION["eating"]["minimum_endpoint_ssim"],
            0.95,
        )

    def test_standing_keeps_existing_second_half_timing(self) -> None:
        command = self.runtime_command("standing")
        video_filter = command[command.index("-vf") + 1]

        self.assertEqual(command[command.index("-ss") + 1], "5.0175")
        self.assertEqual(command[command.index("-t") + 1], "5.0625")
        self.assertNotIn("setpts=", video_filter)

    def test_standing_tongue_extends_and_returns_to_loop_phase(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            start = root / "start.ppm"
            extended = root / "extended.ppm"
            end = root / "end.ppm"
            for path in (start, extended, end):
                self.write_standing_frame(path)

            start_metrics = builder._animate_standing_ppm(start, 0)
            extended_metrics = builder._animate_standing_ppm(extended, 20)
            end_metrics = builder._animate_standing_ppm(end, 80)

        self.assertEqual(start_metrics["extension_scale"], 1.0)
        self.assertEqual(end_metrics["extension_scale"], 1.0)
        self.assertEqual(
            start_metrics["tongue_tip_y"],
            end_metrics["tongue_tip_y"],
        )
        self.assertGreaterEqual(
            extended_metrics["tongue_tip_y"] - start_metrics["tongue_tip_y"],
            builder.STANDING_MOUTH_TONGUE_MOTION[
                "minimum_tongue_tip_y_span_pixels"
            ],
        )
        self.assertGreaterEqual(
            start_metrics["visible_mouth_cavity_dark_pixels"],
            builder.STANDING_MOUTH_TONGUE_MOTION[
                "minimum_visible_mouth_cavity_dark_pixels"
            ],
        )


if __name__ == "__main__":
    unittest.main()
