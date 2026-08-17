from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.assets import build_dog_white_model_driving_assets as builder


class DogWhiteModelAssetBuilderTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
