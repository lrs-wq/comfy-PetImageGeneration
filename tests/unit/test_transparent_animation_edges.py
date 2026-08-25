from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.comfy_pet_pipeline.media.transparent_animation import (
    GIF_ALPHA_THRESHOLD,
    _binary_alpha_edge_metrics,
    _encode_soft_alpha_apng,
    _gif_alpha,
    validate_animation,
)
from src.comfy_pet_pipeline.pipeline.subject_replacement import (
    transparent_guidance_component_area_fraction_minimum,
    transparent_guidance_component_coverage_minimum,
    transparent_guidance_dynamic_component_area_fraction_maximum,
    transparent_guidance_dynamic_component_radius,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_transparent_video_cli():
    script_path = PROJECT_ROOT / "tools/media/scail2_transparent_gif_from_video.py"
    spec = importlib.util.spec_from_file_location(
        "scail2_transparent_gif_from_video",
        script_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TransparentVideoCliTests(unittest.TestCase):
    def test_all_subject_replacement_actions_are_accepted(self):
        cli = load_transparent_video_cli()

        self.assertEqual(
            set(cli.SUPPORTED_ACTIONS),
            {
                "eating",
                "grooming",
                "drinking",
                "running",
                "sleeping",
                "standing",
                "toileting",
            },
        )
        for action in cli.SUPPORTED_ACTIONS:
            with self.subTest(action=action):
                args = cli.build_parser().parse_args(
                    ["video.mp4", "mask.mp4", "output.gif", "--action", action]
                )
                self.assertEqual(args.action, action)

    def test_drinking_and_running_use_bounded_limb_proximity(self):
        for action in ("drinking", "running"):
            with self.subTest(action=action):
                self.assertEqual(
                    transparent_guidance_component_coverage_minimum(action),
                    0.45,
                )
                self.assertEqual(
                    transparent_guidance_component_area_fraction_minimum(action),
                    0.005,
                )
                self.assertEqual(
                    transparent_guidance_dynamic_component_area_fraction_maximum(
                        action
                    ),
                    0.10,
                )
                self.assertEqual(
                    transparent_guidance_dynamic_component_radius(action),
                    12,
                )
        for action in (
            "eating",
            "grooming",
            "sleeping",
            "standing",
            "toileting",
            None,
        ):
            with self.subTest(action=action):
                self.assertEqual(
                    transparent_guidance_component_coverage_minimum(action),
                    0.45,
                )
                self.assertEqual(
                    transparent_guidance_component_area_fraction_minimum(action),
                    0.005,
                )
                self.assertEqual(
                    transparent_guidance_dynamic_component_area_fraction_maximum(
                        action
                    ),
                    0.0,
                )
                self.assertEqual(
                    transparent_guidance_dynamic_component_radius(action),
                    0,
                )


class CleanGifAlphaTests(unittest.TestCase):
    def test_uniform_soft_alpha_is_not_turned_into_ordered_dots(self):
        below = np.full((64, 64), GIF_ALPHA_THRESHOLD - 16, dtype=np.uint8)
        above = np.full((64, 64), GIF_ALPHA_THRESHOLD + 16, dtype=np.uint8)

        self.assertEqual(int(np.count_nonzero(_gif_alpha(below))), 0)
        self.assertEqual(
            int(np.count_nonzero(_gif_alpha(above))),
            int(above.size),
        )

    def test_tiny_detached_dust_is_removed_without_dropping_real_components(self):
        alpha = np.zeros((96, 96), dtype=np.uint8)
        alpha[18:74, 16:65] = 255
        alpha[68:80, 72:86] = 255
        alpha[8, 8] = 255
        alpha[88, 90] = 255

        hard = _gif_alpha(alpha)
        metrics = _binary_alpha_edge_metrics(hard)

        self.assertEqual(int(hard[8, 8]), 0)
        self.assertEqual(int(hard[88, 90]), 0)
        self.assertEqual(int(hard[40, 40]), 255)
        self.assertEqual(int(hard[72, 78]), 255)
        self.assertEqual(metrics["foreground_components"], 2)
        self.assertEqual(metrics["tiny_foreground_components"], 0)

    def test_soft_round_contour_has_no_checkerboard_alpha_pattern(self):
        alpha = np.zeros((128, 128), dtype=np.uint8)
        cv2.circle(alpha, (64, 64), 42, 255, thickness=-1)
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=2.0, sigmaY=2.0)

        metrics = _binary_alpha_edge_metrics(_gif_alpha(alpha))

        self.assertEqual(metrics["tiny_foreground_components"], 0)
        self.assertEqual(metrics["checkerboard_2x2_blocks"], 0)


class SoftAlphaApngTests(unittest.TestCase):
    def test_apng_preserves_soft_alpha_frames(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths: list[Path] = []
            for index in range(3):
                alpha = np.zeros((64, 64), dtype=np.uint8)
                cv2.circle(alpha, (24 + index * 4, 32), 15, 255, thickness=-1)
                alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=1.5, sigmaY=1.5)
                rgba = np.zeros((64, 64, 4), dtype=np.uint8)
                rgba[:, :, :3] = (210, 180, 150)
                rgba[:, :, 3] = alpha
                path = root / f"frame_{index:03d}.png"
                Image.fromarray(rgba, mode="RGBA").save(path)
                paths.append(path)

            target = root / "soft.png"
            _encode_soft_alpha_apng(paths, target, fps=20)
            metrics = validate_animation(
                target,
                size=64,
                frame_count=3,
                duration_ms=150,
            )

            self.assertEqual(metrics["soft_alpha_frames"], 3)
            self.assertGreater(metrics["soft_alpha_pixels"], 0)
            self.assertTrue(metrics["full_2px_border_transparent"])


if __name__ == "__main__":
    unittest.main()
