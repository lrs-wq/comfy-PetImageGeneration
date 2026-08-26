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
    REFERENCE_ALIGNED_ADAPTIVE_EXTRACTION_POLICY,
    REFERENCE_ALIGNED_BOUNDED_CANVAS_EXTRACTION_STRATEGY,
    REFERENCE_ALIGNED_SEMANTIC_EXTRACTION_STRATEGY,
    REFERENCE_ALIGNED_UNIFORM_BACKGROUND_POLICY,
    _reference_guided_uniform_background_mask,
    encode_static_mask_video,
    reference_aligned_transparent_extraction_strategy,
    transparent_guidance_area_ratio_minimum,
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

    def test_reference_aligned_uniform_background_flag_is_opt_in(self):
        cli = load_transparent_video_cli()

        default = cli.build_parser().parse_args(
            ["video.mp4", "mask.mp4", "output.gif", "--action", "toileting"]
        )
        enabled = cli.build_parser().parse_args(
            [
                "video.mp4",
                "mask.mp4",
                "output.gif",
                "--action",
                "toileting",
                "--reference-aligned-uniform-background",
            ]
        )
        adaptive = cli.build_parser().parse_args(
            [
                "video.mp4",
                "mask.mp4",
                "output.gif",
                "--action",
                "toileting",
                "--reference-aligned-adaptive-extraction",
            ]
        )

        self.assertFalse(default.reference_aligned_uniform_background)
        self.assertTrue(enabled.reference_aligned_uniform_background)
        self.assertTrue(adaptive.reference_aligned_uniform_background)

    def test_reference_aligned_extraction_is_adaptive(self):
        self.assertEqual(
            REFERENCE_ALIGNED_ADAPTIVE_EXTRACTION_POLICY,
            "reference_guided_adaptive_semantic_or_bounded_canvas_matte_v3",
        )
        self.assertEqual(
            reference_aligned_transparent_extraction_strategy(7.99),
            REFERENCE_ALIGNED_BOUNDED_CANVAS_EXTRACTION_STRATEGY,
        )
        self.assertEqual(
            reference_aligned_transparent_extraction_strategy(20.70),
            REFERENCE_ALIGNED_SEMANTIC_EXTRACTION_STRATEGY,
        )

    def test_drinking_and_running_use_bounded_limb_proximity(self):
        for action in ("drinking", "running"):
            with self.subTest(action=action):
                self.assertEqual(
                    transparent_guidance_component_coverage_minimum(action),
                    0.45,
                )
                self.assertEqual(
                    transparent_guidance_area_ratio_minimum(action),
                    0.55,
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
            None,
        ):
            with self.subTest(action=action):
                self.assertEqual(
                    transparent_guidance_component_coverage_minimum(action),
                    0.45,
                )
                self.assertEqual(
                    transparent_guidance_area_ratio_minimum(action),
                    0.55,
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

    def test_toileting_requires_the_validated_tray_footprint(self):
        self.assertEqual(
            transparent_guidance_component_coverage_minimum("toileting"),
            0.70,
        )
        self.assertEqual(
            transparent_guidance_area_ratio_minimum("toileting"),
            0.78,
        )
        self.assertEqual(
            transparent_guidance_component_area_fraction_minimum("toileting"),
            0.005,
        )
        self.assertEqual(
            transparent_guidance_dynamic_component_area_fraction_maximum(
                "toileting"
            ),
            0.0,
        )
        self.assertEqual(transparent_guidance_dynamic_component_radius("toileting"), 0)

    def test_uniform_background_extraction_retains_dog_and_wide_tray(self):
        rgb = np.full((128, 128, 3), (234, 245, 246), dtype=np.uint8)
        guidance_u8 = np.zeros((128, 128), dtype=np.uint8)
        cv2.rectangle(rgb, (16, 82), (112, 112), (178, 184, 181), thickness=-1)
        cv2.rectangle(guidance_u8, (16, 82), (112, 112), 1, thickness=-1)
        cv2.ellipse(rgb, (64, 58), (27, 43), 0, 0, 360, (224, 188, 132), -1)
        cv2.ellipse(guidance_u8, (64, 58), (27, 43), 0, 0, 360, 1, -1)
        cv2.circle(rgb, (54, 48), 4, (30, 24, 18), thickness=-1)
        cv2.circle(rgb, (74, 48), 4, (30, 24, 18), thickness=-1)
        guidance = guidance_u8.astype(bool)

        output_rgb, alpha, metrics = _reference_guided_uniform_background_mask(
            rgb,
            guidance,
            frame_index=0,
            guidance_component_coverage_required_min=0.70,
            guidance_area_ratio_required_min=0.78,
            guidance_component_area_fraction_required_min=0.005,
        )

        self.assertEqual(
            REFERENCE_ALIGNED_UNIFORM_BACKGROUND_POLICY,
            "reference_guided_bounded_canvas_subject_and_prop_matte_v2",
        )
        self.assertEqual(int(alpha[0, 0]), 0)
        self.assertGreaterEqual(int(alpha[96, 24]), 128)
        self.assertGreaterEqual(int(alpha[58, 64]), 128)
        self.assertGreater(metrics["guidance_coverage"], 0.95)
        self.assertGreater(metrics["area_ratio"], 0.95)
        self.assertLess(metrics["background_border_distance_p999"], 1.0)
        self.assertTrue(np.all(output_rgb[alpha == 0] == 0))

    def test_bounded_canvas_gradient_retains_reference_aligned_tray(self):
        x_gradient = np.linspace(-10.0, 10.0, 128, dtype=np.float32)[None, :]
        y_gradient = np.linspace(-5.0, 5.0, 128, dtype=np.float32)[:, None]
        rgb = np.empty((128, 128, 3), dtype=np.float32)
        rgb[:, :, 0] = 230.0 + x_gradient + y_gradient
        rgb[:, :, 1] = 241.0 + x_gradient * 0.7 - y_gradient
        rgb[:, :, 2] = 240.0 - x_gradient * 0.5 + y_gradient * 0.8
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        guidance_u8 = np.zeros((128, 128), dtype=np.uint8)
        cv2.rectangle(rgb, (16, 82), (112, 112), (170, 180, 176), -1)
        cv2.rectangle(guidance_u8, (16, 82), (112, 112), 1, -1)
        cv2.ellipse(rgb, (64, 58), (27, 43), 0, 0, 360, (222, 184, 126), -1)
        cv2.ellipse(guidance_u8, (64, 58), (27, 43), 0, 0, 360, 1, -1)

        _output_rgb, alpha, metrics = _reference_guided_uniform_background_mask(
            rgb,
            guidance_u8.astype(bool),
            frame_index=0,
            guidance_component_coverage_required_min=0.70,
            guidance_area_ratio_required_min=0.78,
            guidance_component_area_fraction_required_min=0.005,
        )

        self.assertGreater(metrics["background_border_distance_p999"], 8.0)
        self.assertLess(metrics["background_border_distance_p999"], 32.0)
        self.assertGreater(metrics["guidance_coverage"], 0.80)
        self.assertGreater(metrics["area_ratio"], 0.80)
        self.assertEqual(int(alpha[0, 0]), 0)
        self.assertGreaterEqual(int(alpha[96, 24]), 128)

    @unittest.skipUnless(importlib.util.find_spec("av"), "PyAV is not installed")
    def test_static_reference_mask_normalizes_black_background_losslessly(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "reference-mask.png"
            target = root / "reference-mask.mp4"
            mask = np.zeros((512, 512, 3), dtype=np.uint8)
            cv2.rectangle(mask, (54, 56), (458, 460), (0, 0, 255), -1)
            Image.fromarray(mask, mode="RGB").save(source)

            metrics = encode_static_mask_video(
                source,
                target,
                frame_count=3,
                fps=16.0,
            )

        self.assertEqual(metrics["decoded_frames"], 3)
        self.assertEqual(metrics["unknown_fraction"], 0.0)
        self.assertGreater(metrics["blue_fraction"], 0.60)
        self.assertGreater(metrics["white_fraction"], 0.30)
        self.assertEqual(
            metrics["normalization"],
            "blue_red_roles_on_pure_white_background_v1",
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
