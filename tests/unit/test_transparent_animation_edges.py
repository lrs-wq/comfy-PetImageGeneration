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
    DRINKING_STATIC_BOWL_ALPHA_RETENTION_MIN,
    DRINKING_STATIC_BOWL_RECOVERY_DILATION_RADIUS,
    EATING_STATIC_BOWL_ALPHA_RETENTION_MIN,
    EATING_STATIC_BOWL_ALPHA_THRESHOLD,
    EATING_STATIC_BOWL_GUIDANCE_SUPPORT_RATIO_MIN,
    REFERENCE_ALIGNED_ADAPTIVE_EXTRACTION_POLICY,
    REFERENCE_ALIGNED_BOUNDED_CANVAS_EXTRACTION_STRATEGY,
    REFERENCE_ALIGNED_HYBRID_EXTRACTION_STRATEGY,
    REFERENCE_ALIGNED_SEMANTIC_EXTRACTION_STRATEGY,
    REFERENCE_ALIGNED_UNIFORM_BACKGROUND_POLICY,
    _canvas_border_color_distance_p999,
    _eating_static_bowl_supported_core,
    _eating_static_bowl_temporal_reference,
    _merge_reference_aligned_static_prop,
    _reference_guided_uniform_background_mask,
    clean_transparent_edge_halo,
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
            "reference_guided_adaptive_semantic_bounded_hybrid_matte_v5",
        )
        self.assertEqual(
            reference_aligned_transparent_extraction_strategy(7.99),
            REFERENCE_ALIGNED_BOUNDED_CANVAS_EXTRACTION_STRATEGY,
        )
        self.assertEqual(
            reference_aligned_transparent_extraction_strategy(20.70),
            REFERENCE_ALIGNED_HYBRID_EXTRACTION_STRATEGY,
        )
        self.assertEqual(
            reference_aligned_transparent_extraction_strategy(40.0),
            REFERENCE_ALIGNED_SEMANTIC_EXTRACTION_STRATEGY,
        )

    def test_canvas_measurement_ignores_guided_tail_touching_border(self):
        rgb = np.full((128, 128, 3), 255, dtype=np.uint8)
        guidance = np.zeros((128, 128), dtype=np.uint8)
        cv2.rectangle(rgb, (116, 30), (127, 90), (178, 112, 62), -1)
        cv2.rectangle(guidance, (112, 26), (127, 94), 1, -1)

        unmasked = _canvas_border_color_distance_p999(rgb)
        masked = _canvas_border_color_distance_p999(
            rgb,
            exclusion_mask=guidance.astype(bool),
        )

        self.assertGreater(unmasked, 32.0)
        self.assertLess(masked, 1.0)

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
            0.75,
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
            "reference_guided_bounded_canvas_subject_and_prop_matte_v3",
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

    def test_static_red_prop_guidance_recovers_low_contrast_toilet_tray(self):
        rgb = np.full((128, 128, 3), (214, 226, 217), dtype=np.uint8)
        identity = np.zeros((128, 128), dtype=np.uint8)
        static_prop = np.zeros((128, 128), dtype=np.uint8)
        cv2.ellipse(rgb, (64, 54), (25, 39), 0, 0, 360, (210, 172, 116), -1)
        cv2.ellipse(identity, (64, 54), (25, 39), 0, 0, 360, 1, -1)
        cv2.rectangle(rgb, (14, 84), (114, 113), (220, 229, 222), -1)
        cv2.rectangle(static_prop, (14, 84), (114, 113), 1, -1)
        guidance = (identity | static_prop).astype(bool)

        output_rgb, alpha, metrics = _reference_guided_uniform_background_mask(
            rgb,
            guidance,
            frame_index=0,
            guidance_component_coverage_required_min=0.70,
            guidance_area_ratio_required_min=0.75,
            guidance_component_area_fraction_required_min=0.005,
            static_prop_guidance=static_prop.astype(bool),
        )

        self.assertGreaterEqual(int(alpha[100, 24]), 128)
        self.assertGreater(metrics["static_prop_recovered_pixels"], 1000)
        self.assertTrue(metrics["static_prop_guidance_preserved"])
        self.assertGreater(metrics["guidance_coverage"], 0.95)
        self.assertEqual(
            int(output_rgb[100, 24, 0]),
            int(output_rgb[100, 24, 1]),
        )
        self.assertEqual(
            int(output_rgb[100, 24, 1]),
            int(output_rgb[100, 24, 2]),
        )

    def test_eating_static_bowl_recovery_rejects_unsupported_guide_overhang(self):
        rgb = np.full((128, 128, 3), (238, 240, 238), dtype=np.uint8)
        identity = np.zeros((128, 128), dtype=np.uint8)
        static_prop = np.zeros((128, 128), dtype=np.uint8)
        cv2.ellipse(rgb, (64, 52), (25, 38), 0, 0, 360, (205, 155, 92), -1)
        cv2.ellipse(identity, (64, 52), (25, 38), 0, 0, 360, 1, -1)
        cv2.rectangle(rgb, (16, 82), (112, 108), (170, 180, 176), -1)
        cv2.rectangle(static_prop, (16, 78), (112, 118), 1, -1)
        guidance = (identity | static_prop).astype(bool)

        _output_rgb, bounded_alpha, metrics = _reference_guided_uniform_background_mask(
            rgb,
            guidance,
            frame_index=0,
            guidance_component_coverage_required_min=0.45,
            guidance_area_ratio_required_min=0.55,
            guidance_component_area_fraction_required_min=0.005,
            static_prop_guidance=static_prop.astype(bool),
            force_static_prop_guidance=False,
        )
        supported_core, support_metrics = _eating_static_bowl_supported_core(
            bounded_alpha,
            static_prop.astype(bool),
        )

        self.assertGreaterEqual(int(bounded_alpha[96, 24]), 128)
        self.assertEqual(int(bounded_alpha[116, 24]), 0)
        self.assertFalse(metrics["static_prop_guidance_forced"])
        self.assertGreater(
            support_metrics["static_prop_bounded_support_ratio"],
            EATING_STATIC_BOWL_GUIDANCE_SUPPORT_RATIO_MIN,
        )
        self.assertTrue(supported_core[96, 24])
        self.assertFalse(supported_core[116, 24])

    def test_eating_static_bowl_temporal_reference_restores_missing_frame_without_overhang(
        self,
    ) -> None:
        static_prop = np.zeros((96, 96), dtype=np.uint8)
        cv2.rectangle(static_prop, (18, 58), (78, 88), 1, -1)
        bounded_frames: list[tuple[np.ndarray, np.ndarray]] = []
        guidances: list[np.ndarray] = []
        for index in range(5):
            rgb = np.zeros((96, 96, 3), dtype=np.uint8)
            alpha = np.zeros((96, 96), dtype=np.uint8)
            if index != 2:
                cv2.rectangle(rgb, (20, 60), (76, 80), (172, 180, 176), -1)
                cv2.rectangle(alpha, (20, 60), (76, 80), 255, -1)
            bounded_frames.append((rgb, alpha))
            guidances.append(static_prop.astype(bool))

        canonical_rgb, canonical_alpha, core, metrics, frame_metrics = (
            _eating_static_bowl_temporal_reference(
                bounded_frames,
                guidances,
            )
        )
        output_rgb, output_alpha, recovery = _merge_reference_aligned_static_prop(
            np.zeros((96, 96, 3), dtype=np.uint8),
            np.zeros((96, 96), dtype=np.uint8),
            *bounded_frames[2],
            static_prop.astype(bool),
            temporal_reference_rgb=canonical_rgb,
            temporal_reference_alpha=canonical_alpha,
        )

        self.assertEqual(metrics["reliable_frame_count"], 4)
        self.assertLess(
            frame_metrics[2]["static_prop_bounded_support_ratio"],
            EATING_STATIC_BOWL_GUIDANCE_SUPPORT_RATIO_MIN,
        )
        self.assertTrue(core[68, 48])
        self.assertGreaterEqual(int(output_alpha[68, 48]), 220)
        self.assertEqual(tuple(output_rgb[68, 48]), (172, 180, 176))
        self.assertEqual(int(output_alpha[86, 48]), 0)
        self.assertGreater(
            recovery["static_prop_temporal_reference_added_pixels"],
            500,
        )

    def test_drinking_static_bowl_recovery_does_not_replace_animal_matte(self):
        rgb = np.zeros((96, 96, 3), dtype=np.uint8)
        alpha = np.zeros((96, 96), dtype=np.uint8)
        bounded_rgb = np.zeros((96, 96, 3), dtype=np.uint8)
        bounded_alpha = np.zeros((96, 96), dtype=np.uint8)
        static_bowl = np.zeros((96, 96), dtype=np.uint8)

        cv2.circle(rgb, (40, 36), 18, (212, 176, 125), -1)
        cv2.circle(alpha, (40, 36), 18, 255, -1)
        bounded_rgb[:] = rgb
        bounded_rgb[64:78, 46:82] = (172, 178, 176)
        bounded_alpha[:] = alpha
        bounded_alpha[64:78, 46:82] = 255
        static_bowl[66:76, 48:80] = 1
        # Simulate noisy bounded-canvas pixels far from the bowl. They must not
        # expand the otherwise clean semantic animal silhouette.
        bounded_rgb[10:16, 10:16] = (220, 220, 220)
        bounded_alpha[10:16, 10:16] = 255

        output_rgb, output_alpha, metrics = (
            _merge_reference_aligned_static_prop(
                rgb,
                alpha,
                bounded_rgb,
                bounded_alpha,
                static_bowl.astype(bool),
            )
        )

        self.assertEqual(int(output_alpha[70, 60]), 255)
        self.assertEqual(int(output_alpha[12, 12]), 0)
        self.assertEqual(tuple(output_rgb[70, 60]), (172, 178, 176))
        self.assertGreaterEqual(
            metrics["static_prop_output_retention_ratio"],
            DRINKING_STATIC_BOWL_ALPHA_RETENTION_MIN,
        )
        self.assertEqual(
            metrics["static_prop_recovery_dilation_radius"],
            float(DRINKING_STATIC_BOWL_RECOVERY_DILATION_RADIUS),
        )
        self.assertGreater(metrics["static_prop_recovered_pixels"], 300)

    def test_eating_static_bowl_core_survives_local_edge_cleanup(self):
        rgb = np.zeros((128, 128, 3), dtype=np.uint8)
        alpha = np.zeros((128, 128), dtype=np.uint8)
        cv2.circle(rgb, (38, 48), 28, (196, 145, 82), -1)
        cv2.circle(alpha, (38, 48), 28, 255, -1)
        cv2.circle(rgb, (38, 48), 28, (28, 27, 25), 5)

        bowl_alpha = np.zeros_like(alpha)
        cv2.rectangle(bowl_alpha, (75, 78), (120, 108), 255, -1)
        bowl_alpha = cv2.GaussianBlur(
            bowl_alpha,
            (0, 0),
            sigmaX=1.2,
            sigmaY=1.2,
        )
        alpha = np.maximum(alpha, bowl_alpha)
        cv2.rectangle(rgb, (75, 78), (120, 108), (205, 190, 168), -1)
        protected_core = np.zeros_like(alpha, dtype=bool)
        protected_core[83:104, 80:116] = True

        _cleaned_rgb, cleaned_alpha, metrics = clean_transparent_edge_halo(
            rgb,
            alpha,
            action="eating",
            protected_alpha_core=protected_core,
            minimum_candidate_pixels=8,
            minimum_candidate_fraction=0.001,
            minimum_component_pixels=2,
        )

        self.assertTrue(np.array_equal(cleaned_alpha[protected_core], alpha[protected_core]))
        self.assertGreaterEqual(
            float(
                np.count_nonzero(
                    cleaned_alpha[protected_core] >= EATING_STATIC_BOWL_ALPHA_THRESHOLD
                )
            )
            / float(np.count_nonzero(protected_core)),
            EATING_STATIC_BOWL_ALPHA_RETENTION_MIN,
        )
        self.assertGreaterEqual(
            int(metrics.get("protected_alpha_core_pixels", 0)),
            int(np.count_nonzero(protected_core)),
        )

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
