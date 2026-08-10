from __future__ import annotations

import argparse
import json
from pathlib import Path

import local_video_api as pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and summarize the current grooming guide without Qwen."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--runner-python", type=Path, required=True)
    args = parser.parse_args()

    pipeline.OUTPUT_ROOT = args.target.parent / "logs"
    pipeline.SCAIL_RUNNER = args.runner.resolve()
    pipeline.SCAIL_PYTHON = args.runner_python.resolve()
    metadata = pipeline.prepare_grooming_qwen_geometry_guide(
        "codex-grooming-guide-v9-validation",
        args.source.resolve(),
        args.target.resolve(),
        frames=81,
        fps=16.0,
    )
    qwen = metadata["qwen_guide"]
    private = metadata["private_pose_gate"]
    print(
        json.dumps(
            {
                "policy": metadata["policy"],
                "qwen_role": qwen["role"],
                "qwen_sha256": qwen["sha256"],
                "private_sha256": private["guide_sha256"],
                "foreground_rgb": qwen["foreground_rgb"],
                "foreground_palette_rgb": qwen["foreground_palette_rgb"],
                "foreground_pixels": qwen["foreground_pixels"],
                "private_foreground_pixels": qwen["private_foreground_pixels"],
                "foreground_retained_fraction": qwen[
                    "foreground_retained_fraction"
                ],
                "synthetic_foreleg_semantic": qwen[
                    "synthetic_foreleg_semantic"
                ],
                "structural_negative_space": qwen[
                    "structural_negative_space"
                ],
                "appearance_exposed": qwen[
                    "main_case_appearance_exposed_to_qwen"
                ],
                "private_exposed": private["exposed_to_qwen"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
