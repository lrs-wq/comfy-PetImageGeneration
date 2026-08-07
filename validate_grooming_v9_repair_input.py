from __future__ import annotations

import argparse
import json
from pathlib import Path

import local_video_api as pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build and audit v2.21.24 context-masked grooming repair inputs "
            "without running Qwen."
        )
    )
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--anatomy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    repair_target = args.output_dir / "grooming_foreleg_repair_source.png"
    noise_mask_target = (
        args.output_dir / "grooming_foreleg_repair_source_noise_mask.png"
    )
    anatomy_target = (
        args.output_dir / "grooming_foreleg_repair_anatomy_guide.png"
    )
    repair = pipeline.prepare_grooming_foreleg_repair_canvas(
        args.candidate.resolve(),
        args.geometry.resolve(),
        args.canonical.resolve(),
        repair_target.resolve(),
        noise_mask_target=noise_mask_target.resolve(),
    )
    anatomy = pipeline.prepare_grooming_foreleg_repair_anatomy_guide(
        args.anatomy.resolve(),
        anatomy_target.resolve(),
    )
    print(
        json.dumps(
            {
                "repair": repair,
                "anatomy": anatomy,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
