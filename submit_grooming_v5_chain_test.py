from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests


SUPPORTED_ACTIONS = ("eating", "grooming")


def parse_actions(raw: str) -> list[str]:
    actions = [item.strip() for item in raw.split(",") if item.strip()]
    if not actions:
        raise argparse.ArgumentTypeError("at least one action is required")
    unsupported = [item for item in actions if item not in SUPPORTED_ACTIONS]
    if unsupported:
        raise argparse.ArgumentTypeError(
            f"unsupported actions: {unsupported}; expected eating and/or grooming"
        )
    if len(actions) != len(set(actions)):
        raise argparse.ArgumentTypeError("actions must not contain duplicates")
    return actions


def read_api_key(path: Path) -> str:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() == "LOCAL_VIDEO_API_KEY":
            return value.strip().strip('"').strip("'")
    raise RuntimeError(f"LOCAL_VIDEO_API_KEY is missing from {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Clone frozen identity views into one current eating/grooming test job."
        )
    )
    parser.add_argument("--source-job", required=True)
    parser.add_argument(
        "--actions",
        type=parse_actions,
        default=parse_actions("grooming"),
        help="comma-separated actions: eating,grooming (default: grooming)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/comfy-pet-pipeline/outputs/local_video_api"),
    )
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8770/api/local-video/jobs",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path("/root/comfy-pet-pipeline/local_video_api.env"),
    )
    args = parser.parse_args()

    source_dir = args.output_root / args.source_job
    view_paths = {
        "image": source_dir / "input_identity_source.png",
        "image_left": source_dir / "input_left_identity_source.png",
        "image_right": source_dir / "input_right_identity_source.png",
    }
    missing = [str(path) for path in view_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Frozen identity views are missing: {missing}")

    handles = []
    try:
        files = {}
        for field, path in view_paths.items():
            handle = path.open("rb")
            handles.append(handle)
            files[field] = (path.name, handle, "image/png")
        response = requests.post(
            args.api_url,
            headers={"X-API-Key": read_api_key(args.env_file)},
            data={
                "generation_mode": "subject_replacement",
                "actions": json.dumps(args.actions),
                "subject_profile": "sleek_tabby_cat_v1",
                "output_format": "transparent_gif",
                "scail_steps": "6",
                "seed": "20260715",
                "prompt": "",
            },
            files=files,
            timeout=120,
        )
    finally:
        for handle in handles:
            handle.close()
    response.raise_for_status()
    payload = response.json()
    print(
        json.dumps(
            {
                "job_id": payload.get("job_id"),
                "status": payload.get("status"),
                "requested_actions": args.actions,
                "actions": payload.get("actions"),
                "source_job": args.source_job,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
