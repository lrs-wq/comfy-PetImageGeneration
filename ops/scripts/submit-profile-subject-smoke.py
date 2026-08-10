#!/usr/bin/env python3
"""Submit one profile-backed subject-replacement smoke through port 8050."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("eating", "grooming"))
    parser.add_argument("--profile", default="sleek_tabby_cat_v1")
    parser.add_argument("--base-seed", type=int, default=20260726)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument(
        "--platform-url",
        default="http://127.0.0.1:8050",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    username = os.environ.get("PET_PLATFORM_USER", "").strip()
    password = os.environ.get("PET_PLATFORM_PASSWORD", "").strip()
    if not username or not password:
        raise RuntimeError("PET_PLATFORM_USER/PET_PLATFORM_PASSWORD are required")

    response = requests.post(
        f"{args.platform_url.rstrip('/')}/api/jobs",
        auth=(username, password),
        data={
            "generation_mode": "subject_replacement",
            "mode": "video",
            "actions": json.dumps([args.action]),
            "subject_profile": args.profile,
            "output_format": "transparent_gif",
            "scail_steps": str(args.steps),
            "seed": str(args.base_seed),
        },
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    job_id = str(payload["job_id"])
    Path(f"/tmp/profile-subject-{args.action}-job-id").write_text(
        job_id,
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                key: payload.get(key)
                for key in (
                    "job_id",
                    "status",
                    "stage",
                    "generation_mode",
                    "actions",
                    "subject_profile",
                    "output_format",
                    "seed",
                )
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
