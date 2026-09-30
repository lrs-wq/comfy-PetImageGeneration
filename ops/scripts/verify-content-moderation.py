#!/usr/bin/env python3
"""Record real adapter probes. These few fixtures are not a safety benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image, ImageDraw  # noqa: E402

from src.comfy_pet_pipeline.api.content_moderation import ContentReviewClient  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    client = ContentReviewClient()
    report = {
        "evidence_type": "live_content_review_adapter",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "delivery_mode": "async_output",
        "coverage_boundary": (
            "Only these synthetic text/static-image/animation fixtures; "
            "no illegal-image/video accuracy benchmark, no human approval evidence."
        ),
        "cases": [],
    }
    try:
        report["provider_health"] = client.health()
    except Exception as exc:
        report["provider_health"] = {"status": "unavailable", "error": type(exc).__name__}
    with tempfile.TemporaryDirectory(prefix="comfy-review-probe-") as temp:
        root = Path(temp)
        static = root / "safe-shape.png"
        image = Image.new("RGB", (256, 256), "white")
        ImageDraw.Draw(image).ellipse((70, 70, 186, 186), fill="#3366cc")
        image.save(static)
        animation = root / "safe-animation.gif"
        frames = []
        for offset in (0, 24, 48):
            frame = Image.new("RGB", (256, 256), "white")
            ImageDraw.Draw(frame).rectangle((30 + offset, 80, 80 + offset, 130), fill="#3366cc")
            frames.append(frame)
        frames[0].save(animation, save_all=True, append_images=frames[1:], duration=500, loop=0)
        cases = [
            ("neutral_text", None, "一只可爱的猫正在吃饭。", {"allow"}),
            ("configured_fraud_term", None, "代开发票", {"block"}),
            ("benign_static_transport", static, None, {"allow", "review"}),
            ("benign_animation_video_transport", animation, None, {"allow", "review"}),
        ]
        for name, path, text, actions in cases:
            fixture_hash = hashlib.sha256(path.read_bytes() if path else text.encode()).hexdigest()
            item = {
                "fixture_id": name,
                "fixture_sha256": fixture_hash,
                "expected_actions": sorted(actions),
            }
            try:
                result = client.audit(
                    job_id="verification", stage="input", role=name, path=path, text=text
                )
                item.update(result=result, passed=result["action"] in actions)
            except Exception as exc:
                item.update(passed=False, error=type(exc).__name__)
            report["cases"].append(item)
            print(
                f"fixture={name} passed={item['passed']} "
                f"action={item.get('result', {}).get('action', 'error')}",
                flush=True,
            )
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["passed"] = all(item["passed"] for item in report["cases"])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
