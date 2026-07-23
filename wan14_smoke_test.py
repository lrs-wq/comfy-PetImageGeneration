from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import local_video_api as api


INPUT_NAME = "local_video_api/20260715T060927Z-f3d14327bc/pose_04_wan_input.png"
OUTPUT_DIR = Path("/data/comfy-pet-pipeline/outputs/wan14_smoke")


def main() -> None:
    if api.WAN_ENGINE != "wan2.2-i2v-a14b-fp8":
        raise RuntimeError(f"Expected 14B engine, got {api.WAN_ENGINE}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    prefix = f"wan14_smoke/{stamp}"
    workflow = api.wan_workflow(
        INPUT_NAME,
        (
            "Animate the exact stylized 3D cartoon cat with subtle natural breathing and one slow "
            "tail-tip movement. Preserve face, fur markings, body proportions and the fixed camera. "
            "Uniform pure green background, no clothes or accessories."
        ),
        api.WAN_NEGATIVE + ", " + api.CARTOON_WAN_NEGATIVE + ", " + api.CHROMA_NEGATIVE,
        2026071514,
        512,
        512,
        prefix,
        cfg=api.WAN_CFG,
    )
    workflow_path = OUTPUT_DIR / f"{stamp}_workflow.json"
    workflow_path.write_text(json.dumps(workflow, indent=2, ensure_ascii=False), encoding="utf-8")
    api.free_comfy_models()
    started = time.monotonic()
    entry = api.queue_and_wait(workflow, "Wan2.2 I2V A14B FP8 smoke test", timeout=21600)
    source = api.first_output(entry, {".mp4"}, "Wan2.2 I2V A14B FP8 smoke test")
    target = OUTPUT_DIR / f"{stamp}_wan22_i2v_a14b_fp8.mp4"
    shutil.copy2(source, target)
    print(
        json.dumps(
            {
                "engine": api.WAN_ENGINE,
                "frames": api.FRAME_COUNT,
                "fps": api.FPS,
                "steps": api.WAN_STEPS,
                "split_step": api.WAN_SPLIT_STEP,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "output": str(target),
                "workflow": str(workflow_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
