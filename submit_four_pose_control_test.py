from __future__ import annotations

import json
from pathlib import Path

import requests


ENV_PATH = Path("/root/comfy-pet-pipeline/local_video_api.env")
IMAGE_PATH = Path(
    "/data/comfy-pet-pipeline/outputs/local_video_api/20260715T020936Z-f147ad9342/input.png"
)
URL = "http://127.0.0.1:8770/api/local-video/jobs"


def read_env_value(name: str) -> str:
    for raw_line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == name:
            return value.strip().strip('"').strip("'")
    raise RuntimeError(f"{name} is missing from {ENV_PATH}")


def main() -> None:
    api_key = read_env_value("LOCAL_VIDEO_API_KEY")
    prompt = (
        "根据上传的猫咪照片生成身份一致的高品质可爱3D卡通猫，严格保留原猫毛色、花纹位置、"
        "脸型、眼睛颜色、耳朵、口鼻、身体比例和尾巴特征。猫咪不穿衣服，不佩戴眼镜、项圈或任何配饰。"
        "分别生成奔跑、吃饭、喝水和趴卧四个完整全身动作；镜头固定，主体居中，动作自然，四肢结构正确，"
        "脸部稳定，背景用于透明动图抠像。"
    )
    with IMAGE_PATH.open("rb") as image_file:
        response = requests.post(
            URL,
            headers={"X-API-Key": api_key},
            files={"image": (IMAGE_PATH.name, image_file, "image/png")},
            data={
                "prompt": prompt,
                "pose_count": "4",
                "output_format": "transparent_gif",
                "seed": "2026071502",
            },
            timeout=120,
        )
    response.raise_for_status()
    payload = response.json()
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
