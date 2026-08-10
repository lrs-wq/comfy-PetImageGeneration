#!/usr/bin/env bash
set -euo pipefail

set -a
source /etc/comfy-pet-pipeline/api.env
set +a

response=$(
    curl -fsS -X POST http://127.0.0.1:8770/api/local-video/jobs \
        --header "X-API-Key: ${LOCAL_VIDEO_API_KEY}" \
        --form 'image=@/data/comfy-pet-pipeline/inputs/1.jpg' \
        --form-string 'prompt=Preserve every identifying feature of the uploaded cat. Use smooth, coherent and natural motion. Keep the camera, subject root, scale and background stable. Preserve the same character identity and premium cute 3D rendering throughout the complete five-second animation.' \
        --form-string 'pose_count=1' \
        --form-string 'output_format=transparent_gif' \
        --form-string 'seed=20260715'
)

printf '%s\n' "${response}" > /tmp/lora-style-test-submit.json
/root/comfy-pet-pipeline/.venv/bin/python - <<'PY'
import json
from pathlib import Path

payload = json.loads(Path("/tmp/lora-style-test-submit.json").read_text())
Path("/tmp/lora-style-test-job-id").write_text(payload["job_id"])
fields = (
    "job_id",
    "status",
    "output_format",
    "pose_count",
    "qwen_style_profile",
    "qwen_style_lora",
    "qwen_style_lora_strength",
    "seed",
)
print(json.dumps({key: payload.get(key) for key in fields}, ensure_ascii=False))
PY
