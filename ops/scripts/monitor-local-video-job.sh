#!/usr/bin/env bash
set -euo pipefail

job_id=${1:?job id is required}
status_file="/data/comfy-pet-pipeline/outputs/local_video_api/${job_id}/status.json"

while true; do
    date '+%H:%M:%S'
    grep -E '"status"|"stage"|"progress"|"filename"|"error"' "${status_file}"
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits
    if grep -qE '"status": "(completed|failed)"' "${status_file}"; then
        exit
    fi
    sleep 30
done
