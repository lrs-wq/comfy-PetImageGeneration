#!/usr/bin/env bash
set -euo pipefail

action=${1:?usage: submit-profile-subject-smoke.sh eating|grooming}
case "${action}" in
    eating|grooming) ;;
    *)
        printf 'unsupported action: %s\n' "${action}" >&2
        exit 2
        ;;
esac

set -a
source /root/comfy-pet-pipeline/pet_generation_platform.env
set +a

exec /root/comfy-pet-pipeline/.venv/bin/python \
    /root/comfy-pet-pipeline/deploy/submit-profile-subject-smoke.py \
    "${action}"
