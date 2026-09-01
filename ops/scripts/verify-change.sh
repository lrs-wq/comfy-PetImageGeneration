#!/usr/bin/env bash
set -euo pipefail

project_root="$(git rev-parse --show-toplevel)"
cd "$project_root"

python_bin="${PYTHON_BIN:-$project_root/.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  printf 'Missing project Python: %s\n' "$python_bin" >&2
  exit 1
fi

export PET_PIPELINE_DATA_ROOT="${PET_PIPELINE_DATA_ROOT:-/data/comfy-pet-pipeline}"

load_env_value_if_unset() {
  local env_file="$1"
  local key="$2"
  local line
  local value
  if [[ -n "${!key:-}" || ! -r "$env_file" ]]; then
    return 0
  fi
  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$line" != "$key="* ]]; then
      continue
    fi
    value="${line#*=}"
    if [[ "$value" == \"*\" || "$value" == \'*\' ]]; then
      value="${value:1:${#value}-2}"
    fi
    printf -v "$key" '%s' "$value"
    export "$key"
    return 0
  done < "$env_file"
}

# The import verifier checks that the production provider key is discoverable.
# Read only that named value; never echo or import the rest of the secret file.
load_env_value_if_unset /etc/comfy-pet-pipeline/common.env DASHSCOPE_API_KEY

"$python_bin" -m pytest -q
"$python_bin" ops/scripts/verify_pipeline.py
"$python_bin" ops/scripts/run-historical-regression-gallery.py
