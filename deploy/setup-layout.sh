#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/root/comfy-pet-pipeline
DATA_ROOT=/data/comfy-pet-pipeline
COMFY_ROOT="$PROJECT_ROOT/ComfyUI"
CORE_ARCHIVE=/root/comfyui-core.tar.gz

if [[ "$(realpath -m "$COMFY_ROOT")" != /root/comfy-pet-pipeline/ComfyUI ]]; then
    echo "Refusing unsafe ComfyUI target: $COMFY_ROOT" >&2
    exit 1
fi
if [[ ! -f "$CORE_ARCHIVE" ]]; then
    echo "Missing $CORE_ARCHIVE" >&2
    exit 1
fi

rm -rf -- "$COMFY_ROOT"
mkdir -p "$COMFY_ROOT"
tar -xzf "$CORE_ARCHIVE" -C "$COMFY_ROOT"
rm -f -- "$CORE_ARCHIVE"

mkdir -p \
    "$COMFY_ROOT/custom_nodes" \
    "$DATA_ROOT/models/comfyui" \
    "$DATA_ROOT/comfyui/input" \
    "$DATA_ROOT/comfyui/output" \
    "$DATA_ROOT/comfyui/temp"

ln -sfn ../../comfyui_custom_nodes/ComfyUI-3D-Pack "$COMFY_ROOT/custom_nodes/ComfyUI-3D-Pack"
ln -sfn ../../ComfyUI-DashScope-PetVideo "$COMFY_ROOT/custom_nodes/ComfyUI-DashScope-PetVideo"
ln -sfn ../../ComfyUI-DashScope-PetVideo-I2V "$COMFY_ROOT/custom_nodes/ComfyUI-DashScope-PetVideo-I2V"
ln -sfn ../../ComfyUI-DashScope-PetVideo-TwoActions "$COMFY_ROOT/custom_nodes/ComfyUI-DashScope-PetVideo-TwoActions"

rm -rf -- "$COMFY_ROOT/models" "$COMFY_ROOT/input" "$COMFY_ROOT/output" "$COMFY_ROOT/temp"
ln -s "$DATA_ROOT/models/comfyui" "$COMFY_ROOT/models"
ln -s "$DATA_ROOT/comfyui/input" "$COMFY_ROOT/input"
ln -s "$DATA_ROOT/comfyui/output" "$COMFY_ROOT/output"
ln -s "$DATA_ROOT/comfyui/temp" "$COMFY_ROOT/temp"

echo "Code: $(du -sh "$PROJECT_ROOT" | cut -f1)"
echo "Data: $(du -sh "$DATA_ROOT" | cut -f1)"
ls -ld "$COMFY_ROOT"/{models,input,output,temp} "$COMFY_ROOT"/custom_nodes/ComfyUI*
