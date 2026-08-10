#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PET_PIPELINE_PROJECT_ROOT:-/root/comfy-pet-pipeline}
DATA_ROOT=${PET_PIPELINE_DATA_ROOT:-/data/comfy-pet-pipeline}
COMFY_RUNTIME_ROOT=${COMFY_RUNTIME_ROOT:-/opt/comfyui}
COMFY_ROOT="$COMFY_RUNTIME_ROOT/current"
THIRD_PARTY_NODES="$COMFY_RUNTIME_ROOT/custom-nodes"
COMPAT_LINK="$PROJECT_ROOT/ComfyUI"
CORE_ARCHIVE=${COMFY_CORE_ARCHIVE:-/root/comfyui-core.tar.gz}

case "$(realpath -m "$COMFY_ROOT")" in
    /opt/comfyui/current) ;;
    *)
        echo "Refusing unsafe ComfyUI target: $COMFY_ROOT" >&2
        exit 1
        ;;
esac

if [[ ! -d "$COMFY_ROOT" ]]; then
    if [[ ! -f "$CORE_ARCHIVE" ]]; then
        echo "Missing both $COMFY_ROOT and $CORE_ARCHIVE" >&2
        exit 1
    fi
    install -d -m 0755 "$COMFY_ROOT"
    tar -xzf "$CORE_ARCHIVE" -C "$COMFY_ROOT"
fi

if [[ -e "$COMPAT_LINK" && ! -L "$COMPAT_LINK" ]]; then
    echo "$COMPAT_LINK is a real directory; migrate it before creating the compatibility link" >&2
    exit 1
fi
ln -sfn "$COMFY_ROOT" "$COMPAT_LINK"

install -d -m 0755 \
    "$COMFY_ROOT/custom_nodes" \
    "$THIRD_PARTY_NODES" \
    "$DATA_ROOT/models/comfyui" \
    "$DATA_ROOT/comfyui/input" \
    "$DATA_ROOT/comfyui/output" \
    "$DATA_ROOT/comfyui/temp"

ln -sfn "$THIRD_PARTY_NODES/ComfyUI-3D-Pack" \
    "$COMFY_ROOT/custom_nodes/ComfyUI-3D-Pack"
ln -sfn "$PROJECT_ROOT/extensions/comfyui/ComfyUI-DashScope-PetVideo" \
    "$COMFY_ROOT/custom_nodes/ComfyUI-DashScope-PetVideo"
ln -sfn "$PROJECT_ROOT/extensions/comfyui/ComfyUI-DashScope-PetVideo-I2V" \
    "$COMFY_ROOT/custom_nodes/ComfyUI-DashScope-PetVideo-I2V"
ln -sfn "$PROJECT_ROOT/extensions/comfyui/ComfyUI-DashScope-PetVideo-TwoActions" \
    "$COMFY_ROOT/custom_nodes/ComfyUI-DashScope-PetVideo-TwoActions"

for name in models input output temp; do
    target="$COMFY_ROOT/$name"
    if [[ -e "$target" && ! -L "$target" ]]; then
        echo "Refusing to replace non-symlink runtime path: $target" >&2
        exit 1
    fi
done
ln -sfn "$DATA_ROOT/models/comfyui" "$COMFY_ROOT/models"
ln -sfn "$DATA_ROOT/comfyui/input" "$COMFY_ROOT/input"
ln -sfn "$DATA_ROOT/comfyui/output" "$COMFY_ROOT/output"
ln -sfn "$DATA_ROOT/comfyui/temp" "$COMFY_ROOT/temp"

echo "ComfyUI runtime: $COMFY_ROOT"
echo "Pipeline data: $DATA_ROOT"
ls -ld "$COMPAT_LINK" "$COMFY_ROOT"/{models,input,output,temp} "$COMFY_ROOT"/custom_nodes/ComfyUI*
