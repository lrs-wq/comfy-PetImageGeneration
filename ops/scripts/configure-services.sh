#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PET_PIPELINE_PROJECT_ROOT:-/root/comfy-pet-pipeline}
CONFIG_ROOT=${PET_PIPELINE_CONFIG_ROOT:-/etc/comfy-pet-pipeline}
NGINX_SOURCE="$PROJECT_ROOT/ops/nginx/label-studio-proxy.conf"
NGINX_TARGET=/etc/nginx/conf.d/label-studio-proxy.conf
SYSTEMD_ROOT="$PROJECT_ROOT/ops/systemd"

required_configs=(common.env api.env fun-control.env wan.env platform.env)
for config in "${required_configs[@]}"; do
    if [[ ! -s "$CONFIG_ROOT/$config" ]]; then
        echo "Missing $CONFIG_ROOT/$config; start from ops/env/$config.example" >&2
        exit 1
    fi
done
chmod 0700 "$CONFIG_ROOT"
chmod 0600 "$CONFIG_ROOT"/*.env

PIPELINE_PYTHON="$PROJECT_ROOT/.venv/bin/python"
ffmpeg_path="$($PIPELINE_PYTHON -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"
ln -sfn "$ffmpeg_path" /usr/local/bin/ffmpeg

for unit in \
    comfyui.service \
    local-video-api.service \
    pet-gif-api.service \
    pet-generation-platform.service \
    pet-pipeline.service; do
    install -m 0644 "$SYSTEMD_ROOT/$unit" "/etc/systemd/system/$unit"
done
install -d -m 0755 /etc/systemd/system/local-video-api.service.d
install -m 0644 \
    "$SYSTEMD_ROOT/local-video-api.service.d/fun-control.conf" \
    /etc/systemd/system/local-video-api.service.d/fun-control.conf
install -m 0644 \
    "$SYSTEMD_ROOT/local-video-api.service.d/wan.conf" \
    /etc/systemd/system/local-video-api.service.d/wan.conf
rm -f /etc/systemd/system/local-video-api.service.d/wan14.conf

if [[ ! -s "$CONFIG_ROOT/web-password" ]]; then
    umask 077
    openssl rand -base64 24 | tr -d '\n' > "$CONFIG_ROOT/web-password"
fi
password="$(<"$CONFIG_ROOT/web-password")"
password_hash="$(openssl passwd -apr1 "$password")"
printf 'petadmin:%s\n' "$password_hash" > /etc/nginx/.htpasswd-pet-pipeline
chown root:nginx /etc/nginx/.htpasswd-pet-pipeline
chmod 0640 /etc/nginx/.htpasswd-pet-pipeline

if ! cmp -s "$NGINX_SOURCE" "$NGINX_TARGET"; then
    install -d -m 0700 /data/comfy-pet-pipeline/archive/backups/nginx
    cp -a "$NGINX_TARGET" \
        "/data/comfy-pet-pipeline/archive/backups/nginx/label-studio-proxy.conf.$(date -u +%Y%m%dT%H%M%SZ)"
    install -m 0644 "$NGINX_SOURCE" "$NGINX_TARGET"
fi

nginx -t
systemctl daemon-reload
systemctl enable --now \
    comfyui.service \
    local-video-api.service \
    pet-gif-api.service \
    pet-generation-platform.service \
    pet-pipeline.service
systemctl reload nginx

curl --fail --silent http://127.0.0.1:8188/system_stats >/dev/null
curl --fail --silent http://127.0.0.1:8770/api/local-video/health >/dev/null
curl --fail --silent http://127.0.0.1:8051/health >/dev/null
curl --fail --silent http://127.0.0.1:8766/health >/dev/null
echo "Pet pipeline services and Nginx routes are ready."
