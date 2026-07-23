#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/root/comfy-pet-pipeline
PIPELINE_PYTHON="$PROJECT_ROOT/.venv/bin/python"
NGINX_SOURCE="$PROJECT_ROOT/deploy/nginx/label-studio-proxy.conf"
NGINX_TARGET=/etc/nginx/conf.d/label-studio-proxy.conf

chmod 600 "$PROJECT_ROOT/.env"

ffmpeg_path="$($PIPELINE_PYTHON -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"
ln -sfn "$ffmpeg_path" /usr/local/bin/ffmpeg

install -m 0644 "$PROJECT_ROOT/deploy/systemd/pet-pipeline.service" /etc/systemd/system/pet-pipeline.service
install -m 0644 "$PROJECT_ROOT/deploy/systemd/comfyui.service" /etc/systemd/system/comfyui.service

if [[ ! -s "$PROJECT_ROOT/.web-password" ]]; then
    umask 077
    openssl rand -base64 24 | tr -d '\n' > "$PROJECT_ROOT/.web-password"
fi
password="$(cat "$PROJECT_ROOT/.web-password")"
password_hash="$(openssl passwd -apr1 "$password")"
printf 'petadmin:%s\n' "$password_hash" > /etc/nginx/.htpasswd-pet-pipeline
chown root:nginx /etc/nginx/.htpasswd-pet-pipeline
chmod 640 /etc/nginx/.htpasswd-pet-pipeline

if ! cmp -s "$NGINX_SOURCE" "$NGINX_TARGET"; then
    cp -a "$NGINX_TARGET" "$NGINX_TARGET.backup-$(date +%Y%m%d-%H%M%S)"
    install -m 0644 "$NGINX_SOURCE" "$NGINX_TARGET"
fi

nginx -t
systemctl daemon-reload
systemctl enable --now pet-pipeline.service
systemctl reload nginx

curl --fail --silent http://127.0.0.1:8766/health
echo
echo "Pet pipeline service and Nginx route are ready."
