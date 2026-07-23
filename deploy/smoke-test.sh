#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/root/comfy-pet-pipeline
password="$(cat "$PROJECT_ROOT/.web-password")"
host_header='Host: 39.106.120.253'

no_auth_status="$(curl -sS -o /dev/null -w '%{http_code}' -H "$host_header" http://127.0.0.1/pet-dressup/)"
auth_status="$(curl -sS -o /dev/null -w '%{http_code}' -u "petadmin:$password" -H "$host_header" http://127.0.0.1/pet-dressup/)"
health_body="$(curl -fsS -u "petadmin:$password" -H "$host_header" http://127.0.0.1/pet-dressup/health)"
label_status="$(curl -sS -o /dev/null -w '%{http_code}' -H "$host_header" http://127.0.0.1/)"

printf 'pet_no_auth=%s\n' "$no_auth_status"
printf 'pet_auth=%s\n' "$auth_status"
printf 'pet_health=%s\n' "$health_body"
printf 'label_studio=%s\n' "$label_status"

[[ "$no_auth_status" == 401 ]]
[[ "$auth_status" == 200 ]]
[[ "$health_body" == '{"status":"ok"}' ]]
[[ "$label_status" == 302 ]]
