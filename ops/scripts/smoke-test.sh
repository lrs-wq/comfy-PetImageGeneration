#!/usr/bin/env bash
set -euo pipefail

password="$(</etc/comfy-pet-pipeline/web-password)"
public_host=ai.hzzootech.com
base_url="https://$public_host"
resolve=(--resolve "$public_host:443:127.0.0.1")

no_auth_status="$(curl -sS -o /dev/null -w '%{http_code}' "${resolve[@]}" "$base_url/pet-dressup/")"
auth_status="$(curl -sS -o /dev/null -w '%{http_code}' -u "petadmin:$password" "${resolve[@]}" "$base_url/pet-dressup/")"
health_body="$(curl -fsS -u "petadmin:$password" "${resolve[@]}" "$base_url/pet-dressup/health")"
label_status="$(curl -sS -o /dev/null -w '%{http_code}' "${resolve[@]}" "$base_url/")"

printf 'pet_no_auth=%s\n' "$no_auth_status"
printf 'pet_auth=%s\n' "$auth_status"
printf 'pet_health=%s\n' "$health_body"
printf 'label_studio=%s\n' "$label_status"

[[ "$no_auth_status" == 401 ]]
[[ "$auth_status" == 200 ]]
[[ "$health_body" == '{"status":"ok"}' ]]
[[ "$label_status" == 302 ]]
