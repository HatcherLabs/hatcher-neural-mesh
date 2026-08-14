#!/usr/bin/env bash
set -euo pipefail

readonly IMAGE_REPOSITORY="hatcher/neural-mesh"
readonly IMAGE_TAG="$(git rev-parse --short=12 HEAD)"
readonly CANDIDATE_IMAGE="${IMAGE_REPOSITORY}:${IMAGE_TAG}"
readonly LIVE_IMAGE="${IMAGE_REPOSITORY}:production"
readonly ROLLBACK_IMAGE="${IMAGE_REPOSITORY}:rollback"
readonly ENV_FILE="${HATCHER_MESH_ENV_FILE:-/etc/hatcher/services/hatcher-neural-mesh.env}"
readonly UNIT_TEMPLATE="ops/systemd/hatcher-neural-mesh.service"
readonly UNIT_FILE="/etc/systemd/system/hatcher-neural-mesh.service"

if [[ ! -f "$UNIT_TEMPLATE" || ! -f Dockerfile ]]; then
  echo "Run this script from the hatcher-neural-mesh repository root." >&2
  exit 2
fi

if [[ -L "$ENV_FILE" || ! -f "$ENV_FILE" ]]; then
  echo "Missing $ENV_FILE. Create it root-owned with mode 0600 before deployment." >&2
  exit 2
fi

if [[ "$(stat -c %u -- "$ENV_FILE")" != "0" || "$(stat -c %a -- "$ENV_FILE")" != "600" ]]; then
  echo "$ENV_FILE must be root-owned with mode 0600." >&2
  exit 2
fi

read_env_value() {
  local key="$1"
  sed -n "s/^${key}=//p" "$ENV_FILE" | tail -n 1
}

token="$(read_env_value HATCHER_MESH_INTERNAL_TOKEN)"
if (( ${#token} < 32 )) || [[ "$token" =~ [[:space:]] ]]; then
  echo "HATCHER_MESH_INTERNAL_TOKEN must contain at least 32 non-whitespace characters." >&2
  exit 2
fi
if [[ "$(read_env_value HATCHER_MESH_REQUIRE_INTERNAL_TOKEN)" != "true" ]]; then
  echo "HATCHER_MESH_REQUIRE_INTERNAL_TOKEN=true is required in production." >&2
  exit 2
fi
if [[ "$(read_env_value HATCHER_MESH_SHADOW_ONLY)" != "true" ]]; then
  echo "HATCHER_MESH_SHADOW_ONLY=true is required in production." >&2
  exit 2
fi
if [[ "$(read_env_value HATCHER_MESH_BIND_ADDR)" != "0.0.0.0" ]]; then
  echo "The production container requires HATCHER_MESH_BIND_ADDR=0.0.0.0." >&2
  exit 2
fi
if [[ "$(read_env_value HATCHER_MESH_ALLOW_NON_LOOPBACK_BIND)" != "true" ]]; then
  echo "The production container requires HATCHER_MESH_ALLOW_NON_LOOPBACK_BIND=true." >&2
  exit 2
fi

echo "==> Building pinned Neural Mesh image ${CANDIDATE_IMAGE}"
docker build --pull --tag "$CANDIDATE_IMAGE" .

had_live_image=false
if docker image inspect "$LIVE_IMAGE" >/dev/null 2>&1; then
  docker tag "$LIVE_IMAGE" "$ROLLBACK_IMAGE"
  had_live_image=true
fi
docker tag "$CANDIDATE_IMAGE" "$LIVE_IMAGE"

unit_tmp="$(mktemp)"
trap 'rm -f "$unit_tmp"' EXIT
sed \
  -e "s|@@ENV_FILE@@|${ENV_FILE}|g" \
  -e "s|@@IMAGE@@|${LIVE_IMAGE}|g" \
  "$UNIT_TEMPLATE" > "$unit_tmp"
sudo install -o root -g root -m 0644 "$unit_tmp" "$UNIT_FILE"
sudo systemctl daemon-reload
sudo systemctl enable hatcher-neural-mesh.service >/dev/null
sudo systemctl restart hatcher-neural-mesh.service

healthy=false
for _ in $(seq 1 30); do
  if curl --fail --silent http://127.0.0.1:3030/health >/dev/null; then
    healthy=true
    break
  fi
  sleep 1
done

if [[ "$healthy" == "true" ]]; then
  echo "==> Neural Mesh is healthy on 127.0.0.1:3030"
  docker image prune --force --filter 'until=168h' >/dev/null || true
  exit 0
fi

echo "Neural Mesh failed its health gate." >&2
sudo journalctl -u hatcher-neural-mesh.service --no-pager -n 80 >&2 || true
if [[ "$had_live_image" == "true" ]]; then
  echo "==> Restoring previous image"
  docker tag "$ROLLBACK_IMAGE" "$LIVE_IMAGE"
  sudo systemctl restart hatcher-neural-mesh.service
fi
exit 1
