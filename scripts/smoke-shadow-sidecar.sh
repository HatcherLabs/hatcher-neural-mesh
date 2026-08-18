#!/usr/bin/env bash
set -euo pipefail

binary="${1:-target/debug/hatcher-ux}"
port="${HATCHER_MESH_SMOKE_PORT:-33030}"
token="smoke-test-token-0123456789abcdef"
log_file="$(mktemp)"
response_file="$(mktemp)"

cleanup() {
  if [[ -n "${sidecar_pid:-}" ]]; then
    kill "$sidecar_pid" 2>/dev/null || true
    wait "$sidecar_pid" 2>/dev/null || true
  fi
  rm -f "$log_file" "$response_file"
}
trap cleanup EXIT

HATCHER_MESH_PORT="$port" \
HATCHER_MESH_INTERNAL_TOKEN="$token" \
HATCHER_MESH_REQUIRE_INTERNAL_TOKEN=true \
HATCHER_MESH_SHADOW_ONLY=true \
  "$binary" serve >"$log_file" 2>&1 &
sidecar_pid=$!

for _ in $(seq 1 30); do
  if curl --fail --silent "http://127.0.0.1:${port}/health" >/dev/null; then
    break
  fi
  if ! kill -0 "$sidecar_pid" 2>/dev/null; then
    cat "$log_file" >&2
    exit 1
  fi
  sleep 0.2
done

unauthorized_status="$(curl --silent --output /dev/null --write-out '%{http_code}' \
  --header 'content-type: application/json' \
  --data '{"agents":[],"task":{"task_id":"smoke","description":"smoke","domain":"general","features":[],"urgency":0.5,"constraints":{"max_latency_ms":null,"max_cost":null},"metadata":{}}}' \
  "http://127.0.0.1:${port}/api/shadow/route")"
test "$unauthorized_status" = "401"

malformed_unauthorized_status="$(curl --silent --output /dev/null --write-out '%{http_code}' \
  --header 'content-type: application/json' \
  --data '{}' \
  "http://127.0.0.1:${port}/api/shadow/route")"
test "$malformed_unauthorized_status" = "401"

curl --fail --silent \
  --header 'content-type: application/json' \
  --header "x-hatcher-mesh-token: ${token}" \
  --data '{"agents":[{"id":"agent-1","label":"Agent 1","role":"Coder","capability":{"intelligence":0.7,"specialization":0.7,"performance":0.8,"context":0.6,"memory":0.5},"resources":{"energy":0.1,"latency":0.1,"observed_latency_ms":100,"observed_cost":1,"observations":3},"confidence":0.7,"expertise":{"general":0.7},"tags":["openclaw","active"]}],"task":{"task_id":"smoke","description":"smoke","domain":"general","features":[0.1],"urgency":0.5,"constraints":{"max_latency_ms":1000,"max_cost":10},"metadata":{"contract_version":"shadow.v1"}}}' \
  "http://127.0.0.1:${port}/api/shadow/route" > "$response_file"
grep -Fq '"contract_version":"shadow.v1"' "$response_file"

curl --fail --silent --show-error \
  --header 'content-type: application/json' \
  --header "x-hatcher-mesh-token: ${token}" \
  --data '{"agents":[{"id":"agent-1","label":"Agent 1","role":"Coder","capability":{"intelligence":0.7,"specialization":0.7,"performance":0.8,"context":0.6,"memory":0.5},"resources":{"energy":0.1,"latency":0.1,"observed_latency_ms":100,"observed_cost":1,"observations":3},"confidence":0.7,"expertise":{"general":0.7},"tags":["openclaw","active"]}],"task":{"task_id":"smoke-live","description":"smoke","domain":"general","features":[0.1],"urgency":0.5,"constraints":{"max_latency_ms":1000,"max_cost":10},"metadata":{"contract_version":"route.v2"}}}' \
  "http://127.0.0.1:${port}/api/route" > "$response_file"
grep -Fq '"contract_version":"route.v2"' "$response_file"
grep -Fq '"recommended_agent_id":"agent-1"' "$response_file"

restricted_status="$(curl --silent --output /dev/null --write-out '%{http_code}' \
  "http://127.0.0.1:${port}/api/mesh/overview")"
test "$restricted_status" = "404"

echo "shadow sidecar smoke passed"
