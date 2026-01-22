#!/usr/bin/env bash
# Provider health checks using endpoints from the selected env file.

set -euo pipefail

# Resolve repo root to allow running from any working directory.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)"

# Default env file path (can be overridden with ENV_FILE).
ENV_FILE="${ENV_FILE:-${REPO_ROOT}/zarf/compose/dev/.env.local}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing env file: ${ENV_FILE}" >&2
  echo "Create it from zarf/compose/dev/env.local.example or set ENV_FILE." >&2
  exit 1
fi

# Export all variables defined in the env file into this process.
set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

fail=0

check_http() {
  local name="$1"
  local base_url="$2"
  local path="$3"
  shift 3

  if [[ -z "${base_url}" ]]; then
    echo "  ${name}: skipped (unset)"
    return
  fi

  # Strip any trailing slash before appending the path.
  local url="${base_url%/}${path}"
  local code
  echo "  ${name}: ${url}"
  if [[ "$#" -gt 0 ]]; then
    code="$(curl -sS -o /dev/null -w "%{http_code}" --max-time 10 "$@" "${url}" || true)"
  else
    code="$(curl -sS -o /dev/null -w "%{http_code}" --max-time 10 "${url}" || true)"
  fi

  if [[ "${code}" =~ ^2 ]]; then
    echo "  ${name}: ok (${code})"
  else
    echo "  ${name}: fail (${code})"
    fail=1
  fi
}

echo "Provider health (from ${ENV_FILE}):"

check_http "Ollama" "${OLLAMA_BASE_URL-}" "/api/tags"

if [[ -n "${QDRANT_API_KEY-}" ]]; then
  check_http "Qdrant" "${QDRANT_URL-}" "/collections" -H "api-key: ${QDRANT_API_KEY}"
else
  check_http "Qdrant" "${QDRANT_URL-}" "/collections"
fi

if [[ -n "${WEAVIATE_API_KEY-}" ]]; then
  check_http "Weaviate" "${WEAVIATE_URL-}" "/v1/meta" -H "Authorization: Bearer ${WEAVIATE_API_KEY}"
else
  check_http "Weaviate" "${WEAVIATE_URL-}" "/v1/meta"
fi

exit "${fail}"
