#!/usr/bin/env bash
# Smoke test wrapper that loads zarf/compose/dev/.env.local and runs the Python test.
# This lets you use the same env overrides as Docker Compose without manual exports.

set -euo pipefail

# Resolve repo root based on this script's location to keep it runnable from anywhere.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)"

# Default env file path used by the zarf dev compose setup.
ENV_FILE="${ENV_FILE:-${REPO_ROOT}/zarf/compose/dev/.env.local}"

# Ensure the env file exists so the user gets an explicit error.
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

# Run the Python smoke test from the repo root to keep imports consistent.
cd "${REPO_ROOT}"
# Pass through any additional CLI args (if provided).
if [[ "$#" -gt 0 ]]; then
  python zarf/scripts/smoke_providers.py "$@"
else
  python zarf/scripts/smoke_providers.py
fi
