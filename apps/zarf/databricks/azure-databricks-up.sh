#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root to allow running from any working directory.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)"

ENV_FILE="${ENV_FILE:-${REPO_ROOT}/zarf/databricks/dev/.env.local}"
CLUSTER_SPEC_FILE="${CLUSTER_SPEC_FILE:-${REPO_ROOT}/zarf/databricks/inatinq-azure-databricks-cluster.yml}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing env file: ${ENV_FILE}" >&2
  echo "Create it from zarf/compose/dev/env.local.example or set ENV_FILE." >&2
  exit 1
fi

if [[ ! -f "${CLUSTER_SPEC_FILE}" ]]; then
  echo "Missing cluster spec file: ${CLUSTER_SPEC_FILE}" >&2
  exit 1
fi

# Export all variables defined in the env file into this process.
set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

if [[ -z "${DATABRICKS_HOST-}" || -z "${DATABRICKS_TOKEN-}" ]]; then
  echo "Missing required Databricks env vars (DATABRICKS_HOST, DATABRICKS_TOKEN)." >&2
  exit 1
fi

if ! command -v databricks >/dev/null 2>&1; then
  echo "Missing databricks CLI. Install it to use this target." >&2
  exit 1
fi

cluster_id="${DATABRICKS_CLUSTER_ID-}"
if [[ -z "${cluster_id}" ]]; then
  cluster_id="$(python3 - "$CLUSTER_SPEC_FILE" <<'PY'
import json
import sys

path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
except Exception:
    sys.exit(0)

cluster_id = data.get("cluster_id") or data.get("spec", {}).get("cluster_id")
if cluster_id:
    print(cluster_id)
PY
)"
fi

if [[ -z "${cluster_id}" ]]; then
  echo "Missing cluster id. Set DATABRICKS_CLUSTER_ID or include it in ${CLUSTER_SPEC_FILE}." >&2
  exit 1
fi

echo "Starting cluster ${cluster_id}..."
databricks clusters start --cluster-id "${cluster_id}"
