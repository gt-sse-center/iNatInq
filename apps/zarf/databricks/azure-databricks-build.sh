#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root to allow running from any working directory.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)"

ENV_FILE="${ENV_FILE:-${REPO_ROOT}/zarf/databricks/dev/.env.local}"
CLUSTER_SPEC_FILE="${CLUSTER_SPEC_FILE:-${REPO_ROOT}/zarf/databricks/dev/inatinq-azure-databricks-cluster.yml}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing env file: ${ENV_FILE}" >&2
  echo "Create it from zarf/databricks/dev/env.local.example or set ENV_FILE." >&2
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

get_cluster_id() {
  if [[ -n "${DATABRICKS_CLUSTER_ID-}" ]]; then
    echo "${DATABRICKS_CLUSTER_ID}"
    return
  fi
  python3 - "$CLUSTER_SPEC_FILE" <<'PY'
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
}

tmp_spec="$(mktemp)"
cleanup() {
  rm -f "${tmp_spec}" "${tmp_edit-}"
}
trap cleanup EXIT

python3 - "$CLUSTER_SPEC_FILE" "$tmp_spec" <<'PY'
import json
import sys

src, dst = sys.argv[1], sys.argv[2]
with open(src, "r", encoding="utf-8") as handle:
    data = json.load(handle)

spec = data.get("spec", data)
with open(dst, "w", encoding="utf-8") as handle:
    json.dump(spec, handle)
PY

cluster_id="$(get_cluster_id)"

if [[ -n "${cluster_id}" ]]; then
  tmp_edit="$(mktemp)"
  python3 - "$tmp_spec" "$cluster_id" "$tmp_edit" <<'PY'
import json
import sys

spec_path, cluster_id, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
with open(spec_path, "r", encoding="utf-8") as handle:
    spec = json.load(handle)

spec["cluster_id"] = cluster_id
with open(out_path, "w", encoding="utf-8") as handle:
    json.dump(spec, handle)
PY

  echo "Updating cluster ${cluster_id} from ${CLUSTER_SPEC_FILE}..."
  databricks clusters edit --json @"${tmp_edit}"
else
  echo "Creating cluster from ${CLUSTER_SPEC_FILE}..."
  databricks clusters create --json @"${tmp_spec}"
fi
