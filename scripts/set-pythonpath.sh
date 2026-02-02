#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

EXPECTED_SUFFIX="/iNatInq/scripts"

if [[ "$SCRIPT_DIR" != */scripts || "$SCRIPT_DIR" != *"$EXPECTED_SUFFIX" ]]; then
  echo "Init script must live under */iNatInq/scripts. Found: $SCRIPT_DIR" >&2
  exit 1
fi

REPO_ROOT="${SCRIPT_DIR%/scripts}"
PYTHONPATH_TARGET="${REPO_ROOT}/src"

if [[ ! -d "$PYTHONPATH_TARGET" ]]; then
  echo "PYTHONPATH target not found: $PYTHONPATH_TARGET" >&2
  exit 1
fi

echo "export PYTHONPATH=${PYTHONPATH_TARGET}:\$PYTHONPATH" >/etc/profile.d/py-path.sh
chmod 644 /etc/profile.d/py-path.sh
