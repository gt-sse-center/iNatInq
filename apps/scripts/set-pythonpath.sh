#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

EXPECTED_DIR="/Workspace/Users/kbhardwaj6@gatech.edu/iNatInq/apps/scripts"

if [[ "$SCRIPT_DIR" != "$EXPECTED_DIR" ]]; then
  echo "Init script must live under $EXPECTED_DIR. Found: $SCRIPT_DIR" >&2
  exit 1
fi

PYTHONPATH_TARGET="/Workspace/Users/kbhardwaj6@gatech.edu/iNatInq/apps/src"

if [[ ! -d "$PYTHONPATH_TARGET" ]]; then
  echo "PYTHONPATH target not found: $PYTHONPATH_TARGET" >&2
  exit 1
fi

echo "export PYTHONPATH=${PYTHONPATH_TARGET}:\$PYTHONPATH" >/etc/profile.d/py-path.sh
chmod 644 /etc/profile.d/py-path.sh
