#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="zarf/compose/dev/.env.local"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing $ENV_FILE; aborting." >&2
  exit 1
fi

# export all vars from .env.local
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
env
set +a

uv run uvicorn main:app --app-dir src --reload --host localhost --port 8000
