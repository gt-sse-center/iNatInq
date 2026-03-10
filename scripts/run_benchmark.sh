#!/usr/bin/env bash
set -euo pipefail

# Required environment variables (set these before running, or use a .env file):
#   QDRANT_URL        — Qdrant cluster URL
#   QDRANT_API_KEY    — Qdrant API key
#   CLIP_URL          — Hosted CLIP endpoint URL
#   CLIP_API_KEY      — CLIP endpoint API key

: "${QDRANT_URL:?Set QDRANT_URL}"
: "${QDRANT_API_KEY:?Set QDRANT_API_KEY}"
: "${CLIP_URL:?Set CLIP_URL}"
: "${CLIP_API_KEY:?Set CLIP_API_KEY}"

export QDRANT_URL
export QDRANT_API_KEY
export EMBEDDING_PROVIDER="${EMBEDDING_PROVIDER:-hosted_clip}"
export CLIP_URL
export CLIP_API_KEY
export CLIP_MODEL="${CLIP_MODEL:-ViT-B/32}"

uv run python -m core.benchmark.cli run \
  --dataset "${1:-benchmarks/inquire/inquire-val-subset.json}" \
  --provider "${2:-qdrant}" \
  --collection "${3:-demo-data-1M_images}" \
  --limit "${4:-50}" \
  --warmup 2
