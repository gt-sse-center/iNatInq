#!/bin/bash
# =============================================================================
# MinIO Initialization Script
# =============================================================================
# Creates the pipeline bucket in MinIO for local development.
# This script is typically not needed when using Docker Compose (minio-init
# service handles it), but useful for manual setup or troubleshooting.
#
# Usage:
#   ./init-minio.sh
#
# Prerequisites:
#   - MinIO running at localhost:9000
#   - MinIO client (mc) installed

set -euo pipefail

MINIO_HOST="${MINIO_HOST:-http://localhost:9000}"
MINIO_USER="${MINIO_USER:-minioadmin}"
MINIO_PASSWORD="${MINIO_PASSWORD:-minioadmin}"
BUCKET_NAME="${BUCKET_NAME:-pipeline}"

echo "🔧 Initializing MinIO..."
echo "   Host: ${MINIO_HOST}"
echo "   Bucket: ${BUCKET_NAME}"

# Wait for MinIO to be ready
echo "⏳ Waiting for MinIO to be ready..."
until curl -sf "${MINIO_HOST}/minio/health/live" > /dev/null 2>&1; do
    echo "   MinIO not ready, retrying in 2s..."
    sleep 2
done
echo "✅ MinIO is ready"

# Configure mc alias
echo "🔧 Configuring MinIO client..."
mc alias set local "${MINIO_HOST}" "${MINIO_USER}" "${MINIO_PASSWORD}"

# Create bucket
echo "📦 Creating bucket: ${BUCKET_NAME}"
mc mb "local/${BUCKET_NAME}" --ignore-existing

# Set bucket policy (optional: make public for development)
echo "🔓 Setting bucket policy..."
mc anonymous set public "local/${BUCKET_NAME}"

echo "✅ MinIO initialization complete!"
echo ""
echo "📝 Bucket details:"
mc ls local/

