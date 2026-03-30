"""E2E tests for ingestion pipeline.

Tests the full ingestion flow: generate synthetic images -> upload to MinIO ->
submit Ray job via API -> poll for completion -> verify vectors in Qdrant.
"""

from __future__ import annotations

import httpx
import pytest

from tests.e2e.helpers import api_url


@pytest.mark.e2e
def test_ingestion_happy_path(ingested_collection: str):
    """Test happy-path ingestion from synthetic images to Qdrant vectors.

    Uses session-scoped ingested_collection fixture which handles:
    - Generating 5 synthetic images programmatically
    - Uploading to MinIO
    - Submitting Ray ingestion job
    - Polling until job succeeds
    - Verifying vectors exist in Qdrant

    This test verifies the fixture worked correctly by checking the collection
    exists and has the expected vector count.
    """
    # Verify the fixture completed successfully
    assert ingested_collection is not None
    assert ingested_collection.startswith("e2e_test_")

    # Verify the collection exists in Qdrant by querying it
    # The fixture already verified the count, but we can double-check
    from qdrant_client import QdrantClient

    client = QdrantClient(url="http://localhost:6333")
    collection_info = client.get_collection(collection_name=ingested_collection)

    # Should have exactly 5 vectors (from 5 synthetic images)
    assert collection_info.points_count == 5


@pytest.mark.e2e
def test_ingestion_with_missing_required_field(docker_compose_stack: None):
    """Test that ingestion rejects payloads missing required fields.

    The s3_bucket field is required by RayImageJobRequest. Omitting it
    should result in a 422 validation error.
    """
    payload = {
        "s3_prefix": "test-prefix/",
        "collection": "test_collection",
        # Missing required "s3_bucket" field
    }

    response = httpx.post(
        api_url("/ray/jobs/images"),
        json=payload,
        timeout=10.0,
    )

    assert response.status_code == 422


@pytest.mark.e2e
def test_ingestion_with_empty_payload(docker_compose_stack: None):
    """Test that ingestion rejects an empty payload."""
    response = httpx.post(
        api_url("/ray/jobs/images"),
        json={},
        timeout=10.0,
    )

    # Missing required fields should give 422
    assert response.status_code == 422
