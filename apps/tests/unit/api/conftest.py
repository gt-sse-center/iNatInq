"""Shared fixtures for API tests.

This module provides common fixtures used across all API test modules,
including mock clients, test FastAPI clients, and service mocks.
"""

# pylint: disable=redefined-outer-name
# Pytest fixtures intentionally redefine fixture names - this is expected behavior

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from api.app import create_app
from clients.interfaces.embedding import EmbeddingProvider
from clients.interfaces.vector_db import VectorDBProvider
from core.models import SearchResultItem, SearchResults
from fastapi.testclient import TestClient

# =============================================================================
# FastAPI Test Client
# =============================================================================


@pytest.fixture
def test_client() -> TestClient:
    """Create a FastAPI test client for testing endpoints.

    Returns:
        TestClient: A configured test client for making HTTP requests.
    """
    app = create_app()
    return TestClient(app)


# =============================================================================
# Mock Providers
# =============================================================================


@pytest.fixture
def mock_embedding_provider() -> MagicMock:
    """Create a mock EmbeddingProvider for testing.

    Returns:
        MagicMock: A mock embedding provider with embed and embed_async methods.
    """
    provider = MagicMock(spec=EmbeddingProvider)
    provider.embed = MagicMock(return_value=[0.1, 0.2, 0.3] * 256)  # 768-dim vector
    provider.embed_async = AsyncMock(return_value=[0.1, 0.2, 0.3] * 256)
    provider.embed_batch = MagicMock(return_value=[[0.1, 0.2, 0.3] * 256, [0.4, 0.5, 0.6] * 256])
    provider.embed_batch_async = AsyncMock(return_value=[[0.1, 0.2, 0.3] * 256, [0.4, 0.5, 0.6] * 256])
    provider.vector_size = 768
    provider.close = MagicMock()
    return provider


@pytest.fixture
def mock_vector_db_provider() -> MagicMock:
    """Create a mock VectorDBProvider for testing.

    Returns:
        MagicMock: A mock vector database provider with search_async method.
    """
    provider = MagicMock(spec=VectorDBProvider)
    provider.search_async = AsyncMock(
        return_value=SearchResults(
            items=[
                SearchResultItem(
                    point_id="d790dd2c-99eb-4901-b9c9-538b58318fe3",
                    score=0.9234,
                    payload={
                        "text": "s3://pipeline/inputs/hello-a01f74c0.txt",
                        "s3_key": "inputs/hello-a01f74c0.txt",
                    },
                ),
                SearchResultItem(
                    point_id="e891ee3d-00fc-5012-c0d0-649c69429gf4",
                    score=0.8567,
                    payload={
                        "text": "s3://pipeline/inputs/world-b02g85d1.txt",
                        "s3_key": "inputs/world-b02g85d1.txt",
                    },
                ),
            ],
            total=2,
        )
    )
    provider.batch_upsert_async = AsyncMock()
    provider.ensure_collection_async = AsyncMock()
    provider.close = MagicMock()
    return provider


# =============================================================================
# Mock Services
# =============================================================================


@pytest.fixture
def mock_search_service() -> MagicMock:
    """Create a mock SearchService for testing.

    Returns:
        MagicMock: A mock search service with search_documents_async method.
    """
    service = MagicMock()
    service.search_documents = MagicMock(
        return_value=SearchResults(
            items=[
                SearchResultItem(
                    point_id="1",
                    score=0.95,
                    payload={"text": "test document", "source": "test.txt"},
                )
            ],
            total=1,
        )
    )
    service.search_documents_async = AsyncMock(
        return_value=SearchResults(
            items=[
                SearchResultItem(
                    point_id="1",
                    score=0.95,
                    payload={"text": "test document", "source": "test.txt"},
                )
            ],
            total=1,
        )
    )
    return service


@pytest.fixture
def mock_spark_service() -> MagicMock:
    """Create a mock SparkService for testing.

    Returns:
        MagicMock: A mock Spark service with job management methods.
    """
    service = MagicMock()
    service.submit_processing_job = MagicMock(
        return_value={
            "job_name": "s3-to-vector-db-20260112-153045-a1b2c3d4",
            "status": "submitted",
            "namespace": "ml-system",
            "s3_prefix": "inputs/",
            "collection": "documents",
            "submitted_at": "2026-01-12T15:30:45.123456Z",
        }
    )
    service.get_job_status = MagicMock(
        return_value={
            "job_name": "s3-to-vector-db-20260112-153045-a1b2c3d4",
            "state": "RUNNING",
            "spark_state": "RUNNING",
            "driver_info": {"podName": "s3-to-vector-db-driver"},
            "execution_attempts": 1,
            "last_submission_attempt_time": "2026-01-12T15:30:45Z",
            "termination_time": None,
        }
    )
    service.list_jobs = MagicMock(
        return_value=[
            {
                "job_name": "s3-to-vector-db-20260112-153045-a1b2c3d4",
                "state": "COMPLETED",
                "created_at": "2026-01-12T15:30:45Z",
            }
        ]
    )
    service.delete_job = MagicMock(
        return_value={
            "job_name": "s3-to-vector-db-20260112-153045-a1b2c3d4",
            "status": "deleted",
        }
    )
    return service


@pytest.fixture
def mock_ray_service() -> MagicMock:
    """Create a mock RayService for testing.

    Returns:
        MagicMock: A mock Ray service with job management methods.
    """
    service = MagicMock()
    service.submit_s3_to_qdrant = MagicMock(return_value="raysubmit_1234567890")
    service.get_job_status = MagicMock(return_value={"status": "RUNNING", "message": None})
    service.get_job_logs = MagicMock(return_value="Processing 1000 documents...\nCompleted successfully.")
    service.stop_job = MagicMock()
    return service


# =============================================================================
# Settings and Config Mocks
# =============================================================================


@pytest.fixture
def mock_settings() -> MagicMock:
    """Create a mock Settings object for testing.

    Returns:
        MagicMock: A mock settings object with all required attributes.
    """
    from config import EmbeddingConfig, MinIOConfig, VectorDBConfig

    settings = MagicMock()
    settings.embedding = EmbeddingConfig(
        provider_type="ollama",
        ollama_url="http://localhost:11434",
        ollama_model="nomic-embed-text",
    )
    settings.vector_db = VectorDBConfig(
        provider_type="qdrant",
        collection="documents",
        qdrant_url="http://localhost:6333",
    )
    settings.minio = MinIOConfig(
        endpoint_url="http://localhost:9000",
        access_key_id="minioadmin",
        secret_access_key="minioadmin",
        bucket="pipeline",
    )
    settings.k8s_namespace = "ml-system"
    return settings


# =============================================================================
# Patch Helpers
# =============================================================================


@pytest.fixture
def patch_create_embedding_provider(mock_embedding_provider: MagicMock):
    """Patch create_embedding_provider to return mock provider.

    Args:
        mock_embedding_provider: Mock embedding provider fixture.

    Yields:
        Mock patch object.
    """
    with patch(
        "api.routes.create_embedding_provider",
        return_value=mock_embedding_provider,
    ) as mock:
        yield mock


@pytest.fixture
def patch_create_vector_db_provider(mock_vector_db_provider: MagicMock):
    """Patch create_vector_db_provider to return mock provider.

    Args:
        mock_vector_db_provider: Mock vector DB provider fixture.

    Yields:
        Mock patch object.
    """
    with patch(
        "api.routes.create_vector_db_provider",
        return_value=mock_vector_db_provider,
    ) as mock:
        yield mock


@pytest.fixture
def patch_get_settings(mock_settings: MagicMock):
    """Patch get_settings to return mock settings.

    Args:
        mock_settings: Mock settings fixture.

    Yields:
        Mock patch object.
    """
    with patch("api.routes.get_settings", return_value=mock_settings) as mock:
        yield mock
