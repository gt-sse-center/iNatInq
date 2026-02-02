"""Shared fixtures for API tests.

This module provides common fixtures used across all API test modules,
including mock clients, test FastAPI clients, and service mocks.
"""

# pylint: disable=redefined-outer-name
# Pytest fixtures intentionally redefine fixture names - this is expected behavior

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from api.app import create_app
from clients.clip import CLIPClient
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
def mock_ray_service() -> MagicMock:
    """Create a mock RayService for testing.

    Returns:
        MagicMock: A mock Ray service with job management methods.
    """
    service = MagicMock()
    service.submit_s3_to_vector_dbs = MagicMock(return_value="raysubmit_1234567890")
    service.submit_image_job = MagicMock(return_value="raysubmit_1234567890")
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


# =============================================================================
# CLIP Client Mocks (for Image Search)
# =============================================================================


@pytest.fixture
def mock_clip_client() -> MagicMock:
    """Create a mock CLIPClient for testing image search.

    Returns:
        MagicMock: A mock CLIP client with text embedding methods.
    """
    client = MagicMock(spec=CLIPClient)
    client.embed_text = MagicMock(return_value=[0.1, 0.2, 0.3] * 170)  # 510-dim vector
    client.embed_text_async = AsyncMock(return_value=[0.1, 0.2, 0.3] * 170)
    client.vector_size = 512
    client.model = "clip-vit-base-patch32"
    return client


@pytest.fixture
def mock_image_vector_db_provider() -> MagicMock:
    """Create a mock VectorDBProvider for image search testing.

    Returns results with image metadata (s3_key, s3_uri, format, etc.).

    Returns:
        MagicMock: A mock vector database provider with image search results.
    """
    provider = MagicMock(spec=VectorDBProvider)
    provider.search_async = AsyncMock(
        return_value=SearchResults(
            items=[
                SearchResultItem(
                    point_id="img-d790dd2c-99eb-4901-b9c9-538b58318fe3",
                    score=0.8234,
                    payload={
                        "s3_key": "images/sunset-001.jpg",
                        "s3_uri": "s3://pipeline/images/sunset-001.jpg",
                        "format": "jpeg",
                        "width": 1920,
                        "height": 1080,
                        "thumbnail_key": "thumbnails/sunset-001.jpg",
                    },
                ),
                SearchResultItem(
                    point_id="img-e891ee3d-00fc-5012-c0d0-649c69429gf4",
                    score=0.7567,
                    payload={
                        "s3_key": "images/beach-002.png",
                        "s3_uri": "s3://pipeline/images/beach-002.png",
                        "format": "png",
                        "width": 1280,
                        "height": 720,
                        "thumbnail_key": None,
                    },
                ),
            ],
            total=2,
        )
    )
    provider.close = MagicMock()
    return provider


@pytest.fixture
def mock_image_embedding_config() -> MagicMock:
    """Create a mock ImageEmbeddingConfig for testing.

    Returns:
        MagicMock: A mock image embedding configuration.
    """
    config = MagicMock()
    config.clip_url = "http://clip.test:8000"
    config.clip_model = "ViT-B/32"
    config.clip_backend = "clip"
    config.clip_timeout = 120
    config.clip_circuit_breaker_threshold = 5
    config.clip_circuit_breaker_timeout = 30
    config.clip_max_batch_size = 8
    config.clip_vector_size = 512
    return config


@pytest.fixture
def patch_clip_client(mock_clip_client: MagicMock):
    """Patch CLIPClient.from_config to return mock client.

    Args:
        mock_clip_client: Mock CLIP client fixture.

    Yields:
        Mock patch object.
    """
    with patch(
        "api.routes.CLIPClient.from_config",
        return_value=mock_clip_client,
    ) as mock:
        yield mock


@pytest.fixture
def patch_image_embedding_config(mock_image_embedding_config: MagicMock):
    """Patch ImageEmbeddingConfig.from_env to return mock config.

    Args:
        mock_image_embedding_config: Mock image embedding config fixture.

    Yields:
        Mock patch object.
    """
    with patch(
        "api.routes.ImageEmbeddingConfig.from_env",
        return_value=mock_image_embedding_config,
    ) as mock:
        yield mock
