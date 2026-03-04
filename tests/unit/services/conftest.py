"""Shared fixtures for service tests.

This module provides common fixtures used across all service test modules,
including mock clients, configs, and service instances.
"""

# pylint: disable=redefined-outer-name
# Pytest fixtures intentionally redefine fixture names - this is expected behavior

from unittest.mock import AsyncMock, MagicMock

import pytest

from clients.interfaces.embedding import EmbeddingProvider
from clients.interfaces.vector_db import VectorDBProvider
from config import RayJobConfig, VectorDBConfig
from core.models import SearchResultItem, SearchResults
from core.services.ray_service import RayService
from core.services.search_service import ImageSearchService

# =============================================================================
# Mock Clients
# =============================================================================


@pytest.fixture
def mock_ray_client() -> MagicMock:
    """Create a mock Ray JobSubmissionClient for testing.

    Returns:
        MagicMock: A mock Ray client with common methods.
    """
    client = MagicMock()
    client.submit_job = MagicMock()
    client.get_job_status = MagicMock()
    client.get_job_info = MagicMock()
    client.get_job_logs = MagicMock()
    client.stop_job = MagicMock()
    return client


@pytest.fixture
def mock_vector_db_provider() -> MagicMock:
    """Create a mock VectorDBProvider for testing.

    Returns:
        MagicMock: A mock vector database provider with search method.
    """
    provider = MagicMock(spec=VectorDBProvider)
    provider.search_async = AsyncMock(
        return_value=SearchResults(
            items=[
                SearchResultItem(
                    point_id="1",
                    score=0.95,
                    payload={"text": "test document 1", "source": "test1.txt"},
                ),
                SearchResultItem(
                    point_id="2",
                    score=0.85,
                    payload={"text": "test document 2", "source": "test2.txt"},
                ),
            ],
            total=2,
        )
    )
    return provider


# =============================================================================
# Configuration Fixtures
# =============================================================================


@pytest.fixture
def vector_db_config() -> VectorDBConfig:
    """Create a VectorDBConfig for testing.

    Returns:
        VectorDBConfig: Test vector database configuration.
    """
    return VectorDBConfig(
        provider_type="qdrant", qdrant_url="http://qdrant.test:6333", collection="test_documents"
    )


@pytest.fixture
def ray_job_config() -> RayJobConfig:
    """Create a RayJobConfig for testing.

    Returns:
        RayJobConfig: Test Ray job configuration.
    """
    return RayJobConfig(ray_address="http://ray-head.ml-system:8265")


# =============================================================================
# Service Fixtures
# =============================================================================


@pytest.fixture
def ray_service() -> RayService:
    """Create a RayService instance.

    Returns:
        RayService: Configured service for testing.
    """
    return RayService()


@pytest.fixture
def mock_embedding_provider() -> MagicMock:
    """Create a mock EmbeddingProvider for search testing."""

    provider = MagicMock(spec=EmbeddingProvider)
    provider.embed_text = AsyncMock(return_value=[0.1, 0.2, 0.3])
    provider.vector_size = 512
    return provider


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
                    point_id="img-1",
                    score=0.92,
                    payload={
                        "s3_key": "images/sunset.jpg",
                        "s3_uri": "s3://pipeline/images/sunset.jpg",
                        "format": "jpeg",
                        "width": 1920,
                        "height": 1080,
                        "thumbnail_key": "thumbnails/sunset.jpg",
                    },
                ),
                SearchResultItem(
                    point_id="img-2",
                    score=0.85,
                    payload={
                        "s3_key": "images/beach.png",
                        "s3_uri": "s3://pipeline/images/beach.png",
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
    return provider


@pytest.fixture
def image_search_service(
    mock_embedding_provider: MagicMock, mock_image_vector_db_provider: MagicMock
) -> ImageSearchService:
    """Create an ImageSearchService instance with mocked providers.

    Args:
        mock_embedding_provider: Mock EmbeddingProvider fixture.
        mock_image_vector_db_provider: Mock VectorDBProvider fixture with image results.

    Returns:
        ImageSearchService: Configured service with mocked providers.
    """
    return ImageSearchService(
        embedding_provider=mock_embedding_provider,
        vector_db_provider=mock_image_vector_db_provider,
    )
