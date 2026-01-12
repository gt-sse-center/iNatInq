"""Shared fixtures for service tests.

This module provides common fixtures used across all service test modules,
including mock clients, configs, and service instances.
"""

# pylint: disable=redefined-outer-name
# Pytest fixtures intentionally redefine fixture names - this is expected behavior

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from clients.interfaces.embedding import EmbeddingProvider
from clients.interfaces.vector_db import VectorDBProvider
from clients.k8s_spark import SparkJobClient
from config import EmbeddingConfig, RayJobConfig, VectorDBConfig
from core.models import SearchResultItem, SearchResults
from core.services.ray_service import RayService
from core.services.search_service import SearchService
from core.services.spark_service import SparkService

# =============================================================================
# Mock Clients
# =============================================================================


@pytest.fixture
def mock_spark_client() -> MagicMock:
    """Create a mock SparkJobClient for testing.

    Returns:
        MagicMock: A mock Spark job client with common methods.
    """
    client = MagicMock(spec=SparkJobClient)
    client.submit_job = MagicMock()
    client.get_job_status = MagicMock()
    client.list_jobs = MagicMock()
    client.delete_job = MagicMock()
    return client


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
def mock_embedding_provider() -> MagicMock:
    """Create a mock EmbeddingProvider for testing.

    Returns:
        MagicMock: A mock embedding provider with embed method.
    """
    provider = MagicMock(spec=EmbeddingProvider)
    provider.embed = MagicMock(return_value=[0.1, 0.2, 0.3])
    provider.embed_async = AsyncMock(return_value=[0.1, 0.2, 0.3])
    provider.vector_size = 768
    return provider


@pytest.fixture
def mock_vector_db_provider() -> MagicMock:
    """Create a mock VectorDBProvider for testing.

    Returns:
        MagicMock: A mock vector database provider with search method.
    """
    provider = MagicMock(spec=VectorDBProvider)
    provider.search = AsyncMock(
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
def embedding_config() -> EmbeddingConfig:
    """Create an EmbeddingConfig for testing.

    Returns:
        EmbeddingConfig: Test embedding configuration.
    """
    return EmbeddingConfig(
        provider_type="ollama",
        ollama_url="http://ollama.test:11434",
        ollama_model="nomic-embed-text",
        vector_size=768,
    )


@pytest.fixture
def vector_db_config() -> VectorDBConfig:
    """Create a VectorDBConfig for testing.

    Returns:
        VectorDBConfig: Test vector database configuration.
    """
    return VectorDBConfig(
        provider_type="qdrant",
        qdrant_url="http://qdrant.test:6333",
        collection="test_documents"
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
def spark_service(mock_spark_client: MagicMock) -> SparkService:
    """Create a SparkService instance with mocked client.

    Args:
        mock_spark_client: Mock SparkJobClient fixture.

    Returns:
        SparkService: Configured service with mocked client.
    """
    service = SparkService(namespace="test-namespace")
    service.client = mock_spark_client
    return service


@pytest.fixture
def ray_service() -> RayService:
    """Create a RayService instance.

    Returns:
        RayService: Configured service for testing.
    """
    return RayService()


@pytest.fixture
def search_service(
    mock_embedding_provider: MagicMock, mock_vector_db_provider: MagicMock
) -> SearchService:
    """Create a SearchService instance with mocked providers.

    Args:
        mock_embedding_provider: Mock EmbeddingProvider fixture.
        mock_vector_db_provider: Mock VectorDBProvider fixture.

    Returns:
        SearchService: Configured service with mocked providers.
    """
    return SearchService(
        embedding_provider=mock_embedding_provider,
        vector_db_provider=mock_vector_db_provider,
    )

