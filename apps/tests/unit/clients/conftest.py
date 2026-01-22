"""Shared fixtures for client tests.

This module provides common fixtures used across all client test modules,
including mock clients, circuit breakers, and client instances.
"""

# pylint: disable=redefined-outer-name
# Pytest fixtures intentionally redefine fixture names - this is expected behavior

from unittest.mock import AsyncMock, MagicMock, patch

import pybreaker
import pytest
import requests

from clients.ollama import OllamaClient
from clients.qdrant import QdrantClientWrapper
from clients.s3 import S3ClientWrapper
from clients.weaviate import WeaviateClientWrapper

# =============================================================================
# Common Fixtures
# =============================================================================


@pytest.fixture
def mock_circuit_breaker() -> MagicMock:
    """Create a mock circuit breaker for testing.

    Returns:
        MagicMock: A mock circuit breaker that passes through function calls.
    """
    breaker = MagicMock(spec=pybreaker.CircuitBreaker)
    breaker.call = MagicMock(side_effect=lambda func: func())
    breaker.current_state = pybreaker.STATE_CLOSED
    return breaker


# =============================================================================
# Ollama Fixtures
# =============================================================================


@pytest.fixture
def mock_session() -> MagicMock:
    """Create a mock requests.Session for testing Ollama client.

    Returns:
        MagicMock: A mock requests.Session with post method.
    """
    session = MagicMock(spec=requests.Session)
    session.post = MagicMock()
    return session


@pytest.fixture
def ollama_client(mock_session: MagicMock) -> OllamaClient:
    """Create an OllamaClient instance with mocked session.

    Args:
        mock_session: Mock requests.Session fixture.

    Returns:
        OllamaClient: Configured client with mocked session.
    """
    client = OllamaClient(
        base_url="http://ollama.example.com:11434",
        model="nomic-embed-text",
        timeout_s=60,
    )
    client.set_session(mock_session)
    return client


# =============================================================================
# Qdrant Fixtures
# =============================================================================


@pytest.fixture
def mock_async_client() -> AsyncMock:
    """Create a mock AsyncQdrantClient for testing.

    Returns:
        AsyncMock: A mock async Qdrant client with common methods.
    """
    client = AsyncMock()
    client.get_collections = AsyncMock()
    client.create_collection = AsyncMock()
    client.search = AsyncMock()
    client.upsert = AsyncMock()
    client.update_collection = AsyncMock()
    client.close = AsyncMock()
    return client


@pytest.fixture
def qdrant_client(mock_async_client: AsyncMock) -> QdrantClientWrapper:
    """Create a QdrantClientWrapper instance with mocked client.

    Args:
        mock_async_client: Mock AsyncQdrantClient fixture.

    Returns:
        QdrantClientWrapper: Configured client with mocked async client.
    """
    with patch("clients.qdrant.AsyncQdrantClient", return_value=mock_async_client):
        client = QdrantClientWrapper(url="http://qdrant.example.com:6333")
    return client


# =============================================================================
# S3 Fixtures
# =============================================================================


@pytest.fixture
def mock_boto3_client() -> MagicMock:
    """Create a mock boto3 S3 client for testing.

    Returns:
        MagicMock: A mock boto3 S3 client with common S3 methods.
    """
    client = MagicMock()
    client.head_bucket = MagicMock()
    client.create_bucket = MagicMock()
    client.put_object = MagicMock()
    client.get_object = MagicMock()
    client.head_object = MagicMock()
    client.get_paginator = MagicMock()
    return client


@pytest.fixture
def s3_client(mock_boto3_client: MagicMock) -> S3ClientWrapper:
    """Create an S3ClientWrapper instance with mocked boto3 client.

    Args:
        mock_boto3_client: Mock boto3 S3 client fixture.

    Returns:
        S3ClientWrapper: Configured client with mocked boto3 client.
    """
    with patch("clients.s3.boto3.client", return_value=mock_boto3_client):
        client = S3ClientWrapper(
            endpoint_url="http://minio.example.com:9000",
            access_key_id="test-key",
            secret_access_key="test-secret",
            region_name="us-east-1",
        )
    return client


# =============================================================================
# Weaviate Fixtures
# =============================================================================


@pytest.fixture
def mock_weaviate_client() -> AsyncMock:
    """Create a mock WeaviateAsyncClient for testing.

    Returns:
        AsyncMock: A mock async Weaviate client with collections API.
    """
    client = AsyncMock()
    client.collections = MagicMock()
    client.collections.exists = AsyncMock()
    client.collections.create = AsyncMock()
    client.collections.get = MagicMock()
    return client


@pytest.fixture
def weaviate_client(mock_weaviate_client: AsyncMock) -> WeaviateClientWrapper:
    """Create a WeaviateClientWrapper instance with mocked client.

    Args:
        mock_weaviate_client: Mock WeaviateAsyncClient fixture.

    Returns:
        WeaviateClientWrapper: Configured client with mocked Weaviate client.
    """
    with patch("clients.weaviate.WeaviateAsyncClient", return_value=mock_weaviate_client):
        client = WeaviateClientWrapper(url="http://weaviate.example.com:8080")
    return client
