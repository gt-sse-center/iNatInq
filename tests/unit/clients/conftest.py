"""Shared fixtures for client tests.

This module provides common fixtures used across all client test modules,
including mock clients, circuit breakers, and client instances.
"""

# pylint: disable=redefined-outer-name
# Pytest fixtures intentionally redefine fixture names - this is expected behavior

from unittest.mock import AsyncMock, MagicMock, patch

import pybreaker
import pytest

from clients.ollama import OllamaClient
from clients.qdrant import QdrantClientWrapper
from clients.s3 import S3ClientWrapper

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
def mock_httpx_async_client() -> AsyncMock:
    """Create a mock httpx.AsyncClient for testing Ollama client.

    Returns:
        AsyncMock: A mock httpx.AsyncClient with post and aclose methods.
    """
    client = AsyncMock()
    client.post = AsyncMock()
    client.aclose = AsyncMock()
    client.is_closed = False
    return client


@pytest.fixture
def ollama_client() -> OllamaClient:
    """Create an OllamaClient instance for testing.

    Returns:
        OllamaClient: Configured client instance.
    """
    return OllamaClient(
        base_url="http://ollama.example.com:11434",
        model="nomic-embed-text",
        timeout_s=60,
    )


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
def mock_sync_client() -> MagicMock:
    """Create a mock QdrantClient (sync) for testing.

    Returns:
        MagicMock: A mock sync Qdrant client with common methods.
    """
    client = MagicMock()
    client.get_collection = MagicMock()
    client.update_collection = MagicMock()
    client.get_collections = MagicMock()
    client.create_collection = MagicMock()
    client.close = MagicMock()
    return client


@pytest.fixture
def qdrant_client(mock_async_client: AsyncMock, mock_sync_client: MagicMock) -> QdrantClientWrapper:
    """Create a QdrantClientWrapper instance with mocked async and sync clients.

    Args:
        mock_async_client: Mock AsyncQdrantClient fixture.
        mock_sync_client: Mock QdrantClient fixture.

    Returns:
        QdrantClientWrapper: Configured client with mocked clients.
    """
    with (
        patch("clients.qdrant.AsyncQdrantClient", return_value=mock_async_client),
        patch("clients.qdrant.QdrantClient", return_value=mock_sync_client),
    ):
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
    client.put_object = MagicMock()
    client.get_object = MagicMock()
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
