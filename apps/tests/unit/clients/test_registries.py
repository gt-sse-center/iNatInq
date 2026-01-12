"""Unit tests for clients.registries module.

This file tests the provider registry functionality which registers default
embedding and vector database providers.

# Test Coverage

The tests cover:
  - Provider Registration: Embedding and vector DB provider registration
  - Registry Functions: register_all_providers, provider lookup
  - Module Import: Providers are registered on import

# Test Structure

Tests use pytest class-based organization. Registry state is tested by checking
that providers can be created using the registry functions.

# Running Tests

Run with: pytest tests/unit/clients/test_registries.py
"""

from unittest.mock import MagicMock, patch

from clients.interfaces.embedding import create_embedding_provider
from clients.interfaces.vector_db import create_vector_db_provider
from clients.ollama import OllamaClient
from clients.qdrant import QdrantClientWrapper
from clients.weaviate import WeaviateClientWrapper
from config import EmbeddingConfig, VectorDBConfig

# =============================================================================
# Provider Registration Tests
# =============================================================================


class TestProviderRegistration:
    """Test suite for provider registration."""

    def test_ollama_provider_is_registered(self) -> None:
        """Test that Ollama provider is registered.

        **Why this test is important:**
          - Provider registration enables factory-based creation
          - Validates that Ollama is available as embedding provider
          - Critical for configuration-driven initialization
          - Validates registry functionality

        **What it tests:**
          - Ollama provider can be created via create_embedding_provider
          - Created instance is OllamaClient
        """
        config = EmbeddingConfig(
            provider_type="ollama",
            ollama_url="http://ollama.example.com:11434",
            ollama_model="test-model",
        )

        with patch("clients.ollama.create_retry_session"):
            provider = create_embedding_provider(config)

        assert isinstance(provider, OllamaClient)
        assert provider.base_url == "http://ollama.example.com:11434"
        assert provider.model == "test-model"

    def test_qdrant_provider_is_registered(self) -> None:
        """Test that Qdrant provider is registered.

        **Why this test is important:**
          - Provider registration enables factory-based creation
          - Validates that Qdrant is available as vector DB provider
          - Critical for configuration-driven initialization
          - Validates registry functionality

        **What it tests:**
          - Qdrant provider can be created via create_vector_db_provider
          - Created instance is QdrantClientWrapper
        """
        config = VectorDBConfig(
            provider_type="qdrant",
            collection="test-collection",
            qdrant_url="http://qdrant.example.com:6333",
        )

        with (
            patch("clients.qdrant.AsyncQdrantClient"),
            patch("clients.qdrant.QdrantClient"),
        ):
            provider = create_vector_db_provider(config)

        assert isinstance(provider, QdrantClientWrapper)
        assert provider.url == "http://qdrant.example.com:6333"

    def test_weaviate_provider_is_registered(self) -> None:
        """Test that Weaviate provider is registered.

        **Why this test is important:**
          - Provider registration enables factory-based creation
          - Validates that Weaviate is available as vector DB provider
          - Critical for configuration-driven initialization
          - Validates registry functionality

        **What it tests:**
          - Weaviate provider can be created via create_vector_db_provider
          - Created instance is WeaviateClientWrapper
        """
        config = VectorDBConfig(
            provider_type="weaviate",
            collection="test-collection",
            weaviate_url="http://weaviate.example.com:8080",
        )

        # Use MagicMock with spec to avoid AsyncMock behavior and patch ConnectionParams
        mock_client = MagicMock()
        with patch("clients.weaviate.WeaviateAsyncClient") as mock_weaviate_cls:
            with patch(
                "clients.weaviate.ConnectionParams.from_params"
            ) as mock_conn_params:
                mock_weaviate_cls.return_value = mock_client
                mock_conn_params.return_value = MagicMock()
                provider = create_vector_db_provider(config)

        assert isinstance(provider, WeaviateClientWrapper)
        assert provider.url == "http://weaviate.example.com:8080"

    def test_multiple_vector_db_providers_registered(self) -> None:
        """Test that multiple vector DB providers are registered.

        **Why this test is important:**
          - Multiple providers enable provider switching
          - Validates that both Qdrant and Weaviate are available
          - Critical for flexibility and provider choice
          - Validates multi-provider support

        **What it tests:**
          - Qdrant provider can be created
          - Weaviate provider can be created
          - Both use the same factory function
        """
        qdrant_config = VectorDBConfig(
            provider_type="qdrant",
            collection="test-collection",
            qdrant_url="http://qdrant.example.com:6333",
        )

        weaviate_config = VectorDBConfig(
            provider_type="weaviate",
            collection="test-collection",
            weaviate_url="http://weaviate.example.com:8080",
        )

        with (
            patch("clients.qdrant.AsyncQdrantClient"),
            patch("clients.qdrant.QdrantClient"),
            patch("clients.weaviate.WeaviateAsyncClient"),
        ):
            qdrant_provider = create_vector_db_provider(qdrant_config)
            weaviate_provider = create_vector_db_provider(weaviate_config)

        assert isinstance(qdrant_provider, QdrantClientWrapper)
        assert isinstance(weaviate_provider, WeaviateClientWrapper)
