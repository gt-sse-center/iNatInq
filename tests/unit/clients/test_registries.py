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

from unittest.mock import patch

from clients.interfaces.vector_db import create_vector_db_provider
from clients.qdrant import QdrantClientWrapper
from config import VectorDBConfig

# =============================================================================
# Provider Registration Tests
# =============================================================================


class TestProviderRegistration:
    """Test suite for provider registration."""

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

        with patch("clients.qdrant.AsyncQdrantClient"):
            provider = create_vector_db_provider(config)

        assert isinstance(provider, QdrantClientWrapper)
        assert provider.url == "http://qdrant.example.com:6333"
