"""Unit tests for core.ingestion.interfaces.factories module.

This file tests the factory classes that encapsulate creation logic for
configs, clients, and vector points.

# Test Coverage

The tests cover:
  - VectorDBConfigFactory: Config creation, environment detection
  - ProcessingClientsFactory: Client bundle creation
  - VectorPointFactory: Vector point creation for both databases

# Test Structure

Tests use pytest class-based organization with mocking for external dependencies.

# Running Tests

Run with: pytest tests/unit/core/ingestion/interfaces/test_factories.py
"""

from unittest.mock import MagicMock, patch

import pytest
from config import EmbeddingConfig
from core.ingestion.interfaces.factories import (
    ProcessingClientsFactory,
    VectorDBConfigFactory,
    VectorPointFactory,
)
from core.ingestion.interfaces.types import (
    ContentResult,
    ProcessingConfig,
)

# =============================================================================
# VectorDBConfigFactory Tests
# =============================================================================


class TestVectorDBConfigFactory:
    """Test suite for VectorDBConfigFactory."""

    def test_creates_factory_with_namespace(self) -> None:
        """Test factory initialization with namespace.

        **Why this test is important:**
          - Namespace is required for service discovery
          - Validates constructor

        **What it tests:**
          - Factory stores namespace
        """
        factory = VectorDBConfigFactory(namespace="ml-system")

        assert factory.namespace == "ml-system"

    def test_creates_qdrant_config(self) -> None:
        """Test creating Qdrant configuration.

        **Why this test is important:**
          - Qdrant is primary vector DB
          - Validates config creation

        **What it tests:**
          - Config has correct provider_type
        """
        factory = VectorDBConfigFactory(namespace="ml-system")
        config = factory.create_qdrant_config()

        assert config.provider_type == "qdrant"

    def test_creates_weaviate_config(self) -> None:
        """Test creating Weaviate configuration.

        **Why this test is important:**
          - Weaviate is secondary vector DB
          - Validates config creation

        **What it tests:**
          - Config has correct provider_type
        """
        factory = VectorDBConfigFactory(namespace="ml-system")
        config = factory.create_weaviate_config()

        assert config.provider_type == "weaviate"

    def test_creates_both_configs(self) -> None:
        """Test creating both configurations.

        **Why this test is important:**
          - Common use case for dual-write
          - Validates tuple return

        **What it tests:**
          - Returns (qdrant, weaviate) tuple
        """
        factory = VectorDBConfigFactory(namespace="ml-system")
        qdrant, weaviate = factory.create_both()

        assert qdrant.provider_type == "qdrant"
        assert weaviate.provider_type == "weaviate"

    @patch.dict("os.environ", {"QDRANT_URL": "http://custom-qdrant:6333"}, clear=False)
    def test_uses_env_var_for_qdrant_url(self) -> None:
        """Test that factory uses QDRANT_URL env var.

        **Why this test is important:**
          - Environment-based configuration
          - Validates override behavior

        **What it tests:**
          - Custom URL from env var used
        """
        factory = VectorDBConfigFactory(namespace="ml-system")
        config = factory.create_qdrant_config()

        assert config.qdrant_url == "http://custom-qdrant:6333"

    @patch.dict("os.environ", {"WEAVIATE_URL": "http://custom-weaviate:8080"}, clear=False)
    def test_uses_env_var_for_weaviate_url(self) -> None:
        """Test that factory uses WEAVIATE_URL env var.

        **Why this test is important:**
          - Environment-based configuration
          - Validates override behavior

        **What it tests:**
          - Custom URL from env var used
        """
        factory = VectorDBConfigFactory(namespace="ml-system")
        config = factory.create_weaviate_config()

        assert config.weaviate_url == "http://custom-weaviate:8080"

    @patch.dict(
        "os.environ",
        {
            "WEAVIATE_URL": "https://my-cluster.weaviate.cloud",
            "WEAVIATE_API_KEY": "cloud-key",
            "WEAVIATE_GRPC_HOST": "grpc-my-cluster.weaviate.cloud",
        },
        clear=False,
    )
    def test_uses_env_vars_for_weaviate_cloud(self) -> None:
        """Test that factory uses Weaviate Cloud env vars.

        **Why this test is important:**
          - Weaviate Cloud requires URL, API key, and gRPC host
          - Validates all cloud configuration is passed through

        **What it tests:**
          - Cloud URL, API key, and gRPC host from env vars
        """
        factory = VectorDBConfigFactory(namespace="ml-system")
        config = factory.create_weaviate_config()

        assert config.weaviate_url == "https://my-cluster.weaviate.cloud"
        assert config.weaviate_api_key == "cloud-key"
        assert config.weaviate_grpc_host == "grpc-my-cluster.weaviate.cloud"

    @patch.dict("os.environ", {"VECTOR_DB_COLLECTION": "my-docs"}, clear=False)
    def test_uses_env_var_for_collection(self) -> None:
        """Test that factory uses VECTOR_DB_COLLECTION env var.

        **Why this test is important:**
          - Collection name from environment
          - Validates shared config

        **What it tests:**
          - Collection name from env var
        """
        factory = VectorDBConfigFactory(namespace="ml-system")
        config = factory.create_qdrant_config()

        assert config.collection == "my-docs"

    def test_defaults_collection_to_documents(self) -> None:
        """Test that collection defaults to 'documents'.

        **Why this test is important:**
          - Sensible default needed
          - Validates fallback

        **What it tests:**
          - Default collection is 'documents'
        """
        factory = VectorDBConfigFactory(namespace="ml-system")
        config = factory.create_qdrant_config()

        # Default when env var not set
        assert config.collection == "documents"


# =============================================================================
# VectorPointFactory Tests
# =============================================================================


class TestVectorPointFactory:
    """Test suite for VectorPointFactory."""

    def test_creates_factory_with_bucket(self) -> None:
        """Test factory initialization with bucket.

        **Why this test is important:**
          - Bucket is required for S3 URI
          - Validates constructor

        **What it tests:**
          - Factory stores bucket
        """
        factory = VectorPointFactory(s3_bucket="pipeline")

        assert factory.s3_bucket == "pipeline"

    def test_creates_qdrant_point(self) -> None:
        """Test creating a Qdrant VectorPoint.

        **Why this test is important:**
          - VectorPoint is core data structure
          - Validates all fields

        **What it tests:**
          - Point has correct structure
        """
        factory = VectorPointFactory(s3_bucket="pipeline")
        content = ContentResult(s3_key="doc.txt", content="Hello")
        vector = [0.1, 0.2, 0.3]

        point = factory.create_qdrant_point(content, vector)

        assert point.vector == vector
        assert point.payload["s3_key"] == "doc.txt"
        assert point.payload["s3_bucket"] == "pipeline"
        assert point.payload["s3_uri"] == "s3://pipeline/doc.txt"
        assert point.payload["text"] == "Hello"

    def test_creates_qdrant_point_with_custom_id(self) -> None:
        """Test creating a point with custom ID.

        **Why this test is important:**
          - Custom IDs for deduplication
          - Validates ID parameter

        **What it tests:**
          - Custom ID is used
        """
        factory = VectorPointFactory(s3_bucket="pipeline")
        content = ContentResult(s3_key="doc.txt", content="Hello")
        vector = [0.1, 0.2, 0.3]

        point = factory.create_qdrant_point(content, vector, point_id="custom-uuid")

        assert point.id == "custom-uuid"

    def test_creates_weaviate_object(self) -> None:
        """Test creating a Weaviate data object.

        **Why this test is important:**
          - Weaviate uses different format
          - Validates properties structure

        **What it tests:**
          - Object has correct structure
        """
        factory = VectorPointFactory(s3_bucket="pipeline")
        content = ContentResult(s3_key="doc.txt", content="Hello")
        vector = [0.1, 0.2, 0.3]

        obj = factory.create_weaviate_object(content, vector)

        assert obj.vector == vector
        assert obj.properties["s3_key"] == "doc.txt"
        assert obj.properties["text"] == "Hello"

    def test_creates_matching_pair(self) -> None:
        """Test creating matching Qdrant and Weaviate points.

        **Why this test is important:**
          - Both databases need same UUID
          - Critical for data consistency

        **What it tests:**
          - UUIDs match between points
        """
        factory = VectorPointFactory(s3_bucket="pipeline")
        content = ContentResult(s3_key="doc.txt", content="Hello")
        vector = [0.1, 0.2, 0.3]

        qdrant_point, weaviate_obj = factory.create_pair(content, vector)

        assert qdrant_point.id == weaviate_obj.uuid
        assert qdrant_point.vector == weaviate_obj.vector

    def test_creates_batch(self) -> None:
        """Test creating batch of vector points.

        **Why this test is important:**
          - Batch creation is common
          - Validates length matching

        **What it tests:**
          - Batch has correct counts
        """
        factory = VectorPointFactory(s3_bucket="pipeline")
        contents = [
            ContentResult(s3_key="doc1.txt", content="Hello"),
            ContentResult(s3_key="doc2.txt", content="World"),
        ]
        vectors = [[0.1, 0.2], [0.3, 0.4]]

        batch = factory.create_batch(contents, vectors)

        assert len(batch.qdrant_points) == 2
        assert len(batch.weaviate_objects) == 2
        assert batch.qdrant_points[0].payload["s3_key"] == "doc1.txt"
        assert batch.qdrant_points[1].payload["s3_key"] == "doc2.txt"

    def test_batch_raises_on_length_mismatch(self) -> None:
        """Test that batch creation fails on length mismatch.

        **Why this test is important:**
          - Data integrity check
          - Prevents silent bugs

        **What it tests:**
          - ValueError raised on mismatch
        """
        factory = VectorPointFactory(s3_bucket="pipeline")
        contents = [ContentResult(s3_key="doc1.txt", content="Hello")]
        vectors = [[0.1, 0.2], [0.3, 0.4]]  # 2 vectors, 1 content

        with pytest.raises(ValueError, match="must have same length"):
            factory.create_batch(contents, vectors)


# =============================================================================
# ProcessingClientsFactory Tests
# =============================================================================


class TestProcessingClientsFactory:
    """Test suite for ProcessingClientsFactory."""

    def test_creates_factory(self) -> None:
        """Test factory creation.

        **Why this test is important:**
          - Factory is entry point for clients
          - Validates initialization

        **What it tests:**
          - Factory created successfully
        """
        factory = ProcessingClientsFactory()

        assert factory is not None

    @patch("core.ingestion.interfaces.factories.S3ClientWrapper")
    @patch("core.ingestion.interfaces.factories.create_retry_session")
    @patch("core.ingestion.interfaces.factories.create_embedding_provider")
    @patch("core.ingestion.interfaces.factories.create_vector_db_provider")
    def test_creates_all_clients(
        self,
        mock_vector_db_factory,
        mock_embedding_factory,
        mock_session_factory,
        mock_s3_class,
    ) -> None:
        """Test that factory creates all required clients.

        **Why this test is important:**
          - Core functionality of factory
          - Validates all clients created

        **What it tests:**
          - All client factories called
        """
        mock_s3 = MagicMock()
        mock_s3_class.return_value = mock_s3
        mock_session = MagicMock()
        mock_session_factory.return_value = mock_session
        mock_embedder = MagicMock()
        mock_embedding_factory.return_value = mock_embedder
        mock_vector_db = MagicMock()
        mock_vector_db_factory.return_value = mock_vector_db

        embed_config = EmbeddingConfig(
            provider_type="ollama",
            ollama_model="nomic-embed-text",
            ollama_url="http://localhost:11434",
        )

        config = ProcessingConfig(
            s3_endpoint="http://localhost:9000",
            s3_access_key="minioadmin",
            s3_secret_key="minioadmin",
            s3_bucket="documents",
            embedding_config=embed_config,
            collection="test-collection",
        )

        factory = ProcessingClientsFactory()
        clients = factory.create(config)

        assert clients.s3 == mock_s3
        assert clients.embedder == mock_embedder
        mock_s3_class.assert_called_once()
        mock_embedding_factory.assert_called_once()
        # Vector DB factory called twice (qdrant + weaviate)
        assert mock_vector_db_factory.call_count == 2

    @patch("core.ingestion.interfaces.factories.S3ClientWrapper")
    @patch("core.ingestion.interfaces.factories.create_retry_session")
    @patch("core.ingestion.interfaces.factories.create_embedding_provider")
    @patch("core.ingestion.interfaces.factories.create_vector_db_provider")
    def test_uses_config_values(
        self,
        mock_vector_db_factory,
        mock_embedding_factory,
        mock_session_factory,
        mock_s3_class,
    ) -> None:
        """Test that factory uses config values correctly.

        **Why this test is important:**
          - Config must be passed to clients
          - Validates parameter passing

        **What it tests:**
          - S3 client uses config values
        """
        mock_session_factory.return_value = MagicMock()
        mock_embedding_factory.return_value = MagicMock()
        mock_vector_db_factory.return_value = MagicMock()

        embed_config = EmbeddingConfig(
            provider_type="ollama",
            ollama_model="nomic-embed-text",
            ollama_url="http://localhost:11434",
        )

        config = ProcessingConfig(
            s3_endpoint="http://minio:9000",
            s3_access_key="mykey",
            s3_secret_key="mysecret",
            s3_bucket="documents",
            embedding_config=embed_config,
            collection="test-collection",
        )

        factory = ProcessingClientsFactory()
        factory.create(config)

        mock_s3_class.assert_called_once_with(
            endpoint_url="http://minio:9000",
            access_key_id="mykey",
            secret_access_key="mysecret",
        )
