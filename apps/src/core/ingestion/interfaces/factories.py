"""Factory classes for ingestion pipeline components.

This module provides factory classes that encapsulate the creation logic
for configs, clients, and vector points. These factories are shared between
Ray and Spark implementations to avoid code duplication.
"""

import os
import uuid

from clients.interfaces.embedding import create_embedding_provider
from clients.interfaces.vector_db import create_vector_db_provider
from clients.s3 import S3ClientWrapper
from clients.weaviate import WeaviateDataObject
from config import VectorDBConfig
from core.models import VectorPoint
from foundation.http import create_retry_session

from .types import (
    BatchEmbeddingResult,
    ContentResult,
    ProcessingClients,
    ProcessingConfig,
)


class VectorDBConfigFactory:
    """Factory for creating vector database configurations.

    Creates properly configured VectorDBConfig instances for Qdrant and Weaviate
    based on environment variables and Kubernetes namespace.

    Example:
        >>> factory = VectorDBConfigFactory(namespace="ml-system")
        >>> qdrant_cfg, weaviate_cfg = factory.create_both()
    """

    def __init__(self, namespace: str = "ml-system") -> None:
        """Initialize the factory.

        Args:
            namespace: Kubernetes namespace for service discovery.
        """
        self.namespace = namespace
        self._in_cluster = self._detect_in_cluster()

    @staticmethod
    def _detect_in_cluster() -> bool:
        """Detect if running inside a Kubernetes cluster."""
        if os.path.exists("/var/run/secrets/kubernetes.io/serviceaccount/token"):
            return True
        if os.getenv("KUBERNETES_SERVICE_HOST"):
            return True
        return os.getenv("PIPELINE_ENV", "").lower() == "cluster"

    @staticmethod
    def _env_str(key: str, default: str | None = None) -> str | None:
        """Read environment variable with default, treating empty as None."""
        v = os.getenv(key)
        if v is None or v == "":
            return default
        return v

    def _get_collection(self) -> str:
        """Get collection name from environment."""
        return self._env_str("VECTOR_DB_COLLECTION", "documents") or "documents"

    def create_qdrant_config(self) -> VectorDBConfig:
        """Create Qdrant configuration.

        Returns:
            VectorDBConfig configured for Qdrant.
        """
        default_url = f"http://qdrant.{self.namespace}:6333" if self._in_cluster else "http://localhost:6333"
        return VectorDBConfig(
            provider_type="qdrant",
            collection=self._get_collection(),
            qdrant_url=self._env_str("QDRANT_URL", default_url),
            qdrant_api_key=self._env_str("QDRANT_API_KEY"),
        )

    def create_weaviate_config(self) -> VectorDBConfig:
        """Create Weaviate configuration.

        Returns:
            VectorDBConfig configured for Weaviate.
        """
        default_url = (
            f"http://weaviate.{self.namespace}:8080" if self._in_cluster else "http://localhost:8080"
        )
        return VectorDBConfig(
            provider_type="weaviate",
            collection=self._get_collection(),
            weaviate_url=self._env_str("WEAVIATE_URL", default_url),
            weaviate_api_key=self._env_str("WEAVIATE_API_KEY"),
        )

    def create_both(self) -> tuple[VectorDBConfig, VectorDBConfig]:
        """Create both Qdrant and Weaviate configurations.

        Returns:
            Tuple of (qdrant_config, weaviate_config).
        """
        return (self.create_qdrant_config(), self.create_weaviate_config())


class ProcessingClientsFactory:
    """Factory for creating processing client bundles.

    Creates all required clients (S3, embedding, vector DBs) from a
    ProcessingConfig instance.

    Example:
        >>> factory = ProcessingClientsFactory()
        >>> clients = factory.create(config)
        >>> try:
        ...     # Use clients
        ... finally:
        ...     clients.close_sync()
    """

    def __init__(self, vector_db_factory: VectorDBConfigFactory | None = None) -> None:
        """Initialize the factory.

        Args:
            vector_db_factory: Optional VectorDBConfigFactory instance.
                If not provided, one will be created from the config's namespace.
        """
        self._vector_db_factory = vector_db_factory

    def create(self, config: ProcessingConfig) -> ProcessingClients:
        """Create all clients needed for processing.

        Args:
            config: Processing configuration.

        Returns:
            Bundle of initialized clients.
        """
        # Create S3 client
        s3 = S3ClientWrapper(
            endpoint_url=config.s3_endpoint,
            access_key_id=config.s3_access_key,
            secret_access_key=config.s3_secret_key,
        )

        # Create HTTP session with retries
        session = create_retry_session()

        # Create embedding provider
        embedder = create_embedding_provider(config.embedding_config, session=session)

        # Create vector DB providers
        db_factory = self._vector_db_factory or VectorDBConfigFactory(config.namespace)
        qdrant_config, weaviate_config = db_factory.create_both()
        qdrant_db = create_vector_db_provider(qdrant_config)
        weaviate_db = create_vector_db_provider(weaviate_config)

        return ProcessingClients(
            s3=s3,
            embedder=embedder,
            qdrant_db=qdrant_db,
            weaviate_db=weaviate_db,
            session=session,
        )


class VectorPointFactory:
    """Factory for creating vector points for Qdrant and Weaviate.

    Converts content and embeddings into database-specific point formats.

    Example:
        >>> factory = VectorPointFactory(s3_bucket="pipeline")
        >>> batch = factory.create_batch(contents, vectors)
    """

    def __init__(self, s3_bucket: str) -> None:
        """Initialize the factory.

        Args:
            s3_bucket: S3 bucket name for metadata.
        """
        self.s3_bucket = s3_bucket

    def _create_payload(self, content: ContentResult) -> dict:
        """Create metadata payload for a vector point.

        Args:
            content: The content result.

        Returns:
            Dictionary with s3_key, s3_bucket, s3_uri, and text.
        """
        return {
            "s3_key": content.s3_key,
            "s3_bucket": self.s3_bucket,
            "s3_uri": f"s3://{self.s3_bucket}/{content.s3_key}",
            "text": content.content,
        }

    def create_qdrant_point(
        self,
        content: ContentResult,
        vector: list[float],
        point_id: str | None = None,
    ) -> VectorPoint:
        """Create a Qdrant VectorPoint.

        Args:
            content: The content result.
            vector: The embedding vector.
            point_id: Optional UUID. If not provided, one is generated.

        Returns:
            VectorPoint for Qdrant.
        """
        return VectorPoint(
            id=point_id or str(uuid.uuid4()),
            vector=vector,
            payload=self._create_payload(content),
        )

    def create_weaviate_object(
        self,
        content: ContentResult,
        vector: list[float],
        object_id: str | None = None,
    ) -> WeaviateDataObject:
        """Create a Weaviate data object.

        Args:
            content: The content result.
            vector: The embedding vector.
            object_id: Optional UUID. If not provided, one is generated.

        Returns:
            WeaviateDataObject for Weaviate.
        """
        return WeaviateDataObject(
            uuid=object_id or str(uuid.uuid4()),
            properties=self._create_payload(content),
            vector=vector,
        )

    def create_pair(
        self,
        content: ContentResult,
        vector: list[float],
    ) -> tuple[VectorPoint, WeaviateDataObject]:
        """Create both Qdrant and Weaviate points with the same UUID.

        Args:
            content: The content result.
            vector: The embedding vector.

        Returns:
            Tuple of (VectorPoint, WeaviateDataObject) with matching UUIDs.
        """
        point_id = str(uuid.uuid4())
        return (
            self.create_qdrant_point(content, vector, point_id),
            self.create_weaviate_object(content, vector, point_id),
        )

    def create_batch(
        self,
        contents: list[ContentResult],
        vectors: list[list[float]],
    ) -> BatchEmbeddingResult:
        """Create vector points for a batch of content.

        Args:
            contents: List of content results.
            vectors: Corresponding embedding vectors.

        Returns:
            BatchEmbeddingResult with points for both databases.

        Raises:
            ValueError: If contents and vectors have different lengths.
        """
        if len(contents) != len(vectors):
            raise ValueError(f"Contents ({len(contents)}) and vectors ({len(vectors)}) must have same length")

        qdrant_points: list[VectorPoint] = []
        weaviate_objects: list[WeaviateDataObject] = []

        for content, vector in zip(contents, vectors, strict=True):
            qdrant_point, weaviate_obj = self.create_pair(content, vector)
            qdrant_points.append(qdrant_point)
            weaviate_objects.append(weaviate_obj)

        return BatchEmbeddingResult(
            qdrant_points=qdrant_points,
            weaviate_objects=weaviate_objects,
        )
