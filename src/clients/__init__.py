"""External service clients (MinIO/S3, Qdrant, Embedding Providers).

This module provides factory functions for creating configured client instances
from centralized configuration.
"""

from typing import cast

from config import EmbeddingConfig, MinIOConfig, VectorDBConfig, get_settings

# Import registries to trigger provider registration
from . import registries as _  # noqa: F401
from .interfaces.embedding import EmbeddingProvider, create_embedding_provider
from .interfaces.vector_db import VectorDBProvider, create_vector_db_provider
from .s3 import S3ClientWrapper


def create_s3_client(config: "MinIOConfig | None" = None) -> "S3ClientWrapper":
    """Create a configured S3/MinIO client.

    Args:
        config: Optional MinIOConfig. If None, uses settings from
        get_settings().

    Returns:
        Configured S3ClientWrapper instance.

    Example:
        ```python
        from clients import create_s3_client

        client = create_s3_client()
        client.ensure_bucket("pipeline")
        ```
    """
    if config is None:
        config = get_settings().minio

    return S3ClientWrapper(
        endpoint_url=config.endpoint_url,
        access_key_id=config.access_key_id,
        secret_access_key=config.secret_access_key,
        region_name=config.region,
    )


def create_embedding_client(
    config: EmbeddingConfig | None = None,
) -> "EmbeddingProvider":
    """Create a configured embedding provider.

    Args:
        config: Optional EmbeddingConfig. If None, uses settings from
        get_settings().

    Returns:
        EmbeddingProvider instance (OllamaClient, OpenAIClient, etc.).

    Example:
        ```python
        from clients import create_embedding_client

        provider = create_embedding_client()
        vector = provider.embed("hello world")
        ```
    """
    if config is None:
        # get_settings().embedding is EmbeddingProviderConfig (alias for
        # EmbeddingConfig) Cast to EmbeddingConfig for type checker
        config = cast(EmbeddingConfig, get_settings().embedding)

    return create_embedding_provider(config)


def create_vector_db_client(
    config: VectorDBConfig | None = None,
) -> "VectorDBProvider":
    """Create a configured vector database provider.

    Args:
        config: Optional VectorDBConfig. If None, uses settings from
        get_settings().

    Returns:
        VectorDBProvider instance (QdrantClientWrapper, WeaviateClient, etc.).

    Example:
        ```python
        from clients import create_vector_db_client

        provider = create_vector_db_client()
        results = provider.search_async(collection="documents", query_vector=[...],
        limit=10)
        ```
    """
    if config is None:
        # get_settings().vector_db is VectorDBProviderConfig (alias for
        # VectorDBConfig) Cast to VectorDBConfig for type checker
        config = cast(VectorDBConfig, get_settings().vector_db)

    return create_vector_db_provider(config)


__all__ = [
    "EmbeddingConfig",
    "EmbeddingProvider",
    "S3ClientWrapper",
    "VectorDBConfig",
    "VectorDBProvider",
    "create_embedding_client",
    "create_embedding_provider",
    "create_s3_client",
    "create_vector_db_client",
    "create_vector_db_provider",
]
