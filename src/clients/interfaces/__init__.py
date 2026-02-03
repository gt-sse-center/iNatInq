"""Abstract base classes (interfaces) for client providers.

This sub-package contains the ABCs, configuration classes, and factory functions
that define the contracts for different types of providers (embedding, vector
database, etc.). Concrete implementations live in the parent `clients` package.
"""

from config import EmbeddingConfig, VectorDBConfig

from .embedding import EmbeddingProvider, ImageEmbeddingProvider, create_embedding_provider
from .embedding import register_provider as register_embedding_provider
from .vector_db import VectorDBProvider, create_vector_db_provider
from .vector_db import register_provider as register_vector_db_provider

__all__ = [
    "EmbeddingConfig",
    "EmbeddingProvider",
    "ImageEmbeddingProvider",
    "VectorDBConfig",
    "VectorDBProvider",
    "create_embedding_provider",
    "create_vector_db_provider",
    "register_embedding_provider",
    "register_vector_db_provider",
]
