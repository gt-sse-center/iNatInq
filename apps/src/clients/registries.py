"""Provider registry initialization.

This module registers default providers for embedding and vector database interfaces.
Registration happens at module import time to avoid circular dependencies between
interface definitions and concrete implementations.
"""

from .interfaces.embedding import register_provider as register_embedding_provider
from .interfaces.vector_db import register_provider as register_vector_db_provider
from .ollama import OllamaClient
from .qdrant import QdrantClientWrapper
from .weaviate import WeaviateClientWrapper


def _register_embedding_providers() -> None:
    """Register default embedding providers."""
    register_embedding_provider("ollama", OllamaClient)


def _register_vector_db_providers() -> None:
    """Register default vector database providers."""
    register_vector_db_provider("qdrant", QdrantClientWrapper)
    register_vector_db_provider("weaviate", WeaviateClientWrapper)


def register_all_providers() -> None:
    """Register all default providers.

    This function is called at module import time to register built-in providers.
    External providers can register themselves directly using the register_provider
    functions from the interface modules.
    """
    _register_embedding_providers()
    _register_vector_db_providers()


# Register all providers on module import
register_all_providers()
