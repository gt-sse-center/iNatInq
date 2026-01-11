"""Embedding provider interface and factory.

This module defines the `EmbeddingProvider` ABC and factory functions for embedding
providers. Configuration classes are in `pipeline.config`.
Concrete implementations live in the parent `clients` package (e.g., `OllamaClient`).
"""


from abc import ABC, abstractmethod

import requests

from config import EmbeddingConfig


class EmbeddingProvider(ABC):
    """Abstract base class for embedding generation providers.

    This class defines the interface that all embedding providers must implement.
    Each provider (OllamaClient, OpenAIClient, etc.) inherits from this class and
    implements the required methods.

    Example:
        ```python
        class MyEmbeddingClient(EmbeddingProvider):
            def embed(self, text: str) -> list[float]:
                # Implementation
                return [0.1, 0.2, ...]

            async def embed_async(self, text: str) -> list[float]:
                # Async implementation
                return [0.1, 0.2, ...]

            @property
            def vector_size(self) -> int:
                return 768

            def close(self) -> None:
                # Close HTTP sessions and cleanup resources
                pass
        ```
    """

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Input text to embed.

        Returns:
            List of floats representing the embedding vector. The dimension
            depends on the provider/model (e.g., 768 for nomic-embed-text).

        Raises:
            UpstreamError: If the embedding service is unreachable or returns
                an error.
        """

    @abstractmethod
    async def embed_async(self, text: str) -> list[float]:
        """Generate embedding for a single text (async).

        Args:
            text: Input text to embed.

        Returns:
            List of floats representing the embedding vector.

        Raises:
            UpstreamError: If the embedding service is unreachable or returns
                an error.
        """

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts in one call.

        This method batches multiple texts into a single API call for better
        performance. Providers that support batch embeddings (e.g., Ollama)
        can process multiple texts more efficiently than individual calls.

        Args:
            texts: List of input texts to embed.

        Returns:
            List of embedding vectors, one per input text. Each vector is a
            list of floats with the same dimension as single embeddings.

        Raises:
            UpstreamError: If the embedding service is unreachable or returns
                an error.
            ValueError: If texts is empty.

        Example:
            ```python
            vectors = provider.embed_batch(["hello", "world", "test"])
            # Returns: [[0.1, 0.2, ...], [0.3, 0.4, ...], [0.5, 0.6, ...]]
            ```

        Note:
            For providers that don't natively support batch embeddings,
            the default implementation falls back to individual calls.
        """

    @abstractmethod
    async def embed_batch_async(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts in one call (async).

        This is the async version of embed_batch(). Use this method when you
        need non-blocking batch embedding generation.

        Args:
            texts: List of input texts to embed.

        Returns:
            List of embedding vectors, one per input text.

        Raises:
            UpstreamError: If the embedding service is unreachable or returns
                an error.
            ValueError: If texts is empty.

        Example:
            ```python
            vectors = await provider.embed_batch_async(["hello", "world"])
            # Returns: [[0.1, 0.2, ...], [0.3, 0.4, ...]]
            ```
        """

    @property
    @abstractmethod
    def vector_size(self) -> int:
        """Return the dimension of vectors produced by this provider.

        Returns:
            Vector dimension (e.g., 768 for nomic-embed-text, 1536 for
            text-embedding-ada-002).
        """

    def close(self) -> None:  # noqa: B027
        """Close HTTP sessions and cleanup resources.

        This method should be called when the provider is no longer needed
        to properly release HTTP connections and prevent resource leaks.

        Note:
            This is not an abstract method because some providers may not
            need cleanup. Subclasses should override if they maintain resources.
        """
        # Default no-op implementation

    @classmethod
    @abstractmethod
    def from_config(
        cls, config: EmbeddingConfig, session: requests.Session | None = None
    ) -> "EmbeddingProvider":
        """Create provider instance from EmbeddingConfig.

        Each provider class must implement this class method to construct itself
        from the configuration. This allows the factory to instantiate providers
        without knowing provider-specific construction details.

        Args:
            config: Embedding configuration.
            session: Optional requests session for connection pooling.

        Returns:
            Configured EmbeddingProvider instance.

        Raises:
            ValueError: If config is invalid or missing required fields.
        """


# Provider registry: maps provider_type to provider class
# Each provider class must inherit from EmbeddingProvider and implement the interface
_PROVIDER_REGISTRY: dict[str, type[EmbeddingProvider]] = {}


def register_provider(provider_type: str, provider_class: type[EmbeddingProvider]) -> None:
    """Register an embedding provider class.

    This function allows providers to register themselves in the factory registry.
    This makes the factory extensible without needing to modify it for each new provider.

    Args:
        provider_type: Provider type identifier (e.g., "ollama", "openai").
        provider_class: Provider class that inherits from EmbeddingProvider.

    Example:
        ```python
        from clients.interfaces.embedding import EmbeddingProvider, register_provider

        class MyProvider(EmbeddingProvider):
            # ... implement interface ...

        register_provider("myprovider", MyProvider)
        ```
    """
    _PROVIDER_REGISTRY[provider_type] = provider_class


def create_embedding_provider(
    config: EmbeddingConfig,
    session: requests.Session | None = None,
) -> "EmbeddingProvider":
    """Create an embedding provider based on configuration.

    This factory function instantiates the appropriate embedding client based
    on the provider type in the configuration. Providers are registered via
    `register_provider()` and must inherit from `EmbeddingProvider`.

    Args:
        config: Embedding configuration.
        session: Optional requests session for connection pooling (used by
            HTTP-based providers like Ollama).

    Returns:
        EmbeddingProvider instance (OllamaClient, OpenAIClient, etc.).

    Raises:
        ValueError: If provider type is not registered or required config is missing.

    Example:
        ```python
        from clients.interfaces.embedding import EmbeddingConfig, create_embedding_provider

        config = EmbeddingConfig.from_env()
        provider = create_embedding_provider(config)
        vector = provider.embed("hello world")
        ```
    """
    provider_class = _PROVIDER_REGISTRY.get(config.provider_type)
    if provider_class is None:
        msg = (
            f"Provider type '{config.provider_type}' is not registered. "
            f"Available providers: {list(_PROVIDER_REGISTRY.keys())}"
        )
        raise ValueError(
            msg
        )

    # Instantiate provider - each provider class knows how to construct itself from config
    # This delegates the construction logic to the provider class
    # Since from_config is abstract, all registered providers must implement it
    return provider_class.from_config(config, session=session)
