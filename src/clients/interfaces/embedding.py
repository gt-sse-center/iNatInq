"""Embedding provider interface and factory.

This module defines the `EmbeddingProvider` ABC and factory functions for embedding
providers. Configuration classes are in `pipeline.config`.
Concrete implementations live in the parent `clients` package (e.g., `CLIPClient`, `InfinityClient`).
"""

from abc import ABC, abstractmethod
from config import EmbeddingConfig, ProviderType


class EmbeddingProvider(ABC):
    """Abstract base class for embedding generation providers.

    This class defines the interface that all embedding providers must implement.
    Each provider (CLIPClient, InfinityClient, etc.) inherits from this class and
    implements the required methods.

    Note:
        - For image embedding, image bytes should be in a supported format (JPEG, PNG, WebP, GIF)
        - Image preprocessing (resize, normalize) is typically handled by the provider
        - Vector dimensions vary by model (CLIP ViT-B/32: 512, ViT-L/14: 768)
    """

    @abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Input text to embed.

        Returns:
            List of floats representing the embedding vector.

        Raises:
            UpstreamError: If the embedding service is unreachable or returns
                an error.
        """

    @abstractmethod
    async def embed_text_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts in one call (async).

        Args:
            texts: List of input texts to embed.

        Returns:
            List of embedding vectors, one per input text.

        Raises:
            UpstreamError: If the embedding service is unreachable or returns
                an error.
            ValueError: If texts is empty.
        """

    @abstractmethod
    async def embed_image(self, image_bytes: bytes) -> list[float]:
        """Generate embedding for a single image (async).

        Args:
            image_bytes: Bytes containing the image to embed.

        Returns:
            List of floats representing the image embedding vector.

        Raises:
            UpstreamError: If the embedding service is unreachable or returns
                an error.
            ValueError: If image_bytes is empty or not a valid image format.
        """

    @abstractmethod
    async def embed_image_batch(self, images_bytes: list[bytes]) -> list[list[float]]:
        """Generate embeddings for multiple images in one call (async).

        Args:
            images_bytes: List of images in byte format.

        Returns:
            List of embedding vectors, one per input image.

        Raises:
            UpstreamError: If the embedding service is unreachable or returns
                an error.
            ValueError: If images is empty or contains invalid image data, or
                if texts is provided but has a different length than images.
        """

    async def close(self) -> None:  # noqa: B027
        """Close HTTP sessions and cleanup resources.

        This method should be called when the provider is no longer needed
        to properly release HTTP connections and prevent resource leaks.

        Note:
            This is not an abstract method because some providers may not
            need cleanup. Subclasses should override if they maintain resources.
        """
        ...  # noqa: PIE790

    @property
    @abstractmethod
    def vector_size(self) -> int:
        """Return the dimension of vectors produced by this provider.

        Returns:
            Vector dimension (e.g., 768 for nomic-embed-text, 1536 for
            text-embedding-ada-002).
        """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the provider's underlying embedding model.

        Returns:
            Canonical model name used by the provider (e.g., "nomic-embed-text"
            or "clip-vit-large-patch14").
        """

    @classmethod
    @abstractmethod
    def from_config(
        cls,
        config: EmbeddingConfig,
    ) -> "EmbeddingProvider":
        """Create provider instance from EmbeddingConfig.

        Each provider class must implement this class method to construct itself
        from the configuration. This allows the factory to instantiate providers
        without knowing provider-specific construction details.

        Args:
            config: Embedding configuration.

        Returns:
            Configured EmbeddingProvider instance.

        Raises:
            ValueError: If config is invalid or missing required fields.
        """


# Provider registry: maps provider_type to provider class
# Each provider class must inherit from EmbeddingProvider and implement the interface
_PROVIDER_REGISTRY: dict[ProviderType, type[EmbeddingProvider]] = {}


def register_provider(provider_type: ProviderType, provider_class: type[EmbeddingProvider]) -> None:
    """Register an embedding provider class.

    This function allows providers to register themselves in the factory registry.
    This makes the factory extensible without needing to modify it for each new provider.

    Args:
        provider_type: Provider type identifier (e.g., "clip", "infinity").
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
) -> "EmbeddingProvider":
    """Create an embedding provider based on configuration.

    This factory function instantiates the appropriate embedding client based
    on the provider type in the configuration. Providers are registered via
    `register_provider()` and must inherit from `EmbeddingProvider`.

    Args:
        config: Embedding configuration.

    Returns:
        EmbeddingProvider instance (CLIPClient, InfinityClient, etc.).

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
        raise ValueError(msg)

    # Instantiate provider - each provider class knows how to construct itself from config
    # This delegates the construction logic to the provider class
    # Since from_config is abstract, all registered providers must implement it
    return provider_class.from_config(config)
