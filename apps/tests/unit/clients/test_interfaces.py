"""Unit tests for client interface protocols.

Tests for EmbeddingProvider ABC and ImageEmbeddingProvider protocol.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from clients.interfaces.embedding import (
    EmbeddingProvider,
    ImageEmbeddingProvider,
)

if TYPE_CHECKING:
    import requests

    from config import EmbeddingConfig


class TestImageEmbeddingProviderProtocol:
    """Tests for ImageEmbeddingProvider protocol definition."""

    def test_protocol_is_runtime_checkable(self) -> None:
        """ImageEmbeddingProvider should be a runtime_checkable protocol."""

        # Create a class that implements the protocol
        class MockImageProvider:
            @property
            def vector_size(self) -> int:
                return 512

            def embed_image(self, image_bytes: bytes, text: str | None = None) -> list[float]:
                return [0.1] * 512

            async def embed_image_async(self, image_bytes: bytes, text: str | None = None) -> list[float]:
                return [0.1] * 512

            def embed_image_batch(
                self, images: list[bytes], texts: list[str] | None = None
            ) -> list[list[float]]:
                return [[0.1] * 512 for _ in images]

            async def embed_image_batch_async(
                self, images: list[bytes], texts: list[str] | None = None
            ) -> list[list[float]]:
                return [[0.1] * 512 for _ in images]

        provider = MockImageProvider()
        assert isinstance(provider, ImageEmbeddingProvider)

    def test_incomplete_implementation_not_instance(self) -> None:
        """Classes missing methods should not be considered instances."""

        class IncompleteProvider:
            @property
            def vector_size(self) -> int:
                return 512

            # Missing all embed methods

        provider = IncompleteProvider()
        # Note: Protocol runtime checks only verify method existence,
        # not full signature. For structural typing, this may still pass
        # if the protocol is not fully checked at runtime.
        # However, type checkers will catch this at static analysis time.
        assert not isinstance(provider, ImageEmbeddingProvider)

    def test_protocol_methods_exist(self) -> None:
        """ImageEmbeddingProvider should define required methods."""
        # Check that the protocol defines the expected methods
        protocol_methods = [
            "embed_image",
            "embed_image_async",
            "embed_image_batch",
            "embed_image_batch_async",
        ]
        for method in protocol_methods:
            assert hasattr(ImageEmbeddingProvider, method)

    def test_vector_size_property_exists(self) -> None:
        """ImageEmbeddingProvider should define vector_size property."""
        assert hasattr(ImageEmbeddingProvider, "vector_size")


class TestMockImageEmbeddingProvider:
    """Tests for a mock implementation of ImageEmbeddingProvider."""

    class MockCLIPClient:
        """Mock CLIP client implementing ImageEmbeddingProvider."""

        def __init__(self, vector_size: int = 512) -> None:
            """Initialize mock client with configurable vector size."""
            self._vector_size = vector_size

        @property
        def vector_size(self) -> int:
            """Return the vector dimension."""
            return self._vector_size

        def embed_image(self, image_bytes: bytes, text: str | None = None) -> list[float]:
            """Generate mock embedding for an image."""
            if not image_bytes:
                msg = "Image bytes cannot be empty"
                raise ValueError(msg)
            return [0.1] * self._vector_size

        async def embed_image_async(self, image_bytes: bytes, text: str | None = None) -> list[float]:
            """Generate mock embedding for an image (async)."""
            return self.embed_image(image_bytes, text)

        def embed_image_batch(self, images: list[bytes], texts: list[str] | None = None) -> list[list[float]]:
            """Generate mock embeddings for multiple images."""
            if not images:
                msg = "Images list cannot be empty"
                raise ValueError(msg)
            return [self.embed_image(img) for img in images]

        async def embed_image_batch_async(
            self, images: list[bytes], texts: list[str] | None = None
        ) -> list[list[float]]:
            """Generate mock embeddings for multiple images (async)."""
            return self.embed_image_batch(images, texts)

    @pytest.fixture
    def mock_client(self) -> TestMockImageEmbeddingProvider.MockCLIPClient:
        """Create a mock CLIP client."""
        return self.MockCLIPClient(vector_size=512)

    def test_is_image_embedding_provider(
        self, mock_client: TestMockImageEmbeddingProvider.MockCLIPClient
    ) -> None:
        """MockCLIPClient should be an ImageEmbeddingProvider."""
        assert isinstance(mock_client, ImageEmbeddingProvider)

    def test_vector_size(self, mock_client: TestMockImageEmbeddingProvider.MockCLIPClient) -> None:
        """vector_size should return configured dimension."""
        assert mock_client.vector_size == 512

    def test_vector_size_configurable(self) -> None:
        """vector_size should be configurable at construction."""
        client = self.MockCLIPClient(vector_size=768)
        assert client.vector_size == 768

    def test_embed_image_returns_correct_size(
        self, mock_client: TestMockImageEmbeddingProvider.MockCLIPClient
    ) -> None:
        """embed_image should return vector of correct size."""
        image_bytes = b"\x89PNG\r\n\x1a\n"  # PNG magic bytes
        result = mock_client.embed_image(image_bytes)
        assert len(result) == 512
        assert all(isinstance(x, float) for x in result)

    def test_embed_image_empty_raises(
        self, mock_client: TestMockImageEmbeddingProvider.MockCLIPClient
    ) -> None:
        """embed_image should raise ValueError for empty bytes."""
        with pytest.raises(ValueError, match="empty"):
            mock_client.embed_image(b"")

    @pytest.mark.asyncio
    async def test_embed_image_async_returns_correct_size(
        self, mock_client: TestMockImageEmbeddingProvider.MockCLIPClient
    ) -> None:
        """embed_image_async should return vector of correct size."""
        image_bytes = b"\x89PNG\r\n\x1a\n"
        result = await mock_client.embed_image_async(image_bytes)
        assert len(result) == 512

    def test_embed_image_batch_returns_correct_count(
        self, mock_client: TestMockImageEmbeddingProvider.MockCLIPClient
    ) -> None:
        """embed_image_batch should return one vector per image."""
        images = [b"image1", b"image2", b"image3"]
        results = mock_client.embed_image_batch(images)
        assert len(results) == 3
        assert all(len(v) == 512 for v in results)

    def test_embed_image_batch_empty_raises(
        self, mock_client: TestMockImageEmbeddingProvider.MockCLIPClient
    ) -> None:
        """embed_image_batch should raise ValueError for empty list."""
        with pytest.raises(ValueError, match="empty"):
            mock_client.embed_image_batch([])

    @pytest.mark.asyncio
    async def test_embed_image_batch_async_returns_correct_count(
        self, mock_client: TestMockImageEmbeddingProvider.MockCLIPClient
    ) -> None:
        """embed_image_batch_async should return one vector per image."""
        images = [b"image1", b"image2"]
        results = await mock_client.embed_image_batch_async(images)
        assert len(results) == 2

    def test_embed_image_accepts_optional_text(
        self, mock_client: TestMockImageEmbeddingProvider.MockCLIPClient
    ) -> None:
        """embed_image should accept optional text parameter."""
        image_bytes = b"\x89PNG\r\n\x1a\n"
        # Should work without text
        result1 = mock_client.embed_image(image_bytes)
        # Should work with text (implementation can ignore it)
        result2 = mock_client.embed_image(image_bytes, text="description")
        assert len(result1) == 512
        assert len(result2) == 512

    def test_embed_image_batch_accepts_optional_texts(
        self, mock_client: TestMockImageEmbeddingProvider.MockCLIPClient
    ) -> None:
        """embed_image_batch should accept optional texts parameter."""
        images = [b"image1", b"image2"]
        # Should work without texts
        result1 = mock_client.embed_image_batch(images)
        # Should work with texts (implementation can ignore it)
        result2 = mock_client.embed_image_batch(images, texts=["desc1", "desc2"])
        assert len(result1) == 2
        assert len(result2) == 2


class TestEmbeddingProviderABC:
    """Tests for EmbeddingProvider abstract base class."""

    def test_cannot_instantiate_abc(self) -> None:
        """EmbeddingProvider should not be instantiable directly."""
        with pytest.raises(TypeError, match="abstract"):
            EmbeddingProvider()  # type: ignore[abstract]

    def test_required_abstract_methods(self) -> None:
        """EmbeddingProvider should require implementation of abstract methods."""
        # Verify abstract methods by checking __abstractmethods__
        abstract_methods = EmbeddingProvider.__abstractmethods__
        expected = {
            "embed",
            "embed_async",
            "embed_batch",
            "embed_batch_async",
            "vector_size",
            "from_config",
        }
        assert abstract_methods == expected

    def test_close_has_default_implementation(self) -> None:
        """close() should have a default no-op implementation."""

        class MinimalProvider(EmbeddingProvider):
            def embed(self, text: str) -> list[float]:
                return [0.1]

            async def embed_async(self, text: str) -> list[float]:
                return [0.1]

            def embed_batch(self, texts: list[str]) -> list[list[float]]:
                return [[0.1]]

            async def embed_batch_async(self, texts: list[str]) -> list[list[float]]:
                return [[0.1]]

            @property
            def vector_size(self) -> int:
                return 1

            @classmethod
            def from_config(
                cls,
                config: EmbeddingConfig,
                session: requests.Session | None = None,
            ) -> MinimalProvider:
                return cls()

        provider = MinimalProvider()
        # close() should not raise
        provider.close()
