"""
Unit tests for client interface protocols.

Tests for EmbeddingProvider abstract base class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from clients.interfaces.embedding import (
    EmbeddingProvider,
)

if TYPE_CHECKING:
    from config import EmbeddingConfig


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
            "embed_text",
            "embed_text_batch",
            "embed_image",
            "embed_image_batch",
            "vector_size",
            "from_config",
        }
        assert abstract_methods == expected

    def test_close_has_default_implementation(self) -> None:
        """close() should have a default no-op implementation."""

        class MinimalProvider(EmbeddingProvider):
            async def embed_text(self, text: str) -> list[float]:
                return [0.1]

            async def embed_text_batch(self, texts: list[str]) -> list[list[float]]:
                return [[0.1]]

            async def embed_image(self, image_bytes: bytes, text: str | None = None) -> list[float]:
                return [0.1]

            async def embed_image_batch(
                self, images: list[bytes], texts: list[str] | None = None
            ) -> list[list[float]]:
                return [[0.1]]

            @property
            def vector_size(self) -> int:
                return 1

            @classmethod
            def from_config(
                cls,
                config: EmbeddingConfig,
            ) -> MinimalProvider:
                return cls()

        provider = MinimalProvider()
        # close() should not raise
        assert hasattr(provider, "close") and callable(provider.close)
        provider.close()
