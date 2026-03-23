# ruff: noqa: S110, SIM105

"""Infinity client for generating image and text embeddings via SigLIP models.

This module provides an InfinityClient that generates embeddings for images
and text using Infinity's OpenAI-compatible API with SigLIP/SigLIP2 models.

Infinity supports both text and image embeddings in the same vector space via
a `modality` parameter, enabling cross-modal search.

## Usage

```python
from clients.infinity import InfinityClient

client = InfinityClient(
    base_url="http://localhost:7997",
    model="google/siglip-so400m-patch14-384",
    timeout_s=120
)

# Image embedding (for indexing)
with open("image.jpg", "rb") as f:
    image_vector = await client.embed_image(f.read())

# Text embedding (for search queries)
text_vector = await client.embed_text("a fluffy cat sitting on a couch")

# Both vectors are in the same space - can compute similarity!
```

## Design

The client class:
- Encapsulates configuration (base_url, model, timeout)
- Provides async methods for single and batch image/text embedding
- Implements `EmbeddingProvider` protocol
- Handles errors consistently via `UpstreamError`
- Uses circuit breaker pattern for resilience
- Uses attrs for concise, correct class definition
"""

import asyncio
import logging
from typing import Any

import aiobreaker
import attrs
import httpx
from pydantic import BaseModel, ValidationError
from typing_extensions import override

from clients.interfaces import EmbeddingProvider
from config import EmbeddingConfig
from foundation.exceptions import UpstreamError
from foundation.metrics.decorators import with_client_metrics_async
from foundation.circuit_breaker import (
    create_async_circuit_breaker,
    with_circuit_breaker_async,
)
from foundation.image import encode_image_base64
from foundation.retry import HTTPErrorClassifier, async_retry_call, create_retry_logger

from .mixins import ConfigValidationMixin, LoggerMixin

logger = logging.getLogger(__name__)
_retry_logger = logging.getLogger("clients.infinity.retry")


# =============================================================================
# Infinity Error Classifier
# =============================================================================


class InfinityErrorClassifier(HTTPErrorClassifier):
    """Infinity-specific error classification for retry logic.

    Classifies httpx exceptions into retriable and non-retriable categories.
    Same pattern as CLIPErrorClassifier (both use httpx).
    """

    @override
    def is_retriable(self, exc: BaseException) -> bool:
        """Classify whether the exception is retriable."""
        if isinstance(exc, (httpx.ConnectError, httpx.ReadError, httpx.WriteError)):
            return True
        if isinstance(exc, httpx.TimeoutException):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            return self.is_retriable_http_status(exc.response.status_code)
        if isinstance(exc, UpstreamError):
            return False
        return False

    @override
    def get_error_details(self, exc: BaseException) -> dict[str, Any]:
        """Extract structured error details for logging."""
        if isinstance(exc, httpx.HTTPStatusError):
            return {"http_status": exc.response.status_code}
        return {}


_infinity_classifier = InfinityErrorClassifier()
_infinity_log_retry = create_retry_logger(
    _retry_logger,
    _infinity_classifier.get_error_details,
    "Infinity async operation failed, retrying",
)

# Known Infinity (SigLIP) model vector sizes
INFINITY_VECTOR_SIZES: dict[str, int] = {
    "google/siglip-so400m-patch14-384": 1152,
    "google/siglip-base-patch16-224": 768,
    "google/siglip2-so400m-patch14-384": 1152,
    "google/siglip2-base-patch16-224": 768,
}


class InfinityResponseEntry(BaseModel):
    """Pydantic model to validate Infinity server response (OpenAI format)."""

    embedding: list[float]
    index: int


class InfinityResponseData(BaseModel):
    """Pydantic model for Infinity response wrapper."""

    data: list[InfinityResponseEntry]


@attrs.define(frozen=False, slots=True)
class InfinityClient(ConfigValidationMixin, LoggerMixin, EmbeddingProvider):
    """Client for generating image and text embeddings via Infinity (SigLIP models).

    Attributes:
        base_url: Base URL for the Infinity embedding service.
        model: Model name to use for embedding (e.g., google/siglip-so400m-patch14-384).
        timeout_s: Request timeout in seconds (default: 120).
        circuit_breaker_failure_threshold: Number of consecutive failures before
            circuit opens. Default: 5.
        circuit_breaker_recovery_timeout_s: Seconds to wait before attempting
            recovery after circuit opens. Default: 30.
        max_batch_size: Maximum items per batch request. Default: 8.
        vector_size_override: Override auto-detected vector size. Use for custom
            models not in the known model map. Default: None (auto-detect).

    Note:
        This class implements the `EmbeddingProvider` abstract base class and can be used
        anywhere that interface is expected.
    """

    # Required parameters
    base_url: str
    model: str

    # Timeout configuration
    timeout_s: int = attrs.field(default=120)

    # Circuit breaker configuration
    circuit_breaker_failure_threshold: int = attrs.field(default=5)
    circuit_breaker_recovery_timeout_s: int = attrs.field(default=30)

    # Batch configuration
    max_batch_size: int | None = attrs.field(default=8)

    # Vector size configuration
    vector_size_override: int | None = attrs.field(default=None)

    # Async retry configuration
    max_retries: int = attrs.field(default=3)
    retry_min_wait: float = attrs.field(default=1.0)
    retry_max_wait: float = attrs.field(default=10.0)

    # Private attributes
    _async_client: httpx.AsyncClient | None = attrs.field(init=False, default=None)
    _async_client_loop: asyncio.AbstractEventLoop | None = attrs.field(init=False, default=None)
    _async_breaker: aiobreaker.CircuitBreaker = attrs.field(init=False)

    def _circuit_breaker_config(self) -> tuple[str, int, int]:
        """Return circuit breaker configuration for Infinity.

        Uses instance configuration for failure threshold and recovery timeout.

        Returns:
            Tuple of (name, failure_threshold, recovery_timeout).
        """
        return (
            "infinity",
            self.circuit_breaker_failure_threshold,
            self.circuit_breaker_recovery_timeout_s,
        )

    def __attrs_post_init__(self) -> None:
        """Initialize async circuit breaker."""
        # Initialize async circuit breaker (aiobreaker)
        name, fail_max, timeout = self._circuit_breaker_config()
        object.__setattr__(self, "_async_breaker", create_async_circuit_breaker(name, fail_max, timeout))

    async def _get_async_client(self) -> httpx.AsyncClient:
        """Get or create a reusable async HTTP client.

        Returns:
            Shared httpx.AsyncClient instance with connection pooling.
        """
        current_loop = asyncio.get_running_loop()

        needs_recreate = (
            self._async_client is None
            or self._async_client.is_closed
            or self._async_client_loop is None
            or self._async_client_loop.is_closed()
            or self._async_client_loop is not current_loop
        )

        if needs_recreate:
            if self._async_client is not None and not self._async_client.is_closed:
                try:
                    await self._async_client.aclose()
                except Exception:
                    # Old client may be bound to a different/closed loop.
                    pass
            self._async_client = httpx.AsyncClient(
                timeout=self.timeout_s,
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            )
            self._async_client_loop = current_loop
        assert self._async_client is not None
        return self._async_client

    @override
    async def close(self) -> None:
        """Close the async HTTP client and release resources."""
        if self._async_client is not None:
            try:
                await self._async_client.aclose()
            except Exception:
                pass
            self._async_client = None
            self._async_client_loop = None

    @property
    @override
    def vector_size(self) -> int:
        """Return the dimension of vectors produced by this model.

        Returns:
            Vector dimension. Known sizes:
            - SigLIP SO400M: 1152
            - SigLIP2 Base: 768

        Raises:
            ValueError: If model is unknown and no override is set.
        """
        if self.vector_size_override is not None:
            return self.vector_size_override

        # Check known models
        model_lower = self.model.lower()

        # Exact match
        if model_lower in INFINITY_VECTOR_SIZES:
            return INFINITY_VECTOR_SIZES[model_lower]

        # Substring match (sorted by length descending for most specific match)
        for known_model, size in sorted(INFINITY_VECTOR_SIZES.items(), key=lambda x: len(x[0]), reverse=True):
            if known_model in model_lower:
                return size

        # Fallback to SigLIP SO400M default
        logger.warning(
            "Unknown Infinity model '%s', using default vector size 1152. Set vector_size_override for accurate dimension.",
            self.model,
        )
        return 1152

    @property
    @override
    def model_name(self) -> str:
        return self.model

    async def _request_embeddings(
        self,
        *,
        inputs: list[str],
        modality: str,
    ) -> list[list[float]]:
        """Request embeddings from Infinity API.

        Args:
            inputs: List of text strings or base64 data URIs.
            modality: "text" or "image".

        Returns:
            List of embedding vectors.

        Raises:
            UpstreamError: If the API request fails.
        """

        async def do_request() -> list[list[float]]:
            client = await self._get_async_client()

            url = f"{self.base_url}/embeddings"
            payload: dict[str, object] = {
                "model": self.model,
                "input": inputs,
            }
            if modality:
                payload["modality"] = modality

            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

            return self._parse_infinity_response(data, count=len(inputs))

        return await async_retry_call(
            do_request,
            max_retries=self.max_retries,
            min_wait=self.retry_min_wait,
            max_wait=self.retry_max_wait,
            is_retriable=_infinity_classifier.is_retriable,
            before_sleep=_infinity_log_retry,
            operation=f"Infinity embed {modality}",
        )

    def _parse_infinity_response(self, data: object, *, count: int) -> list[list[float]]:
        """Parse Infinity API response.

        Args:
            data: Raw JSON response from Infinity.
            count: Expected number of embeddings.

        Returns:
            List of embedding vectors.

        Raises:
            UpstreamError: If response format is invalid.
        """
        try:
            validated_data = InfinityResponseData.model_validate(data)
        except ValidationError as e:
            raise UpstreamError(f"Unexpected response format from Infinity server: {e}") from e

        embeddings = [entry.embedding for entry in validated_data.data]
        if len(embeddings) != count:
            raise UpstreamError(f"Infinity server returned {len(embeddings)} embeddings, expected {count}")
        return embeddings

    @override
    @with_client_metrics_async("infinity", "embed_image")
    @with_circuit_breaker_async("infinity")
    async def embed_image(self, image_bytes: bytes) -> list[float]:
        """Generate embedding for a single image.

        Args:
            image_bytes: Raw image bytes.

        Returns:
            Embedding vector.

        Raises:
            ValueError: If image_bytes is empty.
            UpstreamError: If the API request fails.
        """
        if not image_bytes:
            msg = "Image bytes cannot be empty"
            raise ValueError(msg)

        image_b64 = encode_image_base64(image_bytes, include_mime_type=True)
        return (await self._request_embeddings(inputs=[image_b64], modality="image"))[0]

    @override
    @with_client_metrics_async("infinity", "embed_image_batch")
    @with_circuit_breaker_async("infinity")
    async def embed_image_batch(self, images_bytes: list[bytes]) -> list[list[float]]:
        """Generate embeddings for multiple images.

        Args:
            images_bytes: List of raw image bytes.

        Returns:
            List of embedding vectors.

        Raises:
            ValueError: If images list is empty or exceeds max_batch_size.
            UpstreamError: If the API request fails.
        """
        if not images_bytes:
            msg = "Images list cannot be empty"
            raise ValueError(msg)

        # Apply batch size limit
        if self.max_batch_size is not None and len(images_bytes) > self.max_batch_size:
            msg = (
                f"Batch size {len(images_bytes)} exceeds max_batch_size {self.max_batch_size}. "
                "Split into smaller batches."
            )
            raise ValueError(msg)

        # Encode all images first
        encoded_images = [encode_image_base64(image, include_mime_type=True) for image in images_bytes]

        return await self._request_embeddings(inputs=encoded_images, modality="image")

    @override
    @with_client_metrics_async("infinity", "embed_text")
    @with_circuit_breaker_async("infinity")
    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Input text.

        Returns:
            Embedding vector.

        Raises:
            ValueError: If text is empty.
            UpstreamError: If the API request fails.
        """
        if not text or not text.strip():
            msg = "Text cannot be empty"
            raise ValueError(msg)
        return (await self._request_embeddings(inputs=[text], modality="text"))[0]

    @override
    @with_client_metrics_async("infinity", "embed_text_batch")
    @with_circuit_breaker_async("infinity")
    async def embed_text_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of input texts.

        Returns:
            List of embedding vectors.

        Raises:
            ValueError: If texts list is empty or exceeds max_batch_size.
            UpstreamError: If the API request fails.
        """
        if not texts:
            msg = "Texts list cannot be empty"
            raise ValueError(msg)

        # Apply batch size limit
        if self.max_batch_size is not None and len(texts) > self.max_batch_size:
            msg = (
                f"Batch size {len(texts)} exceeds max_batch_size {self.max_batch_size}. "
                "Split into smaller batches."
            )
            raise ValueError(msg)

        # Validate all texts first
        for text in texts:
            if not text or not text.strip():
                msg = "Text cannot be empty"
                raise ValueError(msg)

        return await self._request_embeddings(inputs=texts, modality="text")

    @override
    @classmethod
    def from_config(
        cls,
        config: EmbeddingConfig,
    ) -> "InfinityClient":
        """Create InfinityClient from EmbeddingConfig.

        Args:
            config: Embedding configuration.

        Returns:
            Configured InfinityClient instance.

        Raises:
            ValueError: If config is missing required fields.
        """
        if not config.infinity_url:
            raise ValueError("infinity_url is required in EmbeddingConfig")
        if not config.infinity_model:
            raise ValueError("infinity_model is required in EmbeddingConfig")

        return cls(
            base_url=config.infinity_url,
            model=config.infinity_model,
            timeout_s=config.infinity_timeout,
            vector_size_override=config.infinity_vector_size,
        )
