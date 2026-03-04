# ruff: noqa: S110, SIM105

"""CLIP client class for generating image and text embeddings.

This module provides a CLIP client class that generates embeddings for images
and text using Ollama's multi-modal models (like LLaVA) or compatible CLIP services.

CLIP's key capability is that both image and text embeddings live in the same
vector space, enabling cross-modal search (e.g., text-to-image search).

## Usage

```python
from clients.clip import CLIPClient

client = CLIPClient(
    base_url="http://ollama:11434",
    model="llava",
    timeout_s=60
)

# Image embedding (for indexing)
with open("image.jpg", "rb") as f:
    image_vector = client.embed_image(f.read())

# Text embedding (for search queries)
text_vector = client.embed_text("a fluffy cat sitting on a couch")

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
from typing import Any, Literal
from pydantic import BaseModel, TypeAdapter, ValidationError
from typing_extensions import override

import aiobreaker
import attrs
import httpx

from clients.interfaces import EmbeddingProvider
from config import EmbeddingConfig, ProviderType
from core.exceptions import UpstreamError
from core.metrics.decorators import with_client_metrics_async
from foundation.circuit_breaker import (
    create_async_circuit_breaker,
    with_circuit_breaker_async,
)
from foundation.image import encode_image_base64
from foundation.retry import HTTPErrorClassifier, async_retry_call, create_retry_logger

from .mixins import CircuitBreakerMixin, ConfigValidationMixin, LoggerMixin

logger = logging.getLogger(__name__)
_retry_logger = logging.getLogger("clients.clip.retry")


# =============================================================================
# CLIP Error Classifier
# =============================================================================


class CLIPErrorClassifier(HTTPErrorClassifier):
    """CLIP-specific error classification for retry logic.

    Classifies httpx exceptions into retriable and non-retriable categories.
    Same pattern as OllamaErrorClassifier (both use httpx).
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


_clip_classifier = CLIPErrorClassifier()
_clip_log_retry = create_retry_logger(
    _retry_logger,
    _clip_classifier.get_error_details,
    "CLIP async operation failed, retrying",
)

# Known CLIP model vector sizes
CLIP_VECTOR_SIZES: dict[str, int] = {
    "clip-vit-base-patch32": 512,
    "clip-vit-base-patch16": 512,
    "clip-vit-large-patch14": 768,
    "openclip-vit-h-14": 1024,
}


class HostedClipRespEntry(BaseModel):
    """Pydantic model to validate Hosted CLIP server response."""

    image_features: list[float]
    text_features: list[float]


HostedClipResponse = TypeAdapter(list[HostedClipRespEntry])


class LocalClipRespEntry(BaseModel):
    """Pydantic model to validate Local CLIP server responses."""

    vector: list[float]


LocalClipResponse = TypeAdapter(list[LocalClipRespEntry])


@attrs.define(frozen=False, slots=True)
class CLIPClient(CircuitBreakerMixin, ConfigValidationMixin, LoggerMixin, EmbeddingProvider):
    """Client for generating image embeddings via CLIP-compatible APIs.

    Attributes:
        base_url: Base URL for the embedding service.
        model: Model name to use for image embedding.
        timeout_s: Request timeout in seconds (default: 120, higher for images).
        circuit_breaker_failure_threshold: Number of consecutive failures before
            circuit opens. Default: 5.
        circuit_breaker_recovery_timeout_s: Seconds to wait before attempting
            recovery after circuit opens. Default: 30.
        max_batch_size: Maximum images per batch request. Image batches are typically
            smaller than text due to memory constraints. Default: 8.
        vector_size_override: Override auto-detected vector size. Use for custom
            models not in the known model map. Default: None (auto-detect).

    Note:
        This class implements the `EmbeddingProvider` abstract base class and can be used
        anywhere that interface is expected.
    """

    # Required parameters
    base_url: str
    model: str

    is_hosted: bool
    clip_api_key: str | None = attrs.field(default=None)

    # Timeout configuration (higher default for images)
    timeout_s: int = attrs.field(default=120)

    # Circuit breaker configuration
    circuit_breaker_failure_threshold: int = attrs.field(default=5)
    circuit_breaker_recovery_timeout_s: int = attrs.field(default=30)

    # Batch configuration (smaller than text due to memory)
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

    @override
    def _circuit_breaker_config(self) -> tuple[str, int, int]:
        """Return circuit breaker configuration for CLIP.

        Uses instance configuration for failure threshold and recovery timeout.

        Returns:
            Tuple of (name, failure_threshold, recovery_timeout).
        """
        return (
            "clip",
            self.circuit_breaker_failure_threshold,
            self.circuit_breaker_recovery_timeout_s,
        )

    def __attrs_post_init__(self) -> None:
        """Initialize the requests session and circuit breakers."""
        # Initialize sync circuit breaker from mixin
        self._init_circuit_breaker()

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
            Vector dimension. Common sizes:
            - LLaVA 7B: 4096
            - CLIP ViT-B/32: 512
            - CLIP ViT-L/14: 768
            - OpenCLIP ViT-H/14: 1024

        Raises:
            ValueError: If model is unknown and no override is set.
        """
        if self.vector_size_override is not None:
            return self.vector_size_override

        # Check known models - try exact match first, then substring
        model_lower = self.model.lower()

        # Exact match
        if model_lower in CLIP_VECTOR_SIZES:
            return CLIP_VECTOR_SIZES[model_lower]

        # Substring match (sorted by length descending for most specific match)
        for known_model, size in sorted(CLIP_VECTOR_SIZES.items(), key=lambda x: len(x[0]), reverse=True):
            if known_model in model_lower:
                return size

        # Fallback to common CLIP size
        logger.warning(
            "Unknown CLIP model '%s', using default vector size 512. Set vector_size_override for accurate dimension.",
            self.model,
        )
        return 512

    @property
    @override
    def model_name(self) -> str:
        return self.model

    def _build_hosted_clip_request_headers(self, *, accept_json: bool = False) -> dict[str, str] | None:
        headers: dict[str, str] = {}
        if accept_json:
            headers["Accept"] = "application/json"
        if self.clip_api_key:
            if self.clip_api_key.lower().startswith("bearer "):
                headers["Authorization"] = self.clip_api_key
            else:
                headers["Authorization"] = f"Bearer {self.clip_api_key}"
        return headers or None

    def _build_hosted_clip_payload(
        self,
        *,
        images: list[str],
        texts: list[str],
    ) -> dict[str, object]:
        if len(images) != len(texts):
            raise ValueError(
                f"Texts list length ({len(texts)}) must match images list length ({len(images)})"
            )
        rows = [[images[i], texts[i]] for i in range(len(images))]
        return {
            "input_data": {
                "columns": ["image", "text"],
                "index": list(range(len(rows))),
                "data": rows,
            }
        }

    async def _request_images(self, images_b64: list[str]) -> list[list[float]]:
        async def do_request() -> list[list[float]]:
            client = await self._get_async_client()

            url = self.base_url if self.is_hosted else f"{self.base_url}/embedding/image"
            payload = (
                self._build_hosted_clip_payload(images=images_b64, texts=[""] * len(images_b64))
                if self.is_hosted
                else {"images": images_b64}
            )
            headers = self._build_hosted_clip_request_headers(accept_json=True) if self.is_hosted else None

            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            if self.is_hosted:
                return self._parse_hosted_clip_response(data, kind="images", count=len(images_b64))
            return self._parse_local_clip_response(data, count=len(images_b64))

        return await async_retry_call(
            do_request,
            max_retries=self.max_retries,
            min_wait=self.retry_min_wait,
            max_wait=self.retry_max_wait,
            is_retriable=_clip_classifier.is_retriable,
            before_sleep=_clip_log_retry,
            operation="CLIP embed_image_async",
        )

    @override
    @with_client_metrics_async("clip", "embed_image")
    @with_circuit_breaker_async("clip")
    async def embed_image(self, image_bytes: bytes) -> list[float]:
        image_b64 = encode_image_base64(image_bytes, include_mime_type=(not self.is_hosted))
        return (await self._request_images([image_b64]))[0]

    @override
    @with_client_metrics_async("clip", "embed_image_batch")
    @with_circuit_breaker_async("clip")
    async def embed_image_batch(self, images_bytes: list[bytes]) -> list[list[float]]:
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
        encoded_images = [
            encode_image_base64(image, include_mime_type=(not self.is_hosted)) for image in images_bytes
        ]

        return await self._request_images(encoded_images)

    def _parse_local_clip_response(self, data: object, *, count: int) -> list[list[float]]:
        try:
            validated_data = LocalClipResponse.validate_python(data)
        except ValidationError as e:
            raise UpstreamError(f"Unexpected response format from local CLIP server: {e}")  # noqa: B904

        vectors = [entry.vector for entry in validated_data]
        if len(vectors) != count:
            raise UpstreamError(f"Local CLIP server returned {len(vectors)} vectors, expected {count}")
        return vectors

    def _parse_hosted_clip_response(
        self, data: object, *, kind: Literal["images", "text"], count: int
    ) -> list[list[float]]:
        try:
            validated_data = HostedClipResponse.validate_python(data)
        except ValidationError as e:
            raise UpstreamError(f"Unexpected response format from hosted CLIP server: {e}")  # noqa: B904
        if kind == "images":
            vectors = [entry.image_features for entry in validated_data]
        elif kind == "text":
            vectors = [entry.text_features for entry in validated_data]
        else:
            raise ValueError(f"Invalid kind value: {kind}")  # pyright: ignore[reportUnreachable]
        if len(vectors) != count:
            raise UpstreamError(f"Hosted CLIP server returned {len(vectors)} vectors, expected {count}")
        return vectors

    async def _request_texts(self, texts: list[str]) -> list[list[float]]:
        async def _do_request() -> list[list[float]]:
            client = await self._get_async_client()
            url = self.base_url if self.is_hosted else f"{self.base_url}/embedding/text"
            payload = (
                self._build_hosted_clip_payload(images=[""] * len(texts), texts=texts)
                if self.is_hosted
                else {"texts": texts}
            )
            headers = self._build_hosted_clip_request_headers(accept_json=True) if self.is_hosted else None
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

            if self.is_hosted:
                return self._parse_hosted_clip_response(data, kind="text", count=len(texts))
            return self._parse_local_clip_response(data, count=len(texts))

        return await async_retry_call(
            _do_request,
            max_retries=self.max_retries,
            min_wait=self.retry_min_wait,
            max_wait=self.retry_max_wait,
            is_retriable=_clip_classifier.is_retriable,
            before_sleep=_clip_log_retry,
            operation="CLIP embed_text_async",
        )

    @override
    @with_client_metrics_async("clip", "embed_text")
    @with_circuit_breaker_async("clip")
    async def embed_text(self, text: str) -> list[float]:
        if not text or not text.strip():
            msg = "Text cannot be empty"
            raise ValueError(msg)
        return (await self._request_texts([text]))[0]

    @override
    @with_client_metrics_async("clip", "embed_text_batch")
    @with_circuit_breaker_async("clip")
    async def embed_text_batch(self, texts: list[str]) -> list[list[float]]:
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

        return await self._request_texts(texts)

    @override
    @classmethod
    def from_config(
        cls,
        config: EmbeddingConfig,
    ) -> "CLIPClient":
        """Create CLIPClient from EmbeddingConfig.

        Args:
            config: Embedding configuration.
            session: Optional requests session for connection pooling.

        Returns:
            Configured CLIPClient instance.

        Raises:
            ValueError: If config is missing required fields.
        """
        if not config.clip_url:
            raise ValueError("clip_url is required in EmbeddingConfig")
        if not config.clip_model:
            raise ValueError("clip_model is required in EmbeddingConfig")

        is_hosted = config.provider_type == ProviderType.HOSTED_CLIP
        if is_hosted and not config.clip_api_key:
            raise ValueError("CLIP_API_KEY is required for hosted_clip backend")

        return cls(
            is_hosted=is_hosted,
            base_url=config.clip_url,
            model=config.clip_model,
            clip_api_key=config.clip_api_key,
            timeout_s=config.clip_timeout,
            circuit_breaker_failure_threshold=config.clip_circuit_breaker_threshold,
            circuit_breaker_recovery_timeout_s=config.clip_circuit_breaker_timeout,
            max_batch_size=config.clip_max_batch_size,
            vector_size_override=config.clip_vector_size,
        )
