# ruff: noqa: S110, SIM105

"""Ollama client class for generating text embeddings.

This module provides an Ollama client class that encapsulates configuration
and provides methods for embedding generation. This replaces the functional
API with an object-oriented approach using attrs.

## Design

The client class:
- Encapsulates configuration (base_url, model, timeout)
- Provides a clean interface for embedding operations
- Handles errors consistently via `UpstreamError`
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

from config import EmbeddingConfig
from foundation.circuit_breaker import (
    create_async_circuit_breaker,
    with_circuit_breaker_async,
)
from foundation.exceptions import UpstreamError
from foundation.image import encode_image_base64
from foundation.metrics.decorators import with_client_metrics_async
from foundation.retry import HTTPErrorClassifier, async_retry_call, create_retry_logger

from .interfaces.embedding import EmbeddingProvider
from .mixins import ConfigValidationMixin, LoggerMixin

_retry_logger = logging.getLogger("clients.ollama.retry")


# =============================================================================
# Ollama Error Classifier
# =============================================================================


class OllamaTextResponse(BaseModel):
    """Pydantic model for Ollama text embedding response."""

    embeddings: list[list[float]]


class OllamaImageResponse(BaseModel):
    """Pydantic model for Ollama image embedding response."""

    embedding: list[float]


class OllamaErrorClassifier(HTTPErrorClassifier):
    """Ollama-specific error classification for retry logic.

    Classifies httpx exceptions into retriable and non-retriable categories.
    """

    @override
    def is_retriable(self, exc: BaseException) -> bool:
        """Classify whether the exception is retriable."""
        # Connection-level errors are always retriable
        if isinstance(exc, (httpx.ConnectError, httpx.ReadError, httpx.WriteError)):
            return True
        # Timeout errors are retriable
        if isinstance(exc, httpx.TimeoutException):
            return True
        # HTTP status errors: 5xx retriable, 4xx not
        if isinstance(exc, httpx.HTTPStatusError):
            return self.is_retriable_http_status(exc.response.status_code)
        # UpstreamError from inner layers is not retriable at this level
        if isinstance(exc, UpstreamError):
            return False
        return False

    @override
    def get_error_details(self, exc: BaseException) -> dict[str, Any]:
        """Extract structured error details for logging."""
        if isinstance(exc, httpx.HTTPStatusError):
            return {"http_status": exc.response.status_code}
        return {}


_ollama_classifier = OllamaErrorClassifier()
_ollama_log_retry = create_retry_logger(
    _retry_logger,
    _ollama_classifier.get_error_details,
    "Ollama async operation failed, retrying",
)

OLLAMA_SIZES = {
    # LLaVA uses LLaMA-based embeddings
    "llava": 4096,
    "llava:7b": 4096,
    "llava:13b": 5120,
    "llava:34b": 8192,
    "bakllava": 4096,
    # Nomic models
    "nomic-embed-text": 768,
    "nomic-embed-text-v1.5": 768,
    # MiniLM models
    "all-minilm": 384,
    "all-minilm:l6-v2": 384,
    "all-minilm:l12-v2": 384,
    # MixedBread models
    "mxbai-embed-large": 1024,
    # Snowflake models
    "snowflake-arctic-embed": 1024,
    "snowflake-arctic-embed:s": 384,
    "snowflake-arctic-embed:m": 768,
    "snowflake-arctic-embed:l": 1024,
    # Google EmbeddingGemma
    "embeddinggemma": 768,
    "embeddinggemma:2b": 768,
    # Qwen3 Embedding models (default to largest common size)
    "qwen3-embedding": 1024,
    "qwen3-embedding:0.6b": 1024,
    "qwen3-embedding:4b": 1024,
    "qwen3-embedding:8b": 1024,
}

OLLAMA_TEXT_EMBED_ENDPOINT = "/api/embed"
OLLAMA_IMAGE_EMBED_ENDPOINT = "/api/embeddings"


@attrs.define(frozen=False, slots=True)
class OllamaClient(ConfigValidationMixin, LoggerMixin, EmbeddingProvider):
    """Client for generating text embeddings via Ollama API.

    Attributes:
        base_url: Base URL for the Ollama service (e.g., `http://ollama.ml-system:11434`).
        model: Ollama model name to use for embedding generation (e.g., `nomic-embed-text`).
        timeout_s: Request timeout in seconds (default: 60).
        batch_timeout_multiplier: Multiplier for batch timeout calculation. The batch
            timeout is `timeout_s * batch_timeout_multiplier * len(texts)`. Default: 1.0.
        circuit_breaker_failure_threshold: Number of consecutive failures before circuit
            opens. Lower values fail faster (good for critical path). Default: 5.
        circuit_breaker_recovery_timeout_s: Seconds to wait before attempting recovery
            after circuit opens. Default: 30.
        max_batch_size: Maximum texts per batch request. Ollama quality may degrade
            above 16. Set to None for unlimited. Default: 12.
        vector_size_override: Override auto-detected vector size. Use for custom models
            not in the known model map. Default: None (auto-detect).

    Note:
        This class is not frozen to allow session reuse and connection pooling.
    """

    # Required parameters
    base_url: str
    model: str

    # Timeout configuration
    timeout_s: int = attrs.field(default=60)
    batch_timeout_multiplier: float = attrs.field(default=1.0)

    # Circuit breaker configuration
    circuit_breaker_failure_threshold: int = attrs.field(default=5)
    circuit_breaker_recovery_timeout_s: int = attrs.field(default=30)

    # Batch configuration
    max_batch_size: int | None = attrs.field(default=12)

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
        """Return circuit breaker configuration for Ollama.

        Uses instance configuration for failure threshold and recovery timeout.

        Returns:
            Tuple of (name, failure_threshold, recovery_timeout).
        """
        return (
            "ollama",
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
    @with_client_metrics_async("ollama", "embed_text")
    @with_circuit_breaker_async("ollama")
    async def embed_text(self, text: str) -> list[float]:
        return (await self._request_texts([text], "embed_text"))[0]

    @override
    @with_client_metrics_async("ollama", "embed_text_batch")
    @with_circuit_breaker_async("ollama")
    async def embed_text_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            raise ValueError("texts list cannot be empty")
        # Enforce max batch size if configured
        if self.max_batch_size is not None and len(texts) > self.max_batch_size:
            msg = f"Batch size {len(texts)} exceeds max_batch_size {self.max_batch_size}"
            raise ValueError(msg)

        return await self._request_texts(texts, "embed_text_batch")

    @property
    @override
    def vector_size(self) -> int:
        """Return the dimension of vectors produced by this provider.

        Returns:
            Vector dimension. Uses vector_size_override if set, otherwise
            looks up the model in known model sizes, defaulting to 768.

        Note:
            The vector size is determined by the model. Common values:
            - nomic-embed-text: 768
            - all-minilm: 384
            - mxbai-embed-large: 1024
            - Custom models: use vector_size_override
        """
        if self.vector_size_override is not None:
            return self.vector_size_override
        return OLLAMA_SIZES.get(self.model, 768)

    @override
    @with_client_metrics_async("ollama", "embed_image")
    @with_circuit_breaker_async("ollama")
    async def embed_image(self, image_bytes: bytes) -> list[float]:
        # Ollama does not require mime type
        image_b64 = encode_image_base64(image_bytes, include_mime_type=False)
        return (await self._request_images([image_b64], "embed_image"))[0]

    @override
    @with_client_metrics_async("ollama", "embed_image_batch")
    @with_circuit_breaker_async("ollama")
    async def embed_image_batch(self, images_bytes: list[bytes]) -> list[list[float]]:
        if not images_bytes:
            raise ValueError("embed_requests list cannot be empty")
        # Enforce max batch size if configured
        if self.max_batch_size is not None and len(images_bytes) > self.max_batch_size:
            msg = f"Batch size {len(images_bytes)} images exceeds max_batch_size {self.max_batch_size}"
            raise ValueError(msg)

        # Ollama does not require mime type
        images_b64 = [encode_image_base64(image, include_mime_type=False) for image in images_bytes]
        return await self._request_images(images_b64, "embed_image_batch")

    @property
    @override
    def model_name(self) -> str:
        return self.model

    @override
    @classmethod
    def from_config(cls, config: EmbeddingConfig) -> "OllamaClient":
        cls._validate_config(config, "ollama", ["ollama_url", "ollama_model"])

        # Type narrowing: _validate_config ensures these are not None
        assert config.ollama_url is not None
        assert config.ollama_model is not None
        return cls(
            vector_size_override=config.vector_size,
            base_url=config.ollama_url,
            model=config.ollama_model,
            timeout_s=config.ollama_timeout,
            circuit_breaker_failure_threshold=config.ollama_circuit_breaker_threshold,
            circuit_breaker_recovery_timeout_s=config.ollama_circuit_breaker_timeout,
            batch_timeout_multiplier=config.ollama_batch_timeout_multiplier,
            max_batch_size=config.ollama_max_batch_size,
        )

    async def _request_texts(self, texts: list[str], operation: str) -> list[list[float]]:
        url = self.base_url.rstrip("/") + OLLAMA_TEXT_EMBED_ENDPOINT
        payload = {"model": self.model, "input": texts}

        # Scale timeout based on batch size and multiplier
        batch_timeout = self.timeout_s * self.batch_timeout_multiplier * max(1, len(texts))

        async def _do_embed() -> list[list[float]]:
            client = await self._get_async_client()
            resp = await client.post(url, json=payload, timeout=batch_timeout)
            resp.raise_for_status()
            data = resp.json()
            try:
                validated_data = OllamaTextResponse.model_validate(data)
            except ValidationError as e:
                raise UpstreamError(f"Unexpected response format from Ollama text endpoint: {e}")  # noqa: B904
            return validated_data.embeddings

        return await async_retry_call(
            _do_embed,
            max_retries=self.max_retries,
            min_wait=self.retry_min_wait,
            max_wait=self.retry_max_wait,
            is_retriable=_ollama_classifier.is_retriable,
            before_sleep=_ollama_log_retry,
            client="ollama",
            operation=operation,
        )

    async def _request_images(self, images_b64: list[str], operation: str) -> list[list[float]]:
        # Scale timeout based on batch size and multiplier
        batch_timeout = self.timeout_s * self.batch_timeout_multiplier * max(1, len(images_b64))

        async def make_image_request(image: str) -> list[float]:
            # NOTE: this endpoint is weird - it requires "embeddings" as the prompt
            # and only embed a single image at a time
            url = self.base_url.rstrip("/") + OLLAMA_IMAGE_EMBED_ENDPOINT
            payload = {
                "model": self.model,
                "prompt": "embeddings",
                "images": [image],
            }
            client = await self._get_async_client()
            response = await client.post(url, json=payload, timeout=batch_timeout)
            response.raise_for_status()
            data = response.json()
            try:
                validated_data = OllamaImageResponse.model_validate(data)
            except ValidationError as e:
                raise UpstreamError(f"Unexpected response format from Ollama image endpoint: {e}")  # noqa: B904
            return validated_data.embedding

        embeddings: list[list[float]] = []
        for image in images_b64:
            embedding = await async_retry_call(
                make_image_request,
                image,
                max_retries=self.max_retries,
                min_wait=self.retry_min_wait,
                max_wait=self.retry_max_wait,
                is_retriable=_ollama_classifier.is_retriable,
                before_sleep=_ollama_log_retry,
                client="ollama",
                operation=operation,
            )
            embeddings.append(embedding)
        if len(embeddings) != len(images_b64):
            raise UpstreamError(f"Expected {len(images_b64)} embeddings, got {len(embeddings)}")
        return embeddings

    @override
    async def close(self) -> None:
        if self._async_client is not None:
            try:
                await self._async_client.aclose()
                self._async_client = None
                self._async_client_loop = None
            except Exception:
                pass
