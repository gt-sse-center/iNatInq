# ruff: noqa: S110, SIM105

"""Ollama client class for generating text embeddings.

This module provides an Ollama client class that encapsulates configuration
and provides methods for embedding generation. This replaces the functional
API with an object-oriented approach using attrs.

## Usage

```python
from clients.ollama import OllamaClient

client = OllamaClient(
    base_url="http://ollama.ml-system:11434",
    model="nomic-embed-text",
    timeout_s=60
)

vector = client.embed("hello world")
```

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
from typing_extensions import override

from config import EmbeddingConfig
from core.exceptions import UpstreamError
from core.metrics.decorators import with_client_metrics_async
from foundation.circuit_breaker import (
    create_async_circuit_breaker,
    with_circuit_breaker_async,
)
from foundation.retry import HTTPErrorClassifier, async_retry_call, create_retry_logger

from .interfaces.embedding import EmbeddingProvider
from .mixins import CircuitBreakerMixin, ConfigValidationMixin, LoggerMixin

_retry_logger = logging.getLogger("clients.ollama.retry")


# =============================================================================
# Ollama Error Classifier
# =============================================================================


class OllamaErrorClassifier(HTTPErrorClassifier):
    """Ollama-specific error classification for retry logic.

    Classifies httpx exceptions into retriable and non-retriable categories.
    """

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


@attrs.define(frozen=False, slots=True)
class OllamaClient(CircuitBreakerMixin, ConfigValidationMixin, LoggerMixin, EmbeddingProvider):
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

    Example:
        ```python
        # Basic usage
        client = OllamaClient(
            base_url="http://ollama.ml-system:11434",
            model="nomic-embed-text"
        )
        vector = client.embed("hello world")
        # Returns: [0.1, 0.2, 0.3, ...]  # 768 floats

        # With custom configuration
        client = OllamaClient(
            base_url="http://ollama.ml-system:11434",
            model="custom-embed-model",
            timeout_s=120,
            max_batch_size=8,
            vector_size_override=1024,
            circuit_breaker_failure_threshold=3,  # Fail faster
        )
        ```

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
        """Initialize circuit breakers."""
        # Initialize sync circuit breaker from base class
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
        return self._async_client

    async def close_async(self) -> None:
        """Close the async HTTP client and release resources."""
        if self._async_client is not None:
            try:
                await self._async_client.aclose()
            except Exception:
                pass
            self._async_client = None
            self._async_client_loop = None

    @override
    @classmethod
    def from_config(cls, config: EmbeddingConfig) -> "OllamaClient":
        """Create OllamaClient from EmbeddingConfig.

        Args:
            config: Embedding configuration with Ollama settings.

        Returns:
            Configured OllamaClient instance.

        Raises:
            ValueError: If Ollama config is missing or invalid.

        Example:
            ```python
            from config import EmbeddingConfig
            from clients.ollama import OllamaClient

            config = EmbeddingConfig.from_env()
            client = OllamaClient.from_config(config)
            ```
        """
        cls._validate_config(config, "ollama", ["ollama_url", "ollama_model"])

        # Type narrowing: _validate_config ensures these are not None
        assert config.ollama_url is not None
        assert config.ollama_model is not None
        return cls(
            base_url=config.ollama_url,
            model=config.ollama_model,
            timeout_s=getattr(config, "ollama_timeout", 60),
            circuit_breaker_failure_threshold=getattr(config, "ollama_circuit_breaker_threshold", 5),
            circuit_breaker_recovery_timeout_s=getattr(config, "ollama_circuit_breaker_timeout", 30),
            batch_timeout_multiplier=getattr(config, "ollama_batch_timeout_multiplier", 1.0),
            max_batch_size=getattr(config, "ollama_max_batch_size", 12),
        )

    @staticmethod
    def _get_model_vector_sizes() -> dict[str, int]:
        """Return known model vector sizes for auto-detection.

        Note:
            Some models like qwen3-embedding come in multiple sizes with
            different dimensions. Use vector_size_override for variants
            not listed here.
        """
        return {
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
        return self._get_model_vector_sizes().get(self.model, 768)

    async def _embed_async_impl(self, text: str) -> list[float]:
        """Internal async embed implementation without circuit breaker."""
        url = f"{self.base_url.rstrip('/')}/api/embeddings"

        async def _do_embed() -> list[float]:
            client = await self._get_async_client()
            resp = await client.post(url, json={"model": self.model, "prompt": text})
            resp.raise_for_status()
            data = resp.json()
            emb = data.get("embedding")
            if not isinstance(emb, list) or not emb:
                raise UpstreamError("Ollama response missing embedding")
            return [float(x) for x in emb]

        return await async_retry_call(
            _do_embed,
            max_retries=self.max_retries,
            min_wait=self.retry_min_wait,
            max_wait=self.retry_max_wait,
            is_retriable=_ollama_classifier.is_retriable,
            before_sleep=_ollama_log_retry,
            operation="Ollama _embed_async_impl",
        )

    @override
    @with_client_metrics_async("ollama", "embed_text")
    @with_circuit_breaker_async("ollama")
    async def embed_text(self, text: str) -> list[float]:
        return await self._embed_async_impl(text)

    @override
    @with_client_metrics_async("ollama", "embed_text_batch")
    @with_circuit_breaker_async("ollama")
    async def embed_text_batch(
        self, texts: list[str], *, fallback_to_individual: bool = False
    ) -> list[list[float]]:
        if not texts:
            raise ValueError("texts list cannot be empty")

        # Enforce max batch size if configured
        if self.max_batch_size is not None and len(texts) > self.max_batch_size:
            msg = f"Batch size {len(texts)} exceeds max_batch_size {self.max_batch_size}"
            raise ValueError(msg)

        # Try batch API first (Ollama 0.3.4+)
        url = f"{self.base_url.rstrip('/')}/api/embed"  # Note: /api/embed not /api/embeddings

        # Scale timeout based on batch size and multiplier
        batch_timeout = self.timeout_s * self.batch_timeout_multiplier * max(1, len(texts))

        async def _do_batch_embed() -> list[list[float]]:
            client = await self._get_async_client()
            resp = await client.post(url, json={"model": self.model, "input": texts}, timeout=batch_timeout)
            resp.raise_for_status()
            data = resp.json()

            embeddings = data.get("embeddings")

            if not embeddings or not isinstance(embeddings, list):
                raise UpstreamError("Ollama response missing embeddings field")

            if len(embeddings) != len(texts):
                msg = f"Ollama returned {len(embeddings)} embeddings for {len(texts)} texts"
                raise UpstreamError(msg)

            return [[float(x) for x in emb] for emb in embeddings]

        try:
            return await async_retry_call(
                _do_batch_embed,
                max_retries=self.max_retries,
                min_wait=self.retry_min_wait,
                max_wait=self.retry_max_wait,
                is_retriable=_ollama_classifier.is_retriable,
                before_sleep=_ollama_log_retry,
                operation="Ollama embed_batch_async",
            )
        except (httpx.HTTPStatusError, httpx.RequestError, UpstreamError) as e:
            # Fall back to individual async calls only if explicitly enabled
            if fallback_to_individual:
                self._logger.warning(  # type: ignore[attr-defined]
                    "Ollama batch embedding failed, falling back to individual calls",
                    extra={"error": str(e), "texts_count": len(texts)},
                )
                # Fall back to individual async embedding calls
                # Note: embed_async is already protected by circuit breaker
                return await asyncio.gather(*[self._embed_async_impl(text) for text in texts])
            # Re-raise the error if fallback is not enabled
            raise

    @override
    async def embed_image(self, image_bytes: bytes, text: str | None = None) -> list[float]:
        raise NotImplementedError("OllamaClient currently does not support image embedding — use CLIPClient")

    @override
    async def embed_image_batch(
        self, images: list[bytes], texts: list[str] | None = None
    ) -> list[list[float]]:
        raise NotImplementedError("OllamaClient currently does not support image embedding — use CLIPClient")

    @override
    async def close(self) -> None:
        await self.close_async()
