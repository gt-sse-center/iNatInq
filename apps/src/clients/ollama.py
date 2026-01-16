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

import aiobreaker
import attrs
import httpx
import pybreaker
import requests

from config import EmbeddingConfig
from core.exceptions import UpstreamError
from foundation.circuit_breaker import (
    create_async_circuit_breaker,
    with_circuit_breaker,
    with_circuit_breaker_async,
)
from foundation.http import create_retry_session

from .interfaces.embedding import EmbeddingProvider
from .mixins import CircuitBreakerMixin, ConfigValidationMixin, LoggerMixin


@attrs.define(frozen=False, slots=True)
class OllamaClient(CircuitBreakerMixin, ConfigValidationMixin, LoggerMixin, EmbeddingProvider):
    """Client for generating text embeddings via Ollama API.

    Attributes:
        base_url: Base URL for the Ollama service (e.g., `http://ollama.ml-system:11434`).
        model: Ollama model name to use for embedding generation (e.g., `nomic-embed-text`).
        timeout_s: Request timeout in seconds (default: 60).
        session: Optional requests.Session for connection pooling and retry logic.
            If not provided, a new session with retry logic will be created.

    Example:
        ```python
        client = OllamaClient(
            base_url="http://ollama.ml-system:11434",
            model="nomic-embed-text"
        )
        vector = client.embed("hello world")
        # Returns: [0.1, 0.2, 0.3, ...]  # 768 floats
        ```

    Note:
        This class is not frozen to allow session reuse and connection pooling.
    """

    base_url: str
    model: str
    timeout_s: int = attrs.field(default=60)
    _session: requests.Session | None = attrs.field(init=False, default=None)
    _breaker: pybreaker.CircuitBreaker = attrs.field(init=False)
    _async_breaker: aiobreaker.CircuitBreaker = attrs.field(init=False)

    def _circuit_breaker_config(self) -> tuple[str, int, int]:
        """Return circuit breaker configuration for Ollama.

        Ollama is on critical path (blocks user requests), so fail fast.

        Returns:
            Tuple of (name, failure_threshold, recovery_timeout).
        """
        return ("ollama", 5, 30)

    def __attrs_post_init__(self) -> None:
        """Initialize the requests session and circuit breakers."""
        if self._session is None:
            self._session = create_retry_session()

        # Initialize sync circuit breaker from base class
        self._init_circuit_breaker()

        # Initialize async circuit breaker (aiobreaker)
        name, fail_max, timeout = self._circuit_breaker_config()
        object.__setattr__(self, "_async_breaker", create_async_circuit_breaker(name, fail_max, timeout))

    @property
    def session(self) -> requests.Session:
        """Get the requests session.

        Creates one if needed.
        """
        if self._session is None:
            self._session = create_retry_session()
        # Type narrowing: we just set _session if it was None
        assert self._session is not None
        return self._session

    def set_session(self, session: requests.Session) -> None:
        """Set a custom requests session for connection pooling.

        Args:
            session: Requests session to use for API calls.
        """
        self._session = session

    @classmethod
    def from_config(cls, config: EmbeddingConfig, session: requests.Session | None = None) -> "OllamaClient":
        """Create OllamaClient from EmbeddingConfig.

        Args:
            config: Embedding configuration with Ollama settings.
            session: Optional requests session for connection pooling.

        Returns:
            Configured OllamaClient instance.

        Raises:
            ValueError: If Ollama config is missing or invalid.
        """
        cls._validate_config(config, "ollama", ["ollama_url", "ollama_model"])

        # Type narrowing: _validate_config ensures these are not None
        assert config.ollama_url is not None
        assert config.ollama_model is not None
        client = cls(base_url=config.ollama_url, model=config.ollama_model)
        if session:
            client.set_session(session)
        return client

    @property
    def vector_size(self) -> int:
        """Return the dimension of vectors produced by this provider.

        Returns:
            Vector dimension. Defaults to 768 for nomic-embed-text, but can
            be overridden for other models.

        Note:
            The vector size is determined by the model. Common values:
            - nomic-embed-text: 768
            - all-minilm: 384
            - Custom models may vary
        """
        # Default to 768 for nomic-embed-text (most common)
        # This could be made configurable or detected from first embedding
        model_vector_sizes: dict[str, int] = {
            "nomic-embed-text": 768,
            "all-minilm": 384,
            "nomic-embed-text-v1.5": 768,
        }
        return model_vector_sizes.get(self.model, 768)

    @with_circuit_breaker("ollama")
    def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for a single text string.

        Args:
            text: Input text string to embed.

        Returns:
            A list of floats representing the embedding vector. The vector dimension
            depends on the model (e.g., `nomic-embed-text` produces 768-dimensional vectors).

        Raises:
            UpstreamError: If Ollama is unreachable, returns an error status code, or
                the response is missing the embedding field. Also raised when circuit
                breaker is open (service unavailable).

        Example:
            ```python
            vector = client.embed("hello world")
            # Returns: [0.1, 0.2, 0.3, ...]  # 768 floats
            ```
        """
        url = f"{self.base_url.rstrip('/')}/api/embeddings"
        try:
            resp = self.session.post(url, json={"model": self.model, "prompt": text}, timeout=self.timeout_s)
        except requests.RequestException as e:
            msg = f"Ollama request failed: {e}"
            raise UpstreamError(msg) from e

        if resp.status_code >= 400:
            msg = f"Ollama error {resp.status_code}: {resp.text}"
            raise UpstreamError(msg)

        data = resp.json()
        emb = data.get("embedding")
        if not isinstance(emb, list) or not emb:
            raise UpstreamError("Ollama response missing embedding")

        return [float(x) for x in emb]

    @with_circuit_breaker("ollama")
    def embed_batch(self, texts: list[str], *, fallback_to_individual: bool = False) -> list[list[float]]:
        """Generate embeddings for multiple texts in one API call.

        Ollama supports batch embeddings via the `input` parameter (not `prompt`).
        This is much more efficient than individual calls.

        Args:
            texts: List of input texts to embed.
            fallback_to_individual: If True, fall back to individual embedding calls
                when the batch API fails (e.g., unsupported Ollama version). If False,
                raise UpstreamError on batch API failure. Defaults to False.

        Returns:
            List of embedding vectors, one per input text. Each vector has the
            same dimension as single embeddings (e.g., 768 for nomic-embed-text).

        Raises:
            UpstreamError: If Ollama is unreachable, returns an error status code,
                or the response is invalid. Also raised when circuit breaker is open.
                When fallback_to_individual is False, also raised on batch API failures.
            ValueError: If texts is empty.

        Example:
            ```python
            vectors = client.embed_batch(["hello world", "foo bar"])
            # Returns: [[0.1, 0.2, ...], [0.3, 0.4, ...]]  # Two 768-d vectors

            # With fallback for older Ollama versions:
            vectors = client.embed_batch(["hello", "world"], fallback_to_individual=True)
            ```

        Note:
            Ollama batch embeddings may degrade in quality at batch sizes >= 16.
            Consider limiting batch size to 8-12 for best results.
        """
        if not texts:
            raise ValueError("texts list cannot be empty")

        # Try batch API first (Ollama 0.3.4+)
        url = f"{self.base_url.rstrip('/')}/api/embed"  # Note: /api/embed not /api/embeddings

        # Scale timeout based on batch size
        batch_timeout = self.timeout_s * max(1, len(texts))

        try:
            resp = self.session.post(
                url,
                json={"model": self.model, "input": texts},  # Use "input" for batch
                timeout=batch_timeout,
            )

            if resp.status_code >= 400:
                # Batch API not supported or failed, fall back to individual calls
                msg = f"Ollama batch API error {resp.status_code}: {resp.text}"
                raise UpstreamError(msg)

            data = resp.json()

            # Ollama batch API returns {"embeddings": [[...], [...]]}
            embeddings = data.get("embeddings")

            if not embeddings or not isinstance(embeddings, list):
                # Batch API not supported, fall back to individual calls
                raise UpstreamError("Ollama response missing embeddings field")

            if len(embeddings) != len(texts):
                msg = f"Ollama returned {len(embeddings)} embeddings for {len(texts)} texts"
                raise UpstreamError(msg)

            return [[float(x) for x in emb] for emb in embeddings]

        except (requests.RequestException, UpstreamError) as e:
            # Fall back to individual calls only if explicitly enabled
            if fallback_to_individual:
                self._logger.warning(  # type: ignore[attr-defined]
                    "Ollama batch embedding failed, falling back to individual calls",
                    extra={"error": str(e), "texts_count": len(texts)},
                )
                # Fall back to individual embedding calls
                return [self.embed(text) for text in texts]
            # Re-raise the error if fallback is not enabled
            raise

    @with_circuit_breaker_async("ollama")
    async def embed_async(self, text: str) -> list[float]:
        """Generate an embedding vector for a single text string (async).

        This is the async version of `embed()` that uses `httpx` for non-blocking
        HTTP requests. Use this method when you need to parallelize multiple
        embedding requests.

        Args:
            text: Input text string to embed.

        Returns:
            A list of floats representing the embedding vector. The vector dimension
            depends on the model (e.g., `nomic-embed-text` produces 768-dimensional vectors).

        Raises:
            UpstreamError: If Ollama is unreachable, returns an error status code, or
                the response is missing the embedding field. Also raised when circuit
                breaker is open.

        Example:
            ```python
            vector = await client.embed_async("hello world")
            # Returns: [0.1, 0.2, 0.3, ...]  # 768 floats
            ```

        Note:
            This method uses `httpx.AsyncClient` for async HTTP requests. For parallel
            embedding generation, use `asyncio.gather()`:
            ```python
            tasks = [client.embed_async(text=t) for t in texts]
            embeddings = await asyncio.gather(*tasks)
            ```
        """
        url = f"{self.base_url.rstrip('/')}/api/embeddings"
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                resp = await client.post(url, json={"model": self.model, "prompt": text})
                resp.raise_for_status()
                data = resp.json()
                emb = data.get("embedding")
                if not isinstance(emb, list) or not emb:
                    raise UpstreamError("Ollama response missing embedding")
                return [float(x) for x in emb]
        except httpx.HTTPStatusError as e:
            msg = f"Ollama error {e.response.status_code}: {e.response.text}"
            raise UpstreamError(msg) from e
        except httpx.RequestError as e:
            msg = f"Ollama request failed: {e}"
            raise UpstreamError(msg) from e

    @with_circuit_breaker_async("ollama")
    async def embed_batch_async(
        self, texts: list[str], *, fallback_to_individual: bool = False
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts in one API call (async).

        Ollama supports batch embeddings via the `input` parameter (not `prompt`).

        Args:
            texts: List of input texts to embed.
            fallback_to_individual: If True, fall back to individual embedding calls
                when the batch API fails (e.g., unsupported Ollama version). If False,
                raise UpstreamError on batch API failure. Defaults to False.

        Returns:
            List of embedding vectors, one per input text.

        Raises:
            UpstreamError: If Ollama is unreachable, returns an error status code,
                or the response is invalid. Also raised when circuit breaker is open.
                When fallback_to_individual is False, also raised on batch API failures.
            ValueError: If texts is empty.

        Example:
            ```python
            vectors = await client.embed_batch_async(["hello", "world"])
            # Returns: [[0.1, 0.2, ...], [0.3, 0.4, ...]]

            # With fallback for older Ollama versions:
            vectors = await client.embed_batch_async(["hello", "world"], fallback_to_individual=True)
            ```
        """
        if not texts:
            raise ValueError("texts list cannot be empty")

        url = f"{self.base_url.rstrip('/')}/api/embed"

        # Scale timeout based on batch size
        batch_timeout = self.timeout_s * max(1, len(texts))

        try:
            async with httpx.AsyncClient(timeout=batch_timeout) as client:
                resp = await client.post(url, json={"model": self.model, "input": texts})
                resp.raise_for_status()
                data = resp.json()

                embeddings = data.get("embeddings")

                if not embeddings or not isinstance(embeddings, list):
                    # Batch API not supported, fall back to individual calls
                    raise UpstreamError("Ollama response missing embeddings field")

                if len(embeddings) != len(texts):
                    msg = f"Ollama returned {len(embeddings)} embeddings for {len(texts)} texts"
                    raise UpstreamError(msg)

                return [[float(x) for x in emb] for emb in embeddings]
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

    async def _embed_async_impl(self, text: str) -> list[float]:
        """Internal async embed implementation without circuit breaker.

        Used by embed_batch_async fallback to avoid double circuit breaker wrapping.
        """
        url = f"{self.base_url.rstrip('/')}/api/embeddings"
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                resp = await client.post(url, json={"model": self.model, "prompt": text})
                resp.raise_for_status()
                data = resp.json()
                emb = data.get("embedding")
                if not isinstance(emb, list) or not emb:
                    raise UpstreamError("Ollama response missing embedding")
                return [float(x) for x in emb]
        except httpx.HTTPStatusError as e:
            msg = f"Ollama error {e.response.status_code}: {e.response.text}"
            raise UpstreamError(msg) from e
        except httpx.RequestError as e:
            msg = f"Ollama request failed: {e}"
            raise UpstreamError(msg) from e

    def close(self) -> None:
        """Close the HTTP session and release resources.

        This method closes the underlying requests.Session to properly
        release HTTP connections and prevent resource leaks.
        """
        if self._session is not None:
            self._session.close()
            self._session = None
