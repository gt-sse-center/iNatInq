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
- Provides sync and async methods for single and batch image/text embedding
- Implements `ImageEmbeddingProvider` protocol
- Handles errors consistently via `UpstreamError`
- Uses circuit breaker pattern for resilience
- Uses attrs for concise, correct class definition
"""

import asyncio
import base64
import logging

import aiobreaker
import attrs
import httpx
import pybreaker
import requests

from config import ImageEmbeddingConfig
from core.exceptions import UpstreamError
from foundation.circuit_breaker import (
    create_async_circuit_breaker,
    with_circuit_breaker,
    with_circuit_breaker_async,
)
from foundation.http import create_retry_session

from .mixins import CircuitBreakerMixin, ConfigValidationMixin, LoggerMixin

logger = logging.getLogger(__name__)

# Known CLIP model vector sizes
CLIP_VECTOR_SIZES: dict[str, int] = {
    "llava": 4096,  # LLaVA uses LLaMA-based embeddings
    "llava:7b": 4096,
    "llava:13b": 5120,
    "llava:34b": 8192,
    "bakllava": 4096,
    "clip-vit-base-patch32": 512,
    "clip-vit-base-patch16": 512,
    "clip-vit-large-patch14": 768,
    "openclip-vit-h-14": 1024,
}


@attrs.define(frozen=False, slots=True)
class CLIPClient(CircuitBreakerMixin, ConfigValidationMixin, LoggerMixin):
    """Client for generating image embeddings via CLIP-compatible APIs.

    This client supports multiple backends:
    - **ollama**: Ollama's multi-modal models (LLaVA, BakLLaVA)
    - **clip**: Dedicated CLIP servers like ai4all/clip (https://hub.docker.com/r/ai4all/clip)

    Attributes:
        base_url: Base URL for the embedding service.
        model: Model name to use for image embedding.
        backend: API backend type. One of "ollama" or "clip". Default: "ollama".
        timeout_s: Request timeout in seconds (default: 120, higher for images).
        circuit_breaker_failure_threshold: Number of consecutive failures before
            circuit opens. Default: 5.
        circuit_breaker_recovery_timeout_s: Seconds to wait before attempting
            recovery after circuit opens. Default: 30.
        max_batch_size: Maximum images per batch request. Image batches are typically
            smaller than text due to memory constraints. Default: 8.
        vector_size_override: Override auto-detected vector size. Use for custom
            models not in the known model map. Default: None (auto-detect).

    Example:
        ```python
        # Using Ollama backend (default)
        client = CLIPClient(
            base_url="http://ollama:11434",
            model="llava"
        )

        # Using ai4all/clip backend
        client = CLIPClient(
            base_url="http://clip-server:8000",
            model="ViT-B/32",
            backend="clip"
        )

        with open("cat.jpg", "rb") as f:
            vector = client.embed_image(f.read())
        ```

    Note:
        This class implements the `ImageEmbeddingProvider` protocol and can be used
        anywhere that protocol is expected.
    """

    # Required parameters
    base_url: str
    model: str

    # Backend type: "ollama" or "clip"
    backend: str = attrs.field(default="ollama")

    # Timeout configuration (higher default for images)
    timeout_s: int = attrs.field(default=120)

    # Circuit breaker configuration
    circuit_breaker_failure_threshold: int = attrs.field(default=5)
    circuit_breaker_recovery_timeout_s: int = attrs.field(default=30)

    # Batch configuration (smaller than text due to memory)
    max_batch_size: int | None = attrs.field(default=8)

    # Vector size configuration
    vector_size_override: int | None = attrs.field(default=None)

    # Private attributes
    _session: requests.Session | None = attrs.field(init=False, default=None)
    _breaker: pybreaker.CircuitBreaker = attrs.field(init=False)
    _async_breaker: aiobreaker.CircuitBreaker = attrs.field(init=False)

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
        if self._session is None:
            self._session = create_retry_session()

        # Initialize sync circuit breaker from mixin
        self._init_circuit_breaker()

        # Initialize async circuit breaker (aiobreaker)
        name, fail_max, timeout = self._circuit_breaker_config()
        object.__setattr__(self, "_async_breaker", create_async_circuit_breaker(name, fail_max, timeout))

    @property
    def session(self) -> requests.Session:
        """Get the requests session, creating one if needed."""
        if self._session is None:
            self._session = create_retry_session()
        return self._session

    @property
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
            "Unknown CLIP model '%s', using default vector size 512. "
            "Set vector_size_override for accurate dimension.",
            self.model,
        )
        return 512

    def _encode_image(self, image_bytes: bytes) -> str:
        """Encode image bytes to base64 string.

        For the 'clip' backend (ai4all/clip), this returns a data URL with
        the `data:image/...;base64,` prefix. For the 'ollama' backend, it
        returns raw base64.

        Args:
            image_bytes: Raw image data.

        Returns:
            Base64-encoded string (with data URL prefix for clip backend).

        Raises:
            ValueError: If image_bytes is empty.
        """
        if not image_bytes:
            msg = "Image bytes cannot be empty"
            raise ValueError(msg)

        b64_data = base64.b64encode(image_bytes).decode("utf-8")

        if self.backend == "clip":
            # ai4all/clip requires data URL format with MIME type prefix
            # Detect image format from magic bytes
            mime_type = self._detect_image_mime_type(image_bytes)
            return f"data:{mime_type};base64,{b64_data}"

        return b64_data

    def _detect_image_mime_type(self, image_bytes: bytes) -> str:
        """Detect MIME type from image magic bytes.

        Args:
            image_bytes: Raw image data.

        Returns:
            MIME type string (e.g., "image/png", "image/jpeg").
        """
        # Check magic bytes for common image formats
        if image_bytes.startswith(b"\x89PNG"):
            return "image/png"
        if image_bytes.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if image_bytes.startswith((b"GIF87a", b"GIF89a")):
            return "image/gif"
        if image_bytes.startswith(b"RIFF") and b"WEBP" in image_bytes[:12]:
            return "image/webp"
        # Default to PNG if unknown
        return "image/png"

    def _make_embed_request(self, image_b64: str, text: str | None = None) -> list[float]:
        """Make synchronous embedding request.

        Supports multiple backends:
        - ollama: Uses /api/embeddings with images array and optional prompt
        - clip: Uses /embed/image with base64 image data (text ignored)

        Args:
            image_b64: Base64-encoded image.
            text: Optional text to embed alongside the image. Used for Ollama
                backend's prompt field. Ignored for clip backend.

        Returns:
            Embedding vector.

        Raises:
            UpstreamError: If request fails.
        """
        try:
            if self.backend == "clip":
                # ai4all/clip API format: POST /embedding/image with {"images": [...]}
                # Returns: list of {image, vector} objects
                # Note: clip backend doesn't support text alongside images
                url = f"{self.base_url}/embedding/image"
                payload = {"images": [image_b64]}
                response = self.session.post(url, json=payload, timeout=self.timeout_s)
                response.raise_for_status()
                data = response.json()

                # ai4all/clip returns a list of {image, vector} objects
                if isinstance(data, list) and len(data) > 0 and "vector" in data[0]:
                    return data[0]["vector"]
                msg = f"Unexpected response format from CLIP server: {data}"
                raise UpstreamError(msg)

            # Ollama API format (default)
            url = f"{self.base_url}/api/embeddings"
            payload = {
                "model": self.model,
                "prompt": text or "",  # Use text if provided, otherwise empty
                "images": [image_b64],
            }
            response = self.session.post(url, json=payload, timeout=self.timeout_s)
            response.raise_for_status()
            data = response.json()

            if "embedding" not in data:
                msg = f"Unexpected response format from Ollama: {data}"
                raise UpstreamError(msg)

            return data["embedding"]

        except requests.RequestException as e:
            msg = f"CLIP embedding request failed: {e}"
            raise UpstreamError(msg) from e

    async def _make_embed_request_async(self, image_b64: str, text: str | None = None) -> list[float]:
        """Make asynchronous embedding request.

        Supports multiple backends:
        - ollama: Uses /api/embeddings with images array and optional prompt
        - clip: Uses /embed/image with base64 image data (text ignored)

        Args:
            image_b64: Base64-encoded image.
            text: Optional text to embed alongside the image. Used for Ollama
                backend's prompt field. Ignored for clip backend.

        Returns:
            Embedding vector.

        Raises:
            UpstreamError: If request fails.
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                if self.backend == "clip":
                    # ai4all/clip API format: POST /embedding/image with {"images": [...]}
                    # Returns: list of {image, vector} objects
                    # Note: clip backend doesn't support text alongside images
                    url = f"{self.base_url}/embedding/image"
                    payload = {"images": [image_b64]}
                    response = await client.post(url, json=payload)
                    response.raise_for_status()
                    data = response.json()

                    if isinstance(data, list) and len(data) > 0 and "vector" in data[0]:
                        return data[0]["vector"]
                    msg = f"Unexpected response format from CLIP server: {data}"
                    raise UpstreamError(msg)

                # Ollama API format (default)
                url = f"{self.base_url}/api/embeddings"
                payload = {
                    "model": self.model,
                    "prompt": text or "",  # Use text if provided, otherwise empty
                    "images": [image_b64],
                }
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()

                if "embedding" not in data:
                    msg = f"Unexpected response format from Ollama: {data}"
                    raise UpstreamError(msg)

                return data["embedding"]

        except httpx.HTTPError as e:
            msg = f"CLIP embedding request failed: {e}"
            raise UpstreamError(msg) from e

    @with_circuit_breaker("clip")
    def embed_image(self, image_bytes: bytes, text: str | None = None) -> list[float]:
        """Generate embedding for a single image.

        Args:
            image_bytes: Raw image bytes (JPEG, PNG, WebP, or GIF format).
            text: Optional text to embed alongside the image. This enables
                multi-modal embeddings where both image and text are encoded
                into the same vector space. Implementations can ignore this
                parameter if not supported. Useful for future filter support
                and combined image+metadata embeddings.

        Returns:
            List of floats representing the image embedding vector.

        Raises:
            UpstreamError: If the embedding service is unreachable or returns an error.
            ValueError: If image_bytes is empty.
        """
        image_b64 = self._encode_image(image_bytes)
        return self._make_embed_request(image_b64, text)

    @with_circuit_breaker_async("clip")
    async def embed_image_async(self, image_bytes: bytes, text: str | None = None) -> list[float]:
        """Generate embedding for a single image (async).

        Args:
            image_bytes: Raw image bytes (JPEG, PNG, WebP, or GIF format).
            text: Optional text to embed alongside the image. This enables
                multi-modal embeddings where both image and text are encoded
                into the same vector space. Implementations can ignore this
                parameter if not supported. Useful for future filter support
                and combined image+metadata embeddings.

        Returns:
            List of floats representing the image embedding vector.

        Raises:
            UpstreamError: If the embedding service is unreachable or returns an error.
            ValueError: If image_bytes is empty.
        """
        image_b64 = self._encode_image(image_bytes)
        return await self._make_embed_request_async(image_b64, text)

    @with_circuit_breaker("clip")
    def embed_image_batch(self, images: list[bytes], texts: list[str] | None = None) -> list[list[float]]:
        """Generate embeddings for multiple images.

        Note: Ollama doesn't natively support batch image embeddings, so this
        method processes images sequentially. For true parallelism, use
        `embed_image_batch_async`.

        Args:
            images: List of raw image bytes to embed.
            texts: Optional list of text strings to embed alongside images.
                If provided, must have the same length as images. Each text
                will be embedded with its corresponding image. Implementations
                can ignore this parameter if not supported. Useful for future
                filter support and combined image+metadata embeddings.

        Returns:
            List of embedding vectors, one per input image.

        Raises:
            UpstreamError: If any embedding request fails.
            ValueError: If images list is empty or contains empty bytes, or
                if texts is provided but has a different length than images.
        """
        if not images:
            msg = "Images list cannot be empty"
            raise ValueError(msg)

        # Validate texts length if provided
        if texts is not None and len(texts) != len(images):
            msg = f"Texts list length ({len(texts)}) must match images list length ({len(images)})"
            raise ValueError(msg)

        # Apply batch size limit
        if self.max_batch_size is not None and len(images) > self.max_batch_size:
            msg = (
                f"Batch size {len(images)} exceeds max_batch_size {self.max_batch_size}. "
                "Split into smaller batches."
            )
            raise ValueError(msg)

        results = []
        for i, image_bytes in enumerate(images):
            image_b64 = self._encode_image(image_bytes)
            text = texts[i] if texts is not None else None
            embedding = self._make_embed_request(image_b64, text)
            results.append(embedding)

        return results

    @with_circuit_breaker_async("clip")
    async def embed_image_batch_async(
        self, images: list[bytes], texts: list[str] | None = None
    ) -> list[list[float]]:
        """Generate embeddings for multiple images (async).

        Uses asyncio.gather for concurrent processing of images.

        Args:
            images: List of raw image bytes to embed.
            texts: Optional list of text strings to embed alongside images.
                If provided, must have the same length as images. Each text
                will be embedded with its corresponding image. Implementations
                can ignore this parameter if not supported. Useful for future
                filter support and combined image+metadata embeddings.

        Returns:
            List of embedding vectors, one per input image.

        Raises:
            UpstreamError: If any embedding request fails.
            ValueError: If images list is empty or contains empty bytes, or
                if texts is provided but has a different length than images.
        """
        if not images:
            msg = "Images list cannot be empty"
            raise ValueError(msg)

        # Validate texts length if provided
        if texts is not None and len(texts) != len(images):
            msg = f"Texts list length ({len(texts)}) must match images list length ({len(images)})"
            raise ValueError(msg)

        # Apply batch size limit
        if self.max_batch_size is not None and len(images) > self.max_batch_size:
            msg = (
                f"Batch size {len(images)} exceeds max_batch_size {self.max_batch_size}. "
                "Split into smaller batches."
            )
            raise ValueError(msg)

        # Encode all images first
        encoded_images = [self._encode_image(img) for img in images]

        # Make concurrent requests with optional text
        tasks = [
            self._make_embed_request_async(img_b64, texts[i] if texts is not None else None)
            for i, img_b64 in enumerate(encoded_images)
        ]
        results = await asyncio.gather(*tasks)

        return list(results)

    # =========================================================================
    # Text Embedding Methods (for cross-modal search)
    # =========================================================================

    def _make_text_embed_request(self, text: str) -> list[float]:
        """Make synchronous text embedding request.

        Supports multiple backends:
        - ollama: Uses /api/embeddings with prompt
        - clip: Uses /embedding/text with texts array

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.

        Raises:
            UpstreamError: If request fails.
        """
        try:
            if self.backend == "clip":
                # ai4all/clip API format: POST /embedding/text with {"texts": [...]}
                # Returns: list of {text, vector} objects
                url = f"{self.base_url}/embedding/text"
                payload = {"texts": [text]}
                response = self.session.post(url, json=payload, timeout=self.timeout_s)
                response.raise_for_status()
                data = response.json()

                if isinstance(data, list) and len(data) > 0 and "vector" in data[0]:
                    return data[0]["vector"]
                msg = f"Unexpected response format from CLIP server: {data}"
                raise UpstreamError(msg)

            # Ollama API format (default)
            url = f"{self.base_url}/api/embeddings"
            payload = {
                "model": self.model,
                "prompt": text,
            }
            response = self.session.post(url, json=payload, timeout=self.timeout_s)
            response.raise_for_status()
            data = response.json()

            if "embedding" not in data:
                msg = f"Unexpected response format from Ollama: {data}"
                raise UpstreamError(msg)

            return data["embedding"]

        except requests.RequestException as e:
            msg = f"CLIP text embedding request failed: {e}"
            raise UpstreamError(msg) from e

    async def _make_text_embed_request_async(self, text: str) -> list[float]:
        """Make asynchronous text embedding request.

        Supports multiple backends:
        - ollama: Uses /api/embeddings with prompt
        - clip: Uses /embedding/text with texts array

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.

        Raises:
            UpstreamError: If request fails.
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                if self.backend == "clip":
                    # ai4all/clip API format: POST /embedding/text with {"texts": [...]}
                    # Returns: list of {text, vector} objects
                    url = f"{self.base_url}/embedding/text"
                    payload = {"texts": [text]}
                    response = await client.post(url, json=payload)
                    response.raise_for_status()
                    data = response.json()

                    if isinstance(data, list) and len(data) > 0 and "vector" in data[0]:
                        return data[0]["vector"]
                    msg = f"Unexpected response format from CLIP server: {data}"
                    raise UpstreamError(msg)

                # Ollama API format (default)
                url = f"{self.base_url}/api/embeddings"
                payload = {
                    "model": self.model,
                    "prompt": text,
                }
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()

                if "embedding" not in data:
                    msg = f"Unexpected response format from Ollama: {data}"
                    raise UpstreamError(msg)

                return data["embedding"]

        except httpx.HTTPError as e:
            msg = f"CLIP text embedding request failed: {e}"
            raise UpstreamError(msg) from e

    @with_circuit_breaker("clip")
    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a text query.

        The resulting vector is in the same space as image embeddings,
        enabling cross-modal similarity search (text-to-image).

        Args:
            text: Text to embed (e.g., "a fluffy cat").

        Returns:
            List of floats representing the text embedding vector.

        Raises:
            UpstreamError: If the embedding service is unreachable or returns an error.
            ValueError: If text is empty.
        """
        if not text or not text.strip():
            msg = "Text cannot be empty"
            raise ValueError(msg)
        return self._make_text_embed_request(text)

    @with_circuit_breaker_async("clip")
    async def embed_text_async(self, text: str) -> list[float]:
        """Generate embedding for a text query (async).

        The resulting vector is in the same space as image embeddings,
        enabling cross-modal similarity search (text-to-image).

        Args:
            text: Text to embed (e.g., "a fluffy cat").

        Returns:
            List of floats representing the text embedding vector.

        Raises:
            UpstreamError: If the embedding service is unreachable or returns an error.
            ValueError: If text is empty.
        """
        if not text or not text.strip():
            msg = "Text cannot be empty"
            raise ValueError(msg)
        return await self._make_text_embed_request_async(text)

    @with_circuit_breaker("clip")
    def embed_text_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple text queries.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors, one per input text.

        Raises:
            UpstreamError: If any embedding request fails.
            ValueError: If texts list is empty or contains empty strings.
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

        results = []
        for text in texts:
            if not text or not text.strip():
                msg = "Text cannot be empty"
                raise ValueError(msg)
            embedding = self._make_text_embed_request(text)
            results.append(embedding)

        return results

    @with_circuit_breaker_async("clip")
    async def embed_text_batch_async(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple text queries (async).

        Uses asyncio.gather for concurrent processing.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors, one per input text.

        Raises:
            UpstreamError: If any embedding request fails.
            ValueError: If texts list is empty or contains empty strings.
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

        # Make concurrent requests
        tasks = [self._make_text_embed_request_async(text) for text in texts]
        results = await asyncio.gather(*tasks)

        return list(results)

    def close(self) -> None:
        """Close HTTP session and cleanup resources."""
        if self._session is not None:
            self._session.close()
            self._session = None

    def set_session(self, session: requests.Session) -> None:
        """Set custom requests session for connection pooling.

        Args:
            session: Requests session to use for HTTP calls.
        """
        if self._session is not None:
            self._session.close()
        self._session = session

    @classmethod
    def from_config(
        cls,
        config: ImageEmbeddingConfig,
        session: requests.Session | None = None,
    ) -> "CLIPClient":
        """Create CLIPClient from ImageEmbeddingConfig.

        Args:
            config: Image embedding configuration.
            session: Optional requests session for connection pooling.

        Returns:
            Configured CLIPClient instance.

        Raises:
            ValueError: If config is missing required fields.
        """
        if not config.clip_url:
            msg = "clip_url is required in ImageEmbeddingConfig"
            raise ValueError(msg)
        if not config.clip_model:
            msg = "clip_model is required in ImageEmbeddingConfig"
            raise ValueError(msg)

        client = cls(
            base_url=config.clip_url,
            model=config.clip_model,
            backend=config.clip_backend,
            timeout_s=config.clip_timeout,
            circuit_breaker_failure_threshold=config.clip_circuit_breaker_threshold,
            circuit_breaker_recovery_timeout_s=config.clip_circuit_breaker_timeout,
            max_batch_size=config.clip_max_batch_size,
            vector_size_override=config.clip_vector_size,
        )

        if session is not None:
            client.set_session(session)

        return client
