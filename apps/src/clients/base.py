"""Base classes for client wrappers.

This module provides base classes for client wrappers that provide common
functionality through composition with mixins.
"""


from abc import ABC, abstractmethod
from typing import Any

import attrs
import pybreaker

from core.exceptions import UpstreamError
from foundation.circuit_breaker import handle_circuit_breaker_error
from .mixins import CircuitBreakerMixin, ConfigValidationMixin, LoggerMixin


@attrs.define(frozen=False, slots=True)
class VectorDBClientBase(CircuitBreakerMixin, ConfigValidationMixin, LoggerMixin, ABC):
    """Base class for vector database clients.

    This base class provides common functionality for vector database clients:
    - Circuit breaker management (via CircuitBreakerMixin)
    - Config validation (via ConfigValidationMixin)
    - Automatic logging (via LoggerMixin)
    - Template method for batch_upsert with common logic

    Subclasses must implement:
    - _circuit_breaker_config(): Circuit breaker configuration
    - _do_batch_upsert(): Provider-specific batch upsert
    - ensure_collection(): Collection creation (defined in VectorDBProvider interface)
    - search(): Vector search (defined in VectorDBProvider interface)

    Note:
        This class is intended to be used with VectorDBProvider via multiple inheritance.
        The interface contract (ensure_collection, search) is defined in VectorDBProvider.
    """

    async def batch_upsert_async(self, *, collection: str, points: list[Any], vector_size: int) -> None:
        """Batch upsert points into a collection with common error handling.

        This template method handles common logic:
        1. Empty points check (early return)
        2. Circuit breaker state check (fail fast)
        3. Collection existence check/creation
        4. Delegation to provider-specific implementation
        5. Consistent error handling

        Args:
            collection: Collection name to upsert into.
            points: List of points/objects to upsert. Must not be empty.
            vector_size: Vector dimension (e.g., 768 for nomic-embed-text).

        Raises:
            UpstreamError: If vector database operations fail.
        """
        if not points:
            return

        # Check circuit breaker state - if open, fail fast
        if self._breaker.current_state == pybreaker.STATE_OPEN:
            service_name, _, _ = self._circuit_breaker_config()
            handle_circuit_breaker_error(service_name)

        # Ensure collection exists before upserting
        # Note: ensure_collection is defined in VectorDBProvider interface.
        # Concrete classes inherit from both VectorDBClientBase and VectorDBProvider.
        await self.ensure_collection_async(collection=collection, vector_size=vector_size)  # type: ignore[attr-defined]

        # Delegate to provider-specific implementation with error handling
        try:
            await self._do_batch_upsert(collection=collection, points=points)
        except Exception as e:
            service_name, _, _ = self._circuit_breaker_config()
            msg = f"{service_name.capitalize()} batch upsert failed: {e}"
            raise UpstreamError(msg) from e

    @abstractmethod
    async def _do_batch_upsert(self, *, collection: str, points: list[Any]) -> None:
        """Provider-specific batch upsert implementation.

        This method is called by batch_upsert() after common checks and
        collection creation. It should only contain provider-specific logic.

        Args:
            collection: Collection name to upsert into.
            points: List of points/objects to upsert (guaranteed non-empty).

        Raises:
            Exception: Any provider-specific exceptions (will be wrapped by caller).
        """
