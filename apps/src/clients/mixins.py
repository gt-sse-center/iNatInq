"""Mixins for client wrapper classes.

This module provides reusable mixins that can be combined with client classes
to add common functionality like circuit breaker support, config validation, and logging.
"""


import logging
from abc import ABC, abstractmethod
from typing import Any

import attrs
import pybreaker

from foundation.circuit_breaker import create_circuit_breaker


@attrs.define(frozen=False, slots=True)
class CircuitBreakerMixin(ABC):
    """Mixin for clients with circuit breaker support.

    This mixin provides automatic circuit breaker initialization and management.
    Subclasses must implement `_circuit_breaker_config()` to specify their
    circuit breaker parameters.

    Example:
        ```python
        @attrs.define(frozen=False, slots=True)
        class MyClient(CircuitBreakerMixin):
            url: str

            def _circuit_breaker_config(self) -> tuple[str, int, int]:
                return ("myclient", 5, 60)

            def __attrs_post_init__(self) -> None:
                self._init_circuit_breaker()
        ```
    """

    _breaker: pybreaker.CircuitBreaker = attrs.field(init=False)

    @abstractmethod
    def _circuit_breaker_config(self) -> tuple[str, int, int]:
        """Return circuit breaker configuration.

        Returns:
            Tuple of (name, failure_threshold, recovery_timeout).
            - name: Service name (e.g., "ollama", "qdrant")
            - failure_threshold: Number of consecutive failures before opening
            - recovery_timeout: Seconds to wait before attempting recovery
        """

    def _init_circuit_breaker(self) -> None:
        """Initialize circuit breaker with configuration from subclass.

        This method should be called in `__attrs_post_init__` after the client
        is fully initialized.
        """
        name, failure_threshold, recovery_timeout = self._circuit_breaker_config()
        object.__setattr__(
            self,
            "_breaker",
            create_circuit_breaker(
                name=name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
            ),
        )


class ConfigValidationMixin:
    """Mixin for clients that support configuration-based instantiation.

    This mixin provides helper methods for validating configuration objects
    in `from_config()` class methods.
    """

    @classmethod
    def _validate_config(
        cls, config: Any, expected_provider: str, required_fields: list[str]
    ) -> None:
        """Validate config has correct provider type and required fields.

        Args:
            config: Configuration object with `provider_type` attribute.
            expected_provider: Expected value of `config.provider_type`.
            required_fields: List of required config attribute names to check.

        Raises:
            ValueError: If provider_type doesn't match or required fields are missing.

        Example:
            ```python
            @classmethod
            def from_config(cls, config: EmbeddingConfig) -> "OllamaClient":
                cls._validate_config(config, "ollama", ["ollama_url", "ollama_model"])
                return cls(base_url=config.ollama_url, model=config.ollama_model)
            ```
        """
        if config.provider_type != expected_provider:
            msg = f"Config provider_type must be '{expected_provider}', got '{config.provider_type}'"
            raise ValueError(
                msg
            )

        missing = [f for f in required_fields if not getattr(config, f, None)]
        if missing:
            msg = f"{expected_provider.capitalize()} provider requires: {', '.join(missing)}"
            raise ValueError(
                msg
            )


class LoggerMixin:
    """Mixin that provides automatic logger creation for client classes.

    This mixin automatically creates a logger based on the class's module name.
    The logger is available as `self._logger` or `cls._logger`.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Automatically create logger for each client subclass.

        Args:
            **kwargs: Additional keyword arguments passed to super().__init_subclass__.
        """
        super().__init_subclass__(**kwargs)
        module = cls.__module__
        cls._logger = logging.getLogger(module)  # type: ignore[attr-defined]

