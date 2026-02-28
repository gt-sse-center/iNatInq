"""Exception classes for foundation utilities.

This module provides exception classes used by foundation components.
"""


class FoundationError(Exception):
    """Base exception class for foundation-related errors."""


class UpstreamError(FoundationError):
    """Exception raised when an upstream dependency service fails.

    This exception indicates that a required external service is unavailable,
    returned an error, or failed to complete a request.

    Args:
        message: Human-readable error description.
        error_kind: Optional structured tag for downstream classification
            (e.g. ``"circuit_open"``).  Defaults to ``None``.
    """

    error_kind: str | None

    def __init__(self, message: str, *, error_kind: str | None = None) -> None:
        super().__init__(message)
        self.error_kind = error_kind
