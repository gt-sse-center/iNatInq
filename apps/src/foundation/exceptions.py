"""Exception classes for foundation utilities.

This module provides exception classes used by foundation components.
"""


class FoundationError(Exception):
    """Base exception class for foundation-related errors."""


class UpstreamError(FoundationError):
    """Exception raised when an upstream dependency service fails.

    This exception indicates that a required external service is unavailable,
    returned an error, or failed to complete a request.
    """

