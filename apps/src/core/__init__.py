"""Core domain models, services, and shared types for the pipeline package.

This module provides shared domain code used across all layers:
- Exception hierarchy for error handling
- Result classes for structured return values
- Business logic services (following Go codebase pattern)
- Checkpoint utilities for job recovery

Following the Go codebase pattern, services are located in core/services/,
matching the structure where core/domain/ contains services in the Go codebase.
"""

from .exceptions import UpstreamError
from .models import SearchResultItem, SearchResults, VectorPoint

__all__ = [
    "SearchResultItem",
    "SearchResults",
    "UpstreamError",
    "VectorPoint",
]
