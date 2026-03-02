"""Prometheus metrics registry for iNatInq custom metrics.

This package provides a centralized definition of all Prometheus metrics used
across the iNatInq codebase. Metrics register on the default
``prometheus_client.REGISTRY`` on first import and coexist with the existing
HTTP metrics from ``prometheus-fastapi-instrumentator``.

Usage::

    from core.metrics import CLIENT_REQUEST_DURATION, classify_error

    CLIENT_REQUEST_DURATION.labels(
        client="ollama", operation="embed", status="success"
    ).observe(0.5)

All label values are drawn from fixed allow-lists defined in ADR-0001 to
prevent cardinality explosion.
"""

from core.metrics.classify import classify_error
from core.metrics.registry import (
    CIRCUIT_BREAKER_STATE,
    CIRCUIT_BREAKER_TRANSITIONS,
    CLIENT_ERRORS_TOTAL,
    CLIENT_REQUEST_DURATION,
    CLIENT_REQUEST_TOTAL,
    FAST_BUCKETS,
    INGESTION_BATCH_DURATION,
    INGESTION_CHECKPOINT_SAVES,
    INGESTION_DOCS_PROCESSED,
    RESULT_COUNT_BUCKETS,
    RETRY_ATTEMPTS_TOTAL,
    RETRY_EXHAUSTIONS_TOTAL,
    SEARCH_EMBEDDING_DURATION,
    SEARCH_RESULT_COUNT,
    SEARCH_VECTOR_QUERY_DURATION,
    SLOW_BUCKETS,
)

__all__ = [  # noqa: RUF022 - organized by category, not alphabetically
    # Bucket profiles
    "FAST_BUCKETS",
    "SLOW_BUCKETS",
    "RESULT_COUNT_BUCKETS",
    # Client metrics
    "CLIENT_REQUEST_DURATION",
    "CLIENT_REQUEST_TOTAL",
    "CLIENT_ERRORS_TOTAL",
    # Circuit breaker metrics
    "CIRCUIT_BREAKER_STATE",
    "CIRCUIT_BREAKER_TRANSITIONS",
    # Retry metrics
    "RETRY_ATTEMPTS_TOTAL",
    "RETRY_EXHAUSTIONS_TOTAL",
    # Search metrics
    "SEARCH_EMBEDDING_DURATION",
    "SEARCH_VECTOR_QUERY_DURATION",
    "SEARCH_RESULT_COUNT",
    # Ingestion metrics
    "INGESTION_DOCS_PROCESSED",
    "INGESTION_BATCH_DURATION",
    "INGESTION_CHECKPOINT_SAVES",
    # Helpers
    "classify_error",
]
