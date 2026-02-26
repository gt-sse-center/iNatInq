"""Prometheus metric definitions and histogram bucket profiles.

All custom metrics use the ``inatinq_`` prefix to avoid collisions with
HTTP metrics from ``prometheus-fastapi-instrumentator`` (``http_`` prefix)
and default Python collectors (``python_info``, ``python_gc_*``).

Metrics register on the default ``prometheus_client.REGISTRY`` on first
import and coexist with the existing instrumentator endpoint.
"""

from prometheus_client import Counter, Gauge, Histogram

# =============================================================================
# Histogram Bucket Profiles
# =============================================================================

# Fast operations: embedding generation, vector search (millisecond-scale)
FAST_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)

# Slow operations: S3 downloads, ingestion batches (second-scale)
SLOW_BUCKETS = (0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0)

# Result count buckets for search result distributions
RESULT_COUNT_BUCKETS = (0, 1, 5, 10, 25, 50, 100, 250, 500, 1000)

# =============================================================================
# Client Request Metrics Client Request Metrics
# =============================================================================

CLIENT_REQUEST_DURATION = Histogram(
    "inatinq_client_request_duration_seconds",
    "Duration of client requests in seconds",
    labelnames=["client", "operation", "status"],
    buckets=FAST_BUCKETS,
)

CLIENT_REQUEST_TOTAL = Counter(
    "inatinq_client_request_total",
    "Total number of client requests",
    labelnames=["client", "operation", "status"],
)

CLIENT_ERRORS_TOTAL = Counter(
    "inatinq_client_errors_total",
    "Total number of client errors by type",
    labelnames=["client", "operation", "error_type"],
)

# =============================================================================
# Circuit Breaker Metrics Circuit Breaker Metrics
# =============================================================================

CIRCUIT_BREAKER_STATE = Gauge(
    "inatinq_circuit_breaker_state",
    "Current circuit breaker state (0=closed, 1=open, 2=half_open)",
    labelnames=["breaker"],
)

CIRCUIT_BREAKER_TRANSITIONS = Counter(
    "inatinq_circuit_breaker_transitions_total",
    "Total number of circuit breaker state transitions",
    labelnames=["breaker", "from_state", "to_state"],
)

# =============================================================================
# Retry Metrics Retry Metrics
# =============================================================================

RETRY_ATTEMPTS_TOTAL = Counter(
    "inatinq_retry_attempts_total",
    "Total number of retry attempts by outcome",
    labelnames=["client", "operation", "outcome"],
)

RETRY_EXHAUSTIONS_TOTAL = Counter(
    "inatinq_retry_exhaustions_total",
    "Total number of retry exhaustions (all retries failed)",
    labelnames=["client", "operation"],
)

# =============================================================================
# Search Endpoint Metrics Search Endpoint Metrics
# =============================================================================

SEARCH_EMBEDDING_DURATION = Histogram(
    "inatinq_search_embedding_duration_seconds",
    "Duration of embedding generation in seconds",
    labelnames=["provider"],
    buckets=FAST_BUCKETS,
)

SEARCH_VECTOR_QUERY_DURATION = Histogram(
    "inatinq_search_vector_query_duration_seconds",
    "Duration of vector database query in seconds",
    labelnames=["provider", "collection"],
    buckets=FAST_BUCKETS,
)

SEARCH_RESULT_COUNT = Histogram(
    "inatinq_search_result_count",
    "Distribution of search result counts",
    labelnames=["collection"],
    buckets=RESULT_COUNT_BUCKETS,
)

# =============================================================================
# Ingestion Pipeline Metrics Ingestion Pipeline Metrics
# =============================================================================

INGESTION_DOCS_PROCESSED = Counter(
    "inatinq_ingestion_documents_processed_total",
    "Total number of documents processed by ingestion pipeline",
    labelnames=["status", "pipeline"],
)

INGESTION_BATCH_DURATION = Histogram(
    "inatinq_ingestion_batch_duration_seconds",
    "Duration of ingestion batch processing in seconds",
    labelnames=["pipeline"],
    buckets=SLOW_BUCKETS,
)

INGESTION_CHECKPOINT_SAVES = Counter(
    "inatinq_ingestion_checkpoint_saves_total",
    "Total number of checkpoint saves during ingestion",
    labelnames=["pipeline"],
)
