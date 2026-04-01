# Metrics

Prometheus metrics registry for iNatInq. All metrics register on the default `prometheus_client.REGISTRY` and coexist with HTTP metrics from `prometheus-fastapi-instrumentator`.

## Components

| Module | Description |
|--------|-------------|
| `registry.py` | Metric definitions (Histograms, Counters, Gauges) |
| `classify.py` | `classify_error()` — maps exceptions to status labels |
| `decorators.py` | `@with_client_metrics` / `@with_client_metrics_async` |
| `job_metrics_reporter.py` | `JobMetricsReporter` — logs ingestion job summaries |

## Usage

```python
from foundation.metrics import CLIENT_REQUEST_DURATION, classify_error

CLIENT_REQUEST_DURATION.labels(
    client="clip", operation="embed", status="success"
).observe(0.5)
```

Or use the decorator to instrument client methods automatically:

```python
from foundation.metrics import with_client_metrics_async

@with_client_metrics_async(client="qdrant", operation="search")
async def search(self, query_vector, limit):
    ...
```

## Metric Categories

| Category | Metrics | Type |
|----------|---------|------|
| Client | `CLIENT_REQUEST_DURATION`, `CLIENT_REQUEST_TOTAL`, `CLIENT_ERRORS_TOTAL` | Histogram, Counter |
| Circuit breaker | `CIRCUIT_BREAKER_STATE`, `CIRCUIT_BREAKER_TRANSITIONS` | Gauge, Counter |
| Retry | `RETRY_ATTEMPTS_TOTAL`, `RETRY_EXHAUSTIONS_TOTAL` | Counter |
| Search | `SEARCH_EMBEDDING_DURATION`, `SEARCH_VECTOR_QUERY_DURATION`, `SEARCH_RESULT_COUNT` | Histogram |
| Cache | `CACHE_HITS_TOTAL`, `CACHE_MISSES_TOTAL`, `CACHE_SIZE`, etc. | Counter, Gauge, Histogram |
| Ingestion | `INGESTION_DOCS_PROCESSED`, `INGESTION_BATCH_DURATION`, `INGESTION_CHECKPOINT_SAVES` | Counter, Histogram |

All label values are drawn from fixed allow-lists to prevent cardinality explosion.
