# Dead Letter Queue (DLQ) Recovery Flow

Shows the DLQ lifecycle: failure capture during ingestion, storage in Redis, and reprocessing via dedicated API endpoints.

```mermaid
flowchart TB
    subgraph Ingestion["Normal Ingestion (Ray/Databricks)"]
        A[Ray Worker processes<br/>image batch]
        B{Image Processing<br/>Succeeded?}
        C[Upsert vector to Qdrant]
        D[Processing failure<br/>embed / fetch / upsert error]
    end

    subgraph DLQCapture["DLQ Capture (@with_dlq decorator)"]
        E["dlq.enqueue_failed_ingestion(<br/>image_id, metadata={error, s3_key, ...})"]
        F[DLQBackend.insert()]
    end

    subgraph Redis["Redis DLQ Storage"]
        G[("Redis SET<br/>dlq:{collection}:{image_id}<br/>+ metadata JSON")]
    end

    subgraph Recovery["DLQ Recovery (API-triggered)"]
        H["POST /ray/jobs/process-dlq<br/>or POST /databricks/jobs/process-dlq"]
        I[Submit job with<br/>pull_from_dlq=True]
        J[Read failed keys from Redis]
        K[Reprocess failed images<br/>via standard pipeline]
        L{Reprocessing<br/>Succeeded?}
        M[Remove from DLQ<br/>Upsert to Qdrant]
        N[Remains in DLQ<br/>for next recovery run]
    end

    subgraph Monitoring["Observability"]
        O[GET /ray/jobs/{id}/logs<br/>Shows DLQ counts]
        P[Prometheus metrics<br/>dlq_enqueued_total]
    end

    A --> B
    B -->|Yes| C
    B -->|No| D
    D --> E
    E --> F
    F --> G

    H --> I
    I --> J
    J --> G
    G --> K
    K --> L
    L -->|Yes| M
    L -->|No| N

    D -.-> P
    M -.-> P

    style A fill:#e1f5fe
    style C fill:#c8e6c9
    style D fill:#ffcdd2
    style G fill:#e8eaf6
    style M fill:#c8e6c9
    style N fill:#ffcdd2
```

## Components

### `@with_dlq` Decorator

Wraps ingestion task functions to automatically inject a `DLQ` instance:

```python
@with_dlq
def process_image_batch(dlq: DLQ, keys: list[str], ...):
    for key in keys:
        try:
            image = fetch_from_s3(key)
            vector = embed_image(image)
            upsert_to_qdrant(vector)
        except Exception as e:
            dlq.enqueue_failed_ingestion(key, metadata={"error": str(e)})
```

### DLQ Backend (Pluggable)

| Component | File | Role |
|-----------|------|------|
| `DLQ` | `foundation/dead_letter_queue/dlq.py` | Facade — `enqueue_failed_ingestion()` |
| `DLQBackend` | `foundation/dead_letter_queue/dlq_backend.py` | Abstract protocol for backends |
| `DLQRedisBackend` | `foundation/dead_letter_queue/dlq_redis_backend.py` | Redis implementation |
| `StubbedDLQBackend` | `foundation/dead_letter_queue/dlq_backend.py` | No-op fallback when Redis unavailable |
| `get_dlq_backend()` | `foundation/dead_letter_queue/dlq_backend_registry.py` | Auto-discovers backend from env |

### Recovery Endpoints

| Endpoint | Service | Behavior |
|----------|---------|----------|
| `POST /ray/jobs/process-dlq` | `RayService.submit_image_job(pull_from_dlq=True)` | Submits Ray job that reads from DLQ |
| `POST /databricks/jobs/process-dlq` | `DatabricksRayService.submit_image_job(pull_from_dlq=True)` | Submits Databricks job that reads from DLQ |

### Redis Key Schema

```
dlq:{collection}:{image_id} → JSON metadata
```

Metadata includes: error message, S3 key, timestamp, attempt count.
