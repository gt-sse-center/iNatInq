# Resilience & Error Handling Flow

Shows how the layered resilience stack composes: retry → circuit breaker → DLQ → checkpoint.

```mermaid
flowchart TB
    subgraph Client["Client Call (e.g. embed_text, search_async)"]
        A[Client Method Invoked]
    end

    subgraph CB["Circuit Breaker Layer"]
        B{Circuit<br/>State?}
        C[CLOSED<br/>Normal Operation]
        D[OPEN<br/>Fail Fast]
        E[HALF_OPEN<br/>Test Recovery]
    end

    subgraph Retry["Retry Layer (tenacity)"]
        F[Execute Request]
        G{Success?}
        H{Retriable<br/>Error?}
        I[Exponential Backoff<br/>wait_min → wait_max]
        J{Attempts<br/>Exhausted?}
    end

    subgraph External["External Service"]
        K[(CLIP / Infinity /<br/>Qdrant / S3)]
    end

    subgraph ErrorHandling["Error Propagation"]
        L[UpstreamError<br/>→ 502 Bad Gateway]
        M[BadRequestError<br/>→ 400 Bad Request]
        N[PipelineTimeoutError<br/>→ 504 Gateway Timeout]
    end

    subgraph Recovery["Recovery Mechanisms"]
        O[Dead Letter Queue<br/>Redis Backend]
        P[Checkpoint<br/>Resume from Last Key]
        Q[Prometheus Metrics<br/>retry_attempts, circuit_state]
    end

    A --> B
    B -->|CLOSED| C --> F
    B -->|OPEN| D --> L
    B -->|HALF_OPEN| E --> F

    F --> K
    K --> G
    G -->|Yes| R[Return Result]
    G -->|No| H

    H -->|4xx Client Error| M
    H -->|5xx / Connection Error| I
    I --> J
    J -->|No| F
    J -->|Yes| L

    %% Recovery paths
    L -.->|Ingestion Jobs| O
    O -.->|POST /ray/jobs/process-dlq| F
    L -.->|Batch Processing| P
    P -.->|Next Run Skips Processed Keys| F

    %% Metrics
    F -.-> Q
    D -.-> Q

    %% Circuit breaker state transitions
    G -->|Consecutive Failures >= Threshold| D
    E -->|Success| C
    E -->|Failure| D

    style A fill:#e1f5fe
    style R fill:#c8e6c9
    style D fill:#ffcdd2
    style L fill:#ffcdd2
    style M fill:#fff3e0
    style N fill:#fff3e0
    style O fill:#e8eaf6
    style P fill:#e8eaf6
    style Q fill:#f3e5f5
```

## Resilience Layers (Inside → Outside)

| Layer | Component | Behavior |
|-------|-----------|----------|
| **Circuit Breaker** | `@with_circuit_breaker` / `@with_circuit_breaker_async` | Fails fast when service is degraded. CLOSED → OPEN after N failures, auto-recovers via HALF_OPEN. |
| **Retry** | `RetryWithBackoff` / `async_retry_call` (tenacity) | Exponential backoff for transient errors (5xx, connection). Non-retriable errors (4xx) fail immediately. |
| **Error Classification** | `ErrorClassifier` protocol | Per-client logic determines retriable vs permanent. S3, Qdrant, CLIP each have custom classifiers. |
| **DLQ** | `@with_dlq` decorator → Redis | Captures failed image ingestions for later reprocessing via dedicated API endpoints. |
| **Checkpoint** | `foundation/checkpoint.py` | Tracks processed S3 keys; next run resumes from last checkpoint. |
| **Metrics** | Prometheus counters/gauges | `retry_attempts_total`, `retry_exhaustions_total`, `circuit_breaker_state`, `circuit_breaker_transitions` |

## Circuit Breaker Configuration

| Service | Failure Threshold | Recovery Timeout | Rationale |
|---------|-------------------|-----------------|-----------|
| CLIP / Infinity | 5 | 30s | Critical path, fail fast |
| Qdrant | 3 | 60s | Database issues are serious |
| S3 / MinIO | 5 | 120s | Transient network issues common |

## Error → HTTP Status Mapping (ExceptionHandlerMiddleware)

| Exception | HTTP Status | When |
|-----------|-------------|------|
| `BadRequestError` | 400 | Invalid query, bad parameters |
| `UpstreamError` | 502 | Service failure, retries exhausted, circuit open |
| `PipelineTimeoutError` | 504 | Operation timeout |
| `PipelineError` | 500 | Unexpected application error |
