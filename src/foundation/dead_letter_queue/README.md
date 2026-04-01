# Dead Letter Queue

Captures failed ingestion items for later inspection and retry. Uses a pluggable backend system with Redis as the default implementation.

## Components

| Module | Description |
|--------|-------------|
| `dlq.py` | `DLQ` class — enqueues failed items via a backend |
| `dlq_backend.py` | `DLQBackend` ABC and `StubbedDLQBackend` (no-op fallback) |
| `dlq_backend_registry.py` | Backend registry — resolves backend from `DLQ_BACKEND` env var |
| `dlq_redis_backend.py` | `RedisDLQBackend` — Redis LIST-based implementation |
| `with_dlq.py` | `@with_dlq` decorator — injects a `DLQ` instance into functions |

## Usage

The `@with_dlq` decorator is the primary integration point. It injects a `DLQ` as the first argument:

```python
from foundation.dead_letter_queue.with_dlq import with_dlq
from foundation.dead_letter_queue.dlq import DLQ

@with_dlq
def process_image(dlq: DLQ, image_id: str) -> None:
    try:
        embed_and_store(image_id)
    except Exception as e:
        dlq.enqueue_failed_ingestion(image_id, metadata={"error": str(e)})
        raise
```

The decorator works with both sync and async functions.

## Backend Selection

Set `DLQ_BACKEND` to select the backend:

| Value | Backend | Notes |
|-------|---------|-------|
| `redis` | `RedisDLQBackend` | Requires `DLQ_REDIS_HOST`, `DLQ_REDIS_PORT` |
| *(unset)* | `StubbedDLQBackend` | No-op, logs warnings |

## Redis Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DLQ_REDIS_HOST` | `localhost` | Redis host |
| `DLQ_REDIS_PORT` | `6379` | Redis port |
| `DLQ_REDIS_DATABASE_NUMBER` | `0` | Redis database number |
