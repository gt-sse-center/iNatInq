# Tests

Test suite organized into three tiers: unit, integration, and end-to-end.

## Structure

```
tests/
├── unit/              # Fast, mocked tests (~1400 tests)
│   ├── api/           # FastAPI routes and middleware
│   ├── cli/           # CLI command tests (Typer CliRunner)
│   ├── clients/       # Client wrapper tests
│   ├── core/          # Domain logic (benchmark, ingestion)
│   ├── e2e_helpers/   # E2E helper utility tests
│   ├── foundation/    # Foundation layer (DLQ, retry, metrics)
│   └── services/      # Service orchestration tests
├── integration/       # Real infrastructure via testcontainers
│   ├── benchmark/     # Benchmark framework E2E
│   ├── clients/       # S3, Qdrant, CLIP against real containers
│   ├── foundation/    # DLQ Redis backend
│   └── ingestion/     # Ingestion strategy tests
├── e2e/               # Full stack tests (Docker Compose)
│   ├── conftest.py    # Stack lifecycle, fixtures
│   ├── test_search.py # Search API validation
│   ├── test_ingestion.py  # Ingestion pipeline
│   ├── test_resilience.py # Failure and recovery
│   ├── test_metrics.py    # Prometheus metrics
│   ├── test_dlq.py        # Dead letter queue
│   └── test_cli.py        # CLI against live stack
└── conftest.py        # Shared fixtures (PYTHONPATH setup)
```

## Running Tests

```bash
# Unit tests (fast, no Docker required)
uv run pytest tests/unit/ -v

# Integration tests (requires Docker for testcontainers)
uv run pytest tests/integration/ -v

# E2E tests (requires full Docker Compose stack)
uv run pytest tests/e2e/ -v

# Single file
uv run pytest tests/unit/clients/test_clip.py -v

# With coverage
uv run pytest tests/unit/ -v --cov=src --cov-report=term

# Parallel integration tests (each worker gets own containers)
uv run pytest tests/integration/ -v -n auto
```

## Test Tiers

### Unit Tests

- Mock all external dependencies
- Target: 70%+ code coverage (currently ~88%)
- Mirror the `src/` directory structure

### Integration Tests

- Use testcontainers for real MinIO, Qdrant, Redis
- Cover 10 categories: happy path, retry, circuit breaker, rate limiting, timeouts, cleanup, observability
- See `tests/integration/README.md` for details

### E2E Tests

- Run against the full Docker Compose stack
- Test real user workflows end-to-end
- Managed by `tests/e2e/conftest.py` which handles stack lifecycle
