# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

iNatInq is a semantic search and document ingestion service built with FastAPI, Ray, and Qdrant vector database. It provides two core capabilities:

1. **Query Engine**: Semantic search over documents using vector similarity (text-to-text and text-to-image)
2. **Ingestion Engine**: Distributed processing of S3 documents into vector databases using Ray or Databricks

**Tech Stack**: FastAPI · Ray · Spark · CLIP · Qdrant · MinIO

## Common Development Commands

### Testing

```bash
# Unit tests only (fast)
uv run pytest tests/unit/ -v

# Integration tests (requires Docker)
uv run pytest tests/integration/ -v

# All tests
uv run pytest tests/ -v

# With coverage
uv run pytest tests/unit/ -v --cov=src --cov-report=html --cov-report=term

# Run a single test file
uv run pytest tests/unit/clients/test_clip.py -v

# Run a specific test
uv run pytest tests/unit/clients/test_clip.py::test_embed_success -v

# Integration tests in parallel (faster, each worker gets own containers)
uv run pytest tests/integration/ -v -n auto
```

### Linting & Formatting

```bash
# Run linter
uv run ruff check src/ tests/

# Auto-fix issues
uv run ruff check --fix src/ tests/

# Format code
uv run ruff format src/ tests/
```

### Local Development

```bash
# Start all services (Docker Compose)
make docker-up

# Start dev server (without Docker)
make dev
# or: uv run uvicorn api.app:app --reload --host 0.0.0.0 --port 8000

# Check service health
make status

# View logs for specific service
make logs-pipeline
make logs-ray
make logs-clip
```

### End-to-End Testing

```bash
# Generate synthetic images and upload to MinIO
make synthetic-images-setup IMAGE_COUNT=100

# Submit Ray image ingestion job
make ray-image-job-submit IMAGE_PREFIX=images/ IMAGE_COLLECTION=documents

# Check document counts
make count-all COLLECTION=documents

# Search documents
make search-qdrant QUERY="your search query"
```

## Architecture

### High-Level Component Structure

```text
iNatInq/
├── src/
│   ├── api/              # FastAPI routes and middleware
│   ├── clients/          # External service client wrappers
│   ├── core/             # Domain logic and business services
│   ├── foundation/       # Cross-cutting utilities (retry, circuit breaker, logging)
│   └── config.py         # Pydantic settings with env var loading
├── tests/
│   ├── unit/             # Unit tests (mocked dependencies)
│   └── integration/      # Integration tests (testcontainers)
├── syntheticdata/        # Test data generation tools
└── zarf/                 # Infrastructure (Docker, Databricks configs)
```

### Layered Architecture

The codebase follows a strict layered architecture to maintain separation of concerns:

1. **Foundation Layer** (`src/foundation/`)
   - Cross-cutting utilities: retry logic, circuit breakers, structured logging
   - No dependencies on clients, services, or API layer
   - Framework-agnostic, reusable components

2. **Client Layer** (`src/clients/`)
   - Thin wrappers around external services (S3, Qdrant, CLIP, Infinity)
   - Uses Abstract Base Classes (ABCs) for provider abstraction
   - Translates external errors to `UpstreamError`
   - NO business logic - only connection, authentication, and basic operations
   - Factory functions (`create_embedding_provider`, `create_vector_db_provider`) create instances from config

3. **Core/Domain Layer** (`src/core/`)
   - Domain models (`models.py`): `SearchResultItem`, `SearchResults`, `VectorPoint`
   - Business logic services (`core/services/`): orchestrate multiple clients
   - Framework-agnostic - can be used from APIs, CLIs, or tests
   - Ingestion pipelines with Strategy Pattern for environment abstraction

4. **API Layer** (`src/api/`)
   - FastAPI routes and request/response models
   - Minimal logic - delegates to services
   - Middleware for logging, error handling, metrics

### Key Architectural Patterns

#### Provider Abstraction (ABCs)

Clients use Abstract Base Classes to allow swapping providers without changing service code:

- **`EmbeddingProvider`** ABC: Implemented by `CLIPClient`, `InfinityClient`
- **`VectorDBProvider`** ABC: Implemented by `QdrantClientWrapper`

Services depend on ABCs, not concrete implementations:

```python
from clients.interfaces.embedding import create_embedding_provider, EmbeddingConfig
from clients.interfaces.vector_db import create_vector_db_provider, VectorDBConfig

# Factory functions return providers matching the configured type
embedding_provider = create_embedding_provider(EmbeddingConfig.from_env())
vector_db_provider = create_vector_db_provider(VectorDBConfig.from_env())
```

#### Strategy Pattern for Ingestion

The ingestion pipeline uses Strategy Pattern to abstract Ray cluster lifecycle across environments:

- **`ClusterStrategy`** protocol: Defines init/shutdown/runtime_env interface
- **`LocalRayStrategy`**: Local Ray cluster initialization
- **`DatabricksStrategy`**: Databricks Spark cluster with Ray integration
- **`IngestionPipeline`**: Unified orchestrator that delegates cluster management to strategies

This eliminates code duplication between Ray and Databricks entrypoints (~90% overlap previously).

#### Configuration Management

All configuration uses Pydantic Settings with environment variable loading:

- **`EmbeddingConfig`**: Provider-agnostic embedding configuration
- **`VectorDBConfig`**: Provider-agnostic vector database configuration
- **`MinIOConfig`**: S3/MinIO configuration
- **`RayJobConfig`**: Ray job execution parameters
- **`DatabricksRayJobConfig`**: Databricks job parameters

Configuration is loaded once per process via `@lru_cache`:

```python
from config import get_settings

settings = get_settings()  # Cached, safe to call multiple times
```

Auto-detects environment (Kubernetes vs local) and sets appropriate service URLs.

### Async vs Sync

- **Vector DB clients**: Async-only (uses `AsyncQdrantClient`)
- **Embedding clients**: Sync with optional async support (`embed_async`)
- **Ray tasks**: Use `asyncio.run()` to call async clients from Ray workers
- **FastAPI routes**: Use `async def` with `await` for async clients

### Resilience Features

The codebase implements defense-in-depth resilience:

1. **Retry with Exponential Backoff** (`foundation/retry.py`)
   - Transient failures (connection errors, 5xx) trigger retries
   - Non-retriable errors (4xx) fail fast
   - Configurable via `RetryWithBackoff` class

2. **Circuit Breaker** (via `aiobreaker`)
   - Opens after threshold failures to prevent cascading failures
   - Automatically recovers after timeout
   - Used in all client wrappers

3. **Rate Limiting** (`core/ingestion/ray/rate_limiter.py`)
   - Ray Actor-based rate limiter for embedding API
   - Prevents overwhelming upstream services

4. **Checkpointing** (`core/ingestion/checkpoint.py`)
   - Tracks processed S3 keys to enable resumption
   - Saves progress periodically during ingestion jobs

5. **Connection Pooling**
   - boto3 and vector DB clients handle pooling internally

## Testing Strategy

### Unit Tests (`tests/unit/`)

- Mock all external dependencies
- Test business logic in isolation
- Fast execution (<1 second per test)
- 91% code coverage target

### Integration Tests (`tests/integration/`)

- Use testcontainers to spin up real services (MinIO, Qdrant)
- Test actual network calls, error handling, resilience features
- Session-scoped fixtures for efficiency
- Cover 10 categories: happy path, retry success, retry exhaustion, non-retriable errors, circuit breaker open/recovery, rate limiting, timeout handling, resource cleanup, observability

### Running Tests

When making changes:

1. Run unit tests for the affected module first (fast feedback)
2. Run integration tests if changing client wrappers or network code
3. Use `pytest -k <pattern>` to run specific tests during development
4. Run full test suite before committing

## Important Development Patterns

### Adding a New Client

1. Create ABC in `src/clients/interfaces/` if it's a new provider type
2. Implement concrete client in `src/clients/` (inherit from ABC)
3. Add factory function to create client from config
4. Raise `UpstreamError` for external service failures
5. Add unit tests in `tests/unit/clients/`
6. Add integration tests in `tests/integration/clients/` with testcontainer
7. Update `README.md` in clients package

### Adding a New API Endpoint

1. Define request/response models in `src/api/models.py` (Pydantic)
2. Create route handler in appropriate router file (e.g., `src/api/routers/search.py`)
3. Delegate to service in `src/core/services/`
4. Service orchestrates multiple clients
5. Add tests in `tests/unit/api/` and `tests/integration/api/`

### Adding Configuration

1. Add environment variable to appropriate config class in `src/config.py`
2. Update docstring with variable name, description, and default
3. Use `os.getenv()` in `from_env()` class method
4. Consider auto-detection for environment-specific defaults (in-cluster vs local)

### Working with Ray Jobs

Ray ingestion jobs follow this pattern:

1. **Entrypoint**: `src/core/ingestion/ray/process_s3_to_vector_dbs.py` or `process_s3_images.py`
2. **Strategy**: Cluster lifecycle managed by `ClusterStrategy` implementations
3. **Pipeline**: `IngestionPipeline` orchestrates list → checkpoint → batch → process → collect
4. **Task Function**: `@ray.remote` decorated functions in `processing.py` or `image_processing.py`
5. **Checkpointing**: Periodic saves of processed keys for resumption

When modifying Ray jobs, ensure:

- Task functions remain stateless (no shared mutable state)
- Use `asyncio.run()` for async client calls within tasks
- Progress logging every N keys (configurable via `RAY_PROGRESS_LOG_INTERVAL`)
- Checkpoint saves don't block processing

### Logging

- Use structured logging via `foundation/logger.py`
- Log format is JSON with context fields (request, response, error)
- Access logs: `logger = logging.getLogger("app.access")`
- Error logs: `logger = logging.getLogger("app.error")`
- Include contextual information in `extra` parameter

## Environment Variables

Key variables to know (see `src/config.py` for full list):

- **`EMBEDDING_PROVIDER`**: `clip`, `hosted_clip`, `infinity` (default: `clip`)
- **`VECTOR_DB_PROVIDER`**: `qdrant` (default: `qdrant`)
- **`QDRANT_URL`**: Auto-detected based on environment
- **`S3_ENDPOINT`**: Auto-detected based on environment
- **`RAY_S3_BATCH_SIZE`**: Keys per Ray task (default: `50`)
- **`RAY_CHECKPOINT_ENABLED`**: Enable checkpointing (default: `true`)

Environment detection uses:

1. `PIPELINE_ENV` explicit override (`cluster` or `local`)
2. Kubernetes service account token presence
3. `KUBERNETES_SERVICE_HOST` environment variable

## Code Style

- Line length: 110 characters
- Use `ruff` for linting and formatting (configured in `pyproject.toml`)
- Type hints required for function arguments and return values
- Docstrings use Google style
- Frozen attrs dataclasses for immutable models
- Async functions use `async def` and `await`

## Databricks Integration

Databricks jobs use Ray on Spark clusters:

- Configuration: `zarf/databricks/dev/.env.local` (gitignored)
- Cluster spec: `zarf/databricks/dev/inatinq-azure-databricks-cluster.json`
- Entry point: `src/core/ingestion/databricks/run_ingest.py`
- Uses `DatabricksStrategy` for cluster lifecycle management

See `zarf/databricks/README.md` for setup details.

## Common Gotchas

1. **Async clients in Ray tasks**: Use `asyncio.run()` to call async methods from sync Ray tasks
2. **Environment detection**: Service URLs auto-resolve based on in-cluster detection, check logs if endpoints are wrong
3. **Circuit breaker state**: If tests fail with "Circuit breaker is open", reset the breaker or wait for timeout
4. **Testcontainers ports**: Always use `container.get_exposed_port()`, never hardcode ports
5. **Provider factories**: Use `create_embedding_provider()` and `create_vector_db_provider()` instead of instantiating clients directly - this ensures provider abstraction works correctly
