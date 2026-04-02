# Developer Guide

Getting started with iNatInq: local setup, CLI, testing, ingestion, API, and benchmarking.

---

## Table of Contents

1. [Local Development Environment](#1-local-development-environment)
2. [CLI Reference](#2-cli-reference)
3. [Running Tests](#3-running-tests)
4. [Running the Ingestion Engine](#4-running-the-ingestion-engine)
5. [FastAPI Endpoints and Postman](#5-fastapi-endpoints-and-postman)
6. [Benchmarking](#6-benchmarking)
7. [Configuration System](#7-configuration-system)
8. [Architecture Overview](#8-architecture-overview)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Local Development Environment

### Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | >= 3.11 | `brew install python@3.11` or [python.org](https://www.python.org/) |
| uv | latest | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Docker Desktop | latest | [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/) |
| Git | latest | `brew install git` |

Docker Desktop must be configured with sufficient resources for the full stack. The Docker Compose file targets ~10 CPUs / ~14.6 GiB RAM.

### Clone and Install

```bash
git clone https://github.com/gt-sse-center/iNatInq.git
cd iNatInq

# Install all dependencies (creates .venv automatically)
uv sync --all-packages
```

The project uses [uv workspaces](https://docs.astral.sh/uv/concepts/projects/workspaces/). The top-level `pyproject.toml` manages dependencies, and `uv sync --all-packages` installs everything needed for development, testing, and the CLI.

To install a specific sub-project only:

```bash
uv sync --package <sub-project>
```

### Python Version

The `.python-version` file pins Python 3.11. uv will respect this automatically.

### Environment Configuration

The project uses environment variables for configuration. For local development, the defaults work out of the box with Docker Compose.

For optional overrides (e.g., Qdrant Cloud, custom CLIP endpoints):

```bash
cp zarf/compose/dev/env.local.example zarf/compose/dev/.env.local
# Edit .env.local with your values
```

Key environment variables (see `src/config.py` for the full list):

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_PROVIDER` | `clip` | Embedding provider: `clip`, `hosted_clip`, `infinity` |
| `VECTOR_DB_PROVIDER` | `qdrant` | Vector database provider |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant HTTP endpoint |
| `S3_ENDPOINT` | `http://localhost:9000` | MinIO/S3 endpoint |
| `PIPELINE_ENV` | `local` | Selects Docker Compose mode |

### Pre-commit Hooks

The repo includes pre-commit hooks that run on every commit:

1. `uv sync --frozen --all-packages` — ensures lockfile is in sync
2. `ruff format --check` — checks code formatting
3. `ruff check` — runs linter

Install them:

```bash
uv run pre-commit install
```

### Start the Stack

```bash
# Start all Docker Compose services
uv run inq up

# Verify everything is healthy
uv run inq status
```

This starts: MinIO, Qdrant, CLIP, Ray (head + worker), Redis, and the Pipeline API.

### Service Endpoints

Once the stack is running:

| Service | URL | Notes |
|---------|-----|-------|
| Pipeline API | http://localhost:8000 | FastAPI application |
| Swagger UI | http://localhost:8000/docs | Interactive API docs |
| MinIO Console | http://localhost:9001 | Login: `minioadmin` / `minioadmin` |
| Qdrant Dashboard | http://localhost:6333/dashboard | Vector DB UI |
| Ray Dashboard | http://localhost:8265 | Ray cluster UI |
| CLIP API | http://localhost:8001 | Embedding service |

### IDE Setup

The `src/` directory is the Python source root. Configure your IDE's `PYTHONPATH` to include `src/` for imports to resolve correctly (e.g., `from clients.interfaces.embedding import ...`).

For VS Code, the `uv sync --all-packages` command ensures all type stubs and dependencies are available.

### Code Style

- **Line length**: 110 characters
- **Formatter/Linter**: `ruff` (configured in `pyproject.toml`)
- **Type checker**: `basedpyright` (recommended mode)
- **Docstrings**: Google style
- **Models**: Frozen `attrs` dataclasses

```bash
# Format code
uv run ruff format src/ tests/

# Lint (with auto-fix)
uv run ruff check --fix src/ tests/

# Type check
uv run inq dev typecheck
```

---

## 2. CLI Reference

The project ships a Typer-based CLI registered as `inq`. All commands are invoked via `uv run inq <command>`.

### Top-Level Convenience Commands

```bash
uv run inq up        # Start all services (alias for inq docker up)
uv run inq down      # Stop all services (alias for inq docker down)
uv run inq status    # Health check (alias for inq docker health)
uv run inq --version # Show CLI version
```

### Command Groups

#### `inq docker` — Docker Compose Operations

```bash
inq docker up              # Start all services
inq docker down            # Stop all services
inq docker logs            # Tail all service logs
inq docker log <service>   # Tail logs for one service (pipeline|qdrant|ray-head|minio|clip|redis)
inq docker ps              # Container status
inq docker health          # Health check all services
inq docker build-base      # Build pipeline base image (heavy deps)
inq docker build           # Build pipeline image
inq docker rebuild         # Rebuild and restart pipeline
inq docker restart         # Restart all services
inq docker clean           # Stop services and remove volumes
inq docker shell <service> # Interactive shell in container
inq docker scale -w 3      # Scale Ray workers to 3 replicas
inq docker lazy            # Open lazydocker TUI
```

#### `inq dev` — Development Tasks

```bash
inq dev lint               # Run ruff check
inq dev format             # Run ruff format + ruff check --fix
inq dev typecheck          # Run mypy
inq dev validate-config    # Validate YAML config files
inq dev serve              # Start uvicorn dev server (port 8000, hot-reload)
```

#### `inq test` — Test Execution

```bash
inq test unit              # Unit tests (tests/unit/)
inq test integration       # Integration tests (requires Docker)
inq test integration -p    # Integration tests in parallel
inq test e2e               # E2E tests (requires full stack)
inq test all               # All tests
inq test cov               # Unit tests with coverage
inq test cov --all         # All tests with coverage

# Pass extra pytest args after --
inq test unit -- -k test_healthz
inq test unit -- --no-header -q
```

#### `inq synthetic` — Synthetic Data Generation

```bash
inq synthetic generate -c 100 -s 512   # Generate 100 synthetic images (512px)
inq synthetic upload                    # Upload generated images to MinIO
inq synthetic setup -c 100             # Generate + upload in one step
inq synthetic clean                    # Remove generated images
```

#### `inq search` — Semantic Search

```bash
inq search images -q "red circle" -l 5         # Text-to-image search
inq search demo -q "red circle"                 # Search with presigned URLs
inq search download -q "red circle" -o ./out    # Download matching images
inq search open -q "red circle"                 # Search and open top result in browser
```

#### `inq ray` — Ray Job Management

```bash
inq ray submit -p images/ -c documents   # Submit image ingestion job
inq ray status                           # Show Ray job statuses
inq ray logs                             # Show latest job logs
```

#### `inq vectordb` — Vector Database and E2E

```bash
inq vectordb count -c documents          # Count docs in collection
inq vectordb clear -c documents          # Delete collection
inq vectordb s3-count -b pipeline -p images/   # Count MinIO objects
inq vectordb s3-clear -b pipeline -p images/   # Clear MinIO objects
inq vectordb e2e -c 100                  # Full E2E: generate, upload, ingest, search
```

#### `inq smoke` — Smoke Tests

```bash
inq smoke health      # Provider health checks
inq smoke providers   # Full smoke test (embed -> upsert -> search)
inq smoke all         # Both
```

#### `inq ui` — Open Web Dashboards

```bash
inq ui all        # Open all UIs
inq ui pipeline   # Open Swagger docs
inq ui minio      # Open MinIO console
inq ui qdrant     # Open Qdrant dashboard
inq ui ray        # Open Ray dashboard
```

#### `inq bench` — Benchmarking

```bash
inq bench validate bench/datasets/sample/sample-gold.json   # Validate dataset
inq bench metrics                                            # List available metrics
inq bench run --dataset gold.json --provider qdrant          # Run single benchmark
inq bench compare --dataset gold.json --provider qdrant --provider clip  # Compare providers
inq bench quantization --dataset d.json --collection c1 --collection c2 --collection c3
```

#### `inq databricks` — Azure Databricks

```bash
inq databricks build          # Create/update cluster
inq databricks up             # Start cluster
inq databricks down           # Terminate cluster
inq databricks configure-s3a  # Configure MinIO S3A access
inq databricks cdc-notebooks  # Upload and run CDC notebooks
```

---

## 3. Running Tests

### Test Structure

```
tests/
├── conftest.py          # Shared fixtures (Ray mock/real switching)
├── unit/                # ~1400 tests, fast, no Docker needed
│   ├── api/             # FastAPI routes and middleware
│   ├── cli/             # CLI command tests (Typer CliRunner)
│   ├── clients/         # Client wrapper tests
│   ├── core/            # Domain logic (benchmark, ingestion)
│   ├── e2e_helpers/     # E2E helper utility tests
│   ├── foundation/      # Foundation layer (DLQ, retry, metrics)
│   └── services/        # Service orchestration tests
├── integration/         # Real services via testcontainers
│   ├── benchmark/       # Benchmark framework tests
│   ├── clients/         # S3, Qdrant, CLIP against real containers
│   ├── foundation/      # DLQ Redis backend
│   └── ingestion/       # Ingestion strategy tests
└── e2e/                 # Full Docker Compose stack
    ├── conftest.py      # Stack lifecycle management
    ├── test_search.py   # Search API validation
    ├── test_ingestion.py    # Ingestion pipeline
    ├── test_resilience.py   # Failure and recovery
    ├── test_metrics.py      # Prometheus metrics
    ├── test_dlq.py          # Dead letter queue
    └── test_cli.py          # CLI against live stack
```

### Unit Tests

Unit tests mock all external dependencies and run without Docker. They are fast (<1s per test).

```bash
# Run all unit tests
uv run inq test unit

# Run a specific test file
uv run pytest tests/unit/clients/test_clip.py -v

# Run a specific test
uv run pytest tests/unit/clients/test_clip.py::test_embed_success -v

# Run with pattern matching
uv run inq test unit -- -k "test_healthz"
```

### Integration Tests

Integration tests use [testcontainers](https://testcontainers-python.readthedocs.io/) to spin up real MinIO, Qdrant, and Redis containers. **Docker must be running.**

```bash
# Run all integration tests
uv run inq test integration

# Run in parallel (each worker gets its own containers)
uv run inq test integration -p
# or: uv run pytest tests/integration/ -v -n auto
```

Integration tests cover 10 categories:
1. Happy path
2. Retry success
3. Retry exhaustion
4. Non-retriable errors
5. Circuit breaker open
6. Circuit breaker recovery
7. Rate limiting
8. Timeout handling
9. Resource cleanup
10. Observability

### E2E Tests

E2E tests require the full Docker Compose stack to be running. The E2E conftest will detect if the stack is already up; if not, it will start it.

```bash
# Start the stack first (or let conftest handle it)
uv run inq up

# Run E2E tests
uv run inq test e2e
# or: uv run pytest tests/e2e/ -v
```

E2E tests validate real user workflows: search, ingestion, resilience under failure, Prometheus metrics accuracy, and CLI commands against the live stack.

### Coverage

```bash
# Unit tests with coverage (HTML + terminal + XML reports)
uv run inq test cov

# All tests with coverage
uv run inq test cov --all

# Direct pytest with coverage
uv run pytest tests/unit/ -v --cov=src --cov-report=html --cov-report=term
```

Coverage reports:
- **Terminal**: printed to stdout
- **HTML**: `htmlcov/index.html`
- **XML**: `coverage.xml`
- **Minimum threshold**: 70% (configured in `pyproject.toml`)

### Pytest Configuration

Key pytest settings from `pyproject.toml`:

- `asyncio_mode = "auto"` — async tests run without explicit `@pytest.mark.asyncio`
- `norecursedirs = ["tests/e2e"]` — E2E tests are excluded from default collection
- Custom markers: `e2e`, `integration`, `slow`

### Ray Mocking

The root `tests/conftest.py` automatically swaps between mock and real Ray depending on the test directory:
- `tests/unit/` → mock Ray (no cluster needed)
- `tests/integration/` and `tests/e2e/` → real Ray module

---

## 4. Running the Ingestion Engine

The ingestion engine processes images from S3 (MinIO) into the Qdrant vector database using Ray for distributed processing.

### Step 1: Start the Stack

```bash
uv run inq up
uv run inq status   # Verify all services are healthy
```

### Step 2: Seed MinIO with Synthetic Data

Generate synthetic images with semantic content (shapes + colors) and upload them to MinIO:

```bash
# Generate 100 images (512px) and upload to MinIO in one command
uv run inq synthetic setup --count 100

# Or step by step:
uv run inq synthetic generate --count 100 --size 512
uv run inq synthetic upload --prefix images/
```

Generated images follow the naming convention `{color}-{shape}-{background}-{index}.png` (e.g., `red-circle-gradient-042.png`) and are stored locally in `bench/synthetic/data/imgs/` before upload.

Image properties:
- 8 colors: red, green, blue, yellow, orange, purple, pink, teal
- 3 shapes: circle, square, triangle
- 2 backgrounds: solid, gradient

Verify the upload:

```bash
uv run inq vectordb s3-count --bucket pipeline --prefix images/
```

You can also browse MinIO directly at http://localhost:9001 (login: `minioadmin` / `minioadmin`).

### Step 3: Submit a Ray Ingestion Job

```bash
# Submit via CLI
uv run inq ray submit --prefix images/ --collection documents

# Or via the API directly
curl -X POST http://localhost:8000/ray/jobs/images \
  -H "Content-Type: application/json" \
  -d '{"s3_prefix": "images/", "s3_bucket": "pipeline", "collection": "documents"}'
```

### Step 4: Monitor the Job

```bash
# Check job status
uv run inq ray status

# View job logs
uv run inq ray logs

# Or use the Ray Dashboard at http://localhost:8265
```

### Step 5: Verify Results

```bash
# Check document count in Qdrant
uv run inq vectordb count --collection documents

# Run a semantic search
uv run inq search images --query "red circle" --limit 5
```

### One-Command E2E

Run the entire pipeline (generate, upload, ingest, wait, verify, search) in one command:

```bash
uv run inq vectordb e2e --count 100 --wait 60
```

### Checkpointing

The ingestion pipeline supports checkpointing for resumability. If a job fails mid-way, resubmitting the same job will skip already-processed keys. Controlled by:

- `RAY_CHECKPOINT_ENABLED=true` (default)
- Checkpoints can be stored locally or on S3

### Scaling Workers

```bash
# Scale Ray workers for higher throughput
uv run inq docker scale --workers 3
```

---

## 5. FastAPI Endpoints and Postman

### Starting the API

The API starts automatically with `uv run inq up`. For development with hot-reload (without Docker):

```bash
uv run inq dev serve
# or: uv run uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

### API Documentation

- **Swagger UI**: http://localhost:8000/docs (interactive, with "Try It Out")
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

### Endpoint Summary

#### Health

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/healthz` | Liveness probe (no dependency checks) |

#### Search

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/search/images` | Text-to-image semantic search |

Query parameters: `q` (required), `limit` (default: 10), `collection`, `provider`, `model`, `image_provider`

Example:
```bash
curl "http://localhost:8000/search/images?q=red+circle&limit=5"
```

#### Ray Jobs

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/ray/jobs/images` | Submit image ingestion job |
| `POST` | `/ray/jobs/process-dlq` | Submit DLQ processing job |
| `GET` | `/ray/jobs/{job_id}` | Get job status |
| `GET` | `/ray/jobs/{job_id}/logs` | Get job logs |
| `DELETE` | `/ray/jobs/{job_id}` | Stop job |

#### Databricks Jobs

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/databricks/jobs/images` | Submit image ingestion job |
| `POST` | `/databricks/jobs/cdc-producer` | Submit CDC producer job |
| `POST` | `/databricks/jobs/cdc-consumer` | Submit CDC consumer job |
| `POST` | `/databricks/jobs/process-dlq` | Submit DLQ processing job |
| `GET` | `/databricks/jobs/{run_id}` | Get run status |
| `GET` | `/databricks/jobs/{run_id}/logs` | Get run output |
| `DELETE` | `/databricks/jobs/{run_id}` | Stop run |

#### Other

| Method | Path | Description |
|--------|------|-------------|
| `DELETE` | `/cache` | Invalidate semantic cache |
| `POST` | `/ingestion/metrics` | Record ingestion metrics |
| `GET` | `/metrics` | Prometheus metrics endpoint |

### Postman Collections

The `postman/` directory contains pre-built Postman collections:

| File | Description |
|------|-------------|
| `iNatInq-Pipeline-API.postman_collection.json` | Full API collection with all endpoints |
| `iNatInq-Local.postman_environment.json` | Local development environment variables |

#### Import into Postman

1. Open Postman
2. Click **Import** (top-left)
3. Drag both JSON files or click "Upload Files"
4. Select the **iNatInq Local** environment from the top-right dropdown

#### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `base_url` | `http://localhost:8000` | API base URL |
| `collection` | `documents` | Vector DB collection |
| `s3_prefix` | `inputs/` | S3 prefix for jobs |
| `job_id` | (auto-set) | Ray job ID (populated on submit) |
| `run_id` | (auto-set) | Databricks run ID (populated on submit) |

#### Workflow

1. Start services: `uv run inq up`
2. Run **Health Check** to verify connectivity
3. **Submit a job** — the response auto-saves `job_id` / `run_id` via test scripts
4. **Check status** — uses the saved job identifier

### Error Responses

All errors return JSON:

```json
{
  "error": "BadRequestError",
  "message": "Query parameter 'q' cannot be empty"
}
```

| Status | Exception | Meaning |
|--------|-----------|---------|
| 400 | `BadRequestError` | Invalid request |
| 404 | — | Collection or resource not found |
| 502 | `UpstreamError` | External service failure (CLIP, Qdrant) |
| 504 | `PipelineTimeoutError` | Upstream timeout |
| 500 | `PipelineError` | Internal error |

---

## 6. Benchmarking

The benchmark framework measures search quality (precision, recall, NDCG) and latency against vector database providers.

### Datasets

Datasets live in `bench/datasets/` and follow a JSON schema (`bench/datasets/schemas/gold-standard.schema.json`).

| Dataset | Path | Description |
|---------|------|-------------|
| **Sample Gold** | `bench/datasets/sample/sample-gold.json` | 10 iNaturalist-style queries with graded relevance. Good for quick smoke tests. |
| **INQUIRE Val** | `bench/datasets/inquire/inquire-val.json` | Validation split from the INQUIRE benchmark (iNaturalist image queries). |
| **INQUIRE Val Subset** | `bench/datasets/inquire/inquire-val-subset.json` | Smaller subset of validation split for faster iteration. |
| **INQUIRE Val Bench 20k** | `bench/datasets/inquire/inquire-val-bench20k.json` | Validation split scoped to a 20k image collection. |
| **INQUIRE Test** | `bench/datasets/inquire/inquire-test.json` | Full test split (~27k lines). |
| **INQUIRE Test Subset** | `bench/datasets/inquire/inquire-test-subset.json` | Smaller subset of test split. |

Each dataset entry contains a query, relevant document IDs, and graded relevance scores (0-3):

```json
{
  "id": "q01",
  "text": "red cardinal bird perched on branch",
  "relevant": ["obs_101", "obs_102", "obs_103"],
  "graded_relevance": {"obs_101": 3, "obs_102": 2, "obs_103": 1}
}
```

### Metrics

View all available metrics:

```bash
uv run inq bench metrics
```

The framework computes:

| Metric | Description |
|--------|-------------|
| **Precision@K** | Fraction of top-K results that are relevant |
| **Recall@K** | Fraction of relevant docs found in top-K |
| **NDCG@K** | Normalized Discounted Cumulative Gain (uses graded relevance) |
| **MAP@K** | Mean Average Precision |
| **MRR** | Mean Reciprocal Rank |

Plus latency statistics: `p50_ms`, `p95_ms`, `qps` (queries per second).

### Validate a Dataset

```bash
uv run inq bench validate bench/datasets/sample/sample-gold.json
```

### Run a Benchmark

```bash
# Single provider benchmark
uv run inq bench run \
  --dataset bench/datasets/sample/sample-gold.json \
  --provider qdrant \
  --limit 10 \
  --warmup 5 \
  --output results.json \
  --format both
```

### Compare Providers

```bash
uv run inq bench compare \
  --dataset bench/datasets/sample/sample-gold.json \
  --provider qdrant \
  --provider clip \
  --limit 10 \
  --format console
```

### Quantization Benchmark

Compare search quality across 4 quantization profiles (float32, scalar, scalar+rescore, binary+rescore):

```bash
uv run inq bench quantization \
  --dataset bench/datasets/inquire/inquire-val-subset.json \
  --collection float32-collection \
  --collection scalar-collection \
  --collection binary-collection \
  --limit 50 \
  --output quantization-results.json \
  --format both
```

Requires three pre-built Qdrant collections, each with the same data but different quantization configurations.

### Output Formats

- `console` — printed to stdout
- `json` — written to file (default: `benchmark-results.json`)
- `both` — printed and written

### Benchmark Tools

Additional scripts in `bench/tools/`:

- **`convert_inquire.py`** — Converts raw INQUIRE benchmark data into the project's JSON dataset format
- **`siglip2_server.py`** — Lightweight FastAPI embedding server for SigLIP2 on Apple MPS GPU (drop-in replacement for Infinity)

```bash
# Run the SigLIP2 embedding server
uv run python bench/tools/siglip2_server.py \
  --model-id google/siglip2-so400m-patch14-384 \
  --port 7997
```

---

## 7. Configuration System

iNatInq uses a layered YAML configuration system that bridges structured config files with the existing environment-variable-based settings. You can configure everything through YAML, environment variables, or a mix of both.

### Directory Structure

```
configs/
├── config.yaml              # Base configuration (all defaults)
├── secrets.example.yaml     # Template for credentials (copy to secrets.yaml)
├── environments/            # Environment-specific overrides
│   ├── local.yaml           # Local development (Docker Compose)
│   ├── dev.yaml             # Shared development
│   ├── staging.yaml         # Pre-production
│   └── prod.yaml            # Production
├── examples/                # Cloud provider templates (reference only)
│   ├── aws.yaml
│   ├── azure.yaml
│   └── gcp.yaml
└── schemas/
    └── config.schema.json   # JSON Schema for validation
```

### How Layering Works

Configuration is merged in order, with later sources overriding earlier ones:

```
config.yaml (base defaults)
    → environments/{ENV}.yaml (environment overlay)
        → secrets.yaml (credentials)
            → environment variables (final override)
```

**Environment variables always win.** YAML values are only applied when the corresponding env var is not already set. This means existing deployments that rely on env vars are fully backward compatible.

Select the environment overlay by setting `ENV` or `PIPELINE_ENV`:

```bash
# Local development (default — uses base config.yaml defaults)
uv run inq up

# Explicit environment
ENV=dev uv run inq up
ENV=staging uv run inq up
ENV=prod uv run inq up
```

### Base Config (`config.yaml`)

The base config contains all settings with sensible defaults for local development. It is organized into sections:

| Section | What It Configures |
|---------|--------------------|
| `storage` | S3/MinIO endpoint, bucket, region, SSL, retry, circuit breaker |
| `vector_databases` | Qdrant/Weaviate URLs, collection name, HNSW index tuning, sharding |
| `embeddings` | Text embedding provider and model (CLIP, Infinity, OpenAI, HuggingFace, SageMaker) |
| `image_embeddings` | Image embedding provider, preprocessing (resize, max size), circuit breaker |
| `processing` | Batch sizes, worker count, checkpoint settings |
| `ray` | Ray cluster resources, task concurrency, rate limits, timeouts |
| `databricks` | Databricks host, job ID, workspace path |
| `api` | FastAPI host, port, search limits |
| `benchmark` | Metrics list, K values, warmup, output format |
| `logging` | Log level and format (`text` or `json`) |
| `environment` | Environment mode (`local` or `cluster`) and K8s namespace |

For local development, the defaults work out of the box — no changes needed.

### Environment Overlays

Each environment overlay only specifies the values that differ from the base config. The deep-merge strategy means you only override what you need.

**`local.yaml`** — tuned for laptops:

```yaml
logging:
  level: "debug"
  format: "text"

processing:
  workers: 2
  batch_size: 10

ray:
  cluster:
    num_workers: 1
```

**`prod.yaml`** — tuned for production with `${VAR}` substitution:

```yaml
storage:
  endpoint: "${STORAGE_ENDPOINT}"
  bucket: "inatinq-prod-data"
  use_ssl: true
  path_style: false

vector_databases:
  qdrant:
    url: "${QDRANT_URL}"
    index:
      hnsw_m: 32
      hnsw_ef_construct: 200
    sharding:
      enabled: true
      shard_count: 3
      replication_factor: 2

processing:
  workers: 16
  batch_size: 100

logging:
  level: "info"
  format: "json"
```

The `${VAR_NAME}` syntax is resolved from `os.environ` at load time. Unresolved variables (not set in the environment) are left as-is.

### Secrets

Credentials are kept separate from configuration:

```bash
# Copy the template
cp configs/secrets.example.yaml configs/secrets.yaml
# Edit with your real credentials
```

The `secrets.yaml` file is gitignored. It follows the same YAML structure:

```yaml
storage:
  access_key: "your-access-key"
  secret_key: "your-secret-key"

vector_databases:
  qdrant:
    api_key: "your-qdrant-api-key"

embeddings:
  openai:
    api_key: "sk-..."

databricks:
  token: "dapi..."
```

For production, use environment variables or a secrets manager (AWS Secrets Manager, Azure Key Vault, K8s Secrets) instead of `secrets.yaml`.

### YAML-to-Environment-Variable Mapping

The config loader (`src/config_loader.py`) maps ~80 YAML paths to environment variables. Some key mappings:

| YAML Path | Environment Variable |
|-----------|---------------------|
| `storage.endpoint` | `S3_ENDPOINT` |
| `storage.bucket` | `S3_BUCKET` |
| `vector_databases.search_provider` | `VECTOR_DB_PROVIDER` |
| `vector_databases.qdrant.url` | `QDRANT_URL` |
| `embeddings.provider` | `EMBEDDING_PROVIDER` |
| `image_embeddings.url` | `CLIP_URL` |
| `processing.batch_size` | `RAY_S3_BATCH_SIZE` |
| `processing.checkpoint.enabled` | `RAY_CHECKPOINT_ENABLED` |
| `ray.address` | `RAY_ADDRESS` |
| `ray.cluster.num_workers` | `RAY_NUM_WORKERS` |
| `logging.level` | — (used directly) |
| `environment.mode` | `PIPELINE_ENV` |

See `YAML_TO_ENV_MAP` in `src/config_loader.py` for the full list.

### How It Integrates at Runtime

The Pydantic Settings classes in `src/config.py` read from `os.getenv()`. The YAML config loader bridges the gap:

```
initialize_config()           # Load & merge YAML files
    ↓
apply_yaml_defaults()         # Set env vars (only if not already set)
    ↓
Settings.from_env()           # Pydantic reads os.getenv() as before
```

This is triggered automatically by `get_settings()`:

```python
from config import get_settings

settings = get_settings()  # Loads YAML, applies defaults, returns Settings
```

`initialize_config()` is idempotent — safe to call multiple times, only runs once per process.

### Validation

Validate your configuration against the JSON Schema:

```bash
# Via the CLI
uv run inq dev validate-config

# Or directly
python configs/validate.py
```

The schema (`configs/schemas/config.schema.json`) checks types, required fields, and valid ranges for all sections.

### Cloud Provider Examples

The `configs/examples/` directory contains reference templates for AWS, Azure, and GCP deployments. These are starting points, not tested deployments:

```bash
# Use a cloud template as your production overlay
cp configs/examples/aws.yaml configs/environments/prod.yaml
# Edit with your specific values
```

### Configuration Troubleshooting

| Problem | Fix |
|---------|-----|
| Config not loading | Check that `configs/config.yaml` exists; run `ls configs/environments/${ENV}.yaml` |
| `${VAR}` not substituted | Ensure the variable is exported: `export VAR_NAME=value`. Syntax must be `${VAR_NAME}`, not `$VAR_NAME` |
| Schema validation fails | Run `python configs/validate.py` for detailed errors |
| YAML overridden by env var | This is by design — env vars always take precedence over YAML |
| Changes not taking effect | `get_settings()` is cached via `@lru_cache`; restart the process to pick up changes |

---

## 8. Architecture Overview

### Directory Structure

```
iNatInq/
├── src/
│   ├── api/              # FastAPI routes, middleware, models
│   ├── cli/              # Typer CLI (10 command groups)
│   ├── clients/          # External service wrappers (S3, Qdrant, CLIP, Infinity)
│   ├── core/             # Domain logic, services, ingestion pipelines, benchmark
│   ├── foundation/       # Cross-cutting: retry, circuit breaker, DLQ, logging, metrics
│   ├── config.py         # Pydantic settings with env var loading
│   └── main.py           # Uvicorn entry point
├── tests/                # Unit, integration, E2E
├── bench/                # Benchmarks, synthetic data, tools, datasets
├── postman/              # Postman collections
├── zarf/                 # Infrastructure
│   ├── compose/dev/      # Docker Compose config
│   ├── docker/           # Dockerfiles
│   ├── databricks/       # Databricks configs
│   └── scripts/          # Health check and smoke test scripts
├── docs/                 # ADRs, specs, architecture diagrams
└── configs/              # YAML configuration files
```

### Layered Architecture

```
Foundation (retry, circuit breaker, logging, DLQ)
    ↑
Client Layer (S3, Qdrant, CLIP, Infinity — via ABCs)
    ↑
Core/Domain Layer (services, models, ingestion pipelines, benchmark)
    ↑
API Layer (FastAPI routes, middleware)
```

Dependencies flow upward only. Each layer depends only on the layers below it.

### Key Patterns

- **Provider Abstraction**: `EmbeddingProvider` and `VectorDBProvider` ABCs allow swapping implementations without changing service code. Use factory functions (`create_embedding_provider`, `create_vector_db_provider`).
- **Strategy Pattern**: Ingestion pipeline uses `ClusterStrategy` to abstract Ray vs. Databricks cluster lifecycle.
- **Dependency Injection**: Configuration is loaded via `get_settings()` (cached with `@lru_cache`).
- **Repository Pattern**: Data access goes through client wrappers, never directly.

### Docker Compose Services

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| `minio` | `minio/minio:latest` | 9000, 9001 | S3-compatible object storage |
| `qdrant` | `qdrant/qdrant:v1.16.3` | 6333, 6334 | Vector database |
| `clip` | `ai4all/clip:latest` | 8001 | CLIP embedding service |
| `infinity` | `michaelf34/infinity:latest` | 7997 | SigLIP embedding (alternative) |
| `ray-head` | `rayproject/ray:2.53.0-py311` | 8265, 6379 | Ray cluster head |
| `ray-worker` | `rayproject/ray:2.53.0-py311` | — | Ray cluster worker |
| `redis` | `redis:8.6` | 8334 | Caching / DLQ storage |
| `pipeline` | (built locally) | 8000 | FastAPI application |

---

## 9. Troubleshooting

### "Circuit breaker is open"

If tests or requests fail with this error, the circuit breaker tripped due to repeated failures. Wait for the timeout period, or restart the affected service.

### Environment Detection

Service URLs auto-resolve based on in-cluster detection. If endpoints are wrong, check logs for the resolved URLs and verify `PIPELINE_ENV` is set correctly.

### Testcontainer Ports

Never hardcode ports in integration tests. Always use `container.get_exposed_port()` — testcontainers maps to random host ports.

### Async Clients in Ray Tasks

Ray tasks are sync. Use `asyncio.run()` to call async clients from within `@ray.remote` functions.

### Docker Resource Issues

If services crash or fail to start, ensure Docker Desktop has enough resources allocated (~10 CPUs, ~15 GiB RAM). Check with:

```bash
uv run inq docker ps
uv run inq docker health
```

### Ray Worker Logs

| Deployment | Command |
|------------|---------|
| Ray Dashboard | Jobs → Select job → Logs tab → Worker logs |
| Local | `cat /tmp/ray/session_latest/logs/worker-*.out` |
| Docker | `docker exec ray-head bash -c 'cat /tmp/ray/session_latest/logs/worker-*.out'` |

### Clean Slate

```bash
# Stop everything and remove all volumes (data will be lost)
uv run inq docker clean
```
