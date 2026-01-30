# iNatInq ML Pipeline

A semantic search and document ingestion service built with FastAPI, Ray/Spark, and vector databases.

## Overview

This service provides two core capabilities:

| Capability | Description |
|------------|-------------|
| **Query Engine** | Semantic search over documents using vector similarity |
| **Ingestion Engine** | Distributed processing of S3 documents into vector databases |

**Stack**: FastAPI · Ray · Spark · Ollama · Qdrant · Weaviate · MinIO

---

## Query Engine

The query engine handles semantic search requests by generating embeddings and performing vector similarity search.

<img src="charts/query_flow.png" alt="Query Engine Flow" width="600"/>

<details>
<summary>Sequence Diagram</summary>

<img src="charts/query_sequence.png" alt="Query Engine Sequence" width="700"/>

</details>

**Endpoints**:

- `GET /search?q=your query&limit=10&provider=qdrant` – Text-to-text semantic search
- `GET /search/images?q=sunset over ocean&limit=10` – Text-to-image search using CLIP

**Flow**: HTTP Request → Embedding (Ollama/CLIP) → Vector Search → Ranked Results

---

## Ingestion Engine

The ingestion engine processes documents from S3 into vector databases using distributed computing (Ray or Spark).

<img src="charts/ingestion_flow.png" alt="Ingestion Engine Flow" width="700"/>

<details>
<summary>Sequence Diagram</summary>

<img src="charts/ingestion_sequence.png" alt="Ingestion Engine Sequence" width="700"/>

</details>

**Endpoints**:

- `POST /ray/jobs` – Submit Ray job
- `POST /spark/jobs` – Submit Spark job
- `POST /databricks/jobs` – Submit Databricks job run
- `GET /databricks/jobs/{run_id}` – Get Databricks run status
- `GET /databricks/jobs/{run_id}/logs` – Get Databricks run output
- `DELETE /databricks/jobs/{run_id}` – Stop Databricks run

**Flow**: Job Submit → S3 List → Parallel Workers → Embed → Upsert to Qdrant + Weaviate

---

## Quick Start

```bash
# Start all services
make up

# View status
make status

# Open all dashboards
make ui-all

# Stop services
make down
```

**Service Endpoints** (after `make up`):

| Service | URL |
|---------|-----|
| Pipeline API | <http://localhost:8000/docs> |
| MinIO Console | <http://localhost:9001> |
| Qdrant Dashboard | <http://localhost:6333/dashboard> |
| Ray Dashboard | <http://localhost:8265> |

### Using External Services (Optional)

To use external/cloud-hosted services instead of local Docker containers:

```bash
# External Ollama (embedding service)
export OLLAMA_BASE_URL=http://your-ollama-host:11434
export OLLAMA_MODEL=nomic-embed-text  # or your preferred model
make docker-up

# Qdrant Cloud (https://cloud.qdrant.io/)
export QDRANT_URL=https://your-cluster.region.cloud.qdrant.io
export QDRANT_API_KEY=your-api-key
make docker-up

# Weaviate Cloud (https://console.weaviate.cloud/)
export VECTOR_DB_PROVIDER=weaviate
export WEAVIATE_URL=https://your-cluster.region.weaviate.cloud
export WEAVIATE_API_KEY=your-api-key
make docker-up

# Azure Databricks (optional - for Databricks job execution/integration tests)
export DATABRICKS_HOST=https://adb-<workspace-id>.<region>.azuredatabricks.net
export DATABRICKS_TOKEN=your-databricks-token
export DATABRICKS_JOB_ID=123
export DATABRICKS_TASK_TYPE=python

# Image Embedding (CLIP) - for image search
export IMAGE_EMBEDDING_PROVIDER=clip  # or "llava" for Ollama LLaVA
export CLIP_URL=http://your-clip-host:8000
export CLIP_MODEL=ViT-B/32
export CLIP_BACKEND=clip  # "clip" for ai4all/clip, "ollama" for Ollama LLaVA

# Image Processing Settings
export IMAGE_BATCH_SIZE=10       # Images per processing batch
export IMAGE_MAX_SIZE_MB=10.0    # Reject images larger than this
export IMAGE_TARGET_SIZE=224     # Resize dimension for CLIP input

# Or use a local config file (gitignored)
cp apps/zarf/databricks/dev/env.local.example apps/zarf/databricks/dev/.env.local
# Edit .env.local with your Databricks credentials

# Manage the Databricks cluster (requires Databricks CLI)
make azure-databricks-build
make azure-databricks-up
make azure-databricks-down
```

For full Databricks setup details, see `zarf/databricks/README.md`.

---

## Developer Guide

### Prerequisites

- Docker & Docker Compose
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended)

### Setup

```bash
cd apps

# Install dependencies
uv sync

# Run tests
make test

# Run with coverage
make test-cov

# Lint & format
make lint
make format
```

### Running Locally (without Docker)

```bash
# Start dev server
make dev

# Or directly
uv run uvicorn api.app:app --reload --port 8000
```

### End-to-End Testing

Seed MinIO with synthetic data and run a complete ingestion + search test:

```bash
# 1. Start all services
make docker-up

# 2. Generate & upload test documents (100 by default)
make syntheticdata-setup COUNT=100

# 3. Submit a Ray ingestion job
curl -X POST http://localhost:8000/ray/jobs \
  -H "Content-Type: application/json" \
  -d '{"s3_prefix": "inputs/", "collection": "documents"}'

# 4. Check job status (use job_id from step 3)
curl http://localhost:8000/ray/jobs/<job_id>

# 5. Search the indexed documents
curl "http://localhost:8000/search?q=whale&limit=5"
```

**One-liner** (generate + upload + ingest + search):

```bash
make syntheticdata-setup COUNT=100 && \
  curl -s -X POST http://localhost:8000/ray/jobs \
    -H "Content-Type: application/json" \
    -d '{"s3_prefix": "inputs/", "collection": "documents"}' | jq .
```

See [syntheticdata/README.md](syntheticdata/README.md) for more options.

### Viewing Ray Worker Logs

Ray jobs have two types of logs:

| Log Type | Contains | Access Method |
|----------|----------|---------------|
| **Driver Logs** | Job orchestration, progress updates | Job Logs API, Dashboard |
| **Worker Logs** | Task execution, circuit breaker events, errors | Worker log files |

**Accessing worker logs by deployment:**

| Deployment | Command |
|------------|---------|
| **Ray Dashboard** | Jobs → Select job → Logs tab → Worker logs |
| **Local** | `cat /tmp/ray/session_latest/logs/worker-*.out` |
| **Docker** | `docker exec ray-head bash -c 'cat /tmp/ray/session_latest/logs/worker-*.out'` |
| **Kubernetes** | `kubectl logs <ray-worker-pod>` or Ray Dashboard |

**Filtering for errors:**

```bash
# Docker: Find circuit breaker and upstream errors
docker exec ray-head bash -c 'cat /tmp/ray/session_latest/logs/worker-*.out | grep -iE "(CIRCUIT_BREAKER|UPSTREAM_ERROR)"'

# Local: Same pattern
cat /tmp/ray/session_latest/logs/worker-*.out | grep -iE "(CIRCUIT_BREAKER|UPSTREAM_ERROR)"
```

---

## Codebase Structure

```
apps/
├── src/
│   ├── api/              # FastAPI routes and models
│   ├── clients/          # External service clients (S3, Qdrant, Ollama, etc.)
│   ├── core/             # Domain logic
│   │   ├── ingestion/    # Ray & Spark processing pipelines
│   │   └── services/     # Business logic (search, job orchestration)
│   ├── foundation/       # Utilities (retry, circuit breaker, logging)
│   └── config.py         # Pydantic settings
├── tests/unit/           # Unit tests
├── syntheticdata/        # Test data generation & S3 upload tools
├── charts/               # Architecture diagrams
└── zarf/                 # Docker & infrastructure
    ├── compose/dev/      # Docker Compose config
    └── docker/dev/       # Dockerfiles
```

### Module READMEs

| Module | Description |
|--------|-------------|
| [api/](src/api/README.md) | HTTP endpoints and middleware |
| [clients/](src/clients/README.md) | Service client abstractions |
| [core/](src/core/README.md) | Domain models and exceptions |
| [core/services/](src/core/services/README.md) | Business logic layer |
| [foundation/](src/foundation/README.md) | Cross-cutting utilities |
| [syntheticdata/](syntheticdata/README.md) | Test data generation & upload |
| [charts/](charts/README.md) | Architecture diagrams |
| [zarf/](zarf/README.md) | Infrastructure configs |

---

## Test Coverage

- **458 tests** across foundation, clients, core, and API
- **>90% code coverage**
- Uses pytest with async support and comprehensive mocking
