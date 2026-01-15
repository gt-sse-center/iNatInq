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

**Endpoint**: `GET /search?q=your query&limit=10&provider=qdrant`

**Flow**: HTTP Request → Ollama Embedding → Vector Search → Ranked Results

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
| [charts/](charts/README.md) | Architecture diagrams |
| [zarf/](zarf/README.md) | Infrastructure configs |

---

## Test Coverage

- **458 tests** across foundation, clients, core, and API
- **>90% code coverage**
- Uses pytest with async support and comprehensive mocking
