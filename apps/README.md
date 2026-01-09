# Apps - Shared Infrastructure

This package contains shared infrastructure components and services for the iNatInq platform.

## Overview

The `apps` package is organized into modular components that can be reused across different services:

### Current Components

- **`foundation/`**: Core utilities for building resilient services
  - Retry logic with exponential backoff (tenacity)
  - Circuit breaker pattern (pybreaker)
  - Rate limiting for API calls
  - HTTP session management with retries
  - Structured JSON logging utilities
  - Custom exception hierarchy

### Planned Components

As we migrate from the `modern-web-application` repository, this package will expand to include:

- **`pipeline/`**: ML pipeline orchestration service (migrating from modern-web-application/apps/pipeline)
  - FastAPI service coordinating S3 → Spark → Ollama → Qdrant workflow
  - Embedding generation and vector database operations
  - Semantic search over indexed documents
  - Kubernetes-native batch job orchestration

The pipeline service will integrate foundation utilities for resilient communication with external services (MinIO, Spark, Ollama, Qdrant).

## Setup

This package is part of the iNatInq workspace. To install dependencies:

```bash
# From the root iNatInq directory
uv sync

# Or install dependencies manually
cd apps
pip install -e .
pip install -e ".[dev]"
```

## Running Tests

Tests require dependencies to be installed. Once installed, run:

```bash
# From the root iNatInq directory
uv run pytest apps/tests/

# Or with specific test file
uv run pytest apps/tests/unit/foundation/test_http.py

# With coverage (already configured in pyproject.toml)
uv run pytest apps/tests/ -v
```

The foundation package is installed in the workspace via uv, so imports like `from foundation import ...` work correctly.

## Structure

```text
apps/
├── src/
│   └── foundation/          # Package source code
├── tests/
│   └── unit/
│       └── foundation/      # Unit tests
└── pyproject.toml           # Package configuration
```

