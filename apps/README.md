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
  - Async resource management utilities

- **`clients/`**: External service client wrappers
  - S3/MinIO client for object storage
  - Qdrant client for vector database operations
  - Weaviate client for vector database operations
  - Ollama client for embedding generation
  - Provider abstraction layer (ABCs) for swappable implementations
  - Registry pattern for provider discovery
  - Circuit breaker integration for fault tolerance

- **`core/`**: Core domain models and exceptions
  - Data models (SearchResult, SearchResults, VectorPoint, etc.)
  - Exception hierarchy (UpstreamError, etc.)
  - Shared types and interfaces

- **`config.py`**: Centralized configuration management
  - Pydantic-based settings
  - Environment variable loading
  - Type-safe configuration for all services

### Planned Components

As we migrate from the `modern-web-application` repository, this package will expand to include:

- **`pipeline/`**: ML pipeline orchestration service (migrating from modern-web-application/apps/pipeline)
  - FastAPI service coordinating S3 → Spark → Ollama → Qdrant workflow
  - Embedding generation and vector database operations
  - Semantic search over indexed documents
  - Kubernetes-native batch job orchestration

The pipeline service will integrate foundation utilities and client wrappers for resilient communication with external services.

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
# From the apps directory
cd apps
uv run pytest

# Or with specific test file
uv run pytest tests/unit/foundation/test_http.py

# Or from root iNatInq directory
uv run pytest apps/tests/ -v

# With coverage report
uv run pytest --cov
```

The packages (`foundation`, `clients`, `core`) are configured in `pyproject.toml` with `pythonpath = ["src"]`, so imports like `from foundation import ...`, `from clients import ...`, and `from core import ...` work correctly in tests.

## Structure

```text
apps/
├── src/
│   ├── foundation/          # Core utilities (retry, circuit breaker, logging, etc.)
│   ├── clients/             # External service clients (S3, Qdrant, Ollama, etc.)
│   ├── core/                # Domain models and exceptions
│   └── config.py            # Configuration management
├── tests/
│   ├── unit/
│   │   ├── foundation/      # Foundation unit tests
│   │   └── clients/         # Client unit tests
│   └── conftest.py          # Shared test fixtures
└── pyproject.toml           # Package configuration
```

### Test Coverage

- **168 tests** covering foundation utilities, clients, and core models
- **92.59% code coverage** (exceeds 70% requirement)
- Tests use pytest with async support and comprehensive mocking
