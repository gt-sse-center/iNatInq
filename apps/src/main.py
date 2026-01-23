"""Dev-only ML pipeline orchestrator service (FastAPI).

This file is the **stable Uvicorn entrypoint** for the pipeline container:

- Docker: `uvicorn src.main:app`
- K8s probes: `GET /healthz`

The implementation has been refactored into a package under `src/` so the codebase
scales beyond a single file without losing clarity.

## High-level architecture (dev-only)

- **MinIO**: S3-compatible object storage (dev replacement for AWS S3)
- **Ray**: distributed compute for batch processing
- **Ollama (CPU)**: embeddings service (dev replacement for SageMaker hosting)
- **Qdrant**: vector database (stores embeddings + payload metadata)
- **Weaviate**: vector database (alternative to Qdrant for comparison)

## High-level data flow

The pipeline processes data via Ray jobs:

1. Write objects to MinIO: `s3://{bucket}/{key}`
2. Submit Ray jobs that process S3 data in parallel
3. Ray workers generate embeddings via Ollama's embeddings API
4. Ray workers upsert embeddings + payload into Qdrant and Weaviate

## Configuration (environment variables)

Configuration is loaded in `src/config.py` (see `Settings`), via env vars:

- `OLLAMA_BASE_URL`, `OLLAMA_MODEL`
- `QDRANT_URL`, `QDRANT_COLLECTION`
- `WEAVIATE_URL`, `WEAVIATE_COLLECTION`
- `S3_ENDPOINT`, `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY`, `S3_BUCKET`
- `RAY_ADDRESS`, `K8S_NAMESPACE`

## Code organization

- `src/config.py`: settings + env loading
- `src/api/models.py`: Pydantic request/response models
- `src/clients/*`: external system clients (MinIO/S3, Qdrant, Weaviate, Ollama, Ray)
- `src/core/services/*`: orchestration logic (no FastAPI dependencies,
  following Go codebase pattern)
- `src/api/routes.py`: HTTP routing + exception translation
- `src/api/app.py`: FastAPI app factory (`create_app`)
"""

from api.app import create_app

app = create_app()
