"""Dev-only ML pipeline orchestrator service (FastAPI).

This file is the **stable Uvicorn entrypoint** for the pipeline container:

- Docker: `uvicorn src.main:app`
- K8s probes: `GET /healthz`

The implementation has been refactored into a package under `src/` so the codebase
scales beyond a single file without losing clarity.

## High-level architecture (dev-only)

- **MinIO**: S3-compatible object storage (dev replacement for AWS S3)
- **Spark**: batch compute (standalone cluster in dev)
- **Ollama (CPU)**: embeddings service (dev replacement for SageMaker hosting)
- **Qdrant**: vector database (stores embeddings + payload metadata)

## High-level data flow

The pipeline processes data via Spark jobs:

1. Write objects to MinIO: `s3://{bucket}/{key}`
2. Submit Spark jobs via Kubernetes `batch/v1 Job` that process S3 data
3. Spark jobs generate embeddings via Ollama's embeddings API
4. Spark jobs upsert embeddings + payload into Qdrant

## Spark driver networking gotcha

When the Spark driver runs inside Kubernetes (our `spark-submit` Job pod), executors run on
Spark workers and must connect back to the driver. To keep this reliable, the submit Job sets:

- `spark.driver.bindAddress=0.0.0.0`
- `spark.driver.host=${POD_IP}`

## Configuration (environment variables)

Configuration is loaded in `src/config.py` (see `Settings`), via env vars:

- `OLLAMA_BASE_URL`, `OLLAMA_MODEL`
- `QDRANT_URL`, `QDRANT_COLLECTION`
- `S3_ENDPOINT`, `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY`, `S3_BUCKET`
- `SPARK_MASTER_URL`, `K8S_NAMESPACE`

## Code organization

- `src/config.py`: settings + env loading
- `src/api/models.py`: Pydantic request/response models
- `src/clients/*`: external system clients (MinIO/S3, Qdrant, Ollama, Kubernetes)
- `src/core/services/*`: orchestration logic (no FastAPI dependencies,
  following Go codebase pattern)
- `src/api/routes.py`: HTTP routing + exception translation
- `src/api/app.py`: FastAPI app factory (`create_app`)
"""

from api.app import create_app

app = create_app()

