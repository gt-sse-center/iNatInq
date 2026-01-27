"""FastAPI HTTP routes for the pipeline service.

This module defines all HTTP endpoints for the pipeline service. It handles:
- Request/response serialization via Pydantic models
- Error translation from `PipelineError` to HTTP status codes
- Dependency injection (settings, clients)

## Endpoints

- `GET /healthz`: Liveness/readiness probe (no dependency checks)
- `GET /search`: Perform semantic search over documents in vector database
(Qdrant or Weaviate)
- `POST /ray/jobs`: Submit Ray job to process S3 data and store embeddings
in vector database
- `GET /ray/jobs/{job_id}`: Check status of a submitted job

## Error Handling

All service layer exceptions (`PipelineError` hierarchy) are handled by the
`ExceptionHandlerMiddleware` which translates them to HTTP status codes:
- `BadRequestError` → 400 (Bad Request)
- `UpstreamError` → 502 (Bad Gateway)
- `PipelineError` (base) → 500 (Internal Server Error)

Route handlers can raise these exceptions directly; the middleware will
catch them and return appropriate HTTP responses.

## OpenAPI Documentation

FastAPI automatically generates OpenAPI/Swagger documentation at:
- `/docs`: Interactive Swagger UI
- `/redoc`: ReDoc documentation
- `/openapi.json`: OpenAPI schema JSON
"""

from datetime import datetime, timezone
from typing import Annotated

from api import models
from clients.interfaces.embedding import create_embedding_provider
from clients.interfaces.vector_db import create_vector_db_provider
from config import EmbeddingConfig, MinIOConfig, VectorDBConfig, get_settings
from core.exceptions import BadRequestError, PipelineError
from core.services.databricks_ray_service import DatabricksRayService
from core.services.ray_service import RayService
from core.services.search_service import SearchService
from fastapi import APIRouter, Query

router = APIRouter()


@router.get("/healthz", tags=["health"])
def healthz() -> dict[str, str]:
    """Liveness and readiness probe endpoint.

    This endpoint is used by Kubernetes liveness and readiness probes. It
    returns a simple success response without checking dependencies (MinIO,
    Ray, Ollama, Qdrant).

    Returns:
        A simple status dict: `{"status": "ok"}`

    Note:
        We intentionally do not check dependencies here because:
        - Dependencies can be slow to start in dev environments
        - Dependency checks can cause restart loops if services are temporarily
        unavailable
        - The service can start accepting requests even if some dependencies
        aren't ready yet

    """
    return {"status": "ok"}


@router.get("/search", response_model=models.SearchResponse, tags=["vector-store"])
async def search(
    q: str,
    limit: int = 10,
    collection: str | None = None,
    model: str | None = None,
    provider: Annotated[
        str | None,
        Query(
            pattern="^(qdrant|weaviate|pinecone|milvus)$",
            description=("Vector database provider (defaults to VECTOR_DB_PROVIDER env var)"),
        ),
    ] = None,
) -> models.SearchResponse:
    """Perform semantic search over documents in vector database.

    This endpoint:
    1. Generates an embedding for the query text via Ollama
    2. Searches vector database (Qdrant or Weaviate) for similar documents using
    vector similarity
    3. Returns ranked results with similarity scores and metadata

    Args:
        q: Natural language query string (required).
        limit: Maximum number of results to return (default: 10, max: 100).
        collection: Optional collection name (defaults to service default).
        model: Optional Ollama model name (defaults to service default).
        provider: Optional vector database provider (qdrant, weaviate, pinecone, milvus).
            If not specified, uses VECTOR_DB_PROVIDER environment variable
            (defaults to "qdrant").

    Returns:
        Response containing search results ordered by similarity (highest
        first).

    Raises:
        HTTPException(400): If query is empty, limit is invalid, or provider is
        invalid.
        HTTPException(502): If Ollama or vector database operations fail.
        HTTPException(404): If collection doesn't exist.

    Example Request:
        ```
        GET /search?q=machine%20learning%20pipeline&limit=5&provider=qdrant
        GET /search?q=machine%20learning%20pipeline&limit=5&provider=weaviate
        ```

    Example Response:
        ```json
        {
            "query": "machine learning pipeline",
            "model": "nomic-embed-text",
            "collection": "documents",
            "provider": "qdrant",
            "results": [
                {
                    "id": "d790dd2c-99eb-4901-b9c9-538b58318fe3",
                    "score": 0.9234,
                    "text": "s3://pipeline/inputs/hello-a01f74c0.txt",
                    "metadata": {
                        "s3_key": "inputs/hello-a01f74c0.txt"
                    }
                }
            ],
            "total": 1
        }
        ```

    Note:
        - Results are ordered by similarity score (highest first)
        - Score ranges from 0.0 to 1.0 (1.0 = identical, 0.0 = completely
        different)
        - Uses cosine similarity for vector comparison
        - Collection must exist (will not be auto-created for search)
        - Both Qdrant and Weaviate can be used simultaneously (specify provider
        per request)
    """
    s = get_settings()
    embedding_provider = create_embedding_provider(s.embedding)

    # Determine provider type: use query parameter if provided, otherwise use
    # settings
    if provider:
        try:
            vector_db_config = VectorDBConfig.from_env_for_provider(
                provider_type=provider.lower(), namespace=s.k8s_namespace
            )
        except ValueError as e:
            raise BadRequestError(str(e)) from e
        provider_type = vector_db_config.provider_type
    else:
        # Use default from settings
        vector_db_config = s.vector_db
        provider_type = vector_db_config.provider_type

    vector_db_provider = create_vector_db_provider(vector_db_config)
    model_name = model or s.embedding.ollama_model
    collection_name = collection or vector_db_config.collection

    search_service = SearchService(
        embedding_provider=embedding_provider,
        vector_db_provider=vector_db_provider,
    )
    search_results = await search_service.search_documents_async(
        collection=collection_name,
        query=q,
        limit=limit,
    )

    # Convert SearchResultItem to Pydantic SearchResult
    pydantic_results = []
    for item in search_results.items:
        # Extract text from payload, keep rest as metadata
        payload_copy = dict(item.payload)
        text = payload_copy.pop("text", "")
        pydantic_results.append(
            models.SearchResult(
                id=item.point_id,
                score=item.score,
                text=text,
                metadata=payload_copy,
            )
        )

    return models.SearchResponse(
        query=q,
        model=model_name or "",
        collection=collection_name,
        provider=provider_type,
        results=pydantic_results,
        total=search_results.total,
    )
    # Note: Providers are not explicitly closed here to avoid event loop issues.
    # They will be garbage collected when the function returns since they're
    # local variables created for this request. The underlying HTTP sessions
    # will be cleaned up by Python's garbage collector.


# ═══════════════════════════════════════════════════════════════════════════════
# Ray Job Management Endpoints
# ═══════════════════════════════════════════════════════════════════════════════


@router.post(
    "/ray/jobs",
    response_model=models.RayJobResponse,
    status_code=202,
    tags=["ray-jobs"],
)
async def submit_ray_job(req: models.RayJobRequest) -> models.RayJobResponse:
    """Submit a new Ray job to process S3 documents.

    This submits a job to the Ray cluster and returns immediately.
    Use GET /ray/jobs/{job_id} to check status.

    The job will:
    1. List documents from S3/MinIO at the specified prefix
    2. Generate embeddings via Ollama
    3. Store vectors in Qdrant and Weaviate

    Args:
        req: Request containing s3_prefix and collection.

    Returns:
        Job metadata including Ray-generated job ID and submission timestamp.

    Raises:
        HTTPException(500): If job submission fails.

    Example Request:
        ```json
        POST /ray/jobs
        {
            "s3_prefix": "inputs/",
            "collection": "documents"
        }
        ```

    Example Response:
        ```json
        {
            "job_id": "raysubmit_1234567890",
            "status": "submitted",
            "namespace": "ml-system",
            "s3_prefix": "inputs/",
            "collection": "documents",
            "submitted_at": "2026-01-12T15:30:45.123456Z"
        }
        ```
    """
    try:
        settings = get_settings()
        namespace = settings.k8s_namespace

        ray_service = RayService()
        minio_cfg = MinIOConfig.from_env(namespace)
        embed_cfg = EmbeddingConfig.from_env(namespace)
        vector_cfg = VectorDBConfig.from_env(namespace)

        job_id = ray_service.submit_s3_to_qdrant(
            namespace=namespace,
            s3_endpoint=minio_cfg.endpoint_url,
            s3_access_key_id=minio_cfg.access_key_id,
            s3_secret_access_key=minio_cfg.secret_access_key,
            s3_bucket=minio_cfg.bucket,
            s3_prefix=req.s3_prefix,
            embedding_config=embed_cfg,
            vector_db_config=vector_cfg,
            collection=req.collection,
        )

        return models.RayJobResponse(
            job_id=job_id,
            status="submitted",
            namespace=namespace,
            s3_prefix=req.s3_prefix,
            collection=req.collection,
            submitted_at=datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        )
    except Exception as e:
        raise PipelineError(f"Failed to submit Ray job: {e!s}") from e


@router.post(
    "/databricks/jobs",
    response_model=models.DatabricksJobResponse,
    status_code=202,
    tags=["databricks-jobs"],
)
async def submit_databricks_job(req: models.DatabricksJobRequest) -> models.DatabricksJobResponse:
    """Submit a new Databricks job to process S3 documents.

    This submits a run for a preconfigured Databricks Job and returns immediately.

    Args:
        req: Request containing s3_prefix and collection.

    Returns:
        Job metadata including Databricks run ID and submission timestamp.

    Raises:
        HTTPException(500): If job submission fails.
    """
    try:
        settings = get_settings()
        namespace = settings.k8s_namespace

        databricks_service = DatabricksRayService()
        minio_cfg = MinIOConfig.from_env(namespace)
        embed_cfg = EmbeddingConfig.from_env(namespace)
        vector_cfg = VectorDBConfig.from_env(namespace)

        run_id = databricks_service.submit_s3_to_qdrant(
            namespace=namespace,
            s3_endpoint=minio_cfg.endpoint_url,
            s3_access_key_id=minio_cfg.access_key_id,
            s3_secret_access_key=minio_cfg.secret_access_key,
            s3_bucket=minio_cfg.bucket,
            s3_prefix=req.s3_prefix,
            embedding_config=embed_cfg,
            vector_db_config=vector_cfg,
            collection=req.collection,
        )

        return models.DatabricksJobResponse(
            run_id=str(run_id),
            status="submitted",
            namespace=namespace,
            s3_prefix=req.s3_prefix,
            collection=req.collection,
            submitted_at=datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        )
    except Exception as e:
        raise PipelineError(f"Failed to submit Databricks job: {e!s}") from e


@router.delete(
    "/databricks/jobs/{run_id}",
    response_model=models.DatabricksJobStopResponse,
    tags=["databricks-jobs"],
)
async def stop_databricks_job(run_id: str) -> models.DatabricksJobStopResponse:
    """Stop a running Databricks job run.

    Args:
        run_id: Databricks run ID to stop.

    Returns:
        Stop confirmation.

    Raises:
        HTTPException(500): If stopping fails.
    """
    try:
        databricks_service = DatabricksRayService()
        databricks_service.stop_run(run_id)
        return models.DatabricksJobStopResponse(run_id=str(run_id), status="stopped")
    except Exception as e:
        raise PipelineError(f"Failed to stop Databricks job: {e!s}") from e


@router.get(
    "/databricks/jobs/{run_id}",
    response_model=models.DatabricksJobStatusResponse,
    tags=["databricks-jobs"],
)
async def get_databricks_job_status(run_id: str) -> models.DatabricksJobStatusResponse:
    """Get the status of a Databricks job run.

    Args:
        run_id: Databricks run ID to query.

    Returns:
        Run status details including lifecycle/result states.

    Raises:
        HTTPException(500): If status query fails.
    """
    try:
        databricks_service = DatabricksRayService()
        status = databricks_service.get_run_status(run_id)
        return models.DatabricksJobStatusResponse(run_id=str(run_id), **status)
    except Exception as e:
        raise PipelineError(f"Failed to get Databricks job status: {e!s}") from e


@router.get(
    "/databricks/jobs/{run_id}/logs",
    response_model=models.DatabricksJobLogsResponse,
    tags=["databricks-jobs"],
)
async def get_databricks_job_logs(run_id: str) -> models.DatabricksJobLogsResponse:
    """Get output/logs for a Databricks job run.

    Args:
        run_id: Databricks run ID to query.

    Returns:
        Run output/logs as a string (best-effort).

    Raises:
        HTTPException(500): If log retrieval fails.
    """
    try:
        databricks_service = DatabricksRayService()
        logs = databricks_service.get_run_output(run_id)
        return models.DatabricksJobLogsResponse(run_id=str(run_id), logs=logs)
    except Exception as e:
        raise PipelineError(f"Failed to get Databricks job logs: {e!s}") from e


@router.get("/ray/jobs/{job_id}", response_model=models.RayJobStatusResponse, tags=["ray-jobs"])
async def get_ray_job_status(job_id: str) -> models.RayJobStatusResponse:
    """Get the status of a Ray job.

    Args:
        job_id: Ray job ID (returned from POST /ray/jobs).

    Returns:
        Current job status including state and optional error message.

    Raises:
        HTTPException(404): If job not found.
        HTTPException(500): If status query fails.

    Example Response:
        ```json
        {
            "job_id": "raysubmit_1234567890",
            "status": "RUNNING",
            "message": null
        }
        ```
    """
    try:
        settings = get_settings()
        ray_service = RayService()
        status = ray_service.get_job_status(job_id, settings.k8s_namespace)

        return models.RayJobStatusResponse(job_id=job_id, **status)
    except Exception as e:
        raise PipelineError(f"Failed to get job status: {e!s}") from e


@router.get(
    "/ray/jobs/{job_id}/logs",
    response_model=models.RayJobLogsResponse,
    tags=["ray-jobs"],
)
async def get_ray_job_logs(job_id: str) -> models.RayJobLogsResponse:
    r"""Get logs from a Ray job.

    Args:
        job_id: Ray job ID.

    Returns:
        Job logs as string.

    Raises:
        HTTPException(404): If job not found.
        HTTPException(500): If log retrieval fails.

    Example Response:
        ```json
        {
            "job_id": "raysubmit_1234567890",
            "logs": "Processing 1000 documents...\nCompleted successfully."
        }
        ```
    """
    try:
        settings = get_settings()
        ray_service = RayService()
        logs = ray_service.get_job_logs(job_id, settings.k8s_namespace)

        return models.RayJobLogsResponse(job_id=job_id, logs=logs)
    except Exception as e:
        raise PipelineError(f"Failed to get job logs: {e!s}") from e


@router.delete("/ray/jobs/{job_id}", response_model=models.RayJobStopResponse, tags=["ray-jobs"])
async def stop_ray_job(job_id: str) -> models.RayJobStopResponse:
    """Stop a running Ray job.

    This terminates the job gracefully.

    Args:
        job_id: Ray job ID to stop.

    Returns:
        Stop confirmation.

    Raises:
        HTTPException(500): If stopping fails.

    Example Response:
        ```json
        {
            "job_id": "raysubmit_1234567890",
            "status": "stopped"
        }
        ```
    """
    try:
        settings = get_settings()
        ray_service = RayService()
        ray_service.stop_job(job_id, settings.k8s_namespace)

        return models.RayJobStopResponse(job_id=job_id, status="stopped")
    except Exception as e:
        raise PipelineError(f"Failed to stop job: {e!s}") from e
