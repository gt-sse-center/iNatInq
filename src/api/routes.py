"""FastAPI HTTP routes for the pipeline service.

This module defines all HTTP endpoints for the pipeline service. It handles:
- Request/response serialization via Pydantic models
- Error translation from `PipelineError` to HTTP status codes
- Dependency injection (settings, clients)

## Endpoints

**Health & Monitoring:**
- `GET /healthz`: Liveness/readiness probe (no dependency checks)
- `GET /metrics`: Prometheus metrics endpoint (auto-exposed by instrumentator)
- `POST /ingestion/metrics`: Ingestion job metrics reporting from ephemeral processes

**Search:**
- `GET /search/images`: Semantic search over images using text queries

**Ray Jobs:**
- `POST /ray/jobs/images`: Submit Ray job to process S3 images
- `POST /ray/jobs/process-dlq`: Process dead letter queue entries with local Ray
- `GET /ray/jobs/{job_id}`: Check Ray job status
- `GET /ray/jobs/{job_id}/logs`: Retrieve Ray job logs
- `DELETE /ray/jobs/{job_id}`: Stop a running Ray job

**Databricks Jobs:**
- `POST /databricks/jobs/images`: Submit Databricks job to process S3 or iNaturalist images
- `POST /databricks/jobs/cdc-producer`: Submit Databricks CDC Auto Loader producer job
- `POST /databricks/jobs/process-dlq`: Process DLQ entries with Ray on Databricks
- `POST /databricks/jobs/cdc-consumer`: Submit Databricks CDC Bronze consumer job
- `GET /databricks/jobs/{run_id}`: Check Databricks run status
- `GET /databricks/jobs/{run_id}/logs`: Retrieve Databricks run logs
- `DELETE /databricks/jobs/{run_id}`: Stop a Databricks run

**Cache:**
- `DELETE /cache`: Invalidate the semantic cache

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

import asyncio
import logging
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Query

from api import models
from clients.cache import get_semantic_cache
from clients.interfaces.embedding import create_embedding_provider
from clients.interfaces.vector_db import create_vector_db_provider
from config import EmbeddingConfig, MinIOConfig, ProviderType, VectorDBConfig, get_settings
from core.exceptions import BadRequestError, PipelineError
from foundation.exceptions import UpstreamError
from core.services.databricks_ray_service import DatabricksRayService
from core.services.ray_service import RayService
from core.services.search_service import ImageSearchService
from foundation.metrics.job_metrics_reporter import INGESTION_METRICS_PATH
from foundation.metrics.registry import (
    INGESTION_BATCH_DURATION,
    INGESTION_CHECKPOINT_SAVES,
    INGESTION_DOCS_PROCESSED,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Model Override Helpers
# =============================================================================


def auto_detect_image_provider(model_name: str) -> ProviderType:
    """Infer the embedding provider type from a model name.

    Args:
        model_name: Model identifier (e.g. ``"google/siglip-so400m-patch14-384"``).

    Returns:
        The most appropriate ``ProviderType`` for the given model.
    """
    lower = model_name.lower()
    if "siglip" in lower:
        return ProviderType.INFINITY
    if "llava" in lower:
        return ProviderType.LOCAL_CLIP
    return ProviderType.LOCAL_CLIP


def _model_override_kwargs(model: str, resolved_provider: ProviderType) -> dict[str, object]:
    """Build ``model_copy(update=...)`` kwargs for an embedding config override.

    Args:
        model: Override model name.
        resolved_provider: The provider type to use.

    Returns:
        Dict suitable for ``EmbeddingConfig.model_copy(update=...)``.
    """
    base: dict[str, object] = {"provider_type": resolved_provider}
    if resolved_provider == ProviderType.INFINITY:
        base["infinity_model"] = model
    elif resolved_provider in (ProviderType.LOCAL_CLIP, ProviderType.HOSTED_CLIP):
        base["clip_model"] = model
    return base


@router.get("/healthz", tags=["health"])
async def healthz() -> dict[str, str]:
    """Liveness and readiness probe endpoint.

    This endpoint is used by liveness and readiness probes. It
    returns a simple success response without checking dependencies (MinIO,
    Ray, CLIP, Qdrant).

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


@router.get("/search/images", response_model=models.ImageSearchResponse, tags=["vector-store"])
async def search_images(
    q: str,
    limit: int = 10,
    collection: str | None = None,
    provider: Annotated[
        str | None,
        Query(
            pattern="^qdrant$",
            description="Vector database provider (defaults to VECTOR_DB_PROVIDER env var)",
        ),
    ] = None,
    model: Annotated[
        str | None,
        Query(description="Override embedding model name (e.g. google/siglip-so400m-patch14-384)"),
    ] = None,
    image_provider: Annotated[
        str | None,
        Query(
            pattern="^(clip|hosted_clip|infinity)$",
            description="Override embedding provider type (auto-detected from model name if omitted)",
        ),
    ] = None,
) -> models.ImageSearchResponse:
    """Perform text-to-image search over image collections using CLIP embeddings.

    CLIP embeddings allow searching images using natural language text queries.
    Both text and image embeddings live in the same vector space, enabling
    cross-modal search (e.g., "sunset over ocean" finds matching images).

    This endpoint:
    1. Generates a CLIP text embedding for the query
    2. Searches the image collection for similar vectors
    3. Returns ranked image results with similarity scores and metadata

    Args:
        q: Natural language text query (e.g., "a fluffy cat sitting on a couch").
        limit: Maximum number of results to return (default: 10, max: 100).
        collection: Optional base collection name (defaults to service default).
        provider: Optional vector database provider (qdrant).
            If not specified, uses VECTOR_DB_PROVIDER environment variable.

    Returns:
        Response containing image search results ordered by similarity (highest first).

    Raises:
        HTTPException(400): If query is empty, limit is invalid, or provider is invalid.
        HTTPException(502): If CLIP or vector database operations fail.
        HTTPException(404): If image collection doesn't exist.

    Example Request:
        ```
        GET /search/images?q=sunset%20over%20ocean&limit=5&provider=qdrant
        GET /search/images?q=fluffy%20cat&collection=photos&provider=qdrant
        ```

    Example Response:
        ```json
        {
            "query": "sunset over ocean",
            "model": "ViT-B/32",
            "collection": "documents",
            "provider": "qdrant",
            "results": [
                {
                    "id": "d790dd2c-99eb-4901-b9c9-538b58318fe3",
                    "score": 0.8234,
                    "s3_key": "images/sunset-001.jpg",
                    "s3_uri": "s3://pipeline/images/sunset-001.jpg",
                    "format": "jpeg",
                    "width": 1920,
                    "height": 1080,
                    "thumbnail_key": "thumbnails/sunset-001.jpg"
                }
            ],
            "total": 1
        }
        ```

    Note:
        - Requires CLIP model that supports both text and image encoding
        - Results are ordered by similarity score (highest first)
        - Score ranges from 0.0 to 1.0 (1.0 = identical, 0.0 = completely different)
        - Image collection must exist (created via image ingestion job)
    """
    if not q or not q.strip():
        raise BadRequestError("Query cannot be empty")

    s = get_settings()

    # Build embedding config — apply model override when requested
    embed_config = EmbeddingConfig.from_env()
    if model:
        resolved_provider = (
            ProviderType(image_provider) if image_provider else auto_detect_image_provider(model)
        )
        embed_config = embed_config.model_copy(update=_model_override_kwargs(model, resolved_provider))

    try:
        embedding_provider = create_embedding_provider(embed_config)
    except ValueError as e:
        raise BadRequestError(str(e)) from e

    # Determine provider type: use query parameter if provided, otherwise use settings
    if provider:
        try:
            vector_db_config = VectorDBConfig.from_env_for_provider(
                provider_type=provider.lower(),
            )
        except ValueError as e:
            raise BadRequestError(str(e)) from e
        provider_type = vector_db_config.provider_type
    else:
        vector_db_config = s.vector_db
        provider_type = vector_db_config.provider_type

    vector_db_provider = create_vector_db_provider(vector_db_config)
    collection_name = collection or vector_db_config.collection

    # Validate vector dimensions when model is overridden
    if model:
        try:
            col_info = await vector_db_provider.get_collection_info_async(collection=collection_name)
            if col_info.vector_size > 0 and col_info.vector_size != embedding_provider.vector_size:
                raise BadRequestError(
                    f"Vector size mismatch: model '{model}' produces "
                    f"{embedding_provider.vector_size}-d vectors but collection "
                    f"'{collection_name}' expects {col_info.vector_size}-d vectors"
                )
        except BadRequestError:
            raise
        except Exception:
            logger.debug("Skipping vector-size validation — collection info unavailable", exc_info=True)

    image_search_service = ImageSearchService(
        embedding_provider=embedding_provider,
        vector_db_provider=vector_db_provider,
        embedding_provider_name=embed_config.provider_type,
        vector_db_provider_name=provider_type,
        cache=get_semantic_cache(),
    )

    search_results = await image_search_service.search_images_async(
        collection=collection_name,
        query=q,
        limit=limit,
    )

    # Convert SearchResultItem to Pydantic ImageSearchResult
    pydantic_results: list[models.ImageSearchResult] = []
    for item in search_results.items:
        payload = item.payload
        pydantic_results.append(
            models.ImageSearchResult(
                id=item.point_id,
                score=item.score,
                s3_key=payload.get("s3_key", ""),  # pyright: ignore[reportArgumentType]
                s3_uri=payload.get("s3_uri", ""),  # pyright: ignore[reportArgumentType]
                format=payload.get("format"),  # pyright: ignore[reportArgumentType]
                width=payload.get("width"),  # pyright: ignore[reportArgumentType]
                height=payload.get("height"),  # pyright: ignore[reportArgumentType]
                thumbnail_key=payload.get("thumbnail_key"),  # pyright: ignore[reportArgumentType]
            )
        )

    return models.ImageSearchResponse(
        query=q,
        model=embedding_provider.model_name,
        collection=collection_name,
        provider=provider_type,
        results=pydantic_results,
        total=search_results.total,
    )


@router.post(
    "/ray/jobs/images",
    response_model=models.RayImageJobResponse,
    status_code=202,
    tags=["ray-jobs"],
)
async def submit_ray_image_job(req: models.RayImageJobRequest) -> models.RayImageJobResponse:
    """Submit a new Ray job to process S3 images.

    This submits a job to the Ray cluster and returns immediately.
    Use GET /ray/jobs/{job_id} to check status and GET /ray/jobs/{job_id}/logs for logs.

    The job will:
    1. List objects under the given S3 bucket and prefix
    2. Filter by image extensions (.jpg, .jpeg, .png, .webp, .gif)
    3. Preprocess and embed images via CLIP
    4. Store vectors in Qdrant image collections

    Args:
        req: Request containing s3_bucket, s3_prefix, and collection.

    Returns:
        Job metadata including Ray-generated job ID and submission timestamp.

    Raises:
        HTTPException(500): If job submission fails.

    Example Request:
        ```json
        POST /ray/jobs/images
        {
            "s3_bucket": "pipeline",
            "s3_prefix": "images/",
            "collection": "documents"
        }
        ```

    Example Response:
        ```json
        {
            "job_id": "raysubmit_1234567890",
            "status": "submitted",

            "s3_bucket": "pipeline",
            "s3_prefix": "images/",
            "collection": "documents",
            "submitted_at": "2026-01-12T15:30:45.123456Z"
        }
        ```
    """
    try:
        ray_service = RayService()
        minio_cfg = MinIOConfig.from_env()

        job_id = await asyncio.to_thread(
            ray_service.submit_image_job,
            s3_endpoint=minio_cfg.endpoint_url,
            s3_access_key_id=minio_cfg.access_key_id,
            s3_secret_access_key=minio_cfg.secret_access_key,
            s3_bucket=req.s3_bucket,
            s3_prefix=req.s3_prefix,
            collection=req.collection,
            image_max_items=req.image_max_items,
            image_page_size=req.image_page_size,
        )

        return models.RayImageJobResponse(
            job_id=job_id,
            status="submitted",
            s3_bucket=req.s3_bucket,
            s3_prefix=req.s3_prefix,
            collection=req.collection,
            image_max_items=req.image_max_items,
            submitted_at=datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        )
    except (PipelineError, UpstreamError):
        raise
    except Exception as e:
        raise PipelineError(f"Failed to submit Ray image job: {e!s}") from e


@router.post(
    "/databricks/jobs/images",
    response_model=models.DatabricksImageJobResponse,
    status_code=202,
    tags=["databricks-jobs"],
)
async def submit_databricks_image_job(
    req: models.DatabricksImageJobRequest,
) -> models.DatabricksImageJobResponse:
    """Submit a new Databricks job to process images.

    This submits a run for a preconfigured Databricks Job and returns immediately.
    The job processes images from either S3/MinIO or iNaturalist Open Data,
    generates embeddings using CLIP or Infinity, and stores vectors in Qdrant.

    Args:
        req: Request containing source, optional s3_prefix, and collection.
            Supported sources:
            - s3: Process images from S3/MinIO bucket at s3_prefix
            - inat: Process images from iNaturalist Open Data dataset

    Returns:
        Job metadata including Databricks run ID and submission timestamp.

    Raises:
        HTTPException(400): If source is invalid or required parameters are missing.
        HTTPException(502): If Databricks API is unreachable.
        HTTPException(500): If job submission fails.

    Example Request (S3 source):
        ```json
        POST /databricks/jobs/images
        {
            "source": "s3",
            "s3_prefix": "images/",
            "collection": "documents"
        }
        ```

    Example Request (iNaturalist source):
        ```json
        POST /databricks/jobs/images
        {
            "source": "inat",
            "collection": "inat-images"
        }
        ```

    Example Response:
        ```json
        {
            "run_id": "123456789",
            "status": "submitted",

            "source": "s3",
            "s3_prefix": "images/",
            "collection": "documents",
            "submitted_at": "2026-03-26T14:15:00.123456Z"
        }
        ```
    """
    try:
        databricks_service = DatabricksRayService()
        embed_config = EmbeddingConfig.from_env()
        effective_s3_prefix: str | None
        if req.source == "inat":
            effective_s3_prefix = None
            run_id = await asyncio.to_thread(
                databricks_service.submit_inat_image_job,
                embedding_config=embed_config,
                collection=req.collection,
            )
        else:
            effective_s3_prefix = req.s3_prefix or ""
            minio_cfg = MinIOConfig.from_env()
            run_id = await asyncio.to_thread(
                databricks_service.submit_image_job,
                s3_endpoint=minio_cfg.endpoint_url,
                s3_access_key_id=minio_cfg.access_key_id,
                s3_secret_access_key=minio_cfg.secret_access_key,
                s3_bucket=minio_cfg.bucket,
                s3_prefix=effective_s3_prefix,
                embedding_config=embed_config,
                collection=req.collection,
                image_max_items=req.image_max_items,
                image_page_size=req.image_page_size,
            )

        return models.DatabricksImageJobResponse(
            run_id=str(run_id),
            status="submitted",
            source=req.source,
            s3_prefix=effective_s3_prefix,
            collection=req.collection,
            submitted_at=datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        )
    except (PipelineError, UpstreamError):
        raise
    except Exception as e:
        raise PipelineError(f"Failed to submit Databricks image job: {e!s}") from e


@router.post(
    "/databricks/jobs/cdc-producer",
    response_model=models.DatabricksCdcProducerJobResponse,
    status_code=202,
    tags=["databricks-jobs"],
)
async def submit_databricks_cdc_producer_job() -> models.DatabricksCdcProducerJobResponse:
    """Submit a Databricks CDC producer job to stream S3 changes to Bronze storage.

    This endpoint submits a Databricks job that uses Auto Loader to detect new or
    modified files in the configured S3 bucket and streams them to a Bronze Delta table.
    The Auto Loader provides exactly-once ingestion guarantees using file metadata
    checkpointing.

    Returns:
        DatabricksCdcProducerJobResponse: Job submission confirmation with run ID.

    Raises:
        HTTPException(500): If Databricks job submission fails.

    Example Response:
        ```json
        {
            "run_id": "123456789",
            "status": "submitted",

            "submitted_at": "2026-03-26T15:30:45.123456Z"
        }
        ```
    """
    try:
        databricks_service = DatabricksRayService()
        run_id = await asyncio.to_thread(databricks_service.submit_s3_autoloader_job)

        return models.DatabricksCdcProducerJobResponse(
            run_id=str(run_id),
            status="submitted",
            submitted_at=str(datetime.now(timezone.utc).isoformat()),  # noqa: UP017
        )
    except (PipelineError, UpstreamError):
        raise
    except Exception as e:
        raise PipelineError(f"Failed to submit Databricks CDC producer job: {e!s}") from e


@router.post(
    "/ray/jobs/process-dlq",
    response_model=models.RayProcessDLQResponse,
    status_code=202,
    tags=["ray-jobs"],
)
async def process_dlq_entries_ray() -> models.RayProcessDLQResponse:
    """Process previously failed image ingestions from the dead letter queue using Ray.

    This endpoint submits a Ray job to retry processing images that failed during
    initial ingestion. Failed images are stored in the DLQ (dead letter queue) with
    error metadata for debugging and retry. This allows recovery from transient
    failures without re-processing successful images.

    The job pulls failed entries from the DLQ backend (currently Redis) and
    attempts to re-embed and re-index each failed image.

    Returns:
        RayProcessDLQResponse: Job submission confirmation with Ray job ID.

    Raises:
        HTTPException(500): If Ray job submission fails.

    Example Response:
        ```json
        {
            "job_id": "raysubmit_1234567890",
            "status": "submitted",

            "submitted_at": "2026-03-26T15:30:45.123456Z"
        }
        ```
    """
    try:
        settings = get_settings()

        ray_service = RayService()
        minio_cfg = MinIOConfig.from_env()

        job_id = await asyncio.to_thread(
            ray_service.submit_image_job,
            s3_endpoint=minio_cfg.endpoint_url,
            s3_access_key_id=minio_cfg.access_key_id,
            s3_secret_access_key=minio_cfg.secret_access_key,
            s3_bucket=minio_cfg.bucket,
            collection=settings.vector_db.collection,
            pull_from_dlq=True,
        )

        return models.RayProcessDLQResponse(
            job_id=job_id,
            status="submitted",
            submitted_at=datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        )
    except (PipelineError, UpstreamError):
        raise
    except Exception as e:
        raise PipelineError(f"Failed to submit Ray process DLQ job: {e!s}") from e


@router.post(
    "/databricks/jobs/process-dlq",
    response_model=models.DatabricksProcessDLQResponse,
    status_code=202,
    tags=["databricks-jobs"],
)
async def process_dlq_entries_databricks() -> models.DatabricksProcessDLQResponse:
    """Process previously failed image ingestions from the DLQ using Databricks.

    This endpoint submits a Databricks job (Ray on Spark cluster) to retry processing
    images that failed during initial ingestion. Failed images are stored in the DLQ
    (dead letter queue) with error metadata for debugging and retry. This allows
    recovery from transient failures without re-processing successful images.

    The job uses Ray running on a Databricks Spark cluster to pull failed entries
    from the DLQ backend (currently Redis) and attempts to re-embed and re-index
    each failed image. This provides the same retry semantics as the Ray endpoint
    but with Databricks' cluster management and monitoring.

    Returns:
        DatabricksProcessDLQResponse: Job submission confirmation with Databricks run ID.

    Raises:
        HTTPException(500): If Databricks job submission fails.

    Example Response:
        ```json
        {
            "run_id": "123456789",
            "status": "submitted",

            "submitted_at": "2026-03-26T15:30:45.123456Z"
        }
        ```
    """
    try:
        settings = get_settings()

        databricks_service = DatabricksRayService()
        embed_config = EmbeddingConfig.from_env()
        minio_cfg = MinIOConfig.from_env()
        run_id = await asyncio.to_thread(
            databricks_service.submit_image_job,
            s3_endpoint=minio_cfg.endpoint_url,
            s3_access_key_id=minio_cfg.access_key_id,
            s3_secret_access_key=minio_cfg.secret_access_key,
            s3_bucket=minio_cfg.bucket,
            embedding_config=embed_config,
            collection=settings.vector_db.collection,
            pull_from_dlq=True,
        )

        return models.DatabricksProcessDLQResponse(
            run_id=str(run_id),
            status="submitted",
            submitted_at=datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        )
    except (PipelineError, UpstreamError):
        raise
    except Exception as e:
        raise PipelineError(f"Failed to submit Databricks process DLQ job: {e!s}") from e


@router.post(
    "/databricks/jobs/cdc-consumer",
    response_model=models.DatabricksCdcConsumerJobResponse,
    status_code=202,
    tags=["databricks-jobs"],
)
async def submit_databricks_cdc_consumer_job() -> models.DatabricksCdcConsumerJobResponse:
    """Submit a Databricks CDC consumer job to process Bronze images with Ray.

    This endpoint submits a Databricks job that reads images from the Bronze Delta
    table (populated by the CDC producer), generates embeddings using the configured
    embedding provider, and indexes them in the vector database. This is the second
    stage of the CDC pipeline: Bronze → Embedding → Vector DB.

    The job uses Ray running on a Databricks Spark cluster to process images in
    parallel, providing scalable ingestion for large datasets.

    Returns:
        DatabricksCdcConsumerJobResponse: Job submission confirmation with run ID.

    Raises:
        HTTPException(500): If Databricks job submission fails.

    Example Response:
        ```json
        {
            "run_id": "123456789",
            "status": "submitted",

            "submitted_at": "2026-03-26T15:30:45.123456Z"
        }
        ```
    """
    try:
        settings = get_settings()

        databricks_service = DatabricksRayService()
        minio_cfg = MinIOConfig.from_env()
        embed_config = EmbeddingConfig.from_env()
        run_id = await asyncio.to_thread(
            databricks_service.submit_s3_bronze_image_job,
            s3_endpoint=minio_cfg.endpoint_url,
            s3_access_key_id=minio_cfg.access_key_id,
            s3_secret_access_key=minio_cfg.secret_access_key,
            s3_bucket=minio_cfg.bucket,
            embedding_config=embed_config,
            collection=settings.vector_db.collection,
        )

        return models.DatabricksCdcConsumerJobResponse(
            run_id=str(run_id),
            status="submitted",
            submitted_at=datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        )
    except (PipelineError, UpstreamError):
        raise
    except Exception as e:
        raise PipelineError(f"Failed to submit Databricks CDC consumer job: {e!s}") from e


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
        await asyncio.to_thread(databricks_service.stop_run, run_id)
        return models.DatabricksJobStopResponse(run_id=str(run_id), status="stopped")
    except (PipelineError, UpstreamError):
        raise
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
        status = await asyncio.to_thread(databricks_service.get_run_status, run_id)
        return models.DatabricksJobStatusResponse(run_id=str(run_id), **status)
    except (PipelineError, UpstreamError):
        raise
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
        logs = await asyncio.to_thread(databricks_service.get_run_output, run_id)
        return models.DatabricksJobLogsResponse(run_id=str(run_id), logs=logs)
    except (PipelineError, UpstreamError):
        raise
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
        ray_service = RayService()
        status = await asyncio.to_thread(ray_service.get_job_status, job_id)

        return models.RayJobStatusResponse(job_id=job_id, **status)
    except (PipelineError, UpstreamError):
        raise
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
        ray_service = RayService()
        logs = await asyncio.to_thread(ray_service.get_job_logs, job_id)

        return models.RayJobLogsResponse(job_id=job_id, logs=logs)
    except (PipelineError, UpstreamError):
        raise
    except Exception as e:
        raise PipelineError(f"Failed to get job logs: {e!s}") from e


@router.delete("/cache", tags=["cache"])
async def bust_cache() -> dict[str, str]:
    """Invalidate the entire semantic cache on demand.

    Returns:
        Confirmation dict: ``{"status": "cache invalidated"}``.

    Raises:
        HTTPException(400): If the semantic cache is not enabled.
    """
    cache = get_semantic_cache()
    if cache is None:
        raise BadRequestError("Semantic cache is not enabled")
    await cache.invalidate()
    return {"status": "cache invalidated"}


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
        ray_service = RayService()
        await asyncio.to_thread(ray_service.stop_job, job_id)

        return models.RayJobStopResponse(job_id=job_id, status="stopped")
    except (PipelineError, UpstreamError):
        raise
    except Exception as e:
        raise PipelineError(f"Failed to stop job: {e!s}") from e


@router.post(INGESTION_METRICS_PATH, tags=["metrics"])
async def submit_ingestion_metrics(payload: models.IngestionMetricsPayload) -> dict[str, str]:
    """Record ingestion job metrics into the Prometheus registry.

    Ingestion jobs (Ray, Databricks) run as ephemeral processes that exit after
    completion => metrics must be forwarded here to survive the job process exit.

    Args:
        payload: Batch stats and optional checkpoint flag from the job process.

    Returns:
        ``{"status": "ok"}`` on success.
    """
    if payload.successful > 0:
        INGESTION_DOCS_PROCESSED.labels(status="success", pipeline=payload.pipeline).inc(payload.successful)
    if payload.failed > 0:
        INGESTION_DOCS_PROCESSED.labels(status="failed", pipeline=payload.pipeline).inc(payload.failed)
    if payload.batch_duration_seconds > 0:
        INGESTION_BATCH_DURATION.labels(pipeline=payload.pipeline).observe(payload.batch_duration_seconds)
    if payload.checkpoint_save:
        INGESTION_CHECKPOINT_SAVES.labels(pipeline=payload.pipeline).inc()
    return {"status": "ok"}
