"""Pydantic request/response models for the pipeline service API.

This module defines all HTTP request and response schemas used by the FastAPI endpoints.
All models use Pydantic for automatic validation, serialization, and OpenAPI schema generation.

## Request Models

- `EmbedRequest`: Request to generate embeddings for one or more texts.
- `UpsertRequest`: Request to embed texts and upsert them into vector database.
- `ProcessRequest`: Request to process S3 data and store embeddings in vector database.

## Response Models

- `EmbedResponse`: Contains generated embeddings and the model used.
- `UpsertResponse`: Confirms successful upsert with collection and point count.
- `SearchResponse`: Contains semantic search results from vector database (Qdrant or Weaviate).
- `ProcessResponse`: Confirms successful Spark job completion.

## Validation

Pydantic automatically validates:
- Required fields are present
- Field types match (e.g., `List[str]` for texts)
- Field constraints (e.g., `min_length=1` ensures non-empty lists)
- Optional fields can be omitted or set to `None`
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EmbedRequest(BaseModel):
    """Request to generate embeddings for one or more text strings.

    Attributes:
        texts: Non-empty list of text strings to embed. Each text will produce
            one embedding vector. The vector dimension depends on the model.
        model: Optional Ollama model name override. If omitted, uses the service's
            default model (configured via `OLLAMA_MODEL` env var).

    Example:
        ```json
        {
            "texts": ["hello world", "foo bar"],
            "model": "nomic-embed-text"
        }
        ```
    """

    texts: list[str] = Field(min_length=1, description="Non-empty list of texts to embed")
    model: str | None = Field(None, description="Optional model name override")


class EmbedResponse(BaseModel):
    """Response containing generated embeddings for the input texts.

    Attributes:
        model: The Ollama model name that was used to generate embeddings.
        embeddings: List of embedding vectors, one per input text. Each vector
            is a list of floats. The vector dimension depends on the model
            (e.g., `nomic-embed-text` produces 768-dimensional vectors).

    Example:
        ```json
        {
            "model": "nomic-embed-text",
            "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]]
        }
        ```
    """

    model: str = Field(description="Model name used for embedding generation")
    embeddings: list[list[float]] = Field(description="One embedding vector per input text")


class UpsertRequest(BaseModel):
    """Request to embed texts and upsert the resulting vectors into Qdrant.

    This endpoint performs two operations:
    1. Generates embeddings for each text via Ollama
    2. Upserts the vectors (with metadata) into the specified Qdrant collection

    Attributes:
        texts: Non-empty list of text strings to embed and upsert.
        collection: Optional Qdrant collection name. If omitted, uses the service's
            default collection (configured via `QDRANT_COLLECTION` env var).
        metadata: Optional list of metadata dictionaries, one per text. If provided,
            must have the same length as `texts`. Each metadata dict is merged into
            the point/object payload (the original `text` is always included).
        model: Optional Ollama model name override. If omitted, uses the service's
            default model.

    Example:
        ```json
        {
            "texts": ["document 1", "document 2"],
            "collection": "my-docs",
            "metadata": [{"source": "file1.txt"}, {"source": "file2.txt"}],
            "model": "nomic-embed-text"
        }
        ```

    Raises:
        HTTPException(400): If `metadata` is provided but its length doesn't match `texts`.
    """

    texts: list[str] = Field(min_length=1, description="Non-empty list of texts to embed and upsert")
    collection: str | None = Field(None, description="Optional Qdrant collection name")
    metadata: list[dict[str, Any]] | None = Field(
        None, description="Optional metadata dicts (one per text, same length as texts)"
    )
    model: str | None = Field(None, description="Optional model name override")


class UpsertResponse(BaseModel):
    """Response confirming successful upsert of vectors into Qdrant.

    Attributes:
        collection: The Qdrant collection name where vectors were upserted.
        model: The Ollama model name used to generate embeddings.
        points_upserted: Number of points (vectors) successfully upserted into Qdrant.
            This should equal the number of input texts.

    Example:
        ```json
        {
            "collection": "documents",
            "model": "nomic-embed-text",
            "points_upserted": 2
        }
        ```
    """

    collection: str = Field(description="Qdrant collection name")
    model: str = Field(description="Model name used for embedding generation")
    points_upserted: int = Field(description="Number of points upserted into Qdrant")


class SearchResult(BaseModel):
    """A single search result from a vector database.

    Attributes:
        id: Point/object ID (UUID string from Qdrant or Weaviate).
        score: Similarity score (0.0 to 1.0, higher is more similar).
        text: Original text stored in the point/object payload.
        metadata: Additional metadata from the payload (excluding 'text').
    """

    id: str = Field(description="Point/object ID (UUID string)")
    score: float = Field(ge=0.0, le=1.0, description="Similarity score")
    text: str = Field(description="Original text from payload")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class SearchResponse(BaseModel):
    """Response containing semantic search results.

    Attributes:
        query: The original query string.
        model: The Ollama model used for embedding generation.
        collection: The collection that was searched.
        provider: The vector database provider that was used ("qdrant" or "weaviate").
        results: List of search results, ordered by similarity (highest first).
        total: Total number of results found
            (may be greater than results.length if limit was applied).
    """

    query: str = Field(description="Original query string")
    model: str = Field(description="Model used for embedding")
    collection: str = Field(description="Collection that was searched")
    provider: str = Field(description="Vector database provider used (qdrant or weaviate)")
    results: list[SearchResult] = Field(description="Search results ordered by similarity")
    total: int = Field(ge=0, description="Total number of results found")


class LoadLocalToS3Request(BaseModel):
    """Request to load local files from persistent volume to S3/MinIO.

    Attributes:
        local_dir: Local directory path containing files (default: `/data/testdata/inputs`).
        s3_prefix: Optional S3 prefix to upload to (default: `inputs/`).
        timeout_s: Optional timeout in seconds (default: 1800 for large datasets).

    Example:
        ```json
        {
            "local_dir": "/data/testdata/inputs",
            "s3_prefix": "inputs/",
            "timeout_s": 1800
        }
        ```
    """

    local_dir: str | None = Field(
        default="/data/testdata/inputs", description="Local directory containing files to upload"
    )
    s3_prefix: str | None = Field(default="inputs/", description="S3 prefix to upload to")
    timeout_s: int | None = Field(default=1800, ge=60, le=7200, description="Timeout in seconds (60-7200)")


class LoadLocalToS3Response(BaseModel):
    """Response confirming Spark job submission for loading local files to S3.

    Attributes:
        job_name: The Kubernetes Job name created for the Spark job.
        local_dir: The local directory that was processed.
        s3_prefix: The S3 prefix where files were uploaded.
        status: Job status (always "submitted" if endpoint returns successfully).

    Example:
        ```json
        {
            "job_name": "load-local-to-s3-abc12345",
            "local_dir": "/data/testdata/inputs",
            "s3_prefix": "inputs/",
            "status": "submitted"
        }
        ```
    """

    job_name: str = Field(description="Kubernetes Job name")
    local_dir: str = Field(description="Local directory that was processed")
    s3_prefix: str = Field(description="S3 prefix where files were uploaded")
    status: str = Field(description="Job status")


# ═══════════════════════════════════════════════════════════════════════════════
# Spark Job Management Models
# ═══════════════════════════════════════════════════════════════════════════════


class SparkJobRequest(BaseModel):
    """Request to submit a Spark processing job.

    Attributes:
        s3_prefix: S3 prefix to process (e.g., "inputs/").
        collection: Vector DB collection name.
        executor_instances: Number of Spark executors (parallelism level).
        executor_memory: Memory per executor (e.g., "512m", "1g", "2g").
        job_name: Optional custom job name (auto-generated if omitted).

    Example:
        ```json
        {
            "s3_prefix": "inputs/",
            "collection": "documents",
            "executor_instances": 2,
            "executor_memory": "1g"
        }
        ```
    """

    s3_prefix: str = Field(..., example="inputs/", description="S3 prefix to process")
    collection: str = Field(..., example="documents", description="Vector DB collection")
    executor_instances: int = Field(1, ge=1, le=10, description="Number of executors")
    executor_memory: str = Field("512m", pattern=r"^\d+(m|g|M|G)$", description="Memory per executor")
    job_name: str | None = Field(None, description="Optional custom job name")


class SparkJobResponse(BaseModel):
    """Response after submitting a Spark job.

    Attributes:
        job_name: Auto-generated or custom job name.
        status: Job status (always "submitted" on success).
        namespace: Kubernetes namespace where job runs.
        s3_prefix: S3 prefix being processed.
        collection: Target vector DB collection.
        submitted_at: ISO 8601 timestamp when job was submitted.

    Example:
        ```json
        {
            "job_name": "s3-to-vector-db-20260112-153045-a1b2c3d4",
            "status": "submitted",
            "namespace": "ml-system",
            "s3_prefix": "inputs/",
            "collection": "documents",
            "submitted_at": "2026-01-12T15:30:45.123456Z"
        }
        ```
    """

    job_name: str
    status: str
    namespace: str
    s3_prefix: str
    collection: str
    submitted_at: str


class SparkJobStatusResponse(BaseModel):
    """Status response for a Spark job.

    Attributes:
        job_name: Name of the job.
        state: Application state (SUBMITTED, RUNNING, COMPLETED, FAILED, etc.).
        spark_state: Spark-specific state.
        driver_info: Information about the driver pod.
        execution_attempts: Number of execution attempts.
        last_submission_attempt_time: ISO 8601 timestamp of last submission.
        termination_time: ISO 8601 timestamp when job terminated (null if running).

    Example:
        ```json
        {
            "job_name": "s3-to-vector-db-20260112-153045-a1b2c3d4",
            "state": "RUNNING",
            "spark_state": "RUNNING",
            "driver_info": {"podName": "s3-to-vector-db-...-driver"},
            "execution_attempts": 1,
            "last_submission_attempt_time": "2026-01-12T15:30:45Z",
            "termination_time": null
        }
        ```
    """

    job_name: str
    state: str
    spark_state: str
    driver_info: dict[str, Any]
    execution_attempts: int
    last_submission_attempt_time: str | None
    termination_time: str | None


class SparkJobSummary(BaseModel):
    """Summary of a Spark job for list responses.

    Attributes:
        job_name: Name of the job.
        state: Current application state.
        created_at: ISO 8601 timestamp when job was created.

    Example:
        ```json
        {
            "job_name": "s3-to-vector-db-20260112-153045-a1b2c3d4",
            "state": "COMPLETED",
            "created_at": "2026-01-12T15:30:45Z"
        }
        ```
    """

    job_name: str
    state: str
    created_at: str | None


class SparkJobListResponse(BaseModel):
    """Response listing all Spark jobs.

    Attributes:
        jobs: List of job summaries.
        total: Total number of jobs.

    Example:
        ```json
        {
            "jobs": [
                {
                    "job_name": "s3-to-vector-db-20260112-153045-a1b2c3d4",
                    "state": "COMPLETED",
                    "created_at": "2026-01-12T15:30:45Z"
                }
            ],
            "total": 1
        }
        ```
    """

    jobs: list[SparkJobSummary]
    total: int


class SparkJobDeleteResponse(BaseModel):
    """Response after deleting a Spark job.

    Attributes:
        job_name: Name of the deleted job.
        status: Deletion status (always "deleted" on success).

    Example:
        ```json
        {
            "job_name": "s3-to-vector-db-20260112-153045-a1b2c3d4",
            "status": "deleted"
        }
        ```
    """

    job_name: str
    status: str


# ═══════════════════════════════════════════════════════════════════════════════
# Ray Job Management Models
# ═══════════════════════════════════════════════════════════════════════════════


class RayJobRequest(BaseModel):
    """Request to submit a Ray processing job.

    Attributes:
        s3_prefix: S3 prefix to process (e.g., "inputs/").
        collection: Vector DB collection name.

    Example:
        ```json
        {
            "s3_prefix": "inputs/",
            "collection": "documents"
        }
        ```
    """

    s3_prefix: str = Field(..., example="inputs/", description="S3 prefix to process")
    collection: str = Field(..., example="documents", description="Vector DB collection")


class RayJobResponse(BaseModel):
    """Response after submitting a Ray job.

    Attributes:
        job_id: Ray-generated job ID (e.g., "raysubmit_1234567890").
        status: Job status (always "submitted" on success).
        namespace: Kubernetes namespace where Ray cluster runs.
        s3_prefix: S3 prefix being processed.
        collection: Target vector DB collection.
        submitted_at: ISO 8601 timestamp when job was submitted.

    Example:
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

    job_id: str
    status: str
    namespace: str
    s3_prefix: str
    collection: str
    submitted_at: str


class RayJobStatusResponse(BaseModel):
    """Response containing Ray job status details.

    Attributes:
        job_id: Ray job ID.
        status: Current job status (PENDING, RUNNING, SUCCEEDED, FAILED, STOPPED).
        message: Optional error message if job failed.

    Example:
        ```json
        {
            "job_id": "raysubmit_1234567890",
            "status": "RUNNING",
            "message": null
        }
        ```
    """

    job_id: str
    status: str
    message: str | None = None


class RayJobLogsResponse(BaseModel):
    r"""Response containing Ray job logs.

    Attributes:
        job_id: Ray job ID.
        logs: Job logs as string.

    Example:
        ```json
        {
            "job_id": "raysubmit_1234567890",
            "logs": "Processing 1000 documents...\nCompleted successfully."
        }
        ```
    """

    job_id: str
    logs: str


class RayJobStopResponse(BaseModel):
    """Response after stopping a Ray job.

    Attributes:
        job_id: Ray job ID that was stopped.
        status: Stop status (always "stopped" on success).

    Example:
        ```json
        {
            "job_id": "raysubmit_1234567890",
            "status": "stopped"
        }
        ```
    """

    job_id: str
    status: str
