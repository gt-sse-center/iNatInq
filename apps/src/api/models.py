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
- `RayJobResponse`: Confirms successful Ray job submission.

## Validation

Pydantic automatically validates:
- Required fields are present
- Field types match (e.g., `List[str]` for texts)
- Field constraints (e.g., `min_length=1` ensures non-empty lists)
- Optional fields can be omitted or set to `None`
"""

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


# ═══════════════════════════════════════════════════════════════════════════════
# Databricks Job Management Models
# ═══════════════════════════════════════════════════════════════════════════════


class DatabricksJobRequest(BaseModel):
    """Request to submit a Databricks processing job.

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


class DatabricksJobResponse(BaseModel):
    """Response after submitting a Databricks job.

    Attributes:
        run_id: Databricks run ID.
        status: Job status (always "submitted" on success).
        namespace: Kubernetes namespace used for config resolution.
        s3_prefix: S3 prefix being processed.
        collection: Target vector DB collection.
        submitted_at: ISO 8601 timestamp when job was submitted.

    Example:
        ```json
        {
            "run_id": "123456789",
            "status": "submitted",
            "namespace": "ml-system",
            "s3_prefix": "inputs/",
            "collection": "documents",
            "submitted_at": "2026-01-12T15:30:45.123456Z"
        }
        ```
    """

    run_id: str
    status: str
    namespace: str
    s3_prefix: str
    collection: str
    submitted_at: str


class DatabricksJobStopResponse(BaseModel):
    """Response after stopping a Databricks job run.

    Attributes:
        run_id: Databricks run ID that was stopped.
        status: Stop status (always "stopped" on success).
    """

    run_id: str
    status: str


class DatabricksJobStatusResponse(BaseModel):
    """Response containing Databricks run status details.

    Attributes:
        run_id: Databricks run ID.
        life_cycle_state: Run lifecycle state (e.g., RUNNING, TERMINATED).
        result_state: Run result state (e.g., SUCCESS, FAILED) if available.
        state_message: Optional state message from Databricks.
    """

    run_id: str
    life_cycle_state: str | None
    result_state: str | None
    state_message: str | None


class DatabricksJobLogsResponse(BaseModel):
    """Response containing Databricks run output/logs.

    Attributes:
        run_id: Databricks run ID.
        logs: Run output/logs (best-effort).
    """

    run_id: str
    logs: str


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
