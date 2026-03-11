"""Pydantic request/response models for the pipeline service API.

This module defines all HTTP request and response schemas used by the FastAPI endpoints.
All models use Pydantic for automatic validation, serialization, and OpenAPI schema generation.

## Validation

Pydantic automatically validates:
- Required fields are present
- Field types match (e.g., `List[str]` for texts)
- Field constraints (e.g., `min_length=1` ensures non-empty lists)
- Optional fields can be omitted or set to `None`
"""

from typing import Literal

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════════════
# Image Search Models
# ═══════════════════════════════════════════════════════════════════════════════


class ImageSearchResult(BaseModel):
    """A single image search result from a vector database.

    Attributes:
        id: Point/object ID (UUID string from Qdrant or Weaviate).
        score: Similarity score (0.0 to 1.0, higher is more similar).
        s3_key: S3 object key for the image.
        s3_uri: Full S3 URI (e.g., s3://bucket/key).
        format: Image format (jpeg, png, webp, gif).
        width: Image width in pixels (if available).
        height: Image height in pixels (if available).
        thumbnail_key: S3 key for thumbnail version (if available).
    """

    id: str = Field(description="Point/object ID (UUID string)")
    score: float = Field(ge=0.0, le=1.0, description="Similarity score")
    s3_key: str = Field(description="S3 object key for the image")
    s3_uri: str = Field(description="Full S3 URI")
    format: str | None = Field(default=None, description="Image format")
    width: int | None = Field(default=None, description="Image width in pixels")
    height: int | None = Field(default=None, description="Image height in pixels")
    thumbnail_key: str | None = Field(default=None, description="S3 key for thumbnail")


class ImageSearchResponse(BaseModel):
    """Response containing text-to-image search results.

    CLIP embeddings allow searching images using text queries (e.g., "sunset over ocean"
    finds matching images). Both text and image embeddings live in the same vector space.

    Attributes:
        query: The original text query string.
        model: The CLIP model used for text embedding.
        collection: The image collection that was searched.
        provider: The vector database provider that was used ("qdrant" or "weaviate").
        results: List of image search results, ordered by similarity (highest first).
        total: Total number of results found.
    """

    query: str = Field(description="Original text query string")
    model: str = Field(description="CLIP model used for text embedding")
    collection: str = Field(description="Image collection that was searched")
    provider: str = Field(description="Vector database provider used (qdrant or weaviate)")
    results: list[ImageSearchResult] = Field(description="Image results ordered by similarity")
    total: int = Field(ge=0, description="Total number of results found")


# ═══════════════════════════════════════════════════════════════════════════════
# Ray Job Management Models
# ═══════════════════════════════════════════════════════════════════════════════


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


class RayImageJobRequest(BaseModel):
    """Request to submit a Ray image ingestion job.

    Attributes:
        s3_bucket: S3 bucket containing images.
        s3_prefix: S3 prefix to process (e.g., "images/").
        collection: Base collection name.
        image_max_items: Optional limit on number of images to process.
        image_page_size: Optional S3 listing page size override.

    Example:
        ```json
        {
            "s3_bucket": "pipeline",
            "s3_prefix": "images/",
            "collection": "documents"
        }
        ```
    """

    s3_bucket: str = Field(
        ..., json_schema_extra={"example": "pipeline"}, description="S3 bucket containing images"
    )
    s3_prefix: str = Field(
        default="",
        json_schema_extra={"example": "images/"},
        description="S3 prefix to process (empty string for bucket root)",
    )
    collection: str = Field(
        ..., json_schema_extra={"example": "documents"}, description="Vector DB base collection name"
    )
    image_max_items: int | None = Field(
        default=None,
        gt=0,
        description="Optional limit on number of images to process",
    )
    image_page_size: int | None = Field(
        default=None,
        gt=0,
        description="Optional S3 listing page size override",
    )


class RayImageJobResponse(BaseModel):
    """Response after submitting a Ray image job.

    Attributes:
        job_id: Ray-generated job ID (e.g., "raysubmit_1234567890").
        status: Job status (always "submitted" on success).
        namespace: Kubernetes namespace where Ray cluster runs.
        s3_bucket: S3 bucket being processed.
        s3_prefix: S3 prefix being processed.
        collection: Target base collection name.
        submitted_at: ISO 8601 timestamp when job was submitted.

    Example:
        ```json
        {
            "job_id": "raysubmit_1234567890",
            "status": "submitted",
            "namespace": "ml-system",
            "s3_bucket": "pipeline",
            "s3_prefix": "images/",
            "collection": "documents",
            "submitted_at": "2026-01-12T15:30:45.123456Z"
        }
        ```
    """

    job_id: str
    status: str
    namespace: str
    s3_bucket: str
    s3_prefix: str
    collection: str
    image_max_items: int | None = None
    submitted_at: str


# ═══════════════════════════════════════════════════════════════════════════════
# Databricks Job Management Models
# ═══════════════════════════════════════════════════════════════════════════════


class DatabricksImageJobRequest(BaseModel):
    """Request to submit a Databricks image processing job.

    Attributes:
        source: Image source to ingest ("s3" or "inat").
        s3_prefix: Optional S3 prefix to process (empty string means bucket root).
        collection: Vector DB base collection name.
        image_max_items: Optional limit on number of images to process.
        image_page_size: Optional S3 listing page size override.

    Example:
        ```json
        {
            "source": "s3",
            "s3_prefix": "images/",
            "collection": "documents"
        }
        ```
    """

    source: Literal["s3", "inat"] = Field(
        "s3",
        description="Image source type",
    )
    s3_prefix: str | None = Field(
        "",
        json_schema_extra={"example": "images/"},
        description="Optional S3 prefix to process (empty string for bucket root)",
    )
    collection: str = Field(
        ...,
        json_schema_extra={"example": "documents"},
        description="Vector DB base collection name",
    )
    image_max_items: int | None = Field(
        default=None,
        gt=0,
        description="Optional limit on number of images to process",
    )
    image_page_size: int | None = Field(
        default=None,
        gt=0,
        description="Optional S3 listing page size override",
    )


class DatabricksImageJobResponse(BaseModel):
    """Response after submitting a Databricks image job.

    Attributes:
        run_id: Databricks run ID.
        status: Job status (always "submitted" on success).
        namespace: Kubernetes namespace used for config resolution.
        source: Image source selected for this run.
        s3_prefix: S3 prefix being processed (present when source is "s3").
        collection: Target vector DB base collection.
        submitted_at: ISO 8601 timestamp when job was submitted.

    Example:
        ```json
        {
            "run_id": "123456789",
            "status": "submitted",
            "namespace": "ml-system",
            "source": "s3",
            "s3_prefix": "images/",
            "collection": "documents",
            "submitted_at": "2026-01-12T15:30:45.123456Z"
        }
        ```
    """

    run_id: str
    status: str
    namespace: str
    source: Literal["s3", "inat"]
    s3_prefix: str | None
    collection: str
    submitted_at: str


class DatabricksCdcProducerJobResponse(BaseModel):
    """Response after submitting a Databricks CDC producer job."""

    run_id: str
    status: str
    namespace: str
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
