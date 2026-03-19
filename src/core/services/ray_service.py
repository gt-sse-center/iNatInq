"""Service layer for Ray job orchestration using Ray Jobs API.

This module provides a service class for submitting and monitoring Ray jobs
using the Ray Jobs API, which provides proper job lifecycle management,
non-blocking submission, and built-in status tracking.

## Design

The service layer uses Ray Jobs API to:
- **Submit jobs**: Non-blocking submission to Ray cluster
- **Track status**: Query job status without blocking
- **Retrieve logs**: Access job logs through Ray API
- **Error handling**: Translates Ray errors into `PipelineError` hierarchy

This allows the API to return immediately while Ray cluster manages job execution.
"""

import logging
from typing import Any

import attrs
from ray.job_submission import JobSubmissionClient

from config import EmbeddingConfig, RayJobConfig
from foundation.exceptions import UpstreamError
from core.services.ingestion_params import build_image_ingestion_env

logger = logging.getLogger("pipeline.ray.service")


@attrs.define(frozen=True, slots=True)
class RayService:
    """Service for orchestrating Ray jobs via Ray Jobs API.

    This service uses the Ray Jobs API to submit and monitor jobs running
    on a Ray cluster. Jobs are submitted directly to the Ray cluster without
    using Kubernetes Jobs.
    """

    def submit_image_job(
        self,
        *,
        namespace: str,
        s3_endpoint: str,
        s3_access_key_id: str,
        s3_secret_access_key: str,
        s3_bucket: str,
        s3_prefix: str = "",
        collection: str,
        embedding_config: EmbeddingConfig | None = None,
        pull_from_dlq: bool = False,
        image_max_items: int | None = None,
        image_page_size: int | None = None,
    ) -> str:
        """Submit a Ray job to process S3 images and store embeddings in vector DB image collections.

        Lists objects under s3_bucket/s3_prefix, filters by image extensions
        (.jpg, .jpeg, .png, .webp, .gif), and runs the image pipeline on Ray workers.
        Returns immediately with a job ID; use get_job_status(job_id) for status.

        Args:
            namespace: Kubernetes namespace.
            s3_endpoint: S3 service endpoint URL.
            s3_access_key_id: S3 access key.
            s3_secret_access_key: S3 secret key.
            s3_bucket: S3 bucket name containing images.
            s3_prefix: S3 prefix to process (default: "" for bucket root).
            collection: Base collection name.
            embedding_config: Image embedding configuration. If None, loaded from env.
            pull_from_dlq: Optional bool indicating image keys should be pulled from the dead letter queue.
            image_max_items: Optional limit on number of images to process.
            image_page_size: Optional S3 listing page size override.

        Returns:
            Ray job ID (e.g., "raysubmit_1234567890").

        Raises:
            UpstreamError: If job submission fails.
        """
        ray_config = RayJobConfig.from_env(namespace)
        if not ray_config.dashboard_address:
            raise UpstreamError(
                "RAY_DASHBOARD_ADDRESS not configured. Cannot submit image job to Ray cluster."
            )

        if embedding_config is None:
            embedding_config = EmbeddingConfig.from_env(namespace)

        dashboard_address = ray_config.dashboard_address
        logger.info(
            "Submitting Ray image job",
            extra={"s3_bucket": s3_bucket, "s3_prefix": s3_prefix, "dashboard_address": dashboard_address},
        )

        env_vars = build_image_ingestion_env(
            namespace=namespace,
            s3_endpoint=s3_endpoint,
            s3_access_key_id=s3_access_key_id,
            s3_secret_access_key=s3_secret_access_key,
            s3_bucket=s3_bucket,
            s3_prefix=s3_prefix,
            embedding_config=embedding_config,
            collection=collection,
            pull_from_dlq=pull_from_dlq,
        )

        if image_max_items is not None:
            env_vars["IMAGE_MAX_ITEMS"] = str(image_max_items)
        if image_page_size is not None:
            env_vars["IMAGE_PAGE_SIZE"] = str(image_page_size)

        try:
            client = JobSubmissionClient(dashboard_address)
            job_id = client.submit_job(
                entrypoint="python -m core.ingestion.ray.process_s3_images",
                runtime_env={
                    "env_vars": {
                        **env_vars,
                        "PYTHONPATH": "/app/src",
                    },
                    "pip": [
                        "boto3",
                        "filetype",
                        "redis",
                        "attrs",
                        "pydantic",
                        "pydantic-settings",
                        "httpx",
                        "qdrant-client==1.16.1",
                        "weaviate-client",
                        "tenacity",
                        "pybreaker",
                        "aiobreaker",
                        "pillow",
                        "filetype>=1.2.0",
                        "redis",
                    ],
                },
            )
            logger.info("Ray image job submitted", extra={"job_id": job_id})
            return str(job_id)
        except Exception as e:
            logger.exception("Failed to submit Ray image job", extra={"error": str(e)})
            raise UpstreamError(f"Failed to submit Ray image job: {e}") from e

    def get_job_status(self, job_id: str, namespace: str) -> dict[str, Any]:
        """Get the status of a Ray job.

        Args:
            job_id: Ray job ID returned from submit_s3_to_vector_dbs.
            namespace: Kubernetes namespace (used for config resolution).

        Returns:
            Dictionary with job status information:
            - status: str (PENDING, RUNNING, SUCCEEDED, FAILED, STOPPED)
            - message: str (optional error message)

        Raises:
            UpstreamError: If status check fails or dashboard not configured.

        Example:
            ```python
            status = service.get_job_status(job_id, "ml-system")
            if status["status"] == "SUCCEEDED":
                print("Job completed!")
            ```
        """
        ray_config = RayJobConfig.from_env(namespace)
        if not ray_config.dashboard_address:
            raise UpstreamError("RAY_DASHBOARD_ADDRESS not configured.")
        dashboard_address = ray_config.dashboard_address

        try:
            client = JobSubmissionClient(dashboard_address)
            status = client.get_job_status(job_id)
            info = client.get_job_info(job_id)

            return {
                "status": getattr(status, "value", str(status)),
                "message": getattr(info, "message", None) if info else None,
            }

        except Exception as e:
            logger.exception("Failed to get Ray job status", extra={"job_id": job_id})
            raise UpstreamError(f"Failed to get job status: {e}") from e

    def get_job_logs(self, job_id: str, namespace: str) -> str:
        """Get the logs from a Ray job.

        Args:
            job_id: Ray job ID.
            namespace: Kubernetes namespace (used for config resolution).

        Returns:
            Job logs as a string.

        Raises:
            UpstreamError: If log retrieval fails or dashboard not configured.
        """
        ray_config = RayJobConfig.from_env(namespace)
        if not ray_config.dashboard_address:
            raise UpstreamError("RAY_DASHBOARD_ADDRESS not configured.")
        dashboard_address = ray_config.dashboard_address

        try:
            client = JobSubmissionClient(dashboard_address)
            logs = client.get_job_logs(job_id)
            return str(logs)

        except Exception as e:
            logger.exception("Failed to get Ray job logs", extra={"job_id": job_id})
            raise UpstreamError(f"Failed to get job logs: {e}") from e

    def stop_job(self, job_id: str, namespace: str) -> None:
        """Stop a running Ray job.

        Args:
            job_id: Ray job ID.
            namespace: Kubernetes namespace (used for config resolution).

        Raises:
            UpstreamError: If stopping the job fails or dashboard not configured.
        """
        ray_config = RayJobConfig.from_env(namespace)
        if not ray_config.dashboard_address:
            raise UpstreamError("RAY_DASHBOARD_ADDRESS not configured.")
        dashboard_address = ray_config.dashboard_address

        try:
            client = JobSubmissionClient(dashboard_address)
            client.stop_job(job_id)
            logger.info("Ray job stopped", extra={"job_id": job_id})

        except Exception as e:
            logger.exception("Failed to stop Ray job", extra={"job_id": job_id})
            raise UpstreamError(f"Failed to stop job: {e}") from e
