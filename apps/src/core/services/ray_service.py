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

from config import EmbeddingConfig, RayJobConfig, VectorDBConfig
from core.exceptions import UpstreamError

logger = logging.getLogger("pipeline.ray.service")


@attrs.define(frozen=True, slots=True)
class RayService:
    """Service for orchestrating Ray jobs via Ray Jobs API.

    This service uses the Ray Jobs API to submit and monitor jobs running
    on a Ray cluster. Jobs are submitted directly to the Ray cluster without
    using Kubernetes Jobs.

    Example:
        ```python
        from core.services.ray_service import RayService
        from config import EmbeddingConfig, VectorDBConfig

        service = RayService()

        job_name = service.submit_s3_to_qdrant(
            namespace="ml-system",
            s3_endpoint="http://minio.ml-system:9000",
            s3_access_key_id="minioadmin",
            s3_secret_access_key="minioadmin",
            s3_bucket="pipeline",
            s3_prefix="inputs/",
            embedding_config=EmbeddingConfig.from_env(),
            vector_db_config=VectorDBConfig.from_env(),
            collection="documents",
        )
        ```
    """

    def submit_s3_to_qdrant(
        self,
        *,
        namespace: str,
        s3_endpoint: str,
        s3_access_key_id: str,
        s3_secret_access_key: str,
        s3_bucket: str,
        s3_prefix: str = "inputs/",
        embedding_config: EmbeddingConfig,
        vector_db_config: VectorDBConfig,
        collection: str,
    ) -> str:
        """Submit a Ray job to process S3 data and store embeddings in vector database.

        This method submits a job to the Ray cluster and returns immediately with a job ID.
        The job runs asynchronously on the Ray cluster.

        Args:
            namespace: Kubernetes namespace.
            s3_endpoint: S3 service endpoint URL.
            s3_access_key_id: S3 access key.
            s3_secret_access_key: S3 secret key.
            s3_bucket: S3 bucket name to read from.
            s3_prefix: S3 prefix to filter objects (default: `inputs/`).
            embedding_config: Embedding provider configuration.
            vector_db_config: Vector database provider configuration.
            collection: Collection name.

        Returns:
            Ray job ID (e.g., "raysubmit_1234567890").

        Raises:
            UpstreamError: If job submission fails.

        Example:
            ```python
            service = RayService()
            job_id = service.submit_s3_to_qdrant(
                namespace="ml-system",
                s3_endpoint="http://minio.ml-system:9000",
                s3_access_key_id="minioadmin",
                s3_secret_access_key="minioadmin",
                s3_bucket="pipeline",
                s3_prefix="inputs/",
                embedding_config=EmbeddingConfig.from_env(),
                vector_db_config=VectorDBConfig.from_env(),
                collection="documents",
            )
            # Later: check status with get_job_status(job_id)
            ```
        """
        # Get Ray configuration with dashboard address
        ray_config = RayJobConfig.from_env(namespace)
        if not ray_config.dashboard_address:
            raise UpstreamError("RAY_DASHBOARD_ADDRESS not configured. Cannot submit job to Ray cluster.")

        dashboard_address = ray_config.dashboard_address
        logger.info(
            "Submitting Ray job",
            extra={"s3_prefix": s3_prefix, "dashboard_address": dashboard_address},
        )

        # Build environment variables for the job
        env_vars = {
            "K8S_NAMESPACE": namespace,
            "S3_PREFIX": s3_prefix,
            # Use S3_* env vars to match MinIOConfig.from_env()
            "S3_ENDPOINT": s3_endpoint,
            "S3_ACCESS_KEY_ID": s3_access_key_id,
            "S3_SECRET_ACCESS_KEY": s3_secret_access_key,
            "S3_BUCKET": s3_bucket,
            "VECTOR_DB_COLLECTION": collection,
            "EMBEDDING_PROVIDER_TYPE": embedding_config.provider_type,
        }

        # Add optional config
        if embedding_config.vector_size is not None:
            env_vars["EMBEDDING_VECTOR_SIZE"] = str(embedding_config.vector_size)
        if embedding_config.ollama_url:
            env_vars["OLLAMA_BASE_URL"] = embedding_config.ollama_url
        if embedding_config.ollama_model:
            env_vars["OLLAMA_MODEL"] = embedding_config.ollama_model
        if vector_db_config.provider_type == "qdrant" and vector_db_config.qdrant_url:
            env_vars["QDRANT_URL"] = vector_db_config.qdrant_url

        try:
            # Create job submission client
            client = JobSubmissionClient(dashboard_address)

            # Submit the job with runtime environment
            # Dependencies are installed on Ray workers; code is mounted at /app/src
            job_id = client.submit_job(
                entrypoint="python -m core.ingestion.ray.process_s3_to_qdrant",
                runtime_env={
                    "env_vars": {
                        **env_vars,
                        "PYTHONPATH": "/app/src",
                    },
                    "pip": [
                        "boto3",
                        "attrs",
                        "pydantic",
                        "pydantic-settings",
                        "httpx",
                        "qdrant-client>=1.12.0,<1.13.0",
                        "weaviate-client",
                        "tenacity",
                        "pybreaker",
                    ],
                },
            )

            logger.info("Ray job submitted", extra={"job_id": job_id})

            return str(job_id)

        except Exception as e:
            logger.exception("Failed to submit Ray job", extra={"error": str(e)})
            raise UpstreamError(f"Failed to submit Ray job: {e}") from e

    def get_job_status(self, job_id: str, namespace: str) -> dict[str, Any]:
        """Get the status of a Ray job.

        Args:
            job_id: Ray job ID returned from submit_s3_to_qdrant.
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
