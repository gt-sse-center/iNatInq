"""Service for managing Spark-based data processing jobs.

This module provides a high-level service layer for submitting and managing
Spark jobs via the Kubernetes Spark Operator.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, UTC

from clients.k8s_spark import SparkJobClient

logger = logging.getLogger("pipeline.spark_service")


class SparkService:
    """Service for managing Spark-based data processing jobs.

    This service orchestrates Spark job submissions through the Kubernetes
    Spark Operator, providing a high-level API for processing S3 documents
    into vector databases.

    Example:
        >>> service = SparkService(namespace="ml-system")
        >>> result = service.submit_processing_job(
        ...     s3_prefix="inputs/",
        ...     collection="documents",
        ...     executor_instances=2,
        ... )
        >>> print(result["job_name"])
        s3-to-vector-db-20260112-153045-a1b2c3d4
    """

    def __init__(self, namespace: str = "ml-system") -> None:
        """Initialize Spark service.

        Args:
            namespace: Kubernetes namespace for Spark jobs.
        """
        self.client = SparkJobClient(namespace=namespace)
        self.namespace = namespace

    def submit_processing_job(
        self,
        s3_prefix: str,
        collection: str,
        executor_instances: int = 1,
        executor_memory: str = "512m",
        job_name: str | None = None,
    ) -> dict[str, str]:
        """Submit a Spark job to process S3 documents into vector DB.

        Creates a SparkApplication that processes documents from S3/MinIO,
        generates embeddings via Ollama, and stores them in Qdrant/Weaviate.

        Args:
            s3_prefix: S3 prefix to process (e.g., "inputs/").
            collection: Vector DB collection name.
            executor_instances: Number of Spark executors (parallelism).
            executor_memory: Memory per executor (e.g., "512m", "1g").
            job_name: Optional custom job name (auto-generated if None).

        Returns:
            Job metadata including name, status, and submission time:
            {
                "job_name": "s3-to-vector-db-20260112-153045-a1b2c3d4",
                "status": "submitted",
                "namespace": "ml-system",
                "s3_prefix": "inputs/",
                "collection": "documents",
                "submitted_at": "2026-01-12T15:30:45.123456Z"
            }

        Raises:
            RuntimeError: If job submission fails.
        """
        # Generate unique job name if not provided
        if not job_name:
            timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            job_name = f"s3-to-vector-db-{timestamp}-{unique_id}"

        logger.info(
            "Submitting Spark job",
            extra={
                "job_name": job_name,
                "s3_prefix": s3_prefix,
                "collection": collection,
                "executors": executor_instances,
                "executor_memory": executor_memory,
            },
        )

        try:
            self.client.submit_job(
                name=job_name,
                s3_prefix=s3_prefix,
                collection=collection,
                executor_instances=executor_instances,
                executor_memory=executor_memory,
            )

            return {
                "job_name": job_name,
                "status": "submitted",
                "namespace": self.namespace,
                "s3_prefix": s3_prefix,
                "collection": collection,
                "submitted_at": datetime.now(UTC).isoformat(),
            }
        except Exception as e:
            logger.exception(
                "Failed to submit Spark job",
                extra={"job_name": job_name, "error": str(e)},
            )
            raise

    def get_job_status(self, job_name: str) -> dict[str, str | int | None]:
        """Get the current status of a Spark job.

        Args:
            job_name: Name of the Spark job.

        Returns:
            Status information including:
            {
                "job_name": "s3-to-vector-db-...",
                "state": "RUNNING",  # SUBMITTED, RUNNING, COMPLETED, FAILED
                "spark_state": "RUNNING",
                "driver_info": {...},
                "execution_attempts": 1,
                "last_submission_attempt_time": "2026-01-12T15:30:45Z",
                "termination_time": None
            }

        Raises:
            RuntimeError: If job not found.
        """
        spark_app = self.client.get_job_status(job_name)

        # Extract relevant status info
        status = spark_app.get("status", {})
        app_state = status.get("applicationState", {})

        return {
            "job_name": job_name,
            "state": app_state.get("state", "UNKNOWN"),
            "spark_state": status.get("sparkApplicationState", "UNKNOWN"),
            "driver_info": status.get("driverInfo", {}),
            "execution_attempts": status.get("executionAttempts", 0),
            "last_submission_attempt_time": status.get("lastSubmissionAttemptTime"),
            "termination_time": status.get("terminationTime"),
        }

    def list_jobs(self) -> list[dict[str, str]]:
        """List all Spark jobs in the namespace.

        Returns:
            List of job summaries:
            [
                {
                    "job_name": "s3-to-vector-db-...",
                    "state": "COMPLETED",
                    "created_at": "2026-01-12T15:30:45Z"
                },
                ...
            ]
        """
        response = self.client.list_jobs()
        items = response.get("items", [])

        return [
            {
                "job_name": item["metadata"]["name"],
                "state": item.get("status", {}).get("applicationState", {}).get("state", "UNKNOWN"),
                "created_at": item["metadata"].get("creationTimestamp"),
            }
            for item in items
        ]

    def delete_job(self, job_name: str) -> dict[str, str]:
        """Delete a Spark job (cleanup after completion).

        Removes the SparkApplication resource and terminates any running pods.

        Args:
            job_name: Name of the job to delete.

        Returns:
            Deletion confirmation:
            {
                "job_name": "s3-to-vector-db-...",
                "status": "deleted"
            }

        Raises:
            RuntimeError: If deletion fails.
        """
        logger.info("Deleting Spark job", extra={"job_name": job_name})
        self.client.delete_job(job_name)
        return {"job_name": job_name, "status": "deleted"}
