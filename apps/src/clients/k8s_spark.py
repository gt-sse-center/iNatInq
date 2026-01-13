"""Kubernetes client for managing Spark jobs via Spark Operator.

This module provides a Python client to programmatically create and manage
SparkApplication custom resources in Kubernetes, enabling API-driven Spark job submission.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from kubernetes import client, config
from kubernetes.client.rest import ApiException

logger = logging.getLogger("pipeline.k8s_spark")


class SparkJobClient:
    """Client for managing Spark jobs via Spark Operator.

    This client interacts with the Spark Operator's SparkApplication CRD to
    programmatically submit, monitor, and manage Spark jobs in Kubernetes.

    Example:
        >>> client = SparkJobClient(namespace="ml-system")
        >>> response = client.submit_job(
        ...     name="my-spark-job",
        ...     s3_prefix="inputs/",
        ...     collection="documents",
        ... )
        >>> status = client.get_job_status("my-spark-job")
    """

    def __init__(self, namespace: str = "ml-system") -> None:
        """Initialize Kubernetes client for SparkApplication CRDs.

        Args:
            namespace: Kubernetes namespace where Spark jobs will run.
        """
        try:
            # Try in-cluster config first (when running in K8s pod)
            config.load_incluster_config()
            logger.info("Loaded in-cluster Kubernetes configuration")
        except config.ConfigException:
            # Fall back to kubeconfig (for local development)
            config.load_kube_config()
            logger.info("Loaded kubeconfig from local machine")

        self.api = client.CustomObjectsApi()
        self.namespace = namespace
        self.group = "sparkoperator.k8s.io"
        self.version = "v1beta2"
        self.plural = "sparkapplications"

    def submit_job(
        self,
        name: str,
        s3_prefix: str,
        collection: str,
        executor_instances: int = 1,
        executor_cores: int = 1,
        executor_memory: str = "512m",
        driver_cores: int = 1,
        driver_memory: str = "512m",
        image: str = "ml/spark-job:0.1.0",
    ) -> dict[str, Any]:
        """Submit a new Spark job to process S3 → Vector DB.

        Creates a SparkApplication CRD that the Spark Operator will process,
        launching driver and executor pods to run the job.

        Args:
            name: Unique name for this Spark job (must be DNS-compliant).
            s3_prefix: S3 prefix to process (e.g., "inputs/").
            collection: Vector database collection name.
            executor_instances: Number of Spark executor pods.
            executor_cores: CPU cores per executor.
            executor_memory: Memory per executor (e.g., "512m", "2g").
            driver_cores: CPU cores for driver pod.
            driver_memory: Memory for driver pod.
            image: Docker image for Spark job.

        Returns:
            The created SparkApplication resource as a dictionary.

        Raises:
            RuntimeError: If job submission fails.
        """
        spark_app = {
            "apiVersion": f"{self.group}/{self.version}",
            "kind": "SparkApplication",
            "metadata": {
                "name": name,
                "namespace": self.namespace,
            },
            "spec": {
                "type": "Python",
                "mode": "cluster",
                "image": image,
                "imagePullPolicy": "IfNotPresent",
                "mainApplicationFile": (
                    "local:///app/src/pipeline/core/ingestion/spark/process_s3_to_qdrant.py"
                ),
                "sparkVersion": "3.5.7",
                "restartPolicy": {"type": "Never"},
                "deps": {"pyFiles": ["local:///app/src/pipeline.zip"]},
                "driver": {
                    "cores": driver_cores,
                    "memory": driver_memory,
                    "serviceAccount": "pipeline",
                    "labels": {
                        "role": "driver",
                        "app": "spark-pipeline",
                        "job-name": name,
                    },
                    "env": self._build_env_vars(s3_prefix, collection),
                },
                "executor": {
                    "cores": executor_cores,
                    "instances": executor_instances,
                    "memory": executor_memory,
                    "labels": {
                        "role": "executor",
                        "app": "spark-pipeline",
                        "job-name": name,
                    },
                    "env": [
                        {"name": "PYTHONUNBUFFERED", "value": "1"},
                    ],
                },
            },
        }

        try:
            response = self.api.create_namespaced_custom_object(
                group=self.group,
                version=self.version,
                namespace=self.namespace,
                plural=self.plural,
                body=spark_app,
            )
            logger.info(
                "Submitted Spark job",
                extra={
                    "job_name": name,
                    "namespace": self.namespace,
                    "s3_prefix": s3_prefix,
                    "collection": collection,
                    "executors": executor_instances,
                },
            )
            return response
        except ApiException as e:
            logger.exception(
                "Failed to submit Spark job",
                extra={
                    "job_name": name,
                    "error": str(e),
                    "status": e.status,
                    "reason": e.reason,
                },
            )
            raise RuntimeError(f"Failed to submit Spark job: {e}") from e

    def get_job_status(self, name: str) -> dict[str, Any]:
        """Get status of a SparkApplication.

        Args:
            name: Name of the Spark job.

        Returns:
            The SparkApplication resource including status.

        Raises:
            RuntimeError: If job not found or error occurs.
        """
        try:
            return self.api.get_namespaced_custom_object(
                group=self.group,
                version=self.version,
                namespace=self.namespace,
                plural=self.plural,
                name=name,
            )
        except ApiException as e:
            logger.exception(
                "Failed to get Spark job status",
                extra={"job_name": name, "error": str(e)},
            )
            raise RuntimeError(f"Failed to get job status: {e}") from e

    def delete_job(self, name: str) -> dict[str, Any]:
        """Delete a SparkApplication.

        This removes the SparkApplication resource and terminates any running pods.

        Args:
            name: Name of the Spark job to delete.

        Returns:
            Deletion response from Kubernetes API.

        Raises:
            RuntimeError: If deletion fails.
        """
        try:
            response = self.api.delete_namespaced_custom_object(
                group=self.group,
                version=self.version,
                namespace=self.namespace,
                plural=self.plural,
                name=name,
            )
            logger.info("Deleted Spark job", extra={"job_name": name})
            return response
        except ApiException as e:
            logger.exception(
                "Failed to delete Spark job",
                extra={"job_name": name, "error": str(e)},
            )
            raise RuntimeError(f"Failed to delete job: {e}") from e

    def list_jobs(self) -> dict[str, Any]:
        """List all SparkApplications in the namespace.

        Returns:
            Dictionary containing list of SparkApplication resources.
        """
        try:
            return self.api.list_namespaced_custom_object(
                group=self.group,
                version=self.version,
                namespace=self.namespace,
                plural=self.plural,
            )
        except ApiException as e:
            logger.exception(
                "Failed to list Spark jobs",
                extra={"namespace": self.namespace, "error": str(e)},
            )
            raise RuntimeError(f"Failed to list jobs: {e}") from e

    def _build_env_vars(self, s3_prefix: str, collection: str) -> list[dict[str, str]]:
        """Build environment variables for the Spark job.

        Args:
            s3_prefix: S3 prefix to process.
            collection: Vector database collection name.

        Returns:
            List of environment variable dictionaries for K8s pod spec.
        """
        return [
            # S3/MinIO configuration
            {"name": "S3_ENDPOINT", "value": f"http://minio.{self.namespace}:9000"},
            {"name": "S3_ACCESS_KEY_ID", "value": os.getenv("S3_ACCESS_KEY_ID", "minioadmin")},
            {
                "name": "S3_SECRET_ACCESS_KEY",
                "value": os.getenv("S3_SECRET_ACCESS_KEY", "minioadmin"),
            },
            {"name": "S3_BUCKET", "value": os.getenv("S3_BUCKET", "pipeline")},
            {"name": "S3_PREFIX", "value": s3_prefix},
            # Embedding configuration
            {"name": "EMBEDDING_PROVIDER", "value": "ollama"},
            {"name": "OLLAMA_BASE_URL", "value": f"http://ollama.{self.namespace}:11434"},
            {"name": "OLLAMA_MODEL", "value": "nomic-embed-text"},
            {"name": "EMBEDDING_VECTOR_SIZE", "value": "768"},
            # Vector database configuration
            {"name": "VECTOR_DB_PROVIDER", "value": "qdrant"},
            {"name": "VECTOR_DB_COLLECTION", "value": collection},
            {"name": "QDRANT_URL", "value": f"http://qdrant.{self.namespace}:6333"},
            {"name": "WEAVIATE_URL", "value": f"http://weaviate.{self.namespace}:8080"},
            # Kubernetes configuration
            {"name": "K8S_NAMESPACE", "value": self.namespace},
            {
                "name": "POD_IP",
                "valueFrom": {"fieldRef": {"fieldPath": "status.podIP"}},
            },
            # Python configuration
            {"name": "PYTHONPATH", "value": "/app/src"},
            {"name": "PYSPARK_PYTHON", "value": "/usr/local/bin/python3"},
            {"name": "PYSPARK_DRIVER_PYTHON", "value": "/usr/local/bin/python3"},
            {"name": "PYTHONUNBUFFERED", "value": "1"},
            # Spark job configuration
            {"name": "SPARK_EXECUTOR_MEMORY", "value": "512m"},
            {"name": "SPARK_EXECUTOR_CORES", "value": "1"},
            {"name": "SPARK_DRIVER_MEMORY", "value": "512m"},
        ]
