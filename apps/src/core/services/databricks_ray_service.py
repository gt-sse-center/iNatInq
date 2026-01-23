"""Service layer for submitting Ray ingestion jobs to Databricks.

This module provides a service class for submitting Ray ingestion jobs using
the Databricks Jobs API. It mirrors the RayService parameter construction and
invokes run_now() on a pre-configured job definition.

## Design

- **Submit jobs**: Calls WorkspaceClient.jobs.run_now to launch a job run.
- **Parameter parity**: Uses the same env-style parameters as Ray jobs.
- **Error handling**: Wraps Databricks SDK errors in UpstreamError.
"""

import logging
import os

import attrs
from databricks.sdk import WorkspaceClient

from config import DatabricksRayJobConfig, EmbeddingConfig, VectorDBConfig
from core.exceptions import UpstreamError

logger = logging.getLogger("pipeline.databricks.ray.service")


@attrs.define(frozen=True, slots=True)
class DatabricksRayService:
    """Service for submitting Ray ingestion jobs via Databricks Jobs API.

    Example:
        ```python
        from core.services.databricks_ray_service import DatabricksRayService
        from config import EmbeddingConfig, VectorDBConfig

        service = DatabricksRayService()
        run_id = service.submit_s3_to_qdrant(
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

    def _build_python_params(
        self,
        *,
        namespace: str,
        s3_endpoint: str,
        s3_access_key_id: str,
        s3_secret_access_key: str,
        s3_bucket: str,
        s3_prefix: str,
        embedding_config: EmbeddingConfig,
        collection: str,
    ) -> list[str]:
        """Build python_params for Databricks Jobs API from env-style values."""
        params: list[tuple[str, str]] = [
            ("K8S_NAMESPACE", namespace),
            ("S3_PREFIX", s3_prefix),
            ("S3_ENDPOINT", s3_endpoint),
            ("S3_ACCESS_KEY_ID", s3_access_key_id),
            ("S3_SECRET_ACCESS_KEY", s3_secret_access_key),
            ("S3_BUCKET", s3_bucket),
            ("VECTOR_DB_COLLECTION", collection),
            ("EMBEDDING_PROVIDER_TYPE", embedding_config.provider_type),
        ]

        if embedding_config.vector_size is not None:
            params.append(("EMBEDDING_VECTOR_SIZE", str(embedding_config.vector_size)))
        if embedding_config.ollama_url:
            params.append(("OLLAMA_BASE_URL", embedding_config.ollama_url))
        if embedding_config.ollama_model:
            params.append(("OLLAMA_MODEL", embedding_config.ollama_model))

        optional_env_keys = (
            "QDRANT_URL",
            "QDRANT_API_KEY",
            "WEAVIATE_URL",
            "WEAVIATE_API_KEY",
            "WEAVIATE_GRPC_HOST",
        )
        for key in optional_env_keys:
            value = os.getenv(key)
            if value:
                params.append((key, value))

        return [f"{key}={value}" for key, value in params]

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
    ) -> int:
        """Submit a Databricks job to process S3 data and store embeddings.

        This method submits a run for a configured Databricks Job. Parameters
        are passed as env-style `KEY=VALUE` strings so the job entrypoint can
        use the same configuration logic as Ray workers.

        Args:
            namespace: Kubernetes namespace (used for service discovery).
            s3_endpoint: S3 service endpoint URL.
            s3_access_key_id: S3 access key.
            s3_secret_access_key: S3 secret key.
            s3_bucket: S3 bucket name to read from.
            s3_prefix: S3 prefix to filter objects (default: `inputs/`).
            embedding_config: Embedding provider configuration.
            vector_db_config: Vector database configuration (kept for parity;
                provider-specific env vars are sourced from the environment).
            collection: Vector DB collection name.

        Returns:
            Databricks run ID for the submitted job.

        Raises:
            UpstreamError: If submission fails.
        """
        databricks_config = DatabricksRayJobConfig.from_env()
        python_params = self._build_python_params(
            namespace=namespace,
            s3_endpoint=s3_endpoint,
            s3_access_key_id=s3_access_key_id,
            s3_secret_access_key=s3_secret_access_key,
            s3_bucket=s3_bucket,
            s3_prefix=s3_prefix,
            embedding_config=embedding_config,
            collection=collection,
        )

        try:
            client = WorkspaceClient(host=databricks_config.host, token=databricks_config.token)
            logger.info(
                "Submitting Databricks job",
                extra={"job_id": databricks_config.job_id, "s3_prefix": s3_prefix},
            )
            response = client.jobs.run_now(
                job_id=databricks_config.job_id,
                python_params=python_params,
            )
            return int(response.run_id)
        except Exception as e:
            logger.exception("Failed to submit Databricks job", extra={"error": str(e)})
            raise UpstreamError(f"Failed to submit Databricks job: {e}") from e

    def stop_run(self, run_id: int | str) -> None:
        """Stop a running Databricks job run.

        Args:
            run_id: Databricks run ID to cancel.

        Raises:
            UpstreamError: If the cancellation fails.
        """
        databricks_config = DatabricksRayJobConfig.from_env()
        try:
            client = WorkspaceClient(host=databricks_config.host, token=databricks_config.token)
            client.jobs.cancel_run(run_id=int(run_id))
        except Exception as e:
            logger.exception("Failed to stop Databricks job", extra={"error": str(e), "run_id": run_id})
            raise UpstreamError(f"Failed to stop Databricks job: {e}") from e

    def get_run_status(self, run_id: int | str) -> dict[str, str | None]:
        """Get Databricks run status details.

        Args:
            run_id: Databricks run ID to query.

        Returns:
            Dictionary with run status fields.

        Raises:
            UpstreamError: If the status query fails.
        """
        databricks_config = DatabricksRayJobConfig.from_env()
        try:
            client = WorkspaceClient(host=databricks_config.host, token=databricks_config.token)
            run = client.jobs.get_run(run_id=int(run_id))
            state = getattr(run, "state", None)
            return {
                "life_cycle_state": getattr(state, "life_cycle_state", None),
                "result_state": getattr(state, "result_state", None),
                "state_message": getattr(state, "state_message", None),
            }
        except Exception as e:
            logger.exception("Failed to get Databricks run status", extra={"error": str(e), "run_id": run_id})
            raise UpstreamError(f"Failed to get Databricks run status: {e}") from e

    def get_run_output(self, run_id: int | str) -> str:
        """Get Databricks run output/logs.

        Args:
            run_id: Databricks run ID to query.

        Returns:
            Run output/logs as a string (best-effort).

        Raises:
            UpstreamError: If output retrieval fails.
        """
        databricks_config = DatabricksRayJobConfig.from_env()
        try:
            client = WorkspaceClient(host=databricks_config.host, token=databricks_config.token)
            output = client.jobs.get_run_output(run_id=int(run_id))

            # Best-effort extraction across task types.
            for attr in ("logs", "error"):
                value = getattr(output, attr, None)
                if value:
                    return str(value)
            notebook_output = getattr(output, "notebook_output", None)
            if notebook_output is not None:
                result = getattr(notebook_output, "result", None)
                if result:
                    return str(result)
            return ""
        except Exception as e:
            logger.exception("Failed to get Databricks run output", extra={"error": str(e), "run_id": run_id})
            raise UpstreamError(f"Failed to get Databricks run output: {e}") from e
