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
from logging.config import dictConfig

import attrs
from databricks.sdk import WorkspaceClient

from config import DatabricksRayJobConfig, EmbeddingConfig, ImageEmbeddingConfig
from core.exceptions import UpstreamError
from core.services.ingestion_params import (
    add_ray_tuning_env,
    build_image_ingestion_env,
    build_inat_image_ingestion_env,
    build_ingestion_env,
)
from foundation.logger import LOGGING_CONFIG

dictConfig(LOGGING_CONFIG)

logger = logging.getLogger("pipeline.databricks.ray.service")


@attrs.define(frozen=True, slots=True)
class DatabricksRayService:
    """Service for submitting Ray ingestion jobs via Databricks Jobs API.

    Example:
        ```python
        from core.services.databricks_ray_service import DatabricksRayService
        from config import EmbeddingConfig

        service = DatabricksRayService()
        run_id = service.submit_s3_to_vector_dbs(
            namespace="ml-system",
            s3_endpoint="http://minio.ml-system:9000",
            s3_access_key_id="minioadmin",
            s3_secret_access_key="minioadmin",
            s3_bucket="pipeline",
            s3_prefix="inputs/",
            embedding_config=EmbeddingConfig.from_env(),
            collection="documents",
        )
        ```
    """

    @staticmethod
    def _submit_python_job(
        *,
        host: str,
        token: str,
        job_id: int,
        python_params: list[str],
    ) -> int:
        """Submit a Databricks python task and return run_id."""
        client = WorkspaceClient(host=host, token=token)
        response = client.jobs.run_now(
            job_id=job_id,
            python_params=python_params,
        )
        return int(response.run_id)

    def submit_s3_to_vector_dbs(
        self,
        *,
        namespace: str,
        s3_endpoint: str,
        s3_access_key_id: str,
        s3_secret_access_key: str,
        s3_bucket: str,
        s3_prefix: str = "inputs/",
        embedding_config: EmbeddingConfig,
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
            collection: Vector DB collection name.

        Returns:
            Databricks run ID for the submitted job.

        Raises:
            UpstreamError: If submission fails.
        """
        databricks_config = DatabricksRayJobConfig.from_env()
        env_vars = build_ingestion_env(
            namespace=namespace,
            s3_endpoint=s3_endpoint,
            s3_access_key_id=s3_access_key_id,
            s3_secret_access_key=s3_secret_access_key,
            s3_bucket=s3_bucket,
            s3_prefix=s3_prefix,
            embedding_config=embedding_config,
            collection=collection,
            extra_env_keys=("INATINQ_SRC_DIR",),
        )
        add_ray_tuning_env(env_vars)
        python_params = [f"{key}={value}" for key, value in env_vars.items()]

        try:
            logger.info(
                "Submitting Databricks job",
                extra={"job_id": databricks_config.job_id, "s3_prefix": s3_prefix},
            )
            return self._submit_python_job(
                host=databricks_config.host,
                token=databricks_config.token,
                job_id=databricks_config.job_id,
                python_params=python_params,
            )
        except Exception as e:
            logger.exception("Failed to submit Databricks job", extra={"error": str(e)})
            raise UpstreamError(f"Failed to submit Databricks job: {e}") from e

    def submit_image_job(
        self,
        *,
        namespace: str,
        s3_endpoint: str,
        s3_access_key_id: str,
        s3_secret_access_key: str,
        s3_bucket: str,
        s3_prefix: str = "",
        image_embedding_config: ImageEmbeddingConfig,
        collection: str,
        image_max_items: int | None = None,
    ) -> int:
        """Submit a Databricks job to process S3 images and store embeddings.

        Parameters are passed as env-style `KEY=VALUE` strings so the Databricks
        entrypoint can reuse the same configuration logic as Ray workers.

        Args:
            namespace: Kubernetes namespace (used for service discovery).
            s3_endpoint: S3 service endpoint URL.
            s3_access_key_id: S3 access key.
            s3_secret_access_key: S3 secret key.
            s3_bucket: S3 bucket name to read from.
            s3_prefix: S3 prefix to filter objects (default: "" for bucket root).
            image_embedding_config: Image embedding provider configuration.
            collection: Vector DB base collection name.
            image_max_items: Optional limit on number of images to process.

        Returns:
            Databricks run ID for the submitted job.

        Raises:
            UpstreamError: If submission fails.
        """
        databricks_config = DatabricksRayJobConfig.from_env()
        env_vars = build_image_ingestion_env(
            namespace=namespace,
            s3_endpoint=s3_endpoint,
            s3_access_key_id=s3_access_key_id,
            s3_secret_access_key=s3_secret_access_key,
            s3_bucket=s3_bucket,
            s3_prefix=s3_prefix,
            image_embedding_config=image_embedding_config,
            collection=collection,
            extra_env_keys=("INATINQ_SRC_DIR",),
        )
        add_ray_tuning_env(env_vars)
        if image_max_items is not None:
            env_vars["IMAGE_MAX_ITEMS"] = str(image_max_items)
        python_params = [f"{key}={value}" for key, value in env_vars.items()]

        try:
            logger.info(
                "Submitting Databricks image job",
                extra={"job_id": databricks_config.job_id, "s3_prefix": s3_prefix},
            )
            return self._submit_python_job(
                host=databricks_config.host,
                token=databricks_config.token,
                job_id=databricks_config.job_id,
                python_params=python_params,
            )
        except Exception as e:
            logger.exception("Failed to submit Databricks image job", extra={"error": str(e)})
            raise UpstreamError(f"Failed to submit Databricks image job: {e}") from e

    def submit_inat_image_job(
        self,
        *,
        namespace: str,
        image_embedding_config: ImageEmbeddingConfig,
        collection: str,
        s3_prefix: str = "images/",
    ) -> int:
        """Submit a Databricks job to process iNaturalist images.

        Uses `DATABRICKS_INAT_JOB_ID` as the first-class job selector.
        """
        databricks_config = DatabricksRayJobConfig.from_env()
        if databricks_config.inat_job_id is None:
            raise ValueError("Missing required Databricks config: DATABRICKS_INAT_JOB_ID")

        env_vars = build_inat_image_ingestion_env(
            namespace=namespace,
            image_embedding_config=image_embedding_config,
            collection=collection,
            extra_env_keys=("INATINQ_SRC_DIR",),
        )
        add_ray_tuning_env(env_vars)
        python_params = [f"{key}={value}" for key, value in env_vars.items()]

        try:
            logger.info(
                "Submitting Databricks iNaturalist image job",
                extra={"job_id": databricks_config.inat_job_id, "s3_prefix": s3_prefix},
            )
            return self._submit_python_job(
                host=databricks_config.host,
                token=databricks_config.token,
                job_id=databricks_config.inat_job_id,
                python_params=python_params,
            )
        except Exception as e:
            logger.exception("Failed to submit Databricks iNaturalist image job", extra={"error": str(e)})
            raise UpstreamError(f"Failed to submit Databricks iNaturalist image job: {e}") from e

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
