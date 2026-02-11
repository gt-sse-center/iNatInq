"""Databricks Ray cluster strategy.

This strategy initializes Ray on a Databricks cluster using ray.util.spark.
Used for running ingestion jobs on Azure Databricks.
"""

from __future__ import annotations

import inspect
import logging
import os
from pathlib import Path
from typing import Any

import attrs
import ray

from config import RayJobConfig, VectorDBConfig

logger = logging.getLogger("pipeline.ray.strategy.databricks")

# Environment variables to pass through to Ray workers
_PASSTHROUGH_ENV_VARS = (
    "K8S_NAMESPACE",
    "S3_PREFIX",
    "S3_ENDPOINT",
    "S3_ACCESS_KEY_ID",
    "S3_SECRET_ACCESS_KEY",
    "S3_BUCKET",
    "S3_REGION",
    "S3_USE_SSL",
    "S3_PATH_STYLE",
    "S3_TIMEOUT",
    "S3_MAX_RETRIES",
    "S3_RETRY_MIN_WAIT",
    "S3_RETRY_MAX_WAIT",
    "S3_CIRCUIT_BREAKER_THRESHOLD",
    "S3_CIRCUIT_BREAKER_TIMEOUT",
    "VECTOR_DB_PROVIDER",
    "VECTOR_DB_COLLECTION",
    "QDRANT_URL",
    "QDRANT_API_KEY",
    "QDRANT_TIMEOUT",
    "QDRANT_CIRCUIT_BREAKER_THRESHOLD",
    "QDRANT_CIRCUIT_BREAKER_TIMEOUT",
    "WEAVIATE_URL",
    "WEAVIATE_API_KEY",
    "WEAVIATE_GRPC_HOST",
    "WEAVIATE_GRPC_PORT",
    "WEAVIATE_TIMEOUT",
    "WEAVIATE_CIRCUIT_BREAKER_THRESHOLD",
    "WEAVIATE_CIRCUIT_BREAKER_TIMEOUT",
    "EMBEDDING_PROVIDER",
    "OLLAMA_BASE_URL",
    "OLLAMA_MODEL",
    "OLLAMA_TIMEOUT",
    "OLLAMA_MAX_RETRIES",
    "OLLAMA_RETRY_MIN_WAIT",
    "OLLAMA_RETRY_MAX_WAIT",
    "OLLAMA_CIRCUIT_BREAKER_THRESHOLD",
    "OLLAMA_CIRCUIT_BREAKER_TIMEOUT",
    "IMAGE_EMBEDDING_PROVIDER",
    "CLIP_URL",
    "CLIP_MODEL",
    "CLIP_BACKEND",
    "CLIP_API_KEY",
    "CLIP_TIMEOUT",
    "CLIP_CIRCUIT_BREAKER_THRESHOLD",
    "CLIP_CIRCUIT_BREAKER_TIMEOUT",
    "CLIP_MAX_BATCH_SIZE",
    "CLIP_VECTOR_SIZE",
    "IMAGE_BATCH_SIZE",
    "IMAGE_MAX_SIZE_MB",
    "IMAGE_TARGET_SIZE",
    "INAT_IMAGE_SIZE",
    "INAT_MAX_ROWS",
    "INAT_METADATA_URL",
    "INAT_PHOTO_BASE_URL",
    "INAT_TIMEOUT_S",
    "INAT_CB_FAILURE_THRESHOLD",
    "INAT_CB_RECOVERY_TIMEOUT_S",
    "INATINQ_SRC_DIR",
)


@attrs.define
class DatabricksStrategy:
    """Strategy for running Ray on Databricks Spark clusters.

    This strategy uses ray.util.spark to initialize Ray on Databricks.
    It manages both the Ray-on-Spark cluster and the client connection.

    Attributes:
        _config: Ray job configuration.
        _cluster: Ray cluster handle from setup_ray_cluster().

    Example:
        ```python
        config = RayJobConfig.from_env()
        strategy = DatabricksStrategy(config)
        strategy.init()
        # ... use Ray ...
        strategy.shutdown()
        ```
    """

    _config: RayJobConfig
    _cluster: Any = attrs.field(default=None, init=False)

    @classmethod
    def from_env(cls, namespace: str = "ml-system") -> DatabricksStrategy:
        """Create strategy from environment variables.

        Args:
            namespace: Kubernetes namespace for config resolution.

        Returns:
            Configured DatabricksStrategy instance.
        """
        return cls(config=RayJobConfig.from_env(namespace))

    @property
    def config(self) -> RayJobConfig:
        """Return the Ray job configuration."""
        return self._config

    def get_runtime_env(self) -> dict[str, Any]:
        """Return runtime environment for Ray workers on Databricks.

        Builds runtime env with:
        - Environment variables passed through from driver
        - Working directory set to INATINQ_SRC_DIR

        Returns:
            Runtime env dict for ray.init().

        Raises:
            RuntimeError: If INATINQ_SRC_DIR is not set.
        """
        runtime_env: dict[str, Any] = {}
        env_vars: dict[str, str] = {}

        # Pass through PYTHONPATH if set
        pythonpath = os.environ.get("PYTHONPATH")
        if pythonpath:
            env_vars["PYTHONPATH"] = pythonpath

        # Pass through all configured env vars
        for key in _PASSTHROUGH_ENV_VARS:
            value = os.environ.get(key)
            if value is not None and value != "":
                env_vars[key] = value

        targets = VectorDBConfig.parse_targets_from_env()
        env_vars["VECTOR_DB_TARGETS"] = ",".join(sorted(targets))
        if env_vars:
            runtime_env["env_vars"] = env_vars

        # Set working directory from INATINQ_SRC_DIR
        workspace_src = os.environ.get("INATINQ_SRC_DIR")
        if not workspace_src:
            raise RuntimeError(
                "INATINQ_SRC_DIR is not set. Please export INATINQ_SRC_DIR to the repo src path "
                "(e.g. /Workspace/Users/<user>/iNatInq/src) before running this job."
            )

        if Path(workspace_src).is_dir():
            runtime_env["working_dir"] = workspace_src
        else:
            logger.warning(
                "Ray working_dir not found",
                extra={"working_dir": workspace_src},
            )

        return runtime_env

    def init(self) -> None:
        """Initialize Ray on Databricks Spark cluster.

        Sets up Ray-on-Spark cluster and initializes client connection.

        Raises:
            RuntimeError: If ray.util.spark is not available or setup fails.
        """
        try:
            from ray.util.spark import MAX_NUM_WORKER_NODES, setup_ray_cluster
        except ImportError as e:
            raise RuntimeError(
                "ray.util.spark is required to run on Databricks. Ensure Ray is installed with Spark support."
            ) from e

        # Setup Ray-on-Spark cluster
        self._cluster = self._setup_spark_cluster(setup_ray_cluster, MAX_NUM_WORKER_NODES)

        # Initialize Ray client
        self._init_ray_client()

    def shutdown(self) -> None:
        """Shutdown Ray and Databricks cluster resources.

        Shuts down Ray client and releases Spark cluster resources.
        """
        try:
            if ray.is_initialized():
                ray.shutdown()
                logger.debug("Ray client shutdown complete")
        except Exception as e:
            logger.warning("Error during Ray shutdown", extra={"error": str(e)})

        if self._cluster is not None:
            shutdown_fn = getattr(self._cluster, "shutdown", None)
            if callable(shutdown_fn):
                try:
                    shutdown_fn()
                    logger.debug("Databricks cluster shutdown complete")
                except Exception as e:
                    logger.warning(
                        "Error during cluster shutdown",
                        extra={"error": str(e)},
                    )
            self._cluster = None

    def _setup_spark_cluster(self, setup_fn: Any, max_workers: int) -> Any:
        """Setup Ray-on-Spark cluster.

        Args:
            setup_fn: The setup_ray_cluster function from ray.util.spark.
            max_workers: Maximum number of worker nodes.

        Returns:
            Cluster handle from setup_ray_cluster().
        """
        kwargs = {
            "num_worker_nodes": self._config.num_workers,
            "cpus_per_node": int(self._config.worker_cpus),
            "memory_per_node": self._config.worker_memory,
        }

        # Filter to only supported parameters
        signature = inspect.signature(setup_fn)
        filtered = {key: value for key, value in kwargs.items() if key in signature.parameters and value}

        logger.info(
            "Initializing Ray on Databricks",
            extra={"params": filtered},
        )

        return setup_fn(max_worker_nodes=max_workers)

    def _init_ray_client(self) -> None:
        """Initialize Ray client connection to Databricks cluster."""
        address = getattr(self._cluster, "address", None) if self._cluster else None
        runtime_env = self.get_runtime_env()

        if ray.is_initialized():
            logger.warning(
                "Ray already initialized; skipping runtime_env updates. "
                "Restart the job/cluster to apply VECTOR_DB_PROVIDER to workers."
            )
            return

        ray.init(
            address=address or "auto",
            namespace=self._config.ray_namespace,
            ignore_reinit_error=True,
            logging_level=logging.WARNING,
            log_to_driver=False,
            runtime_env=runtime_env or None,
        )

        logger.info(
            "Connected to Databricks Ray cluster",
            extra={
                "address": address or "auto",
                "namespace": self._config.ray_namespace,
            },
        )
