"""Databricks Ray job: S3 → embeddings → vector DB.

This entrypoint is a Databricks-specific wrapper that:
- Starts Ray on the Databricks cluster using ray.util.spark
- Initializes a Ray client for the job namespace
- Delegates processing to the shared Ray ingestion logic
- Shuts down Ray and cluster resources on exit

The processing logic itself lives in the shared Ray module to avoid duplicating
business logic across runtime environments.
"""

from __future__ import annotations

import inspect
import logging
from logging.config import dictConfig

import ray

from config import RayJobConfig
from core.ingestion.ray.process_s3_to_qdrant import main as run_ray_job
from foundation.logger import LOGGING_CONFIG

try:
    from ray.util.spark import setup_ray_cluster
except ImportError:  # pragma: no cover - only available on Databricks
    setup_ray_cluster = None

logger = logging.getLogger("pipeline.ray.databricks")
dictConfig(LOGGING_CONFIG)


def _setup_ray_cluster(config: RayJobConfig) -> object | None:
    """Start Ray on Databricks and return the cluster handle when possible."""
    if setup_ray_cluster is None:
        raise RuntimeError("ray.util.spark is required to start Ray on Databricks.")

    kwargs = {
        "num_workers": config.num_workers,
        "cpus_per_node": int(config.worker_cpus),
        "memory_per_node": config.worker_memory,
    }
    signature = inspect.signature(setup_ray_cluster)
    filtered = {key: value for key, value in kwargs.items() if key in signature.parameters and value}

    logger.info("Initializing Ray on Databricks", extra={"params": filtered})
    return setup_ray_cluster(**filtered)


def _init_ray(config: RayJobConfig, ray_cluster: object | None) -> None:
    """Initialize Ray client connection to the Databricks cluster."""
    address = getattr(ray_cluster, "address", None) if ray_cluster is not None else None
    if not ray.is_initialized():
        ray.init(
            address=address or "auto",
            namespace=config.ray_namespace,
            ignore_reinit_error=True,
            logging_level=logging.WARNING,
            log_to_driver=False,
        )


def _shutdown_ray_cluster(ray_cluster: object | None) -> None:
    """Shutdown Ray and Databricks cluster resources."""
    if ray.is_initialized():
        ray.shutdown()
    if ray_cluster is None:
        return
    shutdown = getattr(ray_cluster, "shutdown", None)
    if callable(shutdown):
        shutdown()


def main() -> None:
    """Initialize Ray on Databricks and run the shared ingestion job."""
    ray_cfg = RayJobConfig.from_env()
    ray_cluster = _setup_ray_cluster(ray_cfg)
    _init_ray(ray_cfg, ray_cluster)
    try:
        logger.info("Databricks Ray job started")
        run_ray_job()
    finally:
        logger.info("Databricks Ray job completed; shutting down Ray")
        _shutdown_ray_cluster(ray_cluster)


if __name__ == "__main__":
    main()
