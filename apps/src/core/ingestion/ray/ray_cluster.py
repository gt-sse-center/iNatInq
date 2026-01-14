"""Ray cluster initialization and management.

This module provides utilities for initializing and managing Ray clusters
for the ML pipeline jobs.
"""

import logging
import resource

import ray

from config import RayJobConfig

logger = logging.getLogger("pipeline.ray")


def init_ray_cluster(config: RayJobConfig) -> None:
    """Initialize Ray cluster for job execution.

    If Ray is already initialized, this function does nothing.
    This allows the function to be called multiple times safely.

    Args:
        config: Ray job configuration.

    Example:
        ```python
        from config import RayJobConfig
        from core.ingestion.ray.ray_cluster import init_ray_cluster

        config = RayJobConfig.from_env()
        init_ray_cluster(config)
        ```
    """
    try:
        if ray.is_initialized():
            return

        # Require RAY_ADDRESS for external cluster connection
        if not config.ray_address:
            raise ValueError(
                "RAY_ADDRESS is required. Local Ray execution is not supported. "
                "Set RAY_ADDRESS environment variable to connect to external Ray cluster."
            )

        try:
            # Get current soft and hard limits
            soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
            # Increase soft limit to hard limit (or 8192, whichever is lower)
            new_soft = min(hard, 8192) if hard != resource.RLIM_INFINITY else 8192
            if new_soft > soft:
                resource.setrlimit(resource.RLIMIT_NPROC, (new_soft, hard))
        except (OSError, ValueError) as e:
            logger.error(
                "Failed to increase thread limit - Ray will likely fail",
                extra={"error": str(e), "error_type": type(e).__name__},
                exc_info=True,
            )

        # Connect to external Ray cluster
        runtime_env = config.runtime_env.copy() if config.runtime_env else {}

        try:
            ray.init(
                address=config.ray_address,
                namespace=config.ray_namespace,
                runtime_env=runtime_env if runtime_env else None,
                ignore_reinit_error=True,
                logging_level=logging.WARNING,
                log_to_driver=False,
            )
            logger.info(
                "Connected to Ray cluster",
                extra={"ray_address": config.ray_address, "namespace": config.ray_namespace},
            )
        except Exception as e:
            logger.error(
                "Failed to connect to external Ray cluster",
                extra={
                    "ray_address": config.ray_address,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            raise
    except ImportError:
        logger.exception("Ray is not installed. Install with: pip install ray[default]")
        raise
    except (RuntimeError, ValueError) as e:
        logger.error(
            "Failed to initialize Ray cluster",
            extra={"error": str(e), "error_type": type(e).__name__},
            exc_info=True,
        )
        raise


def shutdown_ray_cluster() -> None:
    """Shutdown Ray cluster.

    This function safely shuts down the Ray cluster if it's initialized.
    """
    try:
        if ray.is_initialized():
            ray.shutdown()
    except ImportError:
        pass
    except (RuntimeError, ValueError) as e:
        logger.warning("Error shutting down Ray cluster", extra={"error": str(e)})
