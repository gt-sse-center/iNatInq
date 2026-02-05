"""Local Ray cluster strategy.

This strategy connects to an external Ray cluster via RAY_ADDRESS.
Used for local development with Docker Compose or Kubernetes deployments.
"""

from __future__ import annotations

import logging
import resource
from typing import Any

import attrs
import ray

from config import RayJobConfig

logger = logging.getLogger("pipeline.ray.strategy.local")


@attrs.define
class LocalRayStrategy:
    """Strategy for connecting to external Ray clusters.

    This strategy requires RAY_ADDRESS to be set and connects to an
    existing Ray cluster. It does not manage cluster lifecycle - only
    the client connection.

    Attributes:
        _config: Ray job configuration.

    Example:
        ```python
        config = RayJobConfig.from_env()
        strategy = LocalRayStrategy(config)
        strategy.init()
        # ... use Ray ...
        strategy.shutdown()
        ```
    """

    _config: RayJobConfig

    @classmethod
    def from_env(cls, namespace: str = "ml-system") -> LocalRayStrategy:
        """Create strategy from environment variables.

        Args:
            namespace: Kubernetes namespace for config resolution.

        Returns:
            Configured LocalRayStrategy instance.
        """
        return cls(config=RayJobConfig.from_env(namespace))

    @property
    def config(self) -> RayJobConfig:
        """Return the Ray job configuration."""
        return self._config

    def get_runtime_env(self) -> dict[str, Any]:
        """Return runtime environment for Ray workers.

        Returns:
            Runtime env dict from config, or empty dict if not set.
        """
        return self._config.runtime_env.copy() if self._config.runtime_env else {}

    def init(self) -> None:
        """Initialize connection to external Ray cluster.

        Connects to the Ray cluster specified by RAY_ADDRESS. Also attempts
        to increase the process thread limit to prevent Ray failures.

        Raises:
            ValueError: If RAY_ADDRESS is not configured.
            RuntimeError: If connection to Ray cluster fails.
        """
        if ray.is_initialized():
            logger.debug("Ray already initialized, skipping")
            return

        if not self._config.ray_address:
            raise ValueError(
                "RAY_ADDRESS is required. Local Ray execution is not supported. "
                "Set RAY_ADDRESS environment variable to connect to external Ray cluster."
            )

        self._increase_thread_limit()

        runtime_env = self.get_runtime_env()

        try:
            ray.init(
                address=self._config.ray_address,
                namespace=self._config.ray_namespace,
                runtime_env=runtime_env or None,
                ignore_reinit_error=True,
                logging_level=logging.WARNING,
                log_to_driver=False,
            )
            logger.info(
                "Connected to Ray cluster",
                extra={
                    "ray_address": self._config.ray_address,
                    "namespace": self._config.ray_namespace,
                },
            )
        except Exception as e:
            logger.error(
                "Failed to connect to Ray cluster",
                extra={
                    "ray_address": self._config.ray_address,
                    "error": str(e),
                },
                exc_info=True,
            )
            raise RuntimeError(f"Failed to connect to Ray cluster: {e}") from e

    def shutdown(self) -> None:
        """Shutdown Ray client connection.

        Safely disconnects from the Ray cluster. Does not shut down
        the cluster itself since we don't own it.
        """
        try:
            if ray.is_initialized():
                ray.shutdown()
                logger.debug("Ray client shutdown complete")
        except Exception as e:
            logger.warning("Error during Ray shutdown", extra={"error": str(e)})

    def _increase_thread_limit(self) -> None:
        """Attempt to increase process thread limit for Ray.

        Ray requires many threads for its workers. This attempts to
        increase the soft limit to prevent failures.
        """
        try:
            soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
            new_soft = min(hard, 8192) if hard != resource.RLIM_INFINITY else 8192
            if new_soft > soft:
                resource.setrlimit(resource.RLIMIT_NPROC, (new_soft, hard))
                logger.debug(
                    "Increased thread limit",
                    extra={"old": soft, "new": new_soft},
                )
        except (OSError, ValueError) as e:
            logger.warning(
                "Failed to increase thread limit - Ray may fail",
                extra={"error": str(e)},
            )
