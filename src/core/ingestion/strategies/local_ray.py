"""Local Ray cluster strategy.

This strategy connects to an external Ray cluster via RAY_ADDRESS.
Used for local development with Docker Compose or Kubernetes deployments.
"""

from __future__ import annotations

import logging
import resource
from typing import Any, Self

import attrs
import ray

from config import RayJobConfig

logger = logging.getLogger("pipeline.ray.strategy.local")


@attrs.define
class LocalRayStrategy:
    """Strategy for connecting to Ray clusters.

    If RAY_ADDRESS is set, connects to an external Ray cluster.
    Otherwise, starts a local Ray cluster in-process.

    Attributes:
        config: Ray job configuration.

    Example:
        ```python
        config = RayJobConfig.from_env()
        strategy = LocalRayStrategy(config=config)
        strategy.init()
        # ... use Ray ...
        strategy.shutdown()
        ```
    """

    config: RayJobConfig | None = None
    num_cpus: int | None = None
    include_dashboard: bool = True
    dashboard_host: str | None = None
    dashboard_port: int | None = None

    def __attrs_post_init__(self) -> None:
        if self.config is None:
            self.config = RayJobConfig.from_env()

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
    def is_active(self) -> bool:
        """Return True if Ray is initialized."""
        return ray.is_initialized()

    def get_runtime_env(self) -> dict[str, Any]:
        """Return runtime environment for Ray workers.

        Returns:
            Runtime env dict from config, or empty dict if not set.
        """
        config = self.config
        return config.runtime_env.copy() if config.runtime_env else {}

    def init(self) -> None:
        """Initialize connection to external Ray cluster.

        Connects to the Ray cluster specified by RAY_ADDRESS. Also attempts
        to increase the process thread limit to prevent Ray failures.

        Raises:
            RuntimeError: If connection to external Ray cluster fails.
        """
        if ray.is_initialized():
            logger.debug("Ray already initialized, skipping")
            return

        config = self.config
        runtime_env = self.get_runtime_env()

        self._increase_thread_limit()

        if config.ray_address:
            try:
                ray.init(
                    address=config.ray_address,
                    namespace=config.ray_namespace,
                    runtime_env=runtime_env or None,
                    ignore_reinit_error=True,
                    logging_level=logging.WARNING,
                    log_to_driver=False,
                )
                logger.info(
                    "Connected to Ray cluster",
                    extra={
                        "ray_address": config.ray_address,
                        "namespace": config.ray_namespace,
                    },
                )
                return
            except Exception as e:
                logger.error(
                    "Failed to connect to Ray cluster",
                    extra={
                        "ray_address": config.ray_address,
                        "error": str(e),
                    },
                    exc_info=True,
                )
                raise RuntimeError(f"Failed to connect to Ray cluster: {e}") from e

        init_kwargs: dict[str, Any] = {
            "namespace": config.ray_namespace,
            "ignore_reinit_error": True,
            "logging_level": logging.WARNING,
            "log_to_driver": False,
        }
        if runtime_env:
            init_kwargs["runtime_env"] = runtime_env
        if self.num_cpus is not None:
            init_kwargs["num_cpus"] = self.num_cpus
        if self.include_dashboard is not None:
            init_kwargs["include_dashboard"] = self.include_dashboard
        if self.dashboard_host:
            init_kwargs["dashboard_host"] = self.dashboard_host
        if self.dashboard_port:
            init_kwargs["dashboard_port"] = self.dashboard_port

        ray.init(**init_kwargs)
        logger.info(
            "Started local Ray cluster",
            extra={
                "num_cpus": self.num_cpus,
                "include_dashboard": self.include_dashboard,
            },
        )

    def initialize(self) -> None:
        """Initialize Ray cluster connection (alias for init)."""
        self.init()

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

    def __enter__(self) -> Self:
        self.initialize()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()

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
