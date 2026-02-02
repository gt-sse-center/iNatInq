"""Base protocol for Ray cluster strategies.

This module defines the abstract interface that all cluster strategies must implement.
Each strategy handles the lifecycle of a Ray cluster in a specific environment
(local, Databricks, Kubernetes, etc.).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from config import RayJobConfig


class ClusterStrategy(Protocol):
    """Protocol for Ray cluster lifecycle management.

    Implementations handle environment-specific initialization, configuration,
    and shutdown of Ray clusters. This enables the IngestionPipeline to work
    across different deployment environments without code changes.

    Example:
        ```python
        strategy = LocalRayStrategy(config)
        strategy.init()
        try:
            # Use Ray cluster
            ray.get([task.remote() for task in tasks])
        finally:
            strategy.shutdown()
        ```
    """

    def init(self) -> None:
        """Initialize Ray cluster connection.

        This method should:
        1. Set up the Ray cluster (if needed)
        2. Initialize the Ray client connection
        3. Configure runtime environment for workers

        Raises:
            RuntimeError: If cluster initialization fails.
        """
        ...

    def shutdown(self) -> None:
        """Shutdown Ray cluster and cleanup resources.

        This method should:
        1. Shutdown Ray client connection
        2. Cleanup any cluster resources (if owned)
        3. Release any held references
        """
        ...

    def get_runtime_env(self) -> dict[str, Any]:
        """Return runtime environment configuration for Ray workers.

        The runtime environment is passed to ray.init() and controls
        how workers are configured (env vars, working directory, etc.).

        Returns:
            Dictionary with Ray runtime_env configuration.
            May include: env_vars, working_dir, pip, conda, etc.
        """
        ...

    @property
    def config(self) -> RayJobConfig:
        """Return the Ray job configuration."""
        ...
