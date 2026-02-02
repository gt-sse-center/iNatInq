"""Logging utilities for Ray workers.

This module provides logging configuration that works across all Ray deployments:
- Local development
- Docker Compose
- Kubernetes
- Databricks
"""

from __future__ import annotations

import logging
from logging.config import dictConfig

from foundation.logger import LOGGING_CONFIG

# Configure logging once at module load
dictConfig(LOGGING_CONFIG)


def get_ray_logger(name: str = "ray.task") -> logging.Logger:
    """Get a logger configured for Ray workers.

    Uses Ray's recommended logging pattern that works across all deployments:
    - Local development
    - Docker Compose
    - Kubernetes
    - Databricks

    Args:
        name: Logger name (e.g., "ray.task", "ray.pipeline", "ray.image_pipeline").

    Returns:
        Configured logger instance.

    Example:
        ```python
        from core.ingestion.shared import get_ray_logger

        logger = get_ray_logger("ray.task")
        logger.info("Processing batch", extra={"batch_id": 1})
        ```
    """
    return logging.getLogger(name)
