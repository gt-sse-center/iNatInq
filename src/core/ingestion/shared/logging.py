"""Logging utilities for Ray workers.

This module provides logging configuration that works across all Ray deployments:
- Local development
- Docker Compose
- Kubernetes
- Databricks
"""

from __future__ import annotations

import logging
import os
from logging.config import dictConfig

from foundation.logger import LOGGING_CONFIG


# Component-to-logger mapping for selective debug logging.
_COMPONENT_LOGGERS: dict[str, tuple[str, ...]] = {
    "s3": ("clients.s3", "clients.s3.retry"),
    "clip": ("clients.clip", "clients.clip.retry"),
    "qdrant": ("clients.qdrant", "clients.qdrant.retry"),
    "ollama": ("clients.ollama", "clients.ollama.retry"),
    "ray": ("pipeline.ray", "pipeline.ray.job", "pipeline.ray.task"),
}


def _configure_component_debug_loggers() -> None:
    """Enable DEBUG level for selected components via PIPELINE_DEBUG_COMPONENTS.

    Accepted values:
    - "all": enable all known component loggers
    - comma-separated component names (e.g., "s3,clip")
    """
    raw_value = os.getenv("PIPELINE_DEBUG_COMPONENTS", "").strip().lower()
    if not raw_value:
        return

    if raw_value == "all":
        logger_names = {logger_name for names in _COMPONENT_LOGGERS.values() for logger_name in names}
    else:
        selected = {item.strip() for item in raw_value.split(",") if item.strip()}
        logger_names = {
            logger_name for component in selected for logger_name in _COMPONENT_LOGGERS.get(component, ())
        }

    for logger_name in logger_names:
        logging.getLogger(logger_name).setLevel(logging.DEBUG)


# Configure logging once at module load
dictConfig(LOGGING_CONFIG)
_configure_component_debug_loggers()


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
