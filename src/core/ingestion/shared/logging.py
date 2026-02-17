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

# Configure logging once at module load
dictConfig(LOGGING_CONFIG)

_COMPONENT_DEBUG_LOGGERS: dict[str, tuple[str, ...]] = {
    "s3": ("clients.s3",),
    "clip": ("clients.clip", "clients.clip.retry"),
    "qdrant": ("clients.qdrant", "clients.qdrant.retry"),
}


def _configure_component_debug_loggers() -> None:
    """Enable DEBUG logging for selected ingestion components.

    Controlled by ``PIPELINE_DEBUG_COMPONENTS`` env var, for example:
    - ``PIPELINE_DEBUG_COMPONENTS=s3,clip,qdrant``
    - ``PIPELINE_DEBUG_COMPONENTS=all``
    """
    raw = os.getenv("PIPELINE_DEBUG_COMPONENTS", "")
    tokens = {token.strip().lower() for token in raw.split(",") if token.strip()}
    if not tokens:
        return

    known_components = set(_COMPONENT_DEBUG_LOGGERS)
    selected_components = known_components if "all" in tokens else (tokens & known_components)
    unknown_components = tokens - known_components - {"all"}

    for component in sorted(selected_components):
        for logger_name in _COMPONENT_DEBUG_LOGGERS[component]:
            logging.getLogger(logger_name).setLevel(logging.DEBUG)

    if selected_components:
        logging.getLogger("pipeline.ray.logging").info(
            "Enabled component debug logging",
            extra={"components": sorted(selected_components)},
        )

    if unknown_components:
        logging.getLogger("pipeline.ray.logging").warning(
            "Ignoring unknown debug components",
            extra={"unknown_components": sorted(unknown_components)},
        )


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
