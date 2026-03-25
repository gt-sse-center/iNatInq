# ruff: noqa: ERA001
"""Ingestion metrics reporter for ephemeral job processes.

Ingestion jobs (Ray, Databricks) run in separate processes that exit after
completion.  Because Prometheus scrapes the long-lived API process, metrics
collected inside a job process would be lost => this module POSTs batch stats
to the API's `POST /ingestion/metrics` endpoint so they land in the Prometheus
registry.
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

import requests

PipelineType = Literal["local ray ingestion pipeline", "databricks ingestion pipeline", ""]

_executor = ThreadPoolExecutor(max_workers=1)

logger = logging.getLogger("pipeline.metrics_reporter")

INGESTION_METRICS_PATH = "/ingestion/metrics"
_POST_TIMEOUT_SECONDS = 5


def _do_post(url: str, payload: dict[str, Any], pipeline: str) -> None:
    try:
        response = requests.post(url, json=payload, timeout=_POST_TIMEOUT_SECONDS)
        response.raise_for_status()
    except Exception as exc:
        logger.warning(
            "Failed to report ingestion metrics",
            extra={"error": str(exc), "pipeline": pipeline},
        )


def report_ingestion_metrics(
    *,
    pipeline: PipelineType,
    successful: int = 0,
    failed: int = 0,
    batch_duration_seconds: float = 0.0,
    checkpoint_save: bool = False,
) -> None:
    """Function to POST ingestion metrics to the API endpoint for addition to the Prometheus registry.

    Reads `APP_URL` from the environment. Errors are only logged not re-raised.

    Args:
        pipeline: String label of the pipeline.
        successful: Number of images successfully processed.
        failed: Number of images that failed.
        batch_duration_seconds: Wall-clock seconds for the batch.
            Zero skips the duration histogram observation.
        checkpoint_save: When `True`, increments the checkpoint saves counter.
    """
    # base_url = os.environ.get("APP_URL", "")
    # if not base_url:
    # logger.warning("APP_URL environment variable is not set, skipping metrics reporting")
    # return

    payload = {
        "pipeline": pipeline,
        "successful": successful,
        "failed": failed,
        "batch_duration_seconds": batch_duration_seconds,
        "checkpoint_save": checkpoint_save,
    }
    # _executor.submit(_do_post, f"{base_url}{INGESTION_METRICS_PATH}", payload, pipeline)
    logger.info("Reporting ingestion metrics", extra={"payload": payload, "pipeline": pipeline})
