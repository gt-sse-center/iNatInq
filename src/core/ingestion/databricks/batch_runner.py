"""Shared batch submission/wait runner for Databricks Ray image jobs."""

from __future__ import annotations

import random
import time
from typing import TYPE_CHECKING, Any, cast

import attrs
import ray

from foundation.dead_letter_queue import DLQ, with_dlq
from foundation.metrics.job_metrics_reporter import PipelineType, report_ingestion_metrics

if TYPE_CHECKING:
    import logging
    from collections.abc import Callable, Iterable


@attrs.define(slots=True)
class BatchRunStats:
    """Aggregated stats for batch processing."""

    submitted_records: int = 0
    num_batches: int = 0
    completed_records: int = 0
    successful: int = 0
    failed: int = 0
    successful_keys: set[str] = attrs.field(factory=set)


@with_dlq
def _drain_ready_futures(
    dlq: DLQ,
    *,
    futures: list[Any],
    stats: BatchRunStats,
    wait_batch_size: int,
    wait_timeout: float | None,
    total_expected_records: int | None,
    job_logger: logging.Logger,
    progress_label: str,
) -> list[Any]:
    """Wait for ready futures, aggregate results, and log progress."""
    ready, not_ready = ray.wait(
        futures,
        num_returns=min(wait_batch_size, len(futures)),
        timeout=wait_timeout,
    )
    if not ready:
        return futures

    batch_results = ray.get(ready)
    for batch_result in batch_results:
        stats.completed_records += len(batch_result)
        for key, ok, err_msg in cast(list[tuple[str, bool, str]], batch_result):
            # Force failure 5% of the time
            if random.random() < 0.05:
                ok = False
            if ok:
                stats.successful += 1
                stats.successful_keys.add(key)
            else:
                dlq.enqueue_failed_ingestion(
                    key,
                    metadata={
                        "key": key,
                        "error": err_msg,
                        "label": progress_label,
                    },
                )
                stats.failed += 1

    if total_expected_records is None:
        job_logger.info(
            "%s: %d completed (%d ok, %d failed, %d submitted)",
            progress_label,
            stats.completed_records,
            stats.successful,
            stats.failed,
            stats.submitted_records,
            extra={
                "completed_records": stats.completed_records,
                "submitted_records": stats.submitted_records,
                "successful": stats.successful,
                "failed": stats.failed,
                "remaining_inflight_batches": len(not_ready),
            },
        )
    else:
        job_logger.info(
            "%s: %d/%d completed (%d ok, %d failed)",
            progress_label,
            stats.completed_records,
            total_expected_records,
            stats.successful,
            stats.failed,
            extra={
                "completed_records": stats.completed_records,
                "total_records": total_expected_records,
                "successful": stats.successful,
                "failed": stats.failed,
                "remaining_records": total_expected_records - stats.completed_records,
                "remaining_inflight_batches": len(not_ready),
            },
        )
    return not_ready


def run_ray_batch_processing(
    *,
    batches: Iterable[list[Any]],
    submit_batch: Callable[[list[Any]], Any],
    wait_batch_size: int,
    wait_timeout: float,
    max_inflight_batches: int,
    job_logger: logging.Logger,
    progress_label: str,
    pipeline: PipelineType = "",
    total_expected_records: int | None = None,
) -> BatchRunStats:
    """Submit and drain Ray tasks with bounded in-flight futures."""
    stats = BatchRunStats()
    futures: list[Any] = []

    resolved_wait_batch = max(1, wait_batch_size)
    resolved_max_inflight = max(1, max_inflight_batches)

    def _drain_and_report(futures: list[Any], timeout: float | None) -> list[Any]:
        prev_successful = stats.successful
        prev_failed = stats.failed
        t0 = time.monotonic()
        result = _drain_ready_futures(
            futures=futures,
            stats=stats,
            wait_batch_size=resolved_wait_batch,
            wait_timeout=timeout,
            total_expected_records=total_expected_records,
            job_logger=job_logger,
            progress_label=progress_label,
        )
        elapsed = time.monotonic() - t0
        delta_successful = stats.successful - prev_successful
        delta_failed = stats.failed - prev_failed
        if pipeline and (delta_successful > 0 or delta_failed > 0):
            report_ingestion_metrics(
                pipeline=pipeline,
                successful=delta_successful,
                failed=delta_failed,
                batch_duration_seconds=elapsed,
            )
        return result

    for batch in batches:
        futures.append(submit_batch(batch))
        stats.submitted_records += len(batch)
        stats.num_batches += 1

        while len(futures) >= resolved_max_inflight:
            futures = _drain_and_report(futures, wait_timeout)

    while futures:
        futures = _drain_and_report(futures, None)

    return stats
