"""Shared batch submission/wait runner for Databricks Ray image jobs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import ray

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    import logging


@dataclass(slots=True)
class BatchRunStats:
    """Aggregated stats for batch processing."""

    submitted_records: int = 0
    num_batches: int = 0
    completed_records: int = 0
    successful: int = 0
    failed: int = 0


def _drain_ready_futures(
    *,
    futures: list[Any],
    stats: BatchRunStats,
    wait_batch_size: int,
    wait_timeout: float,
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
        stats.successful += sum(1 for _, ok, _ in batch_result if ok)
        stats.failed += sum(1 for _, ok, _ in batch_result if not ok)

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
    total_expected_records: int | None = None,
) -> BatchRunStats:
    """Submit and drain Ray tasks with bounded in-flight futures."""
    stats = BatchRunStats()
    futures: list[Any] = []

    resolved_wait_batch = max(1, wait_batch_size)
    resolved_max_inflight = max(1, max_inflight_batches)

    for batch in batches:
        futures.append(submit_batch(batch))
        stats.submitted_records += len(batch)
        stats.num_batches += 1

        while len(futures) >= resolved_max_inflight:
            futures = _drain_ready_futures(
                futures=futures,
                stats=stats,
                wait_batch_size=resolved_wait_batch,
                wait_timeout=wait_timeout,
                total_expected_records=total_expected_records,
                job_logger=job_logger,
                progress_label=progress_label,
            )

    while futures:
        futures = _drain_ready_futures(
            futures=futures,
            stats=stats,
            wait_batch_size=resolved_wait_batch,
            wait_timeout=wait_timeout,
            total_expected_records=total_expected_records,
            job_logger=job_logger,
            progress_label=progress_label,
        )

    return stats
