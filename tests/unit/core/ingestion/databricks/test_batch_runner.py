"""Unit tests for core.ingestion.databricks.batch_runner."""

from __future__ import annotations

from unittest.mock import MagicMock


def test_final_drain_blocks_until_futures_complete(mock_ray) -> None:
    from core.ingestion.databricks.batch_runner import run_ray_batch_processing

    future = MagicMock(name="future")
    mock_ray.wait.return_value = ([future], [])
    mock_ray.get.return_value = [[("file1.txt", True, "")]]
    logger = MagicMock()

    stats = run_ray_batch_processing(
        batches=[["file1.txt"]],
        submit_batch=lambda _: future,
        wait_batch_size=10,
        wait_timeout=0.0,
        max_inflight_batches=10,
        job_logger=logger,
        progress_label="Progress",
    )

    assert stats.submitted_records == 1
    assert stats.completed_records == 1
    assert stats.successful == 1
    assert stats.failed == 0
    assert mock_ray.wait.call_count == 1
    assert mock_ray.wait.call_args.kwargs["timeout"] is None
