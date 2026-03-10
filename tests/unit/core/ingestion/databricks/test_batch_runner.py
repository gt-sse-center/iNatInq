"""Unit tests for core.ingestion.databricks.batch_runner."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch


def test_final_drain_blocks_until_futures_complete(mock_ray: MagicMock) -> None:
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


def test_dlq_called_for_failed_ingestion_results(mock_ray: MagicMock) -> None:
    """Verify that the image ingestion logic, shared between ray, databricks, and databricks-inat entrypoints, invokes the dlq and enqueues any ingestion failures."""
    from core.ingestion.databricks.batch_runner import run_ray_batch_processing

    results = [
        ("fail-1", False, "error one"),
        ("ok-1", True, ""),
    ]

    mock_ray_obj_ref = MagicMock()
    mock_ray.wait.return_value = ([mock_ray_obj_ref], [])
    mock_ray.get.return_value = [results]

    with (
        patch("foundation.dead_letter_queue.with_dlq.DLQ") as mock_dlq_cls,
    ):
        mock_dlq = MagicMock()
        mock_dlq_cls.return_value = mock_dlq

        def submit_batch(_batch: list[object]):
            return mock_ray_obj_ref

        run_ray_batch_processing(
            batches=[["batch"]],
            submit_batch=submit_batch,
            wait_batch_size=1,
            wait_timeout=1,
            max_inflight_batches=1,
            job_logger=logging.getLogger(__name__),
            progress_label="dlq-integration-test",
            total_expected_records=None,
        )

    expected_image_id = results[0][0]
    expected_metadata = {
        "key": expected_image_id,
        "error": results[0][2],
        "label": "dlq-integration-test",
    }

    mock_dlq.enqueue_failed_ingestion.assert_called_once_with(expected_image_id, metadata=expected_metadata)
