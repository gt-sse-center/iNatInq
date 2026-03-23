"""Unit tests for core.ingestion.databricks.batch_runner."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

from core.ingestion.databricks.batch_runner import run_ray_batch_processing


def test_final_drain_blocks_until_futures_complete(mock_ray: MagicMock) -> None:
    future = MagicMock(name="future")
    mock_ray.wait.return_value = ([future], [])
    mock_ray.get.return_value = [[("file1.png", True, "")]]
    logger = MagicMock()

    stats = run_ray_batch_processing(
        batches=[["file1.png"]],
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


class TestBatchRunnerMetricsReporting:
    """Test that run_ray_batch_processing() calls report_ingestion_metrics per drain."""

    def test_reporter_called_when_pipeline_set(self, mock_ray: MagicMock) -> None:
        """Reporter is called once per drain cycle when pipeline is provided."""
        future = MagicMock()
        mock_ray.wait.return_value = ([future], [])
        mock_ray.get.return_value = [[("file1.png", True, "")]]

        with patch("core.ingestion.databricks.batch_runner.report_ingestion_metrics") as mock_report:
            run_ray_batch_processing(
                batches=[["file1.png"]],
                submit_batch=lambda _: future,
                wait_batch_size=10,
                wait_timeout=0.0,
                max_inflight_batches=10,
                job_logger=MagicMock(),
                progress_label="Progress",
                pipeline="pipeline",
            )

        mock_report.assert_called_once()

    def test_reporter_not_called_when_pipeline_empty(self, mock_ray: MagicMock) -> None:
        """Reporter is NOT called when pipeline is not provided."""
        future = MagicMock()
        mock_ray.wait.return_value = ([future], [])
        mock_ray.get.return_value = [[("file1.png", True, "")]]

        with patch("core.ingestion.databricks.batch_runner.report_ingestion_metrics") as mock_report:
            run_ray_batch_processing(
                batches=[["file1.png"]],
                submit_batch=lambda _: future,
                wait_batch_size=10,
                wait_timeout=0.0,
                max_inflight_batches=10,
                job_logger=MagicMock(),
                progress_label="Progress",
            )

        mock_report.assert_not_called()

    def test_reporter_receives_correct_delta_counts(self, mock_ray: MagicMock) -> None:
        """Reporter receives per-drain deltas, not cumulative totals."""
        future = MagicMock()
        mock_ray.wait.return_value = ([future], [])
        mock_ray.get.return_value = [
            [("ok1.png", True, ""), ("ok2.png", True, ""), ("fail1.png", False, "err")]
        ]

        with patch("core.ingestion.databricks.batch_runner.report_ingestion_metrics") as mock_report:
            run_ray_batch_processing(
                batches=[["ok1.png", "ok2.png", "fail1.png"]],
                submit_batch=lambda _: future,
                wait_batch_size=10,
                wait_timeout=0.0,
                max_inflight_batches=10,
                job_logger=MagicMock(),
                progress_label="Progress",
                pipeline="pipeline",
            )

        call_kwargs = mock_report.call_args.kwargs
        assert call_kwargs["successful"] == 2
        assert call_kwargs["failed"] == 1
        assert call_kwargs["pipeline"] == "pipeline"

    def test_reporter_not_called_when_drain_returns_nothing(self, mock_ray: MagicMock) -> None:
        """Reporter is NOT called when ray.wait returns no ready futures."""
        future = MagicMock()
        mock_ray.wait.side_effect = [
            ([], [future]),
            ([future], []),
        ]
        mock_ray.get.return_value = [[("file1.png", True, "")]]

        with patch("core.ingestion.databricks.batch_runner.report_ingestion_metrics") as mock_report:
            run_ray_batch_processing(
                batches=[["file1.png"]],
                submit_batch=lambda _: future,
                wait_batch_size=10,
                wait_timeout=1.0,
                max_inflight_batches=10,
                job_logger=MagicMock(),
                progress_label="Progress",
                pipeline="pipeline",
            )

        assert mock_report.call_count == 1

    def test_reporter_called_once_per_drain_cycle_across_multiple_batches(self, mock_ray: MagicMock) -> None:
        """Reporter is called once per completed drain, not once per submitted batch."""
        f1, f2 = MagicMock(name="f1"), MagicMock(name="f2")
        mock_ray.wait.side_effect = [
            ([f1], [f2]),
            ([f2], []),
        ]
        mock_ray.get.side_effect = [
            [[("a.png", True, "")]],
            [[("b.png", True, "")]],
        ]

        with patch("core.ingestion.databricks.batch_runner.report_ingestion_metrics") as mock_report:
            run_ray_batch_processing(
                batches=[["a.png"], ["b.png"]],
                submit_batch=lambda batch: f1 if batch == ["a.png"] else f2,
                wait_batch_size=1,
                wait_timeout=0.0,
                max_inflight_batches=3,
                job_logger=MagicMock(),
                progress_label="Progress",
                pipeline="pipeline",
            )

        assert mock_report.call_count == 2
