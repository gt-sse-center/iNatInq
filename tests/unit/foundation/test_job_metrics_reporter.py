from typing import Any
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest
import requests

from foundation.metrics.job_metrics_reporter import report_ingestion_metrics


class TestReportIngestionMetrics:
    """Tests for the report_ingestion_metrics function."""

    @pytest.fixture(autouse=True)
    def run_executor_inline(self) -> Generator[None, Any, None]:
        """Patch _executor so submitted work runs synchronously, avoiding race conditions."""
        with patch("foundation.metrics.job_metrics_reporter._executor") as mock_exec:
            mock_exec.submit.side_effect = lambda fn, *args, **kwargs: fn(*args, **kwargs)
            yield

    def test_returns_nothing_and_logs_warning_when_app_url_not_set(
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("APP_URL", raising=False)
        report_ingestion_metrics(pipeline="local ray ingestion pipeline")
        assert "APP_URL environment variable is not set, skipping metrics reporting" in caplog.text

    def test_posts_to_correct_endpoint_with_full_payload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_response = MagicMock()
        monkeypatch.setenv("APP_URL", "http://localhost:8000")
        with patch(
            "foundation.metrics.job_metrics_reporter.requests.post", return_value=mock_response
        ) as mock_post:
            report_ingestion_metrics(
                pipeline="local ray ingestion pipeline",
                successful=5,
                failed=1,
                batch_duration_seconds=2.5,
                checkpoint_save=True,
            )

        mock_post.assert_called_once()
        url = mock_post.call_args[0][0]
        assert url == "http://localhost:8000/ingestion/metrics"
        payload = mock_post.call_args[1]["json"]
        assert payload == {
            "pipeline": "local ray ingestion pipeline",
            "successful": 5,
            "failed": 1,
            "batch_duration_seconds": 2.5,
            "checkpoint_save": True,
        }
        mock_response.raise_for_status.assert_called_once()

    def test_swallows_http_error_and_logs_warning(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("500 Server Error")
        monkeypatch.setenv("APP_URL", "http://localhost:8000")
        with (
            patch("foundation.metrics.job_metrics_reporter.requests.post", return_value=mock_response),
            patch("foundation.metrics.job_metrics_reporter.logger") as mock_logger,
        ):
            report_ingestion_metrics(pipeline="local ray ingestion pipeline")

        mock_logger.warning.assert_called_once()
        call_kwargs = mock_logger.warning.call_args[1]["extra"]
        assert "error" in call_kwargs
        assert call_kwargs["pipeline"] == "local ray ingestion pipeline"

    def test_swallows_connection_error_and_logs_warning(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_URL", "http://localhost:8000")
        with (
            patch(
                "foundation.metrics.job_metrics_reporter.requests.post",
                side_effect=requests.ConnectionError("connection refused"),
            ),
            patch("foundation.metrics.job_metrics_reporter.logger") as mock_logger,
        ):
            report_ingestion_metrics(pipeline="local ray ingestion pipeline")

        mock_logger.warning.assert_called_once()
        call_kwargs = mock_logger.warning.call_args[1]["extra"]
        assert "connection refused" in call_kwargs["error"]
