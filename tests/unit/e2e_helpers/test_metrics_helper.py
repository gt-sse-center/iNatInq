"""Unit tests for Prometheus metrics parser.

These tests verify the metrics_helper utility can parse Prometheus
exposition format and extract metric values by name and labels.
"""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from tests.e2e.metrics_helper import get_metric_value, metric_delta, scrape_metrics

# Hardcoded Prometheus text fixture (no network needed)
SAMPLE_PROMETHEUS_TEXT = """
# HELP http_requests_total Total HTTP requests
# TYPE http_requests_total counter
http_requests_total{method="GET",path="/healthz"} 42.0
http_requests_total{method="POST",path="/search/images"} 17.0

# HELP process_cpu_seconds_total Total user and system CPU time spent in seconds
# TYPE process_cpu_seconds_total counter
process_cpu_seconds_total 123.45

# HELP http_request_duration_seconds HTTP request duration
# TYPE http_request_duration_seconds histogram
http_request_duration_seconds_bucket{le="0.1"} 50
http_request_duration_seconds_bucket{le="0.5"} 80
http_request_duration_seconds_bucket{le="1.0"} 100
http_request_duration_seconds_sum 75.5
http_request_duration_seconds_count 100
""".strip()


def _mock_metrics_response() -> MagicMock:
    """Create a mock httpx response returning SAMPLE_PROMETHEUS_TEXT."""
    mock_response = MagicMock()
    mock_response.text = SAMPLE_PROMETHEUS_TEXT
    mock_response.status_code = 200
    return mock_response


def test_scrape_metrics_parses_counter():
    """Test scrape_metrics correctly parses counter metrics."""
    with patch("httpx.get", return_value=_mock_metrics_response()):
        metrics = scrape_metrics("http://localhost:8000")

    assert "http_requests_total" in metrics
    assert len(metrics["http_requests_total"]) == 2  # Two label combinations
    assert metrics["http_requests_total"][0].value == 42.0
    assert metrics["http_requests_total"][0].labels == {"method": "GET", "path": "/healthz"}


def test_scrape_metrics_parses_histogram():
    """Test scrape_metrics correctly parses histogram metrics."""
    with patch("httpx.get", return_value=_mock_metrics_response()):
        metrics = scrape_metrics("http://localhost:8000")

    # Histograms have separate keys for bucket, sum, count
    assert "http_request_duration_seconds_bucket" in metrics
    assert "http_request_duration_seconds_sum" in metrics
    assert "http_request_duration_seconds_count" in metrics
    # 3 buckets
    assert len(metrics["http_request_duration_seconds_bucket"]) == 3


def test_get_metric_value_exact_match():
    """Test get_metric_value returns correct float for exact label match."""
    with patch("httpx.get", return_value=_mock_metrics_response()):
        metrics = scrape_metrics("http://localhost:8000")

    value = get_metric_value(metrics, "http_requests_total", {"method": "POST", "path": "/search/images"})
    assert value == 17.0


def test_get_metric_value_no_labels():
    """Test get_metric_value works for metrics without labels."""
    with patch("httpx.get", return_value=_mock_metrics_response()):
        metrics = scrape_metrics("http://localhost:8000")

    value = get_metric_value(metrics, "process_cpu_seconds_total", {})
    assert value == 123.45


def test_get_metric_value_missing_metric():
    """Test get_metric_value returns None for missing metric."""
    with patch("httpx.get", return_value=_mock_metrics_response()):
        metrics = scrape_metrics("http://localhost:8000")

    value = get_metric_value(metrics, "nonexistent_metric", {})
    assert value is None


def test_get_metric_value_missing_labels():
    """Test get_metric_value returns None for missing label combination."""
    with patch("httpx.get", return_value=_mock_metrics_response()):
        metrics = scrape_metrics("http://localhost:8000")

    value = get_metric_value(metrics, "http_requests_total", {"method": "DELETE", "path": "/unknown"})
    assert value is None


def test_metric_delta_context_manager():
    """Test metric_delta context manager returns correct before/after difference."""
    before_text = 'http_requests_total{method="GET",path="/healthz"} 42.0'
    after_text = 'http_requests_total{method="GET",path="/healthz"} 50.0'

    mock_response_before = MagicMock()
    mock_response_before.text = before_text
    mock_response_before.status_code = 200

    mock_response_after = MagicMock()
    mock_response_after.text = after_text
    mock_response_after.status_code = 200

    with (
        patch("httpx.get", side_effect=[mock_response_before, mock_response_after]),
        metric_delta(
            "http://localhost:8000", "http_requests_total", {"method": "GET", "path": "/healthz"}
        ) as delta,
    ):
        pass  # Simulate some work that increments the counter

    assert delta.value == 8.0  # 50 - 42


def test_metric_delta_new_metric():
    """Test metric_delta handles case where metric doesn't exist before."""
    before_text = ""
    after_text = "new_counter{} 5.0"

    mock_response_before = MagicMock()
    mock_response_before.text = before_text
    mock_response_before.status_code = 200

    mock_response_after = MagicMock()
    mock_response_after.text = after_text
    mock_response_after.status_code = 200

    with (
        patch("httpx.get", side_effect=[mock_response_before, mock_response_after]),
        metric_delta("http://localhost:8000", "new_counter", {}) as delta,
    ):
        pass

    assert delta.value == 5.0  # 5 - 0


def test_scrape_metrics_handles_network_error():
    """Test scrape_metrics raises on network errors."""
    with (
        patch("httpx.get", side_effect=httpx.RequestError("Connection failed")),
        pytest.raises(httpx.RequestError),
    ):
        scrape_metrics("http://localhost:8000")


def test_scrape_metrics_handles_non_200():
    """Test scrape_metrics raises on non-200 status."""
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError("Server error", request=MagicMock(), response=mock_response)
    )

    with patch("httpx.get", return_value=mock_response), pytest.raises(httpx.HTTPStatusError):
        scrape_metrics("http://localhost:8000")
