"""Prometheus metrics parser for E2E testing.

Provides utilities to fetch /metrics endpoints, parse Prometheus text
exposition format, and extract specific metric values for assertions.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import httpx
from prometheus_client.parser import text_string_to_metric_families

if TYPE_CHECKING:
    from collections.abc import Generator


class MetricDelta:
    """Container for before/after metric delta."""

    def __init__(self):
        self.value: float = 0.0


def scrape_metrics(base_url: str) -> dict[str, list[Any]]:
    """Fetch /metrics endpoint and parse Prometheus text exposition format.

    Args:
        base_url: Base URL of the service (e.g., "http://localhost:8000").

    Returns:
        Dictionary mapping metric names to lists of samples.
        Each sample has .name, .labels (dict), .value (float).

    Raises:
        httpx.RequestError: On network errors.
        httpx.HTTPStatusError: On non-200 status codes.

    Example:
        >>> metrics = scrape_metrics("http://localhost:8000")
        >>> metrics["http_requests_total"][0].value
        42.0
    """
    url = f"{base_url}/metrics"
    response = httpx.get(url, timeout=5.0)
    response.raise_for_status()

    # Parse Prometheus text format into metric families
    # Group samples by their full sample name (not family name)
    metrics_dict: dict[str, list[Any]] = {}
    for family in text_string_to_metric_families(response.text):
        for sample in family.samples:
            if sample.name not in metrics_dict:
                metrics_dict[sample.name] = []
            metrics_dict[sample.name].append(sample)

    return metrics_dict


def get_metric_value(metrics: dict[str, list[Any]], name: str, labels: dict[str, str]) -> float | None:
    """Look up a specific metric sample by name and label dict.

    Args:
        metrics: Output from scrape_metrics().
        name: Metric name (e.g., "http_requests_total").
        labels: Label dict for exact match (e.g., {"method": "GET", "path": "/healthz"}).

    Returns:
        Metric value as float, or None if not found.

    Example:
        >>> value = get_metric_value(metrics, "http_requests_total", {"method": "POST"})
        >>> assert value == 17.0
    """
    if name not in metrics:
        return None

    for sample in metrics[name]:
        if sample.labels == labels:
            return float(sample.value)

    return None


@contextmanager
def metric_delta(
    base_url: str,
    metric_name: str,
    labels: dict[str, str],
) -> Generator[MetricDelta, None, None]:
    """Context manager to measure before/after delta of a metric.

    Scrapes metrics before entering the context, yields, scrapes again
    after exiting, and computes the delta.

    Args:
        base_url: Base URL of the service.
        metric_name: Name of the metric to track.
        labels: Label dict for exact match.

    Yields:
        MetricDelta object with .value attribute (float).

    Example:
        >>> with metric_delta("http://localhost:8000", "http_requests_total", {"method": "GET"}) as delta:
        ...     # Make some requests
        ...     pass
        >>> assert delta.value == 5.0  # 5 new requests
    """
    delta = MetricDelta()

    # Scrape before
    before_metrics = scrape_metrics(base_url)
    before_value = get_metric_value(before_metrics, metric_name, labels) or 0.0

    yield delta

    # Scrape after
    after_metrics = scrape_metrics(base_url)
    after_value = get_metric_value(after_metrics, metric_name, labels) or 0.0

    delta.value = after_value - before_value
