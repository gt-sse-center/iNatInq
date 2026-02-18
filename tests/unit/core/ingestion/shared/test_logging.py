"""Unit tests for core.ingestion.shared.logging helpers."""

from __future__ import annotations

import logging

from core.ingestion.shared import logging as shared_logging


def test_configure_component_debug_loggers_all(monkeypatch) -> None:
    """PIPELINE_DEBUG_COMPONENTS=all should set known component loggers to DEBUG."""
    logger_names = (
        "clients.s3",
        "clients.clip",
        "clients.clip.retry",
        "clients.qdrant",
        "clients.qdrant.retry",
    )
    for name in logger_names:
        logging.getLogger(name).setLevel(logging.INFO)

    monkeypatch.setenv("PIPELINE_DEBUG_COMPONENTS", "all")
    shared_logging._configure_component_debug_loggers()

    for name in logger_names:
        assert logging.getLogger(name).level == logging.DEBUG


def test_configure_component_debug_loggers_subset(monkeypatch) -> None:
    """A subset should only enable DEBUG for selected components."""
    logging.getLogger("clients.s3").setLevel(logging.INFO)
    logging.getLogger("clients.clip").setLevel(logging.INFO)
    logging.getLogger("clients.clip.retry").setLevel(logging.INFO)

    monkeypatch.setenv("PIPELINE_DEBUG_COMPONENTS", "s3")
    shared_logging._configure_component_debug_loggers()

    assert logging.getLogger("clients.s3").level == logging.DEBUG
    assert logging.getLogger("clients.clip").level == logging.INFO
    assert logging.getLogger("clients.clip.retry").level == logging.INFO
