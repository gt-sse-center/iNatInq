"""Unit tests for Databricks ingestion entrypoint helpers."""

from __future__ import annotations

import importlib
import sys
import types


def test_entrypoint_file_uses_explicit_env(monkeypatch) -> None:
    """_entrypoint_file should prefer DATABRICKS_ENTRYPOINT_FILE when set."""
    dummy_ingest = types.ModuleType("core.ingestion.databricks.process_s3_to_qdrant")
    dummy_ingest.main = lambda: None
    dummy_logger = types.ModuleType("foundation.logger")
    dummy_logger.LOGGING_CONFIG = {"version": 1}

    monkeypatch.setitem(sys.modules, "core.ingestion.databricks.process_s3_to_qdrant", dummy_ingest)
    monkeypatch.setitem(sys.modules, "foundation.logger", dummy_logger)

    module = importlib.import_module("core.ingestion.databricks.run_ingest")
    monkeypatch.setenv("DATABRICKS_ENTRYPOINT_FILE", "/tmp/custom_entry.py")

    assert module._entrypoint_file() == "/tmp/custom_entry.py"  # type: ignore[attr-defined]


def test_entrypoint_file_falls_back_to_module_file(monkeypatch) -> None:
    """_entrypoint_file should use module __file__ when no env override exists."""
    dummy_ingest = types.ModuleType("core.ingestion.databricks.process_s3_to_qdrant")
    dummy_ingest.main = lambda: None
    dummy_logger = types.ModuleType("foundation.logger")
    dummy_logger.LOGGING_CONFIG = {"version": 1}

    monkeypatch.setitem(sys.modules, "core.ingestion.databricks.process_s3_to_qdrant", dummy_ingest)
    monkeypatch.setitem(sys.modules, "foundation.logger", dummy_logger)

    module = importlib.import_module("core.ingestion.databricks.run_ingest")
    monkeypatch.delenv("DATABRICKS_ENTRYPOINT_FILE", raising=False)

    expected = str(module.__file__)
    assert module._entrypoint_file() == expected  # type: ignore[attr-defined]
