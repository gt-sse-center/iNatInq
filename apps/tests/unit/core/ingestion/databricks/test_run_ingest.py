"""Unit tests for Databricks ingestion entrypoint helpers."""

from __future__ import annotations

import importlib
import sys
import types


def test_load_params_sets_env(monkeypatch) -> None:
    """_load_params should apply KEY=VALUE pairs and ignore non-assignments."""
    dummy_ingest = types.ModuleType("core.ingestion.databricks.process_s3_to_qdrant")
    dummy_ingest.main = lambda: None
    dummy_logger = types.ModuleType("foundation.logger")
    dummy_logger.LOGGING_CONFIG = {"version": 1}

    monkeypatch.setitem(sys.modules, "core.ingestion.databricks.process_s3_to_qdrant", dummy_ingest)
    monkeypatch.setitem(sys.modules, "foundation.logger", dummy_logger)

    module = importlib.import_module("core.ingestion.databricks.run_ingest")

    monkeypatch.delenv("FOO", raising=False)
    monkeypatch.delenv("BAR", raising=False)
    monkeypatch.delenv("NOPE", raising=False)

    module._load_params(["FOO=1", "NOPE", "BAR=two=three"])  # type: ignore[attr-defined]

    assert module.os.environ["FOO"] == "1"
    assert module.os.environ["BAR"] == "two=three"
    assert module.os.environ.get("NOPE") is None
