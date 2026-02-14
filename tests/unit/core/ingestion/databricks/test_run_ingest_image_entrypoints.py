"""Unit tests for Databricks image ingestion entrypoint helpers."""

from __future__ import annotations

import importlib


def test_run_ingest_image_entrypoint_file_uses_explicit_env(monkeypatch) -> None:
    """run_ingest_image._entrypoint_file should prefer DATABRICKS_ENTRYPOINT_FILE when set."""
    module = importlib.import_module("core.ingestion.databricks.run_ingest_image")
    monkeypatch.setenv("DATABRICKS_ENTRYPOINT_FILE", "/tmp/custom_image_entry.py")

    assert module._entrypoint_file() == "/tmp/custom_image_entry.py"  # type: ignore[attr-defined]


def test_run_ingest_image_entrypoint_file_falls_back_to_module_file(monkeypatch) -> None:
    """run_ingest_image._entrypoint_file should use module __file__ when no env override exists."""
    module = importlib.import_module("core.ingestion.databricks.run_ingest_image")
    monkeypatch.delenv("DATABRICKS_ENTRYPOINT_FILE", raising=False)

    expected = str(module.__file__)
    assert module._entrypoint_file() == expected  # type: ignore[attr-defined]


def test_run_ingest_inat_image_entrypoint_file_uses_explicit_env(monkeypatch) -> None:
    """run_ingest_inat_image._entrypoint_file should prefer DATABRICKS_ENTRYPOINT_FILE when set."""
    module = importlib.import_module("core.ingestion.databricks.run_ingest_inat_image")
    monkeypatch.setenv("DATABRICKS_ENTRYPOINT_FILE", "/tmp/custom_inat_entry.py")

    assert module._entrypoint_file() == "/tmp/custom_inat_entry.py"  # type: ignore[attr-defined]


def test_run_ingest_inat_image_entrypoint_file_falls_back_to_module_file(monkeypatch) -> None:
    """run_ingest_inat_image._entrypoint_file should use module __file__ when no env override exists."""
    module = importlib.import_module("core.ingestion.databricks.run_ingest_inat_image")
    monkeypatch.delenv("DATABRICKS_ENTRYPOINT_FILE", raising=False)

    expected = str(module.__file__)
    assert module._entrypoint_file() == expected  # type: ignore[attr-defined]
