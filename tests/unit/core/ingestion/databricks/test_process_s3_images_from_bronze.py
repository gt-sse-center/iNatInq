"""Unit tests for Bronze-backed Databricks Ray image ingestion helpers."""

from __future__ import annotations

import pytest

from core.ingestion.databricks.process_s3_images_from_bronze import BronzeRayCDCConfig, _iter_key_batches


def test_bronze_cdc_config_requires_bronze_table(monkeypatch) -> None:
    """AUTOLOADER_BRONZE_TABLE is required for incremental reads."""
    monkeypatch.delenv("AUTOLOADER_BRONZE_TABLE", raising=False)
    with pytest.raises(ValueError, match="AUTOLOADER_BRONZE_TABLE"):
        BronzeRayCDCConfig.from_env()


def test_bronze_cdc_config_parses_optional_overrides(monkeypatch) -> None:
    """Optional CDC env vars should override defaults."""
    monkeypatch.setenv("AUTOLOADER_BRONZE_TABLE", "main.default.images_bronze")
    monkeypatch.setenv("CDC_PROGRESS_TABLE", "main.default.images_progress")
    monkeypatch.setenv("CDC_PROGRESS_ID", "test-progress")
    monkeypatch.setenv("CDC_KEY_COL", "custom_key")
    monkeypatch.setenv("CDC_WATERMARK_COL", "custom_ts")
    monkeypatch.setenv("CDC_WINDOW_SIZE", "123")

    cfg = BronzeRayCDCConfig.from_env()

    assert cfg.bronze_table == "main.default.images_bronze"
    assert cfg.progress_table == "main.default.images_progress"
    assert cfg.progress_id == "test-progress"
    assert cfg.key_col == "custom_key"
    assert cfg.watermark_col == "custom_ts"
    assert cfg.window_size == 123


def test_iter_key_batches_emits_fixed_size_batches() -> None:
    """Window keys should be chunked deterministically for Ray tasks."""
    keys = ["k1", "k2", "k3", "k4", "k5"]
    batches = list(_iter_key_batches(keys, batch_size=2))
    assert batches == [["k1", "k2"], ["k3", "k4"], ["k5"]]
