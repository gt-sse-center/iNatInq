"""Unit tests for Bronze-backed Databricks Ray image ingestion helpers."""

from __future__ import annotations

import pytest

from core.ingestion.databricks.cdc import CDCWindowConfig
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


def test_bronze_cdc_config_to_window_config_maps_fields_1_to_1() -> None:
    """to_window_config should provide a centralized, exact field mapping."""
    cfg = BronzeRayCDCConfig(
        bronze_table="main.default.images_bronze",
        progress_table="main.default.images_progress",
        progress_id="s3_bronze_image_ingestion",
        key_col="s3_key",
        watermark_col="discovered_at",
        window_size=5000,
    )

    window_cfg = cfg.to_window_config()

    assert isinstance(window_cfg, CDCWindowConfig)
    assert window_cfg.bronze_table == cfg.bronze_table
    assert window_cfg.progress_table == cfg.progress_table
    assert window_cfg.progress_id == cfg.progress_id
    assert window_cfg.key_col == cfg.key_col
    assert window_cfg.watermark_col == cfg.watermark_col
    assert window_cfg.window_size == cfg.window_size
