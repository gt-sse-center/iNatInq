"""Unit tests for Bronze CDC helper utilities."""

from __future__ import annotations

import datetime as dt
from unittest.mock import MagicMock

import pytest
from core.ingestion.databricks import cdc
from core.ingestion.databricks.cdc import (
    BronzeRecord,
    CDCProgressCursor,
    CDCWindowConfig,
    compute_commit_cursor,
    merge_progress_cursor,
)


def _ts(seconds: int) -> dt.datetime:
    """Create deterministic UTC timestamps for tests."""
    return dt.datetime.fromtimestamp(seconds, tz=dt.UTC)


def test_compute_commit_cursor_returns_none_for_empty_window() -> None:
    """No rows means no cursor advancement."""
    cursor = compute_commit_cursor(window_records=[], successful_keys=set())
    assert cursor is None


def test_compute_commit_cursor_returns_none_when_first_row_fails() -> None:
    """Cursor must not advance if first ordered key is unsuccessful."""
    rows = [
        BronzeRecord(s3_key="a.jpg", discovered_at=_ts(1)),
        BronzeRecord(s3_key="b.jpg", discovered_at=_ts(2)),
    ]
    cursor = compute_commit_cursor(window_records=rows, successful_keys={"b.jpg"})
    assert cursor is None


def test_compute_commit_cursor_stops_at_first_failure_boundary() -> None:
    """Cursor should commit only through contiguous successful prefix."""
    rows = [
        BronzeRecord(s3_key="a.jpg", discovered_at=_ts(1)),
        BronzeRecord(s3_key="b.jpg", discovered_at=_ts(2)),
        BronzeRecord(s3_key="c.jpg", discovered_at=_ts(3)),
    ]
    cursor = compute_commit_cursor(
        window_records=rows,
        successful_keys={"a.jpg", "c.jpg"},
    )
    assert cursor is not None
    assert cursor.last_s3_key == "a.jpg"
    assert cursor.last_discovered_at == _ts(1)


def test_compute_commit_cursor_advances_to_last_when_all_succeed() -> None:
    """Cursor should move to final key when full window succeeds."""
    rows = [
        BronzeRecord(s3_key="a.jpg", discovered_at=_ts(1)),
        BronzeRecord(s3_key="b.jpg", discovered_at=_ts(2)),
    ]
    cursor = compute_commit_cursor(window_records=rows, successful_keys={"a.jpg", "b.jpg"})
    assert cursor is not None
    assert cursor.last_s3_key == "b.jpg"
    assert cursor.last_discovered_at == _ts(2)


def test_compute_commit_cursor_raises_for_duplicate_keys() -> None:
    """CDC windows must not contain duplicate s3_key values."""
    rows = [
        BronzeRecord(s3_key="dup.jpg", discovered_at=_ts(1)),
        BronzeRecord(s3_key="dup.jpg", discovered_at=_ts(2)),
    ]
    with pytest.raises(ValueError, match="Duplicate s3_key"):
        compute_commit_cursor(window_records=rows, successful_keys={"dup.jpg"})


def test_monotonic_progress_update_condition_contains_ahead_only_predicate() -> None:
    """Merge update predicate should allow only forward cursor movement."""
    condition = cdc._monotonic_progress_update_condition()
    assert "source.last_discovered_at > target.last_discovered_at" in condition
    assert "source.last_discovered_at = target.last_discovered_at" in condition
    assert "source.last_s3_key > target.last_s3_key" in condition


def test_merge_progress_cursor_uses_monotonic_when_matched_condition(monkeypatch) -> None:
    """Matched merge updates must be guarded against cursor rewinds."""
    spark = MagicMock()
    update_df = MagicMock()
    update_df.withColumn.return_value = update_df
    spark.createDataFrame.return_value = update_df

    sf = MagicMock()
    sf.current_timestamp.return_value = "now-ts"
    monkeypatch.setattr(cdc, "_spark_functions", lambda: sf)

    merge_builder = MagicMock()
    merge_builder.merge.return_value = merge_builder
    merge_builder.whenMatchedUpdate.return_value = merge_builder
    merge_builder.whenNotMatchedInsert.return_value = merge_builder

    delta_table = MagicMock()
    delta_table.alias.return_value = merge_builder
    monkeypatch.setattr(cdc, "_delta_table_for_name", lambda _spark, _table: delta_table)

    config = CDCWindowConfig(
        bronze_table="main.default.images_bronze",
        progress_table="main.default.images_progress",
        progress_id="test-progress",
    )
    cursor = CDCProgressCursor(
        last_discovered_at=_ts(10),
        last_s3_key="k10.jpg",
    )

    merge_progress_cursor(
        spark,
        config=config,
        collection="documents",
        cursor=cursor,
    )

    merge_builder.whenMatchedUpdate.assert_called_once()
    kwargs = merge_builder.whenMatchedUpdate.call_args.kwargs
    assert kwargs["condition"] == cdc._monotonic_progress_update_condition()


def test_cdc_window_config_rejects_unsafe_bronze_table_name() -> None:
    """Bronze table identifiers should be validated at config construction."""
    with pytest.raises(ValueError, match="Unsafe table identifier"):
        CDCWindowConfig(
            bronze_table="main.default.images bronze",
            progress_table="main.default.images_progress",
            progress_id="test-progress",
        )


def test_cdc_window_config_rejects_unsafe_progress_table_name() -> None:
    """Progress table identifiers should be validated at config construction."""
    with pytest.raises(ValueError, match="Unsafe table identifier"):
        CDCWindowConfig(
            bronze_table="main.default.images_bronze",
            progress_table="main.default.images-progress",
            progress_id="test-progress",
        )
