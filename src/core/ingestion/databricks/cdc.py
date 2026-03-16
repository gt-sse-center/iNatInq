"""CDC helpers for Bronze->Ray incremental ingestion on Databricks.

This module provides a focused API for:
1. Reading the next incremental window from a Bronze Delta table.
2. Tracking progress in a separate Delta table.
3. Advancing progress only through the last contiguous successful record.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datetime import datetime
    from pyspark.sql import SparkSession


@dataclass(frozen=True, slots=True)
class BronzeRecord:
    """Single Bronze row used for incremental Ray processing."""

    s3_key: str
    discovered_at: datetime


@dataclass(frozen=True, slots=True)
class CDCProgressCursor:
    """Progress cursor persisted for incremental reads."""

    last_discovered_at: datetime
    last_s3_key: str


@dataclass(frozen=True, slots=True)
class CDCWindowConfig:
    """Runtime config for Bronze window selection and progress writes."""

    bronze_table: str
    progress_table: str
    progress_id: str
    watermark_col: str = "discovered_at"
    key_col: str = "s3_key"
    window_size: int = 5000


def ensure_progress_table(spark: SparkSession, *, progress_table: str) -> None:
    """Create progress Delta table if it does not already exist."""
    _require_safe_table_name(progress_table)
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {progress_table} (
            progress_id STRING,
            source_table STRING,
            collection STRING,
            last_discovered_at TIMESTAMP,
            last_s3_key STRING,
            updated_at TIMESTAMP
        )
        USING DELTA
        """
    )


def load_progress_cursor(
    spark: SparkSession,
    *,
    config: CDCWindowConfig,
    collection: str,
) -> CDCProgressCursor | None:
    """Load latest committed progress cursor for this Bronze->collection stream."""
    sf = _spark_functions()
    row = (
        spark.table(config.progress_table)
        .where(sf.col("progress_id") == config.progress_id)
        .where(sf.col("source_table") == config.bronze_table)
        .where(sf.col("collection") == collection)
        .orderBy(sf.col("updated_at").desc_nulls_last())
        .select("last_discovered_at", "last_s3_key")
        .limit(1)
        .collect()
    )
    if not row:
        return None

    last_discovered_at = row[0]["last_discovered_at"]
    if last_discovered_at is None:
        return None

    return CDCProgressCursor(
        last_discovered_at=last_discovered_at,
        last_s3_key=row[0]["last_s3_key"] or "",
    )


def load_next_window(
    spark: SparkSession,
    *,
    config: CDCWindowConfig,
    cursor: CDCProgressCursor | None,
) -> list[BronzeRecord]:
    """Load the next deterministic Bronze window after the stored cursor."""
    sf = _spark_functions()
    base_df = (
        spark.table(config.bronze_table)
        .where(sf.col(config.key_col).isNotNull())
        .where(sf.col(config.watermark_col).isNotNull())
    )

    if cursor is not None:
        base_df = base_df.where(
            (sf.col(config.watermark_col) > sf.lit(cursor.last_discovered_at))
            | (
                (sf.col(config.watermark_col) == sf.lit(cursor.last_discovered_at))
                & (sf.col(config.key_col) > sf.lit(cursor.last_s3_key))
            )
        )

    rows = (
        base_df.orderBy(sf.col(config.watermark_col).asc(), sf.col(config.key_col).asc())
        .select(config.key_col, config.watermark_col)
        .limit(max(1, config.window_size))
        .collect()
    )

    return [
        BronzeRecord(
            s3_key=row[config.key_col],
            discovered_at=row[config.watermark_col],
        )
        for row in rows
    ]


def compute_commit_cursor(
    *,
    window_records: list[BronzeRecord],
    successful_keys: set[str],
) -> CDCProgressCursor | None:
    """Compute commit cursor up to the first failed row in this ordered window."""
    if not window_records:
        return None

    _assert_unique_window_keys(window_records)

    committed: BronzeRecord | None = None
    for record in window_records:
        if record.s3_key not in successful_keys:
            break
        committed = record

    if committed is None:
        return None

    return CDCProgressCursor(
        last_discovered_at=committed.discovered_at,
        last_s3_key=committed.s3_key,
    )


def assert_unique_window_keys(window_records: list[BronzeRecord]) -> None:
    """Public invariant check used by callers before running expensive batch work."""
    _assert_unique_window_keys(window_records)


def merge_progress_cursor(
    spark: SparkSession,
    *,
    config: CDCWindowConfig,
    collection: str,
    cursor: CDCProgressCursor,
) -> None:
    """Upsert latest progress cursor for this Bronze->collection stream."""
    sf = _spark_functions()
    _require_safe_table_name(config.progress_table)

    update_df = spark.createDataFrame(
        [
            (
                config.progress_id,
                config.bronze_table,
                collection,
                cursor.last_discovered_at,
                cursor.last_s3_key,
            )
        ],
        schema=[
            "progress_id",
            "source_table",
            "collection",
            "last_discovered_at",
            "last_s3_key",
        ],
    ).withColumn("updated_at", sf.current_timestamp())

    delta_table = _delta_table_for_name(spark, config.progress_table)
    (
        delta_table.alias("target")
        .merge(
            source=update_df.alias("source"),
            condition=(
                "target.progress_id = source.progress_id "
                "AND target.source_table = source.source_table "
                "AND target.collection = source.collection"
            ),
        )
        .whenMatchedUpdate(
            condition=_monotonic_progress_update_condition(),
            set={
                "last_discovered_at": "source.last_discovered_at",
                "last_s3_key": "source.last_s3_key",
                "updated_at": "source.updated_at",
            },
        )
        .whenNotMatchedInsert(
            values={
                "progress_id": "source.progress_id",
                "source_table": "source.source_table",
                "collection": "source.collection",
                "last_discovered_at": "source.last_discovered_at",
                "last_s3_key": "source.last_s3_key",
                "updated_at": "source.updated_at",
            },
        )
        .execute()
    )


def _monotonic_progress_update_condition() -> str:
    """Return merge predicate that prevents progress rewinds on concurrent writers."""
    return (
        "target.last_discovered_at IS NULL "
        "OR source.last_discovered_at > target.last_discovered_at "
        "OR (source.last_discovered_at = target.last_discovered_at "
        "AND source.last_s3_key > target.last_s3_key)"
    )


def _spark_functions() -> Any:
    """Load pyspark SQL functions lazily to keep non-spark tests lightweight."""
    try:
        from pyspark.sql import functions as sf
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError("pyspark is required for Databricks CDC operations") from exc
    return sf


def _require_safe_table_name(table_name: str) -> None:
    """Validate table identifier to reduce accidental SQL misuse."""
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.")
    if not table_name or any(char not in allowed for char in table_name):
        raise ValueError(f"Unsafe table identifier: {table_name}")


def _delta_table_for_name(spark: SparkSession, table_name: str) -> Any:
    """Load DeltaTable lazily to avoid importing delta-spark in unit tests."""
    try:
        from delta.tables import DeltaTable
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("delta-spark is required for CDC progress merge") from exc
    return DeltaTable.forName(spark, table_name)


def _assert_unique_window_keys(window_records: list[BronzeRecord]) -> None:
    """Fail fast when an input window violates the unique s3_key invariant."""
    seen: set[str] = set()
    duplicates: set[str] = set()
    for record in window_records:
        if record.s3_key in seen:
            duplicates.add(record.s3_key)
        else:
            seen.add(record.s3_key)
    if duplicates:
        ordered = ", ".join(sorted(duplicates))
        raise ValueError(f"Duplicate s3_key values detected in CDC window: {ordered}")
