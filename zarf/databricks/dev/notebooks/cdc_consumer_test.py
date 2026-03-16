# Databricks notebook source
# MAGIC %md
# MAGIC # CDC Consumer Test Notebook
# MAGIC
# MAGIC This notebook validates Bronze -> CDC cursor behavior by directly manipulating
# MAGIC the producer-created Bronze Delta table and the CDC progress Delta table.

# COMMAND ----------
# MAGIC %run ./cdc_test_common

# COMMAND ----------
from __future__ import annotations

import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from pyspark.sql import functions as F

# COMMAND ----------
# MAGIC %md
# MAGIC ## Runtime Parameters

# COMMAND ----------
dbutils.widgets.text("env_file", "")
env_file = dbutils.widgets.get("env_file").strip()
env_values, env_source = load_databricks_env_local(explicit_path=env_file or None)
applied_env_keys = apply_env_defaults(
    env_values,
    keys=[
        "DATABRICKS_HOST",
        "DATABRICKS_TOKEN",
        "DATABRICKS_CLUSTER_ID",
        "DATABRICKS_JOB_ID",
        "DATABRICKS_S3_AUTOLOADER_JOB_ID",
        "DATABRICKS_FROM_BRONZE_JOB_ID",
        "INATINQ_SRC_DIR",
    ],
)

default_catalog = env_values.get("CDC_TEST_CATALOG", "hive_metastore")
default_schema = env_values.get("CDC_TEST_SCHEMA", "default")
default_run_suffix = env_values.get("CDC_TEST_RUN_SUFFIX", "").strip() or (
    datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S") + "_" + uuid4().hex[:8]
)
default_window_size = env_values.get("CDC_TEST_WINDOW_SIZE", env_values.get("CDC_WINDOW_SIZE", "5"))
default_collection = f"documents_test_{default_run_suffix}"
default_progress_id = f"s3_bronze_image_ingestion_test_{default_run_suffix}"

dbutils.widgets.text("catalog", default_catalog)
dbutils.widgets.text("schema", default_schema)
dbutils.widgets.text("test_run_suffix", default_run_suffix)
dbutils.widgets.text("progress_id", default_progress_id)
dbutils.widgets.text("collection", default_collection)
dbutils.widgets.text("window_size", default_window_size)

catalog = dbutils.widgets.get("catalog").strip()
schema = dbutils.widgets.get("schema").strip()
test_run_suffix = dbutils.widgets.get("test_run_suffix").strip()
progress_id = dbutils.widgets.get("progress_id").strip()
collection = dbutils.widgets.get("collection").strip()
window_size = int(dbutils.widgets.get("window_size").strip())

bronze_table_name = build_test_table_name(prefix="inatinq_images_bronze", run_suffix=test_run_suffix)
progress_table_name = build_test_table_name(prefix="inatinq_images_progress", run_suffix=test_run_suffix)
bronze_table = qualified_table_name(catalog=catalog, schema=schema, table=bronze_table_name)
progress_table = qualified_table_name(catalog=catalog, schema=schema, table=progress_table_name)
require_test_qualified_table_name(bronze_table, label="bronze_table")
require_test_qualified_table_name(progress_table, label="progress_table")
if "_test_" not in progress_id and not progress_id.endswith("_test"):
    raise ValueError(f"progress_id must be test-only, got: {progress_id!r}")
if "_test_" not in collection and not collection.endswith("_test"):
    raise ValueError(f"collection must be test-only, got: {collection!r}")

print_header("Consumer Test Notebook Configuration")
print(f"env_source={env_source or 'none'}")
print(f"applied_env_keys={applied_env_keys}")
print(f"catalog={catalog}")
print(f"schema={schema}")
print(f"test_run_suffix={test_run_suffix}")
print(f"bronze_table={bronze_table}")
print(f"progress_table={progress_table}")
print(f"progress_id={progress_id}")
print(f"collection={collection}")
print(f"window_size={window_size}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Import CDC Module from `src/`


# COMMAND ----------
def _find_src_dir(start: Path) -> Path | None:
    src_hint = os.getenv("INATINQ_SRC_DIR", "").strip()
    if src_hint:
        hinted = Path(src_hint)
        if (hinted / "core" / "ingestion" / "databricks" / "cdc.py").exists():
            return hinted
    for candidate in [start, *start.parents]:
        src_dir = candidate / "src"
        if (src_dir / "core" / "ingestion" / "databricks" / "cdc.py").exists():
            return src_dir
    repo_root = find_repo_root(start=start)
    if repo_root is not None:
        src_dir = repo_root / "src"
        if (src_dir / "core" / "ingestion" / "databricks" / "cdc.py").exists():
            return src_dir
    return None


src_dir = _find_src_dir(Path.cwd())
if src_dir is None:
    raise RuntimeError(
        "Could not locate src/ directory from current notebook path. "
        "Run this notebook from the iNatInq repo context."
    )

if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from core.ingestion.databricks.cdc import (  # noqa: E402
    CDCProgressCursor,
    CDCWindowConfig,
    compute_commit_cursor,
    ensure_progress_table,
    load_next_window,
    load_progress_cursor,
    merge_progress_cursor,
)

print(f"Imported CDC module from: {src_dir}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Setup Helpers

# COMMAND ----------


def _ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _normalize_ts_for_assert(value: datetime) -> datetime:
    return value.replace(tzinfo=None) if value.tzinfo is not None else value


def _sql_escape(value: str) -> str:
    return value.replace("'", "''")


def make_row(
    s3_key: str | None,
    discovered_at: str | None,
    *,
    source_path_prefix: str = "s3://pipeline/images",
    source_length: int = 128,
) -> dict[str, object]:
    safe_key = s3_key or "null-key.jpg"
    return {
        "source_path": f"{source_path_prefix}/{safe_key}",
        "s3_key": s3_key,
        "source_modified_at": _ts("2026-03-01T00:00:00+00:00"),
        "source_length": source_length,
        "discovered_at": discovered_at,
    }


def cfg(*, current_window_size: int | None = None) -> CDCWindowConfig:
    return CDCWindowConfig(
        bronze_table=bronze_table,
        progress_table=progress_table,
        progress_id=progress_id,
        watermark_col="discovered_at",
        key_col="s3_key",
        window_size=current_window_size if current_window_size is not None else window_size,
    )


def reset_state() -> None:
    truncate_table_if_exists(bronze_table)
    truncate_table_if_exists(progress_table)


def load_cursor(*, for_collection: str | None = None) -> CDCProgressCursor | None:
    return load_progress_cursor(
        spark,
        config=cfg(),
        collection=for_collection or collection,
    )


def load_window(*, current_cursor: CDCProgressCursor | None = None, current_window_size: int | None = None):
    return load_next_window(
        spark,
        config=cfg(current_window_size=current_window_size),
        cursor=current_cursor,
    )


def commit_window(
    *,
    window_records,
    successful_keys: set[str],
    for_collection: str | None = None,
):
    commit_cursor = compute_commit_cursor(window_records=window_records, successful_keys=successful_keys)
    if commit_cursor is not None:
        merge_progress_cursor(
            spark,
            config=cfg(),
            collection=for_collection or collection,
            cursor=commit_cursor,
        )
    return commit_cursor


def progress_snapshot() -> None:
    if table_exists(progress_table):
        display(
            spark.table(progress_table)
            .where(F.col("progress_id") == progress_id)
            .where(F.col("source_table") == bronze_table)
            .orderBy(F.col("updated_at").desc_nulls_last())
        )


print("Setup executes in the final results block to guarantee cleanup on failure.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Test Harness

# COMMAND ----------
scenario_results: list[dict[str, str]] = []


def run_scenario(name: str, fn) -> None:
    print_header(f"Running Scenario: {name}")
    try:
        fn()
        scenario_results.append({"scenario": name, "status": "PASS", "details": ""})
        print(f"PASS: {name}")
    except Exception as exc:
        scenario_results.append({"scenario": name, "status": "FAIL", "details": str(exc)})
        print(f"FAIL: {name}")
        print(traceback.format_exc())


# COMMAND ----------
# MAGIC %md
# MAGIC ## Scenarios


# COMMAND ----------
def scenario_01_ordered_happy_path() -> None:
    reset_state()
    append_bronze_rows(
        bronze_table,
        [
            make_row("img/a.jpg", "2026-03-01T00:00:10+00:00"),
            make_row("img/b.jpg", "2026-03-01T00:00:20+00:00"),
            make_row("img/c.jpg", "2026-03-01T00:00:30+00:00"),
        ],
    )

    window_records = load_window(current_cursor=None, current_window_size=10)
    keys = [r.s3_key for r in window_records]
    assert_equal(keys, ["img/a.jpg", "img/b.jpg", "img/c.jpg"], message="Unexpected first window order")

    committed = commit_window(window_records=window_records, successful_keys=set(keys))
    assert_true(committed is not None, message="Expected commit cursor")
    assert_equal(committed.last_s3_key, "img/c.jpg", message="Cursor should commit to last successful key")

    cursor = load_cursor()
    assert_true(cursor is not None, message="Cursor should be persisted")
    assert_equal(cursor.last_s3_key, "img/c.jpg", message="Persisted cursor mismatch")


def scenario_02_tie_breaker_same_timestamp() -> None:
    reset_state()
    append_bronze_rows(
        bronze_table,
        [
            make_row("img/z.jpg", "2026-03-01T01:00:00+00:00"),
            make_row("img/a.jpg", "2026-03-01T01:00:00+00:00"),
            make_row("img/m.jpg", "2026-03-01T01:00:00+00:00"),
        ],
    )

    keys = [r.s3_key for r in load_window(current_cursor=None, current_window_size=10)]
    assert_equal(
        keys,
        ["img/a.jpg", "img/m.jpg", "img/z.jpg"],
        message="Rows with identical watermark must be ordered by s3_key",
    )


def scenario_03_null_filtering() -> None:
    reset_state()
    append_bronze_rows(
        bronze_table,
        [
            make_row("img/valid-1.jpg", "2026-03-01T02:00:00+00:00"),
            make_row(None, "2026-03-01T02:00:01+00:00"),
            make_row("img/invalid-null-ts.jpg", None),
            make_row("img/valid-2.jpg", "2026-03-01T02:00:02+00:00"),
        ],
    )

    keys = [r.s3_key for r in load_window(current_cursor=None, current_window_size=10)]
    assert_equal(
        keys,
        ["img/valid-1.jpg", "img/valid-2.jpg"],
        message="CDC window should ignore rows with NULL key or NULL watermark",
    )


def scenario_04_window_boundary() -> None:
    reset_state()
    rows = [make_row(f"img/w{i}.jpg", f"2026-03-01T03:00:{i:02d}+00:00") for i in range(1, 9)]
    append_bronze_rows(bronze_table, rows)

    # Run 1
    c1 = load_cursor()
    w1 = load_window(current_cursor=c1, current_window_size=3)
    k1 = [r.s3_key for r in w1]
    assert_equal(k1, ["img/w1.jpg", "img/w2.jpg", "img/w3.jpg"], message="Window 1 mismatch")
    commit_window(window_records=w1, successful_keys=set(k1))

    # Run 2
    c2 = load_cursor()
    w2 = load_window(current_cursor=c2, current_window_size=3)
    k2 = [r.s3_key for r in w2]
    assert_equal(k2, ["img/w4.jpg", "img/w5.jpg", "img/w6.jpg"], message="Window 2 mismatch")
    commit_window(window_records=w2, successful_keys=set(k2))

    # Run 3
    c3 = load_cursor()
    w3 = load_window(current_cursor=c3, current_window_size=3)
    k3 = [r.s3_key for r in w3]
    assert_equal(k3, ["img/w7.jpg", "img/w8.jpg"], message="Window 3 mismatch")
    commit_window(window_records=w3, successful_keys=set(k3))

    # Run 4 should be empty
    c4 = load_cursor()
    w4 = load_window(current_cursor=c4, current_window_size=3)
    assert_equal(len(w4), 0, message="Expected no remaining rows after draining all windows")


def scenario_05_first_row_failure_blocks_commit() -> None:
    reset_state()
    append_bronze_rows(
        bronze_table,
        [
            make_row("img/fail-first.jpg", "2026-03-01T04:00:10+00:00"),
            make_row("img/ok-second.jpg", "2026-03-01T04:00:20+00:00"),
            make_row("img/ok-third.jpg", "2026-03-01T04:00:30+00:00"),
        ],
    )

    window_records = load_window(current_cursor=None, current_window_size=10)
    committed = commit_window(
        window_records=window_records,
        successful_keys={"img/ok-second.jpg", "img/ok-third.jpg"},
    )
    assert_true(committed is None, message="Commit cursor must stay None when first ordered row fails")
    assert_true(load_cursor() is None, message="Progress table should remain empty")


def scenario_06_mid_window_partial_commit() -> None:
    reset_state()
    append_bronze_rows(
        bronze_table,
        [
            make_row("img/p1.jpg", "2026-03-01T05:00:10+00:00"),
            make_row("img/p2.jpg", "2026-03-01T05:00:20+00:00"),
            make_row("img/p3-fail.jpg", "2026-03-01T05:00:30+00:00"),
            make_row("img/p4.jpg", "2026-03-01T05:00:40+00:00"),
        ],
    )

    window_records = load_window(current_cursor=None, current_window_size=10)
    committed = commit_window(
        window_records=window_records,
        successful_keys={"img/p1.jpg", "img/p2.jpg", "img/p4.jpg"},
    )
    assert_true(committed is not None, message="Expected partial commit cursor")
    assert_equal(committed.last_s3_key, "img/p2.jpg", message="Commit should stop at first failure boundary")

    cursor = load_cursor()
    assert_true(cursor is not None, message="Cursor should be written")
    assert_equal(
        cursor.last_s3_key, "img/p2.jpg", message="Persisted cursor should stop at contiguous prefix"
    )


def scenario_07_recovery_after_fix() -> None:
    reset_state()
    append_bronze_rows(
        bronze_table,
        [
            make_row("img/r1.jpg", "2026-03-01T06:00:10+00:00"),
            make_row("img/r2-flaky.jpg", "2026-03-01T06:00:20+00:00"),
            make_row("img/r3.jpg", "2026-03-01T06:00:30+00:00"),
        ],
    )

    # Run 1: middle key fails
    w1 = load_window(current_cursor=None, current_window_size=10)
    c1 = commit_window(window_records=w1, successful_keys={"img/r1.jpg", "img/r3.jpg"})
    assert_true(c1 is not None, message="Expected first run to commit prefix")
    assert_equal(c1.last_s3_key, "img/r1.jpg", message="First run should commit only first key")

    # Run 2: retry succeeds
    cursor_after_run1 = load_cursor()
    w2 = load_window(current_cursor=cursor_after_run1, current_window_size=10)
    k2 = [r.s3_key for r in w2]
    assert_equal(k2, ["img/r2-flaky.jpg", "img/r3.jpg"], message="Retry window mismatch")

    c2 = commit_window(window_records=w2, successful_keys={"img/r2-flaky.jpg", "img/r3.jpg"})
    assert_true(c2 is not None, message="Expected second run commit")
    assert_equal(c2.last_s3_key, "img/r3.jpg", message="Second run should advance to tail key")


def scenario_08_late_old_watermark_ignored() -> None:
    reset_state()
    append_bronze_rows(
        bronze_table,
        [
            make_row("img/l1.jpg", "2026-03-01T07:00:10+00:00"),
            make_row("img/l2.jpg", "2026-03-01T07:00:20+00:00"),
        ],
    )

    w1 = load_window(current_cursor=None, current_window_size=10)
    commit_window(window_records=w1, successful_keys={r.s3_key for r in w1})

    append_bronze_rows(
        bronze_table,
        [make_row("img/late-old.jpg", "2026-03-01T07:00:15+00:00")],
    )

    w2 = load_window(current_cursor=load_cursor(), current_window_size=10)
    assert_equal(
        len(w2),
        0,
        message="Late-arriving row older than cursor watermark should be skipped by current CDC policy",
    )


def scenario_09_duplicate_key_invariant_violation() -> None:
    reset_state()
    append_bronze_rows(
        bronze_table,
        [
            make_row("img/dup.jpg", "2026-03-01T08:00:10+00:00"),
            make_row("img/dup.jpg", "2026-03-01T08:00:20+00:00"),
            make_row("img/tail.jpg", "2026-03-01T08:00:30+00:00"),
        ],
    )

    window_records = load_window(current_cursor=None, current_window_size=10)
    try:
        compute_commit_cursor(
            window_records=window_records,
            successful_keys={"img/dup.jpg"},
        )
    except ValueError as exc:
        assert_true("Duplicate s3_key" in str(exc), message="Expected duplicate-key invariant failure")
        return
    raise AssertionError("Expected duplicate-key invariant failure for duplicate window keys")


def scenario_10_monotonic_merge_no_rewind() -> None:
    reset_state()
    base = CDCProgressCursor(last_discovered_at=_ts("2026-03-01T09:00:30+00:00"), last_s3_key="img/k30.jpg")
    older = CDCProgressCursor(last_discovered_at=_ts("2026-03-01T09:00:20+00:00"), last_s3_key="img/k20.jpg")
    lower_same_ts = CDCProgressCursor(
        last_discovered_at=_ts("2026-03-01T09:00:30+00:00"),
        last_s3_key="img/k10.jpg",
    )
    higher_same_ts = CDCProgressCursor(
        last_discovered_at=_ts("2026-03-01T09:00:30+00:00"),
        last_s3_key="img/k99.jpg",
    )

    merge_progress_cursor(spark, config=cfg(), collection=collection, cursor=base)
    merge_progress_cursor(spark, config=cfg(), collection=collection, cursor=older)
    merge_progress_cursor(spark, config=cfg(), collection=collection, cursor=lower_same_ts)
    merge_progress_cursor(spark, config=cfg(), collection=collection, cursor=higher_same_ts)

    cursor = load_cursor()
    assert_true(cursor is not None, message="Expected persisted cursor")
    assert_equal(cursor.last_s3_key, "img/k99.jpg", message="Monotonic merge should keep farthest cursor")
    expected_ts = _normalize_ts_for_assert(_ts("2026-03-01T09:00:30+00:00"))
    assert_equal(
        _normalize_ts_for_assert(cursor.last_discovered_at),
        expected_ts,
        message="Monotonic merge should preserve latest watermark",
    )


def scenario_11_collection_isolation() -> None:
    reset_state()
    merge_progress_cursor(
        spark,
        config=cfg(),
        collection="documents_a",
        cursor=CDCProgressCursor(
            last_discovered_at=_ts("2026-03-01T10:00:10+00:00"),
            last_s3_key="img/a1.jpg",
        ),
    )
    merge_progress_cursor(
        spark,
        config=cfg(),
        collection="documents_b",
        cursor=CDCProgressCursor(
            last_discovered_at=_ts("2026-03-01T10:00:20+00:00"),
            last_s3_key="img/b1.jpg",
        ),
    )

    cur_a = load_cursor(for_collection="documents_a")
    cur_b = load_cursor(for_collection="documents_b")

    assert_true(cur_a is not None and cur_b is not None, message="Both collection cursors should exist")
    assert_equal(cur_a.last_s3_key, "img/a1.jpg", message="Collection A cursor mismatch")
    assert_equal(cur_b.last_s3_key, "img/b1.jpg", message="Collection B cursor mismatch")


def scenario_12_progress_reset_replays() -> None:
    reset_state()
    append_bronze_rows(
        bronze_table,
        [
            make_row("img/replay-1.jpg", "2026-03-01T11:00:10+00:00"),
            make_row("img/replay-2.jpg", "2026-03-01T11:00:20+00:00"),
        ],
    )

    w1 = load_window(current_cursor=None, current_window_size=10)
    commit_window(window_records=w1, successful_keys={r.s3_key for r in w1})
    assert_equal(
        len(load_window(current_cursor=load_cursor(), current_window_size=10)), 0, message="Expected drained"
    )

    spark.sql(
        f"""
        DELETE FROM {progress_table}
        WHERE progress_id = '{_sql_escape(progress_id)}'
          AND source_table = '{_sql_escape(bronze_table)}'
          AND collection = '{_sql_escape(collection)}'
        """
    )

    reset_cursor = load_cursor()
    assert_true(reset_cursor is None, message="Expected cursor deletion after progress reset")

    replay_window = load_window(current_cursor=reset_cursor, current_window_size=10)
    replay_keys = [r.s3_key for r in replay_window]
    assert_equal(
        replay_keys,
        ["img/replay-1.jpg", "img/replay-2.jpg"],
        message="Reset progress should replay Bronze rows from the beginning",
    )


# COMMAND ----------
# MAGIC %md
# MAGIC ## Results

# COMMAND ----------
execution_error: Exception | None = None
cleanup_errors: list[str] = []

try:
    ensure_schema(catalog=catalog, schema=schema)
    drop_table_if_exists(bronze_table)
    drop_table_if_exists(progress_table)
    create_bronze_table_if_missing(bronze_table)
    ensure_progress_table(spark, progress_table=progress_table)

    print_header("Initial Table State")
    print(f"bronze_rows={fetch_count(bronze_table)}")
    print(f"progress_rows={fetch_count(progress_table)}")
    progress_snapshot()

    run_scenario("01 Ordered happy path", scenario_01_ordered_happy_path)
    run_scenario("02 Tie-breaker on identical watermark", scenario_02_tie_breaker_same_timestamp)
    run_scenario("03 Null filtering", scenario_03_null_filtering)
    run_scenario("04 Window boundary", scenario_04_window_boundary)
    run_scenario("05 First-row failure blocks commit", scenario_05_first_row_failure_blocks_commit)
    run_scenario("06 Mid-window partial commit", scenario_06_mid_window_partial_commit)
    run_scenario("07 Recovery after fix", scenario_07_recovery_after_fix)
    run_scenario("08 Late old watermark ignored", scenario_08_late_old_watermark_ignored)
    run_scenario("09 Duplicate-key invariant violation", scenario_09_duplicate_key_invariant_violation)
    run_scenario("10 Monotonic merge no rewind", scenario_10_monotonic_merge_no_rewind)
    run_scenario("11 Collection isolation", scenario_11_collection_isolation)
    run_scenario("12 Progress reset replay", scenario_12_progress_reset_replays)

    results_df = spark.createDataFrame(scenario_results)
    display(results_df.orderBy("scenario"))
except Exception as exc:
    execution_error = exc
finally:
    for table in (progress_table, bronze_table):
        try:
            drop_table_if_exists(table)
        except Exception as exc:
            cleanup_errors.append(f"{table}: {exc}")

failed = [row for row in scenario_results if row["status"] == "FAIL"]
if execution_error is not None:
    if cleanup_errors:
        raise RuntimeError(
            f"Consumer test execution failed: {execution_error}. Cleanup failed: {' | '.join(cleanup_errors)}"
        ) from execution_error
    raise execution_error
if failed:
    raise AssertionError(f"Consumer test notebook failed scenarios: {failed}")
if cleanup_errors:
    raise RuntimeError("Consumer cleanup failed: " + " | ".join(cleanup_errors))

print_header("Consumer notebook complete")
print(f"All {len(scenario_results)} scenarios passed")
print(f"Dropped test tables: {bronze_table}, {progress_table}")
