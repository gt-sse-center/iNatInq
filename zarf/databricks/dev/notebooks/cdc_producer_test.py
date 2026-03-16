# Databricks notebook source
# MAGIC %md
# MAGIC # CDC Producer Test Notebook
# MAGIC
# MAGIC This notebook validates the Bronze-table contract that the Auto Loader producer writes.
# MAGIC It intentionally tests table behavior using direct Delta writes so scenarios are deterministic.

# COMMAND ----------
# MAGIC %run ./cdc_test_common

# COMMAND ----------
import traceback
from datetime import datetime, timezone
from uuid import uuid4

from pyspark.sql import functions as F
from pyspark.sql import types as T


def _normalize_ts_for_assert(value: datetime) -> datetime:
    return value.replace(tzinfo=None) if value.tzinfo is not None else value


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

dbutils.widgets.text("catalog", default_catalog)
dbutils.widgets.text("schema", default_schema)
dbutils.widgets.text("test_run_suffix", default_run_suffix)

catalog = dbutils.widgets.get("catalog").strip()
schema = dbutils.widgets.get("schema").strip()
test_run_suffix = dbutils.widgets.get("test_run_suffix").strip()

bronze_table_name = build_test_table_name(prefix="inatinq_images_bronze", run_suffix=test_run_suffix)
bronze_table = qualified_table_name(catalog=catalog, schema=schema, table=bronze_table_name)
require_test_qualified_table_name(bronze_table, label="bronze_table")

print_header("Producer Test Notebook Configuration")
print(f"env_source={env_source or 'none'}")
print(f"applied_env_keys={applied_env_keys}")
print(f"catalog={catalog}")
print(f"schema={schema}")
print(f"test_run_suffix={test_run_suffix}")
print(f"bronze_table={bronze_table}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------
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
def scenario_schema_contract() -> None:
    schema_fields = {field.name: field.dataType for field in spark.table(bronze_table).schema.fields}

    expected = {
        "source_path": T.StringType,
        "s3_key": T.StringType,
        "source_modified_at": T.TimestampType,
        "source_length": T.LongType,
        "discovered_at": T.TimestampType,
        "discovered_date": T.DateType,
    }

    for column, expected_type in expected.items():
        assert_true(column in schema_fields, message=f"Missing expected column: {column}")
        assert_true(
            isinstance(schema_fields[column], expected_type),
            message=(
                f"Unexpected type for column {column}. "
                f"actual={schema_fields[column]}, expected={expected_type.__name__}"
            ),
        )


def scenario_append_accumulates_rows() -> None:
    truncate_table_if_exists(bronze_table)

    append_bronze_rows(
        bronze_table,
        [
            {
                "source_path": "s3://pipeline/images/a.jpg",
                "s3_key": "images/a.jpg",
                "source_modified_at": "2026-03-01T00:00:01+00:00",
                "source_length": 111,
                "discovered_at": "2026-03-01T00:00:10+00:00",
            },
            {
                "source_path": "s3://pipeline/images/b.jpg",
                "s3_key": "images/b.jpg",
                "source_modified_at": "2026-03-01T00:00:02+00:00",
                "source_length": 222,
                "discovered_at": "2026-03-01T00:00:20+00:00",
            },
        ],
    )
    append_bronze_rows(
        bronze_table,
        [
            {
                "source_path": "s3://pipeline/images/c.jpg",
                "s3_key": "images/c.jpg",
                "source_modified_at": "2026-03-01T00:00:03+00:00",
                "source_length": 333,
                "discovered_at": "2026-03-01T00:00:30+00:00",
            }
        ],
    )

    assert_equal(fetch_count(bronze_table), 3, message="Append should accumulate rows")
    latest = (
        spark.table(bronze_table)
        .where(F.col("s3_key") == "images/c.jpg")
        .select("discovered_date")
        .collect()[0]["discovered_date"]
    )
    assert_equal(str(latest), "2026-03-01", message="discovered_date must be derived from discovered_at")


def scenario_duplicate_key_revisions() -> None:
    truncate_table_if_exists(bronze_table)

    append_bronze_rows(
        bronze_table,
        [
            {
                "source_path": "s3://pipeline/images/replay.jpg",
                "s3_key": "images/replay.jpg",
                "source_modified_at": "2026-03-01T00:10:01+00:00",
                "source_length": 400,
                "discovered_at": "2026-03-01T00:11:00+00:00",
            },
            {
                "source_path": "s3://pipeline/images/replay.jpg",
                "s3_key": "images/replay.jpg",
                "source_modified_at": "2026-03-01T00:12:01+00:00",
                "source_length": 410,
                "discovered_at": "2026-03-01T00:13:00+00:00",
            },
        ],
    )

    replay_df = spark.table(bronze_table).where(F.col("s3_key") == "images/replay.jpg")
    assert_equal(replay_df.count(), 2, message="Expected two rows for replayed s3_key")

    latest_ts = replay_df.agg(F.max("discovered_at").alias("ts")).collect()[0]["ts"]
    expected_ts = _normalize_ts_for_assert(datetime.fromisoformat("2026-03-01T00:13:00+00:00"))
    assert_equal(
        _normalize_ts_for_assert(latest_ts),
        expected_ts,
        message="Latest replay timestamp mismatch",
    )


def scenario_ordering_signal() -> None:
    truncate_table_if_exists(bronze_table)

    append_bronze_rows(
        bronze_table,
        [
            {
                "source_path": "s3://pipeline/images/z.jpg",
                "s3_key": "images/z.jpg",
                "source_modified_at": "2026-03-01T01:00:01+00:00",
                "source_length": 99,
                "discovered_at": "2026-03-01T01:00:30+00:00",
            },
            {
                "source_path": "s3://pipeline/images/a.jpg",
                "s3_key": "images/a.jpg",
                "source_modified_at": "2026-03-01T01:00:02+00:00",
                "source_length": 98,
                "discovered_at": "2026-03-01T01:00:30+00:00",
            },
            {
                "source_path": "s3://pipeline/images/m.jpg",
                "s3_key": "images/m.jpg",
                "source_modified_at": "2026-03-01T01:00:03+00:00",
                "source_length": 97,
                "discovered_at": "2026-03-01T01:00:20+00:00",
            },
        ],
    )

    ordered_keys = [
        row["s3_key"]
        for row in (
            spark.table(bronze_table)
            .orderBy(F.col("discovered_at").asc(), F.col("s3_key").asc())
            .select("s3_key")
            .collect()
        )
    ]
    assert_equal(
        ordered_keys,
        ["images/m.jpg", "images/a.jpg", "images/z.jpg"],
        message="Ordering by discovered_at + s3_key should be deterministic",
    )


def scenario_data_quality_detection() -> None:
    truncate_table_if_exists(bronze_table)

    append_bronze_rows(
        bronze_table,
        [
            {
                "source_path": "s3://pipeline/images/valid.jpg",
                "s3_key": "images/valid.jpg",
                "source_modified_at": "2026-03-01T02:00:01+00:00",
                "source_length": 121,
                "discovered_at": "2026-03-01T02:00:20+00:00",
            },
            {
                "source_path": "s3://pipeline/images/invalid-null-key.jpg",
                "s3_key": None,
                "source_modified_at": "2026-03-01T02:00:02+00:00",
                "source_length": 122,
                "discovered_at": "2026-03-01T02:00:21+00:00",
            },
            {
                "source_path": "s3://pipeline/images/invalid-null-ts.jpg",
                "s3_key": "images/invalid-null-ts.jpg",
                "source_modified_at": "2026-03-01T02:00:03+00:00",
                "source_length": 123,
                "discovered_at": None,
            },
        ],
    )

    invalid_rows = (
        spark.table(bronze_table).where(F.col("s3_key").isNull() | F.col("discovered_at").isNull()).count()
    )
    assert_equal(invalid_rows, 2, message="Expected to detect two invalid Bronze rows")


# COMMAND ----------
# MAGIC %md
# MAGIC ## Results

# COMMAND ----------
execution_error: Exception | None = None
cleanup_error: str | None = None

try:
    ensure_schema(catalog=catalog, schema=schema)
    drop_table_if_exists(bronze_table)
    create_bronze_table_if_missing(bronze_table)

    print_header("Initial Bronze Snapshot")
    print(f"initial_row_count={fetch_count(bronze_table)}")
    display(show_recent(bronze_table, limit=20))

    run_scenario("Schema contract", scenario_schema_contract)
    run_scenario("Append accumulates rows", scenario_append_accumulates_rows)
    run_scenario("Duplicate key revisions", scenario_duplicate_key_revisions)
    run_scenario("Deterministic ordering signal", scenario_ordering_signal)
    run_scenario("Data-quality invalid-row detection", scenario_data_quality_detection)

    results_df = spark.createDataFrame(scenario_results)
    display(results_df.orderBy("scenario"))
except Exception as exc:
    execution_error = exc
finally:
    try:
        drop_table_if_exists(bronze_table)
    except Exception as exc:
        cleanup_error = str(exc)

failed = [row for row in scenario_results if row["status"] == "FAIL"]
if execution_error is not None:
    if cleanup_error:
        raise RuntimeError(
            f"Producer test execution failed: {execution_error}. "
            f"Cleanup failed for {bronze_table}: {cleanup_error}"
        ) from execution_error
    raise execution_error
if failed:
    raise AssertionError(f"Producer test notebook failed scenarios: {failed}")
if cleanup_error:
    raise RuntimeError(f"Producer cleanup failed for {bronze_table}: {cleanup_error}")

print_header("Producer notebook complete")
print(f"All {len(scenario_results)} scenarios passed")
print(f"Dropped test table: {bronze_table}")
