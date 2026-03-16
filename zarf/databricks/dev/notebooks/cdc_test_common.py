# Databricks notebook source
# MAGIC %md
# MAGIC # CDC Test Common Helpers
# MAGIC Shared helpers for producer/consumer CDC test notebooks.

# COMMAND ----------
from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T


SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_]+$")


def _require_safe_identifier(identifier: str, *, label: str) -> None:
    if not identifier or not SAFE_IDENTIFIER_RE.match(identifier):
        raise ValueError(f"Unsafe {label}: {identifier!r}")


def _require_safe_table_name(qualified_name: str, *, label: str = "table") -> None:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.")
    if not qualified_name or any(char not in allowed for char in qualified_name):
        raise ValueError(f"Unsafe {label}: {qualified_name!r}")


def qualified_table_name(*, catalog: str, schema: str, table: str) -> str:
    _require_safe_identifier(catalog, label="catalog")
    _require_safe_identifier(schema, label="schema")
    _require_safe_identifier(table, label="table")
    return f"{catalog}.{schema}.{table}"


def split_qualified_table_name(qualified: str) -> tuple[str, str, str] | None:
    """Split `catalog.schema.table` into parts, or return None if not fully qualified."""
    parts = [part.strip() for part in qualified.split(".")]
    if len(parts) != 3:
        return None
    catalog, schema, table = parts
    _require_safe_identifier(catalog, label="catalog")
    _require_safe_identifier(schema, label="schema")
    _require_safe_identifier(table, label="table")
    return catalog, schema, table


def require_test_table_name(table: str, *, label: str = "table") -> None:
    """Enforce test-only table naming convention."""
    _require_safe_identifier(table, label=label)
    if "_test_" not in table and not table.endswith("_test"):
        raise ValueError(f"{label} must be a test table name, got: {table!r}")


def require_test_qualified_table_name(qualified: str, *, label: str = "table") -> None:
    """Ensure a fully qualified table points to a test-only table."""
    parts = split_qualified_table_name(qualified)
    if parts is None:
        raise ValueError(f"{label} must be fully qualified as catalog.schema.table: {qualified!r}")
    _catalog, _schema, table = parts
    require_test_table_name(table, label=label)


def build_test_table_name(*, prefix: str, run_suffix: str) -> str:
    """Build deterministic test-only table name for a notebook run."""
    _require_safe_identifier(prefix, label="prefix")
    _require_safe_identifier(run_suffix, label="run_suffix")
    table_name = f"{prefix}_test_{run_suffix}"
    require_test_table_name(table_name)
    return table_name


def ensure_schema(*, catalog: str, schema: str) -> None:
    _require_safe_identifier(catalog, label="catalog")
    _require_safe_identifier(schema, label="schema")
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")


def table_exists(table_name: str) -> bool:
    _require_safe_table_name(table_name)
    return bool(spark.catalog.tableExists(table_name))


def create_bronze_table_if_missing(bronze_table: str) -> None:
    _require_safe_table_name(bronze_table, label="bronze_table")
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {bronze_table} (
            source_path STRING,
            s3_key STRING,
            source_modified_at TIMESTAMP,
            source_length BIGINT,
            discovered_at TIMESTAMP,
            discovered_date DATE
        )
        USING DELTA
        """
    )


def truncate_table_if_exists(table_name: str) -> None:
    _require_safe_table_name(table_name)
    if table_exists(table_name):
        spark.sql(f"TRUNCATE TABLE {table_name}")


def drop_table_if_exists(table_name: str) -> None:
    _require_safe_table_name(table_name)
    spark.sql(f"DROP TABLE IF EXISTS {table_name}")


def _normalize_ts(value: str | datetime | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def append_bronze_rows(bronze_table: str, rows: list[dict[str, Any]]) -> None:
    _require_safe_table_name(bronze_table, label="bronze_table")
    if not rows:
        return

    schema = T.StructType(
        [
            T.StructField("source_path", T.StringType(), True),
            T.StructField("s3_key", T.StringType(), True),
            T.StructField("source_modified_at", T.TimestampType(), True),
            T.StructField("source_length", T.LongType(), True),
            T.StructField("discovered_at", T.TimestampType(), True),
        ]
    )

    normalized: list[tuple[Any, ...]] = []
    for row in rows:
        source_path = row.get("source_path")
        s3_key = row.get("s3_key")
        source_modified_at = _normalize_ts(row.get("source_modified_at"))
        source_length = row.get("source_length", 1)
        discovered_at = _normalize_ts(row.get("discovered_at"))
        normalized.append((source_path, s3_key, source_modified_at, source_length, discovered_at))

    df = spark.createDataFrame(normalized, schema=schema).withColumn(
        "discovered_date", F.to_date(F.col("discovered_at"))
    )

    (
        df.select(
            "source_path",
            "s3_key",
            "source_modified_at",
            "source_length",
            "discovered_at",
            "discovered_date",
        )
        .write.mode("append")
        .format("delta")
        .saveAsTable(bronze_table)
    )


def fetch_rows(table_name: str, *, order_by: list[str] | None = None) -> list[Any]:
    _require_safe_table_name(table_name)
    df = spark.table(table_name)
    if order_by:
        df = df.orderBy(*order_by)
    return df.collect()


def fetch_count(table_name: str) -> int:
    _require_safe_table_name(table_name)
    return spark.table(table_name).count()


def show_recent(table_name: str, *, limit: int = 50) -> DataFrame:
    _require_safe_table_name(table_name)
    return spark.table(table_name).orderBy(F.col("discovered_at").desc_nulls_last()).limit(limit)


def assert_equal(actual: Any, expected: Any, *, message: str) -> None:
    if actual != expected:
        raise AssertionError(f"{message}. expected={expected!r}, actual={actual!r}")


def assert_true(condition: bool, *, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def print_header(title: str) -> None:
    print("=" * 88)
    print(title)
    print("=" * 88)


def find_repo_root(*, start: Path | None = None) -> Path | None:
    """Locate repository root by walking upward for pyproject.toml + zarf/databricks."""
    base = start or Path.cwd()
    for candidate in [base, *base.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "zarf" / "databricks").exists():
            return candidate
    return None


def _parse_env_line(line: str) -> tuple[str, str] | None:
    """Parse KEY=VALUE lines from dotenv-style files."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[len("export ") :].strip()
    if "=" not in stripped:
        return None

    key, value = stripped.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        return None

    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return key, value


def read_env_file(env_file: str | Path) -> dict[str, str]:
    """Read dotenv-style env file into a dict. Missing files return {}."""
    path = Path(env_file)
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_env_line(line)
        if parsed is None:
            continue
        key, value = parsed
        values[key] = value
    return values


def load_databricks_env_local(*, explicit_path: str | None = None) -> tuple[dict[str, str], str | None]:
    """Load .env.local for Databricks dev workflows.

    Resolution order:
    1) explicit_path (if provided),
    2) $INATINQ_ENV_FILE (if set),
    3) <repo_root>/zarf/databricks/dev/.env.local.
    """
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path))

    env_hint = os.getenv("INATINQ_ENV_FILE", "").strip()
    if env_hint:
        candidates.append(Path(env_hint))

    repo_root = find_repo_root()
    if repo_root is not None:
        candidates.append(repo_root / "zarf" / "databricks" / "dev" / ".env.local")

    for candidate in candidates:
        parsed = read_env_file(candidate)
        if parsed:
            return parsed, str(candidate)

    return {}, None


def apply_env_defaults(
    env_values: dict[str, str],
    *,
    keys: list[str] | None = None,
    overwrite: bool = False,
) -> list[str]:
    """Apply selected env values into process env and return keys that were set."""
    target_keys = keys or sorted(env_values.keys())
    applied: list[str] = []
    for key in target_keys:
        value = env_values.get(key)
        if value is None:
            continue
        if not overwrite and key in os.environ and os.environ.get(key):
            continue
        os.environ[key] = value
        applied.append(key)
    return applied
