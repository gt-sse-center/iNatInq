"""Databricks Auto Loader job: S3 path -> Bronze Delta table.

This job is intentionally separate from Ray inference. It only discovers file
arrivals and persists Bronze metadata for downstream incremental consumers.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from logging.config import dictConfig
from typing import TYPE_CHECKING, Any

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

from core.ingestion.databricks.runtime import apply_python_params as _apply_python_params
from foundation.logger import LOGGING_CONFIG

dictConfig(LOGGING_CONFIG)

logger = logging.getLogger("pipeline.autoloader.databricks")


class _AutoLoaderBoolEnv(BaseSettings):
    """Typed boolean env settings for Auto Loader runtime flags."""

    include_existing_files: bool = Field(default=True, validation_alias="AUTOLOADER_INCLUDE_EXISTING_FILES")
    s3_use_ssl: bool = Field(default=True, validation_alias="S3_USE_SSL")
    s3_path_style: bool = Field(default=False, validation_alias="S3_PATH_STYLE")

    model_config = SettingsConfigDict(env_prefix="", extra="ignore", frozen=True)

    @classmethod
    def from_env(cls) -> _AutoLoaderBoolEnv:
        """Load and validate bool env vars via pydantic_settings."""
        try:
            return cls()
        except ValidationError as exc:
            env_by_field = {
                "include_existing_files": "AUTOLOADER_INCLUDE_EXISTING_FILES",
                "s3_use_ssl": "S3_USE_SSL",
                "s3_path_style": "S3_PATH_STYLE",
            }
            invalid = sorted(
                {
                    env_by_field.get(str(error["loc"][0]), str(error["loc"][0]))
                    for error in exc.errors()
                    if error.get("loc")
                }
            )
            details = ", ".join(invalid) if invalid else "boolean env vars"
            raise ValueError(f"Invalid boolean Auto Loader config: {details}") from exc


@dataclass(frozen=True, slots=True)
class AutoLoaderConfig:
    """Runtime config for Databricks Auto Loader ingestion."""

    source_path: str
    bronze_table: str
    schema_location: str
    checkpoint_location: str
    file_format: str = "binaryFile"
    include_existing_files: bool = True
    max_files_per_trigger: int | None = None
    trigger_mode: str = "availableNow"
    processing_time_interval: str = "5 minutes"
    s3_endpoint: str | None = None
    s3_access_key_id: str | None = None
    s3_secret_access_key: str | None = None
    s3_use_ssl: bool = True
    s3_path_style: bool = False

    @classmethod
    def from_env(cls) -> AutoLoaderConfig:
        """Load Auto Loader settings from environment variables."""
        bool_env = _AutoLoaderBoolEnv.from_env()
        s3_bucket = (os.getenv("S3_BUCKET") or "").strip()
        s3_prefix = (os.getenv("S3_PREFIX") or "").strip()
        bronze_table = (os.getenv("AUTOLOADER_BRONZE_TABLE") or "").strip()
        schema_location = (os.getenv("AUTOLOADER_SCHEMA_LOCATION") or "").strip()
        checkpoint_location = (os.getenv("AUTOLOADER_CHECKPOINT_LOCATION") or "").strip()

        missing = [
            key
            for key, value in (
                ("S3_BUCKET", s3_bucket),
                ("AUTOLOADER_BRONZE_TABLE", bronze_table),
                ("AUTOLOADER_SCHEMA_LOCATION", schema_location),
                ("AUTOLOADER_CHECKPOINT_LOCATION", checkpoint_location),
            )
            if not value
        ]
        if missing:
            raise ValueError(f"Missing required Auto Loader config: {', '.join(missing)}")

        max_files_raw = (os.getenv("AUTOLOADER_MAX_FILES_PER_TRIGGER") or "").strip()
        max_files_per_trigger: int | None = None
        if max_files_raw:
            parsed = int(max_files_raw)
            if parsed > 0:
                max_files_per_trigger = parsed

        trigger_mode_raw = (os.getenv("AUTOLOADER_TRIGGER_MODE") or "availableNow").strip()
        trigger_mode_map = {
            "availablenow": "availableNow",
            "once": "once",
            "processingtime": "processingTime",
        }
        trigger_mode = trigger_mode_map.get(trigger_mode_raw.lower())
        if trigger_mode is None:
            raise ValueError("AUTOLOADER_TRIGGER_MODE must be one of: availableNow, once, processingTime")

        s3_endpoint = (os.getenv("S3_ENDPOINT") or "").strip() or None
        s3_access_key_id = (os.getenv("S3_ACCESS_KEY_ID") or "").strip() or None
        s3_secret_access_key = (os.getenv("S3_SECRET_ACCESS_KEY") or "").strip() or None
        minio_group = {
            "S3_ENDPOINT": s3_endpoint,
            "S3_ACCESS_KEY_ID": s3_access_key_id,
            "S3_SECRET_ACCESS_KEY": s3_secret_access_key,
        }
        if any(minio_group.values()) and not all(minio_group.values()):
            missing = [key for key, value in minio_group.items() if not value]
            raise ValueError(f"Missing required MinIO S3 config for Auto Loader: {', '.join(missing)}")

        source_path = _build_source_path(s3_bucket=s3_bucket, s3_prefix=s3_prefix)

        return cls(
            source_path=source_path,
            bronze_table=bronze_table,
            schema_location=schema_location,
            checkpoint_location=checkpoint_location,
            file_format=(os.getenv("AUTOLOADER_FILE_FORMAT") or "binaryFile").strip(),
            include_existing_files=bool_env.include_existing_files,
            max_files_per_trigger=max_files_per_trigger,
            trigger_mode=trigger_mode,
            processing_time_interval=(os.getenv("AUTOLOADER_TRIGGER_INTERVAL") or "5 minutes").strip(),
            s3_endpoint=s3_endpoint,
            s3_access_key_id=s3_access_key_id,
            s3_secret_access_key=s3_secret_access_key,
            s3_use_ssl=bool_env.s3_use_ssl,
            s3_path_style=bool_env.s3_path_style,
        )


def _build_source_path(*, s3_bucket: str, s3_prefix: str) -> str:
    """Build canonical S3 URI source path from bucket and optional prefix."""
    source_path = f"s3://{s3_bucket}"
    normalized_prefix = s3_prefix.strip("/")
    if normalized_prefix:
        source_path = f"{source_path}/{normalized_prefix}"
    return source_path


def _build_autoloader_df(spark: SparkSession, *, cfg: AutoLoaderConfig) -> DataFrame:
    """Create Auto Loader streaming DataFrame and normalize output schema."""
    sf = _spark_functions()
    _configure_s3a_for_minio(spark, cfg=cfg)
    resolved_source_path = _resolve_source_path(cfg.source_path, cfg=cfg)
    normalized_source_prefix = resolved_source_path.rstrip("/") + "/"

    stream_reader = (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", cfg.file_format)
        .option("cloudFiles.schemaLocation", cfg.schema_location)
        .option("cloudFiles.includeExistingFiles", str(cfg.include_existing_files).lower())
    )

    if cfg.max_files_per_trigger is not None:
        stream_reader = stream_reader.option(
            "cloudFiles.maxFilesPerTrigger",
            str(cfg.max_files_per_trigger),
        )

    source_df = stream_reader.load(resolved_source_path)
    escaped_prefix = re.escape(normalized_source_prefix)

    return source_df.select(
        sf.col("path").alias("source_path"),
        sf.when(
            sf.col("path").startswith(normalized_source_prefix),
            sf.regexp_replace(sf.col("path"), f"^{escaped_prefix}", ""),
        )
        .otherwise(sf.col("path"))
        .alias("s3_key"),
        sf.col("modificationTime").alias("source_modified_at"),
        sf.col("length").alias("source_length"),
        sf.current_timestamp().alias("discovered_at"),
        sf.to_date(sf.current_timestamp()).alias("discovered_date"),
    )


def _start_query(df: DataFrame, *, cfg: AutoLoaderConfig):
    """Start Auto Loader writer query with the configured trigger mode."""
    writer = (
        df.writeStream.format("delta")
        .outputMode("append")
        .option("checkpointLocation", cfg.checkpoint_location)
    )

    trigger_mode = cfg.trigger_mode.lower()
    if trigger_mode == "once":
        writer = writer.trigger(once=True)
    elif trigger_mode == "processingtime":
        writer = writer.trigger(processingTime=cfg.processing_time_interval)
    else:
        writer = writer.trigger(availableNow=True)

    return writer.toTable(cfg.bronze_table)


def main() -> None:
    """Run S3 Auto Loader ingestion into Bronze Delta table."""
    _apply_python_params(sys.argv[1:])
    cfg = AutoLoaderConfig.from_env()

    job_logger = logging.getLogger("pipeline.autoloader.job")
    job_logger.info(
        "Databricks Auto Loader job started",
        extra={
            "source_path": cfg.source_path,
            "resolved_source_path": _resolve_source_path(cfg.source_path, cfg=cfg),
            "bronze_table": cfg.bronze_table,
            "trigger_mode": cfg.trigger_mode,
            "file_format": cfg.file_format,
            "using_explicit_s3_endpoint": bool(cfg.s3_endpoint),
        },
    )
    start = time.time()

    spark_session = _spark_session_class()
    spark = spark_session.getActiveSession() or spark_session.builder.getOrCreate()
    autoloader_df = _build_autoloader_df(spark, cfg=cfg)
    query = _start_query(autoloader_df, cfg=cfg)
    query.awaitTermination()

    elapsed = round(time.time() - start, 2)
    job_logger.info(
        "Databricks Auto Loader job completed",
        extra={
            "bronze_table": cfg.bronze_table,
            "checkpoint_location": cfg.checkpoint_location,
            "elapsed_seconds": elapsed,
        },
    )


def _configure_s3a_for_minio(spark: SparkSession, *, cfg: AutoLoaderConfig) -> None:
    """Configure Spark Hadoop S3A settings when explicit MinIO/S3 creds are provided."""
    if not cfg.s3_endpoint:
        return

    spark.conf.set("spark.hadoop.fs.s3a.endpoint", cfg.s3_endpoint)
    spark.conf.set("spark.hadoop.fs.s3a.access.key", cfg.s3_access_key_id)
    spark.conf.set("spark.hadoop.fs.s3a.secret.key", cfg.s3_secret_access_key)
    spark.conf.set("spark.hadoop.fs.s3a.path.style.access", str(cfg.s3_path_style).lower())
    spark.conf.set("spark.hadoop.fs.s3a.connection.ssl.enabled", str(cfg.s3_use_ssl).lower())

    s3a_impl = _first_existing_jvm_class(
        spark,
        candidates=(
            "shaded.databricks.org.apache.hadoop.fs.s3a.S3AFileSystem",
            "org.apache.hadoop.fs.s3a.S3AFileSystem",
        ),
    )
    if s3a_impl:
        spark.conf.set("spark.hadoop.fs.s3a.impl", s3a_impl)

    creds_provider = _first_existing_jvm_class(
        spark,
        candidates=(
            "shaded.databricks.org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
            "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
        ),
    )
    if creds_provider:
        spark.conf.set("spark.hadoop.fs.s3a.aws.credentials.provider", creds_provider)
    else:
        logger.warning(
            "No compatible S3A SimpleAWSCredentialsProvider class found on this runtime; "
            "S3A credential resolution may fail."
        )


def _first_existing_jvm_class(spark: SparkSession, *, candidates: tuple[str, ...]) -> str | None:
    """Return the first JVM class available on the active Spark runtime."""
    for class_name in candidates:
        try:
            spark._jvm.java.lang.Class.forName(class_name)
            return class_name
        except Exception as exc:
            logger.debug("S3A class not available on runtime: %s", class_name, exc_info=exc)
            continue
    return None


def _resolve_source_path(source_path: str, *, cfg: AutoLoaderConfig) -> str:
    """Normalize source path for runtimes that require s3a:// with explicit endpoint creds."""
    if cfg.s3_endpoint and source_path.startswith("s3://"):
        return "s3a://" + source_path[len("s3://") :]
    return source_path


def _spark_session_class() -> Any:
    """Load SparkSession lazily for environments without pyspark."""
    try:
        from pyspark.sql import SparkSession
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError("pyspark is required for Databricks Auto Loader job") from exc
    return SparkSession


def _spark_functions():
    """Load pyspark SQL functions lazily for environments without pyspark."""
    try:
        from pyspark.sql import functions as sf
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError("pyspark is required for Databricks Auto Loader job") from exc
    return sf


if __name__ == "__main__":
    main()
