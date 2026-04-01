"""Unit tests for Databricks Auto Loader config helpers.

Tests Auto Loader stream configuration for incremental S3 file processing with
Bronze table persistence. This is important because Auto Loader enables
efficient CDC-based ingestion without full dataset rescans.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from core.ingestion.databricks.process_s3_autoloader import (
    AutoLoaderConfig,
    _configure_s3a_for_minio,
    _resolve_source_path,
)


@pytest.fixture(autouse=True)
def _reset_minio_env(monkeypatch) -> None:
    """Ensure ambient MinIO env vars do not leak into unit tests."""
    for key in ("S3_ENDPOINT", "S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY", "S3_USE_SSL", "S3_PATH_STYLE"):
        monkeypatch.delenv(key, raising=False)


def test_autoloader_config_rejects_invalid_boolean_values(monkeypatch) -> None:
    """Invalid bool env values should fail fast with a clear message."""
    monkeypatch.setenv("S3_BUCKET", "pipeline")
    monkeypatch.setenv("AUTOLOADER_BRONZE_TABLE", "main.default.images_bronze")
    monkeypatch.setenv("AUTOLOADER_SCHEMA_LOCATION", "dbfs:/schema")
    monkeypatch.setenv("AUTOLOADER_CHECKPOINT_LOCATION", "dbfs:/checkpoints")
    monkeypatch.setenv("AUTOLOADER_INCLUDE_EXISTING_FILES", "not-a-bool")

    with pytest.raises(ValueError, match="Invalid boolean Auto Loader config"):
        AutoLoaderConfig.from_env()


def test_autoloader_config_requires_mandatory_env(monkeypatch) -> None:
    """Required Auto Loader env vars should be validated."""
    monkeypatch.delenv("S3_BUCKET", raising=False)
    monkeypatch.delenv("AUTOLOADER_BRONZE_TABLE", raising=False)
    monkeypatch.delenv("AUTOLOADER_SCHEMA_LOCATION", raising=False)
    monkeypatch.delenv("AUTOLOADER_CHECKPOINT_LOCATION", raising=False)

    with pytest.raises(ValueError, match="Missing required Auto Loader config"):
        AutoLoaderConfig.from_env()


def test_autoloader_config_reads_optional_values(monkeypatch) -> None:
    """Optional env vars should override defaults when provided."""
    monkeypatch.setenv("S3_BUCKET", "pipeline")
    monkeypatch.setenv("S3_PREFIX", "images")
    monkeypatch.setenv("AUTOLOADER_BRONZE_TABLE", "main.default.images_bronze")
    monkeypatch.setenv("AUTOLOADER_SCHEMA_LOCATION", "dbfs:/schema")
    monkeypatch.setenv("AUTOLOADER_CHECKPOINT_LOCATION", "dbfs:/checkpoints")
    monkeypatch.setenv("AUTOLOADER_FILE_FORMAT", "binaryFile")
    monkeypatch.setenv("AUTOLOADER_INCLUDE_EXISTING_FILES", "false")
    monkeypatch.setenv("AUTOLOADER_MAX_FILES_PER_TRIGGER", "200")
    monkeypatch.setenv("AUTOLOADER_TRIGGER_MODE", "processingTime")
    monkeypatch.setenv("AUTOLOADER_TRIGGER_INTERVAL", "10 minutes")

    cfg = AutoLoaderConfig.from_env()

    assert cfg.source_path == "s3://pipeline/images"
    assert cfg.bronze_table == "main.default.images_bronze"
    assert cfg.schema_location == "dbfs:/schema"
    assert cfg.checkpoint_location == "dbfs:/checkpoints"
    assert cfg.file_format == "binaryFile"
    assert cfg.include_existing_files is False
    assert cfg.max_files_per_trigger == 200
    assert cfg.trigger_mode == "processingTime"
    assert cfg.processing_time_interval == "10 minutes"


def test_autoloader_config_rejects_invalid_trigger_mode(monkeypatch) -> None:
    """Invalid AUTOLOADER_TRIGGER_MODE values should fail fast."""
    monkeypatch.setenv("S3_BUCKET", "pipeline")
    monkeypatch.setenv("AUTOLOADER_BRONZE_TABLE", "main.default.images_bronze")
    monkeypatch.setenv("AUTOLOADER_SCHEMA_LOCATION", "dbfs:/schema")
    monkeypatch.setenv("AUTOLOADER_CHECKPOINT_LOCATION", "dbfs:/checkpoints")
    monkeypatch.setenv("AUTOLOADER_TRIGGER_MODE", "invalid-mode")

    with pytest.raises(ValueError, match="AUTOLOADER_TRIGGER_MODE"):
        AutoLoaderConfig.from_env()


def test_autoloader_config_reads_minio_s3_settings(monkeypatch) -> None:
    """MinIO connection settings should be parsed when provided."""
    monkeypatch.setenv("S3_BUCKET", "pipeline")
    monkeypatch.setenv("S3_PREFIX", "images")
    monkeypatch.setenv("AUTOLOADER_BRONZE_TABLE", "main.default.images_bronze")
    monkeypatch.setenv("AUTOLOADER_SCHEMA_LOCATION", "dbfs:/schema")
    monkeypatch.setenv("AUTOLOADER_CHECKPOINT_LOCATION", "dbfs:/checkpoints")
    monkeypatch.setenv("S3_ENDPOINT", "http://minio:9000")
    monkeypatch.setenv("S3_ACCESS_KEY_ID", "access")
    monkeypatch.setenv("S3_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("S3_USE_SSL", "false")
    monkeypatch.setenv("S3_PATH_STYLE", "true")

    cfg = AutoLoaderConfig.from_env()

    assert cfg.s3_endpoint == "http://minio:9000"
    assert cfg.s3_access_key_id == "access"
    assert cfg.s3_secret_access_key == "secret"
    assert cfg.s3_use_ssl is False
    assert cfg.s3_path_style is True


def test_autoloader_config_requires_complete_minio_s3_group(monkeypatch) -> None:
    """Partial MinIO S3 settings should fail fast with a clear error."""
    monkeypatch.setenv("S3_BUCKET", "pipeline")
    monkeypatch.setenv("AUTOLOADER_BRONZE_TABLE", "main.default.images_bronze")
    monkeypatch.setenv("AUTOLOADER_SCHEMA_LOCATION", "dbfs:/schema")
    monkeypatch.setenv("AUTOLOADER_CHECKPOINT_LOCATION", "dbfs:/checkpoints")
    monkeypatch.setenv("S3_ENDPOINT", "http://minio:9000")
    monkeypatch.delenv("S3_ACCESS_KEY_ID", raising=False)
    monkeypatch.setenv("S3_SECRET_ACCESS_KEY", "secret")

    with pytest.raises(ValueError, match="Missing required MinIO S3 config"):
        AutoLoaderConfig.from_env()


def test_resolve_source_path_uses_s3a_for_explicit_endpoint() -> None:
    """S3 paths should be normalized to s3a:// when explicit endpoint is configured."""
    cfg = AutoLoaderConfig(
        source_path="s3://bucket/prefix",
        bronze_table="main.default.images_bronze",
        schema_location="dbfs:/schema",
        checkpoint_location="dbfs:/checkpoints",
        s3_endpoint="http://minio:9000",
        s3_access_key_id="access",
        s3_secret_access_key="secret",
    )

    assert _resolve_source_path(cfg.source_path, cfg=cfg) == "s3a://bucket/prefix"


def test_configure_s3a_for_minio_sets_expected_spark_confs() -> None:
    """Explicit MinIO settings should map to Hadoop S3A Spark configs."""
    cfg = AutoLoaderConfig(
        source_path="s3://bucket/prefix",
        bronze_table="main.default.images_bronze",
        schema_location="dbfs:/schema",
        checkpoint_location="dbfs:/checkpoints",
        s3_endpoint="http://minio:9000",
        s3_access_key_id="access",
        s3_secret_access_key="secret",
        s3_use_ssl=False,
        s3_path_style=True,
    )
    spark = MagicMock()

    _configure_s3a_for_minio(spark, cfg=cfg)

    expected = {
        ("spark.hadoop.fs.s3a.endpoint", "http://minio:9000"),
        ("spark.hadoop.fs.s3a.access.key", "access"),
        ("spark.hadoop.fs.s3a.secret.key", "secret"),
        ("spark.hadoop.fs.s3a.path.style.access", "true"),
        ("spark.hadoop.fs.s3a.connection.ssl.enabled", "false"),
    }
    actual = {(call.args[0], call.args[1]) for call in spark.conf.set.call_args_list}
    for pair in expected:
        assert pair in actual


def test_configure_s3a_for_minio_noops_when_credentials_incomplete() -> None:
    """Direct config construction with missing creds should not set Spark confs."""
    cfg = AutoLoaderConfig(
        source_path="s3://bucket/prefix",
        bronze_table="main.default.images_bronze",
        schema_location="dbfs:/schema",
        checkpoint_location="dbfs:/checkpoints",
        s3_endpoint="http://minio:9000",
        s3_access_key_id=None,
        s3_secret_access_key="secret",
    )
    spark = MagicMock()

    _configure_s3a_for_minio(spark, cfg=cfg)

    spark.conf.set.assert_not_called()
