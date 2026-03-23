"""Unit tests for Bronze-backed Databricks Ray image ingestion helpers."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from core.ingestion.databricks.batch_runner import BatchRunStats
from core.ingestion.databricks.cdc import BronzeRecord, CDCProgressCursor, CDCWindowConfig
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


def test_bronze_cdc_config_rejects_non_integer_window_size(monkeypatch) -> None:
    """CDC_WINDOW_SIZE should raise a clear domain-specific error for non-integers."""
    monkeypatch.setenv("AUTOLOADER_BRONZE_TABLE", "main.default.images_bronze")
    monkeypatch.setenv("CDC_WINDOW_SIZE", "abc")

    with pytest.raises(ValueError, match="CDC_WINDOW_SIZE must be a positive integer"):
        BronzeRayCDCConfig.from_env()


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


def test_main_happy_path_loads_window_processes_batches_and_commits_cursor() -> None:
    """main() should run the full CDC happy path and commit progress."""
    from core.ingestion.databricks.process_s3_images_from_bronze import main

    namespace = "ml-system"
    cdc_cfg = BronzeRayCDCConfig(
        bronze_table="main.default.images_bronze",
        progress_table="main.default.images_progress",
        progress_id="s3_bronze_image_ingestion",
        key_col="s3_key",
        watermark_col="discovered_at",
        window_size=5000,
    )
    window_cfg = CDCWindowConfig(
        bronze_table=cdc_cfg.bronze_table,
        progress_table=cdc_cfg.progress_table,
        progress_id=cdc_cfg.progress_id,
        key_col=cdc_cfg.key_col,
        watermark_col=cdc_cfg.watermark_col,
        window_size=cdc_cfg.window_size,
    )
    ray_cfg = MagicMock(
        image_batch_size=2,
        image_embed_batch_size=8,
        num_workers=2,
        s3_prefix="ingest/",
        task_num_cpus=1,
        task_max_retries=3,
        pipeline_concurrency=4,
        circuit_breaker_threshold=5,
        circuit_breaker_timeout=30,
        embedding_timeout=120,
        upsert_timeout=60,
        retry_max_attempts=4,
        retry_min_wait=0.5,
        retry_max_wait=5.0,
        wait_batch_size=2,
        wait_timeout=1.0,
        disable_indexing_during_ingest=False,
    )
    minio_cfg = MagicMock(
        endpoint_url="http://minio:9000",
        access_key_id="access",
        bucket="images",
    )
    minio_cfg.secret_access_key = os.getenv("UNIT_TEST_MINIO_SECRET_KEY", "")
    embed_cfg = MagicMock(clip_vector_size=512)
    vector_cfg = MagicMock(collection="images_collection", ingestion_targets=frozenset({"qdrant"}))
    strategy = MagicMock()
    spark = MagicMock()
    spark_session_cls = MagicMock()
    spark_session_cls.getActiveSession.return_value = spark

    existing_cursor = CDCProgressCursor(
        last_discovered_at=datetime(2026, 3, 20, tzinfo=UTC),
        last_s3_key="img000.jpg",
    )
    window_records = [
        BronzeRecord(s3_key="img001.jpg", discovered_at=datetime(2026, 3, 21, 1, 0, tzinfo=UTC)),
        BronzeRecord(s3_key="img002.jpg", discovered_at=datetime(2026, 3, 21, 1, 1, tzinfo=UTC)),
        BronzeRecord(s3_key="img003.jpg", discovered_at=datetime(2026, 3, 21, 1, 2, tzinfo=UTC)),
    ]

    stats = BatchRunStats(
        submitted_records=3,
        num_batches=2,
        completed_records=3,
        successful=3,
        failed=0,
        successful_keys={"img001.jpg", "img002.jpg", "img003.jpg"},
    )
    submitted_batches: list[list[str]] = []
    task_future = MagicMock(name="ray_future")
    expected_commit_cursor = CDCProgressCursor(
        last_discovered_at=datetime(2026, 3, 21, 1, 2, tzinfo=UTC),
        last_s3_key="img003.jpg",
    )

    def _fake_run_ray_batch_processing(
        *,
        batches,
        submit_batch,
        wait_batch_size,
        wait_timeout,
        max_inflight_batches,
        job_logger,
        progress_label,
        total_expected_records,
    ):
        assert wait_batch_size == ray_cfg.wait_batch_size
        assert wait_timeout == ray_cfg.wait_timeout
        assert max_inflight_batches == ray_cfg.num_workers
        assert job_logger is not None
        assert progress_label == "Bronze CDC image batch progress"
        assert total_expected_records == len(window_records)

        for batch in batches:
            submitted_batches.append(batch)
            submit_batch(batch)
        return stats

    with (
        patch(
            "core.ingestion.databricks.process_s3_images_from_bronze._apply_python_params"
        ) as mock_apply_params,
        patch(
            "core.ingestion.databricks.process_s3_images_from_bronze.BronzeRayCDCConfig.from_env",
            return_value=cdc_cfg,
        ) as mock_cdc_from_env,
        patch(
            "core.ingestion.databricks.process_s3_images_from_bronze.RayJobConfig.from_env",
            return_value=ray_cfg,
        ) as mock_ray_from_env,
        patch(
            "core.ingestion.databricks.process_s3_images_from_bronze.MinIOConfig.from_env",
            return_value=minio_cfg,
        ) as mock_minio_from_env,
        patch(
            "core.ingestion.databricks.process_s3_images_from_bronze.EmbeddingConfig.from_env",
            return_value=embed_cfg,
        ) as mock_embed_from_env,
        patch(
            "core.ingestion.databricks.process_s3_images_from_bronze.VectorDBConfig.from_env",
            return_value=vector_cfg,
        ) as mock_vector_from_env,
        patch(
            "core.ingestion.databricks.process_s3_images_from_bronze.DatabricksStrategy"
        ) as mock_strategy_cls,
        patch(
            "core.ingestion.databricks.process_s3_images_from_bronze._spark_session_class",
            return_value=spark_session_cls,
        ),
        patch(
            "core.ingestion.databricks.process_s3_images_from_bronze._assert_table_exists"
        ) as mock_assert_table_exists,
        patch(
            "core.ingestion.databricks.process_s3_images_from_bronze.ensure_progress_table"
        ) as mock_ensure_progress_table,
        patch(
            "core.ingestion.databricks.process_s3_images_from_bronze.load_progress_cursor",
            return_value=existing_cursor,
        ) as mock_load_cursor,
        patch(
            "core.ingestion.databricks.process_s3_images_from_bronze.load_next_window",
            return_value=window_records,
        ) as mock_load_window,
        patch(
            "core.ingestion.databricks.process_s3_images_from_bronze.assert_unique_window_keys"
        ) as mock_assert_unique_keys,
        patch(
            "core.ingestion.databricks.process_s3_images_from_bronze.process_image_batch_ray"
        ) as mock_process_task,
        patch(
            "core.ingestion.databricks.process_s3_images_from_bronze.run_ray_batch_processing",
            side_effect=_fake_run_ray_batch_processing,
        ) as mock_run_batches,
        patch(
            "core.ingestion.databricks.process_s3_images_from_bronze.merge_progress_cursor"
        ) as mock_merge_cursor,
        patch(
            "core.ingestion.databricks.process_s3_images_from_bronze.QdrantClientWrapper.from_config"
        ) as mock_qdrant_from_config,
        patch(
            "core.ingestion.databricks.process_s3_images_from_bronze.qdrant_indexing_disabled"
        ) as mock_qdrant_indexing,
        patch.dict("os.environ", {}, clear=False),
    ):
        mock_strategy_cls.from_env.return_value = strategy
        mock_process_task.options.return_value.remote.return_value = task_future

        main()

    mock_apply_params.assert_called_once()
    mock_cdc_from_env.assert_called_once_with()
    mock_ray_from_env.assert_called_once_with(namespace)
    mock_minio_from_env.assert_called_once_with(namespace)
    mock_embed_from_env.assert_called_once_with(namespace)
    mock_vector_from_env.assert_called_once_with(namespace)

    strategy.init.assert_called_once()
    strategy.shutdown.assert_called_once()

    mock_assert_table_exists.assert_called_once_with(spark, table_name=window_cfg.bronze_table)
    mock_ensure_progress_table.assert_called_once_with(spark, progress_table=window_cfg.progress_table)
    mock_load_cursor.assert_called_once_with(spark, config=window_cfg, collection=vector_cfg.collection)
    mock_load_window.assert_called_once_with(spark, config=window_cfg, cursor=existing_cursor)
    mock_assert_unique_keys.assert_called_once_with(window_records)

    assert submitted_batches == [["img001.jpg", "img002.jpg"], ["img003.jpg"]]
    assert mock_process_task.options.call_count == 1
    mock_process_task.options.assert_called_once_with(
        num_cpus=ray_cfg.task_num_cpus, max_retries=ray_cfg.task_max_retries
    )
    assert mock_process_task.options.return_value.remote.call_count == 2
    mock_process_task.options.return_value.remote.assert_any_call(
        s3_keys=["img001.jpg", "img002.jpg"],
        s3_endpoint=minio_cfg.endpoint_url,
        s3_access_key=minio_cfg.access_key_id,
        s3_secret_key=minio_cfg.secret_access_key,
        s3_bucket=minio_cfg.bucket,
        embedding_config=embed_cfg,
        collection=vector_cfg.collection,
        image_batch_size=ray_cfg.image_batch_size,
        image_embed_batch_size=ray_cfg.image_embed_batch_size,
        rate_limiter=None,
        pipeline_concurrency=ray_cfg.pipeline_concurrency,
        circuit_breaker_threshold=ray_cfg.circuit_breaker_threshold,
        circuit_breaker_timeout=ray_cfg.circuit_breaker_timeout,
        embedding_timeout=ray_cfg.embedding_timeout,
        upsert_timeout=ray_cfg.upsert_timeout,
        retry_max_attempts=ray_cfg.retry_max_attempts,
        retry_min_wait=ray_cfg.retry_min_wait,
        retry_max_wait=ray_cfg.retry_max_wait,
        ingestion_targets=vector_cfg.ingestion_targets,
        s3_key_prefix=ray_cfg.s3_prefix,
    )
    mock_run_batches.assert_called_once()
    mock_merge_cursor.assert_called_once_with(
        spark,
        config=window_cfg,
        collection=vector_cfg.collection,
        cursor=expected_commit_cursor,
    )
    mock_qdrant_from_config.assert_not_called()
    mock_qdrant_indexing.assert_not_called()
