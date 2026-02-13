"""Unit tests for core.ingestion.databricks.process_s3_to_vector_dbs module."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest


class TestApplyPythonParams:
    """Tests for _apply_python_params helper function."""

    def test_applies_valid_key_value_pairs(self) -> None:
        from core.ingestion.databricks.process_s3_to_vector_dbs import _apply_python_params

        with patch.dict("os.environ", {}, clear=True):
            _apply_python_params(["S3_PREFIX=inputs/", "K8S_NAMESPACE=test"])
            assert os.environ["S3_PREFIX"] == "inputs/"
            assert os.environ["K8S_NAMESPACE"] == "test"

    def test_ignores_non_key_value_args(self) -> None:
        from core.ingestion.databricks.process_s3_to_vector_dbs import _apply_python_params

        with patch.dict("os.environ", {}, clear=True):
            _apply_python_params(["not_a_key_value", "another"])
            assert "not_a_key_value" not in os.environ
            assert "another" not in os.environ

    def test_ignores_lowercase_keys(self) -> None:
        from core.ingestion.databricks.process_s3_to_vector_dbs import _apply_python_params

        with patch.dict("os.environ", {}, clear=True):
            _apply_python_params(["lowercase=value"])
            assert "lowercase" not in os.environ

    def test_ignores_empty_keys(self) -> None:
        from core.ingestion.databricks.process_s3_to_vector_dbs import _apply_python_params

        with patch.dict("os.environ", {}, clear=True):
            _apply_python_params(["=value"])
            assert "" not in os.environ


class TestDatabricksMain:
    """Tests for main() orchestration flow."""

    @pytest.fixture
    def mock_dependencies(self, mock_ray):
        mock_s3 = MagicMock()
        mock_s3.list_objects.return_value = []

        mock_strategy = MagicMock()
        mock_strategy.init = MagicMock()
        mock_strategy.shutdown = MagicMock()

        with (
            patch("core.ingestion.databricks.process_s3_to_vector_dbs.RayJobConfig.from_env") as mock_ray_cfg,
            patch(
                "core.ingestion.databricks.process_s3_to_vector_dbs.MinIOConfig.from_env"
            ) as mock_minio_cfg,
            patch(
                "core.ingestion.databricks.process_s3_to_vector_dbs.VectorDBConfig.from_env"
            ) as mock_vector_cfg,
            patch(
                "core.ingestion.databricks.process_s3_to_vector_dbs.EmbeddingConfig.from_env"
            ) as mock_embed_cfg,
            patch(
                "core.ingestion.databricks.process_s3_to_vector_dbs.DatabricksStrategy"
            ) as mock_strategy_cls,
            patch("core.ingestion.databricks.process_s3_to_vector_dbs.S3ClientWrapper", return_value=mock_s3),
        ):
            mock_ray_cfg.return_value = MagicMock(
                num_workers=4,
                s3_batch_size=50,
                embed_batch_max=8,
                batch_upsert_size=200,
                checkpoint_enabled=False,
                ollama_requests_per_second=5,
                task_num_cpus=1,
                task_max_retries=3,
                wait_batch_size=10,
                wait_timeout=1.0,
                progress_log_interval=100,
                pipeline_concurrency=10,
                circuit_breaker_threshold=5,
                circuit_breaker_timeout=30,
                embedding_timeout=120,
                upsert_timeout=60,
                retry_max_attempts=3,
                retry_min_wait=1.0,
                retry_max_wait=10.0,
            )
            mock_minio_cfg.return_value = MagicMock(
                endpoint_url="http://minio:9000",
                access_key_id="access",
                secret_access_key="secret",
                bucket="test-bucket",
            )
            mock_vector_cfg.return_value = MagicMock(collection="test")
            mock_embed_cfg.return_value = MagicMock()
            mock_strategy_cls.from_env.return_value = mock_strategy

            yield {
                "s3": mock_s3,
                "strategy": mock_strategy,
            }

    def test_main_initializes_and_shuts_down_cluster(self, mock_dependencies, mock_ray):
        from core.ingestion.databricks.process_s3_to_vector_dbs import main

        with patch.dict("os.environ", {"S3_PREFIX": "inputs/"}, clear=False):
            main()

        mock_dependencies["strategy"].init.assert_called_once()
        mock_dependencies["strategy"].shutdown.assert_called_once()

    def test_main_returns_early_when_no_keys(self, mock_dependencies, mock_ray):
        from core.ingestion.databricks.process_s3_to_vector_dbs import main

        mock_dependencies["s3"].list_objects.return_value = []

        with patch.dict("os.environ", {"S3_PREFIX": "inputs/"}, clear=False):
            main()

        mock_dependencies["strategy"].shutdown.assert_called_once()
        mock_ray.wait.assert_not_called()

    def test_main_processes_keys_when_found(self, mock_dependencies, mock_ray):
        from core.ingestion.databricks.process_s3_to_vector_dbs import main

        mock_dependencies["s3"].list_objects.return_value = ["file1.txt", "file2.txt"]
        mock_ray.wait.return_value = ([MagicMock()], [])
        mock_ray.get.return_value = [[("file1.txt", True, ""), ("file2.txt", True, "")]]

        with patch.dict("os.environ", {"S3_PREFIX": "inputs/"}, clear=False):
            with patch("core.ingestion.databricks.process_s3_to_vector_dbs.RateLimiterActor") as mock_rate:
                mock_rate.remote.return_value = MagicMock()
                with patch(
                    "core.ingestion.databricks.process_s3_to_vector_dbs.process_s3_batch_ray"
                ) as mock_task:
                    mock_task.options.return_value.remote.return_value = MagicMock()
                    main()

        mock_dependencies["strategy"].shutdown.assert_called_once()

    def test_main_handles_s3_error(self, mock_dependencies, mock_ray):
        from botocore.exceptions import ClientError
        from core.ingestion.databricks.process_s3_to_vector_dbs import main

        mock_dependencies["s3"].list_objects.side_effect = ClientError(
            {"Error": {"Code": "500", "Message": "Error"}}, "ListObjects"
        )

        with patch.dict("os.environ", {"S3_PREFIX": "inputs/"}, clear=False):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 1
        mock_dependencies["strategy"].shutdown.assert_called_once()

    def test_main_applies_python_params(self, mock_dependencies, mock_ray):
        from core.ingestion.databricks.process_s3_to_vector_dbs import main

        mock_dependencies["s3"].list_objects.return_value = []

        with patch("sys.argv", ["script.py", "S3_PREFIX=custom/"]):
            with patch.dict("os.environ", {}, clear=False):
                main()

        mock_dependencies["s3"].list_objects.assert_called_once_with(bucket="test-bucket", prefix="custom/")
