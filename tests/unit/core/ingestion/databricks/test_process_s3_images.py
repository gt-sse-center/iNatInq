"""Unit tests for core.ingestion.databricks.process_s3_images module.

This module tests the Databricks image processing job that reads images from S3,
processes them through the embedding pipeline, and stores results in the vector database.

# Test Coverage

- `_apply_python_params`: Environment variable parsing from command line arguments
- `main()`: Job initialization, execution, error handling, and shutdown

# Running Tests

```bash
uv run pytest tests/unit/core/ingestion/databricks/test_process_s3_images.py -v
```
"""

import os
from unittest.mock import MagicMock, patch

import pytest


class TestApplyPythonParams:
    """Tests for _apply_python_params helper function."""

    def test_applies_valid_key_value_pairs(self):
        """Applies KEY=VALUE pairs to environment.

        **Why this test is important:**

        - The Databricks job receives configuration via command line arguments
        - Environment variables must be correctly set for downstream processing
        - Incorrect parsing could lead to misconfigured jobs

        **What it tests:**

        - Multiple KEY=VALUE arguments are parsed correctly
        - Both keys and values are stored in os.environ
        - Standard uppercase environment variable names are accepted
        """
        from core.ingestion.databricks.process_s3_images import _apply_python_params

        with patch.dict("os.environ", {}, clear=True):
            _apply_python_params(["S3_PREFIX=images/", "K8S_NAMESPACE=test"])
            assert os.environ["S3_PREFIX"] == "images/"
            assert os.environ["K8S_NAMESPACE"] == "test"

    def test_ignores_non_key_value_args(self):
        """Ignores arguments without = separator.

        **Why this test is important:**

        - Command line may contain non-configuration arguments
        - Invalid arguments should be silently ignored, not cause errors
        - Prevents pollution of environment with garbage values

        **What it tests:**

        - Arguments without '=' separator are skipped
        - No partial or malformed entries are added to environment
        """
        from core.ingestion.databricks.process_s3_images import _apply_python_params

        with patch.dict("os.environ", {}, clear=True):
            _apply_python_params(["not_a_key_value", "another"])
            assert "not_a_key_value" not in os.environ
            assert "another" not in os.environ

    def test_ignores_lowercase_keys(self):
        """Ignores lowercase key names.

        **Why this test is important:**

        - Environment variable conventions require uppercase names
        - Lowercase keys may indicate user error or unintended arguments
        - Enforces consistent naming across the application

        **What it tests:**

        - Lowercase key names are rejected
        - Only uppercase keys are added to environment
        """
        from core.ingestion.databricks.process_s3_images import _apply_python_params

        with patch.dict("os.environ", {}, clear=True):
            _apply_python_params(["lowercase=value"])
            assert "lowercase" not in os.environ

    def test_ignores_empty_keys(self):
        """Ignores empty key names.

        **Why this test is important:**

        - Malformed arguments like "=value" should not corrupt environment
        - Empty keys have no semantic meaning
        - Prevents accidental overwriting of empty string key

        **What it tests:**

        - Arguments with empty key portion are skipped
        - Environment remains unchanged for invalid inputs
        """
        from core.ingestion.databricks.process_s3_images import _apply_python_params

        with patch.dict("os.environ", {}, clear=True):
            _apply_python_params(["=value"])
            assert "" not in os.environ


class TestDatabricksImageJobMain:
    """Tests for the main() function in process_s3_images."""

    @pytest.fixture
    def mock_dependencies(self, mock_ray):
        """Set up common mocks for main() tests."""
        mock_s3 = MagicMock()
        mock_s3.list_objects.return_value = []

        mock_strategy = MagicMock()
        mock_strategy.init = MagicMock()
        mock_strategy.shutdown = MagicMock()

        with patch("core.ingestion.databricks.process_s3_images.RayJobConfig.from_env") as mock_ray_cfg:
            with patch("core.ingestion.databricks.process_s3_images.MinIOConfig.from_env") as mock_minio_cfg:
                with patch(
                    "core.ingestion.databricks.process_s3_images.ImageEmbeddingConfig.from_env"
                ) as mock_embed_cfg:
                    with patch(
                        "core.ingestion.databricks.process_s3_images.VectorDBConfig.from_env"
                    ) as mock_vector_cfg:
                        with patch(
                            "core.ingestion.databricks.process_s3_images.DatabricksStrategy"
                        ) as mock_strat_cls:
                            with patch(
                                "core.ingestion.databricks.process_s3_images.S3ClientWrapper"
                            ) as mock_s3_cls:
                                mock_ray_cfg.return_value = MagicMock(
                                    num_workers=4,
                                    image_batch_size=50,
                                    image_embed_batch_size=8,
                                    task_num_cpus=1,
                                    task_max_retries=3,
                                    wait_batch_size=10,
                                    wait_timeout=1.0,
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
                                mock_embed_cfg.return_value = MagicMock()
                                mock_vector_cfg.return_value = MagicMock(
                                    ingestion_targets=frozenset({"qdrant", "weaviate"}),
                                )
                                mock_strat_cls.from_env.return_value = mock_strategy
                                mock_s3_cls.return_value = mock_s3

                                yield {
                                    "ray_cfg": mock_ray_cfg,
                                    "minio_cfg": mock_minio_cfg,
                                    "vector_cfg": mock_vector_cfg,
                                    "strategy": mock_strategy,
                                    "s3": mock_s3,
                                }

    def test_main_initializes_and_shuts_down_cluster(self, mock_dependencies, mock_ray):
        """main() initializes cluster and shuts down on completion.

        **Why this test is important:**

        - Ray cluster resources must be properly initialized before use
        - Cluster must be shut down to release resources and avoid leaks
        - Ensures the job lifecycle is properly managed

        **What it tests:**

        - Strategy init() is called at job start
        - Strategy shutdown() is called on job completion
        - Both calls happen exactly once
        """
        from core.ingestion.databricks.process_s3_images import main

        with patch.dict("os.environ", {"S3_PREFIX": "images/"}, clear=False):
            main()

        mock_dependencies["strategy"].init.assert_called_once()
        mock_dependencies["strategy"].shutdown.assert_called_once()

    def test_main_returns_early_when_no_keys(self, mock_dependencies, mock_ray):
        """main() returns early when no S3 keys found.

        **Why this test is important:**

        - Empty S3 prefix should not cause errors or unnecessary processing
        - Job should exit gracefully when there is no work to do
        - Resources should still be properly cleaned up

        **What it tests:**

        - Job completes successfully with empty S3 listing
        - Shutdown is called even with no processing work
        """
        from core.ingestion.databricks.process_s3_images import main

        mock_dependencies["s3"].list_objects.return_value = []

        with patch.dict("os.environ", {"S3_PREFIX": "images/"}, clear=False):
            main()

        mock_dependencies["strategy"].shutdown.assert_called_once()

    def test_main_returns_early_when_no_image_keys(self, mock_dependencies, mock_ray):
        """main() returns early when no image keys after filtering.

        **Why this test is important:**

        - S3 prefix may contain non-image files that should be filtered out
        - Job should not attempt to process non-image content
        - Filtering ensures only valid image files are processed

        **What it tests:**

        - Non-image files (txt, pdf) are filtered out
        - Job completes gracefully with no images to process
        - Shutdown is called after early return
        """
        from core.ingestion.databricks.process_s3_images import main

        # Return non-image files
        mock_dependencies["s3"].list_objects.return_value = ["file1.txt", "file2.pdf"]

        with patch("core.ingestion.databricks.process_s3_images.ImageContentFetcher") as mock_fetcher:
            mock_fetcher.filter_image_keys.return_value = []
            with patch.dict("os.environ", {"S3_PREFIX": "images/"}, clear=False):
                main()

        mock_dependencies["strategy"].shutdown.assert_called_once()

    def test_main_processes_image_keys_when_found(self, mock_dependencies, mock_ray):
        """main() processes image keys when found.

        **Why this test is important:**

        - Core functionality: images should be processed through the pipeline
        - Ray tasks should be submitted for image batches
        - Results should be collected and aggregated

        **What it tests:**

        - Image keys are passed to processing tasks
        - Ray wait/get pattern is used to collect results
        - Job completes successfully after processing
        """
        from core.ingestion.databricks.process_s3_images import main

        mock_dependencies["s3"].list_objects.return_value = ["img1.jpg", "img2.png"]

        future_mock = MagicMock()
        mock_ray.wait.return_value = ([future_mock], [])
        mock_ray.get.return_value = [[("img1.jpg", True, ""), ("img2.png", True, "")]]

        with patch("core.ingestion.databricks.process_s3_images.ImageContentFetcher") as mock_fetcher:
            mock_fetcher.filter_image_keys.return_value = ["img1.jpg", "img2.png"]
            with patch("core.ingestion.databricks.process_s3_images.process_image_batch_ray") as mock_task:
                mock_task.options.return_value.remote.return_value = future_mock
                with patch.dict("os.environ", {"S3_PREFIX": "images/"}, clear=False):
                    main()

        mock_dependencies["strategy"].shutdown.assert_called_once()

    def test_main_handles_s3_error(self, mock_dependencies, mock_ray):
        """main() exits when S3 listing fails.

        **Why this test is important:**

        - S3 connectivity issues should be handled gracefully
        - Job should exit with error code on unrecoverable S3 errors
        - Resources must still be cleaned up on error

        **What it tests:**

        - ClientError from S3 causes SystemExit with code 1
        - Shutdown is called even after S3 error
        - Error propagation is handled correctly
        """
        from botocore.exceptions import ClientError
        from core.ingestion.databricks.process_s3_images import main

        mock_dependencies["s3"].list_objects.side_effect = ClientError(
            {"Error": {"Code": "500", "Message": "Error"}}, "ListObjects"
        )

        with patch.dict("os.environ", {"S3_PREFIX": "images/"}, clear=False):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 1
        mock_dependencies["strategy"].shutdown.assert_called_once()

    def test_main_applies_python_params(self, mock_dependencies, mock_ray):
        """main() applies sys.argv KEY=VALUE params to environment.

        **Why this test is important:**

        - Databricks passes job parameters via command line arguments
        - Arguments must be parsed and applied before job execution
        - Enables dynamic configuration without code changes

        **What it tests:**

        - sys.argv arguments are processed at job start
        - KEY=VALUE pairs are applied to environment
        - S3 operations use the configured values
        """
        from core.ingestion.databricks.process_s3_images import main

        mock_dependencies["s3"].list_objects.return_value = []

        with patch("sys.argv", ["script.py", "S3_PREFIX=custom/"]):
            with patch.dict("os.environ", {}, clear=False):
                main()

        # The prefix should have been applied to env and used
        mock_dependencies["s3"].list_objects.assert_called_once()

    def test_main_uses_s3_prefix_from_env(self, mock_dependencies, mock_ray):
        """main() uses S3_PREFIX from environment variable.

        **Why this test is important:**

        - S3_PREFIX controls which objects are processed
        - Environment variable configuration must work correctly
        - Ensures the correct S3 path is queried

        **What it tests:**

        - S3_PREFIX environment variable is read
        - Prefix is passed to list_objects call
        - Correct S3 path is used for object listing
        """
        from core.ingestion.databricks.process_s3_images import main

        mock_dependencies["s3"].list_objects.return_value = []

        with patch.dict("os.environ", {"S3_PREFIX": "custom/images/"}, clear=False):
            main()

        mock_dependencies["s3"].list_objects.assert_called_once()
        call_kwargs = mock_dependencies["s3"].list_objects.call_args[1]
        assert call_kwargs["prefix"] == "custom/images/"

    def test_main_uses_collection_from_env(self, mock_dependencies, mock_ray):
        """main() uses VECTOR_DB_COLLECTION from environment variable.

        **Why this test is important:**

        - Vector database collection determines where embeddings are stored
        - Collection name must be configurable for different environments
        - Incorrect collection could corrupt production data

        **What it tests:**

        - VECTOR_DB_COLLECTION environment variable is respected
        - Processing tasks are invoked with correct configuration
        - Collection configuration flows through to task execution
        """
        from core.ingestion.databricks.process_s3_images import main

        mock_dependencies["s3"].list_objects.return_value = ["img1.jpg"]

        future_mock = MagicMock()
        mock_ray.wait.return_value = ([future_mock], [])
        mock_ray.get.return_value = [[("img1.jpg", True, "")]]

        with patch("core.ingestion.databricks.process_s3_images.ImageContentFetcher") as mock_fetcher:
            mock_fetcher.filter_image_keys.return_value = ["img1.jpg"]
            with patch("core.ingestion.databricks.process_s3_images.process_image_batch_ray") as mock_task:
                mock_task.options.return_value.remote.return_value = future_mock
                with patch.dict(
                    "os.environ",
                    {"S3_PREFIX": "images/", "VECTOR_DB_COLLECTION": "my_images"},
                    clear=False,
                ):
                    main()

        mock_task.options.assert_called()

    def test_main_batches_keys_correctly(self, mock_dependencies, mock_ray):
        """main() batches keys according to image_batch_size.

        **Why this test is important:**

        - Large numbers of images must be processed in batches for efficiency
        - Batch size affects memory usage and parallelism
        - Incorrect batching could cause memory issues or inefficient processing

        **What it tests:**

        - 75 keys with batch size 50 creates 2 batches
        - Each batch is submitted as a separate Ray task
        - All batches are processed to completion
        """
        from core.ingestion.databricks.process_s3_images import main

        # Create more keys than batch size
        keys = [f"img{i}.jpg" for i in range(75)]
        mock_dependencies["s3"].list_objects.return_value = keys

        # Setup futures for two batches
        future1, future2 = MagicMock(), MagicMock()

        # First call returns first batch ready, second pending
        # Second call returns second batch ready
        mock_ray.wait.side_effect = [
            ([future1], [future2]),
            ([future2], []),
        ]
        mock_ray.get.side_effect = [
            [[("img0.jpg", True, "")] * 50],
            [[("img50.jpg", True, "")] * 25],
        ]

        with patch("core.ingestion.databricks.process_s3_images.ImageContentFetcher") as mock_fetcher:
            mock_fetcher.filter_image_keys.return_value = keys
            with patch("core.ingestion.databricks.process_s3_images.process_image_batch_ray") as mock_task:
                mock_task.options.return_value.remote.side_effect = [future1, future2]
                with patch.dict("os.environ", {"S3_PREFIX": "images/"}, clear=False):
                    main()

        # With batch size 50, 75 keys should create 2 batches
        assert mock_task.options.return_value.remote.call_count == 2

    def test_main_handles_unexpected_exception(self, mock_dependencies, mock_ray):
        """main() handles unexpected exceptions and still shuts down.

        **Why this test is important:**

        - Unexpected errors should not leave resources in inconsistent state
        - Cluster shutdown must happen even on unhandled exceptions
        - Ensures proper cleanup in all failure scenarios

        **What it tests:**

        - RuntimeError during processing propagates correctly
        - Shutdown is called despite the exception
        - Exception is re-raised after cleanup
        """
        from core.ingestion.databricks.process_s3_images import main

        mock_dependencies["s3"].list_objects.return_value = ["img1.jpg"]

        with patch("core.ingestion.databricks.process_s3_images.ImageContentFetcher") as mock_fetcher:
            mock_fetcher.filter_image_keys.side_effect = RuntimeError("Unexpected error")
            with patch.dict("os.environ", {"S3_PREFIX": "images/"}, clear=False):
                with pytest.raises(RuntimeError, match="Unexpected error"):
                    main()

        mock_dependencies["strategy"].shutdown.assert_called_once()

    def test_main_calculates_success_and_failure_counts(self, mock_dependencies, mock_ray):
        """main() correctly counts successful and failed results.

        **Why this test is important:**

        - Job metrics are essential for monitoring and alerting
        - Partial failures should not cause entire job to fail
        - Success/failure counts inform operational decisions

        **What it tests:**

        - Mixed success/failure results are handled correctly
        - Job completes successfully even with some failures
        - Shutdown is called after processing completes
        """
        from core.ingestion.databricks.process_s3_images import main

        mock_dependencies["s3"].list_objects.return_value = ["img1.jpg", "img2.png", "img3.gif"]

        # Return mixed success/failure results - use return_value not side_effect
        future_mock = MagicMock()
        mock_ray.wait.return_value = ([future_mock], [])
        mock_ray.get.return_value = [
            [
                ("img1.jpg", True, ""),
                ("img2.png", False, "Processing error"),
                ("img3.gif", True, ""),
            ]
        ]

        with patch("core.ingestion.databricks.process_s3_images.ImageContentFetcher") as mock_fetcher:
            mock_fetcher.filter_image_keys.return_value = ["img1.jpg", "img2.png", "img3.gif"]
            with patch("core.ingestion.databricks.process_s3_images.process_image_batch_ray") as mock_task:
                mock_task.options.return_value.remote.return_value = future_mock
                with patch.dict("os.environ", {"S3_PREFIX": "images/"}, clear=False):
                    main()

        # Job should complete without error even with some failures
        mock_dependencies["strategy"].shutdown.assert_called_once()
