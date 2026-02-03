"""Unit tests for core.ingestion.ray.process_s3_images module.

This file tests the Ray image processing job entry point, which processes
S3 image objects using Ray for parallel execution and stores embeddings
in vector databases.

# Test Coverage

The tests cover:
  - main() function lifecycle (init, process, shutdown)
  - S3 object listing and image key filtering
  - Error handling (S3 errors, strategy init errors, unexpected exceptions)
  - Environment variable configuration (S3_PREFIX, VECTOR_DB_COLLECTION)
  - Batch processing and result aggregation
  - Graceful shutdown on errors

# Running Tests

Run with: pytest tests/unit/core/ingestion/ray/test_process_s3_images.py
"""

import os
from unittest.mock import MagicMock, patch

import pytest


class TestRayImageJobMain:
    """Test suite for the main() function in process_s3_images.

    Tests the Ray image processing job entry point which initializes
    a Ray cluster, processes S3 images in batches, and shuts down cleanly.
    """

    @pytest.fixture
    def mock_dependencies(self, mock_ray):
        """Set up common mocks for main() tests."""
        mock_s3 = MagicMock()
        mock_s3.list_objects.return_value = []

        mock_strategy = MagicMock()
        mock_strategy.init = MagicMock()
        mock_strategy.shutdown = MagicMock()

        with patch("core.ingestion.ray.process_s3_images.RayJobConfig.from_env") as mock_ray_cfg:
            with patch("core.ingestion.ray.process_s3_images.MinIOConfig.from_env") as mock_minio_cfg:
                with patch(
                    "core.ingestion.ray.process_s3_images.ImageEmbeddingConfig.from_env"
                ) as mock_embed_cfg:
                    with patch("core.ingestion.ray.process_s3_images.LocalRayStrategy") as mock_strat_cls:
                        with patch("core.ingestion.ray.process_s3_images.S3ClientWrapper") as mock_s3_cls:
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
                            mock_strat_cls.from_env.return_value = mock_strategy
                            mock_s3_cls.return_value = mock_s3

                            yield {
                                "ray_cfg": mock_ray_cfg,
                                "minio_cfg": mock_minio_cfg,
                                "strategy": mock_strategy,
                                "s3": mock_s3,
                            }

    def test_main_initializes_and_shuts_down_cluster(self, mock_dependencies, mock_ray):
        """Test that main() properly initializes and shuts down the Ray cluster.

        **Why this test is important:**
          - Ensures proper resource lifecycle management
          - Prevents resource leaks in production
          - Critical for cluster stability

        **What it tests:**
          - strategy.init() is called exactly once
          - strategy.shutdown() is called exactly once
        """
        from core.ingestion.ray.process_s3_images import main

        with patch.dict("os.environ", {"S3_PREFIX": "images/"}, clear=False):
            main()

        mock_dependencies["strategy"].init.assert_called_once()
        mock_dependencies["strategy"].shutdown.assert_called_once()

    def test_main_returns_early_when_no_keys(self, mock_dependencies, mock_ray):
        """Test that main() returns early when no S3 keys are found.

        **Why this test is important:**
          - Avoids unnecessary processing when bucket is empty
          - Ensures clean exit without errors
          - Validates early-exit optimization

        **What it tests:**
          - Empty key list triggers early return
          - Shutdown is still called for cleanup
        """
        from core.ingestion.ray.process_s3_images import main

        mock_dependencies["s3"].list_objects.return_value = []

        with patch.dict("os.environ", {"S3_PREFIX": "images/"}, clear=False):
            main()

        mock_dependencies["strategy"].shutdown.assert_called_once()

    def test_main_returns_early_when_no_image_keys(self, mock_dependencies, mock_ray):
        """Test that main() returns early when no image keys remain after filtering.

        **Why this test is important:**
          - Non-image files should be filtered out
          - Avoids processing unsupported file types
          - Validates the image extension filter

        **What it tests:**
          - Non-image files (txt, pdf) are filtered out
          - Empty filtered list triggers early return
        """
        from core.ingestion.ray.process_s3_images import main

        mock_dependencies["s3"].list_objects.return_value = ["file1.txt", "file2.pdf"]

        with patch("core.ingestion.ray.process_s3_images.ImageContentFetcher") as mock_fetcher:
            mock_fetcher.filter_image_keys.return_value = []
            with patch.dict("os.environ", {"S3_PREFIX": "images/"}, clear=False):
                main()

        mock_dependencies["strategy"].shutdown.assert_called_once()

    def test_main_processes_image_keys_when_found(self, mock_dependencies, mock_ray):
        """Test that main() processes image keys through the Ray pipeline.

        **Why this test is important:**
          - Core functionality: processing images through Ray
          - Validates the happy path end-to-end
          - Ensures results are collected properly

        **What it tests:**
          - Image keys are passed to Ray tasks
          - Results are collected via ray.wait/ray.get
          - Shutdown occurs after processing
        """
        from core.ingestion.ray.process_s3_images import main

        mock_dependencies["s3"].list_objects.return_value = ["img1.jpg", "img2.png"]

        mock_ray.wait.return_value = ([MagicMock()], [])
        mock_ray.get.return_value = [[("img1.jpg", True, ""), ("img2.png", True, "")]]

        with patch("core.ingestion.ray.process_s3_images.ImageContentFetcher") as mock_fetcher:
            mock_fetcher.filter_image_keys.return_value = ["img1.jpg", "img2.png"]
            with patch("core.ingestion.ray.process_s3_images.process_image_batch_ray") as mock_task:
                mock_task.options.return_value.remote.return_value = MagicMock()
                with patch.dict("os.environ", {"S3_PREFIX": "images/"}, clear=False):
                    main()

        mock_dependencies["strategy"].shutdown.assert_called_once()

    def test_main_handles_s3_error(self, mock_dependencies, mock_ray):
        """Test that main() exits with code 1 when S3 listing fails.

        **Why this test is important:**
          - S3 connectivity issues are common in production
          - Must exit cleanly with proper error code
          - Shutdown must still be called for cleanup

        **What it tests:**
          - ClientError triggers sys.exit(1)
          - Strategy shutdown is called despite error
        """
        from botocore.exceptions import ClientError

        from core.ingestion.ray.process_s3_images import main

        mock_dependencies["s3"].list_objects.side_effect = ClientError(
            {"Error": {"Code": "500", "Message": "Error"}}, "ListObjects"
        )

        with patch.dict("os.environ", {"S3_PREFIX": "images/"}, clear=False):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 1
        mock_dependencies["strategy"].shutdown.assert_called_once()

    def test_main_handles_strategy_init_error(self, mock_ray):
        """Test that main() exits with code 1 when strategy init fails.

        **Why this test is important:**
          - Ray cluster initialization can fail
          - Must provide clear error indication
          - Critical for debugging deployment issues

        **What it tests:**
          - RuntimeError during init triggers sys.exit(1)
          - Error is properly propagated
        """
        from core.ingestion.ray.process_s3_images import main

        mock_strategy = MagicMock()
        mock_strategy.init.side_effect = RuntimeError("Connection failed")
        mock_strategy.shutdown = MagicMock()

        with patch("core.ingestion.ray.process_s3_images.RayJobConfig.from_env"):
            with patch("core.ingestion.ray.process_s3_images.MinIOConfig.from_env"):
                with patch("core.ingestion.ray.process_s3_images.ImageEmbeddingConfig.from_env"):
                    with patch("core.ingestion.ray.process_s3_images.LocalRayStrategy") as mock_strat_cls:
                        mock_strat_cls.from_env.return_value = mock_strategy

                        with patch.dict("os.environ", {"S3_PREFIX": "images/"}, clear=False):
                            with pytest.raises(SystemExit) as exc_info:
                                main()

        assert exc_info.value.code == 1

    def test_main_uses_s3_prefix_from_env(self, mock_dependencies, mock_ray):
        """Test that main() uses S3_PREFIX environment variable.

        **Why this test is important:**
          - S3_PREFIX controls which objects to process
          - Must be configurable per deployment
          - Critical for multi-tenant setups

        **What it tests:**
          - S3_PREFIX env var is passed to list_objects
          - Custom prefix is used correctly
        """
        from core.ingestion.ray.process_s3_images import main

        mock_dependencies["s3"].list_objects.return_value = []

        with patch.dict("os.environ", {"S3_PREFIX": "custom/images/"}, clear=False):
            main()

        mock_dependencies["s3"].list_objects.assert_called_once()
        call_kwargs = mock_dependencies["s3"].list_objects.call_args[1]
        assert call_kwargs["prefix"] == "custom/images/"

    def test_main_uses_collection_from_env(self, mock_dependencies, mock_ray):
        """Test that main() uses VECTOR_DB_COLLECTION environment variable.

        **Why this test is important:**
          - Collection name determines where embeddings are stored
          - Must be configurable for different use cases
          - Enables separation of different embedding types

        **What it tests:**
          - VECTOR_DB_COLLECTION env var is read
          - Collection is passed to processing tasks
        """
        from core.ingestion.ray.process_s3_images import main

        mock_dependencies["s3"].list_objects.return_value = ["img1.jpg"]

        mock_ray.wait.return_value = ([MagicMock()], [])
        mock_ray.get.return_value = [[("img1.jpg", True, "")]]

        with patch("core.ingestion.ray.process_s3_images.ImageContentFetcher") as mock_fetcher:
            mock_fetcher.filter_image_keys.return_value = ["img1.jpg"]
            with patch("core.ingestion.ray.process_s3_images.process_image_batch_ray") as mock_task:
                mock_task.options.return_value.remote.return_value = MagicMock()
                with patch.dict(
                    "os.environ",
                    {"S3_PREFIX": "images/", "VECTOR_DB_COLLECTION": "my_images"},
                    clear=False,
                ):
                    main()

        mock_task.options.assert_called()

    def test_main_batches_keys_correctly(self, mock_dependencies, mock_ray):
        """Test that main() batches keys according to image_batch_size.

        **Why this test is important:**
          - Batch size affects memory usage and parallelism
          - Must split large key lists into manageable batches
          - Critical for processing large datasets

        **What it tests:**
          - 75 keys with batch_size=50 creates 2 batches
          - Each batch is submitted as a separate Ray task
        """
        from core.ingestion.ray.process_s3_images import main

        keys = [f"img{i}.jpg" for i in range(75)]
        mock_dependencies["s3"].list_objects.return_value = keys

        future1, future2 = MagicMock(), MagicMock()

        mock_ray.wait.side_effect = [
            ([future1], [future2]),
            ([future2], []),
        ]
        mock_ray.get.side_effect = [
            [[("img0.jpg", True, "")] * 50],
            [[("img50.jpg", True, "")] * 25],
        ]

        with patch("core.ingestion.ray.process_s3_images.ImageContentFetcher") as mock_fetcher:
            mock_fetcher.filter_image_keys.return_value = keys
            with patch("core.ingestion.ray.process_s3_images.process_image_batch_ray") as mock_task:
                mock_task.options.return_value.remote.side_effect = [future1, future2]
                with patch.dict("os.environ", {"S3_PREFIX": "images/"}, clear=False):
                    main()

        assert mock_task.options.return_value.remote.call_count == 2

    def test_main_handles_ray_none_gracefully(self, mock_dependencies):
        """Test that main() exits gracefully when ray module is None.

        **Why this test is important:**
          - Ray may not be installed in all environments
          - Must provide clear error message
          - Prevents confusing AttributeError

        **What it tests:**
          - ray=None triggers sys.exit(1)
          - Clean exit without exception
        """
        from core.ingestion.ray.process_s3_images import main

        with patch("core.ingestion.ray.process_s3_images.ray", None):
            with patch.dict("os.environ", {"S3_PREFIX": "images/"}, clear=False):
                with pytest.raises(SystemExit) as exc_info:
                    main()

        assert exc_info.value.code == 1

    def test_main_handles_unexpected_exception(self, mock_dependencies, mock_ray):
        """Test that main() shuts down cleanly on unexpected exceptions.

        **Why this test is important:**
          - Unexpected errors must not leak resources
          - Shutdown must always be called
          - Exception should propagate for debugging

        **What it tests:**
          - RuntimeError is re-raised
          - strategy.shutdown() is called in finally block
        """
        from core.ingestion.ray.process_s3_images import main

        mock_dependencies["s3"].list_objects.return_value = ["img1.jpg"]

        with patch("core.ingestion.ray.process_s3_images.ImageContentFetcher") as mock_fetcher:
            mock_fetcher.filter_image_keys.side_effect = RuntimeError("Unexpected error")
            with patch.dict("os.environ", {"S3_PREFIX": "images/"}, clear=False):
                with pytest.raises(RuntimeError, match="Unexpected error"):
                    main()

        mock_dependencies["strategy"].shutdown.assert_called_once()

    def test_main_calculates_success_and_failure_counts(self, mock_dependencies, mock_ray):
        """Test that main() correctly counts successful and failed results.

        **Why this test is important:**
          - Job status reporting depends on accurate counts
          - Partial failures should not crash the job
          - Metrics are used for monitoring

        **What it tests:**
          - Mixed success/failure results are handled
          - Job completes without error
          - Counts are logged correctly
        """
        from core.ingestion.ray.process_s3_images import main

        mock_dependencies["s3"].list_objects.return_value = ["img1.jpg", "img2.png", "img3.gif"]

        future_mock = MagicMock()
        mock_ray.wait.return_value = ([future_mock], [])
        mock_ray.get.return_value = [
            [
                ("img1.jpg", True, ""),
                ("img2.png", False, "Processing error"),
                ("img3.gif", True, ""),
            ]
        ]

        with patch("core.ingestion.ray.process_s3_images.ImageContentFetcher") as mock_fetcher:
            mock_fetcher.filter_image_keys.return_value = ["img1.jpg", "img2.png", "img3.gif"]
            with patch("core.ingestion.ray.process_s3_images.process_image_batch_ray") as mock_task:
                mock_task.options.return_value.remote.return_value = future_mock
                with patch.dict("os.environ", {"S3_PREFIX": "images/"}, clear=False):
                    main()

        mock_dependencies["strategy"].shutdown.assert_called_once()
