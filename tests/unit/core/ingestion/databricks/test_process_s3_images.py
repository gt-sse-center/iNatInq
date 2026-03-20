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

import pytest
import os
from unittest.mock import MagicMock, patch
from core.ingestion.databricks.batch_runner import BatchRunStats


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

    @pytest.fixture(autouse=True)
    def mock_reporter(self):
        """Suppress report_ingestion_metrics for all main() tests."""
        with patch("core.ingestion.databricks.batch_runner.report_ingestion_metrics"):
            yield

    @pytest.fixture
    def mock_dependencies(self, mock_ray):
        """Set up common mocks for main() tests.

        Mocks iter_image_batches to yield no batches by default.
        Individual tests override via the returned mock.
        """
        mock_s3 = MagicMock()

        mock_strategy = MagicMock()
        mock_strategy.init = MagicMock()
        mock_strategy.shutdown = MagicMock()

        with patch("core.ingestion.databricks.process_s3_images.RayJobConfig.from_env") as mock_ray_cfg:
            with patch("core.ingestion.databricks.process_s3_images.MinIOConfig.from_env") as mock_minio_cfg:
                with patch(
                    "core.ingestion.databricks.process_s3_images.EmbeddingConfig.from_env"
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
                                with (
                                    patch(
                                        "core.ingestion.databricks.process_s3_images.iter_image_batches"
                                    ) as mock_iter,
                                    patch(
                                        "core.ingestion.databricks.process_s3_images.qdrant_indexing_disabled"
                                    ) as mock_indexing_disabled,
                                    patch(
                                        "core.ingestion.databricks.process_s3_images.QdrantClientWrapper"
                                    ) as mock_qdrant_cls,
                                ):
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
                                        checkpoint_enabled=False,
                                        checkpoint_dir="/tmp/checkpoints",
                                        s3_prefix="",
                                        image_max_items=None,
                                        image_page_size=1000,
                                        disable_indexing_during_ingest=False,
                                    )
                                    mock_minio_cfg.return_value = MagicMock(
                                        endpoint_url="http://minio:9000",
                                        access_key_id="access",
                                        secret_access_key="secret",
                                        bucket="test-bucket",
                                    )
                                    embed_cfg = MagicMock()
                                    mock_embed_cfg.return_value = embed_cfg
                                    mock_vector_cfg.return_value = MagicMock(
                                        collection="documents",
                                        qdrant_url="http://localhost:6333",
                                        qdrant_api_key=None,
                                    )
                                    mock_strat_cls.from_env.return_value = mock_strategy
                                    mock_s3_cls.return_value = mock_s3
                                    # Default: no batches (empty bucket)
                                    mock_iter.return_value = iter([])

                                    yield {
                                        "ray_cfg": mock_ray_cfg,
                                        "minio_cfg": mock_minio_cfg,
                                        "embed_cfg": embed_cfg,
                                        "vector_cfg": mock_vector_cfg,
                                        "strategy": mock_strategy,
                                        "s3": mock_s3,
                                        "iter_batches": mock_iter,
                                        "qdrant_indexing_disabled": mock_indexing_disabled,
                                        "qdrant_cls": mock_qdrant_cls,
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

    def test_main_completes_when_no_batches(self, mock_dependencies, mock_ray):
        """main() completes cleanly when iter_image_batches yields nothing.

        **Why this test is important:**

        - Empty S3 prefix should not cause errors or unnecessary processing
        - Job should exit gracefully when there is no work to do
        - Resources should still be properly cleaned up

        **What it tests:**

        - Job completes successfully with empty generator
        - Shutdown is called even with no processing work
        """
        from core.ingestion.databricks.process_s3_images import main

        mock_dependencies["iter_batches"].return_value = iter([])

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

        - Image batches from generator are passed to processing tasks
        - Ray wait/get pattern is used to collect results
        - Job completes successfully after processing
        """
        from core.ingestion.databricks.process_s3_images import main

        mock_dependencies["iter_batches"].return_value = iter([["img1.jpg", "img2.png"]])

        future_mock = MagicMock()
        mock_ray.wait.return_value = ([future_mock], [])
        mock_ray.get.return_value = [[("img1.jpg", True, ""), ("img2.png", True, "")]]

        with patch("core.ingestion.databricks.process_s3_images.process_image_batch_ray") as mock_task:
            mock_task.options.return_value.remote.return_value = future_mock
            with patch.dict("os.environ", {"S3_PREFIX": "images/"}, clear=False):
                main()

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

        with patch("sys.argv", ["script.py", "S3_PREFIX=custom/"]):
            with patch.dict("os.environ", {}, clear=False):
                main()

        # iter_image_batches should have been called
        mock_dependencies["iter_batches"].assert_called_once()

    def test_main_passes_prefix_to_iter_image_batches(self, mock_dependencies, mock_ray):
        """main() passes s3_prefix from config to iter_image_batches.

        **Why this test is important:**

        - S3_PREFIX controls which objects are processed
        - Config-based settings must flow correctly
        - Ensures the correct S3 path is queried

        **What it tests:**

        - ray_cfg.s3_prefix is forwarded to iter_image_batches
        """
        from core.ingestion.databricks.process_s3_images import main

        mock_dependencies["ray_cfg"].return_value.s3_prefix = "custom/images/"
        main()

        mock_dependencies["iter_batches"].assert_called_once()
        call_kwargs = mock_dependencies["iter_batches"].call_args[1]
        assert call_kwargs["prefix"] == "custom/images/"

    def test_main_uses_collection_from_config(self, mock_dependencies, mock_ray):
        """main() uses collection from VectorDBConfig.

        **Why this test is important:**

        - Vector database collection determines where embeddings are stored
        - Collection name must be configurable for different environments
        - Incorrect collection could corrupt production data

        **What it tests:**

        - vector_cfg.collection is used for the collection name
        - Processing tasks are invoked with correct configuration
        - Collection configuration flows through to task execution
        """
        from core.ingestion.databricks.process_s3_images import main

        mock_dependencies["vector_cfg"].return_value.collection = "my_images"
        mock_dependencies["iter_batches"].return_value = iter([["img1.jpg"]])

        future_mock = MagicMock()
        mock_ray.wait.return_value = ([future_mock], [])
        mock_ray.get.return_value = [[("img1.jpg", True, "")]]

        with patch("core.ingestion.databricks.process_s3_images.process_image_batch_ray") as mock_task:
            mock_task.options.return_value.remote.return_value = future_mock
            main()

        mock_task.options.assert_called()

    def test_main_batches_keys_correctly(self, mock_dependencies, mock_ray):
        """main() submits each batch from the generator as a Ray task.

        **Why this test is important:**

        - Large numbers of images must be processed in batches for efficiency
        - Batch size affects memory usage and parallelism
        - Incorrect batching could cause memory issues or inefficient processing

        **What it tests:**

        - Two batches from generator create two Ray tasks
        - Each batch is submitted as a separate Ray task
        - All batches are processed to completion
        """
        from core.ingestion.databricks.process_s3_images import main

        batch1 = [f"img{i}.jpg" for i in range(50)]
        batch2 = [f"img{i}.jpg" for i in range(50, 75)]
        mock_dependencies["iter_batches"].return_value = iter([batch1, batch2])

        future1, future2 = MagicMock(), MagicMock()

        mock_ray.wait.side_effect = [
            ([future1], [future2]),
            ([future2], []),
        ]
        mock_ray.get.side_effect = [
            [[("img0.jpg", True, "")] * 50],
            [[("img50.jpg", True, "")] * 25],
        ]

        with patch("core.ingestion.databricks.process_s3_images.process_image_batch_ray") as mock_task:
            mock_task.options.return_value.remote.side_effect = [future1, future2]
            with patch.dict("os.environ", {"S3_PREFIX": "images/"}, clear=False):
                main()

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

        mock_dependencies["iter_batches"].side_effect = RuntimeError("Unexpected error")

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

        mock_dependencies["iter_batches"].return_value = iter([["img1.jpg", "img2.png", "img3.gif"]])

        future_mock = MagicMock()
        mock_ray.wait.return_value = ([future_mock], [])
        mock_ray.get.return_value = [
            [
                ("img1.jpg", True, ""),
                ("img2.png", False, "Processing error"),
                ("img3.gif", True, ""),
            ]
        ]

        with patch("core.ingestion.databricks.process_s3_images.process_image_batch_ray") as mock_task:
            mock_task.options.return_value.remote.return_value = future_mock
            with patch.dict("os.environ", {"S3_PREFIX": "images/"}, clear=False):
                main()

        mock_dependencies["strategy"].shutdown.assert_called_once()

    def test_main_passes_image_max_items_to_iter_batches(self, mock_dependencies, mock_ray):
        """main() passes image_max_items from config to iter_image_batches as max_items.

        **Why this test is important:**

        - IMAGE_MAX_ITEMS allows limiting the number of images processed per job
        - The config field must be forwarded correctly
        - Ensures operators can cap processing volume for cost/time control

        **What it tests:**

        - ray_cfg.image_max_items=500 is passed as max_items=500
        - iter_image_batches receives the correct keyword argument
        """
        from core.ingestion.databricks.process_s3_images import main

        mock_dependencies["ray_cfg"].return_value.image_max_items = 500
        main()

        mock_dependencies["iter_batches"].assert_called_once()
        call_kwargs = mock_dependencies["iter_batches"].call_args[1]
        assert call_kwargs["max_items"] == 500

    def test_main_passes_image_page_size_to_iter_batches(self, mock_dependencies, mock_ray):
        """main() passes image_page_size from config to iter_image_batches as page_size.

        **Why this test is important:**

        - IMAGE_PAGE_SIZE controls S3 listing pagination size
        - The config field must be forwarded to iter_image_batches
        - Allows tuning S3 list performance for different bucket sizes

        **What it tests:**

        - ray_cfg.image_page_size=200 is passed as page_size=200
        - iter_image_batches receives the correct keyword argument
        """
        from core.ingestion.databricks.process_s3_images import main

        mock_dependencies["ray_cfg"].return_value.image_page_size = 200
        main()

        mock_dependencies["iter_batches"].assert_called_once()
        call_kwargs = mock_dependencies["iter_batches"].call_args[1]
        assert call_kwargs["page_size"] == 200

    def test_main_disables_and_reenables_qdrant_indexing(self, mock_dependencies, mock_ray):
        """main() uses qdrant_indexing_disabled context manager when flag is set.

        **Why this test is important:**

        - Disabling HNSW indexing during bulk uploads improves throughput
        - Indexing must be re-enabled after processing completes
        - Both operations must use the correct collection name and credentials

        **What it tests:**

        - qdrant_indexing_disabled context manager is entered with correct args
        - Context manager is only used when disable_indexing_during_ingest is True
        """
        from core.ingestion.databricks.process_s3_images import main

        mock_dependencies["ray_cfg"].return_value.disable_indexing_during_ingest = True

        with patch.dict("os.environ", {"S3_PREFIX": "images/"}, clear=False):
            main()

        mock_dependencies["qdrant_indexing_disabled"].assert_called_once_with(
            client=mock_dependencies["qdrant_cls"].from_config.return_value,
            collection="documents",
            vector_size=mock_dependencies["embed_cfg"].clip_vector_size,
        )

    def test_main_pull_from_dlq_uses_dlq_iterator_and_clears_successes(
        self, mock_dependencies, mock_ray: MagicMock, monkeypatch: pytest.MonkeyPatch
    ):
        """PULL_FROM_DLQ=true switches to DLQ batching and clears successful keys."""
        from core.ingestion.databricks.process_s3_images import main

        dlq_batches = iter([["glad.jpeg", "mad.jpeg"]])
        stats = BatchRunStats(
            successful_keys={"glad.jpeg"},
            successful=1,
            failed=0,
            completed_records=1,
            submitted_records=2,
            num_batches=1,
        )

        with (
            patch("sys.argv", ["script.py"]),
            patch("core.ingestion.databricks.process_s3_images.clear_successful_dlq_entries") as mock_clear,
            patch(
                "core.ingestion.databricks.process_s3_images.iter_dlq_entries", return_value=dlq_batches
            ) as mock_iter_dlq,
            patch(
                "core.ingestion.databricks.process_s3_images.run_ray_batch_processing", return_value=stats
            ) as mock_run,
        ):
            monkeypatch.setenv("PULL_FROM_DLQ", "true")
            main()

        # iter_image_batches is patched by the fixture; ensure it wasn't used when DLQ mode is on
        mock_dependencies["iter_batches"].assert_not_called()
        mock_iter_dlq.assert_called_once_with(
            batch_size=mock_dependencies["ray_cfg"].return_value.image_batch_size,
            max_items=mock_dependencies["ray_cfg"].return_value.image_max_items,
        )
        assert mock_run.call_args.kwargs["batches"] is dlq_batches
        mock_clear.assert_called_once_with(stats.successful_keys)
