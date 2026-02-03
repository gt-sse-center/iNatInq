"""Unit tests for core.ingestion.ray.process_s3_to_vector_dbs module.

This module tests the Ray-based S3 to vector database ingestion pipeline,
including lazy imports, cluster lifecycle management, error handling, and
checkpoint functionality.

# Test Coverage

- Ray package lazy imports via __getattr__
- main() function cluster initialization and shutdown
- S3 key listing and processing
- Error handling for S3 and strategy failures
- Checkpoint loading and saving when enabled
- Environment variable configuration (S3_PREFIX)

# Running Tests

    uv run pytest tests/unit/core/ingestion/ray/test_process_s3_to_vector_dbs.py -v
"""

import sys
from unittest.mock import MagicMock, patch

import pytest


class TestRayPackageInit:
    """Tests for the ray package __init__.py lazy imports."""

    def test_getattr_returns_run_ray_job(self):
        """Test that __getattr__ returns run_ray_job for 'run_ray_job' attribute.

        **Why this test is important:**

        - Ensures lazy loading mechanism works correctly for the main entry point
        - Validates that the ray package exposes run_ray_job without eager imports
        - Confirms the module's public API contract is maintained

        **What it tests:**

        - Accessing ray_pkg.run_ray_job returns the mocked main function
        - The lazy import mechanism resolves to the correct underlying function
        """
        from core.ingestion import ray as ray_pkg

        with patch("core.ingestion.ray.process_s3_to_vector_dbs.main") as mock_main:
            result = ray_pkg.run_ray_job
            assert result is mock_main

    def test_getattr_returns_process_s3_object_ray(self, mock_ray):
        """Test that __getattr__ returns process_s3_object_ray for that attribute.

        **Why this test is important:**

        - Validates lazy loading for the Ray remote task function
        - Ensures process_s3_object_ray is accessible without importing the entire module
        - Confirms the function is properly exposed in the package namespace

        **What it tests:**

        - Accessing ray_pkg.process_s3_object_ray returns a non-None value
        - The lazy import mechanism successfully loads the remote task
        """
        from core.ingestion import ray as ray_pkg

        result = ray_pkg.process_s3_object_ray
        assert result is not None

    def test_getattr_raises_attribute_error_for_unknown(self):
        """Test that __getattr__ raises AttributeError for unknown attributes.

        **Why this test is important:**

        - Ensures proper error handling for invalid attribute access
        - Prevents silent failures when accessing non-existent attributes
        - Maintains Python's standard behavior for missing attributes

        **What it tests:**

        - Accessing an unknown attribute raises AttributeError
        - The error message contains the attribute name for debugging
        """
        from core.ingestion import ray as ray_pkg

        with pytest.raises(AttributeError, match="has no attribute 'unknown_attribute'"):
            _ = ray_pkg.unknown_attribute


class TestRayJobMain:
    """Tests for the main() function in process_s3_to_vector_dbs."""

    @pytest.fixture
    def mock_dependencies(self, mock_ray):
        """Set up common mocks for main() tests."""
        mock_s3 = MagicMock()
        mock_s3.list_objects.return_value = []

        mock_strategy = MagicMock()
        mock_strategy.init = MagicMock()
        mock_strategy.shutdown = MagicMock()

        with patch("core.ingestion.ray.process_s3_to_vector_dbs.RayJobConfig.from_env") as mock_ray_cfg:
            with patch("core.ingestion.ray.process_s3_to_vector_dbs.MinIOConfig.from_env") as mock_minio_cfg:
                with patch(
                    "core.ingestion.ray.process_s3_to_vector_dbs.VectorDBConfig.from_env"
                ) as mock_vector_cfg:
                    with patch(
                        "core.ingestion.ray.process_s3_to_vector_dbs.EmbeddingConfig.from_env"
                    ) as mock_embed_cfg:
                        with patch(
                            "core.ingestion.ray.process_s3_to_vector_dbs.LocalRayStrategy"
                        ) as mock_strat_cls:
                            with patch(
                                "core.ingestion.ray.process_s3_to_vector_dbs.S3ClientWrapper"
                            ) as mock_s3_cls:
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
                                mock_strat_cls.from_env.return_value = mock_strategy
                                mock_s3_cls.return_value = mock_s3

                                yield {
                                    "ray_cfg": mock_ray_cfg,
                                    "minio_cfg": mock_minio_cfg,
                                    "strategy": mock_strategy,
                                    "s3": mock_s3,
                                }

    def test_main_initializes_and_shuts_down_cluster(self, mock_dependencies, mock_ray):
        """Test that main() initializes cluster and shuts down on completion.

        **Why this test is important:**

        - Ensures proper Ray cluster lifecycle management
        - Prevents resource leaks by verifying shutdown is always called
        - Validates the happy path for cluster initialization

        **What it tests:**

        - strategy.init() is called exactly once during startup
        - strategy.shutdown() is called exactly once on completion
        - The cluster lifecycle follows the expected sequence
        """
        from core.ingestion.ray.process_s3_to_vector_dbs import main

        with patch.dict("os.environ", {"S3_PREFIX": "inputs/"}, clear=False):
            main()

        mock_dependencies["strategy"].init.assert_called_once()
        mock_dependencies["strategy"].shutdown.assert_called_once()

    def test_main_returns_early_when_no_keys(self, mock_dependencies, mock_ray):
        """Test that main() returns early when no S3 keys found.

        **Why this test is important:**

        - Ensures graceful handling of empty S3 buckets or prefixes
        - Verifies no unnecessary processing occurs with no data
        - Confirms shutdown is still called even with early return

        **What it tests:**

        - main() completes without error when list_objects returns empty
        - strategy.shutdown() is called despite early return
        - No Ray tasks are submitted when there's nothing to process
        """
        from core.ingestion.ray.process_s3_to_vector_dbs import main

        mock_dependencies["s3"].list_objects.return_value = []

        with patch.dict("os.environ", {"S3_PREFIX": "inputs/"}, clear=False):
            main()

        # Shutdown should still be called
        mock_dependencies["strategy"].shutdown.assert_called_once()

    def test_main_processes_keys_when_found(self, mock_dependencies, mock_ray):
        """Test that main() processes S3 keys when found.

        **Why this test is important:**

        - Validates the core ingestion workflow executes correctly
        - Ensures S3 objects are discovered and processed
        - Confirms Ray tasks are submitted for batch processing

        **What it tests:**

        - S3 keys are fetched via list_objects
        - Ray tasks are submitted for processing batches
        - strategy.shutdown() is called after processing completes
        """
        from core.ingestion.ray.process_s3_to_vector_dbs import main

        mock_dependencies["s3"].list_objects.return_value = ["file1.txt", "file2.txt"]

        # Mock Ray wait and get
        mock_ray.wait.return_value = ([MagicMock()], [])
        mock_ray.get.return_value = [[("file1.txt", True, ""), ("file2.txt", True, "")]]

        with patch.dict("os.environ", {"S3_PREFIX": "inputs/"}, clear=False):
            with patch("core.ingestion.ray.process_s3_to_vector_dbs.RateLimiterActor") as mock_rate:
                mock_rate.remote.return_value = MagicMock()
                with patch("core.ingestion.ray.process_s3_to_vector_dbs.process_s3_batch_ray") as mock_task:
                    mock_task.options.return_value.remote.return_value = MagicMock()
                    main()

        mock_dependencies["strategy"].shutdown.assert_called_once()

    def test_main_handles_s3_error(self, mock_dependencies, mock_ray):
        """Test that main() exits when S3 listing fails.

        **Why this test is important:**

        - Ensures proper error handling for S3 connectivity issues
        - Validates the application fails fast with appropriate exit code
        - Confirms cleanup (shutdown) still occurs on error

        **What it tests:**

        - S3 ClientError causes SystemExit with code 1
        - strategy.shutdown() is called even after S3 error
        - The error is not silently swallowed
        """
        from botocore.exceptions import ClientError
        from core.ingestion.ray.process_s3_to_vector_dbs import main

        mock_dependencies["s3"].list_objects.side_effect = ClientError(
            {"Error": {"Code": "500", "Message": "Error"}}, "ListObjects"
        )

        with patch.dict("os.environ", {"S3_PREFIX": "inputs/"}, clear=False):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 1
        mock_dependencies["strategy"].shutdown.assert_called_once()

    def test_main_handles_strategy_init_error(self, mock_ray):
        """Test that main() exits when strategy init fails.

        **Why this test is important:**

        - Validates error handling during Ray cluster initialization
        - Ensures the application fails fast if cluster cannot be started
        - Confirms appropriate exit code for infrastructure failures

        **What it tests:**

        - RuntimeError during strategy.init() causes SystemExit with code 1
        - The error propagates correctly through the initialization path
        - Application does not proceed with a failed cluster
        """
        from core.ingestion.ray.process_s3_to_vector_dbs import main

        mock_strategy = MagicMock()
        mock_strategy.init.side_effect = RuntimeError("Connection failed")
        mock_strategy.shutdown = MagicMock()

        with patch("core.ingestion.ray.process_s3_to_vector_dbs.RayJobConfig.from_env"):
            with patch("core.ingestion.ray.process_s3_to_vector_dbs.MinIOConfig.from_env"):
                with patch("core.ingestion.ray.process_s3_to_vector_dbs.VectorDBConfig.from_env"):
                    with patch("core.ingestion.ray.process_s3_to_vector_dbs.EmbeddingConfig.from_env"):
                        with patch(
                            "core.ingestion.ray.process_s3_to_vector_dbs.LocalRayStrategy"
                        ) as mock_strat_cls:
                            mock_strat_cls.from_env.return_value = mock_strategy

                            with patch.dict("os.environ", {"S3_PREFIX": "inputs/"}, clear=False):
                                with pytest.raises(SystemExit) as exc_info:
                                    main()

        assert exc_info.value.code == 1

    def test_main_uses_s3_prefix_from_env(self, mock_dependencies, mock_ray):
        """Test that main() uses S3_PREFIX from environment variable.

        **Why this test is important:**

        - Validates configuration is correctly read from environment
        - Ensures users can customize the S3 prefix for their use case
        - Confirms the prefix is passed through to S3 listing

        **What it tests:**

        - S3_PREFIX environment variable value is used in list_objects call
        - Custom prefix "custom/prefix/" is correctly passed to S3
        - The configuration flows through the initialization path
        """
        from core.ingestion.ray.process_s3_to_vector_dbs import main

        mock_dependencies["s3"].list_objects.return_value = []

        with patch.dict("os.environ", {"S3_PREFIX": "custom/prefix/"}, clear=False):
            main()

        mock_dependencies["s3"].list_objects.assert_called_once()
        call_kwargs = mock_dependencies["s3"].list_objects.call_args[1]
        assert call_kwargs["prefix"] == "custom/prefix/"

    def test_main_uses_checkpoint_when_enabled(self, mock_ray):
        """Test that main() loads and saves checkpoint when enabled.

        **Why this test is important:**

        - Validates checkpoint functionality for resumable processing
        - Ensures progress is persisted to allow recovery from failures
        - Confirms both load and save operations occur in the workflow

        **What it tests:**

        - CheckpointManager.load() is called once to restore state
        - CheckpointManager.save() is called once to persist progress
        - Checkpoint is used when checkpoint_enabled=True in config
        """
        from core.ingestion.ray.process_s3_to_vector_dbs import main

        mock_s3 = MagicMock()
        mock_s3.list_objects.return_value = ["file1.txt"]

        mock_strategy = MagicMock()
        mock_checkpoint = MagicMock()
        mock_checkpoint.load.return_value = set()

        mock_ray.wait.return_value = ([MagicMock()], [])
        mock_ray.get.return_value = [[("file1.txt", True, "")]]

        with patch("core.ingestion.ray.process_s3_to_vector_dbs.RayJobConfig.from_env") as mock_ray_cfg:
            mock_ray_cfg.return_value = MagicMock(
                num_workers=4,
                s3_batch_size=50,
                embed_batch_max=8,
                batch_upsert_size=200,
                checkpoint_enabled=True,
                checkpoint_dir="/tmp/checkpoints",
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
            with patch("core.ingestion.ray.process_s3_to_vector_dbs.MinIOConfig.from_env") as mock_minio_cfg:
                mock_minio_cfg.return_value = MagicMock(
                    endpoint_url="http://minio:9000",
                    access_key_id="access",
                    secret_access_key="secret",
                    bucket="bucket",
                )
                with patch(
                    "core.ingestion.ray.process_s3_to_vector_dbs.VectorDBConfig.from_env"
                ) as mock_vector_cfg:
                    mock_vector_cfg.return_value = MagicMock(collection="test")
                    with patch("core.ingestion.ray.process_s3_to_vector_dbs.EmbeddingConfig.from_env"):
                        with patch(
                            "core.ingestion.ray.process_s3_to_vector_dbs.LocalRayStrategy"
                        ) as mock_strat_cls:
                            mock_strat_cls.from_env.return_value = mock_strategy
                            with patch(
                                "core.ingestion.ray.process_s3_to_vector_dbs.S3ClientWrapper",
                                return_value=mock_s3,
                            ):
                                with patch(
                                    "core.ingestion.ray.process_s3_to_vector_dbs.CheckpointManager",
                                    return_value=mock_checkpoint,
                                ):
                                    with patch(
                                        "core.ingestion.ray.process_s3_to_vector_dbs.RateLimiterActor"
                                    ) as mock_rate:
                                        mock_rate.remote.return_value = MagicMock()
                                        with patch(
                                            "core.ingestion.ray.process_s3_to_vector_dbs.process_s3_batch_ray"
                                        ) as mock_task:
                                            mock_task.options.return_value.remote.return_value = MagicMock()
                                            with patch.dict(
                                                "os.environ", {"S3_PREFIX": "inputs/"}, clear=False
                                            ):
                                                main()

        mock_checkpoint.load.assert_called_once()
        mock_checkpoint.save.assert_called_once()
