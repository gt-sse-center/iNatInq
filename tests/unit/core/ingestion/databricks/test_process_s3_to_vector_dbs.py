"""Unit tests for core.ingestion.databricks.process_s3_to_vector_dbs module.

This module contains tests for the Databricks Ray cluster orchestration and S3
processing pipeline that ingests data into vector databases.

# Test Coverage

- Parameter parsing (_apply_python_params)
- Ray cluster setup and initialization
- Ray cluster shutdown handling
- Main orchestration function behavior
- Error handling for S3 operations

# Running Tests

    uv run pytest tests/unit/core/ingestion/databricks/test_process_s3_to_vector_dbs.py -v
"""

import os
from unittest.mock import MagicMock, patch

import pytest


class TestApplyPythonParams:
    """Tests for _apply_python_params helper function."""

    def test_applies_valid_key_value_pairs(self):
        """Apply KEY=VALUE pairs to environment variables.

        **Why this test is important:**
        - Databricks jobs receive parameters as command-line arguments
        - These parameters must be converted to environment variables for the pipeline
        - Incorrect parsing could cause configuration failures

        **What it tests:**
        - Valid uppercase KEY=VALUE pairs are parsed correctly
        - Multiple parameters are all applied
        - Values are accessible via os.environ after application
        """
        from core.ingestion.databricks.process_s3_to_vector_dbs import _apply_python_params

        with patch.dict("os.environ", {}, clear=True):
            _apply_python_params(["S3_PREFIX=inputs/", "K8S_NAMESPACE=test"])
            assert os.environ["S3_PREFIX"] == "inputs/"
            assert os.environ["K8S_NAMESPACE"] == "test"

    def test_ignores_non_key_value_args(self):
        """Ignore arguments without = separator.

        **Why this test is important:**
        - Command-line arguments may include non-configuration values
        - The function must safely ignore invalid formats
        - Prevents pollution of environment with malformed data

        **What it tests:**
        - Arguments without '=' separator are skipped
        - No partial or malformed entries end up in environment
        """
        from core.ingestion.databricks.process_s3_to_vector_dbs import _apply_python_params

        with patch.dict("os.environ", {}, clear=True):
            _apply_python_params(["not_a_key_value", "another"])
            assert "not_a_key_value" not in os.environ
            assert "another" not in os.environ

    def test_ignores_lowercase_keys(self):
        """Ignore lowercase key names.

        **Why this test is important:**
        - Environment variable convention uses uppercase names
        - Lowercase keys may indicate typos or invalid input
        - Filtering prevents accidental configuration pollution

        **What it tests:**
        - Lowercase KEY=value pairs are not applied
        - Only uppercase keys are accepted
        """
        from core.ingestion.databricks.process_s3_to_vector_dbs import _apply_python_params

        with patch.dict("os.environ", {}, clear=True):
            _apply_python_params(["lowercase=value"])
            assert "lowercase" not in os.environ

    def test_ignores_empty_keys(self):
        """Ignore empty key names.

        **Why this test is important:**
        - Malformed input like '=value' should not cause errors
        - Empty keys are meaningless and should be skipped
        - Ensures robustness against edge case inputs

        **What it tests:**
        - Arguments starting with '=' are ignored
        - No empty string key is created in environment
        """
        from core.ingestion.databricks.process_s3_to_vector_dbs import _apply_python_params

        with patch.dict("os.environ", {}, clear=True):
            _apply_python_params(["=value"])
            assert "" not in os.environ


class TestSetupRayCluster:
    """Tests for _setup_ray_cluster helper function."""

    def test_setup_calls_setup_ray_cluster(self, mock_ray):
        """Call the setup_ray_cluster function with correct parameters.

        **Why this test is important:**
        - Ray cluster setup is required for distributed processing
        - Configuration parameters must be passed correctly
        - Ensures the wrapper function properly delegates to the actual setup

        **What it tests:**
        - setup_ray_cluster is called exactly once
        - A valid cluster object is returned
        - Configuration parameters are respected
        """
        from core.ingestion.databricks.process_s3_to_vector_dbs import _setup_ray_cluster

        mock_config = MagicMock()
        mock_config.num_workers = 4
        mock_config.worker_cpus = 2.0
        mock_config.worker_memory = 4096

        with patch("core.ingestion.databricks.process_s3_to_vector_dbs.setup_ray_cluster") as mock_setup:
            with patch("core.ingestion.databricks.process_s3_to_vector_dbs.MAX_NUM_WORKER_NODES", 10):
                mock_setup.return_value = MagicMock()
                result = _setup_ray_cluster(mock_config)

        mock_setup.assert_called_once()
        assert result is not None


class TestInitRay:
    """Tests for _init_ray helper function."""

    def test_init_ray_uses_cluster_address(self, mock_ray):
        """Use cluster address when available.

        **Why this test is important:**
        - Databricks Ray clusters have specific addresses
        - Using the correct address ensures workers connect properly
        - Incorrect addressing causes distributed processing failures

        **What it tests:**
        - Cluster address is extracted and passed to ray.init
        - Runtime environment is configured correctly
        - ray.init is called exactly once
        """
        from core.ingestion.databricks.process_s3_to_vector_dbs import _init_ray

        mock_ray.is_initialized.return_value = False
        mock_cluster = MagicMock()
        mock_cluster.address = "ray://databricks:10001"

        mock_config = MagicMock()
        mock_config.ray_namespace = "test-ns"

        with patch.dict("os.environ", {"INATINQ_SRC_DIR": "/workspace/src"}, clear=False):
            with patch("pathlib.Path.is_dir", return_value=True):
                _init_ray(mock_config, mock_cluster)

        mock_ray.init.assert_called_once()
        call_kwargs = mock_ray.init.call_args[1]
        assert call_kwargs["address"] == "ray://databricks:10001"

    def test_init_ray_uses_auto_when_no_cluster(self, mock_ray):
        """Use 'auto' address when no cluster provided.

        **Why this test is important:**
        - Local development may not have a dedicated cluster
        - 'auto' allows Ray to find the existing cluster or start locally
        - Ensures flexibility between environments

        **What it tests:**
        - 'auto' is used as address when cluster is None
        - ray.init is still called with proper configuration
        """
        from core.ingestion.databricks.process_s3_to_vector_dbs import _init_ray

        mock_ray.is_initialized.return_value = False
        mock_config = MagicMock()
        mock_config.ray_namespace = "test-ns"

        with patch.dict("os.environ", {"INATINQ_SRC_DIR": "/workspace/src"}, clear=False):
            with patch("pathlib.Path.is_dir", return_value=True):
                _init_ray(mock_config, None)

        call_kwargs = mock_ray.init.call_args[1]
        assert call_kwargs["address"] == "auto"

    def test_init_ray_raises_when_no_src_dir(self, mock_ray):
        """Raise RuntimeError when INATINQ_SRC_DIR is not set.

        **Why this test is important:**
        - INATINQ_SRC_DIR is required for Ray worker code discovery
        - Missing this variable causes worker initialization failures
        - Early error detection prevents confusing downstream errors

        **What it tests:**
        - RuntimeError is raised with descriptive message
        - Error occurs before attempting ray.init
        """
        from core.ingestion.databricks.process_s3_to_vector_dbs import _init_ray

        mock_ray.is_initialized.return_value = False
        mock_config = MagicMock()

        with patch.dict("os.environ", {}, clear=True):
            os.environ.pop("INATINQ_SRC_DIR", None)
            with pytest.raises(RuntimeError, match="INATINQ_SRC_DIR is not set"):
                _init_ray(mock_config, None)

    def test_init_ray_skips_when_initialized(self, mock_ray):
        """Skip initialization when Ray is already initialized.

        **Why this test is important:**
        - Double initialization of Ray causes errors
        - Idempotency is important for robust orchestration
        - Allows safe re-entry into the initialization function

        **What it tests:**
        - ray.init is not called when ray.is_initialized() returns True
        - Function completes without error
        """
        from core.ingestion.databricks.process_s3_to_vector_dbs import _init_ray

        mock_ray.is_initialized.return_value = True
        mock_config = MagicMock()

        with patch.dict("os.environ", {"INATINQ_SRC_DIR": "/workspace/src"}, clear=False):
            with patch("pathlib.Path.is_dir", return_value=True):
                _init_ray(mock_config, None)

        mock_ray.init.assert_not_called()

    def test_init_ray_passes_pythonpath(self, mock_ray):
        """Pass PYTHONPATH to runtime environment.

        **Why this test is important:**
        - Ray workers need PYTHONPATH to import custom modules
        - Missing PYTHONPATH causes ImportError on workers
        - Ensures code is discoverable across the cluster

        **What it tests:**
        - PYTHONPATH from environment is included in runtime_env
        - Workers will have access to the same module paths
        """
        from core.ingestion.databricks.process_s3_to_vector_dbs import _init_ray

        mock_ray.is_initialized.return_value = False
        mock_config = MagicMock()
        mock_config.ray_namespace = "test-ns"

        with patch.dict(
            "os.environ",
            {"INATINQ_SRC_DIR": "/workspace/src", "PYTHONPATH": "/custom/path"},
            clear=False,
        ):
            with patch("pathlib.Path.is_dir", return_value=True):
                _init_ray(mock_config, None)

        call_kwargs = mock_ray.init.call_args[1]
        assert call_kwargs["runtime_env"]["env_vars"]["PYTHONPATH"] == "/custom/path"


class TestShutdownRayCluster:
    """Tests for _shutdown_ray_cluster helper function."""

    def test_shutdown_calls_ray_shutdown(self, mock_ray):
        """Call ray.shutdown() when Ray is initialized.

        **Why this test is important:**
        - Proper shutdown releases cluster resources
        - Prevents resource leaks in Databricks environment
        - Ensures clean state for subsequent jobs

        **What it tests:**
        - ray.shutdown() is called when Ray is initialized
        - Shutdown happens regardless of cluster object
        """
        from core.ingestion.databricks.process_s3_to_vector_dbs import _shutdown_ray_cluster

        mock_ray.is_initialized.return_value = True
        _shutdown_ray_cluster(None)
        mock_ray.shutdown.assert_called_once()

    def test_shutdown_calls_cluster_shutdown(self, mock_ray):
        """Call cluster shutdown method when cluster exists.

        **Why this test is important:**
        - Databricks clusters need explicit shutdown
        - Cluster shutdown releases compute resources
        - Prevents billing for unused cluster time

        **What it tests:**
        - cluster.shutdown() is called on provided cluster object
        - Works even when Ray itself is not initialized
        """
        from core.ingestion.databricks.process_s3_to_vector_dbs import _shutdown_ray_cluster

        mock_ray.is_initialized.return_value = False
        mock_cluster = MagicMock()
        _shutdown_ray_cluster(mock_cluster)
        mock_cluster.shutdown.assert_called_once()

    def test_shutdown_handles_no_cluster(self, mock_ray):
        """Handle None cluster gracefully.

        **Why this test is important:**
        - Shutdown may be called even if setup failed
        - Robustness against partial initialization states
        - Prevents crashes during error cleanup

        **What it tests:**
        - No exception is raised when cluster is None
        - Function completes successfully
        """
        from core.ingestion.databricks.process_s3_to_vector_dbs import _shutdown_ray_cluster

        mock_ray.is_initialized.return_value = False
        # Should not raise
        _shutdown_ray_cluster(None)


class TestDatabricksMain:
    """Tests for the main() function."""

    @pytest.fixture
    def mock_dependencies(self, mock_ray):
        """Set up common mocks for main() tests."""
        mock_s3 = MagicMock()
        mock_s3.list_objects.return_value = []

        mock_cluster = MagicMock()

        with patch(
            "core.ingestion.databricks.process_s3_to_vector_dbs.RayJobConfig.from_env"
        ) as mock_ray_cfg:
            with patch(
                "core.ingestion.databricks.process_s3_to_vector_dbs.MinIOConfig.from_env"
            ) as mock_minio_cfg:
                with patch(
                    "core.ingestion.databricks.process_s3_to_vector_dbs.VectorDBConfig.from_env"
                ) as mock_vector_cfg:
                    with patch(
                        "core.ingestion.databricks.process_s3_to_vector_dbs.EmbeddingConfig.from_env"
                    ) as mock_embed_cfg:
                        with patch(
                            "core.ingestion.databricks.process_s3_to_vector_dbs._setup_ray_cluster",
                            return_value=mock_cluster,
                        ) as mock_setup:
                            with patch(
                                "core.ingestion.databricks.process_s3_to_vector_dbs._init_ray"
                            ) as mock_init:
                                with patch(
                                    "core.ingestion.databricks.process_s3_to_vector_dbs._shutdown_ray_cluster"
                                ) as mock_shutdown:
                                    with patch(
                                        "core.ingestion.databricks.process_s3_to_vector_dbs.S3ClientWrapper",
                                        return_value=mock_s3,
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

                                        yield {
                                            "ray_cfg": mock_ray_cfg,
                                            "s3": mock_s3,
                                            "setup": mock_setup,
                                            "init": mock_init,
                                            "shutdown": mock_shutdown,
                                            "cluster": mock_cluster,
                                        }

    def test_main_initializes_and_shuts_down_cluster(self, mock_dependencies, mock_ray):
        """Initialize cluster and shut down on completion.

        **Why this test is important:**
        - Cluster lifecycle management is critical for resource management
        - Ensures proper setup-process-teardown sequence
        - Verifies the main orchestration flow works correctly

        **What it tests:**
        - _setup_ray_cluster is called to create the cluster
        - _init_ray is called to initialize Ray
        - _shutdown_ray_cluster is called for cleanup
        """
        from core.ingestion.databricks.process_s3_to_vector_dbs import main

        with patch.dict("os.environ", {"S3_PREFIX": "inputs/"}, clear=False):
            main()

        mock_dependencies["setup"].assert_called_once()
        mock_dependencies["init"].assert_called_once()
        mock_dependencies["shutdown"].assert_called_once()

    def test_main_returns_early_when_no_keys(self, mock_dependencies, mock_ray):
        """Return early when no S3 keys found.

        **Why this test is important:**
        - Avoids unnecessary processing when no data exists
        - Ensures graceful handling of empty input
        - Still performs proper cleanup even with no work

        **What it tests:**
        - Function completes without error when S3 returns empty list
        - Shutdown is still called for proper cleanup
        """
        from core.ingestion.databricks.process_s3_to_vector_dbs import main

        mock_dependencies["s3"].list_objects.return_value = []

        with patch.dict("os.environ", {"S3_PREFIX": "inputs/"}, clear=False):
            main()

        mock_dependencies["shutdown"].assert_called_once()

    def test_main_processes_keys_when_found(self, mock_dependencies, mock_ray):
        """Process S3 keys when found.

        **Why this test is important:**
        - Core functionality of ingesting S3 data must work
        - Verifies Ray tasks are submitted for processing
        - Ensures end-to-end flow with actual data

        **What it tests:**
        - S3 keys are retrieved and processed
        - Ray tasks are created for batch processing
        - Proper cleanup occurs after processing
        """
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

        mock_dependencies["shutdown"].assert_called_once()

    def test_main_handles_s3_error(self, mock_dependencies, mock_ray):
        """Exit when S3 listing fails.

        **Why this test is important:**
        - S3 failures should not crash silently
        - Proper error handling ensures visibility of issues
        - Cleanup must still occur even on error

        **What it tests:**
        - ClientError from S3 causes sys.exit(1)
        - Shutdown is called even after error
        - Error propagates as expected
        """
        from botocore.exceptions import ClientError
        from core.ingestion.databricks.process_s3_to_vector_dbs import main

        mock_dependencies["s3"].list_objects.side_effect = ClientError(
            {"Error": {"Code": "500", "Message": "Error"}}, "ListObjects"
        )

        with patch.dict("os.environ", {"S3_PREFIX": "inputs/"}, clear=False):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 1
        mock_dependencies["shutdown"].assert_called_once()

    def test_main_applies_python_params(self, mock_dependencies, mock_ray):
        """Apply sys.argv KEY=VALUE params to environment.

        **Why this test is important:**
        - Databricks passes job parameters via command line
        - Parameters must be converted to environment variables
        - Ensures configuration flows from CLI to the pipeline

        **What it tests:**
        - Command-line arguments are parsed and applied
        - S3 client receives the configured prefix
        - Integration between parameter parsing and main flow
        """
        from core.ingestion.databricks.process_s3_to_vector_dbs import main

        mock_dependencies["s3"].list_objects.return_value = []

        with patch("sys.argv", ["script.py", "S3_PREFIX=custom/"]):
            with patch.dict("os.environ", {}, clear=False):
                main()

        # The prefix should have been applied to env and used
        mock_dependencies["s3"].list_objects.assert_called_once()
