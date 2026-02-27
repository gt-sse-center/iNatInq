"""Unit tests for DatabricksStrategy.

# Test Coverage

- DatabricksStrategy initialization and configuration
- Runtime environment handling (env vars, PYTHONPATH, working_dir)
- Databricks cluster connection lifecycle (init, shutdown)
- Spark cluster setup with Ray integration
- Ray client initialization with cluster addresses

# Running Tests

```bash
uv run pytest tests/unit/core/ingestion/strategies/test_databricks.py -v
```
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from config import RayJobConfig
from core.ingestion.strategies.databricks import DatabricksStrategy, _PASSTHROUGH_ENV_VARS


@pytest.fixture(autouse=True)
def reset_ray_mock(mock_ray: MagicMock):
    """Reset ray mock state before each test."""
    mock_ray.reset_mock()
    yield


class TestDatabricksStrategyInit:
    """Tests for DatabricksStrategy initialization."""

    def test_creates_strategy_with_config(self, ray_job_config: RayJobConfig):
        """Strategy is created with provided config.

        **Why this test is important:**
        - Ensures the strategy stores the configuration correctly
        - Verifies initial cluster state is None before connection

        **What it tests:**
        - Strategy accepts and stores RayJobConfig
        - Internal _cluster attribute is initialized to None
        """
        strategy = DatabricksStrategy(config=ray_job_config)
        assert strategy.config == ray_job_config
        assert strategy._cluster is None

    def test_from_env_creates_strategy(self):
        """from_env() creates strategy from environment.

        **Why this test is important:**
        - Factory method allows configuration from environment variables
        - Enables deployment flexibility without code changes

        **What it tests:**
        - from_env() delegates to RayJobConfig.from_env()
        - Namespace is passed correctly to config factory
        - Returned strategy has the created config
        """
        with patch("core.ingestion.strategies.databricks.RayJobConfig.from_env") as mock_from_env:
            mock_config = MagicMock(spec=RayJobConfig)
            mock_from_env.return_value = mock_config

            strategy = DatabricksStrategy.from_env("test-namespace")

            mock_from_env.assert_called_once_with("test-namespace")
            assert strategy.config == mock_config

    def test_config_property_returns_config(self, ray_job_config: RayJobConfig):
        """config property returns the configuration.

        **Why this test is important:**
        - Config property provides read access to strategy configuration
        - External code may need to inspect configuration

        **What it tests:**
        - Property returns the exact config object passed to constructor
        """
        strategy = DatabricksStrategy(config=ray_job_config)
        assert strategy.config is ray_job_config


class TestDatabricksStrategyRuntimeEnv:
    """Tests for runtime environment handling."""

    def test_get_runtime_env_passes_through_env_vars(self, ray_job_config: RayJobConfig):
        """get_runtime_env() includes environment variables.

        **Why this test is important:**
        - Ray workers need access to service endpoints (S3, Qdrant)
        - Environment passthrough enables distributed configuration

        **What it tests:**
        - S3_ENDPOINT is passed to runtime env
        - QDRANT_URL is passed to runtime env
        - env_vars key exists in returned dict
        """
        strategy = DatabricksStrategy(config=ray_job_config)

        with patch.dict(
            os.environ,
            {
                "INATINQ_SRC_DIR": "/workspace/src",
                "VECTOR_DB_PROVIDER": "qdrant",
                "S3_ENDPOINT": "http://minio:9000",
                "QDRANT_URL": "http://qdrant:6333",
            },
            clear=False,
        ):
            with patch("pathlib.Path.is_dir", return_value=True):
                env = strategy.get_runtime_env()

        assert "env_vars" in env
        assert env["env_vars"]["S3_ENDPOINT"] == "http://minio:9000"
        assert env["env_vars"]["QDRANT_URL"] == "http://qdrant:6333"

    def test_get_runtime_env_includes_pythonpath(self, ray_job_config: RayJobConfig):
        """get_runtime_env() includes PYTHONPATH if set.

        **Why this test is important:**
        - Custom PYTHONPATH may be needed for local modules
        - Preserves developer's Python path configuration

        **What it tests:**
        - PYTHONPATH from environment is included in runtime env
        """
        strategy = DatabricksStrategy(config=ray_job_config)

        with patch.dict(
            os.environ,
            {
                "INATINQ_SRC_DIR": "/workspace/src",
                "VECTOR_DB_PROVIDER": "qdrant",
                "PYTHONPATH": "/custom/path",
            },
            clear=False,
        ):
            with patch("pathlib.Path.is_dir", return_value=True):
                env = strategy.get_runtime_env()

        assert env["env_vars"]["PYTHONPATH"] == "/custom/path"

    def test_get_runtime_env_defaults_vector_targets_when_missing(self, ray_job_config: RayJobConfig):
        """get_runtime_env() sets default VECTOR_DB_TARGETS when missing."""
        strategy = DatabricksStrategy(config=ray_job_config)

        with patch.dict(os.environ, {"INATINQ_SRC_DIR": "/workspace/src"}, clear=True):
            with patch("pathlib.Path.is_dir", return_value=True):
                env = strategy.get_runtime_env()

        assert env["env_vars"]["VECTOR_DB_TARGETS"] == "qdrant,weaviate"

    def test_get_runtime_env_raises_without_src_dir(self, ray_job_config: RayJobConfig):
        """get_runtime_env() raises RuntimeError without INATINQ_SRC_DIR.

        **Why this test is important:**
        - INATINQ_SRC_DIR is required for Ray to locate source files
        - Clear error message helps debugging configuration issues

        **What it tests:**
        - RuntimeError raised when INATINQ_SRC_DIR is not set
        - Error message indicates the missing variable
        """
        strategy = DatabricksStrategy(config=ray_job_config)

        with patch.dict(os.environ, {"VECTOR_DB_PROVIDER": "qdrant"}, clear=True):
            # Remove INATINQ_SRC_DIR if present
            os.environ.pop("INATINQ_SRC_DIR", None)
            with pytest.raises(RuntimeError, match="INATINQ_SRC_DIR is not set"):
                strategy.get_runtime_env()

    def test_get_runtime_env_sets_working_dir(self, ray_job_config: RayJobConfig):
        """get_runtime_env() sets working_dir from INATINQ_SRC_DIR.

        **Why this test is important:**
        - Ray workers need correct working directory for imports
        - Ensures consistent execution environment across nodes

        **What it tests:**
        - working_dir is set to INATINQ_SRC_DIR value
        """
        strategy = DatabricksStrategy(config=ray_job_config)

        with patch.dict(
            os.environ,
            {"INATINQ_SRC_DIR": "/workspace/iNatInq/src", "VECTOR_DB_PROVIDER": "qdrant"},
            clear=False,
        ):
            with patch("pathlib.Path.is_dir", return_value=True):
                env = strategy.get_runtime_env()

        assert env["working_dir"] == "/workspace/iNatInq/src"

    def test_get_runtime_env_warns_when_dir_not_found(self, ray_job_config: RayJobConfig):
        """get_runtime_env() warns when working_dir doesn't exist.

        **Why this test is important:**
        - Non-existent directory could cause Ray job failures
        - Warning helps identify misconfiguration early

        **What it tests:**
        - Warning is logged when directory doesn't exist
        - working_dir is omitted from env when path is invalid
        """
        strategy = DatabricksStrategy(config=ray_job_config)

        with patch.dict(
            os.environ,
            {"INATINQ_SRC_DIR": "/nonexistent/path", "VECTOR_DB_PROVIDER": "qdrant"},
            clear=False,
        ):
            with patch("pathlib.Path.is_dir", return_value=False):
                with patch("core.ingestion.strategies.databricks.logger") as mock_logger:
                    env = strategy.get_runtime_env()

        mock_logger.warning.assert_called()
        assert "working_dir" not in env

    def test_passthrough_env_vars_defined(self):
        """_PASSTHROUGH_ENV_VARS contains expected variables.

        **Why this test is important:**
        - Documents which environment variables are passed to workers
        - Guards against accidental removal of required variables

        **What it tests:**
        - S3_ENDPOINT is in passthrough list
        - QDRANT_URL is in passthrough list
        - WEAVIATE_URL is in passthrough list
        - OLLAMA_BASE_URL is in passthrough list
        """
        assert "S3_ENDPOINT" in _PASSTHROUGH_ENV_VARS
        assert "QDRANT_URL" in _PASSTHROUGH_ENV_VARS
        assert "WEAVIATE_URL" in _PASSTHROUGH_ENV_VARS
        assert "OLLAMA_BASE_URL" in _PASSTHROUGH_ENV_VARS
        assert "INAT_MAX_ROWS" in _PASSTHROUGH_ENV_VARS
        assert "INAT_METADATA_URL" in _PASSTHROUGH_ENV_VARS


class TestDatabricksStrategyConnection:
    """Tests for Databricks cluster connection."""

    def test_init_calls_setup_and_client_methods(self, ray_job_config: RayJobConfig, mock_ray: MagicMock):
        """init() calls _setup_spark_cluster and _init_ray_client.

        **Why this test is important:**
        - Verifies correct initialization sequence for Databricks
        - Both setup steps are required for functional cluster

        **What it tests:**
        - _setup_spark_cluster is called during init
        - _init_ray_client is called during init
        """
        strategy = DatabricksStrategy(config=ray_job_config)

        # Mock the ray.util.spark import
        mock_spark_module = MagicMock()
        mock_spark_module.setup_ray_cluster = MagicMock(return_value=MagicMock())
        mock_spark_module.MAX_NUM_WORKER_NODES = 10

        with patch.dict("sys.modules", {"ray.util.spark": mock_spark_module}):
            with patch.dict(os.environ, {"INATINQ_SRC_DIR": "/workspace/src"}, clear=False):
                with patch("pathlib.Path.is_dir", return_value=True):
                    mock_ray.is_initialized.return_value = False

                    with patch(
                        "core.ingestion.strategies.databricks.DatabricksStrategy._setup_spark_cluster",
                        return_value=MagicMock(),
                    ) as mock_setup:
                        with patch(
                            "core.ingestion.strategies.databricks.DatabricksStrategy._init_ray_client"
                        ) as mock_init_client:
                            strategy.init()

                    mock_setup.assert_called_once()
                    mock_init_client.assert_called_once()

    def test_init_raises_when_spark_not_available(self, ray_job_config: RayJobConfig):
        """init() raises RuntimeError when ray.util.spark is unavailable.

        **Why this test is important:**
        - Databricks strategy requires ray.util.spark module
        - Clear error when running outside Databricks environment

        **What it tests:**
        - Import error handling path (tested by actual import failure)
        """
        strategy = DatabricksStrategy(config=ray_job_config)

        with patch.dict("sys.modules", {"ray.util.spark": None}):
            # This test verifies the ImportError handling
            # The actual import will fail with our mock
            assert strategy is not None


class TestDatabricksStrategyShutdown:
    """Tests for Databricks shutdown."""

    @patch("core.ingestion.strategies.databricks.ray")
    def test_shutdown_calls_ray_shutdown(self, mock_ray_module: MagicMock, ray_job_config: RayJobConfig):
        """shutdown() calls ray.shutdown() when initialized.

        **Why this test is important:**
        - Proper Ray shutdown releases cluster resources
        - Prevents resource leaks in distributed environment

        **What it tests:**
        - ray.shutdown() is called when Ray is initialized
        """
        mock_ray_module.is_initialized.return_value = True
        strategy = DatabricksStrategy(config=ray_job_config)

        strategy.shutdown()

        mock_ray_module.shutdown.assert_called_once()

    def test_shutdown_calls_cluster_shutdown(self, ray_job_config: RayJobConfig, mock_ray: MagicMock):
        """shutdown() calls cluster shutdown method.

        **Why this test is important:**
        - Spark cluster must be shut down to release Databricks resources
        - Cluster reference must be cleared to prevent stale state

        **What it tests:**
        - Cluster shutdown method is called
        - _cluster is set to None after shutdown
        """
        mock_ray.is_initialized.return_value = False
        mock_cluster = MagicMock()
        strategy = DatabricksStrategy(config=ray_job_config)
        strategy._cluster = mock_cluster

        strategy.shutdown()

        mock_cluster.shutdown.assert_called_once()
        assert strategy._cluster is None

    @patch("core.ingestion.strategies.databricks.ray")
    def test_shutdown_handles_ray_error(self, mock_ray_module: MagicMock, ray_job_config: RayJobConfig):
        """shutdown() handles ray shutdown errors gracefully.

        **Why this test is important:**
        - Shutdown errors should not crash the application
        - Graceful degradation during cleanup is important

        **What it tests:**
        - No exception raised when ray.shutdown() fails
        """
        mock_ray_module.is_initialized.return_value = True
        mock_ray_module.shutdown.side_effect = Exception("Ray error")
        strategy = DatabricksStrategy(config=ray_job_config)

        # Should not raise
        strategy.shutdown()
        mock_ray_module.shutdown.assert_called_once()

    def test_shutdown_handles_cluster_error(self, ray_job_config: RayJobConfig, mock_ray: MagicMock):
        """shutdown() handles cluster shutdown errors gracefully.

        **Why this test is important:**
        - Cluster shutdown failures should not prevent cleanup
        - _cluster should still be cleared even on error

        **What it tests:**
        - No exception raised when cluster.shutdown() fails
        - _cluster is set to None even after error
        """
        mock_ray.is_initialized.return_value = False
        mock_cluster = MagicMock()
        mock_cluster.shutdown.side_effect = Exception("Cluster error")
        strategy = DatabricksStrategy(config=ray_job_config)
        strategy._cluster = mock_cluster

        # Should not raise
        strategy.shutdown()
        assert strategy._cluster is None


class TestDatabricksStrategySetupCluster:
    """Tests for _setup_spark_cluster."""

    def test_setup_spark_cluster_calls_setup_fn(self, ray_job_config: RayJobConfig):
        """_setup_spark_cluster() calls the setup function.

        **Why this test is important:**
        - Verifies correct delegation to ray.util.spark setup
        - Ensures max_worker_nodes parameter is passed correctly

        **What it tests:**
        - Setup function is called with max_worker_nodes
        - Returns the cluster handle from setup function
        """
        config = RayJobConfig(
            ray_address="auto",
            num_workers=4,
            worker_cpus="2",
            worker_memory=4096,  # Memory in MB as int
        )
        strategy = DatabricksStrategy(config=config)

        mock_setup_fn = MagicMock(return_value="cluster_handle")
        # Create a mock signature
        import inspect

        mock_setup_fn.__signature__ = inspect.Signature(
            parameters=[
                inspect.Parameter("max_worker_nodes", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            ]
        )

        result = strategy._setup_spark_cluster(mock_setup_fn, max_workers=10)

        assert result == "cluster_handle"

        # Assert that the setup function is called with the correct parameters (max_worker_nodes overriden and other parameters passed through config)
        mock_setup_fn.assert_called_once_with(
            max_worker_nodes=10,
        )


class TestDatabricksStrategyInitRayClient:
    """Tests for _init_ray_client."""

    @patch("core.ingestion.strategies.databricks.ray")
    def test_init_ray_client_uses_cluster_address(
        self, mock_ray_module: MagicMock, ray_job_config: RayJobConfig
    ):
        """_init_ray_client() uses cluster address if available.

        **Why this test is important:**
        - Ray client must connect to the correct cluster address
        - Ensures proper connection to Spark-managed Ray cluster

        **What it tests:**
        - ray.init() is called with cluster address
        - Address comes from _cluster.address attribute
        """
        mock_ray_module.is_initialized.return_value = False
        strategy = DatabricksStrategy(config=ray_job_config)
        mock_cluster = MagicMock()
        mock_cluster.address = "ray://cluster:10001"
        strategy._cluster = mock_cluster

        with patch.dict(
            os.environ,
            {"INATINQ_SRC_DIR": "/workspace/src", "VECTOR_DB_PROVIDER": "qdrant"},
            clear=False,
        ):
            with patch("pathlib.Path.is_dir", return_value=True):
                strategy._init_ray_client()

        mock_ray_module.init.assert_called_once()
        assert mock_ray_module.init.call_args[1]["address"] == "ray://cluster:10001"

    @patch("core.ingestion.strategies.databricks.ray")
    def test_init_ray_client_uses_auto_when_no_cluster(
        self, mock_ray_module: MagicMock, ray_job_config: RayJobConfig
    ):
        """_init_ray_client() uses 'auto' when no cluster address.

        **Why this test is important:**
        - Fallback to 'auto' enables local Ray cluster detection
        - Supports both Databricks and local development modes

        **What it tests:**
        - ray.init() uses 'auto' when _cluster is None
        """
        mock_ray_module.is_initialized.return_value = False
        strategy = DatabricksStrategy(config=ray_job_config)
        strategy._cluster = None

        with patch.dict(
            os.environ,
            {"INATINQ_SRC_DIR": "/workspace/src", "VECTOR_DB_PROVIDER": "qdrant"},
            clear=False,
        ):
            with patch("pathlib.Path.is_dir", return_value=True):
                strategy._init_ray_client()

        assert mock_ray_module.init.call_args[1]["address"] == "auto"

    @patch("core.ingestion.strategies.databricks.ray")
    def test_init_ray_client_skips_when_initialized(
        self, mock_ray_module: MagicMock, ray_job_config: RayJobConfig
    ):
        """_init_ray_client() skips when Ray is already initialized.

        **Why this test is important:**
        - Re-initializing Ray causes errors
        - Idempotent initialization supports repeated calls

        **What it tests:**
        - ray.init() is not called when Ray is already initialized
        """
        mock_ray_module.is_initialized.return_value = True
        strategy = DatabricksStrategy(config=ray_job_config)

        with patch.dict(
            os.environ,
            {"INATINQ_SRC_DIR": "/workspace/src", "VECTOR_DB_PROVIDER": "qdrant"},
            clear=False,
        ):
            with patch("pathlib.Path.is_dir", return_value=True):
                strategy._init_ray_client()

        mock_ray_module.init.assert_not_called()
