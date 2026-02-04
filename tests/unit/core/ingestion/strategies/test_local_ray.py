"""Unit tests for LocalRayStrategy.

This module tests the LocalRayStrategy class which handles local Ray cluster
connections for distributed data ingestion.

# Test Coverage

- Strategy initialization with config and from environment
- Runtime environment configuration handling
- Ray cluster connection lifecycle (init/shutdown)
- Thread limit management for worker processes
- Error handling for connection failures and resource limits

# Running Tests

    uv run pytest tests/unit/core/ingestion/strategies/test_local_ray.py -v
"""

import resource
from unittest.mock import MagicMock, patch

import pytest

from config import RayJobConfig
from core.ingestion.strategies.local_ray import LocalRayStrategy


@pytest.fixture(autouse=True)
def reset_ray_mock(mock_ray: MagicMock):
    """Reset ray mock state before each test."""
    mock_ray.reset_mock()
    yield


class TestLocalRayStrategyInit:
    """Tests for LocalRayStrategy initialization."""

    def test_creates_strategy_with_config(self, ray_job_config: RayJobConfig):
        """Strategy is created with provided config.

        **Why this test is important:**

        - Ensures the strategy correctly stores the provided configuration
        - Verifies basic object instantiation works as expected

        **What it tests:**

        - Creating a LocalRayStrategy with an explicit RayJobConfig
        - The config property returns the same config object
        """
        strategy = LocalRayStrategy(config=ray_job_config)
        assert strategy.config == ray_job_config

    def test_from_env_creates_strategy(self):
        """from_env() creates strategy from environment.

        **Why this test is important:**

        - Validates the factory method for environment-based configuration
        - Ensures namespace is passed correctly to the underlying config

        **What it tests:**

        - The from_env() class method calls RayJobConfig.from_env
        - The namespace parameter is forwarded correctly
        - The returned strategy has the expected config
        """
        with patch("core.ingestion.strategies.local_ray.RayJobConfig.from_env") as mock_from_env:
            mock_config = MagicMock(spec=RayJobConfig)
            mock_from_env.return_value = mock_config

            strategy = LocalRayStrategy.from_env("test-namespace")

            mock_from_env.assert_called_once_with("test-namespace")
            assert strategy.config == mock_config

    def test_config_property_returns_config(self, ray_job_config: RayJobConfig):
        """config property returns the configuration.

        **Why this test is important:**

        - Verifies the config property provides access to the stored config
        - Ensures identity preservation (same object, not a copy)

        **What it tests:**

        - The config property returns the exact same config object
        - No transformation or copying occurs on access
        """
        strategy = LocalRayStrategy(config=ray_job_config)
        assert strategy.config is ray_job_config


class TestLocalRayStrategyRuntimeEnv:
    """Tests for runtime environment handling."""

    def test_get_runtime_env_returns_copy_of_config(self, ray_job_config: RayJobConfig):
        """get_runtime_env() returns a copy of config runtime_env.

        **Why this test is important:**

        - Prevents accidental mutation of the original config
        - Ensures callers can safely modify the returned dict

        **What it tests:**

        - The returned dict matches the config runtime_env
        - Modifications to the returned dict do not affect the original
        """
        strategy = LocalRayStrategy(config=ray_job_config)
        env = strategy.get_runtime_env()
        assert env == {"pip": ["requests"]}
        # Verify it's a copy
        env["new_key"] = "value"
        assert "new_key" not in strategy.get_runtime_env()

    def test_get_runtime_env_returns_empty_when_none(self):
        """get_runtime_env() returns empty dict when runtime_env is None.

        **Why this test is important:**

        - Ensures graceful handling of missing runtime_env
        - Prevents None-related errors in downstream code

        **What it tests:**

        - Config with no runtime_env returns empty dict
        - No exceptions are raised for None runtime_env
        """
        config = RayJobConfig(ray_address="ray://localhost:10001")
        strategy = LocalRayStrategy(config=config)
        assert strategy.get_runtime_env() == {}


class TestLocalRayStrategyConnection:
    """Tests for Ray cluster connection."""

    def test_init_skips_when_already_initialized(self, ray_job_config: RayJobConfig, mock_ray: MagicMock):
        """init() skips initialization when Ray is already connected.

        **Why this test is important:**

        - Prevents redundant connection attempts
        - Ensures idempotent behavior for repeated init() calls

        **What it tests:**

        - When ray.is_initialized() returns True, ray.init() is not called
        - No side effects occur when already connected
        """
        mock_ray.is_initialized.return_value = True
        strategy = LocalRayStrategy(config=ray_job_config)

        strategy.init()

        mock_ray.init.assert_not_called()

    def test_init_starts_local_when_no_ray_address(self, mock_ray: MagicMock):
        """init() starts a local cluster when RAY_ADDRESS is not set.

        **Why this test is important:**

        - Validates required configuration before connection attempt
        - Provides clear error message for misconfiguration

        **What it tests:**

        - Ray is initialized locally when ray_address is None
        """
        config = RayJobConfig(ray_address=None)
        strategy = LocalRayStrategy(config=config)
        mock_ray.is_initialized.return_value = False

        strategy.init()

        mock_ray.init.assert_called_once()
        call_kwargs = mock_ray.init.call_args[1]
        assert "address" not in call_kwargs or call_kwargs["address"] is None

    def test_init_connects_to_ray_cluster(self, ray_job_config: RayJobConfig, mock_ray: MagicMock):
        """init() connects to Ray cluster with correct parameters.

        **Why this test is important:**

        - Verifies correct Ray initialization parameters
        - Ensures namespace isolation and address configuration

        **What it tests:**

        - ray.init() is called with the correct address
        - Namespace is set from config
        - ignore_reinit_error is True for idempotent behavior
        """
        mock_ray.is_initialized.return_value = False
        strategy = LocalRayStrategy(config=ray_job_config)

        with patch("core.ingestion.strategies.local_ray.LocalRayStrategy._increase_thread_limit"):
            strategy.init()

        mock_ray.init.assert_called_once()
        call_kwargs = mock_ray.init.call_args[1]
        assert call_kwargs["address"] == "ray://localhost:10001"
        assert call_kwargs["namespace"] == "test-namespace"
        assert call_kwargs["ignore_reinit_error"] is True

    def test_init_raises_runtime_error_on_failure(self, ray_job_config: RayJobConfig, mock_ray: MagicMock):
        """init() raises RuntimeError when connection fails.

        **Why this test is important:**

        - Ensures connection failures are properly propagated
        - Provides meaningful error context for debugging

        **What it tests:**

        - RuntimeError is raised when ray.init() throws an exception
        - Error message indicates connection failure
        """
        mock_ray.is_initialized.return_value = False
        mock_ray.init.side_effect = Exception("Connection refused")
        strategy = LocalRayStrategy(config=ray_job_config)

        with patch("core.ingestion.strategies.local_ray.LocalRayStrategy._increase_thread_limit"):
            with pytest.raises(RuntimeError, match="Failed to connect"):
                strategy.init()

        # Reset side_effect for other tests
        mock_ray.init.side_effect = None


class TestLocalRayStrategyShutdown:
    """Tests for Ray shutdown."""

    def test_shutdown_calls_ray_shutdown(self, ray_job_config: RayJobConfig, mock_ray: MagicMock):
        """shutdown() calls ray.shutdown() when initialized.

        **Why this test is important:**

        - Ensures proper resource cleanup on shutdown
        - Verifies Ray cluster connection is properly terminated

        **What it tests:**

        - ray.shutdown() is called when Ray is initialized
        - Shutdown proceeds normally when connected
        """
        mock_ray.is_initialized.return_value = True
        strategy = LocalRayStrategy(config=ray_job_config)

        strategy.shutdown()

        mock_ray.shutdown.assert_called_once()

    def test_shutdown_skips_when_not_initialized(self, ray_job_config: RayJobConfig, mock_ray: MagicMock):
        """shutdown() does nothing when Ray is not initialized.

        **Why this test is important:**

        - Prevents errors from shutting down non-existent connections
        - Ensures safe no-op behavior when not connected

        **What it tests:**

        - ray.shutdown() is not called when Ray is not initialized
        - No exceptions are raised for redundant shutdown calls
        """
        mock_ray.is_initialized.return_value = False
        strategy = LocalRayStrategy(config=ray_job_config)

        strategy.shutdown()

        mock_ray.shutdown.assert_not_called()

    def test_shutdown_handles_errors_gracefully(self, ray_job_config: RayJobConfig, mock_ray: MagicMock):
        """shutdown() catches and logs errors instead of raising.

        **Why this test is important:**

        - Ensures graceful degradation during cleanup
        - Prevents shutdown errors from crashing the application

        **What it tests:**

        - Exceptions from ray.shutdown() are caught
        - No exception propagates to the caller
        """
        mock_ray.is_initialized.return_value = True
        mock_ray.shutdown.side_effect = Exception("Shutdown error")
        strategy = LocalRayStrategy(config=ray_job_config)

        # Should not raise
        strategy.shutdown()


class TestLocalRayStrategyThreadLimit:
    """Tests for thread limit handling."""

    def test_increase_thread_limit_success(self, ray_job_config: RayJobConfig):
        """_increase_thread_limit() increases soft limit.

        **Why this test is important:**

        - Ray workers may need higher thread limits than defaults
        - Ensures proper resource limit adjustment for worker processes

        **What it tests:**

        - resource.setrlimit() is called to increase thread limits
        - The method completes successfully with valid resource limits
        """
        strategy = LocalRayStrategy(config=ray_job_config)

        with patch("core.ingestion.strategies.local_ray.resource") as mock_resource:
            mock_resource.getrlimit.return_value = (256, 8192)
            mock_resource.RLIMIT_NPROC = resource.RLIMIT_NPROC
            mock_resource.RLIM_INFINITY = resource.RLIM_INFINITY

            strategy._increase_thread_limit()

            mock_resource.setrlimit.assert_called_once()

    def test_increase_thread_limit_handles_oserror(self, ray_job_config: RayJobConfig):
        """_increase_thread_limit() handles OSError gracefully.

        **Why this test is important:**

        - Some environments may not allow resource limit changes
        - Strategy should continue working despite permission errors

        **What it tests:**

        - OSError from getrlimit() is caught
        - No exception propagates to the caller
        """
        strategy = LocalRayStrategy(config=ray_job_config)

        with patch("core.ingestion.strategies.local_ray.resource") as mock_resource:
            mock_resource.getrlimit.side_effect = OSError("Permission denied")
            mock_resource.RLIMIT_NPROC = resource.RLIMIT_NPROC

            # Should not raise
            strategy._increase_thread_limit()

    def test_increase_thread_limit_handles_value_error(self, ray_job_config: RayJobConfig):
        """_increase_thread_limit() handles ValueError gracefully.

        **Why this test is important:**

        - Invalid resource values should not crash the strategy
        - Graceful degradation when limits cannot be set

        **What it tests:**

        - ValueError from setrlimit() is caught
        - No exception propagates to the caller
        """
        strategy = LocalRayStrategy(config=ray_job_config)

        with patch("core.ingestion.strategies.local_ray.resource") as mock_resource:
            mock_resource.getrlimit.return_value = (256, 8192)
            mock_resource.setrlimit.side_effect = ValueError("Invalid value")
            mock_resource.RLIMIT_NPROC = resource.RLIMIT_NPROC
            mock_resource.RLIM_INFINITY = resource.RLIM_INFINITY

            # Should not raise
            strategy._increase_thread_limit()

    def test_increase_thread_limit_respects_hard_limit(self, ray_job_config: RayJobConfig):
        """_increase_thread_limit() does not exceed hard limit.

        **Why this test is important:**

        - Hard limits are system-enforced and cannot be exceeded
        - Ensures the method respects operating system constraints

        **What it tests:**

        - setrlimit() is called with hard limit when it is lower than target
        - The soft limit is set to the minimum of target and hard limit
        """
        strategy = LocalRayStrategy(config=ray_job_config)

        with patch("core.ingestion.strategies.local_ray.resource") as mock_resource:
            mock_resource.getrlimit.return_value = (256, 1024)  # Hard limit lower than 8192
            mock_resource.RLIMIT_NPROC = resource.RLIMIT_NPROC
            mock_resource.RLIM_INFINITY = resource.RLIM_INFINITY

            strategy._increase_thread_limit()

            # Should set to hard limit (1024), not 8192
            mock_resource.setrlimit.assert_called_once_with(resource.RLIMIT_NPROC, (1024, 1024))

    def test_increase_thread_limit_skips_when_already_high(self, ray_job_config: RayJobConfig):
        """_increase_thread_limit() skips when soft limit is already high.

        **Why this test is important:**

        - Avoids unnecessary system calls when limits are sufficient
        - Prevents potential errors from redundant limit changes

        **What it tests:**

        - setrlimit() is not called when soft limit >= target
        - Method exits early with no side effects
        """
        strategy = LocalRayStrategy(config=ray_job_config)

        with patch("core.ingestion.strategies.local_ray.resource") as mock_resource:
            mock_resource.getrlimit.return_value = (8192, 16384)  # Already at 8192
            mock_resource.RLIMIT_NPROC = resource.RLIMIT_NPROC
            mock_resource.RLIM_INFINITY = resource.RLIM_INFINITY

            strategy._increase_thread_limit()

            # Should not call setrlimit since soft (8192) >= target (8192)
            mock_resource.setrlimit.assert_not_called()
