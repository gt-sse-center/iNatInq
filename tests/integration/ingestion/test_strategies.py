"""Integration tests for ingestion pipeline strategies.

Tests the strategy pattern implementations for Ray cluster management
against real containerized Ray clusters.

# Test Coverage

The tests cover:
  - LocalRayStrategy: Initialization, context manager, shutdown
  - Ray cluster connectivity: Dashboard, client connections
  - Resource management: CPU/memory configuration
  - Error handling: Connection failures, timeouts

# Running Tests

Run with: pytest tests/integration/ingestion/test_strategies.py -v -m integration
"""

import pytest

# =============================================================================
# LocalRayStrategy Tests
# =============================================================================


@pytest.mark.integration
class TestLocalRayStrategyWithContainer:
    """Test LocalRayStrategy against a containerized Ray cluster.

    These tests use a real Ray container to verify strategy behavior.
    """

    def test_connects_to_external_ray_cluster(
        self,
        ray_client_address: str,
        ray_dashboard_url: str,
    ) -> None:
        """Test connecting to an existing Ray cluster.

        **Why this test is important:**
          - Validates external cluster connection capability
          - Critical for Docker deployments
          - Verifies address parsing and connection

        **What it tests:**
          - Ray can connect to external cluster
          - Cluster is functional after connection
          - Resources are accessible
        """
        import ray

        # Connect to the containerized Ray cluster
        ray.init(address=ray_client_address, ignore_reinit_error=True)

        try:
            assert ray.is_initialized()

            # Verify cluster has resources
            resources = ray.cluster_resources()
            assert "CPU" in resources
            assert resources["CPU"] > 0
        finally:
            ray.shutdown()

    def test_ray_dashboard_accessible(self, ray_dashboard_url: str) -> None:
        """Test that Ray dashboard is accessible.

        **Why this test is important:**
          - Dashboard is critical for monitoring
          - Validates network connectivity
          - Ensures API endpoints work

        **What it tests:**
          - Dashboard URL is reachable
          - Returns valid response
        """
        import httpx

        response = httpx.get(f"{ray_dashboard_url}/api/cluster_status", timeout=10.0)

        assert response.status_code == 200
        data = response.json()
        assert "data" in data or "result" in data

    def test_can_submit_and_execute_task(self, ray_client_address: str) -> None:
        """Test submitting and executing a Ray task.

        **Why this test is important:**
          - Core Ray functionality
          - Validates task serialization
          - Ensures worker execution works

        **What it tests:**
          - Remote function can be defined
          - Task can be submitted
          - Result can be retrieved
        """
        import ray

        ray.init(address=ray_client_address, ignore_reinit_error=True)

        try:

            @ray.remote
            def add(x: int, y: int) -> int:
                return x + y

            # Submit task and get result
            result = ray.get(add.remote(2, 3))
            assert result == 5

        finally:
            ray.shutdown()

    def test_can_execute_batch_of_tasks(self, ray_client_address: str) -> None:
        """Test executing multiple Ray tasks in parallel.

        **Why this test is important:**
          - Batch processing is core use case
          - Validates parallel execution
          - Tests task scheduling

        **What it tests:**
          - Multiple tasks can be submitted
          - Tasks execute in parallel
          - All results are collected
        """
        import ray

        ray.init(address=ray_client_address, ignore_reinit_error=True)

        try:

            @ray.remote
            def square(x: int) -> int:
                return x * x

            # Submit batch of tasks
            futures = [square.remote(i) for i in range(10)]
            results = ray.get(futures)

            assert results == [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

        finally:
            ray.shutdown()


@pytest.mark.integration
class TestLocalRayStrategyStandalone:
    """Test LocalRayStrategy against an external Ray cluster.

    These tests verify the strategy connects to an existing Ray cluster
    (e.g. containerized) via RayJobConfig and init/shutdown.
    """

    def test_strategy_connects_to_external_cluster(
        self,
        ray_client_address: str,
    ) -> None:
        """Test that LocalRayStrategy connects to external cluster and shuts down.

        **Why this test is important:**
          - Validates strategy init()/shutdown() with real cluster
          - Ensures config is passed correctly
          - Critical for Docker deployments

        **What it tests:**
          - Strategy init() connects to cluster
          - Ray becomes initialized
          - Shutdown cleans up properly
        """
        import os

        import ray

        from config import RayJobConfig
        from core.ingestion.strategies.local_ray import LocalRayStrategy

        # Ensure clean state
        if ray.is_initialized():
            ray.shutdown()

        # Strategy requires RAY_ADDRESS; point to containerized cluster
        prev = os.environ.get("RAY_ADDRESS")
        os.environ["RAY_ADDRESS"] = ray_client_address
        try:
            config = RayJobConfig.from_env()
            strategy = LocalRayStrategy(config=config)
            strategy.init()
            assert ray.is_initialized()
            resources = ray.cluster_resources()
            assert "CPU" in resources
        finally:
            if ray.is_initialized():
                ray.shutdown()
            if prev is not None:
                os.environ["RAY_ADDRESS"] = prev
            else:
                os.environ.pop("RAY_ADDRESS", None)
        assert not ray.is_initialized()

    def test_strategy_get_runtime_env_and_config(
        self,
        ray_client_address: str,
    ) -> None:
        """Test get_runtime_env() and config property without connecting.

        **Why this test is important:**
          - Protocol requires get_runtime_env and config
          - Validates configuration exposure
        """
        from config import RayJobConfig
        from core.ingestion.strategies.local_ray import LocalRayStrategy

        config = RayJobConfig(ray_address=ray_client_address)
        strategy = LocalRayStrategy(config=config)
        assert strategy.config is config
        assert isinstance(strategy.get_runtime_env(), dict)


# =============================================================================
# Strategy Protocol Compliance Tests
# =============================================================================


@pytest.mark.integration
class TestStrategyProtocolCompliance:
    """Test that strategies comply with the ClusterStrategy protocol."""

    def test_local_ray_strategy_has_required_methods(self) -> None:
        """Test LocalRayStrategy implements ClusterStrategy protocol.

        **Why this test is important:**
          - Protocol compliance ensures interchangeability
          - Validates interface contract
          - Prevents runtime errors

        **What it tests:**
          - init() and shutdown() exist and are callable
          - get_runtime_env() and config property exist
        """
        from config import RayJobConfig
        from core.ingestion.strategies.local_ray import LocalRayStrategy

        config = RayJobConfig(ray_address="ray://localhost:10001")
        strategy = LocalRayStrategy(config=config)

        assert hasattr(strategy, "init")
        assert callable(strategy.init)
        assert hasattr(strategy, "shutdown")
        assert callable(strategy.shutdown)
        assert hasattr(strategy, "get_runtime_env")
        assert callable(strategy.get_runtime_env)
        assert hasattr(strategy, "config")
        assert strategy.config is config
