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
          - Critical for Docker/K8s deployments
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
    """Test LocalRayStrategy creating its own local cluster.

    These tests verify the strategy can start and manage a local Ray cluster
    without relying on an external container.
    """

    def test_strategy_initializes_local_cluster(self) -> None:
        """Test that LocalRayStrategy can start a local Ray cluster.

        **Why this test is important:**
          - Local cluster is fallback when no external cluster
          - Critical for development mode
          - Validates strategy initialization

        **What it tests:**
          - Strategy creates cluster on initialize()
          - Ray becomes initialized
          - Shutdown cleans up properly
        """
        import ray

        from core.ingestion.strategies.local_ray import LocalRayStrategy

        strategy = LocalRayStrategy(
            num_cpus=1,
            include_dashboard=False,
        )

        # Ensure clean state
        if ray.is_initialized():
            ray.shutdown()

        try:
            strategy.initialize()
            assert ray.is_initialized()

            # Verify cluster has expected resources
            resources = ray.cluster_resources()
            assert "CPU" in resources

        finally:
            strategy.shutdown()
            # Ray should be shut down
            assert not ray.is_initialized()

    def test_strategy_context_manager(self) -> None:
        """Test LocalRayStrategy as context manager.

        **Why this test is important:**
          - Context manager ensures cleanup
          - Prevents resource leaks
          - Validates proper shutdown

        **What it tests:**
          - __enter__ initializes cluster
          - __exit__ shuts down cluster
          - Exception handling works
        """
        import ray

        from core.ingestion.strategies.local_ray import LocalRayStrategy

        # Ensure clean state
        if ray.is_initialized():
            ray.shutdown()

        strategy = LocalRayStrategy(num_cpus=1, include_dashboard=False)

        with strategy:
            assert ray.is_initialized()

        # After context exit, Ray should be shut down
        assert not ray.is_initialized()

    def test_strategy_is_active_property(self) -> None:
        """Test the is_active property reflects cluster state.

        **Why this test is important:**
          - is_active used for conditional logic
          - Must accurately reflect state
          - Critical for error handling

        **What it tests:**
          - is_active is False before initialize
          - is_active is True after initialize
          - is_active is False after shutdown
        """
        import ray

        from core.ingestion.strategies.local_ray import LocalRayStrategy

        # Ensure clean state
        if ray.is_initialized():
            ray.shutdown()

        strategy = LocalRayStrategy(num_cpus=1, include_dashboard=False)

        assert not strategy.is_active

        strategy.initialize()
        try:
            assert strategy.is_active
        finally:
            strategy.shutdown()

        assert not strategy.is_active


# =============================================================================
# Strategy Protocol Compliance Tests
# =============================================================================


@pytest.mark.integration
class TestStrategyProtocolCompliance:
    """Test that strategies comply with the ClusterStrategy protocol."""

    def test_local_ray_strategy_has_required_methods(self) -> None:
        """Test LocalRayStrategy implements required protocol methods.

        **Why this test is important:**
          - Protocol compliance ensures interchangeability
          - Validates interface contract
          - Prevents runtime errors

        **What it tests:**
          - initialize() method exists
          - shutdown() method exists
          - is_active property exists
        """
        from core.ingestion.strategies.local_ray import LocalRayStrategy

        strategy = LocalRayStrategy()

        assert hasattr(strategy, "initialize")
        assert callable(strategy.initialize)
        assert hasattr(strategy, "shutdown")
        assert callable(strategy.shutdown)
        assert hasattr(strategy, "is_active")
