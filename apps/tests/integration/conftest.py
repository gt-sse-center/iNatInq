"""Shared fixtures for integration tests.

This module provides common fixtures for integration tests that use real
infrastructure via testcontainers. Fixtures are scoped appropriately to
balance test isolation with startup cost.

## Design Principles

1. **Real Infrastructure**: Use actual containers, not mocks
2. **Self-Contained**: Tests don't depend on external docker-compose
3. **Deterministic**: Same behavior locally and in CI
4. **Efficient**: Session-scoped containers where safe

## Container Scoping

- **Session-scoped**: Shared across all tests in a run (e.g., MinIO)
- **Module-scoped**: Shared within a test file
- **Function-scoped**: Fresh per test (when isolation is critical)

Most containers are session-scoped for performance, but tests must clean
up their own data to avoid cross-test pollution.

## Port Conflicts & Parallel Execution

Testcontainers automatically handles port conflicts:
- Random ephemeral host ports are assigned (never hardcoded)
- Each pytest process gets independent containers
- Works safely with pytest-xdist parallel execution

## Orphan Container Cleanup (Ryuk)

Testcontainers includes a "Ryuk" sidecar container that automatically
cleans up orphaned containers if tests crash without proper teardown.
This runs by default and requires no configuration.

To disable Ryuk (e.g., in CI with privileged container cleanup):
    export TESTCONTAINERS_RYUK_DISABLED=true

To change Ryuk image:
    export RYUK_CONTAINER_IMAGE=testcontainers/ryuk:0.5.1
"""

import logging

import pytest

# Note: testcontainers deprecation warnings are filtered in pyproject.toml
# via [tool.pytest.ini_options].filterwarnings

# Configure logging for integration tests
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# Reduce noise from third-party libraries
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("docker").setLevel(logging.WARNING)


@pytest.fixture(scope="session")
def integration_test_session():
    """Marker fixture to identify integration test sessions.

    This can be used to run setup/teardown at the session level.
    """
    logging.info("Starting integration test session")
    yield
    logging.info("Ending integration test session")
