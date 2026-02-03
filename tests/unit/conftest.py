"""Top-level conftest for unit tests.

This module provides mocks for external dependencies that are not available
or not needed during unit testing. Mocks are installed before any test modules
are imported.

Mocked modules:
- ray: Not available on Python 3.14+
- databricks.sdk: External service dependency
"""

import sys
from unittest.mock import MagicMock

# =============================================================================
# Mock Ray module
# =============================================================================
# Ray doesn't support Python 3.14+, so we mock it for unit tests.
# The mock provides the basic structure needed for imports to succeed.

mock_ray = MagicMock()
mock_ray.job_submission = MagicMock()
mock_ray.job_submission.JobSubmissionClient = MagicMock()
mock_ray.job_submission.JobStatus = MagicMock()


def mock_ray_remote(*args, **kwargs):
    """Mock @ray.remote decorator that adds .remote attribute."""

    def decorator(fn):
        fn.remote = MagicMock(return_value=MagicMock())
        fn.options = MagicMock(return_value=fn)
        return fn

    # Handle both @ray.remote and @ray.remote(num_cpus=1) syntax
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return decorator(args[0])
    return decorator


mock_ray.remote = mock_ray_remote

sys.modules["ray"] = mock_ray
sys.modules["ray.job_submission"] = mock_ray.job_submission

# =============================================================================
# Mock Databricks SDK
# =============================================================================
# Databricks SDK is an external service dependency that we mock for unit tests.

mock_databricks = MagicMock()
mock_databricks.sdk = MagicMock()
mock_databricks.sdk.WorkspaceClient = MagicMock()

sys.modules["databricks"] = mock_databricks
sys.modules["databricks.sdk"] = mock_databricks.sdk
