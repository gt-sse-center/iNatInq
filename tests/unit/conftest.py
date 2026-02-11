"""Top-level conftest for unit tests.

This module provides mocks for external dependencies that are not available
or not needed during unit testing. Ray is mocked only for tests under
tests/unit/ (via autouse fixture) so integration tests use the real ray.
"""

import sys
from unittest.mock import MagicMock

import pytest

# =============================================================================
# Mock Ray module (injected only during unit tests via _inject_ray_mock)
# =============================================================================

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

    if len(args) == 1 and callable(args[0]) and not kwargs:
        return decorator(args[0])
    return decorator


mock_ray.remote = mock_ray_remote


@pytest.fixture(autouse=True)
def _inject_ray_mock():
    """Replace ray in sys.modules for the duration of each unit test only."""
    real_ray = sys.modules.get("ray")
    real_ray_js = sys.modules.get("ray.job_submission")
    sys.modules["ray"] = mock_ray
    sys.modules["ray.job_submission"] = mock_ray.job_submission
    try:
        yield
    finally:
        if real_ray is not None:
            sys.modules["ray"] = real_ray
        else:
            sys.modules.pop("ray", None)
        if real_ray_js is not None:
            sys.modules["ray.job_submission"] = real_ray_js
        else:
            sys.modules.pop("ray.job_submission", None)


# =============================================================================
# Mock Databricks SDK
# =============================================================================
# Databricks SDK is an external service dependency that we mock for unit tests.

mock_databricks = MagicMock()
mock_databricks.sdk = MagicMock()
mock_databricks.sdk.WorkspaceClient = MagicMock()

sys.modules["databricks"] = mock_databricks
sys.modules["databricks.sdk"] = mock_databricks.sdk
