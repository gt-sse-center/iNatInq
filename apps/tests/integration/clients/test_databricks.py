"""Integration tests for Databricks workspace connectivity.

These tests exercise a real Azure Databricks workspace using the official SDK.
They are skipped unless the required environment variables are set.
"""

import logging
import os

import pytest
from databricks.sdk import WorkspaceClient

pytestmark = pytest.mark.integration
logger = logging.getLogger(__name__)


def _require_env(name: str) -> str:
    """Return the environment value or skip the test if missing."""
    value = os.getenv(name)
    if not value:
        pytest.skip(f"{name} is not set")
    return value


def test_workspace_connection() -> None:
    """Verify the SDK can authenticate and reach the workspace."""
    host = _require_env("DATABRICKS_HOST")
    token = _require_env("DATABRICKS_TOKEN")

    client = WorkspaceClient(host=host, token=token)
    current_user = client.current_user.me()

    assert current_user is not None
    logger.info("Databricks connectivity verified for user: %s", current_user.user_name)
