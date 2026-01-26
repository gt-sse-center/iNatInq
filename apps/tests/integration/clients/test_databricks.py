"""Integration tests for Databricks workspace connectivity.

These tests exercise a real Azure Databricks workspace using the official SDK.
They are skipped unless the required environment variables are set.
"""

import logging
from pathlib import Path

import pytest
from databricks.sdk import WorkspaceClient

pytestmark = pytest.mark.integration
logger = logging.getLogger(__name__)
ENV_LOCAL_PATH = Path(__file__).resolve().parents[3] / "zarf/compose/dev/.env.local"


def _read_env_file(path: Path) -> dict[str, str]:
    """Read a .env-style file and return key/value pairs."""
    if not path.exists():
        return {}

    data: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].strip()
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        data[key.strip()] = value.strip().strip('"').strip("'")
    return data


def test_workspace_connection() -> None:
    """Verify the SDK can authenticate and reach the workspace."""
    env_data = _read_env_file(ENV_LOCAL_PATH)
    host = env_data.get("DATABRICKS_HOST")
    token = env_data.get("DATABRICKS_TOKEN")
    if not host or not token:
        pytest.skip("Skipping: DATABRICKS_HOST and DATABRICKS_TOKEN not set in zarf/compose/dev/.env.local")

    client = WorkspaceClient(host=host, token=token)
    current_user = client.current_user.me()

    assert current_user is not None
    logger.info("Databricks connectivity verified for user: %s", current_user.user_name)
