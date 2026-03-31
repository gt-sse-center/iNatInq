"""E2E tests for CLI commands.

Tests CLI commands against the live Docker Compose stack to verify
the CLI→Docker bridge works correctly.
"""

from __future__ import annotations

import subprocess

import pytest


@pytest.mark.e2e
def test_docker_health_command(docker_compose_stack: None):
    """Test that 'inq docker health' succeeds against the live stack.

    Verifies the CLI can check Docker service health and reports success
    when all services are up.
    """
    # Run: uv run inq docker health
    result = subprocess.run(
        ["uv", "run", "inq", "docker", "health"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    # Should exit with 0 (success)
    assert result.returncode == 0, f"docker health failed: {result.stderr}"

    # Output should mention services are healthy
    output = result.stdout.lower()
    assert "healthy" in output or "ok" in output or "✓" in output


@pytest.mark.e2e
def test_e2e_help_command():
    """Test that 'inq test e2e --help' shows help text correctly."""
    # Run: uv run inq test e2e --help
    result = subprocess.run(
        ["uv", "run", "inq", "test", "e2e", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    # Should exit with 0
    assert result.returncode == 0

    # Help text should describe the command
    help_text = result.stdout.lower()
    assert "e2e" in help_text
    assert "docker" in help_text or "compose" in help_text or "running" in help_text


@pytest.mark.e2e
def test_e2e_collection_option():
    """Test that 'inq test e2e' can collect tests with --co flag."""
    # Run: uv run inq test e2e -- --co -q
    result = subprocess.run(
        ["uv", "run", "inq", "test", "e2e", "--", "--co", "-q"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    # Should exit with 0
    assert result.returncode == 0

    # Should list E2E test modules and tests
    output = result.stdout
    assert "e2e" in output
    # Should have collected some tests
    assert "test_" in output


@pytest.mark.e2e
def test_cli_test_unit_command():
    """Test that 'inq test unit' command is registered and shows help."""
    # Verify the unit command exists
    result = subprocess.run(
        ["uv", "run", "inq", "test", "unit", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    # Should exit with 0
    assert result.returncode == 0

    # Help text should describe unit tests
    assert "unit" in result.stdout.lower()


@pytest.mark.e2e
def test_cli_test_integration_command():
    """Test that 'inq test integration' command is registered and shows help."""
    # Verify the integration command exists
    result = subprocess.run(
        ["uv", "run", "inq", "test", "integration", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    # Should exit with 0
    assert result.returncode == 0

    # Help text should describe integration tests
    help_text = result.stdout.lower()
    assert "integration" in help_text
    assert "docker" in help_text or "parallel" in help_text
