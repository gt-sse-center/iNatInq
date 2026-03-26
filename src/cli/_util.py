"""Shared utilities for CLI commands."""

from __future__ import annotations

import os
import subprocess
import sys
import webbrowser
from pathlib import Path

import typer

# Constants
REPO_ROOT: Path = Path(__file__).resolve().parents[2]
COMPOSE_FILE: str = "zarf/compose/dev/docker-compose.yaml"
SMOKE_ENV_FILE: str = "zarf/compose/dev/.env.local"
DATABRICKS_ENV_FILE: str = "zarf/databricks/dev/.env.local"
DATABRICKS_CLUSTER_SPEC: str = "zarf/databricks/dev/inatinq-azure-databricks-cluster.json"
DEFAULT_S3_BUCKET: str = "pipeline"


def run(
    cmd: list[str],
    *,
    check: bool = True,
    capture: bool = False,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a command with subprocess.

    Args:
        cmd: Command and arguments as a list.
        check: If True, raise typer.Exit on non-zero exit code.
        capture: If True, capture stdout/stderr; if False, stream to terminal.
        env: Optional environment variables to pass to subprocess.

    Returns:
        CompletedProcess result.
    """
    merged_env = {**os.environ, **env} if env else None
    try:
        if capture:
            result = subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                check=check,
                capture_output=True,
                text=True,
                env=merged_env,
            )
        else:
            result = subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                check=check,
                text=True,
                env=merged_env,
            )
        return result
    except subprocess.CalledProcessError as e:
        if e.stderr:
            typer.echo(e.stderr, err=True)
        raise typer.Exit(code=1) from e


def run_interactive(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    """Run an interactive command with TTY passthrough.

    Args:
        cmd: Command and arguments as a list.
        env: Optional environment variables to pass to subprocess.
    """
    merged_env = {**os.environ, **env} if env else None
    try:
        subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            stdin=sys.stdin,
            stdout=sys.stdout,
            stderr=sys.stderr,
            check=True,
            env=merged_env,
        )
    except subprocess.CalledProcessError as e:
        typer.echo(f"Command failed with exit code {e.returncode}", err=True)
        raise typer.Exit(code=1) from e


def docker_compose(*args: str) -> list[str]:
    """Build a docker compose command with the standard compose file.

    Args:
        args: Additional arguments to pass to docker compose.

    Returns:
        Full command as a list.
    """
    return ["docker", "compose", "-f", COMPOSE_FILE, *args]


def ensure_databricks_env(env_file: str = DATABRICKS_ENV_FILE) -> None:
    """Ensure Databricks environment file exists.

    Args:
        env_file: Path to the Databricks env file.

    Raises:
        typer.Exit: If the env file does not exist.
    """
    env_path = REPO_ROOT / env_file
    if not env_path.exists():
        msg = (
            f"Databricks env file not found: {env_file}\n"
            f"Copy {env_file}.example to {env_file} and configure your Databricks credentials."
        )
        typer.echo(msg, err=True)
        raise typer.Exit(code=1)


def open_browser(url: str) -> None:
    """Open a URL in the default web browser.

    Args:
        url: URL to open.
    """
    webbrowser.open(url)
