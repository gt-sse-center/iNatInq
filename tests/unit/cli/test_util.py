"""Unit tests for CLI utilities.

Tests shared CLI utility functions (command building, validation, environment
detection) to ensure correct argument construction and error handling. This is
important because utilities are used across all command groups and bugs here
cascade to multiple commands.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer

from cli._util import (
    DEFAULT_S3_BUCKET,
    REPO_ROOT,
    docker_compose,
    ensure_databricks_env,
    open_browser,
    run,
    run_interactive,
)


class TestConstants:
    """Tests for utility constants."""

    def test_repo_root_is_project_root(self):
        """REPO_ROOT points to the project root directory."""
        assert REPO_ROOT.exists()
        assert (REPO_ROOT / "pyproject.toml").exists()

    def test_default_s3_bucket_defined(self):
        """DEFAULT_S3_BUCKET has a value."""
        assert DEFAULT_S3_BUCKET == "pipeline"


class TestRun:
    """Tests for run() subprocess wrapper."""

    @patch("subprocess.run")
    def test_run_passes_cwd_repo_root(self, mock_subprocess):
        """run() always passes cwd=REPO_ROOT."""
        mock_subprocess.return_value = MagicMock(returncode=0)
        run(["echo", "test"])
        mock_subprocess.assert_called_once()
        assert mock_subprocess.call_args[1]["cwd"] == REPO_ROOT

    @patch("subprocess.run")
    def test_run_no_capture_streams_output(self, mock_subprocess):
        """run() with capture=False does not capture output."""
        mock_subprocess.return_value = MagicMock(returncode=0)
        run(["echo", "test"], capture=False)
        call_kwargs = mock_subprocess.call_args[1]
        assert "capture_output" not in call_kwargs or not call_kwargs.get("capture_output")

    @patch("subprocess.run")
    def test_run_capture_returns_output(self, mock_subprocess):
        """run() with capture=True captures and returns output."""
        mock_result = MagicMock(returncode=0, stdout="test output")
        mock_subprocess.return_value = mock_result
        result = run(["echo", "test"], capture=True)
        assert result.stdout == "test output"
        assert mock_subprocess.call_args[1]["capture_output"] is True

    @patch("subprocess.run")
    def test_run_merges_env_with_os_environ(self, mock_subprocess):
        """run() merges env dict with os.environ so PATH is preserved."""
        mock_subprocess.return_value = MagicMock(returncode=0)
        run(["echo", "test"], env={"KEY": "value"})
        passed_env = mock_subprocess.call_args[1]["env"]
        # Custom var is present
        assert passed_env["KEY"] == "value"
        # System PATH is preserved (merged from os.environ)
        assert "PATH" in passed_env

    @patch("subprocess.run")
    def test_run_no_env_passes_none(self, mock_subprocess):
        """run() without env= passes env=None (inherits parent environment)."""
        mock_subprocess.return_value = MagicMock(returncode=0)
        run(["echo", "test"])
        assert mock_subprocess.call_args[1]["env"] is None

    @patch("subprocess.run")
    def test_run_check_false_ignores_errors(self, mock_subprocess):
        """run() with check=False doesn't raise on non-zero exit."""
        mock_subprocess.return_value = MagicMock(returncode=1)
        run(["false"], check=False)  # Should not raise

    @patch("subprocess.run")
    def test_run_check_true_raises_on_error(self, mock_subprocess):
        """run() with check=True raises typer.Exit on non-zero exit."""
        mock_subprocess.side_effect = subprocess.CalledProcessError(1, ["false"], stderr="error")
        with pytest.raises(typer.Exit) as exc_info:
            run(["false"], check=True)
        assert exc_info.value.exit_code == 1


class TestRunInteractive:
    """Tests for run_interactive() subprocess wrapper."""

    @patch("subprocess.run")
    def test_run_interactive_passes_cwd_repo_root(self, mock_subprocess):
        """run_interactive() always passes cwd=REPO_ROOT."""
        mock_subprocess.return_value = MagicMock(returncode=0)
        run_interactive(["bash"])
        mock_subprocess.assert_called_once()
        assert mock_subprocess.call_args[1]["cwd"] == REPO_ROOT

    @patch("subprocess.run")
    @patch("sys.stdin")
    @patch("sys.stdout")
    @patch("sys.stderr")
    def test_run_interactive_inherits_stdio(self, mock_stderr, mock_stdout, mock_stdin, mock_subprocess):
        """run_interactive() explicitly passes stdin/stdout/stderr for TTY."""
        mock_subprocess.return_value = MagicMock(returncode=0)
        run_interactive(["bash"])
        call_kwargs = mock_subprocess.call_args[1]
        assert "stdin" in call_kwargs
        assert "stdout" in call_kwargs
        assert "stderr" in call_kwargs


class TestDockerCompose:
    """Tests for docker_compose() command builder."""

    def test_docker_compose_builds_correct_command(self):
        """docker_compose() builds the correct docker compose command."""
        cmd = docker_compose("up", "-d")
        assert cmd[0] == "docker"
        assert cmd[1] == "compose"
        assert "-f" in cmd
        assert "zarf/compose/dev/docker-compose.yaml" in cmd
        assert "up" in cmd
        assert "-d" in cmd


class TestEnsureDatabricksEnv:
    """Tests for ensure_databricks_env() validation."""

    def test_ensure_databricks_env_exists(self, tmp_path):
        """ensure_databricks_env() passes when file exists."""
        env_file = tmp_path / ".env.local"
        env_file.write_text("TEST=value")
        ensure_databricks_env(str(env_file))  # Should not raise

    def test_ensure_databricks_env_missing_raises(self):
        """ensure_databricks_env() exits when file is missing."""
        with pytest.raises(typer.Exit) as exc_info:
            ensure_databricks_env("/nonexistent/.env.local")
        assert exc_info.value.exit_code == 1


class TestOpenBrowser:
    """Tests for open_browser() wrapper."""

    @patch("webbrowser.open")
    def test_open_browser_calls_webbrowser(self, mock_open):
        """open_browser() delegates to webbrowser.open()."""
        open_browser("http://localhost:8000")
        mock_open.assert_called_once_with("http://localhost:8000")
