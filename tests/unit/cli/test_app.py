"""Unit tests for CLI app (top-level commands)."""

from __future__ import annotations

from unittest.mock import patch

from typer.testing import CliRunner

from cli.app import app


class TestHelp:
    """Tests for --help flag."""

    def test_help_shows_all_groups(self):
        """--help lists all command groups."""
        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "docker" in result.output
        assert "dev" in result.output
        assert "test" in result.output
        assert "ui" in result.output
        assert "smoke" in result.output
        assert "synthetic" in result.output
        assert "search" in result.output
        assert "ray" in result.output
        assert "vectordb" in result.output
        assert "databricks" in result.output

    def test_help_shows_top_level_aliases(self):
        """--help shows top-level convenience aliases."""
        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "up" in result.output
        assert "down" in result.output
        assert "status" in result.output


class TestVersion:
    """Tests for --version flag."""

    def test_version_shows_version_string(self):
        """--version prints version and exits."""
        runner = CliRunner()
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "version" in result.output.lower()


class TestTopLevelAliases:
    """Tests for top-level convenience aliases (up, down, status)."""

    @patch("cli.docker.run")
    def test_up_alias_calls_docker_up(self, mock_run):
        """'inq up' is an alias for 'inq docker up'."""
        runner = CliRunner()
        result = runner.invoke(app, ["up"])
        assert result.exit_code == 0
        # Verify docker compose up -d was called
        mock_run.assert_called()
        call_args = mock_run.call_args[0][0]
        assert "docker" in call_args
        assert "compose" in call_args
        assert "up" in call_args

    @patch("cli.docker.run")
    def test_down_alias_calls_docker_down(self, mock_run):
        """'inq down' is an alias for 'inq docker down'."""
        runner = CliRunner()
        result = runner.invoke(app, ["down"])
        assert result.exit_code == 0
        mock_run.assert_called()
        call_args = mock_run.call_args[0][0]
        assert "docker" in call_args
        assert "compose" in call_args
        assert "down" in call_args

    @patch("cli.docker.run")
    @patch("subprocess.run")  # For health checks
    def test_status_alias_calls_docker_health(self, mock_subprocess, mock_run):
        """'inq status' is an alias for 'inq docker health'."""
        runner = CliRunner()
        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        # Health command calls docker compose ps
        mock_run.assert_called()
