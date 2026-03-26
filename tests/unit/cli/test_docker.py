"""Unit tests for docker command group."""

from __future__ import annotations

from unittest.mock import patch

from typer.testing import CliRunner

from cli.docker import app


class TestDockerHelp:
    """Tests for docker --help."""

    def test_help_lists_commands(self):
        """docker --help lists all sub-commands."""
        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "up" in result.output
        assert "down" in result.output
        assert "build" in result.output
        assert "logs" in result.output
        assert "health" in result.output


class TestDockerUp:
    """Tests for docker up command."""

    @patch("cli.docker.run")
    def test_up_runs_docker_compose(self, mock_run):
        """docker up runs docker compose up -d."""
        runner = CliRunner()
        result = runner.invoke(app, ["up"])
        assert result.exit_code == 0
        mock_run.assert_called()
        call_args = mock_run.call_args[0][0]
        assert "docker" in call_args
        assert "compose" in call_args
        assert "up" in call_args
        assert "-d" in call_args


class TestDockerDown:
    """Tests for docker down command."""

    @patch("cli.docker.run")
    def test_down_runs_docker_compose(self, mock_run):
        """docker down runs docker compose down."""
        runner = CliRunner()
        result = runner.invoke(app, ["down"])
        assert result.exit_code == 0
        mock_run.assert_called()
        call_args = mock_run.call_args[0][0]
        assert "docker" in call_args
        assert "compose" in call_args
        assert "down" in call_args


class TestDockerLog:
    """Tests for docker log command (service-specific logs)."""

    @patch("cli.docker.run")
    def test_log_pipeline_tails_logs(self, mock_run):
        """docker log pipeline tails pipeline service logs."""
        runner = CliRunner()
        result = runner.invoke(app, ["log", "pipeline"])
        assert result.exit_code == 0
        mock_run.assert_called()
        call_args = mock_run.call_args[0][0]
        assert "logs" in call_args
        assert "-f" in call_args
        assert "pipeline" in call_args


class TestDockerShell:
    """Tests for docker shell command."""

    @patch("cli.docker.run_interactive")
    def test_shell_pipeline_opens_bash(self, mock_run_interactive):
        """docker shell pipeline opens bash in pipeline container."""
        runner = CliRunner()
        result = runner.invoke(app, ["shell", "pipeline"])
        assert result.exit_code == 0
        mock_run_interactive.assert_called()
        call_args = mock_run_interactive.call_args[0][0]
        assert "exec" in call_args
        assert "pipeline" in call_args
        assert "/bin/bash" in call_args


class TestDockerHealth:
    """Tests for docker health command."""

    @patch("cli.docker.run")
    @patch("subprocess.run")
    def test_health_runs_checks(self, mock_subprocess, mock_run):
        """docker health runs compose ps and health checks."""
        runner = CliRunner()
        result = runner.invoke(app, ["health"])
        assert result.exit_code == 0
        # Should call docker compose ps
        mock_run.assert_called()
