"""Unit tests for databricks command group.

Tests Databricks cluster lifecycle management commands (build, up, down) to
ensure correct Databricks CLI invocation and JSON payload construction. This
is important because cluster management errors can leave expensive resources
running or cause job submission failures.
"""

from __future__ import annotations

from unittest.mock import patch

from typer.testing import CliRunner

from cli.databricks import app


class TestDatabricksHelp:
    """Tests for databricks --help."""

    def test_help_lists_commands(self):
        """databricks --help lists all sub-commands."""
        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "build" in result.output
        assert "up" in result.output
        assert "down" in result.output
        assert "cdc-notebooks" in result.output
        assert "configure-s3a" in result.output


class TestDatabricksBuild:
    """Tests for databricks build command."""

    @patch("cli.databricks.ensure_databricks_env")
    @patch("cli.databricks.run")
    def test_build_checks_env_and_runs_script(self, mock_run, mock_ensure):
        """databricks build checks env file then runs build script."""
        runner = CliRunner()
        result = runner.invoke(app, ["build"])
        assert result.exit_code == 0
        mock_ensure.assert_called_once()
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert any("azure-databricks-build.py" in str(arg) for arg in call_args)


class TestDatabricksUp:
    """Tests for databricks up command."""

    @patch("cli.databricks.ensure_databricks_env")
    @patch("cli.databricks.run")
    def test_up_checks_env_and_runs_script(self, mock_run, mock_ensure):
        """databricks up checks env file then runs up script."""
        runner = CliRunner()
        result = runner.invoke(app, ["up"])
        assert result.exit_code == 0
        mock_ensure.assert_called_once()
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert any("azure-databricks-up.py" in str(arg) for arg in call_args)


class TestDatabricksDown:
    """Tests for databricks down command."""

    @patch("cli.databricks.ensure_databricks_env")
    @patch("cli.databricks.run")
    def test_down_checks_env_and_runs_script(self, mock_run, mock_ensure):
        """databricks down checks env file then runs down script."""
        runner = CliRunner()
        result = runner.invoke(app, ["down"])
        assert result.exit_code == 0
        mock_ensure.assert_called_once()
        mock_run.assert_called_once()


class TestDatabricksCdcNotebooks:
    """Tests for databricks cdc-notebooks command."""

    @patch("cli.databricks.ensure_databricks_env")
    @patch("cli.databricks.run")
    def test_cdc_notebooks_runs_with_flags(self, mock_run, mock_ensure):
        """databricks cdc-notebooks runs script with notebook flags."""
        runner = CliRunner()
        result = runner.invoke(app, ["cdc-notebooks"])
        assert result.exit_code == 0
        mock_ensure.assert_called_once()
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert any("azure-databricks-cdc-notebooks.py" in str(arg) for arg in call_args)
        assert "--upload-notebooks" in call_args
        assert "--run-notebooks" in call_args
