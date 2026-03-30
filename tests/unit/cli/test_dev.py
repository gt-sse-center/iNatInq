"""Unit tests for dev command group."""

from __future__ import annotations

from unittest.mock import patch

from typer.testing import CliRunner

from cli.dev import app


class TestDevHelp:
    """Tests for dev --help."""

    def test_help_lists_commands(self):
        """dev --help lists all sub-commands."""
        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "lint" in result.output
        assert "format" in result.output.lower()
        assert "typecheck" in result.output
        assert "serve" in result.output


class TestDevLint:
    """Tests for dev lint command."""

    @patch("cli.dev.run")
    def test_lint_runs_ruff_check(self, mock_run):
        """dev lint runs ruff check."""
        runner = CliRunner()
        result = runner.invoke(app, ["lint"])
        assert result.exit_code == 0
        mock_run.assert_called()
        call_args = mock_run.call_args[0][0]
        assert "uv" in call_args
        assert "run" in call_args
        assert "ruff" in call_args
        assert "check" in call_args


class TestDevFormat:
    """Tests for dev format command."""

    @patch("cli.dev.run")
    def test_format_runs_ruff_format(self, mock_run):
        """dev format runs ruff format then ruff check --fix."""
        runner = CliRunner()
        result = runner.invoke(app, ["format"])
        assert result.exit_code == 0
        # Should be called twice (format + check --fix)
        assert mock_run.call_count == 2


class TestDevTypecheck:
    """Tests for dev typecheck command."""

    @patch("cli.dev.run")
    def test_typecheck_runs_mypy(self, mock_run):
        """dev typecheck runs mypy."""
        runner = CliRunner()
        result = runner.invoke(app, ["typecheck"])
        assert result.exit_code == 0
        mock_run.assert_called()
        call_args = mock_run.call_args[0][0]
        assert "uv" in call_args
        assert "run" in call_args
        assert "mypy" in call_args


class TestDevServe:
    """Tests for dev serve command."""

    @patch("cli.dev.run")
    def test_serve_runs_uvicorn(self, mock_run):
        """dev serve runs uvicorn with reload."""
        runner = CliRunner()
        result = runner.invoke(app, ["serve"])
        assert result.exit_code == 0
        mock_run.assert_called()
        call_args = mock_run.call_args[0][0]
        assert "uv" in call_args
        assert "run" in call_args
        assert "uvicorn" in call_args
        assert "--reload" in call_args
