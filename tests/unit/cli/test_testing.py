"""Unit tests for test command group (named test_testing to avoid collision)."""

from __future__ import annotations

from unittest.mock import patch

from typer.testing import CliRunner

from cli.test import app


class TestTestHelp:
    """Tests for test --help."""

    def test_help_lists_commands(self):
        """test --help lists all sub-commands."""
        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "unit" in result.output
        assert "integration" in result.output
        assert "all" in result.output.lower()
        assert "cov" in result.output


class TestTestUnit:
    """Tests for test unit command."""

    @patch("cli.test.run")
    def test_unit_runs_pytest(self, mock_run):
        """test unit runs pytest on tests/unit/."""
        runner = CliRunner()
        result = runner.invoke(app, ["unit"])
        assert result.exit_code == 0
        mock_run.assert_called()
        call_args = mock_run.call_args[0][0]
        assert "uv" in call_args
        assert "run" in call_args
        assert "pytest" in call_args
        assert "tests/unit/" in call_args

    @patch("cli.test.run")
    def test_unit_passes_extra_args(self, mock_run):
        """test unit -- -k test_name passes extra args to pytest."""
        runner = CliRunner()
        result = runner.invoke(app, ["unit", "--", "-k", "test_name"])
        assert result.exit_code == 0
        mock_run.assert_called()
        call_args = mock_run.call_args[0][0]
        assert "-k" in call_args
        assert "test_name" in call_args


class TestTestIntegration:
    """Tests for test integration command."""

    @patch("cli.test.run")
    def test_integration_runs_pytest(self, mock_run):
        """test integration runs pytest on tests/integration/."""
        runner = CliRunner()
        result = runner.invoke(app, ["integration"])
        assert result.exit_code == 0
        mock_run.assert_called()
        call_args = mock_run.call_args[0][0]
        assert "pytest" in call_args
        assert "tests/integration/" in call_args

    @patch("cli.test.run")
    def test_integration_parallel_adds_n_auto(self, mock_run):
        """test integration --parallel adds -n auto."""
        runner = CliRunner()
        result = runner.invoke(app, ["integration", "--parallel"])
        assert result.exit_code == 0
        call_args = mock_run.call_args[0][0]
        assert "-n" in call_args
        assert "auto" in call_args


class TestTestCov:
    """Tests for test cov command."""

    @patch("cli.test.run")
    def test_cov_runs_with_coverage(self, mock_run):
        """test cov runs pytest with --cov flags."""
        runner = CliRunner()
        result = runner.invoke(app, ["cov"])
        assert result.exit_code == 0
        call_args = mock_run.call_args[0][0]
        assert "--cov=src" in call_args
        assert "--cov-report=html" in call_args

    @patch("cli.test.run")
    def test_cov_all_runs_all_tests(self, mock_run):
        """test cov --all runs coverage on all tests."""
        runner = CliRunner()
        result = runner.invoke(app, ["cov", "--all"])
        assert result.exit_code == 0
        call_args = mock_run.call_args[0][0]
        assert "tests/" in call_args
