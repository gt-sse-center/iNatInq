"""Unit tests for smoke command group."""

from __future__ import annotations

from unittest.mock import patch

from typer.testing import CliRunner

from cli.smoke import app


class TestSmokeHelp:
    """Tests for smoke --help."""

    def test_help_lists_commands(self):
        """smoke --help lists all sub-commands."""
        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "health" in result.output
        assert "providers" in result.output
        assert "all" in result.output.lower()


class TestSmokeHealth:
    """Tests for smoke health command."""

    @patch("cli.smoke.run")
    def test_health_delegates_to_script(self, mock_run):
        """smoke health delegates to providers_health.sh."""
        runner = CliRunner()
        result = runner.invoke(app, ["health"])
        assert result.exit_code == 0
        mock_run.assert_called()
        call_args = mock_run.call_args[0][0]
        assert any("providers_health.sh" in str(arg) for arg in call_args)


class TestSmokeProviders:
    """Tests for smoke providers command."""

    @patch("cli.smoke.run")
    def test_providers_delegates_to_script(self, mock_run):
        """smoke providers delegates to smoke_providers.sh."""
        runner = CliRunner()
        result = runner.invoke(app, ["providers"])
        assert result.exit_code == 0
        mock_run.assert_called()
        call_args = mock_run.call_args[0][0]
        assert any("smoke_providers.sh" in str(arg) for arg in call_args)


class TestSmokeAll:
    """Tests for smoke all command."""

    @patch("cli.smoke.run")
    def test_all_runs_health_then_providers(self, mock_run):
        """smoke all runs health then providers in sequence."""
        runner = CliRunner()
        result = runner.invoke(app, ["all"])
        assert result.exit_code == 0
        # Should be called twice (health + providers)
        assert mock_run.call_count == 2
