"""Unit tests for ui command group."""

from __future__ import annotations

from unittest.mock import patch

from typer.testing import CliRunner

from cli.ui import app


class TestUiHelp:
    """Tests for ui --help."""

    def test_help_lists_commands(self):
        """ui --help lists all sub-commands."""
        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "all" in result.output.lower()
        assert "pipeline" in result.output
        assert "minio" in result.output
        assert "qdrant" in result.output
        assert "ray" in result.output


class TestUiAll:
    """Tests for ui all command."""

    @patch("cli.ui.open_browser")
    def test_all_opens_all_dashboards(self, mock_open):
        """ui all opens all 4 dashboards."""
        runner = CliRunner()
        result = runner.invoke(app, ["all"])
        assert result.exit_code == 0
        # Should open 4 URLs
        assert mock_open.call_count == 4


class TestUiPipeline:
    """Tests for ui pipeline command."""

    @patch("cli.ui.open_browser")
    def test_pipeline_opens_api_docs(self, mock_open):
        """ui pipeline opens Pipeline API docs."""
        runner = CliRunner()
        result = runner.invoke(app, ["pipeline"])
        assert result.exit_code == 0
        mock_open.assert_called_once()
        # Check URL contains expected port
        call_arg = mock_open.call_args[0][0]
        assert "8000" in call_arg


class TestUiMinio:
    """Tests for ui minio command."""

    @patch("cli.ui.open_browser")
    def test_minio_opens_console_and_prints_creds(self, mock_open):
        """ui minio opens MinIO console and prints credentials."""
        runner = CliRunner()
        result = runner.invoke(app, ["minio"])
        assert result.exit_code == 0
        mock_open.assert_called_once()
        # Should print login info
        assert "minioadmin" in result.output.lower()
