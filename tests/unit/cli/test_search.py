"""Unit tests for search command group."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from cli.search import app


class TestSearchHelp:
    """Tests for search --help."""

    def test_help_lists_commands(self):
        """search --help lists all sub-commands."""
        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "images" in result.output
        assert "demo" in result.output
        assert "download" in result.output


class TestSearchImages:
    """Tests for search images command."""

    @patch("requests.get")
    def test_images_calls_api(self, mock_get):
        """search images calls the Pipeline API."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": [{"s3_key": "test.png", "score": 0.95}]}
        mock_get.return_value = mock_response

        runner = CliRunner()
        result = runner.invoke(app, ["images", "--query", "red circle", "--limit", "3"])
        assert result.exit_code == 0
        mock_get.assert_called_once()
        assert "test.png" in result.output

    @patch("requests.get")
    def test_images_handles_api_error(self, mock_get):
        """search images handles API errors gracefully."""
        mock_get.side_effect = Exception("Connection error")

        runner = CliRunner()
        result = runner.invoke(app, ["images"])
        assert result.exit_code == 1


class TestSearchDemo:
    """Tests for search demo command."""

    @patch("cli.search.run")
    def test_demo_delegates_to_script(self, mock_run):
        """search demo delegates to image_search_demo.py."""
        runner = CliRunner()
        result = runner.invoke(app, ["demo", "--query", "red circle"])
        assert result.exit_code == 0
        mock_run.assert_called()
        call_args = mock_run.call_args[0][0]
        assert "scripts/image_search_demo.py" in call_args
        assert "red circle" in call_args
