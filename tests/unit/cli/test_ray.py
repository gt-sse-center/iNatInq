"""Unit tests for ray command group."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from cli.ray import app


class TestRayHelp:
    """Tests for ray --help."""

    def test_help_lists_commands(self):
        """ray --help lists all sub-commands."""
        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "submit" in result.output
        assert "status" in result.output
        assert "logs" in result.output


class TestRaySubmit:
    """Tests for ray submit command."""

    @patch("requests.post")
    def test_submit_posts_to_api(self, mock_post):
        """ray submit posts job to Pipeline API."""
        mock_response = MagicMock()
        mock_response.status_code = 202
        mock_response.json.return_value = {"job_id": "test-job-123"}
        mock_post.return_value = mock_response

        runner = CliRunner()
        result = runner.invoke(app, ["submit", "--prefix", "images/", "--collection", "documents"])
        assert result.exit_code == 0
        mock_post.assert_called_once()
        assert "test-job-123" in result.output

    @patch("requests.post")
    def test_submit_handles_api_error(self, mock_post):
        """ray submit handles API errors gracefully."""
        mock_post.side_effect = Exception("Connection error")

        runner = CliRunner()
        result = runner.invoke(app, ["submit"])
        assert result.exit_code == 1


class TestRayStatus:
    """Tests for ray status command."""

    @patch("requests.get")
    def test_status_gets_from_ray_api(self, mock_get):
        """ray status gets job status from Ray API."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{"job_id": "test-123", "status": "SUCCEEDED"}]
        mock_get.return_value = mock_response

        runner = CliRunner()
        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        mock_get.assert_called_once()
