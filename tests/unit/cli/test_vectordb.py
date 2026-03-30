"""Unit tests for vectordb command group."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from cli.vectordb import app


class TestVectordbHelp:
    """Tests for vectordb --help."""

    def test_help_lists_commands(self):
        """vectordb --help lists all sub-commands."""
        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "count" in result.output
        assert "clear" in result.output
        assert "s3-count" in result.output
        assert "s3-clear" in result.output
        assert "e2e" in result.output


class TestVectordbCount:
    """Tests for vectordb count command."""

    @patch("requests.get")
    def test_count_gets_collection_info(self, mock_get):
        """vectordb count gets collection info from Qdrant."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": {"points_count": 100}}
        mock_get.return_value = mock_response

        runner = CliRunner()
        result = runner.invoke(app, ["count", "--collection", "documents"])
        assert result.exit_code == 0
        mock_get.assert_called_once()
        assert "100" in result.output


class TestVectordbClear:
    """Tests for vectordb clear command."""

    @patch("requests.delete")
    def test_clear_deletes_collection(self, mock_delete):
        """vectordb clear deletes Qdrant collection."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_delete.return_value = mock_response

        runner = CliRunner()
        result = runner.invoke(app, ["clear", "--collection", "documents"])
        assert result.exit_code == 0
        mock_delete.assert_called_once()


class TestVectordbS3Count:
    """Tests for vectordb s3-count command."""

    @patch("cli.vectordb.run")
    def test_s3_count_uses_minio_mc(self, mock_run):
        """vectordb s3-count uses MinIO mc ls."""
        mock_result = MagicMock(stdout="file1.png\nfile2.png\nfile3.png\n")
        mock_run.return_value = mock_result

        runner = CliRunner()
        result = runner.invoke(app, ["s3-count", "--bucket", "pipeline", "--prefix", "images/"])
        assert result.exit_code == 0
        mock_run.assert_called()
        call_args = mock_run.call_args[0][0]
        assert "mc" in call_args
        assert "ls" in call_args
        assert "3" in result.output


class TestVectordbE2E:
    """Tests for vectordb e2e command."""

    @patch("cli.vectordb.clear")
    @patch("cli.synthetic.setup")
    @patch("cli.ray.submit")
    @patch("cli.vectordb.count")
    @patch("cli.search.images")
    @patch("time.sleep")
    def test_e2e_orchestrates_pipeline(
        self, mock_sleep, mock_search, mock_count, mock_ray_submit, mock_synth_setup, mock_clear
    ):
        """vectordb e2e orchestrates full pipeline."""
        runner = CliRunner()
        result = runner.invoke(app, ["e2e", "--count", "10", "--wait", "5"])
        assert result.exit_code == 0
        # Verify orchestration order
        mock_clear.assert_called_once()
        mock_synth_setup.assert_called_once()
        mock_ray_submit.assert_called_once()
        mock_sleep.assert_called_once_with(5)
        mock_count.assert_called_once()
        mock_search.assert_called_once()
