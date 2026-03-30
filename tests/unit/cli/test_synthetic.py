"""Unit tests for synthetic command group."""

from __future__ import annotations

from unittest.mock import patch

from typer.testing import CliRunner

from cli.synthetic import app


class TestSyntheticHelp:
    """Tests for synthetic --help."""

    def test_help_lists_commands(self):
        """synthetic --help lists all sub-commands."""
        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "generate" in result.output
        assert "upload" in result.output
        assert "setup" in result.output
        assert "clean" in result.output


class TestSyntheticGenerate:
    """Tests for synthetic generate command."""

    @patch("cli.synthetic.run")
    def test_generate_delegates_to_script(self, mock_run):
        """synthetic generate delegates to synthetic_data.py generate-images."""
        runner = CliRunner()
        result = runner.invoke(app, ["generate", "--count", "50", "--size", "256"])
        assert result.exit_code == 0
        mock_run.assert_called()
        call_args = mock_run.call_args[0][0]
        assert "syntheticdata/synthetic_data.py" in call_args
        assert "generate-images" in call_args
        assert "--count" in call_args
        assert "50" in call_args


class TestSyntheticUpload:
    """Tests for synthetic upload command."""

    @patch("cli.synthetic.run")
    def test_upload_delegates_to_script(self, mock_run):
        """synthetic upload delegates to synthetic_data.py upload-images."""
        runner = CliRunner()
        result = runner.invoke(app, ["upload"])
        assert result.exit_code == 0
        mock_run.assert_called()
        call_args = mock_run.call_args[0][0]
        assert "upload-images" in call_args


class TestSyntheticSetup:
    """Tests for synthetic setup command."""

    @patch("cli.synthetic.run")
    def test_setup_delegates_to_script(self, mock_run):
        """synthetic setup delegates to synthetic_data.py setup-images."""
        runner = CliRunner()
        result = runner.invoke(app, ["setup", "--count", "100"])
        assert result.exit_code == 0
        mock_run.assert_called()
        call_args = mock_run.call_args[0][0]
        assert "setup-images" in call_args


class TestSyntheticClean:
    """Tests for synthetic clean command."""

    @patch("shutil.rmtree")
    @patch("pathlib.Path.exists", return_value=True)
    def test_clean_removes_directory(self, mock_exists, mock_rmtree):
        """synthetic clean removes syntheticdata/data/imgs/."""
        runner = CliRunner()
        result = runner.invoke(app, ["clean"])
        assert result.exit_code == 0
        mock_rmtree.assert_called_once()
