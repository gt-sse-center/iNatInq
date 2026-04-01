"""Unit tests for synthetic command group.

Tests synthetic data generation commands (setup, generate, upload) to ensure
correct image generation parameters and S3 upload orchestration. This is
important because synthetic data is used for development, testing, and
benchmarking.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from cli.synthetic import app


class TestSyntheticHelp:
    """Tests for synthetic --help."""

    def test_help_lists_commands(self):
        """Synthetic --help lists all sub-commands."""
        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "generate" in result.output
        assert "upload" in result.output
        assert "setup" in result.output
        assert "clean" in result.output


class TestSyntheticGenerate:
    """Tests for synthetic generate command."""

    @patch("cli.synthetic.ImageGenerator", autospec=False)
    def test_generate_calls_image_generator(self, mock_cls):
        """Synthetic generate creates ImageGenerator and calls .generate()."""
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance

        runner = CliRunner()
        result = runner.invoke(app, ["generate", "--count", "50", "--size", "256"])
        assert result.exit_code == 0
        mock_cls.assert_called_once()
        call_kwargs = mock_cls.call_args
        assert call_kwargs.kwargs["size"] == 256
        mock_instance.generate.assert_called_once_with(count=50)


class TestSyntheticUpload:
    """Tests for synthetic upload command."""

    @patch("cli.synthetic.MinIOUploader", autospec=False)
    def test_upload_calls_minio_uploader(self, mock_cls):
        """Synthetic upload creates MinIOUploader and calls .upload_directory()."""
        mock_instance = MagicMock()
        mock_instance.upload_directory.return_value = (10, 0)
        mock_cls.return_value = mock_instance

        runner = CliRunner()
        result = runner.invoke(app, ["upload", "--endpoint", "http://minio:9000", "--prefix", "test/"])
        assert result.exit_code == 0
        mock_cls.assert_called_once_with(endpoint="http://minio:9000")
        mock_instance.upload_directory.assert_called_once()
        call_kwargs = mock_instance.upload_directory.call_args
        assert call_kwargs.kwargs["prefix"] == "test/"
        assert call_kwargs.kwargs["pattern"] == "*.png"


class TestSyntheticSetup:
    """Tests for synthetic setup command."""

    @patch("cli.synthetic.MinIOUploader", autospec=False)
    @patch("cli.synthetic.ImageGenerator", autospec=False)
    def test_setup_calls_both(self, mock_gen_cls, mock_upload_cls):
        """Synthetic setup creates both ImageGenerator and MinIOUploader."""
        mock_gen = MagicMock()
        mock_gen_cls.return_value = mock_gen

        mock_uploader = MagicMock()
        mock_uploader.upload_directory.return_value = (100, 0)
        mock_upload_cls.return_value = mock_uploader

        runner = CliRunner()
        result = runner.invoke(app, ["setup", "--count", "100"])
        assert result.exit_code == 0
        mock_gen.generate.assert_called_once_with(count=100)
        mock_uploader.upload_directory.assert_called_once()


class TestSyntheticClean:
    """Tests for synthetic clean command."""

    @patch("shutil.rmtree")
    @patch("pathlib.Path.exists", return_value=True)
    def test_clean_removes_directory(self, mock_exists, mock_rmtree):
        """Synthetic clean removes bench/synthetic/data/imgs/."""
        runner = CliRunner()
        result = runner.invoke(app, ["clean"])
        assert result.exit_code == 0
        mock_rmtree.assert_called_once()
