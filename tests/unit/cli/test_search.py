"""Unit tests for search command group."""
# ruff: noqa: S106, S105

from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from cli.search import app


class TestSearchHelp:
    """Tests for search --help."""

    def test_help_lists_commands(self):
        """Search --help lists all sub-commands."""
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
        """Search images calls the Pipeline API."""
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
        """Search images handles API errors gracefully."""
        mock_get.side_effect = Exception("Connection error")

        runner = CliRunner()
        result = runner.invoke(app, ["images"])
        assert result.exit_code == 1


class TestSearchDemo:
    """Tests for search demo command."""

    @patch("boto3.client")
    @patch("requests.get")
    def test_demo_searches_and_displays_results(self, mock_get, mock_boto3_client):
        """Search demo calls API and displays formatted results."""
        # Arrange
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": [{"s3_key": "images/test.jpg", "score": 0.95}]}
        mock_get.return_value = mock_response

        mock_s3 = MagicMock()
        mock_s3.generate_presigned_url.return_value = "http://presigned.com/test.jpg"
        mock_boto3_client.return_value = mock_s3

        runner = CliRunner()

        # Act
        result = runner.invoke(app, ["demo", "--query", "red circle", "--limit", "1"])

        # Assert
        assert result.exit_code == 0
        assert "test.jpg" in result.output
        assert "presigned.com" in result.output  # URLs shown by default


class TestSearchImagesHelper:
    """Tests for _search_images helper function."""

    @patch("requests.get")
    def test_search_images_calls_api_with_correct_params(self, mock_get):
        """_search_images calls API with q= parameter (not query=)."""
        from cli.search import _search_images

        # Arrange
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": [{"s3_key": "test.png", "score": 0.95}]}
        mock_get.return_value = mock_response

        # Act
        results = _search_images("red circle", 5, "http://localhost:8000")

        # Assert
        mock_get.assert_called_once_with(
            "http://localhost:8000/search/images",
            params={"q": "red circle", "limit": 5},
            timeout=30,
        )
        assert results == [{"s3_key": "test.png", "score": 0.95}]

    @patch("requests.get")
    def test_search_images_raises_on_api_error(self, mock_get):
        """_search_images propagates RequestException on API failure."""
        from cli.search import _search_images
        import requests
        import pytest

        # Arrange
        mock_get.side_effect = requests.RequestException("Connection error")

        # Act & Assert
        with pytest.raises(requests.RequestException, match="Connection error"):
            _search_images("query", 5, "http://localhost:8000")


class TestMakeS3Client:
    """Tests for _make_s3_client helper function."""

    @patch("boto3.client")
    def test_make_s3_client_without_s3v4(self, mock_boto3_client):
        """_make_s3_client creates client with path-style addressing only."""
        from cli.search import _make_s3_client
        from config import MinIOConfig

        # Arrange
        config = MinIOConfig(
            endpoint_url="http://localhost:9000",
            access_key_id="test-key",
            secret_access_key="test-secret",
            bucket="pipeline",
            region="us-east-1",
            path_style=True,
        )

        # Act
        _make_s3_client(config, use_s3v4=False)

        # Assert
        call_args = mock_boto3_client.call_args
        assert call_args[0][0] == "s3"
        kwargs = call_args[1]
        assert kwargs["endpoint_url"] == "http://localhost:9000"
        assert kwargs["aws_access_key_id"] == "test-key"
        assert kwargs["aws_secret_access_key"] == "test-secret"
        assert kwargs["region_name"] == "us-east-1"
        # Should have config with path-style addressing
        assert "config" in kwargs
        config_obj = kwargs["config"]
        assert config_obj.s3 == {"addressing_style": "path"}

    @patch("boto3.client")
    def test_make_s3_client_with_s3v4(self, mock_boto3_client):
        """_make_s3_client creates client with s3v4 + path-style addressing."""
        from cli.search import _make_s3_client
        from config import MinIOConfig

        # Arrange
        config = MinIOConfig(
            endpoint_url="http://localhost:9000",
            access_key_id="test-key",
            secret_access_key="test-secret",
            bucket="pipeline",
            region="us-east-1",
            path_style=True,
        )

        # Act
        _make_s3_client(config, use_s3v4=True)

        # Assert
        call_args = mock_boto3_client.call_args
        kwargs = call_args[1]
        # Should have config with both s3v4 and path-style
        config_obj = kwargs["config"]
        assert config_obj.signature_version == "s3v4"
        assert config_obj.s3 == {"addressing_style": "path"}


class TestGetPresignedUrl:
    """Tests for _get_presigned_url helper function."""

    def test_get_presigned_url_generates_url(self):
        """_get_presigned_url generates presigned URL using s3_client."""
        from cli.search import _get_presigned_url
        from config import MinIOConfig
        from unittest.mock import MagicMock

        # Arrange
        mock_s3_client = MagicMock()
        mock_s3_client.generate_presigned_url.return_value = "http://presigned-url.com/image.jpg"
        config = MinIOConfig(
            endpoint_url="http://localhost:9000",
            access_key_id="test",
            secret_access_key="test",
            bucket="pipeline",
        )

        # Act
        url = _get_presigned_url("images/test.jpg", mock_s3_client, config)

        # Assert
        assert url == "http://presigned-url.com/image.jpg"
        mock_s3_client.generate_presigned_url.assert_called_once_with(
            "get_object",
            Params={"Bucket": "pipeline", "Key": "images/test.jpg"},
            ExpiresIn=3600,
        )


class TestDownloadImage:
    """Tests for _download_image helper function."""

    def test_download_image_downloads_to_output_dir(self):
        """_download_image downloads S3 object to local file."""
        from cli.search import _download_image
        from config import MinIOConfig
        from pathlib import Path
        from unittest.mock import MagicMock

        # Arrange
        mock_s3_client = MagicMock()
        config = MinIOConfig(
            endpoint_url="http://localhost:9000",
            access_key_id="test",
            secret_access_key="test",
            bucket="pipeline",
        )
        output_dir = Path("/tmp/test-output")

        # Act
        result = _download_image("images/test.jpg", output_dir, mock_s3_client, config)

        # Assert
        assert result == output_dir / "test.jpg"
        mock_s3_client.download_file.assert_called_once_with(
            "pipeline",
            "images/test.jpg",
            str(output_dir / "test.jpg"),
        )


class TestDisplayResults:
    """Tests for _display_results helper function."""

    def test_display_results_without_urls(self, capsys):
        """_display_results prints formatted table without URLs."""
        from cli.search import _display_results
        from config import MinIOConfig
        from unittest.mock import MagicMock

        # Arrange
        results = [
            {"s3_key": "images/test1.jpg", "score": 0.95},
            {"s3_key": "images/test2.jpg", "score": 0.87},
        ]
        mock_s3_client = MagicMock()
        config = MinIOConfig(
            endpoint_url="http://localhost:9000",
            access_key_id="test",
            secret_access_key="test",
            bucket="pipeline",
        )

        # Act
        _display_results(results, show_urls=False, s3_client=mock_s3_client, config=config)

        # Assert
        captured = capsys.readouterr()
        assert "test1.jpg" in captured.out
        assert "test2.jpg" in captured.out
        assert "0.9500" in captured.out
        assert "0.8700" in captured.out
        assert "URL:" not in captured.out

    def test_display_results_with_urls(self, capsys):
        """_display_results prints formatted table with presigned URLs."""
        from cli.search import _display_results
        from config import MinIOConfig
        from unittest.mock import MagicMock

        # Arrange
        results = [
            {"s3_key": "images/test1.jpg", "score": 0.95},
        ]
        mock_s3_client = MagicMock()
        mock_s3_client.generate_presigned_url.return_value = "http://presigned-url.com/test1.jpg"
        config = MinIOConfig(
            endpoint_url="http://localhost:9000",
            access_key_id="test",
            secret_access_key="test",
            bucket="pipeline",
        )

        # Act
        _display_results(results, show_urls=True, s3_client=mock_s3_client, config=config)

        # Assert
        captured = capsys.readouterr()
        assert "test1.jpg" in captured.out
        assert "URL: http://presigned-url.com/test1.jpg" in captured.out


class TestSearchDownload:
    """Tests for search download command."""

    @patch("subprocess.run")
    @patch("boto3.client")
    @patch("requests.get")
    def test_download_downloads_search_results(self, mock_get, mock_boto3_client, mock_subprocess):
        """Search download downloads images to output directory."""
        # Arrange
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"s3_key": "images/test1.jpg", "score": 0.95},
                {"s3_key": "images/test2.jpg", "score": 0.85},
            ]
        }
        mock_get.return_value = mock_response

        mock_s3 = MagicMock()
        mock_boto3_client.return_value = mock_s3

        runner = CliRunner()

        # Act
        with runner.isolated_filesystem():
            result = runner.invoke(app, ["download", "--query", "test", "--output", "./downloads"])

            # Assert
            assert result.exit_code == 0
            assert "Downloaded:" in result.output
            assert mock_s3.download_file.call_count == 2


class TestSearchOpen:
    """Tests for search open command."""

    @patch("cli.search.open_browser")
    @patch("boto3.client")
    @patch("requests.get")
    def test_open_opens_top_result_in_browser(self, mock_get, mock_boto3_client, mock_open_browser):
        """Search open generates presigned URL and opens browser."""
        # Arrange
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": [{"s3_key": "images/test.jpg", "score": 0.95}]}
        mock_get.return_value = mock_response

        mock_s3 = MagicMock()
        mock_s3.generate_presigned_url.return_value = "http://presigned.com/test.jpg"
        mock_boto3_client.return_value = mock_s3

        runner = CliRunner()

        # Act
        result = runner.invoke(app, ["open", "--query", "test"])

        # Assert
        assert result.exit_code == 0
        assert "Opening top result" in result.output
        mock_open_browser.assert_called_once_with("http://presigned.com/test.jpg")


class TestS3ErrorHandling:
    """Tests for S3/botocore error handling in search commands."""

    @patch("boto3.client")
    @patch("requests.get")
    def test_demo_handles_s3_endpoint_error(self, mock_get, mock_boto3_client):
        """Demo command handles EndpointConnectionError gracefully."""
        from botocore.exceptions import EndpointConnectionError

        # Arrange
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": [{"s3_key": "images/test.jpg", "score": 0.95}]}
        mock_get.return_value = mock_response

        mock_boto3_client.side_effect = EndpointConnectionError(endpoint_url="http://localhost:9000")

        runner = CliRunner()

        # Act
        result = runner.invoke(app, ["demo", "--query", "test"])

        # Assert
        assert result.exit_code == 1
        assert "S3 error" in result.output

    @patch("boto3.client")
    @patch("requests.get")
    def test_download_handles_s3_client_error(self, mock_get, mock_boto3_client):
        """Download command handles ClientError gracefully."""
        from botocore.exceptions import ClientError

        # Arrange
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": [{"s3_key": "images/test.jpg", "score": 0.95}]}
        mock_get.return_value = mock_response

        mock_s3 = MagicMock()
        mock_s3.download_file.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "Access Denied"}}, "GetObject"
        )
        mock_boto3_client.return_value = mock_s3

        runner = CliRunner()

        # Act
        with runner.isolated_filesystem():
            result = runner.invoke(app, ["download", "--query", "test", "--output", "./downloads"])

        # Assert
        assert result.exit_code == 1
        assert "S3 error" in result.output
