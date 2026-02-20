"""Unit tests for benchmark CLI."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from core.benchmark.cli import app


@pytest.fixture()
def valid_dataset(tmp_path):
    """Create a valid dataset JSON file."""
    data = {
        "name": "test-dataset",
        "description": "A test dataset",
        "modality": "text",
        "queries": [
            {
                "id": "q1",
                "text": "red bird",
                "relevant": ["d1", "d2"],
            },
            {
                "id": "q2",
                "text": "blue fish",
                "relevant": ["d3"],
            },
        ],
    }
    path = tmp_path / "test-dataset.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class TestValidateCommand:
    """Tests for 'validate' command."""

    def test_valid_dataset(self, valid_dataset):
        """validate reports success for a valid dataset."""
        runner = CliRunner()
        result = runner.invoke(app, ["validate", str(valid_dataset)])
        assert result.exit_code == 0
        assert "test-dataset" in result.output
        assert "valid" in result.output.lower()
        assert "Queries:  2" in result.output

    def test_invalid_json(self, tmp_path):
        """validate reports error for invalid JSON."""
        runner = CliRunner()
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not json", encoding="utf-8")
        result = runner.invoke(app, ["validate", str(bad_file)])
        assert result.exit_code != 0

    def test_invalid_schema(self, tmp_path):
        """validate reports error for valid JSON but invalid schema."""
        runner = CliRunner()
        bad_file = tmp_path / "bad-schema.json"
        bad_file.write_text(json.dumps({"name": "test"}), encoding="utf-8")
        result = runner.invoke(app, ["validate", str(bad_file)])
        assert result.exit_code != 0

    def test_nonexistent_file(self):
        """validate reports error for non-existent file."""
        runner = CliRunner()
        result = runner.invoke(app, ["validate", "/nonexistent/path.json"])
        assert result.exit_code != 0


class TestMetricsCommand:
    """Tests for 'metrics' command."""

    def test_lists_metrics(self):
        """metrics lists available metrics."""
        runner = CliRunner()
        result = runner.invoke(app, ["metrics"])
        assert result.exit_code == 0
        assert "precision@k" in result.output
        assert "recall@k" in result.output
        assert "map" in result.output
        assert "ndcg" in result.output
        assert "mrr" in result.output

    def test_output_contains_header(self):
        """metrics output contains 'Available metrics' header."""
        runner = CliRunner()
        result = runner.invoke(app, ["metrics"])
        assert "Available metrics" in result.output


class TestCompareCommand:
    """Tests for 'compare' command."""

    def test_missing_dataset(self):
        """compare errors when --dataset is not provided."""
        runner = CliRunner()
        result = runner.invoke(app, ["compare", "--provider", "qdrant"])
        assert result.exit_code != 0
        assert "dataset" in result.output.lower()

    def test_missing_provider(self, valid_dataset):
        """compare errors when --provider is not provided."""
        runner = CliRunner()
        result = runner.invoke(app, ["compare", "--dataset", str(valid_dataset)])
        assert result.exit_code != 0
        assert "provider" in result.output.lower()

    @patch("core.benchmark.provider_factory.resolve_search_pipeline")
    def test_valid_args_runs_benchmark(self, mock_resolve, valid_dataset):
        """compare with valid args resolves provider and runs benchmark."""
        mock_provider = AsyncMock()
        mock_pipeline = AsyncMock()
        from core.benchmark.search_pipeline import SearchPipelineResult
        from core.models import SearchResults

        mock_pipeline.search.return_value = SearchPipelineResult(
            doc_ids=["d1"], raw_results=SearchResults(items=[], total=0)
        )
        mock_resolve.return_value = (mock_provider, mock_pipeline)

        runner = CliRunner()
        result = runner.invoke(app, ["compare", "--dataset", str(valid_dataset), "--provider", "qdrant"])
        assert result.exit_code == 0
        assert "complete" in result.output.lower()
        mock_resolve.assert_called()

    @patch("core.benchmark.provider_factory.resolve_search_pipeline")
    def test_provider_resolution_error(self, mock_resolve, valid_dataset):
        """compare exits with error when provider resolution fails."""
        mock_resolve.side_effect = ValueError("Unknown provider")

        runner = CliRunner()
        result = runner.invoke(app, ["compare", "--dataset", str(valid_dataset), "--provider", "bad"])
        assert result.exit_code != 0
        assert "error" in result.output.lower()


class TestRunCommand:
    """Tests for 'run' command."""

    def test_missing_dataset(self):
        """run errors when --dataset is not provided."""
        runner = CliRunner()
        result = runner.invoke(app, ["run", "--provider", "qdrant"])
        assert result.exit_code != 0
        assert "dataset" in result.output.lower()

    def test_missing_provider(self, valid_dataset):
        """run errors when --provider is not provided."""
        runner = CliRunner()
        result = runner.invoke(app, ["run", "--dataset", str(valid_dataset)])
        assert result.exit_code != 0
        assert "provider" in result.output.lower()

    @patch("core.benchmark.provider_factory.resolve_search_pipeline")
    def test_valid_args_runs_benchmark(self, mock_resolve, valid_dataset):
        """run with valid args resolves provider and runs benchmark."""
        mock_provider = AsyncMock()
        mock_pipeline = AsyncMock()
        from core.benchmark.search_pipeline import SearchPipelineResult
        from core.models import SearchResults

        mock_pipeline.search.return_value = SearchPipelineResult(
            doc_ids=["d1"], raw_results=SearchResults(items=[], total=0)
        )
        mock_resolve.return_value = (mock_provider, mock_pipeline)

        runner = CliRunner()
        result = runner.invoke(app, ["run", "--dataset", str(valid_dataset), "--provider", "qdrant"])
        assert result.exit_code == 0
        assert "complete" in result.output.lower()
        mock_resolve.assert_called()

    @patch("core.benchmark.provider_factory.resolve_search_pipeline")
    def test_collection_option(self, mock_resolve, valid_dataset):
        """run passes --collection to resolve_search_pipeline."""
        mock_provider = AsyncMock()
        mock_pipeline = AsyncMock()
        from core.benchmark.search_pipeline import SearchPipelineResult
        from core.models import SearchResults

        mock_pipeline.search.return_value = SearchPipelineResult(
            doc_ids=["d1"], raw_results=SearchResults(items=[], total=0)
        )
        mock_resolve.return_value = (mock_provider, mock_pipeline)

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "run",
                "--dataset",
                str(valid_dataset),
                "--provider",
                "qdrant",
                "--collection",
                "my-collection",
            ],
        )
        assert result.exit_code == 0
        mock_resolve.assert_called_once_with("qdrant", collection="my-collection")

    @patch("core.benchmark.provider_factory.resolve_search_pipeline")
    def test_custom_limit_and_warmup(self, mock_resolve, valid_dataset):
        """run accepts --limit and --warmup options."""
        mock_provider = AsyncMock()
        mock_pipeline = AsyncMock()
        from core.benchmark.search_pipeline import SearchPipelineResult
        from core.models import SearchResults

        mock_pipeline.search.return_value = SearchPipelineResult(
            doc_ids=["d1"], raw_results=SearchResults(items=[], total=0)
        )
        mock_resolve.return_value = (mock_provider, mock_pipeline)

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "run",
                "--dataset",
                str(valid_dataset),
                "--provider",
                "qdrant",
                "--limit",
                "20",
                "--warmup",
                "10",
            ],
        )
        assert result.exit_code == 0

    @patch("core.benchmark.provider_factory.resolve_search_pipeline")
    def test_provider_resolution_error(self, mock_resolve, valid_dataset):
        """run exits with error when provider resolution fails."""
        mock_resolve.side_effect = ValueError("clip_url is required")

        runner = CliRunner()
        result = runner.invoke(app, ["run", "--dataset", str(valid_dataset), "--provider", "qdrant"])
        assert result.exit_code != 0
        assert "error" in result.output.lower()
