"""Unit tests for JSONDataset loader."""

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError

from core.benchmark.datasets.base import Query
from core.benchmark.datasets.json_dataset import JSONDataset


@pytest.fixture()
def sample_dataset_path(tmp_path: Path) -> Path:
    """Write a valid sample dataset JSON and return its path."""
    data = {
        "name": "test-gold",
        "description": "Test gold standard dataset",
        "modality": "image",
        "queries": [
            {
                "id": "q1",
                "text": "red bird",
                "relevant": ["img1", "img3"],
                "graded_relevance": {"img1": 3, "img3": 1},
            },
            {
                "id": "q2",
                "text": "blue fish",
                "relevant": ["img2"],
            },
        ],
        "metadata": {
            "created_at": "2026-02-09T12:00:00Z",
            "annotators": ["human-1"],
        },
    }
    path = tmp_path / "gold.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class TestJSONDatasetFromFile:
    """Tests for JSONDataset.from_file()."""

    def test_loads_valid_dataset(self, sample_dataset_path: Path):
        """from_file loads and parses a valid JSON dataset."""
        ds = JSONDataset.from_file(sample_dataset_path)

        assert ds.name == "test-gold"
        assert ds.modality == "image"
        assert len(ds) == 2

    def test_queries_returns_iterator(self, sample_dataset_path: Path):
        """queries() returns an Iterator of Query objects."""
        ds = JSONDataset.from_file(sample_dataset_path)

        result = ds.queries()
        assert isinstance(result, Iterator)

        items = list(result)
        assert len(items) == 2
        assert all(isinstance(q, Query) for q in items)

    def test_query_fields_populated(self, sample_dataset_path: Path):
        """Query objects have correct field values."""
        ds = JSONDataset.from_file(sample_dataset_path)
        queries = list(ds.queries())

        q1 = queries[0]
        assert q1.id == "q1"
        assert q1.text == "red bird"
        assert q1.relevant == frozenset({"img1", "img3"})
        assert q1.graded_relevance == {"img1": 3, "img3": 1}

        q2 = queries[1]
        assert q2.id == "q2"
        assert q2.text == "blue fish"
        assert q2.relevant == frozenset({"img2"})
        assert q2.graded_relevance == {}

    def test_file_not_found_raises(self, tmp_path: Path):
        """from_file raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            JSONDataset.from_file(tmp_path / "nonexistent.json")

    def test_invalid_json_raises(self, tmp_path: Path):
        """from_file raises JSONDecodeError for invalid JSON."""
        path = tmp_path / "bad.json"
        path.write_text("not json {{{", encoding="utf-8")

        with pytest.raises(json.JSONDecodeError):
            JSONDataset.from_file(path)

    def test_invalid_schema_raises(self, tmp_path: Path):
        """from_file raises ValidationError for invalid schema."""
        path = tmp_path / "bad_schema.json"
        path.write_text(json.dumps({"name": "x"}), encoding="utf-8")

        with pytest.raises(ValidationError):
            JSONDataset.from_file(path)

    def test_text_modality(self, tmp_path: Path):
        """from_file correctly loads text modality datasets."""
        data = {
            "name": "text-ds",
            "description": "A text dataset",
            "modality": "text",
            "queries": [
                {"id": "q1", "text": "hello world", "relevant": ["doc1"]},
            ],
        }
        path = tmp_path / "text.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        ds = JSONDataset.from_file(path)
        assert ds.modality == "text"

    def test_queries_are_cached(self, sample_dataset_path: Path):
        """Multiple calls to queries() iterate over the same cached data."""
        ds = JSONDataset.from_file(sample_dataset_path)

        first = list(ds.queries())
        second = list(ds.queries())

        assert first == second

    def test_name_and_modality_properties(self, sample_dataset_path: Path):
        """name and modality properties return correct values."""
        ds = JSONDataset.from_file(sample_dataset_path)

        assert ds.name == "test-gold"
        assert ds.modality == "image"
