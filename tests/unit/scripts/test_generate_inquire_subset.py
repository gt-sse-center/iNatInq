"""Unit tests for generate_inquire_subset script."""

from __future__ import annotations

import json
from uuid import NAMESPACE_URL, uuid5

import pytest

# Import the internal functions directly for unit testing
from scripts.generate_inquire_subset import (
    _build_output_dataset,
    _collect_unique_doc_ids,
    _doc_id_to_uuid5,
    _filter_queries,
)


def _make_dataset(queries: list[dict], name: str = "test-dataset") -> dict:
    """Build a minimal dataset dict for testing."""
    return {
        "name": name,
        "description": "Test dataset",
        "modality": "image",
        "queries": queries,
    }


class TestCollectUniqueDocIds:
    """Tests for _collect_unique_doc_ids."""

    def test_collects_all_ids(self):
        """Collects doc IDs from all queries."""
        dataset = _make_dataset(
            [
                {"id": "q1", "text": "bird", "relevant": ["1", "2"]},
                {"id": "q2", "text": "fish", "relevant": ["2", "3"]},
            ]
        )
        result = _collect_unique_doc_ids(dataset)
        assert result == {"1", "2", "3"}

    def test_empty_queries(self):
        """Returns empty set for no queries."""
        dataset = _make_dataset([])
        result = _collect_unique_doc_ids(dataset)
        assert result == set()


class TestDocIdToUuid5:
    """Tests for _doc_id_to_uuid5."""

    def test_deterministic(self):
        """Same input produces same UUID5."""
        assert _doc_id_to_uuid5("123") == _doc_id_to_uuid5("123")

    def test_matches_uuid5(self):
        """Matches stdlib uuid5(NAMESPACE_URL, doc_id)."""
        doc_id = "1316621"
        expected = str(uuid5(NAMESPACE_URL, doc_id))
        assert _doc_id_to_uuid5(doc_id) == expected


class TestFilterQueries:
    """Tests for _filter_queries."""

    def test_full_coverage_keeps_all_relevant(self):
        """Query with all relevant docs present keeps full list."""
        dataset = _make_dataset(
            [
                {"id": "q1", "text": "bird", "relevant": ["1", "2"]},
            ]
        )
        present = {"1", "2", "3"}
        result = _filter_queries(dataset, present)
        assert len(result) == 1
        assert result[0]["relevant"] == ["1", "2"]

    def test_partial_coverage_trims_relevant(self):
        """Query with partial coverage keeps only present docs."""
        dataset = _make_dataset(
            [
                {"id": "q1", "text": "bird", "relevant": ["1", "2", "99"]},
            ]
        )
        present = {"1", "2"}
        result = _filter_queries(dataset, present)
        assert len(result) == 1
        assert result[0]["relevant"] == ["1", "2"]

    def test_no_relevant_present_drops_query(self):
        """Query with zero present relevant docs is dropped."""
        dataset = _make_dataset(
            [
                {"id": "q1", "text": "bird", "relevant": ["99", "100"]},
            ]
        )
        present = {"1", "2"}
        result = _filter_queries(dataset, present)
        assert len(result) == 0

    def test_mixed_queries(self):
        """Keeps all queries that have at least one present relevant doc."""
        dataset = _make_dataset(
            [
                {"id": "q1", "text": "bird", "relevant": ["1"]},
                {"id": "q2", "text": "fish", "relevant": ["1", "99"]},
                {"id": "q3", "text": "tree", "relevant": ["2", "3"]},
                {"id": "q4", "text": "rock", "relevant": ["99"]},
            ]
        )
        present = {"1", "2", "3"}
        result = _filter_queries(dataset, present)
        assert len(result) == 3
        assert [q["id"] for q in result] == ["q1", "q2", "q3"]
        # q2 should have trimmed relevant list
        assert result[1]["relevant"] == ["1"]

    def test_empty_present_set(self):
        """No queries survive if no docs are present."""
        dataset = _make_dataset(
            [
                {"id": "q1", "text": "bird", "relevant": ["1"]},
            ]
        )
        result = _filter_queries(dataset, set())
        assert len(result) == 0

    def test_preserves_order(self):
        """Trimmed relevant list preserves original ordering."""
        dataset = _make_dataset(
            [
                {"id": "q1", "text": "bird", "relevant": ["3", "1", "99", "2"]},
            ]
        )
        present = {"1", "2", "3"}
        result = _filter_queries(dataset, present)
        assert result[0]["relevant"] == ["3", "1", "2"]


class TestBuildOutputDataset:
    """Tests for _build_output_dataset."""

    def test_output_has_subset_name(self):
        """Output name is '<original>-subset'."""
        source = _make_dataset([{"id": "q1", "text": "t", "relevant": ["1"]}])
        filtered = [{"id": "q1", "text": "t", "relevant": ["1"]}]
        output = _build_output_dataset(
            source,
            filtered,
            collection="test-col",
            indexed_count=5,
            total_doc_ids=10,
            present_count=5,
        )
        assert output["name"] == "test-dataset-subset"

    def test_output_preserves_modality(self):
        """Output preserves original modality."""
        source = _make_dataset([])
        output = _build_output_dataset(
            source,
            [],
            collection="col",
            indexed_count=0,
            total_doc_ids=0,
            present_count=0,
        )
        assert output["modality"] == "image"

    def test_output_contains_metadata(self):
        """Output metadata includes provenance fields."""
        source = _make_dataset(
            [{"id": "q1", "text": "t", "relevant": ["1"]}],
            name="inquire-val",
        )
        output = _build_output_dataset(
            source,
            [{"id": "q1", "text": "t", "relevant": ["1"]}],
            collection="demo-1M",
            indexed_count=100,
            total_doc_ids=200,
            present_count=100,
        )
        meta = output["metadata"]
        assert meta["source_dataset"] == "inquire-val"
        assert meta["qdrant_collection"] == "demo-1M"
        assert meta["indexed_count"] == 100
        assert meta["total_doc_ids"] == 200
        assert meta["present_doc_ids"] == 100
        assert meta["query_coverage"] == "1/1"

    def test_output_queries_match_filtered(self):
        """Output queries are exactly the filtered queries."""
        source = _make_dataset(
            [
                {"id": "q1", "text": "t", "relevant": ["1"]},
                {"id": "q2", "text": "t2", "relevant": ["2"]},
            ]
        )
        filtered = [{"id": "q1", "text": "t", "relevant": ["1"]}]
        output = _build_output_dataset(
            source,
            filtered,
            collection="col",
            indexed_count=1,
            total_doc_ids=2,
            present_count=1,
        )
        assert len(output["queries"]) == 1
        assert output["queries"][0]["id"] == "q1"

    def test_output_is_valid_json(self):
        """Output can be serialized to JSON without error."""
        source = _make_dataset([{"id": "q1", "text": "t", "relevant": ["1"]}])
        output = _build_output_dataset(
            source,
            [{"id": "q1", "text": "t", "relevant": ["1"]}],
            collection="col",
            indexed_count=1,
            total_doc_ids=1,
            present_count=1,
        )
        serialized = json.dumps(output, indent=2)
        assert json.loads(serialized) == output
