"""Unit tests for sample gold standard dataset.

Tests sample dataset loading, query/document pair parsing, and relevance score
validation. This is important because the sample dataset provides a lightweight
test case for development and CI without requiring large external datasets.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.benchmark.datasets.json_dataset import JSONDataset

SAMPLE_PATH = Path(__file__).resolve().parents[4] / "benchmarks" / "sample" / "sample-gold.json"


class TestSampleDataset:
    """Tests for sample gold standard dataset."""

    @pytest.fixture()
    def dataset(self) -> JSONDataset:
        """Load the sample dataset."""
        return JSONDataset.from_file(SAMPLE_PATH)

    def test_loads_successfully(self, dataset: JSONDataset):
        """Sample dataset loads without errors via JSONDataset."""
        assert dataset is not None

    def test_dataset_name(self, dataset: JSONDataset):
        """Dataset has the expected name."""
        assert dataset.name == "inat-sample-gold"

    def test_dataset_modality(self, dataset: JSONDataset):
        """Dataset modality is text."""
        assert dataset.modality == "text"

    def test_query_count(self, dataset: JSONDataset):
        """Dataset contains exactly 10 queries."""
        assert len(dataset) == 10

    def test_all_queries_have_relevant_docs(self, dataset: JSONDataset):
        """Every query has at least one relevant document."""
        for query in dataset.queries():
            assert len(query.relevant) >= 1, f"Query {query.id} has no relevant documents"

    def test_all_queries_have_graded_relevance(self, dataset: JSONDataset):
        """Every query has graded_relevance data."""
        for query in dataset.queries():
            assert len(query.graded_relevance) >= 1, f"Query {query.id} has no graded relevance"

    def test_graded_relevance_keys_are_subset_of_relevant(self, dataset: JSONDataset):
        """Graded relevance keys are a subset of relevant document IDs."""
        for query in dataset.queries():
            graded_keys = set(query.graded_relevance.keys())
            assert graded_keys <= set(query.relevant), (
                f"Query {query.id}: graded keys {graded_keys} not subset of relevant {query.relevant}"
            )

    def test_query_ids_are_unique(self, dataset: JSONDataset):
        """All query IDs are unique."""
        ids = [q.id for q in dataset.queries()]
        assert len(ids) == len(set(ids))

    def test_queries_have_text(self, dataset: JSONDataset):
        """All queries have non-empty text."""
        for query in dataset.queries():
            assert query.text.strip(), f"Query {query.id} has empty text"

    def test_graded_relevance_values_are_positive(self, dataset: JSONDataset):
        """All graded relevance values are positive integers."""
        for query in dataset.queries():
            for doc_id, grade in query.graded_relevance.items():
                assert grade >= 1, f"Query {query.id}, doc {doc_id}: grade {grade} is not positive"
