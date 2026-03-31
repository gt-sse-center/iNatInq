"""Unit tests for INQUIRE gold standard datasets (val + test splits).

Tests INQUIRE biodiversity search benchmark dataset loading, query parsing, and
relevance judgment validation for both validation and test splits. This is
important because INQUIRE provides the authoritative benchmark for biodiversity
search quality evaluation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.benchmark.datasets.json_dataset import JSONDataset

_INQUIRE_DIR = Path(__file__).resolve().parents[4] / "bench" / "datasets" / "inquire"
VAL_PATH = _INQUIRE_DIR / "inquire-val.json"
TEST_PATH = _INQUIRE_DIR / "inquire-test.json"


class TestInquireValDataset:
    """Tests for INQUIRE validation split (50 queries)."""

    @pytest.fixture()
    def dataset(self) -> JSONDataset:
        """Load the INQUIRE val dataset."""
        return JSONDataset.from_file(VAL_PATH)

    def test_loads_successfully(self, dataset: JSONDataset):
        """Val dataset loads without errors via JSONDataset."""
        assert dataset is not None

    def test_dataset_name(self, dataset: JSONDataset):
        """Dataset has the expected name."""
        assert dataset.name == "inquire-val"

    def test_dataset_modality(self, dataset: JSONDataset):
        """Dataset modality is image (text-to-image retrieval)."""
        assert dataset.modality == "image"

    def test_query_count(self, dataset: JSONDataset):
        """Val split contains exactly 50 queries."""
        assert len(dataset) == 50

    def test_all_queries_have_relevant(self, dataset: JSONDataset):
        """Every query has at least one relevant document."""
        for query in dataset.queries():
            assert len(query.relevant) >= 1, f"Query {query.id} has no relevant documents"

    def test_query_ids_are_unique(self, dataset: JSONDataset):
        """All query IDs are unique."""
        ids = [q.id for q in dataset.queries()]
        assert len(ids) == len(set(ids))

    def test_queries_have_text(self, dataset: JSONDataset):
        """All queries have non-empty text."""
        for query in dataset.queries():
            assert query.text.strip(), f"Query {query.id} has empty text"

    def test_relevant_ids_are_nonempty_strings(self, dataset: JSONDataset):
        """All relevant document IDs are non-empty strings."""
        for query in dataset.queries():
            for doc_id in query.relevant:
                assert isinstance(doc_id, str), f"Query {query.id}: doc_id {doc_id!r} is not str"
                assert doc_id.strip(), f"Query {query.id}: doc_id is empty"


class TestInquireTestDataset:
    """Tests for INQUIRE test split (200 queries)."""

    @pytest.fixture()
    def dataset(self) -> JSONDataset:
        """Load the INQUIRE test dataset."""
        return JSONDataset.from_file(TEST_PATH)

    def test_loads_successfully(self, dataset: JSONDataset):
        """Test dataset loads without errors via JSONDataset."""
        assert dataset is not None

    def test_dataset_name(self, dataset: JSONDataset):
        """Dataset has the expected name."""
        assert dataset.name == "inquire-test"

    def test_dataset_modality(self, dataset: JSONDataset):
        """Dataset modality is image (text-to-image retrieval)."""
        assert dataset.modality == "image"

    def test_query_count(self, dataset: JSONDataset):
        """Test split contains exactly 200 queries."""
        assert len(dataset) == 200

    def test_all_queries_have_relevant(self, dataset: JSONDataset):
        """Every query has at least one relevant document."""
        for query in dataset.queries():
            assert len(query.relevant) >= 1, f"Query {query.id} has no relevant documents"

    def test_query_ids_are_unique(self, dataset: JSONDataset):
        """All query IDs are unique."""
        ids = [q.id for q in dataset.queries()]
        assert len(ids) == len(set(ids))

    def test_queries_have_text(self, dataset: JSONDataset):
        """All queries have non-empty text."""
        for query in dataset.queries():
            assert query.text.strip(), f"Query {query.id} has empty text"

    def test_relevant_ids_are_nonempty_strings(self, dataset: JSONDataset):
        """All relevant document IDs are non-empty strings."""
        for query in dataset.queries():
            for doc_id in query.relevant:
                assert isinstance(doc_id, str), f"Query {query.id}: doc_id {doc_id!r} is not str"
                assert doc_id.strip(), f"Query {query.id}: doc_id is empty"
