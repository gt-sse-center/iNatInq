"""Unit tests for benchmark dataset base classes.

Tests the abstract BenchmarkDataset interface and base implementations to ensure
correct data loading contracts and schema validation. This is important because
datasets provide the ground truth for all benchmark measurements.
"""

from collections.abc import Iterator
from typing import Literal

import attrs
import pytest

from core.benchmark.datasets.base import Dataset, Query


class TestQuery:
    """Tests for Query attrs class."""

    def test_query_holds_required_fields(self):
        """Query stores id, text, and relevant doc IDs."""
        q = Query(id="q1", text="red bird", relevant={"img1", "img3"})

        assert q.id == "q1"
        assert q.text == "red bird"
        assert q.relevant == frozenset({"img1", "img3"})

    def test_query_graded_relevance_defaults_to_empty(self):
        """graded_relevance defaults to an empty dict."""
        q = Query(id="q1", text="red bird", relevant={"img1"})

        assert q.graded_relevance == {}

    def test_query_with_graded_relevance(self):
        """Query accepts graded relevance mapping."""
        graded = {"img1": 3, "img3": 1}
        q = Query(
            id="q1",
            text="red bird",
            relevant={"img1", "img3"},
            graded_relevance=graded,
        )

        assert q.graded_relevance == {"img1": 3, "img3": 1}
        assert q.graded_relevance["img1"] == 3

    def test_query_relevant_accepts_set(self):
        """relevant field accepts a plain set and converts to frozenset."""
        q = Query(id="q1", text="test", relevant={"a", "b"})

        assert isinstance(q.relevant, frozenset)
        assert q.relevant == frozenset({"a", "b"})

    def test_query_relevant_accepts_list(self):
        """relevant field accepts a list and converts to frozenset."""
        q = Query(id="q1", text="test", relevant=["a", "b", "c"])

        assert isinstance(q.relevant, frozenset)
        assert q.relevant == frozenset({"a", "b", "c"})

    def test_query_is_frozen(self):
        """Query instances are immutable."""
        q = Query(id="q1", text="test", relevant={"a"})

        with pytest.raises(attrs.exceptions.FrozenInstanceError):
            q.id = "q2"

    def test_query_equality(self):
        """Two queries with the same fields are equal."""
        q1 = Query(id="q1", text="test", relevant={"a"})
        q2 = Query(id="q1", text="test", relevant={"a"})

        assert q1 == q2

    def test_query_empty_relevant_set(self):
        """Query can have an empty relevant set."""
        q = Query(id="q1", text="test", relevant=set())

        assert q.relevant == frozenset()
        assert len(q.relevant) == 0


class TestDataset:
    """Tests for Dataset ABC."""

    def test_dataset_is_abstract(self):
        """Dataset cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Dataset()

    def test_concrete_dataset_can_be_instantiated(self):
        """A concrete Dataset with all methods implemented can be instantiated."""

        class ConcreteDataset(Dataset):
            @property
            def name(self) -> str:
                return "test-dataset"

            @property
            def modality(self) -> Literal["text", "image"]:
                return "text"

            def queries(self) -> Iterator[Query]:
                return iter([])

            def __len__(self) -> int:
                return 0

        ds = ConcreteDataset()
        assert ds is not None

    def test_incomplete_dataset_raises(self):
        """A Dataset missing abstract methods cannot be instantiated."""

        class IncompleteDataset(Dataset):
            @property
            def name(self) -> str:
                return "incomplete"

        with pytest.raises(TypeError):
            IncompleteDataset()

    def test_name_property(self):
        """name property returns the dataset name."""

        class NamedDataset(Dataset):
            @property
            def name(self) -> str:
                return "my-dataset"

            @property
            def modality(self) -> Literal["text", "image"]:
                return "text"

            def queries(self) -> Iterator[Query]:
                return iter([])

            def __len__(self) -> int:
                return 0

        ds = NamedDataset()
        assert ds.name == "my-dataset"

    def test_modality_returns_text(self):
        """modality property can return 'text'."""

        class TextDataset(Dataset):
            @property
            def name(self) -> str:
                return "text-ds"

            @property
            def modality(self) -> Literal["text", "image"]:
                return "text"

            def queries(self) -> Iterator[Query]:
                return iter([])

            def __len__(self) -> int:
                return 0

        ds = TextDataset()
        assert ds.modality == "text"

    def test_modality_returns_image(self):
        """modality property can return 'image'."""

        class ImageDataset(Dataset):
            @property
            def name(self) -> str:
                return "image-ds"

            @property
            def modality(self) -> Literal["text", "image"]:
                return "image"

            def queries(self) -> Iterator[Query]:
                return iter([])

            def __len__(self) -> int:
                return 0

        ds = ImageDataset()
        assert ds.modality == "image"

    def test_queries_returns_iterator(self):
        """queries() returns an Iterator of Query objects."""
        sample_queries = [
            Query(id="q1", text="red bird", relevant={"img1"}),
            Query(id="q2", text="blue fish", relevant={"img2", "img3"}),
        ]

        class IterDataset(Dataset):
            @property
            def name(self) -> str:
                return "iter-ds"

            @property
            def modality(self) -> Literal["text", "image"]:
                return "text"

            def queries(self) -> Iterator[Query]:
                return iter(sample_queries)

            def __len__(self) -> int:
                return len(sample_queries)

        ds = IterDataset()
        result = list(ds.queries())

        assert len(result) == 2
        assert result[0].id == "q1"
        assert result[1].id == "q2"
        assert isinstance(ds.queries(), Iterator)

    def test_len_returns_query_count(self):
        """__len__() returns the number of queries."""
        sample_queries = [
            Query(id="q1", text="a", relevant={"d1"}),
            Query(id="q2", text="b", relevant={"d2"}),
            Query(id="q3", text="c", relevant={"d3"}),
        ]

        class SizedDataset(Dataset):
            @property
            def name(self) -> str:
                return "sized-ds"

            @property
            def modality(self) -> Literal["text", "image"]:
                return "text"

            def queries(self) -> Iterator[Query]:
                return iter(sample_queries)

            def __len__(self) -> int:
                return len(sample_queries)

        ds = SizedDataset()
        assert len(ds) == 3

    def test_len_returns_zero_for_empty_dataset(self):
        """__len__() returns 0 for an empty dataset."""

        class EmptyDataset(Dataset):
            @property
            def name(self) -> str:
                return "empty-ds"

            @property
            def modality(self) -> Literal["text", "image"]:
                return "text"

            def queries(self) -> Iterator[Query]:
                return iter([])

            def __len__(self) -> int:
                return 0

        ds = EmptyDataset()
        assert len(ds) == 0
