"""Base classes for benchmark datasets.

This module provides the foundational abstractions for benchmark datasets:
- Query: An attrs data class holding a single benchmark query with relevance judgments
- Dataset: An abstract base class defining the dataset interface

Example:
    ```python
    from core.benchmark.datasets.base import Query, Dataset

    class MyDataset(Dataset):
        name = "my-dataset"
        modality = "text"

        def __init__(self, queries):
            self._queries = queries

        def queries(self):
            return iter(self._queries)

        def __len__(self):
            return len(self._queries)

    q = Query(id="q1", text="red bird", relevant={"img1", "img3"})
    ds = MyDataset([q])
    ```
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Literal

import attrs


@attrs.define(frozen=True, slots=True)
class Query:
    """A single benchmark query with relevance judgments.

    Attributes:
        id: Unique query identifier.
        text: Query text (e.g., "red bird").
        relevant: Set of relevant document IDs for binary relevance.
        graded_relevance: Mapping of document IDs to integer relevance grades.
            Higher grades indicate more relevant documents. Empty dict by default.

    Example:
        >>> q = Query(
        ...     id="q1",
        ...     text="red bird",
        ...     relevant={"img1", "img3"},
        ...     graded_relevance={"img1": 3, "img3": 1},
        ... )
        >>> "img1" in q.relevant
        True
        >>> q.graded_relevance["img1"]
        3
    """

    id: str
    text: str
    relevant: frozenset[str] = attrs.field(converter=frozenset)
    graded_relevance: dict[str, int] = attrs.Factory(dict)


class Dataset(ABC):
    """Abstract base class for benchmark datasets.

    Subclasses must define:
        - name: A unique identifier for the dataset
        - modality: The data modality ("text" or "image")
        - queries(): Iterator over Query objects
        - __len__(): Number of queries in the dataset

    Attributes:
        name: Unique dataset identifier (property).
        modality: Data modality, either "text" or "image" (property).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the dataset name."""
        ...

    @property
    @abstractmethod
    def modality(self) -> Literal["text", "image"]:
        """Return the data modality ("text" or "image")."""
        ...

    @abstractmethod
    def queries(self) -> Iterator[Query]:
        """Iterate over queries in the dataset.

        Returns:
            Iterator of Query objects.
        """
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of queries in the dataset."""
        ...
