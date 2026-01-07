"""Abstract base class for vector database backends."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass
class DataPoint:
    """A single data point in the dataset, which includes the embedding vector and additional metadata."""

    id: int
    vector: Sequence[float]
    metadata: dict[str, object]


@dataclass
class Query:
    """A class encapsulating the query vector and optional filters."""

    vector: Sequence[float]
    filters: object | None = None


@dataclass
class SearchResult:
    """The result of a search query.

    Contains the data point ID and the similarity score.
    """

    id: int
    score: float


class VectorDatabase(ABC):
    """Abstract base class for an adaptor to a vector database.

    This class serves as the base class for wrappers around a vector database client,
    providing a clean and consistent interface which can be used by the benchmarking code.
    """

    @abstractmethod
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        """Constructor for the vector database adaptor.

        Args:
            dataset (HuggingFaceDataset): The dataset which to load to the database.
            metric (str | Metric): The distance/similarity metric to use for the vector database.
            *args (Sequence[object]): Optional positional arguments.
            **kwargs (dict[object, object]): Optional key-word arguments.
        """
        self.dim = 0

    @abstractmethod
    def search(self, q: Query, topk: int) -> Sequence[SearchResult]:
        """Search for top-k nearest neighbors.

        Args:
            q (Query): A single query point.
            topk (int): The number of closest results to return.

        Returns:
            Sequence[SearchResult]: A list of SearchResult objects.
        """
