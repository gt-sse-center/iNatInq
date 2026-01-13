"""Domain models for pipeline operations.

This module defines structured domain model classes to replace tuple returns throughout
the codebase. Using classes instead of tuples provides:
- Type safety and IDE autocomplete
- Self-documenting code (field names vs. positional indices)
- Easy extension with methods or properties
- Better error messages when fields are accessed incorrectly

All classes use `attrs` for concise, correct class definitions.
"""

from typing import Any

import attrs
from qdrant_client.models import PointStruct as QdrantPointStruct


@attrs.define(frozen=False, slots=True)
class VectorPoint:
    """Wrapper around Qdrant's PointStruct for vector database operations.

    This wrapper provides a stable interface that abstracts away the underlying
    Qdrant client implementation, making it easier to swap vector databases
    or add custom behavior.

    Attributes:
        id: Point identifier (str, int, or uuid.UUID).
        vector: Vector embeddings (list[float] or dict[str, list[float]] for named vectors).
        payload: Optional metadata payload (dict[str, Any]).
    """

    id: str | int | Any  # uuid.UUID is also supported
    vector: Any  # Accepts list[float] or dict[str, list[float]] for named vectors
    payload: dict[str, Any] | None = None

    def to_qdrant(self) -> QdrantPointStruct:
        """Convert to Qdrant PointStruct for use with Qdrant client.

        Returns:
            QdrantPointStruct instance.
        """
        return QdrantPointStruct(
            id=self.id,
            vector=self.vector,
            payload=self.payload,
        )

    @classmethod
    def from_qdrant(cls, point: QdrantPointStruct) -> "VectorPoint":
        """Create VectorPoint from Qdrant PointStruct.

        Args:
            point: Qdrant PointStruct instance.

        Returns:
            VectorPoint instance.
        """
        return cls(
            id=point.id,
            vector=point.vector,
            payload=point.payload,
        )


@attrs.define(frozen=True, slots=True)
class SearchResultItem:
    """A single search result item from a vector database.

    Attributes:
        point_id: Point/object ID (UUID string from Qdrant or Weaviate).
        score: Similarity score (0.0 to 1.0, higher is more similar).
        payload: Full payload/metadata dictionary from the vector database.
    """

    point_id: str
    score: float
    payload: dict[str, Any]


@attrs.define(frozen=True, slots=True)
class SearchResults:
    """Complete search results from a vector database query.

    Attributes:
        items: List of search result items, ordered by similarity (highest first).
        total: Total number of results found
            (may be greater than items.length if limit was applied).
    """

    items: list[SearchResultItem]
    total: int
