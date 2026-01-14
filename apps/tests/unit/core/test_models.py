"""Unit tests for core.models module.

This file tests domain model classes that provide structured data types for pipeline
operations, replacing tuple returns with type-safe, self-documenting classes.

# Test Coverage

The tests cover:
  - VectorPoint: Initialization, Qdrant conversion, immutability
  - SearchResultItem: Creation, attribute access, frozen state
  - SearchResults: Result aggregation, item lists, total counts
  - Model Integration: Round-trip conversions, payload handling

# Test Structure

Tests use pytest class-based organization with direct model instantiation.
No mocking required as models are pure data classes using attrs.

# Running Tests

Run with: pytest tests/unit/core/test_models.py
"""

import attrs.exceptions
import pytest
from qdrant_client.models import PointStruct as QdrantPointStruct

from src.core.models import SearchResultItem, SearchResults, VectorPoint

# =============================================================================
# VectorPoint Tests
# =============================================================================


class TestVectorPoint:
    """Test suite for VectorPoint model."""

    def test_creates_point_with_required_fields(self) -> None:
        """Test that VectorPoint is created with id and vector.

        **Why this test is important:**
          - VectorPoint is core to vector DB operations
          - Validates basic initialization
          - Ensures required fields are accepted
          - Critical for all vector operations

        **What it tests:**
          - Point is created with id and vector
          - Fields are accessible as attributes
          - Optional payload defaults to None
        """
        point = VectorPoint(id="test-id", vector=[0.1, 0.2, 0.3])

        assert point.id == "test-id"
        assert point.vector == [0.1, 0.2, 0.3]
        assert point.payload is None

    def test_creates_point_with_payload(self) -> None:
        """Test that VectorPoint accepts optional payload.

        **Why this test is important:**
          - Payload stores document metadata
          - Essential for rich search results
          - Validates optional field handling
          - Critical for metadata storage

        **What it tests:**
          - Point is created with payload
          - Payload is stored correctly
          - Arbitrary metadata is supported
        """
        payload = {"text": "test document", "source": "api"}
        point = VectorPoint(id="test-id", vector=[0.1, 0.2, 0.3], payload=payload)

        assert point.payload == payload
        assert point.payload["text"] == "test document"
        assert point.payload["source"] == "api"

    def test_creates_point_with_int_id(self) -> None:
        """Test that VectorPoint accepts integer IDs.

        **Why this test is important:**
          - Different vector DBs use different ID types
          - Integer IDs are common in some systems
          - Validates flexible ID handling
          - Critical for compatibility

        **What it tests:**
          - Point accepts int ID
          - ID is stored without conversion
        """
        point = VectorPoint(id=12345, vector=[0.1, 0.2, 0.3])

        assert point.id == 12345
        assert isinstance(point.id, int)

    def test_creates_point_with_named_vectors(self) -> None:
        """Test that VectorPoint supports named vectors.

        **Why this test is important:**
          - Named vectors enable multi-vector search
          - Qdrant supports named vector fields
          - Validates dict[str, list[float]] handling
          - Critical for advanced use cases

        **What it tests:**
          - Point accepts dict of vectors
          - Named vectors are stored correctly
        """
        named_vectors = {
            "title": [0.1, 0.2],
            "content": [0.3, 0.4, 0.5],
        }
        point = VectorPoint(id="test-id", vector=named_vectors)

        assert point.vector == named_vectors
        assert point.vector["title"] == [0.1, 0.2]
        assert point.vector["content"] == [0.3, 0.4, 0.5]

    def test_to_qdrant_converts_correctly(self) -> None:
        """Test that to_qdrant() converts to Qdrant PointStruct.

        **Why this test is important:**
          - Qdrant client requires PointStruct objects
          - Conversion enables vector DB operations
          - Validates serialization
          - Critical for Qdrant integration

        **What it tests:**
          - to_qdrant() returns PointStruct
          - All fields are preserved
          - Payload is passed through correctly
        """
        payload = {"text": "document"}
        point = VectorPoint(id="test-id", vector=[0.1, 0.2], payload=payload)

        qdrant_point = point.to_qdrant()

        assert isinstance(qdrant_point, QdrantPointStruct)
        assert qdrant_point.id == "test-id"
        assert qdrant_point.vector == [0.1, 0.2]
        assert qdrant_point.payload == payload

    def test_from_qdrant_converts_correctly(self) -> None:
        """Test that from_qdrant() creates VectorPoint from PointStruct.

        **Why this test is important:**
          - Enables reading from Qdrant
          - Validates deserialization
          - Round-trip compatibility
          - Critical for Qdrant integration

        **What it tests:**
          - from_qdrant() returns VectorPoint
          - All fields are preserved
          - Payload is extracted correctly
        """
        payload = {"text": "document"}
        qdrant_point = QdrantPointStruct(
            id="test-id",
            vector=[0.1, 0.2],
            payload=payload,
        )

        point = VectorPoint.from_qdrant(qdrant_point)

        assert isinstance(point, VectorPoint)
        assert point.id == "test-id"
        assert point.vector == [0.1, 0.2]
        assert point.payload == payload

    def test_round_trip_conversion(self) -> None:
        """Test that round-trip conversion preserves data.

        **Why this test is important:**
          - Ensures lossless conversion
          - Validates bidirectional compatibility
          - Critical for data integrity
          - Prevents serialization bugs

        **What it tests:**
          - VectorPoint -> PointStruct -> VectorPoint
          - All fields preserved through round trip
          - Payload survives conversion
        """
        original = VectorPoint(
            id="test-id",
            vector=[0.1, 0.2, 0.3],
            payload={"text": "doc", "score": 0.95},
        )

        qdrant_point = original.to_qdrant()
        restored = VectorPoint.from_qdrant(qdrant_point)

        assert restored.id == original.id
        assert restored.vector == original.vector
        assert restored.payload == original.payload

    def test_point_is_mutable(self) -> None:
        """Test that VectorPoint is mutable (frozen=False).

        **Why this test is important:**
          - VectorPoint uses frozen=False for flexibility
          - Allows modification after creation
          - Validates attrs configuration
          - Critical for processing pipelines

        **What it tests:**
          - Fields can be modified after creation
          - No FrozenInstanceError is raised
        """
        point = VectorPoint(id="test-id", vector=[0.1, 0.2])

        # Should not raise FrozenInstanceError
        point.payload = {"new": "data"}

        assert point.payload == {"new": "data"}


# =============================================================================
# SearchResultItem Tests
# =============================================================================


class TestSearchResultItem:
    """Test suite for SearchResultItem model."""

    def test_creates_result_item(self) -> None:
        """Test that SearchResultItem is created with required fields.

        **Why this test is important:**
          - SearchResultItem represents single search results
          - Validates basic initialization
          - Ensures required fields are enforced
          - Critical for search functionality

        **What it tests:**
          - Item is created with point_id, score, and payload
          - All fields are accessible as attributes
          - Score is stored as float
        """
        item = SearchResultItem(
            point_id="uuid-123",
            score=0.95,
            payload={"text": "relevant doc"},
        )

        assert item.point_id == "uuid-123"
        assert item.score == 0.95
        assert item.payload == {"text": "relevant doc"}

    def test_score_as_float(self) -> None:
        """Test that score is stored as float.

        **Why this test is important:**
          - Scores are similarity values (0.0 to 1.0)
          - Float precision is required
          - Validates type handling
          - Critical for ranking

        **What it tests:**
          - Score accepts float values
          - Score maintains precision
        """
        item = SearchResultItem(
            point_id="test",
            score=0.123456789,
            payload={},
        )

        assert isinstance(item.score, float)
        assert item.score == 0.123456789

    def test_payload_with_nested_data(self) -> None:
        """Test that payload supports nested structures.

        **Why this test is important:**
          - Real-world payloads contain complex data
          - Nested objects are common in documents
          - Validates flexible payload handling
          - Critical for rich metadata

        **What it tests:**
          - Payload accepts nested dicts
          - Nested data is accessible
          - Lists and dicts work correctly
        """
        payload = {
            "text": "document",
            "metadata": {
                "author": "Jane",
                "tags": ["ml", "ai"],
            },
        }
        item = SearchResultItem(point_id="test", score=0.9, payload=payload)

        assert item.payload["metadata"]["author"] == "Jane"
        assert item.payload["metadata"]["tags"] == ["ml", "ai"]

    def test_result_item_is_frozen(self) -> None:
        """Test that SearchResultItem is immutable (frozen=True).

        **Why this test is important:**
          - Immutability prevents accidental modification
          - Search results should be read-only
          - Validates attrs frozen configuration
          - Critical for data integrity

        **What it tests:**
          - Attributes cannot be modified
          - FrozenInstanceError is raised on modification
        """
        item = SearchResultItem(point_id="test", score=0.9, payload={})

        with pytest.raises(attrs.exceptions.FrozenInstanceError):
            item.score = 0.8


# =============================================================================
# SearchResults Tests
# =============================================================================


class TestSearchResults:
    """Test suite for SearchResults model."""

    def test_creates_results_with_items(self) -> None:
        """Test that SearchResults aggregates search result items.

        **Why this test is important:**
          - SearchResults is the top-level search response
          - Validates list aggregation
          - Ensures item + total tracking
          - Critical for search API

        **What it tests:**
          - Results created with list of items
          - Total count is stored
          - Items are accessible
        """
        items = [
            SearchResultItem("id1", 0.95, {"text": "doc1"}),
            SearchResultItem("id2", 0.85, {"text": "doc2"}),
        ]
        results = SearchResults(items=items, total=2)

        assert len(results.items) == 2
        assert results.total == 2
        assert results.items[0].score == 0.95
        assert results.items[1].score == 0.85

    def test_creates_results_with_empty_list(self) -> None:
        """Test that SearchResults handles no results.

        **Why this test is important:**
          - Empty results are common (no matches)
          - Validates edge case handling
          - Prevents null pointer errors
          - Critical for robustness

        **What it tests:**
          - Empty items list is accepted
          - Total can be 0
          - No errors on empty results
        """
        results = SearchResults(items=[], total=0)

        assert results.items == []
        assert results.total == 0
        assert len(results.items) == 0

    def test_total_greater_than_items(self) -> None:
        """Test that total can exceed items length (pagination).

        **Why this test is important:**
          - Limit parameter restricts returned items
          - Total shows all matches found
          - Validates pagination support
          - Critical for large result sets

        **What it tests:**
          - Total can be greater than items.length
          - Models supports pagination use case
        """
        items = [
            SearchResultItem("id1", 0.95, {"text": "doc1"}),
            SearchResultItem("id2", 0.85, {"text": "doc2"}),
        ]
        results = SearchResults(items=items, total=100)

        assert len(results.items) == 2
        assert results.total == 100

    def test_items_ordered_by_score(self) -> None:
        """Test that items can be ordered by score.

        **Why this test is important:**
          - Search results are typically sorted by relevance
          - High scores should come first
          - Validates ordering preservation
          - Critical for search quality

        **What it tests:**
          - Items maintain order
          - Scores are descending (highest first)
        """
        items = [
            SearchResultItem("id1", 0.95, {}),
            SearchResultItem("id2", 0.90, {}),
            SearchResultItem("id3", 0.85, {}),
        ]
        results = SearchResults(items=items, total=3)

        assert results.items[0].score == 0.95
        assert results.items[1].score == 0.90
        assert results.items[2].score == 0.85
        # Verify descending order
        for i in range(len(results.items) - 1):
            assert results.items[i].score >= results.items[i + 1].score

    def test_results_is_frozen(self) -> None:
        """Test that SearchResults is immutable (frozen=True).

        **Why this test is important:**
          - Immutability prevents accidental modification
          - Search results should be read-only
          - Validates attrs frozen configuration
          - Critical for data integrity

        **What it tests:**
          - Attributes cannot be modified
          - FrozenInstanceError is raised on modification
        """
        results = SearchResults(items=[], total=0)

        with pytest.raises(attrs.exceptions.FrozenInstanceError):
            results.total = 10


# =============================================================================
# Integration Tests
# =============================================================================


class TestModelsIntegration:
    """Test suite for model integration and workflows."""

    def test_search_workflow_with_models(self) -> None:
        """Test complete search workflow using all models.

        **Why this test is important:**
          - Validates models work together correctly
          - Tests real-world usage pattern
          - Ensures API contracts are met
          - Critical for end-to-end functionality

        **What it tests:**
          - SearchResultItem -> SearchResults workflow
          - Multiple items aggregation
          - Data flows through models correctly
        """
        # Simulate search results from vector DB
        item1 = SearchResultItem(
            point_id="doc-uuid-1",
            score=0.95,
            payload={"text": "Machine learning tutorial", "category": "ML"},
        )
        item2 = SearchResultItem(
            point_id="doc-uuid-2",
            score=0.85,
            payload={"text": "Deep learning basics", "category": "DL"},
        )

        # Aggregate into search results
        results = SearchResults(items=[item1, item2], total=2)

        # Verify complete workflow
        assert len(results.items) == 2
        assert results.total == 2
        assert results.items[0].payload["category"] == "ML"
        assert results.items[1].payload["category"] == "DL"

    def test_vector_point_workflow(self) -> None:
        """Test VectorPoint workflow with Qdrant conversion.

        **Why this test is important:**
          - Validates upsert workflow
          - Tests model -> Qdrant integration
          - Ensures data flows correctly
          - Critical for vector operations

        **What it tests:**
          - VectorPoint creation
          - Conversion to Qdrant PointStruct
          - Payload preservation
        """
        # Create point for upsert
        point = VectorPoint(
            id="doc-123",
            vector=[0.1, 0.2, 0.3, 0.4],
            payload={"text": "Example document", "source": "test"},
        )

        # Convert for Qdrant upsert
        qdrant_point = point.to_qdrant()

        # Verify conversion
        assert isinstance(qdrant_point, QdrantPointStruct)
        assert qdrant_point.id == "doc-123"
        assert len(qdrant_point.vector) == 4
        assert qdrant_point.payload["text"] == "Example document"

