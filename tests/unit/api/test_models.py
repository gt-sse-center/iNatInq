"""Unit tests for API Pydantic models.

This file tests all Pydantic request and response models used by the FastAPI endpoints.
All models use Pydantic for automatic validation, serialization, and OpenAPI schema generation.

# Test Coverage

The tests cover:
  - Request model validation (field types, constraints, required fields)
  - Response model serialization (field inclusion, defaults)
  - Model field constraints (min_length, ge, le, pattern)
  - Optional vs required fields
  - Model defaults and factories

# Test Structure

Tests use pytest class-based organization for grouping related tests.

# Running Tests

Run with: pytest tests/unit/api/test_models.py
"""

import pytest
from pydantic import ValidationError
from api import models

# =============================================================================
# Embedding Models Tests
# =============================================================================


class TestEmbedRequest:
    """Test suite for EmbedRequest model."""

    def test_valid_request_with_single_text(self) -> None:
        """Test that EmbedRequest validates correctly with single text.

        **Why this test is important:**
          - Validates basic model construction
          - Ensures required fields are working
          - Tests single-item list handling
        """
        req = models.EmbedRequest(texts=["hello world"])

        assert req.texts == ["hello world"]
        assert req.model is None

    def test_valid_request_with_multiple_texts(self) -> None:
        """Test that EmbedRequest validates correctly with multiple texts.

        **Why this test is important:**
          - Validates multiple texts handling
          - Tests list field behavior
        """
        req = models.EmbedRequest(texts=["hello", "world", "test"])

        assert len(req.texts) == 3
        assert req.texts[0] == "hello"

    def test_valid_request_with_model_override(self) -> None:
        """Test that EmbedRequest accepts optional model parameter.

        **Why this test is important:**
          - Validates optional field handling
          - Tests model override functionality
        """
        req = models.EmbedRequest(texts=["hello"], model="custom-model")

        assert req.texts == ["hello"]
        assert req.model == "custom-model"

    def test_empty_texts_list_fails_validation(self) -> None:
        """Test that EmbedRequest rejects empty texts list.

        **Why this test is important:**
          - Validates min_length constraint
          - Ensures API receives at least one text
          - Prevents unnecessary API calls
        """
        with pytest.raises(ValidationError) as exc_info:
            models.EmbedRequest(texts=[])

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("texts",) for e in errors)
        assert any("at least 1 item" in str(e["msg"]).lower() for e in errors)

    def test_missing_texts_field_fails_validation(self) -> None:
        """Test that EmbedRequest requires texts field.

        **Why this test is important:**
          - Validates required field enforcement
          - Tests Pydantic's required field validation
        """
        with pytest.raises(ValidationError) as exc_info:
            models.EmbedRequest()  # type: ignore[call-arg]

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("texts",) for e in errors)
        assert any(e["type"] == "missing" for e in errors)

    def test_wrong_type_for_texts_fails_validation(self) -> None:
        """Test that EmbedRequest rejects non-list texts.

        **Why this test is important:**
          - Validates field type enforcement
          - Tests type coercion boundaries
        """
        with pytest.raises(ValidationError):
            models.EmbedRequest(texts="not a list")  # type: ignore[arg-type]


class TestEmbedResponse:
    """Test suite for EmbedResponse model."""

    def test_valid_response_with_single_embedding(self) -> None:
        """Test that EmbedResponse serializes correctly with single embedding.

        **Why this test is important:**
          - Validates response model construction
          - Tests embedding list structure
        """
        resp = models.EmbedResponse(model="nomic-embed-text", embeddings=[[0.1, 0.2, 0.3]])

        assert resp.model == "nomic-embed-text"
        assert len(resp.embeddings) == 1
        assert resp.embeddings[0] == [0.1, 0.2, 0.3]

    def test_valid_response_with_multiple_embeddings(self) -> None:
        """Test that EmbedResponse handles multiple embeddings.

        **Why this test is important:**
          - Validates batch embedding response
          - Tests list of lists structure
        """
        resp = models.EmbedResponse(
            model="nomic-embed-text",
            embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        )

        assert len(resp.embeddings) == 2
        assert resp.embeddings[1] == [0.4, 0.5, 0.6]

    def test_missing_required_fields_fails_validation(self) -> None:
        """Test that EmbedResponse requires all fields.

        **Why this test is important:**
          - Validates required field enforcement
          - Ensures API responses are complete
        """
        with pytest.raises(ValidationError):
            models.EmbedResponse()  # type: ignore[call-arg]


# =============================================================================
# Upsert Models Tests
# =============================================================================


class TestUpsertRequest:
    """Test suite for UpsertRequest model."""

    def test_valid_request_minimal(self) -> None:
        """Test that UpsertRequest validates with minimal fields.

        **Why this test is important:**
          - Validates minimal valid request
          - Tests optional field defaults
        """
        req = models.UpsertRequest(texts=["document 1", "document 2"])

        assert len(req.texts) == 2
        assert req.collection is None
        assert req.metadata is None
        assert req.model is None

    def test_valid_request_with_all_fields(self) -> None:
        """Test that UpsertRequest accepts all optional fields.

        **Why this test is important:**
          - Validates full request with all parameters
          - Tests optional field handling
        """
        req = models.UpsertRequest(
            texts=["doc1", "doc2"],
            collection="my-collection",
            metadata=[{"source": "file1.txt"}, {"source": "file2.txt"}],
            model="custom-model",
        )

        assert req.texts == ["doc1", "doc2"]
        assert req.collection == "my-collection"
        assert len(req.metadata) == 2  # type: ignore[arg-type]
        assert req.model == "custom-model"

    def test_empty_texts_fails_validation(self) -> None:
        """Test that UpsertRequest rejects empty texts.

        **Why this test is important:**
          - Validates min_length constraint
          - Prevents empty upserts
        """
        with pytest.raises(ValidationError):
            models.UpsertRequest(texts=[])


class TestUpsertResponse:
    """Test suite for UpsertResponse model."""

    def test_valid_response(self) -> None:
        """Test that UpsertResponse serializes correctly.

        **Why this test is important:**
          - Validates response model construction
          - Tests all required fields
        """
        resp = models.UpsertResponse(
            collection="documents",
            model="nomic-embed-text",
            points_upserted=2,
        )

        assert resp.collection == "documents"
        assert resp.model == "nomic-embed-text"
        assert resp.points_upserted == 2


# =============================================================================
# Search Models Tests
# =============================================================================


class TestSearchResult:
    """Test suite for SearchResult model."""

    def test_valid_search_result_minimal(self) -> None:
        """Test that SearchResult validates with minimal fields.

        **Why this test is important:**
          - Validates required fields work correctly
          - Tests default factory for metadata
        """
        result = models.SearchResult(
            id="test-id",
            score=0.95,
            text="test document",
        )

        assert result.id == "test-id"
        assert result.score == 0.95
        assert result.text == "test document"
        assert result.metadata == {}

    def test_valid_search_result_with_metadata(self) -> None:
        """Test that SearchResult accepts metadata.

        **Why this test is important:**
          - Validates metadata field handling
          - Tests arbitrary dict content
        """
        result = models.SearchResult(
            id="test-id",
            score=0.95,
            text="test document",
            metadata={"source": "file.txt", "page": 1},
        )

        assert result.metadata["source"] == "file.txt"
        assert result.metadata["page"] == 1

    def test_score_constraint_validation(self) -> None:
        """Test that SearchResult validates score constraints.

        **Why this test is important:**
          - Validates ge=0.0, le=1.0 constraints
          - Ensures scores are in valid range
        """
        # Valid scores
        models.SearchResult(id="id", score=0.0, text="text")
        models.SearchResult(id="id", score=0.5, text="text")
        models.SearchResult(id="id", score=1.0, text="text")

        # Invalid scores
        with pytest.raises(ValidationError):
            models.SearchResult(id="id", score=-0.1, text="text")

        with pytest.raises(ValidationError):
            models.SearchResult(id="id", score=1.1, text="text")


class TestSearchResponse:
    """Test suite for SearchResponse model."""

    def test_valid_response_with_results(self) -> None:
        """Test that SearchResponse serializes correctly with results.

        **Why this test is important:**
          - Validates complete search response
          - Tests nested model handling
        """
        resp = models.SearchResponse(
            query="test query",
            model="nomic-embed-text",
            collection="documents",
            provider="qdrant",
            results=[
                models.SearchResult(id="1", score=0.95, text="doc1", metadata={}),
                models.SearchResult(id="2", score=0.85, text="doc2", metadata={}),
            ],
            total=2,
        )

        assert resp.query == "test query"
        assert resp.provider == "qdrant"
        assert len(resp.results) == 2
        assert resp.total == 2

    def test_valid_response_with_empty_results(self) -> None:
        """Test that SearchResponse handles empty results.

        **Why this test is important:**
          - Validates no-results scenario
          - Tests empty list handling
        """
        resp = models.SearchResponse(
            query="test query",
            model="nomic-embed-text",
            collection="documents",
            provider="weaviate",
            results=[],
            total=0,
        )

        assert len(resp.results) == 0
        assert resp.total == 0

    def test_total_constraint_validation(self) -> None:
        """Test that SearchResponse validates total >= 0.

        **Why this test is important:**
          - Validates ge=0 constraint on total
          - Ensures valid count values
        """
        models.SearchResponse(
            query="q",
            model="m",
            collection="c",
            provider="qdrant",
            results=[],
            total=0,
        )

        with pytest.raises(ValidationError):
            models.SearchResponse(
                query="q",
                model="m",
                collection="c",
                provider="qdrant",
                results=[],
                total=-1,
            )


# =============================================================================
# Ray Job Models Tests
# =============================================================================


class TestRayJobRequest:
    """Test suite for RayJobRequest model."""

    def test_valid_request(self) -> None:
        """Test that RayJobRequest validates correctly.

        **Why this test is important:**
          - Validates required fields
          - Tests basic model construction
        """
        req = models.RayJobRequest(
            s3_prefix="inputs/",
            collection="documents",
        )

        assert req.s3_prefix == "inputs/"
        assert req.collection == "documents"


class TestRayJobResponse:
    """Test suite for RayJobResponse model."""

    def test_valid_response(self) -> None:
        """Test that RayJobResponse serializes correctly.

        **Why this test is important:**
          - Validates response model construction
          - Tests all required fields
        """
        resp = models.RayJobResponse(
            job_id="raysubmit_1234567890",
            status="submitted",
            namespace="ml-system",
            s3_prefix="inputs/",
            collection="documents",
            submitted_at="2026-01-12T15:30:45.123456Z",
        )

        assert resp.job_id == "raysubmit_1234567890"
        assert resp.status == "submitted"


class TestRayJobStatusResponse:
    """Test suite for RayJobStatusResponse model."""

    def test_valid_status_without_message(self) -> None:
        """Test that RayJobStatusResponse handles successful status.

        **Why this test is important:**
          - Validates optional message field
          - Tests success case
        """
        resp = models.RayJobStatusResponse(
            job_id="raysubmit_123",
            status="RUNNING",
        )

        assert resp.job_id == "raysubmit_123"
        assert resp.status == "RUNNING"
        assert resp.message is None

    def test_valid_status_with_error_message(self) -> None:
        """Test that RayJobStatusResponse handles failure status.

        **Why this test is important:**
          - Validates message field with error
          - Tests failure case
        """
        resp = models.RayJobStatusResponse(
            job_id="raysubmit_123",
            status="FAILED",
            message="Job failed due to timeout",
        )

        assert resp.status == "FAILED"
        assert resp.message == "Job failed due to timeout"
